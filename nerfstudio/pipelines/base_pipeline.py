# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Abstracts for the Pipeline class.
"""
from __future__ import annotations

import typing
from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple, Type, Union, cast

import torch
import torch.distributed as dist
from PIL import Image
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from torch import nn
from torch.nn import Parameter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp.grad_scaler import GradScaler

from nerfstudio.configs import base_config as cfg
from nerfstudio.data.datamanagers.base_datamanager import (
    DataManager,
    DataManagerConfig,
    VanillaDataManager,
)
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import profiler


def module_wrapper(ddp_or_model: Union[DDP, Model]) -> Model:
    """
    If DDP, then return the .module. Otherwise, return the model.
    """
    if isinstance(ddp_or_model, DDP):
        return cast(Model, ddp_or_model.module)
    return ddp_or_model


class Pipeline(nn.Module):
    """The intent of this class is to provide a higher level interface for the Model
    that will be easy to use for our Trainer class.

    This class will contain high level functions for the model like getting the loss
    dictionaries and visualization code. It should have ways to get the next iterations
    training loss, evaluation loss, and generate whole images for visualization. Each model
    class should be 1:1 with a pipeline that can act as a standardized interface and hide
    differences in how each model takes in and outputs data.

    This class's function is to hide the data manager and model classes from the trainer,
    worrying about:
    1) Fetching data with the data manager
    2) Feeding the model the data and fetching the loss
    Hopefully this provides a higher level interface for the trainer to use, and
    simplifying the model classes, which each may have different forward() methods
    and so on.

    Args:
        config: configuration to instantiate pipeline
        device: location to place model and data
        test_mode:
            'train': loads train/eval datasets into memory
            'test': loads train/test dataset into memory
            'inference': does not load any dataset into memory
        world_size: total number of machines available
        local_rank: rank of current machine

    Attributes:
        datamanager: The data manager that will be used
        model: The model that will be used
    """

    datamanager: DataManager
    _model: Model
    world_size: int

    @property
    def model(self):
        """Returns the unwrapped model if in ddp"""
        return module_wrapper(self._model)

    @property
    def device(self):
        """Returns the device that the model is on."""
        return self.model.device

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: Optional[bool] = None):
        is_ddp_model_state = True
        model_state = {}
        for key, value in state_dict.items():
            if key.startswith("_model."):
                # remove the "_model." prefix from key
                model_state[key[len("_model.") :]] = value
                # make sure that the "module." prefix comes from DDP,
                # rather than an attribute of the model named "module"
                if not key.startswith("_model.module."):
                    is_ddp_model_state = False
        # remove "module." prefix added by DDP
        if is_ddp_model_state:
            model_state = {key[len("module.") :]: value for key, value in model_state.items()}

        pipeline_state = {key: value for key, value in state_dict.items() if not key.startswith("_model.")}

        try:
            self.model.load_state_dict(model_state, strict=True)
        except RuntimeError:
            if not strict:
                self.model.load_state_dict(model_state, strict=False)
            else:
                raise

        super().load_state_dict(pipeline_state, strict=False)

    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict. This will be responsible for
        getting the next batch of data from the DataManager and interfacing with the
        Model class, feeding the data to the model's forward function.

        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """
        if self.world_size > 1 and step:
            assert self.datamanager.train_sampler is not None
            self.datamanager.train_sampler.set_epoch(step)
        ray_bundle, batch = self.datamanager.next_train(step)
        model_outputs = self.model(ray_bundle, batch)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)

        return model_outputs, loss_dict, metrics_dict

    @profiler.time_function
    def get_eval_loss_dict(self, step: int):
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        self.eval()
        if self.world_size > 1:
            assert self.datamanager.eval_sampler is not None
            self.datamanager.eval_sampler.set_epoch(step)
        ray_bundle, batch = self.datamanager.next_eval(step)
        model_outputs = self.model(ray_bundle, batch)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        self.train()
        return model_outputs, loss_dict, metrics_dict

    @abstractmethod
    @profiler.time_function
    def get_eval_image_metrics_and_images(self, step: int):
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """

    @abstractmethod
    @profiler.time_function
    def get_average_eval_image_metrics(
        self, step: Optional[int] = None, output_path: Optional[Path] = None, get_std: bool = False
    ):
        """Iterate over all the images in the eval dataset and get the average.

        Args:
            step: current training step
            output_path: optional path to save rendered images to
            get_std: Set True if you want to return std with the mean metric.
        """

    def load_pipeline(self, loaded_state: Dict[str, Any], step: int) -> None:
        """Load the checkpoint from the given path

        Args:
            loaded_state: pre-trained model state dict
            step: training step of the loaded checkpoint
        """

    @abstractmethod
    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        """Returns the training callbacks from both the Dataloader and the Model."""

    @abstractmethod
    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Get the param groups for the pipeline.

        Returns:
            A list of dictionaries containing the pipeline's param groups.
        """


@dataclass
class VanillaPipelineConfig(cfg.InstantiateConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: VanillaPipeline)
    """target class to instantiate"""
    datamanager: DataManagerConfig = DataManagerConfig()
    """specifies the datamanager config"""
    model: ModelConfig = ModelConfig()
    """specifies the model config"""


class VanillaPipeline(Pipeline):
    """The pipeline class for the vanilla nerf setup of multiple cameras for one or a few scenes.

    Args:
        config: configuration to instantiate pipeline
        device: location to place model and data
        test_mode:
            'val': loads train/val datasets into memory
            'test': loads train/test dataset into memory
            'inference': does not load any dataset into memory
        world_size: total number of machines available
        local_rank: rank of current machine
        grad_scaler: gradient scaler used in the trainer

    Attributes:
        datamanager: The data manager that will be used
        model: The model that will be used
    """

    def __init__(
        self,
        config: VanillaPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super().__init__()
        self.config = config
        self.test_mode = test_mode
        self.datamanager: DataManager = config.datamanager.setup(
            device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank
        )
        self.datamanager.to(device)
        # TODO(ethan): get rid of scene_bounds from the model
        assert self.datamanager.train_dataset is not None, "Missing input dataset"

        self._model = config.model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=len(self.datamanager.train_dataset),
            metadata=self.datamanager.train_dataset.metadata,
            device=device,
            grad_scaler=grad_scaler,
        )
        self.model.to(device)

        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(Model, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True))
            dist.barrier(device_ids=[local_rank])

    @property
    def device(self):
        """Returns the device that the model is on."""
        return self.model.device

    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict. This will be responsible for
        getting the next batch of data from the DataManager and interfacing with the
        Model class, feeding the data to the model's forward function.

        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """
        ray_bundle, batch = self.datamanager.next_train(step)

        # MULTIVIEW CONSISTENCY CHECK
        image_batch = self.datamanager.next_bundle(step)
        images = image_batch["image"]
        depths = image_batch["depth_image"]
        image_shape = images.shape
        num_cam, H, W = image_shape[0], image_shape[1], image_shape[2]
        cam_to_world = self.datamanager.train_dataset.cameras.camera_to_worlds  # Shape: [N, 3, 4]
        intrinsics = self.datamanager.train_dataset.cameras.get_intrinsics_matrices()[0]

        from .vis_correspondence import vis_depth_images

        # vis_depth_images(depths)  # Uncomment this line for debugging

        indices = batch["indices"]  # Shape: [N, 3] [camera_id, y, x]
        camera_ids = indices[:, 0]
        unique_camera_ids, counts = torch.unique(camera_ids, return_counts=True)
        camera_id_count_dict = {int(uid.item()): int(c.item()) for uid, c in zip(unique_camera_ids, counts)}
        # print("Unique Camera IDs and their counts:", camera_id_count_dict)

        cam_0_indice_mask = indices[:, 0] == 0
        indices_where_camera_id_is_0 = torch.nonzero(cam_0_indice_mask).squeeze()
        # print("Indices where camera_id is 0:", indices_where_camera_id_is_0.tolist())

        # Use the boolean mask to filter the rows where camera_id is 0
        cam_0_indices = indices[cam_0_indice_mask]
        cam_0_yx_coordinates = cam_0_indices[:, 1:]
        # cam_0_yx_coordinates = cam_0_yx_coordinates[:10, :] # Take the first sample
        # cam_0_yx_coordinates = torch.tensor([360, 700], dtype=torch.float64, device="cuda:0").unsqueeze(0).double()
        # cam_0_xy_coordinates = cam_0_yx_coordinates[:, [1, 0]] # Flip the x and y coordinates
        # cam_0_xy_list = [tuple(coord.tolist()) for coord in cam_0_yx_coordinates]
        # print("XY coordinates where camera_id is 0 (list of tuples):", xy_list)

        from collections import defaultdict

        camera_indices_dict = defaultdict(list)
        camera_yx_coordinates_dict = defaultdict(list)

        # Find unique camera_ids
        unique_camera_ids = torch.unique(indices[:, 0])

        # Loop through each unique camera_id
        for camera_id in unique_camera_ids:
            # Create a boolean mask for the current camera_id
            camera_indice_mask = indices[:, 0] == camera_id

            # Find indices where camera_id matches the current camera_id
            indices_where_camera_id_is_current = torch.nonzero(camera_indice_mask).squeeze()

            # Use the boolean mask to filter the rows where camera_id matches
            current_camera_indices = indices[camera_indice_mask]

            # Extract yx_coordinates for the current camera_id
            current_camera_yx_coordinates = current_camera_indices[:, 1:]

            camera_indices_dict[camera_id.item()] = indices_where_camera_id_is_current
            camera_yx_coordinates_dict[camera_id.item()] = current_camera_yx_coordinates

        cam_0_yx_coordinates = camera_yx_coordinates_dict[0]
        # cam_0_yx_coordinates = cam_0_yx_coordinates[:128]  # Take the partial sample
        cam_0_to_world = cam_to_world[0]
        depths_for_camera_id_0 = batch["depth_image"][indices_where_camera_id_is_0]

        device = torch.device("cuda:0")  # or "cpu"
        depths_for_camera_id_0 = depths_for_camera_id_0.to(device)
        cam_0_to_world = cam_0_to_world.to(device)
        intrinsics = intrinsics.to(device)

        # Convert from OpenGL to OpenCV coordinate system
        cam_0_to_world = self.convert_pose(cam_to_world[0]).to(device).double()
        cam_0_world_coordinates = self.deproject_pixels_to_world(
            depths[0], cam_0_to_world, intrinsics, cam_0_yx_coordinates
        )
        cam_0_cam_coordinates = self.deproject_pixels_to_cam(
            depths[0], cam_0_to_world, intrinsics, cam_0_yx_coordinates
        )
        # Here we truncate the cam_0_world_coordinates to the first 250 samples
        cam_0_world_coordinates = cam_0_world_coordinates[:300, :]
        cam_0_cam_coordinates = cam_0_cam_coordinates[:300, :]

        cam_1_to_world = self.convert_pose(cam_to_world[1])
        cam_2_to_world = self.convert_pose(cam_to_world[2])

        from .single_point_debug import debug_point  # Uncomment this line for debugging

        # debug_point(depths, cam_0_to_world, cam_1_to_world)  # Uncomment this line for debugging

        # TODO: Inplement the input for the network
        projected_coordinates_0_i_dict = {}
        projected_rgb_cam_0_i_dict = {}
        projected_depth_cam_0_i_dict = {}
        deprojected_coordinates_cam_0_i_dict = {}
        # point_3d_depth_dict = {}
        # point_3d_cam_0_i_dict = {}

        for i in range(0, num_cam):
            cam_i_img = images[i]
            cam_i_depth = depths[i]
            cam_i_to_world = self.convert_pose(cam_to_world[i])

            projected_coordinates_0_i = self.project_world_to_pixels(
                cam_0_world_coordinates, cam_i_to_world, intrinsics
            )
            projected_rgb_cam_0_i = self.fetch_rgb_from_image(projected_coordinates_0_i, cam_i_img)
            projected_depth_cam_0_i = self.fetch_depth_from_image(projected_coordinates_0_i, cam_i_depth)
            # if torch.nonzero(projected_depth_cam_0_i).size(0) != 0:
            #     deprojected_coordinates_cam_0_i = self.deproject_pixels_to_cam(
            #         cam_i_depth, cam_i_to_world, intrinsics, projected_coordinates_0_i
            #     )
            # else:
            #     deprojected_coordinates_cam_0_i = torch.zeros_like(projected_coordinates_0_i)
            # point_3d_depth_cam_0_i = self.get_depth_transform_world_to_camera(cam_0_world_coordinates, cam_i_to_world)
            # point_3d_cam_0_i = self.transform_world_to_camera(cam_0_world_coordinates, cam_i_to_world)

            projected_coordinates_0_i_dict[i] = projected_coordinates_0_i
            projected_rgb_cam_0_i_dict[i] = projected_rgb_cam_0_i
            projected_depth_cam_0_i_dict[
                i
            ] = projected_depth_cam_0_i  # This depth is input to the network to be refined (many EMPTY values)
            # deprojected_coordinates_cam_0_i_dict[i] = deprojected_coordinates_cam_0_i
            # point_3d_depth_dict[i] = point_3d_depth_cam_0_i  # This depth is transformed from cam0 to cami
            # point_3d_cam_0_i_dict[i] = point_3d_cam_0_i  # This 3d point is transformed from cam0 to cami (NEED CAM0)

        refinement_dicts = {
            "pixel_coordinates": projected_coordinates_0_i_dict,
            "rgb": projected_rgb_cam_0_i_dict,
            "depth": projected_depth_cam_0_i_dict,
            "cam_coordinates": deprojected_coordinates_cam_0_i_dict,
        }

        for i in range(1, num_cam):
            print(
                "Non-zero projected depth values for camera",
                i,
                ":",
                torch.nonzero(projected_depth_cam_0_i_dict[i]).size(0),
                "/",
                projected_depth_cam_0_i_dict[i].size(0),
            )
        from .depth_refinement_model import DepthRefinementModel
        from torch.utils.data import DataLoader
        from .dataloader import DepthRefineDataset

        depth_refine_dataset = DepthRefineDataset(refinement_dicts)
        train_loader = DataLoader(
            dataset=depth_refine_dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=False, drop_last=False
        )

        refinement_model = DepthRefinementModel(
            cam_0_cam_coordinates, projected_coordinates_0_i_dict, projected_depth_cam_0_i_dict
        ).to(device)
        refinement_model.train()
        optimizer = torch.optim.Adam(params=refinement_model.parameters(), lr=0.001)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.001)
        pho_criterion = nn.MSELoss()
        depth_criterion = nn.L1Loss()

        # #######################################################################
        # USED FOR TESTING AND DEBUGGING
        # #######################################################################
        projected_coordinates_0_1 = self.project_world_to_pixels(cam_0_world_coordinates, cam_1_to_world, intrinsics)
        projected_coordinates_0_2 = self.project_world_to_pixels(cam_0_world_coordinates, cam_2_to_world, intrinsics)

        # Fetch RGB values from the images
        sample_rgb_cam_0 = self.fetch_rgb_from_image(cam_0_yx_coordinates, images[0])
        projected_rgb_cam_0_1 = self.fetch_rgb_from_image(projected_coordinates_0_1, images[1])
        projected_rgb_cam_0_2 = self.fetch_rgb_from_image(projected_coordinates_0_2, images[2])

        # Fetch depth values from the depth images
        projected_depth_cam_0_1 = self.fetch_depth_from_image(projected_coordinates_0_1, depths[1])
        point_3d_depth_cam_0_1 = self.get_depth_transform_world_to_camera(cam_0_world_coordinates, cam_1_to_world)

        # Build L1 Depth Loss for Multiview Consistency Check
        l1_loss = self.compute_l1_loss_for_valid_depths(point_3d_depth_cam_0_1, projected_depth_cam_0_1)

        from .vis_correspondence import vis_correspondence_three_imgs

        vis_correspondence_three_imgs(
            image_batch, cam_0_yx_coordinates, projected_coordinates_0_1, projected_coordinates_0_2, H, W
        )

        model_outputs = self._model(ray_bundle)  # train distributed data parallel model if world_size > 1
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)

        if self.config.datamanager.camera_optimizer is not None:
            camera_opt_param_group = self.config.datamanager.camera_optimizer.param_group
            if camera_opt_param_group in self.datamanager.get_param_groups():
                # Report the camera optimization metrics
                metrics_dict["camera_opt_translation"] = (
                    self.datamanager.get_param_groups()[camera_opt_param_group][0].data[:, :3].norm()
                )
                metrics_dict["camera_opt_rotation"] = (
                    self.datamanager.get_param_groups()[camera_opt_param_group][0].data[:, 3:].norm()
                )

        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)

        return model_outputs, loss_dict, metrics_dict

    def convert_pose(self, C2W):
        import numpy as np

        flip_yz = np.eye(4)
        flip_yz[1, 1] = -1
        flip_yz[2, 2] = -1
        C2W = np.matmul(C2W, flip_yz)
        return C2W

    def compute_l1_loss_for_valid_depths(self, depth_from_cam1, projected_depth_0to1):
        """
        Compute the L1 loss between two depth tensors only for valid (non-NaN) depths.

        Parameters:
        - depth_from_cam1 (torch.Tensor): Depth values from camera 1. Shape [N, 1].
        - projected_depth_0to1 (torch.Tensor): Projected depth values from camera 0 to camera 1. Shape [N, 1].

        Returns:
        - torch.Tensor: The computed L1 loss.
        """
        import torch.nn.functional as F

        # Create masks for valid depth values (not NaN)
        valid_cam1 = ~torch.isnan(depth_from_cam1)
        valid_proj_0to1 = ~torch.isnan(projected_depth_0to1)

        # Create a combined mask for positions where both depth values are valid
        valid_combined = valid_cam1 & valid_proj_0to1

        # Compute L1 loss only for valid positions
        if valid_combined.any():  # Check if there are any valid points
            l1_loss = F.l1_loss(depth_from_cam1[valid_combined], projected_depth_0to1[valid_combined])
        else:
            l1_loss = torch.tensor(0.0)  # Or handle this case as you see fit

        return l1_loss

    def get_depth_transform_world_to_camera(self, world_coordinates, camera_to_world):
        device = world_coordinates.device
        dtype = world_coordinates.dtype

        camera_to_world = camera_to_world.to(device).to(dtype)
        # Create a 4x4 version of the 3x4 camera-to-world matrix
        cam_to_world_4x4 = torch.zeros((4, 4), dtype=dtype, device=device)
        cam_to_world_4x4[:3, :] = camera_to_world
        cam_to_world_4x4[3, 3] = 1.0

        # Compute the inverse transformation matrix
        rotational_part = cam_to_world_4x4[:3, :3]
        translational_part = cam_to_world_4x4[:3, 3]
        inverse_rotation = rotational_part.T
        inverse_translation = torch.matmul(-inverse_rotation, translational_part)
        world_to_cam_4x4 = torch.zeros((4, 4), dtype=dtype, device=device)
        world_to_cam_4x4[:3, :3] = inverse_rotation
        world_to_cam_4x4[:3, 3] = inverse_translation
        world_to_cam_4x4[3, 3] = 1.0

        # Homogenize world_coordinates to [x, y, z, 1]
        homogeneous_world_coordinates = torch.cat(
            [world_coordinates, torch.ones((world_coordinates.shape[0], 1), device=device, dtype=dtype)], dim=1
        )

        camera_coordinates = torch.matmul(homogeneous_world_coordinates, world_to_cam_4x4.T)

        # Normalize the coordinates by the third (z) coordinate
        z_coords = camera_coordinates[:, 2:3]

        return z_coords

    def transform_world_to_camera(self, world_coordinates, camera_to_world):
        device = world_coordinates.device
        dtype = world_coordinates.dtype

        camera_to_world = camera_to_world.to(device).to(dtype)
        # Create a 4x4 version of the 3x4 camera-to-world matrix
        cam_to_world_4x4 = torch.zeros((4, 4), dtype=dtype, device=device)
        cam_to_world_4x4[:3, :] = camera_to_world
        cam_to_world_4x4[3, 3] = 1.0

        # Compute the inverse transformation matrix
        rotational_part = cam_to_world_4x4[:3, :3]
        translational_part = cam_to_world_4x4[:3, 3]
        inverse_rotation = rotational_part.T
        inverse_translation = torch.matmul(-inverse_rotation, translational_part)
        world_to_cam_4x4 = torch.zeros((4, 4), dtype=dtype, device=device)
        world_to_cam_4x4[:3, :3] = inverse_rotation
        world_to_cam_4x4[:3, 3] = inverse_translation
        world_to_cam_4x4[3, 3] = 1.0

        # Homogenize world_coordinates to [x, y, z, 1]
        homogeneous_world_coordinates = torch.cat(
            [world_coordinates, torch.ones((world_coordinates.shape[0], 1), device=device, dtype=dtype)], dim=1
        )

        camera_coordinates = torch.matmul(homogeneous_world_coordinates, world_to_cam_4x4.T)
        camera_coordinates = camera_coordinates[:, 0:3]

        return camera_coordinates

    def fetch_rgb_from_image(self, projected_coordinates, project_image):
        """
        Fetch the RGB values of pixels from an image at given coordinates.

        Parameters:
        - projected_coordinates (torch.Tensor): The 2D coordinates to fetch pixel values from. Shape [N, 2].
        - project_image (torch.Tensor): The source image from which to fetch pixel values. Shape [H, W, 3].

        Returns:
        - torch.Tensor: The RGB values corresponding to the input coordinates. Shape [N, 3].
        """

        # Round the coordinates to the nearest integers
        rounded_coordinates = torch.round(projected_coordinates).long()

        # Clip the coordinates to be within valid range
        rounded_coordinates[:, 0] = torch.clamp(rounded_coordinates[:, 0], 0, project_image.shape[1] - 1)
        rounded_coordinates[:, 1] = torch.clamp(rounded_coordinates[:, 1], 0, project_image.shape[0] - 1)

        # Move rounded_coordinates to the same device as project_image
        rounded_coordinates = rounded_coordinates.to(project_image.device)

        # Fetch the RGB values
        rgb_values = project_image[rounded_coordinates[:, 1], rounded_coordinates[:, 0], :]

        return rgb_values

    def fetch_depth_from_image(self, projected_coordinates, depth_image):
        """
        Fetch the depth values of pixels from an image at given coordinates.

        Parameters:
        - projected_coordinates (torch.Tensor): The 2D coordinates to fetch pixel values from. Shape [N, 2].
        - depth_image (torch.Tensor): The source depth image from which to fetch pixel values. Shape [H, W, 1].

        Returns:
        - torch.Tensor: The depth values corresponding to the input coordinates. Shape [N, 1].
        """

        # Initialize output tensor filled with NaN values
        N = projected_coordinates.shape[0]
        # depth_values = torch.full((N, 1), float("nan"), dtype=depth_image.dtype, device=depth_image.device)
        depth_values = torch.full((N, 1), float("0.0000"), dtype=depth_image.dtype, device=depth_image.device)

        # Round the coordinates to the nearest integers
        rounded_coordinates = torch.round(projected_coordinates).long()

        # Create a mask for coordinates that are within the image boundary
        valid_x = (rounded_coordinates[:, 0] >= 0) & (rounded_coordinates[:, 0] < depth_image.shape[1])
        valid_y = (rounded_coordinates[:, 1] >= 0) & (rounded_coordinates[:, 1] < depth_image.shape[0])
        valid_coordinates = valid_x & valid_y

        # Clip the coordinates to be within valid range (only for valid ones)
        rounded_coordinates[valid_coordinates, 0] = torch.clamp(
            rounded_coordinates[valid_coordinates, 0], 0, depth_image.shape[1] - 1
        )
        rounded_coordinates[valid_coordinates, 1] = torch.clamp(
            rounded_coordinates[valid_coordinates, 1], 0, depth_image.shape[0] - 1
        )

        # Move rounded_coordinates to the same device as depth_image
        rounded_coordinates = rounded_coordinates.to(depth_image.device)

        # Fetch the depth values for valid coordinates
        depth_values[valid_coordinates] = depth_image[
            rounded_coordinates[valid_coordinates, 1], rounded_coordinates[valid_coordinates, 0], :
        ]

        return depth_values

    def deproject_pixels_to_world(self, cam_0_depth, cam_0_to_world, intrinsics, cam_0_yx_coordinates):
        device = cam_0_depth.device
        dtype = cam_0_depth.dtype

        # Convert all inputs to the same device and dtype
        camera_to_world = cam_0_to_world.to(device).to(dtype)
        intrinsics = intrinsics.to(device).to(dtype)
        cam_0_yx_coordinates = cam_0_yx_coordinates.to(device).to(dtype).long()
        depth_values = cam_0_depth[cam_0_yx_coordinates[:, 0], cam_0_yx_coordinates[:, 1], 0].to(device).to(dtype)

        # Create a mask to exclude zero depth values
        valid_depth_mask = depth_values != 0

        # Use the mask to filter out invalid coordinates and depth values
        cam_0_yx_coordinates = cam_0_yx_coordinates[valid_depth_mask]
        depth_values = depth_values[valid_depth_mask]

        if len(depth_values) == 0:
            return None  # return None if no valid depths are found

        homogeneous_pixel_coordinates = torch.cat(
            [
                cam_0_yx_coordinates,
                torch.ones(cam_0_yx_coordinates.shape[0], 1).to(device).to(dtype),
            ],
            dim=1,
        )

        homogeneous_pixel_coordinates[:, :2] = homogeneous_pixel_coordinates[:, :2].flip(dims=[1])

        # Deproject to camera space
        inv_intrinsics = torch.inverse(intrinsics).to(device).to(dtype)
        cam_0_coordinates = torch.matmul(homogeneous_pixel_coordinates, inv_intrinsics.T) * depth_values.unsqueeze(-1)

        # Create homogeneous coordinates in camera space
        homogeneous_cam_coordinates_3D = torch.cat(
            [
                cam_0_coordinates,
                torch.ones(cam_0_coordinates.shape[0], 1).to(device).to(dtype),
            ],
            dim=1,
        )

        world_coordinates = torch.matmul(homogeneous_cam_coordinates_3D, camera_to_world.T)

        return world_coordinates

    def deproject_pixels_to_cam(self, cam_0_depth, cam_0_to_world, intrinsics, cam_0_yx_coordinates):
        device = cam_0_depth.device
        dtype = cam_0_depth.dtype

        # Convert all inputs to the same device and dtype
        intrinsics = intrinsics.to(dtype)
        cam_0_yx_coordinates = cam_0_yx_coordinates.to(device).to(dtype).long()
        depth_values = cam_0_depth[cam_0_yx_coordinates[:, 0], cam_0_yx_coordinates[:, 1], 0].to(dtype)

        # Create a mask to exclude zero depth values
        valid_depth_mask = depth_values != 0

        # Use the mask to filter out invalid coordinates and depth values
        cam_0_yx_coordinates = cam_0_yx_coordinates[valid_depth_mask]
        depth_values = depth_values[valid_depth_mask]

        if len(depth_values) == 0:
            return None  # return None if no valid depths are found

        homogeneous_pixel_coordinates = torch.cat(
            [
                cam_0_yx_coordinates,
                torch.ones(cam_0_yx_coordinates.shape[0], 1).to(device).to(dtype),
            ],
            dim=1,
        )

        homogeneous_pixel_coordinates[:, :2] = homogeneous_pixel_coordinates[:, :2].flip(dims=[1])

        # Deproject to camera space
        inv_intrinsics = torch.inverse(intrinsics).to(device).to(dtype)
        cam_0_coordinates = torch.matmul(homogeneous_pixel_coordinates, inv_intrinsics.T) * depth_values.unsqueeze(-1)

        return cam_0_coordinates

    def project_world_to_pixels(self, world_coordinates, camera_to_world, intrinsics):
        # Ensure all tensors are on the same device and dtype
        device = world_coordinates.device
        dtype = world_coordinates.dtype

        camera_to_world = camera_to_world.to(device).to(dtype)
        # Create a 4x4 version of the 3x4 camera-to-world matrix
        cam_to_world_4x4 = torch.zeros((4, 4), dtype=dtype, device=device)
        cam_to_world_4x4[:3, :] = camera_to_world
        cam_to_world_4x4[3, 3] = 1.0

        # Compute the inverse transformation matrix
        rotational_part = cam_to_world_4x4[:3, :3]
        translational_part = cam_to_world_4x4[:3, 3]
        inverse_rotation = rotational_part.T
        inverse_translation = torch.matmul(-inverse_rotation, translational_part)
        world_to_cam_4x4 = torch.zeros((4, 4), dtype=dtype, device=device)
        world_to_cam_4x4[:3, :3] = inverse_rotation
        world_to_cam_4x4[:3, 3] = inverse_translation
        world_to_cam_4x4[3, 3] = 1.0
        intrinsics = intrinsics.to(device).to(dtype)

        # Homogenize world_coordinates to [x, y, z, 1]
        homogeneous_world_coordinates = torch.cat(
            [world_coordinates, torch.ones((world_coordinates.shape[0], 1), device=device, dtype=dtype)], dim=1
        )

        camera_coordinates = torch.matmul(homogeneous_world_coordinates, world_to_cam_4x4.T)
        projected_coordinates_homogeneous = torch.mm(camera_coordinates[:, :3], intrinsics.t())

        # Normalize the coordinates by the third (z) coordinate
        z_coords = projected_coordinates_homogeneous[:, 2:3]
        projected_coordinates = projected_coordinates_homogeneous[:, :2] / z_coords
        projected_coordinates = projected_coordinates[:, [1, 0]]

        return projected_coordinates

    def forward(self):
        """Blank forward method

        This is an nn.Module, and so requires a forward() method normally, although in our case
        we do not need a forward() method"""
        raise NotImplementedError

    @profiler.time_function
    def get_eval_loss_dict(self, step: int) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        self.eval()
        ray_bundle, batch = self.datamanager.next_eval(step)
        model_outputs = self.model(ray_bundle)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        self.train()
        return model_outputs, loss_dict, metrics_dict

    @profiler.time_function
    def get_eval_image_metrics_and_images(self, step: int):
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        self.eval()
        image_idx, camera_ray_bundle, batch = self.datamanager.next_eval_image(step)
        outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
        metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, batch)
        assert "image_idx" not in metrics_dict
        metrics_dict["image_idx"] = image_idx
        assert "num_rays" not in metrics_dict
        metrics_dict["num_rays"] = len(camera_ray_bundle)
        self.train()
        return metrics_dict, images_dict

    @profiler.time_function
    def get_average_eval_image_metrics(
        self, step: Optional[int] = None, output_path: Optional[Path] = None, get_std: bool = False
    ):
        """Iterate over all the images in the eval dataset and get the average.

        Args:
            step: current training step
            output_path: optional path to save rendered images to
            get_std: Set True if you want to return std with the mean metric.

        Returns:
            metrics_dict: dictionary of metrics
        """
        self.eval()
        metrics_dict_list = []
        assert isinstance(self.datamanager, VanillaDataManager)
        num_images = len(self.datamanager.fixed_indices_eval_dataloader)
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("[green]Evaluating all eval images...", total=num_images)
            for camera_ray_bundle, batch in self.datamanager.fixed_indices_eval_dataloader:
                # time this the following line
                inner_start = time()
                height, width = camera_ray_bundle.shape
                num_rays = height * width
                outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
                metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, batch)

                if output_path is not None:
                    camera_indices = camera_ray_bundle.camera_indices
                    assert camera_indices is not None
                    for key, val in images_dict.items():
                        Image.fromarray((val * 255).byte().cpu().numpy()).save(
                            output_path / "{0:06d}-{1}.jpg".format(int(camera_indices[0, 0, 0]), key)
                        )
                assert "num_rays_per_sec" not in metrics_dict
                metrics_dict["num_rays_per_sec"] = num_rays / (time() - inner_start)
                fps_str = "fps"
                assert fps_str not in metrics_dict
                metrics_dict[fps_str] = metrics_dict["num_rays_per_sec"] / (height * width)
                metrics_dict_list.append(metrics_dict)
                progress.advance(task)
        # average the metrics list
        metrics_dict = {}
        for key in metrics_dict_list[0].keys():
            if get_std:
                key_std, key_mean = torch.std_mean(
                    torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list])
                )
                metrics_dict[key] = float(key_mean)
                metrics_dict[f"{key}_std"] = float(key_std)
            else:
                metrics_dict[key] = float(
                    torch.mean(torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list]))
                )
        self.train()
        return metrics_dict

    def load_pipeline(self, loaded_state: Dict[str, Any], step: int) -> None:
        """Load the checkpoint from the given path

        Args:
            loaded_state: pre-trained model state dict
            step: training step of the loaded checkpoint
        """
        state = {
            (key[len("module.") :] if key.startswith("module.") else key): value for key, value in loaded_state.items()
        }
        self.model.update_to_step(step)
        self.load_state_dict(state)

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        """Returns the training callbacks from both the Dataloader and the Model."""
        datamanager_callbacks = self.datamanager.get_training_callbacks(training_callback_attributes)
        model_callbacks = self.model.get_training_callbacks(training_callback_attributes)
        callbacks = datamanager_callbacks + model_callbacks
        return callbacks

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Get the param groups for the pipeline.

        Returns:
            A list of dictionaries containing the pipeline's param groups.
        """
        datamanager_params = self.datamanager.get_param_groups()
        model_params = self.model.get_param_groups()
        # TODO(ethan): assert that key names don't overlap
        return {**datamanager_params, **model_params}
