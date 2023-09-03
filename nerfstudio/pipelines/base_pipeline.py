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

        indices = batch["indices"]  # Shape: [N, 3] [camera_id, y, x]
        camera_ids = indices[:, 0]
        unique_camera_ids, counts = torch.unique(camera_ids, return_counts=True)
        camera_id_count_dict = {int(uid.item()): int(c.item()) for uid, c in zip(unique_camera_ids, counts)}
        # print("Unique Camera IDs and their counts:", camera_id_count_dict)

        cam_0_indice_mask = indices[:, 0] == 0
        cam_1_indice_mask = indices[:, 0] == 1
        cam_2_indice_mask = indices[:, 0] == 2
        indices_where_camera_id_is_0 = torch.nonzero(cam_0_indice_mask).squeeze()
        indices_where_camera_id_is_1 = torch.nonzero(cam_1_indice_mask).squeeze()
        indices_where_camera_id_is_2 = torch.nonzero(cam_2_indice_mask).squeeze()
        # print("Indices where camera_id is 0:", indices_where_camera_id_is_0.tolist())

        # Use the boolean mask to filter the rows where camera_id is 0
        cam_0_indices = indices[cam_0_indice_mask]
        cam_0_yx_coordinates = cam_0_indices[:, 1:]
        # cam_0_yx_coordinates = cam_0_yx_coordinates[:1, :] # Take the first sample
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
        cam_0_img = images[0]
        cam_0_depth = depths[0]
        cam_0_to_world = cam_to_world[0]
        depths_for_camera_id_0 = batch["depth_image"][indices_where_camera_id_is_0]

        device = torch.device("cuda:0")  # or "cpu"
        depths_for_camera_id_0 = depths_for_camera_id_0.to(device)
        cam_0_to_world = cam_0_to_world.to(device)
        intrinsics = intrinsics.to(device)

        cam_0_to_world = torch.tensor(
            [
                [-0.3407, 0.0572, -0.9384, 1.1704],
                [0.9397, 0.0525, -0.3380, 1.39677],
                [0.0299, -0.9970, -0.0717, -2.5307],
            ],
            device="cuda:0",
        ).double()  # This is used for testing
        # Convert from OpenGL to OpenCV coordinate system
        cam_0_to_world = self.convert_pose(cam_to_world[0]).to(device).double()
        cam_0_world_coordinates = self.deproject_pixels_to_world(
            cam_0_depth, cam_0_to_world, intrinsics, cam_0_yx_coordinates
        )
        cam_1_to_world = torch.tensor(
            [
                [-0.4591, 0.1727, -0.8714, 1.2658],
                [0.8881, 0.1157, -0.4449, 1.204033],
                [0.0240, -0.9782, -0.2065, -2.4878],
            ],
            device="cuda:0",
        ).double()  # This is used for testing
        # cam_1_to_world = self.convert_pose(self.datamanager.train_dataset.cameras.camera_to_worlds[1])
        # cam_2_to_world = self.convert_pose(self.datamanager.train_dataset.cameras.camera_to_worlds[2])
        cam_1_to_world = self.convert_pose(cam_to_world[1])
        cam_2_to_world = self.convert_pose(cam_to_world[2])
        # THIS IS USED FOR DEBUGGING AND TESTING
        #####################################################################
        intrinsics_test = torch.tensor(
            [[702.0630, 0.0000, 566.0000], [0.0000, 701.9382, 437.0000], [0.0000, 0.0000, 1.0000]], device="cuda:0"
        ).double()
        depth_value = torch.tensor(cam_0_depth[360, 700], device="cuda:0", dtype=torch.float64).double()
        # Step 1: Create homogeneous coordinate
        homogeneous_pixel_coordinates = torch.tensor([700, 360, 1], dtype=torch.float64, device="cuda:0").double()
        # Step 2: Calculate the inverse of the intrinsic matrix
        inv_intrinsics = torch.inverse(intrinsics_test.double())
        # Step 3: Multiply by the inverse of the intrinsic and by the depth
        cam_0_coordinates = torch.matmul(homogeneous_pixel_coordinates, inv_intrinsics.T) * depth_value
        # Step 4: Convert to homogeneous coordinates in camera space
        homogeneous_cam_coordinates_3D = torch.cat(
            [cam_0_coordinates, torch.ones(1, dtype=torch.float64, device="cuda:0")]
        )
        # Step 5: Multiply by the camera-to-world matrix to get world coordinates
        world_coordinates_verify = torch.matmul(homogeneous_cam_coordinates_3D, cam_0_to_world.T)
        print("Deprojected 3D world coordinates:", world_coordinates_verify)
        homogeneous_world_coordinates = torch.cat(
            [world_coordinates_verify, torch.tensor([1.0], device=device, dtype=torch.float64)]
        )
        camera_coordinates = self.transform_world_to_camera(homogeneous_world_coordinates, cam_1_to_world)
        z = camera_coordinates[2]
        projected_coordinates_verify = camera_coordinates[:3] / z
        projected_coordinates_verify = torch.matmul(projected_coordinates_verify, intrinsics_test.T)
        x, y = projected_coordinates_verify[0], projected_coordinates_verify[1]
        print("Projected coordinates verify:", x, y)
        ###########################################################################################################

        projected_coordinates_0_1 = self.project_world_to_pixels(cam_0_world_coordinates, cam_1_to_world, intrinsics)
        projected_coordinates_0_2 = self.project_world_to_pixels(cam_0_world_coordinates, cam_2_to_world, intrinsics)
        cam_1_img = images[1]
        cam_2_img = images[2]
        cam_1_depth = depths[1]
        cam_2_depth = depths[2]
        sample_rgb_cam_0 = self.fetch_rgb_from_image(cam_0_yx_coordinates, cam_0_img)
        projected_rgb_cam_0_1 = self.fetch_rgb_from_image(projected_coordinates_0_1, cam_1_img)
        projected_rgb_cam_0_2 = self.fetch_rgb_from_image(projected_coordinates_0_2, cam_2_img)
        projected_depth_cam_0_1 = self.fetch_depth_from_image(projected_coordinates_0_1, cam_1_depth)
        projected_depth_cam_0_2 = self.fetch_depth_from_image(projected_coordinates_0_2, cam_2_depth)

        # TODO: Build Loss for Multiview Consistency Check

        ########################################################################################
        import matplotlib.pyplot as plt
        import numpy as np

        image_0_tensor = image_batch["image"][0].cpu().numpy()  # Convert to numpy array
        image_1_tensor = image_batch["image"][1].cpu().numpy()  # Convert to numpy array
        image_2_tensor = image_batch["image"][2].cpu().numpy()  # Convert to numpy array

        # Normalize the pixel values to the range [0, 1]
        first_image_normalized = (image_0_tensor - np.min(image_0_tensor)) / (
            np.max(image_0_tensor) - np.min(image_0_tensor)
        )
        second_image_normalized = (image_1_tensor - np.min(image_1_tensor)) / (
            np.max(image_1_tensor) - np.min(image_1_tensor)
        )
        third_image_normalized = (image_2_tensor - np.min(image_2_tensor)) / (
            np.max(image_2_tensor) - np.min(image_2_tensor)
        )

        # Create a figure with subplots
        fig, axs = plt.subplots(1, 3, figsize=(20, 10))

        # Plot the first image and its projected coordinates
        axs[0].imshow(first_image_normalized)
        for i, (y, x) in enumerate(cam_0_yx_coordinates):
            axs[0].text(x, y, str(i), color="red", fontsize=10, ha="center", va="center")
        axs[0].set_title("First Image with Cam 0 Coordinates")
        axs[0].axis("off")

        # Plot the second image and its projected coordinates
        axs[1].imshow(second_image_normalized)
        for i, (y, x) in enumerate(projected_coordinates_0_1):
            if 0 <= y < H and 0 <= x < W:
                axs[1].text(x, y, str(i), color="blue", fontsize=10, ha="center", va="center")
        axs[1].set_title("Second Image with Projected Coordinates")
        axs[1].axis("off")

        # Plot the third image and its projected coordinates
        axs[2].imshow(third_image_normalized)
        for i, (y, x) in enumerate(projected_coordinates_0_2):
            if 0 <= y < H and 0 <= x < W:
                axs[2].text(x, y, str(i), color="green", fontsize=10, ha="center", va="center")
        axs[2].set_title("Third Image with Projected Coordinates")
        axs[2].axis("off")

        # Show the plots
        plt.tight_layout()
        plt.show()

        ########################################################################################
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

    def transform_world_to_camera(self, homogeneous_world_coordinates, cam_to_world_3x4):
        # Ensure the input tensors are on the same device and dtype
        device = homogeneous_world_coordinates.device
        dtype = homogeneous_world_coordinates.dtype

        cam_to_world_3x4 = cam_to_world_3x4.to(device).to(dtype)

        # Create a 4x4 version of the 3x4 camera-to-world matrix
        cam_to_world_4x4 = torch.zeros((4, 4), dtype=dtype, device=device)
        cam_to_world_4x4[:3, :] = cam_to_world_3x4
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

        # Transform the point from world to camera coordinates
        homogeneous_camera_coordinates = torch.matmul(world_to_cam_4x4, homogeneous_world_coordinates)

        # Normalize to get the x, y, z coordinates in camera coordinates
        camera_coordinates = homogeneous_camera_coordinates[:3] / homogeneous_camera_coordinates[3]

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

        # Round the coordinates to the nearest integers
        rounded_coordinates = torch.round(projected_coordinates).long()

        # Clip the coordinates to be within valid range
        rounded_coordinates[:, 0] = torch.clamp(rounded_coordinates[:, 0], 0, depth_image.shape[1] - 1)
        rounded_coordinates[:, 1] = torch.clamp(rounded_coordinates[:, 1], 0, depth_image.shape[0] - 1)

        # Move rounded_coordinates to the same device as depth_image
        rounded_coordinates = rounded_coordinates.to(depth_image.device)

        # Fetch the depth values
        depth_values = depth_image[rounded_coordinates[:, 1], rounded_coordinates[:, 0], :]

        return depth_values

    def deproject_pixels_to_world(self, cam_0_depth, cam_0_to_world, intrinsics, cam_0_yx_coordinates):
        device = cam_0_depth.device
        dtype = cam_0_depth.dtype
        print("device:", device)
        print("dtype:", dtype)
        print(torch.isnan(cam_0_yx_coordinates).any(), torch.isinf(cam_0_yx_coordinates).any())
        print(cam_0_yx_coordinates.shape)

        # Convert all inputs to the same device and dtype
        camera_to_world = cam_0_to_world.to(device).to(dtype)
        intrinsics = intrinsics.to(device).to(dtype)
        cam_0_yx_coordinates = cam_0_yx_coordinates.to(device).to(dtype).long()
        depth_values = cam_0_depth[cam_0_yx_coordinates[:, 0], cam_0_yx_coordinates[:, 1], 0].to(device).to(dtype)

        homogeneous_pixel_coordinates = torch.cat(
            [
                cam_0_yx_coordinates.to(device).to(dtype),
                torch.ones(cam_0_yx_coordinates.shape[0], 1).to(device).to(dtype),
            ],
            dim=1,
        )

        homogeneous_pixel_coordinates[:, :2] = homogeneous_pixel_coordinates[:, :2].flip(dims=[1])

        # Deproject to camera space
        inv_intrinsics = torch.inverse(intrinsics).to(device).to(dtype)
        homogeneous_pixel_coordinates = homogeneous_pixel_coordinates.double()
        inv_intrinsics = inv_intrinsics.double()
        depth_values = depth_values.double()

        cam_0_coordinates = torch.matmul(homogeneous_pixel_coordinates, inv_intrinsics.T) * depth_values.unsqueeze(-1)

        # Create homogeneous coordinates in camera space
        homogeneous_cam_coordinates_3D = torch.cat(
            [
                cam_0_coordinates,
                torch.ones(cam_0_coordinates.shape[0], 1, dtype=torch.float64, device="cuda:0").to(device),
            ],
            dim=1,
        )

        world_coordinates = torch.matmul(homogeneous_cam_coordinates_3D, camera_to_world.T)

        return world_coordinates

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
