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
Datamanager.
"""

from __future__ import annotations

from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from functools import cached_property
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    Union,
    cast,
    ForwardRef,
    get_origin,
    get_args,
)

import torch
from torch import nn
from torch.nn import Parameter
from torch.utils.data.distributed import DistributedSampler
from typing_extensions import TypeVar

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.cameras.cameras import CameraType
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.configs.dataparser_configs import AnnotatedDataParserUnion
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.pixel_samplers import (
    EquirectangularPixelSampler,
    PatchPixelSampler,
    PixelSampler,
)
from nerfstudio.data.utils.dataloaders import (
    CacheDataloader,
    FixedIndicesEvalDataloader,
    RandIndicesEvalDataloader,
)
from nerfstudio.data.utils.nerfstudio_collate import nerfstudio_collate
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes
from nerfstudio.model_components.ray_generators import RayGenerator
from nerfstudio.utils.misc import IterableWrapper
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils.misc import get_orig_class

from PIL import Image
import numpy as np
import os

def variable_res_collate(batch: List[Dict]) -> Dict:
    """Default collate function for the cached dataloader.
    Args:
        batch: Batch of samples from the dataset.
    Returns:
        Collated batch.
    """
    images = []
    imgdata_lists = defaultdict(list)
    for data in batch:
        image = data.pop("image")
        images.append(image)
        topop = []
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                # if the value has same height and width as the image, assume that it should be collated accordingly.
                if len(val.shape) >= 2 and val.shape[:2] == image.shape[:2]:
                    imgdata_lists[key].append(val)
                    topop.append(key)
        # now that iteration is complete, the image data items can be removed from the batch
        for key in topop:
            del data[key]

    new_batch = nerfstudio_collate(batch)
    new_batch["image"] = images
    new_batch.update(imgdata_lists)

    return new_batch


@dataclass
class DataManagerConfig(InstantiateConfig):
    """Configuration for data manager instantiation; DataManager is in charge of keeping the train/eval dataparsers;
    After instantiation, data manager holds both train/eval datasets and is in charge of returning unpacked
    train/eval data at each iteration
    """

    _target: Type = field(default_factory=lambda: DataManager)
    """Target class to instantiate."""
    data: Optional[Path] = None
    """Source of data, may not be used by all models."""
    camera_optimizer: Optional[CameraOptimizerConfig] = None
    """Specifies the camera pose optimizer used during training. Helpful if poses are noisy."""
    masks_on_gpu: Optional[bool] = None
    """Process masks on GPU for speed at the expense of memory, if True."""


class DataManager(nn.Module):
    """Generic data manager's abstract class

    This version of the data manager is designed be a monolithic way to load data and latents,
    especially since this may contain learnable parameters which need to be shared across the train
    and test data managers. The idea is that we have setup methods for train and eval separately and
    this can be a combined train/eval if you want.

    Usage:
    To get data, use the next_train and next_eval functions.
    This data manager's next_train and next_eval methods will return 2 things:

    1. A Raybundle: This will contain the rays we are sampling, with latents and
        conditionals attached (everything needed at inference)
    2. A "batch" of auxiliary information: This will contain the mask, the ground truth
        pixels, etc needed to actually train, score, etc the model

    Rationale:
    Because of this abstraction we've added, we can support more NeRF paradigms beyond the
    vanilla nerf paradigm of single-scene, fixed-images, no-learnt-latents.
    We can now support variable scenes, variable number of images, and arbitrary latents.


    Train Methods:
        setup_train: sets up for being used as train
        iter_train: will be called on __iter__() for the train iterator
        next_train: will be called on __next__() for the training iterator
        get_train_iterable: utility that gets a clean pythonic iterator for your training data

    Eval Methods:
        setup_eval: sets up for being used as eval
        iter_eval: will be called on __iter__() for the eval iterator
        next_eval: will be called on __next__() for the eval iterator
        get_eval_iterable: utility that gets a clean pythonic iterator for your eval data


    Attributes:
        train_count (int): the step number of our train iteration, needs to be incremented manually
        eval_count (int): the step number of our eval iteration, needs to be incremented manually
        train_dataset (Dataset): the dataset for the train dataset
        eval_dataset (Dataset): the dataset for the eval dataset
        includes_time (bool): whether the dataset includes time information

        Additional attributes specific to each subclass are defined in the setup_train and setup_eval
        functions.

    """

    train_dataset: Optional[InputDataset] = None
    eval_dataset: Optional[InputDataset] = None
    train_sampler: Optional[DistributedSampler] = None
    eval_sampler: Optional[DistributedSampler] = None
    includes_time: bool = False

    def __init__(self):
        """Constructor for the DataManager class.

        Subclassed DataManagers will likely need to override this constructor.

        If you aren't manually calling the setup_train and setup_eval functions from an overriden
        constructor, that you call super().__init__() BEFORE you initialize any
        nn.Modules or nn.Parameters, but AFTER you've already set all the attributes you need
        for the setup functions."""
        super().__init__()
        self.train_count = 0
        self.eval_count = 0
        if self.train_dataset and self.test_mode != "inference":
            self.setup_train()
        if self.eval_dataset and self.test_mode != "inference":
            self.setup_eval()

    def forward(self):
        """Blank forward method

        This is an nn.Module, and so requires a forward() method normally, although in our case
        we do not need a forward() method"""
        raise NotImplementedError

    def iter_train(self):
        """The __iter__ function for the train iterator.

        This only exists to assist the get_train_iterable function, since we need to pass
        in an __iter__ function for our trivial iterable that we are making."""
        self.train_count = 0

    def iter_eval(self):
        """The __iter__ function for the eval iterator.

        This only exists to assist the get_eval_iterable function, since we need to pass
        in an __iter__ function for our trivial iterable that we are making."""
        self.eval_count = 0

    def get_train_iterable(self, length=-1) -> IterableWrapper:
        """Gets a trivial pythonic iterator that will use the iter_train and next_train functions
        as __iter__ and __next__ methods respectively.

        This basically is just a little utility if you want to do something like:
        |    for ray_bundle, batch in datamanager.get_train_iterable():
        |        <eval code here>
        since the returned IterableWrapper is just an iterator with the __iter__ and __next__
        methods (methods bound to our DataManager instance in this case) specified in the constructor.
        """
        return IterableWrapper(self.iter_train, self.next_train, length)

    def get_eval_iterable(self, length=-1) -> IterableWrapper:
        """Gets a trivial pythonic iterator that will use the iter_eval and next_eval functions
        as __iter__ and __next__ methods respectively.

        This basically is just a little utility if you want to do something like:
        |    for ray_bundle, batch in datamanager.get_eval_iterable():
        |        <eval code here>
        since the returned IterableWrapper is just an iterator with the __iter__ and __next__
        methods (methods bound to our DataManager instance in this case) specified in the constructor.
        """
        return IterableWrapper(self.iter_eval, self.next_eval, length)

    @abstractmethod
    def setup_train(self):
        """Sets up the data manager for training.

        Here you will define any subclass specific object attributes from the attribute"""

    @abstractmethod
    def setup_eval(self):
        """Sets up the data manager for evaluation"""

    @abstractmethod
    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train data manager.

        Args:
            step: the step number of the eval image to retrieve
        Returns:
            A tuple of the ray bundle for the image, and a dictionary of additional batch information
            such as the groundtruth image.
        """
        raise NotImplementedError

    @abstractmethod
    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the eval data manager.

        Args:
            step: the step number of the eval image to retrieve
        Returns:
            A tuple of the ray bundle for the image, and a dictionary of additional batch information
            such as the groundtruth image.
        """
        raise NotImplementedError

    @abstractmethod
    def next_eval_image(self, step: int) -> Tuple[int, RayBundle, Dict]:
        """Retrieve the next eval image.

        Args:
            step: the step number of the eval image to retrieve
        Returns:
            A tuple of the step number, the ray bundle for the image, and a dictionary of
            additional batch information such as the groundtruth image.
        """
        raise NotImplementedError

    @abstractmethod
    def get_train_rays_per_batch(self) -> int:
        """Returns the number of rays per batch for training."""
        raise NotImplementedError

    @abstractmethod
    def get_eval_rays_per_batch(self) -> int:
        """Returns the number of rays per batch for evaluation."""
        raise NotImplementedError

    @abstractmethod
    def get_datapath(self) -> Path:
        """Returns the path to the data. This is used to determine where to save camera paths."""

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        """Returns a list of callbacks to be used during training."""
        return []

    @abstractmethod
    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Get the param groups for the data manager.

        Returns:
            A list of dictionaries containing the data manager's param groups.
        """
        return {}


@dataclass
class VanillaDataManagerConfig(DataManagerConfig):
    """A basic data manager"""

    _target: Type = field(default_factory=lambda: VanillaDataManager)
    """Target class to instantiate."""
    dataparser: AnnotatedDataParserUnion = BlenderDataParserConfig()
    """Specifies the dataparser used to unpack the data."""
    train_num_rays_per_batch: int = 1024
    """Number of rays per batch to use per training iteration."""
    train_num_images_to_sample_from: int = -1
    """Number of images to sample during training iteration."""
    train_num_times_to_repeat_images: int = -1
    """When not training on all images, number of iterations before picking new
    images. If -1, never pick new images."""
    eval_num_rays_per_batch: int = 1024
    """Number of rays per batch to use per eval iteration."""
    eval_num_images_to_sample_from: int = -1
    """Number of images to sample during eval iteration."""
    eval_num_times_to_repeat_images: int = -1
    """When not evaluating on all images, number of iterations before picking
    new images. If -1, never pick new images."""
    eval_image_indices: Optional[Tuple[int, ...]] = (0,)
    """Specifies the image indices to use during eval; if None, uses all."""
    camera_optimizer: CameraOptimizerConfig = CameraOptimizerConfig()
    """Specifies the camera pose optimizer used during training. Helpful if poses are noisy, such as for data from
    Record3D."""
    collate_fn: Callable[[Any], Any] = cast(Any, staticmethod(nerfstudio_collate))
    """Specifies the collate function to use for the train and eval dataloaders."""
    camera_res_scale_factor: float = 1.0
    """The scale factor for scaling spatial data such as images, mask, semantics
    along with relevant information about camera intrinsics
    """
    patch_size: int = 1
    """Size of patch to sample from. If >1, patch-based sampling will be used."""


TDataset = TypeVar("TDataset", bound=InputDataset, default=InputDataset)


class VanillaDataManager(DataManager, Generic[TDataset]):
    """Basic stored data manager implementation.

    This is pretty much a port over from our old dataloading utilities, and is a little jank
    under the hood. We may clean this up a little bit under the hood with more standard dataloading
    components that can be strung together, but it can be just used as a black box for now since
    only the constructor is likely to change in the future, or maybe passing in step number to the
    next_train and next_eval functions.

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: VanillaDataManagerConfig
    train_dataset: TDataset
    eval_dataset: TDataset
    train_dataparser_outputs: DataparserOutputs
    train_pixel_sampler: Optional[PixelSampler] = None
    eval_pixel_sampler: Optional[PixelSampler] = None

    def __init__(
        self,
        config: VanillaDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,
    ):
        self.config = config
        self.device = device
        self.world_size = world_size
        self.local_rank = local_rank
        self.sampler = None
        self.test_mode = test_mode
        self.test_split = "test" if test_mode in ["test", "inference"] else "val"
        self.dataparser_config = self.config.dataparser
        if self.config.data is not None:
            self.config.dataparser.data = Path(self.config.data)
        else:
            self.config.data = self.config.dataparser.data
        self.dataparser = self.dataparser_config.setup()
        if test_mode == "inference":
            self.dataparser.downscale_factor = 1  # Avoid opening images
        self.includes_time = self.dataparser.includes_time
        self.train_dataparser_outputs: DataparserOutputs = self.dataparser.get_dataparser_outputs(split="train")

        self.train_dataset = self.create_train_dataset()
        self.eval_dataset = self.create_eval_dataset()
        self.exclude_batch_keys_from_device = self.train_dataset.exclude_batch_keys_from_device
        if self.config.masks_on_gpu is True:
            self.exclude_batch_keys_from_device.remove("mask")

        if self.train_dataparser_outputs is not None:
            cameras = self.train_dataparser_outputs.cameras
            if len(cameras) > 1:
                for i in range(1, len(cameras)):
                    if cameras[0].width != cameras[i].width or cameras[0].height != cameras[i].height:
                        CONSOLE.print("Variable resolution, using variable_res_collate")
                        self.config.collate_fn = variable_res_collate
                        break
        super().__init__()

    @cached_property
    def dataset_type(self) -> Type[TDataset]:
        """Returns the dataset type passed as the generic argument"""
        default: Type[TDataset] = cast(TDataset, TDataset.__default__)  # type: ignore
        orig_class: Type[VanillaDataManager] = get_orig_class(self, default=None)  # type: ignore
        if type(self) is VanillaDataManager and orig_class is None:
            return default
        if orig_class is not None and get_origin(orig_class) is VanillaDataManager:
            return get_args(orig_class)[0]

        # For inherited classes, we need to find the correct type to instantiate
        for base in getattr(self, "__orig_bases__", []):
            if get_origin(base) is VanillaDataManager:
                for value in get_args(base):
                    if isinstance(value, ForwardRef):
                        if value.__forward_evaluated__:
                            value = value.__forward_value__
                        elif value.__forward_module__ is None:
                            value.__forward_module__ = type(self).__module__
                            value = getattr(value, "_evaluate")(None, None, set())
                    assert isinstance(value, type)
                    if issubclass(value, InputDataset):
                        return cast(Type[TDataset], value)
        return default

    def create_train_dataset(self) -> TDataset:
        """Sets up the data loaders for training"""
        return self.dataset_type(
            dataparser_outputs=self.train_dataparser_outputs,
            scale_factor=self.config.camera_res_scale_factor,
        )

    def create_eval_dataset(self) -> TDataset:
        """Sets up the data loaders for evaluation"""
        return self.dataset_type(
            dataparser_outputs=self.dataparser.get_dataparser_outputs(split=self.test_split),
            scale_factor=self.config.camera_res_scale_factor,
        )

    def _get_pixel_sampler(self, dataset: TDataset, *args: Any, **kwargs: Any) -> PixelSampler:
        """Infer pixel sampler to use."""
        if self.config.patch_size > 1:
            return PatchPixelSampler(*args, **kwargs, patch_size=self.config.patch_size)

        # If all images are equirectangular, use equirectangular pixel sampler
        is_equirectangular = dataset.cameras.camera_type == CameraType.EQUIRECTANGULAR.value
        if is_equirectangular.all():
            return EquirectangularPixelSampler(*args, **kwargs)
        # Otherwise, use the default pixel sampler
        if is_equirectangular.any():
            CONSOLE.print("[bold yellow]Warning: Some cameras are equirectangular, but using default pixel sampler.")
        return PixelSampler(*args, **kwargs)

    def setup_train(self):
        """Sets up the data loaders for training"""
        assert self.train_dataset is not None
        CONSOLE.print("Setting up training dataset...")
        self.train_image_dataloader = CacheDataloader(
            self.train_dataset,
            num_images_to_sample_from=self.config.train_num_images_to_sample_from,
            num_times_to_repeat_images=self.config.train_num_times_to_repeat_images,
            device=self.device,
            num_workers=self.world_size * 4,
            pin_memory=True,
            collate_fn=self.config.collate_fn,
            exclude_batch_keys_from_device=self.exclude_batch_keys_from_device,
        )
        self.iter_train_image_dataloader = iter(self.train_image_dataloader)
        self.train_pixel_sampler = self._get_pixel_sampler(self.train_dataset, self.config.train_num_rays_per_batch)
        self.train_camera_optimizer = self.config.camera_optimizer.setup(
            num_cameras=self.train_dataset.cameras.size, device=self.device
        )
        self.train_ray_generator = RayGenerator(
            self.train_dataset.cameras.to(self.device),
            self.train_camera_optimizer,
        )
        # for loading full images
        self.fixed_indices_train_dataloader = FixedIndicesEvalDataloader(
            input_dataset=self.train_dataset,
            device=self.device,
            num_workers=self.world_size * 2,
            shuffle=False,
        )
        self.iter_fixed_indices_train_dataloader = iter(self.fixed_indices_train_dataloader)

    def setup_eval(self):
        """Sets up the data loader for evaluation"""
        assert self.eval_dataset is not None
        CONSOLE.print("Setting up evaluation dataset...")
        self.eval_image_dataloader = CacheDataloader(
            self.eval_dataset,
            num_images_to_sample_from=self.config.eval_num_images_to_sample_from,
            num_times_to_repeat_images=self.config.eval_num_times_to_repeat_images,
            device=self.device,
            num_workers=self.world_size * 4,
            pin_memory=True,
            collate_fn=self.config.collate_fn,
            exclude_batch_keys_from_device=self.exclude_batch_keys_from_device,
        )
        self.iter_eval_image_dataloader = iter(self.eval_image_dataloader)
        self.eval_pixel_sampler = self._get_pixel_sampler(self.eval_dataset, self.config.eval_num_rays_per_batch)
        self.eval_camera_optimizer = self.config.camera_optimizer.setup(
            num_cameras=self.eval_dataset.cameras.size, device=self.device
        )
        self.eval_ray_generator = RayGenerator(
            self.eval_dataset.cameras.to(self.device),
            self.eval_camera_optimizer,
        )
        # for loading full images
        self.fixed_indices_eval_dataloader = FixedIndicesEvalDataloader(
            input_dataset=self.eval_dataset,
            device=self.device,
            num_workers=self.world_size * 4,
        )
        self.eval_dataloader = RandIndicesEvalDataloader(
            input_dataset=self.eval_dataset,
            device=self.device,
            num_workers=self.world_size * 4,
        )

    def arange_pixels(self, resolution=(128, 128), batch_size=1, image_range=(-1.0, 1.0), device=torch.device("cpu")):
        """Arranges pixels for given resolution in range image_range.

        The function returns the unscaled pixel locations as integers and the
        scaled float values.

        Args:
            resolution (tuple): image resolution
            batch_size (int): batch size
            image_range (tuple): range of output points (default [-1, 1])
            device (torch.device): device to use
        """
        h, w = resolution

        # Arrange pixel location in scale resolution
        pixel_locations = torch.meshgrid(torch.arange(0, h, device=device), torch.arange(0, w, device=device))
        pixel_locations = (
            torch.stack([pixel_locations[1], pixel_locations[0]], dim=-1).long().view(1, -1, 2).repeat(batch_size, 1, 1)
        )
        pixel_scaled = pixel_locations.clone().float()

        # Shift and scale points to match image_range
        scale = image_range[1] - image_range[0]
        loc = (image_range[1] - image_range[0]) / 2
        pixel_scaled[:, :, 0] = scale * pixel_scaled[:, :, 0] / (w - 1) - loc
        pixel_scaled[:, :, 1] = scale * pixel_scaled[:, :, 1] / (h - 1) - loc
        return pixel_locations, pixel_scaled

    def to_pytorch(self, tensor, return_type=False):
        """Converts input tensor to pytorch.

        Args:
            tensor (tensor): Numpy or Pytorch tensor
            return_type (bool): whether to return input type
        """
        is_numpy = False
        import numpy as np

        if type(tensor) == np.ndarray:
            tensor = torch.from_numpy(tensor)
            is_numpy = True

        tensor = tensor.clone()
        if return_type:
            return tensor, is_numpy
        return tensor

    def transform_to_world(
        self, pixels, depth, camera_mat, world_mat=None, scale_mat=None, invert=True, device=torch.device("cuda")
    ):
        """Transforms pixel positions p with given depth value d to world coordinates.

        Args:
            pixels (tensor): pixel tensor of size B x N x 2
            depth (tensor): depth tensor of size B x N x 1
            camera_mat (tensor): camera matrix
            world_mat (tensor): world matrix
            scale_mat (tensor): scale matrix
            invert (bool): whether to invert matrices (default: true)
        """
        assert pixels.shape[-1] == 2
        if world_mat is None:
            world_mat = torch.tensor(
                [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]], dtype=torch.float32, device=device
            )
        if scale_mat is None:
            scale_mat = torch.tensor(
                [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]], dtype=torch.float32, device=device
            )
        # Convert to pytorch
        pixels, is_numpy = self.to_pytorch(pixels, True)
        depth = self.to_pytorch(depth).to(pixels.device)
        camera_mat = self.to_pytorch(camera_mat).to(pixels.device)
        world_mat = self.to_pytorch(world_mat)
        scale_mat = self.to_pytorch(scale_mat)

        # Invert camera matrices
        if invert:
            camera_mat = torch.inverse(camera_mat)
            world_mat = torch.inverse(world_mat)
            scale_mat = torch.inverse(scale_mat)

        # Transform pixels to homogen coordinates
        pixels = pixels.permute(0, 2, 1)
        pixels = torch.cat([pixels, torch.ones_like(pixels)], dim=1)

        # Project pixels into camera space
        # pixels[:, :3] = pixels[:, :3] * depth.permute(0, 2, 1)
        pixels_depth = pixels.clone()
        pixels_depth[:, :3] = pixels[:, :3] * depth.permute(0, 2, 1)

        # Transform pixels to world space
        p_world = scale_mat @ world_mat @ camera_mat @ pixels_depth

        # Transform p_world back to 3D coordinates
        p_world = p_world[:, :3].permute(0, 2, 1)

        if is_numpy:
            p_world = p_world.numpy()
        return p_world

    def get_tensor_values(self, tensor, p, mode="nearest", scale=True, detach=True, detach_p=True, align_corners=False):
        """
        Returns values from tensor at given location p.

        Args:
            tensor (tensor): tensor of size B x C x H x W
            p (tensor): position values scaled between [-1, 1] and
                of size B x N x 2
            mode (str): interpolation mode
            scale (bool): whether to scale p from image coordinates to [-1, 1]
            detach (bool): whether to detach the output
            detach_p (bool): whether to detach p
            align_corners (bool): whether to align corners for grid_sample
        """

        batch_size, _, h, w = tensor.shape

        # p = pe.clone()
        # p = pe
        if detach_p:
            p = p.detach()
        if scale:
            p[:, :, 0] = 2.0 * p[:, :, 0] / w - 1
            p[:, :, 1] = 2.0 * p[:, :, 1] / h - 1
        p = p.unsqueeze(1)
        values = torch.nn.functional.grid_sample(tensor, p, mode=mode, align_corners=align_corners)
        values = values.squeeze(2)

        if detach:
            values = values.detach()
        values = values.permute(0, 2, 1)

        return values

    def project_to_cam(self, points, camera_mat, device):
        """
        points: (B, N, 3)
        camera_mat: (B, 4, 4)
        """
        # breakpoint()
        B, N, D = points.size()
        points, is_numpy = self.to_pytorch(points, True)
        points = points.permute(0, 2, 1)
        points = torch.cat([points, torch.ones(B, 1, N, device=device)], dim=1)

        xy_ref = camera_mat @ points

        xy_ref = xy_ref[:, :3].permute(0, 2, 1)
        xy_ref = xy_ref[..., :2] / xy_ref[..., 2:]

        valid_points = xy_ref.abs().max(dim=-1)[0] <= 1
        valid_mask = valid_points.unsqueeze(-1).bool()
        if is_numpy:
            xy_ref = xy_ref.numpy()
        return xy_ref, valid_mask

    def get_pc_loss(self, Xt, Yt):
        # compute  error
        match_method = "dense"
        if match_method == "dense":
            loss1 = self.comp_point_point_error(Xt[0].permute(1, 0), Yt[0].permute(1, 0))
            loss2 = self.comp_point_point_error(Yt[0].permute(1, 0), Xt[0].permute(1, 0))
            loss = loss1 + loss2
        return loss

    def comp_closest_pts_idx_with_split(self, pts_src, pts_des):
        """
        :param pts_src:     (3, S)
        :param pts_des:     (3, D)
        :param num_split:
        :return:
        """
        pts_src_list = torch.split(pts_src, 500000, dim=1)
        idx_list = []
        for pts_src_sec in pts_src_list:
            import numpy as np

            diff = pts_src_sec[:, :, np.newaxis] - pts_des[:, np.newaxis, :]  # (3, S, 1) - (3, 1, D) -> (3, S, D)
            dist = torch.linalg.norm(diff, dim=0)  # (S, D)
            closest_idx = torch.argmin(dist, dim=1)  # (S,)
            idx_list.append(closest_idx)
        closest_idx = torch.cat(idx_list)
        return closest_idx

    def comp_point_point_error(self, Xt, Yt):
        closest_idx = self.comp_closest_pts_idx_with_split(Xt, Yt)
        pt_pt_vec = Xt - Yt[:, closest_idx]  # (3, S) - (3, S) -> (3, S)
        pt_pt_dist = torch.linalg.norm(pt_pt_vec, dim=0)
        eng = torch.mean(pt_pt_dist)
        return eng

    def get_rgb_s_loss(self, rgb1, rgb2, valid_points):
        diff_img = (rgb1 - rgb2).abs()
        diff_img = diff_img.clamp(0, 1)
        compute_ssim_loss = SSIM().to("cuda")
        ssim_map = compute_ssim_loss(rgb1, rgb2)
        diff_img = 0.15 * diff_img + 0.85 * ssim_map
        loss = self.mean_on_mask(diff_img, valid_points)
        return loss

    # compute mean value given a binary mask
    def mean_on_mask(self, diff, valid_mask):
        mask = valid_mask.expand_as(diff)
        if mask.sum() > 0:
            mean_value = (diff[mask]).sum() / mask.sum()
            # mean_value = (diff * mask).sum() / mask.sum()
        else:
            print("============invalid mask==========")
            mean_value = torch.tensor(0).float().cuda()
        return mean_value

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        image_batch = next(self.iter_train_image_dataloader)

        image_idx_6_data = image_batch["image_idx"][0]
        image_6_data = image_batch["image"][0]
        depth_image_6_data = image_batch["depth_image"][0]

        image_idx_5_data = image_batch["image_idx"][1]
        image_5_data = image_batch["image"][1]
        depth_image_5_data = image_batch["depth_image"][1]

        Ks = self.train_dataset.cameras.get_intrinsics_matrices()
        # K = Ks[image_idx_6_data].to(self.device)
        import numpy as np

        K_num = np.array(
            [[2 * 702.0630 / 913, 0, 0, 0], [0, -2 * 701.9382 / 1138, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        ).astype(np.float32)
        K = torch.tensor(K_num, device=self.device)
        print(K)
        # K_h = torch.zeros((4, 4), device="cuda:0")
        # K_h[:3, :3] = K
        # K_h[3, 3] = 1.0
        K = K.unsqueeze(0)
        cam5 = self.train_dataset.cameras[image_idx_6_data].camera_to_worlds
        cam6 = self.train_dataset.cameras[image_idx_5_data].camera_to_worlds

        # Convert to homogeneous representation (4x4 matrix)
        cam5_h = torch.cat([cam5, torch.tensor([[0.0, 0.0, 0.0, 1.0]])], dim=0).to(self.device)
        cam6_h = torch.cat([cam6, torch.tensor([[0.0, 0.0, 0.0, 1.0]])], dim=0).to(self.device)
        world_mat = torch.inverse(cam5_h).unsqueeze(0)

        ref_Rt = torch.inverse(cam6_h).unsqueeze(0).to(self.device)
        # ref_Rt = cam6_h.unsqueeze(0).to(self.device)

        print(ref_Rt.shape)
        d1 = depth_image_5_data.to(cam5_h.device)
        d2 = depth_image_6_data.to(cam6_h.device)
        img1 = image_5_data.to(cam5_h.device)
        img2 = image_6_data.to(cam6_h.device)

        # Rt_rel_12 = ref_Rt @ torch.inverse(cam5_h).to(self.device)
        Rt_rel_12 = ref_Rt @ torch.inverse(world_mat).to(self.device)
        # Rt_rel_12 = torch.inverse(ref_Rt) @ cam5_h.to(self.device)
        # Rt_rel_12 = torch.eye(4).to(self.device)
        # Rt_rel_12 = torch.inverse(cam5_h) @ torch.inverse(ref_Rt).to(self.device)
        R_rel_12 = Rt_rel_12[:, :3, :3]
        t_rel_12 = Rt_rel_12[:, :3, 3]

        ratio = 8
        h_depth, w_depth = d1.shape[1:]
        h_depth, w_depth, _ = d1.shape

        sample_resolution = (int(h_depth / ratio), int(w_depth / ratio))
        pixel_locations, p_pc = self.arange_pixels(resolution=sample_resolution, device=self.device)
        from torch.nn import functional as F

        d1_reshaped = d1.permute(2, 0, 1).unsqueeze(0)
        d2_reshaped = d2.permute(2, 0, 1).unsqueeze(0)
        d1 = F.interpolate(d1_reshaped, sample_resolution, mode="nearest")
        d2 = F.interpolate(d2_reshaped, sample_resolution, mode="nearest")
        pc1 = self.transform_to_world(p_pc, d1.view(1, -1, 1), K)
        pc2 = self.transform_to_world(p_pc, d2.view(1, -1, 1), K)
        img1_reshaped = img1.permute(2, 0, 1).unsqueeze(0).to(self.device)
        img2_reshaped = img2.permute(2, 0, 1).unsqueeze(0).to(self.device)
        img1 = F.interpolate(img1_reshaped, sample_resolution, mode="bilinear")
        img2 = F.interpolate(img2_reshaped, sample_resolution, mode="bilinear")
        rgb_pc1 = self.get_tensor_values(
            img1, p_pc, mode="bilinear", scale=False, detach=False, detach_p=False, align_corners=True
        )
        nl = 0.0001
        pc1_rotated = pc1 @ R_rel_12.transpose(1, 2) + t_rel_12
        mask_pc1_invalid = (-pc1_rotated[:, :, 2:] < nl).expand_as(pc1_rotated)
        pc1_rotated[mask_pc1_invalid] = nl
        pc1_rotated[mask_pc1_invalid] = nl
        p_reprojected, valid_mask = self.project_to_cam(pc1_rotated, K, device=self.device)
        p_reprojected = p_reprojected.to(img2.device)
        rgb_pc1_proj = self.get_tensor_values(
            img2, p_reprojected, mode="bilinear", scale=False, detach=False, detach_p=False, align_corners=True
        )
        rgb_pc1 = rgb_pc1.view(1, sample_resolution[0], sample_resolution[1], 3)
        rgb_pc1_proj = rgb_pc1_proj.view(1, sample_resolution[0], sample_resolution[1], 3)
        valid_mask = valid_mask.view(1, sample_resolution[0], sample_resolution[1], 1)

        valid_points = valid_mask
        X = pc1 @ R_rel_12.transpose(1, 2) + t_rel_12
        Y = pc2

        pc_loss = self.get_pc_loss(X, Y)
        rgb_s_loss = self.get_rgb_s_loss(rgb_pc1, rgb_pc1_proj, valid_points)



        Image.fromarray(((rgb_pc1[0] * 255).detach().cpu().numpy()).astype(np.uint8)).convert("RGB").save(
            os.path.join("%04d_img1.png" % (5))
        )
        Image.fromarray(((rgb_pc1_proj[0] * 255).detach().cpu().numpy()).astype(np.uint8)).convert("RGB").save(
            os.path.join("%04d_img2.png" % (6))
        )

        assert self.train_pixel_sampler is not None
        assert isinstance(image_batch, dict)
        batch = self.train_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]
        # Extract the image indices from the tensor (first column)
        image_indices = ray_indices[:, 0]

        # Find unique image indices and their counts
        unique_image_indices, counts = torch.unique(image_indices, return_counts=True)

        # Display the result
        for idx, count in zip(unique_image_indices, counts):
            print(f"Image index {idx} has {count} items.")

        ray_bundle = self.train_ray_generator(ray_indices)
        # Extract the image indices from the tensor (first column)
        image_indices = ray_bundle[:].camera_indices

        # Find unique image indices and their counts
        unique_image_indices, counts = torch.unique(image_indices, return_counts=True)

        # Display the result
        # for idx, count in zip(unique_image_indices, counts):
        #     print(f"Image index {idx} has {count} items.")

        # Find the indices where items have the same image index
        indices_where_same = [torch.nonzero(image_indices == idx).flatten() for idx in unique_image_indices]

        # Display the result
        for idx, indices_list in zip(unique_image_indices, indices_where_same):
            print(f"Image index {idx} appears at the following indices: {indices_list.tolist()}")
        return ray_bundle, batch

    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the eval dataloader."""
        self.eval_count += 1
        image_batch = next(self.iter_eval_image_dataloader)
        assert self.eval_pixel_sampler is not None
        assert isinstance(image_batch, dict)
        batch = self.eval_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]
        ray_bundle = self.eval_ray_generator(ray_indices)
        return ray_bundle, batch

    def next_eval_image(self, step: int) -> Tuple[int, RayBundle, Dict]:
        for camera_ray_bundle, batch in self.eval_dataloader:
            assert camera_ray_bundle.camera_indices is not None
            image_idx = int(camera_ray_bundle.camera_indices[0, 0, 0])
            return image_idx, camera_ray_bundle, batch
        raise ValueError("No more eval images")

    def get_train_rays_per_batch(self) -> int:
        return self.config.train_num_rays_per_batch

    def get_eval_rays_per_batch(self) -> int:
        return self.config.eval_num_rays_per_batch

    def get_datapath(self) -> Path:
        return self.config.dataparser.data

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Get the param groups for the data manager.
        Returns:
            A list of dictionaries containing the data manager's param groups.
        """
        param_groups = {}

        camera_opt_params = list(self.train_camera_optimizer.parameters())
        if self.config.camera_optimizer.mode != "off":
            assert len(camera_opt_params) > 0
            param_groups[self.config.camera_optimizer.param_group] = camera_opt_params
        else:
            assert len(camera_opt_params) == 0

        return param_groups


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images"""

    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01**2
        self.C2 = 0.03**2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x**2) - mu_x**2
        sigma_y = self.sig_y_pool(y**2) - mu_y**2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x**2 + mu_y**2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)
