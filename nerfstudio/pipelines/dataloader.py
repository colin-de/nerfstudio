import copy

import numpy as np
import torch
from skimage.transform import resize
from torch.utils.data import Dataset


def eye_like(tensor):
    return torch.ones_like(tensor) * torch.eye(tensor.shape[-1]).to(tensor.device)


class DepthRefineDataset(Dataset):
    def __init__(self, refinement_dicts):
        super().__init__()
        self.refinement_dicts = dict(refinement_dicts)

        self.num_frames = len(self.refinement_dicts)

        self.ref_idx = 0
        self.ref_rgb = self.refinement_dicts["rgb"][self.ref_idx]
        # self.ref_cam_coordinates = self.refinement_dicts["cam_coordinates"][self.ref_idx]
        self.ref_cam_depth = self.refinement_dicts["depth"][self.ref_idx]
        self.ref_pixel_coordinates = self.refinement_dicts["pixel_coordinates"][self.ref_idx]

    def __len__(self):
        return self.num_frames - 1

        # Repeat above for each query (qry) frame

    def __getitem__(self, idx):
        idx = idx + 1  # Never return reference frame, avoid division by zero
        # idx = self.frames[idx]  # Grab subset frame idx

        sample = dict()
        sample["frame_idx"] = idx

        rgb = self.refinement_dicts["rgb"][idx]
        depth = self.refinement_dicts["depth"][idx]
        cam_coordinates = self.refinement_dicts["cam_coordinates"][idx]
        pixel_coordinates = self.refinement_dicts["pixel_coordinates"][idx]

        sample["rgb"] = rgb
        sample["depth"] = depth
        # sample["cam_coordinates"] = cam_coordinates
        sample["pixel_coordinates"] = pixel_coordinates

        return sample
