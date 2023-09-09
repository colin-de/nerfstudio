import torch
import torch.nn as nn
import torch.nn.functional as F

# from utils.utils import *
from functools import partial
import numpy as np


class DepthRefinementModel(nn.Module):
    def __init__(self, ref_pts_3d_cam0, projected_coordinates_0_i_dict, projected_depth_cam_0_i_dict):
        super().__init__()

        self.ref_pts_3d_cam0 = ref_pts_3d_cam0
        self.projected_coordinates_0_i_dict = projected_coordinates_0_i_dict
        self.projected_depth_cam_0_i_dict = projected_depth_cam_0_i_dict
        self.num_frames = len(projected_coordinates_0_i_dict)
        self.frame_id_embedding = nn.Embedding(self.num_frames + 1, embedding_dim=30)
        self.net = ImplicitDepthNet()

    def forward(self, ref_depth, qry_depth, frame_idx):
        with torch.no_grad():
            pass

        ref_depth_refined = self.net(ref_depth)  # ref_depth(1, 43, 300), ref_depth_refined(1, 43, 300)
        qry_depth_refined = self.net(qry_depth)  # qry_depth(1, 43, 300), qry_depth_refined(1, 43, 300)

        return ref_depth_refined, qry_depth_refined


class ImplicitDepthNet(nn.Module):
    def __init__(self):
        super().__init__()

        M = 6
        self.positional_encoder = PositionalEncoding(num_encoding_functions=M)
        self.FCNet = FCBlock(
            in_features=(2 * M + 1) * 3 + 3,
            out_features=1,
            num_hidden_layers=4,
            hidden_features=256,
            outermost_linear=True,
            nonlinearity="relu",
        )

    def forward(self, depths):
        yx = torch.clone(depths[:, :3])  # rays(1, 43, 1), yx(1, 3, 1)
        yx[:, :2] += 5e-2 * torch.randn_like(yx[:, :2])
        yx[:, :2] = yx[:, :2] / self.dims  # Normalize to 0 to 1

        yx_encoded = self.positional_encoder(yx)  # yx_encoded(1, 39, 1)

        mlp_in = torch.cat((yx_encoded), dim=1)

        z_out = self.FCNet(mlp_in)  # ([1, 1, 1])

        mlp_rays = torch.clone(depths)  # ([1, 43, 1])
        z_out = mlp_rays[:, 2:3] + z_out  # ([1, 1, 1])
        mlp_rays[:, 2:3] = z_out  # replaces the original depth in mlp_rays with the estimated depth from the network.

        return mlp_rays


# Taken from: https://github.dev/computational-imaging/ACORN/
class Sine(nn.Module):
    def __init__(self, w0=30):
        super().__init__()
        self.w0 = w0

    def forward(self, input):
        return torch.sin(self.w0 * input)


def init_weights_zero(m):
    if type(m) == nn.Linear:
        if hasattr(m, "weight"):
            nn.init.zeros_(m.weight)


def init_weights_normal(m):
    if type(m) == nn.Linear:
        if hasattr(m, "weight"):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity="relu", mode="fan_in")


def init_weights_xavier(m):
    if type(m) == nn.Linear:
        if hasattr(m, "weight"):
            nn.init.xavier_normal_(m.weight)


def sine_init(m, w0=30):
    with torch.no_grad():
        if hasattr(m, "weight"):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-np.sqrt(6 / num_input) / w0, np.sqrt(6 / num_input) / w0)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, "weight"):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)


class FCBlock(nn.Module):
    """A fully connected neural network that also allows swapping out the weights when used with a hypernetwork.
    Can be used just as a normal neural network though, as well.
    """

    def __init__(
        self,
        in_features,
        out_features,
        num_hidden_layers,
        hidden_features,
        outermost_linear=True,
        nonlinearity="sine",
        weight_init=None,
        w0=30,
    ):
        super().__init__()

        self.first_layer_init = None

        # Dictionary that maps nonlinearity name to the respective function, initialization, and, if applicable,
        # special first-layer initialization scheme
        nls_and_inits = {
            "sine": (Sine(w0=w0), partial(sine_init, w0=w0), first_layer_sine_init),
            "relu": (nn.ReLU(inplace=True), init_weights_xavier, None),
        }

        nl, nl_weight_init, first_layer_init = nls_and_inits[nonlinearity]

        if weight_init is not None:  # Overwrite weight init if passed
            self.weight_init = weight_init
        else:
            self.weight_init = nl_weight_init

        self.net = []
        self.net.append(nn.Sequential(nn.Linear(in_features, hidden_features), nl))

        for i in range(num_hidden_layers):
            self.net.append(nn.Sequential(nn.Linear(hidden_features, hidden_features), nl))

        if outermost_linear:
            self.net.append(nn.Sequential(nn.Linear(hidden_features, out_features)))
        else:
            self.net.append(nn.Sequential(nn.Linear(hidden_features, out_features), nl))

        self.net = nn.Sequential(*self.net)
        if self.weight_init is not None:
            self.net.apply(self.weight_init)

        if first_layer_init is not None:  # Apply special initialization to first layer, if applicable.
            self.net[0].apply(first_layer_init)

    def forward(self, rays):
        B, C, N = rays.shape
        rays = rays.permute(0, 2, 1)  # B,N,C
        rays = rays.reshape(B * N, C)  # stack rays

        output = self.net(rays)
        output = output.reshape(B, N, 1)
        output = output.permute(0, 2, 1)
        return output


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        num_encoding_functions=6,
        include_input=True,
        log_sampling=True,
        normalize=False,
        input_dim=3,
        gaussian_pe=False,
        gaussian_variance=38,
    ):
        super().__init__()
        self.num_encoding_functions = num_encoding_functions
        self.include_input = include_input
        self.log_sampling = log_sampling
        self.normalize = normalize
        self.gaussian_pe = gaussian_pe
        self.normalization = None

        if self.gaussian_pe:
            # this needs to be registered as a parameter so that it is saved in the model state dict
            # and so that it is converted using .cuda(). Doesn't need to be trained though
            self.gaussian_weights = nn.Parameter(
                gaussian_variance * torch.randn(num_encoding_functions, input_dim), requires_grad=False
            )
        else:
            self.frequency_bands = None
            if self.log_sampling:
                self.frequency_bands = 2.0 ** torch.linspace(
                    0.0, self.num_encoding_functions - 1, self.num_encoding_functions
                )
            else:
                self.frequency_bands = torch.linspace(
                    2.0**0.0, 2.0 ** (self.num_encoding_functions - 1), self.num_encoding_functions
                )

            if normalize:
                self.normalization = torch.tensor(1 / self.frequency_bands)

    def forward(self, tensor) -> torch.Tensor:
        r"""Apply positional encoding to the input.

        Args:
            tensor (torch.Tensor): Input tensor to be positionally encoded.
            encoding_size (optional, int): Number of encoding functions used to compute
                a positional encoding (default: 6).
            include_input (optional, bool): Whether or not to include the input in the
                positional encoding (default: True).

        Returns:
        (torch.Tensor): Positional encoding of the input tensor.
        """

        encoding = [tensor] if self.include_input else []
        if self.gaussian_pe:
            for func in [torch.sin, torch.cos]:
                encoding.append(func(torch.matmul(tensor, self.gaussian_weights.T)))
        else:
            for idx, freq in enumerate(self.frequency_bands):
                for func in [torch.sin, torch.cos]:
                    if self.normalization is not None:
                        encoding.append(self.normalization[idx] * func(tensor * freq))
                    else:
                        encoding.append(func(tensor * freq))

        # Special case, for no positional encoding
        if len(encoding) == 1:
            return encoding[0]
        else:
            return torch.cat(encoding, dim=1)  # concat channels


class Conv2x(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=1, deconv=False, concat=True, bn=True, relu=True
    ):
        super().__init__()
        self.concat = concat

        self.conv1 = BasicConv(in_channels, out_channels, deconv, bn=bn, relu=True, kernel_size=2, stride=2, padding=0)

        if self.concat:
            self.conv2 = BasicConv(
                out_channels * 2, out_channels, False, bn, relu, kernel_size=kernel_size, stride=stride, padding=padding
            )
        else:
            self.conv2 = BasicConv(
                out_channels, out_channels, False, bn, relu, kernel_size=kernel_size, stride=stride, padding=padding
            )

    def forward(self, x, rem):
        x = self.conv1(x)
        if x.size() != rem.size():
            raise Exception("X size", x.size(), "!= rem size", rem.size())

        if self.concat:
            x = torch.cat((x, rem), 1)
        else:
            x = x + rem
        x = self.conv2(x)
        return x


class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, deconv=False, bn=True, relu=True, **kwargs):
        super().__init__()
        self.relu = relu
        self.use_bn = bn
        if deconv:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=False)

    def forward(self, x):
        x = self.conv(x)
        if self.relu:
            x = self.leaky_relu(x)
        return x
