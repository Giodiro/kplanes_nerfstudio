# Copyright 2022 The Nerfstudio Team. All rights reserved.
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
Fields for K-Planes (https://sarafridov.github.io/K-Planes/).
"""

from typing import Dict, Iterable, List, Optional, Tuple, Sequence

import torch
from rich.console import Console
from torch import nn
from torchtyping import TensorType

from nerfstudio.cameras.rays import RaySamples, Frustums
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.encodings import KPlanesEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field, shift_directions_for_tcnn

try:
    import tinycudann as tcnn
except ImportError:
    # tinycudann module doesn't exist
    pass

CONSOLE = Console(width=120)


def interpolate_ms_features(
    pts: torch.Tensor,
    grid_encodings: Iterable[KPlanesEncoding],
    concat_features: bool,
) -> torch.Tensor:
    """Combines/interpolates features across multiple dimensions and scales.

    Args:
        pts: Coordinates to query
        grid_encodings: Grid encodings to query
        concat_features: Whether to concatenate features at different scales

    Returns:
        Feature vectors
    """

    multi_scale_interp = [] if concat_features else 0.0
    for grid in grid_encodings:
        grid_features = grid(pts)

        if concat_features:
            multi_scale_interp.append(grid_features)
        else:
            multi_scale_interp = multi_scale_interp + grid_features

    if concat_features:
        multi_scale_interp = torch.cat(multi_scale_interp, dim=-1)

    return multi_scale_interp


class KPlanesField(Field):
    """K-Planes field.

    Args:
        aabb: Parameters of scene aabb bounds
        num_images: How many images exist in the dataset
        geo_feat_dim: Dimension of 'geometry' features. Controls output dimension of sigma network
        grid_base_resolution: Base grid resolution
        grid_feature_dim: Dimension of feature vectors stored in grid
        concat_across_scales: Whether to concatenate features at different scales
        multiscale_res: Multiscale grid resolutions
        spatial_distortion: Spatial distortion to apply to the scene
        appearance_embedding_dim: Dimension of appearance embedding. Set to 0 to disable
        use_average_appearance_embedding: Whether to use average appearance embedding or zeros for inference
        linear_decoder: Whether to use a linear decoder instead of an MLP
        linear_decoder_layers: Number of layers in linear decoder
    """

    def __init__(
        self,
        aabb: TensorType,
        num_images: int,
        geo_feat_dim: int = 15,  # TODO: This should be removed
        concat_across_scales: bool = True,  # TODO: Maybe this should be removed
        grid_base_resolution: Sequence[int] = (128, 128, 128),
        grid_feature_dim: int = 32,
        multiscale_res: Sequence[int] = (1, 2, 4),
        spatial_distortion: Optional[SpatialDistortion] = None,
        appearance_embedding_dim: int = 0,
        use_average_appearance_embedding: bool = True,
        linear_decoder: bool = False,
        linear_decoder_layers: Optional[int] = None,
    ) -> None:

        super().__init__()

        self.register_buffer("aabb", aabb)
        self.num_images = num_images
        self.geo_feat_dim = geo_feat_dim
        self.grid_base_resolution = list(grid_base_resolution)
        self.concat_across_scales = concat_across_scales
        self.spatial_distortion = spatial_distortion
        self.linear_decoder = linear_decoder
        self.has_time_planes = len(grid_base_resolution) > 3

        # Init planes
        self.grids = nn.ModuleList()
        for res in multiscale_res:
            # Resolution fix: multi-res only on spatial planes
            resolution = [r * res for r in self.grid_base_resolution[:3]] + self.grid_base_resolution[3:]
            self.grids.append(KPlanesEncoding(resolution, grid_feature_dim))
        self.feature_dim = (
            grid_feature_dim * len(multiscale_res) if self.concat_across_scales
            else grid_feature_dim
        )

        # Init appearance code-related parameters
        self.appearance_embedding_dim = appearance_embedding_dim
        if self.appearance_embedding_dim > 0:
            assert self.num_images is not None, "'num_images' must not be None when using appearance embedding"
            self.appearance_ambedding = Embedding(self.num_images, self.appearance_embedding_dim)
            self.use_average_appearance_embedding = use_average_appearance_embedding  # for test-time

        # Init decoder network
        if self.linear_decoder:
            assert linear_decoder_layers is not None
            # The NN learns a basis that is used instead of spherical harmonics
            # Input is an encoded view direction, output is weights for combining the color
            # features into RGB.
            # Architecture based on instant-NGP
            self.color_basis = tcnn.Network(
                n_input_dims=3 + self.appearance_embedding_dim,
                n_output_dims=3 * self.feature_dim,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 128,
                    "n_hidden_layers": linear_decoder_layers,
                },
            )
            # sigma_net just does a linear transformation on the features to get density
            self.sigma_net = tcnn.Network(
                n_input_dims=self.feature_dim,
                n_output_dims=1,
                network_config={
                    "otype": "CutlassMLP",
                    "activation": "None",
                    "output_activation": "None",
                    "n_neurons": 128,
                    "n_hidden_layers": 0,
                },
            )
        else:
            self.sigma_net = tcnn.Network(
                n_input_dims=self.feature_dim,
                n_output_dims=self.geo_feat_dim + 1,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 1,
                },
            )
            self.direction_encoding = tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "SphericalHarmonics",
                    "degree": 4,
                },
            )
            in_dim_color = (
                self.direction_encoding.n_output_dims + self.geo_feat_dim + self.appearance_embedding_dim
            )
            self.color_net = tcnn.Network(
                n_input_dims=in_dim_color,
                n_output_dims=3,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "Sigmoid",
                    "n_neurons": 64,
                    "n_hidden_layers": 2,
                },
            )

    def get_density(self, ray_samples: RaySamples) -> Tuple[TensorType, TensorType]:
        """Computes and returns the densities."""
        positions = ray_samples.frustums.get_positions()
        if self.spatial_distortion is not None:
            positions = self.spatial_distortion(positions)
            positions = positions / 2  # from [-2, 2] to [-1, 1]
        else:
            # From [0, 1] to [-1, 1]
            positions = SceneBox.get_normalized_positions(positions, self.aabb) * 2.0 - 1.0

        if self.has_time_planes:
            assert ray_samples.times is not None, "Initialized model with time-planes, but no time data is given"
            # Normalize timestamps from [0, 1] to [-1, 1]
            timestamps = ray_samples.times * 2.0 - 1.0
            positions = torch.cat((positions, timestamps), dim=-1)  # [n_rays, n_samples, 4]

        positions_flat = positions.view(-1, positions.shape[-1])
        features = interpolate_ms_features(
            positions_flat, grid_encodings=self.grids, concat_features=self.concat_across_scales
        )
        if len(features) < 1:
            features = torch.zeros((0, 1), device=features.device, requires_grad=True)
        if self.linear_decoder:
            density_before_activation = self.sigma_net(features).view(*ray_samples.frustums.shape, -1)
        else:
            features = self.sigma_net(features).view(*ray_samples.frustums.shape, -1)
            features, density_before_activation = torch.split(features, [self.geo_feat_dim, 1], dim=-1)

        # Rectifying the density with an exponential is much more stable than a ReLU or
        # softplus, because it enables high post-activation (float32) density outputs
        # from smaller internal (float16) parameters.
        density = trunc_exp(density_before_activation.to(positions) - 1)
        return density, features

    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[TensorType] = None
    ) -> Dict[FieldHeadNames, TensorType]:
        assert density_embedding is not None

        output_shape = ray_samples.frustums.shape
        directions = ray_samples.frustums.directions.reshape(-1, 3)

        if self.linear_decoder:
            color_features = [density_embedding]
        else:
            directions = shift_directions_for_tcnn(directions)
            d = self.direction_encoding(directions)
            color_features = [d, density_embedding.view(-1, self.geo_feat_dim)]

        if self.appearance_embedding_dim > 0:
            if self.training:
                assert ray_samples.camera_indices is not None
                camera_indices = ray_samples.camera_indices.squeeze()
                embedded_appearance = self.appearance_ambedding(camera_indices)
            elif self.use_average_appearance_embedding:
                embedded_appearance = torch.ones(
                    (*output_shape, self.appearance_embedding_dim),
                    device=directions.device,
                ) * self.appearance_ambedding.mean(dim=0)
            else:
                embedded_appearance = torch.zeros(
                    (*output_shape, self.appearance_embedding_dim),
                    device=directions.device,
                )

            if not self.linear_decoder:
                color_features.append(embedded_appearance)

        color_features = torch.cat(color_features, dim=-1)
        if self.linear_decoder:
            basis_input = directions
            if self.appearance_ambedding_dim > 0:
                basis_input = torch.cat([directions, embedded_appearance], dim=-1)
            basis_values = self.color_basis(basis_input) # [batch, color_feature_len * 3]
            basis_values = basis_values.view(basis_input.shape[0], 3, -1)  # [batch, color_feature_len, 3]
            rgb = torch.sum(color_features[:, None, :] * basis_values, dim=-1)  # [batch, 3]
            rgb = torch.sigmoid(rgb).view(*output_shape, -1).to(directions)
        else:
            rgb = self.color_net(color_features).view(*output_shape, -1)

        return {FieldHeadNames.RGB: rgb}


class KPlanesDensityField(Field):
    """A lightweight density field module.

    Args:
        aabb: Parameters of scene aabb bounds
        resolution: Grid resolution
        num_output_coords: dimension of grid feature vectors
        spatial_distortion: Spatial distortion to apply to the scene
        linear_decoder: Whether to use a linear decoder instead of an MLP
    """

    def __init__(
        self,
        aabb: TensorType,
        resolution: List[int],
        num_output_coords: int,
        spatial_distortion: Optional[SpatialDistortion] = None,
        linear_decoder: bool = False,
    ):
        super().__init__()

        self.register_buffer("aabb", aabb)

        self.spatial_distortion = spatial_distortion
        self.has_time_planes = len(resolution) > 3
        self.feature_dim = num_output_coords
        self.linear_decoder = linear_decoder

        self.grids = KPlanesEncoding(resolution, num_output_coords, init_a=0.1, init_b=0.15)

        self.sigma_net = tcnn.Network(
            n_input_dims=self.feature_dim,
            n_output_dims=1,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "None" if self.linear_decoder else "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            },
        )

        CONSOLE.log(f"Initialized KPlaneDensityField. with time-planes={self.has_time_planes} - resolution={resolution}")

    # pylint: disable=arguments-differ
    def density_fn(self, positions: TensorType["bs":..., 3], times: Optional[TensorType["bs", 1]] = None) -> TensorType["bs":..., 1]:
        """Returns only the density. Overrides base function to add times in samples

        Args:
            positions: the origin of the samples/frustums
            times: the time of rays
        """
        if times is not None and (len(positions.shape) == 3 and len(times.shape) == 2):
            # position is [ray, sample, 3]; times is [ray, 1]
            times = times[:, None]  # RaySamples can handle the shape
        # Need to figure out a better way to descibe positions with a ray.
        ray_samples = RaySamples(
            frustums=Frustums(
                origins=positions,
                directions=torch.ones_like(positions),
                starts=torch.zeros_like(positions[..., :1]),
                ends=torch.zeros_like(positions[..., :1]),
                pixel_area=torch.ones_like(positions[..., :1]),
            ),
            times=times,
        )
        density, _ = self.get_density(ray_samples)
        return density

    def get_density(self, ray_samples: RaySamples) -> Tuple[TensorType, None]:
        """Computes and returns the densities."""
        positions = ray_samples.frustums.get_positions()
        if self.spatial_distortion is not None:
            positions = self.spatial_distortion(positions)
            positions = positions / 2  # from [-2, 2] to [-1, 1]
        else:
            # From [0, 1] to [-1, 1]
            positions = SceneBox.get_normalized_positions(positions, self.aabb) * 2.0 - 1.0

        if self.has_time_planes:
            assert ray_samples.times is not None, "Initialized model with time-planes, but no time data is given"
            # Normalize timestamps from [0, 1] to [-1, 1]
            timestamps = ray_samples.times * 2.0 - 1.0
            positions = torch.cat((positions, timestamps), dim=-1)  # [n_rays, n_samples, 4]

        positions_flat = positions.view(-1, positions.shape[-1])
        features = interpolate_ms_features(
            positions_flat, grid_encodings=[self.grids], concat_features=False
        )
        if len(features) < 1:
            features = torch.zeros((0, 1), device=features.device, requires_grad=True)
        density_before_activation = self.sigma_net(features).view(*ray_samples.frustums.shape, -1)
        density = trunc_exp(density_before_activation.to(positions) - 1)
        return density, None

    def get_outputs(self, ray_samples: RaySamples, density_embedding: Optional[TensorType] = None) -> dict:
        return {}
