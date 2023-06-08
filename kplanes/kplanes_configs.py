from __future__ import annotations

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig
from nerfstudio.data.dataparsers.dnerf_dataparser import DNeRFDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import CosineDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.plugins.types import MethodSpecification

from .kplanes import KPlanesModelConfig


kplanes_method = MethodSpecification(
    config=TrainerConfig(
        method_name="kplanes",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        steps_per_eval_all_images=30000,
        max_num_iterations=30001,
        mixed_precision=True,
        pipeline=VanillaPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                dataparser=BlenderDataParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
            ),
            model=KPlanesModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                grid_base_resolution=[128, 128, 128],
                grid_feature_dim=32,
                multiscale_res=[1, 2, 4],
                proposal_net_args_list=[
                    {"num_output_coords": 8, "resolution": [128, 128, 128]},
                    {"num_output_coords": 8, "resolution": [256, 256, 256]}
                ],
                loss_coefficients={
                    "interlevel": 1.0,
                    "distortion": 0.01,
                    "plane_tv": 0.01,
                    "plane_tv_proposal_net": 0.0001,
                },
                background_color="white",
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-12),
                "scheduler": CosineDecaySchedulerConfig(warm_up_end=512, max_steps=30000),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-12),
                "scheduler": CosineDecaySchedulerConfig(warm_up_end=512, max_steps=30000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="K-Planes NeRF model for static scenes"
)

kplanes_dynamic_method = MethodSpecification(
    config=TrainerConfig(
        method_name="kplanes-dynamic",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        steps_per_eval_all_images=30000,
        max_num_iterations=30001,
        mixed_precision=True,
        pipeline=VanillaPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                dataparser=DNeRFDataParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                camera_res_scale_factor=0.5,  # DNeRF train on 400x400
            ),
            model=KPlanesModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                grid_base_resolution=[128, 128, 128, 25],  # time-resolution should be half the time-steps
                grid_feature_dim=32,
                multiscale_res=[1, 2, 4],
                proposal_net_args_list=[
                    # time-resolution should be half the time-steps
                    {"num_output_coords": 8, "resolution": [128, 128, 128, 25]},
                    {"num_output_coords": 8, "resolution": [256, 256, 256, 25]},
                ],
                loss_coefficients={
                    "interlevel": 1.0,
                    "distortion": 0.01,
                    "plane_tv": 0.1,
                    "plane_tv_proposal_net": 0.0001,
                    "l1_time_planes": 0.001,
                    "l1_time_planes_proposal_net": 0.0001,
                    "time_smoothness": 0.1,
                    "time_smoothness_proposal_net": 0.001,
                },
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-12),
                "scheduler": CosineDecaySchedulerConfig(warm_up_end=512, max_steps=30000),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-12),
                "scheduler": CosineDecaySchedulerConfig(warm_up_end=512, max_steps=30000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="K-Planes NeRF model for dynamic scenes"
)
