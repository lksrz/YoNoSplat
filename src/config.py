from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Type, TypeVar, Any

from dacite import Config, from_dict
from omegaconf import DictConfig, OmegaConf

from .dataset import DatasetCfgWrapper
from .dataset.data_module import DataLoaderCfg
from .loss import LossCfgWrapper
from .model.decoder import DecoderCfg
from .model.encoder import EncoderCfg
from .model.model_wrapper import OptimizerCfg, TestCfg, TrainCfg


@dataclass
class CheckpointingCfg:
    load: Optional[str]  # Not a path, since it could be something like wandb://...
    every_n_train_steps: int
    save_top_k: int
    save_weights_only: bool


@dataclass
class ModelCfg:
    decoder: DecoderCfg
    encoder: EncoderCfg


@dataclass
class TrainerCfg:
    max_steps: int
    val_check_interval: int | float | None
    gradient_clip_val: int | float | None
    num_nodes: int = 1


@dataclass
class RootCfg:
    wandb: dict
    mode: Literal["train", "test"]
    dataset: list[Any]
    data_loader: DataLoaderCfg
    model: ModelCfg
    optimizer: OptimizerCfg
    checkpointing: CheckpointingCfg
    trainer: TrainerCfg
    loss: list[LossCfgWrapper]
    test: TestCfg
    train: TrainCfg
    seed: int


TYPE_HOOKS = {
    Path: Path,
}


T = TypeVar("T")


def load_typed_config(
    cfg: DictConfig,
    data_class: Type[T],
    extra_type_hooks: dict = {},
) -> T:
    return from_dict(
        data_class,
        OmegaConf.to_container(cfg),
        config=Config(type_hooks={**TYPE_HOOKS, **extra_type_hooks}),
    )


def separate_loss_cfg_wrappers(joined: dict) -> list[LossCfgWrapper]:
    # The dummy allows the union to be converted.
    @dataclass
    class Dummy:
        dummy: LossCfgWrapper

    return [
        load_typed_config(DictConfig({"dummy": {k: v}}), Dummy).dummy
        for k, v in joined.items()
    ]


def separate_dataset_cfg_wrappers(joined: dict) -> list[DatasetCfgWrapper]:
    # The dummy allows the union to be converted.
    @dataclass
    class Dummy:
        dummy: DatasetCfgWrapper

    res = []
    for k, v in joined.items():
        try:
            res.append(load_typed_config(DictConfig({"dummy": {k: v}}), Dummy).dummy)
        except Exception as e:
            if k == "shoes":
                from src.dataset.dataset_shoes import DatasetShoesCfg, DatasetShoesCfgWrapper
                from src.dataset.view_sampler.view_sampler_bounded import ViewSamplerBoundedCfg
                
                v_copy = OmegaConf.to_container(DictConfig(v))
                # Handle view_sampler separately
                vs_dict = v_copy.pop("view_sampler")
                
                # Ensure all bounded sampler fields exist
                for field, default in [
                    ("min_distance_between_context_views", 2),
                    ("max_distance_between_context_views", 6),
                    ("min_distance_to_context_views", 0),
                    ("warm_up_steps", 0),
                    ("initial_min_distance_between_context_views", 2),
                    ("initial_max_distance_between_context_views", 6)
                ]:
                    if field not in vs_dict:
                        vs_dict[field] = default
                
                vs_cfg = from_dict(ViewSamplerBoundedCfg, vs_dict, config=Config(type_hooks=TYPE_HOOKS))
                
                # Handle root dataset
                # Manually ensure these fields exist in the dict for dacite
                for field, default in [
                    ("original_image_shape", [512, 512]),
                    ("input_image_shape", [512, 512]),
                    ("background_color", [0.0, 0.0, 0.0]),
                    ("cameras_are_circular", False),
                    ("overfit_to_scene", None),
                    ("make_baseline_1", True),
                    ("relative_pose", True),
                    ("augment", True),
                    ("skip_bad_shape", True),
                    ("baseline_min", 1e-3),
                    ("baseline_max", 1e10),
                    ("max_fov", 100.0),
                    ("pose_norm_method", "max_pairwise_d")
                ]:
                    if field not in v_copy:
                        v_copy[field] = default
                
                v_copy["view_sampler"] = vs_cfg
                shoes_cfg = from_dict(DatasetShoesCfg, v_copy, config=Config(type_hooks=TYPE_HOOKS))
                res.append(DatasetShoesCfgWrapper(shoes_cfg))
            else:
                raise e
    return res


def load_typed_root_config(cfg: DictConfig) -> RootCfg:
    from typing import Any
    return load_typed_config(
        cfg,
        RootCfg,
        {list[Any]: separate_dataset_cfg_wrappers,
         list[LossCfgWrapper]: separate_loss_cfg_wrappers},
    )
