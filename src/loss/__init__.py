from .loss import Loss
from .loss_depth import LossDepth, LossDepthCfgWrapper
from .loss_intrinsic import LossIntrinsic, LossIntrinsicCfgWrapper
from .loss_lpips import LossLpips, LossLpipsCfgWrapper
from .loss_mse import LossMse, LossMseCfgWrapper
from .loss_opacity import LossOpacity, LossOpacityCfgWrapper
from .loss_perceptual import LossPerceptual, LossPerceptualCfgWrapper
from .loss_pose_cfg import LossPose, LossPoseCfgWrapper
from .loss_silhouette import LossSilhouette, LossSilhouetteCfgWrapper

LOSSES = {
    LossDepthCfgWrapper: LossDepth,
    LossIntrinsicCfgWrapper: LossIntrinsic,
    LossLpipsCfgWrapper: LossLpips,
    LossMseCfgWrapper: LossMse,
    LossOpacityCfgWrapper: LossOpacity,
    LossPerceptualCfgWrapper: LossPerceptual,
    LossPoseCfgWrapper: LossPose,
    LossSilhouetteCfgWrapper: LossSilhouette,
}

LossCfgWrapper = (
    LossDepthCfgWrapper
    | LossIntrinsicCfgWrapper
    | LossLpipsCfgWrapper
    | LossMseCfgWrapper
    | LossOpacityCfgWrapper
    | LossPerceptualCfgWrapper
    | LossPoseCfgWrapper
    | LossSilhouetteCfgWrapper
)


def get_losses(cfgs: list[LossCfgWrapper]) -> list[Loss]:
    return [LOSSES[type(cfg)](cfg) for cfg in cfgs]
