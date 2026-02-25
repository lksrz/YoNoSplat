from .decoder import Decoder
from .decoder_splatting_cuda import DecoderSplattingCUDA, DecoderSplattingCUDACfg
from .decoder_splatting_gsplat import DecoderSplattingGSPlat, DecoderSplattingSplatCfg

DECODERS = {
    "splatting_cuda": DecoderSplattingCUDA,
    "splatting_gsplat": DecoderSplattingGSPlat,
}

DecoderCfg = DecoderSplattingCUDACfg | DecoderSplattingSplatCfg


def get_decoder(decoder_cfg: DecoderCfg) -> Decoder:
    return DECODERS[decoder_cfg.name](decoder_cfg)
