"""
Model architectures for protein functionality prediction with MoE.
"""

from .moe import MoELayer, MoEModel
from .experts import Expert
from .encoders import SequenceEncoder
from .ranking_loss import RankingLoss, CombinedLoss, compute_combined_loss
from .ema import EMA, EMAWrapper
from .multimodal_fusion import (
    MSAEncoder,
    CrossAttentionFusion,
    MultiModalMoEModel,
    create_multimodal_model
)

__all__ = [
    'MoELayer', 'MoEModel', 'Expert', 'SequenceEncoder',
    'RankingLoss', 'CombinedLoss', 'compute_combined_loss',
    'EMA', 'EMAWrapper',
    'MSAEncoder', 'CrossAttentionFusion', 'MultiModalMoEModel', 'create_multimodal_model'
]