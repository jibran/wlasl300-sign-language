"""Models package for WLASL300 Sign Language Recognition.

Exposes the three core model classes::

    from models import SignModel, I3DBackbone, ProjectionHead

    cfg  = Config.from_yaml("config/config.yaml")
    model = SignModel.from_config(cfg)
    model.apply_phase(1)  # freeze backbone for warm-up
"""

from models.i3d_backbone import I3DBackbone
from models.projection_head import ProjectionHead
from models.sign_model import SignModel

__all__ = [
    "SignModel",
    "I3DBackbone",
    "ProjectionHead",
]
