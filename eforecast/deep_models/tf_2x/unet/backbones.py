
from __future__ import absolute_import

import warnings

depreciated_warning = ("\n----------------------------------------\n`backbones`is depreciated, "
                       "use `eforecast.deep_models.tf_2x.unet.base` "
                       "instead.\ne.g.\nfrom eforecast.deep_models.tf_2x.unet import base\nbase.unet_2d_base(...);"
                       "\n----------------------------------------")
warnings.warn(depreciated_warning)

from eforecast.deep_models.tf_2x.unet._model_unet_2d import unet_2d_base as unet_2d_backbone
from eforecast.deep_models.tf_2x.unet._model_unet_plus_2d import unet_plus_2d_base as unet_plus_2d_backbone
from eforecast.deep_models.tf_2x.unet._model_r2_unet_2d import r2_unet_2d_base as r2_unet_2d_backbone
from eforecast.deep_models.tf_2x.unet._model_att_unet_2d import att_unet_2d_base as att_unet_2d_backbone
from eforecast.deep_models.tf_2x.unet._model_resunet_a_2d import resunet_a_2d_base as resunet_a_2d_backbone
from eforecast.deep_models.tf_2x.unet._model_u2net_2d import u2net_2d_base as u2net_2d_backbone
from eforecast.deep_models.tf_2x.unet._model_unet_3plus_2d import unet_3plus_2d_base as unet_3plus_2d_backbone
