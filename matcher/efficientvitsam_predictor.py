from utils.registry import SAMPREDICTOR_REGISTRY
import logging
from utils.logger import get_logger
logger:logging = get_logger()
try:
    from efficientvit.sam_model_zoo import create_efficientvit_sam_model
    from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor
except ImportError:
    logger.WARNING("efficientvit sam not installed")
import numpy as np
import torch

@SAMPREDICTOR_REGISTRY.register
class EfficientViTSAMPredictor:
    """
    SAM1 image mask predictor
    """
    def __init__(self, name: str, weight_url: str):
        """
        name: model name
        m: SAM model weights path
        """
        self.name = name
        self.sam = create_efficientvit_sam_model(name=name,pretrained=True,weight_url=weight_url)
        if torch.cuda.is_available():
            self.sam.to(device="cuda")
        self.predictor = EfficientViTSamPredictor(self.sam)
    
    def __repr__(self):
        return f"EfficientVitSAM-{self.name}"
    
    def encode_image(self,image:np.ndarray,image_format="BGR")->torch.Tensor:
        """
        encode image and return image feature
        """

        self.predictor.set_image(image,image_format=image_format)
        return self.predictor.features

    def predict_mask(self,features:torch.Tensor,coord_xy:torch.Tensor,coord_labels:torch.Tensor)->torch.Tensor:
        """
        predict mask for the image
        return mask as torch.Tensor of boolean type
        """
        tar_masks, scores, logits, _ = self.predictor.predict_torch(
                point_coords=coord_xy[:, None, :],
                point_labels=coord_labels[:, None],
                # mask_input=mask_inputs,
                features=features,
                multimask_output=False, 
            )
        tar_masks = tar_masks > self.predictor.model.mask_threshold
        return tar_masks