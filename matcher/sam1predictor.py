from utils.registry import SAMPREDICTOR_REGISTRY
import logging
from utils.logger import get_logger
logger:logging = get_logger()
try:
    from segment_anything import sam_model_registry, SamPredictor
    from segment_anything import SamAutomaticMaskGenerator
except ImportError:
    logger.WARNING("sam1 not installed")
import numpy as np
import torch

@SAMPREDICTOR_REGISTRY.register
class SAM1Predictor:
    """
    SAM1 image mask predictor
    """
    def __init__(self, sam_size: str, weight_url: str):
        """
        sam_size: SAM model size
        sam_weights: SAM model weights path
        """
        self.sam_size = sam_size
        self.sam = sam_model_registry[sam_size](checkpoint=weight_url)
        #check if torch cuda is available
        if torch.cuda.is_available():
            self.sam.to(device="cuda")
        self.predictor = SamPredictor(self.sam)

    def __repr__(self):
        return f"SAM1-{self.sam_size}"
    
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