from utils.registry import SAMPREDICTOR_REGISTRY
import logging
from utils.logger import get_logger
logger:logging = get_logger()
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    logger.WARNING("sam2 not installed")
import numpy as np
import torch

@SAMPREDICTOR_REGISTRY.register
class SAM2Predictor:
    """
    SAM1 image mask predictor
    """
    def __init__(self, model_cfg: str, weight_url: str):
        """
        model_cfg:model config file path
        sam_weights: SAM model weights path
        """
        self.model_cfg = model_cfg
        self.sam = build_sam2(model_cfg, weight_url)
        #check if torch cuda is available
        if torch.cuda.is_available():
            self.device = "cuda"
            self.sam.to(device="cuda")
        else:
            self.device = "cpu"
        self.predictor = SAM2ImagePredictor(self.sam)

    def __repr__(self):
        return f"SAM2-{self.model_cfg}"
    
    def encode_image(self,image:np.ndarray,image_format="BGR")->torch.Tensor:
        """
        encode image and return image feature
        """
        with torch.no_grad():
            self.predictor.set_image(image)
        feats = self.predictor._features["image_embed"] # 1, c, h, w
        return feats
    
    def predict_mask(self,features:torch.Tensor,coord_xy:np.ndarray,coord_labels:np.ndarray)->torch.Tensor:
        """
        predict mask for the image
        return mask as torch.Tensor of boolean type
        """
        masks,_,_ = self.predictor.predict(point_coords=coord_xy[:, None, :],
                                   point_labels=coord_labels[:, None],
                                   multimask_output=False)
        masks = masks>0
        masks = torch.from_numpy(masks).to(self.device)
        return masks