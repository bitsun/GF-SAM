import argparse
import grpc
from concurrent import futures
import service_pb2 as pb2
import service_pb2_grpc as pb2_grpc
import logging
from logging.handlers import TimedRotatingFileHandler
import numpy as np
import os
import PIL.Image as Image
logger:logging.Logger = logging.getLogger(__name__)
from matcher.FewShotsSegmenter import FewShotsSegmenter
class FewShotsSegmenterService(pb2_grpc.FewShotSegmenterService):
    def __init__(self,nshot,ref_dir,dinov2_size,dinov2_weights,sam_size,sam_weights,mask_quality_th):
        self.nshot = nshot
        self.ref_dir = ref_dir
        self.dinov2_size = dinov2_size
        self.dinov2_weights = dinov2_weights
        self.sam_size = sam_size
        self.sam_weights = sam_weights
        self.segmenter:FewShotsSegmenter = FewShotsSegmenter(dinov2_size=self.dinov2_size, 
                                  dinov2_weights=self.dinov2_weights, 
                                  sam_size=self.sam_size, sam_weights=self.sam_weights,
                                 nshot=self.nshot,mask_th=mask_quality_th)
        self._init()
    def _init(self):
        #read sub dirs in ref-dir
        ref_dirs = [os.path.join(self.ref_dir, d) for d in os.listdir(self.ref_dir)]
        #sort ref_dirs
        ref_dirs.sort()
        
        for ref_dir in ref_dirs:
            #read jpg images in ref-dir
            ref_imgs = [os.path.join(ref_dir, f) for f in os.listdir(ref_dir) if f.endswith(".jpg")]
            #if there are more than args.nshot images, take only args.nshot images
            if len(ref_imgs) < self.nshot:
                logger.warning(f"Not enough images in reference directory {ref_dir}")
            ref_imgs = ref_imgs[:min(len(ref_imgs),self.nshot)]
            images = []
            masks = []
            for i, ref_img_name in enumerate(ref_imgs):
                #find its corresponding mask image, which ends png
                #replace jpg with png
                img_name = os.path.basename(ref_img_name)
                ref_img = Image.open(ref_img_name)
                ref_mask = Image.open(os.path.join(ref_dir, img_name.replace(".jpg", ".png")))
                ref_img = ref_img.convert('RGB')
                images.append(ref_img)
                masks.append(ref_mask)
            self.segmenter.add_reference(images, masks, os.path.basename(ref_dir))
    def get_classes(self,request,context): 
        """
        get the classes of the segmenter
        """
        response = pb2.StringListResponse()
        for class_name in self.segmenter.references.keys():
            response.value.append(class_name)
        return response
    def get_nshot(self,request,context):
        """
        how many shots are used for segmentation
        """
        return pb2.IntResponse(value=self.segmenter.nshot,error_msg="")
    
    def segment(self,request,context):
        #get the rgb image
        image_bgr = np.frombuffer(request.data, np.uint8)
        image_bgr = np.reshape(image_bgr,(request.height,request.width,request.num_channels))
        mask = self.segmenter.segment(image_bgr)
        mask = mask.squeeze().detach().cpu().numpy().astype(np.uint8)
        mask_image = pb2.Image(height=mask.shape[0],width=mask.shape[1],num_channels=1,data=mask.tobytes())
        return pb2.SegmentMapResponse(org_img_width= image_bgr.shape[1],
                                      org_img_height= image_bgr.shape[0],
                                       map = mask_image,error_message="")
import json
def serve(max_workers,port,config_path):  
    with open(config_path) as f:
        config = json.load(f)
    if "nshot" in config:
        nshot = int(config["nshot"])
    else:
        nshot = 1
    segmenter_service =FewShotsSegmenterService(nshot=nshot,
                                                dinov2_weights=config["dinov2_weights"],
                                                dinov2_size="vit_large",
                                                sam_size="vit_h",
                                                sam_weights=config["sam_weights"],
                                                ref_dir=config["ref_dir"])
    if "do_postprocess" in config:
        do_postprocess = bool(config["do_postprocess"])
        segmenter_service.segmenter.do_post_process = do_postprocess
    
    if "mask_th" in config:
        mask_quality_th = float(config["mask_th"])
        segmenter_service.segmenter.mask_quality_th = mask_quality_th
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    pb2_grpc.add_FewShotSegmenterServiceServicer_to_server(segmenter_service,server)
    server.add_insecure_port('[::]:{}'.format(port))
    server.start()
    logging.info("few shots segmenter server started, listening port {}".format(port))
    server.wait_for_termination()  

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Few Shots segmentation server")
    parser.add_argument('--max-workers',type=int,default=2,help='max number of workers')
    parser.add_argument('--port',type=int,default=50051,help='port number')
    parser.add_argument('--config-path',type=str,required=True,help='few shots segmenter config path')
    args = parser.parse_args()
    formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s')
    handler = TimedRotatingFileHandler('few_shots_segmenter_{}.log'.format(args.port), 
                                   when='midnight',
                                   backupCount=10)
    handler.setFormatter(formatter)
    logger = logging.getLogger(__name__)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    serve(args.max_workers,args.port,args.config_path)