import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import argparse
import PIL.Image as Image
import os
from matcher.FewShotsSegmenter import FewShotsSegmenter
from matcher.GFSAM import build_model
import numpy as np
import cv2
mask_pallet =[0, 0, 0, 
            0, 0, 255, 
            0, 255, 0,
            255, 0, 0,
            0, 255, 255, 
            255, 0, 255,
            255, 255, 0, 
            0, 0, 128,
            0, 128, 0, 
            128, 0, 0, 
            0, 128, 128, 
            128, 0, 128, 
            128, 128, 0, 
            64, 128, 0, 
            64, 0, 192,
            128, 64, 128, 
            0, 64, 128,
            64, 128, 128, 
            255, 255, 64, 128, 64, 64, 64, 64, 192, 192, 192, 192, 255, 128, 128, 192, 128, 128, 192, 64, 64, 255, 192, 192, 64, 64, 255, 128, 64, 0, 255, 64, 255, 64, 255, 64, 128, 192, 255, 64, 255, 255, 128, 192, 64, 255, 0, 64, 128, 64, 255, 255, 128, 255, 64, 192, 192, 128, 255, 192, 128, 192, 0, 64, 64, 64, 0, 128, 255, 192, 192, 128, 64, 64, 128, 0, 64, 64, 255, 64, 0, 255, 64, 128, 192, 128, 192, 192, 255, 64, 0, 64, 0, 64, 192, 128, 192, 192, 64, 192, 255, 192, 128, 192, 128, 192, 0, 64, 255, 255, 192, 255, 192, 255, 192, 255, 128, 128, 0, 64, 64, 192, 64, 192, 0, 0, 0, 192, 255, 255, 255, 128, 192, 64, 255, 64, 0, 255, 0, 255, 64, 128, 128, 255, 64, 255, 0, 0, 64, 255, 192, 255, 255, 128, 0, 255, 128, 128, 192, 64, 0, 0, 128, 255, 64, 0, 64, 192, 128, 128, 64, 64, 64, 0, 0, 0, 64, 255, 255, 255, 255, 192, 64, 64, 128, 64, 128, 255, 255, 192, 192, 255, 255, 192, 0, 255, 192, 128, 0, 0, 192, 64, 255, 192, 0, 128, 64, 192, 128, 255, 128, 0, 192, 128, 192, 192, 255, 128, 64, 64, 0, 128, 0, 255, 192, 255, 0, 192, 255, 128, 0, 128, 255, 128, 128, 64, 192, 0, 128, 192, 192, 0, 192, 255, 0, 128, 64, 128, 192, 0, 255, 128, 64, 128, 255, 192, 255, 0, 0, 192, 64, 0, 192, 128, 255, 128, 192, 192, 128, 64, 192, 128, 0, 128, 255, 0, 0, 192, 0, 255, 64, 64, 192, 0, 128, 64, 255, 128, 192, 192, 0, 192, 64, 0, 128, 128, 128, 64, 0, 64, 64, 192, 0, 255, 64, 192, 64, 192, 255, 0, 192, 192, 192, 64, 192, 192, 64, 128, 192, 0, 255
		]
def blend_mask_on_image(image, mask):
    mask = mask.squeeze().cpu().numpy()
    max_class_id = int(np.max(mask))
    mask_img = np.zeros((mask.shape[-2],mask.shape[-1],3)).astype(np.uint8)
    for i in range(max_class_id):
        class_id = i+1
        if class_id == 0:
            continue
        mask_img[mask == class_id] = mask_pallet[3*class_id:3*class_id+3]
    #blend image with cv2
    result = cv2.addWeighted(image, 0.5, mask_img, 0.5, 0)
    return result
def main():
    parser = argparse.ArgumentParser(description='ivitec few shot demo')
    parser.add_argument('--nshot', type=int, required=False, default=1)
    parser.add_argument('--ref-dir', type=str, required=True)
    parser.add_argument('--img-size', type=int, default=1024)
    parser.add_argument('--dinov2-size', type=str, default="vit_large")
    parser.add_argument('--dinov2-weights', type=str, default="models/dinov2_vitl14_pretrain.pth")
    parser.add_argument('--sam-size', type=str, default="vit_h")
    parser.add_argument('--sam-weights', type=str, default="models/sam_vit_h_4b8939.pth")

    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    segmenter = FewShotsSegmenter(dinov2_size=args.dinov2_size, 
                                  dinov2_weights=args.dinov2_weights, 
                                  sam_size=args.sam_size, sam_weights=args.sam_weights,
                                    input_size=args.img_size, nshot=args.nshot,mask_th=0.5)
    #read sub dirs in ref-dir
    ref_dirs = [os.path.join(args.ref_dir, d) for d in os.listdir(args.ref_dir)]
    for ref_dir in ref_dirs:
        #read jpg images in ref-dir
        ref_imgs = [os.path.join(ref_dir, f) for f in os.listdir(ref_dir) if f.endswith(".jpg")]
        #if there are more than args.nshot images, take only args.nshot images
        if len(ref_imgs) < args.nshot:
            raise ValueError(f"Not enough images in reference directory {ref_dir}")
        ref_imgs = ref_imgs[:args.nshot]
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
        segmenter.add_reference(images, masks, os.path.basename(ref_dir))
    
    #now make query
    #query_img = Image.open("C:\\Users\\bliu\\Documents\\helmet.jpg")
    #image_tensor1 = segmenter.transform(query_img)
    query_img = cv2.imread("C:\\Users\\bliu\\Documents\\helmet.jpg")
    #query_img = Image.open ("E:\\Data\\TVCheck\\test\\helmet\\000000805.jpg")
    #tar_img_tensor = torch.from_numpy(cv2.dnn.blobFromImage(query_img,1/255.0,segmenter.input_size,swapRB=True))
    mask = segmenter.segment(query_img)
    #draw mask on image
    #swap r and b channel
    #query_img = np.array(query_img)[:,:,::-1]
    result = blend_mask_on_image(query_img, mask)
    cv2.imshow("result",result)    
    cv2.waitKey(0)
main()