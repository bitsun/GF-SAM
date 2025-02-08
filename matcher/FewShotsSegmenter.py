from utils.logger import get_logger
logger = get_logger()
from os import path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import numpy as np
import math
#import matplotlib.pyplot as plt

from segment_anything import sam_model_registry, SamPredictor
from segment_anything import SamAutomaticMaskGenerator
from dinov2.models import vision_transformer as vits
import dinov2.utils.utils as dinov2_utils
from dinov2.data.transforms import MaybeToTensor, make_normalize_transform
from segment_anything.utils.amg import (
    batch_iterator, 
)
from efficientvit.sam_model_zoo import create_efficientvit_sam_model
from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor
from scipy.sparse import csgraph
import PIL.Image as Image
#from .GFSAM import GFSAM
import cv2
class Reference:
    def __init__(self, img_tensors:torch.Tensor, masks:torch.Tensor, label:str,feats:torch.Tensor):
        """
        a reference for some label, if it is a nshot reference, img_tensors and masks are lists of tensors or
        a tensor with shape (nshot, c, h, w)
        """
        if img_tensors is not None:
            self.img_tensors = img_tensors.to(torch.device("cpu"))
        else:
            self.img_tensors = None
        if masks is not None:
            self.masks = masks.to(torch.device("cpu"))
        else:
            self.masks = None
        self.label = label
        if feats is not None:
            self.feats = feats.to(torch.device("cpu"))
        else:
            self.feats = None

class FewShotsSegmenter:
    def __init__(
            self,
            dinov2_size="vit_large",
            dinov2_weights="models/dinov2_vitl14_pretrain.pth",
            sam_size="vit_h",
            sam_weights="models/sam_vit_h_4b8939.pth",
            input_size=1024,
            nshot=1,
            mask_th=0.6,
            do_post_process = False
    ):
        # DINOv2, Image Encoder
        dinov2_kwargs = dict(
            img_size=518,
            patch_size=14,
            init_values=1e-5,
            ffn_layer='mlp',
            block_chunks=0,
            qkv_bias=True,
            proj_bias=True,
            ffn_bias=True,
        )
        dinov2 = vits.__dict__[dinov2_size](**dinov2_kwargs)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        
        dinov2_utils.load_pretrained_weights(dinov2, dinov2_weights, "teacher")
        dinov2.to(device=self.device)
        dinov2.eval()
        
        # SAM
        #sam = sam_model_registry[sam_size](checkpoint=sam_weights)
        sam = create_efficientvit_sam_model(name="efficientvit-sam-xl0",pretrained=True,weight_url="/home/bliu/Network/Robin2/User/bliu/Dataset/Model/SAM/efficientvit_sam_xl0.pt")
        sam.to(device=self.device)
        #predictor = SamPredictor(sam)
        predictor = EfficientViTSamPredictor(sam)
        self.encoder = dinov2
        self.predictor = predictor

        if not isinstance(input_size, tuple):
            input_size = (input_size, input_size)
        self.input_size = input_size

        img_size = 518
        feat_size = img_size // self.encoder.patch_size

        self.encoder_img_size = img_size
        self.encoder_feat_size = feat_size

        # self.gf = GFSAM(
        # encoder=dinov2,
        # generator=predictor,
        # device=self.device
        # )
        # transforms for image encoder
        self.encoder_transform = transforms.Compose([
            MaybeToTensor(),
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BILINEAR),
            make_normalize_transform(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        self.nshot = nshot
        self.references:dict[str,Reference] = {}
        self.transform = transforms.Compose([
            transforms.Resize(size=input_size),
            transforms.ToTensor()
        ])
        self.mask_th = mask_th
        self.do_post_process = do_post_process
        self.half = False
        if self.half:
            self.encoder = self.encoder.half()
        
    @torch.no_grad()
    def extract_img_feats(self,image_tensors:torch.Tensor):
        """
        extract dinov2 features from image tensors
        image tensors are preprocessed(normalized) but only resized
        """
        image_tensors = image_tensors.to(self.device)
        with torch.no_grad():
            image_tensors = self.encoder_transform(image_tensors)
            if self.half:
                image_tensors = image_tensors.half()
            feats = self.encoder.forward_features(image_tensors)["x_prenorm"][:, 1:]
            feats = F.normalize(feats, dim=1, p=2) # normalize for cosine similarity
        if self.half:
            feats = feats.float()
        return feats
    
    def add_reference(self, imgs:list[Image.Image], masks:list[Image.Image],label:str):
        """
        add a nshot reference images and masks
        """
        if len(imgs)==0:
            self.references[label] = Reference(None, None, label,None)
            return
        if label in self.references:
            raise ValueError("ref label {} already exists".format(label))
        assert len(imgs) == len(masks)
        ref_img_tensors = []
        ref_mask_tensors = []
        for i in range(self.nshot):
            ref_img = imgs[i]
            ref_mask = masks[i]
            ref_img_tensor = self.transform(ref_img)
            ref_mask_tensor = torch.tensor(np.array(ref_mask)/255.0)
            ref_img_tensors.append(ref_img_tensor.unsqueeze(0))
            ref_mask_tensors.append(ref_mask_tensor.unsqueeze(0))
        
        ref_img_tensors = torch.cat(ref_img_tensors,dim=0)
        ref_mask_tensors = torch.cat(ref_mask_tensors,dim=0)
        ref_mask_tensors = F.interpolate(ref_mask_tensors.unsqueeze(0).float(), ref_img_tensors.size()[-2:], mode='nearest')
        def reference_masks_verification(masks):
            if masks.sum() == 0:
                _, _, sh, sw = masks.shape
                masks[..., (sh // 2 - 7):(sh // 2 + 7), (sw // 2 - 7):(sw // 2 + 7)] = 1
            return masks

        # process reference masks
        ref_mask_tensors = reference_masks_verification(ref_mask_tensors)
        ref_mask_tensors = ref_mask_tensors.permute(1, 0, 2, 3)  # ns, 1, h, w 
        feats = self.extract_img_feats(ref_img_tensors)
        self.references[label] = Reference(ref_img_tensors, ref_mask_tensors, label,feats)
    
    @torch.no_grad()
    def extract_sam_feats(self, img:np.ndarray|torch.Tensor)->torch.Tensor:
        """
        extract sam features from image
        img is torch image tensor with values between 0 and 1
        """
        #img_np = img.mul(255).byte()
        #img_np = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
        #assert isinstance(img, np.ndarray) and img.ndim == 3 and img.shape[2] == 3
        if isinstance(img,np.ndarray):
            preprocess = True
            rz_img = cv2.resize(img, self.input_size)
            #rz_img = rz_img/255.0
            self.predictor.set_image(rz_img,image_format="BGR")
        else:
            preprocess = False
            input = img
            self.predictor.set_image(input,preprocess=preprocess)
        return self.predictor.features # 1,c,h,w
    
    def compute_sim_ref2query(self,tar_feats,ref_feats, similarities,query_mask,reference_masks,reference_labels)->float:
        """
        compute the similarity from reference mask to the query mask,
        then take the mean of the similarity value to verify the quality of query mask again
        via ref2query similarity
        """
        reference_masks = reference_masks.to(self.device)
        #ref_feats = ref_feats.to(self.device)
        #tar_feats1 = torch.masked_select(tar_feats,query_mask_small>0).reshape((-1,tar_feats.shape[-1]))
        #tar_feats1_norm = tar_feats1/torch.norm(tar_feats1,p=2,dim=1,keepdim=True)
        
        sp_sz2 = similarities[0].shape[-1]
        sp_sz = int(math.sqrt(sp_sz2))
        query_mask_small = F.interpolate(query_mask.unsqueeze(0), size=(sp_sz, sp_sz), mode='bilinear', align_corners=True)
        query_mask_small = query_mask_small.flatten(2).transpose(-1, -2) # [bs, h*w, 1]
        
        #tmp_mask = F.interpolate(tmp_mask, size=(sp_sz, sp_sz), mode='bilinear', align_corners=True)
        mask_qualities = np.zeros(len(similarities))
        for st, _ in enumerate(similarities):
            #for each reference point
            ref_mask = reference_masks[st].unsqueeze(0)
            ref_mask = F.interpolate(ref_mask, size=(sp_sz, sp_sz), mode='bilinear', align_corners=True)
            ref_mask = ref_mask.flatten(2).transpose(-1, -2)
            #ref_feats1 = torch.masked_select(ref_feats[st],ref_mask>0).reshape((-1,ref_feats.shape[-1]))
            #ref_feats1_norm = ref_feats1/torch.norm(ref_feats1,p=2,dim=1,keepdim=True)
            #max_sims = torch.zeros(ref_feats1_norm.shape[0])
            #mean_sims = torch.zeros_like(max_sims)
            #for n1 in range(ref_feats1_norm.shape[0]):
            #    tmp = (tar_feats1_norm*ref_feats1_norm[n1]).sum(-1)
            #    max_sims[n1] = torch.max(tmp)
            #    mean_sims[n1] = torch.mean(tmp)
            similarity = similarities[st].permute(0,2,1)*query_mask_small # [bs, h*w, h*w]
            mask_qualities[st]= ((torch.sum(similarity.max(1)[0]*ref_mask.squeeze(2))/torch.sum(ref_mask)).item())
            #print(st,reference_labels,(torch.sum(similarity.mean(1)*ref_mask.squeeze(2))/torch.sum(ref_mask)).item())
        #compute the mean value of mask_qualities
        return np.mean(mask_qualities)
    def generate_prior(self, query_feat_high, supp_feat_high, s_mask):
        """
        generate prior similarity maps
        s_mask is the masks of supporting or reference images
        """
        bsize, sp_sz2, _= query_feat_high.size()[:]
        sp_sz = int(math.sqrt(sp_sz2))

        similarities = self.generate_pixelwise_comparison(query_feat_high, supp_feat_high)
        corr_query_mask_list = []
        cos_similarity_list = []
        cosine_eps = 1e-7
        for st, supp_feat in enumerate(supp_feat_high):
            tmp_mask = s_mask[st].unsqueeze(0)
            tmp_mask = F.interpolate(tmp_mask, size=(sp_sz, sp_sz), mode='bilinear', align_corners=True)
            tmp_mask = tmp_mask.flatten(2).transpose(-1, -2) # [bs, h*w, 1]

            similarity = similarities[st] * tmp_mask # [bs, h*w, h*w]
            cos_similarity = similarity.mean(1).view(bsize, 1, sp_sz, sp_sz) / (tmp_mask.sum() / sp_sz2 + cosine_eps)
            similarity = similarity.max(1)[0].view(bsize, sp_sz2)   
            # similarity = (similarity - similarity.min(1)[0].unsqueeze(1))/(similarity.max(1)[0].unsqueeze(1) - similarity.min(1)[0].unsqueeze(1) + cosine_eps)
            corr_query = similarity.view(bsize, 1, sp_sz, sp_sz)
            # corr_query = F.interpolate(corr_query, size=(fts_size[0], fts_size[1]), mode='bilinear', align_corners=True)
            corr_query_mask_list.append(corr_query)  
            cos_similarity_list.append(cos_similarity)
        corr_query_mask = torch.cat(corr_query_mask_list, 1)
        corr_query_mask = corr_query_mask.mean(1)
        cos_similarity = torch.cat(cos_similarity_list, 1).mean(1)
        return corr_query_mask, cos_similarity,similarities
    
    def segment(self, img:np.ndarray)->torch.Tensor:
        """
        segment a single image  based on the references
        """
        #assert isinstance(img, np.ndarray) and img.ndim == 3 and img.shape[2] == 3
        #img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #tar_img_tensor = img_rgb.transpose(2,0,1)/255.0
        #tar_img_tensor = self.transform(img).unsqueeze(0)
        tar_img_tensor = torch.from_numpy(cv2.dnn.blobFromImage(img,1/255.0,self.input_size,swapRB=True))
        tar_img_tensor = tar_img_tensor.to(self.device)
        #extract sam features from target image
        tar_feats = self.extract_sam_feats(tar_img_tensor)
        #extract dinov2 features from target image
        tar_feats_sem = self.extract_img_feats(tar_img_tensor)
        result_masks = []
        #for each reference label, segment the image
        for k,reference in enumerate(self.references.values()):
            class_id = k+1
            if reference.feats is None or reference.masks is None:
                continue
            ref_feats_sem = reference.feats.to(self.device)
            ref_masks = reference.masks.to(self.device)
            # positive and negative similarity maps
            neg_sim_map, neg_mean_sim_map,neg_sim_mat = self.generate_prior(tar_feats_sem, ref_feats_sem, 1-ref_masks)
            sim_map, mean_sim_map,sim_mat = self.generate_prior(tar_feats_sem, ref_feats_sem, ref_masks)

            # mid-value of similarity map
            mean_sim_map_half = (mean_sim_map.max() + mean_sim_map.min()) / 2

            # mix the similarity map and align the value to [0, 1]
            cross_sim_map = mean_sim_map * sim_map
            cross_sim_map = (cross_sim_map - cross_sim_map.min()) / (cross_sim_map.max() - cross_sim_map.min() + 1e-6)

            neg_mean_sim_map_std = (neg_mean_sim_map - neg_mean_sim_map.min()) / (neg_mean_sim_map.max() - neg_mean_sim_map.min() + 1e-6)
            neg_region = neg_mean_sim_map_std > cross_sim_map
            mean_sim_map_fil = cross_sim_map * ~neg_region
            coord_xy, coord_labels, sim_map_hot, coord_f = self.find_points(mean_sim_map_fil, tar_feats)
        
            tar_masks_list = self.generate_sam_masks(tar_feats, coord_xy, coord_labels)

            if len(tar_masks_list) == 0:
                tar_masks = torch.zeros((1, 1, 1024, 1024), device=self.device)
                pred_masks = tar_masks.squeeze(0)
                prob_masks = torch.zeros_like(pred_masks)
            else:
                tar_masks = torch.cat(tar_masks_list, dim=0)
                # sim_map_large = F.interpolate(sim_map.unsqueeze(0), size=(1024, 1024), mode='bilinear')
                # mask_quality = []
                # for idx, tar_mask in enumerate(tar_masks):
                #     mask_quality.append(sim_map_large[0,0][tar_mask[0]].mean().item())
                components_weak, labels_weak, components_strong, labels_strong = self.mask_cluster(tar_masks, coord_f, sim_map_hot)

                fgbg_com_labels, fgbg_labels, pseudo_masks, cls_scores = self.cluster_classification(tar_masks, labels_weak, components_weak, 
                                                                                     mean_sim_map * mean_sim_map, neg_mean_sim_map * mean_sim_map_half, coord_f)

                selected_points = self.point_consistency_dis(tar_feats_sem, tar_masks, labels_weak, components_weak, fgbg_labels, coord_f)
                mask_quality_list = []
                labels_weak_tensor = torch.as_tensor(labels_weak, device=self.device, dtype=torch.long)
                for component in range(components_weak):
                    com_args = torch.where(torch.logical_and(labels_weak_tensor == component, selected_points==1))[0]
                    if len(com_args) == 0:
                        mask_quality_list.append(0)
                        selected_points[com_args] = 3
                        continue
                    dense_union_mask = (tar_masks[com_args].sum(dim=0) > 0).float()
                    #dense_union_mask is a candidate mask in query image
                    #now we can reuse the similarity matrix to compute the mean similarity between this query
                    #mask and all the reference masks
                    union_mask = F.interpolate(dense_union_mask.unsqueeze(0), (self.encoder_feat_size, self.encoder_feat_size), mode='nearest').squeeze(0)>0
                    mask_quality = sim_map[0][union_mask[0]].mean().item()
                    mask_quality_list.append(mask_quality)
                    if(mask_quality<self.mask_th):
                        selected_points[com_args] = 3
                        continue
                    mask_quality1 = self.compute_sim_ref2query(tar_feats_sem,reference.feats, sim_mat, dense_union_mask,reference.masks,reference.label)
                    if(mask_quality1<self.mask_th):
                        selected_points[com_args] = 4
                        continue
                    result_masks.append((dense_union_mask>0,class_id, (mask_quality+mask_quality1)/2))
                   
                #pred_masks, prob_masks = self.triplet_selection_b(tar_masks, labels_weak, selected_points, mean_sim_map, coord_f, cls_scores)
        #compare all the masks to check if this masks to check if this mask should be added to the final result
        mask2del = []
        for k,(mask1,class_id1,mask1_quality) in enumerate(result_masks):
            for mask2,class_id2,mask2_quality in result_masks:
                if class_id1 == class_id2:
                    continue
                if mask1_quality>mask2_quality:
                    continue
                #check mask1 overlap with mask 2
                if(torch.sum(mask1 & mask2)>=torch.sum(mask1)*0.2):
                    mask2del.append(k)
                    break
        #delete the masks with index in mask2del,result_masks is a list
        for i in sorted(mask2del, reverse=True):
            del result_masks[i]
        if len(result_masks) == 0:
            return torch.zeros((img.shape[0], img.shape[1])).float(),torch.zeros((img.shape[0], img.shape[1])).float()
        final_mask = torch.zeros_like(result_masks[0][0]).float()
        prob_mask = torch.zeros_like(final_mask)
        for mask,class_id,mask_quality in result_masks:
            if self.do_post_process:
                mask = self.remove_small_isolated_masks(mask)
            final_mask[(mask>0.5) & (mask_quality>prob_mask)] = class_id
            prob_mask = torch.max(prob_mask,mask_quality*mask)
        #interpolate final mask
        final_mask = F.interpolate(final_mask.unsqueeze(0).float(), (img.shape[0], img.shape[1]), mode='nearest').squeeze()  
        prob_mask = F.interpolate(prob_mask.unsqueeze(0).float(), (img.shape[0], img.shape[1]), mode='nearest').squeeze()            
        return final_mask,prob_mask
    
    def remove_small_isolated_masks(self,mask:np.ndarray | torch.Tensor)->np.ndarray | torch.Tensor:
        is_torch_tensor = isinstance(mask, torch.Tensor)   
        if is_torch_tensor:
            device = mask.device
            result_mask = mask.squeeze().cpu().numpy().astype(np.uint8)
        else:
            result_mask = mask.copy()
        _, labels, stats, _= cv2.connectedComponentsWithStats(result_mask,connectivity=8,ltype=cv2.CV_32S)
        #get the toal size of forground pixels
        if labels.shape[0] == 0:
            return result_mask
        #get the biggest foreground mask by sorting, and return both the sorted list and the list of indices
        sorted_with_indices = sorted(enumerate(stats[1:]), key=lambda x: x[1][4],reverse=True)
        #remove small isolated masks
        max_area = sorted_with_indices[0][1][4]
        for label,area in sorted_with_indices:
            #because label 0 as background was ignored in the sorting, so we need to add 1 to label
            label += 1
            if area[4] < 0.1*max_area:
                result_mask[labels == label] = 0
        #any mask
        if is_torch_tensor:
            return torch.tensor(result_mask).to(device)
        return result_mask
    def generate_sam_masks(self, tar_feats, coord_xy, coord_labels):
        """Generate masks using SAM"""
        tar_masks_list = []
        for (points,), (labels,) in zip(batch_iterator(64, coord_xy), batch_iterator(64, coord_labels)):
            in_points = torch.as_tensor(points, device=self.device, dtype=torch.int)
            in_labels = torch.as_tensor(labels, device=self.device, dtype=torch.int)

            tar_masks, scores, logits, _ = self.predictor.predict_torch(
                point_coords=in_points[:, None, :],
                point_labels=in_labels[:, None],
                # mask_input=mask_inputs,
                features=tar_feats,
                multimask_output=False, 
            )
            tar_masks = tar_masks > self.predictor.model.mask_threshold
            tar_masks_list.append(tar_masks)
        return tar_masks_list
    
    def generate_pixelwise_comparison(self, query_feat_high, supp_feat_high):
        pixelwise_coms = []
        for st, supp_feat in enumerate(supp_feat_high):
            tmp_supp_feat = supp_feat.unsqueeze(0)
            tmp_query = query_feat_high.contiguous().permute(0, 2, 1)  # [bs, c, h*w]
            tmp_query_norm = torch.norm(tmp_query, 2, 1, True)
            tmp_supp = tmp_supp_feat.contiguous()
            tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True)

            similarity = torch.bmm(tmp_supp, tmp_query)/(torch.bmm(tmp_supp_norm, tmp_query_norm) + 1e-7) # [bs, h*w, h*w]
            pixelwise_coms.append(similarity)
        return pixelwise_coms
    
    def find_points(self, sim_map, tar_feats):
        """Select points for prompting"""
        
        sum_sim = sim_map.sum()
        topk = min(int(sum_sim), 128) # set maximum to 128 for efficiency
        if topk == 0 and sum_sim > 0:
            topk = 1

        sim_map_flt = sim_map.flatten(0)
        sim_map_topk_args = sim_map_flt.topk(topk)[1]
        sim_map_hot = torch.zeros_like(sim_map_flt)
        sim_map_hot[sim_map_topk_args] = 1
        sim_map_hot = sim_map_hot.view(1, sim_map.shape[1], sim_map.shape[2])
        sim_map_hot = sim_map_hot.squeeze(0).cpu().numpy()

        # translate all points to coordinates
        points_f = np.argwhere(sim_map_hot.T > 0)
        #points1 = self.predictor.transform.apply_coords(points_f, sim_map.shape[-2:])
        points = np.zeros((points_f.shape[0], 2), dtype=np.float32)
        for idx, (x, y) in enumerate(points_f):
            points[idx, 0] = x / sim_map.shape[-1]*self.input_size[0]
            points[idx, 1] = y / sim_map.shape[-2]*self.input_size[1]
        coord_labels = np.ones(points.shape[0], dtype=np.int32)

        return points, coord_labels, sim_map_hot, points_f
    
    def triplet_selection_b(self, tar_masks, cluster_labels, coord_se_labels, mean_sim_map, coord_f, cls_scores):
        """Select and merge the masks"""
        coord_f = torch.as_tensor(coord_f, device=self.device, dtype=torch.long)
        cluster_labels = torch.as_tensor(cluster_labels, device=self.device, dtype=torch.long)
        
        pred_masks = torch.zeros(self.input_size, device=self.device).unsqueeze(0)
        prob_masks = torch.zeros(self.input_size, device=self.device).unsqueeze(0)
        sim_map_rsz = F.interpolate(mean_sim_map.unsqueeze(0), self.input_size, mode='bilinear', align_corners=False).squeeze(0)
        for idx, coord_se_label in enumerate(coord_se_labels):
            if coord_se_label == 1:
                pred_masks += tar_masks[idx]
                curr_prob_mask = (sim_map_rsz * tar_masks[idx]).sum() / tar_masks[idx].sum() * tar_masks[idx]
                prob_masks = torch.max(prob_masks, curr_prob_mask)
        pred_masks = (pred_masks > 0).float()

        # if pred_masks.sum() == 0:
        #     cls_scores[coord_se_labels != 2] = 0
        #     _, max_cls_scores_arg = cls_scores.max(dim=0)
        #     pred_masks += tar_masks[max_cls_scores_arg]
        #     prob_masks = (sim_map_rsz * tar_masks[max_cls_scores_arg]).sum() / tar_masks[max_cls_scores_arg].sum() * tar_masks[max_cls_scores_arg]

        return pred_masks, prob_masks

    
    def cluster_classification(self, tar_masks, cluster_labels, n_components, mean_sim_map, neg_map, coord_f):
        """Classify each points with the guidance from cluster labels and similarity maps"""
        coord_f = torch.as_tensor(coord_f, device=self.device, dtype=torch.long)
        cluster_labels = torch.as_tensor(cluster_labels, device=self.device, dtype=torch.long)

        fgbg_labels = torch.zeros_like(cluster_labels)
        fgbg_com_labels = torch.zeros(n_components, device=self.device, dtype=torch.long)
        cls_scores = torch.zeros_like(cluster_labels, dtype=torch.float)
        pos_region = (mean_sim_map > neg_map).float()
        for component in range(n_components):
            com_args = torch.where(cluster_labels == component)[0]
            union_mask = (tar_masks[com_args].sum(dim=0) > 0).float()
            union_mask = F.interpolate(union_mask.unsqueeze(0), (self.encoder_feat_size, self.encoder_feat_size), mode='bilinear', align_corners=False).squeeze(0)
            pos_score = (pos_region * union_mask).sum()
            neg_score = ((1 - pos_region) * union_mask).sum()

            fgbg_com_labels[component] = 1
            tar_mask_rsz = F.interpolate(tar_masks[com_args].float(), (self.encoder_feat_size, self.encoder_feat_size), mode='bilinear', align_corners=False)
            mask_scores = (tar_mask_rsz * (mean_sim_map - neg_map).unsqueeze(0)).sum(dim=(-1, -2, -3))
            mask_scores = mask_scores / (tar_mask_rsz.sum(dim=(-1, -2, -3)) + 1e-6)
            mask_args = mask_scores.argsort(descending=True)
            inner_mask = torch.zeros_like(union_mask, device=self.device, dtype=torch.float)
            # Mask Growth
            for idx in mask_args:
                curr_mask = tar_mask_rsz[idx]
                curr_mask = curr_mask * (1 - inner_mask)
                pos_score = (pos_region * curr_mask).sum()
                neg_score = ((1 - pos_region) * curr_mask).sum()
                if pos_score > neg_score:
                    inner_mask += curr_mask
                    fgbg_labels[com_args[idx]] = 1
            cls_scores[com_args] = pos_score - neg_score

        pseudo_masks = (tar_masks[fgbg_labels == 1].sum(dim=0) > 0).float()

        return fgbg_com_labels, fgbg_labels, pseudo_masks, cls_scores
             
    def point_consistency_dis(self, tar_feats_sem, tar_masks, cluster_labels, n_components, fgbg_labels, coord_f):
        coord_f = torch.as_tensor(coord_f, device=self.device, dtype=torch.long)
        cluster_labels = torch.as_tensor(cluster_labels, device=self.device, dtype=torch.long)
        fgbg_labels = torch.as_tensor(fgbg_labels, device=self.device, dtype=torch.long)

        union_masks = []
        union_similarities = []
        for component in range(n_components):
            com_args = torch.where(cluster_labels == component)[0]
            union_mask = (tar_masks[com_args].sum(dim=0) > 0).float()
            union_mask = F.interpolate(union_mask.unsqueeze(0), (self.encoder_feat_size, self.encoder_feat_size), mode='bilinear', align_corners=False).squeeze(0)
            union_masks.append(union_mask)
            _, union_similarity,_ = self.generate_prior(tar_feats_sem, tar_feats_sem, union_mask.unsqueeze(0))
            union_similarities.append(union_similarity)
        union_similarities = torch.cat(union_similarities, dim=0) # nc, h, w

        # get similarities from coordinates
        coord_similarities = union_similarities[:, coord_f[:, 1], coord_f[:, 0]] # nc, nm

        # compute distance to the nearest point in the cluster
        coord_distance = torch.cdist(coord_f.float(), coord_f.float()) # nm, nm
        coord_distance_labels = torch.zeros_like(coord_similarities)
        for idx in range(coord_f.shape[0]):
            for compo in range(n_components):
                com_args = torch.where(cluster_labels == compo)[0]
                coord_distance_labels[compo, idx] = coord_distance[idx, com_args].min()
        coord_distance_labels[coord_distance_labels == 0] = 1
        coord_similarities = coord_similarities / coord_distance_labels

        coord_similarities_max, coord_similarities_max_args = coord_similarities.max(dim=0) # nm

        # labeling the points for self consistency
        coord_selection_labels = torch.where(coord_similarities_max_args == cluster_labels, 1, 0)
        for component in range(n_components):
            com_args = torch.where(cluster_labels == component)[0]
            const_count = coord_selection_labels[com_args].sum()
            if const_count < len(com_args) - const_count:
                coord_selection_labels[com_args] = 0
        coord_pos_labels = (coord_selection_labels * fgbg_labels) > 0
        coord_waiting_labels = ((coord_selection_labels + fgbg_labels) > 0).long() * 2
        coord_waiting_labels[coord_pos_labels] = 1

        return coord_waiting_labels

    def mask_cluster(self, tar_masks, coord_f, sim_map_hot):
        """Mask clustering using connected components"""
        sim_map_hot = torch.as_tensor(sim_map_hot>0, device=self.device, dtype=torch.float)

        tar_masks_rsz = F.interpolate(tar_masks.float(), (sim_map_hot.shape[-2], sim_map_hot.shape[-1]), mode='nearest').squeeze(1)
        adjacent = tar_masks_rsz * sim_map_hot.unsqueeze(0) # Compute the coverage

        # squeeze adjacent matrix according to the coordinate
        adjacent = adjacent.flatten(1) # nm, h*w
        coord_f = torch.as_tensor(coord_f, device=self.device, dtype=torch.long)
        coord_f = coord_f[:, 1] * sim_map_hot.shape[-1] + coord_f[:, 0]
        adjacent = adjacent[:, coord_f] # nm, nm

        # strong connected components
        n_components_strong, labels_strong = csgraph.connected_components(adjacent.cpu().numpy(), directed=True, connection='strong', return_labels=True)
        n_components_weak, labels_weak = csgraph.connected_components(adjacent.cpu().numpy(), directed=True, connection='weak', return_labels=True)

        # return n_components, labels
        return n_components_weak, labels_weak, n_components_strong, labels_strong