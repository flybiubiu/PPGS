import os
import random
import argparse

import numpy as np
import torch
from tqdm import tqdm
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2

from dataclasses import dataclass, field
from typing import Tuple, Type
from copy import deepcopy

import torch
import torchvision
from torch import nn
import torch.nn.functional as F

try:
    import open_clip
except ImportError:
    assert False, "open_clip is not installed, install it with `pip install open-clip-torch`"


@dataclass
class OpenCLIPNetworkConfig:
    _target: Type = field(default_factory=lambda: OpenCLIPNetwork)
    clip_model_type: str = "ViT-B-16"
    clip_model_pretrained: str = "laion2b_s34b_b88k"
    clip_n_dims: int = 512
    negatives: Tuple[str] = ("object", "things", "stuff", "texture")
    positives: Tuple[str] = ("",)

class OpenCLIPNetwork(nn.Module):
    def __init__(self, config: OpenCLIPNetworkConfig):
        super().__init__()
        self.config = config
        self.process = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
        model, _, _ = open_clip.create_model_and_transforms(
            self.config.clip_model_type,  # e.g., ViT-B-16
            pretrained=self.config.clip_model_pretrained,  # e.g., laion2b_s34b_b88k
            precision="fp16",
        )
        model.eval()
        self.tokenizer = open_clip.get_tokenizer(self.config.clip_model_type)
        self.model = model.to("cuda")
        self.clip_n_dims = self.config.clip_n_dims

        self.positives = self.config.positives    
        self.negatives = self.config.negatives
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to("cuda")
            self.pos_embeds = model.encode_text(tok_phrases)
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.negatives]).to("cuda")
            self.neg_embeds = model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)
        self.neg_embeds /= self.neg_embeds.norm(dim=-1, keepdim=True)

        assert (
            self.pos_embeds.shape[1] == self.neg_embeds.shape[1]
        ), "Positive and negative embeddings must have the same dimensionality"
        assert (
            self.pos_embeds.shape[1] == self.clip_n_dims
        ), "Embedding dimensionality must match the model dimensionality"

    @property
    def name(self) -> str:
        return "openclip_{}_{}".format(self.config.clip_model_type, self.config.clip_model_pretrained)

    @property
    def embedding_dim(self) -> int:
        return self.config.clip_n_dims
    
    def gui_cb(self,element):
        self.set_positives(element.value.split(";"))

    def set_positives(self, text_list):
        self.positives = text_list
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to("cuda")
            self.pos_embeds = self.model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)

    def get_relevancy(self, embed: torch.Tensor, positive_id: int) -> torch.Tensor:
        phrases_embeds = torch.cat([self.pos_embeds, self.neg_embeds], dim=0)
        p = phrases_embeds.to(embed.dtype)  # phrases x 512
        output = torch.mm(embed, p.T)  # rays x phrases
        positive_vals = output[..., positive_id : positive_id + 1]  # rays x 1
        negative_vals = output[..., len(self.positives) :]  # rays x N_phrase
        repeated_pos = positive_vals.repeat(1, len(self.negatives))  # rays x N_phrase

        sims = torch.stack((repeated_pos, negative_vals), dim=-1)  # rays x N-phrase x 2
        softmax = torch.softmax(10 * sims, dim=-1)  # rays x n-phrase x 2
        best_id = softmax[..., 0].argmin(dim=1)  # rays x 2
        return torch.gather(softmax, 1, best_id[..., None, None].expand(best_id.shape[0], len(self.negatives), 2))[:, 0, :]

    def encode_image(self, input):
        processed_input = self.process(input).half()
        return self.model.encode_image(processed_input)


def create(image_list, data_list, save_folder):
    assert image_list is not None, "image_list must be provided to generate features"
    embed_size=512
    #seg_maps = []
    #total_lengths = []
    timer = 0
    img_embeds = torch.zeros(1, *image_list[0].shape[1:], embed_size)
    #seg_imgs = torch.zeros(50, 224, 224, 3)
    #seg_masks = torch.zeros(50, *image_list[0].shape[1:])
    seg_maps = torch.zeros(1, *image_list[0].shape[1:], 1)
    h,w = image_list[0].shape[1:]
    #print(h, w)
    #print(img_embeds.shape)torch.Size([177, 300, 512])
    #print(seg_maps.shape)torch.Size([177, 4, 730, 988])
    mask_generator.predictor.model.to('cuda')

    #y, x = torch.meshgrid(torch.arange(0, *image_list[0].shape[1:2]), torch.arange(0, *image_list[0].shape[2:]))
    #x = x.reshape(-1, 1)
    #y = y.reshape(-1, 1)

    for i, img in tqdm(enumerate(image_list), desc="Embedding images", leave=False):
        timer += 1
        try:
            #print(img.unsqueeze(0).shape)torch.Size([1, 3, 730, 988])
            #img_embed, seg_map, seg_img, seg_mask = _embed_clip_sam_tiles(img.unsqueeze(0), sam_encoder)
            img_embed, seg_img, seg_map, seg_mask = _embed_clip_sam_tiles(img.unsqueeze(0), sam_encoder)
        except:
            raise ValueError(timer)

        lengths = [len(v) for k, v in seg_img.items()]
        #print(len(lengths))4
        total_length = sum(lengths)
        #total_lengths.append(total_length)

        #if total_length > seg_imgs.shape[0]:
            #pad = total_length - seg_imgs.shape[0]
        seg_maps_pad = len(lengths) - seg_maps.shape[0]
            #img_embeds = torch.cat([img_embeds,torch.zeros((len(image_list), pad, embed_size))], dim=1)
            #seg_imgs = torch.cat([seg_imgs, torch.zeros(pad, 224, 224, 3)], dim=0)
            #seg_masks = torch.cat([seg_masks, torch.zeros(pad, *image_list[0].shape[1:])], dim=0)
        if seg_maps_pad > 0:
            seg_maps = torch.cat([seg_maps, torch.zeros(seg_maps_pad, *image_list[0].shape[1:], 1)], dim=0)
            img_embeds = torch.cat([img_embeds, torch.zeros(seg_maps_pad, *image_list[0].shape[1:], embed_size)], dim=0)

        #img_embed = torch.cat([v for k, v in img_embed.items()], dim=0)
        #seg_img = torch.cat([v for k, v in seg_img.items()], dim=0)
        #seg_mask = torch.cat([v for k, v in seg_mask.items()], dim=0)
        #assert img_embed.shape[0] == total_length
        #assert seg_img.shape[0] == total_length
        #assert seg_mask.shape[0] == total_length
        #img_embeds[i, :total_length] = img_embed#torch.Size([300, 512])
        #print(img_embeds[i, :total_lengths[i]].shape)torch.Size([300, 512])
        #seg_imgs[:total_length] = seg_img.permute(0,2,3,1)
        #print(seg_img.shape)torch.Size([329, 3, 224, 224])
        #seg_masks[:total_length] = seg_mask

        #for (k, v) in enumerate(seg_map.items()):
        #    print(k, v[1].shape)
        # 0 (730, 988)
        # 1 (730, 988)
        # 2 (730, 988)
        # 3 (730, 988)

        seg_map_tensor = []
        lengths_cumsum = lengths.copy()
        #print(lengths_cumsum)[53, 214, 47, 15]
        for j in range(1, len(lengths)):
            lengths_cumsum[j] += lengths_cumsum[j-1]
        #print(lengths_cumsum)[88, 205, 267, 300]

        for j, (k, v) in enumerate(seg_map.items()):
            if j == 0:
                seg_map_tensor.append(torch.from_numpy(v))
                continue
            assert v.max() == lengths[j] - 1, f"{j}, {v.max()}, {lengths[j]-1}"
            v[v != -1] += lengths_cumsum[j-1]
            seg_map_tensor.append(torch.from_numpy(v))
        seg_map = torch.stack(seg_map_tensor, dim=0)
        #print(seg_map.shape)torch.Size([4, 730, 988])
        seg_maps = seg_map
        #print(seg_maps.shape)torch.Size([4, 520, 779])

        clip_img_embed = []
        for j, (key1, key2) in enumerate(zip(img_embed.keys(), seg_mask.keys())):
            #print(k)default
            #print(v.shape)torch.Size([53, 512])
            #print(img_embed[key1].dtype)torch.float16
            clip_tensor = torch.zeros(h, w, embed_size, dtype = torch.float16)
            #print(clip_tensor.dtype)torch.float16
            for l in range(img_embed[key1].shape[0]):
                #print(img_embed[key1][l].shape)torch.Size([512])
                new_v = img_embed[key1][l].repeat(h, w, 1) * seg_mask[key2][l].unsqueeze(-1)
                clip_tensor += new_v
                #print(new_v.shape)torch.Size([520, 779, 512])
                #print(seg_mask[key2][l].shape)torch.Size([520, 779])
            clip_tensor = F.interpolate(clip_tensor.unsqueeze(0).permute(0, 3, 1, 2), size=(60, 90), mode='nearest').squeeze(0)
            clip_img_embed.append(clip_tensor)
        clip_img_embed = torch.stack(clip_img_embed, dim=0)
        img_embeds = clip_img_embed


            #if j == 0:
                #seg_map_tensor.extend(torch.from_numpy(v))
                #continue




    #for i in range(seg_imgs.shape[0]):
        save_path = os.path.join(save_folder, data_list[i].split('.')[0])
        #assert total_lengths[i] == int(seg_imgs[i].max() + 1)
        save_path_imgs_embeds = save_path + '_imgs_embeds.npy'
        #save_path_mask = save_path + '_masks.npy'
        save_path_seg_maps = save_path + '_seg_maps.npy'
        np.save(save_path_imgs_embeds, img_embeds.numpy())
        #np.save(save_path_mask, seg_masks.numpy())
        np.save(save_path_seg_maps, seg_maps.numpy())
    '''
        curr = {
            #'feature': img_embeds[i, :total_lengths[i]],
            #'seg_maps': seg_maps[i],
            'seg_imgs': seg_imgs[i],
            'seg_masks': seg_masks[i]
        }
        sava_numpy(save_path, curr)
    '''
    mask_generator.predictor.model.to('cpu')

def sava_numpy(save_path, data):
    #save_path_s = save_path + '_s.npy'
    #save_path_f = save_path + '_f.npy'
    save_path_seg_img = save_path + '_seg_imgs.npy'
    save_path_mask = save_path + '_masks.npy'

    #np.save(save_path_s, data['seg_maps'].numpy())
    #np.save(save_path_f, data['feature'].numpy())
    np.save(save_path_seg_img, data['seg_imgs'].numpy())
    np.save(save_path_mask, data['seg_masks'].numpy())

def _embed_clip_sam_tiles(image, sam_encoder):
    #print(image.shape)torch.Size([1, 3, 730, 988])
    aug_imgs = torch.cat([image])
    #print(aug_imgs.shape)torch.Size([1, 3, 730, 988])
    seg_images, seg_map, seg_masks = sam_encoder(aug_imgs)
    #print(seg_masks['default'].shape)torch.Size([53, 520, 779])
    #print(seg_map['default'].shape)(520, 779)
    #clip_maps = np.zeros((4, image.shape[2], image.shape[3], 512), dtype = np.float32)
    #clip_hw = np.zeros((4, image.shape[2], image.shape[3], 512), dtype=np.float32)
    #print(clip_map.shape)(4, 520, 779, 512)
    '''考虑下怎么写
    y, x = torch.meshgrid(torch.arange(0, self.image_height), torch.arange(0, self.image_width))
    x = x.reshape(-1, 1)  # torch.Size([721240, 1])
    y = y.reshape(-1, 1)  # torch.Size([721240, 1])
    seg = seg_map[:, y, x].squeeze(-1).long()
    '''
    clip_embeds = {}
    for mode in ['defalut']:#, 's', 'm', 'l']:
        tiles = seg_images[mode]# 3  224 224
        tiles = tiles.to("cuda")
        with torch.no_grad():
            clip_embed = model.encode_image(tiles)
        clip_embed /= clip_embed.norm(dim=-1, keepdim=True)
        clip_embeds[mode] = clip_embed.detach().cpu().half()
        #print(clip_embed.shape)torch.Size([53, 512])
        #for i in range(len(clip_embed)):
            #print(clip_embed[i].shape)torch.Size([512])
            #clip_hw = clip_embed[i]
            #clip_hw * seg_masks[mode]
        #print(clip_embed.shape)torch.Size([88, 512])
        #torch.Size([117, 512])
        #torch.Size([62, 512])
        #torch.Size([33, 512])


    #return clip_embeds, seg_map, seg_images, seg_masks
    return clip_embeds, seg_images, seg_map, seg_masks

def get_seg_img(mask, image):
    #print(mask.keys())dict_keys(['segmentation', 'area', 'bbox',
    # 'predicted_iou', 'point_coords', 'stability_score', 'crop_box'])
    image = image.copy()
    #cv2.imwrite("2.jpg", mask['segmentation'].astype(int)*255.0)
    image[mask['segmentation']==0] = np.array([0, 0,  0], dtype=np.uint8)
    x,y,w,h = np.int32(mask['bbox'])
    seg_img = image[y:y+h, x:x+w, ...]
    #cv2.imwrite("1.jpg", seg_img)
    return seg_img

def pad_img(img):
    h, w, _ = img.shape
    l = max(w,h)
    pad = np.zeros((l,l,3), dtype=np.uint8)
    if h > w:
        pad[:,(h-w)//2:(h-w)//2 + w, :] = img
    else:
        pad[(w-h)//2:(w-h)//2 + h, :, :] = img
    return pad

def filter(keep: torch.Tensor, masks_result) -> None:
    keep = keep.int().cpu().numpy()
    result_keep = []
    for i, m in enumerate(masks_result):
        if i in keep: result_keep.append(m)
    return result_keep

def mask_nms(masks, scores, iou_thr=0.7, score_thr=0.1, inner_thr=0.2, **kwargs):
    """
    Perform mask non-maximum suppression (NMS) on a set of masks based on their scores.
    
    Args:
        masks (torch.Tensor): has shape (num_masks, H, W)
        scores (torch.Tensor): The scores of the masks, has shape (num_masks,)
        iou_thr (float, optional): The threshold for IoU.
        score_thr (float, optional): The threshold for the mask scores.
        inner_thr (float, optional): The threshold for the overlap rate.
        **kwargs: Additional keyword arguments.
    Returns:
        selected_idx (torch.Tensor): A tensor representing the selected indices of the masks after NMS.
    """

    scores, idx = scores.sort(0, descending=True)
    num_masks = idx.shape[0]
    
    masks_ord = masks[idx.view(-1), :]
    masks_area = torch.sum(masks_ord, dim=(1, 2), dtype=torch.float)

    iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
    inner_iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
    for i in range(num_masks):
        for j in range(i, num_masks):
            intersection = torch.sum(torch.logical_and(masks_ord[i], masks_ord[j]), dtype=torch.float)
            union = torch.sum(torch.logical_or(masks_ord[i], masks_ord[j]), dtype=torch.float)
            iou = intersection / union
            iou_matrix[i, j] = iou
            # select mask pairs that may have a severe internal relationship
            if intersection / masks_area[i] < 0.5 and intersection / masks_area[j] >= 0.85:
                inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                inner_iou_matrix[i, j] = inner_iou
            if intersection / masks_area[i] >= 0.85 and intersection / masks_area[j] < 0.5:
                inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                inner_iou_matrix[j, i] = inner_iou

    iou_matrix.triu_(diagonal=1)
    iou_max, _ = iou_matrix.max(dim=0)
    inner_iou_matrix_u = torch.triu(inner_iou_matrix, diagonal=1)
    inner_iou_max_u, _ = inner_iou_matrix_u.max(dim=0)
    inner_iou_matrix_l = torch.tril(inner_iou_matrix, diagonal=1)
    inner_iou_max_l, _ = inner_iou_matrix_l.max(dim=0)
    
    keep = iou_max <= iou_thr
    keep_conf = scores > score_thr
    keep_inner_u = inner_iou_max_u <= 1 - inner_thr
    keep_inner_l = inner_iou_max_l <= 1 - inner_thr
    
    # If there are no masks with scores above threshold, the top 3 masks are selected
    if keep_conf.sum() == 0:
        index = scores.topk(3).indices
        keep_conf[index, 0] = True
    if keep_inner_u.sum() == 0:
        index = scores.topk(3).indices
        keep_inner_u[index, 0] = True
    if keep_inner_l.sum() == 0:
        index = scores.topk(3).indices
        keep_inner_l[index, 0] = True
    keep *= keep_conf
    keep *= keep_inner_u
    keep *= keep_inner_l

    selected_idx = idx[keep]
    return selected_idx

def masks_update(*args, **kwargs):
    # remove redundant masks based on the scores and overlap rate between masks
    masks_new = ()
    for masks_lvl in (args):
        seg_pred =  torch.from_numpy(np.stack([m['segmentation'] for m in masks_lvl], axis=0))
        iou_pred = torch.from_numpy(np.stack([m['predicted_iou'] for m in masks_lvl], axis=0))
        stability = torch.from_numpy(np.stack([m['stability_score'] for m in masks_lvl], axis=0))

        scores = stability * iou_pred
        keep_mask_nms = mask_nms(seg_pred, scores, **kwargs)
        masks_lvl = filter(keep_mask_nms, masks_lvl)

        masks_new += (masks_lvl,)
    return masks_new

def sam_encoder(image):
    image = cv2.cvtColor(image[0].permute(1,2,0).numpy().astype(np.uint8), cv2.COLOR_BGR2RGB)
    # pre-compute masks
    masks_default, masks_s, masks_m, masks_l = mask_generator.generate(image)
    #print(image.shape) (730, 988, 3)
    #print(len(masks_default))212
    #print(len(masks_s))168
    #print(len(masks_m))98
    #print(len(masks_l))41

    # pre-compute postprocess
    masks_default, masks_s, masks_m, masks_l = \
        masks_update(masks_default, masks_s, masks_m, masks_l, iou_thr=0.8, score_thr=0.7, inner_thr=0.5)
    #print(len(masks_default))88
    #print(len(masks_s))117
    #print(len(masks_m))62
    #print(len(masks_l))33

    def mask2segmap(masks, image):
        seg_img_list = []
        seg_mask_list = []
        seg_map = -np.ones(image.shape[:2], dtype=np.int32)
        for i in range(len(masks)):
            mask = masks[i]
            seg_img = get_seg_img(mask, image)
            pad_seg_img = cv2.resize(pad_img(seg_img), (224,224))
            seg_img_list.append(pad_seg_img)
            seg_map[masks[i]['segmentation']] = i
            seg_mask_list.append(masks[i]['segmentation'])#(519, 778)


        #cv2.imwrite("3.jpg", seg_map*3.0)
        seg_imgs = np.stack(seg_img_list, axis=0) # b,224,224,3
        seg_imgs = (torch.from_numpy(seg_imgs.astype("float32")).permute(0,3,1,2) / 255.0).to('cuda')
        #print(seg_masks.shape)(89, 519, 778)
        seg_masks = np.stack(seg_mask_list, axis = 0)
        seg_masks = torch.from_numpy(seg_masks)

        return seg_imgs, seg_map, seg_masks

    seg_images, seg_maps, seg_masks = {}, {}, {}
    seg_images['default'], seg_maps['default'], seg_masks['default'] = mask2segmap(masks_default, image)
    #seg_images['s'], seg_maps['s'], seg_masks['s'] = mask2segmap(masks_s, image)
    #print(seg_images['default'].shape)torch.Size([88, 3, 224, 224])
    '''
    if len(masks_s) != 0:
        seg_images['s'], seg_maps['s'], seg_masks['s']  = mask2segmap(masks_s, image)
        #print(seg_images['s'].shape)torch.Size([117, 3, 224, 224])
    if len(masks_m) != 0:
        seg_images['m'], seg_maps['m'], seg_masks['m'] = mask2segmap(masks_m, image)
        #print(seg_images['m'].shape)torch.Size([62, 3, 224, 224])
    if len(masks_l) != 0:
        seg_images['l'], seg_maps['l'], seg_masks['l'] = mask2segmap(masks_l, image)
        #print(seg_images['l'].shape)torch.Size([33, 3, 224, 224])
    '''

    # 0:default 1:s 2:m 3:l
    return seg_images, seg_maps, seg_masks

def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    seed_num = 42
    seed_everything(seed_num)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--resolution', type=int, default=1)
    parser.add_argument('--sam_ckpt_path', type=str, default="../ckpts/sam_vit_h_4b8939.pth")
    args = parser.parse_args()
    torch.set_default_dtype(torch.float32)

    dataset_path = args.dataset_path
    sam_ckpt_path = args.sam_ckpt_path
    img_folder = os.path.join(dataset_path, 'images')
    data_list = os.listdir(img_folder)
    data_list.sort()

    model = OpenCLIPNetwork(OpenCLIPNetworkConfig)
    sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt_path).to('cuda')
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.7,
        box_nms_thresh=0.7,
        stability_score_thresh=0.85,
        crop_n_layers=1,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=100,
    )

    img_list = []
    WARNED = False
    for data_path in data_list:
        if data_path.lower().endswith(".jpg"):
            image_path = os.path.join(img_folder, data_path)
            image = cv2.imread(image_path)
            orig_w, orig_h = image.shape[1], image.shape[0]
            if args.resolution in [1, 2, 4, 8]:
                resolution = (round(orig_w / args.resolution), round(orig_h / args.resolution))
            image = cv2.resize(image, resolution)
            image = torch.from_numpy(image)
            img_list.append(image)
    images = [img_list[i].permute(2, 0, 1)[None, ...] for i in range(len(img_list))]
    imgs = torch.cat(images)
    #print(imgs.shape)torch.Size([177, 3, 730, 988])

    save_folder = os.path.join(dataset_path, 'language_features')
    os.makedirs(save_folder, exist_ok=True)
    create(imgs, data_list, save_folder)