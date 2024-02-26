import argparse
import os
import copy
import shutil

import numpy as np
import json
import torch
from PIL import Image, ImageDraw, ImageFont

# Grounding DINO
import sys

sys.path.append("/path/to/Grounded-Segment-Anything")
# change to your "Grounded-Segment-Anything" installation folder!!!!!
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import (
    sam_model_registry,
    sam_hq_model_registry,
    SamPredictor
)
import cv2
import numpy as np
import matplotlib.pyplot as plt
def load_image_to_resize(image_path, left=0, right=0, top=0, bottom=0, size = 512):
    if type(image_path) is str:
        image = np.array(Image.open(image_path))[:, :, :3]
    else:
        image = image_path
    h, w, c = image.shape
    left = min(left, w-1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h-bottom, left:w-right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    image = np.array(Image.fromarray(image).resize((size, size)))
    return image


def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    ax.text(x0, y0, label)


def save_mask_data(output_dir, mask_list, box_list, label_list):
    value = 0  # 0 for background

    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

    json_data = [{
        'value': value,
        'label': 'background'
    }]
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1] # the last is ')'
        json_data.append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })
    with open(os.path.join(output_dir, 'mask.json'), 'w') as f:
        json.dump(json_data, f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--sam_version", type=str, default="vit_h", required=False, help="SAM ViT version: vit_b / vit_l / vit_h")
    parser.add_argument("--sam_checkpoint", type=str, required=False, help="path to sam checkpoint file")
    parser.add_argument("--sam_hq_checkpoint", type=str, default=None, help="path to sam-hq checkpoint file")
    parser.add_argument("--use_sam_hq", action="store_true", help="using sam-hq for prediction")
    parser.add_argument("--text_prompt", type=str, required=True, help="text prompt")
    
    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    parser.add_argument("--device", type=str, default="cpu", help="running on cpu only!, default=False")
    parser.add_argument("--name", type=str, default="", help="name of the input image folder")
    parser.add_argument("--size", type=int, default=1024, help="image size")

    args = parser.parse_args()
    args.base_folder = "/path/to/Grounded-Segment-Anything"
    # change to your "Grounded-Segment-Anything" installation folder!!!!!
    input_folder = os.path.join(".", args.name)
    
    args.config = os.path.join(args.base_folder,"GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    args.grounded_checkpoint = "groundingdino_swint_ogc.pth"
    args.sam_checkpoint="sam_vit_h_4b8939.pth"
    args.box_threshold = 0.3
    args.text_threshold = 0.25
    args.device = "cuda"
    # cfg
    
    config_file = args.config  # change the path of the model config file
    grounded_checkpoint = os.path.join(args.base_folder,args.grounded_checkpoint)  # change the path of the model
    sam_version = args.sam_version
    sam_checkpoint = os.path.join(args.base_folder,args.sam_checkpoint)
    if args.sam_hq_checkpoint is not None:
        sam_hq_checkpoint = os.path.join(args.base_folder,args.sam_hq_checkpoint)
    use_sam_hq = args.use_sam_hq
    # image_path = args.input_image
    text_prompt = args.text_prompt
    # output_dir = args.output_dir
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    device = args.device
    
    output_dir = input_folder
    os.makedirs(output_dir, exist_ok=True)
    
    # unify names

    if len(os.listdir(input_folder)) == 1:
        for filename in os.listdir(input_folder):
            imgtype = "." + filename.split(".")[-1]
            shutil.move(os.path.join(input_folder, filename), os.path.join(input_folder, "img"+imgtype)) 
            
            
            
    ### resizing and save
    if os.path.exists(os.path.join(input_folder, "img.jpg")):
        image_path = os.path.join(input_folder, "img.jpg")
    else:
        image_path = os.path.join(input_folder, "img.png")
    image = load_image_to_resize(image_path, size = args.size)
    image =Image.fromarray(image)
    resized_image_path = os.path.join(input_folder, "img_{}.png".format(args.size))
    image.save(resized_image_path)
 
    image_path = resized_image_path
    # load image
    image_pil, image = load_image(image_path)
    # load model
    model = load_model(config_file, grounded_checkpoint, device=device)

    # # visualize raw image
    # image_pil.save(os.path.join(output_dir, "raw_image.jpg"))

    # run grounding dino model
    boxes_filt, pred_phrases = get_grounding_output(
        model, image, text_prompt, box_threshold, text_threshold, device=device
    )

    # initialize SAM
    if use_sam_hq:
        predictor = SamPredictor(sam_hq_model_registry[sam_version](checkpoint=sam_hq_checkpoint).to(device))
    else:
        predictor = SamPredictor(sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(device))
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    size = image_pil.size
    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu()
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)

    masks, _, _ = predictor.predict_torch(
        point_coords = None,
        point_labels = None,
        boxes = transformed_boxes.to(device),
        multimask_output = False,
    )
    
    tot_detect = len(masks)
    # draw output image
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for idx, (mask,label) in enumerate(zip(masks,pred_phrases)):
        show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
        np.save( os.path.join(output_dir, "maskSAM{}_{}.npy".format(idx, label)) ,mask[0].cpu().numpy())
        
    for idx, (box, label) in enumerate(zip(boxes_filt, pred_phrases)):
        label = label + "_{}".format(idx)
        show_box(box.numpy(), plt.gca(), label)
    
    rec_mask = np.zeros_like(mask[0].cpu().numpy()).astype(np.bool_)
    for idx, box in enumerate(boxes_filt):
        up = box[0].numpy().astype(np.int32)
        down = box[2].numpy().astype(np.int32)
        left = box[1].numpy().astype(np.int32)
        right = box[3].numpy().astype(np.int32)
        rec_mask[left:right, up:down] = True

    plt.axis('off')
    plt.savefig(
        os.path.join(output_dir, "seg_init_SAM.png"),
        bbox_inches="tight", dpi=300, pad_inches=0.0
    )
    
    mask_detected = np.logical_or.reduce([mask[0].cpu().numpy() for mask in masks ])
    mask_undetected = np.logical_not(mask_detected)
    np.save( os.path.join(output_dir, "SAM_detected.npy") ,mask_detected)
    np.save( os.path.join(output_dir, "maskSAM{}_rest.npy".format(len(masks)))     ,mask_undetected)
    plt.imsave( os.path.join(output_dir,"mask_SAM-detected.png"), np.repeat(np.expand_dims( mask_detected.astype(float), axis=2), 3, axis = 2))
    