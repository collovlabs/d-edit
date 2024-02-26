
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from PIL import Image
import torch
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches
import os
import numpy as np
import argparse
import matplotlib

def load_image(image_path, left=0, right=0, top=0, bottom=0, size = 512):
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

def draw_panoptic_segmentation(segmentation, segments_info,save_folder=None, noseg = False):
    if torch.max(segmentation)==torch.min(segmentation)==-1:
        print("nothing is detected!")
        noseg=True
        viridis = matplotlib.colormaps['viridis'].resampled(1)
    else:
        viridis = matplotlib.colormaps['viridis'].resampled(torch.max(segmentation)-torch.min(segmentation)+1)
    fig, ax = plt.subplots()
    ax.imshow(segmentation)
    instances_counter = defaultdict(int)
    handles = []
    label_list = []
    if not noseg:
        if torch.min(segmentation) == 0: 
            mask = segmentation==0
            mask = mask.cpu().detach().numpy()   # [512,512]   bool
            segment_label = "rest"
            np.save( os.path.join(save_folder, "mask{}_{}.npy".format(0,"rest")) , mask)
            color = viridis(0)
            label = f"{segment_label}-{0}"
            handles.append(mpatches.Patch(color=color, label=label))
            label_list.append(label)
       
        for  segment in segments_info:
            segment_id = segment['id']
            mask = segmentation==segment_id
            if torch.min(segmentation) != 0: 
                segment_id -= 1
            mask = mask.cpu().detach().numpy()   # [512,512] bool
            
            segment_label = model.config.id2label[segment['label_id']]
            instances_counter[segment['label_id']] += 1
            np.save( os.path.join(save_folder, "mask{}_{}.npy".format(segment_id,segment_label)) , mask)
            color = viridis(segment_id)
            
            label = f"{segment_label}-{segment_id}"
            handles.append(mpatches.Patch(color=color, label=label))
            label_list.append(label)
    else:
        mask = np.full(segmentation.shape, True)
        segment_label = "all"
        np.save( os.path.join(save_folder, "mask{}_{}.npy".format(0,"all")) , mask)
        color = viridis(0)
        label = f"{segment_label}-{0}"
        handles.append(mpatches.Patch(color=color, label=label))
        label_list.append(label)

    plt.xticks([])
    plt.yticks([])
    # plt.savefig(os.path.join(save_folder, 'mask_clear.png'), dpi=500)
    ax.legend(handles=handles)
    plt.savefig(os.path.join(save_folder, 'seg_init.png'), dpi=500 )
    print("; ".join(label_list))



parser = argparse.ArgumentParser()
parser.add_argument("--name",  type=str, default="obama")
parser.add_argument("--size",  type=int, default=512)
parser.add_argument("--noseg", default=False, action="store_true" )
args = parser.parse_args()
base_folder_path = "."

processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-base-coco-panoptic")
model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-base-coco-panoptic")
input_folder = os.path.join(base_folder_path, args.name )
try:
    image = load_image(os.path.join(input_folder, "img.png" ), size = args.size)
except:
    image = load_image(os.path.join(input_folder, "img.jpg" ), size = args.size)

image =Image.fromarray(image)
image.save(os.path.join(input_folder,"img_{}.png".format(args.size)))
inputs = processor(image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

panoptic_segmentation = processor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
save_folder = os.path.join(base_folder_path, args.name)
os.makedirs(save_folder, exist_ok=True)
draw_panoptic_segmentation(**panoptic_segmentation, save_folder = save_folder, noseg = args.noseg)