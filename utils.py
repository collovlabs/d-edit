from transformers import PretrainedConfig
from PIL import Image
import torch
import numpy as np
import PIL
import os
from tqdm.auto import tqdm
from diffusers.models.attention_processor import (
    AttnProcessor2_0,
    LoRAAttnProcessor2_0,
    LoRAXFormersAttnProcessor,
    XFormersAttnProcessor,
)

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

def myroll2d(a, delta_x, delta_y):
    h, w = a.shape[0],  a.shape[1]
    delta_x = -delta_x
    delta_y = -delta_y
    if isinstance(a, np.ndarray):
        b = np.zeros   ([h,w]).astype(np.uint8)
    elif isinstance(a, torch.Tensor):
        b = torch.zeros([h,w]).to(torch.uint8)
    if delta_x > 0:
        left_a = delta_x
        right_a = w
        left_b = 0
        right_b = w - delta_x
    else:
        left_a = 0
        right_a = w + delta_x
        left_b = -delta_x
        right_b =  w
    if delta_y > 0:
        top_a = delta_y
        bot_a = h
        top_b = 0
        bot_b = h-delta_y
    else:
        top_a = 0
        bot_a = h + delta_y
        top_b = -delta_y
        bot_b = h
    b[left_b: right_b, top_b: bot_b] = a[left_a: right_a, top_a: bot_a]
    return b

def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision = None, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection
        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")

@torch.no_grad()
def image2latent(image, vae = None, dtype=None):
    with torch.no_grad():
        if type(image) is Image or type(image) is PIL.PngImagePlugin.PngImageFile or type(image) is PIL.JpegImagePlugin.JpegImageFile:
            image = np.array(image)
        if type(image) is torch.Tensor and image.dim() == 4:
            latents = image
        else:
            image = torch.from_numpy(image).float() / 127.5 - 1
            image = image.permute(2, 0, 1).unsqueeze(0).to(device, dtype= dtype)
            latents = vae.encode(image).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
    return latents

@torch.no_grad()
def latent2image(latents, return_type = 'np', vae = None):
    # needs_upcasting = vae.dtype == torch.float16 and vae.config.force_upcast
    needs_upcasting = True
    if needs_upcasting:
        upcast_vae(vae)
        latents = latents.to(next(iter(vae.post_quant_conv.parameters())).dtype)
    image = vae.decode(latents /vae.config.scaling_factor, return_dict=False)[0]
    
    if return_type == 'np':
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()#[0]
        image = (image * 255).astype(np.uint8)
    if needs_upcasting:
        vae.to(dtype=torch.float16)
    return image

def upcast_vae(vae):
    dtype = vae.dtype
    vae.to(dtype=torch.float32)
    use_torch_2_0_or_xformers = isinstance(
        vae.decoder.mid_block.attentions[0].processor,
        (
            AttnProcessor2_0,
            XFormersAttnProcessor,
            LoRAXFormersAttnProcessor,
            LoRAAttnProcessor2_0,
        ),
    )
    # if xformers or torch_2_0 is used attention block does not need
    # to be in float32 which can save lots of memory
    if use_torch_2_0_or_xformers:
        vae.post_quant_conv.to(dtype)
        vae.decoder.conv_in.to(dtype)
        vae.decoder.mid_block.to(dtype)

def prompt_to_emb_length_sdxl(prompt, tokenizer, text_encoder, length = None):
    text_input = tokenizer(
        [prompt],
        padding="max_length",
        max_length=length,
        truncation=True,
        return_tensors="pt",
    )
    prompt_embeds = text_encoder(text_input.input_ids.to(device),output_hidden_states=True)
    pooled_prompt_embeds = prompt_embeds[0]

    prompt_embeds = prompt_embeds.hidden_states[-2]
    bs_embed, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    
    return  {"prompt_embeds": prompt_embeds, "pooled_prompt_embeds": pooled_prompt_embeds}




def prompt_to_emb_length_sd(prompt, tokenizer, text_encoder,  length = None):
    text_input = tokenizer(
        [prompt],
        padding="max_length",
        max_length=length,
        truncation=True,
        return_tensors="pt",
    )
    emb = text_encoder(text_input.input_ids.to(device))[0]
    return  emb 

def sdxl_prepare_input_decom(
    set_string_list,
    tokenizer,
    tokenizer_2,
    text_encoder_1,
    text_encoder_2,
    length = 20,
    bsz = 1,
    weight_dtype = torch.float32,
    resolution = 1024,
    normal_token_id_list = []
):
    encoder_hidden_states_list = []
    pooled_prompt_embeds = 0

    for m_idx in range(len(set_string_list)):
        prompt_embeds_list = []
        if ("#" in set_string_list[m_idx] or "$" in set_string_list[m_idx]) and m_idx not in normal_token_id_list :  ###
            out = prompt_to_emb_length_sdxl(
                set_string_list[m_idx], tokenizer, text_encoder_1, length = length
            )
        else:
            out = prompt_to_emb_length_sdxl(
                set_string_list[m_idx], tokenizer, text_encoder_1, length = 77
            )
            print(m_idx, set_string_list[m_idx])
        prompt_embeds, _ = out["prompt_embeds"].to(dtype=weight_dtype), out["pooled_prompt_embeds"].to(dtype=weight_dtype)
        prompt_embeds = prompt_embeds.repeat(bsz, 1, 1)
        prompt_embeds_list.append(prompt_embeds)
        if ("#" in set_string_list[m_idx] or "$" in set_string_list[m_idx]) and m_idx not in  normal_token_id_list:
            out = prompt_to_emb_length_sdxl(
                set_string_list[m_idx], tokenizer_2, text_encoder_2, length = length
            )
        else:
            out = prompt_to_emb_length_sdxl(
                set_string_list[m_idx], tokenizer_2, text_encoder_2, length = 77
            )
            print(m_idx, set_string_list[m_idx])

        prompt_embeds = out["prompt_embeds"].to(dtype=weight_dtype)
        pooled_prompt_embeds += out["pooled_prompt_embeds"].to(dtype=weight_dtype)
        prompt_embeds = prompt_embeds.repeat(bsz, 1, 1)
        prompt_embeds_list.append(prompt_embeds)
            
        encoder_hidden_states_list.append(torch.concat(prompt_embeds_list, dim=-1))
        
    add_text_embeds = pooled_prompt_embeds /len(set_string_list)
    target_size, original_size,crops_coords_top_left = (resolution,resolution),(resolution,resolution),(0,0)
    add_time_ids = list(original_size + crops_coords_top_left + target_size)

    add_time_ids = torch.tensor([add_time_ids], dtype=prompt_embeds.dtype,device = pooled_prompt_embeds.device) #[B,6]
    return encoder_hidden_states_list, add_text_embeds, add_time_ids

def sd_prepare_input_decom(
    set_string_list,
    tokenizer,
    text_encoder_1,
    length = 20,
    bsz = 1,
    weight_dtype = torch.float32,
    normal_token_id_list = []
):
    encoder_hidden_states_list = []
    for m_idx in range(len(set_string_list)):
        if ("#" in set_string_list[m_idx] or "$" in set_string_list[m_idx]) and m_idx not in normal_token_id_list :  ###
            encoder_hidden_states = prompt_to_emb_length_sd(
                set_string_list[m_idx], tokenizer, text_encoder_1, length = length
            )
        else:
            encoder_hidden_states = prompt_to_emb_length_sd(
                set_string_list[m_idx], tokenizer, text_encoder_1, length = 77
            )
            print(m_idx, set_string_list[m_idx])
        encoder_hidden_states = encoder_hidden_states.repeat(bsz, 1, 1)
        encoder_hidden_states_list.append(encoder_hidden_states.to(dtype=weight_dtype))
    return encoder_hidden_states_list


def load_mask (input_folder):
    np_mask_dtype = 'uint8'
    mask_np_list = []
    mask_label_list = []
    files = [
        file_name for file_name in os.listdir(input_folder) \
        if "mask" in file_name and ".npy" in file_name \
        and "_" in file_name and "Edited"  not in file_name 
    ]
    files = sorted(files, key = lambda x: int(x.split("_")[0][4:]))

    for idx, file_name in enumerate(files):
        if "mask" in file_name and ".npy" in file_name and "_" in file_name \
            and "Edited"  not in file_name:
            mask_np =  np.load(os.path.join(input_folder, file_name)).astype(np_mask_dtype) 
            mask_np_list.append(mask_np)  
            mask_label = file_name.split("_")[1][:-4]
            mask_label_list.append(mask_label)
    mask_list = []
    for mask_np in mask_np_list:
        mask = torch.from_numpy(mask_np)
        mask_list.append(mask)
    try: 
        assert torch.all(sum(mask_list)==1)
    except:
        print("please check mask")
        # plt.imsave( "out_mask.png", mask_list_edit[0]) 
        import pdb; pdb.set_trace()
    return mask_list, mask_label_list

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

def mask_union_torch(*masks):
    masks = [m.to(torch.float) for m in masks]
    res = sum(masks)>0
    return res

def load_mask_edit(input_folder):
    np_mask_dtype = 'uint8'
    mask_np_list = []
    mask_label_list = []

    files = [file_name for file_name in os.listdir(input_folder)  if "mask" in file_name and ".npy" in file_name and "_" in file_name and "Edited" in file_name and "-1" not in file_name]
    files = sorted(files, key = lambda x: int(x.split("_")[0][10:]))
    
    for idx, file_name in enumerate(files):
        if "mask" in file_name and ".npy" in file_name and "_" in file_name and "Edited" in file_name and "-1" not in file_name:
            mask_np =  np.load(os.path.join(input_folder, file_name)).astype(np_mask_dtype) 
            mask_np_list.append(mask_np)  
            mask_label = file_name.split("_")[1][:-4]
            # mask_label = mask_label.split("-")[0]
            mask_label_list.append(mask_label)
    mask_list = []
    for mask_np in mask_np_list:
        mask = torch.from_numpy(mask_np)
        mask_list.append(mask)
    try: 
        assert torch.all(sum(mask_list)==1)
    except:
        print("Make sure maskEdited is in the folder, if not, generate using the UI")
        import pdb; pdb.set_trace()
    return mask_list, mask_label_list

def save_images(images,filename, num_rows=1, offset_ratio=0.02):
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    folder = os.path.dirname(filename)
    for i, image in enumerate(images):
        pil_img = Image.fromarray(image)
        name = filename.split("/")[-1]
        name = name.split(".")[-2]+"_{}".format(i) +"."+filename.split(".")[-1]
        pil_img.save(os.path.join(folder, name))
        print("saved to ", os.path.join(folder, name))