import os
import torch
import numpy as np
import argparse
from peft import LoraConfig
from pipeline_dedit_sdxl import DEditSDXLPipeline
from pipeline_dedit_sd import DEditSDPipeline
from utils import load_image, load_mask, load_mask_edit
from utils_mask import process_mask_move_torch, process_mask_remove_torch, mask_union_torch, mask_substract_torch, create_outer_edge_mask_torch
from utils_mask import check_mask_overlap_torch, check_cover_all_torch, visualize_mask_list, get_mask_difference_torch, save_mask_list_to_npys

parser = argparse.ArgumentParser()
parser.add_argument("--name",  type=str,required=True, default=None)
parser.add_argument("--name_2", type=str,required=False, default=None)
parser.add_argument("--dpm",   type=str,required=True, default="sd")
parser.add_argument("--resolution",  type=int, default=1024)
parser.add_argument("--seed",  type=int, default=42)
parser.add_argument("--embedding_learning_rate",  type=float, default=1e-4)
parser.add_argument("--max_emb_train_steps",  type=int, default=200)
parser.add_argument("--diffusion_model_learning_rate", type=float, default=5e-5)
parser.add_argument("--max_diffusion_train_steps", type=int, default=200)
parser.add_argument("--train_batch_size",  type=int, default=1)
parser.add_argument("--gradient_accumulation_steps",  type=int, default=1)
parser.add_argument("--num_tokens",  type=int, default=1)


parser.add_argument("--load_trained", default=False, action="store_true" )
parser.add_argument("--num_sampling_steps",  type=int, default=50)
parser.add_argument("--guidance_scale", type=float, default = 3 )
parser.add_argument("--strength",  type=float, default=0.8)

parser.add_argument("--train_full_lora", default=False, action="store_true" )
parser.add_argument("--lora_rank",  type=int, default=4)
parser.add_argument("--lora_alpha",  type=int, default=4)

parser.add_argument("--prompt_auxin_list", nargs="+", type=str, default = None)
parser.add_argument("--prompt_auxin_idx_list", nargs="+", type=int, default = None)

# general editing configs
parser.add_argument("--load_edited_mask", default=False, action="store_true")
parser.add_argument("--load_edited_processed_mask", default=False, action="store_true")
parser.add_argument("--edge_thickness", type=int, default=20)
parser.add_argument("--num_imgs", type=int, default = 1 )
parser.add_argument('--active_mask_list', nargs="+", type=int)
parser.add_argument("--tgt_index",  type=int, default=None)

# recon
parser.add_argument("--recon", default=False, action="store_true" )
parser.add_argument("--recon_an_item", default=False, action="store_true" )
parser.add_argument("--recon_prompt",  type=str, default=None)

# text-based editing
parser.add_argument("--text", default=False, action="store_true")
parser.add_argument("--tgt_prompt",  type=str, default=None)

# image-based editing
parser.add_argument("--image", default=False, action="store_true" )
parser.add_argument("--src_index",  type=int, default=None)
parser.add_argument("--tgt_name",   type=str, default=None)

# mask-based move
parser.add_argument("--move_resize", default=False, action="store_true" )
parser.add_argument('--tgt_indices_list', nargs="+", type=int)
parser.add_argument("--delta_x_list", nargs="+", type=int)
parser.add_argument("--delta_y_list", nargs="+", type=int)
parser.add_argument("--priority_list", nargs="+", type=int)
parser.add_argument("--force_mask_remain", type=int, default=None)
parser.add_argument("--resize_list", nargs="+", type=float)

# remove
parser.add_argument("--remove", default=False, action="store_true" )
parser.add_argument("--load_edited_removemask", default=False, action="store_true")

args = parser.parse_args()

torch.cuda.manual_seed_all(args.seed)
torch.manual_seed(args.seed)  
base_input_folder = "."
base_output_folder  = "."

input_folder = os.path.join(base_input_folder, args.name)


mask_list, mask_label_list = load_mask(input_folder)
assert mask_list[0].shape[0] == args.resolution, "Segmentation should be done on size {}".format(args.resolution)
try:
    image_gt = load_image(os.path.join(input_folder, "img_{}.png".format(args.resolution) ), size = args.resolution)
except:
    image_gt = load_image(os.path.join(input_folder, "img_{}.jpg".format(args.resolution) ), size = args.resolution)

if args.image:
    input_folder_2 = os.path.join(base_input_folder, args.name_2)
    mask_list_2, mask_label_list_2 = load_mask(input_folder_2)
    assert mask_list_2[0].shape[0] == args.resolution, "Segmentation should be done on size {}".format(args.resolution)
    try:
        image_gt_2 = load_image(os.path.join(input_folder_2, "img_{}.png".format(args.resolution) ), size = args.resolution)
    except:
        image_gt_2 = load_image(os.path.join(input_folder_2, "img_{}.jpg".format(args.resolution) ), size = args.resolution)
    output_dir = os.path.join(base_output_folder, args.name + "_" + args.name_2)
    os.makedirs(output_dir, exist_ok = True)
else:
    output_dir = os.path.join(base_output_folder, args.name)
    os.makedirs(output_dir, exist_ok = True)

if args.dpm == "sd":
    if args.image:
        pipe = DEditSDPipeline(mask_list, mask_label_list, mask_list_2, mask_label_list_2, resolution = args.resolution, num_tokens = args.num_tokens)
    else:
        pipe = DEditSDPipeline(mask_list, mask_label_list, resolution = args.resolution, num_tokens = args.num_tokens)
        
elif args.dpm == "sdxl":
    if args.image:
        pipe = DEditSDXLPipeline(mask_list, mask_label_list, mask_list_2, mask_label_list_2, resolution = args.resolution, num_tokens = args.num_tokens)
    else:
        pipe = DEditSDXLPipeline(mask_list, mask_label_list, resolution = args.resolution, num_tokens = args.num_tokens)

else:
    raise NotImplementedError

set_string_list = pipe.set_string_list
if args.prompt_auxin_list is not None:
    for auxin_idx, auxin_prompt in zip(args.prompt_auxin_idx_list, args.prompt_auxin_list):
        set_string_list[auxin_idx] = auxin_prompt.replace("*", set_string_list[auxin_idx] )
print(set_string_list)

if args.image: 
    set_string_list_2 = pipe.set_string_list_2
    print(set_string_list_2)

if args.load_trained:
    unet_save_path = os.path.join(output_dir, "unet.pt")
    unet_state_dict = torch.load(unet_save_path)
    text_encoder1_save_path = os.path.join(output_dir, "text_encoder1.pt")
    text_encoder1_state_dict = torch.load(text_encoder1_save_path)
    if args.dpm == "sdxl":
        text_encoder2_save_path = os.path.join(output_dir, "text_encoder2.pt")
        text_encoder2_state_dict = torch.load(text_encoder2_save_path)

    if 'lora' in ''.join(unet_state_dict.keys()):
        unet_lora_config = LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                init_lora_weights="gaussian",
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            )
        pipe.unet.add_adapter(unet_lora_config) 
        
    pipe.unet.load_state_dict(unet_state_dict)
    pipe.text_encoder.load_state_dict(text_encoder1_state_dict)
    if args.dpm == "sdxl":
        pipe.text_encoder_2.load_state_dict(text_encoder2_state_dict)
else:
    if args.image:
        pipe.mask_list = [m.cuda() for m in pipe.mask_list]
        pipe.mask_list_2 = [m.cuda() for m in pipe.mask_list_2] 
        pipe.train_emb_2imgs(
            image_gt,
            image_gt_2, 
            set_string_list,
            set_string_list_2,
            gradient_accumulation_steps = args.gradient_accumulation_steps,
            embedding_learning_rate = args.embedding_learning_rate,
            max_emb_train_steps = args.max_emb_train_steps,
            train_batch_size = args.train_batch_size,
        )
        
        pipe.train_model_2imgs(
            image_gt,
            image_gt_2, 
            set_string_list,
            set_string_list_2,
            gradient_accumulation_steps = args.gradient_accumulation_steps,
            max_diffusion_train_steps = args.max_diffusion_train_steps,
            diffusion_model_learning_rate = args.diffusion_model_learning_rate ,
            train_batch_size =args.train_batch_size,
            train_full_lora = args.train_full_lora,
            lora_rank = args.lora_rank, 
            lora_alpha = args.lora_alpha
        )
        
    else:
        pipe.mask_list = [m.cuda() for m in pipe.mask_list] 
        pipe.train_emb(
            image_gt,
            set_string_list,
            gradient_accumulation_steps = args.gradient_accumulation_steps,
            embedding_learning_rate = args.embedding_learning_rate,
            max_emb_train_steps = args.max_emb_train_steps,
            train_batch_size = args.train_batch_size,
        )

        pipe.train_model(
            image_gt,
            set_string_list,
            gradient_accumulation_steps = args.gradient_accumulation_steps,
            max_diffusion_train_steps = args.max_diffusion_train_steps,
            diffusion_model_learning_rate = args.diffusion_model_learning_rate ,
            train_batch_size = args.train_batch_size,
            train_full_lora = args.train_full_lora,
            lora_rank = args.lora_rank, 
            lora_alpha = args.lora_alpha
        )

    
    unet_save_path = os.path.join(output_dir, "unet.pt")
    torch.save(pipe.unet.state_dict(),unet_save_path )
    text_encoder1_save_path = os.path.join(output_dir, "text_encoder1.pt")
    torch.save(pipe.text_encoder.state_dict(), text_encoder1_save_path)
    if args.dpm == "sdxl":
        text_encoder2_save_path = os.path.join(output_dir, "text_encoder2.pt")
        torch.save(pipe.text_encoder_2.state_dict(), text_encoder2_save_path )
    

if args.recon:
    output_dir = os.path.join(output_dir, "recon")
    os.makedirs(output_dir, exist_ok = True)
    if args.recon_an_item:
        mask_list = [torch.from_numpy(np.ones_like(mask_list[0].numpy()))]
        tgt_string = set_string_list[args.tgt_index]
        tgt_string = args.recon_prompt.replace("*", tgt_string)
        set_string_list = [tgt_string]
    print(set_string_list)
    save_path = os.path.join(output_dir, "out_recon.png")
    pipe.inference_with_mask(
        save_path,
        guidance_scale = args.guidance_scale,
        num_sampling_steps = args.num_sampling_steps,
        seed = args.seed,
        num_imgs = args.num_imgs,
        set_string_list = set_string_list,
        mask_list = mask_list
    )

if args.text:
    output_dir = os.path.join(output_dir, "text")
    os.makedirs(output_dir, exist_ok = True)
    save_path = os.path.join(output_dir, "out_text.png")
    set_string_list[args.tgt_index] = args.tgt_prompt
    mask_active = torch.zeros_like(mask_list[0])
    mask_active = mask_union_torch(mask_active, mask_list[args.tgt_index])
    
    if args.active_mask_list is not None:
        for midx in args.active_mask_list:
            mask_active = mask_union_torch(mask_active, mask_list[midx])

    if args.load_edited_mask:
        mask_list_edited, mask_label_list_edited = load_mask_edit(input_folder)
        mask_diff = get_mask_difference_torch(mask_list_edited,  mask_list)
        mask_active = mask_union_torch(mask_active, mask_diff)
        mask_list = mask_list_edited
        save_path = os.path.join(output_dir, "out_textEdited.png")
    
    mask_hard = mask_substract_torch(torch.ones_like(mask_list[0]), mask_active)
    mask_soft = create_outer_edge_mask_torch(mask_active, edge_thickness = args.edge_thickness)
    mask_hard = mask_substract_torch(mask_hard, mask_soft)

    pipe.inference_with_mask(
        save_path,
        orig_image = image_gt,
        set_string_list = set_string_list,
        guidance_scale = args.guidance_scale,
        strength = args.strength,
        num_imgs = args.num_imgs,
        mask_hard= mask_hard,
        mask_soft = mask_soft,
        mask_list = mask_list,
        seed = args.seed,
        num_sampling_steps = args.num_sampling_steps
    )

if args.remove:
    output_dir = os.path.join(output_dir, "remove")
    save_path = os.path.join(output_dir, "out_remove.png")
    os.makedirs(output_dir, exist_ok = True)
    mask_active = torch.zeros_like(mask_list[0])
    
    if args.load_edited_mask:
        mask_list_edited, _ = load_mask_edit(input_folder)
        mask_diff = get_mask_difference_torch(mask_list_edited,  mask_list)
        mask_active = mask_union_torch(mask_active, mask_diff)
        mask_list = mask_list_edited
        
    if args.load_edited_processed_mask:
        # manually edit or draw masks after removing one index, then load
        mask_list_processed, _ = load_mask_edit(output_dir)
        mask_remain = get_mask_difference_torch(mask_list_processed, mask_list)
    else:
        # generate masks after removing one index, using nearest neighbor algorithm
        mask_list_processed, mask_remain = process_mask_remove_torch(mask_list, args.tgt_index)
        save_mask_list_to_npys(output_dir, mask_list_processed, mask_label_list, name = "mask")
        visualize_mask_list(mask_list_processed, os.path.join(output_dir, "seg_removed.png"))
    check_cover_all_torch(*mask_list_processed)
    mask_active = mask_union_torch(mask_active, mask_remain)
    
    if args.active_mask_list is not None:
        for midx in args.active_mask_list:
            mask_active = mask_union_torch(mask_active, mask_list[midx])

    mask_hard = 1 - mask_active
    mask_soft = create_outer_edge_mask_torch(mask_remain, edge_thickness = args.edge_thickness)
    mask_hard = mask_substract_torch(mask_hard, mask_soft)    

    pipe.inference_with_mask(
        save_path, 
        orig_image = image_gt,
        guidance_scale = args.guidance_scale,
        strength = args.strength,
        num_imgs = args.num_imgs,
        mask_hard= mask_hard,
        mask_soft = mask_soft,
        mask_list = mask_list_processed, 
        seed = args.seed,
        num_sampling_steps = args.num_sampling_steps
    )

if args.image:
    output_dir = os.path.join(output_dir, "image")
    save_path = os.path.join(output_dir, "out_image.png")
    os.makedirs(output_dir, exist_ok = True)
    mask_active = torch.zeros_like(mask_list[0])
    
    if None not in (args.tgt_name, args.src_index, args.tgt_index):
        if args.tgt_name == args.name:
            set_string_list_tgt = set_string_list
            set_string_list_src = set_string_list_2
            image_tgt = image_gt
            if args.load_edited_mask:
                mask_list_edited, _ = load_mask_edit(input_folder)
                mask_diff = get_mask_difference_torch(mask_list_edited,  mask_list)
                mask_active = mask_union_torch(mask_active, mask_diff)
                mask_list = mask_list_edited
                save_path = os.path.join(output_dir, "out_imageEdited.png")
            mask_list_tgt = mask_list
            
        elif args.tgt_name == args.name_2:
            set_string_list_tgt = set_string_list_2
            set_string_list_src = set_string_list
            image_tgt = image_gt_2
            if args.load_edited_mask:
                mask_list_2_edited, _ = load_mask_edit(input_folder_2)
                mask_diff = get_mask_difference_torch(mask_list_2_edited,  mask_list_2)
                mask_active = mask_union_torch(mask_active, mask_diff)
                mask_list_2 = mask_list_2_edited
                save_path = os.path.join(output_dir, "out_imageEdited.png")
            mask_list_tgt = mask_list_2
        else:
            exit("tgt_name should be either name or name_2")
            
        set_string_list_tgt[args.tgt_index] = set_string_list_src[args.src_index]
        
        mask_active = mask_list_tgt[args.tgt_index]
        mask_frozen = (1-mask_active.float()).to(mask_active.device)
        mask_soft = create_outer_edge_mask_torch(mask_active.cpu(), edge_thickness = args.edge_thickness)
        mask_hard = mask_substract_torch(mask_frozen.cpu(), mask_soft.cpu())
        
        mask_list_tgt = [m.cuda() for m in mask_list_tgt]

        pipe.inference_with_mask(
            save_path,
            set_string_list = set_string_list_tgt,
            mask_list = mask_list_tgt, 
            guidance_scale = args.guidance_scale,
            num_sampling_steps = args.num_sampling_steps,
            mask_hard = mask_hard.cuda(),
            mask_soft = mask_soft.cuda(), 
            num_imgs = args.num_imgs,
            orig_image = image_tgt,
            strength = args.strength,
        )

if args.move_resize:
    output_dir = os.path.join(output_dir, "move_resize")
    os.makedirs(output_dir, exist_ok = True)
    save_path = os.path.join(output_dir, "out_moveresize.png")
    mask_active = torch.zeros_like(mask_list[0])
    
    if args.load_edited_mask:
        mask_list_edited, _ = load_mask_edit(input_folder)
        mask_diff = get_mask_difference_torch(mask_list_edited,  mask_list)
        mask_active = mask_union_torch(mask_active, mask_diff)
        mask_list = mask_list_edited
        # save_path = os.path.join(output_dir, "out_moveresizeEdited.png")
        
    if args.load_edited_processed_mask:
        mask_list_processed, _ = load_mask_edit(output_dir)
        mask_remain = get_mask_difference_torch(mask_list_processed, mask_list)
    else:
        mask_list_processed, mask_remain = process_mask_move_torch(
            mask_list,
            args.tgt_indices_list, 
            args.delta_x_list,
            args.delta_y_list, args.priority_list, 
            force_mask_remain = args.force_mask_remain,
            resize_list = args.resize_list
        )
        save_mask_list_to_npys(output_dir, mask_list_processed, mask_label_list, name = "mask")
        visualize_mask_list(mask_list_processed, os.path.join(output_dir, "seg_move_resize.png"))
    active_idxs = args.tgt_indices_list
    
    mask_active = mask_union_torch(mask_active, *[m for midx, m in enumerate(mask_list_processed) if midx in active_idxs])
    mask_active = mask_union_torch(mask_remain, mask_active)
    if args.active_mask_list is not None:
        for midx in args.active_mask_list:
            mask_active = mask_union_torch(mask_active, mask_list_processed[midx])

    mask_frozen =(1 - mask_active.float())
    mask_soft = create_outer_edge_mask_torch(mask_active, edge_thickness = args.edge_thickness)
    mask_hard = mask_substract_torch(mask_frozen, mask_soft)

    check_mask_overlap_torch(mask_hard, mask_soft)

    pipe.inference_with_mask(
        save_path,
        strength = args.strength,
        orig_image = image_gt, 
        guidance_scale = args.guidance_scale,
        num_sampling_steps =  args.num_sampling_steps,
        num_imgs = args.num_imgs,
        mask_hard= mask_hard,
        mask_soft = mask_soft,
        mask_list = mask_list_processed,
        seed = args.seed
    )
