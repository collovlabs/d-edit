export IMAGE_NAME="example1"
CUDA_VISIBLE_DEVICES=1 python main.py --name=$IMAGE_NAME \
    --dpm="sdxl" \
    --resolution=1024 \
    --num_tokens=5 \
    --load_edited_mask \
    --load_trained \
    --move_resize \
    --seed=2023 \
    --num_sampling_step=20 \
    --strength=0.5 \
    --edge_thickness=20 \
    --guidance_scale=2.8 \
    --num_imgs=2 \
    --tgt_indices_list 0 \
    --active_mask_list 2 \
    --delta_x 200 --delta_y 140  \
    --resize_list 0.5 \
    --priority_list 1  
    
