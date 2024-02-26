export IMAGE_NAME="example1"
CUDA_VISIBLE_DEVICES=1 python main.py --name=$IMAGE_NAME \
    --dpm="sd" \
    --resolution=512 \
    --num_tokens=5 \
    --load_trained \
    --move_resize \
    --seed=2023 \
    --num_sampling_step=50 \
    --strength=0.6 \
    --edge_thickness=10 \
    --guidance_scale=2 \
    --num_imgs=1 \
    --tgt_indices_list 0 \
    --active_mask_list 2 \
    --delta_x 100 --delta_y 60  \
    --resize_list 0.6 \
    --priority_list 1
