# export IMAGE_NAME="example1"
# CUDA_VISIBLE_DEVICES=1 python main.py --name=$IMAGE_NAME \
#     --dpm="sd" \
#     --resolution=512 \
#     --num_tokens=5 \
#     --load_trained \
#     --load_edited_mask \
#     --remove \
#     --seed=2023 \
#     --num_sampling_step=50 \
#     --strength=0.7 \
#     --edge_thickness=10 \
#     --guidance_scale=2 \
#     --num_imgs=1 \
#     --tgt_index=0



# export IMAGE_NAME="example1"
# CUDA_VISIBLE_DEVICES=1 python main.py --name=$IMAGE_NAME \
#     --dpm="sd" \
#     --resolution=512 \
#     --num_tokens=5 \
#     --load_trained \
#     --load_edited_processed_mask \
#     --remove \
#     --seed=2024 \
#     --num_sampling_step=50 \
#     --strength=0.5 \
#     --edge_thickness=10 \
#     --guidance_scale=7 \
#     --num_imgs=1 \
#     --tgt_index=2




export IMAGE_NAME="example1"
CUDA_VISIBLE_DEVICES=1 python main.py --name=$IMAGE_NAME \
    --dpm="sd" \
    --resolution=512 \
    --num_tokens=5 \
    --load_trained \
    --load_edited_processed_mask \
    --remove \
    --seed=1 \
    --num_sampling_step=50 \
    --strength=0.6 \
    --edge_thickness=10 \
    --guidance_scale=7 \
    --num_imgs=1 \
    --tgt_index=2 



