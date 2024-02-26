export IMAGE_NAME="example1"
CUDA_VISIBLE_DEVICES=1 python main.py --name=$IMAGE_NAME \
    --dpm="sdxl" \
    --resolution=1024 \
    --num_tokens=5 \
    --load_edited_mask \
    --load_trained \
    --remove \
    --seed=2023 \
    --num_sampling_step=20 \
    --strength=0.3 \
    --edge_thickness=30 \
    --guidance_scale=3 \
    --num_imgs=1 \
    --tgt_index=0
