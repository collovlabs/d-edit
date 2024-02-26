# export IMAGE_NAME="example1"
# python main.py --name=$IMAGE_NAME \
#     --dpm="sd" \
#     --resolution=512 \
#     --load_trained \
#     --text \
#     --num_tokens=5 \
#     --seed=2024 \
#     --guidance_scale=7 \
#     --num_sampling_step=50 \
#     --strength=0.7 \
#     --edge_thickness=15 \
#     --num_imgs=1 \
#     --tgt_prompt="a red bag" \
#     --tgt_index=0


export IMAGE_NAME="example1"
python main.py --name=$IMAGE_NAME \
    --dpm="sd" \
    --resolution=512 \
    --load_trained \
    --text \
    --num_tokens=5 \
    --seed=2024 \
    --guidance_scale=6 \
    --num_sampling_step=50 \
    --strength=0.5 \
    --edge_thickness=15 \
    --num_imgs=2 \
    --tgt_prompt="a black bag" \
    --tgt_index=0

