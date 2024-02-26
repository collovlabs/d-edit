export IMAGE_NAME="example1"
export IMAGE_NAME_2="example2"
python main.py --name=$IMAGE_NAME \
    --name_2=$IMAGE_NAME_2 \
    --dpm="sd" \
    --resolution=512 \
    --image \
    --load_trained \
    --guidance_scale=2 \
    --num_imgs=2 \
    --seed=2024 \
    --strength=0.5 \
    --edge_thickness=10 \
    --src_index=1   --tgt_index=0  \
    --tgt_name=$IMAGE_NAME