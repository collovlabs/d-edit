export IMAGE_NAME="example1"
export IMAGE_NAME_2="example2"
python main.py --name=$IMAGE_NAME \
    --name_2=$IMAGE_NAME_2 \
    --dpm="sdxl" \
    --image \
    --load_trained \
    --resolution=1024 \
    --guidance_scale=2.8 \
    --num_imgs=2 \
    --seed=2023 \
    --strength=0.5 \
    --edge_thickness=20 \
    --src_index=2   --tgt_index=0  \
    --tgt_name=$IMAGE_NAME