export IMAGE_NAME="example1"
python main.py --name=$IMAGE_NAME \
    --dpm="sdxl" \
    --resolution=1024 \
    --num_tokens=5 \
    --load_trained \
    --load_edited_mask \
    --text \
    --seed=23 \
    --num_sampling_step=50 \
    --strength=0.7 \
    --edge_thickness=30 \
    --num_imgs=2 \
    --tgt_prompt="a white handbag" \
    --tgt_index=0
