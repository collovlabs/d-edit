export IMAGE_NAME="example1"
python main.py --name=$IMAGE_NAME \
    --dpm="sdxl" \
    --resolution=1024 \
    --num_tokens=5 \
    --load_trained \
    --recon \
    --seed=2024 \
    --guidance_scale=3 \
    --num_sampling_step=20 \
    --num_imgs=2 \
