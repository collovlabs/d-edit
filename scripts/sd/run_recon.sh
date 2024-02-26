export IMAGE_NAME="example1"
python main.py --name=$IMAGE_NAME \
    --dpm="sd" \
    --resolution=512 \
    --num_tokens=5 \
    --load_trained \
    --recon \
    --seed=2024 \
    --guidance_scale=2 \
    --num_sampling_step=20 \
    --num_imgs=1 \
