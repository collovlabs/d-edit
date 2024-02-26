# export IMAGE_NAME="example1"
# python main.py --name=$IMAGE_NAME \
#     --dpm="sdxl" \
#     --resolution=1024 \
#     --load_trained \
#     --recon \
#     --recon_an_item \
#     --seed=23 \
#     --guidance_scale=6 \
#     --num_sampling_step=20 \
#     --num_imgs=2  \
#     --tgt_index=0 \
#     --recon_prompt="a photo of a * handbag on a table"

export IMAGE_NAME="example1"
python main.py --name=$IMAGE_NAME \
    --dpm="sdxl" \
    --resolution=1024 \
    --load_trained \
    --recon \
    --recon_an_item \
    --seed=23 \
    --guidance_scale=6 \
    --num_sampling_step=20 \
    --num_imgs=2  \
    --tgt_index=2 \
    --recon_prompt="a photo of a * model on a chair"