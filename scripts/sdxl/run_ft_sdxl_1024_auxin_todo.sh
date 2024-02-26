export IMAGE_NAME="example1"
python main.py --name=$IMAGE_NAME \
    --dpm="sdxl" \
    --resolution=1024 \
    --train_full_lora \
    --embedding_learning_rate=1e-4  \
    --diffusion_model_learning_rate=1e-3 \
    --max_emb_train_steps=500  \
    --max_diffusion_train_steps=500 \
    --train_batch_size=2 \
    --gradient_accumulation_steps=5 \
    --prompt_auxin_idx_list 0 2 \
    --prompt_auxin_list "a photo of * handbag" "a photo of * model" 
    
export IMAGE_NAME="example1"
python main.py --name=$IMAGE_NAME \
    --dpm="sdxl" \
    --resolution=1024 \
    --train_full_lora \
    --load_trained \
    --recon \
    --seed=23 \
    --guidance_scale=7 \
    --num_sampling_step=20 \
    --num_imgs=2 \
    --prompt_auxin_idx_list 0 2 \
    --prompt_auxin_list "a photo of * handbag" "a photo of * model" 
    

