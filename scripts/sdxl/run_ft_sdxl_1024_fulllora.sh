export IMAGE_NAME="example1"
python main.py --name=$IMAGE_NAME \
    --dpm="sdxl" \
    --train_full_lora \
    --resolution=1024 \
    --embedding_learning_rate=1e-4  \
    --diffusion_model_learning_rate=1e-4 \
    --max_emb_train_steps=500  \
    --max_diffusion_train_steps=500 \
    --train_batch_size=2 \
    --gradient_accumulation_steps=5

