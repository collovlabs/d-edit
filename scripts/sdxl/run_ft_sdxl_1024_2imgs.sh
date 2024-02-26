export IMAGE_NAME="example1"
export IMAGE_NAME_2="example2"
python main.py --name=$IMAGE_NAME \
    --dpm="sdxl" \
    --image \
    --name_2=$IMAGE_NAME_2 \
    --resolution=1024 \
    --embedding_learning_rate=1e-4  \
    --diffusion_model_learning_rate=5e-5 \
    --max_emb_train_steps=500  \
    --max_diffusion_train_steps=500 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=5

