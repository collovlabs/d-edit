import torch
from utils import import_model_class_from_model_name_or_path
from transformers import AutoTokenizer
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DDIMScheduler,
    UNet2DConditionModel,
)
from accelerate import Accelerator
from tqdm.auto import tqdm
from utils import sd_prepare_input_decom, save_images
import torch.nn.functional as F
import itertools
from peft import LoraConfig
from controller import GroupedCAController, register_attention_disentangled_control, DummyController
from utils import  image2latent, latent2image
import matplotlib.pyplot as plt
from utils_mask import  check_mask_overlap_torch

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

class DEditSDPipeline:
    def __init__(
        self,
        mask_list,
        mask_label_list, 
        mask_list_2 = None,
        mask_label_list_2 = None, 
        resolution = 1024,
        num_tokens = 1
    ):
        super().__init__()
        model_id = "runwayml/stable-diffusion-v1-5"
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer", use_fast=False)
        text_encoder_cls_one = import_model_class_from_model_name_or_path(model_id, subfolder = "text_encoder")
        self.text_encoder = text_encoder_cls_one.from_pretrained(model_id, subfolder="text_encoder" ).to(device)
        
        self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
        self.unet.ca_dim = 768
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
        self.scheduler = DDPMScheduler.from_pretrained(model_id , subfolder="scheduler")
        self.scheduler = DDIMScheduler(
                            beta_start=0.00085, 
                            beta_end=0.012, 
                            beta_schedule="scaled_linear", 
                            clip_sample=False, 
                            set_alpha_to_one=True,
                            rescale_betas_zero_snr = False,
                        )
        self.mixed_precision = "fp16"
        self.resolution = resolution
        self.num_tokens = num_tokens
        
        self.mask_list = mask_list
        self.mask_label_list = mask_label_list
        notation_token_list = [phrase.split(" ")[-1] for phrase in mask_label_list]
        placeholder_token_list = ["#"+word+"{}".format(widx) for widx, word in enumerate(notation_token_list)]
        self.set_string_list, placeholder_token_ids = self.add_tokens(placeholder_token_list)
        self.min_added_id = min(placeholder_token_ids)
        self.max_added_id = max(placeholder_token_ids)

        if mask_list_2 is not None:
            self.mask_list_2 = mask_list_2
            self.mask_label_list_2 = mask_label_list_2
            notation_token_list_2  = [phrase.split(" ")[-1] for phrase in mask_label_list_2]
            
            placeholder_token_list_2 = ["$"+word+"{}".format(widx) for widx, word in enumerate(notation_token_list_2)]
            self.set_string_list_2, placeholder_token_ids_2 = self.add_tokens(placeholder_token_list_2)
            self.max_added_id = max(placeholder_token_ids_2)

    def add_tokens_text_encoder_random_init(self, placeholder_token, num_tokens=1):
        # Add the placeholder token in tokenizer
        placeholder_tokens = [placeholder_token]
        # add dummy tokens for multi-vector
        additional_tokens = []
        for i in range(1, num_tokens):
            additional_tokens.append(f"{placeholder_token}_{i}")
        placeholder_tokens += additional_tokens
        num_added_tokens = self.tokenizer.add_tokens(placeholder_tokens) # 49408

        if num_added_tokens != num_tokens:
            raise ValueError(
                f"The tokenizer already contains the token {placeholder_token}. Please pass a different"
                " `placeholder_token` that is not already in the tokenizer."
            )
        placeholder_token_ids = self.tokenizer.convert_tokens_to_ids(placeholder_tokens)
        
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))
        token_embeds = self.text_encoder.get_input_embeddings().weight.data
        std, mean = torch.std_mean(token_embeds)
        with torch.no_grad():
            for token_id in placeholder_token_ids:         
                token_embeds[token_id] = torch.randn_like(token_embeds[token_id])*std + mean

        set_string = " ".join(self.tokenizer.convert_ids_to_tokens(placeholder_token_ids))

        return set_string, placeholder_token_ids

    def add_tokens(self, placeholder_token_list):
        set_string_list = []
        placeholder_token_ids_list = []
        for str_idx in range(len(placeholder_token_list)):
            placeholder_token = placeholder_token_list[str_idx]
            set_string, placeholder_token_ids = self.add_tokens_text_encoder_random_init(placeholder_token,  num_tokens=self.num_tokens)
            set_string_list.append(set_string)
            placeholder_token_ids_list.append(placeholder_token_ids)
        placeholder_token_ids = list(itertools.chain(*placeholder_token_ids_list)) 
        return set_string_list, placeholder_token_ids

    def train_emb(
        self, 
        image_gt,
        set_string_list,
        gradient_accumulation_steps = 5,
        embedding_learning_rate = 1e-4,
        max_emb_train_steps = 100,
        train_batch_size = 1,
    ):
        decom_controller =  GroupedCAController(mask_list = self.mask_list)
        register_attention_disentangled_control(self.unet, decom_controller)
        
        accelerator = Accelerator(mixed_precision=self.mixed_precision, gradient_accumulation_steps=gradient_accumulation_steps)
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)

        self.text_encoder.requires_grad_(True)

        self.text_encoder.text_model.encoder.requires_grad_(False)
        self.text_encoder.text_model.final_layer_norm.requires_grad_(False)
        self.text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)
        
        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        self.unet.to(device, dtype=weight_dtype)
        self.vae.to(device, dtype=weight_dtype)

        trainable_embmat_list_1 = [param for param in self.text_encoder.get_input_embeddings().parameters()]
        optimizer = torch.optim.AdamW(trainable_embmat_list_1, lr=embedding_learning_rate)
        
        self.text_encoder, optimizer = accelerator.prepare(self.text_encoder, optimizer)

        orig_embeds_params_1 = accelerator.unwrap_model(self.text_encoder).get_input_embeddings().weight.data.clone()

        self.text_encoder.train()
        
        effective_emb_train_steps = max_emb_train_steps//gradient_accumulation_steps
        
        if accelerator.is_main_process:
            accelerator.init_trackers("DEdit EmbSteps", config={
                    "embedding_learning_rate": embedding_learning_rate,
                    "text_embedding_optimization_steps": effective_emb_train_steps,
                })
        global_step = 0
        noise_scheduler = self.scheduler
        progress_bar = tqdm(range(0, effective_emb_train_steps), initial = global_step, desc="EmbSteps")
        latents0 = image2latent(image_gt, vae = self.vae, dtype = weight_dtype)
        latents0 = latents0.repeat(train_batch_size, 1, 1, 1)
        
        for _ in range(max_emb_train_steps):
            with accelerator.accumulate(self.text_encoder):
                latents = latents0.clone().detach()
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                encoder_hidden_states_list = sd_prepare_input_decom(
                    set_string_list, 
                    self.tokenizer, 
                    self.text_encoder, 
                    length = 40,
                    bsz = train_batch_size, 
                    weight_dtype = weight_dtype
                )

                model_pred = self.unet(
                    noisy_latents,
                    timesteps, 
                    encoder_hidden_states = encoder_hidden_states_list,
                ).sample

                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                index_no_updates = torch.ones((len(self.tokenizer),), dtype=torch.bool)
                index_no_updates[self.min_added_id : self.max_added_id + 1] = False
                with torch.no_grad():
                    accelerator.unwrap_model(self.text_encoder).get_input_embeddings().weight[
                        index_no_updates] = orig_embeds_params_1[index_no_updates]

            logs = {"loss": loss.detach().item(), "lr": embedding_learning_rate}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
            if global_step >= max_emb_train_steps:
                break

        accelerator.wait_for_everyone()
        accelerator.end_training()
        self.text_encoder = accelerator.unwrap_model(self.text_encoder).to(dtype = weight_dtype)

    def train_model(
        self, 
        image_gt, 
        set_string_list,
        gradient_accumulation_steps = 5,
        max_diffusion_train_steps = 100,
        diffusion_model_learning_rate = 1e-5,
        train_batch_size = 1,
        train_full_lora = False,
        lora_rank = 4,
        lora_alpha = 4
    ):
        self.unet = UNet2DConditionModel.from_pretrained(self.model_id, subfolder="unet").to(device)
        self.unet.ca_dim = 768
        decom_controller =  GroupedCAController(mask_list = self.mask_list)
        register_attention_disentangled_control(self.unet, decom_controller)
        
        mixed_precision = "fp16"
        accelerator = Accelerator(gradient_accumulation_steps = gradient_accumulation_steps, mixed_precision = mixed_precision)
        
        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        
        self.vae.requires_grad_(False)
        self.vae.to(device, dtype=weight_dtype)
        
        self.unet.requires_grad_(False)
        self.unet.train()
        
        self.text_encoder.requires_grad_(False)
        
        if not train_full_lora:
            trainable_params_list = []
            for _, module in self.unet.named_modules():
                module_name = type(module).__name__
                if module_name == "Attention":
                    if module.to_k.in_features == self.unet.ca_dim: # this is cross attention:
                        module.to_k.weight.requires_grad = True
                        trainable_params_list.append(module.to_k.weight)
                        if module.to_k.bias is not None:
                            module.to_k.bias.requires_grad = True
                            trainable_params_list.append(module.to_k.bias)                 
                        module.to_v.weight.requires_grad = True
                        trainable_params_list.append(module.to_v.weight)
                        if module.to_v.bias is not None:
                            module.to_v.bias.requires_grad = True
                            trainable_params_list.append(module.to_v.bias)
                        module.to_q.weight.requires_grad = True
                        trainable_params_list.append(module.to_q.weight)
                        if module.to_q.bias is not None:
                            module.to_q.bias.requires_grad = True
                            trainable_params_list.append(module.to_q.bias)
        else:
            unet_lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                init_lora_weights="gaussian",
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            )
            self.unet.add_adapter(unet_lora_config) 
            print("training full parameters using lora!")
            trainable_params_list = list(filter(lambda p: p.requires_grad, self.unet.parameters()))
            
        self.text_encoder.to(device, dtype=weight_dtype)
   
        optimizer = torch.optim.AdamW(trainable_params_list, lr=diffusion_model_learning_rate)                                 
        self.unet, optimizer = accelerator.prepare(self.unet, optimizer)
        psum2 = sum(p.numel() for p in trainable_params_list)

        effective_diffusion_train_steps = max_diffusion_train_steps // gradient_accumulation_steps
        if accelerator.is_main_process:
            accelerator.init_trackers("textual_inversion", config={
                    "diffusion_model_learning_rate": diffusion_model_learning_rate,
                    "diffusion_model_optimization_steps": effective_diffusion_train_steps,
                })
    
        global_step = 0
        progress_bar = tqdm( range(0, effective_diffusion_train_steps),initial=global_step, desc="ModelSteps")

        noise_scheduler = DDPMScheduler.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0" , subfolder="scheduler")
        
        latents0 = image2latent(image_gt, vae = self.vae, dtype=weight_dtype)
        latents0 = latents0.repeat(train_batch_size, 1, 1, 1)

        with torch.no_grad():
            encoder_hidden_states_list = sd_prepare_input_decom(
                set_string_list,
                self.tokenizer,
                self.text_encoder,
                length = 40, 
                bsz = train_batch_size,
                weight_dtype = weight_dtype
            )

        for _ in range(max_diffusion_train_steps):
            with accelerator.accumulate(self.unet):
                latents = latents0.clone().detach()
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                model_pred = self.unet(
                    noisy_latents,
                    timesteps, 
                    encoder_hidden_states=encoder_hidden_states_list,
                ).sample
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            logs = {"loss": loss.detach().item(), "lr": diffusion_model_learning_rate}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
            if global_step >=max_diffusion_train_steps:
                break
        accelerator.wait_for_everyone()
        accelerator.end_training()
        self.unet = accelerator.unwrap_model(self.unet).to(dtype = weight_dtype)

    def train_emb_2imgs(
        self,
        image_gt_1,
        image_gt_2, 
        set_string_list_1,
        set_string_list_2,
        gradient_accumulation_steps = 5,
        embedding_learning_rate = 1e-4,
        max_emb_train_steps = 100,
        train_batch_size = 1,
    ):
        decom_controller_1 = GroupedCAController(mask_list = self.mask_list)
        decom_controller_2 = GroupedCAController(mask_list = self.mask_list_2)
        accelerator = Accelerator(mixed_precision=self.mixed_precision, gradient_accumulation_steps=gradient_accumulation_steps)
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)

        self.text_encoder.requires_grad_(True)
        
        self.text_encoder.text_model.encoder.requires_grad_(False)
        self.text_encoder.text_model.final_layer_norm.requires_grad_(False)
        self.text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)
        

        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        self.unet.to(device, dtype=weight_dtype)
        self.vae.to(device, dtype=weight_dtype)

        
        trainable_embmat_list_1 = [param for param in self.text_encoder.get_input_embeddings().parameters()]

        optimizer = torch.optim.AdamW(trainable_embmat_list_1, lr=embedding_learning_rate)
        self.text_encoder, optimizer= accelerator.prepare(self.text_encoder,  optimizer)  ###
        orig_embeds_params_1 = accelerator.unwrap_model(self.text_encoder)  .get_input_embeddings().weight.data.clone()

        self.text_encoder.train()

        effective_emb_train_steps = max_emb_train_steps//gradient_accumulation_steps
        
        if accelerator.is_main_process:
            accelerator.init_trackers("EmbFt", config={
                    "embedding_learning_rate": embedding_learning_rate,
                    "text_embedding_optimization_steps": effective_emb_train_steps,
                })
        
        global_step = 0
        
        noise_scheduler = DDPMScheduler.from_pretrained(self.model_id , subfolder="scheduler")
        progress_bar = tqdm(range(0, effective_emb_train_steps),initial=global_step,desc="EmbSteps")
        latents0_1 = image2latent(image_gt_1, vae = self.vae, dtype=weight_dtype)
        latents0_1 = latents0_1.repeat(train_batch_size,1,1,1)
        
        latents0_2 = image2latent(image_gt_2, vae = self.vae, dtype=weight_dtype)
        latents0_2 = latents0_2.repeat(train_batch_size,1,1,1)
        
        for step in range(max_emb_train_steps):
            with accelerator.accumulate(self.text_encoder):
                latents_1 = latents0_1.clone().detach()
                noise_1 = torch.randn_like(latents_1)
                
                latents_2 = latents0_2.clone().detach()
                noise_2 = torch.randn_like(latents_2)
                
                bsz = latents_1.shape[0]

                timesteps_1 = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents_1.device)
                timesteps_1 = timesteps_1.long()
                noisy_latents_1 = noise_scheduler.add_noise(latents_1, noise_1, timesteps_1)
                
                timesteps_2 = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents_2.device)
                timesteps_2 = timesteps_2.long()
                noisy_latents_2 = noise_scheduler.add_noise(latents_2, noise_2, timesteps_2)  
                
                register_attention_disentangled_control(self.unet, decom_controller_1)
                encoder_hidden_states_list_1 = sd_prepare_input_decom(
                    set_string_list_1, 
                    self.tokenizer,
                    self.text_encoder,
                    length = 40,
                    bsz = train_batch_size,
                    weight_dtype = weight_dtype
                )
                
                model_pred_1 = self.unet(
                    noisy_latents_1,
                    timesteps_1, 
                    encoder_hidden_states=encoder_hidden_states_list_1,
                ).sample

                register_attention_disentangled_control(self.unet, decom_controller_2)
                # import pdb; pdb.set_trace()
                encoder_hidden_states_list_2= sd_prepare_input_decom(
                    set_string_list_2, 
                    self.tokenizer,
                    self.text_encoder,
                    length = 40,
                    bsz = train_batch_size,
                    weight_dtype = weight_dtype
                )
                
                model_pred_2 = self.unet(
                    noisy_latents_2,
                    timesteps_2, 
                    encoder_hidden_states = encoder_hidden_states_list_2,
                ).sample

                loss_1 = F.mse_loss(model_pred_1.float(), noise_1.float(), reduction="mean") /2
                loss_2 = F.mse_loss(model_pred_2.float(), noise_2.float(), reduction="mean") /2
                loss = loss_1 + loss_2
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                index_no_updates = torch.ones((len(self.tokenizer),), dtype=torch.bool)
                index_no_updates[self.min_added_id : self.max_added_id + 1] = False
                with torch.no_grad():
                    accelerator.unwrap_model(self.text_encoder).get_input_embeddings().weight[
                        index_no_updates] = orig_embeds_params_1[index_no_updates]
                
            logs = {"loss": loss.detach().item(), "lr": embedding_learning_rate}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
            if global_step >= max_emb_train_steps:
                break
        accelerator.wait_for_everyone()
        accelerator.end_training()
        self.text_encoder = accelerator.unwrap_model(self.text_encoder)  .to(dtype = weight_dtype)

    def train_model_2imgs(
        self,
        image_gt_1,
        image_gt_2,
        set_string_list_1,
        set_string_list_2,
        gradient_accumulation_steps = 5,
        max_diffusion_train_steps = 100,
        diffusion_model_learning_rate = 1e-5,
        train_batch_size = 1,
        train_full_lora = False,
        lora_rank = 4,
        lora_alpha = 4
    ):
        self.unet = UNet2DConditionModel.from_pretrained(self.model_id, subfolder="unet").to(device)
        self.unet.ca_dim = 768
        decom_controller_1 = GroupedCAController(mask_list = self.mask_list)
        decom_controller_2 = GroupedCAController(mask_list = self.mask_list_2)
        
        mixed_precision = "fp16"
        accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps,mixed_precision=mixed_precision)
        
        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        
        
        self.vae.requires_grad_(False)
        self.vae.to(device, dtype=weight_dtype)
        self.unet.requires_grad_(False)
        self.unet.train()
        
        self.text_encoder.requires_grad_(False)

        if not train_full_lora:
            trainable_params_list = []
            for name, module in self.unet.named_modules():
                module_name = type(module).__name__
                if module_name == "Attention":
                    if module.to_k.in_features == self.unet.ca_dim: # this is cross attention:
                        module.to_k.weight.requires_grad = True
                        trainable_params_list.append(module.to_k.weight)
                        if module.to_k.bias is not None:
                            module.to_k.bias.requires_grad = True
                            trainable_params_list.append(module.to_k.bias)
                            
                        module.to_v.weight.requires_grad = True
                        trainable_params_list.append(module.to_v.weight)
                        if module.to_v.bias is not None:
                            module.to_v.bias.requires_grad = True
                            trainable_params_list.append(module.to_v.bias)
                        module.to_q.weight.requires_grad = True
                        trainable_params_list.append(module.to_q.weight)
                        if module.to_q.bias is not None:
                            module.to_q.bias.requires_grad = True
                            trainable_params_list.append(module.to_q.bias)
        else:
            unet_lora_config = LoraConfig(
                r = lora_rank,
                lora_alpha = lora_alpha,
                init_lora_weights="gaussian",
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            )
            self.unet.add_adapter(unet_lora_config) 
            print("training full parameters using lora!")
            trainable_params_list = list(filter(lambda p: p.requires_grad, self.unet.parameters()))
            
        self.text_encoder.to(device, dtype=weight_dtype)    
        optimizer = torch.optim.AdamW(trainable_params_list, lr=diffusion_model_learning_rate)                                 
        self.unet, optimizer = accelerator.prepare(self.unet, optimizer)
        psum2 = sum(p.numel() for p in trainable_params_list)

        effective_diffusion_train_steps = max_diffusion_train_steps // gradient_accumulation_steps
        if accelerator.is_main_process:
            accelerator.init_trackers("ModelFt", config={
                    "diffusion_model_learning_rate": diffusion_model_learning_rate,
                    "diffusion_model_optimization_steps": effective_diffusion_train_steps,
                })
    
        global_step = 0
        progress_bar = tqdm(range(0, effective_diffusion_train_steps),initial=global_step, desc="ModelSteps")
        noise_scheduler = DDPMScheduler.from_pretrained(self.model_id, subfolder="scheduler")
        
        latents0_1 = image2latent(image_gt_1, vae = self.vae, dtype=weight_dtype)
        latents0_1 = latents0_1.repeat(train_batch_size, 1, 1, 1)

        latents0_2 = image2latent(image_gt_2, vae = self.vae, dtype=weight_dtype)
        latents0_2 = latents0_2.repeat(train_batch_size,1, 1, 1)
        
        with torch.no_grad():
            encoder_hidden_states_list_1 = sd_prepare_input_decom(
                set_string_list_1,
                self.tokenizer,
                self.text_encoder,
                length = 40, 
                bsz = train_batch_size, 
                weight_dtype = weight_dtype
            )
            encoder_hidden_states_list_2 = sd_prepare_input_decom(
                set_string_list_2,
                self.tokenizer,
                self.text_encoder,
                length = 40,
                bsz = train_batch_size,
                weight_dtype = weight_dtype
            )
            
        for _ in range(max_diffusion_train_steps):
            with accelerator.accumulate(self.unet):
                latents_1 = latents0_1.clone().detach()
                noise_1 = torch.randn_like(latents_1)
                bsz = latents_1.shape[0]
                timesteps_1 = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents_1.device)
                timesteps_1 = timesteps_1.long()
                noisy_latents_1 = noise_scheduler.add_noise(latents_1, noise_1, timesteps_1)
                
                latents_2 = latents0_2.clone().detach()
                noise_2 = torch.randn_like(latents_2)
                bsz = latents_2.shape[0]
                timesteps_2 = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents_2.device)
                timesteps_2 = timesteps_2.long()
                noisy_latents_2 = noise_scheduler.add_noise(latents_2, noise_2, timesteps_2)
                
                register_attention_disentangled_control(self.unet, decom_controller_1)
                model_pred_1 = self.unet(
                    noisy_latents_1,
                    timesteps_1, 
                    encoder_hidden_states = encoder_hidden_states_list_1,
                ).sample
                
                register_attention_disentangled_control(self.unet, decom_controller_2)
                model_pred_2 = self.unet(
                    noisy_latents_2,
                    timesteps_2, 
                    encoder_hidden_states = encoder_hidden_states_list_2,
                ).sample
                
                loss_1 = F.mse_loss(model_pred_1.float(), noise_1.float(), reduction="mean")
                loss_2 = F.mse_loss(model_pred_2.float(), noise_2.float(), reduction="mean")
                loss = loss_1 + loss_2
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                

            logs = {"loss": loss.detach().item(), "lr": diffusion_model_learning_rate}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
            if global_step >=max_diffusion_train_steps:
                break
        accelerator.wait_for_everyone()
        accelerator.end_training()
        self.unet = accelerator.unwrap_model(self.unet).to(dtype = weight_dtype)

    @torch.no_grad()
    def backward_zT_to_z0_euler_decom(
        self,
        zT,
        cond_emb_list,
        uncond_emb=None,
        guidance_scale = 1,
        num_sampling_steps = 20,
        cond_controller = None,
        uncond_controller = None,
        mask_hard = None, 
        mask_soft = None,
        orig_image = None,
        return_intermediate = False,
        strength = 1
    ):
        latent_cur = zT
        if uncond_emb is None:
            uncond_emb = torch.zeros(zT.shape[0], 77, self.unet.ca_dim).to(dtype = zT.dtype, device = zT.device)

        if mask_soft is not None:
            init_latents_orig = image2latent(orig_image, self.vae, dtype=self.vae.dtype)
            length = init_latents_orig.shape[-1]
            noise = torch.randn_like(init_latents_orig)
            mask_soft = torch.nn.functional.interpolate(mask_soft.float().unsqueeze(0).unsqueeze(0), (length, length)).to(self.vae.dtype) ###
        
        if mask_hard is not None:
            init_latents_orig = image2latent(orig_image, self.vae, dtype=self.vae.dtype)
            length = init_latents_orig.shape[-1]
            noise = torch.randn_like(init_latents_orig) 
            mask_hard = torch.nn.functional.interpolate(mask_hard.float().unsqueeze(0).unsqueeze(0), (length, length)).to(self.vae.dtype) ###

        intermediate_list = [latent_cur.detach()]
        for i in tqdm(range(num_sampling_steps)):
            t = self.scheduler.timesteps[i]
            latent_input = self.scheduler.scale_model_input(latent_cur, t)
            
            register_attention_disentangled_control(self.unet, uncond_controller)
            noise_pred_uncond = self.unet(
                                    latent_input, 
                                    t,
                                    encoder_hidden_states=uncond_emb,
                                ).sample
            
            register_attention_disentangled_control(self.unet, cond_controller)
            noise_pred_cond = self.unet(
                                latent_input,
                                t,
                                encoder_hidden_states=cond_emb_list,
                              ).sample
            
            noise_pred =  noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            latent_cur = self.scheduler.step(noise_pred, t, latent_cur, generator = None, return_dict=False)[0]

            if return_intermediate is True:
                intermediate_list.append(latent_cur)

            if mask_hard is not None and mask_soft is not None and i <= strength *num_sampling_steps:
                init_latents_proper = self.scheduler.add_noise(init_latents_orig, noise, torch.tensor([t]))
                mask = mask_soft.to(latent_cur.device, latent_cur.dtype) + mask_hard.to(latent_cur.device, latent_cur.dtype)
                latent_cur = (init_latents_proper * mask) + (latent_cur * (1 - mask))

            elif mask_hard is not None and mask_soft is not None and i > strength *num_sampling_steps:
                init_latents_proper = self.scheduler.add_noise(init_latents_orig, noise, torch.tensor([t]))
                mask = mask_hard.to(latent_cur.device, latent_cur.dtype)
                latent_cur = (init_latents_proper * mask) + (latent_cur * (1 - mask))

            elif mask_hard is None and mask_soft is not None and i <= strength *num_sampling_steps:
                init_latents_proper = self.scheduler.add_noise(init_latents_orig, noise, torch.tensor([t]))
                mask = mask_soft.to(latent_cur.device, latent_cur.dtype)
                latent_cur = (init_latents_proper * mask) + (latent_cur * (1 - mask))
                
            elif mask_hard is None and mask_soft is not None and i > strength *num_sampling_steps:
                pass
            
            elif mask_hard is not None and mask_soft is None:
                init_latents_proper = self.scheduler.add_noise(init_latents_orig, noise, torch.tensor([t]))
                mask = mask_hard.to(latent_cur.dtype)
                latent_cur = (init_latents_proper * mask) + (latent_cur * (1 - mask))
                
            else: # hard and soft are both none
                pass
            
        if return_intermediate is True:
            return latent_cur, intermediate_list
        else:
            return latent_cur
        
    @torch.no_grad()
    def sampling(
        self,
        set_string_list, 
        cond_controller = None, 
        uncond_controller = None,
        guidance_scale = 7,
        num_sampling_steps = 20,
        mask_hard = None,
        mask_soft = None,
        orig_image = None,
        strength = 1.,
        num_imgs = 1,
        normal_token_id_list = [],
        seed = 1
    ):
        weight_dtype = torch.float16
        self.scheduler.set_timesteps(num_sampling_steps)
        self.unet.to(device, dtype=weight_dtype)
        self.vae.to(device, dtype=weight_dtype)
        self.text_encoder.to(device, dtype=weight_dtype)
        
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        zT = torch.randn(num_imgs, 4, self.resolution//vae_scale_factor,self.resolution//vae_scale_factor).to(device,dtype=weight_dtype)
        zT = zT * self.scheduler.init_noise_sigma   
        
        cond_emb_list = sd_prepare_input_decom(
                            set_string_list, 
                            self.tokenizer,
                            self.text_encoder,
                            length = 40,
                            bsz = num_imgs,
                            weight_dtype = weight_dtype,
                            normal_token_id_list = normal_token_id_list
                        )

        z0 = self.backward_zT_to_z0_euler_decom(zT, cond_emb_list,
                guidance_scale = guidance_scale, num_sampling_steps = num_sampling_steps,
                cond_controller = cond_controller, uncond_controller = uncond_controller, 
                mask_hard = mask_hard, mask_soft = mask_soft, orig_image = orig_image, strength = strength
             )
        x0 = latent2image(z0, vae = self.vae)
        return x0
    
    @torch.no_grad()                 
    def inference_with_mask(
        self,
        save_path,
        guidance_scale = 3,
        num_sampling_steps = 50,
        strength = 1,
        mask_soft = None,
        mask_hard= None,
        orig_image=None,
        mask_list = None,
        num_imgs = 1,
        seed = 1,
        set_string_list = None
    ):
        if mask_list is not None:
            mask_list = [m.to(device) for m in mask_list]
        else:
            mask_list = self.mask_list
        if set_string_list is not None:
            self.set_string_list = set_string_list

        if mask_hard is not None and mask_soft is not None:
            check_mask_overlap_torch(mask_hard, mask_soft)
        null_controller = DummyController()
        decom_controller = GroupedCAController(mask_list = mask_list)

        x0 = self.sampling(
            self.set_string_list, 
            guidance_scale = guidance_scale,
            num_sampling_steps = num_sampling_steps,
            strength = strength,
            cond_controller = decom_controller,
            uncond_controller = null_controller, 
            mask_soft = mask_soft,
            mask_hard = mask_hard,
            orig_image = orig_image,
            num_imgs = num_imgs,
            seed = seed
        )
        save_images(x0, save_path)
