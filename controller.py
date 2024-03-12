import numpy as np
import torch
import math
import xformers

class DummyController:
    def __call__(self, *args):
        return args[0]
    def __init__(self):
        self.num_att_layers = 0

class GroupedCAController:
    def __init__(self, mask_list = None):
        self.mask_list = mask_list
        if self.mask_list is None:
            self.is_decom = False
        else:
            self.is_decom = True      
                  
    def mask_img_to_mask_vec(self, mask, length):
        mask_vec = torch.nn.functional.interpolate(mask.unsqueeze(0).unsqueeze(0), (length, length)).squeeze()
        mask_vec = mask_vec.flatten()
        return mask_vec
            
    def ca_forward_decom(self, q,  k_list,  v_list,  scale, place_in_unet):
            # attn [Bh, N,   d ]
            #      [8, 4096, 77]
            # q  [Bh, N,   d] [8, 4096, 40]    [8, 1024, 80]   [8, 256,160]    [8, 64, 160]
            # k  [Bh, P,   d] [8, 77  , 40]    [8, 77,   80]   [8, 77, 160]    [8, 77, 160]
            # v  [Bh, P,   d] [8, 77  , 40]    [8, 77,   80]   [8, 77, 160]    [8, 77, 160]
        N = q.shape[1]
        mask_vec_list = []
        for mask in self.mask_list:
            mask_vec = self.mask_img_to_mask_vec(mask,  int(math.sqrt(N)))   # [1,N,1]
            mask_vec = mask_vec.unsqueeze(0).unsqueeze(-1)
            mask_vec_list.append(mask_vec)
        out = 0
        for mask_vec, k, v in zip(mask_vec_list, k_list, v_list):
            sim = torch.einsum("b i d, b j d -> b i j", q, k) * scale   # [8, 4096, 20]
            attn = sim.softmax(dim=-1)                            # [Bh,N,P] [8,4096,20]
            attn = attn.masked_fill(mask_vec==0, 0)
            masked_out = torch.einsum("b i j, b j d -> b i d", attn, v) # [Bh,N,d] [8,4096,320/h] 
            # mask_vec_inf = torch.where(mask_vec>0, 0,   torch.finfo(k.dtype).min)
            # masked_out1 = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=mask_vec_inf, op=None, scale=scale)
            out += masked_out
        return out

def reshape_heads_to_batch_dim(self):
    def func(tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.num_heads
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len, dim // head_size)
    return func

def reshape_batch_dim_to_heads(self):
    def func(tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.num_heads
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
    return func

def register_attention_disentangled_control(unet, controller):
    def ca_forward(self, place_in_unet):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out        
        def forward(x, encoder_hidden_states =None, attention_mask=None):
            if isinstance(controller, DummyController):  # SA CA full
                q = self.to_q(x)
                is_cross = encoder_hidden_states is not None
                encoder_hidden_states = encoder_hidden_states if is_cross else x
                k = self.to_k(encoder_hidden_states)
                v = self.to_v(encoder_hidden_states)
                q = self.head_to_batch_dim(q)
                k = self.head_to_batch_dim(k)
                v = self.head_to_batch_dim(v)
                
                # sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale
                # attn = sim.softmax(dim=-1)
                # attn = controller(attn, is_cross, place_in_unet)
                # out = torch.einsum("b i j, b j d -> b i d", attn, v)
                out = xformers.ops.memory_efficient_attention(
                    q, k, v, attn_bias=None, op=None, scale=self.scale
                ) 
                out = self.batch_to_head_dim(out)
            else: # decom: CA+SA
                is_cross = encoder_hidden_states is not None
                assert is_cross is not None
                encoder_hidden_states_list = encoder_hidden_states if is_cross else x
                q = self.to_q(x)
                q = self.head_to_batch_dim(q) # [Bh, 4096, 320/h ] h: 8
                if is_cross:  #CA
                    k_list = []
                    v_list = []
                    assert type(encoder_hidden_states_list) is list
                    for encoder_hidden_states in encoder_hidden_states_list:
                        k = self.to_k(encoder_hidden_states)
                        k = self.head_to_batch_dim(k) # [Bh, 77,   320/h ] 
                        k_list.append(k)
                        v = self.to_v(encoder_hidden_states)
                        v = self.head_to_batch_dim(v) # [Bh, 77,   320/h ]
                        v_list.append(v)
                    out = controller.ca_forward_decom(q, k_list, v_list, self.scale, place_in_unet)   # [Bh,N,d]
                    out = self.batch_to_head_dim(out)
                else:   # SA
                    exit("decomposing SA!")
                    k = self.to_k(x)
                    v = self.to_v(x)
                    k = self.head_to_batch_dim(k) # [Bh, 77,   320/h ] 
                    v = self.head_to_batch_dim(v) # [Bh, 77,   320/h ]
                    import pdb; pdb.set_trace()
                    if  k.shape[1] <= 1024 ** 2:
                        out = controller.sa_forward(q, k, v, self.scale, place_in_unet)   # [Bh,N,d] 
                    else:
                        print("warining")
                        out = controller.sa_forward_decom(q, k, v, self.scale, place_in_unet)   # [Bh,N,d] 
                    # sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale
                    # attn = sim.softmax(dim=-1)             # [8,4096,4096]   [Bh,N,N] 
                    # out = torch.einsum("b i j, b j d -> b i d", attn, v) #  [Bh,N,d] [8,4096,320/h] 
     
                    out = self.batch_to_head_dim(out)   # [B, H, N, D]
            
            return to_out(out)

        return forward

    if controller is None:
        controller = DummyController()

    def register_recr(net_, count, place_in_unet):   
        if net_.__class__.__name__ == 'Attention' and net_.to_k.in_features == unet.ca_dim:
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = unet.named_children()

    for net in sub_nets:
        if "down" in net[0]:
            down_count = register_recr(net[1], 0, "down")#6
            cross_att_count += down_count
        elif "up" in net[0]:
            up_count = register_recr(net[1], 0, "up")    #9
            cross_att_count += up_count
        elif "mid" in net[0]:
            mid_count = register_recr(net[1], 0, "mid")  #1
            cross_att_count += mid_count
    controller.num_att_layers = cross_att_count
