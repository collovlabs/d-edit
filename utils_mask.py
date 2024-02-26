import os
import numpy as np
from matplotlib import cm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import torch
from utils import myroll2d

def create_outer_edge_mask_torch(mask, edge_thickness = 20):
    mask_down = myroll2d(mask, edge_thickness, 0 )
    mask_edge_down = (mask_down.to(torch.float) -mask.to(torch.float))>0

    mask_up  = myroll2d(mask, -edge_thickness, 0)
    mask_edge_up = (mask_up.to(torch.float) -mask.to(torch.float))>0

    mask_left  = myroll2d(mask, 0, -edge_thickness)
    mask_edge_left = (mask_left.to(torch.float) -mask.to(torch.float))>0

    mask_right  = myroll2d(mask, 0, edge_thickness)
    mask_edge_right = (mask_right.to(torch.float) -mask.to(torch.float))>0

    mask_ur =  myroll2d(mask, -edge_thickness,edge_thickness)
    mask_edge_ur = (mask_ur.to(torch.float) -mask.to(torch.float))>0

    mask_ul =  myroll2d(mask, -edge_thickness,-edge_thickness)
    mask_edge_ul = (mask_ul.to(torch.float) -mask.to(torch.float))>0
    
    mask_dr =  myroll2d(mask, edge_thickness,edge_thickness )
    mask_edge_dr = (mask_dr.to(torch.float) -mask.to(torch.float))>0

    mask_dl =  myroll2d(mask, edge_thickness,-edge_thickness)
    mask_edge_ul = (mask_dl.to(torch.float) -mask.to(torch.float))>0

    mask_edge = mask_union_torch(mask_edge_down, mask_edge_up, mask_edge_left, mask_edge_right,
                            mask_edge_ur, mask_edge_ul, mask_edge_dr, mask_edge_ul)
    return mask_edge

def mask_substract_torch(mask1, mask2):
    return ((mask1.cpu().to(torch.float)-mask2.cpu().to(torch.float))>0).to(torch.uint8)

def check_mask_overlap_torch(*masks):
    assert torch.any(sum([m.float() for m in masks])<=1 )
    
def check_mask_overlap_numpy(*masks):
    assert np.all(sum([m.astype(float) for m in masks])<=1 )
    
def check_cover_all_torch (*masks):     
    assert torch.all(sum([m.cpu().float() for m in masks])==1)
    
def process_mask_to_follow_priority(mask_list, priority_list):
    for idx1, (m1 , p1) in enumerate(zip(mask_list, priority_list)):
        for idx2, (m2 , p2) in enumerate(zip(mask_list, priority_list)):
            if p2 > p1:
                mask_list[idx1] = ((m1.astype(float)-m2.astype(float))>0).astype(np.uint8)
    return mask_list

def mask_union(*masks):
    masks = [m.astype(float) for m in masks]
    res = sum(masks)>0
    return res.astype(np.uint8)

def mask_intersection(mask1, mask2):
    mask_uni =  mask_union(mask1, mask2)
    mask_intersec = ((mask1.astype(float)-mask2.astype(float))==0) * mask_uni
    return mask_intersec

def mask_union_torch(*masks):
    masks = [m.float() for m in masks]
    res = sum(masks)>0
    return res.to(torch.uint8)

def mask_intersection_torch(mask1, mask2):
    mask_uni =  mask_union_torch(mask1, mask2)
    mask_intersec = ((mask1.float()-mask2.float())==0) * mask_uni
    return mask_intersec.cpu().to(torch.uint8)


def visualize_mask_list(mask_list, savepath):
    mask = 0
    for midx, m in enumerate(mask_list):
        try:
            mask += m.astype(float)* midx
        except:
            mask += m.float()*midx 
    viridis = cm.get_cmap('viridis', len(mask_list))
    fig, ax = plt.subplots()
    ax.imshow( mask)

    handles = []
    label_list = []
    for idx , _ in enumerate(mask_list):
        color = viridis(idx)
        label = f"{idx}"
        handles.append(mpatches.Patch(color=color, label=label))
        label_list.append(label)
    ax.legend(handles=handles)
    plt.savefig(savepath)

def visualize_mask_list_clean(mask_list, savepath):
    mask = 0
    for midx, m in enumerate(mask_list):
        try:
            mask += m.astype(float)* midx
        except:
            mask += m.float()*midx 
    viridis = cm.get_cmap('viridis', len(mask_list))
    fig, ax = plt.subplots()
    ax.imshow( mask)

    handles = []
    label_list = []
    for idx , _ in enumerate(mask_list):
        color = viridis(idx)
        label = f"{idx}"
        handles.append(mpatches.Patch(color=color, label=label))
        label_list.append(label)
    # ax.legend(handles=handles)
    plt.savefig(savepath,  dpi=500)

   
def move_mask(mask_select, delta_x, delta_y):
    mask_edit = myroll2d(mask_select, delta_y, delta_x)
    return mask_edit

def stack_mask_with_priority (mask_list_np, priority_list, edit_idx_list):
    mask_sel = mask_union(*[mask_list_np[eid] for eid in edit_idx_list])
    for midx, mask in enumerate(mask_list_np):
        if midx not in edit_idx_list:
            if priority_list[edit_idx_list[0]] >= priority_list[midx]:
                mask = mask.astype(float) - np.logical_and(mask.astype(bool) , mask_sel.astype(bool)).astype(float)
            mask_list_np[midx] = mask.astype("uint8")
    for midx  in edit_idx_list:
        for midx_1 in edit_idx_list:
            if midx != midx_1:
                if priority_list[midx] <= priority_list[midx_1]:
                    mask = mask_list_np[midx].astype(float) - np.logical_and(mask_list_np[midx].astype(bool), mask_list_np[midx_1].astype(bool)).astype(float)
                    mask_list_np[midx] = mask.astype("uint8")    
    return mask_list_np

def process_remain_mask(mask_list, edit_idx_list = None, force_mask_remain = None):
    print("Start to process remaining mask using nearest neighbor")
    width = mask_list[0].shape[0]
    height = mask_list[0].shape[1]
    pixel_ind = np.arange( width* height)
    
    y_axis = np.arange(width)
    ymesh = np.repeat(y_axis[:,np.newaxis], height, axis = 1) #N, N
    ymesh_vec = ymesh.reshape(-1)                           #N *N
    
    x_axis = np.arange(height)
    xmesh = np.repeat(x_axis[np.newaxis, : ], width, axis = 0)
    xmesh_vec = xmesh.reshape(-1)
    
    mask_remain = (1 - sum([m.astype(float) for m in mask_list])).astype(np.uint8)
    if force_mask_remain is not None:
        mask_list[force_mask_remain] = (mask_list[force_mask_remain].astype(float) + mask_remain.astype(float)).astype(np.uint8)
    else:
        if edit_idx_list is not None:
            a = [mask_list[eidx] for eidx in edit_idx_list]
            mask_edit = mask_union(*a)
        else:
            mask_edit = np.zeros_like(mask_remain).astype(np.uint8)
        mask_feasible = (1 - mask_remain.astype(float) - mask_edit.astype(float)).astype(np.uint8)

        edge_width = 2

        mask_feasible_down  = myroll2d(mask_feasible, edge_width, 0)
        mask_edge_down = (mask_feasible_down.astype(float) -mask_feasible.astype(float))<0

        mask_feasible_up  = myroll2d(mask_feasible, -edge_width, 0)
        mask_edge_up = (mask_feasible_up.astype(float) -mask_feasible.astype(float))<0

        mask_feasible_left  = myroll2d(mask_feasible, 0, -edge_width)
        mask_edge_left = (mask_feasible_left.astype(float) -mask_feasible.astype(float))<0

        mask_feasible_right  = myroll2d(mask_feasible, 0, edge_width)
        mask_edge_right = (mask_feasible_right.astype(float) -mask_feasible.astype(float))<0

        mask_feasible_ur =  myroll2d(mask_feasible, -edge_width,edge_width)
        mask_edge_ur = (mask_feasible_ur.astype(float) -mask_feasible.astype(float))<0

        mask_feasible_ul =  myroll2d(mask_feasible, -edge_width,-edge_width )
        mask_edge_ul = (mask_feasible_ul.astype(float) -mask_feasible.astype(float))<0

        mask_feasible_dr =  myroll2d(mask_feasible, edge_width,edge_width )
        mask_edge_dr = (mask_feasible_dr.astype(float) -mask_feasible.astype(float))<0

        mask_feasible_dl =  myroll2d(mask_feasible, edge_width,-edge_width)
        mask_edge_ul = (mask_feasible_dl.astype(float) -mask_feasible.astype(float))<0
            
        mask_edge = mask_union(
            mask_edge_down, mask_edge_up, mask_edge_left, mask_edge_right, mask_edge_ur, mask_edge_ul, mask_edge_dr, mask_edge_ul
        )
        
        mask_feasible_edge = mask_intersection(mask_edge, mask_feasible)
        
        vec_mask_feasible_edge = mask_feasible_edge.reshape(-1)
        vec_mask_remain        = mask_remain.reshape(-1)

        indvec_all = np.arange(width*height)
        vec_region_partition= 0
        for mask_idx, mask in enumerate(mask_list):
            vec_region_partition += mask.reshape(-1) * mask_idx
        vec_region_partition += mask_remain.reshape(-1) * mask_idx
        # assert 0 in vec_region_partition
        
        vec_ind_remain = np.nonzero(vec_mask_remain)[0]
        vec_ind_feasible_edge = np.nonzero(vec_mask_feasible_edge)[0]    
        
        vec_x_remain = xmesh_vec[vec_ind_remain]
        vec_y_remain = ymesh_vec[vec_ind_remain]

        vec_x_feasible_edge =  xmesh_vec[vec_ind_feasible_edge]
        vec_y_feasible_edge =  ymesh_vec[vec_ind_feasible_edge]

        x_dis = vec_x_remain[:,np.newaxis] - vec_x_feasible_edge[np.newaxis,:]
        y_dis = vec_y_remain[:,np.newaxis] - vec_y_feasible_edge[np.newaxis,:]
        dis = x_dis **2 + y_dis **2
        pos = np.argmin(dis, axis = 1)
        nearest_point = vec_ind_feasible_edge[pos]   # closest point to target point
        
        nearest_region = vec_region_partition[nearest_point]
        nearest_region_set = set(nearest_region)
        if edit_idx_list is not None:
            for edit_idx in edit_idx_list:
                assert edit_idx not in nearest_region

        for midx, m in enumerate(mask_list):
            if midx in nearest_region_set:
                vec_newmask = np.zeros_like(indvec_all)
                add_ind = vec_ind_remain [np.argwhere(nearest_region==midx)]
                vec_newmask[add_ind] = 1
            
                mask_list[midx] = mask_list[midx].astype(float)+ vec_newmask.reshape( mask_list[midx].shape).astype(float)
                mask_list[midx] = mask_list[midx] > 0
    
    print("Finish processing remaining mask, if you want to edit, launch the ui")
    return mask_list, mask_remain
       
def resize_mask(mask_np, resize_ratio = 1):
    w, h = mask_np.shape[0],  mask_np.shape[1]
    resized_w, resized_h = int(w*resize_ratio),int(h*resize_ratio) 
    mask_resized = torch.nn.functional.interpolate(torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0), (resized_w, resized_h)).squeeze()

    mask = torch.zeros(w,  h)
    if w > resized_w:
        mask[:resized_w, :resized_h] = mask_resized
    else:
        assert h <= resized_h 
        mask = mask_resized[resized_w//2-w//2: resized_w//2-w//2+w, resized_h//2-h//2: resized_h//2-h//2+h]
    return mask.cpu().numpy().astype(np.uint8)

def process_mask_move_torch(
        mask_list,
        move_index_list,
        delta_x_list = None,
        delta_y_list = None, 
        edit_priority_list = None,
        force_mask_remain = None,
        resize_list = None
    ):
    mask_list_np = [m.cpu().numpy() for m in mask_list]
    priority_list = [0 for _ in range(len(mask_list_np))]
    for idx, (move_index, delta_x, delta_y, priority) in enumerate(zip(move_index_list, delta_x_list, delta_y_list, edit_priority_list)):
        priority_list[move_index] = priority
        if resize_list is not None:
            mask = resize_mask (mask_list_np[move_index], resize_list[idx])
        else:
            mask = mask_list_np[move_index]
        mask_list_np[move_index] = move_mask(mask,  delta_x = delta_x, delta_y = delta_y)
    mask_list_np = stack_mask_with_priority (mask_list_np, priority_list, move_index_list) # exists blank
    check_mask_overlap_numpy(*mask_list_np)
    mask_list_np, mask_remain = process_remain_mask(mask_list_np, move_index_list,force_mask_remain)
    mask_list = [torch.from_numpy(m).to( dtype=torch.uint8) for m in mask_list_np]
    mask_remain = torch.from_numpy(mask_remain).to(dtype=torch.uint8)
    return mask_list, mask_remain

def process_mask_remove_torch(mask_list, remove_idx):
    mask_list_np = [m.cpu().numpy() for m in mask_list]
    mask_list_np[remove_idx] = np.zeros_like(mask_list_np[0])
    mask_list_np, mask_remain = process_remain_mask(mask_list_np)
    mask_list = [torch.from_numpy(m).to(dtype=torch.uint8) for m in mask_list_np]
    mask_remain = torch.from_numpy(mask_remain).to(dtype=torch.uint8)
    return mask_list, mask_remain

def get_mask_difference_torch(mask_list1, mask_list2):
    assert len(mask_list1) == len(mask_list2)
    mask_diff = torch.zeros_like(mask_list1[0])
    for mask1 , mask2 in zip(mask_list1, mask_list2):
        diff = ((mask1.float() - mask2.float())!=0).to(torch.uint8)
        mask_diff = mask_union_torch(mask_diff, diff)
    return mask_diff  

def save_mask_list_to_npys(folder, mask_list, mask_label_list, name = "mask"):
    for midx, (mask, mask_label) in enumerate(zip(mask_list, mask_label_list)):
        np.save(os.path.join(folder, "{}{}_{}.npy".format(name, midx, mask_label)), mask)
