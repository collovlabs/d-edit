
import os
import copy
from PIL import Image
import matplotlib 
import numpy as np
import gradio as gr
from utils import load_mask, load_mask_edit
from utils_mask import process_mask_to_follow_priority, mask_union, visualize_mask_list_clean

LENGTH=512 #length of the square area displaying/editing images
TRANSPARENCY = 150 # transparency of the mask in display

def add_mask(mask_np_list_updated, mask_label_list):
    mask_new = np.zeros_like(mask_np_list_updated[0])
    mask_np_list_updated.append(mask_new)
    mask_label_list.append("new")
    return mask_np_list_updated, mask_label_list

def create_segmentation(mask_np_list):
    viridis = matplotlib.pyplot.get_cmap(name = 'viridis', lut = len(mask_np_list))
    segmentation = 0
    for i, m  in enumerate(mask_np_list):
        color = matplotlib.colors.to_rgb(viridis(i))
        color_mat = np.ones_like(m)                                                                            
        color_mat = np.stack([color_mat*color[0], color_mat*color[1],color_mat*color[2] ], axis = 2)
        color_mat = color_mat * m[:,:,np.newaxis]
        segmentation += color_mat
    segmentation = Image.fromarray(np.uint8(segmentation*255))
    return segmentation

def load_mask_ui(input_folder,load_edit = False):
    if not load_edit:
        mask_list, mask_label_list = load_mask(input_folder)
    else:
        mask_list, mask_label_list = load_mask_edit(input_folder) 
        
    mask_np_list = []
    for  m  in mask_list:
        mask_np_list. append( m.cpu().numpy())

    return mask_np_list, mask_label_list

def load_image_ui(input_folder, load_edit):
    try:
        try:
            try:
                image = Image.open(os.path.join(input_folder, "img_1024.png"))
            except:
                image = Image.open(os.path.join(input_folder, "img_512.png"))
        except:
            try:
                image = Image.open(os.path.join(os.path.dirname(input_folder), "img_1024.png"))
            except:
                image = Image.open(os.path.join(os.path.dirname(input_folder), "img_512.png"))
        mask_np_list, mask_label_list = load_mask_ui(input_folder, load_edit = load_edit)
        image = image.convert('RGB')
        segmentation = create_segmentation(mask_np_list)  
        return image, segmentation, mask_np_list, mask_label_list, image
    except:
        print("image folder invalid")
        return None, None, None, None, None
    
def transparent_paste_with_mask(backimg, foreimg, mask_np,transparency = 128):
    backimg_solid_np =  np.array(backimg)
    bimg = backimg.copy()
    fimg = foreimg.copy()
    fimg.putalpha(transparency)
    bimg.paste(fimg, (0,0), fimg)

    bimg_np = np.array(bimg)
    mask_np = mask_np[:,:,np.newaxis]
    try:
        new_img_np = bimg_np*mask_np + (1-mask_np)* backimg_solid_np
        return Image.fromarray(new_img_np) 
    except:
        import pdb; pdb.set_trace()

def show_segmentation(image, segmentation, flag):
    if flag is False:
        flag = True
        mask_np = np.ones([image.size[0],image.size[1]]).astype(np.uint8)
        image_edit = transparent_paste_with_mask(image, segmentation, mask_np ,transparency = TRANSPARENCY)
        return image_edit, flag
    else:
        flag = False
        return image,flag

def edit_mask_add(canvas,  image, idx, mask_np_list):
    mask_sel = mask_np_list[idx]
    mask_new = np.uint8(canvas["mask"][:, :, 0]/ 255.)
    mask_np_list_updated = []
    for midx, m  in enumerate(mask_np_list):
        if midx == idx:
            mask_np_list_updated.append(mask_union(mask_sel, mask_new))
        else:
            mask_np_list_updated.append(m)
    
    priority_list = [0 for _ in range(len(mask_np_list_updated))]
    priority_list[idx] = 1
    mask_np_list_updated = process_mask_to_follow_priority(mask_np_list_updated, priority_list)
    mask_ones = np.ones([mask_sel.shape[0], mask_sel.shape[1]]).astype(np.uint8)
    segmentation = create_segmentation(mask_np_list_updated)
    image_edit = transparent_paste_with_mask(image, segmentation, mask_ones ,transparency = TRANSPARENCY)
    return mask_np_list_updated, image_edit

def slider_release(index, image,  mask_np_list_updated, mask_label_list):
    if index > len(mask_np_list_updated):
        return image, "out of range"
    else:
        mask_np = mask_np_list_updated[index]
        mask_label = mask_label_list[index]
        segmentation = create_segmentation(mask_np_list_updated)
        new_image = transparent_paste_with_mask(image, segmentation, mask_np, transparency = TRANSPARENCY)
    return new_image, mask_label

def save_as_orig_mask(mask_np_list_updated, mask_label_list, input_folder):
    try: 
        assert np.all(sum(mask_np_list_updated)==1)
    except:
        print("please check mask")
        # plt.imsave( "out_mask.png", mask_list_edit[0]) 
        import pdb; pdb.set_trace()
        
    for midx, (mask, mask_label) in enumerate(zip(mask_np_list_updated, mask_label_list)):
        # np.save(os.path.join(input_folder, "maskEDIT{}_{}.npy".format(midx, mask_label)),mask )
        np.save(os.path.join(input_folder, "mask{}_{}.npy".format(midx, mask_label)),mask )
    savepath = os.path.join(input_folder, "seg_current.png")
    visualize_mask_list_clean(mask_np_list_updated, savepath)
    
def save_as_edit_mask(mask_np_list_updated, mask_label_list, input_folder):
    try: 
        assert np.all(sum(mask_np_list_updated)==1)
    except:
        print("please check mask")
        # plt.imsave( "out_mask.png", mask_list_edit[0]) 
        import pdb; pdb.set_trace()
    for midx, (mask, mask_label) in enumerate(zip(mask_np_list_updated, mask_label_list)):
        np.save(os.path.join(input_folder, "maskEdited{}_{}.npy".format(midx, mask_label)), mask)
    savepath = os.path.join(input_folder, "seg_edited.png")
    visualize_mask_list_clean(mask_np_list_updated, savepath)
  
with gr.Blocks() as demo:
    with gr.Row():
        gr.Markdown("""# Mask Editing""")
    
    # UI components for editing real images
    with gr.Tab(label="Edit mask"):
        image = gr.State() # store mask
        image_loaded = gr.State()
        segmentation    = gr.State()

        mask_np_list    = gr.State([])
        mask_label_list = gr.State([])
        mask_np_list_updated = gr.State([])
        true    = gr.State(True)
        false    = gr.State(False)

        with gr.Row():
            with gr.Column():
                canvas       = gr.Image( value =None, type="numpy", tool="sketch", label="Draw Mask", show_label=True, height=LENGTH, width=LENGTH, interactive=True)
                input_folder = gr.Textbox(value="example1", label="input folder", interactive= True, )
                text_button  = gr.Button("Load original masks")
                text_button.click(load_image_ui, 
                        [input_folder, false] , 
                        [image_loaded, segmentation,  mask_np_list, mask_label_list, canvas] )
            
                load_edit_button = gr.Button("Load edited masks")    
                load_edit_button.click(load_image_ui, 
                        [input_folder, true] , 
                        [image_loaded, segmentation,  mask_np_list, mask_label_list, canvas] )
                
                show_segment = gr.Checkbox(label = "Show Segmentation")
       
                flag = gr.State(False)
                show_segment.select(show_segmentation,
                                    [image_loaded, segmentation, flag], 
                                    [canvas, flag])
       
            mask_np_list_updated = copy.deepcopy(mask_np_list)

            with gr.Column():
                gr.Markdown("""<p style="text-align: center; font-size: 20px">Draw Mask</p>""")
                slider =  gr.Slider(0, 20, step=1,  interactive=True)
                label = gr.Textbox()
                slider.release(slider_release, 
                        inputs = [slider, image_loaded,   mask_np_list_updated, mask_label_list], 
                        outputs= [canvas, label]
                    )
                add_button  = gr.Button("Add")
                add_button.click( edit_mask_add, 
                        [canvas, image_loaded, slider, mask_np_list_updated] , 
                        [mask_np_list_updated, canvas]
                    )

                save_button2  = gr.Button("Set and Save as edited masks")
                save_button2.click( save_as_edit_mask, 
                        [mask_np_list_updated,  mask_label_list, input_folder] , 
                        [] )  
                
                save_button  = gr.Button("Set and Save as original masks")
                save_button.click( save_as_orig_mask, 
                        [mask_np_list_updated,  mask_label_list, input_folder] , 
                        [] )  
                
                back_button  = gr.Button("Back to current seg")
                back_button.click( load_mask_ui, 
                                [input_folder] , 
                                [ mask_np_list_updated,mask_label_list] )

                add_mask_button = gr.Button("Add new empty mask")    
                add_mask_button.click(add_mask, 
                        [mask_np_list_updated, mask_label_list] , 
                        [mask_np_list_updated, mask_label_list] )
                

demo.queue().launch(share=True, debug=True)
