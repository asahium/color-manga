"""
Here we use the CBNetV2 model (https://github.com/VDIGPKU/CBNetV2) which we have trained with our labeling. 
Labels and weights will be provided in the GitHub repository of dataset.

To run this code you need to:
1. Install torch 1.7.0.
2. Compile mmcv-full 1.3.8 with the same CUDA version that torch is compiled with.
3. Run `python setup.py develop` in the repository of CBNetV2.
"""


import numpy as np
import mmcv
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import torch
import cv2
import os
import json
from detectron2.structures import Boxes, Instances, BitMasks
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog

def convert_predictions_to_instances(result):
    boxes = Boxes(result[0][0][:, :4])
    scores = torch.Tensor(result[0][0][:, 4:].transpose(1, 0)[0])
    classes = torch.ones_like(scores, dtype = torch.int64)
    
    instances = Instances(result[1][0][0].shape)
    instances.pred_boxes = boxes
    instances.scores = scores
    instances.pred_classes = classes
    instances.pred_masks = torch.cat([torch.Tensor(x[None]) for x in result[1][0]], dim = 0)
    
    return instances

def binary_mask_to_rle_np(binary_mask):
    rle = {"counts": [], "size": list(binary_mask.shape)}

    flattened_mask = binary_mask.ravel(order="F")
    diff_arr = np.diff(flattened_mask)
    nonzero_indices = np.where(diff_arr != 0)[0] + 1
    lengths = np.diff(np.concatenate(([0], nonzero_indices, [len(flattened_mask)])))

    # note that the odd counts are always the numbers of zeros
    if flattened_mask[0] == 1:
        lengths = np.concatenate(([0], lengths))

    rle["counts"] = lengths.tolist()

    return rle

config_file = 'CBNetV2/configs/cbnet/htc_cbv2_swin_large_patch4_window7_mstrain_400-1400_giou_4conv1f_adamw_1x_coco_MANGA.py'
checkpoint_file = 'CBNetV2/work_dirs/htc_cbv2_swin_large_patch4_window7_mstrain_400-1400_giou_4conv1f_adamw_1x_coco/epoch_10.pth'

model = init_detector(config_file, checkpoint_file, device='cuda:0')

data_path = "/home/myugolyadkin/proj_1504_bench/temp_aligned"
titles = [x for x in os.listdir(data_path) if 'ipynb' not in x and os.path.isdir(os.path.join(data_path, x))][0:16]
print(titles)

for n, title in enumerate(titles):
    if os.path.exists(os.path.join(data_path, title, 'segmentation_maps.json')):
        continue
        
    title_results = []
    print(n)
    print(title)
    
    chapters = [x for x in os.listdir(os.path.join(data_path, title)) if 'ipynb' not in x and os.path.isdir(os.path.join(data_path, title, x))]
    for chapter in chapters:
        print(chapter)
        
        images = [x for x in os.listdir(os.path.join(data_path, title, chapter)) if 'ipynb' not in x and '_bw' in x]
        for image in images:
            img_dict = {}
            
            img_number = image[:image.find('_bw')]
            image_path = os.path.join(data_path, title, chapter, image)
    
            result = inference_detector(model, image_path)
            if len(result[1][0]) == 0:
                continue
            inst = convert_predictions_to_instances(result)
            
            img_dict['file_path'] = os.path.join(title, chapter, img_number)
            img_dict['panels'] = []
            
            for cur_inst_n in range(len(inst)):
                panel_dict = {}
                
                cur_inst = inst[cur_inst_n]
                bbox = np.rint(cur_inst.pred_boxes.tensor.numpy())[0].astype('int').tolist()
                segmentation_dict = binary_mask_to_rle_np(cur_inst.pred_masks[0].numpy())
                score = cur_inst.scores[0].item()
                
                panel_dict['score'] = score
                panel_dict['segmentation'] = segmentation_dict
                panel_dict['bbox'] = bbox
                
                img_dict['panels'].append(panel_dict)
                
            title_results.append(img_dict)
            
            
    
            
    with open(os.path.join(data_path, title, 'segmentation_maps.json'), 'w') as file:
        json.dump(title_results, file)