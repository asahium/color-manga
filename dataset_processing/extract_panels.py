import argparse
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def rle_to_binary_mask(rle_dict):
    """
    Convert RLE encoded data to a binary mask.

    Args:
    rle_dict (dict): A dictionary containing 'counts' and 'size' keys.

    Returns:
    np.ndarray: A binary mask of the given image size.
    """
    counts = rle_dict['counts']
    width, height = rle_dict['size']
    
    # Initialize the binary mask
    binary_mask = np.zeros(height * width, dtype=np.uint8)
    
    # Decode RLE
    index = 0
    for i, count in enumerate(counts):
        if i % 2 == 0:
            index += count
        else:
            binary_mask[index:index + count] = 1
            index += count

    # Reshape the flat binary mask to 2D
    binary_mask = binary_mask.reshape((height, width)).T
    
    return binary_mask

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save filtered images')
    parser.add_argument('--area_ratio', type=float, default=0.014, help='Threshold for panel area ratio')
    args = parser.parse_args()

    data_path = args.data_path
    save_path = args.save_path
    area_ratio_threshold = args.area_ratio

    titles = [x for x in os.listdir(data_path) if 'ipynb' not in x and os.path.isdir(os.path.join(data_path, x))]

    for title_num, title in enumerate(titles):
        title_path = os.path.join(data_path, title)
        seg_path = os.path.join(title_path, 'seg_with_ratio.json')

        with open(seg_path, 'r') as file:
            seg_dict = json.load(file)

        for img_dict in seg_dict:
            file_path = img_dict['file_path']
            img_path = os.path.join(data_path, file_path)
            
            img_bw_file_path = img_path + '_bw.png'
            img_color_file_path = img_path + '_color.png'

            if not os.path.exists(img_bw_file_path) or not os.path.exists(img_color_file_path):
                continue
            
            img_bw = plt.imread(img_bw_file_path)
            img_color = plt.imread(img_color_file_path)

            new_file_name = file_path.replace('/', '__').replace(' ', '___')

            for num, panel in enumerate(img_dict['panels']):
                if 'area_ratio' not in panel or panel['area_ratio'] >= area_ratio_threshold:
                    continue
                    
                cur_seg = panel['segmentation']
                cur_mask = rle_to_binary_mask(cur_seg)
                cur_bbox = panel['bbox']
                x1, y1, x2, y2 = cur_bbox
                
                image_copy_bw = img_bw.copy()
                image_copy_bw[cur_mask == 0] = 1.
                
                image_copy_bw = image_copy_bw[y1:y2, x1:x2]
                
                image_copy_color = img_color.copy()
                image_copy_color[cur_mask == 0] = 1.
                
                image_copy_color = image_copy_color[y1:y2, x1:x2]

                panel_img_path_bw = os.path.join(save_path, 'bw', new_file_name + '_' + str(num) + '.png')
                panel_img_path_color = os.path.join(save_path, 'color', new_file_name + '_' + str(num) + '.png')

                image_copy_bw = (image_copy_bw * 255).astype('uint8')
                image_copy_color = (image_copy_color * 255).astype('uint8')

                os.makedirs(os.path.dirname(panel_img_path_bw), exist_ok=True)
                os.makedirs(os.path.dirname(panel_img_path_color), exist_ok=True)

                cv2.imwrite(panel_img_path_bw, image_copy_bw)
                cv2.imwrite(panel_img_path_color, image_copy_color[:, :, ::-1])

if __name__ == '__main__':
    main()
