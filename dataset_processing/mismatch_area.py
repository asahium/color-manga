import argparse
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from skimage.metrics import structural_similarity as ssim

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

def filter_small_regions(diff_image, min_area):
    contours, _ = cv2.findContours(diff_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_diff = np.zeros_like(diff_image)
    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            cv2.drawContours(filtered_diff, [contour], -1, 255, thickness=cv2.FILLED)
    return filtered_diff

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--area_ratio_threshold', type=float, default=0.005, help='Threshold for panel area ratio')
    parser.add_argument('--min_area', type=int, default=500, help='Minimum area for filtering small regions')
    args = parser.parse_args()

    data_path = args.data_path
    area_ratio_threshold = args.area_ratio_threshold
    min_area = args.min_area

    titles = [x for x in os.listdir(data_path) if 'ipynb' not in x and os.path.isdir(os.path.join(data_path, x))]

    for title_num, title in enumerate(titles):
        title_path = os.path.join(data_path, title)
        seg_path = os.path.join(title_path, 'segmentation_maps.json')

        with open(seg_path, 'r') as file:
            seg_dict = json.load(file)

        for img_dict in seg_dict:
            file_path = img_dict['file_path']
            img_path = os.path.join(data_path, file_path)
            
            img_bw_file_path = img_path + '_bw.png'
            img_color_file_path = img_path + '_color.png'
        
            try:
                img_bw = plt.imread(img_bw_file_path)
                img_color = plt.imread(img_color_file_path)
            except:
                continue
            
            for panel in img_dict['panels']:
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
        
                image_copy_bw = (image_copy_bw * 255).astype('uint8')
                image_copy_color = (image_copy_color * 255).astype('uint8')
        
                image_copy_color = cv2.cvtColor(image_copy_color, cv2.COLOR_RGB2GRAY)
        
                bw_image_blur = cv2.GaussianBlur(image_copy_bw, (5, 5), 0)
                gray_image_blur = cv2.GaussianBlur(image_copy_color, (5, 5), 0)
                
                log_bw = cv2.Laplacian(bw_image_blur, cv2.CV_64F)
                log_gray = cv2.Laplacian(gray_image_blur, cv2.CV_64F)
                
                # Convert LoG results to absolute values
                edges_bw = np.uint8(np.absolute(log_bw))
                edges_color = np.uint8(np.absolute(log_gray))
        
                met, diff = ssim(edges_bw, edges_color, full=True)
        
                diff = (diff * 255).astype("uint8")
                thresh = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY_INV)[1]
                thresh_res = filter_small_regions(thresh, area_ratio_threshold * ((y2 - y1) * (x2 - x1)))
                contours, _ = cv2.findContours(thresh_res.copy() * cur_mask[y1:y2, x1:x2], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
                sum_area = 0
                for contour in contours:
                    sum_area += cv2.contourArea(contour)
        
                area_ratio = sum_area / ((y2 - y1) * (x2 - x1))

                panel['area_ratio'] = area_ratio

        with open(os.path.join(title_path, 'seg_with_ratio.json'), 'w') as file:
            json.dump(seg_dict, file)

if __name__ == '__main__':
    main()
