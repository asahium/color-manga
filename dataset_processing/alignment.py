import argparse
import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--to_path', type=str, required=True, help='Path to save aligned images')
    args = parser.parse_args()

    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher()

    data_path = args.data_path
    to_path = args.to_path
    titles = [x for x in os.listdir(data_path) if 'ipynb' not in x and os.path.isdir(os.path.join(data_path, x))]

    result_list = []
    for title in titles:
        with open(os.path.join(data_path, title, 'matching.json'), 'r') as file:
            cur_json = json.load(file)

        for cur_chapter_matching in cur_json['chapters_matching']:
            bw_folder_name = cur_chapter_matching['bw_folder_name']
            color_folder_name = cur_chapter_matching['color_folder_name']
            
            save_folder_name = os.path.basename(bw_folder_name)
            save_chapter_path = os.path.join(to_path, title, save_folder_name)
            os.makedirs(save_chapter_path, exist_ok=True)
            
            for n, cur_img_matching in enumerate(cur_chapter_matching['img_matching']):
                bw_file_name = os.path.splitext(cur_img_matching['bw_file'])[0] + '.png'
                color_file_name = os.path.splitext(cur_img_matching['color_file'])[0] + '.png'
                
                bw_img_path = os.path.join(data_path, title, bw_folder_name, bw_file_name)
                cl_img_path = os.path.join(data_path, title, color_folder_name, color_file_name)
                
                save_bw_path = os.path.join(save_chapter_path, f"{n}_bw.png")
                save_color_path = os.path.join(save_chapter_path, f"{n}_color.png")

                if os.path.exists(save_bw_path) and os.path.exists(save_color_path):
                    continue
        
                img_query = cv2.imread(bw_img_path, cv2.IMREAD_GRAYSCALE)
                img_target = cv2.imread(cl_img_path, cv2.IMREAD_GRAYSCALE) 
                color_target = True

                if img_query is None or img_target is None:
                    continue

                if img_query.shape[0] * img_query.shape[1] < img_target.shape[0] * img_target.shape[1]:
                    img_query, img_target = img_target, img_query
                    color_target = False
                    
                kp_query, des_query = sift.detectAndCompute(img_query, None)
                kp_target, des_target = sift.detectAndCompute(img_target, None)

                if len(kp_query) == 0 or len(kp_target) == 0 or des_query is None or des_target is None:
                    continue
            
                matches = bf.knnMatch(des_query, des_target, k=2)

                good = []
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        good.append(m)

                if len(good) == 0:
                    continue
                        
                query_pts = np.float32([kp_query[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                target_pts = np.float32([kp_target[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                
                H, inliers = cv2.estimateAffine2D(query_pts, target_pts)
                
                inliers_ratio = inliers.sum() / len(good)
                if inliers_ratio < 0.2:
                    continue
                    
                if not color_target:
                    img_query = cv2.imread(cl_img_path)

                warped_query = cv2.warpAffine(img_query, H, (img_target.shape[1], img_target.shape[0]), borderValue=(255, 255, 255))
                
                if color_target:
                    bw_save_img = warped_query
                    color_save_img = cv2.imread(cl_img_path)
                else:
                    bw_save_img = img_target
                    color_save_img = warped_query
                
                bw_save_img = bw_save_img - bw_save_img.min()
                bw_save_img = np.clip((bw_save_img * (255 / bw_save_img.max())).astype('int'), a_min=0, a_max=256)
                
                mask = np.ones_like(img_query) * 255
                if color_target:
                    mask = np.repeat(mask[:, :, None], 3, axis=2)
                mask_warped = cv2.warpAffine(mask, H, (img_target.shape[1], img_target.shape[0]), borderValue=(10, 0, 0))
                
                crop_mask = (mask_warped.sum(axis=2) > 500)
                if crop_mask.mean() < 1.:
                    left_up_crop_mask = crop_mask[:crop_mask.shape[0] // 2, :crop_mask.shape[1] // 2]
                    cols_sums = (~left_up_crop_mask).sum(axis=0)
                    bad_y = np.where(cols_sums > left_up_crop_mask.shape[1] // 2)[0]
                    if bad_y.shape[0] == 0:
                        min_y = 0
                    else:
                        min_y = bad_y.max() + 1

                    rows_sum = (~left_up_crop_mask).sum(axis=1)
                    bad_x = np.where(rows_sum > left_up_crop_mask.shape[0] // 2)[0]
                    if bad_x.shape[0] == 0:
                        min_x = 0
                    else:
                        min_x = bad_x.max() + 1

                    right_down_crop_mask = crop_mask[crop_mask.shape[0] // 2:, crop_mask.shape[1] // 2:]
                    cols_sums = (~right_down_crop_mask).sum(axis=0)
                    bad_y = np.where(cols_sums > right_down_crop_mask.shape[1] // 2)[0]
                    if bad_y.shape[0] == 0:
                        max_y = crop_mask.shape[1]
                    else:
                        max_y = bad_y.min() + crop_mask.shape[1] // 2 
                    rows_sum = (~right_down_crop_mask).sum(axis=1)
                    bad_x = np.where(rows_sum > right_down_crop_mask.shape[0] // 2)[0]
                    if bad_x.shape[0] == 0:
                        max_x = crop_mask.shape[0]
                    else:
                        max_x = bad_x.min() + crop_mask.shape[0] // 2
                    
                    ratio = (max_x - min_x - 1) * (max_y - min_y - 1) / (crop_mask.shape[0] * crop_mask.shape[1])
                    if ratio < 0.3:
                        continue
                    
                    bw_save_img = bw_save_img[min_x:max_x, min_y:max_y]
                    color_save_img = color_save_img[min_x:max_x, min_y:max_y]
                    
                cv2.imwrite(save_bw_path, bw_save_img)
                cv2.imwrite(save_color_path, color_save_img)
                    
                result_list.append((save_bw_path, save_color_path, inliers_ratio))

if __name__ == '__main__':
    main()
