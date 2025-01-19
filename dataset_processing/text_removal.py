import argparse
import cv2 as cv
import numpy as np
import json
import os
import time
from multiprocessing import Process

def processing_block(files, worknum, data_path, output_path, processed_data_path):
    processed_data = os.path.join(processed_data_path, f'processed_data_{worknum}.json')
    for file in files:
        if os.path.exists(processed_data):
            with open(processed_data) as outfile:
                result_dict = json.load(outfile)
        else:
            result_dict = dict()
        with open(os.path.join(data_path, file), 'r') as f:
            bboxes_and_pathes = json.load(f)

        start_time = time.time()
        save_path = None
        dlina = len(bboxes_and_pathes)
        counter_our = 0
        try:
            for key in bboxes_and_pathes:
                counter_our += 1
                save_path = key.replace("anime_coursework", "anime_cleaned")
                path = key
                img_original_rgb = cv.imread(path, cv.COLOR_RGB2BGR)
                img_original = cv.imread(path, cv.IMREAD_GRAYSCALE)
                boxes = bboxes_and_pathes[key]
                for block in boxes:
                    for i in range(len(block)):
                        if block[i] < 0:
                            block[i] = 0
                    img = img_original[round(block[1]):round(block[3]), round(block[0]):round(block[2])]
                    try:
                        img = cv.medianBlur(img, 3)
                    except Exception as e:
                        continue
                    img = cv.GaussianBlur(img, (3, 3), 0)
                    ret, thresh1 = cv.threshold(img, 30, 255, cv.THRESH_OTSU | cv.THRESH_BINARY_INV)
                    rect_kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
                    dilation = cv.dilate(thresh1, rect_kernel, iterations=1)
                    img_arr = img.reshape(-1, 1)
                    dilation_arr = dilation.reshape(-1, 1)
                    crop_without_text = img_arr[np.where(dilation_arr == 0)[0]]
                    if np.std(crop_without_text) < 11:
                        if 'bw' in key:
                            img_temp = img_original[round(block[1]):round(block[3]),
                                       round(block[0]):round(block[2])].copy()
                            unique, counts = np.unique(img_temp.reshape(-1, 1)[np.where(dilation_arr == 0)[0]], axis=0,
                                                       return_counts=True)
                            img_temp[:, :] = unique[np.argmax(counts)][0]
                            img_original[round(block[1]):round(block[3]), round(block[0]):round(block[2])] = img_temp
                        else:
                            img_temp = img_original_rgb[round(block[1]):round(block[3]),
                                       round(block[0]):round(block[2])].copy()
                            unique, counts = np.unique(img_temp.reshape(-1, 3)[np.where(dilation_arr == 0)[0]], axis=0,
                                                       return_counts=True)
                            img_temp[:, :, 0], img_temp[:, :, 1], img_temp[:, :, 2] = unique[np.argmax(counts)]
                            img_original_rgb[round(block[1]):round(block[3]),
                            round(block[0]):round(block[2])] = img_temp

                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                if 'bw' in key:
                    cv.imwrite(save_path, img_original)
                else:
                    cv.imwrite(save_path, img_original_rgb)
                result_dict[save_path] = True

                with open(processed_data, "w") as outfile:
                    json.dump(result_dict, outfile, indent=4)
        except Exception as e:
            if save_path is not None:
                result_dict[save_path] = False

        print("--- %s seconds ---" % (time.time() - start_time))

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def main(data_path, output_path, processed_data_path, const_split):
    files = text.split('\n')
    files_block = list(split(files, const_split))
    consumers = []
    for i in range(const_split):
        p = Process(target=processing_block, args=(files_block[i], i, data_path, output_path, processed_data_path))
        consumers.append(p)
        p.start()

    for kl in consumers:
        kl.join()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to the input data')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the output data')
    parser.add_argument('--processed_data_path', type=str, required=True, help='Path to save the processed data JSON files')
    parser.add_argument('--const_split', type=int, default=20, help='Number of parallel processes')
    args = parser.parse_args()

    main(args.data_path, args.output_path, args.processed_data_path, args.const_split)
