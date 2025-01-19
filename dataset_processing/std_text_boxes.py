import argparse
import cv2 as cv
import numpy as np
import json
from multiprocessing import Pool, cpu_count

def process_part(data_part):
    dlina = len(data_part)
    counter = 0
    result_dict = dict()
    for key in data_part:
        img_original = cv.imread(key, cv.IMREAD_GRAYSCALE)
        boxes = data_part[key]
        arr_of_block_and_std = []
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
            our_std = np.std(crop_without_text)
            arr_of_block_and_std.append((block, our_std))
        result_dict[key] = arr_of_block_and_std
        counter += 1

    return result_dict

def split_dict_equally(input_dict, num_splits):
    """Splits a dictionary into `num_splits` smaller dictionaries with roughly equal sizes."""
    items = list(input_dict.items())
    chunk_size, remainder = divmod(len(items), num_splits)

    splits = []
    start = 0
    for i in range(num_splits):
        end = start + chunk_size + (1 if i < remainder else 0)
        splits.append(dict(items[start:end]))
        start = end

    return splits

def merge_dicts(dicts):
    """Merges a list of dictionaries into a single dictionary."""
    result = {}
    for d in dicts:
        result.update(d)
    return result

def main(input_file, output_file, num_processes):
    # Read the JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)

    # Split the data into parts
    data_parts = split_dict_equally(data, num_processes)

    # Create a pool of processes
    with Pool(num_processes) as pool:
        processed_parts = pool.map(process_part, data_parts)

    # Merge the processed parts back into a single dictionary
    processed_data = merge_dicts(processed_parts)

    # Write the processed data back to the JSON file
    with open(output_file, 'w') as f:
        json.dump(processed_data, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str, required=True, help='Text containing the list of JSON files to process')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing the input JSON files')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the output JSON files')
    args = parser.parse_args()

    for block in args.text.split('\n'):
        input_file = os.path.join(args.input_dir, block)
        output_file = os.path.join(args.output_dir, block)
        num_processes = cpu_count()

        main(input_file, output_file, num_processes)

        main(input_file, output_file, num_processes)
        print(f"{block} ended")
