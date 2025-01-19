import argparse
from transformers import AutoModel
import numpy as np
from PIL import Image
import torch
import os
import json


def read_image_as_np_array(image_path):
    with open(image_path, "rb") as file:
        image = Image.open(file).convert("L").convert("RGB")
        image = np.array(image)
    return image

def walklevel(some_dir, level=1):
    some_dir = some_dir.rstrip(os.path.sep)
    assert os.path.isdir(some_dir)
    num_sep = some_dir.count(os.path.sep)
    for root, dirs, files in os.walk(some_dir):
        yield root, dirs, files
        num_sep_this = root.count(os.path.sep)
        if num_sep + level <= num_sep_this:
            del dirs[:]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--results_path', type=str, required=True, help='Path to save the results')
    args = parser.parse_args()

    data_path = args.data_path
    results_path = args.results_path

    subdirs = []
    model = AutoModel.from_pretrained("ragavsachdeva/magi", trust_remote_code=True).cuda()
    processed_dict = dict()
    _counter = 0

    for x in walklevel(data_path):
        try:
            if _counter == 0:
                _counter += 1
                continue
                
            dir_block = x
            filepathes = []
            predicat = dir_block[0]

            for _subdir, dirs, files in os.walk(predicat):
                for file in files:
                    filepath = _subdir + os.sep + file
                    if not filepath.endswith(".json"):
                        filepathes.append(filepath)

            filepathes_bw = [x for x in filepathes if '_bw' in x]
            filepathes_colored = [x for x in filepathes if '_color' in x]

            filepathes_final = filepathes_bw + filepathes_colored

            chunk = 5
            result_file_path = os.path.join(results_path, f"bboxes_for_pages_{os.path.basename(predicat)}.json")
            if os.path.exists(result_file_path):
                with open(result_file_path) as outfile:
                    result_dict = json.load(outfile)
            else:
                result_dict = dict()
                
            for i in range(0, len(filepathes_final), chunk):
                text_bboxes_for_all_images = []
                images = [read_image_as_np_array(image) for image in filepathes_final[i:min(i + chunk, len(filepathes_final))]]
                with torch.no_grad():
                    results = model.predict_detections_and_associations(images, text_detection_threshold=0.1)
                    text_bboxes_for_all_images.extend([x["texts"] for x in results])

                for j in range(i, min(i + chunk, len(filepathes_final))):
                    result_dict[filepathes_final[j]] = text_bboxes_for_all_images[j % chunk]

                with open(result_file_path, "w") as outfile:
                    json.dump(result_dict, outfile, indent=4)

            processed_dict[predicat] = 'Finished'
        except Exception as e:
            processed_dict[predicat] = f'Failed {e}'

if __name__ == '__main__':
    main()
