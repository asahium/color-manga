# Manga Dataset Processing Pipeline

This folder contains scripts for processing a manga dataset. The dataset consists of manga titles, each with black and white (bw) and color editions. Each edition is divided into chapters, with each chapter containing images of manga pages.

## Dataset Structure

The dataset is organized as follows:

dataset/
├── Title1/
│   ├── bw/
│   │   ├── Chapter1/
│   │   │   ├── page1.png
│   │   │   ├── page2.png
│   │   │   └── ...
│   │   ├── Chapter2/
│   │   └── ...
│   ├── color/
│   │   ├── Chapter1/
│   │   │   ├── page1.png
│   │   │   ├── page2.png
│   │   │   └── ...
│   │   ├── Chapter2/
│   │   └── ...
│   └── matching.json
├── Title2/
└── ...

## Steps to Process the Dataset

### 1. Remove JPEG Artifacts

Run `dejpeg.py` to remove JPEG artifacts from the images. Since the images were sourced from the internet, they may be heavily corrupted.

### 2. Align Matched Files

Run `alignment.py` to generate a dataset of aligned pairs from matched files. Matching information is provided in JSON files in the directory of each title. This script creates a new dataset where there is no division into bw and color chapters. Instead, each title folder contains chapter folders, and each chapter folder contains both bw and color images.

### 3. Remove Text from Speech Bubbles

#### a. Detect Text Boxes

Run `predict_text_boxes.py` to detect text on manga pages. It creates a JSON file for each title.

#### b. Calculate STD for Bounding Boxes

Run `std_text_boxes.py` to calculate the standard deviation for the text background in predicted bounding boxes to detect false positives. It modifies the JSON files created in the previous step.

#### c. Text Removal

Run `text_removal.py` to filter out false positives by standard deviation and fill good bounding boxes with a flat color chosen from the text background.

After these steps, the page-level dataset is ready.

### 4. Predict Panel Segmentation

Run `panel_segmentation.py` to predict instance segmentation for panels on manga pages. Results are saved in JSON files.

### 5. Calculate Mismatch Area

Run `mismatch_area.py` to calculate the ratio of regions between bw and color images that cannot be pixel-wise matched. Results are saved in JSON files.

### 6. Generate New Panel-Level Dataset

Generate a new dataset by creating new images where each image is a separate panel. The matching and alignment between bw and color are saved. Before creating new files, panels are filtered by mismatch ratio.

## Additional Preprocessing Steps

Further preprocessing steps will be available in the GitHub repository of the dataset.
