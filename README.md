# Vehicle Number Plate Image Enhancement and Sharpening System Using Computer Vision

## Overview

A computer vision system for detecting vehicle number plates, extracting the plate region, enhancing image quality, sharpening character details, and recognizing the plate number using OCR.

## Pipeline

```text
Vehicle Image
      ↓
Number Plate Detection
      ↓
Plate Extraction
      ↓
Blur / Quality Analysis
      ↓
Image Enhancement
      ↓
Laplacian / High-Pass Sharpening
      ↓
OCR
      ↓
Evaluation

Objectives
Detect vehicle number plates from images.
Extract the detected plate region.
Analyze and improve degraded plate images.
Enhance character edges using Laplacian and high-pass filtering.
Apply OCR to recognize vehicle numbers.
Evaluate detection, enhancement, and OCR performance.
Datasets
Indian Number Plates Dataset

Used for number plate detection and localization.

Indian License Plate Images

Used for plate image enhancement and OCR-related processing.

Technologies
Python
OpenCV
NumPy
YOLO (Ultralytics)
scikit-image
Matplotlib
Pillow
OCR
Number Plate Detection

A pretrained YOLO11n model is fine-tuned to detect a single class:

The detection dataset contains 27 annotated images and is divided into:

Train: 22
Validation: 3
Test: 2
Detection Results
Metric	Score
Precision	0.1981
Recall	0.3333
mAP@50	0.3423
mAP@50–95	0.2366

The detected bounding boxes are used to automatically crop the number plate region for subsequent enhancement and OCR.

Project Structure
Vehicle-Number-Plate-Enhancement/
│
├── data/
│   └── processed/
│       └── plate_detection_v2/
│
├── notebooks/
│   └── 01_plate_detection.ipynb
│
├── src/
│   └── detection/
│       ├── train.py
│       ├── evaluate.py
│       └── detect_and_crop.py
│
├── outputs/
├── .gitignore
├── requirements.txt
└── README.md

Dataset files, trained model weights, and generated outputs are excluded from version control.

Future Work
Advanced image enhancement and sharpening
Adaptive blur detection
High-pass and Laplacian filtering comparison
OCR integration
End-to-end evaluation
Real-time CCTV/video processing