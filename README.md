# MV-Group-Project
MACHINE VISION (MCTA 4364) GROUP PROJECT

AUTOMATIC LICENSE PLATE RECOGNITION (ALPR)

SEMESTER 1 2025/2026

GROUP MEMBERS
1. MUHAMMAD AMMAR FARIS BIN ARAFAS
2. AHMAD HAFIZULLAH BIN IBRAHIM (2216185)
3. MUHAMMAD IRFAN BIN ROSDIN 
4. ANAS BIN BADRULZAMAN (2219945)

Poster link: https://www.canva.com/design/DAG-a-JXG4M/6KoJf3zUET0PBgP0Z9SNaw/view?utm_content=DAG-a-JXG4M&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=h885ed9ad59
You could also download the poster in the repository

## Project Overview

This project implements an Automatic License Plate Recognition (ALPR) system designed specifically for Malaysian vehicle plates. The system is built to handle challenging environmental variations such as low lighting, angled captures, and motion blur.   

The Challenge   
Standard ALPR systems often struggle with the non-standard fonts and reflective backgrounds of Malaysian plates, especially when captured from low-resolution CCTV feeds.

## Meaningful Innovation: Dual-Ensemble Methodology

To meet the project's requirement for innovation, this system introduces a Redundant Multi-Stage Pipeline:

-Hierarchical Detection (YOLO Hierarchy): Uses a primary YOLOv11s model for vehicle detection and a custom-trained YOLO model for plates. It includes a Targeted Retry Mechanism that re-scans vehicle crops at a lower confidence threshold (0.15) if a plate isn't found globally.

-Super-Resolution Enhancement (EDSR 4x): Raw plate crops are upscaled by 400% using Enhanced Deep Residual Networks. This reconstructs pixel data lost during capture, which is critical for the subsequent OCR stage

-Dual-Ensemble OCR Engine: Instead of relying on a single model, this system runs EasyOCR and Tesseract in parallel. A consensus-based logic gate compares results and uses confidence scores to settle disagreements, reducing the Character Error Rate.

The system follows a four-phase pipeline:

Phase 1 (Detection): YOLOv11s identifies vehicles; IoU matching associates plates with specific vehicles.

Phase 2 (Enhancement): EDSR 4x sharpening of the extracted plate crop.

Phase 3 (Ensemble OCR): Parallel execution of EasyOCR and Tesseract with a weighted voting decision.

Phase 4 (Dashboard): Generation of a visual dashboard showing the full frame, the enhanced crop, and the accepted reading.

## Installation & Setup
Follow these steps to set up the environment:

Prerequisites
Python 3.8+
Tesseract OCR Engine installed and added to System PATH.

## Steps

1. Clone the Repository:

       https://github.com/anasbdrz/MV-Group-Project.git
       cd ALPR-Dual-Ensemble

2. Install Dependencies:

       pip install -r requirements.txt

3. Model Weights:

   Ensure there is best.pt in the models/ directory
   You can download the EDSR_x4.pb from the link in the .txt given.

4. Paths:

   Make sure to change the path of the directory in each script in the src folder to match your system.


## Usage

To run the ALPR system on your test dataset, run this script in your terminal:
    
    python Dual_ensemble.py
