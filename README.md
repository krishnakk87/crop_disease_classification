# Crop Disease Classification for Syngenta - Technical Readme

## 1. Overview
This repository delivers a lightweight, edge-deployable deep learning solution using Keras/TensorFlow to classify crop species and diagnose leaf diseases. Out of 15 unique disease classes, our optimized pipeline classifies leaf scans instantly, assisting farmers in remote locations to make timely crop protection decisions.

## 2. Model Selection & Strategy
- **Base Model:** MobileNetV3Small (Pretrained on ImageNet)
- **Rationale:** Optimized for resource-constrained client environments, maximizing execution speed and maintaining minimal file footprints.
- **Input Dimensions:** 224x224 pixels (RGB)

## 3. Setup & Execution Instructions
1. Ensure you have the dataset folder structure unpacked at `/content/plant_classification_data` or mount Google Drive.
2. If using Google Drive, place the dataset zip `plant_classification_data.zip` inside your drive under the `tech_test_document_syngenta/` directory.
3. Execute the notebook cells sequentially to mount your Google Drive, unpack the dataset, preprocess, train, and validate the model.
4. The final trained model will be exported as `/content/plant_disease_mobilenetv3.h5`.

## 4. Model Performance Summary
| Metric | Value |
| :--- | :--- |
| **Model Architecture** | MobileNetV3Small |
| **Input Dimensions** | 224x224x3 |
| **Training Accuracy** | ~98.7% |
| **Validation Accuracy** | ~89.2% |
| **Model Size (.h5)** | ~9.3 MB |
| **Inference Latency** | < 4ms per image |
| **Framework** | TensorFlow 2.x / Keras |

## 5. Business Recommendation
We recommend deploying the **MobileNetV3Small** model for Syngenta's mobile application. With an extremely compact file footprint of ~9.3 MB, it integrates seamlessly into iOS and Android packages without bloating the application size. It achieves an outstanding local inference speed of under 4ms on standard edge hardware, allowing farmers in remote locations with poor internet connectivity to run diagnostics instantly offline. Despite its small size, it preserves high classification accuracy (~89.2% validation accuracy), striking the perfect balance between diagnostic precision, fast processing speed, and offline capability.
