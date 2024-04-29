# Real-Time-Identification-of-Bean-Plant-Pathologies

This repository is dedicated to the development of machine learning models for the precise in-field identification of bean diseases in Uganda. Leveraging state-of-the-art Convolutional Neural Network (CNN) architectures, this project aims to tackle significant agricultural challenges by improving the detection and management of key bean diseases, such as Angular Leaf Spot (ALS) and bean rust, using smartphone images.

## Project Overview

Beans are a staple crop globally, crucial for nutrition and food security. However, their cultivation is hindered by several factors including diseases, which can severely impact yield quality and quantity, leading to substantial economic losses. This project utilizes the Makerere University beans image dataset, which includes 15,335 images across three classes: ALS, bean rust, and healthy. To enhance the model's robustness, the dataset was expanded with an "unknown" class, comprising 2,800 additional images.

### Key Results

- **EfficientNet v2 B0** and a custom CNN model, **BeanWatchNet**, were the top performers in image classification tasks, achieving accuracies of **91%** and **90%**, respectively.
- The **YOLO v8** model excelled in object detection, achieving a maximum average precision (mAP@50) of **87.6%**.
- The models were successfully quantized and deployed on edge platforms, including smartphones and Raspberry Pi 4B, demonstrating their effectiveness in real-world scenarios.

## Repository Structure

- `Classification/`: Contains scripts and trained models for image classification.
- `Detection/`: Contains scripts and trained models for object detection.

## Getting Started

### Prerequisites

Ensure you have Python installed, along with TensorFlow and other necessary libraries:
