# Image Classification Models

This folder contains TensorFlow implementations of three advanced convolutional neural network models for image classification: EfficientNetV2, ResNetV2, and MobileNetV3, plus the BeanWatchNet model. These scripts are designed to be run from the command line, allowing for straightforward training and evaluation.

## Setup

Before you begin, ensure you have Python installed, along with TensorFlow and other necessary libraries. Here's how to set up your environment using bash:

```bash
# Create a virtual environment (optional but recommended)
python -m venv env
source env/bin/activate

# Install required Python packages
pip install tensorflow matplotlib numpy seaborn

# BeanWatchNet

#Use this command to run train the custom classification model
python custom_classication_model.py --mode train

#Use this command to inference or evaluate the classification model
python custom_classication_model.py.py --mode evaluate

# EfficientNetV2

#Use this command to run train the custom classification model
python efficientnet_script.py --mode train --data_dir ****/data --batch_size 32 --epochs 10

#Use this command to inference or evaluate the classification model
python efficientnet_script.py --mode evaluate --data_dir ****/data --batch_size 32

# ResNetV2

#Use this command to run train the custom classification model
python resnet_script.py --mode train --data_dir ****/data --batch_size 32 --epochs 10

#Use this command to inference or evaluate the classification model
python resnet_script.py --mode evaluate --data_dir ****/data --batch_size 32

MobileNetV3

#Use this command to run train the custom classification model
python mobilenetv3_script.py --mode train --data_dir ****/data --batch_size 32 --epochs 10

#Use this command to inference or evaluate the classification model
python mobilenetv3_script.py --mode evaluate --data_dir ****/data --batch_size 32

