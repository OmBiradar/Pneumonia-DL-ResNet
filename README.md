# Pneumonia Detection using ResNet-18 on Chest X-rays

## Project Overview
This repository contains a deep learning solution for the automated detection of pneumonia from chest X-ray images using a ResNet-18 architecture achieving 94% accuracy. The model is designed to assist healthcare professionals in diagnosing pneumonia, potentially improving diagnostic accuracy and reducing the time to treatment.

## Features
- 94% accuracy achieved
- Implementation of ResNet-18 architecture for pneumonia classification
- Trained on chest X-ray dataset to distinguish between normal and pneumonia cases
- Pre-trained model weights available for immediate use
- Comprehensive evaluation metrics
- Jupyter notebook with step-by-step implementation

## Dataset
The model was trained on chest X-ray images which were categorized into two classes:
- Normal: X-ray images showing healthy lungs
- Pneumonia: X-ray images showing indicators of pneumonia

## Model Architecture
This project implements the ResNet-18 architecture, a deep residual network with 18 layers. ResNet addresses the vanishing gradient problem through skip connections, allowing the training of deeper networks with improved performance.

## Implementation Details
The implementation is available in the Pneumonia_Detection_CNN_ResNet18.ipynb notebook, which contains:
- Data loading and preprocessing
- Model architecture definition
- Training procedure
- Evaluation metrics and visualization
- Model saving and export

## Results
The ResNet-18 model achieves competitive performance of 94% accuracy in pneumonia detection. Detailed performance metrics including accuracy, precision, recall, and F1-score are available in the notebook.

## Pre-trained Models
Two pre-trained models are included in this repository:
- `best_model.pth`: The best performing PyTorch model
- `pneumonia_model_jit.pt`: JIT-compiled version for production deployment

## Installation and Usage
1. Clone the repository:
   ```
   git clone https://github.com/username/Pneumonia-DL-ResNet.git
   cd Pneumonia-DL-ResNet
   ```

2. Install dependencies:
   ```
   pip install torch torchvision numpy matplotlib jupyter
   ```

3. Run the Jupyter notebook:
   ```
   jupyter notebook "Pneumonia_Detection_CNN_ResNet18(1).ipynb"
   ```

4. To use the pre-trained model for inference:
   ```python
   import torch
   from torchvision import transforms
   from PIL import Image
   
   # Load model
   model = torch.jit.load('pneumonia_model_jit.pt')
   model.eval()
   
   # Prepare image
   transform = transforms.Compose([
       transforms.Resize((224, 224)),
       transforms.ToTensor(),
       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
   ])
   
   image = Image.open('path_to_xray_image.jpg')
   input_tensor = transform(image).unsqueeze(0)
   
   # Make prediction
   with torch.no_grad():
       output = model(input_tensor)
       prediction = torch.sigmoid(output) > 0.5
   ```

## Requirements
- Python 3.6+
- PyTorch 1.7+
- torchvision
- numpy
- matplotlib
- jupyter

## License
This project is licensed under the terms included in the `LICENSE` file.
