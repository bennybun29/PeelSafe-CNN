# Project Documentation for PyTorch CNN Project

## Overview
This project implements a Convolutional Neural Network (CNN) model using PyTorch with the MobileNetV2 architecture. The goal is to provide a framework for training and evaluating the model on custom datasets.

## Directory Structure
# PeelSafe-CNN: Banana Leaf Disease Detection

## Overview
A deep learning model using MobileNetV2 architecture to detect various banana leaf diseases. The model can identify 9 different classes including healthy leaves and 8 disease types.

## Features
- MobileNetV2-based architecture for efficient inference
- Support for 9 disease classes
- Real-time prediction capabilities
- Data augmentation for improved robustness
- Comprehensive test suite

## Quick Start

### Installation
```bash
git clone https://github.com/bennybun29/PeelSafe-CNN.git
cd PeelSafe-CNN
pip install -r requirements.txt
```

### Training
```bash
python src/train.py
# or train verifier:
python src/train_verifier.py
```

### Making Predictions
For a single image:
```bash
python src/predict.py "path/to/image.jpg"
```

For multiple images in a directory:
```bash
python src/predict.py "path/to/directory"
```

## Disease Classes
- Banana Black Sigatoka Disease
- Banana Bract Mosaic Virus Disease
- Banana Cordana Disease
- Banana Healthy
- Banana Insect Pest Disease
- Banana Moko Disease
- Banana Panama Disease
- Banana Pestalotiopsis Disease
- Banana Yellow Sigatoka Disease

## Project Structure
```
PeelSafe-CNN/
├── configs/          # Model and training configurations
├── data/            # Dataset directory
├── src/             # Source code
├── tests/           # Unit tests
├── requirements.txt # Dependencies
└── README.md
```

## Model Performance
The model uses MobileNetV2 pretrained on ImageNet and is fine-tuned on the banana leaf disease dataset. Training metrics and model weights are saved in the `results` directory.

## Running Tests
```bash
python tests/test_model.py
```

## Setup Instructions
1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd pytorch-cnn-project
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Training Script
To train the model, run the following command:
```bash
python src/train.py --config configs/config.yaml
```

## Evaluating the Model
After training, you can evaluate the model using the provided test script:
```bash
python tests/test_model.py
```

## Predicting a Leaf Disease
```bash
python src/predict.py path/to/your/image.jpg
```

## To Run API
```bash
python run_api.py
# then open http://127.0.0.1:8000/docs
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.