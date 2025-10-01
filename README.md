# Project Documentation for PyTorch CNN Project

## Overview
This project implements a Convolutional Neural Network (CNN) model using PyTorch with the MobileNetV2 architecture. The goal is to provide a framework for training and evaluating the model on custom datasets.

## Directory Structure
```
pytorch-cnn-project
├── src
│   ├── models
│   │   ├── mobilenet.py       # MobileNetV2 architecture implementation
│   │   └── __init__.py        # Models package initialization
│   ├── data
│   │   ├── dataset.py         # Dataset class for loading and preprocessing data
│   │   └── __init__.py        # Data package initialization
│   ├── utils
│   │   ├── transforms.py       # Data transformation functions
│   │   └── __init__.py        # Utils package initialization
│   └── train.py               # Main training script
├── tests
│   └── test_model.py          # Unit tests for the model
├── configs
│   └── config.yaml            # Configuration settings for the project
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
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

## License
This project is licensed under the MIT License - see the LICENSE file for details.