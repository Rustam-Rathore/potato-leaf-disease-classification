
# Potato Leaf Disease Classification

A CNN based deep learning project to classify potato leaf images into different disease categories using TensorFlow and Streamlit.

## Features
- Upload potato leaf images
- Predict leaf disease using CNN
- Display prediction confidence scores
- Interactive Streamlit web interface

## Tech Stack
- Python
- TensorFlow / Keras
- CNN
- Streamlit
- NumPy
- Matplotlib

## Installation

```bash
git clone https://github.com/Rustam-Rathore/potato-leaf-disease-classification.git

cd potato-leaf-disease-classification

pip install -r requirements.txt
```

## Run Application

```bash
streamlit run app.py
```

## Project Structure

```text
├── app.py
├── potato_classification_cnn_code.ipynb
├── requirements.txt
├── classification_report.txt
├── README.md
└── .gitignore
```

## Model Details
- Input Size: 128 × 128
- Input Channels: RGB
- Architecture: CNN
- Optimizer: Adam
- Output: 3 Classes
- Activation: Softmax

## Author
Rustam Rathore