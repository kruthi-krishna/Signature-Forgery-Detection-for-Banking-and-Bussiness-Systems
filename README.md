# Signature-Forgery-Detection-for-Banking-and-Bussiness-Systems

Signature forgery is a significant concern in banking and business environments, leading to financial losses and security threats. This project aims to develop an AI-driven system that detects forged signatures using deep learning and image processing techniques, ensuring authenticity and reducing fraud risks.

## Features

- **Deep Learning Model**: Utilizes Convolutional Neural Networks (CNNs) for high-accuracy handwriting forgery detection.
- **Image Preprocessing**: Implements grayscale conversion, noise reduction, and edge detection for enhanced feature extraction.
- **Machine Learning Tools**: Uses OpenCV and Scikit-learn for dataset augmentation and feature analysis.
- **Flask-based Web Application**: Provides a real-time interface for signature verification.
- **Performance Optimization**: Integrates Cython to improve computational efficiency and execution speed.
- **User-Friendly Interface**: Designed for seamless interaction with the forgery detection system.

## Technologies Used

- **Programming Language**: Python
- **Frameworks & Libraries**: Flask, OpenCV, TensorFlow/Keras, Scikit-learn, NumPy, Pandas, Matplotlib.
- **Frontend**: HTML, CSS, JavaScript

## Project Structure

```
Signature-Forgery-Detection/
│── app.py                # Main Flask application
│── train_model.py        # Model training script
│── data_processing.py    # Data preprocessing and augmentation
│── requirements.txt      # Required dependencies
│── static/               # Static files (CSS, JS, images)
│── templates/            # HTML templates
│── models/               # Trained machine learning models
└── dataset/              # Signature images dataset
```

## Installation and Setup

### Prerequisites

Ensure you have the following installed:

- Python 3.x
- Required dependencies listed in `requirements.txt`

### Steps to Run the Project

1. **Clone the Repository**

   ```sh
   git clone https://github.com/kruthi-krishna/Signature-Forgery-Detection-for-Banking-and-Bussiness-Systems.git
   ```

2. **Navigate to the Project Directory**

   ```sh
   cd Signature-Forgery-Detection-for-Banking-and-Bussiness-Systems
   ```

3. **Install Dependencies**

   ```sh
   pip install -r requirements.txt
   ```

4. **Prepare Dataset**

   - Place signature images inside the `dataset/` directory.
   - Run `data_processing.py` to preprocess the images.

5. **Train the Model**

   ```sh
   python train_model.py
   ```

   - The trained model will be saved in the `models/` directory.

6. **Run the Web Application**

   ```sh
   python app.py
   ```

   - Open `http://127.0.0.1:5000/` in a browser to upload and verify signatures.

## Usage

- Upload a scanned signature image via the web interface.
- The model analyzes the signature and classifies it as genuine or forged with high accuracy.
- The system displays the confidence level and verification result.

## Future Enhancements

- **Enhancing Model Accuracy**: Experiment with advanced deep learning models.
- **Real-time Signature Verification**: Implement verification via touchscreen or stylus input.
- **Cloud Deployment**: Host the application for remote access and scalability.




