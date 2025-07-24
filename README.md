# Machine-Learning-Foundations-From-Concepts-to-Real-World-Projects
This is the repository containing the projects mentioned in the book Machine Learning Foundations: From Concepts to Real World Projects, which can be found on amazon. If you have read our book or not this contains 5 project notebooks with a mix of different machine learning concepts. If you have any questions reach out at sb.sg.code@gmail.com.

# Suicide Detection in Large Groups of Teenagers Using Linear Regression

Developers: Sahiel Bose & Shanay Gaitonde

Research Article: https://medium.com/nextgen-innovations/suicide-detection-in-large-groups-of-teenagers-using-linear-regression-d87a05e820f0

Suicide detection is a critical challenge in mental health analytics, and this project aims to provide a data-driven approach using Linear Regression. By analyzing key behavioral, social, and psychological factors, the model predicts the likelihood of suicide risk in teenagers. Built with Python and leveraging machine learning libraries such as scikit-learn, this project offers an interpretable solution for early intervention strategies.

The dataset undergoes rigorous preprocessing, including feature selection, missing value handling, and normalization, to ensure optimal performance. The core model is based on Linear Regression, which identifies patterns in the data and provides probability estimates for at-risk individuals. Evaluation metrics such as Mean Squared Error (MSE) and R² Score are used to assess model accuracy and effectiveness. The project also integrates visualization tools to illustrate trends and highlight significant predictors.

This project is designed for scalability, allowing further refinement through feature engineering and model tuning. It can be expanded with additional data sources, such as social media sentiment analysis or clinical records, to improve predictive capabilities. The goal is to create an AI-assisted tool that can aid professionals in recognizing early warning signs and implementing timely interventions.

To use the project, users can clone the repository, install dependencies, and run the provided Jupyter Notebook. The workflow includes data loading, preprocessing, model training, and evaluation. Contributions from researchers and developers are encouraged to enhance model performance and expand its applicability.

# Improving Heart Disease Prediction Accuracy: An Exploration of Logistic Regression and KNN

Developers: Sahiel Bose & Shanay Gaitonde

Research Article: https://medium.com/nextgen-innovations/improving-heart-disease-prediction-accuracy-an-exploration-of-logistic-regression-and-knn-5e4af2aed66c

HeartKNN is a machine learning-based system designed to classify heart disease risk using K-Nearest Neighbors (KNN) and Linear Regression. By leveraging data-driven techniques, the model provides accurate predictions to aid early detection and prevention of heart disease. The project is built using Python and machine learning libraries such as scikit-learn, offering a scalable and efficient solution for medical data analysis.

Project Highlights
The project focuses on enhancing model robustness through preprocessing, implementing KNN and Linear Regression for classification, and evaluating performance using accuracy, precision, recall, and a confusion matrix. It is scalable and customizable, allowing for fine-tuning with different datasets and parameters.

Preprocessing Pipeline
The preprocessing stage includes feature scaling with normalization to ensure data consistency. Missing values are handled using imputation techniques, and important features are selected based on correlation and importance scores.

Model Architecture
The project uses K-Nearest Neighbors (KNN) for classification, which is a distance-based algorithm, and Linear Regression for trend analysis to understand contributing factors. KNN hyperparameters, such as the number of neighbors, are fine-tuned to optimize performance.

Training Strategy
An adaptive optimizer is used to improve model convergence, and the loss function is evaluated using classification metrics. Cross-validation ensures the model generalizes well and prevents overfitting.

Post-Training Evaluation
After training, the model is tested on a separate dataset, with key evaluation metrics including accuracy, precision, recall, and the confusion matrix. These metrics provide insights into the model’s classification performance.

Dataset
The dataset includes health-related attributes such as age, cholesterol levels, and blood pressure. It is split into a training set for learning, a validation set for hyperparameter fine-tuning, and a test set for final performance assessment.

Results
The model achieves high accuracy after hyperparameter tuning. Visualization tools, such as training curves and the confusion matrix, help analyze the model's effectiveness.

# PneumoVision: Pneumonia Classification Using Deep Learning

Developers: Sahiel Bose & Shanay Gaitonde

Research Article: https://medium.com/nextgen-innovations/enhancing-pneumonia-identification-using-vgg16-and-deep-learning-techniques-44ea3bd8e10f

PneumoVision is a deep learning-based system designed to classify chest X-ray images as either Pneumonia or Normal. Using transfer learning and data preprocessing techniques, the model achieves high accuracy with efficient training times. The project is built with TensorFlow/Keras and leverages state-of-the-art models to streamline medical image analysis.

Project Highlights
Preprocessing: Extensive data augmentation techniques are applied to make the model more robust.
Model: Pretrained VGG16 is used as the backbone, with a custom dense layer classifier.
Callbacks: Adaptive learning with ReduceLROnPlateau and EarlyStopping ensures efficient training.
Class Weighting: Addresses imbalanced datasets for improved model fairness.
Scalability: Handles chest X-ray images with customizable resolutions.
Model Description
1. Preprocessing Pipeline
Normalization: Pixel values are scaled to the range [0, 1] using rescale=1./255.
Data Augmentation: Enhances generalization by randomly applying:
Rotation: Up to ±20°
Shifts: Horizontal and vertical translations by up to 20%.
Shear Transformations: Up to 20%.
Zoom: Randomly zooms images by up to 20%.
Horizontal Flip: Adds robustness to positional variation.
Brightness Adjustments: Varies brightness from 80% to 120%.
Validation Split: Automatically splits the training data into 80% training and 20% validation sets.
2. Model Architecture
Base Network: The pretrained VGG16 architecture is used with weights initialized from ImageNet.
The convolutional layers are frozen during initial training to preserve learned feature extraction capabilities.
Custom Classifier:
A Flatten layer converts the feature maps into a 1D vector.
A Dense Layer (256 units, ReLU) adds learnable parameters for classification.
A Dropout Layer (50%) prevents overfitting.
A Dense Layer (1 unit, Sigmoid) performs binary classification (Pneumonia vs. Normal).
Why VGG16? The convolutional blocks of VGG16 are excellent feature extractors, especially for image classification tasks. By fine-tuning the last block, we adapt these features specifically for the chest X-ray dataset.

3. Training Strategy
Optimizer: Adam optimizer with an initial learning rate of 1e-4.
Loss Function: Binary Cross-Entropy for classification.
Class Weights: Automatically computed to counter class imbalance.
Callbacks:
ReduceLROnPlateau: Reduces learning rate by a factor of 0.5 if validation loss plateaus for 3 epochs.
EarlyStopping: Stops training early if validation loss does not improve for 5 consecutive epochs.
Training Setup:

Batch Size: 32
Target Image Size: Adjustable to either 128×128 or 150×150 pixels.
Epochs: 20 (adaptive based on callbacks).
4. Post-Training Evaluation
Evaluate the model's performance on a test set.
Metrics:
Accuracy
Loss
Confusion matrix visualization (optional for deeper insights).
Dataset
The dataset includes chest X-ray images split into the following:

Train Directory: Used for training the model with augmentation applied.
Validation Split: Automatically extracted from the training set.
Test Directory: Evaluated post-training to test the model's generalizability.
The images are categorized into:

NORMAL
PNEUMONIA
Results
Performance Metrics:

Training and validation accuracy typically exceed 90% after fine-tuning.
Loss decreases consistently due to robust preprocessing and adaptive callbacks.
Visualization:

Accuracy and loss graphs during training show convergence.

Under Dyne Research

# SignSpeak: American Sign Language Classification with CNNs and LSTMs

Developers: Sahiel Bose & Shanay Gaitonde

**ACSEF ASL Recognition** turns raw video of American Sign Language into text with two purpose‑built notebooks:

| Notebook | Model | What it Learns | Best Test Acc.* |
|----------|-------|----------------|----------------:|
| `ACSEF_CNN.ipynb` | **MobileNetV2‑based CNN** | 26 letters + 10 digits (static hand‑shapes) | **95 %** |
| `ACSEF_LSTM.ipynb` | **Bi‑LSTM sequence model** | 200 common words (8‑frame motion clips) | **90 %** |

\*Measured on held‑out Kaggle data after hyper‑parameter tuning.

---

### How It Works  

| Stage | What Happens |
|-------|--------------|
| **1. Data Prep** | • Kaggle ASL sets are pulled into `/asl_dataset/`  <br>• All images resized to **224 × 224 RGB** <br>• Dataset split 80 / 10 / 10 (train/val/test) and cached with `tf.data`  |
| **2. CNN Pipeline (`ACSEF_CNN`)** | • Frozen **MobileNetV2** backbone → GAP → Dense(36, softmax) <br>• Optimizer **Adam**, LR = 1e‑3, batch = 32 <br>• Early‑stopping & lr‑decay after plateau <br>• Best weights exported to **TFLite** for on‑device use |
| **3. LSTM Pipeline (`ACSEF_LSTM`)** | • Key‑points per frame flattened to 1 530‑feature vectors <br>• Sequence length = 8 frames <br>• Stack: `Mask → Bi‑LSTM(128) × 2 → Dense(128) → Dense(200)` <br>• Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint |
| **4. Evaluation** | • Accuracy, precision, recall, F1  <br>• Confusion matrix heat‑map saved to `results/`  |
| **5. Deployment** | • CNN weights auto‑converted to `my_model.tflite` for mobile/web • LSTM best checkpoint (`best_asl_200_model.h5`) ready for server‑side inference |

---

### Why Two Models?  

* **CNN** nails crisp, single‑frame hand‑shapes—perfect for spelling and digits.  
* **LSTM** watches motion across frames—handy when signs depend on movement (most words).

Use one, the other, or blend their predictions for even better real‑time accuracy.

---
