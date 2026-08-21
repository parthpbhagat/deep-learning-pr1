# Breast Cancer Classification using Deep Learning

Welcome to the **Breast Cancer Classification** project! This repository contains an end-to-end deep learning pipeline to classify breast cancer tumors as either **Malignant (0)** or **Benign (1)** using neural networks.

## 🎯 Project Objective
The primary goal of this project is to build a highly robust and accurate deep learning model to assist in medical diagnosis. In the medical field, **False Negatives** (failing to detect a malignant tumor) can be fatal. Therefore, a major focus of this project is maximizing **Recall** for the malignant class and appropriately tuning the classification threshold.

## 📊 Dataset
This project uses the famous **Wisconsin Breast Cancer Dataset** loaded directly via Scikit-Learn.
- **Total Samples:** 569 patients
- **Features:** 30 numeric features (e.g., radius, perimeter, area, smoothness, etc.)
- **Target:** Malignant (0) vs. Benign (1)

## 🛠️ Implementation Steps & Methodology

### 1. Exploratory Data Analysis (EDA) & Preprocessing
- **Class Imbalance:** Observed a mild class imbalance (212 Malignant vs. 357 Benign).
- **Correlation:** Visualized feature relationships using a Seaborn heatmap, identifying highly correlated feature pairs (multicollinearity).
- **Feature Scaling:** Applied `StandardScaler` to normalize the input features. Neural networks converge much faster and more stably when all inputs share the same scale.

### 2. Baseline Model: Single-Layer Perceptron (SLP)
- Built a simple SLP (equivalent to Logistic Regression) with 30 inputs and 1 output using a `Sigmoid` activation function.
- Served as a solid baseline, but struggled to capture complex, non-linear boundaries perfectly.

### 3. Advanced Model: Multi-Layer Perceptron (MLP)
- Expanded the architecture by adding multiple Hidden Layers (64 and 32 neurons) to learn complex, non-linear feature interactions.
- **Activation Functions:** Compared `ReLU`, `Tanh`, and `Sigmoid` in the hidden layers. **ReLU** performed the best by ensuring fast convergence and avoiding the vanishing gradient problem.

### 4. Preventing Overfitting
Deep learning models can easily overfit on small datasets (569 samples). The following techniques were implemented to ensure the model generalizes well to unseen data:
- **Early Stopping:** Monitored the validation loss (`val_loss`). Training was halted automatically if the loss didn't improve for 15-20 epochs. Crucially, the model's *best weights* were restored at the end.
- **Dropout Layers:** Introduced a 30% dropout rate (`Dropout(0.3)`). This randomly disables neurons during training, acting as an implicit ensemble technique and preventing neuron co-adaptation.
- **Regularization:** Experimented with **L1 (Lasso)**, **L2 (Ridge)**, and **ElasticNet (L1+L2)** weight penalties. L2 regularization provided the smoothest decision boundary and lowest variance.

### 5. The Final Combined Model
The final model architecture combines the best of all worlds:
- Deep Hidden Layers (128 -> 64)
- ReLU Activations
- L2 Regularization
- Dropout Layers (0.3 rate)
- Early Stopping

## 💡 Clinical & Business Insights
- **Threshold Tuning:** The default classification threshold of 0.5 was analyzed. Given the high cost of false negatives in cancer diagnosis, it is clinically recommended to lower the threshold (e.g., to 0.2 or 0.3) to aggressively catch malignant cases.
- **Deep Learning Efficacy:** The project demonstrates that with proper regularization, deep neural networks vastly outperform basic linear models on complex medical diagnostic data.

## 🚀 How to Run
1. Ensure you have the required libraries installed:
   ```bash
   pip install -r requirements.txt
   ```
2. Open and run the Jupyter Notebook:
   ```bash
   jupyter notebook deep_learning_pr1.ipynb
   ```
   *Note: All markdown explanations and visualizations are embedded directly in the notebook!*
