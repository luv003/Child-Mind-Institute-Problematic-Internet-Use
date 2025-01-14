# Child-Mind-Institute-Problematic-Internet-Use

# Predicting Problematic Internet Use in Youth

## 🌟 Overview

In today's digital era, excessive internet use among children and adolescents has become a growing concern. Understanding and identifying early signs of problematic internet behavior is crucial for fostering healthier digital habits and ensuring the well-being of our youth. This project aims to develop a predictive model that analyzes children's physical activity and fitness data to detect early indicators of problematic internet use. By leveraging comprehensive datasets and advanced machine learning techniques, we strive to create tools that can help educators, parents, and healthcare professionals intervene proactively.

This project was developed as part of the [Child Mind Institute — Problematic Internet Use](https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use/overview) competition on Kaggle. The competition challenges participants to predict the severity of internet use issues based on physical activity data, contributing to healthier digital habits among youth.


## 📊 Dataset

We utilized the **Healthy Brain Network (HBN)** dataset, which is a rich clinical sample of approximately five thousand individuals aged 5-22. The dataset comprises:

- **Tabular Data**: Includes demographic details, physical measurements, fitness assessments, questionnaires, and behavioral metrics.
- **Actigraphy Series**: Wrist-worn accelerometer data that provides detailed insights into physical activity over extended periods.

The goal is to predict the **Severity Impairment Index (sii)**, which categorizes internet usage into four levels:
- **None (0)**
- **Mild (1)**
- **Moderate (2)**
- **Severe (3)**

Each participant is uniquely identified by an `id`.

## 🛠️ Approach

1. **Data Preprocessing**:
   - **Handling Missing Values**: Applied median imputation for numerical features and constant imputation for categorical ones.
   - **Encoding & Scaling**: Transformed categorical features using ordinal encoding and scaled numerical features with quantile transformers to ensure robust input for the model.

2. **Model Architecture**:
   - **Hybrid Neural Network**: Combines embeddings for categorical data, convolutional and residual blocks for numerical features, and specialized encoders for time-series accelerometer data.
   - **Time-Series Processing**: Utilized convolutional layers and LSTM networks to effectively capture patterns in accelerometer data.
   - **Numerical Feature Encoding**: Employed convolutional and residual blocks to extract meaningful representations from numerical inputs.

3. **Training Strategy**:
   - **Cross-Validation**: Implemented stratified K-Fold cross-validation to maintain balanced class distributions across folds.
   - **Loss Functions**: Combined Cross-Entropy Loss with a custom Quadratic Weighted Kappa (QWK) Loss to directly optimize the evaluation metric.
   - **Learning Rate Scheduling**: Adopted cosine annealing with warmup phases to enhance training efficiency and model performance.

4. **Threshold Optimization**:
   - Fine-tuned prediction thresholds using the Nelder-Mead optimization method to maximize the QWK score, ensuring better alignment between predictions and actual labels.

5. **Ensemble Techniques**:
   - **Neural Networks & LightGBM**: Merged predictions from multiple neural network variants and LightGBM models to boost overall performance and robustness.

## 🏆 Evaluation

The models are evaluated using the **Quadratic Weighted Kappa (QWK)** metric, which measures the agreement between predicted and actual `sii` scores. This metric ranges from 0 (random agreement) to 1 (perfect agreement), with scores below 0 indicating less agreement than expected by chance.

### 📈 Scores

- **Public Score (lb)**: **0.425**
- **Private Score (pb)**: **0.443**

