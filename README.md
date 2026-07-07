****# Calorie Efficiency Prediction using Random Forest Classifier
****
## Project Overview

This project implements a Machine Learning classification model to predict an individual's **Calorie Efficiency** based on health, fitness, and lifestyle metrics. The model has been trained using the **Random Forest Classifier** from Scikit-learn on a large-scale dataset containing **1,000,000 records**.

The objective is to classify individuals into one of three calorie efficiency categories:

* Low Efficiency
* Moderate
* High Efficiency

---

# Dataset Information

### Dataset Size

* Total Records: **1,000,000**
* Features: **14**
* Target Variable: **calorie_efficiency**

### Feature List

| Feature                  | Description                          |
| ------------------------ | ------------------------------------ |
| age                      | Age of the individual                |
| steps_per_day            | Average daily steps                  |
| active_minutes           | Minutes of physical activity per day |
| calories_burned          | Daily calories burned                |
| sleep_hours              | Average sleep duration               |
| hydration_liters         | Daily water intake                   |
| bmi                      | Body Mass Index                      |
| workouts_per_week        | Weekly workout frequency             |
| muscle_mass_ratio        | Muscle mass ratio                    |
| body_fat_percentage      | Body fat percentage                  |
| heart_rate_resting       | Resting heart rate                   |
| heart_rate_avg           | Average heart rate                   |
| continuous_exercise_days | Consecutive exercise days            |
| efficiency_score         | Overall fitness efficiency score     |

---

# Target Class Distribution

The dataset is highly imbalanced.

| Class           | Records |
| --------------- | ------: |
| Low Efficiency  | 938,243 |
| Moderate        |  34,861 |
| High Efficiency |  26,896 |

This imbalance influences the model's predictive performance and should be considered when interpreting the evaluation metrics.

---

# Machine Learning Model

Algorithm:

* Random Forest Classifier

Configuration:

* Number of Trees: 200
* Random State: 42
* Parallel Processing Enabled (`n_jobs=-1`)

---

# Project Workflow

1. Data loading
2. Data preprocessing
3. Label encoding of target variable
4. Train-test split
5. Model training
6. Performance evaluation
7. Prediction

---

# Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib (optional)
* Seaborn (optional)

---

# Model Performance

### Test Accuracy

**94.60%**

### Classification Report

| Class           | Precision | Recall | F1 Score |
| --------------- | --------: | -----: | -------: |
| Moderate        |      0.98 |   0.01 |     0.02 |
| Low Efficiency  |      0.95 |   1.00 |     0.97 |
| High Efficiency |      0.95 |   0.24 |     0.38 |

### Overall Metrics

* Accuracy: **94.60%**
* Weighted F1 Score: **0.93**
* Macro F1 Score: **0.46**

---

# Confusion Matrix

| Actual / Predicted | Moderate | Low Efficiency | High Efficiency |
| ------------------ | -------: | -------------: | --------------: |
| Moderate           |       43 |          5,389 |              49 |
| Low Efficiency     |        0 |        187,507 |              44 |
| High Efficiency    |        1 |          5,316 |           1,651 |

---

# Observations

The model demonstrates strong performance in identifying the majority class (**Low Efficiency**). However, prediction performance for the minority classes (**Moderate** and **High Efficiency**) is considerably lower due to the significant class imbalance in the training dataset.

For production deployment where balanced predictions across all classes are required, additional techniques such as class weighting, oversampling (SMOTE), undersampling, or alternative boosting algorithms may improve minority class performance.

---

# Installation

```bash
pip install -r requirements.txt
```

---

# Run the Training Script

```bash
python train_model.py
```

---

# Generate Predictions

```bash
python predict.py
```

---

# Expected Input

The prediction model expects the following numerical inputs:

* age
* steps_per_day
* active_minutes
* calories_burned
* sleep_hours
* hydration_liters
* bmi
* workouts_per_week
* muscle_mass_ratio
* body_fat_percentage
* heart_rate_resting
* heart_rate_avg
* continuous_exercise_days
* efficiency_score

---

# Output

The model predicts one of the following classes:

* Low Efficiency
* Moderate
* High Efficiency

---

# Future Enhancements

Potential improvements include:

* Addressing class imbalance using SMOTE or class-weighted learning
* Hyperparameter optimization
* Cross-validation
* Feature importance analysis
* Explainable AI using SHAP
* REST API deployment using FastAPI
* Docker containerization
* Cloud deployment

---

# License

This project is intended for educational, research, and demonstration purposes unless otherwise specified.

---

# Author

**Developer:** ____________________

**Date:** ____________________
