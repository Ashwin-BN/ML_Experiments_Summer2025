# ML\_Experiments\_Summer2025

This repository contains solutions for three machine learning experiments involving regression and classification tasks using classical ML algorithms. The datasets and tasks are organized into three folders: Q1, Q2, and Q3.

---

## Overview

### Q1: House Price Estimation

* **Task:** Predict house prices based on two features — number of bedrooms and basement area.
* **Methods:** Multiple Linear Regression and SGD Regressor.
* **Metrics:** MAE, MSE, RMSE, MAPE.
* **Summary:** Both models yielded similar performance; Linear Regression showed slightly better accuracy. SGD Regressor is advantageous for large or streaming data.
* **Key Insight:** RMSE is sensitive to large errors while MAE provides balanced average error. MAPE gives percentage-based error, aiding interpretation.

---

### Q2: Cat and Dog Image Classification

* **Task:** Classify images of cats and dogs using classical classifiers.
* **Methods:** Logistic Regression, K-Nearest Neighbors (KNN), Perceptron.
* **Preprocessing:** Images resized to 64×64, normalized, flattened, and scaled.
* **Results:**

  * Logistic Regression: 60% accuracy (best on test set).
  * KNN and Perceptron: 50% accuracy.
* **Real-world Testing:** The model performed variably on internet images, occasionally misclassifying cats as dogs and vice versa.
* **Analysis:** Logistic Regression performed best overall but struggled on unseen real-world images. Other classifiers showed inconsistent but complementary predictions.
* **Conclusion:** More robust features or advanced models may improve real-world generalization.

---

### Q3: MNIST Digit Classification

* **Task:** Classify handwritten digits from the MNIST dataset using classical ML algorithms.
* **Methods:** Logistic Regression, K-Nearest Neighbors (k=3), Perceptron.
* **Data:** 60,000 training and 10,000 testing flattened grayscale images. Normalized pixel values between 0 and 1.
* **Results:**

  * KNN: 97.05% accuracy (best performer).
  * Logistic Regression: 92.62% accuracy.
  * Perceptron: 86.33% accuracy.
* **Analysis:** KNN outperformed others, showing strong instance-based learning effectiveness. Logistic Regression also surpassed the 90% accuracy goal. Perceptron’s simpler linear approach resulted in lower accuracy.
* **Conclusion:** Classical ML algorithms can reliably classify MNIST digits without deep learning, with KNN achieving the highest accuracy.

---

## Repository Structure

* **Q1/** — House price regression tasks and reports
* **Q2/** — Cat vs. Dog classification scripts, models, and reports
* **Q3/** — MNIST classification scripts, models, and reports

Each folder contains the corresponding Jupyter Notebook files with detailed code, explanations, and results.
