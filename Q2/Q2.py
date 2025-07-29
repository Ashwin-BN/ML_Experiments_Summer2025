import os
import cv2
import numpy as np
import glob
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

# Set the target size to which all images will be resized
IMG_SIZE = (64, 64)

def load_data(folder_path):
    """
    Loads image data from a specified folder structure.
    Each class should have its own subfolder containing .jpg images.

    Parameters:
        folder_path (str): Path to the dataset folder ('train' or 'test').

    Returns:
        X (np.array): Flattened and normalized image feature vectors.
        y (np.array): Corresponding class labels.
    """
    X = []
    y = []
    files = glob.glob(f'{folder_path}/*/*.jpg')
    valid_count = 0

    for filepath in files:
        img = cv2.imread(filepath)
        if img is None:
            continue  # Skip unreadable files

        # Resize, normalize, and flatten the image
        img = cv2.resize(img, IMG_SIZE)
        img = img.flatten() / 255.0

        X.append(img)

        # Extract the label from the folder name
        label = filepath.split(os.sep)[-2].lower()
        y.append(label)

        valid_count += 1

    print(f"Found {len(files)} image files in {folder_path}")
    print(f"Loaded {valid_count} valid images from {folder_path}")
    return np.array(X), np.array(y)


# Load and preprocess training and testing datasets
print("Loading training data...")
X_train, y_train = load_data('train')

print("Loading test data...")
X_test, y_test = load_data('test')

# Encode string labels into numeric format
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

# Standardize the feature vectors
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train and evaluate Logistic Regression model
print("Training Logistic Regression...")
model_lr = LogisticRegression(max_iter=1000, random_state=42)
model_lr.fit(X_train_scaled, y_train_enc)
pred_lr = model_lr.predict(X_test_scaled)
acc_lr = accuracy_score(y_test_enc, pred_lr)
print(f"Logistic Regression Accuracy: {acc_lr:.4f}")

# Train and evaluate K-Nearest Neighbors model
print("Training KNN...")
model_knn = KNeighborsClassifier(n_neighbors=5)
model_knn.fit(X_train_scaled, y_train_enc)
pred_knn = model_knn.predict(X_test_scaled)
acc_knn = accuracy_score(y_test_enc, pred_knn)
print(f"KNN Accuracy: {acc_knn:.4f}")

# Train and evaluate Perceptron model
print("Training Perceptron...")
model_perc = Perceptron(max_iter=1000, tol=1e-3, random_state=42)
model_perc.fit(X_train_scaled, y_train_enc)
pred_perc = model_perc.predict(X_test_scaled)
acc_perc = accuracy_score(y_test_enc, pred_perc)
print(f"Perceptron Accuracy: {acc_perc:.4f}")

# Determine the best-performing model based on test accuracy
accuracies = {
    'logistic_regression': acc_lr,
    'knn': acc_knn,
    'perceptron': acc_perc
}
best_model_name = max(accuracies, key=accuracies.get)
print(f"Best model: {best_model_name} with accuracy {accuracies[best_model_name]:.4f}")

# Create directory for saving models
os.makedirs('models', exist_ok=True)

# Save label encoder and feature scaler
joblib.dump(le, 'models/label_encoder.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

# Save all trained models
joblib.dump(model_lr, 'models/logistic_regression_model.pkl')
joblib.dump(model_knn, 'models/knn_model.pkl')
joblib.dump(model_perc, 'models/perceptron_model.pkl')

# Save the best model with a generic filename for easy access
best_model_map = {
    'logistic_regression': model_lr,
    'knn': model_knn,
    'perceptron': model_perc
}
joblib.dump(best_model_map[best_model_name], 'models/image_model.pkl')

print("All models saved. Best model saved as image_model.pkl")
