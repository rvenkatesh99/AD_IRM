import sys
import os
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA

def train_svm(X_train, X_test, y_train, y_test, verbose = True):
    model = SVC(kernel='linear', random_state=42, verbose = verbose)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"SVM Accuracy: {accuracy:.4f}")
    return accuracy

def main(input_file, verbose = True, n_jobs=-1):
    # Separate features (X) and target (y)
    df = pd.read_csv(input_file)
    df = df.drop(columns = ['FID', 'SampleID'])
    X = df.iloc[:, :-1].values  # Assuming the last column is the target variable
    y = df.iloc[:, -1].values

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train and evaluate models
    print("Evaluating SVM:")
    svm_acc = train_svm(X_train_scaled, X_test_scaled, y_train, y_test, verbose=verbose)
    
    print("\nSummary of Model Accuracies:")
    print(f"SVM Accuracy: {svm_acc:.4f}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train_elasticnet_logistic.py <input_csv_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    main(input_file)
