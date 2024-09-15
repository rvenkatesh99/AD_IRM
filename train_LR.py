import sys
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def train_elasticnet(X_train, X_val, y_train, y_val, verbose=True, n_jobs=-1):
    # Define ElasticNet logistic regression model
    model = LogisticRegression(penalty='elasticnet', solver='saga', 
                               random_state=42, n_jobs=n_jobs, max_iter=200)

    # Set hyperparameter grid for tuning
    param_grid = {
        'l1_ratio': [0.1, 0.5, 0.7, 0.9],  # L1/L2 ratio for elasticnet
        'C': [0.01, 0.1, 1, 10]  # Regularization strength
    }

    # Perform grid search with 5-fold cross-validation using validation set
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', verbose=1 if verbose else 0, n_jobs=n_jobs)
    grid_search.fit(X_train, y_train)

    # Best model from grid search
    best_model = grid_search.best_estimator_

    # Validate the best model on the validation set
    y_val_pred = best_model.predict(X_val)
    y_val_pred_prob = best_model.predict_proba(X_val)[:, 1]  # Probabilities for ROC-AUC

    # Validation metrics
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_roc_auc = roc_auc_score(y_val, y_val_pred_prob)
    conf_matrix = confusion_matrix(y_val, y_val_pred)
    class_report = classification_report(y_val, y_val_pred, output_dict=True)

    # Print validation results
    print(f"Best Model from GridSearch: {grid_search.best_params_}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Validation ROC-AUC Score: {val_roc_auc:.4f}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Classification Report:\n{classification_report(y_val, y_val_pred)}")

    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_val, y_val_pred_prob)
    plt.figure()
    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (area = {val_roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Validation Set Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('Validation_ROC.png')
    plt.show()

    # Return best model and validation metrics
    return best_model, val_accuracy, val_roc_auc


def main(input_file, verbose=True, n_jobs=-1):
    # Load and preprocess data
    df = pd.read_csv(input_file)
    df = df.drop(columns=['FID', 'SampleID'])
    X = df.iloc[:, :-1].values  # last column is the target
    y = df.iloc[:, -1].values

    # Split the data into training+validation and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Split the training+validation set into separate training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

    print('Train-test-validation split complete')

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    print('Data scaled')

    # Train and tune ElasticNet Logistic Regression
    print("\nEvaluating ElasticNet Logistic Regression with Hyperparameter Tuning:")
    best_model, val_acc, val_roc_auc = train_elasticnet(X_train_scaled, X_val_scaled, y_train, y_val, verbose=verbose, n_jobs=n_jobs)

    # Test the best model on the test set
    print("\nEvaluating on Test Set:")
    y_test_pred = best_model.predict(X_test_scaled)
    y_test_pred_prob = best_model.predict_proba(X_test_scaled)[:, 1]
    
    # Test set metrics
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_roc_auc = roc_auc_score(y_test, y_test_pred_prob)
    test_conf_matrix = confusion_matrix(y_test, y_test_pred)
    test_class_report = classification_report(y_test, y_test_pred)

    # Print test results
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test ROC-AUC Score: {test_roc_auc:.4f}")
    print(f"Test Confusion Matrix:\n{test_conf_matrix}")
    print(f"Test Classification Report:\n{test_class_report}")

    # Plot ROC Curve for test set
    fpr, tpr, _ = roc_curve(y_test, y_test_pred_prob)
    plt.figure()
    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (area = {test_roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Test Set Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('Test_ROC.png')
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train_elasticnet_logistic.py <input_csv_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    main(input_file)