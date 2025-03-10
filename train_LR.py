import sys
import os
import pandas as pd
import numpy as np
import json
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def train_elasticnet_with_feature_selection(X_train, X_val, y_train, y_val, output_file, use_feature_selection=True, verbose=True, n_jobs=-1):
    # Feature selection using Lasso (L1 regularization) if enabled
    if use_feature_selection:
        lasso = LogisticRegression(penalty='l1', solver='saga', random_state=42, n_jobs=n_jobs, max_iter=500, C=0.01)
        lasso.fit(X_train, y_train)
        
        # Select only features with non-zero coefficients
        selector = SelectFromModel(lasso, prefit=True, threshold=1e-4)
        X_train = selector.transform(X_train)
        X_val = selector.transform(X_val)
        
        # Log the number of features selected
        with open(f'{output_file}.txt', "a") as f:
            f.write(f'Selected {X_train.shape[1]} features out of {X_train.shape[1]} using Lasso feature selection\n')
    else:
        selector = None  # No feature selection
    
    # Define ElasticNet logistic regression model
    model = LogisticRegression(penalty='elasticnet', solver='saga', random_state=42, n_jobs=n_jobs, max_iter=500)

    # Set hyperparameter grid for tuning
    param_grid = {
        'l1_ratio': [0.1, 0.5, 0.7, 0.9],  # L1/L2 ratio for elasticnet
        'C': [0.01, 0.1, 1, 10]  # Regularization strength
    }

    # Perform grid search with 5-fold cross-validation
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

    # Write validation results to output file
    with open(f'{output_file}.txt', "a") as f:
        f.write(f"Best Model from GridSearch: {grid_search.best_params_}\n")
        f.write(f"Validation Accuracy: {val_accuracy:.4f}\n")
        f.write(f"Validation ROC-AUC Score: {val_roc_auc:.4f}\n")
        f.write(f"Confusion Matrix:\n{conf_matrix}\n")
        f.write(f"Classification Report:\n{json.dumps(class_report, indent=4)}\n")

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
    plt.savefig(f'{output_file}_Validation_ROC.png')
    plt.show()

    return best_model, val_accuracy, val_roc_auc, selector


def main(input_file, output_file, verbose=True, n_jobs=-1):
    # Load and preprocess data
    df = pd.read_csv(input_file)
    df = df.drop(columns=['FID', 'SampleID'])
    X = df.iloc[:, :-1].values  # Last column is the target
    y = df.iloc[:, -1].values

    # Split the data into training+validation and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Split the training+validation set into separate training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

    with open(f'{output_file}.txt', "w") as f:
        f.write('Train-test-validation split complete\n')

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    with open(f'{output_file}.txt', "a") as f:
        f.write('Data scaled\n')

    # Compare performance with and without feature selection
    for use_feature_selection in [False, True]:
        feature_select_str = "with" if use_feature_selection else "without"
        with open(f'{output_file}.txt', "a") as f:
            f.write(f"\nEvaluating ElasticNet Logistic Regression {feature_select_str} Feature Selection:\n")
        
        best_model, val_acc, val_roc_auc, selector = train_elasticnet_with_feature_selection(
            X_train_scaled, X_val_scaled, y_train, y_val, output_file, 
            use_feature_selection=use_feature_selection, verbose=verbose, n_jobs=n_jobs
        )

        # Test the best model on the test set
        y_test_pred = best_model.predict(X_test_scaled)
        y_test_pred_prob = best_model.predict_proba(X_test_scaled)[:, 1]
        
        # Test set metrics
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_roc_auc = roc_auc_score(y_test, y_test_pred_prob)
        test_conf_matrix = confusion_matrix(y_test, y_test_pred)
        test_class_report = classification_report(y_test, y_test_pred, output_dict=True)

        # Write test results to output file
        with open(f'{output_file}.txt', "a") as f:
            f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
            f.write(f"Test ROC-AUC Score: {test_roc_auc:.4f}\n")
            f.write(f"Test Confusion Matrix:\n{test_conf_matrix}\n")
            f.write(f"Test Classification Report:\n{json.dumps(test_class_report, indent=4)}\n")

        # Plot ROC Curve for test set
        fpr, tpr, _ = roc_curve(y_test, y_test_pred_prob)
        plt.figure()
        plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (area = {test_roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='red', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Test Set Receiver Operating Characteristic {feature_select_str} Feature Selection')
        plt.legend(loc="lower right")
        plt.savefig(f'{output_file}_Test_ROC_{feature_select_str}_FS.png')
        plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python train_elasticnet_logistic.py <input_csv_file> <output_txt_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    main(input_file, output_file)
