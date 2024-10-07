import sys
import os
import pandas as pd
import numpy as np
import json
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def train_elasticnet_with_feature_selection(X_train, X_val, y_train, y_val, output_file, feature_selection_method, 
                                            use_feature_selection=True, 
                                            variance_threshold=0.8, verbose=True, n_jobs=-1):
    
    selector = None

    # Feature selection using Lasso (L1 regularization) if enabled
    if use_feature_selection:
        if feature_selection_method == 'lasso':
            # Feature selection using Lasso (L1 regularization)
            lasso = LogisticRegression(penalty='l1', solver='saga', random_state=42, n_jobs=n_jobs, max_iter=500, C=0.1)
            lasso.fit(X_train, y_train)
            
            # Select only the features with non-zero coefficients
            selector = SelectFromModel(lasso, prefit=True, threshold=1e-5)
            X_train = selector.transform(X_train)
            X_val = selector.transform(X_val)
            
            # Log the number of features selected
            with open(f'{output_file}.txt', "a") as f:
                f.write(f'Selected {X_train.shape[1]} features out of {X_train.shape[1]} using Lasso feature selection\n')

        elif feature_selection_method == 'pca':
            # Feature selection using PCA to retain 80% of variance
            pca = PCA(n_components=variance_threshold, random_state=42)
            X_train = pca.fit_transform(X_train)
            X_val = pca.transform(X_val)
            
            # Number of components chosen to explain 80% variance
            n_components_selected = pca.n_components_
            
            # Log the number of components selected
            with open(f'{output_file}.txt', "a") as f:
                f.write(f'Selected {n_components_selected} principal components explaining {variance_threshold * 100}% variance using PCA\n')
            
            # Visualize the first two principal components
            plt.figure(figsize=(10, 7))
            plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', edgecolor='k', s=50)
            plt.xlabel('First Principal Component')
            plt.ylabel('Second Principal Component')
            plt.title('PCA - First two principal components')
            plt.colorbar(label='Class Label')
            plt.savefig(f'{output_file}_pca_visualization.png')
            plt.show()

            # Generate Scree Plot
            plt.figure(figsize=(10, 7))
            plt.plot(np.arange(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, marker='o', linestyle='--')
            plt.xlabel('Principal Component')
            plt.ylabel('Explained Variance Ratio')
            plt.title('Scree Plot')
            plt.savefig(f'{output_file}_scree_plot.png')
            plt.show()

            # Assign PCA object to selector
            selector = pca

        else:
            raise ValueError("Unsupported feature selection method. Use 'lasso' or 'pca'.")
    else:
        selector = None
        
    # Define ElasticNet logistic regression model
    model = LogisticRegression(penalty='elasticnet', solver='saga', 
                               random_state=42, n_jobs=n_jobs, max_iter=500)

    # Set hyperparameter grid for tuning
    param_grid = {
        'l1_ratio': [0.1, 0.5, 0.7, 0.9],  # L1/L2 ratio for elasticnet
        'C': [0.01, 0.1, 1, 10]  # Regularization strength
    }

    # Perform grid search with 5-fold cross-validation using validation set
    grid_search = GridSearchCV(model, param_grid, cv=5, 
                               scoring='accuracy', verbose=1 if verbose else 0, 
                               n_jobs=n_jobs)
    grid_search.fit(X_train, y_train)

    # Best model from grid search
    best_model = grid_search.best_estimator_

    # Training performance
    y_train_pred = best_model.predict(X_train)
    y_train_pred_prob = best_model.predict_proba(X_train)[:, 1]  # Probabilities for ROC-AUC

    # Training metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_roc_auc = roc_auc_score(y_train, y_train_pred_prob)
    train_conf_matrix = confusion_matrix(y_train, y_train_pred)
    train_class_report = classification_report(y_train, y_train_pred, output_dict=True)

    # Write training results to output file
    with open(f'{output_file}.txt', "a") as f:
        f.write(f"\nTraining Accuracy: {train_accuracy:.4f}\n")
        f.write(f"Training ROC-AUC Score: {train_roc_auc:.4f}\n")
        f.write(f"Training Confusion Matrix:\n{train_conf_matrix}\n")
        f.write(f"Training Classification Report:\n{json.dumps(train_class_report, indent=4)}\n")

    # Plot ROC Curve for training set
    fpr, tpr, _ = roc_curve(y_train, y_train_pred_prob)
    plt.figure()
    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (area = {train_roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Training Set Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(f'{output_file}_Training_ROC.png')
    plt.show()


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

    # Return best model, validation metrics, and the selector (if feature selection is used)
    return best_model, val_accuracy, val_roc_auc, selector


# Main function with consistent feature selection/transformation application
def main(input_file, output_file, use_feature_selection, feature_selection_method, verbose=True, n_jobs=-1):
    # Load and preprocess data
    df = pd.read_csv(input_file)
    
    # drop columns starting with 'ADSP'
    for column in df.columns:
        if df[column].astype(str).str.startswith('ADSP').any():
            df = df.drop(columns=[column])
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

    # Train and tune ElasticNet Logistic Regression with or without Feature Selection
    with open(f'{output_file}.txt', "a") as f:
        f.write(f"\nEvaluating ElasticNet Logistic Regression with {'Feature Selection' if use_feature_selection else 'No Feature Selection'} and Hyperparameter Tuning:\n")

    # Train the model and get the selector used
    best_model, val_acc, val_roc_auc, selector = train_elasticnet_with_feature_selection(
        X_train_scaled, X_val_scaled, y_train, y_val, verbose=verbose, n_jobs=n_jobs, output_file=output_file, 
        use_feature_selection=use_feature_selection, feature_selection_method=feature_selection_method, variance_threshold=0.8)

    # Check if selector is correctly initialized
    if use_feature_selection:
        if selector is None:
            raise ValueError(f"Feature selection was requested using '{feature_selection_method}', but the selector is not initialized.")
        else:
            with open(f'{output_file}.txt', "a") as f:
                f.write(f"Feature selection object initialized: {selector}\n")

    # Apply feature selection/transformation on the test set if it was used
    if use_feature_selection and selector is not None:
        if feature_selection_method == 'pca':
            # Apply the same PCA transformation used in training to the test set
            X_test_scaled = selector.transform(X_test_scaled)  # Use the same PCA object returned earlier
        elif feature_selection_method == 'lasso':
            X_test_scaled = selector.transform(X_test_scaled)
    
    # Debugging information to validate feature dimensions
    with open(f'{output_file}.txt', "a") as f:
        f.write(f"Shape of training data: {X_train_scaled.shape}\n")
        f.write(f"Shape of validation data: {X_val_scaled.shape}\n")
        f.write(f"Shape of test data after transformation: {X_test_scaled.shape}\n")

    # Test the best model on the test set
    with open(f'{output_file}.txt', "a") as f:
        f.write("\nEvaluating on Test Set:\n")

    # Ensure the features in the test set match the model's expected input
    try:
        y_test_pred = best_model.predict(X_test_scaled)
    except ValueError as e:
        with open(f'{output_file}.txt', "a") as f:
            f.write(f"Error during prediction: {e}\n")
        raise

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
    plt.title('Test Set Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(f'{output_file}_Test_ROC.png')
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python train_elasticnet_logistic.py <input_csv_file> <output_txt_file> <use_feature_selection> <feature_selection_method>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    use_feature_selection = bool(int(sys.argv[3]))  # Pass 1 for True, 0 for False
    feature_selection_method = str(sys.argv[4])
    main(input_file, output_file, use_feature_selection, feature_selection_method)
