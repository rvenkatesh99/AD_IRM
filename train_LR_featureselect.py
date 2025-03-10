import sys
import os
import pandas as pd
import numpy as np
import json
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, f1_score, \
    balanced_accuracy_score, average_precision_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def train_elasticnet_with_feature_selection(X_train, X_val, X_test, y_train, y_val, y_test, output_file, feature_selection_method, 
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
            X_test = selector.transform(X_test)
            
            # Log the number of features selected
            with open(f'{output_file}.txt', "a") as f:
                f.write(f'Selected {X_train.shape[1]} features out of {X_train.shape[1]} using Lasso feature selection\n')

        elif feature_selection_method == 'pca':
            # Feature selection using PCA to retain 80% of variance
            pca = PCA(n_components=variance_threshold, random_state=42)
            X_train = pca.fit_transform(X_train)
            X_val = pca.transform(X_val)
            X_test = pca.transform(X_test)
            
            # Number of components chosen to explain 80% variance
            n_components_selected = pca.n_components_
            
            # Log the number of components selected
            with open(f'{output_file}.txt', "a") as f:
                f.write(f'Selected {n_components_selected} principal components explaining {variance_threshold * 100}% variance using PCA\n')

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
    train_f1 = f1_score(y_train, y_train_pred)
    train_balanced_acc = balanced_accuracy_score(y_train, y_train_pred)
    train_auprc = average_precision_score(y_train, y_train_pred_prob)
    train_conf_matrix = confusion_matrix(y_train, y_train_pred)
    train_class_report = classification_report(y_train, y_train_pred, output_dict=True)
    train_precision = precision_score(y_train, y_train_pred)
    train_recall = recall_score(y_train, y_train_pred)

    # Write training results to output file
    with open(f'{output_file}.txt', "a") as f:
        f.write(f"\nTraining Accuracy: {train_accuracy:.4f}\n")
        f.write(f"Training ROC-AUC Score: {train_roc_auc:.4f}\n")
        f.write(f"Training F1 Score: {train_f1:.4f}\n")
        f.write(f"Training Balanced Accuracy: {train_balanced_acc:.4f}\n")
        f.write(f"Training AUPRC: {train_auprc:.4f}\n")
        f.write(f"Training Confusion Matrix:\n{train_conf_matrix}\n")
        f.write(f"Training Classification Report:\n{json.dumps(train_class_report, indent=4)}\n")
        f.write(f"Training Precision: {train_precision:.4f}\n")
        f.write(f"Training Recall: {train_recall:.4f}\n")

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
    val_f1 = f1_score(y_val, y_val_pred)
    val_balanced_acc = balanced_accuracy_score(y_val, y_val_pred)
    val_auprc = average_precision_score(y_val, y_val_pred_prob)
    val_conf_matrix = confusion_matrix(y_val, y_val_pred)
    val_class_report = classification_report(y_val, y_val_pred, output_dict=True)
    val_precision = precision_score(y_val, y_val_pred)
    val_recall = recall_score(y_val, y_val_pred)

    # Write validation results to output file
    with open(f'{output_file}.txt', "a") as f:
        f.write(f"Validation Accuracy: {val_accuracy:.4f}\n")
        f.write(f"Validation ROC-AUC Score: {val_roc_auc:.4f}\n")
        f.write(f"Validation F1 Score: {val_f1:.4f}\n")
        f.write(f"Validation Balanced Accuracy: {val_balanced_acc:.4f}\n")
        f.write(f"Validation AUPRC: {val_auprc:.4f}\n")
        f.write(f"Validation Confusion Matrix:\n{val_conf_matrix}\n")
        f.write(f"Validation Classification Report:\n{json.dumps(val_class_report, indent=4)}\n")
        f.write(f"Validation Precision: {val_precision:.4f}\n")
        f.write(f"Validation Recall: {val_recall:.4f}\n")

    # Plot ROC Curve for validation set
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

    # Test the best model on the test set
    y_test_pred = best_model.predict(X_test)
    y_test_pred_prob = best_model.predict_proba(X_test)[:, 1]

    # Test set metrics
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_roc_auc = roc_auc_score(y_test, y_test_pred_prob)
    test_f1 = f1_score(y_test, y_test_pred)
    test_balanced_acc = balanced_accuracy_score(y_test, y_test_pred)
    test_auprc = average_precision_score(y_test, y_test_pred_prob)
    test_conf_matrix = confusion_matrix(y_test, y_test_pred)
    test_class_report = classification_report(y_test, y_test_pred, output_dict=True)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)

    # Write test results to output file
    with open(f'{output_file}.txt', "a") as f:
        f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
        f.write(f"Test ROC-AUC Score: {test_roc_auc:.4f}\n")
        f.write(f"Test F1 Score: {test_f1:.4f}\n")
        f.write(f"Test Balanced Accuracy: {test_balanced_acc:.4f}\n")
        f.write(f"Test AUPRC: {test_auprc:.4f}\n")
        f.write(f"Test Confusion Matrix:\n{test_conf_matrix}\n")
        f.write(f"Test Classification Report:\n{json.dumps(test_class_report, indent=4)}\n")
        f.write(f"Test Precision: {test_precision:.4f}\n")
        f.write(f"Test Recall: {test_recall:.4f}\n")

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

    return (train_accuracy, train_roc_auc, train_f1, train_balanced_acc, train_auprc, train_precision, train_recall,
            val_accuracy, val_roc_auc, val_f1, val_balanced_acc, val_auprc, val_precision, val_recall,
            test_accuracy, test_roc_auc, test_f1, test_balanced_acc, test_auprc, test_precision, test_recall,
            best_model, selector)



# Main function to run multiple iterations with different splits
def train_multiple_iterations(X, y, output_file, feature_selection_method, 
                              n_iterations=10, use_feature_selection=True, variance_threshold=0.8, verbose=True, test_size=0.2):
    results = []

    # Initialize cumulative metrics
    cumulative_train_accuracy = 0
    cumulative_train_roc_auc = 0
    cumulative_train_f1 = 0
    cumulative_train_balanced_acc = 0
    cumulative_train_auprc = 0
    cumulative_train_precision = 0
    cumulative_train_recall = 0
    
    cumulative_val_accuracy = 0
    cumulative_val_roc_auc = 0
    cumulative_val_f1 = 0
    cumulative_val_balanced_acc = 0
    cumulative_val_auprc = 0
    cumulative_val_precision = 0
    cumulative_val_recall = 0
    
    cumulative_test_accuracy = 0
    cumulative_test_roc_auc = 0
    cumulative_test_f1 = 0
    cumulative_test_balanced_acc = 0
    cumulative_test_auprc = 0
    cumulative_test_precision = 0
    cumulative_test_recall = 0

    for iteration in range(1, n_iterations + 1):
        # Split the data into training, validation, and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=iteration)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=iteration) # 0.25 x 0.8 = 0.2
         
        # Standardize the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        # Train the model and get metrics
        (train_accuracy, train_roc_auc, train_f1, train_balanced_acc, train_auprc, train_precision, train_recall,
         val_accuracy, val_roc_auc, val_f1, val_balanced_acc, val_auprc, val_precision, val_recall,
         test_accuracy, test_roc_auc, test_f1, test_balanced_acc, test_auprc, test_precision, test_recall,
         best_model, selector) = \
            train_elasticnet_with_feature_selection(X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, output_file, 
                                                    feature_selection_method, use_feature_selection, 
                                                    variance_threshold, verbose)
        
        # Accumulate metrics for averaging
        cumulative_train_accuracy += train_accuracy
        cumulative_train_roc_auc += train_roc_auc
        cumulative_train_f1 += train_f1
        cumulative_train_balanced_acc += train_balanced_acc
        cumulative_train_auprc += train_auprc
        cumulative_train_precision += train_precision
        cumulative_train_recall += train_recall
        
        cumulative_val_accuracy += val_accuracy
        cumulative_val_roc_auc += val_roc_auc
        cumulative_val_f1 += val_f1
        cumulative_val_balanced_acc += val_balanced_acc
        cumulative_val_auprc += val_auprc
        cumulative_val_precision += val_precision
        cumulative_val_recall += val_recall
        
        cumulative_test_accuracy += test_accuracy
        cumulative_test_roc_auc += test_roc_auc
        cumulative_test_f1 += test_f1
        cumulative_test_balanced_acc += test_balanced_acc
        cumulative_test_auprc += test_auprc
        cumulative_test_precision += test_precision
        cumulative_test_recall += test_recall

        # Store the results for this iteration
        results.append({
            'iteration': iteration,
            'train_accuracy': train_accuracy,
            'train_roc_auc': train_roc_auc,
            'train_f1': train_f1,
            'train_balanced_acc': train_balanced_acc,
            'train_auprc': train_auprc,
            'train_precision': train_precision,
            'train_recall': train_recall,
            'val_accuracy': val_accuracy,
            'val_roc_auc': val_roc_auc,
            'val_f1': val_f1,
            'val_balanced_acc': val_balanced_acc,
            'val_auprc': val_auprc,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'test_accuracy': test_accuracy,
            'test_roc_auc': test_roc_auc,
            'test_f1': test_f1,
            'test_balanced_acc': test_balanced_acc,
            'test_auprc': test_auprc,
            'test_precision': test_precision,
            'test_recall': test_recall
        })

    # Calculate averages
    avg_train_accuracy = cumulative_train_accuracy / n_iterations
    avg_train_roc_auc = cumulative_train_roc_auc / n_iterations
    avg_train_f1 = cumulative_train_f1 / n_iterations
    avg_train_balanced_acc = cumulative_train_balanced_acc / n_iterations
    avg_train_auprc = cumulative_train_auprc / n_iterations
    avg_train_precision = cumulative_train_precision / n_iterations
    avg_train_recall = cumulative_train_recall / n_iterations

    avg_val_accuracy = cumulative_val_accuracy / n_iterations
    avg_val_roc_auc = cumulative_val_roc_auc / n_iterations
    avg_val_f1 = cumulative_val_f1 / n_iterations
    avg_val_balanced_acc = cumulative_val_balanced_acc / n_iterations
    avg_val_auprc = cumulative_val_auprc / n_iterations
    avg_val_precision = cumulative_val_precision / n_iterations
    avg_val_recall = cumulative_val_recall / n_iterations

    avg_test_accuracy = cumulative_test_accuracy / n_iterations
    avg_test_roc_auc = cumulative_test_roc_auc / n_iterations
    avg_test_f1 = cumulative_test_f1 / n_iterations
    avg_test_balanced_acc = cumulative_test_balanced_acc / n_iterations
    avg_test_auprc = cumulative_test_auprc / n_iterations
    avg_test_precision = cumulative_test_precision / n_iterations
    avg_test_recall = cumulative_test_recall / n_iterations

    # Store averages in the results as the last entry
    results.append({
        'iteration': 'Average',
        'train_accuracy': avg_train_accuracy,
        'train_roc_auc': avg_train_roc_auc,
        'train_f1': avg_train_f1,
        'train_balanced_acc': avg_train_balanced_acc,
        'train_auprc': avg_train_auprc,
        'train_precision': avg_train_precision,
        'train_recall': avg_train_recall,
        'val_accuracy': avg_val_accuracy,
        'val_roc_auc': avg_val_roc_auc,
        'val_f1': avg_val_f1,
        'val_balanced_acc': avg_val_balanced_acc,
        'val_auprc': avg_val_auprc,
        'val_precision': avg_val_precision,
        'val_recall': avg_val_recall,
        'test_accuracy': avg_test_accuracy,
        'test_roc_auc': avg_test_roc_auc,
        'test_f1': avg_test_f1,
        'test_balanced_acc': avg_test_balanced_acc,
        'test_auprc': avg_test_auprc,
        'test_precision': avg_test_precision,
        'test_recall': avg_test_recall
    })

    # Convert results to a DataFrame
    df_results = pd.DataFrame(results)

    # Save to CSV
    df_results.to_csv(f'{output_file}.csv', index=False)

    return df_results

def main(input_data_file, output_file, feature_selection_method, n_iterations=10, test_size=0.2, use_feature_selection=True, variance_threshold=0.8):
    # Load input data (assuming input_data_file is a CSV with X as features and y as target)
    df = pd.read_csv(input_data_file)

    for column in df.columns:
        if df[column].astype(str).str.startswith('A').any():
            df = df.drop(columns=[column])

    # Assume the last column is the target variable and the rest are features
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    # Call the train_multiple_iterations function
    results_df = train_multiple_iterations(X, y, output_file, feature_selection_method, 
                                           n_iterations=n_iterations, 
                                           use_feature_selection=use_feature_selection, 
                                           variance_threshold=variance_threshold, 
                                           test_size=test_size)
    
    # Output the results as a confirmation
    print(f'Results saved to {output_file}.csv')


if __name__ == "__main__":
    # Make sure the right number of arguments is passed
    if len(sys.argv) < 5:
        print("Usage: python script_name.py <input_csv_file> <output_file> <feature_selection_method> <use_feature_selection> [n_iterations] [test_size] [variance_threshold]")
        sys.exit(1)

    # Required arguments from the command line
    input_data_file = sys.argv[1]
    output_file = sys.argv[2]
    use_feature_selection = bool(int(sys.argv[3])) # 1 or 0 for True/False
    feature_selection_method = sys.argv[4]  

    # Optional arguments with default values
    n_iterations = int(sys.argv[5]) if len(sys.argv) > 5 else 10
    test_size = float(sys.argv[6]) if len(sys.argv) > 6 else 0.2
    variance_threshold = float(sys.argv[7]) if len(sys.argv) > 7 else 0.8
    
    # Call the main function with command-line arguments
    main(input_data_file, output_file, feature_selection_method, n_iterations, test_size, use_feature_selection, variance_threshold)
