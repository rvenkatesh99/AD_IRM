import sys
import time
import os
import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, \
    f1_score, balanced_accuracy_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def train_random_forest_with_feature_selection(X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, output_file, feature_selection_method, 
                                            use_feature_selection=True, 
                                            variance_threshold=0.8, verbose=True, n_jobs=-1):
    selector = None

    # Feature selection using Lasso (L1 regularization) if enabled
    if use_feature_selection:
        if feature_selection_method == 'pca':
            # Feature selection using PCA to retain 80% of variance
            pca = PCA(n_components=variance_threshold, random_state=42)
            X_train_scaled = pca.fit_transform(X_train_scaled)
            X_val_scaled = pca.transform(X_val_scaled)
            X_test_scaled = pca.transform(X_test_scaled)
            
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

    # Feature selection using Random Forest feature importances
    feature_selector = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    feature_selector.fit(X_train_scaled, y_train)

    # Select important features based on feature importance
    selector = SelectFromModel(feature_selector, threshold="median", prefit=True)
    X_train_selected = selector.transform(X_train_scaled)
    X_val_selected = selector.transform(X_val_scaled)
    X_test_selected = selector.transform(X_test_scaled)

    n_selected_features = X_train_selected.shape[1]
    print("Number of Features:", n_selected_features)
    with open(f'{output_file}.txt', "a") as f:
        f.write(f'Selected {n_selected_features} features\n')

    # Train Random Forest on selected features
    model = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
    
    # Hyperparameter tuning grid
    param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 8, 10],
    'min_samples_split':  [20, 50, 100],
    'min_samples_leaf': [10, 20, 50],
    'max_features': ['sqrt', 'log2', None]  # Limits the number of features considered at each split
    }

    # Perform randomized search with 5-fold cross-validation
    # grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5,
    #                            scoring='roc_auc', verbose=1 if verbose else 0, n_jobs=-1)
    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, cv=5, random_state=42,
                               scoring='roc_auc', verbose=1 if verbose else 0, n_jobs=-1)
    random_search.fit(X_train_selected, y_train)

    # Best model from grid search
    best_model = random_search.best_estimator_
    print("Best Parameters:", random_search.best_params_)

    with open(f'{output_file}.txt', "a") as f:
        f.write(f"Best Parameters: {random_search.best_params_}")

    # Training performance
    y_train_pred = best_model.predict(X_train_selected)
    y_train_pred_prob = best_model.predict_proba(X_train_selected)[:, 1]  # Probabilities for ROC-AUC

    # Training metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_roc_auc = roc_auc_score(y_train, y_train_pred_prob)
    train_f1 = f1_score(y_train, y_train_pred)
    train_balanced_acc = balanced_accuracy_score(y_train, y_train_pred)
    train_auprc = average_precision_score(y_train, y_train_pred_prob)
    train_conf_matrix = confusion_matrix(y_train, y_train_pred)
    train_class_report = classification_report(y_train, y_train_pred, output_dict=True)

    # Write training results to output file
    with open(f'{output_file}.txt', "a") as f:
        f.write(f"\nTraining Accuracy: {train_accuracy:.4f}\n")
        f.write(f"Training ROC-AUC Score: {train_roc_auc:.4f}\n")
        f.write(f"Training F1 Score: {train_f1:.4f}\n")
        f.write(f"Training Balanced Accuracy: {train_balanced_acc:.4f}\n")
        f.write(f"Training AUPRC: {train_auprc:.4f}\n")
        f.write(f"Training Confusion Matrix:\n{train_conf_matrix}\n")
        f.write(f"Training Classification Report:\n{json.dumps(train_class_report, indent=4)}\n")

    # Validate the best model on the validation set
    y_val_pred = best_model.predict(X_val_selected)
    y_val_pred_prob = best_model.predict_proba(X_val_selected)[:, 1]  # Probabilities for ROC-AUC

    # Validation metrics
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_roc_auc = roc_auc_score(y_val, y_val_pred_prob)
    val_f1 = f1_score(y_val, y_val_pred)
    val_balanced_acc = balanced_accuracy_score(y_val, y_val_pred)
    val_auprc = average_precision_score(y_val, y_val_pred_prob)
    val_conf_matrix = confusion_matrix(y_val, y_val_pred)
    val_class_report = classification_report(y_val, y_val_pred, output_dict=True)

    # Write validation results to output file
    with open(f'{output_file}.txt', "a") as f:
        f.write(f"Validation Accuracy: {val_accuracy:.4f}\n")
        f.write(f"Validation ROC-AUC Score: {val_roc_auc:.4f}\n")
        f.write(f"Validation F1 Score: {val_f1:.4f}\n")
        f.write(f"Validation Balanced Accuracy: {val_balanced_acc:.4f}\n")
        f.write(f"Validation AUPRC: {val_auprc:.4f}\n")
        f.write(f"Validation Confusion Matrix:\n{val_conf_matrix}\n")
        f.write(f"Validation Classification Report:\n{json.dumps(val_class_report, indent=4)}\n")

    # Test the best model on the test set
    y_test_pred = best_model.predict(X_test_selected)
    y_test_pred_prob = best_model.predict_proba(X_test_selected)[:, 1]  # Probabilities for ROC-AUC

    # Test metrics
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_roc_auc = roc_auc_score(y_test, y_test_pred_prob)
    test_f1 = f1_score(y_test, y_test_pred)
    test_balanced_acc = balanced_accuracy_score(y_test, y_test_pred)
    test_auprc = average_precision_score(y_test, y_test_pred_prob)
    test_conf_matrix = confusion_matrix(y_test, y_test_pred)
    test_class_report = classification_report(y_test, y_test_pred, output_dict=True)

    # Write test results to output file
    with open(f'{output_file}.txt', "a") as f:
        f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
        f.write(f"Test ROC-AUC Score: {test_roc_auc:.4f}\n")
        f.write(f"Test F1 Score: {test_f1:.4f}\n")
        f.write(f"Test Balanced Accuracy: {test_balanced_acc:.4f}\n")
        f.write(f"Test AUPRC: {test_auprc:.4f}\n")
        f.write(f"Test Confusion Matrix:\n{test_conf_matrix}\n")
        f.write(f"Test Classification Report:\n{json.dumps(test_class_report, indent=4)}\n")

    return (train_accuracy, train_roc_auc, train_f1, train_balanced_acc, train_auprc,
            val_accuracy, val_roc_auc, val_f1, val_balanced_acc, val_auprc,
            test_accuracy, test_roc_auc, test_f1, test_balanced_acc, test_auprc, best_model, selector)

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
    
    cumulative_val_accuracy = 0
    cumulative_val_roc_auc = 0
    cumulative_val_f1 = 0
    cumulative_val_balanced_acc = 0
    cumulative_val_auprc = 0
    
    cumulative_test_accuracy = 0
    cumulative_test_roc_auc = 0
    cumulative_test_f1 = 0
    cumulative_test_balanced_acc = 0
    cumulative_test_auprc = 0

    for iteration in range(1, n_iterations + 1):
        start_time = time.time()
        with open(f'{output_file}.txt', "a") as f:
            f.write(f"\start time: {start_time:.4f}\n")
        # Split the data into training, validation, and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=iteration)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=iteration) # 0.25 x 0.8 = 0.2
         
        # Standardize the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        # Train the model and get metrics
        (train_accuracy, train_roc_auc, train_f1, train_balanced_acc, train_auprc,
         val_accuracy, val_roc_auc, val_f1, val_balanced_acc, val_auprc,
         test_accuracy, test_roc_auc, test_f1, test_balanced_acc, test_auprc, best_model, selector) = \
            train_random_forest_with_feature_selection(X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, output_file, 
                                                    feature_selection_method, use_feature_selection, 
                                                    variance_threshold, verbose)
        
        # Accumulate metrics for averaging
        cumulative_train_accuracy += train_accuracy
        cumulative_train_roc_auc += train_roc_auc
        cumulative_train_f1 += train_f1
        cumulative_train_balanced_acc += train_balanced_acc
        cumulative_train_auprc += train_auprc
        
        cumulative_val_accuracy += val_accuracy
        cumulative_val_roc_auc += val_roc_auc
        cumulative_val_f1 += val_f1
        cumulative_val_balanced_acc += val_balanced_acc
        cumulative_val_auprc += val_auprc
        
        cumulative_test_accuracy += test_accuracy
        cumulative_test_roc_auc += test_roc_auc
        cumulative_test_f1 += test_f1
        cumulative_test_balanced_acc += test_balanced_acc
        cumulative_test_auprc += test_auprc

        # Store the results for this iteration
        results.append({
            'iteration': iteration,
            'train_accuracy': train_accuracy,
            'train_roc_auc': train_roc_auc,
            'train_f1': train_f1,
            'train_balanced_acc': train_balanced_acc,
            'train_auprc': train_auprc,
            'val_accuracy': val_accuracy,
            'val_roc_auc': val_roc_auc,
            'val_f1': val_f1,
            'val_balanced_acc': val_balanced_acc,
            'val_auprc': val_auprc,
            'test_accuracy': test_accuracy,
            'test_roc_auc': test_roc_auc,
            'test_f1': test_f1,
            'test_balanced_acc': test_balanced_acc,
            'test_auprc': test_auprc
        })
        
        end_time = time.time()
        iteration_time = end_time - start_time
        with open(f'{output_file}.txt', "a") as f:
            f.write(f"\Iteration time: {iteration_time:.4f}\n")

    # Calculate averages
    avg_train_accuracy = cumulative_train_accuracy / n_iterations
    avg_train_roc_auc = cumulative_train_roc_auc / n_iterations
    avg_train_f1 = cumulative_train_f1 / n_iterations
    avg_train_balanced_acc = cumulative_train_balanced_acc / n_iterations
    avg_train_auprc = cumulative_train_auprc / n_iterations

    avg_val_accuracy = cumulative_val_accuracy / n_iterations
    avg_val_roc_auc = cumulative_val_roc_auc / n_iterations
    avg_val_f1 = cumulative_val_f1 / n_iterations
    avg_val_balanced_acc = cumulative_val_balanced_acc / n_iterations
    avg_val_auprc = cumulative_val_auprc / n_iterations

    avg_test_accuracy = cumulative_test_accuracy / n_iterations
    avg_test_roc_auc = cumulative_test_roc_auc / n_iterations
    avg_test_f1 = cumulative_test_f1 / n_iterations
    avg_test_balanced_acc = cumulative_test_balanced_acc / n_iterations
    avg_test_auprc = cumulative_test_auprc / n_iterations

    # Store averages in the results as the last entry
    results.append({
        'iteration': 'Average',
        'train_accuracy': avg_train_accuracy,
        'train_roc_auc': avg_train_roc_auc,
        'train_f1': avg_train_f1,
        'train_balanced_acc': avg_train_balanced_acc,
        'train_auprc': avg_train_auprc,
        'val_accuracy': avg_val_accuracy,
        'val_roc_auc': avg_val_roc_auc,
        'val_f1': avg_val_f1,
        'val_balanced_acc': avg_val_balanced_acc,
        'val_auprc': avg_val_auprc,
        'test_accuracy': avg_test_accuracy,
        'test_roc_auc': avg_test_roc_auc,
        'test_f1': avg_test_f1,
        'test_balanced_acc': avg_test_balanced_acc,
        'test_auprc': avg_test_auprc
    })

    # Convert results to a DataFrame
    df_results = pd.DataFrame(results)

    # Save to CSV
    df_results.to_csv(f'{output_file}.csv', index=False)

    return df_results


def main(input_data_file, output_file, feature_selection_method, n_iterations=10, test_size=0.2, 
         use_feature_selection=True, variance_threshold=0.8):
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
    if len(sys.argv) < 4:
        print("Usage: python script_name.py <input_csv_file> <output_file> <feature_selection_method> <use_feature_selection> [n_iterations] [test_size] [variance_threshold]")
        sys.exit(1)

    # Required arguments from the command line
    input_data_file = sys.argv[1]
    output_file = sys.argv[2]
    use_feature_selection = bool(int(sys.argv[3])) # 1 or 0 for True/False

    # Optional arguments with default values
    feature_selection_method = sys.argv[4] if len(sys.argv) > 4 else None
    n_iterations = int(sys.argv[5]) if len(sys.argv) > 5 else 5
    test_size = float(sys.argv[6]) if len(sys.argv) > 6 else 0.2
    variance_threshold = float(sys.argv[7]) if len(sys.argv) > 7 else 0.8
    
    # Call the main function with command-line arguments
    main(input_data_file, output_file, feature_selection_method, n_iterations, test_size, use_feature_selection, variance_threshold)
