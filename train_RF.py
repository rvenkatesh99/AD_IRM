import sys
import os
import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
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

def plot_clustering(X, y, output_file, set_name='Training'):
    """
    Function to perform t-SNE for visualization of clustering and plot the results.
    """
    tsne = TSNE(n_components=2, random_state=42)
    X_embedded = tsne.fit_transform(X)
    
    # Plot t-SNE result
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], hue=y, palette='viridis', s=60, alpha=0.8, edgecolor='k')
    plt.title(f't-SNE Clustering for {set_name} Set')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(title='Class', loc='best')
    plt.savefig(f'{output_file}_{set_name}_tSNE_clustering.png')
    plt.show()

def train_random_forest(X_train, X_val, X_test, y_train, y_val, y_test, output_file, verbose=True, n_jobs=-1):
        
    # Define RandomForestClassifier model
    model = RandomForestClassifier(random_state=42, n_jobs=n_jobs)

    # Set hyperparameter grid for tuning
    param_grid = {
        'n_estimators': [100, 200, 500],  # Number of trees in the forest
        'max_depth': [10, 50, None],      # Maximum depth of the tree
        'min_samples_split': [2, 5, 10],  # Minimum samples required to split a node
        'min_samples_leaf': [1, 2, 4]     # Minimum samples required at a leaf node
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
    y_test_pred = best_model.predict(X_test)
    y_test_pred_prob = best_model.predict_proba(X_test)[:, 1]  # Probabilities for ROC-AUC

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

    # Plot t-SNE clustering for training, validation, and test sets
    plot_clustering(X_train, y_train, output_file, set_name='Training')
    plot_clustering(X_val, y_val, output_file, set_name='Validation')
    plot_clustering(X_test, y_test, output_file, set_name='Test')

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


def main(input_data_file, output_file, n_iterations=10, test_size=0.2, verbose=True):
    # Load input data (assuming input_data_file is a CSV with X as features and y as target)
    df = pd.read_csv(input_data_file)

    for column in df.columns:
        if df[column].astype(str).str.startswith('A').any():
            df = df.drop(columns=[column])

    # Assume the last column is the target variable and the rest are features
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    # Call the train_multiple_iterations function
    results_df = train_multiple_iterations(X, y, output_file, 
                                           n_iterations=n_iterations, 
                                           test_size=test_size)
    
    # Output the results as a confirmation
    print(f'Results saved to {output_file}.csv')

if __name__ == "__main__":
    # Make sure the right number of arguments is passed
    if len(sys.argv) < 5:
        print("Usage: python script_name.py <input_csv_file> <output_file> [n_iterations] [test_size] [variance_threshold]")
        sys.exit(1)

    # Required arguments from the command line
    input_data_file = sys.argv[1]
    output_file = sys.argv[2]
    use_feature_selection = bool(int(sys.argv[3])) # 1 or 0 for True/False
    feature_selection_method = sys.argv[4]  

    # Optional arguments with default values
    n_iterations = int(sys.argv[5]) if len(sys.argv) > 5 else 10
    test_size = float(sys.argv[6]) if len(sys.argv) > 6 else 0.2
    
    # Call the main function with command-line arguments
    main(input_data_file, output_file, n_iterations, test_size)
