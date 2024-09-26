import sys
import os
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Function to save classification reports and confusion matrices
def save_classification_results(filename, set_name, accuracy, roc_auc, conf_matrix, class_report):
    with open(f'{filename}_{set_name}_results.txt', 'w') as f:
        f.write(f"{set_name} Set Results\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"ROC-AUC Score: {roc_auc:.4f}\n")
        f.write("\nConfusion Matrix:\n")
        f.write(f"{conf_matrix}\n\n")
        f.write("Classification Report:\n")
        f.write(class_report)

# Random Forest training function
def train_random_forest(X_train, X_val, y_train, y_val, verbose=True, n_jobs=-1):
    model = RandomForestClassifier(random_state=42, n_jobs=n_jobs)
    param_grid = {
        'n_estimators': [100, 200, 500],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', verbose=1 if verbose else 0, n_jobs=n_jobs)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    # Training metrics
    y_train_pred = best_model.predict(X_train)
    y_train_pred_prob = best_model.predict_proba(X_train)[:, 1]
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_roc_auc = roc_auc_score(y_train, y_train_pred_prob)
    train_conf_matrix = confusion_matrix(y_train, y_train_pred)
    train_class_report = classification_report(y_train, y_train_pred)

    # Validation metrics
    y_val_pred = best_model.predict(X_val)
    y_val_pred_prob = best_model.predict_proba(X_val)[:, 1]
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_roc_auc = roc_auc_score(y_val, y_val_pred_prob)
    val_conf_matrix = confusion_matrix(y_val, y_val_pred)
    val_class_report = classification_report(y_val, y_val_pred)

    # Plot Training ROC Curve
    fpr, tpr, _ = roc_curve(y_train, y_train_pred_prob)
    plt.figure()
    plt.plot(fpr, tpr, color='blue', label=f'Training ROC Curve (area = {train_roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Training Set ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig('Training_ROC.png')
    plt.show()

    # Plot Validation ROC Curve
    fpr, tpr, _ = roc_curve(y_val, y_val_pred_prob)
    plt.figure()
    plt.plot(fpr, tpr, color='blue', label=f'Validation ROC Curve (area = {val_roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Validation Set ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig('Validation_ROC.png')
    plt.show()

    return best_model, train_accuracy, val_accuracy, train_roc_auc, val_roc_auc, train_conf_matrix, train_class_report, val_conf_matrix, val_class_report

def plot_clustering(X, y, model, title):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    y_pred = model.predict(X)
    plt.figure()
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred, cmap='viridis', marker='o', label='Predictions')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(title)
    plt.colorbar()
    plt.savefig(f'{title}.png')
    plt.show()

def main(input_file, output_file, verbose=True, n_jobs=-1):
    # Load and preprocess data
    df = pd.read_csv(input_file)
    df = df.drop(columns=['FID', 'SampleID'])
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest and get metrics
    best_model, train_acc, val_acc, train_roc_auc, val_roc_auc, train_conf_matrix, train_class_report, val_conf_matrix, val_class_report = train_random_forest(X_train_scaled, X_val_scaled, y_train, y_val, verbose=verbose, n_jobs=n_jobs)

    # Save training and validation results
    save_classification_results(output_file, 'Training', train_acc, train_roc_auc, train_conf_matrix, train_class_report)
    save_classification_results(output_file, 'Validation', val_acc, val_roc_auc, val_conf_matrix, val_class_report)

    # Test set metrics
    y_test_pred = best_model.predict(X_test_scaled)
    y_test_pred_prob = best_model.predict_proba(X_test_scaled)[:, 1]
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_roc_auc = roc_auc_score(y_test, y_test_pred_prob)
    test_conf_matrix = confusion_matrix(y_test, y_test_pred)
    test_class_report = classification_report(y_test, y_test_pred)

    # Save test results
    save_classification_results(output_file, 'Test', test_accuracy, test_roc_auc, test_conf_matrix, test_class_report)

    # Plot Test ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_test_pred_prob)
    plt.figure()
    plt.plot(fpr, tpr, color='blue', label=f'Test ROC Curve (area = {test_roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Test Set ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig('Test_ROC.png')
    plt.show()

    # Plotting clustering outcome for test set
    plot_clustering(X_test_scaled, y_test, best_model, "Test Set Clustering Outcome")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python train_random_forest.py <input_csv_file> <output_filename>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    main(input_file, output_file)
