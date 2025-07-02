import torch
from torch import nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score, confusion_matrix, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from statsmodels.stats.contingency_tables import mcnemar
from scipy.stats import chi2_contingency
from sklearn.utils import resample

def compare_models(results_dicts, labels=None, metrics=None):
    """
    Plot evaluation metrics over training epochs for one or more models.

    Args:
        results_dicts (list or dict): One or more result dictionaries from train_model() or run_experiment().
        labels (list of str): Optional names for each model for legend labels.
        metrics (list of str): Optional list of metrics to plot. Defaults to key classification metrics.
    """

    # If a single model dictionary is passed instead of a list, wrap it in a list
    if not isinstance(results_dicts, list):
        results_dicts = [results_dicts]

    # If no labels are given, generate default ones like 'Model 1', 'Model 2', etc.
    if labels is None:
        labels = [f"Model {i+1}" for i in range(len(results_dicts))]

    # Default list of metrics to plot
    if metrics is None:
        metrics = [
            'train_losses',     # training loss per epoch
            'test_losses',      # test loss per epoch
            'f1_scores',        # F1 score per epoch
            'auc_scores',       # AUC score per epoch
            'recall_scores',    # recall score per epoch
            'precision_scores'  # precision score per epoch
        ]

    # Set up subplots: layout will be 2 columns, and enough rows to fit all metrics
    num_metrics = len(metrics)
    rows = (num_metrics + 1) // 2  # e.g., 6 metrics → 3 rows of 2
    fig, axs = plt.subplots(rows, 2, figsize=(14, 4 * rows))  # Adjust figure size for readability
    axs = axs.flatten()  # Flatten the 2D array of axes into a 1D list for easy indexing

    # Loop over each metric
    for idx, metric in enumerate(metrics):
        # For each model
        for res, label in zip(results_dicts, labels):
            if metric in res and len(res[metric]) == len(res['epochs']):
                # Plot the metric over epochs
                axs[idx].plot(res['epochs'], res[metric], label=label)

        # Title = metric name, prettified
        axs[idx].set_title(metric.replace('_', ' ').title())

        # X-axis: Epochs
        axs[idx].set_xlabel('Epoch')

        # Y-axis: First part of metric string (e.g., 'f1_scores' → 'F1')
        axs[idx].set_ylabel(metric.split('_')[0].capitalize())

        # Show which model is which
        axs[idx].legend()

        # Add grid for better readability
        axs[idx].grid(True)

    # Tightly pack the plots so they don't overlap
    plt.tight_layout()

    # Show the entire figure with all subplots
    plt.show()

    # Plot PR curves at final epoch
    plt.figure(figsize=(8, 6))
    for res, label in zip(results_dicts, labels):
        if 'y_true' in res and 'y_prob' in res:  # You must store this in your result dict
            precisions, recalls, _ = precision_recall_curve(res['y_true'], res['y_prob'])
            plt.plot(recalls, precisions, label=f"{label} PR Curve")

    plt.title("Final Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.grid(True)


def truncate_optic_input(X, keep_ratio):
    """
    Truncate each trial's optic input (theta and pitch) to a percentage of its real, unpadded length.
    The gravity and size variables are preserved. Padding is maintained to keep shape consistent.

    Parameters:
    - X: np.ndarray of shape (n_trials, 1246). Each row = [G, size, theta_1...theta_622, pitch_1...pitch_622]
    - keep_ratio: float in (0, 1), e.g., 0.5 means keep 50% of the real (unpadded) optic sequence.

    Returns:
    - X_new: np.ndarray of same shape (n_trials, 1246), but with optic values truncated + zero-padded.
    """

    N = X.shape[0]                      # Total number of trials
    X_new = np.zeros_like(X)           # Initialize new array with same shape to hold truncated data

    for i in range(N):
        # Extract gravity and size (first two elements of each row)
        g = X[i, 0]
        size = X[i, 1]

        # Extract theta (next 622 values) and pitch (final 622 values)
        theta = X[i, 2:624]
        pitch = X[i, 624:]

        # Determine the true (unpadded) length by counting non-zero entries
        theta_len = np.count_nonzero(theta)
        pitch_len = np.count_nonzero(pitch)

        # Sanity check: theta and pitch should have the same real length
        assert theta_len == pitch_len, f"Inconsistent lengths at trial {i}"

        # Calculate how many time steps to keep, based on the keep_ratio
        keep_len = int(np.ceil(keep_ratio * theta_len))

        # Create new arrays (still length 622) and fill in the truncated data
        theta_new = np.zeros(622)
        pitch_new = np.zeros(622)
        theta_new[:keep_len] = np.asarray(theta[:keep_len])   # Copy only the first `keep_len` values
        pitch_new[:keep_len] = np.asarray(pitch[:keep_len])

        # Reassemble the full feature vector
        X_new[i, 0] = g                             # Copy gravity
        X_new[i, 1] = size                          # Copy size
        X_new[i, 2:624] = theta_new                # Truncated + padded theta
        X_new[i, 624:] = pitch_new                 # Truncated + padded pitch

    return X_new


def bootstrap_f1_diff(y_true, y_pred_full, y_pred_cut, n_boot=1000):
    """
    Performs bootstrap resampling to estimate a 95% confidence interval
    for the difference in F1 scores between two classifiers.

    Parameters:
    - y_true: array-like of true labels
    - y_pred_full: predicted labels from the full-input model
    - y_pred_cut: predicted labels from the truncated-input model
    - n_boot: number of bootstrap resamples (default = 1000)

    Prints:
    - 95% confidence interval (CI)
    - Whether the CI includes 0 (i.e., whether the difference is statistically significant)

    Returns:
    - (ci_lower, ci_upper): tuple of float bounds for the CI
    """

    diffs = []  # store F1 differences for each bootstrap resample

    for _ in range(n_boot):
        # Sample indices with replacement
        idx = np.random.choice(len(y_true), size=len(y_true), replace=True)
        
        # Compute F1 scores for both models on the resampled data
        f1_full = f1_score(y_true[idx], y_pred_full[idx])
        f1_cut = f1_score(y_true[idx], y_pred_cut[idx])

        # Append the difference
        diffs.append(f1_full - f1_cut)

    # Compute 2.5 and 97.5 percentiles to get 95% CI
    ci_lower = np.percentile(diffs, 2.5)
    ci_upper = np.percentile(diffs, 97.5)

    print(f"95% CI is[{ci_lower:.4f}, {ci_upper:.4f}]")

    if ci_lower <= 0 <= ci_upper:
        print("No significant difference")
    else:
        print("Significant difference")
    
    return ci_lower, ci_upper


def remove_gravity_column(X_tensor):
    # X shape: [N, features], gravity is first column
    X_np = X_tensor.cpu().numpy() # convert into a numpy array for easy manipulation
    X_noG = X_np.copy() # copy to avoid changing original data
    X_noG[:, 0] = 0  # set first row (gravity) to zero
    return torch.tensor(X_noG, dtype=torch.float32)


def mcnemar_comparison(y_true, preds_a, preds_b, label_a="A", label_b="B"):
    """
    Performs McNemar's test to compare two classifiers' decisions on the same samples.

    Parameters:
    - y_true: np.ndarray or tensor of true labels
    - preds_a: np.ndarray of predictions from model A
    - preds_b: np.ndarray of predictions from model B
    - label_a: str name for model A (for print labels)
    - label_b: str name for model B (for print labels)

    Prints the contingency table and p-value from McNemar's test.
    """
    if hasattr(y_true, 'cpu'):
        y_true = y_true.cpu().numpy()

    # Boolean arrays for correct predictions
    correct_a = preds_a == y_true
    correct_b = preds_b == y_true

    # Construct contingency table
    both_correct = np.sum(correct_a & correct_b)
    only_a_correct = np.sum(correct_a & ~correct_b)
    only_b_correct = np.sum(~correct_a & correct_b)
    both_wrong = np.sum(~correct_a & ~correct_b)

    table = [
        [both_correct, only_a_correct],
        [only_b_correct, both_wrong]
    ]

    print(f"\nMcNemar Contingency Table ({label_a} vs {label_b}):")
    print(np.array(table))

    result = mcnemar(table, exact=False)
    print(f"McNemar p-value ({label_a} vs {label_b}): {result.pvalue:.4f}")

    if result.pvalue > 0.05:
        print("No significant difference in classification decisions.")
    else:
        print("Significant difference in classification decisions.")


def chi_square_accuracy_by_gravity(X_test, y_test, preds_all):
    """
    Runs a chi-square test to determine if prediction accuracy differs across gravity levels.
    
    Parameters:
    - X_test: torch.Tensor [N, features], with gravity as the first column
    - y_test: torch.Tensor [N]
    - preds_all: np.ndarray of shape (N,), binary predictions (0 or 1)
    
    Prints the contingency table and chi-square test results.
    """
    
    # Extract gravity and labels
    G_column = X_test[:, 0].cpu().numpy()
    y_true_np = y_test.cpu().numpy()
    
    # Correctness vector
    correctness = (preds_all == y_true_np).astype(int)
    gravity_levels = np.unique(G_column)
    
    # Build contingency table: rows=[correct, incorrect], cols=gravity levels
    contingency = np.zeros((2, len(gravity_levels)), dtype=int)

    for i, g in enumerate(gravity_levels):
        mask = G_column == g
        correct = correctness[mask]
        contingency[0, i] = np.sum(correct == 1)
        contingency[1, i] = np.sum(correct == 0)

    print("\nContingency Table [Correct / Incorrect vs Gravity Level]:")
    print(contingency)

    # Chi-square test
    chi2, p, dof, expected = chi2_contingency(contingency)

    print(f"\nChi-square statistic: {chi2:.4f}")
    print(f"Degrees of freedom: {dof}")
    print(f"P-value: {p:.4f}")

    if p > 0.05:
        print("No significant association between gravity and prediction accuracy.")
    else:
        print("Significant association: accuracy varies with gravity.")