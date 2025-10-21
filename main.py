#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main training and evaluation script for molecular mobility prediction.

Includes:
1. Data loading & preprocessing
2. Data augmentation (Spline + FGSM)
3. Ensemble learning (Stacking)
4. Evaluation

Author: Jiannan Qi
Date: 2025-10-21
"""

# %%
import os
import pickle
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.stats import pearsonr
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import clone
from xgboost import XGBRegressor

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# =======================
# Global Config
# =======================
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# =======================
# Data Loading
# =======================
def load_csv(file_path, encoding="gbk"):
    """Load CSV file into numpy array."""
    return pd.read_csv(file_path, header=None, encoding=encoding).values

def load_pickle(file_path):
    """Load a pickle file."""
    with open(file_path, 'rb') as file:
        return pickle.load(file)
        

# =======================
# Data Preprocessing
# =======================
def preprocess_smiles_features(X_tmp, smile_feature_molformer, smile_feature_rdkit):
    """Process SMILES features and combine with other features."""
    X = []
    for row in X_tmp:
        smile = row[1]
        feature_molformer = smile_feature_molformer[smile].reshape(-1, 1)
        feature_rdkit = smile_feature_rdkit[smile].reshape(-1, 1)
        row = np.delete(row, 1).reshape(-1, 1)
        X.append(np.vstack((feature_molformer, feature_rdkit, row)))
    return np.array(X).reshape(len(X), -1)


def preprocess_categorical_features(X, categorical_cols):
    """One-hot encode categorical features."""
    encoder = OneHotEncoder(sparse_output=False)
    X_categorical = X[:, categorical_cols]
    return encoder.fit_transform(X_categorical)


def handle_missing_values(X):
    """Replace missing values with column mean."""
    df = pd.DataFrame(X)
    return df.fillna(df.mean()).values


def standardize_data(X_train, X_test):
    """Standardize numeric features."""
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test)


def process_data():
    """Main data processing pipeline."""
    data = load_csv("Train_dataset.csv")
    dntt = load_csv("Test_dataset.csv")
    data = np.vstack((data, dntt))

    smile_feature_molformer = load_pickle(
        "/features/molformer_feature_pca_reduced.pkl"
    )
    smile_feature_rdkit = load_pickle(
        "/features/RDKit_feature.pkl"
    )

    Y = data[:, -2]
    X_tmp = data[:, 1:-2]

    X = preprocess_smiles_features(X_tmp, smile_feature_molformer, smile_feature_rdkit)

    categorical_cols = [221, 222, 223, 225, 226, 231, 237]
    X_categorical_encoded = preprocess_categorical_features(X, categorical_cols)

    continuous_cols = np.setdiff1d(np.arange(X.shape[1]), categorical_cols)
    X_continuous = X[:, continuous_cols]
    X = np.hstack([X_continuous, X_categorical_encoded])
    X = handle_missing_values(X)

    DNTT_X, DNTT_Y = X[-16:], Y[-16:]
    X, Y = X[:-16], Y[:-16]
    X, DNTT_X = standardize_data(X, DNTT_X)

    return X, Y, DNTT_X, DNTT_Y


# =======================
# Data Augmentation
# =======================
def spline_interpolate(X, y, idx, n_samples=10):
    """
    Perform spline interpolation on the data based on the provided index `idx`.

    Parameters:
        X: (num_samples, num_features) Original feature data
        y: (num_samples,) Target values
        idx: List of starting indices for each group
        n_samples: Number of interpolated samples per original data point

    Returns:
        X_interpolated: Interpolated feature data
        y_interpolated: Interpolated target values
    """
    X_interp_list = []
    y_interp_list = []

    for start, end in zip(idx[:-1], idx[1:]):
        X_sub, y_sub = X[start:end, :], y[start:end]

        if len(X_sub) < 3:
            continue  # Skip groups with less than 3 points (CubicSpline requires at least 3)

        # Interpolation indices
        num_samples = len(X_sub)
        x_idx = np.arange(num_samples)
        x_idx_new = np.linspace(0, num_samples - 1, num_samples * (n_samples + 1))

        # Interpolate each feature in X
        X_interpolated = np.zeros((len(x_idx_new), X.shape[1]))
        for j in range(X.shape[1]):
            cs = CubicSpline(x_idx, X_sub[:, j], bc_type='natural')
            X_interpolated[:, j] = cs(x_idx_new)

        # Interpolate the target variable y
        cs_y = CubicSpline(x_idx, y_sub, bc_type='natural')
        y_interpolated = cs_y(x_idx_new)

        X_interp_list.append(X_interpolated)
        y_interp_list.append(y_interpolated)

    # Combine all interpolated results
    X_interpolated = np.vstack(X_interp_list)
    y_interpolated = np.concatenate(y_interp_list)

    return X_interpolated, y_interpolated



# =======================
# Transformer + FGSM
# =======================
class TransformerDNN(nn.Module):
    """Transformer-based DNN for regression."""

    def __init__(self, input_size, num_classes=1, d_model=256, num_heads=8, num_layers=3):
        super(TransformerDNN, self).__init__()
        self.embedding = nn.Linear(input_size, d_model)  
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)  

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)
        x = self.transformer_encoder(x).squeeze(1)
        x = self.fc(x)
        return x
    

def fgsm_attack(model, x, y, epsilon=0.02):
    """
    Performs the FGSM attack to generate adversarial examples.
    
    Parameters:
        model: The trained model.
        x: The input data.
        y: The ground truth labels.
        epsilon: The perturbation magnitude.

    Returns:
        x_adv: The adversarial examples generated by the FGSM attack.
    """
    x.requires_grad = True  
    outputs = model(x)
    loss = F.mse_loss(outputs, y)  # Compute loss
    loss.backward()  # Backpropagate gradients
    perturbation = epsilon * x.grad.sign()  # Compute the perturbation
    x_adv = torch.clamp(x + perturbation, 0, 1)  # Apply perturbation and clip
    return x_adv.detach()


def augment_with_sampling(model, x, y, epsilons=[0.01], sample_ratio=1):
    """
    Generates augmented data by applying FGSM attack with varying epsilon values.
    
    Parameters:
        model: The trained model.
        x: The input data.
        y: The ground truth labels.
        epsilons: List of epsilon values for generating adversarial examples.
        sample_ratio: The proportion of adversarial samples to retain.

    Returns:
        x_final: The augmented feature data.
        y_final: The augmented labels.
    """
    augmented_samples = [x]  # Retain original data
    augmented_labels = [y]   # Retain original labels

    # Generate augmented data using FGSM attack for each epsilon
    for eps in epsilons:
        x_adv = fgsm_attack(model, x, y, epsilon=eps)
        augmented_samples.append(x_adv)
        augmented_labels.append(y)  # Corresponding labels

    # Combine original and augmented data
    x_combined = torch.cat(augmented_samples, dim=0)
    y_combined = torch.cat(augmented_labels, dim=0)

    # Sample a portion of the augmented data
    total_samples = x_combined.shape[0]
    original_samples = x.shape[0]  # Number of original samples
    augmented_samples_count = total_samples - original_samples  # Number of augmented samples

    # Randomly select a subset of augmented samples
    sampled_indices = torch.randperm(augmented_samples_count)[:int(augmented_samples_count * sample_ratio)]
    augmented_samples_to_keep = x_combined[original_samples:][sampled_indices]
    augmented_labels_to_keep = y_combined[original_samples:][sampled_indices]

    # Merge original data and selected augmented data
    x_final = torch.cat([x, augmented_samples_to_keep], dim=0)
    y_final = torch.cat([y, augmented_labels_to_keep], dim=0)

    return x_final, y_final


# =======================
# Ensemble Model (Stacking)
# =======================
base_models = [
    RandomForestRegressor(max_depth=17, min_impurity_decrease=0, min_samples_leaf=1, 
                          min_samples_split=2, n_estimators=194, random_state=42),
    GradientBoostingRegressor(n_estimators=200, learning_rate=0.180, max_depth=4,
                              min_samples_split=6, min_samples_leaf=1, subsample=0.912,
                              random_state=42),
    XGBRegressor(colsample_bytree=0.5195, gamma=0.2252, learning_rate=0.1998,
                 max_depth=12, min_child_weight=4, n_estimators=160, reg_alpha=0.3334,
                 reg_lambda=1.2074, subsample=0.8896, objective='reg:squarederror',
                 random_state=42),
    AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=4), 
                      n_estimators=63, 
                      learning_rate=0.19844035113697056, 
                      random_state=42),
    Lasso(alpha=0.06061583846218687, random_state=42),
]


def stacking_predict(X_train, y_train, X_test, n_folds=5):
    """Generate stacking meta-features."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    meta_train = np.zeros((X_train.shape[0], len(base_models)))
    meta_test = np.zeros((X_test.shape[0], len(base_models)))

    for i, model in enumerate(base_models):
        test_preds = np.zeros((X_test.shape[0], n_folds))
        for j, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            model_ = clone(model)
            model_.fit(X_train[train_idx], y_train[train_idx])
            meta_train[val_idx, i] = model_.predict(X_train[val_idx])
            test_preds[:, j] = model_.predict(X_test)
        meta_test[:, i] = test_preds.mean(axis=1)
    return meta_train, meta_test


# =======================
# Evaluation
# =======================
def cross_validate(X, y, n_folds=5):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    mse_scores, rmse_scores, r2_scores, pcc_scores = [], [], [], []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Generate stacking features
        meta_features = np.zeros((X_train.shape[0], len(base_models)))
        val_meta_features = np.zeros((X_val.shape[0], len(base_models)))

        for i, model in enumerate(base_models):
            model_clone = clone(model)
            model_clone.fit(X_train, y_train)
            meta_features[:, i] = model_clone.predict(X_train)
            val_meta_features[:, i] = model_clone.predict(X_val)

        # Train meta-model
        meta_model = Ridge()
        meta_model.fit(meta_features, y_train)
        y_pred = meta_model.predict(val_meta_features)

        # Compute metrics
        mse = mean_squared_error(y_val, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_val, y_pred)
        pcc, _ = pearsonr(y_val, y_pred)

        mse_scores.append(mse)
        rmse_scores.append(rmse)
        r2_scores.append(r2)
        pcc_scores.append(pcc)

    # Output mean and standard deviation of metrics
    print(f"Average MSE: {np.mean(mse_scores):.4f} ± {np.std(mse_scores):.4f}")
    print(f"Average RMSE: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")
    print(f"Average R²: {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
    print(f"Average PCC: {np.mean(pcc_scores):.4f} ± {np.std(pcc_scores):.4f}")


# =======================
# Main Entry
# =======================
def main():
    """Main execution flow."""
    print("=== Step 1: Loading and Preprocessing Data ===")
    X, Y, DNTT_X, DNTT_Y = process_data()

    print("=== Step 2: Spline Interpolation ===")
    group_idx = [2, 8, 15, 36, 60, 63, 67, 81, 90, 99, 108, 116, 124, 129, 132]
    group_idx = [i - 2 for i in group_idx]  # Subtract 2 from the indices
    X_interp, Y_interp = spline_interpolate(X, Y, group_idx, n_samples=10)  # Perform interpolation

    print("=== Step 3: FGSM Augmentation ===")
    model = TransformerDNN(input_size=246)
    model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
    model.eval()
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    X_tensor, Y_tensor = torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32).view(-1, 1)
    X_adv, Y_adv = augment_with_sampling(model, X_tensor, Y_tensor, [0.01])
    X_aug, Y_aug = np.vstack((X_interp, X_adv.detach().numpy())), np.concatenate((Y_interp, Y_adv.detach().numpy().ravel()))

    print("=== Step 4: Ensemble Stacking Training ===")
    cross_validate(X_aug, Y_aug)  # Perform cross-validation
    print("=== Done ===")

# %%
if __name__ == "__main__":
    main()

# %%
