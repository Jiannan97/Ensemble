# Stability Prediction of Organic Field-Effect Transistors by Machine Learning

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10-blue" alt="Python">
  <img src="https://img.shields.io/badge/pytorch-2.3-red" alt="PyTorch">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
</p>

## 1. Paper Abstract

Organic field-effect transistors (OFETs) have emerged as promising candidates for next-generation organic electronics owing to their flexibility, low cost, and compatibility with large-area fabrication. However, their limited stability remains a critical barrier to commercialization, largely due to the absence of unified and efficient design principles. Conventional evaluation methods, such as accelerated aging tests and microstructural analyses, typically rely on single-variable control and empirical analysis, making it difficult to systematically capture the nonlinear correlations and coupling effects among different influence factors, and insufficient for systematically identifying degradation mechanisms of OFETs. 

To address this challenge, we propose a tailored machine learning (ML) pipeline for predicting and interpreting the stability of OFET devices. The pipeline integrates chemically meaningful molecular descriptors, extracted using RDKit and large-scale chemical language models, with an ensemble of diverse regression strategies to accommodate heterogeneous data modalities and stability metrics. Applied to a curated dataset of organic semiconductors (OSCs), the framework achieves a five-fold cross-validated R² as high as 0.982, while maintaining strong generalization to newly designed device architectures. In addition, SHAP (SHapley Additive exPlanations) analysis provides interpretable insights into how specific molecular features contribute to device degradation, offering guidance for rational materials design. 

This work underscores the promise of ML in enhancing both predictive accuracy and mechanistic insight in organic electronics, enabling the rational design and rapid screening of stable OSCs and OFETs, and exemplifying a broader shift toward data-driven methodologies in materials research and device engineering.

---

## 2. Overview

This repository contains the implementation of a machine learning pipeline for predicting the stability of organic field-effect transistors (OFETs). The pipeline integrates:

1. **Data Loading & Preprocessing** – including SMILES-based molecular descriptors (RDKit and Molformer) and categorical/continuous feature handling.
2. **Data Augmentation** – using cubic spline interpolation and FGSM-based adversarial perturbation.
3. **Ensemble Learning (Stacking)** – combining RandomForest, GradientBoosting, XGBoost, AdaBoost, and Lasso models with a Ridge meta-model.
4. **Evaluation** – cross-validation metrics including MSE, RMSE, R², and Pearson correlation coefficient.

---

## 3. Installation

Clone the repository:

```
bash
git clone https://github.com/Jiannan97/Ensemble.git
cd Ensemble
```

## 4. Dependencies

The main Python packages required are:
```
numpy
pandas
scipy
scikit-learn
xgboost
torch
matplotlib
rdkit
```

You can install them via pip:
```
pip install numpy pandas scipy scikit-learn xgboost torch matplotlib rdkit
```


## 5. Usage

Place your datasets and feature files:
```
Train_dataset.csv
Test_dataset.csv
/features/molformer_feature_pca_reduced.pkl
/features/RDKit_feature.pkl
```

Run the main training and evaluation script:
```
python main.py
```

This will execute the following steps:

1. **Load and preprocess the data**

2. **Perform spline interpolation for data augmentation**

3. **Apply FGSM adversarial augmentation using a pre-trained TransformerDNN**

4. **Train ensemble stacking models and evaluate via cross-validation**

The script prints average metrics (MSE, RMSE, R², PCC) across folds.


## 6. Evaluation Metrics

The pipeline outputs the following metrics during cross-validation:

**MSE** – Mean Squared Error
**RMSE** – Root Mean Squared Error
**R²** – Coefficient of determination
**PCC** – Pearson correlation coefficient


## 7. Citation

If you use this code in your research, please cite:
Jiannan Qi, Shutao Chen, Hui Liu, Lin Zhang*, Xiaosong Chen*, Liqiang Li*, Wenping Hu. Stability Prediction of Organic Field-Effect Transistors by Machine Learning. 2025. (Submitted)


## 8. License

This project is licensed under the MIT License.

