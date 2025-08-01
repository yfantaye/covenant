import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, 
    roc_curve, 
    precision_recall_curve, 
    average_precision_score, 
    auc
)
from lifelines.utils import concordance_index
import shap
import lightgbm as lgb

import logging

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def classification_evaluation(y_true, y_pred_proba):
    """Generates evaluation metrics and plots for classification models."""
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)
    
    metrics = {'roc_auc': roc_auc, 'pr_auc': pr_auc}
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    ax_roc.plot([0, 1], [0, 1], 'k--')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax_roc.legend()
    
    plots = {'roc_curve': fig_roc}

 
    
    # 3. Plot Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    fig_pr, ax_pr = plt.subplots()
    ax_pr.plot(recall, precision, label=f'LightGBM (PR AUC = {pr_auc:.4f})')
    ax_pr.set_xlabel('Recall')
    ax_pr.set_ylabel('Precision')
    ax_pr.set_title('Precision-Recall Curve')
    ax_pr.legend()
    plots['pr_curve'] = fig_pr

    
    return {'metrics': metrics, 'plots': plots}

def lgbm_explainability(model, X_test):
    """Generates SHAP explanations for a model."""
    metrics = {}
    plots = {}

    # 1. Feature Importance
    fig_importance, ax_importance = plt.subplots()
    lgb.plot_importance(model, max_num_features=25, height=0.8, figsize=(10, 8), ax=ax_importance)
    ax_importance.set_title('LightGBM Feature Importance')
    ax_importance.tight_layout()
    plots['importance'] = fig_importance


    # 2. SHAP Summary Plot
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    fig_shap, ax_shap = plt.subplots()
    shap.summary_plot(shap_values[1], X_test, plot_type="dot", show=False, ax=ax_shap)
    ax_shap.set_title("SHAP Summary Plot")
    ax_shap.tight_layout()
    plots['shap'] = fig_shap

    return {'metrics': metrics, 'plots': plots}

def survival_evaluation(model, test_df):
    """Generates evaluation metrics for survival models."""
    c_index = concordance_index(test_df['duration'], -model.predict_partial_hazard(test_df), test_df['event'])
    
    metrics = {'concordance_index': c_index}
    plots = {} # Add survival curve plots here if needed
    
    return {'metrics': metrics, 'plots': plots}
