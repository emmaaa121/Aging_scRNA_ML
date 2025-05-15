import scanpy as sc
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, ParameterSampler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from lightgbm import LGBMRegressor
import shap
import matplotlib.pyplot as plt
import joblib
import os
from scipy.stats import randint as sp_randint, uniform
from tqdm import tqdm

base_dir = "/home/emma"
data_path = os.path.join(base_dir, "data/Aging/specific_combinations_all_genes_raw.h5ad")
output_dir = os.path.join(base_dir, "result/Aging")
os.makedirs(output_dir, exist_ok=True)

adata = sc.read_h5ad(data_path)

selected_ages = [3, 18, 21, 24, 30]
adata.obs['age'] = adata.obs['age'].apply(lambda x: int(x.replace('m','')))
mask = adata.obs['age'].isin(selected_ages)
adata_selected = adata[mask]

sc.pp.normalize_total(adata_selected, target_sum=1e4)
sc.pp.log1p(adata_selected)
sc.pp.filter_genes_dispersion(adata_selected, subset=True, min_disp=0.5, max_disp=None,
                             min_mean=0.025, max_mean=10, n_bins=20, n_top_genes=None)
sc.pp.scale(adata_selected, max_value=10, zero_center=False)

data_selected = adata_selected.X.toarray() if not isinstance(adata_selected.X, np.ndarray) else adata_selected.X
ages_selected = adata_selected.obs['age'].to_numpy()

# First split into train_val and test sets (80% train_val, 20% test)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    data_selected, ages_selected, test_size=0.2, random_state=42
)

print(f"Dataset shape: {data_selected.shape}")
print(f"Training set: {X_train_val.shape}")
print(f"Test set: {X_test.shape}")
print(f"Age range: {np.min(ages_selected)} to {np.max(ages_selected)}")

def run_lightgbm_random_search(X_train_val, y_train_val, param_dist, n_iter_search, n_splits=5, output_csv='LightGBM_ParaTunes.csv'):
    """
    Run randomized parameter search with cross-validation
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = []
    best_score = float('inf')
    best_params = {}
    best_metrics = {}
    
    param_sampler = ParameterSampler(param_dist, n_iter=n_iter_search, random_state=42)
    for params in tqdm(param_sampler, desc="Parameter combinations", total=n_iter_search):
        cv_metrics = {'mse': [], 'rmse': [], 'mae': [], 'r2': [], 'mape': []}
        
        for train_index, val_index in kf.split(X_train_val):
            X_train, X_val = X_train_val[train_index], X_train_val[val_index]
            y_train, y_val = y_train_val[train_index], y_train_val[val_index]
            
            lgbm_model = LGBMRegressor(**params, random_state=42)
            lgbm_model.fit(X_train, y_train)
            predictions = lgbm_model.predict(X_val)
            
            mse = mean_squared_error(y_val, predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_val, predictions)
            r2 = r2_score(y_val, predictions)
            mape = np.mean(np.abs((y_val - predictions) / y_val)) * 100
            
            cv_metrics['mse'].append(mse)
            cv_metrics['rmse'].append(rmse)
            cv_metrics['mae'].append(mae)
            cv_metrics['r2'].append(r2)
            cv_metrics['mape'].append(mape)
        
        avg_metrics = {f'avg_{metric}': np.mean(values) for metric, values in cv_metrics.items()}
        
        if avg_metrics['avg_mse'] < best_score:
            best_score = avg_metrics['avg_mse']
            best_params = params
            best_metrics = avg_metrics
        
        results.append({
            **params,
            **avg_metrics
        })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    
    return results_df, best_params, best_metrics

# Define the parameter distribution for LightGBM
param_dist = {
    'num_leaves': sp_randint(20, 150),
    'max_depth': sp_randint(3, 15),
    'learning_rate': uniform(0.01, 0.3),
    'n_estimators': sp_randint(100, 1000),
    'subsample': uniform(0.5, 0.5),  # This samples between 0.5 and 1.0
    'colsample_bytree': uniform(0.5, 0.5)  # This samples between 0.5 and 1.0
}

# Number of iterations for Randomized Search
n_iter_search = 20

# Define the path for the output file
results_path = os.path.join(output_dir, "LightGBM_ParaTunes.csv")

# Run randomized search with cross-validation on the training set
print("Running parameter search with cross-validation...")
results_df, best_params, best_cv_metrics = run_lightgbm_random_search(
    X_train_val, y_train_val, param_dist, n_iter_search, output_csv=results_path
)

print("Parameter search complete!")
print("Best Parameters:", best_params)
print("Best CV Metrics:", best_cv_metrics)

# Train final model on the entire training set using best parameters
print("\nTraining final model on full training set...")
final_model = LGBMRegressor(**best_params, random_state=42)
final_model.fit(X_train_val, y_train_val)

# Evaluate on held-out test set
y_pred_test = final_model.predict(X_test)
test_metrics = {
    'test_mse': mean_squared_error(y_test, y_pred_test),
    'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
    'test_mae': mean_absolute_error(y_test, y_pred_test),
    'test_mape': mean_absolute_percentage_error(y_test, y_pred_test) * 100,
    'test_r2': r2_score(y_test, y_pred_test)
}

print("\nTest Set Metrics:")
for metric, value in test_metrics.items():
    print(f"  {metric}: {value:.4f}")

# Save model and metrics
model_path = os.path.join(output_dir, "best_lightgbm_model.joblib")
joblib.dump(final_model, model_path)

test_metrics_path = os.path.join(output_dir, "lightgbm_test_metrics.csv")
pd.DataFrame([test_metrics]).to_csv(test_metrics_path, index=False)

print(f"\nResults saved to: {results_path}")
print(f"Model saved to: {model_path}")
print(f"Test metrics saved to: {test_metrics_path}")

# Generate SHAP values on the test set
print("\nCalculating SHAP values...")
explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X_test)

# Create and save SHAP plots
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test, feature_names=adata_selected.var_names, plot_type='bar', show=False)
plt.tight_layout()
shap_bar_path = os.path.join(output_dir, "global_feature_importance.png")
plt.savefig(shap_bar_path, dpi=300)
plt.close()

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test, feature_names=adata_selected.var_names, plot_type='dot', show=False)
plt.tight_layout()
shap_dot_path = os.path.join(output_dir, "local_explanation_summary.png")
plt.savefig(shap_dot_path, dpi=300)
plt.close()

print(f"SHAP plots saved to {output_dir}")

# Training summary
print("\nTraining Summary:")
print(f"Total dataset size: {data_selected.shape[0]} samples")
print(f"Number of features: {data_selected.shape[1]}")
print(f"Best CV MSE: {best_cv_metrics['avg_mse']:.4f}")
print(f"Test MSE: {test_metrics['test_mse']:.4f}")
print(f"Test R²: {test_metrics['test_r2']:.4f}")