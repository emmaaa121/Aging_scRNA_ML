import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
import os
import sys

import celloracle as co
from celloracle.applications import Pseudotime_calculator
from celloracle.applications import Gradient_calculator
from celloracle.applications import Oracle_development_module
#import tensorflow as tf 
import os

os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/lib/cuda'

output_path = '/home/emma/result/Aging/oracle/'
os.makedirs(output_path, exist_ok=True)

adata = sc.read_h5ad('/home/emma/data/Aging/10X_P7_2_3_marrow_bbknn_annotated.h5ad')
cell_types = adata.obs['cell_type'].unique().tolist()
for cell_type in cell_types:
    indices = np.where(adata.obs['cell_type'] == cell_type)[0]
    cluster_points = adata.obsm['X_umap'][indices, :]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                cmap='jet', label=cell_type, marker='o', s=18, alpha=0.6)
    centroid = cluster_points.mean(axis=0)
    plt.text(centroid[0], centroid[1], str(cell_type), fontsize=10, ha='center', va='center', fontweight='bold')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title='Cell Types')
plt.savefig(f'{output_path}/basic_traj_marrow_3m.png')
plt.show()

# Initialize Pseudotime Calculator
pt = Pseudotime_calculator(adata=adata, obsm_key="X_umap", cluster_column_name="cell_type")
lineage_dict = {'the_one': cell_types}
pt.set_lineage(lineage_dictionary=lineage_dict)
#pt.plot_lineages()

# Set root cell and calculate pseudotime
x_coords = adata.obsm['X_umap'][:, 0]
y_coords = adata.obsm['X_umap'][:, 1]
max_y = np.max(y_coords)
distances_to_top_left = np.sqrt((x_coords - np.min(x_coords))**2 + (max_y - y_coords)**2)
index_of_top_left = np.argmin(distances_to_top_left)
top_left_coordinate = (x_coords[index_of_top_left], y_coords[index_of_top_left])
root_cell = adata.obs_names[index_of_top_left]
root_cells = {'the_one': root_cell}
pt.set_root_cells(root_cells=root_cells)
sc.tl.diffmap(pt.adata)
pt.get_pseudotime_per_each_lineage()
pt.plot_pseudotime(cmap="rainbow")
plt.savefig(f'{output_path}/pt.png', dpi=300)

# Load base Gene Regulatory Network (GRN) and initialize Oracle
base_GRN = co.data.load_mouse_scATAC_atlas_base_GRN()
adata.X = adata.layers['raw_count'].copy()
oracle = co.Oracle()
oracle.import_anndata_as_raw_count(adata=adata, cluster_column_name='cell_type', embedding_name='X_umap')

# Import Transcription Factor (TF) data and perform PCA
oracle.import_TF_data(TF_info_matrix=base_GRN)
oracle.perform_PCA()

# Select important Principal Components (PCs)
plt.plot(np.cumsum(oracle.pca.explained_variance_ratio_)[:100])
n_comps = np.where(np.diff(np.diff(np.cumsum(oracle.pca.explained_variance_ratio_)) > 0.002))[0][0]
plt.axvline(n_comps, color="k")
plt.show()
print(n_comps)
n_comps = min(n_comps, 50)

# Perform KNN imputation
n_cell = oracle.adata.shape[0]
k = int(0.025 * n_cell)
oracle.knn_imputation(n_pca_dims=n_comps, k=k, balanced=True, b_sight=k*8, b_maxl=k*4, n_jobs=4)

# Generate links for GRN
links = oracle.get_links(cluster_name_for_GRN_unit="cell_type", alpha=10, verbose_level=10)
links.filter_links(p=0.001, weight="coef_abs", threshold_number=10000)

# Analyze and visualize links
links.plot_scores_as_rank(cluster="precursor B cell", n_gene=30)
links.plot_score_comparison_2D(value="eigenvector_centrality", cluster1="early pro-B cell", cluster2="late pro-B cell", percentile=98)
links.plot_degree_distributions(plot_model=True)
links.get_network_score()
links.plot_score_per_cluster(goi="Ets1")

# Fit Gene Regulatory Network (GRN) for simulation
oracle.get_cluster_specific_TFdict_from_Links(links_object=links)
oracle.fit_GRN_for_simulation(alpha=10, use_cluster_specific_TFdict=True)

# Check gene expression and plot gene expression histogram
goi = "Ets1"
#sc.pl.draw_graph(oracle.adata, color=[goi, oracle.cluster_column_name], layer="imputed_count", use_raw=False, cmap="viridis", save=f'{output_path}/Ets1_expression.png')
sc.get.obs_df(oracle.adata, keys=[goi], layer="imputed_count").hist()
plt.savefig(f'{output_path}/Ets1_histogram.png')
plt.show()

# Simulate perturbation and estimate transition probability
oracle.simulate_shift(perturb_condition={goi: 0.0}, n_propagation=3)
oracle.estimate_transition_prob(n_neighbors=200, knn_random=True, sampled_fraction=1)
oracle.calculate_embedding_shift(sigma_corr=0.05)

# Plot quiver plots for simulated and randomized vectors
fig, ax = plt.subplots(1, 2, figsize=[13, 6])
oracle.plot_quiver(scale=25, ax=ax[0])
ax[0].set_title(f"Simulated cell identity shift vector: {goi} KO")
oracle.plot_quiver_random(scale=25, ax=ax[1])
ax[1].set_title("Randomized simulation vector")
plt.savefig(f'{output_path}/quiverplot.png')
plt.show()


# Define parameters for the grid used in simulation and gradient calculations
n_grid = 40
oracle.calculate_p_mass(smooth=0.8, n_grid=n_grid, n_neighbors=200)
oracle.suggest_mass_thresholds(n_suggestion=12)
min_mass = 100
oracle.calculate_mass_filter(min_mass=min_mass, plot=True)

# Create quiver plots to visualize simulated and randomized simulation vectors
fig, ax = plt.subplots(1, 2, figsize=[13, 6])
scale_simulation = 30
oracle.plot_simulation_flow_on_grid(scale=scale_simulation, ax=ax[0])
ax[0].set_title("Simulated cell identity shift vector: {goi} KO")

# Quiver plot for randomized simulation data
oracle.plot_simulation_flow_random_on_grid(scale=scale_simulation, ax=ax[1])
ax[1].set_title("Randomized simulation vector")
plt.savefig('/home/emma/result/Aging/oracle/simulated_quiverplot.png')
plt.show()


# Instantiate Gradient calculator object
oracle.adata.obs['Pseudotime'] = pt.adata.obs.Pseudotime
gradient = Gradient_calculator(oracle_object=oracle, pseudotime_key="Pseudotime")
gradient.calculate_p_mass(smooth=0.8, n_grid=n_grid, n_neighbors=200)
gradient.calculate_mass_filter(min_mass=min_mass, plot=False)
gradient.transfer_data_into_grid(args={"method": "polynomial", "n_poly":3}, plot=True)

# Calculate gradient and visualize results
gradient.calculate_gradient()
scale_dev = 40
gradient.visualize_results(scale=scale_dev, s=5)


# pseudotime gradient
fig, ax = plt.subplots(figsize=[6, 6])
gradient.plot_dev_flow_on_grid(scale=scale_dev, ax=ax)

# Save the gradient results
#gradient.to_hdf5('/home/emma/result/Marrow.celloracle.gradient')

# Initialize Oracle development module to compare two vector fields
dev = Oracle_development_module()
dev.load_differentiation_reference_data(gradient_object=gradient)
dev.load_perturb_simulation_data(oracle_object=oracle)
dev.calculate_inner_product()
dev.calculate_digitized_ip(n_bins=10)

# Show perturbation scores
vm = 1
fig, ax = plt.subplots(1, 2, figsize=[12, 6])
dev.plot_inner_product_on_grid(vm=vm, s=50, ax=ax[0])
ax[0].set_title("Perturbation Score")

dev.plot_inner_product_random_on_grid(vm=vm, s=50, ax=ax[1])
ax[1].set_title("PS with Randomized Simulation Vector")
plt.savefig('/home/emma/result/Aging/oracle/Marrow_perturbation.png', dpi=300)
plt.show()

# Visualize perturbation scores with simulation vector field
fig, ax = plt.subplots(figsize=[6, 6])
dev.plot_inner_product_on_grid(vm=vm, s=50, ax=ax)
dev.plot_simulation_flow_on_grid(scale=scale_simulation, show_background=False, ax=ax)
plt.savefig('/home/emma/result/Aging/oracle/Marrow_perturbation_vector_field.png', dpi=300)

# Final visualization of the results
fig = dev.visualize_development_module_layout_0(s=5, scale_for_simulation=scale_simulation, s_grid=50, scale_for_pseudotime=scale_dev, vm=vm, return_fig=True)

# Save the figure
fig.savefig('/home/emma/result/Aging/oracle/final_visualization.png', dpi=300)
