import scanpy as sc
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kruskal
from scipy.stats import mannwhitneyu
from scipy import stats

adata = sc.read_h5ad('/home/emma/data/Aging/specific_combinations_all_genes_raw.h5ad')  
sc.pp.log1p(adata)
genes = [
    "Xist", 
    "Lars2", 
    "Mgp", 
    "Wtap", 
    "Map3k1", 
    "Plp1", 
    "Pdcd4", 
    "Ets1", 
    "Ubb", 
    "Rn45s", 
    "Rps29", 
    "Bpifa1", 
    "Rps27", 
    "Cfl1", 
    "Beta-s", 
    "Rpl13a", 
    "C130026I21Rik", 
    "Gm6981", 
    "S100a6", 
    "Tmsb10"
]


ages_of_interest = ['3m', '18m', '21m', '24m', '30m']
adata_filtered = adata[adata.obs['age'].isin(ages_of_interest)]

# Sort the AnnData object by age
adata_filtered.obs['age'] = pd.Categorical(adata_filtered.obs['age'], categories=ages_of_interest, ordered=True)
adata_filtered = adata_filtered[adata_filtered.obs.sort_values('age').index]

# Perform Mann-Whitney U test for each gene between 18m and 24m
p_values_18_24 = {}
for gene in genes:
    expression_data_18 = adata_filtered[adata_filtered.obs['age'] == '18m', gene].X.toarray().flatten()
    expression_data_24 = adata_filtered[adata_filtered.obs['age'] == '24m', gene].X.toarray().flatten()
    
    stat, p_val = mannwhitneyu(expression_data_18, expression_data_24, alternative='two-sided')
    p_values_18_24[gene] = p_val

# Sort genes by p-value between 18m and 24m
sorted_genes = sorted(p_values_18_24, key=p_values_18_24.get)

# Calculate mean expression for each gene across ages
heatmap_data = pd.DataFrame(index=ages_of_interest, columns=sorted_genes)
for gene in sorted_genes:
    for age in ages_of_interest:
        gene_expression = adata_filtered[adata_filtered.obs['age'] == age, gene].X.toarray().flatten()
        heatmap_data.at[age, gene] = np.mean(gene_expression)

# Convert to float64
heatmap_data_float64 = heatmap_data.astype(np.float64)

# Replace the long gene name with 'Rik'
heatmap_data_float64.columns = heatmap_data_float64.columns.str.replace('C130026I21Rik', 'Rik')

# Apply gene-wise Z-score normalization (similar to dotplot with standard_scale='var')
normalized_data = pd.DataFrame(index=heatmap_data_float64.index, columns=heatmap_data_float64.columns)
for gene in heatmap_data_float64.columns:
    # Z-score normalization: (x - mean) / std
    normalized_data[gene] = stats.zscore(heatmap_data_float64[gene])

# Create heatmap with Z-score normalized data
plt.figure(figsize=(10, 8))
# Use a diverging colormap centered at 0 for Z-scores
cmap = sns.diverging_palette(230, 20, as_cmap=True)  # Blue to red colormap
ax = sns.heatmap(normalized_data, cmap=cmap, yticklabels=True, xticklabels=True, 
                 cbar=True, center=0, vmin=-2, vmax=2)  # Limit colorbar for better contrast

# Style settings
ax.tick_params(axis='y', labelsize=15)
ax.tick_params(axis='x', labelsize=15)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=15)
cbar.set_label('Z-score', size=15)

plt.tight_layout()
plt.savefig('/home/emma/result/Aging/Fig3_heatmap_zscore.png', dpi=300, bbox_inches='tight')
print("Z-score normalized heatmap saved as 'Fig3_heatmap_zscore.png'")