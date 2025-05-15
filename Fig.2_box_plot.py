import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import scanpy as sc

def get_gene_expression_data(adata, tissue, gene, age_groups):
    tissue_data = adata[adata.obs['tissue'] == tissue, :]
    tissue_age_data = tissue_data[tissue_data.obs['age'].isin(age_groups)]
    
    # Handle the tissue that doesn't have cells in particular age
    if tissue_age_data.shape[0] == 0:
        print(f"No data available for age groups {age_groups} in tissue '{tissue}'")
        return None
    # Handle the case where the gene doesn't exist in this tissue
    try:
        gene_expression = tissue_age_data[:, gene].X.toarray().flatten()
    except KeyError:
        print(f"Gene '{gene}' does not exist in tissue '{tissue}'")
        return None
    
    log2_expression = np.log2(gene_expression + 1)
    return pd.DataFrame({
        'Age': tissue_age_data.obs['age'],  
        'log2_Expression': log2_expression
    })


def generate_boxplot(plot_data, gene, tissue, age_groups, color_palette, output_directory):
    plt.figure(figsize=(10, 8))
    plt.rcParams['font.size'] = 38
    sns.set_style("white")
    sns.boxplot(x='Age', y='log2_Expression', data=plot_data, order=age_groups, palette=color_palette)
    # Add stripplot on top of the boxplot with jitter
    sns.stripplot(x='Age', y='log2_Expression', data=plot_data, order=age_groups, color='black', size=4, jitter=True)
    
    sns.despine()
    plt.title('')
    plt.ylabel('log2(Expression + 1)')
    plt.xlabel('')
    file_name = f"{output_directory}/{gene}_{tissue}.png"
    plt.tight_layout()
    plt.savefig(file_name, dpi=300)
    plt.show()
    plt.close()

def perform_statistical_comparison(plot_data, age_groups, gene, tissue, output_directory):
    comparisons = list(itertools.combinations(age_groups, 2))
    comparison_results = []
    for group1, group2 in comparisons:
        group1_data = plot_data[plot_data['Age'] == group1]['log2_Expression']
        group2_data = plot_data[plot_data['Age'] == group2]['log2_Expression']
        
        if len(group1_data) > 0 and len(group2_data) > 0:
            stat, p = stats.mannwhitneyu(group1_data, group2_data)
            comparison_results.append({'Group1': group1, 'Group2': group2, 'P-value': p})
        else:
            # Handle the case where one or both groups have no data
            print(f"No data for comparison: '{group1}' vs '{group2}' in gene '{gene}' in tissue '{tissue}'")
    
    if len(comparison_results) > 0:
        results_df = pd.DataFrame(comparison_results)
        bonferroni_correction = len(comparisons)
        results_df['Adjusted P-value'] = results_df['P-value'] * bonferroni_correction
        results_df['Adjusted P-value'] = results_df['Adjusted P-value'].apply(lambda p: p if p <= 1 else 1)
        csv_file_path = f"{output_directory}/Pvalue_boxplot_{gene}_{tissue}.csv"
        results_df.to_csv(csv_file_path, index=False)


output_directory = '/home/emma/result/Aging/boxplot'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
age_groups = ['3m', '18m', '21m', '24m', '30m']
color_palette = ["#4E79A7", "#59A14F", "#9C755F", "#B07AA1", "#5C88B1"]
Genes = ['Beta-s', 'Xist', 'Lars2', 'Mgp', 'Rpl13a',
'Ubb', 'Ets1', 'Cfl1', 'Tmsb10', 'S100a6', 'Rps29',
'Rps27', 'Rn45s', 'Gm6981', 'Map3k1', 'Wtap',
'C130026I21Rik','Plp1', 'Bpifa1', 'Pdcd4']
tissues = ['Heart', 'Kidney', 'Liver', 'Lung', 'Marrow', 
'Brain Non-Myeloid', 'Thymus', 'Trachea', 'Diaphragm', 
'Brain Myeloid', 'Aorta', 'Spleen']
adata = sc.read_h5ad('/home/emma/data/Aging/specific_combinations_all_genes_raw.h5ad')

for gene in Genes:
    for tissue in tissues:
        plot_data = get_gene_expression_data(adata, tissue, gene, age_groups)
        if plot_data is not None:
            generate_boxplot(plot_data, gene, tissue, age_groups, color_palette, output_directory)
            perform_statistical_comparison(plot_data, age_groups, gene, tissue, output_directory)
