import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import scanpy as sc
import genomap.genoVis as gpv  
import gc  
import sys


adata = sc.read_h5ad('/home/emma/data/Aging/tabula-muris-senis-bbknn-processed-official-annotations.h5ad')

map_age_to_group = {
            "1m": "Young",
            "3m": "Young",
            "18m": "Middle",
            "21m": "Middle",
            "24m": "Old",
            "30m": "Old"
        }
adata.obs['age_group'] = adata.obs['age'].map(map_age_to_group)

new_agegroup_color_dict = {
    "Young": '#ce6dbd',   
    "Middle": '#e7ba52',  
    "Old": '#5254a3'    
}

all_possible_ages = ["1m", "3m", "18m", "21m", "24m", "30m"]
fixed_age_colors = plt.cm.jet(np.linspace(0, 1, len(all_possible_ages)))
fixed_age_color_dict = {age: matplotlib.colors.rgb2hex(fixed_age_colors[i]) 
                        for i, age in enumerate(all_possible_ages)}

def plot_visualization_generic(embedding, categories, color_map, xlabel, ylabel, plot_title, file_name, show_title=False):
    plt.figure(figsize=(12, 10))
    plt.rcParams.update({'font.size': 50})
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    unique_categories = np.unique(categories)
    
    if 'by age_group' in plot_title:
        desired_order = ["Young", "Middle", "Old"]
        unique_categories = [cat for cat in desired_order if cat in unique_categories]
    elif 'by age' in plot_title:
        unique_categories = sorted(unique_categories, key=lambda x: int(x.replace('m', '')))
        
    for cat in unique_categories:
        indices = np.where(categories == cat)[0]
        cluster_points = embedding[indices, :]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], alpha=0.6,
                    c=color_map.get(cat, '#000000'), label=cat, marker='o', s=100)
        
        # if str(cat).isdigit() and xlabel != 'genoTraj1':
        #     centroid = cluster_points.mean(axis=0)
        #     plt.text(centroid[0], centroid[1], str(cat), fontsize=40, ha='center', va='center', fontweight='bold', color='white',
        #              bbox=dict(facecolor='black', alpha=0.5))

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    # Only set the title if show_title is True
    if show_title:
        plt.title(plot_title)
    
    # Add legend for all plots
    #leg = plt.legend(fontsize=35, markerscale=2.0)
    #leg.get_frame().set_linewidth(0.0)
    #leg.get_frame().set_facecolor('none')
        
    plt.tight_layout()
    save_path = r'/home/emma/result/Aging/Genovis/' + file_name
    plt.savefig(save_path, dpi=300)
    plt.show()
            
cell_types = ['naive T cell','T cell', 'B cell','basophil','CD8-positive, alpha-beta T cell','mature NK T cell','CD4-positive, alpha-beta T cell','regulatory T cell','immature NKT cell',
              'immature T cell','double negative T cell','mature alpha-beta T cell','DN3 thymocyte','DN4 thymocyte','precursor B cell','late pro-B cell','immature B cell',
              'naive B cell','pancreatic B cell','early pro-B cell','plasma cell','macrophage','alveolar macrophage','lung macrophage','kupffer cell','leukocyte'
              'lymphocyte', 'mast cell', 'astrocyte', 'NK cell', 'monocyte','myeloid cell','myeloid leukocyte','neutrophil', 'classical monocyte', 'dentritic cell',
              'intermediate monocyte','granulocyte','thymocyte','professional antigen presenting cell', 'microglial cell','granulocyte monocyte progenitor cell',
              'granulocytopoietic cell','hematopoietic stem cell']


tissues = adata.obs['tissue'].unique()

for cell_type in cell_types:
  for tissue in tissues:
    try:
      print(f'Processing {cell_type}s in {tissue}...')
      adata_filtered = adata[(adata.obs['cell_ontology_class'] == cell_type) & (adata.obs['tissue'] == tissue)].copy()

      # Perform genoVis on the PCA results
      pca_results = adata_filtered.obsm['X_pca']
      resVis = gpv.genoVis(pca_results, n_clusters=5, colNum=32, rowNum=32)
      adata_filtered.obs['cluster'] = resVis[1]  
      adata_filtered.obsm['X-genovis'] = resVis[0]
      
      file_path = f'/home/emma/data/Aging/{cell_type}_in_{tissue}_genoVis3cluster_bbknn.h5ad'
      sc.write(file_path, adata_filtered)
      
      # Filter the fixed age color dictionary to only include ages present in this dataset
      unique_ages = adata_filtered.obs['age'].unique()
      age_color_dict_filtered = {age: fixed_age_color_dict[age] for age in unique_ages}   
      # Create a combined dictionary of all color maps
      colormap = {
                'age_group': new_agegroup_color_dict,
                'age': age_color_dict_filtered,
                }

      # Define separate directories for different visualization types
      directories = {
          'age_group': 'Age_group',
          'age': 'Individual_age',
      }

      # Plot for all three attributes
      attributes = ['age_group', 'age']
      for attr in attributes:
          file_name = f'{directories[attr]}/{cell_type}_in_{tissue}_by_{attr}_genoVis_bbknn.png'
          
          plot_visualization_generic(
              adata_filtered.obsm['X-genovis'], 
              adata_filtered.obs[attr], 
              colormap[attr],
              'GenoVis1', 'GenoVis2', 
              f'by {attr}', 
              file_name, 
              show_title=False  
          )

      del adata_filtered
      gc.collect() 

    except Exception as e:
        print(f"Error processing {cell_type} in {tissue}: {e}")