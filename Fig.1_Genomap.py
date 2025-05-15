import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy
import genomap as gp  
import scanpy as sc
import gc  

adata = sc.read_h5ad('/home/emma/data/Aging/tabula-muris-senis-bbknn-processed-official-annotations_genovis.h5ad')

age_groups = {'Y': ['1m', '3m'], 'M':['18m', '21m'], 'O':['24m', '30m'], } 
cell_types = ['naive T cell','T cell', 'B cell','basophil','CD8-positive, alpha-beta T cell','mature NK T cell','CD4-positive, alpha-beta T cell','regulatory T cell','immature NKT cell',
              'immature T cell','double negative T cell','mature alpha-beta T cell','DN3 thymocyte','DN4 thymocyte','precursor B cell','late pro-B cell','immature B cell',
              'naive B cell','pancreatic B cell','early pro-B cell','plasma cell','macrophage','alveolar macrophage','lung macrophage','kupffer cell','leukogyte'
              'lymphocyte', 'mast cell', 'astrocyte', 'NK cell', 'monocyte','myeloid cell','myeloid leukocyte','neutrophil', 'classical monocyte', 'dentritic cell',
              'intermediate monocyte','granulocyte','thymocyte','professional antigen presenting cell', 'microglial cell','granulocyte monocyte progenitor cell',
              'granulocytopoietic cell','hematopoietic stem cell']

tissues =  ['Marrow', 'Heart', 'Bladder', 'Skin', 'Large_Intestine', 'Trachea', 'Diaphragm', 
            'Brain_Non-Myeloid', 'Brain_Myeloid', 'BAT', 'GAT', 'MAT', 'SCAT', 'Aorta',
            'Tongue', 'Heart', 'Marrow', 'Mammary_Gland', 'Fat', 'Kidney', 'Liver','Lung',
            'Limb_Muscle', 'Pancreas', 'Spleen', 'Thymus']


output_dir = "/home/emma/result/Aging/genomap"
os.makedirs(output_dir, exist_ok=True)

colNum = 45
rowNum = 45
max_maps = 15

for organ in tissues:
    for group_name, ages in age_groups.items():
        organ_group_adata = adata[(adata.obs['tissue'] == organ) & (adata.obs['age'].isin(ages))].copy()
        print(f'tissue: {organ}, age:{group_name}')
        
        if organ_group_adata.n_obs == 0: 
            continue
        
        for cell in cell_types:
            cell_group_adata = organ_group_adata[organ_group_adata.obs['cell_ontology_class'] == cell].copy()
            print(f'cell ontology classe: {cell}')
            
            if cell_group_adata.n_obs < 2:
                print(f"Not enough cells for {cell} in {organ} for age group {group_name}. Skipping...")
                continue
              
            sc.pp.highly_variable_genes(cell_group_adata, n_top_genes=2000)
            hvg_subset = cell_group_adata[:, cell_group_adata.var['highly_variable']]

            data = hvg_subset.X.toarray()
            dataNorm = scipy.stats.zscore(data, axis=0, ddof=1)
            dataNorm = np.nan_to_num(dataNorm)

            genoMaps = gp.construct_genomap(dataNorm, rowNum, colNum, epsilon=0.0, num_iter=200)

            for i in range(min(max_maps, len(genoMaps))):
                # Retrieve the cell name for the current genoMap
                cell_name = cell_group_adata.obs_names[i] 
                # Sanitize the cell name to create a valid filename
                formatted_cell_name = cell_name.replace(' ', '_').replace(',', '').replace('-', '_').replace('/', '_').replace(':', '_')
                genoMap = genoMaps[i]
                output_filename = os.path.join(output_dir, f"{cell}_{organ}_{group_name}_{formatted_cell_name}.png")
                
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(genoMap, aspect='auto')
                ax.axis('off')  
                plt.savefig(output_filename, dpi=100, bbox_inches='tight', pad_inches=0)
                plt.close(fig) 
            del cell_group_adata
            gc.collect() 



        
          
