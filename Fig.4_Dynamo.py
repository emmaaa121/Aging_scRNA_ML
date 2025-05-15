import scanpy as sc
import scvelo as scv
import numpy as np
import pandas as pd  
import matplotlib.pyplot as plt
#import tensorflow as tf 
import sys
import os
import dynamo as dyn
from dynamo.preprocessing import Preprocessor
dyn.dynamo_logger.main_silence()
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/lib/cuda'

adata = sc.read_10x_mtx('/home/emma/data/Aging/10X_P7_2/outs/filtered_gene_bc_matrix')
ldata = scv.read('/home/emma/data/Aging/10X_P7_2/velocyto/10X_P7_2.loom')
adata1 = scv.utils.merge(adata, ldata)

adata = sc.read_10x_mtx('/home/emma/data/Aging/10X_P7_3/outs/filtered_gene_bc_matrix')
ldata = scv.read('/home/emma/data/Aging/10X_P7_3/velocyto/10X_P7_3.loom')
adata2 = scv.utils.merge(adata, ldata)

adata_combined = adata1.concatenate(adata2, batch_key='batch')
adata_combined.write('/home/emma/data/Aging/10X_P7_2_3_marrow_bbknn.h5ad')

adata = sc.read_h5ad('/home/emma/data/Aging/10X_P7_2_3_marrow_bbknn_annotated.h5ad') 
adata.X = adata.raw.X
genes = ['Lars2', 'Mgp', 'Rpl13a',  'Ubb', 'Ets1', 'Cfl1', 'Tmsb10', 'S100a6', 'Rps29', 'Rps27', 'Gm6981', 'Map3k1', 'Wtap', 'Plp1', 'Pdcd4']
preprocessor = Preprocessor(gene_append_list=genes)
preprocessor.preprocess_adata(adata, recipe="monocle")
print(adata) 

dyn.tl.dynamics(adata, model="stochastic")
dyn.tl.reduceDimension(adata, n_pca_components=30)
dyn.tl.louvain(adata, resolution=0.2)

dyn.tl.cell_velocities(adata, method="pearson", other_kernels_dict={"transform": "sqrt"})
dyn.tl.cell_velocities(adata, basis="pca")
dyn.tl.cell_velocities(adata, basis="genovis")
dyn.vf.VectorField(adata, basis='pca')
dyn.vf.VectorField(adata, basis='genovis')
print(adata)

gene = 'Pdcd4'
dyn.pd.perturbation(adata, gene, [-100], emb_basis="genovis")
print(adata)


scv.pl.velocity_embedding_stream(adata, basis='genovis_perturbation', color='cell_type', save='/home/emma/result/Aging/perturbation/Pdcd4_perturbation.png', dpi=300)

# Gene specific velocity plots
scv.pl.velocity(adata, ['Pdcd4'], perc = [5, 60], save='/home/emma/result/Aging/perturbation/Pdcd4_velocity_expression_01.png', dpi=300)
scv.pl.scatter(adata, basis= 'genovis' , color= 'Pdcd4',  perc = [40, 95],  save='/home/emma/result/perturbation/Pdcd4_expression.png', dpi=300)

# Velocity confidence
scv.tl.velocity(adata)
scv.tl.velocity_graph(adata)
scv.tl.velocity_confidence(adata)
keys = ['velocity_length', 'velocity_confidence']
scv.pl.scatter(adata, c=keys, perc=[5, 95], save='/home/emma/result/Aging/perturbation/a_scvelo_velocity_length_confidence_01.png', dpi=300)

# Velocity pseudotime
scv.tl.velocity_pseudotime(adata)
scv.pl.scatter(adata,basis= 'genovis', color='velocity_pseudotime', save='/home/emma/result/Aging/perturbation/a_scvelo_velocity_pseudotime_01.png', dpi=300)
