import scanpy as sc
import scvelo as scv
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import genomap.genoTraj as gp
import tensorflow as tf 
import sys
import os
import dynamo as dyn
from dynamo.preprocessing import Preprocessor
dyn.dynamo_logger.main_silence()
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/lib/cuda'

adata = sc.read_h5ad('/home/emma/data/Aging/10X_P7_2_3_marrow_bbknn_annotated.h5ad')
adata.X = adata.raw.X
genes = ['Lars2', 'Mgp', 'Rpl13a',  'Ubb', 'Ets1', 'Cfl1', 'Tmsb10', 'S100a6', 'Rps29', 'Rps27', 'Gm6981', 'Map3k1', 'Wtap', 'Plp1', 'Pdcd4', 'Map2k4', 'Map2k7']
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

gene = ['Map2k7', 'Map2k4', 'Map3k1', 'Ets1']
expr_vals = [100, 100, -100,-100]
#dyn.pd.perturbation(adata, gene, [-100], emb_basis="genotraj_wPCA")
dyn.pd.perturbation(adata, gene, expr_vals, emb_basis="genovis")
#dyn.pd.KO(adata, gene)
print(adata)

scv.pl.velocity_embedding_stream(adata, basis='genovis', color='cell_type', color_map = 'jet', save='/home/emma/result/perturbation/general_genovis.png', dpi=300)
scv.pl.velocity_embedding_stream(adata, basis='genovis_perturbation', color='cell_type', color_map = 'tab20', save='/home/emma/result/perturbation/Map2k7_Map2k4_Map3k1_Ets1_perturbation_genovis.png', dpi=300)


# Velocity confidence
scv.tl.velocity(adata)
scv.tl.velocity_graph(adata)
scv.tl.velocity_confidence(adata)
scv.pl.velocity_graph(adata, basis= 'genovis',  color='cell_type', save='/home/emma/result/perturbation/a_scvelo_velocity_graph.png')
scv.pl.proportions(adata, save='/home/emma/result/perturbation/a_scvelo_proportions.png', dpi=300)


keys = ['velocity_length', 'velocity_confidence']
scv.pl.scatter(adata, basis= 'genovis', c=keys, perc=[5, 95], save='/home/emma/result/perturbation/a_scvelo_velocity_length_confidence.png', dpi=300)

# Velocity pseudotime
scv.tl.velocity_pseudotime(adata)
scv.pl.scatter(adata,basis= 'genovis', color='velocity_pseudotime', save='/home/emma/result/perturbation/a_scvelo_velocity_pseudotime.png', dpi=300)
