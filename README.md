This repository contains code for genomic analysis with a focus on cell type identification, gene signatures, pathway analysis, and in silico perturbation.

* **Fig. 1:** Visualization of specific cell types with Genovis and Genomap
  * [Fig.1_Genomap.py](./Fig.1_Genomap.py)
  * [Fig.1_GenoVis.py](./Fig.1_GenoVis.py)

* **Fig. 2:** Identifying signature genes using LightGBM and SHAP
  * [Fig.2_box_plot.py](./Fig.2_box_plot.py)
  * [Fig.2_heatmap.py](./Fig.2_heatmap.py)
  * [Fig.2_LightGBM_Shap.py](./Fig.2_LightGBM_Shap.py)

* **Fig. 3:** Toxicity Pathway Analysis (IPA)
  * [Fig.3_pathway analysis.py](./Fig.3_pathway%20analysis.py)

* **Fig. 4:** In silico gene perturbation analysis using Oracle (based on multi-omics data) and Dynamo (based on RNA velocity)
  * [Fig.4_Dynamo.py](./Fig.4_Dynamo.py)
  * [Fig.4_Oracle.py](./Fig.4_Oracle.py)

* **Fig. 5:** In silico gene perturbation analysis using Oracle and Dynamo (Focus on Map3k1/Map2k4/Jnk B Cell Receptor Signaling Pathway)
  * [Fig.5_Forced_expression_Dynamo.py](./Fig.5_Forced_expression_Dynamo.py)

## Requirements

Dependencies will be added soon.

## Usage

Instructions for running the analysis scripts and reproducing the results.

## Citation

If you use this code in your research, please cite our paper (citation information will be added upon publication).
" > README.md
Now let's handle the graphical abstract image
You'll need to manually copy the graphical abstract image from its location to your code directory:
bash# First, make sure the file exists
ls "/Users/weiwu/Desktop/Aging Revision/Graphical Abstract.png"

# If the file exists, copy it to your repository with the correct name
cp "/Users/weiwu/Desktop/Aging Revision/Graphical Abstract.png" "/Users/weiwu/Desktop/code/Graphical_Abstract.png"
If the above command fails, the file might be named differently or in a different location. You can:

Manually locate the image file:
bashfind "/Users/weiwu/Desktop/Aging Revision" -name "*.png"

Or simply copy the file manually using Finder:

Open two Finder windows
Navigate to "/Users/weiwu/Desktop/Aging Revision" in one
Navigate to "/Users/weiwu/Desktop/code" in the other
Find the graphical abstract image and drag it to your code folder
Rename it to "Graphical_Abstract.png" if needed



After copying the image, add and commit both files
bashcd /Users/weiwu/Desktop/code

# Add README file only first
git add README.md
git commit -m "Add README structure"

# Now verify the image file is there
ls Graphical_Abstract.png

# If the file exists, add and commit it
git add Graphical_Abstract.png
git commit -m "Add graphical abstract image"

# Push both changes
git push
If you still have issues with the image file, you can proceed with just the README for now and add the image later when you have it available in the correct format and location.RetryClaude can make mistakes. Please double-check responses.
