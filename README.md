# MGST：A Multi-modal Graph Fusion-Driven Framework for Spatial Transcriptomics Region Identification and Functional Microenvironment Analysis
# Overview
    Spatial transcriptomics technology enables the systematic acquisition of high-throughput gene expression data on a genome-wide scale while preserving the spatial 
information of tissues in situ, offering significant opportunities to uncover the structural and functional heterogeneity of tissue microenvironments. Existing methods for 
spatial domain identification face limitations in multimodal collaboration and dynamic graph modeling, making it difficult to fully incorporate spatial, expression, and 
morphological information, and often suffer from interference due to high-dimensional noise and data sparsity. In this study, we propose the MGST framework, which leverages a 
multimodal graph fusion strategy to effectively integrate spatial location, gene expression, and tissue morphology information. The framework employs a dynamic adaptive graph 
structure optimization mechanism, overcoming the limitations of traditional graph neural networks that rely on static adjacency relationships, thus enhancing feature 
expression discrimination and generalization performance. Evaluations on multiple datasets (including human DLPFC and breast cancer) demonstrate the superior performance of 
MGST in spatial domain identification. Moreover, MGST not only accurately identifies tumor subtypes associated with immune suppression and metabolic adaptation in breast 
cancer but also reveals functional units regulated by calcium signaling in the mouse forebrain, providing a reliable analytical framework for systematically unveiling the 
complex heterogeneity of tissue microenvironments.
# Overview of the repository
image.py    This file contains functions related to image processing, particularly for extracting features from images and performing cropping operations.
main.py	    This file serves as the main entry point for the script and coordinates the data processing, feature extraction, and model training.
MLPmodel.py	Defines the MLPFusion class, which is a Multi-Layer Perceptron (MLP) used to fuse adjacency matrices from different data sources (e.g., gene expression, spatial 
            data, and image features).
spatial.py	Contains functions related to spatial data processing
VAEmodel.py	Defines the GAE_VAE class, which is a Graph Autoencoder-Variational Autoencoder (GAE-VAE) model used for dimensionality reduction and clustering of graph- 
            structured data.
# Overview of the repository
The DLPFC dataset (http://research.libd.org/spatialLIBD)
human breast cancer and mouse brain FFPE samples (https://db.cngb.org/stomics/datasets/)
Mouse Forebrain(https://doi.org/10.5281/zenodo.10968451).
The OSMFISH platform provided the MERFISH mouse brain dataset (http://linnarssonlab.org/osmFISH/).
