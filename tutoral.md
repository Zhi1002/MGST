MGST Full Pipeline Tutorial
This tutorial systematically describes the complete workflow of the MGST framework, covering data preparation, feature preprocessing, multimodal graph construction, model training, clustering, and result visualization. All steps are clearly explained to facilitate reproducibility and usability.
1. Data Preparation
MGST supports standard 10x Visium spatial transcriptomics data. The required input files include the gene expression matrix, spatial coordinate information, and H&E histological images. Before model execution, all raw data are organized into a standardized folder structure to ensure unified loading and subsequent processing.
2. Gene Expression Preprocessing
The raw gene expression matrix is preprocessed by quality filtering, normalization, and log transformation. Low-quality spots and low-expression genes are removed to eliminate background noise. This step generates a clean and standardized expression matrix for stable feature extraction and model training.
3. Spatial Feature Enhancement
Based on the spatial proximity relationships among spots, spatial smoothing and feature enhancement are performed. This operation effectively suppresses random noise in gene expression while preserving intrinsic spatial structural information, improving the overall robustness of spatial feature representation.
4. Dual-dimensional PCA Dimensionality Reduction
We adopt a dual-dimensional PCA strategy to compress and refine multimodal features. PCA is separately applied to gene expression profiles and morphological spatial features, obtaining compact, low-noise, and discriminative embedding representations. This dual-branch dimensionality reduction effectively retains core biological information while filtering redundant high-dimensional noise.
5. Multimodal Adjacency Graph Construction
Three independent adjacency graphs are constructed based on different modal information, including gene expression similarity, physical spatial proximity, and histological morphological similarity. These three graphs capture transcriptomic, spatial, and structural characteristics respectively, providing comprehensive topological priors for subsequent multimodal fusion.
6. Multimodal Graph Fusion
MGST employs an edge-level MLP fusion module to adaptively integrate the three established adjacency matrices. The model automatically learns optimal fusion weights for each edge across different modalities in an end-to-end manner, generating a unified multimodal adjacency graph. This adaptive weighting mechanism effectively combines complementary information from multiple modalities.
7. GAE-VAE Model Training
The fused adjacency graph and dual-branch features are fed into a graph conditional variational autoencoder (GAE-VAE). By jointly optimizing the reconstruction loss and KL divergence loss, the model learns discriminative and structurally regularized latent embeddings. The encoder aggregates spatial topological and transcriptomic features, while the regularization constraint ensures the stability and separability of the embedding space.
8. Spatial Domain Clustering
The well-learned latent embeddings are utilized for spatial domain identification. Clustering is performed to partition spots into distinct spatial subregions with consistent transcriptomic and spatial characteristics, achieving fine-grained segmentation of tissue spatial structures.
9. Result Visualization and Analysis
We provide comprehensive visualization functions to display spatial domain segmentation, gene expression distribution, and biological heterogeneity patterns. Multiple analytical diagrams are generated to intuitively demonstrate the structural subdivision, marker gene distribution, and functional differences among identified spatial domains, supporting subsequent biological interpretation.
Overall Workflow
In summary, the complete MGST pipeline follows the unified procedure: Data Preparation → Gene Expression Preprocessing → Spatial Feature Enhancement → Dual-dimensional PCA Reduction → Multimodal Adjacency Graph Construction → Adaptive Graph Fusion → GAE-VAE Training → Spatial Clustering → Result Visualization and Analysis.
