import sys

import numpy as np
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split

sys.path.append("../")

from dimension_reduction import DimensionReduction
from clustering_metrics import ClusteringMetrics
from clustering_utils import ClusteringUtils
from word_embedding.fasttext_model import FastText
from word_embedding.fasttext_data_loader import FastTextDataLoader


# Main Function: Clustering Tasks

# 0. Embedding Extraction
# TODO: Using the previous preprocessor and fasttext model, collect all the embeddings of our data and store them.
# 0. Embedding Extraction
def extract_embeddings(data_path):
    ft_data_loader = FastTextDataLoader(data_path)
    texts, _ = ft_data_loader.create_train_data()

    ft_model = FastText()
    ft_model.load_model()

    embeddings = []
    for text in tqdm(texts):
        embedding = ft_model.get_query_embedding(text)
        embeddings.append(embedding)

    return np.array(embeddings)


# 1. Dimension Reduction
# TODO: Perform Principal Component Analysis (PCA):
#     - Reduce the dimensionality of features using PCA. (you can use the reduced feature afterward or use to the whole embeddings)
#     - Find the Singular Values and use the explained_variance_ratio_ attribute to determine the percentage of variance explained by each principal component.
#     - Draw plots to visualize the results.

# TODO: Implement t-SNE (t-Distributed Stochastic Neighbor Embedding):
#     - Create the convert_to_2d_tsne function, which takes a list of embedding vectors as input and reduces the dimensionality to two dimensions using the t-SNE method.
#     - Use the output vectors from this step to draw the diagram.
def perform_dimension_reduction(embeddings, project_name, run_name):
    dim_reducer = DimensionReduction()

    n_components = 50  # Choose the number of components to keep
    reduced_embeddings = dim_reducer.pca_reduce_dimension(embeddings, n_components)
    dim_reducer.wandb_plot_explained_variance_by_components(embeddings, project_name, f"{run_name}_pca")

    # t-SNE
    tsne_embeddings = dim_reducer.convert_to_2d_tsne(embeddings)
    dim_reducer.wandb_plot_2d_tsne(tsne_embeddings, project_name, f"{run_name}_tsne")

    return reduced_embeddings, tsne_embeddings


# 2. Clustering
## K-Means Clustering
# TODO: Implement the K-means clustering algorithm from scratch.
# TODO: Create document clusters using K-Means.
# TODO: Run the algorithm with several different values of k.
# TODO: For each run:
#     - Determine the genre of each cluster based on the number of documents in each cluster.
#     - Draw the resulting clustering using the two-dimensional vectors from the previous section.
#     - Check the implementation and efficiency of the algorithm in clustering similar documents.
# TODO: Draw the silhouette score graph for different values of k and perform silhouette analysis to choose the appropriate k.
# TODO: Plot the purity value for k using the labeled data and report the purity value for the final k. (Use the provided functions in utilities)

## Hierarchical Clustering
# TODO: Perform hierarchical clustering with all different linkage methods.
# TODO: Visualize the results.
def perform_clustering(embeddings, true_labels, project_name, run_name):
    clustering_utils = ClusteringUtils()
    clustering_metrics = ClusteringMetrics()

    k_values = range(2, 11)  # Range of k values to try

    # Implement K-means clustering
    best_k = None
    best_purity = 0
    best_cluster_labels = None

    for k in k_values:
        cluster_centers, cluster_labels = clustering_utils.cluster_kmeans(embeddings, k)

        # Determine genre of each cluster
        cluster_genres = []
        for cluster_id in range(k):
            cluster_docs = [true_labels[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
            cluster_genre = max(set(cluster_docs), key=cluster_docs.count)
            cluster_genres.append(cluster_genre)

        # Visualize clustering
        clustering_utils.visualize_kmeans_clustering_wandb(embeddings, k, project_name, f"{run_name}_kmeans_k{k}")

        # Evaluate clustering
        purity = clustering_metrics.purity_score(true_labels, cluster_labels)
        if purity > best_purity:
            best_k = k
            best_purity = purity
            best_cluster_labels = cluster_labels

    clustering_utils.plot_kmeans_cluster_scores(embeddings, true_labels, k_values, project_name, f"{run_name}_scores")

    print(f"Best k for K-means clustering: {best_k}")
    print(f"Purity score for best k: {best_purity}")

    for linkage_method in ['single', 'complete', 'average', 'ward']:
        cluster_labels = clustering_utils.cluster_hierarchical(embeddings, linkage_method)
        clustering_utils.wandb_plot_hierarchical_clustering_dendrogram(embeddings, project_name, linkage_method,
                                                                       f"{run_name}_hierarchical_{linkage_method}")
        # Evaluate clustering
        purity = clustering_metrics.purity_score(true_labels, cluster_labels)
        print(f"Purity score for {linkage_method} linkage: {purity}")


# 3. Evaluation
# TODO: Using clustering metrics, evaluate how well your clustering method is performing.


if __name__ == "__main__":
    data_path = "../indexer/index/documents.json"
    project_name = "clustering_project"
    run_name = "clustering_run"
    print("Loading data...")
    embeddings = extract_embeddings(data_path)
    reduced_embeddings, tsne_embeddings = perform_dimension_reduction(embeddings, project_name, run_name)
    print("Load true labels for evaluation...")
    # Load true labels (genres) for evaluation
    ft_data_loader = FastTextDataLoader(data_path)
    _, true_labels = ft_data_loader.create_train_data()
    print("Loading embeddings...")
    perform_clustering(tsne_embeddings, true_labels, project_name, run_name)
