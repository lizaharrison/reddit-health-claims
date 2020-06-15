#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
PROGRAM : subreddit_clustering_params.py
AUTHOR : Eliza Harrison

Specifies all parameters required for the clustering of subreddits using LDA and k-means

"""

import itertools

# SUBREDDIT SAMPLE CORPUS #
# Parameters for generation of random sample dataset (n submissions per subreddit)
subreddit_cluster_params = {'n_posts': 100,
                            'sample_id': 1,
                            }

# FEATURE EXTRACTION (TF-IDF) #
# Parameters for vocabulary generation and feature representation
# Min document frequency: Filters out 'rare' words e.g. specific to individual subreddits, typos etc.
# Max document frequency: Filters out high frequency terms present in most/all subreddits
min_df_range = [2, 3, 4, 5, 7, 10, 20]
max_df_range = [0.6]
vocab_params_condensed = {
    'min_df': min_df_range,  # min_df: number of docs
    'max_df': max_df_range,  # max_df: proportion of docs
}
vocab_params_all = [
    dict(zip(list(vocab_params_condensed.keys()), params_tuple))
    for params_tuple in list(itertools.product(*vocab_params_condensed.values()))
]

vocab_params_default = {
    'min_df': 10,
    'max_df': 0.6,
}

# TOPIC MODELLING (LDA) #
# Parameters for performing LDA dimensionality reduction and topic modelling
lda_params_all = {
    'n_topics': [15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 110, 120, 150],
}
lda_params_default = {
    'n_topics': 50,
}
debug_all_topics = False

# DIMENSIONALITY REDUCTION (TSVD) #
# Parameters for performing T-SVD dimensionality reduction
tsvd_params_all = {
    'n_components': [500],
}  # Increasing n_iter from default = 5 to 10 showed minimal improvement in total explained variance (1-2% only)
tsvd_params_default = {
    'n_components': 500,
}

# CLUSTERING (K-MEANS) #
kmeans_params_all = {
    'n_clusters': [15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 110, 120, 150]
}
kmeans_params_default = {
    'n_clusters': 30,
}
debug_all_clusters = False

# VISUALISATION (t-SNE) #
# Parameters for performing TSNE visualisation
tsne_params_all = {
    'perplexity': [50, 70, 100, 120, 150],
}
tsne_params_default = {
    'perplexity': 150,
    'metric': 'cosine',
}

debug_all_tsne = True

# FINAL PARAMS FOR TOP PERFORMING MODELS FOR EACH METHOD #
final_kmeans_params = {
    'sample_id': subreddit_cluster_params['sample_id'],
    'method': 'kmeans',
    'min_df': 10,
    'max_df': 0.6,
    'tsvd_components': 500,
    'n_clusters': 30,
    'top_health_cluster_id': 6,
}
'''
final_lda_params = {
    'sample_id': subreddit_cluster_params['sample_id'],
    'method': 'lda',
    'min_df': 10,
    'max_df': 0.6,
    'tsvd_components': 500,
    'n_clusters': 50,
    'top_health_cluster_id': 47,
}
'''
