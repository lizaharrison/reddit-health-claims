#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
PROGRAM : run_kmeans.py
AUTHOR : Eliza Harrison

This program performs k-means clustering of subreddits.

"""

import argparse
import json
import pickle

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from subreddit_clustering import subreddit_clustering_params as params
from subreddit_clustering.generate_subreddit_corpus import load_corpus, cluster_dir
from subreddit_clustering.run_tfidf import vocab_dir
from subreddit_clustering.run_tsvd import load_tsvd


def get_sub_clusters(cluster_labels, corpus):
    """
    Identifies the clusters containing subreddits in the seeding sample

    Parameters
    ----------
    cluster_labels : array
        Contains cluster labels for all subreddits in the corpus
    corpus : dataframe
        Final subreddit clustering corpus

    Returns
    -------
    (sub_cluster, health_clusters_distinct) : tuple

    """

    sub_cluster = corpus.loc[:, ['subreddit', 'health']]
    sub_cluster['cluster_label'] = cluster_labels
    health_clusters_distinct = sub_cluster.loc[sub_cluster['health'] == 1,
                                               'cluster_label'].value_counts()
    print('Distinct health-related clusters: \n{}'.format(health_clusters_distinct))

    return sub_cluster, health_clusters_distinct


def kmeans(feature_array,
           corpus,
           n_clusters=params.kmeans_params_default['n_clusters']):
    """
    Trains k-means model

    Parameters
    ----------
    feature_array : array
        Reduced dimensionality array (produced following TSVD)
    corpus : dataframe
        Final subreddit clustering corpus
    n_clusters : int
        Total number of clusters to generate using k-means

    Returns
    -------
    (model, array, summary) : tuple
        Contains trained k-means model, array of cluster IDs and dictionary of
        summary information

    """

    print('Performing K-Means...')
    try:
        model = pickle.load(open('kmeans_obj_{}'.format(n_clusters), 'rb'))
        array = pickle.load(open('kmeans_array_{}'.format(n_clusters), 'rb'))
        cluster_labels = pickle.load(open('kmeans_cluster_labels_{}'.format(n_clusters), 'rb'))

    except FileNotFoundError:
        model = KMeans(n_clusters=n_clusters,
                       init='k-means++',
                       n_init=10,
                       random_state=1,
                       )
        array = model.fit_transform(feature_array)

        cluster_labels = model.labels_
        # cluster_centres = model.cluster_centers_

        pickle.dump(model, open('kmeans_obj_{}'.format(n_clusters), 'wb'))
        pickle.dump(array, open('kmeans_array_{}'.format(n_clusters), 'wb'))
        pickle.dump(cluster_labels, open('kmeans_cluster_labels_{}'.format(n_clusters), 'wb'))

    print('K-means function: \n{}'.format(model))
    print('K-means cluster labels: \n{}'.format(cluster_labels))
    print('Cluster-distance array: \n{}'.format(array))

    inertia = model.inertia_

    sub_cluster, health_cluster_distinct = get_sub_clusters(cluster_labels, corpus)

    print('Computing Silhouette coefficient...')
    score = silhouette_score(array,
                             cluster_labels,
                             metric='cosine',
                             random_state=1,
                             )
    print('Silhouette coefficient: {}'.format(score))
    print('\n')

    summary = {
        'n_clusters': n_clusters,
        'inertia': inertia,
        'silhouette_metric': 'cosine',
        'silhouette_score': score,
        'top_health_clusters': health_cluster_distinct.to_dict(),
        'all_sub_clusters': sub_cluster.set_index('subreddit'),
    }

    return model, array, summary


def elbow_plot(n_clusters, distortions):
    """
    Generates elbow plot of distortion values for increasing number of clusters

    Parameters
    ----------
    n_clusters : list
        Ordered list corresponding to the total number of clusters for k-means models
    distortions : list
        Distortion values for each k-means model

    Returns
    -------

    """

    # Elbow plot
    plt.plot(n_clusters,
             distortions,
             marker='o',
             color='black',
             )
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.savefig('K-Means_elbow')
    plt.show()
    plt.close('all')


def silhouette_plot(n_clusters, silhouette_scores):
    """
    Generates plot of silhouette scores for increasing number of clusters

    Parameters
    ----------
    n_clusters : list
        Ordered list corresponding to the total number of clusters for k-means models
    silhouette_scores : list
        Silhouette scores for each k-means model

    Returns
    -------

    """

    # Silhouette plot
    plt.plot(n_clusters,
             silhouette_scores,
             marker='o',
             color='black',
             )
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette co-efficient')
    plt.savefig('K-Means_silhouette')
    plt.show()
    plt.close('all')


def perform_kmeans(feature_array, corpus, *n_clusters):
    """
    Performs k-means clustering of subreddit dataset.

    Parameters
    ----------
    feature_array : array
        Feature array generated following TSVD
    corpus : dataframe
        Final subreddit clustering corpus
    n_clusters : list
        Values for number of clusters to train k-means models with

    Returns
    -------
    (summary_all, health_clusters_all) : tuple
        Contains full summary dictionary with key information for all trained k-means
        models and dataframe with cluster labels for all health subreddits in seeding set

    """

    # Performs K-Means & evaluates Silhouettte score for each number of clusters
    summary_all = []
    distortions = []
    silhouette_scores = []
    health_clusters_all = []
    sub_clusters_all = []
    for n in n_clusters:
        print('Number of clusters: {}'.format(n))
        model, array, summary = kmeans(feature_array,
                                       corpus,
                                       n,
                                       )
        sub_clusters = summary['all_sub_clusters']

        health_sub_clusters = sub_clusters.loc[sub_clusters['health'] == 1].transpose()
        health_sub_clusters.drop('health', inplace=True)
        health_sub_clusters.index = [n]
        health_sub_clusters.index.name = 'cluster_label'
        health_clusters_all.append(health_sub_clusters)

        sub_clusters.index = pd.MultiIndex.from_arrays([sub_clusters.index,
                                                       sub_clusters['health']],
                                                       )
        sub_clusters.drop('health',
                          axis=1,
                          inplace=True,
                          )
        sub_clusters.columns = pd.MultiIndex.from_product([sub_clusters.columns,
                                                           [n],
                                                           ])
        sub_clusters_all.append(sub_clusters)

        del summary['all_sub_clusters']

        distortions.append(summary['inertia'])
        silhouette_scores.append(summary['silhouette_score'])

        summary_all.append(summary)

    health_clusters_all = pd.concat(health_clusters_all)
    health_clusters_all.to_pickle('health_kmeans_clusters.pkl')
    sub_clusters_all = pd.concat(sub_clusters_all,
                                 axis=1,
                                 )
    sub_clusters_all.to_pickle('all_kmeans_clusters.pkl')

    elbow_plot(n_clusters, distortions)
    silhouette_plot(n_clusters, silhouette_scores)

    return summary_all, health_clusters_all


if __name__ == '__main__':

    # Running k-means clustering with option to test all parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nposts',
                        help='number of submissions sampled per subreddit',
                        type=int,
                        default=100,
                        )
    parser.add_argument('-i', '--sampleid',
                        help='random state for sampling submissions for each subreddit',
                        type=int,
                        default=1,
                        )
    parser.add_argument('-a',
                        '--all',
                        help='run tf-idf for all parameter combinations (default = 50)',
                        action='store_true',
                        )
    args = parser.parse_args()

    sample_params = params.subreddit_cluster_params

    if args.nposts != 100:
        sample_params['n_posts'] = args.nposts
    if args.randomstate != 1:
        sample_params['sample_id'] = args.sampleid

    if args.all:
        summary_file = 'sub_cluster_summary_ALL.json'
        vocab_params = params.vocab_params_all
        tsvd_params = params.tsvd_params_all
        kmeans_params = params.kmeans_params_all
    else:
        summary_file = 'sub_cluster_summary_DEFAULT.json'
        vocab_params = [params.vocab_params_default]
        tsvd_params = {'n_components': [params.tsvd_params_default['n_components']]}
        if params.debug_all_clusters:
            kmeans_params = params.kmeans_params_all
        else:
            kmeans_params = {'n_clusters': [params.kmeans_params_default['n_clusters']]}

    subreddit_corpus = load_corpus(**sample_params)

    kmeans_df_all = []
    params_labels = ['min_df',
                     'max_df',
                     'tsvd_components',
                     'method',
                     'n_topics/clusters',
                     'topic_terms',
                     ]

    # VOCAB
    print('All TF-IDF parameters: \n{}'.format(vocab_params))
    for vocab_dict in vocab_params:
        full_summary = json.load(open(summary_file, 'r'))
        print('\nTF-IDF parameters for current test: {}'.format(vocab_dict))
        vocab_dir('set',
                  **vocab_dict,
                  )
        vocab_key = 'vocab_{}_{}'.format(vocab_dict['min_df'],
                                         vocab_dict['max_df'],
                                         )

        # T-SVD
        for n_comp in tsvd_params['n_components']:
            tsvd_dict = {'n_components': n_comp}
            print('T-SVD params: \n{}'.format(tsvd_dict))
            tsvd_key = 'tsvd_{}'.format(tsvd_dict['n_components'])
            tsvd_obj, tsvd_array = load_tsvd(**tsvd_dict)

            # K-MEANS
            kmeans_key = 'kmeans'
            kmeans_summary, kmeans_health = perform_kmeans(tsvd_array,
                                                           subreddit_corpus,
                                                           *kmeans_params['n_clusters'],
                                                           )
            params_values = [[vocab_dict['min_df'],
                              vocab_dict['max_df'],
                              n_comp,
                              'kmeans',
                              x,
                              'n/a',
                              ] for x in kmeans_health.index]
            params_df = pd.DataFrame(params_values,
                                     columns=params_labels,
                                     )
            kmeans_df = pd.DataFrame(kmeans_summary)

            kmeans_df = pd.merge(params_df,
                                 kmeans_df,
                                 left_on='n_topics/clusters',
                                 right_on='n_clusters',
                                 )
            kmeans_df_all.append(kmeans_df)
            full_summary[vocab_key][tsvd_key][kmeans_key] = kmeans_summary

        # Updates results JSON file
        cluster_dir(**sample_params)
        json.dump(full_summary, open(summary_file, 'w'),
                  indent=4,
                  )

    # Saves K-Means data to file
    kmeans_df_all = pd.concat(kmeans_df_all)
    print(kmeans_df_all)
    kmeans_df_all.to_pickle('kmeans_df.pkl')
    kmeans_df_all.to_csv('kmeans_df.csv',
                         sep=';',
                         )

    print('K-Means complete.')
