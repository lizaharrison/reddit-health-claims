# !/usr/bin/python
# -*- coding: utf-8 -*-

"""
PROGRAM : run_tsne.py
AUTHOR : Eliza Harrison

Performs t-SNE on subreddit clustering dataset.
Generates t-SNE plots to visualise subreddit clustering dataset and results in 2D space.

"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.manifold import TSNE

import config
from subreddit_clustering import subreddit_clustering_params as params
from subreddit_clustering.generate_subreddit_corpus import cluster_dir, load_corpus
from subreddit_clustering.run_tfidf import vocab_dir
from subreddit_clustering.run_tsvd import load_tsvd


def load_tsne(perplexity=params.tsne_params_default['perplexity'],
              metric=params.tsne_params_default['metric']):
    """
    Loads t-SNE model and array from files.

    Parameters
    ----------
    perplexity : int
        The perplexity value used for t-SNE.
    metric : str
        Distance metric to be used for t-SNE. Defaults to 'cosine'.

    Returns
    -------
    (model, array) : tuple
        Contains the trained t-SNE model and array representing each subreddit in the
        final subreddit corpus in 2D space.

    """

    model = pickle.load(open('tsne_model_{}_{}.pkl'.format(perplexity, metric), 'rb'))
    array = pickle.load(open('tsne_array_{}_{}.pkl'.format(perplexity, metric), 'rb'))

    return model, array


def tsne(array,
         perplexity=params.tsne_params_default['perplexity'],
         metric=params.tsne_params_default['metric']):
    """
    Performs t-SNE.

    Parameters
    ----------
    array : array
        Array representing subreddit clustering dataset generated following TSVD.
    perplexity : int
        The perplexity value used for t-SNE.
    metric : str
        Distance metric to be used for t-SNE. Defaults to 'cosine'.

    Returns
    -------
    (model, array) : tuple
        Contains the trained t-SNE model and array representing each subreddit in the
        final subreddit corpus in 2D space.

    """

    print('Number of components: 2')
    print('Perplixity: {}'.format(perplexity))
    print('Distance metric: {}'.format(metric))
    print('Performing T-SNE...')
    model = TSNE(n_components=2,
                 perplexity=perplexity,
                 metric=metric,
                 random_state=1,
                 )
    array = model.fit_transform(array)

    format_str = '_{}_{}'.format(perplexity, metric)
    pickle.dump(model, open('tsne_model{}.pkl'.format(format_str), 'wb'))
    pickle.dump(array, open('tsne_array{}.pkl'.format(format_str), 'wb'))

    return model, array


def load_clusters(corpus,
                  n_clusters=params.final_kmeans_params['n_clusters'],
                  health_cluster=params.final_kmeans_params['top_health_cluster_id'],
                  method='kmeans',
                  ):
    """
    Loads clustering results for specified model (cluster labels for each subreddit) from file.

    Parameters
    ----------
    corpus : dataframe
        Final subreddit clustering dataset
    n_clusters : int
        Total number of clusters generated by model.
    health_cluster : int
        Cluster ID corresponding to the top health cluster.
    method : str
        Indicates the method used to generate clusters. Either 'kmeans' or 'lda'.

    Returns
    -------
    (seeding_clusters, new_health_clusters) : tuples
        Dataframes containing the cluster IDs for seeding subreddits, and the set of subreddits
        assigned to the top health cluster (the cluster capturing the greatest number of seeding
        subreddits).

    """

    if method == 'kmeans':
        all_clusters = pd.read_pickle('./all_kmeans_clusters.pkl')

    elif method == 'lda':
        os.chdir('../lda')
        all_clusters = pd.read_pickle('all_lda_topics.pkl')

    else:
        raise ValueError('Valid options for method: kmeans / lda')

    all_clusters.columns = all_clusters.columns.droplevel(0)
    all_clusters.reset_index(inplace=True)

    seeding_clusters = pd.DataFrame(corpus.loc[corpus['health'] == 1, 'subreddit'])
    seeding_clusters['cluster_number'] = all_clusters.loc[seeding_clusters.index, n_clusters]
    new_health_clusters = all_clusters.loc[all_clusters[n_clusters] == health_cluster, ['subreddit', n_clusters]]
    new_health_clusters.rename({n_clusters: 'cluster_number'},
                               axis=1,
                               inplace=True,
                               )
    print(seeding_clusters)
    print(new_health_clusters)

    return seeding_clusters, new_health_clusters


def tsne_colours(seeding_clusters, print_colours=False):
    """

    Parameters
    ----------
    seeding_clusters : dataframe
        Dataframe containing the IDs, display names and cluster IDs for seeding subreddits.
    print_colours : bool
        Indicates whether colour palette for visualising subreddit clustering should be printed.

    Returns
    -------
    (id_2_colour, palette) : tuple
        Contains dictionary which maps subreddit cluster ID to colour mapping and the colour palette for visualising
        subreddit clustering represented as an array.

    """
    cluster_ids = seeding_clusters['cluster_number'].value_counts()
    num_clusters = len(cluster_ids) + 2
    ranked_cluster_id = seeding_clusters.drop_duplicates('cluster_number')['cluster_number'].values
    ranked_cluster_id.sort()
    id_2_colour = dict(zip(ranked_cluster_id, range(len(ranked_cluster_id))))
    palette = np.array(sns.color_palette(['#67135d',  # 5 Purple
                                          '#59ab80',  # 1 Light teal
                                          '#007076',  # 2 Dark teal
                                          '#191331',  # 0 Dark navy
                                          '#e4b354',  # 3 Gold
                                          '#9d000b',  # 4 Red
                                          '#0000c8',  # 6 Medium blue
                                          '#72a425',  # 7 Green
                                          '#e37827',  # 8 Orange
                                          ]).as_hex())
    '''
    palette = list(sns.husl_palette(num_clusters, l=0.63))
    # palette = list(sns.hls_palette(num_clusters, l=0.58))
    '''
    # palette[1] = tuple([x / 225 for x in (194, 122, 23)])
    # palette = np.array([palette[1]] + [palette[3]] + [palette[0]] + [palette[2]] + palette[5:] + [palette[4]])
    # palette = np.array([[x] for x in palette])

    if print_colours:
        sns.palplot(palette)
        plt.title('Original t-SNE palette')
        plt.show()
        sns.palplot(palette)
        plt.title('Ordered t-SNE palette')
        plt.show()
        sns.palplot(palette[-1])
        plt.title('New health colour')
        plt.show()
        sns.palplot(palette[1])
        plt.title('Top health cluster colour')
        plt.show()
        sns.palplot(palette[-2])
        plt.title('Colour for all seeding subreddits')
        plt.show()

    return id_2_colour, palette


def tsne_cluster_plot(tsne_array, tsne_colour_mapping, seeding_clusters, new_health_clusters=None,
                      perplexity=params.tsne_params_default['perplexity'], metric=params.tsne_params_default['metric'],
                      method='kmeans'):
    """
    Generates t-SNE plot which colours subreddits in the seeding set according to the cluster to which
    they were assigned by a particular clustering model.

    Parameters
    ----------
    tsne_array : array
        Array containing the locations of each subreddit in the final subreddit clustering dataset
        represented in 2D space.
    tsne_colour_mapping : tuple
        Tuple returned by the tsne_colours function containing the id_2_colour mapping and colour
        palette
    seeding_clusters : dataframe
        Dataframe containing the IDs, display names and cluster IDs for seeding subreddits.
    new_health_clusters : dataframe
        Dataframe containing the IDs, display names and cluster IDs non-seeding subreddits in the top
        health cluster.
    perplexity : int
        The perplexity value used for t-SNE.
    metric : str
        Distance metric to be used for t-SNE. Defaults to 'cosine'.
    method : str
        Indicates the method used to generate clusters. Either 'kmeans' or 'lda'.

    Returns
    -------

    """

    id_2_colour, palette = tsne_colour_mapping
    '''
    if method == 'kmeans':
        label = 'K-Means'
        loc = (0.8, 0.91)
    else:
        label = 'LDA'
        loc = (0.85, 0.91)
    '''
    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    ax.scatter(tsne_array[:, 0],
               tsne_array[:, 1],
               lw=0,
               s=30,
               c='#e3e1e1',  # Light grey
               )

    if new_health_clusters is not None:
        filename = 'tsne_w_new_health_{}_{}_{}.png'.format(method,
                                                           perplexity,
                                                           metric,
                                                           )
        label = 'seeding + new'
        new_health_tsne = tsne_array[list(new_health_clusters.index)]
        ax.scatter(new_health_tsne[:, 0],
                   new_health_tsne[:, 1],
                   lw=0,
                   s=30,
                   c=palette[-1],
                   )
    else:
        filename = 'tsne_seeding_only_{}_{}_{}.png'.format(method,
                                                           perplexity,
                                                           metric,
                                                           )
        label = 'seeding only'

    seeding_clusters['colour_label'] = seeding_clusters['cluster_number'].map(id_2_colour)
    seeding_tsne = tsne_array[list(seeding_clusters.index)]
    ax.scatter(seeding_tsne[:, 0],
               seeding_tsne[:, 1],
               lw=0,
               s=30,
               c=palette[seeding_clusters['colour_label'].astype(int)]
               )
    ax.text(0.7, 0.91, label,
            fontsize=16,
            bbox=dict(facecolor='none',
                      edgecolor='lightgray',
                      alpha=0.5,
                      ),
            transform=ax.transAxes,
            )
    plt.xlim(-20, 20)
    plt.ylim(-20, 20)
    '''
    ax.text(loc[0], loc[1],
            label,
            fontsize=16,
            bbox=dict(facecolor='none',
                      edgecolor='lightgray',
                      alpha=0.5,
                      ),
            zorder=-1,
            transform=ax.transAxes,
            )
    '''
    ax.axis('tight')
    ax.axis('off')
    plt.savefig(filename)
    plt.show()
    plt.close('all')


def tsne_summary_plot(tsne_array, tsne_colour_mapping, seeding_clusters, new_health_clusters,
                      perplexity=params.tsne_params_default['perplexity'],
                      metric=params.tsne_params_default['metric'],
                      method='kmeans'):
    """
    Generates t-SNE plot which colours subreddits in the seeding set are labelled in one colour,
    while non-seeding subreddits assigned to the top health cluster labelled in a contrasting colour.
    Visualises the final set of health-related subreddits which forms the initial dataset for
    subsequent classification experiments.

    Parameters
    ----------
    tsne_array : array
        Array containing the locations of each subreddit in the final subreddit clustering dataset
        represented in 2D space.
    tsne_colour_mapping : tuple
        Tuple returned by the tsne_colours function containing the id_2_colour mapping and colour
        palette
    seeding_clusters : dataframe
        Dataframe containing the IDs, display names and cluster IDs for seeding subreddits.
    new_health_clusters : dataframe
        Dataframe containing the IDs, display names and cluster IDs non-seeding subreddits in the top
        health cluster.
    perplexity : int
        The perplexity value used for t-SNE.
    metric : str
        Distance metric to be used for t-SNE. Defaults to 'cosine'.
    method : str
        Indicates the method used to generate clusters. Either 'kmeans' or 'lda'.

    Returns
    -------

    """

    id_2_word, palette = tsne_colour_mapping

    if method == 'kmeans':
        label = 'k-means'
        loc = (0.7, 0.91)
    else:
        label = 'LDA'
        loc = (0.75, 0.91)

    seeding_colour = np.array([palette[-2]])
    new_health_colour = np.array([palette[-1]])

    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    ax.scatter(tsne_array[:, 0],
               tsne_array[:, 1],
               lw=0,
               s=30,
               c='#e3e1e1',  # Light grey
               )
    ax.scatter(tsne_array[list(new_health_clusters.index)][:, 0],
               tsne_array[list(new_health_clusters.index)][:, 1],
               lw=0,
               s=30,
               c=new_health_colour,
               )
    ax.scatter(tsne_array[list(seeding_clusters.index)][:, 0],
               tsne_array[list(seeding_clusters.index)][:, 1],
               lw=0,
               s=30,
               c=seeding_colour,
               )
    plt.xlim(-20, 20)
    plt.ylim(-20, 20)
    ax.text(loc[0], loc[1],
            label,
            fontsize=16,
            bbox=dict(facecolor='none',
                      edgecolor='lightgray',
                      alpha=0.5,
                      ),
            zorder=-1,
            transform=ax.transAxes,
            )
    ax.axis('tight')
    ax.axis('off')
    plt.savefig('tsne_summary_{}_{}_{}.png'.format(method,
                                                   perplexity,
                                                   metric,
                                                   ))
    plt.show()
    plt.close('all')


def tsne_percentages(tsne_array, corpus, method):
    """
    Generates t-SNE plot which colours subreddits according to the percentage of models capturing
    each subreddit in the top health cluster of models meeting performance thresholds (either k-means
    models only, LDA models only or both k-means and LDA mdoels)
    Darker points indicate a subreddit was common to the top health clusters of a large proportion
    of models, while lighter points indicate subreddit was captured by only a few models.

    Parameters
    ----------
    tsne_array : array
        Array containing the locations of each subreddit in the final subreddit clustering dataset
        represented in 2D space.
    corpus : dataframe
        Final subreddit clustering dataset
    method : str
        Indicates the method used to generate clusters. Either 'kmeans', 'lda' or 'both'.

    Returns
    -------

    """
    # Loads subreddit counts from file (file generated by subreddit_clustering_results.py)
    counts = pd.read_pickle('{}/{}/all_kmeans_lda_subs.pkl'.format(config.project_dir,
                                                                   config.clustering_dir, ))
    counts = pd.concat([corpus[['subreddit', 'health']], counts[['kmeans', 'lda', 'kmeans+lda']]], axis=1)
    counts.replace(np.nan,
                   0,
                   inplace=True,
                   )

    if method == 'kmeans':
        col = method
        label = 'k-means'
        loc = (0.7, 0.91)
        cmap = LinearSegmentedColormap.from_list(method, ['#e3e1e1'] + sns.cubehelix_palette(6,
                                                                                             start=2.4,
                                                                                             rot=0,
                                                                                             dark=0.1,
                                                                                             light=.8,
                                                                                             ).as_hex(),
                                                 N=100)
    elif method == 'lda':
        col = method
        label = 'LDA'
        loc = (0.75, 0.91)
        cmap = LinearSegmentedColormap.from_list(method,
                                                 ['#4B0504',
                                                  '#870826',
                                                  '#B6212E',
                                                  '#C53E3D',
                                                  '#D35A4A',
                                                  '#F4A886',
                                                  '#e3e1e1', ],
                                                 N=100,
                                                 ).reversed()
    else:
        col = 'kmeans+lda'
        label = 'k-means + lda'
        loc = (0.7, 0.91)
        cmap = LinearSegmentedColormap.from_list(method, ['#e3e1e1'] + sns.cubehelix_palette(6,
                                                                                             start=-0.1,
                                                                                             light=0.7,
                                                                                             ).as_hex(),
                                                 N=100)

    filename = '{}_tsne_pcts.png'.format(col)
    counts[col] = counts[col].apply(lambda x: int(round(x, 0)))

    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc2 = plt.scatter(tsne_array[list(counts.index)][:, 0],
                      tsne_array[list(counts.index)][:, 1],
                      lw=0,
                      s=30,
                      c=counts[col],
                      cmap=cmap,
                      zorder=0
                      )
    none = counts.loc[counts[col] == 0]
    sc3 = plt.scatter(tsne_array[list(none.index)][:, 0],
                      tsne_array[list(none.index)][:, 1],
                      lw=0,
                      s=30,
                      c=[cmap(0)],
                      zorder=1,
                      )
    some = counts.loc[counts[col] > 0, col]
    for i, pct in enumerate(some.sort_values(ascending=True).unique(), 2):
        data = counts.loc[counts[col] == pct]
        sc4 = plt.scatter(tsne_array[list(data.index)][:, 0],
                          tsne_array[list(data.index)][:, 1],
                          lw=0,
                          s=30,
                          c=[cmap(pct)],
                          zorder=i,
                          )

    cbar = plt.colorbar(sc2,
                        cmap=cmap,
                        ticks=[0, 50, 100],
                        fraction=0.02,
                        pad=0.0001,
                        orientation='horizontal'
                        )
    cbar.set_ticklabels(['0%', '50%', '100%'])
    ax.text(loc[0], loc[1],
            label,
            fontsize=16,
            bbox=dict(facecolor='none',
                      edgecolor='lightgray',
                      alpha=0.5,
                      ),
            zorder=-1,
            transform=ax.transAxes,
            )
    ax.axis('off')
    plt.tight_layout()
    cluster_dir()
    plt.savefig(filename, bbox_inches='tight')
    plt.show()
    plt.close('all')


if __name__ == '__main__':

    # Running TSNE with option to test all parameters
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
    parser.add_argument('-m', '--method',
                        help='plot T-SNE using kmeans or lda cluster ids',
                        type=str,
                        default='k',
                        choices=('k', 'l', 'none'),
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

    cluster_dir()
    subreddit_corpus = load_corpus(**sample_params)
    print(subreddit_corpus)

    if args.method == 'k':
        method = 'kmeans'
        final_cluster_params = params.final_kmeans_params
        # group_pct = pd.read_pickle('kmeans_best_counts.pkl')
    else:
        method = 'lda'
        final_cluster_params = params.final_lda_params
        # group_pct = pd.read_pickle('lda_best_counts.pkl')

    if args.all:
        # summary_file = 'sub_cluster_summary_ALL.json'
        vocab_params = params.vocab_params_all
        tsvd_params = params.tsvd_params_all
        tsne_params = params.tsne_params_all
    else:
        # summary_file = 'sub_cluster_summary_DEFAULT.json'
        vocab_params = [{'min_df': final_cluster_params['min_df'],
                         'max_df': final_cluster_params['max_df'],
                         }]
        tsvd_params = {'n_components': [final_cluster_params['tsvd_components']]}
        if params.debug_all_tsne:
            tsne_params = params.tsne_params_all
        else:
            tsne_params = {'perplexity': [params.tsne_params_default['perplexity']]}

    non_health_sub_idx = subreddit_corpus.loc[subreddit_corpus['health'] == 0].index

    # full_summary = json.load(open(summary_file, 'r'))
    print('All TF-IDF parameters: \n{}'.format(vocab_params))
    for vocab_dict in vocab_params:
        cluster_dir(**sample_params)
        vocab_dir(**vocab_dict)
        vocab_key = 'vocab_{}_{}'.format(vocab_dict['min_df'],
                                         vocab_dict['max_df'],
                                         )

        for n in tsvd_params['n_components']:
            print('\nTF-IDF parameters for current test: {}'.format(vocab_dict))
            print('T-SVD components: {}'.format(n))
            os.chdir('./tsvd_{}'.format(n))
            # tsvd_key = 'tsvd_{}'.format(n)

            for px in tsne_params['perplexity']:
                print('T-SNE perplexity: {}'.format(px))
                try:
                    tsne_model, tsne_r = load_tsne(perplexity=px)
                except FileNotFoundError:
                    # Performs T-SNE to visualise subreddits
                    tsvd_model, tsvd_r = load_tsvd(n)
                    tsne_model, tsne_r = tsne(tsvd_r,
                                              perplexity=px,
                                              )
                print('TSNE function: \n{}'.format(tsne_model))
                print('\n')

                clusters = final_cluster_params['n_clusters']
                top_health_cluster = final_cluster_params['top_health_cluster_id']
                health_sub_idx_seeding, health_sub_idx_full = load_clusters(subreddit_corpus,
                                                                            n_clusters=clusters,
                                                                            health_cluster=top_health_cluster,
                                                                            method=method,
                                                                            )

                tsne_colour_mappings = tsne_colours(health_sub_idx_seeding)
                tsne_cluster_plot(tsne_r,
                                  tsne_colour_mappings,
                                  health_sub_idx_seeding,
                                  method=method,
                                  perplexity=px,
                                  )
                tsne_cluster_plot(tsne_r,
                                  tsne_colour_mappings,
                                  health_sub_idx_seeding,
                                  health_sub_idx_full,
                                  method=method,
                                  perplexity=px,
                                  )
                tsne_summary_plot(tsne_r,
                                  tsne_colour_mappings,
                                  health_sub_idx_seeding,
                                  health_sub_idx_full,
                                  method=method,
                                  perplexity=px,
                                  )
                os.chdir('../tsvd_{}'.format(n))

    print('TSNE complete.')