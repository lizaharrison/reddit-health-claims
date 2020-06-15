#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
PROGRAM : subreddit_clustering_results.py
AUTHOR : Eliza Harrison

This program generates output files for inspecting the results of subreddit clustering.

"""

import itertools
import os

import pandas as pd

from subreddit_clustering import subreddit_clustering_params as params
from subreddit_clustering.generate_subreddit_corpus import cluster_dir
from subreddit_clustering.run_tfidf import vocab_dir


def get_cluster_results(method='kmeans'):
    """
    Loads clustering results from files and transforms into dataframe containing
    recall and total subreddit values + top cluster IDs and list of subreddits within each cluster.

    Parameters
    ----------
    method : str
        Specifies whether to load results for k-means or LDA

    Returns
    -------
    final_df : dataframe
        Final results dataframe formatted for inspection
    """

    cluster_dir()
    if method == 'kmeans':
        recall_df = pd.read_pickle('kmeans_average_recall_ALL.pkl').drop('recall@n', axis=1)
        results_df = pd.read_pickle('kmeans_df.pkl')
        cluster_str = 'kmeans_clusters.pkl'
    elif method == 'lda':
        recall_df = pd.read_pickle('lda_average_recall_ALL.pkl').drop('recall@n', axis=1)
        results_df = pd.read_pickle('lda_df.pkl')
        cluster_str = 'lda_topics.pkl'
    else:
        raise ValueError('Please specify "kmeans" or "lda" for method parameter')

    recall_df['key'] = recall_df[['min_df', 'max_df', 'n_topics/clusters']].astype(str).agg('_'.join, axis=1)
    recall_df['key'] = recall_df['key'].apply(lambda x: '{}_{}'.format(method,
                                                                       x))
    recall_df.reset_index(inplace=True)
    recall_df.rename({'index': 'k'}, axis=1, inplace=True)
    recall_df.set_index('key', inplace=True, drop=False)

    recall = recall_df.pivot(columns='k', values='average_recall')
    total_subs = recall_df.pivot(columns='k', values='total_subs')
    recall_subs = recall.join(total_subs,
                              how='outer',
                              lsuffix='_av_recall',
                              rsuffix='_av_subs',
                              )
    recall_df = recall_df[['min_df', 'max_df', 'tsvd_components', 'n_topics/clusters']].drop_duplicates()
    recall_df = recall_df.join(recall_subs, how='outer')

    cluster_data = {}
    for vocab_dict in params.vocab_params_all:
        cluster_dir()
        vocab_dir(**vocab_dict)
        if method == 'kmeans':
            os.chdir('./tsvd_500')
            tsvd_components = 500
            clusters = pd.read_pickle('all_kmeans_clusters.pkl')
        elif method == 'lda':
            os.chdir('./lda')
            clusters = pd.read_pickle('all_lda_topics.pkl')
            tsvd_components = 'n/a'

        clusters.columns = clusters.columns.droplevel(0)
        health_clusters = clusters.loc[clusters.index.get_level_values(1) == 1]
        min_df = vocab_dict['min_df']
        max_df = vocab_dict['max_df']

        for col in clusters.columns:
            key = recall_df.loc[(recall_df['min_df'] == min_df) &
                                (recall_df['max_df'] == max_df) &
                                (recall_df['n_topics/clusters'] == col)].index[0]
            cluster_1 = health_clusters[col].value_counts().index[0]
            cluster_2 = health_clusters[col].value_counts().index[1]
            cluster_3 = health_clusters[col].value_counts().index[2]
            subs_1 = [sub[0] for sub in list(clusters.loc[clusters[col] == cluster_1].index)]
            subs_2 = [sub[0] for sub in list(clusters.loc[clusters[col] == cluster_2].index)]
            subs_3 = [sub[0] for sub in list(clusters.loc[clusters[col] == cluster_3].index)]

            cluster_data[key] = {
                '1_cluster_id': cluster_1,
                '2_cluster_id': cluster_2,
                '3_cluster_id': cluster_3,
                '1_health_subs': subs_1,
                '2_health_subs': subs_2,
                '3_health_subs': subs_3
            }

    final_df = pd.DataFrame.from_dict(cluster_data,
                                      orient='index',
                                      )
    final_df = recall_df.join(final_df, how='outer')
    final_df.sort_values(['min_df', 'n_topics/clusters'], inplace=True)

    return final_df


'''
cluster_data = recall_df.loc[(recall_df['min_df'] == min_df) &
                                         (recall_df['tsvd_components'] == tsvd_components) &
                                         (recall_df['n_topics/clusters'] == col)]
            recall = pd.DataFrame(cluster_data['average_recall']).transpose()
            recall.rename({'average_recall': cluster_data['key'][1]}, inplace=True)
            recall.columns = ['recall@1', 'recall@2', 'recall@3']
            total_subs = pd.DataFrame(cluster_data['total_subs']).transpose()
            total_subs.rename({'total_subs': cluster_data['key'][1]}, inplace=True)
            total_subs.columns = ['subs@1', 'subs@2', 'subs@3']
'''


def get_best_models(results_df, min_recall, max_subs, recall=1, drop=[]):
    """
    Filters top 1, top 2 or top 3 health cluster results for best_df performing parameters according to pre-specified minimum
    recall and cluster size threshold values.

    Parameters
    ----------
    results_df : dataframe
        Final subreddit clustering results dataframe (pre-formatted)
    min_recall : float
        Minimum recall threshold
    max_subs : int
        Maximum number of subreddits in top health cluster
    recall : int
        Number of clusters to examine
    drop : list
        List of models to exclude from analyses

    Returns
    -------
    best_df : dataframe
        Models meeting performance thresholds (top performing models)

    """

    ix_cols = ['min_df', 'max_df', 'tsvd_components', 'n_topics/clusters']
    final_cols = ix_cols + ['recall', 'total_subs', 'possible_health_subs']
    results_df['1_cluster_size'] = results_df['1_health_subs'].apply(lambda x: len(x))
    results_df['2_cluster_size'] = results_df['2_health_subs'].apply(lambda x: len(x))
    results_df['3_cluster_size'] = results_df['3_health_subs'].apply(lambda x: len(x))

    if recall == 1:
        best_df = results_df.loc[(results_df['1_av_recall'] > min_recall) &
                              (results_df['1_cluster_size'] < max_subs)]
        best_df = best_df[ix_cols + ['1_av_recall', '1_cluster_size', '1_health_subs']]
        best_df.columns = final_cols
    elif recall == 2:
        best_df = results_df.loc[(results_df['2_av_recall'] > min_recall) &
                              (results_df['2_cluster_size'] + results_df['1_cluster_size'] < max_subs)]
        best_df['recall'] = best_df['2_av_recall']
        best_df['total_subs'] = best_df['2_cluster_size'] + best_df['1_cluster_size']
        best_df['possible_health_subs'] = best_df['1_health_subs'] + best_df['2_health_subs']
        best_df = best_df[final_cols]
    else:
        best_df = results_df.loc[(results_df['3_av_recall'] > min_recall) &
                              (results_df['3_cluster_size'] + results_df['2_cluster_size'] + results_df[
                                  '1_cluster_size'] < max_subs)]
        best_df['recall'] = best_df['3_av_recall']
        best_df['total_subs'] = best_df['3_cluster_size'] + best_df['2_cluster_size'] + best_df['1_cluster_size']
        best_df['possible_health_subs'] = best_df['3_health_subs'] + best_df['2_health_subs'] + best_df['1_health_subs']
        best_df = best_df[final_cols]
    if len(drop) > 0:
        best_df.drop(drop,
                  inplace=True,
                  )

    return best_df


def compare_subs(best_df, method='kmeans', compare_methods=False):
    """
    Compares the subreddits present or absent between each of the best performing groups.
    Allows for analysis of edge cases (differences in the subreddits captured by the top
    health clusters of different models).

    Parameters
    ----------
    best_df : dataframe
        Dataframe containing subreddit clustering models meeting performance thresholds
    method : str
        Type of model (either 'lda' or 'kmeans') to analyse
    compare_methods : bool
        Whether to compare the results of k-means and LDA methods or simply analyse k-means or
        LDA models

    Returns
    -------
    compare_dict : dictionary
        Contains lists of subreddits common to or differing between pairs of models e.g. subreddit
        x is present in top health cluster of model A but absent in top health cluster of model B

    """

    if method == 'kmeans':
        m = 'K-Means'
    else:
        m = 'LDA'

    print('Best {} results: \n{}'.format(m,
                                         best_df,
                                         ))
    compare_dict = {}
    for pair in itertools.combinations(best_df.index, 2):
        subs_x = best_df.loc[pair[0], 'possible_health_subs']
        subs_y = best_df.loc[pair[1], 'possible_health_subs']
        gap_xy = [sub for sub in subs_x if sub not in subs_y]
        print('Subreddits in {} but not in {}: \n{}'.format(pair[0],
                                                            pair[1],
                                                            gap_xy,
                                                            ))
        key = '{} vs. {}'.format(pair[0],
                                 pair[1],
                                 )
        compare_dict[key] = gap_xy
        gap_yx = [sub for sub in subs_y if sub not in subs_x]
        print('Subreddits in {} but not in {}: \n{}'.format(pair[1],
                                                            pair[0],
                                                            gap_yx,
                                                            ))
        key = '{} vs. {}'.format(pair[1],
                                 pair[0],
                                 )
        compare_dict[key] = gap_yx

    return compare_dict


def get_counts(best_df):
    """
    Compiles all subreddits captured by the top health clusters of the top performing models and
    computes the percentage of models capturing each.

    Parameters
    ----------
    best_df : dataframe
        Dataframe containing subreddit clustering models meeting performance thresholds

    Returns
    -------
    all_counts : dataframe
        Contains the percentage of models capturing each subreddit in the top health cluster, as well
        as compiling all subreddits captured by each top performing model into a single dataframe for
        review.

    """
    all_subs = pd.DataFrame(best_df['possible_health_subs'].explode())
    distinct_subs = all_subs.drop_duplicates('possible_health_subs')
    all_counts = all_subs['possible_health_subs'].value_counts()
    groups_df = pd.DataFrame([],
                             index=distinct_subs['possible_health_subs'].values,
                             )
    for ix in best_df.index:
        group_subs = best_df.loc[ix, 'possible_health_subs']
        groups_df[ix] = groups_df.index.isin(group_subs).astype(int)

    all_counts = pd.concat([all_counts, groups_df], axis=1).reset_index()
    all_counts.rename({'possible_health_subs': 'count',
                       'index': 'subreddit'}, axis=1, inplace=True)
    all_counts['count'] = all_counts['count'].apply(lambda x: x / len(best_df)) * 100
    '''
    all_subs = pd.DataFrame(best_df['possible_health_subs'].explode())
    all_subs.reset_index(inplace=True)
    all_counts = all_subs['possible_health_subs'].value_counts()
    all_groups = all_subs.groupby('possible_health_subs')['index'].unique()
    all_counts = pd.concat([all_counts, all_groups],
                           axis=1,
                           sort=True,
                           ).reset_index()
    all_counts.columns = ['subreddit',
                          'count',
                          'groups_in',
                          ]
    all_counts['count'] = all_counts['count'].apply(lambda x: x / len(best_df)) * 100
    all_counts = pd.DataFrame(all_counts)
    '''
    return all_counts


if __name__ == '__main__':
    sample_params = params.subreddit_cluster_params
    cluster_dir(**sample_params)
    subreddit_corpus = pd.read_pickle('subreddit_sample_dataset_CLEAN_100_1.pkl')
    health_corpus = subreddit_corpus.loc[subreddit_corpus['health'] == 1]

    kmeans_results = get_cluster_results(method='kmeans')
    kmeans_results.drop([ix for ix in kmeans_results.index if
                         (ix.split('_')[2] == '1000') or
                         (ix.split('_')[1] == '7')],
                        inplace=True,
                        )
    kmeans_results.to_pickle('kmeans_results.pkl')
    lda_results = get_cluster_results(method='lda')
    lda_results.drop([ix for ix in lda_results.index if
                      (ix.split('_')[2] == '1000') or
                      (ix.split('_')[1] == '7')],
                     inplace=True,
                     )
    lda_results.to_pickle('lda_results.pkl')

    cluster_dir()
    min_recall = 0.7
    max_subs = 1500

    best_models = {}
    all_sub_counts = []
    for i in range(1, 4)[0:1]:
        print(i)
        kmeans_key = 'kmeans_{}'.format(i)
        lda_key = 'lda_{}'.format(i)
        best_models[kmeans_key] = get_best_models(kmeans_results,
                                                  min_recall,
                                                  max_subs,
                                                  # drop=kmeans_drop[i],
                                                  recall=i,
                                                  )

        best_models[lda_key] = get_best_models(lda_results,
                                               min_recall,
                                               max_subs,
                                               # drop=lda_drop[i],
                                               recall=i,
                                               )

        counts_at = {}
        counts_at[kmeans_key] = get_counts(best_models[kmeans_key])
        counts_at[kmeans_key].rename({'count': 'kmeans'}, axis=1, inplace=True)

        counts_at[lda_key] = get_counts(best_models['lda_{}'.format(i)])
        counts_at[lda_key.format(i)].rename({'count': 'lda'}, axis=1, inplace=True)
        counts_at['both_{}'.format(i)] = get_counts(pd.concat([best_models[kmeans_key],
                                                               best_models[lda_key]]))
        counts_at['both_{}'.format(i)].rename({'count': 'kmeans+lda'}, axis=1, inplace=True)
        final_counts = counts_at[kmeans_key][['subreddit', 'kmeans']].merge(counts_at[lda_key][['subreddit', 'lda']],
                                                                            how='outer',
                                                                            on='subreddit',
                                                                            ).merge(counts_at['both_{}'.format(i)],
                                                                                    how='outer',
                                                                                    on='subreddit',
                                                                                    )
        final_counts.rename({0: 'kmeans',
                             1: 'lda',
                             2: 'kmeans+lda',
                             },
                            axis=1,
                            inplace=True,
                            )
        final_counts.sort_values(['kmeans+lda',
                                  'kmeans',
                                  'lda',
                                  'subreddit',
                                  ], ascending=[False,
                                                False,
                                                False,
                                                True,
                                                ],
                                 inplace=True,
                                 )

        final_counts.to_csv('all_sub_counts@{}.csv'.format(i), index=False)
        '''
        print(pd.concat([best_models[kmeans_key],
                         best_models[lda_key]]).loc[final_counts.columns[4:], 'recall'])
        '''
        all_sub_counts.append(final_counts)

    print('Best K-Means results (@1 cluster): \n{}'.format(best_models['kmeans_1']))
    # print('Best K-Means results (@2 clusters): \n{}'.format(best_models['kmeans_2']))
    # print('Best K-Means results (@3 clusters): \n{}'.format(best_models['kmeans_3']))

    print('Best LDA results (@1 cluster): \n{}'.format(best_models['lda_1']))
    # print('Best LDA results (@2 clusters): \n{}'.format(best_models['lda_2']))
    # print('Best LDA results (@3 clusters): \n{}'.format(best_models['lda_3']))

    all_sub_counts = pd.concat(all_sub_counts)
    all_sub_counts.drop_duplicates(inplace=True)
    all_sub_counts = all_sub_counts.iloc[:, 0:4]
    all_sub_counts = subreddit_corpus[['subreddit', 'health']].merge(all_sub_counts,
                                                                     how='left',
                                                                     on='subreddit',
                                                                     )
    all_sub_counts.fillna('0', inplace=True)
    all_sub_counts.to_pickle('all_kmeans_lda_pcts.pkl')

    print('Inspection of subreddit clustering results complete.')
