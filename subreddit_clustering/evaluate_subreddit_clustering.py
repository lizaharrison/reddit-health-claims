#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
PROGRAM : evaluate_subreddit_clustering.py
AUTHOR : Eliza Harrison

This program computes evaluation metrics for subreddit clustering.

"""
import argparse
import json

import pandas as pd

from subreddit_clustering import subreddit_clustering_params as params
from subreddit_clustering.generate_subreddit_corpus import load_corpus, cluster_dir
from subreddit_clustering.run_lda import lda_dir
from subreddit_clustering.run_tfidf import vocab_dir
from subreddit_clustering.run_tsvd import tsvd_dir


def cluster_recall(cluster_df, corpus, n_samples=50):
    """
    Calculates average recall for top health cluster of k-means and LDA models.
    Splits seeding set into two samples (A and B). The cluster associated with the greatest number
    of seeding subreddits in sample A is the 'top health cluster'. The proportion of seeding subreddits
    in sample B assigned to this cluster corresponds to the recall. This is repeated over n rounds to
    compute the average recall for each model.

    Parameters
    ----------
    cluster_df : dataframe
        Contains clustering results
    corpus : dataframe
        Final subreddit clustering dataset
    n_samples : int
        Number of rounds over which to compute average recall.

    Returns
    -------
    average_stats : dataframe
        Contains the average recall and total cluster size values.

    """
    # Randomly sample health subs for hold-out set
    health_all = corpus.loc[corpus['health'] == 1]
    cluster_ids = cluster_df['top_topic']
    cluster_pcts = cluster_df['top_topic_pct']

    all_recall = {}
    for i in range(0, n_samples):
        print(i)
        sample_A = health_all.sample(n=len(health_all) // 2,
                                     random_state=i + 1,
                                     )
        sample_B = health_all.loc[~health_all.index.isin(sample_A.index)]

        sample_clusters = pd.Series(cluster_ids[sample_A.index]).astype(int).rename('sample_health').value_counts()
        print('\nCluster IDs for sample health subreddits: \n{}'.format(sample_clusters))
        hold_out_clusters = pd.Series(cluster_ids[sample_B.index]).rename('hold_out_health').value_counts()
        print('Cluster IDs for hold-out health subreddits: \n{}'.format(hold_out_clusters))

        all_recall[i] = {}
        all_recall[i]['sample_subs'] = list(zip(sample_clusters.index, sample_clusters.values))
        all_recall[i]['hold-out_subs'] = list(zip(hold_out_clusters.index, hold_out_clusters.values))

        for n in range(1, 4):
            top_n = list(sample_clusters.index[0:n])
            tp = hold_out_clusters.loc[top_n].sum()
            fn = hold_out_clusters.loc[~hold_out_clusters.index.isin(top_n)].sum()
            recall = round(tp / (tp + fn), 4)
            total_subs = len(cluster_df.loc[cluster_df['top_topic'].isin(top_n),
                                            ['subreddit',
                                             'top_topic',
                                             'top_topic_pct',
                                             ]])
            print('Recall @ {}: {}'.format(n, recall))
            print('Total subreddits in top {} clusters/topics: {}'.format(n, total_subs))
            all_recall[i]['recall@{}'.format(n)] = recall
            all_recall[i]['total@{}'.format(n)] = total_subs

    all_recall = pd.DataFrame(all_recall).transpose().rename_axis('random_sample')
    # pd.to_pickle(all_recall, 'recall_calculation_df.pkl')

    av_recall = [round(all_recall['recall@{}'.format(n)].mean(), 4) for n in range(1, 4)]
    av_total = [round(all_recall['total@{}'.format(n)].mean(), 4) for n in range(1, 4)]

    average_stats = pd.DataFrame(zip(av_recall, av_total),
                                 columns=['average_recall', 'total_subs'],
                                 index=['recall@{}'.format(n) for n in range(1, 4)],
                                 )

    return average_stats


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

    parser.add_argument('-m', '--method',
                        help='choose whether to evaluate kmeans or lda methods',
                        type=str,
                        default='k',
                        choices=('k', 'l'),
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
        lda_params = params.lda_params_all
        tsvd_params = params.tsvd_params_all
        kmeans_params = params.kmeans_params_all
        av_recall_pkl = 'average_recall_ALL.pkl'
        av_recall_csv = 'average_recall_ALL.csv'
    else:
        summary_file = 'sub_cluster_summary_DEFAULT.json'
        vocab_params = [params.vocab_params_default]
        lda_params = {'n_topics': [params.lda_params_default['n_topics']]}
        tsvd_params = {'n_components': [params.tsvd_params_default['n_components']]}
        kmeans_params = {'n_clusters': [params.kmeans_params_default['n_clusters']]}
        av_recall_pkl = 'average_recall_DEFAULT.pkl'
        av_recall_csv = 'average_recall_DEFAULT.csv'

    # Loads sample corpus from file
    subreddit_corpus = load_corpus(**sample_params)
    print(subreddit_corpus.loc[:, ['subreddit', 'clean_text']])

    health_subs = subreddit_corpus.loc[subreddit_corpus['health'] == 1]
    other_subs = subreddit_corpus.loc[subreddit_corpus['health'] == 0]
    print('Health subreddits: {}'.format([sub for sub in health_subs['subreddit']]))

    params_labels = ['min_df',
                     'max_df',
                     'tsvd_components',
                     'n_topics/clusters',
                     ]
    all_results_dfs = []

    # TF-IDF
    for vocab_dict in vocab_params:
        print('\nTF-IDF/vocabulary parameters: {}'.format(vocab_dict))
        full_summary = json.load(open(summary_file, 'r'))
        vocab_dir(action='set', **vocab_dict)
        vocab_key = 'vocab_{}_{}'.format(vocab_dict['min_df'],
                                         vocab_dict['max_df']
                                         )
        if args.method == 'l':
            method = 'lda'
            lda_dir(action='set')
            for n_topics in lda_params['n_topics']:
                try:
                    print('Number of LDA topics: {}'.format(n_topics))
                    lda_key = 'lda'.format(n_topics)
                    lda_summary = full_summary[vocab_key][lda_key]
                    lda_topic_df = pd.read_pickle('lda_topic_data_{}.pkl'.format(n_topics))
                    top_topics = lda_topic_df[['subreddit',
                                               'top_topic',
                                               'top_topic_pct',
                                               ]].drop_duplicates()
                    top_topics = top_topics.merge(subreddit_corpus.reset_index()[['index',
                                                                                  'subreddit',
                                                                                  ]],
                                                  how='left',
                                                  on='subreddit',
                                                  ).set_index('index')
                    final_stats = cluster_recall(lda_topic_df,
                                                 subreddit_corpus,
                                                 n_samples=3,
                                                 ).reset_index().rename(columns={'index': 'recall@n'})
                    results_dict = {
                        'average_recall': list(zip(final_stats.index,
                                                   final_stats['average_recall'],
                                                   )),
                        'average_total_subs': list(zip(final_stats.index,
                                                       final_stats['total_subs'],
                                                       )),
                    }
                    # lda_summary.append(results_dict)
                    full_summary[vocab_key][lda_key].append(results_dict)

                    params_values = [[vocab_dict['min_df'],
                                      vocab_dict['max_df'],
                                      'n/a',
                                      n_topics,
                                      ]] * 3
                    params_df = pd.DataFrame(params_values,
                                             columns=params_labels,
                                             )
                    results_df = pd.concat([params_df,
                                            final_stats],
                                           axis=1,
                                           )
                    results_df.index = [x for x in range(1, len(results_df) + 1)]
                    all_results_dfs.append(results_df)

                except FileNotFoundError:
                    print('Cannot locate files for LDA with {} topics'.format(n_topics))

        elif args.method == 'k':
            method = 'kmeans'
            for n_components in tsvd_params['n_components']:
                tsvd_dir(action='set',
                         n_components=n_components,
                         )
                print('Number of T-SVD components: {}'.format(n_components))
                tsvd_key = 'tsvd_{}'.format(n_components)

                for n_clusters in kmeans_params['n_clusters']:
                    try:
                        print('Number of K-Means clusters: {}'.format(n_clusters))
                        kmeans_key = 'kmeans'
                        kmeans_summary = full_summary[vocab_key][tsvd_key][kmeans_key]
                        cluster_labels = pd.read_pickle('kmeans_cluster_labels_{}'.format(n_clusters))

                        final_stats = cluster_recall(cluster_labels,
                                                     subreddit_corpus,
                                                     n_samples=50,
                                                     ).reset_index().rename(columns={'index': 'recall@n'})
                        results_dict = {
                            'average_recall': list(zip(final_stats.index,
                                                       final_stats['average_recall'],
                                                       )),
                            'average_total_subs': list(zip(final_stats.index,
                                                           final_stats['total_subs'],
                                                           )),
                        }
                        # kmeans_summary.append(results_dict)
                        full_summary[vocab_key][tsvd_key][kmeans_key].append(results_dict)
                        params_values = [[vocab_dict['min_df'],
                                          vocab_dict['max_df'],
                                          n_components,
                                          n_clusters,
                                          ]] * 3
                        params_df = pd.DataFrame(params_values,
                                                 columns=params_labels,
                                                 )
                        results_df = pd.concat([params_df,
                                                final_stats],
                                               axis=1,
                                               )
                        results_df.index = [x for x in range(1, len(results_df) + 1)]
                        all_results_dfs.append(results_df)

                    except FileNotFoundError:
                        print('Cannot locate files for K-Means with {} clusters'.format(n_clusters))

        cluster_dir(**sample_params)
        json.dump(full_summary, open(summary_file, 'w'))

    full_results_df = pd.concat(all_results_dfs)
    print(full_results_df)
    full_results_df.to_pickle('{}_{}'.format(method,
                                             av_recall_pkl,
                                             ))
    full_results_df.to_csv('{}_{}'.format(method,
                                          av_recall_csv,
                                          ))

