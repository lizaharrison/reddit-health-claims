# !/usr/bin/python
# -*- coding: utf-8 -*-

"""
PROGRAM : evaluate_classification_results.py
AUTHOR : Eliza Harrison

Evaluates the performance of trained classifiers for the labelling of
Reddit threads containing health claims.

"""

import itertools
import os
import pickle
from colorsys import rgb_to_hls

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score, roc_curve, auc

from health_claims_classification.health_claims_classifiers import feature_set, grid_search_full

rainbow_list = [
    # '#191331',  # Dark navy
    # '#007076',  # Dark teal
    '#0000c8',  # Medium blue / LR
    '#5ac08b',  # Light teal / NB
    # '#72a425',  # Green
    # '#e4b354',  # Gold
    '#de7348',  # Orange / RF
    '#9d000b',  # Red /SVM_linear
    '#67135d',  # Purple / SVM_rbf
]

cfx_cmaps = {
    'NB': ('#bbedd4', '#51bd85', '#14633a'),  # NB
    'LR': ('#ababf5', '#2222ba', '#080866'),  # LR
    'RF': ('#f5c0ab', '#d45b2a', '#9c3811'),  # RF
    'SVM_linear': ('#f5c1c5', '#ab0914', '#750008'),  # SVM_linear
    'SVM_rbf': ('#f2c4ed', '#822c78', '#5e1456'),  # SVM_rbf
}

classifier_names = {
    'LR': 'Logistic Regression',
    'NB': 'Naive Bayes',
    'RF': 'Random Forest',
    'SVM_linear': 'Linear SVM',
    'SVM_rbf': 'Radial Bias Function SVM',
}


def hex_to_hls(hex):
    h = hex.lstrip('#')
    r, g, b = tuple(int(h[i:i + 2], 16) / 225 for i in (0, 2, 4))
    h, l, s = rgb_to_hls(r, g, b)

    return h, l, s


def plot_confusion_x(classifier_key, Y_predict, Y_true):
    # Generate colormap for estimator to match colour scheme in ROC and PR curve plots
    # seed_colour = (seed_colour[0], seed_colour[1], seed_colour[2] - 0.2)
    # lighter = (seed_colour[0], seed_colour[1] + 0.5, seed_colour[2] - 0.3)
    # darker = (seed_colour[0], seed_colour[1] - 0.05, seed_colour[2] - 0.2)
    # all_cols_hls = (lighter, seed_colour, darker)
    # all_cols_rgb = [hls_to_rgb(col[0], col[1], col[2]) for col in all_cols_hls]
    '''
    # Preview colourmap
    fig, ax = plt.subplots(figsize=(5, 1))
    norm = Normalize(vmin=0, vmax=100)
    cb = ColorbarBase(ax,
                      cmap=cmap,
                      norm=norm,
                      orientation='horizontal',)
    plt.title(classifier_name)
    plt.show()
    '''

    classifier_name_full = classifier_names[classifier_key]
    cmap = LinearSegmentedColormap.from_list('clf_cm',
                                             cfx_cmaps[classifier_key],
                                             N=256, )
    # np.set_printoptions(precision=2)
    cf_x = confusion_matrix(Y_true, Y_predict)
    print('Non-normalised confusion matrix: \n{}'.format(cf_x))
    cf_x = cf_x.astype('float') / cf_x.sum(axis=1)[:, np.newaxis]
    cf_x = np.around(cf_x, decimals=2)
    print('Normalised confusion matrix: \n{}'.format(cf_x))
    n_classes = cf_x.shape[0]

    plt.rcParams.update({'font.size': 13})
    fig, ax = plt.subplots()
    plt.title(classifier_name_full)
    im = ax.imshow(cf_x, interpolation='nearest', cmap=cmap)
    im.set_clim(0, 1.0)
    cmap_min, cmap_max = im.cmap(0), im.cmap(256)
    thresh = (cf_x.max() + cf_x.min()) / 2.0
    for i, j in itertools.product(range(n_classes), range(n_classes)):
        color = 'black' if cf_x[i, j] < thresh else 'white'
        ax.text(j, i,
                format(cf_x[i, j], '.2g'),
                ha='center', va='center',
                color=color)
    cbar_ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    cbar = fig.colorbar(im, ax=ax, ticks=cbar_ticks)
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels([str(x) for x in cbar_ticks])
    ticks = (0, 1)
    ax.set(xticks=ticks,
           yticks=ticks,
           xticklabels=ticks,
           yticklabels=ticks,
           ylabel='True label',
           xlabel='Predicted label',

           )
    ax.set_ylim((n_classes - 0.5, -0.5))
    plt.tight_layout()
    plt.savefig('confusion_matrix_{}.png'.format(classifier_key), bbox='tight')
    plt.show()
    plt.close('all')


def gen_curves(clf_names, clf_estimators, X_test, Y_test, curve_type='PR'):
    curve_dict = []

    for i, classifier in enumerate(clf_names):
        estimator = clf_estimators[classifier]
        Y_true = np.array(Y_test)
        Y_predict = estimator.predict(X_test)

        if hasattr(estimator, 'decision_function'):
            Y_proba = estimator.decision_function(X_test)
        else:
            Y_proba = np.array(estimator.predict_proba(X_test).tolist())[:, 1]

        if curve_type == 'ROC':
            fpr, tpr, _ = roc_curve(Y_true, Y_proba)
            auc_var = roc_auc_score(Y_test, Y_predict)
            curve_dict.append(pd.DataFrame({
                'classifier': classifier,
                'fpr': fpr,
                'tpr': tpr,
                'auc': auc_var,
            }))
        else:
            precision, recall, _ = precision_recall_curve(Y_test, Y_proba)
            auc_var = auc(recall, precision)
            curve_dict.append(pd.DataFrame({
                'classifier': classifier,
                'precision': precision,
                'recall': recall,
                'baseline': len(Y_test[Y_test == 1]) / len(Y_test),
                'auc': auc_var,
            }))

    curve_df = pd.concat(curve_dict).set_index('classifier')

    return curve_df


def plot_curves(roc_curve, pr_curve, feature_set='text'):
    """
    Advantages of precision-recall curve vs ROC curve on imbalanced datasets for binary classification
    Saito & Rehmsmeier (2015) The Precision-Recall Plot Is More Informative than the ROC Plot When
    Evaluating Binary Classifiers on Imbalanced Datasets, PLOS One, 10(3): e0118432.
    https://doi.org/10.1371/journal.pone.0118432

    :param roc_curve:
    :param pr_curve:
    :param feature_set:
    :return:
    """

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    ax1, ax2 = axes

    # Plot ROC for trained estimators when tested on the unseen hold-out set
    lw = 1.5
    ax1.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
    i = 0
    for name, group in roc_curve.groupby('classifier'):
        ax1.plot(group['fpr'],
                 group['tpr'],
                 label='{} (AUC={:.3f})'.format(name, group.iloc[0, -1]),
                 color=rainbow_list[i],
                 lw=lw, )
        i += 1
    plt.rcParams.update({'font.size': 14})
    ax1.set_xlim((0.0, 1.0))
    ax1.set_ylim((0.0, 1.05))
    # ax1.set_yticks(np.arange(0.2, 1.0, 0.2))
    ax1.set_xlabel('False Positive Rate', fontsize=14)
    ax1.set_ylabel('True Positive Rate', fontsize=14)
    ax1.title.set_text('ROC Curve')
    ax1.legend(loc='lower right', fontsize=12)

    # Plot precision-recall for trained estimators when tested on the unseen hold-out set
    lw = 1.5
    baseline = pr_curve['baseline'].iloc[0]
    ax2.plot([0, 1], [baseline, baseline], color='black', lw=lw, linestyle='--')
    i = 0
    for name, group in pr_curve.groupby('classifier'):
        print(name)
        print(group)
        ax2.plot(group['recall'],
                 group['precision'],
                 label='{} (AUC={:.3f})'.format(name, group.iloc[0, -1]),
                 color=rainbow_list[i],
                 lw=lw, )
        i += 1
    # random = len(X_test[X_test == 1]) / len(X_test)
    ax2.set_xlim([0., 1.0])
    ax2.set_ylim([0.0, 1.05])
    # ax2.set_yticks(np.arange(0.2, 1.0, 0.2))
    ax2.set_xlabel('Recall', fontsize=14)
    ax2.set_ylabel('Precision', fontsize=14)
    ax2.title.set_text('P-R Curve')
    ax2.legend(loc='lower left', fontsize=12)

    plt.tight_layout()
    plt.savefig('ROC_PR_curves_hold_out_{}.png'.format(feature_set), bbox='tight')
    plt.show()
    plt.close('all')


if __name__ == '__main__':

    wd = '/health_claims_classification'
    if os.getcwd().endswith(wd):
        pass
    else:
        os.chdir('.' + wd)

    # INITIALISING DATASETS AND RESULTS #
    text_by_thread_w_test = pd.read_pickle('text_by_thread.pkl')
    X_train, Y_train = pickle.load(open('training_datasets.pkl', 'rb'))
    X_test, Y_test = pickle.load(open('hold_out_test_datasets.pkl', 'rb'))
    all_results = pickle.load(open('classification_results_text_non-nested.pkl'.format(feature_set), 'rb'), )
    best_estimators = pickle.load(open('trained_estimators_text_non-nested.pkl'.format(feature_set), 'rb'), )
    thesis_table = pd.read_pickle('thesis_results_table_non-nested.pkl')

    # PLOT CONFUSION MATRICES FOR ALL CLASSIFIERS #
    for i, key in enumerate(classifier_names.keys()):
        clf_name_full = classifier_names[key]
        estimator = best_estimators[key]
        results = all_results[key]
        Y_true = Y_test
        Y_predict = results['y_predict']
        print('\n{}'.format(clf_name_full))
        plot_confusion_x(key, Y_predict, Y_true, )

    # PLOT ROC-AUC AND PRECISION-RECALL CURVES FOR ALL CLASSIFIERS #
    roc_curve_df = gen_curves(classifier_names.keys(),
                              best_estimators,
                              X_test,
                              Y_test,
                              curve_type='ROC', )
    pr_curve_df = gen_curves(list(grid_search_full.keys()),
                             best_estimators,
                             X_test,
                             Y_test,
                             curve_type='PR', )
    plot_curves(roc_curve_df, pr_curve_df, feature_set)

    # Top classifiers = SVM_linear + LR
    top_clf_name = 'SVM_linear'
    top_pipe = best_estimators[top_clf_name]
    top_clf = top_pipe['classifier']

    # Analysis of top weighted features
    features = top_pipe['tfidf'].get_feature_names()
    feature_cfs = top_clf.coef_[0]
    feature_coef_df = pd.Series(feature_cfs, index=features).sort_values(ascending=False)
    top_20 = feature_coef_df.iloc[0:20]

    # Analysis of test threads
    test_qs = text_by_thread_w_test.loc[text_by_thread_w_test['send_to_mturk'] == 2]
    X_test_2 = X_test.reset_index(drop=True).reset_index()
    test_qs = test_qs.merge(X_test_2, how='left', on='text')
    top_Y_predict = all_results[top_clf_name]['y_predict'][-20:]
    test_qs['Y_predict'] = top_Y_predict

    incorrect_test_clf = test_qs.loc[test_qs['Y_predict'] == 0]
    correct_test_clf = test_qs.loc[test_qs['Y_predict'] == 1]

    worker_answers = pd.read_pickle('worker_test_q_answers.pkl')
    correct_test_worker = worker_answers.loc[worker_answers['yes_count'] > worker_answers['no_count']]
    incorrect_test_worker = worker_answers.loc[worker_answers['no_score'] > worker_answers['yes_score']]

