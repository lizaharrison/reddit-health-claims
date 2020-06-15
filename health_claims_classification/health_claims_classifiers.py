# !/usr/bin/python
# -*- coding: utf-8 -*-

"""
PROGRAM : health_claims_classifiers.py
AUTHOR : Eliza Harrison

This program contains the code required to run experiments for the classification of threads
based on the presence or absence of health claims.

"""

import argparse
import os
import pickle

import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import make_scorer
from sklearn.metrics import precision_score, average_precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_validate
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC

random_state = 123
hold_out_size = 0.15
feature_set = 'text'
n_folds = 5

arg_dict = {
    'nb': 'NB',
    'lr': 'LR',
    'svml': 'SVM_linear',
    'svmr': 'SVM_rbf',
    'rf': 'RF',
}

classifier_names = {
    'LR': 'Logistic Regression',
    'NB': 'Naive Bayes',
    'RF': 'Random Forest',
    'SVM_linear': 'Linear SVM',
    'SVM_rbf': 'Radial Bias Function SVM',
}

rainbow_list = [
    # '#191331',  # Dark navy
    # '#007076',  # Dark teal
    '#0000c8',  # Medium blue / NB
    '#5ac08b',  # Light teal / LR
    # '#72a425',  # Green
    # '#e4b354',  # Gold
    '#de7348',  # Orange / RF
    '#9d000b',  # Red /SVM_linear
    '#67135d',  # Purple / SVM_rbf
]

cfx_cmaps = {
    'LR': ('#bbedd4', '#51bd85', '#14633a'),  # LR
    'NB': ('#ababf5', '#2222ba', '#080866'),  # NB
    'RF': ('#f5c0ab', '#d45b2a', '#9c3811'),  # RF
    'SVM_linear': ('#f5c1c5', '#ab0914', '#750008'),  # SVM_linear
    'SVM_rbf': ('#f2c4ed', '#822c78', '#5e1456'),  # SVM_rbf
}

# Subset of params (for TESTING)
grid_search_test = {
    'SVM_rbf': {
        'tfidf__max_features': (1000, 5000, 10000, None),
        'tfidf__ngram_range': ((1, 1), (1, 2), (1, 3)),
        'classifier': (SVC(kernel='rbf',
                           gamma='auto',
                           random_state=random_state, ),),
        'classifier__C': (100.0, 1000.0, 10000.0),
    },
}
# Full set of params (for EXPERIMENTS)
grid_search_full = {
    'NB': {
        'tfidf__max_features': (1000, 5000, 10000, None),
        'tfidf__ngram_range': ((1, 1), (1, 2), (1, 3)),
        'classifier': (ComplementNB(),),
    },
    'LR': [
        {
            'tfidf__max_features': (1000, 5000, 10000, None),
            'tfidf__ngram_range': ((1, 1), (1, 2), (1, 3)),
            'classifier': (LogisticRegression(random_state=random_state,
                                              max_iter=10000,
                                              solver='liblinear',
                                              class_weight='balanced',
                                              ),),
            'classifier__C': (1.0, 10.0, 100.0),
            'classifier__penalty': ('l1', 'l2'),
        },
        {
            'tfidf__max_features': (1000, 5000, 10000, None),
            'tfidf__ngram_range': ((1, 1), (1, 2), (1, 3)),
            'classifier': (LogisticRegression(random_state=random_state,
                                              max_iter=10000,
                                              class_weight='balanced',
                                              penalty='l2',
                                              ),),
            'classifier__C': (1.0, 10.0, 100.0),
            'classifier__solver': ('newton-cg', 'sag', 'lbfgs'),
        },
        {
            'tfidf__max_features': (1000, 5000, 10000, None),
            'tfidf__ngram_range': ((1, 1), (1, 2), (1, 3)),
            'classifier': (LogisticRegression(random_state=random_state,
                                              max_iter=10000,
                                              class_weight='balanced',
                                              solver='saga',
                                              penalty='l1',
                                              ),),
            'classifier__C': (1.0, 10.0, 100.0),
        },
    ],
    'SVM_linear': {
        'tfidf__max_features': (1000, 5000, 10000, None),
        'tfidf__ngram_range': ((1, 1), (1, 2), (1, 3)),
        'classifier': (LinearSVC(max_iter=10000,
                                 random_state=random_state, ),),
        'classifier__C': (1.0, 10.0, 100.0),
        'classifier__loss': ('hinge', 'squared_hinge')
    },
    'SVM_rbf': {
        'tfidf__max_features': (1000, 5000, 10000, None),
        'tfidf__ngram_range': ((1, 1), (1, 2), (1, 3)),
        'classifier': (SVC(kernel='rbf',
                           random_state=random_state, ),),
        'classifier__C': (1.0, 10.0, 100.0),
        'classifier__gamma': ('scale', 'auto'),
    },
    'RF': {
        'tfidf__max_features': (1000, 5000, 10000, None),
        'tfidf__ngram_range': ((1, 1), (1, 2), (1, 3)),
        'classifier': (RandomForestClassifier(random_state=random_state,
                                              max_depth=40,
                                              min_samples_split=2,
                                              bootstrap=True,
                                              class_weight='balanced',
                                              ),),
        'classifier__n_estimators': [int(x) for x in range(1500, 2010, 100)],
        # 'classifier__max_depth': (20, 40, 60),
        'classifier__criterion': ('entropy', 'gini'),
        # 'classifier__class_weight': ('balanced', None),
    },
}


class DenseTransformer(TransformerMixin):

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X.todense()


def clf_report_scorer(y_true, y_predict):
    print(classification_report(y_true, y_predict,
                                zero_division=0, ))
    return 0


def confusion_x_scorer(y_true, y_predict):
    # [[true negatives, false negatives]
    # [true positives, false positives]]

    print(confusion_matrix(y_true, y_predict, ))
    return 0


def precision(y_true, y_predict):
    return precision_score(y_true, y_predict, zero_division=0, )


def recall(y_true, y_predict):
    return recall_score(y_true, y_predict, zero_division=0, )


def f1(y_true, y_predict):
    return f1_score(y_true, y_predict, zero_division=0, )


def weighted_avg_precision(y_true, y_predict):
    return average_precision_score(y_true, y_predict, average='weighted', )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--classifier',
                        help='specify the classifier for which to run the experiments',
                        type=str,
                        choices=('lr', 'nb', 'rf', 'svml', 'svmr', 'all'),
                        default='all',
                        )
    parser.add_argument('-n', '--nested',
                        help='specify whether to use nested cross-validation',
                        action='store_true'
                        )

    args = parser.parse_args()
    if args.classifier == 'all':
        grid_search_params = grid_search_full
    else:
        grid_search_params = grid_search_full[arg_dict[args.classifier]]

    file_sfx = 'nested' if args.nested else 'non-nested'

    # INITIALISING DATA #
    os.chdir('./health_claims_classification')
    text_by_thread_w_test = pd.read_pickle('text_by_thread.pkl')
    test_qs = text_by_thread_w_test.loc[text_by_thread_w_test['send_to_mturk'] == 2]
    text_by_thread_no_test = text_by_thread_w_test.drop(test_qs.index)
    thesis_table = []

    # TRAIN / TEST DATASET GENERATION #
    X = text_by_thread_no_test[feature_set]
    Y = text_by_thread_no_test['health_claim']
    l_ix = text_by_thread_no_test['post_fullname']
    l_y = range(0, len(l_ix))

    # Splits into TRAIN (for cross-validation) and TEST (for hold-out test set)
    X_train, X_test, Y_train, Y_test, ix_train, ix_test = train_test_split(X,
                                                                           Y,
                                                                           l_ix,
                                                                           test_size=hold_out_size,
                                                                           random_state=random_state,
                                                                           stratify=Y.values,
                                                                           )

    X_test = pd.concat([X_test, test_qs['text']], axis=0)
    Y_test = pd.concat([Y_test, test_qs['health_claim']], axis=0)
    health_claim_in_train = Y_train.loc[Y_train == 1]
    health_claim_in_test = Y_test.loc[Y_test == 1]

    # Saves training and test data to pickle files
    pickle.dump((X_train, Y_train), open('training_datasets.pkl', 'wb'))
    pickle.dump((X_test, Y_test), open('hold_out_test_datasets.pkl', 'wb'))

    # SCORING METRICS FOR CLASSIFICATION EXPERIMENTS #
    refit_score = 'Precision'
    all_scorer_names = ['Precision',
                        'Recall',
                        'F1',
                        'Accuracy',
                        'Average_Recall',
                        'Weighted_Average_Precision',
                        # 'Weighted_Average_F1',
                        'ROC_AUC',
                        'Classification_report',
                        'Confusion_matrix', ]
    all_scorer_funcs = [precision,
                        recall,
                        f1,
                        accuracy_score,
                        balanced_accuracy_score,
                        weighted_avg_precision,
                        # weighted_avg_f1,
                        roc_auc_score,
                        clf_report_scorer,
                        confusion_x_scorer, ]
    all_scoring = dict(zip(all_scorer_names,
                           [make_scorer(x) for x in all_scorer_funcs]))
    scoring_for_cv = dict(zip(all_scorer_names[:-2],
                              [make_scorer(x) for x in all_scorer_funcs[:-2]]))
    '''
    scoring_inner = dict(zip(all_scorer_names[:-2],
                             [make_scorer(x) for x in all_scorer_funcs[:-2]]))
    scoring_outer = dict(zip(all_scorer_names,
                             [make_scorer(x) for x in all_scorer_funcs]))
    '''

    # FEATURE REPRESENTATION & CLASSIFICATION PIPELINE #
    # Initialise classification pipeline
    pipeline = Pipeline(steps=[('tfidf',
                                TfidfVectorizer(min_df=2,
                                                # ngram_range=(1, 3),
                                                # max_features=None,
                                                )),
                               ('classifier',
                                DummyClassifier(),)])

    # RUN & EVALUATE CLASSIFICATION EXPERIMENTS #
    # Splits into k cross-validation folds
    inner_cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    outer_cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    best_estimators = {}
    all_results = []
    i = 0

    for classifier, params in grid_search_params.items():
        print('\n\n{}\n{}'.format(classifier.upper(),
                                  params, ))

        # NON-NESTED HYPERPARAMETER GRIDSEARCH AND CROSS VALIDATION #
        # Fitting gridsearch computes k-fold cross validation on the whole X_train, Y_train
        # Returns an optimised model (.best_estimator_) which has been refit on the entire training set
        gs_clf = GridSearchCV(pipeline,
                              param_grid=params,
                              cv=inner_cv,
                              refit=refit_score,
                              return_train_score=False,
                              scoring=scoring_for_cv,
                              n_jobs=-4, )
        gs_clf.fit(X_train, Y_train)
        best_params = gs_clf.best_params_
        best_idx = gs_clf.best_index_
        best_estimator = gs_clf.best_estimator_
        print('\nBest classifier from flat gridsearch CV: \n{}'.format(gs_clf.best_estimator_))

        non_nested_results = {
            key: value for (key, value) in gs_clf.cv_results_.items() if
            key.startswith(('mean_t',
                            'std_t',
                            'rank_t',))
        }
        non_nested_results['best_score'] = gs_clf.best_score_
        print('\nNON-NESTED CV RESULTS:')
        print('Mean {} for optimised model: {:.3f} (+/-{:.03f})'.format(refit_score,
                                                                        non_nested_results[
                                                                            'mean_test_' + all_scorer_names[0]][
                                                                            best_idx],
                                                                        non_nested_results[
                                                                            'std_test_' + all_scorer_names[0]][
                                                                            best_idx] / 2,
                                                                        ))
        print('Mean {} for optimised model: {:.3f} (+/-{:.03f})'.format(all_scorer_names[1],
                                                                        non_nested_results[
                                                                            'mean_test_' + all_scorer_names[1]][
                                                                            best_idx],
                                                                        non_nested_results[
                                                                            'std_test_' + all_scorer_names[1]][
                                                                            best_idx] / 2,
                                                                        ))
        print('Mean {} for optimised model: {:.3f} (+/-{:.03f})'.format(all_scorer_names[2],
                                                                        non_nested_results[
                                                                            'mean_test_' + all_scorer_names[2]][
                                                                            best_idx],
                                                                        non_nested_results[
                                                                            'std_test_' + all_scorer_names[2]][
                                                                            best_idx] / 2,
                                                                        ))

        key = (classifier, feature_set)

        if args.nested:
            file_sfx = 'nested'

            # NESTED CROSS VALIDATION #
            # Passes gridsearch function (INNER LOOP) to cross_validate (OUTER LOOP)
            # Instead of computing k-fold cross validation, computes k-fold nested cross validation
            # For each of the k-folds in outer_cv, passes uses the k-1 training folds (outer_train) to the inner loop
            # outer_train is then split into inner_train and inner_validate, and used to perform k-fold CV and gridsearch
            # Optimised model is then tested on outer_test
            # This process is repeated for all k outer folds
            print('\nNESTED CV RESULTS:')
            nested_results = cross_validate(gs_clf,
                                            X=X_train,
                                            y=Y_train,
                                            cv=outer_cv,
                                            scoring=scoring_for_cv,
                                            return_train_score=False,
                                            return_estimator=False, )
            nested_results = {key: value for (key, value) in nested_results.items() if
                              key.startswith('test_')}

            print('Mean {} for algorithm: {:.3f} (+/-{:.03f})'.format(refit_score,
                                                                      nested_results[
                                                                          'test_' + all_scorer_names[0]].mean(),
                                                                      nested_results[
                                                                          'test_' + all_scorer_names[0]].std() / 2,
                                                                      ))
            print('Mean {} for algorithm: {:.3f} (+/-{:.03f})'.format(all_scorer_names[1],
                                                                      nested_results[
                                                                          'test_' + all_scorer_names[1]].mean(),
                                                                      nested_results[
                                                                          'test_' + all_scorer_names[1]].std() / 2,
                                                                      ))
            print('Mean {} for algorithm: {:.3f} (+/-{:.03f})'.format(all_scorer_names[2],
                                                                      nested_results[
                                                                          'test_' + all_scorer_names[2]].mean(),
                                                                      nested_results[
                                                                          'test_' + all_scorer_names[2]].std() / 2,
                                                                      ))

            # CROSS-VALIDATION RESULTS FOR FINAL TABLE #
            # Selects Nested CV results to be included in thesis results table
            cv_4_table = {}
            for i in range(0, 3):
                mean = nested_results['test_' + all_scorer_names[i]].mean()
                std = nested_results['test_' + all_scorer_names[i]].std() / 2
                cv_4_table[all_scorer_names[i]] = '{:.3f}'.format(mean)
            cv_4_table = pd.DataFrame(cv_4_table,
                                      index=pd.MultiIndex.from_tuples([key]), )
            cv_4_table.columns = pd.MultiIndex.from_product([['5_fold_cv'], cv_4_table.columns])

        else:
            file_sfx = 'non-nested'
            nested_results = None

            # CROSS-VALIDATION RESULTS FOR FINAL TABLE #
            # Selects flat CV results to be included in thesis results table
            cv_4_table = {}
            for i in range(0, 3):
                mean = non_nested_results['mean_test_' + all_scorer_names[i]][best_idx]
                std = non_nested_results['std_test_' + all_scorer_names[i]][best_idx] / 2
                cv_4_table[all_scorer_names[i]] = '{:.3f}'.format(mean)
            cv_4_table = pd.DataFrame(cv_4_table,
                                      index=pd.MultiIndex.from_tuples([key]), )
            cv_4_table.columns = pd.MultiIndex.from_product([['5_fold_cv'], cv_4_table.columns])

        # EVALUATION OF TRAINED ESTIMATOR ON HOLD-OUT TEST SET #
        # Predicts classes in hold-out test set (not included in dataset used for cross-validation)
        print('\nApplying best estimator to hold-out test set')
        Y_true, Y_predict = Y_test, best_estimator.predict(X_test)
        all_hold_out_results = {}
        for i, name in enumerate(all_scorer_names):
            scorer = all_scorer_funcs[i]
            results = round(scorer(Y_true,
                                   Y_predict, ), 3)
            if i < len(all_scorer_names) - 1:
                all_hold_out_results[name] = results

        # HOLD-OUT TEST RESULTS FOR FINAL TABLE #
        # Selects hold-out results to be included in thesis results table
        hold_out_4_table = pd.DataFrame(all_hold_out_results,
                                        index=pd.MultiIndex.from_tuples([key])).iloc[:, 0:3]
        hold_out_4_table.columns = pd.MultiIndex.from_product([['hold_out'], hold_out_4_table.columns])

        # Combines CV and hold-out results into single table
        thesis_table.append(pd.concat([cv_4_table, hold_out_4_table],
                                      axis=1, ))

        # Appends all trained estimators to list for later use
        best_estimators[classifier] = best_estimator
        # Appends all results to list for saving to file
        all_results[classifier] = {
            'inner_cv_results': non_nested_results,
            'grid_search_best_params': gs_clf.best_params_,
            'grid_search_best_score': gs_clf.best_score_,
            'outer_cv_results': nested_results,
            'hold_out_results': all_hold_out_results,
            'y_predict': Y_predict,
        }

        # Update index for colours
        i += 1

    # Saves final table to file
    full_thesis_table = pd.concat(thesis_table)
    full_thesis_table.sort_index(level=0,
                                 inplace=True, )
    full_thesis_table.to_pickle('thesis_results_table_{}.pkl'.format(file_sfx))
    print('All 5-fold cross validation results: {}'.format(full_thesis_table))

    # Saves trained estimators and cross-validation results to file
    pickle.dump(all_results,
                open('classification_results_{}_{}.pkl'.format(feature_set, file_sfx), 'wb'), )
    pickle.dump(best_estimators,
                open('trained_estimators_{}_{}.pkl'.format(feature_set, file_sfx), 'wb'), )

    print('\n\nClassification of threads with health claims complete.')
