# reddit-health-claims
This repository contains all code written for the Masters of Research thesis *"Using machine learning methods to detect health claims made in online forums"*

## Repository contents
- *config.py.example :* Example structure of the config.py file used for this project.
- *database_functions.py :* Functions used to connect to the secure database in which all data used for this project is stored (**NOTE:** All data used for this project is publicly available Reddit data, stored on a local database to facilitate analyses).
- *reddit_file_filter.py :* Functions to extract subreddit, submission and comment data from snapshot files containing dumps of public Reddit data generated on a monthly basis.

**subreddit_clustering**
- *subreddit_clustering_params.py :* Contains all parameters required for the clustering of subreddits using *k*-means and Latent Dirichlet Allocation (LDA).
- *generate_subreddit_corpus.py :* Compiles and processes the corpus for subreddit clustering
- *compile_seeding_set.py :* Generates set of seeding subreddits known to discuss health and medical topics, based on the subreddits recommended in the sidebar of r/Health (a subreddit for the sharing of health and medical news).
- *run_tfidf.py :* Generates different feature vocabularies and generates document-term matrices using term frequency inverse document frequency (TF-IDF).
- *run_tsvd.py :* Performs dimensionality reduction using truncated singular value decomposition (TSVD) in preparation for the clustering of subreddits via *k*-means.
- *run_lda.py :* Performs subreddit clustering using LDA.
- *run_kmeans.py :* Performs subreddit clustering using *k*-means.
- *evaluate_subreddit_clustering.py :* Evaluates the results of subreddit clustering using metrics including average recall and the total cluster size.
- *subreddit_clustering_results.py :* Compares the results of different models and methods for the clustering of health-related subreddits.

**health_claims_classification**
- *generate_thread_dataset.py :* Processes the corpus for classification of threads based on the presence or absence of heath claims.
- *health_claims_classifiers.py :* Trains and evaluates a variety of classifiers for the detection of Reddit threads containing health claims
- *evaluate_classification_results.py:* Analyses and visualises the performance of the trained classifiers.
