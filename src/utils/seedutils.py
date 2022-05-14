import numpy as np
import warnings
import pandas as pd
import math
import random
from src.utils.centroidutils import compute_initial_centroids_from_seed_set

def get_init_seed_set(X, y = np.array([]), seed_fraction = 0.1, noise_fraction = 0, completeness_fraction = 1, n_clusters = 0, manually_annotate = False):
    if n_clusters <= 0:
        raise ValueError('No. of clusters should be a positive integer')
    n_seeds = int(seed_fraction * X.shape[0])
    if n_seeds <= n_clusters:
        raise ValueError('No. of seeds should be >= no. of clusters')
    if n_seeds > X.shape[0]:
        raise ValueError('No. of seeds cannot be greater than the population')
    if noise_fraction < 0 or noise_fraction > 1:
        raise ValueError('Noise fraction should be [0, 1]')
    if completeness_fraction < 0 or completeness_fraction > 1:
        raise ValueError('Completeness fraction should be [0, 1]')
    if not y.size and not manually_annotate:
        raise ValueError('manually_annotate should be set to True if true labels are not available')
    if y.size and manually_annotate:
        warnings.warn('True labels provided, manually_annotate flag is overridden to take actual label values')
    if completeness_fraction == 0:
        warnings.warn('completeness_fraction set as 0, seeds will be selected randomly')

    # TODO
    # unseeded_categories = math.floor(completeness_fraction * n_clusters)
    # print('Initial seed set will not have seeds from {} classes'.format(unseeded_categories))


    # Taking seeds from the available true labels. Assuming at least one seed per class is chosen
    if y.size:
        init_seed_set = _sample_n_seeds(n_seeds, n_clusters, pd.DataFrame(np.c_[X, y]), False)
    # Since true labels are not available, user needs to manually annotate the randomly chosen seeds until **at least one seed per class** condition is satisfied
    else:
        print('Manually annotate the seeds selected below with the cluster no. (0 - {}):\n'.format(n_clusters - 1))
        init_seed_set = _sample_n_seeds(n_seeds, n_clusters, pd.DataFrame(X), True)
        cluster = np.zeros((init_seed_set.shape[0], 1))
        for index, seed in enumerate(init_seed_set):
            cluster[index] = input(str(seed) + ': ')
        init_seed_set = np.c_[init_seed_set, cluster]

    # Now that we have init_seed_set with all true labels, we randomly pick noise_fraction*len(init_seed_set) seeds and change their labels to incorrect ones
    if noise_fraction > 0:
        possible_labels = tuple(set(init_seed_set[:, -1]))
        noisy_seeds = random.sample(range(0, init_seed_set.shape[0]), math.floor(noise_fraction * init_seed_set.shape[0]))
        for idx in noisy_seeds:
            # TODO how to introduce label noise so that init_seed_set still has atleast one seed per class?
            # Method 1
            #init_seed_set[idx, -1] = possible_labels[(possible_labels.index(init_seed_set[idx, -1]) + 1) % len(possible_labels)]

            # Method 2
            true_label = init_seed_set[idx, -1]
            while init_seed_set[idx, -1] == true_label:
                init_seed_set[idx, -1] = random.choice(possible_labels)
    return init_seed_set

def _sample_n_seeds(n_seeds, n_clusters, all_data_df, manually_annotate):
    if manually_annotate:
        init_seed_set = all_data_df.sample(n_seeds)
    else:
        init_seed_set = all_data_df.groupby(all_data_df.shape[1] - 1).apply(lambda x: x.sample(n=math.floor(n_seeds / n_clusters), replace = True)).reset_index(drop=True)
        if n_clusters % n_seeds != 0:
            init_seed_set = init_seed_set.append(all_data_df.sample(n_seeds % n_clusters))  # TODO avoid repeated sampling
    return init_seed_set.values

def get_init_centroids_for_KMeans(X, y = np.array([]), seed_fraction = 0.1, noise_fraction = 0, n_clusters = 0):
    init_seed_set = get_init_seed_set(X, y, seed_fraction, noise_fraction, 1, n_clusters, False)
    centroids = compute_initial_centroids_from_seed_set(init_seed_set, n_clusters)
    return centroids