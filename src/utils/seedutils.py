import numpy as np
import warnings
import pandas as pd
import math
import random

def get_init_seed_set(X, y = np.array([]), seed_fraction = 0.1, noise_fraction = 0, incompleteness_fraction = 0, n_clusters = 0, manually_annotate = False):
    if n_clusters <= 0:
        raise ValueError('No. of clusters should be a positive integer')
    n_seeds = int(seed_fraction * X.shape[0])
    if n_seeds <= n_clusters:
        raise ValueError('No. of seeds should be >= no. of clusters')
    if n_seeds > X.shape[0]:
        raise ValueError('No. of seeds cannot be greater than the population')
    if noise_fraction < 0 or noise_fraction > 1:
        raise ValueError('Noise fraction should be [0, 1]')
    if incompleteness_fraction < 0 or incompleteness_fraction > 1:
        raise ValueError('Incompleteness fraction should be [0, 1]')
    if not y.size and not manually_annotate:
        raise ValueError('manually_annotate should be set to True if true labels are not available')
    if y.size and manually_annotate:
        warnings.warn('True labels provided, manually_annotate flag is overridden to take actual label values')
    if incompleteness_fraction == 1:
        warnings.warn('incompleteness_fraction set as 1, seeds will be selected randomly')

    # No. of clusters which will NOT be represented in the initial seed set will be calculated as follows:
    # For example, for a total of 10 clusters & incompleteness_fraction of 0.4, there'll be seeds from only (1-0.4)*10 = 6 clusters represented in the init_seed_set
    # For incompleteness_fraction 1, seeds will be selected randomly (== Random KMeans)
    seeded_categories = math.floor((1 - incompleteness_fraction) * n_clusters)
    if seeded_categories < n_clusters and seeded_categories > 0:
        print('Initial seed set will have seeds ONLY from {} classes of the total {}'.format(seeded_categories, n_clusters))
    elif seeded_categories == 0:
        print('Initial seed set will be selected randomly as incompleteness_fraction is set to 1')
    allowed_clusters = random.sample(range(0, n_clusters), seeded_categories)

    # Taking seeds from the available true labels. Assuming at least one seed per class is chosen
    if y.size:
        init_seed_set = _sample_n_seeds(n_seeds, n_clusters, allowed_clusters, pd.DataFrame(np.c_[X, y]), False)
    # Since true labels are not available, user needs to manually annotate the randomly chosen seeds until **at least one seed per class** condition is satisfied
    else:
        if incompleteness_fraction > 0 and incompleteness_fraction < 1:
            print('Manually annotate the seeds selected below with the cluster no. {}:\n'.format(allowed_clusters))
        else:
            print('Manually annotate the seeds selected below with the cluster no. (0 - {}):\n'.format(n_clusters - 1))
        init_seed_set = _sample_n_seeds(n_seeds, n_clusters, allowed_clusters, pd.DataFrame(X), True)
        cluster = np.zeros((init_seed_set.shape[0], 1))
        for index, seed in enumerate(init_seed_set):
            cluster[index] = input(str(seed) + ': ')
        init_seed_set = np.c_[init_seed_set, cluster]

    # Now that we have init_seed_set with all true labels, we randomly pick noise_fraction*len(init_seed_set) seeds and change their labels to incorrect ones
    if noise_fraction > 0:
        possible_labels = tuple(set(init_seed_set[:, -1]))
        # We don't need to exclude some clusters from here while introducing noise, for the "incomplete seeding" - as
        # the possible labels are being selected from the init_seed_set,
        # which won't have such cluster labels if incompleteness_fraction < 1
        noisy_seeds = random.sample(range(0, init_seed_set.shape[0]), math.floor(noise_fraction * init_seed_set.shape[0]))
        for idx in noisy_seeds:
            # Introducing label noise so that init_seed_set still has atleast one seed per class?
            # Method 1
            # init_seed_set[idx, -1] = possible_labels[(possible_labels.index(init_seed_set[idx, -1]) + 1) % len(possible_labels)]

            # Method 2
            true_label = init_seed_set[idx, -1]
            while init_seed_set[idx, -1] == true_label:
                init_seed_set[idx, -1] = random.choice(possible_labels)
    return init_seed_set

def _sample_n_seeds(n_seeds, n_clusters, allowed_clusters, all_data_df, manually_annotate):
    if manually_annotate:
        init_seed_set = all_data_df.sample(n_seeds)
    else:
        if len(allowed_clusters) > 0 and len(allowed_clusters) < n_clusters:
            all_data_df = all_data_df.loc[all_data_df[all_data_df.shape[1] - 1].isin(allowed_clusters)]
        if len(allowed_clusters) == 0:
            init_seed_set = all_data_df.sample(n_seeds)
        else:
            init_seed_set = all_data_df.groupby(all_data_df.shape[1] - 1).apply(lambda x: x.sample(n=math.floor(n_seeds / n_clusters), replace = True)).reset_index(drop=True)
            if n_clusters % n_seeds != 0:
                init_seed_set = init_seed_set.append(all_data_df.sample(n_seeds % n_clusters))
    return init_seed_set.values
