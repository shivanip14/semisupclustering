import numpy as np
import warnings
import pandas as pd
import math

def get_init_seed_set(X, y = np.array([]), n_seeds = 0, n_clusters = 0, manually_annotate = False):
    if n_clusters <= 0:
        raise ValueError('No. of clusters should be a positive integer')
    if n_seeds <= n_clusters:
        raise ValueError('No. of seeds should be >= no. of clusters')
    if not y.size and not manually_annotate:
        raise ValueError('manually_annotate should be set to True if true labels are not available')
    if y.size and manually_annotate:
        warnings.warn('True labels provided, manually_annotate flag is overridden to take actual label values')


    # Taking seeds from the available true labels. Assuming at least one seed per class is chosen
    if y.size:
        init_seed_set = _sample_n_seeds(n_seeds, n_clusters, pd.DataFrame(np.c_[X, y]), False)
    # Since true labels are not available, user needs to manually annotate the randomly chosen seeds until **at least one seed per class** condition is satisfied
    else:
        # TODO
        print('Manually annotate the seeds selected below with the cluster no. (0 - {}):\n'.format(n_clusters - 1))
        init_seed_set = _sample_n_seeds(n_seeds, n_clusters, pd.DataFrame(X), True)
        cluster = np.zeros((init_seed_set.shape[0], 1))
        for index, seed in enumerate(init_seed_set):
            cluster[index] = input(str(seed) + ': ')
        init_seed_set = np.c_[init_seed_set, cluster]
    return init_seed_set

def _sample_n_seeds(n_seeds, n_clusters, all_data_df, manually_annotate):
    if manually_annotate:
        init_seed_set = all_data_df.sample(n_seeds)
    else:
        init_seed_set = all_data_df.groupby(all_data_df.shape[1] - 1).apply(lambda x: x.sample(n=math.floor(n_seeds / n_clusters))).reset_index(drop=True)
        if n_clusters % n_seeds != 0:
            init_seed_set = init_seed_set.append(all_data_df.sample(n_seeds % n_clusters))  # TODO avoid repeated sampling
    return init_seed_set.values