from src.utils.centroidutils import compute_initial_centroids_from_seed_set, is_same_clustering, recompute_centroids
import numpy as np
from src.utils.visualisationutils import visualise_clusters

class ConstrainedKMeans():
    def __init__(self, n_clusters = 0, max_iter = 100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.X = np.array([])
        self.y = np.array([])
        self.labels = np.array([])
        self.orig_seed_labels = np.array([])
        self.centroids = np.array([])
        self._check_params()

    def _check_params(self):
        if self.n_clusters <= 0:
            raise ValueError('No. of clusters should be a positive integer')
        #TODO - introduce more checks

    def fit(self, X, y, init_seed_set):
        self.X = X
        self.y = y
        self.seeds = init_seed_set[:, :-1]
        self.labels = init_seed_set[:, -1]
        self.orig_seed_labels = self.labels
        print('Initial seed set:\n', init_seed_set)
        for iter in range(self.max_iter):
            print('Running iteration #', iter)
            if iter == 0:
                self.centroids = compute_initial_centroids_from_seed_set(init_seed_set = init_seed_set, n_clusters = self.n_clusters)
            old_labels = self.labels
            self.labels, self.centroids = recompute_centroids(X = X, old_centroids = self.centroids, orig_seeds = self.seeds, orig_seed_labels = self.orig_seed_labels, enable_seed_cluster_change = False)

            if is_same_clustering(old_labels = old_labels, labels = self.labels):
                print('Terminating as cluster membership did not change from previous iteration')
                break

    def visualise_results(self):
        computed_seed_labels = np.array([self.labels[np.flatnonzero((self.X == seed).all(1))[0]] for seed in self.seeds])
        visualise_clusters(self.X, self.y, self.labels, self.seeds, self.orig_seed_labels, computed_seed_labels)

    def predict(self, x_test):
        # TODO
        self.x_test = x_test

