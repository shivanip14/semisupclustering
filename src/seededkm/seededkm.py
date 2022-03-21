from src.utils.centroidutils import compute_initial_centroids_from_seed_set, is_same_clustering, recompute_centroids
import numpy as np
from src.utils.visualisationutils import visualise_clusters
from sklearn import metrics

class SeededKMeans():
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
        self.seed_indices = [np.flatnonzero((self.X == seed).all(1))[0] for seed in self.seeds]
        for iter in range(self.max_iter):
            if iter == 0:
                self.centroids = compute_initial_centroids_from_seed_set(init_seed_set = init_seed_set, n_clusters = self.n_clusters)
            old_labels = self.labels
            self.labels, self.centroids = recompute_centroids(X = X, old_centroids = self.centroids, enable_seed_cluster_change = True)

            if is_same_clustering(old_labels = old_labels, labels = self.labels):
                print('Terminating after {} iterations as cluster membership did not change'.format(iter))
                break

    def visualise_results(self):
        # each of the seed's index in X -> labels[index]
        computed_seed_labels = np.array([self.labels[np.flatnonzero((self.X == seed).all(1))[0]] for seed in self.seeds])
        visualise_clusters('Seeded K-Means', self.X, self.y, self.labels, self.seeds, self.orig_seed_labels, computed_seed_labels)

    def predict(self, x_test):
        # TODO
        self.x_test = x_test

    def evaluate(self):
        print('\nSeeded K-Means evaluation:')
        print('Adjusted RAND Score: {}'.format(metrics.adjusted_rand_score(self.y, self.labels)))
        print('Adjusted Mutual Info Score: {}'.format(metrics.adjusted_mutual_info_score(self.y, self.labels)))

