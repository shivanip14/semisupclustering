from src.utils.centroidutils import compute_initial_centroids_from_seed_set, is_same_clustering, recompute_centroids
import numpy as np
from src.utils.visualisationutils import visualise_clusters
from sklearn import metrics
from scipy.spatial import distance

class SeededKMeans():
    def __init__(self, seed_fraction, noise_fraction, completeness_fraction, n_clusters, dataset_name, max_iter = 100):
        self.n_clusters = n_clusters
        self.dataset_name = dataset_name
        self.seed_fraction = seed_fraction
        self.noise_fraction = noise_fraction
        self.completeness_fraction = completeness_fraction
        self.max_iter = max_iter
        self.X = np.array([])
        self.y = np.array([])
        self.labels = np.array([])
        self.orig_seed_labels = np.array([])
        self.centroids = np.array([])
        self.ari = -1
        self.ami = -1
        self.ari_intermediate = 0
        self.ami_intermediate = 0
        self._check_params()

    def _check_params(self):
        if self.n_clusters <= 0:
            raise ValueError('No. of clusters should be a positive integer')
        if self.seed_fraction <= 0 or self.seed_fraction > 1:
            raise ValueError('seed_fraction should be (0, 1]')
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
        visualise_clusters('Seeded_KMeans', self.dataset_name, self.X, self.y, self.labels, self.n_clusters, self.seeds, self.seed_fraction, self.orig_seed_labels, computed_seed_labels)

    def predict(self, X_test, y_test):
        self.X_test = X_test
        self.y_test = y_test
        self.test_labels = [np.argmin([distance.euclidean(x, centroid) for centroid in self.centroids]) for x in X_test]

    def evaluate_train(self):
        print('\nSeeded K-Means training evaluation:')
        self.evaluate(self.y, self.labels)

    def evaluate_test(self):
        print('\nSeeded K-Means test evaluation:')
        self.evaluate(self.y_test, self.test_labels)

    def evaluate(self, true_labels, computed_labels):
        print('Adjusted RAND Score: {}'.format(metrics.adjusted_rand_score(true_labels, computed_labels)))
        print('Adjusted Mutual Info Score: {}'.format(metrics.adjusted_mutual_info_score(true_labels, computed_labels)))

    def evaluate_fold(self, n_fold, iter):
        if iter == 0:
            self.ari_intermediate = 0
            self.ami_intermediate = 0
        self.ari_intermediate += metrics.adjusted_rand_score(self.y_test, self.test_labels)
        self.ami_intermediate += metrics.adjusted_mutual_info_score(self.y_test, self.test_labels)
        self.ari = -1
        self.ami = -1
        if iter == n_fold - 1:
            self.ari = self.ari_intermediate / n_fold
            self.ami = self.ami_intermediate / n_fold

        return self.ari, self.ami
