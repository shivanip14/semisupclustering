from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from operator import itemgetter

def visualise_clusters(algo_name, X, y, computed_y, n_clusters, seeds, seed_fraction, orig_seed_labels, computed_seed_labels):
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X)
    # each of the seed's index in X -> pca_result[index]
    new_seeds = np.array([pca_result[np.flatnonzero((X == seed).all(1))[0]] for seed in seeds])

    pca_centroids_orig = np.zeros((n_clusters, 2))
    pca_centroids_computed = np.zeros((n_clusters, 2))
    for cluster_no in range(n_clusters):
        pca_cluster_members_orig = np.array(itemgetter(*list(np.where(np.array(y) == cluster_no)[0]))(pca_result))
        pca_cluster_members_computed = np.array(itemgetter(*list(np.where(np.array(computed_y) == cluster_no)[0]))(pca_result))
        pca_centroids_orig[cluster_no] = [np.sum(pca_cluster_members_orig[:, feature_no])/pca_cluster_members_orig.shape[0] for feature_no in range(2)]
        pca_centroids_computed[cluster_no] = [np.sum(pca_cluster_members_computed[:, feature_no]) / pca_cluster_members_computed.shape[0] for feature_no in range(2)]

    fig = plt.figure()
    fig.suptitle(algo_name + ' (seed_fraction = ' + str(seed_fraction) + ' - ' + str(seeds.shape[0]) + ' seeds)')
    subplot_orig = fig.add_subplot(121)
    subplot_computed = fig.add_subplot(122)

    subplot_orig.scatter(pca_result[:, 0], pca_result[:, 1], c = y, cmap = plt.cm.autumn)
    subplot_orig.scatter(new_seeds[:, 0], new_seeds[:, 1], c = orig_seed_labels, marker='x', cmap = plt.cm.winter)
    subplot_orig.scatter(pca_centroids_orig[:, 0], pca_centroids_orig[:, 1], c = 'black', marker = 's')
    subplot_orig.set_title('True clusters')
    subplot_orig.set_xlabel('PCA_1')
    subplot_orig.set_ylabel('PCA_2')

    subplot_computed.scatter(pca_result[:, 0], pca_result[:, 1], c = computed_y, cmap = plt.cm.winter)
    subplot_computed.scatter(new_seeds[:, 0], new_seeds[:, 1], c = computed_seed_labels, marker='x', cmap = plt.cm.autumn)
    subplot_computed.scatter(pca_centroids_computed[:, 0], pca_centroids_computed[:, 1], c = 'black', marker = 's')
    subplot_computed.set_title('Computed clusters')
    subplot_computed.set_xlabel('PCA_1')
    subplot_computed.set_ylabel('PCA_2')
    plt.savefig('../results/' + algo_name + ' clusters.png')
    plt.show()
