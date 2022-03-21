from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

def visualise_clusters(algo_name, X, y, computed_y, seeds, orig_seed_labels, computed_seed_labels):
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X)
    # each of the seed's index in X -> pca_result[index]
    new_seeds = np.array([pca_result[np.flatnonzero((X == seed).all(1))[0]] for seed in seeds])

    fig = plt.figure()
    fig.suptitle(algo_name)
    subplot_orig = fig.add_subplot(121)
    subplot_computed = fig.add_subplot(122)

    subplot_orig.scatter(pca_result[:, 0], pca_result[:, 1], c = y, cmap = plt.cm.autumn)
    subplot_orig.scatter(new_seeds[:, 0], new_seeds[:, 1], c = orig_seed_labels, marker='x', cmap = plt.cm.winter)
    subplot_orig.set_title('True clusters')
    subplot_orig.set_xlabel('PCA_1')
    subplot_orig.set_ylabel('PCA_2')

    subplot_computed.scatter(pca_result[:, 0], pca_result[:, 1], c = computed_y, cmap = plt.cm.winter)
    subplot_computed.scatter(new_seeds[:, 0], new_seeds[:, 1], c = computed_seed_labels, marker='x', cmap = plt.cm.autumn)
    subplot_computed.set_title('Computed clusters')
    subplot_computed.set_xlabel('PCA_1')
    subplot_computed.set_ylabel('PCA_2')
    plt.show()
