import numpy as np
from scipy.spatial import distance
from operator import itemgetter
import pandas as pd
import heapq

def compute_initial_centroids_from_seed_set(init_seed_set, n_clusters, X):
    centroids = np.zeros((n_clusters, init_seed_set.shape[1] - 1))
    cluster_with_no_centroids_assigned = []
    for cluster_no in range(n_clusters):
        cluster_members = np.array([seed[:-1] for seed in init_seed_set if seed[-1] == cluster_no])
        if len(cluster_members) != 0:
            centroids[cluster_no] = [np.sum(cluster_members[:, feature_no]) / cluster_members.shape[0] for feature_no in range(cluster_members.shape[1])]
        else:
            cluster_with_no_centroids_assigned.append(cluster_no)

    # If we have incomplete seeding, need to select remaining initial centroids
    if len(cluster_with_no_centroids_assigned) > 0:
        centroid_distance = [np.sum([distance.euclidean(x, centroid) for centroid in centroids]) for x in X]
        for ind, i in enumerate(cluster_with_no_centroids_assigned):
            centroids[i] = X[centroid_distance.index(heapq.nlargest(14, centroid_distance)[-1])]

    return centroids

def recompute_centroids(X, old_centroids, orig_seed_labels = [], orig_seed_indices = [], enable_seed_cluster_change = True):
    new_centroids = np.zeros((old_centroids.shape))
    if enable_seed_cluster_change:
        new_labels = [np.argmin([distance.euclidean(x, centroid) for centroid in old_centroids]) for x in X]
    else:
        new_labels = [np.argmin([distance.euclidean(x, centroid) for centroid in old_centroids]) if ind not in orig_seed_indices else int(orig_seed_labels[orig_seed_indices.index(ind)]) for ind, x in enumerate(X)]
    for cluster_no in range(old_centroids.shape[0]):
        cluster_members = np.array(itemgetter(list(np.where(np.array(new_labels) == cluster_no)[0]))(X))
        if cluster_members.shape[0] == 0:
            new_centroids[cluster_no] = old_centroids[cluster_no]
        else:
            new_centroids[cluster_no] = [np.sum(cluster_members[:, feature_no])/cluster_members.shape[0] for feature_no in range(X.shape[1])]
    return new_labels, new_centroids

def is_same_clustering(old_labels, labels):
    return (old_labels == labels)

def get_init_centroids_for_KMeans(X, y = np.array([]), n_clusters = 0):
    centroids = pd.DataFrame(X).sample(n_clusters).values
    return centroids
