from src.utils.seedutils import get_init_seed_set, get_init_centroids_for_KMeans
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn import metrics

def run_algo(algorithm, X, y, n_fold, manually_annotate):
    for iter in range(n_fold):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=iter) #TODO average it out to smoothen the curve
        init_seed_set = get_init_seed_set(X=X_train, y=y_train, seed_fraction=algorithm.seed_fraction, noise_fraction=algorithm.noise_fraction, n_clusters=algorithm.n_clusters, manually_annotate=manually_annotate)
        algorithm.fit(X_train, y_train, init_seed_set)
        algorithm.predict(X_test, y_test)
        ari, ami = algorithm.evaluate_fold(n_fold, iter)
        if ari >= 0 and ami >= 0:
            print('{} avg over {} folds; SF = [{:.2f}], NF = [{:.2f}]: ARI = {}, AMI = {}'.format(algorithm, n_fold, algorithm.seed_fraction, algorithm.noise_fraction, ari, ami))
    return ari, ami

def run_KMeans(X, y, n_clusters, n_fold, seed_fraction, noise_fraction):
    ari_intermediate = 0
    ami_intermediate = 0
    for iter in range(n_fold):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=iter)
        init_centroids = get_init_centroids_for_KMeans(X=X_train, y=y_train, seed_fraction=seed_fraction, noise_fraction=noise_fraction, n_clusters=n_clusters)
        kmeans = KMeans(n_clusters=n_clusters, algorithm='full', init=init_centroids, max_iter=100)
        kmeans = kmeans.fit(X_train, y_train)
        predicted_labels = kmeans.predict(X_test, y_test)
        ari_intermediate += metrics.adjusted_rand_score(y_test, predicted_labels)
        ami_intermediate += metrics.adjusted_mutual_info_score(y_test, predicted_labels)
    ari = ari_intermediate/n_fold
    ami = ami_intermediate/n_fold
    print('KMeans avg over {} folds: ARI = {}, AMI = {}'.format(n_fold, ari, ami))
    return ari, ami