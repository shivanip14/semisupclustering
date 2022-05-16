from src.utils.seedutils import get_init_seed_set
from src.utils.centroidutils import get_init_centroids_for_KMeans
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn import metrics

META_COUNT = 1 # TODO increase this for final commit

def run_algo(algorithm, X, y, n_fold, manually_annotate):
    ari_intermediate = 0
    ami_intermediate = 0
    for iter in range(n_fold):
        ari_intermediate_c = 0
        ami_intermediate_c = 0
        for count in range(META_COUNT):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=iter)  # TODO average it out to smoothen the curve
            init_seed_set = get_init_seed_set(X=X_train, y=y_train, seed_fraction=algorithm.seed_fraction, noise_fraction=algorithm.noise_fraction, incompleteness_fraction=algorithm.incompleteness_fraction, n_clusters=algorithm.n_clusters, manually_annotate=manually_annotate)
            algorithm.fit(X_train, y_train, init_seed_set)
            algorithm.predict(X_test, y_test)
            ari_c, ami_c = algorithm.evaluate_scores()
            ari_intermediate_c += ari_c
            ami_intermediate_c += ami_c
        ari_intermediate += ari_intermediate_c / META_COUNT
        ami_intermediate += ami_intermediate_c / META_COUNT
    ari = ari_intermediate / n_fold
    ami = ami_intermediate / n_fold
    print('{} avg over {} folds; SF = [{:.2f}], NF = [{:.2f}]: ARI = {}, AMI = {}'.format(algorithm, n_fold, algorithm.seed_fraction, algorithm.noise_fraction, ari, ami))
    return ari, ami

def run_KMeans(X, y, n_clusters, n_fold):
    ari_intermediate = 0
    ami_intermediate = 0
    for iter in range(n_fold):
        ari_intermediate_c = 0
        ami_intermediate_c = 0
        for count in range(META_COUNT):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=iter)
            init_centroids = get_init_centroids_for_KMeans(X=X_train, y=y_train, n_clusters=n_clusters)
            kmeans = KMeans(n_clusters=n_clusters, algorithm='full', init=init_centroids, max_iter=100)
            kmeans = kmeans.fit(X_train, y_train)
            predicted_labels = kmeans.predict(X_test, y_test)
            ari_intermediate_c += metrics.adjusted_rand_score(y_test, predicted_labels)
            ami_intermediate_c += metrics.adjusted_mutual_info_score(y_test, predicted_labels)
        ari_intermediate += ari_intermediate_c / META_COUNT
        ami_intermediate += ami_intermediate_c / META_COUNT
    ari = ari_intermediate/n_fold
    ami = ami_intermediate/n_fold
    print('KMeans avg over {} folds: ARI = {}, AMI = {}'.format(n_fold, ari, ami))
    return ari, ami