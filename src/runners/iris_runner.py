from src.seededkm.seededkm import SeededKMeans
from src.constrainedkm.constrainedkm import ConstrainedKMeans
from src.utils.seedutils import get_init_seed_set
from sklearn import datasets
from sklearn.model_selection import train_test_split

def cluster(n_clusters, seed_fraction, manually_annotate, n_fold):
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    skm = SeededKMeans(seed_fraction, n_clusters)
    for iter in range(n_fold):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)
        init_seed_set = get_init_seed_set(X=X_train, y=y_train, seed_fraction=seed_fraction, n_clusters=n_clusters, manually_annotate=manually_annotate)
        skm.fit(X_train, y_train, init_seed_set)
        skm.predict(X_test, y_test)
        # skm.visualise_results()
        # skm.evaluate_train()
        # skm.evaluate_test()
        ari, ami = skm.evaluate_fold(n_fold, iter)
        if ari >= 0 and ami >= 0:
            print('Seeded K-Means avg over {} folds: ARI = {}, AMI = {}'.format(n_fold, ari, ami))

    ckm = ConstrainedKMeans(seed_fraction, n_clusters)
    for iter in range(n_fold):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
        init_seed_set = get_init_seed_set(X=X_train, y=y_train, seed_fraction=seed_fraction, n_clusters=n_clusters, manually_annotate=manually_annotate)
        ckm.fit(X_train, y_train, init_seed_set)
        ckm.predict(X_test, y_test)
        # ckm.visualise_results()
        # ckm.evaluate_train()
        # ckm.evaluate_test()
        ari, ami = ckm.evaluate_fold(n_fold, iter)
        if ari >= 0 and ami >= 0:
            print('Constrained K-Means avg over {} folds: ARI = {}, AMI = {}'.format(n_fold, ari, ami))
