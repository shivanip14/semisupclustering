from src.seededkm.seededkm import SeededKMeans
from src.constrainedkm.constrainedkm import ConstrainedKMeans
from sklearn import datasets
from src.utils.runnerutils import run_algo, run_KMeans

def cluster(n_clusters, seed_fraction, noise_fraction, completeness_fraction, manually_annotate, n_fold, run_KM):
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    skm = SeededKMeans(seed_fraction, noise_fraction, completeness_fraction, n_clusters, 'iris')
    seeded_ari, seeded_ami = run_algo(skm, X, y, n_fold, manually_annotate)
    skm.visualise_results()

    ckm = ConstrainedKMeans(seed_fraction, noise_fraction, completeness_fraction, n_clusters, 'iris')
    constrained_ari, constrained_ami = run_algo(ckm, X, y, n_fold, manually_annotate)
    ckm.visualise_results()

    kmeans_ari = 0
    kmeans_ami = 0
    if run_KM:
        kmeans_ari, kmeans_ami = run_KMeans(X, y, n_clusters, n_fold, seed_fraction, noise_fraction)

    return seeded_ari, seeded_ami, constrained_ari, constrained_ami, kmeans_ari, kmeans_ami
