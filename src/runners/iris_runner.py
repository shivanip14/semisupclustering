from src.seededkm.seededkm import SeededKMeans
from src.constrainedkm.constrainedkm import ConstrainedKMeans
from src.utils.seedutils import get_init_seed_set
from sklearn import datasets

def cluster(n_clusters, n_seeds, manually_annotate):
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    init_seed_set = get_init_seed_set(X = X, y = y, n_seeds = n_seeds, n_clusters = n_clusters, manually_annotate = manually_annotate)

    skm = SeededKMeans(n_clusters)
    skm.fit(X, y, init_seed_set)
    skm.visualise_results()
    skm.evaluate()

    ckm = ConstrainedKMeans(n_clusters)
    ckm.fit(X, y, init_seed_set)
    ckm.visualise_results()
    ckm.evaluate()
