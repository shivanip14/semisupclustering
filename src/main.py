from src.seededkm.seededkm import SeededKMeans
from src.constrainedkm.constrainedkm import ConstrainedKMeans
from src.utils.seedutils import get_init_seed_set
from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
X = iris.data
y = iris.target
n_clusters = 3
n_seeds = 5
manually_annotate = True
init_seed_set = get_init_seed_set(X = X, y = y, n_seeds = n_seeds, n_clusters = n_clusters, manually_annotate = manually_annotate)

skm = SeededKMeans(n_clusters)
skm.fit(X, y, init_seed_set)
skm.visualise_results()

ckm = ConstrainedKMeans(n_clusters)
ckm.fit(X, y, init_seed_set)
ckm.visualise_results()
