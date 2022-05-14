from src.constants.datasets import available_datasets
from src.perftrackers.ami_vs_seedfraction import plot_ami_vs_seedfraction_for_runner
from src.perftrackers.ami_vs_noisefraction import plot_ami_vs_noisefraction_for_runner
import numpy as np
from src.runners.iris_runner import cluster

n_fold = 10
n_clusters = 3
manually_annotate = False
dataset = available_datasets.get('iris')

np.random.seed(0)

#plot_ami_vs_seedfraction_for_runner(n_fold, n_clusters, manually_annotate, dataset, True)
plot_ami_vs_noisefraction_for_runner(n_fold, n_clusters, manually_annotate, dataset, True)
