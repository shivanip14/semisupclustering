from src.constants.datasets import available_datasets
from src.perftrackers.ami_vs_seedfraction import plot_ami_vs_seedfraction_for_runner
from src.perftrackers.ami_vs_noisefraction import plot_ami_vs_noisefraction_for_runner
from src.perftrackers.ami_vs_seedingcompleteness import plot_ami_vs_completenessfraction_for_runner
import numpy as np

n_fold = 10
manually_annotate = False

# For iris
n_clusters = 3
dataset = available_datasets.get('iris')

# For waveform
# n_clusters = 3
# dataset = available_datasets.get('waveform')

# For (partial) 20newsgroups
# n_clusters = 4
# dataset = available_datasets.get('twentynewsgroups')

np.random.seed(0)

plot_ami_vs_seedfraction_for_runner(n_fold, n_clusters, manually_annotate, dataset, True)
plot_ami_vs_noisefraction_for_runner(n_fold, n_clusters, manually_annotate, dataset, True)
plot_ami_vs_completenessfraction_for_runner(n_fold, n_clusters, manually_annotate, dataset, True)
