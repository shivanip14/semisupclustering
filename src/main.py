from src.runners.iris_runner import cluster as iris
from src.runners.twentynewsgroups_runner import cluster as twentynewsgroups

available_dataset_runners = [iris, twentynewsgroups]

n_fold = 10
n_clusters = 3
seed_fraction = 0.1
manually_annotate = False
runner = iris

if runner not in available_dataset_runners:
    raise ValueError('Select dataset from a list of available ones!')
else:
    runner(n_clusters, seed_fraction, manually_annotate, n_fold)
