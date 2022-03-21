from src.runners.iris_runner import cluster as iris
from src.runners.twentynewsgroups_runner import cluster as twentynewsgroups

available_dataset_runners = [iris, twentynewsgroups]

n_clusters = 3
n_seeds = 5
manually_annotate = False
runner = iris

if runner not in available_dataset_runners:
    raise ValueError('Select dataset from a list of available ones!')
else:
    runner(n_clusters, n_seeds, manually_annotate)
