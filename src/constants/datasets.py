from src.runners.iris_runner import cluster as iris
from src.runners.twentynewsgroups_runner import cluster as twentynewsgroups
from src.runners.waveform_runner import cluster as waveform

available_datasets = {'iris': {'runner': iris, 'name': 'iris'},
                     'twentynewsgroups': {'runner': twentynewsgroups, 'name': 'twentynewsgroups'},
                     'waveform': {'runner': waveform, 'name': 'waveform'}}