from src.seededkm.seededkm import SeededKMeans
from src.constrainedkm.constrainedkm import ConstrainedKMeans
from src.utils.runnerutils import run_algo
import pandas as pd
from scipy.io.arff import loadarff
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import normalize

def cluster(n_clusters, seed_fraction, noise_fraction, incompleteness_fraction, manually_annotate, n_fold):
    data, meta = loadarff('../data/waveform/waveform.arff')
    df = pd.DataFrame(data)

    y = df['class'].values

    # encoding the label
    le = LabelEncoder()
    y = le.fit_transform(y)

    # normalize the values
    df_num = df.drop('class', axis = 1)
    X = normalize(df_num.values)

    skm = SeededKMeans(seed_fraction, noise_fraction, incompleteness_fraction, n_clusters, 'waveform')
    seeded_ari, seeded_ami = run_algo(skm, X, y, n_fold, manually_annotate)
    skm.visualise_results()

    ckm = ConstrainedKMeans(seed_fraction, noise_fraction, incompleteness_fraction, n_clusters, 'waveform')
    constrained_ari, constrained_ami = run_algo(ckm, X, y, n_fold, manually_annotate)
    ckm.visualise_results()

    return seeded_ari, seeded_ami, constrained_ari, constrained_ami