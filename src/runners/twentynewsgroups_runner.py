from src.seededkm.seededkm import SeededKMeans
from src.constrainedkm.constrainedkm import ConstrainedKMeans
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from src.utils.runnerutils import run_algo, run_KMeans
from scipy.sparse import csr_matrix

def cluster(n_clusters, seed_fraction, noise_fraction, incompleteness_fraction, manually_annotate, n_fold, run_KM):
    print('Fetching <partial> 20NewsGroups dataset')
    newsgroups_train = fetch_20newsgroups(subset='train', categories=['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space'])
    print('Fetched <partial> 20NewsGroups dataset, clustering')

    vectorizer = TfidfVectorizer()
    sparse_X = vectorizer.fit_transform(newsgroups_train.data)
    X = csr_matrix.todense(sparse_X)
    y = newsgroups_train.target

    skm = SeededKMeans(seed_fraction, noise_fraction, incompleteness_fraction, n_clusters, 'twentynewsgroups')
    seeded_ari, seeded_ami = run_algo(skm, 'Seeded K-Means', X, y, n_fold, manually_annotate)
    skm.visualise_results()

    ckm = ConstrainedKMeans(seed_fraction, noise_fraction, incompleteness_fraction, n_clusters, 'twentynewsgroups')
    constrained_ari, constrained_ami = run_algo(ckm, 'Constrained K-Means', X, y, n_fold, manually_annotate)
    ckm.visualise_results()

    kmeans_ari = 0
    kmeans_ami = 0
    if run_KM:
        kmeans_ari, kmeans_ami = run_KMeans(X, y, n_clusters, n_fold)

    return seeded_ari, seeded_ami, constrained_ari, constrained_ami, kmeans_ari, kmeans_ami
