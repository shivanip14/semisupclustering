import numpy as np
import matplotlib.pyplot as plt

def plot_ami_vs_seedfraction_for_runner(n_fold, n_clusters, manually_annotate, dataset, run_KM, run_COPKM, run_agglo, run_gaussianmm):
    completeness_fraction = np.arange(0.1, 1, 0.05)
    seed_fractions = 0.4
    noise_fraction = 0
    perf_metrics = np.zeros((seed_fractions.shape[0], 4))
    for idx, seed_fraction in enumerate(seed_fractions):
        print('\nClustering {} for SF = {:.2f}'.format(str(dataset['name']), seed_fraction))
        perf_metrics[idx] = dataset['runner'](n_clusters, seed_fraction, noise_fraction, manually_annotate, n_fold)

    fig = plt.figure()
    subplot_ami = fig.add_subplot(121)
    subplot_ari = fig.add_subplot(122)
    fig.suptitle('Clustering quality over Seed Fraction (NF = {:.2f})'.format(noise_fraction))

    subplot_ami.plot(seed_fractions, perf_metrics[:, 1], c='#fc4103', marker='s', markersize=4, label='Seeded KM')
    subplot_ami.plot(seed_fractions, perf_metrics[:, 3], c='#03a1fc', marker='o', markersize=4, label='Constrained KM')
    subplot_ami.set_title('AMI vs Seed Fraction')
    subplot_ami.set_xlabel('seed_fraction')
    subplot_ami.set_ylabel('ami')
    subplot_ami.grid()
    subplot_ami.legend()

    subplot_ari.plot(seed_fractions, perf_metrics[:, 0], c='#fc4103', marker='s', markersize=4, label='Seeded KM')
    subplot_ari.plot(seed_fractions, perf_metrics[:, 2], c='#03a1fc', marker='o', markersize=4, label='Constrained KM')
    subplot_ari.set_title('ARI vs Seed Fraction')
    subplot_ari.set_xlabel('seed_fraction')
    subplot_ari.set_ylabel('ari')
    subplot_ari.grid()
    subplot_ari.legend()

    plt.savefig('../results/' + 'AMI_ARI_vs_SeedingCompleteness_' + str(dataset['name']) + '.png')
    plt.show()
    plt.clf()
    plt.cla()
    plt.close()
