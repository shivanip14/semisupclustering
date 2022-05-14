import numpy as np
import matplotlib.pyplot as plt

def plot_ami_vs_noisefraction_for_runner(n_fold, n_clusters, manually_annotate, dataset, run_KM = False, run_COPKM = False, run_agglo = False, run_gaussianmm = False):
    noise_fractions = np.arange(0, 1, 0.05)
    seed_fraction_1 = 0.3
    seed_fraction_2 = 0.7
    completeness_fraction = 1
    perf_metrics_1 = np.zeros((noise_fractions.shape[0], 6))
    perf_metrics_2 = np.zeros((noise_fractions.shape[0], 6))
    for idx, noise_fraction in enumerate(noise_fractions):
        print('\nClustering {} for NF = {:.2f}'.format(str(dataset['name']), noise_fraction))
        perf_metrics_1[idx] = dataset['runner'](n_clusters, seed_fraction_1, noise_fraction, completeness_fraction, manually_annotate, n_fold, run_KM)
        perf_metrics_2[idx] = dataset['runner'](n_clusters, seed_fraction_2, noise_fraction, completeness_fraction, manually_annotate, n_fold, False)
    fig = plt.figure()
    subplot_ami = fig.add_subplot(121)
    subplot_ari = fig.add_subplot(122)

    fig.suptitle('Clustering quality over Noise Fraction (SF = {:.2f}, {:.2f})'.format(seed_fraction_1, seed_fraction_2))

    subplot_ami.plot(noise_fractions, perf_metrics_1[:, 1], c='#fc4103', marker='s', markersize=4, label='Seeded KM, SF = ' + str(seed_fraction_1))
    subplot_ami.plot(noise_fractions, perf_metrics_1[:, 3], c='#03a1fc', marker='o', markersize=4, label='Constrained KM, SF = ' + str(seed_fraction_1))
    subplot_ami.plot(noise_fractions, perf_metrics_2[:, 1], c='#fc4103', marker='s', markersize=4, linestyle='dashed', label='Seeded KM, SF = ' + str(seed_fraction_2))
    subplot_ami.plot(noise_fractions, perf_metrics_2[:, 3], c='#03a1fc', marker='o', markersize=4, linestyle='dashed', label='Constrained KM, SF = ' + str(seed_fraction_2))
    subplot_ami.plot(noise_fractions, perf_metrics_1[:, 5], c='#8fe312', marker='^', markersize=4, label='Random KMeans, SF = ' + str(seed_fraction_1))
    subplot_ami.set_title('AMI vs Noise Fraction')
    subplot_ami.set_xlabel('noise_fraction')
    subplot_ami.set_ylabel('ami')
    subplot_ami.grid()
    subplot_ami.legend()

    subplot_ari.plot(noise_fractions, perf_metrics_1[:, 0], c='#fc4103', marker='s', markersize=4, label='Seeded KM, SF = ' + str(seed_fraction_1))
    subplot_ari.plot(noise_fractions, perf_metrics_1[:, 2], c='#03a1fc', marker='o', markersize=4, label='Constrained KM, SF = ' + str(seed_fraction_1))
    subplot_ari.plot(noise_fractions, perf_metrics_2[:, 0], c='#fc4103', marker='s', markersize=4, linestyle='dashed', label='Seeded KM, SF = ' + str(seed_fraction_2))
    subplot_ari.plot(noise_fractions, perf_metrics_2[:, 2], c='#03a1fc', marker='o', markersize=4, linestyle='dashed', label='Constrained KM, SF = ' + str(seed_fraction_2))
    subplot_ari.plot(noise_fractions, perf_metrics_1[:, 4], c='#8fe312', marker='^', markersize=4, label='Random KMeans, SF = ' + str(seed_fraction_1))
    subplot_ari.set_title('ARI vs Noise Fraction')
    subplot_ari.set_xlabel('noise_fraction')
    subplot_ari.set_ylabel('ari')
    subplot_ari.grid()
    subplot_ari.legend()

    plt.savefig('../results/' + 'AMI_ARI_vs_NF_' + str(dataset['name']) + '.png')
    plt.show()
    plt.clf()
    plt.cla()
    plt.close()
