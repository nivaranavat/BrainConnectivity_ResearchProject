from source.utils.io_utils import readMRIFile, load_txt
from source.utils.matrix_utils import createCorrelationMatrix, findThreshold, binarize, randomizeCorrelationMatrix, compute_correlation_matrix
from source.utils.plot_utils import plot_timeseries, plot_correlation_matrix, plot_degree_histogram, plot_histogram, plot_matrix_difference
from source.utils.phase_utils import *
import numpy as np
import matplotlib.pyplot as plt
import bct
import random

# Only keep workflow/analysis-specific code here. Use utils for general functions.

#np.set_printoptions(threshold=np.inf)
def pick_four_unique_nodes_quickly(n, seed=None):
    '''
    This is equivalent to np.random.choice(n, 4, replace=False)
    Another fellow suggested np.random.random_sample(n).argpartition(4) which is
    clever but still substantially slower.
    '''
    rng = get_rng(seed)
    k = rng.randint(n**4)
    a = k % n
    b = k // n % n
    c = k // n ** 2 % n
    d = k // n ** 3 % n
    if (a != b and a != c and a != d and b != c and b != d and c != d):
        return (a, b, c, d)
    else:
        # the probability of finding a wrong configuration is extremely low
        # unless for extremely small n. if n is extremely small the
        # computational demand is not a problem.

        # In my profiling it only took 0.4 seconds to include the uniqueness
        # check in 1 million runs of this function so I think it is OK.
        return pick_four_unique_nodes_quickly(n, rng)
    
def randmio_und_signed(R, itr, seed=None):
    '''
    This function randomizes an undirected weighted network with positive
    and negative weights, while simultaneously preserving the degree
    distribution of positive and negative weights. The function does not
    preserve the strength distribution in weighted networks.
    Parameters
    ----------
    W : NxN np.ndarray
        undirected binary/weighted connection matrix
    itr : int
        rewiring parameter. Each edge is rewired approximately itr times.
    seed : hashable, optional
        If None (default), use the np.random's global random state to generate random numbers.
        Otherwise, use a new np.random.RandomState instance seeded with the given value.
    Returns
    -------
    R : NxN np.ndarray
        randomized network
    '''
    rng = get_rng(seed)
    R = R.copy()
    n = len(R)

    itr *= int(n * (n -1) / 2)

    max_attempts = int(np.round(n / 2))
    eff = 0

    for it in range(int(itr)):
        att = 0
        while att <= max_attempts:

            a, b, c, d = pick_four_unique_nodes_quickly(n, rng)

            r0_ab = R[a, b]
            r0_cd = R[c, d]
            r0_ad = R[a, d]
            r0_cb = R[c, b]

            #rewiring condition
            if (    np.sign(r0_ab) == np.sign(r0_cd) and
                    np.sign(r0_ad) == np.sign(r0_cb) and
                    np.sign(r0_ab) != np.sign(r0_ad)):

                R[a, d] = R[d, a] = r0_ab
                R[a, b] = R[b, a] = r0_ad

                R[c, b] = R[b, c] = r0_cd
                R[c, d] = R[d, c] = r0_cb

                eff += 1
                break

            att += 1

    return R, eff

def get_rng(seed=None):
    """
    By default, or if `seed` is np.random, return the global RandomState
    instance used by np.random.
    If `seed` is a RandomState instance, return it unchanged.
    Otherwise, use the passed (hashable) argument to seed a new instance
    of RandomState and return it.
    Parameters
    ----------
    seed : hashable or np.random.RandomState or np.random, optional
    Returns
    -------
    np.random.RandomState
    """
    if seed is None or seed == np.random:
        return np.random.mtrand._rand
    elif isinstance(seed, np.random.RandomState):
        return seed
    try:
        rstate =  np.random.RandomState(seed)
    except ValueError:
        rstate = np.random.RandomState(random.Random(seed).randint(0, 2**32-1))
    return rstate

def null_model_und_sign(W, bin_swaps=2, wei_freq=.1, seeds=None):
    '''
    This function randomizes an undirected network with positive and
    negative weights, while preserving the degree and strength
    distributions. This function calls randmio_und.m
    Parameters
    ----------
    W : NxN np.ndarray
        undirected weighted connection matrix
    bin_swaps : int
        average number of swaps in each edge binary randomization. Default
        value is 5. 0 swaps implies no binary randomization.
    wei_freq : float
        frequency of weight sorting in weighted randomization. 0<=wei_freq<1.
        wei_freq == 1 implies that weights are sorted at each step.
        wei_freq == 0.1 implies that weights sorted each 10th step (faster,
            default value)
        wei_freq == 0 implies no sorting of weights (not recommended)
    seed : hashable, optional
        If None (default), use the np.random's global random state to generate random numbers.
        Otherwise, use a new np.random.RandomState instance seeded with the given value.
    Returns
    -------
    W0 : NxN np.ndarray
        randomized weighted connection matrix
    R : 4-tuple of floats
        Correlation coefficients between strength sequences of input and
        output connection matrices, rpos_in, rpos_out, rneg_in, rneg_out
    Notes
    -----
    The value of bin_swaps is ignored when binary topology is fully
        connected (e.g. when the network has no negative weights).
    Randomization may be better (and execution time will be slower) for
        higher values of bin_swaps and wei_freq. Higher values of bin_swaps
        may enable a more random binary organization, and higher values of
        wei_freq may enable a more accurate conservation of strength
        sequences.
    R are the correlation coefficients between positive and negative
        strength sequences of input and output connection matrices and are
        used to evaluate the accuracy with which strengths were preserved.
        Note that correlation coefficients may be a rough measure of
        strength-sequence accuracy and one could implement more formal tests
        (such as the Kolmogorov-Smirnov test) if desired.
    '''
    rng = get_rng(seeds)
    if not np.allclose(W,W.T):
        raise KeyError("Input must be undirected")
    W = W.copy()
    n = len(W)
    np.fill_diagonal(W, 0)  # clear diagonal
    Ap = (W > 0)  # positive adjmat
    An = (W < 0)  # negative adjmat

    if np.size(np.where(Ap.flat)) < (n * (n - 1)):
        W_r, eff = randmio_und_signed(W,bin_swaps,seed = rng)
        Ap_r = W_r > 0
        An_r = W_r < 0
    else:
        Ap_r = Ap
        An_r = An

    W0 = np.zeros((n, n))
    for s in (1, -1):
        if s == 1:
            Acur = Ap
            A_rcur = Ap_r
        else:
            Acur = An
            A_rcur = An_r

        S = np.sum(W * Acur, axis=0)  # strengths
        Wv = np.sort(W[np.where(np.triu(Acur))])  # sorted weights vector
        i, j = np.where(np.triu(A_rcur))
        Lij, = np.where(np.triu(A_rcur).flat)  # weights indices

        P = np.outer(S, S)

        if wei_freq == 0:  # get indices of Lij that sort P
            Oind = np.argsort(P.flat[Lij])  # assign corresponding sorted
            W0.flat[Lij[Oind]] = s * Wv  # weight at this index
        else:
            wsize = np.size(Wv)
            wei_period = np.round(1 / wei_freq)  # convert frequency to period
            lq = np.arange(wsize, 0, -wei_period, dtype=int)
            for m in lq:  # iteratively explore at this period
                # get indices of Lij that sort P
                Oind = np.argsort(P.flat[Lij])
                #[np.array([i for i in range(min(m.item(),wei_period))])]
                R = rng.permutation(m)[:min(m.item(),int(wei_period.item()))]
                for q, r in enumerate(R):
                    # choose random index of sorted expected weight
                    o = Oind[r]
                    W0.flat[Lij[o]] = s * Wv[r]  # assign corresponding weight

                    # readjust expected weighted probability for i[o],j[o]
                    f = 1 - Wv[r] / S[i[o]]
                    P[i[o], :] *= f
                    P[:, i[o]] *= f
                    f = 1 - Wv[r] / S[j[o]]
                    P[j[o], :] *= f
                    P[:, j[o]] *= f

                    # readjust strength of i[o]
                    S[i[o]] -= Wv[r]
                    # readjust strength of j[o]
                    S[j[o]] -= Wv[r]

                O = Oind[R]
                # remove current indices from further consideration
                Lij = np.delete(Lij, O)
                i = np.delete(i, O)
                j = np.delete(j, O)
                Wv = np.delete(Wv, R)

    W0 = W0 + W0.T

    rpos_in = np.corrcoef(np.sum(W * (W > 0), axis=0),
                          np.sum(W0 * (W0 > 0), axis=0))
    rpos_ou = np.corrcoef(np.sum(W * (W > 0), axis=1),
                          np.sum(W0 * (W0 > 0), axis=1))
    rneg_in = np.corrcoef(np.sum(-W * (W < 0), axis=0),
                          np.sum(-W0 * (W0 < 0), axis=0))
    rneg_ou = np.corrcoef(np.sum(-W * (W < 0), axis=1),
                          np.sum(-W0 * (W0 < 0), axis=1))
    return W0, (rpos_in[0, 1], rpos_ou[0, 1], rneg_in[0, 1], rneg_ou[0, 1])

def main(folder_path, start=2, end=152, plot=True):
    """
    Main workflow for brain connectivity analysis using utils functions.
    """
    # Load data
    x, data = readMRIFile(folder_path, size=200)
    if plot:
        plot_timeseries(data, x, title="ROI Time Series", show=plot)
    # Correlation
    corr_matrix = createCorrelationMatrix(data, method="pearson")
    if plot:
        plot_correlation_matrix(corr_matrix, title="Correlation Matrix", show=plot)
    # Binarize
    threshold = findThreshold(corr_matrix, density=0.15)
    binary_matrix = binarize(corr_matrix, threshold)
    if plot:
        plot_correlation_matrix(binary_matrix, title="Binarized Correlation Matrix", show=plot)
    # Network
    network = createNetwork(binary_matrix)
    # Randomize
    null2 = randomizeCorrelationMatrix(corr_matrix)
    null3, _ = bct.randmio_und_signed(binary_matrix, 2)
    print("Done.")
    return {
        'data': data,
        'corr_matrix': corr_matrix,
        'binary_matrix': binary_matrix,
        'network': network,
        'null2': null2,
        'null3': null3,
    }

if __name__ == "__main__":
    main("/path/to/data", start=2, end=152, plot=True)

