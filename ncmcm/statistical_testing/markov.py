import colorsys
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.stats.multitest as smt


# Functions Markov #

def markovian(sequence, sim_memoryless=1000):
    """
        Test for 1st order Markovian behavior in a sequence. H0 is that the process is a 1st order markov process.

        Parameters:
        - sequence: Input sequence.
        - sim_memoryless: Number of simulations for memoryless Markov behavior test statistic.

        Returns:
        - p: Probability of Markovian behavior.
        - P1: Transition matrix for first-order Markov behavior.
    """
    sequence = np.asarray(sequence).astype(int)
    Pz0z1z2, states, M, N = compute_transition_matrix_lag2(sequence)

    # This is done here at the start, so it does not need to be checked after each calculation
    epsilon = 1e-8
    Pz0z1z2 = np.where(Pz0z1z2 == 0, epsilon, Pz0z1z2)
    Pz0z1z2 = Pz0z1z2 / np.sum(Pz0z1z2)  # here I normalize it so the sum is 1 again

    # P1 = P(z[t]|z[t-1]) = P(z[t],z[t-1]) / P(z[t-1]) = Pz0z1 / Pz1
    Pz0z1 = np.sum(Pz0z1z2, axis=0)
    Pz1 = np.sum(Pz0z1z2, axis=(0, 2))
    if 0 in Pz1:
        print('This should not happen!!!')
    P1 = (Pz0z1 / Pz1.reshape(-1, 1))

    # P2 = P(z[t]|z[t-1],z[t-2]) = P(z[t],z[t-1],z[t-2]) / P(z[t-1],z[t-2]) = Pz0z1z2 / Pz1z2
    Pz1z2 = np.sum(Pz0z1z2, axis=2)
    if 0 in Pz1z2:
        print('This should not happen!!!')
    P2 = Pz0z1z2 / np.tile(Pz1z2[:, :, np.newaxis], (1, 1, N))

    # Testing
    TH0 = np.zeros(sim_memoryless)
    for kperm in range(sim_memoryless):
        zH0, _ = simulate_markovian(M=M, P=P1)
        Pz0z1z2H0 = np.zeros((N, N, N))
        for m in range(2, M):
            i = zH0[m]  # col
            j = zH0[m - 1]  # row
            k = zH0[m - 2]  # depth
            Pz0z1z2H0[k, j, i] += 1

        Pz0z1z2H0 = Pz0z1z2H0 / (M - 2)
        Pz1z2H0 = np.sum(Pz0z1z2H0, axis=2)

        # I am replacing zeros in Pz1z2H0 with epsilon, so we do not encounter RuntimeWarnings
        epsilon = 1e-8
        Pz1z2H0 = np.where(Pz1z2H0 == 0, epsilon, Pz1z2H0)
        Pz1z2H0 = Pz1z2H0 / np.sum(Pz1z2H0)

        # P2H0 = P(z[t]|z[t-1],z[t-2]) = P(z[t],z[t-1],z[t-2]) / P(z[t-1],z[t-2]) = Pz0z1z2H0 / Pz1z2H0
        P2H0 = Pz0z1z2H0 / np.tile(Pz1z2H0[:, :, np.newaxis], (1, 1, N))
        TH0[kperm] = np.sum(np.var(P2H0, axis=0).flatten())

    # compute p-value
    T = np.sum(np.var(P2, axis=0).flatten())
    p = 1 - np.mean(T >= TH0)
    return p, P1


def compute_transition_matrix_lag2(sequence, normalize=True):
    """
        Compute a transition matrix for a lag-2 Markov process.

        Parameters:
        - sequence: Input sequence.
        - normalize: Boolean to normalize the transition matrix (default is True).

        Returns:
        - P: Transition matrix.
        - states: List of unique states in the sequence.
        - M: Length of the sequence.
        - N: Number of unique states.
    """
    states = sorted(np.unique(sequence))
    M = len(sequence)
    N = len(states)
    # tensor is created
    P = np.zeros((N, N, N))
    for m in range(2, M):
        i = sequence[m]
        j = sequence[m - 1]
        k = sequence[m - 2]
        # from k to j to i
        P[k, j, i] += 1
    if normalize:
        P = P / np.sum(P)  # same as P / (M - 2)
    return P, states, M, N


def stationarity(sequence, chunks=None, sim_stationary=1000, plot=False, verbose=0):
    """
        Test stationarity in input sequence.

        Parameters:
        - sequence: Input sequence.
        - parts: Number of parts to split sequence.
        - sim_stationary: Number of simulations for stationary behavior.
        - plot: Boolean indicating whether to plot the results.

        Returns:
        - mean_unadjusted_p_value: Mean unadjusted p-value.
        - mean_FDR_adjusted_p_value: Mean False Discovery Rate (FDR) adjusted p-value.
    """
    states = np.unique(sequence)
    num_states = len(states)
    transition_dict = {state: [] for state in np.unique(sequence)}
    for i in range(len(sequence) - 1):
        transition = (sequence[i], sequence[i + 1])
        transition_dict[sequence[i]].append(transition)

    if chunks is None:
        min_length = min(len(lst) for lst in transition_dict.values())
        # approximate amount of transitions to each state from the least populated state
        per_state = min_length / num_states
        purposed_parts = max(2, int(per_state ** 0.5) + 1)
        if verbose == 1:
            print(f'We purpose {purposed_parts} parts')
        chunks = purposed_parts

    # Split each type of transition for each state into parts
    parts = [[] for _ in range(chunks)]
    for state, transitions in transition_dict.items():
        # random.shuffle(transitions)
        state_chunk_length = len(transitions) // chunks
        for p in range(chunks - 1):
            start = int(state_chunk_length * p)
            end = int(state_chunk_length * (1 + p))
            parts[p] += transitions[start:end]
        parts[chunks - 1] += transitions[int(state_chunk_length * (chunks - 1)):]

    # Making test statistic
    test_stats = []
    test_matrices = make_random_adj_matrices(num_matrices=sim_stationary, matrix_shape=(num_states, num_states))
    for idx1, m1 in enumerate(test_matrices):
        for idx2, m2 in enumerate(test_matrices[idx1 + 1:]):
            m_diff = m1 - m2
            frobenius_norm = np.linalg.norm(m_diff, 'fro')
            test_stats.append(frobenius_norm)

    # The 0.05 percentile for significance
    first_percentile = np.percentile(test_stats, 5)

    # calculate the empirical transition matrices from the chunks
    emp_transition_matrices = []
    for c in parts:
        emp_m = np.zeros((num_states, num_states))
        for t in c:
            emp_m[t[0], t[1]] += 1
        # Normalize rows to ensure they sum up to 1
        if 0 in emp_m:
            # print('We fill 0 in the transition matrix with very small values.')
            emp_m[emp_m == 0] = 1e-8
        row_sums = emp_m.sum(axis=1, keepdims=True)
        emp_m /= row_sums
        emp_m_t1 = np.sum(emp_m, axis=0)
        emp_m = emp_m / emp_m_t1
        emp_transition_matrices.append(emp_m)

    # calculate frobenius norms between the empirical transition matrices
    frobenius_norms = []
    for idx_1, emp_P1 in enumerate(emp_transition_matrices):
        for idx_2, emp_P2 in enumerate(emp_transition_matrices[idx_1 + 1:]):
            # print(f'We compare matrx {idx_1} to matrix {idx_1+idx_2+1}')
            m_test = emp_P1 - emp_P2
            frobenius_empirical = np.linalg.norm(m_test, 'fro')
            frobenius_norms.append(frobenius_empirical)

    # plot all the results
    if plot:
        plt.hist(test_stats, bins='auto', edgecolor='black')  # Adjust the number of bins as needed
        plt.axvline(0, color='orange', label='True Norm')
        for f in frobenius_norms:
            plt.axvline(f, color='green')
        plt.axvline(f, color='green', label='Frobenius Norm between chunks')
        plt.axvline(first_percentile, color='red', label='First 0.05 percentile')

        plt.xlabel('Values')
        plt.ylabel('Frequency')
        plt.title('Histogram of Float Values')
        plt.legend()  # Display the legend
        plt.grid(True)
        plt.show()

    p_values = [1 - (np.sum(test_stats >= f) / len(test_stats)) for f in frobenius_norms]
    _, FDR_adjusted_p_values, _, _ = smt.multipletests(p_values, method='fdr_bh')
    mean_unadjusted_p_value = 1 - (np.sum(test_stats >= np.mean(frobenius_norms)) / len(test_stats))

    return np.mean(FDR_adjusted_p_values), mean_unadjusted_p_value


# Sequence Generation #

def simulate_markovian(M, P=None, N=1, order=1):
    """
        Simulate a higher-order Markovian process.

        Parameters:
        - M: Length of the sequence.
        - P: Transition matrix (default is None for random generation).
        - N: Number of states (default is 1).
        - order: Order of the Markov process (default is 1).

        Returns:
        - z: Simulated sequence.
        - P: Used transition matrix.
    """
    if P is None:
        # Initialize the transition matrix for higher-order Markov process
        dims = [N] * (order + 1)
        P = np.random.rand(*dims)
        # Normalize transition matrix probabilities
        P /= np.sum(P, axis=-1, keepdims=True)
    else:
        # Assume P has the correct shape for the given order
        N = P.shape[0]
        order = len(P.shape) - 1

    # Initialize the state sequence
    z = np.zeros(M, dtype=int)
    # Randomly initialize the first 'order' states
    for i in range(order):
        z[i] = np.random.randint(N)

    for m in range(order, M):
        # Extract the previous 'order' states
        prev_states = tuple(z[m - order:m])
        # Get the transition probabilities for the current state
        probabilities = P[prev_states]
        # Choose the next state based on the transition probabilities
        z[m] = np.random.choice(np.arange(N), p=probabilities)

    return z, P


def make_random_adj_matrices(num_matrices=1000, matrix_shape=(10, 10), sparse=False):
    """
        Generate random adjacency matrices.

        Parameters:
        - num_matrices: Number of matrices to generate.
        - matrix_shape: Shape of each matrix.
        - sparse: Can be applied to get more sparse transition matrices.

        Returns:
        - transition_matrices: List of generated matrices.
    """
    transition_matrices = []

    for _ in range(num_matrices):
        if sparse:
            random_matrix = np.random.dirichlet(np.ones(matrix_shape[0]), size=matrix_shape[0])
        else:
            random_matrix = np.random.rand(*matrix_shape)

        # Normalize rows to ensure they add up to 1
        transition_matrix = random_matrix / random_matrix.sum(axis=1, keepdims=True)
        transition_matrices.append(transition_matrix)

    return transition_matrices


def non_stationary_process(M, N, changes=4):
    """
    Generate a non-stationary Markov process. Changes in the process are equally split within length M.

    Parameters:
    - M: Length of the sequence.
    - N: Number of states.
    - changes: Number of changes within the process.

    Returns:
    - seq: Generated sequence.
    """
    l = int(np.floor(M / changes))
    last = M - ((changes - 1) * l)
    seq = []

    for c in range(changes - 1):
        seq += list(simulate_markovian(M=l, N=N, order=1)[0])
    seq += list(simulate_markovian(M=last, N=N, order=1)[0])

    return seq


def simulate_random_sequence(M, N):
    """
        Simulate a random sequence with N states and length M.
    """
    random_sequence = np.random.randint(0, N, size=M)
    return random_sequence
