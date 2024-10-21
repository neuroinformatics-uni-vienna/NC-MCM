"""
@authors:
Michael Hofer
Moritz Grosse-Wentrup
"""
import colorsys
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.stats.multitest as smt
from scipy.stats import ttest_ind, ks_2samp


# Functions Markov #

def markovian(sequence, sim_memoryless=1000):
    """Test for 1st order Markovian behavior in a sequence. H0 is that the process could be a 1st order Markov process,
    while HA states that the sequence is likely not stemming from a 1st order Markov process.

    The test looks at the variance over the [t-2]-axis of the transition matrix P(z[t]|z[t-1],z[t-2]) of a sequence and
    compares this to the same variance of similar transition matrices stemming from 1st order Markov processes with the
    same state probabilities.

    Args:
        - sequence: Input sequence.
        - sim_memoryless: Number of simulations for memoryless Markov behavior test statistic.

    Returns:
        - p: P-value (a small value indicates that the sequence is likely not stemming from a Markov process)
        - P1: Empirical conditional transition matrix for first-order Markov behavior.
    """
    sequence = np.asarray(sequence).astype(int)
    Pz0z1z2, states, M, N = compute_joint_probability_matrix_lag2(sequence)

    # P1 = P(z[t]|z[t-1]) = P(z[t],z[t-1]) / P(z[t-1]) = Pz0z1 / Pz1
    Pz0z1 = np.sum(Pz0z1z2, axis=0)
    Pz1 = np.sum(Pz0z1z2, axis=(0, 2))
    if 0 in Pz1:
        print('Found zero in \'P(z[t-1])\'! This should not happen!!!')
    P1 = (Pz0z1 / Pz1.reshape(-1, 1))

    # P2 = P(z[t]|z[t-1],z[t-2]) = P(z[t],z[t-1],z[t-2]) / P(z[t-1],z[t-2]) = Pz0z1z2 / Pz1z2
    Pz1z2 = np.sum(Pz0z1z2, axis=2)
    if 0 in Pz1z2:
        print('Found zero in \'P(z[t],z[t-1],z[t-2])\'! This should not happen!!!')
    P2 = Pz0z1z2 / np.tile(Pz1z2[:, :, np.newaxis], (1, 1, N))

    # Testing
    TH0 = np.zeros(sim_memoryless)
    for kperm in range(sim_memoryless):
        zH0, _ = simulate_markovian(M=M, P=P1)
        # We need to give the individual states since sometimes not all states are in the simulated sequence
        Pz0z1z2H0, _, _, _ = compute_joint_probability_matrix_lag2(zH0, states=states)
        Pz1z2H0 = np.sum(Pz0z1z2H0, axis=2)
        # P2H0 = P(z[t]|z[t-1],z[t-2]) = P(z[t],z[t-1],z[t-2]) / P(z[t-1],z[t-2]) = Pz0z1z2H0 / Pz1z2H0
        P2H0 = Pz0z1z2H0 / np.tile(Pz1z2H0[:, :, np.newaxis], (1, 1, N))
        TH0[kperm] = np.sum(np.var(P2H0, axis=0).flatten())

    # compute p-value
    T = np.sum(np.var(P2, axis=0).flatten())
    p = 1 - np.mean(T >= TH0)
    return p, P1


def stationarity(sequence, chunks=None, sim_stationary=1000, plot=False, verbose=0, num_states=None):
    """Tests if an input sequence breaks the stationary rule. H0 states that the process could be a stationary process,
    while HA states that the sequence is likely stemming from a non-stationary process.

    The Test divides the original sequence into chunks, where each chunk contains an equal number of state transitions
    per state. For each chunk, a transition matrix is computed. The Frobenius norm is used to quantify the differences
    between these matrices, providing a measure of how much the transition patterns vary across chunks.

    To assess whether the observed differences are statistically significant, the Frobenius norms from the test sequence
    are compared to those from a reference distribution with the same state probabilities as the original sequence. This
    comparison is conducted using a two-sample KS-test and/or a two-sample t-test, evaluating whether the variations in
    transition patterns are larger than expected under stationarity.

    A plot can be generated to display the calculations using histograms.

    Note:
        - In testing I found out that with very long sequences (>5000) or with high values of sim_stationary (>1000) the
        test is influenced by the unequal/skewed sample sizes, and becomes very sensitive. As a rule of thumb, try to
        keep the sim_simulations below 1+1000*np.exp(-0.0007*N) where N is the length of the sequence.

    Args:
        - sequence: Input sequence.
        - chunks: Number of parts to split sequence.
        - sim_stationary: Number of simulations for stationary behavior.
        - plot: Boolean indicating whether to plot the results.
        - num_states: If not every state of the state space is present here one can set the state space size (int).
        - verbose: Either 0 or 1 and gives additional print-outs for value 1.

    Returns:
        - p_value_ks: P-value for the T-Test
        - ks_statistic: Effect size for the KS-Test
        - p_value_tt: P-value for the KS-Test
        - tt_statistic: Effect size for the T-Test
    """
    if num_states is None:
        states = np.unique(sequence)
        num_states = len(states)

    transition_dict = get_trans_dict(sequence)
    if chunks is None:
        chunks = calculate_chunks(transition_dict, num_states, verbose=verbose)

    # Split each type of transition for each state into parts
    parts = get_parts(transition_dict, chunks)

    # calculate the empirical conditional transition matrices from the chunks
    emp_transition_matrices = []
    for c in parts:
        emp_m = get_empirical_conditional_matrix(c, num_states)
        emp_transition_matrices.append(emp_m)

    frobenius_norms = get_frobenius_norms(emp_transition_matrices)

    test_stats = []
    full_emp_matrix = compute_conditional_transition_matrix_lag1(sequence)
    for _ in range(sim_stationary):
        sim_seq, _ = simulate_markovian(P=full_emp_matrix, M=len(sequence))
        sim_transition_dict = get_trans_dict(sim_seq)
        sim_parts = get_parts(sim_transition_dict, chunks)
        sim_emp_transition_matrices = []
        for sim_c in sim_parts:
            sim_emp_m = get_empirical_conditional_matrix(sim_c, num_states)
            sim_emp_transition_matrices.append(sim_emp_m)
        sim_frobenius_norms = get_frobenius_norms(sim_emp_transition_matrices)
        test_stats = test_stats + sim_frobenius_norms

    # plot all the results
    if plot:
        fig, ax = plt.subplots()
        ax.hist(test_stats, bins='auto', color='black', density=True, alpha=0.6,
                label='Underlying distribution')
        ax.hist(frobenius_norms, bins='auto', color='green', density=True, alpha=0.6,
                label='Frobenius distribution')
        ax.axvline(0, color='orange', label='True Norm')
        ax.axvline(np.mean(test_stats), color='black', label='Mean underlying frobenius', linestyle='--')
        ax.axvline(np.mean(frobenius_norms), color='green', label='Mean sample frobenius', linestyle='--')
        ax.set_xlabel('Values')
        ax.set_ylabel('Frequency')
        ax.set_title('Histogram of Float Values')
        ax.legend()
        ax.grid(True)
        plt.show()

    sorted_ref = np.sort(test_stats)
    sorted_sample = np.sort(frobenius_norms)

    ks_statistic, p_value_ks = ks_2samp(sorted_ref,
                                        sorted_sample,
                                        alternative='greater')

    t_statistic, p_value_tt = ttest_ind(sorted_sample,
                                   sorted_ref,
                                   alternative='greater',
                                   equal_var=False)

    return p_value_ks, ks_statistic, p_value_tt, t_statistic


# Addition for "markovian" and "stationarity"

def compute_joint_probability_matrix_lag2(sequence, normalize=True, states=None):
    """Compute a joint probability matrix for a lag-2 Markov process.

    Args:
        - sequence: Input sequence.
        - normalize: Boolean to normalize the transition matrix (default is True).
        - states: List of potential states, needed if a potential state is not present in the input sequence (default ->
        unique values from sequence).

    Returns:
        - P: Transition matrix.
        - states: List of unique states in the sequence.
        - M: Length of the sequence.
        - N: Number of unique states.
    """
    if states is None:
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
        # This is done here at the start, so it does not need to be checked after each calculation
        epsilon = 1e-8
        P = np.where(P == 0, epsilon, P)
        P = P / np.sum(P)  # same as P / (M - 2)
    return P, states, M, N


def compute_conditional_transition_matrix_lag1(sequence):
    """Calculates the empirical conditional transition matrix from a sequence of states (sequence).

    Args:
        - sequence: A sequence of states where transitions between consecutive states are to be used to construct the
        empirical conditional transition matrix.

    Returns:
        - emp_m: A empirical conditional transition matrix where rows sum to 1. Each entry (i, j) represents the
        conditional probability of transitioning to state j when in state i.
    """
    states = np.unique(sequence)
    emp_m = np.zeros((len(states), len(states)))
    for i, s in enumerate(sequence[:-1]):
        s2 = sequence[i + 1]
        emp_m[s, s2] += 1

    epsilon = 1e-8
    emp_m = np.where(emp_m == 0, epsilon, emp_m)
    row_sums = emp_m.sum(axis=1, keepdims=True)
    emp_m /= row_sums
    return emp_m


def get_trans_dict(sequence):
    """Generates a dictionary of transitions for each unique state in the input sequence.

    Args:
        - sequence: A sequence of states where transitions between consecutive states are to be recorded.

    Returns:
        - transition_dict: A dictionary where each key is a unique state, and its value is a list of transitions from that state.
    """
    transition_dict = {state: [] for state in np.unique(sequence)}
    for i in range(len(sequence) - 1):
        transition = (sequence[i], sequence[i + 1])
        transition_dict[sequence[i]].append(transition)
    return transition_dict


def calculate_chunks(transition_dict, num_states, verbose=0):
    """Calculates the number of chunks or parts based on the given transition dictionary and number of states. This is
    done using a formula that tries to ensure that each chunk contains at least one transition to each other state.

    Args:
        - transition_dict: A dictionary where each key is a state, and its value is a list of transitions.
        - num_states: The number of unique states in the transition dictionary.
        - verbose: A flag to print the purposed number of parts if set to 1.

    Returns:
        - purposed_parts:  int
            The calculated number of chunks/parts based on the transition data and number of states.
    """
    min_length = min(len(lst) for lst in transition_dict.values())
    per_state = min_length / num_states
    purposed_parts = max(2, int(per_state ** 0.5) + 1)
    if verbose == 1:
        print(f'We purpose {purposed_parts} parts')
    return purposed_parts


def get_parts(transition_dict, chunks):
    """Splits transitions from each state into a specified number of parts or chunks. Chunks are equally sized lists of
    transitions from each state. The chunks are split in a way to maintain the sequential order of transitions in the
    individual parts. If the number of transition from an individual state is not dividable by the number of chunks,
    the remainder (=n) will be distributed among the first n part(s). Therefore, the first chunk can be at most s
    (=amount of unique states) bigger than the last chunk.

    Note:
        - The number of chunks should not exceed the number of transitions for the least abundant state.

    Examples:
        - We want to create 5 chunks. The "transition_dict" contains 7 transitions from state 1, 6 transitions from
        state 2 and 15 transitions from state 3.

        transition_dict = { 1: [(1, 1), (1, 2), (1, 1), (1, 2), (1, 1), (1, 3), (1, 2)],
                            2: [(2, 1), (2, 2), (2, 2), (2, 2), (2, 3), (2, 3)],
                            3: [...]}

        The resulting parts/chunks would have:
            - 1 or 2 transitions from state 1;
            - 1 transition from state 2;
            - 3 transitions from state 3;

        parts = [   [(1, 1), (1, 3), (2, 1), (2, 3), ...],
                    [(1, 2), (1, 2), (2, 2), ...],
                    [(1, 1), (2, 2) ...],
                    [(1, 3), (2, 2) ...],
                    [(1, 3), (2, 3) ...]]

    Args:
        - transition_dict: A dictionary where each key is a state, and its value is a list of transitions from that state.
        - chunks: The number of parts or chunks into which the transitions are to be divided. This value should not
        exceed the minimum number of transitions for any state (=occurrences of any state).

    Returns:
        - parts: A list where each sublist contains transitions for each part/chunk.
    """
    parts = [[] for _ in range(chunks)]
    for state, transitions in transition_dict.items():
        state_chunk_length = len(transitions) // chunks
        end = 0
        for p in range(chunks):
            start = int(state_chunk_length * p)
            end = int(state_chunk_length * (1 + p))
            parts[p] += transitions[start:end]
        for p, remainder in enumerate(transitions[end:]):
            parts[p].append(remainder)
    return parts


def get_frobenius_norms(transition_matrices):
    """This function computes the Frobenius norms between all pairs of transition matrices within a given list. The
    number of computed norms follows the triangular number sequence, as each matrix is compared to every other matrix
    exactly once. For n matrices, the total number of norms will be the triangular number of n−1.

    Args:
        - transition_matrices: A list of transition matrices, where each matrix represents transition probabilities between states.

    Returns:
        - frobenius_norms: A list of Frobenius norms computed for each pair of matrices.
    """
    frobenius_norms = []
    for idx_1, emp_P1 in enumerate(transition_matrices):
        for idx_2, emp_P2 in enumerate(transition_matrices[idx_1 + 1:]):
            m_test = emp_P1 - emp_P2
            frobenius_empirical = np.linalg.norm(m_test, 'fro')
            frobenius_norms.append(frobenius_empirical)
    return frobenius_norms


def get_empirical_conditional_matrix(transitions, num_states):
    """Constructs an empirical conditional transition matrix from a list of transitions and normalizes it.

   Args:
        - transitions: List of transitions, where each transition is a tuple representing a transition between states.
        - num_states: The total number of unique states in the system.

   Returns:
        - emp_m: A normalized empirical conditional transition matrix where rows sum to 1. Each entry (i, j) represents
        the probability of transitioning from state i to state j.
   """
    emp_m = np.zeros((num_states, num_states))
    for t in transitions:
        emp_m[t[0], t[1]] += 1
    if 0 in emp_m:
        emp_m[emp_m == 0] = 1e-8
    row_sums = emp_m.sum(axis=1, keepdims=True)
    emp_m /= row_sums
    return emp_m


# Sequence Generation #

def simulate_markovian(M, P=None, N=1, order=1):
    """Simulate a higher-order Markovian process.

    Args:
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
    """Generate random adjacency matrices.

    Args:
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
    """Generate a non-stationary Markov process. Changes in the process are equally split within length M.

    Args:
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
    """Simulate a random sequence with N states and length M.
    """
    random_sequence = np.random.randint(0, N, size=M)
    return random_sequence

