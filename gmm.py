import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import multivariate_normal

PLOT_COLORS = ['red', 'green', 'blue', 'orange']  # Colors for your plots
K = 4           # Number of Gaussians in the mixture model
NUM_TRIALS = 3  # Number of trials to run (can be adjusted for debugging)
UNLABELED = -1  # Cluster label for unlabeled data points (do not change)


def main(is_semi_supervised, trial_num):
    """Problem 3: EM for Gaussian Mixture Models (unsupervised and semi-supervised)"""
    print('Running {} EM algorithm...'
          .format('semi-supervised' if is_semi_supervised else 'unsupervised'))

    # Load dataset
    train_path = os.path.join('.', 'train.csv')
    x_all, z_all = load_gmm_dataset(train_path)

    # Split into labeled and unlabeled examples
    labeled_idxs = (z_all != UNLABELED).squeeze()
    x_tilde = x_all[labeled_idxs, :]   # Labeled examples
    z_tilde = z_all[labeled_idxs, :]   # Corresponding labels
    x = x_all[~labeled_idxs, :]        # Unlabeled examples

    n, d = x_all.shape  # n: number of data points, d: dimensionality
    m = x.shape[0]  # m: number of unlabeled examples
    K = np.unique(z_all).size  # Assuming K is the number of unique classes in z_all



    # *** START CODE HERE ***
    # (1) Initialize mu and sigma by splitting the n_examples data points uniformly at random
    # into K groups, then calculating the sample mean and covariance for each group
    indices = np.random.permutation(n)
    mu = np.array([x_all[indices[i::K]].mean(axis=0) for i in range(K)])
    sigma = np.array([np.cov(x_all[indices[i::K]], rowvar=False) for i in range(K)])
    # (2) Initialize phi to place equal probability on each Gaussian
    # phi should be a numpy array of shape (K,)
    phi = np.ones(K) / K  # numpy array of shape (K,)

    # (3) Initialize the w values to place equal probability on each Gaussian
    # w should be a numpy array of shape (m, K)
    w = np.ones((m, K)) / K  # numpy array of shape (m, K)
    # *** END CODE HERE ***

    if is_semi_supervised:
        w = run_semi_supervised_em(x, x_tilde, z_tilde, w, phi, mu, sigma)
    else:
        w = run_em(x, w, phi, mu, sigma)

    # Plot your predictions
    z_pred = np.zeros(m)
    if w is not None:  # Just a placeholder for the starter code
        for i in range(m):
            z_pred[i] = np.argmax(w[i])

    plot_gmm_preds(x, z_pred, is_semi_supervised, plot_id=trial_num)


def run_em(x, w, phi, mu, sigma):
    """Problem 3(d): EM Algorithm (unsupervised).

    See inline comments for instructions.

    Args:
        x: Design matrix of shape (n_examples, dim).
        w: Initial weight matrix of shape (n_examples, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (dim,).
        sigma: Initial cluster covariances, list of k arrays of shape (dim, dim).

    Returns:
        Updated weight matrix of shape (n_examples, k) resulting from EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    eps = 1e-3  # Convergence threshold
    max_iter = 1000
    n_examples, dim = x.shape

    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    k = phi.shape[0]
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        #pass  # Just a placeholder for the starter code
        # *** START CODE HERE
        
        # (1) E-step: Update your estimates in w
        for j in range(phi.shape[0] ):
            likelihood = multivariate_normal.pdf(x, mean=mu[j], cov=sigma[j])
            w[:, j] = phi[j] * likelihood
        
        # (2) M-step: Update the model parameters phi, mu, and sigma
        w = w / w.sum(axis=1, keepdims=True)
        N_k = w.sum(axis=0)  # Effective number of data points assigned to each cluster
        phi = N_k / n_examples
        
        mu = (w.T @ x) / N_k[:, np.newaxis]

        # Update covariances
        for j in range(k):
            x_centered = x - mu[j]
            sigma[j] = (w[:, j, np.newaxis] * x_centered).T @ x_centered / N_k[j]
            sigma[j] += np.eye(dim) * 1e-6  # Adding a small value to the diagonal for numerical stability

        # (3) Compute the log-likelihood of the data to check for convergence.
        prev_ll = ll
        ll = np.sum(np.log(np.sum([phi[j] * multivariate_normal.pdf(x, mean=mu[j], cov=sigma[j]) for j in range(k)], axis=0)))
        it += 1
        
        # By log-likelihood, we mean `ll = sum_x[log(sum_z[p(x|z) * p(z)])]`.
        # We define convergence by the first iteration where abs(ll - prev_ll) < eps.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.
        # *** END CODE HERE ***

    return w


def run_semi_supervised_em(x, x_tilde, z_tilde, w, phi, mu, sigma):
    """Problem 3(e): Semi-Supervised EM Algorithm.

    See inline comments for instructions.

    Args:
        x: Design matrix of unlabeled examples of shape (n_examples_unobs, dim).
        x_tilde: Design matrix of labeled examples of shape (n_examples_obs, dim).
        z_tilde: Array of labels of shape (n_examples_obs, 1).
        w: Initial weight matrix of shape (n_examples, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (dim,).
        sigma: Initial cluster covariances, list of k arrays of shape (dim, dim).

    Returns:
        Updated weight matrix of shape (n_examples, k) resulting from semi-supervised EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    alpha = 20.  # Weight for the labeled examples
    eps = 1e-3   # Convergence threshold
    max_iter = 1000
    n_examples_unobs, dim = x.shape  # Number of unlabeled data points and dimensionality
    n_examples_obs = x_tilde.shape[0]  # Number of labeled data points
    k = phi.shape[0]  # Number of Gaussian components

    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        #pass  # Just a placeholder for the starter code
        # *** START CODE HERE ***
        # (1) E-step: Update your estimates in w
        for j in range(k):
            likelihood = multivariate_normal.pdf(x, mean=mu[j], cov=sigma[j])
            w[:, j] = phi[j] * likelihood
        
        # Normalize w across all clusters for the unlabeled data
        w = w / w.sum(axis=1, keepdims=True)

        # Create a hard assignment matrix for the labeled data (1-hot encoding)
        w_tilde = np.zeros((n_examples_obs, k))
        w_tilde[np.arange(n_examples_obs), z_tilde.flatten().astype(int)] = 1

        
        # (2) M-step: Update the model parameters phi, mu, and sigma
        # Update mixture priors phi
        N_k_unobs = w.sum(axis=0)
        N_k_obs = w_tilde.sum(axis=0)
        N_k = N_k_unobs + alpha * N_k_obs
        phi = N_k / (n_examples_unobs + alpha * n_examples_obs)

        # Update means mu
        mu_unobs = (w.T @ x) / N_k[:, np.newaxis]
        mu_obs = (w_tilde.T @ x_tilde) / N_k[:, np.newaxis]
        mu = (mu_unobs + alpha * mu_obs) / (1 + alpha)

        # Update covariances sigma
        for j in range(k):
            x_centered_unobs = x - mu[j]
            sigma_unobs = (w[:, j, np.newaxis] * x_centered_unobs).T @ x_centered_unobs
            
            x_centered_obs = x_tilde - mu[j]
            sigma_obs = (w_tilde[:, j, np.newaxis] * x_centered_obs).T @ x_centered_obs
            
            sigma[j] = (sigma_unobs + alpha * sigma_obs) / N_k[j]
            sigma[j] += np.eye(dim) * 1e-6  # Adding a small value to the diagonal for numerical stability

        
        # (3) Compute the log-likelihood of the data to check for convergence.
        prev_ll = ll
        ll_unobs = np.sum(np.log(np.sum([phi[j] * multivariate_normal.pdf(x, mean=mu[j], cov=sigma[j]) for j in range(k)], axis=0)))
        ll_obs = np.sum(np.log([phi[z] * multivariate_normal.pdf(x_tilde[i], mean=mu[z], cov=sigma[z]) for i, z in enumerate(z_tilde.astype(int).flatten())]))
        ll = ll_unobs + alpha * ll_obs
        
        it += 1
        # Hint: Make sure to include alpha in your calculation of ll.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.
        # *** END CODE HERE ***

    return w


# *** START CODE HERE ***
# Helper functions
# *** END CODE HERE ***


def plot_gmm_preds(x, z, with_supervision, plot_id):
    """Plot GMM predictions on a 2D dataset `x` with labels `z`.

    Write to the output directory, including `plot_id`
    in the name, and appending 'ss' if the GMM had supervision.

    NOTE: You do not need to edit this function.
    """
    plt.figure(figsize=(12, 8))
    plt.title('{} GMM Predictions'.format('Semi-supervised' if with_supervision else 'Unsupervised'))
    plt.xlabel('x_1')
    plt.ylabel('x_2')

    for x_1, x_2, z_ in zip(x[:, 0], x[:, 1], z):
        print(z_)
        #color = 'green' if z_ < 0 else PLOT_COLORS[int(z_-1)]
        color = 'gray' if z_ < 0 else PLOT_COLORS[int(z_-1)]
        alpha = 0.25 if z_ < 0 else 0.75
        plt.scatter(x_1, x_2, marker='.', c=color, alpha=alpha)

    file_name = 'pred{}_{}.pdf'.format('_ss' if with_supervision else '', plot_id)
    save_path = os.path.join('.', file_name)
    plt.savefig(save_path)


def load_gmm_dataset(csv_path):
    """Load dataset for Gaussian Mixture Model.

    Args:
         csv_path: Path to CSV file containing dataset.

    Returns:
        x: NumPy array shape (n_examples, dim)
        z: NumPy array shape (n_exampls, 1)

    NOTE: You do not need to edit this function.
    """

    # Load headers
    with open(csv_path, 'r') as csv_fh:
        headers = csv_fh.readline().strip().split(',')

    # Load features and labels
    x_cols = [i for i in range(len(headers)) if headers[i].startswith('x')]
    z_cols = [i for i in range(len(headers)) if headers[i] == 'z']

    x = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols, dtype=float)
    z = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=z_cols, dtype=float)

    if z.ndim == 1:
        z = np.expand_dims(z, axis=-1)

    return x, z


if __name__ == '__main__':
    np.random.seed(229)
    # Run NUM_TRIALS trials to see how different initializations
    # affect the final predictions with and without supervision
    for t in range(NUM_TRIALS):
        #main(is_semi_supervised=False, trial_num=t)

        # *** START CODE HERE ***
        # Once you've implemented the semi-supervised version,
        # uncomment the following line.
        # You do not need to add any other lines in this code block.
        main(is_semi_supervised=True, trial_num=t)
        # *** END CODE HERE ***
