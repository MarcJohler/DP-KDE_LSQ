import numpy as np

def laplace_mechanism(true_value, sensitivity, epsilon):
    """
    Adds Laplace noise to a true value to provide differential privacy.
    
    :param true_value: The actual value to privatize.
    :param sensitivity: The sensitivity of the query.
    :param epsilon: The privacy budget.
    :return: The differentially private result.
    """
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale, 1)[0]
    return true_value + noise

def compute_private_extremal_points(X, epsilon: float):
    assert len(X.shape) == 2
    if X.shape[0] == 1:
        raise Exception("Data only contains one row. Differential privacy not applicable.")
    # compute the lowest and highest values
    X_sorted = np.sort(X, axis = 0)
    X_lowest = X_sorted[0, 0]
    X_second_lowest = X_sorted[1, 0]
    X_highest = X_sorted[-1, 0]
    X_second_highest = X_sorted[-2, 0]
    # compute the local sensitivity for min and max
    local_min_sensitivity = abs(X_second_lowest - X_lowest)
    local_max_sensitivity = abs(X_highest - X_second_highest)
    # apply the laplace mechanism
    private_min = laplace_mechanism(X_lowest, local_min_sensitivity, epsilon)
    private_max = laplace_mechanism(X_highest, local_max_sensitivity, epsilon)
    return private_min, private_max

def compute_private_row_count(X, epsilon: float):
    # compute the local mean by composition
    return np.max([laplace_mechanism(len(X), 1, epsilon), 2])

def compute_private_sum(X, X_boundaries: tuple, epsilon: float):
    max_change = np.abs(np.array(X_boundaries)).max()
    return laplace_mechanism(np.sum(X), max_change, epsilon)
    
def compute_private_mean(X, X_boundaries: tuple, epsilon: float):
    private_row_count = compute_private_row_count(X, epsilon)
    private_sum = compute_private_sum(X, X_boundaries, epsilon)
    return private_sum / private_row_count

def compute_private_var(X, X_boundaries: tuple, epsilon: float):
    private_row_count = compute_private_row_count(X, epsilon)
    private_mean = compute_private_mean(X, X_boundaries, epsilon)
    max_diff = np.max([X_boundaries[1] - private_mean, private_mean - X_boundaries[0]])
    # compute the local variance sensitivity
    local_var_sensitivy = max_diff**2 / private_row_count
    return np.max([laplace_mechanism(np.var(X, ddof = 1), local_var_sensitivy, epsilon), 0])

def compute_private_std(X, X_boundaries: tuple, epsilon: float):
    private_var = compute_private_var(X, X_boundaries, epsilon)
    return np.sqrt(private_var)