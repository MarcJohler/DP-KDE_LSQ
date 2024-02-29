import numpy as np

def laplace_mechanism(true_value, sensitivity, epsilon, size = 1):
    """
    Adds Laplace noise to a true value to provide differential privacy.
    
    :param true_value: The actual value to privatize.
    :param sensitivity: The sensitivity of the query.
    :param epsilon: The privacy budget.
    :return: The differentially private result.
    """
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale, size = size)
    if size is None:
        noise = noise[0]
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
    return laplace_mechanism(len(X), 1, epsilon)

def compute_private_sum(X, X_boundaries: tuple, epsilon: float):
    max_change = np.abs(np.array(X_boundaries)).max()
    return laplace_mechanism(np.sum(X), max_change, epsilon)
    
def compute_private_mean(X, X_boundaries: tuple, epsilon: float):
    private_row_count = max([compute_private_row_count(X, epsilon), 2])
    private_sum = compute_private_sum(X, X_boundaries, epsilon)
    return private_sum / private_row_count

def max_local_var_sensitivity(X, X_boundaries: tuple, k: int):
    # this might only work when X_boundaries are domain boundaries
    max_diff = X_boundaries[1] - (X_boundaries[1] - X_boundaries[0]) / 2
    return max_diff**2 / (len(X) - k)

def compute_smooth_var_sensitivity(X, X_boundaries: tuple, epsilon: float, delta: float = 0.001):
    beta = epsilon / (2 * np.log(2 / delta))
    objective_fun = lambda k: np.e**(-beta * k) * max_local_var_sensitivity(X, X_boundaries, k)
    opt_value = 0
    for i in range(1, max(1, len(X) - 1)):
        result = objective_fun(i)
        if result > opt_value:
            opt_value = result
    return 2 * opt_value
        
def compute_private_var(X, X_boundaries: tuple, epsilon: float, delta: float = 0.001):
    # probably the mean computation is not DP conform for local sensitivity
    # private_row_count = compute_private_row_count(X, epsilon)
    # private_mean = compute_private_mean(X, X_boundaries, epsilon)
    
    # compute the local variance sensitivity
    local_var_sensitivity = compute_smooth_var_sensitivity(X, X_boundaries, epsilon, delta)
    return np.max([laplace_mechanism(np.var(X, ddof = 1), local_var_sensitivity, epsilon), 0])

def compute_private_std(X, X_boundaries: tuple, epsilon: float, delta: float = 0.001):
    private_var = compute_private_var(X, X_boundaries, epsilon, delta)
    return np.sqrt(private_var)