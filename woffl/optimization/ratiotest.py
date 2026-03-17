import numpy as np


def check_array_length(*arrays):
    lengths = [len(arr) for arr in arrays]
    if len(set(lengths)) != 1:
        raise ValueError("All numpy arrays must have the same length. Got lengths: {}".format(lengths))


def feasible_point(x_vec: np.ndarray, a_mat: np.ndarray, b_vec: np.ndarray) -> None:
    """Feasible Point

    Verify whether the x_vector is actually inside the feasible set or not.
    The inequalities have to take the form Ax >= b.

    Args:
        x_vec (np.ndarray): Vector of Feasible Points
        p_vec (np.ndarray): Vector of Feasible Directions
        a_mat (np.ndarray): Matrix of Constraint Gradients
        b_vec (float): Vector of Constraint Locations

    Return:
        None
    """
    for a_vec, b_sca in zip(a_mat, b_vec):
        if a_vec @ x_vec.T < b_sca:
            raise ValueError(f"The point {x_vec} not feasible with ai:{a_vec} and bi: {b_sca}")
    return None


def ratio_test(
    xk: np.ndarray, p: np.ndarray, A: np.ndarray, b: np.ndarray, I: np.ndarray  # noqa: E741
) -> tuple[float, int | float]:
    """Ratio Test

    Used for inactive inequality constraints.

    Args:
        xk (np.ndarray): Feasible Point
        p (np.ndarray): Feasible Direction
        A (np.ndarray): Inactive Constraint Gradients
        b (np.ndarray): Inactive Constraint Locations
        I (np.ndarray): Inactive Constraint Index

    Return:
        alpha: (float): Distance Point can move
        idx (int): Index Corresponds to Inequality from Inactive to Active
    """
    feasible_point(xk, A, b)  # check all the points are feasible first

    fdist = A @ xk - b
    fangl = A @ p

    step_sizes = np.where(fangl < 0, fdist / -fangl, np.inf)

    alpha = np.nanmin(step_sizes)

    if np.isinf(alpha).all():  # if everything is infinite, return nan
        return np.inf, np.nan

    idx = I[np.nanargmin(step_sizes)]
    return alpha, idx  # return minimum step size
