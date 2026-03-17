"""
Reduced Newton Method for Picking Approximate Power Fluid Rates at each well
"""

from typing import Callable

import numpy as np


# need to make this functions negative so it works?
def well_func(qp: float, c1: float, c2: float, c3: float):
    return -c1 + c2 * np.exp(-qp * c3)


def well_first_derv(qp: float, c2: float, c3: float):
    return -c2 * c3 * np.exp(-qp * c3)


def well_second_derv(qp: float, c2: float, c3: float):
    return c2 * c3**2 * np.exp(-qp * c3)


def initial_powerfluid_alloc(well_dict: dict, Qp_tot: float) -> np.ndarray:
    """Initial Power Fluid Allocation

    Create a feasible vector of power fluid flowrates that can
    be assigned to each well and used to kick off the optimization
    scheme. The scheme will look at the total surface pump capacity available
    and the maximum powerfluid allowable for each well. It will decide which
    constraints are more binding initially and create a feasible starting point.

    Args:
        well_dict (dict): Well Dictionary of Definied Parameters
        Qp_tot (float): Total Available Power Fluid to Split out

    Return:
        Qp (np.array): Array of gradients for each well"""

    Qpf_max = []
    for well_name, well_params in well_dict.items():
        Qpf_max.append(well_params["qpf_max"])
    Qpf_max_tot = sum(Qpf_max)  # individual max contraints added up together
    Qpf_max = np.asarray(Qpf_max)

    if Qpf_max_tot <= Qp_tot:  # if the individual max power fluid constraints are less than  surface pump capacity
        Qp = Qpf_max
    else:  # use the surface pump capacity as the feasible active constraints
        num_wells = len(well_dict)
        Qp = np.full(num_wells, Qp_tot / num_wells)

        residual = Qpf_max - Qp  # make sure none of the evenly split values exceed an individual value
        while np.any(residual) < 0:
            neg_res = residual[residual >= 0] = 0  # return values where even split exceeds individual maximum

    num_wells = len(well_dict)
    Qp = np.full(num_wells, Qp_tot / num_wells)
    return Qp


def constraint_spaces(well_dict: dict, Qp_tot: float) -> tuple[np.ndarray, np.ndarray]:
    """Create Constraint Spaces

    Create a matrix A and vector b that correspond constraints of the form
    Ax >= b. All constraints are created here. Later the ACTIVE constraints are filtered out.
    Individual well constraints will be placed on top with the total water constraint on bottom.

    Args:
        well_dict (dict): Well Dictionary of Definied Parameters
        Qp_tot (float): Total Available Power Fluid to Split out

    Return:
        A (np.ndarray): Constraint Matrix
        b (np.ndarray): Constraint Vector
    """
    # constraints for each well to be non-negative
    n = len(well_dict)  # number of wells
    A_min = np.eye(n)  # identity matrix to handle min powerfluid constraints
    b_min = []
    A_max = -1 * A_min.copy()
    b_max = []
    for well_name, well_params in well_dict.items():
        b_min.append(well_params["qpf_min"])  # store the minimum water for each well
        b_max.append(-1 * well_params["qpf_max"])  # store the maximum water for each well

    # negative because sign is flipped, not sure why it needs to be unit vector?
    ai_tot = np.full(n, -1)
    bi_tot = -Qp_tot

    A = np.vstack((A_min, ai_tot, A_max))
    b = [*b_min, bi_tot, *b_max]

    return A, np.array(b)


def constraint_active(A: np.ndarray, b: np.ndarray, xk: np.ndarray, tol: float = 1e-3) -> np.ndarray:
    """Constraint Type

    Calculate Ax - b = 0 to identify which constraints are initially active vs inactive.
    Return a boolean vector where active is True, and Inactive if False

    Args:
        A (np.ndarray): Constraint Matrix
        b (np.ndarray): Constraint Vector
        xk (np.ndarray): Feasible Point

    Return:
        cat (np.ndarray): Category True is Active, False is Inactive
    """
    residuals = A @ xk - b
    return np.abs(residuals) <= tol


def qr_split(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """QR Factorization Split

    Perform a QR factorization split on A^T and required splits.
    The splits are of the form Q1, Q2 and R1. These splits are then
    used to generate the right inverse matrix and null space.

    Args:
        A (np.ndarray): Active Linear Constraints

    Returns:
        Z (np.ndarray): Null Space of Active Constraints
        Ar (np.ndarray): Right Inverse of Active Constraints
    """
    Q, R = np.linalg.qr(A.T, mode="complete")

    m, n = A.shape
    r = np.linalg.matrix_rank(A)  # matrix rank

    Q1 = Q[:, :r]
    Q2 = Q[:, r:]
    R1 = R[:r, :]

    Z = Q2
    Ar = Q1 @ np.linalg.inv(R1).T

    return Z, Ar


def update_objective(well_dict: dict, Qp: np.ndarray) -> float:
    """Update Objective Function

    Compute the objective function value at a specific point Qp. This
    is the total oil rate of all the wells with defined oil rate.

    Args:
        well_dict (dict): Well Dictionary of Definied Parameters
        Qp (np.array): Array of power fluid for each to assess

    Return:
        obj_val (np.array): Array of gradients for each well
    """
    obj_list = []
    for idx, (well_name, well_params) in enumerate(well_dict.items()):
        qp = Qp[idx]
        c1 = well_params["c1"]
        c2 = well_params["c2"]
        c3 = well_params["c3"]
        obj_list.append(well_func(qp, c1, c2, c3))
    return sum(obj_list)


def update_gradient(well_dict: dict, Qp: np.ndarray) -> np.ndarray:
    """Update Gradient

    Ccreate the gradient of the objective function with various
    power fluid flow rates.

    Args:
        well_dict (dict): Well Dictionary of Definied Parameters
        Qp (np.array): Array of power fluid for each to assess

    Return:
        dfk (np.array): Array of gradients for each well
    """
    num_wells = len(well_dict)
    if len(Qp) != num_wells:
        raise ValueError(f"Qp length: {len(Qp)} does not match the number of wells: {num_wells}")
    dfk = np.zeros(num_wells)

    # Loop through each well and calculate the gradient
    for idx, (well_name, well_params) in enumerate(well_dict.items()):
        qp = Qp[idx]
        c2 = well_params["c2"]
        c3 = well_params["c3"]
        # Compute the gradient using the derivative function
        dfk[idx] = well_first_derv(qp, c2, c3)
    return dfk


def update_hessian(well_dict: dict, Qp: np.ndarray) -> np.ndarray:
    """Update Hessian

    Create the Hessian of the objective function with various
    power fluid flow rates.

    Args:
        well_dict (dict): Well Dictionary of Definied Parameters
        Qp (np.array): Array of power fluid for each to assess

    Return:
        Hk (np.array): Array of Hessian for each well
    """
    num_wells = len(well_dict)
    if len(Qp) != num_wells:
        raise ValueError(f"Qp length: {len(Qp)} does not match the number of wells: {num_wells}")
    Hk = np.eye(num_wells)

    # Loop through each well and calculate the second derivative
    for idx, (well_name, well_params) in enumerate(well_dict.items()):
        qp = Qp[idx]
        c2 = well_params["c2"]
        c3 = well_params["c3"]
        # Compute the hessian using the second derivative function
        Hk[idx, idx] = well_second_derv(qp, c2, c3)  # update
    return Hk


def optimality_test(
    dfk: np.ndarray, Z: np.ndarray, Ar: np.ndarray, active: np.ndarray, tol: float = 1e-3
) -> tuple[bool, np.ndarray, bool]:
    """Test for Optimality

    Look at the gradient at the location, null space, right inverse and active constraints.
    Make a decision whether you are optimal or not. The index of active constraints can be
    updated if a constraint needs to be dropped to allow further reduction of obj. func.

    Args:
        dfk (np.ndarray): Gradient of f(xk)
        Z (np.ndarray): Null Space of Active Constraints
        Ar (np.ndarray): Right Inverse of Active Constraints
        active (np.ndarray): Boolean Array of Active Constraints
        tol (float): Stopping Criteria

    Return:
        optimal (bool): True or False, whether optimal
        active (np.ndarray): Boolean array of active constraints
        con_update (np.ndarray): Was active constraints updated
    """

    if np.any(active):  # constraings are active
        if np.linalg.norm(Z.T @ dfk) < tol:  # at an optimal point, on the constraints
            Lags = Ar.T @ dfk  # lagrangian multiples
            if np.all(Lags >= 0):  # see if all the lagrangians are greater than or equal to zero
                return True, active, False
            else:
                active_idx = np.where(active)[0]  # update which constraints are active vs inactive
                min_lag_idx = np.argmin(Lags)
                active_min_lag = active_idx[min_lag_idx]  # find minimum active lagrange variable
                active[active_min_lag] = False  # swap from True to False
                return False, active, True
        else:
            return False, active, False
    else:  # no constraints are active
        if np.linalg.norm(dfk) < tol:
            return True, active, False
        else:
            return False, active, False


def newton_reduced(dfk: np.ndarray, Hfk: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Newton Reduced Search Direction

    Calculate the reduced Newton Search Direction. Formula is on Page 550 of the text.
    This formula kind of sucks because it can destroy sparsity of Hfk matrix.

    Args:
        dfk (np.ndarray): Gradient of Function at xk
        Hfk (np.ndarray): Hessian of Function at xk
        Z (np.ndarray): Null Space of A Constraints

    Return:
        p (np.ndarray): Search Direction Vector using Newton Method
    """
    p = -Z @ np.linalg.inv(Z.T @ Hfk @ Z) @ Z.T @ dfk
    return p


def newton_projected(dfk: np.ndarray, Hfk: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Newton Projected Search Direction

    Calculate the projected Newton Search Direction. Formula is on Page 555 of the text.
    Assumes Hfk is positive definite. Maintains Sparsity of Hfk in finding inverse.

    Args:
        dfk (np.ndarray): Gradient of Function at xk
        Hfk (np.ndarray): Hessian of Function at xk
        Z (np.ndarray): Null Space of A Constraints

    Return:
        p (np.ndarray): Search Direction Vector using Newton Method
    """
    Hfk_inv = np.linalg.inv(Hfk)
    p = -(Hfk_inv - Hfk_inv @ A.T @ np.linalg.inv(A @ Hfk_inv @ A.T) @ A @ Hfk_inv) @ dfk
    return p


def line_search_backtrack(
    obj_func: Callable,
    well_dict: dict,
    xk: np.ndarray,
    dfk: np.ndarray,
    p: np.ndarray,
    alpha: float = 1.0,
    rho: float = 0.5,
    mu: float = 1e-4,
) -> float:
    """Line Search Backtracking

    Calculates a value of Alpha that Guarentees that the value of the objective function
    actually decreases with the appropriate search direction alpha. Uses simple backtrack.

    Args:
        obj_func (Callable): Value of Objective Function
        well_dict (dict): Well Properties
        xk (np.ndarray): Currently Point
        dfk (np.ndarray): Gradient at Point
        p (np.ndarray): Search Direction
        alpha (float): Initial Guess of Alpha
        rho (float): How much to reduce alpha
        mu (float): Armijo Condition Constant

    Return:
        alpha (float): Step Size
    """
    while obj_func(well_dict, xk + alpha * p) > obj_func(well_dict, xk) + mu * alpha * p.T @ dfk:
        alpha = alpha * rho
    return alpha
