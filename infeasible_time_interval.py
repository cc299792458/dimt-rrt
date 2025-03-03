import numpy as np
from solve_quadratic import solve_quadratic

def infeasible_time_interval(start_pos, end_pos, start_vel, end_vel, vmax, amax):
    """
    Compute the infeasible time interval for a trajectory between two states.
    """
    assert vmax > 1e-6, "vmax must be > 1e-6 for numerical stability."
    assert amax > 1e-6, "amax must be > 1e-6 for numerical stability."

    p1, p2, v1, v2 = start_pos, end_pos, start_vel, end_vel

    ##### First ditermine if the goal state is in region I. If not, return None. #####
    sign_v1, norm_v1 = np.sign(v1), abs(v1)
    if sign_v1 == 0:
        raise NotImplementedError("Zero initial velocity not supported.")
    if sign_v1 * (p2 - p1) < 0:
        # The goal state is in region IV or region V.
        return None
    delta_t = max(solve_quadratic(a=0.5 * amax, b=norm_v1, c=-abs(p2 - p1)))
    upper_vel_bound = min(norm_v1 + amax * delta_t, vmax)
    if v2 > upper_vel_bound: 
        # The goal state is in region V.
        return None
    brake_p = v1**2 / amax
    if brake_p > abs(p2 - p1):
        delta_t = min(solve_quadratic(a=-0.5 * amax, b=norm_v1, c=-abs(p2 - p1)))
        lower_vel_bound = norm_v1 - amax * delta_t
    else:
        delta_t = np.sqrt(2 * (abs(p2 - p1) - brake_p) / amax)
        lower_vel_bound = min(0.5 * amax * delta_t**2, vmax)
    if v2 < lower_vel_bound: 
        # The goal state is in region II, III, IV, or V.
        return None

    ##### Then calculate the infeasible time interval. #####
    delta_pacc = 0.5 * (v1 + v2) * abs(v2 - v1) / amax
    sigma = np.sign(p2 - p1 - delta_pacc) if np.sign(p2 - p1 - delta_pacc) != 0 else 1
    a1, a2, vlimit= -sigma * amax, sigma * amax, -sigma * vmax
    
    t1, t2 = solve_quadratic(a=a1, b=2*v1, c=(v2**2 - v1**2) / (2*a2) - (p2 - p1))
    t1, t2 = (min(t1, t2), max(t1, t2)) if t1 is not None and t2 is not None else (None, None)

    tlower = 2 * t1 + (v2 - v1) / a2
    if abs(v1 + a1 * t2) > vmax:
        tupper = ((vlimit - v1) / a1 +
                  ((v1**2 + v2**2 - 2 * vlimit**2) / (2 * vlimit * a1) + (p2 - p1) / vlimit) +
                  (v2 - vlimit) / a2)
    else:
        tupper = 2 * t2 + (v2 - v1) / a2

    return {"tlower": tlower, "tupper": tupper}

# ------------------ Testing and Plotting Code ------------------
if __name__ == '__main__':
    np.random.seed(42)  # For reproducibility
    
    start_pos, end_pos = np.array([0]), np.array([5])
    start_vel, end_vel = np.array([10]), np.array([10])
    vmax, amax = np.array([10]), np.array([5])

    # Calculate the infeasible time interval
    interval_info = infeasible_time_interval(start_pos, end_pos, start_vel, end_vel, vmax, amax)
    print(interval_info)