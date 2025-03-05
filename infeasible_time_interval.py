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
    if np.sign(v1) == 0 or np.sign(v2) == 0:
        # The goal state I does not exist in the first case, and the goal state is on the boundary on the second case.
        return None
    if np.sign(v1) * (p2 - p1) < 0:
        # The goal state is in region IV or region V.
        return None
    if np.sign(v1) * np.sign(v2) < 0:
        # The goal state is in region III or region IV.
        return None
    delta_t = max(solve_quadratic(a=0.5 * amax, b=abs(v1), c=-abs(p2 - p1)))
    upper_vel_bound = min(abs(v1) + amax * delta_t, vmax)
    if abs(v2) > upper_vel_bound: 
        # The goal state is in region V.
        return None
    brake_p = v1**2 / (2 * amax)
    if brake_p >= abs(p2 - p1):
        delta_t = min(solve_quadratic(a=-0.5 * amax, b=abs(v1), c=-abs(p2 - p1)))
        lower_vel_bound = abs(v1) - amax * delta_t
        if abs(v2) < lower_vel_bound: 
            # The goal state is in region V.
            return None
    else:
        delta_t = np.sqrt(2 * (abs(p2 - p1) - brake_p) / amax)
        lower_vel_bound = amax * delta_t
        if abs(v2) <= lower_vel_bound: 
            # The goal state is in region II.
            return None

    ##### Then calculate the infeasible time interval. #####
    delta_pacc = 0.5 * (v1 + v2) * abs(v2 - v1) / amax
    sigma = np.sign(p2 - p1 - delta_pacc) if np.sign(p2 - p1 - delta_pacc) != 0 else np.sign(v1)
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

    start_pos, start_vel = np.array([np.random.uniform(0, 1)]), np.array([np.random.uniform(0, 1)])
    end_pos, end_vel = np.array([np.random.uniform(0, 1)]), np.array([np.random.uniform(0, 1)])
    vmax, amax = np.array([np.random.uniform(1, 2)]), np.array([np.random.uniform(1, 2)])

    # # Examples
    vmax, amax = np.array([1]), np.array([1])
    # # Examples set 1
    # start_pos, start_vel = np.array([0]), np.array([0.5])
    # # Example 1.1: region I.
    # end_pos, end_vel = np.array([0.125]), np.array([0.5])
    # # Example 1.2: region II.
    # end_pos, end_vel = np.array([0.3]), np.array([0.5])
    # # Example 1.3: region III.
    # end_pos, end_vel = np.array([0.2]), np.array([-0.5])
    # # Example 1.4: region IV.
    # end_pos, end_vel = np.array([0]), np.array([-0.5])
    # # Example 1.5: region V.
    # end_pos, end_vel = np.array([0]), np.array([0.25])
    # end_pos, end_vel = np.array([0]), np.array([0.75])
    # end_pos, end_vel = np.array([-0.125]), np.array([0.5])

    # # Examples set 2
    # start_pos, start_vel = np.array([0]), np.array([-0.5])
    # # Example 2.1: region I.
    # end_pos, end_vel = np.array([-0.125]), np.array([-0.5])
    # # Example 2.2: region II.
    # end_pos, end_vel = np.array([-0.3]), np.array([-0.5])
    # # Example 2.3: region III.
    # end_pos, end_vel = np.array([-0.2]), np.array([0.5])
    # # Example 2.4: region IV.
    # end_pos, end_vel = np.array([0]), np.array([0.5])
    # # Example 2.5: region V.
    # end_pos, end_vel = np.array([0]), np.array([-0.25])
    # end_pos, end_vel = np.array([0]), np.array([-0.75])
    # end_pos, end_vel = np.array([0.125]), np.array([-0.5])

    # # Example set 3
    # start_pos, start_vel = np.array([0]), np.array([1.0])
    # # Example 3.1: region I.
    # end_pos, end_vel = np.array([0.5]), np.array([1.0])
    # # Example 3.2: region II.
    # end_pos, end_vel = np.array([1.2]), np.array([1.0])
    # # Example 3.3: region III.
    # end_pos, end_vel = np.array([0.5]), np.array([-1.0])
    # # Example 3.4: region IV.
    # end_pos, end_vel = np.array([0]), np.array([-1.0])
    # # Example 3.5: region V.
    # end_pos, end_vel = np.array([0]), np.array([0.8])
    # end_pos, end_vel = np.array([-0.5]), np.array([1.0])

    # # Example set 4
    # start_pos, start_vel = np.array([0]), np.array([0.5])
    # # Example 4.1
    # end_pos, end_vel = np.array([0]), np.array([0.5])   # This is kind of wired.
    # # Example 4.2
    # end_pos, end_vel = np.array([0.125]), np.array([0])
    # # Example 4.3
    # end_pos, end_vel = np.array([0.25]), np.array([0.5])
    # # Example 4.4
    # end_pos, end_vel = np.array([0.15625]), np.array([0.75])
    # # Example 4.5
    # end_pos, end_vel = np.array([0.09375]), np.array([0.25]) 
    # # Example 4.6
    # end_pos, end_vel = np.array([0.5]), np.array([1.0]) 

    # Calculate the infeasible time interval
    interval_info = infeasible_time_interval(start_pos, end_pos, start_vel, end_vel, vmax, amax)