import numpy as np
from solve_quadratic import solve_quadratic

def infeasible_time_interval(start_pos, end_pos, start_vel, end_vel, vmax, amax):
    assert vmax > 1e-6, "vmax must be > 1e-6 for numerical stability."
    assert amax > 1e-6, "amax must be > 1e-6 for numerical stability."

    p1, p2, v1, v2 = start_pos, end_pos, start_vel, end_vel

    # Determine motion direction.
    delta_pacc = 0.5 * (v1 + v2) * abs(v2 - v1) / amax
    sigma = np.sign(p2 - p1 - delta_pacc) if np.sign(p2 - p1 - delta_pacc) != 0 else 1
    a1, a2, vlimit= -sigma * amax, sigma * amax, -sigma * vmax
    
    t1, t2 = solve_quadratic(a=a1, b=2*v1, c=(v2**2 - v1**2) / (2*a2) - (p2 - p1))
    tlower, tupper = (min(t1, t2), max(t1, t2)) if t1 is not None and t2 is not None else (None, None)

    return {"tlower": tlower, "tupper": tupper}

# ------------------ Testing and Plotting Code ------------------
if __name__ == '__main__':
    np.random.seed(42)  # For reproducibility
    
    # Sample random boundary conditions
    start_pos = np.array([np.random.uniform(-10, 10)])
    end_pos = np.array([np.random.uniform(-10, 10)])
    start_vel = np.array([np.random.uniform(-2, 2)])
    end_vel = np.array([np.random.uniform(-2, 2)])
    vmax = np.array([np.random.uniform(2, 4)])
    amax = np.array([np.random.uniform(1, 3)])

    # Examples
    vmax, amax = np.array([1.0]), np.array([1.0])
    # Examples 1, 2, 3, 4: Corresponding to Figure 5 in the original paper
    # start_pos, end_pos, start_vel, end_vel = np.array([0.0]), np.array([1.0]), np.array([0.0]), np.array([0.0])
    # start_pos, end_pos, start_vel, end_vel = np.array([0.0]), np.array([3.0]), np.array([0.0]), np.array([0.0]) 
    # start_pos, end_pos, start_vel, end_vel = np.array([0.0]), np.array([0.0]), np.array([1.0]), np.array([0.0]) 
    # start_pos, end_pos, start_vel, end_vel = np.array([0.0]), np.array([-0.5]), np.array([1.0]), np.array([1.0]) 
    
    # More examples
    # Example 5: This example illustrates that for the P-L+P+ trajectory, just before accelerating with amax, 
    # the velocity must have reached -vmax
    # start_pos, end_pos, start_vel, end_vel = np.array([0.0]), np.array([0.5]), np.array([0.0]), np.array([1.0]) 

    # Example 6, 7, 8, 9:
    # Examples 6 and 9 demonstrate a scenario where, if the distance is insufficient for acceleration, 
    # it must first decelerate backward before accelerating forward
    # start_pos, end_pos, start_vel, end_vel = np.array([0.0]), np.array([0.4]), np.array([0.0]), np.array([1.0]) 
    # start_pos, end_pos, start_vel, end_vel = np.array([0.0]), np.array([0.6]), np.array([0.0]), np.array([1.0]) 
    # start_pos, end_pos, start_vel, end_vel = np.array([0.0]), np.array([0.5]), np.array([0.0]), np.array([0.8])
    # start_pos, end_pos, start_vel, end_vel = np.array([0.0]), np.array([0.5]), np.array([0.0]), np.array([1.2])

    # Example 10:
    # This is a corner case.
    start_pos, end_pos, start_vel, end_vel = np.array([0.0]), np.array([1.0]), np.array([1.0]), np.array([1.0]), np.array([1.0])

    # Compute candidate trajectories and select the optimal one using the previously defined function.
    traj_info = infeasible_time_interval(start_pos, end_pos, start_vel, end_vel, vmax, amax)