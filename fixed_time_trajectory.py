import numpy as np

from solve_quadratic import solve_quadratic

def fixed_time_trajectory(start_pos, end_pos, start_vel, end_vel, vmax, T, a_threshold):
    assert vmax > 1e-6, "vmax must be > 1e-6 for numerical stability."
    assert T > 1e-6, "T must be > 1e-6 for numerical stability."

    p1, p2, v1, v2 = start_pos, end_pos, start_vel, end_vel

    a1 = max(solve_quadratic(a=T**2, b=(2 * T * (v1 + v2) - 4 * (p2 - p1)), c=(v2 - v1)**2))
    a2 = -a1
    sigma = np.sign(a1) if a1 != 0 else 1
    ta1_candidate = 0.5 * ((v2 - v1) / a1 + T)
    if abs(v1 + ta1_candidate * a1) <= vmax:
        # Two-phase trajectory: acceleration then deceleration.
        traj_type = "P+P-" if sigma > 0 else "P-P+"
        ta1 = ta1_candidate
        tv = None
        ta2 = T - ta1

        # Precompute position at the end of the acceleration phase.
        p_acc_end = p1 + v1 * ta1 + 0.5 * a1 * ta1**2
        p_const_end = None
    else:
        # Three-phase trajectory: acceleration, constant velocity, deceleration.
        traj_type = "P+L+P-" if sigma > 0 else "P-L-P+"
        vlimit = sigma * vmax
        a1 = ((vlimit - v1)**2 + (vlimit - v2)**2) / (2 * (vlimit * T - (p2 - p1)))
        a2 = -a1
        