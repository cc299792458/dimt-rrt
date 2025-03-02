import numpy as np

from solve_quadratic import solve_quadratic

def one_dof_minimum_time(start_pos, end_pos, start_vel, end_vel, vmax, amax):
    assert vmax > 1e-6, "vmax must be greater than 1e-6 for numerical stability."
    assert amax > 1e-6, "amax must be greater than 1e-6 for numerical stability."

    p1, p2, v1, v2 = start_pos, end_pos, start_vel, end_vel

    delta_pacc = 0.5 * (v1 + v2) * abs(v2 - v1) / amax
    sigma = np.sign(p2 - p1- delta_pacc)
    a1, a2, vlimit = sigma * amax, -sigma * amax, sigma * vmax

    ta1 = max(solve_quadratic(a=a1, b=2*v1, c=(v2**2-v1**2)/(2*a2)-(p2-p1)))
    if abs(v1 + ta1 * a1) <= vmax:
        traj_type = "P+P-" if a1 > 0 else "P-P+"
        ta2 = (v2 - v1) / a2 + ta1
        T = ta1 + ta2
    else:
        traj_type = "P+L+P-" if a1 > 0 else "P-L-P+"
        ta1 = (vlimit - v1) / a1
        tv = (v1 ** 2 + v2 ** 2 - 2 * vlimit ** 2) / (2 * vlimit * a1) + (p2 - p1) / vlimit
        ta2 = (v2 - vlimit) / a2
        T = ta1 + tv + ta2