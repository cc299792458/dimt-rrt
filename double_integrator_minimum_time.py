import numpy as np

from is_segment_feasible import is_segment_feasible
from one_dof_minimum_time import one_dof_minimum_time
from fixed_time_trajectory import fixed_time_trajectory
from infeasible_time_interval import infeasible_time_interval

def compute_traj_segment(start_state, end_state, vmax, amax, collision_checker, bounds, n_dim):
    traj_time = compute_traj_time(start_state, end_state, vmax, amax, n_dim)
    traj_infos = compute_traj_infos(start_state, end_state, traj_time, vmax, amax, n_dim)

    traj_feasibility = is_segment_feasible(traj_time=traj_time, traj_infos=traj_infos,
                               collision_checker=collision_checker, bounds=bounds, n_dim=n_dim)

    return traj_time, traj_infos, traj_feasibility

def compute_traj_time(start_state, end_state, vmax, amax, n_dim):
    """
    Calculate the maximum time required to traverse a segment across all dimensions,
    considering vmax and amax constraints.
    """
    t_requireds = []
    for dim in range(n_dim):
        traj_info = one_dof_minimum_time(
            start_pos=start_state[0][dim],
            end_pos=end_state[0][dim],
            start_vel=start_state[1][dim],
            end_vel=end_state[1][dim],
            vmax=vmax[dim],
            amax=amax[dim]
        )
        t_requireds.append(traj_info['T'])
    traj_time = max(t_requireds)
    for dim in range(n_dim):
        time_interval = infeasible_time_interval(
            start_pos=start_state[0][dim],
            end_pos=end_state[0][dim],
            start_vel=start_state[1][dim],
            end_vel=end_state[1][dim],
            vmax=vmax[dim],
            amax=amax[dim]
        )
        if time_interval is not None:
            if traj_time >= time_interval['tlower'] and traj_time <= time_interval['tupper']:
                traj_time = time_interval['tupper'] * 1.001 # NOTE: to handle precision issue.

    return traj_time

def compute_traj_infos(start_state, end_state, traj_time, vmax, amax, n_dim):
    """
    Calculate the traj for a single segment using minimum acceleration interpolants.
    """
    # Vectorized calculation for all dimensions
    traj_infos = []
    for dim in range(n_dim):
        trajectories, optimal_label = fixed_time_trajectory(
            start_pos=start_state[0][dim],
            end_pos=end_state[0][dim],
            start_vel=start_state[1][dim],
            end_vel=end_state[1][dim],
            vmax=vmax[dim],
            T=traj_time,
            a_threshold=amax[dim]
        )
        if trajectories is None:
            raise NotImplementedError
            # NOTE: This is actually possible. See consistency_validation.py for an example.
            return None
        traj_infos.append((trajectories[optimal_label], optimal_label))

    return np.array(traj_infos, dtype=object)
