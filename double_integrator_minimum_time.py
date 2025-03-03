import numpy as np
from tqdm import tqdm

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
        traj_info = fixed_time_trajectory(
            start_pos=start_state[0][dim],
            end_pos=end_state[0][dim],
            start_vel=start_state[1][dim],
            end_vel=end_state[1][dim],
            vmax=vmax[dim],
            T=traj_time,
            a_threshold=amax[dim]
        )
        if traj_info is None:
            raise NotImplementedError
        traj_infos.append(traj_info)

    return np.array(traj_infos, dtype=object)

# ------------------ Testing and Main Code ------------------
if __name__ == '__main__':
    np.random.seed(42)  # For reproducibility

    # Default collision checker: returns True for all states
    def default_collision_checker(state):
        # state is in the form (position, velocity)
        return True
    n_dim = 7
    bounds = np.tile(np.array([[-10, 10]]).T, (1, n_dim))

    for i in tqdm(range(10000)):
        # Generate random boundary conditions.
        start_pos = np.random.uniform(-10, 10, n_dim)
        end_pos = np.random.uniform(-10, 10, n_dim)
        start_vel = np.random.uniform(-2, 2, n_dim)
        end_vel = np.random.uniform(-2, 2, n_dim)
        vmax = np.random.uniform(2, 4, n_dim)
        amax = np.random.uniform(2, 4, n_dim)

        start_state, end_state = np.array([start_pos, start_vel]), np.array([end_pos, end_vel])

        traj_infos = compute_traj_segment(start_state=start_state, end_state=end_state, vmax=vmax, amax=amax, 
                                        collision_checker=default_collision_checker, bounds=bounds, n_dim=n_dim)