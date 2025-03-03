import numpy as np
from tqdm import tqdm

from one_dof_minimum_time import one_dof_minimum_time
from fixed_time_trajectory import fixed_time_trajectory

# ------------------ Testing Code ------------------
if __name__ == '__main__':
    np.random.seed(42)  # For reproducibility

    ##### 1. First, let's test the consistency between 'one_dof_minimum_time' and 'fixed_time_trajectory' #####
    for i in tqdm(range(1_000_000)):
        # Sample random boundary conditions
        start_pos = np.random.uniform(-100, 100)
        end_pos = np.random.uniform(-100, 100)
        start_vel = np.random.uniform(-10, 10)
        end_vel = np.random.uniform(-10, 10)
        vmax = np.random.uniform(10, 20)
        amax = np.random.uniform(2, 10)

        traj_info = one_dof_minimum_time(start_pos, end_pos, start_vel, end_vel, vmax, amax)
        T = traj_info["T"]

        traj_info = fixed_time_trajectory(start_pos, end_pos, start_vel, end_vel, vmax, T, amax)
        amin = abs(traj_info['a1'])

        assert np.isclose(amin, amax)
        assert amin <= amax