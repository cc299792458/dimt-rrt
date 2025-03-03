import numpy as np
import matplotlib.pyplot as plt

from solve_quadratic import solve_quadratic
from get_motion_state import get_motion_state_at_local_t

def fixed_time_trajectory(start_pos, end_pos, start_vel, end_vel, vmax, T, a_threshold):
    assert vmax > 1e-6, "vmax must be > 1e-6 for numerical stability."
    assert T > 1e-6, "T must be > 1e-6 for numerical stability."
    p1, p2, v1, v2 = start_pos, end_pos, start_vel, end_vel

    if np.isclose((v1 * T - (p2 - p1)), 0, 1e-12):
        if (np.isclose(abs(v1), vmax, 1e-12) and np.isclose(abs(v2), vmax, 1e-12)):
            a1 = a2 = ta1 = ta2 = np.zeros([1])
            tv, p_acc_end, p_const_end, vlimit, vswitch, traj_type = T, np.zeros([1]), p2, v1, None, ("P+L+P-" if v1 >=0 else "P-L-P+")
        else:
            return None
    else:
        a1, _ = solve_quadratic(a=T**2, b=(2 * T * (v1 + v2) - 4 * (p2 - p1)), c=-(v2 - v1)**2)
        sigma = np.sign(a1) if a1 != 0 else 1
        ta1_candidate = 0.5 * ((v2 - v1) / a1 + T)
        if abs(v1 + ta1_candidate * a1) <= vmax:
            traj_type = "P+P-" if sigma > 0 else "P-P+"
            vlimit = sigma * vmax
            if abs(a1) * 0.999 > a_threshold: return None
            a1 = sigma * np.clip(abs(a1), 0, a_threshold)
            a2 = -a1
            ta1 = ta1_candidate
            tv = None
            ta2 = T - ta1
            assert ta1 >= 0 and ta2 >= 0
            p_acc_end = p1 + v1 * ta1 + 0.5 * a1 * ta1**2
            p_const_end = None
            vswitch = v1 + a1 * ta1
        else:
            traj_type = "P+L+P-" if sigma > 0 else "P-L-P+"
            vlimit = sigma * vmax
            a1 = ((vlimit - v1)**2 + (vlimit - v2)**2) / (2 * (vlimit * T - (p2 - p1)))
            if abs(a1) * 0.999 > a_threshold: return None
            a1 = sigma * np.clip(abs(a1), 0, a_threshold)
            a2 = -a1
            ta1 = (vlimit - v1) / a1
            ta2 = (v2 - vlimit) / a2
            tv = T - ta1 - ta2
            if ta1 < 0 or ta2 < 0 or tv < 0: return None
            p_acc_end = p1 + v1 * ta1 + 0.5 * a1 * ta1**2
            p_const_end = p_acc_end + vlimit * tv
            vswitch = None
        
    return {"T": T, "ta1": ta1, "tv": tv, "ta2": ta2, "p1": p1, "p2": p2, "p_acc_end": p_acc_end, "p_const_end": p_const_end, "v1": v1, "v2": v2, "vlimit": vlimit, "vswitch": vswitch, "a1": a1, "a2": a2, "traj_type": traj_type}

def plot_trajectory(traj_info, num_points=1000):
    """
    Plot position, velocity, and acceleration vs. time using the 
    get_motion_state_at_local_t function based on traj_info.

    Parameters:
        traj_info (dict): Dictionary from one_dof_minimum_time containing keys:
            "T", "ta1", "tv", "ta2", "p1", "p2", "v1", "v2", 
            "vlimit", "vswitch", "a1", "a2", "traj_type".
        num_points (int): Number of time samples for the plot.
    """
    # Get total time and create time array.
    T = traj_info["T"]
    t_vals = np.linspace(0, T, num_points)
    
    # Compute motion state at each time sample.
    pos_vals = np.array([get_motion_state_at_local_t(traj_info, t)[0] for t in t_vals])
    vel_vals = np.array([get_motion_state_at_local_t(traj_info, t)[1] for t in t_vals])
    acc_vals = np.array([get_motion_state_at_local_t(traj_info, t)[2] for t in t_vals])
    
    # Create subplots for position, velocity, and acceleration.
    fig, axs = plt.subplots(3, 1, figsize=(6, 6), sharex=True)
    
    # Plot Acceleration vs. Time.
    axs[0].plot(t_vals, acc_vals, label="Acceleration")
    axs[0].set_ylabel("Acceleration")
    axs[0].legend(loc='upper right')

    # Plot Velocity vs. Time.
    axs[1].plot(t_vals, vel_vals, label="Velocity")
    axs[1].set_ylabel("Velocity")
    axs[1].legend(loc='upper right')

    # Plot Position vs. Time.
    axs[2].plot(t_vals, pos_vals, label="Position")
    axs[2].set_xlabel("Time")
    axs[2].set_ylabel("Position")
    axs[2].legend(loc='upper right')

    # Set overall title with trajectory type.
    plt.suptitle(f"Min Accel: {abs(list(traj_info['a1'])[0]):.3f} m/s^2\nTrajectory type: {traj_info['traj_type']}", fontsize=14)
    plt.tight_layout()
    plt.show()

# ------------------ Testing and Main Code ------------------
if __name__ == '__main__':
    np.random.seed(42)
    
    a_threshold = np.array([100.0])

    # Generate random boundary conditions.
    start_pos = np.random.uniform(-10, 10)
    end_pos = np.random.uniform(-10, 10)
    start_vel = np.random.uniform(-2, 2)
    end_vel = np.random.uniform(-2, 2)
    vmax = np.random.uniform(2, 4)
    T = np.random.uniform(1, 10)

    # Examples
    # Examples 1, 2, 3, 4: Corresponding to Figure 5 in the original paper
    vmax = np.array([1.0])
    # start_pos, end_pos, start_vel, end_vel, T = np.array([0.0]), np.array([1.0]), np.array([0.0]), np.array([0.0]), np.array([2.0])
    # start_pos, end_pos, start_vel, end_vel, T = np.array([0.0]), np.array([3.0]), np.array([0.0]), np.array([0.0]), np.array([4.0])
    # start_pos, end_pos, start_vel, end_vel, T = np.array([0.0]), np.array([0.0]), np.array([1.0]), np.array([0.0]), np.array([2.41])
    # start_pos, end_pos, start_vel, end_vel, T = np.array([0.0]), np.array([-0.5]), np.array([1.0]), np.array([1.0]), np.array([4.5])
    
    # More examples
    # Example 5: This example illustrates that for the P-L+P+ trajectory, just before accelerating with amax, 
    # the velocity must have reached -vmax
    start_pos, end_pos, start_vel, end_vel, T = np.array([0.0]), np.array([0.5]), np.array([0.0]), np.array([1.0]), np.array([1.0]) 

    # Example 6, 7, 8, 9:
    # Examples 6 and 9 demonstrate a scenario where, if the distance is insufficient for acceleration, 
    # it must first decelerate backward before accelerating forward
    # start_pos, end_pos, start_vel, end_vel, T = np.array([0.0]), np.array([0.4]), np.array([0.0]), np.array([1.0]), np.array([1.63])
    # start_pos, end_pos, start_vel, end_vel, T = np.array([0.0]), np.array([0.6]), np.array([0.0]), np.array([1.0]), np.array([1.1]) 
    # start_pos, end_pos, start_vel, end_vel, T = np.array([0.0]), np.array([0.5]), np.array([0.0]), np.array([0.8]), np.array([1.01])
    # start_pos, end_pos, start_vel, end_vel, T = np.array([0.0]), np.array([0.5]), np.array([0.0]), np.array([1.2]), np.array([2.14])

    # Example 10:
    # This is a corner case.
    # start_pos, end_pos, start_vel, end_vel, T = np.array([0.0]), np.array([1.0]), np.array([1.0]), np.array([1.0]), np.array([1.0])

    # Compute trajectory information that corresponding to the minimal acceleration trajectory.
    traj_info = fixed_time_trajectory(start_pos, end_pos, start_vel, end_vel, vmax, T, a_threshold)
    if traj_info is not None:
        plot_trajectory(traj_info=traj_info)