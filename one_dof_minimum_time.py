import numpy as np
import matplotlib.pyplot as plt

from solve_quadratic import solve_quadratic
from get_motion_state import get_motion_state_at_local_t

def one_dof_minimum_time(start_pos, end_pos, start_vel, end_vel, vmax, amax):
    """
    Compute the minimum-time trajectory for a 1-DOF motion and precompute 
    constant state values for later queries.
    
    Parameters:
        start_pos, end_pos: positions (p1, p2)
        start_vel, end_vel: velocities (v1, v2)
        vmax: maximum velocity magnitude
        amax: maximum acceleration magnitude
        
    Returns:
        dict: A dictionary containing:
            "T": total trajectory time,
            "ta1": time at the end of the acceleration phase,
            "tv": duration of the constant-velocity phase (None if not used),
            "ta2": duration of the deceleration phase,
            "p1": starting position,
            "p2": ending position,
            "p_acc_end": position at the end of acceleration phase,
            "p_const_end": position at the end of constant-velocity phase (if applicable; None otherwise),
            "v1": starting velocity,
            "v2": ending velocity,
            "vlimit": signed velocity limit,
            "a1": acceleration during the acceleration phase,
            "a2": acceleration during the deceleration phase,
            "traj_type": trajectory type string,
            "vswitch": the velocity at the end of acceleration phase (only for two-phase; None for three-phase)
    """
    assert vmax > 1e-6, "vmax must be > 1e-6 for numerical stability."
    assert amax > 1e-6, "amax must be > 1e-6 for numerical stability."

    p1, p2, v1, v2 = start_pos, end_pos, start_vel, end_vel

    # Determine motion direction.
    delta_pacc = 0.5 * (v1 + v2) * abs(v2 - v1) / amax
    sigma = np.sign(p2 - p1 - delta_pacc) if np.sign(p2 - p1 - delta_pacc) != 0 else 1
    a1, a2, vlimit = sigma * amax, -sigma * amax, sigma * vmax 

    # Candidate time for acceleration phase.
    ta1_candidate = max(solve_quadratic(a=a1, b=2*v1, c=(v2**2 - v1**2) / (2*a2) - (p2 - p1)))
    
    if abs(v1 + ta1_candidate * a1) <= vmax:
        # Two-phase trajectory: acceleration then deceleration.
        traj_type = "P+P-" if a1 > 0 else "P-P+"
        ta1 = ta1_candidate
        tv = None
        ta2 = (v2 - v1) / a2 + ta1
        T = ta1 + ta2

        # Precompute position at the end of the acceleration phase.
        p_acc_end = p1 + v1 * ta1 + 0.5 * a1 * ta1**2
        p_const_end = None
        # vswitch is the velocity at the end of the acceleration phase.
        vswitch = v1 + a1 * ta1
    else:
        # Three-phase trajectory: acceleration, constant velocity, deceleration.
        traj_type = "P+L+P-" if a1 > 0 else "P-L-P+"
        ta1 = (vlimit - v1) / a1
        tv = (v1**2 + v2**2 - 2*vlimit**2) / (2*vlimit*a1) + (p2 - p1) / vlimit
        ta2 = (v2 - vlimit) / a2
        T = ta1 + tv + ta2

        # Precompute positions at the end of acceleration and constant-velocity phases.
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
    plt.suptitle(f"Total time: {list(traj_info['T'])[0]:.3f}s\nTrajectory type: {traj_info['traj_type']}", fontsize=14)
    plt.tight_layout()
    plt.show()

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
    start_pos, end_pos, start_vel, end_vel = np.array([0.0]), np.array([0.5]), np.array([0.0]), np.array([1.0]) 

    # Example 6, 7, 8, 9:
    # Examples 6 and 9 demonstrate a scenario where, if the distance is insufficient for acceleration, 
    # it must first decelerate backward before accelerating forward
    # start_pos, end_pos, start_vel, end_vel = np.array([0.0]), np.array([0.4]), np.array([0.0]), np.array([1.0]) 
    # start_pos, end_pos, start_vel, end_vel = np.array([0.0]), np.array([0.6]), np.array([0.0]), np.array([1.0]) 
    # start_pos, end_pos, start_vel, end_vel = np.array([0.0]), np.array([0.5]), np.array([0.0]), np.array([0.8])
    # start_pos, end_pos, start_vel, end_vel = np.array([0.0]), np.array([0.5]), np.array([0.0]), np.array([1.2])

    # Compute candidate trajectories and select the optimal one using the previously defined function.
    traj_info = one_dof_minimum_time(start_pos, end_pos, start_vel, end_vel, vmax, amax)
    plot_trajectory(traj_info=traj_info)