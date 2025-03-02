import numpy as np

def get_motion_state_at_local_t(traj_info, t):
    """
    Get the motion state (position, velocity, acceleration) at time t based on the provided trajectory info.
    
    Parameters:
        traj_info (dict): Dictionary from one_dof_minimum_time with keys:
            "T", "ta1", "tv", "ta2", "p1", "p2", "p_acc_end", "p_const_end",
            "v1", "v2", "vlimit", "a1", "a2", "traj_type".
        t (float): Time at which to evaluate the state (0 <= t <= T).
        
    Returns:
        tuple: (pos, vel, acc) at time t, each as a np.array with shape (1,).
    """
    # Extract parameters
    T, ta1, tv, ta2 = traj_info["T"], traj_info["ta1"], traj_info["tv"], traj_info["ta2"]
    p1, p2, p_acc_end, p_const_end = traj_info["p1"], traj_info["p2"], traj_info["p_acc_end"], traj_info["p_const_end"]
    v1, v2, vlimit, vswitch = traj_info["v1"], traj_info["v2"], traj_info["vlimit"], traj_info["vswitch"]
    a1, a2 = traj_info["a1"], traj_info["a2"]
    
    # Helper to convert scalar to an array with shape (1,)
    def to_array(x):
        return np.array([x], dtype=np.float64) if not (isinstance(x, np.ndarray) and x.shape == (1,)) else x

    # Handle boundaries: if t is less than 0 or greater than T.
    if t <= 0:
        return to_array(p1), to_array(v1), to_array(a1)
    if t >= T:
        return to_array(p2), to_array(v2), to_array(a2)

    # Now, for 0 < t < T, compute state based on the phase.
    if tv is None:
        # Two-phase trajectory: acceleration (0 <= t <= ta1) and deceleration (t > ta1).
        if t <= ta1:
            pos = p1 + v1 * t + 0.5 * a1 * t**2
            vel = v1 + a1 * t
            acc = a1
        else:
            dt = t - ta1
            # Use precomputed p_acc_end for the start of deceleration.
            pos = p_acc_end + vswitch * dt + 0.5 * a2 * dt**2
            vel = vswitch + a2 * dt
            acc = a2
    else:
        # Three-phase trajectory: acceleration, constant velocity, then deceleration.
        if t <= ta1:
            pos = p1 + v1 * t + 0.5 * a1 * t**2
            vel = v1 + a1 * t
            acc = a1
        elif t <= ta1 + tv:
            dt = t - ta1
            # Start of constant phase: position p_acc_end.
            pos = p_acc_end + vlimit * dt
            vel = vlimit
            acc = 0.0
        else:
            dt = t - (ta1 + tv)
            pos = p_const_end + vlimit * dt + 0.5 * a2 * dt**2
            vel = vlimit + a2 * dt
            acc = a2
  
    return to_array(pos), to_array(vel), to_array(acc)
