import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle

from is_segment_feasible import is_segment_feasible
from get_motion_state import get_motion_states_at_local_t
from double_integrator_minimum_time import compute_traj_time, compute_traj_infos

# Default collision checker: returns True for all states
def default_collision_checker(state):
    # state is in the form (position, velocity)
    return True

class DIMTRRT:
    def __init__(self, init_state, goal_state, bounds, vmax, amax, 
                 collision_checker=default_collision_checker, delta_t=0.01, 
                 obstacles=None, visualization=False, save_frames=False, save_path='dimt_rrt_frames'):
        self.init_state = init_state
        self.goal_state = goal_state
        self.V1 = [init_state]
        self.V2 = [goal_state]
        self.E1 = {}
        self.E2 = {}
        self.path = None
        self.direction = True
        self.bounds = bounds
        self.vmax = vmax
        self.amax = amax
        self.collision_checker = collision_checker
        self.delta_t = delta_t
        self.obstacles = obstacles
        self.dimension = vmax.shape[0]
        self.visualization = visualization
        self.save_frames = save_frames
        self.save_path = save_path

    def solve(self, max_iteration=100):
        for iteration in range(max_iteration):
            if self.visualization:
                self.iteration = iteration
                self.plot()
            rand_state = self.sample_reachable_state()
            if self.connect(self.V1, self.E1, rand_state, self.direction):
                if self.connect(self.V2, self.E2, rand_state, not self.direction):
                    self.extract_trajectory(rand_state)
                    break
            self.swap()
            self.direction = not self.direction
        if self.visualization:
            self.plot()
        return self.path

    def sample_reachable_state(self):
        while True:
            sampled_p = np.random.uniform(self.bounds[0], self.bounds[1])
            sampled_v = np.random.uniform(-self.vmax, self.vmax)
            sampled_state = np.vstack([sampled_p, sampled_v])
            brake_forward_p = sampled_p + sampled_v ** 2 / (2 * self.amax)
            brake_backward_p = sampled_p - sampled_v ** 2 / (2 * self.amax)
            if np.any(brake_forward_p > self.bounds[1]) or np.any(brake_forward_p < self.bounds[0]):
                continue
            if np.any(brake_backward_p > self.bounds[1]) or np.any(brake_backward_p < self.bounds[0]):
                continue
            if not self.collision_checker(sampled_state):
                continue
            if self.visualization:
                self.sampled_state = sampled_state
                self.plot(plot_sampled_state=True)
            return sampled_state

    def connect(self, V:list, E:dict, rand_state, direction):
        nearest_state, traj_time = self.nearest(V, rand_state, direction)
        traj_infos = self.steer(nearest_state, rand_state, direction, traj_time)
        if self.visualization:
            self.steer_traj_time = traj_time
            self.steer_traj_infos = traj_infos
            self.plot(plot_sampled_state=True, plot_steer_trajectory=True)
        if self.collision_free(traj_time, traj_infos):
            int_states = self.intermediate_states(traj_time, traj_infos, direction)
            V.extend(int_states)
            for int_state in int_states:
                key = (int_state.shape, int_state.tobytes())
                E[key] = (nearest_state)
            if self.visualization:
                self.nearest_state = nearest_state
                self.rand_state = rand_state
                self.int_states = int_states
                self.extend_direction = direction
                self.plot(extend_tree=True)
            return True
        else:
            return False

    def nearest(self, V, rand_state, direction):
        cost = np.inf
        for v in V:
            start_state, end_state = (v, rand_state) if direction else (rand_state, v)
            traj_time = compute_traj_time(start_state=start_state, end_state=end_state, vmax=self.vmax, amax=self.amax, n_dim=self.dimension)
            if traj_time < cost:
                cost = traj_time
                nearest_state = v
        return nearest_state, cost

    def steer(self, nearest_state, rand_state, direction, traj_time):
        start_state, end_state = (nearest_state, rand_state) if direction else (rand_state, nearest_state)
        traj_infos = compute_traj_infos(start_state=start_state, end_state=end_state, traj_time=traj_time, vmax=self.vmax, amax=self.amax, n_dim=self.dimension)
        return traj_infos

    def collision_free(self, traj_time, traj_infos):
        return is_segment_feasible(traj_time, traj_infos, collision_checker=self.collision_checker, bounds=self.bounds, n_dim=self.dimension)

    def intermediate_states(self, traj_time, traj_infos, direction, n_states=10):
        index_range = np.arange(1, n_states + 1) if direction else np.arange(n_states)
        times = np.linspace(0, traj_time, num=n_states+1)[index_range]
        int_states = []
        for t in times:
            int_state = get_motion_states_at_local_t(traj_infos, t, n_dim=self.dimension)
            int_states.append(np.array(int_state))
        
        return int_states

    def swap(self):
        self.V1, self.V2 = self.V2, self.V1
        self.E1, self.E2 = self.E2, self.E1

    def extract_trajectory(self, connection_state):
        # Backtrace from conn to root using the provided edges.
        def backtrace(edges, root, conn):
            path = []
            while True:
                path.append(conn)
                if np.array_equal(conn, root):
                    break
                conn = edges[(conn.shape, conn.tobytes())]
            return path

        # Select edge dictionaries based on current direction.
        forward_edges, backward_edges = (self.E1, self.E2) if self.direction else (self.E2, self.E1)
        
        # Get and reverse the path from init to connection.
        path_from_init = backtrace(forward_edges, self.init_state, connection_state)[::-1]
        # Get the path from connection to goal.
        path_to_goal = backtrace(backward_edges, self.goal_state, connection_state)
        
        # Concatenate paths (omit duplicate connection state).
        self.path = path_from_init + path_to_goal[1:]

    def plot(self, plot_sampled_state=False, plot_steer_trajectory=False, extend_tree=False):
        assert self.dimension == 2
        # Initialize plot only on the first call
        if not hasattr(self, "_fig"):
            self._fig, self._ax = plt.subplots(figsize=(8, 6))
            self._ax.set_xlim(self.bounds[0][0] - 2, self.bounds[1][0] + 2)
            self._ax.set_ylim(self.bounds[0][1] - 2, self.bounds[1][1] + 2)
            self._ax.set_xlabel("X")
            self._ax.set_ylabel("Y")
            self._ax.set_title("DIMT RRT")
            self._ax.grid(True)

            # Add obstacles if exists
            if self.obstacles:
                self._obstacle_label_added = getattr(self, "_obstacle_label_added", False)
                for obs in self.obstacles:
                    if obs[0] == "ellipse":
                        _, center, rx, ry = obs
                        self._ax.add_patch(Ellipse(xy=center, width=2*rx, height=2*ry,
                                                edgecolor='r', facecolor='gray', alpha=0.5, 
                                                label="Obstacle" if not self._obstacle_label_added else ""))
                    elif obs[0] == "rectangle":
                        _, center, width, height = obs
                        self._ax.add_patch(Rectangle((center[0]-width/2, center[1]-height/2), width, height,
                                                    edgecolor='r', facecolor='gray', alpha=0.5,
                                                    label="Obstacle" if not self._obstacle_label_added else ""))
                    self._obstacle_label_added = True
            
            # Plot init_state and goal_state
            self.plot_state(self.init_state, color='r', markersize=9, label='Init State')
            self.plot_state(self.goal_state, color='g', markersize=9, label='Goal State')

            # Plot virtual sampled state to show legend
            self.sampled_state_point, self.sampled_state_arrow = self.plot_state(state=np.zeros([2, 2]), color='y', markersize=6, label='Sampled State')

            # Plot virtual forward and backward tree to show legend
            traj_time = np.array([2])
            traj_infos = compute_traj_infos(start_state=np.zeros([2, 2]), end_state=np.ones([2, 2]), traj_time=traj_time, vmax=self.vmax, amax=self.amax, n_dim=self.dimension)
            self.forward_tree_line = self.plot_trajectory(traj_time=traj_time, traj_infos=traj_infos, color='r', label='Forward Tree')
            self.backward_tree_line = self.plot_trajectory(traj_time=traj_time, traj_infos=traj_infos, color='g', label='Backward Tree')

            # Plot virtual steer trajectory to show legend
            self.steer_trajectory_line = self.plot_trajectory(traj_time=traj_time, traj_infos=traj_infos, color='y', label='Steer Trajectroy')

            # Plot virtual path to show legend
            self.path_line = self.plot_trajectory(traj_time=traj_time, traj_infos=traj_infos, color='b', label='Path')

            # Legend
            self._ax.legend(loc="lower left")

            # Remove virtual sampled state
            self.sampled_state_point[0].remove()
            self.sampled_state_point = None

            # Remove virtual forward and backward state
            self.forward_tree_line[0].remove()
            self.backward_tree_line[0].remove()
            del self.forward_tree_line, self.backward_tree_line
        
            # Remove virtual steer trajectory
            self.steer_trajectory_line[0].remove()
            self.steer_trajectory_line = None

            # Remove virtual path
            self.path_line[0].remove()
            del self.path_line

            self._iteration_text = self._ax.text(0.95, 0.95, "", transform=self._ax.transAxes, 
                                                fontsize=12, color="blue", ha="right", va="top")

        if plot_sampled_state:
            if self.sampled_state_point is None:
                self.sampled_state_point, self.sampled_state_arrow = self.plot_state(state=self.sampled_state, color='y', markersize=6)
        else:
            if self.sampled_state_point is not None:
                self.sampled_state_point[0].remove()
                self.sampled_state_point = None
            if self.sampled_state_arrow is not None:
                self.sampled_state_arrow.remove()
                self.sampled_state_arrow = None

        if plot_steer_trajectory:
            self.steer_trajectory_line = self.plot_trajectory(traj_time=self.steer_traj_time, traj_infos=self.steer_traj_infos, color='y', label='Steer Trajectroy')
        else:
            if self.steer_trajectory_line is not None:
                self.steer_trajectory_line[0].remove()
                self.steer_trajectory_line = None
        
        if extend_tree:
            color = 'r' if self.extend_direction else 'g'
            for state in self.int_states:
                self.plot_state(state=state, color=color)
            self.plot_trajectory(traj_time=self.steer_traj_time, traj_infos=self.steer_traj_infos, color=color)
        
        if self.path is not None:
            for start_state, end_state in zip(self.path, self.path[1:]):
                traj_time = compute_traj_time(start_state, end_state, self.vmax, self.amax, self.dimension)
                traj_infos = compute_traj_infos(start_state, end_state, traj_time, self.vmax, self.amax, self.dimension)
                self.plot_trajectory(traj_time, traj_infos)

        self._iteration_text.set_text(f"Iteration: {self.iteration}")

        # Save the frame if requested
        if self.save_frames:
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
            if not hasattr(self, 'frame_index'):
                self.frame_index = 0
            self._fig.savefig(f"{self.save_path}/frame_{self.frame_index:03d}.png")
            self.frame_index += 1

        # Draw the plot
        self._fig.canvas.draw()
        plt.pause(0.01)

    def plot_state(self, state, color='b', markersize=3, label=None):
        pos, vel = state[0], state[1]
        state_point = self._ax.plot(pos[0], pos[1], f'{color}o', markersize=markersize, label=label)
        if not (vel[0] == 0 and vel[1] == 0):
            arrow_scale = 0.5
            state_arrow = self._ax.arrow(pos[0], pos[1], vel[0] * arrow_scale, vel[1] * arrow_scale,
                head_width=0.15, head_length=0.15, fc=color, ec=color, alpha=0.2
            )
        else:
            state_arrow = None
        
        return state_point, state_arrow
    
    def plot_trajectory(self, traj_time, traj_infos, color='b', markersize=1, label=None):
        traj_pos = np.array([get_motion_states_at_local_t(traj_infos, t, self.dimension)[0] for t in np.linspace(0, traj_time, int((traj_time / 0.01)))])
        trajectory_line = self._ax.plot(traj_pos[:, 0], traj_pos[:, 1], f'-{color}o', markersize=markersize, label=label)

        return trajectory_line