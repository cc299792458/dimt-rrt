import numpy as np

from is_segment_feasible import is_segment_feasible
from get_motion_state import get_motion_states_at_local_t
from double_integrator_minimum_time import compute_traj_time, compute_traj_infos

class DIMTRRT:
    def __init__(self, init_state, goal_state, bounds, vmax, amax, collision_checker, delta_t=0.01):
        self.init_state = init_state
        self.goal_state = goal_state
        self.V1 = [init_state]
        self.V2 = [goal_state]
        self.E1 = {}
        self.E2 = {}
        self.direction = True
        self.bounds = bounds
        self.vmax = vmax
        self.amax = amax
        self.collision_checker = collision_checker
        self.delta_t = delta_t
        self.dimension = vmax.shape[0]

    def solve(self, max_iteration=100, plot=False):
        for i in range(max_iteration):
            rand_state = self.sample_reachable_state()
            if self.connect(self.V1, self.E1, rand_state, self.direction):
                if self.connect(self.V2, self.E2, rand_state, not self.direction):
                    return self.extract_trajectory()
            self.swap()
            self.direction = not self.direction

    def sample_reachable_state(self):
        while True:
            sampled_p = np.random.uniform(self.bounds[0], self.bounds[1])
            sampled_v = np.random.uniform(-self.vmax, self.vmax)
            brake_forward_p = sampled_p + sampled_v ** 2 / (2 * self.amax)
            brake_backward_p = sampled_p - sampled_v ** 2 / (2 * self.amax)
            if np.any(brake_forward_p > self.bounds[1]) or np.any(brake_forward_p < self.bounds[0]):
                continue
            if np.any(brake_backward_p > self.bounds[1]) or np.any(brake_backward_p < self.bounds[0]):
                continue
            return np.vstack([sampled_p, sampled_v])

    def connect(self, V:list, E:dict, rand_state, direction):
        nearest_state, traj_time = self.nearest(V, rand_state, direction)
        traj_infos = self.steer(nearest_state, rand_state, direction, traj_time)
        if self.collision_free(traj_time, traj_infos):
            int_states = self.intermediate_states(traj_time, traj_infos, direction)
            V.extend(int_states)
            for int_state in int_states:
                E[int_state] = nearest_state

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
            int_states.append(int_state)
        
        return int_state

    def swap(self):
        self.V1, self.V2 = self.V2, self.V1
        self.E1, self.E2 = self.E2, self.E1

    def extract_trajectory(self):
        pass

    def plot(self):
        pass