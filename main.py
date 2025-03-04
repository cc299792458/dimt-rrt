import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle

from dimt_rrt import DIMTRRT

class Scene:
    """
    A scene with both elliptical and rectangular obstacles. The scene is 10x10 by default.
    It generates collision-free start and goal points that are at least a specified minimum distance apart.
    """
    def __init__(self, bounds=(0, 10, 0, 10), 
                 num_ellipses=10, num_rectangles=5, 
                 min_axis=0.5, max_axis=1.5, 
                 min_side=0.5, max_side=1.5, 
                 min_distance=5.0):
        
        self.bounds = bounds
        self.obstacles = []
        # Generate elliptical obstacles; each obstacle is stored as ("ellipse", center, rx, ry)
        self.obstacles += self._generate_elliptical_obstacles(num_ellipses, min_axis, max_axis)
        # Generate rectangular obstacles; each obstacle is stored as ("rectangle", center, width, height)
        self.obstacles += self._generate_rectangular_obstacles(num_rectangles, min_side, max_side)
        
        self.start = self._generate_collision_free_point()
        self.goal = self._generate_collision_free_point()
        # Ensure that the start and goal are at least min_distance apart
        while np.linalg.norm(self.goal - self.start) < min_distance:
            self.goal = self._generate_collision_free_point()
    
    def _generate_elliptical_obstacles(self, num_obstacles, min_axis, max_axis):
        obstacles = []
        x_min, x_max, y_min, y_max = self.bounds
        for _ in range(num_obstacles):
            rx = random.uniform(min_axis/2, max_axis/2)
            ry = random.uniform(min_axis/2, max_axis/2)
            cx = random.uniform(x_min + rx, x_max - rx)
            cy = random.uniform(y_min + ry, y_max - ry)
            obstacles.append(("ellipse", np.array([cx, cy]), rx, ry))
        return obstacles
    
    def _generate_rectangular_obstacles(self, num_obstacles, min_side, max_side):
        """
        Generate rectangular obstacles:
          - Each obstacle is represented as ("rectangle", center, width, height)
          - To ensure that the rectangle remains within bounds even after rotation,
            we use half of its diagonal as a buffer from the scene boundaries.
        """
        obstacles = []
        x_min, x_max, y_min, y_max = self.bounds
        for _ in range(num_obstacles):
            width = random.uniform(min_side, max_side)
            height = random.uniform(min_side, max_side)
            half_diag = np.sqrt((width/2)**2 + (height/2)**2)
            cx = random.uniform(x_min + half_diag, x_max - half_diag)
            cy = random.uniform(y_min + half_diag, y_max - half_diag)
            obstacles.append(("rectangle", np.array([cx, cy]), width, height))
        return obstacles
    
    def _is_point_collision_free(self, pos):
        """
        Check if the given point pos collides with any obstacle.
        For ellipses: the point is rotated into the ellipse frame and checked against the ellipse equation.
        For rectangles: the point is rotated into the rectangle's coordinate frame and checked to see if it falls within.
        """
        for obs in self.obstacles:
            if obs[0] == "ellipse":
                _, center, rx, ry = obs
                dx = pos[0] - center[0]
                dy = pos[1] - center[1]
                if (dx**2 / rx**2 + dy**2 / ry**2) <= 1:
                    return False
            elif obs[0] == "rectangle":
                _, center, width, height = obs
                dx = pos[0] - center[0]
                dy = pos[1] - center[1]
                if abs(dx) <= width/2 and abs(dy) <= height/2:
                    return False
        return True
    
    def _generate_collision_free_point(self, max_attempts=1000):
        x_min, x_max, y_min, y_max = self.bounds
        for _ in range(max_attempts):
            candidate = np.array([random.uniform(x_min, x_max), random.uniform(y_min, y_max)])
            if self._is_point_collision_free(candidate):
                return candidate
        raise RuntimeError("Failed to generate a collision-free point after many attempts.")
    
    def collision_checker(self, state):
        pos = state[0]
        return self._is_point_collision_free(pos)
    
# Example usage
if __name__ == "__main__":
    seed = 42
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    scene = Scene(num_ellipses=2, num_rectangles=2, min_distance=5.0)

    vmax, amax = np.array([1.0, 1.0]), np.array([1.0, 1.0])

    dimt_rrt = DIMTRRT(np.vstack([scene.start, np.zeros([2])]), np.vstack([scene.goal, np.zeros([2])]), 
                       np.vstack((scene.bounds[::2], scene.bounds[1::2])), vmax=vmax, amax=amax, 
                       collision_checker=scene.collision_checker, obstacles=scene.obstacles, visualization=True)
    dimt_rrt.solve()