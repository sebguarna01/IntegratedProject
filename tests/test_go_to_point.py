import numpy as np

from activenav.control.go_to_point import go_to_point
from activenav.kinematics import unicycle_step


def test_reaches_square_waypoints_in_kinematic_sim():
    pose = np.array([1.0, -0.5, 0.0])
    dt = 0.02
    for goal in [(2.0, -0.5), (2.0, 0.5), (1.0, 0.5), (1.0, -0.5)] * 2:
        for _ in range(int(60 / dt)):
            v, w, reached = go_to_point(pose, goal)
            if reached:
                break
            pose = unicycle_step(pose, v, w, dt)
        assert reached, f"did not reach {goal}"
