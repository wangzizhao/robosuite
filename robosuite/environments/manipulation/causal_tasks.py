import numpy as np

from robosuite.environments.manipulation.causal import Causal
from robosuite.utils.observables import Observable, sensor


class CausalGoal(Causal):
    def __init__(self, table_coverage=0.8, z_range=0.5, **kwargs):
        """
        :param table_coverage: x y workspace ranges as a coverage factor of the table
        :param z_range: z workspace range
        """
        self.coverage = table_coverage
        self.z_range = z_range
        self.goal_space_low = None
        self.visualize_goal = True
        super().__init__(**kwargs)

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        assert self.num_movable_objects > 0
        self.cube = self.movable_objects[0]
        self.cube_body_id = self.sim.model.body_name2id(self.cube.root_body)

        self.goal_vis_id = self.sim.model.body_name2id(self.model.mujoco_arena.goal_vis.root_body)

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()
        if self.goal_space_low is None:
            table_len_x, table_len_y, _ = self.table_full_size
            table_offset_z = self.table_offset[2]
            coverage = self.coverage
            z_range = self.z_range
            self.goal_space_low = np.array([-table_len_x * coverage / 2,
                                            -table_len_y * coverage / 2,
                                            table_offset_z + 0.02])             # 0.02 is the half-size of the object
            self.goal_space_high = np.array([table_len_x * coverage / 2,
                                             table_len_y * coverage / 2,
                                             table_offset_z + 0.02 + z_range])
        self.goal = np.random.uniform(self.goal_space_low, self.goal_space_high)
        if self.visualize_goal:
            goal_pos = self.goal.copy()
            goal_pos[-1] -= self.table_offset[2]
            self.sim.model.body_pos[self.goal_vis_id] = goal_pos
        else:
            self.sim.model.body_pos[self.goal_vis_id] = np.array([0, 0, -1])

    def reward(self, action):
        raise NotImplementedError

    def check_success(self):
        raise NotImplementedError

    def step(self, action):
        next_obs, reward, done, info = super().step(action)
        info["success"] = self.check_success()
        return next_obs, reward, done, info

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        @sensor(modality="object")
        def goal_pos(obs_cache):
            return self.goal

        observables["goal_pos"] = Observable(name="goal_pos", sensor=goal_pos, sampling_rate=self.control_freq)

        return observables


class CausalReach(CausalGoal):
    def reward(self, action):
        gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
        dist = np.linalg.norm(gripper_site_pos - self.goal)
        r_reach = 1 - np.tanh(10.0 * dist)
        return r_reach

    def check_success(self):
        gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
        dist = np.linalg.norm(gripper_site_pos - self.goal)
        return dist < 0.05


class CausalPush(CausalGoal):
    def __init__(self, **kwargs):
        assert "z_range" not in kwargs, "invalid set of arguments"
        super().__init__(z_range=0, **kwargs)

    def reward(self, action):
        """
        Un-normalized summed components if using reward shaping:
            - Reaching: in [0, reach_mult], to encourage the arm to reach the cube
            - Pushing: in [0, push_mult], to encourage the arm to push the cube to the goal
        Note that the final reward is normalized.
        """
        reach_mult = 0.5
        push_mult = 1.0

        reward = 0

        cube_pos = self.sim.data.body_xpos[self.cube_body_id]
        gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
        dist = np.linalg.norm(gripper_site_pos - cube_pos)
        r_reach = (1 - np.tanh(5.0 * dist)) * reach_mult
        reward += r_reach

        dist = np.linalg.norm(cube_pos - self.goal)
        r_push = (1 - np.tanh(5.0 * dist)) * push_mult
        reward += r_push

        reward /= (reach_mult + push_mult)
        return reward

    def check_success(self):
        cube_pos = self.sim.data.body_xpos[self.cube_body_id]
        dist = np.linalg.norm(cube_pos - self.goal)
        return dist < 0.05


class CausalPick(CausalGoal):
    def reward(self, action):
        """
        Un-normalized summed components if using reward shaping:
            - Reaching: in [0, reach_mult], to encourage the arm to reach the cube
            - Grasping: in {0, grasp_mult}, to encourage the arm to grasp the cube
            - Lifting: in [0, lift_mult], to encourage the arm to lift the cube to the goal
        Note that the final reward is normalized.
        """
        reach_mult = 0.1
        grasp_mult = 0.35
        lift_mult = 0.5

        reward = 0

        cube_pos = self.sim.data.body_xpos[self.cube_body_id]
        gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
        dist = np.linalg.norm(gripper_site_pos - cube_pos)
        r_reach = (1 - np.tanh(5.0 * dist)) * reach_mult

        grasping_cubeA = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cube)
        if grasping_cubeA:
            r_reach += grasp_mult

        reward += r_reach

        dist = np.linalg.norm(cube_pos - self.goal)
        r_lift = (1 - np.tanh(5.0 * dist)) * lift_mult
        reward += r_lift

        reward /= (reach_mult + grasp_mult + lift_mult)
        return reward

    def check_success(self):
        cube_pos = self.sim.data.body_xpos[self.cube_body_id]
        dist = np.linalg.norm(cube_pos - self.goal)
        return dist < 0.05


class CausalGrasp(CausalGoal):
    def __init__(self, **kwargs):
        assert "z_range" not in kwargs, "invalid set of arguments"
        super().__init__(z_range=0, **kwargs)
        self.visualize_goal = False

    def reward(self, action):
        """
        Un-normalized summed components if using reward shaping:
            - Reaching: in [0, reach_mult], to encourage the arm to reach the cube
            - Pushing: in [0, push_mult], to encourage the arm to push the cube to the goal
        Note that the final reward is normalized.
        """
        reach_mult = 0.5
        grasp_mult = 1.0

        gripper_close = action[-1]

        cube_pos = self.sim.data.body_xpos[self.cube_body_id]
        gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
        dist = np.abs(gripper_site_pos - cube_pos).sum()

        grasping_cubeA = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cube)
        if grasping_cubeA:
            reward = grasp_mult
        else:
            reward = (1 - np.tanh(10.0 * dist)) * (gripper_close < 0) * reach_mult

        return reward
