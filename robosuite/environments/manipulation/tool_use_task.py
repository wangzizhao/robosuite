import numpy as np

from robosuite.environments.manipulation.tool_use import ToolUse
from robosuite.utils.observables import Observable, sensor


class ToolUseGoal(ToolUse):
    def __init__(self, xy_range=[0.25, 0.3], z_range=0.15, visualize_goal=True, **kwargs):
        """
        :param table_coverage: x y workspace ranges as a coverage factor of the table
        :param z_range: z workspace range
        """
        self.xy_range = xy_range
        self.z_range = z_range
        self.goal_space_low = None
        self.visualize_goal = visualize_goal
        super().__init__(**kwargs)

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        self.goal_vis_id = self.sim.model.body_name2id(self.model.mujoco_arena.goal_vis.root_body)

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()
        if self.goal_space_low is None:
            table_len_x, table_len_y, _ = self.table_full_size
            table_offset_z = self.table_offset[2]
            x_range, y_range = self.xy_range
            z_range = self.z_range
            self.goal_space_low = np.array([-x_range,
                                            -y_range,
                                            table_offset_z + 0.01])             # 0.02 is the half-size of the object
            self.goal_space_high = np.array([x_range,
                                             y_range,
                                             table_offset_z + z_range])
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

    def reset(self):
        obs = super().reset()
        obs["remain_t"] = np.array([1.])
        return obs

    def observation_spec(self):
        obs_spec = super().observation_spec()
        obs_spec["remain_t"] = np.array([0])
        return obs_spec

    def step(self, action):
        next_obs, reward, done, info = super().step(action)
        success = info["success"] = self.check_success()

        if self.sparse_reward:
            reward = success
        reward *= self.reward_scale

        next_obs["remain_t"] = np.array([1 - float(self.timestep) / self.horizon])
        return next_obs, float(reward), bool(done), info

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


class ToolUseReach(ToolUseGoal):
    def reward(self, action):
        gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
        dist = np.abs(gripper_site_pos - self.goal).sum()
        r_reach = 1 - np.tanh(5.0 * dist)
        return r_reach

    def check_success(self):
        gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
        dist = np.linalg.norm(gripper_site_pos - self.goal)
        return dist < 0.05


class ToolUsePick(ToolUseGoal):
    def __init__(self, object_name="cube", **kwargs):
        super().__init__(**kwargs)
        if object_name == "cube":
            self.obj = self.cube
            self.obj_id = self.cube_id
        elif object_name == "tool":
            self.obj = self.lshape_tool
            self.obj_id = self.lshape_tool_id
        else:
            raise NotImplementedError

    def reward(self, action):
        """
        Un-normalized summed components if using reward shaping:
            - Reaching: in [0, reach_mult], to encourage the arm to reach the cube
            - Grasping: in {0, grasp_mult}, to encourage the arm to grasp the cube
            - Lifting: in [0, lift_mult], to encourage the arm to lift the cube to the goal
        Note that the final reward is normalized.
        """
        reach_mult = 0.05 # 0.1
        grasp_mult = 0.4 # 0.35
        lift_mult = 0.5 # 0.5
        gripper_open = action[-1] < 0

        reward = 0

        obj_pos = self.sim.data.body_xpos[self.obj_id]
        gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
        dist = np.linalg.norm(gripper_site_pos - obj_pos)
        r_reach = (1 - np.tanh(5.0 * dist)) * reach_mult

        reward += r_reach

        r_lift = 0
        grasped_obj = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.obj)
        if grasped_obj:
            reward += grasp_mult

            dist = np.linalg.norm(obj_pos - self.goal)
            r_lift = (1 - np.tanh(5.0 * dist)) * lift_mult
            reward += r_lift

            if dist < 0.05: # success
                reward += 1

        return reward

    def check_success(self):
        obj_pos = self.sim.data.body_xpos[self.obj_id]
        dist = np.linalg.norm(obj_pos - self.goal)
        return dist < 0.05


class ToolUsePickCube(ToolUsePick):
    def __init__(self, **kwargs):
        super().__init__(object_name="cube", **kwargs)


class ToolUsePickTool(ToolUsePick):
    def __init__(self, **kwargs):
        super().__init__(object_name="tool", **kwargs)


class ToolUseSeries(ToolUseGoal):
    def __init__(self, terminal_state="POT_LIFTING", **kwargs):
        super().__init__(**kwargs)

        self.terminal_state = terminal_state

        self.STATES = ["TOOL_GRASPING", "TOOL_MOVING", "PUSHING",
                       "CUBE_PICKING", "CUBE_MOVING", "POT_PICKING", "POT_LIFTING"]
        self.state_idx = {state:i for i, state in enumerate(self.STATES)}

        self.state = "TOOL_GRASPING"
        self.reached_tool_goal = False
        self.push_x_thre = 0.0

    def reset(self):
        self.state = "TOOL_GRASPING"
        self.reached_tool_goal = False
        return super().reset()

    def reward(self, action):
        """
        Un-normalized summed components if using reward shaping:
            - Reaching: in [0, reach_mult], to encourage the arm to reach the cube
            - Pushing: in [0, push_mult], to encourage the arm to push the cube to the goal
        Note that the final reward is normalized.
        """

        reward = 0

        gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
        cube_pos = self.sim.data.body_xpos[self.cube_id]
        tool_pos = self.sim.data.body_xpos[self.lshape_tool_id]
        pot_pos = self.sim.data.body_xpos[self.pot_object_id]

        tool_head_pos = self.sim.data.geom_xpos[self.tool_head_id]
        pot_handle_pos = self.sim.data.geom_xpos[self.pot_right_handle_id]

        cube_pos_x = cube_pos[0]
        cube_pos_z = cube_pos[2]
        tool_head_pos_x = tool_head_pos[0]
        table_height = self.table_offset[2]

        tool_head_goal_pos = cube_pos + np.array([0.1, -0.1, 0])
        tool_head_goal_dist = np.linalg.norm(tool_head_pos - tool_head_goal_pos)

        cube_pot_dist_xy = np.linalg.norm(cube_pos[:2] - pot_pos[:2])

        cube_grasped = tool_grasped = pot_grasped = cube_touching_pot = False
        cube_grasped = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cube)
        tool_grasped = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.lshape_tool)
        pot_grasped = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.pot_object)

        cube_touching_tool = self.check_contact(self.cube, self.lshape_tool)
        cube_touching_pot = self.check_contact(self.cube, self.pot_object)

        cube_lifted = cube_pos_z > table_height + 0.02

        # determine current state
        if self.state == "TOOL_GRASPING":
            if cube_pos_x < self.push_x_thre:
                self.state = "CUBE_PICKING"
            elif tool_grasped:
                if self.reached_tool_goal:
                    self.state = "PUSHING"
                else:
                    self.state = "TOOL_MOVING"
        elif self.state == "TOOL_MOVING":
            if cube_pos_x < self.push_x_thre:
                self.state = "CUBE_PICKING"
            elif not tool_grasped:
                self.state = "TOOL_GRASPING"
            elif tool_head_goal_dist < 0.05:
                self.reached_tool_goal = True
                self.state = "PUSHING"
        elif self.state == "PUSHING":
            if cube_pos_x < self.push_x_thre:
                self.state = "CUBE_PICKING"
            elif not tool_grasped:
                self.state = "TOOL_GRASPING"
        elif self.state == "CUBE_PICKING":
            if cube_grasped:
                self.state = "CUBE_MOVING"
        elif self.state == "CUBE_MOVING":
            if not cube_grasped:
                if cube_pot_dist_xy < 0.02:
                    self.state = "POT_PICKING"
                else:
                    self.state = "CUBE_PICKING"
        elif self.state == "POT_PICKING":
            if pot_grasped:
                self.state = "POT_LIFTING"
        elif self.state == "POT_LIFTING":
            if not pot_grasped:
                self.state = "POT_PICKING"
        else:
            raise NotImplementedError

        if self.state_idx[self.state] > self.state_idx[self.terminal_state]:
            self.state = self.terminal_state

        tool_reach_mult = 0.05
        tool_grasp_mult = 0.2
        tool_mov_mult = 0.2
        push_mult = 0.2
        tool_release_mult = 0.4
        cube_reach_mult = 0.4
        cube_grasp_mult = 0.6
        cube_mov_mult = 0.5
        cube_place_mult = 1.0
        pot_handle_reach_mult = 0.4
        pot_handle_grasp_mult = 1.0
        pot_reach_mult = 0.5

        if self.state == "TOOL_GRASPING":
            eef_tool_dist = np.linalg.norm(gripper_site_pos - tool_pos)
            r_eef_reach_tool = (1 - np.tanh(5.0 * eef_tool_dist)) * tool_reach_mult
            reward = r_eef_reach_tool
        elif self.state == "TOOL_MOVING":
            r_tool_reach_goal = (1 - np.tanh(5.0 * tool_head_goal_dist)) * tool_mov_mult
            reward = tool_reach_mult + tool_grasp_mult + r_tool_reach_goal
        elif self.state == "PUSHING":
            reward = tool_reach_mult + tool_grasp_mult + tool_mov_mult

            tool_head_cube_dist = np.linalg.norm(tool_head_pos - cube_pos)
            r_tool_head_reach_cube = (1 - np.tanh(5.0 * tool_head_cube_dist)) * push_mult
            reward += r_tool_head_reach_cube

            if cube_touching_tool:
                cube_to_push_dist = np.maximum(0 - cube_pos_x, 0)
                r_cube_push = (1 - np.tanh(5.0 * cube_to_push_dist)) * push_mult
                reward += r_cube_push
        elif self.state == "CUBE_PICKING":
            reward = tool_reach_mult + tool_grasp_mult + tool_mov_mult + 2 * push_mult
            if not tool_grasped:
                eef_cube_dist = np.linalg.norm(gripper_site_pos - cube_pos)
                r_eef_reach_cube = (1 - np.tanh(5.0 * eef_cube_dist)) * cube_reach_mult
                reward += tool_release_mult + r_eef_reach_cube
        elif self.state == "CUBE_MOVING":
            reward = tool_reach_mult + tool_grasp_mult + tool_mov_mult + 2 * push_mult + \
                     tool_release_mult + cube_reach_mult

            cube_pot_horiz_dist = np.linalg.norm(cube_pos[:2] - pot_pos[:2])
            cube_to_raise_dist = np.maximum(pot_pos[2] + 0.15 - cube_pos_z, 0)
            r_cube_reach_goal = (1 - np.tanh(5.0 * cube_pot_horiz_dist)) * cube_mov_mult + \
                                (1 - np.tanh(5.0 * cube_to_raise_dist)) * cube_mov_mult
            reward += cube_grasp_mult + r_cube_reach_goal
        elif self.state == "POT_PICKING":
            reward = tool_reach_mult + tool_grasp_mult + tool_mov_mult + 2 * push_mult + \
                     tool_release_mult + cube_reach_mult + cube_grasp_mult + 2 * cube_mov_mult

            eef_pot_handle_dist = np.linalg.norm(gripper_site_pos - pot_handle_pos)
            r_eef_reach_pot_handle = (1 - np.tanh(5.0 * eef_pot_handle_dist)) * pot_handle_reach_mult
            reward += cube_place_mult + r_eef_reach_pot_handle
        elif self.state == "POT_LIFTING":
            reward = tool_reach_mult + tool_grasp_mult + tool_mov_mult + 2 * push_mult + \
                     tool_release_mult + cube_reach_mult + cube_grasp_mult + 2 * cube_mov_mult + \
                     cube_place_mult + pot_handle_reach_mult

            pot_goal_dist = np.linalg.norm(pot_pos - self.goal)
            r_pot_reach_goal = (1 - np.tanh(5.0 * pot_goal_dist)) * pot_reach_mult
            reward += pot_handle_grasp_mult + r_pot_reach_goal

        return reward

    def check_success(self):
        if self.terminal_state == "PUSHING":
            cube_pos = self.sim.data.body_xpos[self.cube_id]
            cube_pos_x = cube_pos[0]
            return cube_pos_x < 0
        elif self.terminal_state == "CUBE_MOVING":
            cube_grasped = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cube)

            cube_pos = self.sim.data.body_xpos[self.cube_id]
            pot_pos = self.sim.data.body_xpos[self.pot_object_id]
            cube_pot_horiz_dist = np.linalg.norm(cube_pos[:2] - pot_pos[:2])

            return cube_grasped and cube_pot_horiz_dist < 0.05
        elif self.terminal_state == "POT_PICKING":
            cube_pos = self.sim.data.body_xpos[self.cube_id]
            cube_pos_z = cube_pos[2]
            table_height = self.table_offset[2]
            cube_lifted = cube_pos_z > table_height + 0.02

            cube_grasped = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cube)

            cube_touching_pot = self.check_contact(self.cube, self.pot_object)

            return not cube_grasped and cube_touching_pot and cube_lifted
        elif self.terminal_state == "POT_LIFTING":
            cube_pos = self.sim.data.body_xpos[self.cube_id]
            cube_pos_z = cube_pos[2]
            table_height = self.table_offset[2]
            cube_lifted = cube_pos_z > table_height + 0.02

            cube_grasped = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cube)

            cube_touching_pot = self.check_contact(self.cube, self.pot_object)

            pot_pos = self.sim.data.body_xpos[self.pot_object_id]
            pot_goal_dist = np.linalg.norm(pot_pos - self.goal)

            return not cube_grasped and cube_touching_pot and cube_lifted and pot_goal_dist < 0.05
        else:
            raise NotImplementedError


class ToolUsePickPlace(ToolUseGoal):
    def __init__(self, **kwargs):
        super().__init__(visualize_goal=False, **kwargs)

    def reward(self, action):
        """
        Un-normalized summed components if using reward shaping:
            - Reaching: in [0, reach_mult], to encourage the arm to reach the cube
            - Grasping: in {0, grasp_mult}, to encourage the arm to grasp the cube
            - Lifting: in [0, lift_mult], to encourage the arm to lift the cube to the goal
        Note that the final reward is normalized.
        """
        reach_mult = 0.05
        grasp_mult = 0.4
        lift_mult = 0.5
        place_mult = 2.0

        reward = 0

        cube_pos = self.sim.data.body_xpos[self.cube_id]
        pot_pos = self.sim.data.body_xpos[self.pot_object_id]
        gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]

        dist = np.linalg.norm(gripper_site_pos - cube_pos)
        r_reach = (1 - np.tanh(5.0 * dist)) * reach_mult

        # grasping reward
        cube_grasped = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cube)
        if cube_grasped:
            r_reach += grasp_mult

        # lifting is successful when the cube is above the table top by a margin
        r_lift = 0
        cube_height = cube_pos[2]
        table_height = self.table_offset[2]
        cube_lifted = cube_height > table_height + 0.02

        # Aligning is successful when cube is right above cubeB
        if cube_grasped:
            horiz_dist = np.abs(cube_pos[:2] - pot_pos[:2]).sum()
            vert_dist = np.maximum(table_height + 0.15 - cube_height, 0)
            r_lift += lift_mult * (2 - np.tanh(5.0 * horiz_dist) - np.tanh(5.0 * vert_dist))

        # stacking is successful when the block is lifted and the gripper is not holding the object
        r_place = 0
        cube_touching_pot = self.check_contact(self.cube, self.pot_object)
        if not cube_grasped and cube_lifted and cube_touching_pot:
            r_place = place_mult

        reward = r_reach + r_lift + r_place

        return reward

    def check_success(self):
        cube_grasped = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cube)
        cube_touching_pot = self.check_contact(self.cube, self.pot_object)
        cube_pos = self.sim.data.body_xpos[self.cube_id]
        cube_height = cube_pos[2]
        table_height = self.table_offset[2]
        cube_lifted = cube_height > table_height + 0.025
        return not cube_grasped and cube_touching_pot and cube_lifted
