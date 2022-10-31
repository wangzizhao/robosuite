from collections import OrderedDict
import numpy as np
from scipy.spatial.transform import Rotation as R

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv

from robosuite.models.arenas import MarkerArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler, SequentialCompositeSampler
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.mjcf_utils import CustomMaterial, array_to_string
from robosuite.utils.transform_utils import convert_quat

from robosuite.models.tool_use import LShapeTool, PotObject

class ToolUse(SingleArmEnv):
    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 1.2, 0.05),
        table_offset=(0.0, 0.0, 0.8),
        table_friction=(1., 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        contact_threshold=2.0,
        cube_x_range=(0.2, 0.3),
        cube_y_range=(-0.3, 0.0),
        tool_x_range=(0.07, 0.07),
        tool_y_range=(-0.05, -0.05),
        num_markers=3,
        marker_x_range=(-0.3, 0.3),
        marker_y_range=(-0.3, 0.3)
    ):
        # settings for table top (hardcoded since it's not an essential part of the environment)
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array(table_offset)

        # reward configuration
        self.reward_scale = reward_scale

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        # ee resets
        self.ee_force_bias = np.zeros(3)
        self.ee_torque_bias = np.zeros(3)

        # Thresholds
        self.contact_threshold = contact_threshold

        # History observations
        self._history_force_torque = None
        self._recent_force_torque = None

        self.cube_x_range = cube_x_range
        self.cube_y_range = cube_y_range
        self.tool_x_range = tool_x_range
        self.tool_y_range = tool_y_range

        self.num_markers = num_markers
        self.marker_x_range = marker_x_range
        self.marker_y_range = marker_y_range

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
        )

    def reward(self, action=None):
        """
        Reward function for the task.

        Sparse un-normalized reward:

            - a discrete reward of 1.0 is provided if the drawer is opened

        Un-normalized summed components if using reward shaping:

            - Reaching: in [0, 0.25], proportional to the distance between drawer handle and robot arm
            - Rotating: in [0, 0.25], proportional to angle rotated by drawer handled
              - Note that this component is only relevant if the environment is using the locked drawer version

        Note that a successfully completed task (drawer opened) will return 1.0 irregardless of whether the environment
        is using sparse or shaped rewards

        Note that the final reward is normalized and scaled by reward_scale / 1.0 as
        well so that the max score is equal to reward_scale

        Args:
            action (np.array): [NOT USED]

        Returns:
            float: reward value
        """
        reward = 0.

        # sparse completion reward
        if self._check_success():
            reward = 1.0

        # Scale reward if requested
        if self.reward_scale is not None:
            reward *= self.reward_scale / 1.0

        return reward

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = MarkerArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
            num_markers=self.num_markers,
            marker_x_range=self.marker_x_range,
            marker_y_range=self.marker_y_range
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        bluewood = CustomMaterial(
            texture="WoodBlue",
            tex_name="bluewood",
            mat_name="handle1_mat",
            tex_attrib={"type": "cube"},
            mat_attrib={"texrepeat": "1 1", "specular": "0.4", "shininess": "0.1"},
        )

        ingredient_size = [0.02, 0.02, 0.02]
        self.cube = BoxObject(
            name="cube",
            size_min=ingredient_size,
            size_max=ingredient_size,
            rgba=[1, 0, 0, 1],
            material=bluewood,
            density=500.,
        )
        
        self.lshape_tool = LShapeTool(
            name="tool",
        )
        self.pot_object = PotObject(
            name="pot",
        )
        pot_object = self.pot_object.get_obj()
        pot_object.set("pos", array_to_string((0.0, 0.2, self.table_offset[2] + 0.03)))

        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
        
        # Create placement initializer
        self.placement_initializer.append_sampler(
        sampler=UniformRandomSampler(
            name="ObjectSampler-cube",
            mujoco_objects=self.cube,
            x_range=self.cube_x_range,
            y_range=self.cube_y_range,
            rotation=[-np.pi, np.pi],
            rotation_axis='z',
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=self.table_offset,
            z_offset=0.01,
        ))

        self.placement_initializer.append_sampler(
        sampler=UniformRandomSampler(
            name="ObjectSampler-lshape",
            mujoco_objects=self.lshape_tool,
            x_range=self.tool_x_range,
            y_range=self.tool_y_range,
            rotation=[0, 0],
            rotation_axis='z',
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=self.table_offset,
            z_offset=0.02,
        ))
        
        mujoco_objects = [
            self.pot_object,
            self.cube,
            self.lshape_tool
        ]

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots], 
            mujoco_objects=mujoco_objects,
        )
        self.objects = {obj.name: obj for obj in [self.pot_object, self.cube, self.lshape_tool]}

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.object_body_ids = dict()

        self.pot_object_id = self.sim.model.body_name2id(self.pot_object.root_body)
        self.lshape_tool_id = self.sim.model.body_name2id(self.lshape_tool.root_body)
        self.cube_id = self.sim.model.body_name2id(self.cube.root_body)

        self.obj_body_id = {}
        for name, obj in self.objects.items():
            self.obj_body_id[name] = self.sim.model.body_name2id(obj.root_body)
        
    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()
        
        # low-level object information
        if self.use_object_obs:
            modality = "object"

            @sensor(modality=modality)
            def robot0_eef_vel(obs_cache):
                return self.robots[0]._hand_vel

            sensors, names = [robot0_eef_vel], ["robot0_eef_vel"]

            for name in self.objects:
                obj_sensors, obj_sensor_names = self._create_obj_sensors(name, modality)
                sensors += obj_sensors
                names += obj_sensor_names

            for i, marker in enumerate(self.model.mujoco_arena.markers):
                marker_sensors, marker_sensor_names = self._create_marker_sensors(i, marker, modality)
                sensors += marker_sensors
                names += marker_sensor_names

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )
                
        return observables

    def _create_marker_sensors(self, i, marker, modality="object"):
        """
        Helper function to create sensors for a given marker. This is abstracted in a separate function call so that we
        don't have local function naming collisions during the _setup_observables() call.

        Args:
            i (int): ID number corresponding to the marker
            marker (MujocoObject): Marker to create sensors for
            modality (str): Modality to assign to all sensors

        Returns:
            2-tuple:
                sensors (list): Array of sensors for the given marker
                names (list): array of corresponding observable names
        """
        @sensor(modality=modality)
        def marker_pos(obs_cache):
            return np.array(self.sim.data.body_xpos[self.sim.model.body_name2id(marker.root_body)])

        sensors = [marker_pos]
        names = [f"marker{i}_pos"]

        return sensors, names

    def _create_obj_sensors(self, obj_name, modality="object"):
        """
        Helper function to create sensors for a given object. This is abstracted in a separate function call so that we
        don't have local function naming collisions during the _setup_observables() call.

        Args:
            obj_name (str): Name of object to create sensors for
            modality (str): Modality to assign to all sensors

        Returns:
            2-tuple:
                sensors (list): Array of sensors for the given obj
                names (list): array of corresponding observable names
        """
        @sensor(modality=modality)
        def obj_pos(obs_cache):
            return np.array(self.sim.data.body_xpos[self.obj_body_id[obj_name]])

        @sensor(modality=modality)
        def obj_quat(obs_cache):
            return convert_quat(self.sim.data.body_xquat[self.obj_body_id[obj_name]], to="xyzw")

        @sensor(modality=modality)
        def object_zrot(obs_cache):
            quat = convert_quat(np.array(self.sim.data.body_xquat[self.obj_body_id[obj_name]]),
                                to="xyzw")
            z = R.from_quat(quat).as_euler('xyz')[2]
            return np.array([np.sin(z), np.cos(z)]).astype(np.float32)

        @sensor(modality=modality)
        def object_grasped(obs_cache):
            grasped = int(self._check_grasp(gripper=self.robots[0].gripper,
                                            object_geoms=[g for g in self.objects[obj_name].contact_geoms]))
            return grasped

        sensors = [obj_pos, obj_quat, object_zrot, object_grasped]
        names = [f"{obj_name}_pos", f"{obj_name}_quat", f"{obj_name}_zrot", f"{obj_name}_grasped"]

        return sensors, names

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:
            self.model.mujoco_arena.reset_arena(self.sim)

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

    def _check_success(self):
        """
        Check if cube is in pot

        Returns:
            bool: True if drawer has been opened
        """

        pot_pos = self.sim.data.body_xpos[self.pot_object_id]
        cube_pos = self.sim.data.body_xpos[self.cube_id]
        object_in_pot = self.check_contact(self.cube, self.pot_object) and \
                        np.linalg.norm(pot_pos[:2] - cube_pos[:2]) < 0.06 and \
                        np.abs(pot_pos[2] - cube_pos[2]) < 0.05
        
        return object_in_pot

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the drawer handle.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

    def step(self, action):
        self.model.mujoco_arena.step_arena(self.sim)
        return super().step(action)

    def obs_delta_range(self):
        max_delta_eef_pos = 0.1 * np.ones(3)
        max_delta_eef_vel = 0.1 * np.ones(3)
        max_delta_joint_vel = 5 * np.ones(6)
        max_delta_gripper_qpos = 0.02 * np.ones(2)
        max_delta_gripper_qvel = 0.5 * np.ones(2)
        max_delta_cube_pos = 0.1 * np.ones(3)
        max_delta_cube_quat = 2 * np.ones(4)
        max_delta_cube_zrot = 2 * np.ones(2)
        max_delta_tool_pos = 0.1 * np.ones(3)
        max_delta_tool_quat = 2 * np.ones(4)
        max_delta_tool_zrot = 2 * np.ones(2)
        max_delta_pot_pos = 0.05 * np.ones(3)
        max_delta_pot_quat = 2 * np.ones(4)
        max_delta_pot_zrot = 0.2 * np.ones(2)
        max_delta_marker_pos = 0.05 * np.ones(3)

        obs_delta_range = {"robot0_eef_pos": [-max_delta_eef_pos, max_delta_eef_pos],
                           "robot0_eef_vel": [-max_delta_eef_vel, max_delta_eef_vel],
                           "robot0_joint_vel": [-max_delta_joint_vel, max_delta_joint_vel],
                           "robot0_gripper_qpos": [-max_delta_gripper_qpos, max_delta_gripper_qpos],
                           "robot0_gripper_qvel": [-max_delta_gripper_qvel, max_delta_gripper_qvel],
                           "cube_pos": [-max_delta_cube_pos, max_delta_cube_pos],
                           "cube_quat": [-max_delta_cube_quat, max_delta_cube_quat],
                           "cube_zrot": [-max_delta_cube_zrot, max_delta_cube_zrot],
                           "tool_pos": [-max_delta_tool_pos, max_delta_tool_pos],
                           "tool_quat": [-max_delta_tool_quat, max_delta_tool_quat],
                           "tool_zrot": [-max_delta_tool_zrot, max_delta_tool_zrot],
                           "pot_pos": [-max_delta_pot_pos, max_delta_pot_pos],
                           "pot_quat": [-max_delta_pot_quat, max_delta_pot_quat],
                           "pot_zrot": [-max_delta_pot_zrot, max_delta_pot_zrot]}
        for i in range(self.num_markers):
            obs_delta_range["marker{}_pos".format(i)] = [-max_delta_marker_pos, max_delta_marker_pos]
        return obs_delta_range
