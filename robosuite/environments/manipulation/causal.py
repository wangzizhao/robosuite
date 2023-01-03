from collections import OrderedDict
import numpy as np
from scipy.spatial.transform import Rotation as R

from robosuite.utils.transform_utils import convert_quat
from robosuite.utils.mjcf_utils import CustomMaterial

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv

from robosuite.models.arenas import MarkerArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler, SequentialCompositeSampler
from robosuite.utils.observables import Observable, sensor


class Causal(SingleArmEnv):
    """
    This class corresponds to the stacking task for a single robot arm.

    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be a single single-arm robot!

        env_configuration (str): Specifies how to position the robots within the environment (default is "default").
            For most single arm environments, this argument has no impact on the robot setup.

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        table_full_size (3-tuple): x, y, and z dimensions of the table.

        table_friction (3-tuple): the three mujoco friction parameters for
            the table.

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object (cube) information in
            the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        placement_initializer (ObjectPositionSampler): if provided, will
            be used to place objects on every reset, else a UniformRandomSampler
            is used by default.

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

    Raises:
        AssertionError: [Invalid number of robots specified]
    """

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
        sparse_reward=False,
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
        num_movable_objects=1,
        num_unmovable_objects=1,
        num_random_objects=1,
        num_markers=5,
        cube_x_range=(-0.3, 0.3),
        cube_y_range=(-0.3, 0.3),
        marker_x_range=(-0.3, 0.3),
        marker_y_range=(-0.3, 0.3),
        normalization_range=((-0.5, -0.5, 0.7), (0.5, 0.5, 1.1))
    ):
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array(table_offset)

        # reward configuration
        self.reward_scale = reward_scale
        self.sparse_reward = sparse_reward

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        self.num_movable_objects = num_movable_objects
        self.num_unmovable_objects = num_unmovable_objects
        self.num_random_objects = num_random_objects
        self.num_markers = num_markers

        self.cube_x_range = cube_x_range
        self.cube_y_range = cube_y_range
        self.marker_x_range = marker_x_range
        self.marker_y_range = marker_y_range

        # global position range for normalization
        global_low, global_high = normalization_range
        self.global_low = np.array(global_low)
        self.global_high = np.array(global_high)
        self.global_mean = (self.global_high + self.global_low) / 2
        self.global_scale = (self.global_high - self.global_low) / 2

        # eef velocity range for normalization
        self.eef_vel_low = np.array([-2, -2, -2])
        self.eef_vel_high = np.array([2, 2, 2])
        self.eef_vel_mean = (self.eef_vel_high + self.eef_vel_low) / 2
        self.eef_vel_scale = (self.eef_vel_high - self.eef_vel_low) / 2

        # gripper angle range for normalization
        self.gripper_qpos_low = np.array([-0.03, -0.03])
        self.gripper_qpos_high = np.array([0.03, 0.03])
        self.gripper_qpos_mean = (self.gripper_qpos_high + self.gripper_qpos_low) / 2
        self.gripper_qpos_scale = (self.gripper_qpos_high - self.gripper_qpos_low) / 2

        # gripper angular velocity range for normalization
        self.gripper_qvel_low = np.array([-0.5, -0.5])
        self.gripper_qvel_high = np.array([0.5, 0.5])
        self.gripper_qvel_mean = (self.gripper_qvel_high + self.gripper_qvel_low) / 2
        self.gripper_qvel_scale = (self.gripper_qvel_high - self.gripper_qvel_low) / 2

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

    def reward(self, action):
        return 0

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
            num_unmovable_objects=self.num_unmovable_objects,
            num_random_objects=self.num_random_objects,
            num_markers=self.num_markers,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # initialize objects of interest
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )

        self.movable_objects = []
        for i in range(self.num_movable_objects):
            movable_obj_name = "movable{}".format(i)
            movable_obj = BoxObject(
                name=movable_obj_name,
                density=1000,   # default
                size=[0.02, 0.02, 0.02],
                rgba=[1, 0, 0, 1],
                material=redwood,
            )
            self.movable_objects.append(movable_obj)

        # Create placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.samplers["MovableObjectSampler"].add_objects(self.movable_objects)
        else:
            self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
            self.placement_initializer.append_sampler(
                UniformRandomSampler(
                    name="MovableObjectSampler",
                    mujoco_objects=self.movable_objects,
                    x_range=self.cube_x_range,
                    y_range=self.cube_x_range,
                    rotation=[-np.pi, np.pi],
                    rotation_axis='z',
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=True,
                    reference_pos=self.table_offset,
                    z_offset=0.01,
                )
            )

        self.objects = self.movable_objects + mujoco_arena.unmovable_objects + mujoco_arena.random_objects
        self.sensor_prefixs = [f"mov{i}" for i in range(self.num_movable_objects)] + \
                              [f"unmov{i}" for i in range(self.num_unmovable_objects)] + \
                              [f"rand{i}" for i in range(self.num_random_objects)]
        self.movable_sensor_prefixs = [f"mov{i}" for i in range(self.num_movable_objects)]

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.movable_objects,
        )

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        if not self.deterministic_reset:
            self.model.mujoco_arena.reset_arena(self.sim)

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

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

    def _create_object_sensors(self, i, object, prefix="obj", modality="object"):
        @sensor(modality=modality)
        def object_pos(obs_cache):
            return np.array(self.sim.data.body_xpos[self.sim.model.body_name2id(object.root_body)])

        @sensor(modality=modality)
        def object_quat(obs_cache):
            quat = convert_quat(np.array(self.sim.data.body_xquat[self.sim.model.body_name2id(object.root_body)]),
                                to="xyzw")
            return quat

        @sensor(modality=modality)
        def object_euler(obs_cache):
            quat = convert_quat(np.array(self.sim.data.body_xquat[self.sim.model.body_name2id(object.root_body)]),
                                to="xyzw")
            return R.from_quat(quat).as_euler('xyz')

        @sensor(modality=modality)
        def object_zrot(obs_cache):
            quat = convert_quat(np.array(self.sim.data.body_xquat[self.sim.model.body_name2id(object.root_body)]),
                                to="xyzw")
            z = R.from_quat(quat).as_euler('xyz')[2]
            return np.array([np.sin(z), np.cos(z)]).astype(np.float32)

        @sensor(modality=modality)
        def object_grasped(obs_cache):
            grasped = int(self._check_grasp(gripper=self.robots[0].gripper,
                                            object_geoms=object))
            return grasped

        @sensor(modality=modality)
        def object_touched(obs_cache):
            touched = int(self.check_contact(self.robots[0].gripper, object))
            return touched

        sensors = [object_pos, object_quat, object_grasped, object_touched]
        names = [f"{prefix}{i}_pos", f"{prefix}{i}_quat", f"{prefix}{i}_grasped", f"{prefix}{i}_touched"]

        return sensors, names

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

            for i, movable_obj in enumerate(self.movable_objects):
                movable_sensors, movable_sensor_names = self._create_object_sensors(i, movable_obj, "mov", modality)
                sensors += movable_sensors
                names += movable_sensor_names

            for i, unmovable_obj in enumerate(self.model.mujoco_arena.unmovable_objects):
                unmovable_sensors, unmovable_sensor_names = \
                    self._create_object_sensors(i, unmovable_obj, "unmov", modality)
                sensors += unmovable_sensors
                names += unmovable_sensor_names

            for i, random_obj in enumerate(self.model.mujoco_arena.random_objects):
                random_sensors, random_sensor_names = self._create_object_sensors(i, random_obj, "rand", modality)
                sensors += random_sensors
                names += random_sensor_names

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
                    enabled="euler" not in name
                )

        return observables

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the cube.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

    def normalize_obs(self, obs):
        for k, v in obs.items():
            if k in ["robot0_eef_pos", "goal_pos"] or (k.startswith(("mov", "unmov", "rand", "marker")) and k.endswith("pos")):
                if not ((v >= self.global_low) & (v <= self.global_high)).all():
                    print(k, "out of range", v, self.global_low, self.global_high)
                    exit()
                obs[k] = (v - self.global_mean) / self.global_scale
            elif k == "robot0_eef_vel":
                if not ((v >= self.eef_vel_low) & (v <= self.eef_vel_high)).all():
                    print(k, "out of range", v)
                    exit()
                obs[k] = (v - self.eef_vel_mean) / self.eef_vel_scale
            elif k == "robot0_gripper_qpos":
                if not ((v >= self.gripper_qpos_low) & (v <= self.gripper_qpos_high)).all():
                    print(k, "out of range", v)
                    exit()
                obs[k] = (v - self.gripper_qpos_mean) / self.gripper_qpos_scale
            elif k == "robot0_gripper_qvel":
                if not ((v >= self.gripper_qvel_low) & (v <= self.gripper_qvel_high)).all():
                    print(k, "out of range", v)
                    exit()
                obs[k] = (v - self.gripper_qvel_mean) / self.gripper_qvel_scale
        return obs

    def reset(self):
        obs = super().reset()
        obs = self.normalize_obs(obs)
        return obs

    def step(self, action):
        # clip action to avoid that eef goes out of workspace
        assert action.shape == (4,)

        global_act_low, global_act_high = np.array([-0.3, -0.4, 0.81]), np.array([0.3, 0.4, 1.0])
        eef_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
        controller_scale = 0.05
        action[:3] = np.clip(action[:3],
                             (global_act_low - eef_pos) / controller_scale,
                             (global_act_high - eef_pos) / controller_scale)

        self.model.mujoco_arena.step_arena(self.sim)
        next_obs, reward, done, info = super().step(action)
        next_obs = self.normalize_obs(next_obs)

        info["success"] = False
        return next_obs, reward, done, info

    def obs_delta_range(self):
        max_delta_eef_pos = 0.1 * np.ones(3) / (2 * self.global_scale)
        max_delta_eef_vel = 2 * np.ones(3) / (2 * self.eef_vel_scale)
        max_delta_gripper_qpos = 0.02 * np.ones(2) / (2 * self.gripper_qpos_scale)
        max_delta_gripper_qvel = 0.5 * np.ones(2) / (2 * self.gripper_qvel_scale)
        max_delta_obj_pos = 0.1 * np.ones(3) / (2 * self.global_scale)
        max_delta_obj_quat = 2 * np.ones(4)
        max_delta_obj_zrot = 2 * np.ones(2)
        max_delta_heavy_obj_pos = 0.05 * np.ones(3) / (2 * self.global_scale)
        max_delta_heavy_obj_quat = 0.2 * np.ones(4)
        max_delta_heavy_obj_zrot = 0.2 * np.ones(2)
        max_delta_marker_pos = 0.05 * np.ones(3) / (2 * self.global_scale)

        obs_delta_range = {"robot0_eef_pos": [-max_delta_eef_pos, max_delta_eef_pos],
                           "robot0_eef_vel": [-max_delta_eef_vel, max_delta_eef_vel],
                           "robot0_gripper_qpos": [-max_delta_gripper_qpos, max_delta_gripper_qpos],
                           "robot0_gripper_qvel": [-max_delta_gripper_qvel, max_delta_gripper_qvel]}
        for i in range(self.num_movable_objects):
            obs_delta_range["mov{}_pos".format(i)] = [-max_delta_obj_pos, max_delta_obj_pos]
            obs_delta_range["mov{}_quat".format(i)] = [-max_delta_obj_quat, max_delta_obj_quat]
            obs_delta_range["mov{}_zrot".format(i)] = [-max_delta_obj_zrot, max_delta_obj_zrot]
        for i in range(self.num_unmovable_objects):
            obs_delta_range["unmov{}_pos".format(i)] = [-max_delta_heavy_obj_pos, max_delta_heavy_obj_pos]
            obs_delta_range["unmov{}_quat".format(i)] = [-max_delta_heavy_obj_quat, max_delta_heavy_obj_quat]
            obs_delta_range["unmov{}_zrot".format(i)] = [-max_delta_heavy_obj_zrot, max_delta_heavy_obj_zrot]
        for i in range(self.num_random_objects):
            obs_delta_range["rand{}_pos".format(i)] = [-max_delta_heavy_obj_pos, max_delta_heavy_obj_pos]
            obs_delta_range["rand{}_quat".format(i)] = [-max_delta_heavy_obj_quat, max_delta_heavy_obj_quat]
            obs_delta_range["rand{}_zrot".format(i)] = [-max_delta_heavy_obj_zrot, max_delta_heavy_obj_zrot]
        for i in range(self.num_markers):
            obs_delta_range["marker{}_pos".format(i)] = [-max_delta_marker_pos, max_delta_marker_pos]
        return obs_delta_range
