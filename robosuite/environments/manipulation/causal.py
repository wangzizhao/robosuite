from collections import OrderedDict
import numpy as np
from scipy.spatial.transform import Rotation as R

from robosuite.utils.transform_utils import convert_quat
from robosuite.utils.mjcf_utils import CustomMaterial

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv

from robosuite.models.arenas import MarkerArena
from robosuite.models.objects import BoxObject, BallObject, CylinderObject, CapsuleObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler, SequentialCompositeSampler
from robosuite.utils.observables import Observable, sensor

from robosuite.utils.transform_utils import quat_multiply


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
        num_movable_objects=1,
        num_unmovable_objects=1,
        num_random_objects=1,
        num_markers=5,
    ):
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        # reward configuration
        self.reward_scale = reward_scale

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        self.num_movable_objects = num_movable_objects
        self.num_unmovable_objects = num_unmovable_objects
        self.num_random_objects = num_random_objects
        self.num_markers = num_markers

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
        # check collision or out of workspace
        eef_x, eef_y, eef_z = self.sim.data.site_xpos[self.robots[0].eef_site_id]
        table_len_x, table_len_y, _ = self.table_full_size
        table_offset_z = self.table_offset[2]
        workspace_x = [-table_len_x / 2, table_len_x / 2]
        workspace_y = [-table_len_y / 2, table_len_y / 2]
        workspace_z = [table_offset_z, table_offset_z + 1]
        out_of_workspace = (eef_x < workspace_x[0]) or (eef_x > workspace_x[1]) or \
                           (eef_y < workspace_y[0]) or (eef_y > workspace_y[1]) or \
                           (eef_z < workspace_z[0]) or (eef_z > workspace_z[1])
        reward = -self.reward_scale if out_of_workspace else 0
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
                    x_range=[-0.2, 0.1],
                    y_range=[-0.3, -0.1],
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

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

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
        def object_euler(obs_cache):
            quat = convert_quat(np.array(self.sim.data.body_xquat[self.sim.model.body_name2id(object.root_body)]),
                                to="xyzw")
            return R.from_quat(quat).as_euler('xyz')

        sensors = [object_pos, object_euler]
        names = [f"{prefix}{i}_pos", f"{prefix}{i}_euler"]

        return sensors, names

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        sensors = []
        names = []
        # low-level object information
        if self.use_object_obs:
            modality = "object"

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
                )

        return observables

    def _get_observations(self, force_update=False):
        observations = super()._get_observations(force_update)
        new_observations = OrderedDict()
        for k, v in observations.items():
            if k.endswith("quat"):
                new_key = k.replace("quat", "euler")
                if np.linalg.norm(v) == 0:
                    new_observations[new_key] = np.zeros(3)
                else:
                    new_observations[new_key] = R.from_quat(v).as_euler('xyz')
            else:
                new_observations[k] = v
        return new_observations

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

    def step(self, action):
        self.model.mujoco_arena.step_arena(self.sim)
        return super().step(action)
