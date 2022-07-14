import numpy as np
from robosuite.models.arenas import TableArena
from robosuite.utils.mjcf_utils import CustomMaterial, find_elements
from robosuite.models.objects import CylinderObject, BoxObject, BallObject
from robosuite.utils.placement_samplers import UniformRandomSampler, SequentialCompositeSampler
from robosuite.utils.transform_utils import quat_multiply, convert_quat


class MarkerArena(TableArena):
    """
    Workspace that contains an empty table with visual markers on its surface.

    Args:
        table_full_size (3-tuple): (L,W,H) full dimensions of the table
        table_friction (3-tuple): (sliding, torsional, rolling) friction parameters of the table
        table_offset (3-tuple): (x,y,z) offset from center of arena when placing table.
            Note that the z value sets the upper limit of the table
    """

    def __init__(
        self,
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(0.01, 0.005, 0.0001),
        table_offset=(0, 0, 0.8),
        num_unmovable_objects=1,
        num_random_objects=1,
        num_markers=5,
        placement_initializer=None
    ):
        self.num_unmovable_objects = num_unmovable_objects
        self.num_random_objects = num_random_objects
        self.num_markers = num_markers
        self.placement_initializer = placement_initializer
        self.unmovable_objects = []
        self.random_objects = []
        self.markers = []

        # run superclass init
        super().__init__(
            table_full_size=table_full_size,
            table_friction=table_friction,
            table_offset=table_offset,
        )

    def configure_location(self):
        """Configures correct locations for this arena"""
        # Run superclass first
        super().configure_location()

        # Define dirt material for markers
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.0",
            "shininess": "0.0",
        }
        dirt = CustomMaterial(
            texture="Dirt",
            tex_name="dirt",
            mat_name="dirt_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
            shared=True,
        )
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }
        greenwood = CustomMaterial(
            texture="WoodGreen",
            tex_name="greenwood",
            mat_name="greenwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        bluewood = CustomMaterial(
            texture="WoodBlue",
            tex_name="bluewood",
            mat_name="bluewood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        self.goal_vis = BallObject(
            name="goal",
            size=[0.015],
            rgba=[1, 1, 0, 1],
            material=redwood,
            obj_type="visual",
            joints=None,
        )
        self.merge_assets(self.goal_vis)
        table = find_elements(root=self.worldbody, tags="body", attribs={"name": "table"}, return_first=True)
        table.append(self.goal_vis.get_obj())

        # Define markers on table
        for i in range(self.num_markers):
            marker_name = "marker{}".format(i)
            marker = CylinderObject(
                name=marker_name,
                size=[0.02, 0.001],
                rgba=[1, 1, 1, 1],
                material=dirt,
                obj_type="visual",
                joints=None,
            )
            # Manually add this object to the arena xml
            self.merge_assets(marker)
            table = find_elements(root=self.worldbody, tags="body", attribs={"name": "table"}, return_first=True)
            table.append(marker.get_obj())

            # Add this marker to our saved list of all markers
            self.markers.append(marker)

        # Define unmovable objects on the table
        for i in range(self.num_unmovable_objects):
            unmovable_obj_name = "unmovable{}".format(i)
            unmovable_obj = BoxObject(
                name=unmovable_obj_name,
                size=[0.02, 0.02, 0.02],
                rgba=[0, 1, 0, 1],
                material=greenwood,
                joints=None,
            )
            self.merge_assets(unmovable_obj)
            table = find_elements(root=self.worldbody, tags="body", attribs={"name": "table"}, return_first=True)
            table.append(unmovable_obj.get_obj())

            self.unmovable_objects.append(unmovable_obj)

        for i in range(self.num_random_objects):
            random_obj_name = "random{}".format(i)
            random_obj = BoxObject(
                name=random_obj_name,
                size=[0.02, 0.02, 0.02],
                rgba=[0, 0, 1, 1],
                material=bluewood,
                joints=None,
            )
            self.merge_assets(random_obj)
            table = find_elements(root=self.worldbody, tags="body", attribs={"name": "table"}, return_first=True)
            table.append(random_obj.get_obj())

            self.random_objects.append(random_obj)

        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.samplers["UnmovableObjectSampler"].add_objects(self.unmovable_objects)
            self.placement_initializer.samplers["RandomObjectSampler"].add_objects(self.random_objects)
            self.placement_initializer.samplers["MarkerSampler"].add_objects(self.markers)
        else:
            self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
            self.placement_initializer.append_sampler(
                UniformRandomSampler(
                    name="MarkerSampler",
                    mujoco_objects=self.markers,
                    x_range=[-0.3, 0.3],
                    y_range=[-0.3, 0.3],
                    ensure_object_boundary_in_range=True,
                    ensure_valid_placement=False,
                    reference_pos=(0, 0, self.table_half_size[2]),
                    z_offset=1e-3,
                )
            )
            self.placement_initializer.append_sampler(
                UniformRandomSampler(
                    name="UnmovableObjectSampler",
                    mujoco_objects=self.unmovable_objects,
                    x_range=[-0.3, 0.3],
                    y_range=[-0.3, 0.3],
                    rotation=[-np.pi, np.pi],
                    rotation_axis='z',
                    ensure_object_boundary_in_range=True,
                    ensure_valid_placement=True,
                    reference_pos=(0, 0, self.table_half_size[2]),
                    z_offset=1e-3,
                )
            )
            self.placement_initializer.append_sampler(
                UniformRandomSampler(
                    name="RandomObjectSampler",
                    mujoco_objects=self.random_objects,
                    x_range=[-0.3, 0.3],
                    y_range=[-0.3, 0.3],
                    rotation=[-np.pi, np.pi],
                    rotation_axis='z',
                    ensure_object_boundary_in_range=True,
                    ensure_valid_placement=True,
                    reference_pos=(0, 0, self.table_half_size[2]),
                    z_offset=0.01,
                )
            )

    def reset_arena(self, sim):
        """
        Reset the visual marker locations in the environment. Requires @sim (MjSim) reference to be passed in so that
        the Mujoco sim can be directly modified

        Args:
            sim (MjSim): Simulation instance containing this arena and visual markers
        """

        # Sample from the placement initializer for all objects
        object_placements = self.placement_initializer.sample()

        # Loop through all objects and reset their positions
        for obj_pos, obj_quat, obj in object_placements.values():
            # Get IDs to the body, geom, and site of each marker
            body_id = sim.model.body_name2id(obj.root_body)
            geom_id = sim.model.geom_name2id(obj.visual_geoms[0])
            site_id = sim.model.site_name2id(obj.sites[0])
            # Set the current marker (body) to this new position and quaternion
            sim.model.body_pos[body_id] = obj_pos
            sim.model.body_quat[body_id] = obj_quat
            # Reset the marker visualization -- setting geom rgba alpha value to 1
            sim.model.geom_rgba[geom_id][3] = 1
            # Hide the default visualization site
            sim.model.site_rgba[site_id][3] = 0
        goal_id = sim.model.body_name2id(self.goal_vis.root_body)
        sim.model.body_pos[goal_id] = np.zeros(3)

    def step_arena(self, sim):
        """
        Step the random object in the environment. Requires @sim (MjSim) reference to be passed in so that
        the Mujoco sim can be directly modified

        Args:
            sim (MjSim): Simulation instance containing this arena and visual markers
        """
        x_range = self.placement_initializer.samplers["RandomObjectSampler"].x_range
        y_range = self.placement_initializer.samplers["RandomObjectSampler"].y_range
        for obj in self.random_objects:
            body_id = sim.model.body_name2id(obj.root_body)
            obj_pos = sim.model.body_pos[body_id]
            obj_quat = convert_quat(np.array(sim.model.body_quat[body_id]), to="xyzw")

            pos_noise = np.random.normal(0, 0.002, size=2)
            obj_pos[:2] = np.clip(obj_pos[:2] + pos_noise, (x_range[0], y_range[0]), (x_range[1], y_range[1]))

            rot_angle = np.pi / 180 * np.random.normal(0, 2)
            rot_quat = np.array([0, 0, np.sin(rot_angle / 2), np.cos(rot_angle / 2)])
            obj_quat = quat_multiply(rot_quat, obj_quat)

            sim.model.body_pos[body_id] = obj_pos
            sim.model.body_quat[body_id] = convert_quat(obj_quat, to="wxyz")

        x_range = self.placement_initializer.samplers["MarkerSampler"].x_range
        y_range = self.placement_initializer.samplers["MarkerSampler"].y_range
        z_range = [0.03, 0.05]
        for obj in self.markers:
            body_id = sim.model.body_name2id(obj.root_body)
            obj_pos = sim.model.body_pos[body_id]

            pos_noise = np.random.normal(0, 0.01, size=3)
            obj_pos = np.clip(obj_pos + pos_noise,
                              (x_range[0], y_range[0], z_range[0]),
                              (x_range[1], y_range[1], z_range[1]))

            sim.model.body_pos[body_id] = obj_pos
