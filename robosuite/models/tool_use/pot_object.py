import numpy as np
from robosuite.models.objects import CompositeObject
from robosuite.utils.mjcf_utils import array_to_string, CustomMaterial, add_to_dict
import robosuite.utils.transform_utils as T


class PotObject(CompositeObject):
    def __init__(
            self,
            name,
            pot_size=(0.06, 0.06, 0.03),
            bottom_half_thickness=0.005,
            side_half_thickness=0.0075,
            handle_half_height=0.01,
            handle_half_length=0.04,
            density=1000,
            use_texture=True):

        self._name = name

        pot_half_x_length, pot_half_y_length, pot_half_height = pot_size
        self.pot_half_x_length = pot_half_x_length
        self.pot_half_y_length = pot_half_y_length
        self.pot_half_height = pot_half_height
        self.bottom_half_thickness = bottom_half_thickness
        self.side_half_thickness = side_half_thickness
        self.handle_half_height = handle_half_height

        self.use_texture = use_texture

        base_args = {
            "total_size": 0.0,
            "name": self.name,
            "locations_relative_to_center": True,
            "obj_types": "all",
        }
        site_attrs = []
        obj_args = {}

        geom_mat = "steel_scratched_mat"
        geom_frictions = (1, 0.005, 0.0001)
        solref = (0.02, 1.)

        # bottom
        add_to_dict(
            dic=obj_args,
            geom_types="box",
            geom_locations=(0., 0., bottom_half_thickness),
            geom_quats=T.convert_quat(T.axisangle2quat(np.array([0, 0, 0])), to="wxyz"),
            geom_sizes=np.array([pot_half_x_length - side_half_thickness,
                                 pot_half_y_length - side_half_thickness,
                                 bottom_half_thickness]),
            geom_names=f"body_0",
            geom_rgbas=None,
            geom_materials=geom_mat,
            geom_frictions=geom_frictions,
            solref=solref,
        )

        add_to_dict(
            dic=obj_args,
            geom_types="box",
            geom_locations=(0, -pot_half_y_length, pot_half_height),
            geom_quats=T.convert_quat(T.axisangle2quat(np.array([0, 0, 0])), to="wxyz"),
            geom_sizes=np.array([pot_half_x_length + side_half_thickness, side_half_thickness, pot_half_height]),
            geom_names=f"body_1",
            geom_rgbas=None,
            geom_materials=geom_mat,
            geom_frictions=geom_frictions,
            solref=solref,
            density=density)

        add_to_dict(
            dic=obj_args,
            geom_types="box",
            geom_locations=(0, pot_half_y_length, pot_half_height),
            geom_quats=T.convert_quat(T.axisangle2quat(np.array([0, 0, 0])), to="wxyz"),
            geom_sizes=np.array([pot_half_x_length + side_half_thickness, side_half_thickness, pot_half_height]),
            geom_names=f"body_2",
            geom_rgbas=None,
            geom_materials=geom_mat,
            geom_frictions=geom_frictions,
            solref=solref,
            density=density)

        add_to_dict(
            dic=obj_args,
            geom_types="box",
            geom_locations=(pot_half_x_length, 0, pot_half_height),
            geom_quats=T.convert_quat(T.axisangle2quat(np.array([0, 0, 0])), to="wxyz"),
            geom_sizes=np.array([side_half_thickness, pot_half_y_length + side_half_thickness, pot_half_height]),
            geom_names=f"body_3",
            geom_rgbas=None,
            geom_materials=geom_mat,
            geom_frictions=geom_frictions,
            solref=solref,
            density=density)

        add_to_dict(
            dic=obj_args,
            geom_types="box",
            geom_locations=(-pot_half_x_length, 0, pot_half_height),
            geom_quats=T.convert_quat(T.axisangle2quat(np.array([0, 0, 0])), to="wxyz"),
            geom_sizes=np.array([side_half_thickness, pot_half_y_length + side_half_thickness, pot_half_height]),
            geom_names=f"body_4",
            geom_rgbas=None,
            geom_materials=geom_mat,
            geom_frictions=geom_frictions,
            solref=solref,
            density=density)

        handle_friction = 1.0

        for (direction, y) in zip(['left', 'right'], [pot_half_y_length, -pot_half_y_length]):
            add_to_dict(
                dic=obj_args,
                geom_types="box",
                geom_locations=(0.0, y, 2 * pot_half_height + handle_half_height),
                geom_quats=T.convert_quat(T.axisangle2quat(np.array([0, 0, 0])), to="wxyz"),
                geom_sizes=np.array([handle_half_length, side_half_thickness, handle_half_height]),
                geom_names=f"handle_{direction}_0",
                geom_rgbas=None,
                geom_materials=geom_mat,
                geom_frictions=(handle_friction, 0.005, 0.0001),
                solref=solref,                                
                density=density)

        obj_args.update(base_args)

        obj_args["sites"] = site_attrs
        obj_args["joints"] = [{"type": "free", "damping": "0.0005"}]

        super().__init__(**obj_args)

        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "3 3",
            "specular": "0.4",
            "shininess": "0.1",
        }
        steel_scratched_material = CustomMaterial(
            texture="SteelScratched",
            tex_name="steel_scratched_tex",
            mat_name="steel_scratched_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
 
        self.append_material(steel_scratched_material)

    @property
    def bottom_offset(self):
        return np.array([0, 0, -self.bottom_half_thickness])

    @property
    def top_offset(self):
        return np.array([0, 0, self.bottom_half_thickness + 2 * self.pot_half_height + 2 * self.handle_half_height])

    @property
    def horizontal_radius(self):
        return np.linalg.norm([self.pot_half_x_length + self.side_half_thickness,
                               self.pot_half_y_length + self.side_half_thickness])
