import numpy as np
from robosuite.models.objects import CompositeObject
from robosuite.utils.mjcf_utils import CustomMaterial, add_to_dict
import robosuite.utils.transform_utils as T


class LShapeTool(CompositeObject):
    def __init__(
            self,
            name,
            density=1000,
            use_texture=True):

        self._name = name

        self.use_texture = use_texture

        base_args = {
            "total_size": 0 / 2.0,
            "name": self.name,
            "locations_relative_to_center": True,
            "obj_types": "all",
        }
        site_attrs = []
        obj_args = {}

        geom_mat = "MatRedWood"
        geom_frictions = (1, 0.005, 0.0001)

        solref = (0.02, 1.)
        
        add_to_dict(
            dic=obj_args,
            geom_types="box",
            geom_locations=(0.05, 0.0, 0.0),
            geom_quats=T.convert_quat(T.axisangle2quat(np.array([0, 0, 0])), to="wxyz"),
            geom_sizes=np.array([0.11, 0.01, 0.02]),
            geom_names=f"body_0",
            geom_rgbas=None,
            geom_materials=geom_mat,
            geom_frictions=geom_frictions,
            solref=solref,
            density=density)

        add_to_dict(
            dic=obj_args,
            geom_types="box",
            geom_locations=(0.16, 0.05, 0.0),
            geom_quats=T.convert_quat(T.axisangle2quat(np.array([0, 0, 0])), to="wxyz"),
            geom_sizes=np.array([0.01, 0.06, 0.02]),
            geom_names=f"body_1",
            geom_rgbas=None,
            geom_materials=geom_mat,
            geom_frictions=geom_frictions,
            solref=solref,
            density=density)

        obj_args.update(base_args)

        obj_args["sites"] = site_attrs
        obj_args["joints"] = [{"type": "free", "damping":"0.0005"}]

        super().__init__(**obj_args)

        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "3 3",
            "specular": "0.4",
            "shininess": "0.1",
        }
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="red_wood",
            mat_name="MatRedWood",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )

        self.append_material(redwood)

    @property
    def bottom_offset(self):
        return np.array([0, 0, -0.02])

    @property
    def top_offset(self):
        return np.array([0, 0, 0.02])

    @property
    def horizontal_radius(self):
        return 0  # np.sqrt(0.11 ** 2 + 0.17 ** 2)
