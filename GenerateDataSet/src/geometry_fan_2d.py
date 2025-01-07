import numpy as np
from src import geometry_base


class GeometryFan2D(geometry_base.GeometryBase):
    """
        2D Fan specialization of Geometry.
    """

    def __init__(self,
                 volume_shape, volume_spacing,
                 detector_shape, detector_spacing,
                 number_of_projections, proj_range_start, proj_range_end,
                 source_detector_distance, source_isocenter_distance):
        # init base Geometry class with 2 dimensional members:
        super().__init__(volume_shape, volume_spacing,
                         [detector_shape], [detector_spacing],
                         number_of_projections, proj_range_start, proj_range_end,
                         source_detector_distance, source_isocenter_distance)

        # defined by geometry so calculate for convenience use
        self.fan_angle = np.arctan(((self.detector_shape[0] - 1) / 2.0 * self.detector_spacing[0]) / self.source_detector_distance)

    def set_trajectory(self, central_ray_vectors):
        """
            Sets the member central_ray_vectors.
        Args:
            central_ray_vectors: np.array defining the trajectory central_ray_vectors.
        """
        self.central_ray_vectors = np.array(central_ray_vectors, self.np_dtype)