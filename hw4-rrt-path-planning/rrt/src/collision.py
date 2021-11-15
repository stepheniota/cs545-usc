from abc import ABC, abstractmethod
from matplotlib.pyplot import disconnect

import numpy as np


class CollisionObject(ABC):
    """
    Abstract class for a parametrically defined collision object.
    """
    @abstractmethod
    def in_collision(self, target):
        """
        Checks whether target point is in collision. Points at the boundary of
        the object are in collision.

        :returns: Boolean indicating target is in collision.
        """
        pass


class CollisionBox(CollisionObject):
    """
    N-dimensional box collision object.
    """
    def __init__(self, location, half_lengths):
        """
        :params location: coordinates of the center
        :params half_lengths: half-lengths of the rectangle along each axis
        """
        self.location = np.asarray(location)
        self.half_lengths = np.asarray(half_lengths)
        self.ndim = self.location.shape[0]

    def in_collision(self, target):
        return min(
            [np.abs(self.location[i] - target[i]) <= self.half_lengths[i]
             for i in range(self.ndim)]
        )


class CollisionSphere(CollisionObject):
    """
    N-dimensional sphere collision object.
    """
    def __init__(self, location, radius):
        """
        :params location: coordinates of the center
        :params radius: radius of the circle
        """
        self.location = np.asarray(location)
        self.radius = radius

    def in_collision(self, target):
        dist_to_center = np.linalg.norm(target - self.location)
        return dist_to_center <= self.radius
