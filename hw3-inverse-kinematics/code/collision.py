import numpy as np


def line_sphere_intersection(p1, p2, c, r):
    """
    Implements the line-sphere intersection algorithm.
    https://en.wikipedia.org/wiki/Line-sphere_intersection

    :param p1: start of line segment
    :param p2: end of line segment
    :param c: sphere center
    :param r: sphere radius
    :returns: discriminant (value under the square root) of the line-sphere
        intersection formula, as a np.float64 scalar
    """

    u_hat = np.subtract(p2, p1) / np.linalg.norm(np.subtract(p2, p1))  
    discriminant = np.square(np.dot(u_hat, np.subtract(p1, c)))
    discriminant += np.square(r)
    discriminant -= np.square(np.linalg.norm(np.subtract(p1, c)))

    assert type(discriminant) == np.float64

    #print(discriminant)

    return discriminant