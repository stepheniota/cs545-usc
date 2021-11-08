import numpy as np

#def rotation_matix(angle, length):
#    R = np.array(
#            [[np.cos(angle), -np.sin(angle), 0, length * np.cos(angle)],
#             [np.sin(angle), np.cos(angle), 0, length * np.sin(angle)],
#             [0, 0, 1, 0],
#             [0, 0, 0, 1]],
#            dtype=np.float64
#        )
#    return R

def fk(angles, link_lengths):
    """
    Computes the forward kinematics of a planar, n-joint robot arm.

    Given below is an illustrative example. Note the end effector frame is at
    the tip of the last link.

        q[0]   l[0]   q[1]   l[1]   end_eff
          O-------------O--------------C

    you would call:
        fk(q, l)

    :param angles: list of angle values for each joint, in radians.
    :param link_lengths: list of lengths for each link in the robot arm.
    :returns: The end effector position (not pose!) with respect to the base
        frame (the frame at the first joint) as a numpy array with dtype
        np.float64
    """
    end_eff = np.zeros(3, dtype=np.float64)

    for (q, elle) in zip(angles, link_lengths):
        end_eff[0] += np.multiply(elle, np.cos(q))
        end_eff[1] += np.multiply(elle, np.sin(q))

    assert end_eff.dtype == np.float64

    return end_eff


if __name__ == '__main__':
    np.set_printoptions(suppress=True)

    print("A:")
    print(fk([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]))
    print("B:")
    print(fk([0.3, 0.4, 0.8], [0.8, 0.5, 1.0]))
    print("C:")
    print(fk([1.0, 0.0, 0.0], [3.0, 1.0, 1.0]))
