"""
Kinematics for the 5-link planar biped walker.
Computes joint positions and draws the robot skeleton.
"""

import numpy as np  # type: ignore
from dynamics import com_positions_fn, skeleton_fn  # type: ignore


def get_skeleton(q_val):

    return tuple(np.array(p, dtype=float).reshape(2) for p in skeleton_fn(q_val))


def get_com_positions(q_val):

    return tuple(np.array(p, dtype=float).reshape(2) for p in com_positions_fn(q_val))


def draw_biped(q_val, ax, x_offset=0.0):

    P3, P1, P5, P2, P4 = get_skeleton(q_val)
    off = np.array([x_offset, 0.0])
    P3, P1, P5, P2, P4 = (
        P3 + off,
        P1 + off,
        P5 + off,
        P2 + off,
        P4 + off,
    )

    # Supporting leg (blue)
    ax.plot(*zip(P3, P1, P5), "b-o", lw=2)
    # Torso (black)
    # ax.plot(*zip(Ph, ), "k-o", lw=3)
    # Free leg (red)
    ax.plot(*zip(P5, Pk2, P4), "r-o", lw=2)

    ax.set_aspect("equal")
    ax.axhline(0, color="gray", lw=1, linestyle="--")
