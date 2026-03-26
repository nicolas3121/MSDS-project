

import numpy as np # type: ignore
from dynamics import M_fn  # type: ignore

# Permutation matrix: swap supporting <-> free legs
P_mat = np.array([
    [0, 1, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1],
], dtype=float)


def impact_map(x_minus):
   
    q_minus  = x_minus[:5]
    dq_minus = x_minus[5:]

    q_plus  = P_mat @ q_minus
    M_minus = np.array(M_fn(q_minus), dtype=float)
    M_plus  = np.array(M_fn(q_plus), dtype=float)
    dq_plus = np.linalg.solve(M_plus, M_minus @ dq_minus)

    return np.concatenate([q_plus, dq_plus])
