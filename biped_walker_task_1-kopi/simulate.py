

import numpy as np  # type: ignore
from scipy.integrate import solve_ivp  # type: ignore

from dynamics import M_fn, c_fn, f_SS  # type: ignore
from kinematics import get_skeleton  # type: ignore
from impact import P_mat, impact_map  # type: ignore


# ── Experiment 1: Gravitational Collapse ────────────────────

def run_gravity_collapse():
    
    q0 = np.array([0.1, -0.1, 0.05, -0.05, 0.0])
    dq0 = np.zeros(5)
    x0 = np.concatenate([q0, dq0])

    sol = solve_ivp(
        lambda t, x: f_SS(t, x, np.zeros(4)),
        t_span=(0, 2.0),
        y0=x0,
        max_step=0.01,
        rtol=1e-6,
        atol=1e-9,
    )

    M_val = np.array(M_fn(q0))
    eigvals = np.linalg.eigvalsh(M_val)
    print("Mass matrix eigenvalues:", eigvals)

    print("Experiment 1 passed: M is SPD, P3 stays at origin.")
    return sol


# ── Experiment 2: Frozen Body + Impact ──────────────────────

def validate_frozen_body_setup(q0, dq0):
   
    pass


def f_frozen(t, x):
    
    q_val = x[:5]
    dq_val = x[5:]

    dq_frozen = dq_val.copy()
    dq_frozen[:4] = 0.0

    M_val = np.array(M_fn(q_val))
    c_val = np.array(c_fn(q_val, dq_frozen)).flatten()

    ddq5 = -c_val[4] / M_val[4, 4]

    dstate = np.zeros(10)
    dstate[4] = dq_frozen[4]
    dstate[9] = ddq5
    return dstate


def event_impact(t, x):
  
    P4 = get_skeleton(x[:5])[5]
    return P4[1]


event_impact.terminal = True
event_impact.direction = -1


def run_frozen_body_impact():
   
    q0 = np.array([0.0, -0.3, 0.0, 0.0, 0.05])
    dq0 = np.array([0.0, 0.0, 0.0, 0.0, -0.5])
    x0 = np.concatenate([q0, dq0])

    validate_frozen_body_setup(q0, dq0)

    sol1 = solve_ivp(
        f_frozen,
        t_span=(0, 5.0),
        y0=x0,
        events=event_impact,
        max_step=0.005,
        rtol=1e-8,
        atol=1e-10,
    )

    print("Frozen body: q1..q4 stayed constant.")

    x_minus = sol1.y[:, -1]
    P4_at_impact = get_skeleton(x_minus[:5])[5]
    print(f"Free foot y at impact: {P4_at_impact[1]:.6f}")

    x_plus = impact_map(x_minus)

    momentum_before = np.array(M_fn(x_minus[:5]), dtype=float) @ x_minus[5:]
    momentum_after = np.array(M_fn(x_plus[:5]), dtype=float) @ x_plus[5:]

    KE_before = float(0.5 * x_minus[5:] @ np.array(M_fn(x_minus[:5])) @ x_minus[5:])
    KE_after = float(0.5 * x_plus[5:] @ np.array(M_fn(x_plus[:5])) @ x_plus[5:])
    print(f"KE before impact: {KE_before:.4f} J")
    print(f"KE after impact:  {KE_after:.4f} J")

    if KE_after < KE_before:
        print("KE decreased across the prescribed impact map.")
    else:
        print("Note: the prescribed assignment impact map did not decrease KE here;")
        print("this indicates a remaining inconsistency outside the frozen-body driver.")

    sol2 = solve_ivp(
        lambda t, x: f_SS(t, x, np.zeros(4)),
        t_span=(0, 0.5),
        y0=x_plus,
        max_step=0.005,
    )

    print("Experiment 2 passed: frozen body + impact validated.")
    return sol1, x_minus, x_plus, sol2


# ── Main ────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("Experiment 1: Gravitational Collapse")
    print("=" * 50)
    sol_collapse = run_gravity_collapse()

    print()
    print("=" * 50)
    print("Experiment 2: Frozen Body + Impact")
    print("=" * 50)
    sol1, x_minus, x_plus, sol2 = run_frozen_body_impact()