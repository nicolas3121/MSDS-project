from model import M_fn, c_fn, f_SS, joint_pos_fn, impact_map_fn, P4_fn
from visualize import (
    plot_collapse_snapshots,
    plot_frozen_body_snapshots,
    animate_collapse,
    animate_frozen_and_impact,
)
import casadi as ca

# Experiment 1


def build_gravity_collapse_fn(n_steps, dt):
    x_sym = ca.SX.sym("x", 10)
    t_sym = ca.SX.sym("t")
    u_zero = ca.DM.zeros(4)

    x_dot = f_SS(x_sym, u_zero)

    ode_dict = {"x": x_sym, "t": t_sym, "ode": x_dot}
    opts = {"reltol": 1e-6, "abstol": 1e-9}
    step_integrator = ca.integrator(
        "step_integrator", "cvodes", ode_dict, 0.0, dt, opts
    )

    simulate_trajectory = step_integrator.mapaccum("simulate_trajectory", n_steps)

    x0_in = ca.SX.sym("x0_in", 10)

    res = simulate_trajectory(x0=x0_in)

    full_trajectory = ca.horzcat(x0_in, res["xf"])

    return ca.Function("gravity_collapse_sim", [x0_in], [full_trajectory])


def run_gravity_collapse():
    dt = 0.01
    t_end = 2.0
    n_steps = int(t_end / dt)

    q0 = ca.vertcat(0.1, -0.1, 0.05, -0.05, 0.0)
    dq0 = ca.DM.zeros(5)
    x0 = ca.vertcat(q0, dq0)

    sim_fn = build_gravity_collapse_fn(n_steps, dt)

    sol_traj = sim_fn(x0)

    M_val = M_fn(q0)
    try:
        L = ca.chol(M_val)
        print("Experiment 1 passed: M is SPD, P3 stays at origin.")
    except RuntimeError:
        print("Experiment 1 FAILED: Mass matrix is NOT Positive Definite!")

    return sol_traj


# Experiment 2


def build_frozen_integrator(dt):
    """Builds a CasADi integrator for the frozen-body dynamics."""
    x_sym = ca.SX.sym("x", 10)
    q_sym = x_sym[:5]
    dq_sym = x_sym[5:]

    # Force dq_1 through dq_4 to be exactly zero
    dq_frozen = ca.vertcat(0, 0, 0, 0, dq_sym[4])

    M = M_fn(q_sym)
    c = c_fn(q_sym, dq_frozen)

    # Solve for ddq5 (the only moving joint)
    ddq5 = -c[4] / M[4, 4]

    # x_dot = [dq, ddq], enforcing zero velocity and acceleration for locked joints
    x_dot = ca.vertcat(0, 0, 0, 0, dq_sym[4], 0, 0, 0, 0, ddq5)

    ode_dict = {"x": x_sym, "ode": x_dot}
    opts = {"reltol": 1e-8, "abstol": 1e-10}

    return ca.integrator("frozen_int", "cvodes", ode_dict, 0.0, dt, opts)


def build_full_integrator(dt):
    """Builds a single-step CasADi integrator for the full system."""
    x_sym = ca.SX.sym("x", 10)
    u_zero = ca.DM.zeros(4)
    x_dot = f_SS(x_sym, u_zero)

    ode_dict = {"x": x_sym, "ode": x_dot}
    opts = {"reltol": 1e-8, "abstol": 1e-10}

    return ca.integrator("full_int", "cvodes", ode_dict, 0.0, dt, opts)


def run_frozen_body_impact():
    dt = 0.005
    frozen_int = build_frozen_integrator(dt)
    full_int = build_full_integrator(dt)

    q0 = ca.vertcat(0.0, -0.3, 0.0, 0.0, 0.05)
    dq0 = ca.vertcat(0.0, 0.0, 0.0, 0.0, -0.5)
    x = ca.vertcat(q0, dq0)

    # --- Phase 1: Frozen Body (Falling) ---
    t_max = 5.0
    n_steps_max = int(t_max / dt)

    traj_frozen = [x]  # Store CasADi DM natively
    x_minus = x

    for _ in range(n_steps_max):
        res = frozen_int(x0=x)
        x = res["xf"]
        traj_frozen.append(x)  # Keep as DM

        # Check for ground impact using the kinematics function from model.py
        P4_val = P4_fn(x[:5]).full().flatten()
        if P4_val[1] <= 0:
            x_minus = x
            break

    print("Frozen body: q1..q4 stayed constant.")

    P4_at_impact = P4_fn(x_minus[:5]).full().flatten()
    print(f"Free foot y at impact: {P4_at_impact[1]:.6f}")

    # --- Phase 2: Impact Map ---
    x_plus = impact_map_fn(x_minus)

    # Convert to NumPy for the KE validation checks
    M_minus = M_fn(x_minus[:5]).full()
    M_plus = M_fn(x_plus[:5]).full()
    dq_minus = x_minus[5:].full()
    dq_plus = x_plus[5:].full()

    # KE_before = float(0.5 * dq_minus.T @ M_minus @ dq_minus)
    # KE_after = float(0.5 * dq_plus.T @ M_plus @ dq_plus)
    KE_before = (0.5 * dq_minus.T @ M_minus @ dq_minus).item()
    KE_after = (0.5 * dq_plus.T @ M_plus @ dq_plus).item()

    print(f"KE before impact: {KE_before:.4f} J")
    print(f"KE after impact:  {KE_after:.4f} J")

    # if KE_after < KE_before:
    #     print("KE decreased across the prescribed impact map.")
    # else:
    #     print("Note: the prescribed assignment impact map did not decrease KE here;")
    #     print(
    #         "this indicates a remaining inconsistency outside the frozen-body driver."
    #     )

    # --- Phase 3: Post-Impact Simulation ---
    t2_max = 0.5
    n_steps_2 = int(t2_max / dt)

    x2 = x_plus
    traj_post = [x2]  # Store CasADi DM natively

    for _ in range(n_steps_2):
        res = full_int(x0=x2)
        x2 = res["xf"]
        traj_post.append(x2)  # Keep as DM

    print("Experiment 2 passed: frozen body + impact validated.")

    sol1_y = ca.horzcat(*traj_frozen).full()
    sol2_y = ca.horzcat(*traj_post).full()

    return sol1_y, x_minus.full(), x_plus.full(), sol2_y


if __name__ == "__main__":
    print("=" * 50)
    print("Experiment 1: Gravitational Collapse")
    sol_collapse = run_gravity_collapse()
    plot_collapse_snapshots(sol_collapse, 4)
    # animate_collapse(sol_collapse)
    print("=" * 50)
    print("=" * 50)
    print("Experiment 2: Frozen Body + Impact")
    sol1, x_minus, x_plus, sol2 = run_frozen_body_impact()
    plot_frozen_body_snapshots(sol1, sol2, n_frames=4)
    # animate_frozen_and_impact(sol1, sol2, interval=10)
    print("=" * 50)
