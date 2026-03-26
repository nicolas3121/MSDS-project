

import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import matplotlib.animation as animation # type: ignore

from kinematics import draw_biped, get_skeleton, get_com_positions  # type: ignore
from simulate import run_gravity_collapse, run_frozen_body_impact # type: ignore


def plot_robot(q_val):
    
    fig, ax = plt.subplots(figsize=(10, 10))

    # Draw stick figure (thicker lines)
    P3, Pk1, Ph, Pt, Pk2, P4 = get_skeleton(q_val)
    ax.plot(*zip(P3, Pk1, Ph), 'b-o', lw=4, ms=10, zorder=3, label='Stance leg')
    ax.plot(*zip(Ph, Pt), 'k-o', lw=5, ms=10, zorder=3, label='Torso')
    ax.plot(*zip(Ph, Pk2, P4), 'r-o', lw=4, ms=10, zorder=3, label='Free leg')
    ax.axhline(0, color='gray', lw=1.5, linestyle='--')

    # Joint labels with individual offsets to avoid overlap
    joint_labels = [
        ('$P_3$ (stance foot)',  P3,  (-15, -20)),
        ('$P_{k1}$ (stance knee)', Pk1, (-120, -5)),
        ('$P_h$ (hip)',          Ph,  (-80, 10)),
        ('$P_t$ (torso top)',    Pt,  (12, 5)),
        ('$P_{k2}$ (free knee)', Pk2, (12, -5)),
        ('$P_4$ (free foot)',    P4,  (12, -20)),
    ]
    for name, pos, offset in joint_labels:
        ax.plot(*pos, 'ks', ms=10, zorder=5)
        ax.annotate(name, pos, textcoords='offset points',
                    xytext=offset, fontsize=11, fontweight='bold',
                    arrowprops=dict(arrowstyle='-', color='gray', lw=0.8))

    # CoM labels with individual offsets
    G1, G2, G3, G4, G5 = get_com_positions(q_val)
    com_labels = [
        ('$G_1$ (stance femur)', G1, (-130, -10)),
        ('$G_2$ (free femur)',   G2, (15, 10)),
        ('$G_3$ (stance tibia)', G3, (-130, -10)),
        ('$G_4$ (free tibia)',   G4, (15, -15)),
        ('$G_5$ (torso)',        G5, (15, 5)),
    ]
    for name, pos, offset in com_labels:
        ax.plot(*pos, 'r^', ms=10, zorder=5)
        ax.annotate(name, pos, textcoords='offset points',
                    xytext=offset, fontsize=11, color='darkred',
                    arrowprops=dict(arrowstyle='-', color='red', lw=0.8))

    # Formatting
    ax.set_title('Biped configuration (joints and CoMs)', fontsize=14, pad=15)
    ax.set_xlabel('x [m]', fontsize=12)
    ax.set_ylabel('y [m]', fontsize=12)
    ax.set_aspect('equal')
    margin = 0.3
    all_x = [p[0] for p in [P3, Pk1, Ph, Pt, Pk2, P4]]
    all_y = [p[1] for p in [P3, Pk1, Ph, Pt, Pk2, P4]]
    ax.set_xlim(min(all_x) - 0.6, max(all_x) + 0.5)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
    ax.grid(True, alpha=0.6, which='both')
    ax.minorticks_on()
    ax.grid(True, which='minor', alpha=0.25)
    ax.legend(fontsize=11, loc='upper right')
    plt.tight_layout()
    plt.savefig('robot_diagram.png', dpi=150)
    plt.show()


def plot_collapse_snapshots(sol, n_frames=8):
    
    fig, ax = plt.subplots(figsize=(10, 6))
    indices = np.linspace(0, sol.y.shape[1] - 1, n_frames, dtype=int)

    for i in indices:
        draw_biped(sol.y[:5, i], ax)

    ax.set_title("Experiment 1: Gravitational Collapse")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("collapse_snapshots.png", dpi=150)
    plt.show()


def plot_frozen_body_snapshots(sol1, sol2, n_frames=6):
    
    # Compute x-offset for post-impact frames
    q_at_impact = sol1.y[:5, -1]
    P4_at_impact = get_skeleton(q_at_impact)[5]
    x_off = P4_at_impact[0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Before impact (frozen body phase)
    ax = axes[0]
    indices = np.linspace(0, sol1.y.shape[1] - 1, n_frames, dtype=int)
    for i in indices:
        draw_biped(sol1.y[:5, i], ax)
    ax.set_title("Before Impact (Frozen Body)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.3)

    # After impact (free dynamics) — offset so new stance foot
    # appears where the old free foot landed
    ax = axes[1]
    indices = np.linspace(0, sol2.y.shape[1] - 1, n_frames, dtype=int)
    for i in indices:
        draw_biped(sol2.y[:5, i], ax)
    ax.set_title("After Impact (Free Dynamics)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("frozen_body_snapshots.png", dpi=150)
    plt.show()


def animate_collapse(sol, interval=30):
    
    fig, ax = plt.subplots(figsize=(6, 6))

    def update(frame):
        ax.clear()
        draw_biped(sol.y[:5, frame], ax)
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-0.5, 1.5)
        ax.set_title(f"Gravitational Collapse  t = {sol.t[frame]:.3f} s")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.grid(True, alpha=0.3)

    ani = animation.FuncAnimation(
        fig, update, frames=300, interval=interval, repeat=True,
    )
    plt.tight_layout()
    plt.show()
    return ani


def animate_frozen_and_impact(sol1, sol2, interval=30):
    
    

    q_at_impact = sol1.y[:5, -1]
    P4_at_impact = get_skeleton(q_at_impact)[5]  # free foot position
    x_off = P4_at_impact[0]

    # Combine the two solutions into one timeline
    t_all = np.concatenate([sol1.t, sol1.t[-1] + sol2.t])
    y_all = np.concatenate([sol1.y, sol2.y], axis=1)
    n_pre = sol1.y.shape[1]

    fig, ax = plt.subplots(figsize=(6, 6))

    def update(frame):
        ax.clear()
        offset = x_off if frame >= n_pre else 0.0
        draw_biped(y_all[:5, frame], ax, x_offset=offset)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.0, 2.0)
        phase = "Frozen Body" if frame < n_pre else "After Impact"
        ax.set_title(f"{phase}  t = {t_all[frame]:.3f} s")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.grid(True, alpha=0.3)

    ani = animation.FuncAnimation(
        #quit after t = 1.5 s to avoid long post-impact animation
        fig, update, frames=300, interval=interval, repeat=True,


    )
    plt.tight_layout()
    plt.show()
    return ani


# ── Main ────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Plotting robot diagram...")
    plot_robot(np.array([0.2, -0.3, -0.1, 0.15, 0.05]))

    print("Running simulations...")
    sol_collapse = run_gravity_collapse()
    sol1, x_minus, x_plus, sol2 = run_frozen_body_impact()

    print("\nPlotting snapshots...")
    plot_collapse_snapshots(sol_collapse)
    plot_frozen_body_snapshots(sol1, sol2)

    print("\nStarting animations (close window to proceed)...")
    ani1 = animate_collapse(sol_collapse)
    ani2 = animate_frozen_and_impact(sol1, sol2)
