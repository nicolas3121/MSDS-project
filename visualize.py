from model import com_pos_fn, joint_pos_fn, P5_END_fn, P4_fn
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


def draw_biped(q_val, ax, x_offset=0.0):
    offset = np.array([x_offset, 0.0])

    # Flatten the CasADi matrices and reshape them to (N, 2) coordinates
    P = np.array(joint_pos_fn(q_val)).flatten().reshape(5, 2) + offset
    G = np.array(com_pos_fn(q_val)).flatten().reshape(5, 2) + offset

    # P5_END is a single point, so flatten makes it a simple (2,) array
    P5_END = np.array(P5_END_fn(q_val)).flatten() + offset

    ax.plot(*P[:3, :].T, "b-o", lw=2)
    ax.plot([P[2, 0], P5_END[0]], [P[2, 1], P5_END[1]], "b-o", lw=2)
    ax.plot(*P[2:].T, "r-o", lw=2)
    ax.plot(G[:, 0], G[:, 1], "k*", ms=8, label="G Points", zorder=5)
    ax.axhline(0, color="gray", lw=1, linestyle="--")


def plot_collapse_snapshots(sol_traj, n_frames=8):

    state_array = sol_traj.full()

    fig, ax = plt.subplots(figsize=(12, 6))

    indices = np.linspace(0, state_array.shape[1] - 1, n_frames, dtype=int)

    for i, idx in enumerate(indices):
        # Optional: Add a slight shift to the right for each frame
        # so they don't draw perfectly on top of each other
        shift = i * 0.0
        draw_biped(state_array[:5, idx], ax, x_offset=shift)

    ax.set_title("Experiment 1: Gravitational Collapse")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.3)

    # THE CRITICAL FIX: Lock the aspect ratio so limbs don't stretch!
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig("collapse_snapshots.png", dpi=150)
    plt.show()


def plot_frozen_body_snapshots(sol1_y, sol2_y, n_frames=6):

    # Compute x-offset for post-impact frames natively
    q_at_impact = sol1_y[:5, -1]
    P4_at_impact = np.array(P4_fn(q_at_impact)).flatten()
    x_off = P4_at_impact[0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Before impact (frozen body phase) ---
    ax = axes[0]
    indices1 = np.linspace(0, sol1_y.shape[1] - 1, n_frames, dtype=int)
    for idx in indices1:
        draw_biped(sol1_y[:5, idx], ax)

    ax.set_title("Before Impact (Frozen Body)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")  # Fixes the limb stretching!

    # --- After impact (free dynamics) ---
    ax = axes[1]
    indices2 = np.linspace(0, sol2_y.shape[1] - 1, n_frames, dtype=int)
    for idx in indices2:
        # BUG FIXED: Added x_offset=x_off so the robot doesn't teleport to 0,0
        draw_biped(sol2_y[:5, idx], ax, x_offset=x_off)

    ax.set_title("After Impact (Free Dynamics)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")  # Fixes the limb stretching!

    plt.tight_layout()
    plt.savefig("frozen_body_snapshots.png", dpi=150)
    plt.show()


def animate_collapse(sol_traj, dt=0.01, interval=30):

    # 1. Convert the CasADi matrix to a standard NumPy array
    state_array = sol_traj.full()

    # 2. Get the total number of frames dynamically based on the array size
    num_frames = state_array.shape[1]

    fig, ax = plt.subplots(figsize=(8, 8))

    def update(frame):
        ax.clear()

        # 3. Extract the joint angles for this specific frame
        draw_biped(state_array[:5, frame], ax)

        # Lock aspect ratio so limbs remain rigid
        ax.set_aspect("equal")

        # Fixed axis limits so the "camera" doesn't jump around
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)

        # 4. Calculate the current time manually
        current_time = frame * dt
        ax.set_title(f"Gravitational Collapse  t = {current_time:.3f} s")

        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.grid(True, alpha=0.3)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=num_frames,  # Use the dynamically calculated number of frames
        interval=interval,
        repeat=True,
    )

    plt.tight_layout()
    plt.show()
    return ani


def animate_frozen_and_impact(sol1_y, sol2_y, dt=0.005, interval=60):

    # 1. Get the pre-impact state to calculate the foot offset
    q_at_impact = sol1_y[:5, -1]

    # 2. Use the native CasADi function to find the free foot (P4)
    P4_at_impact = np.array(P4_fn(q_at_impact)).flatten()
    x_off = P4_at_impact[0]

    # 3. Combine the two NumPy arrays horizontally
    y_all = np.concatenate([sol1_y, sol2_y], axis=1)
    n_pre = sol1_y.shape[1]
    num_frames = y_all.shape[1]

    fig, ax = plt.subplots(figsize=(8, 8))

    def update(frame):
        ax.clear()

        # Apply the x offset only for the post-impact frames (swapping the stance foot)
        offset = x_off if frame >= n_pre else 0.0

        # Extract the joint angles for this specific frame
        draw_biped(y_all[:5, frame], ax, x_offset=offset)

        # THE CRITICAL FIX: Lock the aspect ratio so limbs don't stretch!
        ax.set_aspect("equal")

        # Expand x-limits slightly to account for the robot stepping forward
        ax.set_xlim(-1.5, 2.0)
        ax.set_ylim(-1.0, 2.0)

        phase = "Frozen Body (Falling)" if frame < n_pre else "After Impact"
        current_time = frame * dt
        ax.set_title(f"{phase}  t = {current_time:.3f} s")

        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.grid(True, alpha=0.3)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=num_frames,  # Dynamically matches the combined simulation length
        interval=interval,
        repeat=True,
    )

    plt.tight_layout()
    plt.show()
    return ani
