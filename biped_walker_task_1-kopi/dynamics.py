"""
Euler-Lagrange dynamics for the 5-link biped walker.
Uses CasADi for symbolic derivation, then compiles to fast numeric functions.
"""

import casadi as ca # type: ignore
import numpy as np # type: ignore

# ── Parameters ──────────────────────────────────────────────

# Masses [kg]
m1, m2 = 6.8, 6.8
m3, m4 = 3.2, 3.2
m5     = 20.0

# Rotational inertia [kg*m^2]
I1, I2 = 1.08, 1.08
I3, I4 = 0.93, 0.93
I5     = 2.22

# Link lengths [m]
l1, l2 = 0.4, 0.4
l3, l4 = 0.4, 0.4
l5     = 0.625

# CoM distances from distal end [m]
r1, r2 = 0.163, 0.163
r3, r4 = 0.128, 0.128
r5     = 0.2

g = 9.81

# ── Symbolic variables ──────────────────────────────────────

q  = ca.MX.sym('q',  5)
dq = ca.MX.sym('dq', 5)

q1_s, q2_s, q3_s, q4_s, q5_s = q[0], q[1], q[2], q[3], q[4]

# Absolute angles
theta1 = q5_s + q1_s
theta2 = q5_s + q2_s
theta3 = q5_s + q1_s + q3_s
theta4 = q5_s + q2_s + q4_s
theta5 = q5_s

# ── Kinematics (CasADi symbolic) 

# Supporting leg: foot -> knee -> hip
P3  = ca.vertcat(0.0, 0.0)
Pk1 = P3  + l3 * ca.vertcat(ca.sin(theta3),  ca.cos(theta3))
Ph  = Pk1 + l1 * ca.vertcat(ca.sin(theta1),  ca.cos(theta1))

# Torso: hip -> top
P5  = Ph  + l5 * ca.vertcat(ca.sin(theta5),  ca.cos(theta5))

# Free leg: hip -> knee -> foot
Pk2 = Ph  + l2 * ca.vertcat(-ca.sin(theta2), -ca.cos(theta2))
P4  = Pk2 + l4 * ca.vertcat(-ca.sin(theta4), -ca.cos(theta4))

# Centre of mass positions
G3 = P3  + (l3 - r3) * ca.vertcat(ca.sin(theta3),  ca.cos(theta3))
G1 = Pk1 + (l1 - r1) * ca.vertcat(ca.sin(theta1),  ca.cos(theta1))
G5 = Ph  + r5        * ca.vertcat(ca.sin(theta5),  ca.cos(theta5))
G2 = Ph  + r2        * ca.vertcat(-ca.sin(theta2), -ca.cos(theta2))
G4 = Pk2 + r4        * ca.vertcat(-ca.sin(theta4), -ca.cos(theta4))

# ── Velocities ──────────────────────────────────────────────

masses  = [m1, m2, m3, m4, m5]
G_list  = [G1, G2, G3, G4, G5]
inertia = [I1, I2, I3, I4, I5]
thetas  = ca.vertcat(theta1, theta2, theta3, theta4, theta5)

# Translational velocities via Jacobians
Jac_G = [ca.jacobian(Gn, q) for Gn in G_list]
dG    = [J @ dq for J in Jac_G]

# Angular velocities
dthetas = ca.jacobian(thetas, q) @ dq

# ── Energy 

# Kinetic energy: translational + rotational
T = 0
for mn, dGn in zip(masses, dG):
    T += 0.5 * mn * ca.dot(dGn, dGn)
for In, dth in zip(inertia, ca.vertsplit(dthetas)):
    T += 0.5 * In * dth**2

# Potential energy
V = 0
for mn, Gn in zip(masses, G_list):
    V += mn * g * Gn[1]

L = T - V

# ── Euler-Lagrange equations

# Mass matrix: Hessian of T w.r.t. dq (T is quadratic in dq)
M_sym = ca.hessian(T, dq)[0]   # 5x5

# Coriolis + gravity term
dLddq      = ca.jacobian(L, dq).T           # (5,1)
dLdq       = ca.jacobian(L, q).T            # (5,1)
d_dLddq_dq = ca.jacobian(dLddq, q)          #
c_sym      = d_dLddq_dq @ dq - dLdq         

# ── Compile to fast numeric functions 

M_fn = ca.Function('M', [q],       [M_sym])
c_fn = ca.Function('c', [q, dq],   [c_sym])
skeleton_fn = ca.Function('skeleton', [q], [P3, Pk1, Ph, P5, Pk2, P4]) 
com_positions_fn = ca.Function('com_positions', [q], [G1, G2, G3, G4, G5])

# Free-foot position and its Jacobian (for impact detection and impact map)
P4_fn = ca.Function('P4', [q], [P4])




# ── State-space ODE

# B matrix: torques act on q1, q2, q3, q4 (not q5)
B = np.zeros((5, 4))
B[:4, :4] = np.eye(4)

# print state-space 




def f_SS(t, x, u=np.zeros(4)):
    
    q_val  = x[:5]
    dq_val = x[5:]

    M_val  = np.array(M_fn(q_val))
    c_val  = np.array(c_fn(q_val, dq_val)).flatten()

    ddq = np.linalg.solve(M_val, B @ u - c_val)
    return np.concatenate([dq_val, ddq.flatten()])
