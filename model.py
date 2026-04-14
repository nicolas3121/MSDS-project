import casadi as ca
from config import L1, L2, L3, L4, L5, R1, R2, R3, R4, R5, MASS_LIST, INERTIA_LIST, g

q = ca.SX.sym("q", 5)
dq = ca.SX.sym("dq", 5)
u = ca.SX.sym("u", 4)

x = ca.vertcat(q, dq)

th5 = q[4]
th1 = th5 + q[0]
th2 = th5 + q[1]
th3 = th1 + q[2]
th4 = th2 + q[3]


def link_vector(angle, length):
    return ca.vertcat(-length * ca.sin(angle), length * ca.cos(angle))


P3 = ca.vertcat(0, 0)
P1 = P3 + link_vector(th3, L3)
P5 = P1 + link_vector(th1, L1)
P2 = P5 - link_vector(th2, L2)
P4 = P2 - link_vector(th4, L4)
P5_END = P5 + link_vector(th5, L5)


G3 = P1 - link_vector(th3, R3)
G1 = P5 - link_vector(th1, R1)
G5 = P5 + link_vector(th5, R5)
G2 = P5 - link_vector(th2, R2)
G4 = P2 - link_vector(th4, R4)

G_list = [G1, G2, G3, G4, G5]
theta_vec = ca.vertcat(th1, th2, th3, th4, th5)

J_G = [ca.jacobian(G_i, q) for G_i in G_list]
dG_dt = [J_G_i @ dq for J_G_i in J_G]
dtheta_dt = ca.jacobian(theta_vec, q) @ dq

T = 0
for m_i, dG_i_dt in zip(MASS_LIST, dG_dt):
    T += 0.5 * m_i * ca.sumsqr(dG_i_dt)
for I_i, dtheta_i_dt in zip(INERTIA_LIST, ca.vertsplit(dtheta_dt)):
    T += 0.5 * I_i * dtheta_i_dt**2

V = 0
for m_i, G_i in zip(MASS_LIST, G_list):
    V += m_i * g * G_i[1]

L = T - V

M = ca.hessian(T, dq)[0]  # 5x5

dL_ddq = ca.gradient(L, dq)
dL_dq = ca.gradient(L, q)

d_dt_dL_ddq_partial = ca.jacobian(dL_ddq, q) @ dq

c = d_dt_dL_ddq_partial - dL_dq

B = ca.DM([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]])

ddq = ca.solve(M, B @ u - c)
x_dot = ca.vertcat(dq, ddq)

# Dynamics functions
f_SS = ca.Function("f_SS", [x, u], [x_dot], ["x", "u"], ["x_dot"])
M_fn = ca.Function("M_func", [q], [M], ["q"], ["M"])
c_fn = ca.Function("c_func", [q, dq], [c], ["q", "dq"], ["c"])

# Kinematics functions
joint_pos_fn = ca.Function(
    "joint_pos", [q], [ca.vertcat(P3, P1, P5, P2, P4)], ["q"], ["positions"]
)
com_pos_fn = ca.Function(
    "com_pos", [q], [ca.vertcat(G1, G2, G3, G4, G5)], ["q"], ["com_positions"]
)
P5_END_fn = ca.Function("P5_END", [q], [P5_END], ["q"], ["P5_END"])
P4_fn = ca.Function("P4", [q], [P4], ["q"], ["P4"])

# Impact
P_mat = ca.DM(
    [
        [0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1],
    ]
)

x_minus = ca.SX.sym("x_minus", 10)

q_minus = x_minus[:5]
dq_minus = x_minus[5:]

q_plus = P_mat @ q_minus
M_minus = M_fn(q_minus)
M_plus = M_fn(q_plus)

RHS = M_minus @ dq_minus
dq_plus = ca.solve(M_plus, RHS)

x_plus = ca.vertcat(q_plus, dq_plus)

impact_map_fn = ca.Function("impact_map", [x_minus], [x_plus], ["x_minus"], ["x_plus"])


T_fn = ca.Function("T_func", [q, dq], [T], ["q", "dq"], ["T"])
V_fn = ca.Function("V_func", [q], [V], ["q"], ["V"])