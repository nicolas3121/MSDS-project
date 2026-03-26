import casadi as ca

# q = [q1, q2, q3, q4, q5]^T
q = ca.SX.sym("q", 5)
dq = ca.SX.sym("dq", 5)
u = ca.SX.sym("u", 4)

x = ca.vertcat(q, dq)

m1 = 6.8
m2 = 6.8
m3 = 3.2
m4 = 3.2
m5 = 20

I1 = 1.08
I2 = 1.08
I3 = 0.93
I4 = 0.94
I5 = 2.22

l3 = 0.4
l4 = 0.4
l1 = 0.4
l2 = 0.4
l5 = 0.625

r3 = 0.128
r4 = 0.128
r1 = 0.163
r2 = 0.163
r5 = 0.2

th5 = q[4]
th1 = th5 + q[0]
th2 = th5 + q[1]
th3 = th1 + q[2]
th4 = th2 + q[3]


def link_vector(angle, length):
    return ca.vertcat(-length * ca.sin(angle), length * ca.cos(angle))


P3 = ca.vertcat(0, 0)
P1 = P3 + link_vector(th3, l3)
P5 = P1 + link_vector(th1, l1)
P2 = P5 - link_vector(th2, l2)
P4 = P2 - link_vector(th4, l4)


G3 = P1 - link_vector(th3, r3)
G1 = P5 - link_vector(th1, r1)
G5 = P5 + link_vector(th5, r5)
G2 = P5 - link_vector(th2, r2)
G4 = P2 - link_vector(th4, r4)

# velocity
omega3 = ca.jacobian(th3, q) @ dq
omega1 = ca.jacobian(th1, q) @ dq
omega5 = ca.jacobian(th5, q) @ dq
omega2 = ca.jacobian(th2, q) @ dq
omega4 = ca.jacobian(th4, q) @ dq

v_G3 = ca.jacobian(G3, q) @ dq
v_G1 = ca.jacobian(G1, q) @ dq
v_G5 = ca.jacobian(G5, q) @ dq
v_G2 = ca.jacobian(G2, q) @ dq
v_G4 = ca.jacobian(G4, q) @ dq

# energy
Tl = 0.5 * (
    m3 * ca.sumsqr(v_G3)
    + m1 * ca.sumsqr(v_G1)
    + m5 * ca.sumsqr(v_G5)
    + m2 * ca.sumsqr(v_G2)
    + m4 * ca.sumsqr(v_G4)
)

Tr = 0.5 * (
    I3 * omega3**2 + I1 * omega1**2 + I5 * omega5**2 + I2 * omega2**2 + I4 * omega4**2
)

T = Tl + Tr

V = g * (m3 * G3[1] + m1 * G1[1] + m5 * G5[1] + m2 * G2[1] + m4 * G4[1])

L = T - V

M = ca.hessian(L, dq)[0]

dL_ddq = ca.gradient(L, dq)
dL_dq = ca.gradient(L, q)

d_dt_dL_ddq_partial = ca.jacobian(dL_ddq, q) @ dq

c = d_dt_dL_ddq_partial - dL_dq

B = ca.DM([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]])

ddq = ca.solve(M, B @ u - c)

x_dot = ca.vertcat(dq, ddq)
f_SS = ca.Function("f_SS", [x, u], [x_dot], ["x", "u"], ["x_dot"])

M_func = ca.Function("M_func", [q], [M], ["q"], ["M"])
