#!/usr/bin/env python
import numpy as np
from numpy import sin, cos, tan, exp, sqrt, sign, arctan as atan, abs as Abs
from __builtin__ import min as Min, max as Max

# This file is auto-generated by ekf/codegen.py. DO NOT EDIT.


def Heaviside(x):
    return 1 * (x > 0)


def DiracDelta(x, v=1):
    return x == 0 and v or 0


def initial_state():
    x = np.float32(
        [0.0, 0.0, 0.0, 0.0, 0.0, 2.70000004768372, 1.04999995231628, 2.00000000000000, -0.649999976158142, -1.39999997615814, 0.200000002980232, 3.79999995231628, -35.0000000000000, 125.000000000000, 0.0]
    )
    P = np.diag(
        [1.00000011116208e-6, 0.0100000007078052, 4.00000000000000, 1.00000000000000, 0.160000011324883, 0.0400000028312206, 0.0400000028312206, 16.0000000000000, 0.250000000000000, 0.250000000000000, 0.250000000000000, 0.250000000000000, 10000.0000000000, 10000.0000000000, 1.00000000000000]
    )

    return x, P


def predict(x, P, Delta_t, u_M, u_delta):
    (v, delta, y_error, psi_error, kappa, ml_1, ml_2, ml_3, ml_4, srv_a, srv_b, srv_r, srvfb_a, srvfb_b, o_g) = x

    tmp0 = exp(ml_4)
    tmp1 = Abs(u_M)
    tmp2 = tmp1*exp(ml_2)
    tmp3 = tmp2*v
    tmp4 = tmp1*exp(ml_1)*Heaviside(u_M)
    tmp5 = exp(ml_3)
    tmp6 = -v
    tmp7 = tmp6 + 0.2
    tmp8 = tmp5*Heaviside(tmp7)
    tmp9 = Heaviside(-Delta_t*(-tmp0 - tmp3 + tmp4 - tmp8) - v)
    tmp10 = -Delta_t*(tmp0 + tmp3 - tmp4 + tmp8)
    tmp11 = Heaviside(tmp10 + v)
    tmp12 = Delta_t*tmp11
    tmp13 = tmp12*(tmp2 - tmp5*DiracDelta(tmp7))
    tmp14 = -delta + srv_a*u_delta + srv_b
    tmp15 = Delta_t*srv_r
    tmp16 = Abs(tmp14)
    tmp17 = Min(tmp15, tmp16)
    tmp18 = sign(tmp14)
    tmp19 = 2*tmp17*DiracDelta(tmp14) + tmp18**2*Heaviside(tmp15 - tmp16)
    tmp20 = sin(psi_error)
    tmp21 = Delta_t*(tmp13/2 + tmp9/2 - 1)
    tmp22 = cos(psi_error)
    tmp23 = Max(tmp10, tmp6)
    tmp24 = Delta_t*(tmp23/2 + v)
    tmp25 = tmp22*tmp24
    tmp26 = Delta_t**2
    tmp27 = tmp11*tmp20*tmp26/2
    tmp28 = kappa*y_error
    tmp29 = tmp28 - 1
    tmp30 = 1/tmp29
    tmp31 = kappa*tmp30
    tmp32 = delta + tmp22*tmp31
    tmp33 = tmp20*tmp24
    tmp34 = tmp11*tmp26*tmp32/2

    F = np.eye(15)
    F[0, 0] += -tmp13 - tmp9
    F[0, 5] += tmp12*tmp4
    F[0, 6] += -tmp12*tmp3
    F[0, 7] += -tmp12*tmp8
    F[0, 8] += -tmp0*tmp12
    F[1, 1] += -tmp19
    F[1, 9] += tmp19*u_delta
    F[1, 10] += tmp19
    F[1, 11] += Delta_t*tmp18*Heaviside(-tmp15 + tmp16)
    F[2, 0] += tmp20*tmp21
    F[2, 3] += -tmp25
    F[2, 5] += -tmp27*tmp4
    F[2, 6] += tmp27*tmp3
    F[2, 7] += tmp27*tmp8
    F[2, 8] += tmp0*tmp27
    F[3, 0] += tmp21*tmp32
    F[3, 1] += -tmp24
    F[3, 2] += kappa**2*tmp25/tmp29**2
    F[3, 3] += tmp31*tmp33
    F[3, 4] += tmp25*tmp30*(tmp28*tmp30 - 1)
    F[3, 5] += -tmp34*tmp4
    F[3, 6] += tmp3*tmp34
    F[3, 7] += tmp34*tmp8
    F[3, 8] += tmp0*tmp34
    Q = np.float32([ 4, 0.490000000000000, pow(0.1*v + 0.001, 2), pow(0.15*v + 0.001, 2), pow(0.75*v + 0.001, 2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.00000000000000e-6])
    x[0] += tmp23
    x[1] += tmp17*tmp18
    x[2] += -tmp33
    x[3] += -tmp24*tmp32

    P = np.dot(F, np.dot(P, F.T)) + Delta_t * np.diag(Q)
    return x, P


def step(x, u, Delta_t):
    (v, delta, y_error, psi_error, kappa, ml_1, ml_2, ml_3, ml_4, srv_a, srv_b, srv_r, srvfb_a, srvfb_b, o_g) = x
    (u_M, u_delta) = u

    tmp0 = -v
    tmp1 = exp(ml_4)
    tmp2 = exp(ml_2)
    tmp3 = Abs(u_M)
    tmp4 = tmp2*tmp3
    tmp5 = tmp4*v
    tmp6 = Heaviside(u_M)
    tmp7 = exp(ml_1)
    tmp8 = tmp3*tmp7
    tmp9 = tmp6*tmp8
    tmp10 = exp(ml_3)
    tmp11 = tmp0 + 0.2
    tmp12 = tmp10*Heaviside(tmp11)
    tmp13 = -Delta_t*(tmp1 + tmp12 + tmp5 - tmp9)
    tmp14 = Max(tmp0, tmp13)
    tmp15 = -delta + srv_a*u_delta + srv_b
    tmp16 = sign(tmp15)
    tmp17 = Delta_t*srv_r
    tmp18 = Abs(tmp15)
    tmp19 = Min(tmp17, tmp18)
    tmp20 = sin(psi_error)
    tmp21 = Delta_t*(tmp14/2 + v)
    tmp22 = tmp20*tmp21
    tmp23 = cos(psi_error)
    tmp24 = kappa*y_error
    tmp25 = tmp24 - 1
    tmp26 = 1/tmp25
    tmp27 = kappa*tmp26
    tmp28 = delta + tmp23*tmp27
    tmp29 = Heaviside(-Delta_t*(-tmp1 - tmp12 - tmp5 + tmp9) - v)
    tmp30 = Heaviside(tmp13 + v)
    tmp31 = Delta_t*tmp30
    tmp32 = tmp31*(-tmp10*DiracDelta(tmp11) + tmp4)
    tmp33 = tmp16**2*Heaviside(tmp17 - tmp18) + 2*tmp19*DiracDelta(tmp15)
    tmp34 = Delta_t*(tmp29/2 + tmp32/2 - 1)
    tmp35 = tmp21*tmp23
    tmp36 = Delta_t**2
    tmp37 = tmp20*tmp30*tmp36/2
    tmp38 = tmp28*tmp30*tmp36/2
    tmp39 = sign(u_M)
    tmp40 = -tmp2*tmp39*v + tmp39*tmp6*tmp7 + tmp8*DiracDelta(u_M)

    F = np.eye(15)
    F[0, 0] += -tmp29 - tmp32
    F[0, 5] += tmp31*tmp9
    F[0, 6] += -tmp31*tmp5
    F[0, 7] += -tmp12*tmp31
    F[0, 8] += -tmp1*tmp31
    F[1, 1] += -tmp33
    F[1, 9] += tmp33*u_delta
    F[1, 10] += tmp33
    F[1, 11] += Delta_t*tmp16*Heaviside(-tmp17 + tmp18)
    F[2, 0] += tmp20*tmp34
    F[2, 3] += -tmp35
    F[2, 5] += -tmp37*tmp9
    F[2, 6] += tmp37*tmp5
    F[2, 7] += tmp12*tmp37
    F[2, 8] += tmp1*tmp37
    F[3, 0] += tmp28*tmp34
    F[3, 1] += -tmp21
    F[3, 2] += kappa**2*tmp35/tmp25**2
    F[3, 3] += tmp22*tmp27
    F[3, 4] += tmp26*tmp35*(tmp24*tmp26 - 1)
    F[3, 5] += -tmp38*tmp9
    F[3, 6] += tmp38*tmp5
    F[3, 7] += tmp12*tmp38
    F[3, 8] += tmp1*tmp38

    J = np.zeros((15, 2))
    J[0, 0] = tmp31*tmp40
    J[1, 1] = srv_a*tmp33
    J[2, 0] = -tmp37*tmp40
    J[3, 0] = -tmp38*tmp40
    x[0] += tmp14
    x[1] += tmp16*tmp19
    x[2] += -tmp22
    x[3] += -tmp21*tmp28
    return x, F, J


def update_centerline(x, P, a, b, c, y_c, Rk):
    y_error = x[2]
    psi_error = x[3]
    kappa = x[4]
    tmp0 = a*y_c
    tmp1 = b + 2*tmp0
    tmp2 = tmp1**2 + 1
    tmp3 = 1/sqrt(tmp2)
    tmp4 = a*y_c**2 + b*y_c + c - tmp1*y_c
    tmp5 = 2*tmp2**(-1.5)
    tmp6 = 1/tmp2
    tmp7 = 2*tmp6
    tmp8 = tmp1*tmp4
    tmp9 = 2*a
    tmp10 = tmp2**(-2.5)
    tmp11 = 12.0*tmp1*tmp10

    yk = np.float32(
        [-tmp3*tmp4 - y_error, -psi_error + atan(tmp1), a*tmp5 - kappa])

    Hk = np.float32([
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    Mk = np.float32([
        [tmp3*y_c*(tmp7*tmp8 + y_c), tmp8/tmp2**(3/2), -tmp3, tmp3*tmp9*(tmp6*tmp8 + y_c)],
        [tmp7*y_c, tmp6, 0, a*tmp7],
        [-tmp0*tmp11 + tmp5, -tmp10*tmp9*(3.0*b + 6.0*tmp0), 0, -a**2*tmp11]])
    Rk = np.dot(Mk, np.dot(Rk, Mk.T))

    S = np.dot(Hk, np.dot(P, Hk.T)) + Rk

    LL = -np.dot(yk, np.dot(np.linalg.inv(S), yk)) - 0.5 * np.log(2 * np.pi * np.linalg.det(S))
    K = np.linalg.lstsq(S, np.dot(Hk, P))[0].T
    x += np.dot(K, yk)
    KHk = np.dot(K, Hk)
    P = np.dot((np.eye(len(x)) - KHk), P)
    return x, P, LL


def update_IMU(x, P, g_z):
    v = x[0]
    delta = x[1]
    o_g = x[14]

    yk = np.float32(
        [delta*v + g_z - o_g])

    Hk = np.float32([
        [-delta, -v, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

    Rk = np.diag([
        0.0100000000000000])

    S = np.dot(Hk, np.dot(P, Hk.T)) + Rk

    LL = -np.dot(yk, np.dot(np.linalg.inv(S), yk)) - 0.5 * np.log(2 * np.pi * np.linalg.det(S))
    K = np.linalg.lstsq(S, np.dot(Hk, P))[0].T
    x += np.dot(K, yk)
    KHk = np.dot(K, Hk)
    P = np.dot((np.eye(len(x)) - KHk), P)
    return x, P, LL


def update_encoders(x, P, dsdt, fb_delta):
    v = x[0]
    delta = x[1]
    srvfb_a = x[12]
    srvfb_b = x[13]

    yk = np.float32(
        [dsdt - 63.0316606304536*v, -delta*srvfb_a + fb_delta - srvfb_b])

    Hk = np.float32([
        [63.0316606304536, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, srvfb_a, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, delta, 1, 0]])

    Rk = np.diag([
        14400, 49])

    S = np.dot(Hk, np.dot(P, Hk.T)) + Rk

    LL = -np.dot(yk, np.dot(np.linalg.inv(S), yk)) - 0.5 * np.log(2 * np.pi * np.linalg.det(S))
    K = np.linalg.lstsq(S, np.dot(Hk, P))[0].T
    x += np.dot(K, yk)
    KHk = np.dot(K, Hk)
    P = np.dot((np.eye(len(x)) - KHk), P)
    return x, P, LL


