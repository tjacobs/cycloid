#include <Eigen/Dense>
#include "ekf.h"

// This file is auto-generated by ekf/codegen.py. DO NOT EDIT.

using Eigen::VectorXf;
using Eigen::MatrixXf;

#define Min(x, y) fminf(x, y)
#define Max(x, y) fmaxf(x, y)

static inline float Heaviside(float x) {
  return x < 0 ? 0 : 1;
}

static inline float DiracDelta(float x) {
  return x == 0;
}

EKF::EKF() : x_(15), P_(15, 15) {
  Reset();
}


void EKF::Reset() {
  x_ << 0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        2.70000004768372,
        1.04999995231628,
        2.00000000000000,
        -0.649999976158142,
        -1.39999997615814,
        0.200000002980232,
        3.79999995231628,
        -35.0000000000000,
        125.000000000000,
        0.0;
  P_.setIdentity();
  P_.diagonal() << 1.00000011116208e-6,
    0.0100000007078052,
    4.00000000000000,
    1.00000000000000,
    0.160000011324883,
    0.0400000028312206,
    0.0400000028312206,
    16.0000000000000,
    0.250000000000000,
    0.250000000000000,
    0.250000000000000,
    0.250000000000000,
    10000.0000000000,
    10000.0000000000,
    1.00000000000000;
}

void EKF::Predict(float Delta_t, float u_M, float u_delta) {
  float v = x_[0];
  float delta = x_[1];
  float y_error = x_[2];
  float psi_error = x_[3];
  float kappa = x_[4];
  float ml_1 = x_[5];
  float ml_2 = x_[6];
  float ml_3 = x_[7];
  float ml_4 = x_[8];
  float srv_a = x_[9];
  float srv_b = x_[10];
  float srv_r = x_[11];

  float tmp0 = exp(ml_4);
  float tmp1 = fabsf(u_M);
  float tmp2 = tmp1*exp(ml_2);
  float tmp3 = tmp2*v;
  float tmp4 = tmp1*exp(ml_1)*Heaviside(u_M);
  float tmp5 = exp(ml_3);
  float tmp6 = -v;
  float tmp7 = tmp6 + 0.2;
  float tmp8 = tmp5*Heaviside(tmp7);
  float tmp9 = Heaviside(-Delta_t*(-tmp0 - tmp3 + tmp4 - tmp8) - v);
  float tmp10 = -Delta_t*(tmp0 + tmp3 - tmp4 + tmp8);
  float tmp11 = Heaviside(tmp10 + v);
  float tmp12 = Delta_t*tmp11;
  float tmp13 = tmp12*(tmp2 - tmp5*DiracDelta(tmp7));
  float tmp14 = -delta + srv_a*u_delta + srv_b;
  float tmp15 = Delta_t*srv_r;
  float tmp16 = fabsf(tmp14);
  float tmp17 = Min(tmp15, tmp16);
  float tmp18 = (((tmp14) > 0) - ((tmp14) < 0));
  float tmp19 = 2*tmp17*DiracDelta(tmp14) + pow(tmp18, 2)*Heaviside(tmp15 - tmp16);
  float tmp20 = sin(psi_error);
  float tmp21 = Delta_t*((1.0L/2.0L)*tmp13 + (1.0L/2.0L)*tmp9 - 1);
  float tmp22 = cos(psi_error);
  float tmp23 = Max(tmp10, tmp6);
  float tmp24 = Delta_t*((1.0L/2.0L)*tmp23 + v);
  float tmp25 = tmp22*tmp24;
  float tmp26 = pow(Delta_t, 2);
  float tmp27 = (1.0L/2.0L)*tmp11*tmp20*tmp26;
  float tmp28 = kappa*y_error;
  float tmp29 = tmp28 - 1;
  float tmp30 = 1.0/tmp29;
  float tmp31 = kappa*tmp30;
  float tmp32 = delta + tmp22*tmp31;
  float tmp33 = tmp20*tmp24;
  float tmp34 = (1.0L/2.0L)*tmp11*tmp26*tmp32;

  MatrixXf F(15, 15);
  F.setIdentity();
  F(0, 0) += -tmp13 - tmp9;
  F(0, 5) += tmp12*tmp4;
  F(0, 6) += -tmp12*tmp3;
  F(0, 7) += -tmp12*tmp8;
  F(0, 8) += -tmp0*tmp12;
  F(1, 1) += -tmp19;
  F(1, 9) += tmp19*u_delta;
  F(1, 10) += tmp19;
  F(1, 11) += Delta_t*tmp18*Heaviside(-tmp15 + tmp16);
  F(2, 0) += tmp20*tmp21;
  F(2, 3) += -tmp25;
  F(2, 5) += -tmp27*tmp4;
  F(2, 6) += tmp27*tmp3;
  F(2, 7) += tmp27*tmp8;
  F(2, 8) += tmp0*tmp27;
  F(3, 0) += tmp21*tmp32;
  F(3, 1) += -tmp24;
  F(3, 2) += pow(kappa, 2)*tmp25/pow(tmp29, 2);
  F(3, 3) += tmp31*tmp33;
  F(3, 4) += tmp25*tmp30*(tmp28*tmp30 - 1);
  F(3, 5) += -tmp34*tmp4;
  F(3, 6) += tmp3*tmp34;
  F(3, 7) += tmp34*tmp8;
  F(3, 8) += tmp0*tmp34;

  VectorXf Q(15);
  Q << 4, 0.490000000000000, pow(0.1*v + 0.001, 2), pow(0.15*v + 0.001, 2), pow(0.75*v + 0.001, 2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.00000000000000e-6;
  x_[0] += tmp23;
  x_[1] += tmp17*tmp18;
  x_[2] += -tmp33;
  x_[3] += -tmp24*tmp32;

  P_ = F * P_ * F.transpose();
  P_.diagonal() += Delta_t * Q;
}

bool EKF::UpdateCenterline(float a, float b, float c, float y_c, Eigen::MatrixXf Rk) {
  float y_error = x_[2];
  float psi_error = x_[3];
  float kappa = x_[4];
  float tmp0 = a*y_c;
  float tmp1 = b + 2*tmp0;
  float tmp2 = pow(tmp1, 2) + 1;
  float tmp3 = pow(tmp2, -1.0L/2.0L);
  float tmp4 = a*pow(y_c, 2) + b*y_c + c - tmp1*y_c;
  float tmp5 = 2*pow(tmp2, -1.5);
  float tmp6 = 1.0/tmp2;
  float tmp7 = 2*tmp6;
  float tmp8 = tmp1*tmp4;
  float tmp9 = 2*a;
  float tmp10 = pow(tmp2, -2.5);
  float tmp11 = 12.0*tmp1*tmp10;


  VectorXf yk(3);
  yk << -tmp3*tmp4 - y_error,
        -psi_error + atan(tmp1),
        a*tmp5 - kappa;

  MatrixXf Hk(3, 15);
  Hk << 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
  MatrixXf Mk(3, 4);
  Mk << tmp3*y_c*(tmp7*tmp8 + y_c), tmp8/pow(tmp2, 3.0L/2.0L), -tmp3, tmp3*tmp9*(tmp6*tmp8 + y_c),
        tmp7*y_c, tmp6, 0, a*tmp7,
        -tmp0*tmp11 + tmp5, -tmp10*tmp9*(3.0*b + 6.0*tmp0), 0, -pow(a, 2)*tmp11;
  Rk = Mk * Rk * Mk.transpose();

  Eigen::Matrix3f S = Hk * P_ * Hk.transpose() + Rk;
  MatrixXf K = P_ * Hk.transpose() * S.inverse();

  x_.noalias() += K * yk;
  P_ = (MatrixXf::Identity(15, 15) - K*Hk) * P_;
  return true;
}

bool EKF::UpdateIMU(float g_z) {
  float v = x_[0];
  float delta = x_[1];
  float o_g = x_[14];


  VectorXf yk(1);
  yk << delta*v + g_z - o_g;

  MatrixXf Hk(1, 15);
  Hk << -delta, -v, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;

  VectorXf Rk(1);
  Rk << 0.0100000000000000;

  Eigen::MatrixXf S = Hk * P_ * Hk.transpose();
  S.diagonal() += Rk;
  MatrixXf K = P_ * Hk.transpose() * S.inverse();

  x_.noalias() += K * yk;
  P_ = (MatrixXf::Identity(15, 15) - K*Hk) * P_;
  return true;
}

bool EKF::UpdateEncoders(float dsdt, float fb_delta) {
  float v = x_[0];
  float delta = x_[1];
  float srvfb_a = x_[12];
  float srvfb_b = x_[13];


  VectorXf yk(2);
  yk << dsdt - 63.0316606304536*v,
        -delta*srvfb_a + fb_delta - srvfb_b;

  MatrixXf Hk(2, 15);
  Hk << 63.0316606304536, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, srvfb_a, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, delta, 1, 0;

  VectorXf Rk(2);
  Rk << 14400, 49;

  Eigen::Matrix2f S = Hk * P_ * Hk.transpose();
  S.diagonal() += Rk;
  MatrixXf K = P_ * Hk.transpose() * S.inverse();

  x_.noalias() += K * yk;
  P_ = (MatrixXf::Identity(15, 15) - K*Hk) * P_;
  return true;
}

