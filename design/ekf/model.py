import numpy as np
import sympy as sp
import codegen
import os

sp.init_printing()

# Define our model's state variables:
(v,          # velocity (m/s)
 delta,      # steering curvature (1/m) - inverse turning radius
 y_error,    # lateral distance from centerline; + means car is right of line
 psi_error,  # car's angle w.r.t. centerline; + means car is facing counterclockwise from line
 kappa       # kappa is the centreline line curvature (1/m)
 ) = sp.symbols("v, delta, y_error, psi_error, kappa", real=True)

(ml_1, ml_2, ml_3,     # log of brushless motor model constants
 ml_4,                 # log of static friction constant
 srv_a, srv_b, srv_r,  # servo response model; delta -> srv_a * control + srv_b at rate srv_r
 srvfb_a, srvfb_b,     # servo feedback measurement = srvfb_a * delta + srvfb_b
 o_g                   # gyroscope offset; gyro measures v * delta + o_g
 ) = sp.symbols("ml_1, ml_2, ml_3, ml_4, srv_a, srv_b, srv_r, srvfb_a, srvfb_b, o_g", real=True)

# dt, steering input, motor input
(Delta_t,  # time between prediction updates
 u_delta,  # control input for steering
 u_M       # control input for motor (assumed PWM controlled brushless)
 ) = sp.symbols("Delta_t u_delta u_M", real=True)

# State vector x is all of the above
X = sp.Matrix([v, delta, y_error, psi_error, kappa,
               ml_1, ml_2, ml_3, ml_4,
               srv_a, srv_b, srv_r, srvfb_a, srvfb_b, o_g])

# Print
print "State variables:"
sp.pprint(X.T)

# Gen
ekfgen = codegen.EKFGen(X)

# Define a default initial state
x0 = np.float32([
    # v, delta, y_error, psi_error, kappa
    0, 0, 0, 0, 0,
    # ml_1 (log m/s^2)
    2.7,
    # ml_2 (log 1/s)
    1.05, 
    # ml_3, ml_4 (log m/s^2 static frictional deceleration)
    2.0, -0.65,
    # srv_a, srv_b, srv_r,
    -1.4, 0.2, 3.8,
    # srvfb_a, srvfb_b
    -35, 125,
    # o_g
    0])

# And covariance
P0 = np.float32([
    # v, delta, y_error, psi_error, kappa
    # assume we start stationary
    0.001, 0.1, 2, 1, 0.4,
    # ml_1, ml_2, ml_3, ml_4
    0.2, 0.2, 4, 0.5,
    # srv_a, srv_b, srv_r
    0.5, 0.5, 0.5,
    # srvfb_a, srvfb_b
    100, 100,
    # o_g
    1])**2

try:
    os.mkdir("out_cc")
except:
    pass
try:
    os.mkdir("out_py")
except:
    pass
ekfgen.open("out_cc", "out_py", sp.Matrix(x0), sp.Matrix(P0))

# The motor model has three components:
# The electronic speed controller is really just a voltage source
# and a PWM-controlled electronic switch which either applies a voltage
# to the motor in pulses (when input control signal is positive),
# or shorts the motor out in pulses (when it is negative).
# The motor is also always slowed down by friction.

# The car's acceleration is proportional to the torque produced by the motor,
# so we fold the inertia into the system constants. The car's acceleration is
# thus:
#   k1 * u_V * u_DC - k2 * u_DC * v - k3*(static?) - k4
# where u_V is the control voltage signal (assumed 1 or 0) and u_DC is the
# duty cycle control input (from 0 to 1) and v is the current velocity.
# The ESC takes a positive or negative u_M control input which is transformed
# to u_DC and u_V here first.

# Since k1, k2, k3, and k4 are scale constants, we keep them as logarithms in
# the model. That way they can never go negative, and we avoid huge derivatives
# when the relative scales are very different. This can, however, blow up to
# huge values if we're not carefully managing measurement and process noise.
# units:
# k1: m/s^2 / V  (acceleration per volt)
# k2: 1/s  (EMF decay time constant)
# k3: m/s^2  (coulomb friction, minimum torque to get moving)
# k4: m/s^2  (dynamic friction)
k1, k2, k3, k4 = sp.exp(ml_1), sp.exp(ml_2), sp.exp(ml_3), sp.exp(ml_4)

# Define user voltage
u_DC = sp.Abs(u_M)
u_V = sp.Heaviside(u_M)  # 1 if u_M > 0, else 0

# The static friction coefficient tries to make the velocity exactly 0, up to the friction limit
k3 = k3 * sp.Heaviside(0.2 - v)  # k3 applies only when v < 0.2
dv = Delta_t*(u_V * u_DC * k1 - u_DC * v * k2 - k3 - k4)
dv = sp.Max(dv, -v)  # velocity cannot go negative
av = v + dv / 2  # average velocity during the timestep

# The servo has its own control loop and position feedback built in, but
# we need to model how it relates to the car's actual rotation, so the
# "servo" constants here also encompass the steering geometry of the car.
# We assume here that the servo linearly moves to the desired position, with
# a certain ratio (srv_a) between control input and turning curvature (1/radius),
# a certain offset (srv_b) when the control signal is 0, and a linear moving rate srv_r.
# The math for this is kind of messy; the code would be simpler as some if
# statements, but this needs to be a differentiable function.
ddelta = sp.Min(Delta_t * srv_r, sp.Abs(srv_a * u_delta + srv_b - delta)) * sp.sign(srv_a * u_delta + srv_b - delta)

# The state transition kinematics equations are based on a curvilinear unicycle model.
f = sp.Matrix([
    v + dv,
    delta + ddelta,
    y_error - Delta_t * av * sp.sin(psi_error),
    psi_error + Delta_t * av * (-delta + kappa * sp.cos(psi_error) / (1 - kappa * y_error)),
    kappa,
    ml_1,
    ml_2,
    ml_3,
    ml_4,
    srv_a,
    srv_b,
    srv_r,
    srvfb_a,
    srvfb_b,
    o_g
])

# Print
print "State transition function: x +="
sp.pprint(f - X)

# Q Process noise needs tuning.
Q = sp.Matrix([
    # v, delta, y_error, psi_error, kappa
    2, 0.7, 0.1*v + 1e-3, 0.15*v + 1e-3, 0.75*v + 1e-3,
    # ml_1, ml_2, ml_3, ml_4
    0, 0, 0, 0,
    # srv_a, srv_b, srv_r
    0, 0, 0,
    # srvfb_a, srvfb_b
    0, 0,
    # o_g gyro
    1e-3])

# Generate the prediction code. Motor speed and steering are inputs, along with f, process noise, and delta_t time.
ekfgen.generate_predict(f, sp.Matrix([u_M, u_delta]), Q, Delta_t)

# Now define the measurement models.
# First we measure the road centerline's position, angle, and curvature with
# our camera / image processing pipeline. The result of that is a quadratic
# regression equation ax^2 + bx + c, and a quadratic fit covariance Rk also
# comes from our image processing pipeline.
def centerline_derivation():

    # Make the symbols
    a, b, c, yc, t = sp.symbols("a b c y_c t", real=True)

    # Z is the processed from camera measurement
    z_k = sp.Matrix([a, b, c, yc])

    # And yc is the center of the original datapoints, where our regression should
    # have the least amount of error. We will measure the centerline curvature
    # (kappa) and angle (psi_error) at this point, and then compute y_error as our
    # perpendicular distance to that line. Simples.
    # 
    #              /
    # psi_error   /
    # & kappa -> |___|  <- y_error
    # at    yc   |
    #             \
    #              \

    # The regression line is x = a*y^2 + b*y^1 + c
    xc = a*yc**2 + b*yc + c
    dx = sp.diff(xc, yc)
    dxx = sp.diff(dx, yc)
    kappa_est = dxx / ((dx**2 + 1)**(1.5))  # Curvature at yc

    pc = sp.Matrix([xc, yc])       # regression center on curve
    N = sp.Matrix([-1, dx])        # N is a vector normal to the curve
    Nnorm = sp.sqrt((N.T * N)[0])  # length of normal

    # If curvature is low, assume we have a straight line; project our
    # regression centerpoint onto the unit normal vector to determine distance
    # to centerline, and tan(psi_e) = dx/dy = dx/1
    ye_linear_est = (N.T * pc)[0] / Nnorm
    tanpsi_linear_est = dx

    h_x = sp.Matrix([y_error, psi_error, kappa])
    h_z = sp.Matrix([
        ye_linear_est,
        sp.atan(tanpsi_linear_est),
        kappa_est
    ])

    return h_x, h_z, z_k

# Centreline
h_x_centerline, h_z_centerline, z_k_centerline = centerline_derivation()
ekfgen.generate_measurement( "centerline", h_x_centerline, h_z_centerline, z_k_centerline, sp.symbols("R_k"))

# Delta is backwards from yaw rate, so negative here
h_imu = sp.Matrix([-v * delta + o_g])

# Gyro IMU
g_z = sp.symbols("g_z")
h_gyro = sp.Matrix([g_z])
R_gyro = sp.Matrix([0.1])
ekfgen.generate_measurement("IMU", h_imu, h_gyro, h_gyro, R_gyro)

# Generate measurement for encoders
METERS_PER_ENCODER_TICK = np.pi * 0.101 / 20
dsdt, fb_delta = sp.symbols("dsdt fb_delta")
h_z_encoders = sp.Matrix([dsdt, fb_delta])
h_x_encoders = sp.Matrix([v / METERS_PER_ENCODER_TICK, srvfb_b + delta * srvfb_a])
R_encoders = sp.Matrix([120, 7])
ekfgen.generate_measurement("encoders", h_x_encoders, h_z_encoders, h_z_encoders, R_encoders)

# Save
ekfgen.close()
