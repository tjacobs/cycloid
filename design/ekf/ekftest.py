import numpy as np
import ekf


def ekf_test():
    x, P = ekf.initial_state()

    # Slow
    for u in np.linspace(0, 0.2, 10):
        x, P = ekf.predict(x, P, 0.0625, u, 0) # Delta_t (1/16th second), motor speed, no steering 
        print("Small acceleration: ", u, x[0])

    # GO!
    for u in np.linspace(0.8, 1.0, 3):
        x, P = ekf.predict(x, P, 0.0625, u, 0) # Delta_t (1/16th second), motor speed, no steering 
        print("Acceleration: ", u, x[0])

    # STOP!
    x, P = ekf.predict(x, P, 0.0625, -1, 0)
    x, P = ekf.predict(x, P, 0.0625, -1, 0)
    print('Full brake 2 frames:', x[0])

    # Coast
    for i in range(10):
        x, P = ekf.predict(x, P, 0.0625, -1, 0)
        print(x[0])
    print('After 10 frames of coasting: ', x[0])
#    x, P = ekf.update_centerline(x, P, 0, 0.01, 0.1, np.eye(3) * 0.1)
    print(x)


if __name__ == '__main__':
    ekf_test()
