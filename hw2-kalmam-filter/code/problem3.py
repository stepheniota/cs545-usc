import matplotlib.pyplot as plt
import numpy as np


class ExtendedKalmanFilter():
    """ Implementation of an Extended Kalman Filter. """
    def __init__(self, mu, sigma, g, g_jac, h, h_jac, R=0., Q=0.):
        """
        :param mu: prior mean
        :param sigma: prior covariance
        :param g: process function
        :param g_jac: process function's jacobian
        :param h: measurement function
        :param h_jac: measurement function's jacobian
        :param R: process noise
        :param Q: measurement noise
        """
        # prior
        self.mu = mu
        self.sigma = sigma
        self.mu_init = mu
        self.sigma_init = sigma
        # process model
        self.g = g
        self.g_jac = g_jac
        self.R = R
        # measurement model
        self.h = h
        self.h_jac = h_jac
        self.Q = Q
        # other
        self.random = np.random.default_rng()

    def reset(self):
        """ Reset belief state to initial value. """
        self.mu = self.mu_init
        self.sigma = self.sigma_init

    def run(self, sensor_data):
        """ Run the Kalman Filter using the given sensor updates.
            :param sensor_data: array of T sensor updates as a TxS array.
            :returns: A tuple of predicted means (as a TxD array) and predicted
                    covariances (as a TxDxD array) representing the KF's belief
                    state AFTER each predict/update cycle, over T timesteps.
        """
        T = len(sensor_data)
        D = len(self.mu)
        means = np.zeros((T, D))
        covs = np.zeros((T, D, D))
        means[0] = self.mu_init
        covs[0] = self.sigma_init

        I = np.eye(2)
        for i in range(1, T):
            obs = sensor_data[i]

            # predict priors
            mu_prior = self.g(self.mu)
            G_grad = self.g_jac(self.mu)
            sigma_prior = G_grad @ self.sigma @ G_grad.T + self.R

            # calculate the gain
            H_grad = self.h_jac(mu_prior)
            S = H_grad @ sigma_prior @ H_grad.T + self.Q
            K_gain = sigma_prior @ H_grad.T @ np.linalg.inv(S)

            # update posterior belief
            self.mu = mu_prior + K_gain @ (obs - self.h(mu_prior))
            self.sigma = (I - K_gain @ H_grad) @ sigma_prior
            means[i] = self.mu
            covs[i] = self.sigma

        return means, covs


    def _predict(self):
        pass

    def _update(self, z):
        pass

    def sensor_data(self, truth):  #, timesteps):
        z = lambda x : np.sqrt(np.square(x) + 1)
        sensor = z(truth)
        noise = self.random.normal(0, scale=1, size=len(truth))
        sensor.T[1, :] = 0
        sensor.T[0, :] += noise
        sensor[0,0] = 1
        return sensor

    def true_pos(self, timesteps):
        alpha = 0.1
        x_0 = 2
        truth = np.zeros((len(timesteps), 2))
        truth[0, 0] = x_0
        for i, t in enumerate(timesteps[1:], start=1):
            truth[i, 0] = alpha * truth[i - 1, 0]
        noise = self.random.normal(0, scale=0.5, size=len(timesteps)-1)
        truth.T[0, 1:] += noise
        truth.T[1, :] = alpha
        return truth


def plot_prediction(t, ground_truth, predict_mean, predict_cov):
    """
    Plot ground truth vs. predicted value.
    :param t: 1-dimensional array representing timesteps, in seconds.
    :param ground_truth: Tx2 array of ground truth values
    :param predict_mean: TxD array of mean vectors
    :param predict_cov: TxDxD array of covariance matrices
    """
    gt_x, gt_a = ground_truth[:, 0], ground_truth[:, 1]
    pred_x, pred_a = predict_mean[:, 0], predict_mean[:, 1]
    pred_x_std = np.sqrt(predict_cov[:, 0, 0])
    pred_a_std = np.sqrt(predict_cov[:, 1, 1])

    plt.figure(figsize=(7, 10))
    plt.subplot(211)
    plt.plot(t, gt_x, color='k')
    plt.plot(t, pred_x, color='g')
    #plt.plot(sensor_data, color='b')
    plt.fill_between(t,
                     pred_x-pred_x_std,
                     pred_x+pred_x_std,
                     color='g',
                     alpha=0.5)
    plt.legend(("ground_truth", "prediction"))
    plt.xlabel("time (s)")
    plt.ylabel(r"$x$")
    plt.title(r"EKF estimation: $x$")

    plt.subplot(212)
    plt.plot(t, gt_a, color='k')
    plt.plot(t, pred_a, color='g')
    plt.fill_between(t,
                     pred_a-pred_a_std,
                     pred_a+pred_a_std,
                     color='g',
                     alpha=0.5)
    plt.legend(("ground_truth", "prediction"))
    plt.xlabel("time (s)")
    plt.ylabel(r"$\alpha$")
    plt.title(r"EKF estimation: $\alpha$")
    #plt.savefig('problem3_ekf_estimation')
    plt.show()

def g_func(state):
    """ Maps belief state from current timestep to the next timestep.
        g: belief_state(t) -> belief_state(t+1)
    """
    state[0] *= state[1]
    return state

def g_grad(state):
    """ pdv matrix of state transition func. """
    alpha = state[1]
    x = state[0]
    G_grad = np.array([[alpha, x], [0, 1]])
    return G_grad

def h_func(state):
    """ Maps current belief state to what would be the expected observation.
        h: belief_state(t) -> obs(t)
    """
    z = lambda x : np.sqrt(np.square(x) + 1)
    h = z(state)
    h[1] = 0
    return h

def h_grad(state):
    """ The pdv matrix of the obs mapping func. """
    grad = lambda x : x / np.sqrt(np.square(x) + 1)
    x = state[0]
    x_grad = grad(x)
    H_grad = np.diag([x_grad, 0])
    return H_grad

def problem3():
    Q = np.eye(2)
    R = 0.5 * np.eye(2)  # adding process noise to \alpha helps prediction to converge
    mu_0 = np.array([1, 2])
    sigma_0 = np.diag([2, 2])
    T = 20 + 1

    kf = ExtendedKalmanFilter(mu=mu_0, sigma=sigma_0, g=g_func, g_jac=g_grad,
                              h=h_func, h_jac=h_grad, R=R, Q=Q)
    kf.reset()

    timesteps = [i for i in range(T)]
    truth = kf.true_pos(timesteps)
    sensor_data = kf.sensor_data(truth)
    mean, cov = kf.run(sensor_data)
    print('Sequence of predicted alphas: ', np.round(mean[:, 1], 3))

    plot_prediction(timesteps, truth, mean, cov)


if __name__ == '__main__':
    problem3()