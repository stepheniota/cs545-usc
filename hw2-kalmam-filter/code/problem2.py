import matplotlib.pyplot as plt
import numpy as np


class KalmanFilter():
    """
    Implementation of a Kalman Filter.
    """
    def __init__(self, mu, sigma, A, C, R=0., Q=0.):
        """
        :param mu: prior mean
        :param sigma: prior covariance
        :param A: process model
        :param C: measurement model
        :param R: process noise
        :param Q: measurement noise
        """
        # prior
        self.mu = mu
        self.sigma = sigma
        self.mu_init = mu
        self.sigma_init = sigma
        # process model
        self.A = A
        self.R = R
        # measurement model
        self.C = C
        self.Q = Q
        # other
        self.random = np.random.default_rng()
        self.dt = 0.1
        self.D = len(mu)

    def reset(self):
        """
        Reset belief state to initial value.
        """
        self.mu = self.mu_init
        self.sigma = self.sigma_init

    def run(self, sensor_data):
        """
        Run the Kalman Filter using the given sensor updates.

        :param sensor_data: array of T sensor updates as a TxS array.

        :returns: A tuple of predicted means (as a TxD array) and predicted
                  covariances (as a TxDxD array) representing the KF's belief
                  state AFTER each predict/update cycle, over T timesteps.
        """
        # FILL in your code here
        T = len(sensor_data)
        mean = np.zeros((T, self.D))
        cov = np.zeros((T, self.D, self.D))
        mean[0] = self.mu_init
        cov[0] = self.sigma_init

        for i in range(1, T):
            obs = sensor_data[i]
            mean[i], cov[i] = self.filter(obs)
        
        return mean, cov

    def filter(self, obs):
        """ One step through KF fitltering algorithm """
        # update prior belief
        mu_t_prior = self.A @ self.mu
        sigma_t_prior = (self.A @ self.sigma @ self.A.T) + self.R

        # compute gain
        K_gain = (sigma_t_prior @ self.C.T) 
        K_gain /= (self.C @ sigma_t_prior @ self.C.T + self.Q)

        # update posterior belief
        self.mu = mu_t_prior + K_gain * (obs - self.C @ mu_t_prior)
        self.sigma = (1 - K_gain @ self.C) * sigma_t_prior

        return self.mu, self.sigma

        
    def true_pos(self, timesteps):
        """ Returns the true position for the given timesteps. """
        return np.sin(timesteps)


    def sensor_data(self, timesteps):
        """ Generates noisy sensor data. """
        data = self.true_pos(timesteps)
        noise = self.random.normal(0, scale=self.Q, size=data.shape)
        return data + noise


def plot_prediction(t, ground_truth, measurement, predict_mean, predict_cov, name='a'):
    """
    Plot ground truth vs. predicted value.

    :param t: 1-dimensional array representing timesteps, in seconds.
    :param ground_truth: Tx1 array of ground truth values
    :param measurement: Tx1 array of sensor values
    :param predict_mean: TxD array of mean vectors
    :param predict_cov: TxDxD array of covariance matrices
    """
    predict_pos_mean = predict_mean[:, 0]
    predict_pos_std = predict_cov[:, 0, 0]

    plt.figure()
    plt.plot(t, ground_truth, color='k')
    plt.plot(t, measurement, color='r')
    plt.plot(t, predict_pos_mean, color='g')
    plt.fill_between(t,
                     predict_pos_mean - predict_pos_std,
                     predict_pos_mean + predict_pos_std,
                     color='g', alpha=0.5)
    plt.legend(("ground truth", "measurements", "predictions"))
    plt.xlabel("time (s)")
    plt.ylabel("position (m)")
    plt.title("Predicted Values")
    #plt.savefig(f'problem2{name}_kf_estimation.png')
    plt.show()


def plot_mse(t, ground_truth, predict_means, name='a'):
    """
    Plot MSE of your KF over many trials.

    :param t: 1-dimensional array representing timesteps, in seconds.
    :param ground_truth: Tx1 array of ground truth values
    :param predict_means: NxTxD array of T mean vectors over N trials
    """
    predict_pos_means = predict_means[:, :, 0]
    errors = ground_truth.squeeze() - predict_pos_means
    mse = np.mean(errors, axis=0) ** 2

    plt.figure()
    plt.plot(t, mse)
    plt.xlabel("time (s)")
    plt.ylabel("position MSE (m^2)")
    plt.title("Prediction Mean-Squared Error")
    #plt.savefig(f'problem2{name}_kf_mse.png')
    plt.show()


def problem2a():
    # preliminaries
    T = 100
    dt = 0.1
    A = np.eye(4)
    A[0, 1] = A[1, 2] = A[2, 3] = dt
    C = np.array([1, 0, 0, 0])
    Q = 1.0
    mu_0 = np.array([5, 1, 0, 0])
    D = len(mu_0)
    sigma_0 = 10 * np.eye(4)

    # filter
    kf = KalmanFilter(mu=mu_0, sigma=sigma_0, A=A, C=C, Q=Q)
    kf.reset()

    # truth and sensor data
    timesteps = np.array([i for i in np.arange(0, T*dt, dt)])
    truth = kf.true_pos(timesteps)
    sensor_data = kf.sensor_data(timesteps)

    # run kf algorithm on sensor data
    mean, cov = kf.run(sensor_data)

    # plots
    plot_prediction(t=timesteps, ground_truth=truth, measurement=sensor_data,
                    predict_mean=mean, predict_cov=cov, name='a')

    N = 10_000
    means = np.zeros((N, T, D))
    for i in range(N):
        if i % 1000 == 0: print(f'Running trial {i}...')
        kf.reset()
        data = kf.sensor_data(timesteps)
        means[i], _ = kf.run(data)
    
    plot_mse(t=timesteps, ground_truth=truth, predict_means=means, name='a')


def problem2b():
    R = 0.1 * np.eye(4)
    T = 100
    dt = 0.1
    A = np.eye(4)
    A[0, 1] = A[1, 2] = A[2, 3] = dt
    C = np.array([1, 0, 0, 0])
    Q = 1.0
    mu_0 = np.array([5, 1, 0, 0])
    D = len(mu_0)
    sigma_0 = 10 * np.eye(4)

    # filter
    kf = KalmanFilter(mu=mu_0, sigma=sigma_0, A=A, C=C, Q=Q, R=R)
    kf.reset()

    # truth and sensor data
    timesteps = np.array([i for i in np.arange(0, T*dt, dt)])
    truth = kf.true_pos(timesteps)
    sensor_data = kf.sensor_data(timesteps)

    # run kf algorithm on sensor data
    mean, cov = kf.run(sensor_data)

    # plots
    plot_prediction(timesteps, truth, sensor_data, mean, cov, name='b')

    N = 10_000
    means = np.zeros((N, T, D))
    for i in range(N):
        if i % 1000 == 0: print(f'Running trial {i}...')
        kf.reset()
        data = kf.sensor_data(timesteps)
        means[i], _ = kf.run(data)
    
    plot_mse(t=timesteps, ground_truth=truth, predict_means=means, name='b')

if __name__ == '__main__':
    problem2a()
    problem2b()
