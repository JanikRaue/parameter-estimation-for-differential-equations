import numpy
import pandas
from tqdm import tqdm
from joblib import Parallel, delayed
import click

from src.data import LorenzSystem, SirModel
from src.visualisation import plot_param_process, plot_progress


class KalmanFilter:
    def __init__(self, differential_equation):
        self.differential_equation = differential_equation
        self.input_data = self.differential_equation.get_data()
        self.best_params = {}
        self.df_progress = pandas.DataFrame()

    def kalman_filter(self, params):
        if self.differential_equation.name == 'lorenz':
            return self.kalman_filter_lorenz(params)
        elif self.differential_equation.name == 'sir':
            return self.kalman_filter_sir(params)
        else:
            raise NotImplemented('The Kalman filter is only implement for the Lorenz system and the SIR model.')

    def cost(self, params, loss='l2'):
        p = params.copy()
        y_a, y_f = self.kalman_filter(p)
        if loss == 'l1':
            p.update({'cost': sum(sum(abs(y_f[:, :3] - self.input_data.values)))})
            return p
        elif loss == 'l2':
            p.update({'cost': sum(sum((y_f[:, :3] - self.input_data.values) ** 2))})
            return p

    def get_random_params(self):
        if self.differential_equation.name == 'lorenz':
            param_space = {'rho': [0, 20], 'sigma': [0, 20], 'beta': [0, 20]}
        elif self.differential_equation.name == 'sir':
            param_space = {'beta': [0, 3], 'gamma': [0, 3]}
        return {key: numpy.random.uniform(value[0], value[1]) for key, value in param_space.items()}

    def train(self, epochs=10000):
        df = pandas.DataFrame(Parallel(n_jobs=-1)(delayed(self.cost)(self.get_random_params()) for i in tqdm(range(epochs))))
        param_names = [*self.differential_equation.param.keys()]
        self.best_params = df.sort_values(by='cost')[param_names].iloc[0].to_dict()
        print(f"best parameter: {self.best_params}")

        progress = []
        for i, ind in enumerate(df.index):
            temp_dict = df.iloc[i].to_dict()
            if len(progress) == 0:
                progress.append(temp_dict)
            elif progress[i - 1]['cost'] < temp_dict['cost']:
                progress.append(progress[i - 1])
            else:
                progress.append(temp_dict)
        self.df_progress = pandas.DataFrame(progress)
        return df

    def predict(self):
        return self.kalman_filter(params=self.best_params)

    def kalman_filter_sir(self, params: dict = None):
        t_old = 0
        time = self.input_data.index
        y_hat = self.input_data.values
        # initialise parameters
        if params is None:
            beta, gamma = self.differential_equation.param.values()
        else:
            beta, gamma = params.values()

        # initialise forecast realisations
        y_f = numpy.empty((len(y_hat), 4))
        y_f[0, :3] = y_hat[0]
        N = sum(y_hat[0])
        y_f[0, 3] = sum(y_hat[0])
        # initialise analysis realisations
        y_a = numpy.empty(y_f.shape)
        # initialise forecast error matrix
        sig_b = numpy.sqrt(1.0)  # Set value correct
        P_F = sig_b ** 2 * numpy.eye(4)
        # initialis error covariance matrices
        R_covariance = numpy.eye(3)
        Q_covariance = numpy.eye(4)
        # initialise observation operator
        H = numpy.zeros((3, 4))
        H[:3, :3] = numpy.eye(3)
        for i, t in enumerate(time):
            h = t - t_old
            # compute Kalman gain
            K = P_F @ H.T @ numpy.linalg.inv(H @ P_F @ H.T + R_covariance)
            # analysis step
            y_a[i] = y_f[i] + K @ (y_hat[i] - H @ y_f[i])
            # get current variables from analysis
            S, I, R, N = y_a[i]
            # print(S, I, R, N)
            # print (S+I+R)
            # compute error covariance matrix of analysis step
            P_A = (numpy.eye(4) - K @ H) @ P_F
            if t < max(time):
                # get matrix to produce forecast - via explicit Euler (Runge-Kutta method with s=1)
                M = numpy.eye(4)
                M[:3, :3] += numpy.array(
                    [[0, -(h * beta * S) / N, 0], [(h * beta * I) / N, -h * gamma, 0], [0, h * gamma, 0]])
                # forecast step
                y_f[i + 1] = M @ y_a[i]
                # compute error covariance of forecast step
                P_F = M @ P_A @ M.T + Q_covariance
                # update time variable
                t_old = t
        return y_a, y_f

    def kalman_filter_lorenz(self, params: dict = None):
        t_old = 0
        time = self.input_data.index
        y_hat = self.input_data.values
        # initialise parameters
        if params is None:
            rho, sigma, beta = self.differential_equation.param.values()
        else:
            rho, sigma, beta = params.values()
        # initialise forecast realisations
        y_f = numpy.empty((len(y_hat), 3))
        y_f[0, :3] = y_hat[0]
        # initialise analysis realisations
        y_a = numpy.empty(y_f.shape)
        # initialise forecast error matrix
        sig_b = numpy.sqrt(1.0)  # Set value correct
        P_F = sig_b ** 2 * numpy.eye(3)
        # initialise error covariance matrices
        R = numpy.eye(3)
        Q = numpy.eye(3)
        # initialise observation operator
        H = numpy.eye(3)
        for i, t in enumerate(time):
            h = t - t_old
            # compute Kalman gain
            K = P_F @ H.T @ numpy.linalg.inv(H @ P_F @ H.T + R)
            # analysis step
            y_a[i] = y_f[i] + K @ (y_hat[i] - H @ y_f[i])
            # get current variables from analysis
            y_1, y_2, y_3 = y_a[i]
            # compute error covariance matrix of analysis step
            P_A = (numpy.eye(3) - K @ H) @ P_F

            if t < max(time):
                # get matrix to produce forecast - via explicit Euler (Runge-Kutta method with s=1)
                M = numpy.eye(3)
                M += h * numpy.array([[-rho, rho, 0], [sigma, -1, -y_1], [y_2, 0, -beta]])
                # forecast step
                y_f[i + 1] = M @ y_a[i]
                # compute error covariance of forecast step
                P_F = M @ P_A @ M.T + Q
                # update time variable
                t_old = t
        return y_a, y_f

@click.command()
@click.option('--model_name', default='lorenz', help="Choose 'lorenz' for LorenzSystem or 'sir' for SirModel.")
@click.option('--save_dir', default=None, help="If you want to save the figure - add the path here.")
def main(model_name, save_dir):
    if model_name == 'lorenz':
        kf = KalmanFilter(LorenzSystem())
    elif model_name == 'sir':
        kf = KalmanFilter(SirModel())
    else:
        raise NotImplemented(f"The model_name must be either 'lorenz' or 'sir'.")

    kf.train(epochs=10000)
    analysis, forecast = kf.predict()
    plot_progress(time=numpy.array([[t] for t in kf.input_data.index]),
                  observation=kf.differential_equation.data,
                  true=kf.differential_equation.data,
                  prediction=forecast,
                  analysis=analysis)

    plot_param_process(true_params=kf.differential_equation.param,
                       df_progress=kf.df_progress)


if __name__ == '__main__':
    main()
