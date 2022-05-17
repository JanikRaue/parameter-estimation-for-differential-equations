import numpy
import os.path
import pandas
import matplotlib.pyplot as plt
import pylab
import click
from mpl_toolkits import mplot3d


colors = ['r', 'b', 'g', 'y']

def rk_update(func, t, y, h):
    """
    :param func:
        - function where the system of differential equations is defined
        - input of func should be t and y
        - output should be numpy array
    :param t: current time
    :param y: current observations
    :param h: step-size
    :return: new observations
    """
    k1 = (func(t, y))
    k2 = (func((t + h / 2), (y + h * k1 / 2)))  # c_2 = 1/2, a_21 = 1/2
    k3 = (func((t + h / 2), (y + h * k2 / 2)))  # c_3 = 1/2, a_31 = 0, a_32 = 1/2
    k4 = (func((t + h), (y + h * k3)))  # c_4 = 1, a_41 = 0, a_42 = 0, a_43 = 1
    k = (k1 + 2 * k2 + 2 * k3 + k4) / 6  # b_1 = 1/6, b_2 = 2/6, b_3 = 2/6, b_4 = 1/6
    return y + h * k


class DifferentialEquation:
    def __init__(self, param, initial_condition, end_time, number_of_data, noise_variance):
        self.name = ''
        self.param = param
        self.initial_condition = initial_condition
        self.end_time = end_time
        self.number_of_data = number_of_data
        self.noise_std = numpy.sqrt(noise_variance)
        self.data = pandas.DataFrame()
        self.set_data()

    def set_data(self):
        # step_size
        h = (self.end_time - self.initial_condition['t']) / self.number_of_data
        # times_of_interest
        ob_t = numpy.linspace(self.initial_condition['t'], self.end_time, self.number_of_data)
        ob_y = [[self.initial_condition['y_1'], self.initial_condition['y_2'], self.initial_condition['y_3']]]
        for t in ob_t[1:]:
            ob_y.append(rk_update(func=self.ode, t=t, y=ob_y[-1], h=h))
        ob_y = numpy.array(ob_y)
        ob_y += numpy.random.normal(loc=0, scale=self.noise_std, size=ob_y.shape)
        self.data = pandas.DataFrame(ob_y, index=ob_t, columns=['y_1', 'y_2', 'y_3'])

    def get_data(self, return_type='DataFrame'):
        if return_type == 'DataFrame':
            return self.data
        elif return_type == 'dict':
            return_dict = {'t': numpy.array([[ind] for ind in self.data.index])}
            return_dict.update({key: numpy.array([[ind] for ind in self.data[key]]) for key in self.data.columns})
            return return_dict
        else:
            msg = f"only implemented for return_type 'DataFrame' or 'dict' but '{return_type}' was given"
            raise NotImplemented(msg)

    def plot_data(self, kind='2d', data=None, save_dir=None):
        if data is None:
            data = self.data

        if kind == '2d':
            self.plot_2d()
        elif kind == '3d':
            fig = pylab.figure(figsize=(5, 5), dpi=100)
            self.plot_3d()

        if save_dir is None:
            plt.show()
        elif os.path.isdir(save_dir):
            param_str = r"rho-{:.0f}_sigma-{:.0f}_beta-{:.0f}".format(*self.param.values())
            ic_str = r"t-{:.0f}_y1-{:.0f}_y2-{:.0f}_y3-{:.0f}".format(*self.initial_condition.values())
            plt.savefig(f'{save_dir}/lorenz_system_{kind}_{param_str}_{ic_str}.png', dpi=300)


class SirModel(DifferentialEquation):
    def __init__(self, param=None, initial_condition=None, end_time=99, number_of_data=100, noise_variance=0):
        DifferentialEquation.__init__(self, param, initial_condition, end_time, number_of_data, noise_variance)
        if initial_condition is None:
            initial_condition = {'t': 0, 'y_1': 999.0, 'y_2': 1.0, 'y_3': 0.0}
        if param is None:
            param = {'beta': 0.15, 'gamma': 1 / 8}
        self.name = 'sir'

    def ode(self, t, y):
        # S = y[0], I = y[1], R = y[2]
        dy_dt = [-(self.param['beta'] * y[0] * y[1])/sum(y),
                 (self.param['beta'] * y[0] * y[1])/sum(y) - self.param['gamma']*y[1],
                 self.param['gamma']*y[1]]
        return numpy.array(dy_dt)

    def plot_2d(self):
        if self.noise_std == 0:
            legend = [r"$S$", r"$I$", r"$R$"]
            marker = '-'
        else:
            legend = [r"$S^{\ast}$", r"$I^{\ast}$", r"$R^{\ast}$"]
            marker = 'x'
        plt.figure(figsize=(12, 9))
        for i, col in enumerate(self.data.columns):
            plt.plot(self.data.index, self.data[col], f"{colors[i]}{marker}")
        plt.ylabel(r"$state$")
        plt.xlabel(r"$time$")
        plt.legend(legend, loc='upper center', bbox_to_anchor=(0.5, 1.11), ncol=3)


class LorenzSystem(DifferentialEquation):
    def __init__(self, param=None, initial_condition=None, end_time=3, number_of_data=100, noise_variance=0):
        DifferentialEquation.__init__(self, param, initial_condition, end_time, number_of_data, noise_variance)
        if initial_condition is None:
            initial_condition = {'t': 0, 'y_1': -5, 'y_2': 10, 'y_3': 25}
        if param is None:
            param = {'rho': 8, 'sigma': 16, 'beta': 2}
        self.name = 'lorenz'

    def ode(self, t, y):
        # y_1 = y[0], y_2 = y[1], y_3 = y[2]
        dy_dt = [self.param['rho'] * (y[1] - y[0]),
                 y[0] * (self.param['sigma'] - y[2]) - y[1],
                 y[0] * y[1] - self.param['beta'] * y[2]]
        return numpy.array(dy_dt)

    def plot_3d(self):
        ax = pylab.axes(projection="3d")
        ax.plot3D(self.data['y_1'], self.data['y_2'], self.data['y_3'], 'blue')
        # ax.set_title(f'Lorenz System with {msg_1} and {msg_2}')
        ax.set_xlabel(r"$y_1$")
        ax.set_ylabel(r"$y_2$")
        ax.set_zlabel(r"$y_3$")
        ax.set_xlim(-25, 25)
        ax.set_ylim(-25, 25)
        ax.set_zlim(0, 50)

    def plot_2d(self):
        # plt.figure(figsize=(12, 9))
        if self.noise_std == 0:
            legend = [r"$y_1$", r"$y_2$", r"$y_3$"]
            marker = '-'
        else:

            legend = [r"$y^{\ast}_1$", r"$y^{\ast}_2$", r"$y^{\ast}_3$"]
            marker = 'x'
        for i, col in enumerate(self.data.columns):
            plt.plot(self.data.index, self.data[col], f"{colors[i]}{marker}")
        plt.ylabel(r"$state$")
        plt.xlabel(r"$time$")
        plt.ylim(-10, 30)
        plt.legend(legend, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3)


@click.command()
@click.option('--model_name', default='lorenz', help="Choose 'lorenz' for LorenzSystem or 'sir' for SirModel.")
@click.option('--save_dir', default=None, help="If you want to save the figure - add the path here.")
def main(model_name, save_dir):
    if model_name == 'lorenz':
        diff_equation = LorenzSystem()
    elif model_name == 'sir':
        diff_equation = SirModel()
    else:
        raise NotImplemented(f"The model_name must be either 'lorenz' or 'sir'.")
    diff_equation.plot_data(save_dir=save_dir)


if __name__ == '__main__':
    main()
