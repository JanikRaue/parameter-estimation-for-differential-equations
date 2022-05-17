import numpy
import pandas
import click

# import deepxde and set backend to 'pytorch'
import deepxde
if deepxde.backend.get_preferred_backend() != 'pytorch':
    deepxde.backend.set_default_backend('pytorch')
deepxde.config.real.set_float64()

from src.data import LorenzSystem, SirModel
from src.visualisation import plot_param_process, plot_progress


class PhysicsInformedNeuralNet:
    def __init__(self, differential_equation=LorenzSystem(), size_hidden_layer=None, num_hidden_layer=None):
        self.differential_equation = differential_equation
        self.input_data = self.differential_equation.get_data(return_type='dict')
        self.filename = 'variables_process.dat'

        # set deepxde variables
        self.params = {key: deepxde.Variable(0.1) for key in self.differential_equation.param}
        self.geom = deepxde.geometry.TimeDomain(0, self.differential_equation.end_time)
        # initial values
        ic_y_1 = deepxde.IC(self.geom, lambda x: self.differential_equation.initial_condition['y_1'], self.boundary, component=0)
        ic_y_2 = deepxde.IC(self.geom, lambda x: self.differential_equation.initial_condition['y_2'], self.boundary, component=1)
        ic_y_3 = deepxde.IC(self.geom, lambda x: self.differential_equation.initial_condition['y_3'], self.boundary, component=2)
        # observations
        observe_y_1 = deepxde.PointSetBC(self.input_data['t'], self.input_data['y_1'], component=0)
        observe_y_2 = deepxde.PointSetBC(self.input_data['t'], self.input_data['y_2'], component=1)
        observe_y_3 = deepxde.PointSetBC(self.input_data['t'], self.input_data['y_3'], component=2)

        if self.differential_equation.name == 'lorenz':
            self.pde = self.lorenz_system
            num_hidden_layer = 3 if num_hidden_layer is None else num_hidden_layer
            size_hidden_layer = 32 if size_hidden_layer is None else size_hidden_layer
        elif self.differential_equation.name == 'sir':
            self.pde = self.sir_model
            num_hidden_layer = 5 if num_hidden_layer is None else num_hidden_layer
            size_hidden_layer = 128 if size_hidden_layer is None else size_hidden_layer
            self.N = sum(self.differential_equation.initial_condition.values())
            print(f"population: {self.N}")
        else:
            raise NotImplemented('PINNs are only implemented for the Lorenz system and the SIR model.')

        # summarise for deepxde
        self.pinn_data = deepxde.data.PDE(self.geom,
                                          self.pde,
                                          [ic_y_1, ic_y_2, ic_y_3, observe_y_1, observe_y_2, observe_y_3],
                                          anchors=self.input_data['t'])
        self.net = deepxde.maps.FNN([1] + [size_hidden_layer] * num_hidden_layer + [3], "tanh", "Glorot uniform")
        self.model = deepxde.Model(self.pinn_data, self.net)

    def sir_model(self, x, y):
        S, I, R = y[:, 0:1], y[:, 1:2], y[:, 2:3]
        ds_t = deepxde.grad.jacobian(y, x, i=0)
        di_t = deepxde.grad.jacobian(y, x, i=1)
        dr_t = deepxde.grad.jacobian(y, x, i=2)
        return [ds_t + (self.params['beta'] * S/self.N * I),
                di_t - (self.params['beta'] * S/self.N * I) + self.params['gamma']*I,
                dr_t - self.params['gamma']*I]

    def lorenz_system(self, x, y):
        """Lorenz system.
        dy1/dx = rho * (y2 - y1)
        dy2/dx = y1 * (sigma - y3) - y2
        dy3/dx = y1 * y2 - beta * y3
        """
        y1, y2, y3 = y[:, 0:1], y[:, 1:2], y[:, 2:]
        dy1_x = deepxde.grad.jacobian(y, x, i=0)
        dy2_x = deepxde.grad.jacobian(y, x, i=1)
        dy3_x = deepxde.grad.jacobian(y, x, i=2)
        return [dy1_x - self.params['rho'] * (y2 - y1),
                dy2_x - y1 * (self.params['sigma'] - y3) + y2,
                dy3_x - y1 * y2 + self.params['beta'] * y3]

    def boundary(self, _, on_initial):
        return on_initial

    def train(self, epochs=30000, display_every=1000, loss_weights=None):
        if loss_weights is None:
            self.model.compile("adam", lr=0.001,
                               external_trainable_variables=[*self.params.values()])
        else:
            self.model.compile("adam", lr=0.001,
                               external_trainable_variables=[*self.params.values()],
                               loss_weights=loss_weights)  # loss weights for sir

        variable = deepxde.callbacks.VariableValue([*self.params.values()], period=1, filename=self.filename)

        loss_history, train_state = self.model.train(epochs=epochs, display_every=display_every, callbacks=[variable])

        print('variable: ', variable.get_value())
        return loss_history

    def get_param_history(self):
        lines = open(self.filename, "r").readlines()
        epochs = numpy.array([int(line.split()[0]) for line in lines])
        value_progress = numpy.array([numpy.fromstring(line.replace(']\n', '').split(' [')[1], sep=',') for line in lines])
        df = pandas.DataFrame(value_progress, columns=[*self.differential_equation.param.keys()])
        df.index = epochs
        return df

    def predict(self, pred_input=None):
        if pred_input is None:
            pred_input = self.input_data['t']
        return self.model.predict(pred_input)

@click.command()
@click.option('--model_name', default='lorenz', help="Choose 'lorenz' for LorenzSystem or 'sir' for SirModel.")
@click.option('--save_dir', default=None, help="If you want to save the figure - add the path here.")
def main(model_name, save_dir):
    if model_name == 'lorenz':
        pinn = PhysicsInformedNeuralNet(LorenzSystem())
    elif model_name == 'sir':
        pinn = PhysicsInformedNeuralNet(SirModel())
    else:
        raise NotImplemented(f"The model_name must be either 'lorenz' or 'sir'.")

    pinn.train(epochs=10000)
    plot_progress(time=pinn.input_data['t'],
                  observation=pinn.differential_equation.data,
                  true=pinn.differential_equation.data,
                  prediction=pinn.predict(pinn.input_data['t']))
    plot_param_process(true_params=pinn.differential_equation.param,
                       df_progress=pinn.get_param_history())

if __name__ == '__main__':
    main()
