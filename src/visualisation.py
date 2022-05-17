import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 22,
                     "figure.figsize": (15, 10),
                     'text.usetex': True,
                     'text.latex.preamble': r'\usepackage{lmodern}'})

colors = ['r', 'b', 'g', 'y']
hat_y = r'\hat{y}'
ast = r'\ast'


def plot_progress(time, observation, true, prediction, analysis=None, variables=None, save_path=None):
    if variables is None:
        variables = {'y_1': 'y_1', 'y_2': 'y_2', 'y_3': 'y_3'}
    for i, var in enumerate(variables):
        plt.plot(time, true[var], f"{colors[i]}:", label=f"${variables[var]}$")
        plt.plot(observation.index, observation[var], f"{colors[i]}x", label=f"${variables[var]}^{ast}$")
        if analysis is None:
            hat = [r"\hat{", r"}"]
            plt.plot(time, prediction[:, i], f"{colors[i]}-", label=f"${hat[0]}{variables[var]}{hat[1]}$")
            anchor = 1.21
        else:
            plt.plot(time, prediction[:, i], f"{colors[i]}d", label=f"${variables[var]}^F$")
            plt.plot(time, analysis[:, i], f"{colors[i]}o", label=f"${variables[var]}^A$")
            anchor = 1.29
    plt.xlabel(r"$time$")
    plt.ylabel(r"$state$")
    # plt.ylim(-25, 50)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, anchor), ncol=3)
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=300)
        print(f"save figure to {save_path}")
    plt.close()


def plot_param_process(true_params, df_progress=None, save_path=None, plot_true_params=True):
    anchor = 1.16
    epochs = df_progress.index
    value_progress = df_progress[[*true_params]].values
    for i, param_key in enumerate(true_params):
        plt.plot(epochs, value_progress[:, i], f"{colors[i]}-", label=f"$\{param_key}$")
        if plot_true_params:
            anchor = 1.21
            plt.plot(epochs, numpy.ones(value_progress[:, 0].shape) * true_params[param_key], f"{colors[i]}:",
                     label=f"$\{param_key}_{'{true}'}={true_params[param_key]}$")
        plt.plot(epochs[-1], value_progress[-1, i], f"{colors[i]}o",
                 label=f"$\{param_key}_{'{best}'}={round(value_progress[-1, i], 3)}$")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, anchor), ncol=len([*true_params]))
    plt.xlabel(f'$epoch$')
    plt.ylabel(r'$parameter$')
    plt.xlim(0, max(epochs) + 1)
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=300)
        print(f"save figure to {save_path}")
    plt.close()
