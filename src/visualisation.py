import matplotlib.pyplot as plt
import numpy

plt.rcParams.update({'font.size': 30,
                     "figure.figsize": (16, 8),
                     'text.usetex': True,
                     'text.latex.preamble': r'\usepackage{lmodern}',
                     'font.weight': 'bold',
                     "axes.labelweight": "bold"})

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
            anchor = 1.38
        else:
            plt.plot(time, prediction[:, i], f"{colors[i]}d", label=f"${variables[var]}^F$")
            plt.plot(time, analysis[:, i], f"{colors[i]}o", label=f"${variables[var]}^A$")
            anchor = 1.49
    plt.xlabel(r"$time$")
    plt.ylabel(r"$state$")
    # plt.ylim(-25, 50)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, anchor), ncol=3)
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"save figure to {save_path}")
    plt.close()


def plot_param_process(true_params, df_progress=None, save_path=None, plot_true_params=True):
    anchor = 1.27
    epochs = df_progress.index
    value_progress = df_progress[[*true_params]].values
    for i, param_key in enumerate(true_params):
        plt.plot(epochs, value_progress[:, i], f"{colors[i]}-", label=f"$\{param_key}$")
        if plot_true_params:
            anchor = 1.38
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
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"save figure to {save_path}")
    plt.close()


def plot_error(true, prediction, name='kalman', variables=None, save_path=None):
    error = (true-prediction).abs()
    if variables is None:
        variables = {'y_1': 'y_1', 'y_2': 'y_2', 'y_3': 'y_3'}
    for i, col in enumerate(error):
        if name == 'kalman':
            label = f"${variables[col]} - {variables[col]}^F$"
            anchor = 1.18
        else:
            hat = [r"\hat{", r"}"]
            label = f"${variables[col]} - {hat[0]}{variables[col]}{hat[1]}$"
            anchor = 1.16
        plt.plot(error[col], colors[i], label=label)
        plt.ylabel(r"$error$")
        plt.xlabel(r"$time$")
        plt.legend(ncol=3, loc='upper center',  bbox_to_anchor=(0.5, anchor))
    cost = sum(error.sum() / len(error)) / 3
    print(f"The cost per state is: {cost}.")
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"save figure to {save_path}")
    plt.close()


def plot_param_distribution(df, best_params, true_params=None, name='lorenz', save_path=None):
    fig, axes = plt.subplots(1, len(best_params)-1, sharey=True)
    axes[0].set_ylabel(f"$cost$")
    axes[0].set_ylim(0, max(df['cost'].values))
    for i, param_key in enumerate(best_params):
        if param_key == 'cost':
            continue
        axes[i].plot(df[param_key].values, df['cost'].values, f"{colors[i]}.")
        if true_params is not None:
            axes[i].plot([true_params[param_key]]*2, [0, max(df['cost'].values)], f"k:", label=f"$\{param_key}_{'{true}'}={round(true_params[param_key],2)}$")
        axes[i].plot(best_params[param_key], best_params['cost'], f"ko", label=f"$\{param_key}_{'{best}'}={round(best_params[param_key],2)}$")
        axes[i].legend(loc='upper center', bbox_to_anchor=(0.5, 1.36))
        if name == 'lorenz':
            axes[i].set_xlim(0, 20)
        elif name == 'sir':
            axes[i].set_xlim(0, 3)
        axes[i].set_xlabel(f"$\{param_key}$")
    fig.tight_layout()
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"save figure to {save_path}")
    plt.close()

