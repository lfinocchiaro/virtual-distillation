
import qutip
import numpy as np
import math
import matplotlib.pyplot as plt
#import matplotlib.colors as colors
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def loss_channel(N:int, decay_rate:float, t:float) -> qutip.Qobj:
    """Returns the superoperator of the loss channel for a single bosonic mode.

    Args:
        N (int): Hilbert space dimension of one bosonic mode (truncated at N-1 excitations).
        decay_rate (float): decay rate κ of the cavity.
        t (float): time duration of the loss channel.

    Returns:
        qutip.Qobj: superoperator of the loss channel.
    """
    kraus_list = []
    for i in range(N):
        prefactor = (1 - np.exp(- decay_rate * t))**(i/2) / np.sqrt(float(math.factorial(i)))
        K_i = prefactor * (- decay_rate * t / 2 * qutip.num(N)).expm() * qutip.destroy(N)**i
        kraus_list.append(K_i)
    # Superoperator of the loss channel
    channel = qutip.superop_reps.kraus_to_super(kraus_list)
    return channel


def dephasing_channel(N:int, dephasing_rate:float, t:float) -> qutip.Qobj:
    """Returns the superoperator of the dephasing channel for a single bosonic mode.

    Args:
        N (int): Hilbert space dimension of one bosonic mode (truncated at N-1 excitations).
        dephasing_rate (float): dephasing rate γ of the cavity.
        t (float): time duration of the channel.

    Returns:
        qutip.Qobj: superoperator of the channel.
    """
    kraus_list = []
    for i in range(N):
        prefactor = (dephasing_rate * t)**(i/2)/np.sqrt(math.factorial(i)) if i<20 else 0
        K_i = prefactor * (- dephasing_rate * t *(qutip.num(N)**2)/2).expm() * qutip.num(N)**i
        kraus_list.append(K_i)
    # Superoperator of the loss channel
    channel = qutip.superop_reps.kraus_to_super(kraus_list)
    return channel


def virtual_distillation_expectation_value(N:int, rho: qutip.Qobj, observable: qutip.Qobj, M: int) -> float:
    """Returns the expectation value of an observable after M-mode virtual distillation of a state rho.

    Args:
        N (int): Hilbert space dimension of one bosonic mode (truncated at N-1 excitations).
        rho (qutip.Qobj): density matrix of the state before virtual distillation.
        observable (qutip.Qobj): observable for which the expectation value is computed 
        M (int): number of copies used for virtual distillation.

    Returns:
        float: expectation value of the observable after virtual distillation.
    """
    return (observable * rho**M).tr() / (rho**M).tr()


def plot_theoretical_results(fig, ax, results: np.ndarray, rho_noisy_list: list, M_list: list, t_list: np.ndarray, observable_label: str, noise: tuple, plot_params: dict) -> None:
    """Plots the expectation value of the observable as a function of time for different states and virtual distillation parameters.

    Args:
        fig, ax
        results (np.ndarray): array of shape (number of states, number of M values, number of time samples) containing the expectation values.
        rho_noisy_list (list): list of tuples (rho_noisy, label) where rho_noisy is a list of density matrices at different times and label is a string describing the state.
        M_list (list): list of tuples (M, label) where M is the number of copies used for virtual distillation and label is a string describing it.
        t_list (np.ndarray): array of time samples corresponding to the results.
        observable_label (str): label for the observable.
        noise (tuple<float,str>): noise strength and label.
        plot_params (dict): {'colors':list, 'show_wigner': bool, 'legend_loc':str, 'xmin': float, 'xmax': float, 'tick_list': list, 'tick_params': dict} 
    """
    for i, (_, state_label) in enumerate(rho_noisy_list):
        plt.sca(ax[i])
        plt.plot(t_list,np.full(t_list.shape,(results[i,0,0])), linestyle="--", label='Pure state', c=plot_params['colors'][5]) # Pure state line
        for j, (_, M_label) in enumerate(M_list):
            plt.scatter(noise[0]*t_list, results[i,j,:], label=f'{M_label}', marker='+', s= 70, color=plot_params['colors'][j+2])
        plt.xlabel(f'$\\{noise[1]} t$', fontsize=25)
        plt.ylabel(observable_label, fontsize=25)
        plt.title(state_label, fontsize=25)
        if plot_params['legend_loc'] == 'outside right' and i==0:
            plt.legend(loc= "lower center", fontsize=15)
            plt.legend(loc= 'center left', bbox_to_anchor=(3.4,0.5),fontsize=15)
        elif plot_params['legend_loc'] == 'inside last' and i==results.shape[0]-1:
            ax[i].legend(loc="upper right",fontsize=15)

        plt.xlim(plot_params['xmin'], plot_params['xmax'])

        plt.xticks(fontsize= 18)
        plt.yticks(fontsize= 18)
        # --- Tick spacing (6 ticks total) ---
        ax[i].xaxis.set_major_locator(ticker.MaxNLocator(6))
        ax[i].yaxis.set_major_locator(ticker.MaxNLocator(6))

        # --- Tick formatting (max 2 decimals) ---
        ax[i].xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        ax[i].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        
        ax[i].text( -0.15, 0.95, plot_params['labels'][i],
            transform=ax[i].transAxes,
            fontsize=20,
            verticalalignment='top')

    if plot_params['show_wigner']:
        bound = 5
        xvec = np.linspace(-bound, bound, 50)
        yvec = np.linspace(-bound, bound, 50)
        wigner_list = [qutip.wigner(state_list[0], xvec, yvec) for state_list, _ in rho_noisy_list] # Wigner functions of the states at t=0 (pure states)
        vmax = max([abs(wigner).max() for wigner in wigner_list])
        vmin = -vmax
        ticks = np.round([0, vmax],2)
        for i, (state_list, _) in enumerate(rho_noisy_list):
            # Create inset
            axins = inset_axes(ax[i],width="35%", height="35%",loc="lower left")  # [x, y, width, height] in axes fraction
            axins.set_aspect('equal', adjustable='box')

            im = axins.contourf(xvec, yvec, wigner_list[i],100,cmap='RdBu',vmin=vmin,vmax=vmax)
            if i==2:
                cax = inset_axes(axins,width="5%", height="100%",loc="lower left",bbox_to_anchor=(1.05, 0, 1, 1),bbox_transform=axins.transAxes,borderpad=0)
                cb= plt.colorbar(im, cax= cax)
                cb.set_ticks(ticks)
                cb.ax.tick_params(labelsize=10)

            axins.set_xlim(-bound, bound)
            axins.set_ylim(-bound, bound)

            axins.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
            axins.set_xticks(plot_params['tick_list'])
            axins.set_yticks(plot_params['tick_list'])
            if plot_params['tick_params']:
                axins.tick_params(**plot_params['tick_params'])