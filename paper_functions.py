
import qutip
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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




def create_noisy_list(N, channel_type, rate, t_list, rho_pure_list):
    loss_channel_list = [channel_type(N, rate, t) for t in t_list]
    return [ ([channel(rho) for channel in loss_channel_list], label) for (rho, label) in rho_pure_list]

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



def perform_protocol(N, rho_noisy_list, M_list, observable):
    ''' Perform the virtual distillation protocol 
    Input:
        N: Hilbert space dimension of one bosonic mode (truncated at N-1 excitations)
        rho_noisy_list: list of tuples (rho_noisy, label)
        M_list: list of tuples (M, label)
    Output:
        results: array of shape (len(rho_noisy_list), len(M_list), len'''

    results = np.zeros((len(rho_noisy_list), len(M_list), len(rho_noisy_list[0][0])), dtype='complex')
    for i, (rho_noisy, _) in enumerate(rho_noisy_list):
        for j, (M, _) in enumerate(M_list):
            for k, rho in enumerate(rho_noisy):
                results[i,j,k] = virtual_distillation_expectation_value(N, rho, observable, M)
    return results



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
        plot_params (dict): {'colors':list, 'show_wigner': bool, 'legend_loc':str, 'xmin': float, 'xmax': float, 
                'tick_list': list, 'tick_params': dict, 'labels': list, 'wigner_cmap': str, 'wigner_vmax': float or None, 'idx_wigner_colorbar': int} 
    """
    for i, (_, state_label) in enumerate(rho_noisy_list):
        plt.sca(ax[i])
        plt.plot(t_list,np.full(t_list.shape,(results[i,0,0])), linestyle="--", label='Pure state', c=plot_params['colors'][5]) # Pure state line
        for j, (_, M_label) in enumerate(M_list):
            plt.scatter(noise[0]*t_list, results[i,j,:], label=f'{M_label}', marker='+', s= 70, color=plot_params['colors'][j+2])
        plt.xlabel(f'$\\{noise[1]} t$', fontsize=25)
        plt.ylabel(observable_label, fontsize=25)
        plt.title(state_label, fontsize=25)
        legend_font_size = plot_params.get('legend_font_size', 15)
        if plot_params['legend_loc'] == 'outside right' and i==0:
            plt.legend(loc= "lower center", fontsize=legend_font_size)
            plt.legend(loc= 'center left', bbox_to_anchor=(3.4,0.5),fontsize=15)
        elif plot_params['legend_loc'] == 'inside last' and i==results.shape[0]-1:
            ax[i].legend(loc="upper right",fontsize=legend_font_size)
        elif plot_params['legend_loc'] == 'inside bottom' and i==0:
            ax[i].legend(loc="lower left",fontsize=legend_font_size)

        plt.xlim(plot_params['xmin'], plot_params['xmax'])

        plt.xticks(fontsize= 18)
        plt.yticks(fontsize= 18)
        # --- Tick spacing (6 ticks total) ---
        ax[i].xaxis.set_major_locator(ticker.MaxNLocator(6))
        ax[i].yaxis.set_major_locator(ticker.MaxNLocator(6))

        # --- Tick formatting (max 2 decimals) ---
        ax[i].xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        ax[i].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        
        ax[i].text( -0.182, 1.03, plot_params['labels'][i],
            transform=ax[i].transAxes,
            fontsize=35,
            verticalalignment='top')

    if plot_params['show_wigner']:
        bound = 5
        xvec = np.linspace(-bound, bound, 50)
        yvec = np.linspace(-bound, bound, 50)
        wigner_list = [qutip.wigner(state_list[0], xvec, yvec) for state_list, _ in rho_noisy_list] # Wigner functions of the states at t=0 (pure states)
        if plot_params['wigner_vmax'] is not None:
            vmax = plot_params['wigner_vmax']
        else:
            vmax = max([abs(wigner).max() for wigner in wigner_list])
        vmin = -vmax
        # cheat trick to have the same colorbar for all insets: we create a dummy contourf plot with the same colormap and vmin/vmax, and use it to create the colorbar


        for i, (state_list, _) in enumerate(rho_noisy_list):
            # Create inset
            axins = ax[i].inset_axes([0.01, 0.01, 0.35, 0.35]) #width="35%", height="35%",loc="lower left")  # [x, y, width, height] in axes fraction
            axins.set_aspect('equal', adjustable='box')
            # trick
            wigner_list[i][0,0] = vmax
            wigner_list[i][-1,-1] = vmin
            im = axins.contourf(xvec, yvec, wigner_list[i],100,cmap=plot_params['wigner_cmap'],vmin=vmin,vmax=vmax)
            if i==plot_params['idx_wigner_colorbar']:
                cax = axins.inset_axes([1.05,0,0.05,1])
                #cax = inset_axes(axins,width="5%", height="100%",loc="lower left",bbox_to_anchor=(1.05, 0, 1, 1),bbox_transform=axins.transAxes,borderpad=0)
                #norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=False)
                #cb = plt.colorbar(cmap= plot_params['wigner_cmap'], norm=norm, cax=cax)
                #dummy_im = axins.contourf(xvec, yvec, wigner_list[i], alpha=0.6, cmap=plot_params['wigner_cmap'], vmin=vmin, vmax=vmax)
                cb= plt.colorbar(im, cax= cax) #, cmap=plot_params['wigner_cmap']) #, vmin=vmin, vmax=vmax)
                ticks = np.round(np.linspace(vmin, vmax, 7),1)
                cb.set_ticks(ticks)
                cb.ax.tick_params(labelsize=10)

            axins.set_xlim(-bound, bound)
            axins.set_ylim(-bound, bound)

            axins.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
            axins.set_xticks(plot_params['tick_list'])
            axins.set_yticks(plot_params['tick_list'])
            if plot_params['tick_params']:
                axins.tick_params(**plot_params['tick_params'])

def loss_kraus_op(N, k, t, kappa, nb_copies=1, id_copy=1):
    # returns Kraus operator for loss channel as defined above
    # for multimode (nb_copies > 1), returns 1 x … x 1 x A_k x … x 1 with A_k in id_copy position
    op = (-kappa*t/2*qutip.num(N)).expm()
    op = op * qutip.destroy(N)**k
    prefactor = (1 - np.exp(-kappa*t))**(k/2) / np.sqrt(float(math.factorial(k)))
    op = prefactor * op
    return qutip.tensor([qutip.identity(N)]*(id_copy) + [op] + [qutip.identity(N)]*(nb_copies-id_copy-1))

def apply_loss_channel(N, rho_init, t, kappa_list, nb_copies=1):
    for id_copy in range(nb_copies):
        rho = qutip.tensor([qutip.Qobj(np.zeros((N,N)))]*nb_copies) # 0 matrix
        for k in range(N): # not useful to go beyond N, the operator is 0
            kraus = loss_kraus_op(N, k, t, kappa_list[id_copy], nb_copies, id_copy)
            rho = rho + kraus * rho_init * kraus.dag()
        rho_init = rho
    return rho
    

def compute_noisy_states_losses(N:int, t_list:np.ndarray, rho_0:qutip.qobj.Qobj, nb_protocol_samples:int,
                  kappa_mean:float, kappa_std:float, nb_copies:int=1, print_progress:bool=False) -> list[qutip.qobj.Qobj]:
    ''' Applies the loss channel to the state rho_0 and returns the temporal evolution
    
    Input :
       N : int                     dimension of the Hilbert space
       t_list : np.ndarray[float]  (size m) list with the input times
       rho_0 : qutip.qobj.Qobj    initial pure state
       nb_protocol_samples : int   number of samples to perform for each time (to have error bars)
       kappa_mean : float         mean decay rate κ of the cavity
       kappa_std : float          standard deviation of the decay rate κ (gaussian distribution)
       nb_copies : int            number of copies on which to apply the loss channel
       print_progress : bool      whether to print the progress of the computation (for long computations)
    Output :
      rho_list : list[list[qutip.qobj.Qobj]] (size len(t_list)*nb_protocol_samples)  list of noisy states'''

    rho_list = []
    kappa_arr = np.random.normal(kappa_mean, kappa_std, (len(t_list),nb_protocol_samples,nb_copies))  

    for (it,t) in enumerate(t_list):
        print(f"({it+1}/{len(t_list)})", end="") if print_progress else None
        rho_samples = []
        for ip in range(nb_protocol_samples):
            print("*", end="") if print_progress else None
            kappa_list = kappa_arr[it,ip,:]
            rho_init = qutip.tensor([rho_0 for _ in range(nb_copies)])
            rho = apply_loss_channel(N, rho_init, t, kappa_arr[it,ip,:], nb_copies)
            rho_samples.append(rho)
        rho_list.append(rho_samples)
        #print() if print_progress else None
    return rho_list

def create_F3(N:int, sines:np.ndarray, noise:np.ndarray, prepost:bool=True) -> tuple[qutip.qobj.Qobj]:
    ''' Creates the F3 gate as a triplet of beam-splitter gates with given sines and noise on the angles
    
    Input :
       N : int                     dimension of the Hilbert space
       sines : np.ndarray[float]  (size 3) sines of the angles
       noise : np.ndarray[float]  (size 3) noise on the angles
       prepost : bool             whether to add the redefinitions of the modes before and after the gate
    Output :
       (pre*) bs1, bs2, bs3 (*post): qutip.qobj.Qobj  beam splitters 1, 2 and 3'''
    t1, t2, t3 = np.arcsin(sines) + noise
    a1daga2 = qutip.tensor(qutip.create(N), qutip.destroy(N))
    a2daga3 = qutip.tensor(qutip.create(N), qutip.destroy(N))
    a1daga3 = qutip.tensor(qutip.create(N), qutip.identity(N), qutip.destroy(N))
    bs1 = qutip.tensor((1j*t1*(a1daga2 + a1daga2.dag())).expm(), qutip.identity(N))
    bs2 = qutip.tensor(qutip.identity(N), (1j*t2*(a2daga3 + a2daga3.dag())).expm())
    bs3 = (1j*t3*(a1daga3 + a1daga3.dag())).expm()
    #print(np.shape(bs1))
    if prepost:
        ps1 = (-5j*np.pi/6*qutip.num(N)).expm()
        ps2 = (-2j*np.pi/3*qutip.num(N)).expm()
        prephase_op = qutip.tensor(ps1, ps2, qutip.identity(N))
        ps_end_1 = (-1j*np.pi/2*qutip.num(N)).expm()
        ps_end_2 = (-1j*np.pi/2*qutip.num(N)).expm()
        postphase_op = qutip.tensor(ps_end_1, ps_end_2, qutip.identity(N))
        return prephase_op * bs1, bs3, bs2 * postphase_op # Changed the order to match with the notations in the report
    return bs1, bs3, bs2

def create_F3_list(N:int, initial_sin_angles:np.ndarray, noise_list:np.ndarray, print_progress:bool=False) -> list[list[tuple[qutip.qobj.Qobj]]]:
    ''' Creates a list of F3 gates with noise on the angles '''

    F_list = [] # list of all F gates (triplets of beam splitters)
    for i in range(len(noise_list)):
        print(f"({i+1}/{len(noise_list)})", end=" ") if print_progress else None
        F_line = []
        for noise in noise_list[i]:
            print("*", end="") if print_progress else None
            # Here we create the F3 gates with the noisy angles
            F_line.append(create_F3(N, initial_sin_angles, noise))
        F_list.append(F_line)
    return F_list


def check_S_operator(N, n, bs1, bs2, bs3):
    ''' Check the properties of the S operator
    Input :
       N : int                     dimension of the Hilbert space
       n : int                     number of photons
       bs1, bs2, bs3 : qutip.qobj.Qobj  beam splitters 1, 2 and 3
    Output :
       None'''
    n1 = qutip.tensor(qutip.num(N),  qutip.identity(N), qutip.identity(N))
    n2 = qutip.tensor(qutip.identity(N), qutip.num(N),  qutip.identity(N))
    denominator_op = (2j*np.pi/3*(n1+2*n2)).expm()
    F3 = bs1*bs2*bs3
    S3 = F3.dag() * denominator_op * F3
    print(S3.isunitary) # this should be equal to the shift operator

    for i in range(3):
        x = [n-1 if i==0 else n, n-1 if i==1 else n, n-1 if i==2 else n]
        for j in range(3):
            y = [n-1 if j==0 else n, n-1 if j==1 else n, n-1 if j==2 else n]
            print(f"<{x[0]}{x[1]}{x[2]}|S|{y[0]}{y[1]}{y[2]}> = {
                abs(((qutip.fock([N,N,N],x).dag() * S3 * qutip.fock([N,N,N],y)) )):.5f}") #.data.toarray()[0][0])))
            


def dephasing_kraus_op(N, l, gamma, nb_copies=1, id_copy=0):
    # returns Kraus operator as defined above
    # for multimode (nb_copies > 1), returns 1 x … x 1 x A_k x … x 1 with A_k in id_copy position
    op = (-gamma*(qutip.num(N)**2)/2).expm()
    op = op * qutip.num(N)**l
    prefactor = gamma**(l/2)/np.sqrt(math.factorial(l)) if l<20 else 0
    op = prefactor * op
    return qutip.tensor([qutip.identity(N)]*(id_copy) + [op] + [qutip.identity(N)]*(nb_copies-id_copy-1))

def apply_dephasing_channel(N, rho_init, t, gamma_list, nb_copies=1):
    for id_copy in range(nb_copies):
        rho = qutip.tensor([qutip.Qobj(np.zeros((N,N)))]*nb_copies) # 0 matrix
        for l in range(N): # not useful to go beyond N, the operator is 0
            kraus = loss_kraus_op(N, l, t, gamma_list[id_copy], nb_copies, id_copy)
            rho = rho + kraus * rho_init * kraus.dag()
        rho_init = rho
    return rho

def perform_protocol_errors(N:int,
                     #nb_protocol_samples:int,
                     rho_tot_list : list[list[qutip.qobj.Qobj]], 
                     F_list : list[tuple[qutip.qobj.Qobj]],
                     kappa_mean: float,
                     kappa_std: float,
                     gamma_mean: float,
                     gamma_std: float,
                     interleaved_time: float,
                     operators: list[qutip.qobj.Qobj],
                     print_progression:bool = False)-> tuple[np.ndarray]:
    ''' Performs the full 3-mode VD protocol on each input state and for each noisy F_3 gate
    Returns the 2-dimensional array with all results. Adds interleaved losses and dephasing.

    Input :
       N : int                     dimension of the Hilbert space
       #nb_protocol_samples : int   number of samples to perform for each time (to have error bars)
       rho_tot_list : list[list[[qutip.qobj.Qobj]]  (size n*m)  list of input states
       F_list :   list[tuple[qutip.qobj.Qobj]]  (size n*m) list of noisy beam splitters forming the F_3 gate
       kappa_mean : float         mean decay rate κ of the cavity
       kappa_std : float          standard deviation of the decay rate κ (gaussian distribution)
       gamma_mean : float         mean dephasing rate γ of the cavity
       gamma_std : float          standard deviation of the dephasing rate γ (gaussian distribution)
       interleaved_time : float       time for which to apply the interleaved noise (losses and dephasing)
       operators : list[qutip.qobj.Qobj] list of operators to compute the expectation values on
       print_progress : bool      whether to print the progress of the computation (for long computations)
       
    Output :
       result :    np.array (size (n, m))  numerical results of the simulation
       result_errors : np.array (size (n,m)) error bars
    '''
    
    # initialize the results
    n, m = len(rho_tot_list), len(rho_tot_list[0])
    result = np.zeros((n,m), dtype='complex')

    kappa_arr = np.random.normal(kappa_mean, kappa_std, (n, m, 3))
    gamma_arr = np.random.normal(gamma_mean, gamma_std, (n, m, 3))


    for i in range(n):
        print(f"({i+1}/{n})", end=" ") if print_progression else None
        for (j, rho) in enumerate(rho_tot_list[i]):
            print("*", end="") if print_progression else None
            # apply the F_3 gate
            bs1, bs2, bs3 = F_list[i][j]
            F3 = bs1 * bs2 * bs3 # no losses in between beam splitters
            rho = F3 * rho * F3.dag()
            # compute the expectation values
            for gate in (bs3, bs2, bs1):
                rho = gate * rho * gate.dag()
                rho = apply_loss_channel(N, rho, interleaved_time, kappa_arr[i,j], 3)
                rho = apply_dephasing_channel(N, rho, interleaved_time, gamma_arr[i,j], 3)
            result[i,j] = qutip.expect(operators[0], rho)/qutip.expect(operators[1], rho)
    return result

