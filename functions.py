import qutip
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.colors as colors


def kraus_op(N, k, t, kappa, nb_copies=1, id_copy=1):
    # returns Kraus operator as defined above
    # for multimode (nb_copies > 1), returns 1 x … x 1 x A_k x … x 1 with A_k in id_copy position
    op = (-kappa*t/2*qutip.num(N)).expm()
    op = op * qutip.destroy(N)**k
    prefactor = (1-np.exp(-kappa*t))**(k/2)/(np.sqrt(math.factorial(k)))
    op = prefactor * op
    return qutip.tensor([qutip.identity(N)]*(id_copy) + [op] + [qutip.identity(N)]*(nb_copies-id_copy-1))

# Creation of the states
def create_states(N:int, t_list:np.ndarray, rho_0:qutip.qobj.Qobj,
                  kappa:float,nb_copies:int=1) -> list[qutip.qobj.Qobj]:
    ''' Applies the loss channel to the state rho_0 and returns the temporal evolution
    
    Input :
       N : int                     dimension of the Hilbert space
       t_list : np.ndarray[float]  (size m) list with the input times
       kappa : float               decay rate
       rho_0 : qutip.qobj.Qobj    initial state
       nb_copies : int            number of copies on which to apply the loss channel
    Output :
      rho_list : list[qutip.qobj.Qobj] (size m)  list of noisy states'''

    rho_list = []
    for t in t_list:
        rho_init = rho_0
        for id_copy in range(nb_copies):
            rho = qutip.tensor([qutip.Qobj(np.zeros((N,N)))]*nb_copies) # 0 matrix
            for k in range(N): # not useful to go beyond N, the operator is 0
                kraus = kraus_op(N, k, t, kappa, nb_copies, id_copy)
                rho = rho + kraus * rho_init * kraus.dag()
            rho_init = rho
        rho_list.append(rho)
    return rho_list

def create_F3(N:int, sines:np.ndarray, noise:np.ndarray) -> tuple[qutip.qobj.Qobj]:
    ''' Creates the F3 gate with the given sines and noise on the angles
    
    Input :
       N : int                     dimension of the Hilbert space
       sines : np.ndarray[float]  (size 3) sines of the angles
       noise : np.ndarray[float]  (size 3) noise on the angles
    Output :
       bs1, bs2, bs3 : qutip.qobj.Qobj  beam splitters 1, 2 and 3'''
    a1 = qutip.tensor(qutip.destroy(N),  qutip.identity(N), qutip.identity(N))
    a2 = qutip.tensor(qutip.identity(N), qutip.destroy(N),  qutip.identity(N))
    a3 = qutip.tensor(qutip.identity(N), qutip.identity(N), qutip.destroy(N))
    t1, t2, t3 = np.arcsin(sines) + noise
    bs1 = (1j*t1*(a1.dag()*a2 + a2.dag()*a1)).expm()
    bs2 = (1j*t2*(a3.dag()*a2 + a2.dag()*a3)).expm()
    bs3 = (1j*t3*(a1.dag()*a3 + a3.dag()*a1)).expm()
    return bs1, bs2, bs3

    # redifining the modes up to global phase (useless)

    #left_redef  = (-2j*np.pi/3*n1 - 5j*np.pi/6*n2 ).expm()
    #right_redef = (-1j*np.pi/2*n1 - 1j*np.pi/2*n2).expm()



def perform_protocol(N:int, rho_list : list[qutip.qobj.Qobj], F_list : list[tuple[qutip.qobj.Qobj]],
                     M_list : list[int] =[3,4,1], losses = True,
                     uncoherent_t_loss :float = 5)-> tuple[np.ndarray]:
    ''' Performs the full 3-mode VD protocol on each input state and for each noisy F_3 gate
    Returns the 2-dimensional array with all results. Also returns theoretical results for comparison (see plots)
    
    Input :
       N : int                     dimension of the Hilbert space
       rho_list : list[qutip.qobj.Qobj]  (size n)  list of input states
       F_list :   list[tuple[qutip.qobj.Qobj]]  (size m) list of noisy beam splitters forming the F_3 gate
       M_list :   list[int]              (size k) list of nb_modes for theoretical results
       losses :   bool                   whether to include losses in the simulation
       uncoherent_t_loss : float         if losses=True, compute the losses with this value (in μs)
    Output :
       result :    np.array (size (n, m))  numerical results of the simulation
       result_loss : np.array (size (n,m)) adding losses in between
       result_th : np.array (size (k, n))  theoretical results for a perfect protocol (default: 3-mode, 4-mode and single-mode)
    '''
    # define the operators
    n1 = qutip.tensor(qutip.num(N),  qutip.identity(N), qutip.identity(N))
    n2 = qutip.tensor(qutip.identity(N), qutip.num(N),  qutip.identity(N))
    n3 = qutip.tensor(qutip.identity(N), qutip.identity(N), qutip.num(N))
    denominator_op = (2j*np.pi/3*(n1+2*n2)).expm()
    numerator_op = (n1+n2+n3)/3 * denominator_op
    # initialize the results
    nb_samples = len(rho_list)
    result = np.zeros((nb_samples, len(F_list)), dtype='complex')
    result_loss = np.zeros((nb_samples, len(F_list)), dtype='complex')
    result_th = np.zeros((len(M_list), nb_samples), dtype='complex')

    for (i,rho) in enumerate(rho_list):
        # create the 3 copies
        rho_full = qutip.tensor([rho]*3)
        for (j,(bs1, bs2, bs3)) in enumerate(F_list):
            # apply the F_3 gate
            F3 = bs1 * bs2 * bs3 # no losses in between beam splitters
            rho_tilde = F3 * rho_full * F3.dag()
            # compute the expectation values
            numerator = qutip.expect(numerator_op, rho_tilde)
            denominator = qutip.expect(denominator_op, rho_tilde)
            result[i,j]=(numerator/denominator)
            # WITH losses
            if losses:
                rho_1 = bs3 * rho_full * bs3.dag()
                rho_1_noisy = create_states([uncoherent_t_loss/3], rho_1, 3)[0]
                rho_2 = bs2 * rho_1_noisy * bs2.dag()
                rho_2_noisy = create_states([uncoherent_t_loss/3], rho_2, 3)[0]
                rho_3 = bs1 * rho_2_noisy * bs1.dag()
                rho_3_noisy = create_states([uncoherent_t_loss/3], rho_3, 3)[0]
                result_loss[i,j] = qutip.expect(numerator_op, rho_3_noisy)/qutip.expect(denominator_op, rho_3_noisy)
        # theoretical results
        for (k, nb_mode) in enumerate(M_list):
            result_th[k,i] = (qutip.num(N)* rho**nb_mode).tr() / (rho**nb_mode).tr()

    return result, result_loss, result_th


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
    print(S3.check_isunitary()) # this should be equal to the shift operator

    for i in range(3):
        x = [n-1 if i==0 else n, n-1 if i==1 else n, n-1 if i==2 else n]
        for j in range(3):
            y = [n-1 if j==0 else n, n-1 if j==1 else n, n-1 if j==2 else n]
            print(f"<{x[0]}{x[1]}{x[2]}|S|{y[0]}{y[1]}{y[2]}> = ",
                abs(((qutip.fock([N,N,N],x).dag() * S3 * qutip.fock([N,N,N],y)).data.toarray()[0][0])))
            

def plot_all(t_list :np.ndarray, result :np.ndarray, result_loss: np.ndarray, result_th_wlabels: list, 
             remove_extreme : None | tuple[int] = None, title:str="Fock state", eps_std:float=None,
             uncoherent_t_loss:float=None, kappa:float=None,
             show_curves:int=None, losses=True) -> None:
    ''' Creates the plots with the given results
    
    Input:
        t_list : np.ndarray[float]   size (n)    list of x-axis coordinates
        result : np.ndarray[complex] size (n, m) array of numerical VD results
        result_loss : np.ndarray[complex] size (n,m) same with inter-losses between beam splitters
        result_th_wlabels : list[tuple[np.ndarray size (n), string]]   Other theoretical results for comparison
        remove_extreme : None | tuple[int]   gives a range of values to keep
        title : string            type of input states we are looking at
        eps_std : float           standard deviation of the coherent noise
        uncoherent_t_loss : float time between losses
        kappa : float             decay rate
        show_curves : None | int       exact solutions for Fock state theoretical curves
    '''
    if (remove_extreme is None):
        result_corr = result
        result_loss_corr = result_loss
    else :
        #remove extreme points
        extr_max, extr_min = remove_extreme
        result_corr = result[:, np.all(np.logical_and(np.abs(result)<extr_max, np.abs(result)>extr_min),0)]
        result_loss_corr = result_loss[:, np.all(np.logical_and(np.abs(result_loss)<extr_max, 
                                                                np.abs(result_loss)>extr_min),0)]
    colors = ['green', 'blue', 'pink', 'orange', 'red', 'gray', 'black']
    plt.errorbar(t_list, np.abs(result_corr.mean(1)), result_corr.std(1), np.zeros_like(t_list), 
                    barsabove=True, elinewidth=.5,fmt='.',label="VD (3 copies) coherent noise", c=colors[0], capsize=3)
    if losses:
        plt.errorbar(t_list, np.abs(result_loss_corr.mean(1)), result_loss_corr.std(1), np.zeros_like(t_list), 
                    barsabove=True, elinewidth=.5,fmt='.',label="VD (3 copies) coherent + loss ", c=colors[1], capsize=2)
    for (i,(result_th, label)) in enumerate(result_th_wlabels):
        plt.scatter(t_list, np.abs(result_th), label=label, marker='+', c=colors[i+3])
    plt.plot(t_list,np.full(t_list.shape,(result_th_wlabels[0][0][0]).real),
             linestyle="--", label='Pure state', c=colors[2])
    if show_curves is not None:
        n = show_curves
        t_list_complete = np.linspace(t_list[0], t_list[-1], 1000)
        no_VD_curve_th = n * np.exp(-kappa * t_list_complete)
        plt.plot(t_list_complete, no_VD_curve_th, c=colors[3], linewidth = 0.5)
        # only for n=2
        temporary = (np.exp(kappa * t_list_complete)-1)
        VD_3_curve_th = (2+8*temporary**3)/(1+8*temporary**3+temporary**6)
        plt.plot(t_list_complete, VD_3_curve_th, c=colors[4], linewidth =0.4)
        VD_4_curve_th = (2+16*temporary**4)/(1+16*temporary**4+temporary**8)
        plt.plot(t_list_complete, VD_4_curve_th, c=colors[5], linewidth=0.4)
    plt.title(f"VD efficiency with coherent error ε~N(0,σ), σ={eps_std}"+
              (f"\nUncoherent loss τ={uncoherent_t_loss}μs" if losses else "")+
              f"\nInitial input : noisy "+title)
    plt.xlabel(f"Time (µs)")
    plt.ylabel("Expected value")
    plt.legend()
    plt.show()



# formula for finitely squeezed GKP state
def gkp_state(delta, N):
    x_squeez = -np.log(np.sqrt(np.sinh(delta**2)*np.cosh(delta**2)))
    p_squeez = -np.pi*np.tanh(delta**2)
    squeezed_vac = qutip.squeeze(N, x_squeez) * qutip.basis(N,0)
    s_bound = round(np.sqrt(-2/p_squeez))
    #print(f"x_squeez:{x_squeez}, p_squeez:{p_squeez}, s_bound:{s_bound}")
    
    result = qutip.Qobj(np.zeros((N)))
    for s in range(-s_bound,s_bound+1):
        term = qutip.displace(N, np.sqrt(np.pi)*s) * squeezed_vac
        term = term * np.exp((s**2)*p_squeez)
        result = result + term
    #print(result.norm())
    return result.unit()

def cat_state(alpha, N):
    return (qutip.coherent(N, alpha) + qutip.coherent(N, -alpha)).unit()

# plot the wigner function of a state
def plot_wigner(psi, xvec, yvec, fig, ax, cmap, title='Wigner function'):
    rho = qutip.ket2dm(psi)
    wigner = qutip.wigner(rho, xvec, yvec)
    plot = ax.contourf(xvec, yvec, wigner, 100, norm=colors.CenteredNorm(), cmap=cmap)
    ax.set_title(title)
    fig.colorbar(plot, ax=ax)