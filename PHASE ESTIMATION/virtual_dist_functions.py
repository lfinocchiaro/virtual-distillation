"""Utilities for virtual-distillation diagnostics in bosonic Hilbert spaces.

The module groups state preparation, noise channels, spectral virtual
distillation, finite-shot sampling models, Wigner reconstruction, and plotting
helpers for the example figures.
"""

import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

from scipy.linalg import expm, schur

from functools import lru_cache

from itertools import product

from math import comb, factorial

from pathlib import Path



matplotlib.rcParams['text.usetex'] = False

matplotlib.rcParams['mathtext.fontset'] = 'cm'

matplotlib.rcParams['font.family'] = 'serif'



# --- State preparation and cutoff helpers ---

def basis_state(dim, level):
    ket = np.zeros(dim, dtype=complex)
    ket[level] = 1.0
    return ket

def density_matrix(state):
    state = np.asarray(state, dtype=complex)
    return np.outer(state, state.conj())

def coherent_state(dim, alpha):
    alpha = complex(alpha)
    coeffs = np.zeros(dim, dtype=complex)
    coeffs[0] = np.exp(-0.5 * abs(alpha) ** 2)
    for n in range(1, dim):
        coeffs[n] = coeffs[n - 1] * alpha / np.sqrt(n)
    return coeffs

def even_cat_state(dim, alpha):
    state = coherent_state(dim, alpha) + coherent_state(dim, -alpha)
    state /= np.linalg.norm(state)
    return state

def square_cat_state(dim, alpha):
    alpha = complex(alpha)
    state = (
        coherent_state(dim, alpha)
        + coherent_state(dim, -alpha)
        + coherent_state(dim, 1j * alpha)
        + coherent_state(dim, -1j * alpha)
    )
    return normalize_state(state)

def tri_cat_state(dim, alpha):
    alpha = complex(alpha)
    omega = np.exp(2j * np.pi / 3.0)
    state = (
        coherent_state(dim, alpha)
        + coherent_state(dim, alpha * omega)
        + coherent_state(dim, alpha * (omega ** 2))
    )
    return normalize_state(state)

def two_mode_fock_index(local_dim, n_a, n_b):
    local_dim = int(local_dim)
    n_a = int(n_a)
    n_b = int(n_b)
    if not (0 <= n_a < local_dim and 0 <= n_b < local_dim):
        raise ValueError('mode occupations must lie inside the local cutoff')
    return n_a * local_dim + n_b

def two_mode_basis_state(local_dim, n_a, n_b):
    return np.kron(basis_state(local_dim, n_a), basis_state(local_dim, n_b))

def two_mode_noon_state(local_dim, n):
    n = int(n)
    if n <= 0:
        raise ValueError('n must be positive for a N00N state')
    if n >= int(local_dim):
        raise ValueError('n must be smaller than the local cutoff')
    state = two_mode_basis_state(local_dim, n, 0) + two_mode_basis_state(local_dim, 0, n)
    return normalize_state(state)

def recommended_cat_cutoff(alpha, safety_margin=10.0, sigma_factor=8.0):
    alpha_abs = abs(alpha)
    cutoff = int(np.ceil(alpha_abs ** 2 + sigma_factor * alpha_abs + safety_margin))
    return max(12, cutoff)

def recommended_square_cat_cutoff(alpha, extra_margin=6.0):
    return recommended_cat_cutoff(alpha) + int(np.ceil(extra_margin))

def annihilation_operator(dim):
    op = np.zeros((dim, dim), dtype=complex)
    for n in range(1, dim):
        op[n - 1, n] = np.sqrt(n)
    return op

def normalize_state(state):
    return state / np.linalg.norm(state)

def state_tail_probability(state, tail_levels=4):
    tail_levels = min(tail_levels, state.shape[0])
    return float(np.sum(np.abs(state[-tail_levels:]) ** 2))

def build_adaptive_state(state_builder, parameter, initial_dim, step=10, max_dim=160, tail_levels=4, tail_tol=1e-7):
    dim = int(initial_dim)
    while True:
        state = normalize_state(state_builder(dim, parameter))
        tail = state_tail_probability(state, tail_levels=tail_levels)
        if tail < tail_tol or dim >= max_dim:
            return state, dim, tail
        dim += step



# --- Operators, noise channels, and two-mode models ---

def number_operator(dim):
    return np.diag(np.arange(dim, dtype=float))

def parity_operator(dim):
    return np.diag([(-1) ** n for n in range(dim)])

def trace_normalize_density_matrix(rho, atol=1e-12):
    rho_h = 0.5 * (np.asarray(rho, dtype=complex) + np.asarray(rho, dtype=complex).conj().T)
    trace = np.trace(rho_h)
    if abs(trace) <= atol:
        raise ValueError('density matrix must have non-zero trace')
    if not np.isclose(trace, 1.0, atol=atol):
        rho_h = rho_h / trace
    return rho_h

def apply_kraus_channel(rho, kraus_ops):
    rho = np.asarray(rho, dtype=complex)
    out = np.zeros_like(rho, dtype=complex)
    for kraus in kraus_ops:
        out += kraus @ rho @ kraus.conj().T
    return trace_normalize_density_matrix(out)

def pure_loss_kraus(dim, eta):
    eta = float(np.clip(eta, 0.0, 1.0))
    sqrt_eta = np.sqrt(eta)
    sqrt_loss = np.sqrt(1.0 - eta)
    kraus_ops = []
    for loss in range(dim):
        op = np.zeros((dim, dim), dtype=complex)
        for n in range(loss, dim):
            coeff = np.sqrt(float(comb(n, loss))) * (sqrt_eta ** (n - loss)) * (sqrt_loss ** loss)
            op[n - loss, n] = coeff
        kraus_ops.append(op)
    return kraus_ops

def pure_loss_channel(rho, eta):
    dim = rho.shape[0]
    return apply_kraus_channel(rho, pure_loss_kraus(dim, eta))

def dephasing_decay_matrix(dim, gamma_t):
    gamma_t = max(float(gamma_t), 0.0)
    number_values = np.arange(dim, dtype=float)
    differences = number_values[:, None] - number_values[None, :]
    return np.exp(-0.5 * gamma_t * (differences ** 2))

def dephasing_channel(rho, gamma_t):
    dim = rho.shape[0]
    damping = dephasing_decay_matrix(dim, gamma_t)
    out = np.asarray(rho, dtype=complex) * damping
    return trace_normalize_density_matrix(out)

def dephasing_channel_from_rate(rho, kappa_phi, time):
    return dephasing_channel(rho, float(kappa_phi) * float(time))

def two_mode_noon_loss_density_matrix(local_dim, n, eta):
    local_dim = int(local_dim)
    n = int(n)
    eta = float(np.clip(eta, 0.0, 1.0))
    rho = np.zeros((local_dim ** 2, local_dim ** 2), dtype=complex)
    for survivors in range(n + 1):
        weight = 0.5 * float(comb(n, survivors)) * (eta ** survivors) * ((1.0 - eta) ** (n - survivors))
        idx_a = two_mode_fock_index(local_dim, survivors, 0)
        idx_b = two_mode_fock_index(local_dim, 0, survivors)
        rho[idx_a, idx_a] += weight
        rho[idx_b, idx_b] += weight
    idx_n0 = two_mode_fock_index(local_dim, n, 0)
    idx_0n = two_mode_fock_index(local_dim, 0, n)
    coherence = 0.5 * (eta ** n)
    rho[idx_n0, idx_0n] += coherence
    rho[idx_0n, idx_n0] += coherence
    rho = 0.5 * (rho + rho.conj().T)
    trace = np.trace(rho)
    if not np.isclose(trace, 1.0):
        rho = rho / trace
    return rho

def two_mode_dephasing_decay_matrix(local_dim, gamma_t):
    local_dim = int(local_dim)
    gamma_t = max(float(gamma_t), 0.0)
    occupation_pairs = [(n_a, n_b) for n_a in range(local_dim) for n_b in range(local_dim)]
    damping = np.ones((local_dim ** 2, local_dim ** 2), dtype=float)
    for row, (n_a, n_b) in enumerate(occupation_pairs):
        for col, (m_a, m_b) in enumerate(occupation_pairs):
            diff_sq = (n_a - m_a) ** 2 + (n_b - m_b) ** 2
            damping[row, col] = np.exp(-0.5 * gamma_t * diff_sq)
    return damping

def two_mode_dephasing_channel(rho, local_dim, gamma_t):
    rho = np.asarray(rho, dtype=complex)
    expected_dim = int(local_dim) ** 2
    if rho.shape != (expected_dim, expected_dim):
        raise ValueError('rho shape must match local_dim ** 2')
    damping = two_mode_dephasing_decay_matrix(local_dim, gamma_t)
    return trace_normalize_density_matrix(rho * damping)

def two_mode_dephasing_channel_from_rate(rho, local_dim, kappa_phi, time):
    return two_mode_dephasing_channel(rho, local_dim, float(kappa_phi) * float(time))



# --- Unitary sampling models ---

def _as_rng(seed_or_rng=None):
    if isinstance(seed_or_rng, np.random.Generator):
        return seed_or_rng
    return np.random.default_rng(seed_or_rng)

def assert_unitary(operator, name='operator', atol=1e-8):
    operator = np.asarray(operator, dtype=complex)
    if operator.ndim != 2 or operator.shape[0] != operator.shape[1]:
        raise ValueError(f'{name} must be a square matrix')
    eye = np.eye(operator.shape[0], dtype=complex)
    error = np.linalg.norm(operator.conj().T @ operator - eye)
    if error > atol * max(1, operator.shape[0]):
        raise ValueError(f'{name} must be unitary for eigenvalue sampling; unitarity error={error:.3e}')

def build_unitary_sampling_model(unitary, atol=1e-8):
    unitary = np.asarray(unitary, dtype=complex)
    assert_unitary(unitary, name='measurement unitary', atol=atol)
    schur_matrix, basis = schur(unitary, output='complex')
    diagonal = np.diag(schur_matrix)
    off_diagonal_error = np.linalg.norm(schur_matrix - np.diag(diagonal))
    if off_diagonal_error > atol * max(1, unitary.shape[0]):
        raise ValueError(f'unitary Schur form is not diagonal enough; off-diagonal error={off_diagonal_error:.3e}')
    return {
        'eigenvalues': diagonal,
        'basis': basis,
        'off_diagonal_error': off_diagonal_error,
    }

def sampling_probabilities_from_model(rho_full, model):
    basis = model['basis']
    rotated_rho = basis.conj().T @ rho_full @ basis
    probabilities = np.real(np.diag(rotated_rho))
    probabilities = np.clip(probabilities, 0.0, None)
    total = probabilities.sum()
    if total <= 0.0:
        raise ValueError('sampling probabilities have non-positive total weight')
    return probabilities / total

def sample_unitary_expectation_from_model(rho_full, model, shots, rng=None):
    shots = int(shots)
    if shots <= 0:
        raise ValueError('shots must be positive')
    rng = _as_rng(rng)
    probabilities = sampling_probabilities_from_model(rho_full, model)
    eigenvalues = model['eigenvalues']
    outcomes = rng.choice(eigenvalues.size, size=shots, p=probabilities)
    weights = eigenvalues[outcomes]
    estimate = weights.mean()
    if shots > 1:
        real_standard_error = weights.real.std(ddof=1) / np.sqrt(shots)
        imag_standard_error = weights.imag.std(ddof=1) / np.sqrt(shots)
    else:
        real_standard_error = np.nan
        imag_standard_error = np.nan
    return {
        'estimate': estimate,
        'exact_from_probabilities': np.dot(probabilities, eigenvalues),
        'real_standard_error': real_standard_error,
        'imag_standard_error': imag_standard_error,
    }



# --- Expectation values and figure saving utility ---

def expectation_value(rho, op):
    return float(np.real_if_close(np.trace(op @ rho)))

def picture_path(folder, filename):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    return str(folder / filename)



# --- Number distributions and spectral virtual distillation ---

def number_distribution_from_rho(rho):
    rho_h = 0.5 * (rho + rho.conj().T)
    probabilities = np.real(np.diag(rho_h))
    probabilities = np.clip(probabilities, 0.0, None)
    total = probabilities.sum()
    if total <= 0.0:
        raise ValueError('number distribution has non-positive total weight')
    return probabilities / total

def two_mode_joint_number_distribution_from_rho(rho, local_dim):
    flat = finalize_probability_distribution(number_distribution_from_rho(rho))
    expected = int(local_dim) ** 2
    if flat.size != expected:
        raise ValueError('rho size does not match a two-mode cutoff of local_dim x local_dim')
    return flat.reshape(int(local_dim), int(local_dim))

def discrete_phase_grid(dim):
    return 2.0 * np.pi * np.arange(int(dim), dtype=float) / float(dim)

def two_mode_characteristic_function_from_distribution(joint_probabilities, phase_a_values=None, phase_b_values=None):
    joint = np.asarray(joint_probabilities, dtype=float)
    if joint.ndim != 2 or joint.shape[0] != joint.shape[1]:
        raise ValueError('joint_probabilities must be a square local_dim x local_dim array')
    local_dim = int(joint.shape[0])
    if phase_a_values is None:
        phase_a_values = discrete_phase_grid(local_dim)
    if phase_b_values is None:
        phase_b_values = discrete_phase_grid(local_dim)
    phase_a_values = np.asarray(phase_a_values, dtype=float)
    phase_b_values = np.asarray(phase_b_values, dtype=float)
    phase_kernel_a = np.exp(1j * np.outer(phase_a_values, np.arange(local_dim, dtype=float)))
    phase_kernel_b = np.exp(1j * np.outer(phase_b_values, np.arange(local_dim, dtype=float)))
    chi_values = phase_kernel_a @ joint.astype(complex) @ phase_kernel_b.T
    return phase_a_values, phase_b_values, np.asarray(chi_values, dtype=complex)

def sanitize_probability_distribution(probabilities, clip_tol=1e-10):
    values = np.nan_to_num(np.asarray(probabilities, dtype=complex), nan=0.0, posinf=0.0, neginf=0.0)
    real_values = values.real.copy()
    imag_abs_max = float(np.max(np.abs(values.imag))) if values.size else 0.0
    small_mask = np.abs(real_values) < clip_tol
    small_clipped_mass = float(np.sum(np.abs(real_values[small_mask]))) if np.any(small_mask) else 0.0
    real_values[small_mask] = 0.0
    negative_mask = real_values < 0.0
    negative_mass = float(-np.sum(real_values[negative_mask])) if np.any(negative_mask) else 0.0
    clipped = np.clip(real_values, 0.0, None)
    abs_fallback_used = False
    uniform_fallback_used = False
    total_after_clip = float(clipped.sum())
    if total_after_clip <= 0.0:
        abs_fallback_used = True
        clipped = np.abs(values)
        clipped[np.abs(clipped) < clip_tol] = 0.0
        clipped = np.asarray(clipped, dtype=float)
        total_after_clip = float(clipped.sum())
    if total_after_clip <= 0.0:
        uniform_fallback_used = True
        clipped = np.ones(values.shape[0], dtype=float)
        total_after_clip = float(clipped.sum())
    if total_after_clip <= 0.0:
        raise ValueError('reconstructed distribution has non-positive total weight')
    normalized = clipped / total_after_clip
    diagnostics = {
        'imag_abs_max': imag_abs_max,
        'small_clipped_mass': small_clipped_mass,
        'negative_mass': negative_mass,
        'renorm_before': total_after_clip,
        'renorm_correction': float(abs(1.0 - total_after_clip)),
        'min_real_before_clip': float(real_values.min()) if real_values.size else 0.0,
        'max_real_before_clip': float(real_values.max()) if real_values.size else 0.0,
        'fallback_used': abs_fallback_used or uniform_fallback_used,
        'abs_fallback_used': abs_fallback_used,
        'uniform_fallback_used': uniform_fallback_used,
    }
    return normalized, diagnostics

def finalize_probability_distribution(probabilities, clip_tol=1e-10):
    return sanitize_probability_distribution(probabilities, clip_tol=clip_tol)[0]

def spectral_decomposition_of_density_matrix(rho):
    rho_h = 0.5 * (rho + rho.conj().T)
    evals, evecs = np.linalg.eigh(rho_h)
    evals = np.clip(evals.real, 0.0, None)
    largest = float(evals.max())
    if largest <= 0.0:
        raise ValueError('rho must have at least one positive eigenvalue')
    return {
        'evals': evals,
        'evecs': evecs,
        'diag_weights': np.abs(evecs) ** 2,
        'largest': largest,
    }

def _scaled_spectral_weights(spectral, copies):
    copies = int(copies)
    scaled = (spectral['evals'] / spectral['largest']) ** copies
    denominator = float(np.sum(scaled))
    if denominator <= 0.0:
        raise ValueError('distilled spectral weights have non-positive total weight')
    return scaled, denominator

def exact_distilled_density_matrix_from_spectral(spectral, copies):
    scaled, denominator = _scaled_spectral_weights(spectral, copies)
    evecs = spectral['evecs']
    rho_distilled = (evecs * scaled) @ evecs.conj().T / denominator
    return 0.5 * (rho_distilled + rho_distilled.conj().T)



# --- Two-mode N00N characteristic-function diagnostics ---

def build_noon_characteristic_modsq_snapshot_data(local_dim, n, copies_list, snapshot_time, kappa, phase_points=121, noise_kind='loss'):
    n = int(n)
    local_dim = int(local_dim)
    if local_dim <= n:
        raise ValueError('local_dim must be at least n + 1 for a N00N state')
    phase_points = int(phase_points)
    phase_values = np.linspace(-np.pi, np.pi, phase_points, endpoint=False)
    rho0 = density_matrix(two_mode_noon_state(local_dim, n))
    if noise_kind == 'loss':
        eta = np.exp(-kappa * snapshot_time)
        noisy_rho = two_mode_noon_loss_density_matrix(local_dim, n, eta)
        noise_note = rf'loss snapshot: $t={snapshot_time:.1f}$, $\eta={eta:.3f}$'
    elif noise_kind == 'dephasing':
        gamma_t = float(kappa) * float(snapshot_time)
        noisy_rho = two_mode_dephasing_channel(rho0, local_dim, gamma_t)
        noise_note = rf'dephasing snapshot: $t={snapshot_time:.1f}$, $\gamma t={gamma_t:.3f}$'
    else:
        raise ValueError(f'unknown noise_kind={noise_kind!r}')
    spectral = spectral_decomposition_of_density_matrix(noisy_rho)
    scenario_states = [('Pure state', rho0), ('No VD', noisy_rho)]
    for copies in copies_list:
        scenario_states.append((rf'VD $M={copies}$', exact_distilled_density_matrix_from_spectral(spectral, copies)))
    maps = {}
    for label, rho in scenario_states:
        joint = two_mode_joint_number_distribution_from_rho(rho, local_dim)
        _, _, chi_values = two_mode_characteristic_function_from_distribution(joint, phase_values, phase_values)
        maps[label] = np.abs(chi_values) ** 2
    return {
        'maps': maps,
        'scenario_labels': [label for label, _ in scenario_states],
        'phase_values': phase_values,
        'snapshot_time': float(snapshot_time),
        'noise_kind': noise_kind,
        'noise_note': noise_note,
        'n': n,
        'local_dim': local_dim,
    }

def plot_noon_characteristic_modsq_snapshot(
    n=4,
    dim=8,
    copies_list=(2, 3, 4),
    snapshot_time=50.0,
    kappa=0.2 / 50.0,
    phase_points=121,
    noise_kind='loss',
    cmap='RdBu_r',
    figsize=None,
    save_path=None,
):
    data = build_noon_characteristic_modsq_snapshot_data(dim, n, copies_list, snapshot_time, kappa, phase_points=phase_points, noise_kind=noise_kind)
    scenario_labels = data['scenario_labels']
    if figsize is None:
        figsize = (2.7 * len(scenario_labels), 3.4)
    fig, axes = plt.subplots(1, len(scenario_labels), figsize=figsize, squeeze=False, sharex=True, sharey=True)
    image = None
    extent = (-1.0, 1.0, -1.0, 1.0)
    tick_values = [-1.0, -0.5, 0.0, 0.5, 1.0]
    tick_labels = ['-1.0', '-0.5', '0', '0.5', '1.0']
    for col, label in enumerate(scenario_labels):
        ax = axes[0, col]
        image = ax.imshow(
            data['maps'][label],
            origin='lower',
            extent=extent,
            vmin=0.0,
            vmax=1.0,
            cmap=cmap,
            aspect='equal',
            interpolation='bilinear',
        )
        ax.set_title(label, fontsize=NOON_TITLE_FONTSIZE)
        ax.set_xlabel(r'$\phi_1 / \pi$', fontsize=NOON_LABEL_FONTSIZE)
        if col == 0:
            ax.set_ylabel(r'$\phi_2 / \pi$', fontsize=NOON_LABEL_FONTSIZE)
        ax.set_xticks(tick_values)
        ax.set_yticks(tick_values)
        ax.set_xticklabels(tick_labels)
        ax.set_yticklabels(tick_labels)
        ax.tick_params(labelsize=NOON_TICK_FONTSIZE)
        if col != 0:
            ax.tick_params(axis='y', labelleft=False)
    fig.tight_layout(rect=(0.0, 0.0, 0.93, 1.0), w_pad=WIGNER_TIGHT_LAYOUT_W_PAD)
    colorbar_ax = fig.add_axes([0.94, 0.20, 0.018, 0.62])
    colorbar = fig.colorbar(image, cax=colorbar_ax)
    colorbar.set_label(r'$|\chi_M|^2$', fontsize=NOON_COLORBAR_FONTSIZE)
    colorbar.ax.tick_params(labelsize=NOON_COLORBAR_TICK_FONTSIZE)
    if save_path is not None:
        fig.savefig(save_path, dpi=250, bbox_inches='tight')
    return fig, axes, data



# --- Wigner and displaced-parity utilities ---

def _displacement_operator_from_ladder(a, adag, alpha):
    alpha = complex(alpha)
    generator = alpha * adag - np.conjugate(alpha) * a
    return expm(generator)

def displacement_operator(dim, alpha):
    a = annihilation_operator(dim)
    adag = a.conj().T
    return _displacement_operator_from_ladder(a, adag, alpha)

def displace_density_matrix(rho, alpha):
    D = displacement_operator(rho.shape[0], alpha)
    displaced = D.conj().T @ rho @ D
    return 0.5 * (displaced + displaced.conj().T)

def displaced_parity_operator(dim, alpha):
    D = displacement_operator(dim, alpha)
    parity = parity_operator(dim)
    return D @ parity @ D.conj().T

def wigner_value_from_rho(rho, alpha):
    parity = parity_operator(rho.shape[0])
    displaced = displace_density_matrix(rho, alpha)
    return float(np.real((2.0 / np.pi) * np.trace(parity @ displaced)))

def phase_space_radius(x_values, y_values):
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    return float(np.sqrt(np.max(x_values ** 2) + np.max(y_values ** 2)))

def build_displaced_parity_kernels(dim, x_values, y_values):
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    a = annihilation_operator(dim)
    adag = a.conj().T
    parity = parity_operator(dim)
    kernels = []
    for y in y_values:
        for x in x_values:
            D = _displacement_operator_from_ladder(a, adag, x + 1j * y)
            kernels.append((2.0 / np.pi) * (D @ parity @ D.conj().T))
    return {
        'x_values': x_values,
        'y_values': y_values,
        'kernels': np.asarray(kernels, dtype=complex),
        'shape': (len(y_values), len(x_values)),
    }

def wigner_map_from_kernels(rho, kernel_grid):
    rho_h = 0.5 * (rho + rho.conj().T)
    values = np.real(np.einsum('aij,ji->a', kernel_grid['kernels'], rho_h))
    return values.reshape(kernel_grid['shape'])



# --- Plot labels, state builders, and figure helpers ---

# Shared labels and noise annotations.

def paper_plot_path(filename):
    return picture_path('Paper_plots', filename)

def paper_alpha_title(family_name, alpha):
    return rf'{family_name} $\alpha = {float(alpha):.2f}$'

def paper_fock_title(level):
    return rf'Fock state $|{int(level)}\rangle$'

def paper_vd_label(copies):
    return rf'VD $M={int(copies)}$'

def paper_re_alpha_label():
    return r'$\mathrm{Re}(\alpha)$'

def paper_im_alpha_label():
    return r'$\mathrm{Im}(\alpha)$'

def paper_wigner_label():
    return r'$W(\alpha)$'

def paper_wigner_l2_label():
    return r'$\left\| \tilde{\rho}-|\psi_0\rangle\langle\psi_0|\right\|_2$'

WIGNER_TITLE_FONTSIZE = 16
WIGNER_LABEL_FONTSIZE = 14.5
WIGNER_TICK_FONTSIZE = 13
WIGNER_LEGEND_FONTSIZE = 12
WIGNER_COLORBAR_FONTSIZE = 16
WIGNER_COLORBAR_TICK_FONTSIZE = 13
WIGNER_TIGHT_LAYOUT_W_PAD = 0.2
WIGNER_RIGHT_MARGIN = 0.96
WIGNER_ROW_WIDTH_RATIOS = [1.0, 1.0, 1.0, 1.0, 0.045, 0.16, 1.05]
WIGNER_ROW_WSPACE = 0.08
NOON_TITLE_FONTSIZE = WIGNER_TITLE_FONTSIZE
NOON_LABEL_FONTSIZE = WIGNER_LABEL_FONTSIZE
NOON_TICK_FONTSIZE = WIGNER_TICK_FONTSIZE
NOON_COLORBAR_FONTSIZE = WIGNER_COLORBAR_FONTSIZE
NOON_COLORBAR_TICK_FONTSIZE = WIGNER_COLORBAR_TICK_FONTSIZE

def apply_noise_model(initial_rho, time, kappa, noise_kind='loss'):
    time = float(time)
    if noise_kind == 'loss':
        return pure_loss_channel(initial_rho, np.exp(-float(kappa) * time))
    if noise_kind == 'dephasing':
        return dephasing_channel_from_rate(initial_rho, float(kappa), time)
    raise ValueError(f'unknown noise_kind={noise_kind!r}')

def noise_time_axis_label(noise_kind):
    if noise_kind == 'loss':
        return r'Loss time $t$'
    if noise_kind == 'dephasing':
        return r'Dephasing time $t$'
    raise ValueError(f'unknown noise_kind={noise_kind!r}')

def noise_snapshot_summary(noise_kind, snapshot_time, kappa):
    snapshot_time = float(snapshot_time)
    kappa = float(kappa)
    if noise_kind == 'loss':
        eta = np.exp(-kappa * snapshot_time)
        return rf'loss snapshot: $t={snapshot_time:.1f}$, $\eta={eta:.3f}$'
    if noise_kind == 'dephasing':
        gamma_t = kappa * snapshot_time
        return rf'dephasing snapshot: $t={snapshot_time:.1f}$, $\gamma t={gamma_t:.3f}$'
    raise ValueError(f'unknown noise_kind={noise_kind!r}')

# Finite-shot expectation sampling for unitary observables.

def sampled_unitary_expectation(rho, unitary, shots=10000, rng=None, model=None):
    if model is None:
        model = build_unitary_sampling_model(unitary)
    sample = sample_unitary_expectation_from_model(rho, model, shots, rng=rng)
    return {
        'estimate': float(np.real_if_close(sample['estimate'])),
        'exact': float(np.real_if_close(sample['exact_from_probabilities'])),
        'standard_error': float(sample['real_standard_error']) if np.isfinite(sample['real_standard_error']) else np.nan,
        'model': model,
    }

def sampled_wigner_value_from_model(rho, model, shots=10000, rng=None):
    sample = sample_unitary_expectation_from_model(rho, model, shots, rng=rng)
    scale = 2.0 / np.pi
    return {
        'estimate': float(scale * np.real_if_close(sample['estimate'])),
        'exact': float(scale * np.real_if_close(sample['exact_from_probabilities'])),
        'standard_error': float(scale * sample['real_standard_error']) if np.isfinite(sample['real_standard_error']) else np.nan,
    }


# --- Product-copy VD and noisy Fourier-interferometer helpers ---

def tensor_product(operators):
    result = np.array([[1.0 + 0.0j]])
    for operator in operators:
        result = np.kron(result, np.asarray(operator, dtype=complex))
    return result

def tensor_state_list(rho_list):
    rho_list = [np.asarray(rho, dtype=complex) for rho in rho_list]
    if not rho_list:
        raise ValueError('rho_list must contain at least one density matrix')
    return tensor_product(rho_list)

def embed_single_copy_operator(op, copy_index, copies):
    op = np.asarray(op, dtype=complex)
    copy_index = int(copy_index)
    copies = int(copies)
    if not (0 <= copy_index < copies):
        raise ValueError('copy_index must be between 0 and copies - 1')
    factors = []
    for index in range(copies):
        factors.append(op if index == copy_index else np.eye(op.shape[0], dtype=complex))
    return tensor_product(factors)

def symmetrized_copy_operator(op, copies):
    copies = int(copies)
    return sum(embed_single_copy_operator(op, index, copies) for index in range(copies)) / copies

def cyclic_shift_operator(local_dim, copies):
    local_dim = int(local_dim)
    copies = int(copies)
    dim = local_dim ** copies
    shift = np.zeros((dim, dim), dtype=complex)
    shape = (local_dim,) * copies
    for column in range(dim):
        basis = np.unravel_index(column, shape)
        rotated = basis[-1:] + basis[:-1]
        row = np.ravel_multi_index(rotated, shape)
        shift[row, column] = 1.0
    return shift

@lru_cache(maxsize=None)
def occupation_tuples(modes, local_dim, max_total=None):
    modes = int(modes)
    local_dim = int(local_dim)
    occupations = tuple(product(range(local_dim), repeat=modes))
    if max_total is None:
        return occupations
    max_total = int(max_total)
    return tuple(occupation for occupation in occupations if sum(occupation) <= max_total)

def occupation_indices_with_total_at_most(local_dim, copies, max_total):
    all_occupations = occupation_tuples(copies, local_dim, None)
    max_total = int(max_total)
    return np.asarray(
        [index for index, occupation in enumerate(all_occupations) if sum(occupation) <= max_total],
        dtype=int,
    )

def permanent_ryser(matrix):
    matrix = np.asarray(matrix, dtype=complex)
    size = matrix.shape[0]
    if size == 0:
        return 1.0 + 0.0j
    total = 0.0 + 0.0j
    for mask in range(1, 1 << size):
        bits = mask.bit_count()
        row_sums = np.zeros(size, dtype=complex)
        for column in range(size):
            if mask & (1 << column):
                row_sums += matrix[:, column]
        total += ((-1) ** (size - bits)) * np.prod(row_sums)
    return total

def fock_interferometer_unitary(mode_unitary, local_dim, max_total=None):
    mode_unitary = np.asarray(mode_unitary, dtype=complex)
    if mode_unitary.ndim != 2 or mode_unitary.shape[0] != mode_unitary.shape[1]:
        raise ValueError('mode_unitary must be a square matrix')
    modes = mode_unitary.shape[0]
    local_dim = int(local_dim)
    if max_total is not None and local_dim <= int(max_total):
        raise ValueError('local_dim must be at least max_total + 1')
    all_occupations = occupation_tuples(modes, local_dim, None)
    active_occupations = occupation_tuples(modes, local_dim, max_total)
    index_by_occupation = {occupation: index for index, occupation in enumerate(all_occupations)}
    dim = local_dim ** modes
    lifted = np.zeros((dim, dim), dtype=complex)
    factorials = [factorial(number) for number in range(local_dim)]
    for input_occupation in active_occupations:
        column = index_by_occupation[input_occupation]
        input_modes = []
        for mode, count in enumerate(input_occupation):
            input_modes.extend([mode] * count)
        total_photons = sum(input_occupation)
        input_norm = np.sqrt(np.prod([factorials[count] for count in input_occupation]))
        for output_occupation in active_occupations:
            if sum(output_occupation) != total_photons:
                continue
            row = index_by_occupation[output_occupation]
            output_modes = []
            for mode, count in enumerate(output_occupation):
                output_modes.extend([mode] * count)
            output_norm = np.sqrt(np.prod([factorials[count] for count in output_occupation]))
            if total_photons == 0:
                submatrix = np.zeros((0, 0), dtype=complex)
            else:
                submatrix = mode_unitary[np.ix_(output_modes, input_modes)]
            lifted[row, column] = permanent_ryser(submatrix) / (input_norm * output_norm)
    return lifted

def fourier_phase_observable(local_dim, copies, sign=1):
    copies = int(copies)
    local_dim = int(local_dim)
    phases = []
    for occupation in occupation_tuples(copies, local_dim, None):
        weighted_number = sum(mode * number for mode, number in enumerate(occupation))
        phases.append(np.exp(float(sign) * 2j * np.pi * weighted_number / float(copies)))
    return np.diag(np.asarray(phases, dtype=complex))

def fourier_matrix(modes, inverse=False):
    modes = int(modes)
    indices = np.arange(modes, dtype=float)
    sign = -1.0 if inverse else 1.0
    matrix = np.exp(sign * 2j * np.pi * np.outer(indices, indices) / float(modes))
    return matrix / np.sqrt(float(modes))

def reck_decomposition(unitary, atol=1e-12):
    unitary = np.asarray(unitary, dtype=complex)
    if unitary.ndim != 2 or unitary.shape[0] != unitary.shape[1]:
        raise ValueError('unitary must be a square matrix')
    U = unitary.astype(complex).copy()
    modes = U.shape[0]
    transformations = []
    for col in range(modes - 1):
        for row in range(modes - 1, col, -1):
            target = U[row, col]
            pivot = U[row - 1, col]
            if abs(target) < atol:
                continue
            radius = np.sqrt(abs(pivot) ** 2 + abs(target) ** 2)
            cos_theta = abs(pivot) / radius
            sin_theta = abs(target) / radius
            phi = np.angle(target) - np.angle(pivot)
            theta = np.arctan2(sin_theta, cos_theta)
            inverse_block = np.array(
                [
                    [cos_theta, sin_theta * np.exp(-1j * phi)],
                    [-sin_theta * np.exp(1j * phi), cos_theta],
                ],
                dtype=complex,
            )
            U[row - 1:row + 1, :] = inverse_block @ U[row - 1:row + 1, :]
            transformations.append((row - 1, row, float(theta), float(phi)))
    phases = np.angle(np.diag(U))
    return transformations, np.asarray(phases, dtype=float)

def reconstruct_reck_unitary(transformations, phases):
    phases = np.asarray(phases, dtype=float)
    modes = phases.size
    unitary = np.diag(np.exp(1j * phases))
    for mode_a, mode_b, theta, phi in reversed(list(transformations)):
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        block = np.array(
            [
                [cos_theta, -sin_theta * np.exp(-1j * phi)],
                [sin_theta * np.exp(1j * phi), cos_theta],
            ],
            dtype=complex,
        )
        unitary[[mode_a, mode_b], :] = block @ unitary[[mode_a, mode_b], :]
    return unitary

def perturb_reck_components(transformations, phases, component_noise=0.0, rng=None, angle_floor=None):
    rng = _as_rng(rng)
    component_noise = float(component_noise)
    if component_noise < 0.0:
        raise ValueError('component_noise must be non-negative')
    noisy_transformations = []
    diagnostics = []
    for mode_a, mode_b, theta, phi in transformations:
        theta_std = component_noise
        phi_std = 0.0
        theta_error = rng.normal(0.0, theta_std) if component_noise > 0.0 else 0.0
        phi_error = 0.0
        noisy_transformations.append((mode_a, mode_b, float(theta + theta_error), float(phi)))
        diagnostics.append({
            'modes': (mode_a, mode_b),
            'theta_error': float(theta_error),
            'phi_error': float(phi_error),
            'theta_std': float(theta_std),
            'phi_std': float(phi_std),
        })
    noisy_phases = np.asarray(phases, dtype=float).copy()
    phase_errors = []
    for _ in noisy_phases:
        phase_std = 0.0
        phase_error = 0.0
        phase_errors.append({'phase_error': float(phase_error), 'phase_std': float(phase_std)})
    return noisy_transformations, noisy_phases, {
        'component_noise': component_noise,
        'epsilon_std': component_noise,
        'theta_noise_model': 'additive_normal',
        'phase_noise_model': 'none',
        'beam_splitter_errors': diagnostics,
        'phase_errors': phase_errors,
    }

def build_noisy_fourier_interferometer(modes, component_noise=0.0, rng=None, inverse=False):
    ideal = fourier_matrix(modes, inverse=inverse)
    transformations, phases = reck_decomposition(ideal)
    noisy_transformations, noisy_phases, diagnostics = perturb_reck_components(
        transformations,
        phases,
        component_noise=component_noise,
        rng=rng,
    )
    noisy = reconstruct_reck_unitary(noisy_transformations, noisy_phases)
    return {
        'ideal': ideal,
        'noisy': noisy,
        'transformations': transformations,
        'phases': phases,
        'noisy_transformations': noisy_transformations,
        'noisy_phases': noisy_phases,
        'diagnostics': diagnostics,
        'reconstruction_error': float(np.linalg.norm(reconstruct_reck_unitary(transformations, phases) - ideal)),
        'noise_error': float(np.linalg.norm(noisy - ideal)),
    }

def build_noisy_fourier_shift_operator(local_dim, copies, component_noise=0.0, rng=None, max_total=None):
    interferometer = build_noisy_fourier_interferometer(copies, component_noise=component_noise, rng=rng)
    fock_fourier = fock_interferometer_unitary(interferometer['noisy'], local_dim, max_total=max_total)
    phase_observable = fourier_phase_observable(local_dim, copies, sign=1)
    shift = fock_fourier.conj().T @ phase_observable @ fock_fourier
    ideal_shift = cyclic_shift_operator(local_dim, copies)
    if max_total is None:
        support_indices = np.arange(local_dim ** int(copies), dtype=int)
    else:
        support_indices = occupation_indices_with_total_at_most(local_dim, copies, max_total)
    support = np.ix_(support_indices, support_indices)
    return {
        'shift': shift,
        'support_indices': support_indices,
        'interferometer': interferometer,
        'ideal_support_error': float(np.linalg.norm((shift - ideal_shift)[support])),
    }

def coherent_displacements_from_interferometer_error(ideal_unitary, noisy_unitary, amplitude=0.02, input_profile=None, atol=1e-12):
    ideal_unitary = np.asarray(ideal_unitary, dtype=complex)
    noisy_unitary = np.asarray(noisy_unitary, dtype=complex)
    if ideal_unitary.shape != noisy_unitary.shape:
        raise ValueError('ideal_unitary and noisy_unitary must have the same shape')
    if input_profile is None:
        input_profile = np.ones(ideal_unitary.shape[1], dtype=complex)
    input_profile = np.asarray(input_profile, dtype=complex)
    if input_profile.shape != (ideal_unitary.shape[1],):
        raise ValueError('input_profile must contain one coherent amplitude weight per input mode')
    mismatch = (noisy_unitary - ideal_unitary) @ input_profile
    max_abs = float(np.max(np.abs(mismatch))) if mismatch.size else 0.0
    if max_abs <= float(atol):
        return np.zeros(mismatch.shape, dtype=complex)
    return float(amplitude) * mismatch

def apply_displacement_channel(rho, alpha):
    D = displacement_operator(rho.shape[0], alpha)
    displaced = D @ np.asarray(rho, dtype=complex) @ D.conj().T
    return trace_normalize_density_matrix(displaced)

def apply_loss_dephasing_coherent_noise(initial_rho, time, kappa_loss, kappa_phi=None, alpha=0.0):
    time = float(time)
    kappa_loss = float(kappa_loss)
    if kappa_phi is None:
        kappa_phi = kappa_loss
    rho = pure_loss_channel(initial_rho, np.exp(-kappa_loss * time))
    rho = dephasing_channel_from_rate(rho, float(kappa_phi), time)
    if abs(alpha) > 0.0:
        rho = apply_displacement_channel(rho, alpha)
    return rho

def build_perturbed_copy_states(initial_rho, copies, time, kappa_loss, kappa_phi=None, displacements=None):
    copies = int(copies)
    if displacements is None:
        displacements = np.zeros(copies, dtype=complex)
    displacements = np.asarray(displacements, dtype=complex)
    if displacements.size != copies:
        raise ValueError('displacements must contain one value per copy')
    return [
        apply_loss_dephasing_coherent_noise(
            initial_rho,
            time,
            kappa_loss,
            kappa_phi=kappa_phi,
            alpha=displacements[index],
        )
        for index in range(copies)
    ]

def product_virtual_distillation_operators(local_dim, op, copies, copy_index=0, shift_operator=None, inserted_operator=None):
    if shift_operator is None:
        shift = cyclic_shift_operator(local_dim, copies)
    else:
        shift = np.asarray(shift_operator, dtype=complex)
    if inserted_operator is None:
        inserted = embed_single_copy_operator(op, copy_index, copies)
    else:
        inserted = np.asarray(inserted_operator, dtype=complex)
    return {
        'shift': shift,
        'inserted_shift': inserted @ shift,
        'inserted': inserted,
    }

def exact_product_virtual_distillation(rho_list, op, copy_index=0, shift_operator=None, inserted_operator=None):
    rho_list = [np.asarray(rho, dtype=complex) for rho in rho_list]
    copies = len(rho_list)
    local_dim = rho_list[0].shape[0]
    operators = product_virtual_distillation_operators(
        local_dim,
        op,
        copies,
        copy_index=copy_index,
        shift_operator=shift_operator,
        inserted_operator=inserted_operator,
    )
    rho_full = tensor_state_list(rho_list)
    denominator = np.trace(operators['shift'] @ rho_full)
    numerator = np.trace(operators['inserted_shift'] @ rho_full)
    return {
        'ratio': numerator / denominator,
        'numerator': numerator,
        'denominator': denominator,
    }

def build_product_virtual_distillation_sampling_models(
    local_dim,
    op,
    copies,
    copy_index=0,
    shift_operator=None,
    support_indices=None,
    inserted_operator=None,
):
    operators = product_virtual_distillation_operators(
        local_dim,
        op,
        copies,
        copy_index=copy_index,
        shift_operator=shift_operator,
        inserted_operator=inserted_operator,
    )
    denominator_operator = operators['shift']
    numerator_operator = operators['inserted_shift']
    if support_indices is not None:
        support_indices = np.asarray(support_indices, dtype=int)
        support = np.ix_(support_indices, support_indices)
        denominator_operator = denominator_operator[support]
        numerator_operator = numerator_operator[support]
    return {
        'denominator': build_unitary_sampling_model(denominator_operator),
        'numerator': build_unitary_sampling_model(numerator_operator),
        'operators': operators,
        'local_dim': int(local_dim),
        'copies': int(copies),
        'copy_index': int(copy_index),
        'support_indices': None if support_indices is None else support_indices,
    }

def sample_product_virtual_distillation(
    rho_list,
    op,
    shots=10000,
    rng=None,
    models=None,
    copy_index=0,
    shift_operator=None,
    support_indices=None,
    inserted_operator=None,
):
    rho_list = [np.asarray(rho, dtype=complex) for rho in rho_list]
    copies = len(rho_list)
    local_dim = rho_list[0].shape[0]
    if models is None:
        models = build_product_virtual_distillation_sampling_models(
            local_dim,
            op,
            copies,
            copy_index=copy_index,
            shift_operator=shift_operator,
            support_indices=support_indices,
            inserted_operator=inserted_operator,
        )
    rho_full = tensor_state_list(rho_list)
    if models.get('support_indices') is not None:
        support_indices = np.asarray(models['support_indices'], dtype=int)
        rho_full = rho_full[np.ix_(support_indices, support_indices)]
    rng = _as_rng(rng)
    denominator_sample = sample_unitary_expectation_from_model(rho_full, models['denominator'], shots, rng=rng)
    numerator_sample = sample_unitary_expectation_from_model(rho_full, models['numerator'], shots, rng=rng)
    exact = exact_product_virtual_distillation(
        rho_list,
        op,
        copy_index=copy_index,
        shift_operator=shift_operator,
        inserted_operator=inserted_operator,
    )
    ratio_estimate = numerator_sample['estimate'] / denominator_sample['estimate']
    numerator_real = float(np.real(numerator_sample['estimate']))
    denominator_real = float(np.real(denominator_sample['estimate']))
    numerator_error = float(numerator_sample['real_standard_error'])
    denominator_error = float(denominator_sample['real_standard_error'])
    if abs(denominator_real) > 1e-14 and np.isfinite(numerator_error) and np.isfinite(denominator_error):
        ratio_standard_error = np.sqrt(
            (numerator_error / denominator_real) ** 2
            + ((numerator_real * denominator_error) / (denominator_real ** 2)) ** 2
        )
    else:
        ratio_standard_error = np.nan
    return {
        'ratio_estimate': ratio_estimate,
        'ratio_exact_from_samples': numerator_sample['exact_from_probabilities'] / denominator_sample['exact_from_probabilities'],
        'ratio_exact': exact['ratio'],
        'numerator_estimate': numerator_sample['estimate'],
        'denominator_estimate': denominator_sample['estimate'],
        'numerator_exact': exact['numerator'],
        'denominator_exact': exact['denominator'],
        'numerator_standard_error': numerator_sample['real_standard_error'],
        'denominator_standard_error': denominator_sample['real_standard_error'],
        'ratio_standard_error': float(ratio_standard_error),
    }

def build_fock_product_noise_parity_data(
    level=2,
    dim=6,
    copies=3,
    dense_times=None,
    vd_times=None,
    kappa=0.2 / 50.0,
    kappa_phi=None,
    shots=10000,
    component_noise=0.02,
    component_noise_samples=32,
    sample_vd=False,
    seed=1234,
):
    if dense_times is None:
        dense_times = np.linspace(0.0, 50.0, 31)
    if vd_times is None:
        vd_times = np.arange(0.0, 51.0, 5.0)
    dense_times = np.asarray(dense_times, dtype=float)
    vd_times = np.asarray(vd_times, dtype=float)
    component_noise_samples = int(component_noise_samples)
    if component_noise_samples <= 0:
        raise ValueError('component_noise_samples must be positive')
    max_total_photons = int(level) * int(copies)
    simulation_dim = max(int(dim), max_total_photons + 1)
    initial_rho = density_matrix(basis_state(simulation_dim, level))
    op = parity_operator(simulation_dim)
    rng = _as_rng(seed)
    raw_model = build_unitary_sampling_model(op)
    support_indices = occupation_indices_with_total_at_most(simulation_dim, copies, max_total_photons)
    product_models = build_product_virtual_distillation_sampling_models(
        simulation_dim,
        op,
        copies,
        copy_index=0,
        support_indices=support_indices,
    )
    noisy_inserted_operator = symmetrized_copy_operator(op, copies)
    noisy_fourier_shift = build_noisy_fourier_shift_operator(
        simulation_dim,
        copies,
        component_noise=component_noise,
        rng=rng,
        max_total=max_total_photons,
    )
    noisy_fourier_shifts = [noisy_fourier_shift]
    if (not sample_vd) and float(component_noise) > 0.0 and component_noise_samples > 1:
        for _ in range(component_noise_samples - 1):
            noisy_fourier_shifts.append(
                build_noisy_fourier_shift_operator(
                    simulation_dim,
                    copies,
                    component_noise=component_noise,
                    rng=rng,
                    max_total=max_total_photons,
                )
            )
    noisy_product_models = None
    if sample_vd:
        noisy_product_models = build_product_virtual_distillation_sampling_models(
            simulation_dim,
            op,
            copies,
            copy_index=0,
            shift_operator=noisy_fourier_shift['shift'],
            support_indices=support_indices,
            inserted_operator=noisy_inserted_operator,
        )
    # Passive Fourier-component errors change the VD measurement operator.
    # They do not create a displacement channel by themselves.
    displacements = np.zeros(copies, dtype=complex)
    noisy_shift_operator = noisy_fourier_shift['shift']
    pure_value = expectation_value(initial_rho, op)
    raw_loss = []
    raw_loss_exact = []
    for time in dense_times:
        rho_loss = pure_loss_channel(initial_rho, np.exp(-float(kappa) * float(time)))
        sample = sampled_unitary_expectation(rho_loss, op, shots=shots, rng=rng, model=raw_model)
        raw_loss.append(sample['estimate'])
        raw_loss_exact.append(sample['exact'])
    vd_loss = []
    vd_loss_exact = []
    vd_all_noise = []
    vd_all_noise_exact = []
    vd_all_noise_imag = []
    vd_all_noise_error = []
    vd_all_noise_realizations = []
    for time in vd_times:
        rho_loss = pure_loss_channel(initial_rho, np.exp(-float(kappa) * float(time)))
        loss_sample = sample_product_virtual_distillation(
            [rho_loss] * copies,
            op,
            shots=shots,
            rng=rng,
            models=product_models,
        )
        vd_loss_value = loss_sample['ratio_estimate'] if sample_vd else loss_sample['ratio_exact']
        vd_loss.append(float(np.real(vd_loss_value)))
        vd_loss_exact.append(float(np.real(loss_sample['ratio_exact'])))
        rho_list = build_perturbed_copy_states(
            initial_rho,
            copies,
            time,
            kappa,
            kappa_phi=kappa_phi,
            displacements=displacements,
        )
        if sample_vd:
            all_noise_sample = sample_product_virtual_distillation(
                rho_list,
                op,
                shots=shots,
                rng=rng,
                models=noisy_product_models,
                shift_operator=noisy_shift_operator,
                inserted_operator=noisy_inserted_operator,
            )
            all_noise_value = all_noise_sample['ratio_estimate']
            all_noise_exact = all_noise_sample['ratio_exact']
            all_noise_error = all_noise_sample['ratio_standard_error']
            realization_values = np.asarray([all_noise_exact], dtype=complex)
        else:
            realization_values = np.asarray(
                [
                    exact_product_virtual_distillation(
                        rho_list,
                        op,
                        shift_operator=shift_data['shift'],
                        inserted_operator=noisy_inserted_operator,
                    )['ratio']
                    for shift_data in noisy_fourier_shifts
                ],
                dtype=complex,
            )
            all_noise_value = np.mean(realization_values)
            all_noise_exact = all_noise_value
            if realization_values.size > 1:
                all_noise_error = float(np.std(np.real(realization_values), ddof=1))
            else:
                all_noise_error = 0.0
        vd_all_noise.append(float(np.real(all_noise_value)))
        vd_all_noise_exact.append(float(np.real(all_noise_exact)))
        vd_all_noise_imag.append(float(np.imag(all_noise_exact)))
        vd_all_noise_error.append(float(all_noise_error))
        vd_all_noise_realizations.append(np.real(realization_values))
    return {
        'dense_times': dense_times,
        'vd_times': vd_times,
        'pure': float(pure_value),
        'raw_loss': np.asarray(raw_loss, dtype=float),
        'raw_loss_exact': np.asarray(raw_loss_exact, dtype=float),
        'vd_loss': np.asarray(vd_loss, dtype=float),
        'vd_loss_exact': np.asarray(vd_loss_exact, dtype=float),
        'vd_all_noise': np.asarray(vd_all_noise, dtype=float),
        'vd_all_noise_exact': np.asarray(vd_all_noise_exact, dtype=float),
        'vd_all_noise_imag': np.asarray(vd_all_noise_imag, dtype=float),
        'vd_all_noise_error': np.asarray(vd_all_noise_error, dtype=float),
        'vd_all_noise_realizations': np.asarray(vd_all_noise_realizations, dtype=float),
        'displacements': displacements,
        'interferometer': noisy_fourier_shift['interferometer'],
        'noisy_fourier_shift': noisy_fourier_shift,
        'noisy_fourier_shift_errors': np.asarray(
            [shift_data['ideal_support_error'] for shift_data in noisy_fourier_shifts],
            dtype=float,
        ),
        'max_total_photons': int(max_total_photons),
        'shots': int(shots),
        'kappa': float(kappa),
        'kappa_phi': float(kappa if kappa_phi is None else kappa_phi),
        'component_noise': float(component_noise),
        'component_noise_samples': int(len(noisy_fourier_shifts)),
        'sample_vd': bool(sample_vd),
        'level': int(level),
        'dim': int(simulation_dim),
        'requested_dim': int(dim),
        'copies': int(copies),
    }

def plot_fock_product_noise_parity_panel(ax, data, show_legend=True):
    x_dense = data['kappa'] * data['dense_times']
    x_vd = data['kappa'] * data['vd_times']
    ax.axhline(data['pure'], color='gray', linestyle='--', linewidth=1.2, label='Pure state')
    ax.plot(
        x_dense,
        data['raw_loss'],
        linestyle='None',
        marker='+',
        markersize=7.0,
        markeredgewidth=1.4,
        color='navy',
        label='no distillation',
    )
    ax.plot(
        x_vd,
        data['vd_loss'],
        linestyle='None',
        marker='+',
        markersize=7.0,
        markeredgewidth=1.4,
        color='#d6271d',
        label=rf'{int(data["copies"])}-mode VD',
    )
    noisy_label = rf'{int(data["copies"])}-mode VD with $N_c + N_l + N_d$'
    noisy_error = np.asarray(data.get('vd_all_noise_error', np.zeros_like(data['vd_all_noise'])), dtype=float)
    show_noisy_error = np.any(np.isfinite(noisy_error) & (noisy_error > 0.0))
    if show_noisy_error:
        ax.errorbar(
            x_vd,
            data['vd_all_noise'],
            yerr=noisy_error,
            linestyle='None',
            marker='o',
            markersize=5.0,
            markeredgewidth=1.0,
            markerfacecolor='#009ca6',
            markeredgecolor='#009ca6',
            color='#009ca6',
            ecolor='#009ca6',
            elinewidth=0.9,
            capsize=3.0,
            label=noisy_label,
        )
    else:
        ax.plot(
            x_vd,
            data['vd_all_noise'],
            linestyle='None',
            marker='o',
            markersize=5.0,
            markeredgewidth=1.0,
            markerfacecolor='#009ca6',
            markeredgecolor='#009ca6',
            color='#009ca6',
            label=noisy_label,
        )
    ax.set_xlabel(r'$\kappa t$', fontsize=9)
    ax.set_ylabel(r'Parity expectation $\langle (-1)^{\hat n} \rangle$', fontsize=9)
    ax.set_xlim(float(x_dense[0]), float(x_dense[-1]))
    noisy_values_for_limits = np.asarray(data['vd_all_noise'], dtype=float)
    if show_noisy_error:
        noisy_values_for_limits = np.concatenate(
            [noisy_values_for_limits - noisy_error, noisy_values_for_limits + noisy_error]
        )
    ax.set_ylim(*sampled_curve_zoom_limits({
        'pure': data['pure'],
        'raw': data['raw_loss'],
        'vd': {
            'loss': data['vd_loss'],
            'all_noise': noisy_values_for_limits,
        },
    }))
    ax.grid(False)
    ax.tick_params(labelsize=8)
    if show_legend:
        ax.legend(fontsize=11, loc='lower left', frameon=True)

def create_fock_product_noise_parity_figure(
    level=2,
    dim=6,
    copies=3,
    dense_times=None,
    vd_times=None,
    kappa=0.2 / 50.0,
    kappa_phi=None,
    shots=10000,
    component_noise=0.02,
    component_noise_samples=32,
    sample_vd=False,
    seed=1234,
    figsize=(7.2, 4.2),
    save_path=None,
    dpi=300,
):
    data = build_fock_product_noise_parity_data(
        level=level,
        dim=dim,
        copies=copies,
        dense_times=dense_times,
        vd_times=vd_times,
        kappa=kappa,
        kappa_phi=kappa_phi,
        shots=shots,
        component_noise=component_noise,
        component_noise_samples=component_noise_samples,
        sample_vd=sample_vd,
        seed=seed,
    )
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plot_fock_product_noise_parity_panel(ax, data, show_legend=True)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 1.0))
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    return fig, ax, data

# Adaptive state builders used by the example figures.

def build_paper_even_cat_state(alpha, scan_radius=0.0, max_dim=220, tail_tol=1e-8):
    initial_dim = recommended_cat_cutoff(abs(alpha) + float(scan_radius)) + 1
    return build_adaptive_state(even_cat_state, alpha, initial_dim, step=10, max_dim=max_dim, tail_tol=tail_tol)

def build_paper_square_cat_state(alpha, scan_radius=0.0, max_dim=240, tail_tol=1e-8):
    initial_dim = recommended_square_cat_cutoff(abs(alpha) + float(scan_radius)) + 1
    return build_adaptive_state(square_cat_state, alpha, initial_dim, step=10, max_dim=max_dim, tail_tol=tail_tol)

def build_paper_tri_cat_state(alpha, scan_radius=0.0, max_dim=240, tail_tol=1e-8):
    initial_dim = recommended_cat_cutoff(abs(alpha) + float(scan_radius)) + 1
    return build_adaptive_state(tri_cat_state, alpha, initial_dim, step=10, max_dim=max_dim, tail_tol=tail_tol)

# Sampled parity time-series figures.

def build_sampled_parity_time_data(initial_rho, copies_list, dense_times, vd_times, kappa, noise_kind='loss', shots=10000, seed=1234):
    dense_times = np.asarray(dense_times, dtype=float)
    vd_times = np.asarray(vd_times, dtype=float)
    dim = initial_rho.shape[0]
    op = parity_operator(dim)
    model = build_unitary_sampling_model(op)
    rng = _as_rng(seed)
    pure_value = expectation_value(initial_rho, op)
    raw = []
    vd = {copies: [] for copies in copies_list}
    for time in dense_times:
        noisy_rho = apply_noise_model(initial_rho, time, kappa, noise_kind=noise_kind)
        raw_sample = sampled_unitary_expectation(noisy_rho, op, shots=shots, rng=rng, model=model)
        raw.append(raw_sample['estimate'])
    for time in vd_times:
        noisy_rho = apply_noise_model(initial_rho, time, kappa, noise_kind=noise_kind)
        spectral = spectral_decomposition_of_density_matrix(noisy_rho)
        for copies in copies_list:
            vd_rho = exact_distilled_density_matrix_from_spectral(spectral, copies)
            vd_sample = sampled_unitary_expectation(vd_rho, op, shots=shots, rng=rng, model=model)
            vd[copies].append(vd_sample['estimate'])
    return {
        'dense_times': dense_times,
        'vd_times': vd_times,
        'pure': float(pure_value),
        'raw': np.asarray(raw, dtype=float),
        'vd': {copies: np.asarray(values, dtype=float) for copies, values in vd.items()},
        'shots': int(shots),
        'noise_kind': noise_kind,
        'kappa': float(kappa),
    }

def sampled_curve_zoom_limits(data, pad_fraction=0.10, min_span=0.03):
    values = [float(data['pure'])]
    values.extend(data['raw'].tolist())
    for copies in data['vd']:
        values.extend(data['vd'][copies].tolist())
    ymin = min(values)
    ymax = max(values)
    span = max(float(ymax - ymin), float(min_span))
    pad = max(float(min_span) * 0.5, pad_fraction * span)
    return max(-1.05, ymin - pad), min(1.05, ymax + pad)

def plot_sampled_parity_time_panel(ax, data, title, zoom=False, show_legend=True):
    palette = ['orange', 'teal', 'forestgreen']
    markers = ['+', 'x', 'o']
    x_dense = data['kappa'] * data['dense_times']
    x_vd = data['kappa'] * data['vd_times']
    ax.plot(x_dense, data['raw'], color='red', linewidth=1.4, label='No VD')
    ax.axhline(data['pure'], color='pink', linestyle='--', linewidth=1.1, label='Pure state')
    for index, copies in enumerate(sorted(data['vd'])):
        color = palette[index % len(palette)]
        marker = markers[index % len(markers)]
        markerfacecolor = 'none' if marker == 'o' else color
        ax.plot(
            x_vd,
            data['vd'][copies],
            linestyle='None',
            marker=marker,
            markersize=5.5,
            markeredgewidth=1.1,
            markerfacecolor=markerfacecolor,
            color=color,
            label=paper_vd_label(copies),
        )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(r'$\kappa t$', fontsize=9)
    ax.set_ylabel(r'Parity expectation $\langle (-1)^{\hat n} \rangle$', fontsize=9)
    ax.set_xlim(float(x_dense[0]), float(x_dense[-1]))
    if zoom:
        ax.set_ylim(*sampled_curve_zoom_limits(data))
    else:
        ax.set_ylim(-1.05, 1.05)
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=8)
    if show_legend:
        ax.legend(fontsize=7, loc='lower left', frameon=False)

def create_sampled_parity_paper_figure(
    initial_rho,
    title,
    copies_list,
    dense_times,
    vd_times,
    kappa,
    noise_kind='loss',
    shots=10000,
    seed=1234,
    save_path=None,
    dpi=300,
):
    data = build_sampled_parity_time_data(
        initial_rho,
        copies_list,
        dense_times,
        vd_times,
        kappa,
        noise_kind=noise_kind,
        shots=shots,
        seed=seed,
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.2), squeeze=False)
    plot_sampled_parity_time_panel(axes[0, 0], data, title, zoom=False, show_legend=True)
    plot_sampled_parity_time_panel(axes[0, 1], data, rf'{title} (zoom)', zoom=True, show_legend=False)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 1.0))
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    return fig, axes, data

def create_zoomed_sampled_parity_state_list_figure(
    state_entries,
    copies_list,
    dense_times,
    vd_times,
    kappa,
    noise_kind='loss',
    shots=10000,
    seed=1234,
    y_limits=None,
    figsize=None,
    save_path=None,
    dpi=300,
):
    state_entries = list(state_entries)
    if not state_entries:
        raise ValueError('state_entries must not be empty')
    if figsize is None:
        figsize = (4.9 * len(state_entries), 4.1)
    fig, axes = plt.subplots(1, len(state_entries), figsize=figsize, squeeze=False)
    outputs = []
    for index, (title, rho) in enumerate(state_entries):
        state_seed = None if seed is None else int(seed) + 1009 * index
        data = build_sampled_parity_time_data(
            rho,
            copies_list,
            dense_times,
            vd_times,
            kappa,
            noise_kind=noise_kind,
            shots=shots,
            seed=state_seed,
        )
        plot_sampled_parity_time_panel(axes[0, index], data, title, zoom=True, show_legend=(index == 0))
        if y_limits is not None:
            axes[0, index].set_ylim(float(y_limits[0]), float(y_limits[1]))
        outputs.append({'title': title, 'data': data})
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 1.0))
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    return fig, axes, outputs

# Sampled Wigner maps and phase-space error figures.

def build_displaced_parity_sampling_models(dim, x_values, y_values):
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    models = []
    for y in y_values:
        for x in x_values:
            alpha = x + 1j * y
            models.append(build_unitary_sampling_model(displaced_parity_operator(dim, alpha)))
    return {
        'x_values': x_values,
        'y_values': y_values,
        'models': models,
        'shape': (len(y_values), len(x_values)),
    }

def sampled_wigner_map_from_models(rho, model_grid, shots=10000, rng=None):
    rng = _as_rng(rng)
    values = []
    errors = []
    for model in model_grid['models']:
        sample = sampled_wigner_value_from_model(rho, model, shots=shots, rng=rng)
        values.append(sample['estimate'])
        errors.append(sample['standard_error'])
    return {
        'map': np.asarray(values, dtype=float).reshape(model_grid['shape']),
        'error': np.asarray(errors, dtype=float).reshape(model_grid['shape']),
    }

def build_noise_scenario_states(initial_rho, copies_list, snapshot_time, kappa, noise_kind='loss'):
    noisy_rho = apply_noise_model(initial_rho, snapshot_time, kappa, noise_kind=noise_kind)
    spectral = spectral_decomposition_of_density_matrix(noisy_rho)
    scenario_states = [('Pure state', initial_rho), ('No VD', noisy_rho)]
    for copies in copies_list:
        scenario_states.append((paper_vd_label(copies), exact_distilled_density_matrix_from_spectral(spectral, copies)))
    return scenario_states

def build_sampled_wigner_grid_l2_error_data(initial_rho, copies_list, times, kappa, x_values, y_values, noise_kind='loss'):
    times = np.asarray(times, dtype=float)
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    kappa = float(kappa)
    dx = float(np.mean(np.diff(x_values))) if len(x_values) > 1 else 1.0
    dy = float(np.mean(np.diff(y_values))) if len(y_values) > 1 else 1.0
    cell_area = abs(dx * dy)
    kernel_grid = build_displaced_parity_kernels(initial_rho.shape[0], x_values, y_values)
    pure_map = wigner_map_from_kernels(initial_rho, kernel_grid)
    raw_error = []
    vd_error = {copies: [] for copies in copies_list}
    for time in times:
        noisy_rho = apply_noise_model(initial_rho, time, kappa, noise_kind=noise_kind)
        raw_map = wigner_map_from_kernels(noisy_rho, kernel_grid)
        raw_error.append(float(np.sqrt(np.sum((raw_map - pure_map) ** 2) * cell_area)))
        spectral = spectral_decomposition_of_density_matrix(noisy_rho)
        for copies in copies_list:
            vd_rho = exact_distilled_density_matrix_from_spectral(spectral, copies)
            vd_map = wigner_map_from_kernels(vd_rho, kernel_grid)
            vd_error[copies].append(float(np.sqrt(np.sum((vd_map - pure_map) ** 2) * cell_area)))
    return {
        'times': times,
        'kappa': kappa,
        'raw_error': np.asarray(raw_error, dtype=float),
        'vd_error': {copies: np.asarray(values, dtype=float) for copies, values in vd_error.items()},
        'noise_kind': noise_kind,
        'grid_shape': kernel_grid['shape'],
        'point_count': int(pure_map.size),
        'dx': dx,
        'dy': dy,
        'cell_area': cell_area,
    }

def plot_sampled_wigner_l2_panel(ax, data, title=None):
    palette = ['orange', 'teal', 'forestgreen']
    x_values = float(data.get('kappa', 1.0)) * data['times']
    ax.plot(x_values, data['raw_error'], color='red', linewidth=1.2, label='No VD')
    for index, copies in enumerate(sorted(data['vd_error'])):
        color = palette[index % len(palette)]
        ax.plot(x_values, data['vd_error'][copies], color=color, linewidth=1.15, label=paper_vd_label(copies))
    ax.set_xlabel(r'$\kappa t$', fontsize=WIGNER_LABEL_FONTSIZE)
    ax.set_ylabel(paper_wigner_l2_label(), fontsize=WIGNER_LABEL_FONTSIZE)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position('right')
    ax.tick_params(axis='y', labelleft=False, labelright=True)
    ax.yaxis.set_label_coords(1.18, 0.5)
    ax.set_xlim(float(x_values[0]), float(x_values[-1]))
    max_error = max([float(np.max(data['raw_error']))] + [float(np.max(values)) for values in data['vd_error'].values()])
    ax.set_ylim(0.0, 1.05 * max(1e-6, max_error))
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=WIGNER_TICK_FONTSIZE)
    ax.legend(fontsize=WIGNER_LEGEND_FONTSIZE, loc='best', frameon=False)

def add_colorbar_between_axes(fig, image, left_ax, right_ax, label, width=0.012, left_gap_fraction=0.0):
    left_box = left_ax.get_position()
    right_box = right_ax.get_position()
    gap = max(0.0, right_box.x0 - left_box.x1)
    colorbar_width = min(float(width), 0.6 * gap) if gap > 0.0 else float(width)
    remaining_gap = max(0.0, gap - colorbar_width)
    colorbar_x = left_box.x1 + float(left_gap_fraction) * remaining_gap
    colorbar_ax = fig.add_axes([colorbar_x, left_box.y0, colorbar_width, left_box.height])
    colorbar = fig.colorbar(image, cax=colorbar_ax)
    colorbar.ax.set_title(label, fontsize=WIGNER_COLORBAR_FONTSIZE, pad=5)
    colorbar.ax.tick_params(labelsize=WIGNER_COLORBAR_TICK_FONTSIZE)
    return colorbar

def create_sampled_wigner_paper_figure(
    initial_rho,
    title,
    copies_list,
    snapshot_time,
    times,
    kappa,
    noise_kind='loss',
    x_values=np.linspace(-2.0, 2.0, 31),
    y_values=np.linspace(-2.0, 2.0, 31),
    shots=10000,
    seed=1234,
    save_path=None,
    dpi=300,
):
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    model_grid = build_displaced_parity_sampling_models(initial_rho.shape[0], x_values, y_values)
    scenario_states = build_noise_scenario_states(initial_rho, copies_list, snapshot_time, kappa, noise_kind=noise_kind)
    scenario_labels = [label for label, _ in scenario_states]
    sampled_maps = {}
    for index, (label, rho) in enumerate(scenario_states):
        panel_seed = None if seed is None else int(seed) + 7919 * index
        sampled_maps[label] = sampled_wigner_map_from_models(rho, model_grid, shots=shots, rng=panel_seed)['map']
    error_data = build_sampled_wigner_grid_l2_error_data(
        initial_rho,
        copies_list,
        times,
        kappa,
        x_values,
        y_values,
        noise_kind=noise_kind,
    )

    fig, axes = plt.subplots(2, 3, figsize=(13.4, 7.7), squeeze=False)
    wigner_axes = [
        axes[0, 0],
        axes[0, 1],
        axes[0, 2],
        axes[1, 0],
        axes[1, 1],
    ]
    l2_ax = axes[1, 2]
    wigner_abs_max = 2.0 / np.pi
    image = None
    for panel_index, label in enumerate(scenario_labels):
        ax = wigner_axes[panel_index]
        image = ax.imshow(
            sampled_maps[label],
            origin='lower',
            extent=(x_values[0], x_values[-1], y_values[0], y_values[-1]),
            cmap='RdBu_r',
            vmin=-wigner_abs_max,
            vmax=wigner_abs_max,
            aspect='equal',
            interpolation='bilinear',
        )
        ax.set_title(label, fontsize=WIGNER_TITLE_FONTSIZE)
        if panel_index >= 3:
            ax.set_xlabel(paper_re_alpha_label(), fontsize=WIGNER_LABEL_FONTSIZE)
        if panel_index in (0, 3):
            ax.set_ylabel(paper_im_alpha_label(), fontsize=WIGNER_LABEL_FONTSIZE)
        else:
            ax.tick_params(axis='y', left=False, labelleft=False)
        ax.tick_params(labelsize=WIGNER_TICK_FONTSIZE)
    plot_sampled_wigner_l2_panel(
        l2_ax,
        error_data,
    )
    note = noise_snapshot_summary(noise_kind, snapshot_time, kappa)
    fig.subplots_adjust(left=0.07, right=0.84, bottom=0.10, top=0.91, wspace=0.16, hspace=0.23)
    top_right_box = wigner_axes[2].get_position()
    bottom_right_box = l2_ax.get_position()
    colorbar_x = 0.900
    colorbar_width = 0.014
    colorbar_ax = fig.add_axes([
        colorbar_x,
        top_right_box.y0,
        colorbar_width,
        top_right_box.height,
    ])
    l2_label_x = (colorbar_x + 0.5 * colorbar_width - bottom_right_box.x0) / bottom_right_box.width
    l2_ax.yaxis.set_label_coords(l2_label_x, 0.5)
    l2_ax.yaxis.label.set_size(WIGNER_LABEL_FONTSIZE + 3)
    colorbar = fig.colorbar(image, cax=colorbar_ax)
    colorbar.ax.set_title(paper_wigner_label(), fontsize=WIGNER_COLORBAR_FONTSIZE, pad=5)
    colorbar.ax.tick_params(labelsize=WIGNER_COLORBAR_TICK_FONTSIZE)
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    return fig, axes, {'scenario_labels': scenario_labels, 'noise_note': note, 'point_count': error_data['point_count'], 'grid_shape': error_data['grid_shape'], 'dx': error_data['dx'], 'dy': error_data['dy'], 'cell_area': error_data['cell_area']}

def create_sampled_wigner_paper_figure_row_no_pure(
    initial_rho,
    title,
    copies_list,
    snapshot_time,
    times,
    kappa,
    noise_kind='loss',
    x_values=np.linspace(-2.0, 2.0, 31),
    y_values=np.linspace(-2.0, 2.0, 31),
    shots=10000,
    seed=1234,
    save_path=None,
    dpi=300,
):
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    model_grid = build_displaced_parity_sampling_models(initial_rho.shape[0], x_values, y_values)
    scenario_states = build_noise_scenario_states(initial_rho, copies_list, snapshot_time, kappa, noise_kind=noise_kind)
    display_states = scenario_states[1:]
    display_labels = [label for label, _ in display_states]
    sampled_maps = {}
    for index, (label, rho) in enumerate(display_states, start=1):
        panel_seed = None if seed is None else int(seed) + 7919 * index
        sampled_maps[label] = sampled_wigner_map_from_models(rho, model_grid, shots=shots, rng=panel_seed)['map']
    error_data = build_sampled_wigner_grid_l2_error_data(
        initial_rho,
        copies_list,
        times,
        kappa,
        x_values,
        y_values,
        noise_kind=noise_kind,
    )

    fig = plt.figure(figsize=(18.0, 3.9))
    grid = fig.add_gridspec(
        1,
        7,
        width_ratios=WIGNER_ROW_WIDTH_RATIOS,
        wspace=WIGNER_ROW_WSPACE,
    )
    axes_row = np.asarray(
        [
            fig.add_subplot(grid[0, 0]),
            fig.add_subplot(grid[0, 1]),
            fig.add_subplot(grid[0, 2]),
            fig.add_subplot(grid[0, 3]),
            fig.add_subplot(grid[0, 6]),
        ],
        dtype=object,
    )
    axes = axes_row.reshape(1, -1)
    colorbar_ax = fig.add_subplot(grid[0, 4])
    wigner_abs_max = 2.0 / np.pi
    image = None
    for panel_index, label in enumerate(display_labels):
        ax = axes_row[panel_index]
        image = ax.imshow(
            sampled_maps[label],
            origin='lower',
            extent=(x_values[0], x_values[-1], y_values[0], y_values[-1]),
            cmap='RdBu_r',
            vmin=-wigner_abs_max,
            vmax=wigner_abs_max,
            aspect='equal',
            interpolation='bilinear',
        )
        ax.set_title(label, fontsize=WIGNER_TITLE_FONTSIZE)
        ax.set_xlabel(paper_re_alpha_label(), fontsize=WIGNER_LABEL_FONTSIZE)
        if panel_index == 0:
            ax.set_ylabel(paper_im_alpha_label(), fontsize=WIGNER_LABEL_FONTSIZE)
        else:
            ax.tick_params(axis='y', labelleft=False)
        ax.tick_params(labelsize=WIGNER_TICK_FONTSIZE)
    plot_sampled_wigner_l2_panel(
        axes_row[4],
        error_data,
    )
    note = noise_snapshot_summary(noise_kind, snapshot_time, kappa)
    colorbar = fig.colorbar(image, cax=colorbar_ax)
    colorbar.ax.set_title(paper_wigner_label(), fontsize=WIGNER_COLORBAR_FONTSIZE, pad=5)
    colorbar.ax.tick_params(labelsize=WIGNER_COLORBAR_TICK_FONTSIZE)
    fig.subplots_adjust(left=0.045, right=0.985, bottom=0.20, top=0.84)
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    return fig, axes, {'scenario_labels': display_labels, 'noise_note': note, 'point_count': error_data['point_count'], 'grid_shape': error_data['grid_shape'], 'dx': error_data['dx'], 'dy': error_data['dy'], 'cell_area': error_data['cell_area']}




# --- Default simulation and plotting parameters ---
COMMON_PLOT_CONFIG = {
    'copies_list': (2, 3, 4),
    'kappa': 0.2 / 50.0,
    'dense_times_medium': np.linspace(0.0, 50.0, 121),
    'vd_times': np.arange(0.0, 51.0, 5.0),
}

PHASE_RECONSTRUCTION_CONFIG = {
    'copies_list': (2, 3, 4),
    'snapshot_time': 50.0,
}

NOON_CHARACTERISTIC_GRID_CONFIG = {
    'n': 4,
    'dim': 8,
    'phase_points': 121,
    'figsize': (13.5, 3.6),
}

PAPER_PLOT_CONFIG = {
    'alpha': 1.50,
    'copies_list': (2, 3, 4),
    'shots': 50000,
    'kappa': COMMON_PLOT_CONFIG['kappa'],
    'times': COMMON_PLOT_CONFIG['dense_times_medium'],
    'vd_times': COMMON_PLOT_CONFIG['vd_times'],
    'snapshot_time': 50.0,
    'wigner_extent': 2.0,
    'fock_wigner_extent': 2.0,
    'wigner_grid_points': 31,
    'fock_wigner_dim': 40,
    'seed_base': 20260504,
}


def paper_wigner_grid(extent=None, points=None):
    extent = PAPER_PLOT_CONFIG['wigner_extent'] if extent is None else float(extent)
    points = PAPER_PLOT_CONFIG['wigner_grid_points'] if points is None else int(points)
    values = np.linspace(-extent, extent, points)
    return values, values.copy()


def build_default_paper_states():
    x_values, y_values = paper_wigner_grid(PAPER_PLOT_CONFIG['wigner_extent'])
    scan_radius = phase_space_radius(x_values, y_values)
    even_state, even_dim, even_tail = build_paper_even_cat_state(PAPER_PLOT_CONFIG['alpha'], scan_radius=scan_radius)
    square_state, square_dim, square_tail = build_paper_square_cat_state(PAPER_PLOT_CONFIG['alpha'], scan_radius=scan_radius)
    tri_state, tri_dim, tri_tail = build_paper_tri_cat_state(PAPER_PLOT_CONFIG['alpha'])
    fock3_rho = density_matrix(basis_state(PAPER_PLOT_CONFIG['fock_wigner_dim'], 3))
    return {
        'even_cat': {
            'rho': density_matrix(even_state),
            'title': paper_alpha_title('Even cat', PAPER_PLOT_CONFIG['alpha']),
            'dim': even_dim,
            'tail': even_tail,
        },
        'square_cat': {
            'rho': density_matrix(square_state),
            'title': paper_alpha_title('Square cat', PAPER_PLOT_CONFIG['alpha']),
            'dim': square_dim,
            'tail': square_tail,
        },
        'tri_cat': {
            'rho': density_matrix(tri_state),
            'title': paper_alpha_title('Tri cat', PAPER_PLOT_CONFIG['alpha']),
            'dim': tri_dim,
            'tail': tri_tail,
        },
        'fock3': {
            'rho': fock3_rho,
            'title': paper_fock_title(3),
            'dim': PAPER_PLOT_CONFIG['fock_wigner_dim'],
            'tail': 0.0,
        },
    }
