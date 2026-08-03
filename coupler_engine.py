import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# --- REFRACTIVE INDEX FUNCTIONS ---

def sellmeier_sio2(lam_um):
    """Cladding: Thermal SiO2 Sellmeier"""
    b1, c1 = 0.6961663, 0.0684043
    b2, c2 = 0.4079426, 0.1162414
    b3, c3 = 0.8974794, 9.896161
    n_sq = 1.0 + (b1 * lam_um**2) / (lam_um**2 - c1**2) + \
                 (b2 * lam_um**2) / (lam_um**2 - c2**2) + \
                 (b3 * lam_um**2) / (lam_um**2 - c3**2)
    return np.sqrt(n_sq)

def n_sin_stoch(lam_um):
    """Stoichiometric Si3N4 (Cauchy)"""
    return 1.981800 + (1.407700e-02 / (lam_um**2))

def n_sin_lowstress(lam_um):
    """Low-Stress SiN (Cauchy)"""
    return 2.087000 + (3.109100e-02 / (lam_um**2))

def n_al2o3(lam_um):
    """Alumina (Al2O3) Core - ALUVIA PDK Sellmeier"""
    eps_inf = 1.0
    A = 1.912
    E = 0.09566
    P = 0.00306
    n_sq = eps_inf + (A * lam_um**2) / (lam_um**2 - E**2) - P * lam_um**2
    return np.sqrt(np.maximum(n_sq, 1.0))

def n_silicon(lam_um):
    """Crystalline Silicon (Malitson Sellmeier)"""
    n_sq = 1.0 + (10.6684293 * lam_um**2) / (lam_um**2 - 0.301516485**2) + \
                 (0.0030434748 * lam_um**2) / (lam_um**2 - 1.13475115**2) + \
                 (1.54133408 * lam_um**2) / (lam_um**2 - 1104.0**2)
    return np.sqrt(np.maximum(n_sq, 1.0))

def get_core_index(lam_um, material_name):
    if material_name == "Si3N4 (Stoichiometric)":
        return n_sin_stoch(lam_um)
    elif material_name == "SiN (Low Stress)":
        return n_sin_lowstress(lam_um)
    elif material_name == "Al2O3 (Alumina)":
        return n_al2o3(lam_um)
    elif material_name == "Si (Silicon)":
        return n_silicon(lam_um)
    else:
        return n_sin_stoch(lam_um)

# --- MESH GENERATION FOR ASYMMETRIC COUPLER ---

def waveguidemesh_asymmetric(w1, w2, h_core, gap, side, bottom_ox, top_ox, dx, dy, n_core, n_clad):
    total_width = w1 + w2 + gap + 2 * side
    total_height = bottom_ox + h_core + top_ox
    
    nx = int(np.round(total_width / dx)) + 1
    ny = int(np.round(total_height / dy)) + 1
    
    x = np.linspace(-total_width / 2.0, total_width / 2.0, nx)
    y = np.linspace(0, total_height, ny)
    
    xc = 0.5 * (x[:-1] + x[1:])
    yc = 0.5 * (y[:-1] + y[1:])
    
    eps = np.full((len(xc), len(yc)), n_clad**2)
    
    # Coordinates of Waveguide 1 (Left) and Waveguide 2 (Right)
    x1_l = -gap / 2.0 - w1
    x1_r = -gap / 2.0
    x2_l = gap / 2.0
    x2_r = gap / 2.0 + w2
    
    y_b = bottom_ox
    y_t = bottom_ox + h_core
    
    wg1_mask = (xc[:, None] >= x1_l) & (xc[:, None] <= x1_r) & (yc[None, :] >= y_b) & (yc[None, :] <= y_t)
    wg2_mask = (xc[:, None] >= x2_l) & (xc[:, None] <= x2_r) & (yc[None, :] >= y_b) & (yc[None, :] <= y_t)
    
    eps[wg1_mask] = n_core**2
    eps[wg2_mask] = n_core**2
    
    return xc, yc, eps, x1_l, x1_r, x2_l, x2_r, y_b, y_t

def single_waveguide_mesh(w_core, h_core, side, bottom_ox, top_ox, dx, dy, n_core, n_clad):
    total_width = w_core + 2 * side
    total_height = bottom_ox + h_core + top_ox
    
    nx = int(np.round(total_width / dx)) + 1
    ny = int(np.round(total_height / dy)) + 1
    
    x = np.linspace(-total_width / 2.0, total_width / 2.0, nx)
    y = np.linspace(0, total_height, ny)
    
    xc = 0.5 * (x[:-1] + x[1:])
    yc = 0.5 * (y[:-1] + y[1:])
    
    eps = np.full((len(xc), len(yc)), n_clad**2)
    mask = (xc[:, None] >= -w_core/2.0) & (xc[:, None] <= w_core/2.0) & (yc[None, :] >= bottom_ox) & (yc[None, :] <= bottom_ox + h_core)
    eps[mask] = n_core**2
    
    return xc, yc, eps

# --- 2D SVFD SOLVER ---

def svmodes_2d(lam_um, guess, nmodes, dx, dy, eps_mesh, polarization='ex'):
    nx, ny = eps_mesh.shape
    k0 = 2.0 * np.pi / lam_um
    eps_padded = np.pad(eps_mesh, ((1, 1), (1, 1)), mode='edge')
    
    ep = eps_padded[1:nx+1, 1:ny+1]
    en = eps_padded[1:nx+1, 2:ny+2]
    es = eps_padded[1:nx+1, 0:ny]
    ee = eps_padded[2:nx+2, 1:ny+1]
    ew = eps_padded[0:nx,   1:ny+1]
    
    n_mat = np.full((nx, ny), dy)
    s_mat = np.full((nx, ny), dy)
    e_mat = np.full((nx, ny), dx)
    w_mat = np.full((nx, ny), dx)
    p_mat = np.full((nx, ny), dx)
    q_mat = np.full((nx, ny), dy)
    
    if polarization.lower() == 'ex':
        an = 2.0 / (n_mat * (n_mat + s_mat))
        as_ = 2.0 / (s_mat * (n_mat + s_mat))
        num_e = 8.0 * (p_mat * (ep - ew) + 2.0 * w_mat * ew) * ee
        den_e = (p_mat * (ep - ee) + 2.0 * e_mat * ee) * (p_mat**2 * (ep - ew) + 4.0 * w_mat**2 * ew) + \
                (p_mat * (ep - ew) + 2.0 * w_mat * ew) * (p_mat**2 * (ep - ee) + 4.0 * e_mat**2 * ee)
        ae = num_e / den_e
        num_w = 8.0 * (p_mat * (ep - ee) + 2.0 * e_mat * ee) * ew
        aw = num_w / den_e
        ap = ep * (k0**2) - an - as_ - ae * (ep / ee) - aw * (ep / ew)
    else:
        num_n = 8.0 * (q_mat * (ep - es) + 2.0 * s_mat * es) * en
        den_n = (q_mat * (ep - en) + 2.0 * n_mat * en) * (q_mat**2 * (ep - es) + 4.0 * s_mat**2 * es) + \
                (q_mat * (ep - es) + 2.0 * s_mat * es) * (q_mat**2 * (ep - en) + 4.0 * n_mat**2 * en)
        an = num_n / den_n
        as_ = 8.0 * (q_mat * (ep - en) + 2.0 * n_mat * en) * es / den_n
        ae = 2.0 / (e_mat * (e_mat + w_mat))
        aw = 2.0 / (w_mat * (e_mat + w_mat))
        ap = ep * (k0**2) - an * (ep / en) - as_ - ae - aw

    N = nx * ny
    main_diag = ap.flatten('F')
    ae_diag = ae.flatten('F')[:-1]
    aw_diag = aw.flatten('F')[1:]
    an_diag = an.flatten('F')[:-nx]
    as_diag = as_.flatten('F')[nx:]
    
    A = sp.diags([main_diag, ae_diag, aw_diag, an_diag, as_diag], [0, 1, -1, nx, -nx], shape=(N, N), format='csc')
    shift = (2.0 * np.pi * guess / lam_um)**2
    vals, vecs = spla.eigs(A, k=nmodes, sigma=shift, which='LM')
    
    neff_vals = (lam_um / (2.0 * np.pi)) * np.sqrt(np.real(vals))
    phi_modes = np.zeros((nx, ny, nmodes))
    
    for idx in range(nmodes):
        mode_2d = np.real(vecs[:, idx]).reshape((nx, ny), order='F')
        max_abs = np.max(np.abs(mode_2d))
        if max_abs > 0:
            mode_2d /= max_abs
        phi_modes[:, :, idx] = mode_2d
        
    return phi_modes, neff_vals

def run_asymmetric_simulation(w1, w2, h_core, gap, coupler_L, ring_R, lambda_start, lambda_end, n_lambda, polarization, res_mode, top_ox, bottom_ox=4.0, core_material="Si3N4 (Stoichiometric)", progress_callback=None):
    dx = dy = 0.005 if "hr" in res_mode else (0.01 if "mr" in res_mode else 0.02)
    side = 2.0
    
    lambda_vec = np.linspace(lambda_start, lambda_end, n_lambda)
    neff1_vec = np.zeros(n_lambda)
    neff2_vec = np.zeros(n_lambda)
    neff_even_vec = np.zeros(n_lambda)
    neff_odd_vec = np.zeros(n_lambda)
    
    delta_vec = np.zeros(n_lambda)
    kappa_eff_vec = np.zeros(n_lambda)
    kappa_pure_vec = np.zeros(n_lambda)
    f_max_vec = np.zeros(n_lambda)
    
    l_residual_vec = np.zeros(n_lambda)
    l_total_vec = np.zeros(n_lambda)
    p_cross_vec = np.zeros(n_lambda)
    p_bar_vec = np.zeros(n_lambda)
    
    idx_center = n_lambda // 2
    
    for i in range(n_lambda):
        if progress_callback:
            progress_callback(i + 1, n_lambda)
            
        current_lambda = lambda_vec[i]
        k0 = 2.0 * np.pi / current_lambda
        
        n_core = get_core_index(current_lambda, core_material)
        n_clad = sellmeier_sio2(current_lambda)
        
        guess = (n_core + n_clad) / 2.0
        
        # 1. Single Waveguide 1
        _, _, eps_wg1 = single_waveguide_mesh(w1, h_core, side, bottom_ox, top_ox, dx, dy, n_core, n_clad)
        _, neff1_val = svmodes_2d(current_lambda, guess, 1, dx, dy, eps_wg1, polarization)
        neff1_vec[i] = neff1_val[0]
        
        # 2. Single Waveguide 2
        _, _, eps_wg2 = single_waveguide_mesh(w2, h_core, side, bottom_ox, top_ox, dx, dy, n_core, n_clad)
        _, neff2_val = svmodes_2d(current_lambda, guess, 1, dx, dy, eps_wg2, polarization)
        neff2_vec[i] = neff2_val[0]
        
        # 3. Coupled Asymmetric System
        xc, yc, eps_mesh, x1_l, x1_r, x2_l, x2_r, y_b, y_t = waveguidemesh_asymmetric(
            w1, w2, h_core, gap, side, bottom_ox, top_ox, dx, dy, n_core, n_clad
        )
        phi_modes, neff_vals = svmodes_2d(current_lambda, guess, 2, dx, dy, eps_mesh, polarization)
        
        sorted_indices = np.argsort(neff_vals)[::-1]
        neff_even_vec[i] = neff_vals[sorted_indices[0]]
        neff_odd_vec[i] = neff_vals[sorted_indices[1]]
        
        # Physics Calculations for Asymmetric Coupling
        delta_vec[i] = (np.pi / current_lambda) * (neff1_vec[i] - neff2_vec[i])
        kappa_eff_vec[i] = (np.pi / current_lambda) * (neff_even_vec[i] - neff_odd_vec[i])
        
        # Pure coupling kappa = sqrt(kappa_eff^2 - delta^2)
        diff_sq = kappa_eff_vec[i]**2 - delta_vec[i]**2
        kappa_pure_vec[i] = np.sqrt(np.maximum(diff_sq, 0.0))
        
        # Power transfer fraction F = P_max
        f_max_vec[i] = (kappa_pure_vec[i] / kappa_eff_vec[i])**2 if kappa_eff_vec[i] > 0 else 0.0
        
        if ring_R > 0:
            n_eff_avg = (neff_even_vec[i] + neff_odd_vec[i]) / 2.0
            gamma_val = k0 * np.sqrt(max(n_eff_avg**2 - n_clad**2, 1e-4))
            l_residual_vec[i] = np.sqrt(np.pi * ring_R / gamma_val)
        else:
            l_residual_vec[i] = 0.0
            
        l_total_vec[i] = coupler_L + l_residual_vec[i]
        p_cross_vec[i] = f_max_vec[i] * (np.sin(kappa_eff_vec[i] * l_total_vec[i]))**2 * 100.0
        p_bar_vec[i] = 100.0 - p_cross_vec[i]
        
        if i == idx_center:
            eps_mesh_center = eps_mesh.copy()
            phi_even = phi_modes[:, :, sorted_indices[0]]
            phi_odd = phi_modes[:, :, sorted_indices[1]]
            
            if np.sum(phi_even) < 0: phi_even = -phi_even
            phi_even /= np.max(np.abs(phi_even))
            
            mid_x_idx = len(xc) // 2
            if np.sum(phi_odd[mid_x_idx:, :]) < 0: phi_odd = -phi_odd
            phi_odd /= np.max(np.abs(phi_odd))
            
            xc_center, yc_center = xc, yc
            lambda_center_val = current_lambda
            mid_y_idx = np.argmin(np.abs(yc - (y_b + y_t) / 2.0))

    L_ring_um = (2 * np.pi * ring_R + 2 * coupler_L) if ring_R > 0 else (2 * coupler_L)
    L_ring_cm = L_ring_um * 1e-4
    alpha_db_vals = np.array([0.5, 1.5, 5.0])
    alpha_cm = alpha_db_vals * (np.log(10) / 10.0)
    round_trip_loss_pct = (1.0 - np.exp(-alpha_cm * L_ring_cm)) * 100.0
    
    neff_avg_vec = (neff_even_vec + neff_odd_vec) / 2.0
    lambda_cm_center = lambda_center_val * 1e-4
    dneff_dlambda = (neff_avg_vec[-1] - neff_avg_vec[0]) / ((lambda_vec[-1] - lambda_vec[0]) * 1e-4)
    n_group = neff_avg_vec[idx_center] - lambda_cm_center * dneff_dlambda
    
    Q0_vals = (2.0 * np.pi * n_group) / (lambda_cm_center * alpha_cm)
    QL_vals = Q0_vals / 2.0
    
    return {
        'xc': xc_center, 'yc': yc_center, 'eps_center': eps_mesh_center,
        'phi_even': phi_even, 'phi_odd': phi_odd, 'mid_y_idx': mid_y_idx,
        'lambda_vec': lambda_vec, 'neff1_vec': neff1_vec, 'neff2_vec': neff2_vec,
        'neff_even': neff_even_vec, 'neff_odd': neff_odd_vec,
        'delta_vec': delta_vec, 'kappa_eff_vec': kappa_eff_vec, 'kappa_pure_vec': kappa_pure_vec,
        'f_max_vec': f_max_vec, 'l_residual_vec': l_residual_vec, 'l_total_vec': l_total_vec,
        'p_cross_vec': p_cross_vec, 'p_bar_vec': p_bar_vec, 'round_trip_loss_pct': round_trip_loss_pct,
        'QL_vals': QL_vals, 'alpha_db_vals': alpha_db_vals, 'L_ring_um': L_ring_um,
        'lambda_center_val': lambda_center_val, 'idx_center': idx_center,
        'box1_l': x1_l, 'box1_r': x1_r, 'box2_l': x2_l, 'box2_r': x2_r,
        'b_y': y_b, 't_y': y_t, 'polarization': polarization,
        'w1': w1, 'w2': w2, 'h_core': h_core, 'gap': gap, 'coupler_L': coupler_L,
        'ring_R': ring_R, 'top_ox': top_ox, 'bottom_ox': bottom_ox,
        'core_material': core_material
    }
