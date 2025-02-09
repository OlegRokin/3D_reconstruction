NAME = "cameramodel"

import numpy as np
# from scipy.sparse.linalg import eigsh
# import scipy.ndimage as ndimage
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.patches as patches
from matplotlib.collections import LineCollection, PolyCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from matplotlib.widgets import Slider
from matplotlib.image import imread

def error(X_pix, X_model):
    X_visible = ~np.isnan(X_pix)[:,:,0]
    X_pix_center = X_pix + 0.5
    X_diff = np.nan_to_num(X_model - X_pix_center)
    return np.dot(X_diff.ravel(), X_diff.ravel()) / (2 * X_visible.sum())

def rx(theta):
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    zeros, ones = np.zeros_like(theta), np.ones_like(theta)
    R = np.array([[ones, zeros, zeros],
                  [zeros, cos_theta, - sin_theta],
                  [zeros, sin_theta, cos_theta]])
    return np.transpose(R, (2, 0, 1))

def ry(theta):
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    zeros, ones = np.zeros_like(theta), np.ones_like(theta)
    R = np.array([[cos_theta, zeros, sin_theta],
                  [zeros, ones, zeros],
                  [- sin_theta, zeros, cos_theta]])
    return np.transpose(R, (2, 0, 1))

def rz(theta):
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    zeros, ones = np.zeros_like(theta), np.ones_like(theta)
    R = np.array([[cos_theta, - sin_theta, zeros],
                  [sin_theta, cos_theta, zeros],
                  [zeros, zeros, ones]])
    return np.transpose(R, (2, 0, 1))

def ryxz(Theta):
    return ry(Theta[:,1]) @ rx(Theta[:,0]) @ rz(Theta[:,2])

def d_rx(theta):
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    zeros = np.zeros_like(theta)
    R = np.array([[zeros, zeros, zeros],
                  [zeros, - sin_theta, - cos_theta],
                  [zeros, cos_theta, - sin_theta]])
    return np.transpose(R, (2, 0, 1))

def d_ry(theta):
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    zeros = np.zeros_like(theta)
    R = np.array([[- sin_theta, zeros, cos_theta],
                  [zeros, zeros, zeros],
                  [- cos_theta, zeros, - sin_theta]])
    return np.transpose(R, (2, 0, 1))

def d_rz(theta):
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    zeros = np.zeros_like(theta)
    R = np.array([[- sin_theta, - cos_theta, zeros],
                  [cos_theta, - sin_theta, zeros],
                  [zeros, zeros, zeros]])
    return np.transpose(R, (2, 0, 1))

def project(X, C, Theta, f=1.0, dir_only=False):
    X_rot = np.einsum('jlk,ijl->ijk', ryxz(Theta), X[:,np.newaxis,:] - C[np.newaxis,:,:])
    X_proj = np.full_like(X_rot, f)
    X_proj[:,:,:2] = f * X_rot[:,:,:2] / X_rot[:,:,2][:,:,np.newaxis]
    X_proj = np.einsum('jkl,ijl->ijk', ryxz(Theta), X_proj)
    if dir_only: return X_proj
    else: return X_proj + C[np.newaxis,:,:]

def reverse_project(X_pix, W, r, C, Theta, phi_x, f=1.0, dir_only=False):
    X_pix_center = X_pix + 0.5
    X_proj = np.full((*X_pix.shape[:2], 3), f)
    X_proj[:,:,:2] = 2 * f * np.tan(phi_x / 2) * (X_pix_center[:,:,:2] / W - np.array([0.5, 0.5 / r]))
    X_proj = np.einsum('jkl,ijl->ijk', ryxz(Theta), X_proj)
    if dir_only: return X_proj
    else: return X_proj + C[np.newaxis,:,:]

def transform(X, C, Theta, phi_x, W, r, delete_back_points=True, delete_boundary_points=False):
    X_rot = np.einsum('jlk,ijl->ijk', ryxz(Theta), X[:,np.newaxis,:] - C[np.newaxis,:,:])
    if delete_back_points: X_rot[np.where(X_rot[:,:,2] <= 0)] = np.nan
    # X_model = W / 2 * (X_rot[:,:,:2] / X_rot[:,:,2][:,:,np.newaxis] / np.tan(phi_x / 2) + np.array([1.0, 1.0 / r])[np.newaxis,np.newaxis,:])
    X_model = W / 2 * (X_rot[:,:,:2] / X_rot[:,:,2][:,:,np.newaxis] / np.tan(phi_x / 2) + np.array([1.0, 1.0 / r]))
    if delete_boundary_points:
        I, J = np.where((X_model[:,:,0] < 0) | (X_model[:,:,0] >= W) | (X_model[:,:,1] < 0) | (X_model[:,:,1] >= W / r))
        X_model[I, J] = np.nan
    return X_model

def distance(X, C, Theta):
    X_rot = np.einsum('jlk,ijl->ijk', ryxz(Theta), X[:,np.newaxis,:] - C[np.newaxis,:,:])
    return X_rot[:,:,2]

def nearest_dot(X, V):
    n = X.shape[0]
    M = np.full((n, n), -1)
    np.fill_diagonal(M, n - 1)
    A = V @ V.T * M
    b = - np.sum(V @ X.T * M, axis=1, keepdims=True)
    lmbda = np.linalg.solve(A, b)
    X_sol = X + lmbda * V
    x_sol = np.mean(X_sol, axis=0)
    return x_sol

def find_dots(X_pix, W, r, C, Theta, phi_x):
    X_proj = reverse_project(X_pix, W, r, C, Theta, phi_x, dir_only=True)
    nearest_dots = np.zeros((X_proj.shape[0], 3))
    for i, x_proj in enumerate(X_proj):
        j_notnan = np.where(~np.isnan(x_proj)[:,0])[0]
        nearest_dots[i,:] = nearest_dot(C[j_notnan,:], x_proj[j_notnan,:])
    return nearest_dots

def sphere_project(x, o, R):
    return o + R * (x - o) / np.linalg.norm(x - o)

def sgd(x, grad_x, lr):
    return x - lr * grad_x

def gd_adam(x, grad_x, lr, s, r, t, rho_1=0.9, rho_2=0.999):
    s_new = rho_1 * s + (1 - rho_1) * grad_x
    r_new = rho_2 * r + (1 - rho_2) * grad_x**2
    s_corr = s_new / (1 - rho_1**t)
    r_corr = r_new / (1 - rho_2**t)
    return x - lr * s_corr / (1e-8 + np.sqrt(r_corr)), s_new, r_new

def fit(X_pix, W, r, lr, max_iters, X_0, C_0, Theta_0, phi_x_0,
        X_mask=True, phi_x_mask=True,
        optimizer='Adam', patience=1000, factor=2.0,
        stop_value=0.0, stop_diff=0.0, print_step=1000, ret_arrays=False):
    X_visible = ~np.isnan(X_pix)[:,:,0]
    NK_nan = X_visible.sum()
    X_pix_center = X_pix + 0.5
    N, K = X_pix.shape[:2]
    X, C, Theta, phi_x = X_0.copy(), C_0.copy(), Theta_0.copy(), phi_x_0
    X_release, C_release, Theta_release, phi_x_release = X, C, Theta, phi_x
    X_model = np.zeros_like(X_pix_center)

    D = {}
    D['R', 'Theta'] = np.zeros((K, 3, 3, 3))
    D['X_model', 'X_rot'] = np.zeros((N, K, 2, 3))

    if not phi_x_mask:
        tan_phi_x_2 = np.tan(phi_x / 2)
        f = W / 2 / tan_phi_x_2

    s_X = r_X = np.zeros_like(X)
    s_C = r_C = np.zeros_like(C)
    s_Theta = r_Theta = np.zeros_like(Theta)
    s_phi_x = r_phi_x = 0
    
    iters = 0
    E = np.zeros(max_iters)
    E_min, E_min_temp = np.inf, np.inf
    # patience_timer = patience
    patience_timer = 0
    success_timer = patience
    began_decreasing = False
    increased = False
    # diff_timer = patience

    # R_scale = np.linalg.norm(C[main_indexes[0],:] - C[main_indexes[1],:])

    if ret_arrays:
        X_list, C_list, Theta_list, phi_x_list = [], [], [], []

    for iters in range(max_iters):
        if ret_arrays:
            X_list.append(X)
            C_list.append(C)
            Theta_list.append(Theta)
            phi_x_list.append(phi_x)

        X_tr = X[:,np.newaxis,:] - C[np.newaxis,:,:]

        Rx, Ry, Rz = rx(Theta[:,0]), ry(Theta[:,1]), rz(Theta[:,2])
        if iters == 0: R = Ry @ Rx @ Rz
        X_rot = np.einsum('jlk,ijl->ijk', R, X_tr)

        if phi_x_mask:
            tan_phi_x_2 = np.tan(phi_x / 2)
            f = W / 2 / tan_phi_x_2
        # X_model = W / 2 * (X_rot[:,:,:2] / X_rot[:,:,2][:,:,np.newaxis] / tan_phi_x_2 + np.array([1.0, 1.0 / r])[np.newaxis,np.newaxis,:])      # должно считаться быстрее, чем через задание X_model[:,:,0] и X_model[:,:,1] по отдельности
        X_model = W / 2 * (X_rot[:,:,:2] / X_rot[:,:,2][:,:,np.newaxis] / tan_phi_x_2 + np.array([1.0, 1.0 / r]))      # должно считаться быстрее, чем через задание X_model[:,:,0] и X_model[:,:,1] по отдельности

        X_diff = np.nan_to_num(X_model - X_pix_center)
        E[iters] = np.dot(X_diff.ravel(), X_diff.ravel()) / (2 * NK_nan)     # должно считаться в >2 раза быстрее, чем np.sum(X_diff**2) / (2 * NK_nan)
        
        # if iters % print_step == 0: print(f'{iters} : {E[iters]}')
        if iters == 0:
            E_min = E[iters]
        else:
            if E[iters] < E_min:
                E_min = E[iters]
                if not began_decreasing: began_decreasing = True
                X_release, C_release, Theta_release, phi_x_release = X, C, Theta, phi_x
                if patience_timer != patience:
                    patience_timer = patience
                # if not increased:
                if success_timer > 0:
                    success_timer -= 1
                else:
                    print(f'{iters} : Icrease LR to {lr * factor}')
                    lr *= factor
                    success_timer = patience
                    increased = True
                    E_min_temp = E_min
                    if optimizer == 'Adam':
                        s_X_temp, r_X_temp = s_X.copy(), r_X.copy()
                        s_C_temp, r_C_temp = s_C.copy(), r_C.copy()
                        s_Theta_temp, r_Theta_temp = s_Theta.copy(), r_Theta.copy()
                        s_phi_x_temp, r_phi_x_temp = s_phi_x, r_phi_x
                        s_X = r_X = np.zeros_like(X)
                        s_C = r_C = np.zeros_like(C)
                        s_Theta = r_Theta = np.zeros_like(Theta)
                        s_phi_x = r_phi_x = 0
            elif E[iters] > E_min:
                if increased:
                    patience_timer = 0
                if success_timer != patience:
                    success_timer = patience
                if patience_timer > 0:
                    patience_timer -= 1
                else:
                    print(f'{iters} : Decrease LR to {lr / factor}')
                    lr /= factor
                    if began_decreasing:
                        patience_timer = patience
                    X, C, Theta, phi_x = X_release, C_release, Theta_release, phi_x_release
                    X_tr = X[:,np.newaxis,:] - C[np.newaxis,:,:]
                    Rx, Ry, Rz = rx(Theta[:,0]), ry(Theta[:,1]), rz(Theta[:,2])
                    R = Ry @ Rx @ Rz
                    X_rot = np.einsum('jlk,ijl->ijk', R, X_tr)
                    if phi_x_mask:
                        tan_phi_x_2 = np.tan(phi_x / 2)
                        f = W / 2 / tan_phi_x_2
                    # X_model = W / 2 * (X_rot[:,:,:2] / X_rot[:,:,2][:,:,np.newaxis] / tan_phi_x_2 + np.array([1.0, 1.0 / r])[np.newaxis,np.newaxis,:])      # должно считаться быстрее, чем через задание X_model[:,:,0] и X_model[:,:,1] по отдельности
                    X_model = W / 2 * (X_rot[:,:,:2] / X_rot[:,:,2][:,:,np.newaxis] / tan_phi_x_2 + np.array([1.0, 1.0 / r]))      # должно считаться быстрее, чем через задание X_model[:,:,0] и X_model[:,:,1] по отдельности
                    X_diff = np.nan_to_num(X_model - X_pix_center)
                    if optimizer == 'Adam':
                        if not increased or (increased and E_min_temp > E_min):
                        # if E_min_temp > E_min:
                            s_X = r_X = np.zeros_like(X)
                            s_C = r_C = np.zeros_like(C)
                            s_Theta = r_Theta = np.zeros_like(Theta)
                            s_phi_x = r_phi_x = 0
                        else:
                            s_X, r_X = s_X_temp.copy(), r_X_temp.copy()
                            s_C, r_C = s_C_temp.copy(), r_C_temp.copy()
                            s_Theta, r_Theta = s_Theta_temp.copy(), r_Theta_temp.copy()
                            s_phi_x, r_phi_x = s_phi_x_temp, r_phi_x_temp
                            increased = False
            # if iters >= patience and np.abs(E[iters - patience] - E[iters]) <= stop_diff:
            #     if diff_timer > 0:
            #         diff_timer -= 1
            #     else:
            #         break
            if iters > 2 * patience and E[:iters-2*patience].min() - E_min  < stop_diff:
                # print(f'E[:{iters-2*patience}].min() - E_min = {E[:iters-2*patience].min() - E_min}')
                break
        if E[iters] <= stop_value:
            break

        if iters % print_step == 0: print(f'{iters} : {E_min}')

        D['R', 'Theta'][:,:,:,0] = Ry @ d_rx(Theta[:,0]) @ Rz
        D['R', 'Theta'][:,:,:,1] = d_ry(Theta[:,1]) @ Rx @ Rz
        D['R', 'Theta'][:,:,:,2] = Ry @ Rx @ d_rz(Theta[:,2])
        D['X_rot', 'Theta'] = np.einsum('jnkl,ijn->ijkl', D['R', 'Theta'], X_tr)

        D['X_model', 'X_rot'][:,:,0,0] = D['X_model', 'X_rot'][:,:,1,1] = f / X_rot[:,:,2]
        D['X_model', 'X_rot'][:,:,:,2] = - f / (X_rot[:,:,2]**2)[:,:,np.newaxis] * X_rot[:,:,:2]

        D['X_model', 'Theta'] = np.einsum('ijkn,ijnl->ijkl', D['X_model', 'X_rot'], D['X_rot', 'Theta'])

        D['X_model', 'X'] = np.einsum('ijkn,jln->ijkl', D['X_model', 'X_rot'], R)

        D['E', 'X_model'] = X_diff / NK_nan

        if X_mask:
            D['E', 'X'] = np.einsum('ijl,ijlk->ik', D['E', 'X_model'], D['X_model', 'X'])

        D['E', 'C'] = - np.einsum('ijl,ijlk->jk', D['E', 'X_model'], D['X_model', 'X'])
        # if X_mask:
        #     D['E', 'C'][main_indexes[0],:] = 0.0

        D['E', 'Theta'] = np.einsum('ijl,ijlk->jk', D['E', 'X_model'], D['X_model', 'Theta'])
        # if X_mask:
        #     D['E', 'Theta'][main_indexes[0],:] = 0.0

        if phi_x_mask:
            D['X_model', 'phi_x'] = - f / X_rot[:,:,2][:,:,np.newaxis] * X_rot[:,:,:2] / np.sin(phi_x)
            D['E', 'phi_x'] = np.dot(D['E', 'X_model'].ravel(), D['X_model', 'phi_x'].ravel())      # должно считаться в >2 раза быстрее, чем np.sum(D['E', 'X_model'] * D['X_model', 'alpha_x'])

        if optimizer == 'SGD':
            if X_mask:
                X = sgd(X, D['E', 'X'], lr)
            C = sgd(C, D['E', 'C'], lr)
            Theta = sgd(Theta, D['E', 'Theta'], lr)
            if phi_x_mask:
                phi_x = sgd(phi_x, D['E', 'phi_x'], lr)
        elif optimizer == 'Adam':
            if X_mask:
                X, s_X, r_X = gd_adam(X, D['E', 'X'], lr, s_X, r_X, t=iters+1)
            C, s_C, r_C = gd_adam(C, D['E', 'C'], lr, s_C, r_C, t=iters+1)
            Theta, s_Theta, r_Theta = gd_adam(Theta, D['E', 'Theta'], lr, s_Theta, r_Theta, t=iters+1)
            if phi_x_mask:
                phi_x, s_phi_x, r_phi_x = gd_adam(phi_x, D['E', 'phi_x'], lr, s_phi_x, r_phi_x, t=iters+1)

        if X_mask:
            X, C, Theta, R = align_scene(X, C, Theta, ret_R=True)
            X, C = normalize_scene(X, C)

    print(f'{E_min = }')

    if not ret_arrays: return X_release, C_release, Theta_release, phi_x_release, E[E > 0]
    else: return X_release, C_release, Theta_release, phi_x_release, E[E > 0], np.array(X_list), np.array(C_list), np.array(Theta_list), np.array(phi_x_list)

def fit_minibatch(X_pix, W, r, lr, max_iters, X_0, C_0, Theta_0, phi_x_0,
        X_mask=True, phi_x_mask=True,
        optimizer='Adam', patience=1000, factor=2.0,
        stop_value=0.0, stop_diff=0.0, print_step=1000, ret_arrays=False):
    X_visible = ~np.isnan(X_pix)[:,:,0]
    NK_nan = X_visible.sum()
    X_pix_center = X_pix + 0.5
    N, K = X_pix.shape[:2]
    X, C, Theta, phi_x = X_0.copy(), C_0.copy(), Theta_0.copy(), phi_x_0
    X_release, C_release, Theta_release, phi_x_release = X, C, Theta, phi_x
    X_model = np.zeros_like(X_pix_center)

    D = {}
    # D['R', 'Theta'] = np.zeros((K, 3, 3, 3))
    # D['X_model', 'X_rot'] = np.zeros((N, K, 2, 3))

    if not phi_x_mask:
        tan_phi_x_2 = np.tan(phi_x / 2)
        f = W / 2 / tan_phi_x_2

    s_X = r_X = np.zeros_like(X)
    s_C = r_C = np.zeros_like(C)
    s_Theta = r_Theta = np.zeros_like(Theta)
    s_phi_x = r_phi_x = 0
    
    iters = 0
    E = np.zeros(max_iters)
    E_min, E_min_temp = np.inf, np.inf
    patience_timer = 0
    success_timer = patience
    began_decreasing = False
    increased = False
    diff_timer = patience

    if ret_arrays:
        X_list, C_list, Theta_list, phi_x_list = [], [], [], []

    for iters in range(max_iters):
        if ret_arrays:
            X_list.append(X)
            C_list.append(C)
            Theta_list.append(Theta)
            phi_x_list.append(phi_x)

        X_tr = X[:,np.newaxis,:] - C[np.newaxis,:,:]

        Rx, Ry, Rz = rx(Theta[:,0]), ry(Theta[:,1]), rz(Theta[:,2])
        if iters == 0: R = Ry @ Rx @ Rz
        X_rot = np.einsum('jlk,ijl->ijk', R, X_tr)

        if phi_x_mask:
            tan_phi_x_2 = np.tan(phi_x / 2)
            f = W / 2 / tan_phi_x_2
        X_model = W / 2 * (X_rot[:,:,:2] / X_rot[:,:,2][:,:,np.newaxis] / tan_phi_x_2 + np.array([1.0, 1.0 / r]))      # должно считаться быстрее, чем через задание X_model[:,:,0] и X_model[:,:,1] по отдельности

        X_diff = np.nan_to_num(X_model - X_pix_center)
        E[iters] = np.dot(X_diff.ravel(), X_diff.ravel()) / (2 * NK_nan)     # должно считаться в >2 раза быстрее, чем np.sum(X_diff**2) / (2 * NK_nan)

        if iters == 0:
            E_min = E[iters]
        else:
            if E[iters] < E_min:
                E_min = E[iters]
                if not began_decreasing: began_decreasing = True
                X_release, C_release, Theta_release, phi_x_release = X, C, Theta, phi_x
                if patience_timer != patience:
                    patience_timer = patience
                if success_timer > 0:
                    success_timer -= 1
                else:
                    print(f'{iters} : Icrease LR to {lr * factor}')
                    lr *= factor
                    success_timer = patience
                    increased = True
                    E_min_temp = E_min
                    if optimizer == 'Adam':
                        s_X_temp, r_X_temp = s_X.copy(), r_X.copy()
                        s_C_temp, r_C_temp = s_C.copy(), r_C.copy()
                        s_Theta_temp, r_Theta_temp = s_Theta.copy(), r_Theta.copy()
                        s_phi_x_temp, r_phi_x_temp = s_phi_x, r_phi_x
                        s_X = r_X = np.zeros_like(X)
                        s_C = r_C = np.zeros_like(C)
                        s_Theta = r_Theta = np.zeros_like(Theta)
                        s_phi_x = r_phi_x = 0
            elif E[iters] > E_min:
                if increased:
                    patience_timer = 0
                if success_timer != patience:
                    success_timer = patience
                if patience_timer > 0:
                    patience_timer -= 1
                else:
                    print(f'{iters} : Decrease LR to {lr / factor}')
                    lr /= factor
                    if began_decreasing:
                        patience_timer = patience
                    X, C, Theta, phi_x = X_release, C_release, Theta_release, phi_x_release
                    X_tr = X[:,np.newaxis,:] - C[np.newaxis,:,:]
                    Rx, Ry, Rz = rx(Theta[:,0]), ry(Theta[:,1]), rz(Theta[:,2])
                    R = Ry @ Rx @ Rz
                    X_rot = np.einsum('jlk,ijl->ijk', R, X_tr)
                    if phi_x_mask:
                        tan_phi_x_2 = np.tan(phi_x / 2)
                        f = W / 2 / tan_phi_x_2
                    X_model = W / 2 * (X_rot[:,:,:2] / X_rot[:,:,2][:,:,np.newaxis] / tan_phi_x_2 + np.array([1.0, 1.0 / r]))      # должно считаться быстрее, чем через задание X_model[:,:,0] и X_model[:,:,1] по отдельности
                    X_diff = np.nan_to_num(X_model - X_pix_center)
                    if optimizer == 'Adam':
                        if not increased or (increased and E_min_temp > E_min):
                            s_X = r_X = np.zeros_like(X)
                            s_C = r_C = np.zeros_like(C)
                            s_Theta = r_Theta = np.zeros_like(Theta)
                            s_phi_x = r_phi_x = 0
                        else:
                            s_X, r_X = s_X_temp.copy(), r_X_temp.copy()
                            s_C, r_C = s_C_temp.copy(), r_C_temp.copy()
                            s_Theta, r_Theta = s_Theta_temp.copy(), r_Theta_temp.copy()
                            s_phi_x, r_phi_x = s_phi_x_temp, r_phi_x_temp
                            increased = False
            if iters >= patience and np.abs(E[iters - patience] - E[iters]) <= stop_diff:
                if diff_timer > 0:
                    diff_timer -= 1
                else:
                    break
        if E[iters] <= stop_value:
            break

        if iters % print_step == 0: print(f'{iters} : {E_min}')

        # d_Rx, d_Ry, d_Rz = d_rx(Theta[:,0]), d_ry(Theta[:,1]), d_rz(Theta[:,2])

        for i in range(N):
            j_subset = np.where(X_visible[i,:])[0]
            # print(f'{i = }, {j_subset = }')

            D['R_sub', 'Theta_sub'] = np.zeros((j_subset.size, 3, 3, 3))
            D['X_model_sub', 'X_rot_sub'] = np.zeros((j_subset.size, 2, 3))

            if phi_x_mask:
                tan_phi_x_2 = np.tan(phi_x / 2)
                f = W / 2 / tan_phi_x_2

            if i > 0:
                X_tr_sub = X[i] - C[j_subset]
                Rx_sub, Ry_sub, Rz_sub = rx(Theta[j_subset,0]), ry(Theta[j_subset,1]), rz(Theta[j_subset,2])
                # if iters == 0: R = Ry @ Rx @ Rz
                # R_sub = R[j_subset]
                R_sub = Rx_sub @ Ry_sub @ Rz_sub
                # if iters == 0: R_sub = R[j_subset]
                # я пока не придумал, как на расчетах R_sub экономить
                X_rot_sub = np.einsum('jlk,jl->jk', R_sub, X_tr_sub)
                X_model_sub = W / 2 * (X_rot_sub[:,:2] / X_rot_sub[:,2][:,np.newaxis] / tan_phi_x_2 + np.array([1.0, 1.0 / r]))      # должно считаться быстрее, чем через задание X_model[:,:,0] и X_model[:,:,1] по отдельности
                X_diff_sub = np.nan_to_num(X_model_sub - X_pix_center[i,j_subset,:])
            else:
                X_tr_sub = X_tr[i,j_subset]
                Rx_sub, Ry_sub, Rz_sub = Rx[j_subset], Ry[j_subset], Rz[j_subset]
                R_sub = R[j_subset]
                X_rot_sub = X_rot[i,j_subset]
                X_diff_sub = X_diff[i,j_subset]

            D['R_sub', 'Theta_sub'][:,:,:,0] = Ry_sub @ d_rx(Theta[j_subset,0]) @ Rz_sub
            D['R_sub', 'Theta_sub'][:,:,:,1] = d_ry(Theta[j_subset,1]) @ Rx_sub @ Rz_sub
            D['R_sub', 'Theta_sub'][:,:,:,2] = Ry_sub @ Rx_sub @ d_rz(Theta[j_subset,2])
            D['X_rot_sub', 'Theta_sub'] = np.einsum('jnkl,jn->jkl', D['R_sub', 'Theta_sub'], X_tr_sub)

            D['X_model_sub', 'X_rot_sub'][:,0,0] = D['X_model_sub', 'X_rot_sub'][:,1,1] = f / X_rot_sub[:,2]
            D['X_model_sub', 'X_rot_sub'][:,:,2] = - f / (X_rot_sub[:,2]**2)[:,np.newaxis] * X_rot_sub[:,:2]

            D['X_model_sub', 'Theta_sub'] = np.einsum('jkn,jnl->jkl', D['X_model_sub', 'X_rot_sub'], D['X_rot_sub', 'Theta_sub'])

            D['X_model_sub', 'X_sub'] = np.einsum('jkn,jln->jkl', D['X_model_sub', 'X_rot_sub'], R_sub)

            D['E_sub', 'X_model_sub'] = X_diff_sub / j_subset.size

            # if X_mask:
            #     D['E_sub', 'X_sub'] = np.einsum('jl,jlk->k', D['E_sub', 'X_model_sub'], D['X_model_sub', 'X_sub'])

            D['E_sub', 'C_sub'] = - np.einsum('jl,jlk->jk', D['E_sub', 'X_model_sub'], D['X_model_sub', 'X_sub'])

            if X_mask:
                D['E_sub', 'X_sub'] = - D['E_sub', 'C_sub'].sum(axis=0)

            D['E_sub', 'Theta_sub'] = np.einsum('jl,jlk->jk', D['E_sub', 'X_model_sub'], D['X_model_sub', 'Theta_sub'])

            if phi_x_mask:
                D['X_model_sub', 'phi_x'] = - f / X_rot_sub[:,2][:,np.newaxis] * X_rot_sub[:,:2] / np.sin(phi_x)
                D['E_sub', 'phi_x'] = np.dot(D['E_sub', 'X_model_sub'].ravel(), D['X_model_sub', 'phi_x'].ravel())      # должно считаться в >2 раза быстрее, чем np.sum(D['E', 'X_model'] * D['X_model', 'alpha_x'])

            # print(f'{i = }, {X[i] = }, {C[j_subset[0]] = }')

            if optimizer == 'SGD':
                if X_mask:
                    X[i] = sgd(X[i], D['E_sub', 'X_sub'], lr)
                C[j_subset] = sgd(C[j_subset], D['E_sub', 'C_sub'], lr)
                Theta[j_subset] = sgd(Theta[j_subset], D['E_sub', 'Theta_sub'], lr)
                if phi_x_mask:
                    phi_x = sgd(phi_x, D['E_sub', 'phi_x'], lr)
            elif optimizer == 'Adam':
                if X_mask:
                    X[i], s_X[i], r_X[i] = gd_adam(X[i], D['E_sub', 'X_sub'], lr, s_X[i], r_X[i], t=iters+1)
                C[j_subset], s_C[j_subset], r_C[j_subset] = gd_adam(C[j_subset], D['E_sub', 'C_sub'], lr, s_C[j_subset], r_C[j_subset], t=iters+1)
                Theta[j_subset], s_Theta[j_subset], r_Theta[j_subset] = gd_adam(Theta[j_subset], D['E_sub', 'Theta_sub'], lr, s_Theta[j_subset], r_Theta[j_subset], t=iters+1)
                if phi_x_mask:
                    phi_x, s_phi_x, r_phi_x = gd_adam(phi_x, D['E_sub', 'phi_x'], lr, s_phi_x, r_phi_x, t=iters+1)

            # print(f'{i = }, {X[i] = }, {C[j_subset[0]] = }')

            # print(f'{i = }, {np.linalg.norm(C - C[0], axis=1).max() = }')

            if X_mask:
                X, C, Theta, R = align_scene(X, C, Theta, ret_R=True)
                X, C = normalize_scene(X, C)

    print(f'{E_min = }')
    if not ret_arrays: return X_release, C_release, Theta_release, phi_x_release, E[E > 0]
    else: return X_release, C_release, Theta_release, phi_x_release, E[E > 0], np.array(X_list), np.array(C_list), np.array(Theta_list), np.array(phi_x_list)


def glue_scenes(X_scenes, C_scenes, Theta_scenes, lr,
                max_iters=100000, patience = 200, factor = 2.0,
                stop_value=0.0, print_step=1000):
    N, K = X_scenes.shape[:-1]
    S = np.ones(K)
    V = np.zeros((K, 3))
    Psi = np.zeros((K, 3))

    S_release, V_release, Psi_release = S, V, Psi

    D = {}
    D['R', 'Psi'] = np.zeros((K, 3, 3, 3))

    E = np.zeros(max_iters)
    E_min = np.inf
    patience_timer = patience
    success_timer = patience
    increased = False

    s_S = r_S = np.zeros_like(S)
    s_V = r_V = np.zeros_like(V)
    s_Psi = r_Psi = np.zeros_like(Psi)

    for iters in range(max_iters):
        Rx, Ry, Rz = rx(Psi[:,0]), ry(Psi[:,1]), rz(Psi[:,2])
        R = Ry @ Rx @ Rz
        X_rot = np.einsum('jkl,ijl->ijk', R, X_scenes)
        X_new = S[np.newaxis,:,np.newaxis] * X_rot + V[np.newaxis,:,:]
        X_mean = np.nanmean(X_new, axis=1, keepdims=True)
        X_diff_1 = X_new - X_mean
        X_diff_2 = X_new - X_scenes[:,0,:][:,np.newaxis,:]

        E[iters] = (np.dot(np.nan_to_num(X_diff_1).ravel(), np.nan_to_num(X_diff_1).ravel()) + np.dot(np.nan_to_num(X_diff_2).ravel(), np.nan_to_num(X_diff_2).ravel())) / (N * K)
        if iters % print_step == 0: print(f'{iters} : {E[iters]}, {S.mean() = }')
        if iters == 0:
            E_min = E[iters]
        else:
            if E[iters] < E_min:
                E_min = E[iters]
                S_release, V_release, Psi_release = S, V, Psi
                if patience_timer != patience:
                    patience_timer = patience
                if not increased:
                    if success_timer > 0:
                        success_timer -= 1
                    else:
                        print(f'{iters} : Icrease LR to {lr * factor}')
                        lr *= factor
                        success_timer = patience
                        increased = True
                        E_min_temp = E_min
                        s_S_temp, r_S_temp = s_S.copy(), r_S.copy()
                        s_V_temp, r_V_temp = s_V.copy(), r_V.copy()
                        s_Psi_temp, r_Psi_temp = s_Psi.copy(), r_Psi.copy()
                        s_S = r_S = np.zeros_like(S)
                        s_V = r_V = np.zeros_like(V)
                        s_Psi = r_Psi = np.zeros_like(Psi)
            elif E[iters] > E_min:
                if increased and E[iters] > E_min:
                    patience_timer = 0
                if success_timer != patience:
                    success_timer = patience
                if patience_timer > 0:
                    patience_timer -= 1
                else:
                    print(f'{iters} : Decrease LR to {lr / factor}')
                    lr /= factor
                    patience_timer = patience
                    S, V, Psi = S_release, V_release, Psi_release
                    Rx, Ry, Rz = rx(Psi[:,0]), ry(Psi[:,1]), rz(Psi[:,2])
                    R = Ry @ Rx @ Rz
                    X_rot = np.einsum('jkl,ijl->ijk', R, X_scenes)
                    X_new = S[np.newaxis,:,np.newaxis] * X_rot + V[np.newaxis,:,:]
                    X_mean = np.nanmean(X_new, axis=1, keepdims=True)
                    X_diff_1 = X_new - X_mean
                    X_diff_2 = X_new - X_scenes[:,0,:][:,np.newaxis,:]
                    if not increased or (increased and E_min_temp > E_min):
                        s_S = r_S = np.zeros_like(S)
                        s_V = r_V = np.zeros_like(V)
                        s_Psi = r_Psi = np.zeros_like(Psi)
                    else:
                        s_S, r_S = s_S_temp.copy(), r_S_temp.copy()
                        s_V, r_V = s_V_temp.copy(), r_V_temp.copy()
                        s_Psi, r_Psi = s_Psi_temp.copy(), r_Psi_temp.copy()
                    increased = False
        if E[iters] <= stop_value:
            break

        D['E', 'S'] = 2 * (np.einsum('ijk,ijk->j', np.nan_to_num(X_diff_1 - np.nanmean(X_diff_1, axis=1, keepdims=True)), np.nan_to_num(X_rot)) \
                           + np.einsum('ijk,ijk->j', np.nan_to_num(X_diff_2), np.nan_to_num(X_rot))) / (N * K)
        D['E', 'S'][0] = 0.0

        D['E', 'V'] = 2 * np.sum(np.nan_to_num(X_diff_1 - np.nanmean(X_diff_1, axis=1, keepdims=True)) + np.nan_to_num(X_diff_2), axis=0) / (N * K)
        D['E', 'V'][0] = 0.0

        D['R', 'Psi'][:,:,:,0] = Ry @ d_rx(Psi[:,0]) @ Rz
        D['R', 'Psi'][:,:,:,1] = d_ry(Psi[:,1]) @ Rx @ Rz
        D['R', 'Psi'][:,:,:,2] = Ry @ Rx @ d_rz(Psi[:,2])
        D['X_rot', 'Psi'] = np.einsum('jknl,ijn->ijkl', D['R', 'Psi'], np.nan_to_num(X_scenes))
        D['E', 'Psi'] = 2 * (np.einsum('ijn,j,ijnk->jk', np.nan_to_num(X_diff_1 - np.nanmean(X_diff_1, axis=1, keepdims=True)), S, D['X_rot', 'Psi']) \
                             + np.einsum('ijn,j,ijnk->jk', np.nan_to_num(X_diff_2), S, D['X_rot', 'Psi'])) / (N * K)
        D['E', 'Psi'][0] = 0.0

        S, s_S, r_S = gd_adam(S, D['E', 'S'], lr, s_S, r_S, t=iters+1)
        S = np.abs(S)
        V, s_V, r_V = gd_adam(V, D['E', 'V'], lr, s_V, r_V, t=iters+1)
        Psi, s_Psi, r_Psi = gd_adam(Psi, D['E', 'Psi'], lr, s_Psi, r_Psi, t=iters+1)

    print(S_release, V_release, Psi_release)

    Rx, Ry, Rz = rx(Psi_release[:,0]), ry(Psi_release[:,1]), rz(Psi_release[:,2])
    R = Ry @ Rx @ Rz

    X_rot = np.einsum('jkl,ijl->ijk', R, X_scenes)
    X_new = S_release[np.newaxis,:,np.newaxis] * X_rot + V_release[np.newaxis,:,:]

    C_rot = np.einsum('jkl,jnl->jnk', R, C_scenes)
    C_new = S_release[:,np.newaxis,np.newaxis] * C_rot + V_release[:,np.newaxis,:]

    R_old = np.zeros((K, 2, 3, 3))
    for j in range(K):
        R_old[j,:,:,:] = ryxz(Theta_scenes[j,:,:])
    R_new = np.einsum('jkl,jsln->jskn', R, R_old)
    Theta_new = get_Theta(R_new)

    return X_new, C_new, Theta_new

# def get_pixels(image):
#     labeled_image, num_features = ndimage.label(image > 1e-1)
#     pixels = np.zeros((num_features, 2))
#     for label in range(1, num_features + 1):
#         mask = labeled_image == label
#         if image.ndim == 2: coords = np.column_stack(np.where(mask))[:,::-1]
#         elif image.ndim == 3: coords = np.column_stack(np.where(mask)[1:])[:,::-1]
#         avg_coords = np.average(coords, weights=image[mask], axis=0)
#         pixels[label-1] = avg_coords
#     if image.ndim == 2: return pixels
#     elif image.ndim == 3: return pixels.reshape(image.shape[0], -1, 2).transpose(1, 0, 2)

def optimal_subplot_layout(n_plots, r):
    cols = np.ceil(np.sqrt(n_plots))
    rows = np.ceil(n_plots / cols)
    return int(rows), int(cols)

def safe_direction(X, id):
    x_A = X[id,:]
    X_diff = x_A[np.newaxis,:] - X
    X_dir = X_diff / np.linalg.norm(X_diff)
    x_dir = np.sum(X_dir, axis=0)
    return x_dir / np.linalg.norm(x_dir)

def draw_2d_views(axes, X_pix, W, H, X_model=None, object_corner_list=None, picture=None, draw_pixels=True, fontsize=10, fontcolor='black'):
    if X_model is None: X_pix_center = X_pix.astype(np.float64) + 0.5
    for j, ax in enumerate(axes):
        if j < X_pix.shape[1]:
            if picture is None:
                ax.set_xlim(0.0, W)
                ax.set_ylim(H, 0.0)
                ax.set_aspect('equal')
            else: ax.imshow(picture[j,:,:], cmap='gray')
            ax.set_title(f'Camera {j}')
            ax.grid(True)
            if object_corner_list:
                for corner in object_corner_list:
                    if X_model is None: ax.plot(*np.vstack((X_pix_center[corner[0],j,:], X_pix_center[corner[1],j,:])).T, color='red', linewidth=1)
                    else: ax.plot(*np.vstack((X_model[corner[0],j,:], X_model[corner[1],j,:])).T, color='red', linewidth=1)
            if X_model is None: ax.scatter(*X_pix_center[:,j,:].T, marker='o', s=10, color='red', zorder=10)           
            else: ax.scatter(*X_model[:,j,:].T, marker='o', s=10, color='red', zorder=10) 
            for i in range(X_pix.shape[0]):
                if draw_pixels:
                    ax.add_patch(patches.Rectangle(
                                    xy=X_pix[i,j,:],
                                    width=1, height=1, linewidth=1,
                                    color='red', fill=False, zorder=10))
                if X_model is None: vec = safe_direction(X_pix_center[:,j,:], i)
                else: vec = safe_direction(X_model[:,j,:], i)
                if vec[0] >= 0.33: ha='left'
                elif vec[0] <= - 0.33: ha='right'
                else: ha='center'
                if vec[1] >= 0.33: va='top'
                elif vec[1] <= - 0.33: va='bottom'
                else: va='center'
                if X_model is None: ax.annotate(f'${i}$', xy=X_pix_center[i,j,:], va=va, ha=ha, color=fontcolor, size=fontsize, zorder=10)
                else: ax.annotate(f'${i}$', xy=X_model[i,j,:], va=va, ha=ha, color=fontcolor, size=fontsize, zorder=10)
        else: ax.axis('off')

def rotate_array(V, R):
    if V.ndim == 2: return V @ R.T
    elif V.ndim == 3: return np.einsum('kl,ijl->ijk', R, V)

def align_scene(X, C, Theta, ret_R=False):
    X_new = X - C[0]
    C_new = C - C[0]

    R_zero = ryxz(Theta[0,:].reshape(1, -1))[0].T
    X_new = rotate_array(X_new, R_zero)
    C_new = rotate_array(C_new, R_zero)
    R_new = R_zero @ ryxz(Theta)
    Theta_new = get_Theta(R_new)

    if not ret_R: return X_new, C_new, Theta_new
    else: return X_new, C_new, Theta_new, R_new

def normalize_scene(X, C, main_indexes=None, R_scale=1.0, ret_j=False):
    if main_indexes is None:
        j0 = 0
        j = np.argmax(np.linalg.norm(C - C[0], axis=1))
    else: j0, j = main_indexes
    scale = np.linalg.norm(C[j] - C[j0])
    X_new = X * R_scale / scale
    C_new = C * R_scale / scale
    if not ret_j: return X_new, C_new
    else: return X_new, C_new, j

# def draw_3d_scene(ax, X, C, Theta, phi_x, r, f, object_corner_list=None, show_projections=True, show_traces=False, show_nums=False):
#     R = ryxz(Theta)

#     camera_dir = np.array([0.0, 0.0, 1.0])
#     camera_dir = np.einsum('ijk,k->ij', R, camera_dir)
#     tan_phi_x_2 = np.tan(phi_x / 2)
#     camera_corners = np.array([[tan_phi_x_2, tan_phi_x_2 / r, 1],
#                                [- tan_phi_x_2, tan_phi_x_2 / r, 1],
#                                [tan_phi_x_2, - tan_phi_x_2 / r, 1],
#                                [- tan_phi_x_2, - tan_phi_x_2 / r, 1]])
#     camera_corners = np.einsum('ijk,nk->inj', R, camera_corners)

#     if show_projections:
#         X_proj = project(X, C, Theta, f=f)

#     # Поворот всей системы вокруг оси x на угол -pi/2
#     Ryz = np.array([[1, 0, 0],
#                     [0, 0, 1],
#                     [0, -1, 0]])
#     X = rotate_array(X, Ryz)
#     C = rotate_array(C, Ryz)
#     camera_dir = rotate_array(camera_dir, Ryz)
#     camera_corners = rotate_array(camera_corners, Ryz)
#     if show_projections:
#         X_proj = rotate_array(X_proj, Ryz)


#     if show_traces:
#         for j in range(C.shape[0]):
#             ax.add_collection(Line3DCollection([[X[i], C[j]] for i in range(X.shape[0])],
#                                                linewidth=0.5, colors='red'))

#     if object_corner_list:
#         for corner in object_corner_list:
#             ax.add_collection(Line3DCollection([[X[corner[0],:], X[corner[1],:]]],
#                                                linewidth=1.5, colors='red'))
#         for j in range(C.shape[0]):
#             for corner in object_corner_list:
#                 ax.add_collection(Line3DCollection([[X_proj[corner[0],j,:], X_proj[corner[1],j,:]]],
#                                                    linewidth=1.0, colors='red'))

#     ax.scatter(*X.T, marker='o', s=40, color='red')

#     if show_projections:
#         ax.plot(*X_proj.reshape(X_proj.shape[0] * X_proj.shape[1], 3).T, linestyle=' ', marker='o', markersize=3, color='red')

#     ax.add_collection(Line3DCollection([[C[j,:], C[j,:] + f * camera_dir[j,:]] for j in range(C.shape[0])],
#                                        linewidth=1.0, linestyle='--', colors='blue'))
#     for k in range(4):
#         ax.add_collection(Line3DCollection([[C[j,:], C[j,:] + f * camera_corners[j,k,:]] for j in range(C.shape[0])],
#                                            linewidth=1.0, colors='blue'))

#     camera_corner_list = [[0, 1], [1, 3], [3, 2], [2, 0]]
#     for corner in camera_corner_list:
#         ax.add_collection(Line3DCollection([[C[j,:] + f * camera_corners[j,corner[0],:], C[j,:] + f * camera_corners[j,corner[1],:]] for j in range(C.shape[0])],
#                                            linewidth=1.0, colors='blue'))
#     for j in range(C.shape[0]):
#         verts = np.vstack(([C[j,:] + f * camera_corners[j,0,:]],
#                            [C[j,:] + f * camera_corners[j,1,:]],
#                            [C[j,:] + f * camera_corners[j,3,:]],
#                            [C[j,:] + f * camera_corners[j,2,:]]))
#         ax.add_collection3d(Poly3DCollection([verts], facecolor='blue', alpha=0.25))

#     ax.scatter(*C.T, marker='o', s=40, color='blue')

#     if show_nums:
#         for i in range(X.shape[0]):
#             ax.text(*X[i,:], f'x{i}')
#         for j in range(C.shape[0]):
#             ax.text(*C[j,:], f'c{j}')

def draw_2d_3d_scene(fig, j_slider, X, C, Theta, phi_x, W, H, X_pix=None, image_paths=None,
                     dist_scale=10.0, f=0.2, trajectory_markers=False, show_image_lines=False,
                     axis_off=False, grid_off=False, ret_axis=False):
    if X_pix.shape[0] != X.shape[0]:
        raise ValueError('Expected X_pix.shape[0] == X.shape[0]')
    if X_pix.shape[1] != C.shape[0]:
        raise ValueError('Expected X_pix.shape[1] == C.shape[0]')
    if image_paths is not None:
        if image_paths.size != X_pix.shape[1]:
            raise ValueError('Expected image_paths.size == X_pix.shape[1]')

    assert isinstance(j_slider, Slider)

    ax0 = fig.add_subplot(1, 2, 1, projection='3d')
    assert isinstance(ax0, Axes3D)
    xmin, ymin, zmin = np.vstack((X.min(axis=0), C.min(axis=0))).min(axis=0)
    xmax, ymax, zmax = np.vstack((X.max(axis=0), C.max(axis=0))).max(axis=0)

    ax0.set_xlim(xmin, xmax)
    ax0.set_ylim(zmin, zmax)
    ax0.set_zlim(-ymax, -ymin)
    ax0.set_zticks(np.round(np.array(ax0.get_zticks()), 6))
    ax0.set_zticklabels(-ax0.get_zticks())
    ax0.set_box_aspect([ub - lb for lb, ub in (getattr(ax0, f'get_{a}lim')() for a in 'xyz')])
    ax0.set_xlabel('x')
    ax0.set_ylabel('z')
    ax0.set_zlabel('y')
    if axis_off: ax0.set_axis_off()
    if grid_off: ax0.grid(False)

    N = X.shape[0]

    R = ryxz(Theta[j_slider.val].reshape(1, -1))[0]
    camera_dir = np.array([0.0, 0.0, 1.0]).reshape(1, -1)
    camera_dir_rot = camera_dir @ R.T
    tan_phi_x_2 = np.tan(phi_x / 2)
    r = W / H
    camera_corners = np.array([[tan_phi_x_2, tan_phi_x_2 / r, 1],
                               [- tan_phi_x_2, tan_phi_x_2 / r, 1],
                               [- tan_phi_x_2, - tan_phi_x_2 / r, 1],
                               [tan_phi_x_2, - tan_phi_x_2 / r, 1]])
    camera_corners_rot = camera_corners @ R.T
    # Поворот всей системы вокруг оси x на угол -pi/2
    Ryz = np.array([[1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, -1.0, 0.0]])
    X_show = rotate_array(X, Ryz)
    C_show = rotate_array(C, Ryz)
    camera_dir_rot = rotate_array(camera_dir_rot, Ryz)
    camera_dir_rot = camera_dir_rot.ravel()
    camera_corners_rot = rotate_array(camera_corners_rot, Ryz)

    X_model = transform(X, C[j_slider.val].reshape(1, -1), Theta[j_slider.val].reshape(1, -1), phi_x, W, r,
                        delete_back_points=True, delete_boundary_points=True)[:,0,:]
    I_pix_visible = np.where(~np.isnan(X_pix[:,j_slider.val,0]))[0]
    I_model_visible = np.where(~np.isnan(X_model[:,0]))[0]
    I_model_semivisible = np.array(list(set(I_model_visible) - set(I_pix_visible)), dtype=np.int64)
    I_both_visible = np.array(list(set(I_pix_visible) & set(I_model_visible)), dtype=np.int64)
    # задавать alpha нормально не получается, возникают баги
    X_visible_points = ax0.scatter(*X_show[I_model_visible,:].T, marker='o', s=16, color='red', depthshade=False)
    I_model_not_visible = np.array(list(set(np.arange(N)) - set(I_model_visible)), dtype=np.int64)
    X_not_visible_points = ax0.scatter(*X_show[I_model_not_visible,:].T,  marker='o', s=12, color='red', depthshade=False, alpha=0.5)

    if not trajectory_markers:
        ax0.plot(*C_show.T, color='blue')
    else:
        ax0.plot(*C_show.T, color='blue', marker='o', markersize=2)
    C_current_point = ax0.scatter(*C_show[j_slider.val,:], marker='o', s=16, color='blue', depthshade=False)

    if show_image_lines:
        X_proj = project(X[I_model_visible], C[j_slider.val].reshape(1, -1), Theta[j_slider.val].reshape(1, -1), f)[:,0,:]
        X_proj_show = rotate_array(X_proj, Ryz)

        X_proj_traces = Line3DCollection(np.stack((X_show[I_both_visible], X_proj_show[np.searchsorted(I_model_visible, I_both_visible)]), axis=1),
                                         linewidth=1.0, colors='red', alpha=0.5)
        ax0.add_collection(X_proj_traces)
        X_proj_semitraces = Line3DCollection(np.stack((X_show[I_model_semivisible], X_proj_show[np.searchsorted(I_model_visible, I_model_semivisible)]), axis=1),
                                             linewidth=1.0, colors='red', linestyle='--', alpha=0.5)
        ax0.add_collection(X_proj_semitraces)
        X_proj_points = ax0.scatter(*X_proj_show.T, marker='o',  s=3, color='red', alpha=0.5)

    camera_dir_line, = ax0.plot(*np.vstack((C_show[j_slider.val,:], C_show[j_slider.val,:] + f * camera_dir_rot)).T,
                                linewidth=1.0, linestyle='--', color='blue')

    center_to_corners = Line3DCollection(np.stack((np.tile(C_show[j_slider.val,:], (4, 1)),
                                                   C_show[j_slider.val,:] + f * camera_corners_rot), axis=1),
                                         linewidth=1.0, colors='blue')
    ax0.add_collection(center_to_corners)
    
    corners_surface = Poly3DCollection([C_show[j_slider.val,:] + f * camera_corners_rot], linewidths=1.0, edgecolors='blue', facecolor='blue', alpha=0.25)
    ax0.add_collection3d(corners_surface)

    ax1 = fig.add_subplot(1, 2, 2)
    assert isinstance(ax1, Axes)
    ax1.set_xlim(0, W)
    ax1.set_ylim(H, 0)
    ax1.set_aspect('equal')
    ax1.set_xticks(np.linspace(0, W, 5))
    ax1.set_yticks(np.linspace(H, 0, 5))

    dist = distance(X, C, Theta)
    dist /= dist.max()
    if I_model_visible.size > 0:
        X_model_points = ax1.scatter(*X_model[I_model_visible,:].T, s=dist_scale/dist[I_model_visible,j_slider.val],
                                     marker='o', color='red', alpha=np.full(I_model_visible.size, 0.5)+0.5*np.isin(I_model_visible, I_both_visible), zorder=10)
    else:
        X_model_points = ax1.scatter([], [], marker='o', color='red', zorder=10)

    if X_pix is not None:
        X_pix_center = X_pix[:,j_slider.val,:] + 0.5
        X_pix_points = ax1.scatter(*X_pix_center.T, s=dist_scale, marker='o', edgecolors='red', facecolors='none', zorder=10)
        pix_corners = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])
        X_pix_squares = PolyCollection(X_pix[:,j_slider.val,:][:,np.newaxis,:] + pix_corners[np.newaxis,:,:],
                                       linewidths=1.0, edgecolors='red', facecolor='none')
        ax1.add_collection(X_pix_squares)

        if I_both_visible.size > 0:
            X_lines = LineCollection(np.stack((X_model[I_both_visible,:], X_pix_center[I_both_visible,:]), axis=1),
                                     color='red', linewidth=1.0, alpha=0.5)
        else:
            X_lines = LineCollection([], color='red', linewidth=1.0, alpha=0.5)
        ax1.add_collection(X_lines)

    if image_paths is not None:
        img_show = ax1.imshow(imread(image_paths[j_slider.val]), extent=(0, W, H, 0), alpha=0.5)

    def update(val):
        C_current_point._offsets3d = C_show[j_slider.val,:].reshape(-1, 1)

        X_model = transform(X, C[j_slider.val].reshape(1, -1), Theta[j_slider.val].reshape(1, -1), phi_x, W, r,
                            delete_back_points=True, delete_boundary_points=True)[:,0,:]
        I_pix_visible = np.where(~np.isnan(X_pix[:,j_slider.val,0]))[0]
        I_model_visible = np.where(~np.isnan(X_model[:,0]))[0]
        I_model_semivisible = np.array(list(set(I_model_visible) - set(I_pix_visible)), dtype=np.int64)
        I_both_visible = np.array(list(set(I_pix_visible) & set(I_model_visible)), dtype=np.int64)
        I_model_not_visible = np.array(list(set(np.arange(N)) - set(I_model_visible)), dtype=np.int64)
        X_visible_points._offsets3d = X_show[I_model_visible,:].T
        X_not_visible_points._offsets3d = X_show[I_model_not_visible,:].T

        if show_image_lines:
            X_proj = project(X[I_model_visible], C[j_slider.val].reshape(1, -1), Theta[j_slider.val].reshape(1, -1), f)[:,0,:]
            X_proj_show = rotate_array(X_proj, Ryz)
            X_proj_traces.set_segments(np.stack((X_show[I_both_visible], X_proj_show[np.searchsorted(I_model_visible, I_both_visible)]), axis=1))
            X_proj_semitraces.set_segments(np.stack((X_show[I_model_semivisible], X_proj_show[np.searchsorted(I_model_visible, I_model_semivisible)]), axis=1))
            X_proj_points._offsets3d = X_proj_show.T

        R = ryxz(Theta[j_slider.val].reshape(1, -1))[0]
        camera_dir_rot = camera_dir @ R.T
        camera_dir_rot = rotate_array(camera_dir_rot, Ryz)
        camera_dir_rot = camera_dir_rot.ravel()
        camera_corners_rot = camera_corners @ R.T
        camera_corners_rot = rotate_array(camera_corners_rot, Ryz)

        camera_dir_array = np.vstack((C_show[j_slider.val,:], C_show[j_slider.val,:] + f * camera_dir_rot))
        camera_dir_line.set_data(*camera_dir_array[:,:2].T)
        camera_dir_line.set_3d_properties(camera_dir_array[:,2])

        center_to_corners.set_segments(np.stack((np.tile(C_show[j_slider.val,:], (4, 1)),
                                                 C_show[j_slider.val,:] + f * camera_corners_rot), axis=1))
        corners_surface.set_verts([C_show[j_slider.val,:] + f * camera_corners_rot])

        if I_model_visible.size > 0:
            X_model_points.set_visible(True)
            X_model_points.set_offsets(X_model[I_model_visible,:])
            X_model_points.set_sizes(dist_scale/dist[I_model_visible,j_slider.val])
            X_model_points.set_alpha(np.full(I_model_visible.size, 0.5) + 0.5 * np.isin(I_model_visible, I_both_visible))
        else:
            X_model_points.set_visible(False)

        if X_pix is not None:
            X_pix_center = X_pix[:,j_slider.val,:] + 0.5
            X_pix_points.set_offsets(X_pix_center)
            X_pix_squares.set_verts(X_pix[:,j_slider.val,:][:,np.newaxis,:] + pix_corners[np.newaxis,:,:])
            if I_both_visible.size > 0:
                X_lines.set_visible(True)
                X_lines.set_segments(np.stack((X_model[I_both_visible,:], X_pix_center[I_both_visible,:]), axis=1))
            else:
                X_lines.set_visible(False)
        if image_paths is not None:
            img_show.set_array(imread(image_paths[j_slider.val]))

    j_slider.on_changed(update)

    if ret_axis: return ax0, ax1

def draw_fitting(fig, t_slider, X_array, C_array, Theta_array, phi_x_array, W, H, X_pix,
                 j_ids, image_paths=None, dist_scale=10.0, f=0.2, trajectory_markers=False,
                 axis_off=False, grid_off=False, ret_axis=False):
    if X_pix.shape[0] != X_array.shape[1]:
        raise ValueError('Expected X_pix.shape[0] == X_array.shape[1]')
    if X_pix.shape[1] != C_array.shape[1]:
        raise ValueError('Expected X_pix.shape[1] == C_array.shape[1]')
    if image_paths is not None:
        if image_paths.size != 3:
            raise ValueError('Expected image_paths to have size 3')

    assert isinstance(t_slider, Slider)

    gs = fig.add_gridspec(nrows=3, ncols=2)
    ax0 = fig.add_subplot(gs[:,0], projection='3d')
    assert isinstance(ax0, Axes3D)
    xmin, ymin, zmin = np.vstack((X_array[-1].min(axis=0), C_array[-1].min(axis=0))).min(axis=0)
    xmax, ymax, zmax = np.vstack((X_array[-1].max(axis=0), C_array[-1].max(axis=0))).max(axis=0)

    ax0.set_xlim(xmin, xmax)
    ax0.set_ylim(zmin, zmax)
    ax0.set_zlim(-ymax, -ymin)
    ax0.set_zticks(np.round(np.array(ax0.get_zticks()), 6))
    ax0.set_zticklabels(-ax0.get_zticks())
    ax0.set_box_aspect([ub - lb for lb, ub in (getattr(ax0, f'get_{a}lim')() for a in 'xyz')])
    ax0.set_xlabel('x')
    ax0.set_ylabel('z')
    ax0.set_zlabel('y')
    if axis_off: ax0.set_axis_off()
    if grid_off: ax0.grid(False)

    R = ryxz(Theta_array[t_slider.val,j_ids])
    camera_dir = np.array([0.0, 0.0, 1.0])
    camera_dir_rot = np.einsum('jkl,l->jk', R, camera_dir)
    tan_phi_x_2 = np.tan(phi_x_array[t_slider.val] / 2)
    r = W / H
    camera_corners = np.array([[tan_phi_x_2, tan_phi_x_2 / r, 1],
                               [- tan_phi_x_2, tan_phi_x_2 / r, 1],
                               [- tan_phi_x_2, - tan_phi_x_2 / r, 1],
                               [tan_phi_x_2, - tan_phi_x_2 / r, 1]])
    camera_corners_rot = np.einsum('jkl,nl->jnk', R, camera_corners)
    # Поворот всей системы вокруг оси x на угол -pi/2
    Ryz = np.array([[1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, -1.0, 0.0]])
    X_show = rotate_array(X_array[t_slider.val], Ryz)
    C_show = rotate_array(C_array[t_slider.val], Ryz)
    C_points_show = C_show[j_ids]
    camera_dir_rot = rotate_array(camera_dir_rot, Ryz)
    camera_corners_rot = rotate_array(camera_corners_rot, Ryz)

    X_points = ax0.scatter(*X_show.T, marker='o', s=16, color='red', depthshade=False)

    if not trajectory_markers:
        C_points, = ax0.plot(*C_show.T, color='blue')
    else:
        C_points, = ax0.plot(*C_show.T, color='blue', marker='o', markersize=2)

    camera_dir_line = Line3DCollection(np.stack((C_points_show,
                                                 C_points_show + f * camera_dir_rot), axis=1),
                                       linewidth=1.0, linestyle='--', colors='blue')
    ax0.add_collection(camera_dir_line)

    center_to_corners = Line3DCollection(np.stack((np.repeat(C_points_show, 4, axis=0).reshape(-1, 4, 3),
                                                   np.repeat(C_points_show, 4, axis=0).reshape(-1, 4, 3) + f * camera_corners_rot), axis=2).reshape(-1, 2, 3),
                                        linewidth=1.0, colors='blue')
    ax0.add_collection(center_to_corners)

    corners_surface = Poly3DCollection(np.repeat(C_points_show, 4, axis=0).reshape(-1, 4, 3) + f * camera_corners_rot, linewidths=1.0, edgecolors='blue', facecolor='blue', alpha=0.25)
    ax0.add_collection3d(corners_surface)

    axs = [fig.add_subplot(gs[i,1]) for i in range(3)]
    for ax in axs:
        assert isinstance(ax, Axes)
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.set_aspect('equal')
        ax.set_xticks(np.linspace(0, W, 5))
        ax.set_yticks(np.linspace(H, 0, 5))

    X_pix_center = X_pix[:,j_ids,:] + 0.5
    X_model = transform(X_array[t_slider.val], C_array[t_slider.val,j_ids], Theta_array[t_slider.val,j_ids], phi_x_array[t_slider.val],
                        W, r, delete_back_points=True, delete_boundary_points=True)
    I_pix_visible = np.where(~np.isnan(X_pix[:,j_ids,0]) == True)[0]
    I_model_visible = np.where(~np.isnan(X_model[:,:,0]) == True)[0]
    I_both_visible = np.array(list(set(I_pix_visible) & set(I_model_visible)))

    dist = distance(X_array[t_slider.val], C_array[t_slider.val], Theta_array[t_slider.val])
    dist /= dist.max()
    X_pix_points, X_model_points, X_lines = 3 * [None], 3 * [None], 3 * [None]
    for j, ax in enumerate(axs):
        assert isinstance(ax, Axes)
        if image_paths is not None:
            ax.imshow(imread(image_paths[j]), extent=(0, W, H, 0), alpha=0.5)
        X_pix_points[j] = ax.scatter(*X_pix_center[:,j,:].T, s=dist_scale, marker='o', edgecolors='red', facecolors='none', zorder=10)
        X_model_points[j] = ax.scatter(*X_model[I_model_visible,j,:].T, s=dist_scale/dist[I_model_visible,j], marker='o', color='red', zorder=10)
        X_lines[j] = LineCollection(np.stack((X_model[I_both_visible,j,:], X_pix_center[I_both_visible,j,:]), axis=1),
                                    color='red', linewidth=1.0, alpha=0.5)
        ax.add_collection(X_lines[j])

    def update(val):
        R = ryxz(Theta_array[t_slider.val,j_ids])
        camera_dir_rot = np.einsum('jkl,l->jk', R, camera_dir)
        tan_phi_x_2 = np.tan(phi_x_array[t_slider.val] / 2)
        camera_corners = np.array([[tan_phi_x_2, tan_phi_x_2 / r, 1],
                                   [- tan_phi_x_2, tan_phi_x_2 / r, 1],
                                   [- tan_phi_x_2, - tan_phi_x_2 / r, 1],
                                   [tan_phi_x_2, - tan_phi_x_2 / r, 1]])
        camera_corners_rot = np.einsum('jkl,nl->jnk', R, camera_corners)
        # Поворот всей системы вокруг оси x на угол -pi/2
        X_show = rotate_array(X_array[t_slider.val], Ryz)
        C_show = rotate_array(C_array[t_slider.val], Ryz)
        C_points_show = C_show[j_ids]
        camera_dir_rot = rotate_array(camera_dir_rot, Ryz)
        camera_corners_rot = rotate_array(camera_corners_rot, Ryz)

        X_points._offsets3d = X_show.T
        C_points.set_data(*C_show[:,:2].T)
        C_points.set_3d_properties(C_show[:,2])

        camera_dir_line.set_segments(np.stack((C_points_show,
                                               C_points_show + f * camera_dir_rot), axis=1))
        center_to_corners.set_segments(np.stack((np.repeat(C_points_show, 4, axis=0).reshape(-1, 4, 3),
                                                 np.repeat(C_points_show, 4, axis=0).reshape(-1, 4, 3) + f * camera_corners_rot), axis=2).reshape(-1, 2, 3))
        corners_surface.set_verts(np.repeat(C_points_show, 4, axis=0).reshape(-1, 4, 3) + f * camera_corners_rot)

        X_model = transform(X_array[t_slider.val], C_array[t_slider.val,j_ids], Theta_array[t_slider.val,j_ids], phi_x_array[t_slider.val],
                                        W, r, delete_back_points=True, delete_boundary_points=True)
        I_model_visible = np.where(~np.isnan(X_model[:,:,0]) == True)[0]
        I_both_visible = np.array(list(set(I_pix_visible) & set(I_model_visible)))

        dist = distance(X_array[t_slider.val], C_array[t_slider.val], Theta_array[t_slider.val])
        dist /= dist.max()
        for j, ax in enumerate(axs):
            X_model_points[j].set_offsets(X_model[I_model_visible,j,:])
            X_model_points[j].set_sizes(dist_scale/dist[I_model_visible,j])
            X_lines[j].set_segments(np.stack((X_model[I_both_visible,j,:], X_pix_center[I_both_visible,j,:]), axis=1))

    t_slider.on_changed(update)

    if ret_axis: return ax0, axs

def get_Theta(R):
    if (np.abs(np.einsum('jkl,jkn->jln', R, R) - np.eye(3)) > 1e-9).any():
        raise ValueError('Expected R[i] to be normal for all i (R[i] @ R[i].T must be approximately equal to eye(3))')
    if (np.abs(np.linalg.det(R) - 1.0) > 1e-9).any():
        raise ValueError('Expected R[i] to have determinant approximately 1.0 for all i')

    C_x = np.sqrt(R[:,1,0]**2 + R[:,1,1]**2)
    S_x = - R[:,1,2]
    C_y, S_y = R[:,2,2] / C_x, R[:,0,2] / C_x
    C_z, S_z = R[:,1,1] / C_x, R[:,1,0] / C_x
    Theta = np.column_stack((np.arctan2(S_x, C_x), np.arctan2(S_y, C_y), np.arctan2(S_z, C_z)))
    return Theta

def get_Theta_orth(R_array):
    Theta = np.zeros(R_array.shape[:2])
    for j, R in enumerate(R_array):
        U, d, Vh = np.linalg.svd(R)
        R_rot = U @ Vh
        Theta[j] = get_Theta(R_rot.reshape(1, 3, 3))[0]
    return Theta

def get_R(x, y):
    v = np.cross(x, y)
    v_norm = np.linalg.norm(v)
    if v_norm > 0.0:
        xy = np.linalg.norm(x) * np.linalg.norm(y)
        n = v / v_norm
        sin_phi = v_norm / xy
        cos_phi = np.dot(x, y) / xy

        R = cos_phi * np.eye(3)
        R += (1 - cos_phi) * (n.reshape(-1, 1) @ n.reshape(1, -1))
        R += sin_phi * np.array([[0.0, -n[2], n[1]],
                                 [n[2], 0.0, -n[0]],
                                 [-n[1], n[0], 0.0]])
    else:
        R = np.eye(3)
    return R

def fundamental_matrix(X_pix, make_rank_2=True, ret_A=False, normalize=False, W=None, H=None):
    if X_pix.shape[0] < 8:
        raise ValueError('Expected X_pix to have shape[0]>=8')
    if X_pix.shape[1] != 2:
        raise ValueError('Expected X_pix to have shape[1]=2')
    if normalize and (W is None or H is None):
        raise ValueError('With normalize=True W and H have to be passed')

    if not normalize:
        X_pix_center = X_pix + 0.5
    else:
        # X_pix_center = (X_pix + 0.5) / np.array([W, H]) - 0.5
        # r = W / H
        # X_pix_center = (X_pix + 0.5) / W - np.array([0.5, 0.5 / r])
        X_pix_center = (X_pix + 0.5 - 0.5 * np.array([W, H])) / max(W, H)
    X_hom = np.concatenate((X_pix_center, np.ones((*X_pix_center.shape[:-1], 1))), axis=-1)

    A = np.column_stack((X_hom[:,1,0] * X_hom[:,0,0],
                         X_hom[:,1,0] * X_hom[:,0,1],
                         X_hom[:,1,0] * X_hom[:,0,2],
                         X_hom[:,1,1] * X_hom[:,0,0],
                         X_hom[:,1,1] * X_hom[:,0,1],
                         X_hom[:,1,1] * X_hom[:,0,2],
                         X_hom[:,1,2] * X_hom[:,0,0],
                         X_hom[:,1,2] * X_hom[:,0,1],
                         X_hom[:,1,2] * X_hom[:,0,2]))
    U, d, Vh = np.linalg.svd(A)
    f = Vh[-1]
    # f = eigsh(A.T @ A, k=1, which='SM')[1]    # вроде иногда должно считаться быстрее, чем с np.linalg.svd(A)
    F = f.reshape(3, 3)

    if make_rank_2 and np.linalg.matrix_rank(F) != 2:
        U, d, Vh = np.linalg.svd(F)
        d[-1] = 0.0
        F = U @ np.diag(d) @ Vh

    if not ret_A: return F
    else: return F, A

def get_scene_from_F(X_pix, F, W, H, phi_x, ret_status=False, normalized=False):
    if X_pix.shape[1] != 2:
        raise ValueError('Expected X_pix to have shape[1]=2')
    
    if not normalized:
        X_pix_center = X_pix + 0.5
    else:
        # X_pix_center = (X_pix + 0.5) / np.array([W, H]) - 0.5
        # r = W / H
        # X_pix_center = (X_pix + 0.5) / W - np.array([0.5, 0.5 / r])
        X_pix_center = (X_pix + 0.5 - 0.5 * np.array([W, H])) / max(W, H)

    if not normalized:
        f_x = f_y = W / 2 / np.tan(phi_x / 2)
        p_x, p_y = W / 2, H / 2
    else:
        # r = W / H
        # f_x, f_y = np.array([1, r]) / 2 / np.tan(phi_x / 2)
        # f_x = f_y = 1 / 2 / np.tan(phi_x / 2)
        f_x = f_y = W / max(W, H) / 2 / np.tan(phi_x / 2)
        p_x = p_y = 0.0
    K_ = np.array([[f_x, 0.0, p_x],
                   [0.0, f_y, p_y],
                   [0.0, 0.0, 1.0]])

    P1 = np.column_stack((K_, np.zeros(3)))
    E = K_.T @ F @ K_
    U, d, Vh = np.linalg.svd(E)

    W_ = np.array([[0.0, -1.0, 0.0],
                   [1.0, 0.0, 0.0],
                   [0.0, 0.0, 1.0]])
    
    status = False
    for k in range(4):
        if k == 0: 
            R = U @ W_ @ Vh
            t = U[:,-1]
        elif k == 1:
            R = U @ W_ @ Vh
            t = - U[:,-1]
        elif k == 2:
            R = U @ W_.T @ Vh
            t = U[:,-1]
        elif k == 3:
            R = U @ W_.T @ Vh
            t = - U[:,-1]
        
        if np.linalg.det(R) < 0: R = - R

        P2 = K_ @ np.column_stack((R, t))

        X1, Y1 = X_pix_center[:,0,:].T
        X2, Y2 = X_pix_center[:,1,:].T
        A_array = np.stack((X1[:,np.newaxis] * P1[2,:] - P1[0,:],
                            Y1[:,np.newaxis] * P1[2,:] - P1[1,:],
                            X2[:,np.newaxis] * P2[2,:] - P2[0,:],
                            Y2[:,np.newaxis] * P2[2,:] - P2[1,:]), axis=1)
        A_array_mat, A_array_vec = A_array[:,:,:-1], A_array[:,:,-1]
        lstsq_mat = np.einsum('ilj,ilk->ijk', A_array_mat, A_array_mat)
        lstsq_vec = np.einsum('ilj,il->ij', A_array_mat, A_array_vec)
        X = - np.linalg.solve(lstsq_mat, lstsq_vec)
        
        C = np.vstack((np.zeros(3), - R.T @ t))

        theta = get_Theta(R.T.reshape(1, 3, 3))[0]
        Theta = np.vstack((np.zeros(3), theta))

        if (distance(X, C, Theta) > 0).all():
            status = True
            break
    
    if not ret_status: return X, C, Theta
    else: return X, C, Theta, status

def find_phi_x(X_pix, W, H, attempts=5, delete_outliers=True, normalize=False):
    K = X_pix.shape[1]
    X_visible = ~np.isnan(X_pix)[:,:,0]
    r = W / H
    
    std_matrix, i_size_matrix = np.full((K, K), np.nan), np.zeros((K, K), dtype=np.int32)
    for i in range(K):
        for j in range(i + 1, K):
            std_matrix[i,j] = np.linalg.norm(np.nanstd(X_pix[:,i,:] - X_pix[:,j,:], axis=0))
            i_size_matrix[i,j] = np.where(X_visible[:,i] * X_visible[:,j])[0].size

    min_std = np.nanpercentile(std_matrix, 25)

    std_matrix_ids = np.where(std_matrix >= min_std)
    matrix_ids = np.argsort(std_matrix[std_matrix_ids] / 10.0 + i_size_matrix[std_matrix_ids])[::-1][:attempts]
    i_ids, j_ids = std_matrix_ids[0][matrix_ids], std_matrix_ids[1][matrix_ids]
    
    phi_x_opt_array = np.zeros(i_ids.size)
    for k, (i, j) in enumerate(zip(i_ids, j_ids)):
        print(f'{k = }')
        i_subset = np.where(X_visible[:,i] * X_visible[:,j])[0]
        ij_subset = np.ix_(i_subset, [i, j])
        F = fundamental_matrix(X_pix[ij_subset], normalize=normalize, W=W, H=H)

        phi_x_array, h = np.linspace(0, np.pi, 53, retstep=True)
        phi_x_array = phi_x_array[1:-1]

        E_diff = np.inf
        while E_diff > 1e-5:
            E_array = np.full_like(phi_x_array, np.inf)
            X_test, C_test, Theta_test, phi_x_test = [], [], [], []
            for phi_x in phi_x_array[(phi_x_array > 0.0) & (phi_x_array < np.pi)]:
                X_, C_, Theta_, success = get_scene_from_F(X_pix[ij_subset], F, W, H, phi_x, ret_status=True, normalized=normalize)
                if success:
                    X_test.append(X_)
                    C_test.append(C_[1,:])
                    Theta_test.append(Theta_[1,:])
                    phi_x_test.append(phi_x)
            if not X_test: break
            X_test, C_test, Theta_test, phi_x_test = np.array(X_test), np.array(C_test), np.array(Theta_test), np.array(phi_x_test)
            R_test = ryxz(Theta_test)
            X_test_rot = np.zeros((*X_test.shape[:2], 2, 3))
            X_test_rot[:,:,0,:] = X_test
            X_test_rot[:,:,1,:] = np.einsum('nlk,nil->nik', R_test, X_test - C_test[:,np.newaxis,:])
            X_model_test = W / 2 * (X_test_rot[:,:,:,:2] / X_test_rot[:,:,:,2][:,:,:,np.newaxis] / np.tan(phi_x_test / 2)[:,np.newaxis,np.newaxis,np.newaxis] \
                                    + np.array([1.0, 1.0 / r]))
            X_pix_center = X_pix[ij_subset] + 0.5
            X_diff = X_model_test - X_pix_center
            E_array = np.sum(X_diff**2, axis=(1, 2, 3)) / (4 * i_subset.size)   # 2 * N * K = 2 * i_subset.size * 2

            phi_x_opt = phi_x_array[(phi_x_array > 0.0) & (phi_x_array < np.pi)][np.argmin(E_array)]
            print(f'{phi_x_opt = }')
            phi_x_array, h = np.linspace(phi_x_opt - h, phi_x_opt + h, 51, retstep=True)
            E_diff = np.nan_to_num(E_array.max() - E_array.min())

        phi_x_opt_array[k] = phi_x_opt

    if delete_outliers:
        Q1, Q3 = np.percentile(phi_x_opt_array, [25, 75])
        IQR = Q3 - Q1
        phi_x_opt_array = phi_x_opt_array[np.where((phi_x_opt_array >= Q1 - 1.5 * IQR) & (phi_x_opt_array <= Q3 + 1.5 * IQR))[0]]

    return np.mean(phi_x_opt_array)

def get_pair_subscenes(X_pix, phi_x, W, H, min_std, max_size=50, normalize=False):
    K = X_pix.shape[1]
    X_visible = ~np.isnan(X_pix)[:,:,0]

    j_subset_list = []
    j_subset_all = np.zeros(0)
    X_pairs_list = []
    C_pairs_list = []
    Theta_pairs_list = []

    iter = 0
    finish = False
    while j_subset_all.size < max_size:
        X_pairs = []
        C_pairs = []
        Theta_pairs = []

        if iter >= 2:            
            if iter % 2 == 0:
                id_1, id_2 = 0, 1
                j_1, j_2 = j_subset_all[id_1], j_subset_all[id_2]
                while j_2 - j_1 < 2:
                    id_1 += 1
                    id_2 += 1
                    if id_2 > j_subset_all.size - 1:
                        finish = True
                    j_1, j_2 = j_subset_all[id_1], j_subset_all[id_2]
                j = int(np.floor((j_1 + j_2) / 2))
            else:
                id_1, id_2 = j_subset_all.size - 1, j_subset_all.size - 2
                j_1, j_2 = j_subset_all[id_1], j_subset_all[id_2]
                while j_1 - j_2 < 2:
                    id_1 -= 1
                    id_2 -= 1
                    if id_1 < 0:
                        finish = True
                    j_1, j_2 = j_subset_all[id_1], j_subset_all[id_2]
                j = int(np.ceil((j_1 + j_2) / 2))
            if finish: break
        else:
            if iter == 0: j = 0
            else: j = K - 1
        j_ = []
        done = False
        success = True
        while True:
            std = 0.0
            if success:
                j_.append(j)
                step = 0
            while std < min_std or np.isnan(std):
                if iter % 2 == 0: step += 1
                else: step -= 1
                if (iter % 2 == 0 and j + step >= K - 1) or (iter % 2 == 1 and j + step <= 0):
                    done = True
                    break
                if j + step in j_subset_all:
                    continue
                std = np.linalg.norm(np.nanstd(X_pix[:,j,:] - X_pix[:,j+step,:], axis=0))
            if done: break
            
            print(f'{j = }, {step = }, {std = }')
            i_subset = np.where(X_visible[:,j] * X_visible[:,j+step])[0]
            print(f'{i_subset.size = }')
            if i_subset.size < 10:
                success = False
                continue
            if iter % 2 == 0: ij_subset = np.ix_(i_subset, [j, j+step])
            else: ij_subset = np.ix_(i_subset, [j+step, j])
            F = fundamental_matrix(X_pix[ij_subset], normalize=normalize, W=W, H=H)
            X_, C, Theta, success = get_scene_from_F(X_pix[ij_subset], F, W, H, phi_x, ret_status=True, normalized=normalize)
            
            print(f'{success = }')
            if not success: continue

            X = np.full((X_pix.shape[0], 3), np.nan)
            X[i_subset] = X_

            X_pairs.append(X)
            C_pairs.append(C)
            Theta_pairs.append(Theta)

            j += step

        j_ = np.array(j_)
        X_pairs = np.array(X_pairs)
        C_pairs = np.array(C_pairs)
        Theta_pairs = np.array(Theta_pairs)
        if iter % 2 == 1:
            j_ = j_[::-1]
            X_pairs = X_pairs[::-1]
            C_pairs = C_pairs[::-1]
            Theta_pairs = Theta_pairs[::-1]

        j_subset_list.append(j_)
        j_subset_all = np.sort(np.concatenate([j_subset for j_subset in j_subset_list]))
        X_pairs_list.append(X_pairs)
        C_pairs_list.append(C_pairs)
        Theta_pairs_list.append(Theta_pairs)
        iter += 1

    return j_subset_list, X_pairs_list, C_pairs_list, Theta_pairs_list

def get_C_Theta_weighted(C_pairs, Theta_pairs, errors, j_pairs_ids_array):
    K = j_pairs_ids_array.max() + 1
    weights_double = np.full((K, 2), np.nan)
    C_double, R_double = np.full((K, 2, 3), np.nan), np.full((K, 2, 3, 3), np.nan)
    
    weights_double[j_pairs_ids_array[:,0],0] = weights_double[j_pairs_ids_array[:,1],1] = 1 / errors
    weights_double /= np.nansum(weights_double, axis=1, keepdims=True)

    C_double[j_pairs_ids_array[:,0],0,:] = C_pairs[:,0,:]
    C_double[j_pairs_ids_array[:,1],1,:] = C_pairs[:,1,:]

    R_double[j_pairs_ids_array[:,0],0,:,:] = ryxz(Theta_pairs[:,0,:])
    R_double[j_pairs_ids_array[:,1],1,:,:] = ryxz(Theta_pairs[:,1,:])

    C_weighted = np.nansum(C_double * weights_double[:,:,np.newaxis], axis=1)
    R_weighted = np.nansum(R_double * weights_double[:,:,np.newaxis,np.newaxis], axis=1)
    Theta_weighted = get_Theta(R_weighted)
    
    return C_weighted, Theta_weighted

def moving_average(X, h):
    X_new = np.zeros_like(X)
    for i in range(X.shape[0]):
        h_ = min(h, i, X.shape[0] - i - 1)
        X_new[i] = np.mean(X[i-h_:i+h_+1], axis=0)
    return X_new

def approximate(j_small, j_big, X_small, extrapolate=False):
    # предполагается, что j_big[0] <= j_small[0] и j_big[-1] >= j_small[-1]
    if X_small.ndim == 1: X_big = np.full(j_big.size, np.nan)
    else: X_big = np.full((j_big.size, *X_small.shape[1:]), np.nan)

    if not extrapolate:
        X_big[:j_small[0]+1] = X_small[0]
        X_big[j_small[-1]:] = X_small[-1]
    else:
        for l in range(j_small[0] + 1):
            X_big[l] = X_small[0] + (j_big[l] - j_small[0]) / (j_small[1] - j_small[0]) * (X_small[1] - X_small[0])
        for l in range(j_small[-1], j_big[-1] + 1):
            X_big[l] = X_small[-1] + (j_big[l] - j_small[-1]) / (j_small[-1] - j_small[-2]) * (X_small[-1] - X_small[-2])
    
    k, l = 0, j_small[0] + 1
    while k < j_small.size - 1:
        if j_big[l] >= j_small[k] and j_big[l] < j_small[k+1]:
            X_big[l] = X_small[k] + (j_big[l] - j_small[k]) / (j_small[k+1] - j_small[k]) * (X_small[k+1] - X_small[k])
            l += 1
        elif j_big[l] >= j_small[k+1]:
            k += 1

    return X_big