NAME = "cameramodel"

import numpy as np
import scipy.ndimage as ndimage
import matplotlib.patches as patches
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from matplotlib.widgets import Slider

def error(X_pix, X_model):
    N, K = X_pix.shape[:2]
    X_pix_center = X_pix + 0.5
    # X_diff = X_model - X_pix_center
    X_diff = np.nan_to_num(X_model - X_pix_center)
    return np.dot(X_diff.ravel(), X_diff.ravel()) / (2 * N * K) 

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

def transform(X, C, Theta, phi_x, W, r):
    X_rot = np.einsum('jlk,ijl->ijk', ryxz(Theta), X[:,np.newaxis,:] - C[np.newaxis,:,:])
    X_rot[np.where(X_rot[:,:,2] <= 0)] = np.nan
    X_model = W / 2 * (X_rot[:,:,:2] / X_rot[:,:,2][:,:,np.newaxis] / np.tan(phi_x / 2) + np.array([1.0, 1.0 / r])[np.newaxis,np.newaxis,:])
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

# def sgd(x, grad_x, lr, ret_delta=False):
    # delta_x = - lr * grad_x
    # # if ~ret_delta: return x + delta_x
    # # else: return x + delta_x, delta_x
    # if ret_delta: return x + delta_x, delta_x
    # else: return x + delta_x
def sgd(x, grad_x, lr):
    return x - lr * grad_x

# def gd_adam(x, grad_x, lr, s, r, t, rho_1, rho_2, ret_delta=False):
def gd_adam(x, grad_x, lr, s, r, t, rho_1, rho_2):
    s_new = rho_1 * s + (1 - rho_1) * grad_x
    r_new = rho_2 * r + (1 - rho_2) * grad_x**2
    s_corr = s_new / (1 - rho_1**t)
    r_corr = r_new / (1 - rho_2**t)
    # delta_x = - lr * s_corr / (1e-8 + np.sqrt(r_corr))
    # # if ~ret_delta: return x + delta_x, s_new, r_new
    # # else: return x + delta_x, s_new, r_new, delta_x
    # if ret_delta: return x + delta_x, s_new, r_new, delta_x
    # else: return x + delta_x, s_new, r_new
    return x - lr * s_corr / (1e-8 + np.sqrt(r_corr)), s_new, r_new

def fit(X_pix, W, r, lr, max_iters, X_0, C_0, Theta_0, phi_x_0,
        X_mask=True, phi_x_mask=True, main_indexes=[0, -1],
        optimizer='SGD', patience=2000, factor=2, print_step=1000, ret_arrays=False):
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
        beta = W / 2 / tan_phi_x_2

    # if eps > 0.0: delta_X, delta_C = np.zeros_like(X), np.zeros_like(C)
    # Delta_X, Delta_C = np.zeros((patience, *X.shape)), np.zeros((patience, *C.shape))     # замедляет процесс аж в 1.5 раза

    s_X = r_X = np.zeros_like(X)
    s_C = r_C = np.zeros_like(C)
    s_Theta = r_Theta = np.zeros_like(Theta)
    s_phi_x = r_phi_x = 0
    
    iters = 0
    E = np.zeros(max_iters)
    E_min = np.inf

    R_scale = np.linalg.norm(C[main_indexes[0],:] - C[main_indexes[1],:])

    # V = np.zeros(3 * N + 6 * K + 1)
    # V[: 3*N] = X.ravel()
    # V[3*N : 3*N+3*K] = C.ravel()
    # V[3*N+3*K : -1] = Theta.ravel()
    # V[-1] = phi_x
    # Delta_V, s_V, r_V = np.zeros_like(V), np.zeros_like(V), np.zeros_like(V)

    if ret_arrays:
        X_list, C_list, Theta_list,phi_x_list = [], [], [], []

    for iters in range(max_iters):
        if ret_arrays and iters % 100 == 0:
            X_list.append(X)
            C_list.append(C)
            Theta_list.append(Theta)
            phi_x_list.append(phi_x)

        X_tr = X[:,np.newaxis,:] - C[np.newaxis,:,:]

        Rx, Ry, Rz = rx(Theta[:,0]), ry(Theta[:,1]), rz(Theta[:,2])
        R = Ry @ Rx @ Rz
        X_rot = np.einsum('jlk,ijl->ijk', R, X_tr)

        if phi_x_mask:
            tan_phi_x_2 = np.tan(phi_x / 2)
            beta = W / 2 / tan_phi_x_2
        X_model = W / 2 * (X_rot[:,:,:2] / X_rot[:,:,2][:,:,np.newaxis] / tan_phi_x_2 + np.array([1.0, 1.0 / r])[np.newaxis,np.newaxis,:])      # должно считаться быстрее, чем через задание X_model[:,:,0] и X_model[:,:,1] по отдельности

        # X_diff = X_model - X_pix_center
        X_diff = np.nan_to_num(X_model - X_pix_center)
        E[iters] = np.dot(X_diff.ravel(), X_diff.ravel()) / (2 * N * K)     # должно считаться в >2 раза быстрее, чем np.sum(X_diff**2) / (2 * N * K)
        if iters % print_step == 0: print(f"{iters} : {E[iters]}")
        # if iters % print_step == 0: print(f"{iters} : {E[iters]}, {np.max(np.abs(np.sum(Delta_X, axis=0))) = }, {np.max(np.abs(np.sum(Delta_C, axis=0))) = }")
        # if iters % print_step == 0: print(f"{iters} : {E[iters]}, {np.max(np.abs(delta_X)) = }, {np.max(np.abs(delta_C)) = }")
        if iters >= 1:
            if E[iters] < E_min:
                E_min = E[iters]
                X_release, C_release, Theta_release, phi_x_release = X, C, Theta, phi_x
            # if iters < patience:
            #     Delta_X[iters,:,:] = delta_X
            #     Delta_C[iters,:,:] = delta_C
            if iters >= patience:
            # else:
                # Delta_X = np.concatenate((Delta_X[1:,:,:], delta_X[np.newaxis,:,:]), axis=0)
                # Delta_C = np.concatenate((Delta_C[1:,:,:], delta_C[np.newaxis,:,:]), axis=0)
                if E[iters] >= E[iters - patience]:
                    print(f"{iters} : Decrease LR to {lr / factor}")
                    lr /= factor
                    if lr == 0.0 or E[iters] - E[iters - patience] == 0.0: break
            # if eps > 0.0 and (np.abs(delta_X) < eps).all() and (np.abs(delta_C) < eps).all():
            #     print('Almost zero changes')
            #     break
            # if optimizer == 'SGD' and E[iters] < 1.0: print(f"{iters} : Turn to Adam"); optimizer = 'Adam'
        if optimizer == 'SGD' and E[iters] < 1.0:
            print(f"{iters} : Turn to Adam")
            optimizer = 'Adam'

        D['R', 'Theta'][:,:,:,0] = Ry @ d_rx(Theta[:,0]) @ Rz
        D['R', 'Theta'][:,:,:,1] = d_ry(Theta[:,1]) @ Rx @ Rz
        D['R', 'Theta'][:,:,:,2] = Ry @ Rx @ d_rz(Theta[:,2])
        D['X_rot', 'Theta'] = np.einsum('jnkl,ijn->ijkl', D['R', 'Theta'], X_tr)

        D['X_model', 'X_rot'][:,:,0,0] = D['X_model', 'X_rot'][:,:,1,1] = beta / X_rot[:,:,2]
        # D['X_model', 'X_rot'][:,:,0,1] = D['X_model', 'X_rot'][:,:,1,0] = 0
        D['X_model', 'X_rot'][:,:,:,2] = - beta / (X_rot[:,:,2]**2)[:,:,np.newaxis] * X_rot[:,:,:2]

        D['X_model', 'Theta'] = np.einsum('ijkn,ijnl->ijkl', D['X_model', 'X_rot'], D['X_rot', 'Theta'])

        D['X_model', 'X'] = np.einsum('ijkn,jln->ijkl', D['X_model', 'X_rot'], R)

        D['E', 'X_model'] = X_diff / (N * K)

        if X_mask:
            D['E', 'X'] = np.einsum('ijl,ijlk->ik', D['E', 'X_model'], D['X_model', 'X'])
        # if (X_mask == 'default').all():
        #     D['E', 'X'][:2,:] = 0
        #     D['E', 'X'][2,1] = 0
        # elif (X_mask == 'zDot').all():
        #     D['E', 'X'][:2,:] = 0
        #     D['E', 'X'][2,2] = 0
        # else:
        #     D['E', 'X'] = D['E', 'X'] * X_mask

        # if C_mask == 'default': D['E', 'C'] = - np.einsum('ijl,ijlk->jk', D['E', 'X_model'], D['X_model', 'X'])
        # elif C_mask != 'none': D['E', 'C'] = D['E', 'C'] * C_mask
        # D['E', 'C'] = - np.einsum('ijl,ijlk->jk', D['E', 'X_model'], D['X_model', 'X']) * C_mask
        D['E', 'C'] = - np.einsum('ijl,ijlk->jk', D['E', 'X_model'], D['X_model', 'X'])
        # D['E', 'C'][0,:] = D['E', 'C'][-1,:] = 0.0
        D['E', 'C'][main_indexes[0],:] = 0.0

        # if Theta_mask == 'default': D['E', 'Theta'] = np.einsum('ijl,ijlk->jk', D['E', 'X_model'], D['X_model', 'Theta'])
        # elif Theta_mask != 'none': D['E', 'Theta'] = D['E', 'Theta'] * Theta_mask
        # D['E', 'Theta'] = np.einsum('ijl,ijlk->jk', D['E', 'X_model'], D['X_model', 'Theta']) * Theta_mask
        D['E', 'Theta'] = np.einsum('ijl,ijlk->jk', D['E', 'X_model'], D['X_model', 'Theta'])
        D['E', 'Theta'][main_indexes[0],:] = 0.0

        if phi_x_mask:
            D['X_model', 'phi_x'] = - beta / X_rot[:,:,2][:,:,np.newaxis] * X_rot[:,:,:2] / np.sin(phi_x)
            D['E', 'phi_x'] = np.dot(D['E', 'X_model'].ravel(), D['X_model', 'phi_x'].ravel())      # должно считаться в >2 раза быстрее, чем np.sum(D['E', 'X_model'] * D['X_model', 'alpha_x'])

        # # с таким подходом возможно иногда работает побыстрее
        # Delta_V[: 3*N] = D['E', 'X'].ravel()
        # Delta_V[3*N : 3*N+3*K] = D['E', 'C'].ravel()
        # Delta_V[3*N+3*K : -1] = D['E', 'Theta'].ravel()
        # Delta_V[-1] = D['E', 'phi_x']

        if optimizer == 'SGD':
            if X_mask:
                # if eps > 0.0: X, delta_X = sgd(X, D['E', 'X'], lr, ret_delta=True)
                # else: X = sgd(X, D['E', 'X'], lr, ret_delta=False)
                X = sgd(X, D['E', 'X'], lr)
            # if C_mask != 'none': C = sgd(C, D['E', 'C'], lr)
            # if eps > 0.0: C, delta_C = sgd(C, D['E', 'C'], lr, ret_delta=True)
            # else: C = sgd(C, D['E', 'C'], lr, ret_delta=False)
            C = sgd(C, D['E', 'C'], lr)
            # if Theta_mask != 'none': Theta = sgd(Theta, D['E', 'Theta'], lr)
            Theta = sgd(Theta, D['E', 'Theta'], lr)
            if phi_x_mask: phi_x = sgd(phi_x, D['E', 'phi_x'], lr)
            # V = sgd(V, Delta_V, lr)
        elif optimizer == 'Adam':
            if X_mask:
                # if eps > 0.0: X, s_X, r_X, delta_X = gd_adam(X, D['E', 'X'], lr, s_X, r_X, t=iters+1, rho_1=0.9, rho_2=0.999, ret_delta=True)
                # else: X, s_X, r_X = gd_adam(X, D['E', 'X'], lr, s_X, r_X, t=iters+1, rho_1=0.9, rho_2=0.999, ret_delta=False)
                X, s_X, r_X = gd_adam(X, D['E', 'X'], lr, s_X, r_X, t=iters+1, rho_1=0.9, rho_2=0.999)
            # if C_mask != 'none': C, s_C, r_C = gd_adam(C, D['E', 'C'], lr, s_C, r_C, t=iters+1, rho_1=0.9, rho_2=0.999)
            # if eps > 0.0: C, s_C, r_C, delta_C = gd_adam(C, D['E', 'C'], lr, s_C, r_C, t=iters+1, rho_1=0.9, rho_2=0.999, ret_delta=True)
            # else: C, s_C, r_C = gd_adam(C, D['E', 'C'], lr, s_C, r_C, t=iters+1, rho_1=0.9, rho_2=0.999, ret_delta=False)
            C, s_C, r_C = gd_adam(C, D['E', 'C'], lr, s_C, r_C, t=iters+1, rho_1=0.9, rho_2=0.999)
            # if Theta_mask != 'none': Theta, s_Theta, r_Theta = gd_adam(Theta, D['E', 'Theta'], lr, s_Theta, r_Theta, t=iters+1, rho_1=0.9, rho_2=0.999)
            Theta, s_Theta, r_Theta = gd_adam(Theta, D['E', 'Theta'], lr, s_Theta, r_Theta, t=iters+1, rho_1=0.9, rho_2=0.999)
            if phi_x_mask: phi_x, s_phi_x, r_phi_x = gd_adam(phi_x, D['E', 'phi_x'], lr, s_phi_x, r_phi_x, t=iters+1, rho_1=0.9, rho_2=0.999)
            # V, s_V, r_V = gd_adam(V, Delta_V, lr, s_V, r_V, t=iters+1, rho_1=0.9, rho_2=0.999)

        # X = V[: 3*N].reshape(X.shape)
        # C = V[3*N : 3*N+3*K].reshape(C.shape)
        # Theta = V[3*N+3*K : -1].reshape(Theta.shape)
        # phi_x = V[-1]

        C[main_indexes[1],:] = sphere_project(C[main_indexes[1],:], C[main_indexes[0],:], R_scale)

    print(f'{E_min = }')
    if not ret_arrays: return X_release, C_release, Theta_release, phi_x_release, E[E > 0]
    else: return X_release, C_release, Theta_release, phi_x_release, E[E > 0], np.array(X_list), np.array(C_list), np.array(Theta_list), np.array(phi_x_list)

def get_pixels(image):
    labeled_image, num_features = ndimage.label(image > 1e-1)
    pixels = np.zeros((num_features, 2))
    for label in range(1, num_features + 1):
        mask = labeled_image == label
        if image.ndim == 2: coords = np.column_stack(np.where(mask))[:,::-1]
        elif image.ndim == 3: coords = np.column_stack(np.where(mask)[1:])[:,::-1]
        avg_coords = np.average(coords, weights=image[mask], axis=0)
        pixels[label-1] = avg_coords
    if image.ndim == 2: return pixels
    elif image.ndim == 3: return pixels.reshape(image.shape[0], -1, 2).transpose(1, 0, 2)

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

def rotate_scene(V, R):
    if V.ndim == 2: return V @ R.T
    elif V.ndim == 3: return np.einsum('kl,ijl->ijk', R, V)

def draw_3d_scene(ax, X, C, Theta, phi_x, r, f, object_corner_list=None, show_projections=True, show_traces=False, show_nums=False):
    R = ryxz(Theta)

    camera_dir = np.array([0.0, 0.0, 1.0])
    camera_dir = np.einsum('ijk,k->ij', R, camera_dir)
    tan_phi_x_2 = np.tan(phi_x / 2)
    camera_corners = np.array([[tan_phi_x_2, tan_phi_x_2 / r, 1],
                               [- tan_phi_x_2, tan_phi_x_2 / r, 1],
                               [tan_phi_x_2, - tan_phi_x_2 / r, 1],
                               [- tan_phi_x_2, - tan_phi_x_2 / r, 1]])
    camera_corners = np.einsum('ijk,nk->inj', R, camera_corners)

    if show_projections:
        X_proj = project(X, C, Theta, f=f)

    # Поворот всей системы вокруг оси x на угол -pi/2
    Ryz = np.array([[1, 0, 0],
                    [0, 0, 1],
                    [0, -1, 0]])
    X = rotate_scene(X, Ryz)
    C = rotate_scene(C, Ryz)
    camera_dir = rotate_scene(camera_dir, Ryz)
    camera_corners = rotate_scene(camera_corners, Ryz)
    if show_projections:
        X_proj = rotate_scene(X_proj, Ryz)


    if show_traces:
        for j in range(C.shape[0]):
            ax.add_collection(Line3DCollection([[X[i], C[j]] for i in range(X.shape[0])],
                                               linewidth=0.5, colors='red'))

    if object_corner_list:
        for corner in object_corner_list:
            ax.add_collection(Line3DCollection([[X[corner[0],:], X[corner[1],:]]],
                                               linewidth=1.5, colors='red'))
        for j in range(C.shape[0]):
            for corner in object_corner_list:
                ax.add_collection(Line3DCollection([[X_proj[corner[0],j,:], X_proj[corner[1],j,:]]],
                                                   linewidth=1.0, colors='red'))

    ax.scatter(*X.T, marker='o', s=40, color='red')

    if show_projections:
        ax.plot(*X_proj.reshape(X_proj.shape[0] * X_proj.shape[1], 3).T, linestyle=' ', marker='o', markersize=3, color='red')

    ax.add_collection(Line3DCollection([[C[j,:], C[j,:] + f * camera_dir[j,:]] for j in range(C.shape[0])],
                                       linewidth=1.0, linestyle='--', colors='blue'))
    for k in range(4):
        ax.add_collection(Line3DCollection([[C[j,:], C[j,:] + f * camera_corners[j,k,:]] for j in range(C.shape[0])],
                                           linewidth=1.0, colors='blue'))

    camera_corner_list = [[0, 1], [1, 3], [3, 2], [2, 0]]
    for corner in camera_corner_list:
        ax.add_collection(Line3DCollection([[C[j,:] + f * camera_corners[j,corner[0],:], C[j,:] + f * camera_corners[j,corner[1],:]] for j in range(C.shape[0])],
                                           linewidth=1.0, colors='blue'))
    for j in range(C.shape[0]):
        verts = np.vstack(([C[j,:] + f * camera_corners[j,0,:]],
                           [C[j,:] + f * camera_corners[j,1,:]],
                           [C[j,:] + f * camera_corners[j,3,:]],
                           [C[j,:] + f * camera_corners[j,2,:]]))
        ax.add_collection3d(Poly3DCollection([verts], facecolor='blue', alpha=0.25))

    ax.scatter(*C.T, marker='o', s=40, color='blue')

    if show_nums:
        for i in range(X.shape[0]):
            ax.text(*X[i,:], f'x{i}')
        for j in range(C.shape[0]):
            ax.text(*C[j,:], f'c{j}')

def draw_2d_3d_scene(fig, j_slider, X, C, Theta, phi_x, W, H, dist_scale=20, f=0.2,
                     axis_off=False, grid_off=False, show_traces=False):
    assert isinstance(j_slider, Slider)
    ax0 = fig.add_subplot(1, 2, 1, projection='3d')
    xmin, ymin, zmin = np.row_stack((X.min(axis=0), C.min(axis=0))).min(axis=0)
    xmax, ymax, zmax = np.row_stack((X.max(axis=0), C.max(axis=0))).max(axis=0)
    ax0.set_xlim(xmin, xmax)
    ax0.set_ylim(zmin, zmax)
    ax0.set_zlim(-ymax, -ymin)
    ax0.set_zticks(np.round(np.array(ax0.get_zticks()), 6))
    ax0.set_zticklabels(ax0.get_zticks()[::-1])
    ax0.set_box_aspect([ub - lb for lb, ub in (getattr(ax0, f'get_{a}lim')() for a in 'xyz')])
    ax0.set_xlabel('x')
    ax0.set_ylabel('z')
    ax0.set_zlabel('y')
    if axis_off: ax0.set_axis_off()
    if grid_off: ax0.grid(False)

    ax1 = fig.add_subplot(1, 2, 2)
    ax1.set_xlim(0, W)
    ax1.set_ylim(H, 0)
    ax1.set_aspect('equal')

    R = ryxz(Theta)
    camera_dir = np.array([0.0, 0.0, 1.0]).reshape(1, -1)
    camera_dir_rot = camera_dir @ R[j_slider.val].T
    tan_phi_x_2 = np.tan(phi_x / 2)
    r = W / H
    camera_corners = np.array([[tan_phi_x_2, tan_phi_x_2 / r, 1],
                               [- tan_phi_x_2, tan_phi_x_2 / r, 1],
                               [tan_phi_x_2, - tan_phi_x_2 / r, 1],
                               [- tan_phi_x_2, - tan_phi_x_2 / r, 1]])
    camera_corners_rot = camera_corners @ R[j_slider.valinit].T
    # Поворот всей системы вокруг оси x на угол -pi/2
    Ryz = np.array([[1, 0, 0],
                    [0, 0, 1],
                    [0, -1, 0]])
    X_show = rotate_scene(X, Ryz)
    C_show = rotate_scene(C, Ryz)
    camera_dir_rot = rotate_scene(camera_dir_rot, Ryz)
    camera_dir_rot = camera_dir_rot.ravel()
    camera_corners_rot = rotate_scene(camera_corners_rot, Ryz)

    X_model = transform(X, C, Theta, phi_x, W, r)
    X_pix = np.floor(X_model)
    I, J = np.where((X_pix[:,:,0] < 0) | (X_pix[:,:,0] >= W) | (X_pix[:,:,1] < 0) | (X_pix[:,:,1] >= H))
    X_pix[I, J] = np.nan

    X_visible = ~np.isnan(X_pix)[:,:,0]
    I_visible = np.where(X_visible[:,j_slider.val] == True)[0]
    ax0.plot(*X_show[I_visible,:].T, marker='o', linestyle=' ', markersize=4, color='red')
    I_not_visible = np.where(X_visible[:,j_slider.val] == False)[0]
    ax0.plot(*X_show[I_not_visible,:].T, marker='o', linestyle=' ', markersize=3, color='red', alpha=0.5)
    ax0.plot(*C_show.T, color='blue')
    ax0.plot(*C_show[j_slider.valinit,:], marker='o', linestyle=' ', markersize=4, color='blue')

    if show_traces:
        X_proj = project(X[I_visible], C[j_slider.val].reshape(1, -1), Theta[j_slider.val].reshape(1, -1), f)
        X_proj_show = rotate_scene(X_proj, Ryz)[:,0,:]
        ax0.add_collection(Line3DCollection(np.stack((X_show[I_visible], X_proj_show), axis=1),
                                            linewidth=0.5, colors='red', alpha=0.5))
        ax0.plot(*X_proj_show.T, marker='o', linestyle=' ', markersize=1.5, color='red', alpha=0.5)

    ax0.plot(*np.vstack((C_show[j_slider.val,:], C_show[j_slider.val,:] + f * camera_dir_rot)).T,
             linewidth=1.0, linestyle='--', color='blue')

    camera_corner_list = [[0, 1], [1, 3], [3, 2], [2, 0]]
    for camera_corner in camera_corners_rot:
        ax0.plot(*np.vstack((C_show[j_slider.val,:],
                             C_show[j_slider.val,:] + f * camera_corner)).T,
                 linewidth=1.0, color='blue')
    for corner in camera_corner_list:
        ax0.plot(*np.vstack((C_show[j_slider.val,:] + f * camera_corners_rot[corner[0],:],
                             C_show[j_slider.val,:] + f * camera_corners_rot[corner[1],:])).T,
                 linewidth=1.0, color='blue')
    verts = np.vstack(([C_show[j_slider.val,:] + f * camera_corners_rot[0,:]],
                       [C_show[j_slider.val,:] + f * camera_corners_rot[1,:]],
                       [C_show[j_slider.val,:] + f * camera_corners_rot[3,:]],
                       [C_show[j_slider.val,:] + f * camera_corners_rot[2,:]]))
    ax0.add_collection3d(Poly3DCollection([verts], facecolor='blue', alpha=0.25))

    dist = distance(X, C, Theta)
    points = ax1.scatter(*X_pix[I_visible, j_slider.val, :].T, s=dist_scale/dist[I_visible, j_slider.val], marker='o', color='red', zorder=10)


    def update(val):
        ax0.clear()
        ax0.set_xlim(xmin, xmax)
        ax0.set_ylim(zmin, zmax)
        ax0.set_zlim(-ymax, -ymin)
        ax0.set_zticks(np.round(np.array(ax0.get_zticks()), 6))
        ax0.set_zticklabels(ax0.get_zticks()[::-1])
        ax0.set_xlabel('x')
        ax0.set_ylabel('z')
        ax0.set_zlabel('y')
        if axis_off: ax0.set_axis_off()
        if grid_off: ax0.grid(False)

        I_visible = np.where(X_visible[:,j_slider.val] == True)[0]
        ax0.plot(*X_show[I_visible,:].T, marker='o', linestyle=' ', markersize=4, color='red')
        I_not_visible = np.where(X_visible[:,j_slider.val] == False)[0]
        ax0.plot(*X_show[I_not_visible,:].T, marker='o', linestyle=' ', markersize=3, color='red', alpha=0.5)
        ax0.plot(*C_show.T, color='blue')
        ax0.plot(*C_show[j_slider.val,:], marker='o', linestyle=' ', markersize=4, color='blue')
        
        if show_traces:
            X_proj = project(X[I_visible], C[j_slider.val].reshape(1, -1), Theta[j_slider.val].reshape(1, -1), f)
            X_proj_show = rotate_scene(X_proj, Ryz)[:,0,:]
            ax0.add_collection(Line3DCollection(np.stack((X_show[I_visible], X_proj_show), axis=1),
                                                linewidth=0.5, colors='red', alpha=0.5))
            ax0.plot(*X_proj_show.T, marker='o', linestyle=' ', markersize=1.5, color='red', alpha=0.5)

        camera_dir_rot = camera_dir @ R[j_slider.val].T
        camera_dir_rot = rotate_scene(camera_dir_rot, Ryz)
        camera_dir_rot = camera_dir_rot.ravel()
        ax0.plot(*np.vstack((C_show[j_slider.val,:], C_show[j_slider.val,:] + f * camera_dir_rot)).T,
                 linewidth=1.0, linestyle='--', color='blue')

        camera_corners_rot = camera_corners @ R[j_slider.val].T
        camera_corners_rot = rotate_scene(camera_corners_rot, Ryz)
        for camera_corner in camera_corners_rot:
            ax0.plot(*np.vstack((C_show[j_slider.val,:],
                                 C_show[j_slider.val,:] + f * camera_corner)).T,
                     linewidth=1.0, color='blue')
        for corner in camera_corner_list:
            ax0.plot(*np.vstack((C_show[j_slider.val,:] + f * camera_corners_rot[corner[0],:],
                                 C_show[j_slider.val,:] + f * camera_corners_rot[corner[1],:])).T,
                     linewidth=1.0, color='blue')
        verts = np.vstack(([C_show[j_slider.val,:] + f * camera_corners_rot[0,:]],
                           [C_show[j_slider.val,:] + f * camera_corners_rot[1,:]],
                           [C_show[j_slider.val,:] + f * camera_corners_rot[3,:]],
                           [C_show[j_slider.val,:] + f * camera_corners_rot[2,:]]))
        ax0.add_collection3d(Poly3DCollection([verts], facecolor='blue', alpha=0.25))


        points.set_offsets(X_pix[I_visible, j_slider.val, :])
        points.set_sizes(dist_scale/dist[I_visible, j_slider.val])
        
        
    j_slider.on_changed(update)

def fundamental_matrix(X_pix, make_rank_2=True, ret_A=False):
    if X_pix.shape[0] < 8:
        raise ValueError(f"Expected X_pix to have shape[0]>=8")
    if X_pix.shape[1] != 2:
        raise ValueError(f"Expected X_pix to have shape[1]=2")

    X_pix_center = X_pix + 0.5
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
    F = f.reshape(3, 3)

    if make_rank_2 and np.linalg.matrix_rank(F) != 2:
        U2, d2, Vh2 = np.linalg.svd(F)
        F = U2 @ np.diag(np.concatenate((d2[:-1], [0.0]))) @ Vh2

    if not ret_A: return F
    else: F, A

def get_theta(R):
    if (np.abs(R @ R.T - np.eye(3)) > 1e-4).any():
        raise ValueError(f"Expected R to be normal (R @ R.T must be approximately equal to eye(3))")
    if np.abs(np.linalg.det(R) - 1.0) > 1e-4:
        raise ValueError(f"Expected R to have determinant approximately 1.0")

    for l in range(2):
        if l == 0:
            c_x, s_x = np.sqrt(R[1,0]**2 + R[1,1]**2), - R[1,2]
        if l == 1:
            c_x, s_x = - np.sqrt(R[1,0]**2 + R[1,1]**2), - R[1,2]
        c_y, s_y = R[[2,0],2] / c_x
        c_z, s_z = R[1,[1,0]] / c_x
        theta_x = np.arctan2(s_x, c_x)
        theta_y = np.arctan2(s_y, c_y)
        theta_z = np.arctan2(s_z, c_z)
        theta = np.array([theta_x, theta_y, theta_z])

        if np.linalg.norm(R - ryxz(theta.reshape(1, -1))[0]) <= 1e-10:
            if l == 1: print('Helped')
            break
    
    return theta

def get_scene_from_F(X_pix, F, W, H, phi_x, ret_status=False):
    if X_pix.shape[1] != 2:
        raise ValueError(f"Expected X_pix to have shape[1]=2")
    
    X_pix_center = X_pix + 0.5

    alpha_x = alpha_y = W / (2 * np.tan(phi_x / 2))
    p_x, p_y = W / 2, H / 2
    K_ = np.array([[alpha_x, 0.0, p_x],
                   [0.0, alpha_y, p_y],
                   [0.0, 0.0, 1]])
    
    # P1 = K_ @ np.column_stack((np.eye(3), np.zeros(3)))
    P1 = np.column_stack((K_, np.zeros(3)))
    E = K_.T @ F @ K_
    U, d, Vh = np.linalg.svd(E)

    W_ = np.array([[0, -1, 0],
                   [1, 0, 0],
                   [0, 0, 1]])
    
    k_values = []
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
        
        # непонятная ситуация
        if np.linalg.det(R) < 0:
            # print('Weird')
            R = - R

        P2 = K_ @ np.column_stack((R, t))

        X = np.zeros((X_pix_center.shape[0], 3))
        for i in range(X.shape[0]):
            x1, y1 = X_pix_center[i,0,:]
            x2, y2 = X_pix_center[i,1,:]
            A = np.vstack((x1 * P1[2] - P1[0],
                           y1 * P1[2] - P1[1],
                           x2 * P2[2] - P2[0],
                           y2 * P2[2] - P2[1]))
            U2, d2, Vh2 = np.linalg.svd(A)
            x = Vh2[-1]
            x = x / x[-1]
            X[i] = x[:-1]
        
        RT = R.T
        C = np.vstack((np.zeros(3), - RT @ t))

        theta = get_theta(RT)
        Theta = np.row_stack((np.zeros(3), theta))
        
        # if (distance(X, C, Theta) <= 0).any():
        #     print(f'Weird, {k = }, {distance(X, C, Theta) = }')
        if (distance(X, C, Theta) > 0).all():
            k_values.append(k)
            # break

    if len(k_values) != 1:
        status = False
        # print(f'IDK, {k_values = }')
    else:
        status = True
        # print(f'Yay, {k_values = }')
        k = k_values[0]
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
        
        # непонятная ситуация
        if np.linalg.det(R) < 0: R = - R

        P2 = K_ @ np.column_stack((R, t))

        X = np.zeros((X_pix_center.shape[0], 3))
        for i in range(X.shape[0]):
            x1, y1 = X_pix_center[i,0,:]
            x2, y2 = X_pix_center[i,1,:]
            A = np.vstack((x1 * P1[2] - P1[0],
                           y1 * P1[2] - P1[1],
                           x2 * P2[2] - P2[0],
                           y2 * P2[2] - P2[1]))
            U2, d2, Vh2 = np.linalg.svd(A)
            x = Vh2[-1]
            x = x / x[-1]
            X[i] = x[:-1]
        
        RT = R.T
        C = np.vstack((np.zeros(3), - RT @ t))

        theta = get_theta(RT)
        Theta = np.row_stack((np.zeros(3), theta))
    
    if np.linalg.norm(RT - ryxz(Theta)[1]) > 1e-10:
            print(f'OH NO, {np.linalg.norm(RT - ryxz(Theta)[1]) = }')
            print(f'{ryxz(Theta)[1] = }')
            print(f'{RT = }')
    
    if not ret_status: return X, C, Theta
    else: return X, C, Theta, status