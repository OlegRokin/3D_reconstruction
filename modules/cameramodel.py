NAME = "cameramodel"

import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.optimize import linprog
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.patches as patches
from matplotlib.collections import LineCollection, PolyCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from matplotlib.widgets import Slider
from matplotlib.image import imread
import functools
import time
from tqdm import tqdm


def validate_scene_arrays_shapes(X=None, C=None, Theta=None, R=None, X_pix=None, ret_shapes=False):
    N, K = None, None
    
    if X is not None:
        if X.ndim != 2:
            raise ValueError(f'Expected X to have ndim = 2, got {X.ndim}')
        if X.shape[1] != 3:
            raise ValueError(f'Expected X to have shape[1] = 3, got {X.shape[1]}')
        if ret_shapes or X_pix is not None:
            N = X.shape[0]
    if C is not None:
        if C.ndim != 2:
            raise ValueError(f'Expected C to have ndim = 2, got {C.ndim}')
        if C.shape[1] != 3:
            raise ValueError(f'Expected C to have shape[1] = 3, got {C.shape[1]}')
        if ret_shapes or Theta is not None or R is not None or X_pix is not None:
            K = C.shape[0]
    if Theta is not None:
        if Theta.ndim != 2:
            raise ValueError(f'Expected Theta to have ndim = 2, got {Theta.ndim}')
        if Theta.shape[1] != 3:
            raise ValueError(f'Expected Theta to have shape[1] = 3, got {Theta.shape[1]}')
        if K is not None and Theta.shape[0] != K:
            raise ValueError(f'Expected Theta to have shape[0] = {K}, got {Theta.shape[0]}')
        elif ret_shapes or R is not None or X_pix is not None:
            K = Theta.shape[0]
    if R is not None:
        if R.ndim != 3:
            raise ValueError(f'Expected R to have ndim = 3, got {R.ndim}')
        if R.shape[1:] != (3, 3):
            raise ValueError(f'Expected R to have shape[1:] = (3, 3), got {R.shape[1:]}')
        if K is not None and R.shape[0] != K:
            raise ValueError(f'Expected R to have shape[0] = {K}, got {R.shape[0]}')
        elif ret_shapes or X_pix is not None:
            K = R.shape[0]
    if X_pix is not None:
        if X_pix.ndim != 3:
            raise ValueError(f'Expected X_pix to have ndim = 3, got {X_pix.ndim}')
        if X_pix.shape[2] != 2:
            raise ValueError(f'Expected X_pix to have shape[2] = 2, got {X_pix.shape[2]}')
        if N is not None and X_pix.shape[0] != N:
            raise ValueError(f'Expected X_pix to have shape[0] = {N}, got {X_pix.shape[0]}')
        elif ret_shapes:
            N = X_pix.shape[0]
        if K is not None and X_pix.shape[1] != K:
            raise ValueError(f'Expected X_pix to have shape[1] = {K}, got {X_pix.shape[1]}')
        elif ret_shapes:
            K = X_pix.shape[1]
    
    if ret_shapes: return N, K


def error(X_pix, X_model):
    validate_scene_arrays_shapes(X_pix=X_pix)
    if X_model.shape != X_pix.shape:
        raise ValueError(f'Expected X_pix and X_model to have the same shape, got {X_pix.shape} != {X_model.shape}')

    X_pix_visible = X_pix[:,:,0] >= 0
    X_model_visible = ~np.isnan(X_model)[:,:,0]
    X_visible = X_pix_visible * X_model_visible
    X_pix_center_flat = X_pix[~X_visible].ravel() + 0.5
    X_model_flat = X_model[~X_visible].ravel()
    X_diff = X_model_flat - X_pix_center_flat

    return np.dot(X_diff, X_diff) / (2 * X_visible.sum())


def rx(theta):
    R = np.zeros((theta.size, 3, 3))
    R[:,0,0] = 1.0
    R[:,1,1] = R[:,2,2] = np.cos(theta)
    R[:,2,1] = np.sin(theta)
    R[:,1,2] = - R[:,2,1]
    return R

def ry(theta):
    R = np.zeros((theta.size, 3, 3))
    R[:,1,1] = 1.0
    R[:,0,0] = R[:,2,2] = np.cos(theta)
    R[:,0,2] = np.sin(theta)
    R[:,2,0] = - R[:,0,2]
    return R

def rz(theta):
    R = np.zeros((theta.size, 3, 3))
    R[:,2,2] = 1.0
    R[:,0,0] = R[:,1,1] = np.cos(theta)
    R[:,1,0] = np.sin(theta)
    R[:,0,1] = - R[:,1,0]
    return R

def ryxz(Theta, _validate=True):
    if _validate: validate_scene_arrays_shapes(Theta=Theta)
    return ry(Theta[:,1]) @ rx(Theta[:,0]) @ rz(Theta[:,2])

# def d_rx(theta):
#     R = np.zeros((theta.size, 3, 3))
#     R[:,1,1] = R[:,2,2] = - np.sin(theta)
#     R[:,2,1] = np.cos(theta)
#     R[:,1,2] = - R[:,2,1]
#     return R

# def d_ry(theta):
#     R = np.zeros((theta.size, 3, 3))
#     R[:,0,0] = R[:,2,2] = - np.sin(theta)
#     R[:,0,2] = np.cos(theta)
#     R[:,2,0] = - R[:,0,2]
#     return R

# def d_rz(theta):
#     R = np.zeros((theta.size, 3, 3))
#     R[:,0,0] = R[:,1,1] = - np.sin(theta)
#     R[:,1,0] = np.cos(theta)
#     R[:,0,1] = - R[:,1,0]
#     return R


def project(X, C, Theta, f=1.0, dir_only=False, _validate=True):
    if _validate: validate_scene_arrays_shapes(X=X, C=C, Theta=Theta)

    R = ryxz(Theta, _validate=False)
    X_rot = ((X[:,np.newaxis,:] - C[np.newaxis,:,:]).transpose(1, 0, 2) @ R).transpose(1, 0, 2)     # np.einsum('jlk,ijl->ijk', R, X[:,np.newaxis,:] - C[np.newaxis,:,:])
    X_proj = np.full_like(X_rot, f)
    X_proj[:,:,:2] = f * X_rot[:,:,:2] / X_rot[:,:,2][:,:,np.newaxis]
    X_proj = (X_proj.transpose(1, 0, 2) @ R.transpose(0, 2, 1)).transpose(1, 0, 2)      # np.einsum('jkl,ijl->ijk', R, X_proj)
    
    if dir_only: return X_proj
    else: return X_proj + C[np.newaxis,:,:]


def enforce_int_img_arg(position):
    """
    Check if function argument in the position is H instead of r.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            value = args[position]
            if not isinstance(value, int):
                raise TypeError(f'This function now gets argument H instead of r')
            return func(*args, **kwargs)
        return wrapper
    return decorator


@enforce_int_img_arg(5)
def reverse_project(X_pix, C, Theta, phi_x, W, H, f=1.0, dir_only=False, _validate=True):
    if _validate: validate_scene_arrays_shapes(X_pix=X_pix, C=C, Theta=Theta)

    WH = np.array([W, H])
    X_pix_center = X_pix + 0.5
    X_pix_center[X_pix[:,:,0] < 0] = np.nan
    X_proj = np.full((*X_pix.shape[:2], 3), f)
    X_proj[:,:,:2] = f * np.tan(phi_x / 2.0) / W * (2.0 * X_pix_center[:,:,:2] - WH)
    R = ryxz(Theta, _validate=False)
    X_proj = (X_proj.transpose(1, 0, 2) @ R.transpose(0, 2, 1)).transpose(1, 0, 2)        # np.einsum('jkl,ijl->ijk', ryxz(Theta), X_proj)
    
    if dir_only: return X_proj
    else: return X_proj + C[np.newaxis,:,:]


@enforce_int_img_arg(5)
def transform(X, C, Theta, phi_x, W, H, delete_back_points=True, delete_boundary_points=False, _validate=True):
    if _validate: validate_scene_arrays_shapes(X=X, C=C, Theta=Theta)

    WH = np.array([W, H])
    R = ryxz(Theta, _validate=False)
    X_rot = ((X[:,np.newaxis,:] - C[np.newaxis,:,:]).transpose(1, 0, 2) @ R).transpose(1, 0, 2)       # np.einsum('jlk,ijl->ijk', ryxz(Theta), X[:,np.newaxis,:] - C[np.newaxis,:,:])
    if delete_back_points: X_rot[np.where(X_rot[:,:,2] <= 0)] = np.nan
    X_model = (W / np.tan(phi_x / 2.0) * X_rot[:,:,:2] / X_rot[:,:,2][:,:,np.newaxis] + WH) / 2.0
    if delete_boundary_points:
        I, J = np.where((X_model[:,:,0] < 0) | (X_model[:,:,0] >= W) | (X_model[:,:,1] < 0) | (X_model[:,:,1] >= H))
        X_model[I, J] = np.nan
    
    return X_model


def distance(X, C, Theta, _validate=True):
    if _validate: validate_scene_arrays_shapes(X=X, C=C, Theta=Theta)
    R = ryxz(Theta, _validate=False)
    return np.einsum('jl,ijl->ij', R[:,:,2], X[:,np.newaxis,:] - C[np.newaxis,:,:])


def orthogonolize_R(Q, check_det=True, _validate=True):
    if _validate:
        if Q.ndim != 3:
            raise ValueError(f'Expected Q to have ndim = 3, got {Q.ndim}')
        if Q.shape[1:] != (3, 3):
            raise ValueError(f'Expected Q to have shape[1:] = (3, 3), got {Q.shape[1:]}')
    
    U, Sigma, Vh = np.linalg.svd(Q)
    R = U @ Vh

    if check_det:
        det_R = np.linalg.det(R)
        j_bad = np.where(det_R < 0.0)[0]
        if j_bad.size > 0:
            print(f'Bad: {j_bad}')
            R[j_bad] *= -1.0
    
    return R, Sigma


def get_Theta(R, orthogonolize=False, _validate=True):
    if _validate: validate_scene_arrays_shapes(R=R)
    if orthogonolize:
        R, _ = orthogonolize_R(R, _validate=False)
    else:
        if (np.abs(R.transpose(0, 2, 1) @ R - np.eye(3)) > 1e-9).any():     # np.einsum('jkl,jkn->jln', R, R)
            raise ValueError('Expected R[i] to be normal for all i (R[i] @ R[i].T must be approximately equal to eye(3)).')
        if (np.abs(np.linalg.det(R) - 1.0) > 1e-9).any():
            raise ValueError('Expected R[i] to have determinant approximately 1.0 for all i.')

    C_x = np.sqrt(R[:,1,0]**2 + R[:,1,1]**2)
    S_x = - R[:,1,2]
    C_y, S_y = R[:,2,2] / C_x, R[:,0,2] / C_x
    C_z, S_z = R[:,1,1] / C_x, R[:,1,0] / C_x
    Theta = np.column_stack((np.arctan2(S_x, C_x),
                             np.arctan2(S_y, C_y),
                             np.arctan2(S_z, C_z)))
    return Theta


def align_scene(X, C, Theta=None, R=None, ret_R=False, _validate=True):
    if Theta is None and R is None:
        raise ValueError('Expected Theta or R to be passed')
    elif Theta is not None and R is not None:
        raise ValueError('Expected only Theta or R to be passed, not both')
    if _validate: validate_scene_arrays_shapes(X=X, C=C, Theta=Theta, R=R)

    X_new = X - C[0]
    C_new = C - C[0]

    if Theta is not None:
        R_zero = ryxz(Theta[0,:].reshape(1, -1), _validate=False)[0].T
    elif R is not None:
        R_zero = R[0].T

    X_new = X_new @ R_zero.T
    C_new = C_new @ R_zero.T
    
    if Theta is not None:
        R_new = R_zero @ ryxz(Theta, _validate=False)
    elif R is not None:
        R_new = R_zero @ R
    Theta_new = get_Theta(R_new, _validate=False)

    if not ret_R: return X_new, C_new, Theta_new
    else: return X_new, C_new, Theta_new, R_new


def normalize_scene(X, C, main_indexes=None, r_scale=1.0, ret_j=False, _validate=True):
    if _validate:
        validate_scene_arrays_shapes(X=X, C=C)
        if main_indexes is not None and len(main_indexes) != 2:
            raise ValueError(f'Expected main_indexes to have length 2, got {len(main_indexes)}')
    
    if main_indexes is None:
        j0 = 0
        j1 = 1 + np.argmax(np.linalg.norm(C[1:] - C[0], axis=1))
    else: j0, j1 = main_indexes

    scale = np.linalg.norm(C[j1] - C[j0])
    factor = r_scale / scale
    X_new = X * factor
    C_new = C * factor

    if not ret_j: return X_new, C_new
    else: return X_new, C_new, j1


def sgd(x, grad_x, lr):
    x_new = x - lr * grad_x
    return x_new, {}

def gd_adam(x, grad_x, lr, s, r, t, rho_1=0.9, rho_2=0.999):
    s_new = rho_1 * s + (1.0 - rho_1) * grad_x
    r_new = rho_2 * r + (1.0 - rho_2) * grad_x**2
    s_corr = s_new / (1.0 - rho_1**t)
    r_corr = r_new / (1.0 - rho_2**t)
    x_new = x - lr * s_corr / (1e-12 + np.sqrt(r_corr))
    t_new = t + 1
    return x_new, {'s': s_new, 'r': r_new, 't': t_new}


def lr_manager(lr, iters, E, E_min, E_min_temp,
               patience_timer, success_timer, began_decreasing, increased,
               patience, factor, params, ret_arrays, func, func_args, func_kwargs):
    returns_list = []

    lr_, iters_, E_min_, E_min_temp_ = lr, iters, E_min, E_min_temp
    patience_timer_, success_timer_, began_decreasing_, increased_ = patience_timer, success_timer, began_decreasing, increased
    
    if iters_ == 0:
        E_min_ = E[iters]
    else:
        if E[iters] < E_min_:
            E_min_ = E[iters]
            if not began_decreasing_:
                began_decreasing_ = True

            for key, val in params.items():
                val['release'] = val['main'].copy()

            if patience_timer_ != patience:
                patience_timer_ = patience
            if success_timer_ > 0:
                success_timer_ -= 1
            else:
                print(f'{iters_} : Icrease LR to {lr_ * factor}')
                lr_ *= factor
                success_timer_ = patience
                increased_ = True
                E_min_temp_ = E_min_

                for key1, val1 in params.items():
                    for key2 in val1['gd']:
                        val1['gd_temp'][key2] = val1['gd'][key2].copy()
                        val1['gd'][key2] = val1['gd_init'][key2].copy()
        
        elif E[iters] > E_min_:
            if increased_:
                patience_timer_ = 0
                iters_ -= 1
            if success_timer_ != patience:
                success_timer_ = patience
            if patience_timer_ > 0:
                patience_timer_ -= 1
            else:
                if began_decreasing_:
                    patience_timer_ = patience
                else:
                    iters_ -= 1

                print(f'{iters_} : Decrease LR to {lr_ / factor}')
                lr_ /= factor

                for key, val in params.items():
                    val['main'][:] = val['release']

                results = func(*func_args, **func_kwargs)
                returns_list.append(results)

                if not increased_ or (increased_ and E_min_temp_ > E_min_):
                    for key1, val1 in params.items():
                        for key2 in val1['gd']:
                            val1['gd'][key2] = val1['gd_init'][key2].copy()
                else:
                    for key1, val1 in params.items():
                        for key2 in val1['gd']:
                            val1['gd'][key2] = val1['gd_temp'][key2].copy()
                    increased_ = False

    if ret_arrays:
        for key, val in params.items():
            val['list'][iters] = val['main'].copy()

    returns_tuple = (lr_, iters_, E_min_, E_min_temp_, patience_timer_, success_timer_, began_decreasing_, increased_)
    return returns_tuple, returns_list


@enforce_int_img_arg(2)
def fit(X_pix, W, H, lr, max_iters, X_0, C_0, Theta_0, phi_x_0,
        X_mask=True, C_Theta_mask=True, phi_x_mask=True, optimizer='Adam', patience=1000,
        factor=2.0, stop_value=0.0, stop_diff=1e-12, print_step=1000, ret_arrays=False):
    # N, K = X_pix.shape[:2]
    N, K = validate_scene_arrays_shapes(X=X_0, C=C_0, Theta=Theta_0, X_pix=X_pix, ret_shapes=True)

    WH_2 = np.array([W, H]) / 2.0
    X_visible = X_pix[:,:,0] >= 0
    NK_nan = X_visible.sum()
    X_pix_center = X_pix + 0.5
    X_pix_center[~X_visible] = np.nan
    X, C, Theta, phi_x = X_0.copy(), C_0.copy(), Theta_0.copy(), np.array([phi_x_0])

    if C_Theta_mask:
        D_Rx, D_Ry, D_Rz = np.zeros((3, K, 3, 3))

    D = dict()
    D['X_model', 'X_rot'] = np.zeros((N, K, 2, 3))
    if C_Theta_mask: D['R', 'Theta'] = np.zeros((K, 3, 3, 3))

    def inner_transform(X, C, Theta, phi_x, calc_Rs, R=None, f=None):
        returns_dict = dict()
        
        X_tr = X[:,np.newaxis,:] - C[np.newaxis,:,:]
        if calc_Rs:
            Rx, Ry, Rz = rx(Theta[:,0]), ry(Theta[:,1]), rz(Theta[:,2])
            returns_dict['Rx'], returns_dict['Ry'], returns_dict['Rz'] = Rx, Ry, Rz
        if R is None:
            R_ = Ry @ Rx @ Rz
            returns_dict['R'] = R_
        else:
            R_ = R
        X_rot = (X_tr.transpose(1, 0, 2) @ R_).transpose(1, 0, 2)       # np.einsum('jlk,ijl->ijk', R_, X_tr)

        if f is None:
            f_ = W / 2.0 / np.tan(phi_x / 2.0)
            returns_dict['f'] = f_
        else:
            f_ = f
        X_model = f_ * X_rot[:,:,:2] / X_rot[:,:,2][:,:,np.newaxis] + WH_2
        X_diff = np.nan_to_num(X_model - X_pix_center)

        return (X_diff, X_rot, X_tr), returns_dict

    if not phi_x_mask:
        f = W / 2.0 / np.tan(phi_x / 2.0)

    params_names = []
    if X_mask: params_names.append('X')
    if C_Theta_mask: params_names.extend(['C', 'Theta'])
    if phi_x_mask: params_names.append('phi_x')

    params = {name: dict() for name in params_names}
    if X_mask: params['X']['main'] = X
    if C_Theta_mask: params['C']['main'], params['Theta']['main'] = C, Theta
    if phi_x_mask: params['phi_x']['main'] = phi_x

    for key, val in params.items():
        val['release'] = val['main'].copy()

    gd_func_dict = {'SGD': sgd, 'Adam': gd_adam}

    for key, val in params.items():
        val['gd'], val['gd_temp'], val['gd_init'] = dict(), dict(), dict()

    if optimizer == 'Adam':
        for key, val in params.items():
            val['gd']['s'], val['gd_temp']['s'], val['gd_init']['s'] = np.zeros_like(val['main']), np.zeros_like(val['main']), np.zeros_like(val['main'])
            val['gd']['r'], val['gd_temp']['r'], val['gd_init']['r'] = np.zeros_like(val['main']), np.zeros_like(val['main']), np.zeros_like(val['main'])
            val['gd']['t'], val['gd_temp']['t'], val['gd_init']['t'] = np.array([1]), np.array([1]), np.array([1])

    E = np.zeros(max_iters)
    E_min, E_min_temp = np.inf, np.inf
    patience_timer = 0
    success_timer = patience
    began_decreasing = False
    increased = False

    if ret_arrays:
        for key, val in params.items():
            val['list'] = max_iters * [None]

    iters = 0
    while iters < max_iters:
        calculate_R = (iters == 0 and X_mask) or not X_mask
        transform_kwargs = dict()
        if not calculate_R: transform_kwargs['R'] = R
        if not phi_x_mask: transform_kwargs['f'] = f
        # calculate_Rs = iters == 0
        calculate_Rs = iters == 0 or not X_mask
        transorm_res_tuple, transorm_res_dict = inner_transform(X, C, Theta, phi_x, calc_Rs=calculate_Rs, **transform_kwargs)
        X_diff, X_rot, X_tr = transorm_res_tuple
        if calculate_Rs: Rx, Ry, Rz = transorm_res_dict['Rx'], transorm_res_dict['Ry'], transorm_res_dict['Rz']
        if calculate_R: R = transorm_res_dict['R']
        if phi_x_mask: f = transorm_res_dict['f']

        E[iters] = np.dot(X_diff.ravel(), X_diff.ravel()) / (2 * NK_nan)     # должно считаться в >2 раза быстрее, чем np.sum(X_diff**2) / (2 * NK_nan)

        transform_kwargs = dict()
        if not C_Theta_mask: transform_kwargs['R'] = R
        if not phi_x_mask: transform_kwargs['f'] = f
        manager_res_tuple, manager_res_list = lr_manager(lr, iters, E, E_min, E_min_temp,
                                                         patience_timer, success_timer, began_decreasing, increased,
                                                         patience, factor, params, ret_arrays,
                                                         func=inner_transform,
                                                         func_args=(X, C, Theta, phi_x, C_Theta_mask),
                                                         func_kwargs=transform_kwargs)
        lr, iters, E_min, E_min_temp, patience_timer, success_timer, began_decreasing, increased = manager_res_tuple

        if lr == 0.0:
            print(f'{iters} : Reached lr = 0.0')
            break

        if stop_diff > 0.0 and iters > 2 * patience:
            E_prev_min = E[:iters-2*patience].min()
            E_min_diff = E_prev_min - E_min
            if E_min_diff > 0.0 and E_min_diff < stop_diff:
                print(f'{iters} : Reached stopping difference: {E_min_diff} < {stop_diff}')
                break
        
        if E[iters] < stop_value:
            print(f'{iters} : Reached stopping value: {E[iters]} < {stop_value}')
            break
        
        if iters == max_iters - 1:
            print(f'{iters} : Reached maximum iterations')
            break

        if manager_res_list:
            # предполагается, что manager_res_list имеет максимальную длину 1
            transorm_res_tuple, transorm_res_dict = manager_res_list[0]
            X_diff, X_rot, X_tr = transorm_res_tuple
            if C_Theta_mask:
                Rx, Ry, Rz, R = transorm_res_dict['Rx'], transorm_res_dict['Ry'], transorm_res_dict['Rz'], transorm_res_dict['R']
            if phi_x_mask: f = transorm_res_dict['f']

        if iters % print_step == 0: print(f'{iters} : {E_min}')

        D['X_model', 'X_rot'][:,:,0,0] = D['X_model', 'X_rot'][:,:,1,1] = f / X_rot[:,:,2]
        D['X_model', 'X_rot'][:,:,:,2] = - f / (X_rot[:,:,2]**2)[:,:,np.newaxis] * X_rot[:,:,:2]

        D['X_model', 'X'] = D['X_model', 'X_rot'] @ R.transpose(0, 2, 1)[np.newaxis,:,:,:]      # np.einsum('ijkn,jln->ijkl', D['X_model', 'X_rot'], R)

        D['E', 'X_model'] = X_diff / NK_nan

        if X_mask:
            D['E', 'X'] = (D['E', 'X_model'].reshape(N, -1)[:,np.newaxis,:] @ D['X_model', 'X'].reshape(N, -1, 3))[:,0,:]       # np.einsum('ijl,ijlk->ik', D['E', 'X_model'], D['X_model', 'X'])

        if C_Theta_mask:
            if not calculate_Rs and not manager_res_list:
                Rx[:,1,1] = Rx[:,2,2] = np.sign(np.pi - (Theta[:,0] + np.pi/2.0) % (2.0*np.pi)) * np.sqrt(R[:,1,0]**2 + R[:,1,1]**2)
                Rx[:,1,2] = R[:,1,2]
                Rx[:,2,1] = - Rx[:,1,2]

                Ry[:,0,0] = Ry[:,2,2] = R[:,2,2] / Rx[:,1,1]
                Ry[:,0,2] = R[:,0,2] / Rx[:,1,1]
                Ry[:,2,0] = - Ry[:,0,2]

                Rz[:,0,0] = Rz[:,1,1] = R[:,1,1] / Rx[:,1,1]
                Rz[:,1,0] = R[:,1,0] / Rx[:,1,1]
                Rz[:,0,1] = - Rz[:,1,0]

            D_Rx[:,1,1] = D_Rx[:,2,2] = Rx[:,1,2]
            D_Rx[:,2,1] = Rx[:,1,1]
            D_Rx[:,1,2] = - D_Rx[:,2,1]

            D_Ry[:,0,0] = D_Ry[:,2,2] = Ry[:,2,0]
            D_Ry[:,0,2] = Ry[:,0,0]
            D_Ry[:,2,0] = - D_Ry[:,0,2]

            D_Rz[:,0,0] = D_Rz[:,1,1] = Rz[:,0,1]
            D_Rz[:,1,0] = Rz[:,0,0]
            D_Rz[:,0,1] = - D_Rz[:,1,0]

            D['R', 'Theta'][:,:,:,0] = Ry @ D_Rx @ Rz
            D['R', 'Theta'][:,:,:,1] = D_Ry @ Rx @ Rz
            D['R', 'Theta'][:,:,:,2] = Ry @ Rx @ D_Rz

            D['X_rot', 'Theta'] = (X_tr.transpose(1, 0, 2) @ D['R', 'Theta'].reshape(K, 3, 9)).reshape(K, N, 3, 3).transpose(1, 0, 2, 3)        # np.einsum('jnkl,ijn->ijkl', D['R', 'Theta'], X_tr)
            D['X_model', 'Theta'] = D['X_model', 'X_rot'] @ D['X_rot', 'Theta']         # np.einsum('ijkn,ijnl->ijkl', D['X_model', 'X_rot'], D['X_rot', 'Theta'])

            D['E', 'C'] = - (D['E', 'X_model'].transpose(1, 0, 2).reshape(K, -1)[:,np.newaxis,:] @ D['X_model', 'X'].transpose(1, 0, 2, 3).reshape(K, -1, 3))[:,0,:]        #  - np.einsum('ijl,ijlk->jk', D['E', 'X_model'], D['X_model', 'X'])
            D['E', 'Theta'] = (D['E', 'X_model'].transpose(1, 0, 2).reshape(K, -1)[:,np.newaxis,:] @ D['X_model', 'Theta'].transpose(1, 0, 2, 3).reshape(K, -1, 3))[:,0,:]      # np.einsum('ijl,ijlk->jk', D['E', 'X_model'], D['X_model', 'Theta'])

        if phi_x_mask:
            D['X_model', 'phi_x'] = - f / np.sin(phi_x) * X_rot[:,:,:2] / X_rot[:,:,2][:,:,np.newaxis]
            D['E', 'phi_x'] = np.array([np.dot(D['E', 'X_model'].ravel(), D['X_model', 'phi_x'].ravel())])      # должно считаться в >2 раза быстрее, чем np.sum(D['E', 'X_model'] * D['X_model', 'alpha_x'])
        
        for key, val in params.items():
            gd_result, gd_results_dict = gd_func_dict[optimizer](val['main'], D['E', key], lr, **val['gd'])
            val['main'][:] = gd_result
            for key1, val1 in val['gd'].items():
                val1[:] = gd_results_dict[key1]
        
        if X_mask and C_Theta_mask:
            X[:], C[:], Theta[:], R = align_scene(X, C, Theta=Theta, ret_R=True, _validate=False)
            X[:], C[:] = normalize_scene(X, C, _validate=False)

        iters += 1

    print(f'{E_min = }')

    if phi_x_mask:
        params['phi_x']['main'] = params['phi_x']['main'][0]     # вытасикиваем единственный элемент из массива phi_x
        if ret_arrays:
            params['phi_x']['list'] = [phi_x[0] for phi_x in params['phi_x']['list']]

    returns = [*[val['main'] for key, val in params.items()], E[:iters+1]]
    if ret_arrays:
        returns.extend([np.array(val['list']) for key, val in params.items()])
    return returns


def fit_X_sets(X, lr, max_iters, S_0=None, T_0=None, Psi_0=None, optimizer='Adam', patience=1000,
               factor=2.0, stop_value=0.0, stop_diff=1e-12, print_step=1000, ret_arrays=False):
    if X.ndim != 3:
        raise ValueError(f'Expected X to have ndim = 3, got {X.ndim}')
    if X.shape[2] != 3:
        raise ValueError(f'Expected X to have shape[2] = 3, got {X.shape[2]}')
    N, P = X.shape[:2]
    if N < 3:
        raise ValueError(f'Expected X to have shape[0] >= 3, got {N}')
    if P == 1:
        raise ValueError(f'Expected X to have shape[1] >= 2, got 1')

    X_visible = ~np.isnan(X)[:,:,0]
    NP_nan = X_visible.sum()
    
    if S_0 is None: S = np.ones(P)
    else: S = S_0.copy()
    if T_0 is None: T = np.zeros((P, 3))
    else: T = T_0.copy()
    if Psi_0 is None: Psi = np.zeros((P, 3))
    else: Psi = Psi_0.copy()

    D_Rx, D_Ry, D_Rz = np.zeros((3, P, 3, 3))

    D = dict()
    D['R', 'Psi'] = np.zeros((P, 3, 3, 3))

    def inner_transform(X, S, T, Psi, calc_Rs, ret_X_new=False, R=None):
        returns_dict = dict()

        if calc_Rs:
            Rx, Ry, Rz = rx(Psi[:,0]), ry(Psi[:,1]), rz(Psi[:,2])
            returns_dict['Rx'], returns_dict['Ry'], returns_dict['Rz'] = Rx, Ry, Rz
        if R is None:
            R_ = Ry @ Rx @ Rz
            returns_dict['R'] = R_
        else:
            R_ = R
        
        X_rot = (X.transpose(1, 0, 2) @ R_.transpose(0, 2, 1)).transpose(1, 0, 2)       # np.einsum('jkl,ijl->ijk', R_, X)
        X_new = S[np.newaxis,:,np.newaxis] * X_rot + T
        if ret_X_new:
            returns_dict['X_new'] = X_new
        X_mean = np.nanmean(X_new, axis=1, keepdims=True)
        X_diff = X_new - X_mean

        return (X_diff, X_rot), returns_dict
    
    params_names = ['S', 'T', 'Psi']

    params = {name: dict() for name in params_names}
    params['S']['main'] = S
    params['T']['main'] = T
    params['Psi']['main'] = Psi

    for key, val in params.items():
        val['release'] = val['main'].copy()

    gd_func_dict = {'SGD': sgd, 'Adam': gd_adam}

    for key, val in params.items():
        val['gd'], val['gd_temp'], val['gd_init'] = dict(), dict(), dict()

    if optimizer == 'Adam':
        for key, val in params.items():
            val['gd']['s'], val['gd_temp']['s'], val['gd_init']['s'] = np.zeros_like(val['main']), np.zeros_like(val['main']), np.zeros_like(val['main'])
            val['gd']['r'], val['gd_temp']['r'], val['gd_init']['r'] = np.zeros_like(val['main']), np.zeros_like(val['main']), np.zeros_like(val['main'])
            val['gd']['t'], val['gd_temp']['t'], val['gd_init']['t'] = np.array([1]), np.array([1]), np.array([1])

    E = np.zeros(max_iters)
    E_min, E_min_temp = np.inf, np.inf
    patience_timer = 0
    success_timer = patience
    began_decreasing = False
    increased = False

    if ret_arrays:
        for key, val in params.items():
            val['list'] = max_iters * [None]

    iters = 0
    while iters < max_iters:
        calculate_R = calculate_Rs = iters == 0
        transform_kwargs = dict()
        if not calculate_R: transform_kwargs['R'] = R
        transorm_res_list, transorm_res_dict = inner_transform(X, S, T, Psi, calc_Rs=calculate_Rs, **transform_kwargs)
        X_diff, X_rot = transorm_res_list
        if calculate_Rs: Rx, Ry, Rz = transorm_res_dict['Rx'], transorm_res_dict['Ry'], transorm_res_dict['Rz']
        if calculate_R: R = transorm_res_dict['R']

        E[iters] = np.dot(np.nan_to_num(X_diff).ravel(), np.nan_to_num(X_diff).ravel()) / (2.0 * NP_nan)

        manager_res_tuple, manager_res_list = lr_manager(lr, iters, E, E_min, E_min_temp,
                                                         patience_timer, success_timer, began_decreasing, increased,
                                                         patience, factor, params, ret_arrays,
                                                         func=inner_transform,
                                                         func_args=(X, S, T, Psi, True),
                                                         func_kwargs={})
        lr, iters, E_min, E_min_temp, patience_timer, success_timer, began_decreasing, increased = manager_res_tuple

        if lr == 0.0:
            print(f'{iters} : Reached lr = 0.0')
            break

        if stop_diff > 0.0 and iters > 2 * patience:
            E_prev_min = E[:iters-2*patience].min()
            E_min_diff = E_prev_min - E_min
            if E_min_diff > 0.0 and E_min_diff < stop_diff:
                print(f'{iters} : Reached stopping difference: {E_min_diff} < {stop_diff}')
                break

        if E[iters] < stop_value:
            print(f'{iters} : Reached stopping value: {E[iters]} < {stop_value}')
            break

        if iters == max_iters - 1:
            print(f'{iters} : Reached maximum iterations')
            break

        if manager_res_list:
            # предполагается, что manager_res_list имеет максимальную длину 1
            transorm_res_tuple, transorm_res_dict = manager_res_list[0]
            X_diff, X_rot = transorm_res_tuple
            Rx, Ry, Rz, R = transorm_res_dict['Rx'], transorm_res_dict['Ry'], transorm_res_dict['Rz'], transorm_res_dict['R']
        
        if iters % print_step == 0: print(f'{iters} : {E_min}')

        X_diff_centered = np.nan_to_num(X_diff - np.nanmean(X_diff, axis=1, keepdims=True))

        D['E', 'S'] = 2.0 / NP_nan * np.einsum('ijk,ijk->j', X_diff_centered, np.nan_to_num(X_rot))
        
        D['E', 'T'] = 2.0 / NP_nan * np.sum(X_diff_centered, axis=0)

        if not calculate_Rs and not manager_res_list:
            Rx[:,1,1] = Rx[:,2,2] = np.sign(np.pi - (Psi[:,0] + np.pi/2.0) % (2.0*np.pi)) * np.sqrt(R[:,1,0]**2 + R[:,1,1]**2)
            Rx[:,1,2] = R[:,1,2]
            Rx[:,2,1] = - Rx[:,1,2]

            Ry[:,0,0] = Ry[:,2,2] = R[:,2,2] / Rx[:,1,1]
            Ry[:,0,2] = R[:,0,2] / Rx[:,1,1]
            Ry[:,2,0] = - Ry[:,0,2]

            Rz[:,0,0] = Rz[:,1,1] = R[:,1,1] / Rx[:,1,1]
            Rz[:,1,0] = R[:,1,0] / Rx[:,1,1]
            Rz[:,0,1] = - Rz[:,1,0]

        D_Rx[:,1,1] = D_Rx[:,2,2] = Rx[:,1,2]
        D_Rx[:,2,1] = Rx[:,1,1]
        D_Rx[:,1,2] = - D_Rx[:,2,1]

        D_Ry[:,0,0] = D_Ry[:,2,2] = Ry[:,2,0]
        D_Ry[:,0,2] = Ry[:,0,0]
        D_Ry[:,2,0] = - D_Ry[:,0,2]

        D_Rz[:,0,0] = D_Rz[:,1,1] = Rz[:,0,1]
        D_Rz[:,1,0] = Rz[:,0,0]
        D_Rz[:,0,1] = - D_Rz[:,1,0]

        D['R', 'Psi'][:,:,:,0] = Ry @ D_Rx @ Rz
        D['R', 'Psi'][:,:,:,1] = D_Ry @ Rx @ Rz
        D['R', 'Psi'][:,:,:,2] = Ry @ Rx @ D_Rz

        D['X_rot', 'Psi'] = (np.nan_to_num(X).transpose(1, 0, 2) \
                             @ D['R', 'Psi'].transpose(0, 2, 1, 3).reshape(P, 3, 9)).reshape(P, N, 3, 3).transpose(1, 0, 2, 3)      # np.einsum('jknl,ijn->ijkl', D['R', 'Psi'], np.nan_to_num(X))
        D['E', 'Psi'] = 2.0 / NP_nan * ((X_diff_centered.transpose(1, 0, 2).reshape(P, 3 * N) * S[:,np.newaxis])[:,np.newaxis,:] \
                                        @ D['X_rot', 'Psi'].transpose(1, 0, 2, 3).reshape(P, 3 * N, 3))[:,0,:]        # np.einsum('ijn,j,ijnk->jk', X_diff_centered, S, D['X_rot', 'Psi'])

        for key, val in params.items():
            gd_result, gd_results_dict = gd_func_dict[optimizer](val['main'], D['E', key], lr, **val['gd'])
            val['main'][:] = gd_result
            for key1, val1 in val['gd'].items():
                val1[:] = gd_results_dict[key1]

        R[:] = ryxz(Psi, _validate=False)
        T[:] = (T - T[0]) @ R[0] / S[0]     # np.einsum('kl,jk->jl', R[0], T - T[0]) / S[0]
        R[:] = R[0].T @ R       # np.einsum('kl,jkn->jln', R[0], R)
        Psi[:] = get_Theta(R, _validate=False)
        S[:] = np.abs(S)
        S[:] /= S[0]

        iters += 1

    print(f'{E_min = }')

    transorm_res_list, transorm_res_dict = inner_transform(X, S, T, Psi, calc_Rs=True, ret_X_new=True)
    X_new = transorm_res_dict['X_new']

    returns = [X_new, *[val['main'] for key, val in params.items()], E[:iters+1]]
    if ret_arrays: returns.extend([np.array(val['list'][:iters+1]) for key, val in params.items()])
    return returns


def adapt_X_sets(X, normalize=False):
    if X.ndim != 3:
        raise ValueError(f'Expected X to have ndim = 3, got {X.ndim}')
    if X.shape[2] != 3:
        raise ValueError(f'Expected X to have shape[2] = 3, got {X.shape[2]}')
    N, P = X.shape[:2]
    if N < 3:
        raise ValueError(f'Expected X to have shape[0] >= 3, got {N}')
    if P == 1:
        raise ValueError(f'Expected X to have shape[1] >= 2, got 1')

    start_time = time.time()

    # N, P = X.shape[:2]
    X_visible = ~np.isnan(X)[:,:,0]

    print(f'Calculating A:\t\t{time.time() - start_time}')
    W = np.eye(1 + 12 * (P - 1), dtype=np.bool_)
    Y_array = np.full((1 + 12 * (P - 1), *X.shape), np.nan)
    Y_array[0,:,0,:] = X[:,0,:]
    Y_array[1:,X_visible[:,0],0,:] = 0.0

    Q_array = W[:, 1:1+9*(P-1)].reshape(1 + 12 * (P - 1), P - 1, 3, 3)
    Q_mask = np.nonzero(Q_array)
    Q_X = np.zeros((1 + 12 * (P - 1), N, P - 1, 3))
    Q_X[:,~X_visible[:,1:],:] = np.nan
    for n, j, k, l in zip(*Q_mask):
        Q_X[n,:,j,k] += X[:,1:,:][:,j,l]
    Y_array[:,:,1:,:] = Q_X

    T_array = W[:, 1+9*(P-1):].reshape(1 + 12 * (P - 1), P - 1, 3)
    T_mask = np.nonzero(T_array)
    for n, j, k in zip(*T_mask):
        Y_array[n,:,j+1,k] += T_array[n,j,k]

    M_array = np.nanmean(Y_array, axis=2, keepdims=True)
    A = (Y_array - M_array).T.reshape(3 * N * P, 1 + 12 * (P - 1))
    A = A[~np.isnan(A[:,0])]

    print(f'Calculating w:\t\t{time.time() - start_time}')   

    A_mat, A_vec = A[:,1:], A[:,0]
    if not normalize:
        w = - np.linalg.solve(A_mat.T @ A_mat, A_mat.T @ A_vec)
    else:
        A_mat_norm = np.linalg.norm(A_mat, axis=0)
        A_mat /= A_mat_norm
        w = - np.linalg.solve(A_mat.T @ A_mat, A_mat.T @ A_vec) / A_mat_norm
    
    Q, T = np.zeros((P, 3, 3)), np.zeros((P, 3))
    Q[0] = np.eye(3)
    Q[1:] = w[: 9*(P-1)].reshape(P - 1, 3, 3)
    T[1:] = w[9*(P-1) :].reshape(P - 1, 3)

    X_new = (X.transpose(1, 0, 2) @ Q.transpose(0, 2, 1)).transpose(1, 0, 2) + T        # np.einsum('jkl,ijl->ijk', Q, X)
    X_mean = np.nanmean(X_new, axis=1, keepdims=True)
    X_diff = np.nan_to_num(X_new - X_mean).ravel()
    NP_nan = X_visible.sum()

    X_errors = np.nanmean(np.linalg.norm(X_new - np.nanmean(X_new, axis=1, keepdims=True), axis=2), axis=0)
    print(X_errors)

    print(f'Calculating S, T, Psi:\t{time.time() - start_time}')
    R, Sigma = orthogonolize_R(Q)
    S = Sigma.mean(axis=1)

    Psi = get_Theta(R, _validate=False)

    X_new = S[np.newaxis,:,np.newaxis] * (X.transpose(1, 0, 2) @ R.transpose(0, 2, 1)).transpose(1, 0, 2) + T       # np.einsum('jkl,ijl->ijk', R, X)

    X_mean = np.nanmean(X_new, axis=1, keepdims=True)
    X_diff = np.nan_to_num(X_new - X_mean).ravel()
    NP_nan = X_visible.sum()
    print(f'E = {np.dot(X_diff, X_diff) / (2.0 * NP_nan)}')

    return X_new, S, T, Psi


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


def rotate_array(V, R, _validate=True):
    if _validate and R.shape != (3, 3):
        raise ValueError(f'Expected R to have shape (3, 3), got {R.shape}')
    return (V.reshape(-1, 3) @ R.T).reshape(*V.shape)


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


def draw_2d_3d_scene(fig, j_slider, X, C, Theta, phi_x, W, H, X_pix=None, image_paths=None,
                     dist_scale=10.0, f=0.2, trajectory_markers=False, show_image_lines=False,
                     axis_off=False, grid_off=False, ret_axis=False):
    N, K = validate_scene_arrays_shapes(X=X, C=C, Theta=Theta, X_pix=X_pix, ret_shapes=True)
    if image_paths is not None:
        # if image_paths.size != X_pix.shape[1]:
        #     raise ValueError('Expected image_paths.size == X_pix.shape[1]')
        if image_paths.size != K:
            raise ValueError('Expected image_paths.size == K')

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

    # N = X.shape[0]

    R = ryxz(Theta[j_slider.val].reshape(1, -1), _validate=False)[0]
    # camera_dir = np.array([0.0, 0.0, 1.0]).reshape(1, -1)
    camera_dir = np.array([[0.0, 0.0, 1.0]])
    camera_dir_rot = camera_dir @ R.T
    tan_phi_x_2 = np.tan(phi_x / 2.0)
    tan_phi_y_2 = H / W * tan_phi_x_2
    camera_corners = np.array([[  tan_phi_x_2,   tan_phi_y_2, 1.0],
                               [- tan_phi_x_2,   tan_phi_y_2, 1.0],
                               [- tan_phi_x_2, - tan_phi_y_2, 1.0],
                               [  tan_phi_x_2, - tan_phi_y_2, 1.0]])
    camera_corners_rot = camera_corners @ R.T
    # Поворот всей системы вокруг оси x на угол -pi/2
    Ryz = np.array([[1.0,  0.0, 0.0],
                    [0.0,  0.0, 1.0],
                    [0.0, -1.0, 0.0]])
    X_show = rotate_array(X, Ryz, _validate=False)
    C_show = rotate_array(C, Ryz, _validate=False)
    camera_dir_rot = rotate_array(camera_dir_rot, Ryz, _validate=False)
    camera_dir_rot = camera_dir_rot.ravel()
    camera_corners_rot = rotate_array(camera_corners_rot, Ryz, _validate=False)

    X_model = transform(X, C[j_slider.val].reshape(1, -1), Theta[j_slider.val].reshape(1, -1), phi_x, W, H,
                        delete_back_points=True, delete_boundary_points=True)[:,0,:]
    # I_pix_visible = np.where(~np.isnan(X_pix[:,j_slider.val,0]))[0]
    I_pix_visible = np.where(X_pix[:,j_slider.val,0] >= 0)[0]
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
        X_proj = project(X[I_model_visible], C[j_slider.val].reshape(1, -1), Theta[j_slider.val].reshape(1, -1), f, _validate=False)[:,0,:]
        X_proj_show = rotate_array(X_proj, Ryz, _validate=False)

        X_proj_traces = Line3DCollection(np.stack((X_show[I_both_visible], X_proj_show[np.searchsorted(I_model_visible, I_both_visible)]), axis=1),
                                         linewidth=1.0, colors='red', alpha=0.5)
        ax0.add_collection(X_proj_traces)
        X_proj_semitraces = Line3DCollection(np.stack((X_show[I_model_semivisible], X_proj_show[np.searchsorted(I_model_visible, I_model_semivisible)]), axis=1),
                                             linewidth=1.0, colors='red', linestyle='--', alpha=0.5)
        ax0.add_collection(X_proj_semitraces)
        X_proj_points = ax0.scatter(*X_proj_show.T, marker='o', s=3, color='red', alpha=0.5)

    camera_dir_line, = ax0.plot(*np.vstack((C_show[j_slider.val,:], C_show[j_slider.val,:] + f * camera_dir_rot)).T,
                                linewidth=1.0, linestyle='--', color='blue')

    center_to_corners = Line3DCollection(np.stack((np.tile(C_show[j_slider.val,:], (4, 1)),
                                                   C_show[j_slider.val,:] + f * camera_corners_rot), axis=1),
                                         linewidth=1.0, colors='blue')
    ax0.add_collection(center_to_corners)
    
    corners_surface = Poly3DCollection([C_show[j_slider.val,:] + f * camera_corners_rot],
                                       linewidths=1.0, edgecolors='blue', facecolor='blue', alpha=0.25)
    ax0.add_collection3d(corners_surface)

    ax1 = fig.add_subplot(1, 2, 2)
    assert isinstance(ax1, Axes)
    ax1.set_xlim(0, W)
    ax1.set_ylim(H, 0)
    ax1.set_aspect('equal')
    ax1.set_xticks(np.linspace(0, W, 5))
    ax1.set_yticks(np.linspace(H, 0, 5))

    dist = distance(X, C, Theta, _validate=False)
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

        X_model = transform(X, C[j_slider.val].reshape(1, -1), Theta[j_slider.val].reshape(1, -1), phi_x, W, H,
                            delete_back_points=True, delete_boundary_points=True, _validate=False)[:,0,:]
        I_pix_visible = np.where(X_pix[:,j_slider.val,0] >= 0)[0]
        I_model_visible = np.where(~np.isnan(X_model[:,0]))[0]
        I_model_semivisible = np.array(list(set(I_model_visible) - set(I_pix_visible)), dtype=np.int64)
        I_both_visible = np.array(list(set(I_pix_visible) & set(I_model_visible)), dtype=np.int64)
        I_model_not_visible = np.array(list(set(np.arange(N)) - set(I_model_visible)), dtype=np.int64)
        X_visible_points._offsets3d = X_show[I_model_visible,:].T
        X_not_visible_points._offsets3d = X_show[I_model_not_visible,:].T

        if show_image_lines:
            X_proj = project(X[I_model_visible], C[j_slider.val].reshape(1, -1), Theta[j_slider.val].reshape(1, -1), f, _validate=False)[:,0,:]
            X_proj_show = rotate_array(X_proj, Ryz, _validate=False)
            X_proj_traces.set_segments(np.stack((X_show[I_both_visible], X_proj_show[np.searchsorted(I_model_visible, I_both_visible)]), axis=1))
            X_proj_semitraces.set_segments(np.stack((X_show[I_model_semivisible], X_proj_show[np.searchsorted(I_model_visible, I_model_semivisible)]), axis=1))
            X_proj_points._offsets3d = X_proj_show.T

        R = ryxz(Theta[j_slider.val].reshape(1, -1), _validate=False)[0]
        camera_dir_rot = camera_dir @ R.T
        camera_dir_rot = rotate_array(camera_dir_rot, Ryz, _validate=False)
        camera_dir_rot = camera_dir_rot.ravel()
        camera_corners_rot = camera_corners @ R.T
        camera_corners_rot = rotate_array(camera_corners_rot, Ryz, _validate=False)

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
    if X_array.ndim != 3:
        raise ValueError(f'Expected X_array to have ndim = 3, got {X_array.ndim}')
    # if X_array.shape[2] != 3:
    #     raise ValueError(f'Expected X_array to have shape[2] = 3, got {X_array.shape[1]}')
    P = X_array.shape[0]
    if C_array.ndim != 3:
        raise ValueError(f'Expected C_array to have ndim = 3, got {C_array.ndim}')
    # if C_array.shape[2] != 3:
    #     raise ValueError(f'Expected C_array to have shape[2] = 3, got {C_array.shape[1]}')
    if C_array.shape[0] != P:
        raise ValueError(f'Expected C_array.shape[0] = X_array.shape[0], got {C_array.shape[0]} != {X_array.shape[0]}')
    if Theta_array.shape != C_array.shape:
        raise ValueError(f'Expected C_array and Theta_array to have the same shape, got {C_array.shape} != {Theta_array.shape}')
    if phi_x_array.ndim != 1:
        raise ValueError(f'Expected phi_x_array to have ndim = 1, got {phi_x_array.ndim}')
    if phi_x_array.size != P:
        raise ValueError(f'Expected phi_x_array.size = X_array.shape[0], got {phi_x_array.size} != {X_array.shape[0]}')
    validate_scene_arrays_shapes(X=X_array[0], C=C_array[0], Theta=Theta_array[0], X_pix=X_pix)
    if image_paths is not None:
        if image_paths.size != 3:
            raise ValueError('Expected image_paths to have size 3.')

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

    R = ryxz(Theta_array[t_slider.val, j_ids], _validate=False)
    camera_dir = np.array([0.0, 0.0, 1.0])
    camera_dir_rot = R @ camera_dir     # np.einsum('jkl,l->jk', R, camera_dir)
    tan_phi_x_2 = np.tan(phi_x_array[t_slider.val] / 2.0)
    tan_phi_y_2 = H / W * tan_phi_x_2
    camera_corners = np.array([[  tan_phi_x_2,   tan_phi_y_2, 1.0],
                               [- tan_phi_x_2,   tan_phi_y_2, 1.0],
                               [- tan_phi_x_2, - tan_phi_y_2, 1.0],
                               [  tan_phi_x_2, - tan_phi_y_2, 1.0]])
    camera_corners_rot = np.einsum('jkl,nl->jnk', R, camera_corners)
    # Поворот всей системы вокруг оси x на угол -pi/2
    Ryz = np.array([[1.0,  0.0, 0.0],
                    [0.0,  0.0, 1.0],
                    [0.0, -1.0, 0.0]])
    X_show = rotate_array(X_array[t_slider.val], Ryz, _validate=False)
    C_show = rotate_array(C_array[t_slider.val], Ryz, _validate=False)
    C_points_show = C_show[j_ids]
    camera_dir_rot = rotate_array(camera_dir_rot, Ryz, _validate=False)
    camera_corners_rot = rotate_array(camera_corners_rot, Ryz, _validate=False)

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

    corners_surface = Poly3DCollection(np.repeat(C_points_show, 4, axis=0).reshape(-1, 4, 3) + f * camera_corners_rot,
                                       linewidths=1.0, edgecolors='blue', facecolor='blue', alpha=0.25)
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
                        W, H, delete_back_points=True, delete_boundary_points=True, _validate=False)
    I_pix_visible = np.where(X_pix[:,j_ids,0] >= 0)[0]
    I_model_visible = np.where(~np.isnan(X_model[:,:,0]))[0]
    I_both_visible = np.array(list(set(I_pix_visible) & set(I_model_visible)))

    dist = distance(X_array[t_slider.val], C_array[t_slider.val], Theta_array[t_slider.val], _validate=False)
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
        R = ryxz(Theta_array[t_slider.val,j_ids], _validate=False)
        camera_dir_rot = R @ camera_dir     # np.einsum('jkl,l->jk', R, camera_dir)
        tan_phi_x_2 = np.tan(phi_x_array[t_slider.val] / 2.0)
        tan_phi_y_2 = H / W * tan_phi_x_2
        camera_corners = np.array([[  tan_phi_x_2,   tan_phi_y_2, 1.0],
                                   [- tan_phi_x_2,   tan_phi_y_2, 1.0],
                                   [- tan_phi_x_2, - tan_phi_y_2, 1.0],
                                   [  tan_phi_x_2, - tan_phi_y_2, 1.0]])
        camera_corners_rot = np.einsum('jkl,nl->jnk', R, camera_corners)
        # Поворот всей системы вокруг оси x на угол -pi/2
        X_show = rotate_array(X_array[t_slider.val], Ryz, _validate=False)
        C_show = rotate_array(C_array[t_slider.val], Ryz, _validate=False)
        C_points_show = C_show[j_ids]
        camera_dir_rot = rotate_array(camera_dir_rot, Ryz, _validate=False)
        camera_corners_rot = rotate_array(camera_corners_rot, Ryz, _validate=False)

        X_points._offsets3d = X_show.T
        C_points.set_data(*C_show[:,:2].T)
        C_points.set_3d_properties(C_show[:,2])

        camera_dir_line.set_segments(np.stack((C_points_show,
                                               C_points_show + f * camera_dir_rot), axis=1))
        center_to_corners.set_segments(np.stack((np.repeat(C_points_show, 4, axis=0).reshape(-1, 4, 3),
                                                 np.repeat(C_points_show, 4, axis=0).reshape(-1, 4, 3) + f * camera_corners_rot), axis=2).reshape(-1, 2, 3))
        corners_surface.set_verts(np.repeat(C_points_show, 4, axis=0).reshape(-1, 4, 3) + f * camera_corners_rot)

        X_model = transform(X_array[t_slider.val], C_array[t_slider.val,j_ids], Theta_array[t_slider.val,j_ids], phi_x_array[t_slider.val],
                            W, H, delete_back_points=True, delete_boundary_points=True, _validate=False)
        I_model_visible = np.where(~np.isnan(X_model[:,:,0]))[0]
        I_both_visible = np.array(list(set(I_pix_visible) & set(I_model_visible)))

        dist = distance(X_array[t_slider.val], C_array[t_slider.val], Theta_array[t_slider.val], _validate=False)
        dist /= dist.max()
        for j, ax in enumerate(axs):
            X_model_points[j].set_offsets(X_model[I_model_visible,j,:])
            X_model_points[j].set_sizes(dist_scale/dist[I_model_visible,j])
            X_lines[j].set_segments(np.stack((X_model[I_both_visible,j,:], X_pix_center[I_both_visible,j,:]), axis=1))

    t_slider.on_changed(update)

    if ret_axis: return ax0, axs


# def get_R(x, y):
#     v = np.cross(x, y)
#     v_norm = np.linalg.norm(v)
#     if v_norm > 0.0:
#         xy = np.linalg.norm(x) * np.linalg.norm(y)
#         n = v / v_norm
#         sin_phi = v_norm / xy
#         cos_phi = np.dot(x, y) / xy

#         R = cos_phi * np.eye(3)
#         R += (1 - cos_phi) * (n.reshape(-1, 1) @ n.reshape(1, -1))
#         R += sin_phi * np.array([[  0.0, -n[2],  n[1]],
#                                  [ n[2],  0.0,  -n[0]],
#                                  [-n[1],  n[0],   0.0]])
#     else:
#         R = np.eye(3)
#     return R


def get_X_from_C_Theta(X_pix, C, Theta, phi_x, W, H, method='LT', normalized=False, _validate=True):
    if _validate: N, K = validate_scene_arrays_shapes(X_pix=X_pix, C=C, Theta=Theta, ret_shapes=True)
    else: N, K = X_pix.shape[:2]

    if method == 'LT':        
        if not normalized:
            X_pix_center = X_pix + 0.5

            f_x = f_y = W / 2.0 / np.tan(phi_x / 2)
            p_x, p_y = W / 2.0, H / 2.0
        else:
            WH_max = max(W, H)
            X_pix_center = (X_pix + 0.5 - 0.5 * np.array([W, H])) / WH_max

            f_x = f_y = W / WH_max / 2.0 / np.tan(phi_x / 2)
            p_x = p_y = 0.0

        X_pix_center[X_pix[:,:,0] < 0] = np.nan
        
        K_ = np.array([[f_x, 0.0, p_x],
                       [0.0, f_y, p_y],
                       [0.0, 0.0, 1.0]])
        
        RT = ryxz(Theta).transpose(0, 2, 1)
        RTC = np.concatenate((RT, - ((RT @ C[:,:,np.newaxis])[:,:,0]).reshape(-1, 3, 1)), axis=2)        # np.einsum('jkl,jl->jk', RT, C)

        P = K_ @ RTC    # np.einsum('kl,jln->jkn', K_, RTC)

        A_array = np.zeros((N, 2 * K, 4))
        A_array[:, ::2,:] = X_pix_center[:,:,0][:,:,np.newaxis] * P[:,2,:] - P[:,0,:]
        A_array[:,1::2,:] = X_pix_center[:,:,1][:,:,np.newaxis] * P[:,2,:] - P[:,1,:]
        A_array = np.nan_to_num(A_array)

        A_array_mat, A_array_vec = A_array[:,:,:-1], A_array[:,:,-1]
        lstsq_mat = A_array_mat.transpose(0, 2, 1) @ A_array_mat        # np.einsum('ilj,ilk->ijk', A_array_mat, A_array_mat)
        lstsq_vec = (A_array_mat.transpose(0, 2, 1) @ A_array_vec[:,:,np.newaxis])[:,:,0]        # np.einsum('ilj,il->ij', A_array_mat, A_array_vec)
        X = - np.linalg.solve(lstsq_mat, lstsq_vec)
    
    elif method == 'GT':
        V = reverse_project(X_pix, C, Theta, phi_x, W, H, dir_only=True, _validate=False)
        V /= np.linalg.norm(V, axis=2, keepdims=True)
        V_flat = V.reshape(N * K, 3)
        eye_VVt = np.eye(3) - (V_flat[:,:,np.newaxis] @ V_flat[:,np.newaxis,:]).reshape(N, K, 3, 3)     #  np.einsum('ijk,ijl->ijkl', V, V)
        X = np.linalg.solve((eye_VVt).sum(axis=1), np.einsum('ijkl,jl->ijk', eye_VVt, C).sum(axis=1))

    return X


def get_C_Theta_from_X(X, X_pix, phi_x, W, H, normalized=False, _validate=True):
    if _validate: N, K = validate_scene_arrays_shapes(X=X, X_pix=X_pix, ret_shapes=True)
    else: N, K = X_pix.shape[:2]
    
    if not normalized:
        X_pix_center = X_pix + 0.5

        f_x = f_y = W / 2.0 / np.tan(phi_x / 2.0)
        p_x, p_y = W / 2.0, H / 2.0
    else:
        X_pix_center = (X_pix + 0.5 - 0.5 * np.array([W, H])) / max(W, H)

        f_x = f_y = W / max(W, H) / 2.0 / np.tan(phi_x / 2.0)
        p_x = p_y = 0.0
    
    X_pix_center[X_pix[:,:,0] < 0] = np.nan

    if (problem := np.where((X_pix[:,:,0] >= 0).sum(axis=0) < 6)[0]).size > 0:
        print(f'{problem = }')

    K_ = np.array([[f_x, 0.0, p_x],
                   [0.0, f_y, p_y],
                   [0.0, 0.0, 1.0]])

    X_hom = np.ones((N, 4))
    X_hom[:,:3] = X

    A_array = np.zeros((K, 2 * N, 12))
    A_array[:,::2,:4] = A_array[:,1::2,4:8] = - X_hom
    A_array[:, ::2,8:12] = X_pix_center[:,:,0].T[:,:,np.newaxis] * X_hom
    A_array[:,1::2,8:12] = X_pix_center[:,:,1].T[:,:,np.newaxis] * X_hom
    # A_array = np.nan_to_num(A_array)
    A_array_nan = np.where(np.isnan(A_array[:,:,0]) | np.isnan(A_array[:,:,4]) | np.isnan(A_array[:,:,8]))
    A_array[A_array_nan] = 0.0

    if N * K <= 25000:
        U, sigma, Vh = np.linalg.svd(A_array)
        P_array = Vh[:,-1,:].reshape(K, 3, 4)
    else:
        P_array = np.zeros((K, 3, 4))
        for j in range(K):
            w = eigsh(A_array[j].T @ A_array[j], k=1, which='SM')[1]
            P_array[j] = w.reshape(3, 4)

    RT = np.linalg.solve(K_, P_array)

    R, T = RT[:,:,:3], RT[:,:,3]

    U, sigma, Vh = np.linalg.svd(R)
    R_rot = (U @ Vh).transpose(0, 2, 1)
    C = - (R_rot @ T[:,:,np.newaxis])[:,:,0] / sigma.mean(axis=1, keepdims=True)       # np.einsum('jkl,jl->jk', R_rot, T)
    R_rot *= np.sign(np.linalg.det(R_rot))[:,np.newaxis,np.newaxis]

    Theta = get_Theta(R_rot)

    return C, Theta


def fundamental_matrix(X_pix, make_rank_2=True, ret_A=False, normalize=False, W=None, H=None, _validate=True):
    if _validate:
        N, K = validate_scene_arrays_shapes(X_pix=X_pix, ret_shapes=True)
        if N < 8:
            raise ValueError('Expected X_pix to have shape[0] >= 8')
        if K != 2:
            raise ValueError('Expected X_pix to have shape[1] = 2')
        if normalize and (W is None or H is None):
            raise ValueError('With normalize=True W and H have to be passed')
    else: N = X_pix.shape[0]

    if not normalize:
        X_pix_center = X_pix + 0.5
    else:
        X_pix_center = (X_pix + 0.5 - 0.5 * np.array([W, H])) / max(W, H)
    
    X_pix_center[X_pix[:,:,0] < 0] = np.nan

    X_hom = np.ones((N, 2, 3))
    X_hom[:,:,:2] = X_pix_center

    A = (X_hom[:,1,:][:,:,np.newaxis] @ X_hom[:,0,:][:,np.newaxis,:]).reshape(N, 9)

    U, sigma, Vh = np.linalg.svd(A)
    f = Vh[-1]
    # f = eigsh(A.T @ A, k=1, which='SM')[1]    # вроде иногда должно считаться быстрее, чем с np.linalg.svd(A)
    F = f.reshape(3, 3)

    if make_rank_2 and np.linalg.matrix_rank(F) != 2:
        U, sigma, Vh = np.linalg.svd(F)
        sigma[-1] = 0.0
        F = U @ np.diag(sigma) @ Vh

    if not ret_A: return F
    else: return F, A


def get_X_C_Theta_from_F(X_pix, F, W, H, phi_x, ret_status=False, normalized=False, _validate=True):
    if _validate:
        N, K = validate_scene_arrays_shapes(X_pix=X_pix, ret_shapes=True)
        if K != 2:
            raise ValueError('Expected X_pix to have shape[1] = 2')
    else: N = X_pix.shape[0]
    
    if not normalized:
        X_pix_center = X_pix + 0.5

        f_x = f_y = W / 2.0 / np.tan(phi_x / 2)
        p_x, p_y = W / 2.0, H / 2.0
    else:
        WH_max = max(W, H)
        X_pix_center = (X_pix + 0.5 - 0.5 * np.array([W, H])) / WH_max

        f_x = f_y = W / WH_max / 2.0 / np.tan(phi_x / 2)
        p_x = p_y = 0.0

    X_pix_center[X_pix[:,:,0] < 0] = np.nan

    K_ = np.array([[f_x, 0.0, p_x],
                   [0.0, f_y, p_y],
                   [0.0, 0.0, 1.0]])

    P = np.zeros((2, 3, 4))
    P[0,:,:3] = K_

    E = K_.T @ F @ K_
    U, sigma, Vh = np.linalg.svd(E)

    W_ = np.array([[0.0, -1.0, 0.0],
                   [1.0,  0.0, 0.0],
                   [0.0,  0.0, 1.0]])
    
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

        P[1,:,:] = K_ @ np.column_stack((R, t))

        A_array = np.zeros((N, 4, 4))
        A_array[:,::2,:]  = X_pix_center[:,:,0][:,:,np.newaxis] * P[:,2,:] - P[:,0,:]
        A_array[:,1::2,:] = X_pix_center[:,:,1][:,:,np.newaxis] * P[:,2,:] - P[:,1,:]

        A_array_mat, A_array_vec = A_array[:,:,:-1], A_array[:,:,-1]
        lstsq_mat = A_array_mat.transpose(0, 2, 1) @ A_array_mat        # np.einsum('ilj,ilk->ijk', A_array_mat, A_array_mat)
        lstsq_vec = (A_array_mat.transpose(0, 2, 1) @ A_array_vec[:,:,np.newaxis])[:,:,0]        # np.einsum('ilj,il->ij', A_array_mat, A_array_vec)
        X = - np.linalg.solve(lstsq_mat, lstsq_vec)
        
        C = np.vstack((np.zeros(3), - R.T @ t))

        theta = get_Theta(R.T.reshape(1, 3, 3))[0]
        Theta = np.vstack((np.zeros(3), theta))

        if (distance(X, C, Theta) > 0).all():
            status = True
            break
    
    if not ret_status: return X, C, Theta
    else: return X, C, Theta, status


def get_H_diff_matrix(X_pix, normalize=False, W=None, H=None, _validate=True):
    if _validate:
        N, K = validate_scene_arrays_shapes(X_pix=X_pix, ret_shapes=True)
        if normalize and (W is None or H is None):
            raise ValueError('With normalize=True W and H have to be passed')
    else: N, K = X_pix.shape[:-1]

    X_visible = X_pix[:,:,0] >= 0

    if not normalize:
        X_pix_center = X_pix + 0.5
    else:
        X_pix_center = (X_pix + 0.5 - 0.5 * np.array([W, H])) / max(W, H)

    X_pix_center[~X_visible] = np.nan

    X_pix_hom = np.ones((N, K, 3))
    X_pix_hom[:,:,:2] = X_pix_center

    H_diff_matrix = np.full((K, K), np.nan)
    i_size_matrix = np.zeros((K, K), dtype=np.int32)

    for i in tqdm(range(K), desc='Calculating H_diff_matrix', ascii=' █'):
        for j in range(i + 1, K):
            i_subset = np.where(X_visible[:,i] * X_visible[:,j])[0]
            if i_subset.size < 10: continue
            i_size_matrix[i,j] = i_size_matrix[j,i] = i_subset.size

            A = np.zeros((2 * i_subset.size, 9))
            A[::2,:3] = A[1::2,3:6] = - X_pix_hom[i_subset,i,:]
            A[::2,6:9]  = X_pix_hom[i_subset,j,0][:,np.newaxis] * X_pix_hom[i_subset,i,:]
            A[1::2,6:9] = X_pix_hom[i_subset,j,1][:,np.newaxis] * X_pix_hom[i_subset,i,:]

            h0 = np.eye(3).ravel()
            h = eigsh(A.T @ A, k=1, which='SM', v0=h0)[1]
            H_ = h.reshape(3, 3)

            X_pix_hom_H = X_pix_hom[i_subset,i,:] @ H_.T
            X_pix_hom_H /= X_pix_hom_H[:,-1][:,np.newaxis]
            X_pix_hom_diff = X_pix_hom[i_subset,j,:2].ravel() - X_pix_hom_H[:,:2].ravel()
            
            H_diff_matrix[i,j] = H_diff_matrix[j,i] = np.dot(X_pix_hom_diff, X_pix_hom_diff) / i_subset.size

    return H_diff_matrix, i_size_matrix


def find_phi_x(X_pix, W, H, i_size_matrix, H_diff_matrix, H_diff_opt, attempts=5, normalize=False, _validate=True):
    if _validate: validate_scene_arrays_shapes(X_pix=X_pix)
    X_visible = X_pix[:,:,0] >= 0
    WH = np.array([W, H])

    H_diff_matrix_log = np.log10(H_diff_matrix)
    H_diff_matrix_log[H_diff_matrix_log < np.log10(H_diff_opt)] = np.nan
    H_diff_matrix_log[H_diff_matrix_log > np.nanpercentile(H_diff_matrix_log, 95.0)] = np.nan
    H_diff_matrix_log -= np.log10(H_diff_opt)
    H_diff_matrix_log /= np.nanmax(H_diff_matrix_log)

    opt_matrix = (i_size_matrix - 10) / (i_size_matrix.max() - 10) + H_diff_matrix_log
    opt_matrix[opt_matrix > np.nanpercentile(opt_matrix, 75.0)] = np.nan
    opt_matrix_notnan = np.where(~np.isnan(opt_matrix))
    opt_matrix_argsort = np.argsort(opt_matrix[opt_matrix_notnan])[::-2][:attempts]
    i_ids, j_ids = opt_matrix_notnan[0][opt_matrix_argsort], opt_matrix_notnan[1][opt_matrix_argsort]
    
    phi_x_array, h = np.linspace(0, np.pi, 53, retstep=True)
    phi_x_array = phi_x_array[1:-1]

    i_subset_list = []
    ij_subset_list = []
    F_array = np.zeros((attempts, 3, 3))
    for k, (i, j) in enumerate(zip(i_ids, j_ids)):
        i_subset = np.where(X_visible[:,i] * X_visible[:,j])[0]
        ij_subset = np.ix_(i_subset, [i, j])
        i_subset_list.append(i_subset)
        ij_subset_list.append(ij_subset)
        F_array[k] = fundamental_matrix(X_pix[ij_subset], normalize=normalize, W=W, H=H)
    i_subset_sizes = np.array([i_subset.size for i_subset in i_subset_list])

    E_diff = np.inf
    while E_diff > 1e-5:
        E_matrix = np.full((phi_x_array.size, attempts), np.inf)
        for k, (i_subset, ij_subset, F) in enumerate(zip(i_subset_list, ij_subset_list, F_array)):
            X_test, C_test, Theta_test, phi_x_test = [], [], [], []
            success_ids = []
            phi_x_good = np.where((phi_x_array > 0.0) & (phi_x_array < np.pi))[0]
            for j, phi_x in enumerate(phi_x_array[phi_x_good]):
                X_, C_, Theta_, success = get_X_C_Theta_from_F(X_pix[ij_subset], F, W, H, phi_x, ret_status=True, normalized=normalize)
                if success:
                    X_test.append(X_)
                    C_test.append(C_[1,:])
                    Theta_test.append(Theta_[1,:])
                    phi_x_test.append(phi_x)
                    success_ids.append(phi_x_good[j])
            X_test, C_test, Theta_test, phi_x_test = np.array(X_test), np.array(C_test), np.array(Theta_test), np.array(phi_x_test)
            R_test = ryxz(Theta_test, _validate=False)
            X_test_rot = np.zeros((*X_test.shape[:2], 2, 3))
            X_test_rot[:,:,0,:] = X_test
            X_test_rot[:,:,1,:] = (X_test - C_test[:,np.newaxis,:]) @ R_test        # np.einsum('nlk,nil->nik', R_test, X_test - C_test[:,np.newaxis,:])
            X_model_test = (W / np.tan(phi_x_test / 2)[:,np.newaxis,np.newaxis,np.newaxis] * X_test_rot[:,:,:,:2] / X_test_rot[:,:,:,2][:,:,:,np.newaxis] + WH) / 2.0
            X_pix_center = X_pix[ij_subset] + 0.5
            X_diff = X_model_test - X_pix_center
            E_matrix[success_ids, k] = np.sum(X_diff**2, axis=(1, 2, 3))

        E_array = np.sum(E_matrix / i_subset_sizes, axis=1)
        phi_x_opt = phi_x_array[np.argmin(E_array)]
        print(f'{phi_x_opt = }')
        phi_x_array, h = np.linspace(phi_x_opt - h, phi_x_opt + h, 51, retstep=True)
        E_diff = np.nan_to_num(E_array.max() - E_array.min())

    return phi_x_opt


def get_pair_subscenes(X_pix, phi_x, W, H, H_diff_bool, max_size=50, normalize=False):
    K = X_pix.shape[1]
    X_visible = X_pix[:,:,0] >= 0

    j_subset_list = []
    j_subset_all = np.zeros(0)  # индексы всех просмотренных кадров
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
            if success:
                j_.append(j)
                step = 0
            found_next = False
            while not found_next:
                if iter % 2 == 0: step += 1
                else: step -= 1
                if (iter % 2 == 0 and j + step >= K - 1) or (iter % 2 == 1 and j + step <= 0):
                    done = True
                    break
                if j + step in j_subset_all:
                    continue
                found_next = H_diff_bool[j, j+step]
            if done: break

            print(f'{j = }, {step = }')
            i_subset = np.where(X_visible[:,j] * X_visible[:,j+step])[0]
            print(f'{i_subset.size = }')
            if iter % 2 == 0: ij_subset = np.ix_(i_subset, [j, j+step])
            else: ij_subset = np.ix_(i_subset, [j+step, j])
            F = fundamental_matrix(X_pix[ij_subset], normalize=normalize, W=W, H=H)
            X_, C, Theta, success = get_X_C_Theta_from_F(X_pix[ij_subset], F, W, H, phi_x, ret_status=True, normalized=normalize)
            
            print(f'{success = }')
            if not success: continue

            X = np.full((X_pix.shape[0], 3), np.nan)
            X[i_subset] = X_

            X_pairs.append(X)
            C_pairs.append(C)
            Theta_pairs.append(Theta)

            j += step

        j_ = np.array(j_)
        j_subset_all = np.sort(np.concatenate((j_subset_all, j_)))
        if j_.size > 1:
            X_pairs = np.array(X_pairs)
            C_pairs = np.array(C_pairs)
            Theta_pairs = np.array(Theta_pairs)
            if iter % 2 == 1:
                j_ = j_[::-1]
                X_pairs = X_pairs[::-1]
                C_pairs = C_pairs[::-1]
                Theta_pairs = Theta_pairs[::-1]
            
            j_subset_list.append(j_)
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


def rotate_around_axis(n, cos_phi, sin_phi):
    R1 = np.tile(np.eye(3), (n.shape[0], 1, 1))
    R2 = n[:,:,np.newaxis] @ n[:,np.newaxis,:]
    R3 = np.zeros((n.shape[0], 3, 3))
    R3[:,0,1] = -n[:,2]
    R3[:,0,2] = n[:,1]
    R3[:,1,2] = -n[:,0]
    R3 -= R3.transpose(0, 2, 1)

    return cos_phi[:,np.newaxis,np.newaxis] * R1 + (1.0 - cos_phi[:,np.newaxis,np.newaxis]) * R2 + sin_phi[:,np.newaxis,np.newaxis] * R3


def generate_cameras(K, r_min=0.0, r_max=1.0, phi_min=0.0, phi_max=2*np.pi, theta_min=0.0, theta_max=np.pi, seed=None, sigma=None):
    if seed is not None:
        np.random.seed(seed)
    r_, phi_, theta_ = np.random.uniform([r_min, phi_min, theta_min], [r_max, phi_max, theta_max], (K, 3)).T
    C = r_[:,np.newaxis] * np.column_stack((np.cos(phi_) * np.sin(theta_), np.sin(phi_) * np.sin(theta_), np.cos(theta_)))
    
    n1 = np.column_stack((C[:,1], -C[:,0], np.zeros(K)))
    n1_norm = np.linalg.norm(n1, axis=1)
    n1 /= n1_norm[:,np.newaxis]
    C_norm = np.linalg.norm(C, axis=1)
    cos_psi1 = - C[:,2] / C_norm
    sin_psi1 = n1_norm / C_norm
    R1 = rotate_around_axis(n1, cos_psi1, sin_psi1)

    psi2 = np.arctan(R1[:,1,0] / np.cross(R1[:,:,0], R1[:,:,2])[:,1])
    cos_psi2, sin_psi2 = np.cos(psi2), np.sin(psi2)
    cos_sin_sign = np.sign(R1[:,1,1] * cos_psi2 - np.cross(R1[:,:,1], R1[:,:,2])[:,1] * sin_psi2)
    cos_psi2 *= cos_sin_sign
    sin_psi2 *= cos_sin_sign
    n2 = R1[:,:,2]
    R2 = rotate_around_axis(n2, cos_psi2, sin_psi2)

    R = R2 @ R1

    if sigma is not None:
        C += np.random.normal(0.0, sigma, (K, 3))

    return C, R


def find_bounding_box(A, b):
    # ищется box (необязательно ограниченный), в который можно вписать множество A @ x >= b
    c = np.zeros(3)
    result = linprog(c, A_ub=-A, b_ub=-b, bounds=[(None, None)] * A.shape[1])
    if not result.success:
        raise ValueError('The space is empty')
    
    bounds = np.zeros((3, 2))
    for i in range(3):
        # минимизация x_i
        c_min = np.zeros(3)
        c_min[i] = 1.0
        res_min = linprog(c_min, A_ub=-A, b_ub=-b, bounds=[(None, None)] * A.shape[1])
        
        # максимизация x_i (минимизация -x_i)
        c_max = np.zeros(3)
        c_max[i] = -1.0
        res_max = linprog(c_max, A_ub=-A, b_ub=-b, bounds=[(None, None)] * A.shape[1])

        min_val = res_min.fun if res_min.status == 0 else -np.inf
        max_val = -res_max.fun if res_max.status == 0 else np.inf

        bounds[i,0] = min_val
        bounds[i,1] = max_val
    
    return bounds


def form_view_matrices(C, R, W, H, phi_x):
    ctan_phi_x_2 = 1.0 / np.tan(phi_x / 2.0)
    ctan_phi_y_2 = W / H * ctan_phi_x_2

    A = np.array([[-ctan_phi_x_2, 0.0, 1.0],
                  [ ctan_phi_x_2, 0.0, 1.0],
                  [0.0, -ctan_phi_y_2, 1.0],
                  [0.0,  ctan_phi_y_2, 1.0]])

    AR = (R @ A.T).transpose(0, 2, 1)       # np.einsum('kl,jnl->jkn', A, R)
    ARC = (AR @ C[:,:,np.newaxis])[:,:,0]       #  np.einsum('jkl,jl->jk', AR, C)

    return AR, ARC


def generate_points(N, AR, ARC, view_bounds, gen_bounds, seed):
    bounds_both = np.stack((view_bounds, gen_bounds), axis=0)
    bounds = np.zeros_like(view_bounds)
    bounds[:,0] = bounds_both[:,:,0].max(axis=0)
    bounds[:,1] = bounds_both[:,:,1].min(axis=0)

    N_ = 0
    X = np.zeros((N, 3))
    np.random.seed(seed)
    while N_ < N:
        X_ = np.random.uniform(*bounds.T, (N - N_, 3))
        ARX = np.einsum('jkl,il->ijk', AR, X_)
        ids = np.where((ARX > ARC).all(axis=(1, 2)))[0]
        X[N_:N_+ids.size] = X_[ids]
        N_ += ids.size

    return X