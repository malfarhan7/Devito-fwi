r"""
Acoustic 3D FWI(VP) with entire data

This example shows how to perform acoustic 3D FWI in a distributed manner using
MPI4py. It mirrors the 2D ``AcousticVel_L2_1stage.py`` script and uses the
3D Overthrust velocity model.

Run as:
    export DEVITO_LANGUAGE=openmp
    export DEVITO_MPI=0
    export OMP_NUM_THREADS=6
    export MKL_NUM_THREADS=6
    export NUMBA_NUM_THREADS=6
    mpiexec -n 9 python AcousticVel3D_L2_1stage.py

Picking ``-n`` equal to the total number of shots (here 9 = 3x3) gives one
shot per rank — the cleanest mapping. Fewer ranks works too; the script
uses ``local_split`` to partition shots evenly.

Memory note: 3D FWI stores the forward wavefield to compute the adjoint,
which is much larger than in 2D. We use ``factor=4`` on the inversion
engine to subsample the saved wavefield, cutting memory ~4x at a small
accuracy cost. Raise ``factor`` (or enable ``checkpointing=True``) if
your nodes run out of memory.
"""

import os
import numpy as np
import time

from matplotlib import pyplot as plt
from mpi4py import MPI
from pylops.basicoperators import Identity
from pylops_mpi.DistributedArray import local_split, Partition

from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize
from devito import configuration

from devitofwi.waveengine.acoustic3d import AcousticWave3D
from devitofwi.loss.l2 import L2
from devitofwi.postproc.acoustic import PostProcessVP
from devitofwi.visual.volume import plot_slices


comm = MPI.COMM_WORLD
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

configuration['log-level'] = 'ERROR'

# Path to save figures
figpath = './figs/AcousticVel3D_L2_1stage'

if rank == 0:
    os.makedirs(figpath, exist_ok=True)


# Callback to track model error and save a checkpoint plot of a depth slice
def fwi_callback(xk, vp, vp_error):
    vp_error.append(np.linalg.norm((xk - vp.reshape(-1)) / vp.reshape(-1)))

    if rank == 0:
        last_loss = ainv.losshistory[-1] if ainv.losshistory else float('nan')
        print(f'iter {len(vp_error):3d}: loss={last_loss:.4e}, err={vp_error[-1]:.4e}',
              flush=True)

        xkv = xk.reshape(vp.shape)
        iz = vp.shape[2] // 3
        plt.figure(figsize=(7, 6))
        plt.imshow(xkv[:, :, iz].T, vmin=m_vmin, vmax=m_vmax, cmap='jet',
                   origin='lower', extent=(x[0], x[-1], y[0], y[-1]))
        plt.colorbar(label='Vp [km/s]')
        plt.title(f'Inverted VP (iter {len(vp_error)}, depth slice iz={iz})')
        plt.tight_layout()
        plt.savefig(os.path.join(figpath, 'InvertedVPtmp.png'))
        plt.close('all')


if rank == 0:
    print(f'Distributed 3D FWI ({size} ranks)')


##################################################################
# Parameters
##################################################################

# Model and acquisition parameters (in km, s, and Hz units)
par = {
    # Model cube (extracted from full Overthrust after downsampling x2)
    'nx': 75,  'dx': 0.050,  'ox': 0.0,
    'ny': 75,  'dy': 0.050,  'oy': 0.0,
    'nz': 50,  'dz': 0.050,  'oz': 0.0,
    # Shot grid (areal, near-surface)
    'nsx': 3,  'dsx': 1.25,  'osx': 0.50,
    'nsy': 3,  'dsy': 1.25,  'osy': 0.50,
    'sz': 0.05,
    # Receiver grid (areal, surface)
    'nrx': 20, 'drx': 0.18,  'orx': 0.05,
    'nry': 20, 'dry': 0.18,  'ory': 0.05,
    'rz': 0.05,
    # Time and source
    'tn': 3.0,
    'freq': 8.0,
}

# Modelling parameters
shape = (par['nx'], par['ny'], par['nz'])
spacing = (par['dx'], par['dy'], par['dz'])
origin = (par['ox'], par['oy'], par['oz'])
space_order = 4
nbl = 15
factor = 4   # save-wavefield subsampling for adjoint pass

# Velocity model
path = '../../data/'
velocity_file = path + 'overthrust/overthrust.vites'

##################################################################
# Acquisition set-up
##################################################################

# Axes
x = np.arange(par['nx']) * par['dx'] + par['ox']
y = np.arange(par['ny']) * par['dy'] + par['oy']
z = np.arange(par['nz']) * par['dz'] + par['oz']

# Sources: 3x3 areal grid -> (nsrc, 3)
sx_1d = np.arange(par['nsx']) * par['dsx'] + par['osx']
sy_1d = np.arange(par['nsy']) * par['dsy'] + par['osy']
nsrc = par['nsx'] * par['nsy']
x_s = np.zeros((nsrc, 3))
x_s[:, 0] = np.tile(sx_1d, par['nsy'])
x_s[:, 1] = np.repeat(sy_1d, par['nsx'])
x_s[:, 2] = par['sz']

# Receivers: nrx x nry areal grid -> (nrec, 3)
rx_1d = np.arange(par['nrx']) * par['drx'] + par['orx']
ry_1d = np.arange(par['nry']) * par['dry'] + par['ory']
nrec = par['nrx'] * par['nry']
x_r = np.zeros((nrec, 3))
x_r[:, 0] = np.tile(rx_1d, par['nry'])
x_r[:, 1] = np.repeat(ry_1d, par['nrx'])
x_r[:, 2] = par['rz']

if rank == 0:
    print(f'{nsrc} sources, {nrec} receivers')

    # Top-down acquisition plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([x[0], x[-1], x[-1], x[0], x[0]],
            [y[0], y[0], y[-1], y[-1], y[0]],
            'k--', lw=1, alpha=0.5, label='Model extent')
    ax.scatter(x_r[:, 0], x_r[:, 1], s=8, c='steelblue',
               label=f'Receivers ({nrec})')
    ax.scatter(x_s[:, 0], x_s[:, 1], s=80, c='red', marker='*',
               edgecolors='black', linewidths=0.5, label=f'Sources ({nsrc})')
    ax.set_xlabel('Inline x [km]')
    ax.set_ylabel('Crossline y [km]')
    ax.set_title('Acquisition geometry (top-down)')
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(figpath, 'Geometry.png'))
    plt.close('all')

##################################################################
# Velocity model
##################################################################

# Load Overthrust, transpose to (nx, ny, nz), and crop the working cube.
# File is big-endian float32 in (nz, ny, nx) order, 25 m spacing, m/s units.
vp_raw = np.fromfile(velocity_file, dtype='>f4').astype(np.float32) / 1000.0
vp_raw = vp_raw.reshape(187, 801, 801)
vp_full = np.transpose(vp_raw, (2, 1, 0))     # -> (nx, ny, nz)
del vp_raw

# Crop 150x150x100 then downsample by 2 -> 75x75x50 at 50 m
vp_true = vp_full[200:350, 200:350, :100][::2, ::2, ::2].copy().astype(np.float32)
del vp_full
assert vp_true.shape == shape, f'expected {shape}, got {vp_true.shape}'

if rank == 0:
    m_vmin, m_vmax = np.percentile(vp_true, [2, 98])
    print(f'vp_true range [km/s]: {vp_true.min():.2f} - {vp_true.max():.2f}')
    print(f'percentile clip for plots: {m_vmin:.2f} - {m_vmax:.2f}')

    fig, _ = plot_slices(vp_true, x, y, z, title='True VP',
                         vmin=m_vmin, vmax=m_vmax)
    plt.savefig(os.path.join(figpath, 'TrueVel.png'), dpi=150,
                bbox_inches='tight')
    plt.close('all')
else:
    m_vmin = m_vmax = None

# Broadcast clip range so every rank has m_vmin/m_vmax for downstream plots/bounds
m_vmin = comm.bcast(m_vmin, root=0)
m_vmax = comm.bcast(m_vmax, root=0)

# Initial model: smoothed true, clipped to the velocity range
vp_init = gaussian_filter(vp_true, sigma=4.0).astype(np.float32)
vp_init = np.clip(vp_init, m_vmin, m_vmax)

if rank == 0:
    fig, _ = plot_slices(vp_init, x, y, z, title='Initial VP',
                         vmin=m_vmin, vmax=m_vmax)
    plt.savefig(os.path.join(figpath, 'InitialVel.png'), dpi=150,
                bbox_inches='tight')
    plt.close('all')


##################################################################
# Data
##################################################################

# Split shots across ranks
ns_rank = local_split((nsrc,), MPI.COMM_WORLD, Partition.SCATTER, 0)
ns_ranks = np.concatenate(MPI.COMM_WORLD.allgather(ns_rank))
isin_rank = np.insert(np.cumsum(ns_ranks)[:-1], 0, 0)[rank]
isend_rank = np.cumsum(ns_ranks)[rank]
print(f'Rank: {rank}, ns: {ns_rank}, isin: {isin_rank}, isend: {isend_rank}')

# Modelling engine (forward only, this rank's shots)
amod = AcousticWave3D(
    shape=shape, origin=origin, spacing=spacing,
    src_x=x_s[isin_rank:isend_rank, 0],
    src_y=x_s[isin_rank:isend_rank, 1],
    src_z=x_s[isin_rank:isend_rank, 2],
    rec_x=x_r[:, 0], rec_y=x_r[:, 1], rec_z=x_r[:, 2],
    t0=0.0, tn=par['tn'],
    vp=vp_true,
    src_type='Ricker', f0=par['freq'],
    space_order=space_order, nbl=nbl,
    base_comm=comm,
)

if rank == 0:
    print('Model data...')
dobs, dtobs = amod.mod_allshots()

# Optional noise
sigman = 0
if sigman > 0:
    dobs = dobs + np.random.normal(0, sigman, dobs.shape).astype(np.float32)

if rank == 0:
    # Plot shot gathers from rank 0's shots: reshape receiver axis to (nry, nrx)
    # and show an inline gather at the middle crossline row
    nt = dobs.shape[1]
    t_axis = np.arange(nt) * dtobs
    nshot_local = dobs.shape[0]
    dobs_4d = dobs.reshape(nshot_local, nt, par['nry'], par['nrx'])
    d_vmin, d_vmax = np.percentile(dobs.ravel(), [2, 98])
    ishots_plot = [0, nshot_local // 2, nshot_local - 1]
    iry_mid = par['nry'] // 2

    fig, axs = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax, ishot in zip(axs, ishots_plot):
        ax.imshow(dobs_4d[ishot, :, iry_mid, :], aspect='auto', cmap='gray',
                  extent=[rx_1d[0], rx_1d[-1], t_axis[-1], t_axis[0]],
                  vmin=-d_vmax, vmax=d_vmax)
        ax.set_title(f'Shot {isin_rank + ishot}')
        ax.set_xlabel('Receiver x [km]')
    axs[0].set_ylabel('Time [s]')
    fig.suptitle(f'Observed data (rank 0) — inline gathers at iry={iry_mid}',
                 y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(figpath, 'Data.png'))
    plt.close('all')


##################################################################
# Gradient
##################################################################

# Per-rank L2 loss against this rank's observed data
l2loss = L2(Identity(int(np.prod(dobs.shape[1:]))),
            dobs.reshape(ns_rank[0], -1))

# Inversion engine with factor=4 for the adjoint memory
ainv = AcousticWave3D(
    shape=shape, origin=origin, spacing=spacing,
    src_x=x_s[isin_rank:isend_rank, 0],
    src_y=x_s[isin_rank:isend_rank, 1],
    src_z=x_s[isin_rank:isend_rank, 2],
    rec_x=x_r[:, 0], rec_y=x_r[:, 1], rec_z=x_r[:, 2],
    t0=0.0, tn=par['tn'],
    vprange=(vp_true.min(), vp_true.max()),
    src_type='Ricker', f0=par['freq'],
    space_order=space_order, nbl=nbl,
    factor=factor,
    loss=l2loss,
    base_comm=comm,
)

# First gradient and scaling factor
postproc = PostProcessVP(scaling=1.0)

if rank == 0:
    print('Compute first gradient...')

loss0, direction = ainv._loss_grad(vp_init, postprocess=postproc.apply)
scaling = direction.max()

if rank == 0:
    print(f'first loss: {loss0:.3e}, gradient max (scaling): {scaling:.3e}')

    direction_np = np.asarray(direction)
    glim = np.percentile(np.abs(direction_np), 99)
    fig, _ = plot_slices(direction_np / scaling, x, y, z,
                         vmin=-glim / scaling, vmax=glim / scaling,
                         cmap='seismic',
                         title='First gradient (scaled)',
                         cbar_label='scaled gradient')
    plt.savefig(os.path.join(figpath, 'Gradient.png'), dpi=150,
                bbox_inches='tight')
    plt.close('all')


##################################################################
# FWI
##################################################################

# L-BFGS parameters
ftol = 1e-10
maxiter = 1000
maxfun = 5000
vp_error = []
convertvp = None

# Use the scaling from the first gradient
postproc = PostProcessVP(scaling=scaling)

if rank == 0:
    print('Run FWI...')
    tstart = time.time()

nl = minimize(
    ainv.loss_grad, vp_init.ravel(),
    method='L-BFGS-B', jac=True,
    args=(convertvp, postproc.apply),
    bounds=[(m_vmin, m_vmax)] * vp_init.size,
    callback=lambda x: fwi_callback(x, vp=vp_true, vp_error=vp_error),
    options={'ftol': ftol, 'maxiter': maxiter, 'maxfun': maxfun,
             'disp': True if rank == 0 else False},
)

if rank == 0:
    print('\nTotal time (s) = %.2f' % (time.time() - tstart))
    print('---------------------------------------------------------\n')
    print(nl)

    # Loss and model-error histories
    fig, axs = plt.subplots(1, 2, figsize=(13, 4))
    axs[0].semilogy(ainv.losshistory, 'k', marker='o', ms=3)
    axs[0].set_xlabel('Function evaluation'); axs[0].set_ylabel('L2 loss')
    axs[0].set_title('Loss history')
    axs[0].grid(True, which='both', alpha=0.3)
    axs[1].plot(vp_error, 'k', marker='o', ms=3)
    axs[1].set_xlabel('L-BFGS iteration'); axs[1].set_ylabel('rel. model error')
    axs[1].set_title('Model error history')
    axs[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figpath, 'LossModelError.png'), dpi=150,
                bbox_inches='tight')
    plt.close('all')

    # Final inverted model and difference
    vp_inv = nl.x.reshape(shape).astype(np.float32)

    fig, _ = plot_slices(vp_inv, x, y, z, title='Inverted VP',
                         vmin=m_vmin, vmax=m_vmax)
    plt.savefig(os.path.join(figpath, 'InvertedVP.png'), dpi=150,
                bbox_inches='tight')
    plt.close('all')

    diff = vp_true - vp_inv
    dlim = max(0.5, np.percentile(np.abs(diff), 99))
    fig, _ = plot_slices(diff, x, y, z, title='True - Inverted',
                         vmin=-dlim, vmax=dlim, cmap='seismic',
                         cbar_label='Delta Vp [km/s]')
    plt.savefig(os.path.join(figpath, 'DifferenceVP.png'), dpi=150,
                bbox_inches='tight')
    plt.close('all')

    print(f'mean abs error: {np.abs(diff).mean():.3f} km/s')
    print(f'max abs error:  {np.abs(diff).max():.3f} km/s')
    print(f'relative model error: '
          f'{np.linalg.norm(diff) / np.linalg.norm(vp_true):.4f}')

    # Save the inverted model
    np.save(os.path.join(figpath, 'vp_inv.npy'), vp_inv)
