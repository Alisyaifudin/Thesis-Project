from matplotlib import pyplot as plt
from corner import corner
from tqdm import tqdm
import numpy as np

def plot_chain(params, labels, burn=0, figsize=(10, 5), path=None, dpi=100):
    fig, axes = plt.subplots(len(labels), figsize=figsize, sharex=True)
    chain_burn = params[burn:]
    for i in tqdm(range(len(labels))):
        ax = axes[i]
        ax.plot(chain_burn[:, :, i], "k", alpha=0.1)
        ax.set_xlim(0, len(chain_burn)-1)
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    if path is not None:
        fig.savefig(path, dpi=dpi)
    axes[-1].set_xlabel("step number")

def plot_corner(params, labels, burn=0, path=None, dpi=100):
    fig = corner(params[burn:].reshape((-1, len(labels))), labels=labels, 
                 quantiles=[0.16, 0.5, 0.84],
                 show_titles=True, title_fmt=".2f", title_kwargs={"fontsize": 12})
    if path is not None:
        fig.savefig(path, dpi=dpi)

sigma_68_3 = 1
sigma_95_4 = 2
sigma_99_7 = 3

def plot_fit(func, zdata, wdata, chain, ndim, n=50000, alpha=0.2, path=None, dpi=100):
    zmid, znum, zerr = zdata
    wmid, wnum, werr = wdata
    
    flat_samples = chain.reshape((-1, ndim))
    print(flat_samples.shape)
    zs = np.linspace(zmid.min()*1.1, zmid.max()*1.1, 100)
    ws = np.linspace(wmid.min()*1.1, wmid.max()*1.1, 100)
    fzs = np.empty((n, len(zs)))
    fws = np.empty((n, len(ws)))
    for i in tqdm(range(n)):
        ind = np.random.randint(len(flat_samples))
        theta = flat_samples[ind]
        fzs[i] = func.fz(zs, theta ,1.)
        fws[i] = func.fw(ws, theta ,1.)
    fz_log_mean = np.log(fzs).mean(axis=0)
    fz_log_std = np.log(fzs).std(axis=0)
    fz_mean = np.exp(fz_log_mean)

    fw_log_mean = np.log(fws).mean(axis=0)
    fw_log_std = np.log(fws).std(axis=0)
    fw_mean = np.exp(fw_log_mean)

    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    axes[0].errorbar(zmid, znum, yerr=zerr, color='k', alpha=0.5, capsize=2, fmt=".")
    axes[0].fill_between(zs, np.exp(fz_log_mean - sigma_99_7*fz_log_std), np.exp(fz_log_mean + sigma_99_7*fz_log_std), alpha=alpha, color="C0")
    axes[0].fill_between(zs, np.exp(fz_log_mean - sigma_95_4*fz_log_std), np.exp(fz_log_mean + sigma_95_4*fz_log_std), alpha=alpha, color="C0")
    axes[0].fill_between(zs, np.exp(fz_log_mean - sigma_68_3*fz_log_std), np.exp(fz_log_mean + sigma_68_3*fz_log_std), alpha=alpha, color="C0")
    axes[0].plot(zs, fz_mean, c="C0", ls="--")
    axes[0].set_ylabel(r'$\nu(z)$')
    axes[0].set_xlabel(r'$z$ [pc]')
    axes[0].set_xlim(zs.min(), zs.max())
    axes[0].set_ylim(np.exp(fz_log_mean - 3*fz_log_std).min(), np.exp(fz_log_mean + 3*fz_log_std).max()*1.5)
    axes[0].set_yscale('log')
    
    axes[1].errorbar(wmid, wnum, yerr=werr, color='k', alpha=0.5, capsize=2, fmt=".")
    axes[1].fill_between(ws, np.exp(fw_log_mean - 3*fw_log_std), np.exp(fw_log_mean + 3*fw_log_std), alpha=alpha, color="C0")
    axes[1].fill_between(ws, np.exp(fw_log_mean - 2*fw_log_std), np.exp(fw_log_mean + 2*fw_log_std), alpha=alpha, color="C0")
    axes[1].fill_between(ws, np.exp(fw_log_mean - fw_log_std), np.exp(fw_log_mean + fw_log_std), alpha=alpha, color="C0")
    axes[1].plot(ws, fw_mean, c="C0", ls="--")
    axes[1].set_ylabel(r'$f_0(w)$')
    axes[1].set_xlabel(r'$w$ [km/s]]')
    axes[1].set_xlim(ws.min(), ws.max())
    axes[1].set_ylim(np.exp(fw_log_mean - 3*fw_log_std).min(), np.exp(fw_log_mean + 3*fw_log_std).max()*1.5)
    axes[1].set_yscale('log')
    if path is not None:
        fig.savefig(path, dpi=dpi)

def calculate_probs(func, chain, ndim, zdata, wdata, locs, scales, batch=100):
    flat_chain = chain.reshape((-1, ndim))
    probs = func.log_prob_par(flat_chain, zdata, wdata, locs, scales, batch)
    return probs