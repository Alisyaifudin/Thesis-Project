from matplotlib import pyplot as plt
from corner import corner
from tqdm import tqdm
import numpy as np
from hammer import vel


def plot_chain(params, labels, burn=0, figsize=(10, 10), path=None, dpi=100, locs=None, scales=None, alpha=0.1):
    fig, axes = plt.subplots(len(labels), figsize=figsize, sharex=True)
    chain_burn = params[burn:]
    for i, label in tqdm(enumerate(labels)):
        ax = axes[i]
        ax.plot(chain_burn[:, :, i], "k", alpha=alpha)
        if locs is not None and scales is not None:
            if i < 1:
                continue
            ax.axhline(locs[i-1], color='r', ls='--')
            ax.axhline(locs[i-1]+scales[i-1], color='g', ls='--')
        ax.set_xlim(0, len(chain_burn)-1)
        ax.set_ylabel(label)
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


sigma_68 = 0.994458
sigma_90 = 1.644854
sigma_95 = 1.959964


def plot_fit_w(wdata, flat_samples, n=5000, alpha=0.1, log=False, c="C0", path=None, dpi=100):
    wmid, wnum, werr = wdata
    ws = np.linspace(wmid.min()*1.1, wmid.max()*1.1, 100)
    fws = np.empty((n, len(ws)))
    for i in tqdm(range(n)):
        ind = np.random.randint(len(flat_samples))
        theta = flat_samples[ind]
        fws[i] = vel.fw(ws, theta)

    fw_log_mean = np.log(fws).mean(axis=0)
    fw_log_std = np.log(fws).std(axis=0)
    fw_mean = np.exp(fw_log_mean)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.errorbar(wmid, wnum, yerr=werr, color='k',
                alpha=0.5, capsize=2, fmt=".")
    ax.plot(ws, fw_mean, c=c, ls="--")
    ax.fill_between(ws, np.exp(fw_log_mean - sigma_95*fw_log_std),
                    np.exp(fw_log_mean + sigma_95*fw_log_std), alpha=alpha, color=c)
    ax.fill_between(ws, np.exp(fw_log_mean - sigma_90*fw_log_std),
                    np.exp(fw_log_mean + sigma_90*fw_log_std), alpha=alpha, color=c)
    ax.fill_between(ws, np.exp(fw_log_mean - sigma_68*fw_log_std),
                    np.exp(fw_log_mean + sigma_68*fw_log_std), alpha=alpha, color=c)
    ax.set_ylabel(r'$f_0(w)$')
    ax.set_xlabel(r'$w$ [km/s]')
    ax.set_xlim(ws.min(), ws.max())
    if log:
        ax.set_yscale("log")
        ax.set_ylim(np.exp(fw_log_mean - sigma_95*fw_log_std).min(),
                    np.exp(fw_log_mean + sigma_95*fw_log_std).max()*1.5)
    else:
        ax.set_ylim(0)
    if path is not None:
        fig.savefig(path, dpi=dpi)
    plt.show()


def plot_fit(func, zdata, wdata, chain, ndim, n=5000, alpha=0.2, c="C0", path=None, dpi=100):
    zmid, znum, zerr = zdata
    wmid, wnum, werr = wdata

    flat_samples = chain.reshape((-1, ndim))
    # filter nan values
    mask = np.isnan(flat_samples).any(axis=1)
    flat_samples = flat_samples[~mask]
    # filter inf
    mask = np.isinf(flat_samples).any(axis=1)
    flat_samples = flat_samples[~mask]
    print(flat_samples.shape)
    zs = np.linspace(zmid.min()*1.1, zmid.max()*1.1, 100)
    ws = np.linspace(wmid.min()*1.1, wmid.max()*1.1, 100)
    fzs = np.empty((n, len(zs)))
    fws = np.empty((n, len(ws)))
    for i in tqdm(range(n)):
        while True:
            ind = np.random.randint(len(flat_samples))
            theta = flat_samples[ind]
            fz_propose = func.fz(zs, theta, 1.)
            fw_propose = func.fw(ws, theta, 1.)
            if (fw_propose < 0).any() or (fz_propose < 0).any():
                continue
            fzs[i] = fz_propose
            fws[i] = fw_propose
            break
    fz_log_mean = np.log(fzs).mean(axis=0)
    fz_log_std = np.log(fzs).std(axis=0)
    fz_mean = np.exp(fz_log_mean)

    fw_log_mean = np.log(fws).mean(axis=0)
    fw_log_std = np.log(fws).std(axis=0)
    fw_mean = np.exp(fw_log_mean)

    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    axes[0].errorbar(zmid, znum, yerr=zerr, color='k',
                     alpha=0.5, capsize=2, fmt=".")
    axes[0].plot(zs, fz_mean, c=c, ls="--")
    axes[0].fill_between(zs, np.exp(fz_log_mean - sigma_95*fz_log_std), np.exp(
        fz_log_mean + sigma_95*fz_log_std), alpha=alpha, color=c)
    axes[0].fill_between(zs, np.exp(fz_log_mean - sigma_90*fz_log_std),
                         np.exp(fz_log_mean + sigma_90*fz_log_std), alpha=alpha, color=c)
    axes[0].fill_between(zs, np.exp(fz_log_mean - sigma_68*fz_log_std),
                         np.exp(fz_log_mean + sigma_68*fz_log_std), alpha=alpha, color=c)
    axes[0].set_ylabel(r'$\nu(z)$')
    axes[0].set_xlabel(r'$z$ [pc]')
    axes[0].set_xlim(zs.min(), zs.max())
    axes[0].set_ylim(np.exp(fz_log_mean - sigma_95*fz_log_std).min(),
                     np.exp(fz_log_mean + sigma_95*fz_log_std).max()*1.5)
    axes[0].set_yscale('log')

    axes[1].errorbar(wmid, wnum, yerr=werr, color='k',
                     alpha=0.5, capsize=2, fmt=".")
    axes[1].plot(ws, fw_mean, c=c, ls="--")
    axes[1].fill_between(ws, np.exp(fw_log_mean - sigma_95*fw_log_std),
                         np.exp(fw_log_mean + sigma_95*fw_log_std), alpha=alpha, color=c)
    axes[1].fill_between(ws, np.exp(fw_log_mean - sigma_90*fw_log_std),
                         np.exp(fw_log_mean + sigma_90*fw_log_std), alpha=alpha, color=c)
    axes[1].fill_between(ws, np.exp(fw_log_mean - sigma_68*fw_log_std),
                         np.exp(fw_log_mean + sigma_68*fw_log_std), alpha=alpha, color=c)
    axes[1].set_ylabel(r'$f_0(w)$')
    axes[1].set_xlabel(r'$w$ [km/s]')
    axes[1].set_xlim(ws.min(), ws.max())
    axes[1].set_ylim(np.exp(fw_log_mean - sigma_95*fw_log_std).min(),
                     np.exp(fw_log_mean + sigma_95*fw_log_std).max()*1.5)
    axes[1].set_yscale('log')
    if path is not None:
        fig.savefig(path, dpi=dpi)


def calculate_probs(func, chain, ndim, zdata, wdata, locs, scales, batch=100):
    flat_chain = chain.reshape((-1, ndim))
    probs = func.log_prob_par(flat_chain, zdata, wdata, locs, scales, batch)
    return probs
