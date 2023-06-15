from typing import List
from matplotlib import pyplot as plt
from corner import corner
from tqdm import tqdm
import numpy as np
import sys
from .mcmc import get_data, Model, func_dict
import numpy as np
from enum import Enum
from hammer import Model as MCMC_Model

sigma_68 = 0.994458
sigma_90 = 1.644854
sigma_95 = 1.959964


def plot_chain(params: np.ndarray, labels: List[str], **options: dict):
    """
    Plot the chains of the parameters.

    Parameters
    ----------
        params: `ndarray(shape=(nstep, nwalker, nparam))` (from `get_params`) \n
        labels: `List[str]` \n
        options:
            burn: `int` = 0 \n
            figsize: `Tuple[int, int]` = (10, 10) \n
            path: `str` = None \n
            alpha: `float` = 0.1 \n
            dpi: `int` = 70 \n
            fig_kw: `Dict` = All additional keyword arguments for `.pyplot.figure`.
            """
    figsize = options.get('figsize', (10, 15))
    path = options.get('path', None)
    alpha = options.get('alpha', 0.1)
    dpi = options.get('dpi', 70)
    burn = options.get('burn', 0)
    fig_kw = options.get('fig_kw', {})

    fig, axes = plt.subplots(len(labels), figsize=figsize, sharex=True, **fig_kw)
    chain_burn = params[burn:]
    for i, label in enumerate(tqdm(labels)):
        ax = axes[i]
        ax.plot(chain_burn[:, :, i], "k", alpha=alpha)
        ax.set_xlim(0, len(chain_burn)-1)
        ax.set_ylabel(label)
        ax.yaxis.set_label_coords(-0.1, 0.5)
    if path is not None:
        fig.savefig(path, dpi=dpi)
    axes[-1].set_xlabel("step number")
    plt.show()


def plot_corner(params: np.ndarray, labels: List[str], **options: dict):
    """required: 
            params: `ndarray(shape=(nstep, nwalker, nparam))` (from `get_params`) \n
            labels: `List[str]` \n
        options:
            burn: `int` = 0 \n
            path: `str` = None \n
            dpi: `int` = 70 \n
            truths: `List[float]` = None (list of the real values) \n
            corner_kw: `Dict` = All additional keyword arguments for `corner.corner`."""
    
    burn = options.get('burn', 0)
    path = options.get('path', None)
    dpi = options.get('dpi', 70)
    truths = options.get('truths', None)
    corner_kw = options.get('corner_kw', {})

    fig = corner(params[burn:]. 
                    reshape((-1, len(labels))), 
                labels=labels,
                quantiles=[0.16, 0.5, 0.84],
                show_titles=True, title_fmt=".2f", 
                title_kwargs={"fontsize": 12},
                truths=truths,
                **corner_kw
                )
    if path is not None:
        fig.savefig(path, dpi=dpi)
    plt.show()

def plot_fit(model: Model, flat_chain: np.ndarray, z_path: str, w_path: str, **options: dict):
    """required: 
            model: `Model` = Model.DM \n
            flat_chain: `ndarray(shape=(nsample, nparam))`\n
            z_path: `str` (path to z data)\n
            w_path: `str` (path to w data)\n
        options:
            res: `int` = 100 \n
            nsample: `int` = 5_000 \n
            figsize: `Tuple[int, int]` = (10, 10) \n
            alpha: `float` = 0.1 \n
            c: `str` = C0 \n
            log: `bool` = False \n
            dpi: `int` = 70 \n
            path: `str` = None \n
            fig_kw: `Dict` = All additional keyword arguments for `.pyplot.figure`.
            """
    res = options.get('res', 100)
    nsample = options.get('nsample', 5_000)
    figsize = options.get('figsize', (10, 10))
    alpha = options.get('alpha', 0.1)
    c = options.get('c', "C0")
    log = options.get('log', False)
    dpi = options.get('dpi', 70)
    path = options.get('path', None)
    fig_kw  = options.get('fig_kw', {})
    func = func_dict.get(model.value, MCMC_Model.DM)

    zdata = get_data(z_path)
    zmid, znum, zerr = zdata
    wdata = get_data(w_path)
    wmid, wnum, werr = wdata
    zs: np.ndarray[np.float64] = np.linspace(zmid.min()*1.1, zmid.max()*1.1, res)
    ws: np.ndarray[np.float64] = np.linspace(wmid.min()*1.1, wmid.max()*1.1, res)
    log_fws = np.empty((nsample, len(ws)))
    log_fzs = np.empty((nsample, len(zs)))
    for i in tqdm(range(nsample)):
        ind = np.random.randint(len(flat_chain))
        theta = flat_chain[ind]
        log_fws[i] = np.log(func.fw(ws, theta))
        log_fzs[i] = np.log(func.fz(zs, theta))

    fz_log_mean = log_fzs.mean(axis=0)
    fz_log_std = log_fzs.std(axis=0)
    fw_log_mean = log_fws.mean(axis=0)
    fw_log_std = log_fws.std(axis=0)

    fig, axes = plt.subplots(2, 1, figsize=figsize, **fig_kw)
    for ax, label, log_mean, log_std, vs, mid, num, err in zip(axes, ["z", "w"], [fz_log_mean, fw_log_mean], [fz_log_std, fw_log_std], [zs, ws], [zmid, wmid], [znum, wnum], [zerr, werr]):
        ax.errorbar(mid, num, yerr=err, color='k',
                    alpha=1, capsize=2, fmt=".")
        ax.plot(vs, np.exp(log_mean), c=c, ls="--")
        for sigma in [sigma_95, sigma_90, sigma_68]:
            ax.fill_between(vs, np.exp(log_mean - sigma*log_std),
                            np.exp(log_mean + sigma*log_std), alpha=alpha, color=c)
        ax.set_ylabel(r'$f_0({})$'.format(label))
        ax.set_xlabel(r'${}$ [km/s]'.format(label))
        ax.set_xlim(vs.min(), vs.max())
        if log:
            ax.set_yscale("log")
            ax.set_ylim(np.exp(log_mean - sigma_95*log_std).min(),
                        np.exp(log_mean + sigma_95*log_std).max()*1.5)
        else:
            ax.set_ylim(0)
    if path is not None:
        fig.savefig(path, dpi=dpi)
    plt.show()
