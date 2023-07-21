from typing import List, Tuple
from matplotlib import pyplot as plt
from corner import corner
from tqdm import tqdm 
import numpy as np
import numpy as np
from hammer import Model

def plot_chain(params: np.ndarray, labels: List[str], **options: dict):
    """required:
            params: `ndarray(shape=(nstep, nwalker, nparam))` (from `get_params`) \n
            labels: `List[str]` \n
        options:
            name: `str` = None \n
            burn: `int` = 0 \n
            figsize: `Tuple[int, int]` = (10, 10) \n
            path: `str` = None \n
            alpha: `float` = 0.1 \n
            dpi: `int` = 70 \n
            fig_kw: `Dict` = All additional keyword arguments for `.pyplot.figure`.
            """
    figsize = options.get('figsize', (10, 15))
    name = options.get('name', None)
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
    axes[0].set_title(name, fontsize=16, y=1.15)
    axes[-1].set_xlabel("Jumlah Langkah")
    # fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=dpi)
    plt.show()


def plot_corner(params: np.ndarray, labels: List[str], **options: dict):
    """required: 
            params: `ndarray(shape=(nstep, nwalker, nparam))` (from `get_params`) \n
            labels: `List[str]` \n
        options:
            name: `str` = None \n
            burn: `int` = 0 \n
            path: `str` = None \n
            dpi: `int` = 70 \n
            truths: `List[float]` = None (list of the real values) \n
            corner_kw: `Dict` = All additional keyword arguments for `corner.corner`."""
    
    name = options.get('name', None)
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
    # fig.tight_layout()
    fig.suptitle(name, fontsize=16)
    if path is not None:
        fig.savefig(path, dpi=dpi)
    plt.show()

def plot_fit(
        model: Model, 
        flat_chain: np.ndarray, 
        zdata: Tuple[np.ndarray, np.ndarray, np.ndarray], 
        wdata: Tuple[np.ndarray, np.ndarray], 
        **options: dict):
    """required: 
            model: `Model` = Model.DM \n
            flat_chain: `ndarray(shape=(nsample, nparam))`\n
            zdata: `Tuple[np.ndarray, np.ndarray]` \n
            wdata: `Tuple[np.ndarray, np.ndarray]` \n
            baryon: `ndarray` = [...rhob, ...sigmaz] \n
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
    alpha = options.get('alpha', 0.05)
    c = options.get('c', "C0")
    log = options.get('log', False)
    dpi = options.get('dpi', 70)
    path = options.get('path', None)
    fig_kw  = options.get('fig_kw', {})
    name = options.get('name', None)

    ind = np.random.choice(np.arange(len(flat_chain)), size=nsample, replace=False)
    theta = flat_chain[ind]
    sigma = np.array([68, 90, 95])
    alpha = 1 - sigma/100

    fit = model.fit_data(theta, zdata, wdata, alpha, res=res)
    zmid = zdata[0]
    wmid = wdata[0]
    zs, ws = fit['b']
    zest = fit['zest']
    west = fit['west']
    zmod = fit['zmod']
    wmod = fit['wmod']
    fig, axes = plt.subplots(2, 1, figsize=figsize, **fig_kw)
    for ax, xlabel, ylabel, mid, est, mod, xs in zip(axes, [r"$z$ [pc]", r"$w$ [km/s]"], [r"$\nu(z)/\nu_0$", r"$f_0(w)$"], [zmid, wmid], [zest, west], [zmod, wmod], [zs, ws]):
        mod_mode = mod[3]
        max = mod_mode.max()
        mod /= max
        mod_lows = mod[:3]
        mod_highs = mod[4:]
        mod_highs = mod_highs[::-1]
        est /= max
        est_mode = est[3]
        est_lows = est[:3]
        est_highs = est[4:]
        est_highs = est_highs[::-1]
        ax.plot(xs, mod_mode, c=c, ls='--')
        for est_low, est_high, mod_low, mod_high in zip(est_lows, est_highs, mod_lows, mod_highs):
            ax.errorbar(mid, est_mode, yerr=(est_mode-est_low, est_high-est_mode), fmt='.', capsize=2, c='k', alpha=0.5)
            ax.fill_between(xs, mod_high, mod_low, alpha=alpha, color=c)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if log:
            ax.set_ylim(mod_lows[0].min(), mod_highs[0].max()*1.5)
            ax.set_yscale("log")
        else:
            ax.set_ylim(0)
        ax.set_xlim(xs.min(), xs.max())
    axes[0].set_title(name, fontsize=16, y=1.05)
    if path is not None:
        fig.savefig(path, dpi=dpi)
    plt.show()
