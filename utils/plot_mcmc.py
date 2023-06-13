from matplotlib import pyplot as plt
from corner import corner
from tqdm import tqdm
import numpy as np
from hammer import vel
import hammer
from scipy.stats import norm
import sys
from .mcmc import get_data
import numpy as np
from glob import glob
from os.path import join


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


class plot_chain:
    """required: 
            params (from `get_params`),
            labels
        optional:
            burn = 0
            figsize = (10, 10)
            path = None
            alpha = 0.1
            dpi = 70
        .run to run the program"""

    def __init__(self):
        self.props = {
            "params": False,
            "labels": False
        }
        self._burn = 0
        self._figsize = (10, 10)
        self._path = None
        self._alpha = 0.1
        self._dpi = 70

    def params(self, p):
        self._params = p
        self.props['params'] = True
        return self

    def labels(self, p):
        self._labels = p
        self.props['labels'] = True
        return self

    def burn(self, p):
        self._burn = p
        return self

    def figsize(self, p):
        self._figsize = p
        return self

    def path(self, p):
        self._path = p
        return self

    def alpha(self, p):
        self._alpha = p
        return self

    def dpi(self, p):
        self._dpi = p
        return self

    def run(self):
        if not all(self.props.values()):
            eprint("props: ", self.props)
            raise ValueError("all requirement must be set first")
        fig, axes = plt.subplots(
            len(self._labels), figsize=self._figsize, sharex=True)
        chain_burn = self._params[self._burn:]
        for i, label in tqdm(enumerate(self._labels)):
            ax = axes[i]
            ax.plot(chain_burn[:, :, i], "k", alpha=self._alpha)
            ax.set_xlim(0, len(chain_burn)-1)
            ax.set_ylabel(label)
            ax.yaxis.set_label_coords(-0.1, 0.5)
        if self._path is not None:
            fig.savefig(self._path, dpi=self._dpi)
        axes[-1].set_xlabel("step number")
        plt.show()


class plot_corner:
    """required: 
            params (from `get_params`),
            labels
        optional:
            burn = 0
            path = None
            dpi = 70
            truths = None (list of the real values)
        .run to run the program"""

    def __init__(self):
        self.props = {
            "params": False,
            "labels": False,
        }
        self._burn = 0
        self._path = None
        self._dpi = 70
        self._truths = None

    def params(self, p):
        self._params = p
        self.props['params'] = True
        return self

    def labels(self, p):
        self._labels = p
        self.props['labels'] = True
        return self

    def burn(self, p):
        self._burn = p
        return self

    def path(self, p):
        self._path = p
        return self

    def dpi(self, p):
        self._dpi = p
        return self

    def truths(self, p):
        self._truths = p
        return self

    def run(self):
        if not all(self.props.values()):
            eprint("props: ", self.props)
            raise ValueError("all requirement must be set first")
        fig = corner(self._params[self._burn:].
                     reshape((-1, len(self._labels))),
                     labels=self._labels,
                     quantiles=[0.16, 0.5, 0.84],
                     show_titles=True, title_fmt=".2f",
                     title_kwargs={"fontsize": 12},
                     truths=self._truths
                     )
        if self._path is not None:
            fig.savefig(self._path, dpi=self._dpi)
        plt.show()


sigma_68 = 0.994458
sigma_90 = 1.644854
sigma_95 = 1.959964


class plot_fit_w:
    """required: index, w_dir_path, flat
        optional:
            nsample = 5_000
            res = 100
            path = None
            dpi = 70
            alpha = 0.1
            c = C0
        .run to run the program"""

    def __init__(self):
        self.props = {
            "index": False,
            "w_dir_path": False,
            "flat": False,
        }
        self._nsample = 5000
        self._res = 100
        self._path = None
        self._dpi = 70
        self._alpha = 0.1
        self._c = "C0"

    def index(self, p):
        self._index = p
        self.props['index'] = True
        return self

    def w_dir_path(self, p):
        self._w_dir_path = p
        self.props['w_dir_path'] = True
        return self

    def flat(self, p):
        self._flat = p
        self.props['flat'] = True
        return self

    def nsample(self, p):
        self._nsample = p
        return self

    def res(self, p):
        self._res = p
        return self

    def path(self, p):
        self._path = p
        return self

    def dpi(self, p):
        self._dpi = p
        return self

    def alpha(self, p):
        self._alpha = p
        return self

    def c(self, p):
        self._c = p
        return self

    def run(self):
        if not all(self.props.values()):
            eprint("props: ", self.props)
            raise ValueError("all requirement must be set first")
        wdata = get_data(self._w_dir_path, self._index, "w")
        wmid, wnum, werr = wdata
        ws = np.linspace(wmid.min()*1.1, wmid.max()*1.1, self._res)
        fws = np.empty((self._nsample, len(ws)))
        for i in tqdm(range(self._nsample)):
            ind = np.random.randint(len(self._flat))
            theta = self._flat[ind]
            fws[i] = vel.fw(ws, theta)

        fw_log_mean = np.log(fws).mean(axis=0)
        fw_log_std = np.log(fws).std(axis=0)
        fw_mean = np.exp(fw_log_mean)

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.errorbar(wmid, wnum, yerr=werr, color='k',
                    alpha=0.5, capsize=2, fmt=".")
        ax.plot(ws, fw_mean, c=self._c, ls="--")
        ax.fill_between(ws, np.exp(fw_log_mean - sigma_95*fw_log_std),
                        np.exp(fw_log_mean + sigma_95*fw_log_std), alpha=self._alpha, color=self._c)
        ax.fill_between(ws, np.exp(fw_log_mean - sigma_90*fw_log_std),
                        np.exp(fw_log_mean + sigma_90*fw_log_std), alpha=self._alpha, color=self._c)
        ax.fill_between(ws, np.exp(fw_log_mean - sigma_68*fw_log_std),
                        np.exp(fw_log_mean + sigma_68*fw_log_std), alpha=self._alpha, color=self._c)
        ax.set_ylabel(r'$f_0(w)$')
        ax.set_xlabel(r'$w$ [km/s]')
        ax.set_xlim(ws.min(), ws.max())
        ax.set_ylim(0)
        # ax.set_yscale("log")
        # ax.set_ylim(np.exp(fw_log_mean - sigma_95*fw_log_std).min(),
        #             np.exp(fw_log_mean + sigma_95*fw_log_std).max()*1.5)

        if self._path is not None:
            fig.savefig(self._path, dpi=self._dpi)
        plt.show()


class plot_fit_z:
    """required: index, z_dir_path, phi_dir_path, flat, zmax, model
        optional:
            nsample = 5_000
            res = 100
            path = None
            dpi = 70 
        .run to run the program"""
    models = ["DM", "DDDM", "no"]
    funcs = {
        "DM": hammer.dm,
        "DDDM": hammer.dddm,
        "no": hammer.no
    }

    def __init__(self):
        self.props = {
            "index": False,
            "z_dir_path": False,
            "phi_dir_path": False,
            "flat": False,
            "zmax": False,
            "model": False
        }
        self._nsample = 5000
        self._res = 100
        self._path = None
        self._dpi = 70

    def model(self, p):
        if not p in self.models:
            raise ValueError("model must be 'DM', 'DDDM', or 'no'")
        self._model = p
        if p == "DM":
            self.i_zsun = 27
            self.i_log_nu0 = 25
        if p == "DDDM":
            self.i_zsun = 29
            self.i_log_nu0 = 27
        if p == "no":
            self.i_zsun = 26
            self.i_log_nu0 = 24
        self.props['model'] = True
        self.func = self.funcs[p]
        return self

    def index(self, p):
        self._index = p
        self.props['index'] = True
        return self

    def z_dir_path(self, p):
        self._z_dir_path = p
        self.props['z_dir_path'] = True
        return self

    def phi_dir_path(self, p):
        self._phi_dir_path = p
        self.props['phi_dir_path'] = True
        return self

    def flat(self, p):
        self._flat = p
        self.props['flat'] = True
        return self

    def zmax(self, p):
        self._zmax = p
        self.props['zmax'] = True
        return self

    def nsample(self, p):
        self._nsample = p
        return self

    def res(self, p):
        self._res = p
        return self

    def path(self, p):
        self._path = p
        return self

    def dpi(self, p):
        self._dpi = p
        return self

    def run(self):
        if not all(self.props.values()):
            eprint("props: ", self.props)
            raise ValueError("all requirement must be set first")
        files = glob(join(self._phi_dir_path, "phi*"))
        files.sort()
        file = files[self._index]
        name = file.split("/")[-1].replace(".npy", "")
        phis = np.load(file)
        pred = np.load(file.replace("phi", "pred"))
        zdata = get_data(self._z_dir_path, self._index, "z")
        zmid, znum, zerr = zdata
        zn = []
        for _ in tqdm(range(self._nsample)):
            th = self._flat[np.random.randint(len(self._flat))]
            zsun = th[self.i_zsun]
            log_nu0 = th[self.i_log_nu0]
            nu0 = np.exp(log_nu0)
            zt = self.func.phi_invers(phis, th, self._zmax*1.5, dz=0.5)
            zrel = zt + zsun
            for i, zt_i in enumerate(zrel):
                pred_i = pred[i]
                znum_mod, znum_weight = pred_i
                sgn = np.random.choice([-1, 1])
                zt_i = zt_i*sgn - zsun
                zn.append((zt_i, [np.log(nu0)+np.log(znum_mod), znum_weight]))
        z_edge = np.linspace(zmid.min()*1.1, zmid.max()*1.1, self._res)
        z_mid = (z_edge[1:]+z_edge[:-1])/2
        ynum = np.empty((2, len(z_edge)-1))
        for i, (z0, z1) in enumerate(zip(tqdm(z_edge[:-1]), z_edge[1:])):
            ys = np.array([zn_i[1][0] for zn_i in zn if (
                zn_i[0] < z1) and (zn_i[0] > z0)]).flatten()
            weights = np.array([zn_i[1][1] for zn_i in zn if (
                zn_i[0] < z1) and (zn_i[0] > z0)]).flatten()
            yn, yedge = np.histogram(
                ys, bins=50, weights=weights, density=True)
            if any(np.isnan(yn)):
                ynum[:, i] = [0, 1]
                continue
            ymid = (yedge[1:]+yedge[:-1])/2
            mu = np.sum(ymid*yn)/np.sum(yn)
            sigma = np.sqrt(np.sum(yn*(ymid-mu)**2)/np.sum(yn))
            ynum[:, i] = [mu, sigma]
        ymid = ynum[0]
        yerr = ynum[1]

        plt.figure(figsize=(10, 5))
        plt.errorbar(zmid, znum, yerr=zerr, fmt='.', color='k', alpha=0.5)
        plt.fill_between(z_mid, np.exp(ymid-yerr),
                         np.exp(ymid+yerr), alpha=0.2, color='C0')
        plt.fill_between(z_mid, np.exp(ymid-2*yerr),
                         np.exp(ymid+2*yerr), alpha=0.2, color='C0')
        plt.fill_between(z_mid, np.exp(ymid-3*yerr),
                         np.exp(ymid+3*yerr), alpha=0.2, color='C0')
        plt.xlabel(r"$z$ (pc)")
        plt.ylabel(r"$\nu(z)/\nu_0$")
        plt.title(name)
        if self._path is not None:
            plt.savefig(self._path, dpi=self._dpi)
            print("saved ", self._path)
        plt.show()
