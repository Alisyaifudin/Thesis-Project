from .plot_mcmc import plot_chain, plot_corner, plot_fit_z
from .mcmc import mcmc_parallel_z, get_data, get_params, generate_init, calculate_prob
from .concat import concat
from datetime import datetime
from glob import glob
from tqdm import tqdm
import pathlib
from os.path import abspath, join
from time import time
import argparse
import numpy as np
from hammer import Model as Model_MCMC
import vaex
current = pathlib.Path(__file__).parent.resolve()
root_dir = abspath(join(current, '..'))
root_data_dir = join(root_dir, 'Data')
# initilization
parser = argparse.ArgumentParser(description='Run mcmc in seperate process')

name = "Baryon"
baryon_dir = join(root_data_dir, name)
# load baryons components
df_baryon = vaex.open(join(baryon_dir, "baryon.hdf5"))
rhob_mean = df_baryon['rho'].to_numpy()
rhob_err = df_baryon['e_rho'].to_numpy()
sigmaz_mean = df_baryon['sigma_z'].to_numpy()
sigmaz_err = df_baryon['e_sigma_z'].to_numpy()


def validate_args(args):
    i = args.data
    try:
        i = int(i)
    except:
        raise ValueError('data must be integer')
    return i


def get_pars(vel, mask):
    nsample = vel[mask, 0].shape[0]
    psi = np.empty((nsample, 30))

    rhob = np.random.normal(rhob_mean, rhob_err, size=(nsample, 12))
    sigmaz = np.random.normal(sigmaz_mean, sigmaz_err, size=(nsample, 12))
    r = np.random.normal(3.4E-3, 0.6E-3, size=(nsample))
    # random_indices = np.random.choice(np.arange(len(psi)), size=nsample, replace=False)
    w0 = vel[mask, 0]
    log_sigmaw = vel[mask, 1]
    q_sigmaw = vel[mask, 2]
    log_a = vel[mask, 3]
    q_a = vel[mask, 4]

    # combine
    psi[:, :12] = rhob
    psi[:, 12:24] = sigmaz
    psi[:, 24] = r
    psi[:, 25] = w0
    psi[:, 26] = log_sigmaw
    psi[:, 27] = q_sigmaw
    psi[:, 28] = log_a
    psi[:, 29] = q_a

    kin = np.empty((nsample, 4))
    kin[:, 0] = np.exp(log_sigmaw)
    kin[:, 1] = kin[:, 0] / q_sigmaw
    kin[:, 2] = np.exp(log_a)
    kin[:, 3] = kin[:, 2] * q_a
    atot = kin[:, 2] + kin[:, 3]
    kin[:, 2] = kin[:, 2] / atot
    kin[:, 3] = kin[:, 3] / atot
    return psi, kin

def get_pot_b(zdata, nsample, psi):
    rhoDM = 0.
    log_nu0 = 0.
    zsun = 0.

    theta = concat(rhoDM, log_nu0, zsun)
    zmid = zdata[0]
    zrange = zmid.max() - zmid.min()
    z_b = np.linspace(zmid.min()-zrange/2, zmid.max()+zrange/2, 3000)

    pot_b = np.empty((nsample, len(z_b)))

    for i in tqdm(range(nsample)):
        ps = psi[i]
        pot_b[i] = Model_MCMC.DM.potential(z_b, theta, ps, dz=0.5)
    return pot_b, z_b

def timestamp_decorator(func):
    def wrapper(*args, **kwargs):
        t0 = time()
        i = validate_args(args[1])
        print(f"[{datetime.now()}] Starting {func.__name__} for index {i}")
        result = func(*args, **kwargs)
        print(
            f"[{datetime.now()}] Finished {func.__name__} in {np.round(time() - t0, 2)} seconds")
        return result
    return wrapper


default_props = {
    'burn': 100,
    'step0': 100,
    'step': 100,
    'it': 1,
    'thin': 2,
    'm': 10,
    'log': True,
    'result_path': None,
    'z_dir_path': None,
    'vel_dir_path': None,
    'alpha': 0.01,
    'model': None,
    'nsample': 10_000,
    'mask': range(0, 60_000, 100),
}

class Program:
    def __init__(self):
        self.props = default_props
        print('Program initilized')

    def add(self, key, val):
        self.props[key] = val

    def ready(self):
        return None not in self.props.values()

    @timestamp_decorator
    def run_mcmc(self, args):
        index = validate_args(args)
        z_dir_path = self.props['z_dir_path']
        z_files = glob(join(z_dir_path, "z*"))
        z_files.sort()
        zdata = get_data(z_files[index])
        vel_dir_path = self.props['vel_dir_path']
        vel_files = glob(join(vel_dir_path, "*.npy"))
        vel_files.sort()
        vel = np.load(vel_files[index])
        name = z_files[index].split(
            "/")[-1].replace(".hdf5", "").replace("z_", "")
        output_path = join(self.props['result_path'],
                           'data', f'chain-{name}.npy')
        
        psi, kin = get_pars(vel, self.props['mask'])
        nsample = len(self.props['mask'])
        pot_b, z_b = get_pot_b(zdata, nsample, psi)
        result = mcmc_parallel_z(
            model=self.props['model'],
            z_path=z_files[index],
            kin=kin,
            pot_b=pot_b,
            z_b=z_b,
            step0=self.props['step0'],
            step=self.props['step'],
            burn=self.props['burn'],
            it=self.props['it'],
            thin=self.props['thin'],
            m=self.props['m'],
        )
        np.save(output_path, result['chain'])
        print(f'\tChain saved to {output_path}')

    @timestamp_decorator
    def plot_chain(self, args):
        index = validate_args(args)
        z_dir_path = self.props['z_dir_path']
        z_files = glob(join(z_dir_path, "z*"))
        z_files.sort()
        name = z_files[index].split(
            "/")[-1].replace(".hdf5", "").replace("z_", "")
        chain_path = join(self.props['result_path'],
                          'data', f'chain-{name}.npy')
        chains = np.load(chain_path)
        _, step, _, ndim = chains.shape
        chain = np.transpose(chains, (1, 0, 2, 3)).reshape((step, -1, ndim))
        print(f'\tLoading chain from\n\t{chain_path}')
        output_path = join(self.props['result_path'],
                           'plots', f'chain-{name}.pdf')
        init = generate_init(self.props['model'])
        indexes = init['indexes']
        labs = init['labs']
        labels = init['labels']

        params = get_params(chain, indexes, labs)

        plot_chain(
            params=params[:, ::10],
            labels=labels,
            alpha=0.01,
            path=output_path,
            figsize=(10, ndim*2)
        )
        print(f'\tChain plot saved to {output_path}')

    @timestamp_decorator
    def plot_corner(self, args):
        index = validate_args(args)
        z_dir_path = self.props['z_dir_path']
        z_files = glob(join(z_dir_path, "z*"))
        z_files.sort()
        name = z_files[index].split(
            "/")[-1].replace(".hdf5", "").replace("z_", "")
        chain_path = join(self.props['result_path'],
                          'data', f'chain-{name}.npy')
        chains = np.load(chain_path)
        _, step, _, ndim = chains.shape
        chain = np.transpose(chains, (1, 0, 2, 3)).reshape((step, -1, ndim))
        print(f'\tLoading chain from\n\t{chain_path}')
        output_path = join(self.props['result_path'],
                             'plots', f'corner-{name}.pdf')
        init = generate_init(self.props['model'])
        indexes = init['indexes']
        labs = init['labs']
        labels = init['labels']
        params = get_params(chain, indexes, labs)
        plot_corner(
            params=params,
            labels=labels,
            path=output_path
        )
        print(f'\tCorner plot saved to {output_path}')

    @timestamp_decorator
    def plot_fit(self, args):
        index = validate_args(args)
        z_dir_path = self.props['z_dir_path']
        z_files = glob(join(z_dir_path, "z*"))
        z_files.sort()
        zdata = get_data(z_files[index])
        vel_dir_path = self.props['vel_dir_path']
        vel_files = glob(join(vel_dir_path, "*.npy"))
        vel_files.sort()
        vel = np.load(vel_files[index])
        name = z_files[index].split(
            "/")[-1].replace(".hdf5", "").replace("z_", "")
        output_path = join(self.props['result_path'],
                           'data', f'chain-{name}.npy')
        
        psi, _ = get_pars(vel, self.props['mask'])
        chain_path = join(self.props['result_path'],
                          'data', f'chain-{name}.npy')
        output_path = join(self.props['result_path'],
                           'plots', f'fit-{name}.pdf')
        chains = np.load(chain_path)
        n_mcmc, _, _, ndim = chains.shape
        flat_chains = chains.reshape(n_mcmc, -1, ndim)
        print(f'\tLoading chain from\n\t{chain_path}')

        plot_fit_z(
            model=self.props['model'],
            flat_chains=flat_chains,
            zdata=zdata,
            psi=psi,
            log=self.props['log'],
            nsample=self.props['nsample'],
            res=100,
            path=output_path
        )
        print(f'\tFit plot saved to {output_path}')

    @timestamp_decorator
    def calculate_prob(self, args):
        index = validate_args(args)
        z_dir_path = self.props['z_dir_path']
        z_files = glob(join(z_dir_path, "z*"))
        z_files.sort()
        zdata = get_data(z_files[index])
        vel_dir_path = self.props['vel_dir_path']
        vel_files = glob(join(vel_dir_path, "*.npy"))
        vel_files.sort()
        vel = np.load(vel_files[index])
        psi, kin = get_pars(vel, self.props['mask'])
        nsample = len(self.props['mask'])
        pot_b, z_b = get_pot_b(zdata, nsample, psi)
        name = z_files[index].split(
            "/")[-1].replace(".hdf5", "").replace("z_", "")
        chain_path = join(self.props['result_path'],
                          'data', f'chain-{name}.npy')
        chains = np.load(chain_path)
        n_mcmc, _, _, ndim = chains.shape
        flat_chains = chains.reshape(n_mcmc, -1, ndim)
        print(f'\tLoading chain from\n\t{chain_path}')
        output_file = join(self.props['result_path'], 'data', f'prob-{name}.hdf5')

        df = calculate_prob(
            model=self.props['model'],
            zdata=zdata,
            flat_chains=flat_chains,
            kin=kin,
            pot_b=pot_b,
            z_b=z_b,
        )
        df.export(output_file, progress=True)
        print(f'\tProbabilities saved to {output_file}')

    @timestamp_decorator
    def all(self, args):
        self.run_mcmc(args)
        self.plot_chain(args)
        self.plot_corner(args)
        self.plot_fit(args)
        self.calculate_prob(args)

    def main(self):
        if not self.ready():
            for k, v in self.props.items():
                if v is None:
                    print(f'{k} is not set')
            raise ValueError('Program not ready yet')
        parser = argparse.ArgumentParser(
            description='Run MCMC in seperate process')

        parser.add_argument(
            '-d', '--data', help='data index (check data folder)', required=True)
        subparsers = parser.add_subparsers(
            title='Subcommands', dest='subcommand')

        run_mcmc_parser = subparsers.add_parser(
            'run_mcmc', help='run_mcmc program')
        run_mcmc_parser.set_defaults(func=self.run_mcmc)

        plot_chain_parser = subparsers.add_parser(
            'plot_chain', help='plot_chain program')
        plot_chain_parser.set_defaults(func=self.plot_chain)

        plot_corner_parser = subparsers.add_parser(
            'plot_corner', help='plot_corner program')
        plot_corner_parser.set_defaults(func=self.plot_corner)

        plot_fit_parser = subparsers.add_parser(
            'plot_fit', help='plot_fit program')
        plot_fit_parser.set_defaults(func=self.plot_fit)

        calculate_prob_parser = subparsers.add_parser(
            'calculate_prob', help='calculate_prob program')
        calculate_prob_parser.set_defaults(func=self.calculate_prob)

        all_parser = subparsers.add_parser(
            'all', help='all program')
        all_parser.set_defaults(func=self.all)

        args = parser.parse_args()

        try:
            args.func(args)
        except AttributeError:
            parser.error(
                "You need to choose a subcommand: run_mcmc, plot_chain, plot_corner, plot_fit, calculate_prob, all")
