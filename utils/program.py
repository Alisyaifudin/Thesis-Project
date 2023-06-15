from datetime import datetime
from glob import glob
import sys
import pathlib
from os.path import abspath, join
from time import time
import argparse
import numpy as np 
import sys
current = pathlib.Path(__file__).parent.resolve()
root_dir = abspath(join(current, '..'))
sys.path.append(root_dir)
from .mcmc import mcmc, get_data, Model, get_params, generate_init, calculate_prob
from .plot_mcmc import plot_chain, plot_corner, plot_fit
# initilization
parser = argparse.ArgumentParser(description='Run mcmc in seperate process')


def validate_args(args):
    i = args.data
    try:
        i = int(i)
    except:
        raise ValueError('data must be integer')
    return i


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
    'burn': 1000,
    'step0': 500,
    'step': 2000,
    'it': 3,
    'thin': 20,
    'm': 10,
    'log': True,
    'result_path': None,
    'z_dir_path': None,
    'w_dir_path': None,
    'alpha': 0.01,
    'model': None,
    'nsample': 5_000,
}

bs =  {
    'DM': 5,
    'DDDM': 7,
    'NO': 4
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
        w_dir_path = self.props['w_dir_path']
        w_files = glob(join(w_dir_path, "w*"))
        w_files.sort()
        name = z_files[index].split("/")[-1].replace(".hdf5", "").replace("z_", "")
        w_dir_path = self.props['w_dir_path']
        output_path = join(self.props['result_path'],
                           'data', f'chain-{name}.npy')
        result = mcmc(
            model=self.props['model'],
            z_path=z_files[index],
            w_path=w_files[index],
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
        name = z_files[index].split("/")[-1].replace(".hdf5", "").replace("z_", "")
        chain_path = join(self.props['result_path'],
                          'data', f'chain-{name}.npy')
        chain = np.load(chain_path)
        print(f'\tLoading chain from\n\t{chain_path}')
        output_path = join(self.props['result_path'],
                           'plots', f'chain-{name}.pdf')
        init = generate_init(self.props['model'])
        indexes = init['indexes']
        labs = init['labs']
        labels = init['labels']
        params = get_params(chain, indexes, labs)

        plot_chain(
            params=params,
            labels=labels,
            alpha=0.01, 
            path=output_path
        )
        print(f'\tChain plot saved to {output_path}')

    @timestamp_decorator
    def plot_corner(self, args):
        index = validate_args(args)
        z_dir_path = self.props['z_dir_path']
        z_files = glob(join(z_dir_path, "z*"))
        z_files.sort()
        name = z_files[index].split("/")[-1].replace(".hdf5", "").replace("z_", "")
        chain_path = join(self.props['result_path'],
                          'data', f'chain-{name}.npy')
        chain = np.load(chain_path)
        print(f'\tLoading chain from\n\t{chain_path}')
        output_z_path = join(self.props['result_path'],
                           'plots', f'corner-z-{name}.pdf')
        output_w_path = join(self.props['result_path'],
                           'plots', f'corner-w-{name}.pdf')
        init = generate_init(self.props['model'])
        indexes = init['indexes']
        labs = init['labs']
        labels = init['labels']
        params = get_params(chain, indexes, labs)
        b = bs[self.props['model'].value]
        # z
        plot_corner(
            params=params[:,:,:b],
            labels=labels[:b],
            path=output_z_path
        )
        # w
        plot_corner(
            params=params[:,:,b:-1],
            labels=labels[b:-1],
            path=output_w_path
        )
        print(f'\tCorner plot saved to {output_z_path} and {output_w_path}')

    @timestamp_decorator
    def plot_fit(self, args):
        index = validate_args(args)
        z_dir_path = self.props['z_dir_path']
        z_files = glob(join(z_dir_path, "z*"))
        z_files.sort()
        w_dir_path = self.props['w_dir_path']
        w_files = glob(join(w_dir_path, "w*"))
        w_files.sort()
        name = z_files[index].split("/")[-1].replace(".hdf5", "").replace("z_", "")
        chain_path = join(self.props['result_path'],
                          'data', f'chain-{name}.npy')
        output_path = join(self.props['result_path'],
                           'plots', f'fit-{name}.pdf')
        chain = np.load(chain_path)
        print(f'\tLoading chain from\n\t{chain_path}')
        ndim = chain.shape[2]
        flat_chain = chain.reshape((-1, ndim))

        plot_fit(
            model=self.props['model'],
            flat_chain=flat_chain,
            z_path=z_files[index],
            w_path=w_files[index],
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
        w_dir_path = self.props['w_dir_path']
        w_files = glob(join(w_dir_path, "w*"))
        w_files.sort()
        name = z_files[index].split("/")[-1].replace(".hdf5", "").replace("z_", "")
        chain_path = join(self.props['result_path'],
                          'data', f'chain-{name}.npy')
        chain = np.load(chain_path)
        print(f'\tLoading chain from\n\t{chain_path}')
        output_file = join(self.props['result_path'], f'stats.txt')
        ndim = chain.shape[2]
        flat_chain = chain.reshape((-1, ndim))
        zdata = get_data(z_files[index])
        wdata = get_data(w_files[index])

        calculate_prob(
            model=self.props['model'], 
            zdata=zdata,
            wdata=wdata,
            flat_chain=flat_chain,
            name=name,
            path=output_file,
        )
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
            parser.error("You need to choose a subcommand: run_mcmc, plot_chain, plot_corner, plot_fit, calculate_prob, all")
        
