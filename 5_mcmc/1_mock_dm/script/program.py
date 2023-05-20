from datetime import datetime
import sys
import pathlib
from os.path import abspath, join
from hammer import dm
from time import time
import argparse
import numpy as np
import sys
current = pathlib.Path(__file__).parent.resolve()
root_dir = abspath(join(current, '..', '..', '..'))
sys.path.append(root_dir)
from utils import (get_data, get_params, run_mcmc as mcmc,
                   run_calculate_bic_aic, plot_chain as pchain,
                   plot_corner as pcorner, plot_fit as pfit)
# initilization
parser = argparse.ArgumentParser(description='Run mcmc in seperate process')
labels = [r'$\rho_b\times 10^2$', r'$\rho_{\textup{DM}}\times 10^2$', r'$\nu_0$', r'$R\times 10^3$',
          r'$z_{\odot}$', r'$w_0$', r'$\log \sigma_{w1}$', r'$\log a_1$', r'$\log \sigma_{w2}$', r'$\log a_2$']
labs = ['rhob', 'rhoDM', 'log_nu0', 'R', 'zsun', 'w0',
        'log_sigmaw1', 'log_a1', 'log_sigmaw2', 'log_a2']
indexes = [12, 24, 25, 26, 27, 28, 29, 30, 31, 32]
root_data_path = join(root_dir, 'Data', 'MCMC', 'dm_mock')


def timestamp_decorator(func):
    def wrapper(*args, **kwargs):
        t0 = time()
        print(f"[{datetime.now()}] Starting {func.__name__}")
        result = func(*args, **kwargs)
        print(
            f"[{datetime.now()}] Finished {func.__name__} in {np.round(time() - t0, 2)} seconds")
        return result
    return wrapper


def validate_args(args):
    c = args.case
    i = args.data
    if c not in ['z', 'n']:
        raise ValueError('tipe must be "z" or "n"')
    try:
        i = int(i)
    except:
        raise ValueError('data must be integer')
    return c, i


@timestamp_decorator
def run_mcmc(args):
    c, i = validate_args(args)
    zpath = join(root_data_path, 'mock', c)
    wpath = zpath
    data = get_data(zpath, wpath, i)
    output_path = join(root_data_path, 'data', c)
    mcmc(dm, labs, indexes, data, i, steps0=1_500,
         steps=20_000, output_path=output_path)


@timestamp_decorator
def plot_chain(args):
    c, i = validate_args(args)
    chain = np.load(join(root_data_path, 'data', c, f'chain-2-{i}.npy'))
    output_path = join(root_data_path, 'plots', c)
    params = get_params(chain, indexes, labs)
    pchain(params, labels, figsize=(10, 10),
           path=join(output_path, f'chain-2-{i}.pdf'))


@timestamp_decorator
def plot_corner(args):
    c, i = validate_args(args)
    chain = np.load(join(root_data_path, 'data', c, f'chain-2-{i}.npy'))
    output_path = join(root_data_path, 'plots', c)
    params = get_params(chain, indexes, labs)
    pcorner(params, labels, path=join(output_path, f'corner-2-{i}.pdf'))


@timestamp_decorator
def plot_fit(args):
    c, i = validate_args(args)
    chain = np.load(join(root_data_path, 'data', c, f'chain-2-{i}.npy'))
    zpath = join(root_data_path, 'mock', c)
    wpath = zpath
    data = get_data(zpath, wpath, i)
    ndim = chain.shape[2]
    zdata, wdata = data
    pfit(dm, zdata, wdata, chain, ndim, path=join(
        root_data_path, 'plots', c, f'fit-2-{i}.pdf'))


@timestamp_decorator
def calculate_prob(args):
    c, i = validate_args(args)
    chain = np.load(join(root_data_path, 'data', c, f'chain-2-{i}.npy'))
    output_file = join(root_data_path, f'stats-{c}.txt')
    zpath = join(root_data_path, 'mock', c)
    wpath = zpath
    data = get_data(zpath, wpath, i)
    run_calculate_bic_aic(dm, labs, data, i, chain, output_file)


@timestamp_decorator
def all(args):
    run_mcmc(args)
    plot_chain(args)
    plot_corner(args)
    plot_fit(args)
    calculate_prob(args)


def main():
    parser = argparse.ArgumentParser(
        description='Run MCMC in seperate process')

    parser.add_argument('-c', '--case', help='z or n', required=True)
    parser.add_argument(
        '-d', '--data', help='data index (check data folder)', required=True)
    subparsers = parser.add_subparsers(title='Subcommands', dest='subcommand')

    run_mcmc_parser = subparsers.add_parser(
        'run_mcmc', help='run_mcmc program')
    run_mcmc_parser.set_defaults(func=run_mcmc)

    plot_chain_parser = subparsers.add_parser(
        'plot_chain', help='plot_chain program')
    plot_chain_parser.set_defaults(func=plot_chain)

    plot_corner_parser = subparsers.add_parser(
        'plot_corner', help='plot_corner program')
    plot_corner_parser.set_defaults(func=plot_corner)

    plot_fit_parser = subparsers.add_parser(
        'plot_fit', help='plot_fit program')
    plot_fit_parser.set_defaults(func=plot_fit)

    calculate_prob_parser = subparsers.add_parser(
        'calculate_prob', help='calculate_prob program')
    calculate_prob_parser.set_defaults(func=calculate_prob)

    all_parser = subparsers.add_parser(
        'all', help='all program')
    all_parser.set_defaults(func=all)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
