from datetime import datetime
from glob import glob
import pathlib
from os.path import abspath, join
from time import time
import argparse
import numpy as np
import vaex
import re
import json
import sys
current = pathlib.Path(__file__).parent.resolve()
root_dir = abspath(join(current, '..', '..'))
sys.path.append(root_dir)
from utils import plot_chain, plot_corner, plot_fit, mcmc, get_data_w, get_data_z, get_params, generate_init, predictive_posterior, Model, concat, bayes_factor
root_data_dir = join(root_dir, 'Data')
# initilization
parser = argparse.ArgumentParser(description='Run mcmc in seperate process')

name = "Baryon"
baryon_dir = join(root_data_dir, name)
# load baryons components
df_baryon = vaex.open(join(baryon_dir, "baryon.hdf5"))
rhob = np.array(df_baryon["rho"].to_numpy())  # Msun/pc^3
sigmaz = df_baryon["sigma_z"].to_numpy()  # km/s
e_rhob = np.array(df_baryon["e_rho"].to_numpy())  # Msun/pc^3
e_sigmaz = np.array(df_baryon["e_sigma_z"].to_numpy())  # km/s
baryon = np.array([concat(rhob, sigmaz), concat(e_rhob, e_sigmaz)])


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
        print("==========================================")
        print(f"[{datetime.now()}] Starting {func.__name__} for index {i}")
        result = func(*args)
        print(
            f"[{datetime.now()}] Finished {func.__name__} in {np.round(time() - t0, 2)} seconds")
        print("******************************************")
        return result
    return wrapper


default_props = {
    'burn': 500,
    'step0': 500,
    'step': 1_000,
    'it': 2,
    'thin': 10,
    'chain_long': 100,
    'm': 10,
    'log': True,
    'alpha': 0.01,
    'nsample': 20_000,
    'model': None,
    'result_path': None,
    'z_path': None,
    'w_path': None,
}

bs = {
    Model.DM.name: 6,
    Model.DDDM.name: 8,
    Model.NO.name: 5
}

result_paths = {
    'DM': join(root_data_dir, 'MCMC-no', 'tes'),
    'DDDM': join(root_data_dir, 'MCMC-no', 'tes'),
    'NO': join(root_data_dir, 'MCMC-no', 'tes')
}

z_dir_path = join(root_data_dir, 'Vertical-Number', 'no')
w_dir_path = join(root_data_dir, 'Velocity-Distribution', 'no')

HammerModel = {
    'DM': Model.DM,
    'DDDM': Model.DDDM,
    'NO': Model.NO,
}

def extract_c(name):
    pattern = r"\((-?\d+\.\d+)\)_\((-?\d+\.\d+)\)"
    match = re.match(pattern, name)
    c0, c1 = None, None
    if match:
        c0 = match.group(1)
        c1 = match.group(2)
        c0 = float(c0)  # Convert to float if desired
        c1 = float(c1)  # Convert to float if desired
    return c0, c1

zb = 200

def get_props(args):
    model = args.model
    index = args.data
    try:
        index = int(index)
    except:
        raise ValueError('data must be integer')
    props = default_props.copy()
    if model in ['DM', 'DDDM', 'NO']:
        props['result_path'] = result_paths[model]
    else:
        raise ValueError("invalid model")
    z_files = glob(join(z_dir_path, "z*"))
    z_files.sort()
    w_files = glob(join(w_dir_path, "w*"))
    w_files.sort()
    N = len(z_files)
    if len(z_files) != len(w_files):
        print(f"z files: {len(z_files)}")
        print(f"w files: {len(w_files)}")
        raise ValueError("z and w files not match, check again")
    if (index >= N) or (index < 0):
        raise ValueError("index out of range")
    props['z_path'] = z_files[index]
    props['w_path'] = w_files[index]
    props['model'] = HammerModel[model]
    return props

class Program:
    def __init__(self):
        pass

    @timestamp_decorator
    def run_mcmc(self, args):
        name = self.props['z_path'].split("/")[-1].replace(".hdf5", "").replace("z_", "")
        zdata_ori = get_data_z(self.props['z_path'])
        wdata = get_data_w(self.props['w_path'])
        zmid, znum, comp = zdata_ori
        mask = np.abs(zmid) < zb
        zmid = zmid[mask]
        znum = znum[mask]
        comp = comp[mask]
        zdata = (zmid, znum, comp)
        output_path = join(self.props['result_path'], 'data', f'chain-{name}.npy')

        result = mcmc(
            model=self.props['model'],
            zdata=zdata,
            wdata=wdata,
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
        z_path = self.props['z_path']
        name = z_path.split("/")[-1].replace(".hdf5", "").replace("z_", "")
        zdata = get_data_z(self.props['z_path'])
        wdata = get_data_w(self.props['w_path'])
        chain_path = join(self.props['result_path'], 'data', f'chain-{name}.npy')
        chain = np.load(chain_path)
        length = chain.shape[0]
        m = length//self.props['chain_long']
        m = max(m, 1)
        chain = chain[::m]
        print(f'\tLoading chain from\n\t{chain_path}')
        output_path = join(self.props['result_path'],
                            'plots', f'chain-{name}.pdf')
        log_z_max = np.log(zdata[1].max())
        log_a_max = np.log(wdata[1].max())
        init = generate_init(self.props['model'], log_z_max, log_a_max)
        indexes = init['indexes']
        labs = init['labs']
        labels = init['labels']
        c0, c1 = extract_c(name)
        name = r"${c0}<J-K_s<{c1}$".format(c0=c0, c1=c1)

        params, labels = get_params(chain, indexes, labs, labels)
        plot_chain(
            name=name,
            params=params,
            labels=labels,
            alpha=0.01,
            path=output_path,
        )
        print(f'\tChain plot saved to {output_path}')

    @timestamp_decorator
    def plot_corner(self, args):
        z_path = self.props['z_path']
        name = z_path.split("/")[-1].replace(".hdf5", "").replace("z_", "")
        chain_path = join(self.props['result_path'],
                            'data', f'chain-{name}.npy')
        chain = np.load(chain_path) 
        zdata = get_data_z(self.props['z_path'])
        wdata = get_data_w(self.props['w_path'])
        log_z_max = np.log(zdata[1].max())
        log_a_max = np.log(wdata[1].max())
        print(f'\tLoading chain from\n\t{chain_path}')
        output_path_z = join(self.props['result_path'], 'plots', f'corner-z-{name}.pdf')
        output_path_w = join(self.props['result_path'], 'plots', f'corner-w-{name}.pdf')
        init = generate_init(self.props['model'], log_z_max, log_a_max)
        indexes = init['indexes']
        labs = init['labs']
        labels = init['labels']
        params, labels = get_params(chain, indexes, labs, labels)
        b = bs[self.props['model'].name]
        c0, c1 = extract_c(name)
        name = r"${c0}<J-K_s<{c1}$".format(c0=c0, c1=c1)

        plot_corner(
            name=name,
            params=params[:, :, :b],
            labels=labels[:b],
            path=output_path_z
        )
        plot_corner(
            name=name,
            params=params[:, :, b:],
            labels=labels[b:],
            path=output_path_w
        )
        print(f'\tCorner plot saved to {output_path_z} and {output_path_w}')

    @timestamp_decorator
    def plot_fit(self, args):
        z_path = self.props['z_path']
        w_path = self.props['w_path']
        name = z_path.split("/")[-1].replace(".hdf5", "").replace("z_", "")
        zdata = get_data_z(z_path)
        wdata = get_data_w(w_path)
        chain_path = join(self.props['result_path'], 'data', f'chain-{name}.npy')
        output_path = join(self.props['result_path'], 'plots', f'fit-{name}.pdf')
        chain = np.load(chain_path)
        ndim = chain.shape[-1]
        flat_chain = chain.reshape(-1, ndim)
        nsample = np.minimum(self.props['nsample'], flat_chain.shape[0])
        c0, c1 = extract_c(name)
        name = r"${c0}<J-K_s<{c1}$".format(c0=c0, c1=c1)
        print(f'\tLoading chain from\n\t{chain_path}')
        
        plot_fit(
            name=name,
            model=self.props['model'],
            flat_chain=flat_chain,
            zdata=zdata,
            wdata=wdata,
            log=self.props['log'],
            nsample=nsample,
            res=100,
            path=output_path
        )
        print(f'\tFit plot saved to {output_path}')

    @timestamp_decorator
    def calculate_prob(self, args):
        zdata = get_data_z(self.props["z_path"])
        zmid, znum, comp = zdata
        mask = np.abs(zmid) > zb
        zmid_out = zmid[mask]
        znum_out = znum[mask]
        comp_out = comp[mask]
        zdata_out = (zmid_out, znum_out, comp_out)
        zmid_in = zmid[~mask]
        znum_in = znum[~mask]
        comp_in = comp[~mask]
        zdata_in = (zmid_in, znum_in, comp_in)
        wdata = get_data_w(self.props["w_path"])
        name = self.props["z_path"].split(
            "/")[-1].replace(".hdf5", "").replace("z_", "")
        chain_path = join(self.props['result_path'],
                            'data', f'chain-{name}.npy')
        chain = np.load(chain_path)
        ndim = chain.shape[-1]
        flat_chain = chain.reshape(-1, ndim)
        print(f'\tLoading chain from\n\t{chain_path}')
        output_dir = join(self.props['result_path'], 'stats')
        mask = np.abs(zmid) < zb
        zmid_in = zmid[mask]
        znum_in = znum[mask]
        comp_in = comp[mask]
        zdata_in = (zmid_in, znum_in, comp_in)
        log_bf_dm, e_log_bf_dm = bayes_factor(
            model=self.props['model'], 
            flat_chain=flat_chain, 
            zdata=zdata_in,
            wdata=wdata,
            nsample=flat_chain.shape[0],
            batch=100
        )
        stats = {
            'name': name,
            'log_bf': log_bf_dm,
            'e_log_bf': e_log_bf_dm,
            'datetime': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'predictive_posterior': None
        }
        mask = (np.abs(zmid) > zb)
        zmid_out = zmid[mask]
        znum_out = znum[mask]
        comp_out = comp[mask]
        zdata_out = (zmid_out, znum_out, comp_out)
        
        probs = predictive_posterior(
            model=self.props['model'],
            flat_chain=flat_chain,
            zdata=zdata_out,
            nsample=flat_chain.shape[0]
        )
        prob = np.sum(probs*np.log10(np.e))
        stats['predictive_posterior'] = prob
        stat_file = join(output_dir, f'stats-{name}.json')
        with open(stat_file, "w+") as outfile:
            json.dump(stats, outfile)
        print(f'\tProbabilities saved to {stat_file}')

    @timestamp_decorator
    def all(self, args):
        self.run_mcmc(args)
        self.plot_chain(args)
        self.plot_corner(args)
        self.plot_fit(args)
        self.calculate_prob(args)

    def main(self):
        parser = argparse.ArgumentParser(
            description='Run MCMC in seperate process')

        parser.add_argument(
            '-d', '--data', help='data index (check data folder)', required=True)
        parser.add_argument(
            '-m', '--model', help='model (DM, DDDM, NO)', required=True)

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
        self.props = get_props(args)
        try:
            args.func(args)
        except AttributeError:
            parser.error("You need to choose a subcommand: run_mcmc, plot_chain, plot_corner, plot_fit, calculate_prob, all")

if __name__ == '__main__':
    program = Program()
    program.main()