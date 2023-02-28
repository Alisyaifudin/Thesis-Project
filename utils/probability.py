import numpy as np
from scipy.stats import norm, uniform
from .gravity import nu_mod
from .vvd import fw
import vaex
from os.path import join
from os.path import abspath, pardir

root_dir = abspath(pardir)
root_data_dir = abspath(join(root_dir, "Data"))
data_baryon_dir = join(root_data_dir, "Baryon")

# load baryons components
df_baryon = vaex.open(join(data_baryon_dir, "baryon.hdf5"))

rhos = df_baryon["rho"].to_numpy()  # Msun/pc^3
sigmaz = df_baryon["sigma_z"].to_numpy() # km/s

# prior
def log_prior(theta, locs, scales, norm_list, uni_list):
    pars_list = norm_list+uni_list
    result = 0
    for item in pars_list:
        if item in uni_list:
            result += np.sum(uniform.logpdf(theta[item], loc=locs[item], scale=scales[item]))
        elif item in norm_list:
            result += np.sum(norm.logpdf(theta[item], loc=locs[item], scale=scales[item]))
    return result

# likelihood
def log_likelihood(theta, zdata, wdata):
    zmid, zdens, zerr = zdata
    wmid, wdens, werr = wdata
    Fz = nu_mod(zmid, **theta)
    Fw = fw(wmid, **theta)
    resz = np.sum(norm.logpdf(zdens, loc=Fz, scale=zerr))
    resw = np.sum(norm.logpdf(wdens, loc=Fw, scale=werr))
    return resz + resw

# posterior DM only
def log_posterior_simple_DM(theta, data, locs, scales, dim, norm_list, uni_list):
    zdata, wdata = data
    theta_dict = dict(
        rhos=rhos,
        sigmaz=sigmaz,
        log_rhoDM=theta[0],
        log_nu0=theta[1],
        zsun=theta[2],
        R=theta[3],
        w0=theta[4:4+dim],
        log_sigma_w=theta[4+dim:4+2*dim],
        a=theta[4+2*dim:4+3*dim]
    )
    log_prior_ = log_prior(theta_dict, locs, scales, norm_list, uni_list)
    if not np.isfinite(log_prior_):
        return -np.inf

    theta_dict['rhoDM'] = np.exp(theta_dict['log_rhoDM'])
    theta_dict['sigmaDD'] = 0
    theta_dict['hDD'] = 1
    theta_dict['nu0'] = np.exp(theta_dict['log_nu0'])
    theta_dict['sigma_w'] = np.exp(theta_dict['log_sigma_w'])
    
    return log_prior_ + log_likelihood(theta_dict, zdata, wdata)


def log_posterior_dm(theta, data, locs, scales, dim, norm_list, uni_list):
    zdata, wdata = data
    theta_dict = dict(
        rhos=theta[:12],
        sigmaz=theta[12:24],
        log_rhoDM=theta[24],
        log_nu0=theta[25],
        zsun=theta[26],
        R=theta[27],
        w0=theta[28:28+dim],
        log_sigma_w=theta[28+dim:28+2*dim],
        a=theta[28+2*dim:28+3*dim]
    )
    log_prior_ = log_prior(theta_dict, locs, scales, norm_list, uni_list)
    if not np.isfinite(log_prior_):
        return -np.inf

    theta_dict['rhoDM'] = np.exp(theta_dict['log_rhoDM'])
    theta_dict['sigmaDD'] = 0
    theta_dict['hDD'] = 1
    theta_dict['nu0'] = np.exp(theta_dict['log_nu0'])
    theta_dict['sigma_w'] = np.exp(theta_dict['log_sigma_w'])
    
    return log_prior_ + log_likelihood(theta_dict, zdata, wdata)