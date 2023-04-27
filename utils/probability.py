import numpy as np
from scipy.stats import norm, uniform
from .gravity import nu_mod
from .gravity_mond import nu_mond
from .vvd import fw
from .vvd_mond import fw_mond
import vaex
from os.path import join
from os.path import abspath, dirname

current_dir = dirname(abspath(__file__))
root_dir = abspath(join(current_dir, ".."))
root_data_dir = abspath(join(root_dir, "Data"))
data_baryon_dir = join(root_data_dir, "Baryon")

# load baryons components
df_baryon = vaex.open(join(data_baryon_dir, "baryon.hdf5"))

rhos = df_baryon["rho"].to_numpy()  # Msun/pc^3
sigmaz = df_baryon["sigma_z"].to_numpy() # km/s

# ==============================================================
# ==============================================================
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

# ==============================================================
# ==============================================================
# likelihood dm
def log_likelihood_dm(theta, zdata, wdata):
    zmid, zdens, zerr = zdata
    wmid, wdens, werr = wdata
    Fz = nu_mod(zmid, **theta)
    Fw = fw(wmid, **theta)
    resz = np.sum(norm.logpdf(zdens, loc=Fz, scale=zerr))
    resw = np.sum(norm.logpdf(wdens, loc=Fw, scale=werr))
    return resz + resw

def log_likelihood_mond(theta, zdata, wdata):
    zmid, zdens, zerr = zdata
    wmid, wdens, werr = wdata
    Fz = nu_mond(zmid, **theta)
    Fw = fw_mond(wmid, **theta)
    resz = np.sum(norm.logpdf(zdens, loc=Fz, scale=zerr))
    resw = np.sum(norm.logpdf(wdens, loc=Fw, scale=werr))
    return resz + resw
# ==============================================================
# ==============================================================
# posterior dm
def log_posterior_dm(theta, data, locs, scales, dim, norm_list, uni_list, simple=True):
    skip = 0 if simple else 24
    theta_dict = dict()
    zdata, wdata = data
    theta_dict_plus = dict( 
        rhoDM=theta[skip+0],
        log_nu0=theta[skip+1],
        zsun=theta[skip+2],
        R=theta[skip+3],
        w0=theta[skip+4:skip+4+dim],
        log_sigma_w=theta[skip+4+dim:skip+4+2*dim],
        a=theta[skip+4+2*dim:skip+4+3*dim]
    )
    theta_dict.update(theta_dict_plus)
    if simple:
        theta_dict['rhos'] = rhos
        theta_dict['sigmaz'] = sigmaz
    else:   
        theta_dict['rhos'] = theta[:12]
        theta_dict['sigmaz'] = theta[12:24]

    log_prior_ = log_prior(theta_dict, locs, scales, norm_list, uni_list)
    if not np.isfinite(log_prior_):
        return -np.inf
    theta_dict['sigmaDD'] = 0
    theta_dict['hDD'] = 1
    theta_dict['nu0'] = np.exp(theta_dict['log_nu0'])
    theta_dict['sigma_w'] = np.exp(theta_dict['log_sigma_w'])
    log_likelihood_ = log_likelihood_dm(theta_dict, zdata, wdata)
    if np.isnan(log_likelihood_):
        return -np.inf
    return log_prior_ + log_likelihood_

# ==============================================================
# ==============================================================
# posterior dd
def log_posterior_dd(theta, data, locs, scales, dim, norm_list, uni_list, simple=True):
    skip = 0 if simple else 24
    theta_dict = dict()
    zdata, wdata = data
    theta_dict_plus = dict(
        sigmaDD=theta[skip+0],
        log_hDD=theta[skip+1],
        log_nu0=theta[skip+2],
        zsun=theta[skip+3],
        R=theta[skip+4],
        w0=theta[skip+5:skip+5+dim],
        log_sigma_w=theta[skip+5+dim:skip+5+2*dim],
        a=theta[skip+5+2*dim:skip+5+3*dim]
    )
    theta_dict.update(theta_dict_plus)
    if simple:
        theta_dict['rhos'] = rhos
        theta_dict['sigmaz'] = sigmaz
    else:   
        theta_dict['rhos'] = theta[:12]
        theta_dict['sigmaz'] = theta[12:24]

    log_prior_ = log_prior(theta_dict, locs, scales, norm_list, uni_list)
    if not np.isfinite(log_prior_):
        return -np.inf
    theta_dict['rhoDM'] = 0
    theta_dict['hDD'] = np.exp(theta_dict['log_hDD'])
    theta_dict['nu0'] = np.exp(theta_dict['log_nu0'])
    theta_dict['sigma_w'] = np.exp(theta_dict['log_sigma_w'])
    log_likelihood_ = log_likelihood_dm(theta_dict, zdata, wdata)
    if np.isnan(log_likelihood_):
        return -np.inf
    return log_prior_ + log_likelihood_

# ==============================================================
# ==============================================================
# posterior dd & dm
def log_posterior_dd_dm(theta, data, locs, scales, dim, norm_list, uni_list, simple=True):
    skip = 0 if simple else 24
    theta_dict = dict()
    zdata, wdata = data
    theta_dict_plus = dict(
        rhoDM=theta[skip+0],
        sigmaDD=theta[skip+1],
        log_hDD=theta[skip+2],
        log_nu0=theta[skip+3],
        zsun=theta[skip+4],
        R=theta[skip+5],
        w0=theta[skip+6:skip+6+dim],
        log_sigma_w=theta[skip+6+dim:skip+6+2*dim],
        a=theta[skip+6+2*dim:skip+6+3*dim]
    )
    theta_dict.update(theta_dict_plus)
    if simple:
        theta_dict['rhos'] = rhos
        theta_dict['sigmaz'] = sigmaz
    else:   
        theta_dict['rhos'] = theta[:12]
        theta_dict['sigmaz'] = theta[12:24]

    log_prior_ = log_prior(theta_dict, locs, scales, norm_list, uni_list)
    if not np.isfinite(log_prior_):
        return -np.inf
    theta_dict['hDD'] = np.exp(theta_dict['log_hDD'])
    theta_dict['nu0'] = np.exp(theta_dict['log_nu0'])
    theta_dict['sigma_w'] = np.exp(theta_dict['log_sigma_w'])
    log_likelihood_ = log_likelihood_dm(theta_dict, zdata, wdata)
    if np.isnan(log_likelihood_):
        return -np.inf
    return log_prior_ + log_likelihood_

# ==============================================================
# ==============================================================
# posterior mond
def log_posterior_mond(theta, data, locs, scales, dim, norm_list, uni_list, simple=True):
    skip = 0 if simple else 24
    zdata, wdata = data
    theta_dict = dict(
        log_mu0=theta[skip+0],
        log_nu0=theta[skip+1],
        zsun=theta[skip+2],
        R=theta[skip+3],
        w0=theta[skip+4:skip+4+dim],
        log_sigma_w=theta[skip+4+dim:skip+4+2*dim],
        a=theta[skip+4+2*dim:skip+4+3*dim]
    )
    if not simple:
        theta_dict['rhos'] = theta[:12]
        theta_dict['sigmaz'] = theta[12:24]
    else:
        theta_dict['rhos'] = rhos
        theta_dict['sigmaz'] = sigmaz
    log_prior_ = log_prior(theta_dict, locs, scales, norm_list, uni_list)
    if not np.isfinite(log_prior_):
        return -np.inf
    
    theta_dict['mu0'] = np.exp(theta_dict['log_mu0'])
    theta_dict['nu0'] = np.exp(theta_dict['log_nu0'])
    theta_dict['sigma_w'] = np.exp(theta_dict['log_sigma_w'])
    
    return log_prior_ + log_likelihood_mond(theta_dict, zdata, wdata)

# ==============================================================
# ==============================================================
# posterior no
def log_posterior_no(theta, data, locs, scales, dim, norm_list, uni_list, simple=True):
    skip = 0 if simple else 24
    zdata, wdata = data
    theta_dict = dict(
        log_nu0=theta[skip+0],
        zsun=theta[skip+1],
        R=theta[skip+2],
        w0=theta[skip+3:skip+3+dim],
        log_sigma_w=theta[skip+3+dim:skip+3+2*dim],
        a=theta[skip+3+2*dim:skip+3+3*dim]
    )
    if not simple:
        theta_dict['rhos'] = theta[:12]
        theta_dict['sigmaz'] = theta[12:24]
    else:
        theta_dict['rhos'] = rhos
        theta_dict['sigmaz'] = sigmaz
    log_prior_ = log_prior(theta_dict, locs, scales, norm_list, uni_list)
    if not np.isfinite(log_prior_):
        return -np.inf
    
    theta_dict['rhoDM'] = 0
    theta_dict['sigmaDD'] = 0
    theta_dict['hDD'] = 1
    theta_dict['nu0'] = np.exp(theta_dict['log_nu0'])
    theta_dict['sigma_w'] = np.exp(theta_dict['log_sigma_w'])
    
    return log_prior_ + log_likelihood_dm(theta_dict, zdata, wdata)