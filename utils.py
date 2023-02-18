import signal
from time import time
from requests import HTTPError
from time import sleep
import subprocess
from os.path import join, abspath
from os import pardir, mkdir
import vaex
import numpy as np
from scipy import interpolate
from scipy.stats import norm, uniform
from scipy.integrate import quad, odeint, simps
from operator import itemgetter
from scipy.optimize import curve_fit
from scipy import optimize

root_data_dir = abspath(join(pardir, "Data"))

# progress bar
def progressbar(percent=0, width=50, info="", path="", flush=False) -> None:
    left = int((width * percent) // 100)
    right = width - left
    
    tags = "#" * left
    spaces = " " * right
    percents = f"{percent:.0f}%"
    text = f"\r[{tags}{spaces}] {percents} {info}"
    if(flush):
        print(text, end="", flush=True)
    else:
        print(text)
    if(path != ""):
        with open(path, 'a') as f:
            f.write(f"{text}")

# add timeout, such that sending request again after some period of time
def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None, minVal=1, path=""):
    class TimeoutError(Exception):
        pass

    def handler(signum, frame):
        raise TimeoutError()

    # set the timeout handler
    t0 = time()
    signal.signal(signal.SIGALRM, handler) 
    signal.alarm(timeout_duration)
    try:
        result = func(*args, **kwargs)
    except TimeoutError as exc:
        result = default
        t1 = time()
        text = f"too long, requesting again...\ntime = {round(t1-t0,2)}s"
        if(path != ""):
            with open(path, 'a') as f:
                f.write(f"{text}")
        print(text)
    except HTTPError:
        result = default
        t1 = time()
        # a litte hacky, need some fixes
        if(t1-t0 < minVal):
            text = f"service unavailable, sleep for 300s\ntime = {round(t1-t0,2)}s"
            if(path != ""):
                with open(path, 'a') as f:
                    f.write(f"{text}")
            print(text)
            sleep(300)
            text = "continue"
            if(path != ""):
                with open(path, 'a') as f:
                    f.write(f"{text}")
            print(text)
        else:
            text = f"server not responding, try again\nmessage: attr: {HTTPError.__dict__}\ntime = {round(t1-t0,2)}s"
            print(text)
            if(path != ""):
                with open(path, 'a') as f:
                    f.write(f"{text}")
    except KeyboardInterrupt:
        raise KeyboardInterrupt
    except Exception:
        result = default
        t1 = time()
        text = f"some error\ntime = {round(t1-t0,2)}s" 
        print(text)
        if(path != ""):
                with open(path, 'a') as f:
                    f.write(f"{text}")
    finally:
        signal.alarm(0)
    
    return result

# add AS
def appendName(element, name):
    string = element.split(" AS ")
    if(len(string) == 1):
        return f"{name}.\"{element}\""
    else:
        return f"{name}.\"{string[0]}\" AS {string[1]}"

def runcmd(cmd, verbose = False, *args, **kwargs):

    process = subprocess.Popen(
        cmd,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        text = True,
        shell = True
    )
    std_out, std_err = process.communicate()
    if verbose:
        print(std_out.strip(), std_err)
    pass

####################################################################################
####################################################################################
####################################################################################
# need mamajek-spectral-class.hdf5, created at "2. Vertical Number Density/2.3. Survey Completeness.ipynb"
def load_spectral_types():
    file_spectral_class = join(root_data_dir, "mamajek-spectral-class.hdf5")
    df_spectral_class = vaex.open(file_spectral_class)
    df_filtered = df_spectral_class[["SpT", "M_J", "J-H", "H-Ks"]]
    mask = (df_filtered["M_J"] == df_filtered["M_J"])*(df_filtered["H-Ks"] == df_filtered["H-Ks"])*(df_filtered["J-H"] == df_filtered["J-H"])
    df_filtered_good = df_filtered[mask].to_pandas_df()
    df_filtered_good['J-K'] = df_filtered_good['J-H']+df_filtered_good['H-Ks']
    return df_filtered_good

def main_sequence_cut_r(jk,low=False):
    sp = load_spectral_types()
    sp_indx = np.array([(not 'O' in s)*(not 'L' in s)\
                            *(not 'T' in s)*(not 'Y' in s)\
                            *(not '.5V' in s) for s in sp['SpT']],
                        dtype='bool')
    # Cut out the small part where the color decreases
    sp_indx *= (np.roll((sp['J-K']),1)-(sp['J-K'])) <= 0.

    main_locus = interpolate.UnivariateSpline((sp['J-K'])[sp_indx],
                                        sp['M_J'][sp_indx],k=3,s=1.)
    """Main-sequence cut, based on MJ, high as in low"""
    j_locus= main_locus(jk)
    if low:
        dj = 0.4
    else:
        dj= 0.4-0.1*(j_locus-5.)
        dj*= -1.
    return j_locus+dj
####################################################################################
####################################################################################
####################################################################################
# Function to calculate the density of dark matter
def sech(x):
    return 1/np.cosh(x)

def rhoDD(z, sigmaDD, hDD):
    return sigmaDD/(4*hDD)*sech(z/(2*hDD))**2

def frho(rho0, phi, sigmaz ):
    return rho0*np.exp(-phi/sigmaz**2)

def rho_tot(z, phi, rhos, sigmaz, rhoDM, sigmaDD, hDD, R=3.4E-3):
    rho = np.array(list(map(lambda par: frho(par[0], phi, par[1]), zip(rhos, sigmaz))))
    return rho.sum() + rhoDM + rhoDD(z, sigmaDD, hDD) - R

def f(u, z, rhos, sigmaz, rhoDM, sigmaDD, hDD, R=3.4E-3):
    G = 4.30091E-3 # pc/M_sun (km/s)^2
    return (u[1], 4*np.pi*G*rho_tot(z, u[0], rhos, sigmaz, rhoDM, sigmaDD, hDD, R))

# def log_nu(phi, sigma_w):
#   return -phi/sigma_w**2

def log_nu_mod(zz, **theta):
    res = 1000
    args = ('rhos', 'sigmaz', 'rhoDM', 'sigmaDD', 'hDD', 'Nu0', 'zsun', 'R', 'sigma_w')
    rhos, sigmaz, rhoDM, sigmaDD, hDD, Nu0, zsun, R, sigma_w = itemgetter(*args)(theta)
    if 'res' in theta.keys(): res = theta['res']
    phi0 = 0 # (km/s)^2
    Kz0 = 0 # pc (km/s)^2
    y0 = [Kz0, phi0]
    zmax = np.max(np.abs(zz+zsun)) 
    zs = np.linspace(0, zmax, res)
    us = odeint(f, y0, zs, args=(rhos, sigmaz, rhoDM, sigmaDD, hDD, R))
    phi = us[:, 0]
    phi_interp_pos = interpolate.interp1d(zs, phi, kind='cubic')
    phi_interp = lambda z: phi_interp_pos(np.abs(z))
    phii = phi_interp(zz+zsun)
    # print(phii)
    logNu = -phii/sigma_w**2
    return logNu+np.log(Nu0)

def fdist_cum(w, sigma, w0):
  return norm.cdf(w, loc=w0, scale=sigma)
def fdist_pdf(w, sigma, w0):
  return norm.pdf(w, loc=w0, scale=sigma)

# def bootstrap_resampling(func, theta, sigmas, zz):
#   """Bootstrap resampling of a function with a distribution"""
#   # run N bootstrap resampling
#   Zs = np.zeros((len(sigmas), len(zz)*2))
#   Nus = np.zeros((len(sigmas), len(zz)*2))

#   for i, sigma in enumerate(sigmas):
#     Zs[i], Nus[i] = func(zz, theta, sigma)
#     if (i % 100 == 0): print(i, end=" ")
#   print(f"{i} end")
#   Nu_mean = np.mean(Nus, axis=0)
#   Nu_std = np.std(Nus, axis=0)
#   return Zs[0], Nu_mean, Nu_std

# def nu_bootstrap(zz, theta, tipe="A",  res=1000):
#   data_dir = join(root_data_dir, "Uncertainty")
#   df_stats = vaex.open(join(data_dir, "stats.hdf5"))
#   args = ('rhos', 'sigmaz', 'rhoDM', 'sigmaDD', 'hDD', 'Nv', 'zsun')
#   rhos, sigmaz, rhoDM, sigmaDD, hDD, Nv, zsun = itemgetter(*args)(theta)

#   phi0 = 0 # (km/s)^2
#   Kz0 = 0 # pc (km/s)^2

#   y0 = [Kz0, phi0]
#   zmax = np.max(np.abs(zz+zsun))
#   zs = np.linspace(0, zmax*1000, res)
#   us = odeint(f, y0, zs, args=(rhos, sigmaz, rhoDM, sigmaDD, hDD))
#   phi = us[:, 0]
#   phi_interp_pos = interpolate.interp1d(zs, phi, kind='cubic')
#   phi_interp = lambda z: phi_interp_pos(np.abs(z)*1000)
#   phii = phi_interp(zz+zsun)
#   data_dir = join(root_data_dir,"Velocity-Distribution")
#   df_velocity = vaex.open(join(data_dir, "Velocity-Distribution.hdf5"))
#   index = 0 if tipe == "A" else 1 if tipe == "F" else 2
#   sigma_v = df_velocity["sigma"].to_numpy()[index]
#   nus = nu(phii, sigma_v)
#   logNu = np.log(nus)
#   b = df_stats[tipe].to_numpy()[0]
#   logNu_std = b*logNu
#   return logNu, logNu_std

# def double_gaussian_cum(x, sigma1, sigma2, w0):
#   """Cumulative distribution of a double Gaussian"""
#   a = 2*sigma1/(sigma1+sigma2)
#   b = 2*sigma2/(sigma1+sigma2)
#   return np.heaviside(w0-x, 0.5)*norm.cdf(x, loc=w0, scale=sigma1)*a + np.heaviside(x-w0, 0.5)*((norm.cdf(x, loc=w0, scale=sigma2)-0.5)*b+a/2)
# def double_gaussian_pdf(x, sigma1, sigma2, w0):
#   a = 2*sigma1/(sigma1+sigma2)
#   b = 2*sigma2/(sigma1+sigma2)
#   return np.heaviside(w0-x, 0)*norm.pdf(x, loc=w0, scale=sigma1)*a + np.heaviside(x-w0, 1)*norm.pdf(x, loc=w0, scale=sigma2)*b

# def asymmerty_uncertainties(zz, theta, tipe="A"):
#   data_unc_dir = join(root_data_dir, "Uncertainty")
#   df_popt = vaex.open(join(data_unc_dir, "sys.hdf5"))
#   index = 0 if tipe=="A" else 1 if tipe=="F" else 2
#   sigma_v1 = df_popt["sigma_v1"].to_numpy()[index]
#   sigma_v2 = df_popt["sigma_v2"].to_numpy()[index]
#   logNu1 = nu_mod(zz, theta, sigma_v1)
#   logNu2 = nu_mod(zz, theta, sigma_v2)
#   sigma_sys = np.abs(logNu1-logNu2)/2
#   middle = (logNu1+logNu2)/2
#   return middle, sigma_sys

# def nu_mod_total(zz, theta, tipe="A", res=1000):
#   logNu_sys, logNu_sys_std = asymmerty_uncertainties(zz, theta, tipe=tipe)
#   logNu_stat, logNu_stat_std = nu_bootstrap(zz, theta, tipe=tipe, res=res)
#   above = logNu_stat*logNu_stat_std**2+logNu_sys*logNu_sys_std**2
#   below = logNu_stat_std**2+logNu_sys_std**2
#   logNu_mean = safe_div(above, below)
#   logNu_std = np.sqrt(logNu_stat_std**2+logNu_sys_std**2)
#   return logNu_mean, logNu_std

# def safe_div(x, y):
#   c = np.divide(x, y, out=np.zeros_like(x), where=(y!=0))
#   return c

def log_prior(theta, locs, scales):
    args = ('rhos', 'sigmaz', 'rhoDM', 'sigmaDD', 'hDD', 'Nu0', 'zsun', 'R', 'sigma_w', 'w0', 'N0')
    rhos, sigmaz, rhoDM, sigmaDD, hDD, Nu0, zsun, R, sigma_w, w0, N0 = itemgetter(*args)(theta)
    args = ('rhos_loc', 'sigmaz_loc', 'rhoDM_loc', 'sigmaDD_loc',  'hDD_loc', 'Nu0_loc', 
            'zsun_loc', 'R_loc', 'sigma_w_loc', 'w0_loc', 'N0_loc')
    rhos_loc, sigmaz_loc, rhoDM_loc, sigmaDD_loc, hDD_loc, Nu0_loc, zsun_loc, R_loc, sigma_w_loc, w0_loc, N0_loc = itemgetter(*args)(locs)
    args = ('rhos_scale', 'sigmaz_scale', 'rhoDM_scale', 'sigmaDD_scale', 'hDD_scale', 'Nu0_scale',
            'zsun_scale', 'R_scale', 'sigma_w_scale', 'w0_scale', 'N0_scale')
    rhos_scale, sigmaz_scale, rhoDM_scale, sigmaDD_scale, hDD_scale, Nu0_scale, zsun_scale, R_scale, sigma_w_scale, w0_scale, N0_scale = itemgetter(*args)(scales)
    uni_loc = np.array([rhoDM_loc, sigmaDD_loc, hDD_loc, Nu0_loc, zsun_loc, sigma_w_loc, w0_loc, N0_loc])
    uni_scale = np.array([rhoDM_scale, sigmaDD_scale, hDD_scale, Nu0_scale, zsun_scale, sigma_w_scale, w0_scale, N0_scale])
    uni_val = rhoDM, sigmaDD, hDD, Nu0, zsun, sigma_w, w0, N0
    log_uni = np.sum(uniform.logpdf(uni_val, loc=uni_loc, scale=uni_scale))
    result = (np.sum(norm.logpdf(rhos, loc=rhos_loc, scale=rhos_scale))
            +np.sum(norm.logpdf(sigmaz, loc=sigmaz_loc, scale=sigmaz_scale))
            +norm.logpdf(R, loc=R_loc, scale=R_scale)
            +log_uni)
    return result

def log_likelihood(theta, z, w, comp_z):
    nu_f = lambda x: np.exp(log_nu_mod(x, theta))*comp_z(x)
    zz = np.linspace(np.min(z), np.max(z), 1000)  
    integral_nu = simps(nu_f(zz), zz)
    log_nu = np.sum(log_nu_mod(z, theta)+np.log(comp_z(z)))

    mu_f = lambda x: theta['N0']*norm.pdf(x, loc=theta['w0'], scale=theta['sigma_w'])
    integral_mu, _ = quad(mu_f, np.min(w), np.max(w))
    log_mu = np.sum(norm.logpdf(w, loc=theta['w0'], scale=theta['sigma_w'])+np.log(theta['N0']))
    return log_nu-integral_nu+log_mu-integral_mu
 
def log_posterior(x, priors, data, comp_z):
    theta = dict(rhos=x[0:12], sigmaz=x[12:24], rhoDM=x[24], sigmaDD=x[25], hDD=x[26], Nu0=x[27], zsun=x[28], R=x[29], sigma_w=x[30], w0=x[31], N0=x[32])
    locs, scales = itemgetter('locs', 'scales')(priors)
    z, w = itemgetter('z', 'w')(data)
    log_prior_ = log_prior(theta, locs, scales)
    if not np.isfinite(log_prior_):
        return -np.inf
    log_likelihood_ = log_likelihood(theta, z, w, comp_z)
    return log_prior_ + log_likelihood_


##############################################################3

def hdi_interval(func, a, b, ymax, xmax, std, level=10, hdi=0.68):
    yup = ymax*0.999
    ydown = 0
    lower = a
    upper = b
    for i in range(level):
        dy = (yup-ydown)/10
        for ynew in np.arange(yup, ydown, -dy):
            sol = optimize.root(lambda y: func(y) - ynew, [xmax-std, xmax+std])
            area = quad(func, sol.x[0], sol.x[1])[0]
            if area > hdi:
                yup = ynew+dy
                ydown = ynew
                lower = sol.x[0]
                upper = sol.x[1]
                break
    return lower, upper

def inference_bayesian(post, x, hdi=0.68, level=10):
    interp_raw = interpolate.interp1d(x, post, kind='cubic', fill_value='extrapolate')
    norm = np.trapz(interp_raw(x), x)
    interp = lambda x: interp_raw(x)/norm
    xmax = optimize.fmin(lambda x: -1*interp(x), x[np.argmax(interp(x))])
    mean = quad(lambda x: x*interp(x), np.min(x), np.max(x))[0]
    std = np.sqrt(quad(lambda x: (x-mean)**2*interp(x), np.min(x), np.max(x))[0])
    lower, upper = hdi_interval(interp, x[0], x[-1], interp(xmax), xmax, std, level=level, hdi=hdi)
    return xmax, lower, upper