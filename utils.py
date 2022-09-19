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
from scipy.stats import norm
from scipy.integrate import quad, odeint
from operator import itemgetter
from scipy.optimize import curve_fit

root_data_dir = abspath(join(pardir, "Data"))

# progress bar
def progressbar(percent=0, width=50) -> None:
    left = int((width * percent) // 100)
    right = width - left
    
    tags = "#" * left
    spaces = " " * right
    percents = f"{percent:.0f}%"
    
    print("\r[", tags, spaces, "]", percents, sep="", end="", flush=True)


# add timeout, such that sending request again after some period of time
def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None, minVal=1):

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
        print("too long, requesting again...")
        print(f"time = {round(t1-t0,2)}s")
    except HTTPError:
        result = default
        t1 = time()
        # a litte hacky, need some fixes
        if(t1-t0 < minVal):
            print("service unavailable, sleep for 300s")
            print(f"time = {round(t1-t0,2)}s")
            sleep(300)
            print("continue")
        else:
            print("server not responding, try again")
            print("message", HTTPError)
            print(f"time = {round(t1-t0,2)}s")
    except KeyboardInterrupt:
        raise KeyboardInterrupt
    except Exception:
        result = default
        t1 = time()
        print("some error")
        print(Exception)
        print(f"time = {round(t1-t0,2)}s")
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

sp = load_spectral_types()
sp_indx = np.array([(not 'O' in s)*(not 'L' in s)\
                        *(not 'T' in s)*(not 'Y' in s)\
                        *(not '.5V' in s) for s in sp['SpT']],
                    dtype='bool')
# Cut out the small part where the color decreases
sp_indx *= (np.roll((sp['J-K']),1)-(sp['J-K'])) <= 0.

main_locus = interpolate.UnivariateSpline((sp['J-K'])[sp_indx],
                                    sp['M_J'][sp_indx],k=3,s=1.)

def main_sequence_cut_r(jk,low=False):
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

def rho_tot(z: np.ndarray, phi: float, rhos: np.ndarray, sigmaz: np.ndarray, rhoDM: float, sigmaDD: float, hDD: float, R=3.4E-3) -> float:
    rho = np.array(list(map(lambda par: frho(par[0], phi, par[1]), zip(rhos, sigmaz))))
    return rho.sum() + rhoDM + rhoDD(z, sigmaDD, hDD) - R

def f(u, z, rhos, sigmaz, rhoDM, sigmaDD, hDD, R=3.4E-3):
    G = 4.30091E-3 # pc/M_sun (km/s)^2
    return (u[1], 4*np.pi*G*rho_tot(z, u[0], rhos, sigmaz, rhoDM, sigmaDD, hDD, R))

#  Function to calculate the density of stars
def fdist(w, sigma):
  return norm.pdf(w, 0, sigma)

# def nu(z, phi, fdist):
#   const = lambda w: 1.0/np.sqrt(w**2-2*phi)
#   a = np.sqrt(2*phi)
#   b = 1000
#   return 2*quad(lambda w: const(w)*fdist(w)*w, a, b)[0]

# created at 4.1. Potential (Dark).ipynb
df_b = vaex.open(join(root_data_dir, "b","b.hdf5"))

diff_interp = interpolate.interp1d(df_b.scale.to_numpy(), df_b['diff'].to_numpy(), kind="cubic")

def log_nu(phi, sigma_w):
  return -phi/(diff_interp(sigma_w)+sigma_w)**2

def nu_mod(zz, theta, sigma_v, zmax=200, res=1000):
  args = ('rhos', 'sigmaz', 'rhoDM', 'sigmaDD', 'hDD', 'Nv', 'zsun')
  rhos, sigmaz, rhoDM, sigmaDD, hDD, Nv, zsun = itemgetter(*args)(theta)

  phi0 = 0 # (km/s)^2
  Kz0 = 0 # pc (km/s)^2

  y0 = [Kz0, phi0]
  zs = np.linspace(0, zmax, res)
  us = odeint(f, y0, zs, args=(rhos, sigmaz, rhoDM, sigmaDD, hDD))
  phi = us[:, 0]
  phi_interp = interpolate.interp1d(zs, phi, kind='cubic')
  phii = phi_interp(zz)
  # nus = np.array(list(map(lambda z: Nv*nu(z, phi_interp(z), lambda w: fdist(w, sigma_v)), zz)))
  lognu = log_nu(phii, sigma_v)+np.log(Nv)
  Z = np.array([-1*zz[::-1],zz]).flatten()/1000 - zsun
  logNu = np.array([lognu[::-1],lognu]).flatten()
  return (Z, logNu)

def fdist_cum(w, sigma, w0):
  return norm.cdf(w, loc=w0, scale=sigma)
def fdist_pdf(w, sigma, w0):
  return norm.pdf(w, loc=w0, scale=sigma)

def bootstrap_resampling(func, theta, sigmas, zz):
  """Bootstrap resampling of a function with a distribution"""
  # run N bootstrap resampling
  Zs = np.zeros((len(sigmas), len(zz)*2))
  Nus = np.zeros((len(sigmas), len(zz)*2))

  for i, sigma in enumerate(sigmas):
    Zs[i], Nus[i] = func(zz, theta, sigma)
    if (i % 100 == 0): print(i, end=" ")
  print(f"{i} end")
  Nu_mean = np.mean(Nus, axis=0)
  Nu_std = np.std(Nus, axis=0)
  return Zs[0], Nu_mean, Nu_std

def double_gaussian_cum(x, sigma1, sigma2, w0):
  """Cumulative distribution of a double Gaussian"""
  a = 2*sigma1/(sigma1+sigma2)
  b = 2*sigma2/(sigma1+sigma2)
  return np.heaviside(w0-x, 0.5)*norm.cdf(x, loc=w0, scale=sigma1)*a + np.heaviside(x-w0, 0.5)*((norm.cdf(x, loc=w0, scale=sigma2)-0.5)*b+a/2)
def double_gaussian_pdf(x, sigma1, sigma2, w0):
  a = 2*sigma1/(sigma1+sigma2)
  b = 2*sigma2/(sigma1+sigma2)
  return np.heaviside(w0-x, 0)*norm.pdf(x, loc=w0, scale=sigma1)*a + np.heaviside(x-w0, 1)*norm.pdf(x, loc=w0, scale=sigma2)*b

def asymmerty_uncertainties(func, theta, zz, tipe="A"):
  data_dir = join(root_data_dir, "Spectral-Class-Velocity")
  df_popt = vaex.open(join(data_dir, "assy.hdf5"))
  index = 0 if tipe=="A" else 1 if tipe=="F" else 2
  sigma_v1 = df_popt["sigma_v1"].to_numpy()[index]
  sigma_v2 = df_popt["sigma_v2"].to_numpy()[index]
  Zs1, Nus1 = func(zz, theta, sigma_v1)
  Zs2, Nus2 = func(zz, theta, sigma_v2)
  sigma_sys = np.abs(Nus1-Nus2)/2
  middle = (Nus1+Nus2)/2
  return Zs1, middle, sigma_sys