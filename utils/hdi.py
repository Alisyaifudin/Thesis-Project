import numpy as np
from scipy import integrate
from scipy.optimize import fmin

# Highest density interval of {perc}% of normalized distribution
def hdi(func_un, perc=0.68, res=1E3, min=0.0, max=1.0):
  r""" Highest density interval of {perc}% of normalized distribution
  perc: percentile
  res: resolution, number of sampling from distributiion
  min: min x-value
  max: max x-value
  """
  x = np.linspace(min, max, int(res))
  # normalize func to 1
  area, _ = integrate.quad(func_un, a=min, b=max)
  func = lambda x: func_un(x)/area
  y = func(x)
  upper = np.max(y)*0.99
  below = 0
  for _ in range(10):
    ys = np.linspace(upper, below, 10)
    for i in range(10):
      mask = y > ys[i]
      x_mask = x[mask]
      integral, _ = integrate.quad(func, a=x_mask[0], b=x_mask[-1])
      if(integral > perc): break
    upper = ys[i-1]
    below = ys[i]
    xMin = x_mask[0]
    xMax = x_mask[-1]
  return (xMin, xMax)

def find_max(func, x0=None):
  r""" Find maximum of a function
  """
  x0 = np.random.rand() if x0 is None else x0
  xMax = fmin(lambda x: -func(x), x0, disp=False)
  return xMax
  