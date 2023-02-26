from scipy.stats import norm

def n_gaussian(x, *args):
    n = len(args) // 3
    y = 0
    for i in range(n):
        a, mu, sigma = args[3*i:3*i+3]
        y += a*norm.pdf(x, mu, sigma)
    return y