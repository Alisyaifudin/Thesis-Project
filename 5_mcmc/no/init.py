def init(model):
    ndim = 30
    if model == 1:
        ndim = 30
    elif model == 2:
        ndim = 32
    else:
        raise ValueError("model must be 1 or 2")
    nwalkers = 2*ndim+2
    return (ndim, nwalkers)