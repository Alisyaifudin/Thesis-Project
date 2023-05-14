def init(model):
    ndim = 31
    if model == 1:
        ndim = 33
    elif model == 2:
        ndim = 35
    else:
        raise ValueError("model must be 1 or 2")
    nwalkers = 2*ndim+2
    return (ndim, nwalkers)