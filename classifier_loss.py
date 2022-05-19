# empirical risk at a specific node

def risk(y, pi = None, n_p = None, n_u = None, n_n = None, estimator = None):
    nwp = (y == 1).sum()
    nwu = (y == 0).sum()
    nwn = (y == -1).sum()
    
    g = 1
    if estimator in ['uPU', 'nnPU']:
        out_pos = (g == -1)*nwp*pi/n_p + (g == 1)*nwu/n_u - (g == 1)*nwp*pi/n_p
    else:
        out_pos = (g == -1)*nwp*pi/n_p + (g == 1)*nwn*(1-pi)/n_n
    
    
    g = -1
    if estimator in ['uPU', 'nnPU']:
        out_neg = (g == -1)*nwp*pi/n_p + (g == 1)*nwu/n_u - (g == 1)*nwp*pi/n_p
    else:
        out_neg = (g == -1)*nwp*pi/n_p + (g == 1)*nwn*(1-pi)/n_n

    return out_pos.item(), out_neg.item()