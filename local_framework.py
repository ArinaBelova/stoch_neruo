import numpy as np
from scipy.linalg import expm

def local_framework(stim: np.array, dt: int, params: dict):
    nd, num_time_bins = stim.shape
    N = params["N"]
    time_delay = params["time_delay"]

    alpha = params["alpha"]
    F_max = params["F_max"]
    F_min = params["F_min"]

    # decay in filtered spike trains
    lambda_d = 1/params["tau_d"]
    # governing dynamics of X
    A = params["A"]

    mu = params["mu"]
    # cost on spiking
    r_lambda = mu * lambda_d**2

    # Generate decoding weights
    W = params["w_mean"] * np.append(-np.ones(int(np.ceil(N/2))), np.ones(int(np.floor(N/2)))) + np.random.random((nd, N)) * params["w_sig"]
    # Threshold
    T = (r_lambda + np.sum(pow(W.T, 2), 1)) / 2

    # Initialise variables
    
    # spikes
    ss = np.zeros((N, num_time_bins))
    # filtered spike train
    rr = np.zeros((N, num_time_bins))
    # true dynamics x
    xx = np.zeros((nd, num_time_bins))
    # network readout, estimate of x
    xh = np.zeros((nd, num_time_bins))
    # internal estimate of x
    zz = np.zeros((nd, num_time_bins))
    # voltage
    vv = np.zeros((N, num_time_bins))
    # probability of spiking 
    spike_prob = np.zeros((N, num_time_bins))
    # penalty on spiking
    penalty = np.zeros(N)

    # exponential integrators
    # x integration 
    A_mult = expm(dt * A) 
    # x integration with delay 
    A_mult_delay = expm(time_delay * dt * A)
    # r integration
    R_mult = np.exp(-lambda_d * dt)
    # r integration with delay
    R_mult_delay = np.exp(-lambda_d * dt * time_delay)

    # Run simulation:
    # start with 1 since in 0 we should have initial values of the system:
    for t in np.arange(1, num_time_bins):
        # update true target variable
        xx[:, t] = A_mult @ xx[:, t - 1] + dt * stim[:, t]

        # update filtered spike trains
        rr[:, t] = rr[:, t] + R_mult * rr[:, t - 1]
        # current network estimate, pre-spike
        xh[:, t] = W @ rr[:, t] # xh[:, t] should be a scalar for 1D system

        # update z, network estimate of x:
        zz[:, t] = A_mult @ zz[:, t - 1] + dt * stim[:, t]

        # update penalty 
        d_penalty = - r_lambda * (ss[:, t - 1] + R_mult_delay * rr[:, t - 1])
        penalty = penalty + dt * d_penalty

        # update voltage 
        vv[:, t] = W * (A_mult_delay @ zz[:, t] - R_mult_delay * W @ rr[:, t]) + penalty

        # COmpute instantaneous spike rate
        rt = alpha * (vv[:, t] - T)
        # conditional intensity
        cond = F_max * 1 / (1 + F_max * np.exp(-rt)) + F_min
        # probability of spiking:
        spike_prob[:, t] = 1 - np.exp(-cond * dt)

        # Spiking iself:
        # Find neurons that spike, Bernoulli RV
        spiking_ind = np.where(np.random.random(N) < spike_prob[:, t])
        if (len(spiking_ind) and (t < num_time_bins - time_delay)):
            ss[spiking_ind, t + time_delay] = 1
            rr[spiking_ind, t + time_delay] += 1

        # update estimate of x
        xh[:, t] = W @ rr[:, t]

    return ss, xh, xx    
            


        

