import numpy as np
from scipy.linalg import expm

def population_framework(stim: np.array, dt: int, params: dict):
    nd, num_time_bins = stim.shape
    N = params["N"]
    time_delay = params["time_delay"]
    beta = params["beta"]

    # decay in filtered spikes
    lambda_d = 1/params["tau_d"]

    # governing dynamics of X
    A = params["A"]

    # Generating decoding weights
    W = (np.dot(params["w_mean"].T, np.ones((1, N//2))) 
         + np.random.randn(nd, N//2) * params["w_sig"] * np.mean(params["w_mean"]))
    
    # Pseudoinverse of W
    W_pinv = np.linalg.pinv(W)

    # Initialise variables
    
    # spikes
    ss = np.zeros((N, num_time_bins, 1))
    # filtered spike train
    rr = np.zeros((int(N/2), num_time_bins))
    # true dynamics x
    xx = np.zeros((nd, num_time_bins))
    # network readout, estimate of x
    xh = np.zeros((nd, num_time_bins))
    # internal estimate of x
    zz = np.zeros((nd, num_time_bins))
    # voltage
    vv = np.zeros((int(N/2), num_time_bins))
    # probability of spiking 
    spike_prob = np.zeros((int(N/2), num_time_bins))

    # exponential integrators
    # x integration 
    A_mult = expm(dt * A) 
    # print(A_mult.shape, A.shape)
    # x integration with delay 
    A_mult_delay = expm(time_delay * dt * A)
    # print(A_mult_delay.shape)
    # r integration
    R_mult = np.exp(-lambda_d * dt)
    # r integration with delay
    R_mult_delay = np.exp(-lambda_d * dt * time_delay)

    # Run simulation

    # start with 1 since in 0 we should have initial values of the system:
    for t in np.arange(1, num_time_bins):
        # update true target variable
        xx[:, t] = A_mult @ xx[:, t - 1] + dt * stim[:, t]
        # print(A_mult.shape, xx.shape, stim.shape, rr.shape, xh.shape, W.shape, W_pinv.shape, zz.shape)
        # update filtered spike trains
        rr[:, t] = rr[:, t] + R_mult * rr[:, t - 1]
        # current network estimate, pre-spike
        xh[:, t] = W @ rr[:, t] 

        # update z, network estimate of x:
        zz[:, t] = A_mult @ zz[:, t - 1] + dt * stim[:, t]
        # print(zz[:, t].shape)
        # print((W @ rr[:, t]).shape, W.shape, rr[:, t].shape)
        # update voltage 
        vv[:, t] = W_pinv @ (A_mult_delay @ zz[:, t] - R_mult_delay * W @ rr[:, t]) 

        # compute spike probabilities
        # pronbability of spiking
        spike_prob[:, t] = (1 / beta) * vv[:, t] 
        # for positive mirror neurons
        p_pos = np.maximum(spike_prob[:, t : t+1], 0)
        # for negative mirror neurons
        p_neg = - np.minimum(spike_prob[:, t : t+1], 0)

        # spiking
        # spikes of positive mirror neurons
        spike_pos = np.random.rand(int(N/2), 1) < p_pos
        # spikes of negative mirror neurons
        spike_neg = np.random.rand(int(N/2), 1) < p_neg
        # print(spike_neg.shape, spike_pos.shape, p_pos.shape, spike_prob[:,t].shape)

        if t < num_time_bins - time_delay:
            # add spike to the spike train 
            ss[:, t + time_delay] = np.concatenate((spike_neg, spike_pos), axis=0)
            # calculate net number of spikes
            # print(spike_pos.astype(int).shape, spike_neg.astype(int).shape)
            net_spikes = spike_pos.astype(int) - spike_neg.astype(int)
            # print(net_spikes.shape)
            # update filtered spike train with time_delay in the future
            rr[:, t + time_delay : t + time_delay + 1] += net_spikes
        
        # update estimate of x
        xh[:, t] = W @ rr[:, t]
    
    return ss, xh, xx