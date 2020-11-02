import numpy as np

from neurolib.utils.collections import dotdict


def loadDefaultParams(Cmat=None, Dmat=None, seed=None):
    """Load default parameters for the augmented Wilson-Cowan model
    
    :param Cmat: Structural connectivity matrix (adjacency matrix) of coupling strengths, will be normalized to 1. If not given, then a single node simulation will be assumed, defaults to None
    :type Cmat: numpy.ndarray, optional
    :param Dmat: Fiber length matrix, will be used for computing the delay matrix together with the signal transmission speed parameter `signalV`, defaults to None
    :type Dmat: numpy.ndarray, optional
    :param seed: Seed for the random number generator, defaults to None
    :type seed: int, optional
    
    :return: A dictionary with the default parameters of the model
    :rtype: dict
    """

    params = dotdict({})

    ### runtime parameters
    params.dt = 0.1  # ms 0.1ms is reasonable
    params.duration = 3500  # Simulation duration (ms)
    params.seed = 0  # seed for RNG of noise and ICs

    # ------------------------------------------------------------------------
    # global whole-brain network parameters
    # ------------------------------------------------------------------------

    # the coupling parameter determines how nodes are coupled.
    # "diffusive" for diffusive coupling, "additive" for additive coupling
    params.coupling = "diffusive"

    # signal transmission speed between areas
    params.signalV = 0
    params.K_gl = 1  # global coupling strength

    if Cmat is None:
        params.N = 1
        params.Cmat = np.zeros((1, 1))
        params.lengthMat = np.zeros((1, 1))

    else:
        params.Cmat = Cmat.copy()  # coupling matrix
        np.fill_diagonal(Cmat, 0)  # no self connections
        params.Cmat = Cmat / np.max(Cmat)  # normalize matrix
        params.N = len(params.Cmat)  # number of nodes
        params.lengthMat = Dmat

    # ------------------------------------------------------------------------
    # local node parameters
    # ------------------------------------------------------------------------

    # external input parameters:
    params.tau_ou = 1  # ms Timescale of the Ornstein-Uhlenbeck noise process
    params.sigma_ou = 0.0  # mV/ms/sqrt(ms) noise intensity
    params.u_ou_mean = 0.0  # mV/ms (OU process) [0-5]
    params.p_ou_mean = 0.0  # mV/ms (OU process) [0-5]
    params.s_ou_mean = 0.0  # mV/ms (OU process) [0-5]

    # neural mass model parameters
    params.tau_u = 10   # excitatory time constant
    params.tau_p = 10  # PV time constant
    params.tau_s = 10  # SST time constant
    params.tau_d1 = 1500 # replenishment time constant
    params.tau_d2 = 20   # depletion time constant

    params.w_ee = 1.1 # local E-E coupling
    params.w_ep = 2.  # local E-PV coupling
    params.w_es = 1.  # local E-SST coupling
    params.w_pe = 1.  # local PV-E coupling
    params.w_pp = 2.  # local PV-PV coupling
    params.w_ps = 2.  # local PV-SST coupling
    params.w_se = 6.  # local SST-E coupling
    params.w_sp = 0.  # local SST-PV coupling
    params.w_ss = 0.  # local SST-SST coupling

    params.w_ee2 = 0.667 # lateral E-E coupling
    params.w_pe2 = 1.25 # lateral PV-E coupling
    params.w_se2 = 0.125 # lateral SST-E coupling

    params.r_u = 3.  # excitatory gain
    params.r_p = 3.  # PV gain
    params.r_s = 3.  # SST gain
    params.u_th = 0.7 # excitatory firing threshold
    params.p_th = 1.0  # PV firing threshold
    params.s_th = 1.0  # SST firing threshold

    params.q = 1.3  # input amplitude

    params.a = 0.5 # degree of depression
    params.b = 1   # degree of facilitation

    params.opt_PV  = np.zeros(((int(params.duration/params.dt)))) # optogenetic PV suppression variable
    params.opt_SST = np.zeros(((int(params.duration/params.dt)))) # optogenetic SST suppression variable

    params.α = 0.65 # percentage of thalamic input
    params.I_ext = np.zeros((params.N,(int(params.duration/params.dt)))) #External input

    # ------------------------------------------------------------------------

    params.us_init = 0.05 * np.random.uniform(0, 1, (params.N, 1))
    params.ps_init = 0.05 * np.random.uniform(0, 1, (params.N, 1))
    params.ss_init = 0.05 * np.random.uniform(0, 1, (params.N, 1))

    # Ornstein-Uhlenbeck noise state variables
    params.u_ou = np.zeros((params.N,))
    params.p_ou = np.zeros((params.N,))
    params.s_ou = np.zeros((params.N,))

    return params


def computeDelayMatrix(lengthMat, signalV, segmentLength=1):
    """Compute the delay matrix from the fiber length matrix and the signal velocity

        :param lengthMat:       A matrix containing the connection length in segment
        :param signalV:         Signal velocity in m/s
        :param segmentLength:   Length of a single segment in mm

        :returns:    A matrix of connexion delay in ms
    """

    normalizedLenMat = lengthMat * segmentLength
    # Interareal connection delays, Dmat(i,j) in ms
    if signalV > 0:
        Dmat = normalizedLenMat / signalV
    else:
        Dmat = lengthMat * 0.0
    return Dmat
