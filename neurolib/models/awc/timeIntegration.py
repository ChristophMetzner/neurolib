import numpy as np
import numba
import matplotlib.pyplot as plt

from neurolib.models.awc import loadDefaultParams as dp


def timeIntegration(params):
    """Sets up the parameters for time integration

    :param params: Parameter dictionary of the model
    :type params: dict
    :return: Integrated activity variables of the model
    :rtype: (numpy.ndarray,)
    """

    dt = params["dt"]  # Time step for the Euler intergration (ms)
    duration = params["duration"]  # imulation duration (ms)
    RNGseed = params["seed"]  # seed for RNG

    # ------------------------------------------------------------------------
    # local parameters
    # See Park&Geffen, A unifying mechanistic model of excitatory-inhibitory interactions in the auditory cortex,
    # bioRxiv, 2020

    #Time constants
    tau_u = params["tau_u"]  # excitatory time constant
    tau_p = params["tau_p"]  # PV time constant
    tau_s = params["tau_s"]  # SST time constant
    tau_d1 = params["tau_d1"] # replenishment time constant
    tau_d2 = params["tau_d2"] # depletion time constant

    #Weights
    w_ee = params["w_ee"]  # local E-E coupling
    w_ep = params["w_ep"]  # local E-PV coupling
    w_es = params["w_es"]  # local E-SST coupling
    w_pe = params["w_pe"]  # local PV-E coupling
    w_pp = params["w_pp"]  # local PV-PV coupling
    w_ps = params["w_ps"]  # local PV-SST coupling
    w_se = params["w_se"]  # local SST-E coupling
    w_sp = params["w_sp"]  # local SST-PV coupling
    w_ss = params["w_ss"]  # local SST-SST coupling
    w_ee2 = params["w_ee2"] # lateral E-E coupling
    w_pe2 = params["w_pe2"] # lateral PV-E coupling
    w_se2 = params["w_se2"] # lateral SST-E coupling

    #gain & thresholds
    r_u = params["r_u"]  # excitatory gain
    r_p = params["r_p"]  # PV gain
    r_s = params["r_s"]  # SST gain
    u_th = params["u_th"]  # excitatory firing threshold
    p_th = params["p_th"]  # PV firing threshold
    s_th = params["s_th"]  # SST firing threshold

    #Input amplitude
    q = params["q"]  # Input amplitude

    # degree of plasticity params
    a = params["a"] # degree of depression
    b = params["b"] # degree of facilitation

    #Optogenetic suppression variables
    opt_pv = params["opt_PV"] # optogenetic PV suppression variable
    opt_sst = params["opt_SST"] # optogenetic SST suppression variable

    α = params["α"]  # percentage of thalamic input

    # external input parameters:
    # Parameter of the Ornstein-Uhlenbeck process for the external input(ms)
    tau_ou = params["tau_ou"]
    # Parameter of the Ornstein-Uhlenbeck (OU) process for the external input ( mV/ms/sqrt(ms) )
    sigma_ou = params["sigma_ou"]
    # Mean external excitatory input (OU process) (mV/ms)
    u_ou_mean = params["u_ou_mean"]
    # Mean external PV input (OU process) (mV/ms)
    p_ou_mean = params["p_ou_mean"]
    # Mean external SSt input (OU process) (mV/ms)
    s_ou_mean = params["s_ou_mean"]

    # ------------------------------------------------------------------------
    # global coupling parameters

    # Connectivity matrix
    # Interareal relative coupling strengths (values between 0 and 1), Cmat(i,j) connection from jth to ith
    Cmat = params["Cmat"]
    N = len(Cmat)  # Number of nodes
    K_gl = params["K_gl"]  # global coupling strength
    # Interareal connection delay
    lengthMat = params["lengthMat"]
    signalV = params["signalV"]

    if N == 1:
        Dmat = np.zeros((N, N))
        w_ee2 = 0 #No lateral connections for single unit
        w_pe2 = 0
        w_se2 = 0
        a = 0
        b = 0
        α = 0
    else:
        # Interareal connection delays, Dmat(i,j) Connnection from jth node to ith (ms)
        Dmat = dp.computeDelayMatrix(lengthMat, signalV)
        Dmat[np.eye(len(Dmat)) == 1] = np.zeros(len(Dmat))

    Dmat_ndt = np.around(Dmat / dt).astype(int)  # delay matrix in multiples of dt
    params["Dmat_ndt"] = Dmat_ndt
    # ------------------------------------------------------------------------
    # Initialization
    t = np.arange(0, duration, dt)  # Time variable (ms)

    #plasticity terms
    g = np.ones((N, len(t)))  # depressing term, g = D for three unit model
    F = np.zeros((N, len(t))) # facilitating term F = 1- g for three unit model

    #Input
    I_ext = params["I_ext"]   # external input
    I = np.zeros((N, len(t))) # Thalamic adapted input = I_ext * g

    #Optogenetic terms
    opt_PV = params["opt_PV"] # Optogenetic PV depression
    opt_SST = params["opt_SST"] # Optogenetic SST depression

    sqrt_dt = np.sqrt(dt)

    max_global_delay = np.max(Dmat_ndt)
    startind = int(max_global_delay + 1)  # timestep to start integration at

    u_ou = params["u_ou"]
    p_ou = params["p_ou"]
    s_ou = params["s_ou"]

    # set of the state variable arrays
    us = np.zeros((N, len(t)))
    ps = np.zeros((N, len(t)))
    ss = np.zeros((N, len(t)))

    # ------------------------------------------------------------------------
    # Set initial values
    # if initial values are just a Nx1 array
    if np.shape(params["us_init"])[1] == 1:
        us_init = np.dot(params["us_init"], np.ones((1, startind)))
        ps_init = np.dot(params["ps_init"], np.ones((1, startind)))
        ss_init = np.dot(params["ss_init"], np.ones((1, startind)))
    # if initial values are a Nxt array
    else:
        us_init = params["us_init"][:, -startind:]
        ps_init = params["ps_init"][:, -startind:]
        ss_init = params["ss_init"][:, -startind:]

    # xsd = np.zeros((N,N))  # delayed activity
    us_input_d = np.zeros(N)  # delayed input to u
    ps_input_d = np.zeros(N)  # delayed input to p
    ss_input_d = np.zeros(N)  # delayed input to s

    if RNGseed:
        np.random.seed(RNGseed)

    # Save the noise in the activity array to save memory
    # us[:, startind:] = np.random.standard_normal((N, len(t) - startind))
    # ps[:, startind:] = np.random.standard_normal((N, len(t) - startind))
    # ss[:, startind:] = np.random.standard_normal((N, len(t) - startind))
    #
    # us[:, :startind] = us_init
    # ps[:, :startind] = ps_init
    # ss[:, :startind] = ss_init

    noise_us = np.zeros((N,))
    noise_ps = np.zeros((N,))
    noise_ss = np.zeros((N,))

    # ------------------------------------------------------------------------

    return timeIntegration_njit_elementwise(
        startind,
        t,
        dt,
        sqrt_dt,
        N,
        Cmat,
        K_gl,
        Dmat_ndt,
        us,
        ps,
        ss,
        g,
        F,
        us_input_d,
        ps_input_d,
        ss_input_d,
        I,
        I_ext,
        tau_u,
        tau_p,
        tau_s,
        tau_d1,
        tau_d2,
        w_ee,
        w_ep,
        w_es,
        w_pe,
        w_pp,
        w_ps,
        w_se,
        w_sp,
        w_ss,
        w_ee2,
        w_pe2,
        w_se2,
        noise_us,
        noise_ps,
        noise_ss,
        u_ou,
        p_ou,
        s_ou,
        u_ou_mean,
        p_ou_mean,
        s_ou_mean,
        tau_ou,
        sigma_ou,
        q,
        r_u,
        r_p,
        r_s,
        u_th,
        p_th,
        s_th,
        a,
        b,
        α,
        opt_PV,
        opt_SST
    )


@numba.njit
def timeIntegration_njit_elementwise(
        startind,
        t,
        dt,
        sqrt_dt,
        N,
        Cmat,
        K_gl,
        Dmat_ndt,
        us,
        ps,
        ss,
        g,
        F,
        us_input_d,
        ps_input_d,
        ss_input_d,
        I,
        I_ext,
        tau_u,
        tau_p,
        tau_s,
        tau_d1,
        tau_d2,
        w_ee,
        w_ep,
        w_es,
        w_pe,
        w_pp,
        w_ps,
        w_se,
        w_sp,
        w_ss,
        w_ee2,
        w_pe2,
        w_se2,
        noise_us,
        noise_ps,
        noise_ss,
        u_ou,
        p_ou,
        s_ou,
        u_ou_mean,
        p_ou_mean,
        s_ou_mean,
        tau_ou,
        sigma_ou,
        q,
        r_u,
        r_p,
        r_s,
        u_th,
        p_th,
        s_th,
        a,
        b,
        α,
        opt_PV,
        opt_SST
):
    ### integrate ODE system:

    def f(x, r):
        if x <= 0:
            return 0
        elif x <= 1/r:
            return r*x
        else:
            return 1

    for i in range(startind, len(t)):

        # loop through all the nodes
        for no in range(N):

            # To save memory, noise is saved in the activity array
            noise_us[no] = us[no, i]
            noise_ps[no] = ps[no, i]
            noise_ss[no] = ss[no, i]

            # delayed input to each node
            us_input_d[no] = 0
            ps_input_d[no] = 0
            ss_input_d[no] = 0

            #Count no: of connecting nodes
            count = 0
            for l in range(N):
                if Cmat[no,l]:
                    count += 1

            #Loop for adding Exc and sst inputs from connecting nodes
            sum_I = 0
            for l in range(N):
                # Note K_gl should be 1 or remove K_gl for Park-Geffen
                us_input_d[no] += K_gl * Cmat[no, l] * (us[l, i-1])
                sum_I += Cmat[no, l] * I_ext[l, i-1] * g[l,i-1]
                ss_input_d[no] += Cmat[no,l] * ss[l,i-1]
            I[no, i] = I_ext[no, i-1] * g[no,i-1]  + α * sum_I
            if count >= 2:
                us_input_d[no] = us_input_d[no]/count

            # augmented Wilson-Cowan model
            #Synaptic depression, g
            g_rhs = (
                (
                        (1-g[no,i-1])/tau_d1
                )
                -
                (
                        (g[no,i-1] * I_ext[no,i-1])/tau_d2
                )
            )

            #Synaptic facilitation, F
            f_rhs = (
                    (
                            (- F[no, i - 1]) / tau_d1
                    )
                    +
                    (
                            (I_ext[no, i - 1]) / tau_d2
                    )
            )

            #Excitatory population - us[t]
            u_rhs = (
                1
                / tau_u
                * (
                    -us[no, i - 1]
                    +
                    f(
                        (w_ee * us[no, i - 1]  # input from within the excitatory population
                        - (w_ep - a * (1-g[no,i-1])) * ps[no, i - 1]  # input from the PV population
                        - (w_es * ss[no, i - 1]) - b * F[no,i-1] * ss_input_d[no]   # input from the SST population
                        + w_ee2 * us_input_d[no]  # input from other unit Exc nodes
                        + q*I[no,i-1]) - u_th, r_u # external input
                    )
                    + u_ou[no]  # ou noise
                )
            )
            #PV population - ps[t]
            p_rhs = (
                    1
                    / tau_p
                    * (
                        -ps[no, i - 1]
                        +
                        f(
                             (w_pe * us[no, i - 1]  # input from within the excitatory population
                             - w_pp * ps[no, i - 1]  # input from the PV population
                             - w_ps * ss[no, i - 1]  # input from the SST population
                             + w_pe2 * us_input_d[no]  # input from other nodes
                             + opt_PV[i-1]             # level of optogenetic inactivation
                             + q * I[no,i-1]) - p_th, r_p # external input
                        )
                            + p_ou[no]  # ou noise
                    )
            )
            #SST population - sst[t]
            s_rhs = (
                    1
                    / tau_s
                    * (
                            -ss[no, i - 1]
                            +
                            f(
                                (w_se * us[no, i - 1]  # input from within the excitatory population
                                - w_sp * ps[no, i - 1]  # input from the PV population
                                - w_ss * ss[no, i - 1]  # input from the SST population
                                + w_se2 * us_input_d[no] # input from other nodes
                                + opt_SST[i-1])          # level of optogenetic inactivation
                                - s_th,r_s
                            )
                           + s_ou[no]  # ou noise
                    )
            )

            # Euler integration
            g[no, i] = g[no,i-1] + dt* g_rhs
            F[no, i] = F[no, i - 1] + dt * f_rhs
            us[no, i] = us[no, i - 1] + dt * u_rhs
            ps[no, i] = ps[no, i - 1] + dt * p_rhs
            ss[no, i] = ss[no, i - 1] + dt * s_rhs

            # Ornstein-Uhlenbeck process
            u_ou[no] = u_ou[no] + (u_ou_mean - u_ou[no]) * dt / tau_ou + sigma_ou * sqrt_dt * noise_us[no]  # mV/ms
            p_ou[no] = p_ou[no] + (p_ou_mean - p_ou[no]) * dt / tau_ou + sigma_ou * sqrt_dt * noise_ps[no]  # mV/ms
            s_ou[no] = s_ou[no] + (s_ou_mean - s_ou[no]) * dt / tau_ou + sigma_ou * sqrt_dt * noise_ss[no]  # mV/ms


    

    return t, us, ps, ss, u_ou, p_ou, s_ou
