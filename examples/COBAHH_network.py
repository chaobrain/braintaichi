# -*- coding: utf-8 -*-

import time

import brainstate as bst
import brainunit as u

import braintaichi

s = 1e-2
Cm = 200 * s  # Membrane Capacitance [pF]
gl = 10. * s  # Leak Conductance   [nS]
g_Na = 20. * 1000 * s
g_Kd = 6. * 1000 * s  # K Conductance      [nS]
El = -60.  # Resting Potential [mV]
ENa = 50.  # reversal potential (Sodium) [mV]
EK = -90.  # reversal potential (Potassium) [mV]
VT = -63.
V_th = -20.
taue = 5.  # Excitatory synaptic time constant [ms]
taui = 10.  # Inhibitory synaptic time constant [ms]
Ee = 0.  # Excitatory reversal potential (mV)
Ei = -80.  # Inhibitory reversal potential (Potassium) [mV]
we = 6. * s  # excitatory synaptic conductance [nS]
wi = 67. * s  # inhibitory synaptic conductance [nS]


class HH(bst.nn.Neuron):
    def __init__(self, size, method='exp_auto'):
        super(HH, self).__init__(size)

    def init_state(self, *args, **kwargs):
        # variables
        self.V = bst.State(El + bst.random.randn(*self.varshape) * 5 - 5.)
        self.m = bst.State(u.math.zeros(self.varshape))
        self.n = bst.State(u.math.zeros(self.varshape))
        self.h = bst.State(u.math.zeros(self.varshape))
        self.rate = bst.State(u.math.zeros(self.varshape))
        self.spike = bst.State(u.math.zeros(self.varshape, dtype=bool))

    def dV(self, V, t, m, h, n, Isyn):
        Isyn = self.sum_current_inputs(Isyn, self.V.value)  # sum projection inputs
        gna = g_Na * (m * m * m) * h
        n2 = n * n
        gkd = g_Kd * (n2 * n2)
        dVdt = (-gl * (V - El) - gna * (V - ENa) - gkd * (V - EK) + Isyn) / Cm
        return dVdt

    def dm(self, m, t, V, ):
        m_alpha = 1.28 / u.math.exprel((13 - V + VT) / 4)
        m_beta = 1.4 / u.math.exprel((V - VT - 40) / 5)
        dmdt = (m_alpha * (1 - m) - m_beta * m)
        return dmdt

    def dh(self, h, t, V):
        h_alpha = 0.128 * u.math.exp((17 - V + VT) / 18)
        h_beta = 4. / (1 + u.math.exp(-(V - VT - 40) / 5))
        dhdt = (h_alpha * (1 - h) - h_beta * h)
        return dhdt

    def dn(self, n, t, V):
        n_alpha = 0.16 / u.math.exprel((15 - V + VT) / 5.)
        n_beta = 0.5 * u.math.exp((10 - V + VT) / 40)
        dndt = (n_alpha * (1 - n) - n_beta * n)
        return dndt

    def update(self, inp=0.):
        t = bst.environ.get('t')
        V = bst.nn.exp_euler_step(self.dV, self.V.value, t, self.m.value, self.h.value, self.n.value, inp)
        m = bst.nn.exp_euler_step(self.dm, self.m.value, t, self.V.value)
        n = bst.nn.exp_euler_step(self.dn, self.n.value, t, self.V.value)
        h = bst.nn.exp_euler_step(self.dh, self.h.value, t, self.V.value)
        self.spike.value = u.math.logical_and(self.V.value < V_th, V >= V_th)
        self.m.value = m
        self.h.value = h
        self.n.value = n
        self.V.value = V
        self.rate.value += self.spike.value
        return self.spike.value


class CSRLinear(bst.nn.Module):
    def __init__(self, n_pre, n_post, g_max, prob):
        super().__init__()
        self.g_max = g_max
        self.n_pre = n_pre
        self.n_post = n_post
        self.prob = prob

    def update(self, spk):
        return braintaichi.jitc_event_mv_prob_homo(
            spk, self.g_max, conn_prob=self.prob, shape=(self.n_pre, self.n_post,), seed=123, transpose=True
        )


class Exponential(bst.nn.Projection):
    def __init__(self, num_pre, post, prob, g_max, tau, E):
        super().__init__()

        self.proj = bst.nn.AlignPostProj(
            comm=CSRLinear(num_pre, post.varshape[0], g_max, prob),
            syn=bst.nn.Expon.desc(post.varshape, tau=tau, g_initializer=bst.init.ZeroInit()),
            out=bst.nn.COBA.desc(E=E),
            post=post
        )

    def update(self, spk):
        self.proj.update(spk)


class COBA_HH_Net(bst.nn.DynamicsGroup):
    def __init__(self, scale=1.):
        super(COBA_HH_Net, self).__init__()
        self.num_exc = int(3200 * scale)
        self.num_inh = int(800 * scale)
        self.num = self.num_exc + self.num_inh

        self.N = HH(self.num)
        self.E = Exponential(self.num_exc, self.N, prob=80 / self.num, g_max=we, tau=taue, E=Ee)
        self.I = Exponential(self.num_inh, self.N, prob=80 / self.num, g_max=wi, tau=taui, E=Ei)

    def update(self):
        self.E(self.N.spike.value[:self.num_exc])
        self.I(self.N.spike.value[self.num_exc:])
        self.N()

    def step_run(self, i):
        with bst.environ.context(i=i, t=i * bst.environ.get_dt()):
            self.update()


def run_a_simulation(scale=10, duration=1e3):
    net = COBA_HH_Net(scale=scale)
    bst.nn.init_all_states(net)

    indices = u.math.arange(int(duration / bst.environ.get_dt()))

    t0 = time.time()
    # if the network size is big, please turn on "progress_bar"
    # otherwise, the XLA may compute wrongly
    r = bst.compile.for_loop(net.step_run, indices)
    t1 = time.time()

    rate = net.N.rate.value.sum() / net.num / duration * 1e3

    print(f'scale={scale}, size={net.num}, time = {t1 - t0} s, firing rate = {rate} Hz')


def check_firing_rate(x64=True, platform='cpu'):
    with bst.environ.context(dt=0.1):
        for scale in [1, 2, 4, 6, 8, 10, 20, 30, 40, 50, 80, 100]:
            run_a_simulation(scale=scale, duration=2e3)


if __name__ == '__main__':
    check_firing_rate()
