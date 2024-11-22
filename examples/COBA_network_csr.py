# -*- coding: utf-8 -*-

import time

import brainstate as bst
import jax.lax
import jax.numpy as jnp
import numpy as np

import braintaichi

taum = 20
taue = 5
taui = 10
Vt = -50
Vr = -60
El = -60
Erev_exc = 0.
Erev_inh = -80.
Ib = 20.
ref = 5.0
we = 0.6
wi = 6.7


class LIF(bst.nn.Neuron):
    def __init__(self, size, V_init: callable, **kwargs):
        super(LIF, self).__init__(size, **kwargs)

        # parameters
        self.V_rest = Vr
        self.V_reset = El
        self.V_th = Vt
        self.tau = taum
        self.tau_ref = ref

        self.V_init = V_init

    def init_state(self, *args, **kwargs):
        # variables
        self.V = bst.init.state(self.V_init, self.varshape)
        self.spike = bst.init.state(lambda s: jnp.zeros(s, dtype=bool), self.varshape)
        self.t_last_spike = bst.init.state(bst.init.Constant(-1e7), self.varshape)

    def update(self, inp):
        inp = self.sum_current_inputs(inp, self.V.value)  # sum all projection inputs
        refractory = (bst.environ.get('t') - self.t_last_spike.value) <= self.tau_ref
        V = self.V.value + (-self.V.value + self.V_rest + inp) / self.tau * bst.environ.get_dt()
        V = self.sum_delta_inputs(V)
        V = jnp.where(refractory, self.V.value, V)
        spike = self.V_th <= V
        self.t_last_spike.value = jnp.where(spike, bst.environ.get('t'), self.t_last_spike.value)
        self.V.value = jnp.where(spike, self.V_reset, V)
        self.spike.value = spike
        return spike


class CSRLinear2(bst.nn.Module):
    def __init__(self, n_pre, n_post, g_max, prob):
        super().__init__()
        self.g_max = g_max
        self.n_pre = n_pre
        self.n_post = n_post
        self.indices = np.random.randint(0, n_post, size=(n_pre, int(n_post * prob)))

    def update(self, spk):
        def scan_fn(post, spi):
            sp, ids = spi
            post = jax.lax.cond(sp, lambda: post.at[ids].add(self.g_max), lambda _: post)
            return post, None

        return jax.lax.scan(scan_fn, jnp.zeros((self.n_post,)), (spk, self.indices))[0]


class CSRLinear(bst.nn.Module):
    def __init__(self, n_pre, n_post, g_max, prob):
        super().__init__()
        self.g_max = g_max
        self.n_pre = n_pre
        self.n_post = n_post
        self.prob = prob
        self.indices = np.random.randint(0, n_post, size=(n_pre, int(n_post * prob))).flatten()
        self.indptr = np.arange(0, n_pre + 1) * int(n_post * prob)

    def update(self, spk):
        return braintaichi.event_csrmv(
            self.g_max,
            self.indices,
            self.indptr,
            spk,
            transpose=True,
            shape=(self.n_pre, self.n_post)
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


class COBA(bst.nn.DynamicsGroup):
    def __init__(self, scale):
        super().__init__()
        self.num_exc = int(3200 * scale)
        self.num_inh = int(800 * scale)
        self.N = LIF(self.num_exc + self.num_inh, V_init=bst.init.Normal(-55., 5.))
        self.E = Exponential(self.num_exc, self.N, prob=80. / self.N.varshape[0], E=Erev_exc, g_max=we, tau=taue)
        self.I = Exponential(self.num_inh, self.N, prob=80. / self.N.varshape[0], E=Erev_inh, g_max=wi, tau=taui)

    def init_state(self, *args, **kwargs):
        self.rate = bst.init.state(jnp.zeros, self.N.varshape)

    def update(self, inp=Ib):
        self.E(self.N.spike.value[:self.num_exc])
        self.I(self.N.spike.value[self.num_exc:])
        self.N(inp)
        self.rate.value += self.N.spike.value

    def step_run(self, i):
        with bst.environ.context(i=i, t=i * bst.environ.get_dt()):
            self.update()


def run_a_simulation(scale=10, duration=1e3, ):
    net = COBA(scale=scale)
    bst.nn.init_all_states(net)
    indices = jnp.arange(int(duration / bst.environ.get_dt()))
    t0 = time.time()
    bst.compile.for_loop(net.step_run, indices)
    t1 = time.time()

    # running
    rate = net.rate.value.sum() / net.N.varshape[0] / duration * 1e3
    print(f'scale={scale}, size={net.N.varshape}, time = {t1 - t0} s, firing rate = {rate} Hz')


def check_firing_rate():
    with bst.environ.context(dt=0.1):
        for s in [1, 2, 4, 6, 8, 10, 20, 40, 60, 80, 100]:
            run_a_simulation(scale=s, duration=5e3)


if __name__ == '__main__':
    check_firing_rate()
