from time import time
import functools

import brainpy as bp
import brainpy.math as bm
from brainpy import neurons

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jax import vmap
import os
from scipy.io import loadmat

from multiple_area_customized_op import (multiple_area_customized_op_E as syn_fun_E,
                                          multiple_area_customized_op_I as syn_fun_I)

bm.set_platform('gpu')
bm.disable_gpu_memory_preallocation()

PATH = os.path.dirname(os.path.abspath(__file__))

neuron_nums = [
    200, 
    2000, 
    20000, 
    200000, 
    # 2000000
]
conn_nums = [
    20, 
    40, 
    80,
    100 
]

E_PROB = 0.8
I_PROB = 0.2
muIE = 0.0475
muEE = 0.0375
wII = 0.075
wEE = 0.01
wIE = 0.075
wEI = 0.0375
extE = 14.2
extI = 14.7
alpha = 4.

RUNNING_TIME = 10
if bm.get_platform() == 'cpu':
    RUNNING_TIME = 3

syn_func_types = [
    'event_mv_prob_homo',
    # 'event_mv_prob_uniform',
    # 'event_mv_prob_normal',
]

# hierarchy values
hierVals = loadmat(f'{PATH}/Matlab/hierValspython.mat')
hierValsnew = hierVals['hierVals'].flatten()
hier = bm.asarray(hierValsnew / max(hierValsnew))  # hierarchy normalized.

# fraction of labeled neurons
flnMatp = loadmat(f'{PATH}/Matlab/efelenMatpython.mat')
conn = bm.asarray(flnMatp['flnMatpython'].squeeze())  # fln values..Cij is strength from j to i

# Distance
speed = 3.5  # axonal conduction velocity
distMatp = loadmat(f'{PATH}/Matlab/subgraphWiring29.mat')
distMat = distMatp['wiring'].squeeze()  # distances between areas values..
delayMat = bm.asarray(distMat / speed)


class MultiAreaNet_brainpylib(bp.Network):
    def __init__(
            self, hier, conn, delay_mat, neuron_num, conn_prob, syn_fun_type, syn_fun
    ):
        super(MultiAreaNet_brainpylib, self).__init__()

        # data
        self.hier = hier
        self.conn = conn
        self.delay_mat = delay_mat

        # parameters
        self.exc_num = int(neuron_num * E_PROB)
        self.inh_num = int(neuron_num * I_PROB)
        self.conn_prob = conn_prob
        self.syn_fun_type = syn_fun_type

        self.muIE = muIE
        self.muEE = muEE
        self.wII = wII
        self.wEE = wEE
        self.wIE = wIE
        self.wEI = wEI
        self.extE = extE
        self.extI = extI
        self.alpha = alpha
        num_area = hier.size
        self.num_area = num_area
        self.num_E = self.exc_num
        self.num_I = self.inh_num

        # neuron models
        self.E = neurons.LIF((num_area, self.exc_num),
                             V_th=-50., V_reset=-60.,
                             V_rest=-70., tau=20., tau_ref=2.,
                             noise=3. / bm.sqrt(20.),
                             V_initializer=bp.init.Uniform(-70., -50.),
                             method='exp_auto',
                             keep_size=True,
                             ref_var=True)
        self.I = neurons.LIF((num_area, self.inh_num), V_th=-50., V_reset=-60.,
                             V_rest=-70., tau=10., tau_ref=2., noise=3. / bm.sqrt(10.),
                             V_initializer=bp.init.Uniform(-70., -50.),
                             method='exp_auto',
                             keep_size=True,
                             ref_var=True)

        # delays
        self.intra_delay_step = int(2. / bm.get_dt())
        self.E_delay_steps = bm.asarray(delay_mat.T / bm.get_dt(), dtype=int)
        bm.fill_diagonal(self.E_delay_steps, self.intra_delay_step)
        self.Edelay = bm.LengthDelay(self.E.spike, delay_len=int(self.E_delay_steps.max()))
        self.Idelay = bm.LengthDelay(self.I.spike, delay_len=self.intra_delay_step)

        # synapse model
        self.f_EE_current = vmap(functools.partial(syn_fun, prob=conn_prob, shape=(self.num_E, self.num_E)))
        self.f_EI_current = vmap(functools.partial(syn_fun, prob=conn_prob, shape=(self.num_E, self.num_I)))
        self.f_IE_current = vmap(functools.partial(syn_fun, prob=conn_prob, shape=(self.num_I, self.num_E)),
                                in_axes=(0, None, 0))
        self.f_II_current = vmap(functools.partial(syn_fun, prob=conn_prob, shape=(self.num_I, self.num_I)),
                                in_axes=(0, None, 0))


        # synapses from I
        # self.intra_I2E_conn = bm.random.random((num_area, self.inh_num, self.exc_num)) < conn_prob
        # self.intra_I2I_conn = bm.random.random((num_area, self.inh_num, self.inh_num)) < conn_prob
        self.intra_I2E_weight = -wEI
        self.intra_I2I_weight = -wII

        # synapses from E
        # self.E2E_conns = [bm.random.random((num_area, 1600, 1600)) < 0.1 for _ in range(num_area)]
        # self.E2I_conns = [bm.random.random((num_area, self.exc_num, self.inh_num)) < conn_prob for _ in range(num_area)]
        self.E2E_weights = (1 + alpha * hier) * muEE * conn.T  # inter-area connections
        bm.fill_diagonal(self.E2E_weights, (1 + alpha * hier) * wEE)  # intra-area connections
        self.E2I_weights = (1 + alpha * hier) * muIE * conn.T  # inter-area connections
        bm.fill_diagonal(self.E2I_weights, (1 + alpha * hier) * wIE)  # intra-area connections
        self.E_seeds = bm.random.randint(0, 100000, (num_area, num_area * 2))
        self.I_seeds = bm.random.randint(0, 100000, (num_area * 2))

    def update(self, v1_input):
        self.E.input[0] += v1_input
        self.E.input += self.extE
        self.I.input += self.extI
        E_not_ref = bm.logical_not(self.E.refractory)
        I_not_ref = bm.logical_not(self.I.refractory)

        if self.syn_fun_type == 'event_mv_prob_homo':
            # synapses from E
            for i in range(self.num_area):
                delayed_E_spikes = self.Edelay(self.E_delay_steps[i], i).astype(float)
                current = self.f_EE_current(delayed_E_spikes, self.E2E_weights[i], self.E_seeds[i, :self.num_area])
                self.E.V += current * E_not_ref  # E2E
                current = self.f_EI_current(delayed_E_spikes, self.E2I_weights[i], self.E_seeds[i, self.num_area:])
                self.I.V += current * I_not_ref  # E2I

            # synapses from I
            delayed_I_spikes = self.Idelay(self.intra_delay_step).astype(float)
            current = self.f_IE_current(delayed_I_spikes, self.intra_I2E_weight, self.I_seeds[:self.num_area])
            self.E.V += current * E_not_ref  # I2E
            current = self.f_II_current(delayed_I_spikes, self.intra_I2I_weight, self.I_seeds[self.num_area:])
            self.I.V += current * I_not_ref  # I2I
        elif self.syn_fun_type == 'event_mv_prob_uniform':
            # synapses from E
            for i in range(self.num_area):
                delayed_E_spikes = self.Edelay(self.E_delay_steps[i], i).astype(float)
                current = self.f_EE_current(delayed_E_spikes, self.E2E_weights[i] - 0.03, self.E2E_weights[i] + 0.03, self.E_seeds[i, :self.num_area])
                self.E.V += current * E_not_ref  # E2E
                current = self.f_EI_current(delayed_E_spikes, self.E2E_weights[i] - 0.03, self.E2E_weights[i] + 0.03, self.E_seeds[i, self.num_area:])
                self.I.V += current * I_not_ref  # E2I

            # synapses from I
            delayed_I_spikes = self.Idelay(self.intra_delay_step).astype(float)
            current = self.f_IE_current(delayed_I_spikes, self.intra_I2E_weight - 0.03, self.intra_I2E_weight + 0.03, self.I_seeds[:self.num_area])
            self.E.V += current * E_not_ref  # I2E
            current = self.f_II_current(delayed_I_spikes, self.intra_I2E_weight - 0.03, self.intra_I2E_weight + 0.03, self.I_seeds[self.num_area:])
            self.I.V += current * I_not_ref  # I2I
        elif self.syn_fun_type == 'event_mv_prob_normal':
            # synapses from E
            for i in range(self.num_area):
                delayed_E_spikes = self.Edelay(self.E_delay_steps[i], i).astype(float)
                current = self.f_EE_current(delayed_E_spikes, self.E2E_weights[i], 0.01, self.E_seeds[i, :self.num_area])
                self.E.V += current * E_not_ref  # E2E
                current = self.f_EI_current(delayed_E_spikes, self.E2I_weights[i], 0.01, self.E_seeds[i, self.num_area:])
                self.I.V += current * I_not_ref  # E2I

            # synapses from I
            delayed_I_spikes = self.Idelay(self.intra_delay_step).astype(float)
            current = self.f_IE_current(delayed_I_spikes, self.intra_I2E_weight, 0.01, self.I_seeds[:self.num_area])
            self.E.V += current * E_not_ref  # I2E
            current = self.f_II_current(delayed_I_spikes, self.intra_I2I_weight, 0.01, self.I_seeds[self.num_area:])
            self.I.V += current * I_not_ref  # I2I

        # updates
        self.Edelay.update(self.E.spike)
        self.Idelay.update(self.I.spike)
        self.E.update()
        self.I.update()


class MultiAreaNet_taichi(bp.Network):
    def __init__(
            self, hier, conn, delay_mat, neuron_num, conn_prob, syn_fun_type, syn_fun_E, syn_fun_I
    ):
        super(MultiAreaNet_taichi, self).__init__()

        # data
        self.hier = hier
        self.conn = conn
        self.delay_mat = delay_mat

        # parameters
        self.exc_num = int(neuron_num * E_PROB)
        self.inh_num = int(neuron_num * I_PROB)
        self.conn_prob = conn_prob
        self.syn_fun_type = syn_fun_type

        self.muIE = muIE
        self.muEE = muEE
        self.wII = wII
        self.wEE = wEE
        self.wIE = wIE
        self.wEI = wEI
        self.extE = extE
        self.extI = extI
        self.alpha = alpha
        num_area = hier.size
        self.num_area = num_area
        self.num_E = self.exc_num
        self.num_I = self.inh_num

        # neuron models
        self.E = neurons.LIF((num_area, self.exc_num),
                             V_th=-50., V_reset=-60.,
                             V_rest=-70., tau=20., tau_ref=2.,
                             noise=3. / bm.sqrt(20.),
                             V_initializer=bp.init.Uniform(-70., -50.),
                             method='exp_auto',
                             keep_size=True,
                             ref_var=True)
        self.I = neurons.LIF((num_area, self.inh_num), V_th=-50., V_reset=-60.,
                             V_rest=-70., tau=10., tau_ref=2., noise=3. / bm.sqrt(10.),
                             V_initializer=bp.init.Uniform(-70., -50.),
                             method='exp_auto',
                             keep_size=True,
                             ref_var=True)

        # delays
        self.intra_delay_step = int(2. / bm.get_dt())
        self.E_delay_steps = bm.asarray(delay_mat.T / bm.get_dt(), dtype=int)
        bm.fill_diagonal(self.E_delay_steps, self.intra_delay_step)
        self.Edelay = bm.LengthDelay(self.E.spike, delay_len=int(self.E_delay_steps.max()))
        self.Idelay = bm.LengthDelay(self.I.spike, delay_len=self.intra_delay_step)

        # synapse model
        self.f_EE_current = functools.partial(syn_fun_E, prob=conn_prob, shape=(self.num_E, self.num_E), area_num=num_area)
        self.f_EI_current = functools.partial(syn_fun_E, prob=conn_prob, shape=(self.num_E, self.num_I), area_num=num_area)
        self.f_IE_current = functools.partial(syn_fun_I, prob=conn_prob, shape=(self.num_I, self.num_E), area_num=num_area)
        self.f_II_current = functools.partial(syn_fun_I, prob=conn_prob, shape=(self.num_I, self.num_I), area_num=num_area)


        # synapses from I
        # self.intra_I2E_conn = bm.random.random((num_area, self.inh_num, self.exc_num)) < conn_prob
        # self.intra_I2I_conn = bm.random.random((num_area, self.inh_num, self.inh_num)) < conn_prob
        self.intra_I2E_weight = -wEI
        self.intra_I2I_weight = -wII

        # synapses from E
        # self.E2E_conns = [bm.random.random((num_area, 1600, 1600)) < 0.1 for _ in range(num_area)]
        # self.E2I_conns = [bm.random.random((num_area, self.exc_num, self.inh_num)) < conn_prob for _ in range(num_area)]
        self.E2E_weights = (1 + alpha * hier) * muEE * conn.T  # inter-area connections
        bm.fill_diagonal(self.E2E_weights, (1 + alpha * hier) * wEE)  # intra-area connections
        self.E2I_weights = (1 + alpha * hier) * muIE * conn.T  # inter-area connections
        bm.fill_diagonal(self.E2I_weights, (1 + alpha * hier) * wIE)  # intra-area connections
        self.E_seeds = bm.random.randint(0, 100000, (num_area, num_area * 2))
        self.I_seeds = bm.random.randint(0, 100000, (num_area * 2))

    def update(self, v1_input):
        self.E.input[0] += v1_input
        self.E.input += self.extE
        self.I.input += self.extI
        E_not_ref = bm.logical_not(self.E.refractory)
        I_not_ref = bm.logical_not(self.I.refractory)

        if self.syn_fun_type == 'event_mv_prob_homo':
            # synapses from E
            for i in range(self.num_area):
                delayed_E_spikes = self.Edelay(self.E_delay_steps[i], i).astype(float)
                delayed_E_spikes = delayed_E_spikes[i]
                current = self.f_EE_current(delayed_E_spikes, self.E2E_weights[i], self.E_seeds[i, :self.num_area])
                self.E.V += current * E_not_ref  # E2E
                current = self.f_EI_current(delayed_E_spikes, self.E2I_weights[i], self.E_seeds[i, self.num_area:])
                self.I.V += current * I_not_ref  # E2I

            # synapses from I
            delayed_I_spikes = self.Idelay(self.intra_delay_step).astype(float)
            current = self.f_IE_current(delayed_I_spikes, self.intra_I2E_weight, self.I_seeds[:self.num_area])
            self.E.V += current * E_not_ref  # I2E
            current = self.f_II_current(delayed_I_spikes, self.intra_I2I_weight, self.I_seeds[self.num_area:])
            self.I.V += current * I_not_ref  # I2I
        elif self.syn_fun_type == 'event_mv_prob_uniform':
            # synapses from E
            for i in range(self.num_area):
                delayed_E_spikes = self.Edelay(self.E_delay_steps[i], i).astype(float)
                current = self.f_EE_current(delayed_E_spikes, self.E2E_weights[i] - 0.03, self.E2E_weights[i] + 0.03, self.E_seeds[i, :self.num_area])
                self.E.V += current * E_not_ref  # E2E
                current = self.f_EI_current(delayed_E_spikes, self.E2E_weights[i] - 0.03, self.E2E_weights[i] + 0.03, self.E_seeds[i, self.num_area:])
                self.I.V += current * I_not_ref  # E2I

            # synapses from I
            delayed_I_spikes = self.Idelay(self.intra_delay_step).astype(float)
            current = self.f_IE_current(delayed_I_spikes, self.intra_I2E_weight - 0.03, self.intra_I2E_weight + 0.03, self.I_seeds[:self.num_area])
            self.E.V += current * E_not_ref  # I2E
            current = self.f_II_current(delayed_I_spikes, self.intra_I2E_weight - 0.03, self.intra_I2E_weight + 0.03, self.I_seeds[self.num_area:])
            self.I.V += current * I_not_ref  # I2I
        elif self.syn_fun_type == 'event_mv_prob_normal':
            # synapses from E
            for i in range(self.num_area):
                delayed_E_spikes = self.Edelay(self.E_delay_steps[i], i).astype(float)
                current = self.f_EE_current(delayed_E_spikes, self.E2E_weights[i], 0.01, self.E_seeds[i, :self.num_area])
                self.E.V += current * E_not_ref  # E2E
                current = self.f_EI_current(delayed_E_spikes, self.E2I_weights[i], 0.01, self.E_seeds[i, self.num_area:])
                self.I.V += current * I_not_ref  # E2I

            # synapses from I
            delayed_I_spikes = self.Idelay(self.intra_delay_step).astype(float)
            current = self.f_IE_current(delayed_I_spikes, self.intra_I2E_weight, 0.01, self.I_seeds[:self.num_area])
            self.E.V += current * E_not_ref  # I2E
            current = self.f_II_current(delayed_I_spikes, self.intra_I2I_weight, 0.01, self.I_seeds[self.num_area:])
            self.I.V += current * I_not_ref  # I2I

        # updates
        self.Edelay.update(self.E.spike)
        self.Idelay.update(self.I.spike)
        self.E.update()
        self.I.update()

def raster_plot(xValues, yValues, duration):
    ticks = np.round(np.arange(0, 29) + 0.5, 2)
    areas = ['V1', 'V2', 'V4', 'DP', 'MT', '8m', '5', '8l', 'TEO', '2', 'F1',
             'STPc', '7A', '46d', '10', '9/46v', '9/46d', 'F5', 'TEpd', 'PBr',
             '7m', '7B', 'F2', 'STPi', 'PROm', 'F7', '8B', 'STPr', '24c']
    N = len(ticks)
    plt.figure(figsize=(8, 6))
    plt.plot(xValues, yValues / (4 * 400), '.', markersize=1)
    plt.plot([0, duration], np.arange(N + 1).repeat(2).reshape(-1, 2).T, 'k-')
    plt.ylabel('Area')
    plt.yticks(np.arange(N))
    plt.xlabel('Time [ms]')
    plt.ylim(0, N)
    plt.yticks(ticks, areas)
    plt.xlim(0, duration)
    plt.tight_layout()
    plt.show()


def syn_fun_homo_brainpylib(pre_spike, weight, seed, prob, shape):
    return bm.jitconn.event_mv_prob_homo(pre_spike, weight, prob, seed, shape=shape, transpose=True)

def syn_fun_homo_taichi_E(pre_spike, weight, seed, prob, shape, area_num):
    return syn_fun_E(pre_spike, weight, prob, seed, shape=shape, area_num=area_num, transpose=True)

def syn_fun_homo_taichi_I(pre_spike, weight, seed, prob, shape, area_num):
    return syn_fun_I(pre_spike, weight, prob, seed, shape=shape, area_num=area_num, transpose=True)


def syn_fun_uniform_brainpylib(pre_spike, w_low, w_high, seed, prob, shape):
    return bm.jitconn.event_mv_prob_uniform(pre_spike, w_low, w_high, prob, seed, shape=shape, transpose=True)


def syn_fun_uniform_taichi(pre_spike, w_low, w_high, seed, prob, shape):
    return bm.jitconn.event_mv_prob_uniform_taichi(pre_spike, w_low, w_high, prob, seed, shape=shape, transpose=True)


def syn_fun_normal_brainpylib(pre_spike, w_mu, w_sigma, prob, seed, shape):
    return bm.jitconn.event_mv_prob_normal(pre_spike, w_mu, w_sigma, prob, seed, shape=shape, transpose=True)


def syn_fun_normal_taichi(pre_spike, w_mu, w_sigma, prob, seed, shape):
    return bm.jitconn.event_mv_prob_normal_taichi(pre_spike, w_mu, w_sigma, prob, seed, shape=shape, transpose=True)


def test(neuron_num, conn_num, syn_func_type):
    print(f'neuron nums: {neuron_num}, conn num: {conn_num} syn func type: {syn_func_type}')
    conn_prob = conn_num / neuron_num
    if syn_func_type == 'event_mv_prob_homo':
        net_brainpylib = MultiAreaNet_brainpylib(hier, conn, delayMat, neuron_num, conn_prob, syn_func_type, syn_fun_homo_brainpylib)
        net_taichi = MultiAreaNet_taichi(hier, conn, delayMat, neuron_num, conn_prob, syn_func_type, syn_fun_homo_taichi_E, syn_fun_homo_taichi_I)
    elif syn_func_type == 'event_mv_prob_uniform':
        net_brainpylib = MultiAreaNet_brainpylib(hier, conn, delayMat, neuron_num, conn_prob, syn_func_type, syn_fun_uniform_brainpylib)
        net_taichi = MultiAreaNet_taichi(hier, conn, delayMat, neuron_num, conn_prob, syn_func_type, syn_fun_uniform_taichi)
    elif syn_func_type == 'event_mv_prob_normal':
        net_brainpylib = MultiAreaNet_brainpylib(hier, conn, delayMat, neuron_num, conn_prob, syn_func_type, syn_fun_normal_brainpylib)
        net_taichi = MultiAreaNet_taichi(hier, conn, delayMat, neuron_num, conn_prob, syn_func_type, syn_fun_normal_taichi)
    else:
        raise NotImplementedError
    
    indices = bm.arange(RUNNING_TIME)
    
    bm.for_loop(lambda i: net_brainpylib.step_run(i, 20.), indices)
    time0 = time()
    bm.for_loop(lambda i: net_brainpylib.step_run(i, 20.), indices)
    time1 = time()
    time2 = time()
    bm.for_loop(lambda i: net_brainpylib.step_run(i, 20.), indices)
    time3 = time()
    time4 = time()
    bm.for_loop(lambda i: net_brainpylib.step_run(i, 20.), indices)
    time5 = time()
    time6 = time()
    bm.for_loop(lambda i: net_brainpylib.step_run(i, 20.), indices)
    time7 = time()
    time8 = time()
    bm.for_loop(lambda i: net_brainpylib.step_run(i, 20.), indices)
    time9 = time()
    time10 = time()
    bm.for_loop(lambda i: net_brainpylib.step_run(i, 20.), indices)
    time11 = time()
    time12 = time()
    bm.for_loop(lambda i: net_brainpylib.step_run(i, 20.), indices)
    time13 = time()
    time14 = time()
    bm.for_loop(lambda i: net_brainpylib.step_run(i, 20.), indices)
    time15 = time()
    time16 = time()
    bm.for_loop(lambda i: net_brainpylib.step_run(i, 20.), indices)
    time17 = time()
    time18 = time()
    bm.for_loop(lambda i: net_brainpylib.step_run(i, 20.), indices)
    time19 = time()

    bm.for_loop(lambda i: net_taichi.step_run(i, 20.), indices)
    time20 = time()
    bm.for_loop(lambda i: net_taichi.step_run(i, 20.), indices)
    time21 = time()
    time22 = time()
    bm.for_loop(lambda i: net_taichi.step_run(i, 20.), indices)
    time23 = time()
    time24 = time()
    bm.for_loop(lambda i: net_taichi.step_run(i, 20.), indices)
    time25 = time()
    time26 = time()
    bm.for_loop(lambda i: net_taichi.step_run(i, 20.), indices)
    time27 = time()
    time28 = time()
    bm.for_loop(lambda i: net_taichi.step_run(i, 20.), indices)
    time29 = time()
    time30 = time()
    bm.for_loop(lambda i: net_taichi.step_run(i, 20.), indices)
    time31 = time()
    time32 = time()
    bm.for_loop(lambda i: net_taichi.step_run(i, 20.), indices)
    time33 = time()
    time34 = time()
    bm.for_loop(lambda i: net_taichi.step_run(i, 20.), indices)
    time35 = time()
    time36 = time()
    bm.for_loop(lambda i: net_taichi.step_run(i, 20.), indices)
    time37 = time()
    time38 = time()
    bm.for_loop(lambda i: net_taichi.step_run(i, 20.), indices)
    time39 = time()

    print(f'BrainPylib: {(time1 - time0)*1000}, {(time3 - time2)*1000}, {(time5 - time4)*1000}, {(time7 - time6)*1000}, {(time9 - time8)*1000}, {(time11 - time10)*1000}, {(time13 - time12)*1000}, {(time15 - time14)*1000}, {(time17 - time16)*1000}, {(time19 - time18)*1000}')
    print(f'Taichi: {(time21 - time20)*1000}, {(time23 - time22)*1000}, {(time25 - time24)*1000}, {(time27 - time26)*1000}, {(time29 - time28)*1000}, {(time31 - time30)*1000}, {(time33 - time32)*1000}, {(time35 - time34)*1000}, {(time37 - time36)*1000}, {(time39 - time38)*1000}')

    return (time1 - time0)*1000, (time3 - time2)*1000, (time5 - time4)*1000, (time7 - time6)*1000, (time9 - time8)*1000, (time11 - time10)*1000, (time13 - time12)*1000, (time15 - time14)*1000, (time17 - time16)*1000, (time19 - time18)*1000, \
        (time21 - time20)*1000, (time23 - time22)*1000, (time25 - time24)*1000, (time27 - time26)*1000, (time29 - time28)*1000, (time31 - time30)*1000, (time33 - time32)*1000, (time35 - time34)*1000, (time37 - time36)*1000, (time39 - time38)*1000

df = pd.DataFrame(columns=['neuron_num', 'conn_num', 'syn_func_type', 'device',
                           'brainpylib_time1', 'brainpylib_time2', 'brainpylib_time3', 'brainpylib_time4', 'brainpylib_time5', 'brainpylib_time6', 'brainpylib_time7', 'brainpylib_time8', 'brainpylib_time9', 'brainpylib_time10',
                           'taichi_time1', 'taichi_time2', 'taichi_time3', 'taichi_time4', 'taichi_time5', 'taichi_time6', 'taichi_time7', 'taichi_time8', 'taichi_time9', 'taichi_time10'])


if bm.get_platform() == 'cpu':
    for neuron_num in neuron_nums:
        for conn_num in conn_nums:
            for syn_func_type in syn_func_types:
                brainpylib_time1, brainpylib_time2, brainpylib_time3, brainpylib_time4, brainpylib_time5, brainpylib_time6, brainpylib_time7, brainpylib_time8, brainpylib_time9, brainpylib_time10, \
                    taichi_time1, taichi_time2, taichi_time3, taichi_time4, taichi_time5, taichi_time6, taichi_time7, taichi_time8, taichi_time9, taichi_time10 = test(neuron_num, conn_num, syn_func_type)
                df.loc[df.shape[0]] = [neuron_num, conn_num, syn_func_type, 'cpu',
                                    brainpylib_time1, brainpylib_time2, brainpylib_time3, brainpylib_time4, brainpylib_time5, brainpylib_time6, brainpylib_time7, brainpylib_time8, brainpylib_time9, brainpylib_time10,
                                    taichi_time1, taichi_time2, taichi_time3, taichi_time4, taichi_time5, taichi_time6, taichi_time7, taichi_time8, taichi_time9, taichi_time10]
                df.to_csv(f'{PATH}/benchmark_MultipleArea_cpu.csv', index=False)

if bm.get_platform() == 'gpu':
    for neuron_num in neuron_nums:
        for conn_num in conn_nums:
            for syn_func_type in syn_func_types:
                brainpylib_time1, brainpylib_time2, brainpylib_time3, brainpylib_time4, brainpylib_time5, brainpylib_time6, brainpylib_time7, brainpylib_time8, brainpylib_time9, brainpylib_time10, \
                    taichi_time1, taichi_time2, taichi_time3, taichi_time4, taichi_time5, taichi_time6, taichi_time7, taichi_time8, taichi_time9, taichi_time10 = test(neuron_num, conn_num, syn_func_type)
                df.loc[df.shape[0]] = [neuron_num, conn_num, syn_func_type, 'gpu',
                                    brainpylib_time1, brainpylib_time2, brainpylib_time3, brainpylib_time4, brainpylib_time5, brainpylib_time6, brainpylib_time7, brainpylib_time8, brainpylib_time9, brainpylib_time10,
                                    taichi_time1, taichi_time2, taichi_time3, taichi_time4, taichi_time5, taichi_time6, taichi_time7, taichi_time8, taichi_time9, taichi_time10]
                df.to_csv(f'{PATH}/benchmark_MultipleArea_gpu.csv', index=False)
                

