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

E_PROB = 0.75
I_PROB = 0.25
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
    RUNNING_TIME = 10

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
        return self.E.spike.value, self.I.spike.value


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

        return self.E.spike.value, self.I.spike.value

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
    print(f'neuron nums: {neuron_num}, conn num: {conn_num} sys func type: {syn_func_type}')
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

    brainpylib_e_spks, brainpylib_i_spks = bm.for_loop(lambda i: net_brainpylib.step_run(i, 20.), indices)

    taichi_e_spks, taichi_i_spks = bm.for_loop(lambda i: net_taichi.step_run(i, 20.), indices)

    brainpylib_e_spks = brainpylib_e_spks.sum() / (RUNNING_TIME/1000) / net_brainpylib.num_E
    brainpylib_i_spks = brainpylib_i_spks.sum() / (RUNNING_TIME/1000) / net_brainpylib.num_I
    taichi_e_spks = taichi_e_spks.sum() / (RUNNING_TIME/1000) / net_taichi.num_E
    taichi_i_spks = taichi_i_spks.sum() / (RUNNING_TIME/1000) / net_taichi.num_I

    return brainpylib_e_spks, brainpylib_i_spks, taichi_e_spks, taichi_i_spks


df = pd.DataFrame(columns=['neuron_num', 'conn_num', 'syn_func_type', 'device',
                           'brainpylib_e_spk', 'brainpylib_i_spk',
                           'taichi_e_spk', 'taichi_i_spk'])


if bm.get_platform() == 'cpu':
    for neuron_num in neuron_nums:
        for conn_num in conn_nums:
            for syn_func_type in syn_func_types:
                brainpylib_e_spks, brainpylib_i_spks, taichi_e_spks, taichi_i_spks = test(neuron_num, conn_num, syn_func_type)
                df.loc[df.shape[0]] = [neuron_num, conn_num, syn_func_type, 'cpu',
                                    brainpylib_e_spks, brainpylib_i_spks, taichi_e_spks, taichi_i_spks]
                df.to_csv(f'{PATH}/benchmark_MultipleArea_spks_cpu.csv', index=False)

if bm.get_platform() == 'gpu':
    for neuron_num in neuron_nums:
        for conn_num in conn_nums:
            for syn_func_type in syn_func_types:
                brainpylib_e_spks, brainpylib_i_spks, taichi_e_spks, taichi_i_spks = test(neuron_num, conn_num, syn_func_type)
                df.loc[df.shape[0]] = [neuron_num, conn_num, syn_func_type, 'gpu',
                                    brainpylib_e_spks, brainpylib_i_spks, taichi_e_spks, taichi_i_spks]
                df.to_csv(f'{PATH}/benchmark_MultipleArea_spks_gpu.csv', index=False)