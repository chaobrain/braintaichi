import time
import functools

import brainpy as bp
import brainpy.math as bm
from brainpy import neurons

import matplotlib.pyplot as plt
import numpy as np
from jax import vmap
from scipy.io import loadmat


def syn_fun(pre_spike, weight, seed, prob, shape):
  return bm.jitconn.event_mv_prob_homo(pre_spike, weight, prob, seed, shape=shape, transpose=True)


class MultiAreaNet(bp.Network):
  def __init__(
      self, hier, conn, delay_mat, muIE=0.0475, muEE=.0375, wII=.075,
      wEE=.01, wIE=.075, wEI=.0375, extE=15.4, extI=14.0, alpha=4.,
  ):
    super(MultiAreaNet, self).__init__()

    # data
    self.hier = hier
    self.conn = conn
    self.delay_mat = delay_mat

    # parameters
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
    self.num_E = 1600
    self.num_I = 400

    # neuron models
    self.E = neurons.LIF((num_area, 1600),
                         V_th=-50., V_reset=-60.,
                         V_rest=-70., tau=20., tau_ref=2.,
                         noise=3. / bm.sqrt(20.),
                         V_initializer=bp.init.Uniform(-70., -50.),
                         method='exp_auto',
                         keep_size=True,
                         ref_var=True)
    self.I = neurons.LIF((num_area, 400), V_th=-50., V_reset=-60.,
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
    self.f_EE_current = vmap(functools.partial(syn_fun, prob=0.1, shape=(self.num_E, self.num_E)))
    self.f_EI_current = vmap(functools.partial(syn_fun, prob=0.1, shape=(self.num_E, self.num_I)))
    self.f_IE_current = vmap(functools.partial(syn_fun, prob=0.1, shape=(self.num_I, self.num_E)),
                             in_axes=(0, None, 0))
    self.f_II_current = vmap(functools.partial(syn_fun, prob=0.1, shape=(self.num_I, self.num_I)),
                             in_axes=(0, None, 0))

    # synapses from I
    self.intra_I2E_conn = bm.random.random((num_area, 400, 1600)) < 0.1
    self.intra_I2I_conn = bm.random.random((num_area, 400, 400)) < 0.1
    self.intra_I2E_weight = -wEI
    self.intra_I2I_weight = -wII

    # synapses from E
    # self.E2E_conns = [bm.random.random((num_area, 1600, 1600)) < 0.1 for _ in range(num_area)]
    self.E2I_conns = [bm.random.random((num_area, 1600, 400)) < 0.1 for _ in range(num_area)]
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


# hierarchy values
hierVals = loadmat('Matlab/hierValspython.mat')
hierValsnew = hierVals['hierVals'].flatten()
hier = bm.asarray(hierValsnew / max(hierValsnew))  # hierarchy normalized.

# fraction of labeled neurons
flnMatp = loadmat('Matlab/efelenMatpython.mat')
conn = bm.asarray(flnMatp['flnMatpython'].squeeze())  # fln values..Cij is strength from j to i

# Distance
speed = 3.5  # axonal conduction velocity
distMatp = loadmat('Matlab/subgraphWiring29.mat')
distMat = distMatp['wiring'].squeeze()  # distances between areas values..
delayMat = bm.asarray(distMat / speed)

pars = dict(extE=14.2, extI=14.7, wII=.075, wEE=.01, wIE=.075, wEI=.0375, muEE=.0375, muIE=0.0475)
inps = dict(value=15, duration=150)

inputs, length = bp.inputs.section_input(values=[0, inps['value'], 0.],
                                         durations=[300., inps['duration'], 500],
                                         return_length=True)

# bm.set_platform('cpu')

net = MultiAreaNet(hier, conn, delayMat, **pars)
step_run = bm.jit(net.step_run)

t0 = time.time()
step_run(0, 0.)
t1 = time.time()

t2 = time.time()
step_run(1, 0.)
t3 = time.time()

print(f'Compilation time {(t1 - t0) - (t3 - t2)} s')
