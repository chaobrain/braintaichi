import brainpy as bp
import brainpy.math as bm
import pandas as pd
import os
from time import time
from functools import partial
from linear import EventCSRLinear, EventJitFPHomoLinear, EventJitFPUniformLinear, EventJitFPNormalLinear

bm.set_platform('gpu')
bm.disable_gpu_memory_preallocation()


EventCSRLinear_brainpylib = partial(EventCSRLinear, method='brainpylib')
EventCSRLinear_taichi = partial(EventCSRLinear, method=None)
EventJitFPHomoLinear_brainpylib = partial(EventJitFPHomoLinear, method='brainpylib')
EventJitFPHomoLinear_taichi = partial(EventJitFPHomoLinear, method=None)
EventJitFPUniformLinear_brainpylib = partial(EventJitFPUniformLinear, method='brainpylib')
EventJitFPUniformLinear_taichi = partial(EventJitFPUniformLinear, method=None)
EventJitFPNormalLinear_brainpylib = partial(EventJitFPNormalLinear, method='brainpylib')
EventJitFPNormalLinear_taichi = partial(EventJitFPNormalLinear, method=None)


post_nums = [400, 4000, 40000, 400000, 4000000]
conn_nums = [20, 40, 80, 100]
# conn_nums = [80]
comm_types = {
    # 'CSRLinear': (bp.dnn.CSRLinear, bp.dnn.CSRLinear_taichi),
    # 'EventCSRLinear': (EventCSRLinear_brainpylib, EventCSRLinear_taichi),
    # 'JitFPHomoLinear': (bp.dnn.JitFPHomoLinear, bp.dnn.JitFPHomoLinear_taichi),
    # 'JitFPUniformLinear': (bp.dnn.JitFPUniformLinear, bp.dnn.JitFPUniformLinear_taichi),
    # 'JitFPNormalLinear': (bp.dnn.JitFPNormalLinear, bp.dnn.JitFPNormalLinear_taichi),
    'EventJitFPHomoLinear': (EventJitFPHomoLinear_brainpylib, EventJitFPHomoLinear_taichi),
    'EventJitFPUniformLinear': (EventJitFPUniformLinear_brainpylib, EventJitFPUniformLinear_taichi),
    'EventJitFPNormalLinear': (EventJitFPNormalLinear_brainpylib, EventJitFPNormalLinear_taichi)
    }
E_PROB = 0.75
I_PROB = 0.25
SEED = 1234
E_WEIGHT = 0.6
E_W_LOW = 0.3
E_W_HIGH = 0.9
E_W_MU = 0.6
E_W_SIGMA = 0.01
I_WEIGHT = 6.7
I_W_LOW = 6.4
I_W_HIGH = 7.0
I_W_MU = 6.7
I_W_SIGMA = 0.01

RUNNING_TIME = 1000
# if bm.get_platform() == 'cpu':
#     RUNNING_TIME = 10


class EINet(bp.DynSysGroup):
    def __init__(self, post_num, conn_num, comm_type, comm_class):
        super().__init__()
        self.exc_num = int(post_num * E_PROB)
        self.inh_num = int(post_num * I_PROB)
        self.N = bp.dyn.LifRef(post_num, V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.,
                           V_initializer=bp.init.Normal(-55., 2.))
        self.delay = bp.VarDelay(self.N.spike, entries={'I': None})
        if comm_type == 'CSRLinear' or comm_type == 'EventCSRLinear':
            self.E = bp.dyn.HalfProjAlignPostMg(comm=comm_class(bp.conn.FixedProb(conn_num/post_num, pre = self.exc_num, post = post_num, seed=SEED, allow_multi_conn=True), weight=E_WEIGHT),
                                        syn=bp.dyn.Expon.desc(size=post_num, tau=5.),
                                        out=bp.dyn.COBA.desc(E=0.),
                                        post=self.N)
            self.I = bp.dyn.HalfProjAlignPostMg(comm=comm_class(bp.conn.FixedProb(conn_num/post_num, pre = self.inh_num, post = post_num, seed=SEED, allow_multi_conn=True), weight=I_WEIGHT),
                                        syn=bp.dyn.Expon.desc(size=post_num, tau=5.),
                                        out=bp.dyn.COBA.desc(E=-80.),
                                        post=self.N)
        elif comm_type == 'JitFPHomoLinear' or comm_type == 'EventJitFPHomoLinear':
            self.E = bp.dyn.HalfProjAlignPostMg(comm=comm_class(self.exc_num, post_num, prob=conn_num/post_num, weight=E_WEIGHT, seed=SEED, atomic=True),
                                        syn=bp.dyn.Expon.desc(size=post_num, tau=5.),
                                        out=bp.dyn.COBA.desc(E=0.),
                                        post=self.N)
            self.I = bp.dyn.HalfProjAlignPostMg(comm=comm_class(self.inh_num, post_num, prob=conn_num/post_num, weight=I_WEIGHT, seed=SEED, atomic=True),
                                        syn=bp.dyn.Expon.desc(size=post_num, tau=5.),
                                        out=bp.dyn.COBA.desc(E=-80.),
                                        post=self.N)
        elif comm_type == 'JitFPUniformLinear' or comm_type == 'EventJitFPUniformLinear':
            self.E = bp.dyn.HalfProjAlignPostMg(comm=comm_class(self.exc_num, post_num, prob=conn_num/post_num, w_low=E_W_LOW, w_high=E_W_HIGH, seed=SEED, atomic=True),
                                        syn=bp.dyn.Expon.desc(size=post_num, tau=5.),
                                        out=bp.dyn.COBA.desc(E=0.),
                                        post=self.N)
            self.I = bp.dyn.HalfProjAlignPostMg(comm=comm_class(self.inh_num, post_num, prob=conn_num/post_num, w_low=I_W_LOW, w_high=I_W_HIGH, seed=SEED, atomic=True),
                                        syn=bp.dyn.Expon.desc(size=post_num, tau=5.),
                                        out=bp.dyn.COBA.desc(E=-80.),
                                        post=self.N)
        elif comm_type == 'JitFPNormalLinear' or comm_type == 'EventJitFPNormalLinear':
            self.E = bp.dyn.HalfProjAlignPostMg(comm=comm_class(self.exc_num, post_num, prob=conn_num/post_num, w_mu=E_W_MU, w_sigma=E_W_SIGMA, seed=SEED, atomic=True),
                                        syn=bp.dyn.Expon.desc(size=post_num, tau=5.),
                                        out=bp.dyn.COBA.desc(E=0.),
                                        post=self.N)
            self.I = bp.dyn.HalfProjAlignPostMg(comm=comm_class(self.inh_num, post_num, prob=conn_num/post_num, w_mu=I_W_MU, w_sigma=I_W_SIGMA, seed=SEED, atomic=True),
                                        syn=bp.dyn.Expon.desc(size=post_num, tau=5.),
                                        out=bp.dyn.COBA.desc(E=-80.),
                                        post=self.N)
        else:
            raise NotImplementedError
        
    def update(self, input):
        spk = self.delay.at('I')
        self.E(spk[:self.exc_num])
        self.I(spk[self.exc_num:])
        self.delay(self.N(input))
        return self.N.spike.value
    
def test_spks(post_num, conn_num, comm_type):
    print(f'Post num: {post_num}, conn num: {conn_num}, comm type: {comm_type}')
    comm_brainpylib = comm_types[comm_type][0]
    comm_taichi = comm_types[comm_type][1]
    net_brainpylib = EINet(post_num, conn_num, comm_type, comm_brainpylib)
    net_taichi = EINet(post_num, conn_num, comm_type, comm_taichi)

    indices = bm.arange(RUNNING_TIME)

    brainpylib_spks = bm.for_loop(lambda i: net_brainpylib.step_run(i, 20.), indices)
    taichi_spks = bm.for_loop(lambda i: net_taichi.step_run(i, 20.), indices)
    
    brainpylib_spk = brainpylib_spks.sum() / (RUNNING_TIME/1000) / post_num
    taichi_spk = taichi_spks.sum() / (RUNNING_TIME/1000) / post_num
    

    print(f'BrainPyLib spks: {brainpylib_spk}')
    print(f'Taichi spks: {taichi_spk}')
    
    return brainpylib_spk, taichi_spk

df = pd.DataFrame(columns=['post_num', 'conn_num', 'comm_type', 'device',
                           'brainpylib_spk',
                           'taichi_spk',])
PATH = os.path.dirname(os.path.abspath(__file__))
if bm.get_platform() == 'cpu':
    for post_num in post_nums:
        for conn_num in conn_nums:
            for comm_type in comm_types:
                brainpylib_spks, taichi_spks = test_spks(post_num, conn_num, comm_type)
                df.loc[df.shape[0]] = [post_num, conn_num, comm_type, 'cpu',
                                       brainpylib_spks,
                                       taichi_spks]
                df.to_csv(f'{PATH}/benchmark_spks_cpu.csv', index=False)

if bm.get_platform() == 'gpu':
    for post_num in post_nums:
        for conn_num in conn_nums:
            for comm_type in comm_types:
               brainpylib_spks, taichi_spks = test_spks(post_num, conn_num, comm_type)
               df.loc[df.shape[0]] = [post_num, conn_num, comm_type, 'gpu',
                                      brainpylib_spks,
                                      taichi_spks]
               df.to_csv(f'{PATH}/benchmark_spks_gpu.csv', index=False)