import brainpy as bp
import brainpy.math as bm
import pandas as pd
import os
from time import time
from functools import partial

from linear import EventCSRLinear, EventJitFPHomoLinear, EventJitFPUniformLinear, EventJitFPNormalLinear

EventCSRLinear_brainpylib = partial(EventCSRLinear, method='brainpylib')
EventCSRLinear_taichi = partial(EventCSRLinear, method=None)
EventJitFPHomoLinear_brainpylib = partial(EventJitFPHomoLinear, method='brainpylib')
EventJitFPHomoLinear_taichi = partial(EventJitFPHomoLinear, method=None)
EventJitFPUniformLinear_brainpylib = partial(EventJitFPUniformLinear, method='brainpylib')
EventJitFPUniformLinear_taichi = partial(EventJitFPUniformLinear, method=None)
EventJitFPNormalLinear_brainpylib = partial(EventJitFPNormalLinear, method='brainpylib')
EventJitFPNormalLinear_taichi = partial(EventJitFPNormalLinear, method=None)


bm.set_platform('gpu')
bm.disable_gpu_memory_preallocation()

post_nums = [400, 4000, 40000, 400000, 4000000]
conn_nums = [20, 40, 80, 100]
# conn_nums = [80]
comm_types = {
    # 'CSRLinear': (bp.dnn.CSRLinear, bp.dnn.CSRLinear_taichi),
    'EventCSRLinear': (EventCSRLinear_brainpylib, EventCSRLinear_taichi),
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

RUNNING_TIME = 100
if bm.get_platform() == 'cpu':
    RUNNING_TIME = 10


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
            self.E = bp.dyn.HalfProjAlignPostMg(comm=comm_class(self.exc_num, post_num, prob=conn_num/post_num, weight=E_WEIGHT, seed=SEED),
                                        syn=bp.dyn.Expon.desc(size=post_num, tau=5.),
                                        out=bp.dyn.COBA.desc(E=0.),
                                        post=self.N)
            self.I = bp.dyn.HalfProjAlignPostMg(comm=comm_class(self.inh_num, post_num, prob=conn_num/post_num, weight=I_WEIGHT, seed=SEED),
                                        syn=bp.dyn.Expon.desc(size=post_num, tau=5.),
                                        out=bp.dyn.COBA.desc(E=-80.),
                                        post=self.N)
        elif comm_type == 'JitFPUniformLinear' or comm_type == 'EventJitFPUniformLinear':
            self.E = bp.dyn.HalfProjAlignPostMg(comm=comm_class(self.exc_num, post_num, prob=conn_num/post_num, w_low=E_W_LOW, w_high=E_W_HIGH, seed=SEED),
                                        syn=bp.dyn.Expon.desc(size=post_num, tau=5.),
                                        out=bp.dyn.COBA.desc(E=0.),
                                        post=self.N)
            self.I = bp.dyn.HalfProjAlignPostMg(comm=comm_class(self.inh_num, post_num, prob=conn_num/post_num, w_low=I_W_LOW, w_high=I_W_HIGH, seed=SEED),
                                        syn=bp.dyn.Expon.desc(size=post_num, tau=5.),
                                        out=bp.dyn.COBA.desc(E=-80.),
                                        post=self.N)
        elif comm_type == 'JitFPNormalLinear' or comm_type == 'EventJitFPNormalLinear':
            self.E = bp.dyn.HalfProjAlignPostMg(comm=comm_class(self.exc_num, post_num, prob=conn_num/post_num, w_mu=E_W_MU, w_sigma=E_W_SIGMA, seed=SEED),
                                        syn=bp.dyn.Expon.desc(size=post_num, tau=5.),
                                        out=bp.dyn.COBA.desc(E=0.),
                                        post=self.N)
            self.I = bp.dyn.HalfProjAlignPostMg(comm=comm_class(self.inh_num, post_num, prob=conn_num/post_num, w_mu=I_W_MU, w_sigma=I_W_SIGMA, seed=SEED),
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
    
def test(post_num, conn_num, comm_type):
    print(f'Post num: {post_num}, conn num: {conn_num}, comm type: {comm_type}')
    comm_brainpylib = comm_types[comm_type][0]
    comm_taichi = comm_types[comm_type][1]
    net_brainpylib = EINet(post_num, conn_num, comm_type, comm_brainpylib)
    net_taichi = EINet(post_num, conn_num, comm_type, comm_taichi)

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
    
    bm.clear_buffer_memory()

    print(f'BrainPylib: {(time1 - time0)*1000}, {(time3 - time2)*1000}, {(time5 - time4)*1000}, {(time7 - time6)*1000}, {(time9 - time8)*1000}, {(time11 - time10)*1000}, {(time13 - time12)*1000}, {(time15 - time14)*1000}, {(time17 - time16)*1000}, {(time19 - time18)*1000}')
    print(f'Taichi: {(time21 - time20)*1000}, {(time23 - time22)*1000}, {(time25 - time24)*1000}, {(time27 - time26)*1000}, {(time29 - time28)*1000}, {(time31 - time30)*1000}, {(time33 - time32)*1000}, {(time35 - time34)*1000}, {(time37 - time36)*1000}, {(time39 - time38)*1000}')

    return (time1 - time0)*1000, (time3 - time2)*1000, (time5 - time4)*1000, (time7 - time6)*1000, (time9 - time8)*1000, (time11 - time10)*1000, (time13 - time12)*1000, (time15 - time14)*1000, (time17 - time16)*1000, (time19 - time18)*1000, \
        (time21 - time20)*1000, (time23 - time22)*1000, (time25 - time24)*1000, (time27 - time26)*1000, (time29 - time28)*1000, (time31 - time30)*1000, (time33 - time32)*1000, (time35 - time34)*1000, (time37 - time36)*1000, (time39 - time38)*1000

df = pd.DataFrame(columns=['post_num', 'conn_num', 'comm_type', 'device',
                           'brainpylib_time1', 'brainpylib_time2', 'brainpylib_time3', 'brainpylib_time4', 'brainpylib_time5', 'brainpylib_time6', 'brainpylib_time7', 'brainpylib_time8', 'brainpylib_time9', 'brainpylib_time10',
                           'taichi_time1', 'taichi_time2', 'taichi_time3', 'taichi_time4', 'taichi_time5', 'taichi_time6', 'taichi_time7', 'taichi_time8', 'taichi_time9', 'taichi_time10'])
PATH = os.path.dirname(os.path.abspath(__file__))
if bm.get_platform() == 'cpu':
    for post_num in post_nums:
        for conn_num in conn_nums:
            for comm_type in comm_types:
                brainpylib_time1, brainpylib_time2, brainpylib_time3, brainpylib_time4, brainpylib_time5, brainpylib_time6, brainpylib_time7, brainpylib_time8, brainpylib_time9, brainpylib_time10, \
                    taichi_time1, taichi_time2, taichi_time3, taichi_time4, taichi_time5, taichi_time6, taichi_time7, taichi_time8, taichi_time9, taichi_time10 = test(post_num, conn_num, comm_type)
                df.loc[df.shape[0]] = [post_num, conn_num, comm_type, 'cpu',
                                   brainpylib_time1, brainpylib_time2, brainpylib_time3, brainpylib_time4, brainpylib_time5, brainpylib_time6, brainpylib_time7, brainpylib_time8, brainpylib_time9, brainpylib_time10,
                                   taichi_time1, taichi_time2, taichi_time3, taichi_time4, taichi_time5, taichi_time6, taichi_time7, taichi_time8, taichi_time9, taichi_time10]
                df.to_csv(f'{PATH}/benchmark_cpu.csv', index=False)

if bm.get_platform() == 'gpu':
    for post_num in post_nums:
        for conn_num in conn_nums:
            for comm_type in comm_types:
                brainpylib_time1, brainpylib_time2, brainpylib_time3, brainpylib_time4, brainpylib_time5, brainpylib_time6, brainpylib_time7, brainpylib_time8, brainpylib_time9, brainpylib_time10, \
                    taichi_time1, taichi_time2, taichi_time3, taichi_time4, taichi_time5, taichi_time6, taichi_time7, taichi_time8, taichi_time9, taichi_time10 = test(post_num, conn_num, comm_type)
                df.loc[df.shape[0]] = [post_num, conn_num, comm_type, 'gpu',
                                   brainpylib_time1, brainpylib_time2, brainpylib_time3, brainpylib_time4, brainpylib_time5, brainpylib_time6, brainpylib_time7, brainpylib_time8, brainpylib_time9, brainpylib_time10,
                                   taichi_time1, taichi_time2, taichi_time3, taichi_time4, taichi_time5, taichi_time6, taichi_time7, taichi_time8, taichi_time9, taichi_time10]
                df.to_csv(f'{PATH}/benchmark_gpu.csv', index=False)