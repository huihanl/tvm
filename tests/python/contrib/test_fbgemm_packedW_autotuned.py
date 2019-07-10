import tvm
import numpy as np
from tvm.contrib import fbgemm
from collections import namedtuple
from tvm import autotvm
import sys
import logging
import os
import random
from conv_with_requant_ref import *

QuantParams = namedtuple("QuantParams", "scale zero_point")

@autotvm.template
def tune_fbgemm_packed_weights(m, n, k):

    MCBs = [56]
    NCBs = [32]
    KCBs = [256]
    MRs = [14]
    NRs = [32]
    NR_MINs = [16]

    ROW_INTERLEAVE = 4

    MCBs = [48, 98, 144, 192, 240]
    NCBs = [16, 32, 64, 128, 48, 98, 192, 384]
    KCBs = [256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 960, 1024]
    MRs = [24, 12, 6, 3, 8, 4, 2, 1]
    NRs = [16, 32]
    NR_MINs = [16]

    configs = autotvm.get_config()
    configs.define_knob("MCBs", MCBs)
    configs.define_knob("NCBs", NCBs)
    configs.define_knob("KCBs", KCBs)
    configs.define_knob("MRs", MRs)
    configs.define_knob("NRs", NRs)
    configs.define_knob("NR_MINs", NR_MINs)
    configs.add_flop(2 * m * n * k)

    ctx = tvm.cpu(0)
    W = tvm.placeholder((k, n), name='W', dtype="uint8")
    w = tvm.nd.array(np.random.uniform(1, 1, size=(k, n)).astype(W.dtype), ctx)

    my_packedw = tvm.get_global_func("tvm.contrib.fbgemm.pack_matrixB_int8")
    ww = my_packedw(w, 1,
                    configs["MCBs"].val,
                    configs["NCBs"].val,
                    configs["KCBs"].val,
                    configs["MRs"].val,
                    configs["NRs"].val,
                    configs["NR_MINs"].val,
		    ROW_INTERLEAVE)

    get_co_offsets = tvm.get_global_func(
        "tvm.contrib.fbgemm.compute_col_offsets_int8")
    co = get_co_offsets(w, 1, 1)

    X = tvm.placeholder((m, k), name='X', dtype="int8")
    B = tvm.placeholder((n,), name='B', dtype="int")

    # quantization parameters will be got from Operator arguments
    X_qparams = QuantParams(scale=1.0, zero_point=0)
    W_qparams = QuantParams(scale=1.0, zero_point=0)
    Y_qparams = QuantParams(scale=1.0, zero_point=0)

    C = fbgemm.gemm_int8acc32_prepacked_for_tuning(m, n, W,
                                        X, X_qparams,
                                        ww, W_qparams,
                                        B, Y_qparams, co, 1, True,
                                        configs["MCBs"].val,
                                        configs["NCBs"].val,
                                        configs["KCBs"].val,
                                        configs["MRs"].val,
                                        configs["NRs"].val,
                                        configs["NR_MINs"].val,
                                        ROW_INTERLEAVE)

    s = tvm.create_schedule(C.op)
    #f = tvm.build(s, [X,W, B, C], target="llvm", name="packedmatmul")
    return s, [X, W, B, C]

def fbgemm_packed_weights(m, n, k):

    MCB = 56
    NCB = 32
    KCB = 256
    MR = 14
    NR = 32
    NR_MIN = 16
    ROW_INTERLEAVE = 4

    MCB = 48
    NCB = 16
    KCB = 640
    MR = 24
    NR = 16
    NR_MIN = 16
    ROW_INTERLEAVE = 4


    ctx = tvm.cpu(0)
    W = tvm.placeholder((k, n), name='W', dtype="uint8")
    w = tvm.nd.array(np.random.uniform(3, 3, size=(k, n)).astype(W.dtype), ctx)

    my_packedw = tvm.get_global_func("tvm.contrib.fbgemm.pack_matrixB_int8")
    ww = my_packedw(w, 1,
                    MCB,
                    NCB,
                    KCB,
                    MR,
                    NR,
                    NR_MIN,
		    ROW_INTERLEAVE)

    print_packed_b =tvm.get_global_func("tvm.contrib.fbgemm.print_packb")
    #print_packed_b(ww)

    get_co_offsets = tvm.get_global_func(
        "tvm.contrib.fbgemm.compute_col_offsets_int8")
    co = get_co_offsets(w, 1, 1)
    print_co_offsets = tvm.get_global_func("tvm.contrib.fbgemm.print_col_offsets")

    X = tvm.placeholder((m, k), name='X', dtype="int8")
    B = tvm.placeholder((n,), name='B', dtype="int")

    # quantization parameters will be got from Operator arguments
    X_qparams = QuantParams(scale=1.0, zero_point=0)
    W_qparams = QuantParams(scale=1.0, zero_point=0)
    Y_qparams = QuantParams(scale=1.0, zero_point=0)

    C = fbgemm.gemm_int8acc32_prepacked(m, n,
                                        X, X_qparams,
                                        ww, W_qparams,
                                        B, Y_qparams, co, 1, True,
                                        MCB,
                                        NCB,
                                        KCB,
                                        MR,
                                        NR,
                                        NR_MIN,
		                        ROW_INTERLEAVE)

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [X, B, C], target="llvm", name="packedmatmul")

    x = tvm.nd.array(np.random.uniform(2, 2, size=(m, k)).astype(X.dtype), ctx)
    b = tvm.nd.array(np.random.uniform(0, 0, size=(n,)).astype(B.dtype), ctx)
    y = tvm.nd.array(np.zeros((m, n), dtype=C.dtype), ctx)
    f(x,b,y)
    '''
    f_evaluator = f.time_evaluator(f.entry_name, ctx, 10)
    result = f_evaluator(x,b,y)
    print(result)
    gops_per_mm = 2*m*n*k
    gops_per_sec = gops_per_mm/result.mean/1e9
    print("M:{}, N:{}, K:{}".format(m,n,k))
    print(gops_per_sec)
    '''

    tvm.testing.assert_allclose(
           y.asnumpy(), np.matmul(x.asnumpy(), w.asnumpy()) + b.asnumpy(), rtol=1e-5)

def test_fbgemm_packed_weights_with_requant(m, n, k, w_val, x_val, b_val, A_trans, W_trans):
    ctx = tvm.cpu(0)
    W = tvm.placeholder((k, n), name='W', dtype="uint8")
    w1 = np.random.uniform(w_val - 1, w_val + 2, size=(k, n)).astype(W.dtype)
    if W_trans:
        w = tvm.nd.array(w1.transpose(), ctx)
    else:
        w = tvm.nd.array(w1, ctx)
    my_packedw = tvm.get_global_func("tvm.contrib.fbgemm.pack_matrixB_int8")
    ww = my_packedw(w, 1, W_trans)

    get_co_offsets = tvm.get_global_func("tvm.contrib.fbgemm.compute_col_offsets_int8")
    co = get_co_offsets(w, 1, 1, W_trans)

    if A_trans:
        X = tvm.placeholder((k, m), name='X', dtype="int8")
    else:
        X = tvm.placeholder((m, k), name='X', dtype="int8")
    B = tvm.placeholder((n,), name='B', dtype="int")

    # quantization parameters will be got from Operator arguments
    X_qparams = QuantParams(scale=1.0, zero_point=0)
    W_qparams = QuantParams(scale=1.0, zero_point=0)
    Y_qparams = QuantParams(scale=1.0, zero_point=0)

    C = fbgemm.gemm_int8acc32_prepacked_with_requant(m, n, X, X_qparams, ww, W_qparams,
						     B, Y_qparams, co, A_trans)
    #Y = tvm.compute((m, n), lambda i, j: C[i][j], name="Y")
    #s = tvm.create_schedule(Y.op)
    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [X, B, C], target="llvm", name="packedmatmul_with_requant")
    #print(tvm.lower(s, [X, B, C], simple_mode=True))
    f_evaluator = f.time_evaluator(f.entry_name, ctx, 10)
    x1 = np.random.uniform(x_val - 1, x_val + 2, size=(m, k)).astype(X.dtype)
    if A_trans:
        x = tvm.nd.array(x1.transpose(), ctx)
    else:
        x = tvm.nd.array(x1, ctx)
    b = tvm.nd.array(np.random.uniform(b_val, b_val + 2, size=(n,)).astype(B.dtype), ctx)
    y = tvm.nd.array(np.zeros((m, n), dtype=C.dtype), ctx)
    f(x,b,y)

    #result = f_evaluator(x,b,y)
    #print(result)
    #gops_per_mm = 2*m*n*k
    #gops_per_sec = gops_per_mm/result.mean/1e9
    print("M:{}, N:{}, K:{}".format(m,n,k))
    #print(gops_per_sec)
    #print(y.asnumpy())
    #print(np.matmul(x1, w1) + b.asnumpy())

    tvm.testing.assert_allclose(
           y.asnumpy(), np.matmul(x1, w1) + b.asnumpy(), rtol=1e-5)


@autotvm.template
def test_fbgemm_conv_int8_autotuned(MB, IC, OC, IN_DIM_lst, G, K_lst, stride_lst, pad_lst):

    ROW_INTERLEAVE = 4

    MCBs = [48, 98, 144, 192, 240]
    NCBs = [16, 32, 64, 128, 48, 98, 192, 384]
    KCBs = [256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 960, 1024]
    MRs = [24, 12, 6, 3, 8, 4, 2, 1]
    NRs = [16, 32]
    NR_MINs = [16]

    configs = autotvm.get_config()
    configs.define_knob("MCBs", MCBs)
    configs.define_knob("NCBs", NCBs)
    configs.define_knob("KCBs", KCBs)
    configs.define_knob("MRs", MRs)
    configs.define_knob("NRs", NRs)
    configs.define_knob("NR_MINs", NR_MINs)

    ctx = tvm.cpu(0)
    spatial_dim = 2
    IN_DIM = tvm.nd.array(np.array(IN_DIM_lst).astype("int32"), ctx)
    K = tvm.nd.array(np.array(K_lst).astype("int32"), ctx)
    stride = tvm.nd.array(np.array(stride_lst).astype("int32"), ctx)
    pad = tvm.nd.array(np.array(pad_lst).astype("int32"), ctx)

    IN_DIMP = [0, 0]
    OUT_DIM = [0, 0]

    IN_DIMP[0] = IN_DIM_lst[0] + pad_lst[0] + pad_lst[2];
    OUT_DIM[0] = (IN_DIMP[0] - K_lst[0]) / stride_lst[0] + 1;

    IN_DIMP[1] = IN_DIM_lst[1] + pad_lst[1] + pad_lst[3];
    OUT_DIM[1] = (IN_DIMP[1] - K_lst[1]) / stride_lst[1] + 1;

    MDim = MB * OUT_DIM[0] * OUT_DIM[1];
    NDim = OC / G;
    KDim = K_lst[0] * K_lst[1] * IC;
    
    nops = 2 * MDim * NDim * KDim
    configs.add_flop(nops)

    # shapes
    input_shape = (MB, IN_DIM_lst[0], IN_DIM_lst[1], IC) #NHWC
    W_shape = (K_lst[0], K_lst[1], IC, OC / G) #RSCK
    Y_shape = (MB, OUT_DIM[0], OUT_DIM[1], OC) #NHWK
    # weight
    W = tvm.placeholder(W_shape, name='W', dtype="int8")
    wa_length = K_lst[0] * K_lst[1] * IC * OC / G
    wa = [random.randint(-4, 4) for i in range(wa_length)]
    w = tvm.nd.array(np.reshape(np.array(wa), W_shape).astype(W.dtype), ctx)

    # packing of weight
    my_packedw = tvm.get_global_func("tvm.contrib.fbgemm.pack_matrixB_int8_conv")

    ww = my_packedw(w, spatial_dim, MB, IC, OC, IN_DIM, G, K, stride, pad)

    # input (X)
    X = tvm.placeholder(input_shape, name='X', dtype="uint8")

    # quantization parameters will be got from Operator arguments
    X_zero_point = 4
    W_zero_point = -2
    Y_zero_point = 5

    # column offset
    get_co_offsets = \
    tvm.get_global_func("tvm.contrib.fbgemm.compute_col_offsets_int8_conv")
    co = get_co_offsets(w, W_zero_point, spatial_dim,
                        MB, IC, OC, IN_DIM, G, K, stride, pad)

    C_multiplier = 0.0878014

    IN_DIM0, IN_DIM1 = IN_DIM_lst
    K0, K1 = K_lst
    stride0, stride1 = stride_lst
    pad0, pad1, pad2, pad3 = pad_lst

    C = fbgemm.conv_int8(Y_shape, X, X_zero_point, ww, W,
                         W_zero_point, Y_zero_point, C_multiplier, co,
                         MB, IC, OC,
                         IN_DIM0, IN_DIM1, G, K0, K1, stride0, stride1,
                         pad0, pad1, pad2, pad3,
                         1, True,
                         configs["MCBs"].val,
                         configs["NCBs"].val,
                         configs["KCBs"].val,
                         configs["MRs"].val,
                         configs["NRs"].val,
                         configs["NR_MINs"].val,
                         ROW_INTERLEAVE)
			
    s = tvm.create_schedule(C.op)
    return s, [X, W, C]

if __name__ == "__main__":
    shapes = (
        [64, 800, 320],
        [64, 768, 512],
        #[16, 256, 512],
        [128, 128, 128],
        [256, 512, 256],
        [1024, 1024, 1024])

    shapes_others = (
        [156800,    4,    36],
        [156800,    8,    36],
        [156800,    16,    36],
        [1,    128,    512],
        [1,    1024,    256],
        [1,    2048,   512],
        [1,    4096,    1024],
        [6,    256,    1024],
        [6,    256,    2048],
        [6,    512,    512],
        [6,    1024,    256],
        [6,    2048,    256],
        [6,    2048,    512],
        [6,    4096,    256],
        [6,    4096,    1024],
        [6,    4096,    2048],
        [10,    2048,    256],
        [10,    4096,    1024],
        [20,    2048,    256],
        [20,    4096,    1024],
        [102,    1024,    512],
        [102,    2323,    256],
        [102,    512,    256],
        [1,    800,    3200],
        [1,    800,    8000],
        [16,    256,    1500],
        [16,    256,    1567],
        [1,    128,    2876],
        [16,    128,    1567],
        [1,    128,    2722])

    configs = [
		[1, 32, 32, [3, 3], 8, [3, 3], [1, 1], [1, 1, 1, 1]],
        [1, 32, 32, [4, 4], 8, [3, 3], [1, 1], [1, 1, 1, 1]],
        [1, 32, 32, [3, 5], 8, [3, 3], [1, 1], [1, 1, 1, 1]],
        [1, 32, 32, [5, 3], 8, [3, 3], [1, 1], [1, 1, 1, 1]],
        [1, 8, 8, [5, 5], 2, [3, 3], [1, 1], [1, 1, 1, 1]],
        [1, 128, 128, [56, 48], 32, [3, 3], [1, 1], [1, 1, 1, 1]],
        [1, 128, 128, [48, 56], 32, [3, 3], [1, 1], [1, 1, 1, 1]],
        [1, 64, 64, [3, 3], 8, [3, 3], [1, 1], [1, 1, 1, 1]],
        [1, 64, 64, [4, 4], 8, [3, 3], [1, 1], [1, 1, 1, 1]],
        [1, 64, 64, [3, 5], 8, [3, 3], [1, 1], [1, 1, 1, 1]],
        [1, 64, 64, [5, 3], 8, [3, 3], [1, 1], [1, 1, 1, 1]],
        [1, 16, 16, [5, 5], 2, [3, 3], [1, 1], [1, 1, 1, 1]],
        [1, 256, 256, [56, 48], 32, [3, 3], [1, 1], [1, 1, 1, 1]],
        [1, 256, 256, [48, 56], 32, [3, 3], [1, 1], [1, 1, 1, 1]],
        [1, 256, 256, [56, 56], 32, [3, 3], [1, 1], [1, 1, 1, 1]],
        [2, 256, 256, [56, 56], 32, [3, 3], [1, 1], [1, 1, 1, 1]]]

   
    if False:

        for i in range(4, 5):
            config = configs[i]
            test_fbgemm_conv_int8(config[0], config[1], config[2], config[3],
                                  config[4], config[5], config[6], config[7])

    else:
         for config in configs:
              task = autotvm.task.create(
                      test_fbgemm_conv_int8_autotuned,
                      args=(config[0], config[1], config[2], config[3],
                      config[4], config[5], config[6], config[7]), target='llvm')
              #print(task.config_space)
              print(len(task.config_space))
              # logging config (for printing tuning log to the screen)
              logging.getLogger('autotvm').setLevel(logging.DEBUG)
              logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))
              measure_option = autotvm.measure_option(
                  builder='local',
                  runner=autotvm.LocalRunner(number=10, timeout=100000))
              tuner = autotvm.tuner.RandomTuner(task)
              name = str([config[0], config[1], config[2], config[3], config[4], config[5]])
              log_file_name = "fbgemm_results_" + name + ".log"
              tuner.tune(n_trial=150,
                         measure_option=measure_option,
                         callbacks=[autotvm.callback.log_to_file(log_file_name)])  
