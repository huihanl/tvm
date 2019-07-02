import tvm
import numpy as np
from tvm.contrib import fbgemm
from collections import namedtuple
from tvm import autotvm
import sys
import logging
import os

#print(os.getpid())
#raw_input("dummy breakpoint")

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

def test_fbgemm_conv_int8():

    ctx = tvm.cpu(0)
      #MB, IC, OC, {IT, IH, IW}, G, {KT, KH, KW}, {stride_t, stride_h, stride_w},
      #{pad_prev, pad_h_top, pad_w_left, pad_next, pad_h_bottom, pad_w_right}


    get_conv_params = tvm.get_global_func("tvm.contrib.fbgemm.get_conv_params")
    conv_params = get_conv_params(MB, IC, OC, IN_DIM, G, K, stride, pad)

    print("finish conv_params")

    spatial_dim = 2

    MB = 1
    IC = 4
    OC = 4
    #IN_DIM = [56, 56]
    IN_DIM = tvm.nd.array(np.array([5, 5]).astype("int32"), ctx)
    G = 1
    K = tvm.nd.array(np.array([3, 3]).astype("int32"), ctx)
    stride = tvm.nd.array(np.array([1, 1]).astype("int32"), ctx)
    #pad = [1, 1, 1, 1]
    pad = tvm.nd.array(np.array([1, 1, 1, 1]).astype("int32"), ctx)
    # conv_params = [1, 128, 128, [56, 56], 1, [3, 3], [1, 1], [1, 1, 1, 1]]

    #conv_params = [MB, IC, OC, IN_DIM, G, K, stride, pad]


    # compute out_dim, i.e. shape for Y (the output for convolution)
    IN_DIMP = [0, 0]
    OUT_DIM = [0, 0]

    IN_DIM1 = [5, 5]
    K1 = [3, 3]
    stride1 = [1, 1]
    pad1 = [1, 1, 1, 1]

    IN_DIMP[0] = IN_DIM1[0] + pad1[0] + pad1[2];
    OUT_DIM[0] = (IN_DIMP[0] - K1[0]) / stride1[0] + 1;

    IN_DIMP[1] = IN_DIM1[1] + pad1[1] + pad1[3];
    OUT_DIM[1] = (IN_DIMP[1] - K1[1]) / stride1[1] + 1;

    # shapes
    input_shape = (MB, IN_DIM1[0], IN_DIM1[1], IC) #NHWC

    W_shape = (K1[0], K1[1], IC, OC / G) #RSCK

    Y_shape = (MB, OUT_DIM[0], OUT_DIM[1], OC) #NHWK

    # weight
    W = tvm.placeholder(W_shape, name='W', dtype="int8")
    w = tvm.nd.array(np.random.uniform(1, 10, size=W_shape).astype(W.dtype), ctx)

    # packing of weight
    my_packedw = tvm.get_global_func("tvm.contrib.fbgemm.pack_matrixB_int8_conv")

    ww = my_packedw(w, spatial_dim, MB, IC, OC, IN_DIM, G, K, stride, pad)

    # input (X)
    X = tvm.placeholder(input_shape, name='X', dtype="uint8")

    # quantization parameters will be got from Operator arguments
    X_zero_point = 4
    W_zero_point = -1
    Y_zero_point = 5

    # column offset
    get_co_offsets = tvm.get_global_func("tvm.contrib.fbgemm.compute_col_offsets_int8_conv")
    co = get_co_offsets(w, W_zero_point, spatial_dim, MB, IC, OC, IN_DIM, G, K, stride, pad)

    # ReQuant Multiplier
    C_multiplier = 0.0878014

    in_dim_v = create_pointer_vector_int(IN_DIM, 2)

    k_v = create_pointer_vector_int(K, 2)

    stride_v = create_pointer_vector_int(stride, 2)

    pad_v = create_pointer_vector_int(pad, 4)

    C = fbgemm.conv_int8(Y_shape, X, X_zero_point, ww, W_zero_point, Y_zero_point, C_multiplier, co,
			 MB, IC, OC, in_dim_v, G, k_v, stride_v, pad_v)

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [X, C], target="llvm", name="conv_int8")

    # applying the formula
    x = tvm.nd.array(np.random.uniform(1, 3, size=input_shape).astype(X.dtype), ctx)
    #x = tvm.nd.array(np.array([0,0,4,2,3,1,0,4,4,5,2,3,4,0,0,3,4,0,2,0,2,4,3,5,5,3,0,3,2,4,5,4,1,0,4,1,3,4,5,2,1,5,4,4,3,0,3,5,1,2,4,2,1,1,2,0,2,5,5,0,5,3,3,1,5,2,1,0,5,0,3,2,1,5,3,2,5,0,4,4,4,0,0,4,5,3,4,4,5,5,1,1,2,3,3,5,2,5,1,2]).astype(X.dtype), ctx)
    #b = tvm.nd.array(np.random.uniform(b_val - 1, b_val + 2, size=(n,)).astype(B.dtype), ctx)
    #y = tvm.nd.array(np.zeros(Y_shape, dtype=C.dtype), ctx)
    y = tvm.nd.array(np.zeros(Y_shape, dtype=C.dtype), ctx)

    f(x,y)

    print(y.asnumpy())

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


    test_fbgemm_conv_int8()
    """
    if True:


	 shapes = (
		[4, 8, 2],
		[2, 16, 1],
		[4, 4, 2],
		[1, 8, 4],
		[16, 1, 1],
		[16, 2, 2],
		[8, 2, 4],
		[2, 2, 8])


	 values = (
		[1.0, 2.0, 0.0],
		[2.0, 2.0, 0.0],
		[3.0, 1.0, 0.0],
		[2.0, 3.0, 0.0],
		[1.0, 3.0, 0.0],
		[2.0, 3.0, 3.0],
		[2.0, 1.0, 2.0])

	 comb = []
	 for shape in shapes:
		for value in values:
			c = shape + value
			comb.append(c)

         for c in comb:
	 	test_fbgemm_packed_weights_with_requant(c[0], c[1], c[2], c[3], c[4], c[5], True, True)
                test_fbgemm_packed_weights_with_requant(c[0], c[1], c[2], c[3], c[4], c[5], True, False)
                test_fbgemm_packed_weights_with_requant(c[0], c[1], c[2], c[3], c[4], c[5], False, True)
                test_fbgemm_packed_weights_with_requant(c[0], c[1], c[2], c[3], c[4], c[5], False, False)
         #fbgemm_packed_weights(16, 4, 8)
         #for shape in shapes_others:
         #     fbgemm_packed_weights(shape[0], shape[1], shape[2])

    else:

         for shape in shapes_others:
              task = autotvm.task.create(
                  tune_fbgemm_packed_weights, args=(
                      shape[0] , shape[1] , shape[2] ), target='llvm')
              #print(task.config_space)
              print(len(task.config_space))
              # logging config (for printing tuning log to the screen)
              logging.getLogger('autotvm').setLevel(logging.DEBUG)
              logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))

              measure_option = autotvm.measure_option(
                  builder='local',
                  runner=autotvm.LocalRunner(number=10, timeout=100000))

              tuner = autotvm.tuner.RandomTuner(task)
              log_file_name = "fbgemm_results_"+str(shape[0])+"_"+str(shape[1])+"_"+str(shape[2])+".log"
              tuner.tune(n_trial=150,
                         measure_option=measure_option,
                         callbacks=[autotvm.callback.log_to_file(log_file_name)])
    """
