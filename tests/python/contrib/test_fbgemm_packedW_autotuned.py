import tvm
import numpy as np
from tvm.contrib import fbgemm
from collections import namedtuple
from tvm import autotvm
import sys
import logging
import os

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

def test_fbgemm_packed_weights_with_requant(m, n, k, w_val, x_val, b_val):
    W_trans = False
    ctx = tvm.cpu(0)
    W = tvm.placeholder((k, n), name='W', dtype="uint8")
    w = tvm.nd.array(np.random.uniform(w_val - 1, w_val + 2, size=(k, n)).astype(W.dtype), ctx)
    my_packedw = tvm.get_global_func("tvm.contrib.fbgemm.pack_matrixB_int8")
    ww = my_packedw(w, 1, W_trans)

    get_co_offsets = tvm.get_global_func("tvm.contrib.fbgemm.compute_col_offsets_int8")
    co = get_co_offsets(w,1,1, W_trans)

    X = tvm.placeholder((m, k), name='X', dtype="int8")
    B = tvm.placeholder((n,), name='B', dtype="int")

    # quantization parameters will be got from Operator arguments
    X_qparams = QuantParams(scale=1.0, zero_point=0)
    W_qparams = QuantParams(scale=1.0, zero_point=0)
    Y_qparams = QuantParams(scale=1.0, zero_point=0)

    C = fbgemm.gemm_int8acc32_prepacked_with_requant(m, n, X, X_qparams, ww, W_qparams, B, Y_qparams, co)
    #Y = tvm.compute((m, n), lambda i, j: C[i][j], name="Y")
    #s = tvm.create_schedule(Y.op)
    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [X, B, C], target="llvm", name="packedmatmul_with_requant")
    #print(tvm.lower(s, [X, B, C], simple_mode=True))
    f_evaluator = f.time_evaluator(f.entry_name, ctx, 10)

    x = tvm.nd.array(np.random.uniform(x_val - 1, x_val + 2, size=(m, k)).astype(X.dtype), ctx)
    b = tvm.nd.array(np.random.uniform(b_val - 1, b_val + 2, size=(n,)).astype(B.dtype), ctx)
    y = tvm.nd.array(np.zeros((m, n), dtype=C.dtype), ctx)
    f(x,b,y)

    #result = f_evaluator(x,b,y)
    #print(result)
    #gops_per_mm = 2*m*n*k
    #gops_per_sec = gops_per_mm/result.mean/1e9
    print("M:{}, N:{}, K:{}".format(m,n,k))
    #print(gops_per_sec)
    print(y.asnumpy())
    print(np.matmul(x.asnumpy(), w.asnumpy()) + b.asnumpy())

    tvm.testing.assert_allclose(
           y.asnumpy(), np.matmul(x.asnumpy(), w.asnumpy()) + b.asnumpy(), rtol=1e-5)

def test_fbgemm_conv_int8(shape):
    if (conv_p.IC % conv_p.G != 0 || conv_p.OC % conv_p.G != 0) {
      // invalid shapes
      continue;
    }
    int im_in_dim = accumulate(
        conv_p.IN_DIM.begin(), conv_p.IN_DIM.end(), 1, multiplies<int>());
    aligned_vector<uint8_t> Aint8(conv_p.MB * im_in_dim * conv_p.IC);

    int kernel_dim =
        accumulate(conv_p.K.begin(), conv_p.K.end(), 1, multiplies<int>());
    aligned_vector<int8_t> Bint8(
        kernel_dim * conv_p.IC * (conv_p.OC / conv_p.G));

    int im_out_dim = accumulate(
        conv_p.OUT_DIM.begin(), conv_p.OUT_DIM.end(), 1, multiplies<int>());
    aligned_vector<int32_t> Cint32_ref(conv_p.MB * im_out_dim * conv_p.OC);
    aligned_vector<uint8_t> Cint8_ref(Cint32_ref.size(), 0);
    aligned_vector<int32_t> Cint32_fb(Cint32_ref.size());
    aligned_vector<uint8_t> Cint8_fb(Cint32_ref.size(), 0);
    aligned_vector<uint8_t> Cint8_fb2(Cint32_ref.size(), 0);
    aligned_vector<int32_t> Cint32_fb2(Cint32_ref.size());

    // A matrix (input activations)
    randFill<uint8_t>(Aint8, 0, 5);
    int32_t Aint8_zero_point = 4;

    // B matrix (weights)
    randFill<int8_t>(Bint8, -4, 4);
    aligned_vector<int32_t> Bint8_zero_point(1);
    randFill(Bint8_zero_point, -3, -1);

    aligned_vector<float> C_multiplier(Bint8_zero_point.size());
    randFill(C_multiplier, 0.1234f / 2, 0.1234f * 3 / 2);
    int32_t C_zero_point = 5;

    aligned_vector<float> Bfp32(Bint8.begin(), Bint8.end());








    PackWeightsForConv<SPATIAL_DIM> packedB(conv_p, Bint8.data());

    // no-op output process objects
    DoNothing<> doNothingObj{};
    ReQuantizeOutput<false, QuantizationGranularity::TENSOR> outputProcObj(
        doNothingObj,
        C_multiplier.data(),
        C_zero_point,
        Aint8_zero_point,
        Bint8_zero_point.data(),
        nullptr, // row offsets
        col_offsets.data(),
        nullptr, // bias
        conv_p.OC,
        conv_p.G);

    fbgemmConv(
      conv_p,
      Aint8.data(),
      packedB,
      Cint8_fb.data(),
      Cint32_fb.data(),
      outputProcObj,
      0,
      1);


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
			print(c)
			comb.append(c)

         for c in comb:
	 	test_fbgemm_packed_weights_with_requant(c[0], c[1], c[2], c[3], c[4], c[5])
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
