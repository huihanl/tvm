import tvm
import numpy as np
from tvm.contrib import fbgemm
from collections import namedtuple
from tvm import autotvm
import sys
import logging
import os
import random

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


def conv_ref(MB, IC, OC, IN_DIM, OUT_DIM, G, K, stride, pad,
             A, A_zero_point, B):
    c_ref = [0 for i in range(MB * OUT_DIM[0] * OUT_DIM[1] * OC)]
    for n in range(MB):
        for h in range(OUT_DIM[0]):
            for w in range(OUT_DIM[1]):
                for g in range(G):
                    for m in range(OC / G):
                        sum = 0
                        for r in range(K[0]):
                            h_in = -pad[0] + h * stride[0] + r
                            for s in range(K[1]):
                                w_in = -pad[1] + w * stride[1] + s
                                for c in range(IC / G):
                                    a = 0
                                    if h_in < 0 or h_in >= IN_DIM[0] or w_in < 0 or w_in >= IN_DIM[1]:
                                        a = A_zero_point
                                    else:
                                        a = A[((n * IN_DIM[0] + h_in) * IN_DIM[1] + w_in) * IC + g * (IC / G) + c]
                                    b = B[(((g * K[0] + r) * K[1] + s) * (IC / G) + c) * (OC / G) + m]
                                    sum += a * b;

                        c_ref[((n * OUT_DIM[0] + h) * OUT_DIM[1] + w) * OC + g * (OC / G) + m] = sum;

    return c_ref



def im2col_ref(MB, IC, OC, IN_DIM, OUT_DIM, G, K, stride, pad, A, A_zero_point, length):
    Ao = [0 for i in range(length)]
    for n in range(MB):
        for h in range(OUT_DIM[0]):
            for w in range(OUT_DIM[1]):
                for r in range(K[0]):
                    h_in = -pad[0] + h * stride[0] + r
                    for s in range(K[1]):
                        w_in = -pad[1] + w * stride[1] + s
                        if h_in < 0 or h_in >= IN_DIM[0] or w_in < 0 or w_in >= IN_DIM[1]:
                            for g in range(G):
                                for c_ in range(IC / G):
                                    id = (((((n * OUT_DIM[0] + h) * OUT_DIM[1] + w) * G + g) * K[0] + r) * K[1] + s) * (IC / G)
                                    Ao[id + c_] = A_zero_point

                        else:
                            for g in range(G):
                                for c_ in range(IC / G):
                                    id = (((((n * OUT_DIM[0] + h) * OUT_DIM[1] + w) * G + g) * K[0] + r) * K[1] + s) * (IC / G)
                                    id_src = ((n * IN_DIM[0] + h_in) * IN_DIM[1] + w_in) * IC + g * (IC / G)
                                    Ao[id + c_] = A[id_src + c_]
    return Ao




def requantize_u8acc32_ref(M, N, ld, inp, C_multiplier, C_zero_point,
                           A_zero_point, B_zero_point, row_offsets, col_offsets,
                           ncols_per_quant_group, NDim, NDim_OC, G):
    out = [0 for i in range(len(inp))]

    for g in range(G):
        for i in range(M):
            for j in range(N):
                raw = inp[NDim * g + i * ld + j]
                if A_zero_point:
                    raw -= A_zero_point * col_offsets[j + NDim * g];
                raw -= B_zero_point[j / ncols_per_quant_group + NDim_OC * g] * row_offsets[i]
                result = raw * C_multiplier[NDim_OC * g + j / ncols_per_quant_group]
                rounded = round(result) + C_zero_point
                out[NDim * g + i * ld + j] = max(0, min(255, rounded))

    return out


def row_offsets_u8acc32_ref(M, K, ld, Aint8, length, KDimPerGroup, G):
    row_offsets = [0 for i in range(length)]
    for g in range(G):
        for i in range(M):
            sum = 0
            for k in range(K):
                sum += Aint8[KDimPerGroup * g + i * ld + k]
            row_offsets[i] = sum
    return row_offsets


def col_offsets_with_zero_pt_s8acc32_ref(K, N, ld, OC, Bint8, B_zero_point, ncols_per_quant_group, G, col_lead, w_lead):
    col_offsets = [0 for i in range(OC)]
    for g in range(G):
        for j in range(N):
            total = 0
            for k in range(K):
                total += Bint8[g * w_lead + k * ld + j]
            col_offsets[g * col_lead + j] = total - B_zero_point[j / ncols_per_quant_group] * K

    return col_offsets


def reference_solution(A, A_zero_point, W, MB, IC, OC, IN_DIM, OUT_DIM, G, K, stride, pad, C_multiplier, B_zero_point, C_zero_point):
    Cint32_ref = conv_ref(MB, IC, OC, IN_DIM, OUT_DIM, G, K, stride, pad,
                 A, A_zero_point, W)

    im_in_dim = IN_DIM[0] * IN_DIM[1]
    kernel_dim = K[0] * K[1]
    im_out_dim = OUT_DIM[0] * OUT_DIM[1]

    MDim = MB * im_out_dim
    NDim = OC / G
    KDim = kernel_dim * IC
    KDimPerGroup = KDim / G

    OC_per_G = OC / G

    length_im2col = MDim * KDim
    A_im2col = im2col_ref(MB, IC, OC, IN_DIM, OUT_DIM, G, K, stride, pad, A, A_zero_point, length_im2col)
    print("A_im2col: ")
    print(A_im2col)
    col_lead = OC_per_G
    w_lead = KDimPerGroup * OC_per_G
    col_offsets = col_offsets_with_zero_pt_s8acc32_ref(KDimPerGroup, OC_per_G, OC_per_G, OC,
                                                       W, B_zero_point, OC, G, col_lead, w_lead)
    print("col_offsets: ")
    print(col_offsets)

    row_offsets = row_offsets_u8acc32_ref(MDim, KDimPerGroup, KDim, A_im2col, MDim, KDimPerGroup, G)
    print("row_offsets: ")
    print(row_offsets)

    NDim_OC = NDim / OC
    output = requantize_u8acc32_ref(MDim, NDim, G * NDim, Cint32_ref, C_multiplier, C_zero_point,
                                    A_zero_point, B_zero_point, row_offsets, col_offsets,
                                    OC, NDim, NDim_OC, G)

    return output


def test_fbgemm_conv_int8(xref, wref, yref, MBi, ICi, OCi, IN_DIMi, Gi, Ki, stridei, padi):
    ctx = tvm.cpu(0)

    spatial_dim = 2

    MB = MBi
    IC = ICi
    OC = OCi
    #IN_DIM = [56, 56]
    IN_DIM = tvm.nd.array(np.array(IN_DIMi).astype("int32"), ctx)
    G = Gi
    K = tvm.nd.array(np.array(Ki).astype("int32"), ctx)
    stride = tvm.nd.array(np.array(stridei).astype("int32"), ctx)
    #pad = [1, 1, 1, 1]
    pad = tvm.nd.array(np.array(padi).astype("int32"), ctx)
    # conv_params = [1, 128, 128, [56, 56], 1, [3, 3], [1, 1], [1, 1, 1, 1]]

    conv_params = [MB, IC, OC, IN_DIM, G, K, stride, pad]


    # compute out_dim, i.e. shape for Y (the output for convolution)
    IN_DIMP = [0, 0]
    OUT_DIM = [0, 0]

    IN_DIM1 = IN_DIMi
    K1 = Ki
    stride1 = stridei
    pad1 = padi

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
    wa_length = K1[0] * K1[1] * IC * OC / G
    wa = wref
    w = tvm.nd.array(np.reshape(np.array(wa), W_shape).astype(W.dtype), ctx)
    # packing of weight
    my_packedw = tvm.get_global_func("tvm.contrib.fbgemm.pack_matrixB_int8_conv")

    ww = my_packedw(w, spatial_dim, MB, IC, OC, IN_DIM, G, K, stride, pad)
    # bias
    #B = tvm.placeholder((n,), name='B', dtype="int")

    # input (X)
    X = tvm.placeholder(input_shape, name='X', dtype="uint8")

    # quantization parameters will be got from Operator arguments
    X_zero_point = 4
    W_zero_point = -2 
    create_pointer_vector_int = tvm.get_global_func("tvm.contrib.fbgemm.create_pointer_vector_int")
    #W_zero_point = [1]
    Y_zero_point = 5

    # column offset
    get_co_offsets = tvm.get_global_func("tvm.contrib.fbgemm.compute_col_offsets_int8_conv")
    co = get_co_offsets(w, W_zero_point, spatial_dim, MB, IC, OC, IN_DIM, G, K, stride, pad)
    # ReQuant Multiplier
    #C_multiplier = np.random.uniform(0.1234 / 2, 0.1234 * 3 / 2, size=(1,))
    C_multiplier = 0.0878014
# formula for calculation
    in_dim_v = create_pointer_vector_int(IN_DIM, 2)
    k_v = create_pointer_vector_int(K, 2)
    stride_v = create_pointer_vector_int(stride, 2)
    pad_v = create_pointer_vector_int(pad, 4)

    C = fbgemm.conv_int8(Y_shape, X, X_zero_point, ww, W_zero_point, Y_zero_point, C_multiplier, co,
			 MB, IC, OC, in_dim_v, G, k_v, stride_v, pad_v)
    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [X, C], target="llvm", name="conv_int8")
    # applying the formula
    x_length = MB * IN_DIM1[0] * IN_DIM1[1] * IC
    xa = xref
    x = tvm.nd.array(np.reshape(np.array(xa), input_shape).astype(X.dtype), ctx)
    y = tvm.nd.array(np.zeros(Y_shape, dtype=C.dtype), ctx)
    f(x,y)

    y_ref = reference_solution(xa, X_zero_point, wa, MB, IC, OC, IN_DIM1, OUT_DIM, G, K1, stride1, pad1, [C_multiplier], [W_zero_point], Y_zero_point)
    y_ref = np.reshape(np.array(y_ref), Y_shape)

    #tvm.testing.assert_allclose(y_ref, np.reshape(np.array(yref), Y_shape), rtol=1e-5)
    #print("reference_solution correct")

    #tvm.testing.assert_allclose(y.asnumpy(), np.reshape(np.array(yref), Y_shape), rtol=1e-5)
    #print("implementation correct")

    tvm.testing.assert_allclose(y.asnumpy(), y_ref, rtol=1e-5)


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
		#[1, 4, 4, [5, 5], 1, [3, 3], [1, 1], [1, 1, 1, 1]], 
		#[1, 6, 6, [3, 3], 1, [3, 3], [1, 1], [1, 1, 1, 1]],    
		[1, 32, 32, [3, 3], 8, [3, 3], [1, 1], [1, 1, 1, 1]],
[1, 32, 32, [4, 4], 8, [3, 3], [1, 1], [1, 1, 1, 1]],    [1, 32, 32, [3, 5], 8, [3, 3], [1, 1], [1, 1, 1, 1]],    [1, 32, 32, [5, 3], 8, [3, 3], [1, 1], [1, 1, 1, 1]],    [1, 8, 8, [5, 5], 2, [3, 3], [1, 1], [1, 1, 1, 1]],    [1, 128, 128, [56, 48], 32, [3, 3], [1, 1], [1, 1, 1, 1]],    [1, 128, 128, [48, 56], 32, [3, 3], [1, 1], [1, 1, 1, 1]],     [1, 64, 64, [3, 3], 8, [3, 3], [1, 1], [1, 1, 1, 1]],     [1, 64, 64, [4, 4], 8, [3, 3], [1, 1], [1, 1, 1, 1]],     [1, 64, 64, [3, 5], 8, [3, 3], [1, 1], [1, 1, 1, 1]],     [1, 64, 64, [5, 3], 8, [3, 3], [1, 1], [1, 1, 1, 1]],     [1, 16, 16, [5, 5], 2, [3, 3], [1, 1], [1, 1, 1, 1]],     [1, 256, 256, [56, 48], 32, [3, 3], [1, 1], [1, 1, 1, 1]],     [1, 256, 256, [48, 56], 32, [3, 3], [1, 1], [1, 1, 1, 1]],     [1, 256, 256, [56, 56], 32, [3, 3], [1, 1], [1, 1, 1, 1]],     [2, 256, 256, [56, 56], 32, [3, 3], [1, 1], [1, 1, 1, 1]] ]


    f= open("diff.txt","r")
    valid = f.readlines()
    for i in range(len(valid) / 3):
	config = configs[i]
	print(i)
        xref = valid[i * 3]
        wref = valid[i * 3 + 1]
	yref = valid[i * 3 + 2]
        xref = xref.split()
	xref = [int(x) for x in xref]
        wref = wref.split()
        wref = [int(w) for w in wref]
        yref = yref.split()
        yref = [int(y) for y in yref]
	#print(xref, wref, yref)
        test_fbgemm_conv_int8(xref, wref, yref, config[0], config[1], config[2], config[3], config[4], config[5], config[6], config[7])
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


