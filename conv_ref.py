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
                    h_in = -conv_p.pad[0] + h * conv_p.stride[0] + r
                    for s in range(K[1]):
                        w_in = -conv_p.pad[1] + w * conv_p.stride[1] + s
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
                           ncols_per_quant_group, NDim, NDim_OC):
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


def row_offsets_u8acc32_ref(M, K, ld, Aint8, length, KDimPerGroup):
    row_offsets = [0 in range(length)]
    for g in range(G):
        for i in range(M):
            sum = 0
            for k in range(K):
                sum += Aint8[KDimPerGroup * g + i * ld + k]
            row_offsets[i] = sum
    return row_offsets


def col_offsets_with_zero_pt_s8acc32_ref(K, N, ld, Bint8, B_zero_point, ncols_per_quant_group, G, col_lead, w_lead):
    col_offsets = [0 in range(OC)]
    for g in range(G):
        for j in range(N):
            sum = 0
            for k in range(K):
                sum += Bint8[g * w_lead + k * ld + j]
            col_offsets[g * col_lead + j] = sum - B_zero_point[j / ncols_per_quant_group] * K

    return col_offsets



def reference_solution(A, A_zero_point, W, MB, IC, OC, IN_DIM, OUT_DIM, G, K, stride, pad, C_multiplier):
    Cint32_ref = conv_ref(MB, IC, OC, IN_DIM, OUT_DIM, G, K, stride, pad,
                 A, A_zero_point, W)

    MDim = MB * im_out_dim
    NDim = OC / G
    KDim = kernel_dim * IC
    KDimPerGroup = KDim / G

    OC_per_G = OC / G

    length_im2col = MDim * KDim
    A_im2col = im2col_ref(MB, IC, OC, IN_DIM, OUT_DIM, G, K, stride, pad, A, A_zero_point, length_im2col)

    col_lead = OC_per_G
    w_lead = KDimPerGroup * OC_per_G
    col_offsets = col_offsets_with_zero_pt_s8acc32_ref(KDimPerGroup, OC_per_G, OC_per_G,
                                                       W, B_zero_point, ncols_per_quant_group, col_lead, w_lead)


    row_offsets = row_offsets_u8acc32_ref(MDim, KDimPerGroup, KDim, A_im2col, MDim, KDimPerGroup)

    output = requantize_u8acc32_ref(MDim, NDim, G * NDim, Cint32_ref, C_multiplier, C_zero_point,
                                    A_zero_point, B_zero_point, row_offsets, col_offsets,
                                    OC, NDim, NDim_OC)

    return output
