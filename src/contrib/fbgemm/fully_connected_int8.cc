/*!
 *  Copyright (c) 2019 by Contributors
 * \file Use external fbgemm library call.
 */

#include <cpuinfo.h>
#include <dmlc/logging.h>
#include <fbgemm/Fbgemm.h>
#include <fbgemm/FbgemmFP16.h>
#include <fbgemm/QuantUtilsAvx2.h>

#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/util.h>
#include <memory>
#include <random>
#include "fbgemm_utils.h"

#include <sys/types.h>
#include <unistd.h>
#include <chrono>

namespace tvm {
namespace runtime {
using namespace fbgemm;

using packbmatrix = PackBMatrix<std::int8_t, std::int32_t>;
template <>
struct extension_class_info<packbmatrix> {
  static const int code = 19;
};

TVM_REGISTER_EXT_TYPE(packbmatrix);
}  // namespace runtime
}  // namespace tvm

namespace tvm {
namespace contrib {

using namespace runtime;
using namespace fbgemm;

TVM_REGISTER_GLOBAL("tvm.contrib.fbgemm.print_packb")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      void* pck_b = args[0];
      packbmatrix* B =
          reinterpret_cast<PackBMatrix<std::int8_t, std::int32_t>*>(pck_b);
      B->printPackedMatrix("B");
    });

TVM_REGISTER_GLOBAL("tvm.contrib.fbgemm.print_col_offsets")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      void* co = args[0];
      std::vector<std::int32_t>* coffsts =
          reinterpret_cast<std::vector<std::int32_t>*>(co);
      std::cout << "size of col offsets" << coffsts->size() << " " << std::endl;

      for(int i=0; i<coffsts->size(); i++) {
        std::cout << coffsts->at(i) << ' ';
      }

    });

TVM_REGISTER_GLOBAL("tvm.contrib.fbgemm.pack_matrixB_int8")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      bool trans = args[2];
      if (!trans) {  //K * N, not transposed
        DLTensor* W = args[0];
        int threads = args[1];

        CHECK_EQ(W->ndim, 2);

        int k = W->shape[0];
        int n = W->shape[1];

        BlockingFactors params;
        if (args.size() > 3) {
          int cntr = 3;
          params.MCB = args[cntr];
          params.NCB = args[cntr + 1];
          params.KCB = args[cntr + 2];
          params.MR = args[cntr + 3];
          params.NR = args[cntr + 4];
          params.NR_MIN = args[cntr + 5];
          params.ROW_INTERLEAVE = args[cntr + 6];

          auto packB = new PackBMatrix<std::int8_t, std::int32_t>(
              matrix_op_t::NoTranspose, k, n,
              reinterpret_cast<const std::int8_t*>(W->data), n, nullptr, 1,
              &params);
         //packB->printPackedMatrix("packingB");
          *ret = packB;
        } else {
          auto packB = new PackBMatrix<std::int8_t, std::int32_t>(
              matrix_op_t::NoTranspose, k, n,
              reinterpret_cast<const std::int8_t*>(W->data), n, nullptr, 1);
          *ret = packB;
        }

      } else { // N*K quantized int8 weight, transposed
        DLTensor* W = args[0];
        int threads = args[1];

        CHECK_EQ(W->ndim, 2);

        int k = W->shape[1];
        int n = W->shape[0];

        BlockingFactors params;
        if (args.size() > 3) {
          int cntr = 3;
          params.MCB = args[cntr];
          params.NCB = args[cntr + 1];
          params.KCB = args[cntr + 2];
          params.MR = args[cntr + 3];
          params.NR = args[cntr + 4];
          params.NR_MIN = args[cntr + 5];
          params.ROW_INTERLEAVE = args[cntr + 6];

          auto packB = new PackBMatrix<std::int8_t, std::int32_t>(
              matrix_op_t::Transpose, k, n,
              reinterpret_cast<const std::int8_t*>(W->data), n, nullptr, 1,
              &params);
         //packB->printPackedMatrix("packingB");
          *ret = packB;
        } else {
          auto packB = new PackBMatrix<std::int8_t, std::int32_t>(
              matrix_op_t::Transpose, k, n,
              reinterpret_cast<const std::int8_t*>(W->data), n, nullptr, 1);
          *ret = packB;
        }

      }

    });

TVM_REGISTER_GLOBAL("tvm.contrib.fbgemm.compute_col_offsets_int8")
    .set_body([](TVMArgs args, TVMRetValue* ret) {

      bool trans = args[3];

      if (!trans) { // K * N; not transposed
        DLTensor* W = args[0];
        int threads = args[1];
        std::int32_t w_zero_point = args[2];

        //std::cout << W->shape << ' ';
        //CHECK_EQ(W->ndim, 2);
        int k = W->shape[0];
        int n = W->shape[1];

        std::vector<TensorQuantizationParams> temp_qparams;
        temp_qparams.push_back(TensorQuantizationParams{1.0, w_zero_point});

        std::vector<std::int32_t>* column_offsets_ =
            new std::vector<std::int32_t>;
        ComputeColumnOffsets<std::int8_t>(
            k, n, reinterpret_cast<const std::int8_t*>(W->data), temp_qparams,
            *column_offsets_);
        *ret = column_offsets_;

      } else { // N * K quantized int8 weight; transposed
        DLTensor* W = args[0];
        int threads = args[1];
        std::int32_t w_zero_point = args[2];

        //std::cout << W->shape << ' ';
        CHECK_EQ(W->ndim, 2);
        int k = W->shape[1];
        int n = W->shape[0];

        std::vector<TensorQuantizationParams> temp_qparams;
        temp_qparams.push_back(TensorQuantizationParams{1.0, w_zero_point});

        std::vector<std::int32_t>* column_offsets_ =
            new std::vector<std::int32_t>;
        ComputeColumnOffsets<std::int8_t>(
            k, n, reinterpret_cast<const std::int8_t*>(W->data), temp_qparams,
            *column_offsets_);
        *ret = column_offsets_;

      }

    });

TVM_REGISTER_GLOBAL("tvm.contrib.fbgemm.gemmint8acc32packedwt")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      DLTensor* X = args[0];  // M*K quantized int8 input

      std::uint64_t wt = args[1];
      void* weight = reinterpret_cast<void*>(static_cast<uint64_t>(wt));
      packbmatrix* packB =
          reinterpret_cast<PackBMatrix<std::int8_t, std::int32_t>*>(weight);
      packB->printPackedMatrix("packB");
      DLTensor* B = args[2];  // N quantized int8 bias
      DLTensor* Y = args[3];
      int threads = args[9];

      //CHECK_EQ(X->ndim, 2);
      // TODO: Ensure correctness here
      // CHECK_EQ(W->ndim, 2);
      // CHECK_EQ(X->shape[1], W->shape[1]);
      //CHECK_EQ(B->ndim, 1);
      // TODO: Ensure correctness here
      // CHECK_EQ(W->shape[0], B->shape[0]);

      float ReQuant_multiplier = (double)args[7];
      std::int32_t x_zero_point = args[4];
      std::int32_t w_zero_point = args[5];
      std::int32_t y_zero_point = args[6];

      int m = X->shape[0];
      int n = Y->shape[1];
      int k = X->shape[1];

      //std::uint64_t co = args[8];
      //void* col_offsts = reinterpret_cast<void*>(static_cast<uint64_t>(co));
      //std::vector<std::int32_t>* column_offsets_ =
      //   reinterpret_cast<std::vector<std::int32_t>*>(col_offsts);

     BlockingFactors params;
     if(args.size() > 10){
        int cntr = 10;
        params.MCB = args[cntr];
        params.NCB = args[cntr + 1];
        params.KCB = args[cntr + 2];
        params.MR = args[cntr + 3];
        params.NR = args[cntr + 4];
        params.NR_MIN = args[cntr + 5];
        params.ROW_INTERLEAVE = args[cntr + 6];


      PackAMatrix<std::uint8_t> packA(
          matrix_op_t::NoTranspose, m, k,
          reinterpret_cast<const std::uint8_t*>(X->data), k, nullptr, 1, &params);

      DoNothing<std::int32_t, std::int32_t> doNothing32BitObj;
      memCopy<> memcopyObj (doNothing32BitObj);

      fbgemmPacked(packA, *packB, reinterpret_cast<std::int32_t*>(Y->data),
                  reinterpret_cast<std::int32_t*>(Y->data), n, memcopyObj, 0,
                   threads, &params);

	    } /*else{

      PackAWithRowOffset<std::uint8_t> packA(
          matrix_op_t::NoTranspose, m, k,
          reinterpret_cast<const std::uint8_t*>(X->data), k, nullptr, 1);

      DoNothing<std::int32_t, std::int32_t> doNothing32BitObj;
      memCopy<> memcopyObj (doNothing32BitObj);

      fbgemmPacked(packA, *packB, reinterpret_cast<std::int32_t*>(Y->data),
                  reinterpret_cast<std::int32_t*>(Y->data), n, memcopyObj, 0,
                   threads);
     }*/
});

TVM_REGISTER_GLOBAL("tvm.contrib.fbgemm.gemmint8acc32packedwt_with_requant")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      DLTensor* X = args[0];  // M*K quantized int8 input
      std::uint64_t wt = args[1];
      void* weight = reinterpret_cast<void*>(static_cast<uint64_t>(wt));
      packbmatrix* packB =
          reinterpret_cast<PackBMatrix<std::int8_t, std::int32_t>*>(weight);

      DLTensor* B = args[2];  // N quantized int8 bias
      // ignore the axis and axis_w now for testing purpose
      DLTensor* Y = args[3];
      int threads = args[9];
      //DLTensor* W = args[10];

      //CHECK_EQ(X->ndim, 2);
      //CHECK_EQ(W->ndim, 2);
      //CHECK_EQ(B->ndim, 1);
      //CHECK_EQ(X->shape[1], W->shape[1]);
      //CHECK_EQ(W->shape[0], B->shape[0]);

      float ReQuant_multiplier = (double)args[7];
      std::int32_t x_zero_point = args[4];
      std::int32_t w_zero_point = args[5];
      std::int32_t y_zero_point = args[6];

      int m = X->shape[0];
      int n = Y->shape[1];
      int k = X->shape[1];

      BlockingFactors params;

      if(args.size() > 10) {
        int cntr = 10;
        params.MCB = args[cntr];
        params.NCB = args[cntr + 1];
        params.KCB = args[cntr + 2];
        params.MR = args[cntr + 3];
        params.NR = args[cntr + 4];
        params.NR_MIN = args[cntr + 5];
        params.ROW_INTERLEAVE = args[cntr + 6];
      }

      std::vector<std::int32_t> row_offsets_(
          PackAWithRowOffset<uint8_t>::rowOffsetBufferSize());
      std::vector<std::int32_t> Y_int32_(n * m);

      std::uint64_t co_addr = args[8];
      void* co = reinterpret_cast<void*>(static_cast<uint64_t>(co_addr));

      std::vector<std::int32_t>* column_offsets_ =
          reinterpret_cast<std::vector<std::int32_t>*>(co);

      std::vector<TensorQuantizationParams> temp_qparams;
      temp_qparams.push_back(TensorQuantizationParams{1.0, w_zero_point});

      if(args.size() > 10){

        PackAWithRowOffset<std::uint8_t> packA(
            matrix_op_t::NoTranspose, m, k,
            reinterpret_cast<const std::uint8_t*>(X->data), k, nullptr, 1,
            row_offsets_.data(), &params);

        DoNothing<> doNothingObj{};
        ReQuantizeOutput<false> outputProcObj(
            doNothingObj, &ReQuant_multiplier, y_zero_point, x_zero_point,
            &w_zero_point, packA.getRowOffsetBuffer(), (*column_offsets_).data(),
            reinterpret_cast<const std::int32_t*>(B->data), n);

        fbgemmPacked(packA, *packB, reinterpret_cast<std::uint8_t*>(Y->data),
                     Y_int32_.data(), n, outputProcObj, 0, threads,
                     &params);  // num_threads

      }  else {

        PackAWithRowOffset<std::uint8_t> packA(
            matrix_op_t::NoTranspose, m, k,
            reinterpret_cast<const std::uint8_t*>(X->data), k, nullptr, 1,
            row_offsets_.data());

        DoNothing<> doNothingObj{};
        ReQuantizeOutput<false> outputProcObj(
            doNothingObj, &ReQuant_multiplier, y_zero_point, x_zero_point,
            &w_zero_point, packA.getRowOffsetBuffer(), (*column_offsets_).data(),
            reinterpret_cast<const std::int32_t*>(B->data), n);

        fbgemmPacked(packA, *packB, reinterpret_cast<std::uint8_t*>(Y->data),
                     Y_int32_.data(), n, outputProcObj, 0,
                     threads);  // num_threads
      }
});

bool isValid(BlockingFactors *param)
{
   if (param->MCB % param->MR)
     return false;
   if (param->NCB % param->NR)
     return false;
   if (fbgemmHasAvx512Support()) {
     if (param->MR * (param->NCB / param->NR) > 24)
       return false;
   } else if (fbgemmHasAvx2Support()) {
     if (param->MR * (param->NCB / param->NR) > 16)
       return false;
   }

   return true;

}

TVM_REGISTER_GLOBAL("tvm.contrib.fbgemm.gemmint8acc32packedwt_for_tuning")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      DLTensor* X = args[0];  // M*K quantized int8 input
      DLTensor* W = args[1];  // K*N quantized uint8 weight

      DLTensor* B = args[2];  // N quantized int8 bias
      DLTensor* Y = args[3];
      int threads = args[9];

      float ReQuant_multiplier = (double)args[7];
      std::int32_t x_zero_point = args[4];
      std::int32_t w_zero_point = args[5];
      std::int32_t y_zero_point = args[6];

      int m = X->shape[0];
      int n = Y->shape[1];
      int k = X->shape[1];

      // For tuning these will be garbage values
      std::uint64_t co = args[8];
      //void* col_offsts = reinterpret_cast<void*>(static_cast<uint64_t>(co));
      //std::vector<std::int32_t>* column_offsets_ =
      //    reinterpret_cast<std::vector<std::int32_t>*>(col_offsts);

     BlockingFactors params;
     if(args.size() > 10){
        int cntr = 10;
        params.MCB = args[cntr];
        params.NCB = args[cntr + 1];
        params.KCB = args[cntr + 2];
        params.MR = args[cntr + 3];
        params.NR = args[cntr + 4];
        params.NR_MIN = args[cntr + 5];
        params.ROW_INTERLEAVE = args[cntr + 6];

        assert (isValid(&params) == true  && "incorrect configuration");


        static PackBMatrix<std::int8_t, std::int32_t> packB_ (
            matrix_op_t::NoTranspose, k, n,
            reinterpret_cast<const std::int8_t*>(W->data), n, nullptr, 1,
            &params);

      PackAMatrix<std::uint8_t> packA(
          matrix_op_t::NoTranspose, m, k,
          reinterpret_cast<const std::uint8_t*>(X->data), k, nullptr, 1,
          &params);

      DoNothing<std::int32_t, std::int32_t> doNothing32BitObj;
      memCopy<> memcopyObj (doNothing32BitObj);

      fbgemmPacked(packA, packB_, reinterpret_cast<std::int32_t*>(Y->data),
                  reinterpret_cast<std::int32_t*>(Y->data), n, memcopyObj, 0,
                   threads, &params);
	}
    });

TVM_REGISTER_GLOBAL("tvm.contrib.fbgemm.fully_connected_int8")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      DLTensor* X = args[0];  // M*K quantized int8 input
      DLTensor* W = args[1];  // N*K quantized int8 weight
      DLTensor* B = args[2];  // N quantized int8 bias
      // ignore the axis and axis_w now for testing purpose
      DLTensor* Y = args[3];
      int threads = args[8];

      //CHECK_EQ(X->ndim, 2);
      //CHECK_EQ(W->ndim, 2);
      //CHECK_EQ(B->ndim, 1);
      //CHECK_EQ(X->shape[1], W->shape[1]);
      //CHECK_EQ(W->shape[0], B->shape[0]);

      float ReQuant_multiplier = (double)args[7];
      std::int32_t x_zero_point = args[4];
      std::int32_t w_zero_point = args[5];
      std::int32_t y_zero_point = args[6];

      int m = X->shape[0];
      int n = Y->shape[1];
      int k = X->shape[1];

      BlockingFactors params;

      if (args.size() > 9) {
        params.MCB = args[9];
        params.NCB = args[10];
        params.KCB = args[11];
        params.MR = args[12];
        params.NR = args[13];
        params.NR_MIN = args[14];
        params.ROW_INTERLEAVE = args[15];
      }

      std::vector<std::int32_t> row_offsets_(
          PackAWithRowOffset<uint8_t>::rowOffsetBufferSize());
      std::vector<std::int32_t> Y_int32_(n * m);
      std::vector<std::int32_t> column_offsets_;

      std::vector<TensorQuantizationParams> temp_qparams;
      temp_qparams.push_back(TensorQuantizationParams{1.0, w_zero_point});

      if (args.size() > 9) {
        PackBMatrix<std::int8_t, std::int32_t> packB(
            matrix_op_t::NoTranspose, k, n,
            reinterpret_cast<const std::int8_t*>(W->data), k, nullptr, 1,
            &params);

        PackAWithRowOffset<std::uint8_t> packA(
            matrix_op_t::NoTranspose, m, k,
            reinterpret_cast<const std::uint8_t*>(X->data), k, nullptr, 1,
            row_offsets_.data(), &params);

        ComputeColumnOffsets<std::int8_t>(
            k, n, reinterpret_cast<const std::int8_t*>(W->data), temp_qparams,
            column_offsets_);

        DoNothing<> doNothingObj{};
        ReQuantizeOutput<false> outputProcObj(
            doNothingObj, &ReQuant_multiplier, y_zero_point, x_zero_point,
            &w_zero_point, packA.getRowOffsetBuffer(), column_offsets_.data(),
            reinterpret_cast<const std::int32_t*>(B->data), n);

        fbgemmPacked(packA, packB, reinterpret_cast<std::uint8_t*>(Y->data),
                     Y_int32_.data(), n, outputProcObj, 0, threads,
                     &params);  // num_threads

      } else {
        PackBMatrix<std::int8_t, std::int32_t> packB(
            matrix_op_t::NoTranspose, k, n,
            reinterpret_cast<const std::int8_t*>(W->data), k, nullptr, 1);

        PackAWithRowOffset<std::uint8_t> packA(
            matrix_op_t::NoTranspose, m, k,
            reinterpret_cast<const std::uint8_t*>(X->data), k, nullptr, 1,
            row_offsets_.data());

        ComputeColumnOffsets<std::int8_t>(
            k, n, reinterpret_cast<const std::int8_t*>(W->data), temp_qparams,
            column_offsets_);

        DoNothing<> doNothingObj{};
        ReQuantizeOutput<false> outputProcObj(
            doNothingObj, &ReQuant_multiplier, y_zero_point, x_zero_point,
            &w_zero_point, packA.getRowOffsetBuffer(), column_offsets_.data(),
            reinterpret_cast<const std::int32_t*>(B->data), n);

        fbgemmPacked(packA, packB, reinterpret_cast<std::uint8_t*>(Y->data),
                     Y_int32_.data(), n, outputProcObj, 0,
                     threads);  // num_threads
      }
    });

void col_offsets_with_zero_pt_s8acc32_ref(
    int K,
    int N,
    int ld,
    const int8_t* Bint8,
    const int32_t* B_zero_point,
    int32_t* col_offsets,
    int ncols_per_quant_group) {
  for (int j = 0; j < N; ++j) {
    int32_t sum = 0;
    for (int k = 0; k < K; ++k) {
      sum += Bint8[k * ld + j];
    }
    col_offsets[j] = sum - B_zero_point[j / ncols_per_quant_group] * K;
  }
}

TVM_REGISTER_GLOBAL("tvm.contrib.fbgemm.conv_int8")
    .set_body([](TVMArgs args, TVMRetValue* ret) {

    DLTensor* A = arg[0];
    DLTensor* B = arg[1];
    DLTensor* Y = arg[2];
    std::int32_t Aint8_zero_point = arg[3];
    std::int32_t Bint8_zero_point = arg[4];
    std::int32_t C_zero_point = arg[5];
    int C_multiplier = arg[6];
    shape = arg[7]; // edit array representation ISSUE 1
    std::cout  << typeid(shape).name() << '\n';
    std::cout  << typeid(shape[3]).name() << '\n';
    std::cout  << shape[3] << '\n';
    int nthreads = arg[8];

    //conv_param_t<> shape = conv_param_t<>(1, 128, 128, {56, 56}, 1, {3, 3}, {1, 1}, {1, 1, 1, 1});
    //ISSUE 2
    conv_param_t<> conv_p = conv_param_t<>(shape[0],
                                          shape[1],
                                          shape[2],
                                          shape[3],
                                          shape[4],
                                          shape[5],
                                          shape[6],
                                          shape[7]);

    CHECK_EQ(conv_p.IC % conv_p.G, 0);
    CHECK_EQ(conv_p.OC % conv_p.G, 0);
    //if (conv_p.IC % conv_p.G != 0 || conv_p.OC % conv_p.G != 0) {
      // invalid shapes
      //continue;
    //}

    int im_in_dim = accumulate(
        conv_p.IN_DIM.begin(), conv_p.IN_DIM.end(), 1, multiplies<int>());

    int kernel_dim =
        accumulate(conv_p.K.begin(), conv_p.K.end(), 1, multiplies<int>());

    int im_out_dim = accumulate(
        conv_p.OUT_DIM.begin(), conv_p.OUT_DIM.end(), 1, multiplies<int>());

    //aligned_vector<int32_t> Cint32_fb(conv_p.MB * im_out_dim * conv_p.OC);
    //aligned_vector<uint8_t> Cint8_fb(conv_p.MB * im_out_dim * conv_p.OC, 0);

    // matrix dimensions after im2col
    int MDim = conv_p.MB * im_out_dim;
    int NDim = conv_p.OC / conv_p.G;
    int KDim = kernel_dim * conv_p.IC;
    int KDimPerGroup = KDim / conv_p.G;
    int OC_per_G = conv_p.OC / conv_p.G;

    // computing column offset
    vector<int32_t> col_offsets(conv_p.OC);
    for (int g = 0; g < conv_p.G; ++g) {
      col_offsets_with_zero_pt_s8acc32_ref(
          KDimPerGroup,
          OC_per_G,
          OC_per_G,
          Bint8.data() + g * KDimPerGroup * OC_per_G,
          Bint8_zero_point.data(),
          col_offsets.data() + g * OC_per_G,
          conv_p.OC);
    }

    PackWeightsForConv<SPATIAL_DIM> packedB(conv_p, B.data());


    std::vector<std::int32_t> Y_int32_(conv_p.MB * im_out_dim * conv_p.OC);

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
        A.data(),
        packedB,
        Y.data(),
        Y_int32_.data(),
        outputProcObj,
        0,
        1);
    });

}  // namespace contrib
}  // namespace tvm
