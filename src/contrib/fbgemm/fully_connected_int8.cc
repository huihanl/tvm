/*!
 *  Copyright (c) 2019 by Contributors
 * \file Use external fbgemm library call.
 */

#include <cpuinfo.h>
#include <dmlc/logging.h>
#include <fbgemm/Fbgemm.h>
#include <fbgemm/FbgemmFP16.h>
#include <fbgemm/QuantUtilsAvx2.h>
#include <fbgemm/AlignedVec.h>

#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/util.h>
#include <memory>
#include <random>
#include "fbgemm_utils.h"

#include <sys/types.h>
#include <unistd.h>
#include <chrono>
#include <vector>
#include <array>

namespace tvm {
namespace runtime {

using namespace fbgemm;
using namespace std;

using packbmatrix = PackBMatrix<std::int8_t, std::int32_t>;
using packweight = PackWeightsForConv<2>;

template <>
struct extension_class_info<packbmatrix> {
  static const int code = 19;
};

template <>
struct extension_class_info<packweight> {
  static const int code = 20;
};

TVM_REGISTER_EXT_TYPE(packbmatrix);
TVM_REGISTER_EXT_TYPE(packweight);
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
/*
TVM_REGISTER_GLOBAL("tvm.contrib.fbgemm.pack_matrixB_int8")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      bool trans = args[2];
      if (!trans) {  //K * N, not transposed
        DLTensor* W = args[0];
        int threads = args[1];


      float ReQuant_multiplier = (double)args[7];
      std::int32_t x_zero_point = args[4];
      std::int32_t w_zero_point = args[5];
      std::int32_t y_zero_point = args[6];

      int m = X->shape[0];
      int n = Y->shape[1];
      int k = X->shape[1];

      int row = X->shape[0];
      int col = X->shape[1];

      matrix_op_t trans_param = matrix_op_t::NoTranspose;
      if (trans) {
	m = X->shape[1];
	k = X->shape[0];
	trans_param = matrix_op_t::Transpose;
      }

      BlockingFactors params;

      if(args.size() > 11) {
        int cntr = 11;
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

      if(args.size() > 11){

        PackAWithRowOffset<std::uint8_t> packA(
            trans_param, row, col,
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
            trans_param, row, col,
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
*/

//Add different implementation for transposed and untransposed weight.
TVM_REGISTER_GLOBAL("tvm.contrib.fbgemm.pack_matrixB_int8")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      bool trans = args[2];
      matrix_op_t trans_params = matrix_op_t::NoTranspose;
      DLTensor* W = args[0];
      int threads = args[1];

      CHECK_EQ(W->ndim, 2);

      int k = W->shape[0]; 
      int n = W->shape[1];
      int ld = W->shape[1];
      if (trans) {
        trans_params = matrix_op_t::Transpose;
	k = W->shape[1];
	n = W->shape[0];
      }
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
            trans_params, k, n,
            reinterpret_cast<const std::int8_t*>(W->data), ld, nullptr, 1,
            &params);
        *ret = packB;

      } else {
        auto packB = new PackBMatrix<std::int8_t, std::int32_t>(
            trans_params, k, n,
            reinterpret_cast<const std::int8_t*>(W->data), ld, nullptr, 1);
        *ret = packB;
      }

    });

//Add different implementation for transposed and untransposed weight.
TVM_REGISTER_GLOBAL("tvm.contrib.fbgemm.compute_col_offsets_int8")
    .set_body([](TVMArgs args, TVMRetValue* ret) {

      bool trans = args[3];

      DLTensor* W = args[0];
      int threads = args[1];
      std::int32_t w_zero_point = args[2];

      int k = W->shape[0];
      int n = W->shape[1];

      if (trans) { // N * K; transposed
        int inter = k;
        k = n;
        n = inter;
      }

      std::vector<TensorQuantizationParams> temp_qparams;
      temp_qparams.push_back(TensorQuantizationParams{1.0, w_zero_point});

      std::vector<std::int32_t>* column_offsets_ =
          new std::vector<std::int32_t>;
      ComputeColumnOffsets<std::int8_t>(
          k, n, reinterpret_cast<const std::int8_t*>(W->data), temp_qparams,
          *column_offsets_, trans);
      *ret = column_offsets_;

    });

TVM_REGISTER_GLOBAL("tvm.contrib.fbgemm.gemmint8acc32packedwt")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      DLTensor* X = args[0];  // M*K quantized int8 input

      std::uint64_t wt = args[1];
      void* weight = reinterpret_cast<void*>(static_cast<uint64_t>(wt));
      packbmatrix* packB =
          reinterpret_cast<PackBMatrix<std::int8_t, std::int32_t>*>(weight);

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

TVM_RhGISTER_GLOBAL("tvm.contrib.fbgemm.gemmint8acc32packedwt_for_tuning")
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


/*
 Supports prepacked weight matrix B and requantization. 
 It will receive a pointer for prepacked weight directly as its argument;
 It will also receive a pointer for col_offsets and other parameters for
 requantization.
*/
TVM_REGISTER_GLOBAL("tvm.contrib.fbgemm.gemmint8acc32packedwt_with_requant")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      std::cout << "REACH C++";
      DLTensor* X = args[0];  // M*K quantized int8 input
      std::uint64_t wt = args[1];
      void* weight = reinterpret_cast<void*>(static_cast<uint64_t>(wt));
      packbmatrix* packB =
          reinterpret_cast<PackBMatrix<std::int8_t, std::int32_t>*>(weight);

      DLTensor* B = args[2];  // N quantized int8 bias
      DLTensor* Y = args[3];
      bool trans = args[9];
      int threads = args[10];
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
      int ld = k;

      matrix_op_t trans_param = matrix_op_t::NoTranspose;
      if (trans) {
	m = X->shape[1];
	k = X->shape[0];
	trans_param = matrix_op_t::Transpose;
	ld = m;
      }


      BlockingFactors params;

      if(args.size() > 11) {
        int cntr = 11;
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

      if(args.size() > 11){

        PackAWithRowOffset<std::uint8_t> packA(
            trans_param, m, k,
            reinterpret_cast<const std::uint8_t*>(X->data), ld, nullptr, 1,
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
            trans_param, m, k,
            reinterpret_cast<const std::uint8_t*>(X->data), ld, nullptr, 1,
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


/*
TVM_REGISTER_GLOBAL("tvm.contrib.fbgemm.fully_connected_int8")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      DLTensor* X = args[0];  // M*K quantized int8 input
      DLTensor* W = args[1];  // N*K quantized int8 weight
      DLTensor* B = args[2];  // N quantized int8 bias
      // ignore the axis and axis_w now for testing purpose
      DLTensor* Y = args[3];
      int threads = args[8ecthhi
      j/CHECK_EQ(B->ndim, 1);
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
*/

TVM_REGISTER_GLOBAL("tvm.contrib.fbgemm.pack_matrixB_int8_conv")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
  freopen ("myfile.txt","w",stdout);
  printf ("This sentence is redirected to a file.");
	        std::cout << "000";
         DLTensor* W = args[0];
        std::cout << "00" << " " << std::endl;
        int spatial_dim = args[1];
        std::cout << "0." << " " << std::endl;
        int cntr = 2; //TODO: make changes on spatial_dim.
        int MB = args[cntr];
        int IC = args[cntr + 1];
        int OC = args[cntr + 2];
        //std::array<int, 2> IN_DIM = args[cntr + 3];

        //std::uint64_t id_addr = args[cntr + 3];
        //void* id_d = reinterpret_cast<void*>(static_cast<uint64_t>(id_addr));
        //std::array<int, 2>* IN_DIM =
        //reinterpret_cast<std::array<int, 2>*>(id_d);
	std::cout << "0" << " " << std::endl;
        DLTensor* id_addr = args[cntr + 3];
	std::cout << id_addr;
	int* id_pr = reinterpret_cast<int*>(id_addr->data);
	std::cout << "1" << " " << std::endl;
	std::array<int, 2> IN_DIM = {0, 0};
        std::cout << "2" << " " << std::endl;
	IN_DIM[0] = id_pr[0];
        std::cout << "3" << " " << std::endl;
        IN_DIM[1] = id_pr[1];	
        std::cout << "4" << " " << std::endl;
        int G = args[cntr + 4];
        //std::array<int, 2> K = args[cntr + 5];
        //std::uint64_t k_addr = args[cntr + 5];
        //void* k_d = reinterpret_cast<void*>(static_cast<uint64_t>(k_addr));
        //std::array<int, 2>* K =
        //reinterpret_cast<std::array<int, 2>*>(k_d);
        std::cout << "5" << " " << std::endl;
        DLTensor* k_addr = args[cntr + 5];
        int* k_pr = reinterpret_cast<int*>(k_addr->data);
        std::array<int, 2> K = {0, 0};
        K[0] = k_pr[0];
        K[1] = k_pr[1];
        std::cout << "6" << " " << std::endl;
        //std::array<int, 2> stride = args[cntr + 6];
        //std::uint64_t s_addr = args[cntr + 6];
        //void* s_d = reinterpret_cast<void*>(static_cast<uint64_t>(s_addr));
        //std::array<int, 2>* stride =
        //reinterpret_cast<std::array<int, 2>*>(s_d);
        DLTensor* s_addr = args[cntr + 6];
        int* s_pr = reinterpret_cast<int*>(s_addr->data);
        std::array<int, 2> stride = {0, 0};
        stride[0] = s_pr[0];
        stride[1] = s_pr[1];
        std::cout << "7" << " " << std::endl;
        
        //std::array<int, 4> pad = args[cntr + 7];
        //std::uint64_t p_addr = args[cntr + 7];
        //void* p_d = reinterpret_cast<void*>(static_cast<uint64_t>(p_addr));
        //std::array<int, 4>* pad =
        //reinterpret_cast<std::array<int, 4>*>(p_d);
        DLTensor* pad_addr = args[cntr + 7];
        int* p_pr = reinterpret_cast<int*>(pad_addr->data);
        std::array<int, 4> pad = {0, 0, 0, 0};
        pad[0] = p_pr[0];
        pad[1] = p_pr[1];
        pad[2] = p_pr[2];
        pad[3] = p_pr[3];
        std::cout << "8" << " " << std::endl;
        std::cout << "pad[0]" << pad[0] << "pad[1]" << pad[1] << "pad[2]" << pad[2] << "pad[3]" << pad[3] <<" " << std::endl;
         //conv_param_t<> shape = conv_param_t<>(1, 128, 128, {56, 56}, 1, {3, 3}, {1, 1}, {1, 1, 1, 1});
        conv_param_t<> conv_p = conv_param_t<>(MB, IC, OC, IN_DIM, G, K, stride, pad);
        std::cout << "9" << " " << std::endl;
fclose (stdout);
         BlockingFactors params;

         if (args.size() > 11) {
          int cntr = 10;
          params.MCB = args[cntr];
          params.NCB = args[cntr + 1];
          params.KCB = args[cntr + 2];
          params.MR = args[cntr + 3];
          params.NR = args[cntr + 4];
          params.NR_MIN = args[cntr + 5];
          params.ROW_INTERLEAVE = args[cntr + 6];

           PackWeightsForConv<2> packedB(conv_p, reinterpret_cast<std::int8_t*>(W->data), &params);
         //packB->printPackedMatrix("packingB"); invalid conversion from ‘int8_t* {aka signed char*}’ to ‘int’ [-fpermissive]
          *ret = packedB;

         } else {

           PackWeightsForConv<2> packedB(conv_p, reinterpret_cast<std::int8_t*>(W->data));
          *ret = packedB;

         }

     });

TVM_REGISTER_GLOBAL("tvm.contrib.fbgemm.conv_int8")
    .set_body([](TVMArgs args, TVMRetValue* ret) {

    DLTensor* A = args[0];

    std::uint64_t wt = args[1];
    void* weight = reinterpret_cast<void*>(static_cast<uint64_t>(wt));
    PackWeightsForConv<2>* packedB =
        reinterpret_cast<PackWeightsForConv<2>*>(weight);

    DLTensor* Y = args[2];
    std::int32_t Aint8_zero_point = args[3];
    //aligned_vector<float> Bint8_zero_point = args[4];

    std::uint64_t zp_addr = args[4];
    void* zp = reinterpret_cast<void*>(static_cast<uint64_t>(zp_addr));
    aligned_vector<int32_t>* Bint8_zero_point =
        reinterpret_cast<aligned_vector<int32_t>*>(zp);

    std::int32_t C_zero_point = args[5];

    //aligned_vector<float> C_multiplier = ;
    //reinterpret_cast<std::vector<float>*>(args[6])

    std::uint64_t mul_addr = args[6];
    void* mula = reinterpret_cast<void*>(static_cast<uint64_t>(mul_addr));
    aligned_vector<float>* C_multiplier =
        reinterpret_cast<aligned_vector<float>*>(mula);

    std::uint64_t co_addr = args[7];
    void* co = reinterpret_cast<void*>(static_cast<uint64_t>(co_addr));
    std::vector<std::int32_t>* column_offsets_ =
        reinterpret_cast<std::vector<std::int32_t>*>(co);

    int cntr = 8;
    int MB = args[cntr];
    int IC = args[cntr + 1];
    int OC = args[cntr + 2];
    //std::array<int, 2> IN_DIM = args[cntr + 3];
    std::uint64_t id_addr = args[cntr + 3];
    void* id_d = reinterpret_cast<void*>(static_cast<uint64_t>(id_addr));
    std::array<int, 2>* IN_DIM =
        reinterpret_cast<std::array<int, 2>*>(id_d);
    int G = args[cntr + 4];
    
        //std::array<int, 2> K = args[cntr + 5];
        std::uint64_t k_addr = args[cntr + 5];
        void* k_d = reinterpret_cast<void*>(static_cast<uint64_t>(k_addr));
        std::array<int, 2>* K =
        reinterpret_cast<std::array<int, 2>*>(k_d);
        
        //std::array<int, 2> stride = args[cntr + 6];
        std::uint64_t s_addr = args[cntr + 6];
        void* s_d = reinterpret_cast<void*>(static_cast<uint64_t>(s_addr));
        std::array<int, 2>* stride =
        reinterpret_cast<std::array<int, 2>*>(s_d);
        
        //std::array<int, 4> pad = args[cntr + 7];
        std::uint64_t p_addr = args[cntr + 7];
        void* p_d = reinterpret_cast<void*>(static_cast<uint64_t>(p_addr));
        std::array<int, 4>* pad =
        reinterpret_cast<std::array<int, 4>*>(p_d);

         //conv_param_t<> shape = conv_param_t<>(1, 128, 128, {56, 56}, 1, {3, 3}, {1, 1}, {1, 1, 1, 1});
        conv_param_t<> conv_p = conv_param_t<>(MB, IC, OC, *IN_DIM, G, *K, *stride, *pad);

//std::array<int, 2> K = args[cntr + 5];
    //std::array<int, 2> stride = args[cntr + 6];
    //std::array<int, 4> pad = args[cntr + 7];
    //int nthreads = args[cntr + 8];

    //conv_param_t<> shape = conv_param_t<>(1, 128, 128, {56, 56}, 1, {3, 3}, {1, 1}, {1, 1, 1, 1});
    //ISSUE 2

//    conv_param_t<> conv_p = conv_param_t<>(MB, IC, OC, IN_DIM, G, K, stride, pad);

    CHECK_EQ(conv_p.IC % conv_p.G, 0);
    CHECK_EQ(conv_p.OC % conv_p.G, 0);

    //if (conv_p.IC % conv_p.G != 0 || conv_p.OC % conv_p.G != 0) {
      // invalid shapes
      //continue;
    //}

    //int im_in_dim = accumulate(
    //    conv_p.IN_DIM.begin(), conv_p.IN_DIM.end(), 1, multiplies<int>());

    int kernel_dim =
        accumulate(conv_p.K.begin(), conv_p.K.end(), 1, multiplies<int>());

    int im_out_dim = accumulate(
        conv_p.OUT_DIM.begin(), conv_p.OUT_DIM.end(), 1, multiplies<int>());

    //aligned_vector<int32_t> Cint32_fb(conv_p.MB * im_out_dim * conv_p.OC);
    //aligned_vector<uint8_t> Cint8_fb(conv_p.MB * im_out_dim * conv_p.OC, 0);

    // matrix dimensions after im2col
    //int MDim = conv_p.MB * im_out_dim;
    //int NDim = conv_p.OC / conv_p.G;
    int KDim = kernel_dim * conv_p.IC;
    int KDimPerGroup = KDim / conv_p.G;
    int OC_per_G = conv_p.OC / conv_p.G;

    std::vector<std::int32_t> Y_int32_(conv_p.MB * im_out_dim * conv_p.OC);

    // no-op output process objects
    DoNothing<> doNothingObj{};
    ReQuantizeOutput<false, QuantizationGranularity::TENSOR> outputProcObj(
        doNothingObj,
        //C_multiplier.data(),
        C_multiplier->data(),
        C_zero_point,
        Aint8_zero_point,
        //Bint8_zero_point.data(),
        Bint8_zero_point->data(),
        nullptr, // row offsets
        (*column_offsets_).data(),
        nullptr, // bias
        conv_p.OC,
        conv_p.G);

    fbgemmConv(
        conv_p,
        reinterpret_cast<const std::uint8_t*>(A->data),
        *packedB,
        reinterpret_cast<std::uint8_t*>(Y->data),
        Y_int32_.data(),
        outputProcObj,
        0,
        1);

    });

}  // namespace contrib
}  // namespace tvm
