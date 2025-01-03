#include <algorithm>
#include <cstddef>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <vector>

#include "ciphertext.h"
#include "matrix_mul.cuh"
#include "utils.cuh"

using namespace std;
using namespace phantom::util;
using namespace phantom::arith;
using namespace nexus;

__global__ void kernel_compress_ciphertext(uint64_t *plain_data, size_t plain_scale, size_t degree,
                                           const DModulus *moduli, const double *values) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < degree) {
    auto coeffd = std::round(values[idx] * plain_scale);
    bool is_negative = std::signbit(coeffd);
    auto coeffu = static_cast<std::uint64_t>(std::fabs(coeffd));

    if (is_negative) {
      for (std::size_t j = 0; j < 2; j++) {
        plain_data[idx + (j * degree)] = negate_uint64_mod(
            barrett_reduce_uint64_uint64(coeffu, moduli[j].value(), moduli[j].const_ratio()[1]), moduli[j].value());
      }
    } else {
      for (std::size_t j = 0; j < 2; j++) {
        plain_data[idx + (j * degree)] = barrett_reduce_uint64_uint64(coeffu, moduli[j].value(), moduli[j].const_ratio()[1]);
      }
    }
  }
}

__global__ void kernel_negacyclic_shift(const uint64_t *cipher_data, const size_t cipher_count, const uint64_t coeff_count, const size_t mod_count,
                                        const int shift, const DModulus *moduli, uint64_t *dest_data) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < cipher_count * mod_count * coeff_count) {
    if (shift == 0) {
      dest_data[idx] = cipher_data[idx];
      return;
    }

    size_t i = idx / (mod_count * coeff_count);
    size_t j = (idx / coeff_count) % mod_count;
    size_t k = idx % coeff_count;
    size_t mask = coeff_count - 1;
    uint64_t modulus_value = moduli[j].value();

    size_t index = (shift + k) & mask;
    size_t result_index = i * mod_count * coeff_count + j * coeff_count + index;
    if (cipher_data[idx] == 0 || ((shift + k) & coeff_count) == 0) {
      dest_data[result_index] = cipher_data[idx];
    } else {
      dest_data[result_index] = modulus_value - cipher_data[idx];
    }
  }
}

// FIXME: 2x speedup if correctly implemented
// void MMEvaluator::multiply_power_of_x(PhantomCiphertext &encrypted, PhantomCiphertext &destination, int index) {
//   auto context = ckks->context;
//   auto coeff_count = ckks->degree;
//   auto param = context->get_context_data(encrypted.params_id()).parms();
//   auto moduli = context->gpu_rns_tables().modulus();
//   auto coeff_mod_count = param.coeff_modulus().size();
//   auto encrypted_count = encrypted.size();
//   auto total_coeff_count = encrypted_count * coeff_mod_count * coeff_count;

//   const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream;
//   const auto &stream = stream_wrapper.get_stream();

//   destination = encrypted;
//   ckks->evaluator.transform_from_ntt_inplace(destination);
//   PhantomCiphertext destination_copy = destination;

//   uint64_t gridDimGlb = total_coeff_count / blockDimGlb.x;
//   kernel_negacyclic_shift<<<gridDimGlb, blockDimGlb, total_coeff_count * sizeof(uint64_t), stream>>>(
//       destination_copy.data(), encrypted_count, coeff_count, coeff_mod_count, index, moduli, destination.data());

//   ckks->evaluator.transform_to_ntt_inplace(destination);
// }

void MMEvaluator::multiply_power_of_x(PhantomCiphertext &encrypted, PhantomCiphertext &destination, int index) {
  auto context = ckks->context;
  auto coeff_count = ckks->degree;
  auto param = context->get_context_data(encrypted.params_id()).parms();
  auto moduli = param.coeff_modulus();
  auto coeff_mod_count = param.coeff_modulus().size();
  auto encrypted_count = encrypted.size();
  auto rns_coeff_count = coeff_count * coeff_mod_count;

  const auto &stream = phantom::util::global_variables::default_stream->get_stream();

  destination = encrypted;
  ckks->evaluator.transform_from_ntt_inplace(destination);

  auto dest_data = new uint64_t[rns_coeff_count * encrypted_count];
  auto dest_data_copy = new uint64_t[rns_coeff_count * encrypted_count];
  cudaMemcpyAsync(dest_data, destination.data(), encrypted_count * rns_coeff_count * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream);
  std::copy(dest_data, dest_data + rns_coeff_count * encrypted_count, dest_data_copy);

  for (int i = 0; i < encrypted_count; i++) {
    for (int j = 0; j < coeff_mod_count; j++) {
      uint64_t *poly = dest_data_copy + i * rns_coeff_count + j * coeff_count;
      uint64_t *result = dest_data + i * rns_coeff_count + j * coeff_count;

      uint64_t index_raw = index;
      uint64_t coeff_count_mod_mask = static_cast<uint64_t>(coeff_count) - 1;
      for (size_t k = 0; k < coeff_count; k++, poly++, index_raw++) {
        uint64_t index = index_raw & coeff_count_mod_mask;
        if (!(index_raw & static_cast<uint64_t>(coeff_count)) || !*poly) {
          result[index] = *poly;
        } else {
          result[index] = moduli[j].value() - *poly;
        }
      }
    }
  }

  cudaMemcpyAsync(destination.data(), dest_data, encrypted_count * rns_coeff_count * sizeof(uint64_t), cudaMemcpyHostToDevice, stream);

  delete[] dest_data;
  delete[] dest_data_copy;

  ckks->evaluator.transform_to_ntt_inplace(destination);
}

void MMEvaluator::enc_compress_ciphertext(vector<double> &values, PhantomCiphertext &ct) {
  size_t plain_scale = 10000000000;

  auto &context_data = ckks->context->first_context_data();
  auto param = context_data.parms();
  auto moduli = ckks->context->gpu_rns_tables().modulus();
  auto coeff_modulus_size = param.coeff_modulus().size();
  auto poly_modulus_degree = param.poly_modulus_degree();

  const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream;
  const auto &stream = stream_wrapper.get_stream();

  PhantomPlaintext p;
  p.resize(coeff_modulus_size, poly_modulus_degree, stream);

  auto gpu_values = make_cuda_auto_ptr<double>(values.size(), stream);
  cudaMemcpyAsync(gpu_values.get(), values.data(), values.size() * sizeof(double), cudaMemcpyHostToDevice, stream);

  kernel_compress_ciphertext<<<poly_modulus_degree / blockDimGlb.x, blockDimGlb, 0, stream>>>(
      p.data(), plain_scale, poly_modulus_degree, moduli, gpu_values.get());

  // Transform polynomials to the NTT domain
  nwt_2d_radix8_forward_inplace(p.data(), ckks->context->gpu_rns_tables(), coeff_modulus_size, 0, stream);

  // Update plaintext parameters
  p.parms_id() = context_data.parms().parms_id();
  p.set_chain_index(context_data.chain_index());
  p.scale() = plain_scale;

  // Create a ciphertext encrypting zero
  PhantomPlaintext zero_pt;
  PhantomCiphertext zero;
  ckks->encoder.encode(0.0, plain_scale, zero_pt);
  ckks->encryptor.encrypt(zero_pt, zero);

  // Encrypt the plaintext
  ckks->evaluator.add_plain(zero, p, ct);
}

vector<PhantomCiphertext> MMEvaluator::decompress_ciphertext(PhantomCiphertext &encrypted) {
  auto N = ckks->degree;
  uint32_t logN = ceil(log2(N));

  vector<PhantomCiphertext> temp;
  temp.push_back(encrypted);

  PhantomCiphertext tempctxt_rotated;
  PhantomCiphertext tempctxt_shifted;
  PhantomCiphertext tempctxt_rotatedshifted;

  for (uint32_t i = 0; i < logN; i++) {
    vector<PhantomCiphertext> newtemp(temp.size() << 1);

    uint32_t galois_elt = ckks->galois_elts[i];
    int index_raw = (N << 1) - (1 << i);
    int index = (index_raw * galois_elt) % (N << 1);

    for (uint32_t a = 0; a < temp.size(); a++) {
      ckks->evaluator.apply_galois(temp[a], galois_elt, *(ckks->galois_keys), tempctxt_rotated);  // sub
      ckks->evaluator.add(temp[a], tempctxt_rotated, newtemp[a]);
      multiply_power_of_x(temp[a], tempctxt_shifted, index_raw);  // x**-1
      // if (temp.size() == 1) ckks->print_decrypted_ct(tempctxt_shifted, 10);
      multiply_power_of_x(tempctxt_rotated, tempctxt_rotatedshifted, index);
      // if (temp.size() == 1) ckks->print_decrypted_ct(tempctxt_rotatedshifted, 10);
      ckks->evaluator.add(tempctxt_shifted, tempctxt_rotatedshifted, newtemp[a + temp.size()]);
    }

    temp = newtemp;
  }

  return temp;
}

void MMEvaluator::matrix_mul(vector<vector<double>> &x, vector<vector<double>> &y, vector<PhantomCiphertext> &res) {
  const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream;
  const auto &stream = stream_wrapper.get_stream();

  auto timer = Timer();

  // Encode plaintext
  vector<PhantomPlaintext> a_pts;
  a_pts.reserve(768);

  for (int i = 0; i < 768; i++) {
    PhantomPlaintext pt;
    ckks->encoder.encode(x[i], ckks->scale, pt);
    a_pts.push_back(pt);
  }

  // Ciphertext encoding & compression
  timer.start();

  int b_cts_count = 768 * 64 / ckks->degree;
  vector<PhantomCiphertext> b_compressed_cts;
  b_compressed_cts.reserve(b_cts_count);

  for (int i = 0; i < b_cts_count; i++) {
    PhantomCiphertext ct;
    enc_compress_ciphertext(y[i], ct);
    b_compressed_cts.push_back(ct);
  }

  timer.stop();
  cout << "Compression took: " << timer.duration<milliseconds>() << " milliseconds" << endl;

  // Ciphertext decompression
  timer.start();

  vector<PhantomCiphertext> b_expanded_cts;

  for (auto i = 0; i < b_compressed_cts.size(); i++) {
    vector<PhantomCiphertext> temp_cts = decompress_ciphertext(b_compressed_cts[i]);
    cout << "Expanded ciphertext #" << i + 1 << endl;
    // ckks->print_decrypted_ct(temp_cts[0], 10);
    b_expanded_cts.insert(b_expanded_cts.end(), make_move_iterator(temp_cts.begin()), make_move_iterator(temp_cts.end()));
  }

  timer.stop();
  cout << "Decompression took: " << timer.duration<seconds>() << " seconds" << endl;

  // Perform plain-cipher matrix multiplication
  timer.start();

  for (int i = 0; i < 64; i++) {
    PhantomCiphertext res_col_ct;
    vector<PhantomCiphertext> temp_cts(768);

    for (int j = 0; j < 768; j++) {
      ckks->evaluator.multiply_plain(b_expanded_cts[i * 768 + j], a_pts[j], temp_cts[j]);
    }

    res_col_ct.scale() = temp_cts[0].scale();
    ckks->evaluator.add_many(temp_cts, res_col_ct);

    res_col_ct.scale() *= 4096;
    res.push_back(res_col_ct);
  }

  for (auto &ct : res) {
    while (ct.coeff_modulus_size() > 1) {
      ckks->evaluator.rescale_to_next_inplace(ct);
    }
  }

  timer.stop();
  cout << "Result calculation time: " << timer.duration<milliseconds>() << " milliseconds" << endl;
}

// TODO: move to device side
inline vector<double> rotate(vector<double>& x, int steps) {
  vector<double> out(x.size());
  for (int i = 0; i < x.size(); i++) {
    out[i] = x[(i+steps) % x.size()];
  }
  return out;
}

__global__ void kernel_pt_encoding_128x128(double* pt, double* rot, double* out) {
  // pt.len = 32768 = 2x128x128; out.len = 256x32768
  // i=blockIdx.x, j/x=threadIdx.x
  constexpr size_t slot_count = 32768;
  int rot_offset = blockIdx.x * slot_count;

  for (int y=0; y<slot_count/128; y++) {
    int xx = (threadIdx.x + y - blockIdx.x + 128) % 128;
    int yy = ((threadIdx.x+y)*128 + xx) % (128*128);
    assert(xx >= 0);
    if (y < 128)
      rot[rot_offset+xx+128*y] = pt[yy];
    else
      rot[rot_offset+xx+128*y] = pt[128*128 + yy];
  }

  __syncthreads();

  for (int y=0; y<slot_count/128; y++) {
    int idx = 128*y + 127- threadIdx.x;
    if(threadIdx.x < blockIdx.x) {
      out[(blockIdx.x+128) * slot_count + idx] = 0;
      out[(blockIdx.x) * slot_count + idx] = rot[rot_offset+idx];
    } else {
      out[(blockIdx.x+128) * slot_count + idx] = rot[rot_offset+idx];
      out[(blockIdx.x) * slot_count + idx] = 0;
    }
  }
}

void MMEvaluator::matrix_mul_ct128x128_pt128x128(PhantomCiphertext& ct, vector<double>& pt, PhantomCiphertext &res) {
  const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream;
  const auto &stream = stream_wrapper.get_stream();
  assert (pt.size() == slot_count);
  vector<vector<double>> cleartexts(256, vector<double>(slot_count));
  double *d_pt, *tmp, *out_buf;
  
  cudaMalloc(&d_pt, slot_count*sizeof(double));
  cudaMalloc(&tmp, 128*slot_count*sizeof(double));
  cudaMalloc(&out_buf, 256*slot_count*sizeof(double));
  cudaMemcpy(d_pt, pt.data(), slot_count*sizeof(double), cudaMemcpyHostToDevice);
  kernel_pt_encoding_128x128<<<128, 128>>>(d_pt, tmp, out_buf);
  for (int i=0; i<256; i++)
    cudaMemcpy(cleartexts[i].data(), out_buf+i*slot_count, slot_count*sizeof(double), cudaMemcpyDeviceToHost);
  cudaFreeAsync(d_pt, stream);
  cudaFreeAsync(tmp, stream);
  cudaFreeAsync(out_buf, stream);

  for (auto gs=-128; gs<128; gs+=16)
    for (int bs=0; bs<16; bs++) {
      cleartexts[bs+gs+128] = rotate(cleartexts[bs+gs+128], -gs);
    }
  vector<PhantomCiphertext> babysteps(16);
  for (int bs=0; bs<16; bs++) {
    ckks->evaluator.rotate_vector(ct, bs, *(ckks->galois_keys), babysteps[bs]);
  }

  PhantomCiphertext tmpct, bsSum;
  PhantomPlaintext tmppt;
  for (int gs=-128; gs<128; gs+=16) {
    for (int bs=0; bs<16; bs++) {
      ckks->encoder.encode(cleartexts[bs+gs+128], babysteps[bs].chain_index(), ckks->scale, tmppt);
      ckks->evaluator.multiply_plain(babysteps[bs], tmppt, tmpct);
      if (bs == 0)
        bsSum = tmpct;
      else
        ckks->evaluator.add_inplace(bsSum, tmpct);
    }
    ckks->evaluator.rotate_vector_inplace(bsSum, gs, *(ckks->galois_keys));
    if (gs == -128)
      res = bsSum;
    else
      ckks->evaluator.add_inplace(res, bsSum);
  }
  ckks->evaluator.rescale_to_next_inplace(res);
}

void MMEvaluator::matrix_mul_ct128x768_pt768x128(vector<PhantomCiphertext>& ct, vector<vector<double>>& pt, PhantomCiphertext &res) {
  PhantomCiphertext product, rotProduct;
  for (int i=0; i<3; i++) {
    matrix_mul_ct128x128_pt128x128(ct[i], pt[i], product);
    ckks->evaluator.rotate_vector(product, slot_count/2, *(ckks->galois_keys), rotProduct);
    ckks->evaluator.add_inplace(product, rotProduct);
    if (i == 0)
      res = product;
    else
      ckks->evaluator.add_inplace(res, product);
  }
}

void MMEvaluator::matrix_mul_ct128x768_pt768x64x2(vector<PhantomCiphertext>& ct, vector<vector<double>>& pt, PhantomCiphertext &res) {
  vector<PhantomCiphertext> buf0(3), buf1(3);
  for (int i=0; i<3; i++) {
    matrix_mul_ct128x128_pt128x128(ct[i],pt[i], buf0[i]);
    matrix_mul_ct128x128_pt128x128(ct[i],pt[i+3], buf1[i]);
  }
  PhantomCiphertext result0, result1;
  ckks->evaluator.add_many(buf0, result0);
  ckks->evaluator.add_many(buf1, result1);
  ckks->evaluator.rotate_vector_inplace(result1, slot_count / 2, *(ckks->galois_keys));
  ckks->evaluator.add(result0, result1, res);
}

void MMEvaluator::matrix_mul_ct128x768_pt768x768(vector<PhantomCiphertext>& ct, vector<vector<vector<double>>>& pt, vector<PhantomCiphertext> &res) {
  res.resize(3);
  for (int i=0; i<3; i++) {
    matrix_mul_ct128x768_pt768x64x2(ct, pt[i], res[i]);
  }
}

void MMEvaluator::matrix_mul_ct128x64_ct128x64_transpose(PhantomCiphertext& ct1, PhantomCiphertext& ct2, PhantomCiphertext &res) {
  vector<PhantomCiphertext> babyStepsL(8), babyStepsR(8);
  for (int bs=0; bs<8; bs++) {
    ckks->evaluator.rotate_vector(ct2, 128*bs, *(ckks->galois_keys), babyStepsL[bs]);
    ckks->evaluator.rotate_vector(ct2, -128*(128-bs), *(ckks->galois_keys), babyStepsR[bs]);
  }
  for (int gs=0; gs<16; gs++) {
    PhantomCiphertext ct1Rot, sumBs;
    ckks->evaluator.rotate_vector(ct1, -1024*gs, *(ckks->galois_keys), ct1Rot);
    for (int bs=0; bs<8; bs++) {
      int i = bs + 8*gs;
      vector<double> maskL(slot_count, 0.0);
      vector<double> maskR(slot_count, 0.0);
      for (int k=0; k<2; k++) {
        for (int y=0; y<slot_count / 128 / 2; y++) {
          for (int x=0; x<128; x++) {
            if (y < i)
              maskR[128*(127 - y) + 128*128*k + x] = 1;
            else
              maskL[128*(127 - y) + 128*128*k + x] = 1;
          }
        }
      }
      PhantomCiphertext ct2RotL, ct2RotR, ct2Rot, sumi, tmp;
      PhantomPlaintext pt;
      
      maskL = rotate(maskL,-1024*gs);
      maskR = rotate(maskR,-1024*gs);

      ckks->encoder.encode(maskL, babyStepsL[bs].chain_index(), ckks->scale, pt);
      ckks->evaluator.multiply_plain(babyStepsL[bs], pt, ct2RotL);
      ckks->encoder.encode(maskR, babyStepsR[bs].chain_index(), ckks->scale, pt);
      ckks->evaluator.multiply_plain(babyStepsR[bs], pt, ct2RotR);
      ckks->evaluator.add(ct2RotL, ct2RotR, ct2Rot);
      ckks->evaluator.rescale_to_next_inplace(ct2Rot);
      ckks->evaluator.multiply_reduced_error(ct1Rot, ct2Rot, *(ckks->relin_keys), sumi);
      ckks->evaluator.rescale_to_next_inplace(sumi);
      for (int j=1; j<64; j*=2) {
        ckks->evaluator.rotate_vector(sumi, j, *(ckks->galois_keys), tmp);
        ckks->evaluator.add_inplace(sumi, tmp);
      }

      // TODO: attention mask
      vector<double> mask(slot_count, 0.0);
      for (int k=0; k<2; k++) {
        for (int j=0; j<128; j++) {
          // mask[128*128*k + 128*jj] = att_mask[(i + jj) % 128];
          mask[128*128*k + 128*j] = 1;
        }
      }
      mask = rotate(mask,-1024*gs);
      ckks->encoder.encode(mask, sumi.chain_index(), ckks->scale, pt);
      ckks->evaluator.multiply_plain_inplace(sumi, pt);
      ckks->evaluator.rescale_to_next_inplace(sumi);

      if (bs == 0)
        sumBs = sumi;
      else {
        ckks->evaluator.rotate_vector_inplace(sumi, -bs, *(ckks->galois_keys));
        ckks->evaluator.add_inplace(sumBs, sumi);
      }
    }
    if (gs == 0)
      res = sumBs;
    else {
      ckks->evaluator.rotate_vector_inplace(sumBs, (1024 - 8)*gs, *(ckks->galois_keys));
      ckks->evaluator.add_inplace(res, sumBs);
    }
  }
}

void MMEvaluator::matrix_mul_ct128x128_ct128x128(PhantomCiphertext& ct1, PhantomCiphertext& ct2, PhantomCiphertext &res) {
  vector<PhantomCiphertext> rotBS(8);
  for (int bs=0; bs<8; bs++) {
    ckks->evaluator.rotate_vector(ct2, 128*bs, *(ckks->galois_keys), rotBS[bs]);
  }

  PhantomCiphertext diag;
  for (int gs=0; gs<16; gs++) {
    PhantomCiphertext diagBS;
    for (int bs=0; bs<8; bs++) {
      int i = bs + 8*gs;
      vector<double> mask(slot_count, 0.0);
      for (int y=0; y<slot_count / 128; y++) {
        mask[i+128*y] = 1;
      }
      mask = rotate(mask,-128*8*gs);

      PhantomCiphertext rot;
      PhantomPlaintext pt;

      ckks->encoder.encode(mask, rotBS[bs].chain_index(), ckks->scale, pt);
      ckks->evaluator.multiply_plain(rotBS[bs], pt, rot);
      ckks->evaluator.rescale_to_next_inplace(rot);

      if (bs == 0)
        diagBS = rot;
      else {
        ckks->evaluator.add_inplace(diagBS, rot);
      }
    }
    if (gs == 0)
      diag = diagBS;
    else {
      ckks->evaluator.rotate_vector_inplace(diagBS, 128*8*gs, *(ckks->galois_keys));
      ckks->evaluator.add_inplace(diag, diagBS);
    }
  }

  vector<PhantomCiphertext> diagBS(8);
  for (int bs=0; bs<8; bs++) {
    ckks->evaluator.rotate_vector(diag, 128*bs, *(ckks->galois_keys), diagBS[bs]);
  }

  for (int gs=0; gs<16; gs++) {
    PhantomCiphertext sumbs;
    for (int bs=0; bs<8; bs++) {
      int i = bs + 8*gs;
      vector<double> maskL(slot_count, 1.0);
      vector<double> maskR(slot_count, 1.0);
      for (int y = 0; y < slot_count / 128; ++y) {
        for (int x = 0; x < 128; ++x) {
            int idx = 127 - x;
            if (x < i) {
                maskL[128 * y + idx] = 0;
            } else {
                maskR[128 * y + idx] = 0;
            }
            if (y < 128 && idx >= 64) {
                maskL[128 * y + idx] = 0;
                maskR[128 * y + idx] = 0;
            } else if (y >= 128 && idx < 64) {
                maskL[128 * y + idx] = 0;
                maskR[128 * y + idx] = 0;
            }
        }
      }
      maskL = rotate(maskL,-128*8*gs);
      maskR = rotate(maskR,-128*8*gs);

      PhantomCiphertext rotL, rotR, diagL, diagR, vec;
      PhantomPlaintext pt;

      ckks->evaluator.rotate_vector(ct1, bs-127*8*gs, *(ckks->galois_keys), rotL);
      ckks->evaluator.rotate_vector(ct1, -128+bs-127*8*gs, *(ckks->galois_keys), rotR);

      ckks->encoder.encode(maskL, diagBS[bs].chain_index(), ckks->scale, pt);
      ckks->evaluator.multiply_plain(diagBS[bs], pt, diagL);
      ckks->evaluator.rescale_to_next_inplace(diagL);
      ckks->encoder.encode(maskR, diagBS[bs].chain_index(), ckks->scale, pt);
      ckks->evaluator.multiply_plain(diagBS[bs], pt, diagR);
      ckks->evaluator.rescale_to_next_inplace(diagR);

      ckks->evaluator.multiply_inplace_reduced_error(diagL, rotL, *(ckks->relin_keys));
      ckks->evaluator.multiply_inplace_reduced_error(diagR, rotR, *(ckks->relin_keys));
      ckks->evaluator.add(diagL, diagR, vec);
      ckks->evaluator.rescale_to_next_inplace(vec);

      if (bs == 0)
        sumbs = vec;
      else {
        ckks->evaluator.add_inplace(sumbs, vec);
      }
    }
    if (gs == 0)
      res = sumbs;
    else {
      ckks->evaluator.rotate_vector_inplace(sumbs, 128*8*gs, *(ckks->galois_keys));
      ckks->evaluator.add_inplace(res, sumbs);
    }
  }
}