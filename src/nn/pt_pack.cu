#include "nn/pt_pack.cuh"

namespace nexus {

using namespace std;

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

PackedPt pt_pack(FlatVec& pt, shared_ptr<CKKSEvaluator> ckks) {
  const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream;
  const auto &stream = stream_wrapper.get_stream();
  PackedPt cleartexts;
  double *d_pt, *tmp, *out_buf;
  
  cudaMalloc(&d_pt, slot_count*sizeof(double));
  cudaMalloc(&tmp, 128*slot_count*sizeof(double));
  cudaMalloc(&out_buf, 256*slot_count*sizeof(double));
  cudaMemcpy(d_pt, pt.data(), slot_count*sizeof(double), cudaMemcpyHostToDevice);
  kernel_pt_encoding_128x128<<<128, 128>>>(d_pt, tmp, out_buf);
  for (int i=0; i<256; i++) {
    cleartexts[i].resize(slot_count);
    cudaMemcpy(cleartexts[i].data(), out_buf+i*slot_count, slot_count*sizeof(double), cudaMemcpyDeviceToHost);
  }
  cudaFreeAsync(d_pt, stream);
  cudaFreeAsync(tmp, stream);
  cudaFreeAsync(out_buf, stream);

  for (int gs=-128; gs<128; gs+=16)
    for (int bs=0; bs<16; bs++) {
      cleartexts[bs+gs+128] = rotate(cleartexts[bs+gs+128], -gs);
    }

  return cleartexts;
}

} // namespace nexus