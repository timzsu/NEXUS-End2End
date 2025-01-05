#include "nn/row_pack.h"

namespace nexus {

using namespace std;

// Row pack for 128x768 matrix
FlatVecArray row_pack_128x768(torch::Tensor matrix) {
    auto splitted_matrix = torch::split(matrix, 128, 1);
    return FlatVecArray{
        flatten_pack(splitted_matrix[0], splitted_matrix[1]), 
        flatten_pack(splitted_matrix[2], splitted_matrix[3]), 
        flatten_pack(splitted_matrix[4], splitted_matrix[5]), 
    };
}

// Row pack for 768x64x2 matrices
FlatVecArray row_pack_768x64x2(torch::Tensor matrix1, torch::Tensor matrix2) {
    std::array<torch::Tensor, 6> mat_res;

    for (int i = 0; i < 3; ++i) {
        mat_res[i] = torch::zeros({2*128, 128}, torch::kDouble);
        mat_res[i+3] = torch::zeros({2*128, 128}, torch::kDouble);

        mat_res[i].slice(1, 0, 64) = torch::concat({
            matrix1.slice(0, i*256, i*256+128), 
            matrix2.slice(0, i*256+128, i*256+256)});
        mat_res[i+3].slice(1, 0, 64) = torch::concat({
            matrix2.slice(0, i*256, i*256+128), 
            matrix1.slice(0, i*256+128, i*256+256)});
    }

    FlatVecArray result(6);
    for (int i = 0; i < 6; i++) {
        result[i] = vector_from_tensor(mat_res[i]);
    }
    return result;
}

// Row pack for 768x768 matrix
FlatVecMat row_pack_768x768(torch::Tensor matrix) {
    vector<vector<torch::Tensor>> mat_res(3, vector<torch::Tensor>(6));
    FlatVecMat results(3);

    for (int m = 0; m < 3; ++m) {
        torch::Tensor matrix1 = matrix.slice(1, m*256, m*256+128);
        torch::Tensor matrix2 = matrix.slice(1, m*256+128, m*256+256);
        for (int i = 0; i < 3; ++i) {
            mat_res[m][i] = torch::concat({
                matrix1.slice(0, i*256, i*256+128), 
                matrix2.slice(0, i*256+128, i*256+256)});
            mat_res[m][i+3] = torch::concat({
                matrix2.slice(0, i*256, i*256+128), 
                matrix1.slice(0, i*256+128, i*256+256)});
        }
        results[m].resize(6);
        for (int i = 0; i < 6; i++) {
            results[m][i] = vector_from_tensor(mat_res[m][i]);
        }
    }

    return results;
}

// Row pack for 768x128 matrix
FlatVecArray row_pack_768x128(torch::Tensor matrix) {
    auto splitted_matrix = torch::split(matrix, 128, 0);
    return FlatVecArray{
        flatten_pack(splitted_matrix[0], splitted_matrix[1]), 
        flatten_pack(splitted_matrix[2], splitted_matrix[3]), 
        flatten_pack(splitted_matrix[4], splitted_matrix[5]), 
    };
}

// Row pack for 128x1 vector
FlatVec row_pack_128x1(torch::Tensor vector) {
    return vector_from_tensor(torch::tile(vector, {2*128, 1}));
}

// Row pack for 64x1x2 vectors
FlatVec row_pack_64x1x2(torch::Tensor vector1, torch::Tensor vector2) {
    torch::Tensor buf = torch::zeros({2*128, 128}, torch::kDouble);
    buf.slice(1, 0, 64) = torch::concat({
        torch::tile(vector1.unsqueeze(0), {128, 1}), 
        torch::tile(vector2.unsqueeze(0), {128, 1})});
    return vector_from_tensor(buf);
}

// Row pack for 768x1 vector
FlatVecArray row_pack_768x1(torch::Tensor vector) {
    std::vector<torch::Tensor> mm_res(3);
    for (int m = 0; m < 3; m++) {
        torch::Tensor vector1 = vector.slice(0, m * 256, m * 256 + 128);
        torch::Tensor vector2 = vector.slice(0, m * 256 + 128, m * 256 + 256);
        mm_res[m] = torch::concat({
            torch::tile(vector1.unsqueeze(0), {128, 1}),
            torch::tile(vector2.unsqueeze(0), {128, 1})});
    }
    FlatVecArray result(3);
    for (int i = 0; i < 3; i++) {
        result[i] = vector_from_tensor(mm_res[i]);
    }
    return result;
}

} // namespace nexus