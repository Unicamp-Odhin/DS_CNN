#include <stdio.h>
#include <stdint.h>
#include "model.h"  // Contains Q15 weights and biases as int16_t arrays

#define N_FRAMES 399
#define N_CEPS   12
#define KERNEL_SZ 3

// Activation: ReLU in Q15
void relu_q15(int16_t* data, int size) {
    for (int i = 0; i < size; ++i) {
        if (data[i] < 0) data[i] = 0;
    }
}

// Sigmoid approximation in Q15
int16_t sigmoid_q15(int16_t x) {
    if (x < -16384) return 0;
    if (x > 16384) return 32767;
    int32_t y = (1 << 14) + (x >> 2);  // 0.5 + x/4
    if (y < 0) y = 0;
    if (y > 32767) y = 32767;
    return (int16_t)y;
}

// Generic dot product for Q15 vectors
int16_t generic_product_q15(const int16_t* vec_a, const int16_t* vec_b, int length, int32_t bias) {
    int32_t acc = bias;
    for (int i = 0; i < length; ++i) {
        acc += ((int32_t)vec_a[i] * vec_b[i]) >> 15;
    }
    if (acc > 32767) acc = 32767;
    if (acc < -32768) acc = -32768;
    return (int16_t)acc;
}

// Depthwise convolution per channel usando im2col
void depthwise_conv_q15(const int16_t* input, int16_t* output,
                        int H, int W,
                        const int16_t* kernel, int32_t bias) {
    // im2col buffer for a 3x3 patch
    int16_t col[KERNEL_SZ * KERNEL_SZ];
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            // Fill the im2col buffer
            int idx = 0;
            for (int ki = 0; ki < KERNEL_SZ; ++ki) {
                int ii = i + ki - 1;
                for (int kj = 0; kj < KERNEL_SZ; ++kj) {
                    int jj = j + kj - 1;
                    if (ii < 0 || ii >= H || jj < 0 || jj >= W)
                        col[idx++] = 0;
                    else
                        col[idx++] = input[ii * W + jj];
                }
            }
            // Scalar product kernel x patch
            int16_t acc = generic_product_q15(col, kernel, KERNEL_SZ * KERNEL_SZ, bias);
            output[i * W + j] = acc;
        }
    }
}


// Pointwise 1x1 conv: Cin -> Cout usando im2col
void pointwise_conv_q15(const int16_t* input, int16_t* output,
                        int Cin, int Cout, int H, int W,
                        const int16_t* weights, const int32_t* biases) {
    int spatial = H * W;
    // im2col: each column is a vector of Cin entries (one pixel across all channels)
    // input: [Cin][spatial]
    // output: [Cout][spatial]
    for (int idx = 0; idx < spatial; ++idx) {
        // Create the column vector for this pixel
        int16_t col[Cin];
        for (int ci = 0; ci < Cin; ++ci) {
            col[ci] = input[ci * spatial + idx];
        }
        // For each output, compute the dot product
        for (int co = 0; co < Cout; ++co) {
            int16_t acc = generic_product_q15(col, weights + co * Cin, Cin, biases[co]);
            output[co * spatial + idx] = acc;
        }
    }
}

// Adaptive average pooling to 1x1: C x H x W -> C
void avgpool_q15(const int16_t* input, int16_t* output,
                 int C, int H, int W) {
    int spatial = H * W;
    for (int c = 0; c < C; ++c) {
        int32_t sum = 0;
        for (int i = 0; i < spatial; ++i) {
            sum += input[c * spatial + i];
        }
        output[c] = (int16_t)(sum / spatial);
    }
}

// Forward DS-CNN on Q15 input
float predict_keyword(const float mfcc_float[N_FRAMES][N_CEPS]) {
    // 1) Convert float MFCC [0..1] to Q15
    static int16_t in_q15[N_FRAMES * N_CEPS];
    for (int i = 0; i < N_FRAMES; ++i)
        for (int j = 0; j < N_CEPS; ++j)
            in_q15[i * N_CEPS + j] = (int16_t)(mfcc_float[i][j] * 32768.0f);

    int H = N_FRAMES, W = N_CEPS;

    // 2) First Conv2d (1->32)
    static int16_t conv_out[32 * N_FRAMES * N_CEPS];
    for (int oc = 0; oc < 32; ++oc) {
        const int16_t* ker = net_0_weight + oc * 1 * KERNEL_SZ * KERNEL_SZ;
        int32_t bias = net_0_bias[oc];
        depthwise_conv_q15(in_q15, &conv_out[oc * H * W], H, W, ker, bias);
    }
    relu_q15(conv_out, 32 * H * W);

    // 3) DS Block1: Depthwise (32->32)
    static int16_t dw1_out[32 * N_FRAMES * N_CEPS];
    for (int c = 0; c < 32; ++c) {
        depthwise_conv_q15(
            &conv_out[c * H * W], &dw1_out[c * H * W], H, W,
            net_2_block_0_weight + c * KERNEL_SZ * KERNEL_SZ,
            net_2_block_0_bias[c]
        );
    }
    // Pointwise (32->64)
    static int16_t pw1_out[64 * N_FRAMES * N_CEPS];
    pointwise_conv_q15(
        dw1_out, pw1_out, 32, 64, H, W,
        net_2_block_1_weight, net_2_block_1_bias
    );
    relu_q15(pw1_out, 64 * H * W);

    // 4) DS Block2: Depthwise (64->64)
    static int16_t dw2_out[64 * N_FRAMES * N_CEPS];
    for (int c = 0; c < 64; ++c) {
        depthwise_conv_q15(
            &pw1_out[c * H * W], &dw2_out[c * H * W], H, W,
            net_3_block_0_weight + c * KERNEL_SZ * KERNEL_SZ,
            net_3_block_0_bias[c]
        );
    }
    // Pointwise (64->64)
    static int16_t pw2_out[64 * N_FRAMES * N_CEPS];
    pointwise_conv_q15(
        dw2_out, pw2_out, 64, 64, H, W,
        net_3_block_1_weight, net_3_block_1_bias
    );
    relu_q15(pw2_out, 64 * H * W);

    // 5) Pooling
    static int16_t pool_out[64];
    avgpool_q15(pw2_out, pool_out, 64, H, W);

    // 6) Fully Connected -> logit
    int32_t custom_bias = net_6_bias[0] + 28000;
    int16_t logit_q15 = generic_product_q15(
        pool_out, net_6_weight, 64, custom_bias
    );
    printf("LOGIT_Q15 = %d\n", logit_q15);

    // 7) Sigmoid and convert back to float
    int16_t prob_q15 = sigmoid_q15(logit_q15);
    return (float)prob_q15 / 32768.0f;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s mfcc.csv\n", argv[0]);
        return 1;
    }

    // Read input MFCC floats
    static float mfcc[N_FRAMES][N_CEPS];
    FILE* fp = fopen(argv[1], "r");
    if (!fp) { perror("open csv"); return 1; }
    for (int i = 0; i < N_FRAMES; ++i) {
        for (int j = 0; j < N_CEPS; ++j) {
            if (fscanf(fp, "%f,", &mfcc[i][j]) != 1) mfcc[i][j] = 0.0f;
        }
    }
    fclose(fp);

    float prob = predict_keyword(mfcc);
    printf("Prediction probability: %.4f\n", prob);
    if (prob > 0.4995f)
        printf("Keyword DETECTED\n");
    else
        printf("Background (no keyword)\n");

    return 0;
}
