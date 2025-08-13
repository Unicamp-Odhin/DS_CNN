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

// Fully connected: input length = in_dim, produces one Q15 output
int16_t fully_connected_q15(const int16_t* input, const int16_t* weights,
                            int32_t bias, int in_dim) {
    int32_t acc = bias;
    for (int i = 0; i < in_dim; ++i) {
        acc += ((int32_t)input[i] * weights[i]) >> 15;
    }
    if (acc > 32767) acc = 32767;
    if (acc < -32768) acc = -32768;
    return (int16_t)acc;
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

// Generic 2D convolution that handles both depthwise and pointwise cases
void conv2d_q15(const int16_t* input, int16_t* output,
               int Cin, int Cout, int H, int W,
               const int16_t* weights, const int32_t* biases,
               int kernel_size) {
    int spatial = H * W;
    int groups = (Cout == Cin && kernel_size > 1) ? Cin : 1; // depthwise if Cout==Cin and kernel>1
    int filters_per_group = Cout / groups;
    
    // For each output channel
    for (int co = 0; co < Cout; ++co) {
        // Determine which input channels to use for this output channel
        int group_idx = co / filters_per_group;
        int cin_start = group_idx * (Cin / groups);
        int cin_end = cin_start + (Cin / groups);
        
        // For each spatial location in the output
        for (int i = 0; i < H; ++i) {
            for (int j = 0; j < W; ++j) {
                int32_t acc = biases[co];
                
                // For each input channel in this group
                for (int ci = cin_start; ci < cin_end; ++ci) {
                    // For each kernel position
                    for (int ki = 0; ki < kernel_size; ++ki) {
                        int ii = i + ki - (kernel_size > 1 ? 1 : 0);
                        if (ii < 0 || ii >= H) continue;
                        
                        for (int kj = 0; kj < kernel_size; ++kj) {
                            int jj = j + kj - (kernel_size > 1 ? 1 : 0);
                            if (jj < 0 || jj >= W) continue;
                            
                            // Calculate input and weight indices
                            int in_idx = ci * spatial + ii * W + jj;
                            int w_idx;
                            if (kernel_size > 1) {
                                // Depthwise: each output channel uses its own kernel
                                w_idx = co * kernel_size * kernel_size + ki * kernel_size + kj;
                            } else {
                                // Pointwise: weights are arranged as [Cout][Cin]
                                w_idx = co * Cin + ci;
                            }
                            
                            acc += ((int32_t)input[in_idx] * weights[w_idx]) >> 15;
                        }
                    }
                }
                
                // Clamp and store result
                if (acc > 32767) acc = 32767;
                if (acc < -32768) acc = -32768;
                output[co * spatial + i * W + j] = (int16_t)acc;
            }
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
        int32_t bias_arr = net_0_bias[oc];
        conv2d_q15(in_q15, &conv_out[oc * H * W], 1, 1, H, W, ker, &bias_arr, KERNEL_SZ);
    }
    relu_q15(conv_out, 32 * H * W);

    // 3) DS Block1: Depthwise (32->32)
    static int16_t dw1_out[32 * N_FRAMES * N_CEPS];
    conv2d_q15(conv_out, dw1_out, 32, 32, H, W,
               net_2_block_0_weight, net_2_block_0_bias, KERNEL_SZ);
    
    // Pointwise (32->64)
    static int16_t pw1_out[64 * N_FRAMES * N_CEPS];
    conv2d_q15(dw1_out, pw1_out, 32, 64, H, W,
               net_2_block_1_weight, net_2_block_1_bias, 1);
    relu_q15(pw1_out, 64 * H * W);

    // 4) DS Block2: Depthwise (64->64)
    static int16_t dw2_out[64 * N_FRAMES * N_CEPS];
    conv2d_q15(pw1_out, dw2_out, 64, 64, H, W,
               net_3_block_0_weight, net_3_block_0_bias, KERNEL_SZ);
    
    // Pointwise (64->64)
    static int16_t pw2_out[64 * N_FRAMES * N_CEPS];
    conv2d_q15(dw2_out, pw2_out, 64, 64, H, W,
               net_3_block_1_weight, net_3_block_1_bias, 1);
    relu_q15(pw2_out, 64 * H * W);

    // 5) Pooling
    static int16_t pool_out[64];
    avgpool_q15(pw2_out, pool_out, 64, H, W);

    // 6) Fully Connected -> logit
    int32_t custom_bias = net_6_bias[0] + 28000; // Custom bias adjustment
    int16_t logit_q15 = fully_connected_q15(
        pool_out, net_6_weight, custom_bias, 64
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
