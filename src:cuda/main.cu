// main.cu
#include <iostream>
#include <fstream>
#include <string>

#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <algorithm>

#define WORDLEN 3
#define PARTITION_SIZE 16
#define SIGNATURE_LEN 64
#define DENSITY 21

//#define DEBUG // Uncomment to enable debug output

#ifdef DEBUG
int debug_counter = 0;
#endif

#define SIGNING_KERNEL_THREADS(line_length) (line_length - WORDLEN + 1) + ((int)(PARTITION_SIZE / 2) - WORDLEN + 1) * (int)std::ceil(2.0f * (std::max(line_length, PARTITION_SIZE) - PARTITION_SIZE) / PARTITION_SIZE)
#define REDUCING_KERNEL_THREADS(line_length) (int)std::ceil((double)(SIGNING_KERNEL_THREADS(line_length)) / (double)(PARTITION_SIZE - WORDLEN + 1)) * 8

__device__ __forceinline__ uint32_t pcg_random(uint32_t input) {
    uint32_t pcg_state = input * 747796405u + 2891336453u;
    uint32_t pcg_word = ((pcg_state >> ((pcg_state >> 28u) + 4u)) ^ pcg_state) * 277803737u;
    return (pcg_word >> 22u) ^ pcg_word;
}

__global__ void signingKernel(const char* input, signed char* signatures, int numThreads) {
    // Each thread processes one word
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numThreads) {
        return;
    }

    // Calculate the starting position of the word in the input and output signature
    //int input_offset = (int)(idx / (PARTITION_SIZE - (WORDLEN - 1))) * WORDLEN + idx;
    int input_offset = idx + (-PARTITION_SIZE + (WORDLEN - 1) + (int)(PARTITION_SIZE / 2)) * (int)(idx / (PARTITION_SIZE - (WORDLEN - 1)));
    int output_offset = idx * SIGNATURE_LEN;

    // Initialise seed from the word
    uint32_t seed = static_cast<uint32_t>(input[input_offset]);
    for (int i = 1; i < WORDLEN; i++) {
        seed = (seed << 8) | static_cast<uint32_t>(input[input_offset + i]);
    }
    
    int non_zero = SIGNATURE_LEN * DENSITY / 100;
    int positive = 0;
    while (positive < non_zero/2)
    {
        uint32_t hash = pcg_random(seed);
        seed ^= hash; // Update seed for next iteration
        short pos = hash % SIGNATURE_LEN;
        if (signatures[output_offset + pos] == 0) 
	    {
            signatures[output_offset + pos] = 1;
            positive++;
        }
    }

    int negative = 0;
    while (negative < non_zero/2)
    {
        uint32_t hash = pcg_random(seed);
        seed ^= hash; // Update seed for next iteration
        short pos = hash % SIGNATURE_LEN;
        if (signatures[output_offset + pos] == 0) 
	    {
            signatures[output_offset + pos] = -1;
            negative++;
        }
    }
}

__global__ void reducingKernel(signed char* signatures, uint8_t* finalOutput, int numThreads, int remainingSignatures) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numThreads) {
        return;
    }
    
    // Calculate the starting position of the input signature
    int partitionOffset = (int)(idx / 8) * (PARTITION_SIZE - WORDLEN + 1) * SIGNATURE_LEN;
    int signatureOffset = (idx * 8) % SIGNATURE_LEN;
    int signaturesToProcess = (idx == numThreads - 1 && remainingSignatures != 0) ? remainingSignatures : (PARTITION_SIZE - WORDLEN + 1);
    
    for (int i = 1; i < signaturesToProcess; i++) {
        for (int j = 0; j < 8; j++) {
            signatures[partitionOffset + signatureOffset + j] += signatures[partitionOffset + signatureOffset + j + i * SIGNATURE_LEN];
        }
    }

    uint8_t c = 0;

    for (int i = 0; i < 8; i++) {
        c |= (signatures[(partitionOffset + signatureOffset + i)] > 0) << (7-i);
    }

    finalOutput[idx] = c;
}

void launchKernels(const char* line_data, const int lineLength, uint8_t* output_signatures) {
    int numThreads_signingKernel = SIGNING_KERNEL_THREADS(lineLength);
    int numThreads_reducingKernel = REDUCING_KERNEL_THREADS(lineLength);
    int remainingSignatures_lastReducingKernel = (SIGNING_KERNEL_THREADS(lineLength)) % (PARTITION_SIZE - WORDLEN + 1);

    // Allocate memory on GPU
    char* gpu_input;
    signed char* gpu_signatures;
    uint8_t* gpu_output;

    cudaMalloc((void**)&gpu_input, lineLength);
    cudaMalloc((void**)&gpu_signatures, (numThreads_signingKernel) * SIGNATURE_LEN);
    cudaMalloc((void**)&gpu_output, (numThreads_reducingKernel));

    // Copy input data from CPU to GPU
    cudaMemcpy(gpu_input, line_data, lineLength, cudaMemcpyHostToDevice);
    cudaMemset(gpu_signatures, 0, (numThreads_signingKernel) * SIGNATURE_LEN);
    cudaMemset(gpu_output, 0, (numThreads_reducingKernel));

    // Launch kernels
    dim3 blockSize(256);
    dim3 gridSize_signingKernel(numThreads_signingKernel);
    dim3 gridSize_reducingKernel(numThreads_reducingKernel);

#ifdef DEBUG
    printf("Launching signingKernel with %d threads\n", numThreads_signingKernel);
#endif

    signingKernel<<<gridSize_signingKernel, blockSize>>>(gpu_input, gpu_signatures, numThreads_signingKernel);

#ifdef DEBUG
    cudaDeviceSynchronize();

    signed char* debug_signatures = (signed char*)malloc(numThreads_signingKernel * SIGNATURE_LEN * sizeof(signed char));
    cudaMemcpy(debug_signatures, gpu_signatures, (numThreads_signingKernel) * SIGNATURE_LEN, cudaMemcpyDeviceToHost);
    
    int pos = 0;
    int neg = 0;
    int part = 0;
    for (int i = 0; i < numThreads_signingKernel * SIGNATURE_LEN; i++) {
        if (debug_signatures[i] > 0) {
            printf("+ ");
            pos++;
        } else if (debug_signatures[i] < 0) {
            printf("- ");
            neg++;
        } else {
            printf("0 ");
        }
        if ((i + 1) % SIGNATURE_LEN == 0) {
            printf("pos: %d, neg: %d, sig: %d\n", pos, neg, debug_counter++);
            pos = 0;
            neg = 0;
        }
        if ((i + 1) % (SIGNATURE_LEN * (PARTITION_SIZE - WORDLEN + 1)) == 0) {
            printf("End of partition %d\n\n", part++);
        }
    }
    printf("\n");
    free(debug_signatures);

    printf("Launching reducingKernel with %d threads and %d remaining signatures\n\n", numThreads_reducingKernel, remainingSignatures_lastReducingKernel);
#endif

    reducingKernel<<<gridSize_reducingKernel, blockSize>>>(gpu_signatures, gpu_output, numThreads_reducingKernel, remainingSignatures_lastReducingKernel);

    cudaMemcpy(output_signatures, gpu_output, (numThreads_reducingKernel), cudaMemcpyDeviceToHost);
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    // Wait for all kernels to finish execution
    cudaDeviceSynchronize();

    // Clean up GPU memory
    cudaFree(gpu_input);
    cudaFree(gpu_signatures);
    cudaFree(gpu_output);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <filename>\n", argv[0]);
        return 1;
    }

    const char* filename = argv[1];

    auto start = std::chrono::high_resolution_clock::now();

    char out_filename[256];
    snprintf(out_filename, sizeof(out_filename), "%s.part%d_sigs%02d_%d_cuda", filename, PARTITION_SIZE, WORDLEN, SIGNATURE_LEN);

    FILE* file = fopen64(filename, "r");
    FILE* sig_file = fopen64(out_filename, "w");

    if (file == NULL)
    {
        fprintf(stderr, "Error: failed to open file %s\n", filename);
        return 1;
    }

    char line_data[10000];
    int line_length = 0;
    int total_size = 0;
    while (!feof(file))
    {
        fgets(line_data, 10000, file); // skip meta data line
        fgets(line_data, 10000, file);
        line_length = (int)strlen(line_data) - 1;
        
        line_data[line_length] = 0; // terminate string

        size_t numUint8_ts = REDUCING_KERNEL_THREADS(line_length);
        total_size += numUint8_ts;
        uint8_t* output_signatures = (uint8_t*)malloc(numUint8_ts * sizeof(uint8_t));
        if (output_signatures == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            return 1;
        }

        memset(output_signatures, 0, numUint8_ts);

        #ifdef DEBUG
        printf("Processing line of length %d\n", line_length);
        #endif

        launchKernels(line_data, line_length, output_signatures);
        //fwrite(output_signatures, sizeof(unsigned char), (REDUCING_KERNEL_THREADS(line_length) * (SIGNATURE_LEN / 8)), sig_file);
        for (int i = 0; i < numUint8_ts; i++) {
            fprintf(sig_file, "%02x", output_signatures[i]);
            if ((i + 1) % 8 == 0) {
                fprintf(sig_file, "\n");
            }
        }
        fprintf(sig_file, "\n");

        free(output_signatures);
    }
    fclose(file);
    fclose(sig_file);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

#ifdef DEBUG
    printf("Total signatures size: %d uint8_ts\n", total_size);
#endif
    printf("%s %f seconds\n", filename, duration.count());

    return 0;
}