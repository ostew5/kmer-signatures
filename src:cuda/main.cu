// main.cu
#include <iostream>
#include <fstream>
#include <string>
#include <cctype>

#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <vector>

#define WORDLEN 3
#define PARTITION_SIZE 16
#define SIGNATURE_LEN 64
#define DENSITY 21
#define BATCH_SIZE 2048

#define DEBUG // Uncomment to enable debug output 

#ifdef DEBUG
int debug_counter = 0;
#endif

#define SIGNING_KERNEL_THREADS(line_length) (line_length - WORDLEN + 1) + ((int)(PARTITION_SIZE / 2) - WORDLEN + 1) * (int)std::ceil(2.0f * (std::max(line_length, PARTITION_SIZE) - PARTITION_SIZE) / PARTITION_SIZE)
#define REDUCING_KERNEL_THREADS(line_length) (int)std::ceil((double)(SIGNING_KERNEL_THREADS(line_length)) / (double)(PARTITION_SIZE - WORDLEN + 1)) * 8

__device__ __forceinline__ uint32_t pcg_random(uint32_t input)
{
        uint32_t pcg_state = input * 747796405u + 2891336453u;
        uint32_t pcg_word = ((pcg_state >> ((pcg_state >> 28u) + 4u)) ^ pcg_state) * 277803737u;
        return (pcg_word >> 22u) ^ pcg_word;
}

__global__ void signingKernel(const char *input, signed char *signatures, int numThreads)
{
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= numThreads)
        {
                return;
        }

        int input_offset = idx + (-PARTITION_SIZE + (WORDLEN - 1) + (int)(PARTITION_SIZE / 2)) * (int)(idx / (PARTITION_SIZE - (WORDLEN - 1)));
        int output_offset = idx * SIGNATURE_LEN;

        uint32_t seed = static_cast<uint32_t>(input[input_offset]);
        for (int i = 1; i < WORDLEN; i++)
        {
                seed = (seed << 8) | static_cast<uint32_t>(input[input_offset + i]);
        }

        int non_zero = SIGNATURE_LEN * DENSITY / 100;
        int positive = 0;
        while (positive < non_zero / 2)
        {
                uint32_t hash = pcg_random(seed);
                seed ^= hash;
                short pos = hash % SIGNATURE_LEN;
                if (signatures[output_offset + pos] == 0)
                {
                        signatures[output_offset + pos] = 1;
                        positive++;
                }
        }

        int negative = 0;
        while (negative < non_zero / 2)
        {
                uint32_t hash = pcg_random(seed);
                seed ^= hash;
                short pos = hash % SIGNATURE_LEN;
                if (signatures[output_offset + pos] == 0)
                {
                        signatures[output_offset + pos] = -1;
                        negative++;
                }
        }
}

__global__ void reducingKernel(signed char *signatures, uint8_t *finalOutput, int numThreads, int remainingSignatures)
{
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= numThreads)
        {
                return;
        }

        int partitionOffset = (int)(idx / 8) * (PARTITION_SIZE - WORDLEN + 1) * SIGNATURE_LEN;
        int signatureOffset = (idx * 8) % SIGNATURE_LEN;
        int signaturesToProcess = (idx == numThreads - 1 && remainingSignatures != 0) ? remainingSignatures : (PARTITION_SIZE - WORDLEN + 1);

        for (int i = 1; i < signaturesToProcess; i++)
        {
                for (int j = 0; j < 8; j++)
                {
                        signatures[partitionOffset + signatureOffset + j] += signatures[partitionOffset + signatureOffset + j + i * SIGNATURE_LEN];
                }
        }

        uint8_t c = 0;

        for (int i = 0; i < 8; i++)
        {
                c |= (signatures[(partitionOffset + signatureOffset + i)] > 0) << (7 - i);
        }

        finalOutput[idx] = c;
}

struct PinnedFastaPointer {
        const char* data;
        int length;
};

std::pair<std::vector<PinnedFastaPointer>, size_t> parseFastaPointers(const char *buffer, size_t size)
{
        std::vector<PinnedFastaPointer> pointers;
        size_t maxLength = 0;
        const char *ptr = buffer;
        const char *end = buffer + size;

        while (ptr < end)
        {
                do ptr++; while (ptr < end && *ptr != '\n' && *ptr != '\r');

                if (ptr >= end) break;

                while(ptr < end && (*ptr == '\n' || *ptr == '\r')) ptr++;

                if (ptr >= end) break;

                const char *seq_start = ptr;
                int len = 0;

                do {
                        ptr++;
                        len++;
                } while (ptr < end && *ptr != '\n' && *ptr != '\r');

                while(ptr < end && (*ptr == '\n' || *ptr == '\r')) ptr++;

                if (len > 0)
                {
                        PinnedFastaPointer p;
                        p.data = seq_start;
                        p.length = len;
                        if (len > maxLength) maxLength = len;
                        pointers.push_back(p);
                }
        }

        return {pointers, maxLength};
}

int main(int argc, char **argv)
{
        if (argc < 2)
        {
                fprintf(stderr, "Usage: %s <filename>\n", argv[0]);
                return 1;
        }

        const char *filename = argv[1];

        char out_filename[256];
        snprintf(out_filename, sizeof(out_filename), "%s.part%d_sigs%02d_%d_cuda", filename, PARTITION_SIZE, WORDLEN, SIGNATURE_LEN);

        FILE *file = fopen64(filename, "rb");
        if (!file)
        {
                fprintf(stderr, "Error: failed to open file %s\n", filename);
                return 1;
        }
        
        fseek(file, 0, SEEK_END);
        size_t file_size = ftell(file);
        fseek(file, 0, SEEK_SET);

        cudaError_t err = cudaSetDevice(0);
        if (err != cudaSuccess)
        {
                fprintf(stderr, "Error: failed to set CUDA device: %s\n", cudaGetErrorString(err));
                fclose(file);
                return 1;
        }
        
        char *hostPinnedBuffer = nullptr;
        err = cudaMallocHost((void **)&hostPinnedBuffer, file_size);
        if (err != cudaSuccess)
        {
                fprintf(stderr, "Error: failed to allocate pinned memory: %s\n", cudaGetErrorString(err));
                return 1;
        }
        
        size_t read_size = fread(hostPinnedBuffer, 1, file_size, file);
        fclose(file);
        if (read_size != file_size)
        {
                fprintf(stderr, "Error: failed to read file %s\n", filename);
                cudaFreeHost(hostPinnedBuffer);
                return 1;
        }

        auto [fastaPointers, maxFastaLength] = parseFastaPointers(hostPinnedBuffer, file_size);

        size_t totalSequences = fastaPointers.size();
        if (totalSequences == 0)
        {
                fprintf(stderr, "Error: no sequences found in file %s\n", filename);
                cudaFreeHost(hostPinnedBuffer);
                return 1;
        }

        FILE *sig_file = fopen64(out_filename, "wb");
        if (!sig_file)
        {
                fprintf(stderr, "Error: failed to open output file %s\n", out_filename);
                cudaFreeHost(hostPinnedBuffer);
                return 1;
        }

        std::vector<cudaStream_t> cudaStreams(BATCH_SIZE);
        for (cudaStream_t &stream : cudaStreams)
        {
                err = cudaStreamCreate(&stream);
                if (err != cudaSuccess)
                {
                        fprintf(stderr, "Error: failed to create CUDA stream: %s\n", cudaGetErrorString(err));
                        fclose(sig_file);
                        cudaFreeHost(hostPinnedBuffer);
                        return 1;
                }
        }

        size_t worstCaseSigningBytes = (SIGNING_KERNEL_THREADS((int)maxFastaLength)) * BATCH_SIZE * SIGNATURE_LEN;
        size_t worstCaseReducingBytes = (REDUCING_KERNEL_THREADS((int)maxFastaLength)) * BATCH_SIZE;

        unsigned int totalFileSize = 0;
        for (const auto &p : fastaPointers)
        {
                totalFileSize += (REDUCING_KERNEL_THREADS(p.length));
        }

        uint8_t* resultSignatures = (uint8_t*)malloc(totalFileSize);
        if (resultSignatures == nullptr)
        {
                fprintf(stderr, "Error: failed to allocate memory for resultSignatures\n");
                fclose(sig_file);
                cudaFreeHost(hostPinnedBuffer);
                for (cudaStream_t &stream : cudaStreams) cudaStreamDestroy(stream);
                return 1;
        }
        size_t resultOffset = 0;
        
        signed char *signatures;
        err = cudaMalloc((void **)&signatures, worstCaseSigningBytes * sizeof(signed char));
        if (err != cudaSuccess)
        {
                fprintf(stderr, "Error: failed to allocate device memory for signatures: %s\n", cudaGetErrorString(err));
                fclose(sig_file);
                cudaFreeHost(hostPinnedBuffer);
                for (cudaStream_t &stream : cudaStreams) cudaStreamDestroy(stream);
                return 1;
        }
        
        uint8_t *gpuOutput;
        err = cudaMalloc((void **)&gpuOutput, worstCaseReducingBytes * sizeof(uint8_t));
        if (err != cudaSuccess)
        {
                fprintf(stderr, "Error: failed to allocate device memory for gpuOutput: %s\n", cudaGetErrorString(err));
                fclose(sig_file);
                cudaFreeHost(hostPinnedBuffer);
                cudaFree(signatures);
                for (cudaStream_t &stream : cudaStreams) cudaStreamDestroy(stream);
                return 1;
        }

        uint8_t *gpuOutput_copy = (uint8_t *)malloc(worstCaseReducingBytes);
        
        auto start = std::chrono::high_resolution_clock::now();

        size_t currentIndex = 0;
        while (currentIndex < totalSequences)
        {
                int actualBatchSize = (int)std::min((size_t)BATCH_SIZE, totalSequences - currentIndex);

                int totalReducingBytes = 0;
                
                for (int i = 0; i < actualBatchSize; i++)
                {
                        totalReducingBytes += REDUCING_KERNEL_THREADS(fastaPointers[currentIndex + i].length);
                }

                size_t signatures_offset = 0;
                size_t gpuOutput_offset = 0;
                for (int i = 0; i < actualBatchSize; i++)
                {
                        const int numThreads_signingKernel = SIGNING_KERNEL_THREADS(fastaPointers[currentIndex + i].length);
                        const int numThreads_reducingKernel = REDUCING_KERNEL_THREADS(fastaPointers[currentIndex + i].length);
                        int remainingSignatures_lastReducingKernel = (SIGNING_KERNEL_THREADS(fastaPointers[currentIndex + i].length)) % (PARTITION_SIZE - WORDLEN + 1);

                        if (signatures_offset + numThreads_signingKernel * SIGNATURE_LEN > worstCaseSigningBytes || gpuOutput_offset + numThreads_reducingKernel > worstCaseReducingBytes)
                        {
                                fprintf(stderr, "\nError: Exceeded preallocated GPU memory.\nAllocated %zu bytes for signatures and %zu bytes for gpuOutput\n", worstCaseSigningBytes, worstCaseReducingBytes);
                                fclose(sig_file);
                                cudaFreeHost(hostPinnedBuffer);
                                cudaFree(signatures);
                                cudaFree(gpuOutput);
                                for (cudaStream_t &stream : cudaStreams) cudaStreamDestroy(stream);
                                return 1;
                        }

                        cudaMemsetAsync(reinterpret_cast<void*>(signatures + signatures_offset), 0, numThreads_signingKernel * SIGNATURE_LEN, cudaStreams[i]);
                        err = cudaGetLastError();
                        if (err != cudaSuccess)
                        {
                                fprintf(stderr, "\nError in main loop at %d in cudaMemsetAsync for signatures: %s\n", i, cudaGetErrorString(err));
                                fclose(sig_file);
                                cudaFreeHost(hostPinnedBuffer);
                                cudaFree(signatures);
                                cudaFree(gpuOutput);
                                for (cudaStream_t &stream : cudaStreams) cudaStreamDestroy(stream);
                                return 1;
                        }

                        cudaMemsetAsync(reinterpret_cast<void*>(gpuOutput + gpuOutput_offset), 0, numThreads_reducingKernel, cudaStreams[i]);
                        err = cudaGetLastError();
                        if (err != cudaSuccess)
                        {
                                fprintf(stderr, "Error in main loop at %d in cudaMemsetAsync for gpuOutput: %s\n", i, cudaGetErrorString(err));
                                fclose(sig_file);
                                cudaFreeHost(hostPinnedBuffer);
                                cudaFree(signatures);
                                cudaFree(gpuOutput);
                                for (cudaStream_t &stream : cudaStreams) cudaStreamDestroy(stream);
                                return 1;
                        }

                        dim3 blockSize(256);
                        int blocksSigning = (numThreads_signingKernel + blockSize.x - 1) / blockSize.x;
                        int blocksReducing = (numThreads_reducingKernel + blockSize.x - 1) / blockSize.x;

                        dim3 gridSize_signingKernel((numThreads_signingKernel + blockSize.x - 1) / blockSize.x);
                        signingKernel<<<blocksSigning, blockSize, 0, cudaStreams[i]>>>(fastaPointers[currentIndex + i].data, signatures + signatures_offset, numThreads_signingKernel);
                        err = cudaGetLastError();
                        if (err != cudaSuccess)
                        {
                                fprintf(stderr, "Error in signingKernel: %s\n", cudaGetErrorString(err));
                                fclose(sig_file);
                                cudaFreeHost(hostPinnedBuffer);
                                cudaFree(signatures);
                                cudaFree(gpuOutput);
                                for (cudaStream_t &stream : cudaStreams) cudaStreamDestroy(stream);
                                return 1;
                        }

                        reducingKernel<<<blocksReducing, blockSize, 0, cudaStreams[i]>>>(signatures + signatures_offset, gpuOutput + gpuOutput_offset, numThreads_reducingKernel, remainingSignatures_lastReducingKernel);
                        err = cudaGetLastError();
                        if (err != cudaSuccess)
                        {
                                fprintf(stderr, "Error in reducingKernel: %s\n", cudaGetErrorString(err));
                                fclose(sig_file);
                                cudaFreeHost(hostPinnedBuffer);
                                cudaFree(signatures);
                                cudaFree(gpuOutput);
                                for (cudaStream_t &stream : cudaStreams) cudaStreamDestroy(stream);
                                return 1;
                        }

                        signatures_offset += numThreads_signingKernel * SIGNATURE_LEN;
                        gpuOutput_offset += numThreads_reducingKernel;
                }

                err = cudaDeviceSynchronize();
                if (err != cudaSuccess)
                {
                        fprintf(stderr, "Error in cudaStreamSynchronize: %s\n", cudaGetErrorString(err));
                        fclose(sig_file);
                        cudaFreeHost(hostPinnedBuffer);
                        cudaFree(signatures);
                        cudaFree(gpuOutput);
                        for (cudaStream_t &stream : cudaStreams) cudaStreamDestroy(stream);
                        return 1;
                }

                err = cudaMemcpy(resultSignatures + resultOffset, gpuOutput, totalReducingBytes, cudaMemcpyDeviceToHost);
                if (err != cudaSuccess)
                {
                        fprintf(stderr, "Error in cudaMemcpy: %s\n", cudaGetErrorString(err));
                        fclose(sig_file);
                        cudaFreeHost(hostPinnedBuffer);
                        cudaFree(signatures);
                        cudaFree(gpuOutput);
                        for (cudaStream_t &stream : cudaStreams) cudaStreamDestroy(stream);
                        return 1;
                }
                
                resultOffset += totalReducingBytes;
                
                currentIndex += actualBatchSize;
        }
        
        cudaFree(signatures);
        cudaFree(gpuOutput);

        for (cudaStream_t &stream : cudaStreams)
        {
                cudaStreamDestroy(stream);
        }
        
        cudaFreeHost(hostPinnedBuffer);

        size_t written = fwrite(resultSignatures, 1, resultOffset, sig_file);
        if (written != resultOffset)
        {
                fprintf(stderr, "Error: failed to write signatures to file %s\n", out_filename);
        }
        
        free(resultSignatures);
        
        fclose(sig_file);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;

        printf("%s %f seconds\n", filename, duration.count());

        return 0;
}