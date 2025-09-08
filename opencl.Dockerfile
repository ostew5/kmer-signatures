# Dockerfile
FROM debian:bookworm

# Headers + loader + Mesa OpenCL (Rusticl) + clinfo
RUN apt-get update && apt-get install -y \
    g++ build-essential \
    opencl-headers ocl-icd-libopencl1 clinfo \
    mesa-opencl-icd mesa-vulkan-drivers libdrm2 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/kmer-signatures

# Copy sources correctly (two separate dirs as examples)
# If you only have ./src, keep the first line and remove the second.
COPY src/     ./src/

# Compile (include both src trees if you have both)
RUN g++ -O2 -std=gnu++17 -o kmer-signatures src/*.cpp -lOpenCL

CMD ["./kmer-signatures"]
