FROM nvidia/cuda:12.4.1-devel-ubuntu22.04
RUN apt-get update && apt-get install -y build-essential cmake git libnuma-dev
  
WORKDIR /app
COPY . ./src/

RUN nvcc -std=c++17 -o kmer-signatures src/*.cu

ENTRYPOINT ["./kmer-signatures"]
CMD ["./data/qut3.fasta"]