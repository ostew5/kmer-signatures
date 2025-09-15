FROM nvidia/cuda:12.4.1-devel-ubuntu22.04
  
WORKDIR /app
COPY . ./src/
  
RUN apt-get update && apt-get install -y build-essential cmake git libnuma-dev

RUN nvcc -std=c++17 -o kmer-signatures src/*.cpp

ENTRYPOINT ["./kmer-signatures"]
CMD ["./data/qut3.fasta"]