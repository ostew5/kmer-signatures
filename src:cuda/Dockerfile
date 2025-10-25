FROM nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04
RUN apt-get update && apt-get install -y build-essential cmake git libnuma-dev
  
WORKDIR /app
COPY . ./src/

RUN nvcc -std=c++17 -o kmer-signatures src/*.cu

CMD ["sh", "./data/load_test.sh"]