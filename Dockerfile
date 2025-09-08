FROM gcc:latest

WORKDIR /usr/src/kmer-signatures
COPY . .

RUN apt-get update && apt-get install -y libsafec-dev && rm -rf /var/lib/apt/lists/*

RUN g++ -O2 -std=gnu++17 -D__STDC_WANT_LIB_EXT1__=1 -o kmer-signatures src/*.cpp -lsafec

CMD ["./kmer-signatures"]