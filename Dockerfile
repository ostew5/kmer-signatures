FROM gcc:latest

WORKDIR /usr/src/kmer-signatures
COPY src/ ./src/

RUN g++ -O2 -std=gnu++17 -o kmer-signatures src/*.cpp

ENTRYPOINT ["./kmer-signatures"]
CMD ["./data/qut3.fasta"]
