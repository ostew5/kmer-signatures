FROM gcc:latest

WORKDIR /app
COPY . ./src/

RUN g++ -O2 -std=gnu++17 -o kmer-signatures src/*.cpp

CMD ["sh", "./data/load_test.sh"]