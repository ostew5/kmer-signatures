#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "uthash.h"
extern "C" {
    void seed_random(char* term, int length);
    short random_num(short max);
}

inline uint32_t pcg_random(uint32_t input) {
    uint32_t pcg_state = input * 747796405u + 2891336453u;
    uint32_t pcg_word = ((pcg_state >> ((pcg_state >> 28u) + 4u)) ^ pcg_state) * 277803737u;
    return (pcg_word >> 22u) ^ pcg_word;
}

#include <chrono>

//#define DEBUG // Uncomment to enable debug output

#ifdef DEBUG
int debug_counter = 0;
#endif

typedef unsigned char byte;

#define SIGNATURE_LEN 64

int DENSITY  = 21;
int PARTITION_SIZE;

int inverse[256];
const char* alphabet = "CSTPAGNDEQHRKMILVFYW";


void Init();

int doc_sig[SIGNATURE_LEN];

int WORDLEN;
FILE *sig_file;

typedef struct
{
    char term[100];
    short sig[SIGNATURE_LEN];
    UT_hash_handle hh;
} hash_term;

hash_term *vocab = NULL;


short* compute_new_term_sig(char* term, short *term_sig)
{
    uint32_t seed = static_cast<uint32_t>(term[0]);
    for (int i = 1; i < WORDLEN; i++) {
        seed = (seed << 8) | static_cast<uint32_t>(term[i]);
    }

    int non_zero = SIGNATURE_LEN * DENSITY/100;

    int positive = 0;
    while (positive < non_zero/2)
    {
        uint32_t hash = pcg_random(seed);
        seed ^= hash; // Update seed for next iteration
        short pos = hash % SIGNATURE_LEN;
        if (term_sig[pos] == 0) 
	{
            term_sig[pos] = 1;
            positive++;
        }
    }

    int negative = 0;
    while (negative < non_zero/2)
    {
        uint32_t hash = pcg_random(seed);
        seed ^= hash; // Update seed for next iteration
        short pos = hash % SIGNATURE_LEN;
        if (term_sig[pos] == 0) 
	{
            term_sig[pos] = -1;
            negative++;
        }
    }
    return term_sig;
}

short *find_sig(char* term)
{
    hash_term *entry;
    HASH_FIND(hh, vocab, term, WORDLEN, entry);
    if (entry == NULL)
    {
        entry = (hash_term*)malloc(sizeof(hash_term));
        strncpy(entry->term, term, sizeof(entry->term) - 1);
        entry->term[sizeof(entry->term) - 1] = '\0';
        memset(entry->sig, 0, sizeof(entry->sig));
        compute_new_term_sig(term, entry->sig);
        HASH_ADD(hh, vocab, term, WORDLEN, entry);
    }

    return entry->sig;
}


void signature_add(char* term)
{
	short* term_sig = find_sig(term);
#ifdef DEBUG
    int pos = 0, neg = 0;
#endif
	for (int i=0; i<SIGNATURE_LEN; i++){
#ifdef DEBUG
        if (term_sig[i] > 0) {
            printf("+ ");
            pos++;
        } else if (term_sig[i] < 0) {
            printf("- ");
            neg++;
        } else {
            printf("0 ");
        }
#endif
		doc_sig[i] += term_sig[i];
    }
#ifdef DEBUG
    printf("pos: %d, neg: %d, sig: %d\n", pos, neg, debug_counter++);
    pos = 0, neg = 0;
#endif
}

int doc = 0;

void compute_signature(char* sequence, int length)
{
    memset(doc_sig, 0, sizeof(doc_sig)); // reset doc_sig to all zeros

    for (int i=0; i<length-WORDLEN+1; i++)
        signature_add(sequence+i);

    if (length < WORDLEN)
        printf("Warning: sequence length %d is shorter than WORDLEN %d\n", length, WORDLEN);

    // save document number to sig file
    // document is the same as the .fasta line, doc is set at partition()
    //fwrite(&doc, sizeof(int), 1, sig_file);
    
    
    // flatten and output to sig file
    for (int i = 0; i < SIGNATURE_LEN; i += 8) 
    {
        byte c = 0;
        for (int j = 0; j < 8; j++) 
            c |= (doc_sig[i + j] > 0) << (7-j);
        //fwrite(&c, sizeof(byte), 1, sig_file);
        fprintf(sig_file, "%02x", c);
    }
    fprintf(sig_file, "\n");
}

#define min(a,b) ((a) < (b) ? (a) : (b))

void partition(char* sequence, int length)
{
    int i=0;
    int part = 0;
    do
    {
        compute_signature(sequence+i, min(PARTITION_SIZE, length-i));
#ifdef DEBUG
        printf("End of partition %d\n\n", part++);
#endif
        i += PARTITION_SIZE/2;
    }
    while (i+PARTITION_SIZE/2 < length);
    fprintf(sig_file, "\n");
    doc++;
}

int power(int n, int e)
{
    int p = 1;
    for (int j=0; j<e; j++)
        p *= n;
    return p;
}

int main(int argc, char* argv[])
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <filename>\n", argv[0]);
        return 1;
    }
    const char* filename = argv[1];
    
    WORDLEN = 3;
    PARTITION_SIZE = 16;
    int WORDS = power(20, WORDLEN);

    for (int i=0; i<strlen(alphabet); i++)
        inverse[alphabet[i]] = i;

    auto start = std::chrono::high_resolution_clock::now();

    FILE* file = fopen64(filename, "r");

    if (file == NULL)
    {
        fprintf(stderr, "Error: failed to open file %s\n", filename);
        return 1;
    }

    char outfile[256];
    sprintf(outfile, "%s.part%d_sigs%02d_%d", filename, PARTITION_SIZE, WORDLEN, SIGNATURE_LEN);
    sig_file = fopen64(outfile, "w");

    char buffer[10000];
    while (!feof(file))
    {
        fgets(buffer, 10000, file); // skip meta data line
        fgets(buffer, 10000, file);
        int n = (int)strlen(buffer) - 1;
        buffer[n] = 0;
#ifdef DEBUG
        printf("Processing line of length %d\n", n);
#endif
        partition(buffer, n);
#ifdef DEBUG
        printf("\n");
#endif
    }
    fclose(file);

    fclose(sig_file);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    printf("%s %f seconds\n", filename, duration.count());

    return 0;
}
