// int_to_address.cl

#ifndef MT19937_CL
#define MT19937_CL

typedef unsigned int uint;

// MT19937 structure and constants
typedef struct {
    uint state[624]; // State vector
    uint index;      // Current index in the state vector
} mt19937_t;

// MT19937-32 constants
#define MT19937_N         624
#define MT19937_M         397
#define MT19937_MATRIX_A  0x9908B0DF
#define MT19937_UPPER_MASK 0x80000000
#define MT19937_LOWER_MASK 0x7FFFFFFF
#define MT19937_F         1812433253

// Initialize MT19937 with a seed
inline void mt19937_init(mt19937_t *mt, uint seed) {
    int i;
    mt->state[0] = seed;
    for (i = 1; i < MT19937_N; i++) {
        mt->state[i] = (MT19937_F * (mt->state[i-1] ^ (mt->state[i-1] >> 30)) + i) & 0xFFFFFFFF;
    }
    mt->index = MT19937_N;
}

// Generate a random number using MT19937
inline uint mt19937_generate(mt19937_t *mt) {
    int i;
    uint y;

    if (mt->index >= MT19937_N) {
        for (i = 0; i < MT19937_N; i++) {
            uint x = (mt->state[i] & MT19937_UPPER_MASK) | (mt->state[(i+1) % MT19937_N] & MT19937_LOWER_MASK);
            uint xA = x >> 1;
            if (x & 0x1) {
                xA ^= MT19937_MATRIX_A;
            }
            mt->state[i] = mt->state[(i + MT19937_M) % MT19937_N] ^ xA;
        }
        mt->index = 0;
    }

    y = mt->state[mt->index++];

    // Tempering transformations
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9D2C5680;
    y ^= (y << 15) & 0xEFC60000;
    y ^= (y >> 18);

    return y;
}

#endif // MT19937_CL

// Base58 encoding constants
#define BASE58_ALPHABET "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"

// Function to count leading zero bytes
inline int count_leading_zeros(const uchar* input, int length) {
    int count = 0;
    for(int i = 0; i < length; i++) {
        if(input[i] == 0x00) {
            count++;
        } else {
            break;
        }
    }
    return count;
}

// Function to perform Base58 encoding
inline void base58_encode(const uchar* input, int input_len, char* output) {
    // Initialize an array to hold the Base58 digits
    int max_output_size = 35; // Maximum 34 characters for 25-byte input + null terminator
    int digits[35];
    for(int i = 0; i < 35; i++) {
        digits[i] = 0;
    }

    // Copy input bytes into a temporary array for manipulation
    uchar temp_input[25];
    for(int i = 0; i < input_len; i++) {
        temp_input[i] = input[i];
    }

    // Convert byte array to big integer representation
    int length = input_len;
    for(int i = 0; i < input_len; i++) {
        temp_input[i] = input[i];
    }

    // Count leading zeros
    int leading_zeros = count_leading_zeros(input, input_len);

    // Perform Base58 encoding
    int digit_index = 0;
    bool done = false;
    while(!done) {
        int remainder = 0;
        bool any_non_zero = false;
        for(int i = 0; i < input_len; i++) {
            int accumulator = (remainder << 8) + temp_input[i];
            temp_input[i] = accumulator / 58;
            remainder = accumulator % 58;
            if(temp_input[i] != 0) {
                any_non_zero = true;
            }
        }
        digits[digit_index] = remainder;
        digit_index++;
        if(!any_non_zero) {
            done = true;
        }
    }

    // Add '1's for leading zeros
    int output_index = 0;
    for(int i = 0; i < leading_zeros; i++) {
        output[output_index++] = BASE58_ALPHABET[0];
    }

    // Convert digits to characters
    for(int i = digit_index - 1; i >= 0; i--) {
        output[output_index++] = BASE58_ALPHABET[digits[i]];
    }

    // Null-terminate the string
    if(output_index < max_output_size) {
        output[output_index] = 0;
    } else {
        output[max_output_size - 1] = 0; // Ensure null termination
    }
}

#define ENTROPY_BITS 256
#define ENTROPY_BYTES (ENTROPY_BITS / 8) // 32
#define CHECKSUM_BITS (ENTROPY_BITS / 32) // 8
#define TOTAL_BITS (ENTROPY_BITS + CHECKSUM_BITS) // 264
#define WORDS 24
#define BITS_PER_WORD 11 // 264 / 24 = 11
#define MAX_MNEMONIC_LENGTH 256 // Adjust as needed

__kernel void int_to_address(
    uint seed, // Single seed value
    __global uchar * target_mnemonic,
    __global uchar * found_mnemonic,
    __global char * addresses_checked, // Changed to char for Base58 strings
    __global uchar * mnemonics_buffer // New buffer to store all mnemonics
) {
    ulong idx = get_global_id(0);

    // Initialize MT19937-32 with the single seed
    mt19937_t mt;
    mt19937_init(&mt, seed + (uint)idx); // Ensure uniqueness per work-item if needed

    // Generate 32 bytes (256 bits) of entropy
    uchar bytes[ENTROPY_BYTES];
    for(int i = 0; i < ENTROPY_BYTES; i++) {
        bytes[i] = (uchar)(mt19937_generate(&mt) & 0xFF); // Equivalent to rng() % 256
    }

    // Continue with existing mnemonic generation and address checking
    uchar mnemonic_hash[32];
    sha256(&bytes, ENTROPY_BYTES, &mnemonic_hash);
    uchar checksum = mnemonic_hash[0]; // 8 bits

    // Construct a bitstream
    uchar bitstream[TOTAL_BITS] = {0}; // 256 bits entropy + 8 bits checksum
    for(int i = 0; i < ENTROPY_BYTES; i++) {
        for(int b = 0; b < 8; b++) {
            bitstream[i*8 + b] = (bytes[i] >> (7 - b)) & 0x01;
        }
    }
    // Add checksum bits
    for(int b = 0; b < CHECKSUM_BITS; b++) {
        bitstream[ENTROPY_BITS + b] = (checksum >> (CHECKSUM_BITS - 1 - b)) & 0x01;
    }

    // Extract 11-bit indices
    ushort indices[WORDS];
    for(int i = 0; i < WORDS; i++) {
        indices[i] = 0;
        for(int b = 0; b < BITS_PER_WORD; b++) {
            indices[i] = (indices[i] << 1) | bitstream[i*BITS_PER_WORD + b];
        }
    }

    // Generate mnemonic using the indices
    uchar mnemonic[MAX_MNEMONIC_LENGTH] = {0};
    int mnemonic_length = 0;
    int mnemonic_index = 0;
    for (int i = 0; i < WORDS; i++) {
        int word_index = indices[i];
        int word_length = word_lengths[word_index];

        for(int j = 0; j < word_length; j++) {
            mnemonic[mnemonic_index] = words[word_index][j];
            mnemonic_index++;
        }
        // Add space separator
        if (i < WORDS - 1) { // No space after last word
            mnemonic[mnemonic_index] = 32; // ASCII space
            mnemonic_index++;
        }
    }
    mnemonic[mnemonic_index] = 0; // Null-terminate
    mnemonic_length = mnemonic_index;

    // Copy mnemonic to mnemonics_buffer
    for(int i = 0; i < MAX_MNEMONIC_LENGTH; i++) {
        if (i < mnemonic_length) {
            mnemonics_buffer[idx * MAX_MNEMONIC_LENGTH + i] = mnemonic[i];
        } else {
            mnemonics_buffer[idx * MAX_MNEMONIC_LENGTH + i] = 0;
        }
    }

    // Derive seed from mnemonic (HMAC-SHA512 or similar steps)
    uchar ipad_key[128];
    uchar opad_key[128];
    for(int x = 0; x < 128; x++) {
        ipad_key[x] = 0x36;
        opad_key[x] = 0x5C;
    }

    for(int x = 0; x < mnemonic_length; x++) {
        ipad_key[x] ^= mnemonic[x];
        opad_key[x] ^= mnemonic[x];
    }

    uchar derived_seed[64] = { 0 };
    uchar sha512_result[64] = { 0 };
    uchar key_previous_concat[256] = { 0 };
    uchar salt[12] = { 109, 110, 101, 109, 111, 110, 105, 99, 0, 0, 0, 1 };

    // Concatenate ipad_key and salt
    for(int x = 0; x < 128; x++) {
        key_previous_concat[x] = ipad_key[x];
    }
    for(int x = 0; x < 12; x++) {
        key_previous_concat[x + 128] = salt[x];
    }

    sha512(&key_previous_concat, 140, &sha512_result);
    copy_pad_previous(&opad_key, &sha512_result, &key_previous_concat);
    sha512(&key_previous_concat, 192, &sha512_result);
    xor_seed_with_round(&derived_seed, &sha512_result);

    for(int x = 1; x < 2048; x++) {
        copy_pad_previous(&ipad_key, &sha512_result, &key_previous_concat);
        sha512(&key_previous_concat, 192, &sha512_result);
        copy_pad_previous(&opad_key, &sha512_result, &key_previous_concat);
        sha512(&key_previous_concat, 192, &sha512_result);
        xor_seed_with_round(&derived_seed, &sha512_result);
    }

    uchar network = BITCOIN_MAINNET;
    extended_private_key_t master_private;
    extended_public_key_t master_public;

    new_master_from_seed(network, &derived_seed, &master_private);
    public_from_private(&master_private, &master_public);

    uchar serialized_master_public[33];
    serialized_public_key(&master_public, &serialized_master_public);
    extended_private_key_t target_key;
    extended_public_key_t target_public_key;

    hardened_private_child_from_private(&master_private, &target_key, 49);
    hardened_private_child_from_private(&target_key, &target_key, 0);
    hardened_private_child_from_private(&target_key, &target_key, 0);
    normal_private_child_from_private(&target_key, &target_key, 0);
    normal_private_child_from_private(&target_key, &target_key, 0);
    public_from_private(&target_key, &target_public_key);

    uchar raw_address[25] = {0};
    p2shwpkh_address_for_public_key(&target_public_key, &raw_address);

    // Define your target address here (ensure it's correctly formatted)
    uchar target_address[25] = {0x05, 0xAD, 0xA1, 0x2B, 0x11, 0x3D, 0x9B, 0x19, 
                                  0x61, 0x47, 0x57, 0xD1, 0x9F, 0xC0, 0x8D, 0xDD, 
                                  0x53, 0x4B, 0xF0, 0x22, 0x76, 0xBD, 0x3A, 0x31, 
                                  0x46};

    bool found_target = true;
    for(int i = 0; i < 25; i++) {
        if(raw_address[i] != target_address[i]){
            found_target = false;
            break;
        }
    }

    // **New Code to Log Checked Addresses with Base58 Encoding**
    // Perform Base58 encoding of the raw_address
    char address_base58[35] = {0}; // Maximum 34 characters + null terminator
    base58_encode(raw_address, 25, address_base58);

    // Write the Base58-encoded address to the addresses_checked buffer
    for(int i = 0; i < 34; i++) {
        addresses_checked[idx * 34 + i] = address_base58[i];
    }

    if(found_target) {
        found_mnemonic[0] = 0x01;
        for(int i = 0; i < mnemonic_index; i++) {
            target_mnemonic[i] = mnemonic[i];
        }
        target_mnemonic[mnemonic_index] = 0; // Null-terminate
    }
}
