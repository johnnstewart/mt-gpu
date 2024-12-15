// MT19937-32 Mersenne Twister parameters
#define MT_N 624
#define MT_M 397
#define MT_MATRIX_A 0x9908b0dfUL // constant vector a
#define MT_UPPER_MASK 0x80000000UL // most significant w-r bits
#define MT_LOWER_MASK 0x7fffffffUL // least significant r bits

// Mersenne Twister state and index per thread
__private uint mt[MT_N];
__private int mti = MT_N + 1;

// Initialize Mersenne Twister with a seed
void init_mt(uint seed) {
    mt[0] = seed;
    for (mti = 1; mti < MT_N; mti++) {
        mt[mti] = (1812433253UL * (mt[mti - 1] ^ (mt[mti - 1] >> 30)) + mti);
    }
}

// Extract the next random number from the generator
uint mt_extract() {
    uint y;
    static const uint mag01[2] = {0x0UL, MT_MATRIX_A};

    if (mti >= MT_N) {
        int kk;

        for (kk = 0; kk < MT_N - MT_M; kk++) {
            y = (mt[kk] & MT_UPPER_MASK) | (mt[kk + 1] & MT_LOWER_MASK);
            mt[kk] = mt[kk + MT_M] ^ (y >> 1) ^ mag01[y & 0x1UL];
        }
        for (; kk < MT_N - 1; kk++) {
            y = (mt[kk] & MT_UPPER_MASK) | (mt[kk + 1] & MT_LOWER_MASK);
            mt[kk] = mt[kk + (MT_M - MT_N)] ^ (y >> 1) ^ mag01[y & 0x1UL];
        }
        y = (mt[MT_N - 1] & MT_UPPER_MASK) | (mt[0] & MT_LOWER_MASK);
        mt[MT_N - 1] = mt[MT_M - 1] ^ (y >> 1) ^ mag01[y & 0x1UL];

        mti = 0;
    }

    y = mt[mti++];

    // Tempering
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680UL;
    y ^= (y << 15) & 0xefc60000UL;
    y ^= (y >> 18);

    return y;
}

// Kernel function for mnemonic-based seed generation
__kernel void just_seed(__global uchar * target_mnemonic, __global uchar * found_mnemonic) {
    ulong idx = get_global_id(0);

    // Initialize Mersenne Twister with a unique seed per thread
    init_mt((uint)idx);

    // Generate entropy values from Mersenne Twister RNG
    ulong mnemonic_lo = (ulong)mt_extract() | ((ulong)mt_extract() << 32);
    ulong mnemonic_hi = (ulong)mt_extract() | ((ulong)mt_extract() << 32);

    uchar bytes[16];
    bytes[0] = mnemonic_lo & 0xFF;
    bytes[1] = (mnemonic_lo >> 8) & 0xFF;
    bytes[2] = (mnemonic_lo >> 16) & 0xFF;
    bytes[3] = (mnemonic_lo >> 24) & 0xFF;
    bytes[4] = (mnemonic_lo >> 32) & 0xFF;
    bytes[5] = (mnemonic_lo >> 40) & 0xFF;
    bytes[6] = (mnemonic_lo >> 48) & 0xFF;
    bytes[7] = (mnemonic_lo >> 56) & 0xFF;
    bytes[8] = mnemonic_hi & 0xFF;
    bytes[9] = (mnemonic_hi >> 8) & 0xFF;
    bytes[10] = (mnemonic_hi >> 16) & 0xFF;
    bytes[11] = (mnemonic_hi >> 24) & 0xFF;
    bytes[12] = (mnemonic_hi >> 32) & 0xFF;
    bytes[13] = (mnemonic_hi >> 40) & 0xFF;
    bytes[14] = (mnemonic_hi >> 48) & 0xFF;
    bytes[15] = (mnemonic_hi >> 56) & 0xFF;

    uchar mnemonic_hash[32];
    sha256(&bytes, 16, &mnemonic_hash);
    uchar checksum = mnemonic_hash[0] >> 4;

    ushort indices[12];
    indices[0] = (mnemonic_hi & (2047 << 53)) >> 53;
    indices[1] = (mnemonic_hi & (2047 << 42)) >> 42;
    indices[2] = (mnemonic_hi & (2047 << 31)) >> 31;
    indices[3] = (mnemonic_hi & (2047 << 20)) >> 20;
    indices[4] = (mnemonic_hi & (2047 << 9)) >> 9;
    indices[5] = ((mnemonic_hi << 55) >> 53) | ((mnemonic_lo & (3 << 62)) >> 62);
    indices[6] = (mnemonic_lo & (2047 << 51)) >> 51;
    indices[7] = (mnemonic_lo & (2047 << 40)) >> 40;
    indices[8] = (mnemonic_lo & (2047 << 29)) >> 29;
    indices[9] = (mnemonic_lo & (2047 << 18)) >> 18;
    indices[10] = (mnemonic_lo & (2047 << 7)) >> 7;
    indices[11] = ((mnemonic_lo << 57) >> 53) | checksum;

    uchar mnemonic[180];
    int mnemonic_index = 0;

    for (int i=0; i < 12; i++) {
        int word_index = indices[i];
        int word_length = word_lengths[word_index];

        for(int j=0;j<word_length;j++) {
            mnemonic[mnemonic_index] = words[word_index][j];
            mnemonic_index++;
        }
        mnemonic[mnemonic_index] = 32; // Space between words
        mnemonic_index++;
    }
    mnemonic[mnemonic_index - 1] = 0; // Null terminator

    uchar ipad_key[128];
    uchar opad_key[128];
    for(int x=0;x<128;x++){
        ipad_key[x] = 0x36;
        opad_key[x] = 0x5c;
    }

    for(int x=0;x<mnemonic_index;x++){
        ipad_key[x] ^= mnemonic[x];
        opad_key[x] ^= mnemonic[x];
    }

    uchar seed[64] = { 0 };
    uchar sha512_result[64] = { 0 };
    uchar key_previous_concat[256] = { 0 };
    uchar salt[12] = { 109, 110, 101, 109, 111, 110, 105, 99, 0, 0, 0, 1 };

    for(int x=0;x<128;x++){
        key_previous_concat[x] = ipad_key[x];
    }
    for(int x=0;x<12;x++){
        key_previous_concat[x+128] = salt[x];
    }

    sha512(&key_previous_concat, 140, &sha512_result);
    copy_pad_previous(&opad_key, &sha512_result, &key_previous_concat);
    sha512(&key_previous_concat, 192, &sha512_result);
    xor_seed_with_round(&seed, &sha512_result);

    for(int x=1;x<2048;x++){
        copy_pad_previous(&ipad_key, &sha512_result, &key_previous_concat);
        sha512(&key_previous_concat, 192, &sha512_result);
        copy_pad_previous(&opad_key, &sha512_result, &key_previous_concat);
        sha512(&key_previous_concat, 192, &sha512_result);
        xor_seed_with_round(&seed, &sha512_result);
    }

    // Check if generated seed matches target criteria (mock condition)
    if (/* condition for found mnemonic */) {
        found_mnemonic[0] = 0x01;
    }
    // Copy the generated mnemonic to the output
    for (int i = 0; i < mnemonic_index; i++) {
        target_mnemonic[i] = mnemonic[i];
    }
}
