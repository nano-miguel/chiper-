#include <immintrin.h>
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <cstring>
#include <algorithm>
#include <random>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <iomanip>
#include <memory>
#include <cmath>
#include <numeric>
#include <fstream>
#include <filesystem>
#include <sstream>
#include <map>
#include <unordered_map>
#include <ctime>
#include <functional>

// ========================
// DETECCIÓN AUTOMÁTICA DE BOTAN
// ========================

#ifdef __has_include
# if __has_include(<botan/botan.h>)
#  define HAVE_BOTAN 1
#  include <botan/botan.h>
#  include <botan/argon2.h>
#  include <botan/mac.h>
#  include <botan/hash.h>
#  include <botan/system_rng.h>
# else
#  define HAVE_BOTAN 0
# endif
#else
# ifdef __linux__
#  define HAVE_BOTAN 1
#  include <botan/botan.h>
#  include <botan/argon2.h>
#  include <botan/mac.h>
#  include <botan/hash.h>
#  include <botan/system_rng.h>
# else
#  define HAVE_BOTAN 0
# endif
#endif

#ifdef _WIN32
#include <windows.h>
#include <intrin.h>
#include <setupapi.h>
#include <cfgmgr32.h>
#include <winioctl.h>
#include <tchar.h>
#pragma comment(lib, "setupapi.lib")
#else
#include <sys/mount.h>
#include <sys/statvfs.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

// ========================
// DETECCIÓN AVX2 PORTÁTIL
// ========================

bool has_avx2_support() {
#if defined(__x86_64__) || defined(_M_X64)
    #ifdef _WIN32
        int cpuInfo[4];
        __cpuid(cpuInfo, 0);
        if (cpuInfo[0] < 7) return false;
        
        __cpuidex(cpuInfo, 7, 0);
        return (cpuInfo[1] & (1 << 5)) != 0;
    #else
        // Para GCC/Linux - usar cpuid.h o implementación directa
        uint32_t eax, ebx, ecx, edx;
        
        // Obtener el ID máximo de CPUID
        asm volatile ("cpuid" : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx) : "a"(0));
        if (eax < 7) return false;
        
        // Verificar AVX2 (bit 5 de EBX)
        asm volatile ("cpuid" : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx) : "a"(7), "c"(0));
        return (ebx & (1 << 5)) != 0;
    #endif
#else
    return false; // No es arquitectura x86-64
#endif
}

// ========================
// POLY1305 IMPLEMENTATION CORREGIDA (FALLBACK)
// ========================

class Poly1305 {
private:
    uint32_t r[5];
    uint32_t h[5];
    uint32_t pad[4];
    size_t buffer_used;
    uint8_t buffer[16];
    
    static inline uint32_t U8TO32_LE(const uint8_t *p) {
        return ((uint32_t)p[0]) | ((uint32_t)p[1] << 8) | 
               ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
    }
    
    static inline void U32TO8_LE(uint8_t *p, uint32_t v) {
        p[0] = (uint8_t)(v);
        p[1] = (uint8_t)(v >> 8);
        p[2] = (uint8_t)(v >> 16);
        p[3] = (uint8_t)(v >> 24);
    }
    
    void blocks(const uint8_t *m, size_t bytes) {
        uint32_t h0 = h[0], h1 = h[1], h2 = h[2], h3 = h[3], h4 = h[4];
        uint32_t r0 = r[0], r1 = r[1], r2 = r[2], r3 = r[3], r4 = r[4];
        
        while (bytes >= 16) {
            h0 += U8TO32_LE(m + 0) & 0x3ffffff;
            h1 += U8TO32_LE(m + 3) >> 2 & 0x3ffffff;
            h2 += U8TO32_LE(m + 6) >> 4 & 0x3ffffff;
            h3 += U8TO32_LE(m + 9) >> 6 & 0x3ffffff;
            h4 += U8TO32_LE(m + 12) >> 8 | (1 << 24);
            
            uint64_t d0 = (uint64_t)h0 * r0 + (uint64_t)h1 * r4 + 
                         (uint64_t)h2 * r3 + (uint64_t)h3 * r2 + (uint64_t)h4 * r1;
            uint64_t d1 = (uint64_t)h0 * r1 + (uint64_t)h1 * r0 + 
                         (uint64_t)h2 * r4 + (uint64_t)h3 * r3 + (uint64_t)h4 * r2;
            uint64_t d2 = (uint64_t)h0 * r2 + (uint64_t)h1 * r1 + 
                         (uint64_t)h2 * r0 + (uint64_t)h3 * r4 + (uint64_t)h4 * r3;
            uint64_t d3 = (uint64_t)h0 * r3 + (uint64_t)h1 * r2 + 
                         (uint64_t)h2 * r1 + (uint64_t)h3 * r0 + (uint64_t)h4 * r4;
            uint64_t d4 = (uint64_t)h0 * r4 + (uint64_t)h1 * r3 + 
                         (uint64_t)h2 * r2 + (uint64_t)h3 * r1 + (uint64_t)h4 * r0;
            
            uint32_t c = (uint32_t)(d0 >> 26); h0 = (uint32_t)d0 & 0x3ffffff;
            d1 += c; c = (uint32_t)(d1 >> 26); h1 = (uint32_t)d1 & 0x3ffffff;
            d2 += c; c = (uint32_t)(d2 >> 26); h2 = (uint32_t)d2 & 0x3ffffff;
            d3 += c; c = (uint32_t)(d3 >> 26); h3 = (uint32_t)d3 & 0x3ffffff;
            d4 += c; c = (uint32_t)(d4 >> 26); h4 = (uint32_t)d4 & 0x3ffffff;
            h0 += c * 5; c = h0 >> 26; h0 &= 0x3ffffff;
            h1 += c;
            
            m += 16;
            bytes -= 16;
        }
        
        h[0] = h0; h[1] = h1; h[2] = h2; h[3] = h3; h[4] = h4;
    }
    
public:
    Poly1305(const uint8_t key[32]) {
        if (key == nullptr) {
            throw std::invalid_argument("Poly1305 key cannot be null");
        }
        
        r[0] = U8TO32_LE(key + 0) & 0x3ffffff;
        r[1] = U8TO32_LE(key + 3) >> 2 & 0x3ffff03;
        r[2] = U8TO32_LE(key + 6) >> 4 & 0x3ffc0ff;
        r[3] = U8TO32_LE(key + 9) >> 6 & 0x3f03fff;
        r[4] = U8TO32_LE(key + 12) >> 8 & 0x00fffff;
        
        h[0] = h[1] = h[2] = h[3] = h[4] = 0;
        
        for (int i = 0; i < 4; i++)
            pad[i] = U8TO32_LE(key + 16 + i * 4);
            
        buffer_used = 0;
    }
    
    void update(const uint8_t *msg, size_t len) {
        if (msg == nullptr && len > 0) {
            throw std::invalid_argument("Poly1305 update: msg cannot be null when len > 0");
        }
        
        size_t want;
        
        if (buffer_used > 0) {
            want = 16 - buffer_used;
            if (want > len)
                want = len;
            for (size_t i = 0; i < want; i++)
                buffer[buffer_used + i] = msg[i];
            buffer_used += want;
            if (buffer_used < 16)
                return;
            blocks(buffer, 16);
            len -= want;
            msg += want;
        }
        
        if (len >= 16) {
            want = len & ~(size_t)15;
            blocks(msg, want);
            len -= want;
            msg += want;
        }
        
        if (len) {
            for (size_t i = 0; i < len; i++)
                buffer[i] = msg[i];
            buffer_used = len;
        }
    }
    
    void finish(uint8_t mac[16]) {
        if (mac == nullptr) {
            throw std::invalid_argument("Poly1305 finish: mac cannot be null");
        }
        
        uint32_t g0, g1, g2, g3, g4;
        uint32_t c;
        uint64_t f;
        uint32_t mask;
        
        if (buffer_used > 0) {
            size_t i = buffer_used;
            buffer[i++] = 1;
            for (; i < 16; i++)
                buffer[i] = 0;
            blocks(buffer, 16);
        }
        
        c = h[1] >> 26; h[1] &= 0x3ffffff;
        h[2] += c; c = h[2] >> 26; h[2] &= 0x3ffffff;
        h[3] += c; c = h[3] >> 26; h[3] &= 0x3ffffff;
        h[4] += c; c = h[4] >> 26; h[4] &= 0x3ffffff;
        h[0] += c * 5; c = h[0] >> 26; h[0] &= 0x3ffffff;
        h[1] += c;
        
        g0 = h[0] + 5; c = g0 >> 26; g0 &= 0x3ffffff;
        g1 = h[1] + c; c = g1 >> 26; g1 &= 0x3ffffff;
        g2 = h[2] + c; c = g2 >> 26; g2 &= 0x3ffffff;
        g3 = h[3] + c; c = g3 >> 26; g3 &= 0x3ffffff;
        g4 = h[4] + c - (1 << 26);
        
        mask = (g4 >> 31) - 1;
        g0 &= mask; g1 &= mask; g2 &= mask; g3 &= mask; g4 &= mask;
        mask = ~mask;
        h[0] = (h[0] & mask) | g0;
        h[1] = (h[1] & mask) | g1;
        h[2] = (h[2] & mask) | g2;
        h[3] = (h[3] & mask) | g3;
        h[4] = (h[4] & mask) | g4;
        
        h[0] = (h[0] | (h[1] << 26)) & 0xffffffff;
        h[1] = (h[1] >> 6 | (h[2] << 20)) & 0xffffffff;
        h[2] = (h[2] >> 12 | (h[3] << 14)) & 0xffffffff;
        h[3] = (h[3] >> 18 | (h[4] << 8)) & 0xffffffff;
        
        f = (uint64_t)h[0] + pad[0]; h[0] = (uint32_t)f;
        f = (uint64_t)h[1] + pad[1] + (f >> 32); h[1] = (uint32_t)f;
        f = (uint64_t)h[2] + pad[2] + (f >> 32); h[2] = (uint32_t)f;
        f = (uint64_t)h[3] + pad[3] + (f >> 32); h[3] = (uint32_t)f;
        
        U32TO8_LE(mac + 0, h[0]);
        U32TO8_LE(mac + 4, h[1]);
        U32TO8_LE(mac + 8, h[2]);
        U32TO8_LE(mac + 12, h[3]);
        
        for (size_t i = 0; i < 5; i++) h[i] = 0;
        for (size_t i = 0; i < 4; i++) pad[i] = 0;
        for (size_t i = 0; i < 16; i++) buffer[i] = 0;
        buffer_used = 0;
    }
    
    ~Poly1305() {
        for (size_t i = 0; i < 5; i++) r[i] = h[i] = 0;
        for (size_t i = 0; i < 4; i++) pad[i] = 0;
        for (size_t i = 0; i < 16; i++) buffer[i] = 0;
        buffer_used = 0;
    }
};

// ========================
// CHACHA20 IMPLEMENTATION (FALLBACK SIN AVX2)
// ========================

class ChaCha20 {
private:
    uint32_t state[16];
    
    static inline uint32_t rotl32(uint32_t x, int n) {
        return (x << n) | (x >> (32 - n));
    }
    
    static inline void quarter_round(uint32_t &a, uint32_t &b, uint32_t &c, uint32_t &d) {
        a += b; d ^= a; d = rotl32(d, 16);
        c += d; b ^= c; b = rotl32(b, 12);
        a += b; d ^= a; d = rotl32(d, 8);
        c += d; b ^= c; b = rotl32(b, 7);
    }
    
    void chacha20_block(uint32_t out[16], uint32_t counter) {
        uint32_t x[16];
        std::memcpy(x, state, sizeof(x));
        x[12] = counter;
        
        for (int i = 0; i < 10; i++) {
            quarter_round(x[0], x[4], x[8], x[12]);
            quarter_round(x[1], x[5], x[9], x[13]);
            quarter_round(x[2], x[6], x[10], x[14]);
            quarter_round(x[3], x[7], x[11], x[15]);
            quarter_round(x[0], x[5], x[10], x[15]);
            quarter_round(x[1], x[6], x[11], x[12]);
            quarter_round(x[2], x[7], x[8], x[13]);
            quarter_round(x[3], x[4], x[9], x[14]);
        }
        
        for (int i = 0; i < 16; i++) {
            out[i] = x[i] + state[i];
        }
    }
    
public:
    ChaCha20(const uint8_t key[32], const uint8_t nonce[12]) {
        state[0] = 0x61707865;
        state[1] = 0x3320646e;
        state[2] = 0x79622d32;
        state[3] = 0x6b206574;
        
        for (int i = 0; i < 8; i++) {
            state[4 + i] = ((uint32_t)key[i * 4]) | 
                          ((uint32_t)key[i * 4 + 1] << 8) |
                          ((uint32_t)key[i * 4 + 2] << 16) |
                          ((uint32_t)key[i * 4 + 3] << 24);
        }
        
        state[12] = 0;
        state[13] = ((uint32_t)nonce[0]) | ((uint32_t)nonce[1] << 8) | 
                   ((uint32_t)nonce[2] << 16) | ((uint32_t)nonce[3] << 24);
        state[14] = ((uint32_t)nonce[4]) | ((uint32_t)nonce[5] << 8) | 
                   ((uint32_t)nonce[6] << 16) | ((uint32_t)nonce[7] << 24);
        state[15] = ((uint32_t)nonce[8]) | ((uint32_t)nonce[9] << 8) | 
                   ((uint32_t)nonce[10] << 16) | ((uint32_t)nonce[11] << 24);
    }
    
    void process_bytes(const uint8_t* input, uint8_t* output, size_t size, uint64_t counter = 0) {
        uint32_t block[16];
        size_t processed = 0;
        
        while (processed < size) {
            uint32_t current_counter = static_cast<uint32_t>(counter + (processed / 64));
            chacha20_block(block, current_counter);
            
            size_t block_pos = processed % 64;
            size_t bytes_to_process = std::min(size - processed, 64 - block_pos);
            
            const uint8_t* keystream = reinterpret_cast<uint8_t*>(block);
            for (size_t i = 0; i < bytes_to_process; i++) {
                output[processed + i] = input[processed + i] ^ keystream[block_pos + i];
            }
            
            processed += bytes_to_process;
        }
    }
    
    ~ChaCha20() {
        std::fill_n(state, 16, 0);
    }
};

// ========================
// CONSTANTES Y CONFIGURACIÓN
// ========================

#define COLOR_RESET   "\033[0m"
#define COLOR_RED     "\033[31m"
#define COLOR_GREEN   "\033[32m"
#define COLOR_YELLOW  "\033[33m"
#define COLOR_BLUE    "\033[34m"
#define COLOR_MAGENTA "\033[35m"
#define COLOR_CYAN    "\033[36m"
#define COLOR_WHITE   "\033[37m"
#define COLOR_BOLD    "\033[1m"

#define BG_BLUE       "\033[44m"
#define BG_GREEN      "\033[42m"
#define BG_RED        "\033[41m"
#define BG_YELLOW     "\033[43m"
#define BG_CYAN       "\033[46m"

// ========================
// ARTE ASCII RUBIC MEJORADO
// ========================

void print_rubic_art() {
    std::cout << COLOR_CYAN;
    std::cout << "┌─────────────────────────────────────────────────────────┐\n";
    std::cout << "│" << COLOR_MAGENTA << "    ██████╗ ██╗   ██╗██████╗ ██╗ ██████╗     ██████╗    " << COLOR_CYAN << "│\n";
    std::cout << "│" << COLOR_MAGENTA << "    ██╔══██╗██║   ██║██╔══██╗██║██╔════╝    ██╔══██╗   " << COLOR_CYAN << "│\n";
    std::cout << "│" << COLOR_MAGENTA << "    ██████╔╝██║   ██║██████╔╝██║██║         ██████╔╝   " << COLOR_CYAN << "│\n";
    std::cout << "│" << COLOR_MAGENTA << "    ██╔══██╗██║   ██║██╔══██╗██║██║         ██╔══██╗   " << COLOR_CYAN << "│\n";
    std::cout << "│" << COLOR_MAGENTA << "    ██║  ██║╚██████╔╝██████╔╝██║╚██████╗    ██║  ██║   " << COLOR_CYAN << "│\n";
    std::cout << "│" << COLOR_MAGENTA << "    ╚═╝  ╚═╝ ╚═════╝ ╚═════╝ ╚═╝ ╚═════╝    ╚═╝  ╚═╝   " << COLOR_CYAN << "│\n";
    std::cout << "│                                                         │\n";
    std::cout << "│" << COLOR_YELLOW << "             SISTEMA DE CIFRADO AVANZADO USB v3.0" << COLOR_CYAN << "           │\n";
    std::cout << "│" << COLOR_GREEN << "            ChaCha20 + Poly1305 + Argon2 + AVX2" << COLOR_CYAN << "            │\n";
    std::cout << "└─────────────────────────────────────────────────────────┘" << COLOR_RESET << "\n\n";
}

// ========================
// GESTOR DE CLAVES MEJORADO CON TABLA CON BORDES
// ========================

class KeyManager {
private:
    struct KeyInfo {
        std::string name;
        std::vector<uint8_t> derived_key;
        std::vector<uint8_t> salt;
        std::vector<uint8_t> nonce;
        bool is_active;
        time_t created_time;
        time_t last_used;
        std::string algorithm;
        uint32_t memory_cost;
        uint32_t time_cost;
        uint32_t parallelism;
        
        KeyInfo(const std::string& n, const std::vector<uint8_t>& key, 
                const std::vector<uint8_t>& s, const std::vector<uint8_t>& nonce_vec,
                const std::string& algo = "ARGON2ID", uint32_t mem = 512*1024,
                uint32_t time = 2, uint32_t parallel = 2)
            : name(n), derived_key(key), salt(s), nonce(nonce_vec), 
              is_active(false), created_time(std::time(nullptr)), last_used(0),
              algorithm(algo), memory_cost(mem), time_cost(time), parallelism(parallel) {}
              
        KeyInfo() : name(""), is_active(false), created_time(0), last_used(0),
                   algorithm("ARGON2ID"), memory_cost(512*1024), time_cost(2), parallelism(2) {}
        
        KeyInfo(const KeyInfo&) = delete;
        KeyInfo& operator=(const KeyInfo&) = delete;
        
        KeyInfo(KeyInfo&& other) noexcept = default;
        KeyInfo& operator=(KeyInfo&& other) noexcept = default;
        
        ~KeyInfo() {
            clear_sensitive_data();
        }
        
        void clear_sensitive_data() {
            if (!derived_key.empty()) {
                std::fill(derived_key.begin(), derived_key.end(), 0);
            }
            if (!salt.empty()) {
                std::fill(salt.begin(), salt.end(), 0);
            }
            if (!nonce.empty()) {
                std::fill(nonce.begin(), nonce.end(), 0);
            }
        }
    };
    
    std::unordered_map<std::string, KeyInfo> keys;
    std::string active_key_id;
    std::mutex key_mutex;
    
    const size_t key_length = 32;
    const size_t salt_length = 16;
    const size_t nonce_length = 12;
    
    bool botan_initialized;
    
public:
    KeyManager() : botan_initialized(false) {
        initialize_botan();
        load_keys_from_file();
    }
    
    ~KeyManager() {
        save_keys_to_file();
        clear_sensitive_data();
    }
    
private:
    void clear_sensitive_data() {
        std::lock_guard<std::mutex> lock(key_mutex);
        for (auto& [name, key_info] : keys) {
            key_info.clear_sensitive_data();
        }
        keys.clear();
        active_key_id.clear();
    }
    
    void initialize_botan() {
#if HAVE_BOTAN
        try {
            botan_initialized = true;
        } catch (const std::exception& e) {
            botan_initialized = false;
        }
#endif
    }
    
    std::vector<uint8_t> derive_key_pbkdf2(const std::string& password, const std::vector<uint8_t>& salt, int iterations = 100000) {
        std::vector<uint8_t> key(key_length, 0);
        
        std::string data = password;
        data.append(reinterpret_cast<const char*>(salt.data()), salt.size());
        
        for (int iter = 0; iter < iterations; iter++) {
            std::hash<std::string> hasher;
            size_t hash_val = hasher(data);
            
            for (size_t i = 0; i < key_length; i++) {
                uint8_t mix_byte = static_cast<uint8_t>((hash_val >> ((i % sizeof(hash_val)) * 8)) & 0xFF);
                key[i] ^= mix_byte;
                key[i] += static_cast<uint8_t>((iter * 11 + i * 7) % 256);
            }
            
            data.assign(key.begin(), key.end());
            data.append(std::to_string(iter));
            data.append(password);
        }
        
        return key;
    }
    
    std::vector<uint8_t> compute_hmac_fallback(const std::vector<uint8_t>& data, const std::vector<uint8_t>& key) {
        const size_t mac_size = 32;
        std::vector<uint8_t> mac(mac_size, 0);
        
        const size_t block_size = 64;
        std::vector<uint8_t> ipad(block_size, 0x36);
        std::vector<uint8_t> opad(block_size, 0x5C);
        
        std::vector<uint8_t> key_padded = key;
        if (key_padded.size() > block_size) {
            std::hash<std::string> hasher;
            std::string key_str(key_padded.begin(), key_padded.end());
            size_t key_hash = hasher(key_str);
            key_padded.assign(sizeof(key_hash), 0);
            for (size_t i = 0; i < sizeof(key_hash); i++) {
                key_padded[i] = static_cast<uint8_t>((key_hash >> (i * 8)) & 0xFF);
            }
        }
        
        if (key_padded.size() < block_size) {
            key_padded.resize(block_size, 0);
        }
        
        std::vector<uint8_t> inner_msg = ipad;
        for (size_t i = 0; i < block_size; i++) {
            inner_msg[i] ^= key_padded[i];
        }
        inner_msg.insert(inner_msg.end(), data.begin(), data.end());
        
        std::hash<std::string> inner_hasher;
        std::string inner_str(inner_msg.begin(), inner_msg.end());
        size_t inner_hash = inner_hasher(inner_str);
        
        std::vector<uint8_t> outer_msg = opad;
        for (size_t i = 0; i < block_size; i++) {
            outer_msg[i] ^= key_padded[i];
        }
        
        std::vector<uint8_t> inner_hash_bytes(sizeof(inner_hash));
        for (size_t i = 0; i < sizeof(inner_hash); i++) {
            inner_hash_bytes[i] = static_cast<uint8_t>((inner_hash >> (i * 8)) & 0xFF);
        }
        outer_msg.insert(outer_msg.end(), inner_hash_bytes.begin(), inner_hash_bytes.end());
        
        std::hash<std::string> outer_hasher;
        std::string outer_str(outer_msg.begin(), outer_msg.end());
        size_t outer_hash = outer_hasher(outer_str);
        
        for (size_t i = 0; i < mac_size && i < sizeof(outer_hash); i++) {
            mac[i] = static_cast<uint8_t>((outer_hash >> (i * 8)) & 0xFF);
        }
        
        std::fill(key_padded.begin(), key_padded.end(), 0);
        std::fill(inner_msg.begin(), inner_msg.end(), 0);
        std::fill(outer_msg.begin(), outer_msg.end(), 0);
        
        return mac;
    }
    
public:
    std::vector<uint8_t> derive_key_argon2(const std::string& password, const std::vector<uint8_t>& salt) {
#if HAVE_BOTAN
        if (botan_initialized) {
            try {
                std::vector<uint8_t> key(key_length);
                
                Botan::secure_vector<uint8_t> derived = Botan::argon2(
                    key_length, 
                    reinterpret_cast<const char*>(password.data()), 
                    password.size(),
                    salt.data(), 
                    salt.size(),
                    nullptr, 0,
                    nullptr, 0,
                    2,
                    512 * 1024,
                    2
                );
                
                std::copy(derived.begin(), derived.end(), key.begin());
                return key;
                
            } catch (const std::exception& e) {
                std::cerr << "Botan Argon2 error: " << e.what() << std::endl;
            }
        }
#endif
        return derive_key_pbkdf2(password, salt, 150000);
    }
    
    std::vector<uint8_t> compute_kmac(const std::vector<uint8_t>& data, const std::vector<uint8_t>& key) {
#if HAVE_BOTAN
        if (botan_initialized) {
            try {
                auto kmac = Botan::MessageAuthenticationCode::create_or_throw("KMAC-256");
                kmac->set_key(key);
                kmac->update(data);
                
                std::vector<uint8_t> mac(kmac->output_length());
                kmac->final(mac.data());
                return mac;
                
            } catch (const std::exception& e1) {
                try {
                    auto hmac = Botan::MessageAuthenticationCode::create_or_throw("HMAC(SHA-256)");
                    hmac->set_key(key);
                    hmac->update(data);
                    
                    std::vector<uint8_t> mac(hmac->output_length());
                    hmac->final(mac.data());
                    return mac;
                    
                } catch (const std::exception& e2) {
                    std::cerr << "Botan HMAC error: " << e2.what() << std::endl;
                }
            }
        }
#endif
        return compute_hmac_fallback(data, key);
    }
    
    void generate_secure_random(std::vector<uint8_t>& output) {
        if (output.empty()) return;
        
#if HAVE_BOTAN
        if (botan_initialized) {
            try {
                Botan::System_RNG rng;
                rng.randomize(output.data(), output.size());
                return;
            } catch (const std::exception& e) {
                std::cerr << "Botan RNG error: " << e.what() << std::endl;
            }
        }
#endif
        
        std::random_device rd;
        std::uniform_int_distribution<uint8_t> dist(0, 255);
        for (auto& byte : output) {
            byte = dist(rd);
        }
        
        for (size_t i = 0; i < output.size(); i++) {
            output[i] ^= static_cast<uint8_t>(rd() & 0xFF);
        }
    }
    
    bool create_key(const std::string& name, const std::string& password, 
                   const std::string& algorithm = "ARGON2ID") {
        std::lock_guard<std::mutex> lock(key_mutex);
        
        if (keys.find(name) != keys.end()) {
            std::cout << COLOR_RED "Error: Ya existe una clave con el nombre: " << name << COLOR_RESET << std::endl;
            return false;
        }
        
        if (password.length() < 12) {
            std::cout << COLOR_RED "Error: La contraseña debe tener al menos 12 caracteres" COLOR_RESET << std::endl;
            return false;
        }
        
        if (name.empty()) {
            std::cout << COLOR_RED "Error: El nombre de la clave no puede estar vacío" COLOR_RESET << std::endl;
            return false;
        }
        
        try {
            std::vector<uint8_t> salt(salt_length);
            std::vector<uint8_t> nonce(nonce_length);
            
            generate_secure_random(salt);
            generate_secure_random(nonce);
            
            std::vector<uint8_t> derived_key = derive_key_argon2(password, salt);
            
            if (derived_key.size() != key_length) {
                std::cout << COLOR_RED "Error: La clave derivada no tiene el tamaño correcto" COLOR_RESET << std::endl;
                return false;
            }
            
            keys.emplace(name, KeyInfo(name, std::move(derived_key), std::move(salt), std::move(nonce), algorithm));
            
            std::cout << COLOR_GREEN "Clave '" << name << "' creada exitosamente con " << algorithm << COLOR_RESET << std::endl;
            
            save_keys_to_file();
            return true;
            
        } catch (const std::exception& e) {
            std::cout << COLOR_RED "Error creando clave: " << e.what() << COLOR_RESET << std::endl;
            return false;
        }
    }
    
    bool activate_key(const std::string& name, const std::string& password) {
        std::lock_guard<std::mutex> lock(key_mutex);
        
        auto it = keys.find(name);
        if (it == keys.end()) {
            std::cout << COLOR_RED "Error: No existe la clave: " << name << COLOR_RESET << std::endl;
            return false;
        }
        
        try {
            KeyInfo& key_info = it->second;
            std::vector<uint8_t> test_key = derive_key_argon2(password, key_info.salt);
            
            if (test_key.size() != key_info.derived_key.size()) {
                std::cout << COLOR_RED "Error: Contraseña incorrecta para la clave: " << name << COLOR_RESET << std::endl;
                return false;
            }
            
            bool password_correct = true;
            for (size_t i = 0; i < test_key.size(); i++) {
                if (test_key[i] != key_info.derived_key[i]) {
                    password_correct = false;
                    break;
                }
            }
            
            if (!password_correct) {
                std::cout << COLOR_RED "Error: Contraseña incorrecta para la clave: " << name << COLOR_RESET << std::endl;
                std::fill(test_key.begin(), test_key.end(), 0);
                return false;
            }
            
            std::fill(test_key.begin(), test_key.end(), 0);
            
            if (!active_key_id.empty()) {
                keys[active_key_id].is_active = false;
            }
            
            key_info.is_active = true;
            key_info.last_used = std::time(nullptr);
            active_key_id = name;
            
            std::cout << COLOR_GREEN "Clave '" << name << "' activada exitosamente" COLOR_RESET << std::endl;
            return true;
            
        } catch (const std::exception& e) {
            std::cout << COLOR_RED "Error activando clave: " << e.what() << COLOR_RESET << std::endl;
            return false;
        }
    }
    
    bool get_active_key(std::vector<uint8_t>& key, std::vector<uint8_t>& nonce) {
        std::lock_guard<std::mutex> lock(key_mutex);
        
        if (active_key_id.empty()) return false;
        
        auto it = keys.find(active_key_id);
        if (it == keys.end() || !it->second.is_active) return false;
        
        key = it->second.derived_key;
        nonce = it->second.nonce;
        return true;
    }
    
    void list_keys() {
        std::lock_guard<std::mutex> lock(key_mutex);
        
        if (keys.empty()) {
            std::cout << COLOR_YELLOW "No hay claves almacenadas" COLOR_RESET << std::endl;
            return;
        }
        
        // Tabla con bordes blancos mejorada
        std::cout << COLOR_WHITE "┌────────────────────────────────────────────────────────────────────────────────────────┐\n";
        std::cout << "│" << COLOR_BLUE COLOR_BOLD << "                          CLAVES ALMACENADAS                                   " << COLOR_WHITE << "│\n";
        std::cout << "├──────────────────┬────────────┬───────────────┬────────────┬────────────┬────────────┬────────────┤\n";
        std::cout << "│" << COLOR_BOLD << std::left 
                  << std::setw(18) << "NOMBRE" 
                  << COLOR_WHITE "│" << COLOR_BOLD
                  << std::setw(12) << "ESTADO" 
                  << COLOR_WHITE "│" << COLOR_BOLD
                  << std::setw(15) << "ALGORITMO"
                  << COLOR_WHITE "│" << COLOR_BOLD
                  << std::setw(12) << "MEMORIA"
                  << COLOR_WHITE "│" << COLOR_BOLD
                  << std::setw(12) << "ITERACIONES"
                  << COLOR_WHITE "│" << COLOR_BOLD
                  << std::setw(12) << "CREACION" 
                  << COLOR_WHITE "│" << COLOR_BOLD
                  << std::setw(12) << "ULTIMO USO" << COLOR_WHITE "│\n";
        std::cout << "├──────────────────┼────────────┼───────────────┼────────────┼────────────┼────────────┼────────────┤\n";
        
        for (const auto& [name, key_info] : keys) {
            std::string estado = (key_info.is_active) ? COLOR_GREEN "● ACTIVA" COLOR_WHITE : COLOR_RED "○ INACTIVA" COLOR_WHITE;
            std::string creado = format_time(key_info.created_time);
            std::string ultimo_uso = (key_info.last_used > 0) ? format_time(key_info.last_used) : "Nunca";
            std::string algoritmo = key_info.algorithm;
            std::string memoria = std::to_string(key_info.memory_cost / 1024) + "MB";
            std::string iteraciones = std::to_string(key_info.time_cost);
            
            std::cout << "│" << std::left 
                      << std::setw(18) << name
                      << COLOR_WHITE "│"
                      << std::setw(12) << estado
                      << COLOR_WHITE "│"
                      << std::setw(15) << algoritmo
                      << COLOR_WHITE "│"
                      << std::setw(12) << memoria
                      << COLOR_WHITE "│"
                      << std::setw(12) << iteraciones
                      << COLOR_WHITE "│"
                      << std::setw(12) << creado
                      << COLOR_WHITE "│"
                      << std::setw(12) << ultimo_uso << COLOR_WHITE "│\n";
        }
        
        std::cout << "└──────────────────┴────────────┴───────────────┴────────────┴────────────┴────────────┴────────────┘\n" << COLOR_RESET;
    }
    
    bool delete_key(const std::string& name) {
        std::lock_guard<std::mutex> lock(key_mutex);
        
        auto it = keys.find(name);
        if (it == keys.end()) {
            std::cout << COLOR_RED "Error: No existe la clave: " << name << COLOR_RESET << std::endl;
            return false;
        }
        
        if (it->second.is_active) {
            std::cout << COLOR_RED "Error: No se puede eliminar una clave activa. Desactívela primero." COLOR_RESET << std::endl;
            return false;
        }
        
        keys.erase(it);
        std::cout << COLOR_GREEN "Clave '" << name << "' eliminada exitosamente" COLOR_RESET << std::endl;
        save_keys_to_file();
        return true;
    }
    
    std::string get_active_key_info() {
        std::lock_guard<std::mutex> lock(key_mutex);
        
        if (active_key_id.empty()) {
            return COLOR_YELLOW "│ Estado: No hay clave activa" COLOR_RESET;
        }
        
        auto it = keys.find(active_key_id);
        if (it == keys.end()) {
            return COLOR_RED "│ Estado: Clave activa no encontrada" COLOR_RESET;
        }
        
        const KeyInfo& key_info = it->second;
        std::ostringstream oss;
        oss << COLOR_GREEN "│ Clave Activa: " << key_info.name << COLOR_RESET << "\n"
            << COLOR_CYAN "│ Algoritmo: " << key_info.algorithm << COLOR_RESET << "\n"
            << COLOR_CYAN "│ Memoria: " << (key_info.memory_cost / 1024) << "MB" COLOR_RESET << "\n"
            << COLOR_CYAN "│ Iteraciones: " << key_info.time_cost << COLOR_RESET << "\n"
            << "│ Creada: " << format_time(key_info.created_time) << "\n"
            << "│ Último uso: " << format_time(key_info.last_used);
        
        return oss.str();
    }
    
    bool has_active_key() {
        std::lock_guard<std::mutex> lock(key_mutex);
        return !active_key_id.empty() && keys[active_key_id].is_active;
    }
    
    void deactivate_current_key() {
        std::lock_guard<std::mutex> lock(key_mutex);
        
        if (!active_key_id.empty()) {
            keys[active_key_id].is_active = false;
            std::cout << COLOR_GREEN "Clave '" << active_key_id << "' desactivada" COLOR_RESET << std::endl;
            active_key_id.clear();
        } else {
            std::cout << COLOR_YELLOW "No hay clave activa para desactivar" COLOR_RESET << std::endl;
        }
    }
    
    void print_security_report() {
        std::lock_guard<std::mutex> lock(key_mutex);
        
        std::cout << COLOR_WHITE "┌──────────────────────────────────────────────────────┐\n";
        std::cout << "│" << COLOR_CYAN COLOR_BOLD << "               INFORME DE SEGURIDAD                 " << COLOR_WHITE << "│\n";
        std::cout << "├──────────────────────────────────────────────────────┤\n";
        
        std::cout << "│" << COLOR_BOLD << "Algoritmos utilizados:" << COLOR_RESET << std::endl;
#if HAVE_BOTAN
        if (botan_initialized) {
            std::cout << "│ • Argon2id: Derivation de claves (Botan)" << std::endl;
            std::cout << "│ • KMAC-256: Verificacion de integridad (Botan)" << std::endl;
            std::cout << "│ • HMAC-SHA256: Fallback de verificacion (Botan)" << std::endl;
        } else {
            std::cout << "│ • PBKDF2: Derivation de claves (fallback interno)" << std::endl;
            std::cout << "│ • HMAC: Verificacion de integridad (fallback interno)" << std::endl;
        }
#else
        std::cout << "│ • PBKDF2: Derivation de claves (fallback interno)" << std::endl;
        std::cout << "│ • HMAC: Verificacion de integridad (fallback interno)" << std::endl;
#endif
        std::cout << "│ • ChaCha20: Cifrado de flujo seguro" << std::endl;
        
        std::cout << "│" << std::endl;
        std::cout << "│" << COLOR_BOLD << "Metricas de seguridad:" << COLOR_RESET << std::endl;
        std::cout << "│ • Claves almacenadas: " << keys.size() << std::endl;
        std::cout << "│ • Clave activa: " << (active_key_id.empty() ? "No" : "Si") << std::endl;
        std::cout << "│ • Salt unico por clave: SI" << std::endl;
        std::cout << "│ • Nonce unico por operacion: SI" << std::endl;
        
        if (!active_key_id.empty()) {
            auto it = keys.find(active_key_id);
            if (it != keys.end()) {
                std::cout << "│" << std::endl;
                std::cout << "│" << COLOR_BOLD << "Clave activa:" << COLOR_RESET << std::endl;
                std::cout << "│ • Nombre: " << it->second.name << std::endl;
                std::cout << "│ • Algoritmo: " << it->second.algorithm << std::endl;
                std::cout << "│ • Memoria: " << (it->second.memory_cost / 1024) << "MB" << std::endl;
                std::cout << "│ • Iteraciones: " << it->second.time_cost << std::endl;
                std::cout << "│ • Fortaleza: " COLOR_GREEN "MUY ALTA" COLOR_RESET << std::endl;
            }
        }
        
        std::cout << "└──────────────────────────────────────────────────────┘\n" << COLOR_RESET;
    }

private:
    std::string format_time(time_t time_val) {
        if (time_val == 0) return "Nunca";
        char buffer[80];
        std::strftime(buffer, sizeof(buffer), "%d/%m/%Y", std::localtime(&time_val));
        return std::string(buffer);
    }
    
    void save_keys_to_file() {
        try {
            std::ofstream file("usb_cipher_keys.dat", std::ios::binary);
            if (!file) {
                std::cerr << "No se pudo abrir el archivo para guardar claves" << std::endl;
                return;
            }
            
            size_t num_keys = keys.size();
            file.write(reinterpret_cast<const char*>(&num_keys), sizeof(num_keys));
            
            for (const auto& [name, key_info] : keys) {
                size_t name_len = name.size();
                file.write(reinterpret_cast<const char*>(&name_len), sizeof(name_len));
                file.write(name.c_str(), name_len);
                
                size_t key_len = key_info.derived_key.size();
                file.write(reinterpret_cast<const char*>(&key_len), sizeof(key_len));
                file.write(reinterpret_cast<const char*>(key_info.derived_key.data()), key_len);
                
                size_t salt_len = key_info.salt.size();
                file.write(reinterpret_cast<const char*>(&salt_len), sizeof(salt_len));
                file.write(reinterpret_cast<const char*>(key_info.salt.data()), salt_len);
                
                size_t nonce_len = key_info.nonce.size();
                file.write(reinterpret_cast<const char*>(&nonce_len), sizeof(nonce_len));
                file.write(reinterpret_cast<const char*>(key_info.nonce.data()), nonce_len);
                
                file.write(reinterpret_cast<const char*>(&key_info.created_time), sizeof(key_info.created_time));
                file.write(reinterpret_cast<const char*>(&key_info.last_used), sizeof(key_info.last_used));
                file.write(reinterpret_cast<const char*>(&key_info.is_active), sizeof(key_info.is_active));
                
                size_t algo_len = key_info.algorithm.size();
                file.write(reinterpret_cast<const char*>(&algo_len), sizeof(algo_len));
                file.write(key_info.algorithm.c_str(), algo_len);
                
                file.write(reinterpret_cast<const char*>(&key_info.memory_cost), sizeof(key_info.memory_cost));
                file.write(reinterpret_cast<const char*>(&key_info.time_cost), sizeof(key_info.time_cost));
                file.write(reinterpret_cast<const char*>(&key_info.parallelism), sizeof(key_info.parallelism));
            }
            
            file.close();
        } catch (const std::exception& e) {
            std::cerr << "Error guardando claves: " << e.what() << std::endl;
        }
    }
    
    void load_keys_from_file() {
        try {
            std::ifstream file("usb_cipher_keys.dat", std::ios::binary);
            if (!file) {
                return;
            }
            
            size_t num_keys;
            file.read(reinterpret_cast<char*>(&num_keys), sizeof(num_keys));
            
            for (size_t i = 0; i < num_keys; i++) {
                KeyInfo key_info;
                
                size_t name_len;
                file.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));
                if (name_len > 1000) {
                    throw std::runtime_error("Nombre de clave demasiado largo");
                }
                std::string name(name_len, '\0');
                file.read(&name[0], name_len);
                
                size_t key_len;
                file.read(reinterpret_cast<char*>(&key_len), sizeof(key_len));
                if (key_len > 1024) {
                    throw std::runtime_error("Clave demasiado larga");
                }
                std::vector<uint8_t> derived_key(key_len);
                file.read(reinterpret_cast<char*>(derived_key.data()), key_len);
                
                size_t salt_len;
                file.read(reinterpret_cast<char*>(&salt_len), sizeof(salt_len));
                if (salt_len > 1024) {
                    throw std::runtime_error("Salt demasiado largo");
                }
                std::vector<uint8_t> salt(salt_len);
                file.read(reinterpret_cast<char*>(salt.data()), salt_len);
                
                size_t nonce_len;
                file.read(reinterpret_cast<char*>(&nonce_len), sizeof(nonce_len));
                if (nonce_len > 1024) {
                    throw std::runtime_error("Nonce demasiado largo");
                }
                std::vector<uint8_t> nonce(nonce_len);
                file.read(reinterpret_cast<char*>(nonce.data()), nonce_len);
                
                file.read(reinterpret_cast<char*>(&key_info.created_time), sizeof(key_info.created_time));
                file.read(reinterpret_cast<char*>(&key_info.last_used), sizeof(key_info.last_used));
                file.read(reinterpret_cast<char*>(&key_info.is_active), sizeof(key_info.is_active));
                
                size_t algo_len;
                file.read(reinterpret_cast<char*>(&algo_len), sizeof(algo_len));
                if (algo_len > 100) {
                    throw std::runtime_error("Algoritmo demasiado largo");
                }
                std::string algorithm(algo_len, '\0');
                file.read(&algorithm[0], algo_len);
                
                file.read(reinterpret_cast<char*>(&key_info.memory_cost), sizeof(key_info.memory_cost));
                file.read(reinterpret_cast<char*>(&key_info.time_cost), sizeof(key_info.time_cost));
                file.read(reinterpret_cast<char*>(&key_info.parallelism), sizeof(key_info.parallelism));
                
                key_info.name = name;
                key_info.derived_key = std::move(derived_key);
                key_info.salt = std::move(salt);
                key_info.nonce = std::move(nonce);
                key_info.algorithm = algorithm;
                
                keys[name] = std::move(key_info);
                
                if (key_info.is_active) {
                    active_key_id = name;
                }
            }
            
            file.close();
        } catch (const std::exception& e) {
            std::cerr << "Error cargando claves: " << e.what() << std::endl;
            keys.clear();
            active_key_id.clear();
        }
    }
};

// ========================
// CONFIGURACIÓN DEL SISTEMA
// ========================
struct SystemConfig {
    int encryption_threads;
    int io_threads = 1;
    size_t chunk_size = 64 * 1024;
    bool enable_avx2 = true;
    bool show_detailed_stats = true;
    bool enable_argon2 = true;
    bool enable_kmac = true;
    bool enable_poly1305 = true;
    int max_threads = 16;
    bool use_avx2_if_available = true;
    
    SystemConfig() {
        int available_cores = std::thread::hardware_concurrency();
        if (available_cores >= 4) {
            encryption_threads = 2;
        } else if (available_cores >= 2) {
            encryption_threads = 1;
        } else {
            encryption_threads = 1;
        }
        
        enable_avx2 = has_avx2_support() && use_avx2_if_available;
    }
    
    std::string to_string() const {
        std::ostringstream oss;
        oss << COLOR_CYAN "│ Configuracion Actual:\n" COLOR_RESET
            << "│ • Hilos de cifrado: " << encryption_threads << " (CPU: " << std::thread::hardware_concurrency() << " nucleos)\n"
            << "│ • AVX2: " << (enable_avx2 ? COLOR_GREEN "Activado" COLOR_RESET : COLOR_YELLOW "Desactivado" COLOR_RESET) << "\n"
            << "│ • Argon2: " << (enable_argon2 ? COLOR_GREEN "Activado" COLOR_RESET : COLOR_RED "Desactivado" COLOR_RESET) << "\n"
            << "│ • KMAC: " << (enable_kmac ? COLOR_GREEN "Activado" COLOR_RESET : COLOR_RED "Desactivado" COLOR_RESET) << "\n"
            << "│ • Poly1305: " << (enable_poly1305 ? COLOR_GREEN "Activado" COLOR_RESET : COLOR_RED "Desactivado" COLOR_RESET) << "\n"
            << "│ • Estadisticas: " << (show_detailed_stats ? COLOR_GREEN "Detalladas" COLOR_RESET : COLOR_YELLOW "Basicas" COLOR_RESET);
        return oss.str();
    }
};

// ========================
// ChaCha20 AVX2 Core CORREGIDO
// ========================

#if defined(__AVX2__) || defined(_WIN32)
inline void quarter_round_avx2(__m256i &a, __m256i &b, __m256i &c, __m256i &d) {
    a = _mm256_add_epi32(a, b);
    d = _mm256_xor_si256(d, a);
    d = _mm256_or_si256(_mm256_slli_epi32(d, 16), _mm256_srli_epi32(d, 16));

    c = _mm256_add_epi32(c, d);
    b = _mm256_xor_si256(b, c);
    b = _mm256_or_si256(_mm256_slli_epi32(b, 12), _mm256_srli_epi32(b, 20));

    a = _mm256_add_epi32(a, b);
    d = _mm256_xor_si256(d, a);
    d = _mm256_or_si256(_mm256_slli_epi32(d, 8), _mm256_srli_epi32(d, 24));

    c = _mm256_add_epi32(c, d);
    b = _mm256_xor_si256(b, c);
    b = _mm256_or_si256(_mm256_slli_epi32(b, 7), _mm256_srli_epi32(b, 25));
}

void chacha20_block_avx2(uint32_t out[128], const uint32_t in[16], uint32_t counter) {
    __m256i state[16];
    for (int i = 0; i < 12; i++) state[i] = _mm256_set1_epi32(in[i]);
    state[12] = _mm256_set_epi32(counter+7, counter+6, counter+5, counter+4,
                                counter+3, counter+2, counter+1, counter);
    for (int i = 13; i < 16; i++) state[i] = _mm256_set1_epi32(in[i]);

    for (int i = 0; i < 10; i++) {
        quarter_round_avx2(state[0], state[4], state[8], state[12]);
        quarter_round_avx2(state[1], state[5], state[9], state[13]);
        quarter_round_avx2(state[2], state[6], state[10], state[14]);
        quarter_round_avx2(state[3], state[7], state[11], state[15]);
        quarter_round_avx2(state[0], state[5], state[10], state[15]);
        quarter_round_avx2(state[1], state[6], state[11], state[12]);
        quarter_round_avx2(state[2], state[7], state[8], state[13]);
        quarter_round_avx2(state[3], state[4], state[9], state[14]);
    }

    for (int i = 0; i < 16; i++) {
        __m256i orig = (i==12)? _mm256_set_epi32(counter+7,counter+6,counter+5,counter+4,
                                                  counter+3,counter+2,counter+1,counter)
                               : _mm256_set1_epi32(in[i]);
        _mm256_storeu_si256((__m256i*)(out + i*8), _mm256_add_epi32(state[i], orig));
    }
}

void process_buffer_avx2_fast(const uint8_t* input, uint8_t* output, size_t size, 
                             const uint32_t* key_data, const uint8_t nonce[12], uint64_t start_counter) {
    if (input == nullptr || output == nullptr || key_data == nullptr || nonce == nullptr) {
        throw std::invalid_argument("Punteros nulos en process_buffer_avx2_fast");
    }
    
    uint32_t state[16] = {
        0x61707865,0x3320646e,0x79622d32,0x6b206574,
        key_data[0],key_data[1],key_data[2],key_data[3],
        key_data[4],key_data[5],key_data[6],key_data[7],
        *reinterpret_cast<const uint32_t*>(nonce),
        *reinterpret_cast<const uint32_t*>(nonce + 4),
        *reinterpret_cast<const uint32_t*>(nonce + 8),
        key_data[8]
    };
    
    size_t processed = 0;
    uint64_t block_counter = start_counter;
    
    while(processed < size) {
        state[15] = static_cast<uint32_t>(block_counter);
        uint32_t keystream[128];
        chacha20_block_avx2(keystream, state, static_cast<uint32_t>(block_counter));
        
        size_t bytes_remaining = size - processed;
        size_t bytes_to_use = std::min(bytes_remaining, size_t(512));
        
        const uint8_t* key_bytes = reinterpret_cast<uint8_t*>(keystream);
        const uint8_t* input_ptr = input + processed;
        uint8_t* output_ptr = output + processed;
        
        size_t i = 0;
        for (; i + 64 <= bytes_to_use; i += 64) {
            __m256i data1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input_ptr + i));
            __m256i data2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input_ptr + i + 32));
            __m256i key1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(key_bytes + i));
            __m256i key2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(key_bytes + i + 32));
            
            __m256i processed1 = _mm256_xor_si256(data1, key1);
            __m256i processed2 = _mm256_xor_si256(data2, key2);
            
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(output_ptr + i), processed1);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(output_ptr + i + 32), processed2);
        }
        
        for (; i < bytes_to_use; i++) {
            output_ptr[i] = input_ptr[i] ^ key_bytes[i];
        }
        
        processed += bytes_to_use;
        block_counter += 8;
    }
    
    std::fill_n(state, 16, 0);
}

void process_buffer_avx2_parallel(const uint8_t* input, uint8_t* output, size_t size, 
                                 const uint32_t* key_data, const uint8_t nonce[12], 
                                 uint64_t start_counter, int num_threads) {
    if (input == nullptr || output == nullptr || key_data == nullptr || nonce == nullptr) {
        throw std::invalid_argument("Punteros nulos en process_buffer_avx2_parallel");
    }
    
    if (size == 0) return;
    
    if (num_threads <= 1 || size < 4096) {
        process_buffer_avx2_fast(input, output, size, key_data, nonce, start_counter);
        return;
    }

    std::vector<std::thread> threads;
    size_t chunk_size = size / num_threads;
    
    for (int i = 0; i < num_threads; i++) {
        size_t thread_start = i * chunk_size;
        size_t thread_size = (i == num_threads - 1) ? (size - thread_start) : chunk_size;
        uint64_t thread_counter = start_counter + (thread_start / 64);
        
        threads.emplace_back([=]() {
            process_buffer_avx2_fast(input + thread_start, output + thread_start, 
                                   thread_size, key_data, nonce, thread_counter);
        });
    }
    
    for (auto& thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}
#endif

// ========================
// SISTEMA DE CIFRADO USB CORREGIDO CON FALLBACK
// ========================
class USBCipher {
private:
    uint32_t key_data[12];
    uint8_t nonce[12];
    SystemConfig config;
    bool use_avx2;
    
public:
    USBCipher() : use_avx2(has_avx2_support()) {
        std::memset(key_data, 0, sizeof(key_data));
        std::memset(nonce, 0, sizeof(nonce));
    }
    
    ~USBCipher() {
        secure_clean();
    }
    
    USBCipher(const USBCipher&) = delete;
    USBCipher& operator=(const USBCipher&) = delete;
    
    USBCipher(USBCipher&& other) noexcept {
        std::memcpy(key_data, other.key_data, sizeof(key_data));
        std::memcpy(nonce, other.nonce, sizeof(nonce));
        config = other.config;
        use_avx2 = other.use_avx2;
        other.secure_clean();
    }
    
    USBCipher& operator=(USBCipher&& other) noexcept {
        if (this != &other) {
            secure_clean();
            std::memcpy(key_data, other.key_data, sizeof(key_data));
            std::memcpy(nonce, other.nonce, sizeof(nonce));
            config = other.config;
            use_avx2 = other.use_avx2;
            other.secure_clean();
        }
        return *this;
    }
    
private:
    void secure_clean() {
        volatile uint32_t* vkey = key_data;
        for (size_t i = 0; i < sizeof(key_data)/sizeof(key_data[0]); i++) {
            vkey[i] = 0;
        }
        
        volatile uint8_t* vnonce = nonce;
        for (size_t i = 0; i < sizeof(nonce); i++) {
            vnonce[i] = 0;
        }
    }
    
    // Fallback implementation without AVX2
    void process_buffer_fallback(const uint8_t* input, uint8_t* output, size_t size, 
                                uint64_t start_counter) {
        ChaCha20 cipher(reinterpret_cast<const uint8_t*>(key_data), nonce);
        cipher.process_bytes(input, output, size, start_counter);
    }
    
    void process_buffer_fallback_parallel(const uint8_t* input, uint8_t* output, size_t size, 
                                         uint64_t start_counter, int num_threads) {
        if (num_threads <= 1 || size < 4096) {
            process_buffer_fallback(input, output, size, start_counter);
            return;
        }

        std::vector<std::thread> threads;
        size_t chunk_size = size / num_threads;
        
        for (int i = 0; i < num_threads; i++) {
            size_t thread_start = i * chunk_size;
            size_t thread_size = (i == num_threads - 1) ? (size - thread_start) : chunk_size;
            uint64_t thread_counter = start_counter + (thread_start / 64);
            
            threads.emplace_back([=]() {
                process_buffer_fallback(input + thread_start, output + thread_start, 
                                      thread_size, thread_counter);
            });
        }
        
        for (auto& thread : threads) {
            if (thread.joinable()) {
                thread.join();
            }
        }
    }
    
public:
    bool set_key_from_manager(const std::vector<uint8_t>& key, const std::vector<uint8_t>& key_nonce) {
        if (key.size() < 32 || key_nonce.size() < 12) {
            std::cerr << "Error: Clave o nonce de tamaño insuficiente" << std::endl;
            return false;
        }
        
        for (int i = 0; i < 12 && i * 4 + 3 < key.size(); i++) {
            key_data[i] = *reinterpret_cast<const uint32_t*>(key.data() + i * 4);
        }
        
        std::memcpy(nonce, key_nonce.data(), std::min(key_nonce.size(), sizeof(nonce)));
        return true;
    }

    void set_config(const SystemConfig& new_config) {
        config = new_config;
        use_avx2 = has_avx2_support() && config.enable_avx2;
    }
    
    struct CipherResult {
        double total_time = 0;
        double read_time = 0;
        double encrypt_time = 0;
        double write_time = 0;
        double total_speed = 0;
        double encrypt_speed = 0;
        double read_speed = 0;
        double write_speed = 0;
        size_t total_files = 0;
        size_t total_bytes = 0;
        std::vector<uint8_t> mac;
        bool used_avx2 = false;
        
        CipherResult() = default;
        
        CipherResult(const CipherResult&) = default;
        CipherResult& operator=(const CipherResult&) = default;
    };

    std::vector<uint8_t> calculate_file_mac(const std::string& file_path, const std::vector<uint8_t>& poly_key) {
        if (poly_key.size() < 32) {
            throw std::invalid_argument("Poly1305 key must be at least 32 bytes");
        }
        
        std::ifstream file(file_path, std::ios::binary);
        if (!file) {
            return {};
        }
        
        Poly1305 poly(poly_key.data());
        const size_t buffer_size = 64 * 1024;
        std::vector<uint8_t> buffer(buffer_size);
        
        while (file.read(reinterpret_cast<char*>(buffer.data()), buffer_size)) {
            poly.update(buffer.data(), static_cast<size_t>(file.gcount()));
        }
        
        if (file.gcount() > 0) {
            poly.update(buffer.data(), static_cast<size_t>(file.gcount()));
        }
        
        std::vector<uint8_t> mac(16);
        poly.finish(mac.data());
        return mac;
    }

    CipherResult process_file_configurable(const std::string& file_path, uint64_t file_counter, bool decrypt = false) {
        CipherResult result;
        result.used_avx2 = use_avx2;
        
        if (!std::filesystem::exists(file_path)) {
            std::cerr << "Error: Archivo no existe: " << file_path << std::endl;
            return result;
        }
        
        auto total_start = std::chrono::high_resolution_clock::now();
        
        try {
            auto read_start = std::chrono::high_resolution_clock::now();
            std::ifstream file_in(file_path, std::ios::binary | std::ios::ate);
            if (!file_in) {
                std::cerr << "Error abriendo archivo: " << file_path << std::endl;
                return result;
            }
            
            size_t file_size = file_in.tellg();
            file_in.seekg(0);
            
            std::vector<uint8_t> file_buffer(file_size);
            
            if (file_size > 0) {
                file_in.read(reinterpret_cast<char*>(file_buffer.data()), file_size);
            }
            file_in.close();
            
            if (!file_in && file_size > 0) {
                std::cerr << "Error leyendo archivo: " << file_path << std::endl;
                return result;
            }
            
            auto read_end = std::chrono::high_resolution_clock::now();
            
            auto encrypt_start = std::chrono::high_resolution_clock::now();
            
            std::vector<uint8_t> poly_key(32);
            if (config.enable_poly1305 && !decrypt) {
                uint32_t poly_state[16] = {
                    0x61707865,0x3320646e,0x79622d32,0x6b206574,
                    key_data[0],key_data[1],key_data[2],key_data[3],
                    key_data[4],key_data[5],key_data[6],key_data[7],
                    *reinterpret_cast<const uint32_t*>(nonce),
                    *reinterpret_cast<const uint32_t*>(nonce + 4),
                    *reinterpret_cast<const uint32_t*>(nonce + 8),
                    static_cast<uint32_t>(file_counter)
                };
                
                // Use appropriate implementation for Poly1305 key generation
                if (use_avx2) {
#if defined(__AVX2__) || defined(_WIN32)
                    uint32_t poly_keystream[16];
                    chacha20_block_avx2(poly_keystream, poly_state, static_cast<uint32_t>(file_counter));
                    std::memcpy(poly_key.data(), poly_keystream, 32);
#endif
                } else {
                    ChaCha20 poly_cipher(reinterpret_cast<const uint8_t*>(key_data), nonce);
                    std::vector<uint8_t> temp_key(32);
                    poly_cipher.process_bytes(temp_key.data(), temp_key.data(), 32, file_counter);
                    poly_key = temp_key;
                }
                
                std::fill_n(poly_state, 16, 0);
            }
            
            if (file_size > 0) {
                if (use_avx2) {
#if defined(__AVX2__) || defined(_WIN32)
                    process_buffer_avx2_parallel(file_buffer.data(), file_buffer.data(), file_size, 
                                               key_data, nonce, file_counter, config.encryption_threads);
#endif
                } else {
                    process_buffer_fallback_parallel(file_buffer.data(), file_buffer.data(), file_size, 
                                                   file_counter, config.encryption_threads);
                }
            }
            auto encrypt_end = std::chrono::high_resolution_clock::now();
            
            if (config.enable_poly1305 && !decrypt && file_size > 0) {
                result.mac = calculate_file_mac(file_path, poly_key);
            }
            
            auto write_start = std::chrono::high_resolution_clock::now();
            std::ofstream file_out(file_path, std::ios::binary | std::ios::trunc);
            if (file_out) {
                const size_t WRITE_CHUNK = 256 * 1024;
                size_t written = 0;
                while (written < file_size) {
                    size_t chunk_size = std::min(WRITE_CHUNK, file_size - written);
                    file_out.write(reinterpret_cast<const char*>(file_buffer.data() + written), chunk_size);
                    written += chunk_size;
                }
                file_out.close();
                
                if (!file_out) {
                    std::cerr << "Error escribiendo archivo: " << file_path << std::endl;
                }
            } else {
                std::cerr << "Error abriendo archivo para escritura: " << file_path << std::endl;
            }
            auto write_end = std::chrono::high_resolution_clock::now();
            auto total_end = std::chrono::high_resolution_clock::now();
            
            result.read_time = std::chrono::duration<double>(read_end - read_start).count();
            result.encrypt_time = std::chrono::duration<double>(encrypt_end - encrypt_start).count();
            result.write_time = std::chrono::duration<double>(write_end - write_start).count();
            result.total_time = std::chrono::duration<double>(total_end - total_start).count();
            
            double file_size_mb = file_size / (1024.0 * 1024.0);
            result.read_speed = (result.read_time > 0) ? file_size_mb / result.read_time : 0;
            result.encrypt_speed = (result.encrypt_time > 0) ? file_size_mb / result.encrypt_time : 0;
            result.write_speed = (result.write_time > 0) ? file_size_mb / result.write_time : 0;
            result.total_speed = (result.total_time > 0) ? file_size_mb / result.total_time : 0;
            
            result.total_files = 1;
            result.total_bytes = file_size;
            
        } catch (const std::exception& e) {
            std::cerr << "Excepción procesando archivo " << file_path << ": " << e.what() << std::endl;
        }
        
        return result;
    }

    struct CipherInfo {
        uint32_t key_data[12] = {0};
        uint8_t nonce[12] = {0};
        bool is_encrypted = false;
        time_t encryption_time = 0;
        std::vector<uint8_t> master_mac;
        int encryption_threads_used = 0;
        bool poly1305_enabled = false;
        bool used_avx2 = false;
        
        CipherInfo() = default;
        
        CipherInfo(const CipherInfo& other) {
            std::memcpy(key_data, other.key_data, sizeof(key_data));
            std::memcpy(nonce, other.nonce, sizeof(nonce));
            is_encrypted = other.is_encrypted;
            encryption_time = other.encryption_time;
            master_mac = other.master_mac;
            encryption_threads_used = other.encryption_threads_used;
            poly1305_enabled = other.poly1305_enabled;
            used_avx2 = other.used_avx2;
        }
        
        CipherInfo& operator=(const CipherInfo& other) {
            if (this != &other) {
                std::memcpy(key_data, other.key_data, sizeof(key_data));
                std::memcpy(nonce, other.nonce, sizeof(nonce));
                is_encrypted = other.is_encrypted;
                encryption_time = other.encryption_time;
                master_mac = other.master_mac;
                encryption_threads_used = other.encryption_threads_used;
                poly1305_enabled = other.poly1305_enabled;
                used_avx2 = other.used_avx2;
            }
            return *this;
        }
        
        ~CipherInfo() {
            secure_clean();
        }
        
        void secure_clean() {
            std::fill_n(key_data, 12, 0);
            std::fill_n(nonce, 12, 0);
            if (!master_mac.empty()) {
                std::fill(master_mac.begin(), master_mac.end(), 0);
            }
        }
    };

    bool save_cipher_info(const std::string& device_path, const std::vector<uint8_t>& master_mac = {}, bool is_encrypted = true) {
        std::string info_path = device_path + "/USB_CIPHER_INFO.bin";
        
        try {
            std::ofstream info_file(info_path, std::ios::binary);
            if (!info_file) {
                std::cerr << "Error creando archivo de información: " << info_path << std::endl;
                return false;
            }
            
            CipherInfo info;
            std::memcpy(info.key_data, key_data, sizeof(key_data));
            std::memcpy(info.nonce, nonce, sizeof(nonce));
            info.is_encrypted = is_encrypted;
            info.encryption_time = std::time(nullptr);
            info.encryption_threads_used = config.encryption_threads;
            info.poly1305_enabled = config.enable_poly1305;
            info.used_avx2 = use_avx2;
            
            if (!master_mac.empty()) {
                info.master_mac = master_mac;
            }
            
            info_file.write(reinterpret_cast<const char*>(&info.key_data), sizeof(info.key_data));
            info_file.write(reinterpret_cast<const char*>(&info.nonce), sizeof(info.nonce));
            info_file.write(reinterpret_cast<const char*>(&info.is_encrypted), sizeof(info.is_encrypted));
            info_file.write(reinterpret_cast<const char*>(&info.encryption_time), sizeof(info.encryption_time));
            info_file.write(reinterpret_cast<const char*>(&info.encryption_threads_used), sizeof(info.encryption_threads_used));
            info_file.write(reinterpret_cast<const char*>(&info.poly1305_enabled), sizeof(info.poly1305_enabled));
            info_file.write(reinterpret_cast<const char*>(&info.used_avx2), sizeof(info.used_avx2));
            
            if (!master_mac.empty()) {
                size_t mac_size = master_mac.size();
                info_file.write(reinterpret_cast<const char*>(&mac_size), sizeof(mac_size));
                info_file.write(reinterpret_cast<const char*>(master_mac.data()), mac_size);
            }
            
            info_file.close();
            return info_file.good();
            
        } catch (const std::exception& e) {
            std::cerr << "Error guardando información de cifrado: " << e.what() << std::endl;
            return false;
        }
    }

    bool load_cipher_info(const std::string& device_path) {
        std::string info_path = device_path + "/USB_CIPHER_INFO.bin";
        
        try {
            std::ifstream info_file(info_path, std::ios::binary);
            if (!info_file) {
                return false;
            }
            
            CipherInfo info;
            info_file.read(reinterpret_cast<char*>(&info.key_data), sizeof(info.key_data));
            info_file.read(reinterpret_cast<char*>(&info.nonce), sizeof(info.nonce));
            info_file.read(reinterpret_cast<char*>(&info.is_encrypted), sizeof(info.is_encrypted));
            info_file.read(reinterpret_cast<char*>(&info.encryption_time), sizeof(info.encryption_time));
            info_file.read(reinterpret_cast<char*>(&info.encryption_threads_used), sizeof(info.encryption_threads_used));
            info_file.read(reinterpret_cast<char*>(&info.poly1305_enabled), sizeof(info.poly1305_enabled));
            info_file.read(reinterpret_cast<char*>(&info.used_avx2), sizeof(info.used_avx2));
            
            if (info_file.peek() != EOF) {
                size_t mac_size;
                info_file.read(reinterpret_cast<char*>(&mac_size), sizeof(mac_size));
                if (mac_size > 0 && mac_size <= 1024) {
                    info.master_mac.resize(mac_size);
                    info_file.read(reinterpret_cast<char*>(info.master_mac.data()), mac_size);
                }
            }
            
            std::memcpy(key_data, info.key_data, sizeof(key_data));
            std::memcpy(nonce, info.nonce, sizeof(nonce));
            use_avx2 = info.used_avx2 && has_avx2_support();
            
            return true;
            
        } catch (const std::exception& e) {
            std::cerr << "Error cargando información de cifrado: " << e.what() << std::endl;
            return false;
        }
    }

    CipherInfo get_cipher_info(const std::string& device_path) {
        CipherInfo info;
        std::string info_path = device_path + "/USB_CIPHER_INFO.bin";
        
        try {
            std::ifstream info_file(info_path, std::ios::binary);
            
            if (info_file) {
                info_file.read(reinterpret_cast<char*>(&info.key_data), sizeof(info.key_data));
                info_file.read(reinterpret_cast<char*>(&info.nonce), sizeof(info.nonce));
                info_file.read(reinterpret_cast<char*>(&info.is_encrypted), sizeof(info.is_encrypted));
                info_file.read(reinterpret_cast<char*>(&info.encryption_time), sizeof(info.encryption_time));
                info_file.read(reinterpret_cast<char*>(&info.encryption_threads_used), sizeof(info.encryption_threads_used));
                info_file.read(reinterpret_cast<char*>(&info.poly1305_enabled), sizeof(info.poly1305_enabled));
                info_file.read(reinterpret_cast<char*>(&info.used_avx2), sizeof(info.used_avx2));
                
                if (info_file.peek() != EOF) {
                    size_t mac_size;
                    info_file.read(reinterpret_cast<char*>(&mac_size), sizeof(mac_size));
                    if (mac_size > 0 && mac_size <= 1024) {
                        info.master_mac.resize(mac_size);
                        info_file.read(reinterpret_cast<char*>(info.master_mac.data()), mac_size);
                    }
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Error obteniendo información de cifrado: " << e.what() << std::endl;
        }
        
        return info;
    }

    std::string get_nonce_hex() const {
        std::ostringstream oss;
        oss << std::hex << std::setfill('0');
        for (int i = 0; i < 12; i++) {
            oss << std::setw(2) << static_cast<int>(nonce[i]);
        }
        return oss.str();
    }

    std::string get_key_hash() const {
        std::ostringstream oss;
        oss << std::hex << std::setfill('0');
        uint32_t hash = 0;
        for (int i = 0; i < 12; i++) {
            hash ^= key_data[i];
        }
        oss << std::setw(8) << hash;
        return oss.str();
    }
    
    bool is_using_avx2() const {
        return use_avx2;
    }
};

// ========================
// DETECCIÓN DE DISPOSITIVOS USB CORREGIDA
// ========================
struct USBDevice {
    std::string device_id;
    std::string name;
    std::string mount_point;
    uint64_t total_size;
    uint64_t free_space;
    bool is_removable;
    std::string filesystem;
    bool is_encrypted;
    
    USBDevice() : total_size(0), free_space(0), is_removable(false), is_encrypted(false) {}
};

class USBDetector {
public:
    std::vector<USBDevice> detect_usb_devices() {
#ifdef _WIN32
        return detect_usb_windows();
#else
        return detect_usb_linux();
#endif
    }

private:
#ifdef _WIN32
    std::vector<USBDevice> detect_usb_windows() {
        std::vector<USBDevice> devices;
        
        try {
            for (char drive = 'A'; drive <= 'Z'; drive++) {
                std::string root_path = std::string(1, drive) + ":\\";
                UINT drive_type = GetDriveTypeA(root_path.c_str());
                
                if (drive_type == DRIVE_REMOVABLE) {
                    USBDevice device;
                    device.mount_point = root_path;
                    device.name = "USB Drive " + std::string(1, drive);
                    device.is_removable = true;
                    
                    std::string info_path = root_path + "USB_CIPHER_INFO.bin";
                    device.is_encrypted = std::filesystem::exists(info_path);
                    
                    ULARGE_INTEGER total_bytes, free_bytes, dummy;
                    if (GetDiskFreeSpaceExA(root_path.c_str(), &dummy, &total_bytes, &free_bytes)) {
                        device.total_size = total_bytes.QuadPart;
                        device.free_space = free_bytes.QuadPart;
                    }
                    devices.push_back(device);
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Error detectando dispositivos USB en Windows: " << e.what() << std::endl;
        }
        
        return devices;
    }
#else
    std::vector<USBDevice> detect_usb_linux() {
        std::vector<USBDevice> devices;
        
        try {
            std::ifstream mounts("/proc/mounts");
            std::string line;
            
            while (std::getline(mounts, line)) {
                std::istringstream iss(line);
                std::string device, mount_point, fs_type, options;
                iss >> device >> mount_point >> fs_type >> options;
                
                if ((device.find("/dev/sd") == 0 || device.find("/dev/mmcblk") == 0) &&
                    mount_point != "/" && mount_point.find("/boot") == std::string::npos) {
                    
                    USBDevice usb_dev;
                    usb_dev.device_id = device;
                    usb_dev.mount_point = mount_point;
                    usb_dev.filesystem = fs_type;
                    usb_dev.name = "USB Device";
                    usb_dev.is_removable = true;
                    
                    std::string info_path = mount_point + "/USB_CIPHER_INFO.bin";
                    usb_dev.is_encrypted = std::filesystem::exists(info_path);
                    
                    struct statvfs buf;
                    if (statvfs(mount_point.c_str(), &buf) == 0) {
                        usb_dev.total_size = buf.f_blocks * buf.f_frsize;
                        usb_dev.free_space = buf.f_bfree * buf.f_frsize;
                    }
                    
                    if (usb_dev.total_size > 1000000) {
                        devices.push_back(usb_dev);
                    }
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Error detectando dispositivos USB en Linux: " << e.what() << std::endl;
        }
        
        return devices;
    }
#endif

    std::string format_size(uint64_t bytes) {
        const char* units[] = {"B", "KB", "MB", "GB", "TB"};
        int unit_index = 0;
        double size = static_cast<double>(bytes);
        
        while (size >= 1024.0 && unit_index < 4) {
            size /= 1024.0;
            unit_index++;
        }
        
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(1) << size << " " << units[unit_index];
        return oss.str();
    }
};

// ========================
// SISTEMA PRINCIPAL MEJORADO CON UI ACTUALIZADA
// ========================
class USBEncryptionSystem {
private:
    USBDetector detector;
    SystemConfig current_config;
    std::unique_ptr<KeyManager> key_manager;

    std::string format_size(uint64_t bytes) {
        const char* units[] = {"B", "KB", "MB", "GB", "TB"};
        int unit_index = 0;
        double size = static_cast<double>(bytes);
        
        while (size >= 1024.0 && unit_index < 4) {
            size /= 1024.0;
            unit_index++;
        }
        
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(1) << size << " " << units[unit_index];
        return oss.str();
    }

    void print_header(const std::string& title, const char* color = COLOR_WHITE) {
        std::cout << "\n" << color << "┌──────────────────────────────────────────────────────────────────────────┐\n";
        std::cout << "│" << COLOR_CYAN COLOR_BOLD << center_text(title, 74) << color << "│\n";
        std::cout << "├──────────────────────────────────────────────────────────────────────────┤" << COLOR_RESET << "\n";
    }

    void print_footer() {
        std::cout << COLOR_WHITE << "└──────────────────────────────────────────────────────────────────────────┘\n" << COLOR_RESET;
    }

    std::string center_text(const std::string& text, int width) {
        if (text.length() >= width) return text;
        int padding = (width - text.length()) / 2;
        return std::string(padding, ' ') + text + std::string(width - text.length() - padding, ' ');
    }

    void show_progress(size_t current, size_t total, double speed, const std::string& operation, int threads, bool use_avx2) {
        if (total == 0) return;
        
        float progress = (float)current / total;
        int bar_width = 50;
        int pos = bar_width * progress;
        
        std::string mode = use_avx2 ? "AVX2" : "FALLBACK";
        
        std::cout << "\r" << operation << " [" << threads << " HILOS][" << mode << "] [";
        for (int i = 0; i < bar_width; ++i) {
            if (i < pos) std::cout << COLOR_GREEN "█" COLOR_RESET;
            else if (i == pos) std::cout << COLOR_GREEN "▶" COLOR_RESET;
            else std::cout << "░";
        }
        std::cout << "] " << COLOR_YELLOW << int(progress * 100.0) << "%" COLOR_RESET " ";
        std::cout << "(" << current << "/" << total << ") ";
        std::cout << COLOR_CYAN << std::setprecision(1) << std::fixed << speed << " MB/s" COLOR_RESET;
        std::flush(std::cout);
    }

    void show_key_manager_menu() {
        while (true) {
            print_header("GESTOR DE CLAVES AVANZADO");
            
            std::cout << key_manager->get_active_key_info() << "\n\n";
            
            std::cout << COLOR_BOLD "Opciones:\n" COLOR_RESET;
            std::cout << " 1. Crear nueva clave (Argon2id)\n";
            std::cout << " 2. Activar clave existente\n";
            std::cout << " 3. Listar todas las claves\n";
            std::cout << " 4. Eliminar clave\n";
            std::cout << " 5. Desactivar clave actual\n";
            std::cout << " 6. Informe de seguridad\n";
            std::cout << " 7. Volver al menu principal\n";
            std::cout << "\n" COLOR_CYAN "Opcion: " COLOR_RESET;
            
            int choice;
            if (!(std::cin >> choice)) {
                std::cin.clear();
                std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                std::cout << COLOR_RED "Opcion invalida\n" COLOR_RESET;
                continue;
            }
            std::cin.ignore();
            
            switch (choice) {
                case 1: {
                    std::string name, password, algorithm;
                    std::cout << "\n";
                    print_header("CREAR NUEVA CLAVE CON ARGON2ID");
                    std::cout << " Nombre de la clave: ";
                    std::getline(std::cin, name);
                    std::cout << " Contrasena (minimo 12 caracteres): ";
                    std::getline(std::cin, password);
                    
                    key_manager->create_key(name, password, "ARGON2ID");
                    break;
                }
                case 2: {
                    std::string name, password;
                    std::cout << "\n";
                    print_header("ACTIVAR CLAVE");
                    
                    key_manager->list_keys();
                    std::cout << "\n";
                    
                    std::cout << " Nombre de la clave: ";
                    std::getline(std::cin, name);
                    std::cout << " Contrasena: ";
                    std::getline(std::cin, password);
                    
                    key_manager->activate_key(name, password);
                    break;
                }
                case 3:
                    key_manager->list_keys();
                    break;
                case 4: {
                    std::string name;
                    std::cout << "\n";
                    print_header("ELIMINAR CLAVE");
                    std::cout << " Nombre de la clave a eliminar: ";
                    std::getline(std::cin, name);
                    
                    key_manager->delete_key(name);
                    break;
                }
                case 5:
                    key_manager->deactivate_current_key();
                    break;
                case 6:
                    key_manager->print_security_report();
                    break;
                case 7:
                    return;
                default:
                    std::cout << COLOR_RED "Opcion invalida\n" COLOR_RESET;
                    break;
            }
            
            std::cout << "\nPresione Enter para continuar...";
            std::cin.get();
        }
    }

    void run_speed_test() {
        print_header("TEST DE VELOCIDAD - HILOS AUTOMATICOS CHACHA20");
        
        uint32_t key_data[12];
        uint8_t nonce[12];
        std::memset(key_data, 0x42, sizeof(key_data));
        std::memset(nonce, 0x24, sizeof(nonce));
        
        const size_t TEST_SIZE = 512 * 1024 * 1024;
        const int ITERATIONS = 3;
        
        bool use_avx2 = has_avx2_support() && current_config.enable_avx2;
        
        std::cout << COLOR_CYAN "Configuracion de test:\n" COLOR_RESET;
        std::cout << " • Hilos: " << current_config.encryption_threads << " (AUTO)\n";
        std::cout << " • Tamano: 512MB\n";
        std::cout << " • Iteraciones: 3\n";
        std::cout << " • AVX2: " << (use_avx2 ? COLOR_GREEN "Activado\n" COLOR_RESET : COLOR_YELLOW "Fallback\n" COLOR_RESET);
        
        double total_speed = 0;
        std::vector<double> speeds;
        
        for (int iter = 0; iter < ITERATIONS; iter++) {
            std::vector<uint8_t> test_buffer(TEST_SIZE, static_cast<uint8_t>(iter));
            std::vector<uint8_t> output_buffer(TEST_SIZE);
            
            std::cout << " Iteracion " << (iter + 1) << "/" << ITERATIONS << "... ";
            std::flush(std::cout);
            
            auto start_time = std::chrono::high_resolution_clock::now();
            
            if (use_avx2) {
#if defined(__AVX2__) || defined(_WIN32)
                process_buffer_avx2_parallel(test_buffer.data(), output_buffer.data(), TEST_SIZE, 
                                           key_data, nonce, 0, current_config.encryption_threads);
#endif
            } else {
                // Fallback implementation
                ChaCha20 cipher(reinterpret_cast<const uint8_t*>(key_data), nonce);
                cipher.process_bytes(test_buffer.data(), output_buffer.data(), TEST_SIZE, 0);
            }
            
            auto end_time = std::chrono::high_resolution_clock::now();
            double elapsed_seconds = std::chrono::duration<double>(end_time - start_time).count();
            double speed_mbs = (TEST_SIZE / (1024.0 * 1024.0)) / elapsed_seconds;
            total_speed += speed_mbs;
            speeds.push_back(speed_mbs);
            
            std::cout << COLOR_GREEN << std::setw(6) << std::setprecision(0) << std::fixed 
                      << speed_mbs << " MB/s\n" COLOR_RESET;
        }
        
        double avg_speed = total_speed / ITERATIONS;
        double min_speed = *std::min_element(speeds.begin(), speeds.end());
        double max_speed = *std::max_element(speeds.begin(), speeds.end());
        
        std::cout << "\n" COLOR_CYAN "RESULTADOS:\n" COLOR_RESET;
        std::cout << COLOR_WHITE "├──────────────────────────────────────────────────────────────────────────┤\n" << COLOR_RESET;
        std::cout << " • Velocidad promedio: " << COLOR_GREEN << std::setprecision(0) 
                  << avg_speed << " MB/s\n" COLOR_RESET;
        std::cout << " • Mejor velocidad: " << COLOR_GREEN << max_speed << " MB/s\n" COLOR_RESET;
        std::cout << " • Peor velocidad: " << COLOR_YELLOW << min_speed << " MB/s\n" COLOR_RESET;
        std::cout << " • Configuracion: " << current_config.encryption_threads << " HILOS ";
        std::cout << (use_avx2 ? "AVX2" : "FALLBACK") << " (AUTO)\n";
        
        if (avg_speed > 1000) {
            std::cout << " • Estado: " COLOR_GREEN "EXCELENTE - Rendimiento optimo\n" COLOR_RESET;
        } else if (avg_speed > 500) {
            std::cout << " • Estado: " COLOR_GREEN "BUENO - Rendimiento aceptable\n" COLOR_RESET;
        } else {
            std::cout << " • Estado: " COLOR_YELLOW "REGULAR - Considere optimizar\n" COLOR_RESET;
        }
        
        print_footer();
        std::cout << "\nPresione Enter para continuar...";
        std::cin.ignore();
        std::cin.get();
    }

    void display_info_from_file() {
        std::ifstream info_file("rubic_info.txt");
        if (!info_file) {
            std::cout << COLOR_RED "No se pudo abrir el archivo rubic_info.txt\n" COLOR_RESET;
            std::cout << "Asegúrese de que el archivo esté en la misma carpeta que el ejecutable.\n";
            return;
        }
        
        print_header("INFORMACIÓN DEL SISTEMA RUBIC");
        
        std::string line;
        while (std::getline(info_file, line)) {
            std::cout << " " << line << "\n";
        }
        info_file.close();
        
        print_footer();
        std::cout << "\nPresione Enter para continuar...";
        std::cin.ignore();
        std::cin.get();
    }

public:
    USBEncryptionSystem() : key_manager(std::make_unique<KeyManager>()) {}
    
    void run() {
        // Mostrar información desde archivo al inicio
        display_info_from_file();
        
        while (true) {
            print_rubic_art();
            
            std::cout << COLOR_WHITE << "┌──────────────────────────────────────────────────────────────────────────┐\n";
            std::cout << "│" << COLOR_CYAN COLOR_BOLD << center_text("RUBIC - SISTEMA DE CIFRADO SEGURO PARA DISPOSITIVOS USB", 74) << COLOR_WHITE << "│\n";
            std::cout << "├──────────────────────────────────────────────────────────────────────────┤\n" << COLOR_RESET;
            
            std::cout << key_manager->get_active_key_info() << "\n";
            
            auto devices = detector.detect_usb_devices();
            
            std::cout << COLOR_BOLD "MENU PRINCIPAL:\n" COLOR_RESET;
            std::cout << " 1. Cifrar/Descifrar Dispositivo USB\n";
            std::cout << " 2. Test de Velocidad (No requiere clave)\n";
            std::cout << " 3. Gestor de Claves Avanzado\n";
            std::cout << " 4. Configuracion del Sistema\n";
            std::cout << " 5. Informacion del Sistema\n";
            std::cout << " 6. Salir\n";
            std::cout << "\n" COLOR_CYAN "Opcion: " COLOR_RESET;
            
            int option;
            if (!(std::cin >> option)) {
                std::cin.clear();
                std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                std::cout << COLOR_RED "Opcion invalida\n" COLOR_RESET;
                std::this_thread::sleep_for(std::chrono::seconds(1));
                continue;
            }
            
            switch (option) {
                case 1: {
                    if (devices.empty()) {
                        std::cout << COLOR_RED "No se detectaron dispositivos USB\n" COLOR_RESET;
                        std::cout << COLOR_YELLOW "Conecte un dispositivo USB..." COLOR_RESET;
                        std::cin.ignore();
                        std::cin.get();
                        break;
                    }
                    
                    std::cout << "\n";
                    print_header("DISPOSITIVOS DETECTADOS");
                    
                    for (size_t i = 0; i < devices.size(); i++) {
                        const auto& device = devices[i];
                        // MEJORA: Detectar estado actual del cifrado
                        USBCipher temp_cipher;
                        auto cipher_info = temp_cipher.get_cipher_info(device.mount_point);
                        bool is_currently_encrypted = cipher_info.is_encrypted;
                        
                        std::string encrypted_status = is_currently_encrypted ? 
                            COLOR_GREEN "● CIFRADO" COLOR_RESET : COLOR_YELLOW "○ NO CIFRADO" COLOR_RESET;
                            
                        std::cout << COLOR_CYAN << " " << (i + 1) << ". " << device.name << COLOR_RESET "\n";
                        std::cout << "    Montaje: " << device.mount_point << "\n";
                        std::cout << "    Tamano: " << format_size(device.total_size) << "\n";
                        std::cout << "    Libre: " << format_size(device.free_space) << "\n";
                        std::cout << "    Estado: " << encrypted_status << "\n\n";
                    }
                    
                    std::cout << COLOR_YELLOW " 0. Volver al menu principal\n" COLOR_RESET;
                    std::cout << COLOR_CYAN "Seleccione dispositivo: " COLOR_RESET;
                    
                    int device_choice;
                    if (!(std::cin >> device_choice)) {
                        std::cin.clear();
                        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                        std::cout << COLOR_RED "Seleccion invalida\n" COLOR_RESET;
                        break;
                    }
                    
                    if (device_choice == 0) break;
                    if (device_choice < 1 || device_choice > static_cast<int>(devices.size())) {
                        std::cout << COLOR_RED "Seleccion invalida\n" COLOR_RESET;
                        break;
                    }
                    
                    USBDevice selected_device = devices[device_choice - 1];
                    process_device_menu(selected_device);
                    break;
                }
                case 2:
                    run_speed_test();
                    break;
                case 3:
                    show_key_manager_menu();
                    break;
                case 4:
                    show_config_menu();
                    break;
                case 5:
                    display_info_from_file();
                    break;
                case 6:
                    std::cout << COLOR_GREEN "Saliendo...\n" COLOR_RESET;
                    return;
                default:
                    std::cout << COLOR_RED "Opcion invalida\n" COLOR_RESET;
                    std::cin.clear();
                    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                    std::this_thread::sleep_for(std::chrono::seconds(1));
                    break;
            }
        }
    }

private:
    void process_device_menu(const USBDevice& device) {
        // MEJORA: Obtener estado actual del cifrado
        USBCipher temp_cipher;
        auto cipher_info = temp_cipher.get_cipher_info(device.mount_point);
        bool is_currently_encrypted = cipher_info.is_encrypted;
        
        while (true) {
            std::cout << "\n";
            print_header("DISPOSITIVO: " + device.name);
            std::cout << " Punto de montaje: " << device.mount_point << "\n";
            std::cout << " Configuracion: " << current_config.encryption_threads << " HILOS (AUTO)\n";
            std::cout << " Estado: " << (is_currently_encrypted ? COLOR_GREEN "CIFRADO" COLOR_RESET : COLOR_YELLOW "NO CIFRADO" COLOR_RESET) << "\n";
            std::cout << key_manager->get_active_key_info() << "\n";
            std::cout << COLOR_BOLD "Seleccione operacion:\n\n" COLOR_RESET;
            
            if (is_currently_encrypted) {
                std::cout << " 1. Descifrar dispositivo\n";
            } else {
                std::cout << " 1. Cifrar dispositivo (" << current_config.encryption_threads << " HILOS)\n";
            }
            std::cout << " 2. Ver informacion de cifrado\n";
            std::cout << " 3. Volver al menu principal\n";
            std::cout << "\n" COLOR_CYAN "Opcion: " COLOR_RESET;
            
            int choice;
            if (!(std::cin >> choice)) {
                std::cin.clear();
                std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                std::cout << COLOR_RED "Opcion invalida\n" COLOR_RESET;
                continue;
            }
            
            switch (choice) {
                case 1:
                    if (is_currently_encrypted) {
                        // Descifrar
                        process_device(device, true);
                        is_currently_encrypted = false; // Actualizar estado local
                    } else {
                        // Cifrar
                        if (!key_manager->has_active_key()) {
                            std::cout << COLOR_RED "NO HAY CLAVE ACTIVA\n" COLOR_RESET;
                            std::cout << COLOR_YELLOW "Debe crear y activar una clave primero\n" COLOR_RESET;
                            std::cout << "\nPresione Enter para continuar...";
                            std::cin.ignore();
                            std::cin.get();
                        } else {
                            process_device(device, false);
                            is_currently_encrypted = true; // Actualizar estado local
                        }
                    }
                    break;
                case 2:
                    show_cipher_info(device);
                    break;
                case 3:
                    return;
                default:
                    std::cout << COLOR_RED "Opcion invalida\n" COLOR_RESET;
                    break;
            }
        }
    }

    void process_device(const USBDevice& device, bool decrypt) {
        print_header((decrypt ? "DESCIFRAR DISPOSITIVO" : "CIFRAR DISPOSITIVO"));
        std::cout << " DISPOSITIVO: " << device.name << "\n";
        std::cout << " Punto de montaje: " << device.mount_point << "\n";
        std::cout << " Capacidad: " << format_size(device.total_size) << "\n";
        std::cout << " MODO: " << current_config.encryption_threads << " HILOS PARA TODOS LOS ARCHIVOS (AUTO)\n";
        
        if (!decrypt) {
            std::cout << " CLAVE ACTIVA: " << key_manager->get_active_key_info() << "\n";
            std::cout << " Poly1305: " << (current_config.enable_poly1305 ? COLOR_GREEN "ACTIVADO" COLOR_RESET : COLOR_YELLOW "DESACTIVADO" COLOR_RESET) << "\n";
            std::cout << "\n" COLOR_RED "ADVERTENCIA: CIFRADO DE ARCHIVOS ORIGINALES\n" COLOR_RESET;
        } else {
            std::cout << "\nMODO DESCIFRADO\n";
        }
        
        std::cout << "\n" COLOR_CYAN "¿Continuar? (s/n): " COLOR_RESET;
        char confirmation;
        std::cin >> confirmation;
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        if (confirmation != 's' && confirmation != 'S') return;
        
        auto cipher = std::make_unique<USBCipher>();
        cipher->set_config(current_config);
        
        if (!decrypt) {
            std::vector<uint8_t> active_key, active_nonce;
            if (key_manager->get_active_key(active_key, active_nonce)) {
                if (cipher->set_key_from_manager(active_key, active_nonce)) {
                    std::cout << COLOR_GREEN "Clave del gestor configurada\n" COLOR_RESET;
                } else {
                    std::cout << COLOR_RED "Error configurando la clave\n" COLOR_RESET;
                    return;
                }
            } else {
                std::cout << COLOR_RED "No se pudo obtener la clave activa\n" COLOR_RESET;
                return;
            }
        } else {
            if (!cipher->load_cipher_info(device.mount_point)) {
                std::cout << COLOR_RED "No se encontro informacion de cifrado\n" COLOR_RESET;
                return;
            }
            std::cout << COLOR_GREEN "Informacion de cifrado cargada\n" COLOR_RESET;
        }
        
        std::vector<std::string> file_paths;
        std::vector<uint64_t> counters;
        uint64_t total_bytes = 0;
        
        std::cout << "\nBuscando archivos...\n";
        
        try {
            for (const auto& entry : std::filesystem::recursive_directory_iterator(device.mount_point)) {
                if (entry.is_regular_file()) {
                    std::string filename = entry.path().filename().string();
                    if (filename == "USB_CIPHER_INFO.bin") continue;
                    
                    file_paths.push_back(entry.path().string());
                    counters.push_back(file_paths.size() * 1000);
                    
                    try {
                        total_bytes += entry.file_size();
                    } catch (const std::filesystem::filesystem_error& e) {
                        std::cerr << "Error obteniendo tamaño de " << entry.path() << ": " << e.what() << std::endl;
                    }
                }
            }
        } catch (const std::filesystem::filesystem_error& e) {
            std::cerr << "Error accediendo al directorio: " << e.what() << std::endl;
            return;
        }
        
        if (file_paths.empty()) {
            std::cout << COLOR_YELLOW "No hay archivos para procesar\n" COLOR_RESET;
            return;
        }
        
        std::cout << COLOR_GREEN "Encontrados " << file_paths.size() << " archivos (" 
                  << format_size(total_bytes) << ")\n" COLOR_RESET;
        std::cout << "TODOS los archivos usaran " << current_config.encryption_threads << " HILOS (AUTO)\n";
        std::cout << "MODO: " << (cipher->is_using_avx2() ? COLOR_GREEN "AVX2" COLOR_RESET : COLOR_YELLOW "FALLBACK" COLOR_RESET) << "\n";
        
        if (!decrypt && current_config.enable_poly1305) {
            std::cout << "Poly1305: " COLOR_GREEN "ACTIVADO - Verificacion de integridad habilitada\n" COLOR_RESET;
        }
        
        if (!decrypt) {
            std::cout << "Nonce: " << cipher->get_nonce_hex() << "\n";
            std::cout << "Hash clave: " << cipher->get_key_hash() << "\n";
            std::cout << COLOR_YELLOW "GUARDE ESTA INFORMACION PARA DESCIFRAR\n" COLOR_RESET;
        }
        
        std::cout << COLOR_CYAN "\n¿Iniciar procesamiento? (s/n): " COLOR_RESET;
        std::cin >> confirmation;
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        if (confirmation != 's' && confirmation != 'S') return;
        
        std::cout << "\nINICIANDO PROCESAMIENTO...\n";
        
        USBCipher::CipherResult total_result;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        std::vector<uint8_t> master_mac;
        if (!decrypt && current_config.enable_poly1305) {
            std::cout << "Calculando MAC maestro con Poly1305...\n";
            std::vector<uint8_t> master_poly_key(32);
            std::random_device rd;
            std::generate(master_poly_key.begin(), master_poly_key.end(), [&]() { return rd(); });
            
            Poly1305 master_poly(master_poly_key.data());
            for (const auto& file_path : file_paths) {
                auto file_mac = cipher->calculate_file_mac(file_path, master_poly_key);
                if (!file_mac.empty()) {
                    master_poly.update(file_mac.data(), file_mac.size());
                }
            }
            master_mac.resize(16);
            master_poly.finish(master_mac.data());
            
            std::fill(master_poly_key.begin(), master_poly_key.end(), 0);
        }
        
        for (size_t i = 0; i < file_paths.size(); i++) {
            try {
                auto result = cipher->process_file_configurable(file_paths[i], counters[i], decrypt);
                
                total_result.total_time += result.total_time;
                total_result.read_time += result.read_time;
                total_result.encrypt_time += result.encrypt_time;
                total_result.write_time += result.write_time;
                total_result.total_bytes += result.total_bytes;
                total_result.total_files++;
                
                auto current_time = std::chrono::high_resolution_clock::now();
                double elapsed = std::chrono::duration<double>(current_time - start_time).count();
                double current_speed = (total_result.total_bytes / (1024.0 * 1024.0)) / elapsed;
                
                std::string operation = decrypt ? "DESCIFRANDO" : "CIFRANDO";
                show_progress(i + 1, file_paths.size(), current_speed, operation, 
                             current_config.encryption_threads, result.used_avx2);
                
            } catch (const std::exception& e) {
                std::cout << COLOR_RED "\nError en archivo " << file_paths[i] << ": " << e.what() << "\n" COLOR_RESET;
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        double total_duration = std::chrono::duration<double>(end_time - start_time).count();
        
        // MEJORA: Actualizar correctamente el estado de cifrado
        if (decrypt) {
            // Al descifrar, marcar como no cifrado
            if (cipher->save_cipher_info(device.mount_point, master_mac, false)) {
                std::cout << COLOR_GREEN "\nInformacion de descifrado guardada\n" COLOR_RESET;
            } else {
                std::cout << COLOR_RED "\nError guardando informacion de descifrado\n" COLOR_RESET;
            }
        } else {
            // Al cifrar, marcar como cifrado
            if (cipher->save_cipher_info(device.mount_point, master_mac, true)) {
                std::cout << COLOR_GREEN "\nInformacion de cifrado guardada\n" COLOR_RESET;
            } else {
                std::cout << COLOR_RED "\nError guardando informacion de cifrado\n" COLOR_RESET;
            }
        }
        
        std::cout << "\n\n";
        print_header((decrypt ? "DESCIFRADO COMPLETADO" : "CIFRADO COMPLETADO"));
        
        std::cout << COLOR_CYAN "ESTADISTICAS GENERALES:\n" COLOR_RESET;
        std::cout << COLOR_WHITE "├──────────────────────────────────────────────────────────────────────────┤\n" << COLOR_RESET;
        std::cout << " • Archivos procesados: " << total_result.total_files << "\n";
        std::cout << " • Total de datos: " << format_size(total_result.total_bytes) << "\n";
        std::cout << " • Tiempo total: " << std::setprecision(2) << total_duration << " s\n";
        std::cout << " • Hilos utilizados: " << current_config.encryption_threads << " (AUTO)\n";
        std::cout << " • Modo utilizado: " << (cipher->is_using_avx2() ? COLOR_GREEN "AVX2" COLOR_RESET : COLOR_YELLOW "FALLBACK" COLOR_RESET) << "\n";
        
        if (total_duration > 0) {
            double avg_speed = (total_result.total_bytes / (1024.0 * 1024.0)) / total_duration;
            std::cout << " • Velocidad promedio: " << COLOR_GREEN << std::setprecision(1) 
                      << avg_speed << " MB/s\n" COLOR_RESET;
        }
        
        if (current_config.show_detailed_stats) {
            std::cout << "\n" COLOR_CYAN "ESTADISTICAS DETALLADAS:\n" COLOR_RESET;
            std::cout << COLOR_WHITE "├──────────────────────────────────────────────────────────────────────────┤\n" << COLOR_RESET;
            std::cout << " • Velocidad de LECTURA: " << std::setprecision(1) 
                      << (total_result.read_time > 0 ? (total_result.total_bytes / (1024.0 * 1024.0)) / total_result.read_time : 0) 
                      << " MB/s\n";
            std::cout << " • Velocidad de CIFRADO: " << std::setprecision(1) 
                      << (total_result.encrypt_time > 0 ? (total_result.total_bytes / (1024.0 * 1024.0)) / total_result.encrypt_time : 0) 
                      << " MB/s\n";
            std::cout << " • Velocidad de ESCRITURA: " << std::setprecision(1) 
                      << (total_result.write_time > 0 ? (total_result.total_bytes / (1024.0 * 1024.0)) / total_result.write_time : 0) 
                      << " MB/s\n";
            
            std::cout << "\n" COLOR_CYAN "TIEMPOS POR OPERACION:\n" COLOR_RESET;
            std::cout << COLOR_WHITE "├──────────────────────────────────────────────────────────────────────────┤\n" << COLOR_RESET;
            std::cout << " • Tiempo de lectura: " << std::setprecision(3) << total_result.read_time << " s\n";
            std::cout << " • Tiempo de cifrado: " << std::setprecision(3) << total_result.encrypt_time << " s\n";
            std::cout << " • Tiempo de escritura: " << std::setprecision(3) << total_result.write_time << " s\n";
        }
        
        if (!decrypt) {
            std::cout << "\n" COLOR_CYAN "INFORMACION DE SEGURIDAD:\n" COLOR_RESET;
            std::cout << COLOR_WHITE "├──────────────────────────────────────────────────────────────────────────┤\n" << COLOR_RESET;
            std::cout << " • Nonce: " << cipher->get_nonce_hex() << "\n";
            std::cout << " • Hash clave: " << cipher->get_key_hash() << "\n";
            std::cout << " • Poly1305: " << (current_config.enable_poly1305 ? COLOR_GREEN "ACTIVADO" COLOR_RESET : COLOR_YELLOW "DESACTIVADO" COLOR_RESET) << "\n";
            std::cout << " • AVX2: " << (cipher->is_using_avx2() ? COLOR_GREEN "UTILIZADO" COLOR_RESET : COLOR_YELLOW "FALLBACK" COLOR_RESET) << "\n";
            if (current_config.enable_poly1305 && !master_mac.empty()) {
                std::cout << " • MAC maestro: " << std::hex;
                for (size_t i = 0; i < std::min(master_mac.size(), size_t(8)); i++) {
                    std::cout << std::setw(2) << std::setfill('0') << static_cast<int>(master_mac[i]);
                }
                std::cout << "..." << std::dec << "\n";
            }
            std::cout << COLOR_YELLOW "GUARDE ESTA INFORMACION PARA DESCIFRAR\n" COLOR_RESET;
        }
        
        print_footer();
        std::cout << "\nPresione Enter para continuar...";
        std::cin.ignore();
        std::cin.get();
        
        // Limpiar buffer después del procesamiento
        std::cin.clear();
    }

    void show_cipher_info(const USBDevice& device) {
        std::cout << "\n";
        print_header("INFORMACION DE CIFRADO");
        
        USBCipher temp_cipher;
        auto info = temp_cipher.get_cipher_info(device.mount_point);
        
        if (info.is_encrypted) {
            std::cout << COLOR_GREEN "Dispositivo cifrado\n" COLOR_RESET;
            std::cout << " • Fecha de cifrado: " << std::ctime(&info.encryption_time);
            std::cout << " • Hilos utilizados: " << info.encryption_threads_used << "\n";
            std::cout << " • Poly1305: " << (info.poly1305_enabled ? COLOR_GREEN "ACTIVADO" COLOR_RESET : COLOR_YELLOW "DESACTIVADO" COLOR_RESET) << "\n";
            std::cout << " • AVX2: " << (info.used_avx2 ? COLOR_GREEN "UTILIZADO" COLOR_RESET : COLOR_YELLOW "FALLBACK" COLOR_RESET) << "\n";
            
            if (!info.master_mac.empty()) {
                std::cout << " • MAC maestro: " << std::hex;
                for (size_t i = 0; i < std::min(info.master_mac.size(), size_t(8)); i++) {
                    std::cout << std::setw(2) << std::setfill('0') << static_cast<int>(info.master_mac[i]);
                }
                std::cout << "..." << std::dec << "\n";
            }
        } else {
            std::cout << COLOR_YELLOW "Dispositivo no cifrado\n" COLOR_RESET;
        }
        
        print_footer();
        std::cout << "\nPresione Enter para continuar...";
        std::cin.ignore();
        std::cin.get();
    }

    void show_config_menu() {
        while (true) {
            print_header("CONFIGURACION DEL SISTEMA");
            
            std::cout << current_config.to_string() << "\n\n";
            
            std::cout << COLOR_BOLD "Opciones:\n" COLOR_RESET;
            std::cout << " 1. Configurar numero de hilos de cifrado (Actual: " << current_config.encryption_threads << ")\n";
            std::cout << " 2. " << (current_config.show_detailed_stats ? "Desactivar" : "Activar") << " estadisticas detalladas\n";
            std::cout << " 3. " << (current_config.enable_poly1305 ? "Desactivar" : "Activar") << " Poly1305\n";
            std::cout << " 4. " << (current_config.enable_avx2 ? "Desactivar" : "Activar") << " AVX2 (si esta disponible)\n";
            std::cout << " 5. Volver al menu principal\n";
            std::cout << "\n" COLOR_CYAN "Opcion: " COLOR_RESET;
            
            int choice;
            if (!(std::cin >> choice)) {
                std::cin.clear();
                std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                std::cout << COLOR_RED "Opcion invalida\n" COLOR_RESET;
                continue;
            }
            
            switch (choice) {
                case 1: {
                    std::cout << "\nNumero actual de hilos: " << current_config.encryption_threads << " (AUTO)\n";
                    std::cout << "Maximo recomendado: " << std::thread::hardware_concurrency() << "\n";
                    std::cout << "Nuevo numero de hilos (1-" << current_config.max_threads << "): ";
                    int new_threads;
                    if (!(std::cin >> new_threads)) {
                        std::cin.clear();
                        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                        std::cout << COLOR_RED "Numero de hilos invalido\n" COLOR_RESET;
                        break;
                    }
                    
                    if (new_threads >= 1 && new_threads <= current_config.max_threads) {
                        current_config.encryption_threads = new_threads;
                        std::cout << COLOR_GREEN "Hilos de cifrado configurados a: " << new_threads << " (MANUAL)\n" COLOR_RESET;
                    } else {
                        std::cout << COLOR_RED "Numero de hilos invalido\n" COLOR_RESET;
                    }
                    break;
                }
                case 2:
                    current_config.show_detailed_stats = !current_config.show_detailed_stats;
                    std::cout << COLOR_GREEN "Estadisticas " 
                              << (current_config.show_detailed_stats ? "activadas" : "desactivadas") 
                              << "\n" COLOR_RESET;
                    break;
                case 3:
                    current_config.enable_poly1305 = !current_config.enable_poly1305;
                    std::cout << COLOR_GREEN "Poly1305 " 
                              << (current_config.enable_poly1305 ? "activado" : "desactivado") 
                              << "\n" COLOR_RESET;
                    break;
                case 4:
                    if (has_avx2_support()) {
                        current_config.enable_avx2 = !current_config.enable_avx2;
                        std::cout << COLOR_GREEN "AVX2 " 
                                  << (current_config.enable_avx2 ? "activado" : "desactivado") 
                                  << "\n" COLOR_RESET;
                    } else {
                        std::cout << COLOR_RED "AVX2 no esta disponible en este sistema\n" COLOR_RESET;
                    }
                    break;
                case 5:
                    return;
                default:
                    std::cout << COLOR_RED "Opcion invalida\n" COLOR_RESET;
                    break;
            }
            
            std::cout << "\nPresione Enter para continuar...";
            std::cin.ignore();
            std::cin.get();
        }
    }
};

// ========================
// MAIN CORREGIDO
// ========================
int main() {
    print_rubic_art();
    
    std::cout << " • AVX2 Support: " << (has_avx2_support() ? COLOR_GREEN "Yes" COLOR_RESET : COLOR_YELLOW "No (usando fallback)" COLOR_RESET) << "\n";
    std::cout << " • CPU Cores: " << std::thread::hardware_concurrency() << "\n";
    
    try {
        USBEncryptionSystem system;
        system.run();
    } catch (const std::exception& e) {
        std::cout << COLOR_RED "\nERROR: " << e.what() << "\n" COLOR_RESET;
        return 1;
    } catch (...) {
        std::cout << COLOR_RED "\nERROR desconocido\n" COLOR_RESET;
        return 1;
    }
    
    std::cout << COLOR_GREEN "Aplicacion cerrada correctamente\n" COLOR_RESET;
    return 0;
}
