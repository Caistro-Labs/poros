/* poros.sss.gather native reader: one slab chunk gathered in one call (no GIL held).
 *
 * The Python reader (gather_into) issues one os.preadv per merged read and one ctypes
 * memmove per segment: each call releases and re-takes the GIL, ~14 GIL hand-offs per
 * slab, and a training main thread that runs Python between kernel launches pays
 * for every one of them. This file does the same work in a single call made through
 * ctypes (which releases the GIL for the whole call): the reads, then the copies.
 *
 * reads: nreads x 5 int64 {file, offset, length, need, scratch_off}
 *   read file `file` (an index into fds) from `offset` into scratch + scratch_off until
 *   at least `need` bytes landed (an aligned O_DIRECT read may run past the end of the
 *   file); drop[file] != 0 then drops that range from the page cache.
 * ops: nops x 4 int64 {kind, src, dst, nbytes}
 *   kind 0: copy nbytes from scratch + src to out + dst;
 *   kind 1: zero nbytes at out + dst;
 *   kind 2: float16 at scratch + src widened to float32 at out + dst (nbytes: output);
 *   kind 3: bfloat16 likewise.
 * Returns 0, or -errno (EIO for a read that hit the end of the file early, EINVAL for an
 * unknown op).
 *
 * The widenings are exact for every non-NaN value (poros.sss.gather never emits one over
 * a NaN); tests compare them with torch over every float16 / bfloat16 bit pattern.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>

int64_t poros_sss_gather_abi(void) { return 1; }

static uint32_t f16_to_f32_bits(uint16_t h) {
    uint32_t s = (uint32_t)(h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1fu;
    uint32_t m = h & 0x3ffu;
    if (e == 0x1fu) return s | 0x7f800000u | (m << 13);
    if (e == 0) {
        if (m == 0) return s;
        e = 113;
        while (!(m & 0x400u)) {
            m <<= 1;
            e--;
        }
        return s | (e << 23) | ((m & 0x3ffu) << 13);
    }
    return s | ((e + 112u) << 23) | (m << 13);
}

int64_t poros_sss_gather(const int64_t *fds, const uint8_t *drop, const int64_t *reads,
                         int64_t nreads, const int64_t *ops, int64_t nops, uint8_t *scratch,
                         uint8_t *out) {
    for (int64_t i = 0; i < nreads; i++) {
        const int64_t *r = reads + 5 * i;
        int fd = (int)fds[r[0]];
        int64_t off = r[1], len = r[2], need = r[3], got = 0;
        uint8_t *buf = scratch + r[4];
        while (got < need) {
            ssize_t n = pread(fd, buf + got, (size_t)(len - got), (off_t)(off + got));
            if (n < 0) {
                if (errno == EINTR) continue;
                return -(int64_t)errno;
            }
            if (n == 0) return -(int64_t)EIO;
            got += n;
        }
        if (drop[r[0]]) posix_fadvise(fd, (off_t)off, (off_t)len, POSIX_FADV_DONTNEED);
    }
    for (int64_t i = 0; i < nops; i++) {
        const int64_t *o = ops + 4 * i;
        int64_t kind = o[0], src = o[1], dst = o[2], n = o[3];
        if (kind == 0) {
            memcpy(out + dst, scratch + src, (size_t)n);
        } else if (kind == 1) {
            memset(out + dst, 0, (size_t)n);
        } else if (kind == 2 || kind == 3) {
            for (int64_t k = 0; k < n / 4; k++) {
                uint16_t h;
                uint32_t w;
                memcpy(&h, scratch + src + 2 * k, 2);
                w = kind == 2 ? f16_to_f32_bits(h) : (uint32_t)h << 16;
                memcpy(out + dst + 4 * k, &w, 4);
            }
        } else {
            return -(int64_t)EINVAL;
        }
    }
    return 0;
}
