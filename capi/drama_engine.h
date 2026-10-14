/* C ABI of drama_engine (drama-rs). Strings are UTF-8, NUL-terminated. Audio is mono f32 at
 * drama_sample_rate() Hz. One DramaEngine serves one caller at a time; drama_cancel may be called from
 * any thread while drama_speak runs. */
#ifndef DRAMA_ENGINE_H
#define DRAMA_ENGINE_H
#include <stddef.h>
#include <stdint.h>

#define DRAMA_OK 0
#define DRAMA_ERROR 1
#define DRAMA_CANCELLED 2

typedef struct DramaEngine DramaEngine;

typedef struct DramaParams {
    float temperature; /* 1.0 */
    int32_t top_k;     /* 30 */
    float top_p;       /* 0.9 */
    uint64_t seed;
    uint32_t max_new;  /* most frames (21.5 per second of speech) of each sentence */
} DramaParams;

const char *drama_version(void);
uint32_t drama_sample_rate(void);
/* The last error of this thread, valid until its next call into the engine. */
const char *drama_last_error(void);
/* Weights folder (apiplant/drama-q4 layout); NULL on error. */
DramaEngine *drama_load(const char *model_dir, uint32_t max_len, uint32_t prefill_chunk);
void drama_free(DramaEngine *engine);
/* Speaks `text` in the voice of `ref_wav` (any rate, its exact transcript in `ref_text`). The reference is
 * encoded once per (path, transcript, size, modified time) and kept. On DRAMA_OK *out_audio holds *out_len
 * samples, to be released with drama_free_audio. */
int32_t drama_speak(DramaEngine *engine, const char *text, const char *ref_wav, const char *ref_text,
                    const DramaParams *params, float **out_audio, size_t *out_len);
void drama_free_audio(float *audio, size_t len);
void drama_cancel(DramaEngine *engine);

#endif
