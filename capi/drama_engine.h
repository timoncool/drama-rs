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
/* Weights folder (apiplant/drama-q4 layout); NULL on error. Same as drama_load_ex with no VRAM limit. */
DramaEngine *drama_load(const char *model_dir, uint32_t max_len, uint32_t prefill_chunk);
/* vram_limit_mb: most VRAM the engine takes, 0 = what is free less 1 GB. When the whole model does not fit, the
 * routed experts (18 GB of the 21 GB of weights) stay in page-locked RAM and a VRAM cache of the ones in use takes the
 * rest of the limit: the output is the same, less VRAM only decodes slower. */
DramaEngine *drama_load_ex(const char *model_dir, uint32_t max_len, uint32_t prefill_chunk, uint32_t vram_limit_mb);
#define DRAMA_EXPERTS_VRAM 1
#define DRAMA_EXPERTS_RAM 2
/* Where the experts live (DRAMA_EXPERTS_VRAM or _RAM) and, in RAM, the cache's expert reads since loading that
 * hit and that were copied from RAM. Any output may be NULL. */
int32_t drama_experts(DramaEngine *engine, uint32_t *out_experts, uint64_t *out_hits, uint64_t *out_misses);
void drama_free(DramaEngine *engine);
/* Speaks `text` in the voice of `ref_wav` (any rate, its exact transcript in `ref_text`). The reference is
 * encoded once per (path, transcript, size, modified time) and kept. On DRAMA_OK *out_audio holds *out_len
 * samples, to be released with drama_free_audio. */
int32_t drama_speak(DramaEngine *engine, const char *text, const char *ref_wav, const char *ref_text,
                    const DramaParams *params, float **out_audio, size_t *out_len);
/* Called with each decoded piece of audio (a sentence or more) while drama_speak_stream runs, on another
 * thread; `audio` is valid only during the call. */
typedef void (*DramaChunkFn)(const float *audio, size_t len, void *user);
/* As drama_speak, and on_chunk (may be NULL) gets the audio as it is decoded; *out_audio still holds all of it. */
int32_t drama_speak_stream(DramaEngine *engine, const char *text, const char *ref_wav, const char *ref_text,
                           const DramaParams *params, DramaChunkFn on_chunk, void *user, float **out_audio,
                           size_t *out_len);
/* A scene of several speakers in one generation: every line of `text` starts with <|speaker:N|>, voice N is ref_wavs[N]
 * with its exact transcript ref_texts[N] (n_voices 1..16). The model hears each earlier line, so the scene sounds
 * like one conversation; params->max_new caps the frames of the whole scene. Output as drama_speak_stream. */
int32_t drama_speak_scene(DramaEngine *engine, const char *text, const char *const *ref_wavs, const char *const *ref_texts,
                          uint32_t n_voices, const DramaParams *params, DramaChunkFn on_chunk, void *user,
                          float **out_audio, size_t *out_len);
void drama_free_audio(float *audio, size_t len);
void drama_cancel(DramaEngine *engine);

#endif
