#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "qwen_tts.h"

extern int qwen_speech_decoder_decode(qwen_tts_ctx_t *ctx, const int *codes, int n_frames,
                                      float **audio_out, int *n_samples);

#define QDT_NCB 16

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <codes.txt> [model_dir] [out.wav]\n", argv[0]);
        fprintf(stderr, "  codes.txt = QWEN_DUMP_CODES output (16 ints per line, one frame per line)\n");
        return 1;
    }

    const char *codes_file = argv[1];
    const char *model_dir = (argc > 2) ? argv[2] : "qwen3-tts-0.6b";
    const char *output = (argc > 3) ? argv[3] : "decoder_test.wav";

    FILE *f = fopen(codes_file, "r");
    if (!f) {
        fprintf(stderr, "Cannot open %s\n", codes_file);
        return 1;
    }
    int cap = 256, n_frames = 0;
    int *codes = malloc((size_t)cap * QDT_NCB * sizeof(int));
    if (!codes) { fprintf(stderr, "OOM\n"); fclose(f); return 1; }
    for (;;) {
        if (n_frames >= cap) {
            cap *= 2;
            int *t = realloc(codes, (size_t)cap * QDT_NCB * sizeof(int));
            if (!t) { fprintf(stderr, "OOM\n"); free(codes); fclose(f); return 1; }
            codes = t;
        }
        int *row = codes + (size_t)n_frames * QDT_NCB, got = 0;
        while (got < QDT_NCB && fscanf(f, "%d", &row[got]) == 1) got++;
        if (got == 0) break;
        if (got < QDT_NCB) {
            fprintf(stderr, "Truncated frame %d in %s (%d/%d codes)\n",
                    n_frames, codes_file, got, QDT_NCB);
            free(codes); fclose(f);
            return 1;
        }
        n_frames++;
    }
    fclose(f);
    if (n_frames == 0) {
        fprintf(stderr, "No frames in %s\n", codes_file);
        free(codes);
        return 1;
    }
    printf("Loaded %d frames x %d codebooks from %s\n", n_frames, QDT_NCB, codes_file);

    printf("Loading model from %s...\n", model_dir);
    qwen_tts_ctx_t *ctx = qwen_tts_load(model_dir);
    if (!ctx) {
        fprintf(stderr, "Model load failed (%s)\n", model_dir);
        free(codes);
        return 1;
    }

    float *audio = NULL;
    int n_samples = 0;
    if (qwen_speech_decoder_decode(ctx, codes, n_frames, &audio, &n_samples) != 0) {
        fprintf(stderr, "Decoding failed!\n");
        qwen_tts_unload(ctx);
        free(codes);
        return 1;
    }

    if (qwen_tts_write_wav(output, audio, n_samples, 24000) != 0) {
        fprintf(stderr, "Cannot write %s\n", output);
        free(audio); qwen_tts_unload(ctx); free(codes);
        return 1;
    }
    printf("Wrote %s (%d samples, %.2fs)\n", output, n_samples, (float)n_samples / 24000.0f);

    free(audio);
    qwen_tts_unload(ctx);
    free(codes);
    return 0;
}
