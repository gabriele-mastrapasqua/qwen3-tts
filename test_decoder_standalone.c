#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* Simple program to decode test codes using the C speech decoder */

extern int qwen_decode_codes(const char *model_dir, const int *codes, int n_frames,
                              int n_codebooks, float **audio_out, int *n_samples);

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <codes.bin> [model_dir]\n", argv[0]);
        return 1;
    }
    
    const char *codes_file = argv[1];
    const char *model_dir = (argc > 2) ? argv[2] : "qwen3-tts-0.6b";
    
    /* Read codes */
    FILE *f = fopen(codes_file, "rb");
    if (!f) {
        fprintf(stderr, "Cannot open %s\n", codes_file);
        return 1;
    }
    
    int n_frames, n_codebooks;
    fread(&n_frames, sizeof(int), 1, f);
    fread(&n_codebooks, sizeof(int), 1, f);
    
    int *codes = malloc(n_frames * n_codebooks * sizeof(int));
    fread(codes, sizeof(int), n_frames * n_codebooks, f);
    fclose(f);
    
    printf("Loaded %d frames x %d codebooks from %s\n", n_frames, n_codebooks, codes_file);
    
    /* Decode */
    float *audio;
    int n_samples;
    
    printf("Decoding with model from %s...\n", model_dir);
    int ret = qwen_decode_codes(model_dir, codes, n_frames, n_codebooks, &audio, &n_samples);
    
    if (ret != 0) {
        fprintf(stderr, "Decoding failed!\n");
        free(codes);
        return 1;
    }
    
    /* Write WAV */
    const char *output = "/tmp/c_decoder_test.wav";
    FILE *wav = fopen(output, "wb");
    if (!wav) {
        fprintf(stderr, "Cannot write %s\n", output);
        free(audio);
        free(codes);
        return 1;
    }
    
    /* Simple WAV header for 24kHz, 16-bit mono */
    int sr = 24000;
    int bits = 16;
    int channels = 1;
    int data_size = n_samples * channels * (bits/8);
    int file_size = 36 + data_size;
    
    fwrite("RIFF", 1, 4, wav);
    fwrite(&file_size, 4, 1, wav);
    fwrite("WAVE", 1, 4, wav);
    fwrite("fmt ", 1, 4, wav);
    int fmt_size = 16;
    fwrite(&fmt_size, 4, 1, wav);
    short audio_fmt = 1;
    fwrite(&audio_fmt, 2, 1, wav);
    fwrite(&channels, 2, 1, wav);
    fwrite(&sr, 4, 1, wav);
    int byte_rate = sr * channels * (bits/8);
    fwrite(&byte_rate, 4, 1, wav);
    short block_align = channels * (bits/8);
    fwrite(&block_align, 2, 1, wav);
    fwrite(&bits, 2, 1, wav);
    fwrite("data", 1, 4, wav);
    fwrite(&data_size, 4, 1, wav);
    
    /* Write samples (clamp and convert to 16-bit) */
    for (int i = 0; i < n_samples; i++) {
        float s = audio[i];
        if (s < -1.0f) s = -1.0f;
        if (s > 1.0f) s = 1.0f;
        int16_t sample = (int16_t)(s * 32767.0f);
        fwrite(&sample, 2, 1, wav);
    }
    
    fclose(wav);
    printf("Wrote %s (%d samples, %.2fs)\n", output, n_samples, (float)n_samples/sr);
    
    free(audio);
    free(codes);
    return 0;
}
