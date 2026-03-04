/* qwen_tts_audio.c - Stub */
#include "qwen_tts.h"
int qwen_tts_write_wav(const char *path, const float *samples, int n_samples, int sample_rate) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    int bits = 16, channels = 1;
    int data_size = n_samples * channels * (bits/8);
    int file_size = 36 + data_size;
    int byte_rate = sample_rate * channels * (bits/8);
    short block_align = channels * (bits/8);
    fwrite("RIFF", 1, 4, f);
    fwrite(&file_size, 4, 1, f);
    fwrite("WAVEfmt ", 1, 8, f);
    int fmt_size = 16; short audio_fmt = 1;
    fwrite(&fmt_size, 4, 1, f);
    fwrite(&audio_fmt, 2, 1, f);
    fwrite(&channels, 2, 1, f);
    fwrite(&sample_rate, 4, 1, f);
    fwrite(&byte_rate, 4, 1, f);
    fwrite(&block_align, 2, 1, f);
    fwrite(&bits, 2, 1, f);
    fwrite("data", 1, 4, f);
    fwrite(&data_size, 4, 1, f);
    for (int i = 0; i < n_samples; i++) {
        float s = samples[i];
        if (s < -1) s = -1; if (s > 1) s = 1;
        int16_t sample = (int16_t)(s * 32767);
        fwrite(&sample, 2, 1, f);
    }
    fclose(f);
    return 0;
}
