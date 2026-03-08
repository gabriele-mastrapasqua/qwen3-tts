/*
 * main.c - Qwen3-TTS CLI
 */

#include "qwen_tts.h"
#include "qwen_tts_audio.h"
#include "qwen_tts_kernels.h"
#include "qwen_tts_server.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>

/* Streaming callback state */
typedef struct {
    FILE *file;            /* WAV file or stdout */
    int is_stdout;         /* 1 = raw PCM to stdout, 0 = WAV file */
    int total_samples;     /* running count of samples written */
} stream_state_t;

static int stream_audio_callback(const float *samples, int n_samples, void *userdata) {
    stream_state_t *st = (stream_state_t *)userdata;
    if (!st->file) return -1;
    for (int i = 0; i < n_samples; i++) {
        float s = samples[i];
        if (s < -1.0f) s = -1.0f;
        if (s > 1.0f) s = 1.0f;
        int16_t sample = (int16_t)(s * 32767);
        fwrite(&sample, 2, 1, st->file);
    }
    fflush(st->file);
    st->total_samples += n_samples;
    return 0;
}

/* Write a WAV header with placeholder data size (will be updated at end) */
static void write_wav_header(FILE *f, int sample_rate) {
    int bits = 16, channels = 1;
    int data_size = 0x7FFFFFFF;  /* placeholder for unknown length */
    int file_size = 36 + data_size;
    int byte_rate = sample_rate * channels * (bits/8);
    short block_align = channels * (bits/8);
    int fmt_size = 16; short audio_fmt = 1;
    fwrite("RIFF", 1, 4, f);
    fwrite(&file_size, 4, 1, f);
    fwrite("WAVEfmt ", 1, 8, f);
    fwrite(&fmt_size, 4, 1, f);
    fwrite(&audio_fmt, 2, 1, f);
    fwrite(&channels, 2, 1, f);
    fwrite(&sample_rate, 4, 1, f);
    fwrite(&byte_rate, 4, 1, f);
    fwrite(&block_align, 2, 1, f);
    fwrite(&bits, 2, 1, f);
    fwrite("data", 1, 4, f);
    fwrite(&data_size, 4, 1, f);
}

/* Update WAV header with actual data size */
static void finalize_wav_header(FILE *f, int total_samples) {
    int data_size = total_samples * 2;  /* 16-bit mono */
    int file_size = 36 + data_size;
    fseek(f, 4, SEEK_SET);
    fwrite(&file_size, 4, 1, f);
    fseek(f, 40, SEEK_SET);
    fwrite(&data_size, 4, 1, f);
}

int main(int argc, char **argv) {
    const char *model_dir = NULL;
    const char *text = NULL;
    const char *output = "output.wav";
    int speaker_id = -1;
    const char *language = NULL;
    const char *instruct = NULL;
    float temperature = 0.9f;
    int top_k = 50;
    float top_p = 1.0f;
    float rep_penalty = 1.05f;
    int max_tokens = 8192;
    int silent = 0;
    int debug = 0;
    int threads = 0;  /* 0 = auto-detect */
    int do_stream = 0;
    int do_stdout = 0;
    int stream_chunk = 10;
    int serve_port = 0;  /* 0 = not serving */
    int seed = -1;       /* -1 = use time-based seed */
    float max_duration = 0;  /* 0 = no limit */
    int voice_design = 0;

    static struct option long_options[] = {
        {"model-dir",     required_argument, 0, 'd'},
        {"text",          required_argument, 0, 't'},
        {"output",        required_argument, 0, 'o'},
        {"speaker",       required_argument, 0, 's'},
        {"language",      required_argument, 0, 'l'},
        {"temperature",   required_argument, 0, 'T'},
        {"top-k",         required_argument, 0, 'k'},
        {"top-p",         required_argument, 0, 'p'},
        {"rep-penalty",   required_argument, 0, 'r'},
        {"max-tokens",    required_argument, 0, 'm'},
        {"threads",       required_argument, 0, 'j'},
        {"instruct",      required_argument, 0, 'I'},
        {"stream",        no_argument,       0, 1001},
        {"stdout",        no_argument,       0, 1002},
        {"stream-chunk",  required_argument, 0, 1003},
        {"serve",         required_argument, 0, 1004},
        {"seed",          required_argument, 0, 1005},
        {"max-duration",  required_argument, 0, 1006},
        {"voice-design",  no_argument,       0, 1007},
        {"silent",        no_argument,       0, 'S'},
        {"debug",         no_argument,       0, 'D'},
        {"help",          no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "d:t:o:s:l:T:k:p:r:m:j:I:SDh", long_options, NULL)) != -1) {
        switch (opt) {
            case 'd': model_dir = optarg; break;
            case 't': text = optarg; break;
            case 'o': output = optarg; break;
            case 's': speaker_id = qwen_tts_speaker_id(optarg); break;
            case 'l': language = optarg; break;
            case 'T': temperature = (float)atof(optarg); break;
            case 'k': top_k = atoi(optarg); break;
            case 'p': top_p = (float)atof(optarg); break;
            case 'r': rep_penalty = (float)atof(optarg); break;
            case 'm': max_tokens = atoi(optarg); break;
            case 'j': threads = atoi(optarg); break;
            case 'I': instruct = optarg; break;
            case 1001: do_stream = 1; break;
            case 1002: do_stdout = 1; do_stream = 1; break;  /* --stdout implies --stream */
            case 1003: stream_chunk = atoi(optarg); break;
            case 1004: serve_port = atoi(optarg); break;
            case 1005: seed = atoi(optarg); break;
            case 1006: max_duration = (float)atof(optarg); break;
            case 1007: voice_design = 1; break;
            case 'S': silent = 1; break;
            case 'D': debug = 1; break;
            case 'h':
            default:
                fprintf(stderr, "Usage: %s -d <model_dir> -t <text> [options]\n", argv[0]);
                fprintf(stderr, "Options:\n");
                fprintf(stderr, "  -d, --model-dir <path>     Model directory\n");
                fprintf(stderr, "  -t, --text <string>        Text to synthesize\n");
                fprintf(stderr, "  -o, --output <path>        Output WAV file\n");
                fprintf(stderr, "  -s, --speaker <name>       Speaker name\n");
                fprintf(stderr, "  -l, --language <name>      Language\n");
                fprintf(stderr, "  -T, --temperature <float>  Sampling temperature\n");
                fprintf(stderr, "  -k, --top-k <int>          Top-k sampling\n");
                fprintf(stderr, "  -p, --top-p <float>        Top-p sampling\n");
                fprintf(stderr, "  -r, --rep-penalty <float>  Repetition penalty\n");
                fprintf(stderr, "  -m, --max-tokens <int>     Max tokens\n");
                fprintf(stderr, "  -j, --threads <int>        Number of threads (0=auto)\n");
                fprintf(stderr, "  -I, --instruct <text>      Style instruction (1.7B only)\n");
                fprintf(stderr, "                             e.g. \"Speak in an angry tone\"\n");
                fprintf(stderr, "  --stream                   Stream audio (decode during generation)\n");
                fprintf(stderr, "  --stdout                   Output raw s16le PCM to stdout (implies --stream)\n");
                fprintf(stderr, "  --stream-chunk <n>         Frames per stream chunk (default: 10)\n");
                fprintf(stderr, "  --serve <port>             Start HTTP server on port\n");
                fprintf(stderr, "  --seed <n>                 Random seed (default: time-based)\n");
                fprintf(stderr, "  --max-duration <secs>      Max audio duration in seconds\n");
                fprintf(stderr, "  --voice-design             VoiceDesign mode (create voice from --instruct)\n");
                fprintf(stderr, "  -S, --silent               Silent mode\n");
                fprintf(stderr, "  -D, --debug                Debug mode\n");
                return opt == 'h' ? 0 : 1;
        }
    }

    if (!model_dir) {
        fprintf(stderr, "Error: --model-dir is required\n");
        return 1;
    }
    if (!text && serve_port <= 0) {
        fprintf(stderr, "Error: --text or --serve is required\n");
        return 1;
    }

    if (!silent) {
        fprintf(stderr, "Model dir: %s\n", model_dir);
        fprintf(stderr, "Text: \"%s\"\n", text);
        fprintf(stderr, "Output: %s\n", output);
    }

    /* Initialize threading: auto-detect or user override */
    if (threads > 0) qwen_set_threads(threads);
    else qwen_init_threads();

    /* Load model */
    qwen_tts_ctx_t *ctx = qwen_tts_load(model_dir);
    if (!ctx) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    /* Set parameters */
    ctx->temperature = temperature;
    ctx->top_k = top_k;
    ctx->top_p = top_p;
    ctx->rep_penalty = rep_penalty;
    ctx->max_tokens = max_tokens;
    ctx->silent = silent;
    ctx->debug = debug;

    if (speaker_id >= 0) ctx->speaker_id = speaker_id;
    if (language) ctx->language_id = qwen_tts_language_id(language);
    if (seed >= 0) ctx->seed = (uint32_t)seed;
    if (max_duration > 0) ctx->max_tokens = (int)(max_duration * 12.5f);
    if (voice_design) ctx->voice_design = 1;
    if (instruct) {
        if (ctx->config.hidden_size < 2048) {
            fprintf(stderr, "Warning: --instruct is only supported on 1.7B model (ignored)\n");
        } else {
            ctx->instruct = strdup(instruct);
        }
    }

    /* Server mode: start HTTP server and block */
    if (serve_port > 0) {
        int ret = qwen_tts_serve(ctx, serve_port);
        qwen_tts_unload(ctx);
        return ret;
    }

    /* Streaming setup */
    stream_state_t stream_state = {0};
    ctx->stream = do_stream;
    ctx->stream_chunk_frames = stream_chunk;

    if (do_stream) {
        if (do_stdout) {
            /* Raw s16le 24kHz mono PCM to stdout */
            stream_state.file = stdout;
            stream_state.is_stdout = 1;
            /* Force silent mode — all status goes to stderr, audio to stdout */
            silent = 1;
            ctx->silent = 1;
        } else {
            /* Streaming WAV: write header now, update at end */
            stream_state.file = fopen(output, "wb");
            if (!stream_state.file) {
                fprintf(stderr, "Error: cannot open %s for writing\n", output);
                qwen_tts_unload(ctx);
                return 1;
            }
            write_wav_header(stream_state.file, QWEN_TTS_SAMPLE_RATE);
        }
        qwen_tts_set_audio_callback(ctx, stream_audio_callback, &stream_state);
        if (!silent)
            fprintf(stderr, "Streaming: chunk=%d frames (%.1fs), %s\n",
                    stream_chunk, stream_chunk / 12.5f,
                    do_stdout ? "raw PCM to stdout" : output);
    }

    /* Generate */
    float *audio = NULL;
    int n_samples = 0;

    if (!silent) fprintf(stderr, "Starting generation...\n");
    if (qwen_tts_generate(ctx, text, &audio, &n_samples) != 0) {
        fprintf(stderr, "Generation failed\n");
        if (do_stream && !do_stdout && stream_state.file) fclose(stream_state.file);
        qwen_tts_unload(ctx);
        return 1;
    }

    if (do_stream) {
        /* Finalize streaming output */
        if (!do_stdout && stream_state.file) {
            finalize_wav_header(stream_state.file, stream_state.total_samples);
            fclose(stream_state.file);
            if (!silent)
                fprintf(stderr, "Wrote %s (%d samples, %.2fs) [streamed]\n",
                        output, stream_state.total_samples,
                        (float)stream_state.total_samples / QWEN_TTS_SAMPLE_RATE);
        }
        /* Free the full decode output (streaming already wrote everything) */
        free(audio);
    } else {
        /* Non-streaming: write WAV from full decode */
        if (audio && n_samples > 0) {
            if (qwen_tts_write_wav(output, audio, n_samples, QWEN_TTS_SAMPLE_RATE) == 0) {
                if (!silent)
                    fprintf(stderr, "Wrote %s (%d samples, %.2fs)\n", output, n_samples,
                            (float)n_samples / QWEN_TTS_SAMPLE_RATE);
            } else {
                fprintf(stderr, "Failed to write WAV\n");
            }
            free(audio);
        }
    }

    qwen_tts_unload(ctx);
    return 0;
}
