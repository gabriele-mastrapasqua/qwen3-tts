/*
 * main.c - Qwen3-TTS CLI
 */

#include "qwen_tts.h"
#include "qwen_tts_audio.h"
#include "qwen_tts_kernels.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>

int main(int argc, char **argv) {
    const char *model_dir = NULL;
    const char *text = NULL;
    const char *output = "output.wav";
    int speaker_id = -1;
    const char *language = NULL;
    float temperature = 0.9f;
    int top_k = 50;
    float top_p = 1.0f;
    float rep_penalty = 1.05f;
    int max_tokens = 8192;
    int silent = 0;
    int debug = 0;
    int threads = 0;  /* 0 = auto-detect */

    static struct option long_options[] = {
        {"model-dir",   required_argument, 0, 'd'},
        {"text",        required_argument, 0, 't'},
        {"output",      required_argument, 0, 'o'},
        {"speaker",     required_argument, 0, 's'},
        {"language",    required_argument, 0, 'l'},
        {"temperature", required_argument, 0, 'T'},
        {"top-k",       required_argument, 0, 'k'},
        {"top-p",       required_argument, 0, 'p'},
        {"rep-penalty", required_argument, 0, 'r'},
        {"max-tokens",  required_argument, 0, 'm'},
        {"threads",     required_argument, 0, 'j'},
        {"silent",      no_argument,       0, 'S'},
        {"debug",       no_argument,       0, 'D'},
        {"help",        no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "d:t:o:s:l:T:k:p:r:m:j:SDh", long_options, NULL)) != -1) {
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
                fprintf(stderr, "  -S, --silent               Silent mode\n");
                fprintf(stderr, "  -D, --debug                Debug mode\n");
                return opt == 'h' ? 0 : 1;
        }
    }

    if (!model_dir || !text) {
        fprintf(stderr, "Error: --model-dir and --text are required\n");
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

    /* Generate */
    float *audio = NULL;
    int n_samples = 0;

    fprintf(stderr, "[MAIN] Starting generation...\n");
    if (qwen_tts_generate(ctx, text, &audio, &n_samples) != 0) {
        fprintf(stderr, "[MAIN] Generation failed\n");
        qwen_tts_unload(ctx);
        return 1;
    }
    fprintf(stderr, "[MAIN] Generation done, n_samples=%d, audio=%p\n", n_samples, (void *)audio);

    /* Write WAV */
    if (audio && n_samples > 0) {
        fprintf(stderr, "[MAIN] Writing WAV (%d samples)...\n", n_samples);
        if (qwen_tts_write_wav(output, audio, n_samples, QWEN_TTS_SAMPLE_RATE) == 0) {
            if (!silent)
                fprintf(stderr, "Wrote %s (%d samples, %.2fs)\n", output, n_samples, (float)n_samples / QWEN_TTS_SAMPLE_RATE);
        } else {
            fprintf(stderr, "[MAIN] Failed to write WAV\n");
        }
        free(audio);
    } else {
        fprintf(stderr, "[MAIN] No audio to write (audio=%p, n_samples=%d)\n", (void *)audio, n_samples);
    }

    qwen_tts_unload(ctx);
    return 0;
}
