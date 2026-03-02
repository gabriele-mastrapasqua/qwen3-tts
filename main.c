#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <time.h>

#include "qwen_tts.h"
#include "qwen_tts_kernels.h"
#include "qwen_tts_audio.h"

static void usage(const char *prog) {
    fprintf(stderr,
        "qwen_tts -- Qwen3-TTS Pure C Inference\n"
        "\n"
        "Usage: %s [options]\n"
        "\n"
        "Required:\n"
        "  -d, --model-dir <path>     Model directory\n"
        "  --text <string>            Text to synthesize\n"
        "\n"
        "Output:\n"
        "  -o, --output <path>        Output WAV file (default: output.wav)\n"
        "  --stdout                   Write raw s16le 24kHz mono to stdout\n"
        "\n"
        "Voice control:\n"
        "  --speaker <name|id>        Speaker name or codec ID (default: serena)\n"
        "                             Names: serena, vivian, uncle_fu, ryan, aiden,\n"
        "                                    ono_anna, sohee, eric, dylan\n"
        "  --language <lang>          Target language (default: auto)\n"
        "  --ref-audio <path>         Reference audio for voice cloning (Base model)\n"
        "  --ref-text <string>        Reference audio transcript (Base model)\n"
        "  --voice-desc <string>      Voice description (VoiceDesign model)\n"
        "\n"
        "Sampling:\n"
        "  --temperature <f>          Sampling temperature (default: 0.9)\n"
        "  --top-k <n>                Top-k sampling (default: 50)\n"
        "  --top-p <f>                Top-p nucleus sampling (default: 1.0)\n"
        "  --rep-penalty <f>          Repetition penalty (default: 1.05)\n"
        "  --max-tokens <n>           Max audio tokens (default: 8192)\n"
        "  --seed <n>                 Random seed (default: time-based)\n"
        "\n"
        "Misc:\n"
        "  --silent                   Suppress status on stderr\n"
        "  --debug                    Verbose diagnostics\n"
        "  -h, --help                 Show this help\n"
        "\n"
        "Examples:\n"
        "  %s -d qwen3-tts-0.6b --text \"Hello world\" -o hello.wav\n"
        "  %s -d qwen3-tts-0.6b --text \"Ciao!\" --speaker vivian --language Italian\n"
        , prog, prog, prog);
}

int main(int argc, char **argv) {
    /* Defaults */
    const char *model_dir = NULL;
    const char *text = NULL;
    const char *output_path = "output.wav";
    const char *speaker_name = "serena";
    const char *language = NULL;
    const char *ref_audio = NULL;
    const char *ref_text = NULL;
    const char *voice_desc = NULL;
    const char *decode_codes_path = NULL;
    float temperature = 0.9f;
    int top_k = 50;
    float top_p = 1.0f;
    float rep_penalty = 1.05f;
    float cp_temperature = -1.0f;  /* -1 = use default (0.9) */
    int cp_top_k = -1;             /* -1 = use default (50) */
    int max_tokens = 8192;
    uint64_t seed = 0;
    int silent = 0;
    int debug = 0;
    int use_stdout = 0;

    static struct option long_opts[] = {
        {"model-dir",    required_argument, 0, 'd'},
        {"text",         required_argument, 0, 't'},
        {"output",       required_argument, 0, 'o'},
        {"stdout",       no_argument,       0, 'S'},
        {"speaker",      required_argument, 0, 's'},
        {"language",     required_argument, 0, 'l'},
        {"ref-audio",    required_argument, 0, 'R'},
        {"ref-text",     required_argument, 0, 'T'},
        {"voice-desc",   required_argument, 0, 'V'},
        {"temperature",  required_argument, 0, 1001},
        {"top-k",        required_argument, 0, 1002},
        {"top-p",        required_argument, 0, 1003},
        {"rep-penalty",  required_argument, 0, 1004},
        {"max-tokens",   required_argument, 0, 1005},
        {"seed",         required_argument, 0, 1006},
        {"cp-temperature", required_argument, 0, 1007},
        {"cp-top-k",    required_argument, 0, 1008},
        {"silent",       no_argument,       0, 1010},
        {"debug",        no_argument,       0, 1011},
        {"decode-codes", required_argument, 0, 1014},
        {"bf16-round",   no_argument,       0, 1012},
        {"no-bf16-round", no_argument,      0, 1013},
        {"help",         no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "d:t:o:s:l:h", long_opts, NULL)) != -1) {
        switch (opt) {
        case 'd': model_dir = optarg; break;
        case 't': text = optarg; break;
        case 'o': output_path = optarg; break;
        case 'S': use_stdout = 1; break;
        case 's': speaker_name = optarg; break;
        case 'l': language = optarg; break;
        case 'R': ref_audio = optarg; break;
        case 'T': ref_text = optarg; break;
        case 'V': voice_desc = optarg; break;
        case 1001: temperature = (float)atof(optarg); break;
        case 1002: top_k = atoi(optarg); break;
        case 1003: top_p = (float)atof(optarg); break;
        case 1004: rep_penalty = (float)atof(optarg); break;
        case 1005: max_tokens = atoi(optarg); break;
        case 1006: seed = (uint64_t)atoll(optarg); break;
        case 1007: cp_temperature = (float)atof(optarg); break;
        case 1008: cp_top_k = atoi(optarg); break;
        case 1010: silent = 1; break;
        case 1011: debug = 1; break;
        case 1012: qwen_bf16_precision = 1; break;
        case 1013: qwen_bf16_precision = 0; break;
        case 1014: decode_codes_path = optarg; break;
        case 'h': usage(argv[0]); return 0;
        default: usage(argv[0]); return 1;
        }
    }

    if (!model_dir) {
        fprintf(stderr, "Error: --model-dir is required\n\n");
        usage(argv[0]);
        return 1;
    }
    if (!text && !decode_codes_path) {
        fprintf(stderr, "Error: --text or --decode-codes is required\n\n");
        usage(argv[0]);
        return 1;
    }

    /* Default seed from time */
    if (seed == 0) {
        seed = (uint64_t)time(NULL);
    }

    /* Resolve speaker name to codec ID */
    int speaker_id = qwen_tts_speaker_id(speaker_name);
    if (speaker_id < 0) {
        /* Try parsing as numeric ID */
        speaker_id = atoi(speaker_name);
        if (speaker_id == 0 && strcmp(speaker_name, "0") != 0) {
            fprintf(stderr, "Error: unknown speaker '%s'\n", speaker_name);
            return 1;
        }
    }

    if (!silent) {
        fprintf(stderr, "Model dir: %s\n", model_dir);
        fprintf(stderr, "Text: \"%s\"\n", text);
        fprintf(stderr, "Speaker: %s (id=%d), Language: %s\n",
                speaker_name, speaker_id, language ? language : "auto");
        fprintf(stderr, "Sampling: temp=%.2f top_k=%d top_p=%.2f rep_pen=%.2f\n",
                temperature, top_k, top_p, rep_penalty);
        fprintf(stderr, "Seed: %llu\n", (unsigned long long)seed);
    }

    /* Suppress unused warnings for now */
    (void)ref_audio;
    (void)ref_text;
    (void)voice_desc;

    /* Initialize RNG */
    qwen_tts_sampling_seed(seed);

    /* Load model */
    qwen_tts_ctx_t *ctx = qwen_tts_load(model_dir);
    if (!ctx) {
        fprintf(stderr, "Failed to load model from %s\n", model_dir);
        return 1;
    }

    /* Configure */
    qwen_tts_set_speaker(ctx, speaker_id);
    if (language) qwen_tts_set_language(ctx, language);
    ctx->temperature = temperature;
    ctx->top_k = top_k;
    ctx->top_p = top_p;
    ctx->rep_penalty = rep_penalty;
    ctx->max_tokens = max_tokens;
    if (cp_temperature >= 0.0f) ctx->cp_temperature = cp_temperature;
    if (cp_top_k >= 0) ctx->cp_top_k = cp_top_k;
    ctx->debug = debug;
    ctx->silent = silent;

    /* Generate or decode from codes file */
    float *audio = NULL;
    int n_samples = 0;
    int ret;

    if (decode_codes_path) {
        /* Decode-only mode: read codes from text file and run speech decoder */
        extern int qwen_speech_dec_forward(qwen_tts_ctx_t *ctx, const int *codes,
                                           int n_frames, float **audio_out, int *n_samples_out);
        FILE *cf = fopen(decode_codes_path, "r");
        if (!cf) { fprintf(stderr, "Error: cannot open %s\n", decode_codes_path); qwen_tts_free(ctx); return 1; }
        int n_fr, n_cb;
        if (fscanf(cf, "%d %d", &n_fr, &n_cb) != 2) { fprintf(stderr, "Error: bad codes file header\n"); fclose(cf); qwen_tts_free(ctx); return 1; }
        fprintf(stderr, "Decoding %d frames x %d codebooks from %s\n", n_fr, n_cb, decode_codes_path);
        /* Read frame-major [n_fr][n_cb], store frame-major [n_fr][16] */
        int *codes = (int *)calloc((int64_t)n_fr * QWEN_TTS_NUM_CODEBOOKS, sizeof(int));
        for (int f = 0; f < n_fr; f++) {
            for (int cb = 0; cb < n_cb && cb < QWEN_TTS_NUM_CODEBOOKS; cb++) {
                int v; if (fscanf(cf, "%d", &v) == 1) codes[f * QWEN_TTS_NUM_CODEBOOKS + cb] = v;
            }
        }
        fclose(cf);
        ret = qwen_speech_dec_forward(ctx, codes, n_fr, &audio, &n_samples);
        free(codes);
        if (ret != 0) { fprintf(stderr, "Speech decoder failed\n"); qwen_tts_free(ctx); return 1; }
        fprintf(stderr, "Decoded %d samples (%.1fs)\n", n_samples, (double)n_samples / QWEN_TTS_SAMPLE_RATE);
    } else {
        ret = qwen_tts_generate(ctx, text, &audio, &n_samples);
        if (ret != 0) {
            fprintf(stderr, "Generation failed\n");
            qwen_tts_free(ctx);
            return 1;
        }
    }

    /* Output */
    if (use_stdout) {
        int16_t *buf = (int16_t *)malloc(n_samples * sizeof(int16_t));
        if (buf) {
            qwen_tts_float_to_s16(audio, buf, n_samples);
            fwrite(buf, sizeof(int16_t), n_samples, stdout);
            free(buf);
        }
    } else {
        qwen_tts_write_wav(output_path, audio, n_samples, QWEN_TTS_SAMPLE_RATE);
        if (!silent) fprintf(stderr, "Wrote %s (%d samples, %.1fs)\n",
                             output_path, n_samples, (double)n_samples / QWEN_TTS_SAMPLE_RATE);
    }

    free(audio);
    qwen_tts_free(ctx);
    return 0;
}
