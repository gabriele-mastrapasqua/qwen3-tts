/*
 * main.c - Qwen3-TTS CLI
 */

#include "qwen_tts.h"
#include "qwen_tts_audio.h"
#include "qwen_tts_kernels.h"
#include "qwen_tts_server.h"

#include <stdio.h>
#include <lz4.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <math.h>
#include <dirent.h>
#include <sys/stat.h>

/* Print info about a .qvoice file. Returns 0 on success. */
static int print_qvoice_info(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;

    char magic[4];
    uint32_t version;
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "QVCE", 4) != 0) {
        fclose(f); return -1;
    }
    if (fread(&version, sizeof(uint32_t), 1, f) != 1) {
        fclose(f); return -1;
    }

    /* Read/skip speaker embedding (v2+ has enc_dim header, v1 assumes 1024) */
    uint32_t file_enc_dim = 1024;
    if (version >= 2) {
        if (fread(&file_enc_dim, sizeof(uint32_t), 1, f) != 1) { fclose(f); return -1; }
    }
    fseek(f, file_enc_dim * sizeof(float), SEEK_CUR);

    /* Read ref_text */
    uint32_t ref_text_len = 0;
    if (fread(&ref_text_len, sizeof(uint32_t), 1, f) != 1) { fclose(f); return -1; }
    char ref_text[256] = {0};
    if (ref_text_len > 0) {
        int read_len = ref_text_len < 255 ? (int)ref_text_len : 255;
        if (fread(ref_text, 1, read_len, f) != (size_t)read_len) { fclose(f); return -1; }
        ref_text[read_len] = '\0';
        if (ref_text_len > 255) fseek(f, ref_text_len - 255, SEEK_CUR);
    }

    /* Read n_ref_frames */
    uint32_t n_ref_frames = 0;
    if (fread(&n_ref_frames, sizeof(uint32_t), 1, f) != 1) { fclose(f); return -1; }
    /* Skip ref_codes data */
    if (n_ref_frames > 0)
        fseek(f, (long)n_ref_frames * 16 * sizeof(int), SEEK_CUR);

    /* v3 metadata */
    char meta_lang_name[16] = {0};
    char meta_voice_name[64] = {0};
    float meta_ref_dur = 0;
    uint32_t meta_model_size = 0;
    int has_meta = 0;
    if (version >= 3) {
        char meta_magic[4];
        if (fread(meta_magic, 1, 4, f) == 4 && memcmp(meta_magic, "META", 4) == 0) {
            has_meta = 1;
            uint32_t lang_id;
            fread(&lang_id, sizeof(uint32_t), 1, f);
            fread(meta_lang_name, 1, 16, f);
            meta_lang_name[15] = '\0';
            fread(&meta_model_size, sizeof(uint32_t), 1, f);
            uint32_t enc_dim_meta;
            fread(&enc_dim_meta, sizeof(uint32_t), 1, f);
            fread(&meta_ref_dur, sizeof(float), 1, f);
            fread(meta_voice_name, 1, 64, f);
            meta_voice_name[63] = '\0';
        }
    }

    fclose(f);

    /* File size */
    struct stat st;
    stat(path, &st);

    /* Extract just filename */
    const char *basename = strrchr(path, '/');
    basename = basename ? basename + 1 : path;

    printf("  %-30s  v%u  %3u frames (%.1fs ref)  %5.1f KB",
           basename, version, n_ref_frames, n_ref_frames / 12.5f,
           (float)st.st_size / 1024.0f);
    if (has_meta) {
        if (meta_voice_name[0])
            printf("  [%s]", meta_voice_name);
        if (meta_lang_name[0])
            printf("  lang=%s", meta_lang_name);
        printf("  model=%s", meta_model_size >= 2048 ? "1.7B" : (meta_model_size > 0 ? "0.6B" : "?"));
    }
    if (ref_text_len > 0)
        printf("  \"%s\"", ref_text);
    printf("\n");
    return 0;
}

/* List all .qvoice files in a directory */
static int list_voices(const char *dir_path) {
    DIR *dir = opendir(dir_path);
    if (!dir) {
        /* Maybe it's a single file */
        if (strstr(dir_path, ".qvoice")) {
            printf("Voice profiles:\n");
            if (print_qvoice_info(dir_path) != 0) {
                fprintf(stderr, "Error: %s is not a valid .qvoice file\n", dir_path);
                return 1;
            }
            return 0;
        }
        fprintf(stderr, "Error: cannot open directory %s\n", dir_path);
        return 1;
    }

    printf("Voice profiles in %s:\n\n", dir_path);
    int count = 0;
    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        size_t len = strlen(entry->d_name);
        if (len > 7 && strcmp(entry->d_name + len - 7, ".qvoice") == 0) {
            char fullpath[4096];
            snprintf(fullpath, sizeof(fullpath), "%s/%s", dir_path, entry->d_name);
            if (print_qvoice_info(fullpath) == 0)
                count++;
        }
    }
    closedir(dir);

    if (count == 0)
        printf("  (no .qvoice files found)\n");
    else
        printf("\n  %d voice profile(s)\n", count);
    return 0;
}

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

/* Write a tensor to .qvoice file, optionally as WDELTA (int8 compressed delta vs CV) */
static void write_tensor_impl(FILE *vf, FILE *cv_sf, const char *cv_hdr_json, size_t cv_data_off,
                               const char *tname, const void *ptr, size_t nbytes,
                               int use_wdelta, int is_bf16,
                               int64_t *total_bytes, int *count) {
    uint16_t nl = (uint16_t)strlen(tname);
    fwrite(&nl, sizeof(uint16_t), 1, vf);
    fwrite(tname, 1, nl, vf);
    uint32_t raw = (uint32_t)nbytes;
    fwrite(&raw, sizeof(uint32_t), 1, vf);

    if (use_wdelta && is_bf16 && cv_sf && cv_hdr_json) {
        /* Find this tensor in CV safetensors */
        char key[300];
        snprintf(key, sizeof(key), "\"%s\"", tname);
        const char *p = strstr(cv_hdr_json, key);
        long cvs = -1, cve = -1;
        if (p) {
            const char *doff = strstr(p, "data_offsets");
            if (doff) {
                const char *br = strchr(doff, '[');
                if (br) sscanf(br, "[%ld,%ld]", &cvs, &cve);
            }
        }
        if (cvs >= 0 && (uint32_t)(cve - cvs) == raw) {
            /* Read CV tensor */
            uint8_t *cv_buf = (uint8_t *)malloc(raw);
            fseek(cv_sf, cv_data_off + cvs, SEEK_SET);
            fread(cv_buf, 1, raw, cv_sf);
            /* Compute int16 delta (lossless — no clamping) */
            size_t n16 = raw / 2;
            int16_t *delta = (int16_t *)malloc(n16 * sizeof(int16_t));
            const uint16_t *base16 = (const uint16_t *)ptr;
            const uint16_t *cv16 = (const uint16_t *)cv_buf;
            for (size_t i = 0; i < n16; i++)
                delta[i] = (int16_t)((int)base16[i] - (int)cv16[i]);
            unsigned long delta_bytes = n16 * sizeof(int16_t);
            /* Compress delta with LZ4 (~7x faster decompress than zlib) */
            int lz4_bound = LZ4_compressBound((int)delta_bytes);
            uint8_t *compressed = (uint8_t *)malloc(lz4_bound);
            int comp_size = LZ4_compress_default((const char *)delta, (char *)compressed,
                                                  (int)delta_bytes, lz4_bound);
            /* dtype=4: int16 delta + LZ4 */
            uint8_t dtype = 4;
            fwrite(&dtype, 1, 1, vf);
            uint32_t csz = (uint32_t)comp_size;
            fwrite(&csz, sizeof(uint32_t), 1, vf);
            fwrite(compressed, 1, comp_size, vf);
            *total_bytes += comp_size;
            free(cv_buf); free(delta); free(compressed);
        } else {
            /* Fallback: raw */
            uint8_t dtype = 0;
            fwrite(&dtype, 1, 1, vf);
            uint32_t csz = raw;
            fwrite(&csz, sizeof(uint32_t), 1, vf);
            fwrite(ptr, 1, raw, vf);
            *total_bytes += raw;
        }
    } else {
        /* WFULL mode or F32 tensor: write raw */
        uint8_t dtype = 0;
        fwrite(&dtype, 1, 1, vf);
        uint32_t csz = raw;
        fwrite(&csz, sizeof(uint32_t), 1, vf);
        fwrite(ptr, 1, raw, vf);
        *total_bytes += raw;
    }
    (*count)++;
}

int main(int argc, char **argv) {
    const char *model_dir = NULL;
    const char *text = NULL;
    const char *output = "output.wav";
    int speaker_id = -1;
    const char *language = NULL;
    const char *instruct = NULL;
    float temperature = 0.5f;
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
    const char *ref_audio = NULL;
    const char *ref_text_str = NULL;
    int xvector_only = 0;
    const char *save_voice = NULL;
    const char *load_voice = NULL;
    const char *list_voices_dir = NULL;
    const char *delete_voice = NULL;
    const char *voice_name = NULL;
    const char *target_cv_dir = NULL;
    int ctx_greedy_warmup = 0;
    float max_ref_duration = 30.0f;  /* default: use first 30s of ref audio */
    int use_int8 = 0;
    int use_int4 = 0;
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
        {"ref-audio",     required_argument, 0, 1008},
        {"ref-text",      required_argument, 0, 1009},
        {"xvector-only",  no_argument,       0, 1010},
        {"save-voice",    required_argument, 0, 1011},
        {"load-voice",    required_argument, 0, 1012},
        {"max-ref-duration", required_argument, 0, 1013},
        {"silent",        no_argument,       0, 'S'},
        {"debug",         no_argument,       0, 'D'},
        {"list-voices",   required_argument, 0, 1016},
        {"delete-voice",  required_argument, 0, 1017},
        {"int8",          no_argument,       0, 1014},
        {"int4",          no_argument,       0, 1015},
        {"voice-name",    required_argument, 0, 1022},
        {"greedy-warmup", required_argument, 0, 1023},
        {"target-cv",     required_argument, 0, 1024},
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
            case 1008: ref_audio = optarg; break;
            case 1009: ref_text_str = optarg; break;
            case 1010: xvector_only = 1; break;
            case 1011: save_voice = optarg; break;
            case 1012: load_voice = optarg; break;
            case 1013: max_ref_duration = (float)atof(optarg); break;
            case 1014: use_int8 = 1; break;
            case 1015: use_int4 = 1; break;
            case 1022: voice_name = optarg; break;
            case 1023: { int gw = atoi(optarg); ctx_greedy_warmup = gw; } break;
            case 1024: target_cv_dir = optarg; break;
            case 1016: list_voices_dir = optarg; break;
            case 1017: delete_voice = optarg; break;
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
                fprintf(stderr, "  --ref-audio <path>         Reference audio for voice cloning (Base model)\n");
                fprintf(stderr, "  --xvector-only             Use speaker embedding only (no ref text/codes)\n");
                fprintf(stderr, "  --save-voice <path>        Save voice (.qvoice = full profile, .bin = embedding only)\n");
                fprintf(stderr, "                             Without --text: create voice profile and exit\n");
                fprintf(stderr, "  --load-voice <path>        Load voice (.qvoice = full profile, .bin = embedding only)\n");
                fprintf(stderr, "  --voice-name <name>        Name for the voice (stored in .qvoice metadata)\n");
                fprintf(stderr, "  --list-voices <dir>        List .qvoice files in directory\n");
                fprintf(stderr, "  --delete-voice <path>      Delete a .qvoice file\n");
                fprintf(stderr, "  --max-ref-duration <secs>  Max ref audio for embedding (default: 30, 0=all)\n");
                fprintf(stderr, "  --int8                     INT8 quantized Talker + Code Predictor\n");
                fprintf(stderr, "  --int4                     Q4_0 quantized Talker (1.7B only, smallest memory)\n");
                fprintf(stderr, "  -S, --silent               Silent mode\n");
                fprintf(stderr, "  -D, --debug                Debug mode\n");
                return opt == 'h' ? 0 : 1;
        }
    }

    /* Voice library management (no model loading needed) */
    if (list_voices_dir) {
        return list_voices(list_voices_dir);
    }
    if (delete_voice) {
        /* Validate it's a .qvoice file */
        size_t dlen = strlen(delete_voice);
        if (dlen <= 7 || strcmp(delete_voice + dlen - 7, ".qvoice") != 0) {
            fprintf(stderr, "Error: --delete-voice only works with .qvoice files\n");
            return 1;
        }
        /* Check it exists and is valid */
        FILE *vf = fopen(delete_voice, "rb");
        if (!vf) {
            fprintf(stderr, "Error: file not found: %s\n", delete_voice);
            return 1;
        }
        char magic[4];
        int valid = (fread(magic, 1, 4, vf) == 4 && memcmp(magic, "QVCE", 4) == 0);
        fclose(vf);
        if (!valid) {
            fprintf(stderr, "Error: %s is not a valid .qvoice file\n", delete_voice);
            return 1;
        }
        if (remove(delete_voice) != 0) {
            fprintf(stderr, "Error: failed to delete %s\n", delete_voice);
            return 1;
        }
        printf("Deleted %s\n", delete_voice);
        return 0;
    }

    if (!model_dir) {
        fprintf(stderr, "Error: --model-dir is required\n");
        return 1;
    }
    /* --save-voice without --text = create voice only (no generation) */
    int create_voice_only = (save_voice && !text && serve_port <= 0);
    if (!text && serve_port <= 0 && !create_voice_only) {
        fprintf(stderr, "Error: --text or --serve is required\n");
        return 1;
    }

    if (!silent) {
        fprintf(stderr, "Model dir: %s\n", model_dir);
        if (text) fprintf(stderr, "Text: \"%s\"\n", text);
        fprintf(stderr, "Output: %s\n", output);
    }

    /* Early validation: check ref-audio format BEFORE loading the model.
     * This saves the user from waiting for model load (~2s) only to discover
     * their input file is MP4/MP3/wrong format. */
    if (ref_audio) {
        /* Extension check */
        const char *ext = strrchr(ref_audio, '.');
        if (ext && (strcasecmp(ext, ".mp4") == 0 || strcasecmp(ext, ".m4a") == 0 ||
                    strcasecmp(ext, ".mp3") == 0 || strcasecmp(ext, ".ogg") == 0 ||
                    strcasecmp(ext, ".opus") == 0 || strcasecmp(ext, ".flac") == 0 ||
                    strcasecmp(ext, ".aac") == 0 || strcasecmp(ext, ".wma") == 0 ||
                    strcasecmp(ext, ".webm") == 0 || strcasecmp(ext, ".mkv") == 0 ||
                    strcasecmp(ext, ".avi") == 0 || strcasecmp(ext, ".mov") == 0)) {
            fprintf(stderr, "Error: %s is not a WAV file (detected %s format)\n", ref_audio, ext);
            fprintf(stderr, "Voice cloning requires 24 kHz WAV (PCM, 16-bit, mono).\n");
            fprintf(stderr, "Convert first:\n");
            fprintf(stderr, "  ffmpeg -i \"%s\" -ar 24000 -ac 1 output.wav\n", ref_audio);
            return 1;
        }
        /* Quick header check — read first 12 bytes */
        FILE *check_f = fopen(ref_audio, "rb");
        if (check_f) {
            unsigned char hdr[12];
            size_t n = fread(hdr, 1, 12, check_f);
            fclose(check_f);
            int bad = 0;
            if (n >= 8 && memcmp(hdr + 4, "ftyp", 4) == 0) {
                fprintf(stderr, "Error: %s is an MP4/M4A file, not a WAV file\n", ref_audio);
                bad = 1;
            } else if (n >= 3 && hdr[0] == 0xFF && (hdr[1] & 0xE0) == 0xE0) {
                fprintf(stderr, "Error: %s is an MP3 file, not a WAV file\n", ref_audio);
                bad = 1;
            } else if (n >= 4 && memcmp(hdr, "OggS", 4) == 0) {
                fprintf(stderr, "Error: %s is an OGG/Opus file, not a WAV file\n", ref_audio);
                bad = 1;
            } else if (n >= 4 && memcmp(hdr, "fLaC", 4) == 0) {
                fprintf(stderr, "Error: %s is a FLAC file, not a WAV file\n", ref_audio);
                bad = 1;
            } else if (n >= 3 && memcmp(hdr, "ID3", 3) == 0) {
                fprintf(stderr, "Error: %s is an MP3 file (ID3 tagged), not a WAV file\n", ref_audio);
                bad = 1;
            } else if (n >= 4 && memcmp(hdr, "RIFF", 4) != 0) {
                fprintf(stderr, "Error: %s is not a WAV file (unrecognized format)\n", ref_audio);
                bad = 1;
            }
            if (bad) {
                fprintf(stderr, "Voice cloning requires 24 kHz WAV (PCM, 16-bit, mono).\n");
                fprintf(stderr, "Convert first:\n");
                fprintf(stderr, "  ffmpeg -i \"%s\" -ar 24000 -ac 1 output.wav\n", ref_audio);
                return 1;
            }
        }
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
    ctx->use_int8 = use_int8;
    ctx->use_int4 = use_int4;

    if (speaker_id >= 0) ctx->speaker_id = speaker_id;
    if (language) ctx->language_id = qwen_tts_language_id(language);
    if (seed >= 0) ctx->seed = (uint32_t)seed;
    if (max_duration > 0) ctx->max_tokens = (int)(max_duration * 12.5f);
    if (ctx_greedy_warmup > 0) ctx->greedy_warmup = ctx_greedy_warmup;
    if (voice_design) {
        if (ctx->config.hidden_size < 2048) {
            fprintf(stderr, "Error: --voice-design requires the 1.7B VoiceDesign model\n");
            fprintf(stderr, "Download it with: ./download_model.sh --model voice-design\n");
            qwen_tts_unload(ctx);
            return 1;
        }
        ctx->voice_design = 1;
    }
    /* Voice cloning setup */
    if (ref_audio || load_voice) {
        if (!ctx->is_base_model && ref_audio) {
            /* --ref-audio requires speaker encoder (Base model only) */
            fprintf(stderr, "Error: --ref-audio requires a Base model (not CustomVoice)\n");
            fprintf(stderr, "Extract a speaker embedding first with the Base model:\n");
            fprintf(stderr, "  ./qwen_tts -d qwen3-tts-1.7b-base --ref-audio %s --save-voice voice.bin\n", ref_audio);
            fprintf(stderr, "Then use it here with --load-voice voice.bin\n");
            qwen_tts_unload(ctx);
            return 1;
        }
        if (!ctx->is_base_model && load_voice) {
            /* Cross-model voice injection: use ECAPA-TDNN embedding from Base model
             * in CustomVoice/VoiceDesign model. This works because the embedding spaces
             * are compatible (cosine similarity ~0.94 between ECAPA and discrete speakers). */
            if (!silent)
                fprintf(stderr, "Cross-model voice: loading speaker embedding into %s model\n",
                        ctx->voice_design ? "VoiceDesign" : "CustomVoice");
        }
        ctx->voice_clone = 1;
        ctx->xvector_only = xvector_only ? 1 : (ref_text_str ? 0 : 1);
        ctx->max_ref_seconds = max_ref_duration;
        if (ref_audio) ctx->ref_audio_path = strdup(ref_audio);
        if (ref_text_str) ctx->ref_text = strdup(ref_text_str);

        /* Use speaker encoder dim if available, otherwise model hidden_size */
        int enc_dim = ctx->speaker_enc.enc_dim > 0 ? ctx->speaker_enc.enc_dim : ctx->config.hidden_size;
        ctx->speaker_embedding = (float *)malloc(enc_dim * sizeof(float));
        if (!ctx->speaker_embedding) {
            fprintf(stderr, "Error: failed to allocate speaker embedding\n");
            qwen_tts_unload(ctx);
            return 1;
        }

        /* Check if file has .qvoice extension */
        int load_is_qvoice = load_voice && strlen(load_voice) > 7 &&
            strcmp(load_voice + strlen(load_voice) - 7, ".qvoice") == 0;
        int save_is_qvoice = save_voice && strlen(save_voice) > 7 &&
            strcmp(save_voice + strlen(save_voice) - 7, ".qvoice") == 0;

        if (load_voice) {
            if (load_is_qvoice) {
                /* Load .qvoice file: speaker embedding + ref_codes + ref_text */
                FILE *vf = fopen(load_voice, "rb");
                if (!vf) {
                    fprintf(stderr, "Error: cannot open voice file %s\n", load_voice);
                    qwen_tts_unload(ctx);
                    return 1;
                }
                /* Read and validate magic */
                char magic[4];
                uint32_t version;
                if (fread(magic, 1, 4, vf) != 4 || memcmp(magic, "QVCE", 4) != 0) {
                    fprintf(stderr, "Error: %s is not a valid .qvoice file\n", load_voice);
                    fclose(vf); qwen_tts_unload(ctx); return 1;
                }
                if (fread(&version, sizeof(uint32_t), 1, vf) != 1 || (version < 1 || version > 3)) {
                    fprintf(stderr, "Error: unsupported .qvoice version %u\n", version);
                    fclose(vf); qwen_tts_unload(ctx); return 1;
                }
                /* Read speaker embedding (v2 has enc_dim header, v1 assumes model's enc_dim) */
                int file_enc_dim = enc_dim;
                if (version >= 2) {
                    uint32_t d;
                    if (fread(&d, sizeof(uint32_t), 1, vf) != 1) {
                        fprintf(stderr, "Error: failed to read enc_dim from %s\n", load_voice);
                        fclose(vf); qwen_tts_unload(ctx); return 1;
                    }
                    file_enc_dim = (int)d;
                }
                if (file_enc_dim != enc_dim) {
                    fprintf(stderr, "Error: .qvoice has enc_dim=%d but model expects %d\n", file_enc_dim, enc_dim);
                    fprintf(stderr, "Re-create the .qvoice with the matching Base model.\n");
                    fclose(vf); qwen_tts_unload(ctx); return 1;
                }
                if (fread(ctx->speaker_embedding, sizeof(float), enc_dim, vf) != (size_t)enc_dim) {
                    fprintf(stderr, "Error: failed to read speaker embedding from %s\n", load_voice);
                    fclose(vf); qwen_tts_unload(ctx); return 1;
                }
                /* Read ref_text */
                uint32_t ref_text_len;
                if (fread(&ref_text_len, sizeof(uint32_t), 1, vf) != 1) {
                    fclose(vf); qwen_tts_unload(ctx); return 1;
                }
                if (ref_text_len > 0) {
                    char *rt = (char *)malloc(ref_text_len + 1);
                    if (fread(rt, 1, ref_text_len, vf) != ref_text_len) {
                        free(rt); fclose(vf); qwen_tts_unload(ctx); return 1;
                    }
                    rt[ref_text_len] = '\0';
                    free(ctx->ref_text);
                    ctx->ref_text = rt;
                }
                /* Read ref_codes */
                uint32_t n_ref_frames;
                if (fread(&n_ref_frames, sizeof(uint32_t), 1, vf) != 1) {
                    fclose(vf); qwen_tts_unload(ctx); return 1;
                }
                if (n_ref_frames > 0) {
                    int n_codes = (int)n_ref_frames * 16;
                    ctx->cached_ref_codes = (int *)malloc(n_codes * sizeof(int));
                    if (fread(ctx->cached_ref_codes, sizeof(int), n_codes, vf) != (size_t)n_codes) {
                        free(ctx->cached_ref_codes);
                        ctx->cached_ref_codes = NULL;
                        fclose(vf); qwen_tts_unload(ctx); return 1;
                    }
                    ctx->cached_ref_n_frames = (int)n_ref_frames;
                    ctx->xvector_only = 0;  /* ICL mode */
                }
                /* Save original mmap'd pointers for WDELTA (before WOVR modifies them) */
                uint16_t *orig_tok_emb = ctx->tok_embeddings_bf16;
                uint16_t *orig_fc1 = ctx->text_proj_fc1_bf16;
                uint16_t *orig_fc2 = ctx->text_proj_fc2_bf16;
                uint16_t *orig_codec = ctx->codec_embedding_bf16;
                uint16_t *orig_codec_head = ctx->codec_head_bf16;
                (void)orig_tok_emb; (void)orig_fc1; (void)orig_fc2;
                (void)orig_codec; (void)orig_codec_head;

                /* v3 metadata */
                char meta_lang_name[16] = {0};
                char meta_voice_name[64] = {0};
                uint32_t meta_lang_id = 0;
                uint32_t meta_model_size = 0;
                float meta_ref_dur = 0;
                int has_meta = 0;
                int has_tpad = 0;
                if (version >= 3) {
                    char meta_magic[4];
                    if (fread(meta_magic, 1, 4, vf) == 4 && memcmp(meta_magic, "META", 4) == 0) {
                        has_meta = 1;
                        fread(&meta_lang_id, sizeof(uint32_t), 1, vf);
                        fread(meta_lang_name, 1, 16, vf);
                        meta_lang_name[15] = '\0';
                        fread(&meta_model_size, sizeof(uint32_t), 1, vf);
                        uint32_t meta_enc_dim;
                        fread(&meta_enc_dim, sizeof(uint32_t), 1, vf);
                        fread(&meta_ref_dur, sizeof(float), 1, vf);
                        fread(meta_voice_name, 1, 64, vf);
                        meta_voice_name[63] = '\0';
                        uint32_t meta_flags;
                        fread(&meta_flags, sizeof(uint32_t), 1, vf);
                        /* Auto-set language from metadata if not specified on CLI */
                        if (!language && meta_lang_id > 0) {
                            ctx->language_id = (int)meta_lang_id;
                            if (!silent)
                                fprintf(stderr, "  Auto-set language from voice: %s\n", meta_lang_name);
                        }
                        /* Warn if CLI language doesn't match voice metadata */
                        if (language && meta_lang_id > 0 && ctx->language_id != (int)meta_lang_id) {
                            fprintf(stderr, "WARNING: voice was created with language '%s' but you specified '%s'\n",
                                    meta_lang_name, language);
                            fprintf(stderr, "  Voice fidelity may be reduced. Consider using -l %s\n", meta_lang_name);
                        }
                        /* Warn if model size doesn't match */
                        if (meta_model_size > 0 && meta_model_size != (uint32_t)ctx->config.hidden_size) {
                            fprintf(stderr, "WARNING: voice was created with model hidden=%u but current model has hidden=%d\n",
                                    meta_model_size, ctx->config.hidden_size);
                            fprintf(stderr, "  Cross-size injection may reduce quality.\n");
                        }
                    }
                    /* TPAD section: source model's tts_pad/bos/eos embeddings. */
                    char tpad_magic[4];
                    if (fread(tpad_magic, 1, 4, vf) == 4 && memcmp(tpad_magic, "TPAD", 4) == 0) {
                        uint32_t tpad_hidden;
                        if (fread(&tpad_hidden, sizeof(uint32_t), 1, vf) == 1 &&
                            (int)tpad_hidden == ctx->config.hidden_size &&
                            ctx->cached_tts_pad_embed) {
                            int h = ctx->config.hidden_size;
                            fread(ctx->cached_tts_pad_embed, sizeof(float), h, vf);
                            fread(ctx->cached_tts_bos_embed, sizeof(float), h, vf);
                            fread(ctx->cached_tts_eos_embed, sizeof(float), h, vf);
                            has_tpad = 1;
                            if (!silent)
                                fprintf(stderr, "  Loaded source tts_pad/bos/eos embeddings\n");
                        }
                    }
                    /* WOVR section: source model weights override.
                     * Replaces text_projection and codec_embedding with source model's
                     * weights to eliminate ALL per-frame divergence from weight diffs. */
                    char wovr_magic[4];
                    if (fread(wovr_magic, 1, 4, vf) == 4 && memcmp(wovr_magic, "WOVR", 4) == 0) {
                        uint32_t wh, wth, wcv;
                        if (fread(&wh, sizeof(uint32_t), 1, vf) == 1 &&
                            fread(&wth, sizeof(uint32_t), 1, vf) == 1 &&
                            fread(&wcv, sizeof(uint32_t), 1, vf) == 1 &&
                            (int)wh == ctx->config.hidden_size &&
                            (int)wth == ctx->config.text_hidden_size) {
                            int h = (int)wh, th = (int)wth, cv = (int)wcv;
                            /* Allocate owned copies (can't write to mmap'd weights) */
                            uint16_t *fc1 = (uint16_t *)malloc((size_t)th * th * sizeof(uint16_t));
                            float *fc1_b = (float *)malloc((size_t)th * sizeof(float));
                            uint16_t *fc2 = (uint16_t *)malloc((size_t)h * th * sizeof(uint16_t));
                            float *fc2_b = (float *)malloc((size_t)h * sizeof(float));
                            uint16_t *ce = (uint16_t *)malloc((size_t)cv * h * sizeof(uint16_t));
                            if (fc1 && fc1_b && fc2 && fc2_b && ce) {
                                fread(fc1, sizeof(uint16_t), (size_t)th * th, vf);
                                fread(fc1_b, sizeof(float), th, vf);
                                fread(fc2, sizeof(uint16_t), (size_t)h * th, vf);
                                fread(fc2_b, sizeof(float), h, vf);
                                fread(ce, sizeof(uint16_t), (size_t)cv * h, vf);
                                /* Override model weights */
                                ctx->text_proj_fc1_bf16 = fc1;
                                ctx->text_proj_fc1_bias = fc1_b;
                                ctx->text_proj_fc2_bf16 = fc2;
                                ctx->text_proj_fc2_bias = fc2_b;
                                /* Override codec_embedding: need owned copy since original is mmap'd */
                                int full_codec_vocab = ctx->config.codec_vocab_size;
                                uint16_t *ce_full = (uint16_t *)malloc((size_t)full_codec_vocab * h * sizeof(uint16_t));
                                if (ce_full) {
                                    /* Copy entire original table (includes speaker presets) */
                                    memcpy(ce_full, ctx->codec_embedding_bf16, (size_t)full_codec_vocab * h * sizeof(uint16_t));
                                    /* Override codebook entries 0-2047 with source model's */
                                    int copy_entries = cv < full_codec_vocab ? cv : full_codec_vocab;
                                    memcpy(ce_full, ce, (size_t)copy_entries * h * sizeof(uint16_t));
                                    ctx->codec_embedding_bf16 = ce_full;
                                }
                                free(ce);
                                /* Recompute tts_pad/bos/eos with new text_projection
                                 * (only if TPAD section wasn't loaded — TPAD has exact Base values) */
                                int tts_pad_id = 151671, tts_bos_id = 151672, tts_eos_id = 151673;
                                float *tmp1 = (float *)malloc(th * sizeof(float));
                                float *tmp2 = (float *)malloc(th * sizeof(float));
                                if (tmp1 && tmp2 && !has_tpad) {
                                    for (int tid_i = 0; tid_i < 3; tid_i++) {
                                        int tid = (tid_i == 0) ? tts_pad_id : (tid_i == 1) ? tts_bos_id : tts_eos_id;
                                        float *out = (tid_i == 0) ? ctx->cached_tts_pad_embed :
                                                     (tid_i == 1) ? ctx->cached_tts_bos_embed :
                                                                    ctx->cached_tts_eos_embed;
                                        /* text_embedding lookup */
                                        const uint16_t *emb = ctx->tok_embeddings_bf16 + (int64_t)tid * th;
                                        for (int j = 0; j < th; j++) {
                                            uint32_t bits = (uint32_t)emb[j] << 16;
                                            memcpy(&tmp1[j], &bits, 4);
                                        }
                                        /* fc1 + bias + SiLU */
                                        for (int i = 0; i < th; i++) {
                                            float sum = fc1_b[i];
                                            const uint16_t *row = fc1 + (size_t)i * th;
                                            for (int j = 0; j < th; j++) {
                                                uint32_t bits = (uint32_t)row[j] << 16;
                                                float w; memcpy(&w, &bits, 4);
                                                sum += w * tmp1[j];
                                            }
                                            /* SiLU = x * sigmoid(x) */
                                            tmp2[i] = sum / (1.0f + expf(-sum));
                                        }
                                        /* fc2 + bias */
                                        for (int i = 0; i < h; i++) {
                                            float sum = fc2_b[i];
                                            const uint16_t *row = fc2 + (size_t)i * th;
                                            for (int j = 0; j < th; j++) {
                                                uint32_t bits = (uint32_t)row[j] << 16;
                                                float w; memcpy(&w, &bits, 4);
                                                sum += w * tmp2[j];
                                            }
                                            out[i] = sum;
                                        }
                                    }
                                }
                                free(tmp1); free(tmp2);
                                if (!silent) {
                                    int64_t wovr_bytes = (int64_t)th*th*2 + th*4 + (int64_t)h*th*2 + h*4 + (int64_t)cv*h*2;
                                    fprintf(stderr, "  Loaded source model weights (%.1f MB) — full cross-model fidelity\n",
                                            wovr_bytes / 1024.0f / 1024.0f);
                                    fprintf(stderr, "  tts_pad_embed[:3]=[%.6f,%.6f,%.6f] (recomputed from source weights)\n",
                                            ctx->cached_tts_pad_embed[0], ctx->cached_tts_pad_embed[1], ctx->cached_tts_pad_embed[2]);
                                }
                            } else {
                                free(fc1); free(fc1_b); free(fc2); free(fc2_b); free(ce);
                            }
                        }
                    }
                    /* WFULL section: full talker weight override from source model.
                     * Replaces ALL talker weights in the target model with source weights
                     * stored in the .qvoice file. This achieves bit-identical output to
                     * the source model without requiring it to be present. */
                    char wfull_magic[5] = {0};
                    int is_wdelta = 0;
                    size_t magic_read = fread(wfull_magic, 1, 5, vf);
                    if (magic_read >= 4 && memcmp(wfull_magic, "WDLT", 4) == 0) {
                        /* WDELTA format (4-byte magic) — push back the 5th byte */
                        fseek(vf, -1, SEEK_CUR);
                        is_wdelta = 1;
                    }
                    if ((magic_read == 5 && memcmp(wfull_magic, "WFULL", 5) == 0) || is_wdelta) {
                        /* WDELTA: validate target model */
                        if (is_wdelta) {
                            uint32_t target_h;
                            if (fread(&target_h, sizeof(uint32_t), 1, vf) == 1) {
                                if (ctx->is_base_model) {
                                    fprintf(stderr, "ERROR: this .qvoice contains weight deltas for CustomVoice,\n");
                                    fprintf(stderr, "  but you're loading it on a Base model. This would corrupt weights.\n");
                                    fprintf(stderr, "  Use --load-voice on the CustomVoice model instead:\n");
                                    fprintf(stderr, "    ./qwen_tts -d qwen3-tts-%s --load-voice %s ...\n",
                                            target_h >= 2048 ? "1.7b" : "0.6b", load_voice);
                                    fclose(vf); qwen_tts_unload(ctx); return 1;
                                }
                                if ((int)target_h != ctx->config.hidden_size) {
                                    fprintf(stderr, "ERROR: .qvoice was created for %s model (hidden=%u)\n",
                                            target_h >= 2048 ? "1.7B" : "0.6B", target_h);
                                    fprintf(stderr, "  but current model has hidden=%d. Recreate with matching --target-cv.\n",
                                            ctx->config.hidden_size);
                                    fclose(vf); qwen_tts_unload(ctx); return 1;
                                }
                            }
                        }
                        uint32_t n_tensors;
                        fread(&n_tensors, sizeof(uint32_t), 1, vf);
                        int loaded = 0;
                        int64_t wfull_bytes = 0;
                        int h = ctx->config.hidden_size;
                        int th = ctx->config.text_hidden_size;
                        int nl = ctx->config.num_layers;
                        int q_dim = ctx->config.num_heads * ctx->config.head_dim;
                        int kv_dim = ctx->config.num_kv_heads * ctx->config.head_dim;
                        int inter = ctx->config.intermediate_size;

                        for (uint32_t t = 0; t < n_tensors; t++) {
                            uint16_t name_len;
                            if (fread(&name_len, sizeof(uint16_t), 1, vf) != 1) break;
                            char tname[256] = {0};
                            if (name_len >= 256) break;
                            fread(tname, 1, name_len, vf);
                            uint32_t data_bytes;
                            fread(&data_bytes, sizeof(uint32_t), 1, vf);

                            /* Match tensor name to ctx field and replace */
                            void **target_ptr = NULL;
                            int is_f32 = 0;

                            /* Global tensors */
                            if (strcmp(tname, "talker.model.text_embedding.weight") == 0)
                                target_ptr = (void **)&ctx->tok_embeddings_bf16;
                            else if (strcmp(tname, "talker.text_projection.linear_fc1.weight") == 0)
                                target_ptr = (void **)&ctx->text_proj_fc1_bf16;
                            else if (strcmp(tname, "talker.text_projection.linear_fc1.bias") == 0)
                                { target_ptr = (void **)&ctx->text_proj_fc1_bias; is_f32 = 1; }
                            else if (strcmp(tname, "talker.text_projection.linear_fc2.weight") == 0)
                                target_ptr = (void **)&ctx->text_proj_fc2_bf16;
                            else if (strcmp(tname, "talker.text_projection.linear_fc2.bias") == 0)
                                { target_ptr = (void **)&ctx->text_proj_fc2_bias; is_f32 = 1; }
                            else if (strcmp(tname, "talker.model.codec_embedding.weight") == 0)
                                target_ptr = (void **)&ctx->codec_embedding_bf16;
                            else if (strcmp(tname, "talker.codec_head.weight") == 0)
                                target_ptr = (void **)&ctx->codec_head_bf16;
                            else if (strcmp(tname, "talker.model.norm.weight") == 0)
                                { target_ptr = (void **)&ctx->talker_norm; is_f32 = 1; }
                            /* Code Predictor tensors */
                            else if (strstr(tname, "code_predictor.model.norm.weight"))
                                { target_ptr = (void **)&ctx->cp_norm; is_f32 = 1; }
                            else if (strstr(tname, "code_predictor.small_to_mtp_projection.weight"))
                                target_ptr = (void **)&ctx->cp_mtp_proj_bf16;
                            else if (strstr(tname, "code_predictor.small_to_mtp_projection.bias"))
                                { target_ptr = (void **)&ctx->cp_mtp_proj_bias; is_f32 = 1; }
                            else if (strstr(tname, "code_predictor.model.codec_embedding.")) {
                                int ci = -1;
                                sscanf(tname, "talker.code_predictor.model.codec_embedding.%d.", &ci);
                                if (ci >= 0 && ci < 15)
                                    target_ptr = (void **)&ctx->cp_codec_emb_bf16[ci];
                            }
                            else if (strstr(tname, "code_predictor.lm_head.")) {
                                int ci = -1;
                                sscanf(tname, "talker.code_predictor.lm_head.%d.", &ci);
                                if (ci >= 0 && ci < 15)
                                    target_ptr = (void **)&ctx->cp_lm_head_bf16[ci];
                            }
                            else if (strstr(tname, "code_predictor.model.layers.")) {
                                int layer = -1;
                                sscanf(tname, "talker.code_predictor.model.layers.%d.", &layer);
                                if (layer >= 0 && layer < ctx->config.cp_num_layers) {
                                    qwen_cp_layer_t *cl = &ctx->cp_layers[layer];
                                    /* Find last occurrence of the weight type suffix */
                                    char *suffix = strstr(tname, "self_attn.");
                                    if (!suffix) suffix = strstr(tname, "mlp.");
                                    if (!suffix) suffix = strstr(tname, "input_layernorm");
                                    if (!suffix) suffix = strstr(tname, "post_attention");
                                    if (suffix) {
                                        if (strstr(suffix, "q_proj.weight"))
                                            target_ptr = (void **)&cl->wq_bf16;
                                        else if (strstr(suffix, "k_proj.weight"))
                                            target_ptr = (void **)&cl->wk_bf16;
                                        else if (strstr(suffix, "v_proj.weight"))
                                            target_ptr = (void **)&cl->wv_bf16;
                                        else if (strstr(suffix, "o_proj.weight"))
                                            target_ptr = (void **)&cl->wo_bf16;
                                        else if (strstr(suffix, "q_norm.weight"))
                                            { target_ptr = (void **)&cl->q_norm; is_f32 = 1; }
                                        else if (strstr(suffix, "k_norm.weight"))
                                            { target_ptr = (void **)&cl->k_norm; is_f32 = 1; }
                                        else if (strstr(suffix, "input_layernorm.weight"))
                                            { target_ptr = (void **)&cl->input_norm; is_f32 = 1; }
                                        else if (strstr(suffix, "post_attention_layernorm.weight"))
                                            { target_ptr = (void **)&cl->post_attn_norm; is_f32 = 1; }
                                        else if (strstr(suffix, "gate_proj.weight"))
                                            target_ptr = (void **)&cl->gate_bf16;
                                        else if (strstr(suffix, "up_proj.weight"))
                                            target_ptr = (void **)&cl->up_bf16;
                                        else if (strstr(suffix, "down_proj.weight"))
                                            target_ptr = (void **)&cl->down_bf16;
                                    }
                                }
                            }
                            else {
                                /* Per-layer tensors: talker.model.layers.N.xxx */
                                int layer = -1;
                                sscanf(tname, "talker.model.layers.%d.", &layer);
                                if (layer >= 0 && layer < nl) {
                                    qwen_talker_layer_t *l = &ctx->layers[layer];
                                    char *suffix = strstr(tname, "self_attn.");
                                    if (!suffix) suffix = strstr(tname, "mlp.");
                                    if (!suffix) suffix = strstr(tname, "input_layernorm");
                                    if (!suffix) suffix = strstr(tname, "post_attention");
                                    if (suffix) {
                                        if (strstr(suffix, "q_proj.weight"))
                                            target_ptr = (void **)&l->wq_bf16;
                                        else if (strstr(suffix, "k_proj.weight"))
                                            target_ptr = (void **)&l->wk_bf16;
                                        else if (strstr(suffix, "v_proj.weight"))
                                            target_ptr = (void **)&l->wv_bf16;
                                        else if (strstr(suffix, "o_proj.weight"))
                                            target_ptr = (void **)&l->wo_bf16;
                                        else if (strstr(suffix, "q_norm.weight"))
                                            { target_ptr = (void **)&l->q_norm; is_f32 = 1; }
                                        else if (strstr(suffix, "k_norm.weight"))
                                            { target_ptr = (void **)&l->k_norm; is_f32 = 1; }
                                        else if (strstr(suffix, "input_layernorm.weight"))
                                            { target_ptr = (void **)&l->input_norm; is_f32 = 1; }
                                        else if (strstr(suffix, "post_attention_layernorm.weight"))
                                            { target_ptr = (void **)&l->post_attn_norm; is_f32 = 1; }
                                        else if (strstr(suffix, "gate_proj.weight"))
                                            target_ptr = (void **)&l->gate_bf16;
                                        else if (strstr(suffix, "up_proj.weight"))
                                            target_ptr = (void **)&l->up_bf16;
                                        else if (strstr(suffix, "down_proj.weight"))
                                            target_ptr = (void **)&l->down_bf16;
                                    }
                                }
                            }

                            /* Read dtype flag and compressed size */
                            uint8_t dtype_flag = 0;
                            uint32_t compressed_size = data_bytes;
                            fread(&dtype_flag, 1, 1, vf);
                            fread(&compressed_size, sizeof(uint32_t), 1, vf);

                            if (target_ptr) {
                                if (dtype_flag == 4) {
                                    /* WDELTA: LZ4-compressed int16 deltas vs CV weights */
                                    uint8_t *lz4_data = (uint8_t *)malloc(compressed_size);
                                    size_t n16 = data_bytes / 2;
                                    uint16_t *result = (uint16_t *)malloc(data_bytes);
                                    int16_t *delta16 = (int16_t *)malloc(n16 * sizeof(int16_t));
                                    /* Use ORIGINAL mmap'd CV weight (before WOVR modified it). */
                                    const uint16_t *cv_orig = (const uint16_t *)*target_ptr;
                                    if (strcmp(tname, "talker.model.text_embedding.weight") == 0)
                                        cv_orig = orig_tok_emb;
                                    else if (strcmp(tname, "talker.text_projection.linear_fc1.weight") == 0)
                                        cv_orig = orig_fc1;
                                    else if (strcmp(tname, "talker.text_projection.linear_fc2.weight") == 0)
                                        cv_orig = orig_fc2;
                                    else if (strcmp(tname, "talker.model.codec_embedding.weight") == 0)
                                        cv_orig = orig_codec;
                                    else if (strcmp(tname, "talker.codec_head.weight") == 0)
                                        cv_orig = orig_codec_head;
                                    if (lz4_data && result && delta16 &&
                                        fread(lz4_data, 1, compressed_size, vf) == compressed_size) {
                                        LZ4_decompress_safe((const char *)lz4_data, (char *)delta16,
                                                             (int)compressed_size, (int)(n16 * sizeof(int16_t)));
                                        for (size_t i = 0; i < n16; i++)
                                            result[i] = (uint16_t)((int)cv_orig[i] + (int)delta16[i]);
                                        *target_ptr = result;
                                        loaded++;
                                        wfull_bytes += compressed_size;
                                    } else {
                                        free(result);
                                        fseek(vf, compressed_size, SEEK_CUR);
                                    }
                                    free(lz4_data); free(delta16);
                                } else if (dtype_flag == 2 || dtype_flag == 3) {
                                    /* Legacy zlib-compressed deltas — no longer supported */
                                    fprintf(stderr, "Error: .qvoice uses legacy zlib compression (dtype=%d)\n", dtype_flag);
                                    fprintf(stderr, "Recreate with: --target-cv (now uses LZ4)\n");
                                    fseek(vf, compressed_size, SEEK_CUR);
                                } else {
                                    /* WFULL: raw data, just read and replace */
                                    void *buf = malloc(compressed_size);
                                    if (buf && fread(buf, 1, compressed_size, vf) == compressed_size) {
                                        *target_ptr = buf;
                                        loaded++;
                                        wfull_bytes += compressed_size;
                                    } else {
                                        free(buf);
                                        fseek(vf, compressed_size, SEEK_CUR);
                                    }
                                }
                            } else {
                                /* Unknown tensor — skip */
                                fseek(vf, compressed_size, SEEK_CUR);
                            }
                        }

                        /* Rebuild gate_up_fused for all talker layers */
                        for (int li = 0; li < nl; li++) {
                            qwen_talker_layer_t *l = &ctx->layers[li];
                            if (l->gate_bf16 && l->up_bf16 && l->gate_up_fused_bf16) {
                                size_t row_bytes = (size_t)h * sizeof(uint16_t);
                                for (int r = 0; r < inter; r++) {
                                    memcpy(l->gate_up_fused_bf16 + (size_t)(2*r)*h,
                                           l->gate_bf16 + (size_t)r*h, row_bytes);
                                    memcpy(l->gate_up_fused_bf16 + (size_t)(2*r+1)*h,
                                           l->up_bf16 + (size_t)r*h, row_bytes);
                                }
                            }
                        }
                        /* Rebuild gate_up_fused for all CP layers */
                        {
                            int cp_h2 = ctx->config.cp_hidden_size;
                            int cp_inter2 = ctx->config.cp_intermediate_size;
                            for (int li = 0; li < ctx->config.cp_num_layers; li++) {
                                qwen_cp_layer_t *cl = &ctx->cp_layers[li];
                                if (cl->gate_bf16 && cl->up_bf16 && cl->gate_up_fused_bf16) {
                                    size_t row_bytes = (size_t)cp_h2 * sizeof(uint16_t);
                                    for (int r = 0; r < cp_inter2; r++) {
                                        memcpy(cl->gate_up_fused_bf16 + (size_t)(2*r)*cp_h2,
                                               cl->gate_bf16 + (size_t)r*cp_h2, row_bytes);
                                        memcpy(cl->gate_up_fused_bf16 + (size_t)(2*r+1)*cp_h2,
                                               cl->up_bf16 + (size_t)r*cp_h2, row_bytes);
                                    }
                                }
                            }
                        }

                        /* Recompute tts_pad/bos/eos from new text_embedding + text_projection */
                        if (loaded > 0 && ctx->cached_tts_pad_embed) {
                            extern void embed_one_text_token_compute(qwen_tts_ctx_t *ctx, int tid, float *out);
                            embed_one_text_token_compute(ctx, 151671, ctx->cached_tts_pad_embed);
                            embed_one_text_token_compute(ctx, 151672, ctx->cached_tts_bos_embed);
                            embed_one_text_token_compute(ctx, 151673, ctx->cached_tts_eos_embed);
                        }

                        if (!silent && loaded > 0)
                            fprintf(stderr, "  Loaded %d/%u source talker tensors (%.1f MB) — full weight override\n",
                                    loaded, n_tensors, wfull_bytes / 1024.0f / 1024.0f);
                    }
                }
                fclose(vf);
                if (!silent) {
                    fprintf(stderr, "Voice clone: loaded .qvoice v%u from %s", version, load_voice);
                    if (ctx->cached_ref_n_frames > 0)
                        fprintf(stderr, " (%d ICL frames)", ctx->cached_ref_n_frames);
                    fprintf(stderr, "\n");
                    if (has_meta) {
                        fprintf(stderr, "  Voice: %s | Language: %s | Model: %s | Ref: %.0fs\n",
                                meta_voice_name[0] ? meta_voice_name : "(unnamed)",
                                meta_lang_name[0] ? meta_lang_name : "auto",
                                meta_model_size >= 2048 ? "1.7B" : (meta_model_size > 0 ? "0.6B" : "unknown"),
                                meta_ref_dur);
                    }
                }
            } else {
                /* Load legacy voice file: raw speaker embedding only */
                FILE *vf = fopen(load_voice, "rb");
                if (!vf) {
                    fprintf(stderr, "Error: cannot open voice file %s\n", load_voice);
                    qwen_tts_unload(ctx);
                    return 1;
                }
                size_t n = fread(ctx->speaker_embedding, sizeof(float), enc_dim, vf);
                fclose(vf);
                if ((int)n != enc_dim) {
                    fprintf(stderr, "Error: voice file has %zu floats, expected %d\n", n, enc_dim);
                    qwen_tts_unload(ctx);
                    return 1;
                }
                if (!silent) {
                    fprintf(stderr, "Voice clone: loaded speaker embedding from %s (%d floats)\n", load_voice, enc_dim);
                    /* Debug: print embedding stats */
                    float norm = 0;
                    for (int i = 0; i < enc_dim; i++) norm += ctx->speaker_embedding[i] * ctx->speaker_embedding[i];
                    norm = sqrtf(norm);
                    fprintf(stderr, "  embedding norm=%.4f, first5=[%.4f,%.4f,%.4f,%.4f,%.4f]\n",
                            norm, ctx->speaker_embedding[0], ctx->speaker_embedding[1],
                            ctx->speaker_embedding[2], ctx->speaker_embedding[3], ctx->speaker_embedding[4]);
                }
            }
        } else {
            /* Extract speaker embedding from reference audio */
            if (qwen_extract_speaker_embedding(ctx, ref_audio, ctx->speaker_embedding) != 0) {
                fprintf(stderr, "Error: failed to extract speaker embedding from %s\n", ref_audio);
                qwen_tts_unload(ctx);
                return 1;
            }
            if (!silent)
                fprintf(stderr, "Voice clone: extracted speaker embedding from %s\n", ref_audio);
        }

        /* If ICL mode (not xvector_only), load the speech encoder for ref audio encoding */
        if (!ctx->xvector_only && ref_audio) {
            if (qwen_speech_encoder_load(ctx) != 0) {
                fprintf(stderr, "Warning: failed to load speech encoder, falling back to x-vector only\n");
                ctx->xvector_only = 1;
            }
        }

        /* For .qvoice save: need to encode ref audio now to get ref_codes */
        if (save_is_qvoice && ref_audio && ref_text_str && !ctx->cached_ref_codes) {
            /* Ensure speech encoder is loaded */
            if (!ctx->xvector_only) {
                float *ref_audio_samples = NULL;
                int ref_n_samples = 0, ref_sr = 0;
                if (qwen_read_wav(ref_audio, &ref_audio_samples, &ref_n_samples, &ref_sr) == 0) {
                    int *codes = NULL;
                    int n_frames = 0;
                    if (qwen_speech_encoder_encode(ctx, ref_audio_samples, ref_n_samples,
                                                    &codes, &n_frames) == 0) {
                        ctx->cached_ref_codes = codes;
                        ctx->cached_ref_n_frames = n_frames;
                        if (!silent)
                            fprintf(stderr, "Encoded ref audio: %d ICL frames\n", n_frames);
                    }
                    free(ref_audio_samples);
                }
            }
        }

        /* Save voice file */
        if (save_voice) {
            if (save_is_qvoice) {
                /* Save .qvoice v3: enc_dim + embedding + ref_text + ref_codes + metadata */
                FILE *vf = fopen(save_voice, "wb");
                if (!vf) {
                    fprintf(stderr, "Error: cannot write voice file %s\n", save_voice);
                } else {
                    fwrite("QVCE", 1, 4, vf);
                    uint32_t version = 3;
                    fwrite(&version, sizeof(uint32_t), 1, vf);
                    uint32_t saved_dim = (uint32_t)enc_dim;
                    fwrite(&saved_dim, sizeof(uint32_t), 1, vf);
                    fwrite(ctx->speaker_embedding, sizeof(float), enc_dim, vf);
                    /* ref_text */
                    uint32_t ref_text_len = ref_text_str ? (uint32_t)strlen(ref_text_str) : 0;
                    fwrite(&ref_text_len, sizeof(uint32_t), 1, vf);
                    if (ref_text_len > 0)
                        fwrite(ref_text_str, 1, ref_text_len, vf);
                    /* ref_codes */
                    uint32_t n_ref_frames = ctx->cached_ref_codes ? (uint32_t)ctx->cached_ref_n_frames : 0;
                    fwrite(&n_ref_frames, sizeof(uint32_t), 1, vf);
                    if (n_ref_frames > 0)
                        fwrite(ctx->cached_ref_codes, sizeof(int), (int)n_ref_frames * 16, vf);
                    /* v3 metadata section */
                    fwrite("META", 1, 4, vf);
                    uint32_t lang_id = (uint32_t)(ctx->language_id >= 0 ? ctx->language_id : 0);
                    fwrite(&lang_id, sizeof(uint32_t), 1, vf);
                    /* language name (16 bytes, null-padded) */
                    char lang_name[16] = {0};
                    if (language) strncpy(lang_name, language, 15);
                    fwrite(lang_name, 1, 16, vf);
                    /* source model size: 0=unknown, 600=0.6B, 1700=1.7B */
                    uint32_t model_size = (uint32_t)ctx->config.hidden_size;
                    fwrite(&model_size, sizeof(uint32_t), 1, vf);
                    /* source enc_dim */
                    fwrite(&saved_dim, sizeof(uint32_t), 1, vf);
                    /* ref audio duration in seconds */
                    float ref_dur = max_ref_duration;
                    fwrite(&ref_dur, sizeof(float), 1, vf);
                    /* voice name (64 bytes, null-padded) */
                    char vname[64] = {0};
                    if (voice_name) strncpy(vname, voice_name, 63);
                    fwrite(vname, 1, 64, vf);
                    /* flags: bit 0=xvector_only, bit 1=has_icl, bit 2=is_base_model */
                    uint32_t flags = 0;
                    if (ctx->xvector_only) flags |= 1;
                    if (n_ref_frames > 0) flags |= 2;
                    if (ctx->is_base_model) flags |= 4;
                    fwrite(&flags, sizeof(uint32_t), 1, vf);
                    /* TPAD section: save tts_pad/bos/eos embeddings for cross-model fidelity.
                     * When a Base voice is loaded into CustomVoice, these embeddings override
                     * the target model's own, eliminating per-frame drift from micro-differences
                     * in text_projection weights. */
                    if (ctx->cached_tts_pad_embed) {
                        int h = ctx->config.hidden_size;
                        fwrite("TPAD", 1, 4, vf);
                        uint32_t hidden = (uint32_t)h;
                        fwrite(&hidden, sizeof(uint32_t), 1, vf);
                        fwrite(ctx->cached_tts_pad_embed, sizeof(float), h, vf);
                        fwrite(ctx->cached_tts_bos_embed, sizeof(float), h, vf);
                        fwrite(ctx->cached_tts_eos_embed, sizeof(float), h, vf);
                    }
                    /* WOVR section: source model weights for cross-model fidelity.
                     * Stores text_projection + codec_embedding as BF16 so any target
                     * model can override its own weights with the source model's.
                     * This eliminates per-frame drift from weight micro-differences
                     * without requiring the source Base model to be present. */
                    if (ctx->text_proj_fc1_bf16 && ctx->codec_embedding_bf16) {
                        int h = ctx->config.hidden_size;
                        int th = ctx->config.text_hidden_size;
                        int cv = 2048;  /* codebook entries only, not speaker presets */
                        fwrite("WOVR", 1, 4, vf);
                        uint32_t wh = (uint32_t)h, wth = (uint32_t)th, wcv = (uint32_t)cv;
                        fwrite(&wh, sizeof(uint32_t), 1, vf);
                        fwrite(&wth, sizeof(uint32_t), 1, vf);
                        fwrite(&wcv, sizeof(uint32_t), 1, vf);
                        /* text_proj fc1: [th × th] BF16 + [th] F32 bias */
                        fwrite(ctx->text_proj_fc1_bf16, sizeof(uint16_t), (size_t)th * th, vf);
                        if (ctx->text_proj_fc1_bias)
                            fwrite(ctx->text_proj_fc1_bias, sizeof(float), th, vf);
                        /* text_proj fc2: [h × th] BF16 + [h] F32 bias */
                        fwrite(ctx->text_proj_fc2_bf16, sizeof(uint16_t), (size_t)h * th, vf);
                        if (ctx->text_proj_fc2_bias)
                            fwrite(ctx->text_proj_fc2_bias, sizeof(float), h, vf);
                        /* codec_embedding codebook entries: [cv × h] BF16 */
                        fwrite(ctx->codec_embedding_bf16, sizeof(uint16_t), (size_t)cv * h, vf);
                        if (!silent) {
                            int64_t wovr_bytes = (int64_t)th*th*2 + th*4 + (int64_t)h*th*2 + h*4 + (int64_t)cv*h*2;
                            fprintf(stderr, "  Saved source model weights (%.1f MB) for cross-model fidelity\n",
                                    wovr_bytes / 1024.0f / 1024.0f);
                        }
                    }
                    /* WFULL section: dump ALL talker weights from source model.
                     * This enables perfect cross-model voice fidelity by replacing
                     * the target model's entire talker with the source model's weights.
                     * Size: ~840 MB for 0.6B, ~3.3 GB for 1.7B. */
                    /* WFULL/WDELTA: write talker weights for cross-model fidelity.
                     * Only written when --target-cv is specified (WDELTA mode).
                     * Without --target-cv, the .qvoice contains only TPAD+WOVR (~16MB). */
                    if (target_cv_dir) {
                        int h = ctx->config.hidden_size;
                        int th = ctx->config.text_hidden_size;
                        int nl_layers = ctx->config.num_layers;
                        int q_dim = ctx->config.num_heads * ctx->config.head_dim;
                        int kv_dim = ctx->config.num_kv_heads * ctx->config.head_dim;
                        int inter = ctx->config.intermediate_size;
                        int vocab = 151936; /* text vocab size */
                        int codec_vocab = ctx->config.codec_vocab_size;
                        int head_dim = ctx->config.head_dim;

                        /* Open target CV safetensors for WDELTA if specified */
                        FILE *cv_sf = NULL;
                        char *cv_hdr_json = NULL;
                        size_t cv_data_off = 0;
                        int use_wdelta = 0;
                        if (target_cv_dir) {
                            char cv_sf_path[512];
                            snprintf(cv_sf_path, sizeof(cv_sf_path), "%s/model.safetensors", target_cv_dir);
                            cv_sf = fopen(cv_sf_path, "rb");
                            if (cv_sf) {
                                uint64_t cv_hs;
                                fread(&cv_hs, 8, 1, cv_sf);
                                cv_hdr_json = (char *)malloc(cv_hs + 1);
                                fread(cv_hdr_json, 1, cv_hs, cv_sf);
                                cv_hdr_json[cv_hs] = '\0';
                                cv_data_off = 8 + cv_hs;
                                use_wdelta = 1;
                                if (!silent)
                                    fprintf(stderr, "  Computing deltas vs %s for WDELTA encoding\n", target_cv_dir);
                            }
                        }

                        /* Helper: find tensor offset in CV safetensors JSON header */
                        /* Returns data offset+start in cv_sf, sets *out_size. Returns -1 if not found. */
                        /* Helper macros for WFULL/WDELTA tensor writing */
                        #define WRITE_TENSOR_BF16(tname_str, ptr, nbytes) \
                            write_tensor_impl(vf, cv_sf, cv_hdr_json, cv_data_off, \
                                              tname_str, ptr, nbytes, use_wdelta, 1, \
                                              &wfull_bytes, &wfull_written)
                        #define WRITE_TENSOR_F32(tname_str, ptr, nbytes) \
                            write_tensor_impl(vf, cv_sf, cv_hdr_json, cv_data_off, \
                                              tname_str, ptr, nbytes, use_wdelta, 0, \
                                              &wfull_bytes, &wfull_written)
                        /* Legacy macro for compatibility — BF16 by default */
                        #define WRITE_TENSOR(tname_str, ptr, nbytes) \
                            WRITE_TENSOR_BF16(tname_str, ptr, nbytes)

                        fwrite(use_wdelta ? "WDLT" : "WFULL", 1, use_wdelta ? 4 : 5, vf);
                        /* For WDELTA: store target model hidden_size for validation */
                        if (use_wdelta) {
                            uint32_t target_h = (uint32_t)ctx->config.hidden_size;
                            fwrite(&target_h, sizeof(uint32_t), 1, vf);
                        }
                        int cp_nl = ctx->config.cp_num_layers;
                        int cp_h = ctx->config.cp_hidden_size;
                        int cp_inter = ctx->config.cp_intermediate_size;
                        int cp_q_dim = ctx->config.cp_num_heads * ctx->config.head_dim;
                        int cp_kv_dim = ctx->config.cp_num_kv_heads * ctx->config.head_dim;
                        /* Count: 8 global + 11 per talker layer + CP tensors + mtp_proj */
                        uint32_t n_tensors = 8 + nl_layers * 11 + 1 + 15 + 15 + cp_nl * 11;
                        if (ctx->cp_mtp_proj_bf16) n_tensors += 2;
                        fwrite(&n_tensors, sizeof(uint32_t), 1, vf);
                        int64_t wfull_bytes = 0;
                        int wfull_written = 0;

                        /* Global tensors */
                        WRITE_TENSOR("talker.model.text_embedding.weight",
                                     ctx->tok_embeddings_bf16, (size_t)vocab * th * 2);
                        WRITE_TENSOR("talker.text_projection.linear_fc1.weight",
                                     ctx->text_proj_fc1_bf16, (size_t)th * th * 2);
                        WRITE_TENSOR_F32("talker.text_projection.linear_fc1.bias",
                                     ctx->text_proj_fc1_bias, (size_t)th * 4);
                        WRITE_TENSOR("talker.text_projection.linear_fc2.weight",
                                     ctx->text_proj_fc2_bf16, (size_t)h * th * 2);
                        WRITE_TENSOR_F32("talker.text_projection.linear_fc2.bias",
                                     ctx->text_proj_fc2_bias, (size_t)h * 4);
                        WRITE_TENSOR("talker.model.codec_embedding.weight",
                                     ctx->codec_embedding_bf16, (size_t)codec_vocab * h * 2);
                        WRITE_TENSOR("talker.codec_head.weight",
                                     ctx->codec_head_bf16, (size_t)codec_vocab * h * 2);
                        WRITE_TENSOR_F32("talker.model.norm.weight",
                                     ctx->talker_norm, (size_t)h * 4);

                        /* Per-layer tensors */
                        for (int li = 0; li < nl_layers; li++) {
                            qwen_talker_layer_t *l = &ctx->layers[li];
                            char tn[256];
                            #define LT(field, suffix, sz) do { \
                                snprintf(tn, sizeof(tn), "talker.model.layers.%d.%s", li, suffix); \
                                WRITE_TENSOR(tn, l->field, sz); \
                            } while(0)
                            LT(wq_bf16, "self_attn.q_proj.weight", (size_t)q_dim * h * 2);
                            LT(wk_bf16, "self_attn.k_proj.weight", (size_t)kv_dim * h * 2);
                            LT(wv_bf16, "self_attn.v_proj.weight", (size_t)kv_dim * h * 2);
                            LT(wo_bf16, "self_attn.o_proj.weight", (size_t)h * q_dim * 2);
                            #define LTF(field, suffix, sz) do { \
                                snprintf(tn, sizeof(tn), "talker.model.layers.%d.%s", li, suffix); \
                                WRITE_TENSOR_F32(tn, l->field, sz); \
                            } while(0)
                            LTF(q_norm, "self_attn.q_norm.weight", (size_t)head_dim * 4);
                            LTF(k_norm, "self_attn.k_norm.weight", (size_t)head_dim * 4);
                            LTF(input_norm, "input_layernorm.weight", (size_t)h * 4);
                            LTF(post_attn_norm, "post_attention_layernorm.weight", (size_t)h * 4);
                            #undef LTF
                            LT(gate_bf16, "mlp.gate_proj.weight", (size_t)inter * h * 2);
                            LT(up_bf16, "mlp.up_proj.weight", (size_t)inter * h * 2);
                            LT(down_bf16, "mlp.down_proj.weight", (size_t)h * inter * 2);
                            #undef LT
                        }
                        /* Code Predictor tensors */
                        WRITE_TENSOR_F32("talker.code_predictor.model.norm.weight",
                                     ctx->cp_norm, (size_t)cp_h * 4);
                        for (int ci = 0; ci < 15; ci++) {
                            char tn2[256];
                            snprintf(tn2, sizeof(tn2), "talker.code_predictor.model.codec_embedding.%d.weight", ci);
                            if (ctx->cp_codec_emb_bf16[ci])
                                WRITE_TENSOR(tn2, ctx->cp_codec_emb_bf16[ci],
                                             (size_t)ctx->config.codebook_size * ctx->cp_emb_dim * 2);
                            snprintf(tn2, sizeof(tn2), "talker.code_predictor.lm_head.%d.weight", ci);
                            if (ctx->cp_lm_head_bf16[ci])
                                WRITE_TENSOR(tn2, ctx->cp_lm_head_bf16[ci],
                                             (size_t)ctx->config.codebook_size * cp_h * 2);
                        }
                        for (int li = 0; li < cp_nl; li++) {
                            qwen_cp_layer_t *cl = &ctx->cp_layers[li];
                            char tn2[256];
                            #define CLT(field, suffix, sz) do { \
                                snprintf(tn2, sizeof(tn2), "talker.code_predictor.model.layers.%d.%s", li, suffix); \
                                WRITE_TENSOR(tn2, cl->field, sz); \
                            } while(0)
                            CLT(wq_bf16, "self_attn.q_proj.weight", (size_t)cp_q_dim * cp_h * 2);
                            CLT(wk_bf16, "self_attn.k_proj.weight", (size_t)cp_kv_dim * cp_h * 2);
                            CLT(wv_bf16, "self_attn.v_proj.weight", (size_t)cp_kv_dim * cp_h * 2);
                            CLT(wo_bf16, "self_attn.o_proj.weight", (size_t)cp_h * cp_q_dim * 2);
                            #define CLTF(field, suffix, sz) do { \
                                snprintf(tn2, sizeof(tn2), "talker.code_predictor.model.layers.%d.%s", li, suffix); \
                                WRITE_TENSOR_F32(tn2, cl->field, sz); \
                            } while(0)
                            CLTF(q_norm, "self_attn.q_norm.weight", (size_t)head_dim * 4);
                            CLTF(k_norm, "self_attn.k_norm.weight", (size_t)head_dim * 4);
                            CLTF(input_norm, "input_layernorm.weight", (size_t)cp_h * 4);
                            CLTF(post_attn_norm, "post_attention_layernorm.weight", (size_t)cp_h * 4);
                            #undef CLTF
                            CLT(gate_bf16, "mlp.gate_proj.weight", (size_t)cp_inter * cp_h * 2);
                            CLT(up_bf16, "mlp.up_proj.weight", (size_t)cp_inter * cp_h * 2);
                            CLT(down_bf16, "mlp.down_proj.weight", (size_t)cp_h * cp_inter * 2);
                            #undef CLT
                        }
                        /* MTP projection (1.7B only: 2048→1024) */
                        if (ctx->cp_mtp_proj_bf16) {
                            WRITE_TENSOR("talker.code_predictor.small_to_mtp_projection.weight",
                                         ctx->cp_mtp_proj_bf16, (size_t)cp_h * h * 2);
                            if (ctx->cp_mtp_proj_bias)
                                WRITE_TENSOR_F32("talker.code_predictor.small_to_mtp_projection.bias",
                                             ctx->cp_mtp_proj_bias, (size_t)cp_h * 4);
                        }

                        #undef WRITE_TENSOR

                        if (cv_sf) fclose(cv_sf);
                        free(cv_hdr_json);
                        #undef WRITE_TENSOR

                        if (!silent)
                            fprintf(stderr, "  Saved %d tensors (%.1f MB%s) for full cross-model fidelity\n",
                                    wfull_written, wfull_bytes / 1024.0f / 1024.0f,
                                    use_wdelta ? " WDELTA compressed" : " WFULL raw");
                    }
                    fclose(vf);
                    if (!silent)
                        fprintf(stderr, "Saved .qvoice v3 to %s (embedding + %u ICL frames + metadata + weights)\n",
                                save_voice, n_ref_frames);
                    if (!silent && language)
                        fprintf(stderr, "  Language: %s, Voice: %s, Model: %s\n",
                                language, voice_name ? voice_name : "(unnamed)",
                                ctx->config.hidden_size >= 2048 ? "1.7B" : "0.6B");
                }
            } else {
                /* Save legacy format: raw speaker embedding only */
                FILE *vf = fopen(save_voice, "wb");
                if (!vf) {
                    fprintf(stderr, "Error: cannot write voice file %s\n", save_voice);
                } else {
                    fwrite(ctx->speaker_embedding, sizeof(float), enc_dim, vf);
                    fclose(vf);
                    if (!silent)
                        fprintf(stderr, "Saved speaker embedding to %s (%d floats)\n", save_voice, enc_dim);
                }
            }
        }

        if (!silent) {
            if (ctx->xvector_only && !ctx->cached_ref_codes)
                fprintf(stderr, "Mode: x-vector only (no reference transcription)\n");
            else if (ctx->cached_ref_codes)
                fprintf(stderr, "Mode: ICL with %d cached ref frames\n", ctx->cached_ref_n_frames);
            else
                fprintf(stderr, "Mode: ICL with ref text: \"%s\"\n", ref_text_str);
        }
    }

    if (instruct) {
        if (ctx->config.hidden_size < 2048) {
            fprintf(stderr, "Warning: --instruct is only supported on 1.7B model (ignored)\n");
        } else if (ctx->voice_clone && ctx->is_base_model) {
            fprintf(stderr, "Warning: --instruct with voice cloning on a Base model is not officially supported.\n");
            fprintf(stderr, "  For best results, extract the voice with the Base model and use it with CustomVoice:\n");
            fprintf(stderr, "    ./qwen_tts -d qwen3-tts-1.7b-base --ref-audio ref.wav --save-voice voice.bin\n");
            fprintf(stderr, "    ./qwen_tts -d qwen3-tts-1.7b --load-voice voice.bin --instruct \"...\" --text \"...\"\n");
            ctx->instruct = strdup(instruct);
        } else {
            ctx->instruct = strdup(instruct);
        }
    }

    /* Create voice only: save and exit without generating */
    if (create_voice_only) {
        if (!save_voice) {
            fprintf(stderr, "Error: --save-voice is required when no --text is provided\n");
            qwen_tts_unload(ctx);
            return 1;
        }
        if (!ref_audio) {
            fprintf(stderr, "Error: --ref-audio is required to create a voice profile\n");
            qwen_tts_unload(ctx);
            return 1;
        }
        /* Voice was already saved above in the voice clone setup block */
        if (!silent)
            fprintf(stderr, "Voice profile created. Use --load-voice to generate speech.\n");
        qwen_tts_unload(ctx);
        return 0;
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
