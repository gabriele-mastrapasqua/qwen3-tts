# Voice Clone Reference Samples — Attribution

All clips are 30-second excerpts starting at the 30-second mark of the source recording
(skipping the LibriVox intro), downsampled to 24 kHz mono PCM WAV.

All source recordings are from **LibriVox** (public domain recordings of public domain
texts, released under Public Domain Mark 1.0 / CC0).

| File | Language | Source | Reader | Book / Author | License |
|------|----------|--------|--------|---------------|---------|
| `it_galatea_fasol.wav` | Italian | [archive.org/details/galatea_0908_librivox](https://archive.org/details/galatea_0908_librivox) | Riccardo Fasol (solo) | *Galatea* — Anton Giulio Barrili | PD (LibriVox) |
| `en_ohenry_chenevert.wav` | English | [archive.org/details/5belovedstories_ohenry_pc_librivox](https://archive.org/details/5belovedstories_ohenry_pc_librivox) | Phil Chenevert (solo) | *Five Beloved Stories* (The Gifts of the Magi) — O. Henry | CC0 / PD |
| `es_quijote_lu.wav` | Spanish | [archive.org/details/donquijote_2507_librivox](https://archive.org/details/donquijote_2507_librivox) | Lu (solo) | *El ingenioso hidalgo Don Quijote de la Mancha* — Cervantes | PD (LibriVox) |
| `fr_hugo_bidou.wav` | French | [archive.org/details/dernierjour_2203_librivox](https://archive.org/details/dernierjour_2203_librivox) | Bidou (solo) | *Le dernier jour d'un condamné* — Victor Hugo | PD (LibriVox) |

The original MP3 sources (`*.mp3`) are kept alongside for provenance. Regenerate the
trimmed WAVs with:

```bash
ffmpeg -y -ss 30 -t 30 -i it_galatea_fasol.mp3 -ac 1 -ar 24000 -c:a pcm_s16le it_galatea_fasol.wav
```
