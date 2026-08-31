#!/usr/bin/env python3
import argparse, csv, glob, json, os
from collections import Counter
import librosa, soundfile as sf

EMOTION_INSTRUCT = {
    "neutral":  "",
    "joy":      "Speak happily, bright and warm, smiling through the words.",
    "happy":    "Speak happily, bright and warm, smiling through the words.",
    "anger":    "Speak with hot, furious anger, sharp and forceful.",
    "sadness":  "Speak with a sad, sorrowful, downcast tone, voice low and heavy.",
    "sad":      "Speak with a sad, sorrowful, downcast tone, voice low and heavy.",
    "fear":     "Speak with fear, tense and trembling, your voice wary.",
    "surprise": "Speak with surprise, startled and taken aback, held through the sentence.",
    "disgust":  "Speak with physical disgust, repulsed and recoiling.",
    "boredom":  "Speak in a bored, flat, disinterested monotone, low energy.",
}

EMOVO_SENT = {
    "b1": "Gli operai si alzano presto.", "b2": "I vigili sono muniti di pistola.",
    "b3": "La cascata fa molto rumore.",
    "l1": "L'autunno prossimo Tony partira' per la Spagna nella prima meta' di ottobre.",
    "l2": "Ora prendo la felpa di la' ed esco per fare una passeggiata.",
    "l3": "Un attimo dopo s'e' incamminato ed e' inciampato.",
    "l4": "Vorrei il numero telefonico del Signor Piatti.",
    "n1": "La casa forte vuole col pane.", "n2": "La forza trova il passo e l'aglio rosso.",
    "n3": "Il gatto sta scorrendo nella pera.", "n4": "Insalata pastasciutta coscia d'agnello limoncello.",
    "n5": "Uno quarantatre' dieci mille cinquantasette venti.",
    "d1": "Sabato sera cosa fara'?", "d2": "Porti con te quella cosa?",
}
EMOVO_EMO = {"neu": "neutral", "gio": "joy", "rab": "anger", "tri": "sadness",
             "pau": "fear", "sor": "surprise", "dis": "disgust"}

EMODB_SENT = {
    "a01": "Der Lappen liegt auf dem Eisschrank.",
    "a02": "Das will sie am Mittwoch abgeben.",
    "a04": "Heute abend koennte ich es ihm sagen.",
    "a05": "Das schwarze Stueck Papier befindet sich da oben neben dem Holzstueck.",
    "a07": "In sieben Stunden wird es soweit sein.",
    "b01": "Was sind denn das fuer Tueten, die da unter dem Tisch stehen.",
    "b02": "Sie haben es gerade hochgetragen und jetzt gehen sie wieder runter.",
    "b03": "An den Wochenenden bin ich jetzt immer nach Hause gefahren und habe Agnes besucht.",
    "b09": "Ich will das eben wegbringen und dann mit Karl was trinken gehen.",
    "b10": "Die wird auf dem Platz sein, wo wir sie immer hinlegen.",
}
EMODB_EMO = {"W": "anger", "L": "boredom", "E": "disgust", "A": "fear",
             "F": "joy", "T": "sadness", "N": "neutral"}

MESD_EMO = {"anger": "anger", "disgust": "disgust", "fear": "fear",
            "happiness": "happy", "neutral": "neutral", "sadness": "sadness"}

CAFE_EMO = {"C": "anger", "D": "disgust", "J": "happy", "N": "neutral",
            "P": "fear", "S": "surprise", "T": "sadness"}
CAFE_SENT = {
    "1": "Un cheval fou dans mon jardin.",
    "2": "Deux ânes aigris au pelage brun.",
    "3": "Trois cygnes aveugles au bord du lac.",
    "4": "Quatre vieilles truies éléphantesques.",
    "5": "Cinq pumas fiers et passionnés.",
    "6": "Six ours aimants domestiqués.",
}

def manifest_rows(args):
    """Yield (src_audio_path, text, emotion_label). Customize for your dataset."""
    if args.emovo:
        for wav in sorted(glob.glob(f"{args.emovo}/*/*.wav")):
            emo, _actor, sent = os.path.basename(wav)[:-4].split("-")[:3]
            if emo in EMOVO_EMO and sent in EMOVO_SENT:
                yield wav, EMOVO_SENT[sent], EMOVO_EMO[emo]
    elif args.emodb:
        for wav in sorted(glob.glob(f"{args.emodb}/**/*.wav", recursive=True)):
            base = os.path.basename(wav)[:-4]
            if len(base) < 6:
                continue
            code, emo = base[2:5], base[5]
            if code in EMODB_SENT and emo in EMODB_EMO:
                yield wav, EMODB_SENT[code], EMODB_EMO[emo]
    elif args.mesd:
        for wav in sorted(glob.glob(f"{args.mesd}/**/*.wav", recursive=True)):
            parts = os.path.basename(wav)[:-4].split("_")
            if len(parts) < 4:
                continue
            emo = parts[0].lower()
            rest = parts[3:]
            if rest and rest[0] in ("L1", "L2"):
                rest = rest[1:]
            word = " ".join(rest).strip().lower()
            if emo in MESD_EMO and word:
                yield wav, word, MESD_EMO[emo]
    elif args.cafe:
        for wav in sorted(glob.glob(f"{args.cafe}/**/*.wav", recursive=True)):
            parts = os.path.basename(wav)[:-4].split("-")
            if len(parts) != 4:
                continue
            emo, sent = parts[1], parts[3]
            if emo in CAFE_EMO and sent in CAFE_SENT:
                yield wav, CAFE_SENT[sent], CAFE_EMO[emo]
    else:
        with open(args.manifest, newline="") as f:
            dialect = csv.Sniffer().sniff(f.read(2048)); f.seek(0)
            for r in csv.DictReader(f, dialect=dialect):
                yield r["audio"], r["text"], r["emotion"].strip().lower()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", help="CSV/TSV with columns: audio,text,emotion")
    ap.add_argument("--emovo", help="EMOVO root dir (built-in Italian example)")
    ap.add_argument("--emodb", help="original EmoDB root dir (built-in German example)")
    ap.add_argument("--mesd", help="MESD root dir (built-in Mexican-Spanish example)")
    ap.add_argument("--cafe", help="CaFE root dir (built-in Canadian-French example)")
    ap.add_argument("--out_dir", default="data")
    args = ap.parse_args()
    if not (args.manifest or args.emovo or args.emodb or args.mesd or args.cafe):
        ap.error("provide --manifest, --emovo, --emodb, --mesd, or --cafe")

    wav_dir = os.path.join(args.out_dir, "wav24k"); os.makedirs(wav_dir, exist_ok=True)
    out_jsonl = os.path.join(args.out_dir, "train_raw.jsonl")
    rows, skipped = [], 0
    for src, text, emo in manifest_rows(args):
        if emo not in EMOTION_INSTRUCT:
            skipped += 1; continue
        out = os.path.join(wav_dir, os.path.basename(src))
        if not os.path.exists(out):
            y, _ = librosa.load(src, sr=24000, mono=True)
            sf.write(out, y, 24000, subtype="PCM_16")
        rows.append({"audio": out, "text": text, "ref_audio": out,
                     "instruct": EMOTION_INSTRUCT[emo], "emotion": emo})

    with open(out_jsonl, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows (skipped {skipped} unknown-emotion) -> {out_jsonl}")
    print("emotions:", dict(Counter(r["emotion"] for r in rows)))
    print("NEXT: run the upstream prepare_data.py on this jsonl to add audio_codes.")

if __name__ == "__main__":
    main()
