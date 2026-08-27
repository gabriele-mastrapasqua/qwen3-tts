#!/usr/bin/env python3
"""load_test.py — banco di carico per il server TTS in streaming, con il TTFA come
METRICA DI PRIMA CLASSE.

L'obiettivo di prodotto, testuale (PLAN 0.nonies):

    servire 2-4 richieste parallele con TTFA <= 500 ms per utente,
    senza degradare e senza picchi.

RTF e throughput sono SECONDARI. Un p99 che esplode e' un incidente di qualita' in
un call center anche quando la media e' ottima — quindi qui la distribuzione conta
piu' della media, e i picchi vanno ATTRIBUITI, non solo contati.

── PERCHE' GLI ARRIVI SCAGLIONATI (la ragione per cui questo file e' cambiato) ──

La modalita' storica (`--arrival all-at-once`) spara tutte le richieste insieme
all'inizio del livello. E' il CASO PEGGIORE SINCRONIZZATO, ed e' utile: resta il
default e resta confrontabile con tutte le misure fatte finora. Ma non e' il
traffico vero, e soprattutto NON PUO' DISTINGUERE due cose che vogliamo separare:

    TTFA alto perche' la macchina e' satura
    TTFA alto perche' la richiesta e' arrivata mentre un'altra stava finendo

Quella distinzione e' esattamente cio' che dobbiamo ottimizzare (ammissione,
priorita' ai nuovi, prefill spezzato — PLAN S7). Con arrivi tutti a t=0 ogni
richiesta e' contemporaneamente "in coda" e "in contesa": non c'e' nessuna
richiesta che arrivi da sola sulla macchina carica, quindi nessun controllo.

Da qui `--arrival poisson --rate <req/s>` e `--arrival uniform --interval <s>`:
apertura di anello (open loop), gli arrivi NON aspettano che il server si liberi.
Il seme del generatore e' fisso e DICHIARATO (stampato e scritto nel JSON), cosi'
due corse sono confrontabili: senza quello un p95 diverso puo' essere solo un'altra
estrazione di numeri casuali, e ci avremmo attribuito un'ottimizzazione.

── COSA SIGNIFICA "c" CON GLI ARRIVI SCAGLIONATI ────────────────────────────────

In open loop `c` non e' un semaforo. Se non passi `--rate`/`--interval`, il ritmo
degli arrivi viene CALIBRATO sulla macchina:

    intervallo medio fra arrivi = S / c        (S = durata di UNA richiesta da sola)

Per Little (L = lambda * W): se il server reggesse c stream SENZA degradare, in
volo ce ne sarebbero esattamente c. Quindi "c=4" = "carico offerto tale che un
server che scala perfettamente mostrerebbe 4 richieste in volo". Il numero di
richieste davvero in volo viene misurato e stampato accanto: se e' maggiore di c,
il server NON sta reggendo l'offerta — ed e' quello il risultato, non un difetto
della misura. S si misura con una richiesta sonda a macchina scarica (`--service-s`
lo salta se lo sai gia').

⚠️ Con gli arrivi scaglionati **Q (throughput) non e' piu' una misura di capacita'**:
dipende dal carico offerto, e se gli arrivi sono radi la macchina resta ferma fra
uno e l'altro. Q serve confrontato A PARITA' DI MODALITA' DI ARRIVO, mai fra
modalita' diverse. Per la capacita' pura resta `all-at-once`.

── L'ATTRIBUZIONE DEI PICCHI ────────────────────────────────────────────────────

Per ogni richiesta sopra soglia si dice COSA STAVA SUCCEDENDO quando e' arrivata:

  in volo all'arrivo      quante richieste erano gia' aperte (dal client: esatto)
  arrivi nei 200 ms prima quante altre sono partite subito prima (dal client: esatto)
  fini durante l'attesa   quante altre hanno chiuso mentre questa aspettava
                          - dal client (ultimo byte)  -> esatto
                          - dal server ([BATCH] done) -> se passi --server-log

L'ipotesi che vogliamo FALSIFICARE o confermare e' "questo picco coincide con
l'arrivo o la fine di un'altra richiesta". Il controllo che la rende falsificabile
e' il caso `in volo all'arrivo = 0`: se una richiesta arriva su una macchina
SCARICA e paga comunque 2 s di TTFA, la contesa non c'entra — e' il prefill, il
modello, o la partenza a freddo. Senza quella riga si ottimizza lo scheduler per
un problema che non e' dello scheduler.

`--server-log` legge lo stderr del server IN DIRETTA (tail) e mette un timestamp
del client su ogni riga: il motore stampa `[BATCH] done #N (..., in-flight
admitted=K)` senza orologio, quindi il tempo lo mettiamo noi al momento in cui la
riga compare. stderr in C e' senza buffer, quindi lo scarto e' l'attraversamento
del file, non un buffer. E' una correlazione a ~ms, dichiarata come tale: le
ammissioni (`sink_next_job`) NON sono loggate, quindi il contatore `admitted` lo
vediamo solo quando qualcosa finisce — per gli arrivi la sorgente esatta resta il
client.

── DEFINIZIONI (tutte wall-clock, lato client) ──────────────────────────────────
  TTFA      primo byte audio     - richiesta inviata
  total     ultimo byte audio    - richiesta inviata
  audio_s   byte PCM / 2 / 24000 (int16 mono 24 kHz)
  RTF       total / audio_s                   (<1 = piu' veloce del realtime)
  Q         somma(audio_s) / wall             (audio-secondi per secondo)
  degrado   TTFA p95(c) / TTFA p95(c=1)       "servire 4 utenti costa 4,6x al peggiore"
  stabilita TTFA p95 / TTFA p50               se si stacca, c'e' un picco anche con
                                              la mediana buona

Solo stdlib — asyncio + un client HTTP/1.1 scritto a mano: il tempo di ARRIVO dei
chunk e' esattamente cio' che misuriamo, e il buffering di una libreria lo
nasconderebbe.

Uso:
  # prima si accende un server:
  #   ./qwen_tts -d qwen3-tts-0.6b --serve 8900 --int8 --batch-size 4 -j 4
  tests/load_test.py --speaker ryan --concurrency 1,2,4 --requests 8
  tests/load_test.py --concurrency 1,2,3,4 --arrival poisson --requests 6 \
                     --server-log /tmp/tts/mini_bench/int8_server.log
  tests/load_test.py --concurrency 4 --arrival uniform --interval 1.5 --csv /tmp/l.csv

Variabili d'ambiente (per chi ci arriva attraverso un altro script che non gli
passa i flag — es. the mini-bench wrapper, che questo file non modifica):
  QWEN_LT_ARRIVAL  QWEN_LT_RATE  QWEN_LT_INTERVAL  QWEN_LT_ARRIVAL_SEED
  QWEN_LT_TTFA_BUDGET_MS  QWEN_LT_SERVER_LOG  QWEN_LT_SERVICE_S
"""
import argparse, asyncio, csv, json, math, os, random, statistics, sys, time
from urllib.parse import urlparse

SR = 24000          # il server emette PCM grezzo int16 mono 24 kHz su /v1/tts/stream
BYTES_PER_SAMPLE = 2
# The DEFAULT text bank is English, and that is a phase decision rather than a taste one.
#
#   PHASE 1 (now): characterise the ENGINE on open Qwen3-TTS models, on rented servers.
#   Compute, bandwidth and disk are the same whatever the checkpoint is -- a finetune
#   only changes the tuning on top -- so open weights are enough to measure RTF, TTFA,
#   throughput, the concurrency ceiling and RSS, and there is no reason to ship private
#   weights to a machine you rent.
#
#   FASE 2 (dopo): quando sapremo spremere la CPU, tornano il finetune, il the target language e
#   il gate sul language identity — che NON si buttano: `load_texts_the target language.txt` resta qui, con
#   la copertura delle sette function word, e si sceglie con --text-file.
#
# Il default sbagliato non fallisce: fa misurare l'OSS con testi the target language in silenzio.
# Per questo il default e' quello che serve ORA, e l'altro e' esplicito.
DEFAULT_TEXTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "load_texts_en.txt")

# finestra di co-occorrenza per l'attribuzione. 200 ms non e' un numero magico: e'
# l'ordine di grandezza di un passo di ammissione + un frame (80 ms) + il margine di
# schedulazione. Sotto, si perdono coincidenze vere; sopra, tutto "coincide con tutto".
NEIGHBOR_MS = 200.0


# ── banco di testi ──────────────────────────────────────────────────────────
def load_texts(path, only_classes=None):
    """-> [(cls, text)]. Le righe sono '<classe>\\t<testo>'; '#' e vuote ignorate."""
    out = []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.rstrip("\n")
            if not ln.strip() or ln.lstrip().startswith("#"):
                continue
            if "\t" in ln:
                cls, txt = ln.split("\t", 1)
            else:
                cls, txt = "medium", ln
            cls, txt = cls.strip(), txt.strip()
            if not txt:
                continue
            if only_classes and cls not in only_classes:
                continue
            out.append((cls, txt))
    if not out:
        sys.exit(f"{path}: nessun testo usabile (filtro={only_classes})")

    # ── INTERLEAVE per classe, e non e' cosmetica ────────────────────────────────
    # Il banco e' scritto RAGGRUPPATO per classe (prima tutte le short, poi le medium,
    # poi le long) e la scelta a valle e' `texts[idx % len(texts)]`, cioe' round-robin
    # dalla testa. Conseguenza misurata il 2026-08-18: un livello c=4 con REQS=4 emetteva
    # QUATTRO FRASI CORTE quasi identiche, e in un giro FULL (REQS=8) la classe `long`
    # non usciva MAI — l'opposto del disegno "lunghezze miste" dichiarato in cima a
    # tests/load_texts_en.txt.
    #
    # Perche' e' il caso peggiore proprio per quello che vogliamo misurare: con richieste
    # di pari lunghezza tutti gli slot del batch arrivano a EOS nello stesso frame, quindi
    # il batch si riempie e si svuota tutto insieme e non si esercita mai il regime
    # RAGGED — che e' esattamente il regime del batching continuo in produzione, dove
    # slot che finiscono vengono rimpiazzati mentre gli altri proseguono.
    #
    # L'interleave si fa QUI e non riordinando il file, perche' `only_classes` puo'
    # selezionare un sottoinsieme: l'alternanza deve valere su cio' che resta dopo il
    # filtro. Ordine delle classi = prima apparizione nel file (deterministico, nessun
    # random: il banco deve restare riproducibile fra due giri).
    buckets = {}
    for cls, txt in out:
        buckets.setdefault(cls, []).append(txt)
    order = list(buckets.keys())
    mixed = []
    i = 0
    while len(mixed) < len(out):
        for cls in order:
            b = buckets[cls]
            if i < len(b):
                mixed.append((cls, b[i]))
        i += 1
    return mixed


# ── tail dello stderr del server ────────────────────────────────────────────
class ServerLogTail:
    """Legge lo stderr del server MENTRE gira e timestampa le righe al momento in
    cui compaiono. Serve perche' il motore stampa `[BATCH] done #N` senza orologio:
    l'unico modo di sapere QUANDO e' successo, senza toccare il C, e' guardare il
    file in diretta. stderr in C non e' bufferizzato -> la riga compare all'istante
    dell'evento, e il ritardo che aggiungiamo e' il periodo di campionamento (20 ms),
    non un buffer di libreria. Va detto: e' una correlazione a ~decine di ms, non un
    tracing. Basta per rispondere a "un'altra ha finito mentre questa aspettava".

    Quello che il tail NON puo' vedere: le AMMISSIONI. `sink_next_job` incrementa il
    contatore ma non stampa niente, quindi `admitted=K` lo leggiamo solo appiccicato
    a un `done`. Per gli arrivi la sorgente esatta e' il client, che li genera lui.
    """

    def __init__(self, path, poll_s=0.02):
        self.path = path
        self.poll_s = poll_s
        self.events = []          # [(t_rel_ms, riga)]
        self._stop = False
        self._fh = None
        self._buf = b""

    def open_at_end(self):
        """Aperto e posizionato in fondo PRIMA del livello: cosi' non si contano gli
        eventi della scaldata o del livello precedente come se fossero di questo."""
        try:
            self._fh = open(self.path, "rb")
            self._fh.seek(0, os.SEEK_END)
            return True
        except Exception:
            self._fh = None
            return False

    async def run(self, t0):
        if self._fh is None:
            return
        while not self._stop:
            try:
                data = self._fh.read()
            except Exception:
                data = b""
            if data:
                now = (time.perf_counter() - t0) * 1000.0
                self._buf += data
                *lines, self._buf = self._buf.split(b"\n")
                for ln in lines:
                    self.events.append((now, ln.decode("utf-8", "replace").rstrip()))
            await asyncio.sleep(self.poll_s)

    def stop(self):
        self._stop = True
        if self._fh:
            try:
                self._fh.close()
            except Exception:
                pass

    def done_events(self):
        """[(t_ms, admitted_cumulativo)] dalle righe `[BATCH] done #N (..., admitted=K)`."""
        out = []
        for t, ln in self.events:
            if "[BATCH] done" in ln:
                adm = None
                if "admitted=" in ln:
                    tail = ln.split("admitted=", 1)[1]
                    num = ""
                    for ch in tail:
                        if ch.isdigit():
                            num += ch
                        else:
                            break
                    adm = int(num) if num else None
                out.append((t, adm))
        return out


# ── una richiesta ───────────────────────────────────────────────────────────
async def one_request(host, port, path, payload, req_id, cls, out_dir, save_audio,
                      timeout, t0, arrival_sched_ms, inflight, gate):
    """Lancia una richiesta in streaming e cronometra a parte il PRIMO byte del corpo.

    `t0` e' l'origine dei tempi DEL LIVELLO: tutti gli istanti sono relativi a quello,
    perche' l'attribuzione dei picchi ha bisogno di una linea del tempo comune fra
    richieste diverse. Una durata da sola non permette di attribuire niente.
    """
    body = json.dumps(payload).encode()
    head = (
        f"POST {path} HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\n"
        f"Content-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\n"
        f"Connection: close\r\n\r\n"
    ).encode()

    rec = {"request_id": req_id, "class": cls, "text": payload["text"],
           "ttfa_ms": None, "total_ms": None, "audio_s": 0.0, "rtf": None,
           "bytes": 0, "status": None, "error": "",
           # ── linea del tempo, relativa all'inizio del livello ──
           "arrival_sched_ms": arrival_sched_ms,   # quando DOVEVA arrivare
           "arrival_ms": None,                     # quando e' partita davvero
           "first_ms": None,                       # primo byte audio
           "end_ms": None,                         # ultimo byte
           "admit_wait_ms": 0.0}                   # attesa nel client (solo --max-inflight)

    if gate is not None:
        g0 = time.perf_counter()
        await gate.acquire()
        rec["admit_wait_ms"] = (time.perf_counter() - g0) * 1000.0

    t_start = time.perf_counter()
    rec["arrival_ms"] = (t_start - t0) * 1000.0
    inflight.append(req_id)
    reader = writer = None
    try:
        reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout)
        writer.write(head + body)
        await writer.drain()

        # ── header ──────────────────────────────────────────────────────────
        raw = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout)
        status_line = raw.split(b"\r\n", 1)[0].decode("latin1")
        rec["status"] = int(status_line.split()[1]) if len(status_line.split()) > 1 else 0
        chunked = b"transfer-encoding: chunked" in raw.lower()

        # ── corpo; il TTFA si timbra sul primo byte che e' davvero audio ─────
        chunks, first = [], None
        while True:
            if chunked:
                size_line = await asyncio.wait_for(reader.readuntil(b"\r\n"), timeout)
                n = int(size_line.strip().split(b";")[0], 16)
                if n == 0:
                    break
                data = await asyncio.wait_for(reader.readexactly(n), timeout)
                await reader.readexactly(2)          # CRLF finale
            else:
                data = await asyncio.wait_for(reader.read(65536), timeout)
                if not data:
                    break
            if data and first is None:
                first = time.perf_counter()
                rec["ttfa_ms"] = (first - t_start) * 1000.0
                rec["first_ms"] = (first - t0) * 1000.0
            chunks.append(data)

        payload_bytes = b"".join(chunks)
        t_end = time.perf_counter()
        rec["total_ms"] = (t_end - t_start) * 1000.0
        rec["end_ms"] = (t_end - t0) * 1000.0
        rec["bytes"] = len(payload_bytes)
        rec["audio_s"] = len(payload_bytes) / BYTES_PER_SAMPLE / SR
        if rec["audio_s"] > 0:
            rec["rtf"] = (rec["total_ms"] / 1000.0) / rec["audio_s"]
        else:
            rec["error"] = rec["error"] or "corpo vuoto"

        if save_audio and payload_bytes:
            _write_wav(os.path.join(out_dir, f"req{req_id:04d}_{cls}.wav"), payload_bytes)

    except Exception as e:                        # noqa: BLE001 — ogni fallimento e' un dato
        rec["error"] = f"{type(e).__name__}: {e}"
        t_end = time.perf_counter()
        rec["total_ms"] = (t_end - t_start) * 1000.0
        rec["end_ms"] = (t_end - t0) * 1000.0
    finally:
        try:
            inflight.remove(req_id)
        except ValueError:
            pass
        if gate is not None:
            gate.release()
        if writer is not None:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
    return rec


def _write_wav(path, pcm):
    """PCM int16 grezzo -> un vero file RIFF. Il server strema PCM senza header; un
    client che lo scrive dritto in .wav produce un file che nessun player apre."""
    import struct
    n = len(pcm)
    hdr = (b"RIFF" + struct.pack("<I", 36 + n) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, SR, SR * 2, 2, 16)
           + b"data" + struct.pack("<I", n))
    with open(path, "wb") as f:
        f.write(hdr)
        f.write(pcm)


# ── processo degli arrivi ───────────────────────────────────────────────────
def arrival_offsets(mode, n, conc, rate, interval, service_s, seed):
    """Istanti di arrivo (secondi dall'inizio del livello) per n richieste.

    Il seme e' FISSO e viene dichiarato nel report: due corse con lo stesso seme
    hanno la stessa sequenza di arrivi, quindi una differenza di p95 e' la macchina
    e non un'altra estrazione. Senza questo, su 6-8 richieste, il rumore del
    processo di Poisson e' facilmente piu' grande dell'effetto che cerchiamo.
    """
    if mode == "all-at-once":
        return [0.0] * n, None
    if mode == "uniform":
        gap = interval if interval and interval > 0 else (service_s / max(conc, 1))
        return [i * gap for i in range(n)], 1.0 / gap if gap > 0 else None
    if mode == "poisson":
        lam = rate if rate and rate > 0 else (max(conc, 1) / service_s)
        rng = random.Random(seed)
        out, t = [], 0.0
        for _ in range(n):
            out.append(t)
            # inter-arrivi esponenziali = processo di Poisson (memoryless: e' il
            # modello standard di arrivi indipendenti, ed e' anche il caso in cui
            # i "grappoli" capitano da soli, senza doverli costruire a mano).
            t += -math.log(1.0 - rng.random()) / lam
        return out, lam
    raise SystemExit(f"modalita' di arrivo sconosciuta: {mode}")


# ── un livello di concorrenza ───────────────────────────────────────────────
async def run_level(args, host, port, texts, conc, service_s):
    out_dir = os.path.join(args.output_dir, f"c{conc}")
    if args.save_audio:
        os.makedirs(out_dir, exist_ok=True)

    tail = None
    if args.server_log:
        tail = ServerLogTail(args.server_log)
        if not tail.open_at_end():
            print(f"  ⚠️  --server-log {args.server_log}: non apribile, attribuzione solo lato client")
            tail = None

    def payload_for(idx):
        cls, txt = texts[idx % len(texts)]
        return cls, {"text": txt, "speaker": args.speaker, "language": args.language,
                     "temperature": args.temperature, "seed": args.seed + idx}

    inflight = []                     # id delle richieste aperte in questo istante
    gate = asyncio.Semaphore(args.max_inflight) if args.max_inflight > 0 else None
    records = []
    t0 = time.perf_counter()
    tail_task = asyncio.create_task(tail.run(t0)) if tail else None

    if args.arrival == "all-at-once" and args.duration:
        # soak storico: si tengono `conc` richieste in volo finche' scade il tempo.
        # E' closed loop per costruzione, e resta com'era: e' il regime di tenuta.
        stop_at = t0 + args.duration
        sem = asyncio.Semaphore(conc)

        async def worker(idx):
            cls, pl = payload_for(idx)
            async with sem:
                if time.perf_counter() > stop_at:
                    return None
                return await one_request(host, port, args.path, pl, idx, cls, out_dir,
                                         args.save_audio, args.timeout, t0, 0.0, inflight, None)

        pending, idx = set(), 0
        while time.perf_counter() < stop_at or pending:
            while len(pending) < conc and time.perf_counter() < stop_at:
                pending.add(asyncio.create_task(worker(idx))); idx += 1
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            records.extend(r for r in (d.result() for d in done) if r)
        lam = None

    elif args.arrival == "all-at-once":
        # IL CASO PEGGIORE SINCRONIZZATO, e resta il default: tutte insieme a t=0,
        # `conc` in volo, la successiva parte quando una finisce. E' il regime in cui
        # ogni numero storico di questo progetto e' stato preso — cambiarlo di
        # nascosto renderebbe incomparabile tutto il pregresso.
        sem = asyncio.Semaphore(conc)

        async def worker(idx):
            cls, pl = payload_for(idx)
            async with sem:
                return await one_request(host, port, args.path, pl, idx, cls, out_dir,
                                         args.save_audio, args.timeout, t0, 0.0, inflight, None)

        results = await asyncio.gather(*(worker(i) for i in range(args.requests)))
        records = [r for r in results if r]
        lam = None

    else:
        # ── ARRIVI SCAGLIONATI, open loop: nessuno trattiene le richieste. E' l'unico
        # modo di avere richieste che arrivano su una macchina GIA' carica ma non
        # sincronizzate con le altre — cioe' di poter attribuire un picco.
        n = args.requests
        offs, lam = arrival_offsets(args.arrival, n, conc, args.rate, args.interval,
                                    service_s, args.arrival_seed)
        if args.duration:
            offs = [o for o in offs if o <= args.duration] or offs[:1]
        tasks = []
        for i, off in enumerate(offs):
            now = time.perf_counter() - t0
            if off > now:
                await asyncio.sleep(off - now)
            cls, pl = payload_for(i)
            tasks.append(asyncio.create_task(
                one_request(host, port, args.path, pl, i, cls, out_dir, args.save_audio,
                            args.timeout, t0, off * 1000.0, inflight, gate)))
        results = await asyncio.gather(*tasks)
        records = [r for r in results if r]

    wall = time.perf_counter() - t0
    if tail_task:
        tail.stop()
        try:
            await asyncio.wait_for(tail_task, 1.0)
        except Exception:
            tail_task.cancel()
    done_ev = tail.done_events() if tail else []
    attribute(records, done_ev, args.ttfa_budget_ms)
    return records, wall, len(records), lam, done_ev


# ── attribuzione dei picchi ─────────────────────────────────────────────────
def attribute(records, server_done_events, budget_ms):
    """Per ogni richiesta: cosa stava succedendo nel server quando e' arrivata, e
    durante l'attesa del primo byte.

    Le colonne lato client (in volo, arrivi vicini, fini durante l'attesa) sono
    ESATTE: siamo noi a generare gli arrivi e a vedere l'ultimo byte. Quelle dal log
    del server sono una conferma indipendente, timestampata dal tail.

    Il controllo che rende l'ipotesi falsificabile e' `inflight_at_arrival == 0`:
    un picco su macchina scarica NON e' contesa, ed e' l'unico caso in cui si puo'
    escludere lo scheduler senza ricorrere ad altri strumenti.
    """
    ev = [(r["arrival_ms"], r["end_ms"], r["request_id"]) for r in records
          if r["arrival_ms"] is not None]
    for r in records:
        a = r["arrival_ms"]
        f = r["first_ms"] if r["first_ms"] is not None else r["end_ms"]
        if a is None:
            continue
        others = [(s, e, i) for (s, e, i) in ev if i != r["request_id"]]
        r["inflight_at_arrival"] = sum(
            1 for (s, e, _i) in others if s is not None and s <= a and (e is None or e > a))
        r["arrivals_prev_200ms"] = sum(
            1 for (s, _e, _i) in others if s is not None and a - NEIGHBOR_MS <= s < a)
        r["finishes_prev_200ms"] = sum(
            1 for (_s, e, _i) in others if e is not None and a - NEIGHBOR_MS <= e < a)
        if f is None:
            r["arrivals_in_wait"] = r["finishes_in_wait"] = 0
            r["max_inflight_in_wait"] = r["inflight_at_arrival"]
        else:
            r["arrivals_in_wait"] = sum(
                1 for (s, _e, _i) in others if s is not None and a <= s <= f)
            r["finishes_in_wait"] = sum(
                1 for (_s, e, _i) in others if e is not None and a <= e <= f)
            # in volo massimo durante l'attesa: il picco di contesa che ha davvero
            # visto, non quello all'istante dell'arrivo (che puo' essere basso).
            hi = r["inflight_at_arrival"]
            for (s, _e, _i) in others:
                if s is not None and a <= s <= f:
                    hi += 1
            r["max_inflight_in_wait"] = hi
        # conferma indipendente dal server: quante `[BATCH] done` sono state stampate
        # mentre questa richiesta aspettava il primo byte.
        if f is not None:
            r["srv_done_in_wait"] = sum(1 for (t, _k) in server_done_events if a <= t <= f)
            r["srv_done_prev_200ms"] = sum(
                1 for (t, _k) in server_done_events if a - NEIGHBOR_MS <= t < a)
            adm = [k for (t, k) in server_done_events if a <= t <= f and k is not None]
            adm0 = [k for (t, k) in server_done_events if t < a and k is not None]
            # il contatore `admitted` e' cumulativo e lo vediamo solo appiccicato a un
            # `done`: se e' cresciuto nella finestra, qualcun altro E' STATO AMMESSO.
            # Se non ci sono `done` nella finestra non possiamo dire niente -> None,
            # che nel report si legge "n/d", non "zero".
            r["srv_admits_in_wait"] = (max(adm) - max(adm0)) if (adm and adm0) else None
        else:
            r["srv_done_in_wait"] = r["srv_done_prev_200ms"] = 0
            r["srv_admits_in_wait"] = None
        r["spike"] = bool(r["ttfa_ms"] is not None and r["ttfa_ms"] > budget_ms)
        r["attribution"] = _verdict(r)


def _verdict(r):
    """Una riga corta che dice a cosa COINCIDE il picco. Non e' una spiegazione
    causale: e' la coincidenza, che e' l'ipotesi da confermare o falsificare."""
    if not r.get("spike"):
        return ""
    if r.get("inflight_at_arrival", 0) == 0 and r.get("max_inflight_in_wait", 0) <= 1:
        return "macchina SCARICA -> NON e' contesa (prefill/modello/partenza a freddo)"
    bits = []
    if r.get("arrivals_prev_200ms", 0):
        bits.append(f"grappolo in arrivo (+{r['arrivals_prev_200ms']} nei 200 ms prima)")
    if r.get("finishes_in_wait", 0) or r.get("srv_done_in_wait", 0):
        n = max(r.get("finishes_in_wait", 0), r.get("srv_done_in_wait", 0))
        bits.append(f"ha aspettato la fine di un'altra ({n} chiusa/e durante l'attesa)")
    if r.get("srv_admits_in_wait"):
        bits.append(f"altre {r['srv_admits_in_wait']} ammesse mentre aspettava")
    if not bits:
        bits.append(f"contesa continua ({r.get('inflight_at_arrival', 0)} gia' in volo, "
                    f"nessun evento nella finestra)")
    return " · ".join(bits)


# ── statistiche ─────────────────────────────────────────────────────────────
def pct(values, p):
    if not values:
        return float("nan")
    v = sorted(values)
    if len(v) == 1:
        return v[0]
    k = (len(v) - 1) * (p / 100.0)
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def mean_inflight(records, wall):
    """Richieste in volo, mediate nel tempo (integrale delle sovrapposizioni / wall).
    Serve a dire se il carico OFFERTO si e' tradotto nella concorrenza voluta: con
    arrivi scaglionati "c=4" e' un bersaglio, non un fatto."""
    tot = 0.0
    hi = 0
    edges = []
    for r in records:
        if r["arrival_ms"] is None or r["end_ms"] is None:
            continue
        tot += (r["end_ms"] - r["arrival_ms"]) / 1000.0
        edges.append((r["arrival_ms"], 1))
        edges.append((r["end_ms"], -1))
    edges.sort()
    cur = 0
    for _t, d in edges:
        cur += d
        hi = max(hi, cur)
    return (tot / wall) if wall > 0 else 0.0, hi


def summarize(records, wall, conc, budget_ms, arrival, lam, seed, service_s):
    ok = [r for r in records if not r["error"] and r["status"] == 200]
    bad = [r for r in records if r not in ok]
    ttfa = [r["ttfa_ms"] for r in ok if r["ttfa_ms"] is not None]
    rtf = [r["rtf"] for r in ok if r["rtf"] is not None]
    audio = sum(r["audio_s"] for r in ok)
    over = [r for r in ok if r["ttfa_ms"] is not None and r["ttfa_ms"] > budget_ms]
    p50, p95 = pct(ttfa, 50), pct(ttfa, 95)
    mi, hi = mean_inflight(records, wall)
    return {
        "concurrency": conc, "requests": len(records), "ok": len(ok), "errors": len(bad),
        "wall_s": wall, "audio_s": audio, "throughput_Q": (audio / wall) if wall else 0.0,
        # ── TTFA: la metrica di prima classe ──
        "ttfa_p50": p50, "ttfa_p95": p95, "ttfa_p99": pct(ttfa, 99),
        "ttfa_max": max(ttfa) if ttfa else float("nan"),
        "ttfa_mean": statistics.fmean(ttfa) if ttfa else float("nan"),
        "ttfa_budget_ms": budget_ms,
        "ttfa_over_budget": len(over),
        "ttfa_over_budget_pct": (100.0 * len(over) / len(ok)) if ok else float("nan"),
        "ttfa_stability": (p95 / p50) if (p50 and p50 == p50 and p50 > 0) else float("nan"),
        "ttfa_within_budget": bool(ok) and (p95 <= budget_ms),
        # ── secondarie ──
        "rtf_p50": pct(rtf, 50), "rtf_p95": pct(rtf, 95),
        # ── com'e' arrivato il carico (senza questo la riga non e' riproducibile) ──
        "arrival": arrival, "arrival_rate_hz": lam, "arrival_seed": seed,
        "service_s_estimate": service_s,
        "mean_inflight": mi, "max_inflight_observed": hi,
        "spikes": [{k: r[k] for k in ("request_id", "class", "ttfa_ms", "arrival_ms",
                                      "inflight_at_arrival", "arrivals_prev_200ms",
                                      "arrivals_in_wait", "finishes_in_wait",
                                      "srv_done_in_wait", "srv_admits_in_wait",
                                      "attribution")}
                   for r in sorted(over, key=lambda x: -x["ttfa_ms"])],
    }


# ── stampa ──────────────────────────────────────────────────────────────────
def print_table(rows, budget_ms):
    base = next((s for s in rows if s["concurrency"] == 1), None)
    b95 = base["ttfa_p95"] if base else None
    print()
    print("=============== TTFA — LA METRICA DI PRIMA CLASSE")
    hdr = (f"{'conc':>5}{'req':>5}{'err':>4} | {'TTFA p50':>9}{'p95':>8}{'p99':>8}{'max':>8}"
           f" | {'degrado':>8}{'stab':>6}{'>soglia':>9} | {'inflight':>9}{'Q':>7}"
           f"{'RTF p50':>9}{'p95':>7}")
    print(hdr)
    print("-" * len(hdr))
    for s in rows:
        deg = (s["ttfa_p95"] / b95) if (b95 and b95 > 0) else float("nan")
        over = f"{s['ttfa_over_budget']}/{s['ok']}"
        print(f"{s['concurrency']:>5}{s['requests']:>5}{s['errors']:>4} | "
              f"{s['ttfa_p50']:>9.0f}{s['ttfa_p95']:>8.0f}{s['ttfa_p99']:>8.0f}{s['ttfa_max']:>8.0f}"
              f" | {deg:>7.2f}x{s['ttfa_stability']:>6.2f}{over:>9} | "
              f"{s['mean_inflight']:>9.2f}{s['throughput_Q']:>7.2f}"
              f"{s['rtf_p50']:>9.2f}{s['rtf_p95']:>7.2f}")
    print()
    print(f"degrado  = TTFA p95(c) / TTFA p95(c=1). \"servire c utenti costa Nx di latenza al peggiore\"")
    print(f"stab     = p95/p50. Se si stacca da 1, c'e' un PICCO anche quando la mediana e' buona")
    print(f">soglia  = richieste con TTFA > {budget_ms:.0f} ms (il bersaglio di prodotto)")
    print(f"inflight = richieste in volo mediate nel tempo: con arrivi scaglionati dice se il")
    print(f"           carico offerto si e' davvero tradotto nella concorrenza voluta")
    print(f"Q e RTF sono SECONDARI (PLAN 0.nonies). ⚠️ con arrivi scaglionati Q dipende dal")
    print(f"carico offerto, non e' una misura di capacita': confrontalo solo a parita' di --arrival")
    nmin = min((s["ok"] for s in rows), default=0)
    stag = any(s.get("arrival") != "all-at-once" for s in rows)
    if stag and nmin:
        cmax = max(s["concurrency"] for s in rows)
        reach = nmin * cmax / (nmin - 1 + cmax)
        print(f"⚠️ con {nmin} richieste per livello l'inflight medio RAGGIUNGIBILE e'")
        print(f"   N*c/(N-1+c) = {reach:.1f} a c={cmax}, non {cmax}: il livello finisce prima di")
        print(f"   entrare in regime. Va bene per vedere la FORMA del degrado; per misurare")
        print(f"   davvero c utenti in volo serve --requests >= 8-16.")
    if nmin < 100:
        print(f"⚠️ con {nmin} richieste per livello la p99 E' il massimo (serve N>=100 perche' sia")
        print(f"   una p99): leggila come 'la peggiore', non come un percentile.")
    if base and base.get("arrival") != "all-at-once" and base.get("max_inflight_observed", 0) > 1:
        print(f"⚠️ la BASELINE non e' pulita: a c=1 si sono viste fino a {base['max_inflight_observed']}")
        print(f"   richieste in volo insieme — con Poisson i grappoli capitano anche a carico basso.")
        print(f"   Il 'degrado' e' quindi normalizzato su un c=1 che gia' contiene contesa: se serve")
        print(f"   una baseline pulita usa --arrival uniform, che non fa grappoli per costruzione.")


def print_verdict(rows, budget_ms):
    print()
    print(f"=============== VERDETTO — TTFA p95 <= {budget_ms:.0f} ms (PLAN 0.nonies, bersaglio di prodotto)")
    best = 0
    for s in sorted(rows, key=lambda r: r["concurrency"]):
        ok = s["ttfa_within_budget"] and s["errors"] == 0
        if ok:
            best = max(best, s["concurrency"])
        mark = "✅" if ok else "❌"
        why = []
        if s["errors"]:
            why.append(f"{s['errors']} errori")
        if not s["ttfa_within_budget"]:
            why.append(f"p95 {s['ttfa_p95']:.0f} ms > {budget_ms:.0f}")
        if s["ttfa_over_budget"]:
            why.append(f"{s['ttfa_over_budget']}/{s['ok']} richieste sopra soglia (max {s['ttfa_max']:.0f} ms)")
        print(f"  c={s['concurrency']:<3} {mark}  p95 {s['ttfa_p95']:>7.0f} ms"
              + (("   " + " · ".join(why)) if why else "   dentro soglia, nessun picco"))
    print()
    if best:
        print(f"  -> MASSIMA CONCORRENZA CHE REGGE TTFA p95 <= {budget_ms:.0f} ms:  c = {best}")
    else:
        print(f"  -> MASSIMA CONCORRENZA CHE REGGE TTFA p95 <= {budget_ms:.0f} ms:  NESSUNA (nemmeno c=1)")
    print("     E' il numero di prodotto: quante conversazioni parallele questa macchina")
    print("     serve senza che l'utente senta il silenzio. Q e RTF vengono dopo.")


def print_spikes(rows, budget_ms, limit=12):
    spikes = [(s["concurrency"], sp) for s in rows for sp in s["spikes"]]
    print()
    print(f"=============== ATTRIBUZIONE DEI PICCHI (TTFA > {budget_ms:.0f} ms)")
    if not spikes:
        print("  nessun picco: nessuna richiesta sopra soglia a nessun livello.")
        return
    print("  L'ipotesi da confermare o FALSIFICARE: \"il picco coincide con l'arrivo o la fine")
    print("  di un'altra richiesta\". La riga che la falsifica e' 'macchina SCARICA'.")
    print()
    hdr = (f"  {'c':>2}{'req':>5}{'classe':>8}{'arrivo':>9}{'TTFA':>8}{'volo':>6}"
           f"{'arr-200':>8}{'fini':>6}{'srv':>5}  coincidenza")
    print(hdr)
    print("  " + "-" * (len(hdr) + 30))
    for c, sp in sorted(spikes, key=lambda x: -x[1]["ttfa_ms"])[:limit]:
        srv = sp["srv_done_in_wait"]
        print(f"  {c:>2}{sp['request_id']:>5}{sp['class']:>8}{sp['arrival_ms']:>8.0f}m"
              f"{sp['ttfa_ms']:>7.0f}m{sp['inflight_at_arrival']:>6}"
              f"{sp['arrivals_prev_200ms']:>8}{sp['finishes_in_wait']:>6}{srv:>5}  {sp['attribution']}")
    if len(spikes) > limit:
        print(f"  ... e altri {len(spikes) - limit} (tutti nel CSV/JSON)")
    print()
    print("  volo    = richieste gia' aperte quando questa e' arrivata (lato client, esatto)")
    print("  arr-200 = altre arrivate nei 200 ms precedenti (grappolo)")
    print("  fini    = altre CHIUSE mentre questa aspettava il primo byte (lato client)")
    print("  srv     = righe `[BATCH] done` del server nella stessa finestra (--server-log)")


# ── calibrazione del ritmo di arrivo ────────────────────────────────────────
async def calibrate(args, host, port, texts):
    """Una richiesta SOLA su macchina scarica -> S, la durata di servizio di
    riferimento. Serve solo a scegliere il ritmo degli arrivi quando non lo passi:
    intervallo = S/c rende "c" confrontabile fra macchine diverse senza numeri
    cablati (su M1 e su un C3 la stessa `--rate` significherebbe carichi diversi).
    NON entra nelle statistiche: e' fuori dalla misura, per costruzione.

    ⚠️ La sonda usa UN testo, ma il livello ne usa una MISCELA di lunghezze diverse:
    prendere la durata della sonda come S paga un errore proporzionale al rapporto
    fra le lunghezze, e con banco misto e' un fattore 3-5 (misurato: sonda 17 s
    contro media reale ~3 s -> arrivi cinque volte troppo radi, e la concorrenza
    offerta non si realizza mai). Quindi si RISCALA sulla lunghezza media del banco,
    che e' il predittore piu' semplice della durata dell'audio. Resta una stima: il
    ritmo vero viene poi RICALIBRATO sul livello c=1, che e' l'unico senza contesa.
    """
    mid = texts[len(texts) // 2]
    pl = {"text": mid[1], "speaker": args.speaker, "language": args.language,
          "temperature": args.temperature, "seed": args.seed}
    t0 = time.perf_counter()
    r = await one_request(host, port, args.path, pl, -1, mid[0], args.output_dir, False,
                          args.timeout, t0, 0.0, [], None)
    if r["error"] or not r["total_ms"]:
        return None, r
    mean_len = sum(len(t) for _c, t in texts) / len(texts)
    scale = mean_len / max(len(mid[1]), 1)
    return (r["total_ms"] / 1000.0) * scale, r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    env = os.environ.get
    ap.add_argument("--url", default="http://localhost:8900")
    ap.add_argument("--path", default="/v1/tts/stream")
    ap.add_argument("--concurrency", default="1,2,4,8", help="sweep separato da virgole, es. 1,2,4,8")
    ap.add_argument("--requests", type=int, default=16, help="richieste per livello di concorrenza")
    ap.add_argument("--duration", type=float, default=0.0, help="secondi; scavalca --requests (soak)")
    ap.add_argument("--text-file", default=DEFAULT_TEXTS)
    ap.add_argument("--classes", default="", help="filtro sulle classi di testo, es. short,long")
    ap.add_argument("--speaker", default="ryan")
    ap.add_argument("--language", default="english")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--output-dir", default="/tmp/tts/loadtest")
    ap.add_argument("--no-save-audio", dest="save_audio", action="store_false")
    ap.add_argument("--csv", default="", help="scrive qui i record per richiesta")
    ap.add_argument("--json", default="", help="scrive qui il riepilogo per livello")
    # ── arrivi ──
    ap.add_argument("--arrival", default=env("QWEN_LT_ARRIVAL", "all-at-once"),
                    choices=["all-at-once", "poisson", "uniform"],
                    help="all-at-once = caso peggiore sincronizzato (default, storico); "
                         "poisson = inter-arrivi esponenziali; uniform = spaziatura fissa")
    ap.add_argument("--rate", type=float, default=float(env("QWEN_LT_RATE", "0") or 0),
                    help="richieste/s per --arrival poisson (0 = calibrato: c/S)")
    ap.add_argument("--interval", type=float, default=float(env("QWEN_LT_INTERVAL", "0") or 0),
                    help="secondi fra arrivi per --arrival uniform (0 = calibrato: S/c)")
    ap.add_argument("--arrival-seed", type=int, default=int(env("QWEN_LT_ARRIVAL_SEED", "1234")),
                    help="seme del processo di arrivo: FISSO e dichiarato, cosi' due corse "
                         "vedono la stessa sequenza di arrivi")
    ap.add_argument("--service-s", type=float, default=float(env("QWEN_LT_SERVICE_S", "0") or 0),
                    help="durata di una richiesta da sola (s); 0 = misurata con una sonda")
    ap.add_argument("--max-inflight", type=int, default=0,
                    help="valvola di sicurezza in open loop (0 = nessun tetto). ⚠️ se scatta, "
                         "l'attesa e' NEL CLIENT e non e' del server: viene registrata a parte")
    # ── soglia di prodotto ──
    ap.add_argument("--ttfa-budget-ms", type=float,
                    default=float(env("QWEN_LT_TTFA_BUDGET_MS", "500")),
                    help="soglia di TTFA per il verdetto (default 500 ms, PLAN 0.nonies)")
    ap.add_argument("--server-log", default=env("QWEN_LT_SERVER_LOG", ""),
                    help="stderr del server, letto in diretta: correla i picchi con le "
                         "righe `[BATCH] done`")
    args = ap.parse_args()

    u = urlparse(args.url)
    host, port = u.hostname or "localhost", u.port or 80
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    texts = load_texts(args.text_file, classes or None)
    levels = [int(c) for c in args.concurrency.split(",") if c.strip()]

    print(f"target   {args.url}{args.path}")
    print(f"speaker  {args.speaker} · lingua {args.language} · temp {args.temperature}")
    print(f"testi    {len(texts)} da {os.path.basename(args.text_file)}"
          f"{' (' + ','.join(classes) + ')' if classes else ''}")
    mode = f"soak {args.duration:.0f}s" if args.duration else f"{args.requests} richieste"
    print(f"sweep    concorrenza {levels} · {mode} per livello")
    print(f"soglia   TTFA p95 <= {args.ttfa_budget_ms:.0f} ms  (bersaglio di prodotto)")

    # ── ritmo degli arrivi: calibrato una volta sola, a macchina scarica ──
    service_s = args.service_s or None
    if args.arrival != "all-at-once" and not service_s and not (args.rate or args.interval):
        print("arrivi   calibrazione: una richiesta sola a macchina scarica...", end=" ", flush=True)
        service_s, probe = asyncio.run(calibrate(args, host, port, texts))
        if service_s:
            print(f"S = {service_s:.2f} s  (audio {probe['audio_s']:.1f} s, "
                  f"TTFA {probe['ttfa_ms']:.0f} ms)")
        else:
            service_s = 10.0
            print(f"FALLITA ({probe['error']}) -> assumo S = {service_s:.0f} s")
    service_s = service_s or 10.0
    if args.arrival == "all-at-once":
        print("arrivi   all-at-once — tutte a t=0 (caso peggiore sincronizzato)")
    else:
        if args.arrival == "poisson":
            ritmo = (f"rate {args.rate:.3f} req/s (dichiarato)" if args.rate
                     else f"rate = c/S = c/{service_s:.2f}s (calibrato)")
        else:
            ritmo = (f"intervallo {args.interval:.2f} s (dichiarato)" if args.interval
                     else f"intervallo = S/c = {service_s:.2f}s/c (calibrato)")
        print(f"arrivi   {args.arrival} · {ritmo} · seme {args.arrival_seed} (FISSO: due corse "
              f"vedono gli stessi arrivi)")
    if args.server_log:
        print(f"log srv  {args.server_log}  (tail in diretta per l'attribuzione)")

    all_records, summaries = [], []
    for conc in levels:
        recs, wall, n, lam, dev = asyncio.run(run_level(args, host, port, texts, conc, service_s))
        for r in recs:
            r["concurrency"] = conc
        all_records.extend(recs)
        s = summarize(recs, wall, conc, args.ttfa_budget_ms, args.arrival, lam,
                      args.arrival_seed, service_s)
        # L'ENDPOINT VA NELLA RIGA, non solo nella riga di comando. /v1/tts/stream e
        # /v1/tts producono lo stesso audio ma NON la stessa metrica: sullo stream il
        # TTFA e' il primo chunk (il silenzio che l'utente vive), su /v1/tts il primo
        # byte arriva a generazione finita, quindi "TTFA" li' vale la latenza totale e
        # non e' un vincolo di prodotto. Senza questo campo un aggregatore che mette
        # insieme due corse non ha modo di accorgersene, e stampa un ❌ falso.
        s["path"] = args.path
        s["workload"] = "stream" if args.path.endswith("/stream") else "offline"
        summaries.append(s)
        # ── il ritmo vero lo dice il livello c=1: e' misurato sulla MISCELA di testi
        # vera e su questa macchina, mentre la sonda e' un testo solo. La sonda serve
        # a partire; da qui in poi i livelli successivi sono calibrati su un numero
        # misurato. Si usa la MEDIANA e non la media: con Poisson qualche grappolo
        # capita anche a c=1, e una singola richiesta che ha aspettato in coda
        # gonfierebbe la media -> arrivi troppo radi ai livelli dopo, cioe' l'errore
        # che stiamo correggendo, di nuovo. ──
        if conc == 1 and args.arrival != "all-at-once" and not (args.rate or args.interval):
            tot = [r["total_ms"] / 1000.0 for r in recs if not r["error"] and r["total_ms"]]
            if tot:
                new_s = statistics.median(tot)
                if abs(new_s - service_s) / max(service_s, 1e-9) > 0.15:
                    print(f"      ritmo ricalibrato su c=1 (senza contesa): "
                          f"S {service_s:.2f} s -> {new_s:.2f} s")
                service_s = new_s
        err = f" · {s['errors']} ERRORI" if s["errors"] else ""
        print(f"  c={conc:<3} {n:>3} req in {wall:6.1f}s · TTFA p50 {s['ttfa_p50']:.0f}ms "
              f"p95 {s['ttfa_p95']:.0f}ms max {s['ttfa_max']:.0f}ms · "
              f"{s['ttfa_over_budget']}/{s['ok']} sopra soglia · Q {s['throughput_Q']:.2f}"
              f"{' · ' + str(len(dev)) + ' [BATCH] done' if dev else ''}{err}")

    print_table(summaries, args.ttfa_budget_ms)
    print_verdict(summaries, args.ttfa_budget_ms)
    print_spikes(summaries, args.ttfa_budget_ms)

    bad = [r for r in all_records if r["error"]]
    if bad:
        print(f"\n{len(bad)} richieste fallite; le prime:")
        for r in bad[:5]:
            print(f"  c={r['concurrency']} req{r['request_id']} status={r['status']} {r['error']}")

    if args.csv and all_records:
        keys = sorted({k for r in all_records for k in r})
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in all_records:
                w.writerow({k: r.get(k, "") for k in keys})
        print(f"record per richiesta -> {args.csv}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2, default=str)
        print(f"riepilogo per livello -> {args.json}")
    if args.save_audio:
        print(f"audio                -> {args.output_dir}/c*/")

    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
