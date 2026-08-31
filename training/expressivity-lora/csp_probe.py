import argparse, json, os, random, sys
import torch
import torch.nn as nn

class LayerProbe(nn.Module):
    """Softmax-weighted combination of per-layer hidden states -> emotion classifier.

    n_layers = number of layer outputs being weighted (= talker num_layers + 1, embeddings + each block).
    Only this module's params are trained; the TTS backbone stays frozen.
    """

    def __init__(self, n_layers, hidden, n_emotions, p_drop=0.1):
        super().__init__()
        self.layer_logits = nn.Parameter(torch.zeros(n_layers))
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden), nn.Dropout(p_drop),
            nn.Linear(hidden, hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, n_emotions),
        )

    def weights(self):
        return torch.softmax(self.layer_logits, dim=0)

    def forward(self, layer_pooled):
        w = self.weights().view(1, -1, 1)
        mixed = (layer_pooled * w).sum(dim=1)
        return self.classifier(mixed)

def select_layers(weights, top_k, n_layers):
    """Pick the characteristic-specific layers from per-layer importance.

    Returns layer INDICES in talker-block space (0..num_layers-1). hidden_states index 0 = the embedding
    output (NOT a transformer block) -> we drop it from selection and map block i -> hidden index i+1.
    Paper picks {argmax, argmin}; we return both that pair and a top_k list for our own sweeps.
    """
    w = weights.tolist()
    block = [(i - 1, w[i]) for i in range(1, n_layers)]
    ranked = sorted(block, key=lambda t: t[1], reverse=True)
    topk = sorted(b for b, _ in ranked[:top_k])
    hi = ranked[0][0]
    lo = sorted(block, key=lambda t: t[1])[0][0]
    paper_pair = sorted({hi, lo})
    return {"top_k": topk, "paper_pair": paper_pair,
            "ranked_blocks": [{"layer": b, "weight": round(wt, 5)} for b, wt in ranked]}

def run_probe(args):
    from accelerate import Accelerator
    from gpu_dataset_expr_lang import TTSDataset
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from transformers import AutoConfig

    acc = Accelerator(mixed_precision="bf16")
    qwen3tts = Qwen3TTSModel.from_pretrained(args.init_model_path, torch_dtype=torch.bfloat16,
                                             attn_implementation="eager")
    config = AutoConfig.from_pretrained(args.init_model_path)
    for p in qwen3tts.model.parameters():
        p.requires_grad = False
    qwen3tts.model.eval()

    data = [json.loads(l) for l in open(args.train_jsonl)]
    emos = sorted({r["emotion"] for r in data})
    emo2idx = {e: i for i, e in enumerate(emos)}
    acc.print(f"[probe] {len(data)} rows, {len(emos)} emotions: {emos}")

    ds = TTSDataset(data, qwen3tts.processor, config)
    labels_by_idx = [emo2idx[r["emotion"]] for r in data]

    def collate(batch_items):
        dicts = [b[0] for b in batch_items]
        labs = torch.tensor([b[1] for b in batch_items], dtype=torch.long)
        out = ds.collate_fn(dicts)
        out["emotion_label"] = labs
        return out

    paired = list(zip([ds[i] for i in range(len(ds))], labels_by_idx)) if args.preload else None
    if paired is None:
        class _Paired(torch.utils.data.Dataset):
            def __len__(self): return len(ds)
            def __getitem__(self, i): return (ds[i], labels_by_idx[i])
        paired = _Paired()
    dl = DataLoader(paired, batch_size=args.batch_size, shuffle=True, collate_fn=collate)

    n_layers = config.talker_config.num_hidden_layers + 1
    hidden = config.talker_config.hidden_size
    probe = LayerProbe(n_layers, hidden, len(emos), p_drop=args.dropout)
    opt = AdamW(probe.parameters(), lr=args.lr, weight_decay=0.01)
    model, probe, opt, dl = acc.prepare(qwen3tts.model, probe, opt, dl)
    lossf = nn.CrossEntropyLoss()

    probe.train()
    for epoch in range(args.epochs):
        seen = correct = 0
        for step, b in enumerate(dl):
            input_ids = b["input_ids"]; B = input_ids.shape[0]
            tmask = b["text_embedding_mask"]; cmask = b["codec_embedding_mask"]
            te = model.talker.model.text_embedding(input_ids[:, :, 0]) * tmask
            ce = model.talker.model.codec_embedding(input_ids[:, :, 1]) * cmask
            emb = te + ce
            for i in range(1, 16):
                emb = emb + model.talker.code_predictor.get_input_embeddings()[i - 1](b["codec_ids"][:, :, i]) * b["codec_mask"].unsqueeze(-1)
            with torch.no_grad():
                out = model.talker(inputs_embeds=emb[:, :-1, :], attention_mask=b["attention_mask"][:, :-1],
                                   output_hidden_states=True)
            hs_tuple = out.hidden_states[0]
            assert len(hs_tuple) == n_layers, f"expected {n_layers} layer outputs, got {len(hs_tuple)}"
            cm = b["codec_mask"][:, :-1].unsqueeze(-1).to(emb.dtype)
            denom = cm.sum(dim=1).clamp(min=1.0)
            pooled = torch.stack([(h.to(emb.dtype) * cm).sum(dim=1) / denom for h in hs_tuple], dim=1)
            logits = probe(pooled.float())
            loss = lossf(logits, b["emotion_label"])
            acc.backward(loss)
            opt.step(); opt.zero_grad()
            seen += B; correct += (logits.argmax(-1) == b["emotion_label"]).sum().item()
            if step % 10 == 0:
                acc.print(f"epoch {epoch} step {step} loss {loss.item():.4f} acc {correct/max(seen,1):.3f}")
        acc.print(f"[probe] epoch {epoch} train-acc {correct/max(seen,1):.3f}")

    if acc.is_main_process:
        w = acc.unwrap_model(probe).weights().detach().cpu()
        sel = select_layers(w, args.top_k, n_layers)
        result = {"emotions": emos, "n_layers_incl_embed": n_layers,
                  "hidden_index_note": "hidden_states[0]=embedding; block i -> hidden i+1",
                  "per_hidden_weight": [round(x, 5) for x in w.tolist()],
                  "selected": sel, "final_train_acc": round(correct / max(seen, 1), 4)}
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(result, f, indent=2)
        acc.print(f"[probe] selected paper_pair={sel['paper_pair']} top_k={sel['top_k']} -> {args.out_json}")
        acc.print(json.dumps(result, indent=2))

def _self_test():
    """Plant emotion signal in ONE layer of synthetic hidden states; the probe must select that layer."""
    torch.manual_seed(0)
    n_layers, hidden, n_emo, N = 8, 64, 4, 512
    PLANT = 5
    y = torch.randint(0, n_emo, (N,))
    centers = torch.randn(n_emo, hidden) * 3.0
    pooled = torch.randn(N, n_layers, hidden) * 0.5
    pooled[:, PLANT, :] += centers[y]
    probe = LayerProbe(n_layers, hidden, n_emo, p_drop=0.0)
    opt = torch.optim.AdamW(probe.parameters(), lr=5e-3)
    lossf = nn.CrossEntropyLoss()
    for it in range(400):
        idx = torch.randint(0, N, (64,))
        logits = probe(pooled[idx])
        loss = lossf(logits, y[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    w = probe.weights().detach()
    acc = (probe(pooled).argmax(-1) == y).float().mean().item()
    sel = select_layers(w, top_k=2, n_layers=n_layers)
    print("per-layer weights:", [round(x, 3) for x in w.tolist()])
    print(f"argmax hidden-idx: {int(w.argmax())} (planted {PLANT})   train-acc: {acc:.3f}")
    print("selected:", sel)
    assert int(w.argmax()) == PLANT, "probe did NOT concentrate weight on the planted layer"
    assert (PLANT - 1) in sel["top_k"], "planted block not in top_k selection"
    assert acc > 0.9, "probe failed to classify the planted signal"
    print("SELF-TEST PASS — probe concentrates weight on the emotion-carrying layer and selects it.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true", help="run the model-free sanity check and exit")
    ap.add_argument("--train_jsonl")
    ap.add_argument("--init_model_path", default="/root/qwen-ft/models/1.7B-CustomVoice")
    ap.add_argument("--out_json", default="/root/qwen-ft/csp_layers.json")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--top_k", type=int, default=2, help="how many top-weighted blocks to fine-tune in CSP-FT")
    ap.add_argument("--preload", action="store_true", help="materialise all encoded items up-front (more RAM)")
    args = ap.parse_args()
    if args.self_test:
        _self_test(); return
    if not args.train_jsonl:
        ap.error("--train_jsonl required (or use --self-test)")
    run_probe(args)

if __name__ == "__main__":
    main()
