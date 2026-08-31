#!/usr/bin/env python3
import argparse, json, os, random
import torch
from accelerate import Accelerator
from dataset import TTSDataset
from peft import LoraConfig, get_peft_model
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig

PRESET_IDS = [3066, 3065, 3010, 3061, 2861, 2873, 2864, 2875, 2878]
TRAIN_LAYERS = list(range(16, 27))

def train():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init_model_path", required=True, help="Qwen3-TTS 1.7B CustomVoice dir")
    ap.add_argument("--train_jsonl", required=True, help="rows with text/instruct/audio_codes")
    ap.add_argument("--output_dir", default="out_lora")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--num_epochs", type=int, default=8)
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=64)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    args = ap.parse_args()

    acc = Accelerator(gradient_accumulation_steps=4, mixed_precision="bf16")
    qwen3tts = Qwen3TTSModel.from_pretrained(args.init_model_path, torch_dtype=torch.bfloat16,
                                             attn_implementation="eager")
    config = AutoConfig.from_pretrained(args.init_model_path)

    lcfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"],
        layers_to_transform=TRAIN_LAYERS, layers_pattern="layers",
    )
    model = get_peft_model(qwen3tts.model, lcfg)
    if acc.is_main_process:
        model.print_trainable_parameters()

    data = [json.loads(l) for l in open(args.train_jsonl)]
    ds = TTSDataset(data, qwen3tts.processor, config)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=ds.collate_fn)
    opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    model, opt, dl = acc.prepare(model, opt, dl)
    model.train()

    for epoch in range(args.num_epochs):
        for step, b in enumerate(dl):
            with acc.accumulate(model):
                input_ids = b["input_ids"]; B = input_ids.shape[0]
                tmask = b["text_embedding_mask"]; cmask = b["codec_embedding_mask"]
                spk_pos = b["spk_pos"]
                tk = model.talker
                te = tk.model.text_embedding(input_ids[:, :, 0]) * tmask
                ce = tk.model.codec_embedding(input_ids[:, :, 1]) * cmask
                spk_ids = torch.tensor([random.choice(PRESET_IDS) for _ in range(B)], device=model.device)
                spk_emb = tk.model.codec_embedding(spk_ids).to(ce.dtype)
                for j in range(B):
                    ce[j, spk_pos[j], :] = spk_emb[j]
                emb = te + ce
                for i in range(1, 16):
                    emb = emb + tk.code_predictor.get_input_embeddings()[i - 1](b["codec_ids"][:, :, i]) * b["codec_mask"].unsqueeze(-1)
                out = tk(inputs_embeds=emb[:, :-1, :], attention_mask=b["attention_mask"][:, :-1],
                         labels=b["codec_0_labels"][:, 1:], output_hidden_states=True)
                hs = out.hidden_states[0][-1][b["codec_mask"][:, :-1]]
                sub_ids = b["codec_ids"][b["codec_mask"]]
                _, sub_loss = tk.forward_sub_talker_finetune(sub_ids, hs)
                loss = out.loss + 0.3 * sub_loss
                acc.backward(loss)
                if acc.sync_gradients:
                    acc.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad()
            if step % 10 == 0:
                acc.print(f"epoch {epoch} step {step} loss {loss.item():.4f}")

    if acc.is_main_process:
        out_dir = os.path.join(args.output_dir, "adapter-final")
        acc.unwrap_model(model).save_pretrained(out_dir)
        sz = sum(os.path.getsize(os.path.join(out_dir, f)) for f in os.listdir(out_dir) if f.endswith(".safetensors"))
        acc.print(f"[save] adapter -> {out_dir}  ({sz/1e6:.1f} MB)")

if __name__ == "__main__":
    train()
