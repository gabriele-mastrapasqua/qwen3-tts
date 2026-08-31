import torch
from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram
from torch.utils.data import Dataset

class TTSDataset(Dataset):
    def __init__(self, data_list, processor, config: Qwen3TTSConfig, lag_num=-1):
        self.data_list = data_list; self.processor = processor
        self.lag_num = lag_num; self.config = config

    def __len__(self): return len(self.data_list)

    def _build_assistant_text(self, text):
        return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"

    def _build_instruct(self, instruct):
        return f"<|im_start|>user\n{instruct}<|im_end|>\n"

    def _tok(self, text):
        inp = self.processor(text=text, return_tensors="pt", padding=True)["input_ids"]
        return inp.unsqueeze(0) if inp.dim() == 1 else inp

    @torch.inference_mode()
    def extract_mels(self, audio, sr):
        assert sr == 24000, "Only support 24kHz audio"
        return mel_spectrogram(torch.from_numpy(audio).unsqueeze(0), n_fft=1024, num_mels=128,
                               sampling_rate=24000, hop_size=256, win_size=1024, fmin=0, fmax=12000).transpose(1, 2)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        text_ids = self._tok(self._build_assistant_text(item["text"]))[:, :-5]
        instruct = item.get("instruct", "") or ""
        instruct_ids = self._tok(self._build_instruct(instruct)) if instruct.strip() else torch.zeros((1, 0), dtype=torch.long)
        audio_codes = torch.tensor(item["audio_codes"], dtype=torch.long)
        return {"instruct_ids": instruct_ids, "text_ids": text_ids, "audio_codes": audio_codes}

    def collate_fn(self, batch):
        assert self.lag_num == -1
        cfg, tk = self.config, self.config.talker_config
        lens = [b["instruct_ids"].shape[1] + b["text_ids"].shape[1] + b["audio_codes"].shape[0] for b in batch]
        T = max(lens) + 8
        B = len(batch)
        input_ids = torch.zeros((B, T, 2), dtype=torch.long)
        codec_ids = torch.zeros((B, T, 16), dtype=torch.long)
        tmask = torch.zeros((B, T), dtype=torch.bool)
        cmask = torch.zeros((B, T), dtype=torch.bool)
        codec_mask = torch.zeros((B, T), dtype=torch.bool)
        attn = torch.zeros((B, T), dtype=torch.long)
        labels = torch.full((B, T), -100, dtype=torch.long)

        for i, d in enumerate(batch):
            ins = d["instruct_ids"][0]; p = ins.shape[0]
            tids = d["text_ids"]; ac0 = d["audio_codes"][:, 0]; acs = d["audio_codes"]
            tlen = tids.shape[1]; clen = ac0.shape[0]
            if p > 0:
                input_ids[i, :p, 0] = ins
                input_ids[i, :p, 1] = tk.codec_pad_id
                tmask[i, :p] = True; cmask[i, :p] = True; attn[i, :p] = 1
            o = p
            input_ids[i, o:o+3, 0]       = tids[0, :3]
            input_ids[i, o+3:o+7, 0]     = cfg.tts_pad_token_id
            input_ids[i, o+7, 0]         = cfg.tts_bos_token_id
            input_ids[i, o+8:o+8+tlen-3, 0] = tids[0, 3:]
            input_ids[i, o+8+tlen-3, 0]  = cfg.tts_eos_token_id
            input_ids[i, o+8+tlen-2:o+8+tlen+clen, 0] = cfg.tts_pad_token_id
            tmask[i, :o+8+tlen+clen] = True
            input_ids[i, o+3:o+8, 1] = torch.tensor([tk.codec_nothink_id, tk.codec_think_bos_id,
                                                     tk.codec_think_eos_id, 0, tk.codec_pad_id])
            input_ids[i, o+8:o+8+tlen-3, 1] = tk.codec_pad_id
            input_ids[i, o+8+tlen-3, 1]     = tk.codec_pad_id
            input_ids[i, o+8+tlen-2, 1]     = tk.codec_bos_id
            input_ids[i, o+8+tlen-1:o+8+tlen-1+clen, 1] = ac0
            input_ids[i, o+8+tlen-1+clen, 1] = tk.codec_eos_token_id
            labels[i, o+8+tlen-1:o+8+tlen-1+clen] = ac0
            labels[i, o+8+tlen-1+clen] = tk.codec_eos_token_id
            codec_ids[i, o+8+tlen-1:o+8+tlen-1+clen, :] = acs
            cmask[i, o+3:o+8+tlen+clen] = True
            cmask[i, o+6] = False
            codec_mask[i, o+8+tlen-1:o+8+tlen-1+clen] = True
            attn[i, :o+8+tlen+clen] = 1

        return {"input_ids": input_ids, "attention_mask": attn,
                "text_embedding_mask": tmask.unsqueeze(-1), "codec_embedding_mask": cmask.unsqueeze(-1),
                "codec_0_labels": labels, "codec_ids": codec_ids, "codec_mask": codec_mask,
                "spk_pos": torch.tensor([b["instruct_ids"].shape[1] + 6 for b in batch], dtype=torch.long)}
