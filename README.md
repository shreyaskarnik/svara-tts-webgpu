---
title: Svara TTS WebGPU
emoji: 🗣️
colorFrom: indigo
colorTo: purple
sdk: static
pinned: false
license: apache-2.0
short_description: Multilingual Indic TTS in your browser, via WebGPU
---

# Svara TTS · WebGPU

Browser-native multilingual TTS for **19 Indian languages** powered by [Svara](https://huggingface.co/kenpath/svara-tts-v1), [SNAC](https://huggingface.co/hubertsiuzdak/snac_24khz), and [Transformers.js v4](https://huggingface.co/docs/transformers.js). Runs 100% locally — no server, no upload.

> **Status: starting-point scaffold.** The LM + SNAC ONNX inference pipeline is wired end-to-end (no streaming yet — full generation, then SNAC decode in one shot). Voice selector covers all 38 Svara voices. First-load is ~2.0 GB (cached after).

## Architecture

```
text → tokenizer → Llama-3.2-3B (q4f16, transformers.js v4 + WebGPU) →
  audio token IDs in [128266, 156938) →
  group every 7 → SNAC frame (3 hierarchical levels) →
  SNAC decoder ONNX (q4f16/fp16 from onnx-community/snac_24khz-ONNX) →
  24 kHz mono PCM → WAV blob → <audio>
```

## Models

| Repo | Size | Notes |
|------|------|-------|
| [`shreyask/svara-tts-v1-ONNX`](https://huggingface.co/shreyask/svara-tts-v1-ONNX) | ~1.95 GB | Llama-3.2-3B q4f16, GQA, KV-cache |
| [`onnx-community/snac_24khz-ONNX`](https://huggingface.co/onnx-community/snac_24khz-ONNX) | ~26 MB (fp16) | SNAC decoder |

## Run locally

```sh
npm install
npm run dev   # http://localhost:5173
```

First run downloads ~2.0 GB into the browser cache (LM + codec + tokenizer). Subsequent runs are instant.

## Voices

Use a string of the form `"<Language Name> (<Gender>)"`. **38 voices across 19 languages**: Hindi, Bengali, Marathi, Telugu, Kannada, Tamil, Malayalam, Gujarati, Punjabi, Assamese, Bhojpuri, Magahi, Maithili, Chhattisgarhi, Bodo, Dogri, Nepali, Sanskrit, English (Indian) — male + female each.

## TODO

- Streaming generation (yield audio chunks every 4 SNAC frames instead of decoding the full clip at the end). Plumb through `model.generate({ ..., streamer })` once the v4 streamer API stabilises for our use case.
- Verify the SNAC ONNX input names — the current code uses `snacSession.inputNames[0..2]` positionally. Inspect `await snacSession.inputNames` to confirm order matches `[layer_1, layer_2, layer_3]`.
- Test the prompt format end-to-end on a Mac. Compare the generated audio against the MLX [`mlx-community/svara-tts-v1-4bit`](https://huggingface.co/mlx-community/svara-tts-v1-4bit) for the same input.
- Add `<happy>`, `<sad>`, etc. emotion tags as a UI toggle (Svara supports them).
- Service worker for offline caching of the 2.0 GB model.
- Mobile Safari memory ceiling — q4f16 may OOM on iPhone. Consider an even more aggressive quant (q4 + tied embeddings) for that target.

## Credits

- [Kenpath](https://huggingface.co/kenpath) — Svara TTS v1 base model.
- [Canopy Labs](https://huggingface.co/canopylabs) — Orpheus 3B Hindi base.
- [Hugging Face](https://github.com/huggingface/transformers.js-examples/tree/main/text-to-speech-webgpu) — original `text-to-speech-webgpu` scaffold this project forked from.
- License: Apache 2.0.
