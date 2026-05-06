---
title: Svāra TTS WebGPU
emoji: 🗣️
colorFrom: yellow
colorTo: red
sdk: static
app_build_command: npm run build
app_file: dist/index.html
pinned: false
license: apache-2.0
short_description: Multilingual Indic TTS in your browser, via WebGPU
---

# Svāra TTS · WebGPU

Browser-native multilingual TTS for **19 Indian languages** powered by [Svara](https://huggingface.co/kenpath/svara-tts-v1), [SNAC](https://huggingface.co/hubertsiuzdak/snac_24khz), and [Transformers.js v4](https://huggingface.co/docs/transformers.js). Runs 100% locally in the browser after the one-time model download.

This build adds an explicit model load step, browser-side caching, multilingual voice switching, prompt presets, and a WebGPU worker tuned around the ONNX-exported Svāra model.

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

First run downloads the selected model into the browser cache (LM + codec + tokenizer). Subsequent runs reuse the cached weights.

## Voices

Use a string of the form `"<Language Name> (<Gender>)"`. **38 voices across 19 languages**: Hindi, Bengali, Marathi, Telugu, Kannada, Tamil, Malayalam, Gujarati, Punjabi, Assamese, Bhojpuri, Magahi, Maithili, Chhattisgarhi, Bodo, Dogri, Nepali, Sanskrit, English (Indian) — male + female each.

## Notes

- `q4f16` is the fastest cold-start option and works well for short prompts.
- `q8` is heavier but can sound cleaner on more difficult prompts.
- Emotion tags such as `<happy>` and `<sad>` can be appended at the end of a line.
- Everything stays local to the browser after the model has loaded.

## Credits

- [Kenpath](https://huggingface.co/kenpath) — Svara TTS v1 base model.
- [Canopy Labs](https://huggingface.co/canopylabs) — Orpheus 3B Hindi base.
- [Hugging Face](https://github.com/huggingface/transformers.js-examples/tree/main/text-to-speech-webgpu) — original `text-to-speech-webgpu` scaffold this project forked from.
- License: Apache 2.0.
