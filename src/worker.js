// Svara TTS WebGPU worker.
//
// Architecture:
//   1) Llama-3.2-3B causal LM (loaded via @huggingface/transformers v4) emits
//      audio token IDs in the range [128266, 156938).
//   2) We group every 7-token bundle into a SNAC frame.
//   3) Offline decode mirrors Kenpath's streaming path: decode a sliding
//      4-frame SNAC window and keep samples [2048:4096] from each window.
//      That matches the codec's stable synthesis region and avoids the
//      "behind a fan" smear seen when decoding the whole sequence in one shot.

import {
  AutoTokenizer,
  AutoModelForCausalLM,
  LogitsProcessor,
  LogitsProcessorList,
  Tensor,
} from "@huggingface/transformers";
import * as ort from "onnxruntime-web/webgpu";

// ORT-Web's .wasm/.mjs files aren't served by Vite by default; vite.config.js
// copies them from node_modules to /ort-wasm/ via vite-plugin-static-copy.
ort.env.wasm.wasmPaths = "/ort-wasm/";

// --- WebGPU feature detection -----------------------------------------------
let fp16_supported = false;
try {
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("WebGPU is not supported (no adapter found)");
  fp16_supported = adapter.features.has("shader-f16");
  self.postMessage({ status: "feature-success", fp16: fp16_supported });
} catch (e) {
  self.postMessage({ status: "feature-error", data: e.toString() });
  throw e;
}

// --- Constants matching upstream Svara inference -----------------------------
const EOT = 128009;
const SOS = 128257, EOS = 128258;
const SOH = 128259, EOH = 128260;
const SOAI = 128261;
const AUDIO_OFFSET = 128266;
const AUDIO_END = AUDIO_OFFSET + 7 * 4096;
const WINDOW_FRAMES = 4;
const WINDOW_AUDIO_START = 2048;
const WINDOW_AUDIO_END = 4096;
const SAMPLE_RATE = 24000;

const SVARA_REPO = "shreyask/svara-tts-v1-ONNX";
const SNAC_REPO = "onnx-community/snac_24khz-ONNX";
const SUPPORTED_DTYPES = new Set(["q4f16", "q8"]);

// Lazy load the tokenizer once -- it's the same across dtypes.
let tokenizerPromise = null;
function getTokenizer() {
  return (tokenizerPromise ??= AutoTokenizer.from_pretrained(SVARA_REPO));
}

// SNAC decoder is small (~26 MB at fp16); load once, share across LM dtypes.
let snacPromise = null;
function getSnac() {
  return (snacPromise ??= (async () => {
    const url = `https://huggingface.co/${SNAC_REPO}/resolve/main/onnx/decoder_model${fp16_supported ? "_fp16" : ""}.onnx`;
    return ort.InferenceSession.create(url, { executionProviders: ["webgpu"] });
  })());
}

// LM is per-dtype. Cache by dtype string so switching back is instant.
const lmCache = new Map();
function getLM(dtype) {
  if (!lmCache.has(dtype)) {
    lmCache.set(
      dtype,
      AutoModelForCausalLM.from_pretrained(SVARA_REPO, {
        dtype,
        device: "webgpu",
        // Number of external data chunks to fetch alongside the .onnx graph.
        // q4f16 is one .onnx_data file; q8 is sharded into 3 chunks
        // (.onnx_data, _data_1, _data_2) to stay under the ~2 GB browser
        // ArrayBuffer ceiling. transformers.js v4 accepts a number here per
        // its types: `false` | `true` (=1) | <number of chunks>.
        use_external_data_format: dtype === "q8" ? 3 : true,
      }),
    );
  }
  return lmCache.get(dtype);
}

// --- Generation guards ------------------------------------------------------
// Svara should only emit 7-band audio tokens followed by END_OF_SPEECH. If we
// let the sampler wander into the text/control vocab, the rest of the clip
// turns phasey/robotic because frame alignment is lost.
class SvaraLogitsProcessor extends LogitsProcessor {
  constructor(promptLength) {
    super();
    this.promptLength = promptLength;
  }

  _call(inputIds, logits) {
    for (let i = 0; i < inputIds.length; i++) {
      const data = logits[i].data;
      const step = inputIds[i].length - this.promptLength;

      if (step === 0) {
        data.fill(-Infinity);
        data[SOAI] = 0;
        continue;
      }

      if (step === 1) {
        data.fill(-Infinity);
        data[SOS] = 0;
        continue;
      }

      const eosLogit = data[EOS];
      data.subarray(0, AUDIO_OFFSET).fill(-Infinity);
      data.subarray(AUDIO_END).fill(-Infinity);
      data[EOS] = eosLogit;
    }
    return logits;
  }
}

function buildLogitsProcessor(promptLength) {
  const list = new LogitsProcessorList();
  list.push(new SvaraLogitsProcessor(promptLength));
  return list;
}

function estimateAudioTokenBudget(text) {
  const spokenText = stripTrailingEmotionTag(text);
  const graphemeCount = Array.from(
    new Intl.Segmenter(undefined, { granularity: "grapheme" }).segment(spokenText),
    ({ segment }) => segment,
  ).filter((segment) => /\S/u.test(segment)).length;
  const punctuationGroups = Array.from(
    spokenText.matchAll(/[.,!?;:।॥…\-—]+/gu),
  ).length;
  const wordCount = spokenText.split(/\s+/u).filter(Boolean).length;

  const roughBudget = graphemeCount * 12 + wordCount * 20 + punctuationGroups * 28 + 84;
  const clampedBudget = Math.max(224, Math.min(1120, roughBudget));
  return Math.ceil(clampedBudget / 7) * 7;
}

function getTrailingEmotionTag(text) {
  return text.match(/\s*(<[^>]+>)\s*$/u)?.[1] ?? "";
}

function stripTrailingEmotionTag(text) {
  return text.replace(/\s*<[^>]+>\s*$/u, "").trim();
}

function normalizeTextForSvara(text) {
  return text
    .replace(/\.{2,}/gu, ",")
    .replace(/…+/gu, ",")
    .replace(/[—–]+/gu, ",")
    .replace(/\s+/gu, " ")
    .replace(/\s*([,.;!?।॥])\s*/gu, "$1 ")
    .trim();
}

function countChunkGraphemes(chunk) {
  return Array.from(
    new Intl.Segmenter(undefined, { granularity: "grapheme" }).segment(chunk),
    ({ segment }) => segment,
  ).filter((segment) => /\S/u.test(segment)).length;
}

function countChunkWords(chunk) {
  return chunk.split(/\s+/u).filter(Boolean).length;
}

function splitLongChunk(chunk) {
  const graphemeCount = countChunkGraphemes(chunk);
  const wordCount = countChunkWords(chunk);

  if (graphemeCount <= 28 || wordCount <= 5) return [chunk];

  const parts = chunk.split(/\s*,\s*/u).map((part) => part.trim()).filter(Boolean);
  return parts.length > 1 ? parts : [chunk];
}

function mergeTinyChunks(chunks) {
  const merged = [];

  for (const chunk of chunks) {
    const graphemeCount = countChunkGraphemes(chunk);
    const wordCount = countChunkWords(chunk);
    const shouldAttach =
      merged.length > 0 &&
      !/[.!?।॥]$/u.test(merged.at(-1)) &&
      (graphemeCount < 10 || wordCount < 3);

    if (shouldAttach) {
      merged[merged.length - 1] = `${merged.at(-1)}, ${chunk}`;
      continue;
    }

    merged.push(chunk);
  }

  return merged;
}

function splitTextForSvara(text) {
  const emotionTag = getTrailingEmotionTag(text);
  const spokenText = normalizeTextForSvara(stripTrailingEmotionTag(text));

  if (!spokenText) return [];

  const chunks = mergeTinyChunks(
    spokenText
      .match(/[^.!?।॥]+[.!?।॥]?/gu)
      ?.map((part) => part.trim())
      .filter(Boolean)
      .flatMap(splitLongChunk) ?? [],
  );

  if (!emotionTag) return chunks;
  return chunks.map((chunk, index) =>
    index === chunks.length - 1 ? `${chunk} ${emotionTag}` : chunk,
  );
}

function mergeTinyLeadingChunks(chunks) {
  const merged = [];

  for (let i = 0; i < chunks.length; i++) {
    const chunk = chunks[i];
    const graphemeCount = countChunkGraphemes(chunk);
    const wordCount = countChunkWords(chunk);

    if (graphemeCount < 10 && wordCount < 3) {
      if (i + 1 < chunks.length) {
        chunks[i + 1] = `${chunk}, ${chunks[i + 1]}`;
        continue;
      }
      if (merged.length > 0) {
        merged[merged.length - 1] = `${merged.at(-1)}, ${chunk}`;
        continue;
      }
    }

    merged.push(chunk);
  }

  return merged;
}

function splitEmotionSafeTextForSvara(text) {
  const emotionTag = getTrailingEmotionTag(text);
  const spokenText = normalizeTextForSvara(stripTrailingEmotionTag(text));

  if (!spokenText) return [];

  const chunks = spokenText
    .match(/[^.!?।॥]+[.!?।॥]?/gu)
    ?.map((part) => part.trim())
    .filter(Boolean)
    .flatMap((sentence) => {
      const commaParts = sentence
        .split(/\s*,\s*/u)
        .map((part) => part.trim())
        .filter(Boolean);
      return mergeTinyLeadingChunks(commaParts);
    }) ?? [];

  if (!emotionTag) return chunks;
  return chunks.map((chunk, index) =>
    index === chunks.length - 1 ? `${chunk} ${emotionTag}` : chunk,
  );
}

function splitFinalEmotionClauseTextForSvara(text) {
  const emotionTag = getTrailingEmotionTag(text);
  const spokenText = normalizeTextForSvara(stripTrailingEmotionTag(text));

  if (!spokenText) return [];

  const chunks = mergeTinyLeadingChunks(
    spokenText.split(/\s*,\s*/u).map((part) => part.trim()).filter(Boolean),
  );

  if (!emotionTag) return chunks;
  return chunks.map((chunk, index) =>
    index === chunks.length - 1 ? `${chunk} ${emotionTag}` : chunk,
  );
}

function buildPromptVariants(text) {
  const rawText = text.trim();
  const spokenText = normalizeTextForSvara(stripTrailingEmotionTag(text));
  if (!rawText && !spokenText) return [];

  const variants = rawText ? [[rawText]] : [];
  variants.push(
    splitTextForSvara(text),
    splitEmotionSafeTextForSvara(text),
    splitFinalEmotionClauseTextForSvara(text),
  );

  if (getTrailingEmotionTag(text)) {
    variants.push([spokenText]);
    variants.push(splitEmotionSafeTextForSvara(spokenText));
  }

  const seen = new Set();
  return variants.filter((chunks) => {
    if (chunks.length === 0) return false;
    const key = chunks.join("\u241e");
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function pauseDurationForChunk(chunk, isLast) {
  if (isLast) return 0;
  const trimmed = chunk.trim();
  if (/[!?]$/u.test(trimmed)) return 0.26;
  if (/[.]$/u.test(trimmed)) return 0.3;
  return 0.18;
}

function concatFloat32Arrays(chunks) {
  const totalLength = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const merged = new Float32Array(totalLength);
  let offset = 0;
  for (const chunk of chunks) {
    merged.set(chunk, offset);
    offset += chunk.length;
  }
  return merged;
}

function pcmStats(samples) {
  let peak = 0;
  let sumSquares = 0;

  for (let i = 0; i < samples.length; i++) {
    const value = Math.abs(samples[i]);
    if (value > peak) peak = value;
    sumSquares += value * value;
  }

  const rms = samples.length > 0 ? Math.sqrt(sumSquares / samples.length) : 0;
  return { peak, rms };
}

function isNearlySilent(samples) {
  const { peak, rms } = pcmStats(samples);
  return peak < 0.006 && rms < 0.0015;
}

function isComplexQ4Prompt(text) {
  const spokenText = stripTrailingEmotionTag(text);
  const wordCount = countChunkWords(spokenText);
  const punctuationGroups = Array.from(
    spokenText.matchAll(/[.,!?;:।॥…\-—]+/gu),
  ).length;
  return punctuationGroups >= 3 || wordCount >= 8 || (
    getTrailingEmotionTag(text) && punctuationGroups >= 1 && wordCount >= 5
  );
}

async function synthesizeChunks(tokenizer, lm, speaker_id, chunks, generation) {
  const pcmChunks = [];

  for (let index = 0; index < chunks.length; index++) {
    const chunk = chunks[index];
    const promptIds = buildPrompt(tokenizer, chunk, speaker_id);
    const inputIds = new Tensor(
      "int64",
      BigInt64Array.from(promptIds.map(BigInt)),
      [1, promptIds.length],
    );

    const maxAudioTokens = estimateAudioTokenBudget(chunk);
    const out = await lm.generate({
      inputs: inputIds,
      max_new_tokens: maxAudioTokens + 3,
      logits_processor: buildLogitsProcessor(promptIds.length),
      ...generation,
      repetition_penalty: 1.0,
      eos_token_id: EOS,
    });

    const allIds = Array.from(out.data, (x) => Number(x));
    const audioIds = extractAudioTokens(allIds, promptIds.length);
    if (audioIds.length === 0) {
      throw new Error(`LM produced no audio tokens for chunk ${index + 1}/${chunks.length}.`);
    }

    const pcm = await decodeSnacStable(audioIds);
    pcmChunks.push(pcm);

    const pauseSeconds = pauseDurationForChunk(chunk, index === chunks.length - 1);
    if (pauseSeconds > 0) {
      pcmChunks.push(new Float32Array(Math.round(SAMPLE_RATE * pauseSeconds)));
    }
  }

  return concatFloat32Arrays(pcmChunks);
}

// --- Token-stream → SNAC code conversion ------------------------------------
// Reference: mlx_audio/tts/models/llama/llama.py:codes_to_layers
//   layer_1 (band 0):           [c0]                — 1 code per coarse frame
//   layer_2 (bands 1, 4):       [c1, c4]            — 2 codes per coarse frame
//   layer_3 (bands 2, 3, 5, 6): [c2, c3, c5, c6]    — 4 codes per coarse frame
function codesToLayers(audioTokenIds) {
  const N = Math.floor(audioTokenIds.length / 7);
  const l1 = new BigInt64Array(N);
  const l2 = new BigInt64Array(N * 2);
  const l3 = new BigInt64Array(N * 4);
  for (let i = 0; i < N; i++) {
    const base = i * 7;
    l1[i]         = BigInt(audioTokenIds[base    ] - AUDIO_OFFSET - 0 * 4096);
    l2[2 * i + 0] = BigInt(audioTokenIds[base + 1] - AUDIO_OFFSET - 1 * 4096);
    l3[4 * i + 0] = BigInt(audioTokenIds[base + 2] - AUDIO_OFFSET - 2 * 4096);
    l3[4 * i + 1] = BigInt(audioTokenIds[base + 3] - AUDIO_OFFSET - 3 * 4096);
    l2[2 * i + 1] = BigInt(audioTokenIds[base + 4] - AUDIO_OFFSET - 4 * 4096);
    l3[4 * i + 2] = BigInt(audioTokenIds[base + 5] - AUDIO_OFFSET - 5 * 4096);
    l3[4 * i + 3] = BigInt(audioTokenIds[base + 6] - AUDIO_OFFSET - 6 * 4096);
  }
  return { l1, l2, l3, N };
}

async function decodeSnacWindow(audioTokenIds) {
  const snac = await getSnac();
  const { l1, l2, l3, N } = codesToLayers(audioTokenIds);
  const feeds = {
    [snac.inputNames[0]]: new ort.Tensor("int64", l1, [1, N]),
    [snac.inputNames[1]]: new ort.Tensor("int64", l2, [1, N * 2]),
    [snac.inputNames[2]]: new ort.Tensor("int64", l3, [1, N * 4]),
  };
  const out = await snac.run(feeds);
  return out[snac.outputNames[0]].data;
}

async function decodeSnacStable(audioTokenIds) {
  const numFrames = Math.floor(audioTokenIds.length / 7);
  if (numFrames === 0) return new Float32Array(0);

  if (numFrames < WINDOW_FRAMES) {
    return await decodeSnacWindow(audioTokenIds);
  }

  const chunks = [];
  let totalLength = 0;

  for (let start = 0; start <= numFrames - WINDOW_FRAMES; start++) {
    const windowIds = audioTokenIds.slice(start * 7, (start + WINDOW_FRAMES) * 7);
    const decoded = await decodeSnacWindow(windowIds);
    const stable = decoded.slice(WINDOW_AUDIO_START, WINDOW_AUDIO_END);
    chunks.push(stable);
    totalLength += stable.length;
  }

  const merged = new Float32Array(totalLength);
  let offset = 0;
  for (const chunk of chunks) {
    merged.set(chunk, offset);
    offset += chunk.length;
  }
  return merged;
}

// Match the exported ONNX repo README:
//   [SOH, BOS, "<voice>: <text>" tokens, EOT, EOH]
// The model predicts SOAI -> SOS -> audio tokens -> EOS itself.
function buildPrompt(tokenizer, text, voice) {
  const body = tokenizer.encode(`${voice}: ${text}`, { add_special_tokens: false });
  return [SOH, tokenizer.bos_token_id, ...body, EOT, EOH];
}

// Keep audio tokens after the first START_OF_SPEECH emitted by the model.
function extractAudioTokens(allTokenIds, promptLength) {
  let sosIdx = -1;
  for (let i = promptLength; i < allTokenIds.length; i++) {
    if (allTokenIds[i] === SOS) {
      sosIdx = i;
      break;
    }
  }
  if (sosIdx === -1) return [];

  const audio = [];
  for (let i = sosIdx + 1; i < allTokenIds.length; i++) {
    const tokenId = allTokenIds[i];
    if (tokenId === EOS) break;
    if (tokenId >= AUDIO_OFFSET && tokenId < AUDIO_END) {
      audio.push(tokenId);
    }
  }
  return audio.slice(0, audio.length - (audio.length % 7));
}

// --- WAV encoder (24 kHz, mono, PCM16) --------------------------------------
function pcmFloat32ToWav(samples, sampleRate) {
  const bufLen = 44 + samples.length * 2;
  const buf = new ArrayBuffer(bufLen);
  const v = new DataView(buf);
  let p = 0;
  const w = (s) => { for (let i = 0; i < s.length; i++) v.setUint8(p++, s.charCodeAt(i)); };
  w("RIFF");
  v.setUint32(p, 36 + samples.length * 2, true); p += 4;
  w("WAVEfmt ");
  v.setUint32(p, 16, true); p += 4;
  v.setUint16(p, 1, true); p += 2;
  v.setUint16(p, 1, true); p += 2;
  v.setUint32(p, sampleRate, true); p += 4;
  v.setUint32(p, sampleRate * 2, true); p += 4;
  v.setUint16(p, 2, true); p += 2;
  v.setUint16(p, 16, true); p += 2;
  w("data");
  v.setUint32(p, samples.length * 2, true); p += 4;
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    v.setInt16(p, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    p += 2;
  }
  return buf;
}

// --- Sampling defaults per dtype --------------------------------------------
// Transformers.js v4 currently ignores top-k/top-p on this path, so unconstrained
// sampling drifts badly on quantized Svara and turns later words robotic. Use
// greedy decoding by default for stability; q8 can tolerate a little sampling.
function generationFor(dtype) {
  return dtype === "q8"
    ? { do_sample: true, temperature: 0.35, min_new_tokens: 30 }
    : { do_sample: false, min_new_tokens: 30 };
}

function generationPlansFor(dtype, text) {
  const base = generationFor(dtype);
  if (dtype !== "q4f16" || !isComplexQ4Prompt(text)) {
    return [base];
  }

  return [
    {
      do_sample: true,
      temperature: 0.6,
      top_k: 40,
      top_p: 0.9,
      min_new_tokens: 30,
    },
    base,
  ];
}

// --- Message handler --------------------------------------------------------
self.addEventListener("message", async (e) => {
  const { type, text, speaker_id, dtype: requested } = e.data;
  const dtype = SUPPORTED_DTYPES.has(requested) ? requested : "q4f16";

  try {
    if (type === "preload") {
      // Triggered by the explicit "Load model" action in the UI.
      self.postMessage({ status: "loading", dtype });
      await Promise.all([getTokenizer(), getSnac(), getLM(dtype)]);
      self.postMessage({ status: "ready", dtype });
      return;
    }

    self.postMessage({ status: "loading", dtype });
    const [tokenizer, lm] = await Promise.all([getTokenizer(), getLM(dtype)]);
    await getSnac(); // warm

    const variants = buildPromptVariants(text);
    if (variants.length === 0) {
      throw new Error("No speakable text found after normalization.");
    }

    const generations = generationPlansFor(dtype, text);
    let mergedPcm = null;
    let lastError = null;

    for (const generation of generations) {
      for (const chunks of variants) {
        try {
          const candidate = await synthesizeChunks(
            tokenizer,
            lm,
            speaker_id,
            chunks,
            generation,
          );
          if (isNearlySilent(candidate)) {
            lastError = new Error("Generated near-silent audio.");
            continue;
          }
          mergedPcm = candidate;
          break;
        } catch (err) {
          lastError = err;
        }
      }
      if (mergedPcm) {
        break;
      }
    }

    if (!mergedPcm) {
      throw lastError ?? new Error("Synthesis failed for all prompt variants.");
    }

    const wav = pcmFloat32ToWav(mergedPcm, SAMPLE_RATE);
    const blob = new Blob([wav], { type: "audio/wav" });
    self.postMessage({
      status: "complete",
      audio: URL.createObjectURL(blob),
      text,
      voice: speaker_id,
      dtype,
    });
  } catch (err) {
    self.postMessage({ status: "error", data: String(err), dtype });
    console.error(err);
  }
});
