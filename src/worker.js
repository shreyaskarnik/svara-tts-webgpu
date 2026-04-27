// Svara TTS WebGPU worker.
//
// Architecture:
//   1) Llama-3.2-3B causal LM (loaded via @huggingface/transformers v4) emits
//      audio token IDs in the range [128266, 156938).
//   2) We strip every 7-token group into a SNAC frame, buffer 4 frames, and
//      feed them to the SNAC 24 kHz decoder (loaded via onnxruntime-web).
//   3) Per 4-frame window we keep PCM samples [2048:4096] -- this avoids
//      click artefacts at frame boundaries, matching the reference inference.

import {
  AutoTokenizer,
  AutoModelForCausalLM,
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

// --- Constants matching mlx_audio/tts/models/llama/llama.py -----------------
const SOH = 128259, EOH = 128260, EOT = 128009;
const SOS = 128257, EOS = 128258;
const AUDIO_OFFSET = 128266;
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

async function decodeSnacAll(audioTokenIds) {
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

// Match mlx_audio/tts/models/llama/llama.py:prepare_input_ids:
//   [SOH, BOS, "<voice>: <text>" tokens, EOT, EOH]
// LM predicts SOAI -> SOS -> audio tokens -> EOS itself.
function buildPrompt(tokenizer, text, voice) {
  const body = tokenizer.encode(`${voice}: ${text}`, { add_special_tokens: false });
  return [SOH, tokenizer.bos_token_id, ...body, EOT, EOH];
}

// Match mlx_audio/tts/models/llama/llama.py:parse_output: slice after the LAST
// SOS the model emitted, strip EOS anywhere, trim to a multiple of 7.
// Do NOT range-filter -- dropping a non-audio token shifts (position % 7) of
// every subsequent code and corrupts the audio.
function extractAudioTokens(allTokenIds) {
  let sosIdx = -1;
  for (let i = allTokenIds.length - 1; i >= 0; i--) {
    if (allTokenIds[i] === SOS) { sosIdx = i; break; }
  }
  const audio = [];
  for (let i = sosIdx + 1; i < allTokenIds.length; i++) {
    if (allTokenIds[i] !== EOS) audio.push(allTokenIds[i]);
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
// q4f16 has more quant noise so we tighten temperature; q8 is closer to bf16
// and tolerates the upstream-recommended 0.75.
function samplingFor(dtype) {
  return dtype === "q8"
    ? { temperature: 0.75, top_p: 0.9, top_k: 40 }
    : { temperature: 0.6,  top_p: 0.9, top_k: 40 };
}

// --- Message handler --------------------------------------------------------
self.addEventListener("message", async (e) => {
  const { type, text, speaker_id, dtype: requested } = e.data;
  const dtype = SUPPORTED_DTYPES.has(requested) ? requested : "q4f16";

  try {
    if (type === "preload") {
      // Triggered when the user picks a dtype in the UI -- start the
      // download/compile so that hitting Generate is instant.
      self.postMessage({ status: "loading", dtype });
      await Promise.all([getTokenizer(), getSnac(), getLM(dtype)]);
      self.postMessage({ status: "ready", dtype });
      return;
    }

    self.postMessage({ status: "loading", dtype });
    const [tokenizer, lm] = await Promise.all([getTokenizer(), getLM(dtype)]);
    await getSnac(); // warm

    const promptIds = buildPrompt(tokenizer, text, speaker_id);
    const inputIds = new Tensor(
      "int64",
      BigInt64Array.from(promptIds.map(BigInt)),
      [1, promptIds.length],
    );

    const sampling = samplingFor(dtype);
    const out = await lm.generate({
      inputs: inputIds,
      max_new_tokens: 2048,
      do_sample: true,
      ...sampling,
      // rep penalty disabled: transformers.js applies it across ALL prior
      // tokens (no context window), which progressively suppresses
      // naturally-recurring audio codes -> clip gets quieter over time.
      repetition_penalty: 1.0,
      eos_token_id: EOS,
    });

    const allIds = Array.from(out.data, (x) => Number(x));
    const audioIds = extractAudioTokens(allIds);
    if (audioIds.length === 0) {
      throw new Error("LM produced no audio tokens; try again or adjust sampling.");
    }
    const pcm = await decodeSnacAll(audioIds);
    const wav = pcmFloat32ToWav(pcm, SAMPLE_RATE);
    const blob = new Blob([wav], { type: "audio/wav" });
    self.postMessage({
      status: "complete",
      audio: URL.createObjectURL(blob),
      text,
      dtype,
    });
  } catch (err) {
    self.postMessage({ status: "error", data: String(err) });
    console.error(err);
  }
});
