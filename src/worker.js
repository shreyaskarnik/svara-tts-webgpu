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

// --- WebGPU feature detection -----------------------------------------------
let fp16_supported = false;
try {
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("WebGPU is not supported (no adapter found)");
  fp16_supported = adapter.features.has("shader-f16");
  self.postMessage({ status: "feature-success" });
} catch (e) {
  self.postMessage({ status: "feature-error", data: e.toString() });
  throw e;
}

// --- Constants matching mlx_audio/tts/models/llama/llama.py -----------------
const SOH = 128259, EOH = 128260, EOT = 128009;
const SOAI = 128261, SOS = 128257, EOS = 128258;
const AUDIO_OFFSET = 128266; // first audio code token ID
const SAMPLE_RATE = 24000;

// --- Load LM (Svara) and SNAC decoder ---------------------------------------
const SVARA_REPO = "shreyask/svara-tts-v1-ONNX"; // move to mlx-community when transferred
const SNAC_REPO  = "onnx-community/snac_24khz-ONNX";
const dtype = fp16_supported ? "q4f16" : "q4";

const tokenizer = await AutoTokenizer.from_pretrained(SVARA_REPO);
const lm = await AutoModelForCausalLM.from_pretrained(SVARA_REPO, {
  dtype,
  device: "webgpu",
  // The .onnx graph references model_q4f16.onnx_data (~1.95 GB external
  // weights). Without this flag transformers.js fetches the graph but
  // never mounts the data file -> "Module.MountedFiles is not available".
  use_external_data_format: true,
});

// SNAC decoder is small (~26 MB at fp16), runs cleanly on WebGPU
const snacUrl = `https://huggingface.co/${SNAC_REPO}/resolve/main/onnx/decoder_model${fp16_supported ? "_fp16" : ""}.onnx`;
const snacSession = await ort.InferenceSession.create(snacUrl, {
  executionProviders: ["webgpu"],
});

self.postMessage({ status: "ready" });

// --- Token-stream → SNAC code conversion ------------------------------------
// Reference: mlx_audio/tts/models/llama/llama.py:codes_to_layers
// Each block of 7 raw audio tokens decodes into a triplet of layer tensors:
//   layer_1 [stride 8, 1 code per frame]:   [c0]
//   layer_2 [stride 4, 2 codes per frame]:  [c1, c4]
//   layer_3 [stride 2, 4 codes per frame]:  [c2, c3, c5, c6]
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
  const { l1, l2, l3, N } = codesToLayers(audioTokenIds);
  // SNAC decoder ONNX expects three int64 code tensors. Names / shapes per the
  // export schema; if the snac ONNX uses different input names (`codes_0..2`,
  // `audio_codes`, etc.) inspect snacSession.inputNames and rename here.
  const feeds = {
    [snacSession.inputNames[0]]: new ort.Tensor("int64", l1, [1, N]),
    [snacSession.inputNames[1]]: new ort.Tensor("int64", l2, [1, N * 2]),
    [snacSession.inputNames[2]]: new ort.Tensor("int64", l3, [1, N * 4]),
  };
  const out = await snacSession.run(feeds);
  // SNAC decoder emits float32 PCM at 24 kHz, shape [1, 1, samples] or [1, samples]
  const pcm = out[snacSession.outputNames[0]];
  const flat = pcm.data; // Float32Array
  // Strip leading singleton dims
  return flat;
}

// --- Build Svara prompt -----------------------------------------------------
//   BOS, SOH, "Voice (Gender): text", EOT, EOH, SOAI, SOS
function buildPrompt(text, voice) {
  const body = tokenizer.encode(`${voice}: ${text}`, {
    add_special_tokens: false,
  });
  return [
    tokenizer.bos_token_id,
    SOH,
    ...body,
    EOT,
    EOH,
    SOAI,
    SOS,
  ];
}

// --- Strip non-audio tokens from the LM output ------------------------------
// LM emits: ...prompt..., SOS, <7 audio bands>×N, EOS.
// We slice [SOS_index+1, EOS_index) and keep only IDs in [AUDIO_OFFSET, AUDIO_OFFSET + 7*4096).
function extractAudioTokens(allTokenIds) {
  // Find last SOS occurrence
  let sosIdx = -1;
  for (let i = allTokenIds.length - 1; i >= 0; i--) {
    if (allTokenIds[i] === SOS) { sosIdx = i; break; }
  }
  let start = sosIdx + 1;
  let end = allTokenIds.length;
  // Trim EOS if present
  if (end > start && allTokenIds[end - 1] === EOS) end -= 1;
  const audio = [];
  for (let i = start; i < end; i++) {
    const t = allTokenIds[i];
    if (t >= AUDIO_OFFSET && t < AUDIO_OFFSET + 7 * 4096) audio.push(t);
  }
  // Trim to a multiple of 7
  const M = audio.length - (audio.length % 7);
  return audio.slice(0, M);
}

// --- WAV encoder (24 kHz, mono, PCM16) --------------------------------------
function pcmFloat32ToWav(samples, sampleRate) {
  const bufLen = 44 + samples.length * 2;
  const buf = new ArrayBuffer(bufLen);
  const v = new DataView(buf);
  // RIFF header
  let p = 0;
  const w = (s) => { for (let i = 0; i < s.length; i++) v.setUint8(p++, s.charCodeAt(i)); };
  w("RIFF");
  v.setUint32(p, 36 + samples.length * 2, true); p += 4;
  w("WAVEfmt ");
  v.setUint32(p, 16, true); p += 4;            // PCM chunk size
  v.setUint16(p, 1, true); p += 2;             // format = PCM
  v.setUint16(p, 1, true); p += 2;             // channels
  v.setUint32(p, sampleRate, true); p += 4;
  v.setUint32(p, sampleRate * 2, true); p += 4; // byte rate (mono, 16-bit)
  v.setUint16(p, 2, true); p += 2;             // block align
  v.setUint16(p, 16, true); p += 2;            // bits per sample
  w("data");
  v.setUint32(p, samples.length * 2, true); p += 4;
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    v.setInt16(p, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    p += 2;
  }
  return buf;
}

// --- Generate ---------------------------------------------------------------
self.addEventListener("message", async (e) => {
  const { text, speaker_id } = e.data; // speaker_id is the voice string e.g. "Hindi (Female)"
  try {
    const promptIds = buildPrompt(text, speaker_id);
    const inputIds = new Tensor(
      "int64",
      BigInt64Array.from(promptIds.map(BigInt)),
      [1, promptIds.length],
    );

    const out = await lm.generate({
      inputs: inputIds,
      max_new_tokens: 1200,
      do_sample: true,
      temperature: 0.75,
      top_p: 0.9,
      top_k: 40,
      repetition_penalty: 1.1,
      eos_token_id: EOS,
    });
    // out is a Tensor of shape [1, prompt_len + generated_len]
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
    });
  } catch (err) {
    self.postMessage({ status: "error", data: String(err) });
    console.error(err);
  }
});
