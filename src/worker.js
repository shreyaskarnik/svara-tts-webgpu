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
// Diagnostic: surface the actual input names so we can confirm our
// positional [layer_1, layer_2, layer_3] feed matches what ORT expects.
console.log("[svara] SNAC inputs:", snacSession.inputNames);
console.log("[svara] SNAC outputs:", snacSession.outputNames);

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
// Match mlx_audio/tts/models/llama/llama.py:prepare_input_ids exactly:
//   [SOH, BOS, "<voice>: <text>" tokens, EOT, EOH]
// SOH comes before BOS (Orpheus quirk -- the reference relies on the HF
// tokenizer auto-prepending BOS inside its input_ids, then prepends SOH).
// We deliberately do NOT append SOAI/SOS here; the LM predicts those itself
// before emitting audio tokens.
function buildPrompt(text, voice) {
  const body = tokenizer.encode(`${voice}: ${text}`, {
    add_special_tokens: false,
  });
  return [SOH, tokenizer.bos_token_id, ...body, EOT, EOH];
}

// --- Strip non-audio tokens from the LM output ------------------------------
// Match mlx_audio/tts/models/llama/llama.py:parse_output exactly:
//   1) Slice after the LAST SOS (=128257) the LM emitted (model predicts it)
//   2) Strip EOS (=128258) anywhere in the slice -- DO NOT range-filter,
//      since dropping a non-audio token shifts the (position % 7) band index
//      of every subsequent code -> garbled audio mid-clip.
//   3) Trim to a multiple of 7.
function extractAudioTokens(allTokenIds) {
  let sosIdx = -1;
  for (let i = allTokenIds.length - 1; i >= 0; i--) {
    if (allTokenIds[i] === SOS) { sosIdx = i; break; }
  }
  const audio = [];
  for (let i = sosIdx + 1; i < allTokenIds.length; i++) {
    const t = allTokenIds[i];
    if (t !== EOS) audio.push(t);
  }
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

    console.log(`[svara] prompt length: ${promptIds.length}, calling generate...`);
    const out = await lm.generate({
      inputs: inputIds,
      max_new_tokens: 2048,
      do_sample: true,
      temperature: 0.75,
      top_p: 0.9,
      top_k: 40,
      // NB: rep penalty intentionally disabled. transformers.js applies it
      // across ALL prior tokens (no context window), which for an audio
      // token stream progressively penalises naturally-recurring codes
      // (silence, voiced-region) and yields a clip that gets quieter and
      // less articulate over time. The MLX reference uses a 20-token
      // context window so it doesn't see this drift; the upstream Svara
      // recipe uses 1.1 with vLLM whose impl differs. 1.0 == off.
      repetition_penalty: 1.0,
      eos_token_id: EOS,
    });
    // out is a Tensor of shape [1, prompt_len + generated_len]
    const allIds = Array.from(out.data, (x) => Number(x));
    console.log(
      `[svara] LM output: total ${allIds.length} ids (${allIds.length - promptIds.length} generated). first 8: [${allIds.slice(0, 8)}], last 8: [${allIds.slice(-8)}]`,
    );
    const audioIds = extractAudioTokens(allIds);
    if (audioIds.length === 0) {
      throw new Error("LM produced no audio tokens; try again or adjust sampling.");
    }

    // Diagnostic: how many tokens fell outside their expected band?
    // For a clean stream every position i should have:
    //   AUDIO_OFFSET + (i % 7) * 4096 <= token < AUDIO_OFFSET + (i % 7 + 1) * 4096
    // Mid-clip muffling usually correlates with a cluster of bad bands.
    let badBands = 0;
    const badPositions = [];
    for (let i = 0; i < audioIds.length; i++) {
      const band = i % 7;
      const lo = AUDIO_OFFSET + band * 4096;
      const hi = lo + 4096;
      if (audioIds[i] < lo || audioIds[i] >= hi) {
        badBands++;
        if (badPositions.length < 10) badPositions.push({ i, band, token: audioIds[i] });
      }
    }
    console.log(
      `[svara] generated ${allIds.length} total ids, ${audioIds.length} audio (${audioIds.length / 7} frames). bad-band count: ${badBands}`,
    );
    if (badBands > 0) console.log("[svara] first bad bands:", badPositions);

    const pcm = await decodeSnacAll(audioIds);
    let peak = 0, sumsq = 0;
    for (let i = 0; i < pcm.length; i++) {
      const v = Math.abs(pcm[i]);
      if (v > peak) peak = v;
      sumsq += pcm[i] * pcm[i];
    }
    const rms = Math.sqrt(sumsq / pcm.length);
    console.log(
      `[svara] decoded PCM: ${pcm.length} samples (${(pcm.length / SAMPLE_RATE).toFixed(2)}s), peak=${peak.toFixed(4)}, rms=${rms.toFixed(4)}`,
    );
    if (peak < 0.01) {
      console.warn("[svara] PCM is essentially silent -- likely an LM degenerate state. See last-8 token IDs above.");
    }
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
