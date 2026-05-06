#!/usr/bin/env python3

import argparse
import math
import wave
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer, LogitsProcessor, LogitsProcessorList


EOT = 128009
SOS = 128257
EOS = 128258
SOH = 128259
EOH = 128260
SOAI = 128261
AUDIO_OFFSET = 128266
AUDIO_END = AUDIO_OFFSET + 7 * 4096
WINDOW_FRAMES = 4
WINDOW_AUDIO_START = 2048
WINDOW_AUDIO_END = 4096
SAMPLE_RATE = 24000


class SvaraLogitsProcessor(LogitsProcessor):
    def __init__(self, prompt_length: int) -> None:
        self.prompt_length = prompt_length

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        for row in range(scores.shape[0]):
            step = input_ids[row].shape[0] - self.prompt_length

            if step == 0:
                scores[row].fill_(-float("inf"))
                scores[row, SOAI] = 0
                continue

            if step == 1:
                scores[row].fill_(-float("inf"))
                scores[row, SOS] = 0
                continue

            eos_logit = scores[row, EOS].item()
            scores[row, :AUDIO_OFFSET] = -float("inf")
            scores[row, AUDIO_END:] = -float("inf")
            scores[row, EOS] = eos_logit

        return scores


def build_prompt(tokenizer: AutoTokenizer, text: str, voice: str) -> list[int]:
    body = tokenizer.encode(f"{voice}: {text}", add_special_tokens=False)
    return [SOH, tokenizer.bos_token_id, *body, EOT, EOH]


def extract_audio_tokens(all_token_ids: list[int], prompt_length: int) -> list[int]:
    try:
        sos_idx = next(i for i in range(prompt_length, len(all_token_ids)) if all_token_ids[i] == SOS)
    except StopIteration:
        return []

    audio = []
    for token_id in all_token_ids[sos_idx + 1 :]:
        if token_id == EOS:
            break
        if AUDIO_OFFSET <= token_id < AUDIO_END:
            audio.append(token_id)

    return audio[: len(audio) - (len(audio) % 7)]


def codes_to_layers(audio_token_ids: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    n = len(audio_token_ids) // 7
    layer_1 = np.zeros((1, n), dtype=np.int64)
    layer_2 = np.zeros((1, n * 2), dtype=np.int64)
    layer_3 = np.zeros((1, n * 4), dtype=np.int64)

    for i in range(n):
        base = i * 7
        layer_1[0, i] = audio_token_ids[base] - AUDIO_OFFSET
        layer_2[0, 2 * i] = audio_token_ids[base + 1] - AUDIO_OFFSET - 1 * 4096
        layer_3[0, 4 * i] = audio_token_ids[base + 2] - AUDIO_OFFSET - 2 * 4096
        layer_3[0, 4 * i + 1] = audio_token_ids[base + 3] - AUDIO_OFFSET - 3 * 4096
        layer_2[0, 2 * i + 1] = audio_token_ids[base + 4] - AUDIO_OFFSET - 4 * 4096
        layer_3[0, 4 * i + 2] = audio_token_ids[base + 5] - AUDIO_OFFSET - 5 * 4096
        layer_3[0, 4 * i + 3] = audio_token_ids[base + 6] - AUDIO_OFFSET - 6 * 4096

    return layer_1, layer_2, layer_3, n


def decode_snac_window(session: ort.InferenceSession, audio_token_ids: list[int]) -> np.ndarray:
    layer_1, layer_2, layer_3, n = codes_to_layers(audio_token_ids)
    outputs = session.run(
        None,
        {
            session.get_inputs()[0].name: layer_1,
            session.get_inputs()[1].name: layer_2,
            session.get_inputs()[2].name: layer_3,
        },
    )
    return outputs[0].reshape(-1).astype(np.float32, copy=False)


def decode_snac_stable(session: ort.InferenceSession, audio_token_ids: list[int]) -> np.ndarray:
    num_frames = len(audio_token_ids) // 7
    if num_frames == 0:
        return np.zeros(0, dtype=np.float32)

    if num_frames < WINDOW_FRAMES:
        return decode_snac_window(session, audio_token_ids)

    chunks = []
    for start in range(0, num_frames - WINDOW_FRAMES + 1):
        window_ids = audio_token_ids[start * 7 : (start + WINDOW_FRAMES) * 7]
        decoded = decode_snac_window(session, window_ids)
        chunks.append(decoded[WINDOW_AUDIO_START:WINDOW_AUDIO_END])

    return np.concatenate(chunks, axis=0)


def write_wav(path: Path, samples: np.ndarray) -> None:
    pcm = np.clip(samples, -1.0, 1.0)
    pcm16 = np.where(pcm < 0, pcm * 32768.0, pcm * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm16.tobytes())


def audio_stats(samples: np.ndarray) -> tuple[float, float, float, float]:
    if samples.size == 0:
        return 0.0, 0.0, -float("inf"), -float("inf")

    peak = float(np.max(np.abs(samples)))
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    peak_db = 20.0 * math.log10(max(peak, 1e-12))
    rms_db = 20.0 * math.log10(max(rms, 1e-12))
    return peak, rms, peak_db, rms_db


def generation_kwargs(dtype: str) -> dict:
    if dtype == "q8":
        return {"do_sample": True, "temperature": 0.35, "min_new_tokens": 30}
    return {"do_sample": False, "min_new_tokens": 30}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=".hf-models/svara-tts-v1-ONNX")
    parser.add_argument("--snac-dir", default=".hf-models/snac_24khz-ONNX/onnx")
    parser.add_argument("--dtype", choices=["q4f16", "q8"], default="q4f16")
    parser.add_argument("--provider", default="CPUExecutionProvider")
    parser.add_argument("--voice", default="Hindi (Female)")
    parser.add_argument("--text", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--fix-mistral-regex", action="store_true")
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    snac_dir = Path(args.snac_dir)
    model_file = "model_q4f16.onnx" if args.dtype == "q4f16" else "model_quantized.onnx"

    print(f"loading tokenizer from {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
        fix_mistral_regex=args.fix_mistral_regex,
    )

    print(f"loading model {model_file} on {args.provider}")
    model = ORTModelForCausalLM.from_pretrained(
        model_dir,
        subfolder="onnx",
        file_name=model_file,
        provider=args.provider,
        use_io_binding=False,
        local_files_only=True,
    )

    decoder_file = "decoder_model.onnx"
    print(f"loading snac decoder {decoder_file}")
    snac = ort.InferenceSession(
        str(snac_dir / decoder_file),
        providers=[args.provider, "CPUExecutionProvider"],
    )

    prompt_ids = build_prompt(tokenizer, args.text, args.voice)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long)
    logits_processor = LogitsProcessorList([SvaraLogitsProcessor(len(prompt_ids))])

    print(f"prompt_length={len(prompt_ids)} max_new_tokens={args.max_new_tokens}")
    print(f'prompt={args.voice}: {args.text}')

    gen_kwargs = generation_kwargs(args.dtype)
    if args.do_sample:
        gen_kwargs["do_sample"] = True
    if args.temperature is not None:
        gen_kwargs["temperature"] = args.temperature
    if args.top_k is not None:
        gen_kwargs["top_k"] = args.top_k
    if args.top_p is not None:
        gen_kwargs["top_p"] = args.top_p

    output = model.generate(
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        logits_processor=logits_processor,
        repetition_penalty=1.0,
        eos_token_id=EOS,
        **gen_kwargs,
    )

    if isinstance(output, torch.Tensor):
        all_ids = output[0].tolist()
    else:
        all_ids = output.sequences[0].tolist()

    audio_ids = extract_audio_tokens(all_ids, len(prompt_ids))
    print(f"total_tokens={len(all_ids)} audio_tokens={len(audio_ids)} frames={len(audio_ids) // 7}")
    if not audio_ids:
        raise RuntimeError("no audio tokens produced")

    pcm = decode_snac_stable(snac, audio_ids)
    peak, rms, peak_db, rms_db = audio_stats(pcm)
    print(
        "samples="
        f"{pcm.size} duration_s={pcm.size / SAMPLE_RATE:.3f} "
        f"peak={peak:.6f} peak_db={peak_db:.2f} rms={rms:.6f} rms_db={rms_db:.2f}"
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_wav(out_path, pcm)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
