import { useEffect, useMemo, useRef, useState } from "react";
import { motion } from "motion/react";

const LANGUAGES = [
  ["Hindi", "नमस्ते, आप कैसे हैं?"],
  ["Bengali", "নমস্কার, আপনি কেমন আছেন?"],
  ["Marathi", "नमस्कार, तुम्ही कसे आहात?"],
  ["Telugu", "నమస్కారం, మీరు ఎలా ఉన్నారు?"],
  ["Kannada", "ನಮಸ್ಕಾರ, ನೀವು ಹೇಗಿದ್ದೀರಿ?"],
  ["Tamil", "வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?"],
  ["Malayalam", "നമസ്കാരം, സുഖമാണോ?"],
  ["Gujarati", "નમસ્તે, તમે કેમ છો?"],
  ["Punjabi", "ਸਤ ਸ੍ਰੀ ਅਕਾਲ, ਤੁਸੀਂ ਕਿਵੇਂ ਹੋ?"],
  ["Assamese", "নমস্কাৰ, আপুনি কেনে আছে?"],
  ["Bhojpuri", "नमस्कार, राउर का हाल बा?"],
  ["Magahi", "नमस्कार, तू कैसन हे?"],
  ["Maithili", "नमस्कार, अहाँ कोना छी?"],
  ["Chhattisgarhi", "नमस्कार, आप कइसन हन?"],
  ["Bodo", "नमस्कार, नोँ बेसेबा डंनो?"],
  ["Dogri", "नमस्ते, तुसें कि’यां ओ?"],
  ["Nepali", "नमस्ते, तपाईं कस्तो हुनुहुन्छ?"],
  ["Sanskrit", "नमस्कारः, भवान् कथमस्ति?"],
  ["English (Indian)", "Hello, how are you?"],
];

const LANGUAGE_DETAILS = {
  Hindi: { script: "Devanagari", region: "North India" },
  Bengali: { script: "Bengali", region: "Eastern India" },
  Marathi: { script: "Devanagari", region: "Maharashtra" },
  Telugu: { script: "Telugu", region: "Andhra Pradesh + Telangana" },
  Kannada: { script: "Kannada", region: "Karnataka" },
  Tamil: { script: "Tamil", region: "Tamil Nadu" },
  Malayalam: { script: "Malayalam", region: "Kerala" },
  Gujarati: { script: "Gujarati", region: "Gujarat" },
  Punjabi: { script: "Gurmukhi", region: "Punjab" },
  Assamese: { script: "Assamese", region: "Assam" },
  Bhojpuri: { script: "Devanagari", region: "Bihar + Eastern UP" },
  Magahi: { script: "Devanagari", region: "Bihar" },
  Maithili: { script: "Devanagari", region: "Mithila" },
  Chhattisgarhi: { script: "Devanagari", region: "Chhattisgarh" },
  Bodo: { script: "Devanagari", region: "Northeast India" },
  Dogri: { script: "Devanagari", region: "Jammu" },
  Nepali: { script: "Devanagari", region: "Nepal + India" },
  Sanskrit: { script: "Devanagari", region: "Classical Indic" },
  "English (Indian)": { script: "Latin", region: "Indian English" },
};

const VOICES = LANGUAGES.flatMap(([lang]) => [
  `${lang} (Female)`,
  `${lang} (Male)`,
]);

const DTYPES = [
  { value: "q4f16", label: "q4f16", note: "~1.95 GB · fastest cold start" },
  { value: "q8", label: "q8", note: "~4.32 GB · cleaner, slower preload" },
];

const STACK_FACTS = [
  { label: "Model", value: "Svāra-TTS v1" },
  { label: "Codec", value: "SNAC 24 kHz" },
  { label: "Runtime", value: "WebGPU + Transformers.js" },
];

function withEmotionTag(text, tag) {
  return `${text.replace(/\s*<[^>]+>\s*$/u, "").trim()} ${tag}`;
}

export default function App() {
  const worker = useRef(null);
  const runtimeReadyRef = useRef(false);
  const loadedDtypesRef = useRef([]);

  const [selectedVoice, setSelectedVoice] = useState("Hindi (Female)");
  const [inputText, setInputText] = useState(LANGUAGES[0][1]);
  const [dtype, setDtype] = useState("q4f16");
  const [status, setStatus] = useState(null);
  const [error, setError] = useState(null);
  const [runtimeReady, setRuntimeReady] = useState(false);
  const [loadingDtype, setLoadingDtype] = useState(null);
  const [loadedDtypes, setLoadedDtypes] = useState([]);
  const [loadingMessage, setLoadingMessage] = useState(
    "Detecting WebGPU support...",
  );
  const [results, setResults] = useState([]);

  const selectedLanguage = selectedVoice.split(" (")[0];
  const selectedGender = selectedVoice.includes("(Male)") ? "Male" : "Female";
  const languageDetail = LANGUAGE_DETAILS[selectedLanguage] ?? {
    script: "Indic",
    region: "South Asia",
  };
  const currentSample =
    LANGUAGES.find(([lang]) => lang === selectedLanguage)?.[1] ?? inputText;
  const currentDtype = DTYPES.find((entry) => entry.value === dtype) ?? DTYPES[0];
  const isCurrentDtypeLoaded = loadedDtypes.includes(dtype);
  const isLoadingCurrentDtype =
    status === "loading" && loadingDtype === currentDtype.value;

  const promptChips = useMemo(
    () => [
      { label: "Sample line", value: currentSample },
      { label: "Sample + <sad>", value: withEmotionTag(currentSample, "<sad>") },
      {
        label: "Sample + <happy>",
        value: withEmotionTag(currentSample, "<happy>"),
      },
    ],
    [currentSample],
  );

  useEffect(() => {
    runtimeReadyRef.current = runtimeReady;
  }, [runtimeReady]);

  useEffect(() => {
    loadedDtypesRef.current = loadedDtypes;
  }, [loadedDtypes]);

  useEffect(() => {
    worker.current ??= new Worker(new URL("./worker.js", import.meta.url), {
      type: "module",
    });

    const onMessageReceived = (e) => {
      switch (e.data.status) {
        case "feature-success":
          runtimeReadyRef.current = true;
          setRuntimeReady(true);
          setError(null);
          setStatus("idle");
          setLoadingMessage(
            "WebGPU is available. Load a model when you want to start local synthesis.",
          );
          break;
        case "feature-error":
          setError(e.data.data);
          break;
        case "loading":
          setError(null);
          if (loadedDtypesRef.current.includes(e.data.dtype)) {
            setLoadingDtype(null);
            setStatus("running");
          } else {
            setLoadingDtype(e.data.dtype);
            setLoadingMessage(
              e.data.dtype === "q8"
                ? "Loading q8 weights (~4.32 GB, sharded). First run can take a minute..."
                : "Loading q4f16 weights (~1.95 GB). First run downloads once, then stays cached...",
            );
            setStatus("loading");
          }
          break;
        case "ready":
          setLoadingDtype(null);
          setError(null);
          setLoadedDtypes((prev) => {
            if (prev.includes(e.data.dtype)) return prev;
            const next = [...prev, e.data.dtype];
            loadedDtypesRef.current = next;
            return next;
          });
          setStatus("ready");
          break;
        case "complete":
          setResults((prev) => [
            {
              text: e.data.text,
              src: e.data.audio,
              voice: e.data.voice,
              dtype: e.data.dtype,
              createdAt: new Date().toLocaleTimeString([], {
                hour: "numeric",
                minute: "2-digit",
              }),
            },
            ...prev,
          ]);
          setError(null);
          setStatus("ready");
          break;
        case "error":
          setLoadingDtype(null);
          setError(e.data.data);
          setStatus(
            loadedDtypesRef.current.includes(e.data.dtype)
              ? "ready"
              : runtimeReadyRef.current
                ? "idle"
                : null,
          );
          break;
      }
    };

    worker.current.addEventListener("message", onMessageReceived);
    worker.current.addEventListener("error", (event) => console.error(event));

    return () => {
      worker.current.removeEventListener("message", onMessageReceived);
    };
  }, []);

  const handleSubmit = (event) => {
    event.preventDefault();
    if (!isCurrentDtypeLoaded) return;
    setStatus("running");
    setError(null);
    worker.current.postMessage({
      type: "generate",
      text: inputText.trim(),
      speaker_id: selectedVoice,
      dtype,
    });
  };

  const handleLoadModel = () => {
    if (!runtimeReady || isCurrentDtypeLoaded) return;
    setError(null);
    setLoadingDtype(dtype);
    setLoadingMessage(
      dtype === "q8"
        ? "Loading q8 weights (~4.32 GB, sharded). First run can take a minute..."
        : "Loading q4f16 weights (~1.95 GB). First run downloads once, then stays cached...",
    );
    setStatus("loading");
    worker.current?.postMessage({ type: "preload", dtype });
  };

  const onLanguageChange = (lang) => {
    const sample = LANGUAGES.find(([entry]) => entry === lang)?.[1] ?? inputText;
    setInputText(sample);
    setSelectedVoice(`${lang} (Female)`);
  };

  const onDtypeChange = (next) => {
    if (next === dtype) return;
    setDtype(next);
    setError(null);
    setLoadingDtype(null);
    setStatus(loadedDtypesRef.current.includes(next) ? "ready" : "idle");
  };

  let statusHeadline = "Checking browser runtime";
  let statusBody = loadingMessage;

  if (error) {
    statusHeadline = runtimeReady ? "Load issue" : "Startup issue";
    statusBody = error;
  } else if (status === "running") {
    statusHeadline = "Rendering speech locally";
    statusBody = `Synthesizing with ${selectedVoice} on ${currentDtype.label}.`;
  } else if (status === "loading") {
    statusHeadline = "Loading model weights";
  } else if (runtimeReady && !isCurrentDtypeLoaded) {
    statusHeadline = "Ready to load model";
    statusBody = `${currentDtype.label} is a one-time ${currentDtype.note.split("·")[0].trim()} download. Tap Load model to cache it in this browser.`;
  } else if (isCurrentDtypeLoaded) {
    statusHeadline = "Model ready in this browser";
    statusBody = `${selectedVoice} is ready on ${currentDtype.label}. Everything runs locally after the one-time model download.`;
  }

  const statusActivityLabel = status === "running"
    ? "Generating audio..."
    : status === "loading"
      ? "Loading in the background"
      : null;
  const statusCardBusy = !error && (
    status === "loading" || status === "running" || status === null
  );
  const loadButtonLabel = isLoadingCurrentDtype
    ? `Loading ${currentDtype.label}...`
    : `Load ${currentDtype.label}`;

  return (
    <div className="app-shell">
      <div className="ornament ornament-top" aria-hidden="true">
        <img src="/warli-strip.svg" alt="" />
      </div>

      <main className="app-main">
        <header className="hero">
          <span className="hero-kicker">Svāra TTS · WebGPU</span>
          <h1 className="hero-title">Svāra</h1>
          <span className="hero-subline">स्वरा · Indic text-to-speech in the browser</span>
          <p className="hero-copy">
            A warmer frontend for the same local synthesis engine: 19 languages,
            38 voices, SNAC decoding, and no server round-trip once the model is
            cached in this browser.
          </p>
          <div className="hero-links">
            <a
              href="https://huggingface.co/kenpath/svara-tts-v1"
              target="_blank"
              rel="noreferrer"
            >
              Base model
            </a>
            <a
              href="https://huggingface.co/shreyask/svara-tts-v1-ONNX"
              target="_blank"
              rel="noreferrer"
            >
              ONNX export
            </a>
            <a
              href="https://huggingface.co/onnx-community/snac_24khz-ONNX"
              target="_blank"
              rel="noreferrer"
            >
              SNAC codec
            </a>
          </div>
        </header>

        <section
          className={`card status-card ${statusCardBusy ? "is-busy" : ""}`}
        >
          <div className="status-main">
            <p className="section-kicker">Session</p>
            <h2>{statusHeadline}</h2>
            <p className={`status-copy ${error ? "is-error" : ""}`}>
              {statusBody}
            </p>
            {statusActivityLabel && !error && (
              <div className="inline-loader" aria-hidden="true">
                <span className="inline-loader-dot"></span>
                <span className="inline-loader-label">{statusActivityLabel}</span>
              </div>
            )}
            {runtimeReady && !isCurrentDtypeLoaded && status !== "loading" && (
              <div className="model-gate">
                <div>
                  <p className="model-gate-copy">
                    Model load is explicit in this build.
                  </p>
                  <span className="model-gate-sub">
                    {loadedDtypes.length > 0
                      ? `Cached here: ${loadedDtypes.join(", ")}`
                      : "Nothing cached in this browser session yet."}
                  </span>
                </div>
                <button
                  type="button"
                  className="primary-button load-button"
                  onClick={handleLoadModel}
                  disabled={!runtimeReady || status === "running"}
                >
                  {loadButtonLabel}
                </button>
              </div>
            )}
          </div>
          <div className="pill-row">
            <span className="pill">19 languages</span>
            <span className="pill">38 voices</span>
            <span className="pill">24 kHz mono</span>
            <span className="pill">Runs locally</span>
          </div>
        </section>

        <div className="workspace">
          <section className="card composer-card">
            <div className="card-header">
              <div>
                <p className="section-kicker">Compose</p>
                <h2>Switch language, adjust voice, synthesize</h2>
              </div>
              <button
                type="button"
                className="ghost-button"
                onClick={() => setInputText(currentSample)}
              >
                Use sample
              </button>
            </div>

            <form onSubmit={handleSubmit} className="composer-form">
              <div className="control-grid">
                <label className="field">
                  <span className="field-label">Language</span>
                  <select
                    value={selectedLanguage}
                    onChange={(event) => onLanguageChange(event.target.value)}
                  >
                    {LANGUAGES.map(([lang]) => (
                      <option key={lang} value={lang}>
                        {lang}
                      </option>
                    ))}
                  </select>
                </label>

                <label className="field">
                  <span className="field-label">Voice</span>
                  <select
                    value={selectedVoice}
                    onChange={(event) => setSelectedVoice(event.target.value)}
                  >
                    {VOICES.filter((voice) => voice.startsWith(`${selectedLanguage} (`)).map(
                      (voice) => (
                        <option key={voice} value={voice}>
                          {voice.split("(")[1].replace(")", "")}
                        </option>
                      ),
                    )}
                  </select>
                </label>

                <label className="field field-wide">
                  <span className="field-label">Quantization</span>
                  <select
                    value={dtype}
                    onChange={(event) => onDtypeChange(event.target.value)}
                    disabled={status === "running" || status === "loading"}
                  >
                    {DTYPES.map((entry) => (
                      <option key={entry.value} value={entry.value}>
                        {entry.label}
                      </option>
                    ))}
                  </select>
                  <small className="field-note">{currentDtype.note}</small>
                </label>
              </div>

              <label className="field">
                <div className="label-row">
                  <span className="field-label">Prompt</span>
                  <span className="field-meta">
                    {languageDetail.script} · {languageDetail.region}
                  </span>
                </div>
                <textarea
                  placeholder="Enter text in any supported Indian language..."
                  value={inputText}
                  onChange={(event) => setInputText(event.target.value)}
                  rows={Math.min(8, Math.max(4, inputText.split("\n").length))}
                />
              </label>

              <div className="chip-bar">
                {promptChips.map((chip) => (
                  <button
                    key={chip.label}
                    type="button"
                    className="utility-chip"
                    onClick={() => setInputText(chip.value)}
                  >
                    {chip.label}
                  </button>
                ))}
              </div>

              <div className="composer-footer">
                <p className="helper-copy">
                  Emotion tags can be appended at the end of the sentence, for
                  example <code>&lt;sad&gt;</code> or <code>&lt;happy&gt;</code>.
                  Use <code>q8</code> if you want the cleanest output and can
                  afford the larger one-time download.
                </p>
                <button
                  type="submit"
                  className="primary-button"
                  disabled={
                    status !== "ready" ||
                    !isCurrentDtypeLoaded ||
                    inputText.trim() === ""
                  }
                >
                  {status === "running"
                    ? "Generating audio..."
                    : isLoadingCurrentDtype
                      ? `Loading ${currentDtype.label}...`
                      : !isCurrentDtypeLoaded
                        ? "Load model to continue"
                      : "Generate speech"}
                </button>
              </div>
            </form>
          </section>

          <aside className="sidebar">
            <section className="card inspector-card">
              <p className="section-kicker">Inspector</p>
              <h3 className="inspector-title">{selectedVoice}</h3>

              <dl className="compact-meta-grid">
                <div className="compact-meta">
                  <dt>Script</dt>
                  <dd>{languageDetail.script}</dd>
                </div>
                <div className="compact-meta">
                  <dt>Region</dt>
                  <dd>{languageDetail.region}</dd>
                </div>
                <div className="compact-meta">
                  <dt>Type</dt>
                  <dd>{selectedGender}</dd>
                </div>
                <div className="compact-meta">
                  <dt>Quant</dt>
                  <dd>{currentDtype.label}</dd>
                </div>
              </dl>

              <div className="stack-chip-list">
                {STACK_FACTS.map((fact) => (
                  <div key={fact.label} className="stack-chip">
                    <span>{fact.label}</span>
                    <strong>{fact.value}</strong>
                  </div>
                ))}
              </div>

              <details className="debug-notes">
                <summary>Usage notes</summary>
                <ul className="note-list note-list-compact">
                  <li>Model and codec are browser-cached after the first load.</li>
                  <li>Short prompts are the best way to compare voices and quant levels.</li>
                  <li>The results archive below preserves each render with the actual voice used.</li>
                </ul>
              </details>
            </section>
          </aside>
        </div>

        {results.length > 0 && (
          <section className="results-section">
            <div className="results-header">
              <div>
                <p className="section-kicker">Archive</p>
                <h2>Generated takes</h2>
              </div>
              <span className="results-meta">Newest first</span>
            </div>

            <div className="results-grid">
              {results.map((result, index) => (
                <motion.article
                  key={`${result.voice}-${result.createdAt}-${index}`}
                  initial={{ y: 24, opacity: 0 }}
                  animate={{ y: 0, opacity: 1 }}
                  transition={{ duration: 0.35, delay: index * 0.04 }}
                  className="card result-card"
                >
                  <div className="result-head">
                    <div>
                      <h3>{result.voice}</h3>
                      <p>{result.createdAt}</p>
                    </div>
                    <span className="result-pill">{result.dtype}</span>
                  </div>
                  <p className="result-text">{result.text}</p>
                  <audio controls src={result.src} className="result-audio">
                    Your browser does not support the audio element.
                  </audio>
                </motion.article>
              ))}
            </div>
          </section>
        )}
      </main>

      <div className="ornament ornament-bottom" aria-hidden="true">
        <img src="/warli-strip.svg" alt="" />
      </div>
    </div>
  );
}
