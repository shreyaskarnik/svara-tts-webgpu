import React, { useRef, useState, useEffect } from "react";
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

const VOICES = LANGUAGES.flatMap(([lang]) => [
  `${lang} (Female)`,
  `${lang} (Male)`,
]);

export default function App() {
  const worker = useRef(null);

  const [selectedVoice, setSelectedVoice] = useState("Hindi (Female)");
  const [inputText, setInputText] = useState(LANGUAGES[0][1]);

  const [status, setStatus] = useState(null);
  const [error, setError] = useState(null);
  const [loadingMessage, setLoadingMessage] = useState(
    "Detecting WebGPU support...",
  );
  const [results, setResults] = useState([]);

  useEffect(() => {
    worker.current ??= new Worker(new URL("./worker.js", import.meta.url), {
      type: "module",
    });

    const onMessageReceived = (e) => {
      switch (e.data.status) {
        case "feature-success":
          setLoadingMessage(
            "Loading Svara LM (~2.6 GB) and SNAC codec — only downloaded once...",
          );
          break;
        case "feature-error":
          setError(e.data.data);
          break;
        case "ready":
          setStatus("ready");
          break;
        case "complete":
          setResults((prev) => [
            { text: e.data.text, src: e.data.audio, voice: selectedVoice },
            ...prev,
          ]);
          setStatus("ready");
          break;
        case "error":
          setError(e.data.data);
          setStatus("ready");
          break;
      }
    };

    worker.current.addEventListener("message", onMessageReceived);
    worker.current.addEventListener("error", (e) => console.error(e));

    return () => {
      worker.current.removeEventListener("message", onMessageReceived);
    };
  }, [selectedVoice]);

  const handleSubmit = (e) => {
    e.preventDefault();
    setStatus("running");
    setError(null);
    worker.current.postMessage({
      type: "generate",
      text: inputText.trim(),
      speaker_id: selectedVoice,
    });
  };

  const onLanguageChange = (lang) => {
    const sample = LANGUAGES.find(([l]) => l === lang)?.[1] ?? inputText;
    setInputText(sample);
    setSelectedVoice(`${lang} (Female)`);
  };

  return (
    <div className="relative w-full min-h-screen bg-gradient-to-br from-gray-900 to-gray-700 flex flex-col items-center justify-center p-4 overflow-hidden font-sans">
      <motion.div
        initial={{ opacity: 1 }}
        animate={{ opacity: status === null ? 1 : 0 }}
        transition={{ duration: 0.5 }}
        className="absolute w-screen h-screen justify-center flex flex-col items-center z-10 bg-gray-800/95 backdrop-blur-md"
        style={{ pointerEvents: status === null ? "auto" : "none" }}
      >
        <div className="w-[250px] h-[250px] border-4 border-white shadow-[0_0_0_5px_#4973ff] rounded-full overflow-hidden">
          <div className="loading-wave"></div>
        </div>
        <p
          className={`text-3xl my-5 text-center px-6 ${error ? "text-red-500" : "text-white"}`}
        >
          {error ?? loadingMessage}
        </p>
      </motion.div>

      <div className="max-w-3xl w-full space-y-8 relative z-[2]">
        <div className="text-center">
          <h1 className="text-5xl font-extrabold text-gray-100 mb-2 drop-shadow-lg">
            Svara TTS · WebGPU
          </h1>
          <p className="text-2xl text-gray-300 font-semibold">
            Multilingual Indian-language TTS, 100% in your browser. Powered by{" "}
            <a
              href="https://huggingface.co/kenpath/svara-tts-v1"
              target="_blank"
              rel="noreferrer"
              className="underline"
            >
              Svara
            </a>
            ,{" "}
            <a
              href="https://huggingface.co/hubertsiuzdak/snac_24khz"
              target="_blank"
              rel="noreferrer"
              className="underline"
            >
              SNAC
            </a>
            , and{" "}
            <a
              href="https://huggingface.co/docs/transformers.js"
              target="_blank"
              rel="noreferrer"
              className="underline"
            >
              Transformers.js v4
            </a>
            .
          </p>
        </div>

        <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-lg p-6">
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="flex gap-3">
              <select
                value={selectedVoice.split(" (")[0]}
                onChange={(e) => onLanguageChange(e.target.value)}
                className="flex-1 bg-gray-700/50 border-2 border-gray-600 rounded-xl text-gray-100 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                {LANGUAGES.map(([lang]) => (
                  <option key={lang} value={lang}>
                    {lang}
                  </option>
                ))}
              </select>
              <select
                value={selectedVoice}
                onChange={(e) => setSelectedVoice(e.target.value)}
                className="bg-gray-700/50 border-2 border-gray-600 rounded-xl text-gray-100 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                {VOICES.filter((v) => v.startsWith(selectedVoice.split(" (")[0])).map(
                  (v) => (
                    <option key={v} value={v}>
                      {v.split("(")[1].replace(")", "")}
                    </option>
                  ),
                )}
              </select>
            </div>
            <textarea
              placeholder="Enter text in any of 19 Indian languages..."
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              className="w-full min-h-[100px] max-h-[300px] bg-gray-700/50 border-2 border-gray-600 rounded-xl resize-y text-gray-100 placeholder-gray-400 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
              rows={Math.min(8, inputText.split("\n").length)}
            />
            <button
              type="submit"
              className="w-full inline-flex justify-center items-center px-6 py-2 text-lg font-semibold bg-gradient-to-t from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 transition-colors duration-300 rounded-xl text-white disabled:opacity-50"
              disabled={status === "running" || inputText.trim() === ""}
            >
              {status === "running" ? "Generating..." : "Generate"}
            </button>
          </form>
        </div>

        {results.length > 0 && (
          <motion.div
            initial={{ y: 50, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ duration: 0.5 }}
            className="max-h-[300px] overflow-y-auto px-2 mt-4 space-y-6 relative z-[2]"
          >
            {results.map((result, i) => (
              <div key={i}>
                <div className="text-white bg-gray-800/70 backdrop-blur-sm border border-gray-700 rounded-lg p-4 z-10">
                  <span className="absolute right-5 font-bold text-gray-400">
                    {result.voice}
                  </span>
                  <p className="mb-3 max-w-[80%]">{result.text}</p>
                  <audio controls src={result.src} className="w-full">
                    Your browser does not support the audio element.
                  </audio>
                </div>
              </div>
            ))}
          </motion.div>
        )}
      </div>

      <div className="bg-[#015871] pointer-events-none absolute left-0 w-full h-[5%] bottom-[-50px]">
        <div className="wave"></div>
        <div className="wave"></div>
      </div>
    </div>
  );
}
