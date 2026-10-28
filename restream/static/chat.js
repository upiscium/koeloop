import { SpeechPlayer } from "/static/chat-audio.js";
import { Dictation } from "/static/dictation.js";
import { VoiceTurn } from "/static/voice-turn.js";

const $ = (id) => document.getElementById(id);
const STORAGE = "voicevox-chat-settings-v1";
const SYSTEM = "日本語で自然に会話してください。返答は音声でも読み上げます。通常は簡潔な文章で答え、不要な見出しや箇条書きは避けてください。";
const MODE_HINTS = {
  balanced: "文末を優先し、長い文は読点などで区切ります。",
  sentence: "原則として文末まで待つ比較基準です。",
  syntax: "文末が未確定で、括弧の外の読点まで24文字以上、その後ろに20文字以上届いた場合に解析します。文末が先に届けば解析0回でも正常です。",
  raw: "LLM の content 差分を再分割せず、受信単位のまま合成します。文末待ち・構文解析なし。細切れで不自然になりやすい比較用です（最大160文字の安全制限あり）。句読点の間の調整は別設定です。",
};
const length = (text) => Array.from(text).length;
let turns = [];
let current = null;
let audioContext = null;
let speakerRequest = null;
let defaultsReady = false;
let dictation = null;
let voiceTurn = null;

function refreshComposer() {
  $("send").disabled = !defaultsReady || !!current || !!dictation?.busy || !!voiceTurn?.busy;
  $("settingsFields").disabled = !defaultsReady || !!current || !!voiceTurn?.busy;
  dictation?.render();
  voiceTurn?.render();
}

function setStatus(message, error = false) {
  $("status").textContent = message;
  $("status").classList.toggle("error", error);
}
function setSidebar(open) {
  $("sidebar").hidden = !open;
  $("toggleSettings").setAttribute("aria-expanded", String(open));
}
function scrollToBottom(force = false) {
  const panel = $("conversation");
  if (force || panel.scrollHeight - panel.scrollTop - panel.clientHeight < 180) {
    panel.scrollTop = panel.scrollHeight;
  }
}
function settings() {
  return {
    llmUrl: $("llmUrl").value.trim(), model: $("model").value.trim(),
    voicevoxUrl: $("voicevoxUrl").value.trim(), speaker: Number($("speaker").value),
    reasoning: $("reasoning").value, speed: Number($("speed").value),
    mode: $("mode").value, audio: $("audioEnabled").checked,
    adjustPauses: $("adjustPauses").checked, sentenceExtra: Number($("sentenceExtra").value),
    commaScale: Number($("commaScale").value),
    voiceSilenceMs: Number($("voiceSilence").value), voicePreMs: Number($("voicePre").value),
    voicePostMs: Number($("voicePost").value), voiceThreshold: Number($("voiceThreshold").value),
    voiceBargeIn: $("voiceBargeIn").checked,
  };
}
function saveSettings() {
  try { localStorage.setItem(STORAGE, JSON.stringify(settings())); } catch (_) { /* Private mode may block storage. */ }
  $("modelLabel").textContent = $("model").value.trim() || "KoeLoop";
}
function updateModeHint() {
  $("modeHint").textContent = MODE_HINTS[$("mode").value] || "";
}
function addMessage(role, content = "") {
  const article = document.createElement("article");
  article.className = `message ${role}`;
  article.setAttribute("aria-label", role === "user" ? "あなた" : "アシスタント");
  if (role === "assistant") {
    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.textContent = "◉";
    avatar.setAttribute("aria-hidden", "true");
    article.append(avatar);
  }
  const body = document.createElement("div");
  body.className = "body";
  body.textContent = content;
  article.append(body);
  const note = document.createElement("div");
  note.className = "message-note";
  if (role === "assistant") article.append(note);
  $("messages").append(article);
  return {article, body, note};
}
function historyFor(prompt) {
  const retained = [];
  let size = length(SYSTEM) + length(prompt);
  const completed = turns.filter((turn) => turn.complete);
  for (const turn of completed.slice().reverse()) {
    const count = length(turn.user) + length(turn.assistant);
    if (size + count > 14000 || retained.length >= 20) break;
    retained.unshift(turn);
    size += count;
  }
  $("contextInfo").textContent = retained.length < completed.length
    ? `長さ上限のため、直近 ${retained.length} 往復を送信します。画面上の会話は残ります。`
    : "会話はこのページ内のみ。中断・失敗した未完了の返答は次の送信履歴に含めません。";
  return [
    {role: "system", content: SYSTEM},
    ...retained.flatMap((turn) => [{role: "user", content: turn.user}, {role: "assistant", content: turn.assistant}]),
    {role: "user", content: prompt},
  ];
}
function finish(run, message, error = false, completed = false, interrupted = false) {
  if (current !== run) return;
  current = null;
  clearTimeout(run.connectionTimer);
  run.player?.stop();
  if (run.ws && run.ws.readyState < WebSocket.CLOSING) run.ws.close();
  run.view.article.classList.remove("pending");
  if (run.warning) message += ` ${run.warning}`;
  const analysis = run.segmentation?.requested_mode === "syntax"
    ? ` / 解析 ${run.segmentation.analysis_calls} 回・${run.segmentation.analysis_ms} ms` : "";
  run.view.note.textContent = message + analysis;
  run.view.note.classList.toggle("error", error);
  run.turn.complete = !interrupted && run.textComplete && !!run.turn.assistant.trim();
  $("settingsFields").disabled = false;
  $("send").disabled = false;
  $("send").hidden = false;
  $("stop").hidden = true;
  $("prompt").disabled = false;
  refreshComposer();
  setStatus(message, error);
  if (run.fromVoice) voiceTurn?.replyFinished(run.voiceSession, completed && !error && !run.warning && run.textComplete, run.voiceReply);
  $("prompt").focus({preventScroll: true});
}
function stop() {
  voiceTurn?.cancel();
  if (!current) return;
  const run = current;
  if (run.ws?.readyState === WebSocket.OPEN) run.ws.send(JSON.stringify({type: "cancel"}));
  finish(run, run.textComplete ? "読み上げを停止しました。" : "停止しました。この未完了の返答は次の会話履歴に含めません。");
}
async function submit(event, {fromVoice = false, voiceSession = null, voiceReply = null} = {}) {
  event.preventDefault();
  if (current || !defaultsReady || dictation?.busy || (voiceTurn?.busy && !(fromVoice && voiceTurn.canSubmit(voiceSession)))) return false;
  const prompt = $("prompt").value.trim();
  if (!prompt) return false;
  const config = settings();
  try {
    for (const value of [config.llmUrl, config.voicevoxUrl]) {
      const url = new URL(value);
      if (!["http:", "https:"].includes(url.protocol) || url.username || url.password || url.search || url.hash) {
        throw new Error("接続先は認証情報やクエリを含まない HTTP(S) の URL にしてください。");
      }
    }
    if (!config.model) throw new Error("モデル名を入力してください。");
    if (!Object.hasOwn(MODE_HINTS, config.mode)) throw new Error("読み上げの区切りを選択してください。");
    if (!Number.isInteger(config.speaker) || config.speaker < 0) throw new Error("話者を選択してください。");
    if (!Number.isFinite(config.speed) || config.speed < 0.5 || config.speed > 2) throw new Error("速度は 0.5～2.0 にしてください。");
    if (config.adjustPauses && (!Number.isInteger(config.sentenceExtra) || config.sentenceExtra < 0 || config.sentenceExtra > 1000
        || !Number.isFinite(config.commaScale) || config.commaScale < 0 || config.commaScale > 2)) {
      throw new Error("句点の追加は 0～1000 ms、読点の倍率は 0～2 にしてください。");
    }
  } catch (error) { setSidebar(true); setStatus(error.message, true); return false; }
  const messages = historyFor(prompt);
  const backend = {llm_url: config.llmUrl, llm_model: config.model, voicevox_url: config.voicevoxUrl,
    reasoning_effort: config.reasoning};
  if ($("apiKey").value) backend.llm_api_key = $("apiKey").value;
  const request = JSON.stringify({type: "start", source: "llm", messages, backend,
    audio: config.audio, speaker: config.speaker, speed: config.speed, mode: config.mode,
    adjust_pauses: config.adjustPauses, sentence_pause_extra_ms: config.adjustPauses ? config.sentenceExtra : 200,
    comma_pause_scale: config.adjustPauses ? config.commaScale : 0.8});
  if (new TextEncoder().encode(request).length > 65536) {
    setStatus("会話と設定が送信上限を超えています。「新しい会話」でやり直してください。", true);
    return false;
  }
  saveSettings();
  speakerRequest?.abort();
  $("welcome").hidden = true;
  addMessage("user", prompt);
  const view = addMessage("assistant");
  view.article.classList.add("pending");
  const turn = {user: prompt, assistant: "", complete: false};
  turns.push(turn);
  const run = {turn, view, textComplete: false, started: performance.now(), fromVoice, voiceSession, voiceReply};
  current = run;
  refreshComposer();
  $("prompt").value = "";
  $("prompt").disabled = true;
  $("settingsFields").disabled = true;
  $("send").disabled = true;
  $("send").hidden = true;
  $("stop").hidden = false;
  if (matchMedia("(max-width: 760px)").matches) setSidebar(false);
  scrollToBottom(true);
  setStatus("接続中…");
  try {
    if (config.audio) {
      audioContext ||= new AudioContext();
      await audioContext.resume();
      if (current !== run) return;
      if (audioContext.state !== "running") throw new Error("音声再生を許可するか、「音声で返答する」をオフにしてください。");
      run.player = new SpeechPlayer(audioContext,
        (seq) => {
          if (current === run && run.ws?.readyState === WebSocket.OPEN) {
            run.ws.send(JSON.stringify({type: "played", seq}));
          }
        }, (message) => finish(run, message, true));
    }
    if (current !== run) return;
    const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
    run.ws = ws;
    run.connectionTimer = setTimeout(() => finish(run, "接続がタイムアウトしました。", true), 20000);
    ws.onopen = () => {
      if (current !== run) return ws.close();
      ws.send(request);
    };
    ws.onmessage = ({data}) => {
      if (current !== run) return;
      try {
        const event = JSON.parse(data);
        if (event.type === "ready") {
          clearTimeout(run.connectionTimer);
          setStatus("返答を待っています…");
        } else if (event.type === "delta") {
          const panel = $("conversation");
          const wasNearBottom = panel.scrollHeight - panel.scrollTop - panel.clientHeight < 180;
          run.turn.assistant += event.text;
          run.view.body.textContent = run.turn.assistant;
          run.view.article.classList.remove("pending");
          setStatus(config.audio ? "返答を生成・読み上げ中…" : "返答を生成中…");
          scrollToBottom(wasNearBottom);
        } else if (event.type === "audio") {
          run.player?.enqueue(event);
        } else if (event.type === "text_done") {
          run.textComplete = true;
          run.segmentation = event.segmentation;
          setStatus(config.audio ? "残りの返答を読み上げています…" : "返答を受信しました。");
        } else if (event.type === "warning") {
          run.warning = event.message;
          run.view.note.textContent = event.message;
        } else if (event.type === "done") {
          finish(run, run.turn.assistant.trim() ? "完了" : "本文が返りませんでした。推論モードやモデルを確認してください。", !run.turn.assistant.trim(), true);
        } else if (event.type === "error") {
          finish(run, event.message, true);
        } else if (event.type === "cancelled") {
          finish(run, "停止しました。");
        }
      } catch (_) { finish(run, "返答を処理できませんでした。接続先を確認してください。", true); }
    };
    ws.onerror = () => finish(run, "サーバーへの接続に失敗しました。", true);
    ws.onclose = () => {
      if (current === run) finish(run, "完了前に接続が切れました。", true);
    };
    return true;
  } catch (error) { finish(run, error.message, true); return false; }
}

async function loadSpeakers() {
  if (current) return;
  speakerRequest?.abort();
  const controller = new AbortController();
  speakerRequest = controller;
  $("loadSpeakers").disabled = true;
  $("speakerStatus").textContent = "接続しています…";
  try {
    const response = await fetch("/api/backends/speakers", {method: "POST",
      headers: {"Content-Type": "application/json"}, signal: controller.signal,
      body: JSON.stringify({voicevox_url: $("voicevoxUrl").value.trim()})});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "接続に失敗しました。");
    if (controller.signal.aborted || current) return;
    const options = [];
    for (const speaker of data) for (const style of speaker.styles) {
      if (style.type && style.type !== "talk") continue;
      const option = document.createElement("option");
      option.value = style.id;
      option.textContent = `${speaker.name} / ${style.name} (${style.id})`;
      options.push(option);
    }
    if (!options.length) throw new Error("読み上げ可能な話者が見つかりませんでした。");
    const previous = $("speaker").value;
    $("speaker").replaceChildren(...options);
    if (options.some((option) => option.value === previous)) $("speaker").value = previous;
    $("speakerStatus").textContent = `${options.length} 種類の話者・スタイルを取得しました。`;
    saveSettings();
  } catch (error) {
    if (!controller.signal.aborted) $("speakerStatus").textContent = error.message;
  } finally {
    if (speakerRequest === controller) $("loadSpeakers").disabled = false;
  }
}

$("composer").addEventListener("submit", submit);
$("prompt").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing && event.keyCode !== 229) {
    event.preventDefault();
    $("composer").requestSubmit();
  }
});
$("stop").addEventListener("click", stop);
$("newChat").addEventListener("click", () => {
  voiceTurn?.cancel();
  dictation?.reset();
  stop();
  turns = [];
  $("messages").replaceChildren();
  $("welcome").hidden = false;
  $("prompt").value = "";
  $("contextInfo").textContent = "会話はこのページ内のみ。接続設定だけをこのブラウザに保存します。";
  setStatus("新しい会話を始められます。");
  $("prompt").focus();
});
$("toggleSettings").addEventListener("click", () => setSidebar($("sidebar").hidden));
$("loadSpeakers").addEventListener("click", loadSpeakers);
$("voicevoxUrl").addEventListener("input", () => { speakerRequest?.abort(); $("speakerStatus").textContent = "接続先を変更した場合は、話者を再取得してください。"; });
$("settingsFields").addEventListener("change", saveSettings);
$("mode").addEventListener("change", updateModeHint);
function updatePauseInputs() {
  $("sentenceExtra").disabled = !$("adjustPauses").checked;
  $("commaScale").disabled = !$("adjustPauses").checked;
}
$("adjustPauses").addEventListener("change", updatePauseInputs);
for (const button of document.querySelectorAll("[data-prompt]")) button.addEventListener("click", () => {
  if (voiceTurn?.busy) return;
  $("prompt").value = button.dataset.prompt;
  $("prompt").focus();
});
setSidebar(!matchMedia("(max-width: 760px)").matches);
dictation = new Dictation({
  button: $("mic"), cancelButton: $("micCancel"), status: $("sttStatus"), draft: $("prompt"),
  pendingPanel: $("sttPendingPanel"), pending: $("sttPending"), insertButton: $("insertStt"), discardButton: $("discardStt"),
  modelSelect: $("sttBackend"),
  canStart: () => defaultsReady && !current && !voiceTurn?.busy, changed: refreshComposer,
});
voiceTurn = new VoiceTurn({button: $("voiceStart"), continuousButton: $("voiceContinuous"), cancelButton: $("voiceCancel"), status: $("voiceStatus"), draft: $("prompt"),
  silenceInput: $("voiceSilence"), preInput: $("voicePre"), postInput: $("voicePost"), thresholdInput: $("voiceThreshold"), hint: $("voiceHint"),
  bargeInput: $("voiceBargeIn"),
  canStart: () => defaultsReady && !current && !dictation?.busy, changed: refreshComposer,
  prepareAudio: async () => {
    speakerRequest?.abort();
    if ($("audioEnabled").checked) {
      audioContext ||= new AudioContext();
      await audioContext.resume();
      if (audioContext.state !== "running") throw new Error("playback blocked");
    }
  }, committed: (voiceSession, voiceReply) => submit({preventDefault() {}}, {fromVoice: true, voiceSession, voiceReply}),
  stopReply: () => { if (current?.fromVoice) stop(); },
  interruptReply: (reply) => {
    const run = current;
    if (!run || run.voiceReply !== reply) return false;
    if (run.ws?.readyState === WebSocket.OPEN) run.ws.send(JSON.stringify({type: "cancel"}));
    finish(run, "発話を検出したため返答を中断しました。この返答は次の会話履歴に含めません。", false, false, true);
    return true;
  },
});
$("send").disabled = true;
$("settingsFields").disabled = true;
try {
  let server = {};
  try {
    const response = await fetch("/api/config", {signal: AbortSignal.timeout(5000)});
    if (response.ok) server = await response.json();
  } catch (_) { /* Defaults remain editable even if discovery is unavailable. */ }
  let saved = {};
  try { saved = JSON.parse(localStorage.getItem(STORAGE)) || {}; } catch (_) { /* Ignore damaged settings. */ }
  const defaultUrl = server.llm_configured ? server.llm_url : "http://127.0.0.1:11434/v1";
  $("llmUrl").value = saved.llmUrl ?? defaultUrl ?? "";
  $("model").value = saved.model ?? (server.llm_model || "qwen3.5:4b");
  $("voicevoxUrl").value = saved.voicevoxUrl ?? server.voicevox_url ?? "http://127.0.0.1:50021";
  $("reasoning").value = saved.reasoning ?? (server.llm_configured ? server.llm_reasoning_effort || "" : "none");
  $("speed").value = saved.speed ?? 1;
  $("adjustPauses").checked = saved.adjustPauses ?? true;
  $("sentenceExtra").value = saved.sentenceExtra ?? 200;
  $("commaScale").value = saved.commaScale ?? 0.8;
  updatePauseInputs();
  $("mode").value = Object.hasOwn(MODE_HINTS, saved.mode) ? saved.mode : "balanced";
  updateModeHint();
  $("syntaxStatus").textContent = server.syntax?.installed
    ? `日本語解析器 ${server.syntax.model} 導入済み（初回に読み込み）。慎重な分割のため、文末まで待つ場合があります。`
    : "日本語解析器は未導入です。構文解析を選んだ場合は文末方式で処理し、通知します。";
  $("audioEnabled").checked = saved.audio ?? true;
  dictation.configure(server.stt, server.stt_backends);
  voiceTurn.configure(server.voice_vad, server.stt_backends, saved);
  if (Number.isInteger(saved.speaker) && saved.speaker >= 0) {
    const option = document.createElement("option");
    option.value = saved.speaker;
    option.textContent = `保存した話者 (ID ${saved.speaker})`;
    $("speaker").replaceChildren(option);
  }
  $("modelLabel").textContent = $("model").value || "KoeLoop";
} finally {
  defaultsReady = true;
  $("send").disabled = false;
  $("settingsFields").disabled = false;
  refreshComposer();
}
