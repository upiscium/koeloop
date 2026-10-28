"use strict";
const $ = (id) => document.getElementById(id);
const reasons = {raw: "受信チャンク", sentence: "文末", clause: "節＋先読み", syntax_clause: "構文で確認した節", forced_limit: "上限で強制分割", end_of_input: "入力終了"};
const boundaryReasons = {not_predicate: "名詞句・副詞など／述語ではない", relative_clause: "後続名詞への修飾が残る", not_clause: "節の係り受けが未確認", no_connective: "接続表現がない", right_predicate_pending: "後続の述語が未確定", crossing_dependency: "他の係り受けが境界をまたぐ", unstable_dependency: "先読みで係り先が変わる", stable_clause: "2つの解析で節境界が一致", not_token_boundary: "単語境界が未確定"};
let current = null;
let context = null;
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
function status(text, error = false) {
  $("status").textContent = text;
  $("status").classList.toggle("error", error);
}
function updateModeHint() {
  if ($("mode").value !== "syntax") {
    $("modeHint").textContent = $("mode").value === "raw"
      ? "受信した content 差分ごとに合成します。文末・構文でまとめません。句読点の間の調整は別設定です。"
      : "文末が届けば文単位で確定します。構文解析を使う場合は分割方式を「構文解析」に変更してください。";
    return;
  }
  const min = Number($("minChars").value), lookahead = Number($("lookahead").value);
  if (!$("minChars").value || !$("lookahead").value || !Number.isInteger(min) || min < 1
      || !Number.isInteger(lookahead) || lookahead < 0) {
    $("modeHint").textContent = "構文解析の条件を確認するには、最小文字数と先読み文字数を入力してください。";
    return;
  }
  $("modeHint").textContent = `構文解析は文末が未確定で、括弧の外の読点まで最小文字数（現在 ${min}）以上、その後ろに ${Math.max(12, lookahead) + 8} 文字以上届いたときに実行します。再解析は8文字以上増えたときだけ。文末が先に届いた場合などは解析0回でも正常です。`;
}
for (const id of ["mode", "minChars", "lookahead"]) $(id).addEventListener("input", updateModeHint);
updateModeHint();
function metric(run, id) {
  if ($(id).textContent === "—") $(id).textContent = `${Math.round(performance.now() - run.started)} ms`;
}
function finish(run, message, error = false) {
  if (current !== run) return;
  current = null;
  run.stopped = true;
  if (run.echo) run.echo();
  for (const source of run.sources) { source.onended = null; try { source.stop(); source.disconnect(); } catch (_) {} }
  run.sources.clear();
  if (run.ws && run.ws.readyState < WebSocket.CLOSING) run.ws.close();
  $("start").disabled = false;
  $("stop").disabled = true;
  status(message, error);
  const article = document.createElement("article");
  const heading = document.createElement("b");
  heading.textContent = `${new Date().toLocaleTimeString()} / ${run.mode} / ${message}`;
  const detail = document.createElement("p");
  detail.textContent = `入力 ${run.received.length} 文字・${run.rows.size} 発話 / 初回確定 ${$("firstSegment").textContent} / 初回再生予定 ${$("firstAudio").textContent} / 空白 ${$("gaps").textContent} / 話者 ${run.speaker}・速度 ${run.speed} / 間調整 ${run.adjustPauses ? `句点 +${run.sentenceExtra}ms・読点 ×${run.commaScale}` : "オフ"} / ${$("analysisMetrics").textContent} ${$("syntaxWarning").textContent}`;
  const sample = document.createElement("details");
  const summary = document.createElement("summary");
  summary.textContent = "入力と分割結果";
  const text = document.createElement("pre");
  text.textContent = run.segmentTexts.join("\n────\n");
  sample.append(summary, text);
  article.append(heading, detail, sample);
  $("history").prepend(article);
  while ($("history").children.length > 10) $("history").lastElementChild.remove();
}
async function scheduleAudio(run, event) {
  if (current !== run) return;
  const bytes = Uint8Array.from(atob(event.wav_base64), (c) => c.charCodeAt(0));
  const buffer = await context.decodeAudioData(bytes.buffer);
  if (current !== run) return;
  const start = Math.max(context.currentTime + 0.03, run.nextTime);
  if (run.nextTime) run.gaps += Math.max(0, start - run.nextTime);
  if ($("firstAudio").textContent === "—") {
    $("firstAudio").textContent = `${Math.round(performance.now() - run.started + (start - context.currentTime) * 1000)} ms`;
  }
  $("gaps").textContent = `${Math.round(run.gaps * 1000)} ms`;
  const source = context.createBufferSource();
  source.buffer = buffer;
  source.connect(context.destination);
  run.sources.add(source);
  source.onended = () => {
    source.disconnect();
    run.sources.delete(source);
    if (current === run && run.ws.readyState === WebSocket.OPEN) {
      run.ws.send(JSON.stringify({type: "played", seq: event.seq}));
    }
  };
  source.start(start);
  run.nextTime = start + buffer.duration;
  const cell = run.rows.get(event.seq);
  if (cell) cell.textContent = `${event.synthesis_ms} ms`;
}
async function sendText(run) {
  const chars = Array.from(run.text);
  for (let i = 0; i < chars.length && current === run; i += run.chunkSize) {
    // Wait for the server's delta echo to pace input and avoid an unbounded send queue.
    const echoed = new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error("入力確認がタイムアウトしました。")), 180000);
      run.echo = () => { clearTimeout(timer); resolve(); };
    });
    run.ws.send(JSON.stringify({type: "text", text: chars.slice(i, i + run.chunkSize).join("")}));
    await echoed;
    run.echo = null;
    if (run.interval) await sleep(run.interval);
  }
  if (current === run) run.ws.send(JSON.stringify({type: "end"}));
}
$("start").addEventListener("click", async () => {
  if (current) return;
  const number = (id) => $(id).value.trim() === "" ? NaN : Number($(id).value);
  const speed = number("speed"), speaker = number("speaker");
  if (!Number.isFinite(speed) || speed < 0.5 || speed > 2) {
    return status("読み上げ速度は 0.5～2.0 の数値を入力してください。", true);
  }
  if (!Number.isInteger(speaker) || speaker < 0 || speaker > 100000) {
    return status("有効な話者を選択してください。必要なら「話者を再取得」を押してください。", true);
  }
  const min = number("minChars"), max = number("maxChars"), lookahead = number("lookahead");
  const chunkSize = $("source").value === "mock" ? number("chunkSize") : 4096;
  const interval = $("source").value === "mock" ? number("interval") : 0;
  const adjustPauses = $("adjustPauses").checked;
  const sentenceExtra = adjustPauses ? number("sentenceExtra") : 200;
  const commaScale = adjustPauses ? number("commaScale") : 0.8;
  if (!Number.isInteger(sentenceExtra) || sentenceExtra < 0 || sentenceExtra > 1000
      || !Number.isFinite(commaScale) || commaScale < 0 || commaScale > 2) {
    return status("句点の追加は 0～1000 ms、読点の倍率は 0～2 にしてください。", true);
  }
  if ($("source").value !== "external" && !$("text").value.trim()) return status("テキストを入力してください。", true);
  if ($("source").value === "llm" && Array.from($("text").value).length > 16000) {
    return status("LLM への入力は 16000 文字以下にしてください。", true);
  }
  if (![min, max, lookahead, chunkSize, interval].every(Number.isInteger) || min < 1 || max < min || max > 1000 || lookahead < 0 || lookahead > max || chunkSize < 1 || chunkSize > 8192 || interval < 0 || interval > 5000) {
    return status("分割・送信設定の範囲を確認してください。", true);
  }
  const run = {started: performance.now(), mode: $("mode").value, speaker, speed,
    adjustPauses, sentenceExtra, commaScale,
    text: $("text").value, source: $("source").value, chunkSize, interval, audio: $("audio").checked,
    received: "", rows: new Map(), segmentTexts: [], boundaries: [], sources: new Set(), nextTime: 0, gaps: 0, audioChain: Promise.resolve()};
  const request = JSON.stringify({type: "start", source: ["llm", "external"].includes(run.source) ? run.source : "text", mode: run.mode,
    speaker: run.speaker, speed: run.speed, audio: run.audio, min_chars: min, max_chars: max, lookahead,
    adjust_pauses: run.adjustPauses, sentence_pause_extra_ms: run.sentenceExtra, comma_pause_scale: run.commaScale,
    prompt: run.source === "llm" ? run.text : ""});
  if (new TextEncoder().encode(request).length > 65536) {
    return status("開始設定が送信上限（65536バイト）を超えています。入力を短くしてください。", true);
  }
  current = run;
  $("start").disabled = true;
  $("stop").disabled = false;
  $("segments").replaceChildren();
  $("received").textContent = "";
  $("externalUrl").value = "";
  for (const id of ["syntaxLog", "syntaxWarning", "analysisMetrics"]) $(id).textContent = "";
  for (const id of ["firstDelta", "firstSegment", "firstAudio", "gaps"]) $(id).textContent = "—";
  try {
    if (run.audio) {
      context = context || new AudioContext();
      await context.resume();
      if (context.state !== "running") throw new Error("ブラウザの音声再生を許可してください。");
    }
    if (current !== run) return;
    status("接続中…");
    run.ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
    run.ws.onopen = () => {
      if (current !== run) return run.ws.close();
      run.ws.send(request);
    };
    run.ws.onmessage = ({data}) => {
      if (current !== run) return;
      try {
        const event = JSON.parse(data);
        if (event.type === "ready") {
          status("受信・処理中…");
          if (run.source === "external") {
            $("externalUrl").value = `${location.origin}/api/sessions/${event.session_id}/events`;
            status("外部入力を待っています。表示された URL に POST してください（無操作180秒で終了）。");
          } else if (run.source !== "llm") sendText(run).catch((e) => finish(run, e.message, true));
        } else if (event.type === "delta") {
          metric(run, "firstDelta");
          run.received += event.text;
          $("received").textContent = run.received;
          if (run.echo) run.echo();
        } else if (event.type === "segment") {
          metric(run, "firstSegment");
          run.segmentTexts.push(event.text);
          const row = document.createElement("tr");
          if (event.reason === "forced_limit") row.classList.add("forced");
          for (const value of [event.seq, event.text, reasons[event.reason] || event.reason, `${event.confirmed_ms} ms`, "—"]) {
            const cell = document.createElement("td"); cell.textContent = value; row.append(cell);
          }
          run.rows.set(event.seq, row.lastElementChild);
          $("segments").append(row);
        } else if (event.type === "audio") {
          run.audioChain = run.audioChain.then(() => scheduleAudio(run, event)).catch((e) => finish(run, `再生失敗: ${e.message}`, true));
        } else if (event.type === "text_done") {
          if (event.segmentation?.requested_mode === "syntax") {
            $("analysisMetrics").textContent = `構文解析 ${event.segmentation.analysis_calls} 回・${event.segmentation.analysis_ms} ms（初回モデル読み込みを含む） / 実効方式 ${event.segmentation.effective_mode}`;
          }
          status(run.audio ? "入力完了・残りの音声を再生中…" : "入力完了");
        } else if (event.type === "warning") {
          $("syntaxWarning").textContent = event.message;
        } else if (event.type === "boundary") {
          run.boundaries.push(`${event.selected ? "採用" : event.accepted ? "候補可" : "保留"}「…${event.preview}」: ${boundaryReasons[event.reason] || event.reason}`);
          run.boundaries = run.boundaries.slice(-50);
          $("syntaxLog").textContent = run.boundaries.join("\n");
        } else if (event.type === "done") {
          finish(run, "完了");
        } else if (event.type === "error") {
          finish(run, event.message, true);
        } else if (event.type === "cancelled") {
          finish(run, "停止しました");
        }
      } catch (e) { finish(run, e.message, true); }
    };
    run.ws.onerror = () => finish(run, "WebSocket 接続に失敗しました。", true);
    run.ws.onclose = () => { if (current === run) finish(run, "完了前に接続が切れました。", true); };
  } catch (e) { finish(run, e.message, true); }
});
$("stop").addEventListener("click", () => {
  if (!current) return;
  const run = current;
  if (run.ws?.readyState === WebSocket.OPEN) run.ws.send(JSON.stringify({type: "cancel"}));
  if (run.echo) run.echo();
  finish(run, "停止しました");
});
$("reuse").addEventListener("click", () => {
  if (current) return status("停止または完了してから入力を変更してください。");
  $("text").value = $("received").textContent;
  $("source").value = "mock";
  status("受信済みテキストを比較用入力にしました。");
});
async function loadSpeakers() {
  try {
    const response = await fetch("/api/speakers");
    const speakers = await response.json();
    if (!response.ok) throw new Error(speakers.error);
    const options = [];
    for (const speaker of speakers) for (const style of speaker.styles) {
      if (style.type && style.type !== "talk") continue;
      const option = document.createElement("option");
      option.value = style.id; option.textContent = `${speaker.name} / ${style.name} (${style.id})`;
      options.push(option);
    }
    const previous = $("speaker").value;
    if (!options.length) throw new Error("読み上げ可能な話者が見つかりませんでした。");
    $("speaker").replaceChildren(...options);
    if (options.some((o) => o.value === previous)) $("speaker").value = previous;
    if (!current) status("準備完了");
  } catch (e) { if (!current) status(`${e.message} 音声をオフにすると分割のみ試せます。`, true); }
}
$("speakers").addEventListener("click", loadSpeakers);
for (const [id, text] of [
  ["relativeExample", "私が昨日の夕方に駅前の喫茶店でゆっくり話を聞いた、近所に住んでいる友人が今日は新しい資料を持って私の家を訪ねてくれる予定です。"],
  ["clauseExample", "昨日から降り続いていた激しい雨がようやくやんだので、私は近所の公園に出かけて、濡れた木々や花の様子をゆっくり眺めながら散歩を楽しむことにしました。"],
]) $(id).addEventListener("click", () => {
  if (current) return;
  $("text").value = text;
  $("source").value = "mock";
  $("chunkSize").value = "4";
  $("interval").value = "100";
  status("同じ例文で文末・バランス・構文解析を切り替えて比較してください。");
});
fetch("/api/config").then((r) => r.json()).then((c) => {
  $("config").textContent = c.llm_configured ? `LLM モデル: ${c.llm_model} / reasoning_effort: ${c.llm_reasoning_effort || "接続先の既定値"}` : "LLM 未設定。サーバーで LLM_MODEL を設定すると接続できます。";
  $("syntaxStatus").textContent = c.syntax?.installed ? `構文解析モデル: ${c.syntax.model}（導入済み・初回に読み込み）` : "構文解析器は未導入です。構文解析を選ぶと文末方式へ切り替えます。";
}).catch(() => status("サーバーに接続できません。", true));
loadSpeakers();
