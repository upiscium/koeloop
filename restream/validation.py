"""Human-readable start errors without echoing keys, prompts, or submitted values."""

from pydantic import ValidationError

START_HINTS = {
    "type": "type は start にしてください。",
    "source": "入力元 (source) は text / llm / external のいずれかです。",
    "mode": "分割方式 (mode) は raw / sentence / balanced / syntax のいずれかです。",
    "audio": "音声設定 (audio) は true / false で指定してください。",
    "speaker": "話者 ID (speaker) は 0～100000 の整数で指定してください。",
    "speed": "読み上げ速度 (speed) は 0.5～2.0 の数値で指定してください。",
    "min_chars": "節の最小文字数 (min_chars) は 1～1000 の整数で指定してください。",
    "max_chars": "発話の最大文字数 (max_chars) は 1～1000 の整数で指定してください。",
    "lookahead": "先読み文字数 (lookahead) は 0～1000 の整数で指定してください。",
    "adjust_pauses": "句読点の間の設定 (adjust_pauses) は true / false で指定してください。",
    "sentence_pause_extra_ms": "句点の追加時間 (sentence_pause_extra_ms) は 0～1000 ms の整数で指定してください。",
    "comma_pause_scale": "読点の倍率 (comma_pause_scale) は 0～2 の数値で指定してください。",
    "prompt": "LLM への入力 (prompt) は 16000 文字以下にしてください。",
    "messages": "会話履歴 (messages) の形式・役割・長さを確認してください。",
    "backend": "接続設定 (backend) の URL・モデル名・認証設定を確認してください。",
}


def start_error_message(error: ValidationError) -> str:
    hints = []
    for issue in error.errors(include_input=False, include_context=False, include_url=False):
        if issue["type"] == "extra_forbidden":
            hint = "未対応の設定項目があります。サーバーを再起動し、画面を再読み込みしてください。"
        elif issue["type"] == "json_invalid":
            hint = "開始メッセージを有効な JSON で送信してください。"
        else:
            path = issue["loc"]
            hint = START_HINTS.get(path[0] if path else None,
                                   "会話履歴の順序・合計文字数と prompt の併用を確認してください。")
        if hint not in hints:
            hints.append(hint)
        if len(hints) >= 4:
            break
    return "start の設定が不正です。" + " ".join(hints)
