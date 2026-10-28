# 構文解析・音声認識に利用する第三者コンポーネント

## spaCy

- 作者：Explosion および spaCy contributors
- 利用バージョン（固定 Nix 環境）：3.8.16
- ソース：https://github.com/explosion/spaCy
- ライセンス：MIT
- ライセンス本文：https://github.com/explosion/spaCy/blob/master/LICENSE

## ja_core_news_sm

- 作者：Explosion
- モデル：Japanese CPU pipeline `ja_core_news_sm` 3.8.0
- モデル配布：https://github.com/explosion/spacy-models/releases/tag/ja_core_news_sm-3.8.0
- メタデータ：https://github.com/explosion/spacy-models/blob/master/meta/ja_core_news_sm-3.8.0.json
- ライセンス：**Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)**
- ライセンス本文：https://creativecommons.org/licenses/by-sa/4.0/legalcode
- 学習データ：UD Japanese GSD v2.8、UD Japanese GSD v2.8 NER（作者・出典は上記モデルメタデータ参照）

本プロジェクトでは、配布済みモデルを変更・再学習せずに利用します。実行時は固有表現認識コンポーネントをロードせず、品詞・係り受けの情報を利用します。モデルを再配布・改変する場合は、そのライセンス条件を確認してください。モデル作者による本プロジェクトへの推奨・保証を意味しません。

## SudachiPy / SudachiDict-core

- 作者：Works Applications および contributors
- 固定 Nix 環境のバージョン：SudachiPy 0.6.11 / SudachiDict-core 20260428
- SudachiPy：https://github.com/WorksApplications/SudachiPy
- 辞書：https://github.com/WorksApplications/SudachiDict
- ライセンス：Apache License 2.0。辞書には追加の出典・著作権表示があります。
- 辞書の法的表示：https://github.com/WorksApplications/SudachiDict/blob/develop/LEGAL

モデルと辞書は Nix/Python の依存パッケージとして取得し、このソースツリーへコピーしていません。VOICEVOX と各話者の利用条件は、これら NLP コンポーネントの条件とは別です。

## whisper.cpp / Whisper base Q5_1

- whisper.cpp：ggml-org contributors、MIT License。
- 確認済みバージョン：1.9.5（固定 Nixpkgs のパッケージ）。
- ソースとライセンス：https://github.com/ggml-org/whisper.cpp / https://github.com/ggml-org/whisper.cpp/blob/master/LICENSE
- 元の Whisper モデル：OpenAI、MIT License。
- モデルの出典とライセンス：https://github.com/openai/whisper / https://github.com/openai/whisper/blob/main/LICENSE
- 使用する量子化済み配布物：`ggerganov/whisper.cpp` の `ggml-base-q5_1.bin`（多言語版）。
- 配布元：https://huggingface.co/ggerganov/whisper.cpp
- 固定リビジョン：`5359861c739e955e79d9a303bcbc70fb988958b1`
- SHA-256：`422f1ae452ade6f30a004d7e5c6a43195e4433bc370bf23fac9cc591f01a8898`
- ファイルサイズ：59,707,625バイト。これはモデルファイルのサイズであり、実行時メモリ使用量ではありません。

配布済みの量子化モデルをそのまま使用し、本プロジェクトで再学習・追加量子化はしていません。モデルは環境構築時に取得し、音声入力はローカル CPU で処理します。

## Parakeet TDT-CTC 0.6B Japanese / ONNX int8（任意導入）

- 元モデルの作者：NVIDIA NeMo team。
- 正式名：`nvidia/parakeet-tdt_ctc-0.6b-ja`。
- モデルカード：https://huggingface.co/nvidia/parakeet-tdt_ctc-0.6b-ja
- ライセンス：**CC BY 4.0**。https://creativecommons.org/licenses/by/4.0/legalcode
- 変換・量子化版の配布者：OpenVoiceOS。
- 使用する配布物：https://huggingface.co/OpenVoiceOS/nvidia-parakeet-tdt_ctc-0.6b-ja-onnx
- 固定リビジョン：`4353e7b9e2e9ebdd35e85b8140c7351d03c2219c`
- 使用する部分：TDT の encoder と joint decoder、int8版。CTC版やfp32版は取得していません。

配布者が行った NeMo → ONNX 変換と int8 量子化を含む派生物です。本プロジェクトでは配布済みファイルを変更・再学習せず使用しています。変換元チェックポイントの厳密なリビジョンは配布カードからは確認できず、本プロジェクトで固定しているのは変換版のリビジョンと各ファイルのハッシュです。NVIDIA / OpenVoiceOS による本プロジェクトへの推奨や、元モデルと同等の精度の保証を意味しません。

| ファイル | SHA-256 |
|---|---|
| `encoder-model.int8.onnx` | `3c9f3f8840b105fad60ac743e50a2bbe01e467205ea0978f3a3407b2b18148b2` |
| `decoder_joint-model.int8.onnx` | `1de334dbbf66233309e18b6242b5dbc34005851f282cd89e36597c3cb8564498` |
| `config.json` | Nix SRI `sha256-eOp4CWX5lnMKw53v5nGP86/uaG0Ds2PmLr9RLPqORyg=` |
| `vocab.txt` | Nix SRI `sha256-cy9kxTkJ8mIMcT9BBrSH2S5vVKaRWzzT0dvTL59POSo=` |

ランタイム：

- **onnx-asr 0.12.0**：istupakov / contributors、MIT License。https://github.com/istupakov/onnx-asr
- **ONNX Runtime 1.27.1**：Microsoft / contributors、MIT License。https://github.com/microsoft/onnxruntime
- CPU Execution Provider とローカルファイルのみで実行。モデルコードのリモートロードはしません。

## Silero VAD 6.2（音声会話用）

- 作者／配布元：Silero team / snakers4。
- ソース：https://github.com/snakers4/silero-vad/tree/be95df9152c0d7618fa1edfeb296fc3dae32376f
- 固定コミット：`be95df9152c0d7618fa1edfeb296fc3dae32376f`。
- 使用ファイル：`src/silero_vad/data/silero_vad.onnx`（`silero_vad.onnx`）。
- ファイルの Nix SRI：`sha256-GhU6IvRQnikqlOZ9b5uF6N6yW0mIaCt+F0xlJ52HiOM=`。
- ライセンス：MIT。https://github.com/snakers4/silero-vad/blob/be95df9152c0d7618fa1edfeb296fc3dae32376f/LICENSE
- ランタイム：ONNX Runtime 1.27.1（上記 Parakeet 環境と同じバージョン）。

配布されたCPU ONNXグラフは変更せず使用します。音声会話の発話ごとに再帰状態を分離し、ローカルCPUで検出します。PyTorchや外部STTサービスへの音声送信は使用しません。VADは発話区間を検出するだけで文字起こしは行わず、認識は別途Parakeetで実行します。

## Ivydata 日本語微調整モデル（任意導入）

- 作者／配布者：Ivydata / Kosuke Suzuki。
- `Ivydata/whisper-small-japanese`：https://huggingface.co/Ivydata/whisper-small-japanese
  - 元モデル：OpenAI Whisper-small。
  - 固定リビジョン：`d889b2b90cae2e7df49238fb84f7ad13248fc1cd`
  - 重み SHA-256：`24a5a8e7beb868ab93e888e5b4b47c05a45cd0f51592ac6bbff34a65b4c90871`
- `Ivydata/wav2vec2-large-xlsr-53-japanese`：https://huggingface.co/Ivydata/wav2vec2-large-xlsr-53-japanese
  - 元モデル：Meta / Facebook wav2vec2-large-xlsr-53。
  - 固定リビジョン：`3feaaf2fd0e2139ca3343c5a7058e172225579b4`
  - 重み SHA-256：`3383026ffd844f69c1e4e30a00f6fc9d378a4cb7c5cfb06027e39cf952d8dab2`
- 両モデルカードのライセンス表記：Apache-2.0。https://www.apache.org/licenses/LICENSE-2.0
- 学習データはモデルカードによると Common Voice、JVS、JSUT。本プロジェクトでそれらを再配布・再学習していません。
- 配布済みの重み・トークナイザーを無変更で使用。CPU FP32で評価し、量子化はしていません。
- 設定・語彙などのファイルも `nix/japanese-stt-models.nix` でハッシュ固定。公開ファイルは PyTorch `.bin` なので、現行 PyTorch の制限付き weights-only ロードを使用します。
- ランタイム：Hugging Face Transformers 5.17.0（Apache-2.0）、PyTorch 2.13.0（BSD-style license）。
  - https://github.com/huggingface/transformers/blob/main/LICENSE
  - https://github.com/pytorch/pytorch/blob/main/LICENSE

## Vosk 日本語小型モデル（任意導入）

- 作者／配布者：Alpha Cephei / Vosk contributors。
- モデル：`vosk-model-small-ja-0.22`。
- 公式配布と評価値：https://alphacephei.com/vosk/models
- モデルzip SHA-256（Nix SRI）：`sha256-76CS0oAVOndhXp4MfXKD6T5gDePRnTvsaGxX7xnVLqw=`。
- API：Vosk 0.3.45、公式 PyPI wheel。Nix で共有ライブラリの参照先を調整して利用。
- ソースとライセンス：https://github.com/alphacep/vosk-api / https://github.com/alphacep/vosk-api/blob/master/COPYING
- モデル・API の公開ライセンス：Apache-2.0。Kaldi系のモデルと復号グラフをそのまま使用し、辞書や文法を変更していません。

モデル配布元の評価値は、本プロジェクトの測定値や対象環境での動作保証を意味しません。データセットを別途取得する場合、その音声自体のライセンスと利用条件も確認してください。
