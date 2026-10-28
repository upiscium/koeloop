{
  description = "KoeLoop development and integration-test environment";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" ];
      forAllSystems = nixpkgs.lib.genAttrs systems;
      pythonPackages = p: [
        p.fastapi p.pydantic p.uvicorn p.httpx p.websockets p.pytest
        p.spacy p.sudachipy p.sudachidict-core p.spacy-models.ja_core_news_sm
      ];
      pythonFor = pkgs: pkgs.python3.withPackages pythonPackages;
      sttModel = pkgs: pkgs.fetchurl {
        url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/ggml-base-q5_1.bin";
        sha256 = "422f1ae452ade6f30a004d7e5c6a43195e4433bc370bf23fac9cc591f01a8898";
      };
      vadModel = pkgs: pkgs.fetchurl {
        url = "https://raw.githubusercontent.com/snakers4/silero-vad/be95df9152c0d7618fa1edfeb296fc3dae32376f/src/silero_vad/data/silero_vad.onnx";
        hash = "sha256-GhU6IvRQnikqlOZ9b5uF6N6yW0mIaCt+F0xlJ52HiOM=";
      };
      parakeetShell = pkgs: withBrowser: pkgs.mkShell {
        packages = [
          (pkgs.python3.withPackages (p: (pythonPackages p) ++ [ p.onnx-asr ]
            ++ pkgs.lib.optionals withBrowser [ p.playwright ]))
          pkgs.whisper-cpp pkgs.ruff pkgs.nodejs pkgs.curl
        ] ++ pkgs.lib.optionals withBrowser [ pkgs.chromium ];
        STT_BACKEND = "parakeet";
        STT_PARAKEET_MODEL_DIR = "${import ./nix/parakeet-model.nix { inherit pkgs; }}";
        STT_VAD_MODEL_PATH = "${vadModel pkgs}";
        STT_MODEL_PATH = "${sttModel pkgs}";
        STT_COMMAND = "${pkgs.whisper-cpp}/bin/whisper-cli";
        HF_HUB_OFFLINE = "1";
        HF_HUB_DISABLE_TELEMETRY = "1";
      };
      japaneseSttShell = pkgs: selection: withBrowser:
        let
          models = import ./nix/japanese-stt-models.nix { inherit pkgs; };
          all = selection == "all";
          voice = selection == "voice";
          useVosk = all || selection == "vosk";
          useTransformers = all || builtins.elem selection [ "whisper-ja" "wav2vec-ja" ];
        in pkgs.mkShell {
          packages = [
            (pkgs.python3.withPackages (p: (pythonPackages p)
              ++ pkgs.lib.optionals useVosk [ (import ./nix/vosk-python.nix { inherit pkgs; pythonPackages = p; }) ]
              ++ pkgs.lib.optionals useTransformers [ p.torch p.transformers ]
              ++ pkgs.lib.optionals (all || voice) [ p.onnx-asr ]
              ++ pkgs.lib.optionals withBrowser [ p.playwright ]))
            pkgs.whisper-cpp pkgs.ruff pkgs.nodejs
          ] ++ pkgs.lib.optionals withBrowser [ pkgs.chromium ];
          STT_BACKEND = if all || voice then "parakeet" else selection;
          STT_MODEL_PATH = "${sttModel pkgs}";
          STT_COMMAND = "${pkgs.whisper-cpp}/bin/whisper-cli";
          STT_VOSK_MODEL_DIR = if useVosk then "${models.vosk}" else "";
          STT_WHISPER_JA_MODEL_DIR = if all || selection == "whisper-ja" then "${models.whisper-ja}" else "";
          STT_WAV2VEC_JA_MODEL_DIR = if all || selection == "wav2vec-ja" then "${models.wav2vec-ja}" else "";
          STT_PARAKEET_MODEL_DIR = if all || voice then "${import ./nix/parakeet-model.nix { inherit pkgs; }}" else "";
          STT_VAD_MODEL_PATH = if all || voice then "${vadModel pkgs}" else "";
          HF_HUB_OFFLINE = "1";
          HF_HUB_DISABLE_TELEMETRY = "1";
        };
    in {
      devShells = forAllSystems (system:
        let pkgs = import nixpkgs { inherit system; };
        in {
          default = pkgs.mkShell {
            packages = [ (pythonFor pkgs) pkgs.ruff pkgs.nodejs pkgs.curl pkgs.whisper-cpp ];
            STT_BACKEND = "whisper";
            STT_MODEL_PATH = "${sttModel pkgs}";
            STT_COMMAND = "${pkgs.whisper-cpp}/bin/whisper-cli";
            # Official CPU image resolved on 2026-10-21. Docker daemon is host-managed.
            VOICEVOX_IMAGE = "voicevox/voicevox_engine@sha256:eb8c7f46a7d01217d1ff2b6f018261faedeceded3cc756b4fbbf371791ad6c90";
          };
          browser = pkgs.mkShell {
            packages = [
              (pkgs.python3.withPackages (p: (pythonPackages p) ++ [ p.playwright ]))
              pkgs.chromium pkgs.whisper-cpp
            ];
            STT_BACKEND = "whisper";
            STT_MODEL_PATH = "${sttModel pkgs}";
            STT_COMMAND = "${pkgs.whisper-cpp}/bin/whisper-cli";
          };
          # Optional evaluation environment: keep the smaller Whisper default intact.
          parakeet = parakeetShell pkgs false;
          parakeet-browser = parakeetShell pkgs true;
          vosk = japaneseSttShell pkgs "vosk" false;
          voice = japaneseSttShell pkgs "voice" false;
          voice-browser = japaneseSttShell pkgs "voice" true;
          whisper-ja = japaneseSttShell pkgs "whisper-ja" false;
          wav2vec-ja = japaneseSttShell pkgs "wav2vec-ja" false;
          stt-eval = japaneseSttShell pkgs "all" false;
          stt-eval-browser = japaneseSttShell pkgs "all" true;
        });

      checks = forAllSystems (system:
        let
          pkgs = import nixpkgs { inherit system; };
          source = pkgs.lib.cleanSourceWith {
            src = self;
            filter = path: type:
              let name = baseNameOf path;
              in !(builtins.elem name [
                "artifacts" ".venv" ".pytest_cache" ".ruff_cache" "__pycache__" ".env"
              ]) && pkgs.lib.cleanSourceFilter path type;
          };
        in {
          unit = pkgs.runCommand "voicevox-restream-tests" {
            nativeBuildInputs = [ (pythonFor pkgs) pkgs.nodejs ];
          } ''
            cp -r ${source} source
            chmod -R u+w source
            cd source
            export HOME="$TMPDIR"
            python -c "from restream.syntax import installed; assert installed()"
            python -m pytest -q
            node --check restream/static/app.js
            node --check restream/static/chat.js
            node --check restream/static/chat-audio.js
            node --check restream/static/dictation.js
            node --check restream/static/mic-worklet.js
            node --check restream/static/voice-turn.js
            node --check restream/static/voice-worklet.js
            node --test tests/microphone.test.mjs
            touch "$out"
          '';
        });
    };
}
