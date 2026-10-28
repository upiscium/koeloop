{ pkgs }:
let
  hfModel = name: repo: revision: files: pkgs.linkFarm name (pkgs.lib.mapAttrsToList (file: hash: {
    name = file;
    path = pkgs.fetchurl {
      url = "https://huggingface.co/${repo}/resolve/${revision}/${file}";
      sha256 = hash;
    };
  }) files);
  voskArchive = pkgs.fetchurl {
    url = "https://alphacephei.com/vosk/models/vosk-model-small-ja-0.22.zip";
    hash = "sha256-76CS0oAVOndhXp4MfXKD6T5gDePRnTvsaGxX7xnVLqw=";
  };
in {
  whisper-ja = hfModel "ivydata-whisper-small-japanese" "Ivydata/whisper-small-japanese"
    "d889b2b90cae2e7df49238fb84f7ad13248fc1cd" {
      "pytorch_model.bin" = "24a5a8e7beb868ab93e888e5b4b47c05a45cd0f51592ac6bbff34a65b4c90871";
      "config.json" = "sha256-SrK529uimY76fP+fEPzwETCnh1ONtjERM0C9OWTOX0o=";
      "generation_config.json" = "sha256-WNIM478EdG/d+WLcvs+3vX9lEMZlAkVNHbErRI/IVOc=";
      "preprocessor_config.json" = "sha256-pqdtKMk+2yc2aeueCwY2or3bsScsMmHke3ym39usG40=";
      "tokenizer_config.json" = "sha256-FPhL30uezb1HON3IHBehuu38AruTxuBJyVHhWhtAtw0=";
      "special_tokens_map.json" = "sha256-sKrGlQOBxK4vYLPw+rBLz+OZpqMgyz5Meu4tUpJNcz4=";
      "vocab.json" = "sha256-UNapGfCgYB1WoE61g8eA0YVTqjiCVLoxWOtqAPE+LBo=";
      "merges.txt" = "sha256-ePJ/gB/rQoOulp1Mrayyj8BRtlVkdAKxolLHB56hHEY=";
      "added_tokens.json" = "sha256-zpSf5yDBQxHLbERuac/jQNxmnXsAYHem/u1q5XHdfog=";
      "normalizer.json" = "sha256-vxxQfchyTKnPmQNkDaz7adri8A7e5PIc66EGpzkvJt0=";
    };
  wav2vec-ja = hfModel "ivydata-wav2vec2-xlsr-japanese" "Ivydata/wav2vec2-large-xlsr-53-japanese"
    "3feaaf2fd0e2139ca3343c5a7058e172225579b4" {
      "pytorch_model.bin" = "3383026ffd844f69c1e4e30a00f6fc9d378a4cb7c5cfb06027e39cf952d8dab2";
      "config.json" = "sha256-N9VrnD9PGtxvmmed0EUKYyq6BAIoQI69Mxmc9hYV524=";
      "preprocessor_config.json" = "sha256-/8cvR1zXloDV1ln7rvxpHNvO5clwOhX69wil6RRSqNg=";
      "tokenizer_config.json" = "sha256-3Rr2qSQeHmM1xlhTV5lWuqODDOdChhkDcIsXV5GvICY=";
      "special_tokens_map.json" = "sha256-Z7hHp4CIOArhFrEPAZpib7EVLvuI+sMCkc/h61WWlXU=";
      "vocab.json" = "sha256-M9hMcvurgi1R25drJbObu0vFRyMJp5ODhdVal6uZB+k=";
      "added_tokens.json" = "sha256-FVfDQKIAcqzCRVJqq4ue+wGycEoRMp1uC2D/ETho75M=";
    };
  vosk = pkgs.runCommand "vosk-model-small-ja-0.22" { nativeBuildInputs = [ pkgs.unzip ]; } ''
    unzip -q ${voskArchive} -d unpacked
    cp -r unpacked/vosk-model-small-ja-0.22 "$out"
  '';
}
