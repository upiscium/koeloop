{ pkgs }:
let
  # NVIDIA's Japanese hybrid model, third-party ONNX export using the TDT decoder.
  # This is NOT the European-language parakeet-tdt-0.6b-v3 model.
  base = "https://huggingface.co/OpenVoiceOS/nvidia-parakeet-tdt_ctc-0.6b-ja-onnx/resolve/4353e7b9e2e9ebdd35e85b8140c7351d03c2219c";
  files = {
    "config.json" = "sha256-eOp4CWX5lnMKw53v5nGP86/uaG0Ds2PmLr9RLPqORyg=";
    "vocab.txt" = "sha256-cy9kxTkJ8mIMcT9BBrSH2S5vVKaRWzzT0dvTL59POSo=";
    "encoder-model.int8.onnx" = "3c9f3f8840b105fad60ac743e50a2bbe01e467205ea0978f3a3407b2b18148b2";
    "decoder_joint-model.int8.onnx" = "1de334dbbf66233309e18b6242b5dbc34005851f282cd89e36597c3cb8564498";
  };
in pkgs.linkFarm "parakeet-ja-0.6b-onnx-int8" (pkgs.lib.mapAttrsToList (name: hash: {
  inherit name;
  path = pkgs.fetchurl { url = "${base}/${name}"; sha256 = hash; };
}) files)
