{ pkgs, pythonPackages }:
let
  wheel = if pkgs.stdenv.hostPlatform.isx86_64 then {
    url = "https://files.pythonhosted.org/packages/fc/ca/83398cfcd557360a3d7b2d732aee1c5f6999f68618d1645f38d53e14c9ff/vosk-0.3.45-py3-none-manylinux_2_12_x86_64.manylinux2010_x86_64.whl";
    sha256 = "25e025093c4399d7278f543568ed8cc5460ac3a4bf48c23673ace1e25d26619f";
  } else {
    url = "https://files.pythonhosted.org/packages/a4/23/3130a69fa0bf4f5566a52e415c18cd854bf561547bb6505666a6eb1bb625/vosk-0.3.45-py3-none-manylinux2014_aarch64.whl";
    sha256 = "54efb47dd890e544e9e20f0316413acec7f8680d04ec095c6140ab4e70262704";
  };
in pythonPackages.buildPythonPackage {
  pname = "vosk";
  version = "0.3.45";
  format = "wheel";
  src = pkgs.fetchurl wheel;
  nativeBuildInputs = [ pkgs.autoPatchelfHook ];
  buildInputs = [ pkgs.stdenv.cc.cc.lib ];
  dependencies = with pythonPackages; [ cffi requests tqdm srt websockets ];
  pythonImportsCheck = [ "vosk" ];
  meta = {
    description = "Vosk offline recognition (official binary wheel)";
    homepage = "https://github.com/alphacep/vosk-api";
    license = pkgs.lib.licenses.asl20;
    platforms = [ "x86_64-linux" "aarch64-linux" ];
  };
}
