{ pkgs ? import <nixpkgs> {} }:
pkgs.mkShell {
  packages = [
    (pkgs.python3.withPackages (p: [ p.fastapi p.uvicorn p.httpx p.websockets p.pytest ]))
  ];
}
