{
  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    treefmt-nix.url = "github:numtide/treefmt-nix";
    gitignore.url = "github:hercules-ci/gitignore.nix";
  };
  outputs = {...} @ inputs:
    inputs.flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = (import inputs.nixpkgs) {
          inherit system;
          config = {
            allowUnfree = true;
          };
        };
      in {
        devShells = rec {
          pyShell = let
            pythonPkg = pythonPackages: with pythonPackages; [
              ipykernel
              pip
              langchain
              langchain-community
              langchain-ollama
              sqlalchemy
              pydantic
              ollama
            ];
          in pkgs.mkShell {
            packages = with pkgs; [
              podman-compose
              podman-compose
              (pkgs.python3.withPackages (pythonPkg))
              ruff
              ollama-rocm
            ];
          };
          default = pyShell;
        };
        formatter = let
          treefmtconfig = inputs.treefmt-nix.lib.evalModule pkgs {
            projectRootFile = "flake.nix";
            programs = {
              alejandra.enable = true;
              ruff-format.enable = true;
              toml-sort.enable = true;
              yamlfmt.enable = true;
              mdformat.enable = true;
              shellcheck.enable = true;
              shfmt.enable = true;
            };
            settings.formatter.shellcheck.excludes = [".envrc"];
          };
        in
          treefmtconfig.config.build.wrapper;
        apps = rec {
        };
        packages = rec {
        };
      }
    );
}
