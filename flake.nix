{
  description = "Python development environment with ML libraries";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            allowBroken = true;
            allowInsecure = false;
          };
        };
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            (python3.withPackages (ps: with ps; [
              opencv4
              numpy
              pillow
              torch-bin
              torchvision-bin
              plotext
              pycocotools
              exif
              torchmetrics
            ]))
          ];

          shellHook = ''
            export PS1="\[\033[1;32m\]\W\[\033[0m\] $ "
            echo "Welcome to the Python shell with Torch, OpenCV, and other ML libraries!"
            echo "Python version: $(python --version)"
            echo "Torch version: $(python -c 'import torch; print(torch.__version__)')"
            echo "OpenCV version: $(python -c 'import cv2; print(cv2.__version__)')"
          '';
        };
      }
    );
}