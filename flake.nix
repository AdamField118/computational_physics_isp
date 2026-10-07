{
  description = "Computational physics: Python, JAX, C/C++, Fortran, Rust and Julia";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";

  outputs = { self, nixpkgs }:
    let
      # The existing native code uses GNU OpenMP and Linux shared-library names.
      systems = [ "x86_64-linux" ];
      forAllSystems = nixpkgs.lib.genAttrs systems;
      shellFor = system: cuda:
        let
          pkgs = import nixpkgs { inherit system; config.allowUnfreePredicate = pkg:
            pkgs.lib.getName pkg == "triangle" || cuda; };
          python = pkgs.python312.withPackages (ps: with ps; let
            pyjulia = buildPythonPackage {
              pname = "julia";
              version = "0.6.2";
              format = "wheel";
              src = pkgs.fetchPypi {
                pname = "julia"; version = "0.6.2";
                format = "wheel"; python = "py2.py3";
                hash = "sha256-kHUvcTdv25kZQ50gSWwtqyRIbfpP6Kgx1t0Uoby8BNE=";
              };
              doCheck = false; # PyCall is initialized by comphys-build julia.
              pythonImportsCheck = [ "julia" ];
            };
            triangle = buildPythonPackage {
              pname = "triangle";
              version = "20250106";
              format = "wheel";
              src = pkgs.fetchPypi {
                pname = "triangle"; version = "20250106";
                format = "wheel"; dist = "cp312"; python = "cp312"; abi = "cp312";
                platform = "manylinux_2_17_x86_64.manylinux2014_x86_64";
                hash = "sha256-OMIhu25AP4HeSQUImVAqgybJVl9ixNaRcQcLQdwly2k=";
              };
              nativeBuildInputs = [ pkgs.autoPatchelfHook ];
              buildInputs = [ pkgs.stdenv.cc.cc.lib ];
              dependencies = [ numpy ];
              pythonImportsCheck = [ "triangle" ];
              meta.license = pkgs.lib.licenses.unfree;
            };
          in [
            numpy scipy matplotlib pandas triangle pybind11 pyjulia
            setuptools pip wheel meson-python pytest pillow charset-normalizer
            (jax.override { cudaSupport = cuda; })
          ]);
          command = name: script: pkgs.writeShellScriptBin name ''
            root="$(git rev-parse --show-toplevel 2>/dev/null)"
            if [ ! -f "$root/flake.nix" ] || [ ! -f "$root/scripts/${script}" ]; then
              echo "Run ${name} from inside the computational_physics_isp checkout." >&2
              exit 1
            fi
            exec ${python}/bin/python "$root/scripts/${script}" "$@"
          '';
        in pkgs.mkShell {
          name = "comphys-${if cuda then "cuda" else "cpu"}";
          packages = [
            python pkgs.gcc pkgs.gfortran pkgs.gnumake pkgs.pkg-config
            pkgs.meson pkgs.ninja pkgs.cmake pkgs.cargo pkgs.rustc
            pkgs.maturin pkgs.julia_110-bin pkgs.git pkgs.ffmpeg-headless pkgs.util-linux
            (command "comphys-build" "build.py")
            (command "comphys-check" "check_environment.py")
          ];
          # LAPACK calls in the Fortran sources use 32-bit INTEGER (LP64).
          buildInputs = [ pkgs.openblasCompat ];
          # Avoid ABI contamination from user-installed Python packages.
          PYTHONNOUSERSITE = "1";
          PYO3_PYTHON = "${python}/bin/python";
          COMPHYS_PYTHON = "${python}/bin/python";
          COMPHYS_JULIA = "${pkgs.julia_110-bin}/bin/julia";
          CC = "gcc";
          CXX = "g++";
          FC = "gfortran";
          # A dev shell has no install output. Avoid its synthetic RPATH, which
          # also breaks compiler-wrapper flag splitting in checkouts with spaces.
          NIX_NO_SELF_RPATH = "1";
          shellHook = ''
            export COMPHYS_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
            export JULIA_PROJECT="$COMPHYS_ROOT/nix/julia"
            # Isolate PyCall's Python binding from the user's normal Julia depot.
            export JULIA_DEPOT_PATH="$COMPHYS_ROOT/.cache/julia:"
            export PYTHON="${python}/bin/python"
            export JULIA_PYTHONCALL_EXE="$PYTHON"
            export MPLBACKEND="''${MPLBACKEND:-Agg}"
            export MPLCONFIGDIR="$COMPHYS_ROOT/.cache/matplotlib"
            export JULIA_NUM_THREADS="''${JULIA_NUM_THREADS:-1}"
            export OMP_NUM_THREADS="''${OMP_NUM_THREADS:-1}"
            export OPENBLAS_NUM_THREADS="''${OPENBLAS_NUM_THREADS:-1}"
            ${nixpkgs.lib.optionalString (!cuda) ''export JAX_PLATFORMS=cpu''}
            ${nixpkgs.lib.optionalString cuda ''
              unset JAX_PLATFORMS
              export LD_LIBRARY_PATH="/run/opengl-driver/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            ''}
            if [ -n "''${VIRTUAL_ENV:-}" ] || [ -n "''${CONDA_PREFIX:-}" ]; then
              echo "An outer Python environment is active; exit it before using this shell." >&2
            fi
            echo "comphys: run comphys-build, then comphys-check --built. See docs/NIX.md."
          '';
        };
    in {
      devShells = forAllSystems (system: {
        default = shellFor system false;
      } // nixpkgs.lib.optionalAttrs (system == "x86_64-linux") {
        cuda = shellFor system true;
      });
    };
}
