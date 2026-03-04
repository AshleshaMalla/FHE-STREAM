{
  description = "Nix flake for FHE RaiderSTREAM - Benchmarking FHE kernels";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
        };

        # On Darwin and Linux, OpenMP can be provided by llvmPackages.openmp.
        openmp = pkgs.llvmPackages.openmp;

        # OpenFHE is NOT in nixpkgs, so we define it here.
        openfhe = pkgs.stdenv.mkDerivation rec {
          pname = "openfhe";
          version = "1.2.1";

          src = pkgs.fetchFromGitHub {
            owner = "openfheorg";
            repo = "openfhe-development";
            rev = "v${version}";
            hash = "sha256-Rj05zuDKA9liNzfglyhPDT/a7s3vM3mAdJzbsaiTN3Q=";
          };

          nativeBuildInputs = [ pkgs.cmake pkgs.git ];
          buildInputs = [ pkgs.gmp pkgs.ntl pkgs.cereal openmp ];

          postPatch = ''
            mkdir -p third-party/cereal
            cp -r ${pkgs.cereal}/include third-party/cereal/
          '';

          cmakeFlags = [
            "-DBUILD_UNITTESTS=OFF"
            "-DBUILD_EXAMPLES=OFF"
            "-DBUILD_BENCHMARKS=OFF"
          ];
        };

        gbenchmark = pkgs.gbenchmark;

      in {
        packages.default = pkgs.stdenv.mkDerivation {
          pname = "fhe_raiderstream";
          version = "0.1.0";

          src = ./.;

          nativeBuildInputs = [
            pkgs.cmake
          ];

          buildInputs = [
            openfhe
            gbenchmark
            openmp
            pkgs.openmpi
          ];

          cmakeFlags = [
            "-DCMAKE_BUILD_TYPE=Release"
            "-DBUILD_STATIC=OFF"
          ];

          installPhase = ''
            runHook preInstall
            mkdir -p $out/bin
            cp fhe_raiderstream $out/bin/
            runHook postInstall
          '';

          meta = with pkgs.lib; {
            description = "FHE RaiderSTREAM: Benchmarking FHE kernels";
            homepage = "https://github.com/tyush/FHERaiderSTREAM";
            platforms = platforms.all;
          };
        };

        devShells.default = pkgs.mkShell {
          buildInputs = [
            pkgs.cmake
            pkgs.python315
            openfhe
            gbenchmark
            openmp
            pkgs.openmpi
          ];

          shellHook = ''
            echo "FHE RaiderSTREAM development environment loaded."
            echo "Run './build.sh --run' to build and benchmark."
          '';
        };
      }
    );
}
