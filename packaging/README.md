# Packaging

One package, `drama-rs`, carrying the `drama` binary. Linux x86_64 packages use CUDA and
need an NVIDIA GPU with 24 GB (Ampere or newer). Apple Silicon archives use MLX on macOS 15+. The `packages`, `homebrew`, `apt` and `pacman` jobs in
[`.github/workflows/release.yml`](../.github/workflows/release.yml) substitute
the `@VERSION@`, `@SHA_*@`, `@ARCH@` and `@ARCHIVE_PREFIX@` placeholders with
the tag's version and the checksums of the archives that release just built,
then publish the result. None of the definitions compiles the project — they
install the binary the build jobs already produced. The `crates-io` job
publishes the crate (library + binary) with `cargo publish`.

| File | Publishes to |
| --- | --- |
| `homebrew/drama-rs.rb` | `apiplant/homebrew-tap`, as `Formula/drama-rs.rb` (Linux x86_64 and macOS arm64) |
| `pacman/PKGBUILD` | the release itself, as a `.pkg.tar.zst` asset, and `apiplant/pacman` |
| `debian/control` | the release itself, as a `.deb` asset, and `apiplant/apt` |
| `apt/apt-ftparchive.conf` | `apiplant/apt`, served at `apt.apiplant.com` |

This repository reuses the shared `apiplant` repositories (`apiplant/homebrew-tap`,
`apiplant/pacman` served at `apiplant.github.io/pacman`, `apiplant/apt` served
at `apt.apiplant.com`) and the same secrets as the other apiplant releases:
`HOMEBREW_TAP_TOKEN`, `CARGO_REGISTRY_TOKEN`, `PACMAN_REPO_TOKEN` +
`PACMAN_GPG_PRIVATE_KEY`/`PACMAN_GPG_PASSPHRASE`, `APT_REPO_TOKEN` +
`APT_GPG_PRIVATE_KEY`/`APT_GPG_PASSPHRASE`. They are copied here by the `Sync
release secrets` workflow in `apiplant/gliner-rs`; run it once (with `dry_run`
off) if this repository doesn't have them yet. A publish job whose credential is
absent is skipped rather than failing the release.

## Builds

- `ui`: `pnpm build` in `ui/`, uploaded as the `ui-dist` artifact. `binary`,
  `binary-arch` and `crates-io` download it into `ui/dist` before building, and
  `DRAMA_REQUIRE_UI=1` makes build.rs fail instead of embedding no web UI. The
  crate ships `ui/dist`, so `cargo install` gets the UI without Node.
- `binary`: an `nvidia/cuda:12.8.1-devel-ubuntu22.04` container (nvcc and the
  CUDA libraries, no GPU needed), `cargo build --profile dist` for
  `DRAMA_CUDA_ARCH=sm_80,sm_86,sm_89,sm_90,sm_120` (machine code for each, PTX
  of the newest). The archive, the `.deb` and the Homebrew formula use it; it
  links the CUDA 12 runtime (`libcudart.so.12`, `libcublas.so.12`).
- `binary-arch`: the same build inside Arch against Arch's own `cuda`, used by
  the pacman package (Arch's CUDA has a different soname). If rolling Arch
  breaks that build, the pacman package falls back to the CUDA 12 archive.

The `.deb` depends on `libcuda1`, `libcudart12` and `libcublas12`; the Arch
package on `nvidia-utils` and `cuda`. Homebrew cannot depend on a driver, so
the host needs a CUDA 12 runtime itself. None of these can express "a driver
compatible with this toolkit" exactly.

## Cutting a release

Bump `version` in `Cargo.toml`, commit, then:

```bash
git tag v$(grep -m1 '^version' Cargo.toml | cut -d'"' -f2) && git push origin --tags
```

`workflow_dispatch` runs the build and packaging jobs without publishing
anything (a dry run).

## Using the packages

```bash
brew tap apiplant/tap && brew install apiplant/tap/drama-rs
```

Arch Linux (one-time setup, then `pacman -Syu` picks up new releases):

```bash
curl -sSfL https://apiplant.github.io/pacman/apiplant.gpg -o /tmp/apiplant.gpg
keyid=$(gpg --show-keys --with-colons /tmp/apiplant.gpg | awk -F: '/^pub:/ { print $5; exit }') && sudo pacman-key --add /tmp/apiplant.gpg && sudo pacman-key --finger "$keyid" && sudo pacman-key --lsign-key "$keyid"
printf '\n[apiplant]\nSigLevel = Required DatabaseOptional\nServer = https://apiplant.github.io/pacman/$arch\n' | sudo tee -a /etc/pacman.conf > /dev/null
sudo pacman -Sy drama-rs
```

Debian/Ubuntu (one-time setup, then `apt upgrade` picks up new releases):

```bash
curl -sSfL https://apt.apiplant.com/apiplant-archive-keyring.gpg | sudo tee /usr/share/keyrings/apiplant.gpg > /dev/null
echo "deb [signed-by=/usr/share/keyrings/apiplant.gpg] https://apt.apiplant.com stable main" | sudo tee /etc/apt/sources.list.d/apiplant.list > /dev/null
sudo apt update && sudo apt install drama-rs
```

Then fetch the weights (~21 GB): `hf download apiplant/drama-q4 --local-dir drama-q4`.

## Apple Silicon archives

The native `macos-15` job follows the `fluid-speech-rs` packaging convention: a target-named
`drama-rs-vVERSION-aarch64-apple-darwin.tar.gz` with a matching `.sha256` file, CLI `--version` smoke test,
and Homebrew `on_macos` / `on_arm` URL and checksum. It also runs the MLX library tests. Linux CUDA,
Debian, Arch and repository publication retain their existing paths.

MLX's native library is statically linked, but GPU execution also requires `mlx.metallib`. The job sets
`MLX_RS_METAL_PATH` to a workspace directory and packages its generated `mlx.metallib` alongside `drama`.
The runtime resolves the adjacent file before initializing Metal; the formula instead installs it under
`share/drama-rs`. This avoids depending on the build runner's home directory after installation.
