# Generated from packaging/homebrew/drama-rs.rb in apiplant/drama-rs by the
# release workflow, which fills in the version and checksum and commits the
# result to apiplant/homebrew-tap as Formula/drama-rs.rb. Changes belong in
# the source repository: the next release overwrites this file.
class DramaRs < Formula
  desc "Fish Audio Drama 3 text-to-speech on CUDA or Apple Silicon (MLX)"
  homepage "https://github.com/apiplant/drama-rs"
  version "@VERSION@"
  license "Apache-2.0"

  # The release archives are the binaries; no source build or bottles.
  on_macos do
    depends_on "ffmpeg"
    depends_on arch: :arm64
    depends_on macos: :sequoia
    on_arm do
      url "https://github.com/apiplant/drama-rs/releases/download/v@VERSION@/drama-rs-v@VERSION@-aarch64-apple-darwin.tar.gz"
      sha256 "@SHA_MACOS_ARM64@"
    end
  end
  on_linux do
    depends_on arch: :x86_64
    on_intel do
      url "https://github.com/apiplant/drama-rs/releases/download/v@VERSION@/drama-rs-v@VERSION@-x86_64-unknown-linux-gnu.tar.gz"
      sha256 "@SHA_LINUX_X86_64@"
    end
  end

  def install
    bin.install "drama"
    (share/"drama-rs").install "mlx.metallib" if OS.mac?
    doc.install "README.md"
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/drama --version")
  end
end
