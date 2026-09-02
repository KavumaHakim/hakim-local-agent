"""Fetch what dictation and reading aloud need.

Three separate things, and they are separate because they fail separately:

    vendor/whisper/   a whisper.cpp build      ~8 MB, per platform
    whisper/          a ggml speech model      78-148 MB
    tts/              a Piper voice            ~63 MB

Text-to-speech needs no binary at all - `piper-tts` is a wheel in
`requirements.txt` - so a voice file is the whole of it. Speech-to-text needs
a platform build, which is why this exists rather than being a pip install.

The downloading, unpacking and path-traversal checks are `get_llama.py`'s,
imported rather than copied. That module's `download` grew HTTP `Range` resume
after a 35 MB archive failed three times running on a domestic connection, and
a second copy of that logic would be a second place to fix it.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Aliased at the boundary: it is the same fetcher, and calling a whisper
# failure a LlamaError in an error message would be a small lie.
from get_llama import (  # noqa: E402
    LlamaError as FetchError,
    download,
    make_executable,
    say,
    unpack,
)

ROOT = Path(__file__).resolve().parent.parent

VENDOR = ROOT / "vendor" / "whisper"
WHISPER_MODELS = ROOT / "whisper"
VOICES = ROOT / "tts"

RELEASES = "https://api.github.com/repos/ggml-org/whisper.cpp/releases/latest"
DOWNLOAD = "https://github.com/ggml-org/whisper.cpp/releases/download"

# Unlike llama.cpp, whisper.cpp's binary releases are ordinary releases rather
# than prereleases, so `/releases/latest` finds them and there is no paging
# through history. Pinned anyway, because "latest" changes under you and a
# setup that worked yesterday should work today.
DEFAULT_BUILD = "b4938"

# The asset for each platform, from release b4938's own list. There is no
# macOS build: whisper.cpp ships an xcframework for Xcode and expects everyone
# else to build it, which is stated rather than papered over.
ASSETS = {
    ("windows", "x64"): "whisper-bin-x64.zip",
    ("windows", "x86"): "whisper-bin-Win32.zip",
    ("linux", "x64"): "whisper-bin-ubuntu-x64.tar.gz",
    ("linux", "arm64"): "whisper-bin-ubuntu-arm64.tar.gz",
}

# Speech models, smallest first. `base.en` is the recommendation on hardware
# like this: `tiny.en` is half the size and noticeably worse at anything but
# clear speech, and `small.en` triples the time for a dictated sentence.
MODEL_HOST = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main"
MODELS = {
    "tiny.en": 77_700_000,
    "base.en": 148_000_000,
    "small.en": 488_000_000,
    "tiny": 77_700_000,
    "base": 148_000_000,
    "small": 488_000_000,
}
DEFAULT_MODEL = "base.en"

# Piper voices. The path is the voice's own name taken apart - language,
# locale, speaker, quality - which is how the repository is laid out.
VOICE_HOST = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
VOICES_AVAILABLE = {
    "en_US-lessac-medium": ("en", "en_US", "lessac", "medium", 63_200_000),
    "en_US-amy-medium": ("en", "en_US", "amy", "medium", 63_200_000),
    "en_US-ryan-high": ("en", "en_US", "ryan", "high", 121_000_000),
    "en_GB-alba-medium": ("en", "en_GB", "alba", "medium", 63_200_000),
}
DEFAULT_VOICE = "en_US-lessac-medium"


def platform_tokens() -> tuple[str, str]:
    """The (system, architecture) this machine needs a build for."""
    import platform

    machine = platform.machine().lower()
    if machine in ("arm64", "aarch64"):
        architecture = "arm64"
    elif machine in ("x86", "i386", "i686"):
        architecture = "x86"
    else:
        architecture = "x64"

    system = platform.system().lower()
    if system not in ("windows", "linux", "darwin"):
        raise FetchError(f"Unsupported system: {platform.system()}")
    return system, architecture


def asset_for(system: str, architecture: str) -> str:
    """The release asset this machine needs, or a plain explanation."""
    name = ASSETS.get((system, architecture))
    if name:
        return name
    if system == "darwin":
        raise FetchError(
            "whisper.cpp publishes no macOS binary - only an xcframework for "
            "Xcode. Build it (`brew install whisper-cpp`, or make -j from the "
            "repository) and put whisper-cli on PATH; everything else here "
            "works unchanged."
        )
    raise FetchError(f"No whisper.cpp build for {system}-{architecture}.")


def find_cli(directory: Path) -> Path | None:
    """The whisper-cli inside an unpacked build, wherever it landed."""
    if not directory.is_dir():
        return None
    for name in ("whisper-cli.exe", "whisper-cli", "main.exe", "main"):
        for found in sorted(directory.rglob(name)):
            if found.is_file():
                return found
    return None


def install_whisper(*, build: str = DEFAULT_BUILD, force: bool = False) -> Path | None:
    """Download and unpack a whisper.cpp build. Returns the CLI path."""
    existing = find_cli(VENDOR)
    if existing and not force:
        say(f"  already there: {existing}")
        return existing

    system, architecture = platform_tokens()
    name = asset_for(system, architecture)
    url = f"{DOWNLOAD}/{build}/{name}"
    say(f"  {name}  ({build})")

    if VENDOR.exists() and force:
        shutil.rmtree(VENDOR, ignore_errors=True)

    with tempfile.TemporaryDirectory() as scratch:
        archive = Path(scratch) / name
        # 0 rather than a guessed size: the server reports it, and claiming a
        # total that turns out to be wrong is how a good download gets thrown
        # away for being "truncated".
        download(url, archive, 0)
        say("  unpacking ...")
        # Flattened, because the Windows archive puts everything under
        # Release/ and the Linux one does not - and `speech/whisper.py` looks
        # for the binary directly in vendor/whisper.
        with tempfile.TemporaryDirectory() as staging:
            unpack(archive, Path(staging))
            VENDOR.mkdir(parents=True, exist_ok=True)
            for path in Path(staging).rglob("*"):
                if path.is_file():
                    shutil.copy2(path, VENDOR / path.name)

    cli = find_cli(VENDOR)
    if cli is None:
        raise FetchError(
            f"The archive unpacked but contained no whisper-cli. Look in "
            f"{VENDOR} and report this."
        )
    make_executable(cli)
    say(f"  {cli}")
    return cli


def install_model(name: str = DEFAULT_MODEL, *, force: bool = False) -> Path | None:
    """Download a ggml speech model into whisper/."""
    if name not in MODELS:
        raise FetchError(
            f"Unknown model {name!r}. Known: {', '.join(sorted(MODELS))}."
        )

    WHISPER_MODELS.mkdir(parents=True, exist_ok=True)
    target = WHISPER_MODELS / f"ggml-{name}.bin"
    if target.is_file() and not force:
        say(f"  already there: {target.name}")
        return target

    say(f"  ggml-{name}.bin")
    download(f"{MODEL_HOST}/ggml-{name}.bin", target, MODELS[name])
    return target


def install_voice(name: str = DEFAULT_VOICE, *, force: bool = False) -> Path | None:
    """Download a Piper voice, and the config it cannot load without."""
    if name not in VOICES_AVAILABLE:
        raise FetchError(
            f"Unknown voice {name!r}. Known: {', '.join(sorted(VOICES_AVAILABLE))}. "
            f"Any voice from huggingface.co/rhasspy/piper-voices works - put "
            f"the .onnx and its .onnx.json in {VOICES}."
        )

    language, locale, speaker, quality, size = VOICES_AVAILABLE[name]
    VOICES.mkdir(parents=True, exist_ok=True)
    target = VOICES / f"{name}.onnx"
    config = VOICES / f"{name}.onnx.json"

    if target.is_file() and config.is_file() and not force:
        say(f"  already there: {target.name}")
        return target

    base = f"{VOICE_HOST}/{language}/{locale}/{speaker}/{quality}/{name}.onnx"
    say(f"  {name}.onnx")
    download(base, target, size)
    # The .json carries the phoneme map and the sample rate. Without it the
    # voice will not load at all, so it is not an optional extra.
    say(f"  {name}.onnx.json")
    download(f"{base}.json", config, 0)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch what dictation and reading aloud need."
    )
    parser.add_argument(
        "--what",
        default="all",
        choices=("all", "whisper", "model", "voice"),
        help="which part to fetch (default: all of them)",
    )
    parser.add_argument(
        "--build", default=DEFAULT_BUILD, help="whisper.cpp release tag to pin"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=sorted(MODELS),
        help="which speech model to fetch",
    )
    parser.add_argument(
        "--voice",
        default=DEFAULT_VOICE,
        choices=sorted(VOICES_AVAILABLE),
        help="which Piper voice to fetch",
    )
    parser.add_argument(
        "--force", action="store_true", help="re-download even when it is there"
    )
    arguments = parser.parse_args()

    wanted = arguments.what
    try:
        if wanted in ("all", "whisper"):
            say("whisper.cpp:")
            install_whisper(build=arguments.build, force=arguments.force)
        if wanted in ("all", "model"):
            say("speech model:")
            install_model(arguments.model, force=arguments.force)
        if wanted in ("all", "voice"):
            say("voice:")
            install_voice(arguments.voice, force=arguments.force)
    except FetchError as exc:
        say(f"[X] {exc}")
        return 1
    except KeyboardInterrupt:
        say("\nStopped.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
