"""OCR through Tesseract, as an alternative to the GLM-OCR model.

The two backends are not competitors so much as different trades, and on this
machine the difference is stark:

                      RAM        one page      understands layout
    GLM-OCR         ~1.4 GB      ~30 s         yes - tables, columns, headings
    Tesseract       ~50 MB       <1 s          no - lines of text, in order

So Tesseract is the right default for "read the text off this screenshot", and
GLM-OCR earns its cost when the page has structure worth preserving. Neither is
strictly better, which is why this is a switch rather than a replacement.

Tesseract is called as a subprocess rather than through pytesseract. The
wrapper's job is exactly this - build an argument list, run the binary, read
stdout - and the project already starts and supervises llama-server the same
way. A dependency to avoid twenty lines would not pay for itself.

VERIFIED against Tesseract 5.5.3: `samples/note.png` transcribes correctly in
0.56 s and `samples/table.png` in 0.44 s, against roughly 30 s for the same
images through GLM-OCR. The table comes back as plain rows with no structure,
exactly as the trade above says it will.

Two things that only showed up against the real binary, and are worth knowing
before changing this file:

  * A **relative path** must be made absolute first. `cwd` is set to the
    image's own folder, so "samples/note.png" would be resolved against it
    twice and Tesseract would report only "Error during processing". A stub
    never opens the file, so this was invisible until it ran for real - the
    stub in the tests now opens it too, and there is a regression test.
  * Some builds ship **no language data at all**. Scoop's does: the binary
    installs, `--list-langs` returns nothing, and every read fails with
    "Could not initialize tesseract." The fix is one `eng.traineddata` file in
    the tessdata folder, and the error message here says so.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

# Where Tesseract puts itself on Windows when nobody adds it to PATH, which is
# the default in its own installer. Checking these turns "command not found"
# into "found it, using it" for the common case.
WINDOWS_LOCATIONS = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
)

# Page segmentation mode. 3 is "fully automatic, no orientation detection",
# which is right for a screenshot or a photographed page. 6 ("a single uniform
# block") is the one to try when 3 scrambles a simple image.
DEFAULT_PSM = 3
DEFAULT_LANGUAGE = "eng"

# Tesseract is fast - under a second for a screenshot - so a page that takes
# this long has gone wrong rather than being slow.
DEFAULT_TIMEOUT = 120.0


class TesseractError(Exception):
    """Tesseract is missing, or could not read the image."""


@dataclass(frozen=True)
class TesseractInfo:
    """What was found on this machine."""

    path: str
    version: str
    languages: tuple[str, ...] = ()

    @property
    def summary(self) -> str:
        languages = ", ".join(self.languages[:8]) or "unknown"
        return f"Tesseract {self.version} at {self.path} (languages: {languages})"


def find_tesseract(configured: str = "") -> str:
    """Locate the Tesseract binary, or return "".

    Order: what the config says, then PATH, then the places the Windows
    installer uses. The last step is what makes this work for someone who
    installed Tesseract and never touched their PATH.
    """
    if configured:
        candidate = Path(configured).expanduser()
        if candidate.is_file():
            return str(candidate)
        # A bare command name in the config is still worth resolving on PATH.
        found = shutil.which(configured)
        if found:
            return found
        return ""

    found = shutil.which("tesseract")
    if found:
        return found

    for location in WINDOWS_LOCATIONS:
        if Path(location).is_file():
            return location
    return ""


def probe(command: str = "", *, timeout: float = 15.0) -> TesseractInfo | None:
    """Check that Tesseract runs, and report its version and languages.

    Returns None when it is not installed, which is a supported state: the OCR
    tool then says so and names the other backend, rather than failing at the
    moment someone attaches an image.
    """
    path = find_tesseract(command)
    if not path:
        return None

    try:
        result = subprocess.run(
            [path, "--version"],
            capture_output=True,
            text=True,
            timeout=timeout,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError):
        return None

    # Version goes to stdout on some builds and stderr on others.
    output = (result.stdout or "") + (result.stderr or "")
    first = output.strip().splitlines()[0] if output.strip() else ""
    version = first.replace("tesseract", "").strip() or "unknown"

    return TesseractInfo(path=path, version=version, languages=_languages(path, timeout))


def _languages(path: str, timeout: float) -> tuple[str, ...]:
    """Installed language packs. Empty when they cannot be listed."""
    try:
        result = subprocess.run(
            [path, "--list-langs"],
            capture_output=True,
            text=True,
            timeout=timeout,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError):
        return ()

    output = (result.stdout or "") + (result.stderr or "")
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    # The first line is a header ("List of available languages (4):").
    return tuple(line for line in lines if " " not in line and ":" not in line)


class TesseractBackend:
    """Reads images by shelling out to Tesseract."""

    def __init__(
        self,
        *,
        command: str = "",
        language: str = DEFAULT_LANGUAGE,
        psm: int = DEFAULT_PSM,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.command = command
        self.language = language or DEFAULT_LANGUAGE
        self.psm = int(psm)
        self.timeout = float(timeout)
        self._resolved: str | None = None

    # --- availability ---

    def path(self) -> str:
        """The binary's path, resolved once and remembered."""
        if self._resolved is None:
            self._resolved = find_tesseract(self.command)
        return self._resolved

    def available(self) -> bool:
        return bool(self.path())

    def info(self) -> TesseractInfo | None:
        return probe(self.command)

    def missing_message(self) -> str:
        """What to tell someone who has selected a backend that is not there."""
        return (
            "Tesseract is not installed, or not where this can find it. "
            "Install it from https://github.com/UB-Mannheim/tesseract/wiki "
            "(Windows) or your package manager, then either add it to PATH or "
            "set TESSERACT_CMD to the full path of tesseract.exe. "
            "Alternatively switch the OCR backend back to the GLM-OCR model."
        )

    # --- reading ---

    def read(self, path: Path, *, language: str = "", psm: int | None = None) -> str:
        """Return the text in `path`.

        The caller has already validated the path against the workspace jail
        and the size limit; this only runs the binary.
        """
        binary = self.path()
        if not binary:
            raise TesseractError(self.missing_message())

        # Absolute, and resolved. `cwd` below is set to the image's own folder,
        # so a relative path here would be resolved against it twice -
        # "samples/note.png" becomes "samples/samples/note.png" and Tesseract
        # reports only "Error during processing". A stub binary never opens the
        # file, so this is invisible until it runs against the real thing.
        try:
            image = path.resolve()
        except OSError:
            image = path.absolute()

        # "stdout" is Tesseract's own keyword for the output file, not a path.
        command = [
            binary,
            str(image),
            "stdout",
            "-l",
            language or self.language,
            "--psm",
            str(self.psm if psm is None else int(psm)),
        ]

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                # Tesseract writes temporary files beside its input otherwise.
                cwd=str(path.parent),
                env={**os.environ, "OMP_THREAD_LIMIT": "2"},
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except subprocess.TimeoutExpired:
            raise TesseractError(
                f"Tesseract did not finish within {self.timeout:.0f}s on "
                f"{path.name}."
            ) from None
        except OSError as exc:
            raise TesseractError(f"Could not run Tesseract: {exc}") from None

        if result.returncode != 0:
            detail = (result.stderr or "").strip().splitlines()
            reason = detail[-1] if detail else f"exit code {result.returncode}"
            if "Failed loading language" in (result.stderr or ""):
                reason += (
                    f". Install the '{language or self.language}' language pack, "
                    f"or set TESSERACT_LANG to one that is installed."
                )
            raise TesseractError(f"Tesseract could not read {path.name}: {reason}")

        text = (result.stdout or "").strip()
        if not text:
            raise TesseractError(
                f"Tesseract found no text in {path.name}. If the image is a "
                f"photograph or is low contrast, the GLM-OCR model backend "
                f"handles those better."
            )
        return text
