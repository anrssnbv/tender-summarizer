"""PDF text extraction and scanned-document detection.

The scan classifier (:func:`classify`) is a pure function over already
extracted per-page strings. It has no dependency on PyMuPDF or any file
I/O, which is what makes every threshold in this module unit-testable
without a single PDF fixture.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from enum import Enum

import fitz  # PyMuPDF


class ExtractionStatus(str, Enum):
    """Outcome of classifying a document's extracted text."""

    OK = "ok"
    """Text-bearing throughout (or close to it). Safe to summarize."""

    PARTIAL = "partial"
    """Mostly text-bearing, but some pages appear to be scans/images.
    Safe to summarize, but the caller should attach a confidence note."""

    EMPTY_DOCUMENT = "empty_document"
    """The PDF has zero pages."""

    INSUFFICIENT_TEXT = "insufficient_text"
    """Too little text overall (typically a 1-2 page document with no
    real content, or a single-page scan) to summarize meaningfully."""

    SCANNED_DOCUMENT = "scanned_document"
    """Most pages carry no usable text layer -- this is a scan and
    requires OCR, which is out of scope."""


# A full A4 page of Russian legal/procurement text runs roughly
# 1,500-3,500 characters. A scanned page that happens to carry a text
# layer only for a header, footer, or stamp yields on the order of
# 20-120 characters. 200 sits comfortably above that noise floor and
# comfortably below any genuine content page.
TEXT_PAGE_MIN_CHARS = 200

# Below this total (across the whole document), the ratio-based checks
# degenerate on very short documents (1-2 pages), so an absolute floor
# is needed. This also catches a 30-page document where only a title
# page carries text.
MIN_TOTAL_MEANINGFUL_CHARS = 500

# Below this page count, a text-bearing ratio is not a meaningful
# signal (it can only be 0, 0.5, or 1.0) -- PLAN.md's own framing of
# scan detection is "many pages but almost no text", which presumes
# enough pages for a ratio to say something. Below this count, only
# the absolute character floor decides.
MIN_PAGES_FOR_RATIO_CHECK = 3

# Fraction of pages that must be text-bearing for the document to be
# usable at all.
SCANNED_RATIO_THRESHOLD = 0.30

# Fraction of pages that must be text-bearing for the document to be
# considered fully OK (below this, up to SCANNED_RATIO_THRESHOLD, it
# is PARTIAL: usable, but flagged).
PARTIAL_RATIO_THRESHOLD = 0.80

# If more than this fraction of a page's characters are replacement
# characters or private-use-area glyphs, the page's "text" is really a
# broken CID font mapping and should not count as real content.
GARBAGE_CHAR_RATIO_THRESHOLD = 0.10

# If fewer than this fraction of a document's letters are Cyrillic, we
# flag it (some annexes are legitimately non-Russian) but never fail
# outright -- this is not a scan-detection signal, just a heads-up.
MIN_CYRILLIC_LETTER_RATIO = 0.30


@dataclass(frozen=True)
class ExtractedDocument:
    """Per-page text extracted from a PDF, plus the classification of
    whether that text is usable for summarization."""

    pages: tuple[str, ...]
    status: ExtractionStatus
    text_bearing_pages: tuple[int, ...]  # 1-based page numbers
    notes: tuple[str, ...]

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def full_text(self) -> str:
        return "\n\n".join(self.pages)


class PdfExtractionError(Exception):
    """Raised when the input bytes are not a PDF PyMuPDF can open."""


def _meaningful_chars(text: str) -> int:
    """Count letters and digits, ignoring whitespace/punctuation, after
    Unicode normalization so composed/decomposed forms count the same."""
    normalized = unicodedata.normalize("NFKC", text)
    return sum(1 for ch in normalized if unicodedata.category(ch)[0] in ("L", "N"))


REPLACEMENT_CHAR_CODEPOINT = 0xFFFD
PRIVATE_USE_AREA_RANGE = (0xE000, 0xF8FF)


def _is_garbage(text: str) -> bool:
    """Detect broken CID font maps: extraction "succeeds" but yields
    replacement characters or private-use-area glyphs instead of text.
    Codepoints are compared numerically (not as literal characters) to
    keep this source file plain ASCII."""
    if not text:
        return True
    lo, hi = PRIVATE_USE_AREA_RANGE
    bad = sum(1 for ch in text if ord(ch) == REPLACEMENT_CHAR_CODEPOINT or lo <= ord(ch) <= hi)
    return bad / len(text) > GARBAGE_CHAR_RATIO_THRESHOLD


def _is_text_bearing(text: str) -> bool:
    return _meaningful_chars(text) >= TEXT_PAGE_MIN_CHARS and not _is_garbage(text)


# Unicode Cyrillic block (basic + supplement), by codepoint rather than
# literal character bounds to keep this source file plain ASCII.
CYRILLIC_BLOCK_RANGE = (0x0400, 0x04FF)


def _cyrillic_ratio(pages: tuple[str, ...]) -> float | None:
    lo, hi = CYRILLIC_BLOCK_RANGE
    letters = cyrillic = 0
    for page in pages:
        for ch in page:
            if unicodedata.category(ch)[0] == "L":
                letters += 1
                if lo <= ord(ch) <= hi:
                    cyrillic += 1
    if letters == 0:
        return None
    return cyrillic / letters


def classify(pages: list[str]) -> ExtractedDocument:
    """Classify already-extracted per-page text as usable, partially
    usable, or a scan requiring OCR.

    Pure function: takes plain strings, not a PDF. See module docstring.
    """
    notes: list[str] = []
    pages_t = tuple(pages)
    n = len(pages_t)

    if n == 0:
        return ExtractedDocument(
            pages=pages_t,
            status=ExtractionStatus.EMPTY_DOCUMENT,
            text_bearing_pages=(),
            notes=("Документ не содержит страниц.",),
        )

    text_bearing = tuple(i + 1 for i, p in enumerate(pages_t) if _is_text_bearing(p))
    total_meaningful = sum(_meaningful_chars(p) for p in pages_t)
    ratio = len(text_bearing) / n

    insufficient = ExtractedDocument(
        pages=pages_t,
        status=ExtractionStatus.INSUFFICIENT_TEXT,
        text_bearing_pages=text_bearing,
        notes=(
            "В документе недостаточно текста для анализа "
            f"({total_meaningful} значащих символов).",
        ),
    )
    scanned = ExtractedDocument(
        pages=pages_t,
        status=ExtractionStatus.SCANNED_DOCUMENT,
        text_bearing_pages=text_bearing,
        notes=(
            "Похоже, документ является сканом без текстового слоя "
            "(требуется OCR, что выходит за рамки текущей реализации).",
        ),
    )

    if n < MIN_PAGES_FOR_RATIO_CHECK:
        # Too few pages for a ratio to mean anything (see
        # MIN_PAGES_FOR_RATIO_CHECK) -- the absolute floor alone
        # decides. This deliberately never returns SCANNED_DOCUMENT:
        # a 1-2 page document that is merely short is not a scan.
        if total_meaningful < MIN_TOTAL_MEANINGFUL_CHARS:
            return insufficient
    else:
        if ratio < SCANNED_RATIO_THRESHOLD:
            return scanned
        if total_meaningful < MIN_TOTAL_MEANINGFUL_CHARS:
            # Rare in practice once ratio >= 0.30 (see module notes),
            # but kept as a defensive fallback.
            return insufficient

    cyr_ratio = _cyrillic_ratio(pages_t)
    if cyr_ratio is not None and cyr_ratio < MIN_CYRILLIC_LETTER_RATIO:
        notes.append(
            "Значительная часть текста документа не на русском языке; "
            "качество извлечения может быть ниже ожидаемого."
        )

    if ratio < PARTIAL_RATIO_THRESHOLD:
        missing = sorted(set(range(1, n + 1)) - set(text_bearing))
        notes.append(
            "Страницы без текстового слоя (вероятно, сканы или изображения): "
            + ", ".join(str(p) for p in missing)
            + ". Данные с этих страниц не извлечены."
        )
        return ExtractedDocument(
            pages=pages_t,
            status=ExtractionStatus.PARTIAL,
            text_bearing_pages=text_bearing,
            notes=tuple(notes),
        )

    return ExtractedDocument(
        pages=pages_t,
        status=ExtractionStatus.OK,
        text_bearing_pages=text_bearing,
        notes=tuple(notes),
    )


def extract_text(file_bytes: bytes) -> ExtractedDocument:
    """Extract per-page text from PDF bytes and classify the result.

    Raises:
        PdfExtractionError: if ``file_bytes`` is not a PDF PyMuPDF can
            open (corrupt file, wrong format, etc).
    """
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:  # PyMuPDF raises its own exception types
        raise PdfExtractionError(f"Could not open file as PDF: {exc}") from exc

    try:
        pages = [page.get_text() for page in doc]
    finally:
        doc.close()

    return classify(pages)
