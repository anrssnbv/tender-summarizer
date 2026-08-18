"""Shared pytest fixtures.

PDF fixtures are generated on the fly with PyMuPDF rather than committed
as binary files -- the working rules forbid committing test PDFs, and
generated fixtures are also easier to parameterize (page count, which
pages are "scanned", document length for chunking tests).
"""

from __future__ import annotations

from collections.abc import Callable

import fitz
import pytest


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guarantee the test suite never reaches a real external API.

    Clears provider API keys so a misconfigured test can't accidentally
    pick up real credentials from the developer's shell, and makes any
    outbound httpx call raise immediately. Provider-specific tests that
    need to exercise real request-building opt back in with their own
    httpx.MockTransport, which bypasses this by not calling `.send`.
    """
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(key, raising=False)

    import httpx

    async def _blocked_send(self, request, **kwargs):
        raise RuntimeError(
            f"network access is not allowed in tests (attempted: {request.method} {request.url})"
        )

    monkeypatch.setattr(httpx.AsyncClient, "send", _blocked_send)


PdfFactory = Callable[[list[str | None]], bytes]


@pytest.fixture
def make_pdf() -> PdfFactory:
    """Build a PDF from a list of pages.

    Each entry in `pages` is either:
      - a string: rendered as real, extractable text (Cyrillic-safe --
        uses insert_htmlbox, whose Story engine does font fallback;
        PyMuPDF's base-14 fonts do NOT cover Cyrillic glyphs), or
      - None: rendered as a blank grayscale image with no text layer,
        simulating a scanned page.
    """

    def _make(pages: list[str | None]) -> bytes:
        doc = fitz.open()
        for content in pages:
            page = doc.new_page()
            if content is None:
                pix = fitz.Pixmap(fitz.csGRAY, fitz.IRect(0, 0, 80, 100), False)
                pix.clear_with(200)
                page.insert_image(page.rect, pixmap=pix)
            else:
                page.insert_htmlbox(page.rect, f"<p>{content}</p>")
        data = doc.tobytes()
        doc.close()
        return data

    return _make


@pytest.fixture
def empty_pdf_bytes() -> bytes:
    """A structurally valid PDF with zero pages.

    Hand-built rather than produced with `fitz.open().tobytes()` --
    PyMuPDF refuses to *save* a document with zero pages (`ValueError:
    cannot save with zero pages`), but its lenient parser opens one
    that already exists on disk (e.g. produced by a broken export
    pipeline), which is exactly the real-world case this fixture
    stands in for.
    """
    return (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [] /Count 0 >>\nendobj\n"
        b"trailer\n<< /Size 3 /Root 1 0 R >>\n"
        b"%%EOF\n"
    )


@pytest.fixture
def not_a_pdf_bytes() -> bytes:
    return b"this is definitely not a pdf file, just plain bytes"
