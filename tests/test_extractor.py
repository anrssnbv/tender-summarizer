"""Tests for app.pdf.extractor.

Two layers, as the design intends:
  - classify() tests: pure-function, no PDF involved at all. This is
    where every threshold and edge case is exercised.
  - extract_text() tests: a thin PyMuPDF-backed shell around classify(),
    checked against generated PDFs to confirm real extraction feeds it
    correctly (including that Cyrillic actually round-trips).
"""

from __future__ import annotations

import pytest

from app.pdf.extractor import (
    ExtractionStatus,
    PdfExtractionError,
    classify,
    extract_text,
)

RUSSIAN_PARAGRAPH = (
    "Настоящая тендерная документация регулирует порядок проведения "
    "закупок в рамках государственного заказа. Участник обязан "
    "предоставить банковскую гарантию в размере пяти процентов от "
    "суммы договора не позднее десяти календарных дней с момента "
    "подписания договора, а также обеспечить исполнение обязательств "
    "в соответствии с условиями настоящей документации и приложенным "
    "техническим заданием, включая полный перечень требований к "
    "участникам закупки."
)


class TestClassifyPureFunction:
    def test_empty_document(self):
        result = classify([])
        assert result.status is ExtractionStatus.EMPTY_DOCUMENT
        assert result.text_bearing_pages == ()

    def test_single_page_below_absolute_floor(self):
        # Real content, but not enough of it -- the ratio would be 1.0
        # (100% of pages are "text-bearing" by the per-page threshold
        # if it were low enough), so the absolute floor is what must
        # catch this, not the ratio.
        result = classify(["короткая страница"])
        assert result.status is ExtractionStatus.INSUFFICIENT_TEXT

    def test_fully_text_bearing_document(self):
        pages = [RUSSIAN_PARAGRAPH] * 5
        result = classify(pages)
        assert result.status is ExtractionStatus.OK
        assert result.text_bearing_pages == (1, 2, 3, 4, 5)
        assert result.notes == ()

    def test_fully_scanned_document(self):
        pages = [""] * 10
        result = classify(pages)
        assert result.status is ExtractionStatus.SCANNED_DOCUMENT
        assert result.text_bearing_pages == ()

    def test_mostly_scanned_below_ratio_threshold(self):
        # 2 of 10 pages have text -> ratio 0.2, below 0.30
        pages = [RUSSIAN_PARAGRAPH, RUSSIAN_PARAGRAPH] + [""] * 8
        result = classify(pages)
        assert result.status is ExtractionStatus.SCANNED_DOCUMENT

    def test_mixed_document_partial(self):
        # 7 of 10 pages have text -> ratio 0.7, in [0.30, 0.80) -> PARTIAL
        pages = [RUSSIAN_PARAGRAPH] * 7 + [""] * 3
        result = classify(pages)
        assert result.status is ExtractionStatus.PARTIAL
        assert result.text_bearing_pages == (1, 2, 3, 4, 5, 6, 7)
        assert any("8" in note and "9" in note and "10" in note for note in result.notes)

    def test_mixed_document_above_partial_threshold_is_ok(self):
        # 9 of 10 pages -> ratio 0.9, >= 0.80 -> OK, no note
        pages = [RUSSIAN_PARAGRAPH] * 9 + [""]
        result = classify(pages)
        assert result.status is ExtractionStatus.OK

    def test_garbage_text_treated_as_not_text_bearing(self):
        # Broken CID font mapping: lots of characters, but mostly the
        # Unicode replacement character -- must not count as real text.
        # Two real pages keep total meaningful text above the absolute
        # floor so this exercises the ratio band, not the floor.
        garbage_page = "�" * 500
        pages = [RUSSIAN_PARAGRAPH, RUSSIAN_PARAGRAPH, garbage_page, garbage_page]
        result = classify(pages)
        # ratio 2/4 = 0.5 -> PARTIAL (between the scan and partial thresholds)
        assert result.status is ExtractionStatus.PARTIAL
        assert result.text_bearing_pages == (1, 2)

    def test_non_cyrillic_document_gets_a_note_but_proceeds(self):
        english_paragraph = (
            "This tender documentation governs the procurement process "
            "under the state contract. The bidder must provide a bank "
            "guarantee equal to five percent of the contract value "
            "within ten calendar days of signing, and must fulfil all "
            "obligations described in the attached technical specification."
        )
        pages = [english_paragraph] * 3
        result = classify(pages)
        assert result.status is ExtractionStatus.OK
        assert any("не на русском" in note for note in result.notes)

    def test_two_page_document_below_floor_is_insufficient(self):
        # Regression for the 1-2 page degenerate case: both pages are
        # short, so ratio math never even reaches PARTIAL/OK; the
        # absolute floor must fire first, and this must NOT be
        # reported as a scan (see test below).
        result = classify(["мало текста здесь", "и здесь тоже мало"])
        assert result.status is ExtractionStatus.INSUFFICIENT_TEXT

    def test_short_document_never_reported_as_scanned(self):
        # A 2-page document where neither page individually clears
        # TEXT_PAGE_MIN_CHARS has a ratio of 0.0 -- identical to a
        # true scan's ratio. Below MIN_PAGES_FOR_RATIO_CHECK, ratio is
        # not a meaningful signal, so this must resolve via the
        # absolute floor (INSUFFICIENT_TEXT), never SCANNED_DOCUMENT:
        # a short real document is not a scan.
        result = classify(["мало текста здесь", "и здесь тоже мало"])
        assert result.status is not ExtractionStatus.SCANNED_DOCUMENT


class TestExtractTextFromRealPdf:
    def test_zero_page_pdf_is_empty_document(self, empty_pdf_bytes):
        result = extract_text(empty_pdf_bytes)
        assert result.status is ExtractionStatus.EMPTY_DOCUMENT

    def test_corrupt_bytes_raise(self, not_a_pdf_bytes):
        with pytest.raises(PdfExtractionError):
            extract_text(not_a_pdf_bytes)

    def test_text_bearing_pdf_round_trips_cyrillic(self, make_pdf):
        pdf_bytes = make_pdf([RUSSIAN_PARAGRAPH] * 3)
        result = extract_text(pdf_bytes)
        assert result.status is ExtractionStatus.OK
        assert "тендерная документация" in result.full_text.lower()

    def test_scanned_pdf_is_detected(self, make_pdf):
        pdf_bytes = make_pdf([None, None, None])
        result = extract_text(pdf_bytes)
        assert result.status is ExtractionStatus.SCANNED_DOCUMENT

    def test_mixed_pdf_is_partial(self, make_pdf):
        pdf_bytes = make_pdf([RUSSIAN_PARAGRAPH] * 7 + [None] * 3)
        result = extract_text(pdf_bytes)
        assert result.status is ExtractionStatus.PARTIAL
