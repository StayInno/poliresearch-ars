"""Offline tests for arXiv verification, domain profiles, and unified routing."""

from __future__ import annotations

from poliresearch.arxiv_verifier import ArxivVerifier
from poliresearch.domain import CS_AI, get_domain
from poliresearch.models import Reference
from poliresearch.reference_verification import UnifiedVerifier

_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery</title>
    <published>2024-08-12T00:00:00Z</published>
    <author><name>Chris Lu</name></author>
    <author><name>David Ha</name></author>
  </entry>
</feed>"""

_EMPTY_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"></feed>"""


class _Resp:
    def __init__(self, text, status=200):
        self.text = text
        self.status_code = status


class _ArxivSession:
    def __init__(self, text):
        self.text = text
        self.headers = {}

    def get(self, url, timeout=0):
        return _Resp(self.text)


def test_arxiv_valid_id_resolves_with_metadata():
    v = ArxivVerifier(session=_ArxivSession(_FEED))
    chk = v.verify(Reference(arxiv_id="2408.06292", authors=["Ha, D."], year=2024,
                             title="The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery"))
    assert chk.exists and chk.source == "arxiv"
    assert chk.authors_match is True
    assert chk.year_match is True
    assert chk.title_match is True
    assert chk.ok


def test_arxiv_unknown_id_fails():
    v = ArxivVerifier(session=_ArxivSession(_EMPTY_FEED))
    chk = v.verify(Reference(arxiv_id="9999.99999"))
    assert not chk.exists
    assert not chk.ok


def test_arxiv_id_normalisation():
    v = ArxivVerifier(session=_ArxivSession(_FEED))
    for variant in ["arXiv:2408.06292", "https://arxiv.org/abs/2408.06292", "2408.06292"]:
        assert v.verify(Reference(arxiv_id=variant)).exists, variant


def test_unified_rejects_reference_with_no_identifier():
    # A reference with NO doi/arxiv/pmid AND no title is unverifiable (a title alone now
    # routes to OpenAlex title-search, so that is no longer rejected).
    chk = UnifiedVerifier().verify(Reference())
    assert not chk.exists
    assert "cannot verify" in (chk.error or "")


def test_domain_registry():
    assert get_domain(None) is CS_AI            # ships as CS/AI first
    assert get_domain("cs_ai").enable_experiments is True
    assert get_domain("biomed").enable_experiments is False
    assert get_domain("generic").id_anchors[0] == "doi"


def test_unknown_domain_raises():
    import pytest
    with pytest.raises(ValueError):
        get_domain("astrology")
