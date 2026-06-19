"""Offline tests for OpenAlex + PubMed verifiers and UnifiedVerifier routing."""

from __future__ import annotations

from poliresearch.models import Reference
from poliresearch.openalex_verifier import OpenAlexVerifier
from poliresearch.pubmed_verifier import PubMedVerifier
from poliresearch.reference_verification import UnifiedVerifier


class _Resp:
    def __init__(self, status, payload):
        self.status_code = status
        self._p = payload

    def json(self):
        return self._p


class _Session:
    """Maps a URL-substring -> _Resp."""

    def __init__(self, routes):
        self.routes = routes
        self.headers = {}

    def get(self, url, timeout=0):
        for frag, resp in self.routes.items():
            if frag in url:
                return resp
        return _Resp(404, {})


# --- OpenAlex ---
def test_openalex_by_doi():
    work = {"title": "Autonomous chemical research with large language models",
            "publication_year": 2023, "is_retracted": False,
            "authorships": [{"author": {"display_name": "Daniil A. Boiko"}}]}
    s = _Session({"/works/doi:": _Resp(200, work)})
    chk = OpenAlexVerifier(session=s).verify(
        Reference(doi="10.1038/s41586-023-06792-0", authors=["Boiko, D."], year=2023,
                  title="Autonomous chemical research with large language models"))
    assert chk.exists and chk.source == "openalex"
    assert chk.authors_match and chk.year_match and chk.title_match and chk.ok


def test_openalex_title_fallback_for_reference_with_no_id():
    work = {"title": "Attention Is All You Need", "publication_year": 2017,
            "authorships": [{"author": {"display_name": "Ashish Vaswani"}}]}
    s = _Session({"title.search:": _Resp(200, {"results": [work]})})
    chk = OpenAlexVerifier(session=s).verify(
        Reference(title="Attention Is All You Need", authors=["Vaswani, A."], year=2017))
    assert chk.exists  # verified with NO doi and NO arxiv id
    assert chk.ok


def test_openalex_retracted_flag():
    work = {"title": "X", "publication_year": 2020, "is_retracted": True, "authorships": []}
    s = _Session({"/works/doi:": _Resp(200, work)})
    chk = OpenAlexVerifier(session=s).verify(Reference(doi="10.1/x"))
    assert chk.retracted and not chk.ok


# --- PubMed ---
def test_pubmed_by_pmid():
    rec = {"title": "A biomedical paper", "pubdate": "2021 Mar",
           "pubtype": ["Journal Article"], "authors": [{"name": "Smith J"}]}
    s = _Session({"esummary.fcgi": _Resp(200, {"result": {"12345": rec}})})
    chk = PubMedVerifier(session=s).verify(
        Reference(pmid="12345", authors=["Smith, J."], year=2021, title="A biomedical paper"))
    assert chk.exists and chk.source == "pubmed"
    assert chk.authors_match and chk.year_match and chk.ok


def test_pubmed_retracted_pubtype():
    rec = {"title": "Bad paper", "pubdate": "2019", "pubtype": ["Retracted Publication"],
           "authors": [{"name": "Doe J"}]}
    s = _Session({"esummary.fcgi": _Resp(200, {"result": {"99": rec}})})
    chk = PubMedVerifier(session=s).verify(Reference(pmid="99"))
    assert chk.retracted and not chk.ok


def test_pubmed_missing_pmid():
    s = _Session({"esummary.fcgi": _Resp(200, {"result": {"uids": []}})})
    chk = PubMedVerifier(session=s).verify(Reference(pmid="404404"))
    assert not chk.exists and not chk.ok


# --- UnifiedVerifier routing ---
def test_unified_routes_pmid_to_pubmed():
    v = UnifiedVerifier()
    v.pubmed.session = _Session({"esummary.fcgi": _Resp(
        200, {"result": {"7": {"title": "T", "pubdate": "2020", "pubtype": ["Journal Article"],
                               "authors": []}}})})
    chk = v.verify(Reference(pmid="7"))
    assert chk.source == "pubmed" and chk.exists


def test_unified_routes_titleonly_to_openalex():
    v = UnifiedVerifier()
    v.openalex.session = _Session({"title.search:": _Resp(
        200, {"results": [{"title": "T", "publication_year": 2020, "authorships": []}]})})
    chk = v.verify(Reference(title="T"))
    assert chk.source == "openalex" and chk.exists


def test_openalex_empty_title_results_no_crash():
    # regression: OpenAlex title search returning {"results": []} must not IndexError.
    v = OpenAlexVerifier(session=_Session({"title.search:": _Resp(200, {"results": []})}))
    assert not v.verify(Reference(title="no such paper")).exists
