"""Offline tests for keyed sources: disabled-by-default rule + S2/CORE parsing."""

from __future__ import annotations

from poliresearch.core_verifier import CoreVerifier
from poliresearch.models import Reference
from poliresearch.semantic_scholar_verifier import SemanticScholarVerifier
from poliresearch import sources as src


class _Resp:
    def __init__(self, status, payload):
        self.status_code = status
        self._p = payload

    def json(self):
        return self._p


class _Session:
    def __init__(self, routes):
        self.routes = routes

    def get(self, url, headers=None, params=None, timeout=0):
        self.last_headers = headers
        self.last_params = params
        for frag, resp in self.routes.items():
            if frag in url:
                return resp
        return _Resp(404, {})


# --- enable-by-default rule: free+keyless on; key-required off until key set ---
def test_only_free_keyless_sources_on_without_keys(monkeypatch):
    for s in src.REGISTRY:
        monkeypatch.delenv(s.env_var, raising=False)
    names = {s.name for s in src.enabled_sources()}
    assert names == {"Semantic Scholar"}    # free & keyless -> default on; the rest need keys


def test_key_required_source_enables_with_key(monkeypatch):
    for s in src.REGISTRY:
        monkeypatch.delenv(s.env_var, raising=False)
    assert "CORE" not in {s.name for s in src.enabled_sources()}     # off without key
    monkeypatch.setenv("CORE_API_KEY", "k")
    assert "CORE" in {s.name for s in src.enabled_sources()}         # on with key


def test_all_connectors_implemented():
    assert all(s.implemented for s in src.REGISTRY)  # no more 'pending' connectors


# --- Semantic Scholar ---
def test_semantic_scholar_by_doi():
    paper = {"title": "Autonomous chemical research with large language models", "year": 2023,
             "authors": [{"name": "Daniil A. Boiko"}]}
    v = SemanticScholarVerifier(api_key="k", session=_Session({"/paper/DOI:": _Resp(200, paper)}))
    chk = v.verify(Reference(doi="10.1/x", authors=["Boiko, D."], year=2023,
                             title="Autonomous chemical research with large language models"))
    assert chk.exists and chk.source == "semanticscholar"
    assert chk.authors_match and chk.year_match and chk.title_match and chk.ok


def test_semantic_scholar_title_search():
    sess = _Session({"/paper/search": _Resp(200, {"data": [{"title": "Attention Is All You Need",
                                                            "year": 2017, "authors": []}]})})
    v = SemanticScholarVerifier(api_key="k", session=sess)
    chk = v.verify(Reference(title="Attention Is All You Need"))
    assert chk.exists and chk.title_match


def test_semantic_scholar_sends_key_header():
    sess = _Session({"/paper/DOI:": _Resp(200, {"title": "T", "year": 2020, "authors": []})})
    SemanticScholarVerifier(api_key="secret", session=sess).verify(Reference(doi="10.1/x"))
    assert sess.last_headers.get("x-api-key") == "secret"


# --- CORE ---
def test_core_by_doi_and_fulltext():
    work = {"title": "A paper", "yearPublished": 2021, "authors": [{"name": "Smith, John"}],
            "fullText": "Open access body. " * 30}
    v = CoreVerifier(api_key="k", session=_Session({"search/works": _Resp(200, {"results": [work]})}))
    ref = Reference(doi="10.1/y", authors=["Smith, J."], year=2021, title="A paper")
    chk = v.verify(ref)
    assert chk.exists and chk.source == "core" and chk.ok
    assert v.fetch_fulltext(ref).startswith("Open access body")


def test_core_not_found():
    v = CoreVerifier(api_key="k", session=_Session({"search/works": _Resp(200, {"results": []})}))
    assert not v.verify(Reference(doi="10.1/none")).exists


# --- the four keyed REST connectors (offline) ---
from poliresearch.keyed_rest import (ADSVerifier, ElsevierVerifier, IEEEVerifier,
                                     SpringerVerifier)


def test_ads_verifier():
    doc = {"title": ["A Galaxy Paper"], "year": 2019, "author": ["Doe, Jane", "Roe, R."]}
    s = _Session({"adsabs.harvard.edu": _Resp(200, {"response": {"docs": [doc]}})})
    chk = ADSVerifier(api_key="k", session=s).verify(
        Reference(doi="10.1/a", authors=["Doe, J."], year=2019, title="A Galaxy Paper"))
    assert chk.exists and chk.source == "ads" and chk.authors_match and chk.title_match


def test_ieee_verifier():
    art = {"title": "A Circuit Paper", "publication_year": "2020",
           "authors": {"authors": [{"full_name": "John Smith"}]}}
    s = _Session({"ieeexploreapi": _Resp(200, {"articles": [art]})})
    chk = IEEEVerifier(api_key="k", session=s).verify(
        Reference(doi="10.1/b", authors=["Smith, J."], year=2020))
    assert chk.exists and chk.source == "ieee" and chk.authors_match


def test_springer_verifier():
    rec = {"title": "A Springer Paper", "publicationDate": "2018-05-01",
           "creators": [{"creator": "Mueller, Anna"}]}
    s = _Session({"springernature.com": _Resp(200, {"records": [rec]})})
    chk = SpringerVerifier(api_key="k", session=s).verify(
        Reference(doi="10.1/c", authors=["Mueller, A."], year=2018))
    assert chk.exists and chk.source == "springer" and chk.year_match


def test_elsevier_verifier():
    entry = {"dc:title": "A Scopus Paper", "prism:coverDate": "2022-01-01", "dc:creator": "Lee, K."}
    s = _Session({"api.elsevier.com": _Resp(200, {"search-results": {"entry": [entry]}})})
    chk = ElsevierVerifier(api_key="k", session=s).verify(
        Reference(doi="10.1/d", authors=["Lee, K."], year=2022, title="A Scopus Paper"))
    assert chk.exists and chk.source == "elsevier" and chk.title_match


def test_elsevier_handles_error_entry():
    s = _Session({"api.elsevier.com": _Resp(200, {"search-results": {"entry": [{"error": "x"}]}})})
    assert not ElsevierVerifier(api_key="k", session=s).verify(Reference(doi="10.1/none")).exists


def test_fulltext_core_fallback():
    from poliresearch.fulltext import OpenAccessFetcher

    class _Core:
        def fetch_fulltext(self, ref):
            return "CORE open access full text body " * 20

    f = OpenAccessFetcher(session=_Session({}), core=_Core())
    assert "CORE open access" in (f.fetch(Reference(doi="10.1/oa")) or "")
