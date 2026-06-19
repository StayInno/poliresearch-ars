"""Offline tests for the citation verifier — a fake HTTP session stands in for Crossref,
so these run with no network."""

from __future__ import annotations

import json

from poliresearch.citation_verifier import CitationVerifier
from poliresearch.models import Reference


class _FakeResp:
    def __init__(self, status_code: int, payload: dict | None = None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload


class _FakeSession:
    """Maps a normalised DOI -> _FakeResp."""

    def __init__(self, routes: dict[str, _FakeResp]):
        self.routes = routes
        self.headers = {}

    def get(self, url, timeout=0):
        doi = url.rsplit("/works/", 1)[-1]
        return self.routes.get(doi, _FakeResp(404))


def _msg(**kw):
    return {"message": kw}


def test_valid_doi_passes_all_gates():
    routes = {
        "10.1038/s41586-023-06792-0": _FakeResp(200, _msg(
            title=["Autonomous chemical research with large language models"],
            type="journal-article",
            author=[{"family": "Boiko"}, {"family": "Gomes"}],
            issued={"date-parts": [[2023, 12, 1]]},
        ))
    }
    v = CitationVerifier(session=_FakeSession(routes))
    chk = v.verify(Reference(doi="10.1038/s41586-023-06792-0",
                             authors=["Gomes, G."], year=2023))
    assert chk.exists and not chk.retracted
    assert chk.authors_match is True
    assert chk.year_match is True
    assert chk.ok


def test_missing_doi_fails_gate1():
    v = CitationVerifier(session=_FakeSession({}))
    chk = v.verify(Reference(doi="10.9999/does-not-exist"))
    assert not chk.exists
    assert not chk.ok


def test_retraction_detected_via_update_to():
    routes = {"10.1234/retracted": _FakeResp(200, _msg(
        title=["A retracted paper"],
        type="journal-article",
        **{"update-to": [{"type": "retraction"}]},
    ))}
    v = CitationVerifier(session=_FakeSession(routes))
    chk = v.verify(Reference(doi="10.1234/retracted"))
    assert chk.exists
    assert chk.retracted
    assert not chk.ok  # gate 2 fails


def test_author_mismatch_flagged():
    routes = {"10.1/x": _FakeResp(200, _msg(
        title=["X"], type="journal-article",
        author=[{"family": "Einstein"}], issued={"date-parts": [[2020]]},
    ))}
    v = CitationVerifier(session=_FakeSession(routes))
    chk = v.verify(Reference(doi="10.1/x", authors=["Newton, I."], year=2020))
    assert chk.authors_match is False
    assert not chk.ok


def test_doi_normalisation():
    routes = {"10.1/y": _FakeResp(200, _msg(title=["Y"], type="journal-article"))}
    v = CitationVerifier(session=_FakeSession(routes))
    for variant in ["https://doi.org/10.1/y", "doi:10.1/Y", "10.1/y"]:
        assert v.verify(Reference(doi=variant)).exists, variant


def test_no_doi_is_rejected():
    v = CitationVerifier(session=_FakeSession({}))
    chk = v.verify(Reference(doi=None, title="some claim"))
    assert not chk.ok
    assert "no DOI" in (chk.error or "")


def test_title_mismatch_flagged_wrong_doi():
    # Real DOI, but the claimed title is about a completely different paper.
    routes = {"10.1/real": _FakeResp(200, _msg(
        title=["Autonomous chemical research with large language models"],
        type="journal-article",
    ))}
    v = CitationVerifier(session=_FakeSession(routes))
    chk = v.verify(Reference(doi="10.1/real",
                             title="A totally unrelated paper about marine biology"))
    assert chk.exists
    assert chk.title_match is False
    assert not chk.ok  # gate 3b catches the wrong-DOI


def test_partial_fabricated_author_rejected():
    # Regression: one real + one fabricated co-author must FAIL (all-match, not any-match).
    routes = {"10.1/x": _FakeResp(200, _msg(
        title=["Autonomous chemical research with large language models"],
        type="journal-article",
        author=[{"family": "Boiko"}, {"family": "Gomes"}],
    ))}
    v = CitationVerifier(session=_FakeSession(routes))
    chk = v.verify(Reference(doi="10.1/x", authors=["Boiko, D.A.", "Fakeson, Q."]))
    assert chk.authors_match is False
    assert not chk.ok


def test_paraphrased_title_accepted():
    # Regression: an order-independent, abbreviated paraphrase of the SAME paper must pass,
    # while a totally unrelated title must still fail.
    routes = {"10.1/x": _FakeResp(200, _msg(
        title=["Autonomous chemical research with large language models"],
        type="journal-article",
    ))}
    v = CitationVerifier(session=_FakeSession(routes))
    para = v.verify(Reference(doi="10.1/x", title="Using LLMs for autonomous chemistry research"))
    assert para.title_match is True
    unrelated = v.verify(Reference(doi="10.1/x", title="A study of coral reef fish populations"))
    assert unrelated.title_match is False


def test_title_match_passes_for_correct_title():
    routes = {"10.1/real": _FakeResp(200, _msg(
        title=["Autonomous chemical research with large language models"],
        type="journal-article",
    ))}
    v = CitationVerifier(session=_FakeSession(routes))
    chk = v.verify(Reference(doi="10.1/real",
                             title="Autonomous chemical research with large language models"))
    assert chk.title_match is True
    assert chk.ok


def test_retraction_via_relation_field():
    routes = {"10.1/rel": _FakeResp(200, _msg(
        title=["X"], type="journal-article",
        relation={"is-retracted-by": [{"id": "10.1/notice"}]},
    ))}
    v = CitationVerifier(session=_FakeSession(routes))
    assert v.verify(Reference(doi="10.1/rel")).retracted


class _FlakySession:
    """Returns retry-status responses N times, then a real 200."""

    def __init__(self, fails: int, payload: dict):
        self.fails = fails
        self.calls = 0
        self.payload = payload
        self.headers = {}

    def get(self, url, timeout=0):
        self.calls += 1
        if self.calls <= self.fails:
            return _FakeResp(503)
        return _FakeResp(200, self.payload)


def test_retry_then_success():
    sess = _FlakySession(fails=2, payload=_msg(title=["Y"], type="journal-article"))
    v = CitationVerifier(session=sess, backoff_base=0)  # no real sleeping
    chk = v.verify(Reference(doi="10.1/y"))
    assert chk.exists
    assert sess.calls == 3  # 2 failures + 1 success


def test_cache_avoids_refetch():
    routes = {"10.1/c": _FakeResp(200, _msg(title=["Z"], type="journal-article"))}
    sess = _FakeSession(routes)
    sess.calls = 0
    orig_get = sess.get

    def counting_get(url, timeout=0):
        sess.calls += 1
        return orig_get(url, timeout)

    sess.get = counting_get
    v = CitationVerifier(session=sess)
    v.verify(Reference(doi="10.1/c"))
    v.verify(Reference(doi="10.1/c", title="Z"))  # second call should hit cache
    assert sess.calls == 1
