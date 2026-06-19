"""Offline tests for BM25 retrieval, Karpathy memory, and the OA full-text fetcher."""

from __future__ import annotations

from poliresearch.corpus import Chunk
from poliresearch.fulltext import OpenAccessFetcher, strip_html
from poliresearch.memory import Memory
from poliresearch.models import Reference
from poliresearch.retrieval import BM25Retriever, make_retriever


# --- BM25 ---
def _chunks():
    return [
        Chunk("a#0", "a", "ROCK inhibitor ripasudil increases RPE phagocytosis via ABCA1."),
        Chunk("b#0", "b", "Large language models generate hypotheses for scientific discovery."),
        Chunk("c#0", "c", "Quantum error correction on superconducting qubits."),
    ]


def test_bm25_ranks_relevant_chunk_first():
    r = BM25Retriever(_chunks())
    hits = r.search("ripasudil ABCA1 phagocytosis", k=1)
    assert hits and "ripasudil" in hits[0].text


def test_bm25_term_weighting_beats_raw_count():
    # a rare, informative term should rank its chunk above a chunk repeating common words.
    r = BM25Retriever(_chunks())
    hits = r.search("qubits", k=1)
    assert hits[0].chunk_id == "c#0"


def test_make_retriever_defaults_bm25():
    assert isinstance(make_retriever(_chunks()), BM25Retriever)


# --- Karpathy memory ---
class _DistillLLM:
    available = True
    model = "fake"

    def complete(self, system, prompt, *, max_tokens=1500):
        return '["Lesson 1: verification independence matters", "Lesson 2: role diversity helps"]'


def test_memory_distill_recompiles_and_clears_working():
    m = Memory(goal="g")
    m.scratch("raw note that should be discarded")
    m.add_finding("novel lead A", "neutral", 0)
    m.add_finding("novel lead B", "neutral", 0)
    m.distill(_DistillLLM())
    assert m.working == []                      # RAM cleared on recompile
    assert len(m.notebook) == 2                 # replaced with distilled lessons
    assert "verification independence" in m.notebook[0]


def test_memory_context_uses_notebook_not_raw_history():
    m = Memory(goal="g")
    m.notebook = ["distilled lesson"]
    m.add_finding("a novel lead", "neutral", 0)
    ctx = m.context()
    assert "distilled lesson" in ctx and "novel lead" in ctx


def test_memory_distill_noop_without_llm_still_clears_ram():
    m = Memory(goal="g")
    m.scratch("ephemeral")
    m.add_finding("x", "neutral", 0)
    m.distill(None)
    assert m.working == []


# --- full text ---
def test_strip_html():
    assert strip_html("<p>Hello <b>world</b></p><script>x()</script>") == "Hello world"


class _Resp:
    def __init__(self, status, text="", payload=None):
        self.status_code = status
        self.text = text
        self._p = payload

    def json(self):
        return self._p


class _Session:
    def __init__(self, routes):
        self.routes = routes

    def get(self, url, timeout=0):
        for frag, resp in self.routes.items():
            if frag in url:
                return resp
        return _Resp(404)


def test_fulltext_arxiv_native_html():
    html = "<html><body>" + "Introduction to the method. " * 40 + "</body></html>"
    f = OpenAccessFetcher(session=_Session({"arxiv.org/html/2408.06292": _Resp(200, html)}))
    text = f.fetch(Reference(arxiv_id="2408.06292"))
    assert text and "Introduction" in text


def test_fulltext_arxiv_rejects_conversion_error_page():
    err = "<html><body>No content available. Conversion to HTML had a Fatal error.</body></html>"
    f = OpenAccessFetcher(session=_Session({"ar5iv.org/abs/9999.9": _Resp(200, err),
                                            "arxiv.org/html/9999.9": _Resp(404)}))
    assert f.fetch(Reference(arxiv_id="9999.9")) is None


def test_fulltext_unpaywall_html():
    routes = {
        "api.unpaywall.org": _Resp(200, payload={"best_oa_location": {"url": "https://oa.example/p"}}),
        "oa.example/p": _Resp(200, "<html>" + "Open access body text. " * 20 + "</html>"),
    }
    f = OpenAccessFetcher(session=_Session(routes))
    text = f.fetch(Reference(doi="10.1/x"))
    assert text and "Open access body" in text


def test_fulltext_returns_none_when_no_oa():
    f = OpenAccessFetcher(session=_Session({"api.unpaywall.org": _Resp(
        200, payload={"best_oa_location": None})}))
    assert f.fetch(Reference(doi="10.1/paywalled")) is None
