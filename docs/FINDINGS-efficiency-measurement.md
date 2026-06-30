# Findings memo — measuring LLM token efficiency and its correlation with task quality

**Question.** How do you measure LLM token efficiency, and how does token *optimization* correlate with
output/task *quality*? This memo is the **evidence + methodology + tooling** counterpart to the novel-
hypothesis memo (`FINDINGS-token-optimization.md`).

**Method.** Synthesized from a deep-research run (25 sources, 121 claims extracted, **25 put through
3-vote adversarial verification — 23 confirmed, 2 killed**) plus a web-verified OSS-tool enumeration
(45 tools, real repos + licenses checked). Peer-reviewed vs preprint vs blog is marked throughout; the
two refuted claims are recorded in §6.

---

## TL;DR — the correlation, stated plainly

**Token optimization's effect on quality is technique-dependent, not a single curve:**

| Family | Quality effect | Evidence / where it breaks |
|--------|----------------|-----------------------------|
| **Speculative decoding** (EAGLE, Medusa) | **Lossless — provably** | rejection sampling matches the target distribution exactly (formal theorems, Leviathan 2023; ICLR 2025). Only deviation: FP/hardware numerics |
| **Routing / cascades** (FrugalGPT, RouteLLM) | **Lossy but favorable**; *sometimes raises* accuracy (ensembling) | FrugalGPT (TMLR 2024): match GPT-4 at up to 98% cost cut — but best-case-per-dataset, needs labeled training data, fail-then-retry double-billing |
| **Quantization** (FP8/INT4) | **Near-lossless at large scale; degrades small** | ~0.2% avg drop at 405B, but **degradation concentrates in <7B models and multi-step numeric/code** (500K-eval study, arXiv:2409.11055) |
| **Prompt / context compression** (LLMLingua, Headroom, Caveman) | **Lossy; quality risk real; savings often over-claimed** | thin verified evidence; "near-lossless N×" is the field's most over-claimed idea; real-workload test ≈3.7% vs 60–95% claimed (§5) |
| **Naive token-spending** (majority voting) | **~0% economic benefit** | more tokens ≠ proportionate quality (Cost-of-Pass) |

**The one rule:** *the correlation is invisible from accuracy alone.* At matched accuracy, models/configs
differ **3.3×–5.1× in tokens**. You must **plot cost against a held-constant quality measure** to see the
real efficiency-quality frontier — and to catch *silent degradation* (savings that quietly drop quality on
the long tail). That is the entire methodological point (§4).

---

## 1. Measuring token efficiency

- **Cost-of-pass (CPST) — the de-facto standard.** `v(m,p) = C_m(p) / R_m(p)` = expected **$ per correct
  solution**, with input and output tokens **priced separately** (outputs dominate). Fuses cost and
  success into one number. *Erol et al., arXiv:2504.13359 (preprint); re-adopted by Efficient Agents,
  arXiv:2508.02694.* **High confidence.**
- **OckScore** = `Accuracy − λ·log(T/C)` (T=avg output tokens; λ=10, C=10k are author-chosen, not standard).
  *OckBench, arXiv:2511.05722 (preprint).*
- **Energy/token**: `E = E_prefill + E_decode = E_GPU + E_CPU + E_DRAM + …` via NVML/DCGM + Intel RAPL +
  IPMI/PDU. *TokenPowerBench — **peer-reviewed (AAAI)**.*
- **Tokenizer fertility** (tokens/word) **must be normalized** before cross-model comparison — it drives
  speed, context use, compute, and per-token billing. *Frontiers in AI 2025 — **peer-reviewed**.*
- **No model family dominates** the frontier; use **leave-one-family-out counterfactuals** to attribute
  contribution (removing reasoning models worsens the AIME-2024 frontier 81%). *Cost-of-Pass.*

## 2. Measuring quality (and the traps that corrupt the correlation)

- **LLM-as-judge is biased** — position, verbosity, self-enhancement (*Zheng et al., **NeurIPS 2023***);
  12 quantified bias types (*CALM, arXiv:2410.02736*). **Do NOT treat LLM-judge as a validated human-
  equivalent proxy** (see refuted claim §6). No single judge is most reliable across biases.
- **Benchmark contamination** causes **systematic one-way accuracy overestimation** — the annotated-
  benchmark paradigm is "in trouble." *Sainz et al., **EMNLP 2023 Findings**.*
- **Aggregate accuracy masks long-tail failures** (behavioral testing surfaces failures high accuracy
  hides). *Ribeiro et al., **ACL 2020 Best Paper** (CheckList).* — **This is exactly why a compression
  tool can "preserve benchmark accuracy" while silently breaking verbatim/numeric/multi-hop tasks.**

## 3. The token-optimization ↔ quality correlation (your core question)

The verified evidence (table in TL;DR) separates **provably lossless** (speculative decoding) from
**lossy-but-often-net-positive** (routing, quantization) from **lossy-and-over-claimed** (context/output
compression). The recurring failure mode is that **degradation concentrates on the long tail** — small
models, multi-step numeric, code, verbatim-detail, multi-hop — which **aggregate accuracy hides** (§2). So
the correlation is conditional: optimization is near-free on easy/average tasks and increasingly lossy as
task difficulty / precision-sensitivity rises.

**Verified gap (honest):** the deep-research run yielded **no surviving verified claims** on prompt/context
compression or KV-cache compression *quality* tradeoffs — the established-evidence base there is thin
relative to quantization/routing. (This is also where our own novel-hypothesis memo concentrated its
leads, and where the §5 case study lands.)

## 4. How to measure the correlation rigorously

1. **Cost-quality Pareto frontier** + compare at **iso-accuracy / iso-cost** (not raw token counts).
2. **Hold quality constant on the SAME eval suite across every config** — run **lm-evaluation-harness**
   (or **lighteval**) on each optimization setting; quality is the variable you plot cost against, so you
   **catch silent degradation**.
3. **Report token/cost gaps at matched quality** (accuracy alone hides them): identical-size 7B models at
   similar accuracy differ **>3.3× tokens / >5× latency**; DeepSeek-V3.2 matches GPT-5.2 at iso-accuracy
   but uses **5.1× more tokens**. *(OckBench / Efficient Agents; scaffold-dependent — ratios robust.)*
4. **Probe the long tail explicitly** — score numeric / multi-hop / verbatim / small-model slices
   separately, because that is where lossy optimization degrades and where aggregate accuracy lies.
5. **Honest reporting**: n, variance, total $, **and the compressor's/router's OWN cost** (the recurring
   omission — a routing or compression call that is itself an LLM call must be charged against the saving).

## 5. Real-world case study — the claimed-vs-measured gap

Three popular token-saving tools, pitched as a stack:
- **[Headroom](https://github.com/headroomlabs-ai/headroom)** (*Apache-2.0*, ~54k★) — context-compression
  layer (proxy/MCP) compressing tool outputs/logs/RAG before the LLM; reversible. Claims **60–95% fewer tokens**.
- **RTK (Rust Token Killer)** — proxy compressing CLI command output before it reaches the LLM.
- **[Caveman](https://github.com/juliusbrussee/caveman)** (Claude Code skill) — makes the model's *output*
  terse. Claims ~65% output reduction.

**Independent test on a 614M-token corpus of real Claude Code sessions: the three together saved ≈3.7% of
spend** — versus the 60–95% headlines. *(Source: independent blog benchmark — not peer-reviewed; flag as
such, but it is directionally consistent with the verified evidence below.)*

**Why this is the predicted result, not a surprise:** it triangulates three independent lines —
(a) the deep-research finding that "near-lossless" claims break down on real/long-tail workloads and naive
token reduction buys ~nothing; (b) an adversarial completeness critic naming *"near-lossless N×
compression as a free lunch"* the field's most over-claimed idea (benchmark-recoverable answers mask
failures; the compressor's own cost is excluded; gains don't transfer); (c) the TokenGuard program's own
measurements that hook/CLI-layer compression tops out small while routing and overhead-trimming are the
real levers. **Takeaway: treat marketing compression ratios as favorable-measurement upper bounds; measure
cost-of-pass before/after on YOUR workload with a fixed eval suite.**

*(Security note: Headroom/RTK/Caveman all sit in the prompt/output/command path — a compromised release
sees prompts + API key, or runs arbitrary commands. Pin versions and review.)*

## 6. Refuted claims (killed in 3-vote verification — do not cite)

- **"GPT-4 judges reach >80% agreement = human-human level, validating LLM-as-judge as a human proxy."**
  Vote 1-2. *(arXiv:2306.05685.)* → LLM-as-judge is biased and **not** a validated human-equivalent proxy.
- **"Efficient Agents retains 96.7% of OWL's GAIA performance at 28.4% lower cost-of-pass."** Vote 1-2.
  *(arXiv:2508.02694.)*

## 7. Open-source toolchain (verified: real repos + OSS licenses)

### Measurement
**Cost/token & observability:** [LiteLLM](https://github.com/BerriAI/litellm) (MIT) ·
[Langfuse](https://github.com/langfuse/langfuse) (MIT) · [Helicone](https://github.com/Helicone/helicone)
(Apache-2.0) · [OpenLLMetry](https://github.com/traceloop/openllmetry) (Apache-2.0) ·
[tokencost](https://github.com/AgentOps-AI/tokencost) (MIT)
**Quality + efficiency eval:** [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
(MIT) · [HELM](https://github.com/stanford-crfm/helm) (Apache-2.0, scores accuracy **+ efficiency**) ·
[OpenCompass](https://github.com/open-compass/opencompass) (Apache-2.0) ·
[lighteval](https://github.com/huggingface/lighteval) (MIT) ·
[EvalScope](https://github.com/modelscope/evalscope) (Apache-2.0) ·
[LLMPerf](https://github.com/ray-project/llmperf) (Apache-2.0) · vLLM `bench` (Apache-2.0)
**Energy:** [Zeus](https://github.com/ml-energy/zeus) (Apache-2.0) ·
[CodeCarbon](https://github.com/mlco2/codecarbon) (MIT, *local hardware only — not hosted-API calls*) ·
nvidia-ml-py (BSD) · [carbontracker](https://github.com/lfwa/carbontracker) (MIT)

### Optimization
**Prompt/context compression:** [LLMLingua](https://github.com/microsoft/LLMLingua) (MIT) ·
[LMCache](https://github.com/LMCache/LMCache) (Apache-2.0) ·
[Selective Context](https://github.com/liyucheng09/Selective_Context) (MIT) ·
[Headroom](https://github.com/headroomlabs-ai/headroom) (Apache-2.0) · Caveman/RTK (skills/proxies — see §5)
**Quantization:** [llm-compressor](https://github.com/vllm-project/llm-compressor) (Apache-2.0, current
go-to) · [GPTQModel](https://github.com/ModelCloud/GPTQModel) (Apache-2.0) ·
[bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) (MIT) ·
[llama.cpp/GGUF](https://github.com/ggml-org/llama.cpp) (MIT) · *(AutoGPTQ, AutoAWQ — archived 2025)*
**KV-cache + serving:** [vLLM](https://github.com/vllm-project/vllm) (Apache-2.0, PagedAttention+prefix
cache) · [SGLang](https://github.com/sgl-project/sglang) (Apache-2.0, RadixAttention) ·
[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) · [LMDeploy](https://github.com/InternLM/lmdeploy)
(INT4/8 KV) · [KIVI](https://github.com/jy-yuan/KIVI) (MIT, 2-bit KV)
**Speculative decoding & routing:** [EAGLE](https://github.com/SafeAILab/EAGLE) (Apache-2.0, lossless) ·
[Medusa](https://github.com/FasterDecoding/Medusa) · [LookaheadDecoding](https://github.com/hao-ai-lab/LookaheadDecoding) ·
[RouteLLM](https://github.com/lm-sys/RouteLLM) (Apache-2.0, open FrugalGPT-style router) ·
[Semantic Router](https://github.com/aurelio-labs/semantic-router) (MIT)

## 8. Minimal honest stack (measure optimization↔quality on your own workload, OSS only)

1. **Account** tokens + $/call with **LiteLLM** → **Langfuse** (real cost-of-pass per run).
2. **Hold quality** with **lm-evaluation-harness** (or **lighteval**) on the *same* suite across each
   optimization config — **add long-tail slices** (numeric/multi-hop/verbatim) so you catch silent degradation.
3. **Ground energy** (if self-hosting) with **Zeus**/**CodeCarbon**; then **plot cost/energy vs eval score**
   across configs = your actual efficiency-quality frontier.

## Caveats
- Peer-reviewed core: Zheng (NeurIPS'23), Sainz (EMNLP'23), Ribeiro (ACL'20), FrugalGPT (TMLR'24),
  speculative cascades (ICLR'25), TokenPowerBench (AAAI), Frontiers-fertility. Preprints (provisional
  numbers): Cost-of-Pass, OckBench, Efficient Agents, CALM. §5's 3.7% figure is a single blog benchmark.
- **All absolute $/accuracy figures decay fast** (FrugalGPT uses March-2023 prices); ratios and methodology
  transfer, dollar amounts do not.
- Selection/scaffold effects: OckBench's 7B spread comes from a filter engineered to surface gaps; GAIA
  accuracy is harness-dependent.
