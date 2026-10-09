# ScholAI · Controlled Text Rewriting

**An experimental NLP pipeline for rewriting text and inspecting what changes.**

ScholAI separates text generation from evaluation: it extracts propositions, builds semantic anchors, produces candidate rewrites, checks meaning and scores writing quality. Its current implementation combines **remote DeepSeek generation** with **local embedding and language-inference models**.

中文：一个面向文本改写的研究原型，把“生成候选文本”和“检查语义及质量”拆成可单独检查的环节。重点是流水线设计与评估边界，不是准确率承诺。

## Design

```text
Source text
    ↓
Proposition extraction → Semantic anchors
    ↓
Candidate rewriting
    ↓
Embedding similarity + proposition-level NLI + auxiliary BERTScore
    ↓
Quality checks → retry / coherence repair / final candidate and report
```

| Stage | Implementation | Purpose |
|---|---|---|
| Claim extraction | [proposition_extractor.py](src/pipeline/proposition_extractor.py) | Extract propositions and their importance |
| Semantic anchors | [semantic_anchor.py](src/pipeline/semantic_anchor.py) | Represent source text with BGE-M3 embeddings |
| Generation | [rewriter.py](src/pipeline/rewriter.py) · [llm_client.py](src/models/llm_client.py) | Request candidate rewrites through the DeepSeek API |
| Meaning checks | [semantic_validator.py](src/pipeline/semantic_validator.py) | Compare document similarity and proposition entailment |
| Quality and repair | [quality_controller.py](src/pipeline/quality_controller.py) · [coherence_fixer.py](src/pipeline/coherence_fixer.py) | Combine heuristic signals, LLM scoring and retry orchestration |
| Entry points | [pipeline API](src/pipeline/__init__.py) · [CLI](main.py) | Run the stages and expose attempt counts, scores and warnings |

The useful engineering question is whether separate validators can identify information lost or changed during rewriting. Evaluation scores are diagnostic signals; they do not establish that a rewrite is factually correct.

## Run locally

Requires Python 3.10+, a DeepSeek API key, and capacity to download and run the local models. BGE-M3 and NLI model weights are fetched on first use; generation and LLM quality scoring send text to DeepSeek and incur API usage.

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
export DEEPSEEK_API_KEY="YOUR_API_KEY"
python main.py --help
```

Use an environment variable for the CLI: it checks for the key before importing the configuration module that loads `.env`. Keep credentials out of source and version control.

```bash
# Connectivity, embedding load and proposition-extraction checks
python main.py --health-check

# Rewrite a short input and inspect the report
python main.py "这项实验比较了两种导航策略，并记录各自的完成时间。" --verbose

# Or supply your own UTF-8 input file
python main.py --file input.txt --output output.txt --verbose --max-retries 3
```

The health check is a component smoke test, not full-pipeline validation. `--verbose` reports acceptance status, attempt count, similarity, quality score and warnings. A best-effort result may still be written and the CLI currently exits successfully; callers must inspect the report rather than treating exit code zero as quality approval.

## Tests and evidence

[tests/](tests/) contains component and pipeline tests. Some tests use API calls and local models, and test collection requires a configured key; the full suite is not a hermetic offline check.

```bash
python -m pytest tests -q
```

No independently evaluated accuracy, meaning-preservation rate or detector-evasion performance is claimed. Model versions, inputs and failure cases should accompany any future benchmark.

## Known evaluation limits

- **Fail-open behavior:** orchestration can continue after semantic-validation errors. An accepted candidate therefore does not guarantee that every validator ran successfully.
- **Fallback scores:** LLM-scoring errors or malformed responses can produce default scores. Missing evaluation must be distinguished from positive evidence.
- **Source grounding:** the current LLM quality prompt scores the candidate without the original text, so its “faithfulness” score cannot independently check preservation of source claims.
- **Heuristics:** the quantity named `ppl_estimate` is derived from unigram entropy, not language-model perplexity. Vocabulary diversity and sentence-length variation do not prove human authorship.
- **Model coverage:** thresholds and NLI behavior need calibration against labeled English and Chinese examples. Semantic similarity alone can miss changed numbers, entities and negation.

The next research priorities are explicit “evaluation unavailable” states, source-aware scoring, and a labeled regression set containing factual changes and meaning-preserving edits.

## Scope and project notes

This is a command-line research prototype, not a hosted application. Some earlier [product](PM.md), [architecture](技术架构.md) and [research notes](技术调研报告.md) describe local-model or sampling approaches that are not the current runtime. The source files linked above describe what is implemented today.

Use it for editing and text-transformation research with human review. Preserve attribution and follow the disclosure requirements of the context in which a rewrite is used; generated text and heuristic scores are not evidence of original authorship.

## License and acknowledgements

[MIT](LICENSE). Built with the OpenAI-compatible DeepSeek API, [Sentence Transformers](https://github.com/UKPLab/sentence-transformers), BGE-M3, a DeBERTa NLI cross-encoder and [BERTScore](https://github.com/Tiiiger/bert_score). Third-party models and dependencies retain their own licenses.
