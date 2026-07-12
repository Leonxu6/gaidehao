# ScholAI

> A research prototype for meaning-preserving text transformation with semantic and quality evaluation.

[中文说明](#中文说明) · [Product brief](PM.md) · [Architecture](技术架构.md) · [Research notes](技术调研报告.md)

## Scope

ScholAI explores how a text-transformation pipeline can preserve claims, named entities, and numerical facts while evaluating semantic consistency and language quality. It is an experimental project, not a production service.

The repository combines controlled local-model generation with layered evaluation:

1. Extract claims and constraints from the source text.
2. Generate candidate rewrites with controlled sampling.
3. Check semantic consistency with embeddings and NLI.
4. Evaluate quality using statistical and model-based signals.
5. Retry or reject candidates that miss the configured quality bar.

## Safety and responsible use

This project is intended for legitimate editing, accessibility, localization, and writing-assistance research. It must not be used to misrepresent authorship, evade academic-integrity requirements, or bypass platform safeguards. Any deployment needs its own user policy, human review, and compliance assessment.

## Project status

Research prototype. The repository contains product documentation, architecture notes, experiments, and tests; interfaces and evaluation thresholds may change as the research evolves.

## Getting started

```bash
# Create the environment and install dependencies.
uv sync

# Inspect available development tasks.
make help
```

See the project documents above for model requirements, evaluation design, and hardware guidance. Keep credentials in local environment files; never commit API keys.

## 中文说明

ScholAI 是一个“语义不走样、质量可检查”的文本转换研究原型。它通过受控生成、语义一致性验证和质量评测来探索文本编辑工作流；不是生产服务，也不应用于代写、学术不端或规避平台规则。

- [PM.md](PM.md)：产品边界与问题定义
- [技术架构.md](技术架构.md)：五步流水线与验证设计
- [技术调研报告.md](技术调研报告.md)：模型、硬件和评测调研

## Development workflow

The repository includes focused agent roles under `.claude/agents/` for engineering, product review, and QA critique. Treat generated outputs as drafts: validate meaning, facts, and policy compliance before use.
