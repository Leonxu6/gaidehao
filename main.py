"""
ScholAI CLI entry point.

Usage examples
--------------
  python main.py "要转换的AI文本"
  python main.py --file input.txt
  python main.py --file input.txt --output output.txt
  python main.py --file input.txt --verbose
  python main.py "文本内容" --api-key sk-xxx --max-retries 2
  python main.py --health-check
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scholai",
        description=(
            "ScholAI — probabilistic-layer AI text humaniser.\n"
            "Rewrites AI-generated text so its statistical features match human writing."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Examples:
              python main.py "AI生成的文本内容"
              python main.py --file input.txt --output output.txt --verbose
              python main.py "文本" --api-key sk-xxx --max-retries 2
              python main.py --health-check
            """
        ),
    )

    # Health check
    parser.add_argument(
        "--health-check",
        action="store_true",
        default=False,
        help="Run a pipeline health check and exit.",
    )

    # Input: positional text or --file
    parser.add_argument(
        "text",
        nargs="?",
        default=None,
        metavar="TEXT",
        help="AI-generated text to humanise (pass directly as a string).",
    )
    parser.add_argument(
        "--file", "-f",
        metavar="PATH",
        default=None,
        help="Read input text from FILE instead of the positional argument.",
    )

    # Output
    parser.add_argument(
        "--output", "-o",
        metavar="PATH",
        default=None,
        help="Write the final text to FILE (default: print to stdout).",
    )

    # Verbosity
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help=(
            "Print a detailed report: propositions count, layer-1 similarity, "
            "quality score, attempt count, warnings."
        ),
    )

    # API / pipeline overrides
    parser.add_argument(
        "--api-key",
        metavar="KEY",
        default=None,
        help=(
            "DeepSeek API key. Overrides the DEEPSEEK_API_KEY environment variable. "
            "If neither is set the program exits with code 1."
        ),
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        metavar="N",
        help="Maximum number of rewrite attempts (default: 3).",
    )

    return parser


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def _print_check(label: str, ok: bool, detail: str = "") -> None:
    icon = "✓" if ok else "✗"
    status = "PASS" if ok else "FAIL"
    detail_str = f"  ({detail})" if detail else ""
    print(f"  [{icon}] {label}: {status}{detail_str}", file=sys.stderr)


def run_health_check(api_key: str) -> int:
    """
    Run a pipeline health check covering:
      1. API key validity (DeepSeek reachability)
      2. Embedding model loading (BGE-M3)
      3. Minimal pipeline smoke test (1 proposition + embed + validate)
    """
    print("ScholAI Health Check", file=sys.stderr)
    print("-" * 40, file=sys.stderr)

    all_ok = True

    # 1. API key
    try:
        from src.config import LLMConfig
        from src.models.llm_client import LLMClient
        cfg = LLMConfig(api_key=api_key)
        client = LLMClient(config=cfg)
        # Quick reachability check
        response = client.generate(
            prompt="Reply exactly: ok",
            system_prompt="You are a test assistant. Reply with the exact word 'ok'.",
            temperature=0.0,
            max_tokens=5,
        )
        api_ok = "ok" in response.lower()
        _print_check("DeepSeek API key", api_ok, "API reachable" if api_ok else f"unexpected reply: {response!r}")
        all_ok = all_ok and api_ok
    except Exception as exc:
        _print_check("DeepSeek API key", False, str(exc))
        all_ok = False

    # 2. Embedding model
    try:
        from src.config import EmbeddingConfig
        from src.models.embedding_client import EmbeddingClient
        emb_cfg = EmbeddingConfig()
        emb_client = EmbeddingClient(config=emb_cfg)
        # Trigger model loading and encode
        vec = emb_client.encode("健康检查测试句子")
        emb_ok = vec.shape[0] == 1024
        _print_check(
            "BGE-M3 embedding model",
            emb_ok,
            f"dim={vec.shape[0]}" if emb_ok else "unexpected dimension",
        )
        all_ok = all_ok and emb_ok
    except Exception as exc:
        _print_check("BGE-M3 embedding model", False, str(exc))
        all_ok = False

    # 3. Proposition extractor (LLM)
    try:
        from src.config import LLMConfig
        from src.models.llm_client import LLMClient
        from src.pipeline.proposition_extractor import PropositionExtractor
        cfg = LLMConfig(api_key=api_key)
        llm = LLMClient(config=cfg)
        extractor = PropositionExtractor(llm_client=llm, config=cfg)
        result = extractor.extract("今天香港天气很好，我们去海边散步。")
        prop_ok = len(result.propositions) > 0
        _print_check(
            "Proposition extractor",
            prop_ok,
            f"{len(result.propositions)} propositions" if prop_ok else "no propositions returned",
        )
        all_ok = all_ok and prop_ok
    except Exception as exc:
        _print_check("Proposition extractor", False, str(exc))
        all_ok = False

    print("-" * 40, file=sys.stderr)
    if all_ok:
        print("All checks passed.", file=sys.stderr)
        return 0
    else:
        print("Some checks failed — see above.", file=sys.stderr)
        return 1


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------

def read_input(args: argparse.Namespace) -> str:
    """Return the input text from --file or the positional argument."""
    if args.file:
        try:
            with open(args.file, encoding="utf-8") as fh:
                return fh.read()
        except OSError as exc:
            print(f"[ERROR] Cannot read input file '{args.file}': {exc}", file=sys.stderr)
            sys.exit(1)

    if args.text:
        return args.text

    # Nothing provided — try stdin if it's a pipe
    if not sys.stdin.isatty():
        return sys.stdin.read()

    print(
        "[ERROR] No input provided. Pass text as a positional argument, "
        "use --file, or pipe text via stdin.",
        file=sys.stderr,
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def write_output(text: str, output_path: str | None) -> None:
    """Write *text* to *output_path* or stdout."""
    if output_path:
        try:
            with open(output_path, "w", encoding="utf-8") as fh:
                fh.write(text)
                if not text.endswith("\n"):
                    fh.write("\n")
            print(f"[OK] Output written to '{output_path}'", file=sys.stderr)
        except OSError as exc:
            print(f"[ERROR] Cannot write to '{output_path}': {exc}", file=sys.stderr)
            sys.exit(1)
    else:
        print(text)


def print_verbose_report(result) -> None:
    """Print a structured verbose report to stderr."""
    sep = "-" * 60
    print(sep, file=sys.stderr)
    print("ScholAI Pipeline Report", file=sys.stderr)
    print(sep, file=sys.stderr)
    print(f"  Status          : {'ACCEPTED' if result.accepted else 'BEST EFFORT'}", file=sys.stderr)
    print(f"  Attempts        : {result.total_attempts}", file=sys.stderr)
    print(f"  Propositions    : {result.propositions_count}", file=sys.stderr)

    if result.layer1_similarity is not None:
        print(f"  Layer-1 sim.    : {result.layer1_similarity:.4f}", file=sys.stderr)
    else:
        print("  Layer-1 sim.    : n/a", file=sys.stderr)

    if result.final_quality_score is not None:
        print(f"  Quality score   : {result.final_quality_score:.2f} / 5.00", file=sys.stderr)
    else:
        print("  Quality score   : n/a", file=sys.stderr)

    if result.warning:
        print(f"  Warning         : {result.warning}", file=sys.stderr)

    print(f"  Input length    : {len(result.original_text)} chars", file=sys.stderr)
    print(f"  Output length   : {len(result.final_text)} chars", file=sys.stderr)
    print(sep, file=sys.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # ------------------------------------------------------------------
    # Health check mode
    # ------------------------------------------------------------------
    if args.health_check:
        api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            print(
                "[ERROR] DeepSeek API key is missing.\n"
                "  Set the DEEPSEEK_API_KEY environment variable or pass --api-key.",
                file=sys.stderr,
            )
            return 1
        return run_health_check(api_key)

    # ------------------------------------------------------------------
    # Resolve API key
    # ------------------------------------------------------------------
    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print(
            "[ERROR] DeepSeek API key is missing.\n"
            "  Set the DEEPSEEK_API_KEY environment variable or pass --api-key.",
            file=sys.stderr,
        )
        return 1

    # ------------------------------------------------------------------
    # Read input text
    # ------------------------------------------------------------------
    text = read_input(args)
    if not text.strip():
        print("[ERROR] Input text is empty.", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------
    # Build pipeline
    # ------------------------------------------------------------------
    if args.verbose:
        print("[INFO] Building pipeline …", file=sys.stderr)

    from src.pipeline import PipelineConfig, ScholAIPipeline

    config = PipelineConfig(
        deepseek_api_key=api_key,
        max_retries=args.max_retries,
    )
    pipeline = ScholAIPipeline.from_config(config)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    if args.verbose:
        print("[INFO] Running pipeline …", file=sys.stderr)

    result = pipeline.run(text)

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    write_output(result.final_text, args.output)

    if args.verbose:
        print_verbose_report(result)

    # Non-zero exit when the pipeline hit its retry limit (but a result was
    # still produced so the caller can still use stdout/file output).
    if result.warning:
        print(f"[WARN] {result.warning}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
