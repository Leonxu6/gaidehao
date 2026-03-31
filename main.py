"""
ScholAI CLI entry point.

Usage examples
--------------
  python main.py "要转换的AI文本"
  python main.py --file input.txt
  python main.py --file input.txt --output output.txt
  python main.py --file input.txt --verbose
  python main.py "文本内容" --api-key sk-xxx --max-retries 2
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
            """
        ),
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

    from src.pipeline import ScholAIPipeline, PipelineConfig

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
