"""Bounded file-to-report CLI. Only prepare-model explicitly accesses the network."""
from __future__ import annotations

import argparse
import os
import re
import sys

from . import (ApprovedEncoder, Document, Result, Status, WorkBudget, build_index,
               report_json, search)
from ._io_boundary import BoundaryError, DOCUMENT_JSON_BYTES, parse_documents, read_regular
from .types import MODEL_BYTE_LENGTH, TOKENIZER_BYTE_LENGTH, MAX_TEXT_BYTES


class _Usage(Exception):
    pass


class _Help(Exception):
    def __init__(self, frame: bytes):
        self.frame = frame


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        # argparse's default echoes user arguments and paths.
        raise _Usage()

    def print_help(self, file=None):
        raise _Help(self.format_help().encode("utf-8"))


def _integer(value: str) -> int:
    if not re.fullmatch(r"0|[1-9][0-9]{0,5}", value, flags=re.ASCII):
        raise argparse.ArgumentTypeError("canonical nonnegative integer required")
    return int(value)


def _parser() -> _Parser:
    parser = _Parser(prog="local-semantic-atlas", allow_abbrev=False,
                     description="Local multilingual neural retrieval. Search never downloads artifacts.")
    sub = parser.add_subparsers(dest="command", required=True, parser_class=_Parser)
    query = sub.add_parser("search", allow_abbrev=False, help="Search regular-file snapshots; write one complete report to stdout")
    for flag in ("documents", "query-file", "model", "tokenizer"):
        query.add_argument("--" + flag, required=True)
    query.add_argument("--top-k", type=_integer, default=5)
    query.add_argument("--max-documents", type=_integer, default=256)
    query.add_argument("--max-tokens", type=_integer, default=65536,
                       help="Separate per-operation token ceiling for index and query")
    query.add_argument("--format", choices=("json", "html"), default="json")
    query.add_argument("--locale", choices=("en", "pt-BR"), default="pt-BR")
    prepare = sub.add_parser("prepare-model", allow_abbrev=False,
                             help="Explicit network operation: fetch fixed revision into a NEW exclusive directory")
    prepare.add_argument("--destination", required=True)
    return parser


def _exit(result: Result) -> int:
    if result.status in (Status.COMPLETE, Status.COMPLETE_EMPTY):
        return 0
    if result.status in (Status.INVALID_INPUT, Status.INVALID_ARTIFACT):
        return 2
    return 4 if result.status == Status.NUMERICALLY_INDETERMINATE else 3


def _diagnostic() -> None:
    try:
        sys.stderr.write("local-semantic-atlas: OUTPUT_IO\n")
        sys.stderr.flush()
    except (OSError, ValueError, UnicodeError):
        _quiet_failed_stream(sys.stderr)


def _quiet_failed_stream(stream) -> None:
    # Python otherwise retries its buffered flush at shutdown, changing exit 3
    # into exit 120 and emitting an unraisable traceback after /dev/full/pipe I/O.
    descriptor = None
    try:
        descriptor = os.open(os.devnull, os.O_WRONLY)
        os.dup2(descriptor, stream.fileno())
    except (OSError, ValueError, AttributeError):
        pass
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _write(frame: bytes) -> bool:
    """Stage first; no success claim can be emitted before complete serialization."""
    try:
        stream = sys.stdout.buffer
        written = stream.write(frame)
        if written != len(frame):
            raise OSError("short output")
        stream.flush()
        return True
    except (OSError, ValueError, UnicodeError):
        _diagnostic()
        _quiet_failed_stream(sys.stdout)
        return False


def _search(args: argparse.Namespace) -> Result:
    if not (1 <= args.top_k <= 256 and 0 <= args.max_documents <= 256
            and 0 <= args.max_tokens <= 65536):
        return Result(Status.INVALID_INPUT, "INVALID_OPTIONS")
    rows = parse_documents(read_regular(args.documents, DOCUMENT_JSON_BYTES))
    raw_query = read_regular(args.query_file, MAX_TEXT_BYTES)
    try:
        query = raw_query.decode("utf-8", "strict")
    except UnicodeError:
        return Result(Status.INVALID_INPUT, "QUERY_UTF8")
    # Owned bytes go straight to the approved SDK; it verifies both hashes before
    # any tokenizer/session construction. No hub, path reopen, cache or fallback.
    model_bytes = read_regular(args.model, MODEL_BYTE_LENGTH, exact=MODEL_BYTE_LENGTH)
    tokenizer_bytes = read_regular(args.tokenizer, TOKENIZER_BYTE_LENGTH, exact=TOKENIZER_BYTE_LENGTH)
    admitted = ApprovedEncoder.from_snapshots(model_bytes, tokenizer_bytes)
    del model_bytes, tokenizer_bytes
    if admitted.status != Status.COMPLETE:
        return admitted
    budget = WorkBudget(args.max_documents, args.max_tokens)
    indexed = build_index(tuple(Document(id_, text) for id_, text in rows), admitted.value, budget)
    if indexed.status != Status.COMPLETE:
        return indexed
    return search(indexed.value, query, top_k=args.top_k, budget=budget)


def main(argv: list[str] | None = None) -> int:
    args = None
    try:
        args = _parser().parse_args(argv)
        if args.command == "prepare-model":
            from .prepare_model import prepare_model
            result = prepare_model(args.destination)
        else:
            result = _search(args)
    except _Help as help_:
        return 0 if _write(help_.frame) else 3
    except _Usage:
        result = Result(Status.INVALID_INPUT, "USAGE")
    except BoundaryError as error:
        result = Result(Status.INCOMPLETE if error.operational else Status.INVALID_INPUT, error.code)
    except (OSError, UnicodeError):
        result = Result(Status.INCOMPLETE, "IO_FAILURE")
    # Serialization success does not imply operation success. Keep this result
    # for exit status; report_json(INVALID_INPUT) itself returns COMPLETE(bytes).
    original = result
    if args is not None and args.command == "prepare-model" and original.status == Status.COMPLETE:
        framed = original
    elif args is not None and args.command == "search" and args.format == "html":
        from .html_report import report_html
        framed = report_html(original, locale=args.locale)
    else:
        framed = report_json(original)
    if framed.status != Status.COMPLETE:
        original = framed
        framed = report_json(original)
    if framed.status != Status.COMPLETE or not _write(framed.value):
        return 3
    return _exit(original)


if __name__ == "__main__":
    raise SystemExit(main())
