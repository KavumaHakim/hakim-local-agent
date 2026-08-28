"""Managing the document index from a terminal, with no model server running.

    python -m rag index <path> [--no-recursive] [--force]
    python -m rag search "<query>" [--top-k N] [--min-score S]
    python -m rag list
    python -m rag remove <id|name|path>
    python -m rag rebuild
    python -m rag compact
    python -m rag stats

This exists because ingesting a folder has nothing to do with chatting. It
needs the embedding model and nothing else - no llama-server, no API, no web
UI - and a folder of PDFs takes long enough on this CPU that it wants to be
started and left, not driven through a chat box.

Everything it prints is meant to be read by a person. The same operations are
available over HTTP for anything else.
"""

from __future__ import annotations

import argparse
import sys

from config import load_config
from rag.embeddings import shared_embedder, unload_shared
from rag.manager import RagError, RagManager


def _build(config) -> RagManager:
    return RagManager(
        config.rag_store,
        embedder=shared_embedder(
            model=config.rag_model,
            model_dir=config.rag_model_dir or None,
            threads=config.rag_threads,
            batch_size=config.rag_batch_size,
            # A command-line run does one thing and exits, so nothing should
            # be swept out from under it mid-ingest.
            idle_seconds=0.0,
        ),
        model=config.rag_model,
        dimension=config.rag_dimension,
        chunk_tokens=config.rag_chunk_tokens,
        overlap_tokens=config.rag_overlap_tokens,
        top_k=config.rag_top_k,
        min_score=config.rag_min_score,
        max_file_bytes=config.rag_max_file_bytes,
    )


def _progress(name: str, done: int, total: int) -> None:
    """Progress on stderr, so stdout stays clean enough to pipe."""
    print(f"  [{done}/{total}] {name}", file=sys.stderr, flush=True)


def _command_index(manager: RagManager, args) -> int:
    print(f"Indexing {args.path} ...", file=sys.stderr)
    result = manager.index_path(
        args.path,
        recursive=not args.no_recursive,
        force=args.force,
        progress=_progress,
    )

    for entry in result["indexed"]:
        print(f"indexed  {entry['document']}  ({entry['chunks']} chunks)")
    for name in result["skipped"]:
        print(f"skipped  {name}  (unchanged)")
    for entry in result["failed"]:
        print(f"FAILED   {entry['document']}: {entry['error']}", file=sys.stderr)

    print(
        f"\n{result['documents_total']} document(s), "
        f"{result['chunks_total']} chunk(s) in the index."
    )
    # A run where every file failed is a failed run, even though the ones that
    # worked were still stored.
    return 1 if result["failed"] and not result["indexed"] else 0


def _command_search(manager: RagManager, args) -> int:
    result = manager.search(args.query, top_k=args.top_k, min_score=args.min_score)

    if not result["results"]:
        print(result.get("note", "No results."))
        return 0

    for position, hit in enumerate(result["results"], start=1):
        where = hit["document"]
        if "page" in hit:
            where += f", page {hit['page']}"
        print(f"\n{position}. [{hit['score']:.3f}] {where}")
        print("   " + "\n   ".join(hit["text"].strip().splitlines()[:12]))
    print()
    return 0


def _command_list(manager: RagManager, args) -> int:
    result = manager.list_documents()
    if not result["documents"]:
        print("Nothing indexed yet.")
        return 0

    print(f"{'id':>4}  {'chunks':>6}  {'pages':>5}  document")
    for entry in result["documents"]:
        pages = entry["pages"] if entry["pages"] is not None else "-"
        print(f"{entry['id']:>4}  {entry['chunks']:>6}  {pages:>5}  {entry['document']}")
        print(f"{'':>19}  {entry['path']}")
    print(f"\n{result['count']} document(s), {result['chunks_total']} chunk(s).")
    return 0


def _command_remove(manager: RagManager, args) -> int:
    result = manager.remove(args.document)
    print(f"Removed {result['document']} ({result['removed_chunks']} chunks).")
    return 0


def _command_rebuild(manager: RagManager, args) -> int:
    print("Re-reading and re-embedding every indexed document ...", file=sys.stderr)
    result = manager.rebuild(progress=_progress)
    for entry in result["rebuilt"]:
        print(f"rebuilt  {entry['document']}  ({entry['chunks']} chunks)")
    for name in result["dropped"]:
        print(f"dropped  {name}  (source file is gone)")
    for entry in result["failed"]:
        print(f"FAILED   {entry['document']}: {entry['error']}", file=sys.stderr)
    print(
        f"\n{result['documents_total']} document(s), "
        f"{result['chunks_total']} chunk(s) in the index."
    )
    return 0


def _command_compact(manager: RagManager, args) -> int:
    result = manager.compact()
    if not result["compacted"]:
        print(result["note"])
    else:
        print(
            f"Reclaimed {result['reclaimed_rows']} row(s); "
            f"{result['chunks']} chunk(s) remain."
        )
    return 0


def _command_stats(manager: RagManager, args) -> int:
    stats = manager.stats()
    width = max(len(key) for key in stats if key != "success")
    for key, value in stats.items():
        if key != "success":
            print(f"{key:<{width}}  {value}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rag",
        description="Index and search local documents for the agent.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    index = subcommands.add_parser("index", help="index a file or folder")
    index.add_argument("path", help="file or directory to index")
    index.add_argument(
        "--no-recursive", action="store_true", help="do not descend into sub-folders"
    )
    index.add_argument(
        "--force", action="store_true", help="re-embed even if nothing changed"
    )
    index.set_defaults(handler=_command_index)

    search = subcommands.add_parser("search", help="semantic search")
    search.add_argument("query", help="what to look for")
    search.add_argument("--top-k", type=int, default=None, help="how many passages")
    search.add_argument(
        "--min-score", type=float, default=None, help="similarity threshold"
    )
    search.set_defaults(handler=_command_search)

    listing = subcommands.add_parser("list", help="list indexed documents")
    listing.set_defaults(handler=_command_list)

    remove = subcommands.add_parser("remove", help="remove one document")
    remove.add_argument("document", help="document id, file name or full path")
    remove.set_defaults(handler=_command_remove)

    rebuild = subcommands.add_parser(
        "rebuild", help="re-read and re-embed everything from source"
    )
    rebuild.set_defaults(handler=_command_rebuild)

    compact = subcommands.add_parser(
        "compact", help="reclaim rows left by deleted documents"
    )
    compact.set_defaults(handler=_command_compact)

    stats = subcommands.add_parser("stats", help="index size and settings")
    stats.set_defaults(handler=_command_stats)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        config = load_config()
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1

    try:
        manager = _build(config)
        return args.handler(manager, args)
    except RagError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n(cancelled)", file=sys.stderr)
        return 130
    finally:
        # The whole point of the design: the model does not outlive the work.
        unload_shared()


if __name__ == "__main__":
    raise SystemExit(main())
