"""Local semantic search over the user's own documents.

Ingestion is a straight line, and each step lives in its own module:

    document -> extract -> chunk -> embed -> index (vectors)
                                          -> metadata (text + provenance)

`manager.RagManager` is the only object the rest of the application talks to.
Nothing here reaches the network once the embedding model is on disk.

Note that nothing is imported at package scope. `rag.manager` pulls in numpy,
and document search is optional - the CLI and the API both have to start
without its dependencies installed.
"""
