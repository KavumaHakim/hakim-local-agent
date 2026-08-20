"""HTTP API for the local agent.

The React front end talks to this; the agent loop, tools and model manager
underneath are the same objects the CLI uses, unchanged.

Nothing here is a second implementation of the agent. Every module in this
package is transport: turning one `Agent.send()` call into a stream of events
a browser can render, and serialising those calls so a two-core machine only
ever runs one.
"""
