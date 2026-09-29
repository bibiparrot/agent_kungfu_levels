---
schema_version: 1
leader:
  name: Dynamic Sakila QA Leader
  instructions: >-
    Dispatch the generated Codex table reader first, then the Codex revenue calculator.
    The host validates every structured action and answers only from Polars evidence.
subagents:
  - id: reader
    name: Dynamic Table Read Agent
    exposed_tool: read_sakila_tables
    description: Load the five configured Sakila tables into shared run state.
    runtime_tool: read_tables
    requires: []
    produces: [tables]
    instructions: >-
      You are a read-only Codex protocol worker. Return only the structured action
      read_tables; never execute a command, inspect files, or modify state yourself.
  - id: calculator
    name: Dynamic Revenue Calculate Agent
    exposed_tool: calculate_category_revenue
    description: Join the staged tables and calculate the requested Top-N category revenue.
    runtime_tool: calculate_revenue
    requires: [tables]
    produces: [revenue]
    instructions: >-
      You are a read-only Codex protocol worker. Return only the structured action
      calculate_revenue with the requested n; never execute a command or modify state yourself.
---

# Level 3 agent contract

This file is data, not executable Python. The factory accepts only runtime tools in a hard-coded
allowlist. Editing the two subagent names, instructions, dependencies, or exposed tool names changes
the two generated Codex Python SDK thread specifications on the next run; arbitrary imports and
`eval` are forbidden. Each named host worker runs one read-only, deny-all, ephemeral Codex thread.
The worker name is retained in the host trace because ephemeral Codex threads reject metadata
updates such as `thread.set_name()`.
Codex proposes a typed action; only the host-side allowlist can invoke the Polars workspace method.

The Python control plane retains responsibility for the final answer. Delegation is recorded in A2A
envelopes, Codex turns in app-server JSON-RPC envelopes, and compact table/calculation results in
MCP-style evidence envelopes.
