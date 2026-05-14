# Architecture Diagrams

Mermaid source for SmartRent AI service architecture, written for the
thesis report and advisor review. GitHub renders mermaid natively so
every diagram is viewable straight in the browser; for the thesis PDF
export each block via the mermaid CLI or VSCode "Mermaid Markdown
Syntax" extension.

| # | File | Topic |
|---|---|---|
| 1 | [`01-system-overview.md`](./01-system-overview.md) | End-to-end stack: FE ↔ Backend ↔ AI ↔ external providers |
| 2 | [`02-streaming-sequence.md`](./02-streaming-sequence.md) | SSE chat flow per request (FE → AI → Backend → LLM → events back to FE) |
| 3 | [`03-agent-dispatch.md`](./03-agent-dispatch.md) | Per-round agent loop: Runner.run_streamed → tool calls → ToolContext |
| 4 | [`04-address-bridging.md`](./04-address-bridging.md) | Vietnam 2025-07 admin reform: legacy ↔ new structure bidirectional resolution |
| 5 | [`05-cicd-pipeline.md`](./05-cicd-pipeline.md) | CI/CD: PR merge → image build → Droplet auto-deploy |

## Reading order for the advisor

Start with **01** for the bird's-eye view, then **02** to follow a real
request through the system. **03** drills into the agent loop that is
the core thesis contribution. **04** explains the domain quirk the
contribution had to handle. **05** is operational background, only
relevant if questions go into deployment.

## Update policy

Diagrams are source-of-truth — keep them aligned with code. When you
change orchestrator flow, agent_factory wiring, or the deploy pipeline,
update the matching diagram in the same PR.
