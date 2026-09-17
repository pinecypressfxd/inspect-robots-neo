# inspect-robots-vla

VLA policies for [Inspect Robots](https://github.com/robocurve/inspect-robots):
a pure `umi-replay` policy that speaks the `serve_rlt_inference` HTTP wire
(NPZ submit/poll on `:10055`), and a Helix-style `hybrid` policy where a
frontier LLM decomposes the task and delegates skill segments to the VLA.

Status: under active development (plan 0084). The wire client
(`inspect_robots_vla._client.VlaClient`) is protocol-tested against an
in-test fake of the live service; the policy adapters land next.

> [!WARNING]
> VLA-commanded motion goes through the same four clamp layers as any other
> Inspect Robots action (workspace box, clamp + delta limit, per-tick joint
> rate limit, joint envelope). Never run against real arms without those
> guardrails enabled.
