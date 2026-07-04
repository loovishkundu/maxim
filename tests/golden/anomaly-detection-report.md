# Report

## TL;DR
- STL is solid [F-ai1]

## Method Landscape
| Method | Perspective | Maturity | Community Signal | Best When | Findings |
|---|---|---|---|---|---|
| STL decomposition | Statistics | mature | – | seasonal data | [F-st1] |

## Decision Guide
Start with STL [F-st1].

## Caveats
Coverage gaps remain [F-ai1].


## Sources

### AI / Agentic
- **[F-ai1]** STL catches most injected anomalies on vehicle telemetry.
  - Anomaly detection at Example Corp — <https://eng.example.com/anomaly-detection> (blog, 2026-01) [tier B] ✓
- **[F-ai2]** STL has a low false-positive rate. (not cited in report body)
  - Anomaly detection at Example Corp — <https://eng.example.com/anomaly-detection> (blog, 2026-01) [tier B] ✓
- **[F-ai3]** Prophet needs heavy per-signal tuning. (not cited in report body)
  - Anomaly detection at Example Corp — <https://eng.example.com/anomaly-detection> (blog, 2026-01) [tier B] ✓

### Statistics
- **[F-st1]** STL catches most injected anomalies on vehicle telemetry.
  - Anomaly detection at Example Corp — <https://eng.example.com/anomaly-detection> (blog, 2026-01) [tier B] ✓
- **[F-st2]** STL has a low false-positive rate. (not cited in report body)
  - Anomaly detection at Example Corp — <https://eng.example.com/anomaly-detection> (blog, 2026-01) [tier B] ✓
- **[F-st3]** Prophet needs heavy per-signal tuning. (not cited in report body)
  - Anomaly detection at Example Corp — <https://eng.example.com/anomaly-detection> (blog, 2026-01) [tier B] ✓

### Community
- **[F-cm1]** STL catches most injected anomalies on vehicle telemetry. (not cited in report body)
  - Anomaly detection at Example Corp — <https://eng.example.com/anomaly-detection> (blog, 2026-01) [tier B] ✓
- **[F-cm2]** STL has a low false-positive rate. (not cited in report body)
  - Anomaly detection at Example Corp — <https://eng.example.com/anomaly-detection> (blog, 2026-01) [tier B] ✓
- **[F-cm3]** Prophet needs heavy per-signal tuning. (not cited in report body)
  - Anomaly detection at Example Corp — <https://eng.example.com/anomaly-detection> (blog, 2026-01) [tier B] ✓

## Appendix: Rejected Claims
These claims were drafted during research but did not survive grounding; they are listed for transparency and should not be relied on.

- (AI / Agentic) “STL is obsolete for telemetry.” — mechanical: quote(s) not found in the fetched source text
- (Statistics) “STL is obsolete for telemetry.” — mechanical: quote(s) not found in the fetched source text
- (Community) “STL is obsolete for telemetry.” — mechanical: quote(s) not found in the fetched source text

---
## Run Metadata
- Generated: <normalized> · depth: standard
- Estimated cost: $0.00 · wall time: <normalized>
- Recency horizon: 24 months
- Source tier mix: B:100%
- AI / Agentic: 3 validated, 1 rejected, 0 searches, 0 fetches — ok
- Statistics: 3 validated, 1 rejected, 0 searches, 0 fetches — ok
- Community: 3 validated, 1 rejected, 0 searches, 0 fetches — ok
- Stage usage:
- Costs are estimates from configured pricing, not billing truth.
