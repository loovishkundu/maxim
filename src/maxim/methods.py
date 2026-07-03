"""Method-name canonicalization between the research waves.

Wave-1 researchers name methods independently; without canonicalization
"XGBoost" and "gradient boosted trees" fragment the landscape table and the
community wave researches the same method twice. String normalization handles
casing/whitespace; one cheap haiku call merges true synonyms. Degrades to the
normalized identity mapping on any LLM failure — canonicalization is a
quality boost, never a run-breaker.
"""

from __future__ import annotations

import json

from .config import Settings
from .llm import LLM, LLMError
from .prompts import CANONICALIZER_SYSTEM
from .schemas import CanonicalMethods, ResearchDossier


def _dedupe_normalized(methods: list[str]) -> list[str]:
    """Order-preserving dedupe on casefolded names, keeping the first spelling."""
    seen: set[str] = set()
    unique: list[str] = []
    for method in methods:
        name = " ".join(method.split())
        if name and name.casefold() not in seen:
            seen.add(name.casefold())
            unique.append(name)
    return unique


async def canonicalize_methods(
    methods: list[str],
    settings: Settings,
    llm: LLM,
) -> dict[str, str]:
    """Map every method name (casefolded) to its canonical spelling."""
    unique = _dedupe_normalized(methods)
    identity = {name.casefold(): name for name in unique}
    if len(unique) <= 1 or llm.ledger.over_budget:
        return identity

    try:
        result: CanonicalMethods = await llm.parse(
            stage="canonicalizer",
            system=CANONICALIZER_SYSTEM,
            messages=[{"role": "user", "content": json.dumps(unique, ensure_ascii=False)}],
            output_format=CanonicalMethods,
            model=settings.canonicalizer_model,
            effort=settings.canonicalizer_effort,
        )
    except LLMError:
        return identity

    mapping = dict(identity)  # names the model dropped stay mapped to themselves
    for group in result.groups:
        canonical = " ".join(group.canonical.split())
        if not canonical:
            continue
        for variant in group.variants:
            mapping[variant.casefold()] = canonical
    return mapping


def apply_canonical_names(dossier: ResearchDossier, mapping: dict[str, str]) -> ResearchDossier:
    """Rewrite a dossier's method names through the canonical mapping."""
    if not mapping:
        return dossier
    findings = [
        f.model_copy(update={"method_name": mapping.get(f.method_name.casefold(), f.method_name)})
        for f in dossier.findings
    ]
    methods = _dedupe_normalized([mapping.get(m.casefold(), m) for m in dossier.methods_identified])
    return dossier.model_copy(update={"findings": findings, "methods_identified": methods})


def canonical_names(mapping: dict[str, str]) -> list[str]:
    """Unique canonical names, stable order (by first appearance in the map)."""
    return _dedupe_normalized(list(mapping.values()))
