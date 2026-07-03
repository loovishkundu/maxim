from conftest import make_dossier

from maxim.config import Settings
from maxim.llm import LLMError
from maxim.methods import apply_canonical_names, canonical_names, canonicalize_methods
from maxim.schemas import CanonicalMethods, MethodGroup
from maxim.usage import UsageLedger


class FakeCanonLLM:
    def __init__(self, groups=None, raises=False, budget_usd=10.0):
        self.groups = groups or []
        self.raises = raises
        self.ledger = UsageLedger(budget_usd=budget_usd)
        self.calls = 0

    async def parse(self, *, stage, system, messages, output_format, model, effort, **_):
        self.calls += 1
        if self.raises:
            raise LLMError("canonicalizer: boom")
        return CanonicalMethods(groups=self.groups)


async def test_merges_variants_through_llm():
    llm = FakeCanonLLM(
        groups=[
            MethodGroup(canonical="XGBoost", variants=["XGBoost", "gradient boosted trees"]),
            MethodGroup(canonical="Prophet", variants=["Prophet"]),
        ]
    )
    mapping = await canonicalize_methods(
        ["XGBoost", "gradient boosted trees", "Prophet"], Settings(), llm
    )
    assert mapping["gradient boosted trees"] == "XGBoost"
    assert mapping["xgboost"] == "XGBoost"
    assert canonical_names(mapping) == ["XGBoost", "Prophet"]


async def test_single_method_skips_the_llm_call():
    llm = FakeCanonLLM()
    mapping = await canonicalize_methods(["STL", "stl", "  STL  "], Settings(), llm)
    assert llm.calls == 0  # normalization already collapsed everything
    assert mapping == {"stl": "STL"}


async def test_llm_failure_degrades_to_identity():
    llm = FakeCanonLLM(raises=True)
    mapping = await canonicalize_methods(["A", "B"], Settings(), llm)
    assert mapping == {"a": "A", "b": "B"}


async def test_over_budget_skips_the_llm_call():
    llm = FakeCanonLLM(budget_usd=0.0)
    mapping = await canonicalize_methods(["A", "B"], Settings(), llm)
    assert llm.calls == 0
    assert mapping == {"a": "A", "b": "B"}


async def test_names_dropped_by_the_model_stay_mapped():
    # The model only groups one of the two inputs; the other must not vanish.
    llm = FakeCanonLLM(groups=[MethodGroup(canonical="A", variants=["A"])])
    mapping = await canonicalize_methods(["A", "B"], Settings(), llm)
    assert mapping["b"] == "B"


def test_apply_canonical_names_rewrites_findings():
    dossier = make_dossier()  # finding method_name == "STL decomposition"
    mapping = {"stl decomposition": "STL", "prophet": "Prophet"}
    rewritten = apply_canonical_names(dossier, mapping)
    assert rewritten.findings[0].method_name == "STL"
    assert rewritten.methods_identified == ["STL"]
    # Original untouched (model_copy semantics).
    assert dossier.findings[0].method_name == "STL decomposition"


def test_apply_canonical_names_empty_mapping_is_identity():
    dossier = make_dossier()
    assert apply_canonical_names(dossier, {}) is dossier


async def test_lookup_normalizes_whitespace_like_the_mapping_keys():
    # The model returned a whitespace-mangled variant; mapping keys and
    # lookups must normalize identically or the variant silently misses.
    llm = FakeCanonLLM(
        groups=[
            MethodGroup(canonical="XGBoost", variants=["gradient  boosted\ttrees", "XGBoost"]),
        ]
    )
    mapping = await canonicalize_methods(["gradient boosted trees", "XGBoost"], Settings(), llm)
    assert mapping["gradient boosted trees"] == "XGBoost"

    dossier = make_dossier()
    mangled = dossier.model_copy(
        update={
            "findings": [
                dossier.findings[0].model_copy(update={"method_name": "Gradient  Boosted Trees"})
            ]
        }
    )
    rewritten = apply_canonical_names(mangled, mapping)
    assert rewritten.findings[0].method_name == "XGBoost"
