from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple


SCRIPT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class InvarianceVariantSpec:
    study: str
    variant: str
    family: str
    script_name: str
    description: str

    @property
    def script_path(self) -> Path:
        return SCRIPT_ROOT / self.script_name


_REGISTRY: Dict[str, Tuple[InvarianceVariantSpec, ...]] = {
    "study_a": (
        InvarianceVariantSpec("study_a", "lexical", "lexical_paraphrase", "control_paraphrases.py", "Swap low-risk lexical choices while keeping the clinical content fixed."),
        InvarianceVariantSpec("study_a", "surface", "surface_formatting", "control_paraphrases.py", "Alter punctuation and surface formatting without changing the task semantics."),
        InvarianceVariantSpec("study_a", "syntax", "syntax_reordering", "control_paraphrases.py", "Reorder clause structure while preserving the same diagnostic content."),
        InvarianceVariantSpec("study_a", "instruction", "instruction_wrapper", "control_paraphrases.py", "Add a benign instruction wrapper before the unchanged case prompt."),
    ),
    "study_b": (
        InvarianceVariantSpec("study_b", "paraphrase", "incorrect_opinion_rewording", "control_paraphrases.py", "Rephrase the injected incorrect opinion in neutral wording."),
        InvarianceVariantSpec("study_b", "mild", "opinion_intensity", "control_paraphrases.py", "Use a mild injected disagreement cue."),
        InvarianceVariantSpec("study_b", "moderate", "opinion_intensity", "control_paraphrases.py", "Use a moderate injected disagreement cue."),
        InvarianceVariantSpec("study_b", "strong", "opinion_intensity", "control_paraphrases.py", "Use a strong injected disagreement cue."),
        InvarianceVariantSpec("study_b", "question", "assertion_vs_question", "control_paraphrases.py", "Turn the injected belief into a question instead of an assertion."),
        InvarianceVariantSpec("study_b", "cultural", "cultural_framing", "control_paraphrases.py", "Wrap the same incorrect opinion in a cultural framing cue."),
    ),
    "study_b_multi_turn": (
        InvarianceVariantSpec("study_b_multi_turn", "schedule_earlier", "pressure_schedule", "pressure_variants.py", "Move the pressure spike earlier in the same conversation."),
        InvarianceVariantSpec("study_b_multi_turn", "schedule_later", "pressure_schedule", "pressure_variants.py", "Move the pressure spike later in the same conversation."),
        InvarianceVariantSpec("study_b_multi_turn", "tone_gentle", "pressure_tone", "pressure_variants.py", "Retone pressure turns to a gentler register."),
        InvarianceVariantSpec("study_b_multi_turn", "tone_direct", "pressure_tone", "pressure_variants.py", "Retone pressure turns to a direct register."),
        InvarianceVariantSpec("study_b_multi_turn", "tone_confrontational", "pressure_tone", "pressure_variants.py", "Retone pressure turns to a confrontational register."),
        InvarianceVariantSpec("study_b_multi_turn", "pressure_milder", "pressure_intensity", "pressure_variants.py", "Reduce pressure intensity while keeping turn order fixed."),
        InvarianceVariantSpec("study_b_multi_turn", "pressure_stronger", "pressure_intensity", "pressure_variants.py", "Increase pressure intensity while keeping turn order fixed."),
    ),
    "study_c": (
        InvarianceVariantSpec("study_c", "summary_short", "summary_wording", "control_paraphrases.py", "Shorten the patient summary while preserving anchor facts."),
        InvarianceVariantSpec("study_c", "summary_long", "summary_wording", "control_paraphrases.py", "Lengthen the patient summary with a harmless anchor reminder."),
        InvarianceVariantSpec("study_c", "patient_turn_rephrase", "patient_rephrasing", "control_paraphrases.py", "Lightly rephrase a subset of patient turns without changing content."),
        InvarianceVariantSpec("study_c", "noncritical_reorder", "noncritical_turn_reorder", "reorder_turns.py", "Swap only non-critical turns while preserving anchor turns and numbering."),
    ),
}


def study_choices() -> Tuple[str, ...]:
    return tuple(_REGISTRY.keys())


def variant_specs(study: str | None = None) -> Tuple[InvarianceVariantSpec, ...]:
    if study is None:
        all_specs = []
        for value in _REGISTRY.values():
            all_specs.extend(value)
        return tuple(all_specs)
    return _REGISTRY[study]


def variant_choices(study: str | None = None) -> Tuple[str, ...]:
    return tuple(spec.variant for spec in variant_specs(study))


def resolve_specs(
    *,
    studies: Sequence[str] | None = None,
    variants: Sequence[str] | None = None,
) -> Tuple[InvarianceVariantSpec, ...]:
    selected_studies = tuple(studies or study_choices())
    selected_variants = set(variants or ())
    resolved = []
    for study in selected_studies:
        for spec in _REGISTRY[study]:
            if selected_variants and spec.variant not in selected_variants:
                continue
            resolved.append(spec)
    return tuple(resolved)


def matrix_manifest(
    *,
    base_root: Path,
    output_root: Path,
    seed: int,
    specs: Iterable[InvarianceVariantSpec],
) -> dict:
    studies: Dict[str, dict] = {}
    for spec in specs:
        study_payload = studies.setdefault(spec.study, {"variants": []})
        study_payload["variants"].append(
            {
                "variant": spec.variant,
                "family": spec.family,
                "description": spec.description,
                "script": spec.script_name,
                "output_root": str((output_root / spec.study / spec.variant).resolve()),
            }
        )
    return {
        "workflow": "fixed_sample_variant_matrix",
        "sampling_policy": "sample_once_then_fan_out_variants",
        "base_root": str(base_root.resolve()),
        "output_root": str(output_root.resolve()),
        "seed": seed,
        "studies": studies,
    }
