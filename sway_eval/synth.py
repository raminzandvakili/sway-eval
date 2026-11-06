"""Synthetic rubric candidates: templated records with labels fixed by a stated rule.

Each candidate is a resume-like record with one unit per section. A points
rule, stated in full in the instruction, decides the label, so the correct
answer is determinate for every candidate, including those near the
threshold. Text is filled from fixed templates (no language model), so the
data are deterministic given the seed.

Candidates are drawn so that qualification levels (points minus threshold)
are spread evenly over a chosen range, and each candidate records which
section contributes the most points (its decisive section).
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from .units import Item, Unit

DEGREES = ("no degree", "a bachelor's degree", "a master's degree", "a doctorate")
SKILLS = ("Python", "SQL", "statistics", "cloud deployment", "data visualization")
CERTS = ("a cloud architecture certification", "a data engineering certification", "a security certification")
FILLER = {
    "Summary": (
        "Analyst who enjoys working with cross-functional teams.",
        "Interested in roles that combine data work and communication.",
    ),
    "Experience": ("Worked in teams of five to ten people.",),
    "Education": ("Coursework included a capstone project.",),
    "Skills": ("Comfortable with version control.",),
    "Certifications": ("Keeps certifications current.",),
}


@dataclass(frozen=True)
class RubricSpec:
    """The stated rule. Points per section, a threshold, and the section order.

    Points: one per year of relevant experience (up to ``max_years``); 0 to 3
    for degree level; one per required skill listed; one per certification.
    A candidate advances if total points are at least ``threshold``.
    """

    threshold: int = 8
    max_years: int = 8
    convention_order: tuple[str, ...] = ("Summary", "Experience", "Education", "Skills", "Certifications")
    scoring_sections: tuple[str, ...] = ("Experience", "Education", "Skills", "Certifications")
    tie_priority: tuple[str, ...] = ("Experience", "Education", "Skills", "Certifications")
    options: tuple[str, str] = ("ADVANCE", "DO NOT ADVANCE")

    def instruction(self) -> str:
        skills = ", ".join(SKILLS)
        return (
            "Screen this candidate with the following rule. Score points as follows: "
            f"1 point per year of relevant experience (at most {self.max_years}); "
            "degree: bachelor's 1, master's 2, doctorate 3, none 0; "
            f"1 point per required skill listed (required skills: {skills}); "
            "1 point per certification. "
            f"A candidate advances if their total is at least {self.threshold} points. "
            "Nothing else counts. Should this candidate advance?"
        )


@dataclass(frozen=True)
class Profile:
    years: int
    degree: int
    skills: tuple[str, ...]
    certs: tuple[str, ...]
    extra: dict = field(default_factory=dict)

    def points(self) -> dict[str, int]:
        return {
            "Experience": self.years,
            "Education": self.degree,
            "Skills": len(self.skills),
            "Certifications": len(self.certs),
        }


def decisive_section(profile: Profile, spec: RubricSpec) -> tuple[str, bool]:
    """Section with the most points, and whether that maximum is shared."""
    pts = profile.points()
    best = max(pts[s] for s in spec.scoring_sections)
    tops = [s for s in spec.tie_priority if pts[s] == best]
    return tops[0], len(tops) > 1


def rubric_scorer(spec: RubricSpec) -> Callable[[str], list[float]]:
    """Points of each section, in the order the sections appear in a rendered
    record. Finds section headings in any of the usual heading formats.

    For the mock model: it lets a test know exactly what a rule-following
    reader would count, under any layout or format.
    """
    names = "|".join(re.escape(s) for s in spec.convention_order)
    heading = re.compile(rf"^(?:#+\s*)?\[?({names})\]?:?\s*$", re.IGNORECASE | re.MULTILINE)
    canon = {s.lower(): s for s in spec.convention_order}

    def score(context: str) -> list[float]:
        marks = list(heading.finditer(context))
        out = []
        for i, m in enumerate(marks):
            body = context[m.end(): marks[i + 1].start() if i + 1 < len(marks) else len(context)]
            out.append(float(_section_points(canon[m.group(1).lower()], body)))
        return out

    return score


def _section_points(section: str, body: str) -> int:
    if section == "Experience":
        m = re.search(r"(\d+) years? of relevant experience", body)
        return int(m.group(1)) if m else 0
    if section == "Education":
        for k in range(len(DEGREES) - 1, 0, -1):
            if DEGREES[k] in body:
                return k
        return 0
    if section == "Skills":
        return sum(1 for s in SKILLS if re.search(rf"\b{re.escape(s)}\b", body.split("Comfortable")[0]))
    if section == "Certifications":
        return sum(1 for c in CERTS if c in body)
    return 0


def _profiles_with_total(total: int, spec: RubricSpec) -> list[tuple[int, int, int, int]]:
    out = []
    for y in range(spec.max_years + 1):
        for d in range(len(DEGREES)):
            for s in range(len(SKILLS) + 1):
                c = total - y - d - s
                if 0 <= c <= len(CERTS):
                    out.append((y, d, s, c))
    return out


def render_profile(cid: str, profile: Profile, spec: RubricSpec, rng: random.Random) -> tuple[Unit, ...]:
    """One unit per section, filled from templates."""
    lines = {
        "Summary": [rng.choice(FILLER["Summary"])],
        "Experience": [
            f"{profile.years} {'year' if profile.years == 1 else 'years'} of relevant experience as a data analyst."
            if profile.years
            else "No relevant work experience yet.",
            *FILLER["Experience"],
        ],
        "Education": [f"Holds {DEGREES[profile.degree]}.", *FILLER["Education"]],
        "Skills": [
            f"Required skills listed: {', '.join(profile.skills)}." if profile.skills
            else "No required skills listed.",
            *FILLER["Skills"],
        ],
        "Certifications": [
            *(f"Holds {c}." for c in profile.certs),
            *([] if profile.certs else ["No certifications."]),
            *FILLER["Certifications"],
        ],
    }
    return tuple(Unit(id=f"{cid}:{sec}", heading=sec, lines=tuple(lines[sec])) for sec in spec.convention_order)


def generate(spec: RubricSpec, levels: list[int], per_level: int, seed: int) -> list[Item]:
    """Candidates spread evenly over ``levels`` (points minus threshold)."""
    rng = random.Random(seed)
    items = []
    for level in levels:
        combos = _profiles_with_total(spec.threshold + level, spec)
        if not combos:
            raise ValueError(f"no profile reaches level {level}")
        for k in range(per_level):
            y, d, s, c = rng.choice(combos)
            profile = Profile(
                years=y, degree=d,
                skills=tuple(sorted(rng.sample(SKILLS, s), key=SKILLS.index)),
                certs=tuple(sorted(rng.sample(CERTS, c), key=CERTS.index)),
            )
            cid = f"cand{level:+d}_{k:02d}"
            units = render_profile(cid, profile, spec, rng)
            section, shared = decisive_section(profile, spec)
            items.append(Item(
                id=cid,
                units=units,
                convention_order=tuple(u.id for u in units),
                meta={
                    "level": level,
                    "points": profile.points(),
                    "label": level >= 0,
                    "gold": spec.options[0] if level >= 0 else spec.options[1],
                    "decisive_section": section,
                    "decisive_shared": shared,
                },
            ))
    return items
