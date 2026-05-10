"""Live ClinicalTrials.gov intelligence lane.

Phase 1 goal:
- Pull compact live trial signals from ClinicalTrials.gov on each analysis run.
- Score and synthesize them into the four executive buckets.
- Preserve backend hooks so the same structured study records can later be persisted
  into a database without changing the executive UI contract.

This module intentionally avoids a Streamlit cache. While the prototype is on
Streamlit Community Cloud, each run fetches fresh data with small page sizes and
short timeouts, then fails gracefully if the upstream service is unavailable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from config.clinical_trials_sources import (
    CLINICAL_TRIALS_PAGE_SIZE,
    CLINICAL_TRIALS_TIMEOUT_SECONDS,
    CLINICAL_TRIAL_SEARCH_SPECS,
    ClinicalTrialSearchSpec,
)

API_BASE = "https://clinicaltrials.gov/api/v2/studies"


DIRECT_LANE_ORDER = ["CDH6 / Ovarian ADC", "B7-H4 ADC", "Ovarian ADC"]
SIDE_LANE_ORDER = ["Alzheimer's Side Channel", "Bone Disease Side Channel"]
LANE_DISPLAY = {
    "CDH6 / Ovarian ADC": "CDH6 / ovarian ADC",
    "B7-H4 ADC": "B7-H4 ADC",
    "Ovarian ADC": "ovarian ADC",
    "ADC Oncology": "broader oncology ADC",
    "Alzheimer's Side Channel": "Alzheimer's exploratory area",
    "Bone Disease Side Channel": "bone-disease exploratory area",
}


def _lane_label(lane: str) -> str:
    return LANE_DISPLAY.get(lane, lane.replace(" Side Channel", " exploratory area"))


def _join_labels(lanes: list[str]) -> str:
    labels = [_lane_label(lane) for lane in lanes]
    if not labels:
        return "the monitored clinical landscape"
    if len(labels) == 1:
        return labels[0]
    return ", ".join(labels[:-1]) + f", and {labels[-1]}"


@dataclass(frozen=True)
class TrialRecord:
    nct_id: str
    title: str
    sponsor: str
    phase: str
    status: str
    conditions: str
    interventions: str
    start_date: str
    last_update: str
    source_query: str
    lane: str
    url: str
    enrollment: str
    primary_outcomes: str
    secondary_outcomes: str
    eligibility_criteria: str
    countries: str
    collaborators: str
    sponsor_type: str


@dataclass(frozen=True)
class ClinicalTrialSignal:
    bucket: str
    title: str
    finding: str
    value: str
    evidence: str
    priority: int


@dataclass(frozen=True)
class ClinicalTrialsSummary:
    source_status: str
    fetched_at_utc: str
    total_trials: int
    active_trials: int
    lanes_covered: list[str]
    signals: list[ClinicalTrialSignal]
    trial_table: pd.DataFrame
    persistence_payload: list[dict[str, Any]]
    source_errors: list[str]

    @property
    def new_information(self) -> list[str]:
        return [s.finding for s in self.signals if s.bucket == "new_information"]

    @property
    def value_interpretation(self) -> list[str]:
        return [s.value for s in self.signals if s.bucket == "value"]

    @property
    def trend_inference(self) -> list[str]:
        return [s.finding for s in self.signals if s.bucket == "trend"]

    @property
    def positioning_implications(self) -> list[str]:
        return [s.finding for s in self.signals if s.bucket == "positioning"]


def _extract_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return ", ".join(_extract_text(v) for v in value if _extract_text(v))
    if isinstance(value, dict):
        return ", ".join(_extract_text(v) for v in value.values() if _extract_text(v))
    return str(value).strip()


def _first_date(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("date") or value.get("startDate") or value.get("completionDate") or "")
    return _extract_text(value)


def _phase(protocol: dict[str, Any]) -> str:
    phases = protocol.get("designModule", {}).get("phases")
    text = _extract_text(phases)
    return text or "Not specified"


def _sponsor(protocol: dict[str, Any]) -> str:
    lead = protocol.get("sponsorCollaboratorsModule", {}).get("leadSponsor", {})
    return _extract_text(lead.get("name")) or "Unknown sponsor"


def _interventions(protocol: dict[str, Any]) -> str:
    arms = protocol.get("armsInterventionsModule", {}).get("interventions", []) or []
    names = []
    for item in arms:
        name = item.get("name") if isinstance(item, dict) else None
        if name:
            names.append(str(name))
    return ", ".join(dict.fromkeys(names)) or "Not specified"


def _conditions(protocol: dict[str, Any]) -> str:
    return _extract_text(protocol.get("conditionsModule", {}).get("conditions")) or "Not specified"


def _status(protocol: dict[str, Any]) -> str:
    return _extract_text(protocol.get("statusModule", {}).get("overallStatus")) or "Unknown"


def _title(protocol: dict[str, Any]) -> str:
    id_module = protocol.get("identificationModule", {})
    return _extract_text(id_module.get("briefTitle") or id_module.get("officialTitle")) or "Untitled trial"


def _nct_id(protocol: dict[str, Any]) -> str:
    return _extract_text(protocol.get("identificationModule", {}).get("nctId"))


def _enrollment(protocol: dict[str, Any]) -> str:
    enrollment = protocol.get("designModule", {}).get("enrollmentInfo", {})
    count = enrollment.get("count") if isinstance(enrollment, dict) else None
    if count in (None, ""):
        return "Not specified"
    return str(count)


def _outcomes(protocol: dict[str, Any], key: str) -> str:
    outcomes = protocol.get("outcomesModule", {}).get(key, []) or []
    parts: list[str] = []
    for item in outcomes:
        if not isinstance(item, dict):
            continue
        measure = _extract_text(item.get("measure"))
        description = _extract_text(item.get("description"))
        if measure and description:
            parts.append(f"{measure}: {description}")
        elif measure:
            parts.append(measure)
        elif description:
            parts.append(description)
    return "; ".join(dict.fromkeys(parts)) or "Not specified"


def _eligibility_criteria(protocol: dict[str, Any]) -> str:
    text = _extract_text(protocol.get("eligibilityModule", {}).get("eligibilityCriteria"))
    return text or "Not specified"


def _countries(protocol: dict[str, Any]) -> str:
    locations = protocol.get("contactsLocationsModule", {}).get("locations", []) or []
    countries: list[str] = []
    for item in locations:
        if isinstance(item, dict):
            country = _extract_text(item.get("country"))
            if country and country not in countries:
                countries.append(country)
    return ", ".join(countries) or "Not specified"


def _collaborators(protocol: dict[str, Any]) -> str:
    module = protocol.get("sponsorCollaboratorsModule", {})
    collaborators = module.get("collaborators", []) or []
    names: list[str] = []
    for item in collaborators:
        if isinstance(item, dict):
            name = _extract_text(item.get("name"))
            if name and name not in names:
                names.append(name)
    return ", ".join(names) or "None listed"


def _sponsor_type_from_name(name: str) -> str:
    text = (name or "").lower()
    if any(token in text for token in ["university", "hospital", "institute", "center", "centre", "m.d. anderson", "massachusetts general", "national cancer institute", "nih"]):
        return "Academic / government"
    if any(token in text for token in ["bristol", "merck", "astrazeneca", "genmab", "gilead", "pfizer", "roche", "novartis", "eli lilly", "abbvie", "bayer", "sanofi", "johnson"]):
        return "Large pharma / established oncology"
    if any(token in text for token in ["biotech", "pharma", "therapeutics", "bioscience", "medicines", "biopharma", "bio", "limited", "ltd", "inc", "llc", "gmbh"]):
        return "Biotech / emerging sponsor"
    return "Other sponsor"


def _record_from_study(study: dict[str, Any], spec: ClinicalTrialSearchSpec) -> TrialRecord | None:
    protocol = study.get("protocolSection", {}) if isinstance(study, dict) else {}
    nct_id = _nct_id(protocol)
    if not nct_id:
        return None
    status_module = protocol.get("statusModule", {})
    return TrialRecord(
        nct_id=nct_id,
        title=_title(protocol),
        sponsor=_sponsor(protocol),
        phase=_phase(protocol),
        status=_status(protocol),
        conditions=_conditions(protocol),
        interventions=_interventions(protocol),
        start_date=_first_date(status_module.get("startDateStruct")),
        last_update=_first_date(status_module.get("lastUpdatePostDateStruct")),
        source_query=spec.query,
        lane=spec.label,
        url=f"https://clinicaltrials.gov/study/{nct_id}",
        enrollment=_enrollment(protocol),
        primary_outcomes=_outcomes(protocol, "primaryOutcomes"),
        secondary_outcomes=_outcomes(protocol, "secondaryOutcomes"),
        eligibility_criteria=_eligibility_criteria(protocol),
        countries=_countries(protocol),
        collaborators=_collaborators(protocol),
        sponsor_type=_sponsor_type_from_name(_sponsor(protocol)),
    )


def _request_payload(params: dict[str, str]) -> dict[str, Any]:
    url = f"{API_BASE}?{urlencode(params)}"
    request = Request(url, headers={"User-Agent": "NextCure-Intelligence-Prototype/0.9.13"})
    with urlopen(request, timeout=CLINICAL_TRIALS_TIMEOUT_SECONDS) as response:  # noqa: S310 - fixed public API endpoint
        return json.loads(response.read().decode("utf-8"))


def _fetch_spec(spec: ClinicalTrialSearchSpec) -> tuple[list[TrialRecord], str | None]:
    base_params = {
        "query.term": spec.query,
        "pageSize": str(CLINICAL_TRIALS_PAGE_SIZE),
        "format": "json",
    }
    attempts = [
        # Preferred if accepted by the upstream API: newest/most recently updated first.
        base_params | {"sort": "LastUpdatePostDate:desc"},
        # Safe fallback if the API rejects or changes sort syntax.
        base_params,
    ]
    last_error: str | None = None
    payload: dict[str, Any] | None = None
    for params in attempts:
        try:
            payload = _request_payload(params)
            break
        except Exception as exc:  # network/API failure should never break the dashboard
            last_error = f"{type(exc).__name__}: {exc}"

    if payload is None:
        return [], f"{spec.label}: {last_error or 'unknown upstream error'}"

    records: list[TrialRecord] = []
    for study in payload.get("studies", []) or []:
        record = _record_from_study(study, spec)
        if record is not None:
            records.append(record)
    return records, None


def _is_active(status: str) -> bool:
    text = status.lower()
    return any(token in text for token in ["recruiting", "active", "enrolling", "not yet recruiting"])


def _trial_table(records: list[TrialRecord]) -> pd.DataFrame:
    columns = [
        "Lane", "NCT ID", "Sponsor", "Sponsor Type", "Phase", "Status", "Title",
        "Conditions", "Interventions", "Primary Outcomes", "Secondary Outcomes",
        "Enrollment", "Countries", "Collaborators", "Start Date", "Last Update", "URL",
    ]
    if not records:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame([
        {
            "Lane": r.lane,
            "NCT ID": r.nct_id,
            "Sponsor": r.sponsor,
            "Sponsor Type": r.sponsor_type,
            "Phase": r.phase,
            "Status": r.status,
            "Title": r.title,
            "Conditions": r.conditions,
            "Interventions": r.interventions,
            "Primary Outcomes": r.primary_outcomes,
            "Secondary Outcomes": r.secondary_outcomes,
            "Enrollment": r.enrollment,
            "Countries": r.countries,
            "Collaborators": r.collaborators,
            "Start Date": r.start_date,
            "Last Update": r.last_update,
            "URL": r.url,
        }
        for r in records
    ])


def _summarize_lanes(records: list[TrialRecord]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for r in records:
        lane = summary.setdefault(r.lane, {"count": 0, "active": 0, "sponsors": set(), "phases": set()})
        lane["count"] += 1
        lane["active"] += 1 if _is_active(r.status) else 0
        lane["sponsors"].add(r.sponsor)
        lane["phases"].add(r.phase)
    for lane in summary.values():
        lane["sponsors"] = sorted(lane["sponsors"])
        lane["phases"] = sorted(lane["phases"])
    return summary



def _lane_records(records: list[TrialRecord], lane_name: str) -> list[TrialRecord]:
    return [r for r in records if r.lane == lane_name]


def _active_records(records: list[TrialRecord]) -> list[TrialRecord]:
    return [r for r in records if _is_active(r.status)]


def _unique_values(records: list[TrialRecord], attr: str, exclude: set[str] | None = None) -> list[str]:
    excluded = exclude or set()
    values: list[str] = []
    for r in records:
        raw = getattr(r, attr, "") or ""
        for part in [x.strip() for x in str(raw).split(",") if x.strip()]:
            if part not in excluded and part not in values:
                values.append(part)
    return values


def _sponsor_phrase(records: list[TrialRecord]) -> str:
    sponsors = _unique_values(records, "sponsor", {"Unknown sponsor"})
    if not sponsors:
        return "sponsor detail not clearly listed"
    return ", ".join(sponsors)


def _sponsor_type_mix(records: list[TrialRecord]) -> str:
    counts: dict[str, int] = {}
    for r in records:
        counts[r.sponsor_type] = counts.get(r.sponsor_type, 0) + 1
    ordered = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    return "; ".join(f"{label}: {count}" for label, count in ordered) or "Sponsor type detail unavailable"


def _country_phrase(records: list[TrialRecord]) -> str:
    countries = _unique_values(records, "countries", {"Not specified"})
    if not countries:
        return "country/site geography not consistently listed"
    return ", ".join(countries)


def _enrollment_read(records: list[TrialRecord]) -> str:
    values: list[int] = []
    for r in records:
        try:
            values.append(int(float(str(r.enrollment).replace(",", ""))))
        except Exception:
            pass
    if not values:
        return "enrollment size was not consistently available across the surfaced records"
    return f"listed enrollment sizes range from {min(values):,} to {max(values):,}, with median-style midpoint around {sorted(values)[len(values)//2]:,}"


def _trial_text(r: TrialRecord) -> str:
    return " ".join([
        r.title, r.conditions, r.interventions, r.primary_outcomes, r.secondary_outcomes,
        r.eligibility_criteria, r.countries, r.collaborators, r.sponsor, r.phase, r.status,
    ]).lower()


def _keyword_presence(records: list[TrialRecord], terms: list[str]) -> list[TrialRecord]:
    return [r for r in records if any(term.lower() in _trial_text(r) for term in terms)]


def _differentiation_reads(records: list[TrialRecord]) -> list[str]:
    if not records:
        return []
    reads: list[str] = []
    biomarker = _keyword_presence(records, ["biomarker", "expression", "positive", "selected", "selection", "stratified", "molecular", "ihc", "overexpress"])
    prior_therapy = _keyword_presence(records, ["platinum", "recurrent", "refractory", "resistant", "prior therapy", "previous therapy", "relapsed"])
    combo = _keyword_presence(records, ["combination", "combined", "pembrolizumab", "nivolumab", "paclitaxel", "carboplatin", "bevacizumab", "chemotherapy", "plus"])
    safety = _keyword_presence(records, ["safety", "tolerability", "dose limiting", "maximum tolerated", "recommended phase 2", "adverse event"])
    endpoints = _keyword_presence(records, ["overall response", "objective response", "progression-free", "duration of response", "dose limiting", "recommended phase 2"])

    if biomarker:
        reads.append(f"Patient-selection signal: {len(biomarker)} surfaced oncology record(s) contain biomarker, expression, positivity, or selection language. This is the part to watch because precision of patient selection is where a CDH6 story can become more than generic ADC exposure.")
    else:
        reads.append("Patient-selection signal: the surfaced oncology records did not consistently expose biomarker-selection language. That makes explicit CDH6 rationale and patient-selection clarity a potential messaging edge if supported by company data.")
    if prior_therapy:
        reads.append(f"Treatment-context signal: {len(prior_therapy)} record(s) reference recurrent, resistant, refractory, platinum, or prior-therapy language. That helps identify whether competitors are fighting in late-line salvage settings versus trying to move into cleaner earlier-line narratives.")
    if combo:
        reads.append(f"Combination signal: {len(combo)} record(s) include combination or partner-therapy language. If peers are leaning on combinations, a cleaner single-agent or better-tolerated positioning can become strategically important if the data support it.")
    if safety or endpoints:
        reads.append(f"Endpoint/safety signal: {len(set([r.nct_id for r in safety + endpoints]))} record(s) expose safety, tolerability, response, PFS, DOR, dose-limiting, or RP2D-style endpoint language. That is where the battlefield shifts from 'who has an ADC' to 'who can prove usable clinical benefit.'")
    return reads


def _phase_phrase(phases: list[str] | set[str]) -> str:
    clean = [p for p in sorted(phases) if p and p != "Not specified"]
    return ", ".join(clean[:4]) if clean else "phase detail not consistently specified"


def _clinical_activity_phrase(data: dict[str, Any], lane_name: str) -> str:
    active = int(data.get("active", 0) or 0)
    total = int(data.get("count", 0) or 0)
    label = _lane_label(lane_name)
    if total <= 0:
        return f"{label} did not contribute enough usable clinical signal to elevate this run"
    ratio = active / total
    if active >= 6 and ratio >= 0.75:
        return f"{label} is showing broad active clinical presence in this run"
    if active >= 3:
        return f"{label} remains meaningfully active in the current clinical sample"
    if active > 0:
        return f"{label} is present, but the signal is narrower than the larger monitored lanes"
    return f"{label} appeared in the clinical landscape, but active development was limited in this run"


def _phase_stage_phrase(phases: list[str] | set[str]) -> str:
    clean = {str(p).upper().replace(" ", "") for p in phases if p and p != "Not specified"}
    if any("PHASE3" in p for p in clean):
        return "the landscape includes late-stage programs, so the field is no longer purely exploratory"
    if any("PHASE2" in p for p in clean):
        return "mid-stage studies are present, which suggests the space is moving beyond first-in-human exploration"
    if any("PHASE1" in p for p in clean):
        return "the activity is still mostly early-stage, leaving room for differentiated clinical positioning"
    return "phase detail is inconsistent, so maturity should be interpreted cautiously"


def _maturity_label(phases: list[str] | set[str]) -> str:
    clean = {str(p).upper().replace(" ", "") for p in phases if p and p != "Not specified"}
    if any("PHASE3" in p for p in clean):
        return "late-stage anchor present"
    if any("PHASE2" in p for p in clean):
        return "mid-stage validation emerging"
    if any("PHASE1" in p for p in clean):
        return "early clinical field"
    return "maturity unclear"


def _theme_phrase(theme: str) -> str:
    mapping = {
        "biomarker / patient-selection language": "patient-selection / biomarker language",
        "combination strategy": "combination strategy",
        "ovarian / gynecologic focus": "ovarian / gynecologic focus",
        "antibody / ADC modality language": "antibody / ADC modality language",
    }
    return mapping.get(theme, theme)


def _theme_hits(records: list[TrialRecord]) -> dict[str, int]:
    theme_terms = {
        "biomarker / patient-selection language": ["biomarker", "expression", "positive", "selected", "selection", "stratified", "molecular"],
        "combination strategy": ["combination", "combined", "plus", "with pembrolizumab", "with chemotherapy", "with paclitaxel"],
        "ovarian / gynecologic focus": ["ovarian", "fallopian", "peritoneal", "gynecologic", "gynaecologic"],
        "antibody / ADC modality language": ["adc", "antibody drug", "antibody-drug", "antibody", "conjugate"],
    }
    counts = {theme: 0 for theme in theme_terms}
    for r in records:
        haystack = " ".join([r.title, r.conditions, r.interventions]).lower()
        for theme, terms in theme_terms.items():
            if any(term in haystack for term in terms):
                counts[theme] += 1
    return {theme: count for theme, count in counts.items() if count > 0}


def _top_theme_sentence(records: list[TrialRecord], scope: str) -> tuple[str, str] | None:
    hits = _theme_hits(records)
    if not hits:
        return None
    ranked = sorted(hits.items(), key=lambda item: item[1], reverse=True)
    top_theme, _ = ranked[0]
    other = [_theme_phrase(name) for name, _count in ranked[1:3]]
    detail = f"; secondary themes include {', '.join(other)}" if other else ""
    return (
        f"Across {scope}, the strongest repeated trial-design language is {_theme_phrase(top_theme)}{detail}.",
        "This matters because repeated protocol language reveals what sponsors are choosing to emphasize clinically, which is more useful than simply knowing that studies exist.",
    )


def _fragmentation_read(records: list[TrialRecord], lane_names: list[str]) -> str:
    lane_count = len(lane_names)
    sponsor_count = len({r.sponsor for r in records if r.sponsor and r.sponsor != "Unknown sponsor"})
    phases = {r.phase for r in records if r.phase and r.phase != "Not specified"}
    maturity = _phase_stage_phrase(phases)
    if sponsor_count >= 5 and lane_count >= 2:
        return (
            f"The direct oncology battlefield is active but fragmented across multiple sponsors; {maturity}. "
            "That is not automatically good or bad. The edge is to make the CDH6 / ovarian ADC story sharper than the category itself: why this target, why this patient population, and why the approach can stand out inside a crowded ADC conversation."
        )
    if sponsor_count >= 2:
        return (
            f"The direct oncology battlefield has multiple active sponsors but is not overwhelmingly broad in this sample; {maturity}. "
            "The edge is focus: use the clinical landscape to show that the category is alive while keeping the differentiation narrative specific to NextCure's own program rather than generic ADC momentum."
        )
    return (
        f"The direct oncology signal is present but narrow in this run; {maturity}. "
        "The edge is selectivity: avoid overstating category heat and instead emphasize the most defensible clinical angle supported by NextCure's own data and upcoming catalysts."
    )


def _latest_update_sentence(records: list[TrialRecord]) -> str | None:
    latest = sorted(_active_records(records), key=lambda r: r.last_update or "", reverse=True)[:4]
    if not latest:
        return None
    pieces = []
    for r in latest:
        phase = f", {r.phase}" if r.phase and r.phase != "Not specified" else ""
        pieces.append(f"{r.sponsor} — {_lane_label(r.lane)}{phase} [{r.nct_id}]")
    return "Recent clinical-record movement worth knowing: " + "; ".join(pieces) + "."


def _phase_mix(records: list[TrialRecord]) -> str:
    order = ["EARLY_PHASE1", "PHASE1", "PHASE2", "PHASE3", "PHASE4"]
    counts: dict[str, int] = {}
    for r in records:
        phase = (r.phase or "Not specified").upper().replace(" ", "")
        counts[phase] = counts.get(phase, 0) + 1
    parts = []
    for key in order:
        if key in counts:
            parts.append(f"{key.replace('_', ' ')}: {counts[key]}")
    for key, val in sorted(counts.items()):
        if key not in order and key != "NOTSPECIFIED":
            parts.append(f"{key}: {val}")
    if counts.get("NOTSPECIFIED"):
        parts.append(f"phase not specified: {counts['NOTSPECIFIED']}")
    return "; ".join(parts) or "phase mix not available"


def _phase_anchor_sponsors(records: list[TrialRecord], phase_token: str = "PHASE3") -> list[str]:
    names: list[str] = []
    for r in records:
        if phase_token in (r.phase or "").upper().replace(" ", "") and r.sponsor not in names:
            names.append(r.sponsor)
    return names


def _lane_profile_sentence(records: list[TrialRecord], lane: str) -> str:
    lane_recs = _lane_records(records, lane)
    if not lane_recs:
        return f"{_lane_label(lane)}: no usable live clinical profile in this run."
    anchors = _phase_anchor_sponsors(lane_recs, "PHASE3")
    anchor_phrase = f" Late-stage anchor sponsor(s): {', '.join(anchors)}." if anchors else " No Phase 3 anchor was surfaced in this lane in this run."
    return (
        f"{_lane_label(lane)} profile — sponsors: {_sponsor_phrase(lane_recs)}. "
        f"Phase mix: {_phase_mix(lane_recs)}. "
        f"Sponsor mix: {_sponsor_type_mix(lane_recs)}. "
        f"Geography: {_country_phrase(lane_recs)}. "
        f"Enrollment signal: {_enrollment_read(lane_recs)}."
        f"{anchor_phrase}"
    )


def _battlefield_edge_sentence(ovarian_records: list[TrialRecord], b7h4_records: list[TrialRecord]) -> str:
    ovarian_anchors = _phase_anchor_sponsors(ovarian_records, "PHASE3")
    sponsor_mix = _sponsor_type_mix(ovarian_records) if ovarian_records else "Sponsor type detail unavailable"
    if ovarian_anchors:
        return (
            f"Ovarian ADC is not an empty or purely early-stage field; Phase 3 anchor sponsor(s) surfaced: {', '.join(ovarian_anchors)}. "
            f"The useful edge is not claiming first-mover category novelty. It is sharper CDH6-specific positioning inside a field that still shows sponsor fragmentation ({sponsor_mix}). "
            "That gives leadership a better board/investor framing: the category is validated enough to matter, but not so consolidated that a clear CDH6 rationale, patient-selection story, and catalyst path cannot stand out."
        )
    return (
        f"Ovarian ADC activity is visible but the current live pull did not surface a Phase 3 anchor inside the ovarian-linked set. Sponsor mix: {sponsor_mix}. "
        "That creates a different edge: the field is active enough to validate attention, while the clinical narrative may still be shaped by whoever can communicate the cleanest target rationale and patient-selection logic."
    )



def _sponsor_segments(records: list[TrialRecord]) -> str:
    buckets: dict[str, list[str]] = {
        "large pharma / established oncology": [],
        "biotech / emerging sponsor": [],
        "academic / government": [],
        "other sponsor": [],
    }
    for r in records:
        name = r.sponsor.strip() or "Unknown sponsor"
        if name == "Unknown sponsor":
            continue
        key = r.sponsor_type.lower()
        if "large pharma" in key:
            bucket = "large pharma / established oncology"
        elif "biotech" in key:
            bucket = "biotech / emerging sponsor"
        elif "academic" in key:
            bucket = "academic / government"
        else:
            bucket = "other sponsor"
        if name not in buckets[bucket]:
            buckets[bucket].append(name)
    parts = []
    for label, names in buckets.items():
        if names:
            parts.append(f"{label}: {', '.join(names)}")
    return "; ".join(parts) or "sponsor segmentation was not available"


def _endpoint_strategy_read(records: list[TrialRecord]) -> str:
    if not records:
        return "Endpoint strategy could not be assessed from the surfaced records."
    categories = {
        "response and tumor-control endpoints": ["objective response", "overall response", "orr", "response rate", "duration of response", "dor", "disease control"],
        "time-to-event endpoints": ["progression-free", "pfs", "overall survival", "os", "time to"],
        "dose/safety endpoints": ["safety", "tolerability", "dose limiting", "maximum tolerated", "recommended phase 2", "rp2d", "adverse event"],
    }
    hits: dict[str, list[str]] = {k: [] for k in categories}
    for r in records:
        haystack = " ".join([r.primary_outcomes, r.secondary_outcomes, r.title]).lower()
        for label, terms in categories.items():
            if any(term in haystack for term in terms) and r.nct_id not in hits[label]:
                hits[label].append(r.nct_id)
    ordered = [(label, ids) for label, ids in hits.items() if ids]
    if not ordered:
        return "Endpoint strategy is not consistently exposed in the surfaced records, so trial maturity should be judged more from phase, sponsor type, and enrollment design."
    phrases = [f"{label} in {len(ids)} study/studies" for label, ids in ordered]
    leader = max(ordered, key=lambda item: len(item[1]))[0]
    return f"Endpoint emphasis: {', '.join(phrases)}. The most visible endpoint posture is {leader}, which helps show whether competitors are optimizing for early activity signals, durability, or dose usability."


def _patient_selection_read(records: list[TrialRecord]) -> str:
    biomarker = _keyword_presence(records, ["biomarker", "expression", "positive", "selected", "selection", "stratified", "molecular", "ihc", "overexpress", "cdh6", "b7-h4", "b7h4"])
    prior = _keyword_presence(records, ["platinum", "recurrent", "refractory", "resistant", "relapsed", "prior therapy", "previous therapy", "after", "progressed"])
    if biomarker and prior:
        return f"Patient-selection read: {len({r.nct_id for r in biomarker})} study/studies expose biomarker, target-expression, or selection language and {len({r.nct_id for r in prior})} study/studies expose recurrent, refractory, resistant, platinum, relapsed, or prior-therapy language. The useful edge is seeing whether competitors are defining who should receive the ADC, not just whether they have an ADC."
    if biomarker:
        return f"Patient-selection read: {len({r.nct_id for r in biomarker})} study/studies expose biomarker, target-expression, or selection language. This is where a CDH6 story can become sharper than generic ovarian ADC exposure if the target rationale is communicated clearly."
    if prior:
        return f"Treatment-context read: {len({r.nct_id for r in prior})} study/studies expose recurrent, refractory, resistant, platinum, relapsed, or prior-therapy language. This helps separate late-line salvage positioning from broader ovarian oncology ambition."
    return "Patient-selection read: biomarker and treatment-line language were not strongly visible in the surfaced records. That absence itself matters because a clearer CDH6 patient-selection rationale can become more distinctive if supported by NextCure's own evidence."


def _combination_read(records: list[TrialRecord]) -> str:
    combo = _keyword_presence(records, ["combination", "combined", "pembrolizumab", "nivolumab", "paclitaxel", "carboplatin", "bevacizumab", "chemotherapy", "plus", "with"])
    if not combo:
        return "Combination read: the surfaced records do not strongly point to combination-heavy positioning. That keeps attention on target rationale, monotherapy activity, tolerability, and patient selection rather than assuming combinations are the main battlefield."
    return f"Combination read: {len({r.nct_id for r in combo})} study/studies contain combination or partner-therapy language. If competitors lean on combinations, the strategic question becomes whether a program can show cleaner single-agent contribution, better tolerability, or a clearer role in the treatment sequence."


def _geography_depth_read(records: list[TrialRecord]) -> str:
    countries = _unique_values(records, "countries", {"Not specified"})
    if not countries:
        return "Geography read: trial-site country detail was not consistently visible."
    regions = []
    lower = {c.lower() for c in countries}
    if "united states" in lower:
        regions.append("U.S.")
    if any(c in lower for c in {"china", "hong kong", "taiwan", "korea, republic of", "japan", "singapore"}):
        regions.append("Asia-Pacific")
    if any(c in lower for c in {"france", "germany", "spain", "italy", "united kingdom", "netherlands", "belgium", "poland"}):
        regions.append("Europe")
    region_phrase = f" Region signal: {', '.join(regions)}." if regions else ""
    return f"Geography read: surfaced countries include {', '.join(countries)}.{region_phrase} Broad geography can indicate operational seriousness; narrow geography can indicate earlier or more localized development."


def _enrollment_depth_read(records: list[TrialRecord]) -> str:
    values: list[tuple[int, TrialRecord]] = []
    for r in records:
        try:
            values.append((int(float(str(r.enrollment).replace(',', ''))), r))
        except Exception:
            pass
    if not values:
        return "Enrollment read: enrollment size was not consistently available, so confidence should lean more on phase, sponsor type, and protocol design."
    values.sort(key=lambda x: x[0], reverse=True)
    top_n = values[:3]
    top_text = "; ".join(f"{r.sponsor} {r.phase} {n:,} planned/actual participants" for n, r in top_n)
    return f"Enrollment read: the largest surfaced enrollment signals are {top_text}. Larger enrollment can indicate seriousness or later-stage breadth; smaller enrollment often points to exploratory signal-finding."


def _board_ammunition_read(records: list[TrialRecord]) -> str:
    if not records:
        return "No board-level clinical ammunition was available from this source run."
    sponsors = _sponsor_segments(records)
    endpoint = _endpoint_strategy_read(records)
    selection = _patient_selection_read(records)
    combo = _combination_read(records)
    return (
        "Board/investor ammunition from ClinicalTrials.gov: "
        f"1) sponsor map — {sponsors}. "
        f"2) {endpoint} "
        f"3) {selection} "
        f"4) {combo}"
    )


def _edge_read(records: list[TrialRecord], lane_name: str) -> str:
    lane_recs = _lane_records(records, lane_name)
    if not lane_recs:
        return f"{_lane_label(lane_name)}: no edge read available from this run."
    phase3 = _phase_anchor_sponsors(lane_recs, "PHASE3")
    phase2 = _phase_anchor_sponsors(lane_recs, "PHASE2")
    sponsors = _unique_values(lane_recs, "sponsor", {"Unknown sponsor"})
    sponsor_count = len(sponsors)
    if phase3 and sponsor_count >= 4:
        setup = f"{_lane_label(lane_name)} has late-stage anchor sponsor(s) ({', '.join(phase3)}) plus a broader sponsor set ({', '.join(sponsors)})."
        edge = "That points to a validated but contested field: the edge is not novelty; the edge is whether NextCure can make CDH6 feel more precise, more biologically justified, and better timed than broad ADC category exposure."
    elif phase2 or phase3:
        anchors = phase3 or phase2
        setup = f"{_lane_label(lane_name)} has visible mid/late clinical anchors ({', '.join(anchors)}) but does not look fully consolidated in this pull."
        edge = "That creates room for a differentiated clinical narrative if NextCure can clearly explain target selection, patient fit, and evidence path."
    else:
        setup = f"{_lane_label(lane_name)} appears active but mainly earlier-stage in this pull, with sponsors including {', '.join(sponsors)}."
        edge = "That is a shapeable battlefield: the edge is establishing clinical credibility and narrative specificity before the space becomes more crowded or later-stage."
    return f"{setup} {edge}"


def _edge_read_for_records(label: str, lane_recs: list[TrialRecord]) -> str:
    if not lane_recs:
        return f"{label}: no edge read available from this run."
    phase3 = _phase_anchor_sponsors(lane_recs, "PHASE3")
    phase2 = _phase_anchor_sponsors(lane_recs, "PHASE2")
    sponsors = _unique_values(lane_recs, "sponsor", {"Unknown sponsor"})
    sponsor_count = len(sponsors)
    if phase3 and sponsor_count >= 4:
        setup = f"{label} has late-stage anchor sponsor(s) ({', '.join(phase3)}) plus a broader sponsor set ({', '.join(sponsors)})."
        edge = "That points to a validated but contested field: the edge is not novelty; the edge is whether NextCure can make CDH6 feel more precise, more biologically justified, and better timed than broad ADC category exposure."
    elif phase2 or phase3:
        anchors = phase3 or phase2
        setup = f"{label} has visible mid/late clinical anchors ({', '.join(anchors)}) but does not look fully consolidated in this pull."
        edge = "That creates room for a differentiated clinical narrative if NextCure can clearly explain target selection, patient fit, and evidence path."
    else:
        setup = f"{label} appears active but mainly earlier-stage in this pull, with sponsors including {', '.join(sponsors)}."
        edge = "That is a shapeable battlefield: the edge is establishing clinical credibility and narrative specificity before the space becomes more crowded or later-stage."
    return f"{setup} {edge}"



# --- v0.9.20: clinical edge synthesis helpers ---------------------------------

def _count_phase(records: list[TrialRecord], token: str) -> int:
    token = token.upper().replace(" ", "")
    return sum(1 for r in records if token in (r.phase or "").upper().replace(" ", ""))


def _sponsors_for_phase(records: list[TrialRecord], token: str) -> list[str]:
    token = token.upper().replace(" ", "")
    names: list[str] = []
    for r in records:
        if token in (r.phase or "").upper().replace(" ", "") and r.sponsor not in names and r.sponsor != "Unknown sponsor":
            names.append(r.sponsor)
    return names


def _phase_architecture(records: list[TrialRecord]) -> str:
    p1 = _count_phase(records, "PHASE1")
    p2 = _count_phase(records, "PHASE2")
    p3 = _count_phase(records, "PHASE3")
    parts = []
    if p3:
        parts.append(f"Phase 3 anchor(s): {', '.join(_sponsors_for_phase(records, 'PHASE3'))}")
    if p2:
        parts.append(f"Phase 2/mid-stage presence: {', '.join(_sponsors_for_phase(records, 'PHASE2'))}")
    if p1:
        parts.append(f"early-stage exploration remains active across {p1} surfaced program(s)")
    return "; ".join(parts) if parts else "phase architecture was not consistently exposed"


def _named_sponsor_segments(records: list[TrialRecord]) -> str:
    # Same idea as _sponsor_segments, but phrased as a strategic map and never hides names.
    buckets: dict[str, list[str]] = {
        "established oncology sponsors": [],
        "emerging / specialist developers": [],
        "academic or government sponsors": [],
        "other named sponsors": [],
    }
    for r in records:
        name = r.sponsor.strip() or "Unknown sponsor"
        if name == "Unknown sponsor":
            continue
        key = r.sponsor_type.lower()
        if "large pharma" in key:
            bucket = "established oncology sponsors"
        elif "biotech" in key:
            bucket = "emerging / specialist developers"
        elif "academic" in key:
            bucket = "academic or government sponsors"
        else:
            bucket = "other named sponsors"
        if name not in buckets[bucket]:
            buckets[bucket].append(name)
    parts = []
    for label, names in buckets.items():
        if names:
            parts.append(f"{label}: {', '.join(names)}")
    return "; ".join(parts) or "named sponsor segmentation was not available"


def _binary_signal(records: list[TrialRecord], terms: list[str]) -> str:
    hits = _keyword_presence(records, terms)
    if not hits:
        return "not strongly visible"
    if len({r.nct_id for r in hits}) >= max(2, len(records) // 3):
        return "clearly visible across multiple surfaced protocols"
    return "visible, but not universal"


def _edge_signal_matrix(records: list[TrialRecord]) -> dict[str, str]:
    return {
        "patient selection / target expression": _binary_signal(records, [
            "biomarker", "expression", "positive", "selected", "selection", "stratified", "molecular", "ihc", "overexpress", "cdh6", "b7-h4", "b7h4"
        ]),
        "late-line / resistant disease context": _binary_signal(records, [
            "platinum", "recurrent", "refractory", "resistant", "relapsed", "prior therapy", "previous therapy", "progressed"
        ]),
        "combination-dependence / partner therapy": _binary_signal(records, [
            "combination", "combined", "pembrolizumab", "nivolumab", "paclitaxel", "carboplatin", "bevacizumab", "chemotherapy", " plus ", " with "
        ]),
        "dose, safety, or tolerability proof burden": _binary_signal(records, [
            "safety", "tolerability", "dose limiting", "maximum tolerated", "recommended phase 2", "rp2d", "adverse event"
        ]),
        "response / durability proof burden": _binary_signal(records, [
            "objective response", "overall response", "orr", "duration of response", "dor", "progression-free", "pfs", "overall survival"
        ]),
    }


def _matrix_sentence(records: list[TrialRecord]) -> str:
    matrix = _edge_signal_matrix(records)
    return "; ".join(f"{k}: {v}" for k, v in matrix.items()) + "."


def _fragmentation_level(records: list[TrialRecord]) -> str:
    sponsors = _unique_values(records, "sponsor", {"Unknown sponsor"})
    p3 = _count_phase(records, "PHASE3")
    if len(sponsors) >= 8 and p3 <= 2:
        return "broad but not narratively consolidated"
    if p3 >= 3:
        return "late-stage and increasingly mature"
    if len(sponsors) >= 4:
        return "competitive but still shapeable"
    return "narrow and still developing"


def _specific_edge_thesis(cdh6_records: list[TrialRecord], ovarian_records: list[TrialRecord], b7h4_records: list[TrialRecord]) -> str:
    cdh6_sponsors = _unique_values(cdh6_records, "sponsor", {"Unknown sponsor"})
    ovarian_sponsors = _unique_values(ovarian_records, "sponsor", {"Unknown sponsor"})
    b7_sponsors = _unique_values(b7h4_records, "sponsor", {"Unknown sponsor"})
    if cdh6_records and ovarian_records:
        return (
            "The most useful clinical-trial edge is the separation between the broad ovarian ADC conversation and the narrower CDH6-specific conversation. "
            f"The broad ovarian ADC map is {_fragmentation_level(ovarian_records)} and includes {', '.join(ovarian_sponsors)}. "
            f"The CDH6-specific map is narrower and includes {', '.join(cdh6_sponsors)}. "
            "That gives leadership a sharper answer than 'ADC is active': the category has enough activity to validate investor attention, but CDH6 can still be framed as a more precise target-specific lane rather than another undifferentiated ovarian ADC claim."
        )
    if ovarian_records:
        return (
            f"The broad ovarian ADC map is {_fragmentation_level(ovarian_records)} and includes {', '.join(ovarian_sponsors)}. "
            "The practical edge is not to argue that the space is uncrowded; it is to identify where the field is still under-defined and make the CDH6 rationale more precise than the category headline."
        )
    if b7h4_records:
        return (
            f"B7-H4 is the clearest adjacent gynecologic-oncology comparator in this run, with named sponsors including {', '.join(b7_sponsors)}. "
            "The edge is to use B7-H4 as an attention read-through while keeping the CDH6 thesis distinct."
        )
    return "The current pull did not surface enough direct ovarian/CDH6 structure to claim a clinical-trial edge from this source alone."


def _investor_artillery(records: list[TrialRecord], cdh6_records: list[TrialRecord], ovarian_records: list[TrialRecord], b7h4_records: list[TrialRecord]) -> str:
    focus = cdh6_records or ovarian_records or records
    answers: list[str] = []
    if ovarian_records:
        answers.append(
            f"If asked whether the ovarian ADC space is crowded: yes, but the current structure looks {_fragmentation_level(ovarian_records)}, not fully owned by one narrative. Named ovarian ADC sponsors: {_sponsor_phrase(ovarian_records)}."
        )
    if cdh6_records:
        answers.append(
            f"If asked what is specific to CDH6: the surfaced CDH6-linked sponsors are {_sponsor_phrase(cdh6_records)}, with phase architecture: {_phase_architecture(cdh6_records)}. The answer should stay target-specific, not generic ADC."
        )
    if b7h4_records:
        answers.append(
            f"If asked about B7-H4: it is a useful gynecologic-oncology attention comparator, with named sponsors {_sponsor_phrase(b7h4_records)}, but it should not be blended into the CDH6 story."
        )
    answers.append("Protocol stress-test: " + _matrix_sentence(focus))
    return " ".join(answers)


def _narrative_opening(records: list[TrialRecord], label: str) -> str:
    if not records:
        return f"{label}: no reliable narrative opening was detected in this run."
    matrix = _edge_signal_matrix(records)
    openings: list[str] = []
    if matrix["patient selection / target expression"] != "clearly visible across multiple surfaced protocols":
        openings.append("patient-selection ownership still looks under-defined")
    if matrix["combination-dependence / partner therapy"] == "clearly visible across multiple surfaced protocols":
        openings.append("single-agent clarity or cleaner sequencing could matter if supported by data")
    if matrix["dose, safety, or tolerability proof burden"] == "clearly visible across multiple surfaced protocols":
        openings.append("tolerability and dose usability are likely part of the real competitive burden")
    if not openings:
        openings.append("the main opening is clearer target-specific narrative ownership rather than generic category participation")
    return f"{label}: " + "; ".join(openings) + "."


def _build_signals(records: list[TrialRecord], errors: list[str]) -> list[ClinicalTrialSignal]:
    signals: list[ClinicalTrialSignal] = []
    if not records:
        detail = "ClinicalTrials.gov did not provide enough usable signal to support a clinical-landscape conclusion in this run."
        if errors:
            detail += " Source diagnostics were captured without interrupting the dashboard."
        return [ClinicalTrialSignal(
            bucket="new_information",
            title="ClinicalTrials.gov source check",
            finding=detail,
            value="This prevents the dashboard from overstating external clinical intelligence when the source pull is degraded or empty.",
            evidence="; ".join(errors[:3]) if errors else "No matching records returned.",
            priority=99,
        )]

    lane_summary = _summarize_lanes(records)
    cdh6_records = _lane_records(records, "CDH6 / Ovarian ADC")
    b7h4_records = _lane_records(records, "B7-H4 ADC")
    ovarian_records = _lane_records(records, "Ovarian ADC")
    adc_records = _lane_records(records, "ADC Oncology")
    side_records = [r for r in records if r.lane in SIDE_LANE_ORDER]
    direct_records = [r for r in records if r.lane in DIRECT_LANE_ORDER]
    focus_records = cdh6_records or ovarian_records or direct_records
    direct_lane_names = [lane for lane in DIRECT_LANE_ORDER if lane in lane_summary]
    side_lane_names = [lane for lane in SIDE_LANE_ORDER if lane in lane_summary]

    # 01 — New information: show what changed as a usable clinical map, not a data dump.
    if focus_records:
        signals.append(ClinicalTrialSignal(
            bucket="new_information",
            title="Clinical battlefield map",
            finding=(
                _specific_edge_thesis(cdh6_records, ovarian_records, b7h4_records) + " "
                f"Named sponsor map: {_named_sponsor_segments(focus_records)}. "
                f"Phase architecture: {_phase_architecture(focus_records)}."
            ),
            value=(
                "This matters because the clinical trial pull is no longer just confirming activity; it is separating broad category noise from the narrower target-specific lane where a board or investor answer can become sharper."
            ),
            evidence="; ".join(f"{r.nct_id}: {r.sponsor} — {r.title}" for r in focus_records[:8]),
            priority=1,
        ))

    recent = _latest_update_sentence(records)
    if recent:
        signals.append(ClinicalTrialSignal(
            bucket="new_information",
            title="Recent named movement",
            finding=(
                recent.replace("Recent clinical-record movement worth knowing: ", "Recent live-trial movement to know: ")
                + " The useful read is not that every update is a threat; it is knowing which names just moved so leadership is not caught flat-footed in an investor or board conversation."
            ),
            value="Recent updates become useful only when attached to names, lanes, and maturity, not when treated as generic news flow.",
            evidence="; ".join(f"{r.nct_id}: {r.title}" for r in sorted(_active_records(records), key=lambda r: r.last_update or "", reverse=True)[:6]),
            priority=2,
        ))

    # 02 — Value: convert trial fields into investor/board artillery.
    if focus_records:
        signals.append(ClinicalTrialSignal(
            bucket="value",
            title="Clinical-trial edge read",
            finding=_investor_artillery(records, cdh6_records, ovarian_records, b7h4_records),
            value=(
                "The edge is not saying the space is congested. The edge is answering the next investor question: whether the battlefield is owned, whether CDH6 is still a shapeable lane, whether competitors are relying on combinations, and whether patient-selection or tolerability can become the differentiator."
            ),
            evidence=(
                f"Endpoint posture: {_endpoint_strategy_read(focus_records)} "
                f"Patient-selection posture: {_patient_selection_read(focus_records)} "
                f"Combination posture: {_combination_read(focus_records)}"
            ),
            priority=3,
        ))

        signals.append(ClinicalTrialSignal(
            bucket="value",
            title="Narrative openings",
            finding=(
                f"{_narrative_opening(cdh6_records or ovarian_records, 'CDH6 / ovarian ADC')} "
                f"{_narrative_opening(b7h4_records, 'B7-H4 comparator')} "
                "These openings are the places where leadership can sharpen the story beyond information that investors may already know."
            ),
            value=(
                "This turns protocol structure into a messaging advantage: the company can prepare answers around target rationale, patient definition, treatment sequence, tolerability burden, and why CDH6 is not interchangeable with adjacent gynecologic-oncology targets."
            ),
            evidence=f"Field scan uses titles, conditions, interventions, endpoints, eligibility criteria, enrollment, sponsor, phase, collaborator, and geography fields from ClinicalTrials.gov.",
            priority=4,
        ))

    # 03 — Trend: avoid repeating raw lane profiles; surface the structure of the battlefield.
    trend_parts: list[str] = []
    if cdh6_records:
        trend_parts.append(
            f"CDH6 / ovarian ADC: {_fragmentation_level(cdh6_records)}. Sponsors: {_sponsor_phrase(cdh6_records)}. {_phase_architecture(cdh6_records)}. {_narrative_opening(cdh6_records, 'Opening')}"
        )
    if ovarian_records:
        trend_parts.append(
            f"Broad ovarian ADC: {_fragmentation_level(ovarian_records)}. Sponsors: {_sponsor_phrase(ovarian_records)}. {_phase_architecture(ovarian_records)}. This is the category context, not the whole NXTC thesis."
        )
    if b7h4_records:
        trend_parts.append(
            f"B7-H4 ADC: {_fragmentation_level(b7h4_records)}. Sponsors: {_sponsor_phrase(b7h4_records)}. {_phase_architecture(b7h4_records)}. Use this as gynecologic-oncology attention read-through, not as a substitute for CDH6."
        )
    if trend_parts:
        signals.append(ClinicalTrialSignal(
            bucket="trend",
            title="Battlefield structure, not category noise",
            finding=" ".join(trend_parts),
            value=(
                "The trend is that ADC activity should not be read as one blended market. CDH6, B7-H4, and broad ovarian ADC each create different investor questions, and the dashboard now separates them so the answer can be specific."
            ),
            evidence="; ".join(_lane_profile_sentence(records, lane) for lane in direct_lane_names[:3]),
            priority=5,
        ))

    if adc_records:
        signals.append(ClinicalTrialSignal(
            bucket="trend",
            title="Category weather check",
            finding=(
                f"Broad ADC oncology remains category weather rather than the core positioning thesis. Named broader ADC sponsors surfaced here: {_sponsor_phrase(adc_records)}. "
                f"{_phase_architecture(adc_records)}. {_geography_depth_read(adc_records)}"
            ),
            value=(
                "Broad ADC activity can help investors pay attention to the category, but it does not create the company-specific edge. The edge has to be carried by target specificity, patient-selection logic, and clinical evidence quality."
            ),
            evidence=f"Sponsor segmentation: {_named_sponsor_segments(adc_records)}. Enrollment: {_enrollment_read(adc_records)}.",
            priority=6,
        ))

    if side_records:
        side_parts = []
        for lane in side_lane_names:
            lane_recs = _lane_records(records, lane)
            side_parts.append(f"{_lane_label(lane)}: sponsors include {_sponsor_phrase(lane_recs)}; {_phase_architecture(lane_recs)}")
        signals.append(ClinicalTrialSignal(
            bucket="trend",
            title="Exploratory side-channel discipline",
            finding=(
                "Exploratory areas are tracked but kept separate from the core oncology thesis. " + " ".join(side_parts)
            ),
            value=(
                "This keeps optionality visible without letting Alzheimer's or bone-disease activity dilute the main CDH6 / ovarian ADC investor conversation."
            ),
            evidence="Side-channel records are preserved as separate source records and do not contaminate the direct oncology read.",
            priority=7,
        ))

    # 04 — Positioning: a direct talk track, not another trial list.
    if focus_records:
        signals.append(ClinicalTrialSignal(
            bucket="positioning",
            title="Board-ready clinical positioning",
            finding=(
                "The most useful positioning line is: NXTC should not be framed as generic ADC exposure. "
                "It should be framed against a specific clinical battlefield where broad ovarian ADC is active, B7-H4 is an adjacent attention comparator, and CDH6 remains the sharper target-specific lane to defend. "
                f"The current clinical-trial artillery is: {_investor_artillery(records, cdh6_records, ovarian_records, b7h4_records)}"
            ),
            value=(
                "This gives leadership a concrete answer when challenged on crowding: the category is real, competition is real, but the strategic opening is owning a more precise CDH6 / ovarian narrative with evidence quality, patient selection, and catalyst timing."
            ),
            evidence=(
                f"{_named_sponsor_segments(focus_records)}. {_matrix_sentence(focus_records)} {_geography_depth_read(focus_records)} {_enrollment_depth_read(focus_records)}"
            ),
            priority=8,
        ))
    else:
        signals.append(ClinicalTrialSignal(
            bucket="positioning",
            title="NXTC positioning implication",
            finding="The live clinical pull did not support a strong direct clinical read-through, so NXTC positioning should lean more heavily on market behavior, peer performance, and company-specific catalysts in this run.",
            value="This keeps the system from overstating clinical-trial relevance when the live records do not support it.",
            evidence="ClinicalTrials.gov records are kept below the Executive Summary as supporting evidence.",
            priority=8,
        ))

    return sorted(signals, key=lambda s: s.priority)

def build_clinical_trials_intelligence() -> ClinicalTrialsSummary:
    fetched_at = datetime.now(UTC).isoformat(timespec="seconds")
    by_nct: dict[str, TrialRecord] = {}
    errors: list[str] = []

    for spec in CLINICAL_TRIAL_SEARCH_SPECS:
        records, error = _fetch_spec(spec)
        if error:
            errors.append(error)
        for record in records:
            existing = by_nct.get(record.nct_id)
            # Keep the highest-priority/source-specific lane for duplicates.
            if existing is None:
                by_nct[record.nct_id] = record
            else:
                existing_priority = next((s.priority for s in CLINICAL_TRIAL_SEARCH_SPECS if s.label == existing.lane), 99)
                if spec.priority < existing_priority:
                    by_nct[record.nct_id] = record

    records = list(by_nct.values())
    records.sort(key=lambda r: (r.last_update or "", r.nct_id), reverse=True)
    signals = _build_signals(records, errors)
    table = _trial_table(records)
    payload = [asdict(record) | {"fetched_at_utc": fetched_at, "source": "clinicaltrials.gov"} for record in records]
    active_count = sum(1 for r in records if _is_active(r.status))
    source_status = "live" if records else ("degraded" if errors else "empty")

    return ClinicalTrialsSummary(
        source_status=source_status,
        fetched_at_utc=fetched_at,
        total_trials=len(records),
        active_trials=active_count,
        lanes_covered=sorted({r.lane for r in records}),
        signals=signals,
        trial_table=table,
        persistence_payload=payload,
        source_errors=errors,
    )
