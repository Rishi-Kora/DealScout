"""Output schemas for DealScout agents.

Every agent produces structured output (no free-form prose between agents).
The FactSheet below is the M1 output — produced by the Listing Analyst
and consumed by every downstream specialist in M2-M6.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Fuel(str, Enum):
    PETROL = "petrol"
    DIESEL = "diesel"
    CNG = "cng"
    ELECTRIC = "electric"
    HYBRID = "hybrid"


class Transmission(str, Enum):
    MANUAL = "manual"
    AUTOMATIC = "automatic"


class SellerType(str, Enum):
    INDIVIDUAL = "individual"
    DEALER = "dealer"


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Recommendation(str, Enum):
    BUY = "buy"
    NEGOTIATE = "negotiate"
    WALK_AWAY = "walk_away"
    INVESTIGATE = "investigate"


class Location(BaseModel):
    city: str
    state: str


class FieldConfidence(BaseModel):
    field: str
    confidence: Confidence


class FactSheet(BaseModel):
    """Cleaned-up facts extracted from a used-car listing.

    Fields the listing did not contain MUST be set to null and listed
    in `missing_fields`. Never fabricate.
    """

    make: str = Field(description="Manufacturer, e.g. Maruti, Honda, Hyundai")
    model: str = Field(description="Model name, e.g. Swift, City, i20")
    variant: Optional[str] = Field(description="Trim/variant if specified, e.g. VXi, ZX")
    year: int = Field(description="Year of manufacture")

    fuel: Fuel
    transmission: Optional[Transmission]

    km_driven: int = Field(description="Odometer reading in kilometres")
    owner_count: int = Field(description="Number of previous owners (1, 2, 3, ...)")

    asking_price: int = Field(description="Seller's asking price in INR")
    seller_type: Optional[SellerType]
    listed_on: Optional[str] = Field(description="Listing date if available, free-form string")

    location: Location

    reg_no: Optional[str] = Field(description="Registration number, e.g. TN10AB1234")

    photos: list[str] = Field(description="List of photo URLs from the listing")

    claimed_condition: str = Field(
        description="Seller's free-text claim about condition. Empty string if absent."
    )
    raw_listing_text: str = Field(description="Full original listing text. Always populated.")

    extraction_confidence: list[FieldConfidence] = Field(
        description="One entry per field for which confidence is not 'high'."
    )
    missing_fields: list[str] = Field(
        description="Names of fields the listing did not contain. Set the field to null and list it here."
    )


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class OverallCondition(str, Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"


class ConcernType(str, Enum):
    RUST = "rust"
    PANEL_MISMATCH = "panel_mismatch"
    SCRATCHES_OR_DENTS = "scratches_or_dents"
    INTERIOR_WEAR = "interior_wear"
    ODOMETER_SUSPICION = "odometer_suspicion"
    MISSING_PARTS = "missing_parts"
    AFTERMARKET_MOD = "aftermarket_modification"
    OTHER = "other"


class Concern(BaseModel):
    type: ConcernType
    severity: Severity
    confidence: Confidence
    location: Optional[str] = Field(
        description="Where on the car, e.g. 'rear bumper'. Null if not applicable."
    )
    description: str = Field(description="Specific finding observed in the photo(s).")


class VisualInspectionReport(BaseModel):
    """Visual Inspector's findings from photo analysis (M3+)."""

    concerns: list[Concern] = Field(
        description="Issues identified in the photos. Empty list if everything looks clean."
    )
    missing_angles: list[str] = Field(
        description="Standard photo angles not provided "
                    "(e.g. 'engine bay', 'rear view', 'odometer close-up')."
    )
    overall_condition: OverallCondition = Field(
        description="Overall visual condition based on the photos available."
    )
    photo_count: int = Field(
        description="How many photos were analysed. A very low number is itself a signal."
    )
    notes: str = Field(
        description="Free-text remarks. Empty string if none."
    )


class PriceVerdict(str, Enum):
    FAIR = "fair"
    OVERPRICED = "overpriced"
    UNDERPRICED = "underpriced"
    SUSPICIOUS_LOW = "suspicious_low"


class MarketBand(BaseModel):
    p25: int = Field(description="25th percentile market price in INR.")
    p50: int = Field(description="Median market price in INR.")
    p75: int = Field(description="75th percentile market price in INR.")


class PriceAuditReport(BaseModel):
    """Price Auditor's market comparison (M3+)."""

    market_band: MarketBand = Field(
        description="Market price distribution for comparable cars (same make/model/year/km/location)."
    )
    asking_price: int = Field(
        description="The seller's asking price in INR (echoed from FactSheet for self-contained reading)."
    )
    delta_from_median: int = Field(
        description="asking_price minus market_band.p50, in INR. Positive = above median."
    )
    delta_pct: float = Field(
        description="Percentage difference from median. Positive = above market, negative = below."
    )
    verdict: PriceVerdict = Field(
        description="Overall price assessment. 'suspicious_low' is reserved for prices "
                    "so far below market that fraud, theft, or hidden damage is plausible."
    )
    comparable_listings_count: int = Field(
        description="How many comparable listings the band is based on. Higher = more trustworthy."
    )
    notes: str = Field(description="Free-text remarks. Empty string if none.")


class RCData(BaseModel):
    """What the registration certificate (rc_lookup) returns."""

    reg_no: Optional[str] = Field(description="Registration number, e.g. TN10AB1234.")
    owner_count: Optional[int] = Field(description="Number of registered owners on RC.")
    registration_year: Optional[int] = Field(description="Year of first registration.")
    rto: Optional[str] = Field(description="RTO code, e.g. 'TN10'.")
    fitness_valid: Optional[bool] = Field(
        description="Whether the fitness certificate is currently valid."
    )
    hypothecation: Optional[bool] = Field(
        description="True if a financier has a lien on the vehicle."
    )
    blacklist: bool = Field(description="True if the vehicle is on a blacklist.")
    notes: str = Field(description="Free-text remarks. Empty string if none.")


class InsuranceRecord(BaseModel):
    """What insurance_status returns about the policy."""

    insurer: Optional[str] = Field(description="Name of the insurance provider.")
    valid_until: Optional[str] = Field(
        description="Policy expiry date as a free-form string. Null if unknown."
    )
    lapsed: bool = Field(description="True if the policy is currently lapsed.")
    lapse_duration_days: Optional[int] = Field(
        description="If lapsed, how long ago in days. Null if not lapsed or unknown."
    )
    notes: str = Field(description="Free-text remarks. Empty string if none.")


class ChallanRecord(BaseModel):
    """One challan (traffic fine) entry from challan_check."""

    challan_id: str = Field(description="Challan identifier or reference.")
    offence: str = Field(description="Offence description, e.g. 'overspeeding'.")
    issued_to_name: Optional[str] = Field(
        description="Name the challan was issued to. If different from current "
                    "owner per RC, that is a red flag."
    )
    fine_amount: int = Field(description="Fine amount in INR.")
    paid: bool = Field(description="Whether the fine has been paid.")
    issue_date: Optional[str] = Field(description="Date the challan was issued.")


class HistoryRedFlagType(str, Enum):
    ACCIDENT_HINT = "accident_hint"
    RAPID_OWNERSHIP_CHANGE = "rapid_ownership_change"
    INSURANCE_LAPSED = "insurance_lapsed"
    UNPAID_CHALLAN = "unpaid_challan"
    NAME_MISMATCH = "name_mismatch"
    BLACKLIST = "blacklist"
    HYPOTHECATION_ACTIVE = "hypothecation_active"
    DATA_UNAVAILABLE = "data_unavailable"
    OTHER = "other"


class HistoryRedFlag(BaseModel):
    """A single concern surfaced from the history data."""

    type: HistoryRedFlagType
    severity: Severity
    confidence: Confidence
    description: str = Field(description="Specific finding in plain language.")


class HistoryAssessment(str, Enum):
    CLEAN = "clean"
    MINOR_CONCERNS = "minor_concerns"
    MAJOR_CONCERNS = "major_concerns"
    INSUFFICIENT_DATA = "insufficient_data"


class HistoryReport(BaseModel):
    """History Checker's findings about a vehicle's registration history (M4+).

    Combines RC, insurance, challans, and the IIB accident-lookup guidance.
    The History Checker has no programmatic access to accident records — the
    `iib_recommendation` field carries instructions for the buyer to do that
    check manually.
    """

    rc_data: RCData
    insurance_record: Optional[InsuranceRecord] = Field(
        description="Null if insurance lookup was skipped or unavailable."
    )
    challans: list[ChallanRecord] = Field(
        description="All challans returned by challan_check. Empty list if clean."
    )
    iib_recommendation: str = Field(
        description="Guidance string returned by accident_lookup_hint. Always present."
    )
    red_flags: list[HistoryRedFlag] = Field(
        description="Synthesised red flags. Empty list if the history is genuinely clean."
    )
    overall_assessment: HistoryAssessment
    tools_called: list[str] = Field(
        description="Names of the tools the History Checker actually called "
                    "(for trace / debugging)."
    )
    notes: str = Field(description="Free-text remarks. Empty string if none.")


class Disagreement(BaseModel):
    """A single conflict between specialists that the Coordinator surfaced (M4+).

    Disagreements are first-class output: rather than burying conflicts in
    free-form reasoning, the Coordinator names them explicitly so the buyer
    (and the Critic at M6) can see them.
    """

    between: list[str] = Field(
        description="Names of the parties in conflict, e.g. "
                    "['visual_inspector', 'history_checker'] or "
                    "['seller_claim', 'rc_data']."
    )
    on: str = Field(
        description="What they disagree about, in plain language."
    )
    resolution: str = Field(
        description="How the Coordinator weighted the conflict and what it "
                    "implies for the verdict."
    )


class CriticIssueType(str, Enum):
    UNSUPPORTED_CLAIM = "unsupported_claim"
    IGNORED_DISAGREEMENT = "ignored_disagreement"
    SEVERITY_MISCALIBRATION = "severity_miscalibration"
    MISSING_IIB_NOTE = "missing_iib_note"
    OVERCONFIDENCE = "overconfidence"
    OTHER = "other"


class CriticIssue(BaseModel):
    """One specific issue the Critic flagged in the Coordinator's verdict."""

    type: CriticIssueType
    severity: Severity
    description: str = Field(
        description="What is wrong, in plain language. Quote specific phrasing "
                    "from the verdict where helpful."
    )
    suggested_fix: str = Field(
        description="What the Coordinator should change to address this. "
                    "Concrete enough that the next revision attempt knows what to do."
    )


class CriticReview(BaseModel):
    """The Critic's review of the Coordinator's verdict (M6+).

    `approved=True` means the verdict can ship. `approved=False` means the
    Coordinator should revise; the `issues` list tells it what to fix.
    """

    approved: bool = Field(
        description="True if the verdict is acceptable as-is. False if revision is needed."
    )
    issues: list[CriticIssue] = Field(
        description="Specific issues. Empty list when approved=True."
    )
    summary: str = Field(
        description="One-sentence overall assessment of the verdict's quality."
    )
    revision_round: int = Field(
        description="Which revision round this is. 0 = the first verdict; 1+ = "
                    "subsequent attempts after Critic feedback."
    )


class Verdict(BaseModel):
    """The Coordinator's final synthesis for a single listing.

    Grows in shape across milestones: at M2 it was a thin wrapper around the
    Listing Analyst; at M3 it draws on three specialists; at M4 it adds the
    `disagreements` field to surface conflicts as structured output.
    """

    recommendation: Recommendation = Field(
        description="The single most important field — what should the buyer do?"
    )
    confidence: Confidence = Field(
        description="How sure the Coordinator is about the recommendation."
    )
    headline: str = Field(
        description="One-sentence summary the buyer reads first. Plain English."
    )
    reasoning: str = Field(
        description="2-3 short paragraphs explaining the recommendation, "
                    "drawing on the consulted specialists' findings."
    )
    key_concerns: list[str] = Field(
        description="Bullet-style list of things the buyer should know. "
                    "Empty list if the listing is genuinely clean."
    )
    disagreements: list[Disagreement] = Field(
        description="Conflicts surfaced between specialists. Empty list only "
                    "if all specialists genuinely agree. Do not paper over "
                    "conflicts in `reasoning` — surface them here too."
    )
    specialists_consulted: list[str] = Field(
        description="Names of the specialist agents whose findings were used "
                    "(e.g. 'listing_analyst'). For traceability."
    )


if __name__ == "__main__":
    # Demo: build a sample FactSheet and show what it looks like.
    # This block runs only when you do `python schemas.py` directly.
    import json

    sample = FactSheet(
        make="Maruti",
        model="Swift",
        variant="VXi",
        year=2018,
        fuel=Fuel.PETROL,
        transmission=Transmission.MANUAL,
        km_driven=45000,
        owner_count=2,
        asking_price=485000,
        seller_type=SellerType.INDIVIDUAL,
        listed_on="2026-04-15",
        location=Location(city="Chennai", state="Tamil Nadu"),
        reg_no="TN10AB1234",
        photos=["https://example.com/photo1.jpg", "https://example.com/photo2.jpg"],
        claimed_condition="Single owner from new, lovingly maintained, all service records available.",
        raw_listing_text="2018 Maruti Swift VXi, 45000 km, 2nd owner, Chennai. Asking 4.85L. Reg TN10AB1234.",
        extraction_confidence=[
            FieldConfidence(field="listed_on", confidence=Confidence.MEDIUM),
        ],
        missing_fields=[],
    )

    print("=" * 60)
    print("Sample FactSheet (Python object)")
    print("=" * 60)
    print(sample)

    print("\n" + "=" * 60)
    print("Same FactSheet as JSON (this is what flows between agents)")
    print("=" * 60)
    print(json.dumps(sample.model_dump(mode="json"), indent=2))

    # M2 — the Coordinator's output.
    sample_verdict = Verdict(
        recommendation=Recommendation.NEGOTIATE,
        confidence=Confidence.MEDIUM,
        headline="Reasonable buy at 4.5L, but ask for service records before closing.",
        reasoning=(
            "The Listing Analyst extracted a clean fact sheet for a 2018 Maruti Swift VXi "
            "with 45,000 km and one previous owner. The asking price of 4.85L is in line "
            "with market expectations for this configuration in Chennai. The seller's "
            "claimed condition is plausible. No specialists have flagged red flags yet.\n\n"
            "Recommendation is to negotiate down to ~4.5L on the basis that you have not "
            "personally verified the service history. Walk away if records are not available."
        ),
        key_concerns=[
            "Service records claimed but not yet verified",
            "Insurance validity should be confirmed in person",
        ],
        specialists_consulted=["listing_analyst"],
    )

    print("\n" + "=" * 60)
    print("Sample Verdict (the Coordinator's M2 output)")
    print("=" * 60)
    print(json.dumps(sample_verdict.model_dump(mode="json"), indent=2))
