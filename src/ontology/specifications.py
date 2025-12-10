"""
Specification Ontology - Defines the canonical schema for pressure transmitter specifications.

This module handles:
1. WHAT specs matter (the ontology definition)
2. HOW to normalize units (psi ↔ bar ↔ kPa, °C ↔ °F, mm ↔ inches)
3. HOW to parse raw spec strings into structured data
4. HOW to enable head-to-head comparisons
5. FUZZY MATCHING with aliases and similarity scoring
6. AI-DERIVED ATTRIBUTES for specs not in the ontology

Key Design Decision (from boss's question):
"Will that be handled by a human defining what's important or will the AI try to identify those?"

ANSWER: Hybrid approach
- Human defines the ONTOLOGY (what specs exist, valid units, how to normalize)
- AI extracts and maps data ONTO the ontology
- AI can propose NEW attributes tagged as AI_DERIVED_ATTRIBUTE
- Fuzzy matching allows partial matches with similarity > 0.6
"""

from __future__ import annotations
import re
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from difflib import SequenceMatcher


# =============================================================================
# UNIT CONVERSION TABLES
# =============================================================================

# Pressure conversions (all to psi as canonical)
PRESSURE_TO_PSI = {
    "psi": 1.0,
    "psig": 1.0,
    "psia": 1.0,
    "bar": 14.5038,
    "bara": 14.5038,
    "barg": 14.5038,
    "mbar": 0.0145038,
    "kpa": 0.145038,
    "mpa": 145.038,
    "pa": 0.000145038,
    "kgcm2": 14.2233,  # kg/cm²
    "kg/cm2": 14.2233,
    "kg/cm²": 14.2233,
    "atm": 14.6959,
    "mmhg": 0.0193368,
    "inhg": 0.491154,
    "inh2o": 0.0361273,  # inches of water
    "mmh2o": 0.00142233,
    "torr": 0.0193368,
}

# Reverse mapping for display
PSI_TO_OTHER = {
    "bar": 1 / 14.5038,
    "kpa": 1 / 0.145038,
    "mpa": 1 / 145.038,
    "mbar": 1 / 0.0145038,
}

# Temperature conversions (all to Celsius as canonical)
def fahrenheit_to_celsius(f: float) -> float:
    return (f - 32) * 5 / 9

def celsius_to_celsius(c: float) -> float:
    return c

def kelvin_to_celsius(k: float) -> float:
    return k - 273.15

TEMP_CONVERTERS = {
    "c": celsius_to_celsius,
    "celsius": celsius_to_celsius,
    "°c": celsius_to_celsius,
    "deg c": celsius_to_celsius,
    "degc": celsius_to_celsius,
    "f": fahrenheit_to_celsius,
    "fahrenheit": fahrenheit_to_celsius,
    "°f": fahrenheit_to_celsius,
    "deg f": fahrenheit_to_celsius,
    "degf": fahrenheit_to_celsius,
    "k": kelvin_to_celsius,
    "kelvin": kelvin_to_celsius,
}

# Length conversions (all to mm as canonical)
LENGTH_TO_MM = {
    "mm": 1.0,
    "cm": 10.0,
    "m": 1000.0,
    "in": 25.4,
    "inch": 25.4,
    "inches": 25.4,
    '"': 25.4,
    "ft": 304.8,
    "feet": 304.8,
}

# Time conversions (all to ms as canonical)
TIME_TO_MS = {
    "ms": 1.0,
    "millisecond": 1.0,
    "milliseconds": 1.0,
    "s": 1000.0,
    "sec": 1000.0,
    "second": 1000.0,
    "seconds": 1000.0,
    "min": 60000.0,
    "minute": 60000.0,
    "minutes": 60000.0,
}


# =============================================================================
# SPECIFICATION TYPE DEFINITIONS
# =============================================================================

class SpecType(Enum):
    """Types of specification values."""
    RANGE = "range"           # Min to max (e.g., pressure range: 0-6000 psi)
    VALUE = "value"           # Single value (e.g., accuracy: ±0.075%)
    ENUM = "enum"             # One of allowed values (e.g., output: 4-20mA)
    LIST = "list"             # Multiple values (e.g., materials: [SS316, Hastelloy])
    BOOLEAN = "boolean"       # Yes/no (e.g., HART enabled: true)
    TEXT = "text"             # Free text (e.g., certifications: "ATEX, IECEx, SIL2")


@dataclass
class SpecDefinition:
    """Definition of a specification attribute in the ontology."""
    name: str                         # Canonical name (e.g., "pressure_range")
    display_name: str                 # Human-readable (e.g., "Pressure Range")
    spec_type: SpecType               # Type of value
    canonical_unit: Optional[str]     # Unit for normalization (e.g., "psi")
    allowed_values: Optional[List[str]] = None  # For ENUM types
    extraction_hints: List[str] = None  # Regex/keywords to look for
    aliases: List[str] = None         # Alternative names for fuzzy matching
    importance: int = 1               # 1-5 priority for comparisons
    
    def __post_init__(self):
        if self.extraction_hints is None:
            self.extraction_hints = []
        if self.aliases is None:
            self.aliases = []


@dataclass
class NormalizedValue:
    """A normalized specification value with both raw and converted forms."""
    raw_value: str                    # Original extracted text
    raw_unit: str                     # Original unit
    normalized_value: Union[float, Tuple[float, float], str, List[str]]  # Converted value
    normalized_unit: str              # Canonical unit
    confidence: float = 1.0           # Confidence in the conversion


@dataclass 
class AIExtractedSpec:
    """A specification discovered by AI that may not be in the ontology."""
    name: str                         # Extracted name
    value: str                        # Extracted value
    raw_text: str                     # Source text snippet
    mapped_ontology_key: Optional[str] = None  # If mapped to ontology
    similarity_score: float = 0.0     # Fuzzy match score
    is_ai_derived: bool = False       # True if not in ontology
    source_url: str = ""              # Source URL


# =============================================================================
# THE ONTOLOGY: What specifications matter for pressure transmitters
# =============================================================================

PRESSURE_TRANSMITTER_ONTOLOGY: Dict[str, SpecDefinition] = {
    # === PERFORMANCE SPECIFICATIONS (Highest importance) ===
    "pressure_range": SpecDefinition(
        name="pressure_range",
        display_name="Pressure Range",
        spec_type=SpecType.RANGE,
        canonical_unit="psi",
        extraction_hints=[
            r"(\d+\.?\d*)\s*(?:to|-|–|~)\s*(\d+\.?\d*)\s*(psi|bar|mbar|kpa|mpa)",
            r"range[:\s]+(\d+\.?\d*)\s*(?:to|-|–)\s*(\d+\.?\d*)",
            r"pressure\s+range",
            r"measuring\s+range",
            r"span[:\s]+",
            r"turndown",
        ],
        aliases=[
            "measuring range", "pressure span", "measurement range", 
            "operating range", "span", "pressure scale", "range of measurement",
            "full scale range", "fs range", "measurement span"
        ],
        importance=5,
    ),
    
    "accuracy": SpecDefinition(
        name="accuracy",
        display_name="Accuracy",
        spec_type=SpecType.VALUE,
        canonical_unit="percent_fs",  # Percent of full scale
        extraction_hints=[
            r"accuracy[:\s]+[±]?(\d+\.?\d*)\s*%",
            r"[±](\d+\.?\d*)\s*%\s*(?:fs|full\s*scale|span)",
            r"(\d+\.?\d*)\s*%\s*accuracy",
            r"reference\s+accuracy",
            r"total\s+accuracy",
        ],
        aliases=[
            "reference accuracy", "total accuracy", "measurement accuracy",
            "typical accuracy", "best accuracy", "standard accuracy",
            "accuracy class", "error", "measurement error", "max error"
        ],
        importance=5,
    ),
    
    "repeatability": SpecDefinition(
        name="repeatability",
        display_name="Repeatability",
        spec_type=SpecType.VALUE,
        canonical_unit="percent_fs",
        extraction_hints=[
            r"repeatability[:\s]+[±]?(\d+\.?\d*)\s*%",
            r"[±](\d+\.?\d*)\s*%\s*repeatability",
        ],
        aliases=[
            "repeat accuracy", "reproducibility", "precision",
            "repeat precision", "measurement repeatability"
        ],
        importance=4,
    ),
    
    "stability": SpecDefinition(
        name="stability",
        display_name="Long-term Stability",
        spec_type=SpecType.TEXT,  # Often expressed as "±0.1% over 5 years"
        canonical_unit=None,
        extraction_hints=[
            r"stability[:\s]+(.+?)(?:\n|$)",
            r"long.?term\s+stability",
            r"drift[:\s]+(.+?)(?:\n|$)",
            r"(\d+\.?\d*)\s*%\s*(?:per|\/)\s*year",
        ],
        aliases=[
            "long term stability", "drift", "zero drift", "span drift",
            "annual drift", "yearly stability", "long-term drift"
        ],
        importance=4,
    ),
    
    "response_time": SpecDefinition(
        name="response_time",
        display_name="Response Time",
        spec_type=SpecType.VALUE,
        canonical_unit="ms",
        extraction_hints=[
            r"response\s+time[:\s]+(\d+\.?\d*)\s*(ms|s|sec)",
            r"(\d+\.?\d*)\s*(ms|s)\s+response",
            r"update\s+time",
            r"settling\s+time",
        ],
        aliases=[
            "update time", "settling time", "reaction time", 
            "time constant", "step response", "dynamic response"
        ],
        importance=3,
    ),
    
    "overpressure": SpecDefinition(
        name="overpressure",
        display_name="Overpressure Limit",
        spec_type=SpecType.VALUE,
        canonical_unit="psi",
        extraction_hints=[
            r"overpressure[:\s]+(\d+\.?\d*)\s*(psi|bar|mpa)",
            r"burst\s+pressure",
            r"max(?:imum)?\s+pressure",
            r"proof\s+pressure",
        ],
        aliases=[
            "burst pressure", "maximum pressure", "proof pressure",
            "overload", "pressure limit", "static pressure limit"
        ],
        importance=4,
    ),
    
    "turn_down_ratio": SpecDefinition(
        name="turn_down_ratio",
        display_name="Turn Down Ratio",
        spec_type=SpecType.TEXT,
        canonical_unit=None,
        extraction_hints=[
            r"turn.?down[:\s]+(\d+)[:\s]*1",
            r"rangeability[:\s]+(\d+)",
        ],
        aliases=[
            "rangeability", "turndown", "range ratio", "span ratio"
        ],
        importance=3,
    ),
    
    # === ELECTRICAL SPECIFICATIONS ===
    "output_signal": SpecDefinition(
        name="output_signal",
        display_name="Output Signal",
        spec_type=SpecType.LIST,  # Can have multiple (e.g., "4-20mA with HART")
        canonical_unit=None,
        allowed_values=[
            "4-20mA", "0-20mA", "0-10V", "1-5V", "0-5V",
            "HART", "Profibus PA", "Foundation Fieldbus",
            "Modbus RTU", "Modbus TCP", "IO-Link", "WirelessHART",
            "Bluetooth", "Digital", "Analog"
        ],
        extraction_hints=[
            r"output[:\s]+(4-20\s*ma|0-10\s*v|hart|profibus|modbus)",
            r"(4-20\s*ma|0-20\s*ma)",
            r"(hart|profibus|foundation\s*fieldbus|modbus)",
            r"signal[:\s]+(4-20|0-10|1-5)",
        ],
        aliases=[
            "signal output", "output type", "communication protocol",
            "protocol", "output format", "electrical output",
            "analog output", "digital output", "fieldbus"
        ],
        importance=5,
    ),
    
    "supply_voltage": SpecDefinition(
        name="supply_voltage",
        display_name="Supply Voltage",
        spec_type=SpecType.RANGE,
        canonical_unit="V DC",
        extraction_hints=[
            r"supply[:\s]+(\d+\.?\d*)\s*(?:to|-|–)\s*(\d+\.?\d*)\s*v",
            r"power[:\s]+(\d+\.?\d*)\s*v\s*dc",
            r"(\d+)\s*v\s*dc",
            r"voltage[:\s]+(\d+)",
        ],
        aliases=[
            "power supply", "operating voltage", "input voltage",
            "excitation voltage", "dc supply", "loop voltage"
        ],
        importance=3,
    ),
    
    "load_resistance": SpecDefinition(
        name="load_resistance",
        display_name="Load Resistance",
        spec_type=SpecType.VALUE,
        canonical_unit="ohm",
        extraction_hints=[
            r"load[:\s]+(\d+)\s*(?:ohm|Ω)",
            r"(\d+)\s*(?:ohm|Ω)\s+max",
            r"loop\s+resistance",
        ],
        aliases=[
            "loop resistance", "max load", "load impedance"
        ],
        importance=2,
    ),
    
    "power_consumption": SpecDefinition(
        name="power_consumption",
        display_name="Power Consumption",
        spec_type=SpecType.VALUE,
        canonical_unit="W",
        extraction_hints=[
            r"power\s+consumption[:\s]+(\d+\.?\d*)\s*(w|mw)",
            r"(\d+\.?\d*)\s*(w|mw)\s+(?:max|typical)",
        ],
        aliases=[
            "power draw", "current consumption", "energy consumption"
        ],
        importance=2,
    ),
    
    # === PHYSICAL SPECIFICATIONS ===
    "process_connection": SpecDefinition(
        name="process_connection",
        display_name="Process Connection",
        spec_type=SpecType.LIST,
        canonical_unit=None,
        allowed_values=[
            "1/4 NPT", "1/2 NPT", "3/4 NPT", "1 NPT",
            "G1/4", "G1/2", "G3/4", "G1",
            "M20x1.5", "M14x1.5",
            "Tri-Clamp", "Flange", "DIN", "JIS",
            "SAE", "BSP", "ISO"
        ],
        extraction_hints=[
            r"(1/4|1/2|3/4|1)\s*(npt|bsp)",
            r"(g1/4|g1/2|m20|m14)",
            r"process\s+connection[:\s]+(.+?)(?:\n|,|$)",
            r"(tri.?clamp|flange)",
            r"thread[:\s]+(.+?)(?:\n|$)",
        ],
        aliases=[
            "connection type", "fitting", "port", "thread type",
            "pressure port", "mounting", "process fitting"
        ],
        importance=4,
    ),
    
    "wetted_materials": SpecDefinition(
        name="wetted_materials",
        display_name="Wetted Materials",
        spec_type=SpecType.LIST,
        canonical_unit=None,
        allowed_values=[
            "316 SS", "316L SS", "304 SS",
            "Hastelloy C-276", "Monel 400",
            "Titanium", "Tantalum",
            "PTFE", "Viton", "EPDM", "FKM",
            "Ceramic", "Inconel", "Duplex"
        ],
        extraction_hints=[
            r"wetted[:\s]+(.+?)(?:\n|$)",
            r"material[:\s]+(316|hastelloy|monel|titanium)",
            r"(stainless\s+steel|ss\s*316|hastelloy)",
            r"diaphragm[:\s]+(316|hastelloy|ceramic)",
            r"sensor\s+material",
        ],
        aliases=[
            "wetted parts", "media contact materials", "diaphragm material",
            "sensor material", "process wetted parts", "materials of construction"
        ],
        importance=4,
    ),
    
    "housing_material": SpecDefinition(
        name="housing_material",
        display_name="Housing Material",
        spec_type=SpecType.ENUM,
        canonical_unit=None,
        allowed_values=["Aluminum", "316 SS", "Plastic", "Die-cast Aluminum", "Stainless Steel"],
        extraction_hints=[
            r"housing[:\s]+(.+?)(?:\n|$)",
            r"enclosure[:\s]+(aluminum|stainless|plastic)",
            r"case[:\s]+(aluminum|steel|plastic)",
        ],
        aliases=[
            "enclosure material", "case material", "body material"
        ],
        importance=2,
    ),
    
    "diaphragm_material": SpecDefinition(
        name="diaphragm_material",
        display_name="Diaphragm Material",
        spec_type=SpecType.ENUM,
        canonical_unit=None,
        allowed_values=["316L SS", "Hastelloy C-276", "Monel", "Titanium", "Ceramic", "Inconel"],
        extraction_hints=[
            r"diaphragm[:\s]+(316|hastelloy|monel|titanium|ceramic)",
            r"sensing\s+element[:\s]+(.+?)(?:\n|$)",
        ],
        aliases=[
            "sensing element", "pressure element", "membrane material"
        ],
        importance=4,
    ),
    
    # === ENVIRONMENTAL SPECIFICATIONS ===
    "operating_temp": SpecDefinition(
        name="operating_temp",
        display_name="Operating Temperature",
        spec_type=SpecType.RANGE,
        canonical_unit="celsius",
        extraction_hints=[
            r"operating\s+temp[:\s]+(-?\d+)\s*(?:to|~|-|–)\s*(-?\d+)\s*(°?[cf])",
            r"ambient[:\s]+(-?\d+)\s*(?:to|-|–)\s*(-?\d+)",
            r"(-40|–40)\s*(?:to|-|–)\s*(\d+)\s*(°?c)",
            r"temperature\s+range[:\s]+",
        ],
        aliases=[
            "ambient temperature", "electronics temperature", 
            "environmental temperature", "working temperature"
        ],
        importance=4,
    ),
    
    "process_temp": SpecDefinition(
        name="process_temp",
        display_name="Process/Media Temperature",
        spec_type=SpecType.RANGE,
        canonical_unit="celsius",
        extraction_hints=[
            r"process\s+temp[:\s]+(-?\d+)\s*(?:to|-|–)\s*(-?\d+)",
            r"media\s+temp[:\s]+(-?\d+)\s*(?:to|-|–)\s*(-?\d+)",
            r"fluid\s+temp",
            r"medium\s+temperature",
        ],
        aliases=[
            "media temperature", "fluid temperature", "medium temperature",
            "service temperature", "max process temperature"
        ],
        importance=4,
    ),
    
    "storage_temp": SpecDefinition(
        name="storage_temp",
        display_name="Storage Temperature",
        spec_type=SpecType.RANGE,
        canonical_unit="celsius",
        extraction_hints=[
            r"storage\s+temp[:\s]+(-?\d+)\s*(?:to|-|–)\s*(-?\d+)",
        ],
        aliases=[
            "storage range", "shelf temperature"
        ],
        importance=2,
    ),
    
    "ip_rating": SpecDefinition(
        name="ip_rating",
        display_name="IP/Ingress Protection Rating",
        spec_type=SpecType.ENUM,
        canonical_unit=None,
        allowed_values=["IP65", "IP66", "IP67", "IP68", "IP69K", "NEMA 4X", "NEMA 4", "NEMA 7"],
        extraction_hints=[
            r"(ip\s*6[5-9]k?|ip\s*68|nema\s*4x?|nema\s*7)",
            r"protection[:\s]+(ip\d+)",
            r"ingress[:\s]+(ip\d+)",
            r"enclosure\s+rating",
        ],
        aliases=[
            "protection rating", "ingress protection", "enclosure rating",
            "nema rating", "environmental protection"
        ],
        importance=3,
    ),
    
    "hazardous_area": SpecDefinition(
        name="hazardous_area",
        display_name="Hazardous Area Certification",
        spec_type=SpecType.LIST,
        canonical_unit=None,
        allowed_values=[
            "ATEX", "IECEx", "FM", "CSA", "UL",
            "Class I Div 1", "Class I Div 2",
            "Zone 0", "Zone 1", "Zone 2",
            "Intrinsically Safe", "Explosion Proof"
        ],
        extraction_hints=[
            r"(atex|iecex|fm\s+approved|csa)",
            r"(class\s*i\s*div\s*[12])",
            r"(zone\s*[012])",
            r"explosion.?proof",
            r"intrinsically\s+safe",
            r"hazardous\s+area",
        ],
        aliases=[
            "ex certification", "explosion proof", "intrinsic safety",
            "hazloc", "hazardous location", "atex certification"
        ],
        importance=4,
    ),
    
    "sil_rating": SpecDefinition(
        name="sil_rating",
        display_name="SIL Rating",
        spec_type=SpecType.ENUM,
        canonical_unit=None,
        allowed_values=["SIL1", "SIL2", "SIL3"],
        extraction_hints=[
            r"(sil\s*[123])",
            r"safety\s+integrity\s+level",
            r"iec\s*61508",
            r"iec\s*61511",
        ],
        aliases=[
            "safety integrity level", "functional safety", "sil certified"
        ],
        importance=4,
    ),
    
    # === MEASUREMENT TYPE ===
    "measurement_type": SpecDefinition(
        name="measurement_type",
        display_name="Measurement Type",
        spec_type=SpecType.ENUM,
        canonical_unit=None,
        allowed_values=["Gauge", "Absolute", "Differential", "Sealed Gauge", "Compound"],
        extraction_hints=[
            r"(gauge|absolute|differential|sealed|compound)",
            r"pressure\s+type[:\s]+(gauge|absolute|differential)",
            r"measurement\s+type",
        ],
        aliases=[
            "pressure type", "sensing type", "reference type"
        ],
        importance=5,
    ),
    
    # === PHYSICAL DIMENSIONS ===
    "dimensions": SpecDefinition(
        name="dimensions",
        display_name="Dimensions (LxWxH)",
        spec_type=SpecType.TEXT,
        canonical_unit="mm",
        extraction_hints=[
            r"dimensions[:\s]+(.+?)(?:\n|$)",
            r"(\d+)\s*x\s*(\d+)\s*x\s*(\d+)\s*(mm|in)",
            r"size[:\s]+(.+?)(?:\n|$)",
        ],
        aliases=[
            "size", "physical dimensions", "overall dimensions"
        ],
        importance=2,
    ),
    
    "weight": SpecDefinition(
        name="weight",
        display_name="Weight",
        spec_type=SpecType.VALUE,
        canonical_unit="g",
        extraction_hints=[
            r"weight[:\s]+(\d+\.?\d*)\s*(g|kg|lb|oz)",
            r"(\d+\.?\d*)\s*(kg|g)\s+weight",
            r"mass[:\s]+",
        ],
        aliases=[
            "mass", "net weight", "product weight"
        ],
        importance=2,
    ),
    
    # === CERTIFICATIONS ===
    "certifications": SpecDefinition(
        name="certifications",
        display_name="Certifications",
        spec_type=SpecType.LIST,
        canonical_unit=None,
        allowed_values=["CE", "UL", "CSA", "FM", "ATEX", "IECEx", "CRN", "PED", "CPA", "MID"],
        extraction_hints=[
            r"(ce|ul|csa|fm|ped|crn|cpa|mid)\s+(?:certified|approved|listed)",
            r"certifications?[:\s]+(.+?)(?:\n|$)",
            r"approvals?[:\s]+(.+?)(?:\n|$)",
        ],
        aliases=[
            "approvals", "listings", "compliance", "standards"
        ],
        importance=3,
    ),
    
    # === WARRANTY & LIFECYCLE ===
    "warranty": SpecDefinition(
        name="warranty",
        display_name="Warranty",
        spec_type=SpecType.TEXT,
        canonical_unit=None,
        extraction_hints=[
            r"warranty[:\s]+(\d+)\s*(year|month)",
            r"(\d+)\s*year\s+warranty",
        ],
        aliases=[
            "guarantee", "product warranty"
        ],
        importance=2,
    ),
    
    "mtbf": SpecDefinition(
        name="mtbf",
        display_name="MTBF (Mean Time Between Failures)",
        spec_type=SpecType.VALUE,
        canonical_unit="hours",
        extraction_hints=[
            r"mtbf[:\s]+(\d+\.?\d*)\s*(hours?|years?)",
            r"mean\s+time\s+between\s+failures",
        ],
        aliases=[
            "mean time between failures", "reliability", "service life"
        ],
        importance=3,
    ),
}


# =============================================================================
# FUZZY MATCHING UTILITIES
# =============================================================================

def calculate_similarity(s1: str, s2: str) -> float:
    """Calculate similarity ratio between two strings (0-1)."""
    s1 = s1.lower().strip()
    s2 = s2.lower().strip()
    return SequenceMatcher(None, s1, s2).ratio()


def find_best_ontology_match(
    extracted_name: str,
    threshold: float = 0.6
) -> Tuple[Optional[str], float]:
    """
    Find the best matching ontology key for an extracted specification name.
    
    Uses fuzzy matching with aliases to find the best match.
    
    Args:
        extracted_name: The name extracted from the page
        threshold: Minimum similarity score to consider a match (default 0.6)
    
    Returns:
        Tuple of (ontology_key, similarity_score) or (None, 0.0) if no match
    """
    extracted_lower = extracted_name.lower().strip()
    best_match = None
    best_score = 0.0
    
    for key, spec_def in PRESSURE_TRANSMITTER_ONTOLOGY.items():
        # Check exact match with key
        if extracted_lower == key.replace("_", " "):
            return (key, 1.0)
        
        # Check exact match with display name
        if extracted_lower == spec_def.display_name.lower():
            return (key, 1.0)
        
        # Check similarity with key
        score = calculate_similarity(extracted_lower, key.replace("_", " "))
        if score > best_score:
            best_score = score
            best_match = key
        
        # Check similarity with display name
        score = calculate_similarity(extracted_lower, spec_def.display_name.lower())
        if score > best_score:
            best_score = score
            best_match = key
        
        # Check against aliases
        for alias in spec_def.aliases:
            # Exact alias match
            if extracted_lower == alias.lower():
                return (key, 1.0)
            
            score = calculate_similarity(extracted_lower, alias.lower())
            if score > best_score:
                best_score = score
                best_match = key
    
    if best_score >= threshold:
        return (best_match, best_score)
    
    return (None, 0.0)


def match_partial_phrase(text: str, spec_key: str) -> bool:
    """
    Check if text contains a partial match for any extraction hints or aliases.
    
    Args:
        text: The text to search in
        spec_key: The ontology key to check against
    
    Returns:
        True if a partial match is found
    """
    if spec_key not in PRESSURE_TRANSMITTER_ONTOLOGY:
        return False
    
    spec_def = PRESSURE_TRANSMITTER_ONTOLOGY[spec_key]
    text_lower = text.lower()
    
    # Check extraction hints (these are regexes)
    for hint in spec_def.extraction_hints:
        try:
            if re.search(hint, text_lower, re.IGNORECASE):
                return True
        except re.error:
            # Treat as simple string match if regex fails
            if hint.lower() in text_lower:
                return True
    
    # Check aliases as partial matches
    for alias in spec_def.aliases:
        if alias.lower() in text_lower:
            return True
    
    # Check display name
    if spec_def.display_name.lower() in text_lower:
        return True
    
    return False


# =============================================================================
# UNIT NORMALIZATION FUNCTIONS
# =============================================================================

def normalize_pressure(value: float, unit: str) -> NormalizedValue:
    """Convert pressure to PSI (canonical unit) and return both raw and normalized."""
    unit_lower = unit.lower().replace(" ", "").replace(".", "")
    multiplier = PRESSURE_TO_PSI.get(unit_lower, 1.0)
    normalized = value * multiplier
    
    return NormalizedValue(
        raw_value=str(value),
        raw_unit=unit,
        normalized_value=normalized,
        normalized_unit="psi",
        confidence=1.0 if unit_lower in PRESSURE_TO_PSI else 0.5
    )


def normalize_temperature(value: float, unit: str) -> NormalizedValue:
    """Convert temperature to Celsius (canonical unit)."""
    unit_lower = unit.lower().replace("°", "").strip()
    converter = TEMP_CONVERTERS.get(unit_lower, celsius_to_celsius)
    normalized = converter(value)
    
    return NormalizedValue(
        raw_value=str(value),
        raw_unit=unit,
        normalized_value=normalized,
        normalized_unit="celsius",
        confidence=1.0 if unit_lower in TEMP_CONVERTERS else 0.5
    )


def normalize_length(value: float, unit: str) -> NormalizedValue:
    """Convert length to mm (canonical unit)."""
    unit_lower = unit.lower().strip()
    multiplier = LENGTH_TO_MM.get(unit_lower, 1.0)
    normalized = value * multiplier
    
    return NormalizedValue(
        raw_value=str(value),
        raw_unit=unit,
        normalized_value=normalized,
        normalized_unit="mm",
        confidence=1.0 if unit_lower in LENGTH_TO_MM else 0.5
    )


def normalize_time(value: float, unit: str) -> NormalizedValue:
    """Convert time to milliseconds (canonical unit)."""
    unit_lower = unit.lower().strip()
    multiplier = TIME_TO_MS.get(unit_lower, 1.0)
    normalized = value * multiplier
    
    return NormalizedValue(
        raw_value=str(value),
        raw_unit=unit,
        normalized_value=normalized,
        normalized_unit="ms",
        confidence=1.0 if unit_lower in TIME_TO_MS else 0.5
    )


def normalize_spec_value(spec_key: str, value: Any, unit: str = "") -> NormalizedValue:
    """
    Normalize a specification value based on its ontology definition.
    
    Args:
        spec_key: The ontology key
        value: The raw value
        unit: The unit (if applicable)
    
    Returns:
        NormalizedValue with both raw and normalized forms
    """
    if spec_key not in PRESSURE_TRANSMITTER_ONTOLOGY:
        return NormalizedValue(
            raw_value=str(value),
            raw_unit=unit,
            normalized_value=value,
            normalized_unit=unit,
            confidence=0.5
        )
    
    spec_def = PRESSURE_TRANSMITTER_ONTOLOGY[spec_key]
    canonical_unit = spec_def.canonical_unit
    
    try:
        if canonical_unit == "psi" and isinstance(value, (int, float)):
            return normalize_pressure(float(value), unit)
        elif canonical_unit == "celsius" and isinstance(value, (int, float)):
            return normalize_temperature(float(value), unit)
        elif canonical_unit == "mm" and isinstance(value, (int, float)):
            return normalize_length(float(value), unit)
        elif canonical_unit == "ms" and isinstance(value, (int, float)):
            return normalize_time(float(value), unit)
        else:
            return NormalizedValue(
                raw_value=str(value),
                raw_unit=unit,
                normalized_value=value,
                normalized_unit=canonical_unit or unit,
                confidence=1.0
            )
    except (ValueError, TypeError):
        return NormalizedValue(
            raw_value=str(value),
            raw_unit=unit,
            normalized_value=value,
            normalized_unit=unit,
            confidence=0.3
        )


# =============================================================================
# PARSING UTILITIES
# =============================================================================

def parse_range(text: str, unit_converter=None) -> Optional[Tuple[float, float]]:
    """Parse a range like '0-6000 psi' or '10 to 100 bar'."""
    # Pattern: number (to/-/~) number (unit)
    pattern = r"(-?\d+\.?\d*)\s*(?:to|–|-|~)\s*(-?\d+\.?\d*)\s*(\w+)?"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        min_val = float(match.group(1))
        max_val = float(match.group(2))
        unit = match.group(3) or ""
        
        if unit_converter:
            min_val = unit_converter(min_val, unit)
            max_val = unit_converter(max_val, unit)
        
        return (min_val, max_val)
    return None


def parse_percentage(text: str) -> Optional[float]:
    """Parse accuracy/repeatability like '±0.075%' or '0.1% FS'."""
    pattern = r"[±]?\s*(\d+\.?\d*)\s*%"
    match = re.search(pattern, text)
    if match:
        return float(match.group(1))
    return None


def extract_number_with_unit(text: str) -> Optional[Tuple[float, str]]:
    """Extract a number and its unit from text."""
    pattern = r"(-?\d+\.?\d*)\s*([a-zA-Z°/%]+)"
    match = re.search(pattern, text)
    if match:
        return (float(match.group(1)), match.group(2))
    return None


# =============================================================================
# AI-DERIVED ATTRIBUTE HANDLING
# =============================================================================

# Store for AI-discovered attributes not in the ontology
AI_DERIVED_ATTRIBUTES: Dict[str, Dict[str, Any]] = {}


def register_ai_derived_attribute(
    name: str,
    value: str,
    source_url: str,
    occurrence_count: int = 1
) -> str:
    """
    Register a new AI-discovered attribute that doesn't map to the ontology.
    
    Args:
        name: The attribute name discovered
        value: Example value
        source_url: Where it was found
        occurrence_count: How many times it's been seen
    
    Returns:
        The normalized attribute key
    """
    key = name.lower().replace(" ", "_").replace("-", "_")
    key = re.sub(r'[^a-z0-9_]', '', key)
    
    if key in AI_DERIVED_ATTRIBUTES:
        AI_DERIVED_ATTRIBUTES[key]["occurrence_count"] += occurrence_count
        if source_url not in AI_DERIVED_ATTRIBUTES[key]["sources"]:
            AI_DERIVED_ATTRIBUTES[key]["sources"].append(source_url)
        if value not in AI_DERIVED_ATTRIBUTES[key]["example_values"]:
            AI_DERIVED_ATTRIBUTES[key]["example_values"].append(value)
    else:
        AI_DERIVED_ATTRIBUTES[key] = {
            "original_name": name,
            "normalized_key": key,
            "example_values": [value],
            "sources": [source_url],
            "occurrence_count": occurrence_count,
            "is_ai_derived": True,
        }
    
    return key


def get_ai_derived_attributes() -> Dict[str, Dict[str, Any]]:
    """Get all AI-discovered attributes."""
    return AI_DERIVED_ATTRIBUTES.copy()


def get_frequently_seen_ai_attributes(min_occurrences: int = 3) -> List[Dict[str, Any]]:
    """Get AI-derived attributes seen multiple times (candidates for ontology)."""
    return [
        attr for attr in AI_DERIVED_ATTRIBUTES.values()
        if attr["occurrence_count"] >= min_occurrences
    ]


# =============================================================================
# PROMPT GENERATION FOR LLM
# =============================================================================

def get_ontology_for_prompt() -> str:
    """Generate a prompt-friendly description of the ontology for the LLM."""
    lines = [
        "=== PRESSURE TRANSMITTER SPECIFICATION ONTOLOGY ===",
        "Extract these specific attributes when found in the text.",
        "Use EXACTLY these field names and formats:\n",
    ]
    
    for key, spec in PRESSURE_TRANSMITTER_ONTOLOGY.items():
        type_hint = spec.spec_type.value
        unit_hint = f" (normalize to {spec.canonical_unit})" if spec.canonical_unit else ""
        
        if spec.allowed_values:
            values_hint = f" - Valid values: {', '.join(spec.allowed_values[:5])}"
            if len(spec.allowed_values) > 5:
                values_hint += "..."
        else:
            values_hint = ""
        
        importance_stars = "★" * spec.importance
        
        # Include aliases for better extraction
        alias_hint = ""
        if spec.aliases:
            alias_hint = f" (Also called: {', '.join(spec.aliases[:3])})"
        
        lines.append(
            f"- {key}: [{type_hint}]{unit_hint}{values_hint}{alias_hint} {importance_stars}"
        )
    
    lines.append("\n=== EXTRACTION RULES ===")
    lines.append("1. For RANGE types: Use format 'min-max unit' (e.g., '0-6000 psi')")
    lines.append("2. For VALUE types: Include unit if applicable (e.g., '±0.075%')")
    lines.append("3. For LIST types: Return array of values")
    lines.append("4. For ENUM types: Use ONLY the allowed values listed")
    lines.append("5. Only extract what you SEE in the text - no guessing")
    lines.append("6. Include BOTH raw_value (original text) and normalized_value when units differ")
    lines.append("7. If you find specs NOT in this ontology, include them under 'other_specifications'")
    
    return "\n".join(lines)


def get_aggressive_extraction_prompt() -> str:
    """Get an aggressive extraction prompt that captures ALL possible specs."""
    return """=== AGGRESSIVE SPECIFICATION EXTRACTION ===

Your goal is to extract EVERY possible technical specification from the text.

STEP 1: Extract ALL specifications you can find, even if they don't match the ontology exactly.
Look for:
- Any number followed by a unit (e.g., "0.5% accuracy", "10-30 VDC", "IP67")
- Any specification pattern: "Name: Value" or "Name - Value"
- Tables with specification data
- Bullet points with technical details
- Parenthetical specifications
- Footnotes with specs

STEP 2: For each spec, provide:
{
    "extracted_name": "what it was called in the source",
    "value": "the value you found",
    "unit": "the unit if applicable",
    "raw_text": "exact quote from source (max 100 chars)",
    "confidence": 0.0-1.0
}

STEP 3: Group specs into:
- "ontology_specs": specs that match the known ontology
- "other_specifications": specs that don't match but are still valuable

REMEMBER: Extract EVERYTHING. It's better to capture too much than too little.
"""


# =============================================================================
# COMPARISON UTILITIES
# =============================================================================

def compare_specs(spec1: Dict[str, Any], spec2: Dict[str, Any]) -> Dict[str, Dict]:
    """
    Compare two products' specifications for head-to-head analysis.
    
    Returns a dict with winner for each spec:
    {
        "pressure_range": {"winner": "product1", "reason": "Wider range"},
        "accuracy": {"winner": "product2", "reason": "Better accuracy (0.05% vs 0.1%)"},
        ...
    }
    """
    comparison = {}
    
    for spec_name, spec_def in PRESSURE_TRANSMITTER_ONTOLOGY.items():
        val1 = spec1.get(spec_name)
        val2 = spec2.get(spec_name)
        
        if val1 is None and val2 is None:
            continue
        elif val1 is None:
            comparison[spec_name] = {"winner": "product2", "reason": "Only product2 has this spec"}
        elif val2 is None:
            comparison[spec_name] = {"winner": "product1", "reason": "Only product1 has this spec"}
        else:
            # Compare based on type
            if spec_def.spec_type == SpecType.RANGE:
                # For ranges, wider is usually better
                range1 = val1 if isinstance(val1, tuple) else (0, 0)
                range2 = val2 if isinstance(val2, tuple) else (0, 0)
                span1 = range1[1] - range1[0]
                span2 = range2[1] - range2[0]
                
                if span1 > span2:
                    comparison[spec_name] = {"winner": "product1", "reason": f"Wider range: {span1} vs {span2}"}
                elif span2 > span1:
                    comparison[spec_name] = {"winner": "product2", "reason": f"Wider range: {span2} vs {span1}"}
                else:
                    comparison[spec_name] = {"winner": "tie", "reason": "Same range"}
            
            elif spec_def.spec_type == SpecType.VALUE and spec_name in ["accuracy", "repeatability"]:
                # For accuracy, LOWER is better
                try:
                    v1 = float(val1) if not isinstance(val1, (int, float)) else val1
                    v2 = float(val2) if not isinstance(val2, (int, float)) else val2
                    if v1 < v2:
                        comparison[spec_name] = {"winner": "product1", "reason": f"Better: {v1}% vs {v2}%"}
                    elif v2 < v1:
                        comparison[spec_name] = {"winner": "product2", "reason": f"Better: {v2}% vs {v1}%"}
                    else:
                        comparison[spec_name] = {"winner": "tie", "reason": "Same value"}
                except (ValueError, TypeError):
                    comparison[spec_name] = {"winner": "tie", "reason": "Cannot compare values"}

    return comparison


# Export for easy access
__all__ = [
    "PRESSURE_TRANSMITTER_ONTOLOGY",
    "SpecDefinition", 
    "SpecType",
    "NormalizedValue",
    "AIExtractedSpec",
    "get_ontology_for_prompt",
    "get_aggressive_extraction_prompt",
    "normalize_pressure",
    "normalize_temperature",
    "normalize_length",
    "normalize_time",
    "normalize_spec_value",
    "compare_specs",
    "parse_range",
    "parse_percentage",
    "extract_number_with_unit",
    "find_best_ontology_match",
    "match_partial_phrase",
    "calculate_similarity",
    "register_ai_derived_attribute",
    "get_ai_derived_attributes",
    "get_frequently_seen_ai_attributes",
    "AI_DERIVED_ATTRIBUTES",
    "PRESSURE_TO_PSI",
    "PSI_TO_OTHER",
]
