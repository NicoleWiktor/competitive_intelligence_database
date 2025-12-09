"""
Specification Ontology - Defines the canonical schema for pressure transmitter specifications.

This module handles:
1. WHAT specs matter (the ontology definition)
2. HOW to normalize units (psi ↔ bar, °C ↔ °F, mm ↔ inches)
3. HOW to parse raw spec strings into structured data
4. HOW to enable head-to-head comparisons

Key Design Decision (from boss's question):
"Will that be handled by a human defining what's important or will the AI try to identify those?"

ANSWER: Hybrid approach
- Human defines the ONTOLOGY (what specs exist, valid units, how to normalize)
- AI extracts and maps data ONTO the ontology
- This ensures consistent head-to-head comparison capability
"""

from __future__ import annotations
import re
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


# =============================================================================
# UNIT CONVERSION TABLES
# =============================================================================

# Pressure conversions (all to psi as canonical)
PRESSURE_TO_PSI = {
    "psi": 1.0,
    "bar": 14.5038,
    "mbar": 0.0145038,
    "kpa": 0.145038,
    "mpa": 145.038,
    "pa": 0.000145038,
    "kgcm2": 14.2233,  # kg/cm²
    "atm": 14.6959,
    "mmhg": 0.0193368,
    "inhg": 0.491154,
    "inh2o": 0.0361273,  # inches of water
    "mmh2o": 0.00142233,
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
    "f": fahrenheit_to_celsius,
    "fahrenheit": fahrenheit_to_celsius,
    "°f": fahrenheit_to_celsius,
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
    "ft": 304.8,
    "feet": 304.8,
}

# Time conversions (all to ms as canonical)
TIME_TO_MS = {
    "ms": 1.0,
    "s": 1000.0,
    "sec": 1000.0,
    "second": 1000.0,
    "seconds": 1000.0,
    "min": 60000.0,
    "minute": 60000.0,
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
    importance: int = 1               # 1-5 priority for comparisons
    
    def __post_init__(self):
        if self.extraction_hints is None:
            self.extraction_hints = []


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
            r"(\d+\.?\d*)\s*(?:to|-|~)\s*(\d+\.?\d*)\s*(psi|bar|mbar|kpa|mpa)",
            r"range[:\s]+(\d+\.?\d*)\s*(?:to|-)\s*(\d+\.?\d*)",
            r"pressure\s+range",
            r"measuring\s+range",
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
            "Modbus RTU", "Modbus TCP", "IO-Link",
        ],
        extraction_hints=[
            r"output[:\s]+(4-20\s*ma|0-10\s*v|hart|profibus|modbus)",
            r"(4-20\s*ma|0-20\s*ma)",
            r"(hart|profibus|foundation\s*fieldbus|modbus)",
        ],
        importance=5,
    ),
    
    "supply_voltage": SpecDefinition(
        name="supply_voltage",
        display_name="Supply Voltage",
        spec_type=SpecType.RANGE,
        canonical_unit="V DC",
        extraction_hints=[
            r"supply[:\s]+(\d+\.?\d*)\s*(?:to|-)\s*(\d+\.?\d*)\s*v",
            r"power[:\s]+(\d+\.?\d*)\s*v\s*dc",
            r"(\d+)\s*v\s*dc",
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
        ],
        extraction_hints=[
            r"(1/4|1/2|3/4|1)\s*(npt|bsp)",
            r"(g1/4|g1/2|m20|m14)",
            r"process\s+connection[:\s]+(.+?)(?:\n|,|$)",
            r"(tri.?clamp|flange)",
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
        ],
        extraction_hints=[
            r"wetted[:\s]+(.+?)(?:\n|$)",
            r"material[:\s]+(316|hastelloy|monel|titanium)",
            r"(stainless\s+steel|ss\s*316|hastelloy)",
            r"diaphragm[:\s]+(316|hastelloy|ceramic)",
        ],
        importance=4,
    ),
    
    "housing_material": SpecDefinition(
        name="housing_material",
        display_name="Housing Material",
        spec_type=SpecType.ENUM,
        canonical_unit=None,
        allowed_values=["Aluminum", "316 SS", "Plastic", "Die-cast Aluminum"],
        extraction_hints=[
            r"housing[:\s]+(.+?)(?:\n|$)",
            r"enclosure[:\s]+(aluminum|stainless|plastic)",
        ],
        importance=2,
    ),
    
    # === ENVIRONMENTAL SPECIFICATIONS ===
    "operating_temp": SpecDefinition(
        name="operating_temp",
        display_name="Operating Temperature",
        spec_type=SpecType.RANGE,
        canonical_unit="celsius",
        extraction_hints=[
            r"operating\s+temp[:\s]+(-?\d+)\s*(?:to|~|-)\s*(-?\d+)\s*(°?[cf])",
            r"ambient[:\s]+(-?\d+)\s*(?:to|-)\s*(-?\d+)",
            r"(-40|–40)\s*(?:to|-)\s*(\d+)\s*(°?c)",
        ],
        importance=4,
    ),
    
    "process_temp": SpecDefinition(
        name="process_temp",
        display_name="Process/Media Temperature",
        spec_type=SpecType.RANGE,
        canonical_unit="celsius",
        extraction_hints=[
            r"process\s+temp[:\s]+(-?\d+)\s*(?:to|-)\s*(-?\d+)",
            r"media\s+temp[:\s]+(-?\d+)\s*(?:to|-)\s*(-?\d+)",
            r"fluid\s+temp",
        ],
        importance=4,
    ),
    
    "ip_rating": SpecDefinition(
        name="ip_rating",
        display_name="IP/Ingress Protection Rating",
        spec_type=SpecType.ENUM,
        canonical_unit=None,
        allowed_values=["IP65", "IP66", "IP67", "IP68", "IP69K", "NEMA 4X"],
        extraction_hints=[
            r"(ip\s*6[5-9]k?|ip\s*68|nema\s*4x?)",
            r"protection[:\s]+(ip\d+)",
            r"ingress[:\s]+(ip\d+)",
        ],
        importance=3,
    ),
    
    "hazardous_area": SpecDefinition(
        name="hazardous_area",
        display_name="Hazardous Area Certification",
        spec_type=SpecType.LIST,
        canonical_unit=None,
        allowed_values=[
            "ATEX", "IECEx", "FM", "CSA",
            "Class I Div 1", "Class I Div 2",
            "Zone 0", "Zone 1", "Zone 2",
        ],
        extraction_hints=[
            r"(atex|iecex|fm\s+approved|csa)",
            r"(class\s*i\s*div\s*[12])",
            r"(zone\s*[012])",
            r"explosion.?proof",
            r"intrinsically\s+safe",
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
        ],
        importance=4,
    ),
    
    # === MEASUREMENT TYPE ===
    "measurement_type": SpecDefinition(
        name="measurement_type",
        display_name="Measurement Type",
        spec_type=SpecType.ENUM,
        canonical_unit=None,
        allowed_values=["Gauge", "Absolute", "Differential", "Sealed Gauge"],
        extraction_hints=[
            r"(gauge|absolute|differential|sealed)",
            r"pressure\s+type[:\s]+(gauge|absolute|differential)",
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
        ],
        importance=2,
    ),
}


# =============================================================================
# SPECIFICATION EXTRACTOR: AI-powered extraction onto the ontology
# =============================================================================

@dataclass
class ExtractedSpec:
    """A specification value extracted from text and normalized."""
    spec_name: str              # Ontology key (e.g., "pressure_range")
    raw_value: str              # Original extracted text
    normalized_value: Any       # Normalized for comparison
    unit: Optional[str]         # Canonical unit
    confidence: float           # 0-1 confidence score
    source_text: str            # Text snippet where found
    

def normalize_pressure(value: float, unit: str) -> float:
    """Convert pressure to PSI (canonical unit)."""
    unit_lower = unit.lower().replace(" ", "")
    multiplier = PRESSURE_TO_PSI.get(unit_lower, 1.0)
    return value * multiplier


def normalize_temperature(value: float, unit: str) -> float:
    """Convert temperature to Celsius (canonical unit)."""
    unit_lower = unit.lower().replace("°", "").strip()
    converter = TEMP_CONVERTERS.get(unit_lower, celsius_to_celsius)
    return converter(value)


def normalize_length(value: float, unit: str) -> float:
    """Convert length to mm (canonical unit)."""
    unit_lower = unit.lower().strip()
    multiplier = LENGTH_TO_MM.get(unit_lower, 1.0)
    return value * multiplier


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
        
        lines.append(
            f"- {key}: [{type_hint}]{unit_hint}{values_hint} {importance_stars}"
        )
    
    lines.append("\n=== EXTRACTION RULES ===")
    lines.append("1. For RANGE types: Use format 'min-max unit' (e.g., '0-6000 psi')")
    lines.append("2. For VALUE types: Include unit if applicable (e.g., '±0.075%')")
    lines.append("3. For LIST types: Return array of values")
    lines.append("4. For ENUM types: Use ONLY the allowed values listed")
    lines.append("5. Only extract what you SEE in the text - no guessing")
    
    return "\n".join(lines)


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
                if val1 < val2:
                    comparison[spec_name] = {"winner": "product1", "reason": f"Better: {val1}% vs {val2}%"}
                elif val2 < val1:
                    comparison[spec_name] = {"winner": "product2", "reason": f"Better: {val2}% vs {val1}%"}
                else:
                    comparison[spec_name] = {"winner": "tie", "reason": "Same value"}
    
    return comparison


# Export for easy access
__all__ = [
    "PRESSURE_TRANSMITTER_ONTOLOGY",
    "SpecDefinition", 
    "SpecType",
    "ExtractedSpec",
    "get_ontology_for_prompt",
    "normalize_pressure",
    "normalize_temperature",
    "compare_specs",
    "parse_range",
    "parse_percentage",
]

