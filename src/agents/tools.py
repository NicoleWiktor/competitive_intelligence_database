"""
Agent Tools - The capabilities available to the Competitive Intelligence Agent.

These tools follow the LangChain Tool pattern and can be used by the ReAct agent
to dynamically search, extract, verify, and analyze competitive intelligence.

AGENTIC PRINCIPLE: The AI decides WHEN and HOW to use these tools based on its
reasoning about what data is missing and what would be most valuable to find.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from tavily import TavilyClient

from src.config.settings import get_tavily_api_key, get_openai_api_key
from src.ontology.specifications import (
    PRESSURE_TRANSMITTER_ONTOLOGY,
    get_ontology_for_prompt,
)


# =============================================================================
# TOOL: WEB SEARCH
# =============================================================================

@tool
def search_web(query: str, max_results: int = 3) -> str:
    """
    Search the web for competitive intelligence information.
    
    Use this tool when you need to find:
    - New competitors in the pressure transmitter market
    - Product information for a specific company
    - Pricing data for products
    - Technical specifications for products
    - Industry trends or market analysis
    
    Args:
        query: A specific search query. Be precise and include company/product names.
        max_results: Number of results to return (1-5)
    
    Returns:
        Search results with titles, URLs, and content snippets.
    """
    client = TavilyClient(api_key=get_tavily_api_key())
    
    try:
        response = client.search(
            query=query,
            max_results=min(max_results, 5),
            include_raw_content=True,
            search_depth="advanced"
        )
        
        results = []
        for r in response.get("results", []):
            results.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", "")[:500],  # Truncate for agent context
                "raw_content_length": len(r.get("raw_content", "") or ""),
            })
        
        return json.dumps(results, indent=2)
    
    except Exception as e:
        return f"Search failed: {str(e)}"


@tool
def extract_page_content(url: str) -> str:
    """
    Extract full content from a specific URL for detailed analysis.
    
    Use this tool when you've found a promising URL from search results
    and need the full page content to extract specifications or prices.
    
    Args:
        url: The URL to extract content from.
    
    Returns:
        The full text content of the page (truncated to 10000 chars).
    """
    client = TavilyClient(api_key=get_tavily_api_key())
    
    try:
        response = client.extract(urls=[url], extract_depth="advanced")
        
        for result in response.get("results", []):
            if result.get("url") == url:
                content = result.get("raw_content", "")
                if len(content) > 10000:
                    content = content[:10000] + "\n\n[Content truncated...]"
                return content
        
        return "No content extracted from URL"
    
    except Exception as e:
        return f"Extraction failed: {str(e)}"


# =============================================================================
# TOOL: EXTRACT SPECIFICATIONS
# =============================================================================

@tool
def extract_product_specs(product_name: str, company_name: str, page_content: str) -> str:
    """
    Extract structured specifications from page content using the ontology.
    
    Use this tool after you have page content that contains technical
    specifications for a product. The tool will extract and normalize
    specifications according to the pressure transmitter ontology.
    
    Args:
        product_name: The exact product model name (e.g., "ST800", "A-10")
        company_name: The manufacturer (e.g., "Honeywell", "Wika")
        page_content: The raw text content containing specifications
    
    Returns:
        JSON with extracted and normalized specifications.
    """
    llm = ChatOpenAI(
        api_key=get_openai_api_key(),
        model="gpt-4o-mini",
        temperature=0,
    )
    
    ontology_prompt = get_ontology_for_prompt()
    
    prompt = f"""You are a technical specification extraction expert.

PRODUCT: {product_name} by {company_name}

{ontology_prompt}

=== PAGE CONTENT ===
{page_content[:8000]}

=== YOUR TASK ===
Extract all specifications you can find for {product_name} from the content above.
Return a JSON object with the specification fields as keys.

For each spec found, include:
- The normalized value (convert units as specified)
- The raw text where you found it (max 100 chars)

Example output:
{{
    "pressure_range": {{"value": "0-6000", "unit": "psi", "raw": "Pressure Range: 0 to 6000 psi"}},
    "accuracy": {{"value": 0.075, "unit": "percent_fs", "raw": "Accuracy: ±0.075% FS"}},
    "output_signal": {{"value": ["4-20mA", "HART"], "raw": "Output: 4-20 mA with HART protocol"}},
    "process_connection": {{"value": ["1/2 NPT", "1/4 NPT"], "raw": "Connections: 1/2 or 1/4 NPT"}}
}}

ONLY extract specs you can verify in the text. Skip any spec you're not confident about.
Return valid JSON only, no markdown code blocks."""

    try:
        response = llm.invoke(prompt)
        content = getattr(response, "content", str(response))
        
        # Try to parse as JSON
        try:
            # Clean up response
            content = content.strip()
            if content.startswith("```"):
                content = re.sub(r"```json?\s*", "", content)
                content = content.replace("```", "")
            
            specs = json.loads(content)
            return json.dumps({
                "product": product_name,
                "company": company_name,
                "specifications": specs,
                "extraction_status": "success"
            }, indent=2)
        except json.JSONDecodeError:
            return json.dumps({
                "product": product_name,
                "company": company_name,
                "specifications": {},
                "extraction_status": "failed_to_parse",
                "raw_response": content[:500]
            }, indent=2)
    
    except Exception as e:
        return json.dumps({
            "product": product_name,
            "company": company_name,
            "specifications": {},
            "extraction_status": f"error: {str(e)}"
        }, indent=2)


# =============================================================================
# TOOL: VERIFY INFORMATION
# =============================================================================

@tool
def verify_claim(claim: str, evidence_text: str) -> str:
    """
    Verify if a claim is supported by the evidence text.
    
    Use this tool to validate extracted information before adding it
    to the knowledge base. This prevents hallucinations and ensures
    data quality.
    
    Args:
        claim: The claim to verify (e.g., "Wika A-10 costs $161.09")
        evidence_text: The source text that should support this claim
    
    Returns:
        Verification result with confidence score.
    """
    llm = ChatOpenAI(
        api_key=get_openai_api_key(),
        model="gpt-4o-mini",
        temperature=0,
    )
    
    prompt = f"""You are a fact-checker. Verify if the claim is supported by the evidence.

CLAIM: {claim}

EVIDENCE TEXT:
{evidence_text[:3000]}

Analyze if the evidence DIRECTLY supports the claim.
Return JSON:
{{
    "supported": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation",
    "exact_quote": "the exact text that supports/contradicts" (if found)
}}"""

    try:
        response = llm.invoke(prompt)
        content = getattr(response, "content", str(response))
        
        # Parse response
        try:
            content = content.strip()
            if content.startswith("```"):
                content = re.sub(r"```json?\s*", "", content)
                content = content.replace("```", "")
            return content
        except:
            return json.dumps({"supported": False, "confidence": 0, "reasoning": "Failed to parse"})
    
    except Exception as e:
        return json.dumps({"supported": False, "confidence": 0, "reasoning": str(e)})


# =============================================================================
# TOOL: ANALYZE COMPETITORS
# =============================================================================

@tool  
def analyze_competitive_landscape(current_competitors: str) -> str:
    """
    Analyze the current competitive landscape and suggest next steps.
    
    Use this tool to get strategic guidance on what data to collect next.
    The tool considers what competitors, products, and specs we already
    have and suggests the most valuable next actions.
    
    Args:
        current_competitors: JSON string of current competitor data
    
    Returns:
        Analysis with recommended next actions.
    """
    llm = ChatOpenAI(
        api_key=get_openai_api_key(),
        model="gpt-4o-mini",
        temperature=0.3,
    )
    
    prompt = f"""You are a competitive intelligence analyst for Honeywell pressure transmitters.

CURRENT DATA:
{current_competitors}

Analyze what we have and what's missing. Consider:
1. Do we have enough competitors? (Target: 5 major competitors)
2. Do we have products for each competitor?
3. Do we have prices for comparison?
4. Do we have specifications for head-to-head analysis?

Major pressure transmitter competitors include:
- Emerson (Rosemount)
- Siemens
- ABB
- Endress+Hauser
- Yokogawa
- WIKA
- Danfoss

Return JSON:
{{
    "analysis": "What we have and what's missing",
    "completeness_score": 0-100,
    "priority_actions": [
        {{"action": "description", "target": "specific target", "reason": "why this matters"}}
    ],
    "recommended_search_query": "specific search to run next"
}}"""

    try:
        response = llm.invoke(prompt)
        content = getattr(response, "content", str(response))
        
        try:
            content = content.strip()
            if content.startswith("```"):
                content = re.sub(r"```json?\s*", "", content)
                content = content.replace("```", "")
            return content
        except:
            return json.dumps({"analysis": "Failed to analyze", "priority_actions": []})
    
    except Exception as e:
        return json.dumps({"analysis": str(e), "priority_actions": []})


# =============================================================================
# TOOL: SAVE TO KNOWLEDGE BASE
# =============================================================================

@tool
def save_to_knowledge_base(entity_type: str, entity_data: str, source_url: str) -> str:
    """
    Save verified information to the knowledge base.
    
    Use this tool ONLY after you have:
    1. Found information through search
    2. Extracted structured data
    3. Verified the data is accurate
    
    Args:
        entity_type: One of: "competitor", "product", "price", "specification"
        entity_data: JSON string with the entity details
        source_url: The URL where this information was found
    
    Returns:
        Confirmation of what was saved.
    """
    try:
        data = json.loads(entity_data)
        data["source_url"] = source_url
        data["entity_type"] = entity_type
        
        # This will be processed by the pipeline state
        return json.dumps({
            "status": "queued_for_save",
            "entity_type": entity_type,
            "data": data
        }, indent=2)
    
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": str(e)
        })


# =============================================================================
# ALL TOOLS COLLECTION
# =============================================================================

ALL_TOOLS = [
    search_web,
    extract_page_content,
    extract_product_specs,
    verify_claim,
    analyze_competitive_landscape,
    save_to_knowledge_base,
]

def get_tools_description() -> str:
    """Get a formatted description of all available tools."""
    descriptions = []
    for tool in ALL_TOOLS:
        descriptions.append(f"- {tool.name}: {tool.description.split(chr(10))[0]}")
    return "\n".join(descriptions)

