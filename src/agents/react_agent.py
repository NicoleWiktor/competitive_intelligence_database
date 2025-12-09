"""
ReAct Agent - The core agentic intelligence for competitive analysis.

This agent implements the ReAct (Reasoning + Acting) pattern:
1. OBSERVE: What do I know? What's in the current state?
2. THINK: What am I trying to achieve? What's missing?
3. ACT: Choose and execute a tool to gather information
4. REFLECT: Did that help? What should I do next?

The agent makes AUTONOMOUS DECISIONS about:
- What competitors to research
- What products to investigate
- What specifications are important
- When to verify information
- When to stop (goal achieved)

This is TRUE AGENTIC AI - not just a scripted pipeline!
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from src.config.settings import get_openai_api_key
from src.ontology.specifications import (
    PRESSURE_TRANSMITTER_ONTOLOGY,
    get_ontology_for_prompt,
)


# =============================================================================
# AGENT STATE
# =============================================================================

class AgentPhase(Enum):
    """Current phase of the agent's mission."""
    DISCOVERY = "discovery"           # Finding competitors
    PRODUCT_RESEARCH = "product"      # Finding products for competitors
    SPEC_EXTRACTION = "specs"         # Getting technical specifications
    PRICE_RESEARCH = "pricing"        # Finding prices
    VERIFICATION = "verification"     # Verifying collected data
    COMPLETE = "complete"             # Mission accomplished


@dataclass
class AgentState:
    """
    The agent's working memory and accumulated knowledge.
    """
    # Mission parameters
    target_product: str = "SmartLine ST700"
    target_company: str = "Honeywell"
    target_industry: str = "process industries"
    max_competitors: int = 5
    
    # Accumulated knowledge
    competitors: Dict[str, Dict] = field(default_factory=dict)
    products: Dict[str, Dict] = field(default_factory=dict)
    specifications: Dict[str, Dict] = field(default_factory=dict)
    prices: Dict[str, str] = field(default_factory=dict)
    
    # Source tracking
    sources: List[Dict] = field(default_factory=list)
    
    # Agent reasoning
    current_phase: AgentPhase = AgentPhase.DISCOVERY
    thoughts: List[str] = field(default_factory=list)
    actions_taken: List[str] = field(default_factory=list)
    iteration: int = 0
    max_iterations: int = 30
    
    def to_context_string(self) -> str:
        """Convert state to a context string for the LLM."""
        return f"""=== CURRENT KNOWLEDGE STATE ===
Phase: {self.current_phase.value}
Iteration: {self.iteration}/{self.max_iterations}

COMPETITORS FOUND ({len(self.competitors)}/{self.max_competitors}):
{json.dumps(list(self.competitors.keys()), indent=2) if self.competitors else "None yet"}

PRODUCTS FOUND ({len(self.products)}):
{json.dumps(list(self.products.keys()), indent=2) if self.products else "None yet"}

SPECIFICATIONS EXTRACTED:
{json.dumps({k: list(v.keys()) for k, v in self.specifications.items()}, indent=2) if self.specifications else "None yet"}

PRICES FOUND:
{json.dumps(self.prices, indent=2) if self.prices else "None yet"}

SOURCES USED: {len(self.sources)}
"""

    def get_completeness_score(self) -> Dict[str, float]:
        """Calculate how complete our data collection is."""
        scores = {
            "competitors": min(len(self.competitors) / self.max_competitors * 100, 100),
            "products": min(len(self.products) / max(len(self.competitors), 1) * 100, 100),
            "specifications": sum(1 for p in self.products if p in self.specifications) / max(len(self.products), 1) * 100,
            "prices": len(self.prices) / max(len(self.products), 1) * 100,
        }
        scores["overall"] = sum(scores.values()) / 4
        return scores


# =============================================================================
# TOOL DEFINITIONS FOR FUNCTION CALLING
# =============================================================================

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for competitive intelligence. Use to find competitors, products, prices, or specifications.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query. Be specific - include company names, product models, etc."
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Number of results to return (1-5)",
                        "default": 3
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "extract_page_content",
            "description": "Get full content from a URL for detailed analysis",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to extract content from"
                    }
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_competitor",
            "description": "Save a confirmed competitor to the knowledge base",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Company name (e.g., 'Emerson', 'Siemens', 'ABB')"
                    },
                    "source_url": {
                        "type": "string",
                        "description": "URL where this competitor was found"
                    }
                },
                "required": ["name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_product",
            "description": "Save a product to the knowledge base",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_name": {
                        "type": "string",
                        "description": "Product model name (e.g., 'Rosemount 3051', 'A-10')"
                    },
                    "company_name": {
                        "type": "string",
                        "description": "Company that makes this product"
                    },
                    "source_url": {
                        "type": "string",
                        "description": "URL where this product was found"
                    }
                },
                "required": ["product_name", "company_name"]
            }
        }
    },
    {
        "type": "function", 
        "function": {
            "name": "save_specification",
            "description": "Save a product specification to the knowledge base",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_name": {
                        "type": "string",
                        "description": "Product this spec belongs to"
                    },
                    "spec_name": {
                        "type": "string",
                        "description": "Specification type (pressure_range, accuracy, output_signal, etc.)"
                    },
                    "spec_value": {
                        "type": "string",
                        "description": "The specification value (e.g., '0-6000 psi', '±0.075%')"
                    }
                },
                "required": ["product_name", "spec_name", "spec_value"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_price",
            "description": "Save a product price to the knowledge base",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_name": {
                        "type": "string",
                        "description": "Product this price is for"
                    },
                    "price": {
                        "type": "string",
                        "description": "Price with currency (e.g., '$161.09')"
                    }
                },
                "required": ["product_name", "price"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "complete_mission",
            "description": "Mark the mission as complete when enough data has been gathered",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Summary of what was collected"
                    }
                },
                "required": ["summary"]
            }
        }
    }
]


# =============================================================================
# REACT AGENT
# =============================================================================

class CompetitiveIntelligenceAgent:
    """
    Agentic AI for competitive intelligence gathering.
    """
    
    def __init__(self, state: Optional[AgentState] = None):
        self.state = state or AgentState()
        self.llm = ChatOpenAI(
            api_key=get_openai_api_key(),
            model="gpt-4o",
            temperature=0.1,
        )
        self.messages: List[Dict] = []
        self._init_conversation()
    
    def _init_conversation(self):
        """Initialize the conversation with system prompt."""
        system_prompt = f"""You are an expert Competitive Intelligence Agent for Honeywell.

YOUR MISSION:
Gather comprehensive competitive intelligence on pressure transmitters for process industries.
Focus on {self.state.target_company} {self.state.target_product} and its competitors.

TARGET COMPETITORS (find at least 5):
- Emerson (Rosemount brand)
- Siemens  
- ABB
- Endress+Hauser
- Yokogawa
- WIKA
- Danfoss

IMPORTANT SPECIFICATIONS TO GATHER for each product:
- pressure_range (e.g., "0-6000 psi")
- accuracy (e.g., "±0.075%")
- output_signal (e.g., "4-20mA", "HART")
- process_connection (e.g., "1/2 NPT")
- wetted_materials (e.g., "316 SS")
- ip_rating (e.g., "IP67")
- operating_temp (e.g., "-40 to 85°C")
- hazardous_area (e.g., "ATEX", "IECEx")

STRATEGY:
1. First, search for competitors in the pressure transmitter market
2. For each competitor found, save it with save_competitor
3. Search for specific product models for each competitor
4. Save products with save_product
5. Search for specifications/datasheets for each product
6. Save each specification with save_specification
7. Optionally find and save prices
8. When you have enough data (5 competitors, products with specs), call complete_mission

IMPORTANT: 
- Always use the save_* functions to store data you find
- Be thorough - extract multiple specifications per product
- Use specific search queries (include model numbers)
"""
        self.messages = [{"role": "system", "content": system_prompt}]
    
    def _execute_tool(self, tool_name: str, args: Dict) -> str:
        """Execute a tool and return the result."""
        from tavily import TavilyClient
        from src.config.settings import get_tavily_api_key
        
        if tool_name == "search_web":
            query = args.get("query", "")
            max_results = args.get("max_results", 3)
            
            try:
                client = TavilyClient(api_key=get_tavily_api_key())
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
                        "content": r.get("content", "")[:800],
                    })
                    self.state.sources.append({"url": r.get("url", ""), "query": query})
                
                return json.dumps(results, indent=2)
            except Exception as e:
                return f"Search error: {str(e)}"
        
        elif tool_name == "extract_page_content":
            url = args.get("url", "")
            try:
                client = TavilyClient(api_key=get_tavily_api_key())
                response = client.extract(urls=[url], extract_depth="advanced")
                
                for result in response.get("results", []):
                    if result.get("url") == url:
                        content = result.get("raw_content", "")
                        if len(content) > 5000:
                            content = content[:5000] + "\n\n[Truncated...]"
                        return content
                
                return "No content found"
            except Exception as e:
                return f"Extract error: {str(e)}"
        
        elif tool_name == "save_competitor":
            name = args.get("name", "")
            source_url = args.get("source_url", "")
            if name:
                self.state.competitors[name] = {"name": name, "source_url": source_url}
                return f"✓ Saved competitor: {name}"
            return "Error: No name provided"
        
        elif tool_name == "save_product":
            product_name = args.get("product_name", "")
            company_name = args.get("company_name", "")
            source_url = args.get("source_url", "")
            if product_name and company_name:
                self.state.products[product_name] = {
                    "name": product_name, 
                    "company": company_name,
                    "source_url": source_url
                }
                return f"✓ Saved product: {product_name} by {company_name}"
            return "Error: Missing product_name or company_name"
        
        elif tool_name == "save_specification":
            product_name = args.get("product_name", "")
            spec_name = args.get("spec_name", "")
            spec_value = args.get("spec_value", "")
            if product_name and spec_name and spec_value:
                if product_name not in self.state.specifications:
                    self.state.specifications[product_name] = {}
                self.state.specifications[product_name][spec_name] = spec_value
                return f"✓ Saved spec for {product_name}: {spec_name} = {spec_value}"
            return "Error: Missing required fields"
        
        elif tool_name == "save_price":
            product_name = args.get("product_name", "")
            price = args.get("price", "")
            if product_name and price:
                self.state.prices[product_name] = price
                return f"✓ Saved price for {product_name}: {price}"
            return "Error: Missing product_name or price"
        
        elif tool_name == "complete_mission":
            summary = args.get("summary", "Mission complete")
            self.state.current_phase = AgentPhase.COMPLETE
            return f"✓ Mission complete: {summary}"
        
        return f"Unknown tool: {tool_name}"
    
    def step(self) -> Dict[str, Any]:
        """Execute one step of the agent loop."""
        self.state.iteration += 1
        
        if self.state.iteration > self.state.max_iterations:
            self.state.current_phase = AgentPhase.COMPLETE
            return {"complete": True, "reason": "Max iterations reached"}
        
        # Build context message
        context = self.state.to_context_string()
        scores = self.state.get_completeness_score()
        
        user_message = f"""{context}

COMPLETENESS: {scores['overall']:.0f}%
- Competitors: {scores['competitors']:.0f}%
- Products: {scores['products']:.0f}%  
- Specifications: {scores['specifications']:.0f}%
- Prices: {scores['prices']:.0f}%

What should you do next? Choose a tool to call."""

        self.messages.append({"role": "user", "content": user_message})
        
        # Call LLM with function calling
        try:
            response = self.llm.invoke(
                self.messages,
                tools=TOOL_DEFINITIONS,
                tool_choice="required"  # Force tool use
            )
        except Exception as e:
            print(f"[Agent] LLM error: {e}")
            return {"complete": False, "error": str(e)}
        
        # Check for tool calls
        if hasattr(response, 'tool_calls') and response.tool_calls:
            tool_call = response.tool_calls[0]
            tool_name = tool_call.get("name", "")
            tool_args = tool_call.get("args", {})
            
            print(f"\n[Agent Step {self.state.iteration}]")
            print(f"🔧 Tool: {tool_name}")
            print(f"📝 Args: {json.dumps(tool_args, indent=2)[:200]}")
            
            # Execute tool
            result = self._execute_tool(tool_name, tool_args)
            print(f"📋 Result: {result[:300]}...")
            
            # Add to conversation
            self.messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": tool_name, "arguments": json.dumps(tool_args)}}]
            })
            self.messages.append({
                "role": "tool",
                "tool_call_id": "call_1", 
                "content": result
            })
            
            self.state.actions_taken.append(f"{tool_name}({json.dumps(tool_args)[:50]})")
            
            return {
                "complete": self.state.current_phase == AgentPhase.COMPLETE,
                "tool": tool_name,
                "args": tool_args,
                "result": result[:500]
            }
        
        # No tool call - extract from content
        content = getattr(response, "content", "")
        print(f"[Agent] Response without tool call: {content[:200]}")
        
        return {"complete": False, "message": content}
    
    def run(self, max_steps: Optional[int] = None) -> "AgentState":
        """Run the agent until completion."""
        if max_steps:
            self.state.max_iterations = max_steps
        
        print("="*60)
        print("🤖 COMPETITIVE INTELLIGENCE AGENT STARTING")
        print(f"Target: {self.state.target_company} {self.state.target_product}")
        print(f"Max iterations: {self.state.max_iterations}")
        print("="*60)
        
        while self.state.current_phase != AgentPhase.COMPLETE:
            step_result = self.step()
            
            if step_result.get("complete"):
                break
            
            if step_result.get("error"):
                print(f"[Agent] Error: {step_result['error']}")
                break
        
        print("\n" + "="*60)
        print("🏁 AGENT MISSION COMPLETE")
        print(f"Iterations: {self.state.iteration}")
        print(f"Competitors: {len(self.state.competitors)}")
        print(f"Products: {len(self.state.products)}")
        print(f"Specifications: {sum(len(s) for s in self.state.specifications.values())}")
        print(f"Prices: {len(self.state.prices)}")
        print("="*60)
        
        return self.state
    
    def get_knowledge_for_neo4j(self) -> Dict[str, Any]:
        """Convert collected knowledge to Neo4j-ready format."""
        relationships = []
        
        # Competitor relationships
        for comp_name, comp_data in self.state.competitors.items():
            relationships.append({
                "source": "Honeywell",
                "source_type": "Company",
                "relationship": "COMPETES_WITH",
                "target": comp_name,
                "target_type": "Company",
                "source_url": comp_data.get("source_url", ""),
            })
        
        # Product relationships
        for prod_name, prod_data in self.state.products.items():
            company = prod_data.get("company", "")
            if company:
                relationships.append({
                    "source": company,
                    "source_type": "Company",
                    "relationship": "OFFERS_PRODUCT",
                    "target": prod_name,
                    "target_type": "Product",
                    "source_url": prod_data.get("source_url", ""),
                })
        
        # Specification relationships
        for prod_name, specs in self.state.specifications.items():
            for spec_name, spec_value in specs.items():
                relationships.append({
                    "source": prod_name,
                    "source_type": "Product",
                    "relationship": "HAS_SPEC",
                    "target": f"{spec_name}: {spec_value}",
                    "target_type": "Specification",
                    "spec_name": spec_name,
                    "spec_value": str(spec_value),
                    "source_url": "",
                })
        
        # Price relationships
        for prod_name, price in self.state.prices.items():
            relationships.append({
                "source": prod_name,
                "source_type": "Product",
                "relationship": "HAS_PRICE",
                "target": price,
                "target_type": "Price",
                "source_url": "",
            })
        
        return {
            "Relationships": relationships,
            "Doc": {"source_url": [s.get("url", "") for s in self.state.sources]},
        }


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def run_intelligence_agent(
    target_product: str = "SmartLine ST700",
    target_company: str = "Honeywell",
    max_competitors: int = 5,
    max_iterations: int = 30
) -> Dict[str, Any]:
    """Run the competitive intelligence agent."""
    state = AgentState(
        target_product=target_product,
        target_company=target_company,
        max_competitors=max_competitors,
        max_iterations=max_iterations,
    )
    
    agent = CompetitiveIntelligenceAgent(state)
    agent.run()
    
    return agent.get_knowledge_for_neo4j()


if __name__ == "__main__":
    data = run_intelligence_agent(max_iterations=15)
    print("\n" + "="*60)
    print("EXTRACTED DATA:")
    print("="*60)
    print(json.dumps(data, indent=2))
