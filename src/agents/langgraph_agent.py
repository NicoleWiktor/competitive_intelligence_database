"""
LangGraph Agentic Pipeline - Parallel tool execution with proper state management.

Architecture:
    __start__ → agent → tools → agent (loop) → __end__
    
    The 'tools' node fans out to execute multiple tools in parallel,
    with individual tool nodes shown in the visualization.
"""

from __future__ import annotations

import json
import operator
import time
from typing import Annotated, Any, Dict, List, Sequence, TypedDict, Literal
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from tavily import TavilyClient

from src.config.settings import get_openai_api_key, get_tavily_api_key


# =============================================================================
# STATE DEFINITION
# =============================================================================

class AgentState(TypedDict):
    """State that flows through the LangGraph agent."""
    messages: Annotated[List[BaseMessage], operator.add]
    competitors: Dict[str, Dict]
    products: Dict[str, Dict]
    specifications: Dict[str, Dict]  # product -> {spec_name: {value, snippet, evidence_id, source_url}}
    prices: Dict[str, Dict]          # product -> {value, snippet, evidence_id, source_url}
    reviews: Dict[str, List[Dict]]  # product_name -> list of reviews
    sources: List[str]
    iteration: int
    max_iterations: int
    is_complete: bool


# =============================================================================
# TOOLS
# =============================================================================

@tool
def search_web(query: str, max_results: int = 3) -> str:
    """Search the web for competitive intelligence."""
    client = TavilyClient(api_key=get_tavily_api_key())
    try:
        response = client.search(query=query, max_results=min(max_results, 5), search_depth="advanced")
        results = [{"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("content", "")[:500]} 
                   for r in response.get("results", [])]
        return json.dumps(results, indent=2)
    except Exception as e:
        return f"Search error: {str(e)}"


@tool
def extract_page_content(url: str) -> str:
    """Extract content from a URL."""
    client = TavilyClient(api_key=get_tavily_api_key())
    try:
        response = client.extract(urls=[url])
        results = response.get("results", [])
        if results:
            return results[0].get("raw_content", "")[:3000]
        return "No content found"
    except Exception as e:
        return f"Extraction error: {str(e)}"


@tool
def save_competitor(name: str, source_url: str = "") -> str:
    """Save a competitor company."""
    return json.dumps({"action": "save_competitor", "name": name, "source_url": source_url})


@tool
def save_product(product_name: str, company_name: str, source_url: str = "") -> str:
    """Save a product."""
    return json.dumps({"action": "save_product", "product_name": product_name, "company_name": company_name, "source_url": source_url})


@tool
def save_specification(product_name: str, spec_name: str, spec_value: str, snippet: str = "", evidence_id: str = "", source_url: str = "") -> str:
    """Save a specification with optional snippet and evidence (chunk) id."""
    return json.dumps({
        "action": "save_specification",
        "product_name": product_name,
        "spec_name": spec_name,
        "spec_value": spec_value,
        "snippet": snippet,
        "evidence_id": evidence_id,
        "source_url": source_url,
    })


@tool
def save_price(product_name: str, price: str, snippet: str = "", evidence_id: str = "", source_url: str = "") -> str:
    """Save a price with optional snippet and evidence (chunk) id."""
    return json.dumps({
        "action": "save_price",
        "product_name": product_name,
        "price": price,
        "snippet": snippet,
        "evidence_id": evidence_id,
        "source_url": source_url,
    })


@tool
def save_review(product_name: str, review_text: str, rating: str = "", source: str = "", snippet: str = "", evidence_id: str = "", source_url: str = "") -> str:
    """Save a customer review for a product. Include the review text, rating (e.g., '4/5 stars'), and source."""
    return json.dumps({
        "action": "save_review",
        "product_name": product_name,
        "review_text": review_text,
        "rating": rating,
        "source": source,
        "source_url": source_url or source,
        "snippet": snippet or review_text,
        "evidence_id": evidence_id,
    })


@tool
def mark_complete(summary: str) -> str:
    """Mark the mission as complete."""
    return json.dumps({"action": "complete", "summary": summary})


TOOLS = [search_web, extract_page_content, save_competitor, save_product, save_specification, save_price, save_review, mark_complete]

TOOL_MAP = {t.name: t for t in TOOLS}


# =============================================================================
# AGENT NODE
# =============================================================================

def create_agent_node(llm_with_tools):
    """Create the agent node."""
    
    system_prompt = """You are a Competitive Intelligence Agent.

GOAL: Build a competitive analysis of the pressure transmitter market for Honeywell SmartLine ST700.

BASELINE FIRST:
- Always start by saving Honeywell as the company and SmartLine ST700 as the baseline product.
- Extract specifications, price (or "Contact for quote"), and at least one review for SmartLine ST700 before moving to competitors. If it is not able to be found within a few tries, move on.

FOR EACH COMPETITOR YOU MUST:
1. save_competitor - save the company name
2. save_product - save their pressure transmitter product name  
3. save_specification - save specs (pressure_range, accuracy, output_signal, ip_rating) WITH snippet + evidence_id + source_url
4. save_price - save price OR "Contact for quote" (include snippet + evidence_id + source_url)
5. save_review - find and save customer reviews (search "[product name] review") WITH snippet + evidence_id + source_url
6. Repeat for next competitor

REVIEW RULES:
- Search "[product name] review" or "[product name] customer feedback"
- Look for reviews on industrial forums, Amazon, distributor sites
- Save the review text, rating (e.g., "4/5 stars"), and source
- Even one review per product is valuable!

PRICE RULES:
- If no public price found: save_price with "Contact for quote"

SNIPPET & EVIDENCE RULES (for specs, prices, reviews):
- Include a short snippet/quote from the source page (snippet)
- Include evidence_id (chunk id) if available, and source_url
- These will be shown in verification

REQUIREMENTS BEFORE mark_complete:
- At least 5 competitors saved
- At least 5 products saved
- At least 10 specifications saved  
- At least 5 prices saved
- At least 3 reviews saved

WHEN READY EARLY:
- If thresholds are met but iterations remain, consider another search for new competitors/products or missing specs/reviews before marking complete.

WORKFLOW: search -> save_competitor -> save_product -> extract_page -> save_specification -> save_price -> search reviews -> save_review -> repeat"""

    def agent_node(state: AgentState) -> Dict:
        messages = list(state.get("messages", []))
        
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=system_prompt)] + messages
        
        competitors = state.get("competitors", {})
        products = state.get("products", {})
        specs = state.get("specifications", {})
        
        total_specs = sum(len(s) for s in specs.values())
        prices = state.get("prices", {})
        reviews = state.get("reviews", {})
        total_reviews = sum(len(r) for r in reviews.values())
        can_complete = len(competitors) >= 5 and len(products) >= 5 and total_specs >= 10 and len(prices) >= 5 and total_reviews >= 3
        remaining_iters = state.get('max_iterations', 30) - state.get('iteration', 0)
        show_complete_hint = can_complete and remaining_iters <= 5
        
        context = f"""
PROGRESS:
- Competitors: {len(competitors)}/5 - {list(competitors.keys())}
- Products: {len(products)}/5 - {list(products.keys())}  
- Specs: {total_specs}/10 total
- Prices: {len(prices)}/5 - {list(prices.keys())}
- Reviews: {total_reviews}/3 total
- Iteration: {state.get('iteration', 0)}/{state.get('max_iterations', 30)}
- Ready to complete: {'YES' if can_complete else 'NO - need more data!'}

{'You can call mark_complete now.' if show_complete_hint else 'Keep going. If thresholds are met but time remains, search for more competitors/products/specs/reviews before completing.'}"""
        
        messages.append(HumanMessage(content=context))
        
        # Add delay to avoid rate limits
        time.sleep(1.0)
        
        response = llm_with_tools.invoke(messages)
        
        return {
            "messages": [response],
            "iteration": state.get("iteration", 0) + 1,
        }
    
    return agent_node


# =============================================================================
# TOOL NODES - Each tool is its own node for visualization
# =============================================================================

def create_tool_node(tool_name: str):
    """Create a node for a specific tool."""
    tool_fn = TOOL_MAP[tool_name]
    
    def node(state: AgentState) -> Dict:
        messages = state.get("messages", [])
        if not messages:
            return {"messages": [], "competitors": {}, "products": {}, "specifications": {}, "prices": {}, "reviews": {}}
        
        last_msg = messages[-1]
        if not hasattr(last_msg, "tool_calls") or not last_msg.tool_calls:
            return {"messages": [], "competitors": {}, "products": {}, "specifications": {}, "prices": {}, "reviews": {}}
        
        # Find calls for this specific tool
        calls = [tc for tc in last_msg.tool_calls if tc["name"] == tool_name]
        if not calls:
            return {"messages": [], "competitors": {}, "products": {}, "specifications": {}, "prices": {}, "reviews": {}}
        
        results = []
        # Start with empty dicts - we'll only add new items
        new_competitors = {}
        new_products = {}
        new_specs = {}
        new_prices = {}
        new_reviews = {}
        is_complete = state.get("is_complete", False)
        
        for tc in calls:
            try:
                result = tool_fn.invoke(tc["args"])
                results.append(ToolMessage(content=result, tool_call_id=tc["id"]))
                
                # Parse and update state
                try:
                    parsed = json.loads(result)
                    if isinstance(parsed, dict):
                        action = parsed.get("action", "")
                        
                        if action == "save_competitor":
                            name = parsed.get("name", "")
                            if name:
                                new_competitors[name] = {"name": name, "source_url": parsed.get("source_url", "")}
                                print(f"  ✓ Saved competitor: {name}")
                        
                        elif action == "save_product":
                            name = parsed.get("product_name", "")
                            company = parsed.get("company_name", "")
                            if name:
                                new_products[name] = {"name": name, "company": company}
                                print(f"  ✓ Saved product: {name} by {company}")
                        
                        elif action == "save_specification":
                            product = parsed.get("product_name", "")
                            spec_name = parsed.get("spec_name", "")
                            spec_value = parsed.get("spec_value", "")
                            snippet = parsed.get("snippet", "")
                            evidence_id = parsed.get("evidence_id", "")
                            source_url = parsed.get("source_url", "")
                            if product and spec_name:
                                if product not in new_specs:
                                    new_specs[product] = {}
                                new_specs[product][spec_name] = {
                                    "value": spec_value,
                                    "snippet": snippet,
                                    "evidence_id": evidence_id,
                                    "source_url": source_url,
                                }
                                print(f"  ✓ Saved spec: {product} → {spec_name} = {spec_value}")
                        
                        elif action == "save_price":
                            product = parsed.get("product_name", "")
                            price = parsed.get("price", "")
                            snippet = parsed.get("snippet", "")
                            evidence_id = parsed.get("evidence_id", "")
                            source_url = parsed.get("source_url", "")
                            if product and price:
                                new_prices[product] = {
                                    "value": price,
                                    "snippet": snippet,
                                    "evidence_id": evidence_id,
                                    "source_url": source_url,
                                }
                                print(f"  ✓ Saved price: {product} = {price}")
                        
                        elif action == "save_review":
                            product = parsed.get("product_name", "")
                            review_text = parsed.get("review_text", "")
                            rating = parsed.get("rating", "")
                            source = parsed.get("source", "")
                            snippet = parsed.get("snippet", "")
                            evidence_id = parsed.get("evidence_id", "")
                            source_url = parsed.get("source_url", "")
                            if product and review_text:
                                if product not in new_reviews:
                                    new_reviews[product] = []
                                new_reviews[product].append({
                                    "text": review_text,
                                    "rating": rating,
                                    "source": source,
                                    "source_url": source_url,
                                    "snippet": snippet or review_text,
                                    "evidence_id": evidence_id,
                                })
                                print(f"  ✓ Saved review: {product} - {rating} - {review_text[:50]}...")
                        
                        elif action == "complete":
                            is_complete = True
                            print(f"  🏁 Mission complete!")
                except:
                    pass
                    
            except Exception as e:
                results.append(ToolMessage(content=f"Error: {str(e)}", tool_call_id=tc["id"]))
        
        return {
            "messages": results,
            "competitors": new_competitors,
            "products": new_products,
            "specifications": new_specs,
            "prices": new_prices,
            "reviews": new_reviews,
            "is_complete": is_complete,
        }
    
    return node


# =============================================================================
# ROUTER
# =============================================================================

def route_after_agent(state: AgentState) -> str:
    """Route from agent to appropriate tool or end."""
    if state.get("is_complete", False):
        return "end"
    
    if state.get("iteration", 0) >= state.get("max_iterations", 30):
        return "end"
    
    messages = state.get("messages", [])
    if not messages:
        return "end"
    
    last_msg = messages[-1]
    if not hasattr(last_msg, "tool_calls") or not last_msg.tool_calls:
        return "end"
    
    # Route to the first tool called
    tool_name = last_msg.tool_calls[0]["name"]
    return tool_name


# =============================================================================
# BUILD GRAPH
# =============================================================================

def build_agentic_graph(max_iterations: int = 30):
    """Build the LangGraph with separate tool nodes."""
    
    llm = ChatOpenAI(
        api_key=get_openai_api_key(),
        model="gpt-4o",  # Back to gpt-4o for better quality
        temperature=0.1,
    )
    # Disable parallel tool calls for sequential execution
    llm_with_tools = llm.bind_tools(TOOLS, parallel_tool_calls=False)
    
    graph = StateGraph(AgentState)
    
    # Add agent node
    graph.add_node("agent", create_agent_node(llm_with_tools))
    
    # Add individual tool nodes
    graph.add_node("search_web", create_tool_node("search_web"))
    graph.add_node("extract_page", create_tool_node("extract_page_content"))
    graph.add_node("save_competitor", create_tool_node("save_competitor"))
    graph.add_node("save_product", create_tool_node("save_product"))
    graph.add_node("save_spec", create_tool_node("save_specification"))
    graph.add_node("save_price", create_tool_node("save_price"))
    graph.add_node("save_review", create_tool_node("save_review"))
    graph.add_node("mark_complete", create_tool_node("mark_complete"))
    
    # Edges
    graph.add_edge(START, "agent")
    
    # Agent routes to specific tool
    graph.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "search_web": "search_web",
            "extract_page_content": "extract_page",
            "save_competitor": "save_competitor",
            "save_product": "save_product",
            "save_specification": "save_spec",
            "save_price": "save_price",
            "save_review": "save_review",
            "mark_complete": "mark_complete",
            "end": END,
        }
    )
    
    # All tools go back to agent
    graph.add_edge("search_web", "agent")
    graph.add_edge("extract_page", "agent")
    graph.add_edge("save_competitor", "agent")
    graph.add_edge("save_product", "agent")
    graph.add_edge("save_review", "agent")
    graph.add_edge("save_spec", "agent")
    graph.add_edge("save_price", "agent")
    graph.add_edge("mark_complete", "agent")
    
    return graph.compile()


# =============================================================================
# RUN AGENT
# =============================================================================

def run_langgraph_agent(max_iterations: int = 30, max_competitors: int = 5) -> Dict[str, Any]:
    """Run the LangGraph agent."""
    
    print("="*60)
    print("🤖 LANGGRAPH AGENTIC PIPELINE")
    print(f"Max iterations: {max_iterations}")
    print("="*60)
    
    app = build_agentic_graph(max_iterations)
    
    initial_state = {
        "messages": [],
        "competitors": {},
        "products": {},
        "specifications": {},
        "prices": {},
        "reviews": {},
        "sources": [],
        "iteration": 0,
        "max_iterations": max_iterations,
        "is_complete": False,
    }
    
    print("\n[Agent Starting...]\n")
    
    # Use invoke to get final accumulated state
    final_state = None
    step = 0
    
    try:
        # Stream for progress output, but use the final state from LangGraph
        for event in app.stream(initial_state, {"recursion_limit": 200}):
            node_name = list(event.keys())[0]
            node_output = event[node_name]
            
            # Print progress
            if node_name == "agent":
                step += 1
                msgs = node_output.get("messages", [])
                if msgs and hasattr(msgs[-1], "tool_calls") and msgs[-1].tool_calls:
                    print(f"\n[Step {step}] Agent calling:")
                    for tc in msgs[-1].tool_calls:
                        print(f"  🔧 {tc['name']}")
            
            # Keep track of the latest full state from each node
            # LangGraph accumulates state automatically - we just need to capture it
            if node_output:
                if final_state is None:
                    final_state = dict(initial_state)
                
                # The node output contains only what changed - merge it
                for key, value in node_output.items():
                    if key == "messages" and isinstance(value, list):
                        final_state["messages"] = final_state.get("messages", []) + value
                    elif key == "specifications" and isinstance(value, dict):
                        # DEEP MERGE for specifications: {product: {spec_name: value}}
                        if value:
                            existing = final_state.get(key, {})
                            for product_name, specs_dict in value.items():
                                if product_name not in existing:
                                    existing[product_name] = {}
                                for spec_name, spec_payload in specs_dict.items():
                                    if spec_name not in existing[product_name] or isinstance(existing[product_name].get(spec_name), str):
                                        existing[product_name][spec_name] = spec_payload
                                    else:
                                        existing[product_name][spec_name].update(spec_payload)
                            final_state[key] = existing
                    elif key == "reviews" and isinstance(value, dict):
                        # DEEP MERGE for reviews: {product: [list of reviews]}
                        if value:
                            existing = final_state.get(key, {})
                            for product_name, review_list in value.items():
                                if product_name not in existing:
                                    existing[product_name] = []
                                existing[product_name].extend(review_list)
                            final_state[key] = existing
                    elif key in ["competitors", "products", "prices"] and isinstance(value, dict):
                        if value:  # Only update if non-empty
                            existing = final_state.get(key, {})
                            existing.update(value)
                            final_state[key] = existing
                    elif value is not None:
                        final_state[key] = value
                        
    except Exception as e:
        print(f"\n⚠️ Error: {str(e)[:100]}")
    
    if final_state is None:
        final_state = initial_state
    
    # Extract results
    competitors = final_state.get("competitors", {})
    products = final_state.get("products", {})
    specifications = final_state.get("specifications", {})
    prices = final_state.get("prices", {})
    reviews = final_state.get("reviews", {})
    
    print("\n" + "="*60)
    print("🏁 AGENT COMPLETE")
    print(f"Competitors: {len(competitors)} - {list(competitors.keys())}")
    print(f"Products: {len(products)} - {list(products.keys())}")
    print(f"Specifications: {sum(len(s) for s in specifications.values())} total")
    print(f"Prices: {len(prices)}")
    print(f"Reviews: {sum(len(r) for r in reviews.values())} total")
    print("="*60)
    
    return convert_to_neo4j_format(competitors, products, specifications, prices, reviews)


def convert_to_neo4j_format(competitors, products, specifications, prices, reviews) -> Dict[str, Any]:
    """Convert to Neo4j format."""
    relationships = []
    baseline_company = "Honeywell"
    
    for name, data in competitors.items():
        # Skip self-competition
        if name.strip().lower() == baseline_company.lower():
            continue
        relationships.append({
            "source": baseline_company, "source_type": "Company",
            "relationship": "COMPETES_WITH",
            "target": name, "target_type": "Company",
            "source_url": data.get("source_url", ""),
        })
    
    for name, data in products.items():
        company = data.get("company", "")
        if not company:
            lower_name = name.lower()
            if "honeywell" in lower_name or "smartline" in lower_name or lower_name.startswith("st7"):
                company = baseline_company
        if company:
            relationships.append({
                "source": company, "source_type": "Company",
                "relationship": "OFFERS_PRODUCT",
                "target": name, "target_type": "Product",
            })
    
    for product_name, specs in specifications.items():
        for spec_name, spec_payload in specs.items():
            if isinstance(spec_payload, dict):
                spec_value = spec_payload.get("value", "")
                snippet = spec_payload.get("snippet", "")
                evidence_id = spec_payload.get("evidence_id", "")
                source_url = spec_payload.get("source_url", "")
            else:
                spec_value = spec_payload
                snippet = ""
                evidence_id = ""
                source_url = ""
            relationships.append({
                "source": product_name, "source_type": "Product",
                "relationship": "HAS_SPEC",
                "target": f"{spec_name}: {spec_value}",
                "target_type": "Specification",
                "spec_type": spec_name,
                "spec_value": spec_value,
                "snippet": snippet,
                "evidence_ids": [evidence_id] if evidence_id else [],
                "source_url": source_url,
            })
    
    for product_name, price_payload in prices.items():
        if isinstance(price_payload, dict):
            price_value = price_payload.get("value", "")
            snippet = price_payload.get("snippet", "")
            evidence_id = price_payload.get("evidence_id", "")
            source_url = price_payload.get("source_url", "")
        else:
            price_value = price_payload
            snippet = ""
            evidence_id = ""
            source_url = ""
        relationships.append({
            "source": product_name, "source_type": "Product",
            "relationship": "HAS_PRICE",
            "target": f"{product_name} | {price_value}", "target_type": "Price",
            "snippet": snippet,
            "evidence_ids": [evidence_id] if evidence_id else [],
            "source_url": source_url,
        })
    
    # Add reviews
    for product_name, review_list in reviews.items():
        for i, review in enumerate(review_list):
            review_id = f"{product_name}_review_{i+1}"
            review_text = review.get("text", "")[:100]  # Truncate for node name
            rating = review.get("rating", "")
            source = review.get("source", "")
            snippet = review.get("snippet", review_text)
            evidence_id = review.get("evidence_id", "")
            relationships.append({
                "source": product_name, "source_type": "Product",
                "relationship": "HAS_REVIEW",
                "target": f"{rating}: {review_text}..." if rating else review_text[:50] + "...",
                "target_type": "Review",
                "review_text": review.get("text", ""),
                "rating": rating,
                "review_source": source,
                "source_url": source,
                "snippet": snippet,
                "evidence_ids": [evidence_id] if evidence_id else [],
            })
    
    return {
        "relationships": relationships,
        "competitors": competitors,
        "products": products,
        "specifications": specifications,
        "prices": prices,
        "reviews": reviews,
    }
