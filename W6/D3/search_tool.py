"""Examples of search engine tools that can replace google_search in ADK agents.

These tools work with any model (not just Gemini) and can be used with OpenRouter models.
"""

# ============================================================================
# Option 1: DuckDuckGo Search (FREE, No API Key Required)
# ============================================================================
# Install: pip install ddgs

def duckduckgo_search(query: str, max_results: int = 5) -> dict:
    """Search the web using DuckDuckGo and return relevant results.
    
    Args:
        query: The search query string.
        max_results: Maximum number of results to return (default: 5).
    
    Returns:
        Dictionary with status and search results.
    """
    try:
        from ddgs import DDGS
        
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            
            if not results:
                return {
                    "status": "error",
                    "error_message": f"No results found for query: {query}"
                }
            
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "title": result.get("title", ""),
                    "snippet": result.get("body", ""),
                    "url": result.get("href", "")
                })
            
            return {
                "status": "success",
                "results": formatted_results
            }
    except ImportError:
        return {
            "status": "error",
            "error_message": "ddgs package not installed. Run: pip install ddgs"
        }
    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Search failed: {str(e)}"
        }
