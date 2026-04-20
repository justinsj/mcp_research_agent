"""
MCP Template Server
===================
A generic MCP (Model Context Protocol) server exposing tools for:

  - **web_search**: Searches the web using the Brave Search API and returns
    formatted results with titles, URLs, and descriptions.
  - **research_website**: Delegates to the website_research_agent to
    load a web page and extract structured information via LLM.
  - **instruction_following_check**: A quality-control gate that verifies the
    proposed answer adheres to the domain-specific instructions.
  - **final_answer_check**: Validates that a JSON string matches the
    FinalAnswerDict schema.

Customisation:
    Modify FINAL_ANSWER_STRUCTURE, INSTRUCTIONS, and PROCESS in constants.py
    to adapt this server to your domain.

Transport:
    Runs via stdio (spawned by the MCP client). Start directly with:
        python mcp_server.py
"""

import json
import os
from dotenv import load_dotenv
import anthropic
import httpx
import logging

try:
    from constants import (
        FINAL_ANSWER_STRUCTURE,
        FinalAnswerDict,
        INSTRUCTIONS,
        PROCESS_BY_NAME,
        QualityControlCheckResult,
        get_typeddict_structure,
        LOG_LEVEL,
    )
    from helpers import validate_type
    from website_research_agent import research_website as _research_website
except ImportError:
    from .constants import (
        FINAL_ANSWER_STRUCTURE,
        FinalAnswerDict,
        INSTRUCTIONS,
        PROCESS_BY_NAME,
        QualityControlCheckResult,
        get_typeddict_structure,
        LOG_LEVEL,
    )
    from .helpers import validate_type
    from .website_research_agent import research_website as _research_website

_api_key = os.environ.get("ANTHROPIC_API_KEY")

load_dotenv()

os.environ["FASTMCP_RICH_CONSOLE_WIDTH"] = "2000"
os.environ["COLUMNS"] = "2000"

from fastmcp import FastMCP, Context  # noqa: E402

root = logging.getLogger()
if root.handlers:
    for handler in root.handlers:
        root.removeHandler(handler)
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

mcp = FastMCP("MCPTemplate")


# -- Tool: research a website --
@mcp.tool()
async def research_website(
    url: str, additional_instructions: str, ctx: Context,
) -> str:
    f"""
    Research the website at the given URL and extract information.
    Target structure: {FINAL_ANSWER_STRUCTURE}
    Instructions: {INSTRUCTIONS}
    """
    await ctx.info(f"[research_website] Fetching URL: {url}")
    try:
        result = await _research_website(url, additional_instructions)
        await ctx.info(
            f"[research_website] Completed for: {url}"
        )
        return result
    except Exception as e:
        await ctx.error(f"Error researching {url}: {str(e)}")
        return f"Error researching {url}: {str(e)}"


# -- Tool: instruction / QC check --
@mcp.tool()
async def instruction_following_check(
    proposed_final_answer: FinalAnswerDict,
    instructions: str,
    ctx: Context,
) -> QualityControlCheckResult:
    """
    Checks if the proposed answer follows the instructions.
    """
    anthropic_client = anthropic.AsyncAnthropic(api_key=_api_key)
    await ctx.info("[instruction_following_check] Called")
    await ctx.debug(
        f"[instruction_following_check] Input: {proposed_final_answer}"
    )

    try:
        system_prompt = (
            "You are an expert compliance officer.\n"
            "Your task is to review the proposed answer and ensure "
            "it adheres to the specified rules.\n\n"
            f"Rules:\n{instructions}"
        )

        text = (
            f"Proposed Final Answer: {proposed_final_answer}\n\n"
            "Your response must be a valid JSON string with the "
            "following format:\n"
            f"{get_typeddict_structure(QualityControlCheckResult)}"
        )

        llm_response = await anthropic_client.messages.create(
            model="claude-haiku-4-5",
            system=system_prompt,
            max_tokens=8000,
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": text}],
                }
            ],
        )

        await ctx.info(f"[LLM Response] {llm_response}")

        if hasattr(llm_response, "content") and llm_response.content:
            for block in llm_response.content:
                if getattr(block, "type", None) == "text":
                    response_text = block.text.strip()
                    # Strip markdown code fences if present
                    if response_text.startswith("```"):
                        response_text = response_text.split(
                            "\n", 1,
                        )[-1]
                        if response_text.endswith("```"):
                            response_text = response_text[:-3].strip()
                    await ctx.info(
                        f"LLM instruction check response: "
                        f"{response_text}"
                    )
                    try:
                        result_json = json.loads(response_text)
                        validation_response = validate_type(
                            result_json,
                            QualityControlCheckResult,
                        )
                        await ctx.info(
                            f"Validation response: "
                            f"{validation_response}"
                        )
                        return validation_response
                    except Exception as e:
                        await ctx.error(
                            f"Error parsing LLM response: {e}"
                        )
                        return {
                            "is_quality_standard_met": False,
                            "issues": [],
                            "other_notes": (
                                f"Error parsing LLM response: {e}"
                            ),
                        }

    except Exception as e:
        await ctx.error(
            f"Error instruction_following_check: {str(e)}"
        )
        return f"Error instruction_following_check: {str(e)}"


# -- Tool: validate final answer against schema --
@mcp.tool()
async def final_answer_check(
    json_data_str: str, ctx: Context,
) -> str:
    """
    Validates that json_data_str is a valid JSON object
    matching the FinalAnswerDict schema.
    """
    await ctx.info("[final_answer_check] Called")
    await ctx.debug(
        f"[final_answer_check] Input text: {json_data_str}"
    )

    try:
        json_data = json.loads(json_data_str)
        result = validate_type(json_data, FinalAnswerDict)
        return json.dumps(result)
    except Exception as e:
        await ctx.error(
            f"[final_answer_check] Error validating JSON: {e}"
        )
        return f"Error validating JSON: {e}"


# -- Tool: web search via Brave Search API --
@mcp.tool()
async def web_search(query: str, count: int = 5, ctx: Context = None) -> str:
    """
    Search the web using the Brave Search API.

    Returns formatted search results with titles, URLs, and descriptions.
    Requires the BRAVE_SEARCH_API_KEY environment variable.

    Args:
        query: The search query string.
        count: Maximum number of results to return (default 5).
    """
    brave_search_api_key = os.environ.get("BRAVE_SEARCH_API_KEY")
    if not brave_search_api_key:
        return "Error: BRAVE_SEARCH_API_KEY environment variable is not set."

    if ctx:
        await ctx.info(f"[web_search] Searching for: {query}")

    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": brave_search_api_key,
    }
    params = {"q": query, "count": count}

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(
                url, headers=headers, params=params,
            )
            response.raise_for_status()
            data = response.json()

        results = data.get("web", {}).get("results", [])
        if not results:
            return f"No search results found for: {query}"

        formatted = []
        for r in results:
            title = r.get("title", "No title")
            link = r.get("url", "")
            description = r.get("description", "No description")
            formatted.append(
                f"Title: {title}\nURL: {link}\n"
                f"Description: {description}"
            )

        result_text = "\n\n".join(formatted)
        if ctx:
            await ctx.info(
                f"[web_search] Found {len(results)} results for: {query}"
            )
        return result_text
    except httpx.HTTPStatusError as e:
        error_msg = (
            f"Error performing web search: HTTP {e.response.status_code}"
        )
        if ctx:
            await ctx.error(f"[web_search] {error_msg}")
        return error_msg
    except Exception as e:
        error_msg = f"Error performing web search: {str(e)}"
        if ctx:
            await ctx.error(f"[web_search] {error_msg}")
        return error_msg


# -- Tool: validate data against a PROCESS TypedDict schema --
@mcp.tool()
async def validate_data(
    json_data_str: str, typed_dict_name: str, ctx: Context,
) -> str:
    """
    Validates that json_data_str is a valid JSON object matching
    the named TypedDict schema from PROCESS.

    Args:
        json_data_str: A JSON string of the data to validate.
        typed_dict_name: Name of the TypedDict class (e.g. "DisputeCase").
    """
    await ctx.info(
        f"[validate_data] Validating against {typed_dict_name}"
    )

    td_class = PROCESS_BY_NAME.get(typed_dict_name)
    if not td_class:
        return json.dumps({
            "status": "error",
            "message": f"Unknown TypedDict: {typed_dict_name}. "
                       f"Valid names: {list(PROCESS_BY_NAME.keys())}",
        })

    try:
        json_data = json.loads(json_data_str)
        result = validate_type(json_data, td_class)
        return json.dumps(result)
    except json.JSONDecodeError as e:
        return json.dumps({
            "status": "error",
            "message": f"Invalid JSON: {e}",
        })


if __name__ == "__main__":
    print("Starting MCP template server...")
    mcp.run(transport="stdio")
