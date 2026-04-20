# MCP Template — Structured Agentic Research Loop

A generic, multi-step agentic research framework built on the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/). Define your data structures as Python `TypedDict`s, set their order in a `PROCESS` list, and the framework handles the agentic loop, tool orchestration, validation, and cost tracking automatically.

## Purpose

Enable a declarative, schema-driven approach to building complex multi-step research agents. This empowers you to focus on defining the data you want to extract and the instructions for the agent, while the framework manages the iterative prompting, tool calls, and validation logic.

1. Define typed output schemas as `TypedDict` classes in `constants.py`
2. Order them in the `PROCESS` list (e.g. `[DisputeCase, DefendantInfo, PlaintiffInfo, FinalAnswerDict]`)
3. The agentic loop walks through each step, instructing Claude to fill out the structure using web search and website research tools
4. Each step is validated at runtime against its TypedDict schema via the `validate_data` tool
5. A final `instruction_following_check` + `final_answer_check` gate ensures the output matches your schema before returning

The included example researches legal disputes — but the framework is domain-agnostic. Swap the TypedDicts and instructions for any structured research task.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  mcp_client.py (Chat)                               │
│  ┌───────────────────────────────────────────────┐  │
│  │  chat_loop()                                  │  │
│  │  For each PROCESS step:                       │  │
│  │    1. Send step prompt to Claude              │  │
│  │    2. process_loop() — Claude calls tools     │  │
│  │    3. Log validate_data results               │  │
│  │    4. Check for final_answer_check            │  │
│  └───────────────────────────────────────────────┘  │
│                        │ stdio                       │
│  ┌─────────────────────▼─────────────────────────┐  │
│  │  mcp_server.py (FastMCP)                      │  │
│  │  Tools:                                       │  │
│  │    • web_search (Brave Search API)            │  │
│  │    • research_website (Playwright + LLM)      │  │
│  │    • validate_data (TypedDict validation)     │  │
│  │    • instruction_following_check (QC gate)    │  │
│  │    • final_answer_check (schema validation)   │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

## Project Structure

```
mcp_template/
├── constants.py              # TypedDicts, PROCESS order, instructions, config
├── mcp_client.py             # Agentic loop (Chat class)
├── mcp_server.py             # MCP server with tools
├── helpers.py                # Runtime TypedDict validation
├── pricing.py                # Token tracking & cost calculation
├── website_research_agent/
│   └── agent.py              # Playwright-based web scraper + LLM extraction
├── requirements.txt          # Python dependencies
├── tests/
│   └── test.py               # End-to-end test runner
└── README.md
```

## Installation

### Prerequisites

- Python 3.10.11+
- An [Anthropic API key](https://console.anthropic.com/)
- A [Brave Search API key](https://brave.com/search/api/)

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Install Playwright browsers
python -m playwright install chromium
```

### Environment Variables

Create a `.env` file in the `mcp_template/` directory:

```env
ANTHROPIC_API_KEY=sk-ant-...
BRAVE_SEARCH_API_KEY=BSA...
```

Optional variables:

| Variable | Default | Description |
|---|---|---|
| `LOG_FORMATTED` | `False` | Pretty-print tool args in logs |
| `IS_LOCAL` | - | Set to `true` to skip automatic Playwright path setup |

## Usage

### CLI

```bash
# Run with default question and loop count
python mcp_client.py -q "Research the latest lawsuit involving Katie Perry"

# Specify max loops
python mcp_client.py -q "Your question here" --max-loops 6
```

### Test Runner

```bash
cd mcp_template
python -m tests.test
python -m tests.test --question "Your question" --max-loops 5
```

### Programmatic

```python
from mcp_client import Chat

chat = Chat(model_id="claude-haiku-4-5-20251001")
result = await chat.run(
    question="Research the latest lawsuit involving Katie Perry",
    max_loops=6,
)

print(result["message"])       # The validated final answer
print(result["log_data"])      # Token usage, costs, durations
```

## Customisation Guide

All customisation happens in `constants.py`. No changes to the client or server are needed.

### 1. Define your data structures

Replace the example TypedDicts with your domain-specific schemas:

```python
class CompanyProfile(TypedDict):
    name: str
    website: str
    industry: str
    employee_count: int
    annual_revenue: Optional[float]

class CompetitorAnalysis(TypedDict):
    competitors: List[CompanyProfile]
    market_position: str

class FinalAnswerDict(TypedDict):
    company: CompanyProfile
    competitors: List[CompanyProfile]
    summary: str
```

Supported types: `str`, `int`, `float`, `bool`, `List[T]`, `Tuple[T1, T2]`, `Optional[T]`, `Union[T1, T2]`, `Literal["a", "b"]`, nested `TypedDict`, and `Enum`.

### 2. Set the process order

```python
PROCESS = [
    CompanyProfile,
    CompetitorAnalysis,
    FinalAnswerDict,    # Must always be last
]
```

Each step maps to one iteration of the agentic loop. The LLM fills out one TypedDict per loop, using tools to gather information.

### 3. Write domain instructions

```python
INSTRUCTIONS = """
- Focus on publicly traded companies only.
- Revenue figures must be sourced from SEC filings or official reports.
- Do not include subsidiaries as separate competitors.
"""
```

These are injected into the system prompt and used by `instruction_following_check` to verify the final answer.

### 4. Adjust configuration

```python
MAX_LOOPS = 8       # Max iterations before forcing a final answer
LOG_LEVEL = logging.INFO
```

## How It Works

1. **Client spawns server** — `mcp_client.py` launches `mcp_server.py` via stdio using FastMCP
2. **Tool discovery** — The client discovers all available tools from the server
3. **Step-by-step loop** — For each `PROCESS` step:
   - The client sends a prompt telling Claude which TypedDict to fill out
   - Claude calls tools (`web_search`, `research_website`) to gather data
   - Claude calls `validate_data` with its JSON findings to check against the schema
   - The client logs the validation result
4. **Final answer** — On the last step, Claude calls `instruction_following_check` then `final_answer_check`
5. **Result** — The client returns the validated answer with token usage, costs, and timing metadata

## Tools

| Tool | Description |
|---|---|
| `web_search` | Searches the web via Brave Search API. Returns titles, URLs, and descriptions. |
| `research_website` | Loads a URL with Playwright (stealth mode), extracts page content, and uses an LLM to pull structured information. |
| `validate_data` | Validates a JSON string against any PROCESS TypedDict schema. Returns field-level errors or success. |
| `instruction_following_check` | LLM-powered QC gate that verifies the proposed answer follows domain instructions. |
| `final_answer_check` | Validates the final JSON output matches the `FinalAnswerDict` schema exactly. |

## Example Output

Running the included test with a lawsuit research query:

"Is there any latest lawsuit with Katie Perry?"

```
$ python -m tests.test > sample_output.log 2>&1

[chat_loop] Step 1/4 (DisputeCase) validation:
{"status": "ok", "message": {"defendant_name": "Katy Perry", "plaintiff_name": "Katie Perry (Katie Taylor)", ...}}

[chat_loop] Step 2/4 (DefendantInformation) validation:
{"status": "ok", "message": {"entity_name": "Katy Perry", "founding_or_birth_year": 1984, "profession": "Singer, Songwriter, Television Personality"}}

[chat_loop] Step 3/4 (PlaintiffInformation) validation:
{"status": "ok", "message": {"entity_name": "Katie Perry (Katie Taylor)", "founding_or_birth_year": 1980, "profession": "Fashion Designer"}}

Final Answer:
{
  "result": "Yes, there is a latest lawsuit involving Katy Perry. On March 11, 2026,
    the Australian High Court ruled in favor of Katie Perry (now Katie Taylor),
    an Australian fashion designer, in a trademark dispute against pop star Katy Perry.
    The case lasted nearly 17 years...",
  "defendant_name": "Katy Perry",
  "plaintiff_name": "Katie Perry (Katie Taylor)",
  "reference_items": [
    {
      "intended_query": "Katie Perry vs Katy Perry trademark dispute High Court ruling 2026",
      "source_url": "https://www.bbc.com/news/articles/c9dn8l021l2o",
      "relevant_snippet": "Australian designer Katie Perry has won her High Court appeal..."
    },
    ...
  ]
}

Tools used (16):
  web_search, research_website, validate_data,
  instruction_following_check, final_answer_check

Stats:
  Loops used: 4/4
  Total duration: 103.14s
  LLM call duration: 49.89s

Token Usage:
  Input:  158,650
  Output: 4,312
  Total:  162,962

Cost:
  Total:    $0.180210
```
