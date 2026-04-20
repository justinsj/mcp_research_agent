"""
MCP Template Client
===================
A generic agentic loop built on the Model Context Protocol (MCP).

Architecture:
    1. A FastMCP *Client* connects to the MCP server (mcp_server.py)
       via stdio and discovers available tools.
    2. A *Chat* instance drives an agentic loop powered by the
       Anthropic Claude API. Each iteration:
       a. Sends the conversation to Claude with available tools.
       b. Claude may call tools (web_search, research_website,
          instruction_following_check, final_answer_check) — all served
          by the MCP server.
       c. Tool results are appended back to the conversation.
    3. The loop walks through the PROCESS steps defined in constants.py
       (e.g. GatheredData → FinalAnswerDict), one step per iteration.
    4. Once `final_answer_check` returns status "ok", the loop ends.

Entry Points:
    - CLI: `python mcp_client.py -q "Your question" --max-loops 5`
    - Programmatic: `Chat().run(question="...", max_loops=5)`

Environment Variables:
    - ANTHROPIC_API_KEY (required)
    - LOG_FORMATTED (optional, pretty-print tool args in logs)
    - FASTMCP_LOG_LEVEL (set automatically to INFO, can be overridden)
"""

import os
import logging
# ─── Local imports (support both direct execution and package import) ───
try:
    from constants import (
        MAX_LOOPS,
        INSTRUCTIONS,
        PROCESS,
        get_typeddict_structure,
        LOG_LEVEL,
    )
    from pricing import TokenTracker
except (ModuleNotFoundError, ImportError):
    from .constants import (
        MAX_LOOPS,
        INSTRUCTIONS,
        PROCESS,
        get_typeddict_structure,
        LOG_LEVEL,
    )
    from .pricing import TokenTracker

os.environ["FASTMCP_LOG_LEVEL"] = logging.getLevelName(LOG_LEVEL)

import json
import asyncio
from typing import Dict, Union, cast, Any, List
import logging
import time

import anthropic
from anthropic.types import MessageParam, ToolUnionParam
from fastmcp import Client
from fastmcp.client.sampling import (
    SamplingMessage,
    SamplingParams,
    RequestContext,
)
import dotenv
import argparse
import pandas as pd

dotenv.load_dotenv()


# ─── Configuration ───
anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
log_formatted = os.getenv("LOG_FORMATTED", False)

# ─── Logging ───
root = logging.getLogger()
if root.handlers:
    for handler in root.handlers:
        root.removeHandler(handler)
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# MCP Server Configuration
# ─────────────────────────────────────────────────────────────────────

# Tells FastMCP Client how to spawn the server process.
config = {
    "mcpServers": {
        "tools": {
            "command": "python",
            "args": [
                os.path.abspath(
                    os.path.join(
                        os.path.dirname(__file__), "mcp_server.py",
                    )
                )
            ],
        }
    }
}

# ─────────────────────────────────────────────────────────────────────
# CLI Argument Parsing
# ─────────────────────────────────────────────────────────────────────

args = None
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MCP Template Client — agentic research loop."
    )
    parser.add_argument(
        "--question", "-q", type=str, default="",
        help="The question to research.",
    )
    parser.add_argument(
        "--max-loops", type=int, default=MAX_LOOPS,
        help=f"Maximum number of agentic loops (default: {MAX_LOOPS}).",
    )
    args = parser.parse_args()


# ─────────────────────────────────────────────────────────────────────
# Chat Agent
# ─────────────────────────────────────────────────────────────────────


class Chat:
    """
    Orchestrates the agentic research loop.

    Flow:
        1. `run()` spawns the MCP server and calls `initialize()`.
        2. `chat_loop()` iterates through PROCESS steps, calling
           `process_loop()` each iteration.
        3. `process_loop()` sends the conversation to Claude, handles
           tool calls via MCP server.
        4. Once `final_answer_check` passes, `_build_result()` assembles
           the final output dict with token/cost metadata.
    """

    # Default model for all API calls
    MODEL_ID: str = "claude-haiku-4-5-20251001"

    def __init__(self, model_id: str = MODEL_ID) -> None:
        """
        Initialize a new Chat instance with fresh state.

        Args:
            model_id: Anthropic model identifier to use for all LLM calls.
        """
        self.model_id = model_id

        # Conversation state (reset per run via initialize())
        self.messages: List[MessageParam] = []
        self.available_tools: List[ToolUnionParam] = []
        self.tool_durations: Dict[str, float] = {}
        self.tool_calls_ordered: List[str] = []
        self.system_prompt: str = ""
        self.loop_num: int = 0
        self.total_duration_for_llm_calls: float = 0.0

        # Token & cost tracking (encapsulated in TokenTracker)
        self.tracker = TokenTracker()

    # ─── Initialization ───

    async def initialize(self, client: Client) -> None:
        """
        Discover MCP server tools and build the system prompt.

        Called once at the start of each `run()`. Resets conversation
        state so the same Chat instance can be reused.

        Args:
            client: An open FastMCP Client connected to the server.
        """
        # Discover server tools (including web_search)
        response = await client.list_tools()
        self.available_tools = [
            {
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": tool.inputSchema,
            }
            for tool in response
        ]

        # Reset per-run state
        self.messages = []
        self.tool_calls_ordered = []
        self.tool_durations = {}
        self.loop_num = 0
        self.total_duration_for_llm_calls = 0.0
        self.tracker = TokenTracker()

        # Build the system prompt with process steps
        process_steps_desc = ""
        for i, td_class in enumerate(PROCESS):
            process_steps_desc += (
                f"Step {i + 1}: "
                f"{get_typeddict_structure(td_class)}\n"
            )

        self.system_prompt = (
            "You are a helpful assistant who can call tools to "
            "gather information to answer the question. You will "
            "be given multiple loops to answer a question.\n\n"
            f"The current date is {pd.Timestamp.now()}.\n\n"
            "You must follow this process IN ORDER, completing "
            "each step before moving to the next:\n"
            f"{process_steps_desc}\n"
            "Use available tools (web_search, "
            "research_website, etc.) to gather information "
            "for each step.\n"
            "Each loop you will be told which step to focus on. "
            "Fill out the structure for that step before moving "
            "on.\n\n"
            "IMPORTANT: After completing each step, you MUST call "
            "the validate_data tool with a JSON string of your "
            "findings and the typed_dict_name for that step. "
            "This validates your output matches the expected "
            "schema. Do this for EVERY step including "
            "intermediate ones.\n\n"
            f"Instructions:\n{INSTRUCTIONS}\n\n"
            "Ensure you call the tool instruction_following_check "
            "to verify the answer follows the guide before calling "
            "final_answer_check.\n"
            "You will need to call the tool final_answer_check. "
            "Once it is called successfully, that will be the "
            "final answer, so you can only call it once.\n"
            "When calling the final_answer_check, you must ensure "
            "that the JSON structure matches the one above exactly."
        )

    # ─── Sampling Handler ───

    def get_sampling_handler(self):
        """
        Build and return an async sampling handler for MCP server callbacks.

        The MCP server may request LLM completions via the sampling protocol.
        This handler converts the request to Anthropic API format, calls Claude,
        tracks token usage under the *sampling* bucket, and returns the text.

        Returns:
            An async callable compatible with FastMCP's sampling_handler interface.
        """
        async def sampling_handler(
            messages: List[SamplingMessage],
            params: SamplingParams,
            context: RequestContext,
        ) -> str:
            logger.info(
                f"[sampling_handler] Sampling call "
                f"#{self.tracker.sampling_call_count + 1} started."
            )

            anthropic_client = anthropic.AsyncAnthropic(
                api_key=anthropic_api_key,
            )

            # Convert SamplingMessage to Anthropic message format
            anthropic_messages = []
            for msg in messages:
                content = msg.content
                if hasattr(content, "text"):
                    anthropic_messages.append(
                        {"role": msg.role, "content": content.text},
                    )
                elif isinstance(content, str):
                    anthropic_messages.append(
                        {"role": msg.role, "content": content},
                    )
                elif isinstance(content, list):
                    anthropic_messages.append(
                        {"role": msg.role, "content": content},
                    )
                else:
                    anthropic_messages.append(
                        {"role": msg.role, "content": str(content)},
                    )

            # Call Anthropic API with prompt caching
            llm_response = await anthropic_client.messages.create(
                model=self.model_id,
                system=[
                    {
                        "type": "text",
                        "text": params.systemPrompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                max_tokens=(
                    params.maxTokens
                    if hasattr(params, "maxTokens") and params.maxTokens
                    else 8000
                ),
                messages=anthropic_messages,
            )

            # Track tokens under the sampling bucket
            self.tracker.accumulate_sampling_usage(llm_response)

            logger.info(
                f"[sampling_handler] Sampling call "
                f"#{self.tracker.sampling_call_count} complete."
            )

            # Extract text from response
            if hasattr(llm_response, "content") and llm_response.content:
                for block in llm_response.content:
                    if getattr(block, "type", None) == "text":
                        return block.text

            return json.dumps({"error": "Empty response from LLM"})

        return sampling_handler

    # ─── Single Loop Iteration ───

    async def process_loop(
        self,
        client: Client,
        loop_num: int = 0,
        total_loops: int = MAX_LOOPS,
        max_tool_rounds: int = 10,
    ) -> None:
        """
        Execute one iteration of the agentic loop.

        Sends the current conversation to Claude, processes the response
        (text blocks and tool_use blocks), executes tool calls, and appends
        results back to self.messages. Continues calling Claude after tool
        results until Claude stops issuing tool calls (end_turn).

        Args:
            client: Open FastMCP Client for calling server tools.
            loop_num: Current loop index (0-based), for logging.
            total_loops: Total number of loops allowed.
            max_tool_rounds: Max inner rounds to prevent runaway tool loops.
        """
        anthropic_client = anthropic.AsyncAnthropic(
            api_key=anthropic_api_key,
        )
        system_prompt = self.system_prompt
        available_tools = self.available_tools

        for tool_round in range(max_tool_rounds):
            logger.info(
                "[process_loop] Starting Claude API call."
            )

            # Claude API call
            api_start = time.perf_counter()
            res = await anthropic_client.messages.create(
                model=self.model_id,
                system=system_prompt,
                max_tokens=8000,
                messages=self.messages,
                tools=available_tools,
            )
            api_duration = time.perf_counter() - api_start
            self.total_duration_for_llm_calls += api_duration
            logger.info(
                f"[process_loop] Claude API call duration: "
                f"{api_duration:.2f} seconds"
            )

            # Track token usage
            self.tracker.accumulate_usage(res)

            has_tool_use = any(
                c.type == "tool_use" for c in res.content
            )

            for content in res.content:
                if content.type == "text":
                    self.messages.append(
                        {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": content.text}
                            ],
                        }
                    )

                elif content.type == "tool_use":
                    tool_name = content.name
                    tool_args = content.input

                    tool_args_str = tool_args
                    if log_formatted:
                        tool_args_str = json.dumps(tool_args, indent=2)
                        escaped_newline = "\\n"
                        tool_args_str = tool_args_str.replace(
                            escaped_newline, chr(10),
                        )

                    logger.info(
                        f"[process_loop] Calling tool {tool_name} "
                        f"with args {tool_args_str}"
                    )
                    self.tool_calls_ordered.append(tool_name)
                    result = None
                    try:
                        start_time = time.perf_counter()

                        result = await client.call_tool(
                            tool_name, cast(dict, tool_args),
                        )

                        end_time = time.perf_counter()
                        duration = end_time - start_time
                        if tool_name not in self.tool_durations:
                            self.tool_durations[tool_name] = 0
                        self.tool_durations[tool_name] += duration
                        logger.info(
                            f"[process_loop] Tool {tool_name} duration: "
                            f"{duration:.2f}s"
                        )

                        if not result:
                            result = []
                    except Exception as e:
                        logger.error(
                            f"[process_loop] Error calling tool "
                            f"{tool_name}: {e}"
                        )
                        from types import SimpleNamespace

                        result = [
                            SimpleNamespace(
                                type="text",
                                text=(
                                    f"Error calling tool "
                                    f"{tool_name}: {str(e)}"
                                ),
                            )
                        ]
                        self.messages.append(
                            {
                                "role": "assistant",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": (
                                            f"Error calling tool "
                                            f"{tool_name}: {str(e)}"
                                        ),
                                    }
                                ],
                            }
                        )

                    self.messages.append(
                        {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "tool_use",
                                    "id": content.id,
                                    "name": tool_name,
                                    "input": tool_args,
                                }
                            ],
                        }
                    )

                    # Fix utf-8 encoding issues
                    if hasattr(result, "content"):
                        all_contents = result.content
                    elif isinstance(result, list):
                        all_contents = result
                    else:
                        all_contents = [result]

                    all_contents = [
                        c
                        for c in all_contents
                        if c is not None
                        and hasattr(c, "type")
                        and hasattr(c, "text")
                    ]

                    self.messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": content.id,
                                    "content": [
                                        {
                                            "type": result_content.type,
                                            "text": result_content.text,
                                        }
                                        for result_content in all_contents
                                    ],
                                }
                            ],
                        }
                    )

            # If Claude didn't use any tools, it's done for this loop
            if not has_tool_use:
                break

            logger.info(
                f"[process_loop] Tool round {tool_round + 1} complete, "
                f"re-calling Claude to process tool results."
            )

    # ─── Final Answer Extraction ───

    def check_for_final_answer(
        self, messages: list[MessageParam],
    ) -> Union[Dict, None]:
        """
        Scan the message history for a successful `final_answer_check` call.

        Looks for an assistant message with a tool_use block calling
        `final_answer_check`, followed by a user message with the tool_result.
        If the result has status "ok", returns the parsed JSON dict.

        Args:
            messages: The full conversation history.

        Returns:
            The validated final answer dict, or None if not found.
        """
        for i, message in enumerate(messages):
            if not (
                message["role"] == "assistant"
                and isinstance(message["content"], list)
            ):
                continue
            for content in message["content"]:
                if not (
                    isinstance(content, dict)
                    and content["type"] == "tool_use"
                    and content.get("name") == "final_answer_check"
                ):
                    continue
                tool_use_id = content.get("id")
                next_message = (
                    messages[i + 1]
                    if i + 1 < len(messages)
                    else None
                )
                assert (
                    next_message["role"] == "user"
                ), (
                    f"Expected user message after assistant, "
                    f"got {next_message['role'] if next_message else 'None'}"
                )
                tool_result = [
                    c
                    for c in next_message["content"]
                    if isinstance(c, dict)
                    and c.get("type") == "tool_result"
                    and c.get("tool_use_id") == tool_use_id
                ]
                assert len(tool_result) <= 1, (
                    f"Expected at most one tool_result for "
                    f"tool_use_id {tool_use_id}, "
                    f"got {len(tool_result)}"
                )
                assert len(tool_result) == 1, (
                    f"No tool_result found for "
                    f"tool_use_id {tool_use_id}"
                )
                tool_result = tool_result[0]
                tool_result_json_str = (
                    tool_result
                    .get("content", [{}])[0]
                    .get("text", "{}")
                )
                assert tool_result_json_str, (
                    f"No content found in tool_result for "
                    f"tool_use_id {tool_use_id}"
                )
                logger.info(
                    f"[check_for_final_answer] Found: "
                    f"{tool_result_json_str}"
                )
                tool_result_json = json.loads(tool_result_json_str)
                assert tool_result_json.get("status") == "ok", (
                    f"Tool result status is not ok: "
                    f"{tool_result_json.get('status')}"
                )
                return tool_result_json

        return None

    # ─── Main Chat Loop ───

    async def chat_loop(
        self,
        session: Client,
        question: str = None,
        max_loops: int = MAX_LOOPS,
    ) -> Dict[str, Any]:
        """
        Drive the multi-step agentic loop until a final answer is found.

        Iterates up to `max_loops` times. Each iteration:
          1. Constructs a user message indicating which PROCESS step to focus on.
          2. Calls `process_loop()` which talks to Claude and executes tools.
          3. Checks if `final_answer_check` was called successfully.

        Args:
            session: Open FastMCP Client for tool calls.
            question: The user's research question.
            max_loops: Maximum iterations before giving up.

        Returns:
            The validated final answer dict from `final_answer_check`,
            or None if max loops exhausted.
        """
        question = question or input(
            "Enter your question: "
        ).strip()
        logger.info(f"User question: {question}")

        process_names = [p.__name__ for p in PROCESS]

        for loop_num in range(max_loops):
            logger.info(
                f"[chat_loop] Starting loop "
                f"{loop_num + 1}/{max_loops}"
            )

            # Determine which PROCESS step to focus on
            current_step_index = min(
                loop_num, len(PROCESS) - 1,
            )
            current_step = PROCESS[current_step_index]
            step_structure = get_typeddict_structure(current_step)

            if loop_num == 0:
                loop_message = (
                    f"{question}\n\n"
                    f"Follow the enrichment process in order "
                    f"({' -> '.join(process_names)}).\n\n"
                    f"You are on Step {current_step_index + 1} of "
                    f"{len(PROCESS)}: {current_step.__name__}\n"
                    f"Use available tools (web_search, "
                    f"research_website, etc.) to gather "
                    f"information to fill out this structure:\n"
                    f"{step_structure}\n\n"
                    f"After gathering the data, call validate_data "
                    f"with your JSON findings and "
                    f"typed_dict_name=\"{current_step.__name__}\"."
                )
            elif loop_num + 1 == max_loops:
                loop_message = (
                    f"This is the last loop "
                    f"({loop_num + 1}/{max_loops}). "
                    f"Please finalize and call final_answer_check "
                    f"with your best answer now."
                )
            else:
                completed_steps = process_names[:current_step_index]
                loop_message = (
                    f"Continue the process "
                    f"(loop {loop_num + 1}/{max_loops}).\n"
                    f"Completed steps: "
                    f"{', '.join(completed_steps) if completed_steps else 'None'}\n\n"
                    f"Current Step {current_step_index + 1} of "
                    f"{len(PROCESS)}: {current_step.__name__}\n"
                    f"Use available tools to gather information "
                    f"to fill out:\n{step_structure}\n\n"
                    f"After gathering the data, call validate_data "
                    f"with your JSON findings and "
                    f"typed_dict_name=\"{current_step.__name__}\".\n\n"
                    f"Once this step is complete, move on to the "
                    f"next step."
                )

            self.messages.append(
                {"role": "user", "content": loop_message},
            )
            logger.info(
                f"[chat_loop] asking: {loop_message}..."
            )
            self.loop_num = loop_num
            msg_count_before = len(self.messages)
            await self.process_loop(
                session,
                loop_num=loop_num,
                total_loops=max_loops,
            )

            # Log any validate_data results from this loop
            for msg in self.messages[msg_count_before:]:
                if msg["role"] != "user":
                    continue
                content = msg["content"]
                if not isinstance(content, list):
                    continue
                for block in content:
                    if (
                        isinstance(block, dict)
                        and block.get("type") == "tool_result"
                    ):
                        # Find the matching tool_use to check name
                        tool_use_id = block.get("tool_use_id")
                        for prev_msg in self.messages[msg_count_before:]:
                            if prev_msg["role"] != "assistant":
                                continue
                            prev_content = prev_msg["content"]
                            if not isinstance(prev_content, list):
                                continue
                            for prev_block in prev_content:
                                if (
                                    isinstance(prev_block, dict)
                                    and prev_block.get("type") == "tool_use"
                                    and prev_block.get("id") == tool_use_id
                                    and prev_block.get("name") == "validate_data"
                                ):
                                    result_texts = [
                                        c.get("text", "")
                                        for c in block.get("content", [])
                                        if isinstance(c, dict)
                                    ]
                                    if result_texts:
                                        logger.info(
                                            f"[chat_loop] Step "
                                            f"{current_step_index + 1}/"
                                            f"{len(PROCESS)} "
                                            f"({current_step.__name__}) "
                                            f"validation:\n"
                                            f"{chr(10).join(result_texts)}"
                                        )

            # Check for a valid response
            valid_response = self.check_for_final_answer(
                self.messages,
            )
            if valid_response:
                return valid_response
            logger.info(
                "[chat_loop] No valid final answer, continuing."
            )

        logger.error(
            "[chat_loop] Max loops reached without final answer."
        )

    # ─── Result Assembly ───

    def _build_result(
        self,
        result_json: Dict[str, Any],
        start_time: float,
        max_loops: int,
    ) -> Dict[str, Any]:
        """
        Assemble the final output dict with metadata, token usage, and costs.

        Logs a summary of tool durations, token counts, and USD costs,
        then attaches everything to `result_json` under the `log_data`
        and `messages` keys.

        Args:
            result_json: The raw final answer dict from `final_answer_check`.
            start_time: `time.perf_counter()` value from when `run()` started.
            max_loops: The max_loops setting used for this run.

        Returns:
            The enriched result_json dict.
        """
        end_time = time.perf_counter()

        # Log tool durations
        for tool, duration in self.tool_durations.items():
            logger.info(f"  {tool}: {duration:.2f}s")
        logger.info(
            f"[Chat.run] Tools used in order: {self.tool_calls_ordered}"
        )
        result_json["tools_used"] = self.tool_calls_ordered

        # Log high-level stats
        logger.info(
            f"[Chat.run] Total sampling calls: "
            f"{self.tracker.sampling_call_count}"
        )
        logger.info(
            f"[Chat.run] Total LLM call duration: "
            f"{self.total_duration_for_llm_calls:.2f}s"
        )

        # Log token usage
        t = self.tracker
        logger.info("[Chat.run] Token Usage:")
        logger.info(f"  Total Input: {t.total_input_tokens:,}")
        logger.info(f"  Total Output: {t.total_output_tokens:,}")
        logger.info(
            f"  Total: {t.total_input_tokens + t.total_output_tokens:,}"
        )
        logger.info(f"  Cache Creation: {t.total_cache_creation_tokens:,}")
        logger.info(f"  Cache Read: {t.total_cache_read_tokens:,}")

        # Build token/cost summary via TokenTracker
        summary = t.build_summary(self.model_id)

        logger.info(f"[Chat.run] Cost ({self.model_id}):")
        logger.info(f"  Total: ${summary['cost_usd']['total_cost']:.6f}")
        logger.info(f"  Sampling: ${summary['cost_usd']['sampling_cost']:.6f}")
        logger.info(f"  Main API: ${summary['cost_usd']['main_api_cost']:.6f}")

        # Attach metadata
        result_json["log_data"] = {
            "tool_durations": self.tool_durations,
            "tools_used": self.tool_calls_ordered,
            "total_duration_for_llm_calls": self.total_duration_for_llm_calls,
            "sampling_call_count": t.sampling_call_count,
            "total_loops": self.loop_num + 1,
            "max_loops": max_loops,
            "total_duration": end_time - start_time,
            "model_id": self.model_id,
            **summary,
        }
        result_json["messages"] = self.messages

        return result_json

    # ─── Public Entry Point ───

    async def run(
        self,
        question: str = "",
        max_loops: int = MAX_LOOPS,
    ) -> Dict[str, Any]:
        """
        Execute a complete agentic research run.

        Spawns the MCP server, initializes tools, runs the chat loop,
        and returns a result dict containing the final answer plus
        detailed metadata (tokens, costs, tool durations, messages).

        Args:
            question: The research question to answer.
            max_loops: Maximum number of agentic loop iterations.

        Returns:
            Dict with keys: status, message (the answer), tools_used,
            log_data (token_usage, cost_usd, durations), messages.

        Raises:
            ValueError: If no final answer was produced within max_loops.
        """
        sampling_handler = self.get_sampling_handler()
        client = Client(config, sampling_handler=sampling_handler)

        start = time.perf_counter()
        logger.info(
            f"[Chat.run] Starting run with question: {question}, "
            f"max_loops: {max_loops}"
        )

        async with client:
            await self.initialize(client)
            logger.info("[Chat.run] Client initialized and tools acquired.")

            result_json = await self.chat_loop(
                client, question, max_loops=max_loops,
            )
            logger.info("[Chat.run] Chat loop complete.")

            # Log full message history
            messages_json_str = json.dumps(self.messages, indent=2)
            logger.info(
                f"[Chat.run] Messages:\n"
                f"{messages_json_str.replace(chr(92) + 'n', chr(10))}"
            )
            logger.info(f"[Chat.run] Final result: {result_json}")

            if not result_json:
                raise ValueError(
                    "No result returned from MCP server."
                )

            return self._build_result(result_json, start, max_loops)


# ─────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    chat = Chat()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(
        chat.run(
            question=args.question,
            max_loops=args.max_loops,
        )
    )
