"""
Website Research Agent
======================
Loads a web page using Playwright (with stealth anti-bot evasion) and
uses an LLM to extract structured information from the page content.

This module is MCP-independent — it uses standard ``logging`` so it can be
reused as a standalone library or from any other framework.

Entry point:
    ``await research_website(url, additional_instructions)``

Environment Variables:
    - ANTHROPIC_API_KEY (required for LLM extraction)
    - PLAYWRIGHT_BROWSERS_PATH (optional, auto-detected per OS)
    - IS_LOCAL (set to "true" to skip automatic Playwright path setup)
"""

import json
import logging
import os
import platform

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from playwright.async_api import (
    async_playwright,
    TimeoutError as PlaywrightTimeoutError,
)
from playwright_stealth import stealth_async
import anthropic
from fake_useragent import UserAgent

# ─── Local imports ───
try:
    from constants import FINAL_ANSWER_STRUCTURE, INSTRUCTIONS
except (ModuleNotFoundError, ImportError):
    from ..constants import FINAL_ANSWER_STRUCTURE, INSTRUCTIONS

load_dotenv()

logger = logging.getLogger(__name__)

_api_key = os.environ.get("ANTHROPIC_API_KEY")


# ─────────────────────────────────────────────────────────────────────
# Playwright browser path
# ─────────────────────────────────────────────────────────────────────


def get_default_playwright_path() -> str:
    """Determine the default Playwright browser cache directory for the current OS."""
    env_path = os.getenv("PLAYWRIGHT_BROWSERS_PATH")
    if env_path:
        return env_path
    home = os.path.expanduser("~")
    system = platform.system()
    if system == "Darwin":
        return os.path.join(
            home, "Library", "Caches", "ms-playwright",
        )
    elif system == "Linux":
        return os.path.join(home, ".cache", "ms-playwright")
    elif system == "Windows":
        local = os.getenv(
            "LOCALAPPDATA",
            os.path.join(home, "AppData", "Local"),
        )
        return os.path.join(local, "ms-playwright")
    return os.path.abspath("ms-playwright")


if os.getenv("IS_LOCAL") != "true":
    DEFAULT_PLAYWRIGHT_PATH = get_default_playwright_path()
    os.environ.setdefault(
        "PLAYWRIGHT_BROWSERS_PATH", DEFAULT_PLAYWRIGHT_PATH,
    )


# ─────────────────────────────────────────────────────────────────────
# Page loading
# ─────────────────────────────────────────────────────────────────────


async def load_page(url: str) -> str:
    """
    Load a web page using Playwright with stealth configurations.
    Returns the extracted text content from the page.
    """
    browser = None
    context = None
    page = None

    async with async_playwright() as p:
        try:
            launch_args = [
                "--single-process",
                "--no-zygote",
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--disable-gpu-compositing",
                "--in-process-gpu",
                "--use-gl=swiftshader",
                "--disable-accelerated-2d-canvas",
                "--disable-webgl",
                "--disable-infobars",
                "--ignore-certificate-errors",
                "--no-first-run",
                "--window-size=1920,1080",
                "--disable-blink-features=AutomationControlled",
            ]
            browser = await p.chromium.launch(
                headless=True, args=launch_args,
            )
            logger.debug("Browser launched")

            ua = UserAgent()
            context = await browser.new_context(
                user_agent=ua.chrome,
                viewport={"width": 1920, "height": 1080},
                locale="en-US",
                timezone_id="America/New_York",
                permissions=["geolocation"],
                geolocation={
                    "latitude": 40.7128,
                    "longitude": -74.0060,
                },
                color_scheme="light",
                extra_http_headers={
                    "Accept": (
                        "text/html,application/xhtml+xml,"
                        "application/xml;q=0.9,image/avif,"
                        "image/webp,image/apng,*/*;q=0.8"
                    ),
                    "Accept-Language": "en-US,en;q=0.9",
                    "Accept-Encoding": "gzip, deflate, br",
                    "DNT": "1",
                    "Connection": "keep-alive",
                    "Upgrade-Insecure-Requests": "1",
                    "Sec-Fetch-Dest": "document",
                    "Sec-Fetch-Mode": "navigate",
                    "Sec-Fetch-Site": "none",
                    "Sec-Fetch-User": "?1",
                    "Cache-Control": "max-age=0",
                },
            )
            # Block unnecessary resources to speed up loading
            await context.route(
                "**/*",
                lambda route, request: (
                    route.abort()
                    if request.resource_type
                    in ["image", "font", "stylesheet", "script", "xhr", "fetch"]
                    else route.continue_()
                ),
            )

            page = await context.new_page()
            await stealth_async(page)

            # Add additional evasion techniques
            await page.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                });
                window.chrome = { runtime: {} };
                Object.defineProperty(navigator, 'plugins', {
                    get: () => [1, 2, 3, 4, 5]
                });
                Object.defineProperty(navigator, 'languages', {
                    get: () => ['en-US', 'en']
                });
                Object.defineProperty(screen, 'availWidth', {
                    get: () => 1920
                });
                Object.defineProperty(screen, 'availHeight', {
                    get: () => 1080
                });
            """)

            logger.debug(f"Navigating to {url}")
            response = await page.goto(
                url, wait_until="domcontentloaded", timeout=60000,
            )
            logger.debug(
                f"Page loaded with status: "
                f"{response.status if response else 'N/A'}"
            )

            try:
                await page.wait_for_load_state(
                    "networkidle", timeout=15000,
                )
                logger.debug("Network became idle")
            except PlaywrightTimeoutError:
                logger.debug(
                    "Network did not become idle, continuing anyway"
                )

            # Check for Cloudflare challenge
            cloudflare_selectors = [
                'text="Checking your browser"',
                'text="Just a moment"',
                'text="Please wait"',
                "#challenge-running",
                ".ray_id",
                ".attack-box",
            ]
            for selector in cloudflare_selectors:
                try:
                    element = await page.query_selector(selector)
                    if element:
                        logger.info(
                            "Detected Cloudflare challenge, waiting..."
                        )
                        await page.wait_for_selector(
                            selector, state="detached", timeout=30000,
                        )
                        logger.info("Cloudflare challenge completed")
                        break
                except Exception:
                    pass

            await page.wait_for_timeout(3000)
            html_content = await page.content()
            soup = BeautifulSoup(html_content, "html.parser")
            text_content = soup.get_text(separator="\n", strip=True)
            logger.info(
                f"Loaded {len(text_content)} chars from {url}",
            )
            return text_content

        except PlaywrightTimeoutError as e:
            logger.error(f"Timeout loading {url}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading {url}: {e}")
            raise
        finally:
            if page is not None:
                try:
                    await page.close()
                except Exception:
                    pass
            if context is not None:
                try:
                    await context.close()
                except Exception:
                    pass
            if browser is not None:
                try:
                    await browser.close()
                except Exception:
                    pass


# ─────────────────────────────────────────────────────────────────────
# LLM extraction
# ─────────────────────────────────────────────────────────────────────


async def extract_info(
    text: str,
    additional_instructions: str,
    url: str,
    load_page_passed: bool,
) -> object:
    """
    Use an LLM to extract structured information from researched text.

    If *load_page_passed* is True, the LLM is given the raw page text
    and asked to fill out the FINAL_ANSWER_STRUCTURE. If False (page
    failed to load), the LLM is asked to infer information from the URL alone.

    Args:
        text: The researched page text (may be empty if load failed).
        additional_instructions: Extra guidance appended to the system prompt.
        url: The target URL (used as context when page load fails).
        load_page_passed: Whether Playwright successfully loaded the page.

    Returns:
        The raw Anthropic API response object.
    """
    anthropic_client = anthropic.AsyncAnthropic(api_key=_api_key)
    logger.info("Inside extract_info function")

    if load_page_passed:
        system_prompt = (
            "You are a helpful assistant who can look at "
            "text loaded from a website and extract information.\n"
            f"{INSTRUCTIONS}\n\n"
            f"Additional Instructions:\n{additional_instructions}"
        )
        logger.info("[LLM] Sending text to LLM for extraction")
        user_message = text
    else:
        system_prompt = (
            "You are a helpful assistant who can search for "
            "information about a URL/website.\n"
            f"{INSTRUCTIONS}\n\n"
            f"Additional Instructions:\n{additional_instructions}"
        )
        logger.info(
            "[LLM] load_page failed, asking LLM to search using URL"
        )
        user_message = (
            f"What is {url}? Please provide detailed information "
            "about this website. Extract as much information as "
            "possible to fill out the required format."
        )

    llm_response = await anthropic_client.messages.create(
        model="claude-haiku-4-5",
        system=system_prompt,
        max_tokens=8000,
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": user_message}],
            }
        ],
    )

    logger.info(f"[LLM Response] {llm_response}")
    return llm_response

# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────


async def research_website(
    url: str, additional_instructions: str,
) -> str:
    """
    Research *url* and use an LLM to extract structured information.

    Orchestrates the full pipeline:
        1. Load the page via Playwright with stealth evasion.
        2. Pass the loaded text (or URL alone on failure) to an LLM
           for structured extraction.
        3. Return the LLM's text response.

    Args:
        url: The web page to research.
        additional_instructions: Extra guidance for the LLM extraction.

    Returns:
        Extracted information as a string, or an error message.
    """
    try:
        logger.info(f"Fetching URL: {url}")

        load_page_passed = False
        text_content = ""
        try:
            text_content = await load_page(url)
            load_page_passed = True
        except Exception as e:
            logger.warning(
                f"load_page failed for {url}: {e}"
            )
            load_page_passed = False

        llm_response = await extract_info(
            text_content,
            additional_instructions,
            url,
            load_page_passed,
        )

        if hasattr(llm_response, "content") and llm_response.content:
            for block in llm_response.content:
                if getattr(block, "type", None) == "text":
                    return block.text.strip()
            return json.dumps(
                {"error": "No valid JSON content from LLM"}
            )

        return json.dumps(
            {"error": "Empty LLM response"}
        )

    except Exception as e:
        logger.error(f"Error researching {url}: {e}")
        return f"Error researching {url}: {str(e)}"
