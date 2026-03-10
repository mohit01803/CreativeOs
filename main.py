# story_mcp_server.py — MCP server for Story Generation API
from typing import Any, Dict

# --- FastMCP import shim (supports both package layouts) ---
try:
    from fastmcp import FastMCP
except ImportError:
    from mcp.server.fastmcp import FastMCP

import httpx
import logging
import sys
import json

# Configure logging (stderr only — no file logging for cloud deployment)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger(__name__)

mcp = FastMCP("story_generator")

# API Base URL
API_BASE = "http://49.248.193.5"

logger.info("=" * 60)
logger.info("Story Generation MCP Server Starting...")
logger.info("=" * 60)

async def _api_post(client: httpx.AsyncClient, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Make POST request to the Story API with extended timeout"""
    url = f"{API_BASE}{endpoint}"
    logger.info(f"Making API request to: {url}")
    logger.info(f"Request data: {json.dumps(data, indent=2)}")

    try:
        # Extended timeout for story generation (5 minutes)
        timeout = httpx.Timeout(
            connect=600.0,  # Connection timeout
            read=600.0,    # Read timeout (5 minutes for story generation)
            write=600.0,    # Write timeout
            pool=600.0      # Pool timeout
        )

        r = await client.post(
            url,
            json=data,
            headers={"Content-Type": "application/json"},
            timeout=timeout
        )
        r.raise_for_status()
        logger.info(f"API request successful - Status: {r.status_code}")

        # Try to parse as JSON, fallback to raw text
        try:
            return r.json()
        except json.JSONDecodeError:
            raw_text = r.text
            logger.warning(f"Response is not valid JSON. Raw response (first 500 chars): {raw_text[:500]}")
            return {
                "success": True,
                "raw_response": raw_text
            }
    except httpx.TimeoutException as e:
        logger.error(f"Timeout error: {e}")
        return {
            "error": "Request timed out. The API is taking longer than expected. Please try again.",
            "error_type": "timeout",
            "success": False
        }
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
        return {
            "error": f"API returned error: {e.response.status_code}",
            "error_details": e.response.text,
            "success": False
        }
    except Exception as e:
        logger.error(f"Request error: {e}")
        return {
            "error": f"Request failed: {str(e)}",
            "error_type": type(e).__name__,
            "success": False
        }

@mcp.tool()
async def generate_story(prompt: str) -> str:
    """
    Generate a story with episodic beats based on your prompt.

    Args:
      prompt: Your story prompt describing what kind of story you want to create.
              Example: "Craft a noir-inspired Bollywood murder mystery micro-series following 
              the shocking death of superstar Zoya..."

    Returns:
      Generated story with episodic beats in JSON format

    Note: Story generation may take 1-3 minutes depending on complexity.
    """
    logger.info(f"generate_story called with prompt length: {len(prompt)} chars")
    logger.info(f"Story generation started - this may take a few minutes...")

    try:
        async with httpx.AsyncClient() as client:
            # API expects 'question' parameter
            data = await _api_post(client, "/story", {"question": prompt})

            # Check if there was an error in the response
            if isinstance(data, dict) and data.get("success") == False:
                logger.error(f"Story generation failed: {data.get('error')}")
                return json.dumps(data, ensure_ascii=False, indent=2)

            logger.info(f"Story generated successfully")
            return json.dumps(data, ensure_ascii=False, indent=2)

    except Exception as e:
        logger.error(f"Error in generate_story: {e}", exc_info=True)
        return json.dumps({
            "error": f"Unexpected error: {str(e)}",
            "error_type": type(e).__name__,
            "success": False
        }, ensure_ascii=False)

@mcp.tool()
async def generate_episodic_beats_from_file(file_path: str) -> str:
    """
    Generate episodic beats from a story file.

    Args:
      file_path: Path to the story file (e.g., "./rag_output.txt")
                 The file should contain a story that you want to break down into episodes.

    Returns:
      Episodic beats structure in JSON format

    Note: Processing may take 1-3 minutes depending on file size.
    """
    logger.info(f"generate_episodic_beats_from_file called with file_path: {file_path}")
    logger.info(f"Episodic beat generation started - this may take a few minutes...")

    try:
        async with httpx.AsyncClient() as client:
            data = await _api_post(client, "/episodic_beats_from_file", {"file_path": file_path})

            # Check if there was an error in the response
            if isinstance(data, dict) and data.get("success") == False:
                logger.error(f"Episodic beat generation failed: {data.get('error')}")
                return json.dumps(data, ensure_ascii=False, indent=2)

            logger.info(f"Episodic beats generated successfully from file")
            return json.dumps(data, ensure_ascii=False, indent=2)

    except Exception as e:
        logger.error(f"Error in generate_episodic_beats_from_file: {e}", exc_info=True)
        return json.dumps({
            "error": f"Unexpected error: {str(e)}",
            "error_type": type(e).__name__,
            "success": False
        }, ensure_ascii=False)

@mcp.tool()
async def ask_vector_db(prompt: str) -> str:
    """
    Ask a question to the Vector Database (RAG - Retrieval Augmented Generation).
    This queries previous stories and knowledge stored in the vector database.

    Args:
      prompt: Your question about previous stories or request for content based on stored knowledge.
              Example: "Write a romantic story based on previous stories"

    Returns:
      Answer from the vector database with relevant story content

    Note: Query processing may take 30-60 seconds.
    """
    logger.info(f"ask_vector_db called with prompt length: {len(prompt)} chars")
    logger.info(f"Vector DB query started - this may take up to a minute...")

    try:
        async with httpx.AsyncClient() as client:
            # API expects 'question' parameter
            data = await _api_post(client, "/ask", {"question": prompt})

            # Check if there was an error in the response
            if isinstance(data, dict) and data.get("success") == False:
                logger.error(f"Vector DB query failed: {data.get('error')}")
                return json.dumps(data, ensure_ascii=False, indent=2)

            logger.info(f"Vector DB query successful")
            return json.dumps(data, ensure_ascii=False, indent=2)

    except Exception as e:
        logger.error(f"Error in ask_vector_db: {e}", exc_info=True)
        return json.dumps({
            "error": f"Unexpected error: {str(e)}",
            "error_type": type(e).__name__,
            "success": False
        }, ensure_ascii=False)

# ----------------------------
# MAIN — Streamable HTTP on 127.0.0.1:8001/mcp
# ----------------------------
if __name__ == "__main__":
    logger.info("Starting Story Generation MCP Streamable HTTP server at http://127.0.0.1:8001/mcp")
    logger.info("Available tools:")
    logger.info("  1. generate_story - Generate a story with episodic beats")
    logger.info("  2. generate_episodic_beats_from_file - Generate episodes from a story file")
    logger.info("  3. ask_vector_db - Query the vector database about previous stories")
    logger.info(f"API Base URL: {API_BASE}")

    # HTTP endpoint at `/mcp` that supports POST (streaming responses)
    mcp.run(
        "http",
        host="127.0.0.1",
        port=8001,  # Using 8001 to avoid conflict with YouTube server on 8000
        path="/mcp"
    )
