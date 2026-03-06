"""Defuddle tool for extracting clean article content from web pages."""

import json
import subprocess
from typing import Any

from nanobot.agent.tools.base import Tool


class DefuddleTool(Tool):
    """Extract clean article content from URLs using defuddle."""

    name = "defuddle_extract"
    description = "Extract clean article content from a web page, removing ads and clutter. Returns markdown with metadata."
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "URL to extract content from"},
            "format": {"type": "string", "enum": ["markdown", "json"], "default": "markdown", "description": "Output format"}
        },
        "required": ["url"]
    }

    async def execute(self, url: str, format: str = "markdown", **kwargs: Any) -> str:
        """Run defuddle to extract content from URL."""
        try:
            cmd = ["defuddle", "parse", url]
            if format == "json":
                cmd.append("--json")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0:
                return f"Error extracting content: {result.stderr}"

            output = result.stdout.strip()

            if format == "json":
                try:
                    data = json.loads(output)
                    return json.dumps(data, ensure_ascii=False, indent=2)
                except json.JSONDecodeError:
                    return output

            return output

        except subprocess.TimeoutExpired:
            return "Error: Timeout extracting content (30s)"
        except FileNotFoundError:
            return "Error: defuddle not installed. Run: npm install -g defuddle"
        except Exception as e:
            return f"Error: {e}"
