"""Agent Reach tools for accessing various content channels."""

import json
import subprocess
from typing import Any

from loguru import logger

from nanobot.agent.tools.base import Tool


class YouTubeTool(Tool):
    """Extract YouTube video information and subtitles."""

    name = "youtube_extract"
    description = "Extract YouTube video info, subtitles, and metadata. Supports video URL or ID."
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "YouTube video URL or ID"},
            "language": {"type": "string", "default": "zh-CN", "description": "Subtitle language code (e.g., zh-CN, en)"}
        },
        "required": ["url"]
    }

    async def execute(self, url: str, language: str = "zh-CN", **kwargs: Any) -> str:
        """Extract YouTube video content."""
        try:
            # Use yt-dlp to extract info
            cmd = [
                "yt-dlp",
                "--dump-json",
                "--no-download",
                "--sub-langs", language,
                "--write-subs",
                url
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode != 0 and not result.stdout:
                return f"Error: {result.stderr}"

            # Parse first line (JSON info)
            lines = result.stdout.strip().split('\n')
            if not lines:
                return "Error: No data returned"

            info = json.loads(lines[0])

            output = [
                f"# {info.get('title', 'Unknown')}",
                f"\n**Channel:** {info.get('uploader', 'Unknown')}",
                f"**Duration:** {info.get('duration_string', 'Unknown')}",
                f"**Views:** {info.get('view_count', 'Unknown')}",
            ]

            if description := info.get('description'):
                output.append(f"\n## Description\n{description[:2000]}")

            # Try to get subtitles
            subs_cmd = [
                "yt-dlp",
                "--skip-download",
                "--write-subs",
                "--sub-langs", language,
                "--sub-format", "txt",
                "-o", "/tmp/yt_subs",
                url
            ]

            subprocess.run(subs_cmd, capture_output=True, timeout=60)

            # Check if subtitle file was created
            import glob
            sub_files = glob.glob(f"/tmp/yt_subs*.{language}*")
            if sub_files:
                with open(sub_files[0], 'r', encoding='utf-8') as f:
                    subs_content = f.read()[:5000]
                    output.append(f"\n## Subtitles (first 5000 chars)\n```\n{subs_content}\n```")

            return '\n'.join(output)

        except subprocess.TimeoutExpired:
            return "Error: Timeout (60s)"
        except FileNotFoundError:
            return "Error: yt-dlp not installed"
        except Exception as e:
            return f"Error: {e}"


class RSSFeedTool(Tool):
    """Read and parse RSS/Atom feeds."""

    name = "rss_read"
    description = "Read RSS or Atom feed and return recent entries with title, link, and summary."
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "RSS/Atom feed URL"},
            "limit": {"type": "integer", "default": 10, "description": "Number of entries to return"}
        },
        "required": ["url"]
    }

    async def execute(self, url: str, limit: int = 10, **kwargs: Any) -> str:
        """Parse RSS feed."""
        try:
            import feedparser

            feed = feedparser.parse(url)

            if feed.bozo and hasattr(feed, 'bozo_exception'):
                # Try anyway, might still work
                pass

            output = [f"# {feed.feed.get('title', 'RSS Feed')}", ""]

            if description := feed.feed.get('description'):
                output.append(f"{description}\n")

            entries = feed.entries[:min(limit, 20)]

            for i, entry in enumerate(entries, 1):
                title = entry.get('title', 'No title')
                link = entry.get('link', 'No link')
                published = entry.get('published', entry.get('updated', 'Unknown date'))
                summary = entry.get('summary', entry.get('description', ''))[:500]

                output.append(f"## {i}. {title}")
                output.append(f"**Published:** {published}")
                output.append(f"**Link:** {link}")
                if summary:
                    output.append(f"**Summary:** {summary}\n")

            return '\n'.join(output)

        except ImportError:
            return "Error: feedparser not installed. Run: pip install feedparser"
        except Exception as e:
            return f"Error parsing RSS: {e}"


class TwitterTool(Tool):
    """Search and extract Twitter/X content."""

    name = "twitter_search"
    description = "Search Twitter/X for tweets. Requires xreach CLI to be installed."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "count": {"type": "integer", "default": 10, "description": "Number of tweets to return"}
        },
        "required": ["query"]
    }

    async def execute(self, query: str, count: int = 10, **kwargs: Any) -> str:
        """Search Twitter using xreach."""
        try:
            cmd = ["xreach", "search", query, "--limit", str(min(count, 20))]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0:
                return f"Error: {result.stderr}\n\nMake sure xreach is installed."

            output = result.stdout.strip()
            if not output:
                return "No tweets found or authentication required."

            return f"# Twitter Search: {query}\n\n{output}"

        except FileNotFoundError:
            return "Error: xreach not installed. Agent Reach should have installed it."
        except subprocess.TimeoutExpired:
            return "Error: Timeout (30s)"
        except Exception as e:
            return f"Error: {e}"


class WebSearchTool(Tool):
    """Semantic web search using Agent Reach's mcporter/Exa."""

    name = "web_semantic_search"
    description = "Perform semantic web search using Exa (free, no API key needed). Returns relevant web pages with summaries."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "count": {"type": "integer", "default": 10, "description": "Number of results"}
        },
        "required": ["query"]
    }

    async def execute(self, query: str, count: int = 10, **kwargs: Any) -> str:
        """Search using mcporter if available, fallback to web_search tool."""
        try:
            # Try mcporter first
            cmd = ["mcporter", "search", "exa", query, "--limit", str(min(count, 10))]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0 and result.stdout.strip():
                return f"# Semantic Search: {query}\n\n{result.stdout}"

            # Fallback message
            return (
                f"Exa search not available. Use web_search tool instead.\n\n"
                f"Query: {query}"
            )

        except FileNotFoundError:
            return "Error: mcporter not installed. Run: agent-reach install"
        except Exception as e:
            return f"Error: {e}"


class GitHubRepoTool(Tool):
    """Enhanced GitHub repository operations via Agent Reach."""

    name = "github_reach"
    description = "GitHub operations via Agent Reach: get repo info, README, recent commits, issues."
    parameters = {
        "type": "object",
        "properties": {
            "owner": {"type": "string", "description": "Repository owner"},
            "repo": {"type": "string", "description": "Repository name"},
            "action": {"type": "string", "enum": ["info", "readme", "commits", "issues"], "default": "info", "description": "Action to perform"},
            "limit": {"type": "integer", "default": 10, "description": "Limit for commits/issues"}
        },
        "required": ["owner", "repo"]
    }

    async def execute(self, owner: str, repo: str, action: str = "info", limit: int = 10, **kwargs: Any) -> str:
        """GitHub operations via gh CLI."""
        try:
            if action == "info":
                cmd = ["gh", "repo", "view", f"{owner}/{repo}", "--json", "name,description,stars,forks,createdAt,updatedAt,primaryLanguage,url"]
            elif action == "readme":
                cmd = ["gh", "repo", "view", f"{owner}/{repo}", "--readme"]
            elif action == "commits":
                cmd = ["gh", "api", f"repos/{owner}/{repo}/commits?per_page={min(limit, 20)}"]
            elif action == "issues":
                cmd = ["gh", "issue", "list", "--repo", f"{owner}/{repo}", "--limit", str(min(limit, 20)), "--json", "number,title,state,createdAt,author"]
            else:
                return f"Unknown action: {action}"

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0:
                return f"Error: {result.stderr}"

            output = result.stdout.strip()

            if action in ["commits", "issues"]:
                try:
                    data = json.loads(output)
                    return json.dumps(data, ensure_ascii=False, indent=2)
                except json.JSONDecodeError:
                    pass

            return f"# GitHub: {owner}/{repo} ({action})\n\n{output}"

        except FileNotFoundError:
            return "Error: gh CLI not installed. Run: agent-reach install"
        except subprocess.TimeoutExpired:
            return "Error: Timeout (30s)"
        except Exception as e:
            return f"Error: {e}"
