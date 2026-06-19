"""
Advanced crawler using Playwright for JavaScript-heavy sites
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional
from urllib.parse import urlparse


class CrawlError(Exception):
    """Raised when a URL cannot be fetched or parsed."""


@dataclass
class CrawlResult:
    url: str
    final_url: str
    success: bool
    status_code: int
    content_type: str
    title: str
    description: str
    text: str
    text_length: int
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def validate_url(url: str) -> str:
    """Validate and normalize a URL for crawling."""
    if not url or not url.strip():
        raise CrawlError("URL is empty")

    normalized_url = url.strip()
    parsed = urlparse(normalized_url)

    if parsed.scheme not in ["http", "https"]:
        raise CrawlError("URL must start with http:// or https://")

    if not parsed.netloc:
        raise CrawlError("URL is missing host name")

    return normalized_url


async def fetch_url_playwright(url: str, timeout: int = 30) -> CrawlResult:
    """
    Fetch URL using Playwright (supports JavaScript rendering)
    """
    from playwright.async_api import async_playwright
    
    normalized_url = validate_url(url)
    
    try:
        async with async_playwright() as p:
            # Launch browser
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            )
            page = await context.new_page()
            
            # Navigate - use 'load' instead of 'networkidle' (faster)
            response = await page.goto(normalized_url, wait_until="load", timeout=timeout*1000)
            
            # Wait a bit for dynamic content
            await page.wait_for_timeout(3000)  # 3s
            
            # Extract content
            title = await page.title()
            
            # Try to get description
            description = ""
            desc_meta = await page.query_selector('meta[name="description"]')
            if not desc_meta:
                desc_meta = await page.query_selector('meta[property="og:description"]')
            if desc_meta:
                description = await desc_meta.get_attribute("content") or ""
            
            # Extract article text
            text_parts = []
            
            # Try multiple selectors for article content
            article = (
                await page.query_selector('article') or
                await page.query_selector('[role="article"]') or
                await page.query_selector('.article-content') or
                await page.query_selector('.story-body')
            )
            
            if article:
                paragraphs = await article.query_selector_all('p, h1, h2, h3')
            else:
                paragraphs = await page.query_selector_all('p')
            
            for p in paragraphs:
                try:
                    text = await p.inner_text()
                    text = text.strip()
                    if text and len(text) > 20:
                        text_parts.append(text)
                except:
                    continue
            
            full_text = ' '.join(text_parts)
            final_url = page.url
            
            await browser.close()
            
            return CrawlResult(
                url=normalized_url,
                final_url=final_url,
                success=True,
                status_code=response.status if response else 200,
                content_type="text/html",
                title=title,
                description=description,
                text=full_text,
                text_length=len(full_text)
            )
    
    except Exception as exc:
        raise CrawlError(f"Failed to fetch URL: {exc}") from exc


async def crawl_https_playwright(url: str, timeout: int = 30) -> Dict[str, Any]:
    """
    Public helper using Playwright
    """
    result = await fetch_url_playwright(url=url, timeout=timeout)
    return result.to_dict()


async def crawl_and_clean_playwright(url: str, timeout: int = 30) -> Dict[str, Any]:
    """
    Fetch with Playwright and clean text
    """
    from utils.clean_text import clean_text

    result = await crawl_https_playwright(url=url, timeout=timeout)
    result["clean_text"] = clean_text(result.get("text", ""))
    result["clean_text_length"] = len(result["clean_text"])
    return result
