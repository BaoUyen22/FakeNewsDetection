"""
Utilities to fetch and extract news content from HTTPS URLs.

Uses requests + BeautifulSoup for simple and reliable web scraping.
Perfect for extracting text from news articles.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup


DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
}


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


def extract_text_from_html(html: str, url: str = "") -> Dict[str, str]:
    """
    Extract title, description, and visible text from HTML using BeautifulSoup.
    
    Args:
        html: HTML content
        url: Original URL (for context)
    
    Returns:
        Dict with title, description, and text
    """
    soup = BeautifulSoup(html, 'html.parser')
    
    # Remove script and style elements
    for script in soup(["script", "style", "noscript", "iframe", "nav", "header", "footer", "aside"]):
        script.decompose()
    
    # Extract title
    title = ""
    if soup.title:
        title = soup.title.string or ""
    elif soup.find('meta', property='og:title'):
        title = soup.find('meta', property='og:title').get('content', '')
    elif soup.find('h1'):
        title = soup.find('h1').get_text(strip=True)
    
    # Extract description
    description = ""
    meta_desc = soup.find('meta', attrs={'name': 'description'})
    if meta_desc:
        description = meta_desc.get('content', '')
    elif soup.find('meta', property='og:description'):
        description = soup.find('meta', property='og:description').get('content', '')
    
    # Extract main content - prioritize article tags
    text_parts = []
    
    # Try to find main content area
    main_content = (
        soup.find('article') or 
        soup.find('main') or 
        soup.find('div', class_=lambda x: x and any(c in str(x).lower() for c in ['content', 'article', 'post', 'detail', 'fck_detail']))
    )
    
    if main_content:
        # Extract from main content area - get all text from paragraphs
        for tag in main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li']):
            text = tag.get_text(strip=True)
            if text and len(text) > 15:  # Filter out very short text
                text_parts.append(text)
    
    # Fallback: extract all paragraphs from entire page
    if not text_parts:
        for paragraph in soup.find_all('p'):
            text = paragraph.get_text(strip=True)
            if text and len(text) > 15:
                text_parts.append(text)
    
    # Join all text parts
    full_text = ' '.join(text_parts)
    
    return {
        "title": title.strip(),
        "description": description.strip(),
        "text": full_text.strip()
    }


def fetch_url(
    url: str,
    timeout: int = 15,
    verify_ssl: bool = True,
) -> CrawlResult:
    """
    Fetch a URL and return the extracted news content.
    
    Args:
        url: URL to fetch
        timeout: Request timeout in seconds
        verify_ssl: Whether to verify SSL certificates
    
    Returns:
        CrawlResult with extracted content
    """
    normalized_url = validate_url(url)

    try:
        response = requests.get(
            normalized_url,
            headers=DEFAULT_HEADERS,
            timeout=timeout,
            allow_redirects=True,
            verify=verify_ssl,
        )
        response.raise_for_status()

        content_type = response.headers.get("content-type", "")
        html = response.text
        
        # Extract content using BeautifulSoup
        extracted = extract_text_from_html(html, normalized_url)

        return CrawlResult(
            url=normalized_url,
            final_url=str(response.url),
            success=True,
            status_code=response.status_code,
            content_type=content_type,
            title=extracted["title"],
            description=extracted["description"],
            text=extracted["text"],
            text_length=len(extracted["text"]),
        )
        
    except requests.RequestException as exc:
        raise CrawlError(f"Failed to fetch URL: {exc}") from exc


def crawl_https(url: str, timeout: int = 15, verify_ssl: bool = True) -> Dict[str, Any]:
    """
    Public helper used by API code to crawl a news page.
    
    Args:
        url: URL to crawl
        timeout: Request timeout
        verify_ssl: Whether to verify SSL
    
    Returns:
        Dict with crawl results
    """
    result = fetch_url(url=url, timeout=timeout, verify_ssl=verify_ssl)
    return result.to_dict()


def crawl_and_clean(url: str, timeout: int = 15, verify_ssl: bool = True) -> Dict[str, Any]:
    """
    Fetch a URL and return both raw extracted text and a cleaned variant.
    
    Args:
        url: URL to crawl
        timeout: Request timeout
        verify_ssl: Whether to verify SSL
    
    Returns:
        Dict with crawl results including cleaned text
    """
    from utils.clean_text import clean_text

    result = crawl_https(url=url, timeout=timeout, verify_ssl=verify_ssl)
    result["clean_text"] = clean_text(result.get("text", ""))
    result["clean_text_length"] = len(result["clean_text"])
    return result
