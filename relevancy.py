import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

import streamlit as st
from sentence_transformers import SentenceTransformer, util


# -----------------------
# Scraping + analysis logic
# -----------------------

def fetch_html(url: str, timeout: int = 10) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    }
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def clean_soup(html: str) -> BeautifulSoup:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "header", "footer", "nav", "form"]):
        tag.decompose()
    return soup


def is_valid_href(href: str) -> bool:
    if not href:
        return False
    href = href.strip().lower()
    if href.startswith(("#", "javascript:", "mailto:")):
        return False
    return True


def get_block_parent(node):
    block = node
    while block and block.name not in ("p", "li", "h1", "h2", "h3", "h4", "h5", "h6"):
        block = block.parent
    return block


def extract_article_text(soup: BeautifulSoup) -> str:
    blocks = soup.find_all(["article", "main"])
    if not blocks:
        blocks = soup.find_all(["p", "li"])

    texts = []
    for b in blocks:
        t = b.get_text(" ", strip=True)
        if t:
            texts.append(t)
    return " ".join(texts)


def normalize_context(text: str) -> str:
    return " ".join(text.split()).strip().lower()


def chunk_text(text: str, max_words: int = 120, min_words: int = 30) -> list[str]:
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current = []

    for sent in sentences:
        words = sent.split()
        if len(current) + len(words) > max_words and len(current) >= min_words:
            chunks.append(" ".join(current))
            current = words
        else:
            current.extend(words)

    if current and (len(current) >= min_words or not chunks):
        chunks.append(" ".join(current))

    return chunks


def extract_sentence_with_anchor(context: str, anchor: str) -> str:
    if not anchor or not context:
        return context

    sentences = re.split(r"(?<=[.!?])\s+", context)
    anchor_l = anchor.lower()

    for sent in sentences:
        if anchor_l in sent.lower():
            return sent.strip()

    return context


# -----------------------
# Core analysis (ARTICLE-ONLY)
# -----------------------

def analyse_mentions_and_links_semantic(
    page_url: str,
    client_description: str,
    client_name: str,
    client_domain: str | None = None,
    min_chars_context: int = 40,
):
    html = fetch_html(page_url)
    soup = clean_soup(html)

    client_name_l = client_name.lower().strip() if client_name else ""
    client_domain_l = client_domain.lower().strip() if client_domain else None

    entries: list[dict] = []

    # 1) LINK ENTRIES
    if client_domain_l:
        seen_urls = set()
        for a in soup.find_all("a"):
            href = a.get("href")
            if not is_valid_href(href):
                continue

            absolute_url = urljoin(page_url, href).strip()
            if client_domain_l not in absolute_url.lower():
                continue

            if absolute_url.lower() in seen_urls:
                continue
            seen_urls.add(absolute_url.lower())

            block = get_block_parent(a)
            if not block:
                continue

            context = block.get_text(" ", strip=True)
            if len(context) < min_chars_context:
                continue

            entries.append(
                {
                    "type": "link",
                    "url": absolute_url,
                    "anchor": a.get_text(" ", strip=True),
                    "context": context,
                }
            )

    # 2) BRAND MENTIONS
    for block in soup.find_all(["p", "li", "h1", "h2", "h3", "h4", "h5", "h6"]):
        text = block.get_text(" ", strip=True)
        if len(text) < min_chars_context:
            continue

        if client_name_l and client_name_l in text.lower():
            entries.append(
                {
                    "type": "mention",
                    "url": None,
                    "anchor": None,
                    "context": text,
                }
            )

    if not entries:
        return []

    # De-duplicate
    seen = set()
    uniq = []
    for e in entries:
        key = normalize_context(e["context"])
        if key not in seen:
            seen.add(key)
            uniq.append(e)
    entries = uniq

    # 3) ARTICLE SIMILARITY (CHUNKED, BEST MATCH)
    article_text = extract_article_text(soup)
    if not article_text:
        article_text = " ".join(e["context"] for e in entries)

    chunks = chunk_text(article_text)
    if not chunks:
        chunks = [article_text]

    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb_client = model.encode(client_description, normalize_embeddings=True)
    emb_chunks = model.encode(chunks, normalize_embeddings=True)

    sims = util.cos_sim(emb_client, emb_chunks)[0].tolist()
    article_score = float(max(sims)) if sims else 0.0

    for e in entries:
        e["score"] = article_score

    return entries


# -----------------------
# Streamlit UI
# -----------------------

def main():
    st.title("Article Relevancy Checker (Article-Only)")

    st.markdown(
        "Relevancy score is based **only on the article as a whole**, "
        "using chunked semantic similarity.\n\n"
        "**Score = strongest topical alignment between any part of the article and the client**."
    )

    with st.form("input_form"):
        url = st.text_input("Page URL")
        client_name = st.text_input("Client name (as mentioned in text)")
        client_domain = st.text_input("Client domain (optional)")
        client_description = st.text_area(
            "Client description",
            height=150,
        )
        submitted = st.form_submit_button("Analyse")

    if submitted:
        if not url or not client_description:
            st.error("URL and client description are required.")
            return

        with st.spinner("Analysing article..."):
            results = analyse_mentions_and_links_semantic(
                page_url=url.strip(),
                client_description=client_description.strip(),
                client_name=client_name.strip(),
                client_domain=client_domain.strip() or None,
            )

        if not results:
            st.warning("No mentions or links found.")
            return

        st.subheader("Results")

        for i, r in enumerate(results, start=1):
            link_display = r["url"] if r["url"] else "Brand mention (no link)"

            if r["anchor"]:
                context_display = extract_sentence_with_anchor(
                    r["context"], r["anchor"]
                )
            else:
                context_display = r["context"]

            st.markdown(f"### Result {i}")
            st.markdown(f"**Link:** {link_display}")
            st.markdown(f"**Relevancy score:** `{r['score']:.3f}`")
            st.markdown("**Context:**")
            st.markdown(f"> {context_display}")
            st.markdown("---")


if __name__ == "__main__":
    main()

