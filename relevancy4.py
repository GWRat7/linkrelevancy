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
    # Strip obvious non-content
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
    """
    Find a reasonably small block parent so we don't get giant nested div/section duplication.
    """
    block = node
    while block and block.name not in ("p", "li", "h1", "h2", "h3", "h4", "h5", "h6"):
        block = block.parent
    return block


def extract_article_text(soup: BeautifulSoup) -> str:
    """
    Simple heuristic to grab main article text.
    """
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
    """Normalize context so near-identical snippets dedupe cleanly."""
    return " ".join(text.split()).strip().lower()


def analyse_mentions_and_links_semantic(
    page_url: str,
    client_description: str,
    client_name: str,
    client_domain: str | None = None,
    min_chars_context: int = 40,
):
    """
    Scrape a page and find:
      - text mentions of client_name
      - links whose URL contains client_domain

    Score each using SEMANTIC similarity:
      final_score = 0.5 * sim(client_desc, local_context)
                  + 0.5 * sim(client_desc, full_article)
    """
    html = fetch_html(page_url)
    soup = clean_soup(html)

    client_name_l = client_name.lower().strip() if client_name else ""
    client_domain_l = client_domain.lower().strip() if client_domain else None

    entries: list[dict] = []

    # 1) LINK-BASED ENTRIES – only one per unique URL
    if client_domain_l:
        seen_urls = set()
        for a in soup.find_all("a"):
            href = a.get("href")
            if not is_valid_href(href):
                continue

            absolute_url = urljoin(page_url, href).strip()
            url_l = absolute_url.lower()

            if client_domain_l not in url_l:
                continue

            if url_l in seen_urls:
                # already have an entry for this URL
                continue
            seen_urls.add(url_l)

            block = get_block_parent(a)
            if not block:
                continue

            context_text = block.get_text(" ", strip=True)
            if len(context_text) < min_chars_context:
                continue

            entries.append(
                {
                    "type": "link",
                    "url": absolute_url,
                    "anchor": a.get_text(" ", strip=True),
                    "context": context_text,
                    "reason": "client_domain_in_url",
                }
            )

    # 2) TEXT-ONLY MENTIONS – only scan “real” content blocks
    blocks = soup.find_all(["p", "li", "h1", "h2", "h3", "h4", "h5", "h6"])

    for block in blocks:
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
                    "reason": "client_name_in_text",
                }
            )

    if not entries:
        return []

    # 3) DE-DUPE BY CONTEXT (so same sentence/paragraph appears once)
    seen_contexts = set()
    unique_entries = []
    for e in entries:
        key = normalize_context(e["context"])
        if key in seen_contexts:
            continue
        seen_contexts.add(key)
        unique_entries.append(e)
    entries = unique_entries

    # 4) ARTICLE-LEVEL CONTEXT
    article_text = extract_article_text(soup)
    if not article_text:
        article_text = " ".join(e["context"] for e in entries)

    # 5) SEMANTIC SCORING
    model = SentenceTransformer("all-MiniLM-L6-v2")

    emb_client = model.encode(client_description, normalize_embeddings=True)
    emb_article = model.encode(article_text, normalize_embeddings=True)

    article_sim = float(util.cos_sim(emb_client, emb_article))

    local_texts = [e["context"] for e in entries]
    emb_locals = model.encode(local_texts, normalize_embeddings=True)
    local_sims = util.cos_sim(emb_client, emb_locals)[0].tolist()

    # Fixed weights (simple average)
    weight_local = 0.5
    weight_article = 0.5

    for entry, local_sim in zip(entries, local_sims):
        local_sim = float(local_sim)
        final = weight_local * local_sim + weight_article * article_sim

        entry["local_similarity"] = local_sim
        entry["article_similarity"] = article_sim
        entry["final_score"] = final

    entries.sort(key=lambda x: x["final_score"], reverse=True)
    return entries

import re

def extract_sentence_with_anchor(context: str, anchor: str) -> str:
    """
    From a block of text, return the sentence that contains the anchor text.
    Falls back to full context if sentence cannot be isolated.
    """
    if not anchor or not context:
        return context

    # Split into sentences (simple but effective)
    sentences = re.split(r'(?<=[.!?])\s+', context)

    anchor_l = anchor.lower()
    for sentence in sentences:
        if anchor_l in sentence.lower():
            return sentence.strip()

    return context

# -----------------------
# Streamlit UI
# -----------------------

def main():
    st.title("Client Mention & Relevancy Checker")

    st.markdown(
        "Analyse how contextually relevant a page is to a client, "
        "based on your description.\n\n"
    )

    with st.form("input_form"):
        url = st.text_input(
            "Page URL",
            value="",
        )
        client_name = st.text_input("Client name (as mentioned in text)", value="")
        client_domain = st.text_input("Client domain (e.g. example.com)", value="")
        client_description = st.text_area(
            "Client description (what they do / offer)",
            value=(
                ""
            ),
            height=150,
        )

        submitted = st.form_submit_button("Analyse")

    if submitted:
        if not url.strip():
            st.error("Please enter a URL.")
            return
        if not client_description.strip():
            st.error("Please enter a client description.")
            return
        if not client_name.strip() and not client_domain.strip():
            st.error("Please enter at least a client name or a client domain.")
            return

        try:
            with st.spinner("Fetching and analysing page..."):
                results = analyse_mentions_and_links_semantic(
                    page_url=url.strip(),
                    client_description=client_description.strip(),
                    client_name=client_name.strip(),
                    client_domain=client_domain.strip() or None,
                )

            if not results:
                st.warning("No client mentions or client URLs found on this page.")
                return

            st.subheader("Results")

            for i, r in enumerate(results, start=1):
                # Linking URL or mention
                if r.get("url"):
                    link_display = r["url"]
                else:
                    link_display = "Brand mention (no link)"

                # Anchor text or full context
                # Anchor sentence (for links) or paragraph (for mentions)
                if r.get("anchor"):
                    anchor_or_mention = extract_sentence_with_anchor(
                        context=r["context"],
                        anchor=r["anchor"],
                    )
                else:
                    anchor_or_mention = r["context"]

                st.markdown(f"**Link:** {link_display}")
                st.markdown(f"**Score:** `{r['final_score']:.3f}`")
                st.markdown("**Anchor / Mention context:**")
                st.markdown(f"> {anchor_or_mention}")
                st.markdown("---")

        except Exception as e:
            st.error(f"Something went wrong: {e}")


if __name__ == "__main__":
    main()
