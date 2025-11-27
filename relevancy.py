import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

import streamlit as st
import pandas as pd
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
    # Keep this fairly small to avoid giant nested div/section duplication
    block = node
    while block and block.name not in ("p", "li", "h1", "h2", "h3", "h4", "h5", "h6"):
        block = block.parent
    return block


def extract_article_text(soup: BeautifulSoup) -> str:
    # Simple content grabber – you can tailor this per site if needed
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
    weight_local: float = 0.6,
    weight_article: float = 0.4,
):
    html = fetch_html(page_url)
    soup = clean_soup(html)

    client_name_l = client_name.lower().strip()
    client_domain_l = client_domain.lower().strip() if client_domain else None

    entries = []

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

    for entry, local_sim in zip(entries, local_sims):
        local_sim = float(local_sim)
        final = weight_local * local_sim + weight_article * article_sim

        entry["local_similarity"] = local_sim
        entry["article_similarity"] = article_sim
        entry["final_score"] = final

    entries.sort(key=lambda x: x["final_score"], reverse=True)
    return entries


# -----------------------
# Streamlit UI
# -----------------------

def main():
    st.title("Client Mention & Relevancy Checker")

    st.markdown(
        "Analyse how contextually relevant a page is to a client, "
        "based on brand mentions and links to their domain."
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
            value=(""
            ),
            height=150,
        )

        st.markdown("### Scoring balance")

        weight_local = st.slider(
            "Local context weight",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.05,
        )
        
        # Automatically derived so total = 1.0
        weight_article = round(1.0 - weight_local, 2)
        
        st.slider(
            "Article-level context weight (auto)",
            min_value=0.0,
            max_value=1.0,
            value=weight_article,
            step=0.05,
            disabled=True,
        )
        
        st.caption(
            f"Final score = {weight_local:.2f} × local similarity "
            f"+ {weight_article:.2f} × article similarity"
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
                    client_name=client_name.strip() or "",
                    client_domain=client_domain.strip() or None,
                    weight_local=weight_local,
                    weight_article=weight_article,
                )

            if not results:
                st.warning("No client mentions or client URLs found on this page.")
                return

            # Convert to DataFrame for display
            df = pd.DataFrame(results)
            # Shorten context for table view
            df["context_snippet"] = df["context"].apply(
                lambda x: x[:200] + ("..." if len(x) > 200 else "")
            )

            st.subheader("Results")
            st.write(
                "Sorted by `final_score` (combination of local and article-level similarity)."
            )

            display_cols = [
                "type",
                "reason",
                "url",
                "anchor",
                "local_similarity",
                "article_similarity",
                "final_score",
                "context_snippet",
            ]
            display_cols = [c for c in display_cols if c in df.columns]

            st.dataframe(
                df[display_cols].style.format(
                    {
                        "local_similarity": "{:.3f}",
                        "article_similarity": "{:.3f}",
                        "final_score": "{:.3f}",
                    }
                ),
                use_container_width=True,
            )

        except Exception as e:
            st.error(f"Something went wrong: {e}")


if __name__ == "__main__":
    main()
