"""
OPTIONAL / MANUAL stage: scrapes Sardinian dictionary sources to (re)build the
vocabulary files under data/dictionaries/. Not part of the default pipeline run —
the scraped files are already committed to the repo, and re-scraping takes several
minutes and puts load on third-party public sites (rate-limited to 0.5s/request
here on purpose). Run individual functions manually if a dictionary needs refreshing.

Sources:
  - https://ditzionariu.nor-web.eu/  (mixed varieties; NOT used by the final
    pipeline, kept here for provenance/comparison only)
  - https://sardu.wiki/              (Campidanese-specific, standard + non-standard
    lemmas; both are used by the final pipeline)
"""

import argparse
import os
import re
import string
import time

import requests
from bs4 import BeautifulSoup

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DICT_DIR = os.path.join(SCRIPT_DIR, "..", "..", "data", "dictionaries")
REQUEST_DELAY_S = 0.5
USER_AGENT = "Mozilla/5.0"


def scrape_ditzionariu_page(letter, page):
    """Scrapes a single page of ditzionariu.nor-web.eu for a given letter."""
    headers = {"User-Agent": USER_AGENT}
    url = f"https://ditzionariu.nor-web.eu/it/leghe/{letter}/{page}"

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"[scrape_dictionaries] Request failed for {url}")
        return set()

    soup = BeautifulSoup(response.text, "html.parser")

    entries = set()
    for entry in soup.find_all("div", class_="shadow-z-1"):
        original_text = entry.find("strong").text.strip()
        for word in original_text.split(","):
            clean_word = re.sub(r"[^a-zàèìòùáéíóú']+", "", word.lower()).strip()
            if len(clean_word) > 1:
                entries.add(clean_word)

    return entries


def scrape_ditzionariu():
    """Scrapes the full ditzionariu.nor-web.eu dictionary, letter by letter."""
    letters = list(string.ascii_uppercase) + ["TZ"]  # "TZ" has its own page
    full_dict = set()

    for letter in letters:
        print(f"[scrape_dictionaries] Starting letter: {letter}")
        page = 1
        while True:
            print(f"[scrape_dictionaries] Scraping {letter} - page {page}...", end="\r")
            words_in_page = scrape_ditzionariu_page(letter, page)
            if not words_in_page:
                print(f"\n[scrape_dictionaries] Done with letter {letter}. Found {page - 1} pages.")
                break
            full_dict.update(words_in_page)
            page += 1
            time.sleep(REQUEST_DELAY_S)

    output_path = os.path.join(DICT_DIR, "ditzionariu_in_linia.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        for word in sorted(full_dict):
            f.write(word + "\n")
    print(f"[scrape_dictionaries] Saved {len(full_dict)} words -> {output_path}")
    return full_dict


def scrape_sardu_wiki_page(lemmas_url, session):
    """Scrapes a single lemma page of sardu.wiki, returning (standard, non_standard) forms."""
    headers = {"User-Agent": USER_AGENT}
    response = session.get(lemmas_url, headers=headers)
    if response.status_code != 200:
        return [], []

    soup = BeautifulSoup(response.text, "html.parser")

    def extract(css_class):
        words = []
        for word_el in soup.find_all(class_=css_class):
            for part in re.split(r"[\s']+", word_el.text.lower()):
                clean_word = re.sub(r"[^a-zàèìòùáéíóú\-]+", "", part).strip(" -")
                if len(clean_word) > 1:
                    words.append(clean_word)
        return words

    return extract("standard-lemma campidanese"), extract("nonstandard-lemma campidanese")


def scrape_sardu_wiki():
    """Scrapes the full sardu.wiki Campidanese lemma category, following pagination."""
    start_url = "https://sardu.wiki/index.php/Category:Lemas"
    current_category_url = start_url
    session = requests.Session()

    full_std, full_non_std = set(), set()
    page_num = 1

    while current_category_url:
        print(f"\n[scrape_dictionaries] Scraping page {page_num}...")
        headers = {"User-Agent": USER_AGENT}
        response = session.get(current_category_url, headers=headers)
        if response.status_code != 200:
            print(f"[scrape_dictionaries] Failed to load category: {current_category_url}")
            break

        soup = BeautifulSoup(response.text, "html.parser")
        elements = soup.find_all("div", class_="mw-category-group")

        entry_links = []
        for el in elements:
            for a in el.find_all("a", href=True):
                if a["href"].startswith("/index.php/"):
                    entry_links.append("https://sardu.wiki" + a["href"])

        for i, url in enumerate(entry_links, 1):
            print(f"[scrape_dictionaries] Scraping lemma {i}/{len(entry_links)}...", end="\r")
            std_words, non_std_words = scrape_sardu_wiki_page(url, session)
            full_std.update(std_words)
            full_non_std.update(non_std_words)
            time.sleep(REQUEST_DELAY_S)

        next_page_link = None
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "pagefrom=" in href and "next" in a.text.lower():
                next_page_link = "https://sardu.wiki" + href
                break

        current_category_url = next_page_link
        page_num += 1

    std_path = os.path.join(DICT_DIR, "campidanese_standard.txt")
    with open(std_path, "w", encoding="utf-8") as f:
        for word in sorted(full_std):
            f.write(word + "\n")
    print(f"[scrape_dictionaries] Saved {len(full_std)} words -> {std_path}")

    non_std_path = os.path.join(DICT_DIR, "campidanese_non_standard.txt")
    with open(non_std_path, "w", encoding="utf-8") as f:
        for word in sorted(full_non_std):
            f.write(word + "\n")
    print(f"[scrape_dictionaries] Saved {len(full_non_std)} words -> {non_std_path}")

    return full_std, full_non_std


def parse_args():
    parser = argparse.ArgumentParser(
        description="OPTIONAL/manual: re-scrapes the online Sardinian dictionaries. Slow, hits external sites."
    )
    parser.add_argument("--source", choices=["ditzionariu", "sardu-wiki", "all"], default="all")
    return parser.parse_args()


def main():
    args = parse_args()
    print("[scrape_dictionaries] This will re-scrape the selected dictionary source(s) (several minutes).")
    if args.source in ("ditzionariu", "all"):
        scrape_ditzionariu()
    if args.source in ("sardu-wiki", "all"):
        scrape_sardu_wiki()


if __name__ == "__main__":
    main()
