from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import os
import time
import json

BASE_URL = "https://www.canlii.org"
SCC_INDEX = "https://www.canlii.org/en/ca/scc/"
OUTPUT_DIR = "canlii_cases"


def init_driver():
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)


def get_case_links(index_url, max_cases=10):
    driver = init_driver()
    driver.get(index_url)
    time.sleep(3)
    soup = BeautifulSoup(driver.page_source, "html.parser")
    driver.quit()

    case_links = []
    for a in soup.select("a"):
        href = a.get("href", "")
        if href.startswith("/en/ca/scc/") and href.count("/") > 3:
            case_links.append(BASE_URL + href)

    return list(set(case_links))[:max_cases]


def get_case_data(case_url):
    driver = init_driver()
    driver.get(case_url)
    time.sleep(3)
    soup = BeautifulSoup(driver.page_source, "html.parser")
    driver.quit()

    # Title and date
    title_el = soup.select_one(".headnotes h1") or soup.select_one("h1")
    date_el = soup.select_one(".decision-date") or soup.find(string=lambda s: s and "Dated:" in s)

    title = title_el.get_text(strip=True) if title_el else "Unknown Title"
    date = date_el.get_text(strip=True) if date_el and hasattr(date_el, 'get_text') else date_el.strip() if date_el else "Unknown Date"

    # Core content
    content_el = soup.select_one(".decision-content") or soup.select_one(".contentDecision") or soup.select_one("main")
    content = content_el.get_text(separator="\n").strip() if content_el else ""

    return {
        "url": case_url,
        "title": title,
        "date": date,
        "content": content
    }


def save_case_json(case_data, filename):
    with open(os.path.join(OUTPUT_DIR, filename), "w", encoding="utf-8") as f:
        json.dump(case_data, f, indent=2, ensure_ascii=False)


def preprocess_text(text):
    import re
    text = re.sub(r'\s+', ' ', text)  # normalize whitespace
    text = re.sub(r'Page \d+', '', text)  # remove page headers/footers
    return text.strip()


def scrape_canlii_cases(max_cases=5):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    links = get_case_links(SCC_INDEX, max_cases=max_cases)

    for i, link in enumerate(links):
        print(f"Scraping case {i+1}/{len(links)}: {link}")
        try:
            case = get_case_data(link)
            if len(case['content']) < 100:
                print(f"⚠️ Warning: Very short content. Skipping.")
                continue
            case['content'] = preprocess_text(case['content'])
            save_case_json(case, f"case_{i+1}.json")
            print(f"✅ Saved case_{i+1}.json with {len(case['content'])} characters.")
        except Exception as e:
            print(f"❌ Failed to scrape {link}: {e}")


if __name__ == "__main__":
    scrape_canlii_cases(max_cases=10)
