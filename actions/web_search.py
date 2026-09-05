"""Browser search automation via Selenium."""

from selenium import webdriver
from selenium.webdriver.chrome.options import Options


def search_web(query: str):
    if not query:
        return "I didn't catch what to search for."

    options = Options()
    options.add_argument("--start-maximized")

    driver = webdriver.Chrome(options=options)
    driver.get(f"https://www.google.com/search?q={query}")

    return f"Here are the search results for {query}."
