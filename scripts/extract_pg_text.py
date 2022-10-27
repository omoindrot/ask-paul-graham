# pylint: disable=invalid-name
"""Download all essays from Paul Graham
"""

from pathlib import Path
import re
import time

from bs4 import BeautifulSoup
import requests


# Base url
base = "http://www.paulgraham.com/"

# Get the html
res = requests.get(f"{base}articles.html").text
links = re.findall(
    r'<font size=2 face="verdana"><a href="([a-zA-Z0-9\-\.html]*)">([a-zA-Z0-9\-\ ]*)<',
    res,
)

# Get absolute links
links = [(f"{base}{link}", name) for link, name in links]

out_dir = Path("pg_essays")
out_dir.mkdir(exist_ok=True)


# Parse every link for text
for i, (link, name) in enumerate(links):
    # Get the aricle html, without abusing the site
    time.sleep(0.1)
    res = requests.get(link).text

    # Find the text
    parsed_articles = re.findall(
        r'<font size=2 face="verdana">(.*)<br><br></font>', res, re.DOTALL
    )
    if len(parsed_articles) == 0:
        parsed_articles = re.findall(
            r'<font size=2 face="verdana">(.*)<br><br><br clear=all></font>',
            res,
            re.DOTALL,
        )

    if len(parsed_articles) == 0:
        print(i + 1, ". Could not get text from: ", link)
    else:
        article = parsed_articles[0]

        paragraphs = article.split("<br><br>")
        paragraphs = [
            p.replace("\r\n", " ").replace("\n", " ").strip() for p in paragraphs
        ]
        paragraphs = [p for p in paragraphs if p != ""]
        article = "\n".join(paragraphs)

        article = BeautifulSoup(article, features="html.parser").text.strip()
        paragraphs = article.split("\n")
        paragraphs = [p.strip() for p in paragraphs]
        paragraphs = [p for p in paragraphs if p != ""]
        if paragraphs[-1].startswith("Thanks to"):
            paragraphs = paragraphs[:-1]
        article = "\n".join(paragraphs)

        # Remove call to join YC
        call = "Want to start a startup?  Get funded by\nY Combinator."
        article = article.replace(call, "")
        article = article.strip()

        # Remove end of page references like [1], [2]...
        article = re.sub(r"\[[0-9]\]", "", article)
        # Remove useless whitespaces
        article = re.sub(r"\s\s+", " ", article)

        # Write retreived text in a file
        name = "_".join(name.split(" ")).lower()
        name = re.sub(r"[\W\s]+", "", name)
        out_path = out_dir / f"{i+1:03d}_{name}.txt"
        with open(out_path, "w", encoding="UTF-8") as f:
            f.write(article)
        print(i + 1, ". Text taken from: ", link)
