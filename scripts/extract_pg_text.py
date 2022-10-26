# pylint: disable=invalid-name
"""Download all essays from Paul Graham
"""

from pathlib import Path
import re
import time

import requests


# base url
base = "http://www.paulgraham.com/"

# get the html
res = requests.get(f"{base}articles.html").text
links = re.findall(
    r'<font size=2 face="verdana"><a href="([a-zA-Z0-9\-\.html]*)">([a-zA-Z0-9\-\ ]*)<',
    res,
)

# get absolute links
links = [(f"{base}{link}", name) for link, name in links]

out_dir = Path("paul_graham_essays")
out_dir.mkdir(exist_ok=True)


# parse every link for text
for i, (link, name) in enumerate(links[:2]):

    # get the aricle html, without abusing the site
    time.sleep(0.1)
    res = requests.get(link).text

    # find the text
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
        article = parsed_articles[0].replace("<br>", "\n")
        article += "\n"

        # write retreived text in a file
        name = "_".join(name.split(" ")).lower()
        name = re.sub(r"[\W\s]+", "", name)
        out_path = out_dir / f"{i:03d}_{name}.txt"
        with open(out_path, "w", encoding="UTF-8") as f:
            f.write(article)
        print(i + 1, ". Text taken from: ", link)
