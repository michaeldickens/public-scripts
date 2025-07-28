"""fetch_doi.py
-------------

Author: Michael Dickens <michael@mdickens.me>
Created: 2024-08-23

Look up an article citation given a DOI or URL. The DOI/URL is passed as a
command-line argument, and the citation is printed to stdout across multiple
lines in the following format:

    Lastname et al. (Year)
    All author names (Year)
    Title
    URL
    DOI

I use this script as the backend to an Emacs command that inserts citations into
articles with the press of a button.

"""

import re
import requests
import sys


def fetch_metadata(search_url, email_address=None):
    """
    Get metadata for a DOI or URL from CrossRef.
    """
    crossref_url = f"https://api.crossref.org/works/{search_url}"
    headers = {}
    if email_address:
        # provide contact info in User-Agent header to get the better server
        # that's "reserved for polite users"
        headers = {"User-Agent": f"mailto:{email_address}"}
    response = requests.get(crossref_url, headers=headers)
    response.raise_for_status()
    data = response.json()
    ref = data["message"]
    author_last_names = [author["family"] for author in ref["author"]]
    if len(author_last_names) > 2:
        author_last_names_str = f"{author_last_names[0]} et al."
    elif len(author_last_names) == 2:
        author_last_names_str = " & ".join(author_last_names)
    else:
        author_last_names_str = author_last_names[0]

    authors_full = [
        # MLA/Chicago style, but with names ordered as Given Family
        # f"{author['given']} {author['family']}"
        # APA/Harvard style
        f"{author['family']}, {' '.join([name[0] + '.' for name in re.split('[ -.]', author['given']) if len(name) > 0])}"
        for author in ref["author"]
    ]

    if len(authors_full) > 7:
        # Show at most 7 authors
        authors_full_str = ", ".join(authors_full[:7]) + " et al."
    else:
        authors_full_str = (
            ", ".join(authors_full[:-1])
            + (", & " if len(authors_full) >= 2 else "")
            + authors_full[-1]
        )
    year = ref["issued"]["date-parts"][0][0]
    doi_url = f"https://doi.org/{ref['DOI']}"
    proper_url = search_url if search_url.startswith("http") else doi_url
    full_citation = (
        f"{authors_full_str} ({year})\n{ref['title'][0]}\n{proper_url}\n{ref['DOI']}"
    )
    short_citation = f"{author_last_names_str} ({year})"
    return {"full_citation": full_citation, "short_citation": short_citation}


if __name__ == "__main__":
    # get DOI or URL from command line arguments
    if len(sys.argv) < 2:
        print("Usage: fetch_doi.py <DOI or URL> [email address]")
        sys.exit(1)
    doi = sys.argv[1]
    email_address = sys.argv[2] if len(sys.argv) > 2 else None
    metadata = fetch_metadata(doi, email_address)
    print(metadata["short_citation"] + "\n" + metadata["full_citation"])
