import re
from dataclasses import dataclass
from typing import Iterable, List

import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

@dataclass(frozen=True, repr=True)
class WikipediaPath:
    article: str
    header: str

    def __str__(self):
        return f"{self.article} - {self.header}"

@dataclass(frozen=True, repr=True)
class Section:
    location: WikipediaPath
    text: str

    def __str__(self):
        return f"{self.location}:\n{self.text}"

def wikipedia_splitter(
        contents: str, article_title: str, token_limit: int, split_point_regexes: List[str]
) -> Iterable[Section]:
    split_point_regex = split_point_regexes[0]
    sections = re.split(split_point_regex, contents)

    if not sections[0].strip():
        # Remove the first section if it's empty (this happens when the file starts with a "#" line)
        sections.pop(0)
    else:
        # Otherwise: Wikipedia articles often begin with a section that has no `==` header.
        first_section = sections.pop(0)
        yield Section(
            location=WikipediaPath(article=article_title, header=article_title),
            text=first_section,
        )

    for section in sections:
        if not section.strip():
            # Remove trailing empty sections.
            continue

        header = section.splitlines()[0].strip()
        if "=" in split_point_regex:
            # If we're splitting on equal-sign headers, then we need to remove the trailing equal signs
            header = re.sub(r"=+$", "", header).strip()

        # To be better steer embeddings, we include the article's title and section name with one another above the text.
        emit = Section(
            location=WikipediaPath(article=article_title, header=header),
            text=f"{article_title}: {section}",
        )

        if len(str(section).replace("\n", " ")) > token_limit:
            print(f"Section is too long: {emit.location}, splitting")
            subtitle = f"{article_title} - {header}"
            # If the section is too long, split it on a lower precedence split point

            yield from wikipedia_splitter(
                section, subtitle, token_limit, split_point_regexes[1:]
            )
        else:
            yield emit