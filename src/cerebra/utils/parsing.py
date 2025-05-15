import re
from dataclasses import dataclass
from typing import List, Optional, Dict


@dataclass
class TagExtraction:
    """Represents extracted tag content and its presence."""

    content: List[str]
    found: bool


def extract_xml_tags(text: str, tag: str) -> TagExtraction:
    """
    Extract all instances of content within specified XML-style tags.

    Args:
        text: The text to parse.
        tag: The XML-style tag name to extract content from (e.g., "response").

    Returns:
        A TagExtraction object containing a list of extracted content strings
        and a boolean indicating if any tags were found.
    """

    tag_pattern = rf"<{tag}>(.*?)</{tag}>"  # finds <tag>content</tag>, capturing content non-greedily
    matches = re.findall(tag_pattern, text, re.DOTALL)  # re.DOTALL allows '.' to match newline characters

    return TagExtraction(content=[content.strip() for content in matches], found=bool(matches))


def parse_docstring_params(docstring: Optional[str]) -> Dict[str, str]:
    """
    Parses parameter descriptions from various common docstring formats.

    Attempts to find a parameter section (e.g., "Parameters:", "Args:")
    and parse lines within it. If no section is found, it tries to parse
    all lines in the docstring.

    Supports:
    - Sphinx/reST style keywords (e.g., ":param name: description")
    - Google style (e.g., "name (type): description")
    - Bullet points (e.g., "- name: description")
    - Simple "name: description" lines.

    Args:
        docstring: The docstring to parse.

    Returns:
        A dictionary mapping parameter names to their descriptions.
        Extracts only the first line of a multi-line description.
    """
    if not docstring:
        return {}

    params: Dict[str, str] = {}

    # Regex for parameter sections (case-insensitive).
    # Captures the content block of the section.
    section_pattern = re.compile(r"^\s*(?:Parameters|Args|Arguments|Params):?\s*\n((?:.|\n)*?)(?=\n\s*\n|\Z)", re.MULTILINE | re.IGNORECASE)

    # Regex patterns for individual parameter lines. Order can be important.
    # 1. Sphinx/reST style with keyword (e.g., :param foo: description)
    # Allows for *args, **kwargs by including '*' in name.
    sphinx_kw_pattern = re.compile(r"^\s*:(?:param|parameter|arg|argument)\s+([\w\*]+)\s*:\s*(.+)")
    # 2. Google style (e.g., foo (int): description)
    # Captures name, optional type (not used here), and description.
    google_pattern = re.compile(r"^\s*([\w\*]+)\s*(?:\(([^)]*)\))?:\s*(.+)")
    # 3. Bullet point style (e.g., - foo: description or * foo: description)
    bullet_pattern = re.compile(r"^\s*[-*]\s+([\w\*]+)\s*:\s*(.+)")
    # 4. Simple style (e.g., foo: description) - often a fallback
    simple_pattern = re.compile(r"^\s*([\w\*]+)\s*:\s*(.+)")

    content_to_parse = docstring
    section_match = section_pattern.search(docstring)

    if section_match:
        # If a specific parameter section is found, parse content within that section.
        content_to_parse = section_match.group(1)

    for line in content_to_parse.splitlines():
        stripped_line = line.strip()
        if not stripped_line:  # Skip empty lines
            continue

        # Try matching patterns in a specific order
        # For each pattern, if matched, extract name and description, then continue to next line.
        match_obj = sphinx_kw_pattern.match(stripped_line)
        if match_obj:
            param_name, description = match_obj.groups()
            # Take only the first line of the description if it's multi-line
            params[param_name.strip()] = description.strip().split("\n", 1)[0]
            continue

        match_obj = google_pattern.match(stripped_line)
        if match_obj:
            param_name, _type_info, description = match_obj.groups()
            params[param_name.strip()] = description.strip().split("\n", 1)[0]
            continue

        match_obj = bullet_pattern.match(stripped_line)
        if match_obj:
            param_name, description = match_obj.groups()
            params[param_name.strip()] = description.strip().split("\n", 1)[0]
            continue

        # Fallback to the simple pattern if parsing the whole docstring or within a section
        # and no other pattern matched.
        match_obj = simple_pattern.match(stripped_line)
        if match_obj:
            param_name, description = match_obj.groups()
            params[param_name.strip()] = description.strip().split("\n", 1)[0]

    return params
