import re
from dataclasses import dataclass
from typing import List, Optional, Dict, Any  # Added Any


@dataclass
class TagExtraction:
    """Represents extracted tag content and its presence."""

    content: List[str]
    found: bool


def extract_tags(text: str, tag: str) -> TagExtraction:
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
    Parses parameter descriptions from a specific docstring format.

    Expected format within the docstring:
        Parameters:
          - param_name: Description of the parameter.
          * another_param: Description.

        Args:
          arg1: Description of arg1.
          argument_item: Description of argument_item.

    Returns:
        A dictionary mapping parameter names to their descriptions.
    """
    if not docstring:
        return {}
    params: Dict[str, str] = {}

    # Regex to find parameter sections (Parameters, Args, Arguments - case-insensitive)
    param_section_matches = re.finditer(r"(Parameters|Args|Arguments):\s*\n(.*?)(?:\n\n|\Z)", docstring, re.DOTALL | re.IGNORECASE)

    for match in param_section_matches:
        param_lines = match.group(2).strip().splitlines()
        # Regex to capture '- param_name: description' or '* param_name: description'
        param_line_pattern = re.compile(r"^\s*[-*]\s*(\w+)\s*:(.*)")
        # Regex to capture 'param_name: description' without the bullet point
        simple_param_pattern = re.compile(r"^\s*(\w+)\s*:(.*)")

        for line in param_lines:
            bullet_match = param_line_pattern.match(line)
            if bullet_match:
                param_name = bullet_match.group(1).strip()
                description = bullet_match.group(2).strip()
                params[param_name] = description
                continue  # Move to the next line

            simple_match = simple_param_pattern.match(line)
            if simple_match:
                param_name = simple_match.group(1).strip()
                description = simple_match.group(2).strip()
                params[param_name] = description

    return params
