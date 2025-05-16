from cerebra.utils.parsing import parse_docstring_params, extract_xml_tags, TagExtraction

# --- Tests for extract_xml_tags ---


def test_extract_xml_tags_single_occurrence():
    text = "<note>This is a note.</note>"
    expected = TagExtraction(content=["This is a note."], found=True)
    assert extract_xml_tags(text, "note") == expected


def test_extract_xml_tags_single_occurrence_with_attributes():
    text = "<note type='important' priority='high'>This is a note with attributes.</note>"
    expected = TagExtraction(content=["This is a note with attributes."], found=True)
    assert extract_xml_tags(text, "note") == expected


def test_extract_xml_tags_multiple_occurrences():
    text = "<item>Apple</item><item status='ripe'>Banana</item>"
    expected = TagExtraction(content=["Apple", "Banana"], found=True)
    assert extract_xml_tags(text, "item") == expected


def test_extract_xml_tags_no_occurrence():
    text = "Just some plain text."
    expected = TagExtraction(content=[], found=False)
    assert extract_xml_tags(text, "tag") == expected


def test_extract_xml_tags_empty_content():
    text = "<empty></empty><filled>content</filled><empty_again></empty_again><empty_with_attr class='test'></empty_with_attr>"
    assert extract_xml_tags(text, "empty") == TagExtraction(content=[""], found=True)
    assert extract_xml_tags(text, "filled") == TagExtraction(content=["content"], found=True)
    assert extract_xml_tags(text, "empty_again") == TagExtraction(content=[""], found=True)
    assert extract_xml_tags(text, "empty_with_attr") == TagExtraction(content=[""], found=True)


def test_extract_xml_tags_multiline_content():
    text = "<script type='text/javascript'>\n  let x = 10;\n  console.log(x);\n</script>"
    expected_content = "let x = 10;\n  console.log(x);"
    expected = TagExtraction(content=[expected_content.strip()], found=True)  # .strip() is applied by the function
    assert extract_xml_tags(text, "script") == expected


def test_extract_xml_tags_with_nested_tags_content_only():
    text = "<outer id='1'><inner>Nested Text</inner> More outer text <inner>Another Nest</inner></outer>"
    # Content of <outer> includes the <inner> tags themselves as strings
    expected_outer_content = "<inner>Nested Text</inner> More outer text <inner>Another Nest</inner>"
    expected_outer = TagExtraction(content=[expected_outer_content], found=True)
    assert extract_xml_tags(text, "outer") == expected_outer

    expected_inner_content = ["Nested Text", "Another Nest"]
    expected_inner = TagExtraction(content=expected_inner_content, found=True)
    assert extract_xml_tags(text, "inner") == expected_inner


def test_extract_xml_tags_whitespace_stripping():
    text = "<data>  Padded data  </data>"
    expected = TagExtraction(content=["Padded data"], found=True)
    assert extract_xml_tags(text, "data") == expected


def test_extract_xml_tags_with_special_regex_chars_in_tag_name():
    """Tests that tag names with special regex characters are handled correctly due to re.escape."""
    text_dot = "<my.tag>content with dot</my.tag>"
    expected_dot = TagExtraction(content=["content with dot"], found=True)
    assert extract_xml_tags(text_dot, "my.tag") == expected_dot

    text_q = "<my?tag>content with q</my?tag>"
    expected_q = TagExtraction(content=["content with q"], found=True)
    assert extract_xml_tags(text_q, "my?tag") == expected_q

    text_star = "<my*tag>content with star</my*tag>"
    expected_star = TagExtraction(content=["content with star"], found=True)
    assert extract_xml_tags(text_star, "my*tag") == expected_star

    assert extract_xml_tags(text_dot, "another.tag") == TagExtraction(content=[], found=False)


# --- Tests for parse_docstring_params ---


def test_parse_google_style_docstring():
    doc = '''"""
    Function summary.

    Args:
        a (int): description of a.
        b (str, optional): description of b.
            This is a second line for b's description.
        c: description for c without type.
    """'''
    params = parse_docstring_params(doc)
    # The parser is expected to only extract the first line of the description.
    assert params == {"a": "description of a.", "b": "description of b.", "c": "description for c without type."}


def test_parse_sphinx_style_docstring():
    doc = '''"""
    Summary.

    :param x: x desc
                continues here.
    :type x: float
    :param y: y desc
    :type y: bool
    :param z: z desc
    """'''
    params = parse_docstring_params(doc)
    assert params == {"x": "x desc", "y": "y desc", "z": "z desc"}


def test_parse_bullet_style_docstring():
    doc = '''"""
    My Function.

    Parameters:
      - param1: The first parameter.
        It has two lines.
      * param2 : The second one.
    """'''
    params = parse_docstring_params(doc)
    assert params == {"param1": "The first parameter.", "param2": "The second one."}


def test_parse_simple_colon_style_docstring_in_section():
    doc = '''"""
    Another function.

    Arguments:
      argA: Description for A.
      argB: Description for B.
            Spanning multiple lines.
    """'''
    params = parse_docstring_params(doc)
    assert params == {"argA": "Description for A.", "argB": "Description for B."}


def test_parse_docstring_no_param_section_fallback():
    doc = '''"""
    This function has no explicit parameter section.
    param_one: Description for one.
    param_two (str): Description for two.
    :param sphinx_style_no_section: Desc for sphinx.
    """'''
    params = parse_docstring_params(doc)
    assert params == {"param_one": "Description for one.", "param_two": "Description for two.", "sphinx_style_no_section": "Desc for sphinx."}


def test_parse_docstring_only_summary():
    doc = '''"""Just a summary, no parameters."""'''
    params = parse_docstring_params(doc)
    assert params == {}


def test_parse_empty_docstring():
    doc = ""
    params = parse_docstring_params(doc)
    assert params == {}
    params_none = parse_docstring_params(None)
    assert params_none == {}


def test_parse_docstring_mixed_formats_in_section():
    doc = '''"""
    Mixed parameters.

    Args:
        google_param (type): Google style description.
        - bullet_param: Bullet style description.
        simple_param: Simple style description.
        :param sphinx_param_correct: Sphinx style description.
    """'''
    params = parse_docstring_params(doc)
    assert params == {
        "google_param": "Google style description.",
        "bullet_param": "Bullet style description.",
        "simple_param": "Simple style description.",
        "sphinx_param_correct": "Sphinx style description.",
    }


def test_parse_docstring_param_name_with_asterisks():
    doc = '''"""
    Testing *args and **kwargs.

    Parameters:
        *args: Variable positional arguments.
        **kwargs: Variable keyword arguments.
    """'''
    params = parse_docstring_params(doc)
    assert params == {"*args": "Variable positional arguments.", "**kwargs": "Variable keyword arguments."}


def test_parse_docstring_no_description_after_colon():
    doc = '''"""
    Args:
        param_a:
        param_b: Description for B.
        param_c :
        :param param_d:
    """'''
    # With (.*) for description, empty descriptions should be captured.
    params = parse_docstring_params(doc)
    assert params == {"param_a": "", "param_b": "Description for B.", "param_c": "", "param_d": ""}
