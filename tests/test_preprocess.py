"""Comment extraction and normalization (src/preprocess_comments.py). Needs no torch."""
import pytest

from preprocess_comments import (
    CommentSentence,
    detect_lang,
    extract_comments,
    normalize_sentence,
    split_sentences,
    to_model_input,
)


# ---------------------------------------------------------------- normalization

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("{@link Foo#bar(String)}", "@link foo#bar string"),   # brackets and parentheses become spaces
        ("<li>", "li"),                                         # HTML tags keep only their name
        ("isn't", "isn t"),                                      # apostrophe becomes a space
        ("see test.proto for details", "see test^proto for details"),  # dot inside a word -> ^
        ("Ends with a dot.", "ends with a dot."),                # a sentence-ending dot is kept
        ("snake_case and kebab-case", "snake case and kebab case"),
        ("  many    spaces\there ", "many spaces here"),
        ("Keeps @ # ^ % & ? ! and ,", "keeps @ # ^ % & ? ! and ,"),
    ],
)
def test_normalize_sentence(raw, expected):
    assert normalize_sentence(raw) == expected


def test_normalize_is_idempotent():
    s = normalize_sentence("Call {@link Foo#bar()} on a.b.c, then <b>stop</b>!")
    assert normalize_sentence(s) == s


def test_split_sentences_splits_at_dots_and_drops_empty_lines():
    text = "First sentence. Second one.\n\n----\nThird line without dot"
    assert [norm for _, norm in split_sentences(text)] == [
        "first sentence.",
        "second one.",
        "third line without dot",
    ]


def test_split_sentences_keeps_original_text():
    raw, norm = split_sentences("Uses {@link Map}.")[0]
    assert raw == "Uses {@link Map}."
    assert norm == "uses @link map ."


@pytest.mark.parametrize(
    "name, lang",
    [("Foo.java", "java"), ("foo.py", "python"), ("Foo.class.st", "pharo"), ("Foo.ST", "pharo"),
     ("foo.txt", None), ("Makefile", None)],
)
def test_detect_lang(name, lang):
    assert detect_lang(name) == lang


# ---------------------------------------------------------------- Java

JAVA_SOURCE = '''package demo;

/** File header, not a class comment. */
import java.util.Map;

/**
 * Caches users in memory.
 * Use {@link #get(String)} to read an entry.
 *
 * @author Jane Doe
 */
@Deprecated
public final class UserCache {

    /** Method comment: must be ignored. */
    public User get(String id) { return null; }

    /**
     * Inner helper.
     */
    static class Entry {}
}
'''


def test_java_extracts_class_javadoc_only():
    items = extract_comments(JAVA_SOURCE, "UserCache.java")
    assert [it.sentence for it in items] == [
        "caches users in memory.",
        "use @link #get string to read an entry.",
        "@author jane doe",
        "inner helper.",
    ]


def test_java_class_id_is_file_name_and_line_is_declaration():
    items = extract_comments(JAVA_SOURCE, "UserCache.java")
    assert {it.class_id for it in items} == {"UserCache.java"}   # the dataset uses the file name, inner classes too
    assert items[0].class_name == "UserCache"
    assert items[-1].class_name == "Entry"
    assert items[0].line == JAVA_SOURCE.splitlines().index("public final class UserCache {") + 1


@pytest.mark.parametrize("keyword", ["interface", "enum", "record", "abstract class", "@interface"])
def test_java_other_type_declarations(keyword):
    source = f"/** Short doc. */\npublic {keyword} Thing {{}}\n"
    items = extract_comments(source, "Thing.java")
    assert [it.sentence for it in items] == ["short doc."]


def test_java_without_class_comment():
    assert extract_comments("public class Plain { /** method */ void f() {} }", "Plain.java") == []


# ---------------------------------------------------------------- Python

PYTHON_SOURCE = '''"""Module docstring, not a class comment."""


class RateLimiter:
    """Limits calls per second.

    Args:
        rate: allowed calls per second.
    """

    def allow(self):
        """Method docstring: ignored."""

    class Window:
        """Sliding window."""


class NoDoc:
    pass
'''


def test_python_extracts_class_docstrings_in_order():
    items = extract_comments(PYTHON_SOURCE, "rate_limiter.py")
    assert [(it.class_id, it.sentence) for it in items] == [
        ("RateLimiter", "limits calls per second."),
        ("RateLimiter", "args"),
        ("RateLimiter", "rate allowed calls per second."),
        ("Window", "sliding window."),
    ]


def test_python_line_numbers():
    items = extract_comments(PYTHON_SOURCE, "rate_limiter.py")
    lines = PYTHON_SOURCE.splitlines()
    assert items[0].line == lines.index("class RateLimiter:") + 1
    assert items[-1].line == lines.index("    class Window:") + 1


def test_python_syntax_error_is_raised():
    with pytest.raises(SyntaxError):
        extract_comments("class Broken(:\n    pass\n", "broken.py")


# ---------------------------------------------------------------- Pharo

TONEL_SOURCE = '''"
I represent a queue.
I say ""hello"" to my collaborators.
"
Class {
\t#name : #OrderedQueue,
\t#superclass : #Object,
\t#category : #Demo
}
'''

FILEOUT_SOURCE = """Object subclass: #RetryPolicy
\tinstanceVariableNames: 'count'
\tclassVariableNames: ''
\tpackage: 'Demo'!
!RetryPolicy commentStamp: 'jd 1/1/2024 10:00' prior: 0!
I retry failed jobs!! Use me carefully.!
"""


def test_pharo_tonel_unescapes_double_quotes():
    items = extract_comments(TONEL_SOURCE, "OrderedQueue.class.st")
    assert [it.raw for it in items] == ["I represent a queue.", 'I say "hello" to my collaborators.']
    assert {it.class_id for it in items} == {"OrderedQueue"}


def test_pharo_tonel_trait():
    source = '"\nA comparable trait.\n"\nTrait {\n\t#name : #TComparable\n}\n'
    items = extract_comments(source, "TComparable.trait.st")
    assert [(it.class_id, it.sentence) for it in items] == [("TComparable", "a comparable trait.")]


def test_pharo_file_out_unescapes_exclamation_marks():
    items = extract_comments(FILEOUT_SOURCE, "RetryPolicy.st")
    assert [it.raw for it in items] == ["I retry failed jobs! Use me carefully."]
    assert items[0].class_id == "RetryPolicy"


# ---------------------------------------------------------------- public API

def test_unsupported_extension_raises():
    with pytest.raises(ValueError):
        extract_comments("x", "notes.txt")


def test_explicit_lang_overrides_extension():
    items = extract_comments('class A:\n    """Doc."""\n', "snippet.txt", lang="python")
    assert [it.sentence for it in items] == ["doc."]


def test_to_model_input_matches_training_template():
    item = CommentSentence("java", "Abfss.java", "Abfss", 1, "raw", "this impl delegates to the old filesystem")
    assert to_model_input(item) == "[JAVA] Abfss.java | this impl delegates to the old filesystem"
    assert to_model_input(item, include_lang=False, include_class_id=False) == item.sentence