"""
Extract class comments from .java / .py / .st (Pharo) files and normalize them to the exact sentence
format of the NLBSE'23 dataset (data/raw/sentences.csv), ready to be fed to the model in train.py.

Quick check (no torch needed):
    python src/preprocess_comments.py demo/examples/UserCache.java
"""
import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_TEMPLATE = "[{lang}] {class_id} | {sentence}"

EXT_TO_LANG = {".java": "java", ".py": "python", ".st": "pharo"}


# ===================== Data =====================

@dataclass
class CommentSentence:
    lang: str          # "java" | "python" | "pharo"
    class_id: str      # Java: file name (Foo.java); Python/Pharo: class name (same as the class_id column)
    class_name: str
    line: int          # line of the class declaration in the file
    raw: str           # original sentence in the comment
    sentence: str      # normalized sentence (same as the sentence column)


def detect_lang(filename: str):
    return EXT_TO_LANG.get(Path(filename).suffix.lower())


# ===================== Normalization =====================
# Rules derived from sentences.csv:
#  - everything lowercase
#  - keep only letters, digits and . , @ # ^ % & ? ! ; other characters (brackets, quotes, _, -, /, <, >...) -> space
#    => "{@link Foo#bar(String)}" -> "@link foo#bar string", "<li>" -> "li", "isn't" -> "isn t"
#  - a dot inside a word -> "^"  (e.g. "test.proto" -> "test^proto")
#  - each comment line is further split at sentence-ending dots

_INWORD_DOT = re.compile(r"(?<=[a-z0-9])\.(?=[a-z0-9])")
_SENT_SPLIT = re.compile(r"(?<=\.)\s+")
_DISALLOWED = re.compile(r"[^a-z0-9\s.,@#^%&?!]")
_SPACES = re.compile(r"\s+")


def normalize_sentence(text: str) -> str:
    s = text.lower()
    s = _INWORD_DOT.sub("^", s)
    s = _DISALLOWED.sub(" ", s)
    return _SPACES.sub(" ", s).strip()


def split_sentences(comment_text: str):
    """Return a list of (original sentence, normalized sentence); drops lines without letters/digits."""
    out = []
    for line in comment_text.splitlines():
        for part in _SENT_SPLIT.split(line.strip()):
            norm = normalize_sentence(part)
            if any(c.isalnum() for c in norm):
                out.append((part.strip(), norm))
    return out


# ===================== Java =====================

# A Javadoc block /** ... */ right before a class/interface/enum/record declaration
# (annotations and modifiers allowed in between). The body must not contain "*/" so that
# consecutive comments are not swallowed together.
_JAVA_CLASS_DOC = re.compile(
    r"/\*\*((?:(?!\*/).)*)\*/"
    r"\s*(?:@[\w.]+(?:\([^)]*\))?\s*)*"
    r"(?:(?:public|protected|private|abstract|static|final|sealed|non-sealed|strictfp)\s+)*"
    r"(?:class|interface|enum|record|@interface)\s+(\w+)",
    re.DOTALL,
)
_JAVADOC_LINE_PREFIX = re.compile(r"^\s*\*?[ \t]?")


def _strip_javadoc(body: str) -> str:
    return "\n".join(_JAVADOC_LINE_PREFIX.sub("", ln) for ln in body.splitlines())


def extract_java(source: str, filename: str):
    items = []
    for m in _JAVA_CLASS_DOC.finditer(source):
        class_name = m.group(2)
        # In the dataset, the Java class_id is the file name (also for inner classes)
        class_id = Path(filename).name if filename.endswith(".java") else f"{class_name}.java"
        line = source.count("\n", 0, m.start(2)) + 1
        for raw, sent in split_sentences(_strip_javadoc(m.group(1))):
            items.append(CommentSentence("java", class_id, class_name, line, raw, sent))
    return items


# ===================== Python =====================

def extract_python(source: str):
    tree = ast.parse(source)  # SyntaxError is propagated to the caller
    classes = sorted(
        (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)),
        key=lambda n: n.lineno,
    )
    items = []
    for node in classes:
        doc = ast.get_docstring(node)
        if not doc:
            continue
        for raw, sent in split_sentences(doc):
            items.append(CommentSentence("python", node.name, node.name, node.lineno, raw, sent))
    return items


# ===================== Pharo =====================
# Two common .st file formats:
#  - Tonel (Foo.class.st, used in git repos via Iceberg): a "..." comment right before
#    "Class { #name : #Foo ..." ; a " inside the comment is escaped as ""
#  - File Out (Foo.st, chunk format): "!Foo commentStamp: '...' prior: 0!<comment>!" ;
#    a ! inside the comment is escaped as !!

_TONEL_CLASS_DOC = re.compile(
    r'"((?:[^"]|"")*)"\s*(?:Class|Trait)\s*\{\s*#name\s*:\s*[#\']*(\w+)',
    re.DOTALL,
)
_FILEOUT_CLASS_DOC = re.compile(
    r"!(\w+)\s+commentStamp:\s*'(?:[^']|'')*'\s+prior:\s*\d+!\s*((?:[^!]|!!)*)!",
    re.DOTALL,
)


def extract_pharo(source: str):
    found = []  # (position of the class name, class name, comment)
    for m in _TONEL_CLASS_DOC.finditer(source):
        found.append((m.start(2), m.group(2), m.group(1).replace('""', '"')))
    for m in _FILEOUT_CLASS_DOC.finditer(source):
        found.append((m.start(1), m.group(1), m.group(2).replace("!!", "!")))

    items = []
    for pos, class_name, comment in sorted(found):
        line = source.count("\n", 0, pos) + 1
        for raw, sent in split_sentences(comment):
            items.append(CommentSentence("pharo", class_name, class_name, line, raw, sent))
    return items


# ===================== Public API =====================

def extract_comments(source: str, filename: str, lang: str = None):
    lang = lang or detect_lang(filename)
    if lang == "java":
        return extract_java(source, filename)
    if lang == "python":
        return extract_python(source)
    if lang == "pharo":
        return extract_pharo(source)
    raise ValueError(f"Unsupported file '{filename}' (only .java, .py and .st)")


def to_model_input(item: CommentSentence, template: str = DEFAULT_TEMPLATE,
                   include_lang: bool = True, include_class_id: bool = True) -> str:
    """Same as CodeCommentDataset._build_text in train.py."""
    if include_lang or include_class_id:
        return template.format(lang=item.lang.upper(), class_id=item.class_id, sentence=item.sentence)
    return item.sentence


def main():
    if len(sys.argv) != 2:
        print("Usage: python src/preprocess_comments.py <file.java|file.py|file.st>")
        sys.exit(1)
    path = Path(sys.argv[1])
    try:
        items = extract_comments(path.read_text(encoding="utf-8", errors="replace"), path.name)
    except SyntaxError as e:
        print(f"Python syntax error in {path.name}, line {e.lineno}: {e.msg}")
        sys.exit(1)
    if not items:
        print("No class comment found.")
        return
    for it in items:
        print(f"L{it.line:<5} {to_model_input(it)}")


if __name__ == "__main__":
    main()
