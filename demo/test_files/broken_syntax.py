class Tokenizer:
    """Split source code into tokens.

    Edge case for the demo: this file has a syntax error below, so the
    program must report the error instead of crashing.
    """

    def tokenize(self, text):
        return text.split()