"""Exception types for the known failure modes of the pipeline.

Each error carries a message that tells the user what went wrong and, where
possible, which command fixes it.
"""


class DocQAError(Exception):
    """Base class for all docqa errors."""


class CorpusNotFoundError(DocQAError):
    """Raw documentation or the processed chunk file is missing."""


class MalformedDataError(DocQAError):
    """A data file (chunks JSONL, questions JSON) is malformed or inconsistent."""


class EmptyQuestionError(DocQAError):
    """The user asked an empty or whitespace-only question."""


class ModelUnavailableError(DocQAError):
    """A Transformer model could not be loaded (missing package, no network, bad id)."""


class IndexNotFoundError(DocQAError):
    """A retrieval index/embedding cache is required but missing."""
