"""Module docstring: describes the module, not a class, so the demo ignores it."""
from dataclasses import dataclass, field


@dataclass
class DataLoader:
    """Loads training examples from a CSV file in mini-batches.

    The file is read lazily, so memory usage does not depend on the file size.

    Attributes:
        path: location of the CSV file.
        batch_size: number of rows returned by each call to next_batch.
        shuffle: whether the rows are shuffled at the start of every epoch.

    Example:
        >>> loader = DataLoader("train.csv", batch_size=32)
        >>> batch = loader.next_batch()

    Note:
        Shuffling currently loads every row index into memory, which should be replaced by reservoir sampling.
    """

    path: str
    batch_size: int = 32
    shuffle: bool = True
    _offset: int = field(default=0, repr=False)

    def next_batch(self):
        """Return the next batch of rows."""
        return []

    class Config:
        """Default settings shared by every loader.

        Values can be overridden per instance.
        """

        encoding = "utf-8"
