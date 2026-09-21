"""The distillation pair set, and the passes that build it.

The pair set is the corpus the student is trained to imitate: rows that name
two recipes a chooser decided belong together, the writing prompt designed from
them, and the teacher's answer to it. This package holds the tooling behind
those files. The pool pass ranks every recipe's nearest neighbours and marks the
near copies the chooser may not be offered; the chooser reads that pool and
writes the pairs; the design pass writes one request per pair; the answer pass
puts each request to the teacher and keeps the recipe that comes back.

The re-exports below are the surface a caller outside the package should use.
The pool, the chooser and the answer pass are reached through their own module
instead, so a caller that wants the manifest helpers and nothing else does not
pay for the encoder the pool pass reads the corpus with. The selection loaders
are reached through ``.selection`` by the passes that need them, the same way
the drafter packages are entered.
"""

from .design import design_prompts
from .manifest import file_checksum, write_manifest

__all__ = ["design_prompts", "file_checksum", "write_manifest"]
