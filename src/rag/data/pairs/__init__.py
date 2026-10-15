"""The distillation pair set, and the passes that build it.

The pair set is the corpus the student is trained to imitate: rows that name
two recipes a chooser decided belong together, the writing prompt designed from
them, and the teacher's answer to it. This package holds the tooling behind
those files. The pool pass ranks every recipe's nearest neighbours and marks the
near copies the chooser may not be offered; the chooser reads that pool and
writes the pairs; the design pass writes one request per pair; the answer pass
puts each request to the teacher and keeps the recipe that comes back. Beside
those sits the labelling pass, which reads the pair set's titles and marks each
recipe savoury or sweet, and the reviewed calls that settle the titles the pass
abstained on.

Only the manifest write and the pointer write are re-exported. The pool, the
chooser, the design pass, the answer pass and the labelling pass are each reached
through their own module, so a caller that wants the manifest helper and nothing
else does not pay for what a pass loads on the way in. The selection loaders are
reached through ``.selection`` by the passes that need them, the same way the
drafter packages are entered.
"""

from .manifest import add_to_dvc, write_manifest

__all__ = ["write_manifest", "add_to_dvc"]
