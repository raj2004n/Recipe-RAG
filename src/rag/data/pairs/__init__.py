"""The distillation pair set, and the passes that build it.

The pair set is the corpus the student is trained to imitate: rows that name
two recipes a chooser decided belong together, the writing prompt designed from
them, and the teacher's answer to it. This package holds the tooling behind
those files. The design pass writes one prompt per pair; the manifest helpers
record what each pass read and wrote and are shared by the passes outside this
package as well.

The re-exports below are the surface a caller outside the package should use.
The selection loaders are reached through ``.selection`` by the passes that
need them, the same way the drafter packages are entered.
"""

from .design import design_prompts
from .manifest import file_checksum, write_manifest

__all__ = ["design_prompts", "file_checksum", "write_manifest"]
