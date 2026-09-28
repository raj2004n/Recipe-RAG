"""Fitting the student generator to the teacher's answers.

Distilling the teacher into the student is the training side of the pair set: the
rows the pair builders write are tokenised by the distillation split, and the
trainer here fits an adapter over them. The module holds the student's own side
of a row, which is the system prompt and the chat template its answers are
rendered with, together with the loop that trains on them.

The re-export below is the surface a caller outside the package should use. The
prompt constants, the renderer and the parity assertion are reached through
``.student`` by the passes that need them, such as the tokeniser that pins the
template before it writes a row.
"""

from .student import train_student

__all__ = ["train_student"]
