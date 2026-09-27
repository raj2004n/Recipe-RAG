"""The prompt a student is served with, and the one it was trained on.

A student is served the same two things a training row gave it: a one line system prompt, and a
user turn carrying the retrieved recipes and the request. The user turn is rendered by the same
function the training rows were built from, so a served turn and a trained turn are the same text
rather than two renderings that have to be kept in step by hand. The system line is a config value
rather than a constant, so the words the student opens with can be read from the copy a snapshot
seals and from the live configs a training pass reads, and the tokenised rows record the value they
were built with.

The system prompt is a single line because the shape and the voice of an answer are carried by the
training targets rather than asked for in words. Fine-tuning is what moves style into the weights,
so instructions about style are the part of a prompt a student needs least.
"""

from src.rag.data.source_block import request_message

# The persona whose system line a student opens with.
STUDENT_PERSONA = "student"


def student_system(template: dict) -> str:
    """The student's system line, read from a prompts config's template block.

    A service reads the copy its snapshot sealed and a training or tokenising pass reads the live
    configs, so both sides name the one key. A line edited on one side is then a line the rows
    already built no longer agree with, which the training pass checks before it fits anything.
    """
    try:
        return template["personas"][STUDENT_PERSONA]["generate"]["system"]
    except (KeyError, TypeError) as exc:
        raise KeyError(
            f"prompts_config is missing personas.{STUDENT_PERSONA}.generate.system; "
            f"check configs/prompts_config.yaml"
        ) from exc


def student_prompt(system: str, blocks: list[str], request: str) -> tuple[str, str]:
    """The (system, user) pair a student answers from, given its system line and rendered blocks.

    The blocks are rendered by the caller so a serving path can hold a token budget over them and
    hand on only what fits, and the turn they make up is assembled by the same function a training
    row uses.
    """
    return system, request_message(blocks, request)
