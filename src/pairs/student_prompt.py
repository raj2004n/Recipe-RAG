from src.domain.source_block import request_message

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
