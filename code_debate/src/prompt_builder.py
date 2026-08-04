

import sys
from pathlib import Path

# allow `from prompts.role_prompts import ...` when run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prompts.role_prompts import (
    ANSWER_FORMAT,
    OUTPUT_SCHEMA,
    DEBATE_INSTRUCTION,
    DEBATE_INSTRUCTION_SELFOTHER,
    SELF_REFLECT_INSTRUCTION,
    VANILLA_SYSTEM,
)

_SELFOTHER_MARKER = "[YOUR PREVIOUS RESPONSE]"
_OTHER_MARKER = "[OTHER AGENTS' PREVIOUS RESPONSES]"

def _debate_block(prev_block: str) -> str:

    if prev_block.startswith(_SELFOTHER_MARKER):
        instr = DEBATE_INSTRUCTION_SELFOTHER if _OTHER_MARKER in prev_block else SELF_REFLECT_INSTRUCTION
        return f"{instr}\n\n{prev_block}"
    return (
        f"{DEBATE_INSTRUCTION}\n\n"
        f"## Previous-Round Responses from Other Agents\n{prev_block}"
    )

def _answer_format(qtype: str) -> str:
    if qtype not in ANSWER_FORMAT:
        raise ValueError(f"Unknown qtype: {qtype!r}; must be one of {sorted(ANSWER_FORMAT)}")
    return ANSWER_FORMAT[qtype]

def build_vanilla_prompt(question: str, prev_block: str | None = None,
                         round_: int = 0, qtype: str = "mcq") -> str:

    parts: list[str] = [VANILLA_SYSTEM, f"## Question\n{question}"]
    if round_ >= 1 and prev_block:
        parts.append(_debate_block(prev_block))
    parts.append(_answer_format(qtype))
    parts.append(OUTPUT_SCHEMA)
    return "\n\n".join(parts)

if __name__ == "__main__":
    q = ("What category of the plant is shown?\n"
         "A. orchid\nB. marigold\nC. lily\nD. amaryllis")
    prev_debate = ("[YOUR PREVIOUS RESPONSE]\n...my prev...\n\n"
                   "[OTHER AGENTS' PREVIOUS RESPONSES]\n--- Agent: x ---\n...prev...")
    prev_self = "[YOUR PREVIOUS RESPONSE]\n...my prev..."
    print(f"\n{'='*60}\nROUND 0 (mcq)\n{'='*60}")
    print(build_vanilla_prompt(q, round_=0, qtype="mcq"))
    print(f"\n{'='*60}\nDEBATE — ROUND 1 (mcq)\n{'='*60}")
    print(build_vanilla_prompt(q, prev_block=prev_debate, round_=1, qtype="mcq"))
    print(f"\n{'='*60}\nSELF_REFLECT — ROUND 1 (mcq)\n{'='*60}")
    print(build_vanilla_prompt(q, prev_block=prev_self, round_=1, qtype="mcq"))
