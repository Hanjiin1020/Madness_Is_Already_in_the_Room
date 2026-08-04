

_COT_BASE = (
    "You are a vision-language agent answering a question about the given image. "
    "Look at the image carefully and reason step by step before answering."
)

VANILLA_SYSTEM = _COT_BASE

ANSWER_FORMAT = {
    "mcq":  "This is a multiple-choice question. Your final answer must be the single "
            "letter of the best option (e.g. 'A').",
    "open": "This is an open-ended question. Answer with a short, direct phrase, number, "
            "or expression.",
    "binary": "This is a yes/no question. Your final answer must be exactly 'Yes' or 'No'.",
}

OUTPUT_SCHEMA = (
    "First, think step by step and explain your reasoning based on the image. "
    "Then, on the last line, write exactly 'Answer: ' followed by your final answer."
)

DEBATE_INSTRUCTION_SELFOTHER = (
    "Your own answer from the previous round is shown under [YOUR PREVIOUS RESPONSE]; "
    "the other agents' answers are shown under [OTHER AGENTS' PREVIOUS RESPONSES]. "
    "Review the responses and reconsider the question using the image. "
    "Then provide your best answer based on the available evidence."
)

SELF_REFLECT_INSTRUCTION = (
    "Your own answer from the previous round is shown under [YOUR PREVIOUS RESPONSE]. "
    "Review the responses and reconsider the question using the image. "
    "Then provide your best answer based on the available evidence."
)

DEBATE_INSTRUCTION = (
    "Below are the responses from other agents in the previous round. "
    "Carefully review the previous agents' reasoning."
)
