"""Table 1 of the paper: the only instruction DeepSeek-R1-Zero gets. It fixes the shape of the answer
(<think> ... </think> <answer> ... </answer>) and says nothing about how to reason."""

TEMPLATE = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
    "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, "
    "respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. "
    "User: {prompt}. Assistant:"
)
# Our one addition: the prompt ends with the opening tag, so a 0.5B model starts inside <think> (see the text).
PREFILL = ' <think>'


def make_prompt(question):
    """Put one question into the template. The text is fed to the model as is: no chat template, no examples."""
    return TEMPLATE.format(prompt=question.strip().rstrip('.')) + PREFILL


def full_completion(generated):
    """What the rewards see: the prefilled tag plus the generated text."""
    return '<think>' + generated
