"""§6.2 context builder. The exact string format is part of the model
contract (train and serve must build it identically) and covered by fixtures."""


def build_c1(previous_agent_utterance: str, current_transcript: str,
             agent_tag: str = "[AGENT]", customer_tag: str = "[CUSTOMER]") -> str:
    return f"{agent_tag} {previous_agent_utterance}\n{customer_tag} {current_transcript}"
