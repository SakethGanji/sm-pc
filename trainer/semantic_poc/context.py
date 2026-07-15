"""§6.2 context builder. Pure function shared (conceptually) with the Go server;
the exact string format is part of the model contract and covered by parity fixtures."""


def build_c1(previous_agent_utterance: str, current_transcript: str,
             agent_tag: str = "[AGENT]", customer_tag: str = "[CUSTOMER]") -> str:
    return f"{agent_tag} {previous_agent_utterance}\n{customer_tag} {current_transcript}"
