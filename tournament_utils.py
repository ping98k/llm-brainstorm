import os
from litellm import completion

BUDGET_TOKENS_DEFAULT = int(os.getenv("THINKING_BUDGET_TOKENS", "1024"))


def _completion_kwargs(
    api_base: str | None,
    api_key: str | None,
    temperature: float | None,
) -> dict:
    """Build kwargs for litellm.completion from api settings."""
    kwargs: dict = {}
    if api_base:
        kwargs["api_base"] = api_base
    if api_key:
        kwargs["api_key"] = api_key
    if temperature is not None:
        kwargs["temperature"] = temperature
    return kwargs


def generate_players(
    instruction: str,
    n: int,
    model: str = "gpt-4o-mini",
    *,
    api_base: str | None = None,
    api_key: str | None = None,
    temperature: float | None = None,
    thinking: bool = False,
    budget_tokens: int = BUDGET_TOKENS_DEFAULT,
    return_usage: bool = False,
) -> list[str] | tuple[list[str], object]:
    """Request ``n`` completions for the instruction using the given model.

    When ``return_usage`` is ``True`` the ``usage`` object from the completion
    response is also returned.
    """
    messages = [{"role": "user", "content": instruction}]
    kwargs = _completion_kwargs(api_base, api_key, temperature)
    kwargs["thinking"] = {
        "type": "enabled" if thinking else "disabled",
        "budget_tokens": budget_tokens if thinking else 0,
    }
    response = completion(
        model=model,
        messages=messages,
        n=n,
        **kwargs,
    )
    players = [c.message.content.strip() for c in response.choices]
    if return_usage:
        return players, getattr(response, "usage", None)
    return players


def prompt_score(
    instruction: str,
    criteria_list: list[str],
    criteria_block: str,
    player: str,
    model: str = "gpt-4o-mini",
    *,
    api_base: str | None = None,
    api_key: str | None = None,
    temperature: float | None = None,
    include_instruction: bool = True,
    thinking: bool = False,
    budget_tokens: int = BUDGET_TOKENS_DEFAULT,
    return_usage: bool = False,
) -> str | tuple[str, object]:
    """Return a JSON score string evaluating `player` on the criteria."""
    example_scores = ", ".join(["1-10"] * len(criteria_list)) or "1-10"
    prompt = f"""Evaluate the output below on the following criteria:
{criteria_block}

Return JSON exactly like: {{"scores": [{example_scores}]}}."""
    if include_instruction:
        prompt += f"\n\nInstruction:\n{instruction}"
    prompt += f"\n\nOutput:\n{player}"
    kwargs = _completion_kwargs(api_base, api_key, temperature)
    kwargs["thinking"] = {
        "type": "enabled" if thinking else "disabled",
        "budget_tokens": budget_tokens if thinking else 0,
    }
    response = completion(
        model=model,
        messages=[{"role": "system", "content": prompt}],
        **kwargs,
    )
    text = response.choices[0].message.content.strip()
    if return_usage:
        return text, getattr(response, "usage", None)
    return text


def prompt_pairwise(
    instruction: str,
    criteria_block: str,
    a: str,
    b: str,
    model: str = "gpt-4o-mini",
    *,
    api_base: str | None = None,
    api_key: str | None = None,
    temperature: float | None = None,
    include_instruction: bool = True,
    thinking: bool = False,
    budget_tokens: int = BUDGET_TOKENS_DEFAULT,
    return_usage: bool = False,
) -> str | tuple[str, object]:
    """Return which player wins in JSON using the given criteria."""
    prompt = f"""Compare the two players below using:
{criteria_block}

Return ONLY JSON {{"winner": "A"}} or {{"winner": "B"}}."""
    if include_instruction:
        prompt += f"\n\nInstruction:\n{instruction}"
    prompt += f"\n\nPlayers:\n<A>{a}</A>\n<B>{b}</B>"
    kwargs = _completion_kwargs(api_base, api_key, temperature)
    kwargs["thinking"] = {
        "type": "enabled" if thinking else "disabled",
        "budget_tokens": budget_tokens if thinking else 0,
    }
    response = completion(
        model=model,
        messages=[{"role": "system", "content": prompt}],
        **kwargs,
    )
    text = response.choices[0].message.content.strip()
    if return_usage:
        return text, getattr(response, "usage", None)
    return text
