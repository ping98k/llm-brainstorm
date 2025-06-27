from litellm import completion


def _completion_with_retry(*args, retries: int = 5, **kwargs):
    """Call ``completion`` with retry logic."""
    for _ in range(retries):
        try:
            return completion(*args, **kwargs)
        except Exception:
            pass
    return None


def _completion_kwargs(api_base: str | None, api_key: str | None) -> dict:
    """Build kwargs for litellm.completion from api settings."""
    kwargs: dict = {}
    if api_base:
        kwargs["api_base"] = api_base
    if api_key:
        kwargs["api_key"] = api_key
    return kwargs


def generate_players(
    instruction: str,
    n: int,
    model: str = "gpt-4o-mini",
    *,
    api_base: str | None = None,
    api_key: str | None = None,
    return_usage: bool = False,
) -> list[str] | tuple[list[str], object]:
    """Request ``n`` completions for the instruction using the given model.

    When ``return_usage`` is ``True`` the ``usage`` object from the completion
    response is also returned.
    """
    players = []
    usage_data = {"prompt_tokens": 0, "completion_tokens": 0}
    for _ in range(n):
        resp = _completion_with_retry(
            model=model,
            messages=[{"role": "user", "content": instruction}],
            n=1,
            **_completion_kwargs(api_base, api_key),
        )
        if resp is None:
            continue
        players.append(resp.choices[0].message.content.strip())
        u = getattr(resp, "usage", None)
        if u:
            pt = getattr(u, "prompt_tokens", None)
            if pt is None and isinstance(u, dict):
                pt = u.get("prompt_tokens", 0)
            ct = getattr(u, "completion_tokens", None)
            if ct is None and isinstance(u, dict):
                ct = u.get("completion_tokens", 0)
            usage_data["prompt_tokens"] += pt or 0
            usage_data["completion_tokens"] += ct or 0
    if return_usage:
        return players, usage_data
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
    return_usage: bool = False,
) -> str | tuple[str, object]:
    """Return a JSON score string evaluating `player` on the criteria."""
    example_scores = ", ".join(["1-10"] * len(criteria_list)) or "1-10"
    prompt = f"""Evaluate the output below on the following criteria:
{criteria_block}

Return JSON exactly like: {{"scores": [{example_scores}]}}.

Instruction:
{instruction}

Output:
{player}"""
    response = _completion_with_retry(
        model=model,
        messages=[{"role": "system", "content": prompt}],
        **_completion_kwargs(api_base, api_key),
    )
    if response is None:
        text = "{}"
        usage = None
    else:
        text = response.choices[0].message.content.strip()
        usage = getattr(response, "usage", None)
    if return_usage:
        return text, usage
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
    return_usage: bool = False,
) -> str | tuple[str, object]:
    """Return which player wins in JSON using the given criteria."""
    prompt = f"""Compare the two players below using:
{criteria_block}

Return ONLY JSON {{"winner": "A"}} or {{"winner": "B"}}.

Instruction:
{instruction}

Players:
<A>{a}</A>
<B>{b}</B>"""
    response = _completion_with_retry(
        model=model,
        messages=[{"role": "system", "content": prompt}],
        **_completion_kwargs(api_base, api_key),
    )
    if response is None:
        text = "{}"
        usage = None
    else:
        text = response.choices[0].message.content.strip()
        usage = getattr(response, "usage", None)
    if return_usage:
        return text, usage
    return text
