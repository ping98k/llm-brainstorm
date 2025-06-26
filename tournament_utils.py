from litellm import completion


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
):
    """Request `n` completions for the instruction using the given model."""
    response = completion(
        model=model,
        messages=[{"role": "user", "content": instruction}],
        n=n,
        **_completion_kwargs(api_base, api_key),
    )
    return [c.message.content.strip() for c in response.choices]


def prompt_score(
    instruction: str,
    criteria_list: list[str],
    criteria_block: str,
    player: str,
    model: str = "gpt-4o-mini",
    *,
    api_base: str | None = None,
    api_key: str | None = None,
) -> str:
    """Return a JSON score string evaluating `player` on the criteria."""
    example_scores = ", ".join(["1-10"] * len(criteria_list)) or "1-10"
    prompt = f"""Evaluate the output below on the following criteria:
{criteria_block}

Return JSON exactly like: {{"scores": [{example_scores}]}}.

Instruction:
{instruction}

Output:
{player}"""
    response = completion(
        model=model,
        messages=[{"role": "system", "content": prompt}],
        **_completion_kwargs(api_base, api_key),
    )
    return response.choices[0].message.content.strip()


def prompt_pairwise(
    instruction: str,
    criteria_block: str,
    a: str,
    b: str,
    model: str = "gpt-4o-mini",
    *,
    api_base: str | None = None,
    api_key: str | None = None,
) -> str:
    """Return which player wins in JSON using the given criteria."""
    prompt = f"""Compare the two players below using:
{criteria_block}

Return ONLY JSON {{"winner": "A"}} or {{"winner": "B"}}.

Instruction:
{instruction}

Players:
<A>{a}</A>
<B>{b}</B>"""
    response = completion(
        model=model,
        messages=[{"role": "system", "content": prompt}],
        **_completion_kwargs(api_base, api_key),
    )
    return response.choices[0].message.content.strip()
