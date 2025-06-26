from litellm import completion


def generate_players(instruction: str, n: int, model: str = "gpt-4o-mini"):
    """Request `n` completions for the instruction using the given model."""
    response = completion(
        model=model,
        messages=[{"role": "user", "content": instruction}],
        n=n,
    )
    return [c.message.content.strip() for c in response.choices]


def prompt_score(
    instruction: str,
    criteria_list: list[str],
    criteria_block: str,
    player: str,
    model: str = "gpt-4o-mini",
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
    response = completion(model=model, messages=[{"role": "system", "content": prompt}])
    return response.choices[0].message.content.strip()


def prompt_play(instruction: str, criteria_block: str, a: str, b: str, model: str = "gpt-4o-mini") -> str:
    """Return which player wins in JSON using the given criteria."""
    prompt = f"""Compare the two players below using:
{criteria_block}

Return ONLY JSON {{"winner": "A"}} or {{"winner": "B"}}.

Instruction:
{instruction}

Players:
<A>{a}</A>
<B>{b}</B>"""
    response = completion(model=model, messages=[{"role": "system", "content": prompt}])
    return response.choices[0].message.content.strip()
