from dotenv import load_dotenv
load_dotenv()
import os, json, gradio as gr
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from litellm import completion

NUM_TOP_PICKS_DEFAULT = int(os.getenv("NUM_TOP_PICKS", 5))
POOL_SIZE_DEFAULT = int(os.getenv("POOL_SIZE", 20))
MAX_WORKERS_DEFAULT = int(os.getenv("MAX_WORKERS", 10))
NUM_GENERATIONS_DEFAULT = int(os.getenv("NUM_GENERATIONS", 100))

def generate_players(instruction, n):
    response = completion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": instruction}],
        n=n
    )
    return [c.message.content.strip() for c in response.choices]

def run_tournament(instruction_input, criteria_input, n_gen,
                   num_top_picks, pool_size, max_workers):
    instruction = instruction_input.strip()
    criteria_list = [c.strip() for c in criteria_input.split(",") if c.strip()] or [
        "Factuality", "Instruction Following", "Precision"
    ]
    n_gen = int(n_gen)
    num_top_picks = int(num_top_picks)
    pool_size = int(pool_size)
    max_workers = int(max_workers)

    def criteria_block():
        return "\n".join(f"{i + 1}) {c}" for i, c in enumerate(criteria_list))

    def prompt_score(player):
        prompt = f"""Evaluate the output below on the following criteria:
{criteria_block()}

Return JSON exactly like: {{"score": [{', '.join(['1-10'] * len(criteria_list))}]}}.

Instruction:
{instruction}

Output:
{player}"""
        response = completion(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": prompt}]
        )
        return response.choices[0].message.content.strip()

    def score(player):
        try:
            data = json.loads(prompt_score(player))
        except json.JSONDecodeError:
            data = eval(prompt_score(player))
        lst = data.get("score", data.get("scores", []))
        return sum(lst) / len(lst) if lst else 0.0

    def prompt_play(a, b):
        prompt = f"""Compare the two players below using:
{criteria_block()}

Return ONLY JSON {{"winner": "A"}} or {{"winner": "B"}}.

Instruction:
{instruction}

Players:
<A>{a}</A>
<B>{b}</B>"""
        response = completion(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": prompt}]
        )
        return response.choices[0].message.content.strip()

    def play(a, b):
        try:
            winner_label = json.loads(prompt_play(a, b))["winner"]
        except json.JSONDecodeError:
            winner_label = eval(prompt_play(a, b)).get("winner", "A")
        return a if winner_label == "A" else b

    def precompute_scores(players, executor):
        futures = {executor.submit(score, p): p for p in players}
        scores = {}
        for fut in tqdm(as_completed(futures), total=len(futures)):
            scores[futures[fut]] = fut.result()
        return scores

    def tournament_round(pairs, executor):
        futures = {executor.submit(play, a, b): (a, b) for a, b in pairs}
        results = []
        for fut in tqdm(as_completed(futures), total=len(futures)):
            a, b = futures[fut]
            winner = fut.result()
            loser = b if winner == a else a
            results.append((winner, loser))
        return results

    def tournament(players, executor):
        lost_to = {}
        current = players[:]
        while len(current) > 1:
            pairs = [(current[i], current[i + 1]) for i in range(0, len(current) - 1, 2)]
            for w, l in tournament_round(pairs, executor):
                lost_to[l] = w
            current = [w for w, _ in tournament_round(pairs, executor)]
            if len(players) % 2 == 1:
                current.append(players[-1])
        return current[0], lost_to

    def get_candidates(champion, lost_to):
        return [p for p, o in lost_to.items() if o == champion] + [champion]

    def playoff(candidates, executor):
        wins = {p: 0 for p in candidates}
        pairs = [(candidates[i], candidates[j])
                 for i in range(len(candidates))
                 for j in range(i + 1, len(candidates))]
        futures = {executor.submit(play, a, b): (a, b) for a, b in pairs}
        for fut in tqdm(as_completed(futures), total=len(futures)):
            wins[fut.result()] += 1
        return sorted(candidates, key=lambda p: wins[p], reverse=True)

    def get_top(players, executor):
        champion, lost_to = tournament(players, executor)
        runner_up = lost_to.get(champion)
        finalists = [champion] + ([runner_up] if runner_up else [])
        semifinalists = [p for p, o in lost_to.items()
                         if o in finalists and p not in finalists]
        candidates = set(finalists + semifinalists +
                         get_candidates(champion, lost_to))
        return playoff(list(candidates), executor)[:num_top_picks]

    all_players = generate_players(instruction, n_gen)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        scores = precompute_scores(all_players, ex)
        top_players = sorted(all_players, key=scores.get, reverse=True)[:pool_size]
        top_k = get_top(top_players, ex)
    return ", ".join(top_k)

demo = gr.Interface(
    fn=run_tournament,
    inputs=[
        gr.Textbox(lines=2, label="Instruction"),
        gr.Textbox(lines=1, label="Criteria (comma separated)"),
        gr.Number(value=NUM_GENERATIONS_DEFAULT, label="Number of Generations"),
        gr.Number(value=NUM_TOP_PICKS_DEFAULT, label="Top Picks (k)"),
        gr.Number(value=POOL_SIZE_DEFAULT, label="Filter Size"),
        gr.Number(value=MAX_WORKERS_DEFAULT, label="Max Workers")
    ],
    outputs=gr.Textbox(label="Top picks")
)

if __name__ == "__main__":
    demo.launch()
