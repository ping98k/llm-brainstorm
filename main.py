from dotenv import load_dotenv
load_dotenv("./local.env")
import os, json, re, ast, gradio as gr
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import matplotlib.pyplot as plt
from tournament_utils import generate_players, prompt_score, prompt_pairwise

NUM_TOP_PICKS_DEFAULT = int(os.getenv("NUM_TOP_PICKS", 3))
POOL_SIZE_DEFAULT = int(os.getenv("POOL_SIZE", 5))
MAX_WORKERS_DEFAULT = int(os.getenv("MAX_WORKERS", 100))
NUM_GENERATIONS_DEFAULT = int(os.getenv("NUM_GENERATIONS", 10))
API_BASE_DEFAULT = os.getenv("OPENAI_API_BASE", "")
API_TOKEN_DEFAULT = os.getenv("OPENAI_API_KEY", "")
SCORE_FILTER_DEFAULT = os.getenv("ENABLE_SCORE_FILTER", "true").lower() == "true"
PAIRWISE_FILTER_DEFAULT = os.getenv("ENABLE_PAIRWISE_FILTER", "true").lower() == "true"
GENERATE_MODEL_DEFAULT = os.getenv("GENERATE_MODEL", "gpt-4o-mini")
SCORE_MODEL_DEFAULT = os.getenv("SCORE_MODEL", "gpt-4o-mini")
PAIRWISE_MODEL_DEFAULT = os.getenv("PAIRWISE_MODEL", "gpt-4o-mini")
CRITERIA_DEFAULT = "Factuality,Instruction Following,Precision"
def _clean_json(txt):
    txt = re.sub(r"^```.*?\n|```$", "", txt, flags=re.DOTALL).strip()
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        return ast.literal_eval(txt)

def run_tournament(
    api_base,
    api_token,
    generate_model,
    score_model,
    pairwise_model,
    instruction_input,
    criteria_input,
    n_gen,
    pool_size,
    num_top_picks,
    max_workers,
    enable_score_filter,
    enable_pairwise_filter,
):
    instruction = instruction_input.strip()
    criteria_list = [c.strip() for c in criteria_input.split(",") if c.strip()] or ["Factuality", "Instruction Following", "Precision"]
    n_gen = int(n_gen)
    num_top_picks = int(num_top_picks)
    pool_size = int(pool_size)
    max_workers = int(max_workers)
    if not api_base:
        api_base = API_BASE_DEFAULT
    if not api_token:
        api_token = API_TOKEN_DEFAULT
    if not generate_model:
        generate_model = GENERATE_MODEL_DEFAULT
    if not score_model:
        score_model = SCORE_MODEL_DEFAULT
    if not pairwise_model:
        pairwise_model = PAIRWISE_MODEL_DEFAULT
    enable_score_filter = bool(enable_score_filter)
    enable_pairwise_filter = bool(enable_pairwise_filter)
    process_log = []
    hist_fig = None
    top_picks_str = ""
    prompt_tokens = 0
    completion_tokens = 0
    score_outputs: list[str] = []
    pairwise_outputs: list[str] = []

    def add_usage(usage):
        nonlocal prompt_tokens, completion_tokens
        if not usage:
            return
        pt = getattr(usage, "prompt_tokens", None)
        if pt is None and isinstance(usage, dict):
            pt = usage.get("prompt_tokens")
        ct = getattr(usage, "completion_tokens", None)
        if ct is None and isinstance(usage, dict):
            ct = usage.get("completion_tokens")
        if pt:
            prompt_tokens += pt
        if ct:
            completion_tokens += ct

    def usage_str():
        return (
            f"Prompt tokens: {prompt_tokens}\n"
            f"Completion tokens: {completion_tokens}\n"
            f"Total tokens: {prompt_tokens + completion_tokens}"
        )

    def log_completion(prefix: str, text: str):
        disp = text.replace("\n", " ")
        if len(disp) > 100:
            disp = disp[:100] + "…"
        return log(f"{prefix}{disp}")
    def log(msg):
        process_log.append(msg)
        tqdm.write(msg)
        yield "\n".join(process_log), hist_fig, top_picks_str, usage_str()
    yield from log("Generating players …")
    all_players, usage = generate_players(
        instruction,
        n_gen,
        model=generate_model,
        api_base=api_base,
        api_key=api_token,
        return_usage=True,
    )
    add_usage(usage)
    yield from log(f"{len(all_players)} players generated")
    for i, p in enumerate(all_players, 1):
        yield from log_completion(f"Completion {i}: ", p)
    def criteria_block():
        return "\n".join(f"{i + 1}) {c}" for i, c in enumerate(criteria_list))

    if enable_score_filter:
        def score(player):
            text, usage = prompt_score(
                instruction,
                criteria_list,
                criteria_block(),
                player,
                model=score_model,
                api_base=api_base,
                api_key=api_token,
                return_usage=True,
            )
            add_usage(usage)
            score_outputs.append(text)
            data = _clean_json(text)
            if "scores" in data and isinstance(data["scores"], list):
                vals = data["scores"]
                return sum(vals) / len(vals) if vals else 0.0
            return float(data.get("score", 0))

        yield from log("Histogram generating")
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            scores = {
                p: s
                for p, s in zip(
                    all_players,
                    tqdm(ex.map(score, all_players), total=len(all_players)),
                )
            }
        hist_fig = plt.figure()
        plt.hist(list(scores.values()), bins=10)
        yield from log("Histogram generated")
        top_players = sorted(all_players, key=scores.get, reverse=True)[:pool_size]
        yield from log(f"Filtered to {len(top_players)} players with best scores")
        for i, txt in enumerate(score_outputs, 1):
            yield from log_completion(f"Score completion {i}: ", txt)
    else:
        top_players = all_players
    if enable_pairwise_filter:
        def play(a, b):
            text, usage = prompt_pairwise(
                instruction,
                criteria_block(),
                a,
                b,
                model=pairwise_model,
                api_base=api_base,
                api_key=api_token,
                return_usage=True,
            )
            add_usage(usage)
            pairwise_outputs.append(text)
            winner_label = _clean_json(text).get("winner", "A")
            return a if winner_label == "A" else b

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
                leftover = current[-1] if len(current) % 2 == 1 else None
                pairs = [(current[i], current[i + 1]) for i in range(0, len(current) - 1, 2)]
                round_results = tournament_round(pairs, executor)
                for w, l in round_results:
                    lost_to[l] = w
                current = [w for w, _ in round_results]
                if leftover:
                    current.append(leftover)
            return current[0], lost_to

        def get_candidates(champion, lost_to):
            return [p for p, o in lost_to.items() if o == champion] + [champion]

        def playoff(candidates, executor):
            wins = {p: 0 for p in candidates}
            pairs = [
                (candidates[i], candidates[j])
                for i in range(len(candidates))
                for j in range(i + 1, len(candidates))
            ]
            futures = {executor.submit(play, a, b): (a, b) for a, b in pairs}
            for fut in tqdm(as_completed(futures), total=len(futures)):
                wins[fut.result()] += 1
            return sorted(candidates, key=lambda p: wins[p], reverse=True)

        def get_top(players, executor):
            champion, lost_to = tournament(players, executor)
            runner_up = lost_to.get(champion)
            finalists = [champion] + ([runner_up] if runner_up else [])
            semifinalists = [p for p, o in lost_to.items() if o in finalists and p not in finalists]
            candidates = list(set(finalists + semifinalists + get_candidates(champion, lost_to)))
            return playoff(candidates, executor)[:num_top_picks]

        yield from log("Pairwise generating")
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            top_k = get_top(top_players, ex)
        for i, txt in enumerate(pairwise_outputs, 1):
            yield from log_completion(f"Pairwise completion {i}: ", txt)
    else:
        top_k = top_players[:num_top_picks]
    top_picks_str = "\n\n\n=====================================================\n\n\n".join(top_k)
    yield "\n".join(process_log + ["Done"]), hist_fig, top_picks_str, usage_str()

demo = gr.Interface(
    fn=run_tournament,
    inputs=[
        gr.Textbox(value=API_BASE_DEFAULT, label="API Base Path"),
        gr.Textbox(value="", label="API Token", type="password"),
        gr.Textbox(value=GENERATE_MODEL_DEFAULT, label="Generation Model"),
        gr.Textbox(value=SCORE_MODEL_DEFAULT, label="Score Model"),
        gr.Textbox(value=PAIRWISE_MODEL_DEFAULT, label="Pairwise Model"),
        gr.Textbox(lines=10, label="Instruction"),
        gr.Textbox(value=CRITERIA_DEFAULT, lines=5, label="Criteria (comma separated)"),
        gr.Number(value=NUM_GENERATIONS_DEFAULT, label="Number of Generations"),
        gr.Number(value=POOL_SIZE_DEFAULT, label="Top Picks Score Filter"),
        gr.Number(value=NUM_TOP_PICKS_DEFAULT, label="Top Picks Pairwise"),
        gr.Number(value=MAX_WORKERS_DEFAULT, label="Max Workers"),
        gr.Checkbox(value=SCORE_FILTER_DEFAULT, label="Enable Score Filter"),
        gr.Checkbox(value=PAIRWISE_FILTER_DEFAULT, label="Enable Pairwise Filter"),
    ],
    outputs=[
        gr.Textbox(lines=10, label="Process"),
        gr.Plot(label="Score Distribution"),
        gr.Textbox(lines=50, label="Top picks"),
        gr.Textbox(lines=5, label="Token Usage"),
    ],
    description="Generate multiple completions and use score and pairwise filters to find the best answers.",
)

if __name__ == "__main__":
    demo.launch()
