from dotenv import load_dotenv
load_dotenv("./local.env",override=True)
import os, json, re, ast, gradio as gr
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import matplotlib.pyplot as plt
from tournament_utils import generate_players, prompt_score, prompt_pairwise
import time


class SimpleProgress:
    """Minimal progress helper to compute ETA."""

    def __init__(self, total: int, prefix: str = "Progress"):
        self.total = total
        self.prefix = prefix
        self.start = time.time()
        self.count = 0

    def step(self) -> str:
        self.count += 1
        elapsed = time.time() - self.start
        remaining = (elapsed / self.count) * (self.total - self.count) if self.count else 0
        h, rem = divmod(int(remaining), 3600)
        m, s = divmod(rem, 60)
        if h:
            eta = f"{h:d}:{m:02d}:{s:02d}"
        else:
            eta = f"{m:02d}:{s:02d}"
        return f"{self.prefix} {self.count}/{self.total} - ETA {eta}"

NUM_TOP_PICKS_DEFAULT = int(os.getenv("NUM_TOP_PICKS", 3))
POOL_SIZE_DEFAULT = int(os.getenv("POOL_SIZE", 6))
MAX_WORKERS_DEFAULT = int(os.getenv("MAX_WORKERS", 100))
NUM_GENERATIONS_DEFAULT = int(os.getenv("NUM_GENERATIONS", 10))
API_BASE_DEFAULT = os.getenv("OPENAI_API_BASE", "")
API_TOKEN_DEFAULT = os.getenv("OPENAI_API_KEY", "")
SCORE_FILTER_DEFAULT = os.getenv("ENABLE_SCORE_FILTER", "true").lower() == "true"
PAIRWISE_FILTER_DEFAULT = os.getenv("ENABLE_PAIRWISE_FILTER", "true").lower() == "true"
GENERATE_MODEL_DEFAULT = os.getenv("GENERATE_MODEL", "gpt-4o-mini")
SCORE_MODEL_DEFAULT = os.getenv("SCORE_MODEL", "gpt-4o-mini")
PAIRWISE_MODEL_DEFAULT = os.getenv("PAIRWISE_MODEL", "gpt-4o-mini")
GENERATE_TEMPERATURE_DEFAULT = float(os.getenv("GENERATE_TEMPERATURE", "0.9"))
SCORE_TEMPERATURE_DEFAULT = float(os.getenv("SCORE_TEMPERATURE", "0.6"))
PAIRWISE_TEMPERATURE_DEFAULT = float(os.getenv("PAIRWISE_TEMPERATURE", "0.6"))
SCORE_WITH_INSTRUCTION_DEFAULT = os.getenv("PASS_INSTRUCTION_TO_SCORE", "true").lower() == "true"
PAIRWISE_WITH_INSTRUCTION_DEFAULT = os.getenv("PASS_INSTRUCTION_TO_PAIRWISE", "true").lower() == "true"
GENERATE_THINKING_DEFAULT = os.getenv("ENABLE_GENERATE_THINKING", "false").lower() == "true"
SCORE_THINKING_DEFAULT = os.getenv("ENABLE_SCORE_THINKING", "false").lower() == "true"
PAIRWISE_THINKING_DEFAULT = os.getenv("ENABLE_PAIRWISE_THINKING", "false").lower() == "true"
CRITERIA_DEFAULT = "Factuality,Concise,Precision"

# Regex used to capture the final verdict from judge output
FINAL_VERDICT_RE = re.compile(r"(?im)^final verdict:\s*(.*)$")


def _parse_verdict(txt: str) -> dict:
    """Extract verdict information from judge output."""
    txt = re.sub(r"^```.*?\n|```$", "", txt, flags=re.DOTALL).strip()
    match = FINAL_VERDICT_RE.search(txt)
    if not match:
        return {}
    verdict = match.group(1).strip()
    try:
        verdict_val = ast.literal_eval(verdict)
    except Exception:
        verdict_val = verdict
    if isinstance(verdict_val, list):
        return {"scores": verdict_val}
    return {"winner": str(verdict_val)}

def run_tournament(
    api_base,
    api_token,
    generate_model,
    score_model,
    pairwise_model,
    generate_temperature,
    score_temperature,
    pairwise_temperature,
    instruction_input,
    criteria_input,
    n_gen,
    pool_size,
    num_top_picks,
    max_workers,
    enable_score_filter,
    enable_pairwise_filter,
    score_with_instruction,
    pairwise_with_instruction,
    generate_thinking,
    score_thinking,
    pairwise_thinking,
    score_explain=None,
    pairwise_explain=None,
):
    instruction = instruction_input.strip()
    criteria_list = [c.strip() for c in criteria_input.split(",") if c.strip()] or ["Factuality", "Instruction Following", "Precision"]
    n_gen = int(n_gen)
    num_top_picks = int(num_top_picks)
    pool_size = int(pool_size)
    max_workers = int(max_workers)
    if generate_temperature is None:
        generate_temperature = GENERATE_TEMPERATURE_DEFAULT
    if score_temperature is None:
        score_temperature = SCORE_TEMPERATURE_DEFAULT
    if pairwise_temperature is None:
        pairwise_temperature = PAIRWISE_TEMPERATURE_DEFAULT
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
    if score_with_instruction is None:
        score_with_instruction = SCORE_WITH_INSTRUCTION_DEFAULT
    if pairwise_with_instruction is None:
        pairwise_with_instruction = PAIRWISE_WITH_INSTRUCTION_DEFAULT
    if generate_thinking is None:
        generate_thinking = GENERATE_THINKING_DEFAULT
    if score_thinking is None:
        score_thinking = SCORE_THINKING_DEFAULT
    if pairwise_thinking is None:
        pairwise_thinking = PAIRWISE_THINKING_DEFAULT
    if score_explain is None:
        score_explain = False
    if pairwise_explain is None:
        pairwise_explain = False

    process_log = []
    hist_fig = None
    elo_fig = None
    top_picks_str = ""
    prompt_tokens = 0
    completion_tokens = 0
    score_outputs: list[str] = []
    raw_scores: dict[str, list] = {}
    pairwise_outputs: list[str] = []
    match_cache: dict[tuple[str, str], str] = {}

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

    def log_completion(prefix: str, text: str, player_id: int | None = None):
        disp = text.replace("\n", " ")
        if len(disp) > 1000:
            disp = disp[:1000] + "…"
        if player_id is not None:
            prefix = f"{prefix}(ID {player_id}) "
        return log(f"{prefix}{disp}")
    def log(msg):
        process_log.append(msg)
        tqdm.write(msg)
        yield "\n".join(process_log), hist_fig, elo_fig, top_picks_str, usage_str()
    yield from log("Generating answers …")
    all_players, usage = generate_players(
        instruction,
        n_gen,
        model=generate_model,
        api_base=api_base,
        api_key=api_token,
        temperature=generate_temperature,
        thinking=generate_thinking,
        return_usage=True,
    )
    add_usage(usage)
    yield from log(f"{len(all_players)} players generated")
    for i, p in enumerate(all_players, 1):
        yield from log_completion(f"Completion {i}: ", p, i)
    def criteria_block():
        return "\n".join(f"{i + 1}) {c}" for i, c in enumerate(criteria_list))

    if enable_score_filter:
        players_with_ids = list(enumerate(all_players, 1))

        def score(item):
            idx, player = item
            text, usage = prompt_score(
                instruction,
                criteria_list,
                criteria_block(),
                player,
                model=score_model,
                api_base=api_base,
                api_key=api_token,
                temperature=score_temperature,
                include_instruction=score_with_instruction,
                thinking=score_thinking,
                explain=score_explain,
                return_usage=True,
            )
            add_usage(usage)
            score_outputs.append((idx, text))
            data = _parse_verdict(text)
            raw_vals = None
            if "scores" in data and isinstance(data["scores"], list):
                raw_vals = data["scores"]
                avg = sum(raw_vals) / len(raw_vals) if raw_vals else 0.0
            else:
                try:
                    avg = float(data.get("score", 0))
                    raw_vals = [avg]
                except Exception:
                    avg = 0.0
                    raw_vals = None
            return avg, raw_vals

        yield from log("Histogram generating")
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            prog = SimpleProgress(len(all_players), "Scoring")
            scores = {}
            for (idx, p), (s_val, raw_val) in zip(players_with_ids, ex.map(score, players_with_ids)):
                scores[p] = s_val
                if raw_val is not None:
                    raw_scores[p] = raw_val
                yield from log(prog.step())
        hist_fig = plt.figure()
        plt.hist(list(scores.values()), bins=10)
        yield from log("Histogram generated")
        top_players = sorted(all_players, key=scores.get, reverse=True)[:pool_size]
        yield from log(f"Filtered to {len(top_players)} players with best scores")
        for i, (idx, txt) in enumerate(score_outputs, 1):
            yield from log_completion(f"Score completion {i}: ", txt, idx)
    else:
        top_players = all_players
    if enable_pairwise_filter:
        def play(a, b):
            key = tuple(sorted((a, b)))
            if key in match_cache:
                return match_cache[key]
            text, usage = prompt_pairwise(
                instruction,
                criteria_block(),
                a,
                b,
                model=pairwise_model,
                api_base=api_base,
                api_key=api_token,
                temperature=pairwise_temperature,
                include_instruction=pairwise_with_instruction,
                thinking=pairwise_thinking,
                explain=pairwise_explain,
                return_usage=True,
            )
            add_usage(usage)
            pairwise_outputs.append(text)
            winner_label = _parse_verdict(text).get("winner", "A")
            winner = a if winner_label == "A" else b
            match_cache[key] = winner
            return winner

        def all_pairs(players):
            for i in range(len(players)):
                for j in range(i + 1, len(players)):
                    yield players[i], players[j]

        def rate(players, executor):
            rating = {p: 1000.0 for p in players}
            pairs = list(all_pairs(players))
            futures = {executor.submit(play, a, b): (a, b) for a, b in pairs}
            prog = SimpleProgress(len(futures), "Elo matches")
            K = 32
            for fut in as_completed(futures):
                a, b = futures[fut]
                winner = fut.result()
                loser = b if winner == a else a
                ra, rb = rating[a], rating[b]
                ea = 1 / (1 + 10 ** ((rb - ra) / 400))
                eb = 1 - ea
                if winner == a:
                    rating[a] = ra + K * (1 - ea)
                    rating[b] = rb + K * (0 - eb)
                else:
                    rating[a] = ra + K * (0 - ea)
                    rating[b] = rb + K * (1 - eb)
                yield from log(prog.step())
            return rating

        yield from log("Pairwise generating")
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            rating = yield from rate(top_players, ex)
        elo_fig = plt.figure()
        players_sorted = sorted(rating, key=rating.get, reverse=True)
        plt.bar(range(len(players_sorted)), [rating[p] for p in players_sorted])
        plt.xticks(range(len(players_sorted)), [str(i + 1) for i in range(len(players_sorted))])
        top_k = players_sorted[:num_top_picks]
        for i, txt in enumerate(pairwise_outputs, 1):
            yield from log_completion(f"Pairwise completion {i}: ", txt)
        top_picks_str = "\n\n\n=====================================================\n\n\n".join(
            f"{p}\nElo: {rating[p]:.1f}" + (f"\nScore: {raw_scores.get(p)}" if p in raw_scores else "") for p in top_k
        )
    else:
        top_k = top_players[:num_top_picks]
        top_picks_str = "\n\n\n=====================================================\n\n\n".join(top_k)
    yield "\n".join(process_log + ["Done"]), hist_fig, elo_fig, top_picks_str, usage_str()

demo = gr.Interface(
    fn=run_tournament,
    inputs=[
        gr.Textbox(value=API_BASE_DEFAULT, label="API Base Path"),
        gr.Textbox(value="", label="API Token", type="password"),
        gr.Textbox(value=GENERATE_MODEL_DEFAULT, label="Generation Model"),
        gr.Textbox(value=SCORE_MODEL_DEFAULT, label="Score Model"),
        gr.Textbox(value=PAIRWISE_MODEL_DEFAULT, label="Pairwise Model"),
        gr.Number(value=GENERATE_TEMPERATURE_DEFAULT, label="Generation Temperature"),
        gr.Number(value=SCORE_TEMPERATURE_DEFAULT, label="Score Temperature"),
        gr.Number(value=PAIRWISE_TEMPERATURE_DEFAULT, label="Pairwise Temperature"),
        gr.Textbox(lines=10, label="Instruction"),
        gr.Textbox(value=CRITERIA_DEFAULT, lines=5, label="Criteria (comma separated)"),
        gr.Number(value=NUM_GENERATIONS_DEFAULT, label="Number of Generations"),
        gr.Number(value=POOL_SIZE_DEFAULT, label="Top Picks Score Filter"),
        gr.Number(value=NUM_TOP_PICKS_DEFAULT, label="Top Picks Pairwise"),
        gr.Number(value=MAX_WORKERS_DEFAULT, label="Max Workers"),
        gr.Checkbox(value=SCORE_FILTER_DEFAULT, label="Enable Score Filter"),
        gr.Checkbox(value=PAIRWISE_FILTER_DEFAULT, label="Enable Pairwise Filter"),
        gr.Checkbox(value=SCORE_WITH_INSTRUCTION_DEFAULT, label="Pass Instruction to Score Model"),
        gr.Checkbox(value=PAIRWISE_WITH_INSTRUCTION_DEFAULT, label="Pass Instruction to Pairwise Model"),
        gr.Checkbox(value=GENERATE_THINKING_DEFAULT, label="Enable Thinking (Generate)"),
        gr.Checkbox(value=SCORE_THINKING_DEFAULT, label="Enable Thinking (Score)"),
        gr.Checkbox(value=PAIRWISE_THINKING_DEFAULT, label="Enable Thinking (Pairwise)"),
        gr.Checkbox(value=False, label="Enable Explain (Score)"),
        gr.Checkbox(value=False, label="Enable Explain (Pairwise)"),
    ],
    outputs=[
        gr.Textbox(lines=10, label="Process"),
        gr.Plot(label="Score Distribution"),
        gr.Plot(label="Elo Ratings"),
        gr.Textbox(lines=50, label="Top picks"),
        gr.Textbox(lines=5, label="Token Usage"),
    ],
    description="Generate multiple completions and use score and pairwise filters to find the best answers.",
)

if __name__ == "__main__":
    demo.launch()
