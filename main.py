import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time

NUM_TOP_PICKS = int(os.getenv("NUM_TOP_PICKS", 5))
POOL_SIZE     = int(os.getenv("POOL_SIZE", 20))
MAX_WORKERS   = int(os.getenv("MAX_WORKERS", 10))

def score(player):
    time.sleep(5)
    return random.randint(1, 10)

def play(a, b):
    time.sleep(5)
    return a if score(a) >= score(b) else b

def precompute_scores(players, executor):
    """Compute all scores in parallel once and return a dict."""
    futures = {executor.submit(score, p): p for p in players}
    scores = {}
    for fut in tqdm(as_completed(futures), total=len(futures), desc="Scoring"):
        p = futures[fut]
        scores[p] = fut.result()
    return scores

def tournament_round(pairs, executor):
    """Play a batch of matches in parallel; returns list of (winner, loser)."""
    futures = {executor.submit(play, a, b): (a, b) for a, b in pairs}
    results = []
    for fut in tqdm(as_completed(futures), total=len(futures),
                    desc="Tournament round", leave=False):
        a, b = futures[fut]
        w = fut.result()
        loser = b if w == a else a
        results.append((w, loser))
    return results

def tournament(players, executor):
    lost_to = {}
    current = players[:]
    while len(current) > 1:
        pairs = [(current[i], current[i+1]) for i in range(0, len(current)-1, 2)]
        round_results = tournament_round(pairs, executor)
        next_round = [w for w, _ in round_results]
        for w, loser in round_results:
            lost_to[loser] = w
        if len(current) % 2 == 1:
            next_round.append(current[-1])
        current = next_round
    return current[0], lost_to

def get_candidates(champion, lost_to):
    return [p for p, o in lost_to.items() if o == champion] + [champion]

def playoff(candidates, executor):
    wins = {p: 0 for p in candidates}
    pairs = [(candidates[i], candidates[j])
             for i in range(len(candidates)) for j in range(i+1, len(candidates))]
    futures = {executor.submit(play, a, b): (a, b) for a, b in pairs}
    for fut in tqdm(as_completed(futures), total=len(futures),
                    desc="Playoff", leave=False):
        winner = fut.result()
        wins[winner] += 1
    return sorted(candidates, key=lambda p: wins[p], reverse=True)

def get_top(players, executor, k=NUM_TOP_PICKS):
    champion, lost_to = tournament(players, executor)
    runnerup     = lost_to.get(champion)
    finalists    = [champion] + ([runnerup] if runnerup else [])
    semifinalists = [p for p, o in lost_to.items()
                     if o in finalists and p not in finalists]
    candidates = set(finalists + semifinalists + get_candidates(champion, lost_to))
    ranking = playoff(list(candidates), executor)
    return ranking[:k]

if __name__ == "__main__":
    all_players = [f"S{i}" for i in range(1, 101)]
    # reuse one executor for everything
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 1) Precompute scores
        scores = precompute_scores(all_players, executor)

        # 2) Pick top N by score
        top_n_players = sorted(all_players, key=lambda p: scores[p], reverse=True)[:POOL_SIZE]

        # 3) Run tournament+playoff
        top5 = get_top(top_n_players, executor)
        print("🏆 Top picks:", top5)
