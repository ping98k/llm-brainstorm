import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time

# Number of top picks to return (default 5)
NUM_TOP_PICKS = int(os.getenv("NUM_TOP_PICKS", 5))
# Initial pool size after scoring (default 20)
POOL_SIZE     = int(os.getenv("POOL_SIZE", 20))
# Maximum number of worker threads for parallel execution
MAX_WORKERS   = int(os.getenv("MAX_WORKERS", 10))

# -----------------------------------------------------------------------------
# score: Simulate an expensive scoring operation for a player.
# Sleeps for 5 seconds then returns a random score between 1 and 10.
def score(player):
    time.sleep(5)
    return random.randint(1, 10)

# play: Determine match winner using precomputed scores.
# Compare two players' scores in O(1) time without additional delays.
def play(a, b, scores):
    # Return 'a' if its score >= b's score, else 'b'
    return a if scores[a] >= scores[b] else b

# precompute_scores: Batch parallel scoring of all players.
# Returns a dict mapping player -> their computed score.
def precompute_scores(players, executor):
    """Compute all scores in parallel once and return a dict."""
    # Submit all score() calls to the executor
    futures = {executor.submit(score, p): p for p in players}
    scores = {}
    # Collect results as they complete
    for fut in tqdm(as_completed(futures), total=len(futures), desc="Scoring"):
        p = futures[fut]
        scores[p] = fut.result()
    return scores

# tournament_round: Play one elimination round in parallel.
# Takes pairs of players, returns a list of (winner, loser) tuples.
def tournament_round(pairs, executor, scores):
    """Play a batch of matches in parallel; returns list of (winner, loser)."""
    futures = {executor.submit(play, a, b, scores): (a, b) for a, b in pairs}
    results = []
    # As matches complete, record winners and losers
    for fut in tqdm(as_completed(futures), total=len(futures),
                    desc="Tournament round", leave=False):
        a, b = futures[fut]
        w = fut.result()
        loser = b if w == a else a
        results.append((w, loser))
    return results

# tournament: Run full single-elimination bracket on a player list.
# Returns the champion and a map of who each loser lost to.
def tournament(players, executor, scores):
    lost_to = {}
    current = players[:]
    # Continue until only one player remains
    while len(current) > 1:
        # Pair off adjacent players
        pairs = [(current[i], current[i+1]) for i in range(0, len(current)-1, 2)]
        round_results = tournament_round(pairs, executor, scores)
        next_round = [w for w, _ in round_results]
        # Record loss relationships
        for w, loser in round_results:
            lost_to[loser] = w
        # Handle odd player out (bye)
        if len(current) % 2 == 1:
            next_round.append(current[-1])
        current = next_round
    # Final remaining player is champion
    return current[0], lost_to

# get_candidates: Identify players who lost directly to the champion.
def get_candidates(champion, lost_to):
    # Include all who lost to champion + champion itself
    return [p for p, o in lost_to.items() if o == champion] + [champion]

# playoff: Conduct a round-robin among candidates to refine ranking.
# Returns candidates sorted by number of wins descending.
def playoff(candidates, executor, scores):
    wins = {p: 0 for p in candidates}
    # Generate all unique matchups
    pairs = [(candidates[i], candidates[j])
             for i in range(len(candidates)) for j in range(i+1, len(candidates))]
    futures = {executor.submit(play, a, b, scores): (a, b) for a, b in pairs}
    # Tally wins as matches complete
    for fut in tqdm(as_completed(futures), total=len(futures),
                    desc="Playoff", leave=False):
        winner = fut.result()
        wins[winner] += 1
    # Sort by wins (highest first)
    return sorted(candidates, key=lambda p: wins[p], reverse=True)

# get_top: Main orchestration to get top K players.
# 1) Run tournament to identify top bracket players
# 2) Gather candidates (champion, runner-up, semifinalists)
# 3) Conduct playoff to finalize top K ordering
# Returns a list of top 'k' players.
def get_top(players, executor, scores, k=NUM_TOP_PICKS):
    champion, lost_to = tournament(players, executor, scores)
    runnerup = lost_to.get(champion)
    finalists = [champion] + ([runnerup] if runnerup else [])
    semifinalists = [p for p, o in lost_to.items() if o in finalists and p not in finalists]
    candidates = set(finalists + semifinalists + get_candidates(champion, lost_to))
    ranking = playoff(list(candidates), executor, scores)
    return ranking[:k]

if __name__ == "__main__":
    # Create list of 100 players labeled S1..S100
    all_players = [f"S{i}" for i in range(1, 101)]
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 1) Compute scores once
        scores = precompute_scores(all_players, executor)

        # 2) Select top N players by score
        top_n_players = sorted(all_players, key=lambda p: scores[p], reverse=True)[:POOL_SIZE]

        # 3) Run optimized tournament + playoff using cached scores
        top5 = get_top(top_n_players, executor, scores)
        print("🏆 Top picks:", top5)
