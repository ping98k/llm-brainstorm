import os

TOP_K = int(os.getenv("TOP_K", 5))

def play(a,b):
    return a

def tournament(players):
    lost_to = {}
    current = players[:]
    while len(current) > 1:
        next_round = []
        for i in range(0, len(current) - 1, 2):
            a, b = current[i], current[i + 1]
            w = play(a, b)
            loser = b if w == a else a
            lost_to[loser] = w
            next_round.append(w)
        if len(current) % 2 == 1:
            next_round.append(current[-1])
        current = next_round
    champion = current[0]
    return champion, lost_to

def get_candidates(champion, lost_to):
    c = {champion}
    for player, opponent in lost_to.items():
        if opponent == champion:
            c.add(player)
    return list(c)

def playoff(candidates):
    wins = {p: 0 for p in candidates}
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            w = play(candidates[i], candidates[j])
            wins[w] += 1
    return sorted(candidates, key=lambda p: wins[p], reverse=True)

def get_top(players, k=TOP_K):
    champion, lost_to = tournament(players)
    runnerup = lost_to.get(champion)
    finalists = [champion] + ([runnerup] if runnerup else [])
    semifinalists = [p for p, o in lost_to.items() if o in finalists and p not in finalists]
    candidates = set(finalists + semifinalists)
    candidates.update(get_candidates(champion, lost_to))
    ranking = playoff(list(candidates))
    return ranking[:k]

players = [f"S{i}" for i in range(1, 21)]
top_picks = get_top(players)
print(top_picks)