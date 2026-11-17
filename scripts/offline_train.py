#!/usr/bin/env python3
"""Offline surrogate benchmark. No services, credentials, or live submissions."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random

FEATURES = ("suspicion", "seer_claim", "pressure", "activity")
BASE = {"vote": (1., 0., 0., 0.), "night_wolf": (0., 1.5, 0., 1.),
        "night_seer": (1., 0., 0., 0.)}
PHASES = tuple(BASE)
TRAIN_SEEDS = range(10000, 12400)
VALID_SEEDS = range(20000, 20420)
TEST_SEEDS = range(30000, 30840)
TRAIN_MIX = ("heuristic", "heuristic", "random", "contrarian", "heuristic", "random")
VALID_MIXES = {"mixed": TRAIN_MIX, "heuristic": ("heuristic",)*6}
TEST_MIXES = dict(VALID_MIXES, noisy_coordination=("follower", "random", "follower",
                                                 "contrarian", "heuristic", "follower"))
ROLES = ("villager", "seer", "wolf")


def dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def vote_result(ballots):
    counts = Counter(b for b in ballots if b is not None)
    if not counts:
        return None
    top = max(counts.values())
    winners = [p for p, n in counts.items() if n == top]
    return winners[0] if len(winners) == 1 else None


def night_result(ballots):
    counts = Counter(ballots)
    return max(counts, key=lambda p: (counts[p], -p))


def simulate(seed, policy, mix, sample=False):
    """Separate exogenous random draws from candidate decisions for common seed replay."""
    env = random.Random(seed)
    roles = ["wolf", "wolf", "seer"] + ["villager"]*4
    env.shuffle(roles)
    focal = env.randrange(7)
    mix = list(mix)
    env.shuffle(mix)
    opponents = dict(zip((i for i in range(7) if i != focal), mix))
    wolf = {i for i, r in enumerate(roles) if r == "wolf"}
    seer = roles.index("seer")
    alive = set(range(7))
    known, claims, activity, checks = {}, {}, Counter(), {}
    votes = []
    trace = []
    streak = 0

    def winner():
        nw = len(wolf & alive)
        if nw == 0:
            return "villager"
        if nw * 2 >= len(alive) or streak >= 3:
            return "wolf"
        return None

    def action(actor, phase, legal, day):
        # Seat learns only self/ally, its own investigations and public discussion/tally.
        # Exogenous public discussion cues are generated separately below.
        if phase == "vote" and actor == seer:
            legal = [p for p in legal if checks.get(p) == "wolf"] or legal
        if phase == "night_seer":
            legal = [p for p in legal if p not in checks] or legal
        features = []
        public_counts = Counter(votes)
        max_a = max((activity[p] for p in legal), default=0)
        for p in legal:
            prior = public_counts[p]/len(votes) if votes else 0.
            suspicion = max(-1., min(1., known.get(p, 0.) + (0.5 if p in claims.values() else 0)))
            if actor == seer and phase == "vote" and checks.get(p) == "villager":
                suspicion = -1.
            features.append((suspicion, float(p in claims), prior, activity[p]/max_a if max_a else 0.))
        # Action entropy is deterministic in each seed/decision coordinate, independent of
        # previous choices and candidate's sampling consumption.
        decision_rng = random.Random(seed*10000 + day*100 + actor*4 + PHASES.index(phase))
        if actor != focal:
            mode = opponents[actor]
            if mode == "random":
                return decision_rng.choice(legal)
            weights = {"heuristic": BASE[phase], "contrarian": (-1., 0., -.5, .25),
                       "follower": (.35, .1, 3., .5)}[mode]
            return legal[max(range(len(legal)), key=lambda j: dot(weights, features[j]) +
                       .4*decision_rng.random())]
        logits = [dot(policy[phase], f) for f in features]
        if not sample:
            return legal[max(range(len(legal)), key=lambda j: (logits[j], -legal[j]))]
        weights = [math.exp(x-max(logits)) for x in logits]
        j = decision_rng.choices(range(len(legal)), weights)[0]
        z = sum(weights)
        gradient = [features[j][k]-sum(w*f[k] for w, f in zip(weights, features))/z
                    for k in range(len(FEATURES))]
        trace.append((phase, gradient))
        return legal[j]

    def night(day):
        victim = None
        if day > 1:
            candidates = sorted(alive - wolf)
            victim = night_result([action(w, "night_wolf", candidates, day)
                                   for w in sorted(wolf & alive)])
        if seer in alive:
            p = action(seer, "night_seer", sorted(alive-{seer}), day)
            checks[p] = "wolf" if p in wolf else "villager"
        if victim is not None:
            alive.remove(victim)

    # Generate stylized publicly visible behavior using an explicit model, not actual text.
    # Propensities depend on hidden roles only via observable generated messages, never as inputs.
    for day in range(1, 8):
        night(day)
        side = winner()
        if side:
            break
        for p in range(7):
            msg = random.Random(seed*1000 + day*10 + p)
            cue = msg.uniform(-.65, .65) + (.15 if p in wolf else 0.)
            post_count = msg.randrange(0, 4)
            bluff = msg.random() < .08
            if p not in alive:
                continue
            known[p] = .5*known.get(p, 0.) + cue
            activity[p] += post_count
            if p == seer:
                finds = sorted(k for k, v in checks.items() if v == "wolf" and k in alive)
                if finds:
                    claims[p] = finds[0]
            elif p in wolf and bluff:
                claims[p] = msg.choice(sorted(alive - wolf))
        ballots = [action(p, "vote", sorted(alive-{p}-(wolf if p in wolf else set())), day)
                   for p in sorted(alive)]
        votes.extend(ballots)
        eliminated = vote_result(ballots)
        streak = streak + 1 if eliminated is None else 0
        if eliminated is not None:
            alive.remove(eliminated)
        side = winner()
        if side:
            break
    else:
        raise AssertionError("game did not reach terminal state")
    role = roles[focal]
    return role, int(side == ("wolf" if role == "wolf" else "villager")), trace


def measure(policy, seeds, mixes):
    outcomes = {}
    for name, mix in mixes.items():
        outcomes[name] = [simulate(seed, policy, mix)[:2] for seed in seeds]
    return outcomes


def rate(rows):
    return sum(win for _, win in rows)/len(rows)


def aggregate(outcomes):
    return sum(map(rate, outcomes.values()))/len(outcomes)


def paired_across_mixes(outcomes, paired):
    """Average each seed across mixes, then estimate raw and role-standardized differences."""
    n = len(next(iter(outcomes.values())))
    first = next(iter(outcomes.values()))
    shifts = [sum(rows[i][1]-paired[mix][i][1] for mix, rows in outcomes.items())/len(outcomes)
              for i in range(n)]
    role_weights = {"villager": 4/7, "seer": 1/7, "wolf": 2/7}
    role_indices = {role: [i for i, (r, _) in enumerate(first) if r == role] for role in ROLES}

    def variance(group, mean):
        return sum((shifts[i]-mean)**2 for i in group)/(len(group)*(len(group)-1))

    mean = sum(shifts)/n
    se = math.sqrt(variance(range(n), mean))
    role_means = {r: sum(shifts[i] for i in ix)/len(ix) for r, ix in role_indices.items()}
    standardized = sum(role_weights[r]*role_means[r] for r in ROLES)
    standardized_se = math.sqrt(sum(role_weights[r]**2*variance(ix, role_means[r])
                                    for r, ix in role_indices.items()))
    by_mix = {mix: sum(role_weights[r]*sum(rows[i][1] for i in ix)/len(ix)
                       for r, ix in role_indices.items()) for mix, rows in outcomes.items()}
    return {"seed_count": n, "paired_vs_baseline": mean,
            "paired_normal_95ci": [mean-1.96*se, mean+1.96*se],
            "standardized_role_weights": role_weights,
            "standardized_win_rate_by_mix": by_mix,
            "standardized_mean_win_rate": sum(by_mix.values())/len(by_mix),
            "standardized_paired_vs_baseline": standardized,
            "standardized_paired_normal_95ci": [standardized-1.96*standardized_se,
                                                  standardized+1.96*standardized_se]}


def summary(outcomes, paired):
    report = {}
    for mix, rows in outcomes.items():
        report[mix] = {}
        for role in ("overall", *ROLES):
            indices = [i for i, (r, _) in enumerate(rows) if role == "overall" or r == role]
            wins = sum(rows[i][1] for i in indices)
            n = len(indices)
            diffs = [rows[i][1]-paired[mix][i][1] for i in indices]
            mean = sum(diffs)/n
            se = math.sqrt(sum((x-mean)**2 for x in diffs)/(n*(n-1))) if n > 1 else 0.
            report[mix][role] = {"wins": wins, "n": n, "rate": wins/n,
                                 "paired_vs_baseline": mean,
                                 "paired_normal_95ci": [mean-1.96*se, mean+1.96*se]}
    return report


def train_reinforce():
    policy = {p: list(w) for p, w in BASE.items()}
    baseline = dict.fromkeys(ROLES, 0.)
    for i, seed in enumerate(TRAIN_SEEDS):
        role, win, trace = simulate(seed, policy, TRAIN_MIX, sample=True)
        reward = 2*win-1
        advantage = reward-baseline[role]
        baseline[role] += .04*advantage
        step = .06/math.sqrt(1+i/300)
        for phase, gradient in trace:
            for j, g in enumerate(gradient):
                policy[phase][j] = max(-6., min(6., policy[phase][j] + step*advantage*g))
    return policy


def train_search():
    rng = random.Random(89143)
    train = TRAIN_SEEDS[:300]
    current = BASE
    score = rate(measure(current, train, {"train": TRAIN_MIX})["train"])
    for _ in range(23):
        phase = rng.choice(PHASES)
        weights = list(current[phase])
        j = rng.randrange(len(weights))
        weights[j] += rng.choice((-1., -.5, .5, 1.))
        candidate = dict(current, **{phase: weights})
        new_score = rate(measure(candidate, train, {"train": TRAIN_MIX})["train"])
        if new_score > score:
            current, score = candidate, new_score
    return current


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/offline"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    assert vote_result([1, 2, 1, 2]) is None
    assert vote_result([1, 1, 2]) == 1
    assert night_result([4, 2]) == 2
    policies = {"baseline": BASE, "search": train_search(), "reinforce": train_reinforce()}
    validation = {name: measure(p, VALID_SEEDS, VALID_MIXES) for name, p in policies.items()}
    selected = max(policies, key=lambda name: aggregate(validation[name]))
    test = {name: measure(p, TEST_SEEDS, TEST_MIXES) for name, p in policies.items()}
    weights = {"schema_version": 1, "feature_names": FEATURES, "approved_for_online": False,
               "validation_selected": selected,
               "policies": {name: {phase: dict(zip(FEATURES, w)) for phase, w in p.items()}
                            for name, p in policies.items()}}
    report = {"schema_version": 1, "validation_selected": selected, "seed_ranges": {
                  "train": [TRAIN_SEEDS.start, TRAIN_SEEDS.stop],
                  "validation": [VALID_SEEDS.start, VALID_SEEDS.stop],
                  "test": [TEST_SEEDS.start, TEST_SEEDS.stop]},
              "search_training_range": [TRAIN_SEEDS.start, TRAIN_SEEDS.start+300],
              "seed_range_stop_exclusive": True,
              "train_mix": TRAIN_MIX, "validation_mixes": VALID_MIXES, "test_mixes": TEST_MIXES,
              "limitations": ["surrogate engine, no platform results",
                              "stylized public cues and claims, no language model or timing",
                              "the surrogate policies never abstain or vote for known wolf allies; both day actions are legal on the platform",
                              "revealed deaths affect who can act, but role/death history is not an explicit scoring feature",
                              "simulator baseline approximates phase priorities; it does not run the online WerewolfPolicy",
                              "intervals are normal approximations paired by seed, not online performance bounds"],
              "validation_aggregate": {k: aggregate(v) for k, v in validation.items()},
              "test_aggregate": {k: aggregate(v) for k, v in test.items()},
              "test_seed_paired_aggregate": {k: paired_across_mixes(v, test["baseline"])
                                            for k, v in test.items()},
              "validation": {k: summary(v, validation["baseline"]) for k, v in validation.items()},
              "test": {k: summary(v, test["baseline"]) for k, v in test.items()}}
    for name, data in (("weights", weights), ("benchmark", report)):
        (args.out_dir / f"{name}.json").write_text(json.dumps(data, indent=2)+"\n")
    print(json.dumps({k: report[k] for k in ("validation_selected", "validation_aggregate",
                                              "test_aggregate")}, indent=2))


if __name__ == "__main__":
    main()
