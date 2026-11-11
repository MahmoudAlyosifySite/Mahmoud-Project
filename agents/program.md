# How a team of agents works here

One researcher, one A40, many agents. The scarce things are GPU hours and the researcher's attention, so the team is shaped like a funnel: many cheap ideas and critiques at the top, few experiments in the middle, very few claims at the bottom.

`CLAUDE.md` at the repository root holds the rules every agent follows. This file is about who does what.

## The funnel

1. **Read.** Scan new papers on latent world models, conservation, and surprise. For each, extract the claims that can be tested here and add them to `docs/reading-list.md`.
2. **Propose.** Write a proposal from `agents/proposal-template.md`. It must state the expected outcome before the run.
3. **Attack.** A different agent tries to kill the proposal on paper: is it already done, is the test fair to the baseline, can the result be explained by something dull? Most proposals should die here, which costs nothing.
4. **Run.** One experiment script, results to the ledger.
5. **Audit.** A different session reruns with new seeds, reads the code path for leaks, and tries to break the result.
6. **Decide.** The researcher reads the audited result and decides what it means.

## Roles

| Role | Does | Never does |
|---|---|---|
| Reader | summarises papers into testable claims, with the source of every statement | describes what a paper lacks from its abstract alone |
| Proposer | writes proposals with predictions | runs its own proposal unreviewed |
| Critic | finds the flaw, the prior work, the unfair baseline | softens a criticism to be agreeable |
| Experimenter | runs scripts, logs to the ledger | edits a score and runs it in one change |
| Auditor | reruns with new seeds, checks for leaks | audits a run it launched |
| Scenario author | writes new possible, improbable and impossible scenarios with tests that prove each one keeps or breaks the intended law | ships a scenario without a conservation test |

Scenario authoring is where many agents help most. Every new family needs: a lawful control that is equally unusual, a test that the improbable clips conserve energy and count to 1e-12, and a test that the impossible clips change exactly the intended quantity at exactly one step. `tests/test_worlds.py` shows the pattern.

## How many agents

Start with three to five and measure before adding more:

- experiments completed per day,
- share of proposals that survive the critic,
- share of results that survive the audit,
- cost per audited result.

Add agents only when the GPU queue is empty and agents are the bottleneck. A 2026 preregistered study (arXiv 2608.16951) found that seven of ten agent-loop designs did not beat simple sampling from the same model, and that the full scaffold did worse than blind sampling at seven to eight times the token cost. More agents and more scaffolding are a cost to be justified, not a default.

## With 1000 experiments, some will look good by luck

That is why seed 0 is exploratory, why every seed is reported, why the go rule is fixed in advance, and why nothing counts before the audit.
