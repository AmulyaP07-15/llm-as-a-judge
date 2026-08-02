# LLM as a Judge

Can you trust a language model to grade another language model? We ran 2,646
comparisons to find out. Course project for CS6120 at Northeastern.

Three open weight models each answered the same questions, then took turns
judging each other's answers. Two task types, one where there is a correct
answer and one where there is not.

**The short version.** Some models judge well and some barely beat a coin flip.
The ones that cannot tell a good answer from a bad one just pick whichever
answer they saw first. So position bias is not really its own problem, it is
what a model does when it has nothing else to go on.

**The bit I am most pleased with.** We found what looked like a solid result,
judges preferring longer answers, significant at p below 0.0001. It turned out
to be nothing. Once we accounted for answer order it vanished completely. The
longer answer was winning because it happened to be sitting first, not because
it was longer. A few of our other numbers had the same problem hiding in them.

Python, Groq API, pandas, scipy. 37 tests on the metrics, including ones that
check a pooled statistic is not quietly hiding a confound.

---


## Results, objective task

2,646 judgments across 294 questions and three judges. Everything parsed
cleanly. After dropping ties we were left with 1,418 real comparisons. Full
tables come out of the analysis script.

**Some judges are much better than others.** Looking only at comparisons where
one answer was right and the other wrong, qwen got it right 84 percent of the
time. Llama 76 percent. Allam 59 percent, which is not far off guessing.

**And the worse a judge is, the more it goes on order.** Allam picked the first
answer 74 percent of the time. Llama 61 percent. Qwen 52 percent, which is
basically chance. That is the exact reverse of the accuracy ranking above. If
you can tell which answer is better you do not need to fall back on position.

**The clearest evidence came from the ties.** Almost half the time, two models
gave literally the same answer. There is no better option in that situation, and
judges picked the first one 99.3 percent of the time. With nothing to go on,
order decides it.

**The verbosity result fell apart.** Allam looked like it strongly preferred
longer answers. But split by order, the longer answer won 81 percent of the time
when it was shown first and only 42 percent when shown second. A real preference
for length would show up either way. This one was just position again.

**Two models did favour their own answers, but only in one position each.**
Allam when its answer was shown first, qwen when its answer was shown second,
llama not at all. We nearly missed qwen completely, because pooling the two
positions made the effect mostly cancel out.

**None of them know when they are wrong.** Qwen gave itself five out of five on
every single one of its 471 verdicts, including the ones it got wrong. Llama
averaged 4.89. Allam was the only one that varied and it was also the worst
judge. Confidence tells you nothing here.

**The judges do not agree with each other much.** Cohen's kappa lands between
0.36 and 0.54. It looked better before we dropped the ties, but that was just
three models agreeing to pick whatever came first.

So the answer is not that LLM judges do not work. It is that which model you
pick matters a lot, that position bias mostly shows up when a judge is out of
its depth, and that a judge will not tell you when it is guessing.

### The numbers

The eight metrics the project set out to measure.

**1. Self enhancement**

| model | self rate | neutral rate | difference | p |
|---|---|---|---|---|
| allam | 0.535 | 0.356 | +0.179 | 0.000004 |
| llama | 0.597 | 0.602 | -0.005 | 0.960 |
| qwen | 0.654 | 0.560 | +0.094 | 0.024 |

**2. First position preference**

| judge | rate | n | p |
|---|---|---|---|
| all | 0.623 | 1418 | 2.0e-20 |
| allam | 0.737 | 475 | 1.0e-25 |
| llama | 0.606 | 472 | 4.8e-06 |
| qwen | 0.524 | 471 | 0.311 |

**3. Verbosity correlation**

| judge | spearman | n | p |
|---|---|---|---|
| all | 0.092 | 912 | 0.006 |
| allam | 0.306 | 305 | 4.8e-08 |
| llama | 0.089 | 303 | 0.121 |
| qwen | -0.102 | 304 | 0.077 |

**4. Neither rate**

| judge | neither | unparsed | n |
|---|---|---|---|
| allam | 0.000 | 0.000 | 475 |
| llama | 0.006 | 0.000 | 475 |
| qwen | 0.008 | 0.000 | 475 |

**5. Judge confidence**

| judge | mean | when correct | when wrong |
|---|---|---|---|
| allam | 3.96 | 4.01 | 3.92 |
| llama | 4.89 | 4.95 | 4.83 |
| qwen | 5.00 | 5.00 | 5.00 |

**6. Response time**

| judge | mean sec | median sec |
|---|---|---|
| allam | 4.19 | 0.31 |
| llama | 0.19 | 0.15 |
| qwen | 0.14 | 0.10 |

**7. Ground truth accuracy**

| judge | accuracy | n |
|---|---|---|
| all | 0.518 | 1418 |
| allam | 0.457 | 475 |
| llama | 0.530 | 472 |
| qwen | 0.567 | 471 |

**8. Inter judge agreement**

| pair | kappa | raw agreement | n |
|---|---|---|---|
| allam vs llama | 0.463 | 0.638 | 475 |
| allam vs qwen | 0.361 | 0.566 | 475 |
| llama vs qwen | 0.542 | 0.697 | 475 |

## Supporting evidence

Four of the eight numbers above are misleading on their own. These are the
checks that showed why.

**Ties.** Almost half the comparisons had two identical answers. There is no
better option in that case, so a judge can only go on order. Leaving them in
inflates position bias and adds noise to everything else, so they are excluded
from the headline metrics. They also happen to be the cleanest experiment in the
project, since quality is held perfectly constant.

| | rate | n |
|---|---|---|
| ties | 0.447 | 1144 |
| genuine comparisons | 0.553 | 1418 |
| first picked, on ties | 0.993 | |
| first picked, on genuine | 0.623 | |

**Self enhancement split by position.** Position bias here is strong enough to
fake a self preference effect, or hide one. Splitting by whether a model's own
answer was shown first or second separates the two. Allam only favours itself
from first position, qwen only from second. The pooled test almost missed qwen
entirely because the two directions partly cancel.

| model | position | self rate | neutral rate | difference | p |
|---|---|---|---|---|---|
| allam | first | 0.764 | 0.419 | +0.346 | 1.2e-10 |
| allam | second | 0.300 | 0.292 | +0.008 | 0.961 |
| llama | first | 0.714 | 0.730 | -0.016 | 0.865 |
| llama | second | 0.497 | 0.491 | +0.006 | 0.999 |
| qwen | first | 0.677 | 0.722 | -0.044 | 0.462 |
| qwen | second | 0.629 | 0.379 | +0.250 | 4.8e-05 |

**Verbosity split by position.** Same problem. If a judge mostly picks the first
answer, then whenever the longer answer happens to sit first it wins, and that
reads as a length preference. A real one would hold in both positions. None of
them do. Longer answers actually lose more often than they win when shown second.

| judge | longer shown | longer wins | n | p |
|---|---|---|---|---|
| allam | first | 0.806 | 155 | 5.5e-15 |
| allam | second | 0.420 | 150 | 0.060 |
| llama | first | 0.654 | 153 | 0.0002 |
| llama | second | 0.420 | 150 | 0.060 |
| qwen | first | 0.516 | 155 | 0.748 |
| qwen | second | 0.436 | 149 | 0.140 |

**Discrimination accuracy.** Plain accuracy pools three different situations.
Where both answers are wrong the judge scores zero no matter what it picks, and
where both are right it scores one regardless. Neither says anything about the
judge. Only comparisons with exactly one correct answer actually test whether a
judge can tell them apart, and there the numbers are far higher than the pooled
0.518 suggests.

| case | n | share |
|---|---|---|
| one right | 605 | 0.427 |
| both wrong | 519 | 0.366 |
| both right | 294 | 0.207 |

| judge | accuracy | n | p |
|---|---|---|---|
| all | 0.727 | 605 | 7.9e-30 |
| allam | 0.589 | 202 | 0.014 |
| llama | 0.756 | 201 | 1.8e-13 |
| qwen | 0.837 | 202 | 3.3e-23 |

## Why this exists

What you actually want to know is which answer a person would prefer. Collecting
that does not scale. You need annotators, several per example so the signal is
stable, and you need to redo it every time the model changes.

Older metrics like BLEU compare against a reference answer, which stops working
once there are many valid ways to say the same thing.

So the field started using language models as the graders instead. Cheap, fast,
and close enough to human preference to be useful. The catch is that the grader
is a model too, with its own habits. That is what this project measures.

## Method

Two tasks, same protocol. One with a correct answer, one without.

- **Objective:** factual QA from TriviaQA, stratified sample of 294 questions
- **Subjective:** argument quality from IBM Debater ArgKP
- **Models:** Llama 3.1 8B, Qwen 3.6 27B, Allam 2 7B, all via Groq

Every question is answered by all three models. Answers are paired head to head,
order randomised with a fixed seed and an exact half split. Each pair is then
judged by all three models, including the ones that wrote the answers. That
rotation is what makes self preference measurable.

Judges return a choice, a confidence score, and can decline with Neither.

## Metrics

Self enhancement, position preference, verbosity correlation, neither rate,
judge confidence, response time, ground truth accuracy, and inter judge
agreement via Cohen's kappa.

Each one is also reported split by presentation order where that changes the
interpretation, which for this data is most of them.

## Running it

```bash
python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export GROQ_API_KEY="your key"
```

```
TriviaQA_dataset_preparation.py     -> TriviaQA_sampled.csv
TriviaQA_answer_generator.py        -> TriviaQA_model_answers.csv
TriviaQA_pairwise_combinations.py   -> TriviaQA_pairs.csv
TriviaQA_judge_answers.py           -> TriviaQA_judge_verdicts.csv
TriviaQA_analyze_bias.py            -> results
```

Generation and judging checkpoint and resume, which matters because the Groq
free tier has a daily token cap a full run can hit.

`TriviaQA_analyze_bias.py` needs no API access. 37 tests in
`test_TriviaQA_analyze_bias.py`.

## Limitations

One prompt, one temperature, one run per comparison. Results could be specific
to this wording.

Ground truth is string matching against TriviaQA alias lists. Standard for the
dataset but approximate.

Answers run two to three words, so verbosity had little room to show anything
here. The subjective task has genuinely long outputs, which is part of why both
are in scope.

Order is randomised rather than paired, so we measure a first position
preference rate rather than watching individual verdicts flip on reversal.
Running every pair twice would double the judge calls past what the free tier
covers.

## Results, subjective task

369 judgments across 41 topics (3 pairwise combinations per topic, 3 judges
each). No ties to exclude — free-form arguments don't repeat by accident, and
a direct check confirmed zero content-identical pairs. 345 decisive verdicts
after dropping Neither.

**Two of the three judges are mostly grading position, not content.** Llama
picks whichever argument loads first 87 percent of the time. Qwen does it 76
percent of the time. Allam is close to neutral at 59 percent, and even that is
only marginal (p = 0.053).

**The verbosity effect looked real and wasn't.** Pooled, longer arguments win
59 percent of the time (p = 0.0008). Split by position, longer arguments win
83 percent of the time when shown first and lose more often than they win when
shown second (35 percent). A genuine length preference would hold in both
positions. This one flips sign, so it's position bias wearing a verbosity
costume.

**Self-enhancement mostly evaporates once you look at position.** Llama's raw
self-preference (56 percent) tracks almost exactly with its overall
first-position rate (87 percent) whenever its own argument happens to load
first, and drops to 26 percent when it loads second — which is just what a
judge that always picks position one would do regardless of who wrote what.
Same pattern, more muted, for qwen. Allam is the exception: self rate (42
percent) and baseline rate (41 percent) are statistically indistinguishable
(chi2 p = 1.0) — the most content-driven judge of the three, and the one with
almost no position bias.

**Confidence and speed pull in the same direction.** Allam is both the
slowest judge (0.45s average) and the most confident (4.48 / 5). Qwen is the
fastest (0.24s) and least confident (3.89 / 5). Worth a footnote, not a
headline.

**Judges agree moderately with each other.** Cohen's kappa ranges from 0.44
to 0.61. Llama and qwen agree most (0.61) — plausibly because they share the
same position-bias pattern rather than because they're reasoning about
arguments the same way. Allam, the outlier judge, agrees least with either.

No ground truth exists for this task — there's no "correct" side of an
opinion argument — so unlike the objective task there's no accuracy metric
here. The bias patterns are the finding.

### The numbers (subjective task)

**1. Self enhancement**

| model | self rate | neutral rate | difference | p |
|---|---|---|---|---|
| allam | 0.420 (n=81) | 0.408 (n=76) | +0.012 | 1.0000 |
| llama | 0.561 (n=82) | 0.694 (n=72) | −0.133 | 0.1238 |
| qwen  | 0.493 (n=69) | 0.410 (n=78) | +0.083 | 0.4023 |

Pooled across all three judges: 0.491 (n=232) — see the position split below
before reading this as "no self-enhancement."

**2. First position preference**

| judge | rate | n | p |
|---|---|---|---|
| all | 0.742 | 345 | <0.0001 |
| allam | 0.593 | 118 | 0.0527 |
| llama | 0.870 | 123 | <0.0001 |
| qwen | 0.760 | 104 | <0.0001 |

**3. Verbosity (position split, in place of pooled correlation)**

| when longer argument shown | win rate | n | p |
|---|---|---|---|
| first | 0.825 | 177 | <0.0001 |
| second | 0.345 | 168 | 0.0001 |
| pooled (misleading alone) | 0.591 | 345 | 0.0008 |

**4. Outcome rates**

| outcome | rate | n |
|---|---|---|
| Neither | 0.065 | 24 |
| Unclear (unparseable) | 0.000 | 0 |
| Echoed prompt | 0.000 | 0 |
| API error | 0.000 | 0 |

(all out of 369 total verdicts)

**5. Judge confidence**

| judge | mean | n |
|---|---|---|
| all | 4.22 | 354 |
| allam | 4.48 | 108 |
| llama | 4.33 | 123 |
| qwen | 3.89 | 123 |

**6. Response time**

| judge | mean sec | n |
|---|---|---|
| all | 0.321 | 369 |
| allam | 0.448 | 123 |
| llama | 0.238 | 123 |
| qwen | 0.276 | 123 |

**7. Ground truth accuracy**

N/A — ArgKP arguments are opinion, not fact; there's no correct answer to
score against.

**8. Inter judge agreement**

| pair | kappa | n |
|---|---|---|
| allam vs llama | 0.525 | 118 |
| allam vs qwen | 0.442 | 99 |
| llama vs qwen | 0.607 | 104 |

### Supporting evidence (subjective task)

**Content ties.** Checked directly rather than assumed: zero pairs where the
two arguments being compared were textually identical. Unlike the objective
task's short factual answers, free-form 2-3 sentence arguments essentially
never collide by chance, so no exclusion was needed here.

**Self-enhancement split by position.**

| model | position | self rate | n |
|---|---|---|---|
| allam | first | 0.512 | 43 |
| allam | second | 0.316 | 38 |
| llama | first | 0.897 | 39 |
| llama | second | 0.256 | 43 |
| qwen | first | 0.774 | 31 |
| qwen | second | 0.263 | 38 |

Baseline (neutral-judge) rate isn't yet split by position here, only the self
rate — so this shows self-preference varies sharply by position, but doesn't
fully isolate a self-preference effect net of position the way a complete
comparison would. Splitting the baseline the same way is the natural next
step.

**Verbosity split by position:** see table 3 above — the split is the whole
finding for this metric, not a side check.