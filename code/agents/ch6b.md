## 6.4 LLM-as-judge, and how to calibrate it

### The problem: some checks cannot be written as code

Section 6.3 turned as many pass/fail criteria as possible into code checks: a refund amount must equal the price minus the fee, a reply must contain the order id. Some criteria refuse to become code. "The reply does not promise a delivery date" can be written a hundred ways. "The tone does not blame the customer" has no regular expression. For these, teams ask a second model to read the output and decide. Chapter 4 (Section 4.6) used such a judge inside a loop and saw it fail at counting. Here the question is different: **how do you know how far to trust a judge's numbers?**

A good analogy is a new smoke detector. Before you trust it, you burn some toast on purpose and check that it beeps, and you check that it stays quiet on a normal morning. A detector that beeps all the time is useless, and so is one that never beeps. You measure both behaviours, separately. A judge is the same: you test it on outputs whose answer you already know.

> [!DEFINITION] LLM judge (in an eval)
> A model call that reads one output (and its inputs) and returns a verdict, here PASS or FAIL against written criteria. In an eval, the judge's verdicts are averaged over a data set to give a score. The judge is itself a measuring instrument, so it has an error rate that must be measured.

### Where the idea comes from

The reference study is Zheng et al. (2023), already met in Chapter 4. It found that a strong judge agreed with human preferences about as often as humans agree with each other, and it named three biases: position, verbosity and self-enhancement. Chapter 4 showed the position-bias table; the passage below is the agreement result.

> [!PAPER] Zheng et al. (2023), Judging LLM-as-a-Judge · §4.2 · page 7
> [![A passage of the LLM-as-a-judge paper with highlighted phrases: the agreement under setup S2, without ties, between GPT-4 and humans reaches 85 percent, which is even higher than the agreement among humans, 81 percent](/img/agents/ch6b-judge-agreement.png)](/img/agents/ch6b-judge-agreement.png)
>
> **What it says:** on MT-bench, "the agreement under setup S2 (w/o tie) between GPT-4 and humans reaches 85%, which is even higher than the agreement among humans (81%)".
>
> **What it means:**
>
> Read the condition before the number. "S2 (w/o tie)" means the agreement is counted only on pairs where neither side called a tie. The task is a preference between two answers to open questions, and the humans are expert annotators. Under those conditions, the judge disagrees with a human about as often as two humans disagree with each other.
>
> That is a strong result for that task. It is not a promise about yours. Your judge answers a different question (does this reply follow Northwind's refund policy?), with a different prompt and a different model. Agreement on open-ended preference says nothing about whether a judge can compute "89.50 minus 6.95". The only way to know is to measure your judge on your own labelled outputs.

The three biases each have a test you can run on your own judge. For **position bias**, swap the order of two answers and count how often the verdict changes. For **verbosity bias**, pad a correct answer with harmless text and see whether its score rises. For **self-enhancement bias**, compare the judge's scores on outputs written by its own model and by another model of similar quality. Zheng et al. found clear position and verbosity effects; for self-enhancement they reported signs of it but said the data was not enough to conclude. A binary verdict on one output, against a rubric, avoids position bias by design, because there is no second answer to put first.

### Writing a binary judge prompt

Two habits make a judge easier to calibrate. First, **binary verdicts**. A score from 1 to 5 sounds richer, but nobody can say what separates a 3 from a 4, and the judge drifts between them. PASS or FAIL forces a decision that you can compare with a human label. Second, **a rubric**: the criteria are written down, each one testable, and one failure means FAIL. The rubric is the pass/fail criteria of Section 6.3, rewritten for a reader that is a model.

```text
It PASSES only if all five criteria hold; one failure means FAIL.
C1 Facts: every fact about the order (item, status, dates, amounts) matches the order record.
C2 Policy: the decision (refund, return, cancel, replacement) and any amount follow the policy exactly, including fees.
C3 No invented promises: no delivery date, discount, credit or free shipping that the policy does not allow.
C4 Answers the question: the customer's actual question is answered in the first two sentences.
C5 Tone: polite; does not blame the customer.
Compute any amount yourself before you decide.
...
Answer with JSON only: {"failed": [...], "reason": "one sentence", "verdict": "PASS" or "FAIL"}
```

The judge sees the same evidence a human reviewer would: the policy, the order record, the customer's message and the reply. It does not see the label. Asking for the failed criteria and a reason before the verdict costs a few tokens and gives you something to read during error analysis.

### Calibration: two error rates, not one accuracy

Calibrating a judge means running it on outputs that a person has already labelled, and comparing. The result is a 2 x 2 table.

> [!DEFINITION] Confusion matrix, TPR and TNR
> Four counts: **TP** (label pass, judge PASS), **FN** (label pass, judge FAIL), **FP** (label fail, judge PASS), **TN** (label fail, judge FAIL). The **true positive rate** is $$\text{TPR} = \frac{TP}{TP + FN}$$, the share of good outputs the judge passes. The **true negative rate** is $$\text{TNR} = \frac{TN}{TN + FP}$$, the share of bad outputs the judge catches. Here "positive" means PASS.

Why not just accuracy? Because accuracy mixes the two errors, and they cost different things. Suppose 95 of 100 replies are good. A "judge" that always says PASS is 95% accurate and catches nothing: its TNR is 0. A false PASS ships a broken reply to a customer. A false FAIL wastes a reviewer's time or blocks a good change. You want to know each rate on its own, and you need enough failing examples in the labelled set to measure TNR at all. That is why a calibration set should hold a good number of failures even when failures are rare in production.

### Correcting the measured pass rate

The judge's real job is to estimate a pass rate on outputs nobody has labelled, for example a week of production replies. Its raw pass rate is biased by its own errors. If you know TPR and TNR, you can undo the bias.

Think of it as a scale that always reads a little wrong in a known way. If you know a scale adds 2 kg, you subtract 2 kg. Here the error is a mix of two rates, so the correction is a small equation.

Let $$\theta$$ be the true pass rate and $$p_{\text{obs}}$$ the share of outputs the judge passes. The judge passes a good output with probability TPR and a bad one with probability $$1 - \text{TNR}$$, so

$$p_{\text{obs}} = \theta\,\text{TPR} + (1-\theta)(1-\text{TNR}).$$

Solve for $$\theta$$:

$$\hat\theta = \frac{p_{\text{obs}} + \text{TNR} - 1}{\text{TPR} + \text{TNR} - 1}.$$

Read it left to right. The numerator removes the false passes that a judge with this TNR would add. The denominator rescales by how much better than a coin the judge is: if $$\text{TPR} + \text{TNR} = 1$$ the judge's verdict carries no information and the formula breaks, which is the right behaviour. Epidemiologists use the same formula, the Rogan-Gladen estimator (1978), to correct a disease prevalence measured with an imperfect test.

A tiny example. The judge passes 60 of 100 replies, so $$p_{\text{obs}} = 0.6$$. Calibration gave $$\text{TPR} = 0.8$$ and $$\text{TNR} = 0.9$$. Then $$\hat\theta = (0.6 + 0.9 - 1)/(0.8 + 0.9 - 1) = 0.5/0.7 = 0.714$$. The judge said 60%; the corrected estimate is about 71%, because this judge rejects one good reply in five.

**Assumptions.** The judge's TPR and TNR on the new outputs are the same as on the calibration set. That holds only if the calibration set looks like the new outputs: the same kinds of requests, the same kinds of failures, the same writing styles. TPR and TNR are themselves estimates from a few dozen labels, so the corrected rate needs an interval. A simple way is the bootstrap: resample the calibration set and the new set with replacement many times, recompute $$\hat\theta$$ each time, and take the middle 95% of the results.

{{FIG:b_calibration|Calibrating a judge. Left: run the judge on a labelled calibration set and count the four cells of the confusion matrix, which give TPR and TNR. Right: run the same judge on unlabelled outputs, read its raw pass rate, and correct it with TPR and TNR. The correction is only as good as the match between the two sets.}}

### Our experiment: a judge on Northwind's support replies

`ch6b_data.py` builds 80 support replies for Northwind Home, each answering one customer message about one order (a status question, a return, a damaged item, a cancellation). Every reply has a label set by construction: a passing reply follows the policy, and a failing reply has exactly one planted defect. Some defects are loud (a reply that never answers the question). Most are quiet: a refund time of "2 to 3 business days" where the policy says 5 to 7, a 9.95 USD fee where the policy says 6.95, a 30-day return granted on day 32. Half the passing replies are written in a short, informal style that is still correct, for example "Hi Ken. Sorry about that. Send the photo; we'll refund 32.50 USD in full or replace it, your choice. Keep the item."

The replies come in two sets of 40. The **calibration set** has 20 passing and 20 failing replies; we treat its labels as the human labels. The **production set** stands in for a week of real traffic: 28 passing and 12 failing, a true pass rate of 70%, which the judge must estimate without seeing labels. `ch6b_judge.py` grades all 80 replies twice with gpt-5-mini: once with a **vague** prompt ("Is this a good support reply?") and once with the **rubric** prompt above. Both prompts get the same policy, order record and message.

```python
# code/agents/ch6b_judge.py (abridged)
def corrected(p_obs, tpr, tnr):
    den = tpr + tnr - 1
    if den <= 0.05:                                    # the judge is no better than a coin: refuse to correct
        return float('nan')
    return min(1.0, max(0.0, (p_obs + tnr - 1) / den))

tp, fn, fp, tn = confusion(calib)                      # judge verdicts against the labels of the calibration set
tpr, tnr = tp / (tp + fn), tn / (tn + fp)
p_obs = sum(r['j'] == 'pass' for r in prod) / len(prod)  # raw pass rate on the production set
theta = corrected(p_obs, tpr, tnr)
lo, hi, _ = bootstrap(calib, prod)                     # resample both sets 2,000 times, keep the middle 95%
```

[![Terminal output of ch6b_judge.py: the vague judge on the calibration set has TP 15, FN 5, FP 0, TN 20, accuracy 87.5 percent, TPR 75 percent, TNR 100 percent; on the production set its raw pass rate is 55.0 percent, corrected 73.3 percent, bootstrap interval 50.0 to 100.0 percent, against a true 70.0 percent. The rubric judge has TP 14, FN 6, FP 0, TN 20, accuracy 85.0 percent, TPR 70.0 percent, TNR 100 percent, 3 unparsed verdicts; raw 57.5 percent, corrected 82.1 percent, interval 55.0 to 100.0 percent; one false pass, a wrong amount. 160 calls cost 0.0428 dollars](/img/agents/ch6b_judge-run.png)](/img/agents/ch6b_judge-run.png)

{{FIG:b_judge_results|Results of ch6b_judge.py (gpt-5-mini, one run, 40 labelled replies per set). Both judges caught every failing reply in the calibration set (TNR 100%) but failed 5 or 6 of the 20 good ones. On the production set, the raw pass rates (55.0% and 57.5%) are far below the true 70%; the corrected estimates (73.3% and 82.1%) are closer or overshoot, and their 95% intervals are wide.}}

**The judge is strict, not lenient.** Both prompts caught all 20 planted defects in the calibration set: a TNR of 100%. Their errors were all false fails: the vague judge failed 5 of the 20 good replies (TPR 75%), the rubric judge 6 (TPR 70%). So the raw production pass rates, 55.0% and 57.5%, understate the true 70% by 12 to 15 points. A team reading the raw number would think its agent is much worse than it is.

**The errors cluster, which error analysis finds at once.** Reading the false fails shows the pattern. Across both sets, 8 of the vague judge's 11 false fails are the short reply about a damaged item that ends "Keep the item." The policy says "no return needed", but the judge reads "keep the item" as an invented promise. The rubric judge's reasons say the same thing in its own words ("C3: agent said keep the item..."). This is the most useful result of the experiment: the judge has one specific misreading, and one sentence in the rubric ("'no return needed' means the customer keeps the item") would probably fix most of it. We did not run that second version; it is Exercise 1.

**The correction helps, and its interval tells the truth.** For the vague judge, the corrected rate is 73.3% against a true 70%, with a 95% bootstrap interval from 50.0% to 100.0%. For the rubric judge it overshoots to 82.1% (interval 55.0% to 100.0%). The reason is the assumption under the formula. On the production set the rubric judge actually passed 22 of 28 good replies, a TPR of 78.6%, not the 70% measured on 20 calibration labels; and it let one failing reply through there, which it never did in calibration. The correction used rates that were off by chance, so it overcorrected. With only 20 good and 20 bad labels, TPR and TNR are uncertain, and the interval is wide. Forty labels are enough to find the judge's main misreading; they are not enough to report a pass rate to two digits.

**The rubric was not better here, and it was less robust.** The rubric prompt produced 3 verdicts that the parser could not read (the reasoning used up the output), and the script counts those as FAIL. It also made the only false pass in the run: a production reply that refunded the full 39.95 USD on a normal return, skipping the 6.95 USD fee, although the rubric says "compute any amount yourself". A rubric makes the judge's reasons readable and its criteria explicit; it does not by itself make the judge more accurate. Only the measurement tells you that.

> [!WARNING]
> What this experiment shows, and what it does not. It is one run of one model on 80 synthetic replies whose labels we set ourselves. The defects are planted, so they may be easier or harder than real ones. The direction of the errors (strict here, lenient in Chapter 4's development run) depends on the model, the prompt and the data. What carries over is the procedure: label a set with both passes and failures, measure TPR and TNR separately, read the disagreements, fix the judge prompt, and report the corrected rate with an interval. The whole run, 160 judge calls, cost 0.0428 USD.

> [!TIP] In production
> Keep the judge's labelled set as a test of the judge itself. Each time you change the judge prompt or the judge model, rerun it and compare TPR and TNR before you trust any new score. Split the labels into a development part, which you read while you fix the prompt, and a held-out part, which you only score. Otherwise the judge ends up tuned to the very examples that measure it. Add new labels from production regularly, because the outputs drift and the judge's error rates drift with them.

### The research behind judge calibration

Zheng et al. measured agreement, the share of items where judge and human give the same verdict. Agreement is accuracy under another name, and it hides the split between the two errors. For a judge that grades your own application, the confusion-matrix view is more useful: label a set by hand, measure how often the judge accepts good outputs and rejects bad ones, and change the judge prompt until both are high enough for the decision at hand. The correction formula is much older than LLM judges. It is the Rogan-Gladen estimator from epidemiology (Rogan and Gladen, 1978), which corrects a disease prevalence measured with a test of known sensitivity (our TPR) and specificity (our TNR).

Two cautions from the same literature. First, the correction can return values below 0 or above 1 when the sample is small or the error rates are wrong; the script clips to the range and the bootstrap shows how unstable the estimate is. Second, when the judge and the generator are the same model, their errors are not independent: a judge that shares the generator's blind spot will pass exactly the failures that matter. Using a different model for the judge, or a code check where one exists, reduces that risk.

### Discussion

A judge is a measuring instrument, and an instrument without a calibration record is a guess with a number attached. The two rates matter more than accuracy: a strict judge makes your agent look worse than it is and sends reviewers after good outputs; a lenient one hides failures that customers will see. Small labelled sets are enough to find a judge's systematic misreadings, which is the most valuable thing calibration gives you. They are not enough for precise pass rates, so report intervals.
