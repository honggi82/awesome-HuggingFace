# arXiv:2605.26045v1[cs.CL]25May2026

## Confidence and Calibration of Activation Oracles for Reliable Interpretation of Language Model Internals

Federico Torrielli University of Turin federico.torrielli@unito.it

### Peter Schneider-Kamp

University of Southern Denmark petersk@imada.sdu.dk

### Lukas Galke Poech

University of Southern Denmark galke@imada.sdu.dk

### Abstract

Activation oracles aim to make the activations of other models legible to humans and yield promising results compared to white-box interpretability techniques. However, uncertainty quantification (UQ) for the natural-language outputs of such activation oracles is so far understudied. Here, we investigate 6 different methods for estimating the confidence of activation oracles and evaluate how well-calibrated their confidence scores are. Our experiments on 6,000 samples per oracle (varying verbalizer and context prompts) reveal that bootstrap mode frequency is the best-calibrated method among those tested (ECE 5.7% vs. 25.5% for the answer-word log-probability on Qwen3-8B; 10.3% vs. 13.1% on Qwen3.6-27B), and that the log-prob baseline can serve as a fast triage signal at a fraction of the cost.

Code and the patched trainer are available at https://github.com/federicotorrielli /probabilistic_activation_oracles.

### 1 Introduction

An activation oracle is a language model fine-tuned to translate a target model’s hidden state (fig. 1, top) into a natural language description (Karvonen et al., 2025). The activations of the target model are ingested into the residual stream of the oracle at selected positions and layers. The oracle is commonly instantiated as a base model equipped with a low-rank verbalizer adapter, trained on a mixture of tasks: latent question-answering, binary classification, and self-supervised context prediction, using placeholder positions that stand in for the target’s activations. Through this training procedure, the oracle gains the capability to map the target model’s neural activity into natural language. Activation oracles can thereby recover information that lives in the target’s weights and surfaces only in its activations, such as a memorized biographical fact, a

hidden objective, or a secret word that should not be revealed.

However, the output of an activation oracle is merely a natural-language utterance without any notion of certainty or confidence. This matters for the downstream use cases oracles are being proposed for: alignment auditing (Bricken et al., 2025; Sheshadri et al., 2026), deception detection (Ravindran, 2025), and elicitation of hidden objectives (Dietz et al., 2026). Each of these is an actionable decision that needs a probability: an auditor flags or releases a model, a monitoring pipeline routes or drops a generation, a release process clears or blocks a checkpoint. An oracle that asserts “the secret word is tree” with no notion of its own confidence is either always trusted or never trusted, and the highly-confident-but-wrong answers are exactly the ones a deployed pipeline would act on.

Standard ways of attaching confidence to language model outputs were developed on un-steered models. An activation oracle is a steered model: Its residual stream is overwritten at inference time at one layer and a few positions with a vector drawn from another model. Whether the standard recipes still hold under that perturbation has not, to the best of our knowledge, been measured.

Here, we set out to identify the best way to equip activation oracles with confidence estimation methods and test their calibration: For this, we benchmark six confidence estimation methods on the secret-word taboo task of Karvonen et al. (2025) on two oracles: the released Qwen38B oracle of Karvonen et al. (2025) and a newly trained Qwen3.6-27B oracle. Three of the methods are established baselines that we adapt to the steered setting: the joint token log-probability of the predicted word; temperature-bootstrap mode frequency, which can be considered a short-answer analogue of self-consistency (Wang et al., 2022)); and free-form numeric self-report (Kadavath et al., 2022; Lin et al., 2022). The three further methods

TA R G E T

O R A C L E

###### Target model

###### Oracle

A language model fine-tuned to hide a secret word.

A second language model trained to read the target’s hidden state.

hidden state h ∈ ℝd, one layer

…

+0.93 −0.41 +0.17 −1.32 +0.55 +0.11 −0.38 +1.74 −0.19 +0.87 −1.62 +0.27 +0.08 −0.95

layer 48 secret: “tree”

#### “ tree ” ?

read at layer 48, injected at the oracle’s placeholder positions

…but how sure of that?

###### An AI reads another AI’s hidden thoughts — and decodes an answer. But how confident is the read? Six methods, on two oracles.

Calibration error on the secret-word task Qwen3-8B oracle Qwen3.6-27B oracle well-calibrated (ECE ≤ 0.10)

0.057 CALIBRATED

###### Bootstrap

sample 20 answers; agreement = confidence ✓best on both

0.103

0.255

###### Log-prob

decent

joint token probability of the answer

0.131

0.264

###### MCMC agreement †

decent

10 power-sampling chains; agreement = confidence

0.390

0.404

###### Steering sens. †

poor

vary injection strength; stability = confidence

0.501

0.544

###### MCMC acceptance †

poor

MH chain acceptance rate as confidence

0.740

0.582

###### Direct (numeric)

ask the oracle ‘how confident are you?’ ✗anti-calibrated

0.753

0 0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80

Expected Calibration Error (0 = perfectly calibrated · 1 = worst possible)

Figure 1: Top: an activation oracle reads another model’s hidden state and decodes an answer but emits no calibrated confidence. Bottom: expected calibration error for the six UQ methods we evaluate. †: methods introduced in this work.

are new to this work: The first reads the acceptance ratio of an MCMC power-sampling chain (Karan and Du, 2026) on the steered oracle. The second runs k such chains and reads cross-chain agreement. The third sweeps the steering coefficient over a small grid and reads decoding stability, inspired by the uncertainty/steerability correlation of Zur et al. (2025). Each method emits an answer and a confidence in [0,1]. We score sixteen method-temperature configurations (six bootstrap temperatures, three each for the two MCMC variants, two log-probability variants, and one configuration each for direct self-report and steering sensitivity) against accuracy, expected calibration error (ECE), Brier score, NLL, and AUROC on 6,000 samples per configuration per oracle (20 target words × 3 verbalizer prompts × 100 context prompts).

Our results show that the best-calibrated method on both oracles is the temperature bootstrap at a sampling temperature T near 1.0. The ECEoptimal T tracks task accuracy and migrates flatter on the harder model: T=1.0 when the oracle is right 40% of the time (8B, ECE 5.7%); T=1.5 when it is right 20% of the time (27B, ECE 10.3%). The worst-calibrated method on both oracles is freeform numeric self-report; on the 27B oracle the model is on average more confident in its wrong answers than its right ones (mean confidence 98.9% vs. 97.6%). This replicates the probe-vs-verbalized

confidence gap that Yuan et al. (2026) and Miao and Ungar (2026) report on standard LLMs in the activation-oracle setting studied here. The MCMC acceptance ratio fails for a different reason: the steered conditional distribution is sharply peaked at the greedy answer, so the chain accepts most proposals on correct and on wrong outputs alike (AUROC 0.53–0.60).

##### Contributions.

- • The first UQ benchmark for activation oracles. Six methods, two models, four calibration/ranking metrics, a controlled target-setscaling variant; 6,000 samples per configuration per oracle.
- • Six UQ methods evaluated on activation oracles for the first time. Three are adaptations of established LLM-UQ baselines (answer-word logprobability, temperature bootstrap, free-form numeric self-report); three are designed specifically for the steered-oracle setting (MCMC powersampling acceptance, MCMC power-sampling agreement, and steering-coefficient sensitivity). We give an intuition for why two of the three steered-oracle-specific methods (MCMC acceptance and steering sensitivity) yield negative results.
- • A practical recipe: pick the bootstrap temperature so the mean mode frequency on a held-out word slice matches the empirical accuracy on that slice. On our two oracles this rule picks T=1.0 and

- T=1.5, matching the ECE optimum.
- • A replication, in the activation-oracle setting, of the probe-vs-verbalized confidence gap reported by Yuan et al. (2026); Miao and Ungar (2026): free-form numeric self-report is anti-calibrated on the 27B oracle.
- • The first activation oracle for a hybrid linearplus-full attention architecture. We adapt the activation-oracle trainer of Karvonen et al. (2025) (LatentQA together with binary classification and self-supervised past-lens context prediction) to Qwen3.6-27B and release the verbalizer weights and the patched trainer.

### 2 Background and Related Work

Activation oracles. An activation oracle is a language model that has been fine-tuned to read another model’s hidden state. Mechanically: take a small number of placeholder tokens in the oracle’s input prompt; at one designated injection layer, intercept the oracle’s residual stream and overwrite it at those placeholder positions with vectors collected from a target model’s residual stream (Karvonen et al., 2025). The oracle is trained via a LoRA adapter to treat those overwritten positions as a representation of the target’s hidden state, and to answer natural-language questions about it:

“What is this text about?”, “What is the model’s goal?”, “What word was the target trained to keep secret?”. The injection is norm-matched so that the magnitude of the inserted vector follows the magnitude of the residual it replaces; this keeps the post-injection norm in distribution. We denote the steering coefficient by c, defaulting to 1.

Power sampling. Karan and Du (2026) sample from the unnormalized power distribution p(x)α with α > 1. The power distribution is sharper than the base distribution p where the base is already peaked, but, importantly, it preserves multimodal structure that naïve low-temperature sampling collapses. They implement it as blockwise Metropolis–Hastings: at each block they propose a continuation from a low-temperature proposal and accept or reject based on the ratio pα(prop)/pα(curr), corrected for the proposal asymmetry. On math, code, and reasoning benchmarks the result matches or beats RL post-training while preserving sample diversity. We use the same block-MH procedure on a steered oracle and read confidence in two ways. First, off the empirical acceptance ratio of a single chain: a sharply unimodal

steered posterior accepts almost every proposal (acceptance near 1), while a multimodal posterior produces frequent rejections as the chain crosses between modes. Second, off cross-chain agreement over k independent chains, the multi-chain analogue of self-consistency (Wang et al., 2022) applied to power sampling.

Uncertainty for language models. Guo et al. (2017) document that modern neural networks are systematically miscalibrated and that post-hoc temperature scaling reduces ECE; we do no posthoc rescaling so the calibration numbers we report are the methods’ native calibration. Kadavath et al. (2022) show that large pretrained LLMs assign well-calibrated probabilities to the correct option on multiple-choice questions, and that the model’s own probability of the token “True” when prompted to evaluate its proposed answer (their P(True) score) is informative and few-shot calibrated; RLHF-tuned policies on the same base appear off-the-shelf miscalibrated but recover under a single global temperature rescale. Lin et al. (2022) train models to verbalize calibrated confidence on arithmetic, but the calibration does not transfer to held-out task families without further training. Wang et al. (2022) introduce self-consistency for chain-of-thought: sample k traces, take the modal final answer, read the agreement rate as a confidence. Our temperature-bootstrap method is the short-answer analogue, applied to a steered oracle where the “trace” is just the answer word. Kuhn et al. (2022) compute entropy over semantic equivalence classes of free-text answers, using NLI to cluster paraphrases. We use the closed taboo vocabulary as a hard equivalence relation, which is a coarse but exact stand-in: two outputs are deemed equivalent iff our extractor returns the same target word, which sidesteps the NLI step but does not collapse near-synonyms outside the vocabulary (e.g., outputs containing “rock” and “stone” map to distinct classes).

Activation steering. Several methods edit a transformer’s residual stream by adding a fixed vector at one or more layers: ActAdd (Turner et al., 2024), contrastive activation addition (Rimsky et al., 2024), and the broader representationengineering program of Zou et al. (2025). Activation oracles share the mechanical setup (a controlled intervention on the residual stream) but invert the direction of use: the inserted vector is used to steer a separately trained verbalizer instead of

being used to change the target model’s behavior.

Probe vs. verbalized confidence. Yuan et al. (2026) and Miao and Ungar (2026) report that the activation-level uncertainty signal in standard LLMs is largely inaccessible through the model’s verbalized confidence. A linear probe on hidden states can predict trace correctness at AUROC up to 0.95 even when the model verbally insists it is confident (Yuan et al., 2026); the verbalizedconfidence direction lies orthogonal to the calibration direction in activation space (Miao and Ungar, 2026). We observe the same gap in the activationoracle setting (section 5.4), with a recovery direction (constrained label scoring) that bypasses the free-text decoding channel. Basu et al. (2026) report a parallel failure on a different task: a linear probe identifies clinical hazards at 98% AUROC, but four mechanistic interventions designed to act on that signal close less than a quarter of the resulting knowledge-action gap. The activation oracle is a third instance of the same shape.

Linear representations. Park et al. (2024) formalize the linear-representation hypothesis and connect it to linear probing and steering; Tigges et al. (2023) demonstrate the assumption empirically for sentiment, showing a single direction that is both probe-readable and causally relevant. An activation oracle’s single-layer read implicitly relies on this: one residual-stream slice at one layer is assumed to carry the relevant concept direction. We observe this empirically: only one of the three training-time read layers (25%,50%,75% depth) carries a usable secret-word signal at inference on our 27B model (section E).

### 3 Methods

Setup. We use the 20-word taboo task of Karvonen et al. (2025): 20 LoRA adapters that fine-tune the target model, each one trained to keep a different word secret while still behaving as a conversational assistant. For each (target word, context prompt, verbalizer prompt) triple, the target encodes a clueelicitation prompt (e.g., “Hint me”) under its taboo adapter; we read the residual stream at the trailing K token positions of one designated layer and inject those vectors into the oracle’s residual stream at the placeholder positions of one designated injection layer. We use the same 20 words, 100 context prompts, and three verbalizer prompts as Karvonen et al. (2025). The steered oracle then emits a free-

text reply. Each of the six UQ methods we evaluate produces aˆ, the first taboo-vocabulary word in its decoded output, and a confidence c ∈ [0,1]. Calibration is measured against 1[ˆa = a⋆], with a⋆ being the secret word.

(M1, baseline) Log-probability of the answer word. Decode greedily under the steering hook. To align subword tokens to aˆ we accumulate each generated token’s decoded character span, locate aˆ in the accumulated text via a word-boundary regex, and select the contiguous tokens whose spans overlap the match. The confidence is the joint probability of those tokens:

clp =

p(xt | x<t).

t∈tok(ˆa)

This adapts the self-evaluation family of Kadavath et al. (2022) to the answer word. An offset-free variant (joint probability of the first |tok(ˆa)| generated tokens) agrees within 0.02 AUROC (section A).

(M2) Temperature bootstrap. Draw k=20 samples at temperature T, normalize each to its first taboo-vocabulary word, and report the modal answer with confidence equal to its empirical frequency:

cboot = k1

k

1[ˆai = mode(ˆa1:k)],

i=1

where aˆi is the first taboo-vocabulary word in the i-th sample. This is the short-answer analogue of self-consistency (Wang et al., 2022). We sweep T ∈ {0.3,0.5,0.7,1.0,1.3,1.5}.

- (M3) Direct numeric self-report. We elicit a direct numeric self-report via two turns: Turn 1 greedy-decodes the answer under the hook; turn 2 appends “On a scale of 0 to 100, how confident are you?” and greedy-decodes under the same hook. The response is parsed as an integer in [0,100] and divided by 100.
- (M4) MCMC power-sampling acceptance. A single block Metropolis-Hastings (MH) powersampling chain on the steered oracle (Karan and Du, 2026): B=4 blocks of 5 tokens, S=5 MH steps per block, and three power values α = 1/T with T ∈ {0.5,0.25,0.125}. Each step picks a random position in the generated suffix, resamples from that position to the end of the current block under the low-temperature proposal qT, and

- accepts under the standard MH ratio. The confidence is the empirical acceptance rate cmh = #{accepted}/(B · S). Acceptance saturates when the steered posterior is mode-peaked and drops when it is multimodal; we test whether this tracks correctness.
- (M5) MCMC power-sampling agreement. We run k=10 independent chains with different seeds; apply the M2 mode-frequency readout to the outputs. Same T grid as M4. In theory, power sampling should preserve multimodal diversity that collapses under temperature-scaling. Cross-chain agreement should thus sharpen uncertain items.
- (M6) Steering-coefficient sensitivity Greedy decode at coefficients c ∈ {0.5,0.75,1.0,1.25,1.5} around the default c=1.0; report the modal answer with confidence equal to its mode frequency. Following Zur et al. (2025): a committed activation should decode the same word at every nearby c.

### 4 Experimental Setup

Models and oracles. We evaluate two activation oracles on the secret-word taboo task: the Qwen38B oracle released by Karvonen et al. (2025) and a Qwen3.6-27B oracle we train and release with this paper (see section L). The Qwen3.6-27B base is a hybrid Gated DeltaNet plus Gated Attention architecture that the upstream LatentQA trainer of Karvonen et al. (2025) did not support out of the box; we patched the LoRA target selection and the attention backend to make training converge to comparable out-of-distribution performance. Target adapters for both oracles are the 20 taboo LoRAs from Karvonen et al. (2025) for 8B and our re-trained equivalents for 27B.

Layer readout. We inject activations at layer 1 of the oracle and read from a single layer of the target. For Qwen3-8B we read at 50% depth (layer 18 of 36), matching Karvonen et al. (2025). For Qwen3.6-27B the oracle is trained on activations from {25%,50%,75%} depth, but at inference time only the 75% read (layer 48 of 64) carries a usable secret-word signal: layer 32 yields 6% task accuracy versus 41% at layer 48. Section E reports the layer sweep and discusses why we believe the hybrid architecture is the cause.

Prompts and protocol. We use the 100 context prompts and three verbalizer prompts from Karvonen et al. (2025). Each triple of (target word,

context prompt, verbalizer prompt) makes one sample, for 20 × 100 × 3 = 6,000 samples per method per model. For each sample we run all six methods at every temperature in their grid: 6 temperatures for bootstrap, 3 each for MCMC acceptance and MCMC agreement, 2 variants for log-prob (with and without character-offset alignment, see M1), and 1 each for direct numeric self-report and steering sensitivity, for 16 method/temperature rows per oracle. Hardware, software, and runtime are reported in section M.

Evaluation metrics. Accuracy is exact-match between aˆ and a⋆. ECE (Naeini et al., 2015) with 10 equal-width bins measures calibration, i.e., whether the model is 80%-confident when it attains mean 80% accuracy? Brier score (Brier, 1950) and negative log-likelihood are “proper” scoring rules: their expectation is minimized when the reported probability matches the true label probability, so they jointly penalize miscalibration and poor discrimination and cannot be trivially gamed by a constant predictor. AUROC of confidence-as-predictor measures ranking: do high-confidence outputs outrank low-confidence ones? A method can game ECE by emitting a constant near the overall accuracy, so we report both axes of the trade-off.

### 5 Results

##### 5.1 Method scorecard

Table 1 reports the five considered metrics (accuracy, ECE, Brier, NLL, AUROC) for the eight mostinformative methods on both models, sorted by average rank across the five metrics. Section A provides the full results.

Three patterns recur across both models. The best ECE is achieved by a bootstrap variant. The best AUROC is achieved either by log-prob or by a low-temperature bootstrap variant. The worst ECE and AUROC both come from direct numeric self-report and from raw MCMC acceptance.

Fitting a post-hoc calibrator (temperature scaling, Platt, isotonic, or beta) on a held-out word slice substantially closes the ECE gap between methods (section B): isotonic-rescaled log-prob and isotonicrescaled bootstrap T=0.7 are within 0.01 ECE on 27B. Bootstrap retains a small advantage when no held-out labels are available; once labels are available for fitting a rescale, the choice between log-prob and bootstrap is closer to a cost/latency question (one decode vs. twenty) than a calibrationquality question. The post-hoc-calibrated tables are

Qwen3-8B Qwen3.6-27B

Method Acc ECE Brier NLL AUROC Acc ECE Brier NLL AUROC Log-prob (with offset, baseline) .414 .255 .246 0.730 .840 .234 .131 .158 0.489 .835

- Bootstrap T=0.5 .424 .193 .213 1.029 .810 .231 .327 .222 0.800 .864

- Bootstrap T=0.7 .414 .097 .171 0.558 .830 .230 .218 .157 0.514 .863

- Bootstrap T=1.0 .402 .057 .163 0.498 .829 .225 .147 .130 0.431 .851

- Bootstrap T=1.3 .387 .076 .171 0.519 .823 .209 .125 .125 0.415 .837 Bootstrap T=1.5 .367 .083 .173 0.522 .824 .206 .103 .129 0.422 .812 MCMC agreement T=0.5 .413 .263 .247 1.608 .803 .228 .390 .274 1.383 .858 Steering sensitivity .418 .404 .354 4.401 .763 .232 .501 .413 4.172 .725 MCMC accept T=0.125 .415 .544 .534 10.77 .563 .234 .740 .729 15.61 .528 Direct (numeric) .414 .582 .580 12.66 .516 .234 .753 .752 16.02 .404

- Table 1: Calibration scorecard, n=6,000 samples per row per model. Lower is better for ECE, Brier, NLL; higher for accuracy and AUROC; best per column in bold. Log-prob is the established LLM-UQ baseline (M1). Bootstrap dominates ECE/Brier/NLL on both models; log-prob has the highest AUROC on 8B. The full 16-row table is in section A.

Qwen3-8B Qwen3.6-27B

T Acc ECE Br. AUR. Acc ECE Br. AUR.

- 0.3 .419 .334 .303 .784 .233 .479 .369 .837

- 0.5 .424 .193 .213 .810 .231 .327 .222 .864

- 0.7 .414 .097 .171 .830 .230 .218 .157 .863

- 1.0 .402 .057 .163 .829 .225 .147 .130 .851

- 1.3 .387 .076 .171 .823 .209 .125 .125 .837

- 1.5 .367 .083 .173 .824 .206 .103 .129 .812

- Table 2: Bootstrap temperature sweep, k=20 samples per item. The ECE optimum sits at T=1.0 on 8B (overall accuracy 0.40) and T=1.5 on 27B (overall accuracy 0.21). Br.: Brier. AUR.: AUROC.

##### 5.3 Power sampling does not add information

Single-chain MCMC acceptance ratio does not separate correct from wrong predictions on this task (AUROC 0.53–0.60, ECE 0.54–0.74 across the three proposal temperatures; table 1). The steered oracle’s distribution is mode-peaked at the greedy decode: the verbalizer assigns high probability to the correct answer, and at α > 1 the power distribution sharpens this spike further. MH proposals from the low-temperature proposal distribution rarely propose anything that would be rejected, so acceptance saturates regardless of whether the spike sits at the correct word. The multi-chain agreement variant (M5) reaches competitive AUROC on 27B at 5× the wall-clock of bootstrap and without improving ECE or Brier; the matched-cost comparison and per-temperature breakdown are in section H. The natural extension of Karan and Du (2026) to the activation-oracle UQ setting therefore does not transfer: in a reasoning task the base model’s distribution is multimodal across candidate reasoning paths and α-sharpening picks the high-quality mode; in our setting the conditional distribution given the injected activation is approximately unimodal, so the MH correction adds noise.

in section B, and bootstrap 95% CIs over n=6,000 resamples in section C.

5.2 Bootstrap temperature is tied to task accuracy

- Table 2 sweeps the bootstrap temperature on both models. On 8B (overall accuracy ≈ 0.41), ECE consistency displays a minimum at T=1.0. On 27B (overall accuracy ≈ 0.22), ECE monotonically decreases through T=1.5, the largest temperature we tested, with Brier and NLL minima at T=1.3.

The mechanism is direct: the mode frequency over k samples is an estimator of the probability that a random decode lands on the modal answer. Calibration to the binary correctness signal is best when that estimator matches the per-item accuracy distribution. On 8B, T=1.0 yields a mean mode frequency of 0.40 against an empirical accuracy of 0.40; on the harder 27B oracle the optimum migrates toward flatter sampling (T=1.5 gives mean mode frequency 0.25 against accuracy 0.21, and ECE is still falling at the largest temperature we tested). The tuning strategy we recommend is picking T in such a way that the mean mode frequency on a held-out word slice matches the empirical accuracy on that slice.

##### 5.4 Direct self-report is anti-calibrated on the larger model

Direct numeric self-report assigns nearly identical confidences to correct and wrong predictions on both models. The gap between mean confidence on correct and on wrong is +0.003 on 8B (mean 0.998 correct vs. 0.995 wrong) and −0.013 on 27B (mean 0.976 correct vs. 0.989 wrong). The 27B oracle is, by a small margin, more confident in wrong predictions than in correct ones. AUROC is 0.516 on 8B and 0.404 on 27B, the latter below

[Figure 1]

[Figure 2]

Figure 2: Calibration vs. ranking on Qwen3-8B (top) and Qwen3.6-27B (bottom). Each point is one UQ method; marker size encodes accuracy. The bootstrap family (green) occupies the low-ECE, mid-AUROC region; log-prob (blue) sits at high AUROC with mid ECE. Direct self-report and raw MCMC sit in the high-ECE, low-AUROC corner. The Pareto front (low ECE, high AUROC) is governed by bootstrap and log-prob on both models.

chance. ECE is 0.582 on 8B and 0.753 on 27B.

We attribute the failure to the language-modeling prior dominating the elicitation channel: the question “how confident are you?” has a stronglyattested modal answer (“very confident”, “100”) in the model’s training distribution. The steering hook fires during prefill of the second turn, so the activation-grounded posterior is in principle available; the chat-protocol prior overrides that posterior in the generated text. This matches the probe-vsverbalized gap that Yuan et al. (2026) report for chain-of-thought reasoning models (linear probes on hidden states predict trace correctness at AUROC ≈ 0.95 while verbalized confidence is statistically indistinguishable between right and wrong traces) and the orthogonality result of Miao and Ungar (2026) (the verbalized-confidence direction lies roughly orthogonal to the calibration direction in activation space). The mechanism is the same here: the steered activation carries a usable confidence signal, but the natural-language elicitation

[Figure 3]

Figure 3: Per-word accuracy on Qwen3-8B for selected UQ methods. Rows are target words sorted by mean accuracy. The bootstrap and log-prob families have closely correlated per-word accuracy; direct self-report is constant. Per-word accuracy varies by an order of magnitude on this task.

channel does not preserve it.

To test whether the activation-grounded uncertainty is reachable through a different elicitation channel, we ran a pilot on 8B in which the model is asked “Reply with exactly one of: very low, low, medium, high, very high” and we score the five labels with constrained logits (no free generation). On a n=30 slice (6 words × 5 context prompts, one verbalizer prompt) the expected-value readout achieves AUROC 0.957; the P(very high) readout achieves AUROC 0.957 with a 3× wider confidence gap. Section G reports the full pilot.

The pilot is too small to draw a settled claim from (AUROC standard error at n=30 is roughly 0.05), but the magnitude of the gap and its mechanism (i.e., reading the label distribution) suggest that introspective UQ on a steered oracle is reachable, provided one bypasses the free-form text channel that the standard self-report prompt uses.

##### 5.5 Per-word and target-set-size scaling

Aggregate metrics in table 1 marginalize over the 20 target words. The per-word breakdown for Bootstrap T=1.0 is plotted in fig. 3. On 8B, the accuracy spread is 0.12 (rock) to 0.83 (moon). On 27B it is 0.05 (leaf) to 0.50 (blue). Of the 20 words, two are easy on both models (moon, snow), five are hard on both (leaf, wave, clock, song, rock), and one (blue) inverts: from worst-three on 8B to best-one on 27B.

The wrong-answer failure mode also differs between models. On 8B, when the oracle is wrong, the modal wrong answer is often a topical neighbor of the secret word: flame → fire; rock → tree or stone; cloud → sky; song → music. On 27B the modal wrong answer is empty output or one of secret, word, password. 14% of 27B generations are empty; topical confabulation is rare. The 8B failure mode is harder to detect from confidence alone (the oracle is committing to a plausible alternative), which is part of why 8B AUROC is lower than 27B AUROC for every bootstrap temperature.

We also run a controlled target-set-size variant in which the 20-word vocabulary is restricted to N ∈ {2,5,10,20} with a fixed seed. Bootstrap T=1.0 is the only method whose ECE improves

- as N grows on 8B (ECE improves from 21.3% at N=2 to 5.7% at N=20). Most other methods degrade over the same sweep: steering-sensitivity ECE rises from 16.6% to 40.4%, and MCMCagreement T=0.5 ECE from 4.7% to 26.3%. On

- 27B the ECE of every method shifts by at most ∼ 6 pp across N, so N-scaling is not informative on the larger oracle. The full controlled-N tables are in section D. 6 Discussion

Claims and their scope We present here three claims. (i) Among the six UQ methods we tested, bootstrap mode-frequency at T ≈ 1 has the lowest ECE, Brier, and NLL on the activation-oracle taboo task on two Qwen-family models, with the optimal T tied to the overall task accuracy. (ii) Raw MCMC acceptance ratio is no suitable confidence signal in this setting, with a mechanistic explanation (modepeakedness of the steered posterior). (iii) Freeform numeric self-report is anti-calibrated on the larger model and uninformative on the smaller one; a constrained linguistic readout reaches AUROC 0.957 on a small pilot.

Why scale buys calibration but not accuracy on this task Qwen3.6-27B has roughly half the secret-word accuracy of Qwen3-8B (0.22 vs. 0.41)

- at the layer-48 read. Yet the best-AUROC on 27B (0.864) exceeds that on 8B (0.840) and the bestBrier on 27B (0.125) is lower than on 8B (0.163). Two mechanisms contribute. First, the 27B oracle emits 14% empty answers, all scored as wrong, all with low first-token max probability; this pulls the wrong-pool mean confidence down toward the overall accuracy. Second, the 27B oracle refuses

harder words. Refusal is easier to detect from confidence than topical confabulation because it shows up as low log-prob on the generated tokens. Both mechanisms benefit calibration at the cost of useful signal: refusal is not the same as “I don’t know,” and a downstream auditor would prefer the topicalneighbor failure mode, because a topical neighbor still localizes the concept in semantic space (flame → fire tells the auditor the activation carries something fire-shaped), whereas refusal collapses to no information at all. Exact-match scoring also penalizes topical near-misses as hard as it penalizes refusals; an embedding-based soft accuracy (e.g., a BERTScore-style similarity against the secret word, Zhang et al., 2020) would credit the 8B failure mode more than the 27B failure mode and is a natural follow-up evaluation.

Recommendations For practitioners building on activation oracles:

- • Use bootstrap mode frequency as the production confidence signal. Tune T on a held-out word slice so that mean mode frequency matches empirical accuracy if validation data is available, else use T = 1.0.
- • Use log-prob word probability as a fast triage signal when k=20 samples is too expensive, with a held-out affine-rescale step.
- • Steering-coefficient sensitivity (M6) on the 5point grid we tested yields mode frequencies in {0.2,0.4,0.6,0.8,1.0}, which is too coarse to act as a calibrated probability (ECE ≥ 0.40 on both oracles). It remains usable as a binary selfagreement diagnostic.
- • MCMC acceptance ratio is not a confidence signal on this task; it can still serve as a sampling diagnostic to detect when a chain is not mixing.
- • Free-form numeric self-report does not yield usable confidences in our setting. If introspective UQ is needed, we instead recommend constrained label scoring or training an explicit confidence head.

### 7 Conclusion

We benchmarked six uncertainty-quantification methods on activation oracles over the secretword taboo task on Qwen3-8B and Qwen3.6-27B. Temperature-bootstrap mode frequency at sampling temperature near 1 is the best-calibrated UQ method on both models, with the optimal temperature tied to the overall task accuracy. MCMC power-sampling acceptance ratio does not separate

correct from wrong outputs because the steered oracle’s distribution is mode-peaked. Free-form numeric self-report is anti-calibrated on the larger model; a constrained linguistic readout recovers the signal on a small pilot. The actionable recipe is to pick the bootstrap temperature so that the mean mode frequency on a held-out word slice matches the empirical accuracy on that slice.

### 8 Limitations

We test two oracles, both Qwen-family, and one training recipe. The cross-model agreement of the bootstrap and power-sampling results is consistent with a paradigm-level claim, but is not a proof of generality across architectures.

Scoring is exact-match on the first taboovocabulary word in the output. The answerextraction pipeline is shared across methods, so method ranking is invariant to that choice; absolute accuracy numbers would shift under an embeddingbased soft metric (cf. §6).

The main scorecard reports native, uncalibrated metrics; a held-out Platt, isotonic, or beta rescale closes most of the ECE gap between methods (section B). AUROC is monotone-invariant, so the ranking ordering carries over. Confidence intervals from bootstrap resampling (section C) show that several within-family gaps (bootstrap T=0.7 vs. T=1.0 on 8B AUROC, MCMC agreement T=0.5 vs. bootstrap on 27B AUROC) are within overlapping intervals; the between-family gaps named in the main text are an order of magnitude larger than the CI widths.

### 9 Ethical Considerations

No human-subjects data is used. The taboo task uses 20 synthetic LoRA target adapters that implant a secret word; both the secret words and the context prompts are reused from Karvonen et al. (2025) and contain no PII.

Our research contributes to better calibration of activation oracles, which we do not expect to cause any extra harm. Instead, uncertainty quantification of activation oracles improves reliability and thereby the usefulness of activation oracles in downstream monitoring settings.

All artifacts we build on are released for research, and our use is consistent with that scope. The Qwen3.6-27B oracle, 20 retrained taboo target LoRAs, patched LatentQA trainer, and UQ benchmark code we release with this paper are intended

for the same research purposes. Compute and energy are reported in section M.

### Acknowledgments

This research was supported in part by the MIST project, funded by the Novo Nordisk Foundation under grant reference number NNF25OC0103204.

### References

Sanjay Basu, Sadiq Y. Patel, Parth Sheth, Bhairavi Muralidharan, Namrata Elamaran, Aakriti Kinra, John Morgan, and Rajaie Batniji. 2026. Interpretability without actionability: mechanistic methods cannot correct language model errors despite near-perfect internal representations.

Trenton Bricken, Rowan Wang, Sam Bowman, Euan Ong, Johannes Treutlein, Jeff Wu, Evan Hubinger, and Samuel Marks. 2025. Building and evaluating alignment auditing agents. Alignment Science Blog.

Glenn W. Brier. 1950. Verification of forecasts expressed in terms of probability. Monthly Weather Review, 78(1):1–3.

Florian Dietz, William Wale, Oscar Gilg, Robert McCarthy, Felix Michalak, Gustavo Ewbank Rodrigues Danon, Miguelito de Guzman, and Dietrich Klakow. 2026. Split personality training: revealing latent knowledge through alternate personalities.

Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. 2017. On calibration of modern neural networks. In Proceedings of the 34th International Conference on Machine Learning, pages 1321–1330. PMLR. ShortConferenceName: ICML.

Saurav Kadavath, Tom Conerly, Amanda Askell, Tom Henighan, Dawn Drain, Ethan Perez, Nicholas Schiefer, Zac Hatfield-Dodds, Nova DasSarma, Eli Tran-Johnson, Scott Johnston, Sheer El-Showk, Andy Jones, Nelson Elhage, Tristan Hume, Anna Chen, Yuntao Bai, Sam Bowman, Stanislav Fort, and 17 others. 2022. Language models (mostly) know what they know. arXiv preprint.

Aayush Karan and Yilun Du. 2026. Reasoning with sampling: Your base model is smarter than you think. In The Fourteenth International Conference on Learning Representations.

Adam Karvonen, James Chua, Clément Dumas, Kit Fraser-Taliente, Subhash Kantamneni, Julian Minder, Euan Ong, Arnab Sen Sharma, Daniel Wen, Owain Evans, and Samuel Marks. 2025. Activation oracles: training and evaluating LLMs as general-purpose activation explainers. arXiv preprint.

Lorenz Kuhn, Yarin Gal, and Sebastian Farquhar. 2022. Semantic uncertainty: linguistic invariances for uncertainty estimation in natural language generation. In Proceedings of the 11th International Conference on Learning Representations.

Meelis Kull, Telmo Silva Filho, and Peter Flach. 2017. Beyond sigmoids: how to obtain well-calibrated probabilities from binary classifiers with beta calibration. In Proceedings of the 20th International Conference on Artificial Intelligence and Statistics, pages 623– 631. PMLR.

Stephanie Lin, Jacob Hilton, and Owain Evans. 2022. Teaching models to express their uncertainty in words. Transactions on Machine Learning Research, pages 1–19.

Miranda Muqing Miao and Lyle Ungar. 2026. Closing the confidence-faithfulness gap in large language models.

Mahdi Pakdaman Naeini, Gregory F. Cooper, and Milos Hauskrecht. 2015. Obtaining well calibrated probabilities using bayesian binning. Proceedings of the 29th AAAI Conference on Artificial Intelligence, 29(1):1–7.

Kiho Park, Yo Joong Choe, and Victor Veitch. 2024. The linear representation hypothesis and the geometry of large language models. In Proceedings of the 41st International Conference on Machine Learning, pages 39643–39666. PMLR.

John C. Platt. 1999. Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. In Advances in Large Margin Classifiers, pages 61–74. MIT Press.

Santhosh Kumar Ravindran. 2025. Adversarial activation patching: a framework for detecting and mitigating emergent deception in safety-aligned transformers.

Nina Rimsky, Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger, and Alexander Turner. 2024. Steering llama 2 via contrastive activation addition. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 15504–15522, Bangkok, Thailand. Association for Computational Linguistics.

Abhay Sheshadri, Aidan Ewart, Kai Fronsdal, Isha Gupta, Samuel R. Bowman, Sara Price, Samuel Marks, and Rowan Wang. 2026. AuditBench: evaluating alignment auditing techniques on models with hidden behaviors.

Curt Tigges, Oskar John Hollinsworth, Atticus Geiger, and Neel Nanda. 2023. Linear representations of sentiment in large language models.

Alexander Matt Turner, Lisa Thiergart, Gavin Leech, David Udell, Juan J. Vazquez, Ulisse Mini, and Monte MacDiarmid. 2024. Steering language models with activation engineering.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V. Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. 2022. Self-consistency improves chain of thought reasoning in language models. In Proceedings of the 11th International Conference on Learning Representations.

Aojie Yuan, Zhiyuan Julian Su, Haiyue Zhang, Yi Nian, and Yue Zhao. 2026. Hidden error awareness in chain-of-thought reasoning: the signal is diagnostic, not causal.

Bianca Zadrozny and Charles Elkan. 2002. Transforming classifier scores into accurate multiclass probability estimates. In Proceedings of the Eighth ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pages 694–699. ACM.

Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, and Yoav Artzi. 2020. Bertscore: Evaluating text generation with BERT. In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020. OpenReview.net.

Andy Zou, Long Phan, Sarah Chen, James Campbell, Phillip Guo, Richard Ren, Alexander Pan, Xuwang Yin, Mantas Mazeika, Ann-Kathrin Dombrowski, Shashwat Goel, Nathaniel Li, Michael J. Byun, Zifan Wang, Alex Mallen, Steven Basart, Sanmi Koyejo, Dawn Song, Matt Fredrikson, and 2 others. 2025. Representation engineering: a top-down approach to AI transparency. arXiv preprint.

Amir Zur, Atticus Geiger, Ekdeep Singh Lubana, and Eric Bigelow. 2025. Are language models aware of the road not taken? Token-level uncertainty and hidden state dynamics. arXiv preprint. ArXiv:2511.04527 [cs.CL].

### A Full scorecard

Table 3 reports all sixteen method-temperature pairs (six bootstrap temperatures, three each for the two MCMC variants, two log-probability variants, and one row each for direct self-report and steering sensitivity) sorted by average rank across the five metrics, for both models.

### B Post-hoc calibration baselines

Tables 4 and 5 report test-set ECE after fitting four post-hoc calibrators on a held-out slice. Calibrators: Temperature (one-parameter logit rescale, Guo et al., 2017), Platt (two-parameter logistic regression on logit(p), Platt, 1999), Isotonic (nonparametric monotone, Zadrozny and Elkan, 2002), and Beta (logistic regression on [log p,−log(1 − p)], Kull et al., 2017). We report two splits: Worddisjoint fits on a random 10 of the 20 secret words and tests on the other 10 (cross-word generalization); Random 50/50 is sample-level. Random seeds: 1 (word) and 2 (sample). Implementation: scikit-learn 1.8 for Platt/isotonic; one-parameter scipy.optimize.minimize_scalar for temperature; logistic regression with custom features for beta.

Qwen3-8B Qwen3.6-27B

Method Acc ECE Brier NLL AUROC Acc ECE Brier NLL AUROC Log-prob (no offset, baseline) 0.414 0.256 0.249 0.748 0.824 0.234 0.131 0.157 0.484 0.842 Log-prob (with offset, baseline) 0.414 0.255 0.246 0.730 0.840 0.234 0.131 0.158 0.489 0.835

- Bootstrap T=0.3 0.419 0.334 0.303 2.636 0.784 0.233 0.479 0.369 2.367 0.837

- Bootstrap T=0.5 0.424 0.193 0.213 1.029 0.810 0.231 0.327 0.222 0.800 0.864

- Bootstrap T=0.7 0.414 0.097 0.171 0.558 0.830 0.230 0.218 0.157 0.514 0.863

- Bootstrap T=1.0 0.402 0.057 0.163 0.498 0.829 0.225 0.147 0.130 0.431 0.851

- Bootstrap T=1.3 0.387 0.076 0.171 0.519 0.823 0.209 0.125 0.125 0.415 0.837

- Bootstrap T=1.5 0.367 0.083 0.173 0.522 0.824 0.206 0.103 0.129 0.422 0.812 Steering sensitivity 0.418 0.404 0.354 4.401 0.763 0.232 0.501 0.413 4.172 0.725 MCMC accept T=0.125 0.415 0.544 0.534 10.77 0.563 0.234 0.740 0.729 15.61 0.528 MCMC accept T=0.25 0.403 0.547 0.532 10.28 0.579 0.230 0.738 0.722 15.04 0.547 MCMC accept T=0.5 0.376 0.551 0.531 9.358 0.601 0.211 0.746 0.722 13.97 0.574 MCMC agreement T=0.125 0.415 0.493 0.461 7.823 0.669 0.236 0.668 0.620 10.87 0.653 MCMC agreement T=0.25 0.418 0.408 0.371 4.673 0.737 0.233 0.569 0.482 5.658 0.767 MCMC agreement T=0.5 0.413 0.263 0.247 1.608 0.803 0.228 0.390 0.274 1.383 0.858 Direct (numeric) 0.414 0.582 0.580 12.66 0.516 0.234 0.753 0.752 16.02 0.404

- Table 3: Full method-temperature scorecard, n=6,000 samples per row. ECE/Brier/NLL: lower is better. Accuracy/AUROC: higher is better. Best per column in bold.

Two observations. First, all four calibrators close most of the absolute ECE gap on every method, including the two anti-calibrated methods (direct, raw MCMC acceptance); after isotonic or beta rescaling on the random-split fit slice, these reach ECE ≈ 0.01 on 27B. The closing is partly due to the calibrator absorbing what is effectively a label-trained classifier on top of an uninformative score. AUROC, which is monotone-invariant, is unaffected (it actually rises on direct from 0.40 to ≈ 0.59 under Platt because the logistic regression learns a negative slope, flipping the sign of the anti-correlated score). In the end, the uncalibrated bootstrap remains the best label-free signal; once labels are available, the choice between methods is dominated by cost (a single decoding pass vs. twenty samples) rather than calibration quality.

Second, the word-disjoint split is uniformly harder than the random split: word identity carries systematic difficulty (some words refuse, some confabulate) that a calibrator fit on one subset cannot fully transfer. The largest word-vs-random gap is on bootstrap T=0.7 on 8B (isotonic ECE 0.102 word-disjoint vs. 0.038 random), which we read as the calibrator overfitting word-level frequency structure. Bootstrap T=1.0 is robust on both splits (≤ 0.041).

### C Bootstrap 95% CIs

Table 6 reports 1000-resample bootstrap percentile intervals for the headline scorecard metrics. We sample with replacement from the n=6,000 persample (confidence, correctness) pairs. Standard errors on AUROC are around 0.005–0.008, so several within-family gaps (bootstrap T=0.7 vs. T=1.0

on 8B; MCMC agreement T=0.5 vs. bootstrap on 27B) are within overlapping CIs. The betweenfamily gaps in the main text (bootstrap vs. direct; log-prob vs. raw MCMC) are an order of magnitude larger than the CI widths.

### D Controlled target-set scaling

Table 7 reports the controlled-N scaling, restricting the target vocabulary to N ∈ {2,5,10,20} words with a fixed seed: N=2: {snow, ship}; N=5: {flame, green, leaf, rock, snow}; N=10: {gold, song, chair, wave, ship, blue, snow, smile, flame, flag}; N=20: the full vocabulary. The samples per row are 300 · N for N ≤ 10 and 6,000 for N=20.

### E Per-layer readout sweep

Each verbalizer is trained on activations from three layers (the 25%, 50%, 75% depth points of Karvonen et al. (2025)). At inference time the experiment reads from one of these three. We swept every layer index 0..N−1 on four oracle bases: Qwen3-8B, Qwen3.6-27B, Gemma-2-9B, and Gemma-3-27B. The Qwen3-8B and Gemma2-9B verbalizers are released by Karvonen et al. (2025), that demonstrate the secret-word taboo task on those two models. The Gemma-3-27B verbalizer was trained using the same SFT recipe. Under the active target LoRA we captured the residual stream, fed it to the verbalizer with the matching introspection prefix, and recorded logprob baseline accuracy across 20 words × 5 contexts (n=100 per layer per model). Trained layer indices: {9,18,27} on Qwen3-8B; {16,32,48} on Qwen3.6-27B; {10,21,31} on Gemma-2-9B;

Word-disjoint split Random 50/50 split

Method Uncal Temp Platt Iso Beta Uncal Temp Platt Iso Beta Log-prob (offset) 0.172 0.180 0.128 0.127 0.128 0.258 0.157 0.018 0.021 0.017

- Bootstrap T=0.7 0.151 0.152 0.105 0.102 0.106 0.090 0.086 0.024 0.038 0.031

- Bootstrap T=1.0 0.065 0.027 0.032 0.041 0.034 0.057 0.033 0.015 0.015 0.018 Bootstrap T=1.5 0.114 0.123 0.042 0.048 0.042 0.075 0.085 0.026 0.021 0.021 Direct (numeric) 0.583 0.245 0.002 0.004 0.004 0.588 0.251 0.013 0.014 0.014 MCMC accept T=0.125 0.530 0.212 0.031 0.032 0.033 0.546 0.229 0.007 0.011 0.012 MCMC agreement T=0.5 0.271 0.182 0.070 0.028 0.028 0.267 0.176 0.076 0.025 0.026 Steering sensitivity 0.502 0.335 0.155 0.153 0.154 0.389 0.167 0.035 0.035 0.034

- Table 4: Post-hoc calibration on Qwen3-8B: test-set ECE after fitting each calibrator on the fit slice. Word-disjoint: fit on 10 of 20 secret words, evaluate on the other 10. Random 50/50: random sample-level split. Lower is better.

Word-disjoint split Random 50/50 split

Method Uncal Temp Platt Iso Beta Uncal Temp Platt Iso Beta Log-prob (offset) 0.142 0.090 0.030 0.017 0.021 0.138 0.096 0.030 0.019 0.021

- Bootstrap T=0.7 0.225 0.168 0.020 0.012 0.029 0.211 0.160 0.026 0.019 0.034

- Bootstrap T=1.0 0.164 0.040 0.041 0.041 0.041 0.153 0.027 0.019 0.018 0.019 Bootstrap T=1.5 0.108 0.054 0.023 0.022 0.023 0.098 0.034 0.022 0.016 0.016 Direct (numeric) 0.727 0.386 0.048 0.050 0.046 0.758 0.417 0.009 0.010 0.007 MCMC accept T=0.125 0.772 0.449 0.068 0.067 0.071 0.738 0.415 0.002 0.002 0.002 MCMC agreement T=0.5 0.351 0.297 0.080 0.076 0.076 0.388 0.327 0.021 0.008 0.010 Steering sensitivity 0.531 0.356 0.066 0.065 0.065 0.505 0.325 0.010 0.009 0.016

- Table 5: Post-hoc calibration on Qwen3.6-27B: test-set ECE after fitting each calibrator on the fit slice. Worddisjoint: fit on 10 of 20 secret words, evaluate on the other 10. Random 50/50: random sample-level split. Lower is better.

{15,31,46} on Gemma-3-27B. The two Qwens carry the headline experiments in section 3; the Gemma-2-9B and Gemma-3-27B sweeps are reported here so the layer-selection decision for any follow-up run is on the record.

On every oracle the probe-readable region is a band several layers wide, but its position along the stack varies across the four bases (fig. 4). Qwen38B has a wide plateau of 35–50% across layers 9–32 peaking at layers 22–30; Qwen3.6-27B has a tighter band of 30–38% across layers ∼ 37–58 (∼ 58–91% depth), with everything before layer ∼

- 28 at the overall task rate. Gemma-2-9B reads early: its band sits between ∼ 24% and ∼ 50% depth and decays through the second half of the stack. Gemma-3-27B has the widest band of the four, ∼ 50–56% across layers 31–59 (∼ 50–95% depth), with monotone rise from layer 11 and a sharp drop at the head. The 9.4× gap between layers 32 and 48 on the 27B reported in section 3 is the distance between the edge and the inside of a single readout band; both layers lie inside the probe-readable region.

The trained sample triple {25,50,75}% is sparse with respect to these bands. On Qwen3.627B only the 75% sample lands inside the band;

- on Gemma-2-9B only 25% lands; on Qwen3-8B both the 50% and 75% samples are inside; and
- on Gemma-3-27B both the 50% and 75% samples

land squarely in the plateau. The headline configuration in section 3 reads at 50% on Qwen3-8B and at 75% on Qwen3.6-27B. Trained-vs-peak gaps at the best trained sample are tight on three of the four: 3 pp on Qwen3-8B (layer 27 at 47% vs. peak 50%), 3 pp on Qwen3.6-27B (layer 48 at 35% vs. layer 51 at 38%), 5 pp on Gemma-2-9B (layer 10 at 28% vs. layer 13 at 33%), and 3 pp on Gemma-3-27B (layer 46 at 53% vs. layer 40 at 56%).

Gemma-3-27B is the strongest oracle in this sweep: peak 56%, against 50% on Qwen3-8B, 38% on Qwen3.6-27B, and 33% on Gemma-2-9B. Two of its three trained samples (50% at 50% accuracy, 75% at 53%) land inside the band, which is the most of any of the four oracles.

### F Attention backend numerics

The 27B run is sensitive to the choice of attention backend. With flash_attention_4 on Qwen3.627B at sequence length 171 (roughly the length of one verbalizer plus one context prompt), the cosine similarity between the read-layer hidden state and the flash_attention_2 reference drops to the 0.87–0.97 range, and the maximum absolute logit difference reaches 9.66. eager and sdpa remain numerically faithful at cos. sim. > 0.9999. The regression surfaces only on the non-cached forward pass that the experiment uses for activation collec-

Method ECE Brier NLL AUROC Qwen3-8B Log-prob (offset) 0.255[0.244,0.265] 0.246[0.237,0.254] 0.730[0.705,0.753] 0.840[0.831,0.851]

- Bootstrap T=0.5 0.193[0.182,0.204] 0.213[0.207,0.219] 1.029[0.953,1.113] 0.810[0.798,0.820]

- Bootstrap T=0.7 0.097[0.087,0.108] 0.171[0.166,0.175] 0.558[0.530,0.590] 0.830[0.821,0.840]
- Bootstrap T=1.0 0.057[0.049,0.068] 0.163[0.159,0.167] 0.498[0.487,0.507] 0.829[0.819,0.840]

- Bootstrap T=1.3 0.076[0.067,0.087] 0.171[0.167,0.176] 0.519[0.507,0.530] 0.823[0.811,0.834] Bootstrap T=1.5 0.083[0.075,0.093] 0.173[0.168,0.178] 0.522[0.510,0.532] 0.824[0.814,0.835] MCMC agreement T=0.5 0.263[0.253,0.275] 0.247[0.240,0.255] 1.608[1.489,1.726] 0.803[0.792,0.814] Steering sensitivity 0.404[0.393,0.414] 0.354[0.346,0.363] 4.401[4.191,4.604] 0.763[0.752,0.774] MCMC accept T=0.125 0.544[0.531,0.557] 0.534[0.522,0.546] 10.771[10.496,11.084] 0.563[0.554,0.572] Direct (numeric) 0.582[0.569,0.593] 0.580[0.567,0.591] 12.658[12.369,12.924] 0.516[0.510,0.521]

Qwen3.6-27B Log-prob (offset) 0.131[0.122,0.140] 0.158[0.151,0.165] 0.489[0.469,0.509] 0.835[0.823,0.846]

- Bootstrap T=0.5 0.327[0.318,0.337] 0.222[0.216,0.227] 0.800[0.749,0.858] 0.864[0.850,0.877]

- Bootstrap T=0.7 0.218[0.209,0.227] 0.157[0.153,0.161] 0.514[0.494,0.538] 0.863[0.851,0.875]
- Bootstrap T=1.0 0.147[0.139,0.155] 0.130[0.127,0.134] 0.431[0.423,0.439] 0.851[0.840,0.864]

- Bootstrap T=1.3 0.125[0.117,0.133] 0.125[0.121,0.128] 0.415[0.407,0.424] 0.837[0.823,0.851] Bootstrap T=1.5 0.103[0.096,0.112] 0.129[0.125,0.133] 0.422[0.412,0.432] 0.812[0.799,0.826] MCMC agreement T=0.5 0.390[0.382,0.399] 0.274[0.268,0.280] 1.383[1.290,1.483] 0.858[0.845,0.870] Steering sensitivity 0.501[0.490,0.510] 0.413[0.404,0.420] 4.172[3.971,4.361] 0.725[0.710,0.740] MCMC accept T=0.125 0.740[0.728,0.751] 0.729[0.717,0.740] 15.606[15.322,15.887] 0.528[0.519,0.535] Direct (numeric) 0.753[0.742,0.763] 0.752[0.741,0.762] 16.022[15.749,16.287] 0.404[0.390,0.416]

- Table 6: Bootstrap 95% CIs (1000 resamples) for the headline scorecard rows. Point estimate followed by the 2.5/97.5 percentile. n=6,000 per row.

Qwen3-8B Qwen3.6-27B

Method \ N 2 5 10 20 2 5 10 20 Bootstrap T=1.0 ECE 0.213 0.045 0.049 0.057 0.130 0.139 0.155 0.147

- Bootstrap T=0.7 ECE 0.139 0.108 0.080 0.097 0.194 0.196 0.230 0.218

- Bootstrap T=1.5 ECE 0.244 0.080 0.084 0.083 0.067 0.084 0.106 0.103 Log-prob (offset) ECE 0.487 0.245 0.275 0.255 0.157 0.160 0.119 0.131 MCMC agree T=0.5 ECE 0.047 0.269 0.242 0.263 0.350 0.374 0.395 0.390 Direct (numeric) ECE 0.291 0.584 0.563 0.582 0.710 0.719 0.764 0.753

Bootstrap T=1.0 AUROC 0.795 0.791 0.819 0.829 0.842 0.837 0.852 0.851 Bootstrap T=0.7 AUROC 0.783 0.801 0.815 0.830 0.854 0.847 0.871 0.863 Log-prob (offset) AUROC 0.829 0.815 0.828 0.840 0.835 0.820 0.847 0.835 MCMC agree T=0.5 AUROC 0.775 0.753 0.786 0.803 0.866 0.834 0.865 0.858

- Table 7: Controlled target-set scaling. Bootstrap T=1.0 on 8B is the only method whose ECE improves as N grows; most others degrade. On 27B every method is roughly N-stable.

tion; greedy decoding through model.generate produces token sequences byte-identical to the reference under flash_attention_4. The same comparison on Qwen3-8B at similar sequence lengths shows agreement at the level of bfloat16 numerical noise across all four backends. All experiments in this paper use flash_attention_2.

### G Linguistic-label direct-elicitation pilot

We ran an n=30 pilot on Qwen3-8B testing five direct-elicitation prompt variants on the same generated answer per item: zero-shot numeric (control), four-example few-shot numeric, verbalized linguistic (five labels: very low, low, medium, high, very high, scored by constrained logits), P(True) (Kadavath et al., 2022), and a hedged numeric variant (“be honest, you may be wrong”).

The verbalized-linguistic variant achieves AUROC 0.957 versus AUROC 0.500 for zero-shot numeric, with the modal label being “very high” on every sample; the ranking signal is carried by the probability mass on each label, not by the modal label. P(True) fails (AUROC 0.440) because the model assigns high probability to “yes” whenever the proposed answer is a plausible secret-word phrase (e.g. “secret”, “password”), regardless of activation match. The pilot is small enough that the standard error on AUROC is roughly 0.05 and the absolute number should not be settled; the gap to zero-shot numeric replicates across all 30 samples.

### H MCMC temperature sweeps

Table 8 sweeps the proposal temperature for the single-chain acceptance ratio (M4). Lowering the

Where the secret-word direction is probe-readable: per-layer accuracy of the log-prob baseline.

Qwen3-8B (36 layers, full attention)

Qwen3.6-27B (64 layers, hybrid linear/full)

1.0

taskaccuracy(secret-wordrecovery)

| | | |
|---|---|---|
| | | |

| | | |full attention sliding / linear attention verbalizer training layer|
|---|---|---|---|
| | | | |
| | | | |

0.8

0.6

0.4

0.2

0.0

0 5 10 15 20 25 30 35 layer index

0 10 20 30 40 50 60 layer index

Gemma-2-9B (42 layers, hybrid sliding/full)

Gemma-3-27B (62 layers, hybrid sliding/full)

1.0

taskaccuracy(secret-wordrecovery)

| | | |
|---|---|---|
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |

0.8

0.6

0.4

0.2

0.0

0 5 10 15 20 25 30 35 40 layer index

0 10 20 30 40 50 60 layer index

- Figure 4: Per-layer log-prob accuracy on the secret-word task across all decoder layers of four oracle bases. Dashed lines mark the three layers each verbalizer was trained on; bar shading distinguishes full-attention (indigo) from sliding/linear-attention (slate) layers. Band peaks: Qwen3-8B 50% at layer 28; Qwen3.6-27B 38% at layer 51; Gemma-2-9B 33% at layer 13; Gemma-3-27B 56% at layer 40 (broad plateau ∼ 50–56% across layers 31–59). Gemma-2-9B reads early; the Qwens read late; Gemma-3-27B has the widest probe-readable band of the four.

Qwen3-8B Qwen3.6-27B

T α Acc ECE AUR. Acc ECE AUR.

.500 2 .376 .551 .601 .211 .746 .574 .250 4 .403 .547 .579 .230 .738 .547 .125 8 .415 .544 .563 .234 .740 .528

- Table 8: M4 (MCMC acceptance) temperature sweep. AUR.: AUROC. Acceptance ratio is anti-correlated with confidence as T drops: the chain accepts more on both correct and wrong inputs, shrinking the gap between the two pools.

proposal temperature (raising α) improves task accuracy on both models but worsens AUROC: the chain accepts more proposals on confidentlylocked-in items but also accepts more on uncertain items, so the discriminator weakens. Raw acceptance is therefore the wrong sign at low T; we recommend not using it as a confidence signal.

- Table 10 sweeps the same temperature grid for

the agreement variant (M5). Here the pattern reverses: α = 2 preserves cross-chain diversity and dominates ECE/AUROC on both models; at α = 8 all k = 10 chains collapse to the greedy trajectory and the mode frequency saturates at 1.0 regardless of correctness. The optimal α for agreement-based power sampling is small, which is the same direction as the bootstrap optimum.

Table 9 compares MCMC agreement at its best temperature against bootstrap at its best temperature on each model. Bootstrap has lower ECE and

Method Acc ECE Br. AUR. Qwen3-8B

Bootstrap T=1.0 .402 .057 .163 .829 MCMC agree T=0.5 .413 .263 .247 .803

Qwen3.6-27B

Bootstrap T=1.0 .225 .147 .130 .851 MCMC agree T=0.5 .228 .390 .274 .858

- Table 9: Bootstrap versus power-sampling agreement at the best temperature of each. Br.: Brier. AUR.: AUROC.

Qwen3-8B Qwen3.6-27B

T α Acc ECE AUR. Acc ECE AUR.

.500 2 .413 .263 .803 .228 .390 .858 .250 4 .418 .408 .737 .233 .569 .767 .125 8 .415 .493 .669 .236 .668 .653

- Table 10: M5 (MCMC agreement) temperature sweep. The α = 2 row is competitive with bootstrap on AUROC, especially on 27B (AUR. 0.858), but at 5× the wall-clock cost and worse ECE.

Brier on both; MCMC agreement is +0.007 higher in AUROC on 27B and −0.026 lower on 8B, at roughly 5× the wall-clock cost.

### I Reliability fingerprints and rank summaries

Figure 5 and fig. 6 show reliability diagrams (binned mean confidence vs. accuracy) for the mostinformative methods on both models. Bootstrap at T = 1.0 on 8B and T = 1.5 on 27B track the diagonal; direct self-report and single-chain MCMC sit

[Figure 4]

- Figure 5: Reliability diagrams on Qwen3-8B for the eight most-discussed methods. Marker size encodes bin population. The diagonal is perfect calibration. Bootstrap families are near the diagonal; log-prob is underconfident (below the diagonal); direct self-report and MCMC acceptance are pinned to the right at near-1 confidence with empirical accuracy at the overall task rate.

[Figure 5]

- Figure 6: Reliability diagrams on Qwen3.6-27B. The same qualitative geometry as 8B, with the bootstrap diagonal shifted toward lower confidence (consistent with the lower overall accuracy). on the far-overconfident side of the plot.
- Figure 7 compactly summarizes every method’s

per-metric rank on both models. Bootstrap variants own the top three rows on both models; direct selfreport and MCMC acceptance own the bottom.

### J Confidence separation between correct and wrong pools

A useful diagnostic is the gap between mean confidence on correct predictions and mean confidence on wrong predictions: ∆ = c | aˆ=a⋆ − c | aˆ̸=a⋆. A method with ∆ ≤ 0 is anti-calibrated in the strong sense: its confidence predicts incorrectness.

### K Per-word accuracy: full table

- Table 11 reports per-word accuracy on Bootstrap T=1.0 for both models, sorted by 8B accuracy. Five words are hard on both models (leaf, wave, clock, song, rock); two are easy on both (moon,

[Figure 6]

[Figure 7]

Figure 7: Per-metric rank heatmap on Qwen3-8B (top) and Qwen3.6-27B (bottom). Lower (darker) is better. Columns are the five metrics; rows are methodtemperature pairs sorted by average rank.

snow); one inverts (blue: worst-three on 8B, best on 27B).

Wrong-answer modes differ sharply between models. On 8B, when the oracle is wrong, the top wrong answers are usually topical neighbors of the secret word: flame → fire, rock → tree/stone, cloud → sky, song → music. On 27B the top wrong answers are non-content tokens: secret, password, the empty string, word. About 14% of 27B generations are empty. The 8B failure mode commits to a plausible alternative; the 27B failure mode refuses. Refusal is easier to detect from confidence than committed confabulation, which is part of why 27B AUROC is higher than 8B AUROC on every bootstrap temperature despite the lower task accuracy.

[Figure 8]

[Figure 9]

- Figure 8: Confidence-pool split on Qwen3-8B (top) and Qwen3.6-27B (bottom). For each method, the upper marker shows mean confidence on correct predictions and the lower marker shows mean confidence on wrong predictions. Direct self-report on 27B is the only method with inverted markers (∆ < 0): the oracle is more confident when it is wrong.

### L Training the Qwen3.6-27B activation oracle

The Qwen3-8B oracle is the released checkpoint of Karvonen et al. (2025). For Qwen3.6-27B no oracle existed prior to this work. We adapt their LatentQA trainer to a hybrid Gated DeltaNet (3/4 of token-mixing sublayers) plus Gated Attention (1/4 of sublayers) architecture and train a verbalizer with the same data mixture (systemprompt question answering, binary classification, self-supervised context prediction) and the same LoRA recipe (r=8, one epoch, bf16). Three changes were needed:

LoRA target selection. The upstream regex matches only q_proj|k_proj|v_proj|o_proj and the MLP projections. Those names exist only in the 1/4 full-attention sublayers of Qwen3.6. The 3/4 Gated DeltaNet sublayers expose Lin-

Word Qwen3-8B Qwen3.6-27B ∆ moon .827 .380 −.447 snow .727 .413 −.314 jump .627 .240 −.387 smile .610 .240 −.370 ship .580 .100 −.480 chair .557 .190 −.367 flag .523 .097 −.426 salt .477 .277 −.200 green .473 .450 −.023 dance .350 .197 −.153 book .337 .267 −.070 gold .327 .247 −.080 cloud .307 .163 −.144 flame .297 .147 −.150 leaf .240 .050 −.190 blue .180 .500 +.320 wave .170 .110 −.060 clock .170 .157 −.013 song .150 .100 −.050 rock .120 .170 +.050

Table 11: Per-word accuracy on Bootstrap T=1.0, both models. The 27B oracle is uniformly less accurate except on blue and rock.

ear modules under different names (in_proj_qkv, in_proj_z, in_proj_b, in_proj_a, out_proj). Without expanding the regex, LoRA wraps only one quarter of the token-mixing capacity and training loss flattens well above the homogeneous baseline. Our expanded regex covers both attention types while still excluding the vision tower and lm_head.

Activation-collection slicing. The upstream multi-layer activation collector had an off-by-one in its offset window. We patched the slicing to honor [max_offset:min_offset] consistently, which matters when collecting from the trailing tokens of long inputs.

After these three fixes the verbalizer trains to comparable out-of-distribution validation loss as the homogeneous Qwen3-8B oracle (one B200, ≈ 36 hours wall-clock, single epoch on ≈ 1M examples). The 20 taboo target adapters use the upstream TRL recipe and train in ≈ 2 hours each at the same hardware.

### M Hardware, software, and runtime

All runs use a single NVIDIA B200 GPU with bfloat16 weights and the flash_attention_2 backend. The 8B benchmark sweep completes in ≈ 8 hours wall-clock; the 27B sweep in ≈ 48 hours. Each run is checkpointed and resumable. Software: PyTorch 2.7, Hugging Face Transformers 5.8, PEFT 0.19, flash-attn 2.8.3, Python 3.13.

