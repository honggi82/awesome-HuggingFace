## Reinforcement Learning Elicits Contextual Learning of Unseen Language Translation

Hanxu Hu1 Zdenˇek Šnajdr2 Pinzhen Chen3 Jannis Vamvas1 Rico Sennrich1 1University of Zurich 2ETH Zurich 3Queen’s University Belfast hanxu.hu@uzh.ch

# arXiv:2606.06428v1[cs.CL]4Jun2026

### Abstract

Prior work has shown that large language models (LLMs) can translate unseen or lowresource languages by undergoing continued training or even by encoding a grammar book in their context. However, both methods typically overfit specific languages, with limited zero-shot transfer at test time. To translate extremely low-resource languages at scale, we argue that LLMs must acquire the meta-skill of utilizing in-context linguistic knowledge rather than memorizing specific languages. In this paper, we propose a reinforcement learning (RL) approach to unseen language translation given rich linguistic context, using a surface-level translation metric (chrF) as the reward. Empirically, despite the lightweight reward, our RL-trained models effectively extract and apply relevant linguistic information from the provided context, leading to better translations on completely unseen languages than in-context learning or supervised fine-tuning. Our analyses suggest that outcome-based RL can extend beyond conventional reasoning tasks like math and coding to serve as a recipe for language learning from context.

### 1 Introduction

Large Language Models (LLMs) have achieved unprecedented performance across a wide range of tasks. Yet, as ML models, they remain fundamentally constrained when used in data-scarce, out-ofdistribution scenarios. A particularly revealing example is the processing of extremely low-resource or entirely unseen languages, e.g. unseen language translation (Tanzer et al., 2024), a setting that we argue constitutes an ideal testbed for LLM research for several reasons. First, reasoning over linguistic patterns with limited pre-training data remains challenging even for frontier LLMs (Bean et al., 2024; Kocmi et al., 2025). Second, progress here has direct societal implications: it might enable more

Code at: https://github.com/hanxuhu/rl-new-language

effective documentation and preservation of endangered languages, for which large-scale (parallel) corpora hardly exist (Joshi et al., 2020).

Two paradigms dominate LLM approaches to new languages: post-training the model on additional language/task-specific data (Yong et al., 2023; Iyer et al., 2024); and where resources are even more limited, in-context learning using inductive or deductive examples in the prompt (Garg et al., 2022; Agarwal et al., 2024; Tanzer et al., 2024). We argue that a more principled and scalable approach should instead be languageindependent learning, which is inspired by metalearning (Finn et al., 2017; Gu et al., 2018). Rather than memorize the content or knowledge of any specific language, an LLM should acquire the ability to generalize across new languages by reasoning over an accompanying context with rich linguistic features, grammatical descriptions, morphological paradigms, and lexicons. We refer to this transferable ability as a meta-skill of contextual leveraging, and view it as the key property that any generalpurpose method for very low-resource languages should target.

Motivated by this perspective, we introduce a reinforcement learning (RL) method which can elicit LLMs to leverage linguistic context when translating new unseen languages. The recent success of RL with verifiable rewards (RLVR) on reasoning tasks (OpenAI, 2026; Guo et al., 2025) suggests that outcome-based RL is particularly effective at eliciting transferable skills (e.g. reasoning) rather than memorized fixed solutions. We bring this philosophy to low-resource translation: we treat “translating a new language given its grammar” as a verifiable, context-dependent reasoning problem. Concretely, the model is trained to produce translations conditioned on a context of linguistic knowledge and dictionary entries, with a translation quality metric (e.g., chrF) as the outcome reward. By simply rewarding the translation using RL, we can

elicit LLMs to learn the meta-skill of leveraging linguistic context and can gain better generalization performance for unseen languages. In summary, our contributions are as follows:

- • A reframing of extreme low-resource translation as the acquisition of a language-independent meta-skill, reasoning over in-context linguistic context, rather than memorization of any individual target language, unifying in-context learning and post-training under one objective.
- • An outcome-reward RL recipe for context leveraging. We propose a simple training method that conditions the model on linguistic knowledge and dictionary context and uses translation quality as the outcome reward.
- • Empirical evidence for generalization. Through controlled context learning comparisons, we show that RL can bring better performance on unseen languages, while SFT overfits to in-domain training languages, showing that the method successfully generalizes.

### 2 Related Work

Tanzer et al. (2024) introduced MTOB, a benchmark for translating between English and Kalamang (an extreme low-resource language). By providing a grammar book (Visser, 2022) as context, this work demonstrates how LLMs can leverage explicit linguistic documentation to translate an entirely new language. Zhang et al. (2024b) include bilingual dictionaries, grammar books, and morphologically analyzed text in context to guide LLM translation without parameter updates. Following the DiPMT (Ghazvininejad et al., 2023) framework, where an LLM is fed bilingual lexicons of rare words in the prompt, Zhang et al. (2024a) used dictionary retrieval and in-context learning to teach LLMs the Zhuang language on the fly. Zhang et al. (2025) further explored the effectiveness of grammar by decomposing it into rule retrieval and application, and specifically proposed to convert grammar into pseudocode. On the contrary, more recently, Aycock et al. (2025) showed that LLMs make little use of grammar descriptions but rely on the parallel examples in the book when translating an unseen language. This finding is backed by the study by Pei et al. (2025) on Manchu that bilingual dictionaries are more helpful, while grammar and CoT steps do not bring noticeable benefit. Nonetheless, (Marmonier et al., 2025) found that LLMs have measurable capacity to learn constructed lan-

guages from grammar descriptions and fine-tuning on CoT greatly improves it, but generalizability is limited for complex linguistic phenomena. Other approaches using LLMs to process low/no-resource languages include combining an LLM with rules (Coleman et al., 2024, 2026) or finite-state transducers (Gutierrez et al., 2025).

Orthogonal to above approaches, a recent line of work explores fine-tuning via reinforcement learning (RL). Building on RLVR (Lambert et al., 2025; Guo et al., 2025) and GRPO (Shao et al., 2024), several works have transferred this paradigm to MT despite the absence of a single “correct” output. Feng et al. (2025) introduce MT-R1-Zero, the first R1-Zero-style adaptation for MT, using a hybrid BLEU + COMET-Kiwi reward under GRPO; He et al. (2025) pair COMET-based rewards with format signals to incentivize translation reasoning, while Wang et al. (2026) and Yang et al. (2026) move beyond reference-based metrics via trajectory-level generative or self-rewarding signals. He et al. (2024) earlier showed that qualityestimation models provide a data-efficient reward for MT RLHF. Closer to the low-resource regime, Mosquera et al. (2025) use RL to teach an LLM to consult a bilingual dictionary as an external tool during Spanish–Wayuunaiki translation. and Attia and Aji (2026) propose self-supervised roundtrip RL with intrinsic chrF++ rewards for Aymara, Friulian, and Wolof. However, existing RL-forMT methods optimize translation quality directly from source-target pairs, leaving open how RL can instead teach models to better exploit in-context linguistic resources.

Our setup can be naturally framed as metalearning: rather than optimizing for any single translation instance, we train the model to make better use of a contextual “support set” of linguistic resources at inference time. This perspective has a long history in low-resource MT, where Gu et al. (2018) adapted MAML (Finn et al., 2017) to learn initializations that transfer quickly to new language pairs from limited parallel data. With the rise of in-context learning, Brown et al. (2020) reframed few-shot learning as implicit meta-learning over the pre-training distribution, and a subsequent line of work has sought to make this capability explicit: MetaICL (Min et al., 2022) meta-trains on a distribution of tasks formatted as in-context demonstrations to amplify few-shot performance, in-context tuning (Chen et al., 2022) similarly optimizes models to condition on examples rather than

to fit them, and Garg et al. (2022) give a theoretical account of what transformers can learn in-context. Unlike these supervised meta-learning approaches, we use RLVR to teach the model to exploit qualitatively heterogeneous in-context resources better for low-resource translation.

- 3 Methodology 3.1 Data Curation

This section describes the construction of the corpus used for our reinforcement learning experiments, covering language selection, parallelsentence extraction, the dedicated treatment of Romansh, the synthesis of dictionary entries, and the structure of the assembled prompts.

- 3.1.1 Language Selection and Sources We investigate fourteen (very) low-resource languages, grouped by data-sourcing pipeline rather than by language family. The first group comprises eight languages whose parallel data is extracted from Language Science Press grammar books: Choguita Rarámuri, Gyeli, Japhug, Kagayanen, Kalamang, Tuatschin, Ulwa, and Vamale. We note that Tuatschin is genealogically a dialect of Sursilvan and thus belongs to the Romansh continuum; it is grouped here because its data comes from a grammar book rather than from the parallel corpora and dictionaries used for the standardized Romansh idioms below. The second group consists of the six standardized varieties of RomanshPuter, Vallader, Surmiran, Sursilvan, Sutsilvan, and Rumantsch Grischun—which, despite a shared heritage, diverge enough lexically and orthographically to warrant separate treatment.

The Romansh varieties were selected because they allow studying meta-learning on a spectrum of language relatedness, and due to the availability of dictionary data and grammar books; other languages were selected based on the availability of open-source grammar books. For the non-Romansh languages, candidates were restricted to those with grammars published by Language Science Press under the CC BY 4.0 licence and distributed in both PDF and LATEX source. The detail is shown in Appendix C.

- 3.1.2 Parallel Sentence Extraction Due to the lack of parallel corpora available, we extracted translated examples directly from their grammars. An exception are the five Romansh varieties, where we sample a small amount of parallel

sentences from the back-translated training data used by Vamvas et al. (2026). Although the grammar books occasionally contain OCR or LATEX artefacts, the underlying translations have been verified by an author-linguist. We worked from the LATEX source rather than the rendered PDF to preserve idiosyncratic orthographies, locating interlineargloss examples by pattern-matching the standard \gll and \glt commands as well as user-defined macros. Residual markup was stripped from both sides of each pair.

We retained only pairs with source sentences between 6 and 50 space-delimited words. A stricter filter targeting extraction artefacts (long uppercase sequences, inline numbering, stray digits, ellipses, underscores, residual newlines) was tested but reduced the corpora to as few as 11 sentences for Japhug and 20 for Kalamang—too few for reinforcement learning—so only the length filter was retained. To enable bidirectional training, each pair was duplicated with the source and target swapped.

#### 3.1.3 Romansh Preprocessing

The Romansh varieties differ in two respects: they possess large, high-quality parallel corpora and dictionaries (grammar is available only for Puter and Vallader), but their reference language is German rather than English. We discarded sentences longer than 50 words from both the training and test splits and, where more than 2,000 pairs remained, randomly sampled 2,000 to keep per-language volumes comparable. All prompt components were translated into German so that experimental conditions remain identical across language groups. Dictionary entries where retrieved using the Rumlem tool (Fischer et al., 2026) based on the Romansh segments.

#### 3.1.4 Synthetic Dictionary Augmentation

Every language other than Kalamang and the Romansh varieties lacks sufficient dictionary resources, so we generated synthetic entries using an LLM based on parallel data and grammar books in the format of the MTOB benchmark (Tanzer et al., 2024). Two prompt designs were compared, conditioning on the same parallel sentences and grammar excerpts but differing in the dictionary section: v1 explicitly names each source token and shows two working example entries per token, whereas v2 only describes the general entry format without token-specific demonstrations. For each language and translation direction, we generated dictionar-

###### Component Content Retrieval

Introduction Linguistic and geographic profile of the source language Task instruction Translation direction and test sentence Dictionary section 2 entries per source token LCS Parallel sentences 3 or 5 source–target pairs LCS Grammar passages 2 raw excerpts from the grammar LCS Closing instruction Request for step-by-step meta-linguistic reasoning —

- Table 1: Components of an assembled prompt. The parallel-sentence count (3 or 5) is an experimental condition.

Split Group Langs Dirs Families Train Test Seen

Romansh→De (seen varieties) 4 4×1 1 13,892 1,462 Other↔EN 7 7×2 6 9,695 —

Similar Romansh→De (held-out)‡ 2 2×1 1 7,998 737 Unseen

Kalamang↔EN† 1 1×2 1 750 100 OOD (EN→X only) 4 4×1 4 — 400

Total 18 26 10 23,587 2,699

- Table 2: Data summary. Languages are partitioned by evaluation split. Seen languages appear in both train and test; Similar directions are held out for evaluation but share their family with seen varieties; Unseen directions have no related training data. Only Seen languages are used for training; struck-through numbers (7,998, 750) indicate parallel data that exists for the held-out Similar and Unseen languages but is excluded from training. Dirs reports translation directions as langs × directionality (×2 for bidirectional, ×1 for unidirectional). Each prompt contains ∼20–34 dictionary entries, 5 parallel sentence pairs, and optionally a grammar-book passage (∼2.8k tokens on average). ‡Sursilvan→De and Surmiran→De, held out from training.

ies with both variants, used them in prompts over the training data, and retained the variant with the higher chrF on the resulting translations. The final synthetic dictionary is therefore a per-language, per-direction mixture of v1 and v2.

#### 3.1.5 Prompt Composition

Each prompt is assembled from five components, as summarised in Table 1, and ends with an instruction to produce step-by-step meta-linguistic reasoning before committing to a final translation. Retrieval uses the Longest Common Subsequence (LCS) metric, which outperformed embedding-based similarity in preliminary experiments. Two configurations of the parallel-sentence section—three and five examples—are evaluated independently to study the effect of in-context example count.

- Table 2 summarises the composition of our

dataset across the four language groups. In total, the corpus spans 18 languages from 10 distinct language families, yielding 32,335 training pairs and 2,699 test pairs. The Romansh varieties contribute the largest share of the training data (21,890 pairs across six varieties translated to and from German), reflecting both the availability of highquality parallel corpora and the upper bound of 2,000 sampled sentences per variety. The remain-

ing seven low-resource languages aligned with English contribute 9,695 training pairs but are used exclusively for training, since their grammar bookextracted examples are too scarce to be split further. Kalamang↔EN is treated separately: we adopt the original MTOB split of 750 training and 100 test pairs to ensure direct comparability with prior work, and additionally hold it out together with Sursilvan↔De and Surmiran↔De in our training data configuration to probe cross-lingual generalisation within and across families. Finally, four languages out-of-distribution (one per family) are reserved as a test-only EN → X benchmark from Hus and Anastasopoulos (2024) to assess transfer to languages entirely unseen during RL training, with 100 sentences randomly sampled per direction for each language.

3.2 RL with Linguistic Knowledge in Context With the curated data from Section 3.1, we formulate translation as a reinforcement learning task in which the policy is trained to produce translations conditioned on a meta-linguistic context.

Task formulation. Each training instance is a triple (x,c,y), where x is a source sentence, y its reference translation, and c the languageknowledge context assembled as described in Sec-

tion 3.1, comprising the introductory description, retrieved dictionary entries, parallel sentences, and grammar passages. The policy πθ is a large language model that, given the concatenation of c and x, generates a response containing step-by-step meta-linguistic reasoning followed by a final hypothesis translation yˆ. Only yˆ is scored against y; the reasoning trace is left unconstrained.

Reward. We use chrF (Popovi´c, 2015) between yˆ and y as the reward signal. Since chrF is reported on a 0–100 scale, we rescale it to [0,1], which is a typical range for reward functions: r(ˆy,y) =

1

100chrF(hyp = y,ˆ ref = y). If the model fails to produce a parsable final translation, the reward is

set to zero.

Optimization. We optimize πθ with GRPO (Shao et al., 2024). For each prompt, G responses

{yˆi}Gi=1 are sampled from the current policy and scored independently. Each reward is then standard-

ized within its group, Ai = σ1

(ri − µG), where µG and σG are the mean and standard deviation of {ri}Gi=1, yielding a group-relative advantage that requires no separate value model. The policy is updated with the standard PPO-style clipped objective using Ai, regularized by a KL penalty against a frozen reference policy to limit drift from the initial model. The prompt fed to the policy includes meta-linguistic information about the low-resource language grammar rules, dictionary entries, and parallel sentences, together with the source sentence, allowing the translation quality reward to shape how the policy makes use of these resources when generating yˆ.

G

### 4 Experiments

#### 4.1 Experimental Setup

Models and training. We fine-tune two backbones, Qwen3-4B-Base and Llama-3.2-3BInstruct, with both SFT and RL. SFT and RL share an identical prompt format—the full retrieval context from Section 3.1, with two LCS-retrieved dictionary entries per source token, three or five parallel sentences, and two grammar passages where available—and differ only in the supervision signal: SFT minimises cross-entropy against the gold reference, whereas RL optimises chrF reward via GRPO, as described in Section 3.2. All models are trained for one epoch on the 22 training directions, which exclude the two Romansh varieties (Sursilvan→German, Surmiran→German) held out for

the Similar evaluation setting. SFT uses an effective batch size of 128 and RL a batch size of 64. Remaining hyperparameters are listed in Appendix A.

Evaluation data. For Romansh–German evaluation, we use the WMT24++ benchmark (Vamvas et al., 2025), with each language variety evaluated in the X→De direction. For the five unseen languages, the test data come from two sources. Kalamang (100 pairs in each direction) uses the original MTOB benchmark split (Tanzer et al., 2024), based on the Kalamang grammar book (Visser, 2022). The remaining four languages (Dinka, Wolof, Guarani, Kachin, 100 pairs each in the EN→X direction) use the test sets from Hus and Anastasopoulos (2024).

Test-time conditions and metric. Each checkpoint is evaluated under two prompt context conditions to separate retrieval-dependent skills from those internalized into the policy: full (same prompt as training) and none (only the language description and translation instruction). No model is trained without retrieval. We report chrF results on the 0–1 scale.

#### 4.2 Comparison of SFT and RL

As shown in Table 3, we evaluate three conditions of increasing difficulty: (1) seen languages Romansh language varieties seen during training, (2) similar held-out Romansh varieties from the same family, and (3) five unseen languages from unrelated families. Table 3 reports results under both full retrieval context and no context (task instruction only).

SFT is stronger on seen languages. On seen Romansh→German (averaged over four trained language varieties), Qwen SFT scores 0.60 versus RL’s 0.52; LLaMA SFT scores 0.56 versus RL’s 0.48. A similar gap holds on the held-out Romansh varieties (Qwen SFT 0.55 vs. RL 0.48), suggesting that SFT’s advantage extends within the language family. This is expected: SFT trains on gold translations to fit word-level alignment, whereas RL receives only a sentence-level chrF reward.

RL is stronger on unseen languages. On five unseen languages from different families (Kalamang, Dinka, Wolof, Guarani, Kachin), the ranking reverses. Qwen RL averages 0.27 across these languages, compared to 0.09 for SFT and 0.18 for

Romansh→De Unseen languages (En→X) Method Context Seen Similar Kal Din Wol Gua Kac Avg. Qwen3-4B-Base experiments

Qwen3-4B-Base full 0.3413 0.3203 0.2558 0.1606 0.1505 0.1728 0.1774 0.2255 Our SFT full 0.6017 0.5464 0.2860 0.0506 0.0484 0.0639 0.0128 0.2300 Our RL full 0.5160 0.4785 0.3464 0.2291 0.2253 0.2679 0.2715 0.3335 Qwen3-4B-Base none 0.2150 0.2413 0.1470 0.1452 0.1303 0.1256 0.1195 0.1606 Our SFT none 0.4644 0.4235 0.1358 0.1075 0.0472 0.2083 0.1827 0.2242 Our RL none 0.3048 0.3694 0.1687 0.0643 0.1025 0.1856 0.0752 0.1815 Llama-3.2-3B-Instruct experiments

Llama-3.2-3B-Inst full 0.3012 0.2861 0.2014 0.1044 0.1118 0.1307 0.1545 0.1843 Our SFT full 0.5577 0.4970 0.2444 0.0329 0.0576 0.0713 0.0269 0.2125 Our RL full 0.4765 0.4414 0.3005 0.1949 0.2029 0.2486 0.2493 0.3020 Llama-3.2-3B-Inst none 0.2222 0.2037 0.1485 0.0640 0.0571 0.0784 0.0236 0.1139 Our SFT none 0.4411 0.3826 0.1225 0.0410 0.1131 0.1231 0.0152 0.1769 Our RL none 0.3470 0.3327 0.1669 0.0968 0.1134 0.1577 0.0430 0.1796

- Table 3: Main results on low-resource translation (chrF, ↑). Seen: avg of 4 Romansh varieties→German (Puter, Vallader, Sutsilvan, RumantschGr.), in training data. Similar: avg of 2 held-out varieties→German (Sursilvan, Surmiran). Unseen: 5 languages from unrelated families, never in training—Kalamang (Papuan), Dinka (NiloSaharan), Wolof (Niger-Congo), Guarani (Tupi-Guarani), Kachin (Sino-Tibetan). Avg.: macro-average across all 7 language directions. The full-context block (dictionary + parallel sentences + grammar; shaded) is our primary evaluation; the no-context block (task instruction only) is included to show the expected failure to generalize to unrelated unseen languages. Bold = best fine-tuned row per (column) cell within the full-context block.

Romansh→De Kalamang Train/Test Context

Puter, Val

Surs, Surm

En→ Kal

Kal→ En

Full (dict+sent+gram) .5324 .4785 .3464 .3843 No grammar (dict+sent) .5249 .4766 .3319 .3821 No sent (dict+gram) .5224 .4669 .2733 .2932 No dict (sent+gram) .4483 .4077 .2626 .3146 Task only (no context) .4154 .3770 .1700 .1968

- Table 4: Context ablation (RL, Qwen3-4B-Base). Each row trains and evaluates with a matched context level. Dictionary entries are the most impactful component (−8 chrF on seen languages when removed); grammar contributes minimally (−0.5); parallel sentences are critical for OOD Kalamang (−7 on En→Kal).

- • Seen: SFT retains much of its performance without context (Qwen SFT: 0.60→0.46), indicating that it stores language-specific mappings in the weights. RL drops more (0.52→0.30), consistent with greater reliance on context.
- • Unseen languages: Without context, all models perform significantly worse.
- • The role of context: RL’s advantage on unseen languages (0.27 vs. 0.12) appears only when retrieval context is available. This suggests that RL’s generalization is tied to its use of in-context resources rather than to language-specific knowledge acquired during training.

Taken together, the results point to a trade-off between the two training objectives. SFT fits the training languages more tightly but appears to reduce context utilization for new languages. RL fits training languages less precisely but develops a more transferable ability to translate through the provided context: dictionary entries, parallel sentences, and grammar passages, which generalizes to languages from unseen families when such resources are available at test time.

the untuned base model. Llama-3.2 shows a consistent pattern (RL 0.24 vs. SFT 0.09 vs. base 0.14). The per-language columns in Table 3 show that RL outperforms SFT on each of the five languages individually. Notably, SFT scores below the base model on these languages, suggesting that supervised training on the Romansh directions may reduce the model’s ability to use in-context resources for unfamiliar languages.

#### 4.3 Ablation study with each role in context

Context removal clarifies the source of each method’s strength. Stripping retrieval context at test time (“no context” rows) reveals different behaviours:

To understand the contribution of each retrieval component, we train five matched RL runs on Qwen3-4B-Base. In each run, we remove one com-

###### Train×Test Context Mismatch (RL, Qwen3-4B-Base)

###### Romansh DE

Kalamang

Test: full Test: none

Test: full Test: none

0.6

0.532 #1 0.478

#1 0.384

#4

#4

0.4

0.346

0.5

0.402

0.369

0.4

0.3

Train: full

Train: full

chrF

chrF

0.3

0.169

0.2

0.148

0.2

0.1

0.1

0.0

0.0

0.6

#2

#3

#2

#3

0.4

0.456

0.5

0.419

0.415

0.307

0.377

0.277

0.4

0.3

Train: none

Train: none

chrF

chrF

0.197

0.3

0.170

0.2

0.2

0.1

0.1

0.0

0.0

En Kal Kal En

En Kal Kal En

Puter, Val Surs, Surm

Puter, Val Surs, Surm

Romansh→De En↔Kal

- Figure 1: Train–test context mismatch (RL, Qwen3-4B-Base). Test-time context dominates: no/full > full/no in every panel (En→Kal: 0.28 vs. 0.17).

ponent from both training and test prompts, so the policy never sees a component at training that will be absent at inference. Results are shown in Table 4.

Among the three components, the bilingual dictionary has the largest impact. Removing it causes a drop of 8.4 chrF on seen Romansh (Puter/Vallader: 0.5324 → 0.4483) and 8.4 on En→Kalamang (0.3464 → 0.2626). This is expected because the dictionary provides direct wordlevel grounding that the other two components cannot fully replace.

Parallel sentences rank second. Removing them has only a moderate effect on seen Romansh (−1.0 chrF on Puter/Vallader), but a much larger effect on OOD Kalamang (−7.3 on En→Kal, from 0.3464 to 0.2733). This gap suggests that for languages far from the training distribution, parallel examples provide a complete example for translation.

Grammar passages contribute the least. Removing them causes the smallest drop across all test sets (−0.8 chrF on Puter/Vallader, −1.5 on En→Kal). The model might be unable to extract useful information from grammar passages.

We also examine whether the benefit of context comes from training or from test time (Figure 1). We cross the two conditions: a model trained without context but evaluated with context (no/full), versus a model trained with context but evaluated without (full/no). Across all test sets, no/full consistently outperforms full/no (e.g., En→Kal: 0.28 vs. 0.17). This shows that test-time context availability is the dominant factor: even a model trained with RL that never saw retrieval

during training can exploit it at inference. However, training with context further amplifies this ability. full/full exceeds no/full by +7 chrF on En→Kal (0.35 vs. 0.28), confirming that exposure to retrieval during RL teaches the policy a more systematic way to use in-context evidence.

#### 4.4 Analysis of training rewards

We run GRPO three times on Qwen3-4B-Base over the training split, varying only the retrieval content of the prompt. Full matches Section 4.1 (dictionary, parallel sentences, and grammar passages, all LCS-retrieved). No-dict drops the dictionary block. Task-only drops all three components. The reward function, optimiser, and training budget are unchanged across runs.

Both panels of Figure 2 rank the three runs in the same order: Full > No-dict > Task-only. With the chrF reward, Full reaches 0.68, No-dict reaches 0.62, and Task-only flattens near 0.29 by step 50. The 6-point gap between Full and No-dict isolates the dictionary’s contribution, which is not redundant with parallel sentences. The 33-point gap between No-dict and Task-only shows that without any retrieval, GRPO has nothing language-specific to ground on and saturates near the untuned baseline. On WMT24++, the same order holds, but the differences compress to about 0.06, since the RM weighs fluency more heavily than surface form.

The reward trajectories support the ablation findings from the perspective of training dynamics. The Task-only run plateaus early, confirming that without linguistic grounding, GRPO exhausts its improvement budget within about 50 steps and con-

###### (a) Test Reward

###### (b) Training Reward

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |

| |Solid: EMA smo|othed ( =0.92). Fa<br><br>|int: raw per-step|reward.|
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

RewardScore(WMT24pp-RM,mean@1)

0.7

0.40

TrainingReward(critic/score/mean)

0.6

0.35

0.5

0.30

0.4

0.3

0.25

0.2

0.20

0.1

0 50 100 150

0 50 100 150

Training Step

Training Step

|Full prompt (dict + sentences + grammar) No dictionary Task instruction only<br><br>|
|---|

- Figure 2: RL reward trajectories on Qwen3-4B-Base under three prompt configurations. (a) Held-out WMT24++ reward. (b) chrF training reward; faint lines raw, solid lines EMA-smoothed (α=0.92).

No context SFT No context RL Full context SFT Full context RL

- (1) Reference: Granny Ruslan’s grandmother is still strong.

Nina Ruslan is a strong woman.

Nina Ruslan speaks a strong word.

Granny Ruslan is still strong.

Granny Ruslan’s grandmother is still strong.

- (2) Reference: Rustam wanted to bathe but he felt cold.

Rustam and Waruotkin are the ones who are eating the food.

Rustam sees and hears the sound.

Rustam wants to wash his hands.

Rustam wants to wash and he feels cold.

- Table 5: Kalamang→English case study. Outputs of SFT and RL models with and without retrieval context, for two source sentences.

verges to a policy that relies only on pretraining priors. The No-dict run continues to improve past step 100, showing that parallel sentences alone provide a useful learning signal, but the ceiling is lower. Only the Full run sustains reward growth throughout training. This indicates that the dictionary supplies a complementary gradient signal that keeps the policy improving even after the parallelsentence signal saturates. Together with the ablation results, the reward analysis shows that each retrieval component contributes an additive learning signal, with the dictionary providing the strongest boost and grammar the weakest.

#### 4.5 Case study

- Table 5 illustrates this gap with two examples. Without retrieved context, both SFT and RL produce fluent English that recognizes the person name but is semantically unrelated to the source. With a dictionary and parallel-sentence context, RL pro-

duces near-perfect translations: in (1) it exactly matches the reference, and in (2) it correctly captures both “bathe” and “cold”. SFT, by contrast, captures only partial meaning—rendering “bathe” as “wash his hands” and dropping “cold” entirely.

### 5 Conclusion

In this paper, we propose a reinforcement learning approach for low-resource translation that learns to exploit in-context linguistic information. Using translation quality directly as the reward signal, our empirical results demonstrate that RL-trained LLMs generalize substantially better to unseen languages than their SFT counterparts, and genuinely learn to leverage linguistic context rather than memorize the specific languages seen during training. Our work offers a new perspective on low-resource translation by uniting the complementary strengths of in-context learning and reinforcement learning.

### Limitations

We report chrF++ but do not conduct human evaluation, which would give a sharper picture of fluency and adequacy in low-resource settings; we view the automatic numbers as a reliable signal of relative improvement across methods and leave human evaluation to future work. While our method generalizes to unseen languages far better than SFT, absolute performance on unseen languages still lags behind that for higher-resourced languages, indicating a clear headroom for richer in-context evidence and stronger context-utilization signals.

### Acknowledgments

HH, JV and RS acknowledge funding by the Swiss National Science Foundation (project InvestigaDiff; no. 10000503). We thank RTR and Fundaziun Patrimoni Cultural RTR for their support, and Lia Rumantscha and Uniun dals Grischs for contributing dictionary data.

### References

Rishabh Agarwal, Avi Singh, Lei Zhang, Bernd Bohnet, Luis Rosias, Stephanie Chan, Biao Zhang, Ankesh Anand, Zaheer Abbas, Azade Nova, and 1 others. 2024. Many-shot in-context learning. Advances in Neural Information Processing Systems, 37:76930– 76966.

Ahmed Attia and Alham Fikri Aji. 2026. Improving low-resource machine translation via round-trip reinforcement learning. Preprint, arXiv:2601.12535.

Seth Aycock, David Stap, Di Wu, Christof Monz, and Khalil Sima’an. 2025. Can LLMs really learn to translate a low-resource language from one grammar book? In The Thirteenth International Conference on Learning Representations.

Russell Barlow. 2023. A grammar of Ulwa (Papua New Guinea). Number 6 in Comprehensive Grammar Library. Language Science Press, Berlin.

Andrew Michael Bean, Simeon Hellsten, Harry Mayne, Jabez Magomere, Ethan A Chi, Ryan Andrew Chi, Scott A. Hale, and Hannah Rose Kirk. 2024. LINGOLY: A benchmark of olympiad-level linguistic reasoning puzzles in low resource and extinct languages. In The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track.

Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child,

Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, and 12 others. 2020. Language models are few-shot learners. Preprint, arXiv:2005.14165.

Gabriela Caballero. 2022. A grammar of Choguita Rarámuri. Number 5 in Comprehensive Grammar Library. Language Science Press, Berlin.

Yanda Chen, Ruiqi Zhong, Sheng Zha, George Karypis, and He He. 2022. Meta-learning via language model in-context tuning. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 719–730. Association for Computational Linguistics.

Jared Coleman, Bhaskar Krishnamachari, Ruben Rosales, and Khalil Iskarous. 2024. LLM-assisted rule based machine translation for low/no-resource languages. In Proceedings of the 4th Workshop on Natural Language Processing for Indigenous Languages of the Americas (AmericasNLP 2024), pages 67–87. Association for Computational Linguistics.

Jared Coleman, Ruben Rosales, Kira Toal, Diego Cuadros, Nicholas Leeds, Bhaskar Krishnamachari, and Khalil Iskarous. 2026. Comparing LLM-based translation approaches for extremely low-resource languages. In Proceedings for the Ninth Workshop on Technologies for Machine Translation of Low Resource Languages (LoResMT 2026), pages 49–68, Rabat, Morocco. Association for Computational Linguistics.

Zhaopeng Feng, Shaosheng Cao, Jiahan Ren, Jiayuan Su, Ruizhe Chen, Yan Zhang, Jian Wu, and Zuozhu Liu. 2025. MT-r1-zero: Advancing LLM-based machine translation via r1-zero-like reinforcement learning. In Findings of the Association for Computational Linguistics: EMNLP 2025, pages 18685– 18702, Suzhou, China. Association for Computational Linguistics.

Chelsea Finn, Pieter Abbeel, and Sergey Levine. 2017. Model-agnostic meta-learning for fast adaptation of deep networks. In Proceedings of the 34th International Conference on Machine Learning - Volume 70, ICML’17, page 1126–1135. JMLR.org.

Dominic P. Fischer, Zachary Hopton, and Jannis Vamvas. 2026. RUMLEM: A dictionary-based lemmatizer for Romansh. In Proceedings of the 11th Edition of the Swiss Text Analytics Conference, pages 125–132, Zurich, Switzerland. Association for Computational Linguistics.

Gian Paul Ganzoni. 1983. Grammatica Ladina: Grammatica sistematica dal rumauntsch d’Engiadin’Ota per scolars e creschieus da lingua rumauntscha e tudais-cha. Lia Rumantscha, Chur.

Shivam Garg, Dimitris Tsipras, Percy Liang, and Gregory Valiant. 2022. What can transformers learn incontext? a case study of simple function classes. In Advances in Neural Information Processing Systems.

Marjan Ghazvininejad, Hila Gonen, and Luke Zettlemoyer. 2023. Dictionary-based phrase-level prompting of large language models for machine translation. arXiv preprint arXiv:2302.07856.

Nadine Grimm. 2021. A grammar of Gyeli. Number 2 in Comprehensive Grammar Library. Language Science Press, Berlin.

Jiatao Gu, Yong Wang, Yun Chen, Victor O. K. Li, and Kyunghyun Cho. 2018. Meta-learning for lowresource neural machine translation. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 3622–3631. Association for Computational Linguistics.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, and 175 others. 2025. Deepseek-r1 incentivizes reasoning in llms through reinforcement learning. Nature, 645(8081):633–638.

Ximena Gutierrez, Mikel Segura Elizalde, and Victor Mijangos. 2025. FSTs vs ICL: Generalisation in LLMs for an under-resourced language. In Findings of the Association for Computational Linguistics: EMNLP 2025, pages 15998–16006, Suzhou, China. Association for Computational Linguistics.

Minggui He, Yilun Liu, Shimin Tao, Yuanchang Luo, Hongyong Zeng, Chang Su, Li Zhang, Hongxia Ma, Daimeng Wei, Weibin Meng, Hao Yang, Boxing Chen, and Osamu Yoshie. 2025. R1-t1: Fully incentivizing translation capability in llms via reasoning learning. Preprint, arXiv:2502.19735.

Zhiwei He, Xing Wang, Wenxiang Jiao, Zhuosheng Zhang, Rui Wang, Shuming Shi, and Zhaopeng Tu. 2024. Improving machine translation with human feedback: An exploration of quality estimation as a reward model. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 8164–8180, Mexico City, Mexico. Association for Computational Linguistics.

Jonathan Hus and Antonios Anastasopoulos. 2024. Back to school: Translation using grammar books. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 20207–20219. Association for Computational Linguistics.

Vivek Iyer, Bhavitvya Malik, Wenhao Zhu, Pavel Stepachev, Pinzhen Chen, Barry Haddow, and Alexandra Birch. 2024. Exploring very low-resource translation with LLMs: The University of Edinburgh’s submission to AmericasNLP 2024 translation task. In Proceedings of the 4th Workshop on Natural Language Processing for Indigenous Languages of the Americas (AmericasNLP 2024), pages 209–220. Association for Computational Linguistics.

Guillaume Jacques. 2025. A grammar of Japhug. Number 1 in Comprehensive Grammar Library. Language Science Press, Berlin.

Pratik Joshi, Sebastin Santy, Amar Budhiraja, Kalika Bali, and Monojit Choudhury. 2020. The state and fate of linguistic diversity and inclusion in the NLP world. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 6282–6293. Association for Computational Linguistics.

Tom Kocmi, Sweta Agrawal, Ekaterina Artemova, Eleftherios Avramidis, Eleftheria Briakou, Pinzhen Chen, Marzieh Fadaee, Markus Freitag, Roman Grundkiewicz, Yupeng Hou, Philipp Koehn, Julia Kreutzer, Saab Mansour, Stefano Perrella, Lorenzo Proietti, Parker Riley, Eduardo Sánchez, Patricia Schmidtova, Mariya Shmatova, and Vilém Zouhar. 2025. Findings of the WMT25 multilingual instruction shared task: Persistent hurdles in reasoning, generation, and evaluation. In Proceedings of the Tenth Conference on Machine Translation, pages 414–435, Suzhou, China. Association for Computational Linguistics.

Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James Validad Miranda, Alisa Liu, Nouha Dziri, Xinxi Lyu, Yuling Gu, Saumya Malik, Victoria Graf, Jena D. Hwang, Jiangjiang Yang, Ronan Le Bras, Oyvind Tafjord, Christopher Wilhelm, Luca Soldaini, and 4 others. 2025. Tulu 3: Pushing frontiers in open language model post-training. In Second Conference on Language Modeling.

Malik Marmonier, Rachel Bawden, and Benoît Sagot. 2025. Explicit learning and the LLM in machine translation. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 31372–31422, Suzhou, China. Association for Computational Linguistics.

Philippe Maurer-Cecchini. 2021. A grammar of Tuatschin. Number 3 in Comprehensive Grammar Library. Language Science Press, Berlin.

Sewon Min, Mike Lewis, Luke Zettlemoyer, and Hannaneh Hajishirzi. 2022. MetaICL: Learning to learn in context. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 2791–2809. Association for Computational Linguistics.

Manuel Mosquera, Melissa Robles, Johan Rodriguez, and Ruben Manrique. 2025. Improving low-resource translation with dictionary-guided fine-tuning and rl: A spanish-to-wayuunaiki study. Preprint, arXiv:2508.19481.

OpenAI. 2026. Openai o1 system card. Preprint, arXiv:2412.16720.

Carol J. Pebley and Thomas E. Payne. 2024. A grammar of Kagayanen. Number 8 in Comprehensive Grammar Library. Language Science Press, Berlin.

Renhao Pei, Yihong Liu, Peiqin Lin, François Yvon, and Hinrich Schuetze. 2025. Understanding in-context machine translation for low-resource languages: A case study on Manchu. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 8767– 8788. Association for Computational Linguistics.

Maja Popovi´c. 2015. chrF: character n-gram F-score for automatic MT evaluation. In Proceedings of the Tenth Workshop on Statistical Machine Translation, pages 392–395. Association for Computational Linguistics.

Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong He. 2020. Zero: memory optimizations toward training trillion parameter models. In Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis, pages 1–16.

Jean Rohleder. 2024. A grammar of Vamale. Number 9 in Comprehensive Grammar Library. Language Science Press, Berlin.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. 2024. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. Preprint, arXiv:2402.03300.

Garrett Tanzer, Mirac Suzgun, Eline Visser, Dan Jurafsky, and Luke Melas-Kyriazi. 2024. A benchmark for learning to translate a new language from one grammar book. In The Twelfth International Conference on Learning Representations.

Gion Peder Thöny. 1969. Rumantsch-Surmeir: Grammatica per igl idiom surmiran. Lia Rumantscha, Chur.

Jannis Vamvas, Ignacio Pérez Prat, Not Soliva, Sandra Baltermia-Guetg, Andrina Beeli, Simona Beeli, Madlaina Capeder, Laura Decurtins, Gian Peder Gregori, Flavia Hobi, Gabriela Holderegger, Arina Lazzarini, Viviana Lazzarini, Walter Rosselli, Bettina Vital, Anna Rutkiewicz, and Rico Sennrich. 2025. Expanding the WMT24++ benchmark with rumantsch grischun, sursilvan, sutsilvan, surmiran, puter, and vallader. In Proceedings of the Tenth Conference on Machine Translation, pages 1028–1047, Suzhou, China. Association for Computational Linguistics.

Jannis Vamvas, Ignacio Pérez Prat, Angela Heldstab, Dominic P. Fischer, Sina Ahmadi, and Rico Sennrich. 2026. Translation asymmetry in llms as a data augmentation factor: A case study for 6 romansh language varieties. Preprint, arXiv:2603.25489.

Eline Visser. 2022. A grammar of Kalamang. Language Science Press.

Jiaan Wang, Fandong Meng, and Jie Zhou. 2026. DeepTrans: Deep reasoning translation via reinforcement learning. Transactions of the Association for Computational Linguistics, 14:47–63.

Wenjie Yang, Mao Zheng, Mingyang Song, Zheng Li, and Sitong Wang. 2026. Ssr-zero: Simple selfrewarding reinforcement learning for machine translation. Preprint, arXiv:2505.16637.

Zheng Xin Yong, Hailey Schoelkopf, Niklas Muennighoff, Alham Fikri Aji, David Ifeoluwa Adelani, Khalid Almubarak, M Saiful Bari, Lintang Sutawika, Jungo Kasai, Ahmed Baruwa, Genta Winata, Stella Biderman, Edward Raff, Dragomir Radev, and Vassilina Nikoulina. 2023. BLOOM+1: Adding language support to BLOOM for zero-shot prompting. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 11682–11703. Association for Computational Linguistics.

Chen Zhang, Jiuheng Lin, Xiao Liu, Zekai Zhang, and Yansong Feng. 2025. Read it in two steps: Translating extremely low-resource languages with codeaugmented grammar books. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3977–3997. Association for Computational Linguistics.

Chen Zhang, Xiao Liu, Jiuheng Lin, and Yansong Feng. 2024a. Teaching large language models an unseen language on the fly. In Findings of the Association for Computational Linguistics: ACL 2024, pages 8783–8800. Association for Computational Linguistics.

Kexun Zhang, Yee Choi, Zhenqiao Song, Taiqi He, William Yang Wang, and Lei Li. 2024b. Hire a linguist!: Learning endangered languages in LLMs with in-context linguistic descriptions. In Findings of the Association for Computational Linguistics: ACL 2024, pages 15654–15669. Association for Computational Linguistics.

### A Hyperparameters

We perform full-parameter fine-tuning with DeepSpeed ZeRO-2 (Rajbhandari et al., 2020), using a learning rate of 1e−5 with a cosine scheduler and 10% warmup, an effective batch size of 128 (4 per device × 8 gradient accumulation steps). For RL, we use GRPO with chrF/100 as the outcome reward, a learning rate of 1e−6, batch size 64, n = 8 rollouts per prompt at temperature 1.0, clip ratio 0.25, KL coefficient 1e−4, and entropy coefficient 1e−3; training uses FSDP with SGLang (TP=4) for rollout and all ablation variants (nodict, nogram, nosent, taskonly) share the same hyperparameters.

### B LLM Usage Declaration

We used LLM in this research for two purposes. First, during manuscript writing, we used AI to edit language and polish the overall text. Second,

we used GPT-5 mini to generate synthetic bilingual dictionary entries for languages lacking sufficient dictionary resources, as described in Section 3.1.4. Since none of the authors is a speaker of the target languages, these synthetic entries were not manually validated for linguistic correctness; instead, we performed an indirect, utility-based selection between two prompt variants (v1 and v2) by comparing downstream chrF on the resulting translations and retaining the better-performing variant per language and direction.

We declare that ideation, methodology, experiment execution, and analysis are the original work of the authors. All AI-assisted text editing has been carefully reviewed by the authors to ensure accuracy.

### C Grammar Books Used For Data Curation

Language Grammar Open-access grammars (Language Science Press) Choguita Rarámuri Caballero (2022) Gyeli Grimm (2021) Japhug Jacques (2025) Kagayanen Pebley and Payne (2024) Tuatschin Maurer-Cecchini (2021) Ulwa (PNG) Barlow (2023) Vamale Rohleder (2024) Romansh idiom grammars (Lia Rumantscha, print) Surmiran Thöny (1969) Puter Ganzoni (1983)

- Table 6: Grammar-book sources used for data curation (§3.1). The eight grammars in the upper block are published by Language Science Press under the CC BY 4.0

license with LATEX source available, from which we extracted parallel translation examples. The two Romansh idiom grammars in the lower block are published by Lia Rumantscha as print volumes; for these we extracted examples from the printed text rather than LATEX source.

Language Family Split Dir. Dict. Par. Grammar Pairs Romansh continuum (ref. language: German) Puter Romance Seen →De ✓ ✓ Ganzoni, 1983

Vallader Romance Seen →De ✓ ✓ ✗ Sutsilvan Romance Seen →De ✓ ✓ ✗ Rumantsch Gr. Romance Seen →De ✓ ✓ ✗

13,892/ 1,462

Sursilvan Romance Similar →De ✓ ✓ ✗ 7,998/ Surmiran Romance Similar →De ✓ ✓ Thöny, 1969 737

Grammar-book languages (parallel data from LSP grammars; ref. language: English) Choguita Rarámuri Uto-Aztecan Seen ↔En △ ✓ Caballero, 2022

Gyeli Niger-Congo Seen ↔En △ ✓ Grimm, 2021 Japhug Sino-Tibetan Seen ↔En △ ✓ Jacques, 2025 Kagayanen Austronesian Seen ↔En △ ✓ Pebley and Payne, 2024 Tuatschin Romance Seen ↔En △ ✓ Maurer-Cecchini, 2021 Ulwa (PNG) Keram Seen ↔En △ ✓ Barlow, 2023 Vamale Austronesian Seen ↔En △ ✓ Rohleder, 2024

9,695 (7 langs)

Unseen / OOD languages (test-only; EN→X sets from Hus and Anastasopoulos, 2024)

Kalamang Gr. W. Bomberai Unseen ↔En ✓ ✓ Tanzer et al., 2024 750/100 Dinka Nilo-Saharan Unseen En→X ✓ ✓ 100 Wolof Niger-Congo Unseen En→X ✓ ✓ 100 Guarani Tupi-Guarani Unseen En→X ✓ ✓ 100 Kachin Sino-Tibetan Unseen En→X ✓ ✓ 100

- Table 7: Per-language resources and usage. For each language we list its Family, evaluation Split (Seen: train+test; Similar: held-out variety of a seen family; Unseen: no related training data), translation Dir.ection, the in-context resources used—bilingual Dict.ionary and Par.allel sentences (✓ curated, △ LLM-synthesised, ✗ none)—and the Grammar book source (citation if used, ✗ otherwise; see Table 6). Pairs gives train/test counts; Romansh and the 7-language group report group totals. Struck-through counts exist but are excluded from RL training. Romansh varieties use German as the reference language; all others use English. Kalamang uses the MTOB split (Tanzer et al., 2024); the four OOD EN→X test sets are from Hus and Anastasopoulos (2024).

- Table 8: Structure of the translation prompt for Romansh→German. The prompt consists of a linguistic introduction, a translation instruction, retrieved dictionary entries, parallel sentence examples, a grammar excerpt, and a closing instruction. Placeholder text is shown in italics. Dictionary entries are available for Puter and Vallader.

Component Description Template (German) LINGUISTIC INTRODUCTION

Background paragraph introducing the Romansh language family, its status as a Swiss minority language in Graubünden, and its five written varieties plus Rumantsch Grischun.

Rätoromanisch gehört zum romanischen Zweig der indogermanischen Sprachfamilie. Es ist eine Minderheitensprache im Schweizer Kanton Graubünden ...

TRANSLATION INSTRUCTION

Specifies the source variety, target language, and the sentence to be translated.

Übersetze von [Variety] nach Deutsch: [source sentence]

DICTIONARY ENTRIES (≤100 entries)

Retrieved bilingual dictionary entries for words in the source sentence. Each entry provides the closest match from a varietyspecific dictionary.

Um bei der Übersetzung zu helfen, hier ist einer der ähnlichsten Einträge zu “[word]” im [Variety]Deutschen Wörterbuch: [Variety]: [source term] Deutsch: [target term]

PARALLEL SENTENCES (≤5 pairs)

Retrieved parallel sentence pairs containing words similar to those in the source sentence. Provides in-context translation examples.

Um bei der Übersetzung zu helfen, hier ist ein übersetzter Satz mit Wörtern ähnlich zu “[word]” in einer Liste übersetzter [Variety]-Deutscher Referenzsätze: [Variety]: [source sentence]

Deutsch: [target sentence]

GRAMMAR PASSAGE An excerpt from a grammar book of the source variety, providing morphological and phonological rules.

Um bei der Übersetzung zu helfen, hier ein aus einem [Variety] Grammatiksbuch entnommener Abschnitt:

[grammar excerpt]

CLOSING INSTRUCTION

Instructs the model to provide step-bystep metalinguistic reasoning and place the final translation in a \boxed{} environment.

Schreibe nun die Übersetzung des Satzes: [source sentence]

Bitte lege deine metalinguistischen Überlegungen Schritt für Schritt dar und setze deine endgültige Übersetzung in eine Box: \boxed{}

- Table 9: Structure of the translation prompt for Kalamang→English. The prompt includes dictionary entries, parallel sentence examples, and grammar passages. The same structure applies to other endangered languages (Gyeli, Japhung, Ulwa, etc.) where a bilingual dictionary is available. Placeholder text is shown in italics.

Component Description Template (English) LINGUISTIC INTRODUCTION

Brief description of the language family and geographic location of the source language.

Kalamang is a Papuan language of the Greater West Bomberai family. It is spoken in East Indonesia.

TRANSLATION INSTRUCTION

Specifies the source language, target language, and the sentence to be translated.

Your task is to translate from [Language] to English: [source sentence]

DICTIONARY ENTRIES

Retrieved entries from a bilingual dictionary for words in the source sentence. Each entry provides the closest match with part of speech and English translation.

To help with the translation, here is one of the closest entries to “[word]” in the [Language]-English bilingual dictionary:

[Language] word: [source term] Part of speech: [POS] English translation: [target term]

PARALLEL SENTENCES (≤5 pairs)

Retrieved parallel sentence pairs containing words similar to those in the source sentence. Two examples per source word are provided.

To help with the translation, here is a translated sentence with words similar to “[word]” in a list of translated [Language]-English reference sentences: [Language] sentence: [source sentence]

English translation: [target sentence]

GRAMMAR PASSAGE An excerpt from a descriptive grammar of the source language, covering relevant morphological or syntactic phenomena.

To help with the translation, here is a passage retrieved from a [Language] grammar book: [grammar excerpt]

CLOSING INSTRUCTION

Instructs the model to provide step-bystep metalinguistic reasoning and place the final translation in a \boxed{} environment.

Now write the translation of the sentence: [source sentence]

Please provide your meta-linguistic reasoning step by step, and put your final translation within \boxed{}.

