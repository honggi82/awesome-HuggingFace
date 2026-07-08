arXiv:2603.09160v1[cs.CV]10Mar2026

# RubiCap: Rubric-Guided Reinforcement Learning for Dense Image Captioning

Tzu-Heng Huang‡,§, Sirajul Salekin†, Javier Movellan†, Frederic Sala§, Manjot Bilkhu† †Apple, §University of Wisconsin—Madison, ‡Work done during an internship at Apple

Denseimagecaptioningiscriticalforcross-modalalignmentinvision-languagepretrainingandtext-to-imagegeneration, but scaling expert-quality annotations is prohibitively expensive. While synthetic captioning via strong vision-language models (VLMs) is a practical alternative, supervised distillation often yields limited output diversity and weak generalization. Reinforcement learning (RL) could overcome these limitations, but its successes have so far been concentrated in verifiable domains that rely on deterministic checkers—a luxury not available in open-ended captioning. We address this bottleneck with RubiCap, a novel RL framework that derives fine-grained, sample-specific reward signals from LLMwritten rubrics. RubiCap first assembles a diverse committee of candidate captions, then employs an LLM rubric writer to extract consensus strengths and diagnose deficiencies in the current policy. These insights are converted into explicit evaluation criteria, enabling an LLM judge to decompose holistic quality assessment and replace coarse scalar rewards with structured, multi-faceted evaluations. Across extensive benchmarks, RubiCap achieves the highest win rates on CapArena, outperforming supervised distillation, prior RL methods, human-expert annotations, and GPT-4V-augmented outputs. On CaptionQA, it demonstrates superior word efficiency: our 7B model matches Qwen2.5-VL-32B-Instruct, and our3Bmodelsurpassesits7Bcounterpart. Remarkably,usingthecompactRubiCap-3Basacaptionerproducesstronger pretrained VLMs than those trained on captions from proprietary models.

Correspondence: Manjot Bilkhu: mbilkhu@apple.com; Tzu-Heng Huang: thuang273@wisc.edu Date: March 11, 2026

## 1 Introduction

Dense image captioning shifts traditional vision-language tasks from providing global scene summaries (Chen et al., 2015; Vinyals et al., 2015; Sharma et al., 2018) to generating fine-grained, region-level descriptions of objects, attributes, and spatial relationships (Chen et al., 2020; Park and Paik, 2023; Zhou et al., 2024; Deitke et al., 2025). This richer form of visual understanding has become a core building block for crossmodal alignment in visual-language pretraining (Radford et al., 2021; Zhang et al., 2024), visual instruction tuning (Liu et al., 2023), and controllable text-to-image generation (Kim et al., 2023).

However, scaling dense captioning without sacrificing quality remains challenging. Manual annotation demands expert-level visual perception and precise language grounding, a combination that is prohibitively expensive at the scale required by frontier models. Recent efforts therefore turn to using stronger visionlanguage models (VLMs) to generate synthetic captions, which are usually then distilled into smaller, specialized captioners through supervised fine-tuning (SFT) (Yang et al., 2023; Chen et al., 2024; Li et al., 2024; Singla et al., 2024). However, SFT comes with pitfalls: (i) it tends to collapse linguistic diversity through memorization (Chu et al., 2025) (e.g., replicating teacher’s narrative style rather than improving visual understanding), (ii) induces severe catastrophic forgetting of pretrained capabilities (McCloskey and Cohen, 1989; Lai et al., 2025; Shenfeld et al., 2025), and (iii) degrades when teacher and student distributions mismatch (Gerstgrasser et al., 2024).

These shortcomings motivate a shift toward feedback-driven optimization, specifically through reinforcement learning with verifiable rewards (RLVR) (Team et al., 2025; Guo et al., 2025). RLVR has driven major advances in mathematical reasoning (Shao et al., 2024) and code generation (Luo et al., 2025; Liu and

Zhang, 2025), where solutions are hard to produce but trivial to verify. Dense captioning, however, does not naturally admit such verification: caption quality is inherently open-ended, subjective, and contextdependent, with no deterministic verifier available. This verification bottleneck remains the key obstacle to bringing RL into dense captioning.

Prior attempts to provide RL rewards primarily fall into two categories, each with drawbacks that stymie their adoption:

- • Lexical NLP Metrics: Metrics such as CIDEr (Vedantam et al., 2015), and ROUGE-L (Lin, 2004) measure n-gram overlap against reference captions. While computationally efficient, they are strictly reference-bound and are insensitive to semantic equivalence or compositional variation, rewarding lexical similarity rather than descriptive accuracy.
- • VLM-as-a-Judge: Alternatively, frontier VLMs are increasingly used as holistic evaluators that assign quality ratings (Lee et al., 2024; Chen et al., 2024; Lee et al., 2024). However, these scores are often coarse and opaque; they oversimplify complex quality dimensions into a single scalar, providing minimal diagnostic insight into specific failures.

We address this verification bottleneck by deriving fine-grained, sample-specific evaluation criteria, transforming an inherently subjective judgment into a structured, multi-faceted assessment. Specifically, we propose RubiCap, a novel RL framework that decomposes caption quality using a suite of rubrics and leverages them to produce reliable RL rewards. RubiCap begins with a committee of diverse VLMs producing candidate descriptions for each image. An LLM rubric writer then extracts the consensus while identifying deficiencies in the student’s current policy. These targeted gaps are transformed into interpretable, easy-tocheck rules that an LLM judge can readily apply to caption assessment, yielding more precise feedback that goes far beyond what a global quality score can offer.

This design offers several key advantages. First, it opens multi-dimensional evaluation: rubrics can simultaneously check object presence, attribute correctness, spatial reasoning, and hallucination. Second, rubric creation scales naturally: for human annotators, producing coherent evaluation criteria is cognitively demanding, even harder than writing captions themselves, whereas an LLM rubric writer synthesizes them reliably at scale. Finally, rubrics are steerable by design: by directing the rubric writer toward specific weaknesses, the reward signal can be precisely aimed at the areas most in need of improvement.

We evaluate RubiCap extensively across diverse captioning benchmarks. On CapArena (Cheng et al., 2025), judged by GPT-4.1 (OpenAI, 2024), our 7B model achieves the highest win rates across all compared methods, notably outperforming human-expert annotations and proprietary model outputs. In a blind ranking evaluation, RubiCap-7B earns the highest proportion of rank-1 assignments among all models—including 72B and 32B frontiers—achieving the lowest hallucination penalty and strongest accuracy. Beyond captioning quality, RubiCap exhibits improved word efficiency on CaptionQA (Yang et al., 2025) by prioritizing salient content; specifically, RubiCap-3B models outperform 7B base models, while our 7B models can match 32B-scale frontiers. Additionally, RubiCap-trained models preserve pretrained capabilities across 10 VLM benchmarks, substantially mitigating severe forgetting observed in supervised distillation. Finally, we show that a compact RubiCap-3B captioner can serve as a high-quality data source for VLM pretraining, resulting in stronger pretrained models than those built on GPT-4V captions.

We summarize our contributions as follows:

- • We identify the verification bottleneck in RL for dense captioning and address it through synthetic, samplespecific rubrics for fine-grained, reliable reward signals.
- • We propose an automated rubric synthesis pipeline that leverages diverse model consensus and targeted deficiency analysis to decompose holistic evaluation into multi-faceted quality checks.
- • We conduct extensive experiments across six evaluation axes, demonstrating that RubiCap consistently delivers the largest gains on base models and outperforms diverse baselines in both caption quality and word efficiency.
- • We show that RubiCap-7B outperforms 72B and 32B frontier models in blind ranking evaluations, earning the highest proportion of rank-1 assignments while achieving the lowest hallucination penalty and strongest accuracy.

- • We demonstrate that RubiCap-trained models serve as stronger captioners than proprietary systems, yielding better pretrained VLMs at scale.

## 2 Related Works

Our work sits at the intersection of two research threads: dense image captioning and reinforcement learning for vision-language models.

Dense Image Captioning. Early image captioning research was driven by datasets such as MS COCO (Chen et al., 2015) and Flickr30k (Plummer et al., 2015), where human annotators provided concise, scene-level summaries. Recent efforts have shifted toward dense captioning, which requires both global description and the localization of multiple salient regions. However, collecting such fine-grained annotations at scale is substantially more demanding, motivating researchers to explore synthetic alternatives. One line of research reuses web-crawled alt-text, refining it with LLMs for greater clarity and descriptiveness (Fan et al., 2023; Lai et al., 2024; Singla et al., 2024). Other methods prompt strong VLMs directly (Chen et al., 2024), fuses outputs from multiple VLMs (Yu et al., 2024), or enrich captions with predictions from external vision experts (Li et al., 2024, 2025; Zhu et al., 2025). Most of these approaches rely on SFT to distill synthetic captions into a specialized captioner. RubiCap takes a different path: instead of imitating a fixed teacher, it trains via RL with rewards derived from fine-grained, sample-specific rubrics—letting the model discover better captions rather than merely reproducing existing ones.

Reinforcement Learning in Visual Language Models. Reinforcement learning has recently driven major advances in LLM capabilities, particularly in domains where correctness can be automatically verified (Shao et al., 2024; Liu and Zhang, 2025). In vision-language models, RL-based post-training has been applied to improve grounding (Shen et al., 2025), detection and classification (Liu et al., 2025), chart-based question answering (Sinha et al., 2025), and tool-augmented visual mathematical reasoning (Zhou et al., 2025)—tasks that all admit deterministic verification via IoU, classification accuracy, or multiple-choice correctness. Open-ended dense captioning, where outputs are free-form and no ground-truth verifier exists, presents a fundamentally harder challenge. The closest concurrent work, CapRL (Xing et al., 2025), constructs multiple-choice questions from a strong VLM and treats derived accuracy as a proxy reward. However, MCQ options are inherently limited in coverage—any failure mode absent from the option set goes unpenalized, fundamentally capping evaluation by design. RubiCap sidesteps this limitation. By assembling a diverse

VLM committee to extract consensus-grounded rubrics, it produces open-ended evaluation criteria capable of surfacing failures that no fixed choice set could anticipate.

## 3 Framework

RubiCap addresses the verification challenge that arises when applying RL to dense image captioning. Its key insight is to exploit sample-specific evaluation rubrics—clear, interpretable rules that decompose caption quality into fine-grained, image-conditioned reward signals. As illustrated in Fig. 1, the framework operates in two stages: (1) Automated Rubric Synthesis (Sec. 3.2), which distills teacher consensus into per-image evaluation criteria; and (2) Rubric-Guided Reinforcement Learning (Sec. 3.3), which uses the derived criteria to optimize a student captioning policy.

### 3.1 Setup and Notation

We begin by establishing our notation, then describe each stage in turn. Let x denote an input image. The student VLM being trained is parameterized by θs, with captioning policy πθ

. Given x and a descriptive prompt (e.g., “Describe this image in detail”), the student generates a caption cstudent ∼ πθ

s

(· | x). We assume access to a committee of K teacher VLMs T = {Tk}Kk=1, which collectively produce candidate captions Cteacher(x) = {cteacherk }Kk=1. Together, these capture a broader range of visual elements and descriptions than any single reference caption.

s

[Figure 1]

- Figure 1 Overview of the RubiCap framework. A committee of VLMs first produces diverse candidate captions and consolidate them into a consensus. An LLM rubric writer then diagnoses the student’s specific deficiencies and transforms them into fine-grained, interpretable evaluation criteria. An LLM judge applies these rubrics to assess caption rollouts, replacing coarse scalar rewards with structured, multi-dimensional signals that alleviate the verification bottleneck in RL.

### 3.2 Automated Rubric Synthesis

Rather than relying on a single golden reference, we leverage the collective expertise of teacher models to construct evaluation rubrics tailored to each training sample. Given an image x, we collect teacher captions Cteacher(x) and generate a student caption cstudent from the current policy. An LLM rubric writer is then prompted with the image, the student caption, and the teacher captions to produce targeted evaluation criteria via three sequential steps:

- 1. Identify consensus aspects: The rubric writer extracts key descriptive elements on which the majority of teacher captions agree, conditioned on the image x. An element—whether an object, attribute, spatial relationship, or contextual interpretation—is treated as ground truth only if at least ⌈K/2⌉ teachers describe it accurately, preventing any single noisy teacher from biasing the rubric.
- 2. Diagnose student deficiencies: The rubric writer performs a comparative analysis between cstudent and the teacher consensus. Crucially, the rubric writer focuses on discriminative deficiencies—flagging only the aspects the student failed to capture or misrepresented. This ensures the reward signal is non-redundant and targeted. Identified failures are categorized by severity: (i) critical failures such as main-subject misidentification or hallucination of major elements; (ii) important gaps such as missing secondary objects, imprecise attributes, or incorrect spatial logic; and (iii) minor polish issues such as phrasing clarity or fine-grained detail richness.
- 3. Formulate targeted rubrics. For each diagnosed deficiency, the rubric writer is asked to define a

binary, easy-to-check criterion rm paired with a severity weight wm for minor, important, and critical failures, respectively. These weights can be viewed as hyperparameters; inspired by other rubric-based works (Shao et al., 2025; Gunjal et al., 2025), here we use {1.0,2.0,3.0}. Finally, every criterion must admit an unambiguous pass/fail judgment.

The resulting sample-specific rubric set is:

R x, cstudent, {cteacherk }Kk=1 = (rm, wm) Mm=1,

where each rm is a human-readable binary statement. We present a rubric example in Fig. 1, where the rubric writer asks the student model to identify text inscriptions on the cake; a caption that correctly mentions “24 CARROT CAKE” will be rewarded accordingly. By construction, R is discriminative—every criterion

targets a real quality gap relative to teacher consensus. Moreover, it is image-conditioned, adapting to both the visual content of x and the student’s current failure modes, rather than applying a fixed, generic checklist.

In practice, we develop the teacher committee with five VLMs, including Gemini 2.5 Pro (Comanici et al., 2025), GPT-5 (Singh et al., 2025), Qwen2.5-VL-72B-Instruct (Bai et al., 2025), Gemma-3-27B-IT (Kamath et al., 2025), and Qwen3-VL-30B-A3B-Instruct (Bai et al., 2025) to ensure diversity. We designate Gemini

- 2.5 Pro as our primary rubric writer and anonymize teacher identities within the synthesis prompts to prevent any stylistic bias. The complete prompt templates are provided in Appendix B.
- 3.3 Rubric-Guided Reinforcement Learning

To convert R into a RL reward signal, an LLM-based judge evaluates cstudent against each criterion rm, producing a binary satisfaction score:

yˆm =

1 if rm is fully satisfied, 0 otherwise.

The overall caption quality is then expressed as a normalized weighted scalar reward:

M m=1 wm · yˆm

G(x,cstudent) =

,

M m=1 wm

which measures the proportion of identified quality gaps that the student has successfully addressed, weighted by their severity.

We optimize πθ

using group relative policy optimization (GRPO) (Shao et al., 2024). For each image x, we

s

sample N rollouts {cstudenti }Ni=1 and compute rewards {G(x,cstudenti )}Ni=1. The advantage of each rollout is estimated relative to the group:

G(x,cstudenti ) − mean {Gi} std {Gi}

,

Ai =

rewarding captions that outperform their peers while penalizing those below the group mean. The student policy is updated by minimizing:

LGRPO(θs) = − Ex,{cstudent

i }

N

1 N

min ρiAi, clip ρi, 1−ϵ, 1+ϵ Ai , (3.1)

i=1

(cstudenti | x) is the importance sampling ratio, ϵ is the clipping threshold, and πθ

(cstudenti | x)/πθ

where ρi = πθ

ref

s

is the reference policy.

ref

By training against discriminative, sample-specific rubrics, the student is incentivized to address the precise visual details it previously overlooked, progressively closing the quality gap toward teacher consensus. The full RubiCap algorithm is summarized in Appendix E.

## 4 Experiments

We empirically evaluate the effectiveness of RubiCap by comparing the dense captions it generates against four reference points: those produced by (i) the student base model, (ii) a diverse set of RL and SFT baselines, (iii) human-expert annotations, and (iv) proprietary model outputs. Our evaluation is organized around six core claims:

- C1. Greater Self-Improvement: RubiCap achieves a higher win rate than supervised distillation and RL approaches that rely on NLP metrics or VLM judges, when all methods are compared against the same base model (Sec. 4.1).
- C2. Superior Caption Quality: RubiCap produces higher-quality captions than human-expert annotations, proprietary model captions, and outputs from concurrent work, CapRL (Xing et al., 2025) (Sec. 4.2).

- C3. Knowledge Forgetting Mitigation: Unlike SFT-based methods, RubiCap substantially reduces catastrophic forgetting, better preserving the fine-tuned model’s pretrained capabilities (Sec. 4.3).
- C4. Advantage over Rubric-Augmented SFT: Even when identical rubrics are explicitly incorporated into SFT, RubiCap consistently produces higher-quality captions through better exploration across different model scales (Sec. 4.4).
- C5. Higher Information Density: RubiCap exhibits superior word efficiency, providing more salient

content under strict length constraints. (Sec. 4.5).

C6. Stronger Pretraining Utility: Using RubiCap-trained models as annotators for pretraining yields

stronger VLMs compared to those annotated by proprietary systems (Sec. 4.6).

Datasets. We construct two training datasets from different dense captioning sources: PixMoCap (Deitke et al., 2025) and DenseFusion-4V-100K (Li et al., 2024). From each, we randomly sample 50,000 images for training and reserve 500 images as held-out evaluation sets. These two datasets provide distinct forms of supervision that enable different evaluation objectives: (i) PixMoCap contains high-quality human-expert annotations where three annotators independently produce spoken descriptions for each image, which are subsequently refined and rewritten by an LLM to improve clarity and ensure stylistic consistency. (ii) DenseFusion-4V-100K contains dense captions generated by GPT-4V (OpenAI, 2023) and augmented with multiple perceptual attributes, representing strong outputs from a proprietary VLM. Captions from both datasets serve as reference annotations for quality comparison throughout our experiments.

Models and Baselines. We validate RubiCap through full-parameter fine-tuning on student models from the Qwen VLM family at multiple scales including Qwen2.5-VL-7B-Instruct (Bai et al., 2025), Qwen2.5-VL-3BInstruct (Bai et al., 2025), and Qwen2-VL-2B-Instruct (Wang et al., 2024). Caption rollouts are evaluated against each rubric criterion by a Qwen2.5-7B-Instruct, which serves as our LLM judge and provides binary satisfaction scores. Training configurations and compute details are provided in Appendix F. We compare RubiCap against five families of baselines:

- • Base Model: the untuned student model, serving as a reference point for measuring self-improvement.
- • Supervised Distillation: supervised fine-tuning on captions from PixMoCap, DenseFusion, and a strong frontier model (Qwen2.5-VL-72B-Instruct (Bai et al., 2025)), all treated as ground-truth image descriptions.
- • NLP Metric–based RL: reinforcement learning using ROUGE-L (Lin, 2004) as the reward signal, measuring lexical overlap between caption rollouts and reference captions.
- • VLM Judge–based RL: two baseline variants from the RaR framework (Gunjal et al., 2025): DirectLikert and Reference-Likert. Both use Qwen2.5-VL-7B-Instruct as a VLM judge to assign quality scores on a 0–10 scale. Reference-Likert evaluates quality relative to a reference caption, whereas Direct-Likert scores captions without comparisons. The employed prompts are offered in Appendix C.
- • CapRL-3B (Xing et al., 2025): a method that leverages 75,000 well-curated images and multiple-choice question–based caption quality scores as RL rewards. We compare at the 3B scale where it shares our student base model.

### 4.1 Comparison with Supervised Distillation and Other RL Objectives

Setup. We evaluate caption quality using CapArena (Cheng et al., 2025), a recent dense captioning benchmark that employs a strong VLM judge to compute pairwise win–loss rates between caption pairs. We adopt the official CapArena evaluation protocol and use GPT-4.1 (OpenAI, 2024) as the judge. To the best of our knowledge, CapArena’s annotated preferences exhibit the strongest alignment with human judgments among existing automatic evaluation metrics. The full judging prompt is provided in Appendix D.

In addition to CapArena, we report a comprehensive suite of complementary metrics. These include modelbased measures including CAPTURE (Dong et al., 2024) and SPECS (Chen et al., 2025) (a specialized CLIP-based metric), as well as traditional lexical overlap metrics such as ROUGE-L (Lin, 2004), METEOR (Banerjee and Lavie, 2005), and BLEU (Papineni et al., 2002). Detailed metric descriptions and complete results across model scales appear in Appendix G.

##### PixMoCap

###### v.s. Expert-Annotated Caption

###### v.s. Base Model Generated Caption

BLINK

GQA ChartQA

68

70

AI2D

66

100

64

80

RealWorldQA

60

65

40

DocVQA

20

62

WinRate(%)

WinRate(%)

###### +13.4%

60

###### +20.8%

60

OCRBench

58

InfoVQA

RubiCap-7B-PixMoCap

RubiCap-7B-PixMoCap

55

56

Reference-Likert

Reference-Likert

Direct-Likert

Direct-Likert

OCRVQA

TextVQA

SFT (72B-distilled)

SFT (72B-distilled)

54

Metric-Based RL

Metric-Based RL

SFT (Expert-distilled)

SFT (Expert-distilled)

50

RubiCap-7B-PixMoCap: 70.7%

Metric-Based RL: 44.2%

52

SFT (Expert-distilled): 18.2%

Direct-Likert: 59.9%

0 387 774 1161 1548 1935 2322 2709 3090

0 387 774 1161 1548 1935 2322 2709 3090

SFT (72B-distilled): 12.5%

Reference-Likert: 38.4%

Training Step

Training Step

##### DenseFusion-4V-100K

###### v.s. Base Model Generated Caption

###### v.s. GPT-4V Generated Caption

BLINK

GQA ChartQA

64

52

AI2D

62

50

100

80

###### +8.4%

RealWorldQA

60

60

40

DocVQA

48

20

WinRate(%)

WinRate(%)

58

46

###### +14.4%

56

OCRBench

44

InfoVQA

RubiCap-7B-DenseFusion

RubiCap-7B-DenseFusion

54

42

Reference-Likert

Reference-Likert

Direct-Likert

Direct-Likert

OCRVQA

TextVQA

52

SFT (72B-distilled)

SFT (72B-distilled)

40

Metric-Based RL

Metric-Based RL

SFT (GPT-4V-distilled)

SFT (GPT-4V-distilled)

50

RubiCap-7B-DenseFusion: 70.7%

Metric-Based RL: 43.7%

38

SFT (GPT-4V-distilled): 51.4%

Direct-Likert: 65.2%

0 386 772 1158 1544 1930 2316 2702 3088

0 386 772 1158 1544 1930 2316 2702 3088

SFT (72B-distilled): 9.0%

Reference-Likert: 62.0%

Training Step

Training Step

- Figure 2 Evaluation of RubiCap-7B models trained on PixMoCap and DenseFusion datasets. The top row shows results for PixMoCap, and the bottom row for DenseFusion. For each setting: Left: CapArena win rates against the base model across training steps, demonstrating consistent and strongest self-improvement over SFT variants and RL baselines. Middle: CapArena win rates against high-quality human-expert annotations (PixMoCap) and GPT-4V-augmented captions (DenseFusion), showing RubiCap surpasses professional and proprietary labeling systems. Right: Radar plots over 10 VLM benchmarks, validating RubiCap has better knowledge preservation of pretrained capabilities over supervised distillation. Average performance scores are provided in the boxes.

Results. We begin by comparing RubiCap captions and baseline methods against those produced by the base model. This setting measures self-improvement, where the untuned model serves as the reference point at a 50% win rate. The leftmost panels in Fig. 2 report results for 7B-scale models on the two datasets. RubiCap-7B consistently outperforms all baselines, including SFT variants trained on humanexpert annotations, GPT-4V-augmented captions, and synthetic captions from Qwen2.5-VL-72B-Instruct. Specifically, RubiCap achieves a +20.8% win–rate improvement on PixMoCap and a +14.4% improvement on DenseFusion. These yield the highest self-improvement among all compared methods. We further validate these gains at the 3B scale in Fig. 3 and report 2B-scale results in Appendix G. Across all settings and model sizes, RubiCap consistently achieves the highest win rates relative to base models.

Moreover, we observe a failure mode emerges from the Reference-Likert baseline. At the 3B and 2B scales, this method eventually induces self-praising behavior, obtaining high rewards without producing meaningful or detailed image descriptions 1. This behavior represents a classic form of reward hacking, in which a model discovers a shortcut that maximizes the reward signal while entirely bypassing the intended generation objective (DeepMind, 2023; Amodei et al., 2016). As shown in Fig. 3 (purple bar results), Reference-Likert results in the lowest win rates in the PixMoCap setting (achieving only 7.8% against the base model and 4.0% against human-expert annotations).

These findings expose a fundamental limitation of RL reward formulations: relying on holistic, unspecified

1Across virtually all test images, Reference-Likert–trained models produce captions such as: “Certainly! Here is a detailed description of the image: This image description is absolutely correct and complete. It is absolutely clear and detailed.”

Qwen2.5-VL-3B-Instruct

Qwen2.5-VL-3B-Instruct CapRL-3B Metric-based RL Direct-Likert Reference-Likert RubiCap-3B-PixMoCap

68.6

59.2

70

60

CapRL-3B Metric-based RL Direct-Likert

| |
|---|

64.0

| |
|---|

| |
|---|

53.2

59.4

50.8

| |
|---|

| |
|---|

60

50.0

Reference-Likert RubiCap-3B-DenseFusion

48.6

50

| |
|---|

54.0

45.4

| |
|---|

| |
|---|

50.0

50

40

WinRate(%)

WinRate(%)

42.8

40

37.4

35.8

34.0 34.4

30

26.2

30

23.8

20.0

19.4

20

17.8

20

15.2

10

10

7.8

4.0

0

0

vs. Base Model Caption vs. GPT-4V Generated Caption

vs. Base Model Caption vs. Expert-Annotated Caption

- Figure 3 Performance of RubiCap-3B across PixMoCap and DenseFusion settings. Left: Results under PixMoCap. Right: Results under DenseFusion. In both settings, RubiCap-3B achieves the highest CapArena win rates among all compared methods.

quality judgments—a “vibe check”—creates degenerate reward surfaces that admit shortcut generations and provide arbitrary training signals. In contrast, RubiCap overcomes this by grounding evaluation in explicit, sample-specific criteria, guiding the model toward detailed and accurate captioning.

### 4.2 Comparison with Human Annotations and Proprietary Models

Setup. Next, we conduct a more challenging evaluation by directly comparing RubiCap captions against two strong references: expert-refined annotations from PixMoCap and GPT-4V-enriched outputs from DenseFusion, representing professional human supervision and proprietary model outputs, respectively. We also include a ranking-style comparison where we prompt GPT-4.1 to jointly rank captions from RubiCap models, 72B, 32B frontiers, the base model, and the dataset reference captions. To prevent bias, model identities are anonymized—the judge receives only the captions, with no information about their source. This evaluation is inspired by the EXPERT metric (Kim et al., 2025), which decomposes caption quality into sub-dimensions, and we rank models by their total scores. Our evaluation prompt is offered in Appendix D. Finally, we include a head-to-head comparison against CapRL-3B (Xing et al., 2025), evaluating RubiCap at the 3B and 2B scales.

Results. Results for 7B-scale models are shown in the middle panels of Fig. 2. RubiCap consistently achieves the highest win rates across all compared methods. Relative to the base model, RubiCap-7B trained on PixMoCap and DenseFusion improves win rates by 13.4% and 8.4%, respectively. Importantly, RubiCapgenerated captions are preferred over both expert-refined human annotations and GPT-4Venriched outputs—winning more than half of all pairwise comparisons.

A particularly revealing pattern emerges from the SFT baselines. Fine-tuning on human-annotated captions (PixMoCap) hurts base model performance (red line, middle panel), whereas distilling from a same-family model—Qwen2.5-VL-72B-Instruct—yields consistent gains (yellow line, middle panel). The difference points to a distributional mismatch: even objectively high-quality captions can be misaligned with the student’s inherent distribution, making them poor fine-tuning targets. In contrast, RubiCap avoids this supervision limitation by optimizing against the student’s own rollouts.

We present the blind ranking results in Fig. 4, where RubiCap-7B-PixMoCap earns the highest proportion of rank-1 assignments—surpassing both 72B and 32B frontier models. Moreover, the sub-metric breakdown further reveals why: RubiCap achieves the lowest hallucination penalty and the strongest accuracy score among all models, while matching the 72B model on completeness and clarity. We demonstrate a series of

100

| | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | || |
|---|
<br><br>| |
|---|
| |Rank Rank Rank|1<br>2<br>3<br>| |
| | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |Rank Rank|4<br>5<br>| |
| | | | | | | | | | | | | | || |
|---|
| | | | |
| | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | |

###### PercentageofSamples(%)

80

60

40

20

0

RubiCap-7B-PixMoCapQwen2.5-VL-72B-InstructQwen2.5-VL-32B-InstructBaseModelExpert-Annotated

10

Accuracy Completeness Clarity

9.0 9.0 9.0

| |
|---|

8.9

8.4 8.5

| |
|---|

8.2

Hallucination

7.9 7.8

8

7.8

7.4

7.3

AverageScore

6.9

6.9

6

5.8

4

3.5

2.4 2.3

1.9 2.0

2

0

RubiCap-7B-PixMoCapQwen2.5-VL-72B-InstructQwen2.5-VL-32B-InstructBaseModelExpert-Annotated

- Figure 4 Left: Rank Distribution per Model. Using the PixMoCap setting as an example, RubiCap achieves a substantially higher proportion of rank-1 assignments, demonstrating consistently preferred captions despite its smaller scale. Right: Sub-metric Breakdown per Model. Scores across four evaluation dimensions show that RubiCap achieve lower hallucination penalty and stronger accuracy and clarity, while matching the 72B model overall.

qualitative comparisons in Appendix I.

Fig. 5 (left) shows CapArena win rates against CapRL-3B. RubiCap-3B achieves win rates of 62% on PixMoCap and 59% on DenseFusion, comfortably outperforming CapRL-3B. Notably, even RubiCap-2B remains competitive: models built on Qwen2-VL-2B-Instruct achieve win rates of 54% and 51.4%, despite being trained on a smaller size model and 25% less data than CapRL-3B. We also observe that CapRL-

- 3B exhibits similar self-praising behavior, frequently appending captions with meta-commentary such as: “This detailed description should provide a pure text model with sufficient information to answer any related questions about the image.”

### 4.3 Mitigating Catastrophic Forgetting

Setup. Continued supervised fine-tuning can risk in forgetting pretrained capabilities. To assess this, we test fine-tuned models on 10 VLM benchmarks spanning four categories using VLMEvalKit (Duan et al., 2024): (i) Visual Reasoning (GQA (Hudson and Manning, 2019), BLINK (Fu et al., 2024)), (ii) Scientific Understanding (AI2D (Kembhavi et al., 2016)), (iii) OCR (RealWorldQA (xAI, 2024), OCRBench (Liu et al., 2024), TextVQA (Singh et al., 2019), OCRVQA (Mishra et al., 2019)), and (iv) Document Extraction (InfoVQA (Mathew et al., 2022), DocVQA (Mathew et al., 2020), ChartVQA (Masry et al., 2022)). We report the average performance across all benchmarks as a measure of generalization retention.

Results. We first focus on 7B-scale models, results are shown in the rightmost panels of Fig. 2, with 3B- and

- 2B-scale results provided in Appendix H. Across all model scales, RubiCap consistently achieves the highest average performance over the 10 benchmarks, substantially outperforming SFT-based models, which exhibit the largest degradation. These results underscore the limitations of SFT on open-ended tasks, where it often impairs generalization, whereas RubiCap training more effectively mitigates catastrophic forgetting.

### 4.4 Effectiveness beyond Rubric-Augmented SFT.

Setup. A natural question is whether RubiCap’s gains can be replicated by simply incorporating derived rubrics into an SFT pipeline. To answer this, we construct a rubric-augmented SFT baseline that has access to the same evaluation criteria as RubiCap but trains via supervised imitation rather than RL. Concretely, for each image the student model first generates an initial caption and is then prompted with the full set of rubrics to rewrite it such that all criteria are satisfied. The model is subsequently fine-tuned on these rewritten captions via standard SFT. This baseline controls for teachers’ collective expertise and rubric

| |62.0% Win<br><br>1.0% Tie<br><br>37.0% Lose<br><br>59.0% Win<br><br>0.6% Tie<br><br>40.4% Lose<br><br>54.0% Win<br><br>0.6% Tie<br><br>45.4% Lose<br><br>51.4% Win<br><br>0.0% Tie<br><br>48.6% Lose|
|---|---|
| | |
| | |
| | |
| | |

RubiCap-3B-PixMoCap

v.s. CapRL-3B

RubiCap-3B-DenseFusion

v.s. CapRL-3B

RubiCap-2B-PixMoCap

v.s. CapRL-3B

RubiCap-2B-DenseFusion

v.s. CapRL-3B

80

70.8

68.6

70

66.0

65.8

64.0

59.8

60

50

WinRate(%)

42.8

40.2

40

30

20

Rubric-Augmented SFT (3B)

RubiCap-3B-PixMoCap

10

Rubric-Augmented SFT (7B)

RubiCap-7B-PixMoCap

0

vs. Base Model Caption vs. Expert-Annotated Caption

- Figure 5 Left: Comparison with CapRL-3B in CapArena Evaluation. RubiCap variants achieve higher win rates across both 3B and 2B model sizes while using 25% less training data. Right: Comparison between rubricaugmented SFT and RubiCap across model scales. Even when provided identical rubrics, RubiCap consistently achieves higher win rates.

100 200 300 400 500 600

Caption Limit (Words)

0.550

0.575

0.600

0.625

0.650

0.675

0.700

0.725

0.750

AverageScore

RubiCap-7B-PixMoCap

Qwen2.5-VL-32B-Instruct

RubiCap-3B-PixMoCap RubiCap-2B-PixMoCap

Qwen2.5-VL-7B-Instruct Qwen2.5-VL-3B-Instruct

100 200 300 400 500 600

Caption Limit (Words)

0.550

0.575

0.600

0.625

0.650

0.675

0.700

0.725

0.750

AverageScore

RubiCap-7B-DenseFusion Qwen2.5-VL-32B-Instruct RubiCap-3B-DenseFusion RubiCap-2B-DenseFusion

Qwen2.5-VL-7B-Instruct Qwen2.5-VL-3B-Instruct

- Figure 6 RubiCap achieves word efficiency that surpasses larger models. The left panel shows results on PixMoCap; the right panel reports results on DenseFusion. RubiCap models consistently outperform both their same-size base models and larger counterparts under strict word-count constraints. This shows rubric-guided RL incurs more relevant and comprehensive captions.

availability, allowing us to attribute remaining gap to the RL training paradigm itself. Experiments are conducted at the 3B and 7B scales using PixMoCap.

Results. Fig. 5 (right) compares rubric-augmented SFT and RubiCap across both scales. First, RubiCap outperforms rubric-augmented SFT when evaluated against the base model. At 3B, RubiCap achieves a win rate of 68.6%, compared to 64.0% for rubric-augmented SFT. This gap widens at 7B, where RubiCap reaches 70.8%, a 5.0 percentage point advantage. Second, when compared to human-expert refined captions, this advantage persists: at 7B, RubiCap outperforms rubric-augmented SFT by 6.2 percentage points. These results indicate that conditioning on consensus-driven rubrics during supervised learning is insufficient to fully leverage their corrective potential. RubiCap’s advantage stems not from rubric exposure alone, but from using rubrics as fine-grained reward signals within RL, enabling broader

Table 1 VLM pretraining performance across 9 benchmarks when using captions generated by different annotators. Bold indicates best performance per benchmark; underline indicates the second best. RubiCap-annotated models outperform the GPT-4V baseline (a proprietary system) on average, using open models at 3B–7B scale.

Proprietary RubiCap Annotators (Ours) Benchmark GPT-4V RubiCap-3BPixMoCap RubiCap-3BDenseFusion RubiCap-7BPixMoCap

AI2D 47.67 49.55 48.32 48.10 ChartQA 34.04 35.68 36.24 36.60 MathVista-CoT 33.20 35.60 37.00 33.00 MathVista-Format 32.70 34.80 34.60 31.80 MathVista-Solution 34.90 35.00 35.60 34.90 MMBench (EN) 63.57 65.72 64.43 67.61 MMMU 36.33 35.22 35.89 37.00 MM-Vet 16.93 21.38 22.57 25.05 ScienceQA 72.38 73.97 72.68 74.60

Average 41.75 42.99 43.04 43.18

exploration and more effective optimization.

### 4.5 Caption Quality under Word Constraints

Setup. We further evaluate caption quality under explicit word-count constraints using CaptionQA (Yang et al., 2025), a recent multiple-choice benchmark that assesses whether captions contain sufficient information to answer curated MCQs. Higher CaptionQA scores indicate captions encode more relevant and comprehensive information. We use Qwen2.5-72B-Instruct as the LLM evaluator and prompt the fine-tuned models to generate captions under word limits ranging from 100 to 600 words. Experiments are conducted across three scales: RubiCap-2B, RubiCap-3B, and RubiCap-7B, each compared against its corresponding untuned checkpoint.

Results. As shown in Fig. 6, RubiCap consistently outperforms its base model counterparts across all word-count limits and model scales. The gains are most pronounced in low-token regimes: under a 100-word limit, RubiCap-7B improves CaptionQA by +12.01% over Qwen2.5-VL-7B-Instruct, and RubiCap-

- 3B improves by +9.53% over its 3B counterpart. Notably, performance gains are not strictly tied to model size: RubiCap-3B and RubiCap-2B both surpass the larger 7B base model under the same constraints. This cross-size advantage extends further: under 100–300 word limits, RubiCap-7B outperforms the 32B model, and matches it under more relaxed 400–600 word budgets—delivering 32B-level caption quality at 7B inference cost. These results demonstrate: by covering multiple evaluation dimensions simultaneously, rubric-driven rewards train models to prioritize salient, information-dense content, producing higher-quality captions within tighter token budgets.

### 4.6 Enabling Stronger VLM Pretraining

Setup. Finally, we demonstrate the broader applicability of RubiCap as a scalable captioner for visionlanguage model pretraining. We follow the LLaVA-NeXT training framework (Liu et al., 2024) but replace the Stage 1.5 image captioning data with captions generated by RubiCap-3B and RubiCap-7B. Specifically, we re-caption three large-scale datasets—COCO118K, BLIP558K, and CC3M—for a total of approximately 3.5 million images, while keeping all remaining instruction tuning data unchanged. The resulting VLM uses CLIP-336px (Radford et al., 2021) as the vision encoder and Qwen2-7B (Bai et al., 2025) as the LLM backbone. The performance is evaluated on nine benchmarks: AI2D (Kembhavi et al., 2016), ChartQA (Masry et al., 2022), MathVista (3 variants) (Lu et al., 2023), MMBench (EN) (Liu et al., 2024), MMMU (Yue et al., 2024), MM-Vet (Yu et al., 2023), ScienceQA (Lu et al., 2022). As a direct baseline, we train an identical model under the same pipeline but with GPT-4V-generated captions.

Results. Table 1 reports performance across 9 VLM benchmarks. All three RubiCap-annotated models outperform the GPT-4V baseline on average, yielding a relative improvement of 3.42%. Remarkably, even our compact 3B-scale RubiCap models produce pretraining captions that yield stronger pretrained VLMs than those annotated by GPT-4V, matching or exceeding proprietary model quality at a

fraction of the cost. These results establish RubiCap-trained models as a practical, scalable alternative to proprietary systems for vision-language pretraining.

## 5 Conclusion

We present RubiCap, a framework that addresses the verification bottleneck in RL-based dense captioning by deriving synthetic, sample-specific rubrics as fine-grained reward signals. Extensive evaluations confirm RubiCap’s effectiveness across multiple axes: it achieves the highest win rates in GPT-4.1-judged pairwise comparisons against all baselines, surpasses human-expert annotations and GPT-4V-augmented outputs at the 7B scale, and preserves broad pretrained capabilities far more effectively than supervised distillation. In a blind ranking evaluation, RubiCap-7B obtains the highest proportion of rank-1 assignments, outperforming 72B and 32B frontiers. Under fixed word budgets, RubiCap produces more information-dense captions, enabling a 7B model to match 32B-scale performance—delivering frontier-level caption quality at a fraction of the inference cost. Finally, a compact RubiCap-3B serves as an effective pretraining annotators, yielding stronger VLMs than those built on proprietary GPT-4V captions.

## References

Chen, X.; Fang, H.; Lin, T.-Y.; Vedantam, R.; Gupta, S.; Dollár, P.; Zitnick, C. L. Microsoft coco captions: Data collection and evaluation server. arXiv preprint arXiv:1504.00325 2015,

Vinyals, O.; Toshev, A.; Bengio, S.; Erhan, D. Show and tell: A neural image caption generator. Proceedings of the IEEE conference on computer vision and pattern recognition. 2015; pp 3156–3164.

Sharma, P.; Ding, N.; Goodman, S.; Soricut, R. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2018; pp 2556–2565.

Chen, S.; Jin, Q.; Wang, P.; Wu, Q. Say as you wish: Fine-grained control of image caption generation with abstract scene graphs. Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020; pp 9962–9971.

Park, S.; Paik, J. RefCap: image captioning with referent objects attributes. Scientific Reports 2023, 13, 21577. Zhou, Q.; Huang, J.; Li, Q.; Gao, J.; Wang, Q. Text-only synthesis for image captioning. arXiv preprint

arXiv:2405.18258 2024,

Deitke, M.; Clark, C.; Lee, S.; Tripathi, R.; Yang, Y.; Park, J. S.; Salehi, M.; Muennighoff, N.; Lo, K.; Soldaini, L.; others Molmo and pixmo: Open weights and open data for state-of-the-art vision-language models. Proceedings of the Computer Vision and Pattern Recognition Conference. 2025; pp 91–104.

Radford, A.; Kim, J. W.; Hallacy, C.; Ramesh, A.; Goh, G.; Agarwal, S.; Sastry, G.; Askell, A.; Mishkin, P.; Clark, J.; others Learning transferable visual models from natural language supervision. International conference on machine learning. 2021; pp 8748–8763.

Zhang, B.; Zhang, P.; Dong, X.; Zang, Y.; Wang, J. Long-clip: Unlocking the long-text capability of clip. European conference on computer vision. 2024; pp 310–325.

Liu, H.; Li, C.; Wu, Q.; Lee, Y. J. Visual instruction tuning. Advances in neural information processing systems 2023, 36, 34892–34916.

Kim, Y.; Lee, J.; Kim, J.-H.; Ha, J.-W.; Zhu, J.-Y. Dense text-to-image generation with attention modulation. Proceedings of the IEEE/CVF International Conference on Computer Vision. 2023; pp 7701–7711.

Yang, K.; Deng, J.; An, X.; Li, J.; Feng, Z.; Guo, J.; Yang, J.; Liu, T. Alip: Adaptive language-image pre-training with synthetic caption. Proceedings of the IEEE/CVF International Conference on Computer Vision. 2023; pp 2922–2931.

Chen, L.; Li, J.; Dong, X.; Zhang, P.; He, C.; Wang, J.; Zhao, F.; Lin, D. Sharegpt4v: Improving large multi-modal models with better captions. European Conference on Computer Vision. 2024; pp 370–387.

Li, X.; Zhang, F.; Diao, H.; Wang, Y.; Wang, X.; Duan, L. Densefusion-1m: Merging vision experts for comprehensive multimodal perception. Advances in Neural Information Processing Systems 2024, 37, 18535–18556.

Singla, V.; Yue, K.; Paul, S.; Shirkavand, R.; Jayawardhana, M.; Ganjdanesh, A.; Huang, H.; Bhatele, A.; Somepalli, G.; Goldstein, T. From pixels to prose: A large dataset of dense image captions. arXiv preprint arXiv:2406.10328 2024,

Chu, T.; Zhai, Y.; Yang, J.; Tong, S.; Xie, S.; Schuurmans, D.; Le, Q. V.; Levine, S.; Ma, Y. Sft memorizes, rl

generalizes: A comparative study of foundation model post-training. arXiv preprint arXiv:2501.17161 2025, McCloskey, M.; Cohen, N. J. Psychology of learning and motivation; Elsevier, 1989; Vol. 24; pp 109–165. Lai, S.; Zhao, H.; Feng, R.; Ma, C.; Liu, W.; Zhao, H.; Lin, X.; Yi, D.; Xie, M.; Zhang, Q.; others Reinforcement

fine-tuning naturally mitigates forgetting in continual post-training. arXiv preprint arXiv:2507.05386 2025, Shenfeld, I.; Pari, J.; Agrawal, P. Rl’s razor: Why online reinforcement learning forgets less. arXiv preprint arXiv:2509.04259 2025,

Gerstgrasser, M.; Schaeffer, R.; Dey, A.; Rafailov, R.; Sleight, H.; Hughes, J.; Korbak, T.; Agrawal, R.; Pai, D.; Gromov, A.; others Is model collapse inevitable? breaking the curse of recursion by accumulating real and synthetic data. arXiv preprint arXiv:2404.01413 2024,

Team, K.; Du, A.; Gao, B.; Xing, B.; Jiang, C.; Chen, C.; Li, C.; Xiao, C.; Du, C.; Liao, C.; others Kimi k1. 5: Scaling reinforcement learning with llms. arXiv preprint arXiv:2501.12599 2025,

Guo, D.; Yang, D.; Zhang, H.; Song, J.; Wang, P.; Zhu, Q.; Xu, R.; Zhang, R.; Ma, S.; Bi, X.; others Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948 2025,

Shao, Z.; Wang, P.; Zhu, Q.; Xu, R.; Song, J.; Bi, X.; Zhang, H.; Zhang, M.; Li, Y.; Wu, Y.; others Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300 2024,

Luo, M.; Tan, S.; Huang, R.; Patel, A.; Ariyak, A.; Wu, Q.; Shi, X.; Xin, R.; Cai, C.; Weber, M.; others Deepcoder:

A fully open-source 14b coder at o3-mini level. Notion Blog 2025, 1. Liu, J.; Zhang, L. Code-R1: Reproducing R1 for Code with Reliable Rewards. 2025, Vedantam, R.; Lawrence Zitnick, C.; Parikh, D. Cider: Consensus-based image description evaluation. Proceedings of

the IEEE conference on computer vision and pattern recognition. 2015; pp 4566–4575. Lin, C.-Y. ROUGE: A Package for Automatic Evaluation of Summaries. Text Summarization Branches Out. Barcelona, Spain, 2004; pp 74–81.

Lee, Y.; Park, I.; Kang, M. Fleur: An explainable reference-free evaluation metric for image captioning using a large multimodal model. Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024; pp 3732–3746.

Chen, D.; Chen, R.; Zhang, S.; Wang, Y.; Liu, Y.; Zhou, H.; Zhang, Q.; Wan, Y.; Zhou, P.; Sun, L. Mllm-as-ajudge: Assessing multimodal llm-as-a-judge with vision-language benchmark. Forty-first International Conference on Machine Learning. 2024.

Lee, S.; Kim, S.; Park, S.; Kim, G.; Seo, M. Prometheus-vision: Vision-language model as a judge for fine-grained evaluation. Findings of the Association for Computational Linguistics: ACL 2024. 2024; pp 11286–11315.

Cheng, K.; Song, W.; Fan, J.; Ma, Z.; Sun, Q.; Xu, F.; Yan, C.; Chen, N.; Zhang, J.; Chen, J. CapArena: Benchmarking

and Analyzing Detailed Image Captioning in the LLM Era. 2025; https://arxiv.org/abs/2503.12329. OpenAI GPT-4.1 Model Documentation. 2024; https://platform.openai.com/docs/models/gpt-4.1. Yang, S.; Liu, Y.; Zhai, B.; Sun, X.; Liu, Z.; Barsoum, E.; Li, M.; Xu, C. CaptionQA: Is Your Caption as Useful as

the Image Itself? arXiv preprint arXiv:2511.21025 2025,

Plummer, B. A.; Wang, L.; Cervantes, C. M.; Caicedo, J. C.; Hockenmaier, J.; Lazebnik, S. Flickr30k entities: Collecting region-to-phrase correspondences for richer image-to-sentence models. Proceedings of the IEEE international conference on computer vision. 2015; pp 2641–2649.

Fan, L.; Krishnan, D.; Isola, P.; Katabi, D.; Tian, Y. Improving clip training with language rewrites. Advances in Neural Information Processing Systems 2023, 36, 35544–35575.

Lai, Z.; Zhang, H.; Zhang, B.; Wu, W.; Bai, H.; Timofeev, A.; Du, X.; Gan, Z.; Shan, J.; Chuah, C.-N.; others Veclip: Improving clip training via visual-enriched captions. European Conference on Computer Vision. 2024; pp 111–127.

Yu, Q.; Sun, Q.; Zhang, X.; Cui, Y.; Zhang, F.; Cao, Y.; Wang, X.; Liu, J. Capsfusion: Rethinking image-text data at scale. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024; pp 14022–14032.

Li, X.; Zhang, T.; Li, Y.; Yuan, H.; Chen, S.; Zhou, Y.; Meng, J.; Sun, Y.; Xu, S.; Qi, L.; others Denseworld-1m: Towards detailed dense grounded caption in the real world. arXiv preprint arXiv:2506.24102 2025,

Zhu, Z.; Zhou, F.; Huang, S.; Li, T. EKCA-Cap: Improving dense captioning via external knowledge and context awareness. Neurocomputing 2025, 635, 129867.

Shen, H.; Liu, P.; Li, J.; Fang, C.; Ma, Y.; Liao, J.; Shen, Q.; Zhang, Z.; Zhao, K.; Zhang, Q.; others Vlm-r1: A stable and generalizable r1-style large vision-language model. arXiv preprint arXiv:2504.07615 2025,

Liu, Z.; Sun, Z.; Zang, Y.; Dong, X.; Cao, Y.; Duan, H.; Lin, D.; Wang, J. Visual-rft: Visual reinforcement fine-tuning. Proceedings of the IEEE/CVF International Conference on Computer Vision. 2025; pp 2034–2044.

Sinha, S.; Frunza, O.; Rasul, K.; Nevmyvaka, Y.; Zhang, A. Chart-RVR: Reinforcement Learning with Verifiable Rewards for Explainable Chart Reasoning. arXiv preprint arXiv:2510.10973 2025,

Zhou, Z.; Chen, D.; Ma, Z.; Hu, Z.; Fu, M.; Wang, S.; Wan, Y.; Zhao, Z.; Krishna, R. Reinforced visual perception with tools. arXiv preprint arXiv:2509.01656 2025,

Xing, L.; Dong, X.; Zang, Y.; Cao, Y.; Liang, J.; Huang, Q.; Wang, J.; Wu, F.; Lin, D. Caprl: Stimulating dense image caption capabilities via reinforcement learning. arXiv preprint arXiv:2509.22647 2025,

Shao, R.; Asai, A.; Shen, S. Z.; Ivison, H.; Kishore, V.; Zhuo, J.; Zhao, X.; Park, M.; Finlayson, S. G.; Sontag, D.; others Dr tulu: Reinforcement learning with evolving rubrics for deep research. arXiv preprint arXiv:2511.19399 2025,

Gunjal, A.; Wang, A.; Lau, E.; Nath, V.; He, Y.; Liu, B.; Hendryx, S. Rubrics as rewards: Reinforcement learning beyond verifiable domains. arXiv preprint arXiv:2507.17746 2025,

Comanici, G.; Bieber, E.; Schaekermann, M.; Pasupat, I.; Sachdeva, N.; Dhillon, I.; Blistein, M.; Ram, O.; Zhang, D.; Rosen, E.; others Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261 2025,

Singh, A.; Fry, A.; Perelman, A.; Tart, A.; Ganesh, A.; El-Kishky, A.; McLaughlin, A.; Low, A.; Ostrow, A.; Ananthram, A.; others Openai gpt-5 system card. arXiv preprint arXiv:2601.03267 2025,

Bai, S.; Chen, K.; Liu, X.; Wang, J.; Ge, W.; Song, S.; Dang, K.; Wang, P.; Wang, S.; Tang, J.; others Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923 2025,

Kamath, A.; Ferret, J.; Pathak, S.; Vieillard, N.; Merhej, R.; Perrin, S.; Matejovicova, T.; Ramé, A.; Rivière, M.; Rouillard, L.; others Gemma 3 technical report. arXiv preprint arXiv:2503.19786 2025, 4.

Bai, S.; Cai, Y.; Chen, R.; Chen, K.; Chen, X.; Cheng, Z.; Deng, L.; Ding, W.; Gao, C.; Ge, C.; others Qwen3-vl

technical report. arXiv preprint arXiv:2511.21631 2025, OpenAI GPT-4V(ision) System Card; Technical Report, 2023. Wang, P.; Bai, S.; Tan, S.; Wang, S.; Fan, Z.; Bai, J.; Chen, K.; Liu, X.; Wang, J.; Ge, W.; others Qwen2-vl: Enhancing

vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191 2024, Dong, H.; Li, J.; Wu, B.; Wang, J.; Zhang, Y.; Guo, H. Benchmarking and improving detail image caption. arXiv preprint arXiv:2405.19092 2024,

Chen, X.; Salazar, I.; Kementchedjhieva, Y. SPECS: Specificity-Enhanced CLIP-Score for Long Image Caption Evaluation. Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing. 2025; pp 9406–9418.

Banerjee, S.; Lavie, A. METEOR: An automatic metric for MT evaluation with improved correlation with human judgments. Proceedings of the acl workshop on intrinsic and extrinsic evaluation measures for machine translation and/or summarization. 2005; pp 65–72.

Papineni, K.; Roukos, S.; Ward, T.; Zhu, W.-J. Bleu: a method for automatic evaluation of machine translation. Proceedings of the 40th annual meeting of the Association for Computational Linguistics. 2002; pp 311–318.

###### DeepMind Specification Gaming: The Flip Side of AI Ingenuity. 2023; deepmind.google/blog/

specification-gaming-the-flip-side-of-ai-ingenuity/.

Amodei, D.; Olah, C.; Steinhardt, J.; Christiano, P.; Schulman, J.; Mané, D. Concrete problems in AI safety. arXiv preprint arXiv:1606.06565 2016,

Kim, H.; Kim, S.; Jeong, J.; Cho, Y.; Cho, S. EXPERT: An explainable image captioning evaluation metric with structured explanations. Findings of the Association for Computational Linguistics: ACL 2025. 2025; pp 26642– 26657.

Duan, H.; Yang, J.; Qiao, Y.; Fang, X.; Chen, L.; Liu, Y.; Dong, X.; Zang, Y.; Zhang, P.; Wang, J.; others Vlmevalkit: An open-source toolkit for evaluating large multi-modality models. Proceedings of the 32nd ACM international conference on multimedia. 2024; pp 11198–11201.

Hudson, D. A.; Manning, C. D. Gqa: A new dataset for real-world visual reasoning and compositional question answering. Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2019; pp 6700– 6709.

Fu, X.; Hu, Y.; Li, B.; Feng, Y.; Wang, H.; Lin, X.; Roth, D.; Smith, N. A.; Ma, W.-C.; Krishna, R. BLINK: Multimodal Large Language Models Can See but Not Perceive. arXiv preprint arXiv:2404.12390 2024,

Kembhavi, A.; Salvato, M.; Kolve, E.; Seo, M.; Hajishirzi, H.; Farhadi, A. A diagram is worth a dozen images.

European conference on computer vision. 2016; pp 235–251. xAI RealWorldQA. 2024; https://huggingface.co/datasets/xai-org/RealworldQA. Liu, Y.; Li, Z.; Huang, M.; Yang, B.; Yu, W.; Li, C.; Yin, X.-C.; Liu, C.-L.; Jin, L.; Bai, X. Ocrbench: on the hidden

mystery of ocr in large multimodal models. Science China Information Sciences 2024, 67, 220102.

Singh, A.; Natarajan, V.; Shah, M.; Jiang, Y.; Chen, X.; Batra, D.; Parikh, D.; Rohrbach, M. Towards vqa models that can read. Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2019; pp 8317–8326.

Mishra, A.; Shekhar, S.; Singh, A. K.; Chakraborty, A. OCR-VQA: Visual Question Answering by Reading Text in Images. ICDAR. 2019.

Mathew, M.; Bagal, V.; Tito, R.; Karatzas, D.; Valveny, E.; Jawahar, C. Infographicvqa. Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. 2022; pp 1697–1706.

Mathew, M.; Karatzas, D.; Manmatha, R.; Jawahar, C. DocVQA: A Dataset for VQA on Document Images. CoRR abs/2007.00398 (2020). arXiv preprint arXiv:2007.00398 2020,

Masry, A.; Long, D.; Tan, J. Q.; Joty, S.; Hoque, E. ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning. Findings of the Association for Computational Linguistics: ACL 2022. Dublin, Ireland, 2022; pp 2263–2279.

Liu, H.; Li, C.; Li, Y.; Li, B.; Zhang, Y.; Shen, S.; Lee, Y. J. LLaVA-NeXT: Improved reasoning, OCR, and world knowledge. 2024; https://llava-vl.github.io/blog/2024-01-30-llava-next/.

Lu, P.; Bansal, H.; Xia, T.; Liu, J.; Li, C.; Hajishirzi, H.; Cheng, H.; Chang, K.-W.; Galley, M.; Gao, J. Mathvista:

Evaluating mathematical reasoning of foundation models in visual contexts. arXiv preprint arXiv:2310.02255 2023, Liu, Y.; Duan, H.; Zhang, Y.; Li, B.; Zhang, S.; Zhao, W.; Yuan, Y.; Wang, J.; He, C.; Liu, Z.; others Mmbench: Is

your multi-modal model an all-around player? European conference on computer vision. 2024; pp 216–233.

Yue, X.; Ni, Y.; Zhang, K.; Zheng, T.; Liu, R.; Zhang, G.; Stevens, S.; Jiang, D.; Ren, W.; Sun, Y.; others Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2024; pp 9556–9567.

Yu, W.; Yang, Z.; Li, L.; Wang, J.; Lin, K.; Liu, Z.; Wang, X.; Wang, L. Mm-vet: Evaluating large multimodal models for integrated capabilities. arXiv preprint arXiv:2308.02490 2023,

Lu, P.; Mishra, S.; Xia, T.; Qiu, L.; Chang, K.-W.; Zhu, S.-C.; Tafjord, O.; Clark, P.; Kalyan, A. Learn to explain: Multimodal reasoning via thought chains for science question answering. Advances in neural information processing systems 2022, 35, 2507–2521.

## A Appendix

AppendixRoadmap. Our appendix is structured as follows. We begin with the prompts underlying our framework: Appendix B presents those used to synthesize rubrics and apply them via an LLM judge for RL rewards, Appendix C covers the prompts used by baseline methods with VLM-as-a-judge, and Appendix D details the prompts for assessing caption quality in the CapArena benchmark and the blind ranking experiment. Appendix E then summarizes our full framework in pseudocode, and Appendix F documents the experimental setup and training configurations. We next turn to results: Appendix G reports full evaluation tables across 7B, 3B, and 2B models on comprehensive captioning metrics, while Appendix H presents VLM performance for measuring knowledge retention. Finally, Appendix I offers qualitative comparisons, showcasing sampled captions from our model alongside those from the base model and Qwen2.5-VL-72B-Instruct.

## B Rubric Synthesis and Judging Prompts

We first provide the prompts used to synthesize targeted rubrics from a committee of candidate captions: the system prompt in Prompt 1 and the user prompt in Prompt 2. Both prompts are sent to the rubric writer, which is instructed to return structured JSON outputs. Each rubric item contains the criterion, evaluation rule, severity weight, and a supporting justification traced back to teacher consensus, providing an interpretable audit trail for every scoring decision.

Our framework is agnostic to the specific models in the teacher committee; any set of diverse captioners can serve this role. In our experiments, we instantiate the committee with five VLMs spanning different model families to maximize descriptive diversity. Once rubrics are synthesized, the entire RL training loop requires only a lightweight LLM judge (Qwen2.5-7B-Instruct, a 7B open-source model), with no further dependence on proprietary systems. Notably, the committee is invoked only once per image during rubric construction—it is an offline preprocessing step (a one-time investment), not a recurring labeling cost.

Prompt 1 System Prompt for Rubric Writer

You are an expert evaluator generating highly discriminative rubrics to assess image caption quality. ## Task Identify the most discriminative criteria that separate excellent captions from weak or flawed ones. Focus

→ on subtle but decisive quality differences that generic rubrics typically miss, while covering all

→ critical dimensions of captioning performance. ## Rules you MUST follow:

- 1. Discriminative Power (Highest Priority)

- - **Only** include criteria where the student model actually **fails** relative to teacher majority. Do NOT

→ create rubrics for aspects the student model already handles correctly.

- - Each rubric MUST meaningfully distinguish the weak student caption from teacher-consensus level

→ performance.

- 2. Teacher Consensus as Ground Truth

- - Ground truth = majority agreement among the five teacher models.

- - A visual element, relationship, or interpretation is considered correct only if >= 2 teachers describe it

→ accurately.

- 3. Weighting by Severity

- - 3.0: Critical failures (main subject misidentification, hallucination of major elements, missing

→ essential relationships)

- - 2.0: Important but non-critical (secondary objects, spatial/contextual accuracy, attribute precision)

- - 1.0: Minor polish (style fluency, phrasing clarity, minor detail richness)

- 4. Binary & Verifiable

- Every criterion must have a clear, objective pass/fail rule that can be verified by directly comparing

→ the student caption against teacher consensus.

- 5. Quality over Quantity

- Prefer extremely important and sharp rubrics over many many generic ones.

## Output Requirements: IMPORTANT: Return valid JSON object only, enclosed in triple backticks (‘‘‘json). Do not include any

→ additional text, explanations, or comments outside the JSON. Escape all quotes within string values

→ using backslash (\"). Do not use single quotes or unescaped double quotes within JSON string values:

**JSON Structure:** {

"rubrics": [ {

"criterion": "Clear, specific criterion (e.g., Identifies the red bicycle in foreground)", "description": "Detailed explanation of what this measures and why it matters", "evaluation_rule": "Concrete rule with clear pass/fail condition", "weight": 1.0|2.0|3.0, "justification": "Explain the gap: what you see in the image, what teachers captured, what weak model

→ missed/got wrong", "student_already_met": "True or False - Whether the weak model already satisfies this criterion", "reference_teachers": ["List of teacher model names that correctly satisfied this criterion"], "teacher_consensus": "Description of what the majority of reliable teachers agree on for this element"

} ]

}

Prompt 2 User Prompt for Rubric Writer

|{{IMAGE}}<br><br>**Weak Model Output:** {{WEAK_STUDENT_CAPTION}}<br><br>**Teacher Model Outputs:**<br><br>1. **Model 1:** {{CANDIDATE_1}}<br><br>2. **Model 2:** {{CANDIDATE_2}}<br><br>3. **Model 3:** {{CANDIDATE_3}}<br><br>4. **Model 4:** {{CANDIDATE_4}}<br><br>5. **Model 5:** {{CANDIDATE_5}}<br><br><br>**Task:**<br><br>1. Carefully examine the image and identify all important visual elements.<br><br>2. Determine teacher consensus (what the majority of the five teacher models describe correctly).<br><br>3. Evaluate the weak model’s caption across all important dimensions of caption quality: accuracy,<br><br>→ completeness, clarity, detail, relationships, and contextual interpretation.<br><br>4. For each dimension you choose to include, create one targeted binary rubric item with appropriate weight<br><br>→ (1.0-3.0) using the required JSON structure.<br><br>5. Do NOT create rubrics for aspects the weak model already handles correctly.<br>|
|---|

Next, we present the prompt for the LLM judge that provides RL rewards. Prompt 3 instructs the judge to evaluate generated caption rollouts against a given rubric criterion and its evaluation rule, assigning a binary satisfaction score. The returned JSON includes a reasoning trace alongside the score.

Prompt 3 User Prompt for RubiCap’s Reward Provider

You are a Quality Assurance Auditor for image captioning. Your task is to validate if a generated caption

→ adheres to a specific criterion based on meaning and intent, not just keyword matching. CRITERION TO EVALUATE (THE ONLY THING YOU USE TO CONSIDER):

- - Criteria name: {criterion}

- - Criteria description: {description}

- - Evaluation rule: {evaluation_rule}

GENERATED CAPTION TO EVALUATE: {generated_caption}

EVALUATION INSTRUCTIONS:

- 1. **Analyze Requirements:** Break down the criteria description and its evaluation rule to understand the

→ core facts or concepts required.

- 2. **Check Semantic Equivalence:** Analyze the generated caption. Check if the required concepts are

→ present, even if phrased differently.

- - Synonyms are acceptable (e.g., "automobile" instead of "car").

- - Structural changes are acceptable (e.g., "The man holds a cup" vs "A cup is held by the man").

- 3. **Assign Score:**

- - Score 1 (Pass): The caption conveys the correct *meaning* of the rule. It includes the necessary

→ details, regardless of phrasing.

- - Score 0 (Fail): The caption explicitly misses key information, contradicts the facts, or hallucinates

→ details not found in the rule.

RESPONSE FORMAT: ‘‘‘json {{

"reasoning": "<detailed explanation of the score, why the caption does or does not fully meet the given

→ criterion and its evaluation rule>", "score": <0 or 1>

}} ‘‘‘

IMPORTANT: Return valid JSON object only, enclosed in triple backticks (‘‘‘json). Do not include any

→ additional text, explanations, or comments outside the JSON. Escape all quotes within string values

→ using backslash (\"). Do not use single quotes or unescaped double quotes within JSON string values.

## C Baseline Judging Prompts

We provide the prompts for two baseline reward methods: Direct-Likert and Reference-Likert. Both employ a VLM-as-a-judge approach, using Qwen2.5-VL-7B-Instruct as the judge. Prompts 4 and 5 instruct the judge to rate caption quality on a scale from 0 to 10. In Prompt 5, a reference caption is additionally provided to guide the evaluation.

- Prompt 4 User Prompt for VLM Judge in Direct-Likert Method

|You are an expert image caption evaluator. Generated Caption: {generated_caption} Evaluate the generated caption on these criteria:<br><br>1. Accuracy: Does it correctly describe what’s in the image?<br><br>2. Completeness: Does it cover the important details?<br><br>3. Quality: Is it well-written and coherent? Provide a score from 0 to 10, where:<br><br><br>- 0-3: Poor quality, major inaccuracies or missing key information<br><br>- 4-6: Adequate, captures main elements but lacks detail or has minor issues<br><br>- 7-8: Good, accurate and detailed description<br><br>- 9-10: Excellent, comprehensive and high-quality caption Respond with ONLY a single number from 0 to 10, nothing else.<br><br><br>|
|---|

- Prompt 5 User Prompt for VLM Judge in Reference-Likert Method

You are an expert image caption evaluator. Compare the generated caption with the reference caption for the

→ given image.

Reference Caption: {reference_caption} Generated Caption: {generated_caption}

Evaluate the generated caption on these criteria:

- 1. Accuracy: Does it correctly describe what’s in the image?

- 2. Completeness: Does it cover the important details?

- 3. Quality: Is it well-written and coherent? Provide a score from 0 to 10, where:

- - 0-3: Poor quality, major inaccuracies or missing key information

- - 4-6: Adequate, captures main elements but lacks detail or has minor issues

- - 7-8: Good, accurate and detailed description

- - 9-10: Excellent, comprehensive and high-quality caption

Respond with ONLY a single number from 0 to 10, nothing else.

## D Evaluation Prompts

We provide the evaluation prompts used throughout our experiments. Prompt 6, adopted from CapArena (Cheng

et al., 2025), is used to compute pairwise win-lose rates as judged by GPT-4.1. Caption assignments are randomized to mitigate positional bias, and the judge returns a detailed justification along with the winner decision. Prompt 7 instructs the evaluator to blindly rank captions and assess quality along four dimensions: accuracy, completeness, clarity, and hallucination penalty. We evaluate five captions: those generated by RubiCap, Qwen2.5-VL-72B-Instruct, Qwen2.5-VL-32B-Instruct, Qwen2.5-VL-7B-Instruct (the student base model), and an expert-annotated caption from PixMoCap. The evaluator returns per-dimension scores and an overall ranking.

Prompt 6 Used Prompt for Win–Lose Rate in CapArena

|Given an image and two candidate captions, you are required to determine which of the two captions is<br><br>→ better. Below are some guidelines for your reference: ### What to evaluate<br><br>1. **Precision**: The caption should accurately correspond to the content of the image, providing precise<br><br>→ information about it. Common examples of imprecision include errors in color, quantity, spatial<br><br>→ relationships, or the posture of people.<br><br>2. **Informativeness**: Salient information in the image should be reflected in the caption. Since it is<br><br>→ impossible to include every detail, you will need to subjectively judge which aspects of the image are<br><br>→ important. For instance, describing an otter as "a small animal" is precise, but it is less informative<br><br>→ than specifying "an otter".<br><br>3. **Hallucination**: Captions that include descriptions of objects or elements that are clearly absent<br><br>→ from the image should be significantly penalized.<br><br>4. **Attention to detail**: Annotators should pay close attention to the details in the image to<br><br>→ distinguish the quality of the descriptions.<br><br>5. **Assistive description**: Imagine a visually impaired person asking you to describe the image for them.<br><br>→ How would you convey the image to them?<br><br>6. **Reverse thinking**: What image does the caption lead us to imagine? Does the caption effectively lead<br><br>→ you to imagine the intended image?<br><br>7. **Ties are acceptable**: If you find it genuinely difficult to determine which caption is better (e.g.,<br><br><br>→ both captions are excellent), marking a tie is acceptable. While the above guidelines provide a<br><br>→ framework, they cannot cover all possible cases. Therefore, we encourage you to make **subjective<br><br>→ judgments** based on the specific circumstances and your own reasoning about which caption is better.<br><br>### What to ignore When comparing the captions, **do NOT consider differences in**:<br><br>- Writing style or phrasing<br><br>- **Caption length**<br><br>- Grammatical variations<br><br><br>Caption A: {caption_a}<br><br>Caption B: {caption_b} Format your response in JSON format.<br><br><br>RESPONSE FORMAT: ‘‘‘json {{<br><br>"reason": "<detailed explanation of the judgment>", "judgment": "<A, B, or, Tie>"<br><br>}} ‘‘‘<br><br>|
|---|

Prompt 7 User Prompt to rank captions and assess caption quality by breaking down into four dimentions: accuracy, completeness, clarity, and hallucination penalty.

|You are an expert image captioning evaluator. Given the image above and the 5 captions below, rigorously<br><br>→ assess each caption.<br><br>## Scoring Score each caption on four dimensions (integers 0-10):<br><br>1. accuracy - Are the described objects, actions, text, colors, and spatial relationships factually correct<br><br>→ for this image? Penalize for any wrong attribute, misidentified object, or incorrect action.<br><br>2. completeness - Does the caption cover all visually significant elements (main subjects, notable actions,<br><br>→ background context, on-screen text if present)? Penalize for missing key details.<br><br>3. clarity - Is the caption well-written, specific, grammatically correct, and unambiguous? Penalize for<br><br>→ vague language or redundancy.<br><br>4. hallucination_penalty - Does the caption assert things NOT visible in the image? 0 = zero hallucination;<br><br><br>→ 10 = pervasive fabrication. Be strict: even plausible but unverifiable claims count as mild<br><br>→ hallucination (2-4). This score is applied as a penalty. Compute: total_score = (accuracy + completeness + clarity) / 3.0 - hallucination_penalty x 1.5 ## Captions to Evaluate {captions_text} ## Output Format Respond ONLY with a single valid JSON object - no markdown fences, no extra text. {<br><br>"assessments": {<br><br>"Caption A": { "justification": "<2-3 sentences citing specific visual evidence from the image>", "accuracy": <int 0-10>, "completeness": <int 0-10>, "clarity": <int 0-10>, "hallucination_penalty": <int 0-10>, "total_score": <float><br><br>},<br><br>... "Caption E": { ... }<br><br>}, "ranking": ["Caption X", "Caption X", "Caption X", "Caption X", "Caption X"]<br><br>} The "ranking" list must contain all 5 caption labels ordered from best (index 0) to worst (index 4), sorted<br><br>→ strictly by total_score descending.|
|---|

## E Algorithm Overview

We summarize the two stages of RubiCap as unified pseudocode in Algorithm 1. The prompt used in Stage 1 (Sec. 3.2) is a general, extensible template that any user can steer the rubric’s focus toward specific areas of weakness by design.

## F Training Details

#### We detail the training configurations and computational resources used in our experiments. We sample 50,000 images from PixMoCap 2 and DenseFusion 3, holding out 500 for evaluation and using the remainder for training. We use the Qwen VLM family at different scales as our base student models and treat Qwen2.5-

- 2https://huggingface.co/datasets/allenai/pixmo-cap
- 3https://huggingface.co/datasets/BAAI/DenseFusion-1M

Algorithm 1 RubiCap Require: Image dataset D, student policy πθ

, teacher committee T = {Tk}Kk=1, LLM Rubric Writer, LLM Judge, number of rollouts N, clipping threshold ϵ, severity weights w ∈ {1,2,3}.

s

- 1: {Stage 1: Automated Rubric Synthesis (Sec. 3.2)}
- 2: for each image x ∈ D do
- 3: Generate teacher captions: Cteacher(x) = {cteacherk }Kk=1 where cteacherk ← Tk(x)
- 4: Generate student caption: cstudent ∼ πθ

s

(· | x)

- 5: Step 1: Extract consensus elements agreed upon by ≥ ⌈K/2⌉ teachers conditioned on x
- 6: Step 2: Diagnose discriminative student deficiencies by comparing cstudent against teacher consensus; categorize by severity (critical, important, minor)
- 7: Step 3: Formulate targeted rubrics: R(x,cstudent,Cteacher(x)) = {(rm,wm)}Mm=1, where each rm is a binary criterion with severity weight wm
- 8: end for
- 9: {Stage 2: Rubric-Guided Reinforcement Learning (Sec. 3.3)}
- 10: while not converged do
- 11: Sample a batch of images x ∼ D
- 12: Sample N rollouts: {cstudenti }Ni=1 ∼ πθ

s

(· | x)

- 13: for each rollout i ∈ {1,...,N} do
- 14: LLM Judge evaluates each criterion: yˆi,m ∈ {0,1} for all rm ∈ R(x)
- 15: Compute reward: Gi =

M m=1 wm·yˆi,m

M m=1 wm

- 16: end for
- 17: Estimate advantages: Ai = Gi−mean({Gj}

N j=1)

std({Gj}Nj=1)

- 18: Update θs by minimizing LGRPO(θs)
- 19: end while

7B-Instruct (LLM-only) as the reward provider to evaluate caption rollouts against the synthesized rubrics. All experiments are conducted on 8 NVIDIA H100 GPUs.

We optimize the student policy using GRPO (Shao et al., 2024) with the following hyperparameters: a learning rate of 1 × 10−5 with cosine scheduling and a warmup ratio of 0.01, a maximum completion length of 1,024 tokens, and N = 4 generations per image. Training runs for 1 epoch. All the RL baseline methods employ the same training configuration, except for the reward design. During rollout generation, we prompt the student with: “You are an expert at describing images in detail. Provide comprehensive, accurate captions that describe the main subjects, their actions, the setting, and important visual details in the image.”

For the supervised fine-tuning baseline, we use a learning rate of 1 × 10−5 and train for 1 epoch. We explore diverse supervision signals for SFT, including expert-refined annotations from PixMoCap, GPT-4Vaugmented captions from DenseFusion, and distilled captions from the Qwen2.5-VL-72B-Instruct frontier model.

## G Captioning Results

In addition to CaptionQA (Yang et al., 2025) and the blind ranking evaluation inspired by EXPERT (Kim et al., 2025) used in the main paper, we report full results across all captioning evaluation metrics considered. Each of them has been shown to align well with human preference. We describe each metric below, then present complete results in Table 2 (RubiCap-7B), Table 3 (RubiCap-3B), and Table 4 (RubiCap-2B). They are:

- • CapArena Win Rate (Cheng et al., 2025): A pairwise preference-based metric in which a strong VLM judge compares two captions and selects the better one. The win rate measures how often the model’s caption is preferred over a reference across pairwise comparisons. We use GPT-4.1 as the judge.
- • CAPTURE (Dong et al., 2024): A reference-based metric that uses a T5-based parser to extract structured visual elements (e.g., objects, attributes, and relations) from both the candidate and ground-truth

###### Table 2 Full captioning evaluation results for RubiCap-7B.

CapArena Win Rate (v.s. Base Model)

CapArena Win Rate (v.s. Expert-Annotated)

CAPTURE SPECS ROUGE-L METEOR BLEU-4 Average

Base Model (Qwen2.5-VL-7B-Instruct) 50.00 54.00 57.63 69.39 23.18 18.32 8.05 40.08 GRPO Fine-tuning on 50k Images Sampled from PixMoCap Dataset

NLP Metric-based RL 59.60 56.20 59.66 72.38 24.71 18.71 9.47 42.96 Direct-Likert 66.20 61.00 59.30 75.22 23.60 19.93 7.55 44.69 Reference-Likert 67.20 59.40 59.33 74.54 22.47 20.26 6.56 44.25 RubiCap-7B-PixMoCap 70.80 66.00 59.42 75.47 23.73 19.93 7.53 46.13 Performance Gain +20.80 +12.00 +1.79 +6.08 +0.55 +1.61 -0.52 +6.05

CapArena Win Rate (v.s. Base Model)

CapArena Win Rate (v.s. GPT-4V Augmented)

CAPTURE SPECS ROUGE-L METEOR BLEU-4 Average

Base Model (Qwen2.5-VL-7B-Instruct) 50.00 44.80 60.64 69.58 27.25 19.42 11.54 40.46 GRPO Fine-tuning on 50k Images Sampled from DenseFusion Dataset

NLP Metric-based RL 58.40 45.00 62.35 77.52 28.90 21.50 13.30 43.85 Direct-Likert 59.20 48.20 62.01 74.90 28.38 21.24 12.75 43.81 Reference-Likert 50.60 38.40 62.11 73.43 26.39 23.17 9.89 40.57 RubiCap-7B-DenseFusion 64.40 53.20 62.21 74.51 28.58 21.15 12.80 45.26 Performance Gain +14.40 +8.40 +1.57 +4.93 +1.33 +1.73 +1.26 +4.80

captions, filtering out stop words. Precision and recall are then computed by matching elements via exact, synonym, and soft (cosine similarity-based) methods. CAPTURE has shown higher consistency with expert judgments than other rule-based caption metrics.

- • SPECS (Chen et al., 2025): A reference-free representational similarity metric tailored to long image captioning. SPECS builds on LongCLIP (Zhang et al., 2024) with a training objective that decomposes captions into fine-grained detail units and applies contrastive, positive, and negative losses to reward accurate visual details while penalizing inaccuracies.
- • ROUGE-L (Lin, 2004): A text overlap metric that measures the Longest Common Subsequence (LCS) between the generated and reference captions.
- • METEOR (Banerjee and Lavie, 2005): A reference-based metric that goes beyond exact word matching by incorporating synonyms, stemming, and paraphrase matching.
- • BLEU-4 (Papineni et al., 2002): A precision-based metric that counts how many 4-gram sequences in the generated caption also appear in the reference, with a brevity penalty for short outputs.

Results. RubiCap consistently outperforms all RL baseline methods across both datasets and all model scales (see Tables 2, 3, and 4). The largest gains appear at 7B, with average improvements of +6.05 on PixMoCap and +4.80 on DenseFusion, and CapArena win rates reaching 70.80% against the base model (+20.80). At 3B, RubiCap maintains clear gains (+4.62 and +2.68 on average) while substantially outperforming CapRL-3B, which collapses to a SPECS score of 0.00 on both datasets. Notably, the Reference-Likert baseline fails at 3B and 2B (averaging 7.33 and 11.08), exhibiting severe self-praising behavior and reward hacking—a failure mode that RubiCap avoids entirely.

## H VLM Benchmark Results

We complement the main paper by reporting VLM benchmark results at the 3B and 2B scales, measuring how well fine-tuned models retain the pretrained capabilities of their base models. We evaluate all fine-tuned models on 9 VLM benchmarks using VLMEvalKit (Duan et al., 2024). Results are presented in Table 5 (3B) and Table 6 (2B).

###### Table 3 Full captioning evaluation results for RubiCap-3B.

CapArena Win Rate (v.s. Base Model)

CapArena Win Rate (v.s. Expert-Annotated)

CAPTURE SPECS ROUGE-L METEOR BLEU-4 Average

Base Model (Qwen2.5-VL-3B-Instruct) 50.00 34.00 56.87 68.83 21.76 17.98 6.50 36.56 CapRL-3B 59.40 34.40 59.07 0.00 18.46 20.75 4.08 28.02 GRPO Fine-tuning on 50k Images Sampled from PixMoCap Dataset

NLP Metric-based RL 54.00 35.80 58.12 71.22 24.31 17.83 9.14 38.63 Direct-Likert 64.00 37.40 58.35 69.73 23.11 18.36 7.59 39.79 Reference-Likert 7.80 4.00 13.33 16.80 7.27 2.13 0.01 7.33

RubiCap-3B-PixMoCap 68.60 42.80 57.89 69.49 23.30 18.16 8.05 41.18 Performance Gain +18.60 +8.80 +1.02 +0.66 +1.54 +0.18 +1.55 +4.62

CapArena Win Rate (v.s. Base Model)

CapArena Win Rate (v.s. GPT-4V Augmented)

CAPTURE SPECS ROUGE-L METEOR BLEU-4 Average

Base Model (Qwen2.5-VL-3B-Instruct) 50.00 20.00 58.68 68.03 25.73 18.84 11.06 36.05 CapRL-3B 45.40 15.20 60.65 0.00 21.73 23.49 6.51 24.71 GRPO Fine-tuning on 50k Images Sampled from DenseFusion Dataset

NLP Metric-based RL 50.80 19.40 60.91 77.14 28.68 20.54 12.74 38.60 Direct-Likert 53.20 23.80 60.28 69.44 27.27 18.95 11.35 37.76 Reference-Likert 48.60 17.80 60.25 65.66 25.47 21.62 9.59 35.57 RubiCap-3B-DenseFusion 59.20 26.20 59.67 69.33 27.20 18.52 10.97 38.73 Performance Gain +9.20 +6.20 +0.99 +1.30 +1.47 -0.32 -0.09 +2.68

Table 4 Full captioning evaluation results for RubiCap-2B.

CapArena Win Rate (v.s. Base Model)

CapArena Win Rate (v.s. Expert-Annotated)

CAPTURE SPECS ROUGE-L METEOR BLEU-4 Average

Base Model (Qwen2-VL-2B-Instruct) 50.00 28.80 57.33 67.34 22.14 17.24 6.40 35.61 GRPO Fine-tuning on 50k Images Sampled from PixMoCap Dataset

NLP Metric-based RL 52.20 27.00 57.63 69.02 23.97 16.56 7.89 36.32 Direct-Likert 57.20 31.40 57.07 64.57 22.87 16.35 7.51 36.71 Reference-Likert 9.00 2.60 25.72 22.43 12.03 4.72 1.06 11.08 RubiCap-2B-PixMoCap 61.60 34.60 57.19 64.40 23.08 16.17 7.75 37.83 Performance Gain +11.60 +5.80 -0.14 -2.94 +0.94 -1.07 +1.35 +2.22

CapArena Win Rate (v.s. Base Model)

CapArena Win Rate (v.s. GPT-4V Augmented)

CAPTURE SPECS ROUGE-L METEOR BLEU-4 Average

Base Model (Qwen2-VL-2B-Instruct) 50.00 19.40 59.80 67.49 26.96 19.03 11.60 36.33 GRPO Fine-tuning on 50k Images Sampled from DenseFusion Dataset

NLP Metric-based RL 40.80 13.20 59.84 72.50 28.06 19.10 11.45 34.99 Direct-Likert 49.40 14.20 58.41 63.88 26.34 17.26 9.41 34.13 Reference-Likert 42.20 13.00 58.34 64.20 25.91 18.05 10.39 33.16 RubiCap-2B-DenseFusion 52.40 22.00 58.14 63.54 26.59 16.91 9.47 35.58 Performance Gain +2.40 +2.60 -1.66 -3.95 -0.37 -2.12 -2.13 -0.75

## I Qualitative Examples

We present qualitative caption comparisons sampled from both datasets, comparing RubiCap-7B against its base model (Qwen2.5-VL-7B-Instruct) and the larger Qwen2.5-VL-72B-Instruct. A GPT-4.1 judge selects the preferred caption, with its reasoning shown in each caption block.

###### Table 5 VLM benchmark results for RubiCap-3B

SEEDBench A-OKVQA BLINK ChartQA ScienceQA DocVQA InfoVQA TextVQA OCRBench Average Base Model (Qwen2.5-VL-3B-Instruct) 73.45 85.05 47.32 82.83 80.46 92.60 75.42 78.49 83.00 77.62 GRPO Fine-tuning on 50k Images Sampled from PixMoCap Dataset

Metric-based RL 72.48 83.22 31.32 75.59 57.14 87.80 70.02 66.35 79.00 69.21 Direct-Likert 73.78 85.05 34.37 79.51 60.96 87.39 70.98 68.35 81.00 71.27 Reference-Likert 20.27 18.36 10.74 68.51 24.70 85.36 65.35 66.00 66.70 47.33 RubiCap-3B-PixMoCap 73.71 84.27 46.26 80.03 79.61 90.05 72.23 75.32 83.30 76.09 GRPO Fine-tuning on 50k Images Sampled from DenseFusion Dataset

Metric-based RL 68.48 79.55 30.00 73.91 47.72 87.78 69.96 66.12 80.20 67.08 Direct-Likert 73.76 83.92 38.37 78.39 69.10 88.79 71.86 68.83 79.40 72.49 Reference-Likert 39.90 52.54 7.79 70.23 24.01 85.79 68.89 63.17 79.00 54.59 RubiCap-3B-DenseFusion 74.01 84.62 45.37 80.15 80.56 90.36 71.99 75.73 81.50 76.03

Table 6 VLM benchmark results for RubiCap-2B

SEEDBench A-OKVQA BLINK ChartQA ScienceQA DocVQA InfoVQA TextVQA OCRBench Average Base Model (Qwen2-VL-2B-Instruct) 71.52 83.83 38.95 70.59 76.24 89.19 63.82 74.80 80.50 72.16 GRPO Fine-tuning on 50k Images Sampled from PixMoCap Dataset

Metric-based RL 41.03 56.56 16.32 55.38 54.71 80.73 53.39 68.70 51.19 53.11 Direct-Likert 70.81 79.46 29.63 67.47 70.69 84.22 57.03 71.73 71.72 66.97 Reference-Likert 12.23 18.97 3.90 51.50 35.62 79.92 51.65 60.65 29.90 38.26 RubiCap-2B-PixMoCap 71.48 81.91 36.95 70.03 76.09 87.28 61.15 73.68 78.14 70.75 GRPO Fine-tuning on 50k Images Sampled from DenseFusion Dataset

Metric-based RL 59.31 71.59 24.00 60.58 59.43 82.16 54.80 71.45 61.59 60.55 Direct-Likert 70.90 82.26 28.32 64.47 70.39 83.51 57.62 71.21 67.44 66.24 Reference-Likert 58.41 68.71 23.53 66.99 62.40 83.80 58.04 71.29 68.89 62.45 RubiCap-2B-DenseFusion 71.82 82.08 33.95 69.91 76.39 87.47 60.62 73.46 77.19 70.32

Caption Comparison (Judged by GPT4.1)

|RubiCap-7B-PixMoCap<br><br>Correct object identification (blowtorch, not oil lamp) This caption correctly identifies the object as an antique brass blowtorch, accurately describes the cylindrical body, brass finish, handle with holes, nozzle, and adjustment mechanism. It also notes the plain dark background and lighting. There are no hallucinations, and the description is clear and complete.<br><br>The image shows an antique brass blowtorch, likely used for welding or cutting metal. The device has a cylindrical body with a polished brass finish and a slightly worn appearance, indicating age. The handle is on the left side of the cylinder and features a series of holes along its length, possibly for ventilation or grip enhancement. The top of the blowtorch has a nozzle with a protruding brass tip, which is designed to direct the flame. A small lever or adjustment mechanism is located near the nozzle, suggesting that it can be adjusted for different flame settings. The overall design suggests functionality and durability, typical of tools from the early 20th century. The background is a plain dark surface, which contrasts with the golden hue of the brass, making the object stand out prominently. The lighting is even, highlighting the details of the tool's construction without creating harsh shadows. The image focuses solely on the blowtorch, with no additional objects or distractions present.|
|---|

|Qwen2.5-VL-72B-Instruct<br><br>Hallucination/error identified This caption misidentifies the object as an oil lamp and describes features like a black leather strap and additional leather strap, which are not visible. It also invents a wick holder and medical use, which are not supported by the image. The description is otherwise clear but contains significant hallucinations.<br><br>This image depicts an old-fashioned brass oil lamp that appears to be designed for use as a hand-held or medical lamp. The lamp has a cylindrical body made of polished brass, which shows signs of age with some wear and patina on its surface. A brass nozzle is mounted on top of the cylinder, which likely serves as the wick holder for lighting purposes. The lamp features a black leather strap that wraps around the cylinder's body and extends upwards into a handle with a U-shaped grip. Small rivets secure the strap to the body, and there is an additional leather strap hanging from the side, possibly used as an extra handle or for securing the lamp when not in use. At the top of the lamp near the nozzle, there are intricate cutout designs in the metal plate, possibly for ventilation or aesthetic purposes. The nozzle itself looks adjustable, with a screw mechanism that might control the flow of oil for the wick. The overall design suggests that this is a durable and functional piece, likely used in situations requiring portability and reliability.|
|---|

[Figure 2]

|RubiCap-7B-PixMoCap<br><br>No hallucinated prop (no oval dish stand) This caption accurately describes the watercolor painting, noting the purple crystal cluster (likely amethyst), the brown earthy base, and the green section on the left. It covers the main visual elements, the impressionistic style, and the shadow, with no hallucinated details. The language is clear and specific.<br><br>This is a watercolor painting of a geode or crystal formation, depicted on a textured white paper background. The main subject of the artwork is a cluster of crystals with a purple hue, which appears to be amethyst, given its deep purple color and translucent quality. The crystals have a rough, jagged texture, with some parts appearing more transparent than others. The crystals are perched atop a brown, earthy base that resembles the outer layer of a geode or a piece of rock. This base is painted with a mix of brown and dark brown tones, giving it a natural, earthy appearance. To the left of the crystals, there is a green section, possibly representing another mineral or a different type of crystal, adding contrast to the composition. The painting includes subtle shading and highlights, particularly around the edges of the crystals and the base, to create depth and dimension. The shadow cast by the crystals onto the paper adds realism to the scene, suggesting a light source from above. The overall style of the painting is loose and impressionistic, capturing the essence of the subject rather than focusing on fine details.|
|---|

|Qwen2.5-VL-72B-Instruct<br><br>Hallucination/error identified While the caption describes the crystal and its colors well, it incorrectly states that the stone is on a 'beige, oval-shaped base' resembling a dish or stand, which is not present; the base is just a painted shadow. Otherwise, it is clear and mostly accurate.<br><br>This image appears to be a watercolor painting of a gemstone or mineral specimen. The stone is situated on a beige, oval-shaped base which looks like it might be a dish or stand for displaying the gem. The gem itself has an interesting composition with multiple layers and colors: 1. The top portion is predominantly purple, resembling a cluster of amethyst crystals. The purple hues vary from light lavender to deep violet, creating a sense of depth and texture. 2. Below the purple section, there is a layer of green that could be malachite or another green mineral. This part has a smooth, polished look compared to the rougher texture of the amethyst. 3. The bottom part of the gem transitions into darker brown tones, possibly representing a matrix or base rock from which the crystals grew. The painting style uses soft, blended watercolors to capture the natural appearance of the gem. The brushstrokes are visible but blend seamlessly to create a realistic portrayal of the stone's texture and color variations. The background is plain white, which helps to emphasize the gem as the central focus of the piece.|
|---|

[Figure 3]

|RubiCap-7B-PixMoCap<br><br>No hallucinated background (no body of water or bench) Caption A accurately describes the sign's text, the devil silhouette, the guardrail, and the rusted farm equipment. It notes the rural setting, trees, and sky, with no hallucinated elements. The description is clear, specific, and covers all major visual elements.<br><br>The image shows a roadside sign situated in an open grassy area, bordered by trees and a road in the background. The sign is rectangular with a white background and bold black text that reads "GO TO CHURCH" at the top. Below this, in smaller red text, it says "Or the Devil Will Get You!" accompanied by a red silhouette of a devil figure on the left side of the sign. The sign appears to be mounted on two metal posts and is positioned near the edge of a paved road. In the foreground, there is a metal guardrail, suggesting the photo was taken from inside a vehicle. To the right of the sign, there is a rusted piece of farm equipment, possibly a wheel or part of a tractor, lying on the ground. The surrounding environment includes a mix of green grass and scattered trees, with a cloudy sky above. The overall setting suggests a rural or semi-rural location.|
|---|

|Qwen2.5-VL-72B-Instruct<br><br>Hallucination/error identified Caption B is mostly accurate and detailed, correctly describing the sign, text, devil illustration, guardrail, and farm equipment. However, it mentions a 'small wooden or metal structure resembling a bench or fence' and a 'faint view of a body of water,' neither of which are clearly visible, introducing mild hallucination.<br><br>This image depicts an outdoor scene with a large roadside sign near what appears to be a grassy area surrounded by trees, likely in a rural or countryside setting. The sign has bold red and black text on a white background that reads: "GO TO CHURCH Or the Devil Will Get You!" The word "Devil" is emphasized in red. To the left of the text, there is a red illustration of a devil figure with horns and a tail. The sign is mounted on a metal frame supported by posts. In the background, there are more green trees and a faint view of a body of water, possibly a pond or lake. A small wooden or metal structure resembling a bench or fence is also visible behind the sign. On the right side of the image, part of an old piece of machinery or agricultural equipment can be seen lying on the ground, adding to the rustic feel of the scene. A metal guardrail is visible at the bottom of the image, suggesting that the photo was taken from inside a moving vehicle, as the motion blur at the bottom edge indicates. The sky above is partly cloudy, with patches of blue showing through. The overall composition gives off a slightly ominous but humorous tone due to the sign's message.|
|---|

[Figure 4]

|RubiCap-7B-PixMoCap<br><br>No unverifiable speculation (just 'kitchen', not 'restaurant') This caption accurately describes the stainless steel steamer, the tied crabs, their color, the stovetop, and the kitchen background. It notes the warm lighting and provides a thorough, clear summary. There are no hallucinations; all details are visible.<br><br>The image shows a stainless steel steamer basket containing several live crabs, which appear to be tied with black and white string or twine. The crabs have a reddish-orange hue, indicating they might be cooked or partially cooked. The basket is placed on a stovetop, suggesting the crabs are being steamed. The background features a tiled backsplash with a beige and brown color scheme, and part of a wooden cabinet is visible at the top left corner. The lighting in the scene is warm, highlighting the texture of the crabs' shells and the sheen on the metal basket. The overall setting appears to be a kitchen environment.|
|---|

|Qwen2.5-VL-72B-Instruct<br><br>Hallucination/error identified The caption correctly identifies the cooked crabs, their binding, the steamer setup, and the kitchen setting. However, it speculates about the possibility of a restaurant and the inviting atmosphere, which is not directly verifiable from the image.<br><br>This image shows a steaming pot containing several cooked crabs. The crabs appear to be tied with black ropes, likely to keep their legs and claws from moving during the cooking process. The pot is made of stainless steel and has two handles on either side. It is placed on top of another larger pot, which might be holding water for steaming. The background features a tiled wall with earthy tones, suggesting that the setting might be a kitchen or a restaurant. The overall scene conveys a warm and inviting atmosphere, typical of a meal being prepared.|
|---|

[Figure 5]

|RubiCap-7B-PixMoCap<br><br>No hallucinated garnish (no cinnamon sticks) This caption accurately describes the number of desserts, their glass containers, and the visible layers (crumbly base, cream, banana, whipped cream, chocolate shavings). It notes the arrangement and background details. There are no hallucinations, and the language is clear and specific.<br><br>The image shows a collection of six small glass jars, each containing a layered dessert. The desserts appear to be a type of trifle or layered pudding, with distinct layers visible: 1. **Bottom Layer**: A layer of what looks like a dark, crumbly base, possibly cookie crumbs or a similar texture. 2. **Middle Layer**: A creamy, light-colored layer, which could be cream or custard. 3. **Top Layer**: A layer of sliced bananas, adding a touch of yellow and freshness to the dessert. 4. **Final Layer**: A generous topping of whipped cream, sprinkled with chocolate shavings for added texture and flavor. Each jar is garnished with a few additional chocolate shavings on top, enhancing the visual appeal. The jars are arranged neatly on a white surface, likely a table, with a checkered cloth partially visible in the background. The lighting is bright, highlighting the textures and colors of the dessert layers. The overall presentation suggests a homemade or artisanal style, emphasizing simplicity and elegance.|
|---|

|Qwen2.5-VL-72B-Instruct<br><br>Hallucination/error identified The caption is mostly accurate and detailed, correctly describing the layers and presentation. However, it hallucinates the presence of cinnamon stick pieces, which are not visible in the image. Otherwise, it is clear and covers most elements.<br><br>This image showcases six small glass jars filled with a layered dessert, likely a type of trifle or parfait. Each jar contains multiple layers of different textures and colors: 1.<br><br>**Bottom Layer:** A layer of crushed cookies or biscuit crumbs forms the base. 2. **Next Layer:** A creamy custard-like layer, possibly caramel-flavored, sits atop the cookie crumb base. 3. **Middle Layer:** Slices of banana are arranged neatly on top of the custard layer. 4. **Top Layer:** A generous dollop of whipped cream crowns each dessert. Additionally, each jar is garnished with: - **Chocolate Shavings:** Sprinkled generously over the whipped cream for added texture and flavor. - **Cinnamon Stick Pieces:** Small pieces of cinnamon stick are placed on top, adding a touch of spice and visual appeal. The jars are arranged on a white surface, possibly a plate or tray, which contrasts nicely with the rich colors of the dessert. In the background, there is a hint of a plaid fabric, suggesting a cozy and homely setting. The overall presentation is inviting and suggests a homemade treat, perfect for sharing.|
|---|

[Figure 6]

###### Figure 7 Qualitative comparison to 72B frontier using RubiCap-7B-PixMoCap.

Caption Comparison (Judged by GPT4.1)

|RubiCap-7B-DenseFusion<br><br>Correct exhibition dates (1975, not 1978) Caption A accurately describes the poster's main elements: the abstract painting, the color palette, the exhibition title, venue, and dates, as well as the white background and artist's signature. The only minor issue is that the signature is not clearly legible in the image, but its presence is plausible. All text and layout details are correct, and the description is clear and thorough.<br><br>The image is a poster advertising an exhibition titled "deKooning in East Hampton." The poster features a vibrant abstract painting as its central artwork, which appears to be an expressionist piece with bold brushstrokes and a mix of colors including yellow, blue, red, pink, and black. The composition is dynamic, with the colors blending and overlapping in a non-representational manner. Below the painting, the text "deKooning in East Hampton" is prominently displayed in large, elegant serif font. Beneath this title, smaller text indicates the venue: "THE SOLOMON R. GUGGENHEIM MUSEUM, NEW YORK," along the dates of the exhibition: "FEBRUARY 10 - APRIL 27, 1975." The overall design of the poster is clean and minimalistic, with the focus on the artwork and the event details. The background of the poster is white, which contrasts with the colorful painting and the black text, making the information stand out clearly. The signature of the artist, Willem de Kooning, is faintly visible at the bottom right corner of the painting.|
|---|

|Qwen2.5-VL-72B-Instruct<br><br>Hallucination/error identified Caption B contains several factual errors: the exhibition dates are incorrect (should be 1975, not 1978), and it claims the artist's signature is near the center-right, which is not visible. It also mentions a black border, which does not exist. The rest of the description is mostly accurate and clear, but these errors are significant.<br><br>This image is a promotional poster for an art exhibition of Willem de Kooning, held at The Solomon R. Guggenheim Museum in New York from February 10 to April 23, 1978. The central feature of the poster is an abstract painting by de Kooning. The artwork features dynamic and expressive brushstrokes with a mix of colors, including beige, red, black, yellow, blue, and white. The composition is dominated by bold, gestural lines that create a sense of movement and energy, typical of de Kooning's style. The text on the poster is simple and clear. It reads "de Kooning" in large, bold, black letters at the bottom left, followed by "in East Hampton" in smaller text below it. The name "de Kooning" appears again as a handwritten signature near the center-right. At the bottom of the poster, additional information is provided: "THE SOLOMON R. GUGGENHEIM MUSEUM, NEW YORK" and the dates of the exhibition. The overall design of the poster emphasizes the artwork itself, allowing viewers to appreciate the raw, expressive nature of de Kooning's painting. The choice of a white background for the poster ensures that the focus remains on the vibrant and dynamic painting. The poster is framed with a black border, giving it a polished and professional appearance.|
|---|

[Figure 7]

with

|RubiCap-7B-DenseFusion<br><br>Correct panel count (3 panels, not 4; no hallucinated wine or skewer) This caption accurately describes the three-panel collage, the actions of the people, the kitchen setting, and the food being cooked. It covers all significant visual elements, including the attire, kitchen details, and the mood. The language is clear and specific, with no hallucinated details.<br><br>This image is a collage of three photographs, each capturing different moments in a kitchen setting. 1. **Top Left Panel**: A woman with long dark hair, wearing a white strapless top and a peach-colored skirt, is feeding a man with a piece of food from her hand. The man, dressed in a black vest over a white shirt, is leaning back slightly, appearing to enjoy the moment. The kitchen background includes modern cabinetry, a stainless steel refrigerator, and a hanging light fixture. 2. **Top Right Panel**: This photo shows a close-up of a frying pan on a stovetop containing clams cooking in a broth. The clams are open, indicating they are being cooked, and steam is visible rising from the pan. The background is slightly blurred, focusing attention on the pan. 3. **Bottom Panel**: This image captures the same woman from the first panel, now standing at the stove, actively engaged in cooking. She appears to be stirring or handling a pot on the stove, which contains what looks like a boiling liquid. Another pot is visible on the stove, along with a kettle and a bottle of olive oil on the counter. The kitchen has a modern design with white cabinets, a sink, and various kitchen utensils and appliances. The overall theme of the images suggests a homey, intimate cooking scene, possibly involving a shared meal preparation or a casual dining experience.|
|---|

|Qwen2.5-VL-72B-Instruct<br><br>Hallucination/error identified This caption incorrectly claims there are four photographs (there are three), and hallucinates a glass of wine and a skewer, neither of which are visible. It is otherwise detailed and clear, but these hallucinations and the panel count error are significant.<br><br>This image is a collage of four photographs, capturing scenes from what appears to be a cooking and dining experience. ### Top Left: - **Setting:** A modern kitchen with stainless steel appliances. - **People:** A woman in a white top and red-orange skirt is feeding something to a man who is dressed in a suit jacket over a white shirt, with a black apron. The woman is laughing joyfully, holding a piece of food (likely seafood) on a thin stick or skewer, while the man looks up in anticipation or amusement. They seem to share a playful moment. - **Details:** The woman also holds a glass of wine in her other hand, indicative of a celebratory mood. ### Top Right: - **Focus:** A close-up shot of a frying pan containing clams being cooked in an oil-based sauce. The clams are open, revealing their meat. The pan is placed on a stovetop, suggesting that the dish is being actively prepared. ### Bottom Left: - **Scene:** The woman is seen again, standing near the stove, intently watching or tending to a pot or pan. She uses a spoon or similar utensil, possibly stirring or checking the contents of the pan. - **Background:** The kitchen has white cabinetry, various cooking tools like knives and a cutting board, and sunlight streaming through a window, adding a warm and inviting atmosphere. ### Bottom Right: - **Focus:** A pot on the stovetop with boiling water and visible steam, accompanied by a kettle and other cooking vessels. The scene suggests active cooking and preparation. - **Background:** The stovetop and surrounding area show ingredients and utensils, emphasizing the culinary setting. Together, these images depict a lighthearted and enjoyable cooking session, likely between two people, with a focus on both the process and the interaction. It conveys warmth, enjoyment, and the pleasure of shared food preparation.|
|---|

[Figure 8]

|RubiCap-7B-DenseFusion<br><br>No hallucinated shamrock (asterisk in 'Y', not shamrock) This caption accurately describes the green t-shirt, the white text, the content and style of the text, and the lack of other visible elements. It does not hallucinate any details not present in the image and is clear and concise.<br><br>The image shows a close-up of a green t-shirt with white text printed on it. The text reads "STATE PATTY'S DAY" in large, bold, uppercase letters. Below the main text, in smaller uppercase letters, is the phrase "HAPPY VALLEY, PA." The font style for "STATE PATTY'S DAY" is slightly more stylized compared to the simpler, clean font used for "HAPPY VALLEY, PA." The shirt appears to be made of a standard cotton material, and the fabric is smooth and tightly woven. The lighting in the image is even, highlighting the texture of the fabric and the crispness of the printed text. There are no other visible details or decorations on the shirt apart from the text. The background is entirely filled by the green fabric of the t-shirt, with no additional elements or distractions present.|
|---|

|Qwen2.5-VL-72B-Instruct<br><br>Hallucination/error identified While the caption correctly describes the shirt and text, it hallucinates a shamrock in the 'Y' (which is not present; the 'Y' has a small asterisk), and speculates about the event's association with Penn State and Irish culture, which is not visually verifiable.<br><br>This image shows a close-up of a green T-shirt with white printed text. The design on the shirt reads "STATE PATTY'S DAY" in large, bold letters. The "Y" in "PATTY'S" is stylized with shamrocks, a symbol commonly associated with St. Patrick's Day. Below the main text, it says "HAPPY VALLEY, PA," which likely refers to Happy Valley, Pennsylvania. The shirt appears to be promoting an event or celebration related to State Patty's Day, a local holiday observed in State College, Pennsylvania, near Penn State University. The overall theme and color scheme suggest a connection to Irish culture and St. Patrick's Day festivities.|
|---|

[Figure 9]

|RubiCap-7B-DenseFusion<br><br>No hallucinated objects (no red bucket or shovel on seesaw) This caption accurately describes the book cover, including the title, author, seesaw, outdoor setting, and the educational theme. It notes the seesaw's motion and the presence of trees and shrubs. There are no hallucinations, and the language is clear and specific.<br><br>The image is the cover of an educational book titled "Zoom In on Simple Machines: Levers" by Andrea Rivera. The title is prominently displayed in bold, white font against a green rectangular background at the top of the cover. Below the title, the author's name, "Andrea Rivera," is written in smaller text. The central focus of the cover is a photograph of a seesaw (a type of lever) situated outdoors on a grassy field under a clear blue sky with scattered clouds. The seesaw consists of a wooden plank balanced on a central support, with two red seats attached to either end. The seesaw appears to be in motion, as one end is raised while the other is lowered, suggesting it is being used or demonstrating the principle of a lever. The surrounding environment includes trees and shrubs in the background, adding to the outdoor setting. The overall tone of the image is bright and inviting, emphasizing the educational theme of simple machines.|
|---|

|Qwen2.5-VL-72B-Instruct<br><br>Hallucination/error identified While the caption is detailed and mostly accurate, it incorrectly describes the left side of the seesaw as having a 'red bucket-like object' and the right side as a 'black-handled shovel with a red blade,' which are not present. The seesaw simply has red seats and black handles. This is a significant hallucination.<br><br>This is the cover of an educational book titled **"Levers"**, part of the "Zoom In on Simple Machines" series by Andrea Rivera. The design emphasizes the concept of levers, using a clear and visual representation. ### Description: 1. **Title and Series Information**: - The title **"Levers"** is prominently displayed in large white font in the center. - Above it, the series name **"Zoom In on Simple Machines"** is written in yellow and green text on a green background banner, located at the top of the image. 2.<br><br>**Visual Element**: - A seesaw (a classic example of a lever) dominates the cover. It consists of a wooden plank balanced on a central fulcrum. - The left side of the seesaw has a red bucket-like object attached to the end, while the right side has a black-handled shovel with a red blade attached. - This creates a practical demonstration of how levers work, showing balance or imbalance depending on weight distribution. 3.<br><br>**Background**: - The scene is set outdoors on a bright day, with a lush green lawn as the foreground and a wide-open blue sky with scattered white clouds as the backdrop. Trees line the horizon in the distance, adding depth to the image. 4. **Author Name**:<br><br>- At the bottom of the cover, the author's name, **Andrea Rivera**, is written in a simple white font. The overall design effectively combines educational content with engaging visuals, aimed at teaching readers about levers in a relatable context.|
|---|

[Figure 10]

|RubiCap-7B-DenseFusion<br><br>No incorrect text size claim (text size not decreasing across lines) This caption accurately describes the text, color, font, and lack of additional elements. It also briefly interprets 'CRB' but clearly marks it as likely, not certain. The description is concise and clear, with no hallucinated details.<br><br>The image features a light teal background with bold, black text centered on the page. The text reads: "CRB Adults Attacking & Killing Young Coconut Palms." The font is sans-serif and appears to be a standard typeface used for clarity and readability. There are no additional graphics, illustrations, or other elements present in the image. The focus is entirely on the text, which conveys a message about the impact of CRB (likely referring to Coconut Rhinoceros Beetle) adults on young coconut palms.|
|---|

|Qwen2.5-VL-72B-Instruct<br><br>Hallucination/error identified The caption correctly describes the background, text, and layout, but it incorrectly claims that the text size decreases with each line, which is not visually verifiable. The rest of the description is accurate and clear, but this error affects accuracy and completeness.<br><br>The image is a simple, text-based graphic with a light blue background and black text. The text is centered and consists of four lines, each line larger than the one below it, creating a hierarchical emphasis on the message: 1. **CRB Adults** - This is the largest text on the image, likely indicating the primary subject or title. 2. **Attacking &** Slightly smaller than the first line, it continues the main statement. 3. **Killing Young** - Further elaborates on the action described, continuing the sentence from the previous line. 4. **Coconut Palms** - The smallest text, completing the sentence and providing the target of the attack. The overall message conveyed by the text is that "CRB Adults" (likely referring to Coconut Rhinoceros Beetles) are attacking and killing young coconut palms. The design is straightforward and informational, with no additional graphics or images included.|
|---|

[Figure 11]

###### Figure 8 Qualitative comparison to 72B frontier using RubiCap-7B-DenseFusion.

Caption Comparison (Judged by GPT-4.1)

|RubiCap-7B-DenseFusion<br><br>Correct tool identification (fruit picker with bag; no gloves, coconut, or gear mechanism) This caption accurately describes the fruit picker tool, its metallic structure, the presence of a pole, the inset image, and the 'HARKO' branding. It does not hallucinate any features and covers all significant elements, though it could mention the bag attached to the picker more explicitly. The language is clear and specific.<br><br>The image shows a close-up of a mechanical device, likely a fruit picker or harvesting tool, attached to a tree. The main focus is on the metallic component of the device, which appears to be made of stainless steel or a similar reflective material. This part has multiple circular slots or openings, designed to grip and collect fruits. The device is mounted on a cylindrical metal arm that extends from the tree. In the background, there are green leaves and parts of the tree, indicating the device is being used outdoors. A small inset image in the upper right corner provides a closer view of the metallic component, highlighting its design and structure. The text "HARKO" is visible in the top left corner, suggesting the brand or manufacturer of the device. The overall scene suggests an agricultural or gardening context where the tool is being used for fruit harvesting.|
|---|

|Qwen2.5-VL-7B-Instruct<br><br>Hallucination/error identified This caption is inaccurate and hallucinates gloves, a coconut, and a mechanism with gears, none of which are present. The tool is a standard fruit picker with a bag, not gloves or a coconutspecific device. The description is clear but almost entirely fabricated.<br><br>The image shows a close-up of a mechanical fruit picker, specifically designed for harvesting coconuts. The tool is composed of several components: 1. **Gloves**: The main part of the tool is a pair of large gloves with a reinforced palm area and fingers that are designed to fit around the coconut fruit. These gloves protect the worker's hands from bruising and sharp objects while ensuring a firm grip on the coconut. 2. **Handle**: A metal handle extends downward, which is likely used for gripping and maneuvering the tool. The handle appears to be sturdy and designed for a comfortable grip to prevent fatigue during extended use. 3. **Mechanical Mechanism**: Attached to the handle is a mechanical mechanism that likely includes gears or teeth. This part of the tool is used to securely attach the glove to the coconut when it is being harvested. 4. **Coconut**: In the background, there is a green coconut fruit. The glove is partially wrapped around the coconut, demonstrating how the tool works in practice. The coconut is located on a tree branch, indicating that the tool is being used outdoors in a tropical environment where coconuts are typically found. In the top left corner, there is a logo with the text "HARKO" and an image of a house, suggesting that the brand name of the fruit picker is HARKO. The overall design of the tool appears to be robust and functional, tailored for efficient coconut harvesting.|
|---|

[Figure 12]

|RubiCap-7B-DenseFusion<br><br>No hallucinated props (no wooden spoons, gold oven mitts, or kitchen setting) This caption accurately describes the split-screen format, the costumes, gestures, and the onscreen text. It notes the puppet features of BÖRK and the human features of TERRIANN, as well as their matching outfits and gestures. It does not hallucinate any objects or actions not present in the image, but it misses the watermark and does not mention the background color. The description of the hands as 'furry' is accurate, and the gestures are correctly described.<br><br>The image is a split-screen comparison of two characters, both dressed as chefs, with the left side labeled "BÖRK" and the right side labeled "TERRIANN." - **BÖRK** (left side):<br><br>- The character is a puppet with a large, expressive face featuring a red nose, brown mustache, and eyebrows. - He has a chef's hat on his head, which is made of fabric and appears to be slightly crumpled. - His hands are large and furry, resembling animal paws, and he is wearing a striped shirt with blue and white horizontal stripes. - He is also wearing a red bow tie and a white apron. - BÖRK is making a gesture with his hands, one hand raised with fingers spread out, and the other hand giving a thumbs-up. -<br><br>**TERRIANN** (right side): - The character is a person with short black hair, glasses,<br><br>and a friendly smile. - She is also wearing a chef's hat, similar in style to BÖRK's, but it appears more structured. - Her outfit includes a striped shirt with blue and white horizontal stripes, a red bow tie, and a white apron. - Her hands are large and<br><br>furry, similar to BÖRK's, and she is making a gesture with her hands, one hand raised with fingers spread out, and the other hand giving a thumbs-up. - The text at the bottom identifies her as "Crazy Chinese Chef from Cookies & Clogs." Both sides of the image have a clean background, and the overall theme suggests a playful or comedic take on chef characters, possibly from different media franchises.|
|---|

|Qwen2.5-VL-7B-Instruct<br><br>Hallucination/error identified This caption contains several inaccuracies: it claims both characters are holding wooden spoons (they are not), describes 'gold oven mitts' (the hands are furry, not mitts), and says the chef's hats are silver (they are white). It also asserts a kitchen setting, which is not visible. The rest of the description is mostly accurate, but these hallucinations are significant.<br><br>The image is a side-by-side comparison of two characters dressed as chefs. On the left, there is a character from "The Muppets," specifically Börk, who is identified as the "Crazy Swedish Chef." He has a puppet-like appearance with fur, a red nose, and is wearing a blue and white striped shirt, a white apron, and a silver chef's hat. Börk is holding a wooden spoon in one hand and giving a thumbs-up with the other, while dressed in gold oven mitts. On the right, there is a human character dressed as a chef, identified as TERRJANN, the "Crazy Chinese Chef" from "Cookies & Clogs." This person has long dark hair, glasses, and is wearing a similar outfit to Börk: a blue and white striped shirt, a white apron, and a silver chef's hat. They are also holding a wooden spoon in one hand and giving a thumbs-up with the other. The human character is wearing gold oven mitts as well. Both characters appear to be in a kitchen setting, suggesting they are ready to cook or perform a culinary demonstration. The image is likely meant to humorously compare the two characters, both of whom are known for their exaggerated cooking styles in their respective shows.|
|---|

[Figure 13]

|RubiCap-7B-DenseFusion<br><br>No hallucinated pedestrian (snowy urban scene accurately described; no invented walker) This caption accurately describes the snowy urban scene, the presence of bare branches with brown leaves, and the overlaid text. It mentions vehicles and a motorcycle, which are visible, and correctly interprets the atmosphere. It does not hallucinate any elements and is clear and concise.<br><br>The image captures a snowy urban scene viewed through a window, with the foreground dominated by bare tree branches and scattered brown leaves. The snow-covered ground and rooftops suggest a winter setting. In the background, a street is visible, where vehicles are driving slowly, possibly due to the slippery conditions. The road appears wet and reflective, with some cars and a motorcycle visible. The atmosphere is overcast, with diffused light indicating cloudy weather. Overlaid on the image is text that reads: "On our Way to Kathmandu Part 12" along with a website URL, "www.twistingspokes.com," suggesting this is part of a travel or journey series. The overall tone of the image conveys a cold, wintry day in an urban environment.|
|---|

|Qwen2.5-VL-7B-Instruct<br><br>Hallucination/error identified The caption describes the snowy branches and overlaid text accurately, but it hallucinates a person walking and possible structures in the background, which are not visible. The mention of 'spokes' as possibly referring to bicycles is speculative.<br><br>The image depicts a snowy landscape, likely captured during winter. The foreground features a close-up view of tree branches covered with snow and adorned with dried leaves, suggesting that the trees have lost their foliage for the season. The branches extend outward from the left side of the frame, partially obscuring the view of the snow-covered ground and the background. In the midground, there appears to be a road or pathway, also blanketed in snow. A person can be seen walking along this path, adding a sense of scale and activity to the otherwise serene and quiet scene. The background includes more trees and possibly some structures, though they are mostly obscured by the snow and the tree branches in the foreground. Overlaid on the image is text that reads "On our Way to Kathmandu Part 12" followed by a website URL: "www.twistingspokes.com." This suggests that the image is part of a travel series or blog post documenting a journey towards Kathmandu, which is the capital city of Nepal. The title and the website indicate that the content might be related to travel, possibly cycling given the mention of "spokes," which could refer to bicycle wheels.|
|---|

[Figure 14]

|RubiCap-7B-DenseFusion<br><br>No hallucinated watch details (no black strap, '10:10' time, or outdoor greenery) This caption accurately describes the main visual elements: a close-up of a wristwatch, the overlaid quote, the musixmatch logo, and the general color tone. It avoids hallucinating details about the background, only suggesting a possible setting. The description is clear and covers all significant elements, including the text and branding.<br><br>The image features a close-up of a person's wrist wearing a wristwatch, with the watch face prominently displayed. The watch has a black dial with white hour markers and hands, and it appears to be a classic design with a silver or metallic bezel. The strap of the watch is dark-colored, possibly leather or a similar material. Overlaid on the image is a text quote that reads: "I been blocking my feelings but now I can't hide em." Below the quote, the song title "Leave Me Alone" by Hopsin is written. The text is in a clean, sansserif font, white in color, and positioned centrally over the image. In the bottom left corner, there is a small logo for "musixmatch," which is a music lyric website, indicating that the image may have been sourced from their platform. The background of the image is slightly blurred, showing part of what seems to be a bed or cushioned surface, suggesting a relaxed or casual setting. The overall tone of the image is introspective and emotional, aligning with the theme of the quote.|
|---|

|Qwen2.5-VL-7B-Instruct<br><br>Hallucination/error identified The caption accurately describes the watch, text, and overall tone, but it hallucinates a black strap (the strap is not clearly black), a time of 10:10, and an outdoor setting with greenery, which are not visible. The musixmatch logo is actually bottom center, not bottom right.<br><br>The image is a stylized, sepia-toned photograph that features a close-up of a wrist wearing a wristwatch. The watch has a black strap and a silver case with a round face. The time on the watch appears to be around 10:10. The background is blurred but suggests an outdoor setting with some greenery. Overlaid on the image is a text quote in bold white letters that reads, "I been blocking my feelings but now I can't hide em." Below the quote, there is another line of text that says, "Leave Me Alone" followed by the name "HOPSIN." In the bottom right corner, there is a small logo for "musixmatch," which is likely indicating that the quote is sourced from their website or app. The overall tone of the image conveys a sense of introspection and emotional vulnerability, aligning well with the theme of the quote about hidden emotions.|
|---|

[Figure 15]

|RubiCap-7B-DenseFusion<br><br>No hallucinated actions (no page turning, pointing, or unverifiable monkey species) This caption accurately describes the main elements: a person holding a newspaper, a photo of two monkeys, the headline, and the coffee cup. It avoids specific unverifiable details and does not hallucinate. However, it is slightly verbose and could be more concise.<br><br>The image shows a person holding a newspaper with their left hand, while the right hand is partially visible, suggesting they are reading or about to turn a page. The newspaper features a prominent photograph of two monkeys, likely green monkeys, on the front page. The headline reads "Azul, Azizi debut at zoo," indicating that these animals have made their public debut at a zoo. The text below the headline mentions "Njala Zoo" and other details about the animals' arrival. The background includes a wooden table surface, and part of a white coffee cup with a saucer is visible in the upper right corner, suggesting a casual setting. The lighting is bright, casting soft shadows, and the overall atmosphere appears relaxed and informal. The newspaper's layout is typical of a daily publication, with various articles and sections visible, though the specific content of the articles is not legible in the image.|
|---|

|Qwen2.5-VL-7B-Instruct<br><br>Hallucination/error identified The caption accurately describes the main scene and visible elements, but it hallucinates the monkeys' species, the person's actions (turning the page, pointing), and some article content that is not legible. The rest is clear and relevant.<br><br>The image shows a person sitting at a wooden table, reading a newspaper. The newspaper features an article titled "Azul, Azizi debut at zoo" under the Niabi Zoo section. The article discusses the opening of a new African primates exhibit at the zoo, which includes two male Wolf's Guenon monkeys named Azul and Azizi. The person is holding the newspaper with both hands, and their left hand is turning the page. The right hand appears to be pausing or gesturing as if pointing at something on the page. The top right corner of the newspaper displays a smaller article about U.S. Rep. Eric Swalwell participating in a Times caucus forum. On the table next to the person, there is a white saucer with a black coffee cup filled with coffee. The background of the image is a wooden surface, suggesting that the setting is indoors, possibly at home or in a casual environment. The overall scene conveys a moment of leisure and information gathering.|
|---|

[Figure 16]

###### Figure 9 Qualitative comparison to base model using RubiCap-7B-DenseFusion.

