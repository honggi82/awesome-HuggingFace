# arXiv:2501.09012v3[cs.CV]2Sep2025

## Multimodal LLMs Can Reason about Aesthetics in Zero-Shot

### Ruixiang Jiang

### Chang Wen Chen

rui-x.jiang@connect.polyu.hk Department of Computing The Hong Kong Polytechnic University Hong Kong, China

chen.changwen@polyu.edu.hk Department of Computing The Hong Kong Polytechnic University Hong Kong, China

#### Abstract

The rapid technical progress of generative art (GenArt) has democratized the creation of visually appealing imagery. However, achieving genuine artistic impact – the kind that resonates with viewers on a deeper, more meaningful level – remains formidable as it requires a sophisticated aesthetic sensibility. This sensibility involves a multifaceted cognitive process extending beyond mere visual appeal, which is often overlooked by current computational methods. This paper pioneers an approach to capture this complex process by investigating how the reasoning capabilities of Multimodal LLMs (MLLMs) can be effectively elicited to perform aesthetic judgment. Our analysis reveals a critical challenge: MLLMs exhibit a tendency towards hallucinations during aesthetic reasoning, characterized by subjective opinions and unsubstantiated artistic interpretations. We further demonstrate that these hallucinations can be suppressed by employing an evidence-based and objective reasoning process, as substantiated by our proposed baseline, ArtCoT. MLLMs prompted by this principle produce multifaceted, in-depth aesthetic reasoning that aligns significantly better with human judgment. These findings have direct applications in areas such as AI art tutoring and as reward models for image generation. Ultimately, we hope this work paves the way for AI systems that can truly understand, appreciate, and contribute to art that aligns with human aesthetic values. Project homepage: https://github.com/songrise/MLLM4Art.

#### CCS Concepts

• Applied computing → Fine arts; • Human-centered computing → Empirical studies in HCI; • Computing methodologies → Knowledge representation and reasoning; Natural language generation; Philosophical/theoretical foundations of artificial intelligence.

#### Keywords

Computational Aesthetics; Art appreciation and criticism; User modeling; Multimodal LLMs reasoning; Chain-of-thought; Hallucination

ACM Reference Format:

Ruixiang Jiang and Chang Wen Chen. 2025. Multimodal LLMs Can Reason about Aesthetics in Zero-Shot. In Proceedings of the 33rd ACM International Conference on Multimedia (MM ’25), October 27–31, 2025, Dublin, Ireland. ACM, New York, NY, USA, 25 pages. https://doi.org/10.1145/3746027.3754961

This work is licensed under a Creative Commons Attribution 4.0 International License. MM ’25, Dublin, Ireland

© 2025 Copyright held by the owner/author(s). ACM ISBN 979-8-4007-2035-2/2025/10 https://doi.org/10.1145/3746027.3754961

[Figure 1]

Let me calculate …

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

Existing AI model

[Figure 11]

[Figure 12]

Score: 0.34

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

Emotional impact

[Figure 17]

[Figure 18]

Art History in 1900s

Let me think …

[Figure 19]

[Figure 20]

A girl

[Figure 21]

Cubism

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

Muddy Palette

Picasso

[Figure 26]

[Figure 27]

Abstraction

Human

Figure 1: Existing models assess aesthetics as a black-box, which is oversimplified and can misinterpret the aesthetic value. The complex, multifaceted human aesthetic judging process must be captured for human-aligned aesthetic evaluation. Credit: Girl with a Mandolin (1910), by Pablo Picasso.

#### 1 Introduction

Human aesthetic perception is a multifaceted cognitive process beyond mere visual appeal. It encompasses factors including, but not limited to, uniqueness, narrative, cultural background, and emotional resonance, all of which contribute to the unique aesthetic value of artworks [24, 38, 43, 45]. The importance of understanding this complex process is widely acknowledged by art practitioners and scholars, as it not only deepens our theoretical understanding of beauty but also informs and inspires art creation [24, 71]. Analogously, in the era of rapidly advancing GenArt, an algorithmic understanding of aesthetics is also crucial to generating truly unique and meaningful artistic expressions [8, 11, 27, 37, 68].

Despite its complexity, almost all of the existing computational approaches have oversimplified the concept of aesthetics. These methods predominantly rely on end-to-end learning to obtain blackbox scoring models, without engaging with the complex cognitive processes underlying aesthetic judgment. Specifically, systems like traditional Image Quality Assessment (IQA) [85, 98], AI-Generated IQA (AGIQA) [93], and more recent human preference models (e.g., PickScore [49], VisionPrefer [90]) follow this limited paradigm. While these scoring models may effectively assess image quality in some instances, they provide no insights or interpretation for their predictions. More importantly, the measurement of image quality or visual appeal is a fundamentally superficial understanding of aesthetics [24, 38, 40, 45, 69, 71], and this limited view fosters misalignment with human aesthetic perception in practice [5, 7, 12, 39, 64, 68, 73]. For instance, impactful artworks that are not immediately visually pleasing but involve more in-depth artistic

expression may be undervalued by these models. Therefore, modeling the multifaceted cognitive process in human aesthetic judgment is essential to build a truly human-aligned aesthetic evaluator, as illustrated in Fig. 1.

To capture the underlying process of human aesthetic perception, we operationalize principles from art and philosophy. In particular, the “formal analysis” [5, 47, 82] in art criticism offers valuable insight that informs our approach. This analysis is a structured method that first describes visual elements in a painting, critically analyzes its formulation, and connects these observations to broader aesthetic principles for objective aesthetic judgment. This analysis extends beyond the subconscious feeling of beauty; it actively weaves perception, emotion, cultural context, and artistic knowledge into a coherent evaluative narrative. In other words, it connects observation and interpretation into words, a holistic capability that bears a striking resemblance to the inference-time reasoning paradigm of recent Multimodal Large Language Models (MLLMs) [54, 55, 63, 81, 87]. This parallel presents an opportunity to model the cognitive process behind human aesthetic judgment, inspiring our central research question:

To what extent does reasoning improve the alignment of MLLMs’ aesthetic judgments with humans in a zeroshot setting?

Answering this question requires understanding both human aesthetic judgment and MLLMs’ aesthetic reasoning. We model human judgment through a novel tournament-style aesthetic relationship with philosophy-inspired designs [45, 79, 80] to foster objective comparisons, overcoming the limitation in previous scorebased evaluation approaches. Human expert judgment is collected on a newly proposed dataset, FineArtBench, with large-scale semantic labels to facilitate statistically unbiased correlation studies. In parallel, we task MLLMs with the same evaluation. We leverage Chain-of-Thought (CoT) prompting [87] to elicit not just a final decision, but a detailed reasoning trace for why one artwork is preferred over another. Finally, we analyze the alignment between judgments from humans and MLLMs.

Through extensive experiments across different MLLMs, we identify a key challenge of aesthetic reasoning: hallucination. Specifically, we observe a high tendency for MLLMs to hallucinate subjective, unsubstantiated opinions and artistic interpretations. Our analysis further underscores that the bottleneck lies in the reasoning process rather than the base model. To overcome it, we find that evidence-based and objective reasoning is essential to suppress these hallucinations. With these valuable insights, we demonstrate that a simple baseline, ArtCoT, facilitates exceptionally in-depth and multifaceted aesthetic reasoning. This objective reasoning elevates general-purpose MLLMs into state-of-the-art aesthetics evaluators that outperform existing IQA and preference models by a large margin. Furthermore, this powerful aesthetic reasoning has direct applications in human-computer interaction (HCI) and generative art, such as AI art tutors [22, 92] and reinforcement learning from AI feedback (RLAIF)-enhanced art generation [7, 15, 27, 90], due to its unprecedented interpretability and comprehensiveness.

We summarize our contributions as follows:

(1) We introduce the task of aesthetic reasoning.

- (2) We propose a philosophy-inspired principled approach for objectively modeling human aesthetic judgment.
- (3) We introduce FineArtBench, the first large-scale semantically annotated dataset for broad art-related tasks.
- (4) Our analysis provides valuable insights into the hallucination bottleneck of MLLMs in aesthetic reasoning.
- (5) Our zero-shot baseline ArtCoT points out a promising way for human-aligned aesthetic judgment in AI.

2 Related Works

- 2.1 Generative Art, Fine Art and Aesthetics

At its core, GenArt is a new medium for artistic expression. Neural style transfer (NST) [1, 17, 21, 35, 41, 42, 83] and image generative modeling [18, 59, 61, 67, 75] are two central techniques in this field. Initially, research focused primarily on fidelity challenges, and the goal was to learn the distribution of real artworks, utilizing featurelevel metrics such as Gram Loss [21] and FID [32, 89] to quantify the performance. More recently, the focus has shifted towards the affective challenge [37, 52, 68, 90], ensuring that the generated results are not only plausible but also visually pleasing. Preference models, such as PickScore [49] and HPS [91], are developed to address this challenge. With the rapid scaling of generative models, the acquisition of attractive images has become democratized. This progression allows us to pursue the ultimate challenge of art: striving for artistically meaningful and impactful artworks rather than mere visual appeal. This tri-stage progression was articulated at least 200 years ago by philosophers such as Hegel [30]. It is contextualized by the transition in art history from technical execution to artistic expression [24]. Just as aesthetics played a central role in that transition, an algorithmic understanding of aesthetics is key to advancing generative art. This paper directly addresses this issue by proposing a fundamental shift from aesthetic scoring to reasoning to better align with human values.

- 2.2 Computational Aesthetics

Early aesthetic evaluation algorithms are primarily rule-based, assessing the visual quality of paintings or photos using different principles [8, 23]. While simple and interpretable, these principles lack the generality to explain all cases. With the emergence of deep learning, data-driven methods [31, 70] attempt to directly learn aesthetic evaluators from image collections labeled with human preferences. In parallel to these explorations, a highly related line of work is developing quantitative metrics for NST. Among them, ArtFID [89] and Art Score [12] are two notable methods. Despite the different technical details, all of these black-box models oversimplify the complex aesthetic judgment process and are reportedly misaligned with human perception [5, 7, 39, 73]. Recent methods [6, 36, 46, 95, 97] attempt to incorporate MLLMs for more interpretable and human-aligned aesthetics evaluation. Nevertheless, most require costly instruction-tuning and still focus on visual appeal. This paper focuses on capturing the aesthetics in its full sense by explicitly modeling the aesthetic judgment process.

- 2.3 CoT Reasoning

CoT prompting [28, 50, 87] is a simple yet powerful technique for reasoning in both LLMs and MLLMs. It functions by requiring the

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

Zero-shot CoT

[Figure 33]

[Figure 34]

[Figure 35]

Left: 45% right: 55%

[Figure 36]

[Figure 37]

B>A>C>E>F>D

content Style Artists

[Figure 38]

[Figure 39]

Experts’ Ranking

[Figure 40]

Feedback arc +=1

[Figure 41]

[Figure 42]

Task Distribution and and Painting

ArtCoT

Expert Judgment

Heuristics

[Figure 43]

Input

F

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

Summarizer

CS Analyzer Art Critic

[Figure 48]

Elo: B>C>E>…

- A vs E

[Figure 49]

- B vs F

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

Zero-Shot: D>A>F>E>B>C

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

E

[Figure 58]

[Figure 59]

- B vs D

[Figure 60]

- C vs F …

[Figure 61]

[Figure 62]

[Figure 63]

ArtCoT: A>C>B>E>F>D

Bradley-Terry: B>A>C>…

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

MLLM’s Ranking

RLAIF HCI Others

- a) 2AFC Questions Collection b) Human Judgment Modeling c) Aesthetic Reasoning via MLLMs and Alignment Study

[Figure 72]

FineArtBench Artworks

[Figure 73]

[Figure 74]

[Figure 75]

A

[Figure 76]

B

[Figure 77]

- C D

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

Tournament graph Sampled 2AFC

pruning ranking

[Figure 82]

[Figure 83]

Correlation Analysis

Zero-shot Aesthetic Reasoning

Objective Expert Judgment Modeling

Uniform tournament Scheduling

- Figure 2: Overview of our correlation evaluation pipeline. (a) First, we generate 2AFC comparison set by sampling content and style from the FineArtBench. (b) Next, human experts perform aesthetic judgment on sampled questions. To ensure annotation consistency, these judgments are filtered with two heuristic indicators and are aggregated as global rankings. (c) In parallel, we leverage different reasoning methods to derive the judgment from MLLMs. Finally, we calculate the correlation coefficient between the MLLM and human rankings as indicators of aesthetic alignment.

model to decompose the task before arriving at a final answer. It has been proven effective for various tasks that benefit from enhanced logical reasoning, such as math, logic, or commonsense problems. This paper investigates the application of CoT to aesthetics evaluation, a domain that extends beyond strict logical reasoning. Our analysis reveals a special hallucination issue when CoT is applied in aesthetics, and the proposed ArtCoT overcomes this challenge.

where 𝜋(𝑖) maps the candidate index to its global rank, and 𝑦1 ≻ 𝑦2 means 𝑦1 is judged to be superior than 𝑦2. We measure the alignment as the correlation between the ranks constructed from human experts and the MLLM(s), and our target is to study the effectiveness and potential challenges of reasoning in enhancing this alignment.

#### 3.2 Modeling Human Aesthetic Judgment

#### 3 Methodology 3.1 Problem Definition

Judgment Collection. Directly obtaining the complete ranking as in Eq. 1 by asking users to sort many items at once is cognitively demanding and can lead to inconsistent judgments. Therefore, inspired by findings in psychometrics [73, 78], we instead employ a two-alternative forced choice (2AFC) task for more efficient judgment collection. Specifically, we iteratively present users with the task specification (𝑐,𝑠), along with two candidate artworks (𝑦𝑖,𝑦𝑗). Users are required to judge the relatively superior artwork without the option to indicate a tie or skip the comparison. We model the judged aesthetic superiority relationship probabilistically:

Motivation.Despitethesubjectivenatureofaesthetic taste, philosophers [38, 45, 72, 79, 80] and psychologists [58, 65] generally acknowledge a degree of consensus regarding the universality of aesthetic judgments, suggesting the possibility of objective comparative evaluation. This consensus, however, is not absolute. Comparing vastly different artworks, such as “Mona Lisa” and “Guernica” is much more challenging due to divergent content and style. In contrast, a controlled scenario, such as an art lesson where students draw porcelain vases using the cross-hatching technique, offers a more robust basis for objective aesthetic comparison. This approach aligns with established practices in art criticism and art education, where rubrics are employed for artwork assessment [2, 25]. Overall, these observations guide us to explore aesthetic judgment within comparable content and style context for objective comparison.

𝑄𝑖 𝑄𝑖 + 𝑄𝑗

, (2)

𝑃(𝑦𝑖 ≻ 𝑦𝑗 |𝑐,𝑠) =

where𝑄𝑖 represents a latent aesthetic competence of𝑦𝑖, conditioned on the task context of (𝑐,𝑠).

Sampling Strategies for 2AFC. Given the combinatorial complexity of C × S × A (107 in our case), exhaustive pairwise comparisons are infeasible. Therefore, we generate 2AFC comparisons by sampling, which reduces to a classical tournament scheduling problem [51]. For a specific candidate set Y𝑐,𝑠, all feasible comparisons can be modeled as complete graph 𝐺(𝑉,𝐸), where the node 𝑉 Y𝑐,𝑠 is the set of candidate results and 𝐸 stands for pairwise comparisons. We consider two strategies for sampling 𝐸′ ⊆ 𝐸, corresponding to two real-world scenarios:

Objective. To formalize our motivation, we aim to derive a statistically robust global ranking of a set of artists (or art generators) or artworks (or generated art) from human feedback. Formally, let C, S, and A denote the sets of all content, styles, and artists, respec-

tively (within the studied scope). Each combination (𝑐,𝑠) ∈ C × S constitutes a task presented to the artists. The resulting artworks from artists 𝑎𝑖 ∈ A, denoted as 𝑦𝑖 = 𝑎𝑖(𝑐,𝑠), form the set of candidates for this task: Y𝑐,𝑠 = {𝑦1, . . .,𝑦𝑘}. We then model human aesthetic judgment as a partial ordering on the set Y𝑐,𝑠:

(1) Global sampling: Uniformly sample an arbitrary number of edges without replacement. This approach is suitable for covering a wider range of content and style to facilitate the ranking of artists across all tasks.

𝑦𝜋(1) ≻ 𝑦𝜋(2) ≻ · · · ≻ 𝑦𝜋(𝑘), (1)

[Figure 84]

[Figure 85]

Content labels

Content Attributes

[Figure 86]

[Figure 87]

decoration technology

[Figure 88]

[Figure 89]

[Figure 90]

Color diversity

Interior

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

Limited Diverse

Animal

6.5% 5.9%

[Figure 99]

[Figure 100]

[Figure 101]

9.3%

Element diversity

Architecture

[Figure 102]

[Figure 103]

5.2%

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

Limited Diverse

Content prompt: “A professional real estate photograph of a luxurious […]” Scene labels: [interior, urban, decoration] Color diversity: low Element diversity: low Element density: low Composition: non-object centric Camera setup: wide angle

[Figure 112]

[Figure 113]

|[Figure 114]|
|---|

10.1%

Others

[Figure 115]

[Figure 116]

9.7%

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

Element density

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

Low High

12.1%

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

Activities

14.0%

[Figure 140]

[Figure 141]

[Figure 142]

Composition

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

12.9%

[Figure 155]

[Figure 156]

Non-object centric Object centric

Objects

Content prompt: “Two men are emptying a basket of quid onto […]” Scene labels: [activities] Color diversity: high Element diversity: low Element density: high Composition: non-object centric Camera setup: generic

13.9%

|[Figure 157]|
|---|

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

Camera setup

[Figure 167]

[Figure 168]

Urban

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

Generic Wide Others

Nature

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

Content

###### Style Labels

[Figure 192]

[Figure 193]

Style Attributes

[Figure 194]

[Figure 195]

[Figure 196]

Prompt length

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

Short medium long

|[Figure 209]|
|---|

[Figure 210]

[Figure 211]

Style prompt: “Jackson Pollock abstract expressionism” Prompt label: [artist, art movement] Attribute: [abstract, western, contemporary] Source: WikiArt

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

Abstraction

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

Non-representational figurative abstract

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

Demographic

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

Western Oriental Others

|[Figure 252]|
|---|

[Figure 253]

[Figure 254]

[Figure 255]

Style prompt: “Edward Hopper painting style featuring muted tones, […]” Prompt label : [artist, color and tone, art movement] Attribute: [non-representational, western, modern] Source: DiffusionDB

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

Time

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

Contemporary Classical Modern

Style

artmovementmediumColorandtonetechniqueartworktitlemixedothersgenressentiment

artist

##### Figure 3: The FineArtBench dataset. Left: The label distribution and semantic attributes for both content and style. Right: examples of two content (top) and two style (bottom). Zoom in for a better view.

(2) Per-task sampling: Sample edges |𝐸′| ∈ [|𝑉 | − 1, |𝐸|] such that the sub-graph is connected with maximum node degree uniformness, meaning that each candidate shall be compared for a similar number of times. This is suitable for determining the ranks of artworks within a specific task.

We design a novel greedy algorithm (described in Appendix A.2) to efficiently achieve the requirements of per-task sampling.

Dealing with Divergent Judgment Annotation. Due to the inherent subjectivity of aesthetic perception, human judgment can diverge even within the constrained scenario. This variation can arise from inherent annotation noise and is particularly prevalent when the artists perform equally well or poorly. While addressing these challenging edge cases could be valuable, in this paper, given that this is an initial exploration of aesthetic reasoning, we focus on modeling the universality of aesthetics [45, 72, 79] with suppressed inter-annotator divergence. This suppression is necessary as it allows us to create an unbiased, reproducible testbed for quantitative evaluation. To this end, we employ the following two heuristics to mitigate annotation subjectivity:

- (1) 2AFC disagreement: We exclude the 2AFC questions where judgments tie (i.e., 𝑃(𝑦𝑖 ≻ 𝑦𝑗 |𝑐,𝑠) ≈ 50%).
- (2) Intransitive relationship: To deal with intransitive judgment, such as (𝑎 ≻ 𝑏,𝑏 ≻ 𝑐,𝑐 ≻ 𝑎), we apply a feedback arc set (FAS) algorithm [94] to detect feedback arcs 𝐸′𝑓 and drop the whole task when |𝐸′𝑓 |/|𝐸′| ≥ 𝜂.

GlobalRankDerivation.We utilize Bradley-Terry(BT)model [9]

and Elo [19] algorithm to construct Eq. 1 from 2AFC. In the BT model, 𝑄𝑖 = exp(𝜃𝑖), where the latent parameter {𝜃𝑖}𝑘𝑖=1 is optimized with maximum likelihood estimation. The Elo algorithm uses 𝑄𝑖 = 10𝑅𝑖/400, where 𝑅𝑖 is the rating of candidate-i that is updated

online. The rank of a candidate within a specific task is sorted by its competence 𝑄𝑖, and we treat the average competence among all tasks as the competence of an artist.

#### 3.3 Zero-shot Aesthetics Reasoning via ArtCoT

Recentinference-timescalingparadigms, particularly the CoT prompting [50, 87] demonstrate impressive zero-shot task-solving capability. However, CoT is more widely studied for commonsense and logical reasoning. Applying it to aesthetic judgment, which is not a direct logical task, remains an unexplored direction. Very recently, research suggest the alignment challenge in this application may differ from logical tasks [60, 77]. In practice, the key issue we identified is hallucination, where judgments are made prematurely, relying on subjective and superficial justifications. In other words, during aesthetic reasoning, MLLMs inherently tend to “feel” rather than to “reason”, and this hallucinated feeling is tied with severe degradation of the MLLM’s aesthetic judgment with human alignment. We detail this issue in the Sec. 5.1.

Motivated by the formal analysis in art criticism [5, 47, 82], our key insight is to explicitly ground the MLLMs’ decision-making with evidence and domain knowledge in art, so as to reduce spurious reliance on hallucinated feelings. Based on this principle, we propose a simple yet effective baseline, ArtCoT, which frames the multifaceted aesthetic judgment as an evidence-based reasoning process.

ArtCoT features a two-stage reasoning process. In the initial stage, the MLLM acts as a CS Analyzer, generating a detailed and concrete analysis of the input task (𝑐,𝑠), as well as the paired 2AFC images (𝑦𝑖,𝑦𝑗). This stage provides the context for aesthetic judgment. In the second phase of Art Critic, the MLLM is prompted to critically evaluate its previous observation and to synthetically

##### Table 1: Quantitative comparison of aesthetic judgment correlation. This table presents the statistical correlation (Spearman’s 𝜌 ↑) and significance (𝑝 value, ↓) between rankings generated by different aesthetic evaluators and expert judgments. Both conventional models and MLLMs with different reasoning methods are compared. Performance improvement and decline are calculated as normalized changes relative to the base prompt method for each of the corresponding MLLMs. MLLMs reasoned with ArtCoT consistently demonstrates strong alignment with human judgment.

Per-Artist Alignment Per-Task Alignment Elo Bradley-Terry Elo Bradley-Terry

Model Reasoning

𝜌 ↑ 𝑝-value ↓ 𝜌 ↑ 𝑝-value ↓ 𝜌 ↑ 𝑝-value ↓ 𝜌 ↑ 𝑝-value ↓

Random guess – 0.061 0.896 0.059 0.855 0.068 0.153 0.026 0.290 CLIP-IQA[85] – 0.366 0.038 0.321 0.091 0.324 0.004 0.314 0.002 Compare2Score [98] – 0.091 0.940 0.085 0.950 0.178 0.042 0.173 0.119 CLIP Score [66] – -0.024 0.945 0.030 0.951 -0.333 0.347 0.006 0.987 PickScore [49] – 0.275 0.192 0.329 0.063 0.193 < 10−3 0.201 0.001 HPSv2 [91] – 0.343 0.074 0.352 0.056 0.321 < 10−3 0.285 0.007 Aesthetics Predictor [70] – 0.424 0.016 0.456 0.008 0.385 < 10−3 0.404 < 10−3

GPT-4o Base 0.395 0.084 0.432 0.023 0.328 0.003 0.331 0.006 Claude 3.5-sonnet Base 0.341 0.032 0.217 0.236 0.312 < 10−3 0.367 < 10−3 Gemini 1.5-flash Base 0.496 0.003 0.513 0.001 0.479 < 10−3 0.353 < 10−3

GPT-4o Zero-shot CoT 0.415 +3% 0.034 0.487 +10% 0.015 0.299 -4% 0.097 0.313 -3% 0.031 Claude 3.5-sonnet Zero-shot CoT 0.293 -7% 0.177 0.264 +6% 0.174 0.108 -30% 0.068 0.081 -45% 0.082 Gemini 1.5-flash Zero-shot CoT 0.206 -58% 0.292 0.272 -50% 0.142 0.376 -20% < 10−3 0.327 -4% < 10−3 GPT-4o ArtCoT 0.630 +39% 0.003 0.721 +51% 0.001 0.591 +39% < 10−3 0.548 +32% < 10−3 Claude 3.5-sonnet ArtCoT 0.598 +39% 0.009 0.546 +42% 0.016 0.492 +26% < 10−3 0.487 +19% < 10−3 Gemini 1.5-flash ArtCoT 0.705 +41% < 10−3 0.741 +47% < 10−3 0.624 +28% < 10−3 0.577 +35% < 10−3

Claude-3.7-sonnet-thinking (Built-in) 0.285 0.225 0.154 0.481 0.171 0.014 0.175 0.009 Gemini-2.0-flash-thinking (Built-in) 0.204 0.262 0.254 0.191 0.157 0.004 0.245 0.003

judge [44, 45] the aesthetic value. The analysis in this stage is multifaceted, with discussion not limited to stylistic features of 𝑠, aesthetic principles, technical execution of the two candidates, and potential emotional impact. Overall, these two stages link context and world knowledge into the aesthetic judgment, fundamentally different from the existing paradigm based on vision features alone. Effectively, it establishes a rubric [2, 25], translating the subjective perception of beauty into more well-defined sub-tasks that can be logically reasoned [13, 16, 53].

Importantly, we aim to capture this reasoning trace as a model of the cognitive process behind aesthetic judgment, instead of simply arriving at the 2AFC result. This trace is a rich representation and is highly interpretable, which can benefit a wide range of downstream applications. As this paper focuses on assessing the alignment between MLLMs and humans, we further employ a summarizer that takes the reasoning trace as input and outputs a binary answer for the winner. The overall workflow is visualized in Fig. 2.

#### 3.4 FineArtBench

The collection of content and style (i.e., C, S) is most widely studied in NST literature [17, 34, 41]. However, the majority of these papers rely on ad-hoc sampling from existing datasets, such as MS-COCO [57] and WikiART [88]. This ad-hoc approach has two critical limitations: (1) Limited Scope and Inherent Bias, usually covering a small range of art styles that may lead to bias in aesthetics evaluation. (2) Lack of Semantic Annotation: the lack of

semantic-level labeling of style and content impedes a fine-grained understanding of the methods’ strengths and weaknesses.

To circumvent these challenges, we propose FineArtBench, a dataset for GenArt with unprecedented scale, diversity and quality. In terms of scale, it contains 1,000, and 1,000 densely annotated content and styles, respectively. The data is harvested from diverse sources [48, 57, 86, 88], and we use MLLMs, enabling comprehensive comparison for downstream art-related tasks. Moreover, we provide content and style in two modalities (text and image), further expanding its versatility. Second, we perform a semi-automated annotation process (detailed in Appendix A.3) to provide semantic information, such as scene in content (e.g., nature, portrait) and abstractness of style (e.g., figurative, abstract). These annotations enable an unprecedented, fine-grained analysis of model performance. A comparison of FineArtBench with existing datasets is presented in Tab. 2, and Fig. 3 presents an overview of FineArtBench.

#### 4 Experiments 4.1 Experiment Setup

Stylization Models. Given the size of |C × S|, hiring real human artists for scalable dataset collection is unfortunately infeasible: a single painting requires days of expertise. On the other hand, existing publicly available artworks have a limited style that only covers a small subset of S. Even if we collect some paired

[Figure 281]

MM ’25, October 27–31, 2025, Dublin, Ireland Ruixiang Jiang and Chang Wen Chen

##### Table 2: Comparison of representative GenArt benchmark datasets. The proposed FineArtBench offers significantly more content and style instances, with fine-grained multimodal annotations.

##### Table 3: Inter-Annotator Agreement. Higher goodness-of-fit 𝑅2 indicate more concordant global ranking.

Raw + prune uncertain + drop non-transitive 𝑅2 ↑ 0.17 0.26 0.31

Dataset/Protocol # Content # Style Multimodal Semantic Labeling

StyleID [14] 20 40 ✗ ✗ LAION-Aesthetics [70] - ∼50 ✗ ✗

ArtBench [56] - 10 ✗ ✗ StyleBench [20] 20 73 ✓ ✗ AGIQA-3k [20] ∼ 300 5 ✓ ✗

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

FineArtBench 1000 1000 ✓ ✓

or

paintings, the MLLMs are prone to memorization rather than reasoning [6]. To this end, we primarily rely on neural style transfer (NST) models as our “artists.” We evaluated a total of 10 NST models, all executed with the default configurations suggested by respective authors. The methods include AdaIN [35], ArtFlow [1], ControlNet [96], DDIM [74], DiffArtist [41], DiffStyler [34], InstantStyle [84], Instruct-pix2pix [10], StyleID [14], and Sty-Tr2 [17]. These stylization models encompass a wide range of architectures and exhibit varying degrees of stylization capability. This diversity provides a robust testbed for evaluating the performance of MLLMs in art evaluation.

[Figure 286]

Left, because of it conveys a sense of harmony, making it a better artistic interpretation. So […] (misaligned)

Zero-shot CoT

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

The use of thick brushstroke and impasto aligns with Monet’s Impressionist style, and […] So, the right one is better. (aligned) ArtCoT

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

##### Figure 4: A minimal example of the hallucination issue. The opinionated and subjective reasoning of zero-shot CoT leads to misaligned aesthetic judgment.

Alignment Metrics. Following [12, 41, 97], we use Spearman’s correlation coefficient [76] to quantify the alignment between judgment from experts and MLLMs. A Spearman’s 𝜌 closer to 1 indicates a stronger positive linear correlation of ranking, which suggests better alignment. To ensure robustness, we calculate the average 𝜌 from five random and independent splits for the global (per-artist) sampling scheme, while for per-task sampling, we correlate each independent task. Considering the size of the independent test and typical 𝑝 values, we combine 𝑝 using Pearson’s and Fisher’s methods for per-artist and per-task scenarios, respectively, as suggested by [29]. A lower 𝑝 value means stronger statistical significance.

prune uncertain human feedback with 𝑃(𝑦𝑖 ≻ 𝑦𝑗 |𝑐,𝑠) ∈ [0.4, 0.6], and remove intransitive task with 𝜂 > 0.15. As a result, 20.1% of the feedback is filtered out due to pairwise divergence, while the remaining 15.3% is removed due to a high intransitivity. To quantify the inter-annotation agreement, we use McFadden’s 𝑅2 [62] to assess the transitivity of global ranking constructed from BT and Elo, reported in Tab. 3. An 𝑅2 ≥ 0.2 is generally considered a good fit, which implies the judgments are statistically consistent. Further details are in Appendix A.1.

Compared Methods. As the first exploration on MLLMs for aesthetic reasoning, finding direct comparisons can be challenging. Therefore, we broadly compare three groups of methods. (a), IQA methods: including CLIP-IQA [85] and Compare2Score [98]; (b) Text-image alignment and preference scores: CLIP Score [66], Aesthetic Predictor [70], PickScore [49] and HPSv2 [91]. Despite ArtScore [12] sharing the most similar objective with this paper, we exclude it due to the unavailability of code at the time this paper was written. We also consider (3) Zero-shot MLLMs, where three main-stream MLLMs are inspected, including GPT-4o [63], Gemini 1.5 [81], and Claude 3.5 [3]. We apply different reasoning methods on top of them, including base (directly output the winner), zeroshot CoT [87], and the proposed ArtCoT. We also consider recent reasoning MLLMs, including Claude-3.7-sonnet-reasoning and Gemini-2.0-flash-thinking. The experiment of all MLLMs is repeated thrice with default hyperparameters to ensure robustness.

#### 4.3 Main Result: Human Alignment

We compare the correlation and statistical significance of all compared methods under two sampling strategies. The results in Tab. 1 demonstrate that aesthetic reasoning substantially enhances an MLLM’s aesthetic judgment, achieving average improvements of 44%, 30% in the per-artist and per-task setup, respectively. Compared with other specialized IQA or preference models, MLLMs prompted with ArtCoT demonstrate state-of-the-art alignment even without task-specific optimization (i.e., in zero-shot). Interestingly, the zero-shot CoT adversely affects alignment, decreasing it by 16% and 18% on average. Overall, these results indicate that (1) reasoning can indeed make MLLMs human-aligned aesthetic evaluators, conditioned on (2) proper elicitation of their reasoning capability. Next, we analyze these findings in detail.

#### 4.2 Human Judgment Collection

We recruit 18 human experts with general knowledge of fine art for annotation. We collected a total of 80,000 initial judgments. For per-task sampling, we sample O(𝑘𝑙𝑜𝑔(𝑘)) for each task. We

##### Table 4: Response subjectivity and hallucination from different prompting methods. We report the average length of thinking trace (in words), two subjectivity metrics, and human-annotated hallucination score (normalized) from four perspectives. Reasoning traces generated with ArtCoT contain less subjectivity and hallucination. (*): Trace from Claude-3.7-sonnet-thinking.

###### Response Subjectivity Hallucination (Factuality) Hallucination (Extrinsic)

Method Avg. length

TextBlob ↓ Word frequency (%) ↓ Visual ↓ Textual ↓ Opinionated ↓ Interpretation ↓ Thinking MLLM (*) 578.71 0.44 11.41 0.21 0.15 0.44 0.49

GPT, 0-shot 71.73 0.44 17.34 0.31 0.17 0.45 0.33 Claude, 0-shot 223.51 0.44 20.15 0.27 0.22 0.58 0.48 Gemini, 0-shot 123.07 0.46 21.05 0.31 0.17 0.54 0.59

GPT, ArtCoT 384.45 0.30 -25% 5.04 -14% 0.29 -3% 0.19 +2% 0.32 -24% 0.18 -22% Claude, ArtCoT 580.21 0.29 -26% 6.15 -18% 0.31 +5% 0.13 -12% 0.22 -85% 0.31 -33% Gemini, ArtCoT 740.21 0.23 -43% 6.51 -18% 0.28 -4% 0.19 +5% 0.17 -80% 0.22 -53%

5 Analysis and Discussion

- 5.1 Zero-shot CoT Reinforces Hallucination

During aesthetic reasoning, we identify a special pattern, where MLLMs tend to hallucinate feelings rather than to reason objectively. This tendency significantly degrades an MLLM’s aesthetic judgment, leading to the counter-intuitive result where zero-shot CoT negatively affects alignment with human judgment.

Specifically, in the zero-shot CoT setting, we observe that MLLMs tend to arrive at rapid conclusions, employing subjective and less concrete language to justify their judgments. This hallucination can be further categorized into two types. The first is the use of opinionated and subjective words. For instance, opinion verbs (e.g., “feels”, “senses”) and hedge words (e.g., “appears”, “seemingly”) are frequently employed. These terms often indicate a lack of indepth consideration, weak logical connection to the given task, and are less supportive for aesthetic judgment [72]. Another pattern is the generation of unsubstantiated artistic interpretations that are superficial and not substantiated with facts or explanations. These hallucinations are consistently observed in misaligned reasoning traces across different MLLMs, corroborating recent findings [33, 60]. Detailed conversation examples of ArtCoT are provided in Appendix C.

#### 5.2 Evidence-based Reasoning Suppresses Subjectivity and Hallucination

In stark contrast to base prompting and zero-shot CoT, ArtCoT produces thinking traces that are objective, logical, and well-grounded with evidence. Fig. 4 gives a minimal example to demonstrate this difference. To quantify this improvement, we first measure the subjectivity of responses by calculating the frequency of subjective verbs in the lemmatized responses of MLLMs. We also follow established practices in natural language processing (NLP) by performing lexicon-based sentiment analysis using the TextBlob subjectivity score, which ranges from 0 (objective) to 1 (subjective).

For a fine-grained human evaluation of hallucinations, we analyze cases where the MLLM’s preference disagreed with human judgment. We asked users to read the MLLM’s reasoning and label hallucinations, which we categorize into two primary types. The first, Factual errors, includes incorrect image recognition (visual)

and inaccurate style explanations (textual). They are also called intrinsic hallucinations [4], as they can be directly deduced from the input. Another type is Extrinsic hallucinations, comprising opinionated responses and artistic interpretations, which correspond to the subjective language and unsubstantiated interpretations identified in Sec. 5.1. Users rated the severity of each of the four types of hallucinations on a scale of [0,1,2]. The aggregated results are presented in Tab. 4.

Human evaluation reveals that all models exhibit relatively less severe hallucinations in terms of visual and textual aspects. This suggests that factual inaccuracies are not the primary source of hallucinations in this context. Instead, the prevalent use of subjective language, characterized by opinionated expressions and unsubstantiated artistic interpretations, significantly impedes the ability of MLLMs to align their aesthetic judgments with those of humans. Moreover, the elevated response subjectivity observed in zero-shot CoT, as evidenced by the remarkably consistent TextBlob scores (≈ 0.45) and the frequency of subjective words, provides further support for our finding.

The human evaluation further supports our central hypothesis that the bottleneck for MLLMs in aesthetic reasoning is not visual recognition but reasoning elicitation. All models demonstrate sufficient comprehension of the input; the core challenge is preventing them from defaulting to subjective shortcuts. We illustrate this challenge is surmountable. By enforcing an evidence-based, reflective structure, ArtCoT directly mitigates subjective expression, substantially suppressing hallucinations and driving significant gains in human alignment.

#### 5.3 Multifaceted Aesthetic Reasoning is Versatile

The semantic label in FineArtBench enables fine-grained understanding of MLLM’s performance. For instance, in Fig. 5 , we visualize the alignment performance on different content image complexity and style prompt categories. The proposed ArtCoT outperforms both base and zero-shot CoT prompting in all examined sub-classes, demonstrating its versatility.

The performance gap is most pronounced on tasks with concrete style instructions, such as those involving “long prompt” and prompts specifying particular “art movement”. We posit that such

[Figure 300]

Table 6: Ablation on image resolution and source information. We report the correlation 𝜌 (averaged from BT and Elo) of different input setups: content image, style prompt, and image sub-sampling factor.

[Figure 301]

Base Zero-shot CoT ArtCoT

Content Style Resolution Per-artist 𝜌 ↑ Per-task 𝜌 ↑

✓ ✓ 1/2 0.630 -42% 0.432 -44% ✓ ✓ 1/4 0.502 -91% 0.285 -82%

- ✗ ✗ full 0.476 -100% 0.416 -49%
- ✗ ✓ full 0.678 -23% 0.465 -36%

Figure 5: Fine-grained comparison of different MLLM prompting schemes. We show the Spearman’s 𝜌 for perinstance alignment, grouped by representative attribute in FineArtBench. ArtCoT elicits aesthetic reasoning in all scenarios, especially for tasks with a specific style instruction.

✓ ✗ full 0.557 -69% 0.521 -22%

###### ✓ ✓ full 0.739 0.607

[Figure 302]

[Figure 303]

[Figure 304]

“Good! Can you make the chiaroscuro stronger?"

[Figure 305]

“Good story telling. Try to strengthen the triangular composition"

- Table 5: Ablation on component of ArtCoT. We ablate the content/style analyzer and the art critic. The full two-stage aesthetics reasoning facilitates human-aligned judgment.

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

“Painting of martyrdom of a robot saint, by Caravaggio"

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

CS-analyzer Art Critic Per-artist 𝜌 ↑ Per-task 𝜌 ↑

✗ ✓ 0.630 0.532 ✓ ✗ 0.531 0.366

Task First attempts Second attempts Third attempt

##### Figure 6: A minimal illustration of ArtCoT-enhanced art generation.

###### ✓ ✓ 0.739 0.607

specific instruction sets a more concrete objective for both the art creation and evaluation [4]. Compared with obscure instructions, this specificity facilitates the MLLM to more objectively perform the two-hop reasoning, resulting in reduced hallucination and improved human alignment [4, 26, 33, 87].

#### 5.4 Ablations

Components of ArtCoT. We ablate the key components of ArtCoT, specifically the CS analyzer and the art critic, and report the results in Tab. 5. The complete ArtCoT prompt achieves the highest aesthetic alignment. Particularly, removing the art critic phase induces the most significant decline, underscoring its critical role. This provides further evidence that reasoning is the key bottleneck.

Input to MLLM. We ablate input modalities and image resolutions, with results reported in Tab. 6. The full resolution achieves the best result. This is because artwork details, such as strokes, are important aesthetic factors. For input modalities, providing style information is important in per-artist alignment, while including the reference image affects per-instance alignment the most. For either setting, providing all input modalities achieves the best alignment, meaning that task-constrained 2AFC aids objective aesthetic reasoning and judgment to better align with human value.

#### 5.5 Broader Impact

We havedemonstratedthatevidence-based reasoning makes MLLMs human-aligned art evaluators. Many applications can benefit from this exploration, both in terms of RLAIF and HCI. Fig. 6 illustrates how generative art may benefit from the reward signals with aesthetic reasoners to improve the generation, utilizing a pipeline

akin to [27]. A similar pipeline may also be employed for humancentered and interactive art education.

#### 5.6 Limitations and Future Works

Aesthetics, a subject studied for centuries, possesses complexities that cannot be fully addressed in a single paper. Numerous avenues for improvement exist, with the most significant challenge being the computational modeling of interperson subjectivity. One possible technical direction involves conditioning Eq. 2 on the audience identity to enable personalized aesthetic models. A deeper understanding of the human part provides valuable insights, such as the philosophical theories and discussions on aesthetic judgment [37, 45, 72].

#### 6 Conclusion

Prevailing aesthetic computing approaches are fundamentally limited, reducing rich human judgment to simplistic visual scoring. True alignment requires modeling the cognitive process behind this judgment. Models that fail to do so cannot meaningfully align with human perception. This paper addresses this gap by pioneering the task called aesthetic reasoning with MLLMs. In-depth analysis highlights the hallucination issues stemming from subjective opinions and unsubstantiated interpretations, leading to misaligned judgments. We further identify a promising solution through an evidence-based and objective reasoning process. Ultimately, this work provides a blueprint for developing AI that does not just recognize visual appeal, but can reason about the aesthetics in-depth—a critical paradigm shift toward creating genuinely collaborative and artistically insightful systems.

#### 7 Acknowledgment

This research was supported by the Hong Kong Research Grants Council (GRF-15229423).

#### References

- [1] Jie An, Siyu Huang, Yibing Song, Dejing Dou, Wei Liu, and Jiebo Luo. 2021. Artflow: Unbiased image style transfer via reversible neural flows. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 862–871.
- [2] Heidi Andrade, Joanna Hefferen, and Maria Palma. 2014. Formative assessment in the visual arts. Art Education 67, 1 (2014), 34–40.
- [3] Anthropic. [n.d.]. Meet Claude. https://www.anthropic.com/claude.
- [4] Zechen Bai, Pichao Wang, Tianjun Xiao, Tong He, Zongbo Han, Zheng Zhang, and Mike Zheng Shou. 2024. Hallucination of multimodal large language models: A survey. arXiv preprint arXiv:2404.18930 (2024).
- [5] Sylvan Barnet. 2015. A short guide to writing about art. Pearson Upper Saddle River, NJ.
- [6] Yi Bin, Wenhao Shi, Yujuan Ding, Zhiqiang Hu, Zheng Wang, Yang Yang, SeeKiong Ng, and Heng Tao Shen. 2024. Gallerygpt: Analyzing paintings with large multimodal models. In Proceedings of the 32nd ACM International Conference on Multimedia. 7734–7743.
- [7] Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine.

2023. Training diffusion models with reinforcement learning. arXiv preprint arXiv:2305.13301 (2023).

- [8] Yihang Bo, Jinhui Yu, and Kang Zhang. 2018. Computational aesthetics and applications. Visual computing for industry, biomedicine, and art 1 (2018), 1–19.
- [9] Ralph Allan Bradley and Milton E Terry. 1952. Rank analysis of incomplete block designs: I. The method of paired comparisons. Biometrika 39, 3/4 (1952), 324–345.
- [10] Tim Brooks, Aleksander Holynski, and Alexei A Efros. 2023. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 18392–18402.
- [11] Rebecca Chamberlain, Caitlin Mullin, Bram Scheerlinck, and Johan Wagemans.

2018. Putting the art in artificial: Aesthetic responses to computer-generated art. Psychology of Aesthetics, Creativity, and the Arts 12, 2 (2018), 177.

- [12] Junyu Chen, Jie An, Hanjia Lyu, Christopher Kanan, and Jiebo Luo. 2024. Learning to Evaluate the Artness of AI-generated Images. IEEE Transactions on Multimedia

(2024).

- [13] Bo T Christensen and Linden J Ball. 2016. Dimensions of creative evaluation: Distinct design and reasoning strategies for aesthetic, functional and originality judgments. Design studies 45 (2016), 116–136.
- [14] Jiwoo Chung, Sangeek Hyun, and Jae-Pil Heo. 2024. Style injection in diffusion: A training-free approach for adapting large-scale diffusion models for style transfer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 8795–8805.
- [15] Kevin Clark, Paul Vicol, Kevin Swersky, and David J Fleet. 2023. Directly finetuning diffusion models on differentiable rewards. arXiv preprint arXiv:2309.17400

(2023).

- [16] Francis J Coleman. 1966. A phenomenology of aesthetic reasoning. The Journal of Aesthetics and Art Criticism 25, 2 (1966), 197–203.
- [17] Yingying Deng, Fan Tang, Weiming Dong, Chongyang Ma, Xingjia Pan, Lei Wang, and Changsheng Xu. 2022. Stytr2: Image style transfer with transformers. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 11326–11336.
- [18] Ahmed Elgammal, Bingchen Liu, Mohamed Elhoseiny, and Marian Mazzone.

2017. Can: Creative adversarial networks, generating" art" by learning about styles and deviating from style norms. arXiv preprint arXiv:1706.07068 (2017).

- [19] Arpad E Elo and Sam Sloan. 1978. The rating of chessplayers: Past and present. (No Title) (1978).
- [20] Junyao Gao, Yanchen Liu, Yanan Sun, Yinhao Tang, Yanhong Zeng, Kai Chen, and Cairong Zhao. 2024. Styleshot: A snapshot on any style. arXiv preprint arXiv:2407.01414 (2024).
- [21] Leon A Gatys, Alexander S Ecker, and Matthias Bethge. 2016. Image style transfer using convolutional neural networks. In Proceedings of the IEEE conference on computer vision and pattern recognition. 2414–2423.
- [22] Tia Ghose. 2017. This AI Helps You Paint Like Van Gogh. https://www.livescience. com/60536-ai-helps-you-paint-like-van-gogh.html. Live Science (sep 2017). Accessed: 2025-04-09.
- [23] James Gips and George Stiny. 1975. An investigation of algorithmic aesthetics. Leonardo 8, 3 (1975), 213–220.
- [24] Ernst Hans Gombrich and EH Gombrich. 1995. The story of art. Vol. 12. Phaidon London.
- [25] Talita Groenendijk, Andrea Kárpáti, and Folkert Haanstra. 2020. Self-Assessment in art education through a visual rubric. International Journal of Art & Design Education 39, 1 (2020), 153–175.
- [26] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin

Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. 2025. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv

preprint arXiv:2501.12948 (2025).

- [27] Ziyu Guo,Renrui Zhang, Chengzhuo Tong, Zhizheng Zhao, Peng Gao, Hongsheng Li, and Pheng-Ann Heng. 2025. Can We Generate Images with CoT? Let’s Verify and Reinforce Image Generation Step by Step. arXiv preprint arXiv:2501.13926

(2025).

- [28] Shibo Hao, Sainbayar Sukhbaatar, DiJia Su, Xian Li, Zhiting Hu, Jason Weston, and Yuandong Tian. 2024. Training large language models to reason in a continuous latent space. arXiv preprint arXiv:2412.06769 (2024).
- [29] Nicholas A Heard and Patrick Rubin-Delanchy. 2018. Choosing between methods of combining-values. Biometrika 105, 1 (2018), 239–246.
- [30] Georg Wilhelm Friedrich Hegel and GWF Hegel. 1998. Aesthetics: Lectures on fine art. Vol. 1. Oxford University Press.
- [31] Simon Hentschel, Konstantin Kobs, and Andreas Hotho. 2022. CLIP knows image aesthetics. Frontiers in Artificial Intelligence 5 (2022), 976235.
- [32] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. 2017. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems 30

(2017).

- [33] Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, et al. 2025. A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions. ACM Transactions on Information Systems 43, 2 (2025), 1–55.
- [34] Nisha Huang, Yuxin Zhang, Fan Tang, Chongyang Ma, Haibin Huang, Weiming Dong, and Changsheng Xu. 2024. Diffstyler: Controllable dual diffusion for text-driven image stylization. IEEE Transactions on Neural Networks and Learning Systems (2024).
- [35] Xun Huang and Serge Belongie. 2017. Arbitrary style transfer in real-time with adaptive instance normalization. In Proceedings of the IEEE international conference on computer vision. 1501–1510.
- [36] Yipo Huang, Xiangfei Sheng, Zhichao Yang, Quan Yuan, Zhichao Duan, Pengfei Chen, Leida Li, Weisi Lin, and Guangming Shi. 2024. Aesexpert: Towards multimodality foundation model for image aesthetics perception. In Proceedings of the 32nd ACM International Conference on Multimedia. 5911–5920.
- [37] Jessica Hullman, Ari Holtzman, and Andrew Gelman. 2023. Artificial intelligence and aesthetic judgment. arXiv preprint arXiv:2309.12338 (2023).
- [38] David Hume. 1757. Of the Standard of Taste. Essays, Moral, Political, and Literary

(1757). https://oll.libertyfund.org/titles/hume-essays-moral-political-literary-lfed

- [39] Eleftherios Ioannou and Steve Maddock. 2024. Evaluation in Neural Style Transfer: A Review. In Computer Graphics Forum. Wiley Online Library, e15165.
- [40] Ayse Ilkay Isik and Edward A Vessel. 2021. From visual perception to aesthetic appeal: Brain responses to aesthetically appealing natural landscape movies. Frontiers in Human Neuroscience 15 (2021), 676032.
- [41] Ruixiang Jiang and Changwen Chen. 2024. Artist: Aesthetically Controllable Text-Driven Stylization without Training. arXiv preprint arXiv:2407.15842 (2024).
- [42] Ruixiang Jiang, Can Wang, Jingbo Zhang, Menglei Chai, Mingming He, Dongdong Chen, and Jing Liao. 2023. Avatarcraft: Transforming text into neural human avatars with parameterized shape and pose control. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 14371–14382.
- [43] Dhiraj Joshi, Ritendra Datta, Elena Fedorovskaya, Quang-Tuan Luong, James Z Wang, Jia Li, and Jiebo Luo. 2011. Aesthetics and emotions in images. IEEE Signal Processing Magazine 28, 5 (2011), 94–115.
- [44] Immanuel Kant. 1950. Prolegomena to any future metaphysics (1783). Ed. Lewis White Beck. Indianapolis: Bobbs-Merrill 950 (1950).
- [45] Immanuel Kant. 2024. Critique of judgment. Vol. 10. Minerva Heritage Press.
- [46] Junjie Ke, Keren Ye, Jiahui Yu, Yonghui Wu, Peyman Milanfar, and Feng Yang.

2023. Vila: Learning image aesthetics from user comments with vision-language pretraining. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 10041–10051.

- [47] Diana Kim, Ahmed Elgammal, and Marian Mazzone. 2022. Formal analysis of art: Proxy learning of visual concepts from style through language models. arXiv preprint arXiv:2201.01819 (2022).
- [48] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al.

2023. Segment anything. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 4015–4026.

- [49] Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, and Omer Levy. 2023. Pick-a-pic: An open dataset of user preferences for text-toimage generation. Advances in Neural Information Processing Systems 36 (2023), 36652–36663.
- [50] Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. 2022. Large language models are zero-shot reasoners. Advances in neural information processing systems 35 (2022), 22199–22213.
- [51] HymanG Landau. 1951. On dominance relations and the structure of animal societies: I. Effect of inherent characteristics. The bulletin of mathematical biophysics 13, 1 (1951), 1–19.

- [52] Daiqing Li, Aleks Kamko, Ehsan Akhgari, Ali Sabet, Linmiao Xu, and Suhail Doshi.

2024. Playground v2. 5: Three insights towards enhancing aesthetic quality in text-to-image generation. arXiv preprint arXiv:2402.17245 (2024).

- [53] Leida Li, Yipo Huang, Jinjian Wu, Yuzhe Yang, Yaqian Li, Yandong Guo, and Guangming Shi. 2023. Theme-aware visual attribute reasoning for image aesthetics assessment. IEEE Transactions on Circuits and Systems for Video Technology 33, 9 (2023), 4798–4811.
- [54] Yanshu Li, Hongyang He, Yi Cao, Qisen Cheng, Xiang Fu, and Ruixiang Tang.

2025. M2iv: Towards efficient and fine-grained multimodal in-context learning in large vision-language models. arXiv preprint arXiv:2504.04633 (2025).

- [55] Yanshu Li, Tian Yun, Jianjiang Yang, Pinyuan Feng, Jinfa Huang, and Ruixiang Tang. 2025. TACO: Enhancing Multimodal In-context Learning via Task MappingGuided Sequence Configuration. arXiv preprint arXiv:2505.17098 (2025).
- [56] Peiyuan Liao, Xiuyu Li, Xihui Liu, and Kurt Keutzer. 2022. The artbench dataset: Benchmarking generative models with artworks. arXiv preprint arXiv:2206.11404

(2022).

- [57] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. 2014. Microsoft coco: Common objects in context. In Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13. Springer, 740– 755.
- [58] Annukka K Lindell and Julia Mueller. 2011. Can science account for taste? Psychological insights into art appreciation. Journal of Cognitive Psychology 23, 4 (2011), 453–475.
- [59] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le.

2022. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747

(2022).

- [60] Ryan Liu, Jiayi Geng, Addison J Wu, Ilia Sucholutsky, Tania Lombrozo, and Thomas L Griffiths. 2024. Mind your step (by step): Chain-of-thought can reduce performance on tasks where thinking makes humans worse. arXiv preprint arXiv:2410.21333 (2024).
- [61] Hongxu Ma, Chenbo Zhang, Lu Zhang, Jiaogen Zhou, Jihong Guan, and Shuigeng Zhou. 2025. Fine-Grained Zero-Shot Object Detection. arXiv preprint arXiv:2507.10358 (2025).
- [62] Daniel McFadden. 1972. Conditional logit analysis of qualitative choice behavior. Frontiers in Econometrics (1972).
- [63] R OpenAI. 2023. Gpt-4 technical report. arxiv 2303.08774. View in Article 2, 5

(2023).

- [64] Stefan A Ortlieb and Claus-Christian Carbon. 2019. A functional model of kitsch and art: linking aesthetic appreciation to the dynamics of social motivation. Frontiers in psychology 9 (2019), 2437.
- [65] Stephen E Palmer, Karen B Schloss, and Jonathan Sammartino. 2013. Visual aesthetics and human preference. Annual review of psychology 64, 1 (2013), 77–107.
- [66] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021. Learning transferable visual models from natural language supervision. In International conference on machine learning. PMLR, 8748–8763.
- [67] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. 2022. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 10684–10695.
- [68] Andrew Samo and Scott Highhouse. 2023. Artificial intelligence and art: Identifying the aesthetic judgment factors that distinguish human-and machinegenerated artwork. Psychology of Aesthetics, Creativity, and the Arts (2023).
- [69] Friedrich Wilhelm Joseph Schelling, Douglas W Stott, and David Simpson. 2008. The philosophy of art. University of Minnesota Press.
- [70] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. 2022. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in Neural Information Processing Systems 35 (2022), 25278–25294.
- [71] Frank Sibley. 2017. Aesthetic concepts. In Aesthetics. Routledge, 501–505.
- [72] Guy Sircello. 1968. Subjectivity and Justification in Aesthetic Judgments. The Journal of Aesthetics and Art Criticism 27, 1 (1968), 3–12.
- [73] Chaehan So. 2023. Measuring aesthetic preferences of neural style transfer: More precision with the two-alternative-forced-choice task. International Journal of Human–Computer Interaction 39, 4 (2023), 755–775.
- [74] Jiaming Song, Chenlin Meng, and Stefano Ermon. 2020. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502 (2020).

- [75] Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. 2023. Consistency models. (2023).
- [76] Charles Spearman. 1987. The proof and measurement of association between two things. The American journal of psychology 100, 3/4 (1987), 441–471.
- [77] Zayne Sprague, Fangcong Yin, Juan Diego Rodriguez, Dongwei Jiang, Manya Wadhwa, Prasann Singhal, Xinyu Zhao, Xi Ye, Kyle Mahowald, and Greg Durrett.

2024. To cot or not to cot? chain-of-thought helps mainly on math and symbolic reasoning. arXiv preprint arXiv:2409.12183 (2024).

- [78] Gail M Sullivan and Anthony R Artino Jr. 2013. Analyzing and interpreting data from Likert-type scales. Journal of graduate medical education 5, 4 (2013), 541–542.
- [79] Wladyslaw Tatarkiewicz. 1963. Objectivity and Subjectivity in the History of Aesthetics. Philosophy and phenomenological research 24, 2 (1963), 157–173.
- [80] Władysław Tatarkiewicz. 2012. A history of six ideas: An essay in aesthetics. Vol. 5. Springer Science & Business Media.
- [81] Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. 2023. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805 (2023).
- [82] Theo Van Leeuwen and Carey Jewitt. 2000. The handbook of visual analysis. Sage.
- [83] Can Wang, Ruixiang Jiang, Menglei Chai, Mingming He, Dongdong Chen, and Jing Liao. 2023. Nerf-art: Text-driven neural radiance fields stylization. IEEE Transactions on Visualization and Computer Graphics 30, 8 (2023), 4983–4996.
- [84] Haofan Wang, Matteo Spinelli, Qixun Wang, Xu Bai, Zekui Qin, and Anthony Chen. 2024. Instantstyle: Free lunch towards style-preserving in text-to-image generation. arXiv preprint arXiv:2404.02733 (2024).
- [85] Jianyi Wang, Kelvin CK Chan, and Chen Change Loy. 2023. Exploring clip for assessing the look and feel of images. In Proceedings of the AAAI conference on artificial intelligence, Vol. 37. 2555–2563.
- [86] Zijie J Wang, Evan Montoya, David Munechika, Haoyang Yang, Benjamin Hoover, and Duen Horng Chau. 2022. Diffusiondb: A large-scale prompt gallery dataset for text-to-image generative models. arXiv preprint arXiv:2210.14896 (2022).
- [87] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. 2022. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems 35

(2022), 24824–24837.

- [88] Wikiart. [n. d.]. WikiArt.org - Visual Art Encyclopedia. https://www.wikiart.org. Accessed: 2025-08-02.
- [89] Matthias Wright and Björn Ommer. 2022. Artfid: Quantitative evaluation of neural style transfer. In DAGM German Conference on Pattern Recognition. Springer, 560– 576.
- [90] Xun Wu, Shaohan Huang, Guolong Wang, Jing Xiong, and Furu Wei. 2024. Multimodal large language models make text-to-image generative models align better. Advances in Neural Information Processing Systems 37 (2024), 81287–81323.
- [91] Xiaoshi Wu, Keqiang Sun, Feng Zhu, Rui Zhao, and Hongsheng Li. 2023. Human preference score: Better aligning text-to-image models with human preference. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 2096– 2105.
- [92] Jianing Xu. 2024. Exploration of the Applications of Image-based AIGC in Art Education. Advances in Social Development and Education Research 1, 3 (2024), 60–65.
- [93] Junfeng Yang, Jing Fu, Wei Zhang, Wenzhi Cao, Limei Liu, and Han Peng. 2024. Moe-agiqa: Mixture-of-experts boosted visual perception-driven and semanticaware quality assessment for ai-generated images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 6395–6404.
- [94] D Younger. 1963. Minimum feedback arc sets for a directed graph. IEEE Transactions on Circuit Theory 10, 2 (1963), 238–245.
- [95] Zhengqing Yuan, Yunhong He, Kun Wang, Yanfang Ye, and Lichao Sun. 2023. ArtGPT-4: Towards artistic-understanding large vision-language models with enhanced adapter. arXiv preprint arXiv:2305.07490 (2023).
- [96] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. 2023. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 3836–3847.
- [97] Xinlu Zhang, Yujie Lu, Weizhi Wang, An Yan, Jun Yan, Lianke Qin, Heng Wang, Xifeng Yan, William Yang Wang, and Linda Ruth Petzold. 2023. Gpt-4v (ision) as a generalist evaluator for vision-language tasks. arXiv preprint arXiv:2311.01361

(2023).

- [98] Hanwei Zhu, Haoning Wu, Yixuan Li, Zicheng Zhang, Baoliang Chen, Lingyu Zhu, Yuming Fang, Guangtao Zhai, Weisi Lin, and Shiqi Wang. 2024. Adaptive image quality assessment via teaching large multimodal model to compare. arXiv preprint arXiv:2405.19298 (2024).

Algorithm 1: Sample a Connected Subgraph with Uniform Degree Distribution

Input: 𝐺 = (𝑉,𝐸) ; // Complete graph

𝑛 ; // Number of edges, |𝑉 | − 1 ≤ 𝑛 ≤ |𝑉 |(|𝑉2 |−1) RNG ; // Random Number Generator

Output: 𝐸′ ⊆ 𝐸 ; // Subsampled edge set forming a connected subgraph

- 1. Validate Inputs: if |𝑉 | < 1 or 𝑛 < |𝑉 | − 1 or 𝑛 > |𝑉 |(|𝑉2 |−1) then

Error: Invalid input parameters.;

- 2. Generate Spanning Tree: 𝐸𝑇 ← Kruskal’s MST(𝐺, RNG);
- 3. Initialize Subgraph and Degrees:

𝐸′ ← 𝐸𝑇 ;

Initialize 𝑑(𝑣) = 0, ∀𝑣 ∈ 𝑉; foreach 𝑒 = (𝑢, 𝑣) ∈ 𝐸𝑇 do

𝑑(𝑢) ← 𝑑(𝑢) + 1; 𝑑(𝑣) ← 𝑑(𝑣) + 1;

- 4. Add Remaining Edges:

𝑚 ← 𝑛 − |𝐸𝑇 |; while𝑚 > 0 do

- a. Identify Best Candidate Edges:

𝐵𝑒𝑠𝑡𝐸𝑑𝑔𝑒𝑠 ← 𝑒 ∈ 𝐸 \ 𝐸′ minmax{𝑑(𝑢) + 1,𝑑(𝑣) + 1}

𝐵𝑒𝑠𝑡𝐸𝑑𝑔𝑒𝑠 ← 𝑒 ∈ 𝐵𝑒𝑠𝑡𝐸𝑑𝑔𝑒𝑠 min∑︁{𝑑(𝑢) + 1,𝑑(𝑣) + 1}

- b. Select and Add an Edge: 𝑒∗ ← Randomly select from 𝐵𝑒𝑠𝑡𝐸𝑑𝑔𝑒𝑠 using RNG;

𝐸′ ← 𝐸′ ∪ {𝑒∗};

- c. Update Degrees and Counter: foreach 𝑣 ∈ 𝑒∗ do

𝑑(𝑣) ← 𝑑(𝑣) + 1; 𝑚 ← 𝑚 − 1;

return 𝐸′;

#### Appendix A Additional Experiment Details A.1 Human Aesthetic Judgment Collection

[Figure 315]

- Figure 7: UI for 2AFC annotation. We present user with the source image (top), 2AFC (middle) and style prompt (bottom). The user is required to choose the preferred one by clicking on the “left" or “right" button.

User Interface For the main judgment collection in Sec. 3.2, we develop a web-based application to collect responses from humanexpert annotators. We present users with the 2AFC question and reference content and style. The content is presented as a reference image, while we present the style as a prompt. The name of the stylization model is tracked but not visible to the user during annotation. The screenshot of the user interface (UI) can be found in Fig. 7.

For the hallucination labeling, we also designed a UI, visualized in Fig. 8. We record the user’s response during the 2AFC annotation and present them with the misaligned comparisons. The user shall read the MLLM’s response (the prompting method and the model name is not visible) and determine the hallucination issue from the four aspects. The image on left will be enlarged on mouse hover.

The annotation tools will be open-sourced to benefit future research.

#### A.2 The Sampling algorithm

We formally describe the proposed degree-uniform sub-graph sampling algorithm in Algo. 1.

#### A.3 Detail on FineArtBench

We construct FineArtBenchbyharvesting from existing open-source datasets with the help of MLLMs.

For the Content category, 50% of the content images are generated using the Ideogram-v1 text-to-image (T2I) diffusion model with diverse prompts produced by GPT-4. The remaining 50% are randomly sampled from the SA-1B [48] and MS-COCO [57] datasets, with captions generated by Gemini-v1.5 pro [81]. Images from SA1B are downsampled by a factor of two, while those from MS-COCO retain their original resolution. Overall, all images have an average height and width of 895.7 and 811.9 pixels, respectively. The 5th percentiles for height and width are 480 and 427 pixels, respectively, and the 95th percentiles for both dimensions are 1248 pixels.

To synthesize fine-grained attribute annotations for the content images and their associated prompts, we first employ Gemini v1.5pro to automate the annotation process based on a predefined attribute set. The Style subset of FineArtBench is derived from WikiArt [88] and DiffusionDB [86]. The process involves two main steps:

- (1) WikiArt Processing: We extract keywords from WikiArt, focusing on specifications such as art movement, artist, and genre. These base keywords are then expanded and combined using GPT-4 [63] to create a diverse set of style descriptors.
- (2) DiffusionDB Processing: Given the high noise level in DiffusionDB annotations, we utilize an MLLM to preprocess the text prompts within the dataset. Specifically, we extract the style descriptions from each prompt and merge those with similar style specifications. Subsequently, we subsample from these processed style prompts to ensure quality and diversity.

For both the WikiArt and DiffusionDB prompts, style reference images are generated using StableDiffusion-v2 [67], with all images standardized to a resolution of 512×512 pixels. Similar to content

[Figure 316]

- Figure 8: UI for hallucination annotation. Users are presented with the original 2AFC task and the response of MLLM, when the answer are not aligned. User shall drag discrete-valued slider to determine the hallucination strength from four perspectives.

annotations, we employ MLLMs to annotate the attributes of style prompts.

At least two human annotators manually validate all annotations to ensure quality. The actual number of sampled content and style images is summarized in Table 7. We will open-source FineArtBench to facilitate reproducibility and support future research endeavors.

- Table 7: Content and Style Sources FineArtBench is built from diverse sources to eliminate bias.

Content Style Source Generated MS-COCO SA-1B WikiArt DiffusionDB Number 500 250 250 764 236

#### A.4 Prompt Design

We summarize the full prompt of base prompt, zero-shot CoT prompting and the proposed ArtCoT prompt in Tab. 8. We did not attempt to optimize the prompt design of ArtCoT as it just act as a strong baseline method to identify the key issue and inspire future works.

#### Appendix B Further Discussions B.1 Limitation

Despite the progress we made in this paper, there still exists a wide range of possible venue for future study. We would like to provide a more detailed discussion of them here.

Implementation. We mainly test our experiment with the publicly available and mainstream MLLMs. First, the main objective of this paper is to understand how CoT can be utilized for aesthetic reasoning. This reasoning is in zero-shot and not designed to be model-specific. As evidenced in our experiment, the performance gain and hallucination suppression are already consistent and statistically significant. Second, due to the computational efficiency of MLLMs’ inference, we prototyped with small-scale models like LLaVa and LLaVa-Next. In practice, we find these small-scale models struggle with instruction following. Future work could investigate the performance of ArtCoT with larger, more capable open-source MLLMs.

Art Styles. This paper primarily deals with non-conceptual art, where aesthetic quality is more closely related to visual features (though not solely based on them). This is a key assumption of ArtCoT and FineArtBench, which cover much of mainstream and fine art. However, we acknowledge that avant-garde styles, particularly conceptual art, present different challenges. The value of conceptual art relies heavily on interpretation, which may be conveyed through various media and is often more subjective. A

##### Table 8: Template for different prompting methods. [STYLE] stands for placeholder for the style prompt, and [IMAGE] stands for placeholder for image tokens.

ArtCoT CS Analyzer Art Critic Summarizer

Base Prompt Zero-Shot CoT

‘[IMAGE]‘ You are an expert in fine art. A source image (top) and two different stylized images (bottom) in the style of ‘[STYLE]‘ are presented to you. Consider both the content and style, which stylized image is better in terms of overall aesthetic quality as an artwork? Return your decision in a Python Dict, [’winner’:int]. ‘0‘ means the left is better while ‘1‘ means the right is better. Do not answer any other things.

‘[IMAGE]‘ {"request": "You are an expert in fine art. A source image (top) and two different stylized images (bottom) in the style of ‘[STYLE]‘ are presented to you. Consider both the content preservation and style fidelity, which stylized image is better in terms of overall aesthetic quality as an artwork?". Return the reason and your decision in short in format of a Python Dict ’thinking’:str, ’winner’:int. ‘0‘ means the left is better while ‘1‘ means the right is better.", "response": "{’thinking’: ’ Let’s’ think step by step,

‘[IMAGE]‘ You are an expert in fine art. A source image (top) Two stylized images (bottom left and bottom right) in the style of ‘[STYLE]‘ are presented to you. Compare the content preservation and style fidelity of the two images, which one is better. Return your answer in a Python Dict, [’style_reason’:str, ’content_reason’:str, ’style_winner’:int, ’content_winner’:int]. ‘0‘ means the left is better while ‘1‘ means the right is better. Do not include any other string in your response.

‘[IMAGE]‘ Take a closer look at the two stylized images at the bottom in the style of ‘[STYLE]‘. As an expert in art, do you agree with above analysis? Compare and consider the following questions. What visual features is essential for the style of ‘[STYLE]‘? Is the content at top wellpreserved in the specific art style? Is there any artifact, distortion or inharmonious color patterns in either painting? Return your answer in a Python Dict, [reflection’:str].

‘[IMAGE]‘ Now we summarize. Based on above analysis and reflection, which stylized image at the bottom is better in terms of overall aesthetic quality as an **painting of the original content (top) in another style**? Return your answer in a Python Dict, [’winner’:int]. ‘0‘ means the left is better while ‘1‘ means the right is better. Do not include any other string in your response.

limitation of our work is that ArtCoT, focusing on visual features and evidence-based reasoning, may not be directly applicable to evaluating conceptual art. Future research could explore incorporating contextual information and interpretive frameworks into the aesthetic evaluation process.

#### Appendix C Example Conversations

We present several qualitative comparisons of MLLM’s response generated using different prompts in the following pages. Specifically, we show the input images to Google Gemini and its raw output from different reasoning methods. We highlight the subjective

words (yellow background) and the unsubstantiated interpretation of zero-shot CoT. Compared with zero-shot CoT, the responses from ArtCoT are much more detailed and contain less hallucination.

#### Appendix D Samples of FineArtBench

Examples of content image and style prompt is in Fig. 9, Tab. 9, respectively

[Figure 317]

MM ’25, October 27–31, 2025, Dublin, Ireland Ruixiang Jiang and Chang Wen Chen

|Input|Content|Style: Traditional Japanese Sumi-e ink wash painting, featuring delicate gradients of black and grey on rice paper to emphasize simplicity and natural beauty.<br><br>[Figure 318]|2AFC|Human judgement: 1 (right > left)|
|---|---|---|---|---|
|BaselineReasoning<br><br>[Figure 319]|Base: {‘winner’: 0} (❌ misaligned)<br><br>[Figure 320]| | | |
| |Zero-Shot CoT: {‘thinking’: “The goal is to determine which of the two stylized images better captures the essence of a traditional Japanese Sumi-e ink wash painting while accurately representing the source image‘s content. The left image (0) attempts a desaturated, washed-out effect, but it lacks the characteristic brushstrokes and ink gradations that define Sumi-e. The right image (1), although somewhat simplistic, more successfully evokes the style of Sumi-e through its use of stark black lines and subtle grey shading. It captures the essence of the scene’s composition and objects with a suitable level of abstraction. The right image‘s use of negative space and its emphasis on line work are more consistent with the principles of Sumi-e. While neither perfectly replicates the medium, the right image comes closer to achieving the intended artistic effect.”, {‘winner’: 1}} (✅ aligned)<br><br>[Figure 321]| | | |
|ReasolningviaArtCoT(Ours)<br><br>[Figure 322]|CS-Analyzer: {'style_reason': 'The right image demonstrates a more authentic sumi-e style with its use of ink washes and brushstrokes. The left image attempts the style but lacks the fluidity and expressiveness of traditional sumi-e.', 'content_reason': 'Both images successfully preserve the content of the original image, showing the building and bicycles. However, the right image is slightly more stylized, resulting in a minor loss of detail in the bicycles.', 'style_winner': 1, 'content_winner': 0}| | | |
| |Art Critic:{<br><br>'reflection': "The prompt requests an evaluation of two stylized images at the bottom of the provided image, judging their adherence to the style of Traditional Japanese Sumi-e ink wash painting. Let's analyze this. Essential visual features of Traditional Japanese Sumi-e include:<br><br>* **Monochromatic palette:** Primarily using black ink (sumi) with varying shades of grey achieved through dilution and layering. This is crucial for the simplicity and atmospheric depth sought in the art form.<br>* **Subtle gradations:** Delicate transitions between light and dark are essential for creating a sense of movement, form, and depth. Harsh lines are avoided; soft, hazy edges are favored.<br>* **Emphasis on negative space:** 'Ma' or negative space is very important in Sumi-e, and it plays a vital role in balancing the composition and suggesting a sense of quiet contemplation.<br>* **Brushstrokes:** The visible strokes of the brush are often part of the aesthetic appeal, revealing the artist's hand and contributing to the dynamic quality of the work. They're often less about precise detail and more about suggestive forms.<br>* **Rice paper:** The texture and absorbency of rice paper are integral to Sumi-e's characteristic look, as the ink bleeds and softens on this type of paper. Regarding the bottom images:<br><br><br>The image on the left (sepia-toned photograph) attempts a stylistic transformation of the original photo to mimic a Sumi-e feel. But, it fails to fully achieve the style as it lacks the essential brushstrokes and true 'sumi' ink gradients. The desaturation creates a grayscale, but isn’t the nuanced layering of ink that defines Sumi-e. It maintains too much photorealistic detail, preventing it from conveying the intended simplicity.<br><br>The image on the right (the actual attempt at Sumi-e) fares better in capturing the spirit of Sumi-e. It utilizes a monochromatic palette and shows some attempt at suggesting form and composition through varied shades of grey and black. It demonstrates the essential use of negative space. However, the brushstrokes lack fluidity and subtle gradations common in authentic Sumi-e. It feels rather stiff and lacks the organic movement one would expect. The detail in the bicycles also seems somewhat out of scale and not in harmony with the rest of the painting, causing a minor level of dissonance.<br><br>In summary: While the image on the right captures some aspects of the Sumi-e style, neither image successfully preserves the content of the original photograph in a way that harmoniously integrates with the authentic aesthetic of Sumi-e painting. The 'sumie' style attempted in the right image is a simplified and rudimentary interpretation; it lacks the technical sophistication and sensitivity to ink and brushwork present in master works. There is some inharmoniousness in the scale and detail in the right image, which distracts from the intended simplicity."| | | |
| |Summarizer: {‘winner’: 1} (✅ aligned)<br><br>[Figure 323]| | | |

InputneReasoning

[Figure 324]

Multimodal LLMs Can Reason about Aesthetics in Zero-Shot MM ’25, October 27–31, 2025, Dublin, Ireland

|Input|Content|Style: Art by Lois van Baarle and Loish and Ross Tran and RossDraws and Sam Yang and SamDoesArts and Artgerm and Saruei and Disney, digital art, highly detailed, intricate, sharp focus, trending on ArtStation HQ.<br><br>[Figure 325]|2AFC|Human judgement: 0 (left > right)|
|---|---|---|---|---|
|BaselineReasoning<br><br>[Figure 326]|Base: {‘winner’: 0} (✅ aligned)<br><br>[Figure 327]| | | |
| |Zero-Shot CoT: {‘thinking’: ‘Looking at the bottom images. Both stylized images depart significantly from the source image’s realism. The left image (0) attempts to maintain the scene‘s structure but uses an overly saturated, almost psychedelic color palette that feels jarring and lacks subtlety. The right image (1), while abstract, displays a more cohesive and aesthetically pleasing color harmony. The teal and peach tones work together well, creating a sense of depth and mood despite the lack of literal representation. The right image demonstrates better artistic interpretation and skill in its execution, even if less directly tied to the original.’, ‘winner’: 1} (❌ misaligned)<br><br>[Figure 328]| | | |
|ReasolningviaArtCoT(Ours)<br><br>[Figure 329]|CS-Analyzer: {'style_reason': 'The image on the right better captures the vibrant, stylized aesthetic reminiscent of the specified artists. It demonstrates a stronger sense of movement and fluidity, aligning more closely with the digital art styles mentioned.', 'content_reason': 'The image on the left preserves the basic elements of the original scene (mountains, water, sunset) more effectively, albeit with a heavily stylized color palette. The right image retains less of the original composition.', 'style_winner': 1, 'content_winner': 0}| | | |
| |Art Critic: { 'reflection': """ "The prompt requests an analysis of two stylized images at the bottom, comparing them to the style of several artists and Disney, emphasizing digital art, high detail, intricacy, sharp focus, and popularity on ArtStation HQ.<br><br>Let's examine the visual features essential to the specified style. Artists like Lois van Baarle, Loish, Ross Tran, Sam Yang, Artgerm, and Saruei, alongside Disney's digital work, are known for several key features:<br><br>* **Highly detailed rendering:** This includes intricate textures, smooth gradients, and fine details in both environments and characters (though only environments are present here).<br>* **Dynamic lighting and color palettes:** Often vibrant and saturated, they employ dramatic lighting to create mood and highlight important areas.*<br>**Strong sense of composition:** They use leading lines, rule of thirds, and other compositional techniques to create visual impact.<br>* **Stylized realism:** Although not photorealistic, they aim for a believable depiction of form and light.<br><br>Looking at the bottom images, neither fully captures the essence of the referenced artistic styles. The top image serves as a good reference point for the subject matter - a mountain range reflected in a lake at sunset. The bottom two images are attempts to reproduce this in the requested style, but are less successful:<br><br>* **Image 1 (left):** This image shows a significant departure from the detail and realism. The colors are strongly saturated, but the gradients are not smooth or nuanced, and the rendering of the mountains and water lacks the fine detail and texture seen in the work of the listed artists. There are noticeable distortions and a lack of sharp focus.<br>* **Image 2 (right):** This image displays a more abstract interpretation. While it uses a vibrant color palette, it lacks the detailed rendering and careful composition found in the style guide. The forms are more suggestive than precise, and the ‘reflection’ is highly stylized and doesn’t maintain the integrity of the original reflection. The color shifts are abrupt and inharmonious in places.<br><br><br>In conclusion, while the bottom images attempt to capture the *vibrant color* and *general mood* of the source image and the requested style, they fail to achieve the *high level of detail*, *sharp focus*, *intricate textures*, and *realistic rendering* that define the styles of the mentioned artists and Disney. The content of the top image is not well-preserved in the specific art style due to lack of technical skill and rendering. There are significant artifacts, distortions, and inharmonious color patterns in both bottom paintings, particularly evident in image 1‘s gradient shifts and image 2’s abstract and unclear forms. }| | | |
| |Summarizer: {‘winner’: 0} (✅ aligned)<br><br>[Figure 330]| | | |

Inputning

[Figure 331]

MM ’25, October 27–31, 2025, Dublin, Ireland Ruixiang Jiang and Chang Wen Chen

|Input|Content|Style:<br><br>painting by Georges Seurat, oil on canvas|2AFC|Human judgement: 1 (right > left)|
|---|---|---|---|---|
|BaselineReasoning<br><br>[Figure 332]|Base: {‘winner’: 0} (❌ misaligned)<br><br>[Figure 333]<br><br>[Figure 334]| | | |
| |Zero-Shot CoT: { ‘thinking’: “Both stylized images attempt to capture the essence of Seurat‘s pointillist technique, using small dots of color to create the overall image. However, the left image (0) more effectively mimics Seurat’s style and successfully translates the scene‘s content. The right image (1), while showing a pointillist approach, lacks the cohesiveness and nuanced color blending characteristic of Seurat’s work. The color palette in the left image is more vibrant and evocative of Seurat‘s use of color, while the right image’s colors appear somewhat muted and less expressive. The composition and perspective in the left image are also better preserved, resulting in a more recognizable and engaging scene. Therefore, the left image better captures the aesthetic qualities of Seurat‘s style.”, ’winner‘: 0 } (❌ misaligned)<br><br>[Figure 335]| | | |
|ReasoningviaArtCoT(Ours)<br><br>[Figure 336]|CS-Analyzer: {'style_reason': 'The image on the right more accurately captures the pointillist style of Seurat, with distinct dots of color creating the image, rather than the more impressionistic brushstrokes of the left image.', 'content_reason': 'The image on the left preserves the content of the source image more faithfully. The building and street scene are largely recognizable, even if stylized. The image on the right alters the content significantly.', 'style_winner': 1, 'content_winner': 0}| | | |
| |Art Critic: { 'reflection': """ The two stylized images at the bottom attempt to emulate the pointillist style of Georges Seurat, but with varying degrees of success. Let's analyze the key features:<br><br>**Essential Visual Features of Seurat's Style:**<br><br>* **Pointillism:** Small, distinct dots of pure color applied directly to the canvas, creating the illusion of blended color when viewed from a distance. The dots themselves are not blended on the canvas, but optically mix in the viewer's perception.<br>* **Optical Mixing:** The viewer's eye blends the juxtaposed dots of color, creating a sense of luminosity and vibrancy.<br>* **High level of detail:** Although using dots, the painting possesses a remarkably high level of detail. Seurat carefully positioned and sized the dots to achieve a wide spectrum of tonal variations.<br>* **Strong compositional structure:** Seurat's works are known for their strong compositional unity, creating a balanced and harmonious arrangement of shapes and forms. He often utilized classical approaches to composition.<br>* **Muted Palette with Bright Accents:** Though the palette tends towards muted, natural tones, Seurat uses bright, pure colors sparingly within the pointillist technique for highlights and focal points.<br>**Comparison of Stylized Images:** Neither image fully captures the essence of Seurat's style. The attempt is evident, with the use of small color patches resembling dots. However, several factors detract from a true Seurat interpretation:<br>* **Inconsistent Dot Size and Application:** The dots are not uniform in size or evenly spaced, a key characteristic of Seurat’s technique. This makes the optical mixing less effective.<br>* **Lack of Optical Mixing:** The color blending in both images appears more like digital processing than true optical mixing. The dots do not appear to visually blend in the same way as Seurat's paintings.<br>* **Color Palette:** The colors in both images often lack the subtlety and luminosity achieved through Seurat’s careful selection and juxtaposition of complementary colors.<br>* **Content Preservation:** The content of the top image is only partially well preserved. Both stylized images fail to render the subtle details of the original photograph with the level of detail associated with Seurat's work. They simplify and flatten the perspective. They present a more cartoonish feel than a finely wrought painting would.<br>* **Artifacts and Distortions:** The left image shows evident artifacts which appear to be hallmarks of the style transfer algorithms used, giving the image an unnatural look which Seurat's work avoids. Some areas lack detailed structure and look more like impressionistic washes rather than well-rendered dotwork. This causes disharmony in color and detail. The right image has some of the same digital flaws, though the degree is less jarring.<br>**Conclusion:**<br><br><br>While the intent to recreate Seurat's pointillist style is clear, both bottom images fall short of achieving a genuinely authentic reproduction. The technique is approximated, but the vital aspects of consistent dot application, true optical mixing, and the nuanced color palette are missing. Therefore, as an expert in art, I would not agree with a claim that the bottom images are a successful representation of Seurat's painting style. They are at best, digital interpretations influenced by the broader aesthetic of Post-Impressionism."""}| | | |
| |Summarizer: {‘winner’: 1} (✅ aligned)<br><br>[Figure 337]| | | |

InputBaselineReasoning

Multimodal LLMs Can Reason about Aesthetics in Zero-Shot MM ’25, October 27–31, 2025, Dublin, Ireland

[Figure 338]

|Input|Content|Style:<br><br>Pixel art retro style, digital pixel art<br><br>[Figure 339]|2AFC|Human judgement: 0 (left>right)|
|---|---|---|---|---|
|BaselineReasoning<br><br>[Figure 340]|Base: {‘winner’: 1} (❌ misaligned)<br><br>[Figure 341]| | | |
| |Zero-Shot CoT: { 'thinking': "Both stylized images utilize a pixel art approach, but differ significantly in their aesthetic quality and effectiveness in representing the source image. The left image (0) attempts to capture the scene's details and colors, but the result is a chaotic and visually overwhelming jumble of pixels. The color palette feels jarring and lacks harmony. The right image (1), while also using pixel art, employs a more restrained and effective approach. The color palette is simpler and more cohesive, creating a more unified and visually pleasing image. Though it simplifies the scene significantly, it maintains the essential elements and composition, resulting in a more successful and aesthetically pleasing pixel art interpretation. The right image demonstrates a better understanding of pixel art principles, focusing on effective use of color and simplification to create a stronger visual impact.", 'winner’: 1 } (❌ misaligned)<br><br>[Figure 342]| | | |
|ReasoningviaArtCoT(Ours)<br><br>[Figure 343]|CS-Analyzer: {'style_reason': 'The left image uses a more vibrant and varied color palette, better capturing the retro pixel art aesthetic. The right image is more muted and less evocative of classic pixel art.', 'content_reason': 'The left image maintains a better representation of the scene, including the woman walking on a path, the vegetation, and the general layout. The right image loses significant detail and almost completely obscures the content of the original image.', 'style_winner': 0, 'content_winner': 0}| | | |
| |Art Critic: {<br><br>'reflection': "The two bottom images are attempts at recreating the top image in a pixel art retro style. However, while they capture the general composition – a person walking on a path – the execution is not entirely successful in achieving a cohesive 'pixel art retro' aesthetic.<br><br>Essential visual features of pixel art retro style include: a limited color palette, clearly defined pixel blocks, a low resolution (resulting in blockiness), and a simplified representation of forms. While both bottom images utilize a limited palette and low resolution, they lack the clean, defined pixel blocks characteristic of the style. Instead, they display a dithering effect and color blending that softens the pixel boundaries, reducing the distinctive 'pixelated' look. The resulting images appear more like low-resolution digital paintings than pixel art. The content of the top image is only partially preserved. The basic scene (person on path, vegetation) is recognizable, but many details are lost. The colors in the pixel art versions, particularly in the first example, bear little resemblance to the originals. They are overly vibrant and seem almost arbitrary in their application, lacking the intentional and often limited color choices common to retro pixel art styles.<br><br>Artifacts are present. The first image exhibits a notable amount of color banding and irregular block distribution, suggesting the conversion process was imperfect. The second image suffers less from this but still lacks the crispness and clean linework typical of pixel art. Both show a loss of detail and a lack of precision in the rendering of the subject, and the path/surroundings. Inharmonious color patterns are evident, particularly in the exaggerated, often unnatural color combinations used in the first image. The second is better in terms of harmonious use of colour, despite lacking the other aspects of a good pixel art piece."}| | | |
| |Summarizer: {‘winner’: 0} (✅ aligned)<br><br>[Figure 344]| | | |

nput

[Figure 345]

MM ’25, October 27–31, 2025, Dublin, Ireland Ruixiang Jiang and Chang Wen Chen

|Input<br><br>[Figure 346]|Content|Style:<br><br>painting by Caravaggio, oil on canvas|2AFC|Human judgement: 0 (left>right)|
|---|---|---|---|---|
|BaselineReasoning<br><br>[Figure 347]|Base: {‘winner’: 0} (✅ misaligned)<br><br>[Figure 348]| | | |
| |Zero-Shot CoT: {'thinking’: ‘ The goal is to determine which stylized image has better aesthetic quality. The left image attempts a Caravaggio style with some success in the dark tones and chiaroscuro. However, the background is too simplistic and doesn’t possess the richness and complexity one would expect from a Caravaggio painting. The figures lack the strong anatomical detail and expressive power found in Caravaggio’s work. The right image, while stylistically less coherent, offers a more textured and evocative background. The figures, though somewhat indistinct, convey a sense of weight and presence through their posture. The overall atmosphere is more in keeping with the dramatic and psychological intensity found in Caravaggio’s works. Therefore, I favor the right image for its greater evocation of a Caravaggio-esque atmosphere.', 'winner': 1} (❌ misaligned)<br><br>[Figure 349]| | | |
|ReasoningviaArtCoT(Ours)<br><br>[Figure 350]|CS-Analyzer: {'style_reason': 'The image on the right better captures the chiaroscuro and dramatic use of light and shadow characteristic of Caravaggio, creating a more intense and evocative mood.', 'content_reason': 'Both images preserve the content reasonably well, showing four figures seated on a bench. The left image is slightly more accurate in representing the figures and their clothing.', 'style_winner': 1, 'content_winner': 0}| | | |
| |Art Critic: {<br><br>'reflection’: As an art expert specializing in the style of Caravaggio, I can offer an analysis of the stylized images. The prompt suggests that the two bottom images are rendered in the style of a 'painting by Caravaggio, oil on canvas'. Let's examine this claim.<br><br>**Essential Visual Features of Caravaggio's Style:** Caravaggio's paintings are characterized by several key features:<br><br>* **Tenebrism:** The dramatic use of light and shadow (chiaroscuro), with strong contrasts between brightly lit figures and dark backgrounds. This creates a sense of drama and intensity, focusing the viewer's attention on specific points within the composition.<br>* **Realism:** Although idealized, his figures possess a strong sense of realism, with visible textures, and believable anatomy, even if the overall context is theatrical.<br>* **Emotional Intensity:** His works often depict intense emotional states—suffering, joy, or anger—expressed through the figures' postures, expressions, and the overall composition.<br>* **Composition:** His compositions are often carefully arranged to guide the viewer's eye through the scene. This is achieved through the interplay of light, shadow, and the placement of the figures. He often uses diagonals and triangles to create dynamism.<br>* **Impasto:** In oil painting, impasto involves applying paint thickly, creating texture and enhancing the visual interest. Caravaggio's brushwork often shows impasto technique.<br>**Analysis of the Bottom Images:**<br><br><br>Neither of the bottom images convincingly capture the essence of Caravaggio's style. They lack the key tenebrism, realistic rendering of figures, and the emotional intensity that define his work. The stylistic elements are more reminiscent of other movements or digital art filters instead of Caravaggio's realism and use of light. The figures in both images appear somewhat distorted and lack the detailed anatomy and realistic proportions common in Caravaggio's painting. The color palette is far less rich, nuanced, and natural. It is not a representation of oil on canvas. The lighting in both images is not realistically presented; it does not possess the dramatic contrast that’s signature of his work.<br><br>**Content Preservation:** The content (people sitting on a bench) is somewhat preserved, but the artistic representation fails to capture the spirit of Caravaggio. It doesn't translate the simple scene into the emotional depth or painterly quality for which Caravaggio is famous.<br><br>**Artifacts and Distortions:***<br><br>**Bottom Left:** Shows significant artifacts. The background is a rather crudely painted-looking setting; the characters appear flat, unlike Caravaggio's detailed portrayals. There's a jarring bright blob in the background that’s completely anachronistic to the period.<br><br>* **Bottom Right:** The colors are muddy and lack the vibrancy of Caravaggio's palette. The figures are distorted and lack definition, with a painterly effect that is more akin to impressionism than to Caravaggio's meticulous detail and highly refined realism. The background texture looks unnatural and digital.<br>**Conclusion:** The assertion that the bottom images are in the style of Caravaggio is incorrect. While the subject matter is somewhat similar, the artistic execution completely misses the critical elements of Caravaggio's style. The images lack the required lighting, composition, realism, and emotional intensity to be considered even remotely similar to his works."}<br>| | | |
| |Summarizer: {‘winner’: 0} (✅ aligned)<br><br>[Figure 351]| | | |

InputBaselineReasoning

[Figure 352]

Multimodal LLMs Can Reason about Aesthetics in Zero-Shot MM ’25, October 27–31, 2025, Dublin, Ireland

|Input|Content|Style:<br><br>Anime-style illustration in the style of Hayao Miyazaki, featuring whimsical characters and vibrant landscapes<br><br>[Figure 353]|2AFC|Human judgement: 0 (left>right)|
|---|---|---|---|---|
|BaselineReasoning<br><br>[Figure 354]|Base: {‘winner’: 1} (❌ misaligned)<br><br>[Figure 355]| | | |
| |Zero-Shot CoT: {'thinking': "The left image attempts to render the bookstore in a style reminiscent of anime, but the execution is uneven. The lines are somewhat shaky, and the color palette lacks the vibrancy and harmony typically associated with high-quality anime. While it captures the essence of the bookstore's architecture, it doesn't achieve a cohesive or aesthetically pleasing anime style. The right image, however, is a more successful artistic interpretation. It effectively integrates the characters and landscape into a style that evokes the whimsical and vibrant aesthetic of Hayao Miyazaki's work. The color palette is bold and expressive, the character designs are charming, and the landscape adds a compelling narrative element. The overall composition is stronger and more aesthetically pleasing.", 'winner': 1} (❌ misaligned)<br><br>[Figure 356]| | | |
|ReasoningviaArtCoT(Ours)<br><br>[Figure 357]|CS-Analyzer: {‘style_reason’: ‘The right image demonstrates a more successful and vibrant application of the requested comic and manga style elements, with more dynamic color strokes and exaggerated features. The left image attempts a similar style but lacks the same level of visual energy and clarity.’, ‘content_reason’: ‘Both images retain the core content, a bakery display with pastries and customers, but the right image simplifies it significantly, losing some detail, while the left image attempts to preserve more.', 'style_winner': 1, 'content_winner': 0}| | | |
| |Art Critic: {<br><br>'reflection': "The prompt requests an evaluation of two stylized images (bottom row) based on whether they successfully emulate an 'Animestyle illustration in the style of Hayao Miyazaki, featuring whimsical characters and vibrant landscapes’. Let's analyze this assertion: **Essential Visual Features of the Target Style:** Hayao Miyazaki's style is characterized by several key visual elements. These include:<br><br>* **Whimsical Character Design:** Rounded, expressive characters often with slightly exaggerated features. Emphasis on emotional nuance in their poses and expressions.<br>* **Vibrant, often slightly surreal Landscapes:** Backdrops are lush and detailed, incorporating elements of fantasy and nature, sometimes with a touch of the ethereal or otherworldly.<br>* **Warm Color Palettes:** Miyazaki frequently employs rich, saturated colors, but typically with a harmonious balance to avoid jarring contrasts.<br>* **Detailed Backgrounds:** Environments are meticulously rendered, often showing a remarkable attention to detail.<br>**Linework and Style:** A distinct style of line art, often softer and less rigid than some other anime styles, is used, with a balance between clean lines and a painterly quality.<br>**Image 1 (Anime Style Bookstore):** This image shows a successful translation of the original bookstore image into an anime style. The color palette is relatively harmonious, and there's a clear stylistic choice which evokes anime. However, it falls short of capturing a true Miyazakiesque aesthetic. While vibrant, the colors lack the slightly muted warmth often seen in his work. The whimsical element isn't as strongly present; it simply looks like an anime depiction of a bookstore rather than a Miyazaki inspired scene.<br>**Image 2 (Anime Style Landscape):** This image more strongly reflects a whimsical, anime-style landscape. The color palette is certainly vibrant, leaning toward a more stylized and even slightly artificial look. The characters show a playful design. However, the landscape, while vibrant, lacks the lush detail typical of Miyazaki's backgrounds; it appears more flat in terms of depth and texture. The color choice, while intense, doesn't possess the warmth and harmony associated with Miyazaki's style, creating somewhat of an inharmonious color pattern. While the characters have the potential to be considered whimsical, the style lacks the distinctive linework and soft details observed in Miyazaki's works.<br><br><br>**Content Preservation:** The original image of the bookstore is relatively well-preserved in both versions. The structure and details are recognizable, although the anime rendition alters the visual feel. In the second image (landscape), the new elements (characters and landscapes) replace the original, so this isn’t applicable.<br><br>**Artifacts, Distortions, and Inharmonious Colors:** Both stylized images exhibit some artifacts from the stylization process. Image 1 shows minor inconsistencies in linework and color blending. Image 2 displays more pronounced color choices that, while vibrant, verge on jarring and inharmonious. The landscape feels flatter than it should in a true Miyazaki-style rendering. The artifacts are largely due to the artistic choices made during stylization, not necessarily flaws in the process.<br><br>**Conclusion:** While both images show a successful attempt at stylization, neither fully captures the essence of a ‘Miyazaki-style’ Anime illustration. They borrow some elements, like whimsical characters and vibrant palettes, but lack the specific details of linework, harmonious color blending, and depth of detail that are hallmarks of Miyazaki's art. The attempt is laudable, but not fully realized in the specific style requested."| | | |
| |Summarizer: {‘winner’: 0} (✅ aligned)<br><br>[Figure 358]| | | |

InputaselineReasoning

|Input|Content<br><br>[Figure 359]|Style:<br><br>Impressionist painting, oil on canvas|2AFC<br><br>[Figure 360]|Human judgement: 1 (right > left)|
|---|---|---|---|---|
|BaselineReasoning<br><br>[Figure 361]|Base: {‘winner’: 1} (✅ aligned)<br><br>[Figure 362]| | | |
| |Zero-Shot CoT: {‘thinking’: “The left image attempts to mimic an Impressionistic style through blurring and muted colors. However, the result lacks the vibrancy and expressive brushstrokes characteristic of Impressionism. The right image, while not perfectly replicating an oil on canvas texture, captures a more successful Impressionistic feel. The warm lighting, visible brushstrokes (though digital), and overall composition create a more cohesive and aesthetically pleasing artwork. The left image feels more like a poorly executed filter than a genuine artistic interpretation.”, ‘winner’: 1} (✅ aligned)<br><br>[Figure 363]| | | |
|ReasolningviaArtCoT(Ours)<br><br>[Figure 364]|CS-Analyzer: {'style_reason': 'The right image demonstrates a more successful emulation of Impressionist techniques, particularly in its use of color and light. The brushstrokes are more varied and expressive, creating a sense of movement and atmosphere reminiscent of Impressionist masters. The left image, while attempting a similar style, results in a more muddy and less defined application of paint.', 'content_reason': 'Both images retain the essential components of the source image - the buildings, palm trees, walkway, and kiosks. However, the right image better preserves the overall composition and spatial relationships of these elements. The left image, in its attempt to mimic the Impressionist style, sacrifices some clarity and precision in depicting the scene.', 'style_winner': 1, 'content_winner': 1}| | | |
| |Art Critic: {<br><br>'reflection': "The two stylized images at the bottom attempt to emulate the Impressionist style of oil on canvas, but with varying degrees of success. Let's analyze based on key Impressionist features:<br><br>**Essential Visual Features of Impressionism:** Impressionism is characterized by visible brushstrokes, a focus on light and its effects on color, capturing the fleeting moment, and often depicting everyday scenes. The use of pure, unmixed colors placed side-by-side to create optical mixing is also key. The style prioritizes capturing the sensation of a scene rather than perfect representation.<br><br>**Content Preservation:** The content from the top image is largely preserved in both stylized versions. The main architectural elements (buildings, palm trees), the walkway, and even some smaller details like the kiosks are recognizable. However, the degree of detail varies.<br><br>**Image 1 (left bottom):** This image shows a more muted color palette and somewhat blurry rendering. The brushstrokes are less defined and more like a soft smudge than the characteristic short, visible strokes of Impressionism. While attempting to capture the 'impression' of the scene, it lacks the vibrancy and distinct color contrasts expected in an Impressionist piece. There is a slight artifact of blurring, particularly noticeable in the trees and some building details. It might be described as more of an impressionistic filter applied to a photograph than a true artistic rendering in the style.<br>**Image 2 (right bottom):** This image fares better in capturing some essential elements of Impressionism. The use of color is richer and more varied, with visible brushstrokes, particularly in the sky and foliage, although these strokes are not uniformly 'short' as is typical. The handling of light and shadow is better demonstrated, creating a more evocative mood. However, the level of detail is still significantly simplified. Some color choices appear slightly inharmonious, specifically the unnatural, intense orange/red in the foliage, which detracts from the overall coherence. There's a visible shift in color temperature from the background to the foreground.<br><br><br>**Overall:** Neither image fully captures the essence of Impressionist oil painting. While both retain the scene's core elements, neither consistently showcases the short, distinct brushstrokes, vibrant, optically mixed color palette, nor the spontaneous feeling that defines the style. They are more digital interpretations of an Impressionist effect rather than genuine paintings done in the manner of Monet, Renoir, or Pissarro. The attempts are visible; however they fall short of a true artistic rendition in the Impressionistic style due to several artifacts and distortions of color and brushwork.}| | | |
| |Summarizer: {‘winner’: 1} (✅ aligned)<br><br>[Figure 365]| | | |

###### Inputing

[Figure 366]

Multimodal LLMs Can Reason about Aesthetics in Zero-Shot MM ’25, October 27–31, 2025, Dublin, Ireland

|Input<br><br>[Figure 367]|Content|Style:<br><br>Romanticism rich colors and dramatic landscapes inspired by Friedrich with expressive sky tones, oil on canvas|2AFC|Human judgement: 0 (left>right)|
|---|---|---|---|---|
|BaselineReasoning<br><br>[Figure 368]|Base: {‘winner’: 1} (❌ mis-aligned)<br><br>[Figure 369]| | | |
| |Zero-Shot CoT: {'thinking': "The left image attempts a romanticized portrait style, incorporating surreal elements with the background. However,<br><br>the execution feels somewhat flat and lacks the depth and emotional resonance of true Romanticism. The color palette is muted, and the overall effect is less dramatic than intended. The right image, on the other hand, is a more successful interpretation of the style. The fiery red figure against a dramatic, expressive sky evokes a strong emotional response, aligning with the core principles of Romanticism. The use of light and shadow, and the overall composition, creates a more powerful and aesthetically compelling artwork. The dramatic sky, reminiscent of Caspar David Friedrich's work, adds a significant layer of depth and meaning.", 'winner': 1} (❌ misaligned)<br><br>[Figure 370]| | | |
|ReasolningviaArtCoT(Ours)<br><br>[Figure 371]|CS-Analyzer: {'style_reason': 'The image on the right better captures the dramatic, expressive sky tones and landscape elements characteristic of Romanticism, particularly Friedrich\'s style. The clouds are more dynamic and the overall color palette is richer and more evocative.', 'content_reason': 'Both images preserve the figure's pose and clothing reasonably well. The image on the left distorts the face slightly, losing some of the likeness, while the image on the right is more faithful in its representation of the person.', 'style_winner': 1, 'content_winner': 1}| | | |
| |Art Critic: {<br><br>'reflection': "As an art expert, I would offer a nuanced assessment of the provided images and their relation to the specified artistic style. The prompt requests an analysis of the bottom two images as if they were painted in the style of Romanticism, specifically referencing Caspar David Friedrich's dramatic landscapes and expressive skies, using rich colors and oil on canvas. Let's examine this claim.<br><br>**Essential Visual Features of the Specified Style:** A painting truly embodying this style would necessitate several key features:<br><br>* **Dramatic Landscapes:** Friedrich's works often feature vast, awe-inspiring landscapes – often featuring mountains, forests, or seas – that evoke a sense of sublime power and the insignificance of humanity in the face of nature.<br>* **Expressive Skies:** The sky is frequently a dominant element, conveying emotional intensity through swirling clouds, dramatic lighting (often with a strong contrast between light and shadow), and intense color.<br>* **Rich Colors:** While not necessarily overly vibrant, the colors should be deeply saturated and evocative, contributing to the overall emotional impact of the piece. Think of deep blues, fiery oranges, and somber browns.<br>* **Oil on Canvas:** The medium is crucial for achieving the texture and depth of color associated with Romantic painting.<br>* **Sublime/Emotional Impact:** A sense of awe, mystery, or melancholy is central to Romanticism. The artwork should evoke strong emotions in the viewer.<br>* **Figures:** Often, figures are included but appear small within the vast landscape, emphasizing the immensity of nature.<br>**Content Preservation and Artistic Style:** The top image, the photograph of the woman in a business suit, is not well-preserved in the style of Romantic painting. The bottom two images attempt to emulate the *subject matter* (the woman in a suit) in a Romantic-inspired *style*, but with varying success.<br>* **Image on the Left:** This stylized portrait attempts to create a mood, using warm lighting and a somewhat surreal, dream-like background with repeating architectural forms. However, this is more akin to Surrealism and Symbolist movements than Romanticism. The style is not entirely in line with the intended features (dramatic landscape, expressive sky). The background does suggest an attempt at something grander, but it fails to evoke the vastness and awe associated with the Romantic landscape.<br>* **Image on the Right:** This image is much closer to achieving the desired style. It incorporates dramatic cloud formations in the sky—a characteristic feature of Friedrich's works—and uses a strong contrast in lighting (red-toned subject against a relatively brighter sky). It does, however, depart slightly in using the intensely saturated red tone of the figure; this is more reminiscent of Expressionism than the more subdued palette, generally found in Romanticism. It evokes a strong emotional impact.<br>**Artifacts, Distortions, and Color Inharmonious:** Neither image displays significant artifacts or distortions. Both are digitally created or enhanced, and the level of smooth blending, and the sharp detail are likely a result of digital tools rather than inherent characteristics of oil painting. In the left image, however, there is a slight stiffness and lack of fluidity in the way light plays on the figure's clothing that might be considered inharmonious with the background. In the right image, the jarring redness of the person’s face against the more naturalistic palette might feel dissonant to some viewers. The style is not entirely cohesive.<br><br><br>In summary, while the artist clearly intended to capture the essence of Romantic landscape painting in the right-hand image, it falls short of fully capturing the stylistic conventions and essential features of the described style. The left image employs elements of other stylistic movements more prominently."}| | | |
| |Summarizer: {‘winner’: 0} (✅ aligned)<br><br>[Figure 372]| | | |

InputBaselineReasoning

[Figure 373]

|Input|Content|Style:<br><br>painting by Jean Dubuffet<br><br>[Figure 374]|2AFC|Human judgement: 1 (right>left)|
|---|---|---|---|---|
|BaselineReasoning<br><br>[Figure 375]|Base: {‘winner’: 1} (✅ aligned)<br><br>[Figure 376]| | | |
| |Zero-Shot CoT: { 'thinking': "Both stylized images attempt to capture the essence of a bag of popcorn in the style of Jean Dubuffet. However, the right image (1) is more successful. The left image (0) uses Dubuffet-esque linework and color blocking but the composition feels less resolved and the rendering of the popcorn itself is unclear. It's more of an abstract representation than a stylized popcorn. The right image, while also abstract, better represents the visual texture and form of a bag of popcorn within its overall bold composition. The colors and lines are more integrated to convey a sense of depth and form, leading to a more satisfying aesthetic outcome. Therefore, the right image displays a stronger grasp of both Dubuffet's style and the subject matter.", 'winner’: 1} (✅ aligned)<br><br>[Figure 377]| | | |
|ReasoningviaArtCoT(Ours)<br><br>[Figure 378]|CS-Analyzer: {'style_reason': 'The right image demonstrates a better understanding of Dubuffet's texture and color palette application. It uses bolder outlines and a more varied color scheme, creating a more dynamic and expressive piece that is reminiscent of Dubuffet's work. The left image, while attempting a similar style, appears less refined and lacks the visual richness of the right.', 'content_reason': 'Both images successfully capture the essence of the popcorn bag, although the representation of popcorn is somewhat abstracted in both. The right image arguably does a slightly better job of capturing the form of the bag. The left image abstracts the content almost completely to the point where it becomes ambiguous.', 'style_winner': 1, 'content_winner': 1}| | | |
| |Art Critic: {<br><br>'reflection': \"The bottom two images attempt to emulate the style of Jean Dubuffet, but with varying degrees of success. Dubuffet's work is characterized by several key visual features:<br><br>**Primitivism and Naiveté:** A deliberate rejection of academic painting techniques in favor of a raw, childlike, or 'art brut' aesthetic. His work often features simplified forms and bold outlines.<br><br>* **Texture and Materiality:** Dubuffet was deeply interested in the physical properties of paint, often employing thick impasto or incorporating unconventional materials into his canvases. This creates a sense of tactile roughness.<br>* **Emphasis on Gesture and Process:** The visible brushstrokes or markings in his paintings are often prominent, highlighting the act of creation itself.<br>**Non-representational or Abstract elements:** While sometimes incorporating recognizable imagery, Dubuffet frequently utilized abstract forms and patterns to convey emotion or experience rather than literal representation. Analyzing the bottom images through this lens:<br>**Image 1 (left):** This image exhibits some Dubuffet-esque qualities. The bold black outlines and simplified shapes are reminiscent of his style. However, the color palette is relatively limited, and the texture is absent. It lacks the spontaneous and gestural quality present in most Dubuffet artworks. The overall effect is more graphic design than painting. The content (popcorn) is poorly preserved in the chosen style; the shape is too literal, and the depiction is somewhat stiff and lacks the organic feel of Dubuffet's work.<br>**Image 2 (right):** This image is more successful in capturing the essence of Dubuffet's style. The use of thick lines, varied textures through color application, and the somewhat disjointed nature of the composition create a closer parallel to Dubuffet's approach. The popcorn is more abstractly represented and integrated into the overall painting's composition. There are some distinct brushstroke-like patterns in the blues and oranges that add a gestural quality. Yet, even in this artwork, a deeper exploration of texture and materiality could elevate it closer to the genuine Dubuffet style. The colors, while bold, are somewhat inharmonious but could be interpreted as intentional within this specific style.<br><br><br>**Overall:** Neither image fully captures the nuances of Dubuffet's art. They offer approximations or suggestive references. To achieve a more authentic emulation, a greater emphasis on textured paint application, a richer color palette, and a more spontaneous and less precise application of the lines and shapes would be essential. }| | | |
| |Summarizer: {‘winner’: 1} (✅ aligned)<br><br>[Figure 379]| | | |

###### Inputning

[Figure 380]

Multimodal LLMs Can Reason about Aesthetics in Zero-Shot MM ’25, October 27–31, 2025, Dublin, Ireland

|Input|Content|Style:<br><br>Expressionist color fields<br><br>[Figure 381]|2AFC|Human judgement: 0 (left>right)|
|---|---|---|---|---|
|BaselineReasoning<br><br>[Figure 382]|Base: {‘winner’: 1} (❌ mis-aligned)<br><br>[Figure 383]| | | |
| |Zero-Shot CoT: { 'thinking': "The top image provides a charming, realistic scene of a toddler interacting with a puppy. Both bottom images attempt to capture the essence of this scene through the lens of Expressionist color fields. The left image (0) uses vibrant, almost thermalimaging-like colors to convey emotion and energy. However, the application of these colors feels somewhat arbitrary and lacks the cohesive visual structure of a successful color field painting. The right image (1), while less directly representative of the original scene, demonstrates a more sophisticated understanding of color field principles. The composition is stronger, with a clear sense of depth and visual weight distribution. The colors, although abstract, work together harmoniously to create a unified and aesthetically pleasing whole. While both interpretations utilize expressive color palettes, the right image displays a greater mastery of composition and color harmony, making it the stronger artwork.", 'winner': 1 } (❌ misaligned)<br><br>[Figure 384]| | | |
|ReasoningviaArtCoT(Ours)<br><br>[Figure 385]|CS-Analyzer: {'style_reason': 'The image on the right better captures the essence of Expressionist color fields with its bold, sweeping brushstrokes and simplified forms, creating a more evocative and emotionally resonant piece. The left image, while exhibiting some color field characteristics, appears more like a distorted photograph than a true expressionist work.', 'content_reason': 'The image on the left preserves more of the original image\\'s content, albeit in a highly stylized manner. We can still clearly identify the child and the puppy. The image on the right retains almost no recognizable content from the original photograph.', 'style_winner': 1, 'content_winner': 0}| | | |
| |Art Critic:{ 'reflection': "The two bottom images attempt to emulate Expressionist color field painting, but with varying degrees of success. Let's break down the essential visual features of Expressionist color fields and then assess each image:<br><br>**Essential Features of Expressionist Color Field Painting:**<br><br>**Emphasis on Color and Emotion:** Color is not merely descriptive but conveys raw emotion and psychological states. Vibrant, often nonnaturalistic hues are used to evoke feeling rather than represent reality. The application of the paint itself - thick impasto or thin washes -often communicates emotional intensity.<br><br>* **Abstraction:** Figurative elements are often minimized or entirely absent. The focus is on the interplay of color and form itself.<br>* **Large Scale and Flatness:** Color fields often span large canvases, creating a sense of immersion and immediacy. The paintings aim for a flat pictorial plane, minimizing depth and perspective.<br>* **Gestural Brushwork:** While some color field paintings are highly controlled, many reveal the artist's gestural brushstrokes, giving a sense of dynamism and spontaneity.<br>**Analysis of the Stylized Images:**<br>**Image 1 (Child and Dog):** This image employs a highly saturated, almost psychedelic color palette. The brushstrokes are visible, hinting at a gestural approach. However, the subject matter (the child and dog) is still clearly recognizable, which clashes with the abstract emphasis of true color field painting. The intense, almost random application of color, while vivid, lacks the cohesion and emotional unity often found in Expressionist color fields. The overall impression is one of digital manipulation rather than authentic painterly expression. There's a significant amount of distortion and arguably, inharmonious color combinations that lack a unifying emotional theme.<br>**Image 2 (Landscape):** This image is closer to the style, showcasing a more abstract landscape. The bold blocks of color recall the influence of artists like Barnett Newman, but the brushwork is comparatively more controlled and lacks the raw emotional energy often found in Expressionist color field works. The color palette is less jarring than image 1 and arguably more harmonious, but the overall effect is muted and lacks the expressive intensity of classic examples. While exhibiting some key elements, it feels more like a stylistic interpretation than a true embodiment of the Expressionist color field movement. There are no significant artifacts or distortions in this image.<br><br><br>**Conclusion:**Neither image fully captures the essence of Expressionist color field painting. Image 2 comes closer by virtue of its abstraction and more considered color palette, but both lack the profound emotional resonance and painterly presence found in the work of major exponents of the movement. The top image's content is not well-preserved; it's significantly altered and loses its original meaning. The attempts at stylization are interesting but ultimately fall short of effectively conveying the style's core principles. }| | | |
| |Summarizer: {‘winner’: 0} (✅ aligned)<br><br>[Figure 386]| | | |

InputelineReasoning

[Figure 387]

##### Figure 9: Random sample of content images from FineArtBench

Multimodal LLMs Can Reason about Aesthetics in Zero-Shot MM ’25, October 27–31, 2025, Dublin, Ireland

Table 9: Examples of style prompt

Randomly Sampled Style Prompts

Bauhaus geometric abstraction with primary colors and clean lines The Blue Boy by Thomas Gainsborough Alphonse Mucha, Vincent van Gogh, Egon Schiele-inspired artwork with expressive color palettes and dynamic brush strokes Cafe Terrace at Night by Vincent van Gogh painting by Rumiko Takahashi, manga ink on paper painting by Donato Giancola Symbolist mystical and allegorical imagery inspired by Klimt, oil on canvas with intricate patterns Low-poly 3D model, 3D rendering guido crepax painting epic, megadeth cover, aesthetically inspired by beksinski and dan mumford, trending on artstation, art by greg rutkowski, 8k Imagery rich in symbolic and allegorical content Surrealist dream-like elements reminiscent of Dalí Cubist fragmented forms, oil on canvas Girl with a Pearl Earring, oil on canvas, Baroque portrait renowned for its enigmatic expression and masterful lighting Las Meninas Thomas Gainsborough’s ’The Blue Boy’, a Rococo oil on canvas portrait known for its elegant attire and striking blue palette. salvador dali painting painting by Nanna Ditzel, mixed media The Scream painting by Pierre-Auguste Renoir, oil on canvas in the style of Honor C. Appleton painting by Grandma Moses Anime-style illustration in the style of Hayao Miyazaki, featuring whimsical characters and vibrant landscapes D&D, fantasy, elegant, pale, highly detailed, digital painting, artstation, concept art, illustration, art by alberto scorfano and james jean and jason chan painting by James Favaro Water Lilies, oil on canvas, Impressionist series capturing the serene beauty of Monet’s garden pond painting by Vektroid, digital art Minimalist line art by Saul Steinberg, ink on paper Dark fantasy, Warhammer, ArtStation painted by Zdzisław Beksiński and Wayne Barlowe fauvism style painting, oil on canvas, vibrant colors Organic surrealism Cinematic, dark scenes with film grain and deep, moody color tones painting by Yves Klein, oil on canvas Pop art colorful graphics vaporwave aesthetic, synthwave, digital painting, artstation, concept art, smooth, sharp focus, art by artgerm and greg rutkowski and alphonse mucha painting by Jean-Honoré Fragonard Kitsch art by Jeff Koons Digital hyperrealism with intricate textures by contemporary artists using lifelike color schemes painting by Moebius painting by James Turrell Georgia O’Keeffe Modernist A fusion of Van Gogh, Picasso, Cezanne, and David Hockney styles, oil on canvas with expressive brushstrokes and bold geometry cinematic, highly detailed, masterpiece by craig mullins, ruan jia, greg rutkowski, jakub rebelka, caravaggio, syd mead Impressionist painting with soft pastel hues and loose, visible brushstrokes inspired by Claude Monet Digital hyperrealism with intricate textures by contemporary artists, digital painting with stunning detail Superflat acrylic artwork by Takashi Murakami, blending traditional Japanese motifs with contemporary pop culture generative art, sci-fi, highly detailed, 3d octane render, high contrast, minimalistic by paul lehr and ralph mcquarrie Ophelia by John Everett Millais Neo-Expressionist raw brushstrokes and emotive color palette conveying intense feelings Imagery that evokes surreal, dreamlike scenes

