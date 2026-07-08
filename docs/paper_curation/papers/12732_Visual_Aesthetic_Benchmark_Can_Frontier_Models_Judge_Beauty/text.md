# arXiv:2605.12684v1[cs.CV]12May2026

BakeAI Technical Report

## Visual Aesthetic Benchmark: Can Frontier Models Judge Beauty?

###### Yichen Feng1,2,†* Yuetai Li2* Chunjiang Liu1,6,†* Yuanyuan Chen1 Fengqing Jiang2 Yue Huang5 Hang Hua7 Zhengqing Yuan5 Kaiyuan Zheng2 Luyao Niu2 Bhaskar Ramasubramanian2,8 Basel Alomair9 Xiangliang Zhang5 Misha Sra3 Zichen Chen1,3,4 Radha Poovendran2 Zhangchen Xu1,2

1Bake AI 2University of Washington 3University of California, Santa Barbara 4Stanford University 5University of Notre Dame 6Carnegie Mellon University 7MIT-IBM Watson AI Lab 8Western Washington University 9King Abdulaziz City for Science and Technology

∗Equal contribution. †Work done during internship at Bake AI. Corresponding author.

###### Abstract

Multimodal large language models (MLLMs) are now routinely deployed for visual understanding, generation, and curation. A substantial fraction of these applications require an explicit aesthetic judgment. Most existing solutions reduce this judgment to predicting a scalar score for a single image. We first ask whether such scores faithfully capture comparative preference: in a controlled study with eight expert annotators, score-derived rankings align poorly with the same annotators’ direct comparisons, while direct ranking yields substantially higher inter-annotator agreement on best- and worst-image labels. Motivated by this finding, we introduce the Visual Aesthetic Benchmark (VAB), which casts aesthetic evaluation as comparative selection over candidate sets with matched subject matter. VAB contains 400 tasks and 1,195 images across fine art, photography, and illustration, with labels derived from the consensus of 10 independent expert judges per task. Evaluating 20 frontier MLLMs and six dedicated visual-quality reward models, we find that the strongest system identifies both the best and the worst image correctly across three random permutations of the candidate order in only 26.5% of tasks, far below the 68.9% achieved by human experts. Fine-tuning a 35B-parameter model on 2,000 expert examples brings its accuracy close to that of a 397B-parameter open-weight model, suggesting that the comparative signal in VAB is transferable. Together, these results expose a clear and measurable gap between current multimodal models and expert aesthetic judgment, and VAB provides the first set-based, expert-grounded testbed on which that gap can be tracked and closed.

Code BakeLab/Visual-Aesthetic-Benchmark Dataset BakeLab/Visual-Aesthetic-Benchmark Website vab.bakelab.ai

###### 1 Introduction

Multimodal large language models (MLLMs) have become critical to a widening range of visual tasks such as image generation and curation (Liu et al., 2023; Team et al., 2023; Wang et al., 2024). Many of these applications require aesthetic judgment, asking models to compare visual works and identify differences in execution quality across composition, color harmony, technical execution, and cultural convention. This makes aesthetic judgment a demanding test of higher-order multimodal reasoning that is often omitted by existing visual benchmarks (Yu et al., 2023; Yue et al., 2024).

Existing image-aesthetics datasets and methods typically reduce aesthetic judgment to a mapping from an image to a scalar score (Murray et al., 2012; Kong et al., 2016; He et al., 2022; Lee and Kim, 2019; Hosu et al., 2019; Yi et al., 2023; He et al., 2023), an approach that we argue is poorly suited to the underlying decision being studied. Preference order is inferred from independent scores rather than measured through direct comparison; crowd-averaged labels mix expertise, taste, and annotation noise, so the aesthetic signal is difficult to isolate; and because images in such datasets often depict different subjects, a model can shortcut its way to high accuracy by exploiting content preferences instead of discriminating execution quality.

We first ask whether scoring faithfully captures comparative preference . Eight expert annotators evaluate the same images under both independent absolute scoring and direct comparative ranking, and the two

###### BakeAI 1 Introduction

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

Still Life Sketch

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

Calligraphy

Digital Art Concept Art

Manga Comic

[Figure 20]

Wildlife

|[Figure 21]<br><br>[Figure 22]|
|---|

|[Figure 23]<br><br>[Figure 24]|
|---|

|[Figure 25]<br><br>[Figure 26]|
|---|

|[Figure 27]<br><br>[Figure 28]|
|---|

|[Figure 29]<br><br>[Figure 30]|
|---|

|[Figure 31]<br><br>[Figure 32]|
|---|

|[Figure 33]<br><br>[Figure 34]|
|---|

|[Figure 35]<br><br>[Figure 36]|
|---|

|[Figure 37]<br><br>[Figure 38]|
|---|

|[Figure 39]<br><br>[Figure 40]|
|---|

|[Figure 41]<br><br>[Figure 42]|
|---|

|[Figure 43]<br><br>[Figure 44]|
|---|

|[Figure 45]<br><br>[Figure 46]|
|---|

|[Figure 47]<br><br>[Figure 48]|
|---|

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

Pixel Art Stylized 3D

Art

[Figure 57]

|[Figure 58]<br><br>[Figure 59]|
|---|

|[Figure 60]<br><br>[Figure 61]|
|---|

Comic Manga

|[Figure 62]<br><br>[Figure 63]|
|---|

|[Figure 64]<br><br>[Figure 65]|
|---|

[Figure 66]

[Figure 67]

[Figure 68]

Art

Digital

[Figure 69]

[Figure 70]

Concept

[Figure 71]

[Figure 72]

Quick Sketch

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

Ink & Wash

Stylized 3D

|[Figure 80]<br><br>[Figure 81]|
|---|

|[Figure 82]<br><br>[Figure 83]|
|---|

|[Figure 84]<br><br>[Figure 85]|
|---|

|[Figure 86]<br><br>[Figure 87]|
|---|

[Figure 88]

|[Figure 89]<br><br>[Figure 90]|
|---|

|[Figure 91]<br><br>[Figure 92]|
|---|

Art

Portrait

[Figure 93]

[Figure 94]

[Figure 95]

|[Figure 96]<br><br>[Figure 97]|
|---|

|[Figure 98]<br><br>[Figure 99]|
|---|

[Figure 100]

[Figure 101]

[Figure 102]

Pixel

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

|[Figure 111]<br><br>[Figure 112]|
|---|

[Figure 113]

[Figure 114]

Calligraphy

Architecture

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

Architecture

[Figure 119]

Quick Sketch

[Figure 120]

[Figure 121]

Food/ Product

Street/City

Visual Aesthetic Benchmark

[Figure 122]

[Figure 123]

|[Figure 124]<br><br>[Figure 125]|
|---|

|[Figure 126]<br><br>[Figure 127]|
|---|

|[Figure 128]<br><br>[Figure 129]|
|---|

[Figure 130]

|[Figure 131]<br><br>[Figure 132]|
|---|

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

Landscape Color

Landscape

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

Portrait Color

Portrait Sketch

[Figure 151]

Portrait

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

Landscape Color

[Figure 156]

[Figure 157]

|[Figure 158]<br><br>[Figure 159]|
|---|

|[Figure 160]<br><br>[Figure 161]|
|---|

|[Figure 162]<br><br>[Figure 163]|
|---|

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

Portrait Sketch

[Figure 168]

|[Figure 169]<br><br>[Figure 170]|
|---|

|[Figure 171]<br><br>[Figure 172]|
|---|

Street/City

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

Food/Product Night/Astro

[Figure 178]

[Figure 179]

[Figure 180]

Still ColorLife

|[Figure 181]<br><br>[Figure 182]|
|---|

|[Figure 183]<br><br>[Figure 184]|
|---|

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

Macro

Ink &

[Figure 190]

[Figure 191]

[Figure 192]

|[Figure 193]<br><br>[Figure 194]|
|---|

[Figure 195]

Sports

|[Figure 196]<br><br>[Figure 197]|
|---|

Night/Astro

Still SketchLife

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

Wash

PaintingChinese

Wildlife

[Figure 203]

|[Figure 204]<br><br>[Figure 205]|
|---|

[Figure 206]

[Figure 207]

|[Figure 208]<br><br>[Figure 209]|
|---|

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

Macro

Portrait Color

[Figure 218]

[Figure 219]

[Figure 220]

Landscape

Still Life Color

[Figure 221]

[Figure 222]

[Figure 223]

Sports

Chinese Painting

|[Figure 224]<br><br>[Figure 225]|
|---|

|[Figure 226]<br><br>[Figure 227]|
|---|

|[Figure 228]<br><br>[Figure 229]|
|---|

|[Figure 230]<br><br>[Figure 231]|
|---|

|[Figure 232]<br><br>[Figure 233]|
|---|

|[Figure 234]<br><br>[Figure 235]|
|---|

|[Figure 236]<br><br>[Figure 237]|
|---|

|[Figure 238]<br><br>[Figure 239]|
|---|

|[Figure 240]<br><br>[Figure 241]|
|---|

|[Figure 242]<br><br>[Figure 243]|
|---|

|[Figure 244]<br><br>[Figure 245]|
|---|

|[Figure 246]<br><br>[Figure 247]|
|---|

|[Figure 248]<br><br>[Figure 249]|
|---|

|[Figure 250]<br><br>[Figure 251]|
|---|

|[Figure 252]<br><br>[Figure 253]|
|---|

Figure 1: Overview of the VAB benchmark. VAB spans three visual domains — fine art, photography, and illustration — covering 24 topics across 400 evaluation tasks and 1,195 images. Each task presents a set of candidate

images sharing a common subject.

protocols disagree. Comparative ranking yields 42 percentage points higher inter-annotator agreement on best-image selection than score-derived rankings (Talebi and Milanfar, 2018; Ma et al., 2017; Wu et al., 2023; Zhou et al., 2024a). The two protocols therefore elicit different preference signals from the same annotators, and direct comparison captures the underlying judgment more faithfully than the pointwise scoring used in most existing pipelines.

Building on this finding, we introduce the Visual Aesthetic Benchmark (VAB), which formulates aesthetic judgment as a comparative selection problem over candidate sets with matched subject matter. VAB contains 400 tasks and 1,195 images across fine art, photography, and illustration, covering 24 topics (Figure 1). Each task presents two to six images sharing a common subject, and the model selects the best image and, under the stricter setting, also the worst image. Three design choices distinguish VAB from prior aesthetic benchmarks. Matched subject matter within each task isolates differences in execution from differences in content; each ground-truth label is the consensus of 10 independent expert judges, filtered through a dualthreshold criterion to remove ambiguous cases; and every task is evaluated under three independent random permutations of the candidate order. We report ap@1, the average accuracy across the three evaluations, and pass^3, which credits a task only when the model is correct in all three.

We benchmark 20 frontier MLLMs and six dedicated visual-quality reward models on VAB. The strongest system, Claude Sonnet 4.6, reaches 26.5% on TB-1 pass^3, 42.4 points below the human baseline of 68.9%. Models are also highly sensitive to candidate order, retaining only 34–65% of their ap@1 accuracy when required to be correct across all three permutations, and they degrade much faster than humans as the candidate set grows: a 7.1× drop from two-image to four-or-more-image tasks, against only 2.0× for human experts. The contributions of this paper are as follows.

- • A controlled human study with eight expert annotators establishes that direct comparative ranking is substantially more reliable than independent scoring for subjective aesthetic annotation, and motivates the design of the benchmark.

BakeAI 2 Human Annotation Study: Ranking vs. Scoring

- • VAB is, to our knowledge, the first set-based, expert-grounded, multi-domain aesthetic benchmark with permutation-robust evaluation, covering 400 tasks and 1,195 images across fine art, photography, and illustration.
- • A systematic evaluation of 20 frontier MLLMs and six dedicated reward models exposes a persistent 42.4-point gap to human experts on TB-1 pass^3 together with severe positional instability across permutations of the candidate order.
- • Fine-tuning on 2,000 expert examples brings a 35B-parameter model close to a 397B-parameter openweight model, indicating that expert comparative judgments transfer effectively through lightweight adaptation.

###### 2 Human Annotation Study: Ranking vs. Scoring

Existing aesthetic benchmarks rely on Scoring to collect ground truth labels (Murray et al., 2012; Kong et al., 2016; He et al., 2022; Cao et al., 2025). We conduct a controlled human study to explore the following research question regarding the annotation process:

###### Which protocol yields a more scientifically sound signal for data annotation?

- • Scoring: expert annotates each image independently on an absolute aesthetic scale.
- • Ranking: expert compares images and annotates their relative order.

Setup and Metrics. Eight expert annotators evaluate the same images under two protocols: (1) scoring assigns each image an independent score si,a ∈ [0,10]; (2) ranking orders images directly within groups of size n ∈ {2,3,4,5}. We construct 119 homogeneous-content tasks, in which all candidates share the same underlying content, and 107 heterogeneous-content tasks formed by regrouping images across source groups while preserving topic consistency. We compare the two protocols along three dimensions:

- • Fidelity (Kendall’s τ (Kendall, 1938)): we compute τ = (C − D)/(n2), where C and D are the numbers of concordant and discordant pairs between an annotator’s score-derived ranking σascore and the same annotator’s direct comparison. A low τ means the two protocols elicit different preference orderings from the same person. We additionally report the top-1 self-consistency (SC) rate: the fraction of annotator–task pairs for which σascore and the direct comparison select the same top image.
- • Reproducibility (accbest/worst): accbest is the proportion of tasks where a single image receives ≥ 5 of 8 best-image votes; accworst and accbest&worst are defined analogously. Higher accuracy means the protocol produces ground truth labels that different experts can independently reproduce.
- • Distinguishability (top-1 entropy (Shannon, 1948)): we measure H = − ∑i pi log2 pi, where pi is the fraction of annotators selecting image i as best. Lower entropy means annotators concentrate on a single top choice rather than spreading votes.

Full protocol details, notations, formal definitions, and statistical tests are in Appendix F.

Scoring does not faithfully preserve preference order. Table 1 shows that score-derived rankings often diverge from the direct comparisons given by the same annotator. The mean Kendall’s τ is 0.188 for homogeneous-content and 0.351 for heterogeneous-content tasks; the top-1 self-consistency rates are 45.5% and 52.0%, respectively. In other words, nearly half of the numeric score annotations fail to recover the order that the same annotator expresses under direct comparison.

Ranking yields more reproducible labels. Table 2 shows that ranking yields substantially higher interannotator agreement on best and worst labels. On homogeneous-content tasks, ranking improves accbest by 42.0 points, accworst by 37.0 points, and accbest&worst by 51.3 points (all p < 0.001, McNemar test (McNemar,

- Table 1: We report within-annotator agreement between expert’s score-derived rankings σascore and direct comparison from 8 annotators, measured by Kendall’s τ (↑) and top-1 self-consistency rate (SC rate, ↑), defined as the fraction of annotator–task pairs where the score-derived top-1 matches the direct comparison top-1. The results show that nearly 50% of the numeric score annotations do not reliably preserve the order that the annotator would express in direct comparison.

(a) Results on homogeneous-content tasks.

n N tasks Mean τ Std. τ SC rate

- 2 28 0.098 0.511 54.9%
- 3 38 0.274 0.328 49.7%
- 4 35 0.126 0.243 34.3%
- 5 18 0.267 0.243 43.6%

All 119 0.188 0.354 45.5%

(b) Results on heterogeneous-content tasks.

n N tasks Mean τ Std. τ SC rate

- 2 19 0.342 0.427 67.1%
- 3 37 0.286 0.300 48.0%
- 4 34 0.441 0.172 51.1%
- 5 17 0.324 0.256 45.6%

All 107 0.351 0.291 52.0%

- Table 2: We report inter-annotator consistency on best and worst decisions for 8 annotators, where a majority requires agreement from at least 5 of 8. Each row shows task-level accuracy acc (↑), improvement ∆, and the exact two-sided McNemar p-value (↓) (McNemar, 1947; Fagerland et al., 2013). The Significance level is marked as ∗∗p < 0.001. Results show that ranking is more likely to yield annotation labels that independent experts can reproduce.

Homogeneous-content (N = 119) Heterogeneous-content (N = 107) Metric Scoring Ranking ∆ p Scoring Ranking ∆ p accbest 52.9% 95.0% +42.0% <0.001∗∗ 65.4% 94.4% +29.0% <0.001∗∗

accworst 55.5% 92.4% +37.0% <0.001∗∗ 69.2% 93.5% +24.3% <0.001∗∗ accbest&worst 39.5% 90.8% +51.3% <0.001∗∗ 49.5% 90.7% +41.1% <0.001∗∗

1947)); the same pattern holds for heterogeneous-content tasks. Independent experts can therefore reproduce best/worst labels far more reliably under ranking than under scoring.

Ranking makes the top choice stand out more clearly. Table 3 measures how concentrated their topchoice preference is within each task. It shows that ranking produces lower entropy over top-1 selections (Shannon, 1948) (0.729 vs. 1.213) and a much higher rate of unanimous choices (10.1% vs. 0.8%; Wilcoxon p = 1.7 × 10−15 (Wilcoxon, 1945)). More generally, ranking concentrates votes on a single preferred image rather than spreading them across several plausible winners.

Interpretation. Our findings align with conclusions from two complementary disciplines. Aesthetically, modern aesthetics favors a relational view of beauty (Goldman, 1995; Levinson, 2005): people judge preference more reliably when alternatives are juxtaposed than when items are assessed in isolation. Cognitively, comparative judgment is a more primitive operation than absolute evaluation (Thurstone, 1927). Absolute scoring requires maintaining an internal reference scale and converting qualitative impressions into numbers, imposing heavier cognitive load and greater instability (Stewart et al., 2005; Parducci, 1965).

We therefore adopt ranking as the annotation protocol for VAB. Extended discussion is provided in Appendix F.2.

- 3 The VAB Benchmark

Building on the findings of Section 2, we construct the Visual Aesthetic Benchmark (VAB) to test whether frontier multimodal models can make expert-aligned aesthetic judgments under a ranking-based protocol. In contrast with prior evaluations that aggregate scalar ratings from crowd annotators (Murray et al., 2012; Kong et al., 2016; He et al., 2022), VAB casts aesthetic evaluation as comparative selection over candidate

Table 3: We report top-choice concentration in the homogeneous-content setting (119 tasks, 8 annotators), measured by entropy H (Shannon, 1948) and the rate of unanimous tasks (H = 0). The 95% confidence intervals use a normal approximation for mean H and a Wilson interval (Wilson, 1927) for the unanimous-task rate. A Wilcoxon signed-rank test (Wilcoxon, 1945) yields that the difference is statistically significant (p = 1.7 × 10−15).

Metric Scoring Ranking

Mean H (↓) [95% CI] 1.213 [1.137, 1.289] 0.729 [0.665, 0.793] Unanimous rate (↑) 0.8% [0.1%, 4.6%] 10.1% [5.9%, 16.8%]

sets with matched subject matter: each task asks the model to identify the best image and, in the stricter setting, also the worst. The remainder of this section describes the task layout, the data construction pipeline, and the ground-truth formation procedure.

###### 3.1 Overview

Task formulation. Each benchmark task presents a candidate set of images with matched subject matter, so performance depends on differences in execution rather than differences in depicted content. VAB contains 400 released tasks and 1,195 images across fine art, photography, and illustration, covering 24 topics with candidate set sizes from 2 to 6 images. We consider two settings: in Top-1, the model selects the best image in the set; in TB-1, it identifies both the best and the worst image. Figure 1 provides a visual overview, Figure 2 summarizes the released domain and topic distribution, and Appendix B.1 reports detailed benchmark statistics.

[Figure 257]

[Figure 258]

Macro (1%)

[Figure 259]

[Figure 260]

[Figure 261]

Illustration (25%)

[Figure 262]

[Figure 263]

Sports (0.25%)

[Figure 264]

[Figure 265]

[Figure 266]

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

[Figure 281]

[Figure 282]

Night/Astro (0.5%)

Concept Art (5.50%)

[Figure 283]

[Figure 284]

[Figure 285]

Wildlife (2.25%)

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

Stylized 3D (2.75%)

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

Digital Art (5.75%)

[Figure 294]

[Figure 295]

Fine Art (40.25%)

[Figure 296]

[Figure 297]

[Figure 298]

Pixel Art (5.25%)

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

Visual Aesthetic Benchmark

Manga (3.25%) Comic (2.5%)

[Figure 303]

[Figure 304]

[Figure 305]

Calligraphy (2.75%)

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

ax

[Figure 311]

[Figure 312]

[Figure 313]

Landscape Color (14%)

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

Quick Sketch (2%)

[Figure 321]

Photography (34.75%)

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

Portrait Color (0.5%)

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

Portrait Sketch (8.25%) Still Life Color (6%)

[Figure 339]

[Figure 340]

Architecture (8%) Food/Product (1.25%) Landscape (11.25%) Portrait (0.5%) Street/City (9.75%)

[Figure 341]

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

Ink & Wash (1%)

[Figure 363]

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

Still Life Sketch (3.5%)

[Figure 371]

[Figure 372]

[Figure 373]

[Figure 374]

[Figure 375]

[Figure 376]

[Figure 377]

Chinese Painting (2.25%)

[Figure 378]

[Figure 379]

Figure 2: Detailed distribution of VAB.

Evaluation protocol. Labels are derived from direct comparative judgments by independent experts rather than post hoc aggregation of absolute scores. Each task is evaluated three times, each under an independent random permutation of the candidate order. We report ap@1, the average accuracy across the three evaluations, and pass^3, which counts a task as correct only if the model is correct in all three. Representative topic-level examples are collected in Appendix I.

###### 3.2 Data Construction

Because VAB is annotated through direct ranking over candidate sets, data construction is organized around matched image groups rather than standalone images: in every domain, the members of a set share subject matter but vary in execution quality. Figure 3 summarizes the pipeline; Appendix B.2 reports the full raw-pool statistics, and Appendix B.3 contains the domain-level construction details.

Fine Art. We commissioned 426 painting sets comprising 1,142 images. Each set is built from constrained artist prompts that preserve subject matter while allowing variation in composition, value structure, color handling, and finish. All works were commissioned specifically for VAB.

Photography. We constructed 670 photography sets comprising 1,809 images. Each set originates from a single source photograph, many provided directly by professional photographers, and is expanded through two pipelines: an expert-edit pipeline based on professional retouching tools, which can produce either original/improved pairs or larger multi-expert sets, and a scalable agent-edit pipeline that generates improved and degraded variants. The agent-edit prompt sequence and runtime pipeline are recorded in Appendix H.1.

Step 1: Data Collection by Domain

[Figure 381]

[Figure 382]

[Figure 383]

###### Artwork Photography Illustration

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

Architecture Food & Product

Calligraphy Chinese Painting

Landscape Macro Night Astro

Ink & Wash Landscape Color

Anime/Manga Comic Concept Art

Portrait Color

[Figure 389]

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

Digital/AI Art Pixel Art Stylized 3D

Portrait Sketch

Quick Sketch

Still Life Color

Still Life Sketch

Portrait Sports Street City Wildlife

Single Photo

[Figure 394]

Constrained Prompt

###### Generative Pipeline 3D Pipeline

[Figure 395]

[Figure 396]

Expert-Edit Pipeline Agent-Edit Pipeline

[Figure 397]

[Figure 398]

Template Based Prompts

[Figure 399]

|[Figure 400]| |
|---|---|
| | |

Critic Model

|[Figure 401]| |
|---|---|
|[Figure 402]<br><br>Experts| |

Artists Produce Independent Renditions

Photos with Flaws

Multi-viewpoint Rendering Fixed Lighting & Background

[Figure 403]

3D Models

Image Generation Models

Critiques & Reviews

[Figure 404]

[Figure 405]

Recomposition

MLLM

Color &Tone Correction

[Figure 406]

[Figure 407]

[Figure 408]

###### Image-to-Image Models

[Figure 409]

[Figure 410]

[Figure 411]

[Figure 412]

[Figure 413]

[Figure 414]

[Figure 415]

[Figure 416]

[Figure 417]

[Figure 418]

[Figure 419]

[Figure 420]

[Figure 421]

[Figure 422]

[Figure 423]

[Figure 424]

[Figure 425]

[Figure 426]

[Figure 427]

|[Figure 428]|
|---|

|[Figure 429]|
|---|

[Figure 430]

[Figure 431]

|[Figure 432]|
|---|

|[Figure 433]|
|---|

|[Figure 434]|
|---|

|[Figure 435]|
|---|

Improved Version

Aesthetic Variants Aesthetic Variants

Aesthetic Variants

Better Variant Worse Variant

Step 2: Expert Annotation Step 3: Model Training

Training Data

LoRA Fine-tuning

Training Model

Candidate Sets

Assignment Expert Evaluation

Consensus Filter

Final Benchmark

[Figure 436]

[Figure 437]

[Figure 438]

[Figure 439]

[Figure 440]

[Figure 441]

[Figure 442]

[Figure 443]

[Figure 444]

Annotated training set

Aesthetic learning

Aesthetic quality predictor

Identify Best & Worst

Strong majority agreement

400 tasks, 1,195 images

From Step 1 10 Judges per Set

Figure 3: Overview of the VAB data construction pipeline. Images are organized into candidate sets with matched subject matter and varying aesthetic execution, then passed through expert review and consensus filtering before entering the benchmark.

Illustration. We constructed 271 illustration sets comprising 908 images. These sets are formed either by prompt-based generation with text-to-image models or by rendering CC0 3D assets under controlled viewpoint and presentation changes, producing multiple images with matched subject matter but varying execution. Prompt templates are in Appendix H.2.

All collected sets are deduplicated and reviewed for semantic consistency before annotation.

###### 3.3 Expert Annotation and Ground Truth

Following Section 2, VAB uses ranking-based expert consensus for ground truth.

Review setup. Creation and evaluation are strictly separated. In the raw pre-filter pool, each collected candidate set is reviewed by 10 independent expert judges drawn from a pool of more than 100 evaluators, yielding over 13,000 assessments and more than 300 hours of review. Judges remain blind to artist identity throughout annotation. Rubrics and interface examples are in Appendix C. Additional topic-level examples are in Appendix I.

Annotation protocol. Each task presents k candidate images side by side. Judges first assess each image against the domain-specific rubric and then make a comparative selection over the full set. For k = 2, a judge selects the best image. For k ≥ 3, a judge selects both the best and the worst image, with distinct choices required. Every task receives exactly 10 judgments. A representative annotation example is shown in Figure 4.

[Figure 446]

[Figure 447]

[Figure 448]

[Figure 449]

[Figure 450]

[Figure 451]

[Figure 452]

[Figure 453]

[Figure 454]

[Figure 455]

[Figure 456]

[Figure 457]

[Figure 458]

[Figure 459]

[Figure 460]

[Figure 461]

[Figure 462]

[Figure 463]

[Figure 464]

[Figure 465]

[Figure 466]

[Figure 467]

[Figure 468]

[Figure 469]

[Figure 470]

[Figure 471]

[Figure 472]

[Figure 473]

[Figure 474]

[Figure 475]

[Figure 476]

[Figure 477]

[Figure 478]

[Figure 479]

[Figure 480]

[Figure 481]

[Figure 482]

[Figure 483]

[Figure 484]

[Figure 485]

[Figure 486]

[Figure 487]

[Figure 488]

[Figure 489]

[Figure 490]

[Figure 491]

###### EXPERT CONSENSUS RANKING

###### The ReviewPanel & Mechanism

###### The Options (Painting Option A-D)

###### A

- Expert 1

[Figure 492]

- Expert 2

[Figure 493]

- Expert 3

[Figure 494]

- Expert 4

[Figure 495]

- Expert 5

[Figure 496]

- Expert 6

[Figure 497]

- Expert 7

[Figure 498]

- Expert 8

[Figure 499]

- Expert 9

[Figure 500]

- Expert 10

10 EXPERTS

RANKING PROCESS

###### B

###### Expert Evaluation: Image Quality Ranking

###### Final benchmark output

###### C

###### Image A Image B Image C Image B

Best label: D; Worst label: A Retained: 8/10 experts select D as best, and 7/10 experts select A as worst, both exceeding the consensus threshold.

###### D

RANKING RESULTS

INDIVIDUAL RANKING SHEET (10 EXPERTS’ RESULT)

Figure 4: Example of a single VAB annotation task, illustrating the candidate set presented to expert judges and the resulting consensus decision. The consensus threshold is determined through statistical confidence; see Appendix D for details.

Consensus filter. A task is retained only if expert votes are sufficiently concentrated. For k = 2, the best-image votes must meet the threshold; for k ≥ 3, the best-image and worst-image votes must both meet it, so satisfying only one is insufficient. The thresholds are chosen by statistical confidence under a uniform-random null, and the exact derivation and threshold table are provided in Appendix D.

###### 4 Experiments

###### 4.1 Setup

Evaluation scope. We evaluate 20 MLLMs and six reward models on VAB. Unless otherwise noted, each model is evaluated under its default inference configuration.

We compare all systems against two baselines. The Human Expert baseline is obtained from a fresh round of annotation in which 10 expert judges evaluate each released task under the benchmark interface, and we report their average accuracy against the released labels. The Random Guess baseline is computed analytically for each task size k.

MLLMs. Our MLLM evaluation covers 20 closed-source and open-weight models spanning the Claude family (Anthropic, 2026a; 2025a; 2026b; 2025b; 2026c), the Gemini family (Google DeepMind, 2025b;c; 2026), GPT and o-series models (OpenAI, 2025a;b;c;d;e;f), and additional systems from Doubao, Grok, GLM, Kimi, and Qwen (ByteDance Seed, 2026; xAI, 2025; Z.ai, 2025; Team et al., 2026; Bai et al., 2023; Qwen Team, 2026). For each task, the model selects either the best image (Top-1) or both the best and worst images (TB-1). We evaluate each task three times under independent random permutations of candidate order and report ap@1 and pass^3 as defined in Section 3. Prompts are provided in Appendix H.3.

Reward Models. We additionally evaluate six reward models: ArtiMuse (Cao et al., 2025), PEAS-aes and PEAS-ava (Yun and Choo, 2024), Q-Align (Wu et al., 2023), Q-Instruct (Wu et al., 2024), and RealQA-ava (Li et al., 2025). Unlike MLLMs, these models assign scalar scores to images independently rather than selecting directly from a set. We therefore follow each model’s official scoring interface, score each image three times, average the scores, and then rank images within each task. The main text reports Top-1 and

Table 4: Main VAB results for 20 frontier MLLMs. We report Top-1 and TB-1 accuracy under both ap@1 and

pass^3 (%). Best result in each column is shown in bold.

Overall Fine Art Illustration Photography Top-1 TB-1 Top-1 TB-1 Top-1 TB-1 Top-1 TB-1 Model ap@1 p^3 ap@1 p^3 ap@1 p^3 ap@1 p^3 ap@1 p^3 ap@1 p^3 ap@1 p^3 ap@1 p^3

Random Guess 37.2 6.6 27.5 5.3 41.0 8.9 36.0 8.5 31.5 4.4 19.2 3.3 36.8 5.6 23.7 3.1 Human Expert 77.7 68.9 80.4 74.7 70.3 54.4 79.9 72.4

Closed-Source Models

CLAUDE-HAIKU-4.5 43.3 18.8 32.8 12.5 44.7 19.9 40.4 19.3 32.7 13.0 18.7 5.0 49.4 21.6 34.3 10.1

- CLAUDE-SONNET-4.5 47.2 24.2 35.7 14.5 44.5 20.5 40.2 19.3 37.3 17.0 20.7 8.0 57.6 33.8 41.2 13.7

- CLAUDE-SONNET-4.6 54.8 40.2 40.6 26.5 48.4 37.3 44.7 34.2 45.3 30.0 29.3 19.0 68.8 51.1 43.9 23.0

- CLAUDE-OPUS-4.5 51.2 34.2 39.3 20.2 45.3 28.0 40.4 24.2 40.0 27.0 25.3 14.0 65.9 46.8 48.2 20.1

- CLAUDE-OPUS-4.6 53.0 35.5 38.8 20.0 48.7 27.3 43.3 23.6 42.3 31.0 25.0 17.0 65.7 48.2 43.6 18.0

DOUBAO-SEED-2.0-PRO 51.3 33.2 42.8 23.5 55.9 39.8 49.9 32.9 39.3 18.0 25.0 6.0 54.7 36.7 47.5 25.2 GEMINI-3-FLASH 52.8 28.0 40.7 15.8 51.8 23.0 46.8 20.5 46.0 22.0 26.0 8.0 58.8 38.1 44.1 15.8 GEMINI-3-PRO 55.4 35.0 40.0 22.0 52.0 32.9 46.6 29.8 47.3 28.0 28.7 14.0 65.2 42.4 40.5 18.7 GEMINI-3.1-PRO 54.8 35.0 41.8 22.2 51.8 31.1 45.1 26.1 49.7 32.0 28.3 13.0 62.1 41.7 47.5 24.5

- GPT-4.1 49.3 29.5 39.8 21.2 47.2 28.0 43.1 25.5 43.0 24.0 27.7 15.0 56.4 35.3 44.8 20.9

GPT-5 51.0 32.2 40.0 21.8 49.9 30.4 42.7 25.5 36.7 19.0 20.3 9.0 62.6 43.9 51.1 26.6

- GPT-5.1 52.3 29.5 40.2 20.0 49.9 30.4 44.5 29.8 45.7 25.0 26.7 11.0 60.0 31.7 44.8 15.1

GPT-5.1-CODEX 50.2 31.8 39.2 22.0 47.4 29.8 42.0 27.3 41.3 22.0 24.3 15.0 60.0 41.0 46.8 20.9 GPT-5.2 46.3 24.0 37.4 15.5 40.8 18.0 35.4 14.9 33.7 13.0 21.3 8.0 61.9 38.8 51.3 21.6

GROK-4.1-FAST 44.6 15.5 35.2 12.5 44.5 18.0 39.5 17.4 36.3 15.0 25.0 11.0 50.6 12.9 37.4 7.9

O4-MINI 48.2 25.0 39.8 21.5 44.7 23.6 41.0 23.0 38.0 16.0 20.7 7.0 59.5 33.1 52.3 30.2 Open-Weight Models

GLM-4.6V 43.0 17.5 34.1 11.5 42.7 18.0 39.3 18.6 43.3 19.0 27.0 8.0 43.2 15.8 33.1 5.8

KIMI-K2.5 49.4 26.8 36.8 15.0 43.5 19.9 38.5 18.0 43.0 21.0 29.0 14.0 60.9 38.8 40.5 12.2 QWEN3-VL-235B-A22B 44.9 20.5 35.2 14.0 42.4 22.4 36.9 19.3 38.3 12.0 25.7 8.0 52.5 24.5 40.0 12.2

QWEN3.5-397B-A17B 50.1 29.8 38.9 17.2 42.7 21.1 37.3 17.4 44.3 24.0 26.0 8.0 62.8 43.9 50.1 23.7

TB-1; model-specific scoring interfaces and full result tables, including Bot-1 and topic-level breakdowns, are deferred to Appendix H.4, Appendix E.2, and Appendix E.4.

###### 4.2 Results

Tables 4 and 5 summarize the main results; Appendices E.1–E.4 provide the full tables, topic-level breakdowns, and auxiliary metrics.

###### 4.2.1 Results for Multimodal Large Language Models

Gap to experts. Current frontier models remain far below expert performance. On the strictest metric, TB-1 pass^3, Claude Sonnet 4.6 attains 26.5%, against 68.9% for human experts; even under the more lenient TB-1 ap@1, the strongest model, Doubao Seed 2.0 Pro, reaches 42.8%, still 26.1 points short of the expert baseline.

Progress across families. Improvements are uneven across model families. Claude Sonnet 4.6 nearly doubles its predecessor on TB-1 pass^3 (14.5% → 26.5%), whereas the GPT-5 series regresses from 21.8% (GPT-5) to 20.0% (GPT-5.1) and 15.5% (GPT-5.2). A clear closed/open gap persists under the stricter metric: the strongest open-weight system, Qwen3.5-397B-A17B, reaches 17.2% on TB-1 pass^3, compared with 26.5% for Claude Sonnet 4.6.

Domain difficulty and stability. Accuracy varies strongly across domains and across repeated evaluations. Illustration is the most difficult domain, where Claude Sonnet 4.6 attains only 19.0% on TB-1 pass^3, while photography is comparatively more tractable, with o4-mini reaching 30.2%. The large ap@1-to-pass^3

BakeAI 5 Fine-Tuning on Expert Aesthetic Judgments

Table 5: Main reward-model results on VAB. We report overall and domain-level Top-1 and TB-1 accuracy (%).

Best non-baseline result in each column is shown in bold.

Overall Fine Art Illustration Photography Model Top-1 TB-1 Top-1 TB-1 Top-1 TB-1 Top-1 TB-1

Random Guess 37.2 27.5 41.0 36.0 31.5 19.2 36.8 23.7 Human Expert 77.7 68.9 80.4 74.7 70.3 54.4 79.9 72.4

ARTIMUSE (Cao et al., 2025) 44.2 34.8 50.9 46.0 37.0 20.0 41.7 32.4 PEAS-AES (Yun and Choo, 2024) 48.5 37.8 49.7 44.1 34.0 22.0 57.6 41.7 PEAS-AVA (Yun and Choo, 2024) 49.0 37.0 50.9 44.1 25.0 13.0 64.0 46.0

Q-ALIGN (Wu et al., 2023) 52.2 38.2 52.2 44.7 31.0 16.0 67.6 46.8 Q-INSTRUCT (Wu et al., 2024) 38.5 31.2 37.9 34.2 32.0 21.0 43.9 35.2 REALQA-AVA (Li et al., 2025) 44.2 30.2 40.4 36.6 33.0 15.0 56.8 33.8

gap—for example, GLM-4.6V drops from 34.1% to 11.5% on TB-1—indicates that sensitivity to candidate order and response instability remain major failure modes.

###### 4.2.2 Results for Reward Models

Overall performance. Reward models also remain well below human experts, although some are competitive with MLLMs on Top-1. Q-Align is the strongest overall, reaching 52.2% on Top-1 and 38.2% on TB-1, against 77.7% and 68.9% for the human baseline.

Domain trends and asymmetric failure modes. Domain trends broadly mirror the MLLM results. Appendix E.2 additionally reveals an asymmetry between best and worst predictions: Q-Instruct reaches 51.5% on overall Bot-1 and 77.0% on photography Bot-1, despite considerably weaker Top-1 scores of 38.5% overall and 43.9% in photography. This pattern suggests that detecting aesthetic failure is in some cases easier than identifying the strongest image in a set.

Comparison with MLLMs. The best reward-model Top-1 result , 52.2%, is numerically close to the best MLLM Top-1 ap@1 result , 55.4%, but the comparison is not strictly like-for-like: MLLMs make a single comparative decision over a candidate set, whereas reward models score images independently and are then ranked within the set.

###### 4.3 Ablations and Additional Analysis

Temperature ablation. For models with an exposed temperature parameter, greedy decoding improves Top-1 pass^3 by 1.1 points on average and TB-1 pass^3 by 2.3 points. Decoding stochasticity therefore contributes noise, but it does not explain the main performance gap; full model-by-model results are reported in Appendix E.5.

Candidate set size. Performance degrades sharply as the candidate set grows. Under TB-1 pass^3, the best model falls from 47.3% on two-image tasks to 6.7% on four-image tasks, while human experts decline from 87.1% to 43.6%. Larger candidate sets therefore amplify both comparative difficulty and instability; the full breakdown is reported in Appendix E.6.

###### 5 Fine-Tuning on Expert Aesthetic Judgments

Training Setup. We fine-tune QWEN3.5-35B-A3B using Low-Rank Adaptation (LoRA) (Hu et al., 2022) with rank 64 and α = 128, freezing the vision tower while updating the multimodal projector. Supervision comes from 2,000 additional expert-annotated examples collected with the same annotation pipeline described in Section 3.3; these examples are fully disjoint from the benchmark evaluation set. We train for three epochs with a learning rate of 1 × 10−4 and cosine scheduling. Additional hyperparameters and full fine-tuning results are provided in Appendix G. The fine-tuned model is denoted as KALLISTI-35B-A3B.

BakeAI 6 Conclusion

- Table 6: Fine-tuning results on VAB. We compare open-weight baselines with KALLISTI-35B-A3B, which is obtained by LoRA fine-tuning on 2,000 expert-annotated training examples. All models are evaluated under their default inference configuration. Best result in each column is shown in bold.

Top-1 TB-1 Model ap@1 p^3 ap@1 p^3 Open-Weight Models

GLM-4.6V 43.0 17.5 34.1 11.5 KIMI-K2.5 49.4 26.8 36.8 15.0

QWEN3-VL-235B-A22B 44.9 20.5 35.2 14.0 QWEN3.5-397B-A17B 50.1 29.8 38.9 17.2

Ours QWEN3.5-35B-A3B (base) 47.5 25.5 36.5 15.8

KALLISTI-35B-A3B 51.3+3.8 29.5+4.0 38.8+2.3 17.5+1.7

Improvement over the base model. Table 6 compares KALLISTI-35B-A3B against its base model and a set of open-weight baselines. Fine-tuning yields consistent gains across all four reported metrics: Top-1 pass^3 improves from 25.5% to 29.5%, TB-1 pass^3 from 15.8% to 17.5%, Top-1 ap@1 from 47.5% to 51.3%, and TB-1 ap@1 from 36.5% to 38.8%. The topic-level breakdown is reported in Appendix G.2.

Scaling implication. KALLISTI-35B-A3B approaches the substantially larger Qwen3.5-397B-A17B on Top-1 pass^3, 29.5% versus 29.8%, and lies within roughly one point on the other metrics. This is preliminary evidence that the comparative supervision in VAB carries enough aesthetic signal to compensate for an order-of-magnitude reduction in parameter count under lightweight adaptation.

###### 6 Conclusion

We introduced VAB, a benchmark that recasts aesthetic evaluation as comparative selection over candidate sets with matched subject matter, with labels grounded in expert ranking rather than scalar scoring. VAB contains 400 tasks and 1,195 images spanning fine art, photography, and illustration. A controlled human study motivates the protocol: direct ranking yields 42 percentage points higher inter-annotator agreement than score-derived rankings. Across 20 frontier MLLMs and six reward models, the strongest system reaches 26.5% on TB-1 pass^3 against 68.9% for human experts, and accuracy drops sharply as the candidate set grows; models also exhibit pronounced positional instability across permutations of the candidate order. Fine-tuning a 35B-parameter model on 2,000 expert examples brings its accuracy close to that of a 397B-parameter open-weight model, suggesting that expert comparative supervision transfers efficiently with lightweight adaptation. We see several directions for future work: broadening domain and annotator diversity beyond the present three domains, improving robustness to candidate-order permutations, and scaling expert comparative supervision to larger training corpora. By making expert aesthetic judgment a measurable, comparative target, VAB establishes the foundation on which expert-aligned multimodal aesthetic judgment can now be systematically pursued.

###### Ethics Statement

This study involved human participants providing aesthetic judgments through an online evaluation platform. The University of Washington Human Subjects Division (HSD) reviewed the protocol and determined that the research qualifies as exempt human subjects research (Category 2 and 3; minimal-risk behavioral research) under U.S. federal regulations. Participation was voluntary and participants could stop at any time. No sensitive personal information was collected. Only aggregated benchmark results are publicly released, and no identifiable participant data are included in any publications or datasets. The exempt determination letter is on file with the authors.

All fine art works were created as commissioned works for VAB, with participating artists informed of and consenting to the use of their work for research. Professional photographers whose images served as source material for the photography pipeline contributed their work under appropriate licensing terms. Agent-edited variants derived from these source photographs were produced within the scope of those agreements. Illustration sets generated through text-to-image models used structured prompts without reproducing copyrighted material, and 3D-rendered sets were constructed exclusively from CC0-licensed assets. We are committed to crediting and fairly compensating all human creators whose work contributed to this benchmark.

###### References

Adobe. Adobe Lightroom. Adobe Website, 2026a. URL https://www.adobe.com/products/ photoshop-lightroom.html.

Adobe. Adobe Photoshop. Adobe Website, 2026b. URL https://www.adobe.com/products/photoshop. html.

Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. Advances in neural information processing systems, 35:23716–23736, 2022.

- Anthropic. Introducing Claude Opus 4.5. Anthropic News, 2025a. URL https://www.anthropic.com/ news/claude-opus-4-5.

- Anthropic. Introducing Claude Sonnet 4.5. Anthropic News, 2025b. URL https://www.anthropic.com/ news/claude-sonnet-4-5.

Anthropic. Introducing Claude Haiku 4.5. Anthropic News, 2026a. URL https://www.anthropic.com/ news/claude-haiku-4-5.

Anthropic. Introducing Claude Opus 4.6. Anthropic News, 2026b. URL https://www.anthropic.com/ news/claude-opus-4-6.

- Anthropic. Introducing Claude Sonnet 4.6. Anthropic News, 2026c. URL https://www.anthropic.com/ news/claude-sonnet-4-6.

Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023.

Carlo Bonferroni. Teoria statistica delle classi e calcolo delle probabilita. Pubblicazioni del R istituto superiore

di scienze economiche e commericiali di firenze, 8:3–62, 1936. ByteDance Seed. SEED2. ByteDance Seed Website, 2026. URL https://seed.bytedance.com/en/seed2. Shuo Cao, Nan Ma, Jiayang Li, Xiaohui Li, Lihao Shao, Kaiwen Zhu, Yu Zhou, Yuandong Pu, Jiarui Wu,

Jiaquan Wang, et al. Artimuse: Fine-grained image aesthetics assessment with joint scoring and expertlevel understanding. arXiv preprint arXiv:2507.14533, 2025.

Creative Commons. CC0 1.0 Universal. Creative Commons Website, 2009. URL https://creativecommons. org/publicdomain/zero/1.0/.

Wenliang Dai, Junnan Li, Dongxu Li, Anthony Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale N Fung, and Steven Hoi. Instructblip: Towards general-purpose vision-language models with instruction tuning. Advances in neural information processing systems, 36:49250–49267, 2023.

Morten W Fagerland, Stian Lydersen, and Petter Laake. The McNemar test for binary matched-pairs data: mid-p and asymptotic are better than exact conditional. BMC medical research methodology, 13(1):91, 2013.

Alan H. Goldman. Aesthetic Value. Westview Press, 1995. Google DeepMind. Gemini 3 Pro Image (Nano Banana Pro). Google DeepMind Model Card, 2025a. URL

https://deepmind.google/models/model-cards/gemini-3-pro-image/. Google DeepMind. Gemini 3 Flash. Google DeepMind Model Card, 2025b. URL https://deepmind. google/models/model-cards/gemini-3-flash/. Google DeepMind. Gemini 3 Pro. Google DeepMind Model Card, 2025c. URL https://deepmind.google/ models/model-cards/gemini-3-pro/. Google DeepMind. Gemini 3.1 Pro. Google DeepMind Model Card, 2026. URL https://deepmind.google/ models/model-cards/gemini-3-1-pro/. Shuai He, Yongchang Zhang, Rui Xie, Dongxiang Jiang, and Anlong Ming. Rethinking Image Aesthetics Assessment: Models, Datasets and Benchmarks. In IJCAI, volume 6, page 22, 2022.

Shuai He, Anlong Ming, Yaqi Li, Jinyuan Sun, ShunTian Zheng, and Huadong Ma. Thinking image color aesthetics assessment: Models, datasets and benchmarks. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 21838–21847, 2023.

Vlad Hosu, Bastian Goldlucke, and Dietmar Saupe. Effective aesthetics prediction with multi-level spatially pooled features. In proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9375–9383, 2019.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Liang Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. Iclr, 1(2):3, 2022.

Yipo Huang, Xiangfei Sheng, Zhichao Yang, Quan Yuan, Zhichao Duan, Pengfei Chen, Leida Li, Weisi Lin, and Guangming Shi. Aesexpert: Towards multi-modality foundation model for image aesthetics perception. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 5911–5920, 2024a.

Yipo Huang, Quan Yuan, Xiangfei Sheng, Zhichao Yang, Haoning Wu, Pengfei Chen, Yuzhe Yang, Leida Li, and Weisi Lin. Aesbench: An expert benchmark for multimodal large language models on image aesthetics perception. arXiv preprint arXiv:2401.08276, 2024b.

Maurice G Kendall. A new measure of rank correlation. Biometrika, 30(1-2):81–93, 1938. Maurice G Kendall and B Babington Smith. The problem of m rankings. The annals of mathematical statistics,

10(3):275–287, 1939.

Shu Kong, Xiaohui Shen, Zhe Lin, Radomir Mech, and Charless Fowlkes. Photo aesthetics ranking network with attributes and content adaptation. In European conference on computer vision, pages 662–679. Springer, 2016.

Jun-Tae Lee and Chang-Su Kim. Image aesthetic assessment based on pairwise comparison a unified approach to score regression, binary classification, and personalization. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1191–1200, 2019.

Jerrold Levinson. The Oxford handbook of aesthetics. Oxford University Press, 2005. Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training

with frozen image encoders and large language models. In International conference on machine learning, pages 19730–19742. PMLR, 2023.

Mingxing Li, Rui Wang, Lei Sun, Yancheng Bai, and Xiangxiang Chu. Next token is enough: Realistic image quality and aesthetic scoring with multimodal large language model. arXiv preprint arXiv:2503.06141, 2025.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023.

Shuang Ma, Jing Liu, and Chang Wen Chen. A-lamp: Adaptive layout-aware multi-patch deep convolutional neural network for photo aesthetic assessment. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4535–4544, 2017.

Quinn McNemar. Note on the sampling error of the difference between correlated proportions or percentages. Psychometrika, 12(2):153–157, 1947.

Midjourney. Version. Midjourney Documentation, 2026. URL https://docs.midjourney.com/docs/ models.

Naila Murray, Luca Marchesotti, and Florent Perronnin. AVA: A large-scale database for aesthetic visual analysis. In 2012 IEEE conference on computer vision and pattern recognition, pages 2408–2415. IEEE, 2012.

- OpenAI. GPT-4.1. OpenAI API Documentation, 2025a. URL https://developers.openai.com/api/docs/ models/gpt-4.1.

OpenAI. Introducing GPT-5. OpenAI Website, August 2025b. URL https://openai.com/index/ introducing-gpt-5/.

- OpenAI. Introducing GPT-5.1. OpenAI Website, 2025c. URL https://developers.openai.com/api/ docs/models/gpt-5.1.

OpenAI. GPT-5.1 Codex. OpenAI API Documentation, 2025d. URL https://developers.openai.com/ api/docs/models/gpt-5.1-codex.

- OpenAI. Introducing GPT-5.2. OpenAI Website, 2025e. URL https://openai.com/index/ introducing-gpt-5-2/.

OpenAI. o4-mini. OpenAI API Documentation, 2025f. URL https://developers.openai.com/api/docs/

models/o4-mini. Allen Parducci. Category judgment: a range-frequency model. Psychological review, 72(6):407, 1965. PixelLab. PixelLab. PixelLab Website, 2026. URL https://www.pixellab.ai/.

Daiqing Qi, Handong Zhao, Jing Shi, Simon Jenni, Yifei Fan, Franck Dernoncourt, Scott Cohen, and Sheng Li. The Photographer’s Eye: Teaching Multimodal Large Language Models to See, and Critique Like Photographers. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 24807–24816, 2025.

Qwen Team. Qwen3.5. Qwen Blog, 2026. URL https://qwen.ai/blog?id=qwen3.5. Claude Elwood Shannon. A mathematical theory of communication. The Bell system technical journal, 27(3):

379–423, 1948. Sketchfab. Sketchfab. Sketchfab Website, 2026. URL https://sketchfab.com/. Neil Stewart, Gordon DA Brown, and Nick Chater. Absolute identification by relative judgment. Psychological

review, 112(4):881, 2005. Peize Sun, Yi Jiang, Shoufa Chen, Shilong Zhang, Bingyue Peng, Ping Luo, and Zehuan Yuan. Autoregressive model beats diffusion: Llama for scalable image generation. arXiv preprint arXiv:2406.06525, 2024. Hossein Talebi and Peyman Milanfar. NIMA: Neural image assessment. IEEE transactions on image processing, 27(8):3998–4011, 2018.

Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.

Kimi Team, Tongtong Bai, Yifan Bai, Yiping Bao, SH Cai, Yuan Cao, Y Charles, HS Che, Cheng Chen, Guanduo Chen, et al. Kimi K2. 5: Visual Agentic Intelligence. arXiv preprint arXiv:2602.02276, 2026.

Louis L. Thurstone. A Law of Comparative Judgment. Psychological Review, 34(4):273–286, 1927. doi: 10.1037/h0070288.

Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024.

Wenshan Wang, Su Yang, Weishan Zhang, and Jiulong Zhang. Neural aesthetic image reviewer. IET Computer

Vision, 13(8):749–758, 2019. Frank Wilcoxon. Individual comparisons by ranking methods. Biometrics bulletin, 1(6):80–83, 1945. Edwin B Wilson. Probable inference, the law of succession, and statistical inference. Journal of the American

Statistical Association, 22(158):209–212, 1927.

Haoning Wu, Zicheng Zhang, Weixia Zhang, Chaofeng Chen, Liang Liao, Chunyi Li, Yixuan Gao, Annan Wang, Erli Zhang, Wenxiu Sun, et al. Q-align: Teaching lmms for visual scoring via discrete text-defined levels. arXiv preprint arXiv:2312.17090, 2023.

Haoning Wu, Zicheng Zhang, Erli Zhang, Chaofeng Chen, Liang Liao, Annan Wang, Kaixin Xu, Chunyi Li, Jingwen Hou, Guangtao Zhai, et al. Q-instruct: Improving low-level visual abilities for multi-modality foundation models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 25490–25500, 2024.

xAI. Grok 4.1 Fast. xAI News, 2025. URL https://x.ai/news/grok-4-1-fast. Yuzhe Yang, Liwu Xu, Leida Li, Nan Qie, Yaqian Li, Peng Zhang, and Yandong Guo. Personalized image

aesthetics assessment with rich attributes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19861–19869, 2022.

Ran Yi, Haoyuan Tian, Zhihao Gu, Yu-Kun Lai, and Paul L Rosin. Towards artistic image aesthetics assessment: a large-scale dataset and a new method. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22388–22397, 2023.

Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities. arXiv preprint arXiv:2308.02490, 2023.

Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9556–9567, 2024.

Jooyeol Yun and Jaegul Choo. Scaling up personalized image aesthetic assessment via task vector customiza-

tion. In European Conference on Computer Vision, pages 323–339. Springer, 2024. Z.ai. GLM-4.6V. Z.ai Blog, 2025. URL https://z.ai/blog/glm-4.6v. Hantao Zhou, Longxiang Tang, Rui Yang, Guanyi Qin, Yan Zhang, Yutao Li, Xiu Li, Runze Hu, and Guangtao

Zhai. UniQA: Unified vision-language pre-training for image quality and aesthetic assessment. arXiv preprint arXiv:2406.01069, 2024a.

###### Zhaokun Zhou, Qiulin Wang, Bin Lin, Yiwei Su, Rui Chen, Xin Tao, Amin Zheng, Li Yuan, Pengfei Wan, and Di Zhang. Uniaa: A unified multi-modal image aesthetic assessment baseline and benchmark. arXiv preprint arXiv:2404.09619, 2024b.

###### Appendix

- A Related Work . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
- B Benchmark Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17

- B.1 Final Benchmark Statistics. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
- B.2 Raw Collection Statistics Before Expert Filtering . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
- B.3 Data Construction Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18

- C Annotation Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21

- C.1 Annotation Rubrics . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
- C.2 Annotation Interface . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24

- D Ground Truth Mathematical Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24

- D.1 Selected Thresholds. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24
- D.2 Derivation of Null Pass Probabilities . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24
- D.3 Monte Carlo Verification . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25
- D.4 Special Case for Binary Tasks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25

- E Detailed Experimental Results. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25

- E.1 Comprehensive MLLM Results. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25
- E.2 Comprehensive Reward-Model Results. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26
- E.3 Topic-Level MLLM Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26
- E.4 Topic-Level Reward-Model Results. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 28
- E.5 Temperature Ablation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 28
- E.6 Results by Candidate Set Size . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 38

- F Human Study Details. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 38

- F.1 Detailed Setup and Metric Definitions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 38
- F.2 Extended Interpretation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 39
- F.3 Task Examples. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 40
- F.4 Annotation Protocol . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 40
- F.5 Notation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 45
- F.6 Statistical Tests . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 45
- F.7 Task Lists. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 46

- G Fine-Tuning Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 51

- G.1 Hyperparameters . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 51
- G.2 Detailed Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 52

- H Prompt Templates and Scoring Interfaces . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 56

- H.1 Photography Pipeline Prompts. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 56
- H.2 Illustration Pipeline Prompts. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 58
- H.3 MLLM Evaluation Prompts . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 59
- H.4 Reward-Model Scoring Interfaces . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 60

- I Topic-Level Annotation Examples. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 62

- I.1 Fine Art . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 62
- I.2 Photography. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 71
- I.3 Illustration. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 79

###### A Related Work

Multimodal Large Language Models. Recent MLLMs such as Flamingo, BLIP-2, InstructBLIP, LLaVA, Gemini, and Qwen2-VL have rapidly improved multimodal perception, instruction following, and reasoning (Alayrac et al., 2022; Li et al., 2023; Dai et al., 2023; Liu et al., 2023; Team et al., 2023; Wang et al., 2024). Their capabilities are typically assessed on broad benchmarks such as MM-Vet and MMMU (Yu et al., 2023; Yue et al., 2024), where each example usually admits a single correct answer. Aesthetic judgment is different: it is comparative, convention-sensitive, and not reducible to factual correctness alone. Our focus is therefore whether frontier MLLMs can make stable, expert-aligned selections over candidate sets with matched subject matter rather than pointwise judgments on isolated images.

Image Aesthetics Assessment. Image aesthetics assessment (IAA) has traditionally been posed as singleimage score prediction. Canonical datasets such as AVA and AADB provide large-scale photographic images with scalar ratings. Later resources extend this paradigm through comments, theme-aware labels, and personalization, including AVA-Reviews, TAD66K, and PARA (Murray et al., 2012; Kong et al., 2016; Wang et al., 2019; He et al., 2022; Yang et al., 2022). These resources support a broad line of regression and ranking models, including NIMA, A-Lamp, theme-aware predictors, and more recent vision-language scorers such as Q-Align and UniQA (Talebi and Milanfar, 2018; Ma et al., 2017; He et al., 2022; Wu et al., 2023; Zhou et al., 2024a). Still, the dominant setup remains pointwise, mapping from one image to a score, with preference order inferred only indirectly from independent estimates.

More recently, MLLM-based efforts have moved toward richer aesthetic reasoning. AesBench evaluates aesthetic perception in frontier MLLMs; AesExpert and UNIAA build instruction-tuned systems for critique and assessment; and ArtiMuse and the photography-focused The Photographer’s Eye move closer to expertstyle analysis (Huang et al., 2024b;a; Zhou et al., 2024b; Cao et al., 2025; Qi et al., 2025). Nevertheless, existing resources remain largely single-image, score-centric, or domain-specific. In contrast, VAB is explicitly set-based: each example is a group with matched subject matter, and the task is to identify the strongest and weakest images under controlled content. This formulation targets relative aesthetic discrimination directly, derives labels from expert consensus rather than crowd averages, and spans fine art, photography, and illustration within a unified benchmark.

###### B Benchmark Details

###### B.1 Final Benchmark Statistics

Table 7 reports the final benchmark statistics after expert annotation and consensus filtering. Table 8 shows the candidate set size distribution in the final benchmark by domain, and Table 9 provides the corresponding topic-level breakdown.

###### B.2 Raw Collection Statistics Before Expert Filtering

Before expert filtering, the collected pool contains 1,367 tasks and 3,859 images. Table 10 summarizes the raw collection by domain, Table 11 reports the corresponding topic-level breakdown, and Table 12 further decomposes the photography subset by collection pipeline.

Table 7: Final benchmark statistics for VAB by domain and topic after expert consensus filtering. T/I denotes the

number of tasks and images.

Fine Art (161/458) Photography (139/389) Illustration (100/348) Topic T / I Topic T / I Topic T / I

Landscape Color 56/243 Landscape 45/130 Digital & AI Art 23/84 Portrait Sketch 33/66 Street & City 39/106 Concept Art 22/88 Still Life Color 24/53 Architecture 32/89 Pixel Art 21/84

Still Life Sketch 14/28 Wildlife 9/24 Anime & Manga 13/26 Calligraphy 11/22 Food & Product 5/14 Stylized 3D 11/44 Chinese Painting 9/18 Macro 4/12 Comic 10/22

Quick Sketch 8/16 Night & Astro 2/6 Ink & Wash 4/8 Portrait 2/5 Portrait Color 2/4 Sports 1/3

Table 8: Candidate set size distribution in the final benchmark by domain.

Domain k = 2 k = 3 k = 4 k = 5 k = 6 Fine Art 109 4 13 34 1

Photography 30 107 2 0 0 Illustration 26 0 74 0 0

Total 165 111 89 34 1

###### B.3 Data Construction Details

Fine Art. We commissioned 426 painting sets comprising 1,142 images across nine fine-art topics: calligraphy, Chinese painting, ink and wash, landscape color, portrait color, portrait sketch, quick sketch, still-life color, and still-life sketch. For each topic, artists received constrained prompts that fixed the subject and high-level compositional requirements while allowing variation in execution. The resulting sets preserve semantic intent while differing in composition, value structure, color harmony, and paint handling. All works were commissioned specifically for VAB rather than drawn from existing collections, which reduces the risk of contamination from publicly indexed art. Before annotation, we remove near-duplicates and discard sets with semantic mismatches or trivial cues.

Photography. We constructed 670 photography sets comprising 1,809 images across nine topics: architecture, food and product, landscape, macro, night and astrophotography, portrait, sports, street and city, and wildlife. Each set originates from a single source photograph, many provided directly by professional photographers. We transform these source images into controlled comparison sets through two pipelines.

In the expert-edit pipeline, photographers and retouchers improve each source image using tools such as Adobe Photoshop and Lightroom (Adobe, 2026b;a), applying recomposition, color and tone correction, and content-aware expansion. The original photograph and its improved variant form a pair. For a subset of source images, multiple experts independently edit the same photograph, yielding larger candidate sets with matched subject matter and varying aesthetic quality.

In the agent-edit pipeline, each source image is first sent to a critic model, ArtiMuse (Cao et al., 2025), which produces a detailed aesthetic review. The review and the original image are then passed to a generalpurpose MLLM, GPT-5 (OpenAI, 2025b), to generate two editing prompts: one for improvement and one for degradation. These prompts, together with the original image, are fed to an image-to-image model, Gemini-3-Pro-Image (Nano Banana Pro) (Google DeepMind, 2025a), to generate an improved and a degraded variant. The original, improved, and degraded images form a three-image candidate set. The exact prompt sequence is recorded in Appendix H.1.

Table 9: Topic-level breakdown of the final benchmark after expert consensus filtering.

Domain Topic k = 2 k = 3 k = 4 k = 5 k = 6 Tasks Avg. k

Fine Art Calligraphy 11 0 0 0 0 11 2.00 Fine Art Chinese Painting 9 0 0 0 0 9 2.00 Fine Art Ink and Wash 4 0 0 0 0 4 2.00 Fine Art Landscape Color 6 4 12 33 1 56 4.34 Fine Art Portrait Color 2 0 0 0 0 2 2.00 Fine Art Portrait Sketch 33 0 0 0 0 33 2.00 Fine Art Quick Sketch 8 0 0 0 0 8 2.00 Fine Art Still Life Color 22 0 1 1 0 24 2.21 Fine Art Still Life Sketch 14 0 0 0 0 14 2.00

Photography Architecture 7 25 0 0 0 32 2.78 Photography Food and Product 1 4 0 0 0 5 2.80 Photography Landscape 6 38 1 0 0 45 2.89 Photography Macro 0 4 0 0 0 4 3.00 Photography Night and Astrophotography 0 2 0 0 0 2 3.00 Photography Portrait 1 1 0 0 0 2 2.50 Photography Sports 0 1 0 0 0 1 3.00 Photography Street and City 11 28 0 0 0 39 2.72 Photography Wildlife 4 4 1 0 0 9 2.67

Illustration Anime and Manga 13 0 0 0 0 13 2.00 Illustration Comic 9 0 1 0 0 10 2.20 Illustration Concept Art 0 0 22 0 0 22 4.00 Illustration Digital and AI Art 4 0 19 0 0 23 3.65 Illustration Pixel Art 0 0 21 0 0 21 4.00 Illustration Stylized 3D 0 0 11 0 0 11 4.00

Table 10: Raw collection statistics by domain before expert filtering.

Domain Tasks Images Avg. k Fine Art 426 1,142 2.68

Photography 670 1,809 2.70 Illustration 271 908 3.35

Total 1,367 3,859 2.82

All photography sets from both pipelines are deduplicated and reviewed for semantic consistency before annotation.

Illustration. We constructed 271 illustration sets comprising 908 images across six topics: anime and manga, comic, concept art, digital and AI art, pixel art, and stylized 3D. The sets are derived from two distinct sources, prompt-based generation and 3D asset rendering.

In the generative pipeline, each set originates from a structured prompt that specifies subject identity, scene context, composition, camera viewpoint, lighting, and stylistic attributes. For each set, the prompt is instantiated using a single text-to-image model, such as Midjourney v7 (Midjourney, 2026), Pixellab.ai (Create M-XL) (PixelLab, 2026), or LlamaGen (Sun et al., 2024), to generate candidate images with matched subject matter and varying aesthetic quality. The prompt templates are provided in Appendix H.2.

In the 3D-rendering pipeline, each set is constructed from CC0-licensed assets sourced from Sketchfab (Creative Commons, 2009; Sketchfab, 2026). For each asset, we render multiple images by varying camera viewpoint while keeping object identity, scene semantics, lighting, and background conditions fixed. The resulting renders form candidate sets with matched subject matter and controlled variation in composition and visual presentation.

All illustration sets from both pipelines are deduplicated and reviewed for semantic consistency before annotation.

Table 11: Topic-level raw collection statistics before expert filtering.

Domain Topic Tasks Images k = 2 k = 3 k = 4 k = 5 k = 6 Avg. k

Fine Art Landscape Color 113 502 12 6 20 70 5 4.44 Fine Art Portrait Sketch 93 186 93 0 0 0 0 2.00 Fine Art Still Life Color 73 160 68 0 1 4 0 2.19 Fine Art Still Life Sketch 43 86 43 0 0 0 0 2.00 Fine Art Calligraphy 44 88 44 0 0 0 0 2.00 Fine Art Quick Sketch 30 60 30 0 0 0 0 2.00 Fine Art Chinese Painting 13 26 13 0 0 0 0 2.00 Fine Art Ink and Wash 10 20 10 0 0 0 0 2.00 Fine Art Portrait Color 7 14 7 0 0 0 0 2.00

Photography Landscape 280 761 81 197 2 0 0 2.72 Photography Street and City 165 450 45 120 0 0 0 2.73 Photography Architecture 157 420 51 106 0 0 0 2.68 Photography Wildlife 24 62 11 12 1 0 0 2.58 Photography Portrait 13 31 8 5 0 0 0 2.38 Photography Macro 13 36 3 10 0 0 0 2.77 Photography Night and Astrophotography 7 20 1 6 0 0 0 2.86 Photography Food and Product 7 18 3 4 0 0 0 2.57 Photography Sports 4 11 1 3 0 0 0 2.75

Illustration Digital and AI Art 51 184 10 0 41 0 0 3.61 Illustration Concept Art 50 200 0 0 50 0 0 4.00 Illustration Stylized 3D 50 200 0 0 50 0 0 4.00 Illustration Pixel Art 40 160 0 0 40 0 0 4.00 Illustration Anime and Manga 40 82 39 0 1 0 0 2.05 Illustration Comic 40 82 39 0 1 0 0 2.05

- Table 12: Raw photography collection statistics by topic and collection pipeline before expert filtering. Agent-Edit and Expert-Edit denote task counts from the automated editing pipeline and the expert retouching pipeline, respectively.

Topic Tasks Images k = 2 k = 3 k = 4 Avg. k Agent-Edit Expert-Edit Landscape 280 761 81 197 2 2.72 188 92

Street and City 165 450 45 120 0 2.73 115 50

Architecture 157 420 51 106 0 2.68 104 53 Wildlife 24 62 11 12 1 2.58 10 14 Portrait 13 31 8 5 0 2.38 5 8

Macro 13 36 3 10 0 2.77 10 3 Night and Astrophotography 7 20 1 6 0 2.86 5 2 Food and Product 7 18 3 4 0 2.57 4 3 Sports 4 11 1 3 0 2.75 3 1

###### C Annotation Details

###### C.1 Annotation Rubrics

The rubric serves as a structured comparison guide rather than a separate numerical scoring form. Judges assess each image against domain-specific criteria before making their final comparative selection. The rubrics below describe the criteria used for each domain.

###### Fine Art Rubric

Given a set of 2 to 6 fine art paintings with the same style, subject matter, and medium, and only subtle differences, determine which work demonstrates greater artistic completeness and more mature aesthetic expression.

Core Comparison Criteria

- 1. Composition and Visual Order

- • Is the primary subject more clearly defined?
- • Does the composition feel more stable or intentionally dynamic?
- • Is the spatial arrangement natural, without unnecessary congestion or emptiness?

- 2. Color and Tonal Relationships

- • Are the color relationships more harmonious or layered?
- • Is there stronger control over warmth/coolness and light/dark contrast?
- • Does the work avoid muddiness, dullness, or scattered color impressions?

- 3. Technical Control

- • Are the brushstrokes more confident, fluid, and rhythmic?
- • Are there signs of hesitation, redundancy, or ineffective marks?
- • Does the technique serve the image rather than draw attention to itself?

- 4. Degree of Completion

- • Are there any areas that appear unfinished or perfunctory?
- • Are edges and transitions handled naturally?
- • Does the work feel resolved, at a point where the artist could reasonably stop?

- 5. Artistic Expression and Overall Character

- • Is the mood or atmosphere more focused and coherent?
- • Does the work convey a stronger overall character or presence?
- • Does it have greater artistic persuasiveness?

- 6. Final Selection

- • Considering the set as a whole, which image is the strongest work?
- • If the set contains three or more images, which image is the weakest work?

###### Should Not Be Considered

- • Personal preference for subject matter.
- • Any contextual or background information beyond what is visible in the image itself.

###### Photography Rubric

Given a set of 2 to 6 photographs taken of similar subjects, in similar scenes, and under closely matched conditions, determine which image is more successful in terms of photographic language and visual expression.

Core Comparison Criteria

- 1. Composition and Image Structure

- • Is the subject more clearly emphasized?
- • Is the frame more balanced or intentionally dynamic?
- • Is the background cleaner and more supportive of the subject?

- 2. Quality of Light

- • Is the light more natural or more expressive?
- • Are highlights and shadows better controlled?
- • Does the image avoid crushed blacks or blown highlights?

- 3. Sharpness and Focus

- • Is focus more accurate where it matters?
- • Is the relationship between sharpness and blur more appropriate?
- • Are there any technical errors that detract from the image?

- 4. Color or Tonal Rendering

- • Is white balance or color temperature more accurate or more intentional?
- • Are colors or tonal values more consistent and unified?
- • Does the image avoid unwanted color casts or muddy tones?

- 5. Moment and Emotional Expression

- • Does the image capture a stronger or more meaningful moment?
- • Is the emotional or narrative content more compelling?
- • Does it carry a stronger sense of photographic presence?

- 6. Final Selection

- • Considering the set as a whole, which photograph is the strongest image?
- • If the set contains three or more images, which photograph is the weakest image?

###### Should Not Be Considered

- • The personal appeal or attractiveness of the subject.
- • Assumptions about shooting difficulty.
- • Assumptions about post-processing effort.

###### Digital Illustration Rubric

Given a set of 2 to 6 digital illustrations with similar purpose, style, and level of completion, determine which image demonstrates stronger visual design and higher professional illustration quality. Core Comparison Criteria

###### 1. Visual Focus and Information Clarity

- • Is the visual focal point clearer?
- • Does the visual flow guide the viewer more effectively?
- • Is there less visual noise or competition for attention?

###### 2. Form and Proportion

- • Are the structures of figures or objects more accurate?
- • Are poses and silhouettes more aesthetically refined?
- • Does the image avoid stiffness, distortion, or unnatural forms?

###### 3. Color Design

- • Is the color palette more unified and thematically coherent?
- • Are value and lighting hierarchies clearer?
- • Is there effective use of contrast and visual rhythm?

###### 4. Detail Control and Finish

- • Are details handled appropriately, neither excessive nor careless?
- • Are materials and lighting consistent throughout?
- • Are edges, masks, and gradients clean and well controlled?

###### 5. Stylistic Consistency and Purpose Fit

- • Does the illustration better align with its intended stylistic direction?
- • Is quality more consistent within a unified style?
- • Is the image more suitable for its potential use case?

###### 6. Final Selection

- • Considering the set as a whole, which illustration is the strongest work?
- • If the set contains three or more images, which illustration is the weakest work?

###### Should Not Be Considered

- • Personal preference for illustration style.
- • Software or tools used.
- • Hypothetical or imagined client requirements.

BakeAI D Ground Truth Mathematical Details

Table 13: Consensus thresholds and corresponding null pass probabilities by candidate set size k, with n = 10 judges

per task.

k Threshold m paccept(k, m)

- 2 8 10.94%
- 3 7 0.88%
- 4 6 0.95%
- 5 6 0.15%
- 6 6 0.03%

- C.2 Annotation Interface Figure 5 shows an example annotation interface used by expert judges.

[Figure 518]

[Figure 519]

[Figure 520]

Figure 5: The annotation interface presents all k candidate images side by side. Judges first review each image with respect to the rubric criteria, then make a final comparative selection: selecting the best image for k = 2 and selecting both the best and worst images for k ≥ 3.

- D Ground Truth Mathematical Details This appendix provides additional mathematical details for the consensus filter described in Section 3.3.

###### D.1 Selected Thresholds

Table 13 reports the consensus thresholds used in the benchmark together with their corresponding null pass probabilities.

###### D.2 Derivation of Null Pass Probabilities

For k ≥ 3 candidates with n = 10 judges, each judge’s response is modeled as a directed edge i → j (selecting image i as best and image j as worst, with i ̸= j) drawn uniformly from k(k − 1) possibilities. Let cij denote the count of judges choosing the pair (i, j). The probability of any configuration C under the random null is:

n! ∏i̸=j cij!

Pr(C) =

1 k(k − 1)

n

.

The exact by-chance pass probability is computed by summing over all configurations satisfying the consensus threshold:

n

n! ∏i̸=j cij!

1 k(k − 1)

paccept(k,m) = ∑

1 max

Bi ≥ m 1 max

Wj ≥ m ,

i

j

C: cii=0, ∑i̸=j cij=n

Table 14: Monte Carlo verification of null pass probabilities (107 samples).

k m Exact MC Estimate 95% CI

- 2 8 10.94% 10.9496% [10.9303%, 10.9690%]
- 3 7 0.88% 0.8818% [0.8760%, 0.8876%]
- 4 6 0.95% 0.9507% [0.9447%, 0.9567%]
- 5 6 0.15% 0.1489% [0.1465%, 0.1513%]
- 6 6 0.03% 0.0307% [0.0296%, 0.0318%]

where Bi = ∑j̸=i cij (votes for i as best) and Wj = ∑i̸=j cij (votes for j as worst). The exact probabilities were computed by exhaustive enumeration over all valid count configurations C = (cij) satisfying cii = 0 and ∑i̸=j cij = n, summing the multinomial probability mass over the configurations that satisfy the acceptance condition. For k ≥ 3, this acceptance condition is conjunctive: a task passes only if the best-vote threshold and the worst-vote threshold are both met, so satisfying only one of the two is insufficient.

###### D.3 Monte Carlo Verification

We verify the exact probabilities via Monte Carlo simulation with 107 samples per (k, m) configuration. Table 14 compares the exact values from Table 13 against Monte Carlo estimates and reports 95% Wald confidence intervals.

###### D.4 Special Case for Binary Tasks

For binary tasks (k = 2), the worst image is implied once the best is selected. The consensus filter reduces to requiring maxi Bi ≥ m where B ∼ Binomial(n,1/2):

n

n

- 1

- 2

n b

### ∑

paccept(2, m) = Pr(B ≥ m) + Pr(B ≤ n − m) = 2

.

b=m

For n = 10 and m = 8, this gives paccept(2,8) = 2 × (45 + 10 + 1)/1024 = 112/1024 ≈ 10.94%. We use this lower binary threshold because k = 2 tasks provide only a single comparative decision per annotator; compared with k ≥ 3, demanding the same level of concentration would remove a disproportionate number of otherwise informative binary comparisons.

###### E Detailed Experimental Results

This appendix supplements Section 4 with full result tables that are too large for the main text. We retain the same evaluation focus as in the main paper: overall performance relative to expert baselines, cross-domain variation, sensitivity to candidate order, and auxiliary analyses on decoding and task size.

###### E.1 Comprehensive MLLM Results

- Table 15 reports comprehensive MLLM results, including Bot-1, across overall and domain-level evaluations. These results complement Table 4 in the main text by making the asymmetry between identifying the strongest and weakest images explicit. They also make clear that the gap between ap@1 and pass^3 is pervasive rather than concentrated in a small subset of models, consistent with the interpretation in Section 4.2.1 that many MLLM predictions remain sensitive to candidate order and response instability.

The full table also makes several model-level patterns easier to inspect. Within the Claude family, Claude Sonnet 4.6 improves substantially over Claude Sonnet 4.5 on the stricter TB-1 pass^3 metric, whereas the GPT-5 series declines monotonically from GPT-5 to GPT-5.2. The same table also makes the closed/open gap clearer under stability-sensitive metrics: Qwen3.5-397B-A17B is much closer to the strongest closed models

- Table 15: Comprehensive VAB results for evaluated MLLMs and reference baselines. We report Top-1, Bot-1, and TB-1 accuracy under both ap@1 and pass^3 (%). Best result in each non-baseline column is shown in bold.

Overall Fine Art Illustration Photography Top-1 Bot-1 TB-1 Top-1 Bot-1 TB-1 Top-1 Bot-1 TB-1 Top-1 Bot-1 TB-1 Model ap@1 p^3 ap@1 p^3 ap@1 p^3 ap@1 p^3 ap@1 p^3 ap@1 p^3 ap@1 p^3 ap@1 p^3 ap@1 p^3 ap@1 p^3 ap@1 p^3 ap@1 p^3

Random Guess 37.2 6.6 37.2 6.6 27.5 5.3 41.0 8.9 41.0 8.9 36.0 8.5 31.5 4.4 31.5 4.4 19.2 3.3 36.8 5.6 36.8 5.6 23.7 3.1 Human Expert 77.7 80.1 68.9 80.4 82.7 74.7 70.3 70.4 54.4 79.9 84.0 72.4

Closed-Source Models

CLAUDE-HAIKU-4.5 43.3 18.8 50.4 26.5 32.8 12.5 44.7 19.9 49.1 27.3 40.4 19.3 32.7 13.0 41.7 12.0 18.7 5.0 49.4 21.6 58.3 36.0 34.3 10.1

- CLAUDE-SONNET-4.5 47.2 24.2 53.3 28.5 35.7 14.5 44.5 20.5 51.6 26.7 40.2 19.3 37.3 17.0 40.3 17.0 20.7 8.0 57.6 33.8 64.7 38.8 41.2 13.7

- CLAUDE-SONNET-4.6 54.8 40.2 54.4 36.8 40.6 26.5 48.4 37.3 53.2 34.2 44.7 34.2 45.3 30.0 40.3 23.0 29.3 19.0 68.8 51.1 65.9 49.6 43.9 23.0

- CLAUDE-OPUS-4.5 51.2 34.2 55.5 32.8 39.3 20.2 45.3 28.0 51.3 28.0 40.4 24.2 40.0 27.0 44.7 29.0 25.3 14.0 65.9 46.8 68.1 41.0 48.2 20.1

- CLAUDE-OPUS-4.6 53.0 35.5 53.9 31.2 38.8 20.0 48.7 27.3 50.7 27.3 43.3 23.6 42.3 31.0 42.7 26.0 25.0 17.0 65.7 48.2 65.7 39.6 43.6 18.0

DOUBAO-SEED-2.0-PRO 51.3 33.2 60.8 44.8 42.8 23.5 55.9 39.8 60.0 46.0 49.9 32.9 39.3 18.0 43.3 17.0 25.0 6.0 54.7 36.7 74.1 63.3 47.5 25.2 GEMINI-3-FLASH 52.8 28.0 60.5 38.2 40.7 15.8 51.8 23.0 60.7 40.4 46.8 20.5 46.0 22.0 47.3 15.0 26.0 8.0 58.8 38.1 69.8 52.5 44.1 15.8 GEMINI-3-PRO 55.4 35.0 55.2 33.8 40.0 22.0 52.0 32.9 55.1 32.9 46.6 29.8 47.3 28.0 44.3 19.0 28.7 14.0 65.2 42.4 63.3 45.3 40.5 18.7 GEMINI-3.1-PRO 54.8 35.0 61.1 41.5 41.8 22.2 51.8 31.1 59.4 41.6 45.1 26.1 49.7 32.0 49.0 23.0 28.3 13.0 62.1 41.7 71.7 54.7 47.5 24.5

- GPT-4.1 49.3 29.5 59.3 43.2 39.8 21.2 47.2 28.0 53.0 37.9 43.1 25.5 43.0 24.0 44.7 21.0 27.7 15.0 56.4 35.3 77.2 65.5 44.8 20.9

GPT-5 51.0 32.2 58.1 43.0 40.0 21.8 49.9 30.4 50.5 36.6 42.7 25.5 36.7 19.0 41.7 23.0 20.3 9.0 62.6 43.9 78.7 64.7 51.1 26.6

- GPT-5.1 52.3 29.5 59.1 43.8 40.2 20.0 49.9 30.4 53.8 41.6 44.5 29.8 45.7 25.0 44.3 26.0 26.7 11.0 60.0 31.7 75.8 59.0 44.8 15.1

GPT-5.1-CODEX 50.2 31.8 58.8 42.5 39.2 22.0 47.4 29.8 53.4 39.1 42.0 27.3 41.3 22.0 45.3 19.0 24.3 15.0 60.0 41.0 74.8 63.3 46.8 20.9 GPT-5.2 46.3 24.0 53.8 34.5 37.4 15.5 40.8 18.0 44.9 26.7 35.4 14.9 33.7 13.0 39.7 15.0 21.3 8.0 61.9 38.8 74.3 57.6 51.3 21.6

GROK-4.1-FAST 44.6 15.5 47.8 24.2 35.2 12.5 44.5 18.0 49.5 27.3 39.5 17.4 36.3 15.0 34.3 13.0 25.0 11.0 50.6 12.9 55.6 28.8 37.4 7.9

O4-MINI 48.2 25.0 53.7 32.2 39.8 21.5 44.7 23.6 46.4 26.7 41.0 23.0 38.0 16.0 41.7 15.0 20.7 7.0 59.5 33.1 70.7 51.1 52.3 30.2 Open-Weight Models

GLM-4.6V 43.0 17.5 51.4 25.8 34.1 11.5 42.7 18.0 47.4 24.2 39.3 18.6 43.3 19.0 41.7 14.0 27.0 8.0 43.2 15.8 63.1 36.0 33.1 5.8

KIMI-K2.5 49.4 26.8 52.5 28.2 36.8 15.0 43.5 19.9 52.2 31.7 38.5 18.0 43.0 21.0 41.0 15.0 29.0 14.0 60.9 38.8 61.2 33.8 40.5 12.2 QWEN3-VL-235B-A22B 44.9 20.5 50.2 22.8 35.2 14.0 42.4 22.4 46.8 23.0 36.9 19.3 38.3 12.0 42.7 9.0 25.7 8.0 52.5 24.5 59.5 32.4 40.0 12.2

QWEN3.5-397B-A17B 50.1 29.8 54.8 36.2 38.9 17.2 42.7 21.1 46.2 28.6 37.3 17.4 44.3 24.0 41.3 18.0 26.0 8.0 62.8 43.9 74.3 58.3 50.1 23.7

under ap@1 than under pass^3, suggesting weaker consistency across repeated evaluations. Finally, the domain columns reinforce the main-text headline that illustration is the most difficult domain, photography is generally more tractable, and fine art lies between them.

E.2 Comprehensive Reward-Model Results

- Table 16 reports comprehensive reward-model results, including Bot-1, across overall and domain-level evaluations. In contrast to the condensed main-text presentation in Table 5, this table shows that some reward models are materially stronger at rejecting poor images than at identifying the best one. In particular, the detailed Bot-1 columns make the failure mode discussed in Section 4.2.2 directly visible, especially for Q-Instruct in photography.

###### E.3 Topic-Level MLLM Results

To match the main-text MLLM table, we report topic-level results under all four metrics: Top-1 ap@1, Top-1 pass^3, TB-1 ap@1, and TB-1 pass^3. Together, these tables show that the domain-level trends in Section 4.2.1 are not driven by a single topic or a single evaluation metric. Illustration remains the hardest domain across all four views, while photography is generally more tractable but still highly uneven across topics.

Several topic effects recur across metrics. In fine art, Chinese painting and portrait-color tasks are often relatively tractable, whereas landscape-color and some sketch-based tasks become much less stable under the stricter criteria. In illustration, the remaining signal is concentrated mostly in anime-and-manga and comic tasks, while concept art, digital-and-AI art, and stylized 3D frequently collapse under pass^3. In photography, macro and sports tasks are often close to saturation for strong models, but night-andastrophotography, architecture, and some street-and-city tasks remain much less stable, especially when consistency across repeated evaluations is required.

###### Top-1 ap@1.

###### Table 16: Comprehensive reward-model results on VAB. Each image is scored independently three times and the scores are averaged before within-task ranking. We report Top-1, Bot-1, and TB-1 accuracy across overall and domain-level evaluations (%). Best result in each non-baseline column is shown in bold.

Overall Fine Art Illustration Photography Model Top-1 Bot-1 TB-1 Top-1 Bot-1 TB-1 Top-1 Bot-1 TB-1 Top-1 Bot-1 TB-1

Random Guess 37.2 37.2 27.5 41.0 41.0 36.0 31.5 31.5 19.2 36.8 36.8 23.7 Human Expert 77.7 80.1 68.9 80.4 82.7 74.7 70.3 70.4 54.4 79.9 84.0 72.4

ARTIMUSE 44.2 51.2 34.8 50.9 54.7 46.0 37.0 33.0 20.0 41.7 60.4 32.4 PEAS-AES 48.5 49.5 37.8 49.7 51.6 44.1 34.0 36.0 22.0 57.6 56.8 41.7 PEAS-AVA 49.0 48.2 37.0 50.9 49.1 44.1 25.0 31.0 13.0 64.0 59.7 46.0

Q-ALIGN 52.2 47.8 38.2 52.2 49.7 44.7 31.0 28.0 16.0 67.6 59.7 46.8

Q-INSTRUCT 38.5 51.5 31.2 37.9 39.1 34.2 32.0 36.0 21.0 43.9 77.0 35.2 REALQA-AVA 44.2 42.2 30.2 40.4 43.5 36.6 33.0 27.0 15.0 56.8 51.8 33.8

###### Table 17: Topic-level MLLM Top-1 ap@1 results on fine-art tasks. The final column reports the corresponding domain-level average (%).

Model Calligraphy Chinese Painting Ink & Wash Landscape Color Portrait Color Portrait Sketch Quick Sketch Still Life Color Still Life Sketch Avg.

Random Guess 50.0 50.0 50.0 25.2 50.0 50.0 50.0 47.7 50.0 41.0 Human Expert 89.1 94.4 90.0 66.6 85.0 90.3 88.8 81.7 86.5 80.4

Closed-Source Models

CLAUDE-HAIKU-4.5 33.3 81.5 66.7 29.8 50.0 53.5 54.2 52.8 42.9 44.7

- CLAUDE-SONNET-4.5 42.4 85.2 66.7 27.4 50.0 54.5 62.5 48.6 40.5 44.5

- CLAUDE-SONNET-4.6 54.5 74.1 50.0 39.9 83.3 48.5 62.5 48.6 47.6 48.4

- CLAUDE-OPUS-4.5 42.4 81.5 50.0 32.7 83.3 46.5 58.3 51.4 47.6 45.3

- CLAUDE-OPUS-4.6 48.5 81.5 50.0 37.5 66.7 50.5 62.5 47.2 59.5 48.7

DOUBAO-SEED-2.0-PRO 42.4 96.3 58.3 44.6 66.7 64.6 70.8 61.1 45.2 55.9 GEMINI-3-FLASH 36.4 85.2 66.7 42.9 66.7 59.6 66.7 48.6 50.0 51.8 GEMINI-3-PRO 27.3 88.9 58.3 41.7 100.0 58.6 66.7 58.3 45.2 52.0 GEMINI-3.1-PRO 24.2 81.5 41.7 44.0 100.0 60.6 58.3 50.0 59.5 51.8

- GPT-4.1 33.3 88.9 41.7 32.7 100.0 44.4 66.7 52.8 69.0 47.2

GPT-5 27.3 96.3 58.3 37.5 100.0 44.4 70.8 59.7 61.9 49.9

- GPT-5.1 33.3 92.6 50.0 35.7 83.3 49.5 70.8 54.2 69.0 49.9

GPT-5.1-CODEX 30.3 92.6 50.0 35.1 100.0 39.4 62.5 56.9 66.7 47.4 GPT-5.2 21.2 70.4 50.0 31.5 50.0 40.4 54.2 47.2 52.4 40.8 GROK-4.1-FAST 36.4 55.6 58.3 28.0 33.3 57.6 54.2 59.7 45.2 44.5 O4-MINI 18.2 81.5 58.3 26.2 83.3 43.4 62.5 72.2 52.4 44.7

Open-Weight Models

GLM-4.6V 21.2 63.0 33.3 26.8 66.7 52.5 54.2 58.3 52.4 42.7 KIMI-K2.5 30.3 85.2 50.0 27.4 100.0 46.5 62.5 50.0 52.4 43.5

QWEN3-VL-235B-A22B 15.2 85.2 50.0 26.8 66.7 48.5 66.7 47.2 57.1 42.4 QWEN3.5-397B-A17B 24.2 96.3 66.7 33.9 100.0 37.4 58.3 43.1 45.2 42.7

Tables 17, 18, and 19 report topic-level Top-1 ap@1 results for fine art, illustration, and photography, respectively.

Top-1 pass^3. Tables 20, 21, and 22 report topic-level Top-1 pass^3 results for fine art, illustration, and photography, respectively. TB-1 ap@1. Tables 23, 24, and 25 report topic-level TB-1 ap@1 results for fine art, illustration, and photography, respectively. TB-1 pass^3.

Tables 26, 27, and 28 report topic-level TB-1 pass^3 results for fine art, illustration, and photography, respectively.

Table 18: Topic-level MLLM Top-1 ap@1 results on illustration tasks. The final column reports the corresponding

domain-level average (%).

Model Anime & Manga Comic Concept Art Digital & AI Art Pixel Art Stylized 3D Avg.

Random Guess 50.0 47.5 25.0 29.3 25.0 25.0 31.5 Human Expert 87.7 82.0 66.8 65.2 62.4 71.8 70.3

Closed-Source Models

CLAUDE-HAIKU-4.5 56.4 26.7 40.9 20.3 25.4 33.3 32.7

- CLAUDE-SONNET-4.5 61.5 26.7 33.3 33.3 41.3 27.3 37.3

- CLAUDE-SONNET-4.6 71.8 63.3 28.8 37.7 46.0 45.5 45.3

- CLAUDE-OPUS-4.5 64.1 46.7 27.3 31.9 42.9 42.4 40.0

- CLAUDE-OPUS-4.6 61.5 50.0 31.8 36.2 46.0 39.4 42.3

DOUBAO-SEED-2.0-PRO 53.8 50.0 24.2 34.8 47.6 36.4 39.3 GEMINI-3-FLASH 59.0 56.7 34.8 44.9 52.4 33.3 46.0 GEMINI-3-PRO 69.2 60.0 36.4 43.5 41.3 51.5 47.3 GEMINI-3.1-PRO 71.8 60.0 37.9 42.0 55.6 42.4 49.7

- GPT-4.1 79.5 36.7 28.8 42.0 41.3 39.4 43.0

GPT-5 61.5 30.0 37.9 26.1 33.3 39.4 36.7

- GPT-5.1 69.2 40.0 43.9 39.1 46.0 39.4 45.7

GPT-5.1-CODEX 59.0 26.7 37.9 42.0 39.7 42.4 41.3 GPT-5.2 48.7 30.0 30.3 24.6 38.1 36.4 33.7 GROK-4.1-FAST 61.5 36.7 34.8 39.1 28.6 18.2 36.3 O4-MINI 56.4 33.3 36.4 30.4 42.9 30.3 38.0

Open-Weight Models

GLM-4.6V 51.3 53.3 31.8 37.7 60.3 27.3 43.3 KIMI-K2.5 71.8 56.7 36.4 36.2 39.7 30.3 43.0

QWEN3-VL-235B-A22B 69.2 33.3 30.3 30.4 39.7 36.4 38.3 QWEN3.5-397B-A17B 71.8 53.3 33.3 27.5 50.8 48.5 44.3

###### E.4 Topic-Level Reward-Model Results

To match the main-text reward-model table, we report topic-level results for both Top-1 and TB-1. As in the MLLM case, the domain-level averages in the main text mask substantial variation across topics. The topic-level reward-model tables show that relatively stronger domain-level results in photography are driven by a subset of topics, while illustration remains difficult across most topics for all six reward models.

The topic breakdown also clarifies where reward models do and do not transfer. Fine-art performance is mixed but not uniformly weak, with several models reaching competitive results on Chinese painting and portrait-color tasks. Illustration remains systematically difficult: outside anime-and-manga and, in a few cases, comic, most models stay close to random on concept art, digital-and-AI art, pixel art, and stylized 3D, especially under TB-1. Photography remains the strongest domain, but the gains are concentrated in a subset of topics and do not transfer uniformly to the harder best-and-worst selection setting.

Top-1. Tables 29, 30, and 31 report topic-level Top-1 results for fine art, illustration, and photography, respectively.

TB-1. Tables 32, 33, and 36 report topic-level TB-1 results for fine art, illustration, and photography, respectively.

###### E.5 Temperature Ablation

- Table 34 reports the full model-by-model temperature ablation referenced in Section 4.3. The overall trend is that greedy decoding modestly improves stability, but the effect size varies substantially across models

Table 19: Topic-level MLLM Top-1 ap@1 results on photography tasks. The final column reports the corresponding

domain-level average (%).

Model Architecture Food & Product Landscape Macro Night & Astro Portrait Sports Street & City Wildlife Avg.

Random Guess 37.0 36.7 35.4 33.3 33.3 41.7 33.3 38.0 39.8 36.8 Human Expert 81.6 78.0 78.9 75.0 80.0 85.0 90.0 80.3 78.9 79.9

Closed-Source Models

CLAUDE-HAIKU-4.5 40.6 60.0 49.6 83.3 0.0 66.7 100.0 47.9 66.7 49.4

- CLAUDE-SONNET-4.5 43.8 40.0 61.5 83.3 66.7 33.3 100.0 58.1 81.5 57.6

- CLAUDE-SONNET-4.6 58.3 46.7 75.6 91.7 50.0 100.0 100.0 70.1 63.0 68.8

- CLAUDE-OPUS-4.5 58.3 46.7 69.6 83.3 50.0 83.3 100.0 66.7 70.4 65.9

- CLAUDE-OPUS-4.6 59.4 66.7 65.9 83.3 50.0 100.0 100.0 65.0 74.1 65.7

DOUBAO-SEED-2.0-PRO 39.6 73.3 59.3 83.3 50.0 33.3 100.0 49.6 85.2 54.7 GEMINI-3-FLASH 52.1 86.7 60.0 83.3 33.3 0.0 100.0 58.1 66.7 58.8 GEMINI-3-PRO 58.3 73.3 65.9 83.3 66.7 66.7 100.0 66.7 63.0 65.2 GEMINI-3.1-PRO 47.9 80.0 68.1 100.0 66.7 0.0 100.0 62.4 63.0 62.1

- GPT-4.1 36.5 46.7 60.7 100.0 33.3 50.0 100.0 64.1 59.3 56.4

GPT-5 49.0 66.7 65.2 100.0 50.0 33.3 100.0 67.5 63.0 62.6

- GPT-5.1 47.9 60.0 62.2 75.0 50.0 33.3 100.0 62.4 77.8 60.0

GPT-5.1-CODEX 44.8 66.7 64.4 100.0 50.0 66.7 100.0 62.4 55.6 60.0 GPT-5.2 49.0 53.3 68.9 83.3 50.0 50.0 100.0 62.4 66.7 61.9 GROK-4.1-FAST 45.8 60.0 50.4 58.3 50.0 50.0 100.0 47.9 66.7 50.6 O4-MINI 52.1 60.0 63.7 91.7 50.0 0.0 100.0 57.3 70.4 59.5

Open-Weight Models

GLM-4.6V 35.4 53.3 39.3 58.3 16.7 0.0 100.0 48.7 63.0 43.2 KIMI-K2.5 49.0 80.0 63.7 75.0 50.0 50.0 100.0 61.5 70.4 60.9

QWEN3-VL-235B-A22B 37.5 66.7 54.1 58.3 50.0 50.0 66.7 57.3 66.7 52.5 QWEN3.5-397B-A17B 44.8 66.7 64.4 83.3 50.0 50.0 100.0 72.6 66.7 62.8

and does not alter the main conclusion that the dominant limitation is comparative judgment rather than decoding noise alone.

The gains are not uniform. Gemini 3.1 Pro and Claude Sonnet 4.5 benefit the most from greedy decoding, with especially visible improvements on the stricter TB-1 pass^3 metric. A small set of models, including Claude Sonnet 4.6, Doubao Seed 2.0 Pro, GLM-4.6V, and Qwen3.5-397B-A17B, are flat or slightly worse on at least one metric, reinforcing the point that decoding choice matters but does not dominate the benchmark outcome.

- Table 20: Topic-level MLLM Top-1 pass^3 results on fine-art tasks. The final column reports the corresponding domain-level average (%).

Model Calligraphy Chinese Painting Ink & Wash Landscape Color Portrait Color Portrait Sketch Quick Sketch Still Life Color Still Life Sketch Avg.

Random Guess 12.5 12.5 12.5 2.4 12.5 12.5 12.5 11.6 12.5 8.9 Human Expert 89.1 94.4 90.0 66.6 85.0 90.3 88.8 81.7 86.5 80.4

Closed-Source Models

CLAUDE-HAIKU-4.5 9.1 66.7 50.0 5.4 50.0 24.2 50.0 16.7 21.4 19.9 CLAUDE-SONNET-4.5 9.1 77.8 25.0 7.1 0.0 33.3 37.5 20.8 7.1 20.5 CLAUDE-SONNET-4.6 36.4 66.7 50.0 28.6 50.0 36.4 62.5 37.5 35.7 37.3

CLAUDE-OPUS-4.5 36.4 77.8 50.0 16.1 50.0 21.2 37.5 37.5 21.4 28.0 CLAUDE-OPUS-4.6 36.4 66.7 50.0 19.6 50.0 21.2 37.5 25.0 28.6 27.3

DOUBAO-SEED-2.0-PRO 18.2 88.9 50.0 30.4 50.0 42.4 50.0 45.8 35.7 39.8 GEMINI-3-FLASH 0.0 66.7 50.0 17.9 50.0 24.2 37.5 20.8 14.3 23.0 GEMINI-3-PRO 9.1 77.8 50.0 17.9 100.0 36.4 50.0 45.8 28.6 32.9 GEMINI-3.1-PRO 0.0 77.8 25.0 23.2 100.0 36.4 37.5 33.3 28.6 31.1 GPT-4.1 9.1 77.8 25.0 12.5 100.0 30.3 50.0 25.0 50.0 28.0 GPT-5 0.0 88.9 50.0 19.6 100.0 21.2 50.0 37.5 42.9 30.4 GPT-5.1 9.1 88.9 50.0 12.5 50.0 30.3 62.5 33.3 50.0 30.4 GPT-5.1-CODEX 0.0 88.9 25.0 12.5 100.0 27.3 62.5 37.5 50.0 29.8 GPT-5.2 0.0 55.6 25.0 12.5 50.0 15.2 37.5 12.5 28.6 18.0 GROK-4.1-FAST 9.1 22.2 50.0 5.4 0.0 30.3 25.0 33.3 7.1 18.0 O4-MINI 0.0 66.7 50.0 3.6 50.0 21.2 50.0 41.7 42.9 23.6

Open-Weight Models

GLM-4.6V 9.1 55.6 0.0 5.4 50.0 21.2 25.0 33.3 14.3 18.0 KIMI-K2.5 9.1 77.8 25.0 5.4 100.0 18.2 50.0 16.7 28.6 19.9

QWEN3-VL-235B-A22B 9.1 77.8 50.0 8.9 50.0 24.2 50.0 20.8 21.4 22.4 QWEN3.5-397B-A17B 18.2 88.9 50.0 16.1 100.0 6.1 37.5 16.7 14.3 21.1

- Table 21: Topic-level MLLM Top-1 pass^3 results on illustration tasks. The final column reports the corresponding domain-level average (%).

Model Anime & Manga Comic Concept Art Digital & AI Art Pixel Art Stylized 3D Avg.

Random Guess 12.5 11.4 1.6 3.5 1.6 1.6 4.4 Human Expert 87.7 82.0 66.8 65.2 62.4 71.8 70.3

Closed-Source Models

CLAUDE-HAIKU-4.5 30.8 0.0 18.2 4.3 9.5 18.2 13.0

- CLAUDE-SONNET-4.5 30.8 0.0 9.1 17.4 23.8 18.2 17.0

- CLAUDE-SONNET-4.6 61.5 60.0 4.5 26.1 28.6 27.3 30.0

- CLAUDE-OPUS-4.5 53.8 30.0 4.5 21.7 33.3 36.4 27.0

- CLAUDE-OPUS-4.6 53.8 30.0 9.1 34.8 38.1 27.3 31.0

DOUBAO-SEED-2.0-PRO 7.7 10.0 4.5 26.1 28.6 27.3 18.0 GEMINI-3-FLASH 23.1 20.0 9.1 21.7 38.1 18.2 22.0 GEMINI-3-PRO 46.2 40.0 13.6 26.1 28.6 27.3 28.0 GEMINI-3.1-PRO 53.8 30.0 22.7 21.7 38.1 36.4 32.0

- GPT-4.1 61.5 20.0 13.6 21.7 19.0 18.2 24.0

GPT-5 46.2 10.0 13.6 17.4 14.3 18.2 19.0

- GPT-5.1 46.2 20.0 18.2 21.7 23.8 27.3 25.0

GPT-5.1-CODEX 38.5 10.0 9.1 30.4 19.0 27.3 22.0 GPT-5.2 30.8 10.0 4.5 8.7 14.3 18.2 13.0 GROK-4.1-FAST 38.5 20.0 9.1 17.4 9.5 0.0 15.0 O4-MINI 23.1 10.0 13.6 13.0 23.8 9.1 16.0

Open-Weight Models

GLM-4.6V 23.1 20.0 9.1 17.4 38.1 0.0 19.0 KIMI-K2.5 53.8 30.0 13.6 21.7 9.5 9.1 21.0

QWEN3-VL-235B-A22B 46.2 0.0 4.5 4.3 9.5 18.2 12.0 QWEN3.5-397B-A17B 53.8 10.0 13.6 4.3 33.3 45.5 24.0

###### Table 22: Topic-level MLLM Top-1 pass^3 results on photography tasks. The final column reports the corresponding domain-level average (%).

Model Architecture Food & Product Landscape Macro Night & Astro Portrait Sports Street & City Wildlife Avg.

Random Guess 5.6 5.5 4.8 3.7 3.7 8.1 3.7 6.2 7.4 5.6 Human Expert 81.6 78.0 78.9 75.0 80.0 85.0 90.0 80.3 78.9 79.9

Closed-Source Models

CLAUDE-HAIKU-4.5 15.6 20.0 24.4 75.0 0.0 0.0 100.0 15.4 33.3 21.6 CLAUDE-SONNET-4.5 15.6 0.0 37.8 75.0 50.0 0.0 100.0 35.9 66.7 33.8 CLAUDE-SONNET-4.6 28.1 20.0 60.0 75.0 50.0 100.0 100.0 56.4 55.6 51.1

CLAUDE-OPUS-4.5 31.2 0.0 53.3 75.0 50.0 50.0 100.0 48.7 66.7 46.8 CLAUDE-OPUS-4.6 25.0 60.0 51.1 75.0 50.0 100.0 100.0 53.8 55.6 48.2

DOUBAO-SEED-2.0-PRO 18.8 40.0 44.4 75.0 50.0 0.0 100.0 30.8 66.7 36.7 GEMINI-3-FLASH 28.1 60.0 42.2 75.0 0.0 0.0 100.0 33.3 55.6 38.1 GEMINI-3-PRO 31.2 40.0 44.4 75.0 50.0 50.0 100.0 43.6 44.4 42.4 GEMINI-3.1-PRO 25.0 60.0 46.7 100.0 50.0 0.0 100.0 41.0 44.4 41.7 GPT-4.1 15.6 20.0 40.0 100.0 0.0 50.0 100.0 41.0 33.3 35.3 GPT-5 28.1 40.0 48.9 100.0 50.0 0.0 100.0 46.2 44.4 43.9 GPT-5.1 15.6 20.0 35.6 50.0 50.0 0.0 100.0 35.9 44.4 31.7 GPT-5.1-CODEX 25.0 40.0 48.9 100.0 50.0 50.0 100.0 35.9 44.4 41.0 GPT-5.2 25.0 20.0 48.9 75.0 50.0 0.0 100.0 35.9 44.4 38.8 GROK-4.1-FAST 6.2 20.0 8.9 25.0 50.0 0.0 100.0 12.8 33.3 12.9 O4-MINI 21.9 40.0 33.3 75.0 50.0 0.0 100.0 33.3 44.4 33.1

Open-Weight Models

GLM-4.6V 6.2 20.0 15.6 50.0 0.0 0.0 100.0 15.4 33.3 15.8 KIMI-K2.5 18.8 60.0 44.4 75.0 50.0 50.0 100.0 35.9 55.6 38.8

QWEN3-VL-235B-A22B 6.2 60.0 26.7 50.0 50.0 0.0 0.0 28.2 33.3 24.5 QWEN3.5-397B-A17B 31.2 40.0 42.2 75.0 50.0 0.0 100.0 51.3 55.6 43.9

###### Table 23: Topic-level MLLM TB-1 ap@1 results on fine-art tasks. The final column reports the corresponding domain-level average (%).

Model Calligraphy Chinese Painting Ink & Wash Landscape Color Portrait Color Portrait Sketch Quick Sketch Still Life Color Still Life Sketch Avg.

Random Guess 50.0 50.0 50.0 11.3 50.0 50.0 50.0 46.4 50.0 36.0 Human Expert 89.1 94.4 90.0 50.5 85.0 90.3 88.8 80.8 86.5 74.7

Closed-Source Models

CLAUDE-HAIKU-4.5 33.3 81.5 66.7 17.9 50.0 53.5 54.2 51.4 42.9 40.4

- CLAUDE-SONNET-4.5 42.4 85.2 66.7 15.5 50.0 54.5 62.5 47.2 40.5 40.2

- CLAUDE-SONNET-4.6 54.5 74.1 50.0 28.6 83.3 48.5 62.5 50.0 47.6 44.7

- CLAUDE-OPUS-4.5 42.4 81.5 50.0 19.0 83.3 46.5 58.3 50.0 47.6 40.4

- CLAUDE-OPUS-4.6 48.5 81.5 50.0 22.0 66.7 50.5 62.5 47.2 59.5 43.3

DOUBAO-SEED-2.0-PRO 42.4 96.3 58.3 27.4 66.7 64.6 70.8 61.1 45.2 49.9 GEMINI-3-FLASH 36.4 85.2 66.7 28.6 66.7 59.6 66.7 48.6 50.0 46.8 GEMINI-3-PRO 27.3 88.9 58.3 26.2 100.0 58.6 66.7 58.3 45.2 46.6 GEMINI-3.1-PRO 24.2 81.5 41.7 25.0 100.0 60.6 58.3 50.0 59.5 45.1

- GPT-4.1 33.3 88.9 41.7 22.0 100.0 44.4 66.7 50.0 69.0 43.1

GPT-5 27.3 96.3 58.3 17.9 100.0 44.4 70.8 56.9 61.9 42.7

- GPT-5.1 33.3 92.6 50.0 21.4 83.3 49.5 70.8 51.4 69.0 44.5

GPT-5.1-CODEX 30.3 92.6 50.0 19.6 100.0 39.4 62.5 56.9 66.7 42.0 GPT-5.2 21.2 70.4 50.0 17.3 50.0 40.4 54.2 44.4 52.4 35.4 GROK-4.1-FAST 36.4 55.6 58.3 13.1 33.3 57.6 54.2 61.1 45.2 39.5 O4-MINI 18.2 81.5 58.3 16.7 83.3 43.4 62.5 69.4 52.4 41.0

Open-Weight Models

GLM-4.6V 21.2 63.0 33.3 17.3 66.7 52.5 54.2 58.3 52.4 39.3 KIMI-K2.5 30.3 85.2 50.0 13.1 100.0 46.5 62.5 50.0 52.4 38.5

QWEN3-VL-235B-A22B 15.2 85.2 50.0 10.7 66.7 48.5 66.7 47.2 57.1 36.9 QWEN3.5-397B-A17B 24.2 96.3 66.7 18.5 100.0 37.4 58.3 43.1 45.2 37.3

- Table 24: Topic-level MLLM TB-1 ap@1 results on illustration tasks. The final column reports the corresponding domain-level average (%).

Model Anime & Manga Comic Concept Art Digital & AI Art Pixel Art Stylized 3D Avg.

Random Guess 50.0 45.8 8.3 15.6 8.3 8.3 19.2 Human Expert 87.7 79.0 42.3 49.1 43.3 49.1 54.4

Closed-Source Models

CLAUDE-HAIKU-4.5 56.4 26.7 13.6 15.9 9.5 0.0 18.7 CLAUDE-SONNET-4.5 61.5 23.3 10.6 17.4 12.7 12.1 20.7 CLAUDE-SONNET-4.6 71.8 66.7 16.7 26.1 15.9 3.0 29.3

CLAUDE-OPUS-4.5 64.1 43.3 13.6 20.3 17.5 12.1 25.3 CLAUDE-OPUS-4.6 61.5 43.3 12.1 24.6 19.0 3.0 25.0

DOUBAO-SEED-2.0-PRO 53.8 50.0 16.7 21.7 17.5 6.1 25.0 GEMINI-3-FLASH 59.0 53.3 21.2 18.8 17.5 3.0 26.0 GEMINI-3-PRO 69.2 60.0 12.1 21.7 20.6 15.2 28.7 GEMINI-3.1-PRO 71.8 60.0 7.6 21.7 25.4 9.1 28.3 GPT-4.1 79.5 33.3 9.1 30.4 12.7 21.2 27.7 GPT-5 61.5 30.0 12.1 20.3 7.9 3.0 20.3 GPT-5.1 69.2 36.7 18.2 26.1 14.3 9.1 26.7 GPT-5.1-CODEX 59.0 33.3 9.1 36.2 12.7 3.0 24.3 GPT-5.2 48.7 30.0 15.2 14.5 17.5 15.2 21.3 GROK-4.1-FAST 61.5 40.0 19.7 18.8 17.5 6.1 25.0 O4-MINI 56.4 33.3 13.6 21.7 7.9 3.0 20.7

Open-Weight Models

GLM-4.6V 51.3 50.0 13.6 20.3 28.6 15.2 27.0 KIMI-K2.5 71.8 53.3 18.2 20.3 17.5 18.2 29.0

QWEN3-VL-235B-A22B 69.2 33.3 22.7 14.5 22.2 3.0 25.7 QWEN3.5-397B-A17B 71.8 46.7 18.2 15.9 20.6 0.0 26.0

- Table 25: Topic-level MLLM TB-1 ap@1 results on photography tasks. The final column reports the corresponding domain-level average (%).

Model Architecture Food & Product Landscape Macro Night & Astro Portrait Sports Street & City Wildlife Avg.

Random Guess 24.0 23.3 20.9 16.7 16.7 33.3 16.7 26.1 30.6 23.7 Human Expert 73.8 72.0 71.3 60.0 60.0 80.0 70.0 74.4 72.2 72.4

Closed-Source Models

CLAUDE-HAIKU-4.5 21.9 60.0 32.6 41.7 0.0 66.7 33.3 35.0 66.7 34.3 CLAUDE-SONNET-4.5 33.3 33.3 40.7 41.7 33.3 16.7 33.3 47.9 55.6 41.2 CLAUDE-SONNET-4.6 34.4 26.7 41.5 58.3 50.0 100.0 0.0 50.4 55.6 43.9

CLAUDE-OPUS-4.5 42.7 46.7 44.4 66.7 50.0 83.3 66.7 50.4 59.3 48.2 CLAUDE-OPUS-4.6 37.5 40.0 40.0 58.3 50.0 66.7 66.7 47.9 51.9 43.6

DOUBAO-SEED-2.0-PRO 36.5 53.3 43.7 66.7 50.0 33.3 66.7 49.6 85.2 47.5 GEMINI-3-FLASH 36.5 80.0 36.3 66.7 50.0 33.3 66.7 47.9 63.0 44.1 GEMINI-3-PRO 36.5 73.3 27.4 58.3 66.7 66.7 33.3 44.4 66.7 40.5 GEMINI-3.1-PRO 32.3 66.7 45.2 91.7 50.0 33.3 100.0 50.4 66.7 47.5 GPT-4.1 30.2 20.0 40.7 100.0 50.0 50.0 33.3 57.3 51.9 44.8 GPT-5 40.6 73.3 45.2 91.7 50.0 66.7 66.7 57.3 55.6 51.1 GPT-5.1 33.3 73.3 37.8 75.0 66.7 33.3 100.0 50.4 59.3 44.8 GPT-5.1-CODEX 30.2 66.7 43.7 100.0 33.3 66.7 33.3 53.0 59.3 46.8 GPT-5.2 39.6 26.7 51.1 83.3 33.3 50.0 66.7 56.4 74.1 51.3 GROK-4.1-FAST 27.1 46.7 38.5 41.7 33.3 16.7 0.0 39.3 63.0 37.4 O4-MINI 43.8 53.3 52.6 91.7 50.0 16.7 100.0 52.1 66.7 52.3

Open-Weight Models

GLM-4.6V 28.1 40.0 28.1 33.3 33.3 0.0 66.7 39.3 48.1 33.1 KIMI-K2.5 29.2 80.0 39.3 33.3 16.7 0.0 66.7 45.3 59.3 40.5

QWEN3-VL-235B-A22B 28.1 53.3 36.3 91.7 33.3 66.7 66.7 42.7 51.9 40.0 QWEN3.5-397B-A17B 31.2 66.7 49.6 83.3 50.0 66.7 0.0 57.3 66.7 50.1

- Table 26: Topic-level TB-1 pass^3 results on fine-art tasks. The final column reports the corresponding domainlevel average (%).

Model Calligraphy Chinese Painting Ink & Wash Landscape Color Portrait Color Portrait Sketch Quick Sketch Still Life Color Still Life Sketch Avg.

Random Guess 12.5 12.5 12.5 1.4 12.5 12.5 12.5 11.5 12.5 8.5 Human Expert 89.1 94.4 90.0 50.5 85.0 90.3 88.8 80.8 86.5 74.7

Closed-Source Models

CLAUDE-HAIKU-4.5 9.1 66.7 50.0 3.6 50.0 24.2 50.0 16.7 21.4 19.3 CLAUDE-SONNET-4.5 9.1 77.8 25.0 3.6 0.0 33.3 37.5 20.8 7.1 19.3 CLAUDE-SONNET-4.6 36.4 66.7 50.0 19.6 50.0 36.4 62.5 37.5 35.7 34.2

CLAUDE-OPUS-4.5 36.4 77.8 50.0 5.4 50.0 21.2 37.5 37.5 21.4 24.2 CLAUDE-OPUS-4.6 36.4 66.7 50.0 8.9 50.0 21.2 37.5 25.0 28.6 23.6

DOUBAO-SEED-2.0-PRO 18.2 88.9 50.0 10.7 50.0 42.4 50.0 45.8 35.7 32.9 GEMINI-3-FLASH 0.0 66.7 50.0 10.7 50.0 24.2 37.5 20.8 14.3 20.5 GEMINI-3-PRO 9.1 77.8 50.0 8.9 100.0 36.4 50.0 45.8 28.6 29.8 GEMINI-3.1-PRO 0.0 77.8 25.0 8.9 100.0 36.4 37.5 33.3 28.6 26.1 GPT-4.1 9.1 77.8 25.0 5.4 100.0 30.3 50.0 25.0 50.0 25.5 GPT-5 0.0 88.9 50.0 5.4 100.0 21.2 50.0 37.5 42.9 25.5 GPT-5.1 9.1 88.9 50.0 10.7 50.0 30.3 62.5 33.3 50.0 29.8 GPT-5.1-CODEX 0.0 88.9 25.0 5.4 100.0 27.3 62.5 37.5 50.0 27.3 GPT-5.2 0.0 55.6 25.0 3.6 50.0 15.2 37.5 12.5 28.6 14.9 GROK-4.1-FAST 9.1 22.2 50.0 3.6 0.0 30.3 25.0 33.3 7.1 17.4 O4-MINI 0.0 66.7 50.0 1.8 50.0 21.2 50.0 41.7 42.9 23.0

Open-Weight Models

GLM-4.6V 9.1 55.6 0.0 7.1 50.0 21.2 25.0 33.3 14.3 18.6 KIMI-K2.5 9.1 77.8 25.0 0.0 100.0 18.2 50.0 16.7 28.6 18.0

QWEN3-VL-235B-A22B 9.1 77.8 50.0 0.0 50.0 24.2 50.0 20.8 21.4 19.3 QWEN3.5-397B-A17B 18.2 88.9 50.0 5.4 100.0 6.1 37.5 16.7 14.3 17.4

- Table 27: Topic-level TB-1 pass^3 results on illustration tasks. The final column reports the corresponding domainlevel average (%).

Model Anime & Manga Comic Concept Art Digital & AI Art Pixel Art Stylized 3D Avg.

Random Guess 12.5 11.3 0.1 2.2 0.1 0.1 3.3 Human Expert 87.7 79.0 42.3 49.1 43.3 49.1 54.4

Closed-Source Models

CLAUDE-HAIKU-4.5 30.8 0.0 0.0 4.3 0.0 0.0 5.0 CLAUDE-SONNET-4.5 30.8 0.0 4.5 8.7 4.8 0.0 8.0

CLAUDE-SONNET-4.6 61.5 60.0 9.1 13.0 0.0 0.0 19.0 CLAUDE-OPUS-4.5 53.8 30.0 4.5 13.0 0.0 0.0 14.0 CLAUDE-OPUS-4.6 53.8 30.0 0.0 21.7 9.5 0.0 17.0

DOUBAO-SEED-2.0-PRO 7.7 10.0 0.0 13.0 4.8 0.0 6.0 GEMINI-3-FLASH 23.1 20.0 0.0 8.7 4.8 0.0 8.0

GEMINI-3-PRO 46.2 40.0 0.0 13.0 4.8 0.0 14.0 GEMINI-3.1-PRO 53.8 30.0 0.0 13.0 0.0 0.0 13.0

GPT-4.1 61.5 20.0 0.0 21.7 0.0 0.0 15.0 GPT-5 46.2 10.0 0.0 8.7 0.0 0.0 9.0 GPT-5.1 46.2 20.0 0.0 8.7 4.8 0.0 11.0

GPT-5.1-CODEX 38.5 20.0 0.0 26.1 9.5 0.0 15.0

GPT-5.2 30.8 10.0 4.5 4.3 4.8 0.0 8.0

GROK-4.1-FAST 38.5 20.0 9.1 8.7 0.0 0.0 11.0

O4-MINI 23.1 10.0 4.5 8.7 0.0 0.0 7.0 Open-Weight Models

GLM-4.6V 23.1 20.0 4.5 8.7 0.0 0.0 8.0

KIMI-K2.5 53.8 30.0 4.5 8.7 4.8 0.0 14.0 QWEN3-VL-235B-A22B 46.2 0.0 4.5 0.0 4.8 0.0 8.0

QWEN3.5-397B-A17B 53.8 10.0 0.0 0.0 0.0 0.0 8.0

- Table 28: Topic-level TB-1 pass^3 results on photography tasks. The final column reports the corresponding domain-level average (%).

Model Architecture Food & Product Landscape Macro Night & Astro Portrait Sports Street & City Wildlife Avg.

Random Guess 3.1 2.9 2.1 0.5 0.5 6.5 0.5 3.9 5.8 3.1 Human Expert 73.8 72.0 71.3 60.0 60.0 80.0 70.0 74.4 72.2 72.4

Closed-Source Models

CLAUDE-HAIKU-4.5 3.1 40.0 4.4 25.0 0.0 0.0 0.0 12.8 33.3 10.1 CLAUDE-SONNET-4.5 9.4 0.0 13.3 25.0 0.0 0.0 0.0 15.4 33.3 13.7 CLAUDE-SONNET-4.6 12.5 0.0 17.8 50.0 50.0 100.0 0.0 28.2 44.4 23.0

CLAUDE-OPUS-4.5 18.8 20.0 15.6 25.0 50.0 50.0 0.0 17.9 44.4 20.1 CLAUDE-OPUS-4.6 12.5 0.0 13.3 25.0 50.0 50.0 0.0 23.1 33.3 18.0

DOUBAO-SEED-2.0-PRO 25.0 20.0 17.8 50.0 50.0 0.0 0.0 25.6 55.6 25.2 GEMINI-3-FLASH 9.4 40.0 8.9 25.0 50.0 0.0 0.0 17.9 44.4 15.8 GEMINI-3-PRO 9.4 60.0 8.9 50.0 50.0 50.0 0.0 20.5 44.4 18.7 GEMINI-3.1-PRO 9.4 60.0 17.8 75.0 50.0 0.0 100.0 28.2 44.4 24.5 GPT-4.1 9.4 0.0 8.9 100.0 50.0 50.0 0.0 35.9 22.2 20.9 GPT-5 9.4 40.0 22.2 75.0 50.0 0.0 0.0 35.9 44.4 26.6 GPT-5.1 12.5 20.0 8.9 25.0 50.0 0.0 100.0 12.8 44.4 15.1 GPT-5.1-CODEX 3.1 40.0 13.3 100.0 0.0 50.0 0.0 30.8 33.3 20.9 GPT-5.2 9.4 0.0 17.8 75.0 0.0 0.0 0.0 25.6 66.7 21.6

GROK-4.1-FAST 3.1 0.0 6.7 0.0 0.0 0.0 0.0 10.3 33.3 7.9

O4-MINI 9.4 40.0 37.8 75.0 50.0 0.0 100.0 28.2 44.4 30.2 Open-Weight Models

GLM-4.6V 0.0 20.0 2.2 0.0 0.0 0.0 0.0 10.3 22.2 5.8

KIMI-K2.5 3.1 60.0 6.7 25.0 0.0 0.0 0.0 15.4 33.3 12.2 QWEN3-VL-235B-A22B 3.1 20.0 11.1 75.0 0.0 0.0 0.0 12.8 22.2 12.2

QWEN3.5-397B-A17B 15.6 20.0 17.8 50.0 50.0 0.0 0.0 30.8 44.4 23.7

- Table 29: Topic-level reward-model Top-1 results on fine-art tasks. The final column reports the corresponding domain-level average (%).

Model Calligraphy Chinese Painting Ink & Wash Landscape Color Portrait Color Portrait Sketch Quick Sketch Still Life Color Still Life Sketch Avg.

Random Guess 50.0 50.0 50.0 25.2 50.0 50.0 50.0 47.7 50.0 41.0 Human Expert 89.1 94.4 90.0 66.6 85.0 90.3 88.8 81.7 86.5 80.5 Reward Models

ARTIMUSE 54.5 88.9 50.0 39.3 50.0 54.5 50.0 50.0 64.3 50.9 PEAS-AES 54.5 44.4 50.0 30.4 100.0 72.7 50.0 62.5 42.9 49.7 PEAS-AVA 36.4 66.7 50.0 41.1 100.0 54.5 62.5 58.3 57.1 50.9

Q-ALIGN 63.6 66.7 50.0 39.3 100.0 48.5 50.0 66.7 64.3 52.2 Q-INSTRUCT 9.1 44.4 75.0 21.4 100.0 57.6 50.0 41.7 42.9 37.9

REALQA-AVA 36.4 66.7 50.0 21.4 100.0 48.5 50.0 45.8 57.1 40.4

- Table 30: Topic-level reward-model Top-1 results on illustration tasks. The final column reports the corresponding domain-level average (%).

Model Anime & Manga Comic Concept Art Digital & AI Art Pixel Art Stylized 3D Avg.

Random Guess 50.0 47.5 25.0 29.3 25.0 25.0 31.5 Human Expert 87.7 82.0 66.8 65.2 62.4 71.8 70.3 Reward Models

ARTIMUSE 53.8 60.0 22.7 39.1 33.3 27.3 37.0 PEAS-AES 30.8 20.0 22.7 43.5 47.6 27.3 34.0 PEAS-AVA 30.8 20.0 9.1 30.4 42.9 9.1 25.0

Q-ALIGN 38.5 10.0 36.4 21.7 38.1 36.4 31.0

Q-INSTRUCT 46.2 30.0 18.2 34.8 42.9 18.2 32.0 REALQA-AVA 38.5 50.0 45.5 13.0 38.1 18.2 33.0

- Table 31: Topic-level reward-model Top-1 results on photography tasks. The final column reports the corresponding domain-level average (%).

Model Architecture Food & Product Landscape Macro Night & Astro Portrait Sports Street & City Wildlife Avg.

Random Guess 37.0 36.7 35.4 33.3 33.3 41.7 33.3 38.0 39.8 36.8 Human Expert 81.6 78.0 78.9 75.0 80.0 85.0 90.0 80.3 78.9 79.9 Reward Models

ARTIMUSE 25.0 60.0 46.7 50.0 100.0 50.0 0.0 38.5 66.7 41.7 PEAS-AES 50.0 80.0 57.8 100.0 50.0 0.0 0.0 61.5 55.6 57.6 PEAS-AVA 56.2 80.0 68.9 100.0 50.0 0.0 0.0 66.7 55.6 64.0

Q-ALIGN 62.5 80.0 77.8 75.0 50.0 100.0 100.0 59.0 55.6 67.6 Q-INSTRUCT 37.5 40.0 42.2 25.0 50.0 0.0 0.0 51.3 66.7 43.9

REALQA-AVA 46.9 60.0 57.8 100.0 50.0 50.0 100.0 59.0 55.6 56.8

Table 32: Topic-level reward-model TB-1 results on fine-art tasks. The final column reports the corresponding

domain-level average (%).

Model Calligraphy Chinese Painting Ink & Wash Landscape Color Portrait Color Portrait Sketch Quick Sketch Still Life Color Still Life Sketch Avg.

Random Guess 50.0 50.0 50.0 11.3 50.0 50.0 50.0 46.4 50.0 36.0 Human Expert 89.1 94.4 90.0 50.5 85.0 90.3 88.8 80.8 86.5 74.7 Reward Models

ARTIMUSE 54.5 88.9 50.0 25.0 50.0 54.5 50.0 50.0 64.3 46.0 PEAS-AES 54.5 44.4 50.0 16.1 100.0 72.7 50.0 58.3 42.9 44.1 PEAS-AVA 36.4 66.7 50.0 23.2 100.0 54.5 62.5 54.2 57.1 44.1

Q-ALIGN 63.6 66.7 50.0 19.6 100.0 48.5 50.0 62.5 64.3 44.7 Q-INSTRUCT 9.1 44.4 75.0 10.7 100.0 57.6 50.0 41.7 42.9 34.2

REALQA-AVA 36.4 66.7 50.0 12.5 100.0 48.5 50.0 41.7 57.1 36.6

- Table 33: Topic-level reward-model TB-1 results on illustration tasks. The final column reports the corresponding domain-level average (%).

Model Anime & Manga Comic Concept Art Digital & AI Art Pixel Art Stylized 3D Avg.

Random Guess 50.0 45.8 8.3 15.6 8.3 8.3 19.2 Human Expert 87.7 79.0 42.3 49.1 43.3 49.1 54.4 Reward Models

ARTIMUSE 53.8 60.0 4.5 13.0 4.8 18.2 20.0 PEAS-AES 30.8 20.0 9.1 30.4 28.6 9.1 22.0 PEAS-AVA 30.8 20.0 4.5 17.4 9.5 0.0 13.0

Q-ALIGN 38.5 10.0 13.6 13.0 19.1 0.0 16.0

Q-INSTRUCT 46.2 30.0 9.1 21.7 19.1 9.1 21.0 REALQA-AVA 38.5 50.0 13.6 4.3 4.8 0.0 15.0

- Table 34: Effect of decoding temperature on MLLM performance. We compare default decoding (t=1) with greedy decoding (t=0); ∆ denotes the change from t=1 to t=0 (%).

Top-1 p^3 TB-1 p^3

Model t=1 t=0 ∆ t=1 t=0 ∆

Closed-Source Models

CLAUDE-HAIKU-4.5 18.8 21.2 +2.4 12.5 17.2 +4.7

- CLAUDE-SONNET-4.5 24.2 27.3 +3.1 14.5 20.5 +6.0

- CLAUDE-SONNET-4.6 40.2 39.2 −1.0 26.5 26.8 +0.3

- CLAUDE-OPUS-4.5 34.2 35.2 +1.0 20.2 22.5 +2.3

- CLAUDE-OPUS-4.6 35.5 36.5 +1.0 20.0 21.5 +1.5

DOUBAO-SEED-2.0-PRO 33.2 32.8 −0.4 23.5 22.5 −1.0 GEMINI-3-FLASH 28.0 29.5 +1.5 15.8 21.2 +5.4 GEMINI-3-PRO 35.0 36.2 +1.2 22.0 24.8 +2.8 GEMINI-3.1-PRO 35.0 40.0 +5.0 22.2 27.8 +5.6 GPT-4.1 29.5 30.8 +1.3 21.2 22.5 +1.3 GROK-4.1-FAST 15.5 16.8 +1.3 12.5 13.8 +1.3

Open-Weight Models

GLM-4.6V 17.5 16.5 −1.0 11.5 12.2 +0.7 KIMI-K2.5 26.8 28.2 +1.4 15.0 15.5 +0.5

QWEN3-VL-235B-A22B 20.5 20.8 +0.3 14.0 15.8 +1.8 QWEN3.5-397B-A17B 29.8 29.2 −0.6 17.2 18.5 +1.3

Average 28.2 29.4 +1.1 17.9 20.2 +2.3

- Table 35: TB-1 results by candidate set size k for the evaluated MLLMs and reference baselines. For each set size, we report pass^3 and ap@1 (%).

k = 2 k = 3 k = 4 k = 5 k = 6

Model p^3 ap@1 p^3 ap@1 p^3 ap@1 p^3 ap@1 p^3 ap@1

Random Guess 12.5 50.0 0.5 16.7 0.1 8.3 0.0 5.0 0.0 3.3 Human Expert 87.1 69.3 43.6 45.9 40.0

Claude Haiku 4.5 24.8 49.7 8.1 32.1 0.0 12.7 0.0 6.9 0.0 0.0

- Claude Sonnet 4.5 27.3 53.1 9.9 37.5 2.2 12.0 0.0 7.8 0.0 0.0

- Claude Sonnet 4.6 47.3 58.2 17.1 39.0 6.7 17.2 8.8 22.5 0.0 0.0

Claude Opus 4.5 37.0 55.2 15.3 45.0 3.4 13.5 0.0 12.7 0.0 0.0 Claude Opus 4.6 37.6 58.0 9.9 38.4 6.7 13.9 2.9 13.7 0.0 0.0

Doubao Seed 2.0 Pro 38.8 59.8 20.7 43.8 5.6 17.6 5.9 24.5 0.0 0.0 Gemini 3 Flash 28.5 57.6 11.7 42.0 3.4 16.9 0.0 17.6 0.0 0.0

Gemini 3 Pro 43.6 61.4 12.6 34.8 2.2 16.1 0.0 16.7 0.0 0.0 Gemini 3.1 Pro 37.6 58.8 22.5 45.6 1.1 16.1 2.9 13.7 0.0 33.3

GLM-4.6V 24.2 50.7 1.8 29.1 3.4 19.9 2.9 7.8 0.0 0.0 GPT-4.1 37.0 55.6 18.9 43.2 2.2 15.7 2.9 16.7 0.0 0.0 GPT-5 33.9 54.1 27.0 51.1 0.0 11.6 2.9 10.8 0.0 0.0 GPT-5.1 37.0 57.4 13.5 42.3 1.1 15.4 8.8 15.7 0.0 0.0 GPT-5.1 Codex 35.8 52.5 19.8 45.9 6.7 16.1 2.9 14.7 0.0 0.0 GPT-5.2 23.0 47.1 18.9 48.6 3.4 14.6 0.0 14.7 0.0 0.0

Grok 4.1 Fast 26.1 53.1 4.5 33.6 2.2 14.2 0.0 8.8 0.0 0.0 Kimi K2.5 29.7 53.3 8.1 36.3 2.2 15.4 0.0 15.7 0.0 0.0 o4-mini 30.9 53.1 30.6 53.2 1.1 9.7 0.0 11.8 0.0 0.0

Qwen3-VL-235B 26.7 50.3 9.0 37.2 2.2 14.6 0.0 9.8 0.0 0.0 Qwen3.5-397B 28.5 52.5 18.0 45.6 1.1 14.2 2.9 15.7 0.0 33.3

- Table 36: Topic-level reward-model TB-1 results on photography tasks. The final column reports the corresponding domain-level average (%).

Model Architecture Food & Product Landscape Macro Night & Astro Portrait Sports Street & City Wildlife Avg.

Random Guess 24.0 23.3 20.9 16.7 16.7 33.3 16.7 26.1 30.6 23.7 Human Expert 73.8 72.0 71.3 60.0 60.0 80.0 70.0 74.4 72.2 72.5 Reward Models

ARTIMUSE 15.6 40.0 28.9 50.0 100.0 50.0 0.0 35.9 66.7 32.4 PEAS-AES 37.5 60.0 42.2 75.0 0.0 0.0 0.0 46.2 33.3 41.7 PEAS-AVA 40.6 40.0 46.7 100.0 0.0 0.0 0.0 51.3 44.4 46.0

Q-ALIGN 46.9 40.0 44.4 75.0 50.0 100.0 0.0 46.2 44.4 46.8 Q-INSTRUCT 31.2 20.0 31.1 0.0 50.0 0.0 0.0 43.6 66.7 35.2

REALQA-AVA 25.0 60.0 24.4 50.0 50.0 50.0 0.0 41.0 55.6 33.8

###### E.6 Results by Candidate Set Size

- Table 35 provides the full breakdown by candidate set size. Consistent with the summary in the main text, performance drops sharply as k increases, and the decline is especially severe under the stricter TB-1 pass^3 criterion. The table also shows that the gap between models and human experts widens noticeably on larger candidate sets. F Human Study Details

###### F.1 Detailed Setup and Metric Definitions

This subsection provides the full metric definitions summarized in Section 2.

Eight expert annotators evaluate identical images under both protocols. Under scoring, each annotator independently assigns an absolute score si,a ∈ [0,10] to each image. Under ranking, annotators provide a complete within-task ordering when presented with groups of size n ∈ {2,3,4,5}. We operationalize “more scientifically sound” along three dimensions: fidelity to human comparative preference, reproducibility across annotators, and sharpness of the resulting label signal.

The study comprises 119 homogeneous-content tasks drawn directly from the curated benchmark, where all candidates share the same underlying content, and 107 heterogeneous-content tasks formed by regrouping 370 images from distinct homogeneous task groups while preserving topic consistency. Each image appears in at most one regrouped task. Examples of both settings are given in Appendix F.3.

Fidelity. We ask whether the ranking induced by absolute scores matches the direct ranking produced by the same annotator. Let ρsamea and ρdiffa denote annotator a’s direct rankings for homogeneous-content and heterogeneous-content tasks, and let σascore denote the ranking induced by the absolute scores. We report Kendall’s τ (Kendall, 1938),

C − D (n2)

,

τ =

where C and D are the numbers of concordant and discordant pairs. We compute τ for each annotator within each task, average across annotators to obtain a task-level value, and report the mean and standard deviation across tasks. We additionally report a top-1 self-consistency (SC) rate: the fraction of annotator–task pairs for which σascore and the direct ranking select the same top image.

Reproducibility. We define task-level accuracy acc as the fraction of tasks on which at least five of the eight annotators agree on the relevant label, and report accbest, accworst, and accbest&worst for both protocols.

###### Heterogeneous-Content Example

###### Homogeneous-Content Example

Topic: Digital Illustration / Pixel Art Group size: n = 4 Construction: Candidates are regrouped from different source tasks while preserving topic consistency.

Topic: Photography / Architecture Group size: n = 4 Construction: All four candidates come from the same source task and share the same underlying scene content.

|A<br><br>source task 014 alley scene|
|---|
|C<br><br>source task 081 market scene|

|B<br><br>source task 037 rooftop scene|
|---|
|D<br><br>source task 104 tower scene|

|A<br><br>same facade warm light, clean framing|
|---|
|C<br><br>same facade tighter crop, stronger depth|

|B<br><br>same facade flat light, weaker balance|
|---|
|D<br><br>same facade cluttered background|

Scoring protocol. The same images are scored one by one, without seeing the regrouped set. Ranking protocol. Annotators jointly compare the regrouped candidates and produce a full ordering.

Scoring protocol. Each image is shown independently and receives a separate score in [0,10]. Ranking protocol. The full set is shown jointly and ranked from best to worst.

Figure 6: Schematic examples of the two settings used in the human study. Left: a homogeneous-content task, where all candidates come from the same source task and share the same underlying content. Right: a heterogeneous-content task, where candidates are regrouped from different source tasks. In both settings, the scoring protocol rates images independently, whereas the ranking protocol orders the full set jointly.

Distinguishability. Restricting to the 119 homogeneous-content tasks, where entropy reflects concentration of aesthetic preference rather than content mismatch, we measure the entropy of the eight annotators’ top-1 selections (Shannon, 1948),

H = −∑

pi log2 pi,

i

where pi is the fraction of annotators selecting image i as the top image in a task. Because the maximum entropy depends on task size n, we report the unnormalized quantity so that lower values directly indicate stronger concentration of top-choice judgments.

###### F.2 Extended Interpretation

Philosophical lens. Our results echo the long-standing debate between Aesthetic Absolutism, which posits a context-free, intrinsic standard of beauty, and Aesthetic Relativism, which holds that judgments arise from comparison and context. Modern philosophical aesthetics has largely moved toward the relational view: people can reliably tell which image they prefer when alternatives are juxtaposed.

Statistical lens. Scoring introduces an additional measurement layer. Annotators must first form an internal judgment and then translate it onto a numeric scale, after which pairwise preference is inferred from score differences. This pipeline adds calibration variance, scale-use heterogeneity, and extra measurement noise. Ranking measures the relative decision of interest more directly, which helps explain its higher consistency and sharper signal.

Cognitive lens. Comparative judgment is a natural human operation. People can often say which of two images feels better before they can specify by how much. Absolute scoring is cognitively heavier because it requires maintaining an internal reference scale, calibrating intervals, and converting a qualitative impression into a number. That added abstraction likely makes scoring less stable than direct comparison.

- F.3 Task Examples Figure 6 provides schematic examples of the two task settings used in the human study.
- F.4 Annotation Protocol

Scoring task. Under the scoring protocol, each annotator independently assigned an absolute score si,a ∈ [0,10] to each of the 400 images in the base image pool. Images were presented individually, without access to the other members of the same task group. The resulting scores were later reused to derive score-derived rankings for both the 119 homogeneous-content tasks and the 107 heterogeneous-content regroupings. Domain-specific single-image scoring rubrics are given below.

###### Fine Art Single-Image Scoring Rubric

For a single fine art painting, evaluate the degree of artistic completeness and maturity of aesthetic expression based solely on the image itself, and assign an overall score from 0 to 10, accurate to one decimal place.

Reference Scoring Criteria

- 1. Composition and Visual Order

- • Is the primary subject clearly defined and visually grounded?
- • Does the composition feel stable or intentionally dynamic?
- • Is the spatial arrangement natural, without unnecessary congestion or emptiness?

- 2. Color and Tonal Relationships

- • Are color relationships harmonious and layered?
- • Is there effective control of warmth/coolness and light/dark contrast?
- • Does the work avoid muddiness, dullness, or scattered color impressions?

- 3. Technical Control

- • Are the brushstrokes confident, fluid, and rhythmic?
- • Are there signs of hesitation, redundancy, or ineffective marks?
- • Does technique serve the image rather than draw attention to itself?

- 4. Degree of Completion

- • Are there areas that appear unfinished or perfunctory?
- • Are edges and transitions handled naturally?
- • Does the work feel resolved, at a point where the artist could reasonably stop?

- 5. Artistic Expression and Overall Character

- • Is the mood or atmosphere focused and coherent?
- • Does the work convey a unified overall character or presence?
- • Does it demonstrate strong artistic persuasiveness?

- 6. Overall Score

• Based on all criteria above, assign an overall score from 0 to 10, accurate to one decimal place. Should Not Be Considered

- • Personal preference for subject matter.
- • Artist background, intent, or any contextual information beyond what is visible in the image itself.

###### Photography Single-Image Scoring Rubric

For a single photograph, evaluate its overall quality in terms of photographic language and visual expression, based solely on the image itself, and assign an overall score from 0 to 10, accurate to one decimal place.

Reference Scoring Criteria

1. Composition and Image Structure

- • Is the subject clearly emphasized?
- • Does the frame feel balanced or intentionally dynamic?
- • Is the background clean and supportive of the subject?

###### 2. Quality of Light

- • Is the light natural or expressive?
- • Are highlights and shadows well controlled?
- • Does the image avoid crushed blacks or blown highlights?

###### 3. Sharpness and Focus

- • Is focus accurate where it matters most?
- • Is the relationship between sharpness and blur appropriate?
- • Are there any noticeable technical flaws?

###### 4. Color or Tonal Rendering

- • Is white balance or color temperature accurate or clearly intentional?
- • Are colors or tonal values consistent and unified?
- • Does the image avoid unwanted color casts or muddy tones?

###### 5. Moment and Emotional Expression

- • Does the image capture a meaningful moment?
- • Is the emotional or narrative content clear and compelling?
- • Does the image convey a strong sense of photographic presence?

###### 6. Overall Score

• Based on all criteria above, assign an overall score from 0 to 10, accurate to one decimal place. Should Not Be Considered

- • The personal appeal or attractiveness of the subject.
- • Assumptions about shooting difficulty.
- • Assumptions about post-processing effort.

###### Digital Illustration Single-Image Scoring Rubric

For a single digital illustration, evaluate its overall quality in terms of visual design and professional illustration standards, based solely on the image itself, and assign an overall score from 0 to 10, accurate to one decimal place.

Reference Scoring Criteria

1. Visual Focus and Information Clarity

- • Is the visual focal point clear and well defined?
- • Does the visual flow guide the viewer smoothly?
- • Is there minimal visual noise or competition for attention?

###### 2. Form and Proportion

- • Are the structures of figures or objects accurate?
- • Are poses and silhouettes aesthetically refined?
- • Does the image avoid stiffness, distortion, or unnatural forms?

###### 3. Color Design

- • Is the color palette unified and thematically coherent?
- • Are value and lighting hierarchies clear?
- • Is contrast and visual rhythm used effectively?

###### 4. Detail Control and Finish

- • Are details handled appropriately, neither excessive nor careless?
- • Are materials and lighting consistent throughout the image?
- • Are edges, masks, and gradients clean and well controlled?

###### 5. Stylistic Consistency and Purpose Fit

- • Does the illustration align with its intended stylistic direction?
- • Is quality consistent within a unified style?
- • Is the image suitable for its potential use case?

###### 6. Overall Score

• Based on all criteria above, assign an overall score from 0 to 10, accurate to one decimal place. Should Not Be Considered

- • Personal preference for illustration style.
- • Software or tools used.
- • Hypothetical or imagined client requirements.

When converting absolute scores into induced rankings, no exact within-task score ties were observed in the collected annotations, so no tie-breaking rule was required.

Ranking task. Under the ranking protocol, each annotator provided a complete best-to-worst ordering of all n images presented simultaneously within a group. Annotators were instructed to rank the images by overall aesthetic quality using the following domain-specific comparative rubrics.

###### Fine Art Ranking Rubric

For a set of fine art paintings that share the same style, subject matter, and medium, and differ only subtly in execution, rank the images from best to worst according to artistic completeness and maturity of aesthetic expression. Reference Comparison Criteria

###### 1. Composition and Visual Order

- • Is the primary subject more clearly defined?
- • Does the composition feel more stable or intentionally dynamic?
- • Is the spatial arrangement natural, without unnecessary congestion or emptiness?

###### 2. Color and Tonal Relationships

- • Are the color relationships more harmonious or layered?
- • Is there stronger control over warmth/coolness and light/dark contrast?

- • Does the work avoid muddiness, dullness, or scattered color impressions?

###### 3. Technical Control

- • Are the brushstrokes more confident, fluid, and rhythmic?
- • Are there signs of hesitation, redundancy, or ineffective marks?
- • Does the technique serve the image rather than draw attention to itself?

###### 4. Degree of Completion

- • Are there any areas that appear unfinished or perfunctory?
- • Are edges and transitions handled naturally?
- • Does the work feel resolved, at a point where the artist could reasonably stop?

###### 5. Artistic Expression and Overall Character

- • Is the mood or atmosphere more focused and coherent?
- • Does the work convey a stronger overall character or presence?
- • Does it have greater artistic persuasiveness?

###### 6. Final Selection

• Considering the set as a whole, determine the relative ordering of the images from strongest to weakest.

###### Should Not Be Considered

- • Personal preference for subject matter.
- • Any contextual or background information beyond what is visible in the image itself.

###### Photography Ranking Rubric

For a set of photographs taken of similar subjects, in similar scenes, and under closely matched conditions, rank the images from best to worst in terms of photographic language and visual expression. Reference Comparison Criteria

###### 1. Composition and Image Structure

- • Is the subject more clearly emphasized?
- • Is the frame more balanced or intentionally dynamic?
- • Is the background cleaner and more supportive of the subject?

###### 2. Quality of Light

- • Is the light more natural or more expressive?
- • Are highlights and shadows better controlled?
- • Does the image avoid crushed blacks or blown highlights?

###### 3. Sharpness and Focus

- • Is focus more accurate where it matters?
- • Is the relationship between sharpness and blur more appropriate?
- • Are there any technical errors that detract from the image?

###### 4. Color or Tonal Rendering

- • Is white balance or color temperature more accurate or more intentional?
- • Are colors or tonal values more consistent and unified?
- • Does the image avoid unwanted color casts or muddy tones?

###### 5. Moment and Emotional Expression

- • Does the image capture a stronger or more meaningful moment?
- • Is the emotional or narrative content more compelling?
- • Does it carry a stronger sense of photographic presence?

###### 6. Final Selection

• Considering the set as a whole, determine the relative ordering of the photographs from

strongest to weakest. Should Not Be Considered

- • The personal appeal or attractiveness of the subject.
- • Assumptions about shooting difficulty.
- • Assumptions about post-processing effort.

###### Digital Illustration Ranking Rubric

For a set of digital illustrations with similar purpose, style, and level of completion, rank the images from best to worst according to visual design and professional illustration quality. Reference Comparison Criteria

###### 1. Visual Focus and Information Clarity

- • Is the visual focal point clearer?
- • Does the visual flow guide the viewer more effectively?
- • Is there less visual noise or competition for attention?

###### 2. Form and Proportion

- • Are the structures of figures or objects more accurate?
- • Are poses and silhouettes more aesthetically refined?
- • Does the image avoid stiffness, distortion, or unnatural forms?

###### 3. Color Design

- • Is the color palette more unified and thematically coherent?
- • Are value and lighting hierarchies clearer?
- • Is there effective use of contrast and visual rhythm?

###### 4. Detail Control and Finish

- • Are details handled appropriately, neither excessive nor careless?
- • Are materials and lighting consistent throughout?
- • Are edges, masks, and gradients clean and well controlled?

###### 5. Stylistic Consistency and Purpose Fit

- • Does the illustration better align with its intended stylistic direction?
- • Is quality more consistent within a unified style?
- • Is the image more suitable for its potential use case?

###### 6. Final Selection

• Considering the set as a whole, determine the relative ordering of the illustrations from

strongest to weakest. Should Not Be Considered

- • Personal preference for illustration style.
- • Software or tools used.
- • Hypothetical or imagined client requirements.

###### F.5 Notation

Symbol Meaning A = 8 number of annotators n number of images in a task, n ∈ {2,3,4,5} si,a absolute score assigned to image i by annotator a, si,a ∈ [0,10] σascore score-derived ranking for annotator a, obtained by sorting images in descending order of si,a ρsamea direct ranking provided by annotator a for a homogeneous-content task ρdiffa direct ranking provided by annotator a for a heterogeneous-content task ybesta , yworsta annotator a’s best and worst labels in a task SC top-1 self-consistency rate between score-derived and direct rankings acc fraction of tasks on which annotators reach majority agreement on the relevant label

Task-level accuracy. For the majority-agreement comparison, task-level accuracy is defined as follows:

- • accbest: fraction of tasks on which at least five of the eight annotators select the same best label.
- • accworst: fraction of tasks on which at least five of the eight annotators select the same worst label.
- • accbest&worst: fraction of tasks on which both majority conditions are satisfied simultaneously.

###### F.6 Statistical Tests

We report two-sided paired tests over tasks to assess whether the ranking and scoring protocols differ systematically. Unless otherwise noted, p-values are uncorrected for multiple comparisons.

McNemar’s test for majority-agreement comparisons. For each task and each metric accbest, accworst, and accbest&worst, we construct paired binary outcomes indicating whether the protocol achieves a five-of-eight majority hit. We then apply an exact two-sided McNemar test (McNemar, 1947; Fagerland et al., 2013) to the discordant pairs.

- • Homogeneous-content (N = 119 tasks): p = 1.0 × 10−11 for accbest, p = 2.4 × 10−9 for accworst, and p = 1.2 × 10−14 for accbest&worst.
- • Heterogeneous-content (N = 107 tasks): p = 1.2 × 10−7 for accbest, p = 2.4 × 10−5 for accworst, and p = 1.0 × 10−9 for accbest&worst.

Wilcoxon signed-rank test for entropy comparison. We compare the paired per-task entropies Hrank and Hscore over the 119 homogeneous-content tasks using a two-sided Wilcoxon signed-rank test (Wilcoxon, 1945), which yields p = 1.7 × 10−15.

Multiple comparisons. After Bonferroni correction (Bonferroni, 1936) over the three metrics in each setting, all six tests remain significant at p < 0.001.

###### F.7 Task Lists Homogeneous-Content Tasks (119 Tasks)

Each homogeneous-content task groups images that share the same underlying content but differ in aesthetic treatment. The 119 tasks span four topics across three domains.

Table 37: Homogeneous-content tasks (119 tasks).

Task ID n fine-art_landscape-color_61 4 fine-art_landscape-color_64 2 fine-art_landscape-color_65 4 fine-art_landscape-color_66 4

- fine-art_landscape-color_68 5
- fine-art_landscape-color_69 5
- fine-art_landscape-color_70 5
- fine-art_landscape-color_71 5
- fine-art_landscape-color_72 5
- fine-art_landscape-color_73 5
- fine-art_landscape-color_74 5
- fine-art_landscape-color_75 5
- fine-art_landscape-color_76 5

- fine-art_landscape-color_79 5
- fine-art_landscape-color_80 5
- fine-art_landscape-color_81 5
- fine-art_landscape-color_82 5
- fine-art_landscape-color_83 5
- fine-art_landscape-color_84 2

- fine-art_landscape-color_86 5
- fine-art_landscape-color_87 5
- fine-art_landscape-color_88 5
- fine-art_landscape-color_89 2
- fine-art_landscape-color_90 5 illustration_pixel-art_1 4 illustration_pixel-art_10 4

- illustration_pixel-art_12 4
- illustration_pixel-art_13 4

- illustration_pixel-art_17 4
- illustration_pixel-art_18 4
- illustration_pixel-art_19 4
- illustration_pixel-art_20 4
- illustration_pixel-art_21 4
- illustration_pixel-art_22 4
- illustration_pixel-art_23 4
- illustration_pixel-art_24 4
- illustration_pixel-art_25 4

- illustration_pixel-art_27 4
- illustration_pixel-art_28 4
- illustration_pixel-art_29 4

- illustration_pixel-art_3 4

illustration_pixel-art_30 4 illustration_pixel-art_31 4 illustration_pixel-art_32 4 illustration_pixel-art_33 4 illustration_pixel-art_34 4 illustration_pixel-art_35 4 illustration_pixel-art_36 4 illustration_pixel-art_37 4 illustration_pixel-art_38 4 illustration_pixel-art_39 4

- illustration_pixel-art_4 4

- illustration_pixel-art_40 4

illustration_pixel-art_7 4 illustration_pixel-art_8 4 illustration_pixel-art_9 4 photograph_architecture_103 3

- photograph_architecture_105 2
- photograph_architecture_106 2 photograph_architecture_108 2 photograph_architecture_109 2 photograph_architecture_110 2

- photograph_architecture_112 2
- photograph_architecture_113 2
- photograph_architecture_114 2 photograph_architecture_116 2 photograph_architecture_119 2

- photograph_architecture_12 3

- photograph_architecture_121 2
- photograph_architecture_122 2
- photograph_architecture_123 2
- photograph_architecture_124 2 photograph_architecture_126 2 photograph_architecture_129 2

- photograph_architecture_13 3

- photograph_architecture_133 2
- photograph_architecture_134 2 photograph_architecture_140 2 photograph_architecture_143 2 photograph_architecture_146 2

- photograph_architecture_148 2
- photograph_architecture_149 2 photograph_architecture_15 3 photograph_architecture_153 2 photograph_architecture_154 2 photograph_architecture_20 3 photograph_architecture_22 3

- photograph_architecture_25 3
- photograph_architecture_26 3 photograph_architecture_32 3

- photograph_architecture_35 3
- photograph_architecture_36 3 photograph_architecture_39 3 photograph_architecture_4 3 photograph_architecture_41 3

- photograph_architecture_44 3
- photograph_architecture_45 3 photograph_architecture_50 3

- photograph_architecture_6 3
- photograph_architecture_7 3

- photograph_landscape_1 3 photograph_landscape_10 3 photograph_landscape_15 3

- photograph_landscape_18 3
- photograph_landscape_19 3

- photograph_landscape_2 3 photograph_landscape_22 3

- photograph_landscape_26 3
- photograph_landscape_27 3 photograph_landscape_29 3

- photograph_landscape_3 3 photograph_landscape_48 3 photograph_landscape_59 3

photograph_landscape_62 3 photograph_landscape_68 3 photograph_landscape_75 3 photograph_landscape_81 3 photograph_landscape_83 3 photograph_landscape_86 3

###### Heterogeneous-Content Tasks (107 Tasks)

Each heterogeneous-content task is formed by regrouping images drawn from distinct homogeneous-content tasks, typically within the same topic. Of the original 119 regrouped tasks, 12 are excluded because the score-based Kendall’s W (Kendall and Smith, 1939) is at least 0.825, indicating that the regrouped task already exhibits near-saturated consensus under the scoring protocol. The remaining 107 tasks are retained for analysis. The Source Task IDs column lists the originating homogeneous-content task for each image in the group, in label order A, B, C, ....

Table 38: Heterogeneous-content task construction (107 tasks, after saturation filter).

ID n Source Task IDs

- 1 3 illustration_pixel-art_19, illustration_pixel-art_23, illustration_pixel-art_17
- 2 3 photograph_landscape_27, photograph_landscape_2, photograph_landscape_3
- 3 3 illustration_pixel-art_35, illustration_pixel-art_4, illustration_pixel-art_12
- 4 4 fine-art_landscape-color_81, fine-art_landscape-color_82, fine-art_landscape-color_70, fine-art_landscape-color_69
- 5 4 illustration_pixel-art_31, illustration_pixel-art_40, illustration_pixel-art_8, illustration_pixel-art_3
- 6 2 photograph_landscape_86, photograph_landscape_81
- 7 2 illustration_pixel-art_22, illustration_pixel-art_8
- 8 3 illustration_pixel-art_20, illustration_pixel-art_36, illustration_pixel-art_19
- 9 5 illustration_pixel-art_38, illustration_pixel-art_7, illustration_pixel-art_10, illustration_pixel-art_32, illustration_pixel-art_12
- 10 3 illustration_pixel-art_13, illustration_pixel-art_12, illustration_pixel-art_24
- 11 3 photograph_architecture_32, photograph_architecture_110, photograph_architecture_106
- 12 4 fine-art_landscape-color_79, fine-art_landscape-color_71, fine-art_landscape-color_61, fine-art_landscape-color_88
- 13 5 photograph_architecture_26, photograph_architecture_6, photograph_architecture_41, photograph_architecture_134, photograph_architecture_149
- 14 2 photograph_landscape_62, photograph_landscape_1
- 15 5 illustration_pixel-art_34, illustration_pixel-art_17, illustration_pixel-art_30, illustration_pixel-art_36, illustration_pixel-art_18
- 16 3 fine-art_landscape-color_90, fine-art_landscape-color_74, fine-art_landscape-color_79

- 19 3 fine-art_landscape-color_76, fine-art_landscape-color_69, fine-art_landscape-color_90
- 20 4 illustration_pixel-art_32, illustration_pixel-art_31, illustration_pixel-art_18, illustration_pixel-art_33
- 21 3 illustration_pixel-art_30, illustration_pixel-art_1, illustration_pixel-art_7

Continued on next page

- 22 5 fine-art_landscape-color_87, fine-art_landscape-color_80, fine-art_landscape-color_68, fine-art_landscape-color_76, fine-art_landscape-color_70
- 23 4 fine-art_landscape-color_73, fine-art_landscape-color_83, fine-art_landscape-color_86, fine-art_landscape-color_81
- 24 2 illustration_pixel-art_20, illustration_pixel-art_29

- 26 4 illustration_pixel-art_17, illustration_pixel-art_33, illustration_pixel-art_38, illustration_pixel-art_34
- 27 3 photograph_architecture_39, photograph_architecture_103, photograph_architecture_45
- 28 3 illustration_pixel-art_35, illustration_pixel-art_29, illustration_pixel-art_18
- 29 3 photograph_landscape_18, photograph_landscape_29, photograph_landscape_1
- 30 3 illustration_pixel-art_40, illustration_pixel-art_1, illustration_pixel-art_34

- 32 5 fine-art_landscape-color_73, fine-art_landscape-color_79, fine-art_landscape-color_90, fine-art_landscape-color_70, fine-art_landscape-color_84
- 33 2 photograph_architecture_124, photograph_architecture_26
- 34 3 photograph_architecture_106, photograph_architecture_113, photograph_architecture_4
- 35 4 fine-art_landscape-color_68, fine-art_landscape-color_79, fine-art_landscape-color_83, fine-art_landscape-color_75
- 36 3 fine-art_landscape-color_68, fine-art_landscape-color_81, fine-art_landscape-color_66
- 37 4 fine-art_landscape-color_72, fine-art_landscape-color_73, fine-art_landscape-color_76, fine-art_landscape-color_89
- 38 4 photograph_landscape_2, photograph_landscape_22, photograph_landscape_26, photograph_landscape_10
- 39 3 photograph_architecture_35, photograph_architecture_112, photograph_architecture_7
- 40 4 fine-art_landscape-color_75, fine-art_landscape-color_66, fine-art_landscape-color_81, fine-art_landscape-color_86

- 42 5 fine-art_landscape-color_87, fine-art_landscape-color_89, fine-art_landscape-color_79, fine-art_landscape-color_65, fine-art_landscape-color_86
- 43 3 photograph_landscape_83, photograph_landscape_18, photograph_landscape_3
- 44 4 photograph_landscape_62, photograph_landscape_59, photograph_landscape_75, photograph_landscape_3
- 45 4 photograph_architecture_45, photograph_architecture_13, photograph_architecture_140, photograph_architecture_26
- 46 4 illustration_pixel-art_28, illustration_pixel-art_20, illustration_pixel-art_4, illustration_pixel-art_39
- 47 4 illustration_pixel-art_25, illustration_pixel-art_10, illustration_pixel-art_8, illustration_pixel-art_28
- 48 3 illustration_pixel-art_9, illustration_pixel-art_21, illustration_pixel-art_39
- 49 4 illustration_pixel-art_30, illustration_pixel-art_18, illustration_pixel-art_4, illustration_pixel-art_8
- 50 2 photograph_architecture_114, photograph_architecture_36

- 52 5 photograph_architecture_44, photograph_architecture_153, photograph_architecture_116, photograph_architecture_129, photograph_architecture_39
- 53 3 photograph_landscape_48, photograph_landscape_83, photograph_landscape_59
- 54 4 photograph_architecture_123, photograph_architecture_103, photograph_architecture_12, photograph_architecture_13

- 56 2 photograph_landscape_27, photograph_landscape_75

- 57 2 photograph_architecture_6, photograph_architecture_39
- 58 4 illustration_pixel-art_1, illustration_pixel-art_33, illustration_pixel-art_19, illustration_pixel-art_40
- 59 2 illustration_pixel-art_22, illustration_pixel-art_27
- 60 3 fine-art_landscape-color_75, fine-art_landscape-color_87, fine-art_landscape-color_81
- 61 4 illustration_pixel-art_22, illustration_pixel-art_21, illustration_pixel-art_13, illustration_pixel-art_7
- 62 5 photograph_architecture_105, photograph_architecture_146, photograph_architecture_15, photograph_architecture_36, photograph_architecture_140
- 63 2 photograph_architecture_20, photograph_architecture_6
- 64 5 fine-art_landscape-color_73, fine-art_landscape-color_74, fine-art_landscape-color_70, fine-art_landscape-color_68, fine-art_landscape-color_80

- 66 3 fine-art_landscape-color_64, fine-art_landscape-color_74, fine-art_landscape-color_88
- 67 4 photograph_architecture_35, photograph_architecture_22, photograph_architecture_32, photograph_architecture_109
- 68 4 photograph_architecture_133, photograph_architecture_108, photograph_architecture_143, photograph_architecture_103
- 69 4 illustration_pixel-art_39, illustration_pixel-art_27, illustration_pixel-art_9, illustration_pixel-art_37
- 70 2 photograph_architecture_129, photograph_architecture_134
- 71 3 fine-art_landscape-color_90, fine-art_landscape-color_65, fine-art_landscape-color_76
- 72 3 photograph_landscape_15, photograph_landscape_48, photograph_landscape_1
- 73 4 illustration_pixel-art_3, illustration_pixel-art_37, illustration_pixel-art_31, illustration_pixel-art_1

- 75 4 illustration_pixel-art_31, illustration_pixel-art_4, illustration_pixel-art_22, illustration_pixel-art_28
- 76 5 fine-art_landscape-color_88, fine-art_landscape-color_65, fine-art_landscape-color_74, fine-art_landscape-color_69, fine-art_landscape-color_75

- 78 4 illustration_pixel-art_3, illustration_pixel-art_17, illustration_pixel-art_32, illustration_pixel-art_40
- 79 5 photograph_architecture_44, photograph_architecture_22, photograph_architecture_154, photograph_architecture_153, photograph_architecture_32
- 80 4 fine-art_landscape-color_82, fine-art_landscape-color_68, fine-art_landscape-color_84, fine-art_landscape-color_69
- 81 3 photograph_landscape_22, photograph_landscape_86, photograph_landscape_15
- 82 4 fine-art_landscape-color_66, fine-art_landscape-color_75, fine-art_landscape-color_76, fine-art_landscape-color_83
- 83 3 illustration_pixel-art_37, illustration_pixel-art_39, illustration_pixel-art_33
- 84 4 photograph_architecture_123, photograph_architecture_41, photograph_architecture_113, photograph_architecture_116
- 85 3 fine-art_landscape-color_61, fine-art_landscape-color_87, fine-art_landscape-color_70
- 86 4 illustration_pixel-art_25, illustration_pixel-art_12, illustration_pixel-art_35, illustration_pixel-art_32
- 87 5 photograph_architecture_126, photograph_architecture_112, photograph_architecture_146, photograph_architecture_25, photograph_architecture_50
- 88 4 illustration_pixel-art_30, illustration_pixel-art_36, illustration_pixel-art_37, illustration_pixel-art_24
- 89 3 illustration_pixel-art_25, illustration_pixel-art_20, illustration_pixel-art_10

Continued on next page

- 90 3 photograph_architecture_154, photograph_architecture_110, photograph_architecture_25
- 91 2 photograph_architecture_20, photograph_architecture_149
- 92 5 illustration_pixel-art_23, illustration_pixel-art_13, illustration_pixel-art_38, illustration_pixel-art_29, illustration_pixel-art_19
- 93 5 photograph_architecture_15, photograph_architecture_7, photograph_architecture_41, photograph_architecture_121, photograph_architecture_148
- 94 2 illustration_pixel-art_23, illustration_pixel-art_10

- 96 2 illustration_pixel-art_29, illustration_pixel-art_23
- 97 4 photograph_architecture_121, photograph_architecture_119, photograph_architecture_108, photograph_architecture_22
- 98 3 photograph_landscape_19, photograph_landscape_81, photograph_landscape_68
- 99 2 photograph_architecture_7, photograph_architecture_122
- 100 4 photograph_architecture_45, photograph_architecture_126, photograph_architecture_133, photograph_architecture_122
- 101 4 photograph_landscape_59, photograph_landscape_26, photograph_landscape_10, photograph_landscape_29
- 102 3 photograph_architecture_119, photograph_architecture_36, photograph_architecture_148
- 103 3 illustration_pixel-art_7, illustration_pixel-art_36, illustration_pixel-art_3
- 104 3 illustration_pixel-art_34, illustration_pixel-art_24, illustration_pixel-art_13
- 105 4 fine-art_landscape-color_80, fine-art_landscape-color_86, fine-art_landscape-color_90, fine-art_landscape-color_88
- 106 2 illustration_pixel-art_27, illustration_pixel-art_21
- 107 3 photograph_architecture_12, photograph_architecture_50, photograph_architecture_15

- 109 3 photograph_landscape_68, photograph_landscape_62, photograph_landscape_2
- 110 3 illustration_pixel-art_9, illustration_pixel-art_38, illustration_pixel-art_27
- 111 4 fine-art_landscape-color_83, fine-art_landscape-color_82, fine-art_landscape-color_72, fine-art_landscape-color_71
- 112 2 illustration_pixel-art_9, illustration_pixel-art_35
- 113 3 fine-art_landscape-color_80, fine-art_landscape-color_82, fine-art_landscape-color_87
- 114 2 photograph_architecture_4, photograph_architecture_12
- 115 5 fine-art_landscape-color_72, fine-art_landscape-color_88, fine-art_landscape-color_69, fine-art_landscape-color_71, fine-art_landscape-color_86
- 116 5 photograph_landscape_22, photograph_landscape_27, photograph_landscape_19, photograph_landscape_26, photograph_landscape_29
- 117 5 photograph_architecture_143, photograph_architecture_25, photograph_architecture_20, photograph_architecture_105, photograph_architecture_124
- 118 2 illustration_pixel-art_24, illustration_pixel-art_28
- 119 3 photograph_architecture_109, photograph_architecture_114, photograph_architecture_44

###### G Fine-Tuning Details

###### G.1 Hyperparameters

- Table 39 lists the hyperparameters used to fine-tune KALLISTI-35B-A3B.

- Table 40: Topic-level fine-tuning Top-1 ap@1 results on fine-art tasks. The final column reports the corresponding domain-level average (%).

Model Calligraphy Chinese Painting Ink & Wash Landscape Color Portrait Color Portrait Sketch Quick Sketch Still Life Color Still Life Sketch Avg. Open-Weight Models

GLM-4.6V 21.2 63.0 33.3 26.8 66.7 52.5 54.2 58.3 52.4 42.7 KIMI-K2.5 30.3 85.2 50.0 27.4 100.0 46.5 62.5 50.0 52.4 43.5

QWEN3-VL-235B-A22B 15.2 85.2 50.0 26.8 66.7 48.5 66.7 47.2 57.1 42.4 QWEN3.5-397B-A17B 24.2 96.3 66.7 33.9 100.0 37.4 58.3 43.1 45.2 42.7

Ours

QWEN3.5-35B-A3B (base) 21.2 85.2 75.0 31.0 83.3 41.4 75.0 37.5 47.6 41.8 KALLISTI-35B-A3B 24.2 81.5 58.3 36.3 66.7 52.5 75.0 47.2 50.0 47.0

Table 39: Hyperparameters used for fine-tuning KALLISTI-35B-A3B.

Parameter Value

Base Model Qwen3.5-35B-A3B Fine-tuning Method LoRA

LoRA Rank 64 LoRA Alpha 128 LoRA Target All layers Vision Tower Frozen MLP Projector Trainable Learning Rate 1 × 10−4 LR Scheduler Cosine

Warmup Steps 10 Epochs 3 Batch Size 4 per device Precision BF16 Max Sequence Length 65,536

###### G.2 Detailed Results

- As in the main-text fine-tuning table, we report topic-level detailed results under all four metrics: Top-1 ap@1, Top-1 pass^3, TB-1 ap@1, and TB-1 pass^3. The full breakdown shows that the gains from fine-tuning are uneven across topics: KALLISTI-35B-A3B improves most clearly on several fine-art topics, remains competitive with larger open-weight baselines on illustration, and does not uniformly dominate its base model in photography.

###### Top-1 ap@1.

Tables 40, 41, and 42 report topic-level Top-1 ap@1 results for fine art, illustration, and photography, respectively.

Top-1 pass^3. Tables 43, 44, and 45 report topic-level Top-1 pass^3 results for fine art, illustration, and photography, respectively. TB-1 ap@1. Tables 46, 47, and 48 report topic-level TB-1 ap@1 results for fine art, illustration, and photography, respectively. TB-1 pass^3.

Tables 49, 50, and 51 report topic-level TB-1 pass^3 results for fine art, illustration, and photography, respectively.

- Table 41: Topic-level fine-tuning Top-1 ap@1 results on illustration tasks. The final column reports the corresponding domain-level average (%).

Model Anime & Manga Comic Concept Art Digital & AI Art Pixel Art Stylized 3D Avg. Open-Weight Models

GLM-4.6V 51.3 53.3 31.8 37.7 60.3 27.3 43.3 KIMI-K2.5 71.8 56.7 36.4 36.2 39.7 30.3 43.0

QWEN3-VL-235B-A22B 69.2 33.3 30.3 30.4 39.7 36.4 38.3 QWEN3.5-397B-A17B 71.8 53.3 33.3 27.5 50.8 48.5 44.3

Ours

QWEN3.5-35B-A3B (base) 53.8 46.7 30.3 44.9 41.3 30.3 40.7 KALLISTI-35B-A3B 66.7 50.0 28.8 36.2 44.4 36.4 41.7

- Table 42: Topic-level fine-tuning Top-1 ap@1 results on photography tasks. The final column reports the corresponding domain-level average (%).

Model Architecture Food & Product Landscape Macro Night & Astro Portrait Sports Street & City Wildlife Avg. Open-Weight Models

GLM-4.6V 35.4 53.3 39.3 58.3 16.7 0.0 100.0 48.7 63.0 43.2 KIMI-K2.5 49.0 80.0 63.7 75.0 50.0 50.0 100.0 61.5 70.4 60.9

QWEN3-VL-235B-A22B 37.5 66.7 54.1 58.3 50.0 50.0 66.7 57.3 66.7 52.5 QWEN3.5-397B-A17B 44.8 66.7 64.4 83.3 50.0 50.0 100.0 72.6 66.7 62.8

Ours

QWEN3.5-35B-A3B (base) 51.0 46.7 57.0 83.3 50.0 33.3 100.0 65.0 70.4 59.0 KALLISTI-35B-A3B 50.0 73.3 67.4 75.0 50.0 33.3 100.0 66.7 70.4 63.3

- Table 43: Topic-level fine-tuning Top-1 pass^3 results on fine-art tasks. The final column reports the corresponding domain-level average (%).

Model Calligraphy Chinese Painting Ink & Wash Landscape Color Portrait Color Portrait Sketch Quick Sketch Still Life Color Still Life Sketch Avg. Open-Weight Models

GLM-4.6V 9.1 55.6 0.0 5.4 50.0 21.2 25.0 33.3 14.3 18.0 KIMI-K2.5 9.1 77.8 25.0 5.4 100.0 18.2 50.0 16.7 28.6 19.9

QWEN3-VL-235B-A22B 9.1 77.8 50.0 8.9 50.0 24.2 50.0 20.8 21.4 22.4 QWEN3.5-397B-A17B 18.2 88.9 50.0 16.1 100.0 6.1 37.5 16.7 14.3 21.1

Ours

QWEN3.5-35B-A3B (base) 0.0 66.7 75.0 14.3 50.0 15.2 50.0 16.7 28.6 21.7 KALLISTI-35B-A3B 0.0 77.8 50.0 19.6 50.0 27.3 62.5 33.3 14.3 28.0

- Table 44: Topic-level fine-tuning Top-1 pass^3 results on illustration tasks. The final column reports the corresponding domain-level average (%).

Model Anime & Manga Comic Concept Art Digital & AI Art Pixel Art Stylized 3D Avg. Open-Weight Models

GLM-4.6V 23.1 20.0 9.1 17.4 38.1 0.0 19.0 KIMI-K2.5 53.8 30.0 13.6 21.7 9.5 9.1 21.0

QWEN3-VL-235B-A22B 46.2 0.0 4.5 4.3 9.5 18.2 12.0 QWEN3.5-397B-A17B 53.8 10.0 13.6 4.3 33.3 45.5 24.0

Ours

QWEN3.5-35B-A3B (base) 23.1 20.0 9.1 34.8 14.3 18.2 20.0 KALLISTI-35B-A3B 53.8 10.0 9.1 21.7 19.0 27.3 22.0

- Table 45: Topic-level fine-tuning Top-1 pass^3 results on photography tasks. The final column reports the corresponding domain-level average (%).

Model Architecture Food & Product Landscape Macro Night & Astro Portrait Sports Street & City Wildlife Avg. Open-Weight Models

GLM-4.6V 6.2 20.0 15.6 50.0 0.0 0.0 100.0 15.4 33.3 15.8 KIMI-K2.5 18.8 60.0 44.4 75.0 50.0 50.0 100.0 35.9 55.6 38.8

QWEN3-VL-235B-A22B 6.2 60.0 26.7 50.0 50.0 0.0 0.0 28.2 33.3 24.5 QWEN3.5-397B-A17B 31.2 40.0 42.2 75.0 50.0 0.0 100.0 51.3 55.6 43.9

Ours

QWEN3.5-35B-A3B (base) 21.9 20.0 35.6 75.0 50.0 0.0 100.0 35.9 44.4 33.8 KALLISTI-35B-A3B 21.9 40.0 46.7 50.0 50.0 0.0 100.0 33.3 44.4 36.7

- Table 46: Topic-level fine-tuning TB-1 ap@1 results on fine-art tasks. The final column reports the corresponding domain-level average (%).

Model Calligraphy Chinese Painting Ink & Wash Landscape Color Portrait Color Portrait Sketch Quick Sketch Still Life Color Still Life Sketch Avg. Open-Weight Models

GLM-4.6V 21.2 63.0 33.3 17.3 66.7 52.5 54.2 58.3 52.4 39.3 KIMI-K2.5 30.3 85.2 50.0 13.1 100.0 46.5 62.5 50.0 52.4 38.5

QWEN3-VL-235B-A22B 15.2 85.2 50.0 10.7 66.7 48.5 66.7 47.2 57.1 36.9 QWEN3.5-397B-A17B 24.2 96.3 66.7 18.5 100.0 37.4 58.3 43.1 45.2 37.3

Ours

QWEN3.5-35B-A3B (base) 21.2 85.2 75.0 16.7 83.3 41.4 75.0 37.5 47.6 36.9 KALLISTI-35B-A3B 24.2 81.5 58.3 17.9 66.7 52.5 75.0 47.2 50.0 40.6

- Table 47: Topic-level fine-tuning TB-1 ap@1 results on illustration tasks. The final column reports the corresponding domain-level average (%).

Model Anime & Manga Comic Concept Art Digital & AI Art Pixel Art Stylized 3D Avg. Open-Weight Models

GLM-4.6V 51.3 50.0 13.6 20.3 28.6 15.2 27.0 KIMI-K2.5 71.8 53.3 18.2 20.3 17.5 18.2 29.0

QWEN3-VL-235B-A22B 69.2 33.3 22.7 14.5 22.2 3.0 25.7 QWEN3.5-397B-A17B 71.8 46.7 18.2 15.9 20.6 0.0 26.0

Ours

QWEN3.5-35B-A3B (base) 53.8 43.3 4.5 18.8 22.2 9.1 22.3 KALLISTI-35B-A3B 66.7 46.7 19.7 18.8 15.9 0.0 25.3

- Table 48: Topic-level fine-tuning TB-1 ap@1 results on photography tasks. The final column reports the corresponding domain-level average (%).

Model Architecture Food & Product Landscape Macro Night & Astro Portrait Sports Street & City Wildlife Avg. Open-Weight Models

GLM-4.6V 28.1 40.0 28.1 33.3 33.3 0.0 66.7 39.3 48.1 33.1 KIMI-K2.5 29.2 80.0 39.3 33.3 16.7 0.0 66.7 45.3 59.3 40.5

QWEN3-VL-235B-A22B 28.1 53.3 36.3 91.7 33.3 66.7 66.7 42.7 51.9 40.0 QWEN3.5-397B-A17B 31.2 66.7 49.6 83.3 50.0 66.7 0.0 57.3 66.7 50.1

Ours

QWEN3.5-35B-A3B (base) 41.7 46.7 42.2 50.0 0.0 33.3 0.0 52.1 74.1 46.3 KALLISTI-35B-A3B 38.5 53.3 43.0 41.7 50.0 16.7 33.3 55.6 59.3 46.5

- Table 49: Topic-level TB-1 pass^3 fine-tuning results on fine-art tasks. The final column reports the corresponding domain-level average (%).

Model Calligraphy Chinese Painting Ink & Wash Landscape Color Portrait Color Portrait Sketch Quick Sketch Still Life Color Still Life Sketch Avg. Open-Weight Models

GLM-4.6V 9.1 55.6 0.0 7.1 50.0 21.2 25.0 33.3 14.3 18.6 KIMI-K2.5 9.1 77.8 25.0 0.0 100.0 18.2 50.0 16.7 28.6 18.0

QWEN3-VL-235B-A22B 9.1 77.8 50.0 0.0 50.0 24.2 50.0 20.8 21.4 19.3 QWEN3.5-397B-A17B 18.2 88.9 50.0 5.4 100.0 6.1 37.5 16.7 14.3 17.4

Ours

QWEN3.5-35B-A3B (base) 0.0 66.7 75.0 3.6 50.0 15.2 50.0 16.7 28.6 18.0 KALLISTI-35B-A3B 0.0 77.8 50.0 5.4 50.0 27.3 62.5 33.3 14.3 23.0

- Table 50: Topic-level TB-1 pass^3 fine-tuning results on illustration tasks. The final column reports the corresponding domain-level average (%).

Model Anime & Manga Comic Concept Art Digital & AI Art Pixel Art Stylized 3D Avg. Open-Weight Models

GLM-4.6V 23.1 20.0 4.5 8.7 0.0 0.0 8.0

KIMI-K2.5 53.8 30.0 4.5 8.7 4.8 0.0 14.0 QWEN3-VL-235B-A22B 46.2 0.0 4.5 0.0 4.8 0.0 8.0

QWEN3.5-397B-A17B 53.8 10.0 0.0 0.0 0.0 0.0 8.0 Ours QWEN3.5-35B-A3B (base) 23.1 20.0 0.0 8.7 0.0 0.0 7.0

KALLISTI-35B-A3B 53.8 10.0 0.0 8.7 0.0 0.0 10.0

- Table 51: Topic-level TB-1 pass^3 fine-tuning results on photography tasks. The final column reports the corresponding domain-level average (%).

Model Architecture Food & Product Landscape Macro Night & Astro Portrait Sports Street & City Wildlife Avg. Open-Weight Models

GLM-4.6V 0.0 20.0 2.2 0.0 0.0 0.0 0.0 10.3 22.2 5.8

KIMI-K2.5 3.1 60.0 6.7 25.0 0.0 0.0 0.0 15.4 33.3 12.2 QWEN3-VL-235B-A22B 3.1 20.0 11.1 75.0 0.0 0.0 0.0 12.8 22.2 12.2

QWEN3.5-397B-A17B 15.6 20.0 17.8 50.0 50.0 0.0 0.0 30.8 44.4 23.7 Ours

QWEN3.5-35B-A3B (base) 15.6 20.0 17.8 25.0 0.0 0.0 0.0 20.5 44.4 19.4 KALLISTI-35B-A3B 9.4 20.0 17.8 0.0 50.0 0.0 0.0 20.5 22.2 16.5

###### H Prompt Templates and Scoring Interfaces

This appendix records the prompt templates and scoring interfaces used in data construction and evaluation. We group them by usage: prompts for the photography pipeline, prompts for the illustration pipeline, prompts for MLLM evaluation, and scoring interfaces for reward-model evaluation.

###### H.1 Photography Pipeline Prompts

This subsection records the prompts used in the photography agent-edit pipeline described in Section 3. The pipeline consists of three stages: aspect-wise critique generation with ArtiMuse, edit planning with GPT through LiteLLM, and image editing with Gemini-3-Pro-Image (Nano Banana Pro). In Step 1, ArtiMuse is queried eight times independently with the same image, once per aesthetic aspect. The resulting responses are then concatenated into a single review string that is passed to the planning stage.

###### H.1.1 Step 1: ArtiMuse Aspect-Wise Critique

ArtiMuse Aspect-Wise Critique Calls

The same image is attached in every call. The following eight prompts are sent independently:

- 1. Please evaluate the aesthetic quality of this image from the aspect of Composition & Design.
- 2. Please evaluate the aesthetic quality of this image from the aspect of Visual Elements & Structure.
- 3. Please evaluate the aesthetic quality of this image from the aspect of Technical Execution.
- 4. Please evaluate the aesthetic quality of this image from the aspect of Originality & Creativity.
- 5. Please evaluate the aesthetic quality of this image from the aspect of Theme & Communication.
- 6. Please evaluate the aesthetic quality of this image from the aspect of Emotion & Viewer Response.
- 7. Please evaluate the aesthetic quality of this image from the aspect of Overall Gestalt.
- 8. Please evaluate the aesthetic quality of this image from the aspect of Comprehensive Evaluation.

Critique Aggregation Format

The eight responses are concatenated into a single review string in the following format: [Composition & Design] <response> [Visual Elements & Structure] <response> [Technical Execution] <response> [Originality & Creativity] <response> [Theme & Communication] <response> [Emotion & Viewer Response] <response> [Overall Gestalt] <response> [Comprehensive Evaluation] <response>

###### H.1.2 Step 2: Edit Planning

The planning stage is executed with GPT through LiteLLM. Below we reproduce the literal runtime system and user messages. The placeholder {review} denotes the full ArtiMuse output obtained by concatenating the eight aspect-wise responses from Step 1. We retain the original runtime field names, including substyle.

###### Planning System Message

You are an expert aesthetic analyst and image editing planner for creating aesthetic preference benchmark datasets.

###### Planning User Message

Analyze this photograph and plan a synthetic edit for an aesthetic preference dataset. AESTHETIC ASSESSMENT:

{review} AVAILABLE CLASSIFICATIONS: Domains: Photo Substyles:

{

"Photo": [ "Portrait", "Landscape", "StreetCity", "Architecture", "Wildlife", "Macro", "FoodProduct", "NightAstro", "Sports"

]

} TASK:

- 1. Classify the image: choose the most appropriate domain and substyle based on image itself
- 2. Freely decide how to edit this image to create better and worse versions
- 3. Write a "better" prompt: describe how to edit THIS image to make it aesthetically better
- 4. Write a "worse" prompt: describe how to edit THIS image to make it aesthetically worse

- "Better" means: take the current image and improve its quality - "Worse" means: take the current image and degrade its quality.

EDIT Types - Do NOT always default to color/tone adjustments. Actively consider all edit types:

- 1. **Color & Tonal Adjustments**:

- - White balance, color cast correction or introduction
- - Saturation, vibrance, contrast adjustments
- - Color grading and mood shifts

- 2. **Perspective & Viewpoint Changes**:

- Simulate a slightly different camera angle or position - Adjust the apparent depth or distance from the subject - Change the sense of scale or spatial relationships

- 3. **Element Removal**:

- - Remove contextual elements that enhance (or clutter) the scene
- - Modify background elements to improve or degrade composition

- 4. **Compositional Restructuring**:

- - Adjust cropping or framing emphasis
- - Rebalance visual weight distribution
- - Modify leading lines or focal points

- 5. **Light & Shadow Changes**:

- - Modify shadow depth, highlight recovery, or exposure
- - Change lighting direction or intensity
- - Adjust local brightness to guide visual attention

Choose multiple edit types that best address the specific strengths and weaknesses identified in the aesthetic assessment. WORSE PROMPT RESTRICTIONS:

When creating the "worse" prompt, DO NOT use the following editing techniques:

- - DO NOT add lens flare or light flares
- - DO NOT add vignetting (darkened corners)
- - DO NOT add object outlines, glows, or halos around subjects These techniques are overused and create unrealistic degradation. PROMPT WRITING GUIDELINES:
- - Describe the scene narratively, don’t just list keywords, a narrative, descriptive paragraph will almost always produce a better, more coherent image than a list of disconnected words.
- - Use photographic and cinematic language terms. Mention camera angles, lens types, lighting, and fine details to guide the model toward a photorealistic result.
- - Explain how changes should integrate into the scene (lighting, style, composition)
- - For adding/removing elements: describe what to add/remove and how it should blend in
- - For style changes: describe both the target style and what to preserve Return JSON:

{

"content": {"domain": "Photo", "substyle": "..."}, "planned_prompts": { "better": "...", "worse": "..."

}, "reasoning": "Explain your substyle classification choice and why you chose

these specific photographic edits for better/worse versions" }

###### H.1.3 Step 3: Image Editing

The final stage uses Gemini-3-Pro-Image (Nano Banana Pro) to synthesize the edited variants. Round 1 generates the better image from the original image alone. Round 2 generates the worse image from the original image while conditioning on the better image as additional reference context. For Round 2, we report the file-backed with-context template actually used at runtime. This is the operative version loaded by the pipeline and therefore excludes the fallback sentence beginning with “Important: Apply this edit to the ORIGINAL image” that appears only in the default code path.

- Round 1: Better-Image Editing Prompt You are an expert image editor. Edit the provided image according to these instructions:

{edit_instruction}

- Round 2: Worse-Image Editing Prompt with Context You are an expert image editor.

Context: Previously, the following improvement was made to the original image: "{context_prompt}" (The improved result is shown in the second image for reference) Now, edit the ORIGINAL image (first image) according to these instructions: "{edit_instruction}"

###### H.2 Illustration Pipeline Prompts

This subsection records the prompt templates used in the prompt-based illustration generation pipeline. We use one template for digital and AI art, concept art, and pixel art, and a separate storyboard-style template

for anime and manga and comic generation. The stylized 3D subset is constructed through asset rendering rather than text-to-image prompting.

###### H.2.1 Digital and AI Art / Concept Art / Pixel Art Generation

For digital and AI art, concept art, and pixel art generation, we use a structured template to control semantic content while allowing variation in visual execution. Each prompt is composed in a fixed order: subject or character, motion or action, environment or background, lighting condition, atmosphere or mood, camera or composition, and art style.

Illustration Structured Prompt Template

Generate 25 concise prompts ready for text-to-image artwork generation on the topic of [ART CATEGORY]. Each prompt should describe a clear visual scene using the following order: subject or character, motion, environment/background, lighting condition, atmosphere or mood, camera or composition (view, framing, depth), and art style.

###### H.2.2 Manga / Comic Storyboard Generation

For anime and manga and comic generation, we adopt a storyboard-style template with stronger structural constraints. Each prompt describes a short multi-panel narrative with consistent characters and environment, while each panel specifies shot type, character action, key props, and scene details.

Illustration Storyboard Prompt Template

Generate 25 concise storyboard prompts for [ART CATEGORY] artwork generation. Each prompt should describe a micro-story using panels. Anchors such as character identity and environment must remain consistent. Each panel must specify shot type (wide, medium, close-up), character actions, key props, and environment details. Keep the description visual and concise.

###### H.3 MLLM Evaluation Prompts

This subsection reports the generic prompts used to evaluate MLLMs on VAB. Depending on the task setting, the model is asked to identify the best image, the worst image, or both the best and the worst images in a candidate set.

- H.3.1 Best-Image Selection Prompt

Pick-Best Prompt

You will be shown {n} images labeled {labels}. Evaluate the overall aesthetic quality of each image, considering composition, color, technique, and artistic expression.

Which image has the best overall aesthetic quality? {options}

You may reason step by step. Then, on the very last line of your response, write your final answer in exactly this format:

BEST: X where X is the letter label of the best image.

###### H.3.2 Worst-Image Selection Prompt

Pick-Worst Prompt

You will be shown {n} images labeled {labels}. Evaluate the overall aesthetic quality of each image, considering composition, color, technique, and artistic expression.

Which image has the worst overall aesthetic quality? {options}

You may reason step by step. Then, on the very last line of your response, write your final answer in exactly this format:

WORST: X where X is the letter label of the worst image.

###### H.3.3 Joint Best-and-Worst Selection Prompt

Pick-Best-and-Worst Prompt

You will be shown {n} images labeled {labels}. Evaluate the overall aesthetic quality of each image, considering composition, color, technique, and artistic expression.

Which image has the best overall aesthetic quality, and which has the worst? {options}

You may reason step by step. Then, on the very last lines of your response, write your final answers in exactly this format:

BEST: X WORST: Y

where X and Y are the letter labels.

###### H.4 Reward-Model Scoring Interfaces

This subsection records the model-specific scoring interfaces used to evaluate reward models on VAB. Unlike MLLMs, reward models do not share a single comparative prompt. Instead, each model scores images independently following its released evaluation method, and we convert the resulting per-image scores into within-task rankings. Across all reward models, each image is scored num_trials times, the scores are averaged, and the averaged scores are used to determine the predicted best and worst images. We use the following terminology: explicit prompt denotes prompt text exposed at the call site, implicit prompt denotes prompt text embedded inside the model library, and no prompt denotes a pure vision scorer with no language instruction.

###### H.4.1 ArtiMuse

ArtiMuse uses an implicit prompt: the instruction is hardcoded inside the released library rather than passed from our evaluation script.

###### ArtiMuse Internal Scoring Prompt

Rate the aesthetics score of the image in 0-100. In the output format, numbers are replaced by 2 corresponding letters, and the mapping relationship is: score 0 to 25: 0-aa, 1-ab, 2-ac, 3-ad, ..., 25-az, score 26 to 50: 26-ca, 27-cb, 28-cc, 29-cd, ..., 50-cy, score 51 to 75: 51-da, 52-db, 53-dc, 54-dd, ..., 75-dy, score 76 to 100: 76-ea, 77-eb, 78-ec, 79-ed, ..., 100-ey.

The answer only outputs 2 corresponding letters.

- At runtime, the library automatically prepends an image token before the instruction. The model outputs a two-letter code, which is then decoded back to a numerical score in [0,100].

###### H.4.2 Q-Align

Q-Align also uses an implicit prompt. In our experiments, we use the aesthetic scorer by default; the library additionally exposes a quality scorer.

Q-Align Internal Prompts

Aesthetic scorer (used by default):

USER: How would you rate the aesthetics of this image? <|image|> ASSISTANT: The aesthetics of the image is

Quality scorer (alternative interface):

USER: How would you rate the quality of this image? <|image|> ASSISTANT: The quality of the image is

The scorer reads the last-token logits over the five words excellent, good, fair, poor, and bad, applies a softmax, and computes a weighted sum with weights [1,0.75,0.5,0.25,0].

###### H.4.3 Q-Instruct Q-Instruct uses an explicit prompt passed through the command-line interface.

Q-Instruct Evaluation Prompt

Rate the quality of the image.

The instruction is wrapped in the mplug_owl2 conversation format with a prepended DEFAULT_IMAGE_TOKEN. The default score is the logit difference between good and poor.

###### H.4.4 RealQA RealQA uses an explicit prompt. Under the AVA adapter used in our experiments, the default instruction is:

RealQA Evaluation Prompt

Rate the aesthetics and quality of this image briefly.

The model generates free text, from which a numerical score in [0.0,10.0] is parsed. If parsing fails, the library issues a second constrained prompt:

RealQA Retry Prompt

Output only one numeric score. Range: 0.0 to 10.0. No words, no explanation.

###### H.4.5 PEAS

PEAS-aes and PEAS-ava use no prompt. They are pure vision models with no language instruction. The scorer outputs a 10-bin distribution over scores from 1 to 10, and the final score is the expected value of that distribution.

###### PEAS Scoring Interface

No textual prompt is used. The model outputs a 10-bin score distribution over the range 1 to 10, and the final scalar score is computed as the expected value under that distribution.

###### I Topic-Level Annotation Examples

This appendix provides one representative annotation example for each of the 24 topics in VAB, grouped by domain.

- I.1 Fine Art

Landscape Color

- A
- B
- C
- D

|[Figure 559]|
|---|

|[Figure 560]|
|---|

|[Figure 561]|
|---|

|[Figure 562]|
|---|

|[Figure 563]|
|---|

|[Figure 564]|
|---|

|[Figure 565]|
|---|

|[Figure 566]|
|---|

3rd 4th 1st 2nd

- Expert 1
- Expert 2
- Expert 3
- Expert 4
- Expert 5
- Expert 6
- Expert 7
- Expert 8
- Expert 9

- Expert 10

2nd 4th 1st 3rd

- 1st 4th 2nd 3rd

- 1st 4th 3rd 2nd 3rd 2nd 1st 4th

4th 3rd 1st 2nd

3rd 4th 1st 2nd

4th 2nd 1st 3rd

- 2nd 4th 1st 3rd 2nd 3rd 4th 1st

Still Life Color

- A
- B
- C
- D

|[Figure 568]|
|---|

|[Figure 569]|
|---|

|[Figure 570]|
|---|

|[Figure 571]|
|---|

|[Figure 572]|
|---|

|[Figure 573]|
|---|

|[Figure 574]|
|---|

|[Figure 575]|
|---|

3rd 4th 2nd 1st

- Expert 1
- Expert 2
- Expert 3
- Expert 4
- Expert 5
- Expert 6
- Expert 7
- Expert 8
- Expert 9

- Expert 10

3rd 2nd

4th 4th 4th

1st

1st 3rd

3rd

2nd 2nd 1st

3rd 4th 2nd 1st

3rd 4th 2nd 1st

- 2nd 4th 3rd 1st

- 3rd 4th 2nd 1st

4th

1st

3rd 2nd

2nd 4th 3rd 1st

Calligraphy

- A
- B

|[Figure 577]|
|---|

|[Figure 578]|
|---|

- Expert 1
- Expert 2
- Expert 3
- Expert 4
- Expert 5
- Expert 6
- Expert 7
- Expert 8
- Expert 9

- Expert 10

|[Figure 579]|
|---|

|[Figure 580]|
|---|

1st 2rd 1st 2rd

2rd 1st

1st 2rd

1st 2rd 1st 2rd 1st 2rd 1st 2rd 1st 2rd 1st 2rd 1st 2rd

Chinese Painting

- A
- B

|[Figure 582]|
|---|

|[Figure 583]|
|---|

|[Figure 584]|
|---|

1st 2rd 1st 2rd

- Expert 1
- Expert 2
- Expert 3
- Expert 4
- Expert 5
- Expert 6
- Expert 7
- Expert 8
- Expert 9

- Expert 10

2rd 1st

1st 2rd 1st 2rd 1st 2rd 1st 2rd 1st 2rd 1st 2rd 1st 2rd

|[Figure 585]|
|---|

Ink and Wash

A

B

|[Figure 587]|
|---|

|[Figure 588]|
|---|

|[Figure 589]|
|---|

2rd 1st

1st 2rd

Expert 1 Expert 2 Expert 3 Expert 4 Expert 5 Expert 6 Expert 7 Expert 8 Expert 9 Expert 10

- 1st 2rd

- 2rd 1st 2rd 1st

- 1st 2rd

- 2rd 1st 2rd 1st 2rd 1st 2rd 1st 2rd 1st

2rd 1st

|[Figure 590]|
|---|

1st 2rd

2rd 1st 2rd 1st

Portrait Color

A

B

|[Figure 592]|
|---|

|[Figure 593]|
|---|

- Expert 1
- Expert 2
- Expert 3
- Expert 4
- Expert 5
- Expert 6
- Expert 7
- Expert 8
- Expert 9
- Expert 10

|[Figure 594]|
|---|

|[Figure 595]|
|---|

2rd 1st

21strd 21strd

2rd 1st

- 1st 2rd

- 2rd 1st 2rd 1st

- 1st 2rd

- 2rd 1st 2rd 1st 2rd 1st 2rd 1st 2rd 1st

21strd 21strd

1st 2rd

2rd 1st 2rd 1st

Portrait Sketch

A

B

|[Figure 597]|
|---|

|[Figure 598]|
|---|

|[Figure 599]|
|---|

|[Figure 600]|
|---|

Expert 1 Expert 2 Expert 3 Expert 4 Expert 5 Expert 6 Expert 7 Expert 8 Expert 9 Expert 10

1st 2rd 1st 2rd

2rd 1st

1st 2rd

1st 2rd 1st 2rd 1st 2rd 1st 2rd 1st 2rd 1st 2rd 1st 2rd

Quick Sketch

##### A

##### B

|[Figure 602]|
|---|

|[Figure 603]|
|---|

|[Figure 604]|
|---|

|[Figure 605]|
|---|

Expert 1 Expert 2 Expert 3 Expert 4 Expert 5 Expert 6 Expert 7 Expert 8 Expert 9 Expert 10

1st 2rd 1st 2rd

2rd 1st

- 1st 2rd

- 2rd 1st 21strd 21strd

1st 2rd 1st 2rd 1st 2rd 1st 2rd 1st 2rd 1st 2rd 1st 2rd

Still Life Sketch

A

B

|[Figure 607]|
|---|

|[Figure 608]|
|---|

1st 2rd 1st 2rd

Expert 1 Expert 2 Expert 3 Expert 4 Expert 5 Expert 6 Expert 7 Expert 8 Expert 9 Expert 10

- 1st 2rd

|[Figure 609]|
|---|

|[Figure 610]|
|---|

- 2rd 1st

2rd 1st

1st 2rd 1st 2rd 1st 2rd 1st 2rd 1st 2rd 1st 2rd 1st 2rd

2rd 1st

- I.2 Photography

Architecture

- A
- B

|[Figure 612]|
|---|

|[Figure 613]|
|---|

|[Figure 614]|
|---|

2rd 1st 2rd 1st 2rd 1st 2rd 1st 2rd 1st 2rd 1st 2rd 1st 2rd 1st 2rd 1st 2rd 1st

- Expert 1
- Expert 2
- Expert 3
- Expert 4
- Expert 5
- Expert 6
- Expert 7
- Expert 8
- Expert 9

- Expert 10

|[Figure 615]|
|---|

Food and Product

A

B

|[Figure 617]|
|---|

|[Figure 618]|
|---|

|[Figure 619]|
|---|

|[Figure 620]|
|---|

Expert 1 Expert 2 Expert 3 Expert 4 Expert 5 Expert 6 Expert 7 Expert 8 Expert 9 Expert 10

1st 2rd 1st 2rd

2rd 1st

1st 2rd

1st 2rd 1st 2rd 1st 2rd 1st 2rd 1st 2rd 1st 2rd 1st 2rd

###### Landscape

- A
- B
- C

|[Figure 622]|
|---|

- 1st 2nd 3rd

|[Figure 623]|
|---|

|[Figure 624]|
|---|

|[Figure 625]|
|---|

- 1st 3rd 2nd

- Expert 1
- Expert 2
- Expert 3
- Expert 4
- Expert 5
- Expert 6
- Expert 7
- Expert 8
- Expert 9

- Expert 10

1st 2nd 3rd

|[Figure 626]|
|---|

- 1st 2nd 3rd

- 2nd 1st 3rd

1st 2nd 3rd 1st 2nd 3rd 1st 2nd 3rd 1st 2nd 3rd 1st 2nd 3rd

|[Figure 627]|
|---|

Macro

A

B

C

|[Figure 629]|
|---|

|[Figure 630]|
|---|

|[Figure 631]|
|---|

- Expert 1
- Expert 2
- Expert 3
- Expert 4
- Expert 5
- Expert 6
- Expert 7
- Expert 8
- Expert 9

- Expert 10

|[Figure 632]|
|---|

|[Figure 633]|
|---|

|[Figure 634]|
|---|

2nd 1st 3rd 2nd 1st 3rd 1st 2nd 3rd 2nd 1st 3rd 2nd 1st 3rd 1st 2nd 3rd 2nd 1st 3rd 2nd 1st 3rd 2nd 1st 3rd 2nd 1st 3rd

Night and Astrophotography

|[Figure 636]|
|---|

|[Figure 637]|
|---|

|[Figure 638]|
|---|

|[Figure 639]|
|---|

- A
- B
- C

- 1st 2nd 3rd 1st 2nd 3rd 1st 2nd 3rd 1st 2nd 3rd

- 1st 3rd 2nd

Expert 1 Expert 2 Expert 3 Expert 4 Expert 5 Expert 6 Expert 7 Expert 8 Expert 9 Expert 10

|[Figure 640]|
|---|

1st 3rd 2nd

1st 2nd 3rd 1st 2nd 3rd 1st 2nd 3rd 1st 2nd 3rd

|[Figure 641]|
|---|

Portrait

- A
- B

|[Figure 642]|
|---|

|[Figure 643]|
|---|

- Expert 1
- Expert 2
- Expert 3
- Expert 4
- Expert 5
- Expert 6
- Expert 7
- Expert 8
- Expert 9

- Expert 10

|[Figure 644]|
|---|

|[Figure 645]|
|---|

2rd 1st 2rd 1st 2rd 1st 2rd 1st 2rd 1st 2rd 1st 2rd 1st 2rd 1st 2rd 1st 2rd 1st

21strd 21strd

Sports

###### A

###### B

|[Figure 647]|
|---|

|[Figure 648]|
|---|

|[Figure 649]|
|---|

|[Figure 650]|
|---|

Expert 1 Expert 2 Expert 3 Expert 4 Expert 5 Expert 6 Expert 7 Expert 8 Expert 9 Expert 10

1st 2rd 1st 2rd

2rd 1st

- 1st 2rd

- 2rd 1st

1st 2rd 1st 2rd 1st 2rd 1st 2rd 1st 2rd 1st 2rd 1st 2rd

Street and City

###### A

###### B

|[Figure 652]|
|---|

|[Figure 653]|
|---|

Expert 1 Expert 2 Expert 3 Expert 4 Expert 5 Expert 6 Expert 7 Expert 8 Expert 9 Expert 10

|[Figure 654]|
|---|

|[Figure 655]|
|---|

2rd 1st 2rd 1st 2rd 1st 2rd 1st 2rd 1st 2rd 1st 2rd 1st 2rd 1st 2rd 1st 2rd 1st

Wildlife

#### A

#### B

|[Figure 657]|
|---|

|[Figure 658]|
|---|

|[Figure 659]|
|---|

|[Figure 660]|
|---|

- Expert 1
- Expert 2
- Expert 3
- Expert 4
- Expert 5
- Expert 6
- Expert 7
- Expert 8
- Expert 9

- Expert 10

2rd 1st 2rd 1st 2rd 1st 2rd 1st 2rd 1st 2rd 1st 2rd 1st 2rd 1st 2rd 1st 2rd 1st

- I.3 Illustration

Anime and Manga

- A
- B

|[Figure 662]|
|---|

|[Figure 663]|
|---|

|[Figure 664]|
|---|

2rd 1st

Expert 1 Expert 2 Expert 3 Expert 4 Expert 5 Expert 6 Expert 7 Expert 8 Expert 9 Expert 10

- 1st 2rd

- 2rd 1st 2rd 1st 2rd 1st 2rd 1st 2rd 1st

|[Figure 665]|
|---|

1st 2rd

2rd 1st 2rd 1st

Comic

- A
- B

|[Figure 667]|
|---|

|[Figure 668]|
|---|

- Expert 1
- Expert 2
- Expert 3
- Expert 4
- Expert 5
- Expert 6
- Expert 7
- Expert 8
- Expert 9

- Expert 10

|[Figure 669]|
|---|

|[Figure 670]|
|---|

2rd 1st

- 1st 2rd

- 2rd 1st 2rd 1st

- 1st 2rd

- 2rd 1st 2rd 1st 2rd 1st 2rd 1st 2rd 1st

1st 2rd

2rd 1st 2rd 1st

Concept Art

A

B

C

D

|[Figure 672]|
|---|

|[Figure 673]|
|---|

|[Figure 674]|
|---|

|[Figure 675]|
|---|

|[Figure 676]|
|---|

|[Figure 677]|
|---|

|[Figure 678]|
|---|

|[Figure 679]|
|---|

3rd 4th 2nd 1st 3rd 4th 2nd 1st

Expert 1 Expert 2 Expert 3 Expert 4 Expert 5 Expert 6 Expert 7 Expert 8 Expert 9 Expert 10

- 3rd 4th 2nd 1st

- 4th 3rd 2nd 1st

3rd 2nd 4th 1st 3rd 4th 1st 2nd 2nd 4th 1st 3rd 4th 3rd 2nd 1st

3rd 4th 1st 2nd

4th 3rd 2nd 1st

Digital and AI Art

A

B

C

D

|[Figure 681]|
|---|

|[Figure 682]|
|---|

|[Figure 683]|
|---|

|[Figure 684]|
|---|

|[Figure 685]|
|---|

|[Figure 686]|
|---|

|[Figure 687]|
|---|

|[Figure 688]|
|---|

4th 3rd 1st 2nd

Expert 1 Expert 2 Expert 3 Expert 4 Expert 5 Expert 6 Expert 7 Expert 8 Expert 9 Expert 10

- 4th 2nd 1st 3rd

- 4th 3rd 2nd 1st 3rd 4th 2nd 1st

4th 3rd 1st 2nd

3rd 4th 2nd 1st

4th 3rd 1st 2nd

3rd 4th 1st 2nd 2nd 4th 1st 3rd

4th 2nd 1st 3rd

Pixel Art

- A
- B
- C
- D

|[Figure 690]|
|---|

|[Figure 691]|
|---|

|[Figure 692]|
|---|

|[Figure 693]|
|---|

|[Figure 694]|
|---|

|[Figure 695]|
|---|

|[Figure 696]|
|---|

|[Figure 697]|
|---|

1st 3rd 2nd 4th

- Expert 1
- Expert 2
- Expert 3
- Expert 4
- Expert 5
- Expert 6
- Expert 7
- Expert 8
- Expert 9

- Expert 10

- 2nd 3rd 1st 4th

- 3rd 2nd 1st 4th 2nd 3rd 1st 4th

2nd 4th 1st 3rd 3rd 1st 2nd 4th

3rd 2nd 1st 4th

3rd 2nd 1st 4th

4th 1st 2nd 3rd

3rd 2nd 1st 4th

Stylized 3D

A

B

C

D

|[Figure 699]|
|---|

|[Figure 700]|
|---|

|[Figure 701]|
|---|

|[Figure 702]|
|---|

|[Figure 703]|
|---|

|[Figure 704]|
|---|

- 1st 2nd 4th 3rd

- 2nd 1st 3rd 4th

Expert 1 Expert 2 Expert 3 Expert 4 Expert 5 Expert 6 Expert 7 Expert 8 Expert 9 Expert 10

- 1st 2nd 3rd 4th

- 1st 3rd 2nd 4th 1st 3rd 2nd 4th

- 1st 2nd 3rd 4th

- 1st 3rd 2nd 4th

|[Figure 705]|
|---|

2nd 1st 3rd 4th

1st 2nd 3rd 4th

1st 3rd 2nd 4th

|[Figure 706]|
|---|

