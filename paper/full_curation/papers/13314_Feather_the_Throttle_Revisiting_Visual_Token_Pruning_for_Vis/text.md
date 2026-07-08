# arXiv:2412.13180v2[cs.CV]31Jul2025

## Feather the Throttle: Revisiting Visual Token Pruning for Vision-Language Model Acceleration

Mark Endo, Xiaohan Wang, Serena Yeung-Levy Stanford University

{markendo,xhanwang,syyeung}@stanford.edu

#### Abstract

- (a)
- (b) FEATHER with 64% FLOPS reduction

###### Visual reasoning

[Figure 1]

[Figure 2]

Recent works on accelerating Vision-Language Models achieve strong performance across a variety of visionlanguage tasks despite highly compressing visual information. In this work, we examine the popular acceleration approach of early pruning of visual tokens inside the language model. Surprisingly, we find that while strong performance is maintained across many tasks, it exhibits drastically different behavior for a subset of vision-centric tasks such as localization. Upon further investigation, we uncover a core issue with the acceleration approach where most tokens towards the top of the image are pruned away. Yet, on many benchmarks aiming to evaluate vision-centric capabilities, strong performance persists with the flawed pruning strategy, highlighting these benchmarks’ limited ability to assess fine-grained visual capabilities. Based on these findings, we propose FEATHER (Fast and Effective Acceleration wiTH Ensemble cRiteria), a straightforward approach that resolves the discovered early-layer pruning issue and further enhances the preservation of relevant tokens via multistage pruning with early uniform sampling to ensure broad image coverage. With comparable computational savings, we find that FEATHER achieves more than 5× performance improvement on the vision-centric localization benchmarks compared to the original acceleration approach.

###### FastV

[Figure 3]

cow

Remaining visual tokens

Q: What animal is in front

after layer 3

of the trees?

###### Localization

[Figure 4]

[Figure 5]

FastV

[Figure 6]

[0.12, 0.333, 0.49, 0.72]

Remaining visual tokens

Q: Where is the player in

after layer 3

white shirt and black shorts?

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

###### FEATHER

[Figure 11]

[0.42, 0.23, 0.7, 0.76]

Remaining visual tokens after layer 16

Q: Where is the player in white shirt and black shorts?

Remaining visual tokens

after layer 8

Figure 1. (a) Although FastV prunes most visual tokens from the upper portion of the image, the approach still displays strong performance on a variety of evaluated vision-language tasks except for a small subset of vision-centric tasks like localization. (b) Based on our findings, we propose FEATHER (Fast and Effective Acceleration wiTH Ensemble cRiteria), a straightforward approach that resolves the existing issue of selecting bottom tokens, additionally maintains uniformly sampled tokens to ensure good coverage over the whole image, and prunes in two stages.

#### 1. Introduction

The exploration of Vision-Language Models (VLMs) is a critical area of computer vision and natural language processing research, centered on combining large language models (LLMs) with visual encoders to enable multimodal perception, reasoning, and understanding capabilities. While earlier works explored sophisticated schemes for conditioning language models with visual information [2, 22, 23], more recently the space has shifted to predominately using the simplistic approach of taking patch features from pre-trained visual encoders and projecting them

to the input space of the language model with a light-weight adapter [4, 27, 29, 38]. Using image patches as tokens, however, comes with the drawback of being computationally inefficient. To achieve fine-grained resolution, the im-

age is divided into many patches. This large number of patches significantly increases computational demands due to the quadratic complexity of the attention operation in Transformers. As a result, many recent works have focused on accelerating these methods by compressing visual information, demonstrating that heavy compression can still maintain strong performance across a wide variety of tasks [3, 7, 10, 31, 39]. For instance, FastV [10] prunes 50% of visual tokens after the shallow layers of the Language Model (LLM) while not compromising in performance across a range of image and video understanding tasks [10]. Another work reveals that for a fixed computational budget on visual reasoning tasks, optimal performance is achieved by using the largest LLM possible while sacrificing visual information, often reducing the visual token count to a single token [24]. With such compressed visual information, it is still unclear how these methods achieve high performance on tasks assessing vision capabilities such as visual reasoning and understanding, and whether there are more demanding visual tasks where these methods fail.

The motivation of our study is to get a better understanding of the vision capabilities of accelerated VLMs given that they leverage highly compressed visual information, focusing specifically on FastV approach. Evaluating across a wide range of vision-language tasks, we find that while the approach maintains strong performance across many tasks, there is a substantial decrease in performance for TextVQA and a severe decrease for localization tasks. When analyzing the approach’s poor performance, we uncover a fundamental issue with the pruning criteria in early layers where tokens towards the top of the image are disproportionally removed due to the long-term decay property of RoPE. Since this issue is not task-specific, we examine how performance is maintained across most evaluated tasks and find that even when tokens are entirely discarded using the flawed criteria, performance remains largely unchanged. This highlights how many benchmarks require minimal visual grounding. Our finding underscores a significant challenge in the field of multimodal learning, not only for measuring the effectiveness of VLM acceleration methods but also for benchmarking the visual capabilities of VLMs as a whole.

Given the discovered limitation of the studied approach, we next explore whether the attention-based criteria can be modified to enable robust visual token pruning, leading to our final method, FEATHER (Fast and Effective Acceleration wiTH Ensemble cRiteria). Specifically, we propose a straightforward modification to the attentionbased criteria by removing RoPE, demonstrating that this resolves the positional bias issue and greatly improves performance. Additionally, we incorporate uniform sampling in early token pruning to ensure adequate coverage of all image regions and apply more extensive pruning at a later layer, when the attention-based criteria better discerns im-

portant tokens. This strategy is analogous to how a racecar driver feathers the throttle by gradually pressing the accelerator at the beginning of a turn to maintain grip and then accelerating more aggressively once the car is past the apex. We show that FEATHER results in substantial performance gains compared to the original acceleration approach, improving localization performance more than 5× with comparable computational savings. Strikingly, we find that our approach achieves this performance improvement while only retaining 3.3% of visual tokens for the second half of LLM layers. Overall, our work demonstrates that while visual compression can maintain strong performance even on challenging vision-centric tasks, its effectiveness depends on a well-designed strategy, which is currently difficult to assess due to many vision-language benchmarks not thoroughly evaluating vision capabilities. In summary, our contributions include:

- • Revealing inconsistent effect of early pruning: Our evaluations across 12 benchmarks show that early visual token pruning minimally affects performance on most tasks, but causes a substantial dropoff for TextVQA and a drastic decline for localization tasks.
- • Analyzing poor performance: We uncover that the attention-based criteria for selecting important tokens exhibits a bias towards bottom image tokens in early layers due to the long-term decay property of RoPE.
- • Identifying benchmark limitations in assessing token pruning: We find that performance remains strong even with suboptimal pruning criteria and removal of pruned tokens before transferring information in the LLM, demonstrating how many benchmarks fail to assess fine-grained visual capabilities.
- • Improving attention-based criteria: With the straightforward modification of removing RoPE, we demonstrate that attention can be used as an effective criteria for selecting tokens and performance can be further improved with simple strategies like incorporating uniform sampling and pruning more extensively at a later layer, when the criteria better discerns important tokens.

#### 2. Related Work

Recent efforts to accelerate VLMs can be broadly divided into two main categories: compressing visual information before it enters the LLM, and compressing visual information within the LLM itself. In the first category, ChatUniVi [17] dynamically merges visual tokens with similar semantic meanings. PruMerge [31], FasterVLM [44], and VisionZip [40] select important tokens based on ViT attention ([CLS] or average attention received). PruMerge then merges important tokens with unselected ones while VisionZip merges unselected tokens together. For the LLaVA-NeXT [28] approach of partitioning an image into

sub-images where inefficiency is an even bigger problem, HiRED [3] selects tokens with top feature important on each sub-image with an allocated budget. Other methods argue that the input image alone does not include enough information to select important patches and thus use the textual input to recover visually meaningful tokens [11, 41]. Alternatively, some approaches reduce visual information within the ViT by merging or pruning tokens [6, 8, 30].

For the second category of approaches, where visual information is compressed within the LLM itself, LOOKM reduces the multimodal KV cache size [36]. In our study, we focus on the popular FastV approach [10]. This work identifies that attention over image tokens is sparse in deeper layers of the LLM. Based on this observation, they propose to prune away unimportant vision tokens after the shallow layers of the LLM, achieving a 45% reduction in FLOPS with nearly no performance loss. Works since have proposed alterations to this setup such as adaptively determining the number of pruned tokens instead of using a fixed ratio [14] or pruning in multiple stages [39].

#### 3. Revising Visual Token Pruning

In this section, we aim to get a better understanding of the vision capabilities of the VLM acceleration approach of pruning visual tokens after shallow LLM layers. After outlining preliminaries (§3.1), we take a closer look at how the approach performs across a broad range of tasks. Upon inspection, we discover that while heavily pruning visual tokens after the shallow LLM layers has little effect on performance across a variety of tasks, this approach fails decisively on more vision-centric tasks, particularly localization (§3.2). Next, we examine why this method struggles with localization, uncovering that the poor performance is due to the ineffectiveness of the pruning criteria to select important tokens when applied at an early layer (§3.3). As this defect is not specific to localization, we explore what can be attributed to the method’s high performance on numerous other tasks and find that these tasks require minimal visual grounding (§3.4).

##### 3.1. Preliminaries

Before our analyses, we provide a background on the explored adapter-style VLM with visual token pruning, the evaluated benchmarks, and experimental settings.

VLM and token pruning. Formally, an adapter-style VLM takes as input an image ximg and text prompt tokens xprompt. A pre-trained vision backbone f first encodes visual features zimg = f(ximg) ∈ Rn×d

vision where n is the number of image patches and dvision is the dimensionality of the vision encoder. Next, an adapter p (either a simple linear layer or MLP) projects the vision features to embeddings himg = p(zimg) ∈ Rn×d

text where dtext is the dimensionality of the LLM. Lastly, himg is concatenated with text

prompt embeddings hprompt = embed(xprompt) and passed into the language model LM to generate the output text y = LM([himg;hprompt]).

In this work, we study the inference acceleration of VLMs where visual tokens are pruned within the attention mechanism of LM. We focus on the FastV [10] approach, where after layer K in LM, R% of visual tokens are pruned away based on a ranking function gϕ which ranks tokens based on criteria ϕ. In practice, the attention score received from the last text token is used as the criteria, referred to as ϕoriginal. Note that the positional information is preserved when performing the pruning.

With this approach, we measure the acceleration using the theoretical FLOPS reduction ratio related to the image tokens. For one Transformer layer, the total FLOPS is estimated as C = 4nd2 + 2n2d + 2ndm where d = dtext and

- m is the intermediate size of FNN. Given that the Transfoerm has T layers in total and after layer K, we maintain
- nˆ = (1 − R%) ∗ n visual tokens, we calculate the FLOPS reduction as

K ∗ C + (T − K) ∗ (4ˆnd2 + 2ˆn2d + 2ˆndm) C

. (1)

1 −

Benchmarking. We evaluate the accelerated VLMs on a large suite of benchmarks from [18] which includes evaluations spanning the areas of localization, open-ended visual question answering, and challenge sets. For all benchmarks, we follow the same evaluation protocol as [18].

Localization. We evaluate localization performance using the RefCOCO, RefCOCO+, RefCOCOg [19, 42] and OCID-Ref [37] datasets. RefCOCO contains short, spatially grounded descriptions, while RefCOCO+ focuses on appearance-based descriptions. RefCOCOg, on the other hand, features longer and more detailed descriptions. OCID-Ref is a robotics dataset that tests generalization to out-of-distribution scenarios, particularly for object localization in cluttered environments.

Open-Ended Visual Question Answering. We evaluate general visual reasoning using the VizWiz [5] and VQAv2 [13] datasets, spatial reasoning using the GQA [16] dataset, and reasoning around text using the TextVQA [32] dataset. For TextVQA, following [18], we exclude OCR-system parsed input tokens to better evaluate the effect of pruning on visual capabilities.

Challenge Sets. We additionally evaluate on the VSR [26], TallyQA [1], POPE [25], and AI2D [20] closed-set prediction datasets. VSR includes binary spatial relationship questions, TallyQA consists of counting questions, POPE probes hallucination, and AI2D contains multiplechoice questions referring to scientific diagrams and charts.

Experimental settings. In our experiments, we utilize a VLM with SigLIP ViT-SO400M [43] as f, a one-layer MLP with GELU activation as p, and Llama 2 7B [35] as LM. The model was trained on the multimodal instruction tuning dataset presented in [27] in a single-stage with f frozen.

Challenge

Sets

Open-Ended Visual Question

Answering

| |
|---|

###### Localization

- Figure 2. Contrasting the difference in performance dropoff on the challenging vision-centric localization task (Left) versus the other evaluated tasks (Middle) when pruning visual tokens after the shallow LLM layers. Whereas performance decrease is minimal for most tasks, localization exhibits roughly a linear decrease to zero as the ratio of pruned tokens increases. Right: Per-task performance breakdown across various setups of pruning ratios. Using K = 3 for all pruning setups.

##### 3.2. Impact of early visual token pruning varies drastically by task

##### 3.3. Interpreting poor vision-centric task performance

We next investigate why pruning visual tokens after the shallow LLM layers has such an adverse effect on localization performance. Intuitively, even localization should not require all image tokens, as there are background tokens irrelevant to identifying specific objects that can be pruned. Thus, it is surprising that performance begins to degrade even with minimal token pruning. This suggests that the pruning criteria, intended to retain important tokens, fails to effectively distinguish which tokens contain necessary visual information.

We examine the effect of pruning visual tokens after the shallow LLM layers (K = 3) across a variety of tasks. We compare various pruning ratios (R ∈ {0.25,0.5,0.75,0.9}) and the baseline non-pruned model in Figure 2.

For the majority of evaluated tasks, heavily reducing the number of visual tokens results in minimal performance decrease (3.5% for AI2D, 0.1% for VSR, 3.7% for TallyQA,

- 4.8% for POPE, 5.1% for VizWiz, 7.9% for VQAv2 and 7.7% for GQA when dropping 75% of tokens). However, for a small subset of tasks–particularly TextVQA, and more severely, localization–performance drops substantially, with decreases of 42.0%, 88.9%, 88.9%, 91.0%, and 86.0% for TextVQA, RefCOCO, RefCOCO+, RefCOCOg, and OCID-Ref, respectively. Investigating further, Figure 2 shows that localization performance declines roughly linearly to zero as the pruning ratio progresses from 0 to 1, a pattern that strongly contrasts with most other tasks, where performance remains largely unaffected. While prior works report a much smaller decline in TextVQA performance [9, 39], they incorporate OCR-parsed input tokens into prompts, reducing the reliance on visual information. Discussion. Although localization is undoubtedly challenging for a model that discards visual tokens, as it requires fine-grained visual grounding and precise boundary identification, it is still surprising that (1) localization performance declines to the extent it does even when many tokens are maintained (e.g., dropping 50% of tokens reduces average performance by 45%), and (2) most non-localization benchmarks, except for TextVQA, exhibit a drastically different pattern despite these benchmarks aiming to evaluate visual grounding abilities, such as counting for TallyQA, spatial reasoning for GQA, and chart understanding for AI2D. In the following sections, we seek to explain these unexpected findings and understand their implications.

To assess the efficacy of the pruning criteria, we first examine the distribution of retained tokens across all benchmark examples. As shown in Figure 3(b), we find that pruning visual tokens after the shallow LLM layers (K = 3) retains tokens concentrated at the bottom of the image. For example, when R = 0.75, the average sampled token position across all datasets is situated 80.7% of the way down the image. A Chi-Square test confirms the non-uniformity of selected tokens’ y-positions (p-value < 0.05). We also examine the criteria behavior when pruning after later layers (K ∈ (8,16,24)) and observe that the bias of selecting bottom image tokens is reduced.

In addition, we observe that as the pruning layer increases, not only is the positional bias reduced, but the criteria also begins to correctly select tokens relevant to the text instruction. For example, as shown in Figure 3(a), pruning after the later layers results in a distinct concentration of selected tokens around the area corresponding to the reference expression for localization. To quantify this change in criteria effectiveness across layers, we measure the performance when varying the pruning layer (K ∈ {3,8,16,24}). As shown in Figure 3(c), performance remains stable when pruning is performed at deeper layers (K = 16 and K = 24), whereas pruning at earlier layers (K = 3 and K = 8)

(a) (b)

(c)

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

| |
|---|

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

| |
|---|

[Figure 24]

- Figure 3. (a) Example demonstrating that when pruning in early layers, selected tokens are concentrated in the bottom of the image, whereas in later layers the selected tokens more precisely cover the region of the image important for answering the question. (b) Heatmap illustrating that averaged across all benchmark examples, as the pruning layer increases, the selection of bottom visual tokens by the criteria is reduced. (c) Visualizing the effect of the pruning layer on performance for both localization and non-localization tasks. We find that pruning after layer 16 or later results in little performance dropoff, whereas pruning earlier results in a performance decrease, particularly for localization. Using R = 0.75 for all setups.

leads to performance degradation, particularly for localization.

FastV (K=3, R=0.75)

Baseline

Pruning before LLM

Text Only

| |
|---|

| |
|---|

| |
|---|

Discussion. While previous research attributes the poor performance of early token pruning to the low redundancy of image tokens in shallow layers [39], we reveal a critical insight: the variable performance across pruning layers is linked to the effectiveness of selecting important tokens, as early token pruning leads to suboptimal token selection biased toward bottom tokens. To understand why the criteria disproportionately retains bottom image tokens, we must consider the LM architecture. Specifically, token selection is based on the attention received from the last text token. Since the LM uses RoPE to encode positional information [33], it exhibits a long-term decay property. While this property is well-suited for language modeling, we uncover that it poses an issue for selecting visual tokens, where tokens appearing later in the flattened, rasterscan order have higher attention scores purely due to their position. In language modeling, prior work finds that there is greater emphasis on shorter-distance information in shallow layers [15]. We predict that the visually-conditioned LM exhibits the same behavior, explaining why the positional bias is most pronounced when pruning after shallow layers.

0.8

0.6

Accuracy

0.4

0.2

0.0

GQA VQAv2 VizWiz POPE TallyQA VSR AI2D TextVQA

Benchmark

Figure 4. Assessing whether the strong performance of early visual token pruning for many tasks can be attributed to visual information transfer before pruning or benchmarks’ lack of assessing fine-grained visual capabilities. We observe that for all evaluated tasks, allowing information transfer before pruning (shown in green) does not result in substantial performance improvement over the setup without visual information transfer (shown in light green), highlighting a limitation of many benchmarks.

mance across a diverse range of tasks, including those aiming to evaluate visual grounding abilities? In this section, we test the following contrasting explanations:

##### 3.4. Explaining strong performance on other tasks

Given that early pruning of visual tokens leads to a suboptimal tokens selection where remaining tokens are concentrated towards the bottom of the image, a natural question arises: how does the approach still maintain high perfor-

1. Important visual information from top image tokens is transferred to later tokens before pruning with the unidirectional attention of the LLM.

2. Many questions can still be inferred with access to only the suboptimal set of visual tokens primarily from the bottom of the image.

To evaluate these hypotheses across tasks, we save the tokens retained by the studied pruning approach and, in a subsequent run, remove all other tokens before they reach the LLM, ensuring that no visual information from the pruned tokens is retained. As an additional comparison, we also evaluate a text-only model in which all visual tokens are removed before entering the LLM. As shown in Figure 4, we find that across all evaluated tasks, removing pruned tokens before the LLM results in little to no performance decrease compared to the method that allows information transfer in shallow layers prior to pruning. Discussion. This finding indicates that the high performance of early visual token pruning on many tasks does not stem from the effectiveness of visual information transfer in early layers but rather signifies that many benchmarks do not demand a detailed understanding of visual information to answer questions accurately. Note that for the majority of these benchmarks (except VizWiz and AI2D), performance substantially declines when not including any visual information, signifying that simply comparing visionenabled and disabled setups is insufficient to determine a benchmark’s ability to assess visual grounding.

#### 4. Improving Visual Token Pruning

Based on our findings from §3, we now seek to answer: can attention serve as an effective criteria for token pruning even in early layers? To this end, we propose a straightforward solution to removing the positional bias of the attention-based criteria and demonstrate that our modifications enable effective pruning even when applied after early LLM layers (§4.1). Guided by our insights, we present our final approach, FEATHER (Fast and Effective Acceleration wiTH Ensemble cRiteria) (§4.2), and showcase its impressive visual capabilities while offering high computational efficiency (§4.3).

##### 4.1. Assessing potential of attention-based criteria

An attention-based criteria offers the potential to naturally identify and retain only the most relevant visual tokens for answering a question. However, as shown in §3, positional bias strongly influences token selection, limiting its effectiveness. Thus, we aim to resolve the positional bias issue and explore the true promise of attention-based criteria. We compare against non-attention-based approaches that incorporate an explicit inductive bias toward reducing redundancy. Below, we outline the explored criteria.

ϕ-R: Given the long-term decay property of RoPE, we propose the following straightforward adjustment to the pruning criteria in order to mitigate this bias and more effectively select tokens relevant to the text instruction. Namely,

we still compute the attention score received from the last text token, except we do not apply RoPE to the attention mechanism, thereby removing the long-term decay effect. Note that this lightweight calculation is only done once per pruning, and all other state computations are compatible with FlashAttention [12].

ϕuniform: In this simple criteria, we uniformly sample visual tokens in the image, using a set stride. This criteria ensures that there is good image coverage, but does so by sacrificing the ability to have more densely captured visual information for a particular image region. To use a similar computational saving to the attention-based criteria when R = 0.75, we use a stride of two, resulting in 196 selected tokens.

ϕKNN: Inspired by dynamic visual tokenization [17], we select tokens based on local density. For each visual token zimg[i] where i ∈ {0,1,...,n − 1}, we compute its local density ρi using K-nearest neighbors and calculate distance index δi as:

 

∥zimg[i] − zimg[j]∥2, if ∃j s.t. ρj > ρi. max j

min

j:ρj>ρi

(2)

δi =



∥zimg[i] − zimg[j]∥2, otherwise.

δi represents how far away zimg[i] is from other highdensity tokens. We use ρi ∗δi as the token importance score for the criteria and select 196 tokens.

ϕ-R + ϕuniform: Since ϕ-R focuses on selecting important tokens and ϕuniform aims to reduce redundancy, we propose an ensemble of these two to leverage their respective strengths. Specifically, we apply ϕ-R while incorporating a small number of uniformly sampled tokens using a stride of three.

Improvement of removing RoPE: In Table 1, we report results evaluating the effectiveness of these criteria. Comparing ϕ-R with ϕoriginal, we find that by removing RoPE, at K = 3 there is a 183% average improvement on localization tasks and at K = 8, there is a 17% average improvement. These performance improvements demonstrate that once the impact of token position on attention is removed, attention score can more effectively be used for the criteria when applied after early LLM layers. Note that the narrowing gap in performance between ϕ-R and ϕoriginal from K = 3 to K = 8 is likely because ϕoriginal has less of a bias towards selecting bottom image tokens as the pruning layer increases.

Token selection improves when pruning later: Comparing ϕ-R when K = 8 versus K = 3, we see that the average localization performance improves by 63%. However, it remains unclear whether this performance increase is affected by factors other than the ability of the criteria to select important tokens. Therefore, to directly compare the criteria across different layers, we take the tokens selected by both criteria and evaluate how well the model can perform with only these tokens passed into the LLM (as is

Localization Open-Ended VQA Challenge Sets

FLOPSRed

RefCOCO+

RefCOCOg

RefCOCO

OCID-Ref

TextVQA

TallyQA

VQAv2

VizWiz

POPE

AI2D

GQA

Pruning Layer Criteria

VSR

Avg

Avg

Avg

Attention-based

ϕoriginal 68% 5.9 5.7 5.1 6.1 6.7 54.8 31.8 58.4 72.7 56.3 64.0 83.2 57.1 63.3 52.4 ϕ-R (Ours) 68% 16.7 22.9 15.1 13.3 15.3 59.0 41.6 61.2 76.0 57.3 64.7 85.2 58.2 62.2 53.2

∆ +10.7 +17.2 +10.0 +7.3 +8.6 +4.2 +9.7 +2.8 +3.2 +1.1 +0.7 +1.9 +1.1 -1.1 +0.8

Non-attention-based

K = 3

ϕKNN 66% 23.9 15.1 24.9 26.0 29.6 58.4 39.9 60.9 74.4 58.4 62.8 81.2 55.9 61.5 52.8 ϕuniform 66% 28.0 20.6 28.6 29.7 33.3 59.0 41.4 61.8 75.9 57.1 64.6 85.2 58.1 61.9 53.0

Ensemble

###### ϕ-R + ϕuniform (Ours) 61% 27.2 29.1 27.2 24.7 27.7 61.2 46.6 62.3 77.4 58.4 65.4 86.0 58.9 62.7 54.0

Attention-based

ϕoriginal 56% 23.3 19.4 23.5 24.0 26.3 59.8 45.0 60.3 76.1 57.8 64.6 85.4 57.5 62.6 53.0 ϕ-R (Ours) 56% 27.3 27.1 26.7 26.4 29.2 61.4 49.0 61.5 77.4 57.8 65.5 86.7 58.6 63.0 53.7

∆ +4.0 +7.6 +3.2 +2.5 +2.9 +1.6 +4.0 +1.2 +1.3 +0.0 +0.8 +1.3 +1.1 +0.4 +0.6

Non-attention-based

K = 8

ϕKNN 55% 23.6 15.4 24.4 25.2 29.4 58.6 40.2 61.1 74.5 58.5 62.9 81.4 56.2 60.9 53.0 ϕuniform 55% 30.3 24.6 31.0 30.9 34.8 59.3 42.2 61.8 76.0 57.4 64.4 85.3 57.9 61.0 53.2

Ensemble

###### ϕ-R + ϕuniform (Ours) 50% 35.6 32.0 35.9 35.4 38.8 62.7 51.7 62.4 78.1 58.6 66.0 87.4 59.1 63.6 54.0

- Table 1. Evaluating alternative criteria for token pruning after the early LLM layers. For each task and pruning layer, we bold the best result and underline the second-best result. Our main findings include: (1) our proposed RoPE-free criteria ϕ-R substantially improves pruning performance compared to the original criteria ϕoriginal; (2) pruning later (K = 8) yields higher performance than pruning earlier (K = 3); and (3) integrating uniform sampling into the attention-based criteria with ϕ-R +ϕuniform enhances effectiveness. Using R = 0.75 for ϕoriginal, ϕ-R, and ϕ-R + ϕuniform. See §4.1 for criteria definitions.

K OCID-Ref RefCOCOg RefCOCO+ RefCOCO

3 23.8 16.2 14.5 16.3 8 26.7 29.8 26.8 29.8

- Table 2. Assessing localization performance when using the selected tokens from ϕ-R for the entirety of the LLM. We find that ϕ-R applied at a later LLM layer results in a better token selection.

coverage. When we combine these two criteria types with ϕ-R + ϕuniform, we find that when K = 3 this approach improves upon localization performance of ϕ-R by 63% but has a slight decrease compared to ϕuniform by 2.9%. However, when K = 8, this approach far outperforms both of the individual criteria, improving ϕ-R by 30% and ϕuniform by 17%. Note that this setup results in slightly less computation efficiency than ϕ-R (7% drop in FLOPS reduction for K = 3 and 6% for K = 16), but this is outweighed by the performance improvements.

done in §3.4). As shown in Table 2, we find the selected tokens from ϕ-R applied after layer 8 are superior compared the tokens from ϕ-R applied after layer 3.

We provide qualitative examples of various criteria token selection and additional results using a different model setup in the supplement.

Integrating uniform sampling with attention-based criteria is beneficial in early layers: When comparing the attention-based criteria ϕ-R with the two non-attention based criteria ϕKNN and ϕuniform, we find that ϕ-R results in better performance on some tasks (e.g., OCID-Ref, TextVQA), while ϕKNN and ϕuniform outperform on other tasks (e.g., RefCOCO). This varying effectiveness of criteria types presumably comes from some tasks requiring a more detailed understanding of specific image regions while others benefit from a broader understanding with full image

##### 4.2. Distilling insights

Guided by our insights, we now present our final approach FEATHER (Fast and Effective Acceleration wiTH Ensemble cRiteria). In this approach, we first perform pruning after an early layer (K = 8), utilizing our proposed criteria ϕ-R + ϕuniform to retain (1 − R)% of tokens. Given our finding that the criteria improves as the LLM layer increases, we additionally prune a second time at K = 16.

Challenge Sets

Open-Ended

Visual Question

Answering

| |
|---|

###### Localization

Figure 5. Comparing FEATHER performance against FastV [10] and PyramidDrop [39]. We find that FEATHER far outperforms both compared methods, particularly for the vision-centric task of localization.

#### 5. Conclusion

At this stage, we utilize ϕ-R as the criteria, as uniform sampling should not be necessary when the attention-based criteria reliably selects important tokens. For the reduction ratio, we choose to only retain (1 − R)2% of the remaining tokens since the attention-based criteria has proved highly effective when pruning at later layers, even when using the original criteria ϕoriginal (see Figure 3(c)).

In this work, we examine the visual capabilities of the VLM acceleration approach of pruning visual tokens after shallow LLM layers. While strong performance is maintained on most evaluated tasks, it fails on more vision-centric tasks like TextVQA and localization due to its flawed pruning criteria that predominately selects visual tokens from the bottom part of the image. Observing this same flawed criteria on other tasks, we show that strong performance is largely due to the benchmarks’ inability to assess fine-grained visual capabilities. Next, we propose and evaluate several alternative criteria to improve visual capabilities, ultimately arriving at our final method, FEATHER. This approach refines the attention-based criteria to address the token selection bias while using uniform sampling for better image coverage. It then prunes more aggressively when the criteria more effectively identifies important tokens. We find that FEATHER has more than 5× performance improvement on localization compared to the original acceleration approach.

##### 4.3. Comparison against FastV and PyramidDrop

We compare our approach against the one-stage early pruning approach of FastV (K = 3) [10] and the multi-stage approach of PyramidDrop (K = [8,16,24]) [39]. Note that both of these methods use ϕoriginal for the criteria.

As shown in Figure 5, we find that FEATHER far outperforms the baselines, especially on the localization tasks. Specifically, for comparable computational costs (64% FLOPS reduction for our approach, 68% FLOPS reduction for FastV, and 65% FLOPS reduction for PyramidDrop), we observe that for localization tasks, FEATHER exhibits more than 5× average performance improvement compared to FastV and a 36% average performance improvement compared to PyramidDrop. For non-localization tasks, FEATHER has a 7.8% improvement over FastV and 1.5% improvement over PyramidDrop. Note that PyramidDrop performs substantially better than FastV as it prunes fewer tokens in an early layer. However, it still suffers from an ineffective pruning strategy at this stage, though the impact is less pronounced since it predominantly prunes later.

While we study a particular type of VLM acceleration, our work highlights a broad challenge in evaluating VLMs. Building on [34], which shows that text-only models can perform well on some vision-language benchmarks, we demonstrate that even benchmarks with substantial differences between vision-enabled and disabled setups may not assess fine-grained visual capabilities. We address this issue by focusing on more vision-centric localization tasks, though this limits the analysis to a few specific types of skills. To accurately assess a wider range of visual capabilities, future work may explore how to resolve current dataset biases that models can exploit [21]. Additionally, while removing RoPE for token pruning effectively eliminates positional bias, it may introduce unintended effects on attention weights influencing token selection. Future research could investigate more robust methods for encoding positional information in visually conditioned language models to prevent positional artifacts in cross-modal interactions.

Remarkably, with our 64% FLOPS reduction setup, after layer 16 only 3.3% of tokens are retained, yet the average localization performance decrease compared to the baseline method with no token pruning is only 26% (Figure 1 includes an example of retained tokens after layer 16). This finding illustrates that even for vision-centric tasks, maintaining strong performance while gaining huge acceleration speedups with extensive pruning is possible, but it heavily relies on the effectiveness of the pruning criteria.

Acknowledgments. This work is supported in part by the National Science Foundation (NSF) under Grant No. 2026498, the Stanford AIMI-HAI Partnership Grant, and the NSF Graduate Research Fellowship Program under Grant No. DGE-2146755 (for M.E.). Any opinions, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of any other entity.

#### References

- [1] Manoj Acharya, Kushal Kafle, and Christopher Kanan. Tallyqa: Answering complex counting questions. In Proceedings of the AAAI conference on artificial intelligence, pages 8076–8084, 2019. 3
- [2] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. Advances in neural information processing systems, 35:23716–23736,

2022. 1

- [3] Kazi Hasan Ibn Arif, JinYi Yoon, Dimitrios S Nikolopoulos, Hans Vandierendonck, Deepu John, and Bo Ji. Hired: Attention-guided token dropping for efficient inference of high-resolution vision-language models in resourceconstrained environments. arXiv preprint arXiv:2408.10945,

2024. 2, 3

- [4] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. arXiv preprint arXiv:2308.12966, 2023. 1
- [5] Jeffrey P Bigham, Chandrika Jayant, Hanjie Ji, Greg Little, Andrew Miller, Robert C Miller, Robin Miller, Aubrey Tatarowicz, Brandyn White, Samual White, et al. Vizwiz: nearly real-time answers to visual questions. In Proceedings of the 23nd annual ACM symposium on User interface software and technology, pages 333–342, 2010. 3
- [6] Daniel Bolya, Cheng-Yang Fu, Xiaoliang Dai, Peizhao Zhang, Christoph Feichtenhofer, and Judy Hoffman. Token merging: Your vit but faster. In The Eleventh International Conference on Learning Representations, 2023. 3
- [7] Mu Cai, Jianwei Yang, Jianfeng Gao, and Yong Jae Lee. Matryoshka multimodal models. arXiv preprint arXiv:2405.17430, 2024. 2
- [8] Wenhao Chai, Enxin Song, Yilun Du, Chenlin Meng, Vashisht Madhavan, Omer Bar-Tal, Jeng-Neng Hwang, Saining Xie, and Christopher D Manning. Auroracap: Efficient, performant video detailed captioning and a new benchmark. arXiv preprint arXiv:2410.03051, 2024. 3
- [9] Jieneng Chen, Luoxin Ye, Ju He, Zhao-Yang Wang, Daniel Khashabi, and Alan Yuille. Efficient large multi-modal models via visual context compression. In The Thirty-eighth Annual Conference on Neural Information Processing Systems,

2024. 4

- [10] Liang Chen, Haozhe Zhao, Tianyu Liu, Shuai Bai, Junyang Lin, Chang Zhou, and Baobao Chang. An image is worth 1/2

- tokens after layer 2: Plug-and-play inference acceleration for large vision-language models. ECCV, 2024. 2, 3, 8
- [11] Yi Chen, Jian Xu, Xu-Yao Zhang, Wen-Zhuo Liu, YangYang Liu, and Cheng-Lin Liu. Recoverable compression: A multimodal vision token recovery mechanism guided by text information. arXiv preprint arXiv:2409.01179, 2024. 3
- [12] Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher R´e. Flashattention: Fast and memory-efficient exact attention with io-awareness. Advances in neural information processing systems, 35:16344–16359, 2022. 6
- [13] Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 6904–6913, 2017. 3
- [14] Yefei He, Feng Chen, Jing Liu, Wenqi Shao, Hong Zhou, Kaipeng Zhang, and Bohan Zhuang. Zipvl: Efficient large vision-language models with dynamic token sparsification and kv cache compression. arXiv preprint arXiv:2410.08584, 2024. 3
- [15] Xiangyu Hong, Che Jiang, Biqing Qi, Fandong Meng, Mo Yu, Bowen Zhou, and Jie Zhou. On the token distance modeling ability of higher rope attention dimension. arXiv preprint arXiv:2410.08703, 2024. 5
- [16] Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6700–6709, 2019. 3
- [17] Peng Jin, Ryuichi Takanobu, Wancai Zhang, Xiaochun Cao, and Li Yuan. Chat-univi: Unified visual representation empowers large language models with image and video understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13700– 13710, 2024. 2, 6
- [18] Siddharth Karamcheti, Suraj Nair, Ashwin Balakrishna, Percy Liang, Thomas Kollar, and Dorsa Sadigh. Prismatic VLMs: Investigating the design space of visuallyconditioned language models. In Proceedings of the 41st International Conference on Machine Learning, pages 23123–

23144. PMLR, 2024. 3

- [19] Sahar Kazemzadeh, Vicente Ordonez, Mark Matten, and Tamara Berg. Referitgame: Referring to objects in photographs of natural scenes. In Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP), pages 787–798, 2014. 3
- [20] Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11–14, 2016, Proceedings, Part IV 14, pages 235–

251. Springer, 2016. 3

- [21] Baiqi Li, Zhiqiu Lin, Wenxuan Peng, Jean de Dieu Nyandwi, Daniel Jiang, Zixian Ma, Simran Khanuja, Ranjay Krishna, Graham Neubig, and Deva Ramanan. Naturalbench: Evaluating vision-language models on natural adversarial samples. Advances in Neural Information Processing Systems, 37:17044–17068, 2025. 8

- [22] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In International conference on machine learning, pages 12888–12900. PMLR, 2022. 1
- [23] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International conference on machine learning, pages 19730–

19742. PMLR, 2023. 1

- [24] Kevin Y Li, Sachin Goyal, Joao D Semedo, and J Zico Kolter. Inference optimal vlms need only one visual token but larger models. arXiv preprint arXiv:2411.03312, 2024. 2
- [25] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. arXiv preprint arXiv:2305.10355, 2023. 3
- [26] Fangyu Liu, Guy Emerson, and Nigel Collier. Visual spatial reasoning. Transactions of the Association for Computational Linguistics, 11:635–651, 2023. 3
- [27] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26296–26306, 2024. 1, 3
- [28] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, 2024. 2
- [29] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36, 2024. 1
- [30] Yifei Liu, Mathias Gehrig, Nico Messikommer, Marco Cannici, and Davide Scaramuzza. Revisiting token pruning for object detection and instance segmentation. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 2658–2668, 2024. 3
- [31] Yuzhang Shang, Mu Cai, Bingxin Xu, Yong Jae Lee, and Yan Yan. Llava-prumerge: Adaptive token reduction for efficient large multimodal models. arXiv preprint arXiv:2403.15388,

2024. 2

- [32] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8317–8326, 2019. 3
- [33] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063,

2024. 5

- [34] Shengbang Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Manoj Middepogu, Sai Charitha Akula, Jihan Yang, Shusheng Yang, Adithya Iyer, Xichen Pan, et al. Cambrian1: A fully open, vision-centric exploration of multimodal llms. arXiv preprint arXiv:2406.16860, 2024. 8
- [35] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al.

- Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023. 3
- [36] Zhongwei Wan, Ziang Wu, Che Liu, Jinfa Huang, Zhihong Zhu, Peng Jin, Longyue Wang, and Li Yuan. Look-m: Lookonce optimization in kv cache for efficient multimodal longcontext inference. arXiv preprint arXiv:2406.18139, 2024. 3
- [37] Ke-Jyun Wang, Yun-Hsuan Liu, Hung-Ting Su, Jen-Wei Wang, Yu-Siang Wang, Winston Hsu, and Wen-Chin Chen. OCID-ref: A 3D robotic dataset with embodied language for clutter scene grounding. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 5333–5338, Online, 2021. Association for Computational Linguistics. 3
- [38] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024. 1
- [39] Long Xing, Qidong Huang, Xiaoyi Dong, Jiajie Lu, Pan Zhang, Yuhang Zang, Yuhang Cao, Conghui He, Jiaqi Wang, Feng Wu, et al. Pyramiddrop: Accelerating your large vision-language models via pyramid visual redundancy reduction. arXiv preprint arXiv:2410.17247, 2024. 2, 3, 4, 5, 8
- [40] Senqiao Yang, Yukang Chen, Zhuotao Tian, Chengyao Wang, Jingyao Li, Bei Yu, and Jiaya Jia. Visionzip: Longer is better but not necessary in vision language models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 19792–19802, 2025. 2
- [41] Gaotong Yu, Yi Chen, and Jian Xu. Balancing performance and efficiency: A multimodal large language model pruning method based image text interaction. arXiv preprint arXiv:2409.01162, 2024. 3
- [42] Licheng Yu, Patrick Poirson, Shan Yang, Alexander C Berg, and Tamara L Berg. Modeling context in referring expressions. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part II 14, pages 69–85. Springer, 2016. 3
- [43] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 11975–11986, 2023. 3
- [44] Qizhe Zhang, Aosong Cheng, Ming Lu, Zhiyong Zhuo, Minqi Wang, Jiajun Cao, Shaobo Guo, Qi She, and Shanghang Zhang. [cls] attention is all you need for training-free visual token pruning: Make vlm inference faster. arXiv eprints, pages arXiv–2412, 2024. 2

(a) Criteria evaluated with Dinov2-SigLIP

(b) Criteria performance on non-

encoder

localization tasks (with SigLIP encoder)

## Feather the Throttle: Revisiting Visual Token Pruning for Vision-Language Model Acceleration

### Supplementary Material

#### A1. Results on Different Model Setup

In addition, we show performance with respect to total runtime on a NVIDIA L40S in Figure A1.

We additionally experiment with using a DINOv2 + SigLIP visual encoder. As shown in Table A1, we observe the same behavior that removing RoPE substantially improves performance and incorporating uniform sampling is strong.

#### A2. Additional FEATHER Results

We compare FEATHER performance against FastV and PyramidDrop on all evaluated benchmarks in Table A2.

Figure A1. Total runtime on L40S vs. performance for FastV, PyramidDrop, and FEATHER.

Localization Open-Ended VQA Challenge Sets

FLOPSRed

RefCOCO+

RefCOCOg

RefCOCO

OCID-Ref

TextVQA

TallyQA

VQAv2

VizWiz

POPE

AI2D

GQA

VSR

Avg

Avg

Avg

Criteria

Attention-based

ϕoriginal 68% 27.2 21.9 27.7 27.8 31.1 56.6 35.6 59.1 74.0 57.7 66.1 84.6 60.2 67.1 52.7 ϕ-R 68% 37.2 37.0 38.7 34.9 38.1 60.1 45.4 60.4 76.5 58.2 66.3 85.9 61.1 65.1 52.9

∆ +10.0 +15.0 +11.0 +7.0 +7.0 +3.5 +9.7 +1.2 +2.5 +0.5 +0.1 +1.3 +0.9 -2.0 +0.2

Non-attention-based

ϕKNN 66% 20.5 13.4 22.1 22.0 24.6 54.2 29.9 60.0 70.2 57.0 60.5 77.7 51.9 61.9 50.7 ϕuniform 66% 38.3 32.7 38.8 38.8 42.7 58.3 37.6 61.9 75.8 58.0 65.8 85.9 60.2 65.0 52.2

Ensemble

###### ϕ-R + ϕuniform (Ours) 61% 46.3 41.6 47.3 46.0 50.1 61.3 46.8 62.0 77.7 58.7 66.8 86.9 61.6 65.4 53.3

- Table A1. Evaluating criteria using DINOv2 + SigLIP visual encoder. For each task, we bold the best result and underline the second-best result. Using K = 3 for all setups.

Localization Open-Ended VQA Challenge Sets

Method

FLOPSRed

GPUHours

Avg

OCID-Ref

RefCOCOg

RefCOCO+

RefCOCO

Avg

TextVQA

GQA

VQAv2

VizWiz

Avg

POPE

TallyQA

VSR

AI2D

Baseline 0% 20.3 53.2 40.7 56.3 55.0 60.9 64.1 54.9 63.3 78.9 59.3 66.1 87.4 59.3 63.3 54.3

FastV 68% 15.1 5.9 5.7 5.1 6.1 6.7 54.8 31.8 58.4 72.7 56.3 64.0 83.2 57.1 63.3 52.4 PyramidDrop 65% 15.7 28.9 24.0 29.2 29.7 32.9 60.8 47.1 61.2 76.9 57.9 65.3 86.6 58.2 63.4 53.1

FEATHER 64% 15.7 39.3 33.1 40.1 39.7 44.1 61.9 51.4 61.8 77.9 56.5 66.1 87.7 59.1 63.4 54.2

FastV 45% 16.8 29.1 17.5 29.5 33.1 36.1 61.0 45.8 62.3 77.4 58.4 65.7 86.8 59.2 63.3 53.5 PyramidDrop 46% 16.8 46.6 37.4 48.3 47.8 53.0 63.7 53.8 63.1 78.7 59.1 66.2 87.5 59.4 63.5 54.3

FEATHER 48% 16.5 49.7 39.3 52.1 50.9 56.7 63.9 54.6 63.2 78.8 59.0 66.3 87.7 59.2 64.0 54.6

- Table A2. Comparing FEATHER performance against FastV and PyramidDrop. The best results are bolded (excluding the baseline method).

#### A3. Comparison Against FasterVLM and VisionZip

We present FasterVLM and VisionZip performance in Table A3. We find that these approaches, while performing comparably to our approach on some benchmarks, perform vastly worse on localization benchmarks. We expect this is because positional information is not maintained in these methods, as image tokens are filtered without altering the positional embeddings. We verify the importance of positional embeddings in §A4. Note that since our setup uses the SigLIP encoder, for FasterVLM (which relies on [CLS] attention), we use the proposed solution in VisionZip of averaging attention each token receives from all others in the sequence.

#### A4. Token Shuffling Ablation

To assess the impact of positional embeddings on model performance, we shuffle positional embeddings for the image tokens and evaluate both the original VLM and our FEATHER approach. As shown in Table A3, the localization performance of both methods drops drastically for localization tasks, substantially for TextVQA, and relatively little for other benchmarks. This result supports our key insight that many vision-language benchmarks inadequately capture the shortcomings of efficiency methods due to their limited ability to assess fine-grained visual capabilities, particularly for visual grounding.

#### A5. Token Pruning Visualizations

In this supplemental material section, we provide a qualitative analysis comparing the pruning effectiveness of various criteria as well as the final approaches of FEATHER, FastV, and PyramidDrop. Namely, we visualize the ability of approaches to retain important tokens, particularly for localization. In Figure A2 and Figure A3, we visualize pruning from the various criteria assessed in the main text when pruning is done after layers three and eight, respectively. In Figure A4, we visualize pruning from the final approaches of FEATHER, FastV, and PyramidDrop.

##### A5.1. Comparing pruning criteria

We first visualize the retained tokens of various criteria when pruning is applied after layer three (see Figure A2) and layer eight (see Figure A3). We see that these visualizations support our quantitative results from the main paper. Specifically, (1) ϕ-R removes the criteria tendency of selecting bottom image tokens, resulting in an improved selection of maintained tokens; (2) the attention-based criteria improve when pruning after a later layer; and (3) adding uniform sampling to the attention-based pruning criteria with ϕ-R + ϕuniform improves token selection.

Localization Open-Ended VQA Challenge Sets

FLOPSRed

RefCOCO+

RefCOCOg

RefCOCO

OCID-Ref

TextVQA

TallyQA

VQAv2

VizWiz

POPE

AI2D

GQA

VSR

Avg

Avg

Avg

Method

Baseline 0% 53.2 40.7 56.3 55.0 60.9 64.1 54.9 63.3 78.9 59.3 66.1 87.4 59.3 63.3 54.3 Baseline (pos shuffled) 0% 8.0 9.0 7.8 7.1 8.0 59.2 44.1 60.3 75.8 56.8 63.3 86.6 55.3 59.2 51.9

FasterVLM 65% 5.7 8.0 5.9 4.2 4.7 60.9 50.9 59.9 76.5 56.4 66.6 85.2 62.6 63.7 54.7

VisionZip 65% 8.5 7.3 9.0 8.1 9.5 61.1 50.8 60.2 76.7 56.7 66.5 85.3 62.9 63.7 54.3 FEATHER 64% 39.3 33.1 40.1 39.7 44.1 61.9 51.4 61.8 77.9 56.5 66.1 87.7 59.1 63.4 54.2

FEATHER (pos shuffled) 64% 5.3 5.3 4.8 5.2 5.8 57.8 41.7 58.9 75.0 55.5 63.2 86.0 55.7 58.8 52.5

- Table A3. Comparison against FasterVLM and VisionZip and positional embeddings ablation (where image token positions are shuffled). The best results are bolded.

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

Reference expression: a bowl of blueberries

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

Reference expression: elephant on the left behind tree

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

Reference expression: right giraffe

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

Reference expression: last plane

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

- Figure A2. Visualizing the ability of various pruning criteria to maintain visual tokens relevant to the reference expression when applied after layer three. We observe that ϕ-R resolves ϕoriginal’s tendency of selecting bottom image tokens and that uniform sampling is a robust approach that improves the token selection effectiveness of ϕ-R with ϕ-R + ϕuniform. See the main text for criteria definitions.

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

Reference expression: a bowl of blueberries

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

Reference expression: elephant on the left behind tree

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

Reference expression: right giraffe

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

Reference expression: last plane

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

- Figure A3. Visualizing the ability of various pruning criteria to maintain visual tokens relevant to the reference expression when applied after layer eight. We observe that the attention-based criteria are more effective when pruning after this layer compared to after layer three. See the main text for criteria definitions.

##### A5.2. Comparing FEATHER to FastV and PyramidDrop

Additionally, we visualize the retained tokens for the FEATHER, FastV, and PyramidDrop approaches.

As shown in Figure A4, when comparing the remaining tokens used for prediction (after layer 16 for FEATHER, layer 24 for PyramidDrop, and layer three for FastV), we see that our approach retains substantially more tokens around and inside the reference expression bounding box.

Image FastV PyramidDrop FEATHER

k=8 k=16 k=24 k=8 k=16

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

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

Reference expression: player in white shirt and black shorts

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

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

Reference expression: a bowl of blueberries

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

Reference expression: elephant on the left behind tree

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

Reference expression: right giraffe

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

Reference expression: last plane

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

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

- Figure A4. Visualizing the ability of FEATHER, FastV, and PyramidDrop to retain visual tokens relevant to the reference expression. We observe that our approach retains a substantially higher portion of tokens relevant to the reference expression.

