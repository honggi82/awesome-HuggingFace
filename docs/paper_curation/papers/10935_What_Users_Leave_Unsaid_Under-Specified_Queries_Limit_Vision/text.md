# arXiv:2601.06165v2[cs.CV]13Apr2026

## What Users Leave Unsaid: Under-Specified Queries Limit Vision-Language Models

### Dasol Choi1,2* Guijin Son3* Hanwool Lee1* Minhyuk Kim4 Hyunwoo Ko3 Teabin Lim5 Eungyeol Ahn5 Jungwhan Kim6 Seunghyeok Hong7† Youngsook Song8†

1AIM Intelligence 2Yonsei University 3OneLineAI 4Korea University 5Doodlin Corp. 6NAVER Cloud 7Hankuk University of Foreign Studies 8Lablup Inc.

GitHub HuggingFace Leaderboard dasolchoi@yonsei.ac.kr, spthsrbwls123@yonsei.ac.kr

[Figure 1]

[Figure 2]

[Figure 3]

### Abstract

Current vision-language benchmarks predominantly feature well-structured questions with clear, explicit prompts. However, real user queries are often informal and underspecified. Users naturally leave much unsaid, relying on images to convey context. We introduce HAERAE-Vision, a benchmark of 653 realworld visual questions from Korean online communities (0.76% survival from 86K candidates), each paired with an explicit rewrite, yielding 1,306 query variants in total. Evaluating 45 VLMs, we find that even state-of-the-art models (GPT-5, Gemini 2.5 Pro) achieve under 50% on the original queries. Crucially, query explicitation alone yields 8 to 22 point improvements, with smaller models benefiting most. We further show that even with web search, underspecified queries underperform explicit queries without search, revealing that current retrieval cannot compensate for what users leave unsaid. Our findings demonstrate that a substantial portion of VLM difficulty stems from natural query under-specification instead of model capability, highlighting a critical gap between benchmark evaluation and real-world deployment.

### 1 Introduction

When users ask visual questions, they rarely provide complete, well-structured queries. Instead, they write informally, omit context, and rely on images to convey what they leave unsaid. A user might ask “How do I do this?” alongside an image, expecting the responder to identify the problem, infer the relevant domain, and provide a step-by-step solution. This natural tendency toward under-specification poses a fundamental challenge for vision-language models (VLMs) (Li et al., 2025), yet current benchmarks predominantly feature clean, explicit prompts failing to capture this phenomenon (Kim and Jung, 2025; Ju et al., 2024).

*Equal contribution. †Corresponding authors.

We introduce HAERAE-Vision, a benchmark constructed from authentic user queries in Korean online communities. Starting from 86,052 question-image pairs across nine platforms, we apply a six-stage filtering pipeline to yield 653 rigorously validated items (0.76% survival rate). The resulting questions are ambiguous, informal, and under-specified, mirroring the noisy nature of authentic multimodal interactions. To isolate the effect of query under-specification, we additionally construct HAERAE-Vision-Explicit, a parallel dataset where each question is systematically rewritten to state the missing information explicitly.

Our experiments reveal that query explicitation alone yields up to 22 point improvements across models, with smaller models benefiting most dramatically. Even state-of-the-art models achieve under 50% on original queries but surpass 55% with explicitation (GPT-5: 48.0%→57.6%, Gemini 2.5 Pro: 48.5%→56.7%). Furthermore, we demonstrate that even with web search enabled, under-specified queries still underperform explicit queries without search. This reveals that current retrieval systems cannot compensate for what users leave unsaid, as models must first understand user intent before search becomes effective.

These findings challenge a common assumption in VLM evaluation: that benchmark difficulty reflects model capability limitations. We show that a substantial portion of difficulty stems instead from the natural under-specification of user queries, highlighting a critical gap between benchmark evaluation and real-world deployment.

Our contributions are:

- • Real-world query benchmark: HAERAEVision, comprising 653 user-generated visual questions, filtered from 86K candidates (0.76% survival), spanning 13 domains.
- • Paired explicit rewrites: A parallel dataset

|Natural Objects (Animals/Plants/Insects)| |
|---|---|
|Question: 제주도 바다에서 본 생물 질문합니다. 사진에 있는 저 지렁이 같은 게 뭘까요? (I have a question about a marine organism I saw in the sea around<br><br>Jeju Island. What is that worm-like creature in the photo?)<br><br>Checklist:<br><br>1. 답변에 '덩굴뱀고둥'이라는 정확한 명칭이 포함되었나요? (Does the answer include the accurate name 'Dendropoma maxima<br><br>(덩굴뱀고둥)’?)<br><br>2. 답변에서 '지렁이'가 아닌 '달팽이'임을 명확히 구분했나요? (Does the answer clearly distinguish that it is a 'snail' rather than a<br><br><br>'worm'?)<br><br>[Figure 4]| |

|Daily Life|Question: 강아지가 이런 패턴을 뜯어서 새로 붙이려 하는데 이런|
|---|---|
|[Figure 5]<br><br>[Figure 6]<br><br>건 어디서 살 수 있을까요? (My dog tore off this wood-grain baseboard (skirting board). Where can I buy the same type to replace it?) Checklist:<br><br>1.제품 명칭을 '목재걸레받이'로 정확히 지칭했나요? (Does the answer correctly identify the product as 'wooden<br><br>skirting board’?)<br><br>2. ' 목재걸레받이'를 검색하여 온라인 구매를 권장했나요? (Does the answer recommend searching for and purchasing a<br><br><br>'wooden skirting board’?)| |

|Automotive/Transportation| |
|---|---|
|Question: 자동차 핸들위 동그라미 친 이 장치는 뭐하는용도인가요 (What is the purpose of this circled device above the car steering wheel?)<br><br>[Figure 7]<br><br>Checklist:<br><br>1. 해당 답변이 장치를 '운전자 감시 카메라(DMS)'로 정확히 지칭하고 있나요? (Does the answer correctly identify the device as 'Driver Monitoring System (DMS)’?)<br>2. 해당 답변이 운전자의 졸음운전·주의 산만을 모니터링하는 기능을 설명하고 있나요? (Does the answer explain the function of monitoring driver drowsiness and distraction?)<br><br><br>Question: 자동차 핸들위 동그라미 친 이 장치는 뭐하는용도인가요? (What is the purpose of this circled device above the car steering wheel?)| |

|Science|Question: 일반세균수 검사를 진행했는데 전부 이런 식으로 오염됐어요. 무슨 균에 오염된 걸까요 이것 때문에 산출도 못하고 있습니다|
|---|---|
|[Figure 8]<br><br>? . (I conducted a general bacteria count test, but everything got contaminated like this. What kind of contamination could this be?) Checklist:<br><br>1. 해당 답변은 배지 표면이 완전히 건조되지 않아 오염이 발생했음을 설명하고 있나요?<br><br>(Does the answer explain that contamination occurred because the medium surface was not fully dried?)<br><br>2. 해당 답변은 배지 건조 시간(1~2시간) 및 구체적 건조 방법을 제시하고 있나요?<br><br><br>(Does the answer suggest a drying time of 1–2 hours and provide specific drying methods for the medium?)| |

|Coding| |
|---|---|
|Question: 자동차 핸들위 동그라미 친 이 장치는 뭐하는용도인가요 (What is the purpose of this circled device above the car steering wheel?)Checklist:<br><br>1. 답변에 프로젝트 경로에 한글(또는 비-ASCII 문자) 포함 시 String 타입 정의 시 오류가 발생할 수 있다고 설명하고 있나요?<br><br>(Does the answer explain that an error may occur when defining a String type if the project path contains Korean (or non-ASCII) characters?)<br><br>2. 답변에 프로젝트 경로를 영어로 변경하거나 File → Invalidate Caches & Restart로 IDE 캐시를 초기화하라고 안내하고 있나요?<br><br><br>(Does the answer suggest changing the project path to English or using File → Invalidate<br><br>Caches & Restart to reset the IDE cache?)<br><br>Question: 왜 String타입 변수 정의할 때 빨간줄이 발생하는지 모르겠습니다. 인텔리제이로 새 프로젝트를 시작하면서 뭔가 달라진 걸까요..? (I don’t understand why a red underline appears when defining a String variable. Did something change when I started a new project in IntelliJ?)<br><br>[Figure 9]| |

|Math|Question: 해답 내용은 너무 간결하게 되어서 풀이과정이|
|---|---|
|어렵네요 각ㅂㅁㅈ. 각ㄹㅁㅈ 가 무조건 50각이라는 보장이<br><br>있는건지<br><br>(The solution is too concise, so the solving process is difficult to follow. Is there a guarantee that ∠ㅂㅁㅈ and ∠ㄹㅁㅈ are always 50°?)<br><br>Checklist:<br><br>1. 해당 답변은 접힌 도형에서 겹치는 각이 대칭이므로 같다는 원리를 설명하고 있나요?<br><br>(Does the answer explain the principle that overlapping angles<br><br>in a folded figure are symmetric and therefore equal?)<br><br>2. 해당 답변은 삼각형의 세 각의 합이 180°임을 이용해 각 ㉠을 80°로 구하는 과정을 포함하고 있나요?<br><br><br>(Does the answer use the fact that the sum of the three angles of a triangle is 180° to solve for ∠㉠ = 80°?)<br><br>[Figure 10]| |

- Figure 1: Representative examples from HAERAE-Vision across six of the 13 domains. Each example shows an underspecified Korean question with English translation, the corresponding image, and evaluation checklist criteria. Note the informal, context-dependent nature of the original queries.

- Stage 1: Data Collection. We collect (question, image, answer) triplets, prioritizing those with an accepted answer rewarded by the asker or with high online engagement (views, likes, comments), targeting questions the community finds valuable.
- Stage 2: Appropriateness Filtering. Each triplet is screened along three axes: (i) content safety (political/religious material, discrimination, adult content), (ii) objectivity (overly subjective or unverifiable prompts), and (iii) time-sensitiveness. GPT-4o is used for the automated filtering, flagging problematic items while allowing borderline cases to proceed to human validation. This removes 49.6% of raw data (see Appendix B.1).
- Stage 3: Difficulty Calibration. Following prior benchmarks (Zellers et al., 2019; Hendrycks et al., 2021), we remove questions that strong models solve trivially. Three models (GPT4o, Gemini-1.5-Flash, Claude-3.5) are evaluated against community-provided human answers using semantic-overlap scoring. Items with an average score above 0.6 are removed, eliminating 87.6% of the remaining items.

of clarified queries enabling controlled measurement of under-specification effects.

• Quantifying under-specification: Empirical evidence that explicitation yields up to 22% improvements, with smaller models benefiting most. This demonstrates that query ambiguity accounts for substantial VLM difficulty.

### 2 HAERAE-Vision Benchmark

We present HAERAE-Vision, a benchmark constructed from authentic user queries, designed to capture the under-specified, informal nature of realworld visual questions. Our six-stage pipeline transforms large-scale, noisy community data into highquality evaluation problems while preserving the natural characteristics of user queries.

#### 2.1 Dataset Construction Pipeline

Starting from 86,052 raw question-image pairs from nine Korean platforms spanning general Q&A, gaming, science, and coding forums (see

- Appendix A.1 for detailed platform descriptions), we obtain 653 high-quality problems (0.76% survival rate). Figure 2 illustrates the filtering process.

Original 653

4. Image Dependency 1,040 (1.2%)

6. Human Validation 653

1. Raw Data

2. Appropriateness 43,381 (50.4%)

3. Difficulty 5,360 (6.2%)

86,052 (100%)

Explicitated 653

(0.76%)

Removed 42,671

Removed 387

Removed 38,021

Removed 4,320

- Figure 2: Filtering pipeline showing data reduction at each stage. Numbers indicate pipeline stages described in Section 2.1. The 0.76% survival rate reflects rigorous quality control. Each validated question is paired with an explicitated rewrite, yielding 1,306 query variants.

- Stage 4: Image Dependency Verification. To confirm that each question requires visual reasoning, we generate two responses per item using Gemini 2.0 Flash: one with the image and one without. Both responses are evaluated against the human reference, and items where the quality gap is below 1 point (on a 0-10 scale) are discarded as imageindependent (see Appendix B.2).
- Stage 5: Checklist Generation. Each answer is converted into a structured checklist with 1 to 5 criteria using o4-mini. The model is instructed to define the minimal necessary elements for a response to be deemed correct, focusing on correctness, explanation quality, and reasoning steps rather than exhaustive coverage. This design enables partialcredit scoring and ensures reproducible, automated evaluation across models (see Appendix B.3).
- Stage 6: Human Validation. Seven native Korean annotators conduct three-phase validation: (1) filtering based on image appropriateness, question clarity, and checklist validity, removing any item flagged by at least one annotator; (2) refinement of questions and LLM-generated checklists, where annotators rewrite unclear criteria and remove items not grounded in the original question–image pair;

(3) final audit for category consolidation and consistency. This removes 37.2% of remaining items, yielding 653 problems (see Appendix C.1).

#### 2.2 Dataset Statistics

Our final benchmark contains 653 problems with an average of 3.3 checklist items and 1.3 images per question. Table 1 presents the distribution across 13 categories, where Natural Objects and Gaming are the most represented. The survival rate per platform varies significantly (0.2% to 14.4%), showing distinct community characteristics (see

- Appendix A.2 for detailed breakdown).

Metric Mean Range

Q length (char) 94.4 6–2,030 Images per Q 1.3 1–6 Checklist items 3.3 1–5

###### Category # Items %

Gaming 91 13.9 Entertainment/Arts 50 7.7

Natural Objects 92 14.1 Science 81 12.4 Mathematics 26 4.0

IT/Computer 75 11.5 Coding/Development 45 6.9 Machinery 22 3.4

Daily Life 51 7.8 Business/Economics 37 5.7 Transportation 35 5.4 Shopping/Consumer 27 4.1 Health/Medical 21 3.2

Total 653 100.0

Table 1: Overview of HAERAE-Vision. Statistics of question length, number of images, and checklist items, highlighting the diversity and multimodal nature of HAERAE-Vision.

#### 2.3 HAERAE-Vision-Explicit

To isolate the effect of query under-specification, we construct a parallel dataset where each question is rewritten to explicitly state the missing information while preserving the original intent. Figure 3 illustrates the transformation from under-specified to explicit queries across different domains.

We use GPT-5.1 with web search to rewrite each question following strict guidelines (Appendix B.4): (1) preserve the original intent and scope without broadening or narrowing, (2) make implicit context explicit by specifying domains, entities, and concrete references, (3) replace vague references such as “this,” “that,” or “here,” (4) incorporate visual information from the image into the question, and (5) use web search only to verify proper nouns (e.g., game titles, product names) implied by the original question. Each rewritten question then undergoes human validation. Three annotators reviewed all 653 explicitated questions against their corresponding images,

###### Image Original Explicitated

[Figure 11]

[Figure 12]

어ᄀ는 ᄋ떻ᄀ ᅢᄈ는걸ᄁᄋ? ᄌ ᅩᄀᄅ를 ᅢᄈᄀᄂᄂ ᄌ렇 ᄀ ᄂᄋ는ᄃ ᄌᄇ분ᄋ ᄋ떻ᄀ ᄈᄋᄒᄂᄋ? (How do I remove this? After removing the hook, this part remains—how do I take it out?)

천장ᄋ 설ᄎ된 ᅴᄒᆫ색 ᅩ기ᄅ형 ᅢᄒᆼᄀ를 ᅦᄌᄀ한 ᅮᄒ ᅡᄂᄋ 그ᄉ ᅮᄇ ᄉᄑ을 완전ᄒ 분ᄅᄒᄅ면 ᄋ떻ᄀ ᄒᄋ ᄒᄂᄋ? (How do I completely remove the metal fitting left after detaching the white ceiling hook hanger?)

ᄀ임 ’ᄋᆫᄉᆫ’ᄋᄉ ᄑᄏᄐᆫ NPCᄀ ᄋ뢰ᄒ는 임ᄆ 중 ᄃ장ᄒ는 ᄋ ᅥᄋᄅᆫ ᄋ ᅦᄉ ᄆᄅ 외ᄋ ᅮᄎᄀᄅ 찾ᄋᄋ ᅡᄒ는 ᄋᄋ ᅥᄃ 있ᄂᄋ? (In Genshin Impact, are there additional dragons to find beyond the three baby dragons in Parkatin’s quest?)

[Figure 13]

ᄋᄅᆫᄋ ᄌ 3ᄆᄅ ᄆᄀ ᄃ 있ᄂᄋ? (Are there more besides those 3 baby dragons?)

한ᄀ ᄆᄅᄆ 경ᄀ선 ᅥᆹᄋᄋ는 법. ᄃᄀ리ᄆ ᅵᄎᆫ ᅮᄇ분 없ᄋ ᄉ 있ᄂᄋ? (How to remove header border in Hangul. Can I remove the circled part?)

한ᄀ 문ᄉᄋᄉ ᄆᄅᄆ ᄀᄋ 상단ᄋ ᅭᄑᄉ되는 ᄋ백 경ᄀ선을 ᄌᄀᄒᄅ면 ᄋ떻ᄀ ᄒᄋ ᄒᄂᄋ? (How do I remove the margin border line shown at the top of the header area in Hangul word processor?)

|[Figure 14]|
|---|

- Figure 3: Examples of query explicitation across three domains (Daily Life, Gaming, IT/Software). Original queries contain vague references that depend on images. Explicitated versions include background information to clarify the user request.

verifying factual accuracy, correcting hallucinated details through additional search, and adjusting specificity by removing overly specific terms or adding missing context where necessary. This process yields 653 explicitated questions paired with the original under-specified versions.

#### 2.4 Korean Cultural Grounding

We consider an item culturally grounded if it requires knowledge of Korean institutions, services, policies, local brands or products, or Koreanlanguage UI and text conventions; items solvable through globally shared knowledge are marked non-cultural. Under this criterion, 23.7% of items require distinctively Korean cultural knowledge, including local interfaces (Seoul Metro signage, Naver SmartPlace), region-specific objects (winter road sandbags), or Korean media (drama actors, traditional calligraphy). These items are rarely represented in English-centric training corpora. Figure 4 shows representative examples.

3 Evaluation Framework

#### 3.1 Checklist-based Assessment

To mitigate the subjectivity of single-label scoring and the noise inherent in raw web text, our methodology centers on detailed checklists that decompose complex answers into specific criteria. Supported by recent findings that instance-specific rubrics align better with human judgments (Kim

- et al., 2024), each problem includes 1–5 evaluation points assessing different reasoning aspects. This checklist approach provides several advantages over traditional methods: (1) Fine-grained assessment of partial understanding, (2) Reduced subjectivity through explicit criteria, (3) Diagnostic

capability for pinpointing model weaknesses, and

(4) Scalability for automated evaluation.

#### 3.2 LLM Judge Protocol

GPT-5-Mini is instructed to act as the primary judge, following a structured prompt that enforces consistent scoring across all problems (Appendix D). Each checklist item is scored on a threelevel scale: met (1.0), partially met (0.5), or not met (0.0), based solely on explicit evidence found in the model’s response. Each score is accompanied by supporting evidence and justification, where the evidence is a single line directly extracted from the response and the justification is a short rationale clarifying the decision. The model outputs a structured report containing evidence blocks and fractional totals (e.g., 3.5/5 when one item is partially and three are fully satisfied out of five). The overall score is computed as the average of instance-level means, where each instance has mi checklist items with item scores rij ∈ {0,0.5,1}:

 ,

  1

mi

N

1 N

Sfinal =

rij

mi

j=1

i=1

ensuring comparability across problems with differing checklist lengths.

4 Experimental Setup

#### 4.1 Model Selection

We evaluate 45 VLMs covering a broad range of families and scale. Proprietary models. This group includes OpenAI’s GPT-5 series (GPT-5, GPT-5-Mini, GPT-5-Nano) (OpenAI, 2025a), Google’s Gemini (2.5-Pro, 2.5-Flash, 2.5-FlashLite) (Google DeepMind, 2025), and proprietary

[Figure 15]

- Figure 4: Examples highlighting the cultural specificity of HAERAE-Vision: (a) Seoul subway interface, (b) traditional painting with calligraphy, (c) Korean drama scene requiring celebrity recognition, (d) TV channel settings, (e) historical family registry. Such culturally grounded items require knowledge rarely represented in English-centric datasets.

systems such as Perplexity-Sonar-Pro (Perplexity AI, 2025), xAI-Grok-4 (xAI, 2025), Mistral (Medium-3.1, Small-24B) and Pixtral (Large, 12B) (Mistral AI, 2024; Agrawal et al., 2024). Open-source models. We evaluate Gemma-3 (27B, 12B, 4B) (Gemma Team, Google DeepMind, 2025), Qwen2.5-VL (72B, 7B, 3B) (Bai et al., 2025), Qwen3-VL (235B-A22B, 32B, 30B-A3B, 8B, 4B, 2B; each in Instruct and Thinking variants) (Yang

- et al., 2025), Skywork-R1V3-38B (Shen et al., 2025), InternVL3.5 (38B–1B) (Wang et al., 2025), and AIDC-AI-Ovis2 (34B–1B) (Lu et al., 2025). Korean models. Finally, we include Koreanspecific models, including VARCO-VISION-2.0 (14B, 1.7B) (NCSOFT AI Center, 2025) and HyperCLOVA-3B (Yoo et al., 2024).

- 4.2 Implementation Details

We used temperature=0.6 (1.0 for GPT-5 due to provider constraints), top_p=0.95, and max_tokens=4096 across all models. Each instance was evaluated three times and averaged.

5 Results

- 5.1 Overall Performance

Table 2 summarizes the performance of 18 VLMs across four categories (full results are provided in Appendix E.1). Even the best-performing models—Gemini 2.5 Pro (48.5%) and GPT-5 (48.0%)—fall short of 50% accuracy, highlighting that authentic, culturally grounded multimodal queries remain far from solved. Proprietary systems consistently outperform open-weight counterparts, with the strongest open-weight models (Skywork-R1V3-38B: 27.8%, Qwen2.5-VL72B: 20.6%) reaching roughly half the accuracy of top proprietary models. Neither searchaugmented models (Perplexity Sonar-Pro: 44.3%) nor reasoning-specialized models (Skywork-R1V3) achieve notable gains, suggesting that solving

would require capabilities beyond current retrievalaugmented or reasoning-optimized paradigms.

Korean-specialized models also struggled to achieve competitive results (VARCO-VISION 2.0 14B: 15.6%, HyperCLOVA X-SEED-3B: 12.7%), indicating that dedicated local models have yet to demonstrate clear advantages on this benchmark. See Appendix E for a domain-level analysis.

#### 5.2 Effect of Query Explicitation

Figure 5 shows the effect of query explicitation on model performance. Across all six models, explicitation yields substantial improvements of 7.8 to 21.7 points. Smaller models benefit most from explicitation: GPT-5-Nano improves by 21.7 points (21.2 → 43.0), more than doubling its performance, while larger models like GPT-5 and Gemini 2.5 Pro show gains of 9.6 and 8.1 points respectively. This pattern suggests that under-specified queries disproportionately disadvantage smaller models, which may lack the capacity to infer implicit context from images alone. Even with explicitation, the bestperforming model (GPT-5) achieves only 57.6%, indicating that query under-specification accounts for a substantial portion, but not all, of the difficulty in HAERAE-Vision. Our error analysis (Section 6) reveals that the remaining challenges stem primarily from cultural knowledge gaps.

#### 5.3 Effect of Web Search

To isolate the contributions of query explicitation and retrieval augmentation, we evaluated GPT-5 and GPT-5-Mini across all four conditions: original and explicitated queries, each with and without web search. We use the official OpenAI search API (OpenAI, 2025b).

As shown in Table 3, web search yields moderate improvements for original queries (GPT-5: +7.57; GPT-5-Mini: +5.87), but these gains are smaller than those obtained through explicitation alone (+9.56 and +7.83, respectively). Notably,

Model Entertainment Scientific Technical Daily Life Overall

Proprietary Models Gemini 2.5 Pro 40.520.61 51.440.40 53.890.79 52.640.93 48.540.11 GPT-5 33.070.87 48.140.96 55.710.84 55.980.75 48.010.19 GPT-5 Mini 27.380.81 50.620.93 51.880.74 51.311.32 45.210.70 Perplexity Sonar-Pro 32.840.76 47.980.59 47.171.23 49.640.64 44.280.48 Gemini 2.5 Flash 29.311.09 45.040.98 44.050.53 48.721.38 41.050.79 Grok-4 26.880.67 31.030.64 44.180.80 39.670.55 36.080.30 Gemini 2.5 Flash-Lite 18.390.59 38.171.47 32.740.84 35.470.92 30.290.24 GPT-5 Nano 11.640.53 20.101.24 27.151.36 29.680.54 21.220.26

Open Source Models

Skywork-R1V3-38B 15.030.73 35.310.88 30.220.49 33.750.72 27.760.34 Mistral Medium 3.1 13.740.80 30.770.86 28.870.67 28.781.01 24.860.56 Gemma-3 27B 11.590.58 25.800.61 22.281.04 30.850.61 22.530.16 Qwen2.5-VL-72B 10.890.66 26.711.49 21.600.53 25.610.52 20.580.46 Pixtral Large 11.430.82 21.790.50 21.770.38 25.650.91 20.100.24 InternVL3.5-38B 8.810.46 23.250.61 17.920.73 23.360.78 18.010.22 Ovis2-34B 9.520.47 21.880.55 21.000.51 24.820.58 18.500.02 Mistral Small 24B 6.460.29 10.180.45 13.300.66 16.200.66 11.200.01

Korean-specialized Models

VARCO-VISION 2.0 (14B) 7.870.80 16.560.65 16.880.57 22.130.88 15.550.29 HyperCLOVA X-SEED-3B 6.250.25 14.870.51 11.990.50 17.930.73 12.660.10

Table 2: Performance of 18 models averaged by category. For model families with multiple sizes, only the largest variant is shown. Full results across all model sizes and detailed 13-category breakdowns are in Appendix 15. All scores are reported as meanSE, where SE is the standard error over 3 independent runs (n=3). The highest-scoring model is highlighted in bold.

70

Original Explicitated

+9.6 +8.1

60

+7.8

| |
|---|

+10.3

50

Accuracy(%)

+21.7 +10.9

40

30

20

10

0

GPT-5 Nano Gemini 2.5 Flash-Lite

Gemini 2.5 Flash

GPT-5 Gemini 2.5 Pro

GPT-5 Mini

Model

- Figure 5: Effect of query explicitation on model performance. Models are sorted by improvement magnitude. Smaller models benefit most from explicitation, with GPT-5 Nano showing +21.7 points improvement. All results averaged over 3 runs.

original queries augmented with search still underperform explicit queries without search (GPT-5:

- 55.58 vs. 57.57; GPT-5-Mini: 51.08 vs. 53.04). This indicates that retrieval cannot compensate for under-specified queries; models must first infer user intent for search to be effective. We observe a recurring failure mode in which models rely on textual cues during search while failing to ground visual features, suggesting that current web search integration operates at a largely surface level and is not deeply leveraged by GPT-5. The highest performance is achieved when explicitation and search are combined (GPT-5: 59.72; GPT-5-Mini:
- 56.69), demonstrating additive benefits. However, the marginal improvement from adding search to explicit queries (+2.15 and +3.65) is smaller than when added to original queries, implying that ex-

Model Orig Orig+S Expl Expl+S

GPT-5 48.01 55.58 57.57 59.72 GPT-5-Mini 45.21 51.08 53.04 56.69

Δ from Original (no search)

GPT-5 – +7.57 +9.56 +11.71 GPT-5-Mini – +5.87 +7.83 +11.48

Table 3: Effect of web search and query explicitation. Scores reported as mean over 3 runs. Original+Search still underperforms Explicit alone, indicating retrieval cannot compensate for under-specification.

plicitation already supplies much of the contextual information that search would otherwise retrieve.

#### 5.4 Cross-Lingual Validation

To test whether the explicitation effect generalizes beyond Korean, we conduct a pilot study in English. We collect approximately 3,000 image-containing Q&A pairs from 12 Stack Exchange communities spanning 9 of our 13 categories and apply the same six-stage pipeline (Section 2.1). After filtering, 627 samples survive; we randomly select 100 samples stratified by category for evaluation (see Appendix A.3 for full construction details and perdomain results).

As shown in Table 4, all four models show consistent explicitation gains (+3.2 to +6.6 points), confirming that the effect of query underspecification is not limited to Korean. However, English deltas are consistently smaller than their Korean counterparts (+7.8 to +21.7). Notably, GPT-5-Nano scores 44.4% on English original queries, more

###### Model Original Explicit ∆

GPT-5 60.8 65.6 +4.8 GPT-5-Mini 53.3 59.9 +6.6 Gemini 2.5 Pro 51.3 57.3 +6.0 GPT-5-Nano 44.4 47.6 +3.2

Table 4: English pilot: effect of query explicitation. All four models show consistent gains, confirming that underspecification effects are cross-lingual. Deltas are smaller than in Korean (+3.2–6.6 vs. +7.8–21.7).

than double its Korean score (21.2%), suggesting that smaller models can better compensate for underspecification in high-resource languages.

Root-cause analysis on remaining errors after explicitation reveals why the gap is smaller in English: cultural knowledge accounts for only 2.7% of English errors versus 19.0% in Korean, with general reasoning comprising 96.7% of English failures. This confirms that the larger Korean explicitation gap is driven by the interaction between surfacelevel underspecification and culturally grounded assumptions that are underrepresented in Englishcentric training corpora. Once explicitation resolves surface ambiguity, cultural knowledge remains as a persistent source of difficulty in Korean but not in English.

### 6 Additional Analysis on Explicitation

To understand why explicitation improves performance, we analyzed error patterns across original and explicitated conditions. We collected 3,164 (original) and 2,834 (explicitated) error cases where models scored below 1.0, spanning six models (GPT-5, GPT-5-Mini, GPT-5-Nano, Gemini 2.5 Pro, Gemini 2.5 Flash, Gemini 2.5 Flash-Lite). Each error was annotated by an LLM judge (Claude 3.5 Sonnet) along two dimensions: (1) failure category—how the error manifests (Table 5); and (2) root cause—why the error occurs. Failure categories are multi-label (1–3 per error), while root causes are single-label; the two dimensions are orthogonal, so their percentages are not directly comparable. The full annotation prompt and category definitions are provided in Appendix F.

#### 6.1 What Explicitation Fixes

Table 6 shows the key shifts. The most striking change is the reduction in lack of explicitness failures, which drop from 84.3% to 69.7% (-14.6pp), directly confirming that explicitation addresses surface-level ambiguity. Smaller models show the largest reductions in error cases after explic-

Failure Category Description

Lack of explicitness Missing checklist-required facts Procedural reasoning Failed multi-step execution Object recognition Misidentified visual entities Cultural mismatch Misunderstood Korean conventions Visual-text grounding Wrong image region referenced Spatial reasoning Incorrect spatial relations

###### Root Cause

General reasoning Logic/inference failure Cultural knowledge Missing Korean-specific knowledge Language Korean language misunderstanding

- Table 5: Error annotation taxonomy (abbreviated).

Failure Category Orig Expl Δ

Lack of explicitness 84.3% 69.7% -14.6 Procedural reasoning 66.6% 64.3% -2.3 Object recognition 20.6% 18.5% -2.1 Cultural concept mismatch 13.1% 22.5% +9.4 Visual-text grounding 5.2% 16.6% +11.4

- Table 6: Failure category shifts from original to explicitated queries.

itation (GPT-5-Nano: -83 cases, +12.7pp perfect rate) compared to larger models (GPT-5-Mini: -40 cases, +6.1pp), confirming that under-specification disproportionately impacts smaller models.

Category-level analysis (Figure 6) reveals that explicitation yields the largest gains in Mathematics, Science, Coding, and Shopping—categories where failures primarily stemmed from underspecified problem descriptions. In contrast, Natural Objects and Entertainment remain challenging even after clarification (all-models-pass rate: 0% in both conditions), with failures shifting toward visual-text grounding and cultural knowledge gaps.

Notably, visual-text grounding (VTG) errors increase from 5.2% to 16.6% after explicitation. However, tracking individual error cases reveals that this reflects an unmasking effect rather than a trade-off: 87% of VTG errors in the explicitated condition were already errors under original queries but classified under other categories (primarily lack of explicitness). Explicitation forces models to engage with specific visual regions, exposing latent grounding failures previously obscured by surface-level ambiguity (see Appendix E.5 for detailed analysis).

#### 6.2 Why Retrieval Alone Is Insufficient

Earlier, our results have shown that original queries with search (55.6) underperform explicit queries without search (57.6). This reveals a fundamental limitation: retrieval cannot compensate for query

| |+23.1<br><br>+15.6<br><br>+14.3<br><br>+13.3<br><br>+11.1<br><br>+8.1<br><br>+6.2<br><br>+4.5 +4.4<br><br>+3.9<br><br>+0.0 +0.0 +0.0| | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

Mathematics

Coding Health/Medical

IT/Computer Shopping Business Science Machinery Gaming

Daily Life Transportation Entertainment

Natural Objects

0 5 10 15 20 25

All-models-pass rate (pp)

- Figure 6: Category-level explicitation effects. Categories like Mathematics and Coding show large gains, while Entertainment and Natural Objects remain difficult even after clarification, with failures shifting toward cultural knowledge and visual grounding.

under-specification. Under-specified queries like “ᄋᄀ ᅥᄋ떻ᄀ 효ᄋ?” (How do I do this?) contain no searchable keywords. Since the critical context is embedded solely within the visual modality, current text-based search engines fail to bridge the modality gap without explicit textual grounding. Even when models attempt searches, they lack the specific terms (product names, game titles, error codes) needed to retrieve useful results. In contrast, explicitated queries contain concrete references (e.g., “천장ᄋ 설ᄎ된 ᅴᄒᆫ색 ᅩᄀᄅ형 ᅢᄒᆼᄀ” (white ring-shaped hanger installed on the ceiling)) that enable targeted retrieval. The best performance is achieved when both are combined (59.7), but the key finding is that search on under-specified queries cannot match explicitation alone; models must first understand what to search for.

#### 6.3 Cultural Knowledge Gaps

After explicitation, what errors remain? Analyzing root causes reveals a shift toward cultural knowledge gaps (Table 7). The increase in cultural knowledge attribution (+6.4pp) suggests that once query ambiguity is resolved, the dominant remaining challenge is Korea-specific knowledge. For example, when shown orange bags along a rural road, models identified them as “road safety markers” or “wasp traps,” missing that these are winter snow preparation sandbags, something all native Korean drivers would have known. Similarly, all SOTA models misidentified a Korean folder phone (SKY IM-100) as global brands like Sony or Nokia. Finally, the negligible language error rate (<1.5%) confirms that Korean proficiency is no longer a hurdle for global models, but cultural content is.

Root Cause Orig Expl Δ

General reasoning 86.6% 79.8% -6.9 Cultural knowledge 12.7% 19.0% +6.4 Language 0.7% 1.2% +0.5

- Table 7: Root cause distribution. After explicitation, cultural knowledge becomes more prominent as surface-level ambiguity is resolved.

GPT-5-mini GPT-5 Gem-2.5-Pro Gem-2.5-Flash

GPT-5-mini – 0.87 0.90 0.90 GPT-5 0.87 – 0.90 0.86 Gem-2.5-Pro 0.90 0.90 – 0.89 Gem-2.5-Flash 0.90 0.86 0.89 –

Krippendorff’s α = 0.867

- Table 8: Pairwise Pearson correlations among four LLM judges. Spearman correlations range 0.87–0.90. Krippendorff’s α = 0.867 indicates substantial agreement.

### 7 Reliability of LLM-as-a-Judge

It is widely known that LLM-Judges may be prone to biases (Son et al., 2024). Accordingly, to ensure the credibility of our evaluation, we assess the interjudge agreement among four LLM judges (GPT-5, GPT-5-mini, Gemini-2.5-Pro, Gemini-2.5-Flash). A stratified random sample of 250 model responses (50 per 0.2-score interval) was re-evaluated under identical protocols. Table 8 shows consistently high correlations, with Pearson ranging from 0.863 to 0.903 and Spearman from 0.866 to 0.901. Krippendorff’s α = 0.867 exceeds the conventional 0.80 threshold, indicating substantial agreement across models with different architectures.

Furthermore, to assess alignment with human judgments, the same 250-sample set was evaluated by four independent human annotators, who rated the appropriateness of GPT-5-Mini judgments on a 5-point scale. Agreement was high (Pearson r = 0.820, Spearman ρ = 0.810, p < 0.001), demonstrating that our judge provides a stable and human-aligned evaluation signal. Detailed analyses of low-agreement cases suggest that most discrepancies stem from superficial keyword matching or excessive leniency (examples in Appendix C.2).

### 8 Related Work

Evaluating VLMs. As VLMs become more general-purposed, evaluation has shifted toward diagnostic suites that aim to separate recognition, OCR, and knowledge from higher-level reasoning and instruction following (Liu et al., 2024; Li et al., 2024; Yu et al., 2024). To better probe reasoning, several benchmarks target domain knowl-

edge grounded with visual inputs (Yue et al., 2024, 2025; Lu et al., 2023). This was rapidly followed by the Korean community, first by text benchmarks that measure Korean knowledge (Son et al., 2023, 2025; Hong et al., 2025), then by multimodal benchmarks: KRETA, KViscuit, and KOFFVQA (Hwang et al., 2025; Park et al., 2024; Kim and Jung, 2025). In addition, localized evaluation tools such as KMMB, KSEED, and KDTCBench have been released alongside Korean VLM development efforts (Ju et al., 2024). However, these benchmarks have already been saturated by older-generation models such as GPT-

- 4o (e.g., KRETA (Hwang et al., 2025): 84.6; KVISCUIT (Park et al., 2024): 89.5; K-MMB: 81.01; K-SEED: 76.98; K-DTCBench: 85.80 (Ju et al., 2024)), motivating the creation of a more challenging benchmark.

Query Underspecification. Underspecified or ambiguous queries are pervasive in conversational settings (Rahmani et al., 2023), forcing systems to choose between answering, hedging, or asking for missing constraints. Prior efforts to evaluate LLMs in ambiguity handling include AmbigQA (Min et al., 2020), and clarification-focused resources such as ClariQ (Aliannejadi et al., 2021) and the ConvAI3 shared task (Aliannejadi et al., 2020), which measure how effectively a system reduces uncertainty through clarification. More recently, QuestBench tests minimal question asking as information acquisition for underspecified reasoning (Li et al., 2025). In the multimodal setting, ClearVQA evaluates whether models can ask image grounded clarification questions to resolve ambiguous visual queries (Jian et al., 2025). Overall, however, multimodal resources for query underspecification remain scarce. To bridge this gap, we introduce HAERAE-Vision, which further targets a niche and underexplored setting by focusing on underspecification in Korean language interactions with culturally grounded content and assumptions.

### 9 Conclusion

We introduce HAERAE-Vision, a benchmark of 653 authentic Korean questions from real-life users, each paired with explicit rewrites. Our experiments show that query underspecification accounts for an 8–22 point drop in VLM performance. Retrieval-augmented prompting does not close this gap: search-augmented underspecified queries still underperform explicitated queries with-

out search. We further find that many remaining failures reflect missing cultural knowledge rather than surface-level ambiguity. An English pilot study confirms that the explicitation effect generalizes cross-lingually, with smaller deltas attributable to fewer cultural knowledge barriers. Together, these findings highlight challenges that sanitized, clean-query benchmarks fail to capture.

#### Limitations

Guided by a quality over quantity principle, our filtering procedure yields a 0.76% survival rate. This aggressive filtering may exclude some informative edge cases; however, it should be noted that our goal is not to provide a comprehensive evaluation of Korean knowledge. Rather, we aim to study how LLM behavior changes under different levels of information density in user prompts. Furthermore, our web search augmentation analysis is also limited in scope, as it evaluates only OpenAI’s web search, and results may differ with more advanced retrieval systems. However, based on our observations, the primary bottleneck appears to be less about the search API itself and more about the model’s ability to extract and formulate meaningful questions grounded in the image and accompanying text. Our error annotation relies on an LLM judge, which may introduce systematic biases despite the high inter-judge agreement we observe. Finally, while our English pilot (Section 5.4) confirms that the explicitation effect generalizes beyond Korean, it is limited to 100 samples from a single platform ecosystem; a broader multilingual investigation across diverse languages and cultural contexts remains for future work.

#### Ethics and Data Governance

This study received ethical approval from the Institutional Review Board of Hankuk University of Foreign Studies (HUFS-2510-015). All data were collected from publicly available Korean community platforms. We implemented a rigorous filtering process to exclude sensitive content, and all PII has been systematically removed. AI assistants (Claude and Gemini) were used for grammar editing and code debugging.

#### Acknowledgments

This work was supported by the Hankuk University of Foreign Studies Research Fund (2025) and the TIPS Program (No. RS-2024-00512659) funded by the Ministry of SMEs and Startups (MSS), Korea.

### References

Pravesh Agrawal, Szymon Antoniak, Emma Bou Hanna, Baptiste Bout, Devendra Chaplot, and 1 others. 2024. Pixtral 12b. arXiv.

Mohammad Aliannejadi, Julia Kiseleva, Aleksandr Chuklin, Jeff Dalton, and Mikhail Burtsev. 2020. Convai3: Generating clarifying questions for opendomain dialogue systems (clariq). arXiv preprint arXiv:2009.11352.

Mohammad Aliannejadi, Julia Kiseleva, Aleksandr Chuklin, Jeff Dalton, and Mikhail Burtsev. 2021. Building and evaluating open-domain dialogue corpora with clarifying questions. In EMNLP.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, and 8 others. 2025. Qwen2.5-vl technical report. Preprint, arXiv:2502.13923.

Gemma Team, Google DeepMind. 2025. Gemma 3 technical report. arXiv.

Google DeepMind. 2025. Gemini 2.5 pro: Model card. https://storage.googleapis.com/ model-cards/documents/gemini-2.5-pro.pdf.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. 2021. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874.

Seokhee Hong, Sunkyoung Kim, Guijin Son, Soyeon Kim, Yeonjung Hong, and Jinsik Lee. 2025. From kmmlu-redux to kmmlu-pro: A professional korean benchmark suite for llm evaluation. arXiv preprint

- arXiv:2507.08924.

Taebaek Hwang, Minseo Kim, Gisang Lee, Seonuk Kim, and Hyunjun Eun. 2025. Kreta: A benchmark for korean reading and reasoning in text-rich vqa attuned to diverse visual contexts. arXiv preprint

- arXiv:2508.19944.

Pu Jian, Donglei Yu, Wen Yang, Shuo Ren, and Jiajun Zhang. 2025. Teaching vision-language models to ask: Resolving ambiguity in visual questions. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3619–3638, Vienna, Austria. Association for Computational Linguistics.

Jeongho Ju, Daeyoung Kim, SunYoung Park, and Youngjune Kim. 2024. Varco-vision: Expanding frontiers in korean vision-language models. arXiv preprint arXiv:2411.19103.

Seungone Kim, Juyoung Suk, Ji Yong Cho, Shayne Longpre, Chaeeun Kim, Dongkeun Yoon, Guijin Son, Yejin Cho, Sheikh Shafayat, Jinheon Baek,

and 1 others. 2024. The biggen bench: A principled benchmark for fine-grained evaluation of language models with language models. arXiv preprint arXiv:2406.05761.

Yoonshik Kim and Jaeyoon Jung. 2025. Koffvqa: An objectively evaluated free-form vqa benchmark for large vision-language models in the korean language. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 575– 585.

Belinda Z Li, Been Kim, and Zi Wang. 2025. Questbench: Can llms ask the right question to acquire information in reasoning tasks? arXiv preprint arXiv:2503.22674.

Bohao Li, Yuying Ge, Yixiao Ge, Guangzhi Wang, Rui Wang, Ruimao Zhang, and Ying Shan. 2024. Seedbench: Benchmarking multimodal large language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13299–13308.

Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, and 1 others. 2024. Mmbench: Is your multi-modal model an all-around player? In European conference on computer vision, pages 216–233. Springer.

Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, KaiWei Chang, Michel Galley, and Jianfeng Gao. 2023. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. arXiv preprint arXiv:2310.02255.

Shiyin Lu, Yang Li, Yu Xia, Yuwei Hu, Shanshan Zhao, Yanqing Ma, Zhichao Wei, Yinglun Li, Lunhao Duan, Jianshan Zhao, Yuxuan Han, Haijun Li, Wanying Chen, Junke Tang, Chengkun Hou, Zhixing Du, Tianli Zhou, Wenjie Zhang, Huping Ding, and 23 others. 2025. Ovis2.5 technical report. Preprint, arXiv:2508.11737.

Sewon Min, Julian Michael, Hannaneh Hajishirzi, and Luke Zettlemoyer. 2020. Ambigqa: Answering ambiguous open-domain questions. arXiv preprint arXiv:2004.10645.

Mistral AI. 2024. Pixtral-large-instruct-2411: Model card. https://huggingface.co/mistralai/ Pixtral-Large-Instruct-2411.

NCSOFT AI Center. 2025. Varco-vision-2.0 technical report. arXiv.

- OpenAI. 2025a. Gpt-5 system card. https: //openai.com/index/gpt-5-system-card/. Updated PDF: https://cdn.openai.com/ gpt-5-system-card-aug7.pdf.
- OpenAI. 2025b. Web search — openai api reference.

ChaeHun Park, Yujin Baek, Jaeseok Kim, Yu-Jung Heo, Du-Seong Chang, and Jaegul Choo. 2024. Evaluating visual and cultural interpretation: The k-viscuit benchmark with human-vlm collaboration. arXiv preprint arXiv:2406.16469.

Perplexity AI. 2025. Sonar pro: Model overview. https://docs.perplexity.ai/ getting-started/models/models/sonar-pro.

Hossein A Rahmani, Xi Wang, Yue Feng, Qiang Zhang, Emine Yilmaz, and Aldo Lipani. 2023. A survey on asking clarification questions datasets in conversational systems. arXiv preprint arXiv:2305.15933.

W. Shen and 1 others. 2025. Skywork-r1v3 technical report. arXiv.

Guijin Son, Hyunwoo Ko, Hoyoung Lee, Yewon Kim, and Seunghyeok Hong. 2024. Llm-as-a-judge & reward model: What they can and cannot do. arXiv preprint arXiv:2409.11239.

Guijin Son, Hanwool Lee, Sungdong Kim, Seungone Kim, Niklas Muennighoff, Taekyoon Choi, Cheonbok Park, Kang Min Yoo, and Stella Biderman. 2025. Kmmlu: Measuring massive multitask language understanding in korean. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 4076–4104.

Guijin Son, Hanwool Lee, Suwan Kim, Huiseo Kim, Jaecheol Lee, Je Won Yeom, Jihyu Jung, Jung Woo Kim, and Songseong Kim. 2023. Hae-rae bench: Evaluation of korean knowledge in language models. arXiv preprint arXiv:2309.02706.

Weiyun Wang and 1 others. 2025. Internvl 3.5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv.

xAI. 2025. Grok 4: Model card. https://data.x.ai/ 2025-08-20-grok-4-model-card.pdf.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, and 1 others. 2025. Qwen3 technical report. arXiv preprint arXiv:2505.09388.

Kyung-Min Yoo and 1 others. 2024. Hyperclova x technical report. arXiv.

Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. 2024. Mm-vet: Evaluating large multimodal models for integrated capabilities. In International conference on machine learning. PMLR.

Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, and 1 others. 2024. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi.

In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9556– 9567.

Xiang Yue, Tianyu Zheng, Yuansheng Ni, Yubo Wang, Kai Zhang, Shengbang Tong, Yuxuan Sun, Botao Yu, Ge Zhang, Huan Sun, and 1 others. 2025. Mmmupro: A more robust multi-discipline multimodal understanding benchmark. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 15134– 15186.

Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. 2019. Hellaswag: Can a machine really finish your sentence? arXiv preprint arXiv:1905.07830.

### Appendices

- A Dataset Construction Details 13

- A.1 Detailed Platform Descriptions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13
- A.2 Platform-wise Filtering Statistics . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13
- A.3 English Pilot Study Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13

- B Pipeline Prompts 14

- B.1 Stage 2 (Safety, Objectivity, Temporal) . . . . . . . . . . . . . . . . . . . . . . . . . . . 14
- B.2 Stage 4 Prompt Excerpt (Image Dependency Rubric) . . . . . . . . . . . . . . . . . . . 15
- B.3 Stage 5 (Checklist Generation) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
- B.4 Query Explicitation Prompt . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16

- C Human Annotation 17

- C.1 Annotation Guidelines . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
- C.2 LLM Judge Failure Cases . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19

- D LLM-as-Judge Prompt 19
- E Additional Results & Analysis 22

- E.1 Full Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22
- E.2 Performance by Model Scale . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22
- E.3 Performance by Domain . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22
- E.4 Investigating Failure Modes . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22
- E.5 Visual-Text Grounding Error Analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . 23

- F Error Annotation Methodology 23

- F.1 Annotation Setup . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23
- F.2 Annotation Prompt . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24

### A Dataset Construction Details

#### A.1 Detailed Platform Descriptions

We collected data from nine Korean online platforms representing diverse user communities and domain expertise. Table 10 provides detailed information about each platform.

Platform Category Description Naver KnowledgeIn General Q&A Korea’s largest general Q&A platform covering everyday queries, aca-

demic subjects, and technical issues BRIC Science Community Specialized community for biological research and biotechnology with scientific discussions and professional knowledge sharing Ruliweb Gaming Community Major gaming community covering video games, hardware reviews, game mechanics, and technical gaming issues MonsterZym Fitness Community Fitness and bodybuilding community discussing workout routines, nutrition, supplements, and exercise techniques Quasarzone Hardware Community Hardware enthusiast community focused on computer components, electronics, PC building, and technology reviews i-Boss Business Platform Business and entrepreneurship platform for startup strategies, operations, marketing, and professional development Inflearn Coding Education Online learning platform with community features for programming questions and coding experiences Codeit Coding Education Coding education platform with forums for programming discussions and technical support Okky Developer Community Developer community platform for programming discussions, career advice, and technical problem-solving

Table 9: Korean online platforms used for data collection

These platforms were selected to ensure comprehensive coverage of different user demographics, expertise levels, and domain-specific knowledge, reflecting the diversity of real-world multimodal questions Korean users encounter online.

##### A.2 Platform-wise Filtering Statistics Table 10 provides a detailed breakdown of data collection and filtering across all platforms.

Platform Raw Data Appropri. Difficulty Image Dep. Human Val. Final Survival KnowledgeIn 31,484 10,495 1,404 648 441 441 1.4% BRIC 291 291 163 60 42 42 14.4% Ruliweb 305 240 54 42 32 32 10.5% Coding 27,896 8,369 837 198 135 135 0.5% MonsterZym 3,090 3,090 2,234 8 6 6 0.2% Quasarzone 2,986 896 90 22 15 15 0.5% i-Boss 20,000 20,000 578 62 42 42 0.2% Total 86,052 43,381 5,360 1,040 713 653 0.76%

Table 10: Detailed data collection and filtering statistics by platform (Stages 1–6). Coding platforms include Inflearn, Codeit, and Okky combined.

#### A.3 English Pilot Study Details

To validate the cross-lingual generalizability of our findings, we construct an English pilot dataset by applying the same six-stage pipeline described in Section 2.1.

Data Source. We collect 2,954 image-containing Q&A pairs from 12 Stack Exchange communities: Stack Overflow, Super User, Arqade, DIY Home Improvement, Biology, Gardening & Landscaping, Motor Vehicle Maintenance, Cooking, Bicycles, Chemistry, Board & Card Games, and Mathematics. All data are publicly available under CC BY-SA 4.0.

Filtering and Explicitation. All pipeline stages are applied without modification. Table 11 summarizes the filtering process. Image dependency verification is the most aggressive filter, removing over 60% of candidates where images served as supplementary illustration rather than essential context. Each surviving question is explicitated following the same protocol as the Korean dataset (Appendix B.4), with subsequent human verification. We randomly select 100 samples stratified by domain for the pilot evaluation.

Stage Remaining Removed

Raw collection 2,954 – Image validation 2,876 −78 Under-specification filter 2,457 −419 Difficulty calibration 2,326 −131 Image dependency verification 887 −1,439 Checklist + explicitation 887 – Quality filtering 627 −260 Stratified sampling 100 –

Table 11: English pilot filtering pipeline.

Dataset Statistics. The final 100 samples contain 168 images (avg 1.7 per question) and an average of 3.4 checklist items. Table 12 shows the domain distribution.

Domain # Items %

Natural Objects / Science 23 23.0 Transportation 18 18.0 Gaming / Entertainment 13 13.0 Daily Life / Machinery 12 12.0 Coding 11 11.0 IT / Computer 9 9.0 Daily Life 7 7.0 Science 4 4.0 Mathematics 3 3.0

Total 100 100.0 Table 12: Domain distribution of the English pilot dataset.

Per-Domain Results. Table 13 presents per-domain explicitation effects averaged across all four models. Consistent with the Korean results, Coding shows the largest gain (+14.4), while visually grounded domains such as Natural Objects show smaller improvements.

Domain n Orig Expl ∆

Coding 11 57.5 71.8 +14.4 Gaming / Ent. 13 40.0 49.0 +9.0 Daily Life 7 47.3 55.6 +8.3 Science 4 49.8 57.8 +8.0 Transportation 18 51.1 56.5 +5.4 Nat. Objects / Sci. 23 55.4 58.3 +2.9 IT / Computer 9 55.4 58.1 +2.7 Mathematics 3 49.4 50.6 +1.1 Daily Life / Mach. 12 60.1 56.8 −3.3

Table 13: English pilot: per-domain explicitation effects. Scores averaged across all four evaluated models.

### B Pipeline Prompts

#### B.1 Stage 2 (Safety, Objectivity, Temporal)

We used three LLM-based filters in Stage 2: content safety, objectivity, and temporal dependency. Below we excerpt only the core exclusion criteria from the prompts (full wording omitted).

- B.1.1 Content Safety Mark as inappropriate if the question–image pair includes:

- • Political content (politicians, parties, elections, political opinions)
- • Religious advocacy/criticism or conflicts
- • Hate/discrimination
- • Suicide or self-harm; sensitive mental-health topics
- • Sexual/adult content, nudity, explicit innuendo

- B.1.2 Objectivity Mark as inappropriate if the pair is subjective or ambiguous, e.g.:

- • Preference/aesthetic judgments (“pretty/ugly”, “which outfit is nicer?”)
- • Suitability/personal advice without criteria
- • Moral/intentionality speculation (“who is wrong?”, “good person?”)
- • Multiple valid interpretations or unverifiable answers

- B.1.3 Temporal Dependency Mark as inappropriate if the pair requires time-specific information, e.g.:

- • “today/now” weather, traffic, store hours, last train
- • Current events or status queries (“is it open now?”, “stock price today?”)
- • Questions that become invalid/meaningless as time passes

#### B.2 Stage 4 Prompt Excerpt (Image Dependency Rubric)

Input: (Q), model answer with image, model answer without image, optional gold answer snippet. Task: Compare the two answers and decide image dependency. Decision labels

- • IMAGE_REQUIRED: with-image answer is substantially more accurate/specific; text-only answer is vague, incorrect, or explicitly requests the image.
- • NO_IMAGE_NEEDED: both answers are comparable in correctness and specificity without relying on visual cues.
- • UNCERTAIN: evidence is inconclusive (e.g., partial improvements or conflicting signals).

Scoring (1–5 quality gap)

• 1: negligible difference; 3: clear but moderate gain; 5: decisive gain (critical visual details). Output (natural language)

- • Judgment: IMAGE_REQUIRED / NO_IMAGE_NEEDED / UNCERTAIN
- • Reason: brief comparison citing concrete differences
- • QualityGap: integer in {1,2,3,4,5}

#### B.3 Stage 5 (Checklist Generation)

This appendix provides the instruction prompt used for checklist generation along with illustrative examples of the resulting decompositions. We used GPT-4-mini to derive structured criteria directly from reference answers that users found satisfactory. These checklists therefore represent strict, human-aligned evaluation standards: a model must satisfy all listed criteria to be considered correct.

|Game (Stardew Valley) “What is the circled item in the screenshot?”<br><br>• Identify circled item as a sap tap (ᄉ액ᄎᄎᄀ)<br>• Mention install only on fully grown trees<br>• Explain how to obtain/craft it<br>• Note sap can be collected after time<br>|
|---|

|Economics/Management “Cost allocation: is S2 missing 100,000?”<br><br>• Provide correct S1/S2 values<br>• Reset self-allocation entries to zero<br>• Derive allocation ratios (0.5F, 0.4M)<br>|
|---|

“

“

|Daily Life “Is this ceiling tile asbestos?”<br><br>• Identify material as gypsum, not asbestos<br>• Explain gypsum board contains no asbestos<br>• Explicitly name “ᄉᄀ테ᄉ”<br>• Assure user it is safe<br>|
|---|

|Science “Why does neutron mass ratio decrease?”<br><br>• Explain neutron beta decay<br>• Clarify neutrons inside He nucleus<br>• Relate x-axis to cosmic cooling<br>• Interpret H:He ratio ≈ 3:1<br>|
|---|

“

“

- Figure 7: Examples of checklist decomposition across domains, generated in Stage 5. For brevity, the checklists shown here are abbreviated; full checklists typically contain 1–5 criteria per item.

#### B.4 Query Explicitation Prompt

The following prompt was used with GPT-5.1 (web search enabled) to generate explicitated versions of under-specified queries.

You rewrite incomplete, ambiguous, or context-dependent questions into clear, fully self-contained questions. Your goal is to produce a rewritten question that can be understood and answered on its own, without requiring prior conversation or hidden context. Preserve the original intent, scope, and tone of the question. Do NOT answer the question.

Rules:

- 1. Intent and scope preservation

- • Preserve what the original question is asking and its level of specificity.
- • Do not broaden or narrow the scope.
- • Do not generalize away concrete entities or situations implied by the original question or answer.

- 2. Essential context inclusion

- • Explicitly include essential context if it is implied or required to understand the question, such as: the relevant domain or subject; the specific scenario, task, or situation involved; named entities (e.g., people, organizations, characters, locations); concrete objects, items, or targets referenced.
- • Avoid vague references such as “this,” “that,” “here,” “the scene,” or “the above.”

- 3. Search usage

- • You may use search ONLY to identify widely accepted proper nouns (e.g., titles, names, commonly used labels) that are strongly implied by the original question or associated answer.
- • Do NOT use search to introduce new mechanics, steps, conditions, quantities, or interpretations.
- • Do NOT resolve ambiguity by inventing details.

- 4. Handling missing or ambiguous information

- • If critical context cannot be inferred with high confidence, include a brief clarifying placeholder inside the question, such as: [SPECIFY: missing detail].
- • Do not attempt to guess or “fix” the question beyond what the inputs support.

- 5. Image usage (if an image is provided)

- • You may incorporate information visible in the image to clarify the question.
- • The rewritten question must remain answerable without viewing the image.
- • Do not exhaustively describe the image or convert all visual details into text.
- • Include only visual information that is essential to understanding the question.

- 6. Language and style

- • Maintain a tone consistent with the original question.
- • Do not unnecessarily formalize or casualize the language.
- • Remove slang, conversational fillers, and vague references that reduce clarity.

Output requirements: Output ONLY the rewritten question text. No explanations, no bullet points, no headers. Do not include meta-instructions or commentary.

### C Human Annotation

#### C.1 Annotation Guidelines

Seven Korean-speaking annotators conducted human validation in three phases using custom web-based tools.

- C.1.1 Phase 1: Conservative Filtering Using the annotation interface shown in Figure 8, annotators independently reviewed each item along five dimensions, removing any item flagged by at least one annotator:

- • Image-Question Relevance: Assess whether images provide essential visual information required to answer the question.
- • Question-Answer Quality: Evaluate question clarity, answerability, and reference answer accuracy.
- • Checklist Validation: Review each LLM-generated checklist item for necessity, clarity, and completeness.
- • Category Appropriateness: Verify correct classification into one of 13 domain categories.
- • Overall Assessment: Flag items with fundamental issues such as inappropriate content or unsolvable questions.

- C.1.2 Phase 2: Refinement Three annotators refined surviving items through a separate annotation interface:

- • Question Rewriting: Rewrite unclear or ambiguous questions while preserving original intent and scope.
- • Checklist Revision: Evaluate each LLM-generated checklist item for appropriateness, revising unclear criteria or removing items not grounded in the original question–image pair.
- • Category Re-assignment: Re-assign categories where the original classification was incorrect, with option to propose new categories.

- C.1.3 Phase 3: Final Audit One senior annotator consolidated categories across the dataset and verified cross-item consistency.

[Figure 16]

###### Figure 8: Screenshot of our Phase 1 annotation tool. The interface (shown in Korean) allowed annotators to assess image relevance, question/answer appropriateness, checklist accuracy, and category assignment.

#### C.2 LLM Judge Failure Cases

- Table 14 presents representative examples of human annotator feedback for inappropriate judge evaluations, revealing systematic failure patterns.

Rating Human Reasoning (translated)

Very Inappropriate "Judge awarded points based on superficial word matching rather than actual checklist compliance"

Inappropriate "Judge gave 1 point despite response not addressing checklist criteria, incorrectly interpreting explicit mention as meeting requirements"

Inappropriate "Checklists 1,2,4 satisfied. Item 3 not clearly inappropriate but ambiguous and open to

interpretation" Inappropriate "Even if intent aligns with checklist, response lacks clarity and remains ambiguous" Inappropriate "Judge overlooked insufficient explanations that clearly failed checklist requirements"

Table 14: Representative human feedback explaining inappropriate judge ratings.

Analysis reveals judge failures primarily stem from: (1) superficial keyword matching without semantic understanding, (2) excessive leniency toward incomplete responses, and (3) difficulty distinguishing between implicit intent and explicit satisfaction of requirements.

### D LLM-as-Judge Prompt

This appendix provides the full prompt used for the checklist-based evaluation by the GPT-5-Mini judge. The prompt enforces explicitness, evidence grounding, and consistent scoring across items. For reproducibility, we include the full decision rules, evidence policy, and output format constraints.

[GOAL] Given a Question, Response, and a natural-language Checklist, decide for each checklist item whether the Response explicitly satisfies it: met = 1, partially met = 0.5, not met = 0. Final score = (# met) / (total checklist items).

[INPUT] [Question] QUESTION [Response] RESPONSE [Checklist] CHECKLIST Treat each string as one criterion. Remove numbering such as "1." or "2)". [DECISION RULES]

- • Use only the Response text. No outside knowledge or assumptions. If uncertain → 0.
- • Explicitness: direct fulfillment = 1, implicit or suggestive = 0.5, otherwise = 0.
- • Completeness ("all / every / complete"): explicit = 1, implied = 0.5, absent = 0.
- • Method requirements: actionable steps = 1, vague = 0.5, absent = 0.
- • "Various / multiple types": ≥ 2 specific types = 1, vague or 1 type = 0.5, none = 0.
- • Synonyms: unambiguous = 1, ambiguous = 0.5, different meaning = 0.

[EVIDENCE POLICY]

- • For 1 or 0.5: include a 10–60 character direct quote.
- • For 0: provide a brief explanation.
- • Each item must include: evidence → explanation → met.

[OUTPUT FORMAT — STRICT] <evidence>

- Item 1: evidence: "...direct quote from Response..."

- explanation: Brief justification referencing criteria met: 0 | 0.5 | 1
- Item 2: evidence: "..." explanation: ... met: 0 | 0.5 | 1

... (repeat) </evidence>

<score>

K/N </score>

[NOTES]

- • Output only the two blocks above.
- • No code fences or additional prose.

Qwen3-VL-4B-Instruct20.2321.009.9431.0423.0221.6218.5721.3535.007.6918.4011.2320.8318.051.912.841.323.021.353.071.912.853.561.261.971.102.730.56

Qwen3-VL-2B-Thinking11.8124.585.4319.6717.5313.3213.5816.0926.979.0317.6113.8112.3613.871.333.130.972.241.392.181.472.893.271.382.221.061.880.47

Qwen3-VL-2B-Instruct11.1319.715.2819.8217.4312.219.1714.7212.776.4312.835.8814.0411.151.533.070.862.221.271.951.332.622.331.001.960.662.120.43

InternVL3.514B15.5026.818.2620.7224.6417.4114.6717.7026.457.7415.7612.0519.7216.042.054.461.120.961.183.951.983.091.630.531.581.073.130.37

InternVL3.58B10.2223.117.1420.4420.1416.1611.2711.9922.965.2912.6812.5713.0113.161.083.120.571.871.893.351.562.922.080.791.730.251.220.82

InternVL3.54B7.7023.337.7219.7123.2018.8415.1114.9825.726.4813.8311.7814.9014.091.150.300.521.601.240.421.522.032.640.961.361.372.250.28

Qwen3-VL-4B-Thinking24.2345.0712.8330.6638.5329.7224.8931.2946.6914.9223.8525.3122.6026.182.083.711.352.861.753.392.353.314.211.892.421.392.540.65

Grok439.6436.9629.0044.4440.7047.6340.5736.7322.0924.7750.4330.2939.0236.081.891.161.492.791.131.861.601.652.861.784.901.280.200.53

Qwen3-VL-8B-Instruct25.3145.6513.6129.9734.8527.4625.2727.9835.9410.6624.4120.4025.0724.512.073.881.522.991.702.952.493.473.411.432.601.312.770.64

GPT5Nano22.9945.9810.4624.8111.4726.5921.4926.4223.5612.8126.1725.2732.8421.222.641.021.710.651.217.451.413.266.120.671.631.604.800.46

Qwen3-VL-8B-Thinking28.2749.5511.2133.9242.0729.7326.7032.5347.8314.1024.9028.7524.2028.012.123.611.172.701.783.692.373.873.911.702.301.592.800.67

GPT5Mini49.5960.4529.2250.1952.4951.6850.2844.3358.1925.5449.2241.1757.0245.213.742.401.715.530.441.474.754.963.942.203.112.730.531.21

Qwen3-VL-30B-A3B-Instruct31.1354.7118.8642.0240.4034.1831.9436.5351.3815.1230.2225.1729.6530.922.263.461.653.291.663.532.753.803.941.662.541.492.920.70

GPT559.9562.6132.3458.4136.3152.8546.9355.9654.7033.8054.9753.4255.0748.012.012.592.081.631.604.722.833.144.542.202.431.231.240.32

Qwen3-VL-30B-A3B-Thinking36.1956.3818.0638.4449.9238.4832.3437.9968.6917.8137.4035.4629.9535.412.233.501.733.211.873.812.653.873.712.132.631.663.040.74

Qwen3-VL-32B-Thinking33.9252.3919.7638.2151.9435.6629.3838.1664.5719.1935.5735.0437.5935.492.364.281.722.961.753.532.403.863.652.153.141.723.040.74

Qwen3-VL-32B-Instruct36.7456.3018.1341.2951.3941.7334.2843.3860.9219.6736.0234.4332.2536.082.173.781.623.181.863.762.473.723.771.873.151.663.060.73

Gemma327B20.3140.9013.7531.7134.9327.3626.7224.0723.859.4320.6618.6120.8122.531.181.491.553.211.526.281.122.012.741.302.620.402.150.28

Gemma312B15.1536.6010.5227.9128.7927.4419.2022.4017.257.2321.0113.4323.4118.760.691.321.441.301.393.601.271.472.891.121.651.470.130.63

Gemma34B12.4334.238.9119.6722.5021.2515.5918.2113.546.8419.5614.6813.4515.471.631.080.964.370.121.330.871.212.631.102.121.080.880.78

Ovis2-34B15.9040.159.8719.4423.9729.4619.4320.2722.919.1821.8618.7716.7818.501.352.160.770.450.561.470.583.312.461.412.891.370.260.03

Ovis2-16B11.2038.988.0821.5824.6823.9421.2014.8324.328.7220.2116.4716.1217.181.670.750.181.270.803.503.523.001.311.570.840.631.920.50

Ovis2-8B9.8033.626.0719.1819.4521.0218.3713.5119.818.0417.4213.1714.7714.461.301.540.303.281.851.981.831.335.290.533.080.351.800.37

Qwen3-VL-235B-A22B-Thinking34.1952.1223.9747.3049.1938.3730.0234.1856.5120.2934.1233.0434.8735.472.384.011.873.121.913.972.493.683.952.322.801.703.160.75

Ovis2-4B6.7623.666.0015.8916.1617.0516.4310.6813.167.1717.6514.268.3112.181.753.930.272.761.173.151.512.890.840.503.010.581.000.11

InternVL3.538B14.9430.959.0924.8528.7920.9019.2518.4024.548.5321.1016.4114.7618.010.634.821.571.520.274.441.900.192.470.170.841.982.120.39

Gemini2.5FlashLite25.9243.5417.9738.8441.7338.3030.6728.8245.6218.8234.1027.1632.6330.291.824.481.923.821.473.100.882.387.490.683.780.322.660.42

Gemini2.5Flash42.9856.1026.7048.0539.6245.8644.5946.1351.1431.9248.1244.3739.2641.050.343.951.046.643.141.702.105.193.873.632.370.992.251.38

PixtralLarge19.0935.0911.3324.3227.4023.4124.1619.0119.8911.5421.9318.0822.6420.101.822.331.333.362.291.211.094.661.102.501.340.620.670.41

MistralSmall24B15.3825.077.0022.2920.4721.5313.0715.3418.577.7613.8410.6116.3614.431.934.251.351.841.462.012.673.681.811.093.491.772.460.41

Gemini2.5Pro50.7362.1736.6751.0939.9356.2251.3246.0060.9444.3757.6953.4550.9148.541.632.331.751.721.434.322.575.151.361.182.810.570.920.18

MistralMedium3.124.7737.0116.0128.4833.7034.1424.4125.2238.9911.4625.4619.6231.0924.862.765.961.492.481.231.201.062.513.412.322.652.622.350.98

Pixtral12B8.7624.196.7417.4914.1216.4611.6611.446.836.1715.069.6012.9411.200.773.580.940.530.112.652.401.342.270.352.390.462.800.02

###### Skywork-R1V3-38B27.1247.9415.3032.3736.8437.2526.4328.2741.7114.7630.1027.3826.4227.760.742.921.632.440.691.802.631.954.531.962.730.260.260.58

VARCO-VISION-2.0-14B11.9034.767.9417.8322.0323.4621.8914.0512.687.8018.8414.9713.3115.550.794.780.852.302.713.161.092.801.902.641.750.571.310.50

Qwen2.5VL72B16.5331.3011.8025.5528.4623.5519.7225.8632.329.9721.0219.3625.3120.581.361.382.241.062.621.420.383.147.220.452.620.791.590.80

Qwen2.5VL7B10.3321.045.9518.9620.4918.5013.7017.0013.266.7114.0612.3513.2813.150.704.511.261.053.893.790.924.004.070.281.660.742.860.86

Qwen3-VL-235B-A22B-Instruct37.7554.4423.2843.1651.5147.4239.1440.9854.3122.7536.3337.4440.1038.412.293.961.933.451.764.002.654.034.082.422.921.743.230.76

HyperCLOVA-3B8.4229.746.3315.1718.5415.8013.3813.439.866.1614.2016.219.5312.660.982.980.491.400.412.140.673.832.440.701.750.931.860.18

VARCO-VISION-2.0-1.7B8.0921.345.9516.0717.7916.2212.7012.8812.548.1112.8112.1310.4611.871.211.502.380.960.631.160.321.085.350.681.010.723.570.46

Ovis2-1B4.8312.624.748.077.525.958.038.116.575.058.106.804.436.520.912.580.311.070.711.120.981.972.400.982.551.381.130.25

InternVL3.51B3.217.943.3910.329.125.743.297.7910.223.247.312.935.645.430.432.990.090.070.320.581.021.301.530.571.050.740.430.13

Ovis2-2B6.1416.105.3013.7412.2413.6411.9911.276.577.2811.339.738.989.540.221.010.832.341.704.431.142.011.320.642.190.563.880.22

InternVL3.52B5.3220.865.2415.5016.0512.698.947.6914.185.6310.147.039.079.480.254.130.341.860.872.871.541.601.711.263.140.512.280.49

- Qwen2.5VL3B6.0818.492.8212.7611.5413.769.226.8910.144.8810.317.856.548.202.153.900.441.171.702.510.161.470.980.183.460.380.840.36
- Qwen3-VLFamily

ModelITHealthGameEconSciMachDailyShopMathEntTransNatureCodeAvg

OpenGVLabInternVL3.5Family

Korean-specializedModels

GoogleGemmaFamily

AIDC-AIOvis2Family

Mistral/PixtralFamily

Qwen2.5-VLFamily

OtherOpen-source

Open-sourceModels

ProprietaryModels

Table15:Completeperformanceacrossall13categoriesforallevaluatedmodels(scoresin%).Allscoresarereportedasmean,whereSEisthestandarderrorover3SE

independentruns(n=3).

22.5

20.0

17.5

| |
|---|

AvgScore(%)

15.0

| |
|---|

Gemma-3 InternVL3 InternVL3.5

12.5

| |
|---|

| | | |
|---|---|---|
| | | |

10.0

Ovis2

| |
|---|

7.5

Qwen2.5-VL

VARCO2.0

5.0

HyperCLOVA

0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75

Log10(Parameter Count) [B]

(a) Performance scaling with model size. Accuracy rises up to ∼10B parameters but improves more slowly thereafter.

0.20

mean(30B 4B)

0.15

0.10

0.05

Health/ MedicalMathematicsMachineryScienceTransportationShopping/ ConsumerIT/ComputerDailyLifeNaturalObjectsBusiness/ Economics Coding/ Development Entertainment/Gaming Arts

(b) Domain-level results. Health/Medical yields the highest accuracy, whereas Entertainment/Gaming remains the most challenging.

Figure 9: Scaling and domain-level performance on HAERAE-Vision.

### E Additional Results & Analysis

#### E.1 Full Results

Table 15 reports the full category-wise results for the 45 evaluated models; we will continuously update the leaderboard with newly released models.

#### E.2 Performance by Model Scale

Grouping models by size tiers (Small ≤4B, Medium 8–14B, Large ≥30B) reveals a clear scaling trend: performance improves with size. Large models reach a mean score of 0.3009 (95% CI [0.2974, 0.3046]), more than double Medium (0.1460) and triple Small (0.0854). All pairwise differences are significant (permutation p≈0.001) with large effect sizes (Large–Small ∆ = +0.2155, d≈0.78), confirming that scaling reliably enhances multimodal reasoning.

However, gains become less pronounced beyond about 10B parameters. Accuracy still rises but with smaller marginal improvements (Figure 9a), indicating that scale alone cannot close the gap. Further progress likely depends on advances in reasoning and cultural grounding.

At the family level, commercial systems (Gemini, GPT, Sonar) consistently outperform open-weight models (e.g., InternVL3), with effect sizes around d = 0.7–1.2 (e.g., Gemini-2.5-Pro vs InternVL3 ∆≈0.49, d≈1.21). Thus, both scaling and architectural or cultural factors jointly drive performance.

#### E.3 Performance by Domain

Performance varies widely across the 13 domains (global mean = 0.1987, range 0.1179–0.332). Health/Medical achieves the highest checklist satisfaction (0.332), followed by Science (0.250), while Entertainment/Arts (0.118) and Gaming (0.119) remain the most challenging.

Within all domains, large models (≥30B) consistently outperform small models (≤4B) (permutation p < 0.05), with the largest gains in Health/Medical (∆ = +0.189) and Mathematics (∆ = +0.163). Even in Gaming and Entertainment, scale effects remain positive though absolute performance stays low (Figure 9b).

#### E.4 Investigating Failure Modes

In Table 15, we observe that VARCO-VISION and HyperCLOVA X—two Korean-focused VLMs—underperform multilingual counterparts of similar scale. While the precise reasons remain unclear due to the closed nature of these models and limited information about their training, we propose two possible explanations:

- (A) Training Data Coverage. Current benchmarks that capture progress on culturally grounded,

- information-deficient queries are scarce. Model developers may not have explicitly emphasized such aspects in their training data, leading to weaker performance on this type of evaluation.
- (B) Pretraining Scale and Robustness. Robustness to imperfect or fragmented user queries may emerge from exposure to large-scale, diverse pretraining corpora. Larger multilingual models are more likely to encounter noisy, colloquial, or partially specified inputs, thereby preparing them better for benchmarks of this kind.

#### E.5 Visual-Text Grounding Error Analysis

Table 6 shows that visual-text grounding (VTG) errors increase from 5.2% to 16.6% after explicitation. To understand whether this reflects newly introduced errors or pre-existing failures being reclassified, we tracked individual (question, model) pairs across conditions.

Error Tracking. Of the 461 VTG errors in the explicitated condition, 401 (87.0%) were newly surfacedcases that were already errors under original queries but classified under different failure categories. Only 60 (13.0%) were persistent VTG errors present in both conditions. Additionally, 104 VTG errors from the original condition were resolved by explicitation.

Source of Newly Surfaced VTG Errors. The 401 newly surfaced cases were previously classified under the following failure categories in the original condition (multi-label; percentages sum to >100%):

#### Original Failure Category %

Lack of explicitness 70.8 Object recognition 55.6 Procedural reasoning 29.2 Cultural concept mismatch 21.4

- Table 16: Original failure categories of newly surfaced VTG errors. Most were previously masked by lack-ofexplicitness failures.

Interpretation. This supports an error unmasking interpretation: under-specified queries produce vague responses that fail for surface-level reasons. Once explicitation removes this ambiguity, models are forced to engage with specific visual regions, exposing deeper grounding failures previously masked by the dominant lack-of-explicitness error mode. Supporting this, VTG error severity remains virtually identical across conditions (severe: 84.8% → 83.7%), indicating that explicitation reveals pre-existing failures rather than creating new ones.

Example. In one case (idx=22), the model’s response to the original under-specified query was vague and non-committal, merely identifying the character’s reading while omitting details—annotated as “lack of explicitness.” When given the explicitated query, the model attempted a specific answer but misread the character 惹 as 芯—annotated as “visual-text grounding.” The same question produced different error types because explicitation forced the model to engage with the specific visual region, shifting the failure from surface-level vagueness to a concrete grounding error.

### F Error Annotation Methodology

#### F.1 Annotation Setup

We used Claude 3.5 Sonnet as the LLM judge for error annotation, accessed via OpenRouter API with temperature=0.0 and max_tokens=2048. For each error case (model response with score < 1.0), the judge was provided with the original question, gold answer, checklist items, model response, and metadata (source, category, model name, score).

###### Root Cause (select one)

language Misunderstood Korean grammar, negation, particles, or expressions cultural_knowledge Lacked Korean-specific cultural/institutional knowledge general_reasoning Understood language and context but failed at reasoning

###### Failure Category (select 1–3)

object_recognition Fails to identify key objects in the image spatial_reasoning Misinterprets spatial relations cultural_concept_mismatch Misunderstands Korean-specific concepts or conventions visual_text_grounding Refers to the wrong region/entity relative to the question procedural_reasoning Fails to execute multi-step procedures lack_of_explicitness Misses explicit facts demanded by the checklist other None of the above fit

###### Severity

minor Almost correct; small missing detail moderate Mixed correctness; partially useful severe Largely incorrect or misleading

Table 17: Error annotation taxonomy.

#### F.2 Annotation Prompt

###### System prompt:

You are an impartial error analysis assistant for a Korean multimodal QA benchmark. Your job is to carefully inspect each example and classify the model’s failure according to a predefined taxonomy. Follow the provided schema exactly. Think step by step, but ONLY return the final JSON object in your response. Do NOT include explanations outside the JSON. Be strict and consistent with the taxonomy definitions.

User prompt: You are given one question-answering example from a Korean multimodal benchmark, together with a model’s answer and a detailed checklist used for scoring. Your goal is to analyze WHY the model failed or was only partially correct. Based on the question, gold answer, checklist, and model answer:

- 1. Decide the SINGLE most important root cause of failure: “language”, “cultural_knowledge”, or “general_reasoning”
- 2. Choose 1–3 failure_categories describing HOW the error manifests
- 3. Choose severity: “minor”, “moderate”, or “severe”
- 4. Provide analysis_comment: 2–3 sentences in Korean explaining why the answer is wrong or incomplete [Output format] Return ONLY a single JSON object: {“root_cause”: “...”, “failure_categories”: [...], “severity”: “...”, “analysis_comment”: “...”} [Metadata]

- source: {source} - category: {category} - question_idx: {question_idx} - model_name: {model} - model_score: {score} [Question] {question} [Gold answer] {answer} [Checklist items] {checklist} [Model answer] {model_response}

