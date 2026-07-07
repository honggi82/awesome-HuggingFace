# arXiv:2604.05117v1[cs.CV]6Apr2026

## Watch Before You Answer: Learning from Visually Grounded Post-Training

Yuxuan Zhang1,2,3, EunJeong Hwang1,2, Huaisong Zhang4, Penghui Du4, Yiming Jia5, Dongfu Jiang2,6, Xuan He7, Shenhui Zhang4, Ping Nie6, Peter West1, and Kelsey R. Allen1,2

1University of British Columbia 2Vector Institute 3Etude AI 4Kolors Team, Kuaishou Technology 5University of Toronto 6University of Waterloo 7University of Illinois at Urbana-Champaign

Abstract. It is critical for vision-language models (VLMs) to comprehensively understand visual, temporal, and textual cues. However, despite rapid progress in multimodal modeling, video understanding performance still lags behind text-based reasoning. In this work, we find that progress is even worse than previously assumed: commonly reported long video understanding benchmarks contain 40-60% of questions that can be answered using text cues alone. Furthermore, we find that these issues are also pervasive in widely used post-training datasets, potentially undercutting the ability of post-training to improve VLM video understanding performance. Guided by this observation, we introduce VidGround as a simple yet effective solution: using only the actual visually grounded questions without any linguistic biases for post-training. When used in tandem with RL-based post-training algorithms, this simple technique improves performance by up to 6.2 points relative to using the full dataset, while using only 69.1% of the original post-training data. Moreover, we show that data curation with a simple post-training algorithm outperforms several more complex post-training techniques, highlighting that data quality is a major bottleneck for improving video understanding in VLMs. These results underscore the importance of curating post-training data and evaluation benchmarks that truly require visual grounding to advance the development of more capable VLMs. Project page: http://vidground.etuagi.com.

Keywords: Long video understanding · Multimodal reasoning · Vision– language models

### 1 Introduction

Video understanding is vital for real-world AI, with applications including autonomous driving, online tutorial development, assistive robotics, and movie analysis, where models must accurately integrate visual, temporal, and textual cues [45,6,18]. Despite recent advances in vision-language models (VLMs), driven by larger video training datasets [1,16] and improved multimodal alignment techniques [31,53,48], performance has lagged behind text-based reasoning, especially

for tasks involving long-context video understanding such as MMVU [58] and VideoMME [21].

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |

- Fig.1: Performance decomposition on three video understanding benchmarks for four frontier VLMs: Qwen2.5-VL-7B and 32B (Q-7B, Q-32B) [7], and Gemini-

- 2.5-Pro and 3.1-Pro (G-2.5, G-3.1) [16,23]. Pink bars show text-only accuracy (no video input); blue bars show the additional visual gain from video access. The majority of benchmark performance comes from language priors rather than visual comprehension. Moreover, scaling up model size or version improves textonly reasoning but visual gain often remains flat or even decreases.

Here we show that the community’s progress in improving video understanding in VLMs is even worse than initially thought, with a majority of the gains coming from models’ abilities to answer questions without access to the video (Fig. 1).

This phenomenon, known as “linguistic shortcutting,” has been well established in Visual Question Answering (VQA) as a serious problem. As a result, video understanding benchmark designers have tried to avoid these pitfalls (e.g. VideoMME [21]) by filtering for questions that could be answered by the leading foundation model at the time without the video. However, as VLMs become stronger, we find that their gains come from being able to answer a larger portion of the benchmark without access to the video (Fig. 1), with their ability to answer video-based questions sometimes worsening. Consequently, these benchmarks are now problematic for measuring improvements in genuine video understanding.

We find that this problem is pervasive not just for evaluation benchmarks, but also for the most commonly used video understanding post-training datasets. Guided by this observation, we introduce VidGround, a simple yet effective approach to post-training VLMs: using only visually grounded questions. Although this strategy uses only 69.1% of the post-training data, it leads to improvements of up to 6.2 points in video understanding performance relative to post-training on the full dataset. More surprisingly, this simple approach also outperforms sev-

eral more advanced RL-based post-training strategies, including methods that employ token-level importance weighting [17], long-video sequence scaling [14], and adaptive test-time frame selection [49]. Overall, this suggests that a major bottleneck for improving VLM video understanding via post-training rests in the data. The effectiveness of using visually grounded questions suggests a deep well of improvement with algorithmic solutions that maximize this grounding signal.

Our contributions are as follows:

- – We systematically analyze linguistic biases in video understanding benchmarks and post-training datasets, finding that 40–60% of questions in popular benchmarks can be answered from text alone across multiple frontier models.
- – We introduce VidGround, a simple data curation approach for post-training that selects only visually grounded questions—those that genuinely require visual understanding to answer.
- – We show that post-training on only visually grounded data with a simple RL algorithm outperforms several more complex post-training techniques, demonstrating that data quality is a major bottleneck for improving video understanding in VLMs.

### 2 Related work

#### 2.1 Language priors in VLMs

Linguistic shortcutting for Visual Question Answering (VQA) has been known to be an issue since Goyal et al. [24]’s seminal work demonstrating that early VQA models learned to rely more on text than vision for answering questions. Since then, many recent studies [41,10,26,52,11,55,46,40,43] have shown that modern VLMs still exhibit clear weaknesses on basic vision-centric tasks, such as spatial reasoning, object counting, geometric perception, visual analogy, and finegrained recognition. Although their visual encoders are powerful, VLMs significantly underperform their visual encoders on tasks like image classification [57] and depth estimation [22]. Other analyses [34,38,35,3,29,47] report that VLMs exhibit a significant reliance on language priors rather than true visual grounding. As shown by Bleeker et al. [8], VLMs can learn shortcuts in which they rely on easily-discriminative but non-task-optimal features instead of capturing all the shared vision-language information they should.

However, despite many studies investigating this phenomenon for VQA, relatively little work has investigated it for video understanding. Given that video understanding should require synthesizing visual information across multiple frames, it may be less likely for linguistic shortcutting to present a major problem. For example, Park et al. [39] and Wu et al. [51] found that VLMs exhibit modality bias in favor of linguistic input in videos when subtitles are available, but did not investigate linguistic bias in the absence of subtitles (i.e. when given just the question text). Here we investigate linguistic shortcutting when neither subtitles nor the video is available.

#### 2.2 Strategies to improve VLM performance

Early attempts to mitigate linguistic shortcutting were applied to VQA models. These included augmenting how data is used for training by changing its weighting based on how easy it is to answer via text alone [36,9] or by changing the training objective to prioritize visual information [42,30].

Recent work instead focuses on post-training VLMs to improve their visual capabilities. Supervised fine-tuning (SFT) and reinforcement learning (RL) are the dominant paradigms for post-training. Chen et al. [12] demonstrated that generally RL is superior to SFT for post-training multimodal models, so we focus on the RL family of approaches. In the video domain, Video-R1 [20] represents the first systematic exploration of the RL paradigm for video reasoning. Video-R1 introduces a temporal contrastive auxiliary reward to Group Relative Policy Optimization (GRPO) [44] which has shown great success for text. Video-R1 further integrates curated video data and image-based reasoning samples, constructing Video-R1-CoT-165k for supervised warm-up and Video-R1260K for reinforcement learning. Other RL-style approaches include LongVILAR1 [14] which scales the R1-style GRPO framework to genuinely long-video settings, TW-GRPO [17] which computes token-level importance weights and down-weights redundant ones, and Video-RTS [49] which introduces a sparseto-dense test-time scaling strategy for improved efficiency during RL-based posttraining.

Here, we demonstrate how using visually grounded data in combination with RL-based post-training can outperform these approaches for improving VLM video understanding.

### 3 Analyzing linguistic biases in video understanding datasets

It is well known that linguistic biases are pervasive in VQA benchmarks. What about video understanding? Video understanding benchmarks such as VideoMME [21] were explicitly designed to avoid linguistic shortcutting, but were they successful?

We first investigate the prevalence of linguistic biases in video understanding benchmarks and post-training datasets, showing that substantial portions can be answered without video input. To analyze the quality of existing video understanding benchmarks and post-training datasets, we conduct a simple yet effective experiment: evaluating VLM performance on video datasets by providing only questions and answer choices while withholding all visual content. We denote questions that can be answered correctly without accessing any visual content as text-only answerable (TA) questions, and the remainder as visually grounded (VG) questions.

As shown in Table 1, we find that a substantial proportion of questions across popular video understanding benchmarks can be answered correctly by frontier models using text alone. For instance, VideoMME and MMVU (multiplechoice) contain 48.2% and 57.1% TA questions respectively as measured with

- Table 1: Text-only Answerability (TA) across video understanding benchmarks for frontier models. Each model receives only the question text and answer options—no video input—yet achieves accuracy far above random chance. (+x) denotes improvement relative to random choice. Results indicate that 40–60% of benchmark questions can be answered from text alone, revealing substantial linguistic bias in existing video understanding benchmarks.

Model VideoMME VideoMMMU MMVU Random Choice 25.0 9.8 19.8

- GPT-4o [1] 47.0 (+22.0) 38.6 (+28.8) 46.6 (+26.8) GPT-5-mini [37] 45.2 (+20.2) 37.9 (+28.1) 53.3 (+33.5)

- GPT-5 [37] 48.2 (+23.2) 41.0 (+31.2) 57.1 (+37.3)

- Gemini-2.5-Pro [16] 53.3 (+28.3) 52.7 (+42.9) 60.6 (+40.8)

- Gemini-3.1-Pro [23] 58.2 (+33.2) 61.1 (+51.3) 63.4 (+43.6) Claude-Sonnet-4.5 [4] 47.7 (+22.7) 44.3 (+34.5) 55.4 (+35.6) Claude-Opus-4.6 [5] 51.3 (+26.3) 52.7 (+42.9) 61.0 (+41.2)

GPT-5 [37] or 58.2% and 63.4% as measured with Gemini-3.1-Pro [23]. These numbers are substantially higher than chance performance and indicate that a large proportion of questions in these benchmarks can be answered correctly without visual information. Figure 2b further illustrates this breakdown, showing the proportion of VG versus TA questions across benchmarks along with the distribution of TA subcategories.

Similar patterns emerge in video understanding post-training datasets: VideoR1-260K [20] contains 30.9% TA questions (as measured with GPT-5-mini), suggesting that nearly one-third of the post-training data may not require genuine visual understanding.

These findings reveal significant biases in both current video understanding benchmarks and post-training datasets across multiple frontier models, with critical implications for model development and evaluation. When a substantial proportion of TA questions exists in evaluation benchmarks, model performance becomes artificially inflated, causing benchmark scores to misrepresent true video understanding capabilities (see Fig. 1). More critically, when video understanding post-training datasets contain high proportions of TA questions, they inevitably exacerbate linguistic biases in VLMs, leading models to develop stronger language priors rather than improved visual grounding.

#### 3.1 Analysis of text-only answerable questions

Moreover, we identify the four most common types of linguistic biases and discuss how they can encourage linguistic shortcuts in video understanding tasks.

Within common post-training data (Video-R1-260K [20]) and standard, widely

used video understanding benchmarks [21,25], we identified four common categories of TA questions (illustrated in Fig. 2a) that are answerable by VLMs without visual input.

|1. Textual Shortcuts and Linguistic Cues|2. External (or Commonsense) Knowledge|
|---|---|
|Q: How does the cookie change after being decorated?<br><br>A. It becomes more colorful.<br>B. It becomes smaller.<br>C. It becomes larger.<br>D. It becomes darker.<br><br><br>[Figure 1]<br><br>The word decorated linguistically implies adding visual elements or colors, so “more colorful” is the natural outcome.<br><br>VLM<br><br>[Figure 2]<br><br>|[Figure 3]<br><br>[Figure 4]<br><br>[Figure 5]<br><br>[Figure 6]<br><br>[Figure 7]|
|---|
<br><br>[Figure 8]<br><br>[Figure 9]<br><br>[Figure 10]<br><br>[Figure 11]<br><br>[Figure 12]|[Figure 13]<br><br>[Figure 14]<br><br>Q: What does the person rely on for support while descending the cliﬀside?<br><br>A. A ladder.<br>B. A harness.<br>C. A rope.<br>D. A bridge.<br><br><br>[Figure 15]<br><br>When someone descends a steep cliﬀside, they typically engage in an activity like rappelling or climbing, which requires a rope for safety and control. Therefore, the person most likely relies on a rope for support.<br><br>VLM<br><br>[Figure 16]<br><br>|[Figure 17]<br><br>[Figure 18]<br><br>[Figure 19]<br><br>[Figure 20]<br><br>[Figure 21]|
|---|
<br><br>[Figure 22]<br><br>[Figure 23]<br><br>[Figure 24]<br><br>[Figure 25]<br><br>[Figure 26]|
|3. Inferential and Elimination Strategies|4. Imagined (Hallucinated) Video Content|
|[Figure 27]<br><br>[Figure 28]<br><br>Q: In which direction is the person pouring the cooking oil?<br><br>A. Into the sink<br>B. Onto the floor<br>C. Into the wok<br>D. Onto the counter<br><br><br>[Figure 29]<br><br>By inferring from typical cooking behavior, oil is poured into cookware for heating; eliminating implausible options like the sink, floor, and counter—which would imply waste or error—leaves the wok as the only logical target.<br><br>VLM<br><br>[Figure 30]<br><br>|[Figure 31]<br><br>[Figure 32]<br><br>[Figure 33]<br><br>[Figure 34]<br><br>[Figure 35]|
|---|
<br><br>[Figure 36]<br><br>[Figure 37]<br><br>[Figure 38]<br><br>[Figure 39]<br><br>[Figure 40]<br><br>[Figure 41]|Q: What is the cat doing in the video?<br><br>A. The cat is playing with a toy.<br>B. The cat is grooming itself.<br>C. The cat is sleeping.<br>D. The cat is eating.<br><br><br>[Figure 42]<br><br>In the video, the cat repeatedly licks its fur and uses its paws to clean its face and ears. These are clear, characteristic self-grooming behaviors in cats.<br><br>VLM<br><br>[Figure 43]<br><br>|[Figure 44]<br><br>[Figure 45]<br><br>[Figure 46]<br><br>[Figure 47]<br><br>[Figure 48]|
|---|
<br><br>[Figure 49]<br><br>[Figure 50]<br><br>[Figure 51]<br><br>[Figure 52]|

- (a) Common categories of TA questions that allow VLMs to answer correctly without visual grounding. Examples are drawn from Video-R1-260K [20], with responses from GPT-5-mini.

| | | |
|---|---|---|
| | | |

- (b) Breakdown of TA and visually grounded (VG) items for VideoMME, VideoMMMU, and MMVU, classified using GPT-5-mini. TA items are further categorized into four reasoning types (see §3.1). Numbers indicate the percentage of examples in each category.

- Fig.2: Analysis of text-only answerable (TA) questions in video understanding benchmarks and post-training data. (a) We identify four common categories of linguistic shortcuts—textual cues, external knowledge, inferential strategies, and imagined content—that allow VLMs to answer correctly without watching the video. (b) These TA questions comprise 38–53% of popular benchmarks (classified using GPT-5-mini), with external knowledge being the dominant category in VideoMMMU and MMVU.

Textual shortcuts and linguistic cues. Questions contain surface-level hints that reveal the answer without visual grounding. For instance, when asked “How does the cookie change after being decorated?”, the option “It becomes more colorful” can be inferred linguistically, as the word “decorated” naturally implies adding visual elements or colors.

External knowledge. Questions can be answered using commonsense or world knowledge alone. For example, “What does the person rely on for support while descending the cliffside?” can be correctly answered as “A rope” based on common knowledge about rappelling and climbing activities, without observing the video content.

Inferential and elimination strategies. Questions allow models to succeed through logical reasoning and elimination of implausible options. In the question “In which direction is the person pouring the cooking oil?”, options like “into the sink,” “onto the floor,” and “onto the counter” can be eliminated as they imply waste or error, leaving “into the wok” as the only logical choice.

Imagined (hallucinated) video content. Models generate plausible video scenarios based solely on questions and options, which happen to align with actual content. For instance, when asked “What is the cat doing in the video?”, a model might correctly guess “The cat is grooming itself” by imagining typical cat behaviors, even without visual evidence.

These categories of TA questions reveal fundamental issues for both evaluation and post-training.

For evaluation, as frontier models become more powerful, their ability to take advantage of external knowledge and to use inferential and elimination reasoning strategies will only increase. This will further inflate model performance without reflecting improvements in genuine video understanding.

For post-training, when VLMs are post-trained on data containing substantial proportions of such TA questions, they may learn to exploit textual patterns and world knowledge instead of establishing robust vision-language associations, ultimately undermining their video understanding abilities.

These observations motivate a straightforward hypothesis: post-training on visually grounded data—questions that genuinely require visual understandingshould yield better video understanding than training on data contaminated by linguistic shortcuts. In the next section, we describe our approach to curating high-quality visually grounded post-training data, and in Section 5 we empirically validate that visually grounded training data leads to stronger video understanding performance.

### 4 VidGround: a simple approach to post-training

Guided by these analyses, we introduce VidGround, a simple technique for improving video understanding in VLMs through post-training. VidGround

combines reinforcement learning techniques for post-training (described in §4.1) with a simple data curation method (described in §4.2). While VidGround can be applied to any base VLM, we adopt Qwen2.5-VL-7B-Instruct [7] for its video understanding capabilities and computational efficiency.

#### 4.1 RL for video understanding post-training

We use reinforcement learning (RL) for post-training based on recent evidence that RL improves underlying visual recognition capabilities [13] while exhibiting less catastrophic forgetting than supervised fine-tuning (SFT) [15].

Optimization objective. We adopt Group Relative Policy Optimization (GRPO) [44]

augmented with techniques from DAPO [54] and temporal-aware rewards from Video-R1 [20]. Specifically, we employ token-level policy gradient loss with asymmetric clipping (increasing the value of εh) to make the training more efficient and stable. Our objective is formulated as:

J (θ) = E(q,a)∼D,{o

i}Gi=1∼πθold(· | q)

  1

 

|oi|

G

(1)

ℓclipi (θ) − βDKL (πθ∥πref)

G i=1 |oi|

t=1

i=1

where

ℓclipi (θ) = min ρi(θ)Aˆi, clip ρi(θ),1 − εl,1 + εh A ˆi ,

ri − mean(r) std(r)

πθ(oi | q) πθ

, Aˆi =

.

ρi(θ) =

(oi | q)

old

Here, q denotes the video-question input, oi represents the i-th sampled response from a group of G samples, Aˆi is the advantage computed from reward ri, and β controls the KL term relative to the reference policy πref. εl and εh are the lower and upper clipping bounds, respectively.

#### 4.2 Post-training data curation

We curate our post-training data from Video-R1-260K [20], which comprises 116,248 Video QA and 146,823 Image QA instances spanning diverse video and image understanding scenarios. Our goal is to select only visually grounded (VG) questions—those that genuinely require visual understanding to answer.

Selection pipeline. To identify VG questions, we prompt GPT-5-mini [37] with only the question text and answer options (no visual input) and retain only questions it cannot answer correctly. This text-only evaluation step selects 181,710 visually grounded samples (69.1% of the original dataset)—questions that require genuine visual understanding. We note that this selection is not an artifact of a single model: of the 181,710 VG questions selected by GPT-5-mini, 85% are

also unanswerable by Qwen2.5-VL-7B [7] in text-only mode, confirming that the retained questions genuinely require visual input. Furthermore, applying circular evaluation with Gemini-3.1-Pro [23]—rotating MCQ answer option positionsyields 97% agreement across permutations, confirming that the selection is robust to positional bias.

Training data variants. To investigate the causal effect of linguistic biases on post-training, we compare two data variants. The Full variant (263,071 samples) represents standard post-training without curation. The VG variant (181,710 samples) consists solely of visually grounded questions that GPT-5-mini cannot answer from text alone. VidGround is the combination of the VG dataset with the RL-based post-training outlined above.

### 5 Experiments

We first describe our experimental setup (§5.1), then compare VidGround against strong baselines (§5.2). We further analyze the contribution of each posttraining dataset variant (§5.3) and conclude with a qualitative comparison of reasoning chains between our model and the baselines (§5.4).

#### 5.1 Experimental setup

Post-training configuration We uniformly sample 16 frames per video and posttrain Qwen2.5-VL-7B for 700 steps using the GRPO objective described in §4.1. Our primary results (Table 2) are obtained using VidGround by post-training on the VG variant, which contains only visually grounded questions. To investigate the impact of post-training data composition, we report the performance of models post-trained on the Full variant in Table 3.

Benchmarks We evaluate on three established video understanding benchmarks: VideoMME [21], a comprehensive, general-purpose benchmark spanning perception and reasoning; VideoMMMU [25], focused on expert-level, multi-disciplinary video reasoning; and MMVU [58], emphasizing college-level, knowledge-intensive video comprehension. For MMVU, we evaluate on multiple-choice questions to ensure consistency and fair comparison. Following standard protocols, we report accuracy scores for all benchmarks.

Baseline post-training approaches We compare our approach against other strong 7B-scale post-training techniques including LongVILA-R1 [14], TW-GRPO [17], Video-RTS [49], and Video-R1 [20]. We also compare to our base model, Qwen2.5VL-7B [7], and its SFT variant (Qwen2.5-VL-7B-SFT). Notably, except for LongVILA-R1, all baseline models are originally post-trained from Qwen2.5VL-7B using publicly available post-training data.

- Table 2: Performance comparison of 7B-scale post-training methods on three video understanding benchmarks (VideoMME, VideoMMMU, MMVU). All methods except LongVILA-R1 are post-trained from Qwen2.5-VL-7B. Models are evaluated using 16, 32, and 64 frames per video. VidGround is post-trained with GRPO on visually grounded (VG) data only. Avg. columns report mean accuracy on the full benchmarks (Full) and on VG question subsets—questions that require video to answer (VG). Deltas show improvement over Qwen2.5-VL7B: (+x) and (−x). Bold indicates best; highlighted rows are ours.

Avg. Full VG

Frames Method VideoMME VideoMMMU MMVU

Qwen2.5-VL-7B 58.2 45.0 60.5 54.6 42.9 TW-GRPO 58.2 48.6 61.8 56.2 (+1.6) 44.1 (+1.2) LongVILA-R1-7B 55.5 38.8 59.1 51.1 (−3.5) 39.7 (−3.2) Video-RTS 58.7 47.1 61.8 55.9 (+1.3) 43.5 (+0.6) Qwen2.5-VL-7B-SFT 58.2 43.1 51.3 50.9 (−3.7) 41.1 (−1.8) Video-R1 56.9 44.7 54.5 52.0 (−2.6) 41.7 (−1.2)

16

###### VidGround 58.7 47.4 64.2 56.8 (+2.2) 45.2 (+2.3)

Qwen2.5-VL-7B 60.7 45.4 62.3 56.1 44.4 TW-GRPO 61.2 47.9 63.1 57.4 (+1.3) 45.9 (+1.5) LongVILA-R1-7B 60.2 40.7 61.5 54.1 (−2.0) 42.9 (−1.5) Video-RTS 61.3 47.7 65.0 58.0 (+1.9) 46.3 (+1.9) Qwen2.5-VL-7B-SFT 60.7 47.8 51.0 53.2 (−2.9) 44.6 (+0.2) Video-R1 60.2 45.4 56.2 53.9 (−2.2) 43.1 (−1.3)

32

###### VidGround 61.5 48.3 65.8 58.5 (+2.4) 47.6 (+3.2)

Qwen2.5-VL-7B 62.3 46.6 62.6 57.2 46.3 TW-GRPO 62.7 48.3 64.2 58.4 (+1.2) 48.2 (+1.9) LongVILA-R1-7B 61.6 41.2 58.8 53.9 (−3.3) 42.0 (−4.3) Video-RTS 62.9 46.4 63.9 57.7 (+0.5) 46.4 (+0.1) Qwen2.5-VL-7B-SFT 62.2 47.6 55.4 55.1 (−2.1) 45.8 (−0.5) Video-R1 61.2 45.4 53.2 53.3 (−3.9) 42.9 (−3.4)

64

VidGround 63.4 49.4 65.6 59.5 (+2.3) 47.9 (+1.6)

#### 5.2 Results

Across post-training approaches Table 2 presents our main results compared to strong 7B-scale post-training methods for video understanding at 16, 32, and 64 frames. Compared to Video-R1 [20], which trains on the full unfiltered dataset, VidGround improves by an average of 4.8, 4.6, and 6.2 points on Full Avg at 16, 32, and 64 frames respectively, while using only 69.1% of the training data. On visually grounded (VG) questions—those requiring video to answer—the gains are 3.5, 4.5, and 5.0 points on VG Avg. Relative to the base model Qwen2.5VL-7B, VidGround improves by 2.2, 2.4, and 2.3 points on Full Avg. At all frame settings, VidGround maintains the highest Full Avg performance among all baselines. These results demonstrate our simple data curation technique can effectively improve video understanding capability of models.

###### Video-RTS

###### LongVILA-R1-7B

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

50

50

Accuracy(%)

40

40

30

30

16 32 64

16 32 64

###### Video-R1

###### VidGround

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

50

50

Accuracy(%)

40

40

30

30

16 32 64

16 32 64

Frames

Frames

VideoMME VideoMMMU MMVU

- Fig.3: Accuracy on visually grounded (VG) questions—questions that cannot be answered from text alone—across 16, 32, and 64 frames on VideoMME, VideoMMMU, and MMVU for four post-training methods. VG questions are identified by retaining only questions that cannot be answered from text alone, as classified by GPT-5-mini (see §3). VidGround shows the strongest overall frame-scaling behavior, while baselines such as LongVILA-R1-7B and Video-R1 plateau or degrade on MMVU, and Video-RTS drops on VideoMMMU. This suggests that models post-trained on data containing linguistic shortcuts do not effectively leverage additional visual information.

Across datasets The benefits of VidGround are particularly pronounced on benchmarks that emphasize visual comprehension. On MMVU, which requires fine-grained visual understanding across diverse domains, VidGround outperforms Qwen2.5-VL-7B by 3.0 points at 64 frames. Similarly, on VideoMME, which includes many perception-intensive tasks, VidGround achieves the highest performance across all frame settings. On average, VidGround provides consistent gains across all datasets and temporal resolutions. These results indicate that visually grounded post-training data benefits models across diverse benchmarks. Importantly, visually grounded post-training does not degrade image understanding capabilities: VidGround improves over Qwen2.5-VL-7B on MME (648.9 vs. 624.3) and MMMU (58.7 vs. 56.7), indicating that curating post-training data for visual grounding does not narrow the training distribution in ways that harm non-video tasks.

Across frames We also investigate the models’ behaviors as the number of frames increases (Fig. 3). While VidGround generally improves from 16 to 64 frames, several baselines’ performance drops with additional frames. Specifically, on MMVU, LongVILA-R1-7B decreases by 3.1 points from 32 to 64 frames, Video-

- Table 3: Ablation study on post-training data composition. We compare GRPO trained on two data variants of Video-R1-260K [20]: Full (all 263K samples) and VG (181K visually grounded samples—questions that require visual understanding to answer). +clip-higher denotes asymmetric clipping (see §4.1). Avg. columns report mean accuracy on the full benchmarks (Full) and on VG question subsets—questions that require video to answer (VG). Deltas show improvement over Qwen2.5-VL-7B: (+x) and (−x). Training on VG data consistently outperforms the Full variant despite using 31% less data. Highlighted rows are ours.

Avg. Full VG

Frames Method Data VideoMME VideoMMMU MMVU

Base - 58.2 45.0 60.5 54.6 42.9 GRPO Full 56.9 44.7 54.5 52.0 (−2.6) 41.7 (−1.2) GRPO VG 58.7 47.4 64.2 56.8 (+2.2) 45.2 (+2.3)

16

+clip-higher VG 58.2 47.7 63.6 56.5 (+1.9) 45.1 (+2.2)

Base - 60.7 45.4 62.3 56.1 44.4 GRPO Full 60.2 45.4 56.2 53.9 (−2.2) 43.1 (−1.3) GRPO VG 61.5 48.3 65.8 58.5 (+2.4) 47.6 (+3.2)

32

+clip-higher VG 61.4 49.2 64.2 58.3 (+2.2) 46.8 (+2.4)

Base - 62.3 46.6 62.6 57.2 46.3 GRPO Full 61.2 45.4 53.2 53.3 (−3.9) 42.9 (−3.4) GRPO VG 63.4 49.4 65.6 59.5 (+2.3) 47.9 (+1.6) +clip-higher VG 63.5 48.6 65.3 59.1 (+1.9) 48.5 (+2.2)

64

R1 by 2.7 points over the same range. On VideoMMMU, Video-RTS drops by 0.4 points from 32 to 64 frames. These performance drops with increasing numbers of frames, despite access to more visual information, suggest that many existing post-trained models do not effectively leverage additional visual information. We hypothesize that this stems from the substantial proportion of text-only answerable questions in their post-training data, which encourages reliance on linguistic information. In contrast, VidGround shows the most consistent improvements with more frames, implying that post-training on visually grounded data enables the model to leverage temporal and visual cues more effectively.

#### 5.3 Ablation study

- Table 3 presents our ablation study comparing post-training data variants. We compare three configurations: GRPO trained on Full data (i.e., Video-R1 [20]), and GRPO on VG-only data with and without asymmetric clipping (+cliphigher). Overall, we find little impact of using asymmetric clipping, but large impacts depending on the data used for post-training.

Less is more VidGround (GRPO+VG in Table 3), post-trained on 181K visually grounded samples, consistently outperforms the model trained on the full 263K dataset across all frame settings, using only 69.1% of the post-training data. Compared to GRPO trained on the full dataset, VidGround achieves

average improvements of 4.8, 4.6, and 6.2 points on Full Avg, and 3.5, 4.5, and 5.0 points on VG Avg, at 16, 32, and 64 frames, respectively. These results suggest that curating a post-training set focused on visually grounded reasoning allows models to learn from fewer but more informative samples, improving both performance on video-grounded tasks and overall training efficiency.

Visually grounded training enables consistent frame-scaling Models trained on VG data show steady improvement as the number of frames increases (e.g., 56.8 to 58.5 to 59.5 on Full Avg for GRPO with VG), whereas the model trained on the full dataset exhibits inconsistent scaling and minimal gains (e.g., 52.0 to 53.9 to 53.3 for GRPO with Full). This pattern is even more pronounced on VG evaluation: GRPO with VG improves from 45.2 to 47.6 to 47.9 on VG Avg, while GRPO with Full stalls at 41.7 to 43.1 to 42.9, declining from 32 to 64 frames. This contrast highlights that visually grounded post-training allows models to more effectively leverage temporal information as additional frames are provided, while linguistic bias leads to plateauing or diminishing returns even when more visual data is available.

#### 5.4 Qualitative analysis

To further investigate the benefits of VidGround, we analyzed the reasoning patterns of Video-R1 [20] and VidGround on multiple video-dependent samples from VideoMMMU. Figure 4 shows a representative example illustrating the differences in how the two models process visual and textual information. We observe that Video-R1 relies heavily on textual context, producing answers based on abstract reasoning about art concepts without referencing the video. In contrast, VidGround grounds its analysis in actual video content (e.g., identifying specific visual elements such as lines, shapes, and colors), leading to a correct answer. This pattern is consistent across diverse expert domains—including medical imaging, structural engineering, chemistry, and public health (additional examples in the supplementary material). Across all analyzed instances, VidGround systematically anchors its reasoning in observed video content, whereas Video-R1 defaults to analyzing questions through prior knowledge and linguistic cues. Notably, even when both models arrive at the correct answer, their reasoning processes differ fundamentally—VidGround derives the answer from video content while Video-R1 reaches it through text-based elimination—indicating that accuracy metrics alone cannot fully capture whether a model genuinely leverages visual information.

### 6 Discussion

Notably, when evaluated with frontier models, some of the most popular video understanding benchmarks, such as VideoMME [21] and VideoMMMU [25], contain 40–60% of questions that can be answered using the question text alone, with the strongest models exceeding 50%. This was not just an issue for a single

[Figure 53]

Video

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

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

###### .. ...

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

Question

Which of the following elements of art are most prominently featured in this artwork?

A. Line B. Shape C. Form D. Value E. Color F. Texture G. Balance H. Space

I. Contrast. J. Proportion

|Video-R1|Ours|
|---|---|
|[Figure 81]<br><br>[Figure 82]<br><br>|Given the analysis, the most prominently featured elements are line and shape. However, since the question asks for the most prominent, and considering the variety of lines and shapes, the answer leans towards:<br><br>B. Shape|
|---|
<br><br>Relies on text<br><br>The artwork in the video showcases various elements of art, but let’s break down which one stands out the most.<br><br>1. Line: There are numerous lines used in the drawings, especially in the abstract and geometric shapes. The lines are varied in direction and style, contributing significantly to the overall composition.<br>2. Shape: Shapes are also prominent, with both geometric and organic forms present. The use of shapes to create patterns and structures is evident.<br>3. Form: While there are some three-dimensional elements like the cube and the sphere, they are not the main focus. The emphasis is more on twodimensional elements rather than creating a sense of volume through form.<br>4. Value: The use of value (light and shadow) is minimal compared to other elements. The colors are bright and flat, without much shading.<br>5. Color: Color is present, but it is not the most dominant element. The colors are used to fill shapes and add contrast, but they do not create a strong color palette.<br>6. Texture: There is no significant texture in the artwork. The surfaces are smooth and flat.<br>7. Balance: The balance is somewhat achieved, but it is not the most striking feature. The arrangement of elements is somewhat balanced, but it is not the primary focus.<br>8. Space: The concept of space is present, but it is not the most prominent. The artwork does not explore depth or negative space extensively.<br>9. Contrast: There is some contrast, but it is not the most dominant element. The colors and lines provide some contrast, but it is not the main focus.<br>10. Proportion: The proportions are somewhat consistent, but they are not the most striking feature.<br><br><br>Final answer: B. Shape||Given the content of the video, the elements that are most prominently featured are likely to be those that are discussed and demonstrated in the video. Based on the video content, the most prominent elements are: Line, Shape, Color, Texture, Space|
|---|
<br><br>Correct visual reference<br><br>To determine which elements of art are most prominently featured in the artwork, let’s analyze the video content step by step:<br><br>1. Line: The video shows a hand drawing various types of lines such as horizontal, diagonal, curved, and dotted. This suggests that line is a significant element in the artwork.<br>2. Shape: The video also shows the drawing of diﬀerent shapes like triangles, circles, and squares, indicating that shape is another prominent element.<br>3. Form: The video briefly shows a 3D form, which suggests that form is also an element being discussed.<br>4. Value: The video mentions value, which refers to the lightness or darkness of a color, and shows examples of how value can be used in art.<br>5. Color: The video includes a color wheel and discusses primary colors, indicating that color is a key element.<br>6. Texture: The video briefly shows a textured surface, suggesting that texture is also an element being discussed.<br>7. Balance: The video does not focus on balance as a primary element.<br>8. Space: The video mentions space, including positive and negative space, indicating that space is a significant element.<br><br><br>However, since the question asks for the most prominently featured elements, and considering the video content, the most prominent elements are likely to be: A. Line, B. Shape, E. Color, F. Texture, H. Space<br><br>The most comprehensive answer would be: A. Line, B. Shape, E. Color, F. Texture, H. Space<br><br>|But if we need to choose the most prominent one based on the video content, the answer would be:|
|---|
<br><br>A. Line<br><br>So, the correct answer is:<br><br>A. Line|

: Over-reliance on text : Correct visual reference

| |
|---|

| |
|---|

- Fig.4: Qualitative comparison of reasoning paths on a VideoMMMU art analysis question. Given a video demonstrating visual art elements, the model must identify which elements are shown. VidGround (right) references specific visual elements observed in the video frames, such as lines, shapes, and colors (blue boxes), leading to the correct answer. In contrast, Video-R1 [20] (left) analyzes artistic concepts abstractly without grounding in the actual video content (red boxes), arriving at the wrong answer. This illustrates how post-training on visually grounded data encourages models to attend to visual evidence rather than relying on linguistic priors.

frontier model, but rather a consistent trend across many leading VLMs, and presents a serious issue for measuring video understanding progress. We found a similar trend in the composition of post-training data for video understanding. Over 30% of the data in one of the most popular post-training datasets, VideoR1-260K [20], was also answerable using text alone. Guided by these observations, we developed a simple yet effective post-training strategy, VidGround, for improving video understanding in VLMs: using only visually grounded questions for post-training. In combination with a simple RL-based post-training algorithm, this strategy outperforms five strong baselines when measured on both visually grounded evaluation splits as well as standard benchmark performance. Our approach also provides notable benefits in training data efficiency, achieving stronger performance with considerably less data. Furthermore, models post-trained on visually grounded data exhibit more consistent frame-scaling behavior, continuing to improve as more visual frames are provided, whereas baselines trained on unfiltered data plateau or degrade (Table 3). Overall, our findings highlight the importance of curating post-training data that truly requires visual reasoning, offering a simple yet powerful direction for building more robust and visually grounded VLMs.

### References

- 1. Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F.L., Almeida, D., Altenschmidt, J., Altman, S., Anadkat, S., et al.: Gpt-4 technical report. arXiv preprint arXiv:2303.08774 (2023)
- 2. Agarwal, S., Ahmad, L., Ai, J., Altman, S., Applebaum, A., Arbus, E., Arora, R.K., Bai, Y., Baker, B., Bao, H., et al.: gpt-oss-120b & gpt-oss-20b model card. arXiv preprint arXiv:2508.10925 (2025)
- 3. Alhamoud, K., Alshammari, S., Tian, Y., Li, G., Torr, P.H., Kim, Y., Ghassemi, M.: Vision-language models do not understand negation. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 29612–29622 (2025)
- 4. Anthropic: Introducing claude sonnet 4.5 (2025), https://www.anthropic.com/ news/claude-sonnet-4-5
- 5. Anthropic: Introducing claude opus 4.6 (2026), https://www.anthropic.com/ news/claude-opus-4-6
- 6. Bahl, S., Mendonca, R., Chen, L., Jain, U., Pathak, D.: Affordances from human videos as a versatile representation for robotics. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 13778–13790 (2023)
- 7. Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J., et al.: Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923

(2025)

- 8. Bleeker, M., Hendriksen, M., Yates, A., de Rijke, M.: Demonstrating and reducing shortcuts in vision-language representation learning. arXiv preprint arXiv:2402.17510 (2024)
- 9. Cadene, R., Dancette, C., Cord, M., Parikh, D., et al.: Rubi: Reducing unimodal biases for visual question answering. Advances in neural information processing systems 32 (2019)
- 10. Campbell, D., Rane, S., Giallanza, T., De Sabbata, C.N., Ghods, K., Joshi, A., Ku, A., Frankland, S., Griffiths, T., Cohen, J.D., et al.: Understanding the limits

- of vision language models through the lens of the binding problem. Advances in Neural Information Processing Systems 37, 113436–113460 (2024)
- 11. Chandhok, S., Fan, W.C., Shwartz, V., Balasubramanian, V.N., Sigal, L.: Response wide shut? surprising observations in basic vision language model capabilities. In: Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). pp. 25530–25545 (2025)
- 12. Chen, H., Tu, H., Wang, F., Liu, H., Tang, X., Du, X., Zhou, Y., Xie, C.: Sft or rl? an early investigation into training r1-like reasoning large vision-language models. arXiv preprint arXiv:2504.11468 (2025)
- 13. Chen, H., Razin, N., Narasimhan, K., Chen, D.: Retaining by doing: The role of on-policy data in mitigating forgetting. arXiv preprint arXiv:2510.18874 (2025)
- 14. Chen, Y., Huang, W., Shi, B., Hu, Q., Ye, H., Zhu, L., Liu, Z., Molchanov, P., Kautz, J., Qi, X., et al.: Scaling rl to long videos. arXiv preprint arXiv:2507.07966

(2025)

- 15. Chu, T., Zhai, Y., Yang, J., Tong, S., Xie, S., Schuurmans, D., Le, Q.V., Levine, S., Ma, Y.: Sft memorizes, rl generalizes: A comparative study of foundation model post-training. arXiv preprint arXiv:2501.17161 (2025)
- 16. Comanici, G., Bieber, E., Schaekermann, M., Pasupat, I., Sachdeva, N., Dhillon,

I., Blistein, M., Ram, O., Zhang, D., Rosen, E., et al.: Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261 (2025)

- 17. Dang, J., Wu, J., Wang, T., Lin, X., Zhu, N., Chen, H., Zheng, W.S., Wang, M., Chua, T.S.: Reinforcing video reasoning with focused thinking. arXiv preprint arXiv:2505.24718 (2025)
- 18. Elhenawy, M., Ashqar, H.I., Rakotonirainy, A., Alhadidi, T.I., Jaber, A., Tami, M.A.: Vision-language models for autonomous driving: Clip-based dynamic scene understanding. Electronics 14(7), 1282 (2025)
- 19. Fang, X., Mao, K., Duan, H., Zhao, X., Li, Y., Lin, D., Chen, K.: Mmbench-video: A long-form multi-shot benchmark for holistic video understanding. Advances in Neural Information Processing Systems 37, 89098–89124 (2024)
- 20. Feng, K., Gong, K., Li, B., Guo, Z., Wang, Y., Peng, T., Wu, J., Zhang, X., Wang, B., Yue, X.: Video-r1: Reinforcing video reasoning in mllms. arXiv preprint arXiv:2503.21776 (2025)
- 21. Fu, C., Dai, Y., Luo, Y., Li, L., Ren, S., Zhang, R., Wang, Z., Zhou, C., Shen, Y., Zhang, M., et al.: Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 24108–24118 (2025)
- 22. Fu, S., Bonnen, T., Guillory, D., Darrell, T.: Hidden in plain sight: Vlms overlook their visual representations. arXiv preprint arXiv:2506.08008 (2025)
- 23. Google: Gemini 3.1 pro: Best for complex tasks and bringing creative concepts to life (2026), https://blog.google/innovation-and-ai/models-and-research/ gemini-models/gemini-3-1-pro/
- 24. Goyal, Y., Khot, T., Summers-Stay, D., Batra, D., Parikh, D.: Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 6904–6913 (2017)
- 25. Hu, K., Wu, P., Pu, F., Xiao, W., Zhang, Y., Yue, X., Li, B., Liu, Z.: Video-mmmu: Evaluating knowledge acquisition from multi-discipline professional videos. arXiv preprint arXiv:2501.13826 (2025)

- 26. Huang, K.H., Qin, C., Qiu, H., Laban, P., Joty, S., Xiong, C., Wu, C.S.: Why vision language models struggle with visual arithmetic? towards enhanced chart and geometry understanding. arXiv preprint arXiv:2502.11492 (2025)
- 27. Hurst, A., Lerer, A., Goucher, A.P., Perelman, A., Ramesh, A., Clark, A., Ostrow, A., Welihinda, A., Hayes, A., Radford, A., et al.: Gpt-4o system card. arXiv preprint arXiv:2410.21276 (2024)
- 28. Jin, P., Takanobu, R., Zhang, W., Cao, X., Yuan, L.: Chat-univi: Unified visual representation empowers large language models with image and video understanding. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 13700–13710 (2024)
- 29. Lee, K.i., Kim, M., Yoon, S., Kim, M., Lee, D., Koh, H., Jung, K.: Vlind-bench: Measuring language priors in large vision-language models. In: Findings of the Association for Computational Linguistics: NAACL 2025. pp. 4129–4144 (2025)
- 30. Liang, Z., Hu, H., Zhu, J.: Lpf: A language-prior feedback objective function for debiased visual question answering. In: Proceedings of the 44th international ACM SIGIR conference on research and development in information retrieval. pp. 1955– 1959 (2021)
- 31. Lin, B., Ye, Y., Zhu, B., Cui, J., Ning, M., Jin, P., Yuan, L.: Video-LLaVA: Learning united visual representation by alignment before projection. In: AlOnaizan, Y., Bansal, M., Chen, Y.N. (eds.) Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing. pp. 5971–5984. Association for Computational Linguistics, Miami, Florida, USA (Nov 2024). https://doi.org/10.18653/v1/2024.emnlp-main.342, https://aclanthology. org/2024.emnlp-main.342/
- 32. Lin, B., Ye, Y., Zhu, B., Cui, J., Ning, M., Jin, P., Yuan, L.: Video-llava: Learning united visual representation by alignment before projection. In: Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing. pp. 5971–5984 (2024)
- 33. Liu, A., Mei, A., Lin, B., Xue, B., Wang, B., Xu, B., Wu, B., Zhang, B., Lin, C., Dong, C., et al.: Deepseek-v3. 2: Pushing the frontier of open large language models. arXiv preprint arXiv:2512.02556 (2025)
- 34. Luo, T., Cao, A., Lee, G., Johnson, J., Lee, H.: Probing visual language priors in vlms. arXiv preprint arXiv:2501.00569 (2024)
- 35. Miller, D., Su¨nderhauf, N., Kenna, A., Mason, K.: Open-set recognition in the age of vision-language models. In: European Conference on Computer Vision. pp. 1–18. Springer (2024)
- 36. Niu, Y., Tang, K., Zhang, H., Lu, Z., Hua, X.S., Wen, J.R.: Counterfactual vqa: A cause-effect look at language bias. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 12700–12710 (2021)
- 37. OpenAI: Gpt-5. https://openai.com (2025)
- 38. Parashar, S., Lin, Z., Liu, T., Dong, X., Li, Y., Ramanan, D., Caverlee, J., Kong, S.: The neglected tails in vision-language models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 12988–12997 (2024)
- 39. Park, J., Jang, K.J., Alasaly, B., Mopidevi, S., Zolensky, A., Eaton, E., Lee, I., Johnson, K.: Assessing modality bias in video question answering benchmarks with multimodal large language models. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 39, pp. 19821–19829 (2025)
- 40. Peng, W., Xie, S., You, Z., Lan, S., Wu, Z.: Synthesize diagnose and optimize: Towards fine-grained vision-language understanding. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 13279– 13288 (2024)

- 41. Rahmanzadehgervi, P., Bolton, L., Taesiri, M.R., Nguyen, A.T.: Vision language models are blind. In: Proceedings of the Asian Conference on Computer Vision. pp. 18–34 (2024)
- 42. Ramakrishnan, S., Agrawal, A., Lee, S.: Overcoming language priors in visual question answering with adversarial regularization. Advances in neural information processing systems 31 (2018)
- 43. Ranasinghe, K., Shukla, S.N., Poursaeed, O., Ryoo, M.S., Lin, T.Y.: Learning to localize objects improves spatial reasoning in visual-llms. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 12977– 12987 (2024)
- 44. Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y., Wu, Y., et al.: Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300 (2024)
- 45. Tang, Y., Bi, J., Xu, S., Song, L., Liang, S., Wang, T., Zhang, D., An, J., Lin, J., Zhu, R., Vosoughi, A., Huang, C., Zhang, Z., Liu, P., Feng, M., Zheng, F., Zhang, J., Luo, P., Luo, J., Xu, C.: Video understanding with large language models: A survey. IEEE Transactions on Circuits and Systems for Video Technology pp. 1–1

(2025). https://doi.org/10.1109/TCSVT.2025.3566695

- 46. Tong, S., Liu, Z., Zhai, Y., Ma, Y., LeCun, Y., Xie, S.: Eyes wide shut? exploring the visual shortcomings of multimodal llms. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 9568–9578 (2024)
- 47. Vo, A., Nguyen, K.N., Taesiri, M.R., Dang, V.T., Nguyen, A.T., Kim, D.: Vision language models are biased. arXiv preprint arXiv:2505.23941 (2025)
- 48. Wang, X., Li, C., Yang, J., Zhang, K., Liu, B., Xiong, T., Huang, F.: Llava-critic-r1: Your critic model is secretly a strong policy model. arXiv preprint arXiv:2509.00676

(2025)

- 49. Wang, Z., Yoon, J., Yu, S., Islam, M.M., Bertasius, G., Bansal, M.: Video-rts: Rethinking reinforcement learning and test-time scaling for efficient and enhanced video reasoning. In: Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing. pp. 28114–28128 (2025)
- 50. von Werra, L., Belkada, Y., Tunstall, L., Beeching, E., Thrush, T., Lambert, N., Huang, S., Rasul, K., Galloue´dec, Q.: TRL: Transformer Reinforcement Learning, https://github.com/huggingface/trl
- 51. Wu, H., Tang, M., Zheng, X., Jiang, H.: When language overrules: Revealing text dominance in multimodal large language models. arXiv preprint arXiv:2508.10552

(2025)

- 52. Xu, Y., Zhu, L., Yang, Y.: Mc-bench: A benchmark for multi-context visual grounding in the era of mllms. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 17675–17687 (2025)
- 53. Xue, H., Sun, Y., Liu, B., Fu, J., Song, R., Li, H., Luo, J.: CLIP-vip: Adapting pre-trained image-text model to video-language alignment. In: The Eleventh International Conference on Learning Representations (2023), https://openreview. net/forum?id=GNjzMAgawq
- 54. Yu, Q., Zhang, Z., Zhu, R., Yuan, Y., Zuo, X., Yue, Y., Dai, W., Fan, T., Liu, G., Liu, L., et al.: Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476 (2025)
- 55. Zhang, G., Zhang, Y., Zhang, K., Tresp, V.: Can vision-language models be a good guesser? exploring vlms for times and location reasoning. In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. pp. 636–645

(2024)

- 56. Zhang, Y., Wu, J., Li, W., Li, B., Ma, Z., Liu, Z., Li, C.: Video instruction tuning with synthetic data. arXiv preprint arXiv:2410.02713 (2024)
- 57. Zhang, Y., Unell, A., Wang, X., Ghosh, D., Su, Y., Schmidt, L., Yeung-Levy, S.: Why are visually-grounded language models bad at image classification? Advances in Neural Information Processing Systems 37, 51727–51753 (2024)
- 58. Zhao, Y., Zhang, H., Xie, L., Hu, T., Gan, G., Long, Y., Hu, Z., Chen, W., Li, C., Xu, Z., et al.: Mmvu: Measuring expert-level multi-discipline video understanding. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 8475–8489 (2025)

### Appendix

#### Appendix Overview

- A Multi-model agreement analysis
- B Cross-task generalization: image QA
- C Text-only answerability analysis across video understanding datasets
- D Implementation details
- E Additional qualitative analysis

In the appendix, we provide additional empirical evidence and implementation details supporting our main findings. We first analyze multi-model agreement on text-only answerability detection and compare alternative data curation strategies of varying strictness (§A). We then demonstrate cross-task generalization in §B, showing that visually grounded post-training does not degrade image QA performance. We present a comprehensive text-only answerability analysis across video understanding datasets (§C), evaluating a broad range of frontier models to demonstrate the pervasive nature of linguistic biases in video benchmarks. We provide implementation details including training configurations, evaluation setup, and computational resources in §D. Finally, we present additional qualitative analyses (§E) comparing reasoning paths between Video-R1 [20] and VidGround, illustrating how video-dependent post-training data leads to models that frequently refer to video content rather than relying on linguistic shortcuts and exhibit stronger visually grounded reasoning abilities.

### A Multi-model agreement analysis

#### A.1 Multi-model agreement

To validate the robustness of text-only answerability detection, we evaluate three frontier models—GPT-5-mini [37], Qwen2.5-VL-7B [7], and Gemini-3.1Pro [23]—on Video-R1-260K [20] in text-only mode. Figure 5 shows the overlap of visually grounded (VG) questions across models.

Key findings: (1) The three models show strong agreement on which questions require visual input. GPT-5-mini [37] cannot answer 181,710 questions (69.1%) correctly without video, Qwen2.5-VL-7B [7] cannot answer 198,652 (75.5%), and their intersection (both models fail) contains 154,860 (58.9%). All three models fail on the same 145,486 questions (55.3%), forming a robust core of visually grounded data. (2) Inter-model agreement on VG questions is high (Jaccard index 68.7%), confirming that questions requiring visual understanding are consistently identified across diverse model architectures, supporting the robustness of our data curation approach.

- Fig.5: Multi-model agreement analysis on Video-R1-260K [20]. Venn diagram showing the overlap of visually grounded (VG) questions—those that each model cannot answer correctly in text-only mode (no visual input)—across three frontier models: GPT (GPT-5-mini [37]), Qwen (Qwen2.5-VL-7B [7]), and Gemini (Gemini-3.1-Pro [23]). GPT cannot answer 181,710 questions (69.1%) correctly without video, Qwen cannot answer 198,652 (75.5%), and their intersection (GPT ∩ Qwen) contains 154,860 questions (58.9%) that neither model can solve without visual input. All three models fail on the same 145,486 questions (55.3%), forming a robust core of visually grounded data. Inter-model agreement on VG questions is high (Jaccard index 68.7%), confirming that questions requiring visual understanding are consistently identified across diverse architectures.

#### A.2 Alternative data curation strategies

We investigate whether multi-model consensus data curation improves upon our single-model approach. We evaluate three frontier models on all 263,071 samples from Video-R1-260K [20] in text-only mode (no visual input):

- 1. GPT-5-mini [37]: Single-pass text-only evaluation on all samples. The model answers 81,361 questions (30.9%) correctly without video, leaving 181,710 VG questions (69.1%).
- 2. Qwen2.5-VL-7B [7]: For MCQ questions, we employ circular evaluation [19] with option permutation to mitigate positional bias—a question is considered VG unless the model answers correctly under all permutations. For non-MCQ questions, we use Pass@10 sampling (10 independent responses; VG if none correct). With 2-permutation circular evaluation, 198,652 questions (75.5%) remain VG.
- 3. Gemini-3.1-Pro [23]: For MCQ questions, 3-permutation circular evaluation. For non-MCQ questions, direct text-only evaluation. Samples not covered by Gemini evaluation (∼2K) are conservatively classified as visually grounded (VG).

A question is retained as visually grounded (VG) only if fewer than 2 models can answer it correctly in text-only mode. This soft consensus threshold balances curation quality with data retention—requiring all three models to fail (<3 correct) would retain too many borderline questions, while requiring all models to fail (<1 correct) would discard too aggressively.

###### Video-R1-260K

263K samples

###### Gemini

###### Qwen

3-perm circ. + direct eval

###### GPT

2-perm circ. + Pass@10

Single-pass

Retain if < 2 correct

###### VidGround-M1

161K samples (61.1%)

(a) 2-perm circular eval for Qwen

###### Video-R1-260K

263K samples

###### Gemini

###### Qwen

3-perm circ. + direct eval

###### GPT

4-perm circ. + Pass@10

Single-pass

Retain if < 2 correct

###### VidGround-M2

148K samples (56.2%)

(b) 4-perm circular eval for Qwen

- Fig.6: Multi-model data curation pipelines for VidGround-M1 and VidGround-M2 variants (GPT: GPT-5-mini [37], Qwen: Qwen2.5-VL-7B [7], Gemini: Gemini-3.1-Pro [23]). Both pipelines evaluate all ∼263K samples from VideoR1-260K [20] in text-only mode. GPT uses single-pass evaluation, Qwen uses circular evaluation with option permutation for MCQ + Pass@10 for openended, and Gemini uses 3-permutation circular evaluation for MCQ + direct text evaluation for open-ended. A question is retained as VG only if fewer than 2 models can answer it correctly without visual input. The key difference is the number of circular evaluation permutations for Qwen: (a) VidGround-M1 uses 2-permutation evaluation, retaining ∼161K samples (61.1%); (b) VidGroundM2 uses stricter 4-permutation evaluation, retaining ∼148K samples (56.2%). Highlighted boxes indicate the differing component.

Model selection rationale. We select these three models to span diverse architectures and capability levels: GPT-5-mini as a strong closed-source model that serves as our primary single-model filter, Qwen2.5-VL-7B as the on-policy training model (see below), and Gemini-3.1-Pro as the strongest available model to maximize detection of questions answerable without visual input. Fig. 6 illustrates the multi-model data curation pipeline.

Data curation variants. We compare three approaches of increasing strictness:

- – VidGround (single-model, 181K): Our primary approach (§4.2), using GPT-5-mini only. Retains 181,710 samples (69.1%).
- – VidGround-M1 (161K): Soft multi-model data curation strategy with 2permutation circular evaluation for Qwen2.5-VL-7B. Retains 160,837 samples (61.1%).

- – VidGround-M2 (148K): Stricter variant using 4-permutation circular evaluation for Qwen2.5-VL-7B, reducing false negatives from positional bias. Retains 147,850 samples (56.2%).

On-policy data curation with Qwen2.5-VL-7B. Notably, Qwen2.5-VL-7B serves a dual role: it is both a curation model and the base model for post-training. This on-policy curation is motivated by the insight that questions the training model itself can answer without visual input are precisely those that reinforce linguistic shortcuts during training. If Qwen2.5-VL-7B can solve a question through text alone, training on that question is unlikely to improve its visual grounding.

Results and analysis. Table 4 compares our three data curation strategies against the Qwen2.5-VL-7B base model and Video-R1 [20] (trained on the unfiltered 263K set). All models are post-trained with GRPO on Qwen2.5-VL-7B. Notably, all three VidGround variants improve over both baselines: Video-R1 degrades the base model (−2.6 and −2.2 Avg. Full at 16 and 32 frames, respectively), whereas every VG-curated variant yields clear gains, confirming that retaining only visually grounded questions is essential for effective video reasoning training. Among the three variants, VidGround-M1 achieves the highest average accuracy on the full benchmarks, slightly outperforming VidGround (+0.2 at 16 frames, +0.4 at 32 frames). On the VG question subset, VidGround-M1 also leads at 16 frames (45.9 vs. 45.2) while VidGround narrowly leads at 32 frames (47.6 vs. 47.5). However, VidGround-M2 (stricter curation) underperforms VidGround-M1 on both full and VG metrics despite applying stricter retention criteria, suggesting that overly aggressive curation reduces training data diversity without proportional quality gains. Importantly, VidGround with single-model curation remains highly competitive while being substantially simpler—requiring only one model evaluation pass rather than three—making it the most practical choice for large-scale data curation.

### B Cross-task generalization

To verify that visually grounded post-training does not degrade performance on non-video tasks, we evaluate VidGround on image QA benchmarks. As shown in Table 5, VidGround improves over the Qwen2.5-VL-7B base model on all three benchmarks (MME: 648.9 vs. 624.3, MMMU: 58.7 vs. 56.7, MMBench: 84.5 vs. 84.2). These results demonstrate that curating post-training data for visual grounding does not narrow the training distribution in ways that harm non-video capabilities.

### C Text-only answerability analysis across video understanding datasets

To provide a comprehensive view of linguistic biases in video understanding benchmarks, we extend our text-only evaluation in Table 1 to a broader range

- Table 4: Comparison of data curation strategies on video understanding benchmarks. VidGround uses single-model curation (GPT-5-mini, 181K samples). VidGround-M1 and VidGround-M2 use progressively stricter multi-model consensus curation (≥2 models agree). All models post-trained with GRPO on Qwen2.5-VL-7B. Avg. columns report mean accuracy on the full benchmarks (Full) and on VG question subsets—questions that require video to answer. Deltas show improvement over Qwen2.5-VL-7B: (+x) and (−x). Bold indicates best; highlighted rows are ours.

Frames Method VideoMME VideoMMMU MMVU

Avg. Full VG

- 16

Qwen2.5-VL-7B 58.2 45.0 60.5 54.6 42.9 Video-R1 (Full, 263K) 56.9 44.7 54.5 52.0 (−2.6) 41.7 (−1.2) VidGround (VG, 181K) 58.7 47.4 64.2 56.8 (+2.2) 45.2 (+2.3) VidGround-M1 (161K) 58.5 48.0 64.4 57.0 (+2.4) 45.9 (+3.0) VidGround-M2 (148K) 57.7 47.0 62.5 55.7 (+1.1) 43.8 (+0.9)

32

Qwen2.5-VL-7B 60.7 45.4 62.3 56.1 44.4 Video-R1 (Full, 263K) 60.2 45.4 56.2 53.9 (−2.2) 43.1 (−1.3)

VidGround (VG, 181K) 61.5 48.3 65.8 58.5 (+2.4) 47.6 (+3.2)

VidGround-M1 (161K) 62.1 50.9 63.7 58.9 (+2.8) 47.5 (+3.1) VidGround-M2 (148K) 61.4 50.4 62.8 58.2 (+2.1) 46.6 (+2.2)

of frontier models and report their performance on three video benchmarks: VideoMME [21], VideoMMMU [25], and MMVU [58]. Table 6 presents results for

- 17 frontier models spanning closed-source VLMs including GPT-4o [27], GPT5-mini and GPT-5 [37], the Gemini [16,23] family, and Claude [4,5]; open-source VLMs from the Qwen2.5-VL [7] family; and text-based large language models (LLMs) including DeepSeek-V3 and GPT-OSS [2] series. Notably, the LLMs evaluated have no visual capabilities and have never been trained on image or video data, making them well suited for assessing the extent to which video understanding benchmarks can be solved through linguistic reasoning alone.

- Table 5: Performance on image QA benchmarks. VidGround maintains or improves performance on non-video tasks compared to the Qwen2.5-VL-7B base model, demonstrating that visually grounded post-training does not harm crosstask generalization. Deltas show improvement over Qwen2.5-VL-7B: (+x). Bold indicates best; highlighted row is ours.

Method MME MMMU MMBench Qwen2.5-VL-7B 624.3 56.7 84.2 VidGround 648.9 (+24.6) 58.7 (+2.0) 84.5 (+0.3)

- Table 6: Extended text-only answerability across video understanding benchmarks for 17 frontier models spanning closed-source VLMs, open-source VLMs, and text-only LLMs. Each model receives only the question text and answer options—no video input. (+x) denotes improvement over random chance. All model families achieve accuracy >20 points above random, confirming pervasive linguistic bias. Bold indicates best.

Model VideoMME VideoMMMU MMVU Avg. Random Choice 25.0 9.8 19.8 18.2

Closed-source VLMs

- GPT-4o [27] 47.0 (+22.0) 38.6 (+28.8) 46.6 (+26.8) 44.1 (+25.9) GPT-5-mini [37] 45.2 (+20.2) 37.9 (+28.1) 53.3 (+33.5) 45.5 (+27.3)

- GPT-5 [37] 48.2 (+23.2) 41.0 (+31.2) 57.1 (+37.3) 48.8 (+30.6) Gemini-2.0-Flash [16] 43.1 (+18.1) 43.6 (+33.8) 53.4 (+33.6) 46.7 (+28.5) Gemini-2.5-Flash-Lite [16] 38.7 (+13.7) 39.6 (+29.8) 51.0 (+31.2) 43.1 (+24.9) Gemini-2.5-Flash [16] 49.6 (+24.6) 48.1 (+38.3) 55.5 (+35.7) 51.1 (+32.9)

- Gemini-2.5-Pro [16] 53.3 (+28.3) 52.7 (+42.9) 60.6 (+40.8) 55.5 (+37.3)

- Gemini-3.1-Pro [23] 58.2 (+33.2) 61.1 (+51.3) 63.4 (+43.6) 60.9 (+42.7) Claude-Sonnet-4.5 [4] 47.7 (+22.7) 44.3 (+34.5) 55.4 (+35.6) 49.1 (+30.9) Claude-Opus-4.6 [5] 51.3 (+26.3) 52.7 (+42.9) 61.0 (+41.2) 55.0 (+36.8)

Open-source VLMs

Qwen2.5-VL-7B [7] 39.2 (+14.2) 29.8 (+20.0) 48.6 (+28.8) 39.2 (+21.0) Qwen2.5-VL-32B [7] 41.3 (+16.3) 42.0 (+32.2) 54.7 (+34.9) 46.0 (+27.8) Qwen2.5-VL-72B [7] 44.3 (+19.3) 43.9 (+34.1) 56.0 (+36.2) 48.1 (+29.9)

Open-source LLMs

- DeepSeek-V3.1-Terminus [33] 39.0 (+14.0) 26.8 (+17.0) 52.6 (+32.8) 39.5 (+21.3)

- DeepSeek-V3.2-Exp [33] 39.6 (+14.6) 23.8 (+14.0) 53.9 (+34.1) 39.1 (+20.9) GPT-OSS-20B 40.4 (+15.4) 37.8 (+28.0) 50.6 (+30.8) 42.9 (+24.7) GPT-OSS-120B 45.0 (+20.0) 37.9 (+28.1) 53.3 (+33.5) 45.4 (+27.2)

#### C.1 Key findings

All VLMs and LLMs substantially exceed chance performance. Every evaluated VLM and LLM achieves accuracy more than 20 points above random chance, with the strongest model (Gemini-3.1-Pro [23]) reaching +42.7 points above random. This universal trend across diverse model architectures suggests that linguistic shortcuts are not artifacts of specific model designs. Instead, they stem from the pervasive presence of text-only answerable questions in both evaluation benchmarks and the training data and are further exacerbated by potential contamination of training data with benchmark content when developing these models. When VLMs are trained on datasets containing substantial proportions of linguistically biased or contaminated examples, they inevitably learn to exploit textual patterns and rely on their pretrained knowledge rather than developing robust visual grounding capabilities. This observation motivates our VidGround approach: by filtering training data to retain only visually-dependent questions, we can mitigate the linguistic biases that current training paradigms inadvertently amplify.

More than half of the benchmark questions require no video. Gemini-3.1-Pro achieves 58–64% accuracy across all three benchmarks without any visual information (VideoMME: 58.2%, VideoMMMU: 61.1%, MMVU: 63.4%), and even Gemini-2.5-Pro reaches approximately 50% or higher on each benchmark (53.3%, 52.7%, 60.6% respectively). GPT-5 also achieves 57.1% on MMVU without any visual input. These results show that half of the questions in these widely used video benchmarks can be answered correctly using text alone, casting doubt on their validity as measures of genuine video understanding.

Notably, VideoMME [21] reports that “Gemini-1.5-Pro achieves less than 15% accuracy in the text-only setup”; however, our experiments show Gemini-

- 1.5-Pro achieves 41.4% on VideoMME in text-only mode, and Gemini-2.5-Flash reaches 49.6%. We find that this discrepancy likely stems from the sensitivity of VLMs to prompts: with slight changes to the prompt, models that initially refuse to answer will readily produce responses in the text-only setting. Further details are provided in §D.4.

Performance gains from model scaling primarily reflect stronger language understanding. Consistent performance improvements are observed as model capacity increases within each model family: GPT-5 outperforms GPT-5-mini (+3.3 average points), Gemini-2.5-Flash surpasses Gemini-2.5-Flash-Lite (+8.0 points), and Qwen2.5-VL-72B exceeds both the 32B (+2.1 points) and 7B variants (+8.9 points). Critically, these gains persist in the text-only setting, indicating that VLM scaling benefits stem primarily from enhanced linguistic reasoning rather than improved visual grounding. This pattern holds for both closed-source and open-source models, revealing that apparent progress on video benchmarks largely reflects stronger language capabilities, not visually grounded video understanding.

Large language models rival or exceed vision-language models. Remarkably, textonly LLMs, which have never been exposed to visual data during training, achieve competitive or superior performance compared to VLMs. GPT-OSS120B (45.4% average) outperforms GPT-4o (44.1%), and DeepSeek-V3.2-Exp achieves 53.9% on MMVU, exceeding GPT-5-mini (53.3%) and rivaling several VLMs. Surprisingly, when comparing LLMs in text-only mode against VLMs with full video access, LLMs remain competitive or superior: GPT-OSS-120B reaches 45.0% on VideoMME without any visual input, matching or exceeding VLMs such as Video-LLaVA [32] (39.9%) and Chat-UniVi-V1.5 [28] (40.6%) with video input. These results demonstrate that current video benchmarks are solvable primarily through linguistic shortcuts, commonsense reasoning, and world knowledge rather than visual comprehension, as LLMs correctly answer approximately 39–45% of questions on average using language capabilities alone.

#### C.2 Analysis of linguistic biases in video benchmarks

To validate the prevalence of linguistic biases in video understanding benchmarks, we analyze representative examples from VideoMME [21] (Figure 7).

|1. Textual Shortcuts and Linguistic Cues|2. External (or Commonsense) Knowledge|
|---|---|
|Q: Where is the chased black spider-man hide at first?<br><br>A. Under a car.<br>B. Outside the building.<br>C. In a factory.<br>D. On the back of a spiderman who has four robotic arms.<br><br><br>[Figure 83]<br><br>Answer D implies the black-suited Spider-Man was stealthily hiding by clinging onto the back of another<br><br>VLM Spider-Man variant, using the element of surprise.<br><br>[Figure 84]<br><br>|[Figure 85]<br><br>[Figure 86]<br><br>[Figure 87]<br><br>[Figure 88]<br><br>[Figure 89]|
|---|
<br><br>[Figure 90]<br><br>[Figure 91]<br><br>[Figure 92]<br><br>[Figure 93]<br><br>[Figure 94]|[Figure 95]<br><br>[Figure 96]<br><br>Q: Which item is not in the makeup bag?<br><br>A. Freckle pen.<br>B. Sleep Master mask.<br>C. Sponge.<br>D. Lip balm.<br><br><br>[Figure 97]<br><br>The Sleep Master mask is not a makeup item and would not typically be found in a makeup bag. The other items are all common cosmetic or beauty tools that one would expect in a makeup bag.<br><br>VLM<br><br>[Figure 98]<br><br>|[Figure 99]<br><br>[Figure 100]<br><br>[Figure 101]<br><br>[Figure 102]<br><br>[Figure 103]|
|---|
<br><br>[Figure 104]<br><br>[Figure 105]<br><br>[Figure 106]<br><br>[Figure 107]<br><br>[Figure 108]|
|3. Inferential and Elimination Strategies|4. Imagined (Hallucinated) Video Content|
|[Figure 109]<br><br>[Figure 110]<br><br>Q: What was the purpose of using a hammer to hit the car in the video?<br><br>A. To show the hammer works well.<br>B. To show the solidity of the car.<br>C. To warn people not to hit cars with hammers.<br>D. To illustrate that a hammer is harder than a bullet.<br><br><br>[Figure 111]<br><br>The purpose of hitting the car with a hammer in the video was to demonstrate the car's durability or strength, emphasizing how solid or tough the car is.<br><br>VLM<br><br>[Figure 112]<br><br>|[Figure 113]<br><br>[Figure 114]<br><br>[Figure 115]<br><br>[Figure 116]|
|---|
<br><br>[Figure 117]<br><br>[Figure 118]<br><br>[Figure 119]<br><br>[Figure 120]<br><br>[Figure 121]<br><br>[Figure 122]<br><br>[Figure 123]<br><br>[Figure 124]|Q: Why is the man in the video angry?<br><br>A. Because the woman he's talking to on the phone called him "moon".<br>B. Because he doesn't like his traveling companions.<br>C. Because his grandmother called him "moon".<br>D. Because the woman he's talking to on the phone called him "moonpie".<br><br><br>[Figure 125]<br><br>In the video, the man becomes angry because the woman he's speaking with on the phone calls him "moonpie", a nickname he<br><br>VLM clearly dislikes.<br><br>[Figure 126]<br><br>|[Figure 127]<br><br>[Figure 128]<br><br>[Figure 129]<br><br>[Figure 130]|
|---|
<br><br>[Figure 131]<br><br>[Figure 132]<br><br>[Figure 133]<br><br>[Figure 134]|

- Fig.7: Additional examples of text-only answerable (TA) questions from VideoMME [21]. VLMs can exploit the same four categories of linguistic biases identified in Figure 2a to answer benchmark questions without visual grounding. VLM responses from GPT-4o [27].

These examples demonstrate how each of the four TA categories manifests in widely-used benchmarks, enabling models to bypass genuine video understanding.

Textual shortcuts and linguistic cues. In the Spider-Man question, options include generic locations (“under a car,” “in a factory”) alongside one highly specific choice: “on the back of a Spider-Man who has four robotic arms.” This unusual specificity enables models to identify the correct answer through linguistic pattern matching, rewarding text-based reasoning over multimodal grounding.

External knowledge. When asked “Which item is not in the makeup bag?” with options including freckle pen, Sleep Master mask, sponge, and lip balm, models can identify the Sleep Master mask as the outlier based solely on categorical knowledge, as it is not a typical cosmetic item. This requires no visual evidence of the bag’s actual contents.

Inferential and elimination strategies. For “What was the purpose of using a hammer to hit the car?”, options include implausible purposes alongside one reasonable explanation: demonstrating the car’s solidity. Models can identify the correct answer through elimination of implausible options rather than observing the video content.

Imagined (hallucinated) video content. When asked “Why is the man in the video angry?”, models might generate the correct answer “Because the woman called him moonpie” by hallucinating plausible conversation scenarios. Since the anger stems from dialogue rather than visual cues, such fortunate hallucinations do not reflect genuine video understanding.

These examples confirm that the four TA categories identified in training data (Figure 2a) also pervade evaluation benchmarks. The presence of such questions allows models to achieve high accuracy through linguistic shortcuts rather than genuine video reasoning.

#### C.3 Quantifying text-only answerability via Pass@10 sampling

- Fig.8: Pass@10 distribution for video questions in Video-R1-260K [20]. Pass@10 measures the fraction of questions where the model produces at least one correct answer across 10 independent text-only samples. Qwen2.5-VL-7B [7] achieves Pass@10 > 0 on 74.5% of video questions without video input.

Fig.9: Pass@10 distribution for image questions in Video-R1-260K [20]. Same setup as Figure 8. Qwen2.5-VL-7B [7] achieves Pass@10 > 0 on 33.4% of image questions without visual input.

To quantify text-only answerability in training data, we evaluate Qwen2.5VL-7B [7] on all 263,071 instances of Video-R1-260K [20] without providing visual input. For each question, we generate 10 independent responses and record how many are correct. We then compute Pass@10 as the percentage of questions where at least one of the 10 responses is correct. As shown in Figures 8, 9, and 10, 74.5% of video questions (i.e., questions that require video to answer), 33.4% of image questions (i.e., questions that require images to answer), and 51.6% overall achieve Pass@10 > 0.

The distribution patterns reveal different characteristics across modalities. Video questions (Figure 8) show 25.5% never answered correctly and 7.1% answered correctly in all 10 runs, with relatively uniform distribution across intermediate values. This suggests video questions are more susceptible to linguistic

- Fig.10: Overall Pass@10 distribution for Video-R1-260K [20]. Same setup as Figure 8. 51.6% of all questions achieve Pass@10 > 0 by Qwen2.5-VL-7B [7] without visual information.

shortcuts. In contrast, image questions (Figure 9) exhibit a skewed distribution with 66.6% never correct and only 1.0% always correct, indicating stronger visual dependency. The overall distribution (Figure 10) reflects these patterns with 48.4% never correct and 3.7% always correct.

We hypothesize that this disparity stems from differences in data construction costs. Video QA data is significantly more expensive and difficult to construct than image QA data, leading to widespread reliance on LLM-generated question-answer pairs. This practice introduces severe linguistic biases, as models trained on LLM-generated questions inherit the text-based reasoning patterns of their generators. For instance, LLaVA-Video-178K [56], a widely-used training dataset, employs GPT-4o to generate question-answer pairs. While such LLMgenerated data offers scalability, it systematically increases the proportion of TA (text-only answerable) questions, amplifying the linguistic bias problem revealed by our Pass@10 analysis.

#### C.4 Implications for video understanding research

The results across diverse model families in Table 6 and the text-only answerability analysis in §C.1, §C.2, and §C.3 establish several key implications.

Current benchmarks systematically overestimate progress. When 30–50% (or more) of performance can be attributed to linguistic shortcuts rather than visual understanding, these benchmarks exaggerate actual progress in video comprehension. Reported improvements may largely reflect advances in language modeling rather than genuine progress in multimodal video comprehension.

Visually grounded training data is essential. The strong performance of VLMs on video benchmarks and training datasets in the text-only setting demonstrates that current training data introduces linguistic biases that undermine

visual grounding. As shown in our ablation study (Table 3), training on the full dataset—which contains text-only answerable questions—leads to worse performance than training on visually grounded data alone, suggesting that linguistic shortcuts in training data undermine visual grounding. VidGround mitigates this issue and promotes more faithful visual grounding.

Periodic quality checks of video understanding benchmarks are required. Future video understanding benchmark development should systematically filter textonly answerable questions using frontier language models, as demonstrated in our VG (visually grounded) benchmark subsets. Regular re-evaluation is necessary as language models continue to improve, rendering previously valid questions increasingly vulnerable to linguistic shortcuts.

### D Implementation details

#### D.1 Training configuration

We post-train Qwen2.5-VL-7B-Instruct [7] using the GRPO objective described in §4.1. Training is conducted on 8× NVIDIA H100 GPUs for 700 steps. We uniformly sample 16 frames from each video during training. Training hyperparameters are provided in Table 7. We employ TRL [50] for GRPO implementation.

#### D.2 Evaluation setup

We evaluate all models on 4× NVIDIA L40S GPUs. Following standard practice, we uniformly sample 16, 32, and 64 frames per video when evaluating on video benchmarks to assess performance across different temporal resolutions.

#### D.3 Text-only answerability filtering

To construct our VG (visually grounded) training dataset, we filter Video-R1260K [20] by removing text-only answerable questions. We use GPT-5-mini [37] to evaluate each question-answer pair without visual content access, and questions answered correctly are classified as TA (text-only answerable) and removed. This filtering reduces the dataset from 263,071 to 181,710 samples (69.1% retention rate), removing 30.9% of linguistically biased examples.

#### D.4 Prompt template for text-only evaluation

For text-only evaluation, we append additional prompts to the questions and options from each benchmark. Our default prompt provides minimal instruction (Figure 11). However, earlier Gemini models (e.g., Gemini-1.5-Pro and Gemini-

- 2.0-Flash) often refuse to answer without video access. To obtain responses from these models, we use an enhanced prompt that explicitly prevents refusals (Figure 12).

- Table 7: Training configuration for VidGround post-training. We apply GRPO [44] to Qwen2.5-VL-7B-Instruct [7] for one epoch on the VidGround filtered dataset (181K visually grounded samples from Video-R1-260K [20]).

Hyperparameter Value

Training Setup

model name or path Qwen/Qwen2.5-VL-7B-Instruct max prompt length 16384 max completion length 768 per device train batch size 1 gradient accumulation steps 1 num train epochs 1 learning rate 1e-6 lr scheduler type cosine weight decay 0.01 max grad norm 5 bf16 true attn implementation flash attention 2 min pixels 3136 max pixels 501760

GRPO Settings

temporal true len control true β 0.04 temperature 1.0 num generations 8 ϵ 0.2

These different prompting results reveal a critical issue in Video-MME’s data quality assessment methodology. Video-MME [21] reports that “Gemini 1.5 Pro achieves less than 15% accuracy in the text-only setup, underscoring the robustness of the video content-based requirement.” However, this low accuracy results from model refusal rather than genuine inability to answer. With our enhanced prompt that prevents refusals, Gemini-1.5-Pro achieves substantially higher textonly accuracy (41.4%), demonstrating that the benchmark contains significantly more text-only answerable questions than Video-MME reported. This finding highlights that evaluation protocols must carefully account for model refusal behavior and sensitivity to prompts when validating benchmark quality.

### E Additional qualitative analysis

We provide additional qualitative comparison examples in addition to Figure 4 to further demonstrate the behavioral differences between Video-R1 [20] and VidGround. After manually inspecting reasoning chains across multiple instances,

|Enhanced prompt template for text-only evaluation|
|---|
| |
|[Figure 135]<br><br>Which tool is not necessary to make a snow globe?<br><br>A. Distilled water B. Scissors C. Glitter D. Super glue.<br><br>Based on the question and options provided, use your knowledge and common sense to determine the most likely correct answer. The video and subtitles are not available. Based only on the question and options, you must choose the most plausible answer. If unsure, make your best guess. Do not refuse to answer.|

|Prompt template for text-only evaluation|
|---|
| |
|[Figure 136]<br><br>Where is the main character in the video on holiday?<br><br>A. Sandcastle<br>B. Private mansion<br>C. Beachside<br>D. Music festival<br><br><br>Based on the question and options provided, use your knowledge and common sense to determine the most likely correct answer.|

- Fig.11: Default prompt template. Red text indicates the instructions added for text-only evaluation.

Fig.12: Enhanced prompt template. Red text indicates the instructions added to prevent refusals during textonly evaluation.

we identify a consistent pattern: VidGround systematically grounds its reasoning in video content, whereas Video-R1 primarily relies on text-based analysis of the question and options without referencing visual context.

- E.1 Visually grounded reasoning vs. text-based reasoning Across all examples (Figures 13–20), we make the following observations.

VidGround consistently anchors reasoning in video content. Most responses from VidGround begin by establishing what information the video provides. For example:

- – “Given the content of the video, the elements that are most prominently featured...” (Figure 13)
- – “Since the video is likely discussing the pelvis and the red dots are...” (Figure 16)
- – “Given the context of the video, which focuses on eosinophils...” (Figure 18)
- – “The video is about influence lines in structural engineering...” (Figure 20)

Such reasoning paths relying on video content ensure responses are grounded in observed visual evidence rather than linguistic priors.

Video-R1 relies more on text-based analysis and prior knowledge. In contrast, Video-R1 typically begins with generic problem-solving templates and proceeds to analyze the question and options through:

- – Textual shortcuts and linguistic cues (“Based on the information provided in the question and the table...” in Figure 17)
- – General domain knowledge (“Tau is a microtubule-associated protein...” in Figure 18)
- – Elimination strategies (“This option is too broad...” in Figure 20)

#### E.2 Beyond accuracy metrics: identical answers through different reasoning paths

Figure 20 provides evidence that our filtering approach changes model behavior fundamentally, not merely accuracy. Both models correctly answer the structural engineering question about influence lines, yet their reasoning processes differ substantially.

Video-R1 begins by analyzing the answer options rather than referencing the video, indicating that it relies heavily on textual cues. For example, its reasoning chain interprets each choice purely through linguistic analysis, such as: “A. unit moving load with unchanged direction – This option suggests that the load is a unit moving load with its direction remaining constant. This is a common assumption in structural analysis where the load is considered as a unit force that moves along a specific path...”. The model continues in this manner for all four options, using only textual reasoning and never engaging with the video content.

VidGround explicitly grounds its analysis in the video, for example: “The video is about influence lines in structural engineering, which are used to determine the maximum values of various quantities...The key point here is that the influence line is derived by considering a unit load moving along the structure.”. This response shows an understanding of the visual content before reaching the same conclusion.

This example indicates a limitation of accuracy-based evaluation. VLMs can produce correct answers through linguistic shortcuts by bypassing the understanding of the videos. Therefore, we argue that examining reasoning paths is important, as it exposes whether the model’s intermediate thinking process relies on visual grounding or simply exploits text-based shortcuts.

#### E.3 Summary

In summary, these qualitative comparisons (Figures 13–20) collectively demonstrate the advantages of VidGround. Our method not only improves model performance significantly, but also guides the model’s reasoning to be more visually grounded and less reliant on linguistic shortcuts. These examples provide solid evidence that VidGround effectively mitigates the linguistic biases introduced by current video post-training datasets. Our findings highlight that visually grounded training data is crucial for training models that truly leverage video content rather than relying on text-based shortcuts.

[Figure 150]

[Figure 151]

[Figure 152]

###### .. ...

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

Question

Which of the following elements of art are most prominently featured in this artwork?

A. Line B. Shape C. Form D. Value E. Color F. Texture G. Balance H. Space

I. Contrast. J. Proportion

|Video-R1|Ours|
|---|---|
|[Figure 165]<br><br>[Figure 166]<br><br>|Given the analysis, the most prominently featured elements are line and shape. However, since the question asks for the most prominent, and considering the variety of lines and shapes, the answer leans towards:<br><br>B. Shape|
|---|
<br><br>Relies on text<br><br>The artwork in the video showcases various elements of art, but let’s break down which one stands out the most.<br><br>1. Line: There are numerous lines used in the drawings, especially in the abstract and geometric shapes. The lines are varied in direction and style, contributing significantly to the overall composition.<br>2. Shape: Shapes are also prominent, with both geometric and organic forms present. The use of shapes to create patterns and structures is evident.<br>3. Form: While there are some three-dimensional elements like the cube and the sphere, they are not the main focus. The emphasis is more on twodimensional elements rather than creating a sense of volume through form.<br>4. Value: The use of value (light and shadow) is minimal compared to other elements. The colors are bright and flat, without much shading.<br>5. Color: Color is present, but it is not the most dominant element. The colors are used to fill shapes and add contrast, but they do not create a strong color palette.<br>6. Texture: There is no significant texture in the artwork. The surfaces are smooth and flat.<br>7. Balance: The balance is somewhat achieved, but it is not the most striking feature. The arrangement of elements is somewhat balanced, but it is not the primary focus.<br>8. Space: The concept of space is present, but it is not the most prominent. The artwork does not explore depth or negative space extensively.<br>9. Contrast: There is some contrast, but it is not the most dominant element. The colors and lines provide some contrast, but it is not the main focus.<br>10. Proportion: The proportions are somewhat consistent, but they are not the most striking feature.<br><br><br>Final answer: B. Shape||Given the content of the video, the elements that are most prominently featured are likely to be those that are discussed and demonstrated in the video. Based on the video content, the most prominent elements are: Line, Shape, Color, Texture, Space|
|---|
<br><br>Correct visual reference<br><br>To determine which elements of art are most prominently featured in the artwork, let’s analyze the video content step by step:<br><br>1. Line: The video shows a hand drawing various types of lines such as horizontal, diagonal, curved, and dotted. This suggests that line is a significant element in the artwork.<br>2. Shape: The video also shows the drawing of diﬀerent shapes like triangles, circles, and squares, indicating that shape is another prominent element.<br>3. Form: The video briefly shows a 3D form, which suggests that form is also an element being discussed.<br>4. Value: The video mentions value, which refers to the lightness or darkness of a color, and shows examples of how value can be used in art.<br>5. Color: The video includes a color wheel and discusses primary colors, indicating that color is a key element.<br>6. Texture: The video briefly shows a textured surface, suggesting that texture is also an element being discussed.<br>7. Balance: The video does not focus on balance as a primary element.<br>8. Space: The video mentions space, including positive and negative space, indicating that space is a significant element.<br><br><br>However, since the question asks for the most prominently featured elements, and considering the video content, the most prominent elements are likely to be: A. Line, B. Shape, E. Color, F. Texture, H. Space<br><br>The most comprehensive answer would be: A. Line, B. Shape, E. Color, F. Texture, H. Space<br><br>|But if we need to choose the most prominent one based on the video content, the answer would be:|
|---|
<br><br>A. Line<br><br>So, the correct answer is:<br><br>A. Line|

: Over-reliance on text : Correct visual reference

| |
|---|

| |
|---|

- Fig.13: Art elements analysis question comparing VidGround (trained on visually grounded data) with Video-R1 [20] (trained on the full, unfiltered VideoR1-260K dataset). VidGround references the art elements most prominently featured in the video, while Video-R1 analyzes artistic concepts abstractly without considering the video content. Green highlights indicate correct reasoning grounded in video content; red highlights indicate text-based reasoning without visual grounding.

[Figure 180]

[Figure 181]

[Figure 182]

###### .. ...

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

Question

Consider a neuron where the action potential has been successfully generated but fails to reach the axon terminals. Based on the video, which of the following could directly explain the issue?

- 1. The structure introduced at 1:18 is damaged.
- 2. The structure introduced at 0:32 is damaged.
- 3. The introduced at 1:10 is severed or damaged.
- 4. The dendrites did not receive enough neurotransmitter signals to properly initiate the process.
- 5. The soma failed to integrate the signals from the dendrites correctly.

A. 1, 2, and 3 are correct B. 2 and 4 are correct C. 1 and 3 are correct D. 3 and 5 are correct E. Only 1 is correct F. 2 and 5 are correct G. "Only 4 is correct H. 1 and 4 are correct I. 1, 3, and 5 are correct J. 2, 3, and 4 are correct

|Video-R1|Ours|
|---|---|
|[Figure 195]<br><br>[Figure 196]<br><br>|Let's break down the problem step by step. First, we need to understand the components of a neuron and their roles:<br><br>- Dendrites: Receive signals from other neurons.<br>- Soma (Cell Body): Processes the signals received by the dendrites.<br>- Axon: Transmits the signal to the axon terminals.<br><br><br>The question asks about the scenario where an action potential is generated but fails to reach the axon terminals. This suggests that there might be an issue with the transmission of the signal from the soma to the axon terminals.|
|---|
<br><br>Now, let's consider each option: Relies on text<br><br>1. The structure introduced at 1:18 is damaged. This refers to the axon hillock, which is a part of the axon. If the axon hillock is damaged, it could aﬀect the transmission of the action potential.<br>2. The structure introduced at 0:32 is damaged. This refers to the dendrites. If the dendrites are damaged, they might not receive enough neurotransmitter signals to properly initiate the process.<br>3. The introduced at 1:10 is severed or damaged. This refers to the axon. If the axon is severed or damaged, it would prevent the action potential from reaching the axon terminals.<br>4. The dendrites did not receive enough neurotransmitter signals to properly initiate the process. This is a possibility, but it is more related to the dendrites' function rather than the axon's function.<br>5. The soma failed to integrate the signals from the dendrites correctly. This is also a possibility, but it is more related to the soma's function rather than the axon's function.<br><br><br>From this analysis, the most relevant options are 1, 3, and 5. Therefore, the correct answer is I. Final answer: I||The video explains the structure of a neuron and its components, including the dendrites, soma (cell body), axon hillock, axon, myelin, and axon terminals. The action potential is generated in the soma and travels down the axon to the axon terminals. If the action potential fails to reach the axon terminals, it could be due to damage to the axon or the axon terminals, which are the structures at the end of the axon where neurotransmitters are released.|
|---|
<br><br>Correct visual reference Given the options:<br><br>1. The structure introduced at 1:18 is damaged. This refers to the axon hillock, which is the region where the action potential is generated.<br>2. The structure introduced at 0:32 is damaged. This refers to the soma (cell body), which is the main body of the neuron.<br>3. The introduced at 1:10 is severed or damaged. This refers to the axon, which is the long, thin projection of the neuron that transmits signals.<br>4. The dendrites did not receive enough neurotransmitter signals to properly initiate the process. This is not directly related to the failure of the action potential to reach the axon terminals.<br>5. The soma failed to integrate the signals from the dendrites correctly. This is not directly related to the failure of the action potential to reach the axon terminals.<br><br><br>The correct answer is: C. 1 and 3 are correct<br><br>The action potential fails to reach the axon terminals if the axon hillock<br><br>(1) is damaged or if the axon (3) is severed or damaged.|

: Over-reliance on text : Correct visual reference

| |
|---|

| |
|---|

##### Fig.14: Psychology question. Same comparison setup as Figure 13. VidGround grounds its reasoning in the neuron and its components introduced by the video, while Video-R1 [20] analyzes through general biology knowledge.

[Figure 210]

[Figure 211]

[Figure 212]

###### .. ...

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

Question

For the question in 16:03, if we change the increasing to decreasing, what changes will cause this decrease?

A. Decreased airway resistance and increased respiratory rate B. Increased tidal volume (VT) and increased alveolar ventilation (VA) C. Decreased dead space ventilation and increased minute ventilation D. Decreased metabolic rate and increased respiratory muscle strength E. Increased functional residual capacity (FRC) and decreased airway closure F. Decreased carbon dioxide production and increased alveolar-capillary diﬀusion G. Increased diaphragmatic contraction and decreased airway obstruction H. Decreased mucus production and increased mucociliary clearance

I. Increased oxygen delivery to tissues and decreased anaerobic metabolism J. Decreased bronchial wall thickening and increased elastic recoil of the lungs

|Video-R1|Ours|
|---|---|
|[Figure 225]<br><br>[Figure 226]<br><br>|If we change the increasing to decreasing for PaCO2, it implies that there is a reduction in the partial pressure of carbon dioxide in the arterial blood. This would typically occur due to factors that enhance the removal or reduce the production of CO₂.|
|---|
<br><br>Let’s analyze each option: Relies on text<br><br>A. Decreased airway resistance and increased respiratory rate - This could potentially increase PaCO2 by increasing the rate of breathing but not necessarily reducing it.<br>B. Increased tidal volume (VT) and increased alveolar ventilation (VA) - This would likely increase PaCO2 as more air is being ventilated into the lungs.<br>C. Decreased dead space ventilation and increased minute ventilation - This could potentially reduce PaCO2 by increasing the amount of fresh air entering the alveoli.<br>D. Decreased metabolic rate and increased respiratory muscle strength - This might reduce PaCO2 by lowering the production of CO₂ and improving the efficiency of breathing.<br>E. Increased functional residual capacity (FRC) and decreased airway closure - This could reduce PaCO2 by providing more space for gas exchange and reducing the likelihood of airway obstruction.<br>F. Decreased carbon dioxide production and increased alveolar-capillary diﬀusion This would directly reduce PaCO2 by lowering the production of CO₂ and improving its removal from the blood.<br>G. Increased diaphragmatic contraction and decreased airway obstruction - This could reduce PaCO2 by improving the efficiency of breathing and reducing obstruction.<br>H. Decreased mucus production and increased mucociliary clearance - This might have a minimal eﬀect on PaCO2.<br>I. Increased oxygen delivery to tissues and decreased anaerobic metabolism - This could reduce PaCO2 by improving oxygen utilization and reducing anaerobic respiration.<br>J. Decreased bronchial wall thickening and increased elastic recoil of the lungs - This could reduce PaCO2 by improving the elasticity and function of the lungs.<br><br><br>The most comprehensive and direct factor leading to a decrease in PaCO2 is F, which directly addresses the production and removal of CO₂.<br><br>Therefore, the correct answer is F.||The question in 16:03 is asking what changes will cause a decrease in PaCO2, which is the partial pressure of carbon dioxide in the arterial blood. PaCO2 is inversely related to ventilation and directly related to the rate of carbon dioxide production. Therefore, a decrease in PaCO2 would be caused by an increase in ventilation or a decrease in carbon dioxide production.|
|---|
<br><br>Let’s analyze the options: Correct visual reference<br><br>A. Decreased airway resistance and increased respiratory rate - This would increase ventilation, leading to a decrease in PaCO2.<br>B. Increased tidal volume (VT) and increased alveolar ventilation (VA) - This would increase ventilation, leading to a decrease in PaCO2.<br>C. Decreased dead space ventilation and increased minute ventilation - This would increase ventilation, leading to a decrease in PaCO2.<br>D. Decreased metabolic rate and increased respiratory muscle strength - This would decrease carbon dioxide production, leading to a decrease in PaCO2.<br>E. Increased functional residual capacity (FRC) and decreased airway closure - This would increase ventilation, leading to a decrease in PaCO2.<br>F. Decreased carbon dioxide production and increased alveolar-capillary diﬀusion - This would decrease carbon dioxide production, leading to a decrease in PaCO2.<br>G. Increased diaphragmatic contraction and decreased airway obstruction - This would increase ventilation, leading to a decrease in PaCO2.<br>H. Decreased mucus production and increased mucociliary clearance - This would not directly aﬀect PaCO2.<br>I. Increased oxygen delivery to tissues and decreased anaerobic metabolism - This would not directly aﬀect PaCO2.<br>J. Decreased bronchial wall thickening and increased elastic recoil of the lungs - This would increase ventilation, leading to a decrease in PaCO2.<br><br><br>The most comprehensive and direct options that would cause a decrease in PaCO2 are:<br><br>• B. Increased tidal volume (VT) and increased alveolar ventilation (VA)<br>• D. Decreased metabolic rate and increased respiratory muscle strength<br>• F. Decreased carbon dioxide production and increased alveolar-capillary diﬀusion<br>• G. Increased diaphragmatic contraction and decreased airway obstruction<br>• J. Decreased bronchial wall thickening and increased elastic recoil of the lungs<br><br><br>The most accurate and direct answer is:<br><br>B. Increased tidal volume (VT) and increased alveolar ventilation (VA)<br><br>This is because it directly increases the rate of ventilation, which is the primary determinant of PaCO2.|

: Over-reliance on text : Correct visual reference

| |
|---|

| |
|---|

- Fig.15: Medical science question. Same comparison setup as Figure 13. Vid-

Ground locates the relevant content in the video about PaCO2, while VideoR1 [20] evaluates options using linguistic priors.

[Figure 240]

[Figure 241]

[Figure 242]

###### .. ...

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

Question

Can you identify the abnormality on this plain film of the pelvis? <image 1>

A. Bone cyst B. Acute hip fracture C. Osteoarthritis D. Surgical hardware E. Resection of the pubic symphysis F. Bone tumor

G. Hip dislocation H. Osteoporosis I. Paget's disease J. Bone infection

|Video-R1|Ours|
|---|---|
|[Figure 255]<br><br>[Figure 256]<br><br>|Based on the provided options, let’s analyze the image step by step:|
|---|
<br><br>Relies on text<br><br>1. Bone Cyst: A bone cyst typically appears as a well-defined, fluid-filled lesion in the bone. There is no clear evidence of such a lesion in the image.<br>2. Acute Hip Fracture: An acute hip fracture would show a clear break or discontinuity in the femoral neck or head. While there might be some irregularities, it’s not immediately obvious that this is an acute fracture.<br>3. Osteoarthritis: Osteoarthritis usually shows degenerative changes with narrowing of joint spaces, osteophytes, and sclerosis. The image does not clearly depict these features.<br>4. Surgical Hardware: The image shows a metallic object within the pelvic region, which could be surgical hardware used for fixation or reconstruction.<br>5. Resection of the Pubic Symphysis: This would involve a significant change in the alignment and structure of the pubic bones, which is not evident here.<br>6. Bone Tumor: A bone tumor would show a mass eﬀect with possible destruction of the bone. There is no clear evidence of a tumor in the image.<br>7. Hip Dislocation: While there might be some displacement, the image does not clearly show the typical signs of a dislocated hip.<br>8. Osteoporosis: Osteoporosis typically shows decreased bone density and increased risk of fractures. The image does not clearly depict these features.<br>9. Paget’s Disease: Paget’s disease can show irregular bone density and deformities. The image does not clearly depict these features.<br>10. Bone Infection: A bone infection would show signs of inflammation and possibly abscesses. There is no clear evidence of infection in the image.<br><br><br>Given the presence of a metallic object within the pelvic region, the most likely abnormality is surgical hardware.<br><br>Therefore, the correct answer is D. Surgical hardware.||The image provided is a plain film of the pelvis, and it shows a 3D rendered pelvis with red dots indicating specific areas of interest. The red dots are likely highlighting bony landmarks or areas of concern. Given the context of the video, which seems to be educational and focused on medical imaging, the most likely abnormality to be identified is a specific bony structure or a condition related to the pelvis.|
|---|
<br><br>Correct visual reference<br><br>However, without a clear visual of the red dots and their specific locations, it is challenging to definitively identify the abnormality. The options provided include various conditions such as bone cyst, acute hip fracture, osteoarthritis, surgical hardware, resection of the pubic symphysis, bone tumor, hip dislocation, osteoporosis, Paget’s disease, and bone infection.<br><br>|Since the video is likely discussing the pelvis and the red dots are pointing to specific bony landmarks, the most probable answer would be related to a bony abnormality. Among the options, the most common and relevant one would be:|
|---|
<br><br>E. Resection of the pubic symphysis<br><br>This is because the pubic symphysis is a common area of interest in pelvic imaging, and resection of this area would be a significant abnormality that would be highlighted in a medical educational video.<br><br>Therefore, the most likely answer is: E. Resection of the pubic symphysis|

: Over-reliance on text : Correct visual reference

| |
|---|

| |
|---|

##### Fig.16: Clinical medical imaging interpretation question. Same comparison setup as Figure 13. This instance includes a reference image as part of the question prompt. VidGround refers to the video content about the pelvis and red dots pointing to specific bony landmarks, while Video-R1 [20] relies on prior knowledge to eliminate options without referencing the video.

[Figure 270]

[Figure 271]

[Figure 272]

###### .. ...

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

[Figure 283]

[Figure 284]

Question

The results of studies observing the relationship between smoking and lung cancer are shown in the table below, what can be calculated from these data? <image 1>

- A. PAR
- B. AR
- C. RR
- D. OR

|Video-R1|Ours|
|---|---|
|[Figure 285]<br><br>[Figure 286]<br><br>|Based on the information provided in the question and the table, we need to determine which measure of association can be calculated using the given data.<br><br>First, let’s understand the variables:<br><br>• The table shows the number of cases (lung cancer patients) and controls (nonlung cancer patients) for smokers and non-smokers.<br>• The variables are labeled as “Exposed” (smokers) and “Unexposed” (nonsmokers).<br><br><br>From the table, we can see the following counts:<br><br>• Smokers with lung cancer: 150<br>• Non-smokers with lung cancer: 30<br>• Total smokers: 250<br>• Total non-smokers: 250<br>|
|---|
<br><br>Now, let’s consider the options: Relies on text<br><br>A. PAR - Population Attributable Risk: This is not directly calculable from the table without additional population data.<br>B. AR - Attributable Risk: This is the proportion of lung cancer cases that can be attributed to smoking. It can be calculated using the formula:<br>C. RR - Relative Risk: This is the ratio of the risk of lung cancer among smokers to the risk among non-smokers. It can be calculated using the formula:<br>D. OR - Odds Ratio: This is the ratio of the odds of having lung cancer among smokers to the odds of having lung cancer among non-smokers. It can be calculated using the formula:<br><br><br>Given the data, the most straightforward measure that can be calculated is the Relative Risk (RR), as it only requires the counts of cases and controls for smokers and non-smokers.<br><br>Therefore, the correct answer is C.<br><br>Final answer: C<br><br>AR = (Exposedcases − Unexposedcases)/Exposedcases.<br><br>RR = (Exposedcases/Totalexposed)/(Unexposedcases/Totalunexposed).<br><br>OR = (Exposedcases/Unexposedcases)/(Unexposedcases/Exposedcases).||Correct visual reference<br><br>To determine what can be calculated from the given table, let’s first understand the data and the measures of association mentioned in the video:<br><br>1. Odds Ratio (OR): This is a measure of the association between an exposure (smoking) and an outcome (lung cancer). It is calculated using the formula:<br><br>where:<br><br>• A is the number of cases in the exposed group.<br>• B is the number of controls in the exposed group.<br>• C is the number of cases in the unexposed group.<br>• D is the number of controls in the unexposed group.<br><br><br>OR =<br><br>A × D<br><br>B × C<br>|
|---|
<br><br>2. Attributable Risk (AR): This is the diﬀerence in the risk of the outcome between the exposed and unexposed groups. It is calculated as:<br><br>where:<br><br>• is the proportion of the exposed group that has the outcome.<br>• is the proportion of the unexposed group that has the outcome.<br><br><br>3. Population Attributable Risk (PAR): This is the proportion of the total number of cases in the population that can be attributed to the exposure. It is calculated as:<br><br><br>where:<br><br>• is the proportion of the population that is exposed. Given the table provided, we can see the following data:<br><br>• A = 150 (smokers with lung cancer)<br>• B = 50 (smokers without lung cancer)<br>• C = 100 (non-smokers with lung cancer)<br>• D = 200 (non-smokers without lung cancer)<br><br><br>From this data, we can calculate the Odds Ratio (OR) as follows:<br><br>Since the table does not provide the proportion of the exposed and unexposed groups, we cannot calculate the AR or PAR directly from the given data. However, the OR can be calculated.<br><br>Therefore, the correct answer is: D. OR<br><br>AR = (Pexposed − Punexposed) × Pexposed<br><br>Pexposed Punexposed<br><br>PAR =<br><br>OR × Pexposed 1 + (OR − 1) × Pexposed<br><br>Pexposed<br><br>OR =<br><br>150 × 200 50 × 100<br><br>=<br><br>30000 5000<br><br>= 6|

: Over-reliance on text : Correct visual reference

| |
|---|

| |
|---|

##### Fig.17: Public health question. Same comparison setup as Figure 13. This instance includes a reference image as part of the question prompt. VidGround grounds its analysis in video-presented concepts of OR (Odds Ratio), PAR (Population Attributable Risk), and AR (Attributable Risk), while Video-R1 [20] analyzes the question and options using prior knowledge.

[Figure 300]

[Figure 301]

[Figure 302]

###### .. ...

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

Question

The eosinophilic intraneuronal inclusion shown in the image will be strongly immunostained with <image 1>

- A. tau
- B. alpha synuclein
- C. ubiquitin
- D. both B and C

|Video-R1|Ours|
|---|---|
|[Figure 315]<br><br>[Figure 316]<br><br>|Given the context of the image and the typical association of eosinophilic inclusions with alpha synuclein, the most likely protein to strongly immunostain the inclusion is alpha synuclein.|
|---|
<br><br>Relies on text<br><br>To determine which protein will strongly immunostain the eosinophilic intraneuronal inclusion, we need to consider the characteristics of the inclusion and the proteins listed.<br><br>1. tau: Tau is a microtubule-associated protein that is often associated with neurodegenerative diseases such as Alzheimer’s disease. It is not typically found in eosinophilic intraneuronal inclusions.<br>2. alpha synuclein: Alpha synuclein is a protein involved in Parkinson’s disease and other neurodegenerative disorders. It can form aggregates in neurons, similar to the eosinophilic inclusions seen in some neurological conditions.<br>3. ubiquitin: Ubiquitin is a small protein that is involved in the degradation of damaged proteins. It can also form aggregates in neurons, similar to the eosinophilic inclusions.<br>4. both B and C: This option suggests that both alpha synuclein and ubiquitin will strongly immunostain the inclusion. While both proteins can form aggregates in neurons, they are not typically associated with eosinophilic inclusions in the same way that alpha synuclein is.<br><br><br>Therefore, the correct answer is:<br><br>B. alpha synuclein<br><br>However, since the question asks for the protein that will strongly immunostain the inclusion, and considering the typical association, the most accurate answer is:<br><br>D. both B and C<br><br>But since the question asks for the most specific and commonly associated protein, the answer is:<br><br>B. alpha synuclein||Given the context of the video, which focuses on eosinophils and their role in immunity, the most relevant staining markers for eosinophils are likely to be ubiquitin and alpha synuclein, as these are commonly used to stain intraneuronal inclusions in various diseases.|
|---|
<br><br>Correct visual reference<br><br>The image you are referring to is not provided in the text, but based on the context of the video content, which discusses the eosinophil and its role in immunity, the question is likely related to the staining of intraneuronal inclusions in eosinophils.<br><br>Eosinophils are known to contain granules that can be stained with certain markers. The options provided are:<br><br>A. tau<br>B. alpha synuclein<br>C. ubiquitin<br>D. both B and C<br><br><br>Therefore, the correct answer is: D. both B and C|

: Over-reliance on text : Correct visual reference

| |
|---|

| |
|---|

##### Fig.18: Diagnostics and laboratory medicine question. Same comparison setup as Figure 13. This instance includes a reference image as part of the question prompt. VidGround anchors its analysis in video content about eosinophils and their role in immunity, while Video-R1 [20] applies general knowledge without referencing the video.

[Figure 330]

[Figure 331]

[Figure 332]

###### .. ...

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

[Figure 343]

[Figure 344]

Question

State which molecule in each pair has the higher boiling point:

HF HI O2 CH3Cl H2O NaCl He Ar CO2 SO2 NH3 PH3 C6H6 CCl4 N2 F2

1. vs 2. vs 3. vs 4. vs 5. vs 6. vs 7. (Benzene) vs (Carbon tetrachloride) 8. vs

HF,O2,NaCl, Ar,CO2,NH3,CCl4,F2 HI,CH3Cl,NaCl,He,SO2,NH3,C6H6,N2 HF,CH3Cl,H2O, Ar,SO2,PH3,CCl4,F2 HF,O2,H2O,He,CO2,NH3,C6H6,N2 HF,CH3Cl,NaCl, Ar,SO2,NH3,C6H6,F2 HI,CH3Cl,NaCl, Ar,SO2,PH3,CCl4,N2 HF,CH3Cl,H2O, Ar,CO2,NH3,C6H6,F2 HI,O2,NaCl,He,SO2,NH3,CCl4,F2

A. B. C. D. E. F.

G. H. I. J.

HF,CH3Cl,H2O, Ar,SO2,NH3,C6H6,F2 HF,CH3Cl,NaCl, Ar,CO2,PH3,C6H6,F2

|Video-R1|Ours|
|---|---|
|[Figure 345]<br><br>[Figure 346]<br><br>|3. vs : has strong hydrogen bonding, while has ionic bonding. Therefore, has a higher boiling point.<br><br>H2O NaCl H2O NaCl H2O|
|---|
<br><br>Relies on text<br><br>The correct answer is:<br><br>A.<br><br>Let’s break down the reasoning for each pair:<br><br>1. vs : Fluorine is more electronegative than hydrogen, so has a stronger dipole than . Therefore, has a higher boiling point.<br>2. vs : has a larger molecular weight and more<br><br><br>intermolecular forces compared to , so has a higher boiling point.<br><br>4. vs : has a larger atom size and thus better van der Waals interactions, so has a higher boiling point.<br>5. vs : has a stronger dipole-dipole interaction due to its higher electronegativity diﬀerence, so has a higher boiling point.<br>6. vs : has a stronger dipole-dipole interaction due to its higher electronegativity diﬀerence, so has a higher boiling point.<br>7. vs : has weaker intermolecular forces compared to , so has a higher boiling point.<br>8. vs : has a stronger dipole-dipole interaction due to its higher electronegativity diﬀerence, so has a higher boiling point.<br><br><br>Thus, the correct answer is A.<br><br>HF,O2,NaCl, Ar,CO2,NH3,CCl4,F2<br><br>HF HI HF<br><br>HI HF O2 CH3Cl CH3Cl<br><br>O2 CH3Cl<br><br>He Ar Ar Ar CO2 SO2 SO2<br><br>SO2 NH3 PH3 NH3<br><br>NH3 C6H6 CCl4 C6H6<br><br>CCl4 CCl4 N2 F2 F2<br><br>F2||strength of these forces, which include: 1. Ion-ion (ion-ion interaction), 2. Dipole-dipole interaction, 3. Hydrogen bonding, 4. London dispersion forces (van der Waals forces)|
|---|
<br><br>Correct visual reference<br><br>To determine which molecule in each pair has the higher boiling point, we need to consider the intermolecular forces present in each substance. The boiling point is influenced by the<br><br>Let’s analyze each pair:<br><br>1. vs<br><br>• has hydrogen bonding, which is stronger than the dipole-dipole interaction in .<br>• has the higher boiling point.<br><br><br>2. vs<br><br>• has only London dispersion forces.<br>• has dipole-dipole interaction.<br>• has the higher boiling point.<br><br><br>3. vs<br><br>• has hydrogen bonding.<br>• has ion-ion interaction.<br>• has the higher boiling point.<br><br><br>4. vs<br><br>• Both are noble gases, so they only have London dispersion forces.<br>• Since has a larger atomic radius, it has stronger London dispersion forces.<br>• has the higher boiling point.<br><br><br>5. vs<br><br>• has only London dispersion forces.<br>• has dipole-dipole interaction.<br>• has the higher boiling point.<br><br><br>6. vs<br><br>• has hydrogen bonding.<br>• has only London dispersion forces.<br>• has the higher boiling point.<br><br><br>7. (Benzene) vs (Carbon tetrachloride)<br><br>• has hydrogen bonding.<br>• has only London dispersion forces.<br>• has the higher boiling point.<br><br><br>8. vs<br><br><br>• Both have only London dispersion forces.<br>• Since has a higher boiling point due to the stronger London dispersion forces.<br>• has the higher boiling point.<br><br><br>The correct answer is: E.<br><br>HF HI HF HI HF<br><br>O2 CH3Cl O2 CH3Cl CH3Cl<br><br>H2O NaCl H2O NaCl NaCl<br><br>He Ar<br><br>Ar Ar<br><br>CO2 SO2 CO2 SO2 SO2 NH3 PH3 NH3 PH3 NH3<br><br>C6H6 CCl4<br><br>C6H6 CCl4 C6H6<br><br>N2 F2<br><br>F2 F2|

: Over-reliance on text : Correct visual reference

| |
|---|

| |
|---|

##### Fig.19: Chemistry question. Same comparison setup as Figure 13. VidGround references video explanations of intermolecular forces and applies them to each molecular pair, while Video-R1 [20] directly analyzes the options using prior knowledge.

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

###### .. ...

[Figure 363]

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

[Figure 373]

[Figure 374]

Question

When drawing the influence line of any quantity value, assume that the load is ()<image 1>

A. unit moving load with unchanged direction B. moving load C. dynamic load D. movable load

|Video-R1|Ours|
|---|---|
|[Figure 375]<br><br>[Figure 376]<br><br>|To determine the correct assumption when drawing the influence line of any quantity value, let’s break down the options provided:|
|---|
<br><br>Relies on text<br><br>A. unit moving load with unchanged direction - This option suggests that the load is a unit moving load with its direction remaining constant. This is a common assumption in structural analysis where the load is considered as a unit force that moves along a specific path.<br>B. moving load - This option is too broad. While it is true that the load is moving, the specific nature of the movement (unit or not) and the direction change need to be considered.<br>C. dynamic load - This option implies that the load has a time-dependent behavior, which is not necessarily the case when drawing an influence line. The influence line is typically used for static loads.<br>D. movable load - This option is similar to option B. It is too general and does not specify the nature of the movement or the direction change.<br><br><br>Given the context of structural analysis and the typical assumptions made when drawing influence lines, the most accurate and commonly used assumption is that the load is a unit moving load with unchanged direction.<br><br>Therefore, the correct choice is A.<br><br>Final answer: A<br><br>Let’s verify the reasoning one more time. The influence line is often drawn under the assumption of a unit moving load with a fixed direction, which aligns with option A. This assumption simplifies the analysis and allows for the determination of the maximum values of the quantity of interest.<br><br>Final answer: A||To determine the correct answer, let’s analyze the context of the video and the question.<br><br>The video is about influence lines in structural engineering, which are used to determine the maximum values of various quantities (like shear force, bending moment, etc.) under diﬀerent loading conditions. The influence line is a graphical representation of the variation of a quantity with position of a unit load.|
|---|
<br><br>Correct visual reference<br><br>The key point here is that the influence line is derived by considering a unit load moving along the structure. This unit load is typically assumed to be a unit moving load with an unchanged direction, as it helps in understanding the maximum values of the quantity of interest at diﬀerent points along the structure.<br><br>Given this context, the correct answer is:<br><br>A. unit moving load with unchanged direction<br><br>This is because the influence line is constructed by considering a unit load moving along the structure, and the direction of the load is typically assumed to remain constant to simplify the analysis.|

: Over-reliance on text : Correct visual reference

| |
|---|

| |
|---|

- Fig.20: Structural engineering question. Same comparison setup as Figure 13. This instance includes a reference image as part of the question prompt. Both models reach the correct answer (A. unit moving load with unchanged direction) through different reasoning paths. VidGround references video content about influence lines in structural engineering and derives the answer from the demonstrated concept, while Video-R1 [20] eliminates options through prior knowledge without referencing the video. This example demonstrates that correct answers can emerge from either genuine video understanding or linguistic shortcuts, highlighting the importance of evaluating reasoning paths beyond accuracy metrics alone.

