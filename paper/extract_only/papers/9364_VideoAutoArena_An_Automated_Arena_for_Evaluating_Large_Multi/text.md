# arXiv:2411.13281v2[cs.CV]23Mar2025

## VideoAutoArena: An Automated Arena for Evaluating Large Multimodal Models in Video Analysis through User Simulation

Ziyang Luo1,2, Haoning Wu4, Dongxu Li5, Jing Ma2, Mohan Kankanhalli3, Junnan Li1 1Salesforce AI Research, 2Hong Kong Baptist University, 3National University of Singapore 4Nanyang Technological University, 5The Australian National University

[Figure 1]

VideoAutoArena ELO Rating

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

###### Human Annotation User Simulation (Role Play)

[Figure 7]

A travel enthusiast with a passion for culinary experiences and cultural exploration, particularly drawn to the food and landmarks, making this video especially engaging for their interest in Singapore's cuisine and tourism.

Around 7 Samples per Hour Expensive and Time-Cost (Chatbot Arena Style)

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

###### User Simulation (Question Asking)

###### Rule

As someone passionate about travel and local cuisines, can you provide an overview of the couple's journey through Singapore, highlighting the must-try local dishes they sampled and the famous landmarks they visited?

Fault-Driven Evolution

Chat automatically with two anonymous models and vote for the better response.

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

- 1. Analyze the faults of two models.
- 2. Perform fault-driven question evolution.

###### Model Pool

Automatic Judging GPT-4o-mini Gemini-1.5-Flash

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

GPT-4o Gemini-1.5-Pro

[Figure 22]

[Figure 23]

[Figure 24]

A is better. B is better. Tie.

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

Qwen2-VL-72B LLaVA-Video-72B

Qwen2-VL-7B LLaVA-Video-7B

Peer Battle

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

LLaVA-OneVision-72B LLaVA-OneVision-7B

Aria

[Figure 35]

Model A Response Model B Response

Figure 1. An overview of our VideoAutoArena, where we leverage LMMs for user simulation to automatically evaluate LMMs in video analysis, offering an efficient alternative to costly and time-consuming human annotations, distinct from platforms like LMSYS Chatbot Arena [14] and WildVision Arena [45]. In this figure, we showcase 4 sampled frames from a Singapore travel vlog video.

### Abstract

Large multimodal models (LMMs) with advanced video analysis capabilities have recently garnered significant attention. However, most evaluations rely on traditional methods like multiple-choice question answering in benchmarks such as VideoMME and LongVideoBench, which are prone to lack the depth needed to capture the complex demands of real-world users. To address this limitation—and due to the prohibitive cost and slow pace of human annotation for video tasks—we introduce VideoAutoArena, an arena-style benchmark inspired by LMSYS Chatbot Arena’s framework, designed to automatically assess LMMs’ video analysis abilities. VideoAutoArena utilizes user simulation to generate open-ended, adaptive questions that rigorously assess model performance in video understanding. The benchmark features an automated, scalable evaluation framework, incorporating a modified ELO Rating System

♢Project Page: https://videoautoarena.github.io/

for fair and continuous comparisons across multiple LMMs. To validate our automated judging system, we construct a “gold standard” using a carefully curated subset of human annotations, demonstrating that our arena strongly aligns with human judgment while maintaining scalability. Additionally, we introduce a fault-driven evolution strategy, progressively increasing question complexity to push models toward handling more challenging video analysis scenarios. Experimental results demonstrate that VideoAutoArena effectively differentiates among state-of-the-art LMMs, providing insights into model strengths and areas for improvement. To further streamline our evaluation, we introduce VideoAutoBench as an auxiliary benchmark, where human annotators label winners in a subset of VideoAutoArena battles. We use GPT-4o as a judge to compare responses against these human-validated answers. Together, VideoAutoArena and VideoAutoBench offer a cost-effective, and scalable framework for evaluating LMMs in user-centric

video analysis.

### 1. Introduction

Recently, large multimodal models (LMMs) with advanced video understanding capabilities, such as GPT-4o [26], Gemini-1.5-Pro [2, 51], Aria [31], Qwen2-VL [57], and LLaVa-Video [30, 73], have garnered significant attention within the multimodal community. These models represent a major shift in the capabilities of artificial intelligence by extending beyond traditional image-based LMMs [1, 16, 32, 40] to encompass complex video inputs. Unlike their predecessors, which primarily focused on static visual data, these models are designed to handle dynamic, time-variant data, making them well-suited for processing lengthy and intricate video sequences. By leveraging language-based instructions from users, these LMMs can efficiently conduct a range of video analysis tasks, from understanding fine-grained scene details [31] to comprehending overarching narrative structures [26].

To better evaluate the video analysis capabilities of these models, recent works have introduced several widely used benchmarks, including MVBench [35], VideoMME [20], and LongVideoBench [59]. These benchmarks typically share a common feature: they pre-define core video analysis skills, such as object recognition in a single frame and action reasoning across a sequence of frames. They adopt an ability-centric approach, often using multiple-choice questions to assess performance. While these benchmarks have significantly contributed to the development of LMMs, they place limited emphasis on the types of questions real users might ask when seeking assistance with video analysis. In contrast, practical video analysis scenarios are far more complex and diverse in their requirements [14].

To address this gap, we can draw inspiration from the evaluation methods used for large language models (LLMs) [5, 17, 54, 55, 74]. Traditional language benchmarks, such as MMLU [24], IFEval [77], HumanEval [10], and GSM8k [15], also suffer from limited alignment with real user interactions [14, 72, 75]. To mitigate this issue, platforms like LMSYS Chatbot Arena [14] have been introduced, providing an open, crowdsourced platform for evaluating LLMs based on human preferences. Chatbot Arena employs a pairwise comparison approach, collecting feedback from a diverse set of users, ensuring that evaluation questions are generated by real users. This approach overcomes many of the challenges posed by previous benchmarks and has become one of the most widely adopted methods for evaluating LLMs.

One straightforward solution is to adapt this idea directly to LMMs for video analysis. Indeed, recent work, such as WildVision Arena [45], has attempted to do so. However, an analysis of the WildVision Video Arena leaderboard reveals

certain challenges with this approach. The video arena has received only 256 votes across 11 models, with each model participating in an average of just 23 battles since its release approximately 6 months ago. This limited number of battles can likely be attributed to the increased complexity involved in formulating questions for video-based tasks. Unlike language and image data, where questions can be quickly generated or verified within seconds, videos are typically longer and contain richer, more complex contexts. Annotators must spend significantly more time watching and understanding the video content before formulating high-quality questions. In our preliminary attempts, we hired annotators to complete this task, and the time cost proved significant, with a maximum output of only 7 samples annotated per hour. This time investment hinders the scalability of generating high-quality questions for LMMs, a crucial factor for ensuring the effectiveness of arena-style evaluations.

To address the limitations of existing video analysis benchmarks, we propose VideoAutoArena, a fully automated, arena-style evaluation method for LMMs. Unlike human-driven platforms, VideoAutoArena leverages LMM agents for user simulation and preference selection, eliminating the need for costly human annotators and enabling scalable, efficient evaluations. The framework also integrates fault-driven hard prompt evolution, which generates progressively challenging questions based on model performance, ensuring more rigorous testing. By simulating real-world user behavior, VideoAutoArena bridges the gap between ability-centric evaluations and practical application demands. Our human preference experiments show that 84.20% of the time, questions in VideoAutoArena better mimic real-world user question styles compared to VideoMME and LongVideoBench. Additionally, 87.29% of the time, our automatic judging aligns with human preference selection.

Experiments on 11 well-known proprietary and opensource LMMs reveal that open-source models still lag behind the SOTA closed-source model GPT-4o in video analysis, with a significant performance gap (-385.7). This gap is notably larger than those observed in traditional multiplechoice question-answering benchmarks. The disparity becomes even more pronounced as video length increases or the difficulty of the questions rises. Furthermore, when focusing on user-background relevance and helpfulness, the performance gap widens further. These findings highlight how our benchmark offers a user-centric perspective, providing valuable insights for the development of LMMs.

To complement VideoAutoArena, we also introduce VideoAutoBench, a streamlined benchmark designed for faster, more accessible evaluation of LMMs in video analysis. VideoAutoBench leverages a curated subset of bat-

The numbers are recorded on Nov. 14, 2024 at https:// huggingface.co/spaces/WildVision/vision-arena.

###### Benchmark Venue Long Video Included User-Centric Scalable Open-Ended Automated

MVBench [35] CVPR 24 ✗ ✗ ✗ ✗ ✓ MLVU [78] Arxiv 24 ✓ ✗ ✗ ✗ ✓ LVBench [58] Arxiv 24 ✓ ✗ ✗ ✗ ✓ VideoMME [20] Arxiv 24 ✓ ✗ ✗ ✗ ✓ LongVideoBench [59] NeurIPS 24 ✓ ✗ ✗ ✗ ✓ WildVision Video Arena [45] NeurIPS 24 ? ✓ ✗ ✓ ✗ VideoAutoArena (Ours) CVPR 25 ✓ ✓ ✓ ✓ ✓

Table 1. Comparison of recent popular benchmarks for video analysis. WildVision video data are not yet publicly available.

tles from VideoAutoArena, where human annotators have selected the winning model responses. Using GPT-4o for automatic judging, VideoAutoBench compares model answers against these human-selected and rejected responses, providing an efficient, cost-effective assessment method. Our results show that the rankings from VideoAutoBench align closely with those from VideoAutoArena, with a significant gap between SOTA closed-source and open-source LMMs, underscoring the benchmark’s challenge.

### 2. Related Work

LMMs with advanced video understanding capabilities have garnered significant research attention. For the closed-source models, GPT-4o [26] and Google’s Gemini1.5 [2, 51] demonstrate SOTA video analysis performance. Meanwhile, the open-source community has made notable strides [7, 9, 12, 13, 19, 21, 25, 27, 29, 34, 38, 39, 41, 42, 44, 46, 48, 52, 63–67, 69, 71]. Notably, the LLaVa series [40] has been updated to the LLaVa-Video [73] and LLaVaOneVision models [30], along with the release of all training data. The Qwen-VL model [3] has also been upgraded to the Qwen2-VL version [57], and the first open-source multimodal mixture-of-experts (MoE) model, Aria [31], has recently been introduced. These contributions have significantly narrowed the gap between closed-source and opensource models in video understanding. To accelerate the development of LMMs in video analysis, the establishment of more comprehensive benchmarks is essential.

In the early phase, researchers primarily relied on benchmarks featuring short videos [11, 22, 23, 50], such as MSVD-QA [62], MSRVTT-QA [62], NExT-QA [60], and MVBench [35]. However, these benchmarks have limitations due to their short video durations, averaging less than 50 seconds. This brevity restricts their ability to comprehensively evaluate the temporal understanding capabilities of LMMs, thereby hindering further advancements in LMMs development. To address these limitations, benchmarks like ActivityNet-QA [68] and EgoSchema [49] have extended video durations to approximately 180 seconds on average. More recently, research has introduced even more comprehensive benchmarks [43, 58, 70]. For instance, MovieChat-1K [53] assesses LMMs us-

ing movie videos with an average duration of 500 seconds, while LongVideoBench [59] focuses on long-context interleaved evaluation with an average duration of 473 seconds. Additionally, MLVU [78] presents a LMMs’ ability-centric benchmark, featuring substantial extensions of video lengths. Furthermore, VideoMME [20] introduces a highly comprehensive benchmark that includes short, medium, and long videos, further enhancing the evaluation of LMMs’ temporal understanding abilities. As discussed in the introduction, most current benchmarks are limited by multiple-choice questions that diverge from real user interaction. To address this, we introduce VideoAutoArena, which evaluates LMMs through open-ended, simulated human questions. Table 1 highlights the differences between our benchmark and other recent video-based benchmarks.

### 3. VideoAutoArena

#### 3.1. Overview

As illustrated in Figure 1, the VideoAutoArena pipeline consists of four core components: user simulation, peer battles, automatic judging, and fault-driven evolution. Initially, an agent reviews a video to identify user personas likely to be interested in the content. Adopting one of these personas, the agent formulates a relevant question about the video. Two randomly selected models then engage in peer battles to respond to the question. A judging agent determines which model provides the better response, while an analysis agent evaluates the responses, performs fault analysis, and generates progressively challenging questions to further assess the models’ capabilities. To demonstrate the effectiveness of our VideoAutoArena, we use 2,881 videos from LongVideoBench, with an average duration of 479 seconds. These videos are categorized into four duration ranges—(8s, 15s], (15s, 60s], (180s, 600s], (900s, 3600s]—and 10 categories: Movie, Life Vlogs, Geography, History, News Programs, Art, STEM, Computer Science, Cooking Recipes, and Travel Guides. The data distribution are provided in Figure 2. Our benchmark is not restricted to specific videos, and new videos can be easily incorporated into the evaluation pipeline.

Distribution of Video Categories

Distribution of Video Duration

###### Travel-Guides Movie 13.9%

(900s, 3600s] (180s, 600s] 26.1%

13.8%

26.0%

###### Cooking-Recipes

14.8%

Life-Vlogs13.3%

Total 2,881

Total 2,881

4.5% Computer-Science

5.1%

STEM

8.9%

22.8%

Geography

8.2%

25.1% (8s, 15s]

8.9%

8.6% Art

(15s, 60s]

History

News-Programs

Figure 2. Video statistics by category and duration.

#### 3.2. User Simulation

Ideally, real user queries would offer the direct evaluation of LMM performance in real life video analysis. However, given the complex contextual demands of video content, human annotation is expensive and time-cost. To address this, VideoAutoArena adopts user simulation with role-play, with SOTA LMMs acting as agents to generate realistic, user-centeric questions and preferences, enabling a more practical evaluation. In language-based rolyplay [6, 8, 28, 33], there is typically no need to consider external context, such as videos, allowing role-play to freely generate diverse questions. In video analysis, however, questions are constrained by the video content. To address this, we introduce a novel role-play method called video content-constrained persona generation. Given a video, agents first identify the types of users likely to be interested in it, defining three user types: (1) users with backgrounds highly relevant to the video; (2) users with moderately relevant backgrounds; and (3) users with unrelated backgrounds who encounter the video by chance. This relevance-based categorization aims to emulate real users with varied backgrounds seeking assistance from LMMs for video analysis. Once user types are established, agents adopt these personas to generate persona-specific questions for detailed video analysis. This user-centric process sets our evaluation method apart from previous ability-centric benchmarks. The prompts used are provided in the Appendix A.1.

#### 3.3. Peer Battles and Fault-Driven Evolution

Once the role-play agent generates a question for a specific video, we initiate peer battles between two anonymous models, following the Chatbot Arena style. Two models are randomly chosen from the model pool, presented with the same video and question, and asked to generate responses. Our goal is to fairly compare the quality of these responses to determine which LMM provides a better answer.

Similar to the concept of Hard Prompts in Chatbot Arena [36], incorporating more challenging questions can further push the boundaries of evaluating the abilities of current LMMs in video analysis. Thus, we aim to create a harder question set for evaluation purposes. Unlike Chatbot Arena, which can directly source hard prompts from millions of real user queries, our approach is limited to the

prompts generated by a user simulator. To derive harder questions, we draw inspiration from the famous instruction synthesis method, Evol-Instruct [47, 61], which evolves existing questions into more complex ones using predefined heuristic strategies. However, because Evol-Instruct generates questions based on a similar prompt structure for each evolution, it encounters limitations in question diversity. To address this, we introduce a fault-driven question evolution strategy that iteratively increases question complexity. Rather than relying on isolated prompts, each new evolution generates questions based on the results of the previous model battle. This approach creates a more adaptive and progressively challenging environment for the models, pushing them to respond to increasingly complex questions.

In this framework, a response analysis agent initially reviews responses from two competing models, identifying specific faults and performance weaknesses. Based on this assessment, role-play agents then generate tailored questions aimed at probing these weaknesses, making the question synthesis process progressively more fault-driven. After a new question is generated, a complexity evaluation agent assesses its difficulty. If the new question receives a higher overall complexity score than the previous one, it is retained for subsequent model battles. This iterative approach establishes a rigorous testing environment, challenging models with increasingly complex and contextually nuanced tasks, thus providing a deeper evaluation of each model’s video understanding capabilities. The prompts used are provided in the Appendix A.2.

#### 3.4. Judging and Ranking

A key aspect of arena-style evaluation is determining the winner in each model comparison. In Chatbot Arena, human annotators directly express their preferences, but in VideoAutoArena, human annotation is costly and difficult to scale due to the time-intensive nature of video analysis. To address this, we aim to automate the judgment process. Drawing inspiration from automated judging benchmarks like Arena-Auto-Hard [36] and MT-Bench [76], we first define our judging standards as follows:

- 1. Instruction Following: The response should closely adhere to the user’s instructions, ensuring it directly addresses the specified task.
- 2. Accuracy: The response must utilize information from the video accurately, avoiding fabrication or misquotation. It should maintain factual correctness, avoid hallucinations, and demonstrate contextual coherence with precise terminology and knowledge.
- 3. Relevance: The response should consider the user’s background information and needs, providing a comprehensive, detailed answer that addresses the question directly without straying off-topic. Responses should be thorough, offering multiple perspectives where relevant.

[Figure 36]

Video Caption

This video covers a MoMA workshop and performance, "Walking Tables and Wrestling Foals." It features participants assembling wooden structures and interviews with Douglas Repetto and Mark Allen, who discuss the project’s inspiration. The event includes interactive elements and live music, with both audience and participants actively involved.

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

User Simulation (Role Play) User Simulation (Question Asking)

###### Comparing with Popular Long Video Benchmarks

A person who is an art educator and researcher specializing in Bauhaus studies. They are actively involved in art workshops and exhibitions, making this video directly relevant to their field of work and study.

How does the video illustrate the relationship between Bauhaus principles and contemporary art practices in the context of hands-on workshops, and what role do the participants play in this interaction?

related

[Figure 42]

###### LongVideoBench Question Style

Before the subtitle says 'certain kinds of errors are what give,' what first appears when a girl in a deep red dress kneeling on the left and a girl in a black dress standing on the right are seen together?

A person who is an engineering student with a keen interest in design and collaborative projects. Although their primary focus is not art, they are curious about the intersection of technology and creative expression, which this video exemplifies.

How did the collaboration between the engineers and the creative team at MoMA facilitate the

Multi-Choice QA Our

less related

[Figure 43]

Open Ended QA

construction of the mechanical structures and what was the primary challenge they faced in combining engineering with artistic expression?

VideoMME Question Style

A person who works in finance and has no formal background in art or design. However, they enjoy attending cultural events and are open to

As someone who usually works in finance and is not deeply familiar with art, I'm curious about how this collaboration at MoMA connects concepts from the Bauhaus movement with modern artistic practices. Could you explain how this video showcases that relationship?

not related

What inspires the collaborative project between Douglas Repetto and Mark Allen, and how does it relate to the Bauhaus Lab?

[Figure 44]

exploring new interests. This video piques their curiosity about innovative artistic concepts and experiences.

Multi-Choice QA

- Figure 3. Examples of synthesized personas with three levels of relevance and corresponding synthesized questions. We also compare the style of our questions with those in popular long-video benchmarks, including LongVideoBench and VideoMME.

- 4. Helpfulness: The response should provide valuable information to aid the user in understanding or solving their issue, avoiding irrelevant or vague content.

Based on these standards, we adopt the LMM-as-a-Judge paradigm [36, 45, 76] to automatically determine the better response between two models. The prompts used are provided in the Appendix A.3.

After obtaining the automatic judging results, we utilize the ELO Rating System [18] to establish a dynamic evaluation platform that ranks LMMs through statistical modeling based on direct pairwise comparisons. Here, we briefly explain the Online ELO Rating and statistical estimation methods. The Online ELO Rating system calculates the probability that model i will win against model j using their respective ratings, Ri and Rj, where i,j ∈ N. For each comparison, we define a binary outcome Yij, where Yij = 1 if model i wins and Yij = 0 otherwise. The probability is computed as follows:

1 1 + 10(Rj−Ri)/α ,

P(Yij = 1) =

where α = 400 is the scaling factor in the ELO computation. After each comparison, player ratings are updated by:

Ri′ = Ri + K × (S(i,j) − P(Yij = 1)),

where S(i,j) represents the actual outcome (1 for a win, 0.5 for a tie, and 0 for a loss). Higher-rated players gain fewer points if they win and lose more if defeated, while lower-rated players experience the reverse. Since ELO updates are sensitive to comparison order, we employ the Bradley–Terry model [4] for stable statistical estimation, as in Chatbot Arena.

The Bradley–Terry model refines ELO ratings through a logistic regression model using maximum likelihood esti-

mation. For N models with pairwise comparisons, where Wij is the count of times model i wins over j, the loglikelihood function for all comparisons is:

###### R =

WijYijP(Yij = 1),

i,j∈N,i̸=j

where R = {R1,...,RN} represents each model’s rating. Since this approach doesn’t accommodate ties directly, we split all tie votes, counting half as wins for model i (Yij = 1) and the other half as wins for model j (Yij = 0). This ensures balanced ranking and fair statistical estimation across all model comparisons.

#### 3.5. Experiments

Setup. To demonstrate the effectiveness of VideoAutoArena, we evaluate 11 SOTA LMMs, including GPT4o/mini, Gemini-1.5-Pro/Flash, Aria, Qwen2-VL-72B/7B, LLaVa-Video-72B/7B, and LLaVa-OneVision-72B/7B, all of which have shown strong performance on the VideoMME. For response generation, each video was uniformly sampled to provide 64 frames as input. Since most of these LMMs do not support audio, the audio track was converted to subtitles and combined with the question as input. For automatic judging, each video was sampled to provide 128 frames and combined with subtitles. Additional experimental details are provided in the Appendix B.

User Simulation and Diversity. In our experiments, we generated an average of three personas per video across 2.9k videos, resulting in about 8.6k unique personas. As shown in Figure 3, we showcases examples of three personas with varying levels of relevance to a given video. Each persona includes motivations or reasons for their interest in

Claude-Series LMMs were excluded from our experiments due to limited support for long video input (maximum 20 frames).

t-SNE Visualization of the Persona

PersonaHub Our Persona

80

60

40

20

0

20

40

60

80

75 50 25 0 25 50 75

(a) Visualization of persona distribution.

Ranking of Questions Across Humans Preference

| | | | | |
|---|---|---|---|---|
| |14.2%<br><br>1.7% 1.7%| | | |
| | | | | |
| |84.2%<br><br>82.5%<br><br>96.7%| | | |
| | | | | |
| |15.8%<br><br>3.3%| | | |
| | | | | |

100

80

60

40

20

0

VideoAutoArena VideoMME LongVideoBench

First Place Second Place Third Place

(b) Humans preference ranking.

- Figure 4. Our user simulation offers diverse personas and more effectively mirrors real-world users’ question styles.

the video, enabling more persona-specific question generation. Moreover, in Figure 4a, we compare the distribution of our synthesized personas with those from PersonaHub [6], which includes diverse personas automatically curated from web. To map the distribution, we used one of the SOTA sentence embedding models, gte-large-en-v1.5 [37], to encode each persona description into vectors, then applied t-SNE [56] for dimensionality reduction. The results show that our synthesized personas achieve diversity comparable to that in PersonaHub. Notably, our personas are constrained by video content, while PersonaHub personas are synthesized without such constraints, highlighting that our personas effectively simulate a range of realistic user backgrounds. Additional examples of synthesized personas are provided in Appendix C.1.

After generating personas, our role-play agent adopts each persona to ask relevant questions about the video content, seeking help from AI assistants. In our experiments, we synthesized one question per unique personas for 2.9k videos. To assess how well our questions mimic real-world user queries, we randomly selected 120 questions and compared them with question styles from LongVideoBench and VideoMME. Since our videos are from LongVideoBench, we used the same set for consistency. However, since VideoMME uses different videos, we applied a style transfer method, adapting VideoMME questions to synthesize similar-style questions based on our benchmark’s videos. This allowed a fair comparison across the three benchmarks

Difficulty Level Before and After Evolution

| |4.09<br><br>3.98 3.94| | | | | |
|---|---|---|---|---|---|---|
| |3.38 3.35| | | | | |
| |3.15<br><br>2.95 2.92| | | | | |
| |2.39 2.35| | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

4.0

3.5

3.0

DifficultyLevel

2.5

2.0

1.5

1.0

0.5

0.0

InsFollow Accuracy Relevance Helpfulness Overall

Before Evol After Evol

Figure 5. Our fault-driven evolution strategy generates increasingly challenging questions for video analysis.

on identical video samples. To evaluate the naturalness of each benchmark’s question style, we conducted a blind ranking task, asking three skilled annotators to rank which questions best mimic real-world user queries, with Rank 1 as the best. Annotation guidelines are detailed in the Appendix D.1. As shown in Figure 4b, for 84.2% of the comparisons, VideoAutoArena’s questions ranked first in mimicking real-world user question styles. This result highlights how our benchmark brings a unique perspective to evaluating LMMs for video analysis. Figure 3 further provides example questions to illustrate the style of our questions in mimicking real-world users. In Appendix C.2, we also include more synthesized question examples.

Question Evolution. To analyze the hard prompts generated by our fault-driven evolution, we perform an automated complexity assessment of the synthesized questions. Using the SOTA LMM, GPT-4o, we evaluate the difficulty level of these questions on a 1–5 scale before and after evolution. We further break down the difficulty into 5 categories—instruction following, accuracy, relevance, helpfulness and overall—based on our automatic judging standards. As shown in Figure 5, the evolved questions consistently achieve higher difficulty scores across all categories, demonstrating the effectiveness of our evolution strategy. The analysis prompts are included in Appendix A.4.

Automatic Judging. To establish a “gold standard” for evaluating the accuracy of automatic judging, framed as a 3choice task (A, B, or Tie), we created a benchmark guided by our judging criteria. Since no public human annotations are available for this, we engaged annotators to carefully evaluate a subset of battles. Annotation guidelines are detailed in the Appendix D.2. Given the labor-intensive nature of this work—yielding about 7 annotations per hour—we randomly selected around 300 battles across various video lengths and models for annotation. Figure 6 presents the accuracy comparison among SOTA LMMs and voting methods for automated judging. GPT-4o demonstrated the highest alignment with human preferences, achieving an 87.29% agreement. Notably, employing a voting approach with multiple LMMs (Top2 to Top4) did not result in better

Models Size ELO Win Rates (8s, 15s] (15s, 60s] (180s, 600s] (900s, 3600s] Proprietary Models GPT-4o - 1505.69 89.19 1447.86 1449.59 1575.34 1552.23 GPT-4o-mini - 1323.25 76.90 1293.27 1343.28 1327.75 1349.29 Gemini-1.5-Pro - 1187.01 65.11 1247.65 1171.82 1263.58 1291.64 Gemini-1.5-Flash - 1149.52 62.07 1081.58 1131.27 1140.07 1260.36

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

Open-Source Models Aria 8×3.5B 1119.99 59.54 1147.45 1273.77 1110.67 1111.40 Qwen2-VL 72B 886.52 35.61 985.46 928.23 829.65 826.56 Qwen2-VL 7B 875.56 34.90 969.28 859.33 850.30 829.21 LLaVA-Video 72B 836.62 30.25 796.90 850.12 827.88 782.55 LLaVA-Video 7B 765.61 23.52 672.35 736.14 759.15 721.78 LLaVA-OneVision 72B 763.71 23.11 731.50 710.64 759.29 741.80 LLaVA-OneVision 7B 586.52 9.86 626.70 545.82 556.31 533.18

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

Table 2. Our VideoAutoArena Leaderboard. We show the overall ELO ratings and win rates within four different video lengths.

Accuracy of Different Judges

100

85.57 87.29

86.25 84.19

83.51

79.73 78.35

80

Accuracy(%)

60

40

33.47

20

0

RandomVote(Top2)Vote(Top3)Vote(Top4)Gemini-1.5-FlashGemini-1.5-ProGPT-4o-mini GPT-4o

- Figure 6. Evaluate the accuracy of various judging methods using human annotations as the gold standard. In the Vote (Top N) method, the top N models are used to cast votes.

GPT-4oGPT-4o-miniGemini-1.5-ProGemini-1.5-FlashQwen2-VL-72BAriaQwen2-VL-7BLLaVA-Video-72BLLaVA-OneVision-72BLLaVA-Video-7BLLaVA-OneVision-7B

600

800

1000

1200

1400

1600

ELORating

1512.591498.04

1316.391332.29

1177.681199.56 1144.911155.77 1115.211126.35

886.64886.10876.09874.57 843.65826.79

770.56762.18764.85758.66

591.44579.69

ELO Ratings of Different Models

ELO Rating (Before Evol)

Hard Prompt ELO Rating (After Evol)

- Figure 7. ELO ratings for models competing on questions before and after applying fault-aware evolution.

agreement. As a result, we selected GPT-4o as the primary judge for determining the winning responses due to its strong alignment with human judgments.

Leaderboard. As shown in Table 2, we present the ELO ratings and win rates across 4 video length categories for 11 SOTA LMMs in video analysis. Our evaluation involves a total of 12,479 head-to-head battles, with each model participating in roughly 1,600 battles, and each model pair competing approximately 150 times—significantly surpass-

ing the scale of WildVision Video Arena and demonstrating the scalability of our method. In terms of ELO ratings, the leading open-source LMM, Aria, lags behind the top proprietary LMM, GPT-4o, by a notable margin (-385.7), underscoring the benchmark’s strong discriminative power. Unlike in VideoMME, where the score gap among the top six models is less than 10%. For shorter videos (under 60 seconds), the gap between the top proprietary and opensource LMMs narrows, with Aria even surpassing the wellknown Gemini-1.5-Pro on videos between 15 and 60 seconds. However, as video length increases, performance gaps widen considerably; for instance, Qwen2-VL sees a drop of over 100 ELO points, while Gemini-1.5-Flash gains around 180 ELO points when comparing results on short versus long videos. Notably, Aria demonstrates stable and strong performance across varying video lengths.

In addition, Figure 7 presents the ELO ratings for models competing on questions before and after applying faultaware evolution. When challenged with more difficult prompts, all open-source LMMs except Aria exhibit lower scores, whereas all proprietary LMMs, apart from GPT-4o, show improved scores. Figure 8 further details the performance of various LMMs across different judging standards. Notably, the gap between proprietary and open-source LMMs is most evident in the helpfulness and relevance tracks, compared to the instruction-following and accuracy tracks. This gap suggests that many open-source LMMs are primarily optimized for traditional, ability-focused benchmarks like VideoMME, which overlook user-centered aspects such as contextual relevance and information helpfulness. By focusing on a more comprehensive evaluation, our VideoAutoArena bridges this gap, providing deeper insights into the limitations and future development potential of LMMs. In the Appendix C.3, we also include examples of different LMMs’ responses as case studies to better understand the weaknesses of these models.

ELO Ratings of Different Models Across Tracks

GPT-4o

1485.10

GPT-4o-mini

1600

1441.22

Gemini-1.5-Pro

1367.93 1355.50

Gemini-1.5-Flash

1317.38 1335.40

Aria

1278.03

1400

1241.19

Qwen2-VL-72B

Qwen2-VL-7B

1153.80 1147.16 1149.34 1115.84 1128.43 1130.18 1125.62

1128.03

ELORating

LLaVA-Video-72B

1057.74 1077.75

LLaVA-OneVision-72B

1200

1049.80 1048.40

LLaVA-Video-7B

LLaVA-OneVision-7B

923.33919.03892.33855.14802.23 927.19907.38884.83850.01791.23 907.93899.26877.93830.51773.48 902.49890.33873.75819.52758.84 660.39

1000

682.71

800

625.14 611.21

600

Instruction Following Accuracy Relevance Helpfulness

Figure 8. We evaluate the performance of various models based on four different judging standards.

[Figure 56]

Models vs. Sel. vs. Rej. Avg.

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

GPT-4o 70.98 94.12 82.55 GPT-4o-mini 49.80 92.16 70.98 Gemini-1.5-Pro 28.24 82.74 55.49 Gemini-1.5-Flash 27.25 81.96 54.61 Aria 19.80 76.86 48.33 Qwen2-VL-72B 13.92 64.71 39.32 Qwen2-VL-7B 11.96 60.00 35.98 LLaVA-Video-72B 7.45 56.08 31.77 LLaVA-OneVision-72B 4.12 52.16 28.14 LLaVA-Video-7B 5.29 46.67 25.98 LLaVA-OneVision-7B 3.53 30.98 17.26

[Figure 61]

[Figure 62]

[Figure 63]

In the video, the host discusses the significance of the quetzal in relation to the Guatemalan flag. Could you explain its importance and what other elements are symbolically represented on the flag as described in the video? Question

[Figure 64]

[Figure 65]

The quetzal bird is a significant symbol on the Guatemalan flag, representing the country's national animal. The flag features a vertical tricolor of blue and white, with the blue stripes symbolizing the oceans surrounding Guatemala, and the white stripe representing peace and purity. The Guatemalan coat of arms is centered on the white stripe, which includes a parchment scroll inscribed with "LIBERTAD 15 DE SEPTIEMBRE DE 1821," commemorating the country's independence from Spain on September 15, 1821. Additionally, the coat of arms is surrounded by a wreath of laurel branches, symbolizing victory, and crossed swords and rifles, representing honor and Guatemala's readiness to defend itself.

[Figure 66]

[Figure 67]

[Figure 68]

Aria Response

[Figure 69]

[Figure 70]

The quetzal, a national symbol of Guatemala, is featured on the flag to represent freedom and independence. Other elements on the flag include a laurel wreath symbolizing victory, rifles and swords indicating the country's defense and honor, and a scroll with the date "15 de septiembre de 1821" marking Guatemala's independence from Spain.

[Figure 71]

LLaVA-Video-72B Response

Figure 9. Example of a battle between Aria and LLaVa-Video72B. Red highlights key content, while green highlights important details mentioned only by Aria.

Battle Example. As shown in Figure 9, we present an example of a battle between Aria and LLaVa-Video-72B. Both models follow the question and provide accurate information about the Guatemalan flag. However, only Aria includes additional details, such as the flag’s vertical tricolor, while LLaVa-Video’s response lacks sufficient depth on these aspects. Consequently, Aria’s response is more helpful for users seeking to understand the flag’s symbolism. The automated judge also selects Aria as the winner. Traditional multiple-choice QA benchmarks, like VideoMME, cannot reveal such open-ended limitations in LMMs. More examples are included in the Appendix C.3.

### 4. VideoAutoBench

While VideoAutoArena offers a novel approach to evaluating LMMs in video analysis, it is less immediately user-friendly than multiple-choice benchmarks like VideoMME and LongVideoBench, which provide straightforward scores based on model responses. In contrast, VideoAutoArena requires the target model to engage in comparative battles with other models to generate results. To streamline this evaluation process, we introduce VideoAutoBench, which combines the user-centric assessment strengths of VideoAutoArena with the simplicity and

Table 3. LMMs compete against human selected or rejected answers in our VideoAutoBench.

speed of traditional benchmarks. In our automated judging experiments, we included human annotators to label winners for a subset of battles, using these questions and nontied responses as reference answers. In VideoAutoBench, we employ GPT-4o as the judging model to evaluate each model’s responses against the human-selected or rejected answers, with GPT-4o voting based on the same standards used in VideoAutoArena. When competing with humanselected answers, a model earns 1 point for a win, 0.5 for a tie, and 0 for a loss, with the final score being the average across all battles. When competing with human-rejected answers, only a win earns 1 point. Table 3 presents the performance of different LMMs, showing the similar rank as in VideoAutoArena.

### 5. Conclusion

We introduce VideoAutoArena, an automated arena-style benchmark that addresses the limitations of traditional multiple-choice video QA benchmarks. Using user simulation, peer battles, automated judging, and fault-driven evolution, VideoAutoArena enables a scalable, user-centric evaluation for complex video analysis tasks. Alongside, we present VideoAutoBench, a streamlined evaluation comparing model responses to human-labeled answers. Experiments on 11 SOTA LMMs reveal a notable performance gap between closed and open-source models, particularly for

long videos, challenging questions, and scenarios involving user background relevance and response helpfulness.

### References

- [1] Gpt-4v(ision) system card. 2023. 2
- [2] Rohan Anil, Sebastian Borgeaud, Yonghui Wu, JeanBaptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M. Dai, Anja Hauth, Katie Millican, David Silver, Slav Petrov, Melvin Johnson, Ioannis Antonoglou, Julian Schrittwieser, Amelia Glaese, Jilin Chen, Emily Pitler, Timothy P. Lillicrap, Angeliki Lazaridou, Orhan Firat, James Molloy, Michael Isard, Paul Ronald Barham, Tom Hennigan, Benjamin Lee, Fabio Viola, Malcolm Reynolds, Yuanzhong Xu, Ryan Doherty, Eli Collins, Clemens Meyer, Eliza Rutherford, Erica Moreira, Kareem Ayoub, Megha Goel, George Tucker, Enrique Piqueras, Maxim Krikun, Iain Barr, Nikolay Savinov, Ivo Danihelka, Becca Roelofs, Ana¨ıs White, Anders Andreassen, Tamara von Glehn, Lakshman Yagati, Mehran Kazemi, Lucas Gonzalez, Misha Khalman, Jakub Sygnowski, and et al. Gemini: A family of highly capable multimodal models. CoRR, abs/2312.11805, 2023. 2, 3
- [3] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. 2023. 3
- [4] Ralph Allan Bradley and Milton E. Terry. Rank analysis of incomplete block designs: I. the method of paired comparisons. Biometrika, 39:324, 1952. 5
- [5] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual, 2020. 2
- [6] Xin Chan, Xiaoyang Wang, Dian Yu, Haitao Mi, and Dong Yu. Scaling synthetic data creation with 1,000,000,000 personas. CoRR, abs/2406.20094, 2024. 4, 6
- [7] Guo Chen, Yin-Dong Zheng, Jiahao Wang, Jilan Xu, Yifei Huang, Junting Pan, Yi Wang, Yali Wang, Yu Qiao, Tong Lu, and Limin Wang. Videollm: Modeling video sequence with large language models. CoRR, abs/2305.13292, 2023. 3
- [8] Guhong Chen, Liyang Fan, Zihan Gong, Nan Xie, Zixuan Li, Ziqiang Liu, Chengming Li, Qiang Qu, Shiwen Ni, and Min Yang. Agentcourt: Simulating court with adversarial evolvable lawyer agents. CoRR, abs/2408.08089, 2024. 4

- [9] Lin Chen, Xilin Wei, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Bin Lin, Zhenyu Tang, Li Yuan, Yu Qiao, Dahua Lin, Feng Zhao, and Jiaqi Wang. Sharegpt4video: Improving video understanding and generation with better captions. CoRR, abs/2406.04325,

2024. 3

- [10] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Pond´e de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Joshua Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. Evaluating large language models trained on code. CoRR, abs/2107.03374, 2021. 2
- [11] Xiuyuan Chen, Yuan Lin, Yuchen Zhang, and Weiran Huang. Autoeval-video: An automatic benchmark for assessing large vision language models in open-ended video question answering. CoRR, abs/2311.14906, 2023. 3
- [12] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, Bin Li, Ping Luo, Tong Lu, Yu Qiao, and Jifeng Dai. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. CoRR, abs/2312.14238, 2023. 3
- [13] Zesen Cheng, Sicong Leng, Hang Zhang, Yifei Xin, Xin Li, Guanzheng Chen, Yongxin Zhu, Wenqi Zhang, Ziyang Luo, Deli Zhao, and Lidong Bing. Videollama 2: Advancing spatial-temporal modeling and audio understanding in videollms. CoRR, abs/2406.07476, 2024. 3
- [14] Wei-Lin Chiang, Lianmin Zheng, Ying Sheng, Anastasios Nikolas Angelopoulos, Tianle Li, Dacheng Li, Banghua Zhu, Hao Zhang, Michael I. Jordan, Joseph E. Gonzalez, and Ion Stoica. Chatbot arena: An open platform for evaluating llms by human preference. In Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024. OpenReview.net, 2024. 1, 2
- [15] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. CoRR, abs/2110.14168, 2021. 2
- [16] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven C. H. Hoi. Instructblip: Towards generalpurpose vision-language models with instruction tuning. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems

2023, NeurIPS 2023, New Orleans, LA, USA, December 10

- 16, 2023, 2023. 2

- [17] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, Arun Rao, Aston Zhang, Aur´elien Rodriguez, Austen Gregerson, Ava Spataru, Baptiste Rozi`ere, Bethany Biron, Binh Tang, Bobbie Chern, Charlotte Caucheteux, Chaya Nayak, Chloe Bi, Chris Marra, Chris McConnell, Christian Keller, Christophe Touret, Chunyang Wu, Corinne Wong, Cristian Canton Ferrer, Cyrus Nikolaidis, Damien Allonsius, Daniel Song, Danielle Pintz, Danny Livshits, David Esiobu, Dhruv Choudhary, Dhruv Mahajan, Diego Garcia-Olano, Diego Perino, Dieuwke Hupkes, Egor Lakomkin, Ehab AlBadawy, Elina Lobanova, Emily Dinan, Eric Michael Smith, Filip Radenovic, Frank Zhang, Gabriel Synnaeve, Gabrielle Lee, Georgia Lewis Anderson, Graeme Nail, Gr´egoire Mialon, Guan Pang, Guillem Cucurell, Hailey Nguyen, Hannah Korevaar, Hu Xu, Hugo Touvron, Iliyan Zarov, Imanol Arrieta Ibarra, Isabel M. Kloumann, Ishan Misra, Ivan Evtimov, Jade Copet, Jaewon Lee, Jan Geffert, Jana Vranes, Jason Park, Jay Mahadeokar, Jeet Shah, Jelmer van der Linde, Jennifer Billock, Jenny Hong, Jenya Lee, Jeremy Fu, Jianfeng Chi, Jianyu Huang, Jiawen Liu, Jie Wang, Jiecao Yu, Joanna Bitton, Joe Spisak, Jongsoo Park, Joseph Rocca, Joshua Johnstun, Joshua Saxe, Junteng Jia, Kalyan Vasuden Alwala, Kartikeya Upasani, Kate Plawiak, Ke Li, Kenneth Heafield, Kevin Stone, and et al. The llama 3 herd of models. CoRR, abs/2407.21783, 2024. 2
- [18] A.E. Elo. The USCF Rating System: Its Development, Theory, and Applications. United States Chess Federation, 1966. 5
- [19] Jiajun Fei, Dian Li, Zhidong Deng, Zekun Wang, Gang Liu, and Hui Wang. Video-ccam: Enhancing video-language understanding with causal cross-attention masks for short and long videos. CoRR, abs/2408.14023, 2024. 3
- [20] Chaoyou Fu, Yuhan Dai, Yondong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, Peixian Chen, Yanwei Li, Shaohui Lin, Sirui Zhao, Ke Li, Tong Xu, Xiawu Zheng, Enhong Chen, Rongrong Ji, and Xing Sun. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. CoRR, abs/2405.21075, 2024. 2, 3
- [21] Chaoyou Fu, Haojia Lin, Zuwei Long, Yunhang Shen, Meng Zhao, Yifan Zhang, Xiong Wang, Di Yin, Long Ma, Xiawu Zheng, Ran He, Rongrong Ji, Yunsheng Wu, Caifeng Shan, and Xing Sun. VITA: towards open-source interactive omni multimodal LLM. CoRR, abs/2408.05211, 2024. 3
- [22] Raghav Goyal, Samira Ebrahimi Kahou, Vincent Michalski, Joanna Materzynska, Susanne Westphal, Heuna Kim, Valentin Haenel, Ingo Fr¨und, Peter Yianilos, Moritz MuellerFreitag, Florian Hoppe, Christian Thurau, Ingo Bax, and Roland Memisevic. The ”something something” video database for learning and evaluating visual common sense. In IEEE International Conference on Computer Vision,

ICCV 2017, Venice, Italy, October 22-29, 2017, pages 5843–

5851. IEEE Computer Society, 2017. 3

- [23] Madeleine Grunde-McLaughlin, Ranjay Krishna, and Maneesh Agrawala. AGQA: A benchmark for compositional spatio-temporal reasoning. In IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2021, virtual, June 19-25, 2021, pages 11287–11297. Computer Vision Foundation / IEEE, 2021. 3
- [24] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021. 2
- [25] Bin Huang, Xin Wang, Hong Chen, Zihan Song, and Wenwu Zhu. Vtimellm: Empower LLM to grasp video moments. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024, pages 14271–14280. IEEE, 2024. 3
- [26] OpenAI Aaron Hurst, Adam Lerer, Adam P. Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, Aleksander Mkadry, Alex Baker-Whitcomb, Alex Beutel, Alex Borzunov, Alex Carney, Alex Chow, Alexander Kirillov, Alex Nichol, Alex Paino, Alex Renzin, Alexandre Passos, Alexander Kirillov, Alexi Christakis, Alexis Conneau, Ali Kamali, Allan Jabri, Allison Moyer, Allison Tam, Amadou Crookes, Amin Tootoochian, Amin Tootoonchian, Ananya Kumar, Andrea Vallone, Andrej Karpathy, Andrew Braunstein, Andrew Cann, Andrew Codispoti, Andrew Galu, Andrew Kondrich, Andrew Tulloch, Andrey Mishchenko, Angela Baek, Angela Jiang, Antoine Pelisse, Antonia Woodford, Anuj Gosalia, Arka Dhar, Ashley Pantuliano, Avi Nayak, Avital Oliver, Barret Zoph, B. Ghorbani, Ben Leimberger, Ben Rossen, Benjamin D. Sokolowsky, Ben Wang, Benjamin Zweig, Beth Hoover, Blake Samic, Bob McGrew, Bobby Spero, Bogo Giertler, Bowen Cheng, Brad Lightcap, Brandon Walkin, Brendan Quinn, Brian Guarraci, Brian Hsu, Bright Kellogg, Brydon Eastman, Camillo Lugaresi, Carroll L. Wainwright, Cary Bassin, Cary Hudson, Casey Chu, Chad Nelson, Chak Ming Li, Chan Jun Shern, Channing Conger, Charlotte Barette, Chelsea Voss, Chen Ding, Cheng Lu, Chong Zhang, Chris Beaumont, Chris Hallacy, Chris Koch, Christian Gibson, Christina Kim, Christine Choi, Christine McLeavey, Chris Hesse, Claudia Fischer, Clemens Winter, Coley Czarnecki, Colin Jarvis, Colin Wei, Constantin Koumouzelis, Dane Sherburn, Daniel Kappler, Daniel Levin, Daniel Levy, David Carr, David Farhi, David M´ely, David Robinson, David Sasaki, Denny Jin, Dev Valladares, Dimitris Tsipras, Doug Li, Duc Phong Nguyen, Duncan Findlay, Edede Oiwoh, Edmund Wong, Ehsan Asdar, Elizabeth Proehl, Elizabeth Yang, Eric Antonow, Eric Kramer, Eric Peterson, Eric Sigler, Eric Wallace, Eugene Brevdo, Evan Mays, Farzad Khorasani, Felipe Petroski Such, Filippo Raso, Francis Zhang, Fred von Lohmann, Freddie Sulit, Gabriel Goh, Gene Oden, Geoff Salmon, Giulio Starace, Greg Brockman, Hadi Salman, Hai-Biao Bao, Haitang Hu, Hannah Wong, Haoyu Wang, Heather Schmidt, Heather

Whitney, Heewoo Jun, Hendrik Kirchner, Henrique Pond´e de Oliveira Pinto, Hongyu Ren, Huiwen Chang, Hyung Won Chung, Ian D. Kivlichan, Ian O’Connell, Ian Osband, Ian Silber, Ian Sohl, ˙Ibrahim Cihangir Okuyucu, Ikai Lan, Ilya Kostrikov, Ilya Sutskever, Ingmar Kanitscheider, Ishaan Gulrajani, Jacob Coxon, Jacob Menick, Jakub W. Pachocki, James Aung, James Betker, James Crooks, James Lennon, Jamie Ryan Kiros, Jan Leike, Jane Park, Jason Kwon, Jason Phang, Jason Teplitz, Jason Wei, Jason Wolfe, Jay Chen, Jeff Harris, Jenia Varavva, Jessica Gan Lee, Jessica Shieh, Ji Lin, Jiahui Yu, Jiayi Weng, Jie Tang, Jieqi Yu, Joanne Jang, Joaquin Qui˜nonero Candela, Joe Beutler, Joe Landers, Joel Parish, Johannes Heidecke, John Schulman, Jonathan Lachman, Jonathan McKay, Jonathan Uesato, Jonathan Ward, Jong Wook Kim, Joost Huizinga, Jordan Sitkin, Jos Kraaijeveld, Joshua Gross, Josh Kaplan, Josh Snyder, Josh Achiam, Joy Jiao, Joyce Lee, Juntang Zhuang, Justyn Harriman, Kai Fricke, Kai Hayashi, Karan Singhal, Katy Shi, Kavin Karthik, Kayla Wood, Kendra Rimbach, Kenny Hsu, Kenny Nguyen, Keren Gu-Lemberg, Kevin Button, Kevin Liu, Kiel Howe, Krithika Muthukumar, Kyle Luther, Lama Ahmad, Larry Kai, Lauren Itow, Lauren Workman, Leher Pathak, Leo Chen, Li Jing, Lia Guy, Liam Fedus, Liang Zhou, Lien Mamitsuka, Lilian Weng, Lindsay McCallum, Lindsey Held, Ouyang Long, Louis Feuvrier, Lu Zhang, Lukasz Kondraciuk, Lukasz Kaiser, Luke Hewitt, Luke Metz, Lyric Doshi, Mada Aflak, Maddie Simens, Madelaine Boyd, Madeleine Thompson, Marat Dukhan, Mark Chen, Mark Gray, Mark Hudnall, Marvin Zhang, Marwan Aljubeh, Ma teusz Litwin, Matthew Zeng, Max Johnson, Maya Shetty, Mayank Gupta, Meghan Shah, Mehmet Ali Yatbaz, Meng Jia Yang, Mengchao Zhong, Mia Glaese, Mianna Chen, Michael Janner, Michael Lampe, Michael Petrov, Michael Wu, Michele Wang, Michelle Fradin, Michelle Pokrass, Miguel Castro, Miguel Castro, Mikhail Pavlov, Miles Brundage, Miles Wang, Minal Khan, Mira Murati, Mo Bavarian, Molly Lin, Murat Yesildal, Nacho Soto, Natalia Gimelshein, Natalie Cone, Natalie Staudacher, Natalie Summers, Natan LaFontaine, Neil Chowdhury, Nick Ryder, Nickolas Stathas, Nick Turley, Nikolas A. Tezak, Niko Felix, Nithanth Kudige, Nitish Shirish Keskar, Noah Deutsch, Noel Bundick, Nora Puckett, Ofir Nachum, Ola Okelola, Oleg Boiko, Oleg Murk, Oliver Jaffe, Olivia Watkins, Olivier Godement, Owen Campbell-Moore, Patrick Chao, Paul McMillan, Pavel Belov, Peng Su, Peter Bak, Peter Bakkum, Peter Deng, Peter Dolan, Peter Hoeschele, Peter Welinder, Phil Tillet, Philip Pronin, Phil Tillet, Prafulla Dhariwal, Qim ing Yuan, Rachel Dias, Rachel Lim, Rahul Arora, Rajan Troll, Randall Lin, Raphael Gontijo Lopes, Raul Puri, Reah Miyara, Reimar H. Leike, Renaud Gaubert, Reza Zamani, Ricky Wang, Rob Donnelly, Rob Honsby, Rocky Smith, Rohan Sahai, Rohit Ramchandani, Romain Huet, Rory Carmichael, Rowan Zellers, Roy Chen, Ruby Chen, Ruslan Ramilevich Nigmatullin, Ryan Cheu, Saachi Jain, Sam Altman, Sam S. Schoenholz, Sam Toizer, Samuel Miserendino, Sandhini Agarwal, Sara Culver, Scott Ethersmith, Scott Gray, Sean Grove, Sean Metzger, Shamez Hermani, Shantanu Jain, Shengjia Zhao, Sherwin Wu, Shino Jo-

moto, Shirong Wu, Shuaiqi Xia, Sonia Phene, Spencer Papay, Srinivas Narayanan, Steve Coffey, Steve Lee, Stewart Hall, Suchir Balaji, Tal Broda, Tal Stramer, Tao Xu, Tarun Gogineni, Taya Christianson, Ted Sanders, Tejal A. Patwardhan, Thomas Cunninghman, Thomas Degry, Thomas Dimson, Thomas Raoux, Thomas Shadwell, Tianhao Zheng, Todd Underwood, Todor Markov, Toki Sherbakov, Tom Rubin, Tom Stasi, Tomer Kaftan, Tristan Heywood, Troy Peterson, Tyce Walters, Tyna Eloundou, Valerie Qi, Veit Moeller, Vinnie Monaco, Vishal Kuo, Vlad Fomenko, Wayne Chang, Weiyi Zheng, Wenda Zhou, Wesam Manassra, Will Sheu, Wojciech Zaremba, Yash Patil, Yilei Qian, Yongjik Kim, Youlong Cheng, Yu Zhang, Yuchen He, Yuchen Zhang, Yujia Jin, Yunxing Dai, and Yury Malkov. Gpt-4o system card. 2024. 2, 3

- [27] Peng Jin, Ryuichi Takanobu, Wancai Zhang, Xiaochun Cao, and Li Yuan. Chat-univi: Unified visual representation empowers large language models with image and video understanding. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024, pages 13700–13710. IEEE, 2024. 3
- [28] Chuyi Kong, Ziyang Luo, Hongzhan Lin, Zhiyuan Fan, Yaxin Fan, Yuxi Sun, and Jing Ma. From general to specific: Utilizing general hallucation to automatically measure the role relationship fidelity for specific role-play agents. 2024. 4
- [29] Bo Li, Yuanhan Zhang, Liangyu Chen, Jinghao Wang, Jingkang Yang, and Ziwei Liu. Otter: A multi-modal model with in-context instruction tuning. CoRR, abs/2305.03726,

2023. 3

- [30] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. Llava-onevision: Easy visual task transfer. CoRR, abs/2408.03326, 2024. 2, 3
- [31] Dongxu Li, Yudong Liu, Haoning Wu, Yue Wang, Zhiqi Shen, Bowen Qu, Xinyao Niu, Guoyin Wang, Bei Chen, and Junnan Li. Aria: An open multimodal native mixture-ofexperts model. 2024. 2, 3
- [32] Junnan Li, Dongxu Li, Silvio Savarese, and Steven C. H. Hoi. BLIP-2: bootstrapping language-image pre-training with frozen image encoders and large language models. In International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA, pages 19730–

19742. PMLR, 2023. 2

- [33] Junkai Li, Siyu Wang, Meng Zhang, Weitao Li, Yunghwei Lai, Xinhui Kang, Weizhi Ma, and Yang Liu. Agent hospital: A simulacrum of hospital with evolvable medical agents. CoRR, abs/2405.02957, 2024. 4
- [34] Kunchang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. Videochat: Chat-centric video understanding. CoRR, abs/2305.06355,

2023. 3

- [35] Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Lou, Limin Wang, and Yu Qiao. Mvbench: A comprehensive multimodal video understanding benchmark. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR

2024, Seattle, WA, USA, June 16-22, 2024, pages 22195–

22206. IEEE, 2024. 2, 3

- [36] Tianle Li, Wei-Lin Chiang, Evan Frick, Lisa Dunlap, Tianhao Wu, Banghua Zhu, Joseph E. Gonzalez, and Ion Stoica. From crowdsourced data to high-quality benchmarks: Arena-hard and benchbuilder pipeline, 2024. 4, 5
- [37] Zehan Li, Xin Zhang, Yanzhao Zhang, Dingkun Long, Pengjun Xie, and Meishan Zhang. Towards general text embeddings with multi-stage contrastive learning. arXiv preprint arXiv:2308.03281, 2023. 6
- [38] Bin Lin, Yang Ye, Bin Zhu, Jiaxi Cui, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united visual representation by alignment before projection. CoRR, abs/2311.10122,

- 2023. 3

[39] Ji Lin, Hongxu Yin, Wei Ping, Pavlo Molchanov, Mohammad Shoeybi, and Song Han. VILA: on pre-training for visual language models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024, pages 26679–26689. IEEE,

- 2024. 3

- [40] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023. 2, 3
- [41] Jiajun Liu, Yibing Wang, Hanghang Ma, Xiaoping Wu, Xiaoqi Ma, Xiaoming Wei, Jianbin Jiao, Enhua Wu, and Jie Hu. Kangaroo: A powerful video-language model supporting long-context video input. CoRR, abs/2408.15542, 2024.

- 3

[42] Ruyang Liu, Chen Li, Haoran Tang, Yixiao Ge, Ying Shan, and Ge Li. ST-LLM: large language models are effective temporal learners. In Computer Vision - ECCV 2024 - 18th European Conference, Milan, Italy, September 29-October

- 4, 2024, Proceedings, Part LVII, pages 1–18. Springer, 2024. 3

- [43] Yuanxin Liu, Shicheng Li, Yi Liu, Yuxiang Wang, Shuhuai Ren, Lei Li, Sishuo Chen, Xu Sun, and Lu Hou. Tempcompass: Do video llms really understand videos? In Findings of the Association for Computational Linguistics, ACL 2024, Bangkok, Thailand and virtual meeting, August 11-16, 2024, pages 8731–8772. Association for Computational Linguistics, 2024. 3
- [44] Zuyan Liu, Yuhao Dong, Ziwei Liu, Winston Hu, Jiwen Lu, and Yongming Rao. Oryx MLLM: on-demand spatialtemporal understanding at arbitrary resolution. CoRR, abs/2409.12961, 2024. 3
- [45] Yujie Lu, Dongfu Jiang, Wenhu Chen, William Yang Wang, Yejin Choi, and Bill Yuchen Lin. Wildvision: Evaluating vision-language models in the wild with human preferences. CoRR, abs/2406.11069, 2024. 1, 2, 3, 5
- [46] Ruipu Luo, Ziwang Zhao, Min Yang, Junwei Dong, Minghui Qiu, Pengcheng Lu, Tao Wang, and Zhongyu Wei. Valley: Video assistant with large language model enhanced ability. CoRR, abs/2306.07207, 2023. 3
- [47] Ziyang Luo, Can Xu, Pu Zhao, Qingfeng Sun, Xiubo Geng, Wenxiang Hu, Chongyang Tao, Jing Ma, Qingwei Lin, and

- Daxin Jiang. Wizardcoder: Empowering code large language models with evol-instruct. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024. 4
- [48] Chenyang Lyu, Minghao Wu, Longyue Wang, Xinting Huang, Bingshuai Liu, Zefeng Du, Shuming Shi, and Zhaopeng Tu. Macaw-llm: Multi-modal language modeling with image, audio, video, and text integration. CoRR, abs/2306.09093, 2023. 3
- [49] Karttikeya Mangalam, Raiymbek Akshulakov, and Jitendra Malik. Egoschema: A diagnostic benchmark for very longform video language understanding. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023. 3
- [50] Viorica Patraucean, Lucas Smaira, Ankush Gupta, Adri`a Recasens, Larisa Markeeva, Dylan Banarse, Skanda Koppula, Joseph Heyward, Mateusz Malinowski, Yi Yang, Carl Doersch, Tatiana Matejovicova, Yury Sulsky, Antoine Miech, Alexandre Fr´echette, Hanna Klimczak, Raphael Koster, Junlin Zhang, Stephanie Winkler, Yusuf Aytar, Simon Osindero, Dima Damen, Andrew Zisserman, and Jo˜ao Carreira. Perception test: A diagnostic benchmark for multimodal video models. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023. 3
- [51] Machel Reid, Nikolay Savinov, Denis Teplyashin, Dmitry Lepikhin, Timothy P. Lillicrap, Jean-Baptiste Alayrac, Radu Soricut, Angeliki Lazaridou, Orhan Firat, Julian Schrittwieser, Ioannis Antonoglou, Rohan Anil, Sebastian Borgeaud, Andrew M. Dai, Katie Millican, Ethan Dyer, Mia Glaese, Thibault Sottiaux, Benjamin Lee, Fabio Viola, Malcolm Reynolds, Yuanzhong Xu, James Molloy, Jilin Chen, Michael Isard, Paul Barham, Tom Hennigan, Ross McIlroy, Melvin Johnson, Johan Schalkwyk, Eli Collins, Eliza Rutherford, Erica Moreira, Kareem Ayoub, Megha Goel, Clemens Meyer, Gregory Thornton, Zhen Yang, Henryk Michalewski, Zaheer Abbas, Nathan Schucher, Ankesh Anand, Richard Ives, James Keeling, Karel Lenc, Salem Haykal, Siamak Shakeri, Pranav Shyam, Aakanksha Chowdhery, Roman Ring, Stephen Spencer, Eren Sezener, and et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. CoRR, abs/2403.05530, 2024. 2, 3
- [52] Michael S. Ryoo, Honglu Zhou, Shrikant B. Kendre, Can Qin, Le Xue, Manli Shu, Silvio Savarese, Ran Xu, Caiming Xiong, and Juan Carlos Niebles. xgen-mm-vid (blip-3video): You only need 32 tokens to represent a video even in vlms. 2024. 3
- [53] Enxin Song, Wenhao Chai, Guanhong Wang, Yucheng Zhang, Haoyang Zhou, Feiyang Wu, Haozhe Chi, Xun Guo, Tian Ye, Yanting Zhang, Yan Lu, Jenq-Neng Hwang, and Gaoang Wang. Moviechat: From dense token to sparse memory for long video understanding. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024,

- Seattle, WA, USA, June 16-22, 2024, pages 18221–18232. IEEE, 2024. 3
- [54] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, Aur´elien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language models. CoRR, abs/2302.13971, 2023. 2
- [55] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton-Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aur´elien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and finetuned chat models. CoRR, abs/2307.09288, 2023. 2
- [56] Laurens van der Maaten and Geoffrey Hinton. Visualizing data using t-sne. Journal of Machine Learning Research, 9

(86):2579–2605, 2008. 6

- [57] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. CoRR, abs/2409.12191, 2024. 2, 3
- [58] Weihan Wang, Zehai He, Wenyi Hong, Yean Cheng, Xiaohan Zhang, Ji Qi, Shiyu Huang, Bin Xu, Yuxiao Dong, Ming Ding, and Jie Tang. Lvbench: An extreme long video understanding benchmark. CoRR, abs/2406.08035, 2024. 3
- [59] Haoning Wu, Dongxu Li, Bei Chen, and Junnan Li. Longvideobench: A benchmark for long-context interleaved video-language understanding. CoRR, abs/2407.15754,

2024. 2, 3

- [60] Junbin Xiao, Xindi Shang, Angela Yao, and Tat-Seng Chua. Next-qa: Next phase of question-answering to explaining temporal actions. In IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2021, virtual, June 1925, 2021, pages 9777–9786. Computer Vision Foundation / IEEE, 2021. 3
- [61] Can Xu, Qingfeng Sun, Kai Zheng, Xiubo Geng, Pu Zhao, Jiazhan Feng, Chongyang Tao, Qingwei Lin, and Daxin Jiang. Wizardlm: Empowering large pre-trained language models to follow complex instructions. In The Twelfth International Conference on Learning Representations, ICLR

2024, Vienna, Austria, May 7-11, 2024. OpenReview.net,

2024. 4

- [62] Dejing Xu, Zhou Zhao, Jun Xiao, Fei Wu, Hanwang Zhang, Xiangnan He, and Yueting Zhuang. Video question answering via gradually refined attention over appearance and motion. In ACM Multimedia, 2017. 3
- [63] Lin Xu, Yilin Zhao, Daquan Zhou, Zhijie Lin, See-Kiong Ng, and Jiashi Feng. Pllava : Parameter-free llava extension from images to videos for video dense captioning. CoRR, abs/2404.16994, 2024. 3
- [64] Fuzhao Xue, Yukang Chen, Dacheng Li, Qinghao Hu, Ligeng Zhu, Xiuyu Li, Yunhao Fang, Haotian Tang, Shang Yang, Zhijian Liu, Ethan He, Hongxu Yin, Pavlo Molchanov, Jan Kautz, Linxi Fan, Yuke Zhu, Yao Lu, and Song Han. Longvila: Scaling long-context visual language models for long videos. CoRR, abs/2408.10188, 2024.
- [65] Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He, Qianyu Chen, Huarong Zhou, Zhensheng Zou, Haoye Zhang, Shengding Hu, Zhi Zheng, Jie Zhou, Jie Cai, Xu Han, Guoyang Zeng, Dahai Li, Zhiyuan Liu, and Maosong Sun. Minicpm-v: A GPT-4V level MLLM on your phone. CoRR, abs/2408.01800, 2024.
- [66] Jiabo Ye, Haiyang Xu, Haowei Liu, Anwen Hu, Ming Yan, Qi Qian, Ji Zhang, Fei Huang, and Jingren Zhou. mplugowl3: Towards long image-sequence understanding in multimodal large language models. CoRR, abs/2408.04840, 2024.
- [67] En Yu, Liang Zhao, Yana Wei, Jinrong Yang, Dongming Wu, Lingyu Kong, Haoran Wei, Tiancai Wang, Zheng Ge, Xiangyu Zhang, and Wenbing Tao. Merlin: Empowering multimodal llms with foresight minds. In Computer Vision - ECCV 2024 - 18th European Conference, Milan, Italy, September 29-October 4, 2024, Proceedings, Part IV, pages 425–443. Springer, 2024. 3
- [68] Zhou Yu, Dejing Xu, Jun Yu, Ting Yu, Zhou Zhao, Yueting Zhuang, and Dacheng Tao. Activitynet-qa: A dataset for understanding complex web videos via question answering. In The Thirty-Third AAAI Conference on Artificial Intelligence, AAAI 2019, The Thirty-First Innovative Applications of Artificial Intelligence Conference, IAAI 2019, The Ninth AAAI Symposium on Educational Advances in Artificial Intelligence, EAAI 2019, Honolulu, Hawaii, USA, January 27

- February 1, 2019, pages 9127–9134. AAAI Press, 2019. 3

- [69] Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023 - System Demonstrations, Singapore, December 6-10, 2023, pages 543–553. Association for Computational Linguistics, 2023. 3
- [70] Hongjie Zhang, Yi Liu, Lu Dong, Yifei Huang, Zhen-Hua Ling, Yali Wang, Limin Wang, and Yu Qiao. Movqa: A benchmark of versatile question-answering for long-form movie understanding. CoRR, abs/2312.04817, 2023. 3
- [71] Peiyuan Zhang, Kaichen Zhang, Bo Li, Guangtao Zeng, Jingkang Yang, Yuanhan Zhang, Ziyue Wang, Haoran Tan, Chunyuan Li, and Ziwei Liu. Long context transfer from language to vision. CoRR, abs/2406.16852, 2024. 3

- [72] Shudan Zhang, Hanlin Zhao, Xiao Liu, Qinkai Zheng, Zehan Qi, Xiaotao Gu, Yuxiao Dong, and Jie Tang. NaturalCodeBench: Examining coding performance mismatch on HumanEval and natural user queries. In Findings of the Association for Computational Linguistics: ACL 2024, pages 7907–7928, Bangkok, Thailand, 2024. Association for Computational Linguistics. 2
- [73] Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li. Video instruction tuning with synthetic data. 2024. 2, 3
- [74] Ruochen Zhao, Wenxuan Zhang, Yew Ken Chia, Deli Zhao, and Lidong Bing. Auto arena of llms: Automating LLM evaluations with agent peer-battles and committee discussions. CoRR, abs/2405.20267, 2024. 2
- [75] Wenting Zhao, Xiang Ren, Jack Hessel, Claire Cardie, Yejin Choi, and Yuntian Deng. Wildchat: 1m chatgpt interaction logs in the wild. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024. 2
- [76] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging llm-as-a-judge with mt-bench and chatbot arena. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023. 4, 5
- [77] Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou. Instruction-following evaluation for large language models. CoRR, abs/2311.07911, 2023. 2
- [78] Junjie Zhou, Yan Shu, Bo Zhao, Boya Wu, Shitao Xiao, Xi Yang, Yongping Xiong, Bo Zhang, Tiejun Huang, and Zheng Liu. MLVU: A comprehensive benchmark for multi-task long video understanding. CoRR, abs/2406.04264, 2024. 3

### A. Prompts

- A.1. User Simulation

In Figure 10 and 11, we include the prompts for video content-constrained persona generation and personaconstrained video question asking.

- A.2. Fault-Driven Evolution

- Figure 12 includes the prompt for the fault-driven evolution.

A.3. Automatic Judging

- Figure 13 includes the prompt for the automatic judging. Our VideoAutoBench adopts the same prompt for judging.

A.4. Difficulty Level Evaluation

- Figure 14 includes the prompt for the complexity evaluation.

### B. Experimental Details

#### B.1. Statistics

In VideoAutoArena, Model A wins 5,620 battles, Model B wins 5,941 battles, and 918 ties. VideoAutoBench consists of 255 samples corresponding to 244 unique videos, with an average duration of 478.5 seconds. The duration distribution includes 62 videos for 8-15s, 62 for 15-60s, 60 for 180-600s, and 60 for 900-3,600s. The samples span 10 categories: 29 from Movies, 50 from Life Vlogs, 9 from Geography, 13 from History, 12 from News Programs, 9 from Art, 6 from STEM, 8 from Computer Science, 55 from Cooking Recipes, and 53 from Travel Guides.

##### B.2. LMMs Selection In our experiments, we evaluate 11 SOTA LMMs:

- 1. gpt-4o-2024-05-13,
- 2. gpt-4o-mini-2024-07-18,
- 3. gemini-1.5-pro,
- 4. gemini-1.5-flash,
- 5. rhymes-ai/Aria,
- 6. Qwen/Qwen2-VL-72B-Instruct,
- 7. Qwen/Qwen2-VL-7B-Instruct
- 8. lmms-lab/LLaVA-Video-72B-Qwen2,
- 9. lmms-lab/LLaVA-Video-7B-Qwen2,
- 10. lmms-lab/llava-onevision-qwen2-72b-ov,
- 11. lmms-lab/llava-onevision-qwen2-7b-ov. We selected the top 10 LMMs that support long video analysis, along with their smaller versions, based on their performance on VideoMME as of October 15, 2024. Additional open-source LMMs were excluded for two reasons: first, some were released concurrently with our work, leaving insufficient time for evaluation; second, others exhibited weak performance on VideoMME or lacked support

for long video analysis. Consequently, our experiments are limited to the 11 most popular and SOTA LMMs.

For user simulation, fault-driven evolution, automatic judging, and difficulty level evaluation, we use gpt-4o-2024-08-06, ensuring that the examiner and judge remain distinct from the LMMs throughout the entire evaluation process.

#### B.3. Hyperparameters

For user simulation, fault-driven evolution, automatic judging, and difficulty level evaluation, each video is uniformly sampled into a maximum of 128 frames, while response generation uses up to 64 frames. Each frame is resized to 512 × 512. For API-based LMMs, the max tokens parameter is set to 4096, with other settings using default values. For open-source LMMs, the temperature is set to 0.7, and max new tokens is limited to 2048.

### C. Examples

- C.1. Persona

As shown in Figure 15, we include 15 examples of different personas for 5 different videos.

- C.2. Question

- As shown in Figure 15, we include 15 examples of different questions for 5 different videos.

C.3. Responses and Judging

- As shown in Figure 16, 17, 18, and 19, we include the battle examples between different models.

### D. Human Annotations

- D.1. Question Ranking

- In Figure 20, we include our guideline for the question ranking annotation.

D.2. Judging

- In Figure 21, we include our guideline for the response judging annotation.

### E. Limitation

VideoAutoArena and VideoAutoBench currently lack evaluations for multi-turn and non-English interactions, primarily due to the limited multi-turn conversational capabilities and restricted non-English proficiency of current opensourced LMMs. Moreover, the automatic judging system tends to favor detailed responses, a preference also observed in human evaluations. While detailed responses are often more helpful to users, this introduces challenges in evaluating LMMs. We implemented the style-control method from

You are watching a video. Your task is to create three distinct user personas who might be interested in this video:

- - Persona 1: A user whose background is **closely related** to the video's content.
- - Persona 2: A user whose background is **less related** but who may still develop an interest in learning more about the video's content.
- - Persona 3: A user whose background is **totally not related** but who may still develop an interest in learning more about the video's content. Keep each persona description brief, using only a short paragraph. Please adhere to the following format:

- P1 [Persona 1]: A person xxx
- P2 [Persona 2]: A person xxx
- P3 [Persona 3]: A person xxx

- Figure 10. The prompt for video content-constrained persona generation.

You are given a list of image frames from a video, and you **MUST** pretend that you are watching the video. You will act as a user with the following persona: ```persona {persona} ```

Based on your persona, your task is to craft high-quality [Questions] for Video Question Answering (Video QA). You are required to watch a video. Your [Question] should be based on the video and seek assistance from the AI to analyze its content. **The [Question] must reflect your persona.** To create the [Question], please follow the steps below:

- Step 1: Watch the video carefully and craft a [Question] that aligns with the user's persona. The [Question] should only be answerable by someone who has watched the video. You **CANNOT** reveal any key visual details in the [Question]. If necessary, include response format constraints (Markdown, JSON, Table, List, etc.) in the [Question].
- Step 2: Watch the video carefully and provide a [Response] to the [Question]. The [Response] should be clear and organized, fully addressing the persona's needs without omitting important information. It should provide a targeted answer based on the user's perspective. Please reply strictly in the following format:

- Step 1 [Question]:
- Step 2 [Response]:

- Figure 11. The prompt for persona-constrained video question asking.

LMSYS Chatbot Arena to adjust ELO ratings by penalizing stylistic factors. However, we found this approach unsuitable for evaluating current LMMs. For example, while Aria and Qwen2-VL-72B outperform Gemini-1.5-Pro in ELO ratings, Gemini-1.5-Pro consistently achieves significantly higher win rates. Manual review of Gemini-1.5-Pro’s outputs revealed that its responses are not only more detailed but also of higher quality compared to those from the two open-source LMMs. Additionally, most open-source LMMs tend to generate less detailed responses than proprietary LMMs, causing the style-control mechanism to disproportionately penalize proprietary models. This imbalance leads to unfair evaluations. To address this issue, a more effective style-control mechanism should ensure that competing LMMs produce responses with comparable levels of detail, thereby enabling a fairer evaluation.

To address these limitations, future work will focus on expanding VideoAutoArena and VideoAutoBench to include multiturn and multilingual data. Additionally, we

aim to refine our definitions of user simulation, developing a systematic approach for generating battles that encompass a broader and more inclusive range of scenarios while maintaining high separability and alignment with human judgment. Furthermore, we plan to explore advanced style-control techniques and unbiased LMMs-as-judge to further enhance the robustness and fairness of our LMMbased evaluation framework.

**Remember: You are watching a Video.** A user, characterized by a specific persona, is interacting with two AI assistant models (A and B) to better understand video content using the same question. Here is the user's persona: ```persona {persona} ``` The user's question is: ```question {question} ```

- The response from Model A is:

- ```model_a

- {answer_a} ```

The response from Model B is: ```model_b

- {answer_b} ```

Your task is to carefully evaluate the responses of Model A and Model B to identify any faults and models' weaknesses. Based on these weaknesses, generate a harder [New Question] that explores aspects where the models may struggle. **Remember**, the [New Question] should continue to align with the user's persona, without necessarily being more specific.

You **MUST** follow these steps to generate the [New Question]:

- Step 1: Carefully review the chat history to identify any problems in the two models' responses related to understanding the video's content. Consider:

- - Whether the responses correctly incorporate information from the video.
- - Whether the responses are helpful in fulfilling the user's requirements.
- - Whether the responses are user-aware and align with the user's persona. List all identified faults in [Fault List A] for Model A and [Fault List B] for Model B.

- Step 2: Based on [Fault List A] and [Fault List B], summarize the weaknesses of Model A and Model B. List all identified weaknesses in [Weakness List A] for Model A and [Weakness List B] for Model B. The weaknesses should be more general than the faults.
- Step 3: Based on [Weakness List A] and [Weakness List B], craft a high-quality harder [New Question] that will further explore aspects where the two models may struggle. If both responses do not contain any faults, you still need to create a different harder [New Question] that mimics the interaction between the user and the AI assistant models. The [New Question] is **not** a follow-up question.

**Remember**, the [New Question] should continue to align with the user's persona, without necessarily being more specific. Ensure that:

- - The [New Question] can only be answered by someone who has watched the video.
- - Do **NOT** leak key visual details in the [New Question]. If necessary, include response format requirements (Markdown, JSON, Table, List, etc.) in the [New Question]. Please respond strictly in this format:

- Step 1:

- ```[Fault List A] xxx ```
- ```[Fault List B] xxx ```

- Step 2:

- ```[Weakness List A] xxx ```
- ```[Weakness List B] xxx ```

- Step 3: ```[New Question] Only include the new question here! ```

Figure 12. The prompt for our fault-driven evolution generates new questions based on the responses from the two models.

**Remember: You are watching a Video.** A user, characterized by a specific persona, is interacting with two AI assistant models (A and B) to better understand video content using the same question. Here is the user's persona: ```persona {persona} ``` The user's question is: ```question {question} ``` The response from Model A is: ```model_a {answer_a} ``` The response from Model B is: ```model_b {answer_b} ``` Please act as an impartial judge and carefully evaluate the responses of Model A and Model B to determine which one is better. Use the following standards:

- 1. [Instruction Following]: The response should closely adhere to the user's instructions, ensuring it directly addresses the specified task.
- 2. [Accuracy]: The response must accurately utilize information from the video, avoiding fabrication or misquotation. It should maintain factual correctness, avoid hallucinations, and demonstrate contextual coherence with precise terminology and knowledge.
- 3. [Relevance]: The response should consider the user's background information and needs, providing a comprehensive, detailed answer that addresses the question directly without straying off-topic. Responses should be thorough, offering multiple perspectives where relevant.
- 4. [Helpfulness]: The response should provide valuable information to aid the user in understanding or solving their issue, avoiding irrelevant or vague content. If the responses from Model A and Model B are of similar quality (whether both are good or both are bad), you may declare a tie.

**Please follow these steps for your judgment:**

- - Step 1: Analyze which model provides a better response for the [Instruction Following] standard.
- - Step 2: Analyze which model provides a better response for the [Accuracy] standard.
- - Step 3: Analyze which model provides a better response for the [Relevance] standard.
- - Step 4: Analyze which model provides a better response for the [Helpfulness] standard.
- - Step 5: Based on the results from Steps 1-4, determine the overall outcome: Model A, Model B, Tie (both are good), or Tie (both are bad). Please respond strictly in the following format:

```[Instruction Following] [Your Analysis] ```

```[Accuracy] [Your Analysis] ```

```[Relevance] [Your Analysis] ```

```[Helpfulness] [Your Analysis] ```

```[Overall Judge] A/B/Tie ```

Figure 13. The prompt for our automatic judging.

**Remember: You are watching a Video.** A user, characterized by a specific persona, is seeking help from an AI assistant to better understand video content. Here is the user's persona: ```persona {persona} ``` The first question is:

- ```Question A

- {question_a} ```

The second question is: ```Question B

- {question_b} ``` Please act as an impartial evaluator and assess the difficulty level of these two questions based on the following criteria (1 = easiest, 5 = hardest):

- 1. [Instruction Following]: Rate the complexity (1-5) involved in accurately following the question’s instructions.
- 2. [Accuracy]: Rate the challenge (1-5) in using specific information from the video to address the question without errors or fabrications.
- 3. [Relevance]: Rate the difficulty (1-5) of tailoring the response to the user's background and needs, ensuring alignment with the context.
- 4. [Helpfulness]: Rate the level of difficulty (1-5) in providing information that is genuinely valuable and directly aids the user, avoiding vague or irrelevant content." Please respond strictly in the following format:

- 1. Question A

- ```[Question A Instruction Following Difficulty Level] [Your Score] ```

- ```[Question A Accuracy Difficulty Level] [Your Score] ```

- ```[Question A Relevance Difficulty Level] [Your Score] ```

- ```[Question A Helpfulness Difficulty Level] [Your Score] ```

- ```[Question A Overall Difficulty Level] [Overall Difficulty Score 1-5] ```

2. Question B

- ```[Question B Instruction Following Difficulty Level] [Your Score] ```

- ```[Question B Accuracy Difficulty Level] [Your Score] ```

- ```[Question B Relevance Difficulty Level] [Your Score] ```

- ```[Question B Helpfulness Difficulty Level] [Your Score] ```

- ```[Question B Overall Difficulty Level] [Overall Difficulty Score 1-5] ```

Figure 14. The prompt for question complexity evaluation.

- Persona 1

A person who enjoys crime series and thrillers but hasn't seen "Money Heist" yet. They are curious about popular shows and are looking for recommendations. This recap video might pique their interest and prompt them to start watching the series.

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

- Persona 2

A person who primarily watches comedies and documentaries. They generally avoid intense dramas, but a friend recommended "Money Heist" as an exception. This video could introduce them to the show's narrative and possibly lead them to explore a new genre.

- Persona 3

- Question 1

What was the Professor's strategy in executing the heist in "Money Heist," and how did it involve the team and hostages?

- Question 2

Considering I usually enjoy comedies and documentaries, could you tell me how the initial episodes of "Money Heist" set up the heist and the team dynamics without getting too intense or dramatic? I'm curious about the plan and the characters but want to know if it aligns with my taste.

- Question 3

In the video recap of "Money Heist," how does the Professor ensure that the police and public sympathize with the robbers' cause, and what strategies are highlighted to demonstrate the calculated nature of the heist plan? Include character interactions or decisions that emphasize these strategies.

A person who is a fan of heist dramas and has watched "Money Heist" multiple times. They enjoy analyzing plot twists and character development, and they are always on the lookout for video recaps or discussions to deepen their understanding of the show.

[Figure 76]

- Persona 1

A person with a background in robotics engineering, working primarily on autonomous vehicles. Although their expertise is more focused on mechanical and systems engineering, they are keen to expand their knowledge in computer vision and LiDAR-integrated systems to enhance navigation and tracking capabilities in robotics.

- Persona 2

A person with a background in business management, currently working in the tech industry. Despite having no technical expertise in computer vision or LiDAR technologies, they are interested in learning about cutting-edge advancements in technology to identify potential opportunities for business expansion and innovation.

- Persona 3

###### Question 1

[Figure 77]

A person with a background in computer science, specializing in computer vision and 3D object tracking. They are actively engaged in research related to LiDAR technology and multi-object tracking systems, seeking to implement state-of-the-art techniques like point cloud-based tracking in their projects.

Based on the video presentation, what are the key contributions and results of the PC-DAN (Point Cloud Deep Affinity Network) in 3D multi-object tracking, especially concerning the datasets mentioned? Please summarize in a list format.

[Figure 78]

###### Question 2

How does the PC-DAN method for 3D multi-object tracking utilize point clouds, and what advantages does it offer for autonomous vehicle navigation systems compared to traditional 2D tracking methodologies?

[Figure 79]

###### Question 3

[Figure 80]

Based on the presentation, how does the PC-DAN model for 3D multi-object tracking compare in performance to other models when tested on the JRDB dataset? Can you highlight any key business opportunities or innovations that might arise from its application in real-world scenarios?

[Figure 81]

- Persona 1

A person who works in an urban setting but is enthusiastic about transitioning to a more sustainable lifestyle. They are curious about connecting with nature and finding new hobbies like trying cultural recipes and learning about myths, even though it's different from their usual routine.

- Persona 2

A person working in a fast-paced corporate job, typically focused on technology and modern conveniences. They stumble upon this video, becoming intrigued by the idea of slowing down and appreciating nature, new cultures, and more traditional ways of living as an escape from their daily hustle.

- Persona 3

###### Question 1

[Figure 82]

A person who lives in a rural area, enjoys simple living, and has a deep interest in traditional crafts, herbalism, and folklore. They appreciate a slow-paced lifestyle and enjoy making homemade tea blends and exploring stories from different cultures.

How does the video illustrate the experience of exploring and sourcing herbal ingredients for tea blends and the connection to folklore and personal stories during the winter season?

[Figure 83]

###### Question 2

How does the video illustrate the process of connecting with nature and embracing cultural traditions through activities like tea making and cooking, and what impacts do these activities have on personal well-being according to the narrator?

[Figure 84]

###### Question 3

[Figure 85]

How does the person in the video incorporate elements of nature and traditional living into their daily routine, and what impact does this have on their sense of balance and well-being? I'm curious to explore how stepping away from a technology-driven lifestyle can lead to a more fulfilling life. Please provide the response in a detailed list format.

[Figure 86]

- Persona 1

A person who enjoys dabbling in various DIY projects and has a budding interest in art. They are keen to explore new crafts and may get inspired to try drawing and creating personalized postcards as a new hobby.

- Persona 2

A person who primarily has a background in finance but is looking for a relaxing and creative outlet. They stumble upon this art tutorial video and find the idea of drawing and painting as a form of stress relief intriguing and worth trying out.

- Persona 3

###### Question 1

[Figure 87]

A person who is an art teacher looking for creative ways to teach postcard design and landscape drawing to their students. They are interested in finding engaging tutorials to enhance their class curriculum.

In the video, can you provide a summary of the step-by-step process shown for designing and sketching a landscape postcard? I'm interested in understanding the techniques and tools used as I plan to incorporate similar methods into my art classes.

[Figure 88]

###### Question 2

In the video, the creator is making art inspired by an old postcard. Can you describe the process they use to create their artwork and what materials are involved? How might this inspire someone interested in DIY art projects?

[Figure 89]

###### Question 3

[Figure 90]

As someone new to art and looking for a creative stress relief, how did the video guide the process of drawing and painting a postcard, and what materials were used throughout the tutorial?

[Figure 91]

- Persona 1

A person who is a criminal justice student curious about different aspects of the legal system, including how witness protection operates. They are eager to learn beyond textbooks, especially about real-world applications like WITSEC.

- Persona 2

A person who is a fiction writer looking for inspiration for their next crime thriller. They seek to add realism to their storylines by exploring how witness protection programs function and the dynamics involved in protecting witnesses.

- Persona 3

###### Question 1

[Figure 92]

A person who is a current or former law enforcement officer with experience in witness protection programs. They have a strong interest in understanding the logistical and emotional challenges involved in relocating and protecting high-risk individuals.

In the context of the witness protection program depicted in the video, what logistical measures are in place to ensure the safety and integration of witnesses, and how do these measures address both security and emotional support needs?

[Figure 93]

###### Question 2

Can you describe how the video highlights WITSEC's approach to maintaining participants' safety and their integration into new communities, as inferred from the visual elements and narration?

[Figure 94]

###### Question 3

[Figure 95]

What elements of the witness protection program were highlighted as essential services in the video, and how do they contribute to the overall safety and acclimation of participants into new communities?

[Figure 96]

Figure 15. Examples of our user simulation include five videos from diverse domains: Movies, Computer Science, Life Vlogs, Art, and News Programs. To save space, we only showcase 4 frames of each video.

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

Persona: A person with a background in robotics engineering, working primarily on autonomous vehicles. Although their expertise is more focused on mechanical and systems engineering, they are keen to expand their knowledge in computer vision and LiDAR-integrated systems to enhance navigation and tracking capabilities in robotics. Question: How does the PC-DAN method for 3D multi-object tracking utilize point clouds, and what advantages does it offer for autonomous vehicle navigation systems compared to traditional 2D tracking methodologies?

Model A: Aria

The PC-DAN (Point Cloud-based Deep Affinity Network) method for 3D multi-object tracking leverages point cloud data, which captures 3D information, to enhance tracking accuracy and robustness. This approach extends previous 2D tracking methodologies by integrating deep learning techniques to handle the complexities of 3D spatial data. The system employs a VGG network for feature extraction from point clouds and utilizes a deep affinity network to associate features across frames, ensuring accurate tracking of multiple objects.

One of the primary advantages of PC-DAN over traditional 2D tracking methods is its ability to preserve structural information and avoid perspective distortions, which are common in 2D data. This is particularly beneficial for autonomous vehicle navigation systems, as it allows for more reliable detection and tracking of objects in 3D space, leading to safer and more efficient navigation. Additionally, the use of point clouds enables the system to handle varying perspectives and occlusions more effectively, further improving performance in dynamic environments.

###### Model B: GPT-4o

The PC-DAN (Point Cloud based Deep Affinity Network) method for 3D multi-object tracking leverages point cloud data to enhance tracking performance in scenarios such as autonomous vehicle navigation. Here's an overview of how it utilizes point clouds and its advantages:

### Utilization of Point Clouds:

- 1. **Feature Extraction:**

- PC-DAN uses the PointNet architecture to process point cloud data. PointNet extracts features from the raw point cloud, capturing 3D structural information.

- 2. **Affinity Measurement:**

- The extracted features from consecutive frames are used to measure the affinities between objects in 3D space. This helps in associating objects across different frames accurately.

- 3. **Deep Affinity Network:**

- PC-DAN incorporates a deep affinity network that processes these features through several layers (e.g., VGG, parameter sharing, and compression network) to handle the association tasks.

- 4. **Consistent 3D Tracking:**

- The network generates and maintains consistent 3D object identities by using forward and backward association losses, ensuring robust tracking performance over time. ### Advantages Compared to Traditional 2D Tracking:

- 1. **Enhanced Structural Information:**

- Unlike 2D data, 3D point clouds preserve detailed structural information and are not affected by perspective distortion. This makes it easier to track objects accurately in a threedimensional space.

- 2. **Robustness to Occlusion:**

- 3D data provides a more comprehensive view of the environment, reducing the impact of occlusions and ambiguities that are common with 2D data.

- 3. **Better Spatial Context:**

- Point clouds offer precise spatial context and depth information, crucial for tasks like autonomous navigation where understanding the true distance and relative positions of objects is vital.

- 4. **Adaptability to Dynamic Environments:**

- The use of LIDAR and point cloud data allows better adaptability and robustness in dynamic environments, such as those encountered in self-driving scenarios.

- 5. **Broad Application Use:**

- - LIDAR technology, which feeds point cloud data, is increasingly available in modern automobiles, enhancing the feasibility and scalability of PC-DAN-based systems for real-world

applications. ### Results and Validation:

- - The effectiveness of PC-DAN is validated on datasets like KITTI and JackRabbit (JRDB), showing competitive performance metrics compared to other methods, with high MOTA (MultiObject Tracking Accuracy) scores.

By utilizing these advantages, PC-DAN significantly improves the reliability and accuracy of tracking in autonomous vehicle systems, compared to traditional 2D tracking methodologies that lack depth information and are susceptible to occlusions and distortions.

Automatic Judging

Instruction Following: Both models followed the instructions well, addressing the specifics of how PC-DAN uses point clouds for 3D tracking and comparing it to 2D methods. However, Model B's response is more detailed in outlining the utilization steps and advantages explicitly. Model B is slightly better.

Accuracy: Model A is accurate but less detailed in using specific methodologies like PointNet and the incorporation of various network layers in PC-DAN. Model B accurately reflects the video content, mentioning PointNet, specific network processes, and validation on datasets. Model B is better.

Relevance: Model B provides a more thorough and technical explanation, directly catering to the user's background in autonomous vehicles and interest in LiDAR and point clouds. It also mentions validation datasets which are relevant. Model B is better.

Helpfulness: Model B offers a comprehensive explanation with distinct advantages of the method, catering well to the user's needs to understand 3D tracking systems in autonomous vehicles. It gives more specific technical insights and context. Model B is more helpful.

###### Model B Wins!

Figure 16. Example of the battle between Aria and GPT-4o.

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

Persona: A person who is an art teacher looking for creative ways to teach postcard design and landscape drawing to their students. They are interested in finding engaging tutorials to enhance their class curriculum. Question: In the video, can you provide a summary of the step-by-step process shown for designing and sketching a landscape postcard? I'm interested in understanding the techniques and tools used as I plan to incorporate similar methods into my art classes.

###### Model A: GPT-4o-mini

In the video, the process for designing and sketching a landscape postcard is illustrated step-by-step. Here’s a summary of the techniques and tools used: ### Step-by-Step Process:

- 1. **Materials Preparation**:

- Gather tools such as various colored pencils, a fine-tip pen for outlines, erasers, and a watercolor palette. Paper is divided into sections for planning.

- 2. **Initial Sketch**:

- Start by lightly drawing a basic composition on a sheet of watercolor paper. Reference images (like landscapes) can be placed nearby for inspiration.

- 3. **Detailed Outline**:

- Add details to the initial sketch, refining the shapes, leaves, and any foreground elements like fruits or flowers.

- 4. **Adding Texture**:

- Consider the textures of various elements (such as tree leaves and mountains) and include them in the sketches to convey depth.

- 5. **Incorporating Color**:

- Use watercolor paints to fill in areas, starting with a base layer and gradually adding more vibrant colors. Pay attention to blending techniques for realism.

- 6. **Final Touches**:

- Once the painting dries, outline details with a fine-tip pen to enhance the defined elements and add contrast.

- 7. **Personalization**:

- - Create a postcard format by adding space for a message on the back and potentially a stamp, making the artwork suitable for sending.

### Techniques Used:

- - **Layering**: Build up color gradually for depth.
- - **Blending**: Mixing colors seamlessly on the paper.
- - **Texture Creation**: Using varied strokes to reflect natural patterns. ### Tools Used:
- - **Watercolors**: For vibrant, fluid colors.
- - **Colored Pencils**: For fine detailing and layering.
- - **Fine-Tip Pens**: For outline and finishing touches.

Incorporating these methods into your art classes can encourage students to experiment with both drawing and painting techniques, enhancing their understanding of compositional elements in postcard design.

###### Model B: LLaVa-Vieo-72B

The video begins with a blank sheet of paper, and the artist starts by sketching the basic outline of the landscape. They use a pencil to lightly draw the horizon line, mountains, and trees. The artist then adds more details such as branches and leaves, using a variety of strokes to create texture and depth. After the sketch is complete, the artist uses watercolors to add color to the scene, starting with the sky and moving on to the trees and mountains. The final step involves adding a personal message and address to the postcard, turning it into a functional piece of art.

Automatic Judging

Instruction Following: Model A provides a detailed step-by-step process as requested, while Model B gives a more general overview. Model A follows the instructions more closely by breaking down the process into specific steps and techniques.

Accuracy: Model A accurately describes the steps and tools used, such as colored pencils, watercolors, and fine-tip pens. Model B accurately mentions sketching the landscape first and then adding colors but lacks detail on tools.

Relevance: Model A's response is more relevant as it provides comprehensive details that align with the user\u2019s persona as an art teacher. It gives explicit techniques and tools useful for teaching. Model B, while accurate, lacks depth.

Helpfulness: Model A is more helpful by offering detailed techniques and tool recommendations, which can be directly useful for the user's art classes. Model B is less helpful due to its brevity and lack of detail.

###### Model A Wins!

Figure 17. Example of the battle between GPT-4o-mini and LLaVa-Video-72B.

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

Persona: A person who lives in a rural area, enjoys simple living, and has a deep interest in traditional crafts, herbalism, and folklore. They appreciate a slow-paced lifestyle and enjoy making homemade tea blends and exploring stories from different cultures. Question: How does the video illustrate the experience of exploring and sourcing herbal ingredients for tea blends and the connection to folklore and personal stories during the winter season?

Model A: Qwen2.5-VL-72B

The video captures the process of exploring nature, gathering herbs, and preparing them for use in tea blends. It showcases the connection between the natural world and the traditions of herbalism, highlighting the importance of storytelling and personal experiences in this practice. The video emphasizes the sensory experience of interacting with plants and the significance of passing down knowledge through oral tradition. It provides a glimpse into the rich cultural heritage associated with herbalism and its role in nurturing both physical and emotional wellbeing.

Model B: LLaVa-Video-7B

The video illustrates the experience of exploring and sourcing herbal ingredients for tea blends through the narrator's journey to the Salvation Army, where they find a book on herbs that inspires them. This connection to folklore and personal stories during the winter season is highlighted by the narrator's reflection on their abuela's influence and their own experiences with traditional Puerto Rican recipes and myths. The video shows how these elements come together as the narrator prepares to brew a cup of tea, blending modern self-care practices with cultural heritage.

Automatic Judging

Instruction Following: Model A follows the instructions more closely by focusing on the experience of exploring and sourcing herbal ingredients for tea blends and the connection to folklore and personal stories. Model B focuses on the Salvation Army and a book on herbs, which isn't as prominent in the video.

Accuracy: Model A accurately describes the gathering of herbs and the highlight on folklore and personal experiences, aligning with the video's content. Model B inaccurately describes a visit to the Salvation Army, which is not present in the video.

Relevance: Model A is more relevant as it ties the exploration of herbalism with cultural heritage and storytelling, which resonates with the video's theme. Model B includes details not evident in the video, making it less relevant to the content.

Helpfulness: Model A provides a clearer understanding of the video's focus on herbalism and cultural connections, making it more helpful for understanding the user's interests. Model B introduces unrelated elements, potentially confusing the user.

###### Model A Wins!

###### Figure 18. Example of the battle between Qwen2-VL-72B and LLaVa-Video-7B.

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

Persona: A person who works in the tech industry, with a hobbyist interest in collecting antiques. Although their professional background is far from art, they are captivated by the beauty and craftsmanship of rare historical pieces. They enjoy learning about art through documentaries as a form of relaxation and cultural enrichment. Question: In the video, there is a detailed discussion about a historical artifact known as "The Mantuan Roundel." As someone interested in antiques and their craftsmanship, I'm curious: what is the significance of the materials and techniques used in creating "The Mantuan Roundel," and how do they reflect the artistic practices of the Renaissance period?

Model A: Aria

The Mantuan Roundel is a significant artifact that exemplifies the artistic practices of the Renaissance period. Crafted around 1500, it is made from bronze, partially gilded, and silvered. The intricate detailing and the use of gilding highlight the high level of craftsmanship and the artistic values of the time. The roundel depicts a mythological scene, which was a common theme in Renaissance art, reflecting the period's fascination with classical antiquity. The use of bronze as a medium indicates the importance of durability and the desire to create artworks that would last through the ages. The gilding and silvering techniques showcase the artist's skill in working with precious metals, adding a luxurious and divine quality to the piece. This artifact not only reflects the technical prowess of Renaissance artisans but also their ability to blend classical themes with contemporary artistic innovations.

|"The Mantuan Roundel" is a significant example of Renaissance craftsmanship, showcasing the use of bronze and gold leaf to create intricate details and a sense of depth. The materials and techniques used in its creation, such as the casting process and the application of gold leaf, were indicative of the high level of skill and artistry prevalent during the Renaissance period. These methods not only demonstrate the technical proficiency of the artist but also the cultural value placed on the creation of beautiful and enduring objects.<br><br>Model B: Qwen2-VL-72B|
|---|

Automatic Judging

Instruction Following: Both models follow the instructions well, addressing the user's question about the significance of the materials and techniques used in "The Mantuan Roundel." Model A provides a more detailed explanation of the techniques, such as gilding and silvering, and ties them to Renaissance themes. Model B focuses on materials and the casting process but is less detailed. Model A slightly better follows the instructions with thorough detail.

Accuracy: Model A accurately discusses the materials and techniques used, tying them to Renaissance artistic practices with specific references like gilding and classical themes. Model B accurately identifies the use of bronze and gold leaf but misses specific details compared to Model A. Model A is slightly more accurate due to more detailed information.

Relevance: Model A's response is highly relevant, connecting artistic practices to Renaissance norms and providing detailed context, which matches the user's interest in antiques and craftsmanship. Model B stays relevant too, focusing on craftsmanship but with less depth. Model A is more relevant given the user's interest in detailed craftsmanship.

Helpfulness: Model A provides a deeper context and understanding of the Renaissance artistic practices, offering a comprehensive explanation that aligns with the user's interests. Model B offers useful information but is less detailed in comparison. Model A is more helpful due to its comprehensive and contextual details.

###### Model A Wins!

Figure 19. Example of the battle between Aria and Qwen2-VL-72B.

Each sample includes a video, its video caption, and three questions. From the perspective of a user seeking AI assistance for video analysis, rank these questions. The ranking criterion is whether the question aligns with what a real user interested in the video might ask based on their personas, rather than being a question constructed purely to test the model's capabilities. Questions ranked higher should better meet this standard.

**Example**

- - **Video**: [Video]
- - **Caption:** The video shows a man in a white shirt sitting on a sofa interacting with a doll wearing a hat. In front of them are a few glasses of beer, and several paintings are hanging on the wall. Their conversation seems to cover topics such as World War II history, Cold War information, and the development of games. The man occasionally reads from a piece of paper in his hand, while the doll appears very animated.
- - **Q1:** Can you describe the sequence in which the topics are discussed in the video, including plans for the YouTube channel's future, discussions on the best tank of its era, challenges with creating early videos, and comparisons between different ships and their capabilities?
- - **Q2:** As someone interested in the evolution of military equipment and its impact on modern technology, I'm curious about the main topics discussed in the video regarding military strategy and equipment development. Could you provide a summary in bullet points of the key points and any new insights about the ongoing evolution of military technology mentioned in the video?
- - **Q3:** In front of the wall with the painting hanging on it, there's a chair. In front of the chair, there's a man wearing a white shirt with his left hand raised and a green toy soldier. They are looking at each other. When the subtitle mentions "something they like right yeah well then," what is the man in the shirt doing?

**Ranking:** Q2 > Q1 > Q3

**Reasoning:** Q2 is closer to what a real user might ask about a video they are interested in. Q1 and Q3, on the other hand, are primarily designed to measure specific capabilities of the model.

Figure 20. The guideline for the question ranking annotation.

### Annotation Guidelines Based on the given question and the asker's background information, compare answers A and B across four dimensions and overall: **A, B, Tie Good, Tie Bad**. The four dimensions are **Instruction Following, Accuracy, Relevance, and Helpfulness**. Refer to the following criteria for evaluation: ### Evaluation Dimension Criteria

- #### **1. Instruction Following**

- - **Description:** Ensure the response adheres to the user's instructions.
- - **Criteria:**
- - Fails to meet the core expectations in the user's query or instruction.
- - Errors in context understanding lead to incorrect output. For ambiguous instructions, there is no correct answer in the response.
- - Inaccurate content or factual errors, including hallucinations. Misuse of technical terms or ambiguous expressions.
- - Irrelevant content that does not address the user's query.
- - Incorrect or invalid references in the response.

- #### **2. Accuracy**

- - **Description:** Accurately utilizes information from the video to answer the question without fabrication or misrepresentation.
- - **Criteria:**
- - Generated content must be factually correct and free of hallucinations.
- - The response should be contextually consistent, with accurate and unambiguous use of technical terms and knowledge.

- #### **3. Relevance**

- - **Description:** Considers the asker's background information, ensuring that the output aligns closely with the user's needs and directly addresses the task or question.
- - **Criteria:**
- - Avoids irrelevant or off-topic answers.
- - Responses are complete, detailed, and in-depth, avoiding overly simplistic or single-perspective answers.

- #### **4. Helpfulness**

- - **Description:** The response should effectively meet the user's needs and provide valuable information.
- - **Criteria:**
- - Helps the user better understand or resolve their issue.
- - Avoids providing irrelevant or vague answers.

Figure 21. The guideline for the response judging annotation.

