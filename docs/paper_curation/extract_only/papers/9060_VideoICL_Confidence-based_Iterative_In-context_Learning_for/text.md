## arXiv:2412.02186v1[cs.CV]3Dec2024

### VideoICL: Confidence-based Iterative In-context Learning for Out-of-Distribution Video Understanding

Kangsan Kim1∗ Geon Park1∗ Youngwan Lee1,3 Woongyeong Yeo1 Sung Ju Hwang1,2 1KAIST 2DeepAuto.ai 3ETRI

{kksan07, geon.park, ywlee88, wgcyeo, sjhwang82}@kaist.ac.kr

#### Abstract

Recent advancements in video large multimodal models (LMMs) have significantly improved their video understanding and reasoning capabilities. However, their performance drops on out-of-distribution (OOD) tasks that are underrepresented in training data. Traditional methods like fine-tuning on OOD datasets are impractical due to high computational costs. While In-context learning (ICL) with demonstration examples has shown promising generalization performance in language tasks and image-language tasks without fine-tuning, applying ICL to video-language tasks faces challenges due to the limited context length in Video LMMs, as videos require longer token lengths. To address these issues, we propose VideoICL, a novel video incontext learning framework for OOD tasks that introduces a similarity-based relevant example selection strategy and a confidence-based iterative inference approach. This allows to select the most relevant examples and rank them based on similarity, to be used for inference. If the generated response has low confidence, our framework selects new examples and performs inference again, iteratively refining the results until a high-confidence response is obtained. This approach improves OOD video understanding performance by extending effective context length without incurring high costs. The experimental results on multiple benchmarks demonstrate significant performance gains, especially in domain-specific scenarios, laying the groundwork for broader video comprehension applications. Code will be released at https://github.com/KangsanKim07/VideoICL

#### 1. Introduction

Recent video large multimodal models (LMMs) [11, 34, 47, 57] have shown notable improvement in video understanding and reasoning tasks, e.g., enabling these models to comprehend natural scene videos and answer causal questions. However, as new types of data continue to emerge, these

*Equal contribution.

models are expected to handle previously unseen, out-ofdistribution (OOD) videos [5, 17, 28, 37, 38, 44]. Such OOD videos are rarely encountered during training due to their specialized nature or the need for domain-specific knowledge, such as gymnastic or surgical videos. Consequently, the performance of video LMMs on OOD videos is poor compared to in-distribution videos [23, 36], primarily due to the limited representation of OOD content in training datasets. For instance, video LMMs can readily distinguish between well-represented actions like “dancing” and “exercising” but struggle to differentiate actions like “abuse” from “assault”, as crime videos are seldom included in training datasets. Given that the model faces numerous unseen situations in real-world scenarios, improving OOD video understanding with video LMMs remains a significant challenge.

Fine-tuning the model on the target videos is a straightforward method to enhance its performance. However, finetuning the model for each OOD scenario is time-consuming and often impractical, as it requires substantial amounts of training data to avoid overfitting and incurs significant training costs. In contrast, in the realm of large language models (LLMs), in-context learning (ICL) [7]—which involves providing example inputs alongside a test sample during inference—has been actively explored as an efficient alternative for handling OOD tasks. ICL has shown strong generalization on unseen tasks in LLMs, making it advantageous as it bypasses the need for fine-tuning. Many studies have demonstrated the effectiveness of ICL in language-only and image-language tasks [2, 16, 27, 55, 56]; however, ICL’s potential in video-language tasks has not been fully explored.

A key challenge with ICL in the video domain is that video tokens are significantly longer than image or text tokens, limiting the number of video examples in a single context. For instance, LLaVA-Video [57] can process up to 32K tokens (i.e., context-length), but a 30-second, 384×384 resolution video sampled in 32 frames is converted to approximately 5.5K tokens through the vision encoder of the model with a patch size of 14. This allows for a maximum of only four video samples within the token limit. Considering that

Out-of-Distribution Videos VideoICL (Ours)

...?

[Figure 1]

|Video LMM|
|---|

# ✗

Examples Database

Q. What is the type of crime?

"Arson"

[Figure 2]

[Figure 3]

Unfamiliar video

[Figure 4]

|Video LMM|
|---|

Explosion Arson

Regular ICL

Context too long! Similarity-

Q. What is the type of crime?

based Selection

✗

[Figure 5]

[Figure 6]

[Figure 7]

|Video LMM|
|---|

ConfidenceBased Iteration

Road Accident Shoplifting Q. What is the type of crime?

- Figure 1. Motivation. Top left: Video LMMs perform poorly in out-of-distribution videos, such as crime videos. Bottom left: In-Context Learning (ICL), which is usually employed to solve this problem, is infeasible for video tasks, since the in-context demonstrations are too long. Right: VIDEOICL alleviates this problem by selecting the most relevant demonstrations (e.g., 2-shot) by similarity-based example selection, and iteratively performing inference with different sets of demonstrations at each step (confidence-based iterative inference).

recent text-only and image-text ICL research is advancing toward many-shot learning [1, 6, 21], often utilizing over 1K examples, the number of video examples here is notably constrained in comparison.

This challenge has been approached from two main directions in multimodal ICL research. One approach aims to reduce the token length of each example [14, 59], while the other focuses on selecting a minimal set of highly effective examples [8, 48]. However, the first approach potentially may lead to a loss of crucial information, while the second relies on a limited number of in-context examples, making the model performance highly sensitive to the relevance of these examples to the given query prompt.

To address these issues, we propose VIDEOICL, a training-free video in-context learning framework for OOD video understanding, which efficiently handles multiple examples within a limited context length while preserving example quality. Specifically, we first introduce a similaritybased relevant example selection strategy, ranking demonstrating examples based on both video and text feature similarity to construct an ordered set of relevant examples for inference. Then, we present a confidence-based iterative inference mechanism; VIDEOICL performs in-context learning iteratively, with each iteration leveraging a new subset of highly relevant examples from the top of the similarity ranking. At each iteration, the model calculates a confidence score based on token probabilities and stops iterating once it reaches a sufficient confidence level to generate an accurate answer. This iterative approach enables the model to effectively utilize a larger pool of examples, maintaining informational richness without exceeding context limits.

Extensive experiments on several OOD video understanding benchmarks demonstrate the superiority of our method. Specifically, we validate our approach across six datasets, spanning four distinct video-language tasks: multiple-choice question answering, open-ended question answering, video classification, and video captioning. The results show that our framework, tested with LLaVA-Video7B [57], outperforms zero-shot baselines, achieving an av-

erage improvement of 25.6%p and up to 54.6%p in QA and classification tasks, alongside a gain of 0.143 BLEU4 points in video captioning. Notably, our approach using a 7B model outperforms a 72B model in zero-shot settings and on some datasets, VIDEOICL achieves better results than LoRA fine-tuned counterparts. These results underscore the effectiveness of our video ICL approach, showing that even smaller models can outperform larger and finetuned models on OOD tasks with a robust ICL framework.

Furthermore, VIDEOICL outperforms the state-of-theart video ICL model, Otter [27], which is pre-trained in an in-context manner, by a substantial margin. These results suggest that the training-free nature of VIDEOICL facilitates greater scalability and generalization to diverse OOD videos compared to Otter. We believe that our findings will inspire further research into video in-context learning and drive advancements in enhancing the generalization capabilities of video LMMs for OOD videos.

Our main contributions are summarized below:

- • We introduce VIDEOICL, a novel training-free framework for video in-context learning that enhances out-ofdistribution (OOD) video understanding without requiring model fine-tuning or compromising input quality.
- • We propose a confidence-based iterative in-context learning approach that effectively leverages multiple examples, addressing the token length limitations of video LMMs.
- • VIDEOICL achieves state-of-the-art results on six diverse OOD video-language datasets, with an average improvement of 25.6%p and up to 54.6%p in QA and classification tasks, along with a gain of 0.143 BLEU-4 points in video captioning, significantly outperforming zero-shot and baseline methods.

#### 2. Related works

##### 2.1. Video large multimodal models

The impressive capabilities of LMMs have motivated substantial research into video LMMs [11, 34, 47, 57], demon-

###### Similarity-based Example Selection

###### Confidence-based Iterative Inference

Similarity Ranking

Video Text

“jump.” Confidence

|Ex. #3 Ex. #6<br><br>?| |
|---|---|
| | |

1

|Video LMM|
|---|

Retry

Examples Database

[Figure 8]

Conf.=0.4 ✗

Q. What did the player do after split jump?

Ex. #3 Ex. #6 Ex. #9 Ex. #5 Ex. #11 Ex. #2

|...|
|---|

- Ex. #1
- Ex. #2

|...|
|---|

|[Figure 9]|
|---|

|Q. What did the player do before loop jump?|
|---|

|Ex. #9 Ex. #5<br><br>?|
|---|

- 2

- 3

|Video LMM|
|---|

Retry Test Sample

“flip.”

Conf.=0.3 ✗

Cosine Similarity

|Ex. #11 Ex. #2<br><br>?|
|---|

|Video LMM|
|---|

“salto.”

Accept Answer

Conf. = 0.9

Video Features

Text Features

- Figure 2. Our Methodology. Given a test query Qtest consisting of a video and some text, each are embedded into a vector. Similaritybased Example Selection: Based on the cosine similarity between the query vector and the embeddings in the database of pre-encoded examples, we retrieve top-k most similar examples. This stage takes negligible time cost since it only generates features from test samples and calculates the similarities with pre-encoded features. Confidence-Based Iterative Inference: Starting from the top of the list, each set of m examples are used as in-context examples for the query Qtest, until the confidence for the generated answer exceeds the threshold.

strating significant performance on video understanding benchmarks. However, the benchmarks commonly used for evaluating video LMMs, such as NExT-QA [50], Perception Test [40], and MVBench [30], primarily evaluate models on general tasks that require only basic skills for typical videos. Therefore, the strong performance of video LMMs is largely limited to in-domain videos, which resemble the videos they were trained on.

On the other hand, studies have shown that video LMMs perform poorly when tested on OOD videos. Khattak et al. [23] test different video LMMs using unusual and physically anomalous activities. Additionally, Marcu et al. [36] reveal that video LMMs struggle with autonomous driving videos. Given that the model encounters numerous unseen situations in real-world, these works underscore the need for further research to enhance OOD video understanding within video LMMs.

##### 2.2. Multimodal in-context learning

Beyond the success of ICL on text-only tasks, it has been extended to multimodal tasks involving images [4, 8, 9, 14, 27, 43, 52, 58] and videos [2, 27, 54]. In video ICL, most studies aim to equip LMMs with the ability to comprehend multiple video examples by training them on video-text interleaved datasets. Otter [27] trains image-text ICL model using ICL instruction tuning dataset with video-language demonstrations. Yu et al. [54] propose a strategy to generate a video-text ICL training dataset that induces smallscale language models. However, these studies do not address OOD videos, and fine-tuning is impractical for handling OOD tasks, as it is both costly and time-consuming to train the model for each task.

A significant challenge in multimodal ICL is the increased length of each demonstration. One approach to address this is to compress the tokens of context examples to fit within the model’s limit, using fused virtual tokens [14]

or learnable parameters [9]. However, such compression methods risk losing crucial details. Another approach is selecting the most effective demonstrations [31, 48, 52]. MMICES [8] considers the varying importance of text and image information. However, since these methods still use a limited number of examples during inference, their performance can be less robust and highly sensitive to selection.

##### 2.3. Iterative in-context learning

To enhance the robustness of ICL, some studies have employed iterative methods in language-only tasks. IDS [41] utilizes iterative refinement by selecting multiple sets of demonstrations and performing majority voting among the generated answers to identify the best response. Se2 [33] introduces an example selection approach that considers the sequence of examples, iteratively refining the selection to find the optimal set based on previously chosen examples. Iterative retrieval [10] enhances ICL by using a stateful, policy-learning framework for iterative example selection, improving performance in semantic parsing tasks. Inspired by these works, we adopt an iterative method to address the challenge of limited example numbers in video ICL.

##### 2.4. Factual confidence estimation

Estimating the factual confidence of LLMs has become an important research focus, as higher confidence levels are often associated with more accurate outputs and fewer hallucinations [12, 15, 19]. Xiong et al. [51] and Lin et al. [32] explore verbalization, where models express confidence in natural language. Kumar et al. [24] employ logit-based methods to estimate token-level confidence by assessing token probabilities as indicators, while Liu et al. [33] apply these techniques for selecting ICL examples. Sequencelevel confidence is often derived by aggregating token-level confidence, typically using the minimum probability across all tokens [20]. Azaria and Mitchell [3] and Li et al. [29]

utilize a trained probe approach, using an MLP trained on supervised data to map hidden states to confidence values.

#### 3. Method

In this section, we describe our method to overcome the challenges of video ICL. First, we select a certain number (k) of in-context examples for a given query, constructing a list of demonstrating examples based on similarity (Sec. 3.2). Next, we sequentially fetch a small number (m ≤ k) of examples from the list, iteratively refining the answer based on model confidence (Sec. 3.3). The complete workflow of VIDEOICL is shown in Fig. 2.

##### 3.1. Notation

Throughout the paper, yˆ = M(x;D) denotes the answer a video LMM M generates for a query x using a set of in-context examples D. ConfM(x,yˆ;D) denotes the video LMM M’s estimated confidence in a generated answer yˆ, given a question x and a set of in-context examples D.

##### 3.2. Similarity-based example selection

At test time, we are given a query x = (t,v), where t and v are the text and video in the query prompt, respectively. Based on the relevance to the query, k in-context examples are selected from the set of target task-specific example data. k is a fixed hyperparameter. For this, we use a linear combination of the cosine similarities on the vector representation of the given query and example data:

SQ (t,v), t, ˜ v˜ := αSC(rt(t),rt(t˜)) + (1 − α)SC(rv(v),rv(˜v)),

(1)

where SC denotes the cosine similarity between two vectors. rt(·) and rv(·) denote text and video encoders, respectively, which map arbitrary-length text and video inputs into fixed-length vectors. α is a balancing coefficient between the text and the video similarities. From the set of example data D = {(t˜1,v˜1),...,(t˜n,v˜n)}, we select the top-k examples that maximize SQ:

[SQ(x,x˜)]. (2)

SelectRelevantk(D,x) := Top−k

x˜∈D

Note that this example selection has negligible cost overhead at inference time, since all of the text and video vector representations in the example set can be pre-processed and be stored in a vector database.

##### 3.3. Confidence-based iterative inference

Since all k in-context examples do not fit within the context size of open-source video LMMs, we instead provide the model m examples at each iteration. m ≤ k is a fixed hyperparameter, chosen so that m examples, along with the

Algorithm 1 VIDEOICL Inference Process input Query x, video LMM M, example pool D. output Final answer yˆ.

- 1: D = SelectRelevantk(D,x).
- 2: i ← 1.
- 3: repeat
- 4: Dcur ← D(i−1)m+1:im. ▷ Select next batch of samples
- 5: yˆi = M(x;Dcur). ▷ Generate answer
- 6: ci = ConfM(x,yˆ1;Dcur). ▷ Evaluate confidence
- 7: i ← i + 1.
- 8: until ci > cth or im > k.
- 9: yˆ = yˆarg max

i ci. ▷ Select the largest confidence answer

query, fit within the model’s context length limit. Specifically, from the list of k relevant examples D, we first generate an answer for the given query x using the first m examples: yˆ1 = M(x;D1:m). If ConfM(x,yˆ1;D1:m) > cth, i.e., the model’s confidence for the generated answer exceeds a predefined confidence threshold cth, then we output yˆ1 as the final answer. Otherwise, we retry using the next m examples Dm+1:2m. We repeat this process until either the confidence for the last answer exceeds cth or all of D is exhausted. After multiple iterations are completed, we finally output the answer which had the highest confidence.

For the confidence estimation, inspired by LLM literature [20, 35, 53] that often utilizes token probability, we also quantify the confidence score by taking the minimum token probabilities from the generated response. Through empirical observations, we found that using the minimum token probability is a better measure of confidence. Given a generated sequence of logits, we first apply softmax normalization to convert logits to probabilities, obtaining a sequence of yˆ = {p1,...,pT}. We then compute the confidence score as c = minTi=1 pi, where pi represents the probability of the i-th token in the generated sequence. More confidence measures are explored in Sec. 5.3.

##### 3.4. Theoretical analysis

To substantiate our approach, we further present a theoretical analysis illustrating how our confidence-based iteration method can enhance model accuracy. Let us assume that the probability of a video LMM outputting the correct answer for a given task, given a query and a set of in-context examples, is constant: Pr(M(x;Dcur) = y) = pc, and independent across iterations. For simplicity, we further assume that confidence estimation operates independently at each iteration and that the framework outputs the final iteration’s answer as the final result, rather than selecting the answer with the highest confidence score.

Proposition 1 (Asymptotic model accuracy). Let a(n) be the expected accuracy of VIDEOICL with a maximum of n

###### Question Iteration 1 Iteration 2

[Figure 10]

[Figure 11]

Classify the following video into one of the following categories: <14 crime categories>

###### In-Context Examples

Zero-shot Pred.: Normal Event ✗ Ground Truth: Shoplifting

Normal Event Shoplifting

Prediction: Shoplifting

###### No more iteration!

Confidence: 0.83

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

Do the players perform gainer salto backward stretched with 2.5 twist to side of beam before the salto backward tucked?

[Figure 16]

Salto backward stretched with 1 twist to side after double salto tucked? Yes

Salto backward stretched with 2.5 twist to side after double salto tucked? No

Salto backward stretched with 0.5 twist to side before double salto tucked? No

Salto backward stretched with 3 twist to side after double salto tucked? Yes

Zero-shot Pred.: No ✗ Ground Truth: Yes

Prediction: No

Prediction: Yes

###### Wrong answer

[Figure 17]

Confidence: 0.50

Confidence: 0.68

[Figure 18]

[Figure 19]

[Figure 20]

Classify the following video into one of the following categories: <14 crime categories>

Burglary Arson

Zero-shot Pred.: Fighting ✗ Ground Truth: Assault

Assault Vandalism

Prediction: Assault Confidence: 0.44

Prediction: Assault Confidence: 0.62

Right answer but uncertain

[Figure 21]

Figure 3. Qualitative Results. We show three real hand-picked test samples from the main benchmarks. The first and third examples are from the UCF-Crime [44] (video classification) task, and the second one is from the Sports-QA [28] (open-ended QA) task. The leftmost column shows the given question, which the vanilla (non-ICL) model makes an incorrect prediction. The second and third columns show the first and second confidence-based iterations, with the selected in-context demonstrations at each iteration.

confidence-based iterations. Then,

1 1 + TPRFPR · 1−p

, (3)

lim

a(n) =

n→∞

c pc

where TPR and FPR stand for the true positive rate (i.e., recall) and the false positive rate of the confidence estimation method, respectively.

For example, if pc = 0.5, TPR = 0.9, and FPR = 0.1, then the model’s accuracy is expected to converge to 0.9 as the maximum number of iterations becomes large. In conclusion, if the confidence estimator is accurate enough, then our confidence-based iteration method is theoretically guaranteed to perform better than performing only one iteration. Please refer to Appendix C for the proof and rigorous statement of the proposition.

#### 4. Experiments

In this section, we validate our framework in four videolanguage tasks on a wide range of out-of-distribution benchmarks using six datasets: multiple-choice QA on Animal Kingdom [38], open-ended QA on Sports-QA [28] and PitVQA [17], video classification on UCF-Crime [44] and Drive&Act [37], and video captioning task on CapERA [5].

##### 4.1. Experiment setup

We apply our framework to state-of-the-art open-source video LMMs, such as LLaVA-Video-7B [57], Qwen2-VL7B [47], and Oryx-1.5-7B [34], selected for their strong performance in general video understanding benchmarks. For the similarity-based example selection, we use SentenceBERT [42] and InternVideo2 [49] as text and video encoders, respectively. We sample each video at 1 frame per second across benchmarks, and for videos over 32 seconds, we uniformly select 32 frames from the entire sequence. For the confidence estimation, we use minimum value among the probabilities of generated tokens, and the confidence threshold is set to cth = 0.7 in multiple-choice QA and cth = 0.5 in other tasks. Our reasoning for this choice and additional experiments with various cth settings are provided in Appendix A. The total number of demonstrating examples is set to k = 8. Note that considering the limited context length of the video LMMs, we set the number of in-context examples to m = 2 at each iteration across all benchmarks. Therefore, the maximum number of iterations per test sample is n := k/m = 4.

##### 4.2. Baselines

We compare VIDEOICL with several strong baselines to rigorously assess our method’s performance: GPT4o [46], Gemini-1.5 Pro [45], Otter-7B [27], MMICES [8],

Multiple Choice QA

Video Classification

Open-ended QA

Video Captioning

CapERA

Animal Kingdom

SportsQA

PitVQA

UCFCrime

Drive &Act

n k

BLEU-1 BLEU-2 BLEU-3 BLEU-4 METEOR ROUGE-L

GPT-4o [46] - 0 58.2 - 6.9 58.0 - 0.143 0.065 0.037 0.023 0.142 0.173 Gemini-1.5 Pro [45] - 0 72.9 - 14.7 55.1 - 0.126 0.057 0.031 0.019 0.134 0.176 Otter-7B [27] 1 8 19.4 - 21.8 6.8 - 0.241 0.135 0.088 0.059 0.169 0.167

72B Zero-shot - 0 69.7 25.7 5.7 35.6 14.6 0.133 0.060 0.034 0.020 0.129 0.170 7B LoRA FT - 0 70.2 - 40.5 51.9 - 0.528 0.393 0.302 0.227 0.271 0.181

LLaVA-Video[57]

Zero-shot - 0 68.0 25.5 6.7 39.3 20.2 0.162 0.077 0.045 0.027 0.149 0.181 MMICES [8] 1 2 69.3 43.0 46.4 50.7 51.3 0.462 0.312 0.224 0.160 0.245 0.178 SIMRANKONCE 1 2 69.3 41.8 54.0 50.7 52.0 0.462 0.312 0.224 0.160 0.245 0.178 RANDEXVOTE 4 8 69.6 21.5 11.5 36.6 19.9 0.418 0.256 0.170 0.116 0.189 0.153 SIMRANKVOTE 4 8 70.9 36.3 57.6 50.6 50.6 0.464 0.314 0.228 0.165 0.242 0.175 VIDEOICL (Ours) 4 8 72.3 47.6 61.3 53.3 53.4 0.465 0.320 0.235 0.170 0.252 0.178

∆ +4.3 +22.1 +54.6 +14.0 +33.2 +0.302 +0.242 +0.190 +0.143 +0.104 -0.003

7B Zero-shot - 0 58.6 26.8 5.8 36.1 10.6 0.286 0.159 0.101 0.066 0.149 0.138 SIMRANKONCE 1 2 63.8 43.2 55.3 46.3 45.4 0.457 0.310 0.223 0.158 0.249 0.189 RANDEXVOTE 4 8 62.3 21.0 14.0 36.6 14.3 0.397 0.240 0.157 0.104 0.188 0.170 SIMRANKVOTE 4 8 64.0 50.9 59.4 46.7 45.8 0.448 0.300 0.213 0.151 0.239 0.187 VIDEOICL (Ours) 4 8 66.3 51.5 59.6 48.7 49.3 0.471 0.329 0.244 0.176 0.265 0.189

Qwen2-VL[47]

∆ +7.7 +24.7 +53.8 +12.6 +38.7 +0.185 +0.170 +0.143 +0.110 +0.116 +0.051

7B Zero-shot - 0 58.6 28.3 3.8 11.9 10.7 0.242 0.126 0.077 0.049 0.140 0.151 [34] VIDEOICL (Ours) 4 8 58.5 52.0 58.4 44.0 57.3 0.327 0.195 0.128 0.086 0.188 0.179

Oryx-1.5

∆ -0.1 +23.7 +54.6 +32.1 +46.6 +0.085 +0.069 +0.052 +0.038 +0.047 +0.028

- Table 1. Main Results. We evaluate our framework VIDEOICL on a wide range of out-of-distribution benchmarks. n = k/m denotes the (maximum) number of iterations, where k is the total number of in-context examples, and m is the number of examples used in each iteration. The difference (∆) denotes the score improvement of VIDEOICL over the vanilla 7B models without ICL. The LoRA FT baseline, fine-tuned on each of the downstream tasks, is shown in gray because it is not directly comparable to our method due to its additional training cost. The highest scores for each benchmark are shown in bold.

Zero-shot, SIMRANKONCE, RANDEXVOTE, and SIMRANKVOTE. Otter [27] is a large multimodal model finetuned on MIMIC-IT [26], which is a training set for image and video ICL. We exclude Flamingo [2] and EILeV [54] from the baselines because Flamingo does not provide a checkpoint, and EILeV is specifically trained on egocentric videos for the video captioning task. MMICES [8] is an example selection method originally designed for image ICL, which we have extended to our video multimodal setting. In the zero-shot baseline, responses are generated without any in-context examples.

To further evaluate the impact of each component within VIDEOICL, we use three additional baseline methods. SIMRANKONCE performs ICL a single time with only m relevant examples chosen by our similarity ranking (as detailed in Sec. 3.2), without any iterative inference. RANDEXVOTE incorporates iterative inference but omits confidence-based selection, instead employing majority voting across answers generated using randomly selected in-context examples. SIMRANKVOTE also performs majority voting (i.e., without confidence) but uses k relevant examples chosen by our similarity ranking instead of random examples. We also compare against the zero-shot performance of a larger 72B model. For completeness, we include the performance of LLaVA-Video-7B fine-tuned with

LoRA [18] for some of the benchmark tasks. However, it is not directly comparable to ICL, since fine-tuning incurs a large training cost for each OOD task.

##### 4.3. Datasets

We test our method on six diverse datasets across four tasks, each focusing on specialized domains that are rarely covered in the training data of current video LMMs. For the multiple-choice QA task, we use the Animal Kingdom dataset [38], which includes animal videos annotated with actions across 140 different classes. For open-ended question-answering, we apply Sports-QA [28], covering various sports videos, and PitVQA [17], designed specifically for visual QA in endonasal pituitary surgery videos. For open-ended QA, we utilize UCF-Crime [44], which categorizes types of crime in security camera footage into 13 groups, and Drive&Act [37], which identifies 34 types of activities performed by drivers in Kinect-IR videos. For the video captioning task, we evaluate models on the CapERA dataset [5], which is tailored to describe scenes from an aerial view. For more details, please see Appendix B.

##### 4.4. Quantitative results

Tab. 1 summarizes the performance comparison with the baselines. Overall, VIDEOICL outperforms the baselines

Animal Kingdom PitVQA UCF-Crime Baseline 68.0 6.7 39.3 Random 68.4 (+0.4) 8.3 (+1.6) 38.4 (-0.9) Text only - 33.1 (+24.8) Video only - 29.1 (+22.4) Text + Video 72.3 (+4.3) 61.3 (+54.6) 53.3 (+14.0)

- Table 2. Ablation on the similarity-based example selection method. We investigate which feature types are more helpful in selecting relevant examples.

across a wide variety of domains. See Appendix D.1 for a more detailed discussion on the baseline methods’ performance.

Multiple choice QA. We observe that VIDEOICL achieves a +4.3%p improvement in accuracy for recognizing animal actions compared to zero-shot LLaVA-Video7B. Notably, VIDEOICL even surpasses the larger LLaVAVideo-72B model, despite using the smaller LLaVA-Video7B model. This result indicates that simply increasing model size is not an effective solution for OOD video understanding. Furthermore, video ICL baselines (e.g., SIMRANKVOTE) that rely solely on similarity ranking without confidence-based iteration perform less effectively than VIDEOICL, underscoring the advantages of our approach.

Open-ended QA. We observe the highest accuracy improvement in open-ended QA tasks, with VIDEOICL achieving up to +54.6%p and +22.1%p improvements on PitVQA and Sports-QA, respectively, compared to zeroshot performance. This result highlights that ICL is particularly effective for out-of-distribution tasks where answers adhere to a specific format, even when the question itself is phrased simply. For example, given the question “What is the athlete doing?” with a video showing a gymnast performing a move called salto backward tucked, the model might respond with, “The athlete is jumping from the ground in a crowded gym.” While this answer is not incorrect, it lacks specific domain knowledge users might expect in the context of the gymnastics domain, such as “The athlete is performing salto backward tucked.” ICL enables the model to align with the expected answer style and apply relevant domain knowledge, yielding a response that is both more precise and contextually accurate.

Video Classification. We observe that VIDEOICL significantly outperforms baselines by up to +14.0%p more accuracy gain in UCF-Crime and +38.7%p in Drive&Act, demonstrating the effectiveness of our method in video classification tasks. Surprisingly, VIDEOICL exceeds the LoRA fine-tuned model on UCF-Crime. As Bertsch et al. [6] discovered, VIDEOICL tends to surpass fine-tuning in a small-size dataset. These outstanding results may be due to the fact that similar in-context examples also have a high chance of being from the same class, demonstrating the use-

fulness of our similarity-based example selection process.

Video Captioning. Across various metrics, our method consistently surpasses the baselines, demonstrating its effectiveness in video captioning tasks. The largest improvement was observed in BLEU-1, with the gains gradually decreasing from BLEU-2 to BLEU-4. This suggests that the examples in VIDEOICL provide useful words related to the query video, enhancing the relevance of generated captions. Comparison to proprietary models. In Tab. 1, we compare the performance of VIDEOICL with two leading proprietary APIs: GPT-4o [46] and Gemini-1.5 Pro [45]. Due to cost constraints, we evaluate one benchmark per task format. For this analysis, we use the results of VIDEOICL with the LLaVA-Video-7B model. While Gemini-1.5 Pro outperforms VIDEOICL by +0.6%p in the Animal Kingdom benchmark, it falls short on the PitVQA and CapERA benchmarks. Similarly, GPT-4o achieves a +4.7%p advantage over VIDEOICL in the UCF-Crime benchmark but fails on the Animal Kingdom, PitVQA, and CapERA tasks. Overall, VIDEOICL demonstrates an average performance improvement of +17.8%p over GPT-4o and +14.2%p over Gemini-1.5 Pro. These results highlight the effectiveness and robustness of VIDEOICL across diverse OOD videos, even when compared to significantly larger models. Importantly, VIDEOICL achieves this performance with a 7Bparameter backbone, while GPT-4o and Gemini-1.5 Pro are much larger models. Despite this inherent size disadvantage, VIDEOICL excels due to its efficient iterative incontext learning approach, underscoring its strong capabilities in challenging scenarios.

##### 4.5. Qualitative results

We illustrate some representative samples from our benchmarks in Fig. 3. In the first row, we show an example from UCF-Crime [44]. While the ground truth label is Shoplifting, the zero-shot model answers incorrectly as Normal Event. In contrast, our model selects two relevant demonstrations—a Normal Event and a Shoplifting example in the first round. Using these two in-context examples, our model is able to answer the original question correctly. This demonstrates that our similarity-based example selection allows the model to select relevant and helpful examples.

In the second row, we show an example from SportsQA [28] where the vanilla model makes an incorrect prediction. In the first iteration, the initial response of our model is incorrect too, with low confidence. By the second iteration, our model could answer correctly with high confidence. Without the iterative approach, the model would not have reached the correct answer, highlighting the effectiveness of iterative selection, especially when initial demonstrations are suboptimal.

In contrast, in the third row, even though the model predicts the right answer in the first iteration, it lacks confi-

CapERA

Animal

PitVQA

UCFCrime

Kingdom

BLEU-4 METEOR

Baseline 68.0 6.7 39.3 0.027 0.149 k = 2 69.3 54.0 50.7 0.160 0.245 k = 4 71.0 59.5 52.7 0.168 0.251 k = 8 72.3 61.3 53.3 0.170 0.253

- ∆ +4.3 +54.6 +14.0 +0.143 +0.104 k = 16 73.2 61.2 53.6 0.169 0.250

- ∆ +5.2 +54.5 +14.3 +0.142 +0.101

- Table 3. Ablation on the total number of demonstrations (k). The highest scores for each benchmark are shown in bold.

dence due to insufficiently relevant demonstrations, as there is no example from the same class as the ground truth. Only after observing relevant examples in the second iteration does it gain enough confidence in the correct answer, suggesting that its initial correct prediction was a fluke. We provide more qualitative results in Appendix D.2.

#### 5. Analysis

In this section, we analyze the impact of each component and hyperparameter of our method on its performance. For our analysis, we select one representative dataset for each video-language task: Animal Kingdom [38], PitVQA [17], UCF-Crime [44], and CapERA [5].

##### 5.1. Demonstration selection method

The impact of similarity-based example selection is studied by analyzing which query features (text or video) most influence the effectiveness of example selection. We compare our method with random demonstration selection, text feature-only selection, and video feature-only selection across three datasets. The results, shown in Tab. 2, indicate that our similarity-based selection method using both text and video features outperforms the random selection baseline by a large margin. For instance, on PitVQA, similaritybased selection leads to up to +53.0%p performance increase over random selection. Furthermore, we observe that both text and video features contribute to similarity-based selection, as using only one of the two yields lower scores.

##### 5.2. Total number of available examples (k)

We investigate the impact of confidence-based iterative inference by comparing our method, which uses multiple iterations, with a single-iteration baseline across four datasets. The results, shown in Tab. 3, indicate that increasing the total number of demonstrations generally helps the benchmark performance. For example, compared to performing only one iteration (k = 2), we observe up to +7.2%p performance increase in PitVQA when using eight iterations (k = 16). This is consistent with the general observation

CapERA

Animal

PitVQA

UCFCrime

Kingdom

BLEU-4 METEOR

Baseline 68.0 6.7 39.3 0.027 0.149 Verbalization 69.7 54.6 51.8 0.160 0.245

∆ +1.7 +47.9 +12.5 +0.133 +0.096 Trained Probe 71.7 42.5 52.7 0.162 0.250

- ∆ +3.7 +35.8 +13.4 +0.135 +0.101 Token Prob. 72.3 61.3 53.3 0.170 0.253

- ∆ +4.3 +54.6 +14.0 +0.143 +0.104

Table 4. Ablation on the confidence estimation method. ‘Token Prob.’ refers to Token Probability. The highest scores for each benchmark are shown in bold.

that more ICL demonstrations lead to higher performance in Bertsch et al. [6]. This demonstrates that our confidencebased iterative inference can serve as an alternative to using many demonstrations at once when the context length is limited in a video ICL setting.

##### 5.3. Confidence estimation method

We evaluate our confidence estimation method, token probability [20], against two alternative approaches: verbalization [32, 51] and trained probe [3, 22], across four benchmark datasets. For the trained probe, following Azaria and Mitchell [3], we pre-train a 4-layer MLP as a confidence estimator using the hidden states of the last token on a subset of diverse video-language datasets and benchmarks [13, 25, 30, 57]. The confidence score from the trained probe is used in the same way as our main method. For verbalization, we ask the model if it is confident enough to respond before generating the answer. If it answers yes, we proceed with the response; if not, we iterate with the next set of examples. The results are shown in Tab. 4.

The token probability method outperforms other approaches, with the trained probe following closely, while verbalization performs the worst. This somewhat contradicts the findings in [35], which report that the trained probe estimates model confidence most accurately. A possible explanation is that, unlike other zero-shot methods, the trained probe’s reliance on pre-training makes it more dependent on its specific training data, limiting its ability to generalize, potentially reducing robustness on OOD tasks [39].

##### 5.4. Most confident responses across iterations

We examine which iteration yields the highest confidence scores. Fig. 4 illustrates the confidence across four datasets and highlights the iteration, out of four, that produced the most confident answers. Notably, despite using the most similar examples in the first iteration, only 29% of responses in CapERA and 50% in PitVQA achieved enough confidence to surpass the threshold in the first round. This finding suggests that relying only on the first iteration may

Most Confident Examples

Animal

3931 1041 635 489

PitVQA

5423 2487 1588 1334

UCF-Crime

485 109 46 32

CapERA

408 326 338 319

0% 25% 50% 75% 100%

Iter. 1 Iter. 2 Iter. 3 Iter. 4

Figure 4. Most confident examples. The numbers on each bar represent the number of test samples where the corresponding iteration ended up having the highest confidence score. The x-axis represents the proportion of each iteration.

be inadequate. Multiple iterations allow for adding more examples, which may boost confidence in later rounds.

#### 6. Conclusion

In this paper, we propose VIDEOICL, a novel framework for in-context learning for out-of-distribution videolanguage tasks with large multimodal models. We address the challenge of multimodal in-context examples exceeding the token length limit of LMMs by employing similaritybased demonstration selection and confidence-based iteration. Extensive experimental results highlight the effectiveness of our method for OOD videos. VIDEOICL is trainingfree, enabling rapid adaptation to novel tasks while offering a more efficient and feasible alternative to naive in-context learning at inference time.

#### 7. Acknowledgments

This work was supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (No. RS-2022-00187238, Development of Large Korean Language Model Technology for Efficient Pre-training) and (No. RS-2019-II190075, Artificial Intelligence Graduate School Program(KAIST)).

#### References

- [1] Rishabh Agarwal, Avi Singh, Lei M Zhang, Bernd Bohnet, Luis Rosias, Stephanie Chan, Biao Zhang, Ankesh Anand, Zaheer Abbas, Azade Nova, et al. Many-shot in-context learning. arXiv preprint arXiv:2404.11018, 2024. 2
- [2] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. Advances in neural information processing systems, 35:23716–23736,

2022. 1, 3, 6

- [3] Amos Azaria and Tom Mitchell. The Internal State of an LLM Knows When It’s Lying. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 967–976, Singapore, 2023. Association for Computational Linguistics. 3, 8
- [4] Folco Bertini Baldassini, Mustafa Shukor, Matthieu Cord, Laure Soulier, and Benjamin Piwowarski. What makes multimodal in-context learning work? In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, pages 1539–1550, 2024. 3
- [5] Laila Bashmal, Yakoub Bazi, Mohamad Mahmoud Al Rahhal, Mansour Zuair, and Farid Melgani. Capera: Captioning events in aerial videos. Remote Sensing, 15(8):2139, 2023. 1, 5, 6, 8, 13
- [6] Amanda Bertsch, Maor Ivgi, Uri Alon, Jonathan Berant, Matthew R Gormley, and Graham Neubig. In-context learning with long-context models: An in-depth exploration. arXiv preprint arXiv:2405.00200, 2024. 2, 7, 8
- [7] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In Advances in Neural Information Processing Systems, pages 1877–1901, 2020. 1
- [8] Shuo Chen, Zhen Han, Bailan He, Mark Buckley, Philip Torr, Volker Tresp, and Jindong Gu. Understanding and improving in-context learning on vision-language models. In ICLR 2024 Workshop on Mathematical and Empirical Understanding of Foundation Models, 2024. 2, 3, 5, 6
- [9] Tao Chen, Enwei Zhang, Yuting Gao, Ke Li, Xing Sun, Yan Zhang, Hui Li, and Rongrong Ji. Mmict: Boosting multimodal fine-tuning with in-context examples. ACM Transactions on Multimedia Computing, Communications and Applications, 2024. 3
- [10] Yunmo Chen, Tongfei Chen, Harsh Jhamtani, Patrick Xia, Richard Shin, Jason Eisner, and Benjamin Van Durme. Learning to retrieve iteratively for in-context learning. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 7156–7168, Miami, Florida, USA, 2024. Association for Computational Linguistics. 3
- [11] Zesen Cheng, Sicong Leng, Hang Zhang, Yifei Xin, Xin Li, Guanzheng Chen, Yongxin Zhu, Wenqi Zhang, Ziyang Luo, Deli Zhao, et al. Videollama 2: Advancing spatialtemporal modeling and audio understanding in video-llms. arXiv preprint arXiv:2406.07476, 2024. 1, 2
- [12] Shangbin Feng, Weijia Shi, Yike Wang, Wenxuan Ding, Vidhisha Balachandran, and Yulia Tsvetkov. Don’t hallucinate, abstain: Identifying llm knowledge gaps via multi-llm collaboration. arXiv preprint arXiv:2402.00367, 2024. 3
- [13] Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang

- Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. arXiv preprint arXiv:2405.21075, 2024. 8
- [14] Jun Gao, Qian Qiao, Ziqiang Cao, Zili Wang, and Wenjie Li. Aim: Let any multi-modal large language models embrace efficient in-context learning. arXiv preprint arXiv:2406.07588, 2024. 2, 3
- [15] Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. On calibration of modern neural networks. In Proceedings of the 34th International Conference on Machine Learning, pages 1321–1330. PMLR, 2017. 3
- [16] Zhongyi Han, Guanglin Zhou, Rundong He, Jindong Wang, Tailin Wu, Yilong Yin, Salman Khan, Lina Yao, Tongliang Liu, and Kun Zhang. How well does GPT-4v(ision) adapt to distribution shifts? a preliminary investigation. In ICLR 2024 Workshop on Mathematical and Empirical Understanding of Foundation Models, 2024. 1
- [17] Runlong He, Mengya Xu, Adrito Das, Danyal Z Khan, Sophia Bano, Hani J Marcus, Danail Stoyanov, Matthew J Clarkson, and Mobarakol Islam. Pitvqa: Image-grounded text embedding llm for visual question answering in pituitary surgery. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 488– 498, 2024. 1, 5, 6, 8, 13
- [18] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-Rank Adaptation of Large Language Models,

2021. arXiv:2106.09685 [cs]. 6

- [19] Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, et al. A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions. arXiv preprint arXiv:2311.05232, 2023. 3
- [20] Yuheng Huang, Jiayang Song, Zhijie Wang, Shengming Zhao, Huaming Chen, Felix Juefei-Xu, and Lei Ma. Look before you leap: An exploratory study of uncertainty measurement for large language models. arXiv preprint arXiv:2307.10236, 2023. 3, 4, 8
- [21] Yixing Jiang, Jeremy Andrew Irvin, Ji Hun Wang, Muhammad Ahmed Chaudhry, Jonathan H Chen, and Andrew Y. Ng. Many-shot in-context learning in multimodal foundation models. In ICML 2024 Workshop on In-Context Learning, 2024. 2
- [22] Saurav Kadavath, Tom Conerly, Amanda Askell, Tom Henighan, Dawn Drain, Ethan Perez, Nicholas Schiefer, Zac Hatfield-Dodds, Nova DasSarma, Eli Tran-Johnson, Scott Johnston, Sheer El-Showk, Andy Jones, Nelson Elhage, Tristan Hume, Anna Chen, Yuntao Bai, Sam Bowman, Stanislav Fort, Deep Ganguli, Danny Hernandez, Josh Jacobson, Jackson Kernion, Shauna Kravec, Liane Lovitt, Kamal Ndousse, Catherine Olsson, Sam Ringer, Dario Amodei, Tom Brown, Jack Clark, Nicholas Joseph, Ben Mann, Sam McCandlish, Chris Olah, and Jared Kaplan. Language models (mostly) know what they know. arXiv preprint arXiv:2207.05221, 2022. 8
- [23] Muhammad Uzair Khattak, Muhammad Ferjad Naeem,

- Jameel Hassan, Muzammal Naseer, Federico Tombari, Fahad Shahbaz Khan, and Salman Khan. Complex video reasoning and robustness evaluation suite for video-lmms. arXiv preprint arXiv:2405.03690, 2024. 1, 3
- [24] Abhishek Kumar, Robert Morabito, Sanzhar Umbet, Jad Kabbara, and Ali Emami. Confidence under the hood: An investigation into the confidence-probability alignment in large language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 315–334, Bangkok, Thailand, 2024. Association for Computational Linguistics. 3
- [25] Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. Seed-bench: Benchmarking multimodal llms with generative comprehension. arXiv preprint arXiv:2307.16125, 2023. 8
- [26] Bo Li, Yuanhan Zhang, Liangyu Chen, Jinghao Wang, Fanyi Pu, Jingkang Yang, Chunyuan Li, and Ziwei Liu. Mimicit: Multi-modal in-context instruction tuning. arXiv preprint arXiv:2306.05425, 2023. 6
- [27] Bo Li, Yuanhan Zhang, Liangyu Chen, Jinghao Wang, Jingkang Yang, and Ziwei Liu. Otter: A MultiModal Model with In-Context Instruction Tuning, 2023. arXiv:2305.03726. 1, 2, 3, 5, 6
- [28] Haopeng Li, Andong Deng, Qiuhong Ke, Jun Liu, Hossein Rahmani, Yulan Guo, Bernt Schiele, and Chen Chen. Sports-qa: A large-scale video question answering benchmark for complex and professional sports. arXiv preprint arXiv:2401.01505, 2024. 1, 5, 6, 7, 13
- [29] Kenneth Li, Oam Patel, Fernanda Vi´egas, Hanspeter Pfister, and Martin Wattenberg. Inference-time intervention: Eliciting truthful answers from a language model. In Thirtyseventh Conference on Neural Information Processing Systems, 2023. 3
- [30] Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, Limin Wang, and Yu Qiao. Mvbench: A comprehensive multimodal video understanding benchmark. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 22195–22206, 2024. 3, 8
- [31] Li Li, Jiawei Peng, Huiyi Chen, Chongyang Gao, and Xu Yang. How to configure good in-context sequence for visual question answering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 26710–26720, 2024. 3
- [32] Stephanie Lin, Jacob Hilton, and Owain Evans. Teaching Models to Express Their Uncertainty in Words. Transactions on Machine Learning Research, 2022. 3, 8
- [33] Haoyu Liu, Jianfeng Liu, Shaohan Huang, Yuefeng Zhan, Hao Sun, Weiwei Deng, Furu Wei, and Qi Zhang. se2: Sequential example selection for in-context learning. In Findings of the Association for Computational Linguistics: ACL 2024, pages 5262–5284, Bangkok, Thailand, 2024. Association for Computational Linguistics. 3
- [34] Zuyan Liu, Yuhao Dong, Ziwei Liu, Winston Hu, Jiwen Lu, and Yongming Rao. Oryx mllm: On-demand spatial-temporal understanding at arbitrary resolution. arXiv preprint arXiv:2409.12961, 2024. 1, 2, 5, 6

- [35] Mat´eo Mahaut, Laura Aina, Paula Czarnowska, Momchil Hardalov, Thomas M¨uller, and Lluis Marquez. Factual Confidence of LLMs: on Reliability and Robustness of Current Estimators. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 4554–4570, Bangkok, Thailand, 2024. Association for Computational Linguistics. 4, 8
- [36] Ana-Maria Marcu, Long Chen, Jan H¨unermann, Alice Karnsund, Benoit Hanotte, Prajwal Chidananda, Saurabh Nair, Vijay Badrinarayanan, Alex Kendall, Jamie Shotton, et al. Lingoqa: Visual question answering for autonomous driving. In European Conference on Computer Vision, pages 252–269, 2024. 1, 3
- [37] Manuel Martin, Alina Roitberg, Monica Haurilet, Matthias Horne, Simon Reiß, Michael Voit, and Rainer Stiefelhagen. Drive&act: A multi-modal dataset for fine-grained driver behavior recognition in autonomous vehicles. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2801–2810, 2019. 1, 5, 6, 13
- [38] Xun Long Ng, Kian Eng Ong, Qichen Zheng, Yun Ni, Si Yong Yeo, and Jun Liu. Animal kingdom: A large and diverse dataset for animal behavior understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 19023–19034, 2022. 1, 5, 6, 8, 13
- [39] Hadas Orgad, Michael Toker, Zorik Gekhman, Roi Reichart, Idan Szpektor, Hadas Kotek, and Yonatan Belinkov. Llms know more than they show: On the intrinsic representation of llm hallucinations. arXiv preprint arXiv:2410.02707, 2024. 8
- [40] Viorica Patraucean, Lucas Smaira, Ankush Gupta, Adria Recasens Continente, Larisa Markeeva, Dylan Sunil Banarse, Skanda Koppula, Joseph Heyward, Mateusz Malinowski, Yi Yang, Carl Doersch, Tatiana Matejovicova, Yury Sulsky, Antoine Miech, Alexandre Fr´echette, Hanna Klimczak, Raphael Koster, Junlin Zhang, Stephanie Winkler, Yusuf Aytar, Simon Osindero, Dima Damen, Andrew Zisserman, and Joao Carreira. Perception test: A diagnostic benchmark for multimodal video models. In Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2023. 3
- [41] Chengwei Qin, Aston Zhang, Chen Chen, Anirudh Dagar, and Wenming Ye. In-context learning with iterative demonstration selection. arXiv preprint arXiv:2310.09881, 2023. 3
- [42] Nils Reimers and Iryna Gurevych. Sentence-BERT: Sentence embeddings using Siamese BERT-networks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLPIJCNLP), pages 3982–3992, Hong Kong, China, 2019. Association for Computational Linguistics. 5
- [43] Igor Sterner, Weizhe Lin, Jinghong Chen, and Bill Byrne. Few-shot vqa with frozen llms: A tale of two approaches. arXiv preprint arXiv:2403.11317, 2024. 3
- [44] Waqas Sultani, Chen Chen, and Mubarak Shah. Real-world anomaly detection in surveillance videos. In Proceedings of the IEEE Conference on Computer Vision and Pattern

- Recognition (CVPR), 2018. 1, 5, 6, 7, 8, 13
- [45] Gemini Team. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context, 2024. 5, 6, 7
- [46] OpenAI Team. GPT-4o System Card, 2024. arXiv:2410.21276. 5, 6, 7
- [47] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024. 1, 2, 5, 6
- [48] Siyin Wang, Chao-Han Huck Yang, Ji Wu, and Chao Zhang. Bayesian example selection improves in-context learning for speech, text, and visual modalities. arXiv preprint arXiv:2404.14716, 2024. 2, 3
- [49] Yi Wang, Kunchang Li, Xinhao Li, Jiashuo Yu, Yinan He, Guo Chen, Baoqi Pei, Rongkun Zheng, Jilan Xu, Zun Wang, et al. Internvideo2: Scaling video foundation models for multimodal video understanding. arXiv preprint arXiv:2403.15377, 2024. 5
- [50] Junbin Xiao, Xindi Shang, Angela Yao, and Tat-Seng Chua. Next-qa: Next phase of question-answering to explaining temporal actions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 9777–9786, 2021. 3
- [51] Miao Xiong, Zhiyuan Hu, Xinyang Lu, YIFEI LI, Jie Fu, Junxian He, and Bryan Hooi. Can LLMs express their uncertainty? an empirical evaluation of confidence elicitation in LLMs. In The Twelfth International Conference on Learning Representations, 2024. 3, 8
- [52] Nan Xu, Fei Wang, Sheng Zhang, Hoifung Poon, and Muhao Chen. From introspection to best practices: Principled analysis of demonstrations in multimodal in-context learning. arXiv preprint arXiv:2407.00902, 2024. 3
- [53] Yuchen Yang, Houqiang Li, Yanfeng Wang, and Yu Wang. Improving the reliability of large language models by leveraging uncertainty-aware in-context learning. arXiv preprint arXiv:2310.04782, 2023. 4
- [54] Keunwoo Peter Yu, Zheyuan Zhang, Fengyuan Hu, Shane Storks, and Joyce Chai. Eliciting in-context learning in vision-language models for videos through curated data distributional properties. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 20416–20431, Miami, Florida, USA, 2024. Association for Computational Linguistics. 3, 6
- [55] Lifan Yuan, Yangyi Chen, Ganqu Cui, Hongcheng Gao, Fangyuan Zou, Xingyi Cheng, Heng Ji, Zhiyuan Liu, and Maosong Sun. Revisiting out-of-distribution robustness in nlp: Benchmarks, analysis, and llms evaluations. Advances in Neural Information Processing Systems, 36:58478–58507,

2023. 1

- [56] Xingxuan Zhang, Jiansheng Li, Wenjing Chu, Junjia Hai, Renzhe Xu, Yuqing Yang, Shikai Guan, Jiazheng Xu, and Peng Cui. On the out-of-distribution generalization of multimodal large language models. arXiv preprint arXiv:2402.06599, 2024. 1
- [57] Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li. Video instruction tuning with

- synthetic data. arXiv preprint arXiv:2410.02713, 2024. 1, 2, 5, 6, 8
- [58] Haozhe Zhao, Zefan Cai, Shuzheng Si, Xiaojian Ma, Kaikai An, Liang Chen, Zixuan Liu, Sheng Wang, Wenjuan Han, and Baobao Chang. MMICL: Empowering visionlanguage model with multi-modal in-context learning. In The Twelfth International Conference on Learning Representations, 2024. 3
- [59] Yufan Zhuang, Chandan Singh, Liyuan Liu, Jingbo Shang, and Jianfeng Gao. Vector-icl: In-context learning with continuous vector representations. arXiv preprint arXiv:2410.05629, 2024. 2

Appendices

- A. Discussion on Hyperparameter Choice

Animal Kingdom

PitVQA UCF-Crime Avg. Zero-shot 68.0 6.7 39.3 38.0 cth = 0.1 69.4 53.6 50.3 57.7 cth = 0.3 69.5 58.5 50.6 59.5 cth = 0.5 70.7 61.3* 53.3* 61.8 cth = 0.7 72.3* 61.6 52.7 62.2 cth = 0.9 72.6 61.5 53.6 62.6

Table 5. Ablation on confidence threshold cth. The values used for the main table are marked with *.

We compare the results on varying confidence threshold cth in Tab. 5. While the accuracy generally increases with cth = 0.9, it also increases the cost of the entire process by performing more iterations per query on average. Therefore, we choose cth = 0.5 and 0.7 for the best trade-off between cost and accuracy.

- B. Details on Datasets

Animal Kingdom We use the Animal Kingdom dataset [38] for our multiple choice question answering task. This dataset includes videos of animals with action labels such as Yawning and Struggling, covering 140 unique classes. While it was originally built for action recognition tasks, we modified its format to suit a multiple-choice QA task by pairing one true action label with four randomly chosen alternative labels. The dataset provides 24,004 labeled training examples and 6,096 test examples.

Sports-QA We employ the Sports-QA [28] dataset for open-ended question answering task, which is designed for sports video question answering. This dataset includes various sports, such as basketball, football, and gymnastics, and features diverse question types like descriptions, timelines, causalities, and hypothetical scenarios. The dataset includes 56,385 training examples and 18,718 test examples.

PitVQA We also use PitVQA [17], a dataset designed for VQA in endonasal pituitary surgery videos that requires specific medical knowledge, for the open-ended question answering task. PitVQA provides question-answer annotations at the frame level. For our experiments, we process a sequence of 10 consecutive frames as the video input, with question-answer pairs drawn from the middle, fifth frame. The dataset includes 75,010 training examples and 10,832 test examples.

UCF-Crime UCF-Crime [44], which classifies the type of crime in security camera footage into 13 categories, is used for video classification task. We include all crime categories in the prompt, guiding the model to select the appropriate crime class for the given video. The dataset also includes normal event videos as challenging negative examples. The official split of UCF-Crime provides four different train and test splits, with each split consisting of 532 training samples and 168 test samples. The result is reported as the average performance across the test sets of all four splits.

Drive&Act The Drive&Act dataset [37] is utilized for video classification tasks. This offers comprehensive labels for driver behaviors inside vehicles, including action segmentation information captured in Kinect-IR videos. We extract each segment from the video and ask the model to recognize the action. The official split of Drive&Act provides three different train and test splits. Each split consists of around 2,000 labeled training examples and around 600 test examples. The result is reported as the average performance across the test sets of all three splits.

CapERA For the video captioning task, we evaluate models on the CapERA dataset [5], which is specifically curated for describing scenes captured from an aerial perspective. CapERA provides concise captions for a range of scenarios viewed from above, including concerts, harvesting, and car racing, and consists of 1,473 labeled examples for training and 1,391 for testing.

#### C. Proof of Asymptotic Model Accuracy

Proposition (Asymptotic Model Accuracy). Let a(n) be the expected accuracy of VIDEOICL with a maximum of n confidence-based iterations. Then,

1 1 + TPRFPR · 1−p

,

lim

a(n) =

n→∞

c pc

where TPR and FPR stand for the true positive rate (i.e., recall) and the false positive rate of the confidence estimation method, respectively.

Proof. At each iteration, there are three possibilities:

- • The model returns a correct response and is estimated to be confident, with probability pc · TPR.
- • The model returns an incorrect response, but is estimated to be confident, with a probability of (1 − pc) · FPR.
- • The model returns a response, and is estimated to be unconfident, occurring with probability pu := 1 − (pc · TPR + (1 − pc) · FPR).

For the first two cases, the loop terminates and returns a response, whereas in the third case, the loop continues with a new iteration. Let c(n) represent the probability that the loop ends by correctly returning a response (first case) on the n-th iteration, and l(n) represent the probability that the loop is still ongoing (third case) after n iterations.

The expected accuracy a(n) after n iterations is the sum of the probabilities of ending with a correct response up to the n-th iteration, plus the probability of continuing after the (n − 1)-th iteration, weighted by the probability of a correct response in the next iteration pc:

a(n) =

n

c(i) + l(n − 1) · pc. (4)

i=1

The probability of continuing after the n-th iteration is

l(n) = l(n − 1) · pu, with l(0) = 1, leading to l(n) = pnu by recursion. And the probability c(n) of ending at the n-th iteration with a correct and confident response is:

c(n) = l(n − 1) · (pc · TPR) = pnu−1 · (pc · TPR), (5)

where pc · TPR accounts for the likelihood that a response is classified as confident (pc) and is also correct (TPR). Therefore, we have:

1 − pnu 1 − pu

+ pnu−1 · pc. (6)

a(n) = pc · TPR ·

Since 0 < pu < 1, as n → ∞,

pc · TPR 1 − pu

lim

a(n) =

n→∞

1 1 + TPRFPR · 1−p

. (7)

=

c pc

| |
|---|

#### D. Additional Results

##### D.1. Additional Discussion on Main Results

For LoRA fine-tuning, we use a rank of 32 and train the model for 1 epoch on Animal Kingdom and PitVQA, 5 epochs on UCF-Crime, and 2 epochs on CapERA. In Tab. 1, VIDEOICL outperforms the LoRA fine-tuned model on all datasets except CapERA, showing that in-context examples are more effective than training in OOD video QA when domain knowledge requires extensive data and training.

Interestingly, the LLaVA-Video-72B model underperforms compared to LLaVA-Video-7B model notably in video classification and captioning. For captioning, this is because 72B model often generates excessively long outputs filled with irrelevant details. In video classification, we suspect the limited capacity of 7B model may act as a form

of regularization, helping it generalize better on OOD data, but this needs further investigation.

In addition, VIDEOICL outperforms SIMRANKVOTE in Tab. 1, highlighting the benefits of using confidence-based aggregation instead of majority voting. VIDEOICL also achieves better results than SIMRANKONCE, showing that using more examples leads to better performance. Lastly, SIMRANKVOTE outperforms RANDEXVOTE, demonstrating the effectiveness of selecting similar examples based on video and text features.

##### D.2. Additional Qualitative Results

In the following pages, we present qualitative results of VIDEOICL for each dataset. For each iteration, we use two examples with maximum of 4 iterations, and the outputs of the model are presented together with confidence scores.

#### E. Limitation

While VIDEOICL delivers remarkable performance, it does have some limitations. First, VIDEOICL requires more time compared to single-step in-context learning because it performs multiple rounds of inference. This additional computation may make it less suitable for applications that demand low latency, such as real-time video analysis. However, VIDEOICL mitigates this issue by using early termination when the model confidence in its output is sufficiently high, which significantly reduces computation time. It is also much faster than training a model from scratch.

Second, VIDEOICL relies on having an example pool to select reference examples from. We demonstrate its effectiveness on the UCF-Crime dataset, which contains only 532 training samples, showing that VIDEOICL can perform well even with a relatively small example pool. However, we have not tested its performance with extremely small datasets. Considering the challenges of generating out-of-distribution video data, exploring the effectiveness of VIDEOICL with very limited examples is an important direction for future research.

Multiple Choice QA: Animal Kingdom

###### ▶ Iteration 1

- Example 1: What action is the animal doing in the video? Answer with the option’s letter from the given choices directly. Options: (A) Dancing On Water (B) Urinating (C) Eating (D) Sleeping in its nest (E) Sharing Food The answer is (C)

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

- Example 2: What action is the animal doing in the video? Answer with the option’s letter from the given choices directly. Options: (A) Biting (B) Climbing (C) Dancing (D) Drinking (E) Being Dragged The answer is (D)

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

User: What action is the animal doing in the video? Answer with the option’s letter from the given choices directly. Options: (A) Trapped (B) Attending (C) Walking On Water (D) Sharing Food (E) Showing Affection

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

LLaVA-Video: The answer is (D) (Wrong, Confidence 0.409)

###### ▶ Iteration 2

- Example 3: What action is the animal doing in the video? Answer with the option’s letter from the given choices directly. Options: (A) Chasing (B) Rubbing its head (C) Moving (D) Dying (E) Doing somersault The answer is (C)

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

- Example 4: What action is the animal doing in the video? Answer with the option’s letter from the given choices directly. Options: (A) Puffing its throat (B) Hatching (C) Doing A Back Kick (D) Attending (E) Lying Down The answer is (D)

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

User: What action is the animal doing in the video? Answer with the option’s letter from the given choices directly. Options: (A) Trapped (B) Attending (C) Walking On Water (D) Sharing Food (E) Showing Affection

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

LLaVA-Video: The answer is (B) (Correct, Confidence 0.886)

Figure 5. Qualitative result on the Animal Kingdom dataset.

Open-ended QA: Sports-QA

###### ▶ Iteration 1

- Example 1: What do the players perform after performing transition flight from low bar to high bar? Giant circle backward with 1 turn to handstand

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

- Example 2: What do the players perform after performing transition flight from low bar to high bar? Transition flight from high bar to low bar

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

User: What do the players perform after performing transition flight from low bar to high bar? LLaVA-Video: Giant circle backward with 1 turn to handstand (Correct, Confidence 0.409)

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

###### ▶ Iteration 2

- Example 3: What do the players perform after performing transition flight from low bar to high bar? Transition flight from high bar to low bar

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

- Example 4: What do the players perform after performing transition flight from low bar to high bar? Giant circle backward with 0.5 turn to handstand

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

User: What do the players perform after performing transition flight from low bar to high bar? LLaVA-Video: Giant circle backward with 0.5 turn to handstand (Wrong, Confidence 0.309)

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

###### ▶ Iteration 3

- Example 5: What do the players perform after performing transition flight from low bar to high bar? Giant circle backward

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

- Example 6: What do the players perform after performing transition flight from low bar to high bar? Pike sole circle backward with 0.5 turn to handstand

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

User: What do the players perform after performing transition flight from low bar to high bar? LLaVA-Video: Pike sole circle backward with 0.5 turn to handstand (Wrong, Confidence 0.303)

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

###### ▶ Iteration 4

- Example 7: What do the players perform after performing transition flight from low bar to high bar? Giant circle backward with 1 turn to handstand

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

- Example 8: What do the players perform after performing transition flight from low bar to high bar? Giant circle backward with 0.5 turn to handstand

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

User: What do the players perform after performing transition flight from low bar to high bar? LLaVA-Video: Giant circle backward with 1 turn to handstand (Correct, Confidence 0.586)

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

Figure 6. Qualitative result on the Sports-QA dataset.

Open-ended QA: PitVQA

###### ▶ Iteration 1

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

- Example 1: Where is the surgical instrument freer elevator tip located in the middle of the video? Top-left

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

- Example 2: Where is the surgical instrument freer elevator tip located in the middle of the video? Centre

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

User: Where is the surgical instrument freer elevator tip located in the middle of the video? LLaVA-Video: Bottom-right (Wrong, Confidence 0.167)

###### ▶ Iteration 2

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

- Example 3: Where is the surgical instrument freer elevator tip located in the middle of the video? Top-left

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

- Example 4: Where is the surgical instrument freer elevator tip located in the middle of the video? Centre

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

User: Where is the surgical instrument freer elevator tip located in the middle of the video? LLaVA-Video: Top (Wrong, Confidence 0.204)

###### ▶ Iteration 3

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

- Example 5: Where is the surgical instrument freer elevator tip located in the middle of the video? Bottom-left

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

- Example 6: Where is the surgical instrument freer elevator tip located in the middle of the video? Bottom-left

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

User: Where is the surgical instrument freer elevator tip located in the middle of the video? LLaVA-Video: Bottom-left (Correct, Confidence 0.809)

Figure 7. Qualitative result on the PitVQA dataset.

Video Classification: UCF-Crime

###### ▶ Iteration 1

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

- Example 1: Classify the following video into one of the following categories: {14 categories} Stealing

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

- Example 2: Classify the following video into one of the following categories: {14 categories} Shooting

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

User: Classify the following video into one of the following categories: {14 categories} LLaVA-Video: Stealing (Correct, Confidence 0.484)

###### ▶ Iteration 2

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

- Example 3: Classify the following video into one of the following categories: {14 categories} Stealing

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

- Example 4: Classify the following video into one of the following categories: {14 categories} Vandalism

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

User: Classify the following video into one of the following categories: {14 categories} LLaVA-Video: Stealing (Correct, Confidence 0.591)

Figure 8. Qualitative result on the UCF-Crime dataset.

Video Classification: Drive&Act

###### ▶ Iteration 1

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

- Example 1: Classify the following video into one of the following categories: {34 categories} Putting on sunglasses

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

- Example 2: Classify the following video into one of the following categories: {34 categories} Taking off sunglasses

User: Classify the following video into one of the following categories: {34 categories} LLaVA-Video: Taking off sunglasses (Wrong, Confidence 0.395)

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

###### ▶ Iteration 2

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

- Example 3: Classify the following video into one of the following categories: {34 categories} Eating

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

- Example 4: Classify the following video into one of the following categories: {34 categories} Fetching an object

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

User: Classify the following video into one of the following categories: {34 categories} LLaVA-Video: Eating (Correct, Confidence 0.354)

###### ▶ Iteration 3

- Example 5: Classify the following video into one of the following categories: {34 categories} Eating

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

- Example 6: Classify the following video into one of the following categories: {34 categories} Eating

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

User: Classify the following video into one of the following categories: {34 categories} LLaVA-Video: Eating (Correct, Confidence 0.818)

Figure 9. Qualitative result on the Drive&Act dataset.

Video Captioning: CapERA

###### ▶ Iteration 1

- Example 1: Provide a concise depiction of this video. Two police cars were chasing a white car down a busy street while the white car was walking on the sidewalk.

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

- Example 2: Provide a concise depiction of this video. The winding mountain road is crowded with cars and surrounded by trees.

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

User: Provide a concise depiction of this video. LLaVA-Video: The traffic is moving slowly on the road. (Confidence 0.114)

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

###### ▶ Iteration 2

- Example 3: Provide a concise depiction of this video. Three cars collided on a treelined road and policemen explored the accident.

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

- Example 4: Provide a concise depiction of this video. The road is crowded with cars and surrounded by small buildings and trees.

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

User: Provide a concise depiction of this video. LLaVA-Video: Cars are driving on a road surrounded by trees and greenery. (Confidence 0.090)

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

###### ▶ Iteration 3

- Example 5: Provide a concise depiction of this video. The winding mountain road is crowded with cars and surrounded by trees.

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

- Example 6: Provide a concise depiction of this video. The winding mountain road is crowded with cars and surrounded by trees.

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

User: Provide a concise depiction of this video. LLaVA-Video: The winding mountain road is crowded with cars and surrounded by trees. (Confidence 0.271)

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

###### ▶ Iteration 4

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

- Example 7: Provide a concise depiction of this video. Crowded road with cars.

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

- Example 8: Provide a concise depiction of this video. Winding mountain road crowded with cars and surrounded by trees.

User: Provide a concise depiction of this video. LLaVA-Video: Traffic on a highway surrounded by trees. (Confidence 0.179)

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

Figure 10. Qualitative result on the CapERA dataset.

