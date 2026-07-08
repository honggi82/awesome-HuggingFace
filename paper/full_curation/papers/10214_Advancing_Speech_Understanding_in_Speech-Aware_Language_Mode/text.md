## ADVANCING SPEECH UNDERSTANDING IN SPEECH-AWARE LANGUAGE MODELS WITH GRPO

Avishai Elmakies* , Hagai Aronowitz, Nimrod Shabtay, Eli Schwartz, Ron Hoory, Avihu Dekel IBM Research

# arXiv:2509.16990v1[cs.CL]21Sep2025

ABSTRACT In this paper, we introduce a Group Relative Policy Optimization (GRPO)-based method for training Speech-Aware Large Language Models (SALLMs) on open-format speech understanding tasks, such as Spoken Question Answering and Automatic Speech Translation. SALLMs have proven highly effective for speech understanding tasks. GRPO has recently gained traction for its efficiency in training LLMs, and prior work has explored its application to SALLMs, primarily in multiple-choice tasks. Building on this, we focus on openformat tasks that better reflect the generative abilities of the models. Our approach leverages GRPO with BLEU as the reward signal to optimize SALLMs, and we demonstrate empirically that it surpasses standard SFT across several key metrics. Finally, we explore the potential of incorporating offpolicy samples within GRPO for these tasks, highlighting avenues for further improvement and further research.

Index Terms— Group Relative Policy Optimization, Speech Understanding, Speech Aware Language Models, Speech Question Answering, Automatic Speech Translation.

### 1. INTRODUCTION

Reinforcement Learning (RL) has gained significant traction, due to its effectiveness in improving the reasoning capabilities of LLMs and VLMs [1, 2]. Inspired by the advancements in Multi-Modal RL, we aim to improve Speech-Aware Large Language Models (SALLMs) [3]. SALLMs process spoken language by taking combined speech-and-text inputs and generating text outputs. They are widely used for tasks like Automatic Speech Recognition (ASR), Automatic Speech Translation (AST), and Spoken Question Answering (SQA) [4, 5].

Recent works have applied RL to SALLMs, typically for SQA, yet these approaches often rely on binary rewards, which constrain the model’s ability to provide open-ended answers [6, 7]. Other works use unsupervised methods for reward calculation [8], but these often yield subpar results compared to standard supervised fine-tuning (SFT).

In this work: we propose an RL approach with verifiable rewards (RLVR) to improve SALLMs. Our method leverages the GRPO algorithm [1] and uses the BLEU metric (or other

*Work done during internship at IBM Research

metrics) [9] as a reward signal. We evaluate our approach on speech understanding tasks – SQA and AST. Our approach demonstrates strong performance on open-ended responses, yielding superior results to standard supervised fine-tuning (SFT) as well as the baseline model. We show this empirically using standardized metrics such as BLEU [9], ROUGE [10], METEOR [11], and BERTScore [12]. Our contributions are summarized as follows:

- • We propose to train SALLMs by integrating GRPO with various reward functions.
- • We empirically demonstrate that our approach outperforms standard SFT on SQA and AST across multiple relevant metrics.
- • We study the effectiveness of using off-policy samples as part of our proposed method.

### 2. RELATED WORK

SALLMs have gained recent attention, driven by their potential in speech understanding tasks such as ASR, AST, and SQA. These models leverage a speech encoder, a language model, and train a lightweight projector to align the two modalities. Leading models such as [4, 5] are trained on extensive speech datasets to improve their capabilities. Unlike Speech Language Models (SLMs) [14, 15] which take speech as input and output speech, SALLMs take speech as input but output only text, which makes them more appropriate for chat-oriented tasks.

Reinforcement Learning methods have been used to improve the capabilities of LLMs. Proximal Policy Optimization (PPO) [16] was used to help LLMs learn from human feedback [17], but requires training a reward model and has high memory requirements. Direct Preference Optimization (DPO) demonstrates that when preference response pairs are available, LLMs can be trained without reward [18].

GRPO is an on-policy Reinforcement Learning algorithm, learning from its own generated data. Instead of relying on a value model, it updates the policy and obtains advantage estimates using an empirical normalization of the grouped samples (see Eq. 1). GRPO has been shown to produce SOTA results in reasoning tasks such as math and coding [1].

Table 1. Results on LibriSQA (SQA). SOTA results are taken from LibriSQA paper [13].

BLEU (↑) BERTScore (↑) ROUGE-1 (↑) ROUGE-2 (↑) ROUGE-L (↑) METEOR (↑) SOTA 33.78 93.07 65.38 50.19 62.09 -

Granite Speech 2B 27.74 91.17 56.66 40.25 51.26 53.01 + SFT 40.88 94.15 65.13 49.07 61.50 64.64 + GRPO 44.90 94.45 68.56 53.35 64.88 68.48

Granite Speech 8B 17.85 90.25 49.58 34.31 43.05 53.19 + SFT 42.34 94.49 67.05 51.54 63.76 65.99 + GRPO 46.40 94.76 69.57 57.49 66.16 69.61

Table 2. Results on CoVoST2 English to German (AST). SOTA results are taken from Phi-4-mini paper (Table 5) [5].

BLEU (↑) BERTScore (↑) ROUGE-1 (↑) ROUGE-2 (↑) ROUGE-L (↑) METEOR (↑) SOTA 37.16 - - - - Granite Speech 2B 29.06 86.04 57.25 35.19 54.09 55.03

- + SFT 30.50 86.40 58.53 36.75 55.21 56.18

+ GRPO 31.47 86.90 59.99 37.88 56.75 57.48 Granite Speech 8B 32.48 87.26 60.48 38.78 57.17 58.24

- + SFT 31.62 86.76 59.66 37.91 56.35 57.35

### + GRPO 35.08 87.64 62.90 41.40 59.64 60.40

While GRPO has also been used to improve SQA in SALLMs [6, 7], these studies focused on multi-choice datasets and benchmarks [19, 20, 6, 7]. While easy to evaluate, better performance on those datasets may not reflect the generative capabilities of the evaluated model in tasks like open-ended QA, a key requirement in chat-oriented applications.

A recent approach incorporated off-policy samples into the sample group [21], allowing the model to learn from both on-policy samples and off-policy samples (which were acquired using a different approach/policy). This approach, known as mixed-policy, can be particularly beneficial when high-quality off-policy data is available. Most work on this approach explores text-based models with reasoning tasks using off-policy samples from closed models, such as ChatGPT. In this work, we explore the effect of adding the ground truth to our sample group in GRPO when training SALLMs.

Metrics like BLEU [9], BERTScore [12], ROUGE [10], and METEOR [11] are widely used to evaluate the similarity of a candidate text response to some text reference. While commonly used as a metric for tasks such as machine translation, these metrics have also been adopted as reward functions in RL-based approaches to improve machine translation [22]. More recently, BLEU has also been used as a reward alongside GRPO to improve text-only LLMs on instruction following tasks [23]. We investigate using BLEU and other metrics, computed between the ground-truth and generated text, as a reward for speech-related tasks in SALLMs.

### 3. METHOD 3.1. GRPO

In this work, we propose to use GRPO as our training algorithm [1], which samples different responses from the model, calculates their reward, and uses the GRPO loss to increase the likelihood of high-reward responses. For each prompt q ∼ D, we sample G times from our policy πθ. For each sample G = {o1,...,oG} we calculate a reward R = {r1,...,rG} then use the rewards to estimate the advantages:

ri − mean(R) std(R)

Aˆi =

(1)

Using the above rewards, we can optimize a loss. Specifically, we use the loss introduced in DAPO [24], a slight variation of the original GRPO loss.

li,t = min(si,t(θ)Aˆi,clip[si,t(θ),1 − ϵ,1 + ϵ]Aˆi) (2) where we have si,t(θ) = ππθ(oi,t|p,oi,<t)

old(oi,t|p,oi,<t) is the importance sampling weight. This gives us the final loss:

|oi|

G

1 |G|

(li,t − βDKL[πθ||πref]) (3)

LDAPO(θ) = −

t=1

i=1

where |G| = Gi=1 |oi| i.e., the number of tokens in the completions generated by the policy.

Table 3. Comparing different reward functions when training Granite Speech 2B on LibriSQA Reward BLEU BERTScore ROUGE-1 ROUGE-2 ROUGE-L METEOR AVG BLEU 44.9 94.45 68.56 53.35 64.88 68.48 65.77

- ROUGE-1 38.81 93.54 68.87 53.45 64.76 60.65 63.35
- ROUGE-2 37.82 93.52 68.59 54.15 65.27 58.87 63.04 ROUGE-L 37.95 93.56 68.68 53.84 65.44 59.27 63.12 METEOR 37.69 94.04 66.99 51.74 62.63 70.25 63.89

### 3.2. Mixed-Policy GRPO

Yan et al. [21] define G as a group of both on-policy samples and off-policy samples, estimating rewards as in Eq. 1. They define the off-policy importance sampling weight:

πθ(oj,t|p,oj,<t) πϕ(oj,t|p,oj,<t)

(4)

sˆj,t =

where πϕ is the policy that generated the off-policy samples, and suggest setting πϕ = 1 and disabling clipping for those samples. We explore incorporating the ground truth reference as an off-policy sample in the group to help steer the policy toward high-quality generations, especially early in training. We will denote this method as MP-GRPO. Since we don’t know πϕ, we follow their recommendations and set πϕ = 1 and disable clipping. We use a single off-policy sample in the group, the reference.We note this could be extended to multiple references, but we leave this for future work.

### 3.3. Reward functions

Given some text predicted by our policy, we calculate its reward by computing its BLEU score to a ground-truth reference. The resulting score is within [0,1], indicating the text similarity. In our case we will use BLEU to compare a reference and samples from our policy. BLEU is highly suitable for generative open-ended tasks, with several possible answers. We will note that there are many possible rewards similar to BLEU which could be applicable for our setup, we compare BLEU to other possible rewards in Section 5.4. Neural-based rewards, such as BERT-score [12], or possibly a combination of rewards may also be considered. Due to computational constraints, we focus on BLEU and similar rewards, and leave other possibilities for future work.

4. EXPERIMENTAL SETUP

### 4.1. Tasks and datasets

In this work, we focus on SQA and AST. We also performed some preliminary experiments with ASR and obtained only minor improvements. Unlike ASR, which has a unique valid response for each utterance, tasks like AST and open-ended SQA can have multiple valid outputs. This characteristic

makes them particularly well-suited for our sampling-based approach, where we anticipate greater performance gains. We believe that further work needs to be done on the abilities of RL/GRPO in tasks such as ASR.

For open-ended SQA we use LibriSQA part I [13] as our dataset. This dataset is based on LibriSpeech [25] and contains spoken utterances taken from audio-books as well as questions and open answers created by an LLM. This dataset contains 107K training samples. We use 20% for validation and the test set has about 2500 samples. We use multiple possible prompts for this task, e.g. "Listen to the audio <|audio|> and answer the following question:<|question|>".

For AST we use CoVoST2 [26], focusing on English to German translation. This dataset contains spoken utterances in English and text translations in German. We prompt the model to translate the speech utterance to German directly, without using ASR. This dataset contains about 220K/12K/15K in the train/validation/test splits. We use multiple possible prompts, e.g. "<|audio|> listen to the speech and translate it into German."

### 4.2. Models configuration

It is important to note that SFT and GRPO are two very different approaches. This means that a fully comparable setup is difficult to achieve. Therefore, we report test set performance using the best models selected based on validation results. We believe that this is the fairest comparison between the approaches.

Most of our experiments were done on Granite Speech 2B [4], which is based on Granite 3.3 [27], a CTC Speech encoder and a Window Q-Former projector. Some are based on Granite Speech 8B [4], to test the scalability of our approach. We note that Granite Speech was not trained on SQA. These models were trained on AST, but not on CoVoST2.

We optimized models with AdamW [28] and performed hyperparameter searches for both SFT and GRPO, selecting the best validation model per task. SFT covered learning rates [10−6,5 · 10−5], epochs [1,10], batch sizes [12,48], and warm-up [0,0.15]. GRPO used similar ranges plus β ∈ [0,0.04], group size [4,12], prompt length 256, and max completion length 200. GRPO used temperature = 1 during training. All models used top-p = 0.9, temperature = 0.9 for

50

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

45

BLEUScore

40

35

30

25

SQA AST

SFT GRPO MP-GRPO

Fig. 1. SFT, GRPO and MP-GRPO on SQA and AST

evaluation. With GRPO, a group size of G = 8 and β = 0.02 yielded stable training. Setting β = 0 led to divergence, especially on larger datasets, while higher β reduced reward performance. When following the mixed-policy scheme, we used G − 1 on-policy samples and 1 off-policy. Training used 4 H100 GPUs; GRPO on Granite Speech 2B took up to 24h, compared to far lower SFT cost. Comparable compute budgets for SFT showed minimal gains.

### 5. RESULTS

In this section, we report the results of our experiment. We compare the results of our method to other approaches using BERTScore-F1, BLEU, ROUGE-1/2/L and METEOR, metrics suited to comparing the model’s generated outputs.

### 5.1. Spoken question answering

Table 1 presents SQA results on the LibriSQA dataset. For both model sizes, the baseline scores are lower than the SOTA results reported in the LibriSQA paper [13]. Using either SFT or GRPO improves the results by a large margin, surpassing the SOTA results. Finally, we observe that training with our GRPO approach, outperforms training using SFT. Compared to the base and SFT models, Granite Speech 2B yields BLEU improvements of 61.8% and 9.8% respectively, whereas the 8B model achieves 151% and 6% respectively.

### 5.2. Automatic speech translation

Results for the task of AST on CoVoST2 English to German are presented in Table 2. Granite Speech 2B sees improvements from both SFT and GRPO, with GRPO achieving the strongest results (+8.2% BLEU over base, +3.2% over SFT). In contrast, Granite Speech 8B shows degraded performance with SFT, while GRPO continues to provide gains (+8% over base, +10.9% over SFT).

### 5.3. Mixed-Policy GRPO

We examine the effect of Mixed-Policy GRPO compared to our standard GRPO approach. We report the results for both SQA and AST on Granite Speech 2B. We present results for BLEU in Fig. 1, and note that other metrics show similar trends. We observe that MP-GRPO improves the BLEU score on AST compared to GRPO, suggesting that for AST, adding the reference to the training with GRPO has the potential to improve results. In contrast, when we compare MP-GRPO to GRPO on SQA we observe that the performance degrades when using the reference with GRPO. This difference may be related to the fact that the base model was not trained on SQA but was trained on AST. With SQA, the model had much more to learn, which could have made the off-policy training less stable, resulting in degraded performance. With AST, the model had less to learn, and the off-policy samples were used as anchors that the model could learn from.

### 5.4. Different rewards options

Different text-comparison metrics could be computed and used as a reward. To show that BLEU is the right choice for our approach we trained Granite Speech 2B on LibriSQA, using reward functions including BLEU, ROUGE-1/2/L and METEOR. We tested the overall impact of the reward function on different performance metrics. Results in Table 3 first show that optimizing a given metric yields the best results on that metric, usually at the cost of other metrics, especially BLEU. Among all evaluated metrics, the BLEU score yielded the highest average performance, indicating its overall superiority compared to the other methods considered.

### 6. CONCLUSION

In this work, we propose a method for training Speech Aware Large Language Models. We use GRPO and BLEU to improve the capabilities of SALLMs on open-ended tasks, specifically open-ended Speech Question Answering and Automatic Speech Translation. Unlike multiple-choice tasks, open-format tasks provide a more effective means of assessing the generative abilities of SALLMs.

We show that despite its simplicity, our approach is highly effective. We observe improvements across all metrics; results match or outperform those obtained using the SFT approach. We also show that this approach scales to larger models, showing improvements for models of sizes 2B and 8B.

Finally, we experimented with mixed-policy GRPO, which leverages both on-policy samples and off-policy samples to train our model. While this approach shows promise, further investigation is required, and we encourage the research community to pursue it. We hope this work will inspire the community to test the boundaries of both on-policy algorithms, off-policy algorithms, and even mixed-policy algorithms on different Speech Understanding tasks.

### 7. REFERENCES

- [1] Zhihong Shao et al., “Deepseekmath: Pushing the limits of mathematical reasoning in open language models, 2024,” URL https://arxiv. org/abs/2402.03300, vol. 2, no. 3, pp. 5, 2024.
- [2] Wenxuan Huang et al., “Vision-r1: Incentivizing reasoning capability in multimodal large language models,” arXiv preprint arXiv:2503.06749, 2025.
- [3] Siddhant Arora et al., “On the landscape of spoken language models: A comprehensive survey,” arXiv preprint arXiv:2504.08528, 2025.
- [4] George Saon et al., “Granite-speech: open-source speech-aware llms with strong english asr capabilities,” arXiv preprint arXiv:2505.08699, 2025.
- [5] Abdelrahman Abouelenin et al., “Phi-4-mini technical report: Compact yet powerful multimodal language models via mixture-of-loras,” arXiv preprint arXiv:2503.01743, 2025.
- [6] Andrew Rouditchenko et al., “Omni-r1: Do you really need audio to fine-tune your audio llm?,” arXiv preprint arXiv:2505.09439, 2025.
- [7] Gang Li et al., “Reinforcement learning outperforms supervised fine-tuning: A case study on audio question answering,” arXiv preprint arXiv:2503.11197, 2025.
- [8] Shaowen Wang et al., “Self-improvement for audio large language model using unlabeled speech,” arXiv preprint arXiv:2507.20169, 2025.
- [9] Kishore Papineni et al., “Bleu: a method for automatic evaluation of machine translation,” in Proceedings of the 40th annual meeting of the Association for Computational Linguistics, 2002, pp. 311–318.
- [10] Chin-Yew Lin, “Rouge: A package for automatic evaluation of summaries,” in Text summarization branches out, 2004, pp. 74–81.
- [11] Satanjeev Banerjee et al., “Meteor: An automatic metric for mt evaluation with improved correlation with human judgments,” in Proceedings of the acl workshop on intrinsic and extrinsic evaluation measures for machine translation and/or summarization, 2005, pp. 65–72.
- [12] Tianyi Zhang et al., “Bertscore: Evaluating text generation with bert,” arXiv preprint arXiv:1904.09675, 2019.
- [13] Zihan Zhao et al., “Librisqa: A novel dataset and framework for spoken question answering with large language models,” IEEE Transactions on Artificial Intelligence, 2024.

- [14] Kushal Lakhotia et al., “On generative spoken language modeling from raw audio,” Transactions of the Association for Computational Linguistics, vol. 9, pp. 1336– 1354, 2021.
- [15] Gallil Maimon et al., “Slamming: Training a speech language model on one gpu in a day,” arXiv preprint arXiv:2502.15814, 2025.
- [16] John Schulman et al., “Proximal policy optimization algorithms,” arXiv preprint arXiv:1707.06347, 2017.
- [17] Long Ouyang et al., “Training language models to follow instructions with human feedback,” Advances in neural information processing systems, vol. 35, pp. 27730–27744, 2022.
- [18] Rafael Rafailov et al., “Direct preference optimization: Your language model is secretly a reward model,” Advances in neural information processing systems, vol. 36, pp. 53728–53741, 2023.
- [19] S Sakshi et al., “Mmau: A massive multi-task audio understanding and reasoning benchmark,” arXiv preprint arXiv:2410.19168, 2024.
- [20] Dingdong Wang et al., “Mmsu: A massive multi-task spoken language understanding and reasoning benchmark,” arXiv preprint arXiv:2506.04779, 2025.
- [21] Jianhao Yan et al., “Learning to reason under off-policy guidance,” arXiv preprint arXiv:2504.14945, 2025.
- [22] Leshem Choshen et al., “On the weaknesses of reinforcement learning for neural machine translation,” arXiv preprint arXiv:1907.01752, 2019.
- [23] Yapei Chang et al., “Bleuberi: Bleu is a surprisingly effective reward for instruction following,” arXiv preprint arXiv:2505.11080, 2025.
- [24] Qiying Yu et al., “Dapo: An open-source llm reinforcement learning system at scale, 2025,” URL https://arxiv. org/abs/2503.14476, 2025.
- [25] Vassil Panayotov et al., “Librispeech: an asr corpus based on public domain audio books,” in 2015 IEEE international conference on acoustics, speech and signal processing (ICASSP). IEEE, 2015, pp. 5206–5210.
- [26] Changhan Wang et al., “Covost 2 and massively multilingual speech translation.,” in Interspeech, 2021, vol. 2021, pp. 2247–2251.
- [27] IBM Granite Team, “Granite 3.0 language models,” URL: https://github. com/ibm-granite/granite-3.0language-models, 2024.

#### [28] Ilya Loshchilov et al., “Fixing weight decay regularization in adam,” arXiv preprint arXiv:1711.05101, vol. 5, no. 5, pp. 5, 2017.

