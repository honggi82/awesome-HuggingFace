## arXiv:2401.12187v1[cs.LG]22Jan2024

[Figure 1]

# WARM: On the Benefits of Weight Averaged Reward Models

###### Alexandre Ramé, Nino Vieillard, Léonard Hussenot, Robert Dadashi, Geoffrey Cideron, Olivier Bachem, Johan Ferret

Google DeepMind

Aligning large language models (LLMs) with human preferences through reinforcement learning (RLHF) can lead to reward hacking, where LLMs exploit failures in the reward model (RM) to achieve seemingly high rewards without meeting the underlying objectives. We identify two primary challenges when designing RMs to mitigate reward hacking: distribution shifts during the RL process and inconsistencies in human preferences. As a solution, we propose Weight Averaged Reward Models (WARM), first finetuning multiple RMs, then averaging them in the weight space. This strategy follows the observation that fine-tuned weights remain linearly mode connected when sharing the same pre-training. By averaging weights, WARM improves efficiency compared to the traditional ensembling of predictions, while improving reliability under distribution shifts and robustness to preference inconsistencies. Our experiments on summarization tasks, using best-of-𝑁 and RL methods, shows that WARM improves the overall quality and alignment of LLM predictions; for example, a policy RL fine-tuned with WARM has a 79.4% win rate against a policy RL fine-tuned with a single RM.

Keywords: Alignment, RLHF, Reward Modeling, Model Merging

#### 1. Introduction

Reward modeling. Conversational assistants such as Gemini [1] or GPT-4 [2] have revolutionized the AI community and beyond. These LLMs are capable of completing novel and intricate tasks, including mathematics, coding, and tool use [3]. These advancements are underpinned by a systematic three stage training procedure: pre-training by next token prediction [4, 5, 6], supervised fine-tuning (SFT) to learn to follow instructions [7, 8, 9], and ultimately, reinforcement learning (RL) to maximize a reward encapsulating the desired behaviors [10]. However, defining such rewards for real-world tasks is non-trivial [11]. In reinforcement learning from human feedback (RLHF) [12, 13, 14, 15], rewards are reward models (RMs), trained on binary preference datasets to emulate human judgment. The enhancement of LLM capabilities from RL is strongly tied to the quality of the RMs [16].

Reward hacking. Particularly insidious in RLHF [17, 18] is the reward hacking issue [19, 20, 21, 22] (a.k.a. reward overoptimization), arising from reward misspecification [23, 24] between the proxy RM and actual human preferences. While optimizing for the RM initially provides improvements, in later stages the policy (i.e., the LLM being trained) usually learns to exploit loopholes in the RM and achieves high rewards without truly fulfilling the intended objectives, as illustrated in Figure 1(b). This reward hacking phenomenon poses numerous issues. First, it degrades performances, manifesting as linguistically flawed [25] or unnecessarily verbose [26] outputs, which do not reflect true human preferences. Second, it complicates checkpoint selection due to the unreliability of the proxy RM, echoing Goodhart’s Law [27]: “when a measure becomes a target, it ceases to be a good measure”. Third, it can engender sycophancy [28, 29] or amplify social biases, reflecting the limited and skewed demographics of feedback providers [30, 31]. Lastly and most critically, misalignment [32, 33] due to reward hacking can escalate into safety risks [19, 34, 35], in particular given the rapid integration of LLMs in everyday life and critical decision-making. Such concerns underscore the need to mitigate reward hacking to ensure the beneficial and safe deployment of LLMs.

Corresponding author: alexandrerame@google.com

RL fine-tuning

Aligned LLM

|Compute reward|
|---|

|Sample from policy|
|---|

###### RL fine-tune step

[Figure 2]

Generate output by feeding an unlabeled input data point

Assign a reward to the model’s output.

Use RL to maximize reward by updating weights.

[Figure 3]

Reward function

SFT

WARM: Weight Averaged Reward Model

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

|Collect preference dataset|
|---|

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

Human or AI pairwise feedback.

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

Multiple RM fine-tunings with different hyperparams

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

Weight averaging

[Figure 31]

[Figure 32]

(a) WARM procedure with 𝑀 = 3.

- 4

- 5

- 6

- 7

- 8

- 9

- 10

- 11

Controlreward

WARM M = 10

WARM M = 6 WARM M = 2 ENS M = 2

Ind 2 Ind 1

0 2000 4000 6000 8000

# steps

(b) WARM mitigates reward hacking.

- Figure 1 | Figure 1(a) illustrates the alignment process with WARM. From a SFT-ed LLM, we apply RL fine-tuning to optimize a proxy reward model (RM), in line with RLHF [12]. The innovation of WARM lies in the design of the proxy RM, which is the weight average (WA) of 𝑀 individual RMs, each fine-tuned from a shared pre-trained LLM on the same preference dataset, but with slight differences such as diverse hyperparameters. This WA approach is efficient, while enhancing the reliability under distribution shifts and robustness under inconsistent preferences. Figure 1(b) showcases the impact during RL alignment. The control reward (detailed in Section 5) initially increases but eventually deteriorates, a phenomenon called reward hacking [19]. However, when WARM serves as the proxy RM, increasing 𝑀 (the number of averaged RMs) significantly improves absolute results while delaying the collapse, as indicated by the control rewards maintaining higher values for longer during training. Same plot with KL as the 𝑥-axis in Figure 8(a) and with label corruption in Figure 18.

Challenges. Two primary challenges underlie reward hacking. The first major issue are the distribution shifts encountered by the RM [36, 37]. Indeed, the generations from the policy might deviate substantially from those in the offline preference dataset, posing an out-of-distribution (OOD) challenge. Moreover, those distribution shifts are accentuated by the policy drift during the RL procedure: the policy moves away from its SFT initialization, continually altering the distribution of predictions the RM needs to interpret reliably. Second, preferences are inconsistent: the binary labels in the preference dataset are noisy. Indeed, human labelers often rely on simpler criteria (length, bullet points, politeness) over more nuanced indicators. Moreover, errors can be exacerbated for complex tasks requiring specific expertise [38], and because of the multi-objective nature of alignment [39] requiring handling the heterogeneity of human opinions. Overall, this results in a low inter-labeler agreement (72.6% for InstructGPT [40]), altering the robustness of the RM.

Goal and ensembling baseline. Designing good RMs must meet a tripartite objective: guiding RL efficiently, reliably scoring generations despite the distribution shifts, and providing robust signals amidst label noise. To address these challenges, the seminal work on RLHF from Christiano et al. [12] and more recent works [41, 42] leveraged prediction ensembling (ENS) [43], averaging the rewards from multiple RMs. ENS improves the reliability of the reward and mitigates hacking risks [41, 42]. Yet, ENS suffers from memory and inference overhead causing efficiency challenges; we will also show that ENS fails to improve robustness to label noise in the preference datasets.

WARM. In this paper, we propose weight averaged reward models (WARM), a simple, efficient and scalable strategy to obtain a reliable and robust RM by combining multiple RMs. Starting from a shared pre-trained LLM, we launch multiple RM fine-tunings: in practice, the different runs have different hyperparameters (as in grid search), and see the preference data in different orders, thus

leading to diverse RMs. A key contribution is how the different RMs are merged: by linear interpolation in the weight space. This follows the findings from the linear mode connectivity (LMC) [44, 45] and weight averaging (WA) literature [46, 47, 48]: under shared pre-training, the different weights can be linearly interpolated despite the non-linearities in the architecture.

On the benefits of WARM. Firstly, WARM stands out for its efficiency and practicality. By requiring a single model at inference time, it provides a scalable approximation to the traditional, costlier ensembling of predictions, without its memory and inference burdens. Secondly, WARM improves reliability by inheriting from the generalization abilities of WA under distribution shifts, a quality well-documented in the OOD literature for supervised learning [47, 48, 49]. Lastly, WARM improves robustness to label corruption. We show that WA selects the invariant predictive mechanisms [50, 51] across different runs [52, 53], thus naturally diminishing the memorization of corrupted samples, occurring in each run in different ways. In contrast, ENS simply memorizes the corrupted samples. We also explain why reducing memorization when modeling noisy preferences enhances stability in the RL process. These multifaceted benefits of WARM are further explored in Section 4.

We summarize our contributions as follows.

- 1. Innovation in reward modeling. We introduce WARM, the first instance of weight averaging for reward modeling. This novel strategy efficiently mitigates reward hacking, improves reliability under distribution shifts and robustness to label corruption.
- 2. Theoretical and empirical insights into weight averaging. We validate linear mode connectivity for reward models trained on binary preference datasets. Moreover, we reveal a key difference between weight and prediction averaging, that appears clearly under label corruption; weight averaging only maintains the invariant predictive mechanisms across runs, thereby diminishing memorization and enhancing the focus on generalizable features.

Our experiments on summarization tasks in Section 5 confirm that WARM improves performance without any memory or inference overhead, either when used as the reward selector in best-of-𝑁, or as the proxy RM in RL. WARM mitigates reward hacking, and thus provides better downstream policies; specifically, it leads to a win rate of 79.4% (according to the preference oracle metric) against a policy trained with a standard RM.

#### 2. Context and challenges

###### 2.1. Context

LLMs. We consider an LLM 𝑓𝜃 of a fixed non-linear architecture parameterized by 𝜃, usually a Transformer with attention layers [54]. It defines a policy by mapping prompt inputs 𝑥 to 𝑓𝜃(𝑥). Following the foundation model paradigm [55] and the success of transfer learning [56], the weights 𝜃 are first pre-trained [4] on the vast amount of web data into 𝜃𝑝𝑡, before supervised fine-tuning (SFT) [7] to learn to follow instructions into 𝜃𝑠 𝑓𝑡. However, the high cost and limited scope of instruction data (i.e., prompts and responses) can create a misalignment [19, 32, 33] between the LLM and its intended application. Reinforcement learning (RL) as a third step in the training process of LLMs was shown to help alignment of LLMs with the intended usage [40].

RMs. A notable aspect of RL is the absence of supervised samples to be imitated by the policy; instead, the focus shifts to maximizing the reward of generated samples, that should measure their quality. The challenge is that the oracle reward, perfectly encapsulating the desired behaviors, is not given by the environment. The key innovation from RLHF [12] is that this reward is the output of a reward model (RM), trained in a supervised way to predict and thus reflect human preferences. Specifically,

an RM is an LLM 𝑟𝜙 parameterized by 𝜙, predicting a single scalar as the reward 𝑟𝜙(𝑥, 𝑦) for a prompt 𝑥 and generation 𝑦. The weights 𝜙 are usually initialized from 𝜃𝑠 𝑓𝑡, 𝜔 , where the final linear layer 𝜔 is added on top of the extracted features from the SFT model 𝜃𝑠 𝑓𝑡. Then 𝜙 is trained on a preference dataset D𝑡𝑟𝑎𝑖𝑛 = {𝑥𝑑, 𝑦+

𝑑 to continue 𝑥𝑑. Usually human labelers evaluate those generations, but recent works on RLAIF [57, 58] showed that similar performances can be obtained by prompting an LLM for AI feedback. Following the Bradley-Terry [59] assumption about the distribution of preferences, and by framing the problem as binary classification, the maximum likelihood principle motivates learning 𝜙 by minimizing the following negative log-likelihood loss (where 𝜎 is the logistic function):

𝑑 }𝑑𝐷=1 where the generation 𝑦+

𝑑 has been preferred over 𝑦−

𝑑 , 𝑦−

L𝑅 𝑟𝜙, D𝑡𝑟𝑎𝑖𝑛 = −𝔼(𝑥,𝑦+,𝑦−)∈D𝑡𝑟𝑎𝑖𝑛 log𝜎 𝑟𝜙 𝑥, 𝑦+ − 𝑟𝜙(𝑥, 𝑦−) . (1)

.

Reward inference. With this RM, the literature suggests applying any kind of RL algorithm (usually REINFORCE [60] or PPO [61]) to fine-tuned 𝜃𝑠 𝑓𝑡 into 𝜃𝑟𝑙, as analyzed in Section 5.2. A training-free alternative is best-of-𝑁 (BoN) sampling, analyzed in Section 5.1, which returns the generation that has the highest reward among 𝑁 generations from 𝜃𝑠 𝑓𝑡. Both methods aim to align the policy with human preferences. Yet, the reward misspecification [23] between the proxy RM and the true human preferences can lead to reward hacking [19, 20, 21, 22], where the policy exploits loopholes in the proxy RM to artificially increase the score without matching human preferences.

###### 2.2. Challenges in reward modeling

When handling rich inputs such as text, or when assessing complex behaviours, designing rewards aligned with human preferences is a complex challenge for two main reasons, described below.

Distribution shifts. The primary challenge is the distribution shifts resulting from the offline nature of preference data. Indeed, the generations in the preference dataset and those from the policy 𝜃𝑠 𝑓𝑡 do not necessarily follow the same distributions, and the shifts can become even more pronounced due to model drift during RL. The OOD generalization literature has extensively analyzed the repercussions of these shifts. Firstly, they often lead to a reduction in performance [62, 63]. RMs (of limited capacity) trained on narrow data distributions may rely on spurious correlations [51] or a limited number of features [64], thus failing when encountering OOD examples [65, 66]. Secondly, they complicate the selection of RMs, as ID validation metrics may poorly correlate with real-world OOD performances [67, 68] and the ability to guide the RL [41]. Lastly, RMs can become poorly calibrated [69] in OOD scenarios [70, 71], and predict more extreme values as rewards. Such miscalibration exacerbates the problem in a negative feedback loop, further intensifying model drift and distribution shifts. In conclusion, limited data coverage during reward modeling reduces the reliability of the RM and facilitates reward hacking [36] in regions where the RM is badly specified.

Inconsistent preferences. The second major challenge is the label noise in preference datasets. Human labelers, often grappling with fatigue, misunderstandings [72, 73] and imperfect incentives [74], might default to simpler criteria such as length, bullet points, or politeness rather than more causal indicators. This tendency is exacerbated for complex tasks [38] or when considering multiple objectives, ranging from harmlessness [75] to engagement [76] and representing the heterogeneity of human opinions. Consequently, these factors lead to low inter-rater agreement, where human data appears as an imperfect representation of the underlying ground truth [77, 78]. To mitigate these issues, there has been a shift towards AI-generated preferences [57, 58], which, while reducing human labor costs, introduces its own set of noise and failure cases, such as sensitivity to prompting strategies [79, 80]. These layers of noise and inconsistency challenge the robustness of the RM, and its ability to provide stable signals.

With this in mind, a good RM should ideally satisfy the three following properties.

- Property 1: efficiency. The RM should incur no memory or inference overhead. Then the policy can be optimized efficiently.

- Property 2: reliability. The RM should reliably reward predictions despite the distribution shifts. Then the policy can explore away from its initialization while relying on the RM.

- Property 3: robustness. The RM should be robust to the label inconsistencies in binary preferences. Then the policy can learn from robust signals given by the RM.

###### 2.3. Existing approaches

To tackle those issues, previous works have explored a few research directions, further detailed in our related work from Appendix A.2. During RL, the standard strategy is to encourage the policy to remain close to its SFT initialization with Kullback-Leibler (KL) regularization [81, 82]; KL reduces model drift [83, 84] but can cause underfitting and adds an extra hyperparameter (the regularization strength 𝛼). Collecting, labelling and then training on new data (reflecting the evolving policy) can improve the reliability of the RM [16]. Yet it poses significant efficiency challenges due to the continuous requirement for human annotation and computational resources. In contrast, active learning strategies [85, 86] proactively enrich the preference dataset by seeking out a diverse set of generations and potential failure cases. Concurrent work [87] suggests applying label smoothing and flipping. Finally, and most similar to WARM, prediction ensembling (ENS) [43] strategies average the logits from 𝑀 RMs. From a bias-variance perspective [88], ENS reduces the variance term when members are sufficiently diverse [89], and thus favors reliability under distribution shifts where variance is the key issue [47]. From a RL perspective, ENS was shown to mitigate hacking risks [12, 41, 42]. Despite its advantages, ENS faces efficiency challenges; the memory and inference costs grow linearly with 𝑀, making ENS incompatible with the scaling trend in RMs, where larger architectures consistently perform better [90]. Moreover, we will also show in Section 4.2 that ENS fails to improve robustness to preference inconsistencies.

#### 3. WARM

###### 3.1. Weight averaging of reward models

Facing those challenges in reward modeling and the limitations from existing approaches, we propose Weight Averaged Reward Models (WARM). WARM is a simple and efficient strategy that combines multiple models without the memory and inference overheads of prediction ensembling, enhancing reward reliability (under distribution shifts) and robustness (amidst noisy preference dataset). WARM is illustrated in Figure 1(a) and described below.

- 1. Shared pre-trained initialization. For a given pre-trained LLM, each RM is initialized from 𝜃𝑠 𝑓𝑡, 𝜔 combining SFT weights and a linear probed [91] classifier.
- 2. Diverse fine-tunings. We run 𝑀 RM fine-tunings, optimizing Equation (1) with diverse hyperparameters (as in a grid search), yielding 𝑀 weights {𝜙𝑖}𝑖𝑀=1.
- 3. Weight averaging. We average those 𝑀 weights together to form 𝜙WARM = 𝑀1 𝑖 𝑀=1 𝜙𝑖.

Then 𝑟𝜙WARM serves as the proxy RM to guide the RL procedure, as efficiently as an individual RM, but with the enhanced reliability and robustness provided by the WA strategy, that leverages the strengths and mitigates the weaknesses of the individual RMs.

###### 3.2. Linear mode connectivity

Compared to ENS, the main difference lies in how WARM combines the different RMs: we do so through linear interpolation in the weight space. It relies on the linear mode connectivity (LMC) [44, 45] property across fine-tuned weights, i.e., the fact that the accuracy of the interpolated model is at least as good as the interpolation of the individual accuracies. Precisely, by defining the pairwise accuracy of an RM 𝑟𝜙 w.r.t. a dataset D as Acc 𝑟𝜙, D = 𝔼(𝑥,𝑦+,𝑦−)∈D 1𝑟

𝜙(𝑥,𝑦+)≥𝑟𝜙(𝑥,𝑦−) , the following Observation 1 underpins the success of WARM.

- Observation 1 (LMC). Given two fine-tuned weights 𝜙1 and 𝜙2 with a shared pre-training and a test dataset D𝑡𝑒𝑠𝑡, then for all 𝜆 ∈ [0, 1],

Acc 𝑟(1−𝜆)·𝜙1+𝜆·𝜙2, D𝑡𝑒𝑠𝑡 ≥ (1 − 𝜆) × Acc 𝑟𝜙1, D𝑡𝑒𝑠𝑡 + 𝜆 × Acc 𝑟𝜙2, D𝑡𝑒𝑠𝑡 . (2)

We empirically validate this LMC in Figure 3, by evaluating interpolated RMs on OOD test samples. This follows similar observations for multi-class classification in the context of computer vision [44, 45], which led to a plethora of weight averaging (WA) works such as the model soups [46, 47, 48] variants (detailed in our related work in Appendix A.1).

- Remark 1 (Importance of pre-training and linear probing). The efficacy of WA can be surprising given the non-linearities [54] and permutation symmetries [92] in deep neural network architectures. WA is actually possible only because of the shared pre-training which constrains the divergence during fine-tunings [45], such as the weights remain in convex regions of the loss valley [93]. In contrast, the LMC does not hold when training weights from scratch [45], even if the random initialization is shared. For these reasons and to facilitate the LMC, we follow [47, 48] and use linear probing to initialize the classifier 𝜔; compared to random initialization, such linear probing prevents feature distortion [91].

###### 3.3. Sources of diversity

On one hand, WARM requires shared pre-training so that the fine-tuned weights remain linearly connected. On the other hand, weights must not be identical: actually, the diversity across those fine-tuned weights significantly contributes to the accuracy gains observed in WA [47]. Overall, an effective WARM requires a delicate trade-off between ensuring LMC and diversity across weights.

In practice, we use the following sources of diversity [94], leading the RM fine-tunings to diverse yet linearly connected models. First, the different fine-tunings see the data samples in different orders. Second, we sample slightly different hyperparameters, notably different learning rates and dropout probabilities, as detailed in Appendix B.3. Third, we investigate a new source of diversity in initialization named Baklava, illustrated in Figure 2. Specifically, we initialize the RMs’ featurizers from different checkpoints {𝜃𝑖𝑠 𝑓𝑡}𝑖𝑀=1 collected along a given SFT trajectory. Baklava relaxes the shared initialization constraint from model soups [46] to simply sharing the same pre-training: Baklava is actually an efficient alternative to model ratatouille [48] but without the need of multiple auxiliary tasks. Overall, Baklava increases diversity compared to only initializing from the last SFT checkpoint, while adhering to the shared pre-training requisite for LMC, without incurring any overhead.

𝜃𝑝𝑡 𝜃1𝑠 𝑓𝑡 𝜃2𝑠 𝑓𝑡 𝜃𝑀𝑠 𝑓𝑡

SFT Reward modelings Weight averaging

𝜙1 𝜙2 𝜙𝑀

𝜙WARM = 𝑀1 𝑖 𝑀=1 𝜙𝑖

- Figure 2 | Baklava diversity procedure. Starting from a pre-trained LLM 𝜃𝑝𝑡, we consider different checkpoints

{𝜃𝑖𝑠 𝑓𝑡}𝑖𝑀=1 along a single SFT run (dashed arrow ) collected at different number of SFT training steps. Those checkpoints serve as initializations for 𝑀 RM fine-tunings on the preference dataset (thick solid arrows )

to learn the {𝜙𝑖}𝑖𝑀=1. Finally, those RMs are weight averaged (dotted arrows ) into the final model 𝜙WARM. Following the culinary analogy from model soups [46] and model ratatouille [48], we named this method

Baklava because of its diamond geometric shape.

- Remark 2 (Moving average). Following stochastic weight average [95] or moving average [96], we also tried to average checkpoints collected along a single RM fine-tuning. Though interesting because less costly for training, the lower results in Figure 3(a) suggest that the accuracy-diversity trade-off was not favorable: incorporating early checkpoints would compromise individual accuracies, and considering only later checkpoints would not bring the necessary diversity. As a result, we opted to use in WARM only the last checkpoint from each RM fine-tuning.

#### 4. On the benefits of WARM

We now explore the properties and benefits from the WARM strategy, previously described in Section 3. We ground our analysis on the empirical comparison between WA and ENS for reward modeling, and a novel general theoretical comparison in Section 4.3.

Experimental setup. We leverage the TL;DR summarization benchmark [97], a standard in reward modeling for LLMs, that we briefly describe below and further detail in Appendix B. The goal of the RMs is to score summaries such as they are ranked properly. In training, we use the dataset D𝑡𝑟𝑎𝑖𝑛 from Stiennon et al. [14] where the candidate summaries are generated by GPT-3 [6] variants. To obtain the labels, we follow the RLAIF procedure from [58], where a PaLM-L [98] is prompted with chain-of-thought [99] to generate feedback mimicking human preferences. This strategy performs similarly to human labelers with similar inter-agreement, and will be useful in Section 5 as an oracle metric. The RMs are PaLM-XXS models, pre-trained and SFT-ed on the preferred summaries from D𝑡𝑟𝑎𝑖𝑛, on which we plug a linear probed [91] classification layer. We train the RMs for 10k steps on D𝑡𝑟𝑎𝑖𝑛, with hyperparameters and procedure detailed in Appendix B.3. We report accuracies of those RMs on a novel out-of-distribution (OOD) test dataset D𝑜𝑜𝑑 with 92k pairwise comparisons where the summaries are generated by multiple PaLM-XS policies with high temperature, some of which are pre-trained only, others SFT-ed and others RLHF-ed.

###### 4.1. 1st order analysis: weight averaging for reliable and more efficient ensembling

Previous works [46, 47, 95] have argued that the best way to understand WA is as an efficient approximation of ENS, as clarified in Observation 2.

- Observation 2 (WA and ENS: 1st order analysis). Weight averaging and prediction ensembling perform similarly: i.e., for all 𝜆 ∈ [0, 1] and a test dataset D𝑡𝑒𝑠𝑡,

Acc 𝑟(1−𝜆)·𝜙1+𝜆·𝜙2, D𝑡𝑒𝑠𝑡 ≈ Acc (1 − 𝜆) × 𝑟𝜙1 + 𝜆 × 𝑟𝜙2, D𝑡𝑒𝑠𝑡 . (3)

| |WA<br><br>ENS<br><br>Diag| | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

0.765

0.764

0.763

0.762

Acc.

0.761

0.760

0.759

0.758

0.0 0.2 0.4 0.6 0.8 1.0

(a) 1 RM fine-tuning at 2 different training steps.

| |WA<br><br>ENS<br><br>Diag| | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

0.765

0.764

0.763

0.762

Acc.

0.761

0.760

0.759

0.758

0.0 0.2 0.4 0.6 0.8 1.0

(b) 2 RM fine-tunings with shared config.

| |WA<br><br>ENS<br><br>Diag| | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

0.765

0.764

0.763

0.762

Acc.

0.761

0.760

0.759

0.758

0.0 0.2 0.4 0.6 0.8 1.0

(c) 2 RM fine-tunings with different learning rates.

| |WA<br><br>ENS<br><br>Diag| | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

0.765

0.764

0.763

0.762

Acc.

0.761

0.760

0.759

0.758

0.0 0.2 0.4 0.6 0.8 1.0

(d) 2 RM fine-tunings with different inits: Baklava.

- Figure 3 | Experiments under distribution shifts validating Observations 1 and 2 on the TL;DR summarization benchmark [97]. We report the accuracies on D𝑜𝑜𝑑 when interpolating between two RM weights 𝜙1 and 𝜙2 with the coefficient 𝜆 sliding between 0 and 1. WA stands for weight averaging 𝑟(1−𝜆)·𝜙1+𝜆·𝜙2 while ENS combines the predictions (1−𝜆) ×𝑟𝜙1 +𝜆 ×𝑟𝜙2; Diag is the interpolated accuracy (1 − 𝜆)×Acc 𝑟𝜙1 +𝜆 ×Acc 𝑟𝜙2 . We consider sources of increasing diversity [94] between 𝜙1 and 𝜙2: in Figure 3(a), they are collected at different number of training steps (8k and 10k) along a single RM fine-tuning; in Figure 3(b), they are from two independant RM fine-tunings, with the exact same config, but seeing the data in different orders; in Figure 3(c), they have different learning rates (1e-4 and 4e-5); in Figure 3(d), they are initalized from different SFT checkpoints collected at different number of SFT steps (8k and 12k), per Baklava introduced in Figure 2.

0.0 0.2 0.4 0.6 0.8 1.0

- 0.45

- 0.46

- 0.47

- 0.48

- 0.49

Acc.

WA

ENS

Diag

(a) Train (corrupt).

0.0 0.2 0.4 0.6 0.8 1.0

0.894

0.896

0.898

0.900

0.902

Acc.

WA

ENS

Diag

(b) Train (clean).

0.0 0.2 0.4 0.6 0.8 1.0

0.766

0.768

0.770

0.772

0.774

0.776

0.778

Acc.

WA

ENS

Diag

(c) Validation (ID).

0.0 0.2 0.4 0.6 0.8 1.0

0.708

0.710

0.712

0.714

0.716

Acc.

WA

ENS

Diag

(d) Test (OOD).

- Figure 4 | Corruption experiment validating Observation 3. We consider 𝜙1 and 𝜙2, two RMs fine-tuned independently with the same config as in Figure 3(b), but this time with 25% of the training labels corrupted. We then report the performances of their WA and ENS on the different data subsets. We observe that WA reduces memorization of the corrupted labels in Figure 4(a), and still performs slightly worse than ENS on the clean training samples in Figure 4(b); yet, the performances of WA w.r.t. ENS improves as we move away from the training distribution, in particular on D𝑜𝑜𝑑 in Figure 4(d) where WA generalizes better.

Theoretically, a simple Taylor expansion can justify this similarity when ∥𝜙1 − 𝜙2∥ ≪ 1. Empirically, this is validated in Figure 3 where the accuracy curves on D𝑜𝑜𝑑 for WA and ENS closely match. This similarity justifies that WA is a variance reduction method; then, because variance is the dominant issue under distribution shifts [47], this explains the significant gains in Figure 3 over the individual RMs 𝜙1 and 𝜙2 (validating Observation 1), in particular when weights are sufficiently diverse. This suggests improved reliability in WARM, with efficiency benefits over ENS: indeed, WA maintains a single set of weights, removing the memory and inference overheads from ENS.

###### 4.2. 2nd order analysis: weight averaging for more robust ensembling

A surprising fact remains unexplained. WA is slightly superior to ENS under distribution shifts, which one can see on the plots from Figure 3, and more consistently in Figure B.1 from model soups [46] or in Figure 1 from DiWA [47]. More generally, WA is the state-of-the-art strategy for OOD generalization, consistently outperforming ENS; yet, this was not explained in previous works, thus urging for new insights about the difference between WA and ENS.

Corruption setup. To refine our understanding on the difference between WA and ENS, we propose a new setup where 25% of the binary labels are swapped in training. We then report the per-subset accuracies on Figure 4, enriched in Appendix C.1 and aggregated in Figure 5. On the corrupted subset of training data, the accuracy curve for WA is below the expected accuracies, while it is above on all other subsets. More precisely, we make the following Observation 3.

- Observation 3 (WA and ENS: 2nd order analysis). The accuracy gains of WA over ENS grow as data moves away from the training distribution.

- • WA ≪ ENS on train corrupt: WA is far worse than ENS on train samples with swapped labels, showing reduced memorization and improved robustness to label corruption.
- • WA ≤ ENS on train clean: WA is worse than ENS on train samples with correct labels.
- • WA ⪆ ENS on ID val: WA is better or similar to ENS on samples without distribution shifts.
- • WA ≥ ENS on OOD test: WA is far better than ENS on test samples from new distributions, showing better reliability under distribution shifts.

Train corrupted Train clean Val ID Test OOD

| |
|---|

200

Overall, this suggests that weight averaging memorizes less and generalizes better than ensembling predictions.

| |
|---|

| |
|---|

150

Density

###### 4.3. Weight averaging enforces invariance across runs

100

We now provide theoretical support to this Observation 3. In brief, our simplifying assumptions suggest that WA acts as a regularization towards the predictive mechanisms that are invariant across runs, i.e., learned simultaneously in each independent run. Then, in contrast with ENS, WA would improve robustness to corruption because it would underweight the run-specific features (with low probability of being learned) inducing memorization.

50

0

0.06 0.05 0.04 0.03 0.02 0.01 0.00 0.01 0.02

Accuracy gain of WA over ENS

Figure 5 | Histograms of the differences in accuracy between WA and ENS on different data subsets.

Setup. We follow Lin et al. [53], and consider a simplified binary classification setup with labels 𝑦 ∈ {−1, 1}, related to 𝐹 features {𝑧𝑗}𝐹𝑗=1 such as 𝑧𝑗 ∈ ℝ𝑑. From inputs 𝑥, we train a binary classifier 𝑟(𝑥) = 𝜔⊺ 𝑓 (𝑥). Following [53], we make three key assumptions. First, features orthogonality: we assume that {𝑧𝑗}𝐹𝑗=1 are orthogonal, i.e., (𝑧𝑗)⊺𝑧𝑗′ = 0 when 𝑗 ≠ 𝑗′. Second, input as bag of features: we assume that the input 𝑥 = 𝑥𝑗 𝐹𝑗=1 ∈ ℝ𝐹×𝑑 can be represented as the concatenation of 𝑥𝑗 generated by 𝑥𝑗 ∼ N 𝑦 · 𝑧𝑗, 𝜎 · I𝑑 with 𝜎 ≪ 1. Finally, the binary featurizer assumption: we consider that the featurizer 𝑓 = 𝑓 𝑗 𝐹𝑗=1 ∈ {0, 1}𝐹 is a binary selector of the features that make the input. For example, if 𝑦 = 1, 𝐹 = 3, 𝑥 ≈ [𝑧1, 𝑧2, 𝑧3], and 𝑓 = [1, 0, 1] learns to extract the first and third features, then 𝑓 (𝑥) ≈ 𝑧1 + 𝑧3. We denote 𝑝𝑗 the probability that the featurizer 𝑓 learns to use the 𝑗-th feature dimension (associated with 𝑧𝑗); this means 𝑓 𝑗 is 1 with probability 𝑝𝑗 and 0 otherwise. Moreover, for infinite training samples and under some constraint on 𝜎, Lemma 5 in [53] proved that, to learn 𝑟 = 𝜔⊺ 𝑓, the optimal linear fit 𝜔 on the features selected from 𝑓 would be 𝜔 = 𝐹𝑗=1 𝑓 𝑗 · 𝑧𝑗.

Results. We consider 𝑀 RMs {𝑟𝑖 = 𝜔𝑖⊺ 𝑓𝑖}𝑖𝑀=1, and compare the limit behaviours of their prediction ensembling 𝑟𝑀𝐸𝑁𝑆 and weight averaging 𝑟𝑊𝐴𝑀 when 𝑀 → ∞. In this limit case, the averaged prediction 𝑟𝑀𝐸𝑁𝑆 = 𝑀1 𝑖 𝑀=1 𝜔𝑖⊺ 𝑓𝑖 for an input 𝑥 from label 𝑦 tends towards the expected prediction 𝔼[𝑟(𝑥)] = 𝔼[𝜔⊺ 𝑓 (𝑥)] = 𝔼{𝑓𝑗}𝐹𝑗=1

𝐹 𝑗=1 𝑓 𝑗 · 𝑧𝑗 ⊺( 𝐹𝑗′=1 𝑓 𝑗′ · 𝑥𝑗′) ≈ 𝑦 · 𝐹𝑗=1 𝑝𝑗 · |𝑧𝑗|2, using 𝑥𝑗′ ≈ 𝑦 · 𝑧𝑗′ thus

(𝑧𝑗)⊺𝑥𝑗′ ≈ 0 when 𝑗 ≠ 𝑗′, and ( 𝑓 𝑗)2 = 𝑓 𝑗.

∑︁𝐹

𝒑𝒋 · |𝑧𝑗|2. (4)

###### 𝑟𝑀𝐸𝑁𝑆(𝑥) −−−−−→

𝔼[𝑟(𝑥)] ≈ 𝑦 ·

𝑀→∞

𝑗=1

⊺

In contrast, when considering 𝑟𝑊𝐴𝑀 = 𝑀 1 𝑖 𝑀=1 𝜔𝑖

1

𝑀 𝑖=1 𝑓𝑖 with 𝑀 → ∞, we have

𝑀

𝔼[ 𝑓] = 𝑝𝑗 𝐹𝑗=1 and 𝑀1 𝑖 𝑀=1 𝜔𝑖 −−−−−→

1

𝔼[𝜔] = 𝐹𝑗=1 𝑝𝑗 · 𝑧𝑗, and thus:

###### 𝑀 𝑖=1 𝑓𝑖 −−−−−→

𝑀

𝑀→∞

𝑀→∞

∑︁ 𝐹

∑︁ 𝐹

###### ∑︁𝐹

⊺

𝒑2𝒋 · |𝑧𝑗|2. (5)

𝑝𝑗′ · 𝑥𝑗′ ≈ 𝑦 ·

###### 𝑟𝑊𝐴𝑀 (𝑥) −−−−−→

𝑝𝑗 · 𝑧𝑗

𝑀→∞

𝑗=1

𝑗′=1

𝑗=1

Interpretation. For ENS, the coefficient for a given feature is 𝒑𝒋, the same as the probability of this information being used by any individual network. In contrast, WA involves the square of the

probability 𝒑2𝒋 . Thus WA reduces the reliance on features with low probability, related to minor specific information (such as noise or context) which can be used to fit the corrupted training samples;

this would reduce memorization, and thus explains the robustness of WA under label corruption. Reciprocally, WA tends to prioritize the most probable features, favoring the mechanisms that are consistently learned, in other words the mechanisms invariant across runs. Overall, WA acts as a regularization, improving robustness under label corruption by tackling run-specific mechanisms favoring memorization, and improving reliability under distribution shifts by preserving run-invariant mechanisms favoring generalization.

- Remark 3 (Invariance). We argue that weight averaging only keeps the invariant predictive mechanisms across runs. This is in analogy with the invariance literature [50], popular for domain generalization [51, 100] under spurious correlations, where the key idea is that the predictive mechanisms which are invariant across domains are the causal ones that are stable under distribution shifts. This theoretically connects two key paradigms for OOD generalization, ensembling and invariance, and shows that weight averaging actually benefits from both.
- Remark 4 (Extension to a deeper structure with 𝐿 layers). We obtain a square in 𝒑2𝒋 due to our simplified two-layer architecture. Yet, in full generality, using a deeper structure with 𝐿 layers would

lead to 𝒑𝑳𝒋. Intuitively, WA applies an AND-mask on the information, that need to be found both in the previous feature space and the next layer weights.

- Remark 5 (From reward robustness to learnability). When applied to the design of RMs in WARM, we now argue that WA facilitates WARM’s stability [87] by mitigating the reliance on some non-robust features. Indeed, WA makes the WARM reward more robust to small (potentially adversarial [101]) perturbations [102], i.e., smoother [103] in the input space. This relates to the Lipschitzness property of the reward [104, 105, 106], where the difference in predicted rewards is bounded by the distance in input space. Fortunately, such smoothness is useful in RL [107], in particular for the stability of the policy gradient [108] because “sharp changes in reward value are hard to represent and internalize” [109]. This is studied in Lipschitzness is all you need [109] where the authors argue that “the local Lipschitzness of the reward is a sine qua non condition for good performance”, required “to even learn anything”. In summary, robustness improves stability and hinders the cascade of errors occurring when minor input variations can cause large reward differences.

In conclusion, we summarize the benefits from WARM. First, WARM is efficient, incurring no memory or computation costs, as it returns a single model. Second, WARM reduces variance while leveraging mechanisms invariant across runs, thus improving its reliability under distribution shifts. Lastly, WARM also addresses label corruption, thereby augmenting robustness to noisy preferences.

#### 5. Experiments

To empirically validate WARM’s benefits described in previous section, we train PaLM-XXS RMs on the TL;DR summarization benchmark [97] where preference labels are generated by a PaLM-L model prompted with chain-of-thought [99]. This AI labeling approach, increasingly common in recent research [26, 41, 110] as an efficient alternative to human assessments, is motivated by studies [57, 58] indicating that it correlates well with human preferences: critically, it provides an automatic pairwise oracle preference metric to evaluate reward hacking (in a similar fashion to the distillation setup from [17], discussed in Appendix C.4). In addition, we leverage a PaLM-XS RM for pointwise

WARM M = 6 WARM M = 2 ENS M = 2

0.14

0.12

Ind 2 Ind 1

0.10

Controlrewardgain

0.08

0.06

0.04

0.02

- 0.00

0.2 0.4 0.6 0.8 1.0 1.2

KL : log(N) NN 1

(a) PaLM (clean).

0.25

WARM M = 6 WARM M = 2 ENS M = 2

0.20

Ind 2 Ind 1

Controlrewardgain

0.15

0.10

0.05

0.00

0.2 0.4 0.6 0.8 1.0 1.2

KL : log(N) NN 1

(b) PaLM (corrupt).

0.35

WARM M = 6 WARM M = 2 ENS M = 2

0.30

Ind 2 Ind 1

0.25

Controlrewardgain

0.20

0.15

0.10

0.05

0.00

0 1 2 3 4 5 6

KL : log(N) NN 1

(c) T5 (clean).

WARM M = 6 WARM M = 2 ENS M = 2

0.6

0.5

Ind 2 Ind 1

0.4

Controlrewardgain

0.3

0.2

0.1

0.0

0.1

0 1 2 3 4 5 6

KL : log(N) NN 1

(d) T5 (corrupt).

- Figure 6 | Control reward for BoN experiments: clean preference dataset in Figures 6(a) and 6(c) and 25% corruptions in Figures 6(b) and 6(d). We consider two SFT policies to generate candidate summaries: one based on PaLM architecture [98], the other on T5 architecture [111]. The 𝑥-axis is the KL between the BoN policy and the SFT policy; the 𝑦-axis represents the control reward gains w.r.t. to an RM 𝜙1, which was the best individual RM on D𝑜𝑜𝑑. The blue lines represent WARM with 𝑀 weights: WARM performs higher than the individual RMs (in yellows) or when ensembling their predictions (ENS in red). We report the absolute control rewards for those experiments in Figure 15, where the values range roughly between 3 and 7.

0 1 2 3 4 5 6

KL : log(N) NN 1

0.750

0.775

0.800

0.825

0.850

0.875

0.900

0.925

Winratiovs.SFT

WARM M = 6 WARM M = 2 ENS M = 2

Ind 2 Ind 1

(a) SFT (clean).

0 1 2 3 4 5 6

KL : log(N) NN 1

0.70

0.75

0.80

0.85

0.90

Winratiovs.SFT

WARM M = 6 WARM M = 2 ENS M = 2

Ind 2 Ind 1

(b) SFT (corrupt).

| | | | | | | |
|---|---|---|---|---|---|---|
|WARM M = 2<br><br>ENS M = 2<br><br>Ind 2 Ind 1 y=0.5<br><br>| | | | | | |
| | | | | | | |

0 1 2 3 4 5 6

KL : log(N) NN 1

0.38

0.40

0.42

0.44

0.46

0.48

0.50

Winratiovs.WARM=6M

(c) WARM (clean).

| | | | | | | |
|---|---|---|---|---|---|---|
|WARM M = 2<br><br>ENS M = 2<br><br>Ind 2 Ind 1 y=0.5<br><br>| | | | | | |
| | | | | | | |

0 1 2 3 4 5 6

KL : log(N) NN 1

0.300

0.325

0.350

0.375

0.400

0.425

0.450

0.475

0.500

Winratiovs.WARM=6M

(d) WARM (corrupt).

- Figure 7 | Oracle preference metric for BoN experiments on T5 generations: clean preference dataset in Figures 7(a) and 7(c) and 25% corruptions in Figures 7(b) and 7(d). We plot the win rates for different values of 𝑁 vs. two reference strategies: SFT (i.e., random selection or equivalently BoN with 𝑁 = 1), or selecting the best summary according to WARM 𝑀 = 6. We observe that all strategies beat the SFT reference (they are all above 50% win rate), but that none beat the WARM 𝑀 = 6 reference.

control reward reaching 80.1% accuracy on the OOD dataset D𝑜𝑜𝑑. As verified in our experiments, this control RM also detects hacking, as it benefits from a larger architecture and a disjoint pretraining compared to the PaLM-XXS RMs of interest. Below, we explore two key scenarios: in Section 5.1, WARM reranks outputs in best-of-𝑁 (BoN); in Section 5.2, WARM guides the RL procedure.

###### 5.1. Best-of-𝑁 experiments

Setup. We start with best-of-𝑁 (BoN) sampling experiments in Figures 6 and 7. Given a dataset of 𝐷 text prompts, for each prompt we generate 𝑁 summaries from a SFT policy, and then returns the summary with the highest reward according to different RMs. We actually consider two SFT policies; one based on PaLM architecture [98] (𝑁 = 8, 𝐷 = 15000), the other on T5 architecture [111] (𝑁 = 1000, 𝐷 = 1000). For the 𝑥-axis, we plot the KL between the BoN policy and the SFT policy, which can be approximated by log(𝑁)− 𝑁𝑁−1 [112, 113]. BoN is effective [16], especially in the low-KL regime (i.e., for small 𝑁). We consider two setups, without (clean setup) and with (corrupt setup) 25% label corruption in the preference datasets for reward modeling, and denote in each setup the weights {𝜙𝑖}𝑖𝑀=1 sorted in decreasing accuracy on D𝑜𝑜𝑑.

Control reward. Figure 6 shows that, in terms of pointwise control reward, WARM performs consistently better than ENS (only with 𝑀 = 2 for computational reasons) and the two best individual RMs 𝜙1 and 𝜙2; moreover, the gains get bigger for 𝑀 = 6. As a side note, we also observe that the individual RM 𝜙2 performs better in BoN in Figure 6(c) than 𝜙1 though 𝜙1 was better than 𝜙2 on D𝑜𝑜𝑑, highlighting that selecting the appropriate individual RM is not trivial [41].

Oracle preference. In Figure 7, we leverage the pairwise oracle preference [58] metric to validate better performance with WARM. We observe in Figures 7(a) and 7(b) that summaries selected with WARM have a win rate of up to 92.5% against the random selection of a summary (from SFT). We also see in Figures 7(c) and 7(d) that reciprocally, all selection strategies have a win rate lower than

- 50% against the summaries selected by WARM 𝑀 = 6.

###### 5.2. RL experiments

Setup. For RL fine-tuning of policies, we follow [58] and use their modified version of REINFORCE [60] with a baseline value score for variance reduction, a simpler algorithm than PPO [61] yet still effective for LLMs. Both policy and value LLMs are PaLM-XS, initialized from the same SFT model. We then generate samples with the policy, compute the reward with the RMs and update the weights to optimize this reward. More details are available in Appendix B.4. To reduce forgetting and encourage the policy to remain close to its SFT initialization, we incorporate a KL regularization [81, 82] controlled by a coefficient 𝛼, ablated in Figure 8(c), yet otherwise set to 0.003 in the clean setup and

- 0.01 in the corrupt setup. This KL serves as the 𝑥-axis in our plots to estimate model drift, as done in the literature; same curves with the number of training steps as the 𝑥-axis in Figures 1(b) and 18.

Control reward. In Figure 8, we observe reward hacking; as the policy moves away from its SFT initialization, the control reward collapses. Critically, WARM improves performances: in particular, increasing 𝑀 pushes the Pareto front of solutions to the top left in Figures 8(a) and 8(b). In comparison, policies trained with ENS (with 𝑀 = 2 for computational reasons) are still susceptible to early reward hacking, while reaching absolute control rewards significantly worse than with WARM (even with 𝑀 = 2). In Figure 8(c), we confirm that the 𝛼 hyperparameter plays a crucial role; low values of 𝛼 such as 0.001 correspond to high KL, while high values of 𝛼 such as 0.01 entail low KL but a risk of underfitting. From a practical perspective, this highlights that the optimal value of 𝛼 for WARM is lower than for a single RM; this is because WARM can mitigate reward hacking, and thus the optimal policies are obtained for larger values of KL.

- 3

- 4

- 5

- 6

- 7

- 8

- 9

- 10

- 11

Controlreward

WARM M = 10

WARM M = 6 WARM M = 2 ENS M = 2

Ind 2 Ind 1

0 200 400 600 800 1000

KL

(a) RL (clean).

- 4

- 5

- 6

- 7

Controlreward

WARM M = 6 WARM M = 2 ENS M = 2

Ind 2 Ind 1

0 25 50 75 100 125 150 175

KL

(b) RL (corrupt).

10

8

Controlreward

6

WARM = 0.01

4

WARM = 0.003 WARM = 0.001 Ind = 0.01

Ind = 0.003 Ind = 0.001

2

0 250 500 750 1000 1250 1500 1750

KL

(c) Ablating 𝛼 for RL (clean).

- Figure 8 | Control reward for RL experiments: clean preference dataset in Figures 8(a) and 8(c) and 25% corruptions in Figure 8(b). The blue lines show the RL fine-tuning of policies when averaging 𝑀 weights as the RM; the darker, the higher the 𝑀. It performs higher than when RL fine-tuning with the individual RMs (in yellows) or when ensembling their predictions (in red). Figure 8(c) shows results of policies RL fine-tuned with WARM 𝑀 = 6 or 𝜙1, for different values of 𝛼 controlling the KL regularization strength.

2000 3000 4000 5000 6000

# steps

0.5

0.6

0.7

0.8

0.9

1.0

Winratiovs.SFT

WARM M = 10

WARM M = 6 WARM M = 2 ENS M = 2

Ind 2 Ind 1 y=0.5

(a) SFT.

| | | | | | |
|---|---|---|---|---|---|
|WARM M = 10<br><br>WARM M = 6 WARM M = 2 ENS M = 2<br><br>Ind 2 Ind 1 y=0.5<br><br>| | | | | |
| | | | | | |

2000 3000 4000 5000 6000

# steps

0.0

0.1

0.2

0.3

0.4

0.5

Winratiovs.WARM=6atstep3500M

(b) WARM 𝑀 = 6.

| | | | | | |
|---|---|---|---|---|---|
|WARM M = 10<br><br>WARM M = 6 WARM M = 2 ENS M = 2<br><br>Ind 2 Ind 1 y=0.5<br><br>| | | | | |
| | | | | | |

2000 3000 4000 5000 6000

# steps

0.0

0.1

0.2

0.3

0.4

0.5

0.6

0.7

0.8

Winratiovs.Indatstep30001

(c) 𝜙1 (best individual RM).

- Figure 9 | Oracle preference metric for RL experiments: clean preference dataset. We plot the win rates along RL fine-tuning against three reference policies: the SFT policy, the policy RL fine-tuned with WARM

𝑀 = 6 after 3500 steps, and the policy RL fine-tuned with 𝜙1 after 3000 steps. Figure 19 reports results when comparing policies at fixed number of training steps.

Oracle preference. In Figure 9, we compare the different policies according to our pairwise oracle preference AI labeler [58]. In Figure 9(a), the reference policy is the SFT initialization; all the RL fine-tuned policies outperform this baseline, with WARM 𝑀 = 6 reaching a win rate of 99.8% after 3500 steps (the highest win rate among all policies). We use this policy as the reference in Figure 9(b); no other policy could beat it. Interestingly, we observe that using 𝑀 = 10 rewards can delay reward hacking but does not improve the peak performance; we speculate this is related to our weight selection procedure, as the weights {𝜙𝑖}10𝑖=7 have lower individual accuracy on D𝑜𝑜𝑑 than {𝜙𝑖}6𝑖=1 (more details in Figure 10). Finally, in Figure 9(c), the reference policy is obtained after 3000 steps of RL fine-tuning with 𝜙1 (the best individual RM on D𝑜𝑜𝑑). There is a large region of steps in which policies trained WARM (even with 𝑀 = 2) beat this approach; the previous reference from Figure 9(b) actually has a 79.4% win rate against it.

#### 6. Discussion

Benefits. WARM represents a flexible and pragmatic method to improve the alignment of AI with human values and societal norms. This paper has detailed several of its benefits, and below, we delve into additional, more exploratory advantages. WARM follows the updatable machine learning paradigm [114], eliminating the need for inter-server communication, thus enabling embarrassingly simple parallelization [115] of RMs. This facilitates its use in federated learning scenario [116] where the data should remain private; moreover, WA would add a layer of privacy and bias mitigation by reducing the memorization of private preference [52]. Then, a straightforward extension of WARM would combine RMs trained on different datasets, for example, coming from different (clusters of) labelers. This diversity could help WARM performances, but also from a multi objective perspective [117]; by non-uniform interpolation of RMs, we could learn a set of personalized policies [39]. Furthermore, as WA has been shown to limit catastrophic forgetting [118, 119], WARM could seamlessly support iterative and evolving preferences. Finally, a promising research direction is extending WARM to direct preference optimization (DPO) strategies [120], where averaging the RMs casts back to averaging the DPO policies [121].

Limitations. WARM, while innovative, does face some limitations, notably two when compared to prediction ensembling methods; first, prediction ensembling can benefit from the diversity brought by combining RMs from various architectures and pre-trainings; second, prediction ensembling can incorporate prediction disagreement into the reward to provide uncertainty estimation and limit model drift. However, it’s been noted in [41] that simple averaging of logits often performs comparably to more complex prediction aggregation functions that include uncertainty elements. Another limitation is that, while WARM effectively reduces certain types of memorization, it does not completely eradicate all forms of spurious correlations or biases inherent in the preference data. For instance, if each individual RM predominantly relies on summary length as a criterion, WARM is likely to replicate this tendency. Therefore, alternative methods (from the OOD generalization literature?) might be required, for example those based on invariance regularization [51, 100] or last layer retraining [122]. Finally, WARM only enhances reward modeling without tackling the other challenges in RLHF [18]; thus, to mitigate the safety risks [19, 34, 35] from misalignment [32, 33], WARM must be considered within the larger context of responsible AI.

#### 7. Conclusion

In conclusion, we introduce Weight Averaged Reward Models (WARM) to address two critical challenges in reward modeling: reliability under distribution shifts and robustness under label corruption. By averaging the weights of multiple RMs obtained from diverse fine-tunings, WARM appears as an efficient solution to mitigate reward hacking in reinforcement learning from human feedback. Our empirical results demonstrate its effectiveness when applied to summarization. We anticipate that WARM will contribute to more aligned, transparent, and effective AI systems, encouraging further exploration in reward modeling.

#### References

- [1] Google Gemini Team. Gemini: A family of highly capable multimodal models. 2023. (p. 1)
- [2] OpenAI. Gpt-4 technical report. 2023. (p. 1)
- [3] Sébastien Bubeck, Varun Chandrasekaran, Ronen Eldan, Johannes Gehrke, Eric Horvitz, Ece Kamar, Peter Lee, Yin Tat Lee, Yuanzhi Li, Scott Lundberg, et al. Sparks of artificial general intelligence: Early experiments with gpt-4. arXiv preprint, 2023. (p. 1)
- [4] Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. Improving language understanding by generative pre-training. 2018. (pp. 1 and 3)
- [5] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In NAACL, 2019. (p. 1)
- [6] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In NeurIPS, 2020. (pp. 1, 7, and 27)
- [7] Jason Wei, Maarten Bosma, Vincent Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M. Dai, and Quoc V Le. Finetuned language models are zero-shot learners. In ICLR,

2022. (pp. 1 and 3)

- [8] Yizhong Wang, Swaroop Mishra, Pegah Alipoormolabashi, Yeganeh Kordi, Amirreza Mirzaei, Atharva Naik, Arjun Ashok, Arut Selvan Dhanasekaran, Anjana Arunkumar, David Stap, Eshaan Pathak, Giannis Karamanolakis, Haizhi Lai, Ishan Purohit, Ishani Mondal, Jacob Anderson, Kirby Kuznia, Krima Doshi, Kuntal Kumar Pal, Maitreya Patel, Mehrad Moradshahi, Mihir Parmar, Mirali Purohit, Neeraj Varshney, Phani Rohitha Kaza, Pulkit Verma, Ravsehaj Singh Puri, Rushang Karia, Savan Doshi, Shailaja Keyur Sampat, Siddhartha Mishra, Sujan Reddy A, Sumanta Patro, Tanay Dixit, and Xudong Shen. Super-NaturalInstructions: Generalization via declarative instructions on 1600+ NLP tasks. In ACL, 2022. (p. 1)
- [9] Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B Hashimoto. Stanford Alpaca: An instruction-following LLaMA model,

2023. (p. 1)

- [10] Paul Roit, Johan Ferret, Lior Shani, Roee Aharoni, Geoffrey Cideron, Robert Dadashi, Matthieu Geist, Sertan Girgin, Léonard Hussenot, Orgad Keller, et al. Factually consistent summarization via reinforcement learning with textual entailment feedback. In ACL, 2023. (p. 1)
- [11] Lev McKinney, Yawen Duan, David Krueger, and Adam Gleave. On the fragility of learned reward functions. arXiv preprint, 2023. (p. 1)
- [12] Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. In NeurIPS, 2017. (pp. 1, 2, 3, 5, and 27)
- [13] Daniel M Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B Brown, Alec Radford, Dario Amodei, Paul Christiano, and Geoffrey Irving. Fine-tuning language models from human preferences. arXiv preprint, 2019. (pp. 1 and 27)

- [14] Nisan Stiennon, Long Ouyang, Jeffrey Wu, Daniel Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul F Christiano. Learning to summarize with human feedback. NeurIPS, 2020. (pp. 1, 7, and 27)
- [15] Jeff Wu, Long Ouyang, Daniel M Ziegler, Nisan Stiennon, Ryan Lowe, Jan Leike, and Paul Christiano. Recursively summarizing books with human feedback. arXiv preprint, 2021. (pp. 1 and 27)
- [16] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. LLaMA 2: Open foundation and fine-tuned chat models. arXiv preprint, 2023. (pp. 1, 5, 12, and 27)
- [17] Leo Gao, John Schulman, and Jacob Hilton. Scaling laws for reward model overoptimization. In ICML, 2023. (pp. 1, 11, and 33)
- [18] Stephen Casper, Xander Davies, Claudia Shi, Thomas Krendl Gilbert, Jérémy Scheurer, Javier Rando, Rachel Freedman, Tomasz Korbak, David Lindner, Pedro Freire, et al. Open problems and fundamental limitations of reinforcement learning from human feedback. TMLR, 2023. (pp. 1 and 14)
- [19] Dario Amodei, Chris Olah, Jacob Steinhardt, Paul Christiano, John Schulman, and Dan Mané. Concrete problems in AI safety. arXiv preprint, 2016. (pp. 1, 2, 3, 4, and 14)
- [20] Jack Clark and Dario Amodei. Faulty Reward Functions in the Wild. https://openai.com /research/faulty-reward-functions, 2016. (pp. 1 and 4)
- [21] Amanda Askell, Yuntao Bai, Anna Chen, Dawn Drain, Deep Ganguli, Tom Henighan, Andy Jones, Nicholas Joseph, Ben Mann, Nova DasSarma, Nelson Elhage, Zac Hatfield-Dodds, Danny Hernandez, Jackson Kernion, Kamal Ndousse, Catherine Olsson, Dario Amodei, Tom Brown, Jack Clark, Sam McCandlish, Chris Olah, and Jared Kaplan. A general language assistant as a laboratory for alignment. arXiv preprint, 2021. (pp. 1 and 4)
- [22] Joar Max Viktor Skalse, Nikolaus H. R. Howe, Dmitrii Krasheninnikov, and David Krueger. Defining and characterizing reward gaming. In NeurIPS, 2022. (pp. 1 and 4)
- [23] Alexander Pan, Kush Bhatia, and Jacob Steinhardt. The effects of reward misspecification: Mapping and mitigating misaligned models. In ICLR, 2022. (pp. 1 and 4)
- [24] Nathan Lambert and Roberto Calandra. The alignment ceiling: Objective mismatch in reinforcement learning from human feedback. arXiv preprint, 2023. (p. 1)
- [25] Mike Lewis, Denis Yarats, Yann N Dauphin, Devi Parikh, and Dhruv Batra. Deal or no deal? end-to-end learning for negotiation dialogues. arXiv preprint, 2017. (p. 1)

- [26] Prasann Singhal, Tanya Goyal, Jiacheng Xu, and Greg Durrett. A long way to go: Investigating length correlations in rlhf. arXiv preprint, 2023. (pp. 1 and 11)
- [27] Marilyn Strathern. Improving ratings: audit in the british university system. European Review,

1997. (p. 1)

- [28] Ethan Perez, Sam Ringer, Kamil˙e Lukoši¯ut˙e, Karina Nguyen, Edwin Chen, Scott Heiner, Craig Pettit, Catherine Olsson, Sandipan Kundu, Saurav Kadavath, et al. Discovering language model behaviors with model-written evaluations. arXiv preprint, 2022. (p. 1)
- [29] Mrinank Sharma, Meg Tong, Tomasz Korbak, David Duvenaud, Amanda Askell, Samuel R Bowman, Newton Cheng, Esin Durmus, Zac Hatfield-Dodds, Scott R Johnston, et al. Towards understanding sycophancy in language models. arXiv preprint, 2023. (p. 1)
- [30] Shibani Santurkar, Esin Durmus, Faisal Ladhak, Cinoo Lee, Percy Liang, and Tatsunori Hashimoto. Whose opinions do language models reflect? In ICML, 2023. (p. 1)
- [31] Jochen Hartmann, Jasper Schwenzow, and Maximilian Witte. The political ideology of conversational ai: Converging evidence on chatgpt’s pro-environmental, left-libertarian orientation. arXiv preprint, 2023. (p. 1)
- [32] Jessica Taylor, Eliezer Yudkowsky, Patrick LaVictoire, and Andrew Critch. Alignment for advanced machine learning systems. Ethics of AI, 2016. (pp. 1, 3, and 14)
- [33] Richard Ngo, Lawrence Chan, and Soren Mindermann. The alignment problem from a deep learning perspective. arXiv preprint, 2022. (pp. 1, 3, 14, and 27)
- [34] Dan Hendrycks and Mantas Mazeika. X-risk analysis for AI research. arXiv preprint, 2022.

- (pp. 1 and 14)

[35] Dan Hendrycks. Natural selection favors AIs over humans. arXiv preprint, 2023. (pp. 1 and 14) [36] Simon Zhuang and Dylan Hadfield-Menell. Consequences of misaligned AI. NeurIPS, 2020.

- (pp. 2 and 4)

- [37] Daniel Shin, Anca Dragan, and Daniel S. Brown. Benchmarks and algorithms for offline preference-based reward learning. TMLR, 2023. (p. 2)
- [38] Samuel R Bowman, Jeeyoon Hyun, Ethan Perez, Edwin Chen, Craig Pettit, Scott Heiner, Kamile Lukosuite, Amanda Askell, Andy Jones, Anna Chen, et al. Measuring progress on scalable oversight for large language models. arXiv preprint, 2022. (pp. 2 and 4)
- [39] Alexandre Ramé, Guillaume Couairon, Mustafa Shukor, Corentin Dancette, Jean-Baptiste Gaya, Laure Soulier, and Matthieu Cord. Rewarded soups: towards pareto-optimal alignment by interpolating weights fine-tuned on diverse rewards. In NeurIPS, 2023. (pp. 2, 14, and 26)
- [40] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. NeurIPS, 2022. (pp. 2, 3, and 27)
- [41] Jacob Eisenstein, Chirag Nagpal, Alekh Agarwal, Ahmad Beirami, Alex D’Amour, DJ Dvijotham, Adam Fisch, Katherine Heller, Stephen Pfohl, Deepak Ramachandran, et al. Helping or herding? reward model ensembles mitigate but do not eliminate reward hacking. arXiv preprint, 2023. (pp. 2, 4, 5, 11, 12, 14, and 27)

- [42] Thomas Coste, Usman Anwar, Robert Kirk, and David Krueger. Reward model ensembles help mitigate overoptimization. arXiv preprint, 2023. (pp. 2, 5, and 27)
- [43] Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell. Simple and scalable predictive uncertainty estimation using deep ensembles. In NeurIPS, 2017. (pp. 2 and 5)
- [44] Jonathan Frankle, Gintare Karolina Dziugaite, Daniel M. Roy, and Michael Carbin. Linear mode connectivity and the lottery ticket hypothesis. In ICML, 2020. (pp. 3, 6, and 26)
- [45] Behnam Neyshabur, Hanie Sedghi, and Chiyuan Zhang. What is being transferred in transfer learning? In NeurIPS, 2020. (pp. 3, 6, and 26)
- [46] Mitchell Wortsman, Gabriel Ilharco, Samir Yitzhak Gadre, Rebecca Roelofs, Raphael GontijoLopes, Ari S. Morcos, Hongseok Namkoong, Ali Farhadi, Yair Carmon, Simon Kornblith, and Ludwig Schmidt. Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time. In ICML, 2022. (pp. 3, 6, 7, 9, 26, and 28)
- [47] Alexandre Ramé, Matthieu Kirchmeyer, Thibaud Rahier, Alain Rakotomamonjy, Patrick Gallinari, and Matthieu Cord. Diverse weight averaging for out-of-distribution generalization. In NeurIPS, 2022. (pp. 3, 5, 6, 7, 8, 9, 26, and 28)
- [48] Alexandre Ramé, Kartik Ahuja, Jianyu Zhang, Matthieu Cord, Léon Bottou, and David LopezPaz. Model Ratatouille: Recycling diverse models for out-of-distribution generalization. In ICML, 2023. (pp. 3, 6, 7, 26, and 28)
- [49] Junbum Cha, Sanghyuk Chun, Kyungjae Lee, Han-Cheol Cho, Seunghyun Park, Yunsung Lee, and Sungrae Park. SWAD: Domain generalization by seeking flat minima. In NeurIPS, 2021. (pp. 3 and 26)
- [50] Krikamol Muandet, David Balduzzi, and Bernhard Schölkopf. Domain generalization via invariant feature representation. In ICML, 2013. (pp. 3 and 10)
- [51] Martin Arjovsky, Léon Bottou, Ishaan Gulrajani, and David Lopez-Paz. Invariant risk minimization. arXiv preprint, 2019. (pp. 3, 4, 10, and 14)
- [52] Kerem Zaman, Leshem Choshen, and Shashank Srivastava. Fuse to forget: Bias reduction and selective memorization through model fusion. arXiv preprint, 2023. (pp. 3, 14, and 26)
- [53] Yong Lin, Lu Tan, Yifan Hao, Honam Wong, Hanze Dong, Weizhong Zhang, Yujiu Yang, and Tong Zhang. Spurious feature diversification improves out-of-distribution generalization. In ICLR, 2024. (pp. 3, 9, and 26)
- [54] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NeurIPS, 2017. (pp. 3 and 6)
- [55] Rishi Bommasani, Drew A Hudson, Ehsan Adeli, Russ Altman, Simran Arora, Sydney von Arx, Michael S Bernstein, Jeannette Bohg, Antoine Bosselut, Emma Brunskill, et al. On the opportunities and risks of foundation models. arXiv preprint, 2021. (pp. 3 and 26)
- [56] Maxime Oquab, Leon Bottou, Ivan Laptev, and Josef Sivic. Learning and transferring mid-level image representations using convolutional neural networks. In CVPR, 2014. (p. 3)
- [57] Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, et al. Constitutional AI: Harmlessness from AI feedback. arXiv preprint, 2022. (pp. 4 and 11)

- [58] Harrison Lee, Samrat Phatale, Hassan Mansoor, Thomas Mesnard, Johan Ferret, Kellie Lu, Colton Bishop, Victor Carbune, and Abhinav Rastogi. RLAIF: Scaling reinforcement learning from human feedback with ai feedback. arXiv preprint, 2023. (pp. 4, 7, 11, 12, 13, 27, and 28)
- [59] Ralph Allan Bradley and Milton E Terry. Rank analysis of incomplete block designs: I. the method of paired comparisons. Biometrika, 1952. (p. 4)
- [60] Ronald J Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Reinforcement learning, 1992. (pp. 4, 12, and 28)
- [61] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint, 2017. (pp. 4 and 12)
- [62] Ishaan Gulrajani and David Lopez-Paz. In search of lost domain generalization. In ICLR, 2021. (p. 4)
- [63] Pang Wei Koh, Shiori Sagawa, Henrik Marklund, Sang Michael Xie, Marvin Zhang, Akshay Balsubramani, Weihua Hu, Michihiro Yasunaga, Richard Lanas Phillips, Irena Gao, Tony Lee, Etienne David, Ian Stavness, Wei Guo, Berton Earnshaw, Imran Haque, Sara M Beery, Jure Leskovec, Anshul Kundaje, Emma Pierson, Sergey Levine, Chelsea Finn, and Percy Liang. WILDS: A benchmark of in-the-wild distribution shifts. In ICML, 2021. (p. 4)
- [64] Mohammad Pezeshki, Sékou-Oumar Kaba, Yoshua Bengio, Aaron Courville, Doina Precup, and Guillaume Lajoie. Gradient starvation: A learning proclivity in neural networks. In NeurIPS,

2020. (p. 4)

- [65] Firas Laakom, Jenni Raitoharju, Alexandros Iosifidis, and Moncef Gabbouj. Learning distinct features helps, provably. arXiv preprint, 2021. (p. 4)
- [66] Niv Nayman, Avram Golbert, Asaf Noy, Tan Ping, and Lihi Zelnik-Manor. Diverse ImageNet models transfer better. arXiv preprint, 2022. (p. 4)
- [67] Alexander D’Amour, Katherine Heller, Dan Moldovan, Ben Adlam, Babak Alipanahi, Alex Beutel, Christina Chen, Jonathan Deaton, Jacob Eisenstein, Matthew D Hoffman, et al. Underspecification presents challenges for credibility in modern machine learning. JMLR, 2020. (pp. 4 and 26)
- [68] Damien Teney, Yong Lin, Seong Joon Oh, and Ehsan Abbasnejad. ID and OOD performance are sometimes inversely correlated on real-world datasets. In NeurIPS Workshop, 2023. (p. 4)
- [69] Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. On calibration of modern neural networks. In ICML, 2017. (p. 4)
- [70] Yaniv Ovadia, Emily Fertig, Jie Ren, Zachary Nado, David Sculley, Sebastian Nowozin, Joshua Dillon, Balaji Lakshminarayanan, and Jasper Snoek. Can you trust your model’s uncertainty? evaluating predictive uncertainty under dataset shift. In NeurIPS, 2019. (p. 4)
- [71] Yoav Wald, Amir Feder, Daniel Greenfeld, and Uri Shalit. On calibration and out-of-domain generalization. In NeurIPS, 2021. (p. 4)
- [72] Herbert A Simon. Bounded rationality. Utility and probability, 1990. (p. 4)
- [73] Rohin Shah, Noah Gundotra, Pieter Abbeel, and Anca Dragan. On the feasibility of learning, rather than assuming, human biases for reward inference. In ICML, 2019. (p. 4)

- [74] Timo Kaufmann, Sarah Ball, Jacob Beck, Eyke Hüllermeier, and Frauke Kreuter. On the challenges and practices of reinforcement learning from real human feedback. 2023. (p. 4)
- [75] Deep Ganguli, Liane Lovitt, Jackson Kernion, Amanda Askell, Yuntao Bai, Saurav Kadavath, Ben Mann, Ethan Perez, Nicholas Schiefer, Kamal Ndousse, et al. Red teaming language models to reduce harms: Methods, scaling behaviors, and lessons learned. arXiv preprint, 2022. (p. 4)
- [76] Robert Irvine, Douglas Boubert, Vyas Raina, Adian Liusie, Vineet Mudupalli, Aliaksei Korshuk, Zongyi Liu, Fritz Cremer, Valentin Assassi, Christie-Carol Beauchamp, et al. Rewarding chatbots for real-world engagement with millions of users. arXiv preprint, 2023. (p. 4)
- [77] Condorcet. Essai sur l’application de l’analyse à la probabilité des décisions rendues à la pluralité des voix. 1785. (p. 4)
- [78] Silviu Pitis. Failure modes of learning reward models for llms and other sequence models. In ICML, 2023. (p. 4)
- [79] Melanie Sclar, Yejin Choi, Yulia Tsvetkov, and Alane Suhr. Quantifying language models’ sensitivity to spurious features in prompt design or: How i learned to start worrying about prompt formatting. arXiv preprint, 2023. (p. 4)
- [80] Moran Mizrahi, Guy Kaplan, Dan Malkin, Rotem Dror, Dafna Shahaf, and Gabriel Stanovsky. State of what art? a call for multi-prompt llm evaluation. arXiv preprint, 2023. (p. 4)
- [81] Natasha Jaques, Shixiang Gu, Dzmitry Bahdanau, José Miguel Hernández-Lobato, Richard E Turner, and Douglas Eck. Sequence tutor: Conservative fine-tuning of sequence generation models with kl-control. In ICML, 2017. (pp. 5 and 12)
- [82] Matthieu Geist, Bruno Scherrer, and Olivier Pietquin. A theory of regularized markov decision processes. In ICML, 2019. (pp. 5 and 12)
- [83] Angeliki Lazaridou, Anna Potapenko, and Olivier Tieleman. Multi-agent communication meets natural language: Synergies between functional and structural language learning. In ACL,

2020. (p. 5)

- [84] Yuchen Lu, Soumye Singhal, Florian Strub, Aaron Courville, and Olivier Pietquin. Countering language drift with seeded iterated learning. In ICML, 2020. (p. 5)
- [85] Siddharth Reddy, Anca Dragan, Sergey Levine, Shane Legg, and Jan Leike. Learning human objectives by evaluating hypothetical behavior. In ICML, 2020. (pp. 5 and 27)
- [86] William Saunders, Girish Sastry, Andreas Stuhlmüller, and Owain Evans. Trial without error: Towards safe reinforcement learning via human intervention. In AAMAS, 2018. (p. 5)
- [87] Binghai Wang et al. Secrets of rlhf in large language models part ii: Reward modeling. arXiv preprint, 2023. (pp. 5, 10, and 27)
- [88] Ron Kohavi, David H Wolpert, et al. Bias plus variance decomposition for zero-one loss functions. In ICML, 1996. (p. 5)
- [89] Naonori Ueda and Ryohei Nakano. Generalization error of ensemble estimators. In ICNN,

1996. (p. 5)

- [90] Sandipan Kundu, Yuntao Bai, Saurav Kadavath, Amanda Askell, Andrew Callahan, Anna Chen, Anna Goldie, Avital Balwit, Azalia Mirhoseini, Brayden McLean, et al. Specific versus general principles for constitutional ai. arXiv preprint, 2023. (p. 5)

- [91] Ananya Kumar, Aditi Raghunathan, Robbie Matthew Jones, Tengyu Ma, and Percy Liang. Fine-tuning can distort pretrained features and underperform out-of-distribution. In ICLR,

2022. (pp. 5, 6, 7, and 28)

- [92] Samuel K. Ainsworth, Jonathan Hayase, and Siddhartha Srinivasa. Git re-basin: Merging models modulo permutation symmetries. In ICLR, 2022. (p. 6)
- [93] Almog Gueta, Elad Venezian, Colin Raffel, Noam Slonim, Yoav Katz, and Leshem Choshen. Knowledge is a region in weight space for fine-tuned language models. In EMNLP, 2023. (p. 6)
- [94] Raphael Gontijo-Lopes, Yann Dauphin, and Ekin Dogus Cubuk. No one representation to rule them all: Overlapping features of training methods. In ICLR, 2022. (pp. 6 and 8)
- [95] Pavel Izmailov, Dmitrii Podoprikhin, Timur Garipov, Dmitry Vetrov, and Andrew Gordon Wilson. Averaging weights leads to wider optima and better generalization. In UAI, 2018. (pp. 7 and 26)
- [96] Devansh Arpit, Huan Wang, Yingbo Zhou, and Caiming Xiong. Ensemble of averages: Improving model selection and boosting performance in domain generalization. In NeurIPS, 2021. (pp. 7 and 26)
- [97] Michael Völske, Martin Potthast, Shahbaz Syed, and Benno Stein. Tl; dr: Mining reddit to learn automatic summarization. In ACL Workshop, 2017. (pp. 7, 8, and 11)
- [98] Rohan Anil, Andrew M Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, et al. PaLM 2 technical report. arXiv preprint, 2023. (pp. 7, 11, 12, 27, 28, and 30)
- [99] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Ed Chi, Quoc Le, and Denny Zhou. Chain-of-Thought prompting elicits reasoning in large language models. In NeurIPS,

2022. (pp. 7, 11, and 27)

- [100] Alexandre Ramé, Corentin Dancette, and Matthieu Cord. Fishr: Invariant gradient variances for out-of-distribution generalization. In ICML, 2022. (pp. 10 and 14)
- [101] Christian Szegedy, Wojciech Zaremba, Ilya Sutskever, Joan Bruna, Dumitru Erhan, Ian Goodfellow, and Rob Fergus. Intriguing properties of neural networks. arXiv preprint, 2013. (p. 10)
- [102] Yao-Yuan Yang, Cyrus Rashtchian, Hongyang Zhang, Ruslan Salakhutdinov, and Kamalika Chaudhuri. Adversarial robustness through local lipschitzness. arXiv preprint, 2020. (p. 10)
- [103] Mihaela Rosca, Theophane Weber, Arthur Gretton, and Shakir Mohamed. A case for new neural network smoothness constraints. In NeurIPS ICBINB, 2020. (p. 10)
- [104] Matthias Hein and Maksym Andriushchenko. Formal guarantees on the robustness of a classifier against adversarial manipulation. NeurIPS, 2017. (p. 10)
- [105] Jure Sokolić, Raja Giryes, Guillermo Sapiro, and Miguel RD Rodrigues. Robust large margin deep neural networks. IEEE Transactions on Signal Processing, 2017. (p. 10)
- [106] Jeremy Cohen, Elan Rosenfeld, and Zico Kolter. Certified adversarial robustness via randomized smoothing. In ICML, 2019. (p. 10)
- [107] Roland Hafner and Martin Riedmiller. Reinforcement learning in feedback control: Challenges and benchmarks from technical process control. Machine learning, 2011. (p. 10)

- [108] Matteo Pirotta, Marcello Restelli, and Luca Bascetta. Policy gradient in lipschitz markov decision processes. Machine Learning, 2015. (p. 10)
- [109] Lionel Blondé, Pablo Strasser, and Alexandros Kalousis. Lipschitzness is all you need to tame off-policy generative adversarial imitation learning. Machine Learning, 2022. (p. 10)
- [110] Yann Dubois, Xuechen Li, Rohan Taori, Tianyi Zhang, Ishaan Gulrajani, Jimmy Ba, Carlos Guestrin, Percy Liang, and Tatsunori B Hashimoto. Alpacafarm: A simulation framework for methods that learn from human feedback. arXiv preprint, 2023. (p. 11)
- [111] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. JMLR, 2020. (pp. 11, 12, 30, and 33)
- [112] Jacob Hilton. KL divergence of max-of-n, 2023. (p. 12)
- [113] Ahmad Beirami, Alekh Agarwal, Jonathan Berant, Alexander D’Amour, Jacob Eisenstein, Chirag Nagpal, and Ananda Theertha Suresh. Theoretical guarantees on the best-of-n alignment policy. arXiv preprint, 2024. (p. 12)
- [114] Colin Raffel. Building Machine Learning Models Like Open Source Software. ACM, 2023. (p. 14)
- [115] Margaret Li, Suchin Gururangan, Tim Dettmers, Mike Lewis, Tim Althoff, Noah A Smith, and Luke Zettlemoyer. Branch-Train-Merge: Embarrassingly parallel training of expert language models. arXiv preprint, 2022. (p. 14)
- [116] Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. Communication-efficient learning of deep networks from decentralized data. In AISTATS, 2017. (p. 14)
- [117] Zeqiu Wu, Yushi Hu, Weijia Shi, Nouha Dziri, Alane Suhr, Prithviraj Ammanabrolu, Noah A. Smith, Mari Ostendorf, and Hannaneh Hajishirzi. Fine-grained human feedback gives better rewards for language model training. In NeuriPS, 2023. (p. 14)
- [118] Zafir Stojanovski, Karsten Roth, and Zeynep Akata. Momentum-based weight interpolation of strong zero-shot models for continual learning. In NeurIPS Workshop, 2022. (pp. 14 and 26)
- [119] Steven Vander Eeckt et al. Weight averaging: A simple yet effective method to overcome catastrophic forgetting in automatic speech recognition. arXiv preprint, 2022. (p. 14)
- [120] Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D Manning, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. arXiv preprint, 2023. (pp. 14 and 27)
- [121] Maxime Labonne. NeuralBeagle14-7B. https://huggingface.co/mlabonne/NeuralBe agle14-7B-GGUF, 2024. (pp. 14 and 27)
- [122] Polina Kirichenko, Pavel Izmailov, and Andrew Gordon Wilson. Last layer re-training is sufficient for robustness to spurious correlations. In ICLR, 2023. (p. 14)
- [123] Dan Hendrycks and Thomas Dietterich. Benchmarking neural network robustness to common corruptions and perturbations. In ICLR, 2019. (p. 26)

- [124] John R. Zech, Marcus A. Badgeley, Manway Liu, Anthony B. Costa, Joseph J. Titano, and Eric Karl Oermann. Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: A cross-sectional study. PLOS Medicine, 2018. (p. 26)
- [125] Alex J DeGrave, Joseph D Janizek, and Su-In Lee. AI for radiographic COVID-19 detection selects shortcuts over signal. Nature Machine Intelligence, 2021. (p. 26)
- [126] Mitchell Wortsman, Gabriel Ilharco, Jong Wook Kim, Mike Li, Hanna Hajishirzi, Ali Farhadi, Hongseok Namkoong, and Ludwig Schmidt. Robust fine-tuning of zero-shot models. In CVPR,

2022. (p. 26)

- [127] Gabriel Ilharco, Mitchell Wortsman, Samir Yitzhak Gadre, Shuran Song, Hannaneh Hajishirzi, Simon Kornblith, Ali Farhadi, and Ludwig Schmidt. Patching open-vocabulary models by interpolating weights. In NeurIPS, 2022. (p. 26)
- [128] Shachar Don-Yehiya, Elad Venezian, Colin Raffel, Noam Slonim, Yoav Katz, and Leshem Choshen. ColD fusion: Collaborative descent for distributed multitask finetuning. In ACL,

2023. (p. 26)

- [129] Nikolaos Dimitriadis, Pascal Frossard, and François Fleuret. Pareto manifold learning: Tackling multiple tasks via ensembles of single-task models. arXiv preprint, 2022. (Not cited.)
- [130] Mustafa Shukor, Corentin Dancette, Alexandre Ramé, and Matthieu Cord. Unival: Unified model for image, video, audio and language. TMLR, 2023. (p. 26)
- [131] Francesco Croce, Sylvestre-Alvise Rebuffi, Evan Shelhamer, and Sven Gowal. Seasoning model soups for robustness to adversarial and natural distribution shifts. In CVPR, 2023. (p. 26)
- [132] Jeevesh Juneja, Rachit Bansal, Kyunghyun Cho, João Sedoc, and Naomi Saphra. Linear connectivity reveals generalization strategies. In ICLR, 2023. (p. 26)
- [133] Evgenii Nikishin, Pavel Izmailov, Ben Athiwaratkun, Dmitrii Podoprikhin, Timur Garipov, Pavel Shvechikov, Dmitry Vetrov, and Andrew Gordon Wilson. Improving stability in deep reinforcement learning with weight averaging. 2018. (p. 26)
- [134] Jean-Baptiste Gaya, Laure Soulier, and Ludovic Denoyer. Learning a subspace of policies for online adaptation in reinforcement learning. In ICLR, 2022. (p. 26)
- [135] Daniel Lawson and Ahmed H Qureshi. Merging decision transformers: Weight averaging for forming multi-task policies. In ICLR RRL Workshop, 2023. (p. 26)
- [136] Michael Noukhovitch, Samuel Lavoie, Florian Strub, and Aaron Courville. Language model alignment with elastic reset. In NeurIPS, 2023. (p. 26)
- [137] Gabriel Ilharco, Marco Tulio Ribeiro, Mitchell Wortsman, Suchin Gururangan, Ludwig Schmidt, Hannaneh Hajishirzi, and Ali Farhadi. Editing models with task arithmetic. In ICLR, 2023. (p. 26)
- [138] Nico Daheim, Nouha Dziri, Mrinmaya Sachan, Iryna Gurevych, and Edoardo M Ponti. Elastic weight removal for faithful and abstractive dialogue generation. arXiv preprint, 2023. (p. 26)
- [139] Hwanjun Song, Minseok Kim, Dongmin Park, Yooju Shin, and Jae-Gil Lee. Learning from noisy labels with deep neural networks: A survey. TNNLS, 2022. (p. 26)
- [140] Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, and Oriol Vinyals. Understanding deep learning requires rethinking generalization. ICLR, 2017. (p. 26)

- [141] Ryutaro Tanno, Ardavan Saeedi, Swami Sankaranarayanan, Daniel C Alexander, and Nathan Silberman. Learning from noisy labels by regularized estimation of annotator confusion. In CVPR, 2019. (p. 26)
- [142] Neel Jain, Ping-yeh Chiang, Yuxin Wen, John Kirchenbauer, Hong-Min Chu, Gowthami Somepalli, Brian R Bartoldson, Bhavya Kailkhura, Avi Schwarzschild, Aniruddha Saha, et al. Neftune: Noisy embeddings improve instruction finetuning. arXiv preprint, 2023. (p. 26)
- [143] Aritra Ghosh, Himanshu Kumar, and P Shanti Sastry. Robust loss functions under label noise for deep neural networks. In AAAI, 2017. (p. 26)
- [144] Xiaobo Xia, Tongliang Liu, Bo Han, Mingming Gong, Jun Yu, Gang Niu, and Masashi Sugiyama. Sample selection with uncertainty of losses for learning with noisy labels. In ICLR, 2022. (p. 26)
- [145] Lu Jiang, Zhengyuan Zhou, Thomas Leung, Li-Jia Li, and Li Fei-Fei. Mentornet: Learning data-driven curriculum for very deep neural networks on corrupted labels. In ICML, 2018. (p. 26)
- [146] Bo Han, Quanming Yao, Xingrui Yu, Gang Niu, Miao Xu, Weihua Hu, Ivor Tsang, and Masashi Sugiyama. Co-teaching: Robust training of deep neural networks with extremely noisy labels. NeurIPS, 2018. (p. 26)
- [147] Maryam Sabzevari. Ensemble learning in the presence of noise. PhD thesis, Universidad Autónoma de Madrid, 2019. (p. 26)
- [148] Andrew Y Ng, Stuart Russell, et al. Algorithms for inverse reinforcement learning. In ICML,

2000. (p. 27)

- [149] W Bradley Knox, Stephane Hatgis-Kessell, Sigurdur Orn Adalgeirsson, Serena Booth, Anca Dragan, Peter Stone, and Scott Niekum. Learning optimal advantage from preferences and mistaking it for reward. arXiv preprint, 2023. (p. 27)
- [150] Peter Barnett, Rachel Freedman, Justin Svegliato, and Stuart Russell. Active reward learning from multiple teachers. arXiv preprint, 2023. (p. 27)
- [151] Sian Gooding and Hassan Mansoor. The impact of preference agreement in reinforcement learning from human feedback: A case study in summarization. arXiv preprint, 2023. (p. 27)
- [152] Lei Li, Yekun Chai, Shuohuan Wang, Yu Sun, Hao Tian, Ningyu Zhang, and Hua Wu. Toolaugmented reward modeling. In ICLR, 2023. (p. 27)
- [153] Zhiqing Sun, Sheng Shen, Shengcao Cao, Haotian Liu, Chunyuan Li, Yikang Shen, Chuang Gan, Liang-Yan Gui, Yu-Xiong Wang, Yiming Yang, et al. Aligning large multimodal models with factually augmented rlhf. arXiv preprint, 2023. (p. 27)
- [154] Anonymous. RIME: Robust preference-based reinforcement learning with noisy human preferences. In Submitted to ICLR, 2023. (p. 27)
- [155] Mohammad Gheshlaghi Azar, Mark Rowland, Bilal Piot, Daniel Guo, Daniele Calandriello, Michal Valko, and Rémi Munos. A general theoretical paradigm to understand learning from human preferences. arXiv preprint, 2023. (p. 27)
- [156] Noam Shazeer and Mitchell Stern. Adafactor: Adaptive learning rates with sublinear memory cost. In ICML, 2018. (pp. 27 and 28)

###### [157] Noam Razin, Hattie Zhou, Omid Saremi, Vimal Thilak, Arwen Bradley, Preetum Nakkiran, Joshua Susskind, and Etai Littwin. Vanishing gradients in reinforcement finetuning of language models. arXiv preprint, 2023. (p. 28)

### WARM: On the Benefits of Weight Averaged Reward Models

##### Supplementary material

This supplementary material is organized as follows:

- • Appendix A enriches our related work.
- • Appendix B clarifies some experimental details.
- • Appendix C enriches our experiments.

#### A. Related work

This paper leverages the insights from the OOD generalization literature, in particular from linear mode connectivity (see Appendix A.1), and applies them to the design of efficient, reliable and robust reward models (see Appendix A.2).

###### A.1. Out-of-distribution generalization, linear mode connectivity and memorization

LMC in fine-tuning. Fine-tuning foundation models [55] into specialized models that generalize well to new distributions is critical for many real-world applications [123, 124, 125]. Recently, different variants of weight averaging (WA) were able to improve performance, such as moving average [49, 95, 96], WiSE fine-tuning [126], model soups [46], DiWA [47] and model ratatouille

- [48]. These works rely on the LMC [44, 45] across fine-tuned weights, which was extended to finetunings on different tasks [48, 127, 128], modalities [130] or with different losses [47, 131], although [132] highlighted some limitations. WA was also used recently in RL setups [39, 133, 134, 135, 136], in particular in RLHF in [39, 136] but only to combine policies and not rewards. Insights into WA. Specifically, WA comes with several benefits. First, WA flattens the loss landscape
- [49]. Second, WA approximates prediction ensembling, thus reduces variance of the estimator [46, 47] and tackles model misspecification [67]. Third, WA combines models’ abilities [137, 138], which can be useful for multi-task [127], multi-objective [39] or in continual learning [118] setups. Lastly, it has recently been shown that WA can provide some benefits under spurious correlations [52, 53], with a phenomenon called FalseFalseTrue in [53]. These works [52, 53] share similarities with our memorization experiments from Section 4.2, but we are the first to analyze WA regularization properties under label corruption, and their consequences on generalization. In contrast, in [52] the networks are trained on different datasets while the theory in [53] is actually mostly developed for prediction ensembling.

Memorization. Traditional approaches [139] tackling memorization of corrupted labels [140] usually require explicit regularization [141], specific data augmentation [142], loss adjustment [143] or sample selection [144]. Some other strategies are based on ensembling: they filter out potentially corrupted samples with self-labeling filtering [145, 146] or bagging diversity procedures [147]. As far as we know, with WA we propose the first strategy combining multiple models trained on the same dataset that manages to tackle corruption.

###### A.2. Reward modeling

One of the central challenge in aligning LLMs is the absence of explicit rewards from the environment, a.k.a. the outer alignment challenge [33]. While Inverse Reinforcement Learning [148] attempts to derive a reward model (RM) from expert demonstrations, most recent efforts [12, 13, 14, 15, 40] primarily focus on learning from human preferences. Despite its importance to enhance LLM performances post-RL and for safe deployment in real-world applications, how to best design RMs has arguably receive less attention than it warrants. Some research [149] seeks to refine the loss function from Equation (1). Other approaches are more data oriented: for example, LLaMA-2 [16] involves continual learning of the RM to adjust to new generation distributions; [85, 150] follow an active learning paradigm [151]. Augmenting rewards with tools [152] or additional information [153] represents an even more recent and very promising trend. Limited efforts have been made at the intersection of label corruption and reward modeling; [154] tried to filter the preference dataset for small academic locomotion tasks, while the concurrent [87] suggests applying label smoothing and flipping. Actually, reward ensembling is the most discussed method to mitigate reward hacking [41, 42]; we show that WARM can beat ENS while removing its overheads. Finally, following DPO [120], a recent trend merges reward modeling with policy learning; though, the policies still tend to hack the preference data [155], and thus require only a few training steps and very small learning rates. The WA of DPO policies, theoretically equivalent to the WA of RMs, is a promising research direction with already significant empirical results on public benchmarks, as demonstrated in [121].

#### B. Implementation details

###### B.1. Dataset details

For summarization, we use the Reddit TL;DR dataset [14], containing posts from Reddit that have been filtered to ensure high quality. The training summaries from [14] are generated by OpenAI GPT-3 [6] variants. The dataset contains 123k posts, and ∼5% is held out as the ID validation set. To generate the candidate responses in the OOD dataset D𝑜𝑜𝑑 with 92k pairwise comparisons, we considered multiple PaLM-XS policies with high temperature, some of which are pre-trained only, others SFT-ed and others RLHF-ed; the goal was to get a diverse set of summaries.

###### B.2. AI labeling details

While the ideal approach for evaluating our models would involve human preferences, we resort to the cheaper AI labeling procedure from RLAIF [58]. We query an instruct fine-tuned PaLM-L [98] LLM1, prompted to generate preference mimicking human preferences. Specifically, we follow the “Detailed + CoT 0-shot” prompting strategy from RLAIF [58], the best one according to their results, involving zero-shot prompting with chain-of-thought [99], a maximum decoding length of 512 tokens and temperature 𝑇 = 0.0 (i.e., greedy decoding). To avoid position bias, we run the AI labeler in the two possible orderings. This strategy was shown to perform similarly to human labellers, with similar inter-agreement. For the corruption experiments, we swap the labels for 25% of the training samples.

###### B.3. Reward modeling details

The RMs are PaLM-XXS models [98]. They are first pre-trained, and then supervised fine-tuned on the Reddit TL;DR dataset for 12k steps with a batch size of 128 and the Adafactor [156] optimizer

1Available through Google Cloud’s Vertex AI https://cloud.google.com/vertex-ai/docs/generative-ai/ learn/models.

with a learning rate of 10−5. Following the Baklava recipe, we actually launch the reward modeling from different checkpoints along this SFT fine-tuning, at steps {8k, 10k, 12k}; taking a too-early checkpoint would drastically reduce RM accuracy, as observed in [157]. To convert this LLM into a classifier, we plug a linear probed [91] classification layer (the same for all RMs); said differently, even though the featurizers are actually from different SFT checkpoints, they share the same linear probed classification linear layer. As explained in [91], it prevents features from moving too much away from their initializations, which facilitates the LMC required for WA.

We train all RMs for 10k steps, a batch size of 128, the Adafactor [156] optimizer, a learning rate sampled in {1e-5,4e-5,1e-4}, and a dropout probability in {0.05, 0.1}. This follows the practical recommandations from [47] to leverage hyperparameters in a mild range to preserve the LMC. Training for a longer number of steps could help, as it did not alter the LMC in previous works [48].

In practice, for the main experiments with clean labels, we launch 10 reward modelings; when ranked in decreasing accuracy on D𝑜𝑜𝑑, we denote them {𝜙𝑖}10𝑖=1. Therefore, the RMs named 𝜙1 and 𝜙2 in the different plots are the two best according to their individual performances under distribution shifts. Then, WARM 𝑀 = 2 is actually the RM defined per 𝜙1+2𝜙2, while ENS 𝑀 = 2 averages their predictions. More generally, WARM with 𝑀 weights is the WA of the 𝑀 best weights {𝜙𝑖}𝑖𝑀=1. The main motivation of this weight selection procedure is to remove potentially bad RMs, as validated in Figure 10, in which we consider different permutations across those 10 RMs. As a side note, we speculate that a greedy procedure as in [46] could further improve performances.

0.768

0.766

0.764

0.762

OODAcc.

0.760

0.758

0.756

From best to worst

First best and then random

0.754

Random permutation v1 Random permutation v2 From worst to best

0.752

2 4 6 8 10

M

- Figure 10 | Analysis of the weight selection procedure. We plot the accuracy resulting from averaging 𝑀 weights (out of 10), where these weights are chosen based on various selection procedures. This effectively validates that choosing models from best to worst serves as a reliable heuristic.

###### B.4. Reinforcement learning details

Both policy and value models are PaLM-XS [98], initialized from the same SFT model. We then generate samples from the policy with temperature 𝑇 = 0.9, batch size of 128, the Adafactor [156] optimizer, a learning rate of 10−5 and a policy warmup of 2k steps. We set 𝛼 = 0.003 for the KL regularization in the main experiment without label corruption, and 𝛼 = 0.01 with label corruption. Following [58], we used a modified version of REINFORCE [60] with a baseline value function for variance reduction.

#### C. Additional experiments

###### C.1. 2nd order analysis: weight averaging for more robust ensembling

| |WA<br><br>ENS<br><br>Diag| | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

0.40

0.50

0.50

0.46

0.48

0.38

0.49

0.46

0.45

Acc.

0.44

0.36

0.48

0.42

0.44

0.47

0.34

0.40

0.46

0.43

0.38

0.0 0.5 1.0

0.0 0.5 1.0

0.0 0.5 1.0

0.0 0.5 1.0

0.575

0.395

0.335

0.46

0.550

0.390

0.330

0.45

0.525

0.385

Acc.

0.500

0.380

0.325

0.44

0.475

0.375

0.320

0.43

0.370

0.450

0.0 0.5 1.0

0.0 0.5 1.0

0.0 0.5 1.0

0.0 0.5 1.0

- Figure 11 | Train (corrupt). More results enriching Figure 4(a) with different pairs of RMs.

0.0 0.5 1.0

0.886

0.888

0.890

0.892

0.894

Acc.

WA

ENS

Diag

0.0 0.5 1.0

0.890

0.892

0.894

0.896

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

0.0 0.5 1.0

0.835

0.840

0.845

0.850

0.855

0.0 0.5 1.0

0.8800

0.8825

0.8850

0.8875

0.8900

0.8925

0.8950

0.0 0.5 1.0

0.890

0.895

0.900

0.905

Acc.

0.0 0.5 1.0

0.8425

0.8450

0.8475

0.8500

0.8525

0.8550

0.8575

0.0 0.5 1.0

0.878

0.880

0.882

0.884

0.886

0.888

0.0 0.5 1.0

0.888

0.890

0.892

0.894

0.896

- Figure 12 | Train (clean). More results enriching Figure 4(b) with different pairs of RMs.

0.0 0.5 1.0

0.7725

0.7750

0.7775

0.7800

0.7825

0.7850

Acc.

WA

ENS

Diag

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

0.0 0.5 1.0

0.760

0.765

0.770

0.775

0.0 0.5 1.0

0.74

0.75

0.76

0.77

0.0 0.5 1.0

0.74

0.75

0.76

0.77

0.78

0.0 0.5 1.0

0.74

0.75

0.76

0.77

0.78

Acc.

0.0 0.5 1.0

0.770

0.772

0.774

0.776

0.778

0.780

0.0 0.5 1.0

0.780

0.785

0.790

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

0.0 0.5 1.0

0.7700

0.7725

0.7750

0.7775

0.7800

0.7825

0.7850

- Figure 13 | Validation (ID). More results enriching Figure 4(c) with different pairs of RMs.

0.0 0.5 1.0

0.710

0.712

0.714

0.716

Acc.

WA

ENS

Diag

0.0 0.5 1.0

0.704

0.706

0.708

0.710

0.712

0.0 0.5 1.0

0.700

0.705

0.710

0.715

0.720

0.725

0.0 0.5 1.0

0.695

0.700

0.705

0.710

0.715

0.720

0.725

0.0 0.5 1.0

0.690

0.695

0.700

0.705

0.710

0.715

Acc.

0.0 0.5 1.0

0.718

0.720

0.722

0.724

0.726

0.0 0.5 1.0

0.718

0.720

0.722

0.724

0.726

0.0 0.5 1.0

0.708

0.710

0.712

0.714

0.716

0.718

- Figure 14 | Test (OOD). More results enriching Figure 4(d) with different pairs of RMs.

###### C.2. BoN experiments

WARM M = 6 WARM M = 2 ENS M = 2

5.75

5.50

Ind 2 Ind 1

5.25

Controlreward

5.00

4.75

4.50

4.25

4.00

0.2 0.4 0.6 0.8 1.0 1.2

KL : log(N) NN 1

(a) PaLM (clean).

WARM M = 6 WARM M = 2 ENS M = 2

5.0

Ind 2 Ind 1

4.5

Controlreward

4.0

3.5

3.0

0.2 0.4 0.6 0.8 1.0 1.2

KL : log(N) NN 1

(b) PaLM (corrupt).

WARM M = 6 WARM M = 2 ENS M = 2

- 3

- 4

- 5

- 6

- 7

Ind 2 Ind 1

Controlreward

0 1 2 3 4 5 6

KL : log(N) NN 1

(c) T5 (clean).

WARM M = 6 WARM M = 2 ENS M = 2

6.5

6.0

Ind 2 Ind 1

5.5

Controlreward

5.0

4.5

4.0

3.5

3.0

0 1 2 3 4 5 6

KL : log(N) NN 1

(d) T5 (corrupt).

- Figure 15 | Same as Figure 8, but with absolute values of the control reward for BoN experiments. We consider two SFT policies to generate candidate summaries: one based on PaLM architecture [98], the other on T5 architecture [111]. In both cases, we observe that WARM performs better than ENS and the individual networks in terms of pointwise control RM.

0.2 0.4 0.6 0.8 1.0 1.2

KL : log(N) NN 1

0.00

0.02

0.04

0.06

Controlrewardgain

WARM Baklava M = 2

ENS Baklava M = 2

Ind 3 Baklava

Ind 1

(a) Baklava with PaLM.

0 1 2 3 4 5 6

KL : log(N) NN 1

0.00

0.05

0.10

0.15

Controlrewardgain

WARM Baklava M = 2

ENS Baklava M = 2

Ind 3 Baklava

Ind 1

(b) Baklava with T5.

- Figure 16 | Control reward for BoN experiments (clean setup) with Baklava when the two fine-tunings 𝜙1 and 𝜙3 have different featurizer initializations, collected respectively at steps 12k and 8k from a shared SFT.

| | | | | | | |
|---|---|---|---|---|---|---|
|WARM M = 2<br><br>ENS M = 2<br><br>Ind 2 Ind 1 y=0.5<br><br>| | | | | | |
| | | | | | | |

0.50

0.48

Winratiovs.WARM=6M

0.46

0.44

0.42

0.40

0.38

0.2 0.4 0.6 0.8 1.0 1.2

KL : log(N) NN 1

(a) PaLM.

| | | | | | | |
|---|---|---|---|---|---|---|
|WARM M = 6 WARM M = 2 ENS M = 2<br><br>Ind 2 Ind 1 y=0.5<br><br>| | | | | | |
| | | | | | | |

0.5

Winratiovs.WARM=6with=1000MN

0.4

0.3

0.2

0.1

0 1 2 3 4 5 6

KL : log(N) NN 1

(b) T5 vs. WARM w/ 𝑁 = 1000.

- Figure 17 | Oracle preference metric for BoN experiments (clean setup). Figure 17(a) confirms Figure 7(c) but on generations from PaLM SFT. Figure 17(b) shows win rates for BoN on T5 generations for WARM with 𝑀 = 6 and always 𝑁 = 1000 for BoN vs. other RMs with 1 ≤ 𝑁 ≤ 1000. We validate that BoN limits reward hacking compared to RL, as performances get better when increasing 𝑁.

- C.3. RL experiments

- C.3.1. Experiments with corrupted preference dataset

0 2000 4000 6000 8000 10000

# steps

- 4

- 5

- 6

- 7

Controlreward

WARM M = 6 WARM M = 2 ENS M = 2

Ind 2 Ind 1

- Figure 18 | RL experiments. Same as Figure 1(b) but with 25% corruption in the preference dataset.

###### C.3.2. Experiments with clean preference dataset

| | | | | | |
|---|---|---|---|---|---|
|WARM M = 10<br><br>WARM M = 2<br><br>ENS M = 2<br><br>Ind 2 Ind 1 y=0.5<br><br>| | | | | |
| | | | | | |

0.8

Winratiovs.WARM=6M

0.6

0.4

0.2

0.0

2000 3000 4000 5000 6000

# steps

(a) WARM 𝑀 = 6.

| | | | | | |
|---|---|---|---|---|---|
|= 0.01<br><br>= 0.003 = 0.001<br><br>y=0.5| | | | | |
| | | | | | |

0.8

WinratioofWARM=6vs.IndM1

0.7

0.6

0.5

0.4

0.3

0.2

2000 3000 4000 5000 6000

# steps

(b) Impact of 𝛼.

- Figure 19 | Oracle preference metric for RL experiments at fixed number of training steps (clean setup). Figure 19(a) plots the win rate of the policy with WARM 𝑀 = 6 vs. the other policies, all at the same number of

training steps. Figure 19(b) shows the win rate of WARM 𝑀 = 6 against the policy trained with a single RM 𝜙1 (the best according to OOD accuracy) along training for different values of 𝛼 controlling the KL regularization strength.

0 2500 5000 7500 10000 12500 15000 17500

# steps

- 2

- 3

- 4

- 5

- 6

- 7

- 8

- 9

- 10

Controlreward

WARM M = 6 WARM M = 2 ENS M = 2

Ind 2 Ind 1

(a) Control reward vs. training steps.

0 100 200 300 400 500 600 700

KL

- 2

- 3

- 4

- 5

- 6

- 7

- 8

- 9

- 10

Controlreward

WARM M = 6 WARM M = 2 ENS M = 2

Ind 2 Ind 1

(b) Control reward vs. KL.

- Figure 20 | Control reward for RL experiments with 𝛼 = 0.01 (clean setup).

10.0

7.5

5.0

Controlreward

2.5

0.0

WARM M = 6 WARM M = 2 ENS M = 2

2.5

5.0

Ind 2 Ind 1

7.5

0 2000 4000 6000 8000

# steps

(a) Control reward vs. training steps.

10.0

7.5

Controlreward

5.0

2.5

0.0

WARM M = 6 WARM M = 2 ENS M = 2

2.5

5.0

Ind 2 Ind 1

7.5

0 500 1000 1500 2000

KL

(b) Control reward vs. KL.

###### Figure 21 | Control reward for RL experiments with 𝛼 = 0.001 (clean setup).

- C.4. Distillation experiments

In Figure 22 we reproduce the distillation setup from [17], where the control PaLM-XS RM generates the labels to train PaLM-XXS RMs. As a side note, we observed that distillation changes the diversity across fine-tuned RMs, thus potentially altering the significance of the distillation setup, motivating us in exploring the more realistic RLAIF setup.

WARM M = 6 WARM M = 2 ENS M = 2

0.125

0.100

Ind 2 Ind 1

0.075

Controlrewardgain

0.050

0.025

0.000

0.025

0.050

0.075

0 1 2 3 4 5 6

KL : log(N) NN 1

###### Figure 22 | BoN experiment in the distillation setup from [17]. The labels in the preference dataset are given by the control RM, the same RM which gives the 𝑦-axis. The candidate summaries are generated by a SFT with the T5 architecture [111]. The blue lines represent WARM with 𝑀 weights: WARM performs higher than the individual RMs (in yellows) or when ensembling their predictions (ENS in red).

