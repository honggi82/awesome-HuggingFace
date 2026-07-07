## Think Before Recommend: Unleashing the Latent Reasoning Power for Sequential Recommendation

Jiakai Tang1∗§, Sunhao Dai1∗, Teng Shi1, Jun Xu1, Xu Chen1†, Wen Chen2†, Jian Wu2, Yuning Jiang2

1Gaoling School of Artificial Intelligence, Renmin University of China, Beijing, China 2Alibaba Group, Beijing, China {tangjiakai5704,sunhaodai,shiteng,junxu,xu.chen}@ruc.edu.cn {chenyu.cw,joshuawu.wujian,mengzhu.jyn}@alibaba-inc.com

### Abstract

|Item Embedding<br><br>Seq Representation<br><br>Reasoning Hidden State|
|---|

Sequential Recommendation (SeqRec) aims to predict the next item by capturing sequential patterns from users’ historical interactions, playing a crucial role in many real-world recommender systems. However, existing approaches predominantly adopt a direct forward computation paradigm, where the final hidden state of the sequence encoder serves as the user representation. We argue that this inference paradigm, due to its limited computational depth, struggles to model the complex evolving nature of user preferences and lacks a nuanced understanding of long-tail items, leading to suboptimal performance. To address this issue, we propose ReaRec, the first inference-time computing framework for recommender systems, which enhances user representations through implicit multi-step reasoning. Specifically, ReaRec autoregressively feeds the sequence’s last hidden state into the sequential recommender while incorporating special reasoning position embeddings to decouple the original item encoding space from the multi-step reasoning space. Moreover, we introduce two lightweight reasoning-based learning methods, Ensemble Reasoning Learning (ERL) and Progressive Reasoning Learning (PRL), to further effectively exploit ReaRec’s reasoning potential. Extensive experiments on five public real-world datasets and different SeqRec architectures demonstrate the generality and effectiveness of our proposed ReaRec. Remarkably, post-hoc analyses reveal that ReaRec significantly elevates the performance ceiling of multiple sequential recommendation backbones by approximately 30%-50%. Thus, we believe this work can open a new and promising avenue for future research in inferencetime computing for sequential recommendation.

# arXiv:2503.22675v3[cs.IR]1Aug2025

- (a) Direct Recommendation

|SeqRec Model|
|---|

|Reasoning SeqRec Model|
|---|

[Figure 1]

[Figure 2]

- (b) Multi-step Reasoning-enhanced Recommendation

#### Figure 1: Illustration of traditional direct inference (i.e., reasoning-free) and our proposed multi-step reasoningenhanced sequential recommendation framework.

##### ACM Reference Format:

Jiakai Tang, Sunhao Dai, Teng Shi, Jun Xu, Xu Chen, Wen Chen, Jian Wu, Yuning Jiang. 2025. Think Before Recommend: Unleashing the Latent Reasoning Power for Sequential Recommendation. In . ACM, New York, NY, USA, 15 pages. https://doi.org/10.1145/nnnnnnn.nnnnnnn

### 1 Introduction

Recommender systems (RS) have become ubiquitous in modern daily life, powering personalized services across domains such as e-commerce platforms [47, 81], music recommendation services [9, 80], and video streaming applications [34, 78]. To accurately capture a user’s next interaction intent, sequential recommendation algorithms are designed to analyze historical interactions to mine underlying sequential patterns and model latent user preferences [2, 13, 59]. Current mainstream sequential recommendation models, such as SASRec [31] and UniSRec [26], adopt a Transformer-based architecture, leveraging their power attention mechanisms to adaptively weight past interacted items and use the final position’s encoded output as the user representation, as illustrated in Fig. 1(a). However, we argue this prevailing direct forward inference paradigm may lack nuanced comprehension of dynamic user preferences and evolving interest patterns, leading to suboptimal modeling for long-tail user interest and unpopular items. Despite their efficiency, we argue that these direct inference paradigms often fall short in modeling long-tail users with fewer interactions and less popular items—scenarios that inherently demand more nuanced reasoning and deeper representation learning.

### CCS Concepts

• Information systems → Recommender systems.

### Keywords

Sequential Recommendation, Inference-time Reasoning

∗ Equal Contribution. § Work done during internship at Alibaba Group. † Corresponding author.

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.

Conference’17, Washington, DC, USA © 2025 Copyright held by the owner/author(s). Publication rights licensed to ACM. ACM ISBN 978-1-4503-XXXX-X/2018/06 https://doi.org/10.1145/nnnnnnn.nnnnnnn

Recently, many studies from the natural language processing (NLP) community have demonstrated that Chain-of-Thought (CoT) during inference can significantly improve the performance of Large Language Models (LLMs) on complex tasks like mathematics and coding [19, 45, 53, 62]. By allowing the model to perform multi-step deliberation before generating a final output, CoT-based reasoning enhances the model’s capacity to handle complex problems beyond what direct inference allows. Furthermore, Feng et al. [14] theoretically uncover that the emergent thinking capabilities are attributed to the increased computational depth introduced by CoT-based reasoning, which allows models to overcome the expressivity limitations of direct answer even with constrained parameter sizes.

Motivated by these insights, we explore whether a similar thinkbefore-action paradigm can benefit sequential recommendation, especially for challenging cases such as long-tail users and items. We propose ReaRec, a novel reasoning-enhanced framework that enables SeqRec models to engage in implicit multi-step reasoning during inference. As shown in Fig. 1(b), ReaRec performs autoregressive reasoning over latent representations before producing the final user embedding, thereby deepening feature crossing and improving representational richness. To prevent the recommender from confusing the sequence encoding stage and reasoning stage, we design a specialized positional encoding scheme to explicitly distinguish item representations from reasoning inputs. However, unlike NLP tasks, where explicit reasoning chains naturally provide process supervision to guide model optimization [36, 39, 44], implicit reasoning in sequential recommendation lacks effective intermediate signals. This absence of stepwise guidance could lead to unpredicted reasoning degradation issues, causing the recommender to either replicate prior reasoning patterns or progressively drift away from accurately modeling the user’s true interest distribution. Consequently, this may significantly impair the robustness and generalization capability of the recommendation model.

To address the aforementioned challenges, we propose two simple yet effective reasoning learning strategies, Ensemble Reasoning Learning (ERL) and Progressive Reasoning Learning (PRL), to fully exploit the reasoning power of our ReaRec framework. For the ERL method, it leverages the idea of ensemble learning to construct multi-order user representations to comprehensively capture latent interest distributions from diverse perspectives. Specifically, we introduce multi-step supervised optimization to alleviate the optimization difficulty in deep reasoning processes. Furthermore, to prevent reasoning-pattern degradation, we incorporate a representation diversity regularizer to mitigate output homogeneity in multi-step reasoning. For the PRL method, inspired by curriculum learning, we design a progressive temperature annealing mechanism to guide the model from initial exploitation to the gradual refinement of modeled sequential patterns. This approach enables the model to progressively learn the user’s true interest distributions. Moreover, we also propose a reasoning-aware contrastive learning objective to enhance the reasoning robustness ability by simulating the error self-correction process, thus achieving better generalization performance.

Our extensive experiments on five benchmark datasets demonstrate the effectiveness of the proposed ReaRec framework. In particular, the ReaRec achieves an average performance gain of 7.49%

0.08

70.00

Base

Ours

Ours (Upper Bound)

| |
|---|

| |
|---|

+52.88%

0.07

54.00

+48.04%

+43.64%

+37.21%

Improv.(%)

NDCG@20

0.06

38.00

0.05

22.00

+9.07%

+6.87%

+5.87%

+3.10%

0.04

6.00

0.03

-10.00

SASRec BERT4Rec UniSRec MoRec

0.17

70.00

Base Ours Ours (Upper Bound)

+51.35%

0.16

54.00

+49.47%

Improv.(%)

Recall@20

0.14

38.00

+28.86%

+28.37%

0.13

22.00

+10.69%

+9.74%

+4.85%

0.11

6.00

+2.46%

0.10

-10.00

SASRec BERT4Rec UniSRec MoRec

Figure 2: Empirical performance gains and potential upper bound analysis of optimal reasoning steps (K = 2) on Yelp dataset across different SeqRec models.

across all metrics while incurring only 3.51% additional inference latency (cf. Sec. 4.2 and Sec. 4.3.3). Moreover, further analysis reveals several interesting empirical findings: (1) Enhancing modeling capability for underrepresented groups. The multi-step reasoning process steadily enhances the recommendation quality of users with sparse interactions and long-tail items. (2) Remarkable performance ceiling breakthrough. Post-hoc optimal reasoning step analysis shows that our method elevates the performance ceilings for different backbone models by approximately 30%-50% (as shown in Fig. 2), highlighting its promising capability. We are optimistic that our proposed RecRec will open new avenues for exploring inference-time scaling for recommender systems.

Our main contributions are summarized as follows:

- • We propose ReaRec, a novel reasoning-enhanced sequential recommendation framework that empowers SeqRec models to perform implicit multi-step reasoning during inference. To our knowledge,thisisthefirstworktosystematically explore inferencetime computational power within recommender systems.
- • We introduce two reasoning learning strategies, ERL and PRL, which leverage the ideas of ensemble learning and curriculum learning to efficiently optimize the implicit reasoning process and alleviate reasoning degradation issues.
- • Extensive experiments on five real-world datasets and various representative SeqRec models validate the generality and effectiveness of ReaRec. Notably, our detailed post-hoc analysis reveals that ReaRec can significantly raise the performance ceiling, achieving significant improvements by up to 50%.
- • We identify somechallengesfacedby current reasoning-enhanced recommendation methods and the future opportunities, stimulating a new research direction at the intersection of inference-time computing and sequential recommendation.

### 2 Preliminary

In this section, we formally define the sequential recommendation task and introduce the typical sequential recommendation pipeline.

### 2.1 Problem Definition

Formally, let U and V denote the sets of users and items, respectively, with 𝑀 = |U| and 𝑁 = |V| representing the number of users and items. For each user𝑢 ∈ U, we define their chronological

interaction sequence as S𝑢 = [𝑣𝑢1,𝑣𝑢2, . . .,𝑣𝑛𝑢𝑢], where𝑛𝑢 represents the length of the interaction sequence S𝑢. Each item 𝑣 ∈ V has a unique ID and a set of textual attributes (such as title, product feature, and other side information). These attributes are stored in a dictionary D𝑣 = {𝑘1 : 𝑎1,𝑘2 : 𝑎2, . . .,𝑘𝑚 : 𝑎𝑚}, where 𝑘𝑖 and 𝑎𝑖 represent the key and value of the 𝑖-th attribute, respectively. Here, 𝑚 refers to the total number of attributes associated with item 𝑣. The overall text description for item 𝑣 is constructed by concatenating its attributes in the format of an unordered list: “The item information is as follows: \n- 𝑘1:𝑎1 \n- 𝑘2:𝑎2 \n ...\n- 𝑘𝑚:𝑎𝑚”.

The goal of sequential recommendation is to predict the next item a user will interact with, based on historical interaction data. Given the interaction sequences for all users S = {S𝑢1, S𝑢2, . . ., S𝑢𝑀 },

where S𝑢𝑖 represents the interaction sequence of user𝑢𝑖, and S1:𝑢𝑖𝑡 = [𝑣𝑢1,𝑣𝑢2, . . .,𝑣𝑡𝑢] denotes the first 𝑡 interaction records of user 𝑢𝑖. Given the item embedding matrix E ∈ R𝑁×𝑑, where 𝑑 is the dimen-

sion of the item embedding, the sub-sequence S1:𝑢𝑖𝑡 is encoded to obtain the corresponding item embeddings E𝑢1:𝑖𝑡 = [e𝑣𝑢

𝑡 ]. The recommender’s learning objective is to maximize the prediction probability of the next item 𝑣𝑡𝑢+𝑖1 based on the historical interaction data, which is formally defined as

, e𝑣𝑢

, . . ., e𝑣𝑢

1

2

∑︁

𝑛∑︁𝑢−1

max

𝑃(𝑣𝑡𝑢+1|S1:𝑢𝑡;Θ), (1)

Θ

𝑡=1

𝑢∈U

where Θ denotes the parameters of the recommendation model.

### 2.2 Sequential Recommendation Pipeline

In a typical sequential recommendation pipeline, users’ historical interactions are first encoded into item embeddings. These item embeddings are then fed into a sequential model (e.g., transformerbased models) to produce a sequence representation, typically using the output from the final position (as illustrated in Fig. 1(a)). Finally, this sequence representation is used to calculate similarity scores with candidate item embeddings (such as dot product [77, 78] or cosine similarity[35, 66]) to predict the probability of the user’s interaction with the next item.

In general, mainstream sequential recommendation methods can be broadly categorized into two main types, distinguished primarily by their approaches to encoding item representations:

- (1) ID-based Encoding: The ID-based approach uses one-hot en-

coding for the item’s discrete representation and retrieves the item’s embedding from the embedding matrix. Representative sequential recommendation methods employing this encoding approach include SASRec [31], BERT4Rec[49], etc.

- (2) Text-based Encoding: The text-based item representation

usually involves feeding the item’s string-formatted description into a pre-trained language model (such as BERT [41], LLaMA [17],

etc.), and then utilizing average pooling or extracting hidden state from special positions (e.g., [CLS] or the last position) as the item’s encoding [16, 35, 37]. Popular recommendation models utilizing text-based encoding include UniSRec [26], MoRec [76], etc.

In this paper, since the proposed reasoning framework is modelagnostic, we omit the details of how item representations are obtained and consistently use e𝑣 ∈ E to denote item 𝑣’ representations.

### 3 Methodology

In this section, we introduce ReaRec, a novel, simple, and highly scalable recommendation framework designed to unleash a model’s latent sequential reasoning capability. Instead of the traditional direct recommendation without reasoning, our approach leverages multi-step implicit reasoning to refine user representations, fully exploiting the computational potential of sequential models to approximate the true distribution of user interests.

In what follows, we first introduce ReaRec, our foundational framework for inference-time computation extension (Sec. 3.1). We then propose two lightweight methods—Ensemble Reasoning Learning (Sec. 3.2) and Progressive Reasoning Learning (Sec. 3.3)to address the aforementioned challenges. Moreover, we give a deeper analysis of the proposed ReaRec framework (Sec. 3.4). The overall framework of ReaRec is illustrated in Fig. 3.

### 3.1 ReaRec Backbone

Our proposed ReaRec is model-agnostic and can be easily integrated into a variety of sequential recommenders. To better explain our work, we illustrate our framework using the widely adopted transformer [57] architecture in sequential recommendation tasks as an example, demonstrating how we extend computational capacity during inference with our backbone.

3.1.1 Self-attention Sequence Encoding. Given a user’s historical sequence S𝑢 = [𝑣𝑢1,𝑣𝑢2, . . .,𝑣𝑛𝑢], we can obtain the item embeddings of these 𝑛 items by looking up the embedding matrix E. To fully leverage sequential information, we inject Absolute Position Embeddings into the item embeddings at the input layer. Specifically, for a given item 𝑣 at position 𝑖, the input representation is constructed by summing its item embedding e𝑣 and the corresponding positional embedding p𝑖𝐼:

h𝑖0 = e𝑣 + p𝑖𝐼, (2)

where p𝑖𝐼 is obtained by looking up the learnable positional embedding matrix P𝐼 ∈ R𝑛×𝑑. Next, we develop the item sequence encoder 𝑓 (·) by stacking multiple multi-head self-attention layers (denoted as 𝑀𝐻𝑆𝐴(·)) and point-wise feed-forward networks (denoted as 𝐹𝐹𝑁 (·)) to capture the complicated sequence features:

H𝑙 = 𝑓 (H𝑙−1) = 𝐹𝐹𝑁 (𝑀𝐻𝑆𝐴(H𝑙−1)), (3)

where H𝑙 = [h𝑙1, h𝑙2, . . ., h𝑙𝑛] denotes the concatenated hidden states at the 𝑙-th layer. In the conventional paradigm, the output at the

last position of the final layer is directly used as the final user representation, i.e., h𝑢 = H𝐿[−1], where 𝐿 is the number of layers.

Then, we calculate the predicted probability for the user 𝑢 as

KL Regularization

|AveragePooling| |
|---|---|
| |L|

AveragePooling

𝑦ˆ = softmax(h𝑢 · E⊤) (5) and use cross-entropy loss as the recommendation objective:

|Reasoning SeqRec Model|
|---|

LKL

Rec

LRec = −log𝑦ˆ𝑣+, (6)

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

LERL

where 𝑦ˆ𝑣+ denotes the prediction probability of the ground-truth item 𝑣+ for user 𝑢’s next interaction.

###### ReaRec Framework

Ensemble Reasoning Learning (ERL)

Real

Predicted Distribution Distribution

However, this naive optimization objective still faces a critical issue: the lack of supervision signals for intermediate reasoning states makes the model vulnerable to the risk of reasoning pattern degradation. Next, we introduce two simple yet effective reasoning learning strategies to address these challenges.

[Figure 9]

|Reasoning SeqRec Model|
|---|

Temperature

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

### 3.2 Ensemble Reasoning Learning (ERL)

Reasoning-aware Contrastive Learning (RCL)

LRCL

LPRL LRec

###### Progressive Reasoning Learning (PRL)

To provide effective supervised signals for the implicit reasoning process, we propose an Ensemble Reasoning Learning (ERL) method. This approach uses the hidden states of different reasoning steps as multi-view representations of the user’s evolving interests. In other words, we apply the idea of ensemble learning [11, 43] to aggregate diverse reasoning results from different reasoning steps, thereby avoiding suboptimal performance caused by the final output alone.

Item Embedding Position Embedding Noise Reasoning Hidden State Reasoning Position Embedding

#### Figure 3: Overview of the proposed ReaRec framework and two reasoning-enhanced learning strategies: Ensemble Reasoning Learning and Progressive Reasoning Learning.

- 3.2.1 Multi-Step Reasoning Supervision. Specifically, we treat the reasoning hidden states from multiple steps as multi-vector user representations and apply cross-entropy loss (cf. Eq. (6)) to the ensembled sequence representation to enhance process guidance. Therefore, instead of using only the reasoning state from the last step, ERL employs an average pooling layer to aggregate reasoning hidden states across all steps to obtain the final user representation as h𝑢 = 𝐾1 𝑖 𝐾=0 r𝑖. The output distribution 𝑦ˆ is then computed following Eq. (5).

- 3.2.2 KL Divergence Regularization. However, simply using the above recommendation objective for model training is obviously inefficient. The recommender may take shortcuts by directly copying the previous reasoning output to optimize the parameters, which can lead to a pattern collpase effect, consequently undermining the advantage of computational scaling during inference processes. To this end, inspired by the works [27, 30], we introduce a Kullback-Leibler (KL) divergence constraint, a popular and simple regularization technique to mitigate the homogenization output issue. To be specific, we aim to increase the reasoning output diversity across different steps, encouraging the model’s multi-step reasoning process to gather multi-view insights, and better model the user’s complex interest distribution, ultimately contributing to the overall sequence recommendation performance. Formally, we pair the predictive probability distributions of different reasoning states in pairwise combinations and maximize the KL divergence between these distribution pairs, which is equivalent to minimizing the following regularization term:

- 3.1.2 Extended Inference-Time Reasoning. Existing sequential recommenders that rely only on non-reasoning forward inference struggle to directly model item sequence patterns, fundamentally constrained by their limited computation power to capture nuanced user interest. To address this problem, we propose implicit reasoning mechanism to augment the computational capacity, enabling the enhanced refinement of user interest modeling to more precisely approximate real preference distributions.

Specifically, rather than directly using H𝐿[−1] as the user representation, we autoregressively feed the hidden state of the last position back into the encoder for 𝑲-pass forward computations. By effectively increasing inference-time computation, this approach further unleashes the model’s potential to capture intricate sequential dependencies. However, this inference strategy deviates from the original objective of sequential recommendation models, namely next-item prediction. To bridge this task gap, we introduce the Reasoning Position Embedding (RPE), denoted as P𝑅 ∈ R𝐾×𝑑, to distinguish between the sequence encoding phase and the reasoning phase. At the 𝑘-th reasoning step, the model’s input embedding is defined as H0 ∈ R(𝑛+𝑘−1)×𝑑. The first 𝑛 positions remain unchanged from the original input (i.e., Eq. (2)), while the latent representation h𝑛0+𝑖 at position 𝑛 + 𝑖 is calculated as the summation of the last output h𝑛𝐿+𝑖−1 from the previous step and the 𝑖-th reasoning position embedding p𝑖𝑅:

h𝑛0+𝑖 = h𝑛𝐿+𝑖−1 + p𝑖𝑅. (4)

𝐾∑︁−1

∑︁𝐾

To differentiate between item encoding outputs and reasoning outputs, we denote the hidden states of the model’s final layer from position 𝑛 to 𝑛 + 𝑘 as R = [r0, r1, . . ., r𝑘], where r𝑖 ∈ R𝑑 represents the reasoning hidden state at the 𝑖-th step. To obtain the user representation, a straightforward approach is to follow the traditional paradigm i.e., use the last reasoning output r𝐾 as h𝑢.

KL(𝑦ˆ(𝑖) ∥𝑦ˆ(𝑗)). (7)

LKL = −

𝑖=0

𝑗=𝑖+1

where 𝑦ˆ(𝑖) represents the logit output of the 𝑖-th reasoning step.

By combining the recommendation loss and the above KL regularization term, the overall learning objective for the ERL method

is to minimize the following loss function:

LERL = LRec + 𝜆LKL, (8) where 𝜆 is a hyperparameter that balances the constraint strength.

- 3.2.3 Inference Phase. In the inference phase, we apply an average pooling layer to aggregate the reasoning hidden states from all steps into the final user representation, i.e., h𝑢 = 𝐾1 𝑖 𝐾=0 r𝑖. Then, we compute the inner product or cosine similarity (depending on the specific sequential recommendation algorithm) between user representation and all candidate item representations, with top-scoring items selected as the final recommendation list.

- 3.3 Progressive Reasoning Learning (PRL)

Unlike the ensemble reasoning learning method, we explore another Progressive Reasoning Learning (PRL) mechanism. The core idea is to design a progressive distribution sharpening strategy to guide the intermediate reasoning chains, gradually approximating the user’s true preference distribution. Intuitively, as the computational power allocated to the inference time increases, the recommendation model should be able to more accurately capture the fine-grained sequential features, narrowing the discrepancy between the predicted and actual user interest distribution.

- 3.3.1 Progressive Temperature Annealing (PTA). Drawing an analogy the human cognitive process, as the thinking depth increases, reasoning pathways become progressively refined until converging toward optimal solutions. Similarly, we expect that as the model’s computations increases, the recommender would gradually clarify the user’s interest evolving patterns, which is manifested as sharper predicted distributions. Inspired by this motivation, we propose a simple Progressive Temperature Annealing (PTA) method to guide the reasoning process. To achieve this, we first introduce a

temperature coefficient, 𝜏𝑘, for the 𝑘-th reasoning step to adjust the predicted distribution sharpness, which is formulated as follows:

𝜏𝑘 = 𝜏 ∗ 𝛼𝐾−𝑘, 𝑦ˆ(𝑘) = softmax(r𝑘 · E⊤/𝜏𝑘),

(9)

where 𝜏 is the base temperature, and 𝛼 is a hyperparameter that controls the temperature decay rate.

In contrast to ensemble reasoning learning method, we apply separate recommendation losses to each reasoning hidden state to inject process supervision into the reasoning process, as follows:

∑︁𝐾

log𝑦ˆ𝑣(𝑘+), (10)

LRec = −

𝑘=0

where 𝑦ˆ𝑣(𝑘+) represents the logit corresponding to the 𝑣+ item. With this lean annealing strategy, the model is encouraged to explore a

broader solution space in the early reasoning stage, preventing it from getting stuck in local optima. Then, as the reasoning process progresses, the value of𝜏𝑘 is gradually reduced to narrow the search space, guiding the model towards the global optimum. Thus, the proposed PTA can more effectively approximate the user’s true preference distribution.

- 3.3.2 Reasoning-aware Contrastive Learning (RCL). However, relying solely on the temperature annealing strategy may not be sufficient to support the generalization ability of progressive reasoning learning. This is because, during the reasoning process, the model may suffer from the reasoning bias, where the model’s reasoning direction deviates from the correct user interest distribution, ultimately leading to the accumulation of reasoning errors and deteriorating the reasoning capability. To address the above challenge, we design a novel Reasoning-aware Contrastive Learning (RCL) method to enhance the model’s robust reasoning ability.

Specifically, we simulate the preceding accumulated reasoning error by injecting noise vectors into the reasoning states for each step, producing the noised reasoning input as follows:

h˜𝑛0+𝑖 = h𝑛0+𝑖 + 𝝐, 𝑖 ∈ {1, 2, . . .,𝐾}, (11)

where h𝑛0+𝑖 is defined according to Eq. (2). The vector 𝝐 represents the added noise embedding, sampled from a normal distribution,

i.e., 𝝐 ∼ N(0,𝛾I), where I ∈ R𝑑 is the identity matrix of dimension 𝑑 and 𝛾 controls the noise intensity. Then, we can obtain the new hidden state view R˜ = [r˜1, r˜2, . . ., r˜𝐾] by feeding the noised input into the transformer encoder.

To enhance the model’s robustness in reasoning denoising, we design a self-supervised task based on Mutual Information Maximization (MIM) [56, 58]. Formally, given variables 𝑋 and 𝑌, the Mutual Information (MI) measures the reduction in uncertainty of X after observing Y, which is defined as:

𝐼(𝑋,𝑌) = 𝐻(𝑋) − 𝐻(𝑋|𝑌),

where 𝐻(·) and 𝐻(·|·) denote the entropy and conditional entropy of the random variable, respectively. By maximizing the MI between the original hidden states R and the denoised hidden states R˜, it can effectively force the model to capture the essential sequential information from the user behavior data and historical reasoning process, achieving self-reflection in the implicit thought space.

However, directly maximizing mutual information is not feasible due to the intractability of the high-dimensional probability distribution estimation. Inspired by recent works [55, 63], we propose an InfoNCE-based reasoning contrastive learning method to optimize the lower bound of mutual information, which is defined as:

LRCL = −

∑︁𝐾

𝑘=1

log

exp(sim(r˜𝑘, r𝑘+)/𝜏) exp(sim(r˜𝑘, r𝑘+)/𝜏) + r−

𝑘 ∈R−

𝑘

exp(sim(r˜𝑘, r𝑘−)/𝜏)

,

(12)

where sim(·) denotes the dot product similarity function, r𝑘+ and r𝑘− indicate the positive and negative contrastive hidden states at the

𝑘-th step, respectively. For the negative sample set R𝑘−, analogous to existing methods [51, 74], we utilize the 𝑘-th step reasoning states

corresponding to the other item sequences within the same batch.

By combining the recommendation loss and the reasoning contrastive loss, we can derive the overall objective function for the PRL method as follows:

LPRL = LRec + LRCL. (13)

- 3.3.3 Inference Phase. During inference, we directly adopt the

final reasoning step’s output as the user representation, i.e., h𝑢 = r𝐾. Then, similar to Sec. 3.2.3, we compute similarity scores between

#### Table 1: The statistics of experimental datasets.

Video & Games

CDs & Vinyl

Baby & Products

Dataset Yelp

Software

#Users 13,083 89,021 30,049 35,238 140,292 #Items 10,697 22,933 16,705 87,969 30,689

#Avg. Inter. / User 33.92 5.96 5.59 14.59 5.57 #Avg. Inter. / Item 41.49 23.15 10.06 5.84 25.44

#Avg. Inter. 443,807 530,989 168,029 513,991 780,809 Sparisty 99.68% 99.97% 99.97% 99.98% 99.98%

h𝑢 and the candidate item embedding matrix E to generate the recommendation list for the user 𝑢.

### 3.4 Discussion

- 3.4.1 Principle Analysis. The ReaRec framework fundamentally extends the model’s modeling capability by strategically increasing inference-time computational amounts. By autoregressively feeding the reasoning hidden states into the sequence encoder, the model continuously deepens feature crossing depth, capturing finer-grained sequence characteristics and eventually improving the recommendation performance. Moreover, the proposed ERL and PRL methods unleash the latent reasoning power of sequential recommenders in different ways. The ERL integrates multi-level deep crossing features into the final user representation, while the latter, based on the concept of curriculum learning, gradually uncovers more complex intent evolution patterns as the reasoning process progresses, moving closer to real user interest distribution.
- 3.4.2 Time and Space Complexity. In this part, we provide a detailed analysis of the time and space complexity of the proposed ReaRec framework as follows:

- • Time Complexity. Suppose the user sequence length is 𝐶, we first analyze the base backbone time without reasoning extension. The input sequence passes through 𝐿 layers of 𝑀𝐻𝑆𝐴 modules (𝑂(𝐶2𝑑 +𝐶𝑑2)) and 𝐹𝐹𝑁 modules (𝑂(𝐶𝑑2)), resulting in a total time complexity of𝑂(𝐿(𝐶2𝑑+𝐶𝑑2)). For the reasoning-enhanced phase, we employ KV Caching technique to store history keyvalue pairs, eliminating redundant computations. Specifically, at the 𝑘-th reasoning step, the time complexity of the 𝑀𝐻𝑆𝐴 and 𝐹𝐹𝑁 are 𝑂((𝐶 + 𝑘 − 1)𝑑) and 𝑂(𝑑2), respectively. After applying 𝐿 transformer blocks and 𝐾 steps of reasoning, the total additional time complexity overhead is 𝑂(𝐿(𝐾(𝐶 + 𝐾)𝑑 + 𝐾𝑑2)). Since the number of reasoning steps (e.g., 𝐾 = 2) is usually much smaller than 𝐶, this overhead is simplified to 𝑂(𝐿(𝐾𝐶𝑑 + 𝐾𝑑2)). Therefore, our framework does not bring significant time cost, making it suitable for practical deployment in real-world industry recommender systems.
- • Space Complexity. Our method only adds 𝐾 𝑑-dimensional reasoning position embeddings P𝑅, which is almost negligible compared to the original model parameters. Therefore, our framework is highly lightweight and flexible.

### 4 Experiments

In this section, we conduct extensive experiments and analyses to demonstrate the superiority of our proposed ReaRec framework.

### 4.1 Experimental Setup

- 4.1.1 Datasets. To evaluate the effectiveness of our proposed methods, we conduct extensive experiments on five real-world recommendation datasets from Yelp and Amazon platforms. The detailed statistics of the datasets are summarized in Table 1.

- (1) Yelp1: This dataset originates from a well-known business

review website, providing rich multidimensional data support for studying user behaviors and business attributes. We treat interactions with ratings greater than 3 as positive samples and apply 20-core filtering to preprocess the data. For textual encoding, we retain the name, location (city and state), and business categories as item information. The dataset is chronologically split into training, validation, and test sets based on two timestamp thresholds: September 4, 2018 and May 12, 2020.

- (2) Amazon 20232: This dataset is derived from Amazon, a

leading global e-commerce platform. We select datasets from four domains: Video & Games, Software, CDs & Vinyl, and Baby & Products. For textual features, we retain the product attributes like title, description, and price. Similarly, we treat user-item interactions with user ratings greater than 3 as positive samples. To ensure data quality, we filter out users with fewer than 5 interactions for Video & Games, Software, Baby & Products, and fewer than 10 interactions for CDs & Vinyl. For dataset splitting, we follow the official absolute timestamps to partition item sequences3. This aligns well with real-world scenarios and facilitates fair performance comparisons within the recommendation research community.

- 4.1.2 EvaluationMetrics. We adopt top-kNormalizedDiscounted Cumulative Gain (NDCG) and top-k Recall to measure the recommendation performance, which are widely used in related sequential recommendation research [6, 50, 70]. In this paper, we specifically report NDCG@{10,20}, which assesses both the relevance and ranking quality of the top-k recommended items, and Recall@{10,20}, which evaluates the ability of the model to recall the ground-truth items in the top-k list.
- 4.1.3 Baselines. To thoroughly evaluate the generality of our proposed reasoning-enhanced framework, we conduct comprehensive benchmarking across different types of sequential recommendation models, including both ID-based and text-based encoding methods. The baselines are as follows: For ID-based encoding methods, we compare our methods with the following state-of-the-art models:

- • SASRec [31], a representative baseline for sequential recommendation, employs causal multi-head attention mechanism to capture sequential patterns in user interaction data.
- • BERT4Rec [49], a widely-used sequential model, leverages bidirectional self-attention layers for deeper contextual information infusion across user behavior sequences.

For Text-based encoding methods, we adopt the following algorithms as backbones:

• UniSRec [26] utilizes parameter whitening and a Mixture-ofExperts (MoE) adaptor to learn universal item and sequence representations from textual features, which effectively addresses cold-start and data sparsity challenges.

- 1https://business.yelp.com/data/resources/open-dataset/
- 2https://amazon-reviews-2023.github.io/
- 3https://amazon-reviews-2023.github.io/data_processing/5core.html

#### Table 2: Performance comparison of different ID-based models on five datasets. ‘N’ and ‘R’ indicate NDCG and Recall metrics, respectively. ‘Avg.’ represents the average improvement rate across all metrics (i.e., NDCG@{10,20} and Recall@{10,20}). Performance improvements are indicated by “↑”, while performance declines are indicated by “↓”.

SASRec BERT4Rec N@10 N@20 R@10 R@20 Avg. N@10 N@20 R@10 R@20 Avg.

Dataset Method

Base 0.0347 0.0452 0.0626 0.1047 - 0.0364 0.046 0.0653 0.1038 -

+ERL (Improv.)

0.0383 (↑10.37%)

0.0474 (↑4.87%)

0.0691 (↑10.38%)

0.1056 (↑0.86%) ↑6.62%

0.0371 (↑1.92%)

0.0476 (↑3.48%)

0.0661 (↑1.23%)

0.1077 (↑3.76%) ↑2.60%

Yelp

+PRL (Improv.)

0.0388 (↑11.82%)

0.0493 (↑9.07%)

0.073 (↑16.61%)

0.1149 (↑9.74%) ↑11.81%

0.0377 (↑3.57%)

0.0487 (↑5.87%)

0.0708 (↑8.42%)

0.1149 (↑10.69%) ↑7.14%

Base 0.0284 0.0353 0.0542 0.0816 - 0.0289 0.0355 0.0548 0.0810 -

+ERL (Improv.)

0.0301 (↑5.99%)

0.0385 (↑9.07%)

0.0581 (↑7.20%)

0.0915 (↑12.13%) ↑8.59%

0.0311 (↑7.61%)

0.0375 (↑5.63%)

0.0578 (↑5.47%)

0.0832 (↑2.72%) ↑5.36%

Video & Games

+PRL (Improv.)

0.0299 (↑5.28%)

0.0379 (↑7.37%)

0.0572 (↑5.54%)

0.0890 (↑9.07%) ↑6.81%

0.0306 (↑5.88%)

0.0380 (↑7.04%)

0.0584 (↑6.57%)

0.0879 (↑8.52%) ↑7.00%

Base 0.0696 0.0895 0.1468 0.2264 - 0.0710 0.0893 0.1530 0.2258 -

+ERL (Improv.)

0.0743 (↑6.75%)

0.0935 (↑4.47%)

0.1456 (↓0.82%)

0.2224 (↓1.77%) ↑2.16%

0.0769 (↑8.31%)

0.0964 (↑7.95%)

0.1554 (↑1.57%)

0.2328 (↑3.10%) ↑5.23%

Software

+PRL (Improv.)

0.0739 (↑6.18%)

0.0949 (↑6.03%)

0.1488 (↑1.36%)

0.2324 (↑2.65%) ↑4.06%

0.0762 (↑7.32%)

0.0976 (↑9.29%)

0.1500 (↓1.96%)

0.2350 (↑4.07%) ↑4.68%

Base 0.0148 0.0174 0.0317 0.0419 - 0.0149 0.0185 0.0326 0.0468 -

+ERL (Improv.)

0.0182 (↑22.97%)

0.0212 (↑21.84%)

0.0363 (↑14.51%)

0.0482 (↑15.04%) ↑18.59%

0.0165 (↑10.74%)

0.0208 (↑12.43%)

0.0354 (↑8.59%)

0.0524 (↑11.97%) ↑10.93%

CDs & Vinyl

+PRL (Improv.)

0.0155 (↑4.73%)

0.0195 (↑12.07%)

0.0315 (↓0.63%)

0.0470 (↑12.17%) ↑7.08%

0.0162 (↑8.72%)

0.0202 (↑9.19%)

0.0334 (↑2.45%)

0.0496 (↑5.98%) ↑6.59%

Base 0.0112 0.0157 0.0260 0.0437 - 0.0109 0.0154 0.0257 0.0439 -

+ERL (Improv.)

0.0116 (↑3.57%)

0.0164 (↑4.46%)

0.0228 (↓12.31%)

0.0418 (↓4.35%) ↓2.16%

0.0148 (↑35.78%)

0.0195 (↑26.62%)

0.0293 (↑9.57%)

0.0481 (↑14.01%) ↑21.49%

Baby & Products

+PRL (Improv.)

0.0135 (↑20.54%)

0.0178 (↑13.38%)

0.0281 (↑8.08%)

0.0451 (↑3.20%) ↑11.30%

0.0140 (↑28.44%)

0.0185 (↑20.13%)

0.0291 (↑6.15%)

0.0466 (↑13.23%) ↑16.99%

• MoRec [76] replaces traditional ID features by incorporating advanced text and visual encoders (e.g., RoBERTa [38] and ViT [12]) to model the multimodal representations of items.

- 4.1.4 Implementation Details. We conduct all experiments on 8 NVIDIA A100 GPUs. To ensure a fair comparison, we set the embedding size and batch size for all methods to 256 and 2048, respectively. We optimize all models using the Adam [32] optimizer with a learning rate of 0.001 and follow previous work [49] by adopting GeLU as the activation function. Following the existing works [7, 64], we truncate user sequences to a maximum length of 50 across all datasets. Since our framework is modelagnostic, it can be seamlessly integrated into various sequential recommendation models. In particular, for BERT4Rec’s bidirectional Transformer, we employ a Prefix Masking strategy, where the item sequence part utilizes bidirectional attention, while the reasoning adopts unidirectional attention. Early stopping is triggered if the metrics on the validation set do not improve over 10 consecutive epochs. For item-based methods, we follow previous work [37] by using LLaMA-3.1-8B [17] to encode item textual features. In particular, we apply Principle Component Analysis (PCA) to the averaged hidden states from the last layer, preserving core features and distilling 768-dimensional model representations. For ERL method, we search for the KL regularization hyperparameter

𝜆 within {0.001, 0.005, 0.01, 0.05, 0.1}. For PRL method, we set the noise strength 𝛾 = 0.01 and tune the base temperature 𝜏 and temperature decay rate 𝛼 over the ranges {0.05, 0.1, 0.5, 1.0, 2.0, 5.0} and {1.0, 1.2, 1.5, 2.0, 5.0, 10.0}, respectively. Our codes will be available at https://github.com/TangJiakai/ReaRec.

### 4.2 Overall Performance

The recommendation performance of ID-based and text-based sequential models across all datasets is summarized in Table 2 and Table 3, respectively. We derive the following observations:

- • For ID-based recommenders (i.e., SASRec and BERT4Rec), we can find that BERT4Rec slightly outperforms SASRec at different metrics on most datasets. This suggests that incorporating both left and right contextual information enhances the model’s ability to capture sequential patterns more effectively.
- • Text-based methods (i.e., UniSRec and MoRec) consistently outperform ID-based models across all datasets. For instance, on the Yelp dataset,UniSRecachievesa9.51% improvement in NDCG@20 and a 14.14% increase in Recall@20 compared to SASRec. This improvement can be attributed to the ability of text-based models to leverage powerful language models for encoding item information, effectively mitigating data sparsity issues. In other words, by learning domain-invariant representations from textual feature

#### Table 3: Performance comparison of different Text-based models on five datasets. ‘N’ and ‘R’ indicate NDCG and Recall metrics, respectively. ‘Avg.’ represents the average improvement rate across all metrics (i.e., NDCG@{10,20} and Recall@{10,20}). Performance improvements are indicated by “↑”, while performance declines are indicated by “↓”.

UniSRec MoRec N@10 N@20 R@10 R@20 Avg. N@10 N@20 R@10 R@20 Avg.

Dataset Method

Base 0.0380 0.0495 0.0737 0.1195 - 0.0391 0.0516 0.0757 0.1258 -

+ERL (Improv.)

0.0406 (↑6.84%)

0.0521 (↑5.25%)

0.0770 (↑4.48%)

0.1227 (↑2.68%) ↑4.81%

0.0417 (↑6.65%)

0.0531 (↑2.91%)

0.0832 (↑9.91%)

0.1283 (↑1.99%) ↑5.36%

Yelp

+PRL (Improv.)

0.0413 (↑8.68%)

0.0529 (↑6.87%)

0.0788 (↑6.92%)

0.1253 (↑4.85%) ↑6.83%

0.0410 (↑4.86%)

0.0532 (↑3.10%)

0.0804 (↑6.21%)

0.1289 (↑2.46%) ↑4.16%

Base 0.0328 0.0421 0.0683 0.1054 - 0.0350 0.0438 0.0716 0.1065 -

+ERL (Improv.)

0.0364 (↑10.98%)

0.0440 (↑4.51%)

0.0711 (↑4.10%)

0.1015 (↓3.70%) ↑3.97%

0.0392 (↑12.00%)

0.0485 (↑10.73%)

0.0744 (↑3.91%)

0.1112 (↑4.41%) ↑7.76%

Video & Games

+PRL (Improv.)

0.0352 (↑7.32%)

0.0433 (↑2.85%)

0.0658 (↓3.66%)

0.0982 (↓6.83%) ↓0.08%

0.0371 (↑6.00%)

0.0462 (↑5.48%)

0.0708 (↓1.12%)

0.1067 (↑0.19%) ↑2.64%

Base 0.0820 0.1041 0.1643 0.2522 - 0.0846 0.1050 0.1697 0.2510 -

+ERL (Improv.)

0.0851 (↑3.78%)

0.1075 (↑3.27%)

0.1669 (↑1.58%)

0.2556 (↑1.35%) ↑2.49%

0.0881 (↑4.14%)

0.1071 (↑2.00%)

0.1711 (↑0.82%)

0.2466 (↓1.75%) ↑1.30%

Software

+PRL (Improv.)

0.0869 (↑5.98%)

0.1076 (↑3.36%)

0.1687 (↑2.68%)

0.2518 (↓0.16%) ↑2.96%

0.0917 (↑8.39%)

0.1120 (↑6.67%)

0.1723 (↑1.53%)

0.2532 (↑0.88%) ↑4.37%

Base 0.0150 0.0208 0.0298 0.0527 - 0.0186 0.0235 0.0405 0.0604 -

+ERL (Improv.)

0.0208 (↑38.67%)

0.0259 (↑24.52%)

0.0428 (↑43.62%)

0.0629 (↑19.35%) ↑31.54%

0.0199 (↑6.99%)

0.0248 (↑5.53%)

0.0417 (↑2.96%)

0.0609 (↑0.83%) ↑4.08%

CDs & Vinyl

+PRL (Improv.)

0.0191 (↑27.33%)

0.0253 (↑21.63%)

0.0394 (↑32.21%)

0.0640 (↑21.44%) ↑25.66%

0.0198 (↑6.45%)

0.0249 (↑5.96%)

0.0417 (↑2.96%)

0.0618 (↑2.32%) ↑4.42%

Base 0.0152 0.0199 0.0315 0.0501 - 0.0176 0.0231 0.0371 0.0588 -

+ERL (Improv.)

0.0183 (↑20.39%)

0.0239 (↑20.10%)

0.0367 (↑16.51%)

0.0589 (↑17.56%) ↑18.64%

0.0184 (↑4.55%)

0.0242 (↑4.76%)

0.0373 (↑0.54%)

0.0602 (↑2.38%) ↑3.06%

Baby & Products

+PRL (Improv.)

0.0182 (↑19.74%)

0.0236 (↑18.59%)

0.0359 (↑13.97%)

0.0575 (↑14.77%) ↑16.77%

0.0189 (↑7.39%)

0.0247 (↑6.93%)

0.0376 (↑1.35%)

0.0611 (↑3.91%) ↑4.89%

spaces, these approaches effectively alleviate the recommendation bias, where underrepresented users and items are dominated by popular ones.

• Our proposed ERL and PRL methods, based on the ReaRec framework, consistently and significantly surpass baseline models at most cases. For example, for ID-based methods, ERL and PRL built on SASRec achieve average improvements of 6.76% and 8.21% respectively across all metrics on five datasets. Similarly, for text-based methods, ERL and PRL built on UniSRec outperform the base model by 12.29% and 10.43% on average. Unlike conventional SeqRec models, our reasoning-enhanced framework employs latent-space computations during the inference phase to deepen the feature crossing depth. This effectively unlock the latent reasoning power of various SeqRec backbones, demonstrating that increasing inference-time computation is a promising avenue for improving recommendation performance.

### 4.3 Further Analysis

In this section, we provide a comprehensive evaluation of the proposed ReaRec framework. We first conduct an in-depth analysis of how reasoning depth impacts performance across different user and item groups(Sec. 4.3.1). We then explore the impact of reasoning steps on recommendation performance (Sec. 4.3.2) and inference

latency (Sec. 4.3.3). Next, we perform a detailed ablation study (Sec. 4.3.4) and hyperparameter sensitivity analysis (Sec. 4.3.5). Finally, we investigate the visualization of reasoning hidden states to gain insights into the model’s reasoning process (Sec. 4.3.6). Unless otherwise specified, we primarily conduct detailed experiments on the PRL method based on SASRec backbone using the Yelp and Video & Games datasets.

4.3.1 Robustness Analysis Across User and Item Subgroups. To further analyze the robustness of our proposed ReaRec framework, we split users and items into different subgroups to gain deeper insights into the performance of the multi-step reasoning framework. Specifically, for users, we divide users into four equalsized groups based on sequence length: {UG-0, UG-1, UG-2, UG3}, where higher group numbers indicate longer sequences. For items, following previous work [51, 70], we group them into four groups based on interaction frequency: {IG-0, IG-1, IG-2, IG-3}, where higher group numbers indicate more popular items. We ensure each item group contains the same sample numbers. We fix the reasoning steps for PRL method during training at three, and analyze how recommendation performance changes for different user and item groups as reasoning steps increase during the inference phase. The detailed experimental results are shown in Fig. 4.

| | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | |

| | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | |

Figure 4: Robustness study w.r.t different user and item subgroups on Yelp dataset. ‘Step-𝑥’ represents the recommendation performance at the 𝑥-th reasoning step. ‘UG’ and ‘IG’ denote User and Item Group, respectively, where higher group numbers indicate longer sequences and more popular items.

We can clearly observe distinct performance trends across different user and item subgroups. For short-sequence user groups and unpopular item groups, recommendation quality (NDCG@20) tends to steadily improve as the reasoning steps increase. For example, in the item group IG-1, more reasoning steps bring better performance gains of 12.08%, 16.35%, and 18.69%, respectively. In contrast, performance tends to decline for users with long interaction sequences and popular items as the reasoning steps increase. We speculate that this is primarily because longer user sequences provide richer contextual information, making it easier to mine interest evolution patterns. Beyond a certain point, additional inference computation fails to yield further performance improvements and even leads to performance degradation due to overthinking. Similarly, for high-popularity items, their well-trained representations allow the recommender to easily capture collaborative signals, making deeper feature crossing depth less beneficial. Overall, longtail users and items usually require more thinking space to reason sparse interaction signals, whereas highly active users and items may not need redundant computational expansion. In the future, it may be necessary to develop differentiated fast and slow reasoning mechanism for different user sequences to further improve overall recommendation performance.

- 4.3.2 Impact of Reasoning Steps on Recommendation Performance. We investigate the variation trend of recommendation performance under different inference steps, that is, we train and perform inference using specified numbers of reasoning steps. We adopt NDCG@20 as the main evaluation metric. We compare the following approaches: (1) Base: The original SASRec sequential recommender serves as the baseline without reasoning enhancement; (2) Naive: Based on the Base method, we extend it to a

Table 4: Inference time statistics for different steps. “Cost Inc.” is short for Cost Increase, where higher values indicate greater time overhead. Efficiency experiments are conducted on a single A100-40G GPU. Note that the optimal performance typically corresponds to Step-2.

Base Step-1 Step-2 Step-3 Step-4 Step-5 SASRec 5.6761 5.7985 5.8752 5.9305 6.0310 6.2786

Cost Inc. - 2.16% 3.51% 4.48% 6.25% 10.61%

BERT4Rec 5.6535 5.7685 5.9174 5.9621 6.0862 6.1224 Cost Inc. - 2.03% 4.67% 5.46% 7.65% 8.29% UniSRec 5.6061 5.6312 5.7596 5.8732 6.0303 6.0502 Cost Inc. - 0.45% 2.74% 4.76% 7.57% 7.92%

MoRec 5.6638 5.7143 5.8391 5.9565 5.9659 5.9812 Cost Inc. - 0.89% 3.10% 5.17% 5.33% 5.60%

Note: All time units are in second (s).

Yelp

Video & Games

0.0520

0.0400

0.0375

0.0482

NDCG@20

NDCG@20

0.0350

0.0445

0.0325

0.0407

0.0300

0.0370

1 2 3 4 5

1 2 3 4 5

# Step

# Step

Base Naive RPE ERL PRL

#### Figure 5: The performance variation trend of different methods under different reasoning steps.

multi-step reasoning paradigm, where the last hidden state is autoregressively fed back into the model, and only the final position is used directly as the user representation; (3) RPE: Building on the Naive approach, we further integrate Reasoning Positional Embeddings to bridge the task gap between sequence encoding mode and reasoning mode. Additionally, we also explore the performance of (4) Ensemble Reasoning Learning (ERL) and (5) Progressive Reasoning Learning (PRL) under multi-step reasoning.

As shown in Fig. 5, the Naive method, which lacks a specialized design, does not yield performance improvements and even underperforms compared to the base model. This is likely due to the model’s inability to distinguish between sequence encoding and the reasoning phases. Introducing reasoning positional embeddings (+RPE) effectively mitigates this task gap, yielding obvious performance gains. However, simply optimizing cross-entropy loss on the final-step output does not provide adequate supervision guidance for the intermediate reasoning states, potentially leading to reasoning pattern degradation and error accumulation. In contrast, our ERL and PRL methods significantly alleviate these issues by explicitly injecting stepwise supervision signals, reducing the optimization difficulty to some extent. Notably, as the number of inference steps increases, we observe a consistent performance decline across all methods. This suggests that excessive reasoning

Yelp (NDCG@20)

Yelp (NDCG@20)

Yelp (NDCG@20)

Video & Games (NDCG@20)

Video & Games (NDCG@20)

Video & Games (NDCG@20)

0.0510

0.0500

0.0490

0.0400

0.0390

0.0388

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |

0.0368

0.0363

0.0490

0.0482

0.0472

0.0385

0.0470

0.0465

0.0455

0.0325

0.0345

0.0382

0.0380

0.0448

0.0438

0.0288

0.0450

0.0322

0.0430

0.0430

0.0420

0.0250

0.0300

0.0377

0.05 0.1 0.5 1.0 2.0 5.0

1.0 1.2 1.5 2.0 5.0 10.0

0.001 0.005 0.01 0.05 0.1

0.05 0.1 0.5 1.0 2.0 5.0

1.0 1.2 1.5 2.0 5.0 10.0

0.001 0.005 0.01 0.05 0.1

Yelp (Recall@20)

Yelp (Recall@20)

Yelp (Recall@20)

Video & Games (Recall@20)

Video & Games (Recall@20)

Video & Games (Recall@20)

0.1200

0.1180

0.1150

0.1000

0.0950

0.0950

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

0.1088

0.0888

0.1150

0.1140

0.0900

0.0912

0.1100

0.1100

0.1025

0.0800

0.0825

0.0875

0.0838

0.0763

0.1050

0.1060

0.0700

0.0962

0.1000

0.1020

0.0900

0.0600

0.0700

0.0800

0.05 0.1 0.5 1.0 2.0 5.0

1.0 1.2 1.5 2.0 5.0 10.0

0.001 0.005 0.01 0.05 0.1

0.05 0.1 0.5 1.0 2.0 5.0

1.0 1.2 1.5 2.0 5.0 10.0

0.001 0.005 0.01 0.05 0.1

#### Figure 6: Performance comparison w.r.t. different hyperparameters, including base temperature 𝜏, temperature decay rate 𝛼, and KL regularization strength 𝜆. The green and orange lines represent the PRL and ERL methods, respectively.

- 0.0478

0.0390

0.0510

0.0410

w/o KL w/ KL

w/o RCL w/ RCL

| |
|---|

| |
|---|

0.0475

0.0386

0.0495

0.0387

0.0471

0.0382

0.0480

0.0365

0.0468

0.0379

0.0465

0.0343

0.0465

0.0375

0.0450

0.0320

Yelp Video & Games

Yelp Video & Games

#### Figure 7: Ablation study for key components in ERL and PRL.

may trigger “overthinking”—simple user interaction patterns may not require intensive latent reasoning. Moreover, considering the post-hoc optimal step analysis in Fig. 2, developing an adaptive inference depth selection mechanism to balance reasoning depth and user sequence complexity presents a highly meaningful direction for future research.

- 4.3.3 Impact of Reasoning Steps on Inference Latency. Our ReaRec framework’s expanded computational demands during inference introduce additional overhead. To evaluate this, we use the PRL method as an example, measuring the time cost on the test set as reasoning steps increase, as shown in Table 4. The results indicate that, despite adopting a recurrent autoregressive inference mechanism, the extra latency remains manageable. This efficiency stems from KV Caching technique, which significantly reduces attention computation complexity from 𝑂(𝑁2) to 𝑂(𝑁) by reusing key and value vectors of past steps, thereby effectively minimizing redundant calculations. Further analysis with Fig. 5 reveals that our approaches generally achieve optimal performance at two reasoning steps. This means that our method increases performance by an average of 7.49% across all metrics with only a modest latency overhead of 3.51%, which is acceptable and practical for real-world deployment in industrial recommender systems. These results suggest that our efficient ReaRec framework holds great promise for real-world applications.
- 4.3.4 Ablation Study. In this section, we present the ablation study of our proposed method. Specifically, we focus on two key

components: (1) KL regularization term (KL) in the ERL approach (Sec. 3.2.2) and (2) Reasoning-aware Contrastive Learning (RCL) in the PRL method (Sec. 3.3.2). Specifically, we conduct ablation studies by removing the auxiliary loss terms from both methods and evaluate their performance on NDCG@20.

As shown in Fig. 7, the experimental results clearly indicate that the ERL method without KL regularization performs worse than the full model, suggesting that the model probably suffers from pattern degradation in reasoning states, leading to highly homogeneous outputs. Similarly, the PRL method without RCL also yields suboptimal recommendation performance. While progressive temperature scheduling helps adjust the learned distribution sharpness across different steps, the absence of robust inference mechanisms prevents the recommender from self-correcting deviations in intermediate reasoning states. As a result, it struggles to effectively approximate the user’s true preference distribution.

4.3.5 Sensitivity Analysis. In this section, we examine the effects of three key hyperparameters, 𝜏, 𝛼, and 𝜆 on the Yelp and Video & Games datasets. Here, 𝜏 and 𝛼 represent the base temperature and progressive temperature decay rate in the PRL method, respectively, while 𝜆 denotes the KL regularization strength in the ERL method. We next analyze how variations in each hyperparameter influence model performance, with the experimental results shown in Fig. 6.

Performance Comparison w.r.t Base Temperature 𝝉 in PRL. By tuning the base temperature 𝜏 across {0.05, 0.1, 0.5, 1.0, 2.0, 5.0}, we can observe that as𝜏 increases, model performance gradually improves. This suggests that overly sharp probability distribution does not align with users’ potential preference distributions. In other words, forcing the model to learn extreme positive and negative sample preferences from noisy interaction data hinders generalization ability. However, too large base temperatures also lead to degraded recommendation performance. We hypothesize that a large 𝜏 value may blur the ranking differences among candidate items, making it harder for the recommender to learn meaningful sequential patterns. Thus, setting a moderate 𝜏 is crucial for achieving satisfactory performance.

Performance Comparison w.r.t Temperature Decay Rate 𝜶 in PRL. To enable the model to learn a more precise user preference

200

- Step 0

- Step 1

- Step 2

- Step 0

- Step 1

- Step 2

500

250

100

- Step 0

- Step 1

- Step 2

- Step 0

- Step 1

- Step 2 R20

400

150

200

75

300

150

PredictedScore

PredictedScore

PredictedScore

PredictedScore

50

100

R33

200

R74 R42

R14 R6

100

25

50

R8

100

R4

R12

50

0

0

0

R2

0

100

25

R6

[Figure 19]

50

R22

50

200

50

100

300

10K 8K 6K 4K 2K 0K

10K 8K 6K 4K 2K 0K

10K 8K 6K 4K 2K 0K

10K 8K 6K 4K 2K 0K

(b) Item Rank( = 2.0)

(c) Item Rank( = 1.2)

(a) Item Rank( = 5.0)

(d) Item Rank(w/o RCL)

ERL

#### Figure 8: Case study on rank changes of target item across different reasoning steps. ‘Rx’ represents the predicted rank of the target item, e.g., ‘R42’ indicates the predicted score of the target item ranks 42nd among all candidate items.

1.00

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 20]

1.00

[Figure 21]

|1.0|0|0.9|6|0.9|1|0.8|8|
|---|---|---|---|---|---|---|---|
| | | | | | | | |
|0.9|6|1.0|0|0.9|9|0.9|8|
| | | | | | | | |
|0.9|1|0.9|9|1.0|0|1.0|0|
| | | | | | | | |
|0.8|8|0.9|8|1.0|0|1.0|0|
| | | | | | | | |

[Figure 22]

|1.0|0|0.9|6|0.8|4|0.6|4|
|---|---|---|---|---|---|---|---|
| | | | | | | | |
|0.9|6|1.0|0|0.9|5|0.7|9|
| | | | | | | | |
|0.8|4|0.9|5|1.0|0|0.9|4|
| | | | | | | | |
|0.6|4|0.7|9|0.9|4|1.0|0|
| | | | | | | | |

0123

0.98

0.95

0123

ERL w/o KL

0.90

0.96

0.85

0.94

0.80

0.92

0.75

##### (a) ERL w/o KL

0.90

0.70

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 23]

0.65

0.88

0 1 2 3

0 1 2 3

(a) RPE

(b) PRL

ERL

1.0

1.00

[Figure 24]

|1.0|0|0.9|1|0.4|5|0.7|9|
|---|---|---|---|---|---|---|---|
| | | | | | | | |
|0.9|1|1.0|0|0.6|3|0.8|9|
| | | | | | | | |
|0.4|5|0.6|3|1.0|0|0.9|0|
| | | | | | | | |
|0.7|9|0.8|9|0.9|0|1.0|0|
| | | | | | | | |

[Figure 25]

|1.0|0|0.9|1|0.7|3|0.8|6|
|---|---|---|---|---|---|---|---|
| | | | | | | | |
|0.9|1|1.0|0|0.9|5|0.9|9|
| | | | | | | | |
|0.7|3|0.9|5|1.0|0|0.9|8|
| | | | | | | | |
|0.8|6|0.9|9|0.9|8|1.0|0|
| | | | | | | | |

0123

0123

0.9

0.95

(b) ERL

0.8

0.90

[Figure 26]

| |
|---|

| |
|---|

|KL|
|---|

|full|
|---|

Figure 10: The embedding visualization of the ll ERL method vs. its ablated version without K regularization. Dashed boxes highlight high similarity between different reasoning steps (Step 0 ∼ Step 3) in the ablated version.

0.7

0.85

0.6

ERL w/o KL

0.80

0.5

0.75

0 1 2 3

0 1 2 3

(c) ERL w/o KL

(d) ERL

Performance Comparison w.r.t KL Regularization Strength 𝝀 in ERL.In our ensemble reasoning learning method, we utilize the KL regularization coefficient𝜆 to balance reasoning diversity. Specifically, we explore the impact of different regularization strengths by varying 𝜆 within the range {0.001, 0.005, 0.01, 0.05, 0.1}. From Fig. 6, we can observe that the model is usually not sensitive to the 𝜆. However, the recommendation performance drops significantly when 𝜆 exceeds a certain threshold (e.g., 0.05). We attribute this to that enforcing the model to learn excessively divergent sequential patterns across multi-step reasoning might actually disrupt the sequential modeling capability. Although our designed KL regularization aims to encourage the model to explore diverse reasoning paths, too strong regularization may dominate gradient optimization, increasing the optimization challenges and ultimately leading to performance degradation.

#### Figure 9: Visualization of similarity in multi-step reasoning hidden states for different methods.

distribution, we introduce a progressive temperature annealing mechanism controlled by temperature decay rate 𝛼, adjusting the sharpness of the learned distribution at different reasoning steps. We vary it within {1.0, 1.2, 1.5, 2.0, 5.0, 10.0} to observe the performance changes. A consistent finding is that a moderate 𝛼 usually achieves the best performance, while too small and too large decay rates lead to suboptimal results. This is as expected, as 𝛼 is too small (in the extreme case, 𝛼 = 1.0), the score distributions learned at different reasoning steps remain the same, causing the model to take shortcuts like replicating the prior reasoning state. Such pattern collapse prevents the model from leveraging reasoning enhancement in inference. On the other hand, overly high 𝛼 (e.g., 𝛼 = 10.0) still leads to performance degradation. This is because, under our exponential temperature decay strategy, an aggressive temperature change triggers a rapid distribution sharpness transition from smooth to sharp distribution, disrupting the model’s curriculum-style reasoning process. Therefore, choosing an approximate temperature decay rate is critical for reducing the model’s optimization difficulty.

4.3.6 Embedding Visualization Analysis. To analyze the hidden state dynamics during reasoning, we visualize the similarity heatmaps of multi-step reasoning outputs for different methods, as shown in Fig. 9. Specifically, by comparing Fig. 9(a) and Fig. 9(b), it is obvious that the RPE variant exhibits high homogeneity in reasoning states. For instance, the similarity scores between the final output and the previous two steps are almost identical i.e.,

[Figure 27]

History Items

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

H1 H2 H3 H4 H5

[Figure 32]

|ID|Item Title|
|---|---|
|H1|Halo: The Master Chief Collection|
|H2|Halo 5: Guardians 9 Gold REQ Packs|
|H3|SanDisk 128GB microSDXC-Card|
|H4|TOSTAR Dust Cover for PS5|
|H5|innoAura Vertical Stand for Xbox Series S|
|R0|Conflict Desert Storm - Xbox|
|R1|Xbox 360 controller led mod RING OF LIGHT LEDS|
|R2|Resident Evil 2 - Xbox One|

Reasoning

[Figure 33]

[Figure 34]

[Figure 35]

R0 R1 R2

Figure 11: Case study of multi-step inference on the Video & Games Dataset. ‘H𝑥’ represents historical items, with smaller 𝑥 indicating more recent interactions. ‘R𝑥’ represents the top-1 recommended items at the 𝑥-th reasoning step, with larger 𝑥 indicating later reasoning steps.

- 1.00 and 0.98, which confirms the reasoning pattern degradation issue claimed before. In contrast, by incorporating a progressive reasoning learning approach, PRL effectively leverages reasoningenhanced computation for performance improvement. The ERL method demonstrates analogous issues, where KL regularization encourages the model to capture diverse sequential patterns through aggregating multi-order feature crossing. Additionally, we visualize the specific reasoning representations in Fig. 10, where we can observe that the ERL method without KL constraint reveals more overlapping patterns across different reasoning steps. This further validates that our proposed methods can effectively address core challenges in multi-step reasoning sequential models.

### 4.4 Case Studies

In this section, to better demonstrate the benefits of our reasoningenhanced sequential recommendation, we present illustrative cases showing the rank change trends of target items during mulit-step inference, along with a specific example from Video & Games dataset.

- 4.4.1 Rank Change Analysis of Target Items. We evaluate target item ranking trajectories on the Yelp dataset using PRL methods with varying temperature decay coefficient (𝛼) and an ablated version without RCL. As shown in Fig. 8, we observe that the full PRL method progressively improves the target item ranking within the overall candidate pool as the depth of reasoning increases, aligning with our expectations. Additionally, we find that for smaller 𝛼, the score distribution across different inference steps transitions smoothly, whereas larger 𝛼 induces distribution changes more aggressively, which is consistent with our analysis in Sec. 4.3.5. Moreover, the ablated version without RCL leads to reasoning errors where increasing the number of reasoning steps incorrectly pushes the target item further down the ranking (e.g., target item rank drops from #12 at step 1 to #22 at step 2 in Fig. 8(d)).
- 4.4.2 Case Study in Real-world Recommendation Scenario. We present a case study to illustrate the stepwise preference refinement effect of the PRL method, as shown in Fig. 11. To be specific, the user previously purchased Halo and Halo 5, two First-Person

Shooter (FPS) games for the XBox-One platform on Amazon. After that, the user bought related accessories, i.e., a memory card, a dust cover, and a stand. Next, the corresponding top-1 recommended items are given by the multi-step reasoning outputs, denoted as R0, R1, and R2, respectively. At step R0, the model successfully captures the user’s preference for FPS games on the XBox platform. However, this recommendation (i.e., Conflict Desert Storm) lacks timeliness and may not align with a gaming enthusiast’s tendency to prefer newer releases. At step R1, the model adjusts by recommending a game controller, reflecting the user’s recent purchase habits (i.e., gaming accessories). However, this recommendation remains suboptimal, as it only reflects collaborative relevance rather than sequential characteristics (typically, users phase controllers before accessories like stands) and lacks recommendation diversity (as recent purchases were all accessories). Surprisingly, at the final inference step, the model recommends Resident Evil 2, a newly released shooter game that matches the actual target item and aligns well with the true preference, further validating how recurrent reasoning resolves ambiguity by integrating temporal context, collaborative relevance, and output diversity.

### 5 Related Work

Sequential Recommendation. Sequential recommendation, as one of the core tasks within the field of Recommender Systems (RS), aims to predict the next item with which a user may interact by modeling behavior patterns and evolving user interests from useritem interaction sequences [2, 13, 59]. Early studies focused on item-to-item collaborative transitions, typically employing Markov Chain-based matrix factorization methods [22, 42] to achieve nextitem predictions. With the advent of deep learning, researchers began exploring various sequential modeling architectures, such as Recurrent Neural Networks (RNNs) [23, 24, 40], Convolutional Neural Networks (CNNs) [52, 68, 75], and Transformer [5, 31, 71], to enhance modeling capabilities. Specifically, GRU4Rec [23] first introduced the GRU networks for session-based recommendation. Caser [52] applied convolutional operations by treating the item sequence embedding matrix as an “image” to extract multi-level interaction features. With the emergence of the Transformer architecture, numerous studies in sequential recommendation shifted toward self-attention-based models. For instance, SASRec [31], a classic sequential recommendation baseline, incorporated self-attention to automatically learn the importance weights of historical items. BERT4Rec [49] further advanced this by adopting bidirectional encoding to capture more contextual dependencies from both directions of the sequence. Furthermore, to address data sparsity and cold-start issues, another line of research focused on leveraging common item attributes (such as text and images) to learn universal item and sequence representations [10, 25, 26, 76]. For example, UniSRec [26] utilized multi-domain recommendation data to learn transferable sequential models. However, existing methods remain constrained by reasoning-free forward computation. In this paper, we further explore the potential reasoning capabilities of sequential recommenders by extending the multi-step computational depth at inference time.

Inference-time Reasoning. As the scaling law of large language models (LLMs) at the training stage has gradually reached its

bottleneck [4, 33, 48], researchers have shifted their focus toward inference-time scaling. Notably, OpenAI’s O1 series [28], Qwen’s QwQ series [54], and DeepSeek’s R1 series [18] have emerged

- as key milestone works, marking the transition from conversational AI to reasoning-intensive AI and opening promising pathways toward achieving Artificial General Intelligence (AGI). These works leverage emerging long Chain-of-Thought (CoT) mechanisms to reveal excellent test-time scaling phenomena—where increased computational power (via generating more tokens) during inference substantially improves the model’s problem-solving abilities [1, 8, 61, 62, 69, 72, 79]. Extended reasoning space enables depth-scalable exploration, manifesting self-reflection capabilities (e.g., “Aha Moments”) that surpass traditional short-chain reasoning limitations [4, 29]. Compared with prior methods constrained by token-by-token generation in the discrete language spaces, which restricts the model’s expressive ability, another research direction focuses on implicit chain of thought reasoning in latent spaces, achieving both efficiency and performance gains in large language models [3, 15, 20, 67] and multimodal foundation models [21, 46]. For example, Coconut [20] introduces continuous thinking in the latent reasoning space of LLMs, while Heima [46] compresses the entire multimodal CoT process into a single high-level thinking token (i.e., <CoT>) to eliminate redundant intermediate token generation. Inspired by this think-before-action paradigm, we pioneer the exploration of implicit reasoning-enhanced sequential recommendation framework in this paper, proposing two lightweight reasoning learning methods to push the performance ceiling of sequential recommenders.

6 Conclusion and Future Work

- 6.1 Conclusion

In this work, we pioneer the integration of deep reasoning into sequentialrecommendationbyintroducingReaRec,a novelinferencetime computing framework inspired by the think-before-action paradigm. Unlike traditional direct inference models, ReaRec expands computational depth through multi-step implicit reasoning, enabling the SeqRec model to think before recommendation. We also propose two lightweight learning strategies to address the challenges of multi-step reasoning-process optimization: Ensemble Reasoning Learning (ERL) and Progressive Reasoning Learning (PRL), which enhance reasoning robustness and effectiveness. Extensive experiments across five real-world datasets validate the effectiveness and generalizability of our proposed ReaRec. Notably, ReaRec not only improves performance for long-tail users and items but also raises the performance ceiling of existing SeqRec backbones by up to 50% with post-hoc optimal step selection, highlighting the untapped potential of ReaRec for sequential recommendation. We believe our work opens a promising direction for future research at the intersection of reasoning and recommendation.

- 6.2 Future Work

While our proposed simple inference-time computational strategies successfully unlock the reasoning potential of sequential recommenders and achieve promising performance gains, this work serves primarily as an initial exploratory effort. Consequently,

we have also identified some immediate challenges and opportunities for future research:

- • Adaptive Inference Depth Selection. As shown in Fig. 4, we observe that while our method effectively improves recommendation performance to cold-start users and long-tail items, it paradoxically induces performance degradation for high-activity users and popular items. We attribute this to the overthinking phenomenon—the additional computational steps provide negligible benefits for well-learned patterns, as their preferences may be sufficiently captured through shallow reasoning-free inference. Moreover, complemented by the post-hoc optimal step analysis in Fig. 2, which illustrates the performance upper bound corresponding to the optimal reasoning step, it becomes evident that there is still a significant gap between the model’s current performance and the theoretical upper bound. Therefore, how to develop an adaptive inference depth selection policy to balance computational depth and sequence complexity is an open research direction.
- • Parameter Disentanglement Between Encoding and Reasoning. Our current ReaRec framework adopts an implicit reasoning mechanism similar to large reasoning models, where the item sequence encoding phase shares parameters with the reasoning computations. While this design ensures parameter efficiency, it creates task ambiguity—the same neural modules have to simultaneously handle two distinct objectives: (1) precise item presentation learning and (2) multi-step forward reasoning. Although we propose reasoning position embeddings (cf. Sec. 3.1.2) to alleviate this issue, the suboptimal performance trajectories (improvement followed by decline as steps increase shown in Fig. 5) suggests our solution may not be optimal. A promising future direction is to explore parameter decoupling between item encoding and deep sequential reasoning at the model level. This separation could potentially reduce task interference, allowing for more specialized representation learning and better adaption to multi-step inference, ultimately leading to improved recommendation quality.
- • The Missing Inference-time Scaling Law. In the field of large reasoning models, recent studies [48, 65] suggest that longer reasoning chains often lead to better reasoning capabilities, thereby improving downstream task performance—this phenomenon is known as the inference-time scaling law. However, our experiments (cf. Sec. 4.3.2) demonstrate that as the number of reasoning steps increases, our framework does not achieve the expected scaling law behavior in a perfect manner. This discrepancy raises several intriguing research questions:

- – Does a scaling law exist for inference-time computation in recommendation systems?
- – If so, how can we design more effective reasoning-enhanced sequential recommenders to better realize such a scaling law?

Further exploration in this direction could unlock new insights into the model’s reasoning capabilities and ultimately push the boundaries of reasoning-enhanced recommendation research.

- • TheoreticalAnalysis.Intuitively,increasinginference-time computational depth allows sequential recommenders to capture higher-order sequential feature crossing, leading to more accurate user preference predictions. To solidify this intuition, future

work could focus on theoretical analysis of how multi-step reasoning contributes to improved recommendation performance. Establishing a strong theoretical foundation for reasoning-enhanced sequential recommendation could pave the way for more principled model design and optimization strategies.

• Efficient Inference Mechanism. While our efficiency experiments 4.3.3 confirm that ReaRec introduces only marginal latency overhead, future advancements in sequential recommendation inference-time scaling laws may still raise efficiency concerns with the autoregressive generation paradigm. To address this, we propose several potential optimization strategies for future exploration, including incorporating linear attention mechanisms [60], model quantization [73], and long-to-short reasoning distillation [53] techniques to further achieve lighter and faster inference efficiency for industrial-scale deployment.

### References

- [1] Maciej Besta, Nils Blach, Ales Kubicek, Robert Gerstenberger, Michal Podstawski, Lukas Gianinazzi, Joanna Gajda, Tomasz Lehmann, Hubert Niewiadomski, Piotr Nyczyk, et al. 2024. Graph of thoughts: Solving elaborate problems with large language models. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 38. 17682–17690.
- [2] Tesfaye Fenta Boka, Zhendong Niu, and Rama Bastola Neupane. 2024. A survey of sequential recommendation systems: Techniques, evaluation, and future directions. Information Systems (2024), 102427.
- [3] Haolin Chen, Yihao Feng, Zuxin Liu, Weiran Yao, Akshara Prabhakar, Shelby Heinecke, Ricky Ho, Phil Mui, Silvio Savarese, Caiming Xiong, et al. 2024. Language models are hidden reasoners: Unlocking latent reasoning capabilities via self-rewarding. arXiv preprint arXiv:2411.04282 (2024).
- [4] Qiguang Chen, Libo Qin, Jinhao Liu, Dengyun Peng, Jiannan Guan, Peng Wang, Mengkang Hu, Yuhang Zhou, Te Gao, and Wangxiang Che. 2025. Towards reasoning era: A survey of long chain-of-thought for reasoning large language models. arXiv preprint arXiv:2503.09567 (2025).
- [5] Qiwei Chen, Huan Zhao, Wei Li, Pipei Huang, and Wenwu Ou. 2019. Behavior sequence transformer for e-commerce recommendation in alibaba. In Proceedings of the 1st international workshop on deep learning practice for high-dimensional sparse data. 1–4.
- [6] Xu Chen, Hongteng Xu, Yongfeng Zhang, Jiaxi Tang, Yixin Cao, Zheng Qin, and Hongyuan Zha. 2018. Sequential recommendation with user memory networks. In Proceedings of the eleventh ACM international conference on web search and data mining. 108–116.
- [7] Yongjun Chen, Zhiwei Liu, Jia Li, Julian McAuley, and Caiming Xiong. 2022. Intent contrastive learning for sequential recommendation. In Proceedings of the ACM web conference 2022. 2172–2182.
- [8] Zheng Chu, Jingchang Chen, Qianglong Chen, Weijiang Yu, Tao He, Haotian Wang, Weihua Peng, Ming Liu, Bing Qin, and Ting Liu. 2023. Navigate through enigmatic labyrinth a survey of chain of thought reasoning: Advances, frontiers and future. arXiv preprint arXiv:2309.15402 (2023).
- [9] Sunhao Dai, Ninglu Shao, Jieming Zhu, Xiao Zhang, Zhenhua Dong, Jun Xu, Quanyu Dai, and Ji-Rong Wen. 2024. Modeling user attention in music recommendation. In 2024 IEEE 40th International Conference on Data Engineering (ICDE). IEEE, 761–774.
- [10] Hao Ding, Yifei Ma, Anoop Deoras, Yuyang Wang, and Hao Wang. 2021. Zeroshot recommender systems. arXiv preprint arXiv:2105.08318 (2021).
- [11] Xibin Dong, Zhiwen Yu, Wenming Cao, Yifan Shi, and Qianli Ma. 2020. A survey on ensemble learning. Frontiers of Computer Science 14 (2020), 241–258.
- [12] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. 2020. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929 (2020).
- [13] Hui Fang, Danning Zhang, Yiheng Shu, and Guibing Guo. 2020. Deep learning for sequential recommendation: Algorithms, influential factors, and evaluations. ACM Transactions on Information Systems (TOIS) 39, 1 (2020), 1–42.
- [14] Guhao Feng, Bohang Zhang, Yuntian Gu, Haotian Ye, Di He, and Liwei Wang.

2023. Towards revealing the mystery behind chain of thought: a theoretical perspective. Advances in Neural Information Processing Systems 36 (2023), 70757– 70798.

- [15] Jonas Geiping, Sean McLeish, Neel Jain, John Kirchenbauer, Siddharth Singh, Brian R Bartoldson, Bhavya Kailkhura, Abhinav Bhatele, and Tom Goldstein.

2025. Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach. arXiv preprint arXiv:2502.05171 (2025).

- [16] Binzong Geng, Zhaoxin Huan, Xiaolu Zhang, Yong He, Liang Zhang, Fajie Yuan, Jun Zhou, and Linjian Mo. 2024. Breaking the length barrier: Llm-enhanced CTR prediction in long textual user behaviors. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval. 2311–2315.
- [17] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783

(2024).

- [18] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. 2025. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948 (2025).
- [19] Daya Guo, Qihao Zhu, Dejian Yang, Zhenda Xie, Kai Dong, Wentao Zhang, Guanting Chen, Xiao Bi, Yu Wu, YK Li, et al. 2024. DeepSeek-Coder: When the Large Language Model Meets Programming–The Rise of Code Intelligence. arXiv preprint arXiv:2401.14196 (2024).
- [20] Shibo Hao, Sainbayar Sukhbaatar, DiJia Su, Xian Li, Zhiting Hu, Jason Weston, and Yuandong Tian. 2024. Training large language models to reason in a continuous latent space. arXiv preprint arXiv:2412.06769 (2024).
- [21] Liqi He, Zuchao Li, Xiantao Cai, and Ping Wang. 2024. Multi-modal latent space learning for chain-of-thought reasoning in language models. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 38. 18180–18187.
- [22] Ruining He and Julian McAuley. 2016. Fusing similarity models with markov chains for sparse sequential recommendation. In 2016 IEEE 16th international conference on data mining (ICDM). IEEE, 191–200.
- [23] Balázs Hidasi, Alexandros Karatzoglou, Linas Baltrunas, and Domonkos Tikk.

2015. Session-based recommendations with recurrent neural networks. arXiv preprint arXiv:1511.06939 (2015).

- [24] Balázs Hidasi, Massimo Quadrana, Alexandros Karatzoglou, and Domonkos Tikk. 2016. Parallel recurrent neural network architectures for feature-rich session-based recommendations. In Proceedings of the 10th ACM conference on recommender systems. 241–248.
- [25] Yupeng Hou, Zhankui He, Julian McAuley, and Wayne Xin Zhao. 2023. Learning vector-quantized item representation for transferable sequential recommenders. In Proceedings of the ACM Web Conference 2023. 1162–1171.
- [26] Yupeng Hou, Shanlei Mu, Wayne Xin Zhao, Yaliang Li, Bolin Ding, and Ji-Rong Wen. 2022. Towards universal sequence representation learning for recommender systems. In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 585–593.
- [27] HyeongJoo Hwang, Geon-Hyeong Kim, Seunghoon Hong, and Kee-Eung Kim.

2021. Multi-view representation learning via total correlation objective. Advances in Neural Information Processing Systems 34 (2021), 12194–12207.

- [28] Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. 2024. Openai o1 system card. arXiv preprint arXiv:2412.16720 (2024).
- [29] Yixin Ji, Juntao Li, Hai Ye, Kaixin Wu, Jia Xu, Linjian Mo, and Min Zhang. 2025. Test-time Computing: from System-1 Thinking to System-2 Thinking. arXiv preprint arXiv:2501.02497 (2025).
- [30] Yufei Jin, Heng Lian, Yi He, and Xingquan Zhu. 2024. HGDL: Heterogeneous Graph Label Distribution Learning. Advances in Neural Information Processing Systems 37 (2024), 40792–40830.
- [31] Wang-Cheng Kang and Julian McAuley. 2018. Self-attentive sequential recommendation. In 2018 IEEE international conference on data mining (ICDM). IEEE, 197–206.
- [32] Diederik P Kingma and Jimmy Ba. 2014. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980 (2014).
- [33] Komal Kumar, Tajamul Ashraf, Omkar Thawakar, Rao Muhammad Anwer, Hisham Cholakkal, Mubarak Shah, Ming-Hsuan Yang, Phillip HS Torr, Salman Khan, and Fahad Shahbaz Khan. 2025. Llm post-training: A deep dive into reasoning large language models. arXiv preprint arXiv:2502.21321 (2025).
- [34] Chenyi Lei, Yong Liu, Lingzi Zhang, Guoxin Wang, Haihong Tang, Houqiang Li, and Chunyan Miao. 2021. Semi: A sequential multi-modal information transfer network for e-commerce micro-video recommendations. In Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining. 3161–3171.
- [35] Jiacheng Li, Ming Wang, Jin Li, Jinmiao Fu, Xin Shen, Jingbo Shang, and Julian McAuley. 2023. Text is all you need: Learning language representations for sequential recommendation. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 1258–1267.
- [36] Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. 2023. Let’s verify step by step. In The Twelfth International Conference on Learning Representations.
- [37] Qidong Liu, Xian Wu, Wanyu Wang, Yejing Wang, Yuanshao Zhu, Xiangyu Zhao, Feng Tian, and Yefeng Zheng. 2024. Large language model empowered embedding generator for sequential recommendation. arXiv preprint arXiv:2409.19925 (2024).
- [38] Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019. Roberta: A

robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692

(2019).

- [39] Liangchen Luo, Yinxiao Liu, Rosanne Liu, Samrat Phatale, Harsh Lara, Yunxuan Li, Lei Shu, Yun Zhu, Lei Meng, Jiao Sun, et al. 2024. Improve mathematical reasoning in language models by automated process supervision. arXiv preprint arXiv:2406.06592 2 (2024).
- [40] Massimo Quadrana, Alexandros Karatzoglou, Balázs Hidasi, and Paolo Cremonesi.

2017. Personalizing session-based recommendations with hierarchical recurrent neural networks. In proceedings of the Eleventh ACM Conference on Recommender Systems. 130–137.

- [41] Nils Reimers and Iryna Gurevych. 2019. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP). Association for Computational Linguistics.
- [42] Steffen Rendle, Christoph Freudenthaler, and Lars Schmidt-Thieme. 2010. Factorizing personalized markov chains for next-basket recommendation. In Proceedings of the 19th international conference on World wide web. 811–820.
- [43] Omer Sagi and Lior Rokach. 2018. Ensemble learning: A survey. Wiley interdisciplinary reviews: data mining and knowledge discovery 8, 4 (2018), e1249.
- [44] Amrith Setlur, Chirag Nagpal, Adam Fisch, Xinyang Geng, Jacob Eisenstein, Rishabh Agarwal, Alekh Agarwal, Jonathan Berant, and Aviral Kumar. 2024. Rewarding progress: Scaling automated process verifiers for llm reasoning. arXiv preprint arXiv:2410.08146 (2024).
- [45] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. 2024. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300 (2024).
- [46] Xuan Shen, Yizhou Wang, Xiangxi Shi, Yanzhi Wang, Pu Zhao, and Jiuxiang Gu.

2025. Efficient Reasoning with Hidden Thinking. arXiv preprint arXiv:2501.19201

(2025).

- [47] Uriel Singer, Haggai Roitman, Yotam Eshel, Alexander Nus, Ido Guy, Or Levi, Idan Hasson, and Eliyahu Kiperwasser. 2022. Sequential modeling with multiple attributes for watchlist recommendation in e-commerce. In Proceedings of the fifteenth ACM international conference on web search and data mining. 937–946.
- [48] Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. 2024. Scaling llm testtime compute optimally can be more effective than scaling model parameters. arXiv preprint arXiv:2408.03314 (2024).
- [49] Fei Sun, Jun Liu, Jian Wu, Changhua Pei, Xiao Lin, Wenwu Ou, and Peng Jiang.

2019. BERT4Rec: Sequential recommendation with bidirectional encoder representations from transformer. In Proceedings of the 28th ACM international conference on information and knowledge management. 1441–1450.

- [50] Qiaoyu Tan, Jianwei Zhang, Jiangchao Yao, Ninghao Liu, Jingren Zhou, Hongxia Yang, and Xia Hu. 2021. Sparse-interest network for sequential recommendation. In Proceedings of the 14th ACM international conference on web search and data mining. 598–606.
- [51] Jiakai Tang, Sunhao Dai, Zexu Sun, Xu Chen, Jun Xu, Wenhui Yu, Lantao Hu, Peng Jiang, and Han Li. 2024. Towards Robust Recommendation via Decision Boundary-aware Graph Contrastive Learning. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 2854–2865.
- [52] Jiaxi Tang and Ke Wang. 2018. Personalized top-n sequential recommendation via convolutional sequence embedding. In Proceedings of the eleventh ACM international conference on web search and data mining. 565–573.
- [53] Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, Chenzhuang Du, Chonghua Liao, et al. 2025. Kimi k1. 5: Scaling reinforcement learning with llms. arXiv preprint arXiv:2501.12599 (2025).
- [54] Qwen Team. 2024. QwQ: Reflect Deeply on the Boundaries of the Unknown. https://qwenlm.github.io/blog/qwq-32b-preview/
- [55] Yonglong Tian, Chen Sun, Ben Poole, Dilip Krishnan, Cordelia Schmid, and Phillip Isola. 2020. What makes for good views for contrastive learning? Advances in neural information processing systems 33 (2020), 6827–6839.
- [56] Michael Tschannen, Josip Djolonga, Paul K Rubenstein, Sylvain Gelly, and Mario Lucic. 2019. On mutual information maximization for representation learning. arXiv preprint arXiv:1907.13625 (2019).
- [57] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. Advances in neural information processing systems 30 (2017).
- [58] Paul Viola and William M Wells III. 1997. Alignment by maximization of mutual information. International journal of computer vision 24, 2 (1997), 137–154.
- [59] Shoujin Wang, Liang Hu, Yan Wang, Longbing Cao, Quan Z Sheng, and Mehmet Orgun. 2019. Sequential recommender systems: challenges, progress and prospects. arXiv preprint arXiv:2001.04830 (2019).
- [60] Sinong Wang, Belinda Z Li, Madian Khabsa, Han Fang, and Hao Ma. 2020. Linformer: Self-attention with linear complexity. arXiv preprint arXiv:2006.04768

(2020).

- [61] Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. 2022. Self-consistency improves chain of thought reasoning in language models. arXiv preprint arXiv:2203.11171 (2022).

- [62] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. 2022. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems 35

(2022), 24824–24837.

- [63] Junkang Wu, Jiawei Chen, Jiancan Wu, Wentao Shi, Xiang Wang, and Xiangnan He. 2023. Understanding contrastive learning via distributionally robust optimization. Advances in Neural Information Processing Systems 36 (2023), 23297–23320.
- [64] Xu Xie, Fei Sun, Zhaoyang Liu, Shiwen Wu, Jinyang Gao, Jiandong Zhang, Bolin Ding, and Bin Cui. 2022. Contrastive learning for sequential recommendation. In 2022 IEEE 38th international conference on data engineering (ICDE). IEEE, 1259– 1273.
- [65] Fengli Xu, Qianyue Hao, Zefang Zong, Jingwei Wang, Yunke Zhang, Jingyi Wang, Xiaochong Lan, Jiahui Gong, Tianjian Ouyang, Fanjin Meng, et al. 2025. Towards Large Reasoning Models: A Survey of Reinforced Reasoning with Large Language Models. arXiv preprint arXiv:2501.09686 (2025).
- [66] Jian Xu, Sichun Luo, Xiangyu Chen, Haoming Huang, Hanxu Hou, and Linqi Song. 2025. RALLRec: Improving Retrieval Augmented Large Language Model Recommendation with Representation Learning. arXiv preprint arXiv:2502.06101

(2025).

- [67] Yige Xu, Xu Guo, Zhiwei Zeng, and Chunyan Miao. 2025. Softcot: Soft chainof-thought for efficient reasoning with llms. arXiv preprint arXiv:2502.12134

(2025).

- [68] An Yan, Shuo Cheng, Wang-Cheng Kang, Mengting Wan, and Julian McAuley.

2019. CosRec: 2D convolutional neural networks for sequential recommendation. In Proceedings of the 28th ACM international conference on information and knowledge management. 2173–2176.

- [69] Ling Yang, Zhaochen Yu, Tianjun Zhang, Shiyi Cao, Minkai Xu, Wentao Zhang, Joseph E Gonzalez, and Bin Cui. 2024. Buffer of thoughts: Thought-augmented reasoning with large language models. Advances in Neural Information Processing Systems 37 (2024), 113519–113544.
- [70] Yuhao Yang, Chao Huang, Lianghao Xia, Chunzhen Huang, Da Luo, and Kangyi Lin. 2023. Debiased contrastive learning for sequential recommendation. In Proceedings of the ACM web conference 2023. 1063–1073.
- [71] Yuhao Yang, Chao Huang, Lianghao Xia, Yuxuan Liang, Yanwei Yu, and Chenliang Li. 2022. Multi-behavior hypergraph-enhanced transformer for sequential recommendation. In Proceedings of the 28th ACM SIGKDD conference on knowledge discovery and data mining. 2263–2274.
- [72] Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. 2023. Tree of thoughts: Deliberate problem solving with large language models. Advances in neural information processing systems 36

(2023), 11809–11822.

- [73] Zhewei Yao, Reza Yazdani Aminabadi, Minjia Zhang, Xiaoxia Wu, Conglong Li, and Yuxiong He. 2022. Zeroquant: Efficient and affordable post-training quantization for large-scale transformers. Advances in Neural Information Processing Systems 35 (2022), 27168–27183.
- [74] Junliang Yu, Hongzhi Yin, Xin Xia, Tong Chen, Lizhen Cui, and Quoc Viet Hung Nguyen. 2022. Are graph augmentations necessary? simple graph contrastive learning for recommendation. In Proceedings of the 45th international ACM SIGIR conference on research and development in information retrieval. 1294–1303.
- [75] Fajie Yuan, Alexandros Karatzoglou, Ioannis Arapakis, Joemon M Jose, and Xiangnan He. 2019. A simple convolutional generative network for next item recommendation. In Proceedings of the twelfth ACM international conference on web search and data mining. 582–590.
- [76] Zheng Yuan, Fajie Yuan, Yu Song, Youhua Li, Junchen Fu, Fei Yang, Yunzhu Pan, and Yongxin Ni. 2023. Where to go next for recommender systems? idvs. modality-based recommender models revisited. In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval. 2639–2649.
- [77] Changshuo Zhang, Sirui Chen, Xiao Zhang, Sunhao Dai, Weijie Yu, and Jun Xu.

2024. UOEP: User-Oriented Exploration Policy for Enhancing Long-Term User Experiences in Recommender Systems. arXiv preprint arXiv:2401.09034 (2024).

- [78] Kepu Zhang, Teng Shi, Sunhao Dai, Xiao Zhang, Yinfeng Li, Jing Lu, Xiaoxue Zang, Yang Song, and Jun Xu. 2024. SAQRec: Aligning Recommender Systems to User Satisfaction via Questionnaire Feedback. In Proceedings of the 33rd ACM International Conference on Information and Knowledge Management. 3165–3175.
- [79] Qiyuan Zhang, Fuyuan Lyu, Zexu Sun, Lei Wang, Weixu Zhang, Zhihan Guo, Yufei Wang, Irwin King, Xue Liu, and Chen Ma. 2025. What, How, Where, and How Well? A Survey on Test-Time Scaling in Large Language Models. arXiv preprint arXiv:2503.24235 (2025).
- [80] Xiao Zhang, Sunhao Dai, Jun Xu, Zhenhua Dong, Quanyu Dai, and Ji-Rong Wen.

2022. Counteracting user attention bias in music streaming recommendation via reward modification. In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 2504–2514.

- [81] Guorui Zhou, Xiaoqiang Zhu, Chenru Song, Ying Fan, Han Zhu, Xiao Ma, Yanghui Yan, Junqi Jin, Han Li, and Kun Gai. 2018. Deep interest network for click-through rate prediction. In Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining. 1059–1068.

