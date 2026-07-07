arXiv:2505.00703v2[cs.CV]1Jul2025

# T2I-R1: Reinforcing Image Generation with Collaborative Semantic-level and Token-level CoT

Dongzhi Jiang∗1, Ziyu Guo∗2, Renrui Zhang∗†1, Zhuofan Zong1, Hao Li3 Le Zhuo1,3, Shilin Yan, Pheng-Ann Heng2, Hongsheng Li1

1CUHK MMLab 2CUHK MiuLar Lab 3Shanghai AI Laboratory {dzjiang, ziyuguo, renruizhang}@link.cuhk.edu.hk hsli@ee.cuhk.edu.hk

∗Equal Contribution †Project Leader

## Abstract

Recent advancements in large language models have demonstrated how chain-ofthought (CoT) and reinforcement learning (RL) can improve performance. However, applying such reasoning strategies to the visual generation domain remains largely unexplored. In this paper, we present T2I-R1, a novel reasoning-enhanced text-to-image generation model, powered by RL with a bi-level CoT reasoning process. Specifically, we identify two levels of CoT that can be utilized to enhance different stages of generation: (1) the semantic-level CoT for high-level planning of the prompt and (2) the token-level CoT for low-level pixel processing during patch-by-patch generation. To better coordinate these two levels of CoT, we introduce BiCoT-GRPO with an ensemble of generation rewards, which seamlessly optimizes both generated CoTs within the same training step. By applying our reasoning strategies to the baseline model, Janus-Pro, we achieve superior performance with 13% improvement on T2I-CompBench and 19% improvement on the WISE benchmark, even surpassing the state-of-the-art model FLUX.1. Code is available at: https://github.com/CaraJ7/T2I-R1.

## 1 Introduction

The emergence of advanced Large Language Models (LLMs) [50, 52, 69, 81], such as OpenAI o1 [53] and DeepSeek-R1 [17], has demonstrated considerable reasoning capabilities across domains including mathematics [1, 21, 46] and coding [6, 2, 24]. Through reinforcement learning (RL) [60, 61], these models analyze problems progressively with a comprehensive Chain-of-Thought (CoT) [72, 27, 19, 26, 86, 18] before providing answers, significantly enhancing output accuracy.

The CoT reasoning strategies have also been extended to the visual domain. Recent Large Multi-modal Models (LMMs) [5, 47, 84, 87] have adapted the paradigm to accommodate the visual understanding task [43, 86, 26]. These advanced LMMs can jointly process images and their associated textual queries, performing step-by-step analyses of visual details and integrating them with reasoning steps to derive final answers. Concurrently, CoT-like reasoning has been initially investigated in the visual generation task, particularly in autoregressive text-to-image generation. The pioneering work, ‘Image Generation with CoT’ [19], regards the progressive generation of the image tokens as a kind of CoT analogous to that of the text tokens, and proposes to optimize this intermediate process to enhance the image quality.

CoT in Image Understanding

[Figure 1]

Question

###### Reasoning Process

[Figure 2]

As shown in the figure, AC is the diameter of circle O, if angle OBC = 40 degrees, then the degree of angle AOB is A: 40°B: 50°C: 80°D: 100°

Step 1: In the circle, AC is a diameter and ∠OBC = 40°. Point O is the center.

Step 2: Triangle OBC is isosceles (OB and OC are radii), so∠BOC = 180°- 40°- 40°= 100°.

Step 3: Since AC is a diameter,∠AOB = 360° - 180°- 100°= 80°. The answer is C.

How to obtain the final answer?

Answer: C

CoT in Image Generation

[Figure 3]

Semantic-level CoT

Reasoning Process

[Figure 4]

The mouse is perched on a small branch with leaves sprouting from it, and the background appears to be a lush garden with vibrant flowers and foliage.

How does the whole image look like?

Image Prompt “A black cat and a brown mouse.”

A black cat with shiny fur and bright green eyes is sitting in front of a brown mouse with tiny paws and a twitching nose.

The cat gently gazes at the mouse, while the mouse playfully scurries around the branch.

ü Global Semantic ü High-level Planning

[Figure 5]

Reasoning Process

Token-level CoT

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

How does the next patch look like?

Image Prompt “A black cat and a brown mouse.”

ü Local Details ü Low-level Processing

- Figure 1: The Illustration of CoT in Image Understand and Generation Tasks. In the image understanding task, the CoT is the textual reasoning process. In the autoregressive visual generation task, we identify two levels of CoT: the semantic-level and token-level CoT. The semantic-level CoT is the high-level planning prior to the image generation, in the form of text. The token-level CoT is the intermediate patch-by-patch generation process, focusing on the local pixel details within a patch, in the form of image tokens.

Despite these advances, the exploration of CoT for image generation remains preliminary. Unlike image understanding, image generation requires the complex interpretation of cross-modal alignment and the synthesis of fine-grained visual details. To address these challenges, we identify two distinct levels of CoT reasoning that can be leveraged to enhance image generation, as illustrated in Fig. 1:

- • Semantic-level CoT is the textual reasoning about the image to generate, which is introduced prior to the image generation. The semantic-level CoT designs the global structure of the image, e.g., the appearance and location of objects. In case the prompt requires reasoning shown in Fig. 2, the semantic-level CoT also helps to deduce the objects to generate. Optimizing the semantic-level CoT could explicitly decouple the planning and reasoning of the prompt from the subsequent image tokens generation, making the generation easier.
- • Token-level CoT is the intermediate patch-by-patch generation process of the image, as originally introduced in [19]. This process could be viewed as a form of CoT as it outputs each subsequent token conditioned on all previous tokens within a discrete space, similar to the textual CoT. Unlike semantic-level CoT, token-level CoT focuses on low-level details like pixel generation and maintaining visual coherence between adjacent patches. Optimizing the token-level CoT can enhance both the generation quality and the alignment between the prompt and the resulting images.

Despite recognizing these two levels of CoT, a critical question remains unaddressed: How can we enhance and coordinate them for text-to-image generation? Current mainstream generative models [63, 67, 58, 28] are trained exclusively on generation targets, lacking the explicit textual understanding required for semantic-level CoT reasoning. Although introducing a separate model (e.g., an LLM) specifically for prompt interpretation [9] is technically feasible, this approach would significantly increase computational costs, complexity, and deployment challenges. Recently, a trend has arisen to merge visual understanding and generation within a single model. Building upon LMMs, these unified LMMs (ULMs) [74, 79, 88, 7] could not only understand the visual inputs but also generate images from text prompts. However, their two capabilities are still decoupled, typically pre-trained in two independent stages, with no clear evidence that the understanding capabilities can benefit generation. Given these potentials and issues, we start from a ULM and enhance it to unite both the semantic-level and token-level CoT into one framework for text-to-image generation.

###### T2I-R1 Janus-Pro

Prompt Semantic-level CoT Token-level CoT

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

The image depicts a beautiful and intricate igloo ... The igloo has an aerodynamic dome shape, showcasing.... Its surface appears smooth and translucent... The entrance to the igloo consists of a gently sloping tunnel...

A structure made of ice, traditionally built by the Inuit people

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

A specific type of flower cultivated in the country where Amsterdam is located

The image portrays tulips, the famous flower that is grown in the country where Amsterdam is located. The vibrant colors of these flowers, ranging from yellow and red to purple...

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

An image of a circular emblem representing balance and harmony in Taoist philosophy

A circular emblem depicting the balance and harmony of Taoist philosophy features the Yin and Yang symbol prominently in its center. The emblem showcases a harmonious arrangement...

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

Underneath the train, on the bottommost part of the platform, lies a tiny pig with a brown-yellowish coat. The pig is curled up, resting peacefully, surrounded by the iron structure of the train tracks and the dimly lit platform...

a pig on the bottom of a train

- Figure 2: Visualization of the Image Generation Process of T2I-R1. All the prompts need reasoning or contain an uncommon scenario. We observe that T2I-R1 successfully deduces the true intention behind the prompt or provides a sensible imagination (highlighted in the text) to produce a satisfying result compared with the baseline model, Janus-Pro.

To fulfill our target, we introduce BiCoT-GRPO, an RL method to jointly optimize the two levels of CoT for ULM. We opt for RL instead of supervised fine-tuning (SFT) for two reasons: First, the ULM has possessed the fundamental ability needed for the semantic-level and token-level CoT; our goal is only to elicit the fusion of these two abilities by guiding the model’s self-exploration. Second, RL methods have proven highly effective for enhancing reasoning capabilities, which are essential for both levels of CoT. Specifically, we first instruct the ULM to imagine and plan the image based on the prompt to obtain the semantic-level CoT. Then, we feed it into the ULM as the condition for the subsequent image generation for token-level CoT. We simultaneously generate multiple images from each prompt and then compute group-relative reward to optimize both levels of CoT within the same iteration. Unlike understanding tasks, where clearly defined rules for rewards exist, image generation lacks such standardized rules. Therefore, we propose to utilize an ensemble of diverse vision experts [77, 70, 41, 19] as reward models. This reward design serves two critical purposes: it evaluates generated images from multiple dimensions to ensure reliable quality assessment, while also functioning as a regularization method to prevent the ULM from hacking a single reward model.

Through the proposed reasoning strategies, we obtain T2I-R1, the first reasoning-enhanced text-toimage model combining the semantic-level and token-level CoT. Empirical results show that our approach outperforms baseline models by 13% and 19% improvements on the T2I-CompBench and WISE benchmark, and even surpasses the previous state-of-the-art model FLUX.1. Qualitative analysis reveals that our method empowers the model to generate more human-aligned results by reasoning about the true intentions behind the prompt and demonstrates enhanced robustness when dealing with uncommon scenarios.

Our contributions are summarized as follows:

- 1. We identify a dual-level reasoning process in the autoregressive image generation task by introducing the semantic-level and token-level CoT, which decouple high-level image planning from low-level pixel generation for more reliable generation.
- 2. We develop BiCoT-GRPO, a new reinforcement learning framework that jointly optimizes both levels of CoT reasoning, seamlessly integrating the understanding capabilities of ULMs for image generation. For reward modeling, we investigate a robust reward system utilizing an ensemble of vision experts.
- 3. Our resulting model, T2I-R1, incorporates both levels of CoT using BiCoT-GRPO and demonstrates significant quantitative and qualitative improvements, surpassing FLUX.1 across multiple established benchmarks.

## 2 Related Work

Unified Generation and Understanding LMM. Recently, the effort to unify image generation and understanding in a single LMM has attracted much attention. Building upon large language models (LLMs), it is natural for the LMMs to understand the image and output the text [51, 30, 90, 16, 85]. However, the method of how to generate an image from a LMM is still under exploration. The image generation method diverges into different branches. One line of the method relies on an exterior image generation model to complete generation [11, 66, 65, 34, 68, 13, 89, 29]. The generator often utilizes text-to-image diffusion models [58, 54] due to its powerful generation capability. To deliver the generation information, the LMM passes either the implicit conditional feature or the explicit image prompt to the generator. For example, EMU [66] first trains the LMM to output CLIP [56] image features identical to that input to the LMM. Then, a pretrained UNet [59] of Stable Diffusion [58] receives the output feature as the condition to generate an image. Another line of the method seeks to train the LMM to generate discrete tokens produced by VQGAN [12] to eliminate the need for an additional generator. [71, 32] directly adopts the VQGAN encoder as the image tokenizer for LMM. However, the VQGAN encoder is only pretrained on the image reconstruction task and thereby generates visual tokens less helpful for image understanding. To improve the understanding capability, [74, 7, 45, 39] proposes to tackle the understanding and generation tasks with different vision encoders separately. The CLIP encoder deals with image input for understanding, while the VQGAN encoder is responsible for generation. Moreover, some works [78, 55, 62] attempt to empower the vision encoder with both the understanding and the generation capability. VILAU [78] trains a vision encoder with both the contrastive loss [56] for text-image understanding and reconstruction loss [12] for image detail preserving. Thanks to the joint pretraining, the vision encoder could generate text-aligned discrete visual tokens. The LMM is then trained to receive the discrete tokens for image understanding and predict them for image generation.

Reinforcement Learning for Large Reasoning Models. The emergence of OpenAI o1 [53] has gained tremendous attention in developing the reasoning capability of large language models. Later, DeepSeek-R1 [17] proposes a rule-based reward and GRPO training method. The introduced method instructs the model to perform an extensive reasoning process before generating the final answer. The reward only focuses on the correctness of the final answer and the following of the pre-defined format. Recently, a number of works have applied this method to multi-modal large language models [5, 47, 82, 84, 10, 23] with task-specific rewards like correctness and IoU [42]. This training paradigm largely helps various reasoning-intensive tasks [57, 26, 18] like mathematical problem-solving [21, 46, 43, 86, 87] and code generation [6, 2, 24].

## 3 Method

##### 3.1 Preliminary

Recently, the employment of reinforcement learning has been the dominant approach to elicit the reasoning capability of the large models. [61] introduces GRPO, enhancing PPO by eliminating the value function and estimating the advantage in a group-relative manner. For a specific prompt-answer pair (p,a), a group of G individual responses {oi}Gi=1 is sampled from the old policy πθ

. Each response is then input to a reward function to obtain the individual reward Ri. Then, the advantage of the i-th response is calculated by normalizing the rewards {Ri}Gi=1 of the group:

old

Ai = Ri − mean({Ri}Gi=1)

. (1)

std({Ri}Gi=1)

GRPO adopts a clipped objective similar to PPO. Besides, a KL penalty term between the current policy πθ and the reference model πθ

is directly added in the loss function: JGRPO(θ) = E(q,a)∼D,{o

ref

i}Gi=1∼πθold(·|q)

|oi|

G

1

min ri,t(θ)Aˆi,clip ri,t(θ),1 − ε,1 + ε A ˆi − βDKL(πθ||πref) ,

G i=1 |oi|

t=1

i=1

###### Step 1

Group Relative Policy Optimization

Image Prompt Reasoning Instruction

Advantages

“a pink rabbit right of a green bowl with two paintings at background.”

“Provide a brief, precise visualization of all elements in the prompt…”

a1 a2 a … aG

3 a4

Group Computation

[Figure 38]

Unified LMM

Rewards

r1 r2 r … rG

3 r4

G for GRPO

Semantic-level CoT

###### Step 2

“A pink rabbit stands to the right of a green bowl, their colors creating a vivid contrast. ... Two background paintings add artistic flair with varied styles. ... The rabbit keeps slight distance from the bowl, balancing the composition. ...”

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

Image Prompt

“a pink rabbit right ...”

Output Reward Model

Human Preference Model

Object Detector

VQA Model

G for GRPO

[Figure 43]

Unified LMM

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

…

Token-level Decode CoT

<img_start>

G for GRPO

G

Figure 3: Framework of BiCoT-GRPO. In step 1, we instruct the model to generate the semanticlevel CoT based on the image prompt. In step 2, images are generated conditioned on both the image prompt and semantic-level CoT, with the intermediate generation process serving as token-level CoT. The resulting images are evaluated by an ensemble of vision experts to obtain rewards. We generate N images from each prompt to compute the group-relative reward and perform GRPO training.

where ri,j(θ) is the ratio between the probabilities of πθ and πθ

for outputting the current token:

old

πθ(oi,j | q,oi,<j) πθ

. (2)

ri,j(θ) =

(oi,t | q,oi,<j)

old

In text reasoning tasks like mathematical problem solving, the model is instructed to follow the pre-defined template to output the reasoning process and final answer. The reward functions are rule-based rewards that only check the correctness of the final answer and the output format.

##### 3.2 Semantic-level and Token-level CoT

In the autoregressive text generation tasks of LLMs and LMMs, CoT occurs in the textual reasoning format. However, in autoregressive image generation tasks, we identify two distinct types of CoT that could enhance the image generation at different abstraction levels:

Semantic-level CoT. Semantic-level CoT is defined as the textual reasoning that precedes image generation, serving as an overall semantic planning stage for the intended image. This process mirrors human artistic creation: when given a brief prompt, an artist first thinks about the scene construction, considering object attributes, spatial relationships, and interactions. In addition to the planning for common prompts, we also observe the semantic-level CoT benefits two other scenarios. If the prompt does not directly depict the object to generate, the semantic-level CoT can reason about the true intention from the user’s prompt, providing more aligned images. As illustrated in Fig. 2, the semantic-level CoT reasons that the flower cultivated in the country where Amsterdam is located is tulip. Without this semantic-level CoT, Janus-Pro fails to provide valid results. Additionally, the semantic-level CoT demonstrates importance when handling unusual or potentially ambiguous scenes. In the bottom example of Fig. 2, when given the prompt ‘A pig on the bottom of a train’, semanticlevel CoT introduces the action ‘lying’ for the pig, creating a more sensible scenario. In contrast, direct generation without this interpretive imagination creates significant confusion for Janus-Pro. Formally, each semantic-level CoT si is composed of |si| text tokens {si,1,si,2,...,si,|s

i|}.

Token-level CoT. Unique to the image generation task, a token-level step-by-step thinking exists in the image generation process. The generation of image tokens much resembles a chain of thought: the image tokens are generated patch by patch, where the current patch is generated based on the previous ones. We define the sequential generation of image tokens as token-level CoT. This process parallels how an artist progressively fills a canvas, with the generated patches forming a visual reasoning chain that maintains coherence across the image. This chain of patches is later reshaped to a 2D grid G ∈ Rh×w×c and input to an image decoder D to obtain the image. Unlike semantic-level CoT, which addresses global planning, token-level CoT focuses on local details and visual coherence across the image space. Formally, each token-level CoT ti consists of M image tokens {ti,1,ti,2,...,ti,M}, where M represents the resolution of the generated image, i.e., M = h × w.

##### 3.3 BiCoT-GRPO

GRPO has been proven to be highly effective for exploring the reasoning capability of the LLMs and LMMs. To accommodate both semantic-level and token-level CoT in image generation, we propose BiCoT-GRPO, where the model reasons twice in a single generation process. We instruct the model to first perform semantic-level CoT for global planning, and then dive into the local details by performing token-level CoT.

However, compared with the task of text generation, a great pipeline challenge is posed for incorporating two levels of CoT for image generation. Limited by the training paradigm, most current ULMs cannot generate interleaved images and text themselves. A manual signifier is often needed to instruct the model on which task to perform, either text generation or image generation. For Janus-Pro to generate an image, which is the ULM we use in this work, we need to manually concatenate an image start token (<img_start>) to explicitly instruct the model to start generating image tokens.

To tackle this problem, we propose a novel pipeline to facilitate ULM in generating images with two levels of CoT, as shown in Fig. 3. Specifically, our pipeline is composed of a two-step generation process. The first step is to generate the semantic-level CoT. We input the image prompt and instruct the model to imagine and reason about the details of the image to generate semantic-level CoT {si}Gi=1. The second stage focuses on the token-level CoT generation. We input the image prompt, the generated semantic-level CoT in the first stage, and the image start token to the ULM for generating image tokens {ti}Gi=1. Then, the image tokens are input to the image decoder to obtain the image I. Since there exist two types of CoT in our method, first the semantic-level CoT and then the token-level CoT. Each response oi is composed of two parts, namely oi = (si,ti). In this sense, the ri,j(θ) is converted to:

 

πθ(si,j|q,si,<j)

πθold(si,j|q,si,<j), 0 ≤ j ≤ |si|

πθ(oi,j | q,oi,<j) πθ

(3)

ri,j(θ) =

=



(oi,j | q,oi,<j)

πθ(ti,j|q,si,ti,<j)

πθold(ti,j|q,si,ti,<j), |si| < j ≤ |si| + M

old

Then, we update the ULM by maximizing Equation 3.1. In practice, we incorporate the token-level policy gradient loss in [83], where the loss term is normalized over all the generated tokens to balance the reward on overly long semantic-level CoT.

##### 3.4 Ensemble of Generation Rewards

Unlike DeepSeek-R1 with the rule-based reward, assessing the images based on pre-defined rules is infeasible. The assessment of the image includes various aspects, including the aesthetic appeal and objects’ existence, attributes, and relationships. Considering the complexity, we introduce an ensemble of vision experts to judge the generated image from multiple aspects. Meanwhile, the use of multiple reward functions also serves as a regularization method to prevent the ULM from hacking into a specific reward model. As shown in Fig. 4, the ensemble contains the following experts:

Human Preference Model. Human preference models (HPMs), such as HPS [77] and ImageReward [80], are trained to simulate human aesthetic preferences. These models are developed using datasets of human rankings on synthetic images, where annotators evaluate and compare generated outputs. During inference, these models assess both the aesthetic quality and prompt alignment of

Image

Human Preference Model

Prompt & Image

Reward: 0.27

- o Prompt Alignment?
- o Aesthetic Appeal?

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

- o Object Existence?
- o Spatial Relation?
- o Blue Butterfly?
- o Green Candle?
- o Prompt Alignment? Low ‘Yes’ Yes No Prob.

Object & Image

Reward: 1.00

Object Detector

Prompt “A blue butterfly on the left of a green candle.”

Object w/ Attribute

[Figure 54]

High ‘Yes’ Prob.

GPT-4o mini

Reward: 0.6

VQA Model

Yes No

& Image

Low ‘Yes’ Prob.

Object

Yes No

o Butterfly o Candle

[Figure 55]

Object w/ Attribute

Prompt & Image

Reward: 0.5

Output Reward Model

- o Blue Butterfly
- o Green Candle

- Figure 4: Illustration of the Ensemble of Generation Rewards. We use GPT-4o mini to extract the objects and their attributes before training. Each specialized reward model receives customized information inputs for the reward calculation. We take the average of all the rewards as final reward.

a generated image, producing a composite human preference score RHPM. This expert provides a holistic reward signal from a general perspective.

Object Detector. Another option of the reward model is an object detector, e.g., GroundingDINO [41] and YOLO-world [8]. These open-vocabulary detection models accept an image along with object queries as input and output both the spatial positions and confidence scores for detected objects. This kind of vision expert serves as an ideal tool to evaluate the object’s existence and relationship concerning space and numbers. For implementation, we extract all objects {obji}Ki=1 from the training image prompts, where K represents the total number of objects. We then query the object detector to identify these objects within the generated image. For each object, we assign a binary existence score (1 if detected, 0 otherwise) and average these scores across all objects in the prompt. If the prompt contains a spatial relationship, we further leverage the detected location to validate its correctness. We calculate the relative distance and intersection over union (IoU) between the objects for the spatial score Rspatial. If the number of the object nobj

is specifically pointed out in the prompt, we compare the number with the detected number of the object nˆobj

i

. The reward from the object detector RDet is determined as:

i

 

αRspatial + (1 − α)K1 Ki=1 I(obji detected), if spatial relationship in the prompt,

K i=1 I(nobj

), if number in the prompt,

1 n

RDet =

###### = nˆobj

i

i



K i=1 I(obji detected), else,

1 n

where Rspatial is 1 if the relative distance between the objects is larger than a threshold and the direction is right. If the direction is wrong, the reward is 0. Otherwise, we use the IoU as the spatial reward. We set α as 0.6 to encourage the correctness of the spatial relationship.

Visual Question Answering Model. The visual question answering (VQA) models are trained to answer questions based on the image input. The VQA models include earlier models prior to LLM, e.g., BLIP [33] and GIT [70], and LMMs like LLaVA [38]. We leverage these models to judge the existence and attributes of the objects. For example, if the image prompt is a red dog and a yellow cat, we first reformat each individual object with its attribute as a question to the VQA model, i.e., a red dog? and a yellow cat?. Then, we record the probability for the model to answer Yes as PYesi and No as PNoi . The reward for a prompt is calculated as: RVQA = K1 i P

i

Yes

###### PYesi +PNoi .

T2I-R1 (Semantic-level +Token-level)

T2I-R1 (Semantic-level +Token-level)

Only Semantic-level

Only Token-level

Only Semantic-level

Only Token-level

Janus-Pro

Janus-Pro

[Figure 56]

[Figure 57]

Show a plant that is a symbol of good fortune in Irish culture, and is known for its three-lobed leaves

A symbol of imperial China, a sprawling complex of palaces and temples in Beijing

[Figure 58]

[Figure 59]

Generate an image of a bird and a dog, with the smaller animal on top and the larger below

A bird grooming its feathers

[Figure 60]

[Figure 61]

The animal that emerges from a cocoon, symbolizing transformation

A typical dish from the country where Naples is located

[Figure 62]

[Figure 63]

A specific type of camera used in the 19th century for early photography

National Emblem of the country where New York is located

- Figure 5: Visualization Results. We provide the image generation results of the same prompt from four models: base model, the model with only semantic-level CoT optimized, the model with only token-level CoT optimized, and the model with both levels of CoT optimized.

Output Reward Model. Lastly, we also employ the output reward model (ORM) proposed in [19] as a reward model. The ORM is fine-tuned from an LMM (e.g., LLaVA-OneVision [30]) specifically for evaluating the alignment between the prompt and the image. The fine-tuning is to instruct the model to output Yes if the image perfectly aligns with the image and No otherwise. We calculate RORM using the methodology similar to RVQA, except that we input the whole image prompt to the ORM instead of reformatting the prompt.

We can choose one or multiple reward functions illustrated above, and take the average as the final reward for a specific sample. The detailed experiments of reward model in shown in Table 3.

## 4 Experiment

In this section, we first provide the main results of T2I-R1 in T2I-CompBench [22] and WISE [49] in Section 4.1. Then we present the results of different reward function combinations in Section 4.2 and the ablation study of the effectiveness of two levels of CoT in Section 4.3. Please refer to the Appendix A for more benchmark results (GenAI-Bench [36] and TIIF-Bench [73], detailed experiment setup, and more visualizations.

##### 4.1 Main Results

We compare T2I-R1 with leading text-to-image diffusion and autoregressive models on the T2ICompBench and WISE benchmarks (in Table 1 and 2). We also provide the qualitative results in Fig. 5. Our method demonstrates substantial improvements over the baseline model, with average enhancements of 13% and 19% on T2I-CompBench and WISE, respectively. On T2I-CompBench, the most significant gains appear in attribute binding, with an average improvement of 19%. For the WISE benchmark, improvements are more evenly distributed across categories. When compared to the more powerful state-of-the-art diffusion models, T2I-R1 achieves superior or comparable results across both benchmarks. Notably, on T2I-CompBench, our method leads in five of six subtasks, with an exceptional performance in the spatial subtask (0.3378), surpassing previous SOTA results by over 5%. Similarly, for WISE, T2I-R1 excels in four of seven subtasks and achieves the highest overall score of 0.54, outperforming the robust FLUX.1-dev by 4%. Remarkably, our approach consistently achieves the leading results across all subtasks in both benchmarks when compared to other autoregressive models. Remarkably, the improvement on T2I-Compbench benefits from the planning ability brought by the semantic-level CoT, which designs the complex scenarios before generation. While the enhancement of WISE is due to the reasoning capability from the semantic-level CoT, which deduces the true object or place depicted behind the prompt.

Table 1: T2I-CompBench Result. The best score is in blue , with the second-best score in green .

Attribute Binding Object Relationship

Model

Complex↑ Color ↑ Shape↑ Texture↑ Spatial↑ Non-Spatial↑

Diffusion Models

StructureDiffusion [14] 0.4990 0.4218 0.4900 0.1386 0.3111 0.3355 Composable Diffusion [40] 0.4063 0.3299 0.3645 0.0800 0.2980 0.2898 Attend-and-Excite [3] 0.6400 0.4517 0.5963 0.1455 0.3109 0.3401 PixArt-α [4] 0.6690 0.4927 0.6477 0.2064 0.3197 0.3433 CoMat [25] 0.7827 0.5329 0.6468 0.2428 0.3187 0.3680 SD-v1.5 [58] 0.3758 0.3713 0.4186 0.1165 0.3112 0.3047 SD-XL-base-1.0 [54] 0.5879 0.4687 0.5299 0.2131 0.3119 0.3237 FLUX.1 [28] 0.7407 0.5718 0.6922 0.2863 0.3127 0.3703

AutoRegressive Models Show-o [79] 0.56 0.41 0.46 0.20 0.30 0.29 Show-o + PARM [19] 0.75 0.56 0.66 0.29 0.31 0.37 EMU3 [71] 0.7544 0.5706 0.7164 - - Janus-Pro-7B (Baseline) [7] 0.6359 0.3528 0.4936 0.2061 0.3085 0.3559 T2I-R1 (Ours) 0.8130 0.5852 0.7243 0.3378 0.3090 0.3993

Table 2: WISE Result. The best score is in blue , with the second-best score in green .

Spatio-Temporal Natural Science

Model Cultural↑

Overall Time↑ Space↑ Biology ↑ Physics↑ Chemistry↑

Diffusion Models

PixArt-Alpha [4] 0.45 0.50 0.48 0.49 0.56 0.34 0.47 playground-v2.5 [31] 0.49 0.58 0.55 0.43 0.48 0.33 0.49 SD-v1-5 [58] 0.34 0.35 0.32 0.28 0.29 0.21 0.32 SD-XL-base-0.9 [54] 0.43 0.48 0.47 0.44 0.45 0.27 0.43 FLUX.1-dev [28] 0.48 0.58 0.62 0.42 0.51 0.35 0.50

AutoRegressive Models

Emu3 [71] 0.34 0.45 0.48 0.41 0.45 0.27 0.39 Show-o [79] 0.28 0.40 0.48 0.30 0.46 0.30 0.35 VILA-U [78] 0.26 0.33 0.37 0.35 0.39 0.23 0.31 Janus-1.3B [74] 0.16 0.26 0.35 0.28 0.30 0.14 0.23 Janus-Pro-7B (Baseline) [7] 0.30 0.37 0.49 0.36 0.42 0.26 0.35 T2I-R1 (Ours) 0.56 0.55 0.63 0.54 0.55 0.30 0.54

##### 4.2 Reward Analysis

In this section, we experiment with the choice of reward functions and their combinations. We hope to provide some insights into how to choose the reward functions and combine them. Our results are shown in Table 3. We first experiment with the individual reward model. HPM (H) demonstrates superior performance in attribute binding but shows limited effectiveness in object relationships, likely due to its weak relation comprehension capabilities. The object detector (D) yields the least improvement in attribute binding, which aligns with expectations since our detector-based reward functions do not explicitly evaluate attributes. The improvements observed stem solely from enhanced object existence ratios in the prompts. We observe that VQA model (V) and ORM (O) are both effective reward models with distinct strengths: the VQA model excels at improving attribute binding, while ORM demonstrates superior performance in relationships. Then we experiment with multiple reward models. We start from the composition of HPM and object detector (H + D), and progressively incorporate other reward models. Our findings indicate that both the HPM-object detector combination (H + D) and the three-model integration of HPM, object detector, and VQA (H + D + V) deliver balanced and satisfactory results in both attribute and relationship tasks. To obtain the optimal choice of reward models, we conduct a human study to evaluate the visual quality,

##### Table 3: T2I-CompBench Results with Different Reward Models. ‘Det’ stands for object detector.

Reward Model Attribute Binding Object Relationship

Model

Complex↑ Visual Quality↑ HPM Det VQA ORM Color ↑ Shape↑ Texture↑ Spatial↑ Non-Spatial↑

Janus-Pro-7B - - - - 0.6359 0.3528 0.4936 0.2061 0.3085 0.3559 -

- - ✓ - - - 0.8134 0.6048 0.7311 0.2383 0.3012 0.3899 -
- - - ✓ - - 0.7422 0.5140 0.6494 0.3044 0.3100 0.3872 -
- - - - ✓ - 0.8171 0.6019 0.7307 0.2969 0.3088 0.4052 0.218
- - - - - ✓ 0.7819 0.5638 0.7010 0.3301 0.3103 0.3959 1.775
- - ✓ ✓ - - 0.8210 0.6074 0.7440 0.3189 0.3076 0.4005 1.942 T2I-R1 ✓ ✓ ✓ - 0.8130 0.5852 0.7243 0.3378 0.3090 0.3993 2.063
- - ✓ ✓ ✓ ✓ 0.7599 0.5742 0.6902 0.2796 0.3070 0.3921 Table 4: Ablation Experiments on the Effectiveness of the Two Levels of CoT.

Optimized CoT T2I-CompBench WISE

Model

Diversity↑ Semantic-level Token-level Color↑ Shape↑ Texture↑ Culture↑ Spatio-Temporal↑ Science↑

Janus-Pro-7B 0.6359 0.3528 0.4936 0.3000 0.4232 0.3467 6.976

- - ✓ 0.8082 0.5684 0.7219 0.4900 0.5599 0.4367 8.177
- - ✓ 0.7752 0.5849 0.7451 0.3500 0.4732 0.3900 6.255 T2I-R1 ✓ ✓ 0.8130 0.5852 0.7243 0.5600 0.5855 0.4633 8.203

detailed in Appendix B.3. We adopt the combination of the highest visual quality, the ensemble of three reward models (H + D + V) for our final model.

Token-level CoT Only Semantic-level + Token-level CoT

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

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

[Figure 81]

[Figure 82]

[Figure 83]

a black squirrel and a brown nut

[Figure 84]

[Figure 85]

- Figure 6: Visualization Result of the Image Diversity of a Single Prompt. We showcase the result of only token-level CoT optimized and both semantic-level and token-level CoT optimized.

##### 4.3 Ablation Study

We validate the effectiveness of incorporating both semantic-level and token-level CoT. We first show the advantage of semantic-level CoT by comparing T2I-R1 with a baseline method that generates images using only token-level CoT optimized with GRPO. We witness a consistent gain on the benchmarks. However, we also find that training solely with token-level CoT substantially reduces image diversity, as demonstrated in Fig. 6 and 7. To quantify this, we provide the Vendi Score [15] as the diversity metric. We find that the diversity largely increases with semantic-level CoT while decreases without it. We then consider another setting to only optimize the semantic-level CoT to show the effectiveness of token-level CoT. The second row of Table 4 shows that optimizing semanticlevel CoT exclusively yields smaller improvements compared to joint optimization. Additionally, optimizing both CoT types produces images with better aesthetic quality compared to optimizing semantic-level CoT only, as shown in Fig. 5. More details are in Appendix B.2.

a fabric bag and a glass vase

[Figure 86]

[Figure 87]

a key on the right of a dog

## 5 Conclusion

In this paper, we introduce T2I-R1, the first reasoning-enhanced text-to-image model powered by a bi-level CoT reasoning process. We identify both the semantic-level CoT for high-level planning and the token-level CoT for patch-by-patch generation. We further integrate them through our proposed BiCoT-GRPO, a reinforcement learning framework incorporating two levels of CoT within the same training step. By leveraging a ULM capable of both visual understanding and generation, our approach eliminates the need for separate specialized models while achieving significant performance improvements, +13% on T2I-CompBench and +19% on the WISE benchmark, surpassing even

FLUX.1. Our qualitative analysis demonstrates that T2I-R1 better understands complex prompts, reasons about user intentions, and handles uncommon scenarios with greater robustness, establishing a new paradigm for reasoning-centric generative systems.

## References

- [1] Amini, A., Gabriel, S., Lin, P., Koncel-Kedziorski, R., Choi, Y., Hajishirzi, H.: Mathqa: Towards interpretable math word problem solving with operation-based formalisms. arXiv preprint arXiv:1905.13319 (2019)
- [2] Austin, J., Odena, A., Nye, M.I., Bosma, M., Michalewski, H., Dohan, D., Jiang, E., Cai, C.J., Terry, M., Le, Q.V., Sutton, C.: Program synthesis with large language models. CoRR abs/2108.07732 (2021)
- [3] Chefer, H., Alaluf, Y., Vinker, Y., Wolf, L., Cohen-Or, D.: Attend-and-excite: Attention-based semantic guidance for text-to-image diffusion models. ACM Transactions on Graphics (TOG) 42(4), 1–10 (2023)
- [4] Chen, J., Yu, J., Ge, C., Yao, L., Xie, E., Wu, Y., Wang, Z., Kwok, J., Luo, P., Lu, H., Li, Z.: Pixart-α: Fast training of diffusion transformer for photorealistic text-to-image synthesis (2023)
- [5] Chen, L., Li, L., Zhao, H., Song, Y., Vinci: R1-v: Reinforcing super generalization ability in vision-language models with less than $3. https://github.com/Deep-Agent/R1-V (2025), accessed: 2025-02-02
- [6] Chen, M., Tworek, J., Jun, H., Yuan, Q., de Oliveira Pinto, H.P., Kaplan, J., Edwards, H., Burda, Y., Joseph, N., Brockman, G., Ray, A., Puri, R., Krueger, G., Petrov, M., Khlaaf, H., Sastry, G., Mishkin, P., Chan, B., Gray, S., Ryder, N., Pavlov, M., Power, A., Kaiser, L., Bavarian, M., Winter, C., Tillet, P., Such, F.P., Cummings, D., Plappert, M., Chantzis, F., Barnes, E., Herbert-Voss, A., Guss, W.H., Nichol, A., Paino, A., Tezak, N., Tang, J., Babuschkin, I., Balaji, S., Jain, S., Saunders, W., Hesse, C., Carr, A.N., Leike, J., Achiam, J., Misra, V., Morikawa, E., Radford, A., Knight, M., Brundage, M., Murati, M., Mayer, K., Welinder, P., McGrew, B., Amodei, D., McCandlish, S., Sutskever, I., Zaremba, W.: Evaluating large language models trained on code. CoRR abs/2107.03374 (2021)
- [7] Chen, X., Wu, Z., Liu, X., Pan, Z., Liu, W., Xie, Z., Yu, X., Ruan, C.: Janus-pro: Unified multimodal understanding and generation with data and model scaling. arXiv preprint arXiv:2501.17811 (2025)
- [8] Cheng, T., Song, L., Ge, Y., Liu, W., Wang, X., Shan, Y.: Yolo-world: Real-time openvocabulary object detection. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 16901–16911 (2024)
- [9] Datta, S., Ku, A., Ramachandran, D., Anderson, P.: Prompt expansion for adaptive textto-image generation. In: Ku, L.W., Martins, A., Srikumar, V. (eds.) Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). pp. 3449–3476. Association for Computational Linguistics, Bangkok, Thailand (Aug 2024). https://doi.org/10.18653/v1/2024.acl-long.189, https://aclanthology.org/2024. acl-long.189/
- [10] Deng, Y., Bansal, H., Yin, F., Peng, N., Wang, W., Chang, K.W.: Openvlthinker: An early exploration to complex vision-language reasoning via iterative self-improvement (2025), https: //arxiv.org/abs/2503.17352
- [11] Dong, R., Han, C., Peng, Y., Qi, Z., Ge, Z., Yang, J., Zhao, L., Sun, J., Zhou, H., Wei, H., et al.: Dreamllm: Synergistic multimodal comprehension and creation. arXiv preprint arXiv:2309.11499 (2023)
- [12] Esser, P., Rombach, R., Ommer, B.: Taming transformers for high-resolution image synthesis. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 12873–12883 (2021)

- [13] Fang, R., Duan, C., Wang, K., Li, H., Tian, H., Zeng, X., Zhao, R., Dai, J., Li, H., Liu, X.: Puma: Empowering unified mllm with multi-granular visual generation. arXiv preprint arXiv:2410.13861 (2024)
- [14] Feng, W., He, X., Fu, T.J., Jampani, V., Akula, A., Narayana, P., Basu, S., Wang, X.E., Wang, W.Y.: Training-free structured diffusion guidance for compositional text-to-image synthesis. arXiv preprint arXiv:2212.05032 (2022)
- [15] Friedman, D., Dieng, A.B.: The vendi score: A diversity evaluation metric for machine learning. arXiv preprint arXiv:2210.02410 (2022)
- [16] Gemini Team, G.: Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805 (2023)
- [17] Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., Zhu, Q., Ma, S., Wang, P., Bi, X., et al.: Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948 (2025)
- [18] Guo, Z., Zhang, R., Chen, H., Gao, J., Jiang, D., Wang, J., Heng, P.A.: Sciverse: Unveiling the knowledge comprehension and visual reasoning of lmms on multi-modal scientific problems. arXiv preprint arXiv:2503.10627 (2025)
- [19] Guo, Z., Zhang, R., Tong, C., Zhao, Z., Gao, P., Li, H., Heng, P.A.: Can we generate images with cot? let’s verify and reinforce image generation step by step. arXiv preprint arXiv:2501.13926

(2025)

- [20] Han, J., Liu, J., Jiang, Y., Yan, B., Zhang, Y., Yuan, Z., Peng, B., Liu, X.: Infinity: Scaling bitwise autoregressive modeling for high-resolution image synthesis (2024)
- [21] Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart, S., Tang, E., Song, D., Steinhardt, J.: Measuring mathematical problem solving with the math dataset. NeurIPS (2021)
- [22] Huang, K., Sun, K., Xie, E., Li, Z., Liu, X.: T2i-compbench: A comprehensive benchmark for open-world compositional text-to-image generation. Advances in Neural Information Processing Systems 36, 78723–78747 (2023)
- [23] Huang, W., Jia, B., Zhai, Z., Cao, S., Ye, Z., Zhao, F., Hu, Y., Lin, S.: Vision-r1: Incentivizing reasoning capability in multimodal large language models. arXiv preprint arXiv:2503.06749

(2025)

- [24] Jain, N., Han, K., Gu, A., Li, W., Yan, F., Zhang, T., Wang, S., Solar-Lezama, A., Sen, K., Stoica, I.: Livecodebench: Holistic and contamination free evaluation of large language models for code. CoRR abs/2403.07974 (2024), https://doi.org/10.48550/arXiv.2403.07974
- [25] Jiang, D., Song, G., Wu, X., Zhang, R., Shen, D., Zong, Z., Liu, Y., Li, H.: Comat: Aligning textto-image diffusion model with image-to-text concept matching. arXiv preprint arXiv:2404.03653

(2024)

- [26] Jiang, D., Zhang, R., Guo, Z., Li, Y., Qi, Y., Chen, X., Wang, L., Jin, J., Guo, C., Yan, S., et al.: Mme-cot: Benchmarking chain-of-thought in large multimodal models for reasoning quality, robustness, and efficiency. arXiv preprint arXiv:2502.09621 (2025)
- [27] Kojima, T., Gu, S.S., Reid, M., Matsuo, Y., Iwasawa, Y.: Large language models are zero-shot reasoners. Advances in neural information processing systems 35, 22199–22213 (2022)
- [28] Labs, B.F.: Flux. https://github.com/black-forest-labs/flux (2024)
- [29] Lei, J., Zhang, R., Hu, X., Lin, W., Li, Z., Sun, W., Du, R., Zhuo, L., Li, Z., Li, X., et al.: Imagine-e: Image generation intelligence evaluation of state-of-the-art text-to-image models. arXiv preprint arXiv:2501.13920 (2025)
- [30] Li, B., Zhang, Y., Guo, D., Zhang, R., Li, F., Zhang, H., Zhang, K., Li, Y., Liu, Z., Li, C.: Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326 (2024)

- [31] Li, D., Kamko, A., Akhgari, E., Sabet, A., Xu, L., Doshi, S.: Playground v2. 5: Three insights towards enhancing aesthetic quality in text-to-image generation. arXiv preprint arXiv:2402.17245

(2024)

- [32] Li, H., Tian, C., Shao, J., Zhu, X., Wang, Z., Zhu, J., Dou, W., Wang, X., Li, H., Lu, L., et al.: Synergen-vl: Towards synergistic image understanding and generation with vision experts and token folding. arXiv preprint arXiv:2412.09604 (2024)
- [33] Li, J., Li, D., Xiong, C., Hoi, S.: Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In: International Conference on Machine Learning. pp. 12888–12900. PMLR (2022)
- [34] Li, Y., Zhang, Y., Wang, C., Zhong, Z., Chen, Y., Chu, R., Liu, S., Jia, J.: Mini-gemini: Mining the potential of multi-modality vision language models. arXiv: 2403.18814 (2024)
- [35] Liao, C., Liu, L., Wang, X., Luo, Z., Zhang, X., Zhao, W., Wu, J., Li, L., Tian, Z., Huang, W.: Mogao: An omni foundation model for interleaved multi-modal generation. arXiv preprint arXiv:2505.05472 (2025)
- [36] Lin, Z., Pathak, D., Li, B., Li, J., Xia, X., Neubig, G., Zhang, P., Ramanan, D.: Evaluating text-to-visual generation with image-to-text generation. In: European Conference on Computer Vision. pp. 366–384. Springer (2024)
- [37] Liu, H., Yan, W., Zaharia, M., Abbeel, P.: World model on million-length video and language with ringattention. arXiv e-prints pp. arXiv–2402 (2024)
- [38] Liu, H., Li, C., Wu, Q., Lee, Y.J.: Visual instruction tuning. In: NeurIPS (2023)
- [39] Liu, J., Chen, H., An, P., Liu, Z., Zhang, R., Gu, C., Li, X., Guo, Z., Chen, S., Liu, M., et al.: Hybridvla: Collaborative diffusion and autoregression in a unified vision-language-action model. arXiv preprint arXiv:2503.10631 (2025)
- [40] Liu, N., Li, S., Du, Y., Torralba, A., Tenenbaum, J.B.: Compositional visual generation with composable diffusion models. In: European Conference on Computer Vision. pp. 423–439. Springer (2022)
- [41] Liu, S., Zeng, Z., Ren, T., Li, F., Zhang, H., Yang, J., yue Li, C., Yang, J., Su, H., Zhu, J.J., Zhang, L.: Grounding dino: Marrying dino with grounded pre-training for open-set object detection. ArXiv abs/2303.05499 (2023)
- [42] Liu, Y., Peng, B., Zhong, Z., Yue, Z., Lu, F., Yu, B., Jia, J.: Seg-zero: Reasoning-chain guided segmentation via cognitive reinforcement. arXiv preprint arXiv:2503.06520 (2025)
- [43] Lu, P., Bansal, H., Xia, T., Liu, J., yue Li, C., Hajishirzi, H., Cheng, H., Chang, K.W., Galley, M., Gao, J.: Mathvista: Evaluating math reasoning in visual contexts with gpt-4v, bard, and other large multimodal models. ArXiv abs/2310.02255 (2023)
- [44] Ma, C., Jiang, Y., Wu, J., Yang, J., Yu, X., Yuan, Z., Peng, B., Qi, X.: Unitok: A unified tokenizer for visual generation and understanding. arXiv preprint arXiv:2502.20321 (2025)
- [45] Ma, Y., Liu, X., Chen, X., Liu, W., Wu, C., Wu, Z., Pan, Z., Xie, Z., Zhang, H., Zhao, L., et al.: Janusflow: Harmonizing autoregression and rectified flow for unified multimodal understanding and generation. arXiv preprint arXiv:2411.07975 (2024)
- [46] MAA: American invitational mathematics examination - aime. In: American Invitational Mathematics Examination - AIME 2024 (February 2024), https://maa.org/ math-competitions/american-invitational-mathematics-examination-aime
- [47] Meng, F., Du, L., Liu, Z., Zhou, Z., Lu, Q., Fu, D., Shi, B., Wang, W., He, J., Zhang, K., et al.: Mm-eureka: Exploring visual aha moment with rule-based large-scale reinforcement learning. arXiv preprint arXiv:2503.07365 (2025)
- [48] Midjourney: Midjourney v6.1. https://www.midjourney.com/ (2024)

- [49] Niu, Y., Ning, M., Zheng, M., Lin, B., Jin, P., Liao, J., Ning, K., Zhu, B., Yuan, L.: Wise: A world knowledge-informed semantic evaluation for text-to-image generation. arXiv preprint arXiv:2503.07265 (2025)
- [50] OpenAI: Chatgpt. https://chat.openai.com (2023)
- [51] OpenAI: GPT-4V(ision) system card (2023), https://openai.com/research/ gpt-4v-system-card
- [52] OpenAI: Hello gpt-4o. https://openai.com/index/hello-gpt-4o/ (2024)
- [53] OpenAI: Introducing openai o1, 2024. (2024), https://openai.com/o1/
- [54] Podell, D., English, Z., Lacey, K., Blattmann, A., Dockhorn, T., Müller, J., Penna, J., Rombach, R.: Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952 (2023)
- [55] Qu, L., Zhang, H., Liu, Y., Wang, X., Jiang, Y., Gao, Y., Ye, H., Du, D.K., Yuan, Z., Wu, X.: Tokenflow: Unified image tokenizer for multimodal understanding and generation. arXiv preprint arXiv:2412.03069 (2024)
- [56] Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., Sutskever, I.: Learning transferable visual models from natural language supervision. In: International Conference on Machine Learning (2021), https://api.semanticscholar.org/CorpusID:231591445
- [57] Rein, D., Hou, B.L., Stickland, A.C., Petty, J., Pang, R.Y., Dirani, J., Michael, J., Bowman, S.R.: GPQA: A graduate-level google-proof q&a benchmark. arXiv preprint arXiv:2311.12022

(2023)

- [58] Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent diffusion models. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 10684–10695 (2022)
- [59] Ronneberger, O., Fischer, P., Brox, T.: U-net: Convolutional networks for biomedical image segmentation. In: Medical image computing and computer-assisted intervention–MICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18. pp. 234–241. Springer (2015)
- [60] Schulman, J., Wolski, F., Dhariwal, P., Radford, A., Klimov, O.: Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347 (2017)
- [61] Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Zhang, M., Li, Y., Wu, Y., Guo, D.: Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300 (2024)
- [62] Song, W., Wang, Y., Song, Z., Li, Y., Sun, H., Chen, W., Zhou, Z., Xu, J., Wang, J., Yu, K.: Dualtoken: Towards unifying visual understanding and generation with dual visual vocabularies. arXiv preprint arXiv:2503.14324 (2025)
- [63] Sun, P., Jiang, Y., Chen, S., Zhang, S., Peng, B., Luo, P., Yuan, Z.: Autoregressive model beats diffusion: Llama for scalable image generation. arXiv preprint arXiv:2406.06525 (2024)
- [64] Sun, P., Jiang, Y., Chen, S., Zhang, S., Peng, B., Luo, P., Yuan, Z.: Autoregressive Model Beats Diffusion: Llama for Scalable Image Generation (Jun 2024). https://doi.org/10.48550/arXiv.2406.06525
- [65] Sun, Q., Cui, Y., Zhang, X., Zhang, F., Yu, Q., Luo, Z., Wang, Y., Rao, Y., Liu, J., Huang, T., Wang, X.: Generative multimodal models are in-context learners. arXiv: 2312.13286 (2023)
- [66] Sun, Q., Yu, Q., Cui, Y., Zhang, F., Zhang, X., Wang, Y., Gao, H., Liu, J., Huang, T., Wang, X.: Generative pretraining in multimodality. arXiv: 2307.05222 (2023)

- [67] Tian, K., Jiang, Y., Yuan, Z., Peng, B., Wang, L.: Visual autoregressive modeling: Scalable image generation via next-scale prediction. Advances in neural information processing systems 37, 84839–84865 (2024)
- [68] Tong, S., Fan, D., Zhu, J., Xiong, Y., Chen, X., Sinha, K., Rabbat, M., LeCun, Y., Xie, S., Liu, Z.: Metamorph: Multimodal understanding and generation via instruction tuning. arXiv preprint arXiv:2412.14164 (2024)
- [69] Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.A., Lacroix, T., Rozière, B., Goyal, N., Hambro, E., Azhar, F., et al.: Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971 (2023)
- [70] Wang, J., Yang, Z., Hu, X., Li, L., Lin, K., Gan, Z., Liu, Z., Liu, C., Wang, L.: Git: A generative image-to-text transformer for vision and language. arXiv preprint arXiv:2205.14100 (2022)
- [71] Wang, X., Zhang, X., Luo, Z., Sun, Q., Cui, Y., Wang, J., Zhang, F., Wang, Y., Li, Z., Yu, Q., et al.: Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869 (2024)
- [72] Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q.V., Zhou, D., et al.: Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems 35, 24824–24837 (2022)
- [73] Wei, X., Zhang, J., Wang, Z., Wei, H., Guo, Z., Zhang, L.: Tiif-bench: How does your t2i model follow your instructions? arXiv preprint arXiv:2506.02161 (2025)
- [74] Wu, C., Chen, X., Wu, Z., Ma, Y., Liu, X., Pan, Z., Liu, W., Xie, Z., Yu, X., Ruan, C., et al.: Janus: Decoupling visual encoding for unified multimodal understanding and generation. arXiv preprint arXiv:2410.13848 (2024)
- [75] Wu, J., Jiang, Y., Ma, C., Liu, Y., Zhao, H., Yuan, Z., Bai, S., Bai, X.: Liquid: Language models are scalable multi-modal generators. arXiv preprint arXiv:2412.04332 (2024)
- [76] Wu, X., Bai, Y., Zheng, H., Chen, H.H., Liu, Y., Wang, Z., Ma, X., Shu, W.J., Wu, X., Yang, H., Lim, S.N.: LightGen: Efficient Image Generation through Knowledge Distillation and Direct Preference Optimization (Mar 2025). https://doi.org/10.48550/arXiv.2503.08619
- [77] Wu, X., Hao, Y., Sun, K., Chen, Y., Zhu, F., Zhao, R., Li, H.: Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis. arXiv preprint arXiv:2306.09341 (2023)
- [78] Wu, Y., Zhang, Z., Chen, J., Tang, H., Li, D., Fang, Y., Zhu, L., Xie, E., Yin, H., Yi, L., et al.: Vila-u: a unified foundation model integrating visual understanding and generation. arXiv preprint arXiv:2409.04429 (2024)
- [79] Xie, J., Mao, W., Bai, Z., Zhang, D.J., Wang, W., Lin, K.Q., Gu, Y., Chen, Z., Yang, Z., Shou, M.Z.: Show-o: One single transformer to unify multimodal understanding and generation. arXiv preprint arXiv:2408.12528 (2024)
- [80] Xu, J., Liu, X., Wu, Y., Tong, Y., Li, Q., Ding, M., Tang, J., Dong, Y.: Imagereward: learning and evaluating human preferences for text-to-image generation. In: Proceedings of the 37th International Conference on Neural Information Processing Systems. pp. 15903–15935 (2023)
- [81] Yang, A., Yang, B., Hui, B., Zheng, B., Yu, B., Zhou, C., Li, C., Li, C., Liu, D., Huang, F., Dong, G., Wei, H., Lin, H., Tang, J., Wang, J., Yang, J., Tu, J., Zhang, J., Ma, J., Xu, J., Zhou, J., Bai, J., He, J., Lin, J., Dang, K., Lu, K., Chen, K., Yang, K., Li, M., Xue, M., Ni, N., Zhang, P., Wang, P., Peng, R., Men, R., Gao, R., Lin, R., Wang, S., Bai, S., Tan, S., Zhu, T., Li, T., Liu, T., Ge, W., Deng, X., Zhou, X., Ren, X., Zhang, X., Wei, X., Ren, X., Fan, Y., Yao, Y., Zhang, Y., Wan, Y., Chu, Y., Liu, Y., Cui, Z., Zhang, Z., Fan, Z.: Qwen2 technical report. arXiv preprint arXiv:2407.10671 (2024)
- [82] Yang, Y., He, X., Pan, H., Jiang, X., Deng, Y., Yang, X., Lu, H., Yin, D., Rao, F., Zhu, M., et al.: R1-onevision: Advancing generalized multimodal reasoning through cross-modal formalization. arXiv preprint arXiv:2503.10615 (2025)

- [83] Yu, Q., Zhang, Z., Zhu, R., Yuan, Y., Zuo, X., Yue, Y., Fan, T., Liu, G., Liu, L., Liu, X., et al.: Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476 (2025)
- [84] Zhang, J., Huang, J., Yao, H., Liu, S., Zhang, X., Lu, S., Tao, D.: R1-vl: Learning to reason with multimodal large language models via step-wise group relative policy optimization. arXiv preprint arXiv:2503.12937 (2025)
- [85] Zhang, R., Han, J., Liu, C., Zhou, A., Lu, P., Qiao, Y., Li, H., Gao, P.: Llama-adapter: Efficient fine-tuning of large language models with zero-initialized attention. In: ICLR 2024 (2024)
- [86] Zhang, R., Jiang, D., Zhang, Y., Lin, H., Guo, Z., Qiu, P., Zhou, A., Lu, P., Chang, K.W., Gao, P., et al.: Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? ECCV 2024 (2024)
- [87] Zhang, R., Wei, X., Jiang, D., Zhang, Y., Guo, Z., Tong, C., Liu, J., Zhou, A., Wei, B., Zhang, S., et al.: Mavis: Mathematical visual instruction tuning. arXiv preprint arXiv:2407.08739 (2024)
- [88] Zhou, C., Yu, L., Babu, A., Tirumala, K., Yasunaga, M., Shamis, L., Kahn, J., Ma, X., Zettlemoyer, L., Levy, O.: Transfusion: Predict the next token and diffuse images with one multi-modal model. arXiv preprint arXiv:2408.11039 (2024)
- [89] Zong, Z., Jiang, D., Ma, B., Song, G., Shao, H., Shen, D., Liu, Y., Li, H.: Easyref: Omnigeneralized group image reference for diffusion models via multimodal llm. arXiv preprint arXiv:2412.09618 (2024)
- [90] Zong, Z., Ma, B., Shen, D., Song, G., Shao, H., Jiang, D., Li, H., Liu, Y.: Mova: Adapting mixture of vision experts to multimodal context. arXiv preprint arXiv:2404.13046 (2024)

## A More Experiment Details

- A.1 Experiment Setup

Training Settings. Our training dataset comprises text prompts sourced from the training set of T2I-CompBench [22] and [19], totaling 6,786 prompts with no images. Prior to training, we use GPT4o mini to extract the objects and their attributes from the prompts to facilitate computing the rewards. We use Janus-Pro-7B as the base model. We use a learning rate of 1e-6 and a beta of 0.01. For the reward model, we choose HPS [77] as the human preference model, GroundingDINO [41] as the object detector, and GIT [70] as the VQA model. For the ORM, we finetune LLaVA-OneVision-7B in the same manner as [19].

Benchmark. We test on T2I-CompBench [22], WISE [49], GenAI-Bench [36], and TIIFBench [73] to validate the effectiveness of our method. T2I-CompBench comprises 6,000 compositional text prompts evaluating three categories (attribute binding, object relationships, and complex compositions) and six sub-categories (color binding, shape binding, texture binding, spatial relationships, non-spatial relationships, and complex compositions). WISE consists of 1,000 text prompts spanning three categories (cultural common sense, spatial-temporal reasoning, and natural science) for evaluating world knowledge of the text-to-image models. To correctly generate an image, the model needs to reason about what the exact object or scenario is depicted in the prompt. We slightly modify the reasoning instruction on the WISE benchmark for more aligned results. GenAIBench is a benchmark containing 1,600 complex, real-world text prompts collected from professional designers, which covers a broad spectrum of compositional text-to-visual generation elements, from basic aspects like scenes, attributes, and relationships to more professional ones, including counting, comparison, differentiation, and logical reasoning. TIIF-Bench is a comprehensive benchmark for fine-grained text-to-image model evaluation, featuring 36 novel prompt combinations across six compositional dimensions and 100 real-world designer-level prompts with rich aesthetic judgment. We follow the official evaluation setting of all the benchmarks.

[Figure 88]

Token-level CoT Only Semantic-level + Token-level CoT

[Figure 89]

[Figure 90]

a fabric bag and a glass vase

a key on the right of a dog

[Figure 91]

Figure 7: More Visualization Result of the Image Diversity of a Single Prompt. We showcase the result of only token-level CoT optimized and both semantic-level and token-level CoT optimized.

B More Experiment Results

- B.1 More Results

We provide the experiment results on GenAI-Bench in Table 5 and TIIF-Bench in Table 6. As shown in Table 5, T2I-R1 largely improves the baseline model, and in the meantime, achieves the highest overall score on both the basic and advanced prompts. Again, T2I-R1 surpasses FLUX.1 in both

- Table 5: GenAI-Bench Evaluation Results. The best score is in blue , with the second-best score in green .

Basic Prompt Advanced Prompt Method Attribute↑ Scene↑

Relation

Overall↑ Count↑ Differ↑ Compare↑

Logical

Overall↑ Spatial↑ Action↑ Part↑ Negate↑ Universal↑

Diffusion Models

SD v2.1 [58] 0.80 0.79 0.76 0.77 0.80 0.78 0.68 0.70 0.68 0.54 0.64 0.62 SD-XL [54] 0.84 0.84 0.82 0.83 0.89 0.83 0.71 0.73 0.69 0.50 0.66 0.63 Midjourney v6 [48] 0.88 0.87 0.87 0.87 0.91 0.87 0.78 0.78 0.79 0.50 0.76 0.69 FLUX.1-dev [28] 0.87 0.88 0.87 0.85 0.87 0.87 0.75 0.78 0.74 0.45 0.70 0.64

Auto-Regressive Models

LWM [37] 0.63 0.62 0.65 0.63 0.70 0.63 0.59 0.58 0.54 0.49 0.52 0.53 Show-o [79] 0.72 0.72 0.70 0.70 0.75 0.70 0.70 0.62 0.71 0.51 0.65 0.60 VILA-U [78] 0.78 0.78 0.77 0.78 0.79 0.76 0.70 0.71 0.74 0.53 0.66 0.64 Liquid [75] – – – – – – 0.76 0.73 0.74 0.46 0.74 0.65 UniTok [44] – – – – – – 0.76 0.76 0.79 0.46 0.73 0.67 Mogao-7B [35] – – – – – – 0.77 0.74 0.77 0.53 0.71 0.68 Janus-Pro-7B [7] (Baseline) 0.85 0.87 0.85 0.84 0.85 0.84 0.73 0.73 0.71 0.48 0.65 0.65 T2I-R1 (Ours) 0.87 0.89 0.89 0.87 0.87 0.88 0.81 0.82 0.78 0.60 0.73 0.73

- Table 6: TIIF-Bench Testmini Subset Evaluation Results. The best score is in blue , with the second-best score in green .

Basic Following Advanced Following Designer

Overall

Model

Attribute +Relation

Attribute +Reasoning

Relation +Reasoning

Real World

Avg Attribute Relation Reasoning Avg

Style Text

short long short long short long short long short long short long short long short long short long short long short long short long

Llamagen [64] 41.67 38.22 53.00 50.00 48.33 42.33 59.57 60.32 51.07 47.32 35.89 32.61 38.82 31.57 40.84 47.22 49.59 46.22 46.67 33.33 0.00 0.00 39.73 35.62 LightGen [76] 53.22 43.41 66.58 47.91 55.83 47.33 74.82 45.82 69.07 50.57 46.74 41.53 62.44 40.82 61.71 50.47 50.34 45.34 53.33 53.33 0.00 6.83 50.92 50.55 Show-o [79] 59.72 58.86 73.08 75.83 74.83 79.83 78.82 78.32 65.57 69.32 53.67 50.38 60.95 56.82 68.59 68.96 66.46 56.22 63.33 66.67 3.83 2.83 55.02 50.92 Infinity [20] 62.07 62.32 73.08 75.41 74.33 76.83 72.82 77.57 72.07 71.82 56.64 54.98 60.44 55.57 74.22 64.71 60.22 59.71 80.00 73.33 10.83 23.83 54.28 56.89 Janus-Pro [7] 66.50 65.02 79.33 78.25 79.33 82.33 78.32 73.32 80.32 79.07 59.71 58.82 66.07 56.20 70.46 70.84 67.22 59.97 60.00 70.00 28.83 33.83 65.84 60.25 T2I-R1 (Ours) 68.59 67.19 82.90 81.63 86.50 83.00 83.47 79.43 78.73 82.46 69.05 68.00 71.64 69.47 72.43 69.95 69.40 70.40 60.00 63.33 27.60 26.24 67.54 60.45

types of prompts and showcases a remarkable margin in the advanced prompt, probably attributed to the high-level reasoning capability granted by semantic-level CoT. We provide more qualitative examples in Fig. 8.

##### B.2 More Illustration of Ablation Study

To validate the effectiveness of the semantic-level CoT, we compare T2I-R1 with a baseline method that generates images using only the token-level CoT optimized with the GRPO method. This is the default text-to-image generation setting in Janus, whose result is shown in the third row in Table 4. Comparing the third and fourth row in the table, we find that semantic-level CoT generally brings performance improvements across both benchmarks tested. We witness a particularly significant gain on the WISE benchmark. This enhanced performance can be attributed to the textual reasoning capabilities inherent in semantic-level CoT. As illustrated in Fig. 5, our method could first clearly reason about the objects or phenomena described in the prompt through semantic-level CoT. This effectively decouples the reasoning and generation processes and thereby facilitates superior results. We also observe that training solely with token-level CoT substantially reduces the diversity of generated images, as demonstrated in Fig. 6, 7, 13, and 14. To quantify this effect, we evaluate image diversity by reusing the generated images from T2I-CompBench, where each prompt generates ten images. We compute the Vendi Score [15] across the ten images for each prompt. Results indicate that GRPO training without semantic-level CoT decreases the diversity score, whereas incorporating semantic-level CoT significantly improves diversity through varied textual planning.

We also consider another situation to validate the effectiveness of token-level CoT: the semantic-level CoT is incorporated in the image generation process, as T2I-R1, but GRPO only optimizes the semantic-level CoT without the token-level CoT. This can be viewed as only enhancing the model’s high-level planning capabilities. The second row of Table 4 presents the result. The results show that optimizing semantic-level CoT exclusively yields smaller improvements compared to the joint optimization approach. Additionally, we find that optimizing both CoT types produces images with

T2I-R1 (Semantic-level +Token-level)

T2I-R1 (Semantic-level +Token-level)

Only Semantic-level

Only Token-level

Only Semantic-level

Only Token-level

Janus-Pro

Janus-Pro

[Figure 92]

[Figure 93]

A chameleon perfectly camouflaged against a brown leaf A chameleon perfectly camouflaged against a green leaf

[Figure 94]

[Figure 95]

A specific type of aircraft used in World War II

A form of personal communication before the advent of telephones

[Figure 96]

[Figure 97]

A traditional mode of transportation in Egypt

A famous flower that symbolizes wealth in China

[Figure 98]

[Figure 99]

Copper wire exposed to air for a long time

An animal with a long nose

[Figure 100]

[Figure 101]

A flower with deep cultural and religious significance in India, representing purity and beauty

A massive stone statue of a mythical creature that is a prominent historical landmark in Egypt

- Figure 8: More Visualization Results. We provide the image generation results of the same prompt from four models: base model, the model with only semantic-level CoT optimized, the model with only token-level CoT optimized, and the model with both levels of CoT optimized.

much better aesthetic quality compared with optimizing semantic-level CoT only. This indicates the necessity to jointly optimize both levels of CoT.

Finally, we discuss the zero-shot potential of the baseline model to perform both semantic-level and token-level reasoning. Specifically, we apply the same image generation process of T2I-R1 directly to the baseline model, where the baseline model is first instructed to output the semantic-level CoT and then the token-level CoT. We term this method of generation as ‘Janus-Pro w/ zero-shot semantic-level CoT’ in Figure 9-12. As shown in the figure, zero-shot semantic-level CoT brings very marginal improvement, while T2I-R1 demonstrates a satisfying result. The reasons are twofold: (1) Zero-shot semantic-level CoT misses critical objects in the original prompt. As shown in Figure 12, the zero-shot semantic-level CoT misses the bird in the original prompt. (2) Zero-shot semantic-level CoT does not fit the model’s generation ability or provide useful information for generation. Although the semantic-level CoT in Figure 9-11 includes all the objects and relationships, the baseline model still fails to generate a satisfying result. This highlights the necessity of our proposed BiCoT-GRPO training method to build the synergy between the two levels of CoT and make them work together.

##### B.3 More Details about Reward Analysis

We conduct a human study to evaluate the visual quality of the generated images. Specifically, we select four options of reward models (V, O, H + D, and H + D + V) to generate an image from the same prompt. Then we ask humans to rank the four images and score them according to the rank (rank 1 for 3 points, rank 2 for 2 points, and so on). The humans are instructed to rank the images only based on the visual appeal. We employ eight graduate students to conduct the study to eliminate individual bias. We randomly choose 30 prompts from each of the subtasks from the T2I-CompBench. The result is shown in the visual quality column in Table 3. We observe that ensemble rewards achieve better visual quality, with H + D + V obtaining slightly superior results. This improvement could

be attributed to the implicit regularization provided by multiple rewards, preventing overfitting to a single reward model. Conversely, individual reward models fail to provide satisfactory quality despite high benchmark scores.

##### B.4 Hyperparameters

All of our experiments are conducted on 8 H800. Our training procedure lasts about 16 hours. We provide the detailed training hyperparameters in Table 7.

Table 7: T2I-R1 training hyperparameters. Name

Learning rate 1e-6 Beta β 0.01 Group Size G 8 Classifier-Free Guidance Scale 5 Max Gradient Norm 1.0 Batchsize 8 Training Steps 1,600 Gradient Accumulation Steps 2 Image Resolution h × w 384 × 384

## C Limitations and Future Work

While this work explores the text-to-image generation task, it requires more exploration on how to apply this paradigm to video generation tasks. Video generation tasks are more complex regarding the reward design and the base model. For the reward design, how to apply dense rewards on each generated frame is still an open question. Besides, there exists no understanding and generation unified model for videos, so BiCoT-GRPO cannot be used directly. Meanwhile, the current inference time of video generation is too long for the current GRPO paradigm. How to balance the training time and effect needs further study.

[Figure 102]

Prompt

a red train, a blue bowl, a green flower, and a yellow cake

Janus-Pro w/ zero-shot semantic-level CoT

Janus-Pro

T2I-R1

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

Underneath a blue sky, a red train is moving along the railway track. Next to it, a green flower stands tall, while a yellow cake sits on a nearby table.

Zero-shot Semantic-level CoT

[Figure 108]

[Figure 109]

In the image, the red train occupies the left side of the image, with a sleek and polished metallic finish. Beside it, a blue bowl is sitting gently on a wooden table, its surface reflecting light, and there lies a delicate green flower on the table, its petals spreading like a butterfly's wings. In the background, an appetizing yellow cake with an intricate icing decoration stands next to a table on the right side of the scene. The train appears slightly ahead of the bowl, while the cake sits further behind the table and flower, creating an interesting spatial arrangement of objects and colors against a bright backdrop.

T2I-R1 Semantic-level CoT

- Figure 9: Visualization Results of Semantic-level CoT. We provide the image generation results of the same prompt from three settings: base model, base model with zero-shot semantic-level CoT, and T2I-R1. For the setting of base model with zero-shot semantic-level CoT, we use the same generation pipeline of T2I-R1 directly on the base model. We employ the same prompt of T2I-R1 to instruct the base model to generate a zero-shot semantic-level CoT, which we visualize in the figure and provide a comparison of the semantic-level CoT generated by T2I-R1.

[Figure 110]

Prompt

three paintings and one knife

Janus-Pro w/ zero-shot semantic-level CoT

Janus-Pro

T2I-R1

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

Three paintings hung on the wall, and in the middle stood a sharplooking knife. The three paintings were of varying nature: one depicted sunflowers in a vibrant yellow, the second portrayed a serene countryside with soft blues and greens, while the third had an abstract, multicolored design.

Zero-shot Semantic-level CoT

[Figure 116]

[Figure 117]

The image displays three paintings hung on a wall, each characterized by vibrant colors such as pink, blue, and yellow, reflecting various artistic styles. In front of the paintings, there is a single, slender knife with a metallic polish, positioned horizontally and lying on a polished wooden surface. The knife's reflective glare contrasts with the richness of the paintings, creating an intricate visual interplay. The paintings provide a backdrop to the knife, and their colors accentuate the sharpness of the blade.

T2I-R1 Semantic-level CoT

- Figure 10: Visualization Results of Semantic-level CoT. We provide the image generation results of the same prompt from three settings: base model, base model with zero-shot semantic-level CoT, and T2I-R1. For the setting of base model with zero-shot semantic-level CoT, we use the same generation pipeline of T2I-R1 directly on the base model. We employ the same prompt of T2I-R1 to instruct the base model to generate a zero-shot semantic-level CoT, which we visualize in the figure and provide a comparison of the semantic-level CoT generated by T2I-R1.

[Figure 118]

Prompt

a dog behind the door

Janus-Pro w/ zero-shot semantic-level CoT

Janus-Pro

T2I-R1

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

A dog is standing behind a door, with brown fur, dark eyes, a black collar, and a white patch on its face. The door is wooden, with a metal handle and hinges. The dog's ears are perked up, and it looks curious or alert through the door.

Zero-shot Semantic-level CoT

[Figure 124]

[Figure 125]

The image shows a brown and white dog peeking its head through a small gap in the door, with a gentle and curious expression etched onto its face. The door is slightly ajar, and the dog's body is barely

visible beyond the edge of the frame, hinting at a playful waiting game. The room behind the door is dimly lit, creating an eerie yet intimate atmosphere as if the dog is on a mission or has been guarding the entrance for some time. The door itself is wooden with a simple metal handle. The overall tone of the visual emphasizes curiosity and anticipation.

T2I-R1 Semantic-level CoT

- Figure 11: Visualization Results of Semantic-level CoT. We provide the image generation results of the same prompt from three settings: base model, base model with zero-shot semantic-level CoT, and T2I-R1. For the setting of base model with zero-shot semantic-level CoT, we use the same generation pipeline of T2I-R1 directly on the base model. We employ the same prompt of T2I-R1 to instruct the base model to generate a zero-shot semantic-level CoT, which we visualize in the figure and provide a comparison of the semantic-level CoT generated by T2I-R1.

[Figure 126]

Prompt

a dog, a bird, a lake, and a building

Janus-Pro w/ zero-shot semantic-level CoT

Janus-Pro

T2I-R1

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

Zero-shot Semantic-level CoT

[Figure 131]

A dog wags its tail, plays fetch at the lake, and sits in front of the building.

[Figure 132]

[Figure 133]

A fluffy white dog with bright amber eyes sits by the edge of a serene lake, observing a vibrant black and white bird soaring above. Nearby, a classic building in shades of brown and cream stands tall with numerous windows, giving it an old-world charm. Together, they create a picturesque scene that evokes peace and charm.

T2I-R1 Semantic-level CoT

- Figure 12: Visualization Results of Semantic-level CoT. We provide the image generation results of the same prompt from three settings: base model, base model with zero-shot semantic-level CoT, and T2I-R1. For the setting of base model with zero-shot semantic-level CoT, we use the same generation pipeline of T2I-R1 directly on the base model. We employ the same prompt of T2I-R1 to instruct the base model to generate a zero-shot semantic-level CoT, which we visualize in the figure and provide a comparison of the semantic-level CoT generated by T2I-R1.

### Janus-Pro T2I-R1

[Figure 134]

[Figure 135]

origami cranes unfolding into real birds during cherry blossom season

- Figure 13: More Visualization Result of the Image Diversity of a Single Prompt. We showcase the result of the baseline model, Janus-Pro, and T2I-R1.

#### Janus-Pro T2I-R1

[Figure 136]

[Figure 137]

paint drops falling from brushes creating flowers on a canvas below

- Figure 14: More Visualization Result of the Image Diversity of a Single Prompt. We showcase the result of the baseline model, Janus-Pro, and T2I-R1.

