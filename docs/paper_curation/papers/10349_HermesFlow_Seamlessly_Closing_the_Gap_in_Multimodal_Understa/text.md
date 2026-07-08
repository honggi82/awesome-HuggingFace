## HermesFlow: Seamlessly Closing the Gap in Multimodal Understanding and Generation

Ling Yang*1 Xinchen Zhang*2 Ye Tian1 Chenming Shang2 Minghao Xu13 Wentao Zhang1 Bin Cui1 https://github.com/Gen-Verse/HermesFlow

### Abstract

# arXiv:2502.12148v2[cs.CV]25Sep2025

The remarkable success of the autoregressive paradigm has made significant advancement in Multimodal Large Language Models (MLLMs), with powerful models like Show-o, Transfusion and Emu3 achieving notable progress in unified image understanding and generation. For the first time, we uncover a common phenomenon: the understanding capabilities of MLLMs are typically stronger than their generative capabilities, with a significant gap between the two. Building on this insight, we propose HermesFlow, a simple yet general framework designed to seamlessly bridge the gap between understanding and generation in MLLMs. Specifically, we take the homologous data as input to curate homologous preference data of both understanding and generation. Through Pair-DPO and self-play iterative optimization, HermesFlow effectively aligns multimodal understanding and generation using homologous preference data. Extensive experiments demonstrate the significant superiority of our approach over prior methods, particularly in narrowing the gap between multimodal understanding and generation. These findings highlight the potential of HermesFlow as a general alignment framework for next-generation multimodal foundation models.

[Figure 1]

[Figure 2]

(a) (b)

Text1 Text2

Image1 Image2

Und Preference

Gen Preference data

MLLM

data MLLM

[Figure 3]

[Figure 4]

DPO Training

DPO Training

Caption

Image Text

Text1 Text2

Iterative Optimization

Und

[Figure 5]

MLLM

MLLMGen&Und Pair-DPO Training

Pair preference data

(Image, Caption)

[Figure 6]

Gen

MLLM

Selfcritic

(c)

Image1 Image2

Figure 1. Architecture comparison between (a) DPO training improve multimodal understanding (Zhou et al., 2024c; He et al., 2024), (b) DPO training improve multimodal generation (Wang et al., 2024) and (c) our HermesFlow.

2024b; Tian et al., 2024a; Fan et al., 2024). Recent studies (Team, 2024; Li et al., 2024; Wu et al., 2024b;a; Ma et al., 2024) have focused on developing unified system capable of handling both multimodal understanding and generation. Powerful Multimodal Large Language Models (MLLMs) like Show-o (Xie et al., 2024a), Transfusion (Zhou et al., 2024a), and Emu3 (Wang et al., 2024), employ a single transformer to unify these tasks, demonstrating remarkable performance across both domains.

Recently, there has been growing interest in exploring the synergy between multimodal understanding and generation (Wu et al., 2024b; Tong et al., 2024; Dong et al., 2023). Liquid (Wu et al., 2024b) demonstrates that these two tasks are mutually beneficial: expanding the data for either understanding or generation enhances the performance of the other. Furthermore, MetaMorph (Tong et al., 2024) reveals that understanding data is more effective than generation data in improving both understanding and generation performance. However, these works jointly improve the understanding and generation capabilities of MLLMs from a data-level perspective but fail to consider the gap between them. It remains unclear whether a capability difference

### 1. Introduction

The rapid advancement of Large Language Models (LLMs) (OpenAI, 2024; Guo et al., 2025; Yang et al., 2024b; 2025a) has driven significant development in both multimodal understanding (Liu et al., 2024a; Zhu et al., 2023; Li et al., 2023a) and autoregressive image generation (Sun et al.,

*Equal contribution 1Peking University 2Tsinghua University 3Mila - Qu´ebec AI Institute. Correspondence to: Ling Yang <yangling0818@163.com>. Preprint.

exists between them.

To quantitatively assess the performance of multimodal understanding and generation, we design a general pipeline, as illustrated in Figure 2 (a). For any pretrained MLLM, input consists of (image, prompt/caption) pairs. For understanding tasks, MLLM is presented with multiple questions related to each image, and the final understanding score is calculated as the average accuracy of its answers. MLLM generates an image for each prompt, and these images are evaluated by posing the same set of questions using GPT-4o (Hurst et al., 2024), with the final generation score calculated based on the average accuracy of GPT-4o’s answers. We employed this pipeline to evaluate multiple MLLMs. As shown in Figure 2 (b), models like VILA-U (Wu et al., 2024c), Janus (Wu et al., 2024a) and Show-o (Xie et al., 2024a) exhibit notably stronger understanding capabilities compared to their generation capabilities. Our experiments highlight a recurring phenomenon: MLLMs consistently demonstrate superior understanding abilities over generation abilities, with a significant gap between them.

In the pretraining of MLLMs, simply increasing the training data for understanding or generation does not yield proportional improvements in both aspects (Tong et al., 2024), leaving a significant gap between their understanding and generation capabilities. To bridge the gap between understanding and generation in MLLMs, we propose HermesFlow, which collects paired understanding and generation preferences from homologous input data, and then employ a novel Pair-DPO post-training framework to seamlessly bridge the gap through the paired preference data. To curate understanding preference data, we enable MLLM to generate multiple captions for a single input image and filter paired understanding preference data using BERT similarity scores. To curate generation preference data, we prompt MLLM to generate multiple images from a single prompt and employ a self-critic-like approach to evaluate the images through self-VQA scoring, thereby filtering and selecting the paired generation preference data. Finally, we design Pair-DPO for preference alignment of homologous paired data, and through iterative optimization to simultaneously and progressively reduce the gap between understanding and generation following the same approach. We achieve the self-improvement of both understanding and generation of MLLM without incorporating any external high-quality training data.

We compare HermesFlow with previous work in Figure 1 and summarize our main contributions as follows:

• An insightful discovery regarding a significant gap between the understanding and generation abilities of MLLMs, with understanding consistently outperforming generation.

- • We propose a general multimodal self-improvement framework, HermesFlow, using Pair-DPO based on homologous data to seamlessly close the gap between multimodal understanding and generation.
- • Self-play iterative optimization paradigm is highly compatible with the multi-round enhancement of MLLMs. HermesFlow has potential as a general alignment framework for next-generation multimodal foundation models.
- • Extensive qualitative and quantitative comparisons with previous powerful methods, such as Show-o, Janus and VILA-U, demonstrate the effectiveness and superiority of our method.

### 2. Related Work

#### 2.1. Unified Multimodal Understanding and Generation In recent years, a growing number of studies (Dong et al.,

- 2023; Ge et al., 2024; Wu et al., 2023; Ye et al., 2024; Ma et al., 2024; Shi et al., 2024) have explored unified multimodal models capable of both visual understanding and generation. Early methods (Dong et al., 2023; Tong et al.,
- 2024; Ge et al., 2024; Sun et al., 2024c; Zhuang et al., 2024; Zhang et al., 2024c) leveraged diffusion models as external tools, where MLLMs generate conditions for visual generation (Yang et al., 2024a; Tian et al., 2024b) without having direct generative capabilities. For instance, DreamLLM (Dong et al., 2023) introduces learnable embeddings called dream queries, which encapsulate the semantics encoded by MLLMs and serve as conditions for the diffusion decoder. More recently, inspired by the success of autoregressive paradigms, many studies (Team, 2024; Xie et al., 2024a; Zhou et al., 2024a; Qu et al., 2024; Xie et al., 2024b; Zhang et al., 2024b; Wang et al., 2024) have shifted focus to representing and generating images using discrete visual tokens within a single transformer framework. For instance, Emu3 (Wang et al., 2024) is trained solely with next-token prediction on a mixture of multimodal sequences using a single transformer. Janus (Wu et al., 2024a) separates visual encoding into distinct pathways for multimodal understanding and generation while maintaining a unified transformer architecture. However, no existing research has focused on the relationship between the strengths of understanding and generation capabilities in MLLMs, which is essential for the balanced and sustainable development of these models.

#### 2.2. DPO in Multimodal LLMs

Direct Preference Optimization (DPO) (Rafailov et al., 2024; Zhang et al., 2024d; Yang et al., 2025b;a) enhances the performance of multimodal LLMs through the post-training process. In Figure 1, we categorize these approaches into three types. Some methods (Zhou et al., 2024b;c; He et al.,

[Figure 7]

[Figure 8]

Image1 Image2 Image3

[Figure 9]

[Figure 10]

[Figure 11]

Understanding Score

. . .

Generation Score

- Prompt1: A hedgehog wrapped in a blanket, holding a cup of hot chocolate, sitting by a 'fireplace' made of pebbles.
- Prompt2: A cute orange kitten sliding down an aqua slide. happy excited.
- Prompt3: a blue Porsche 356 parked in front of a yellow brick wall.

MLLM

[Figure 12]

[Figure 13]

[Figure 14]

GPT-4o

[Figure 15]

[Figure 16]

[Figure 17]

. . .

(a) (b)

- Figure 2. Motivation of HermesFlow. (a) A general pipeline to quantitatively assess the MLLM’s performance of multimodal understanding and generation. (b) The imbalance between understanding and generation capabilities is a common phenomenon in MLLMs, and our method ssignificantly narrows this disparity. For detailed descriptions, please refer to Section 5.2.

2024; Zhang et al., 2024a) utilize DPO to enhance understanding capability, as shown in Figure 1 (a). For instance, CSR (Zhou et al., 2024c) enables the model to self-improve by iteratively generating candidate responses, evaluating the reward for each response, and curating preference data for finetuning. Other methods (Wang et al., 2024) improve the generation capability of MLLMs through DPO as illustrated in Figure 1 (b). Emu3 (Wang et al., 2024) generates a data pool and constructs a preference dataset through manual ranking, which is then used to optimize the model’s generation capabilities via DPO. However, these models focus exclusively on enhancing either understanding or generation capabilities. In contrast, our approach uses Pair-DPO to effectively narrow the gap between the two, achieving mutual improvement.

negative log-likelihood loss over the dataset:

T

1 T

log P(xt|x<t) (2)

L = −

t=1

In autoregressive models, next-token prediction facilitates sequential generation by iteratively sampling tokens from the learned distribution P(xt|x<t). This approach is widely applicable multimodal domains such as visual understanding and visual generation.

#### 3.2. Direct Preference Optimization

Direct Preference Optimization (DPO) provides a straightforward and efficient method by directly utilizing pairwise preference data to optimize the policy model. Specifically, given an input prompt x, and a preference data pair (yw,yl), DPO aims to maximize the probability of the preferred output yw and minimize that of the undesirable output yl. The optimization objective is formulated as:

### 3. Preliminary

#### 3.1. Next Token Prediction

Next token prediction is a fundamental task in sequence modeling, where the goal is to estimate the conditional probability of the next token xt given its preceding context x<t = {x1,x2,...,xt−1}. Formally, for a sequence x = {x1,x2,...,xT}, the joint probability is factorized as:

LDPO(θ) = − E(x,y

w,yl)∼D log σ

(3)

πθ(yl|x) πref(yl|x)

πθ(yw |x) πref(yw |x) −β log

β log

T

T

P(xt|x<t) (1)

P(xt|x1,x2,...,xt−1) =

P(x) =

where D is the pair-wise preference dataset, σ is the sigmoid function, πθ(· | x) is the policy model to be optimized, πref(· | x) is the reference model kept unchanged during training, and the hyperparameter β controls the distance from the reference model.

t=1

t=1

This factorization relies on the autoregressive assumption, where each token depends solely on its preceding tokens. During training, the model is optimized by minimizing the

BERT Sim

Self-Play Iterative Optimization ( )

###### Caption1:

###### Caption2:

A hedgehog in ..

Little hedgehog ..

Iterative update preference data

Und

###### Paired preference data

Multimodal LLM

Und pair:

Captions Images

Win: A hedgehog in blanket and hold a ..

Lose: Little hedgehog with a cup of tea ..

[Figure 18]

Give a caption for this image

Und & Gen

[Figure 19]

Multimodal LLM

[Figure 20]

###### Gen pair:

Caption: A hedgehog wrapped

Win: Lose:

[Figure 21]

[Figure 22]

Img1 Questions Img2 Questions

###### in a blanked.. Pair-DPO Training

[Figure 23]

###### Und

Gen

Multimodal LLM

Caption: A hedgehog wrapped in a blanked..

[Figure 24]

[Figure 25]

[Figure 26]

Img2 acc: 0.6

Img1 acc: 0.8

Img1 Img2

- Figure 3. Pipeline of HermesFlow. We begin by curating paired data that captures both understanding and generation preferences from homologous input data. Leveraging this homologous preference data, we design Pair-DPO and employ self-play iterative optimization to seamlessly bridge the gap between multimodal understanding and generation.
- 4. Method

scores (Devlin, 2018) s(y,x) between the original prompt y and each of the n captions. The caption with the highest BERT similarity score is selected as the winning sample yw, while the one with the lowest score is chosen as the losing sample yl. Following this process, we construct the pairwise understanding preference data.

In this section, we present our method, HermesFlow, which curates pairwise preference data for both multimodal understanding and generation using homologous images and prompts, and seamlessly bridging the gap of multimodal understanding and generation through Pair-DPO training. An overview of HermesFlow is illustrated in Figure 3. In Section 4.1, we detail the methods for curating homologous preference data for multimodal understanding and generation, respectively. In Section 4.2, we propose the Pair-DPO training strategy to bridge the gap between multimodal understanding and generation. In Section 4.3, we introduce self-play iterative optimization, enabling the selfimprovement of MLLM over multiple iterations.

Generation Preference Data Starting with the caption or prompt y, we use the pretrained MLLM to randomly generate n images. Given that MLLM’s understanding abilities surpass its generation capabilities, we apply a self-critique or self-selection method for choosing the generated data.

Specifically, given the prompt y, we use TIFA (Hu et al., 2023) to generate q visual question-answer pairs, denoted as {(Q1,A1),(Q2,A2),...,(Qq,Aq)}. For each generated image, we evaluate them based on the accuracy of the VQA responses provided by the MLLM:

#### 4.1. Curating Homologous Preference Data

Homologous Input Data The curation of both multimodal understanding and generation preference data begins with homologous data (x,y), where y represents the caption or prompt of the image x.

q

1 q

I(Rj,i = Ai), ∀j = 1,2,...,n (4)

Acc(xj) =

i=1

Rj,i = MLLM(xj,Qj,i) (5)

Understanding Preference Data We focus on the image captioning task to collect understanding preference data, which reflects the ability of MLLMs to capture visual features, including object attributes, spatial relationships, and detailed elements of both the subject and background. Give an image x, a pretrained MLLM is used to generate n different captions according to the input prompt: ”Give a caption for this image.”. We then calculate the BERT similarity

where Rj,i represents the response of MLLM according to the input of image xj and question Qj,i. We select the image with the highest accuracy as the winning sample xw and the one with the lowest accuracy as the losing sample xl, while also ensuring that the highest accuracy exceeds 0.6. Using this process, we construct the pairwise generation preference data.

Homologous Output Preference Data After curating understanding and generation preference data from homologous input (x,y) as mentioned above, where y represents the caption or prompt of the image x, we obtain the homologous output preference data D, denoted as (x,y,xw,xl,yw,yl).

#### 4.2. Unified Enhancement with Pair-DPO

Homologous preference paired data of understanding and generation indicate the optimized directions for both capabilities of a pretrained MLLM within the same semantic space. To achieve joint optimization and alignment of understanding and generation, we introduce Pair-DPO. The optimization objective of Pair-DPO can be formulated as:

LPair-DPO(θ) = −E(x,y,x

w,xl,yw,yl)∼D [log σ (∆Und∆Gen)]

(6) ∆Und = β log

πθ(yl | x) πref(yl | x)

πθ(yw | x) πref(yw | x) − β log

(7)

πθ(xw | y) πref(xw | y) − β log

πθ(xl | y) πref(xl | y)

(8)

∆Gen = β log

where ∆Gen and ∆Und represent the preference differences in generation and understanding of MLLM, respectively. By using Pair-DPO to optimize homologous preference data jointly, we not only ensure mutual improvement in the understanding and generation capabilities of MLLM but also effectively narrow the gap between them. We provide the detailed derivation of the Pair-DPO optimization objective in Section A.

#### 4.3. Self-Play Iterative Optimization

To achieve comprehensive optimization and achieve a convergence gap in understanding and generation of MLLMs, we introduce a novel yet easy self-play iterative optimization using Pair-DPO with multiple turns.

Take understanding preference data as an example. We denote the preference data curated in round i−1 in Section 4.1 as (ywi−1,yli−1). In the optimization of round i, the optimized MLLM generates n new captions (y1i,y2i,...,yni ) from the input of image x. The preference data is selected based on the following rules:

ymaxi = arg max

s(yki ,y) (9)

k∈{1,...,n}

(ywi ,yli)=

(ymaxi ,ywi−1) if s(ymaxi ,y)>s(ywi−1,y) (ymaxi ,yli−1) otherwise

(10)

where s(yki ,y) denotes the BERT similarity score between the generated caption yki and the homologous input caption y. Select the caption ymaxi with the highest similarity score, which represents the local upper bound of the optimized MLLM’s understanding capability. If s(ymaxi ,y) >

Algorithm 1 The pseudocode of HermesFlow Input: Homologous data (x,y), pretrained model MLLMθ with parameters θ

- 1: for i = 0,...,iter do
- 2: if i = 0 then
- 3: yw,yl = MLLMiθ(x) // Und preference data
- 4: xw,xl = MLLMiθ(y) // Gen preference data
- 5: else
- 6: y1i,y2i,...,yni = MLLMiθ−1(x)
- 7: ymaxi = arg maxk∈{1,...,n} s(yki ,x)
- 8: Update und-preference data using Equation (10)
- 9: xi1,xi2,...,xin = MLLMiθ−1(y)
- 10: ximax = arg maxk∈{1,...,n} Acc(xik)
- 11: Update gen-preference data using Equation (10)
- 12: end if
- 13: Optimize MLLMiθ−1 to MLLMiθ using Equation (6)
- 14: end for

s(ywi−1,y), MLLM has effectively learned preference knowledge from the previous round. Therefore, it needs to be updated and further optimized using the higher-quality sample ymaxi as the benchmark. Conversely, if s(ymaxi ,y) < s(ywi−1,y), effective optimization was not achieved in the previous round. In this case, it is necessary to update with simpler and clearer preference data ymaxi as the winning sample to provide a smoother learning gradient. Through iterative optimization, we achieve self-improvement of MLLM without relying on any external high-quality training data.

### 5. Experiments

#### 5.1. Experimental Setup

Training Setup We randomly select 5,000 image-caption pairs from JourneyDB (Sun et al., 2024a) as our homologous input data. For the Visual Question Answering (VQA) data corresponding to each pair, we combine the VQA from JourneyDB with the VQA generated from TIFA (Hu et al., 2023) for a comprehensive evaluation. Our HermesFlow is trained upon Showo (Xie et al., 2024a), using a batch size of 4 for both caption and generation data over 3,000 steps. We employ the AdamW optimizer with a weight decay of 0.01, and an initial learning rate of 2e-5 with a cosine scheduling. The parameter β for Pair-DPO is set to 0.2. All experiments are conducted under 8*NVIDIA A100 GPUs.

Evaluation Metrics To assess multimodal understanding capabilities, we evaluate using POPE (Li et al., 2023b), MME (Fu et al., 2023), Flickr30k (Plummer et al., 2015), VQAv2 (Goyal et al., 2017), GQA (Hudson & Manning, 2019), and MMMU (Yue et al., 2024). For visual generation capabilities, we use GenEval (Ghosh et al., 2024) and DPGBench (Hu et al., 2024) to evaluate the model’s prompt-

Table 1. Evaluation on multimodal understanding benchmarks. The baseline data is quoted from Show-o (Xie et al., 2024a).

Model # Params POPE↑ MME↑ Flickr30k↑ VQAv2(test)↑ GQA↑ MMMU↑ Gemini-Nano-1 (Team et al., 2023) 1.8B - - - 62.7 - 26.3 CoDI (Tang et al., 2024) - - - 12.8 - - Emu (Sun et al., 2024c) 13B - - 77.4 57.2 - NExT-GPT (Wu et al., 2023) 13B - - 84.5 66.7 - SEED-X (Ge et al., 2024) 17B 84.2 1435.7 52.3 - 47.9 35.6 DreamLLM (Dong et al., 2023) 7B - - - 72.9 - Chameleon (Team, 2024) 34B - - 74.7 66.0 - Show-o (Xie et al., 2024a) 1.3B 80.0 1232.9 67.6 74.7 61.0 27.4 HermesFlow (Ours) 1.3B 81.4 1249.7 69.2 75.3 61.7 28.3

VILA-U Janus Show-o Ours VILA-U Janus Show-o Ours

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

A charming red bicycle with a basket of flowers adds a splash of color to a quaint cobblestone street in a picturesque village.

A picturesque Parisian cafe terrace offers a stunning view of the Eiffel Tower framed by vibrant flowers and lush greenery.

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

A vintage clock stands elegantly on a mantel, flanked by three softly glowing candles.

Four dogs gather under the shade of a grand tree, embodying friendship and tranquility in nature.

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

A lush Japanese garden in spring, with blooming cherry blossoms and a tranquil koi pond.

A peaceful dinosaur roams amidst a prehistoric paradise of cascading waterfalls and vibrant blooms.

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

A mysterious detective examines cryptic letters by candlelight, deep in thought as shadows dance around the room.

Standing tall against the stormy skies, the lighthouse guides the way with its unwavering beacon of hope.

Figure 4. Qualitative comparison between our HermesFlow and three outstanding Multimodal LLMs VILA-U (Wu et al., 2024c), Janus (Wu et al., 2024a), and Show-o (Xie et al., 2024a). Colored text denotes the advantages of HermesFlow in generated images.

image alignment. We further assess image fidelity with FID (Heusel et al., 2017) and CLIP-Score (Radford et al., 2021). Additionally, we conduct a comprehensive user study to objectively compare our model with other baselines.

#### 5.2. Main Results

Multimodal Understanding Performances Table 1 summarizes the comparison between our method and other leading MLLMs on multimodal understanding benchmarks. Notably, HermesFlow achieves similar or superior understand-

Table 2. Evaluation on visual generation benchmarks: GenEval (Ghosh et al., 2024) and DPG-Bench (Hu et al., 2024).

GenEval↑ DPG-Bench↑ Methods #params Single Obj. Two Obj. Counting Colors Position Color Attri. Overall Average

Diffusion Model

LDM (Rombach et al., 2022) 1.4B 0.92 0.29 0.23 0.70 0.02 0.05 0.37 DALL-E 2 (Ramesh et al., 2022) 4.2B 0.94 0.66 0.49 0.77 0.10 0.19 0.52 -

- SD 1.5 (Rombach et al., 2022) 860M 0.94 0.37 0.27 0.72 0.05 0.07 0.40 63.18

- SD 2.1 (Rombach et al., 2022) 865M 0.97 0.50 0.46 0.80 0.07 0.14 0.49 68.09

Autoregressive Model LlamaGen (Sun et al., 2024b) 775M 0.87 0.25 0.23 0.51 0.06 0.04 0.32 65.16 Emu (Sun et al., 2024c) 14B 0.87 0.34 0.26 0.56 0.07 0.06 0.36 Chameleon (Team, 2024) 34B 0.89 0.39 0.28 0.66 0.08 0.07 0.40 LWM (Liu et al., 2024b) 7B 0.93 0.41 0.46 0.79 0.09 0.15 0.47 SEED-X (Ge et al., 2024) 17B 0.97 0.58 0.26 0.80 0.19 0.14 0.49 Show-o (Xie et al., 2024a) 1.3B 0.95 0.52 0.49 0.82 0.11 0.28 0.53 67.48 Janus (Wu et al., 2024a) 1.3B 0.97 0.68 0.30 0.84 0.46 0.42 0.61 HermesFlow (Ours) 1.3B 0.98 0.84 0.66 0.82 0.32 0.52 0.69 70.22

Table 3. MSCOCO zero-shot FID and CLIP-Score.

Method # Params FID↓ CLIP-Score↑

LDM (Rombach et al., 2022) 1.4B 12.64 DALL·E 2 (Ramesh et al., 2022) 6.5B 10.39 -

- SD 1.5 (Rombach et al., 2022) 860M 9.62 30.23

- SD 2.1 (Rombach et al., 2022) 865M 8.03 30.87

LlamaGen (Sun et al., 2024b) 775M 9.45 29.12 Emu (Sun et al., 2024c) 14B 11.02 28.98 LWM (Liu et al., 2024b) 7B 12.68 SEED-X (Ge et al., 2024) 17B 14.99 Show-o (Xie et al., 2024a) 1.3B 9.24 30.63 HermesFlow (Ours) 1.3B 9.07 31.08

|73% 11% 16%<br><br>69% 7% 24%<br><br>69% 10% 21%<br><br>58% 18% 24%|
|---|

Ours vs Show-o

Ours vs Janus

Ours vs VILA-U

Ours vs Chamelon

0%

20% 40% 60% 80% 100%

Win Tie Loss

Figure 5. Results of user study.

ing performance compared to larger models like SEED-X and Chameleon, using less than 1/10 of the parameters. Additionally, HermesFlow demonstrates significant strengths across all metrics compared to Show-o, indicating that PairDPO effectively reduces the understanding-generation gap while maintaining or even enhancing understanding ability.

Image Generation Performances As shown in Figure 4, HermesFlow achieves superior generation results compared to three powerful Multimodal LLMs: VILA-U (Wu et al., 2024c), Janus (Wu et al., 2024a), and Show-o (Xie et al., 2024a). Compared to its backbone, Show-o, HermesFlow demonstrates superior performance in generating object attributes and accurate counting. This improvement stems from its stronger understanding capabilities, which are utilized to filter generated images and achieve mutual refinement through Pair-DPO iteratively.

We compare HermesFlow with other visual generation models on GenEval (Ghosh et al., 2024) and DPG-Bench (Hu et al., 2024), as shown in Table 2. Compared to the diffusionbased generative model SD 2.1 (Rombach et al., 2022), HermesFlow demonstrates remarkable performance across all

benchmarks. Furthermore, it surpasses larger autoregressive models, such as Chameleon (Team, 2024) and LWM (Liu et al., 2024b). When compared to Show-o (Xie et al., 2024a), HermesFlow exhibits significant strengths in object counting and positions, this is attributed to the critique of its superior understanding capability, which greatly enhances its visual generation performance in aspects such as object quantity, location, and attributes. We present the zero-shot FID (Heusel et al., 2017) and CLIP-Score (Radford et al., 2021) of HermesFlow on MSCOCO-30K in Table 3. The results clearly show that after the iterative optimization with Pair-DPO, HermesFlow achieves improved performance in both image fidelity and prompt-image alignment.

Quantitative assess of MLLM’s Understanding and Generation Gap We also conducted a comprehensive user study to evaluate the effectiveness of HermesFlow in visual generation. As illustrated in Figure 5, we randomly selected 25 prompts for each comparison, and invited 35 users from diverse backgrounds to vote on image generation quality, collecting a total of 3,500 votes. Alignment between the generated images and the prompts was used as the pri-

- Table 4. Quantitative assess of MLLM’s Understanding and Generation Gap.

Method # Params Understanding Score↑ Generation Score ↑ Gap ↓ VILA-U (Wu et al., 2024c) (Xie et al., 2024a) 7B 0.646 0.477 0.169 Janus (Wu et al., 2024a) 1.3B 0.599 0.417 0.182 Show-o (Xie et al., 2024a) 1.3B 0.520 0.433 0.087 HermesFlow (Ours) 1.3B 0.533 0.497 0.036

- Table 5. Comparison of Pair-DPO vs. DPO and the Effect of Pair-DPO Iterations.

Understanding Bench Generation Bench Methods POPE↑ MME↑ MMMU↑ GenEval (Overall)↑ DPG-Bench (Average)↑

Show-o (Xie et al., 2024a) 80.0 1232.9 27.4 0.53 67.48 DPO (Understanding) 80.8 1242.2 27.8 0.58 67.88 DPO (Generation) 80.5 1239.3 27.5 0.70 70.03

- Pair-DPO (Iter. 0) (Show-o) 80.0 1232.9 27.4 0.53 67.48

- Pair-DPO (Iter. 1) 81.1 1246.7 28.0 0.68 70.19

- Pair-DPO (Iter. 2) 81.3 1248.3 28.1 0.69 70.21

- Pair-DPO (Iter. 3) 81.4 1249.7 28.3 0.69 70.22

mary evaluation criterion, with aesthetic quality and detail completeness considered under the same conditions. The results demonstrate that HermesFlow received widespread user approval in visual generation.

As shown in Figure 2, we use homologous data consisting of caption/prompt y and image x as input to evaluate the capability of understanding and generation respectively. The homologous data is randomly selected from JourneyDB (Sun et al., 2024a). For the understanding task, to ensure comprehensive and high-quality question-answer (QA) pairs, we first use TIFA (Hu et al., 2023) to generate QA pairs based on the image and caption. These are then augmented with QA pairs from JourneyDB to create a more thorough and indepth dataset. The final understanding score is calculated as the average accuracy of the answers. For the generation task, we use the prompt as input to generate an image for each prompt. These generated images are evaluated by posing the same set of questions to GPT-4o (Hurst et al., 2024), with the final generation score determined by the average accuracy of GPT-4o’s answers. Since the generation capabilities of MLLMs are relatively limited, strict evaluation criteria are applied in cases of severe object blurring or significant loss of details. Therefore, GPT-4o is required to carefully analyze the completeness and authenticity of the objects involved in each question before providing answers. This evaluation pipeline was applied to multiple MLLMs, with the results presented in Table 4.

It is clear that a significant gap exists between multimodal understanding and generation in MLLM. HermesFlow seamlessly bridges this gap through self-play iterative optimization using Pair-DPO from homologous preference data.

#### 5.3. Ablation Study

Pair-DPO vs. DPO Pair-DPO can simultaneously enhance both the understanding and generation capabilities of multimodal LLMs. As shown in Table 5, compared to DPO methods that rely solely on understanding or generation preference, a single round of Pair-DPO achieves superior performance by jointly optimizing both capabilities through the use of multimodal preference data. Furthermore, we observed that when using preference data from only one modality, whether understanding or generation, the capability of the other modality also improves. This demonstrates the same findings in MetaMorph (Tong et al., 2024) and Liquid (Wu et al., 2024b) that multimodal understanding and generation are synergistic.

Self-play Iterative Optimization As shown in Table 5, we conducted an experimental analysis to examine the impact of iterations in self-play iterative optimization. It is evident that the first round of iterative optimization yields the most significant improvements in both understanding and generation. This is because the notable gap between the understanding and generation capabilities of MLLMs is most effectively bridged in the initial iteration. When the number of iterations exceeds 2, we observed that understanding ability continues to improve slightly, while generation ability remains almost stable. We argue that since generation is a fine-grained visual task, cross-capability transfer has limited impact on further enhancing generation ability in subsequent iterations.

The Impact of Each Preference Sample Richness The performance of Pair-DPO is largely influenced by the num-

- 78

- 79

- 80

- 81

- 82

72

Und-MME

Und-Baseline

Gen-DPGBench

70

Gen-DPGBench

Gen-Baseline

###### Und-MME

68

66

64

2 4 6 8 10 12 14 16

Numbers of generated samples n

### References

Devlin, J. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018.

Dong, R., Han, C., Peng, Y., Qi, Z., Ge, Z., Yang, J., Zhao, L., Sun, J., Zhou, H., Wei, H., et al. Dreamllm: Synergistic multimodal comprehension and creation. arXiv preprint arXiv:2309.11499, 2023.

Figure 6. The influence of the richness of each preference sample.

ber of generated samples n for both understanding and generation data. We conducted experiments to analyze the impact of n on both understanding and generation in MLLMs, with results shown in Figure 6. The dashed lines in the figure represent the performance of the baseline model, Show-o (Xie et al., 2024a).

When n is too small, the model’s understanding and generation performance decline. This is due to the insufficient number of samples and the limited capability of the baseline model, which leads to a noisy preference dataset and significantly impacts the results. However, as the sample size increases, it enables more accurate identification of the model’s optimal local upper bound, which in turn facilitates the curation of higher-quality preference data, leading to noticeable improvements in the understanding and generation capabilities of MLLMs.

Furthermore, Figure 6 reveals that achieving performance comparable to the baseline in generation requires more sampling data that understanding. This indicates that the generation capabilities of MLLMs are more sensitive to noise in the preference data, highlighting a greater need for high-quality generation data.

### 6. Conclusion

In this paper, we present a new MLLM alignment paradigm, HermesFlow, to seamlessly bridge the gap between multimodal understanding and generation. By iterative optimized with Pair-DPO using homologous preference data, HermesFlow successfully Improve the capabilities of both multimodal understanding and generation while narrowing the gap between them. Our extensive empirical evaluations across diverse understanding and generation benchmarks demonstrate the effectiveness of HermesFlow. However, due to current limitations in the number and capabilities of open-source MLLMs, HermesFlow has not yet been optimized across a wider range of backbones. In the future, we plan to extend this framework to more models. HermesFlow has the potential as a general alignment framework for next-generation multimodal foundation models.

Fan, L., Li, T., Qin, S., Li, Y., Sun, C., Rubinstein, M., Sun, D., He, K., and Tian, Y. Fluid: Scaling autoregressive text-to-image generative models with continuous tokens. arXiv preprint arXiv:2410.13863, 2024.

Fu, C., Chen, P., Shen, Y., Qin, Y., Zhang, M., Lin, X., Yang, J., Zheng, X., Li, K., Sun, X., et al. Mme: A comprehensive evaluation benchmark for multimodal large language models. arXiv preprint arXiv:2306.13394, 2023.

Ge, Y., Zhao, S., Zhu, J., Ge, Y., Yi, K., Song, L., Li, C., Ding, X., and Shan, Y. Seed-x: Multimodal models with unified multi-granularity comprehension and generation. arXiv preprint arXiv:2404.14396, 2024.

Ghosh, D., Hajishirzi, H., and Schmidt, L. Geneval: An object-focused framework for evaluating text-to-image alignment. Advances in Neural Information Processing Systems, 36, 2024.

Goyal, Y., Khot, T., Summers-Stay, D., Batra, D., and Parikh, D. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 6904–6913, 2017.

Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., Zhu, Q., Ma, S., Wang, P., Bi, X., et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

He, J., Lin, H., Wang, Q., Fung, Y., and Ji, H. Selfcorrection is more than refinement: A learning framework for visual and language reasoning tasks. arXiv preprint arXiv:2410.04055, 2024.

Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., and Hochreiter, S. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017.

Hu, X., Wang, R., Fang, Y., Fu, B., Cheng, P., and Yu, G. Ella: Equip diffusion models with llm for enhanced semantic alignment. arXiv preprint arXiv:2403.05135, 2024.

Hu, Y., Liu, B., Kasai, J., Wang, Y., Ostendorf, M., Krishna, R., and Smith, N. A. Tifa: Accurate and interpretable text-to-image faithfulness evaluation with question answering. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 20406–20417, 2023.

Hudson, D. A. and Manning, C. D. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 6700– 6709, 2019.

Hurst, A., Lerer, A., Goucher, A. P., Perelman, A., Ramesh, A., Clark, A., Ostrow, A., Welihinda, A., Hayes, A., Radford, A., et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.

Li, H., Tian, C., Shao, J., Zhu, X., Wang, Z., Zhu, J., Dou, W., Wang, X., Li, H., Lu, L., et al. Synergen-vl: Towards synergistic image understanding and generation with vision experts and token folding. arXiv preprint arXiv:2412.09604, 2024.

Li, J., Li, D., Savarese, S., and Hoi, S. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International conference on machine learning, pp. 19730–19742. PMLR, 2023a.

Li, Y., Du, Y., Zhou, K., Wang, J., Zhao, W. X., and Wen, J.-R. Evaluating object hallucination in large visionlanguage models. arXiv preprint arXiv:2305.10355, 2023b.

Liu, H., Li, C., Wu, Q., and Lee, Y. J. Visual instruction tuning. Advances in neural information processing systems, 36, 2024a.

Liu, H., Yan, W., Zaharia, M., and Abbeel, P. World model on million-length video and language with blockwise ringattention. CoRR, 2024b.

Ma, Y., Liu, X., Chen, X., Liu, W., Wu, C., Wu, Z., Pan, Z., Xie, Z., Zhang, H., Zhao, L., et al. Janusflow: Harmonizing autoregression and rectified flow for unified multimodal understanding and generation. arXiv preprint arXiv:2411.07975, 2024.

OpenAI. Openai o1 system card. preprint, 2024.

Plummer, B. A., Wang, L., Cervantes, C. M., Caicedo, J. C., Hockenmaier, J., and Lazebnik, S. Flickr30k entities: Collecting region-to-phrase correspondences for richer image-to-sentence models. In Proceedings of the IEEE international conference on computer vision, pp. 2641– 2649, 2015.

Qu, L., Zhang, H., Liu, Y., Wang, X., Jiang, Y., Gao, Y., Ye, H., Du, D. K., Yuan, Z., and Wu, X. Tokenflow:

Unified image tokenizer for multimodal understanding and generation. arXiv preprint arXiv:2412.03069, 2024.

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748–8763. PMLR, 2021.

Rafailov, R., Sharma, A., Mitchell, E., Manning, C. D., Ermon, S., and Finn, C. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36, 2024.

Ramesh, A., Dhariwal, P., Nichol, A., Chu, C., and Chen, M. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1(2):3, 2022.

Rombach, R., Blattmann, A., Lorenz, D., Esser, P., and Ommer, B. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10684–10695, 2022.

Shi, W., Han, X., Zhou, C., Liang, W., Lin, X. V., Zettlemoyer, L., and Yu, L. Llamafusion: Adapting pretrained language models for multimodal generation. arXiv preprint arXiv:2412.15188, 2024.

Sun, K., Pan, J., Ge, Y., Li, H., Duan, H., Wu, X., Zhang, R., Zhou, A., Qin, Z., Wang, Y., et al. Journeydb: A benchmark for generative image understanding. Advances in Neural Information Processing Systems, 36, 2024a.

- Sun, P., Jiang, Y., Chen, S., Zhang, S., Peng, B., Luo, P., and Yuan, Z. Autoregressive model beats diffusion: Llama for scalable image generation. arXiv preprint arXiv:2406.06525, 2024b.
- Sun, Q., Cui, Y., Zhang, X., Zhang, F., Yu, Q., Wang, Y., Rao, Y., Liu, J., Huang, T., and Wang, X. Generative multimodal models are in-context learners. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14398–14409, 2024c.

Tang, Z., Yang, Z., Zhu, C., Zeng, M., and Bansal, M. Anyto-any generation via composable diffusion. Advances in Neural Information Processing Systems, 36, 2024.

Team, C. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024.

Team, G., Anil, R., Borgeaud, S., Alayrac, J.-B., Yu, J., Soricut, R., Schalkwyk, J., Dai, A. M., Hauth, A., Millican, K., et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.

Tian, K., Jiang, Y., Yuan, Z., Peng, B., and Wang, L. Visual autoregressive modeling: Scalable image generation via next-scale prediction. arXiv preprint arXiv:2404.02905, 2024a.

Tian, Y., Yang, L., Yang, H., Gao, Y., Deng, Y., Chen, J., Wang, X., Yu, Z., Tao, X., Wan, P., et al. Videotetris: Towards compositional text-to-video generation. arXiv preprint arXiv:2406.04277, 2024b.

Tong, S., Fan, D., Zhu, J., Xiong, Y., Chen, X., Sinha, K., Rabbat, M., LeCun, Y., Xie, S., and Liu, Z. Metamorph: Multimodal understanding and generation via instruction tuning. arXiv preprint arXiv:2412.14164, 2024.

Wang, X., Zhang, X., Luo, Z., Sun, Q., Cui, Y., Wang, J., Zhang, F., Wang, Y., Li, Z., Yu, Q., et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024.

Wu, C., Chen, X., Wu, Z., Ma, Y., Liu, X., Pan, Z., Liu, W., Xie, Z., Yu, X., Ruan, C., et al. Janus: Decoupling visual encoding for unified multimodal understanding and generation. arXiv preprint arXiv:2410.13848, 2024a.

Wu, J., Jiang, Y., Ma, C., Liu, Y., Zhao, H., Yuan, Z., Bai, S., and Bai, X. Liquid: Language models are scalable multimodal generators. arXiv preprint arXiv:2412.04332, 2024b.

Wu, S., Fei, H., Qu, L., Ji, W., and Chua, T.-S. Nextgpt: Any-to-any multimodal llm. arXiv preprint arXiv:2309.05519, 2023.

Wu, Y., Zhang, Z., Chen, J., Tang, H., Li, D., Fang, Y., Zhu, L., Xie, E., Yin, H., Yi, L., et al. Vila-u: a unified foundation model integrating visual understanding and generation. arXiv preprint arXiv:2409.04429, 2024c.

Xie, J., Mao, W., Bai, Z., Zhang, D. J., Wang, W., Lin, K. Q., Gu, Y., Chen, Z., Yang, Z., and Shou, M. Z. Show-o: One single transformer to unify multimodal understanding and generation. arXiv preprint arXiv:2408.12528, 2024a.

Xie, R., Du, C., Song, P., and Liu, C. Muse-vl: Modeling unified vlm through semantic discrete encoding. arXiv preprint arXiv:2411.17762, 2024b.

Yang, L., Yu, Z., Meng, C., Xu, M., Ermon, S., and Bin, C. Mastering text-to-image diffusion: Recaptioning, planning, and generating with multimodal llms. In Forty-first International Conference on Machine Learning, 2024a.

Yang, L., Yu, Z., Zhang, T., Cao, S., Xu, M., Zhang, W., Gonzalez, J. E., and Cui, B. Buffer of thoughts: Thoughtaugmented reasoning with large language models. Advances in Neural Information Processing Systems, 2024b.

Yang, L., Yu, Z., Cui, B., and Wang, M. Reasonflux: Hierarchical llm reasoning via scaling thought templates. arXiv preprint arXiv:2502.06772, 2025a.

Yang, L., Yu, Z., Zhang, T., Xu, M., Gonzalez, J. E., Cui, B., and Yan, S. Supercorrect: Supervising and correcting language models with error-driven insights. In International Conference on Learning Representations, 2025b.

Ye, H., Huang, D.-A., Lu, Y., Yu, Z., Ping, W., Tao, A., Kautz, J., Han, S., Xu, D., Molchanov, P., et al. Xvila: Cross-modality alignment for large language model. arXiv preprint arXiv:2405.19335, 2024.

Yue, X., Ni, Y., Zhang, K., Zheng, T., Liu, R., Zhang, G., Stevens, S., Jiang, D., Ren, W., Sun, Y., et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9556–9567, 2024.

Zhang, D., Lei, J., Li, J., Wang, X., Liu, Y., Yang, Z., Li, J., Wang, W., Yang, S., Wu, J., et al. Critic-v: Vlm critics help catch vlm errors in multimodal reasoning. arXiv preprint arXiv:2411.18203, 2024a.

Zhang, J., Wu, Z., Liang, Z., Gong, Y., Hu, D., Yao, Y., Cao, X., and Zhu, H. Fate: Full-head gaussian avatar with textural editing from monocular video. arXiv preprint arXiv:2411.15604, 2024b.

Zhang, X., Yang, L., Cai, Y., Yu, Z., Wang, K.-N., Tian, Y., Xu, M., Tang, Y., Yang, Y., Bin, C., et al. Realcompo: Balancing realism and compositionality improves textto-image diffusion models. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024c.

Zhang, X., Yang, L., Li, G., Cai, Y., Xie, J., Tang, Y., Yang, Y., Wang, M., and Cui, B. Itercomp: Iterative composition-aware feedback learning from model gallery for text-to-image generation. arXiv preprint arXiv:2410.07171, 2024d.

Zhou, C., Yu, L., Babu, A., Tirumala, K., Yasunaga, M., Shamis, L., Kahn, J., Ma, X., Zettlemoyer, L., and Levy, O. Transfusion: Predict the next token and diffuse images with one multi-modal model. arXiv preprint arXiv:2408.11039, 2024a.

Zhou, Y., Cui, C., Rafailov, R., Finn, C., and Yao, H. Aligning modalities in vision large language models via preference fine-tuning. arXiv preprint arXiv:2402.11411, 2024b.

Zhou, Y., Fan, Z., Cheng, D., Yang, S., Chen, Z., Cui, C., Wang, X., Li, Y., Zhang, L., and Yao, H. Calibrated self-rewarding vision language models. arXiv preprint arXiv:2405.14622, 2024c.

Zhu, D., Chen, J., Shen, X., Li, X., and Elhoseiny, M. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592, 2023.

Zhuang, Y., He, Y., Zhang, J., Wang, Y., Zhu, J., Yao, Y., Zhu, S., Cao, X., and Zhu, H. Towards native generative model for 3d head avatar. arXiv preprint arXiv:2410.01226, 2024.

- A. Derivation of the Pair-DPO Optimization Objective Considering that the optimization objective of standard Direct Preference Optimization is:

πθ(yl|x) πref(yl|x)

πθ(yw |x) πref(yw |x) −β log

(11)

LDPO(θ) = −E(x,y

w,yl)∼D log σ β log

Pair-DPO simultaneously optimizes understanding and generation using pairedcpreference data, with its loss function comprising these two components:

LPair-DPO(θ) = LUnd(θ) + LGen(θ)

πθ(yw |x) πref(yw |x) −β log

πθ(yl|x) πref(yl|x)

πθ(xw |y) πref(xw |y) −β log

πθ(xl|y) πref(xl|y)

= −E(x,y,x

w,xl,yw,yl)∼D log σ β log

+ log σ β log

πθ(yw |x) πref(yw |x) −β log

πθ(xw |y) πref(xw |y) −β log

πθ(yl|x) πref(yl|x)

πθ(xl|y) πref(xl|y)

= −E(x,y,x

w,xl,yw,yl)∼D log σ β log

β log

(12) Here, ∆Und and ∆Gen are defined as:

πθ(yw | x) πref(yw | x) − β log

πθ(yl | x) πref(yl | x)

(13)

∆Und = β log

πθ(xw | y) πref(xw | y) − β log

πθ(xl | y) πref(xl | y)

(14) Substituting these definitions, the final Pair-DPO objective can be expressed as:

∆Gen = β log

w,xl,yw,yl)∼D [log σ (∆Und∆Gen)] (15)

LPair-DPO(θ) = −E(x,y,x

### B. Example of Paired Preference Data

Homologous Input Data:

[Figure 59]

Prompt/Caption: Two samurais, one black and the other white, fiercely engage in combat as the black samurai skillfully blocks an incoming sword attack amidst a city backdrop, creating a scene of intense and tense action.

Image:

Sampled data (n=10): Win Lose

###### Sampled Captions for Understanding:

- Caption1: Two warriors in armor stand in front of a city, holding swords.
- Caption2: Two warriors in armor, one with a sword and the other with a spear, are standing in front of a city.
- Caption3: A character in the game Shadowrun stands on a platform with two other characters.
- Caption4: A scene of two warriors standing in front of a city. One warrior has a sword in each hand, and they are facing each other.
- Caption5: Two warriors in armor, one with a sword and the other with a spear, are standing in front of a city.
- Caption6: Two knights in armor are standing in front of a city.
- Caption7: The two knights are in a dark city, standing on a street with swords in hand. They seem to have just come out from the shadows, ready for battle.
- Caption8: Two warriors in armor, one with a sword and the other with a spear, are standing in front of a city.
- Caption9: Two warriors in armor stand on a platform, holding swords. One of the warriors has a horned helmet, the other has a helmet with horns, city background.
- Caption10: The image shows two knights on the edge of a cliff. They have two swords and are facing off, preparing to fight. Sampled Images for Generation:

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

Image1 Image2 Image3 Image4 Image5

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

Image6 Image7 Image8 Image9 Image10

###### Paired Preference Data:

###### Understanding:

Caption_win: Two warriors in armor stand on a platform, holding swords. One of the warriors has a horned helmet, the other has a helmet with horns, city background. Caption_lose: A character in the game Shadowrun stands on a platform with two other characters.

###### Generation:

[Figure 70]

[Figure 71]

Image_win: Image_lose:

Figure 7. An example of the curation of paired preference data.

