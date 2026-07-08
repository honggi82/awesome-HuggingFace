## Compositional Video Generation as Flow Equalization

Xingyi Yang Xinchao Wang∗ National University of Singapore xyang@u.nus.edu xinchao@nus.edu.sg

VideoCrafterv2 + Vico (Ours)

# arXiv:2407.06182v1[cs.CV]10Jun2024

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

Missing

Subject Semantic

A wildlife conservationist observing the behavior of a leopard in a dense rainforest.

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

Confusion Motion

Leakage Spatial

A camel walking under the sea.

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

A bird and a cat

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

Mixing

A whale flying a hot air balloon over the ocean.

Figure 1: Examples for compositional video generation of Vico on top of VideCrafterv2 [8]. We identify four types of typical failure in compositional T2V (Row 1) Missing Subject (Row 2) Spatial Confusion (Row 3) Semantic Leakage and (Row 4) Motion Mixing. Vico provides a unified solution to these issues by equalizing the contributions of text tokens.

### Abstract

Large-scale Text-to-Video (T2V) diffusion models have recently demonstrated unprecedented capability to transform natural language descriptions into stunning and photorealistic videos. Despite the promising results, a significant challenge remains: these models struggle to fully grasp complex compositional interactions between multiple concepts and actions. This issue arises when some words dominantly influence the final video, overshadowing other concepts. To tackle this problem, we introduce Vico, a generic framework for compositional video generation that explicitly ensures all concepts are represented properly. At its core, Vico analyzes how input tokens influence the generated video, and adjusts the model to prevent any single concept from dominating. Specifically, Vico extracts attention weights from all layers to build a spatial-temporal attention graph, and then estimates the influence as the max-flow from the source text token to the video target token. Although the direct computation of attention flow in diffusion models is typically infeasible, we devise an efficient approximation based on subgraph flows and employ a fast and vectorized implementation, which in turn makes the flow computation manageable and differentiable. By updating the noisy latent

∗Corresponding Author

Preprint. Under review.

to balance these flows, Vico captures complex interactions and consequently produces videos that closely adhere to textual descriptions. We apply our method to multiple diffusion-based video models for compositional T2V and video editing. Empirical results demonstrate that our framework significantly enhances the compositional richness and accuracy of the generated videos. Visit our website at https://adamdad.github.io/vico/.

### 1 Introduction

Humans recognize the world compositionally. That is to say, we perceive and understand the world by identifying parts of objects and assembling them into a whole. This ability to recognize and recombine elements—making “infinite use of finite mean”—is crucial for understanding and modeling our environment. Similarly, in the realm of generative AI, particularly in video generation, it is crucial to replicate this compositional approach.

Despite advancements in generative models, current diffusion models fail to capture the true compositional nature of inputs. Typically, some words disproportionately influence the generative process, leading to visual content that does not reflect the intended composition of elements. While the compositional text-to-image sythesis [32, 5, 29, 14, 23] has been more studied, the challenge of compositional video generation has received less attention. This oversight is largely due to the high-dimensional nature of video and the complex interplay between concepts and motion.

As an illustration, we highlight some failure cases in Figure 1 (Left), where certain words dominate while others are underrepresented. Common issues include missing subject and spatial confusion, where some concepts do not appear in the video. Even with all concepts present, semantic leakage can occur, causing attributes amplified incorrectly, for example, a bird looks like a cat. A challenge specific to T2V is Motion Mixing, where the action intended for one object mistakenly interacts with another, such as generating a flying wale instead of flying balloon.

To address these challenges, we present Vico, a novel framework for compositional video generation that ensures all concepts are represented equally. Vico operates on the principle that, each textual token should have an equal opportunity to influence the final video output. At our core, Vico first assesses and then rebalances the influence of these tokens. This is achieved through test-time optimization, where we assess and adjust the impact of each token at every reverse time step of our video diffusion model. As shown in 1, Vico resolves the above questions and provides better results.

One significant challenge is accurately attributing text influence. While cross-attention [50, 37, 14, 42] provides faithful attribution in text-to-image diffusion models, it is not well-suited for video models. It is because such cross-attention is only applied on spatial modules along, treating each frame independently, without directly influencing temporal dynamics.

To surmount this, we develop a new attribution method for T2V model, termed Spatial-Temporal Attention Flow (ST-flow). ST-flow considers all attention layers of the diffusion model, and views it as a spatiotemporal flow graph. Using the maximum flow algorithm, it computes the flow values, from input tokens (sources) to video tokens (target). These values serve as our estimated contributions.

Unfortunately, this naive attention max-flow computation is, in fact, both computationally expensive and non-differentiable. We thus derive an efficient and differentiable approximation for the ST-Flow. Rather than computing flow values on the full graph, we instead compute the flow on all subgraphs. The ST-Flow is then estimated as the maximum subgraph flow. Additionally, we have develop a special matrix operation to compute this subgraph flow in a fully vectorized manner, making it approximately 100× faster than the exact ST-flow.

Once we obtain these attribution scores, we proceed to optimize the model to balance such contributions. We do this as a min-max optimization, where we update the latent code, in the direct that, the least represented token should increase its influence.

We implement Vico on multiple video applications, including text-to-video generation and video editing. These applications highlight the framework’s flexibility and effectiveness in managing complex prompt compositions, demonstrating significant improvements over traditional methods in both the accuracy of generated video. Our contributions can be summarized below:

- • We introduce Vico, a framework for compositional video generation. It optimizes the model to ensure each input token fairly influences the final video output.
- • We develop ST-flow, a new attribution method that uses attention max-flow to evaluate the influence of each input token in video diffusion models.
- • We derive a differentiable method to approximate ST-flow by calculating flows within subgraphs. It greatly speed up computations with a fully vectorized implementation.
- • Extensive evaluation of Vico in diverse settings has proven its robust capability, with substantial improvements in video quality and semantic accuracy.

### 2 Preliminaries

Denoising Diffusion Probabilistic Models. Diffusion model reverses a progressive noise process based on latent variables. Given data x0 ∼ q(x0) sampled from the real distribution, we consider perturbing data with Gaussian noise of zero mean and βt variance for T steps/ At the end of day, xT → N(0,I) converge to isometric Gaussian noise. The choice of Gaussian provides a close-form solution to generate arbitrary time-step xt through

xt = √α¯tx0 + √1 − α¯tϵ, where ϵ ∼ N(0,I) (1)

where αt = 1 − βt and α¯t = ts=1 αs. A variational Markov chain in the reverse process is parameterized as a time-conditioned denoising neural network ϵθ(x,t) with pθ(xt−1|xt) =

(xt + βtϵθ(x,t)),βtI). The denoiser is trained to minimize a re-weighted evidence lower bound (ELBO) that fits the noise

N(xt−1; √11−β

t

0,ϵ ||ϵ + √1 − α¯tϵθ(x,t)||22 (2)

LDDPM = Et,x

Training with denoising loss, ϵθ equivalently learns to recover the derivative that maximize the data log-likelihood [48, 25, 52]. With a trained ϵθ∗(x,t) ≈ ∇xt

log p(xt), we generate the data by reversing the Markov chain

1 √1 − βt

(xt + βtϵθ∗(x,t)) + βtϵt; (3)

xt−1 ←

The reverse process could be understood as going along ∇xt

log p(xt) to maximize the likelihood.

Text-to-Video (T2V) Diffusion Models. Given a text prompt y, T2V diffusion models progressively generate a video from Gaussian noise. This generation typically occurs within the latent space of an autoencoder [43] to reduce the complexity. The architecture design of T2V models often follows either a 3D-UNet [22, 4, 21, 19, 60] or diffusion transformer [17, 40, 36]. For computational efficiency, these architectures commonly utilize separate self-attention [51] for spatial and temporal tokens. Moreover, cross-attentions is applied on each frame separately, thereby injecting conditions into the model. More related work is in Appendix C.

Maximum-Flow Problem. [18, 15, 12] Consider a directed graph G(V,E) with a source node s and a target node t. A flow is function on edge f : E → R that satisfies both conservation constraint and capacity constraint at every vertex v ∈ V \{s,t}. This means the total inflow into any node v must equals its total outflow, and the flow on any edge cannot exceed its capacity. The flow value |f| = e

s,v∈E f(s,u) is defined as the total flow out of the source s, which is equal to the total inflow into the target t, |f| = e

u,t∈E f(u,t). The maximum flow problem is to find a flow f∗ that maximizes this value.

### 3 Vico: Compositional Video Generation as Flow Equalization

In this paper, we solve the problem of compositional video generation by equalizing influence among tokens. We calculate this influence using max-flow within the attention graph of the T2V model and ensure efficient computation. We define our problem and optimization scheme in Sec 3.1. The definition of ST-Flow and its efficient computation are discussed in Sections 3.2 and 3.3.

[Figure 21]

Cross-Attn(SpatialOnly) Denoising Step Flow Equalization Step Self-Attn

Target

𝐱 Updated 𝐱

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

spatial temporal

[Figure 30]

[Figure 31]

ST-Flow min 𝐴

max

[Figure 32]

[Figure 33]

𝐱

[Figure 34]

Three men holding handbags are walking.

Source

Attentions as Graph

|men|
|---|

|hold|
|---|

|handbag|
|---|

|walk|
|---|

Figure 2: Overall pipeline of our Vico. Before each denoising step, Vico extracts attention maps from each layer to build a spatiotemporal graph. We calculate the attribution scores as max-flow in the graph and adjust the noisy latent code to balance this flows.

#### 3.1 Overall Pipeline and Optimization

Our objective is to generate a video from an input prompt P, with K text tokens of interest V = {v1,...,vK}. We aim to ensure that each token fairly contributes to the final video. This process is detailed in Figure 2.

Objectives. To achieve this, we define an attribution function Ai = A(vi) ∈ R for each token vi, quantifying its impact on the video. We optimize the attribution scores to ensure fairness:

{A1,...,AK}; (4) Here, Lfair = minv

Lfair(A1,...,AK) = max

max

min

xt

xt

vi

i{A1,...,AK} serves as the fairness function, focusing on the least represented token. By updating the noisy latent xt to maximize Lfair, we ensure equal contributions across all tokens. Specifically, we estimate Ai as flow in attention graph, which will be discussed later.

Optimization. To implement Eq 4, we perform test-time optimization. Before each denoising step, we first feed xt into the model, extract the Ai, and update xt through gradient ascent: xˆt ← xt + η∇xtLfair(A1,...,AK). η is the step size. Then, xˆt is going through a denoising step to get xt−1 according to Eq 3. We repeat these steps until the video is generated.

#### 3.2 Attention Flow Across Space and Time

With above formulation, our focus is to develop an efficient and precise attribution Ai. Recognizing issues with cross-attention, we instead calculate Ai as the flow through the entire attention graph.

Flawed Cross-Attention in Text-to-Video Models. Cross-attention score has been instrumental in attributing [50] and controlling layout and concept composition in text-to-image models [20, 5, 42]. However, applying it to T2V diffusion model introduces new problem.

This problem arises because T2V models typically employ cross-attention on spatial tokens only [53, 7, 55]. It treats the video as a sequence of independent images, and temporal self-attention mixes tokens across different frames. Consequently, this separation hinders cross-attention’s ability to capture video dynamics, making it challenging to manage actions across frames.

For example, applying the cross-attention-based DAAM attribution [50] on VideoCrafterv2 reveals significant issues in visualization. As shown in Figure 3 (Left), cross-attention leads to a flickering pattern in the attention maps, failing to consistently highlight the target object across frames.

Recognizing these limitations, we propose a new measurement termed Spatial-Temporal Flow (STFlow), which estimates the influence throughout the entire spatiotemporal attention graph in the video diffusion model. As seen in Figure 3 (Right), ST-Flow gets heatmap with improved consistency.

Attention as a Graph Over Space and Time. In our approach, we conceptualize the stacked attention layers as a directed graph G = (V,E), where nodes represent tokens and edges weighted by the influence between tokens. A 4-layer example is illustrated in Figure 2 (Right).

Its adjacency matrix is built using attention weights and skip connections [1]. Suppose wi,jatt is the i-th row j-th column element of attention matrix averaged across heads. For self-attention, the edge

VideoCrafterv2

DAAM (Cross-Attention Only) ST-Flow (Cross & Self Attention)

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

kitten

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

meadow

A playful kitten chasing a butterfly in a wildflower meadow

Animate-Diff

DAAM (Cross-Attention Only) ST-Flow (Cross & Self Attention)

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

dog

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

running

A joyful dog is running gracefully, adorned with a regal crown and stylish glasses.

Figure 3: Attribution heatmap comparison between DAAM and ST-Flow.

weight ei,j between any two tokens, i and j, is ei,j = wi,jatt + 1 if i = j, indicating a skip-connection, and ei,j = wi,jatt if i ̸= j. In the case of cross-attention, edge ei,j = wi,jatt connects text to video, and ei,i = 1 for connections within video tokens due to skip connections. Given that connections only exist from one layer to the next, the resulting matrix exhibits block-wise sparsity pattern. This is

  

  . Here, Ei denotes the edge weights within video,

0 Et,1 0 ... Et,l 0 0 E2 ... 0 0 0 0 ... . 0 0 0 ... El

expressed as W =

and Et,i indicates the influence from text to video at each cross-attention layer.

Attribution as Flow on Graph. Given graph G, we compute the attribution Ai by analyzing all paths from a text token vi to video tokens at the output layer. As such, we formulate it as a max-flow problem with capacity matrix W. To facilitate this, we add an auxiliary target node vt to G, connecting it to all output video tokens with inflow edges ev+

= 12. We treat each text token vi as the source, and vt as the sink. The max-flow from source to sink quantifies the influence of vi, termed ST-Flow.

t

- Definition 1 (ST-Flow). In attention graph G with capacity matrix W, a input token vi as source and sink node vt, the attribution value of Ai = |f|∗ is computed as the maximum flow from vi to vt.

Our ST-Flow can be considered as an extension of Attention Flow [1], incorporating all attention layers in diffusion model. It is proved to be a kind of Shapley Value [13], which is an ideal contribution allocation in game theory [45, 39, 64] and interpretable AI model [34, 49].

Exact ST-Flow Computation is infeasible. While theoretically possible, calculating the ST-Flow in T2V diffusion models faces practical issues that render it infeasible:

- • Non-Differentiable. The max-flow algorithm, by its nature, is non-differentiable. This is a problem when we do gradient-based optimization in Eq 4.
- • Efficiency Issue. Solving max-flow for each input token is slow. Even with the Dinic’s algorithm [9] 3, the time complexity is O(K|V |2|E|) for large attention graphs extracted from video diffusion model.

Despite these obstacles, in Sec 3.3, we derive a min-max approximation to circumvent these issues.

2The maximum inflow is 1 for each node due to softmax normalization in the attention. 3Given that the attentions has more edge than tokens, Dinic is best choice in theory. However, our implemen-

tation shows that max-flow on each token takes ∼8s.

#### 3.3 Differentiable ST-Flow with Min-Max Path Flow

As discussed above, exact computation of ST-Flow is challenging. Instead of directly estimating the ST-Flow, we approach this by focusing on approximating its lower bound, which is computationally feasible. This is made possible, since any sub-graph has max-flow smaller than that of full graph.

Theorem 1 (Sub-Graph Flow)4. For any sub-graph g of a graph G, g ⊆ G, the maximum flow fg∗ in g is less than or equal to the maximum flow fG∗ in G, |fg∗| ≤ |fG∗|.

Based on this theorem, we need not compute the ST-Flow directly. Instead, we sample multiple subgraphs g from G, calculate the maximum flow for each, and take the highest value among these:

|fg|; (5)

|fG| ≥ Ai = max ∀g⊆G

This approach allows for a more efficient calculation by focusing on a manageable number of subgraphs, solving the max-flow for each, and identifying the maximum flow.

In this work, we focus on the simplest type of subgraph in graph G: a path from a vi to target vt. We efficiently approximate the ST-Flow by computing the max path flow for each path. We propose two min-max strategies to achieve this:

- • Hard Flow Strategy. For each text token v, we sample all paths vi to vt. The max-flow on each path is calculated as the minimum edge capacity along the path, |f| = minj ej. And the best approximated Ai = max|f| is the maximum of these minimums across all paths.
- • Soft Flow Strategy. Instead of get the hard min-max flow, we use soft-min and soft-max operations using the log-sum-exp trick. This approach provides a smoother approximation of flow values, which can be especially useful in our gradients-based optimization. The soft-min/max is computed as below, with τ as a temperature

ej τ

) ; (6)

softmax(e1, e2, . . . ; τ) = τ log

exp (

j

softmin(e1, e2, . . . ; τ) = −softmax(−e1, −e2, . . . ; τ), (7)

Vectorized Path Flow Computation. While depth-first and breadth-first searches can identify all paths for above min-max optimization, these methods are slow and cannot be parallelized. Instead, we define a special operation called min-max multiplication on the capacity matrix to calculate the maximum flow for each path in a vectorized manner.

- Definition 2 (Min-max Multiplication). Given two matrices A ∈ Rm×k and B ∈ Rk×n, min-max multiplication C = A ⊙ B ∈ Rm×n is defined where each element Ci,j = maxr(min(Ai,r,Br,j)).

This operation computes the minimum value across all r for the i-th row of A and the j-th column of B, and maxr selects the maximum of these minimum values for each Ci,j. We call it a multiplication because it resembles matrix multiplication but replaces element-wise multiplication with a minimum operation and summation with maximization.

A very good property is that, the min-max multiplication of capacity matrix Wk = Wk−1 ⊙ W can be interpreted as the max path flow for all k-hop paths.

Proposition 1 (Max Path Flow using Min-max Multiplication)5. For min-max power of capacity Wk = Wk−1 ⊙ W, element Wi,jk equals the max path flow for all k-hop path from vi to vj.

For attention graph that current layer’s node is only connect to the next layer, all path from text token to output video token has exactly the length of l. In this way, what we do is just to extract the attention graph G, do l times Min-max Multiplication on its flow matrix, and we consider the value as a approximation of ST-Flow. A tine complexity analysis is prepared in Appendix F.

In this way, we get all pieces to build Vico. We first compute attribution using the approximated ST-Flow, then using Eq 4 to update the latent to equalize such flow.

- 4Proof in Appendix A
- 5Proof in Appendix B

Table 1: Quantitative results for different methods on compositional text-to-video generation.

Name Spatial Relation↑ Multiple Object↑ Motion Composition↑ Overall Consistency↑

AnimateDiff [16] 24.80% 33.44% 33.90% 27.75% +Compositional Diffusion [32] 19.43% 7.27% 23.58% 24.07% +Attend-and-Excite [5] 20.88% 31.25% 34.78% 28.05% +Token-Reweight 28.11% 36.89% 37.45% 26.77%

+Vico (hard) 24.22% 29.95% 37.23% 28.85% +Vico (soft) 31.47% 37.20% 37.95% 28.89%

ZeroScopev2 59.52% 52.52% 45.51% 25.83% +Compositional Diffusion [32] 31.77% 8.23% 33.13% 23.02% +Token-Reweight 57.48% 50.00% 40.42% 25.74% +Attend-and-Excite [5] 59.02% 62.27% 45.82% 25.84%

+Vico (hard) 63.60% 63.34% 46.32% 24.89% +Vico (soft) 62.28% 69.05% 45.31% 26.15%

VideoCrafterv2 [7] 35.86% 40.66% 43.82% 28.06% +Compositional Diffusion [32] 23.61% 10.59% 35.49% 24.49% +Token-Reweight 46.08% 49.16% 44.33% 28.29% +Attend-and-Excite [5] 48.11% 66.62% 43.48% 28.33%

+Vico (hard) 49.85% 67.84% 44.46% 28.41% +Vico (soft) 50.40% 73.55% 44.98% 28.52%

### 4 Experiments

In our experiments section, we evaluate Vico through a series of tests. We start by assessing its performance on generating videos from compositional text prompts. Next, we demonstrate ST-Flow accurately attributes token influence through video segmentation and human study. We also conduct an ablation study to validate our key designs. More application results are provide in Appendix D.

#### 4.1 Experiment Setup

Baselines. We build our method on several open-sourced video diffusion model, including VideoCrafterv2 [8], AnimateDiff [16] and Zeroscopev2 6. Since no current video diffusion models specifically focus on generation with compositional conditions, we re-implement several methods designed for text-to-image diffusion models and compare with them. These methods include:

- • Original Model. We directly ask the original base model to produce video based on prompts.
- • Token Re-weight. We use the compel 7 package to directly up-lift the weight of specific concept token, with a fixed weight of 1.5.
- • Compositional Diffusion [32]. This method directly make multiple noise predictions on different text, and sum the noise prediction as the compositional direction for latent update. In our paper, given a prompt, we first split into short phrases. For example “a dog and a cat” is splitted into “a dog” and “a cat”, make individual denoising, and added up.
- • Attend-and-Excite [5]. A&E refines the noisy latents to excite cross-attention units to attend to all subject tokens in the text prompt.

Evaluation and Metrics. We evaluate compositional generation using VBench [24]. Specifically, we focus on evaluating compositional quality in terms of Spatial Relation, Multiple Object Composition. For both metrics, the model processes text containing multiple concepts, generates a video. Then a caption model verifies the accuracy of the concept representations within the generated video.

Additionally, we design a new metric, Motion Composition. This metric evaluates the generated video based on the presence and accuracy of multiple objects performing different motions. We collect 70 prompts of the form "obj1 is motion1 and obj2 is motion2". Using GRiT [61], we generate dense captions on video for each object and verify if each (object, motion) pair appears in the captions.

The score is computed as 1,2(I(obji)+4I(obji,motioni)). Here, I(x) is an indicator function that returns 1 if x is present in the generated captions, and 0 otherwise.

The overall video quality is measured using ViCLIP [56] to compute a score based on text and video alignment, denoted as Overall Consistency.

- 6https://huggingface.co/cerspense/zeroscope_v2_576w
- 7https://github.com/damian0815/compel

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

Crafterv2

And-Excite Video

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

+Attention-

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

(Ours)

+Vico

A crab DJing at a beach party during sunset.

A dog and a horse

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

Crafterv2

And-Excite Video

+Attention-

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

(Ours)

+Vico

A spiderman panda is surfing on waves.

A falcon as a messenger in a sprawling medieval city.

Figure 4: Qualitative comparison of the videos generated by VideoCrafterv2 baseline, Attribute&Excite and our Vico with compositional textual descriptions.

Implementation Details. We use the implementations on diffusers for video generation. All videos are generated by a A6000 GPU. We sample videos from Zeroscopev2 and VideoCrafterv2 using 50-step DPM-Solver++ [33]. AnimateDiff is sampled with 50-step DDIM [47]. We optimize the latent at each sampling steps, and update the latent with Adam [28] optimizer at the learning rate of 1e − 5. We test both the soft and hard-min/max versions of Vico, setting the temperature τ = 0.01 for the soft version. The NLTK package identify all nouns and verbs for equalization.

#### 4.2 Compositional Video Generation

Quantitative Results. In Table 1, we present the scores achieved by Vico compared to other methods across various base models on compositional text-to-video generation. Vico consistently surpasses all baselines on every metric. Notably, our ST-flow based method surpasses cross-attention based techniques like Attend&Excite, thanks to its ability to incorporating influences across full attention graph. Additionally, the soft min-max version of Vico generally achieves better fidelity than the hard version, as it is better suited for gradient optimization.

Specifically, Vico demonstrates its most significant improvements in multi-subject generation tasks. For instance, on VideoCrafterv2, it shows a remarkable increase, improving scores from 40.66% → 73.55%. This suggests that our attention mechanism in T2V is more adept at managing object arrangement. In contrast, compositional diffusion models [32] often fail, as they assume conditions to be independent, which is problematic for realistic condition compositions.

Qualitative Results. We compare the videos generated by different methods in Figure 4. Attend&Excite receive slightly improvements, but still mixes semantics of different subject. For example, on the “a dog and a horse” example (Top Left), both Attend&Excite and the baseline incorrectly combine a dog’s face with a horse’s body. Vico addresses this issue by ensuring each token contributes equally, effectively separating their relationships.

Additionally, cross-attention often leads to temporal inconsistencies in the modified videos. For instance, in the “spider panda” case (Bottom Left), Attend&Excite initially displays a Spider-Man logo but it disappears abruptly in subsequent frames. In contrast, Vico captures dynamics across both spatial and temporal attention, leading to better results. More results is in Appendix D and E.

Attribution Method Temporal Consistency↑ Reasonability↑

Cross-Attention [50] 2.56 2.84 Attention Rollout [1] 3.88 3.45

ST-Flow (Ours) 4.19 3.78

- Table 2: User study on attribution method.

Method

Ref-DAVID2017

J &F ↑ J ↑ F ↑ Supervised Trained ReferFormer-B [62] 61.1 58.1 64.1 OnlineRefer-B [59] 62.4 59.1 65.6 Zero-Shot

Cross-Attention [50] mean 32.1 29.8 34.7 Attention Rollout [1] mean 38.0 33.3 40.0 ST-Flow (Ours) mean 38.2 33.5 40.3

- Table 3: Performance on Ref-DAVID2017.

Min Loss ST-Flow (soft) Multiple Object↑ Overall Consistency↑

- ✗ ✗ 57.86% 28.03%

✓ ✗ 63.62% 28.24%

- ✗ ✓ 69.75% 28.12%

###### ✓ ✓ 73.55% 28.52%

##### Table 4: Ablation study on Vico.

[Figure 103]

[Figure 104]

[Figure 105]

Cross-AttentionST-Flow(Ours)

Frame 1 Frame 5 Frame 24

[Figure 106]

[Figure 107]

[Figure 108]

Frame 1 Frame 5 Frame 24

Table 5: Segmentation results comparison.

#### 4.3 Attribution on Video Diffusion Model

In this section, we aim to demonstrate that our ST-Flow (hard) provides a more accurate measure of token contribution compared to other attention-based indicators.

Objective Evaluation: Zero-shot Video Segmentation. We test several attribution methods using the VideoCrafterv2 model for zero-shot video segmentation on the Ref-DAVIS2017 [27] dataset. To create these maps, we first performed a 25-step DDIM inversion [38] to extract noise patterns, followed by sampling to generate the attribution maps. We specifically use maps from from end of text ([EOT]) token [30] for segmentation. We used the mean value of the map as a threshold for binary segmentation. We compare with cross-attention [50] and Attention Rollout [1]. The more accurate the segmentation is, the attribution is more reasonable for human.

Results are presented in Table 3. Our method outperform others, providing the highest segmentation metrics in zero-shot setting. As visualized in Figure 5, cross-attention maps show inconsistent highlighting and flickering. Attention Rollout also conciders the full attention graph, but overly smoothed weights, resulting in less precise object focus.

Subjective Evaluation: User Study. Besides, segmentation-based validation, we conducted a subjective user study to evaluate the quality of attribution maps generated by various methods. 20 participants rated maps from three different approaches across ten video clips. The evaluation focused on Temporal Consistency, assessing the presence of flickering, and Reasonability, determining alignment with human interpretations. Ratings ranged from 1 to 5, with 5 as the highest. As summarized in Table 2, Our ST-Flow outperforms other attention-based explanation, achieving the highest scores in both Temporal Consistency (4.19) and Reasonability (3.78).

#### 4.4 Ablation Study

In this section, we ablate our two key designs: the loss function and the proposed ST-Flow. We do experiments on VideoCrafterv2 and compare the performance.

Loss Function. We modify the loss function used, from using the “min” as a fairness indicator (as described in Sec 3.1) to a variance loss, defined as Lfair = − i(Ai − A¯)2. It minimizes the differences between each Ai and the average attribution value A¯, making it fair. The results are shown in Table 4, row 3 and 4. We notice while the variance loss ensures uniformity across all tokens, it overly restricts them, often degrading video quality. Conversely, our min-loss focuses on the least represented token, enhancing object composition accuracy without significantly affecting overall quality.

ST-Flow v.s. Cross-Attention. A major contribution of our work is the development of ST-Flow and its efficient computation. We compare it against a variant using cross-attention as attribution. In this method, cross-attention maps are extracted and a mean score is computed for each token as Ai. As demostrated in Table 4, row 2 and 4, using ST-Flow (soft) largely outperforms cross-attention. We also provide the running speed analysis in Appendix F, confirming the efficiency of our approach.

### 5 Conclusion

In this paper, we present Vico, a framework designed for compositional video generation. Vico starts by analyzing how input tokens influence the generated video. It then adjusts the model to ensure that no single concept dominates. To implement Vico practically, we calculate each text token’s contribution to the video token using max flow. This computation is made feasible by approximating the subgraph flow with a vectorized implementation. We have applied our method across various diffusion-based video models, which has enhanced both the visual fidelity and semantic accuracy of the generated videos.

### References

- [1] Samira Abnar and Willem H. Zuidema. Quantifying attention flow in transformers. In Dan Jurafsky, Joyce Chai, Natalie Schluter, and Joel R. Tetreault, editors, Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, ACL 2020, Online, July 5-10, 2020, pages 4190–4197. Association for Computational Linguistics, 2020.
- [2] Amir Bar, Roei Herzig, Xiaolong Wang, Anna Rohrbach, Gal Chechik, Trevor Darrell, and Amir Globerson. Compositional video synthesis with action graphs. In Marina Meila and Tong Zhang, editors, Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pages 662–673. PMLR, 18–24 Jul 2021.
- [3] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023.
- [4] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22563– 22575, 2023.
- [5] Hila Chefer, Yuval Alaluf, Yael Vinker, Lior Wolf, and Daniel Cohen-Or. Attend-and-excite: Attentionbased semantic guidance for text-to-image diffusion models. ACM Trans. Graph., 42(4), jul 2023.
- [6] Hila Chefer, Shir Gur, and Lior Wolf. Generic attention-model explainability for interpreting bi-modal and encoder-decoder transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 397–406, 2021.
- [7] Haoxin Chen, Menghan Xia, Yingqing He, Yong Zhang, Xiaodong Cun, Shaoshu Yang, Jinbo Xing, Yaofang Liu, Qifeng Chen, Xintao Wang, et al. Videocrafter1: Open diffusion models for high-quality video generation. arXiv preprint arXiv:2310.19512, 2023.
- [8] Haoxin Chen, Yong Zhang, Xiaodong Cun, Menghan Xia, Xintao Wang, Chao Weng, and Ying Shan. Videocrafter2: Overcoming data limitations for high-quality video diffusion models. arXiv preprint arXiv:2401.09047, 2024.
- [9] Efim A Dinic. Algorithm for solution of a problem of maximum flow in networks with power estimation. In Soviet Math. Doklady, volume 11, pages 1277–1280, 1970.
- [10] Yilun Du, Conor Durkan, Robin Strudel, Joshua B Tenenbaum, Sander Dieleman, Rob Fergus, Jascha Sohl-Dickstein, Arnaud Doucet, and Will Sussman Grathwohl. Reduce, reuse, recycle: Compositional generation with energy-based diffusion models and mcmc. In International conference on machine learning, pages 8489–8510. PMLR, 2023.
- [11] Yilun Du, Shuang Li, and Igor Mordatch. Compositional visual generation with energy based models. Advances in Neural Information Processing Systems, 33:6637–6647, 2020.
- [12] Jack Edmonds and Richard M Karp. Theoretical improvements in algorithmic efficiency for network flow problems. Journal of the ACM (JACM), 19(2):248–264, 1972.
- [13] Kawin Ethayarajh and Dan Jurafsky. Attention flows are shapley value explanations. In Chengqing Zong, Fei Xia, Wenjie Li, and Roberto Navigli, editors, Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing, ACL/IJCNLP 2021, (Volume 2: Short Papers), Virtual Event, August 1-6, 2021, pages 49–54. Association for Computational Linguistics, 2021.

- [14] Weixi Feng, Xuehai He, Tsu-Jui Fu, Varun Jampani, Arjun Reddy Akula, Pradyumna Narayana, Sugato Basu, Xin Eric Wang, and William Yang Wang. Training-free structured diffusion guidance for compositional text-to-image synthesis. In The Eleventh International Conference on Learning Representations, 2023.
- [15] Lester Randolph Ford and Delbert R Fulkerson. Maximal flow through a network. Canadian journal of Mathematics, 8:399–404, 1956.
- [16] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized text-to-image diffusion models without specific tuning. International Conference on Learning Representations, 2024.
- [17] Agrim Gupta, Lijun Yu, Kihyuk Sohn, Xiuye Gu, Meera Hahn, Li Fei-Fei, Irfan Essa, Lu Jiang, and José Lezama. Photorealistic video generation with diffusion models. CoRR, abs/2312.06662, 2023.
- [18] TE Harris and FS Ross. Fundamentals of a method for evaluating rail net capacities. Technical report, Rand Corporation, 1955.
- [19] William Harvey, Saeid Naderiparizi, Vaden Masrani, Christian Dietrich Weilbach, and Frank Wood. Flexible diffusion modeling of long videos. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho, editors, Advances in Neural Information Processing Systems, 2022.
- [20] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-toprompt image editing with cross attention control. arXiv preprint arXiv:2208.01626, 2022.
- [21] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey A. Gritsenko, Diederik P. Kingma, Ben Poole, Mohammad Norouzi, David J. Fleet, and Tim Salimans. Imagen video: High definition video generation with diffusion models. CoRR, abs/2210.02303, 2022.
- [22] Jonathan Ho, Tim Salimans, Alexey A. Gritsenko, William Chan, Mohammad Norouzi, and David J. Fleet. Video diffusion models. In Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, 2022.
- [23] Kaiyi Huang, Kaiyue Sun, Enze Xie, Zhenguo Li, and Xihui Liu. T2i-compbench: A comprehensive benchmark for open-world compositional text-to-image generation. Advances in Neural Information Processing Systems, 36:78723–78747, 2023.
- [24] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. VBench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.
- [25] Aapo Hyvärinen and Peter Dayan. Estimation of non-normalized statistical models by score matching. Journal of Machine Learning Research, 6(4), 2005.
- [26] Levon Khachatryan, Andranik Movsisyan, Vahram Tadevosyan, Roberto Henschel, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. Text2video-zero: Text-to-image diffusion models are zero-shot video generators. arXiv preprint arXiv:2303.13439, 2023.
- [27] Anna Khoreva, Anna Rohrbach, and Bernt Schiele. Video object segmentation with language referring expressions. In Computer Vision–ACCV 2018: 14th Asian Conference on Computer Vision, Perth, Australia, December 2–6, 2018, Revised Selected Papers, Part IV 14, pages 123–141. Springer, 2019.
- [28] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.
- [29] Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu. Multi-concept customization of text-to-image diffusion. 2023.
- [30] Senmao Li, Joost van de Weijer, taihang Hu, Fahad Khan, Qibin Hou, Yaxing Wang, and jian Yang. Get what you want, not what you don’t: Image content suppression for text-to-image diffusion models. In The Twelfth International Conference on Learning Representations, 2024.
- [31] Nan Liu, Yilun Du, Shuang Li, Joshua B Tenenbaum, and Antonio Torralba. Unsupervised compositional concepts discovery with text-to-image generative models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2085–2095, 2023.

- [32] Nan Liu, Shuang Li, Yilun Du, Antonio Torralba, and Joshua B Tenenbaum. Compositional visual generation with composable diffusion models. In European Conference on Computer Vision, pages 423–439. Springer, 2022.
- [33] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver++: Fast solver for guided sampling of diffusion probabilistic models. arXiv preprint arXiv:2211.01095, 2022.
- [34] Scott M Lundberg and Su-In Lee. A unified approach to interpreting model predictions. Advances in neural information processing systems, 30, 2017.
- [35] Scott M Lundberg and Su-In Lee. A unified approach to interpreting model predictions. In I. Guyon, U. Von Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017.
- [36] Xin Ma, Yaohui Wang, Gengyun Jia, Xinyuan Chen, Ziwei Liu, Yuan-Fang Li, Cunjian Chen, and Yu Qiao. Latte: Latent diffusion transformer for video generation. CoRR, abs/2401.03048, 2024.
- [37] Ron Mokady, Amir Hertz, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Null-text inversion for editing real images using guided diffusion models. arXiv preprint arXiv:2211.09794, 2022.
- [38] Ron Mokady, Amir Hertz, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Null-text inversion for editing real images using guided diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6038–6047, 2023.
- [39] Roger B Myerson. Graphs and cooperation in games. Mathematics of operations research, 2(3):225–229, 1977.
- [40] William Peebles and Saining Xie. Scalable diffusion models with transformers. In IEEE/CVF International Conference on Computer Vision, ICCV 2023, Paris, France, October 1-6, 2023, pages 4172–4182. IEEE, 2023.
- [41] Harish Guruprasad Ramaswamy et al. Ablation-cam: Visual explanations for deep convolutional network via gradient-free localization. In proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 983–991, 2020.
- [42] Royi Rassin, Eran Hirsch, Daniel Glickman, Shauli Ravfogel, Yoav Goldberg, and Gal Chechik. Linguistic binding in diffusion models: Enhancing attribute correspondence through attention map alignment. Advances in Neural Information Processing Systems, 36, 2024.
- [43] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.
- [44] Ramprasaath R Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, and Dhruv Batra. Grad-cam: Visual explanations from deep networks via gradient-based localization. In Proceedings of the IEEE international conference on computer vision, pages 618–626, 2017.
- [45] Lloyd S Shapley et al. A value for n-person games. 1953.
- [46] Karen Simonyan, Andrea Vedaldi, and Andrew Zisserman. Deep inside convolutional networks: Visualising image classification models and saliency maps. arXiv preprint arXiv:1312.6034, 2013.
- [47] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020.
- [48] Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. Advances in neural information processing systems, 32, 2019.
- [49] Mukund Sundararajan, Ankur Taly, and Qiqi Yan. Axiomatic attribution for deep networks. In International conference on machine learning, pages 3319–3328. PMLR, 2017.
- [50] Raphael Tang, Linqing Liu, Akshat Pandey, Zhiying Jiang, Gefei Yang, Karun Kumar, Pontus Stenetorp, Jimmy Lin, and Ferhan Ture. What the DAAM: Interpreting stable diffusion using cross attention. In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki, editors, Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 5644–5659, Toronto, Canada, July 2023. Association for Computational Linguistics.
- [51] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

- [52] Pascal Vincent. A connection between score matching and denoising autoencoders. Neural computation, 23(7):1661–1674, 2011.
- [53] Jiuniu Wang, Hangjie Yuan, Dayou Chen, Yingya Zhang, Xiang Wang, and Shiwei Zhang. Modelscope text-to-video technical report. arXiv preprint arXiv:2308.06571, 2023.
- [54] Xiang Wang, Hangjie Yuan, Shiwei Zhang, Dayou Chen, Jiuniu Wang, Yingya Zhang, Yujun Shen, Deli Zhao, and Jingren Zhou. Videocomposer: Compositional video synthesis with motion controllability. Advances in Neural Information Processing Systems, 36, 2024.
- [55] Yaohui Wang, Xinyuan Chen, Xin Ma, Shangchen Zhou, Ziqi Huang, Yi Wang, Ceyuan Yang, Yinan He, Jiashuo Yu, Peiqing Yang, et al. Lavie: High-quality video generation with cascaded latent diffusion models. arXiv preprint arXiv:2309.15103, 2023.
- [56] Yi Wang, Yinan He, Yizhuo Li, Kunchang Li, Jiashuo Yu, Xin Ma, Xinhao Li, Guo Chen, Xinyuan Chen, Yaohui Wang, et al. Internvid: A large-scale video-text dataset for multimodal understanding and generation. arXiv preprint arXiv:2307.06942, 2023.
- [57] Zhao Wang, Aoxue Li, Enze Xie, Lingting Zhu, Yong Guo, Qi Dou, and Zhenguo Li. Customvideo: Customizing text-to-video generation with multiple subjects. arXiv preprint arXiv:2401.09962, 2024.
- [58] Yujie Wei, Shiwei Zhang, Zhiwu Qing, Hangjie Yuan, Zhiheng Liu, Yu Liu, Yingya Zhang, Jingren Zhou, and Hongming Shan. Dreamvideo: Composing your dream videos with customized subject and motion. arXiv preprint arXiv:2312.04433, 2023.
- [59] Dongming Wu, Tiancai Wang, Yuang Zhang, Xiangyu Zhang, and Jianbing Shen. Onlinerefer: A simple online baseline for referring video object segmentation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2761–2770, 2023.
- [60] Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Stan Weixian Lei, Yuchao Gu, Yufei Shi, Wynne Hsu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. Tune-a-video: One-shot tuning of image diffusion models for text-to-video generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7623–7633, 2023.
- [61] Jialian Wu, Jianfeng Wang, Zhengyuan Yang, Zhe Gan, Zicheng Liu, Junsong Yuan, and Lijuan Wang. Grit: A generative region-to-text transformer for object understanding. arXiv preprint arXiv:2212.00280, 2022.
- [62] Jiannan Wu, Yi Jiang, Peize Sun, Zehuan Yuan, and Ping Luo. Language as queries for referring video object segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4974–4984, 2022.
- [63] Qiucheng Wu, Yujian Liu, Handong Zhao, Trung Bui, Zhe Lin, Yang Zhang, and Shiyu Chang. Harnessing the spatial-temporal attention of diffusion models for high-fidelity text-to-image synthesis. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7766–7776, 2023.
- [64] H Peyton Young. Monotonic solutions of cooperative games. International Journal of Game Theory, 14(2):65–72, 1985.
- [65] Matthew D Zeiler and Rob Fergus. Visualizing and understanding convolutional networks. In Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part I 13, pages 818–833. Springer, 2014.
- [66] Daquan Zhou, Weimin Wang, Hanshu Yan, Weiwei Lv, Yizhe Zhu, and Jiashi Feng. Magicvideo: Efficient video generation with latent diffusion models. arXiv preprint arXiv:2211.11018, 2022.

### A Proof of Theorem 1: Sub-graph Flow

In a network G = (V,E) with a capacity function c : E → R+, and a subgraph g of G, the maximum flow fg in g is less than or equal to the maximum flow fG in G.

#### Proof

- 1. Definition of a Subgraph: A subgraph g of G can be defined as g = (V ′,E′) where V ′ ⊆ V and E′ ⊆ E. All capacities in g are inherited from G, i.e., c′(e) = c(e) for all e ∈ E′.
- 2. Flow Conservation: Both G and g must satisfy the flow conservation law at all intermediate nodes. That is, the sum of the flow entering any node must equal the sum of the flow exiting that node, except for the source (where flow is generated) and the sink (where flow is absorbed).
- 3. Reduced Set of Paths: Since E′ ⊆ E, every path through g is also a path through G, but not every path through G is necessarily a path through g. This reduction in the number of paths (or edges) in g implies that some routes available for flow in G are not available in g.
- 4. Capacity Limitations: For any edge e in E′, the capacity in g (i.e., c′(e)) equals the capacity in G (i.e., c(e)). Therefore, no edge in g can support more flow than it can in G. Additionally, since some edges might be missing in g, the overall capacity of pathways from the source to the sink might be less in g than in G.
- 5. Maximum Flow Reduction: Given the reduction in paths and capacities, any flow that is

feasible in g is also feasible in G, but not vice versa. Hence, the maximum flow fg that can be pushed from the source to the sink in g must be less than or equal to the maximum flow

fG that can be pushed in G.

Conclusion: From these points, it follows directly that the maximum flow in a subgraph g cannot exceed the maximum flow in the original graph G. This proves that fg ≤ fG.

### B Proof of Proposition 1: Max Path Flow using Min-max Multiplication

Definitions and Proposition: Let W be a capacity matrix of a graph where Wi,j is the capacity of the edge from vertex i to vertex j. If there is no edge between i and j, Wi,j = 0 or some representation of non-connectivity. A k-hop path between two vertices i and j is a path that uses exactly k edges.

Proposition: The k-th min-max power of W, denoted Wk, calculated as Wk = Wk−1 ⊙ W, has elements Wi,jk that represent the maximum flow possible on any k-hop path from vertex i to j. Min-max Multiplication: Given matrices A and B, C = A ⊙ B is defined such that:

Ci,j = max

(min(Ai,r,Br,j))

r

Proof by Induction: Base Case (k = 1):

- • Claim: Wi,j1 represents the capacity of the edge from i to j, which is the maximum flow on a 1-hop path.
- • Proof: By definition, W1 = W, and Wi,j1 = Wi,j, which directly corresponds to the edge capacity between i and j. Hence, the base case holds.

#### Inductive Step:

- • Assumption: Assume that for k − 1, Wi,jk−1 correctly represents the maximum flow on any k − 1-hop path from i to j.
- • To Prove: Wi,jk represents the maximum flow on any k-hop path from i to j.

Proof: From the definition of min-max multiplication, Wi,jk = max

(min(Wi,rk−1,Wr,j))

r

- • Wi,rk−1 is the maximum flow from i to r using k − 1 hops.

- • Wr,j is the capacity of the edge from r to j (1-hop).

Interpretation: min(Wi,rk−1,Wr,j) finds the bottleneck flow for the path from i to j through r using k hops. The minimum function ensures the path’s flow is constrained by its weakest segment.

Maximization Step: maxr over all possible intermediate vertices r selects the path with the highest bottleneck value, thus ensuring the selected path is the most capable among all possible k-hop paths.

Conclusion: The inductive step confirms that the flow represented by Wi,jk is indeed the maximum possible flow across any k-hop path from i to j. Hence, by induction, the proposition holds for all k.

### C Related Work

Video Diffusion Models. Video diffusion models generate video frames by gradually denoising a noisy latent space [22]. One of the main challenges with these models is their high computational complexity. Typically, the denoising process is performed in the latent space [66, 4, 3]. The architectural commonly adopt either a 3D-UNet [22, 4, 21, 19, 60] or diffusion transformer [17, 40, 36]. To enhance computational efficiency, these architectures often employ separate self-attention mechanisms for managing spatial and temporal tokens. Conventionally, training these models involves fine-tuning an image-based model for video data [60, 26, 16]. This process includes adding a temporal module while striving to preserve the original visual quality.

Despite their ability to generate photorealistic videos, these models frequently struggle with understanding the complex interactions between elements in a scene. This shortcoming can result in the generation of nonsensical videos when responding to complex prompts.

Compositional Generation. Current generative models often face challenges in creating data from a combination of conditions, with most developments primarily in the image domain. Energy-based models [11, 10, 31], for example, are mathematically inclined to be compositionally friendly, yet they require the conditions to be independent. In practice, many image-based methods utilize crossattention to effectively manage the composition of concepts [14, 5, 63, 42]. However, when it comes to video, compositional generation introduces additional complexities. Some video-focused approaches concentrate specific form of composition, including object-motion composition [58], subject-composition [57], utilize explicit graphs to control content elements [2], or integrate multimodal conditions [54]. Despite these efforts, a generic solution for accurately generating videos from text descriptions involving multiple concepts is still lacking. We present the first solution for compositional video generation using complex text prompts, an area that remains largely underexplored.

Attribution Methods. Attribution methods clarify how specific input features influence a model’s decisions. gradient-based methods [49, 46, 44] identify influential image regions by back-propagating gradients to the input. Attention-based methods [6, 1] that utilize attention scores to emphasize important inputs. Ablation methods[41, 65] modify data parts to assess their impact. Shapley values [35] distribute the contribution of each feature based on cooperative game theory. In our paper, we extend existing techniques of attention flow to video diffusion models. We develop an efficient approximation to solve the max-flow problem. This improvement helps us more accurately identify and balance the impact of each textual elements on synthesized video.

### D Compositional Video Editing

Our system, Vico, can be integrated into video editing workflows to accommodate text prompts that describe a composition of concepts.

Setup. We begin by performing a 50-step DDIM inversion on the input video. Following this, we generate a new video based on the given prompt.

Results. An example of this process is illustrated in Figure 5. The original video demonstrates a strong bias towards a single presented object, making editing with a composition of concepts challenging. However, by applying Vico, we successfully enhance the video to accurately represent the intended compositional concepts.

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

InputVideo VideoCrafterv2

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

+VicoDDIMInverse

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

A vintage car driving on a winding road through a lush forest, with a standing cat.

Figure 5: Video edit results with compositional prompts.

- E More Visualizations Here we provide more example for compositional T2V in Figure 6

[Figure 127]

[Figure 128]

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

A kite surfer practicing on a quiet lake at dawn, the sky just beginning to brighten.

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

A wildlife photographer capturing close-up shots of a rare bird in a dense jungle.

[Figure 151]

[Figure 152]

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

A zebra conducting traffic in a busy urban intersection.

VideoCrafterv2VideoCrafterv2VideoCrafterv2Vico

VicoVico

Figure 6: Video visualization for compositional video generation

- F Speed Analysis

Attribution Speed. In this section, we assess the running speed of our ST-flow. To assess its computational efficiency, we compare ST-flow with cross-attention and Attention Rollout [1] computation, by reporting the theoretical complexity and empirical running time. We assume we have 1 cross attention map of mxn and L self-attention map of n×n, and demonstrated the theoretical results. Specifically, we measure the average running time required for each diffusion model inference, focusing solely on

the time taken for attribution computation, excluding the overall model inference time. We use the VideoCrafterv2 as the base model.

As detailed in Table 6, the cross-attention computation is fast, as it processes only a single layer. Both Attention Rollout and our approximated ST-Flow involve matrix multiplications and consequently share a similar time complexity. However, our ST-Flow approximation benefits from the relatively faster speed of element-wise min-max operations compared to the floating-point multiplications used in Attention Rollout, leading to slightly quicker execution times.

In contrast, the exact ST-Flow method is much slower. This is because it requires independently estimating the flow for each sink-source pair, a process that takes considerable time.

Method Complexity sec/inference

Cross-Attn. O(1) 0.002s Attention Rollout O(Lmn2) 0.042s Exact-ST-Flow O(L3mn4) 8s ST-Flow (soft) O(Lmn2) 0.037s

Table 6: Speed comparison for attribution method.

Diffusion Inference Speed. Our Vico framework includes a iterative optimization process alongside with the denoising. As expected, it should results in longer inference time. We evaluated this using a 50-step DPM denoising process on the VideoCrafterv2 model, at a resolution of 512 × 320 for 16 frames, on a single A6000 GPU.

The results, shown in Table 7, reveal that the baseline VideoCrafterv2 completed in 23 seconds. Adding the Attend&Excite increased the duration to 48 seconds. In comparison, our Vico framework finished in a comparable time of 50 seconds. Despite its additional complexity, Vico’s efficient design keeps the inference time within a reasonable range.

Method Time

VideoCrafterv2 23s + Attend&Excite 48s + Vico (soft&hard) 45s

Table 7: Text-to video model inference time comparison.

### G Implementation details of Vico

ST-Flow Computation. To compute the ST-Flow, we begin by extracting attention weights from all layers. These weights are averaged across all heads and then upscaled to the image size using bicubic interpolation. Due to the block-wise sparse pattern of the connections, min-max matrix multiplication is applied to the capacity matrix for connected layers. Furthermore, given that cross-attention layers include skip connections from previous layers, we divide the network into multiple groups. Within each group, min-max matrix Multiplication is performed. Finally, we aggregate the scores across all groups to obtain the results. The pseudocode for the min-max multiplication is in Algorithm 1.

Latent Step. During the first half of the sampling process, we update the latent variables. We establish a loss threshold of 0.2; once this threshold is reached, no further updates are made.

### H Limitations

Although Vico effectively allocates attribution across different tokens, it does not explicitly bind attributes to subjects. Moreover, there is a critical balance to maintain between latent updates and semantic coherence. Excessive updating can lead to the generation of nonsensical videos.

Algorithm 1 Batched Min-Max Matrix Multiplication

- 1: function BATCHMINMAXMATRIXMULTIPLICATION(A,B)
- 2: Input:
- 3: A is a tensor of shape [B,m,k]
- 4: B is a tensor of shape [B,k,n]
- 5: Output:
- 6: Tensor of shape [B,m,n] containing the maximum values
- 7: Aexpanded ← A.unsqueeze(2) ▷ Shape becomes [B,m,1,k]
- 8: Bexpanded ← B.permute(0,2,1).unsqueeze(1) ▷ Shape becomes [B,1,n,k]
- 9: min_vals ← torch.min(Aexpanded,Bexpanded) ▷ Shape becomes [B,m,n,k]
- 10: max_vals ← torch.max(min_vals,dim = 3).values ▷ Shape becomes [B,m,n]
- 11: return max_vals
- 12: end function

### I Broader Applications

Technically, the computation of attention flow proposed in our system is versatile and can be efficiently applied to a variety of other applications like erase certain concept in diffusion models. Additionally, the principle of fairly distributing the contribution of different input parts can be extended to other domains, such as language modeling.

