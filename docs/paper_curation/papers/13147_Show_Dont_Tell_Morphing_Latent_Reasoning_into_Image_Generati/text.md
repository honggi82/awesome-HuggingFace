## Show, Don’t Tell: Morphing Latent Reasoning into Image Generation

# arXiv:2602.02227v1[cs.CV]2Feb2026

Harold Haodong Chen*12 Xinxiang Yin*13 Wen-Jie Shu2 Hongfei Zhang1 Zixin Zhang14 Chenfei Liao1 Litao Guo1 Qifeng Chen†2 Ying-Cong Chen†12

### Abstract

Text-to-image (T2I) generation has achieved remarkable progress, yet existing methods often lack the ability to dynamically reason and refine during generation–a hallmark of human creativity. Current reasoning-augmented paradigms mostly rely on explicit thought processes, where intermediate reasoning is decoded into discrete text at fixed steps with frequent image decoding and reencoding, leading to inefficiencies, information loss, and cognitive mismatches. To bridge this gap, we introduce LatentMorph, a novel framework that seamlessly integrates implicit latent reasoning into the T2I generation process. At its core, LatentMorph introduces four lightweight components: (i) a condenser for summarizing intermediate generation states into compact visual memory, (ii) a translator for converting latent thoughts into actionable guidance, (iii) a shaper for dynamically steering next image token predictions, and (iv) an RL-trained invoker for adaptively determining when to invoke reasoning. By performing reasoning entirely in continuous latent spaces, LatentMorph avoids the bottlenecks of explicit reasoning and enables more adaptive selfrefinement. Extensive experiments demonstrate that LatentMorph (I) enhances the base model Janus-Pro by 16% on GenEval and 25% on T2ICompBench; (II) outperforms explicit paradigms (e.g., TwiG) by 15% and 11% on abstract reasoning tasks like WISE and IPV-Txt, (III) while reducing inference time by 44% and token consumption by 51%; and (IV) exhibits 71% cognitive alignment with human intuition on reasoning invocation. Our code: LatentMorph.

### 1. Introduction

Text-to-image (T2I) generation has progressed rapidly in recent years, driven by advances in diffusion (Rombach

*Equal contribution 1HKUST(GZ) 2HKUST 3NWPU 4Knowin. Correspondence to: Qifeng Chen <cqf@ust.hk>, Ying-Cong Chen <yingcongchen@usk.hk>.

[Figure 1]

Figure 1. The comparison of reasoning-augmented image generation paradigms: external-loop, internal-loop, and LatentMorph.

et al., 2022; Saharia et al., 2022) and autoregressive (Sun

- et al., 2024; Wang et al., 2024a; Wu et al., 2025) generative models, which now power a wide range of applications. Despite these successes, previous T2I generators primarily function as text-to-pixel mapping systems, with limited capacity for explicit deliberation or self-refinement during generation–key hallmarks of human creativity.

In contrast, large language models (LLMs) (Brown et al., 2020; Touvron et al., 2023) have demonstrated remarkable emergent reasoning abilities, popularized by chainof-thought (CoT) (Wei et al., 2022) prompting and training. This has motivated a growing line of work aiming to endow image generation with “System-2”-like reasoning. Existing approaches can be broadly organized into two paradigms: (I) external-loop paradigm, which couples an (M)LLM with a generator and uses the (M)LLM as an optimizer to refine prompts, critique outputs, or iteratively propose edits (Wang et al., 2024b; Gani et al., 2024; Yang et al., 2024; Zhan et al., 2024); and (II) internal-loop paradigm, increasingly enabled by unified multimodal models (UMMs) (e.g., Janus-series (Chen et al., 2025d)), attempt to interleave reasoning between the understanding and generation branches within a single backbone (Liao et al., 2025; Huang

- et al., 2025b; Qin et al., 2025a; Guo et al., 2025b). While

these reasoning-before, reasoning-after, or reasoning-whilegenerating show effectiveness, a common thread remains: they typically rely on decoupled, explicit CoT, where “thinking” is forced to be decoded into discrete text and then re-ingested at predefined fixed steps.

This reliance on explicit thought introduces three fundamental deficiencies: (i) information loss: forcing intermediate cognition into natural language compresses rich internal states into a narrow symbolic channel (Zhu et al., 2025b); (ii) inefficiency: repeated decode-encode cycles add latency and consume context budget; (iii) cognitive mismatch: human creativity rarely involves verbalizing intermediate judgments at every step; instead, humans rely on continuous, implicit thoughts to guide their actions dynamically. Empirical validations are demonstrated in Section §5.

Given these limitations, latent reasoning offers a compelling alternative, where intermediate reasoning is performed in continuous hidden states rather than explicit text. Representative works in language modeling show that latent reasoning can be induced by special tokens that suppress explicit thought (e.g., Coconut (Hao et al., 2024)), or by compressing verbose rationales into a small set of informative latent vectors (e.g., SoftCoT (Xu et al., 2025)). While these approaches have been extended to multimodal reasoning tasks, such as visual latent thoughts for understanding (Li et al., 2025a; Dong et al., 2025), they remain constrained to a single understanding feature space. Directly applying these methods to image generation is non-trivial, as the task involves a distinct execution process with its own tokenization and dynamics. This introduces a bi-directional interface mismatch: a perception gap, where generation-time states are not directly interpretable by the understanding branch, and an execution gap, where reasoning-time states are not directly actionable for the generation branch. This leads to our pivotal research question:

How can we seamlessly interleave latent reasoning into the image generation, enabling the model to dynamically monitor its evolving visual state and intervene to refine generation?

[Figure 2]

To bridge this gap, we introduce LatentMorph, a dynamic and cognitively-aligned framework that morphs on-the-fly latent reasoning into T2I generation, eliminating the need for explicit textual CoT. At its core, LatentMorph closes the loop between reasoning and generation by introducing four lightweight components: First, a ♣ condenser that summarizes intermediate generation states into compact, semantically meaningful visual memory, enabling the understanding branch to process visual evidence efficiently without requiring prohibitively large state readouts. Second, a ♦ translator that then converts latent thoughts from the understanding branch into generator-compatible guidance, and a ♥ shaper that integrates these signals to steer imagetoken predictions dynamically. Additionally, an RL-trained

♠ invoker that adaptively decides when to invoke reasoning as a cognitive monitor, enabling a more cognitively-aligned process and overcoming the inefficiencies of fixed-step explicit reasoning injections in existing paradigms.

By performing implicit, interleaved reasoning during generation, LatentMorph enables adaptive self-refinement, improving instruction-following and compositional fidelity while avoiding the bottlenecks of explicit thoughts and the burden of constructing additional training data. Moreover, LatentMorph is model-agnostic and can be instantiated in both external-loop and internal-loop settings with autoregressive generators, with further analysis in Appendix §A.3.

Empirical Evaluation. Extensive experiments across five benchmarks and ten baseline strategies demonstrate that LatentMorph achieves ❶ superior fidelity, enhancing the base model Janus-Pro by 16.0% on GenEval and 25.3% on T2I-CompBench; ❷ abstract reasoning, outperforming the explicit reason-while-generation paradigm by 15.6% and 11.3% on the complex WISE and IPV-Txt benchmarks; ❸ inference efficiency, surpassing baselines while reducing inference time and token consumption by 44.3% and 51.0%, respectively; and ❹ cognitive alignment, where LatentMorph’s adaptive invocation strategy achieves a 71.8% alignment with human evaluators.

### 2. Related Work

Reasoning in Image Generation. A growing line of T2I work seeks to incorporate LLM-style reasoning (Wei et al., 2022) to enable more cognitively aligned generation. Existing approaches largely fall into two families: (I) externalloop approaches couple an (M)LLM with a generator, most commonly as reasoning-before-generation (Wang et al., 2024b; Gani et al., 2024), e.g., using an LLM for planning or prompt optimization, or reasoning-after-generation (Yang

- et al., 2024; Zhan et al., 2024), e.g., adopting an MLLM for verification and revision via re-prompting or editing. In contrast, (II) internal-loop approaches, enabled by recent UMMs (Chen et al., 2025d; Xie et al., 2024; Team, 2024; Deng et al., 2025), interleave reasoning between the understanding and generation branches within a single backbone; beyond reasoning-before/after (Liao et al., 2025; Huang
- et al., 2025b; Qin et al., 2025a; Mi et al., 2025), recent work explores reasoning-while-generating (Guo et al., 2025b) by inserting periodic checks or interventions at predefined decoding steps. While effective, these paradigms most rely on decoupled, explicit reasoning, which causes information loss, or, like recent works adapting LatentSeek (Li et al., 2025b) to T2I (Mi et al., 2025), resort to test-time latent search. The latter also necessitates iterative reasonbefore/after interactions guided by external rewards, remaining computationally prohibitive and cognitively disjointed.

Latent Reasoning in LLMs. With LLM reasoning increasingly studied, recent work has shifted from explicit

CoT to latent reasoning, motivated by the representational richness and potential efficiency of latent spaces (Zhu et al., 2025b; Chen et al., 2025e). The core idea is to carry intermediate deliberation in compact latent representations rather than verbose textual rationales (Wang et al., 2023; Goyal et al., 2023; Deng et al., 2023). Recent works broadly follow two directions: (I) special thinking-token based implicit reasoning (Hao et al., 2024; Li et al., 2025a; Dong et al., 2025; Yang et al., 2025; Qin et al., 2025b) introduces dedicated tokens or control mechanisms to induce internal deliberation while suppressing explicit rationales, typically trained with supervision from textual reasoning traces or auxiliary multimodal cues; and (II) distillation/compression approaches (Xu et al., 2025; Shen et al., 2025; Zhang et al., 2025b;a) explicitly compress long CoT rationales into a small set of informative soft tokens that can be consumed more efficiently at inference. Although the aforementioned works mainly focus on the understanding setting within a single feature space, they have strongly inspired our exploration of latent deliberation for visual generation.

### 3. Preliminary: Problem Formalization

We formalize reasoning-augmented T2I generation in the context of a UMM, which consists of an autoregressive T2I generation branch UMMg and a multimodal understanding branch UMMu. Given a user prompt T, UMMg autoregressively generates a sequence of discrete image tokens X = (x1,x2,...,x|X|), where |X| is the total number of tokens. The probability of X is modeled as:

|X|

pθ(xi | T,x<i), (1)

pθ(X | T) =

i=1

where x<i = (x1,...,xi−1) represents the tokens generated up to step i − 1. The sequence X is decoded into the final image Iˆ = Dec(X). To ensure that the generated image Iˆaligns with T, the objective is to maximize a reward function R(I,Tˆ ) over a prompt distribution D, defined as:

ET∼D R(I,Tˆ ) , (2) where R(·) measures alignment metrics, e.g., semantic alignment. To achieve this goal, reasoning interventions are introduced to leverage UMMu’s reasoning capabilities for refining the generation process. At an intervention step k ∈ {0,1,...,|X|}, the partially generated tokens X<k are decoded into an intermediate image Iˆ<k = Dec(X<k). This intermediate image, along with T, is encoded and passed to UMMu as Enc(Iˆ<k,T). UMMu performs reasoning and generates intermediate thoughts Sk = (s1,s2,...,sm,T′), where s1,s2,...,sm denote reasoning steps and T′ represents the refined prompt. T′ is then re-encoded as Enc(T′) and passed back to UMMg to continue generation.

max

θ

For the reason-before/after-generation (e.g., IRG (Huang et al., 2025b) and Uni-CoT (Qin et al., 2025a)), reasoning is

invoked only at k = 0/|X|, respectively, and T = T′ for all subsequent steps (e.g., generate from scratch), while for the reason-while-generation (e.g., TwiG (Guo et al., 2025b)), reasoning is performed at fixed intermediate steps (k ∈ {k1,k2,...}), e.g., every 1/3 |X| generated tokens. The external-loop paradigm (e.g., Idea2Img (Yang et al., 2024)) replaces UMMu and UMMg with separate (M)LLMs and image generators. Our work, which enables reasoning (Sk) entirely in the latent space, eliminates the need for decoding intermediate steps into text or images and re-encoding, along with designing a dynamic intervention mechanism that adaptively determines when (k) to reason, ensuring seamless integration with the generation process.

### 4. Methodology

#### 4.1. LatentMorph: Generating with Latent Reasoning

Just as humans dynamically reflect and refine their thoughts while creating art, image generation can benefit from reasoning. However, existing methods rely on explicit reasoning, where intermediate thoughts are decoded into discrete text or images at fixed steps. To bridge this gap, LatentMorph interleaves latent reasoning directly into the generation process, enabling implicit and adaptive self-refinement.

As shown in Figure 2, given a user prompt T, the autoregressive generation branch UMMg generates image tokens X = (x1,x2,...,x|X|), where the evolving hidden states H1:i = (h1,...,hi) are monitored continuously. LatentMorph evaluates whether reasoning is required and integrates latent thoughts into the generation process when necessary. Specifically, like how humans continuously think while drawing, at every window interval w, a short-term condenser Cshort compresses the most recent Hi−w:i into a local, short-term latent memory:

s×d, (3) which does not incur excessive inference delay, as validated in Section §5.4. A reasoning invoker Iinvoker then evaluates the current generation state si, which includes M(s) and additional features (e.g., prediction uncertainty), to determine whether reasoning should be invoked:

M(s) = Cshort(hi−w,...,hi), M(s) ∈ Rn

πθ(ak | si) = σ(Iinvoker(si)), (4) where ak ∈ {CONTINUE,REASON}. If ak = [CONTINUE], UMMg continues generating tokens. Otherwise (ak = [REASON]), the latent reasoning is activated at the intervention point k = i. Specifically, a long-term condenser Clong1 first summarizes the entire generation history H1:k into a concise long-term visual latent memory M(l) ∈ Rn

l×d by Clong(h1,...,hk). This memory, along with the encoded prompt T, is passed to the reasoning branch UMMu, without decoding and re-encoding itself, which performs latent reasoning and outputs latent thoughts

1We denote both short-term condenser and long-term condenser as the condenser in Section §1 for clarity.

[Figure 3]

- Figure 2. Overview of LatentMorph. LatentMorph seamlessly integrates implicit reasoning into the autoregressive generation stream. (Middle) Dynamic Monitoring: a short-term condenser compresses recent hidden states Hi−w:i into a local memory, enabling the reason invoker Iinvoker to adaptively trigger reasoning interventions. (Bottom) Latent Reasoning: upon invocation, a long-term condenser summarizes the global history H1:i for UMMu. The resulting latent thoughts z are transformed by the translator and shaper into control tokens Ectrl, which are injected directly into the generator’s KV cache to steer subsequent synthesis without explicit text decoding.

z ∈ Rd in continuous hidden states, which represent highlevel insights and refinements derived from the reasoning.

To bridge the gap between reasoning and generation, a latent translator Ttrans converts the latent thoughts z into generation-compatible control signals c ∈ Rd:

c = Ttrans([z;M(l);p]), (5) where p denotes the prompt embedding of T in UMMg. Subsequently, a latent shaper Sshaper injects c into the generation pipeline by generating control tokens Ectrl ∈ R2B×j×d, which are inserted into the key-value (KV) cache of UMMg, modifying subsequent token predictions:

KVnew = Sshaper(c,KVold). (6) This mechanism seamlessly integrates reasoning outputs into the generation process, ensuring implicit guidance without disrupting internal dynamics (e.g., directly replace p). We next detail the implementations of the condensers and invoker (Section §4.2) and the translator and shaper (Section §4.3), along with the training recipe (Section §4.4).

#### 4.2. Learning to Invoke Reasoning with Visual Memory

Existing methods (Huang et al., 2025b; Qin et al., 2025a; Guo et al., 2025b) rely on fixed, predefined reasoning, where partially generated images are decoded and re-encoded into the understanding branch. In this section, LatentMorph aims to address two critical challenges: how to decide when reasoning should be invoked more dynamically; and how to efficiently represent the visual state of the generation process for reasoning. To this end, LatentMorph introduces the

reasoning invoker Iinvoker as a dynamic and adaptive monitor, operating on a compact yet expressive representation of the generation state, constructed through short-term and longterm condensers Cshort and Clong.

Short-Term Condenser for Invoker Monitoring. The short-term condenser Cshort plays a critical role in enabling the reasoning invoker to monitor the recent generation progress. At every window interval w, Cshort compresses the hidden states of the most recent w token hidden states Hi−w:i into a compact memory representation M(s) ∈ Rn

s×d, which encapsulates localized generation dynamics. To extract salient features from Hi−w:i, Cshort employs cross-attention with learnable latent queries Q ∈ Rn

s×d:

M(s),A = CA(Q,K = Hi−w:i,V = Hi−w:i), (7) where these memory tokens are further refined through a lightweight feedforward network, ensuring they retain meaningful information while remaining compact. To facilitate downstream decision-making, a pooled vector m(s) = Mean(M(s)) summarizes the short-term memory into a single representation. By focusing on the most recent generation window, Cshort provides a concise and up-to-date view of the generation process, enabling the reasoning invoker to make informed and efficient decisions.

Reasoning Invoker for Dynamic Invocation. Unlike fixed-step reasoning schedules, Iinvoker, instantiated as a lightweight policy network (e.g., MLP), adaptively determines when to invoke reasoning based on the evolving generation context, mimicking human-like reflection during creating. Specifically, Iinvoker operates on a state feature

..., a vendor arranging fresh fruits in wooden crates.

The soft white snow contrasted with the sharp ...

A realistic underwater ..., a sea turtle swims ...

On the calm lake, there is a deserted boat ...

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

A sunset lakeside scene ..., a moose standing ...

A rabbit near a train, with a detailed description ...

A snowy street with soft snowfall, muted colors.

A photo of three brown birds perched together ...

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

Ours w/o Reasoning

LatentMorph (Ours)

Ours w/o Reasoning

LatentMorph (Ours)

Ours w/o Reasoning LatentMorph (Ours)

Ours w/o Reasoning

LatentMorph (Ours)

- Figure 3. Case study of LatentMorph. The blue stars denote the adaptive reasoning invocations. These interventions align with critical semantic transitions, enabling LatentMorph to correct object omissions or counting errors observed in the baseline without reasoning.

vector si, which comprises multi-dimensional signals:

✤ Semantic Consistency. Semantic similarity ci = cos(m(s),p) between recent state and prompt embedding, which measures alignment with the user’s intent.

✤ Prediction Uncertainty. Entropy of the token logits ui = (oi), capturing model confidence, where oi represents the model’s predicted probability distribution.

✤ Temporal Dynamics. Changes in semantic consistency ∆ci = ci − ci−w and its variance vi = Var(ci−w:i), reflecting deviations and stability.

These signals are combined into si = [ci,ui,∆ci,vi] and Iinvoker computes the invocation decision as:

pi = πθ(ak = REASON | si) = σ(Iinvoker(si)), (8) where ak ∈ {CONTINUE,REASON} is sampled from a Bernoulli distribution parameterized by pi. To encourage Iinvoker to invoke reasoning only when necessary, we maximize an RL objective that balances task performance and reasoning efficiency, which avoids redundant reasoning steps, inspired by (Zhu et al., 2025a; Zhang et al., 2025a):

R(τ) − λ · max(0,p¯(τ) − p¯ref) , (9) where R(τ) is the task reward (detailed in Section §4.4), p¯(τ) is the average invoker probability over trajectory τ, and p¯ref is an adaptive reference level computed from highreward trajectories in the batch.

Eτ∼π

max

θ

θ

#### Long-Term Condenser for Reasoning Integration.

Once reasoning is triggered, the long-term condenser Clong summarizes the complete generation history H1:k into a concise memory representation M(l) ∈ Rn

l×d. This longterm visual memory provides a high-level global overview of the generation trajectory, enabling UMMu to process visual evidence efficiently, rather than decode and re-encode. Different from Cshort, which captures localized trends, Clong employs streaming attention to handle arbitrarily long sequences by incrementally processing non-overlapping chunks of size c. For each chunk Ht:t+c, cross-attention with learnable memory tokens Q ∈ Rn

l×d is applied as in Equation (7), where the memory updates incrementally:

M(l) = Update(M(l),Ht:t+c). (10)

Here, only the top-nl most informative tokens are retained based on attention scores. Similar to Cshort, a pooled vector m(l) = Mean(M(l)) is computed to summarize the longterm memory. These representations are then passed to UMMu, enabling reasoning directly in the hidden space.

#### 4.3. Morphing Reasoning in Latent Space

A naive way to bridge the gap between UMMu and UMMg might directly map the latent thoughts z into the prompt embedding p space, akin to explicit reasoning paradigms that replace prompts (Guo et al., 2025b). However, such a direct replacement treats reasoning as static, neglecting the dynamic nature of autoregressive generation and undermining p’s role as a guiding signal that interacts with evolving generation states. In this section, LatentMorph introduces the latent translator Ttrans and latent shaper Sshaper, which transform z into actionable guidance and seamlessly inject it into the generation pipeline, preserving autoregressive consistency and enabling adaptive refinement.

Latent Translator for Guidance Generation. The latent translator Ttrans is responsible for converting z, along with long-term visual memory m(l) and the prompt embedding p, into generation-compatible control signals c, as formalized in Equation (5). Internally, Ttrans employs a lightweight MLP with residual connections and a gating mechanism to adaptively filter noise and retain salient information:

c′ = MLP([z;m(l);p]), g = tanh(Linear(c′)), (11) where c = c′ ⊙ g. This design allows Ttrans to adaptively modulate the contribution of reasoning outputs, ensuring that only salient features influence the generation process. By incorporating m(l), Ttrans leverages historical generation context, enabling reasoning to influence the process in a globally consistent manner.

Latent Shaper for Control Injection. Inspired by the use of control tokens in conditional generation tasks, Sshaper extends the concept to dynamically inject reasoning-derived guidance during the generation process. Specifically, Sshaper dynamically transforms c into a sequence of control tokens Ectrl ∈ R2B×j×d, where B is the batch size, j is the number of control tokens, and d is the hidden dimension. Ectrl is

Table 1. Evaluation on GenEval, and T2I-CompBench. The best and second best results are highlighted.

GenEval T2I-CompBench Two Object↑ Position↑ Color Attribute↑ Overall↑ Color↑ Shape↑ Texture↑ Spatial↑ Non-Spatial↑ Complex↑ Overall↑

Method

Vanilla 0.89 0.79 0.66 0.80 63.59 35.28 49.36 20.61 30.85 35.59 39.21 SFT 0.87 0.79 0.66 0.79 64.23 34.56 49.46 20.98 31.55 35.80 39.43 GRPO 0.90 0.80 0.66 0.82 67.90 36.31 52.13 23.01 31.29 39.04 41.61 Self-CoT (Deng et al., 2025) 0.90 0.78 0.70 0.83 68.19 37.89 54.10 21.90 30.00 44.01 42.68 T2I-R1 (Jiang et al., 2025) 0.91 0.76 0.65 0.79 81.30 58.52 72.41 33.78 30.90 39.93 52.81 TIR (Khan et al., 2025) 0.93 0.84 0.68 0.84 68.92 49.12 60.10 21.77 31.21 40.12 45.21 T2I-Copilot (Chen et al., 2025a) 0.94 0.86 0.66 0.85 67.42 47.82 61.34 22.12 30.41 41.85 45.16 MILR (Mi et al., 2025) 0.96 0.98 0.91 0.95 85.08 51.17 69.49 46.13 30.78 36.84 53.25 TwiG-ZS (Guo et al., 2025b) 0.94 0.84 0.67 0.86 73.11 41.55 64.77 21.98 30.90 48.16 46.75 TwiG-RL (Guo et al., 2025b) - - - - 82.49 61.28 73.19 34.06 31.99 54.45 56.24 LatentMorph (Ours) 0.97 0.98 0.92 0.96 84.04 69.46 79.89 50.93 39.27 63.60 64.53

then inserted into the KV cache of UMMg, modifying subsequent token predictions without altering the autoregressive structure or occupying prediction positions:

Ectrl = Sshaper(c),KVnew = Update(KVold,Ectrl). (12) By injecting control tokens directly into the KV cache, Sshaper enables reasoning-derived guidance to influence subsequent token predictions implicitly, without requiring explicit decoding or disrupting the autoregressive dynamics. This approach ensures that the generation process remains efficient and cognitively aligned, dynamically adapting to reasoning outputs as needed.

#### 4.4. Two-Stage Training of LatentMorph

Supervised Fine-Tuning (SFT). We train the long-term modules, i.e., the condenser Clong, translator Ttrans, and shaper Sshaper, using 20k text-image pairs from midjourneyprompts (vivym, 2023). Each image is associated with a single randomly triggered reasoning step during generation. Unlike prior works on interleaving reasoning, e.g., (Guo et al., 2025b; Gu et al., 2025), or latent reasoning, e.g., (Li et al., 2025a; Dong et al., 2025), which require extensive process-level supervision or curated datasets for each reasoning step, our approach relies solely on a standard cross-entropy (CE) loss for autoregressive image generation, achieving adaptive learning without additional supervision. Reinforcement Learning (RL). The reasoning invoker Iinvoker and short-term condenser Cshort are optimized using GRPO (Guo et al., 2025a) with a group size of 8, incorporating the penalty mechanism introduced in Equation (9) for adaptive reasoning invocation. Following TwiG (Guo et al., 2025b), we use the training set prompts from T2ICompBench (Huang et al., 2023) for training. For reward models, we adopt HPS-v2.1 (Wu et al., 2023) and CLIP score (Radford et al., 2021), as in DanceGRPO (Xue et al., 2025). All experiments are conducted on 8 NVIDIA H200 GPUs, with additional details provided in Appendix §A.

### 5. Experiments

In this section, we conduct extensive experiments to answer the following research questions:

- RQ1: Can LatentMorph outperform explicit interleaving reasoning paradigms?
- RQ2: Is latent reasoning by LatentMorph effective in understanding abstract and complex user prompts?
- RQ3: Does LatentMorph’s adaptive latent reasoning invocation facilitate more efficient generation?
- RQ4: Does LatentMorph align better with human cognitive processes in artistic creation?

#### 5.1. Experimental Settings

Baselines. We apply LatentMorph to the advanced pure autoregressive UMM Janus-Pro (Chen et al., 2025d). We focus our comparisons on the following ten baseline strategies: (I) Generation-only methods: the vanilla model, SFT, GRPO (Guo et al., 2025a); (II) Reason-before/aftergeneration: Self-CoT (Deng et al., 2025), T2I-R1 (Jiang et al., 2025), TIR (Khan et al., 2025), T2I-Copilot (Chen et al., 2025a), MILR (Mi et al., 2025); and (III) Reasonwhile-generation: TwiG-ZS, TwiG-RL (Guo et al., 2025b). Details on baselines are provided in Appendix §B.1.

Evaluations. We conduct evaluations across five benchmarks focusing on three key aspects: (i) General: GenEval (Ghosh et al., 2023); (ii) Compositional: T2I-CompBench (Huang et al., 2023), T2I-CompBench++ (Huang et al., 2025a); and (iii) Complex: WISE (Niu et al., 2025), IPV-Txt (Bai et al., 2025). Details are placed in Appendix §B.2.

#### 5.2. Main Results

To answer RQ1, we conduct comprehensive comparisons against ten baselines on general and compositional benchmarks in Tables 1, 3, along with qualitative results shown in Figures 4, 8 and 9. Key observations are summarized:

Obs.❶ LatentMorph establishes superior fidelity in both general and compositional generation. As shown in Table 1, LatentMorph consistently outperforms state-ofthe-art reasoning-augmented paradigms overall. Most notably in the challenging Non-Spatial category of T2ICompBench, LatentMorph surpasses the leading reasonwhile-generation baseline TwiG-RL by a significant mar-

Vanilla + TwiG-ZS + LatentMorph (Ours) Vanilla + MILR + LatentMorph (Ours)

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

A sheep standing on top of a table, with a detailed description of its position ...

A dog sitting on top of a sofa, with its paws resting on the armrests, ...

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

Four ducks quacked by the pond, with a vivid depiction of the scene.

The soft white feathers of the owl contrast with the sharp black talons, set ...

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

A man in a black jacket standing in a kitchen next to a gray dog. In this fashion show, the models will be wearing bold and avant-garde outfits.

Figure 4. Qualitative comparison of LatentMorph.

gin of 7.28%. This advantage widens further against iterative reason-before/after strategies; LatentMorph exceeds T2I-Copilot and MILR by 8.86% and 8.49% respectively, achieving better alignment without their heavy computational overhead. We attribute this performance leap to the continuous nature of our latent reasoning: unlike TwiG or T2I-Copilot, which compress intermediate thoughts into discrete text, LatentMorph retains rich semantic cues within continuous high-dimensional states. This robustness extends to fine-grained tasks in T2I-CompBench++ in Table 3, where LatentMorph outperforms TwiG-RL by 8.13% in 3D-Spatial and 7.32% overall. Qualitative results in Figures 4, 8 and 9 visually reinforce these findings, demonstrating LatentMorph’s ability to resolve complex attributes where baselines frequently suffer from hallucination.

#### 5.3. Effectiveness Analysis

To answer RQ2, we extend our evaluation to the more abstract and cognitively demanding benchmark, i.e., WISE, as well as the idea of “impossible prompting” in IPV-Txt,

- as shown in Figure 5 (Left & Middle). To further isolate the efficacy of latent vs. explicit reasoning, we introduce a baseline “LatentMorph w/o latent”, where the latent thought z is forced through a decode-re-encode bottleneck before control injection, and the attention differences are visualized in Figure 5 (Right). Our observations are:

Obs.❷ Latent reasoning captures ineffable semantics lost in textual decoding. Quantitative results in Figure 5 (Left) reveal a consistent performance hierarchy. Notably,

on the IPV-Txt benchmark, which involves counterintuitive physical dynamics, explicit reasoning methods, e.g., TwiGZS, struggle, likely because natural language is insufficient to precisely describe complex concepts. Even our own variant, w/o latent, suffers a performance drop solely due to the information loss from decoding thoughts into text. This is visually explained in Figure 5 (Right). The differential heatmaps show that LatentMorph activates attention in regions characterized by subtle textures and lighting, attributed to the semantically dense but difficult to verbalize explicitly. This confirms that performing reasoning entirely in the continuous latent space preserves critical, non-verbalizable visual cues essential for generation.

#### 5.4. Efficiency Analysis

To answer RQ3, we further quantify the inference latency and total token consumption of LatentMorph against representative baselines on T2I-CompBench, as demonstrated in Figure 6 (Left). Our findings are summarized as follows:

Obs.❸ LatentMorph is a time- and token-efficient autoregressive image generation enhancer. As illustrated in Figure 6 (Left), LatentMorph incurs negligible computational overhead compared to the vanilla baseline, while demonstrating significant efficiency gains over explicit reasoning paradigms. Iterative methods like MILR require multiple full-generation cycles to search for optimal latents, and reason-while-generating frameworks like TwiG-ZS are bottlenecked by the frequent decoding of partial images and the verbose textual thoughts. In contrast, LatentMorph

Ours w/o Reason Ours w/o Latent LatentMorph (Ours) Differential Heatmap

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

A person... walking across the ocean's surface, ...

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

A photoreali stic ... fusion of a sunflower and rose ...

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

- Figure 5. (Left) Evaluation results on WISE and IPV-Txt. (Middle) Qualitative examples on “impossible prompts” of IPV-Txt. (Right) Differential heatmap between latent and explicit reasoning, highlighting the information loss incurred by discrete text thoughts.

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

- Figure 6. (Left) Inference time and token consumption of LatentMorph vs. baselines on T2I-CompBench. (Middle) Reasoning invocation across GenEval, T2I-CompBench, and WISE. (Right) User study on the timing of reasoning intervention.

operates as a seamless single-pass stream. By condensing history into compact visual memory and injecting control signals directly, we eliminate the expensive steps of pixelspace decoding and explicit text tokenization. Notably, even our LatentMorph w/o latent variant outperforms fixedstep baselines (e.g., TwiG-ZS) in efficiency. This further validates the design of our adaptive invoker, which intelligently triggers reasoning only when necessary, avoiding redundant computation on well-aligned generation steps.

#### 5.5. Framework Analysis

To answer RQ4, we further analyze the frequency and positioning of reasoning invocations across benchmarks (each with 50 prompts) in Figure 6 (Middle). Complementing this, we conduct a user study to evaluate the alignment between LatentMorph’s invocation decisions and human cognitive intuition, summarized in Figure 6 (Right), alongside a case study in Figure 3. Our findings are summarized below:

Obs.❹ LatentMorph mimics the adaptive rhythm of human creative cognition. Unlike fixed-step invocation paradigms, e.g., TwiG, that force reasoning at rigid intervals regardless of context, LatentMorph exhibits a contextaware invocation pattern. As shown in Figure 6 (Middle), the invocation frequency correlates positively with task complexity, i.e., triggering sparsely for simple prompts (avg. 1.14 on GenEval) and more frequently for abstract reasoning tasks (avg. 1.60 on WISE). This aligns with human creative processes: pausing to reflect only when encounter-

ing bottlenecks. The user study in Figure 6 (Right) further corroborates this, where human evaluators rated our adaptive timing as significantly more natural and necessary compared to fixed baselines. Case studies in Figure 3. visually demonstrate this behavior, showing the model intervening precisely at critical compositional transitions.

Ablation Study. To further confirm the superiority of our learned invoker policy, Table 4 in Appendix §C shows that our adaptive strategy consistently outperforms both random and fixed-step injections, confirming that when to reason is as critical as how. Comprehensive sensitivity analyses covering the condensers, translator, shaper, as well as invoker hyperparameters are detailed in Appendix §C.

### 6. Conclusion

In this work, we present LatentMorph, a framework that integrates implicit latent reasoning into autoregressive text-toimage generation. By bypassing the bottlenecks of explicit textual decoding, LatentMorph enables continuous reasoning and refinement within high-dimensional latent spaces. Leveraging visual memory condensation, latent translation, and adaptive control injection, it achieves a strong balance between generation fidelity and efficiency. Additionally, the RL-based invoker dynamically aligns the generation process with human-like cognitive rhythms, intervening only when necessary. This paradigm shift from explicit reasoning to implicit intuition sets the foundation for the next generation of cognitively aligned and creatively robust visual generation.

### Impact Statement

This paper presents LatentMorph, a framework aimed at enhancing the fidelity and efficiency of text-to-image generation. On the positive side, LatentMorph significantly reduces the computational overhead associated with reasoningaugmented generation, contributing to the development of more energy-efficient and environmentally sustainable AI systems. Additionally, by better aligning with human cognitive processes, it serves as a more intuitive tool for artistic expression and creative workflows.

However, as with all advancements in high-fidelity generative models, there is a potential for misuse in creating photorealistic, misleading content or deepfakes. It is important to note that LatentMorph operates by optimizing the latent states of Janus-Pro as post-training and does not introduce new pre-training data. Consequently, LatentMorph inherits both the safety guardrails and the potential biases present in the underlying base model. We strongly encourage the deployment of such technologies in conjunction with robust safety filters and watermarking mechanisms.

### References

Bai, Z., Ci, H., and Shou, M. Z. Impossible videos. In Fortysecond International Conference on Machine Learning, 2025. URL https://openreview.net/forum? id=MNSW6U5zUA.

Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al. Language models are few-shot learners. Advances in neural information processing systems, 33: 1877–1901, 2020.

Chen, C.-Y., Shi, M., Zhang, G., and Shi, H. T2i-copilot: A training-free multi-agent text-to-image system for enhanced prompt interpretation and interactive generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 19396–19405, 2025a.

Chen, H. H., Huang, H., Chen, Q., Yang, H., and Lim, S.N. Hierarchical fine-grained preference optimization for physically plausible video generation. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025b. URL https://openreview.net/ forum?id=y0SRR9XGlZ.

Chen, H. H., Wu, X., Shu, W.-J., Guo, R., Lan, D., Yang, H., and Chen, Y.-C. Go with your gut: Scaling confidence for autoregressive image generation. arXiv preprint arXiv:2509.26376, 2025c.

Chen, X., Wu, Z., Liu, X., Pan, Z., Liu, W., Xie, Z., Yu, X., and Ruan, C. Janus-pro: Unified multimodal understand-

ing and generation with data and model scaling. arXiv preprint arXiv:2501.17811, 2025d.

Chen, X., Zhao, A., Xia, H., Lu, X., Wang, H., Chen, Y., Zhang, W., Wang, J., Li, W., and Shen, X. Reasoning beyond language: A comprehensive survey on latent chainof-thought reasoning. arXiv preprint arXiv:2505.16782, 2025e.

Deng, C., Zhu, D., Li, K., Gou, C., Li, F., Wang, Z., Zhong, S., Yu, W., Nie, X., Song, Z., et al. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025.

Deng, Y., Prasad, K., Fernandez, R., Smolensky, P., Chaudhary, V., and Shieber, S. Implicit chain of thought reasoning via knowledge distillation. arXiv preprint arXiv:2311.01460, 2023.

Dong, S., Wang, S., Liu, X., and Wei, Z. Interleaved latent visual reasoning with selective perceptual modeling. arXiv preprint arXiv:2512.05665, 2025.

Gani, H., Bhat, S. F., Naseer, M., Khan, S., and Wonka, P. LLM blueprint: Enabling text-to-image generation with complex and detailed prompts. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum? id=mNYF0IHbRy.

Ghosh, D., Hajishirzi, H., and Schmidt, L. Geneval: An object-focused framework for evaluating text-to-image alignment. Advances in Neural Information Processing Systems, 36:52132–52152, 2023.

Goyal, S., Ji, Z., Rawat, A. S., Menon, A. K., Kumar, S., and Nagarajan, V. Think before you speak: Training language models with pause tokens. arXiv preprint arXiv:2310.02226, 2023.

Gu, J., Hao, Y., Wang, H. W., Li, L., Shieh, M. Q., Choi, Y., Krishna, R., and Cheng, Y. Thinkmorph: Emergent properties in multimodal interleaved chain-of-thought reasoning. arXiv preprint arXiv:2510.27492, 2025.

Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., Zhu, Q., Ma, S., Wang, P., Bi, X., et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025a.

Guo, Z., Zhang, R., Li, H., Zhang, M., Chen, X., Wang, S., Feng, Y., Pei, P., and Heng, P.-A. Thinking-whilegenerating: Interleaving textual reasoning throughout visual generation. arXiv preprint arXiv:2511.16671, 2025b.

Hao, S., Sukhbaatar, S., Su, D., Li, X., Hu, Z., Weston, J., and Tian, Y. Training large language models to reason in a continuous latent space. arXiv preprint arXiv:2412.06769, 2024.

Huang, K., Sun, K., Xie, E., Li, Z., and Liu, X. T2icompbench: A comprehensive benchmark for open-world compositional text-to-image generation. Advances in Neural Information Processing Systems, 36:78723–78747, 2023.

Huang, K., Duan, C., Sun, K., Xie, E., Li, Z., and Liu, X. T2i-compbench++: An enhanced and comprehensive benchmark for compositional text-to-image generation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025a.

Huang, W., Chen, S., Xie, Z., Cao, S., Tang, S., Shen, Y., Yin, Q., Hu, W., Wang, X., Tang, Y., et al. Interleaving reasoning for better text-to-image generation. arXiv preprint arXiv:2509.06945, 2025b.

Jiang, D., Guo, Z., Zhang, R., Zong, Z., Li, H., Zhuo, L., Yan, S., Heng, P.-A., and Li, H. T2i-r1: Reinforcing image generation with collaborative semantic-level and token-level cot. arXiv preprint arXiv:2505.00703, 2025.

Khan, M. A. H., Jain, Y., Bhattacharyya, S., and Vineet, V. Test-time prompt refinement for text-to-image models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 6506–6516, 2025.

Li, B., Sun, X., Liu, J., Wang, Z., Wu, J., Yu, X., Chen, H., Barsoum, E., Chen, M., and Liu, Z. Latent visual reasoning. arXiv preprint arXiv:2509.24251, 2025a.

Li, H., Li, C., Wu, T., Zhu, X., Wang, Y., Yu, Z., Jiang, E. H., Zhu, S.-C., Jia, Z., Wu, Y. N., et al. Seek in the dark: Reasoning via test-time instance-level policy gradient in latent space. arXiv preprint arXiv:2505.13308, 2025b.

Liao, J., Yang, Z., Li, L., Li, D., Lin, K., Cheng, Y., and Wang, L. Imagegen-cot: Enhancing text-to-image incontext learning with chain-of-thought reasoning. arXiv preprint arXiv:2503.19312, 2025.

Mi, Y., Li, H., Zhao, Y., Li, C., Wu, H., Ma, X., Zhu, S.-C., Wu, Y. N., and Li, Q. Milr: Improving multimodal image generation via test-time latent reasoning. arXiv preprint arXiv:2509.22761, 2025.

Niu, Y., Ning, M., Zheng, M., Jin, W., Lin, B., Jin, P., Liao, J., Feng, C., Ning, K., Zhu, B., et al. Wise: A world knowledge-informed semantic evaluation for textto-image generation. arXiv preprint arXiv:2503.07265, 2025.

Qin, L., Gong, J., Sun, Y., Li, T., Yang, M., Yang, X., Qu, C., Tan, Z., and Li, H. Uni-cot: Towards unified chain-ofthought reasoning across text and vision. arXiv preprint arXiv:2508.05606, 2025a.

Qin, Y., Wei, B., Ge, J., Kallidromitis, K., Fu, S., Darrell, T., and Wang, X. Chain-of-visual-thought: Teaching vlms to see and think better with continuous visual tokens. arXiv preprint arXiv:2511.19418, 2025b.

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748–8763. PmLR, 2021.

Rombach, R., Blattmann, A., Lorenz, D., Esser, P., and Ommer, B. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10684–10695, 2022.

Saharia, C., Chan, W., Saxena, S., Li, L., Whang, J., Denton, E. L., Ghasemipour, K., Gontijo Lopes, R., Karagol Ayan, B., Salimans, T., et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in neural information processing systems, 35: 36479–36494, 2022.

Shen, Z., Yan, H., Zhang, L., Hu, Z., Du, Y., and He, Y. Codi: Compressing chain-of-thought into continuous space via self-distillation. arXiv preprint arXiv:2502.21074, 2025.

Sun, P., Jiang, Y., Chen, S., Zhang, S., Peng, B., Luo, P., and Yuan, Z. Autoregressive model beats diffusion: Llama for scalable image generation. arXiv preprint arXiv:2406.06525, 2024.

Team, C. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024.

Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix, T., Rozi`ere, B., Goyal, N., Hambro, E., Azhar, F., et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.

vivym. midjourney-prompts. https://huggingface. co/datasets/vivym/midjourney-prompts,

2023. Accessed: 2026-01-18.

Wang, X., Caccia, L., Ostapenko, O., Yuan, X., Wang, W. Y., and Sordoni, A. Guiding language model reasoning with planning tokens. arXiv preprint arXiv:2310.05707, 2023.

Wang, X., Zhang, X., Luo, Z., Sun, Q., Cui, Y., Wang, J., Zhang, F., Wang, Y., Li, Z., Yu, Q., et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024a.

Wang, Z., Xie, E., Li, A., Wang, Z., Liu, X., and Li, Z. Divide and conquer: Language models can plan and selfcorrect for compositional text-to-image generation. arXiv preprint arXiv:2401.15688, 2024b.

Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q. V., Zhou, D., et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.

Wu, X., Sun, K., Zhu, F., Zhao, R., and Li, H. Human preference score: Better aligning text-to-image models with human preference. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 2096– 2105, 2023.

Wu, X., Bai, Y., Zheng, H., Chen, H. H., Liu, Y., Wang, Z., Ma, X., Shu, W.-J., Wu, X., Yang, H., et al. Lightgen: Efficient image generation through knowledge distillation and direct preference optimization. arXiv preprint arXiv:2503.08619, 2025.

Xie, J., Mao, W., Bai, Z., Zhang, D. J., Wang, W., Lin, K. Q., Gu, Y., Chen, Z., Yang, Z., and Shou, M. Z. Show-o: One single transformer to unify multimodal understanding and generation. arXiv preprint arXiv:2408.12528, 2024.

Zhang, Y., Xu, W., Zhao, X., Wang, W., Feng, F., He, X., and Chua, T.-S. Reinforced latent reasoning for llm-based recommendation. arXiv preprint arXiv:2505.19092, 2025b.

Zhu, H., Xu, S., Zhang, H., Xiao, T., Guo, Z., Zhou, S., Hu, S., and Honavar, V. G. Reinforcement learning for large language models via group preference reward shaping. In Christodoulopoulos, C., Chakraborty, T., Rose, C., and Peng, V. (eds.), Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 21387–21400, Suzhou, China, November 2025a. Association for Computational Linguistics. ISBN 979-8-89176-332-6. doi: 10.18653/v1/2025.emnlp-main. 1085. URL https://aclanthology.org/2025.

emnlp-main.1085/.

Zhu, R.-J., Peng, T., Cheng, T., Qu, X., Huang, J., Zhu, D., Wang, H., Xue, K., Zhang, X., Shan, Y., et al. A survey on latent reasoning. arXiv preprint arXiv:2507.06203, 2025b.

Xu, Y., Guo, X., Zeng, Z., and Miao, C. Softcot: Soft chain-of-thought for efficient reasoning with llms. arXiv preprint arXiv:2502.12134, 2025.

Xue, Z., Wu, J., Gao, Y., Kong, F., Zhu, L., Chen, M., Liu, Z., Liu, W., Guo, Q., Huang, W., et al. Dancegrpo: Unleashing grpo on visual generation. arXiv preprint arXiv:2505.07818, 2025.

Yang, Z., Wang, J., Li, L., Lin, K., Lin, C.-C., Liu, Z., and Wang, L. Idea2img: Iterative self-refinement with gpt-4v for automatic image design and generation. In European Conference on Computer Vision, pp. 167–184. Springer,

- 2024.

Yang, Z., Yu, X., Chen, D., Shen, M., and Gan, C. Machine mental imagery: Empower multimodal reasoning with latent visual tokens. arXiv preprint arXiv:2506.17218,

- 2025.

Zhan, J., Ai, Q., Liu, Y., Pan, Y., Yao, T., Mao, J., Ma, S., and Mei, T. Prompt refinement with image pivot for text-to-image generation. In Ku, L.-W., Martins, A., and Srikumar, V. (eds.), Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 941–954, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.53. URL https: //aclanthology.org/2024.acl-long.53/.

Zhang, G., Fu, M., and Yan, S. Memgen: Weaving generative latent memory for self-evolving agents. arXiv preprint arXiv:2509.24704, 2025a.

### A. Training Details of LatentMorph

#### A.1. More Details of Two-Stage Training

We employ a two-stage training pipeline to progressively equip the model with latent reasoning capabilities. We provide more details of the two training stages in this section:

- A.1.1. STAGE I: SUPERVISED FINE-TUNING (SFT) FOR LATENT CONTROL

The primary goal of the SFT stage is to train the long-term condenser (Clong), latent translator (Ttrans), and latent shaper (Sshaper) to generate effective control signals from visual history.

Random Injection Strategy. Since we do not have ground-truth data indicating when to reason, we employ a randomized injection strategy during SFT. For a given text-image pair (T,I), we randomly sample an intervention step k from a uniform distribution k ∼ U[kmin,kmax], where we set [kmin,kmax] = [150,450] to avoid edge effects at the very beginning or end of generation.

Forward Process & Loss. At step k, we extract the history H1:k and compute the control tokens Ectrl via the long-term reasoning branch:

m(l) = Clong(H1:k), z = UMMu(m(l),T), Ectrl = Sshaper(Ttrans([z;m(l);p])). (13) Crucially, to maintain training efficiency and autoregressive consistency, we inject Ectrl directly into the KV cache of UMMg at step k. This allows us to compute the loss for all subsequent tokens xk+1:|X| in a single forward pass without physically splicing the sequence. The optimization objective is the standard negative log-likelihood over the image tokens, conditioned on the latent injection:

LSFT = −

|X|

t=1

log pθ(xt|x<t,T,It>k · Ectrl), (14)

where It>k indicates that the control tokens only influence predictions after step k.

- A.1.2. STAGE II: REINFORCEMENT LEARNING (RL) FOR ADAPTIVE INVOCATION

In the second stage, we freeze the modules trained in SFT and focus on optimizing the reasoning invoker (Iinvoker) and short-term condenser (Cshort). We utilize Group Relative Policy Optimization (GRPO) (Guo et al., 2025a) to encourage the model to invoke reasoning only when necessary to improve alignment.

State Space Construction. To enable the policy πθ to make informed decisions, we construct a comprehensive state representation si at each check interval (every w = 32 tokens). The state si concatenates four specific signals extracted from the generation stream:

- • Semantic Consistency: Cosine similarity between the short-term memory and prompt: ci = cos(m(s),p).
- • Temporal Dynamics: The change in consistency ∆ci = ci − ci−w.
- • Stability: The variance of consistency scores over the recent window vi = Var(ci−w:i).
- • Uncertainty: The entropy of the current token distribution ui = H(p(xi)).

GRPO Formulation. For each prompt T, we generate a group of G = 8 trajectories {τ1,...,τG}. For each trajectory, Iinvoker samples an action at ∈ {CONTINUE,REASON} at each interval. The policy gradient is estimated as:

G

R(τj) − R¯

1 G

σR ∇θ log πθ(at,j|st,j), (15) where R¯ and σR are the mean and standard deviation of rewards within the group.

∇θJ =

j=1 t

Reward Shaping and Penalty. The trajectory reward R(τ) is a weighted sum of the CLIP score (RCLIP) (Radford et al., 2021) and Human Preference Score (HPS-v2.1) (RHPS) (Wu et al., 2023), with weights wclip = 1.0 and whps = 1.0 respectively. To prevent the model from trivially invoking reasoning at every step, we enforce the penalty term (Equation 9) with an adaptive threshold. Specifically, we penalize the policy if the average invocation probability p¯ deviates from a reference level derived from high-quality samples in the group.

#### A.2. More Details of Parameter Configurations

Table 2. Hyperparameter configurations for LatentMorph.

Setting Hyperparameter Value

Model Architecture

Cshort memory tokens (ns) 4 attention heads 4 MLP ratio 2.0

Clong memory tokens (nl) 8 attention heads 8 chunk size (c) 64 streaming accumulation FP32

Ttrans hidden ratio 2.0 max scale (gate) 1.0

Sshaper control tokens (j) 4 Iinvoker input dimension 4

hidden dimension 32 layers 2

Stage I: Supervised Fine-Tuning (SFT) Optimization optimizer AdamW (β1 = 0.9, β2 = 0.999) learning rate 1e − 4 weight decay 0.0 global batch size 64

Injection random injection range [150, 450] Data image resolution 384 × 384

Stage II: Reinforcement Learning (RL)

Optimization algorithm GRPO group size (G) 8 learning rate 1e − 5 advantage clipping ±5.0

check interval (w) 64 entropy coefficient 0.001 penalty lambda (λ) 0.2

Policy

Reward weights (CLIP / HPS) 1.0/1.0

#### A.3. Compatibility Analysis

In this section, we provide a further analysis to demonstrate that LatentMorph is theoretically compatible with various pure autoregressive generator paradigms. Our framework relies on the universal abstraction of Transformer-based autoregressive generation, making it model-agnostic.

- A.3.1. UNIVERSAL AUTOREGRESSIVE FORMULATION

Consider a generic autoregressive image generator UMMg parameterized by θ. Given a condition T, the generation of an image sequence X = (x1,...,x|X|) is modeled as a joint probability decomposed into conditional probabilities:

|X|

pθ(xi|x<i,T) (16)

pθ(X|T) =

i=1

In Transformer-based architectures, the probability of the i-th token is computed based on the hidden state hLi of the final layer L:

##### hli = Attention(hli−1,Hl<i−1 ∪ HT), p(xi|·) = Softmax(WvocabhLi ), (17)

where H<i represents the history of visual hidden states (cached as KV pairs) and HT represents the condition embeddings.

- A.3.2. INTERFACE COMPATIBILITY

LatentMorph interacts with UMMg solely through the latent space interface, defined by the read operation (Cshort, Clong) and the write operation (Sshaper). We define the compatibility conditions as follows:

State Readout Compatibility (Condenser). The condenser modules (Cshort and Clong) operate on the sequence of hidden states H1:i. For any AR generator with hidden dimension dg, we can introduce a linear projection Wproj ∈ Rd

g×d to map the generator’s state space to the reasoning space:

M(s) = CrossAttn(Q,K = WprojHi−w:i,V = WprojHi−w:i). (18)

Since C relies on Cross-Attention, it is invariant to the specific architecture (e.g., number of layers or heads) of UMMg, provided that H is accessible.

Control Injection Compatibility (Shaper). The shaper Sshaper influences the generation by injecting control signals Ectrl ∈ Rj×d into the attention mechanism. Mathematically, this modifies the attention context for subsequent steps t > k:

Attention(qt,K′,V ′) = Softmax

qt[K<t;Kctrl]⊤ √

d

[V<t;Vctrl], (19)

where Kctrl,Vctrl are the keys and values derived from Ectrl. Crucially, this injection maintains absolute positional consistency, making it strictly compatible with Rotary Positional Embeddings (RoPE). The control signals act as a virtual context attached to the intervention step k without shifting the positional indices of subsequent tokens (xt>k), thereby preserving the internal relative distance dynamics. This formulation shows that LatentMorph acts as a seamless extension of the conditioning set without altering model weights θ:

pθ(xt|x<t,T,Ectrl) ≈ pθ(xt|x<t,{T,LatentThoughts}) (20) Thus, generators that utilize a Key-Value cache mechanism is compatible with our injection method without structural changes.

- A.3.3. ADAPTATION TO EXTERNAL-LOOP PARADIGMS

Based on the formulation above, we analyze the external-loop paradigm (i.e., decoupled autoregressive generators) mentioned in Section §1. In this setting, the generator UMMg (e.g., LlamaGen (Sun et al., 2024)) is separate from the reasoning model UMMu (e.g., a VLM). The generator operates on pixel/token space X, while the reasoner operates on a separate semantic space. LatentMorph bridges this gap via the learnable adapters trained in SFT: ❶ Clong acts as a visual encoder, projecting the generator’s specific hidden states Hg into the reasoner’s input space du. ❷ Sshaper acts as a control adapter, projecting the reasoner’s output c back into the generator’s dimension dg and format. This demonstrates that LatentMorph essentially functions as a differentiable neural interface, enabling “System-2” reasoning on any “System-1” autoregressive generator, regardless of whether they share a backbone.

### B. Experimental Details of LatentMorph

- B.1. More Details of Baseline Implementations In this section, we provide detailed implementation configurations for each baseline method included in our comparison:

- • SFT & GRPO: To ensure a fair comparison, we directly fine-tune the vanilla model (i.e., Janus-Pro (Chen et al., 2025d)) using the same data and training configuration as LatentMorph, such as the dual reward setting employed in GRPO.
- • Self-CoT (Deng et al., 2025): We adopt the Self-CoT strategy from Bagel (Deng et al., 2025) as a representative reason-before-generation paradigm. Since Bagel utilizes a diffusion head incompatible with our autoregressive setting, we implement this strategy directly on our vanilla model to facilitate a direct comparison.
- • T2I-R1 (Jiang et al., 2025): T2I-R1 represents a sophisticated reason-before-generation paradigm specifically designed for Janus-Pro, featuring a CoT RL framework with two-level CoT data construction. We directly employ their released pre-trained model for evaluation on benchmarks such as GenEval.
- • TIR (Khan et al., 2025): TIR operates as a test-time optimization strategy following the reason-after-generation paradigm. As the original implementation did not target Janus-Pro, we adapted and deployed TIR on our vanilla model for our experiments.
- • T2I-Copilot (Chen et al., 2025a): T2I-Copilot is a training-free multi-agent system that optimizes T2I generation by

- interleaving between reason-before and reason-after paradigms. Similar to TIR, we adapted this framework to function with Janus-Pro to enable comparative evaluation.
- • MILR (Mi et al., 2025): MILR represents a distinct test-time latent optimization strategy within the internal-loop paradigm. Following LatentSeek (Li et al., 2025b), it approaches reasoning as a latent search process rather than a generative one. Specifically, MILR iteratively optimizes intermediate latent representations during inference by maximizing feedback from an external reward model via policy gradients. Unlike LatentMorph, which learns when to reason in a single forward pass, MILR relies on computationally intensive iterative search to locate optimal latent states. We adopt this strategy regarding it as a paradigm of interleaved reason-before and reason-after.
- • TwiG-ZS & TwiG-RL (Guo et al., 2025b): TwiG is a recent reason-while-generating framework built upon Janus-Pro. Our comparison includes both the zero-shot (ZS) and GRPO-trained versions. Due to the unavailability of the official code, we reproduced TwiG-ZS following the implementation details in the paper. For TwiG-RL, as the training data is also proprietary, we directly report the results on T2I-CompBench(++) as presented in their original publication.

- B.2. More Details of Evaluation Benchmarks In this section, we provide further details on the benchmarks used in our evaluations:

- • GenEval (Ghosh et al., 2023): GenEval is a widely adopted benchmark designed to evaluate the general alignment capabilities of T2I models across a diverse range of prompts.
- • T2I-CompBench (Huang et al., 2023) & T2I-CompBench++ (Huang et al., 2025a): T2I-CompBench focuses specifically on evaluating compositional generation capabilities. We also include its enhanced version, T2I-CompBench++, which expands the evaluation scope to include additional dimensions such as numeracy.
- • WISE (Niu et al., 2025): WISE is a recently proposed benchmark that emphasizes world knowledge. It features prompts that are significantly more abstract and complex than those in standard datasets, making it particularly suitable for evaluating the capabilities of reasoning-augmented T2I generation models.
- • IPV-Txt (Bai et al., 2025): IPV-Txt focuses on scenarios involving counter-intuitive physical phenomena, aiming to test whether models grasp underlying physical laws rather than merely fitting training data distributions. Although originally designed for text-to-video generation, we select a subset of prompts applicable to the image domain to evaluate the model’s understanding of abstract concepts and physical constraints. Inspired by (Chen et al., 2025c;b), we adopt the Impossible Prompt Following (IPF) from (Bai et al., 2025) as the evaluation metric, which measures the alignment between generated images and the semantic intent of impossible prompts, and employ GPT-4o to perform judgments.

- B.3. System Prompt for Reasoning Core

In this section, we detail the system prompt for the reasoning branch UMMu whenever the invoker triggers the reasoning process. Given that the primary focus of LatentMorph lies in the mechanism of interleaving implicit latent reasoning rather than exploring complex prompt engineering strategies, we make it as simple as possible to be practical here.

Reasoning Prompt for UMMu

"""You are an image generation optimizer. Your task is to monitor the status of generated images and provide optimization guidance.

You are given the currently generated image representation:

**{rep_ids}** This representation corresponds to the original generation objective:

**{base_prompt}** Please carefully analyze the current state of images generated and deliberate on how subsequent generation should be optimized to correct potential biases while maintaining semantic, structural, and visual consistency. Do not output any intermediate reasoning, verification, or explanations. Implicitly validate consistency and generate a prompt that enables continued image generation. Output only the continuation prompt used to generate the remaining image tokens. """

Table 3. Evaluation on T2I-CompBench++ (Huang et al., 2025a). We highlight the best and second best results. Method Color↑ Shape↑ Texture↑ 2D-Spatial↑ 3D-Spatial↑ Non-Spatial↑ Numeracy↑ Comeplex↑

Vanilla 63.59 35.28 49.36 20.61 32.94 30.85 41.52 35.59 SFT 64.23 34.56 49.46 20.98 32.90 31.55 41.09 35.80 GRPO 67.90 36.31 52.13 23.01 31.78 31.29 42.01 39.04 Self-CoT (Deng et al., 2025) 68.19 37.89 54.10 21.90 34.09 30.00 39.89 44.01 T2I-R1 (Jiang et al., 2025) 81.30 58.52 72.41 33.78 34.23 30.90 45.32 39.93 TIR (Khan et al., 2025) 68.92 49.12 60.10 21.77 35.48 31.21 41.20 40.12 T2I-Copilot (Chen et al., 2025a) 67.42 47.82 61.34 22.12 34.02 30.41 43.44 41.85 MILR (Mi et al., 2025) 85.08 51.17 69.49 46.13 38.08 30.78 60.21 36.84 TwiG-ZS (Guo et al., 2025b) 73.11 41.55 64.77 21.98 33.68 30.90 36.58 48.16 TwiG-RL (Guo et al., 2025b) 82.49 61.28 73.19 34.06 38.87 31.99 61.93 54.45 LatentMorph (Ours) 84.04 69.46 79.89 50.93 47.00 39.56 62.33 63.60

- B.4. Captions of Figure 5 In this section, we provide the detailed prompts in Figure 5:

- • Middle: (1st) “A person defies physics by walking confidently across the ocean’s surface, their feet remaining completely dry as if treading on invisible solid ground. The calm seawater, which should naturally engulf anyone stepping on it, appears to have transformed into a firm platform beneath their feet. The surrounding marine environment features gentle waves and a distant horizon, making this supernatural feat even more striking against the realistic backdrop.” (2nd) “A photorealistic timelapse captures the surreal fusion of a sunflower and rose, creating a striking hybrid bloom. The distinctive yellow petals of the sunflower gradually interweave with the deep red rose petals, while maintaining the recognizable features of both flowers. The transformation occurs against a soft, natural background in natural daylight.”
- • Right: (1st) “The smooth black river flows next to the tall green trees. The dark, glossy river creates a stark contrast against the lush, verdant foliage.” (2nd) “A large mountain stands tall in the background, its rugged surface covered in snow and ice. In the foreground, a small tree with delicate leaves and a slender trunk stands proudly, its branches reaching towards the sky.” (3rd) “A wise and ancient tree, symbolic of Russian landscapes, particularly those vast and dense forests of Siberia, stands tall against a backdrop of snowy branches and lush greenery, with sunlight filtering through the canopy, creating a serene and culturally rich atmosphere.” (4th) “A portrait in the style of Leonardo da Vinci, featuring an elderly man with a thoughtful expression, dressed in Renaissance garb, sitting in a dimly lit study adorned with classical paintings and ancient books, with a soft golden light illuminating his weathered face and hands.” (5th) “A rustic peasant scene featuring a thatched-roof cottage with weathered brown walls and red clay tiles, surrounded by vibrant sunflowers and lush green fields under a golden late-afternoon sky, painted in the swirling, expressive style of early Van Gogh.” (6th) “A lone cherry blossom tree stands against a backdrop of autumn’s warm hues, its branches heavy with soft pink blossoms set against a golden yellow sky at sunset.”

- B.5. More Details of User Study

To quantitatively evaluate the cognitive alignment between LatentMorph’s invocation policy and human intuition, as shown in Figure 6 (Right), we conduct a controlled user study.

We first invite 10 human evaluators with experience in visual generation. The evaluation set consists of 20 complex prompts randomly sampled from the WISE and T2I-CompBench benchmarks, ensuring a diverse range of reasoning difficulties. An interactive interface is then developed to simulate the autoregressive generation stream. The procedure is defined as follows: ❶ Streaming Simulation: For each prompt, the image is generated token-by-token. To match the model’s internal monitoring granularity, the process pauses at every check interval of w = 64 tokens. ❷ Blind Judgment: At each pause, the evaluator is presented with the intermediate image decoded from the current partial tokens alongside the text prompt. The evaluator is then asked a binary question: “Given the current progress and the prompt, is it necessary to pause and reason to correct potential errors or refine details?” ❸ Decision Collection: The human decisions (Invoke/Continue) are recorded without revealing the model’s actual choices to avoid bias. The Human Alignment score is calculated as the percentage of intervals where the model’s action (ak ∈ {REASON,CONTINUE}) matches the majority vote of the human evaluators.

### C. More Results & Sensitivity Analysis

#### C.1. Results on T2I-CompBench++

To provide a more granular evaluation of compositional generation capabilities, we extend our evaluation to T2ICompBench++, as shown in Table 3. Consistent with the findings in the main context, LatentMorph demonstrates superior performance across all evaluated dimensions. While recent external reason-while-generation baselines, e.g., TwiG-RL, have made strides in counting capabilities, i.e., Numeracy, LatentMorph pushes the boundary further. We observe consistent gains in additional complex tasks like 3D-Spatial and Numeracy beyond T2I-CompBench. We attribute this to the fact that while explicit text can convey basic quantities, the precise spatial arrangement and disentanglement of multiple objects are better modulated through continuous latent guidance, enabling LatentMorph to resolve intricate spatial-numerical constraints more effectively.

#### C.2. More Analysis of Condensers

In this section, we investigate the impact of visual memory capacity on generation performance by varying the latent token lengths for both the short-term Cshort and long-term Clong condensers. The results are shown in Figure 7.

[Figure 67]

[Figure 68]

Figure 7. Ablation study of latent memory of Cshort (Left) and Clong (Right).

Length of Short-Term Memory. We examine the sensitivity of the short-term condenser by scaling the memory length ns ∈ {2,4,8,16}. As shown in Figure 7 (Left), we observe a consistent improvement in generation quality as the memory token length increases. This suggests that a larger local memory buffer ns enhances the model’s ability to capture fine-grained local dynamics, thereby providing the invoker Iincoker with more discriminative features for robust decision-making.

Length of Long-Term Memory. Similarly, we also ablate the long-term memory length nl ∈ {4,8,16,32}. As demonstrated in Figure 7 (Right), expanding the global memory size yields progressive performance gains. We attribute this to the increased representational capacity of the visual memory. A larger nl effectively alleviates the information bottleneck, enabling the reasoning core UMMu to access a more granular and comprehensive summary of the entire generation history.

#### C.3. More Analysis of Invoker

In this section, we provide a granular analysis of the invoker Iinvoker, validating its design choices regarding invocation timing, monitoring granularity, and state representation.

Invocation Timestep. We first evaluate the necessity of our learned adaptive policy against heuristic baselines. Table 4 compares our method with random injection i.e., with probabilities p ∈ {0.3,0.5,0.7}, and fixed schedules i.e., injecting once at the midpoint or twice at 1/3 and 2/3 intervals. The results demonstrate that our adaptive strategy consistently outperforms both stochastic and rigid interventions. We attribute this to context awareness: while fixed strategies may intervene during trivial generation phases (e.g., wasting computation) or miss critical turning points, our RL-trained Iinvoker learns to trigger reasoning precisely when semantic drift or high uncertainty is detected.

Table 4. Ablation study of invocation strategies. Invocation Color ↑ Shape ↑ Texture ↑ Spatial ↑ Non-Spatial ↑ Comeplex ↑ Overall ↑

Random (p = 0.3) 78.72 63.71 76.00 45.73 37.67 51.73 58.93 Random (p = 0.5) 78.56 64.83 76.07 48.00 37.93 51.87 59.54 Random (p = 0.7) 77.42 65.63 76.07 49.60 39.20 51.67 59.93

- Fixed (1) 78.28 65.29 73.78 46.73 37.80 51.20 58.85

- Fixed (2) 77.61 63.29 73.33 46.27 38.33 50.87 58.28 Ours 84.04 69.46 79.89 50.93 39.56 63.60 64.53

Window Size of Checking. We examine the impact of the monitoring window size w in Table 5. We observe that performance is sensitive to the temporal resolution of monitoring. A small window w = 32 tends to capture local noise in token prediction, leading to erratic decision-making, while an overly large window w = 128 suffers from lag, failing to detect rapid compositional shifts in time. Our default setting of w = 64 offers an optimal trade-off, providing a stable yet responsive signal for the invoker.

Table 5. Ablation study of check interval. Window Size Color ↑ Shape ↑ Texture ↑ Spatial ↑ Non-Spatial ↑ Comeplex ↑ Overall ↑

w = 32 81.67 66.47 77.56 49.40 38.06 57.00 61.69 w = 64 (Ours) 84.04 69.46 79.89 50.93 39.56 63.60 64.53 w = 128 82.01 66.10 76.89 49.68 40.05 60.45 62.53

Multi-dimensional Signals. We further conduct an ablation study on the components of the state vector si in Table 6. The results indicate that all four signal dimensions, i.e., semantic consistency ci, uncertainty ui, temporal dynamics ∆ci, and stability vi, are essential for robust performance. Notably, removing uncertainty ui or semantic consistency ci leads to the most significant performance drops, confirming that model confidence and alignment scores are the primary indicators for determining the necessity of latent reasoning.

Table 6. Ablation study of state vector si. State Signal Color ↑ Shape ↑ Texture ↑ Spatial ↑ Non-Spatial ↑ Comeplex ↑ Overall ↑

w/o Semantic Consistency ci 79.41 65.90 78.13 48.37 39.20 57.93 61.49 w/o Uncertainty ui 79.10 66.32 78.61 49.70 38.32 59.55 61.93 w/o Temporal Dynamics ∆ci 82.78 68.77 79.14 50.57 38.30 61.39 63.49 w/o Stability vi 83.01 68.24 78.99 50.21 39.19 62.00 63.61 Ours 84.04 69.46 79.89 50.93 39.56 63.60 64.53

#### C.4. More Analysis of Translator

In this section, we dissect the composition of the control signals generated by the latent translator Ttrans. Recall that our design in Section §4.3 fuses the latent thought z with two critical context signals, i.e., the long-term visual memory m(l) and the original prompt embedding p. We evaluate the contribution of each component in Table 7.

Control Signals. The ablation results demonstrate that relying solely on latent thoughts z is insufficient for precise control, and both visual context and textual grounding are indispensable.

❶ Impact of Visual Memory: Removing the long-term visual memory m(l) leads to a marked decline in performance. This indicates that m(l) provides essential historical context, allowing the translator to align abstract reasoning with the visual content generated so far. Without it, the control signals risk disrupting global visual consistency.

❷ Impact of Prompt Embedding: Similarly, omitting the prompt embedding p results in inferior alignment scores. We attribute this to the role of p as a semantic anchor, which ensures that the translated control guidance remains strictly grounded in the user’s original intent, preventing the reasoning process from drifting into unconstrained generation.

Table 7. Ablation study of control signals. Contol Signal Color ↑ Shape ↑ Texture ↑ Spatial ↑ Non-Spatial ↑ Comeplex ↑ Overall ↑

w/o Visual Memory m(l) 82.52 66.12 76.90 48.00 38.39 61.98 62.40 w/o Prompt Embedding p 82.20 67.42 77.01 47.21 38.10 62.10 62.34 Ours 84.04 69.46 79.89 50.93 39.56 63.60 64.53

#### C.5. More Analysis of Shaper

Finally, in this section, we validate the efficacy of the latent shaper Sshaper by investigating how reasoning signals are integrated into the generation stream. We compare our KV-cache injection strategy against a baseline variant “w/o Sshaper”, where the translated latent control signals directly replace the original prompt embeddings p instead of being appended as additional context. The results are shown in Table 8.

Control Injection. The experimental results indicate that the direct replacement strategy, i.e., w/o Sshaper, yields suboptimal performance compared to our additive injection approach. We attribute this to the fact that overwriting the prompt embedding is destructive, as it severs the model’s link to the global generation objective, causing it to lose track of the initial user

instruction while focusing on local refinements. In contrast, our Sshaper injects control tokens Ectrl into the KV cache, acting as a soft modulation mechanism. This design preserves the integrity of the original prompt while seamlessly steering the attention dynamics based on latent reasoning cues, ensuring that refinement does not come at the cost of global semantic fidelity.

Table 8. Ablation study of control injection. Contol Injection Color ↑ Shape ↑ Texture ↑ Spatial ↑ Non-Spatial ↑ Comeplex ↑ Overall ↑

w/o Sshaper 79.42 65.42 77.21 47.32 37.90 60.21 61.25 Ours 84.04 69.46 79.89 50.93 39.56 63.60 64.53

- D. Exhibition Board We provide more comparison results here in Figure 8 and 9.

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

w/oReason+LatentMorph (Ours)

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

A single ripe orange with clearly visible natural peel texture and subtle color variation, ...

Three apples placed on a wooden table, with one red apple in the center foreground in sharp ...

... blurred tall grass framing the foreground, a lone traveler standing on a cliff edge in the ...

A closed hardcover book with a plain solid-colored cover placed on a flat surface indoors, ...

... wildflowers and insects in the foreground, a lone tree ... in the midground, and rolling clouds ...

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

w/oReason+LatentMorph(Ours)

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

A historic stone bridge at dusk with textured arches and mossy stones in the foreground, ...

A long suspension bridge scene with steel cables and bolts sharply visible in the foreground, ...

A dense pine forest after snowfall with snow-covered branches partially obscuring the view ...

A dense old-growth forest with fallen logs and mushrooms sharply defined in the ...

A dramatic coastal cliff scene ..., a seabird standing near the cliff edge in the midground ...

Figure 8. More results demonstration of LatentMorph.

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

Janus-Pro+LatentMorph (Ours)

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

A metallic desk lamp with a fluffy blanket, placed on a wooden table, illuminated by a warm, soft light, ...

Two swans, positioned in a serene lake, with their graceful necks extended towards the sky, ...

A black cat with a white whisker, standing in front of a white background, with its body ...

Three apples rest on the countertop, each possessing a unique hue and texture. Bright, warm ...

A green jay and a blue parrot, both perched on a tree branch, with the green jay on the left ...

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

Janus-Pro+LatentMorph(Ours)

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

A rabbit near a train, with a detailed description of its position, proportions, and fine details, ...

The warm light shone down on the colorful fruit market, casting a soft glow on the vibrant ...

The twisted, gnarled roots of the old oak tree snaked through the rich soil, anchoring it firmly to ...

A round doughnut and a square napkin, placed on a wooden table, casting soft shadows on the ...

A white dome with unique architecture, representing the United States' political history, ...

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

Janus-Pro+LatentMorph(Ours)

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

A bright feather stands tall on the hillside, accompanied by a flock of birds dancing lightly ...

A detailed close-up view and a broad, distant perspective of the same majestic mountain range, ...

A cozy and warm fabric blanket with a fluffy and soft fluffy blanket, both placed on a wooden floor, ...

A fabric blanket and a glass cup, placed on a wooden table, with a warm, soft lighting from the ...

A blue backpack and a brown bear, positioned in a forest setting, with the bear standing to the ...

Figure 9. More results demonstration of LatentMorph.

