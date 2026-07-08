# arXiv:2606.14667v1[cs.CV]12Jun2026

## Memento: Reconstruct to Remember for Consistent Long Video Generation

Xuan Wei1,2,‡ Longbin Ji2 Guan Wang2 Xiangrui Liu2,‡ Zhenyu Zhang2,§ Shuohuan Wang2 Yu Sun2 Qingqi Hong1,†

1Xiamen University 2ERNIE Team, Baidu Inc. weixuan@stu.xmu.edu.cn, {jilongbin,zhangzhenyu07}@baidu.com

‡Work done during internship at Baidu. §Project Lead. †Corresponding author.

### Abstract

Long-form video generation requires recurring subjects to remain consistent across various shots, viewpoints, motions, and scene transitions. Existing temporal decomposition methods improve scalability by generating videos shot by shot. However, they mainly focus on optimizing plausible next-shot continuations without verifying whether the historical memory preserves identity-critical subject evidence. Consequently, as generation proceeds, recurring subjects may be diluted, overwritten, or forgotten. In this paper, we propose Memento, a subject-reconstruction-guided framework that treats subject preservation as an explicit identity grounding problem, based on the premise that a memory bank faithfully preserving a subject should support reconstructing that subject from memory alone. Specifically, Memento jointly trains autoregressive next-shot generation with memory-based subject reconstruction, recovering target appearances using historical memory and global story captions. To disentangle long-range subject evidence from short-range cues, Memento introduces a dual-query memory mechanism, where one query retrieves identity-relevant memory and the other selects short-context keyframes for coherent continuation. Additionally, a subject-aware cinematic data pipeline provides precise reconstruction supervision via consistent, pronoun-free subject descriptions. Experiments demonstrate that Memento achieves state-of-the-art performance in long-term subject consistency, cross-shot coherence, and visual quality.

### 1 Introduction

Recent diffusion-based video generation models have achieved remarkable progress in visual fidelity, motion realism, and text alignment [8, 21, 3, 23, 26, 14, 31, 5, 15, 19, 32, 24]. Despite these advances, generating long-form and story-driven videos remains challenging. A long video is not merely an extended sequence of visually plausible frames, it requires recurring subjects to maintain their identities across multiple shots, viewpoints, poses, motions, and scene transitions. However, subtle changes in face, clothing, body shape, or object appearance can accumulate over time, causing subject drift and breaking narrative coherence.

Existing long video generation methods typically address this challenge through temporal decomposition. Specifically, storyboard-based approaches [39, 30, 36, 38, 10, 7, 29, 35, 16, 11] synthesize sparse keyframes before animating them into clips, while joint multi-shot methods [6, 18, 28, 22, 13, 4, 12] generate multiple shots within a shared diffusion context. Although these strategies improve crossshot coordination, they either leave individual clips largely independent or remain constrained by context length and escalating long-range attention costs. Alternatively, memory-conditioned autoregressive methods [33, 1] offer a scalable alternative by generating videos shot by shot with compact

Preprint.

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

Queen on throne with map Advisor reassures her Queen’s expression softens Advisor’s wise eyes Queen rises to decide

Counsel in the Throne Room (subtle interactions)

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

Astronaut floats in cabin Astronaut steps out lander Places flag beside rocks Kneels to pick up rock Inside cabin, helmet off

Footprints on the Moon (large-scale motions)

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

Robinson clings to mast Lights campfire at dusk Sees footprints on beach Unties frightened man Waves toward ship

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

Robinson Crusoe (complex scene transitions)

- Figure 1: We show representative results covering progressively challenging settings. The examples illustrate Memento’s ability to preserve different subject identities and background coherence over long temporal horizons. Whether handling subtle interactions, large-scale motions, or complex scene transitions in a survival story, our method ensures natural motion and consistent character appearance across multiple shots.

historical memory. However, their memory selection mechanisms are usually driven by generic relevance, visual saliency, or short-term context compression, rather than explicit identity preservation. As generation proceeds, identity-critical evidence of recurring subjects may be gradually diluted.

We contend that subject consistency in long video generation is fundamentally an identity grounding problem. Existing memory-conditioned methods typically use historical frames as references for next-shot generation, where the training signal primarily rewards plausible local continuation. While such supervision can improve temporal coherence in the short term, it does not explicitly guarantee that the memory retains fine-grained identity evidence for recurring subjects. Consequently, the model may learn to exploit memory for scene layouts or visual continuity, while failing to preserve the critical appearance cues required to recognize the same subject across distant shots.

To directly supervise identity preservation, we introduce subject reconstruction as an auxiliary training objective. Our key intuition is straightforward: if a memory bank truly preserves a subject, the model should be capable of recovering the subject from memory. We therefore require the model to reconstruct target subject appearances using only historical memory and the global story caption, without relying on direct visual prompts. This transforms subject consistency from an implicit expectation of next-shot generation into an explicitly verifiable objective, thereby forcing the memory to encode and retrieve identity-relevant cues such as face, clothing, body structure, and other stable appearance attributes.

This reconstruction-guided perspective further suggests that a single entangled memory is suboptimal for long video generation. Subject reconstruction and next-shot generation depend on different forms of historical information. Specifically, reconstruction benefits from long-range identity evidence that should remain stable across scene changes and distant shots. In contrast, next-shot generation relies

more heavily on short-range contextual cues, such as recent layout, motion tendency, and camera continuity. Consequently, when both objectives share the same memory selection policy, visually salient recent context can easily dominate the memory, while sparse but identity-critical evidence may be suppressed. This motivates a disentangled memory design that separately retrieves long-context subject evidence for identity grounding and short-context visual references for local shot generation.

Based on this design, we propose Memento, a subject-reconstruction-guided framework for consistent long video generation. Given a global story caption and a sequence of per-shot captions, Memento generates videos autoregressively at the shot level while maintaining a fixed-length historical memory bank. At each generation step, the memory candidate pool is formed from previous memory tokens and visual features of the most recent shot. A dual-query memory mechanism is then introduced to retrieve complementary historical references: story-conditioned long-context queries select subjectrelevant memories for identity preservation, whereas shot-conditioned short-context queries capture local references for the upcoming shot. The retrieved memories are used jointly for diffusionbased next-shot generation and memory-based subject reconstruction, enabling scalable long video generation without full global attention. To support training, we further develop a subject-aware data curation pipeline that generates global story captions, local shot captions, and reconstruction target captions with consistent, pronoun-free subject descriptions, which significantly reduces ambiguity in subject-level supervision.

Our contributions are summarized as follows:

- • We propose a memory-based subject reconstruction objective that explicitly enforces identity preservation for recurring subjects during long video generation.
- • We introduce Memento, a novel subject-reconstruction-guided framework, which leverages a dual-query memory mechanism to separate the retrieval of long-context subject evidence and short-context visual cues, facilitating scalable shot-level autoregressive synthesis.
- • We construct a subject-aware cinematic data curation pipeline with consistent, pronoun-free subject descriptions and reconstruction targets.
- • Extensive experiments demonstrate that Memento achieves superior long-term subject consistency, cross-shot coherence, and visual quality across diverse long-form scenarios.

### 2 Related Work

- 2.1 Single-Shot Video Generation

Recent advances in diffusion models [8] and Diffusion Transformers (DiTs) [21] have substantially improved video generation. Large-scale text-to-video (T2V) and image-to-video (I2V) models [25, 9, 3, 26, 14, 31, 5, 15, 19, 32, 24] can now synthesize short, single-shot clips with high visual fidelity and natural motion. However, these models are primarily designed for temporally bounded single scenes and lack explicit mechanisms for maintaining consistency across multiple shots. Consequently, they remain insufficient for long-form, story-driven video generation, where recurring subjects, scenes, and narrative logic must remain coherent over extended sequences. This limitation motivates the development of multi-shot video generation approaches.

- 2.2 Multi-Shot Long Video Generation

Multi-shot video generation aims to produce long-form narratives by decomposing video synthesis into shot-level generation. Existing methods mainly follow three paradigms.

Storyboard-based generation. Storyboard-based methods [30, 36, 38, 10, 39, 7, 29, 35, 16, 11] first generate sparse keyframes and then animate them with pretrained image-to-video models. Although efficient, they enforce consistency mostly at the keyframe level, leaving individual clips weakly coupled and often causing rigid transitions, detail drift, and limited narrative coherence.

Joint multi-shot generation. Joint generation methods synthesize multiple shots within a single diffusion forward pass. Representative works such as LCT [6] and HoloCine [18] introduce cross-shot attention mechanisms to improve global interaction and shot-level alignment, with later methods further improving efficiency and controllability [28, 22, 13, 4, 12]. However, these methods remain

###### Dual-Path Disentangled Memory Bank

Recon Target Memory Target

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

... ... ...

[story caption] story

[shot caption] shot

learnable query story learnable query shot

VAE Encoder

... ... ...

Caption-Query Cross Attn.

Caption-Query Cross Attn.

recon tokens memory tokens target tokens

story memory

story =  ( story, story) shot =  ( shot, shot )

[Figure 45]

memorybank

[Figure 46]

퐝퐢  

퐫퐞   

Split Self Attention

Memory Cross Attn.

cross attn.

local memory

Split Cross Attention

Q = [ story, shot], K/V=Candidates

[Figure 47]

story

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

shot

Candidates

VAE Decoder

[Figure 53]

update

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

’

Recon Target Memory Target

[Figure 62]

[Figure 63]

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

... ... ...

memory bank  −1 last shot  −1

- Figure 2: Overview of our framework. The left side illustrates the generation process, while the right side details the memory bank update mechanism. We employ split self-attention over overlapping local groups, allowing reconstruction-memory and memory-target interactions while avoiding full global attention. Split cross-attention injects both global story-level and local shot-level captions. Story and shot captions condition separate learnable queries via caption-to-query cross-attention. The fused query retrieves relevant candidates from the memory bank and the last shot, and updates global and local memory states for scalable long-form generation.

bounded by the context window, and their memory and computation costs increase rapidly as the number of shots grows, limiting open-ended generation.

Memory-conditioned autoregressive generation. Autoregressive methods scale better by generating videos shot by shot while conditioning on a compact memory of previous content. StoryMem [33] maintains selected keyframes using semantic and aesthetic criteria, while OneStory [1] compresses historical context for next-shot generation. These designs improve scalability, but their memory selection is primarily based on generic relevance or short-term context compression, rather than explicit subject preservation. As a result, identity-critical cues may be diluted or overwritten over long generation horizons.

In contrast, Memento explicitly formulates long-form consistency as subject grounding in memory. It uses a dual-query memory mechanism to separate long-context subject evidence from short-context generation cues, and jointly optimizes next-shot generation with memory-based subject reconstruction. This encourages the memory to retain identity-critical information while preserving local visual continuity, enabling more stable long video generation.

- 3 Method

Memento follows the principle that consistent long video generation requires both persistent subject evidence and recent contextual cues. To this end, we generate videos autoregressively at the shot level with a compact historical memory bank, and explicitly supervise the memory through subject reconstruction. This design allows the model to preserve recurring identities without attending to all previous frames, while still using recent visual context for local shot continuity.

Given a global story caption Cstory and a sequence of per-shot captions {Cshott }Nt=1, Memento generates a multi-shot video V = S1,...,SN. The first shot is generated by standard text-to-video synthesis. For each subsequent shot t ≥ 2, the model performs memory-conditioned generation using

Cstory, Cshott , and a dynamically updated memory bank Mt. Optionally, the last frame of St−1 can be provided for smoother inter-shot transitions:

##### St = θ Mt, Cstory, Cshott ,St−1 (1)

where St−1 is omitted in the Text-and-Memory-to-Video (TM2V) generation setting. The framework consists of three components: a subject-aware data curation pipeline that provides unambiguous subject supervision, a dual-query memory mechanism that retrieves long-context subject evidence

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

Quality Filtering

Three-Stage Caption

Story Merging

[Figure 86]

Clip Slicing

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

Story Captioning Shot Captioning

Recon Captioning

[Figure 94]

subjects:

[Figure 95]

[Figure 96]

- clip1 caption: [Person A] adjusts his gloves

while standing next to a large vehicle, then turns slightly toward [Person B].

- clip2 caption: [Person A] looks up at a glowing purple grid structure overhead while standing next to [Person B].

- Person A: a man wearing an white

spacesuit

- Person B: a woman with dark hair tied

ByteTrack

back global caption: [Person A] stands beside a large vehicle while adjusting something near its open rear compartment. He turns toward [Person B], who holds the yellow stuffed animal tightly against her chest.

reconstruct caption:

[Figure 97]

- Person A: a man wearing an

white spacesuit

- Person B: a woman with dark

[Figure 98]

[Figure 99]

hair tied back

- Figure 3: Overview of the subject-aware data curation pipeline. We design three-stage captions to support consistent long video generation. Story Caption generates a global narrative and identifies all recurring subjects with fixed, pronoun-free names. Shot Caption produces per-shot descriptions that adhere to the established subject inventory. Recon Caption annotates selected subject-salient frames as reconstruction targets for memory-based identity supervision.

and short-context generation cues, and a subject-anchored multi-task objective that jointly trains next-shot generation with memory-based subject reconstruction.

#### 3.1 Subject-Aware Data Curation Pipeline

Training Memento requires annotations that support both shot-level generation and memory-based subject reconstruction. Beyond conventional video-caption pairs, the data must consistently identify recurring subjects across shots and provide reconstruction targets that contain clear visual evidence of their appearances. However, standard captions often describe subjects with ambiguous references such as ‘the man,” ‘he,” or “she,” making it difficult to associate the same subject across changing scenes, viewpoints, and interactions. To address this issue, we construct a subject-aware data curation pipeline that produces temporally coherent multi-shot videos with explicit, pronoun-free subject descriptions and reconstruction targets.

As shown in Fig. 3, we collect raw footage from complete cinematic productions, including movies, documentaries, and animations. We slice the footage into continuous multi-shot sequences of 30–60 seconds, where each sequence contains multiple natural shots and each shot lasts longer than 5 seconds. Before captioning, we apply quality filtering and crop subtitle regions to reduce visual artifacts that may degrade generation quality.

- Stage 1: Story Captioning. We first process the full multi-shot sequence with Qwen3-VL [2] to generate a global story caption and identify salient recurring subjects throughout the sequence. This stage establishes a subject inventory shared by all subsequent annotations, ensuring that each recurring subject is described with a fixed and explicit textual reference. In parallel, we apply ByteTrack [34] to track the primary subject over time and select two frames where the subject occupies the largest visual area as reconstruction targets. These frames provide high-quality appearance evidence for the reconstruction branch.
- Stage 2: Shot Captioning. Given the global story caption and the subject inventory from Stage 1, we caption each individual shot using Qwen3-VL. Instead of allowing generic noun phrases or pronouns, we enforce the fixed subject descriptions when referring to recurring subjects. This aligns local shot captions with the global subject identities and reduces ambiguity in cross-shot subject association, especially when multiple visually similar subjects appear in the same sequence.
- Stage 3: Reconstruction Target Captioning. Finally, we caption the selected reconstruction target frames using Qwen3-VL. The generated captions are constrained to remain consistent with the global story caption and the established subject descriptions, so that each target image is paired with an unambiguous subject reference. These target image-caption pairs are used to supervise memory-based subject reconstruction, encouraging the memory bank to preserve identity-relevant appearance cues rather than only short-term visual context.

#### 3.2 Dual-Path Disentangled Memory Bank

A fixed-size memory is necessary for scalable autoregressive long video generation, but a single memory selection criterion can entangle different types of historical information. In our setting, subject reconstruction requires long-range identity evidence, while next-shot generation relies more on recent scene context. We therefore design a dual-path memory mechanism that retrieves subjectrelevant and shot-relevant memories separately from a shared candidate pool.

Memory bank construction. At generation step t, Memento maintains a fixed-length memory bank Mt that summarizes useful historical visual information. Before generating the t-th shot, we form a candidate memory pool by combining the previous memory bank Mt−1 with visual features extracted from the most recently generated shot St−1:

cand×D (2) where E(·) denotes a pretrained VAE encoder that maps frames into latent features of dimension D, and Ncand is the number of candidate memory tokens. This construction allows the memory to retain selected information from earlier shots while continuously incorporating newly generated visual evidence.

Mtcand = [Mt−1 ; E(St−1)] ∈ RN

Dual-query disentanglement. To separate identity-oriented retrieval from context-oriented retrieval, we introduce two sets of learnable query tokens: story queries Qstory ∈ RN

story×D and shot queries Qshot ∈ RN

shot×D. The story queries are conditioned on the global story caption to retrieve subject evidence that should remain stable across shots, while the shot queries are conditioned on the current shot caption to retrieve local references useful for the upcoming generation:

Qˆstory = A(Qstory, ϕ(Cstory)), Qˆshot = A Qshot, ϕ(Cshott ) (3) where ϕ(·) encodes text conditions into the shared latent space and A(·,·) denotes cross-attention. Through this conditioning, Qˆstory is guided by the global narrative and recurring subject descriptions, whereas Qˆshot focuses on the local semantics of the current shot.

Adaptive memory selection. The two query sets independently score the candidate memory pool:

sstory = S Q ˆstory, Mtcand , sshot = S Q ˆshot, Mtcand (4)

where S(·,·) denotes the relevance scoring function. We then select the top-K memory tokens for each path and concatenate the selected subsets to form the updated memory bank:

Mt = TopK Mtcand, sstory ; TopK Mtcand, sshot (5)

where TopK(M,s) retrieves the top-K tokens from M according to score s. The story-selected memories provide long-context subject evidence for identity grounding and reconstruction, while the shot-selected memories provide short-context visual cues for local generation. By decoupling the two retrieval paths, the memory bank reduces competition between persistent identity information and transient scene context.

#### 3.3 Subject-Anchored Multi-Task Training

The dual-path memory bank provides candidate identity and context evidence, but next-shot generation alone does not guarantee that the selected memory contains sufficient subject information. We therefore introduce subject-anchored multi-task training, which couples standard next-shot video denoising with memory-based subject reconstruction. This auxiliary objective shares the memoryconditioned generator with next-shot denoising, adding identity supervision without extra training stages or inference-time overhead.

Next-shot generation. The primary task is conditional diffusion denoising for the current shot. Given the noisy shot latent Stτ at diffusion timestep τ, the model predicts the added Gaussian noise conditioned on the memory bank Mt, the global story caption Cstory, and the current shot caption Cshott :

Ldiff = E ϵ − ϵθ Stτ,τ,Mt,Cstory,Cshott 2 (6)

where ϵ ∼ N(0,I) denotes the added Gaussian noise. This objective trains the model to generate the next shot from both textual instructions and historical visual memory.

Subject-role reconstruction. To make identity preservation directly optimizable, we formulate an auxiliary Text-and-Memory-to-Image (TM2I) reconstruction task. Given a target subject image Isub, the model predicts its diffusion noise conditioned only on the memory bank Mt and the global story caption Cstory, without using any direct visual prompt:

Lrecon = E ϵ − ϵθ Isubτ ,τ,Mt,Cstory 2 (7)

Since the reconstruction branch cannot access the target image except through its noisy latent, successful denoising requires the model to retrieve identity-relevant appearance cues from memory under the guidance of the story caption. This discourages the memory from serving only short-term generation context and anchors it to recurring subject identities.

Total loss. The final training objective combines next-shot generation and subject reconstruction:

Ltotal = Ldiff + λLrecon (8)

where λ is a balancing hyperparameter that controls the trade-off between scene progression and subject reconstruction fidelity.

### 4 Experiments

We conduct extensive experiments to evaluate the effectiveness of Memento in generating long videos with high subject consistency and temporal coherence. First, we provide implementation details on model training and inference (Sec. 4.1). Next, we provide a comprehensive evaluation against state-of-the-art multi-shot video generation methods through quantitative metrics, qualitative visual results, and a user study (Sec.4.2). Then, we investigate the individual contributions of our core components in Sec.4.3. Finally, we show advanced capabilities of our method in Sec.4.4.

- 4.1 Implementation Details

Training Setup. Our video generation backbone is built upon Wan2.2 [26] 14B. The training data is processed through our subject-aware data curation pipeline, resulting in 20K clips at a resolution of 480 × 832. We employ the AdamW [17] optimizer with a base learning rate of 1e − 5. The model is trained on 16 NVIDIA H100 GPUs. The loss weight parameter λ in Eq. 8 is empirically set to 0.5. To facilitate robust classifier-free guidance (CFG) during inference, we randomly drop the textual conditions Cstory and Cshot with a probability of 10% during training.

Inference Configurations. We follow the STBench autoregressive evaluation protocol and adopt the scene transition strategy of StoryMem [33]. The first shot S1 is generated from Gaussian noise with standard T2V. For later shots, we use TMI2V within the same scene, injecting the last frame of the previous shot for pixel-level continuity, and switch to TM2V across scene changes, relying on text prompts and the updated memory bank Mt for long-term identity preservation. We use the UniPC [37] sampler with 40 denoising steps for both generation and reconstruction, and set the CFG scale for global and local text conditions to 3.5.

- 4.2 Comparison

- 4.2.1 Quantitative Results

We evaluate generated videos along four dimensions: aesthetic quality, semantic consistency, background consistency, and subject consistency. Subject consistency is measured at three temporal granularities: intra-shot, inter-shot, and inter-scene, capturing stability within a shot, across consecutive shots, and across scene transitions.

We compare Memento with representative methods from the three main paradigms: StoryDiffusion+Wan2.2-I2V [39, 26] for storyboard-based generation, StoryMem [33] for memoryconditioned autoregressive generation, and HoloCine [18] for joint multi-shot generation. For HoloCine, we compute shot- and scene-level metrics at their corresponding granularities to avoid biases from temporal slicing.

As shown in Table 1, Memento achieves the best performance on long-term subject consistency. It obtains the highest inter-shot consistency of 0.7338, outperforming StoryMem (0.6606), the highest

- Table 1: Quantitative comparison of video generation capabilities. We report aesthetic quality, semantic consistency at both story and shot levels, background consistency, and subject consistency across different granularities. The best results are highlighted in bold, and the second-best results are underlined.

Semantic Consistency Background Consistency

Subject Consistency Story Shot Inter-shot Intra-shot Inter-scene

Method Aesthetic

StoryDiffusion + Wan2.2-I2V [39] 0.5310 0.2671 0.2689 0.9767 0.5525 0.8448 0.6732 StoryMem [33] 0.4937 0.2793 0.2681 0.9732 0.6606 0.8146 0.6692 HoloCine [18] 0.4568 0.2720 0.2854 0.9770 0.5791 0.8128 0.6594

###### Ours 0.4977 0.3063 0.2893 0.9805 0.7338 0.8578 0.7268

PhD student typing at dorm room computer

PhD Student surprised by results

PHD Student shows results to his professor

Student walks through airport pulling a suitcase

Student presents on conference hall stage

StoryMemStoryMemOursHoloCineOursHoloCine

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

Jack boards eagerly, contrasting with passengers nearby.

Rose gazes at sea. Jack and Rose meet by the railing at sunset.

Rose laughs as Jack steadies her.

Amid chaos, Jack and Rose hold hands.

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

- Figure 4: Qualitative comparison of multi-shot video generation against StoryMem [33] and HoloCine [18]. Each row shows one method and each column shows a key story moment. Blue/red boxes highlight cross-shot subject inconsistencies, while red boxes also mark reconstruction frames. Our method better preserves subject identity across diverse scenes, viewpoints, and story progression.

intra-shot consistency of 0.8578, surpassing StoryDiffusion+Wan2.2-I2V (0.8448), and the highest inter-scene consistency of 0.7268, surpassing StoryDiffusion+Wan2.2-I2V (0.6732). These results indicate stronger identity preservation across both consecutive shots and scene transitions. Memento also achieves the best story-level semantic consistency (0.3063) and shot-level semantic consistency (0.2893), suggesting improved narrative coherence. Meanwhile, it obtains the second-best aesthetic quality (0.4977), showing that stronger subject preservation does not compromise visual fidelity or scene stability.

#### 4.2.2 Qualitative Results

Fig. 4 compares long multi-shot generation results from StoryMem [33], HoloCine [18], and our method. Each column shows a key story moment, and each row corresponds to one method. The highlighted regions indicate visible subject inconsistencies across shots.

[Figure 133]

- Figure 5: User study results. We report pairwise human preference comparisons between our method and each baseline over 30 generation cases evaluated by 10 participants. For each criterion, Win, Tie, and Lose indicate the percentage of comparisons where our method is preferred over, judged comparable to, or preferred less than the corresponding baseline.

StoryMem preserves identity in early shots but gradually drifts as generation proceeds, suggesting that its selected memory frames do not always retain subject cues required for future shots. HoloCine achieves reasonable within-scene consistency, but suffers from large identity changes across scene transitions, especially when viewpoint, background, or narrative context changes.

In contrast, our method maintains more stable subject appearance across both intra-scene motion and cross-scene transitions. The reconstruction frames highlighted in red provide explicit subject anchors, helping preserve identity under changes in camera view, lighting, background, and story context. These results show that subject-aware captioning and reconstruction-guided training effectively reduce identity drift and improve long-term narrative coherence.

#### 4.2.3 User Study

To complement automatic evaluation, we conduct a user study with 10 participants on 30 diverse generation cases. For each case, evaluators perform pairwise comparisons between our method and each baseline, including StoryMem [33] and HoloCine [18]. The comparisons are conducted under three criteria: cross-shot consistency, prompt following, and aesthetic quality. For each criterion and each baseline, evaluators choose whether our method is better, comparable, or worse, corresponding to Win, Tie, and Lose, respectively. The reported percentages are computed over 300 pairwise judgments, corresponding to 30 cases evaluated by 10 participants.

As shown in Fig. 5, our method is consistently preferred over both baselines across all three criteria. For cross-shot consistency, our method achieves a win rate of 57.3% against StoryMem and 69.0% against HoloCine, indicating stronger long-term subject preservation across shots. For prompt following, our method also obtains clear advantages, with win rates of 60.3% against StoryMem and 63.0% against HoloCine. In terms of aesthetic quality, our method achieves win rates of 45.3% and 56.7% against StoryMem and HoloCine, respectively, while also receiving substantial tie rates. These results suggest that our method improves perceived cross-shot coherence and prompt adherence while maintaining competitive or better visual quality. “‘

#### 4.3 Ablation Studies

Tab. 2 validates the effectiveness of the proposed reconstruction objective and disentangled memory. Adding subject reconstruction improves inter-shot subject consistency from 0.7227 to 0.7489, indicating that explicit reconstruction encourages the model to preserve and reuse subject identity from memory. However, after introducing disentangled memory, the inter-shot score slightly decreases to 0.7338. This is because the memory bank is no longer optimized only for the immediate next shot; instead, it may introduce memories of other recurring subjects that are not directly related to the next shot. Such additional subject evidence can slightly weaken adjacent-shot similarity, but it provides richer identity cues for long-range consistency.

Despite the slight drop in inter-shot consistency, the disentangled memory module improves other consistency metrics, including semantic consistency, background consistency, intra-shot subject consistency, and inter-scene subject consistency. These results suggest that disentangled memory has a positive effect on global subject preservation. By separating long-term identity evidence from short-term contextual cues, the model can better maintain recurring subjects across the whole video rather than overfitting to local next-shot continuity.

- Table 2: Ablation studies on the individual components of Memento. Starting from the baseline, we cumulatively add each module and report aesthetic quality, semantic consistency (story and shot-level), background consistency, and subject consistency across different granularities. The best results are highlighted in bold.

Semantic Consistency Background Consistency

Subject Consistency Story Shot Inter-shot Intra-shot Inter-scene

Method Aesthetic

Baseline 0.4937 0.2793 0.2681 0.9744 0.6606 0.8146 0.6692 + learnable query 0.4975 0.2793 0.2622 0.9752 0.7227 0.7616 0.7177 + recon. task 0.4901 0.2799 0.2679 0.9723 0.7489 0.7243 0.7082 + disentangled memory 0.4977 0.3063 0.2893 0.9805 0.7338 0.8578 0.7268

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

Age-Consistent Identity Preservation

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

Infinite-Length Generation

- Figure 6: Advanced capabilities of Memento. Top: our method can generate narratives involving age variation while preserving a recognizable subject identity across different life stages. Bottom: our autoregressive multi-shot generation framework scales to long-form video synthesis; we further generate a 5-minute video to demonstrate its ability to maintain coherent story progression and subject consistency over extended duration.

#### 4.4 Advanced Capabilities

Beyond standard multi-shot generation, Memento exhibits two advanced capabilities, as shown in Fig. 6. First, it supports age-consistent identity preservation: the model can generate narratives where the same character changes age while maintaining stable identity cues such as facial structure, hairstyle tendency, and overall appearance. This suggests that Memento does not merely copy a fixed reference, but separates persistent identity from controllable temporal variations. Second, our autoregressive generation with dynamic memory update scales naturally to long-form synthesis. In a 5minute generation example, Memento maintains coherent subject appearance, scene progression, and visual quality across many connected shots, demonstrating practical scalability toward minute-level narrative video generation.

### 5 Conclusion

We present Memento, a subject-reconstruction-guided framework for consistent long video generation. By combining dual-query memory retrieval with memory-based subject reconstruction, Memento explicitly preserves long-term identity evidence while maintaining short-term contextual continuity for next-shot generation. Together with a subject-aware data curation pipeline, Memento achieves strong cross-shot subject consistency, narrative coherence, and visual quality, while naturally extending to long-form generation without architectural modification.

#### 5.1 Limitation and Future Work

Memento has two limitations. First, autoregressive generation may propagate errors when degraded shots are written into memory. Second, physical plausibility remains limited by the video backbone, which lacks explicit modeling of gravity, object permanence, and rigid-body dynamics. Future work will study robust memory filtering and physics-aware generation.

### References

- [1] Zhaochong An, Menglin Jia, Haonan Qiu, Zijian Zhou, Xiaoke Huang, Zhiheng Liu, Weiming Ren, Kumara Kahatapitiya, Ding Liu, Sen He, et al. Onestory: Coherent multi-shot video generation with adaptive memory. arXiv preprint arXiv:2512.07802, 2025.
- [2] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025.
- [3] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023.
- [4] Shengqu Cai, Ceyuan Yang, Lvmin Zhang, Yuwei Guo, Junfei Xiao, Ziyan Yang, Yinghao Xu, Zhenheng Yang, Alan Yuille, Leonidas Guibas, et al. Mixture of contexts for long video generation. arXiv preprint arXiv:2508.21058, 2025.
- [5] Google DeepMind. Veo 3 video model. https://deepmind.google/models/veo/, 2026.
- [6] Yuwei Guo, Ceyuan Yang, Ziyan Yang, Zhibei Ma, Zhijie Lin, Zhenheng Yang, Dahua Lin, and Lu Jiang. Long context tuning for video generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 17281–17291, 2025.
- [7] Jingwen He, Hongbo Liu, Jiajun Li, Ziqi Huang, Qiao Yu, Wanli Ouyang, and Ziwei Liu. Cut2next: Generating next shot via in-context tuning. In Proceedings of the SIGGRAPH Asia 2025 Conference Papers, pages 1–11, 2025.
- [8] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.
- [9] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. Advances in neural information processing systems, 35:8633–8646, 2022.
- [10] Panwen Hu, Jin Jiang, Jianqi Chen, Mingfei Han, Shengcai Liao, Xiaojun Chang, and Xiaodan Liang. Storyagent: Customized storytelling video generation via multi-agent collaboration. arXiv preprint arXiv:2411.04925, 2024.
- [11] Lianghua Huang, Wei Wang, Zhi-Fan Wu, Yupeng Shi, Huanzhang Dou, Chen Liang, Yutong Feng, Yu Liu, and Jingren Zhou. In-context lora for diffusion transformers. arXiv preprint arXiv:2410.23775, 2024.
- [12] Weinan Jia, Yuning Lu, Mengqi Huang, Hualiang Wang, Binyuan Huang, Nan Chen, Mu Liu, Jidong Jiang, and Zhendong Mao. Moga: Mixture-of-groups attention for end-to-end long video generation. arXiv preprint arXiv:2510.18692, 2025.
- [13] Ozgur Kara, Krishna Kumar Singh, Feng Liu, Duygu Ceylan, James M Rehg, and Tobias Hinz. Shotadapter: Text-to-multi-shot video generation with diffusion models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 28405–28415, 2025.
- [14] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.
- [15] Kuaishou. Kling video model. https://kling.kuaishou.com, 2026.

- [16] Fuchen Long, Zhaofan Qiu, Ting Yao, and Tao Mei. Videostudio: Generating consistent-content and multi-scene videos. In European Conference on Computer Vision, pages 468–485. Springer, 2024.
- [17] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.
- [18] Yihao Meng, Hao Ouyang, Yue Yu, Qiuyu Wang, Wen Wang, Ka Leong Cheng, Hanlin Wang, Yixuan Li, Cheng Chen, Yanhong Zeng, et al. Holocine: Holistic generation of cinematic multi-shot long video narratives. arXiv preprint arXiv:2510.20822, 2025.
- [19] OpenAI. Sora 2 video model. https://openai.com/research/sora-2, 2026.
- [20] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.
- [21] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205, 2023.
- [22] Tianhao Qi, Jianlong Yuan, Wanquan Feng, Shancheng Fang, Jiawei Liu, SiYu Zhou, Qian He, Hongtao Xie, and Yongdong Zhang. Maskˆ 2dit: Dual mask-based diffusion transformer for multi-scene long video generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 18837–18846, 2025.
- [23] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.
- [24] Team Seedance, De Chen, Liyang Chen, Xin Chen, Ying Chen, Zhuo Chen, Zhuowei Chen, Feng Cheng, Tianheng Cheng, Yufeng Cheng, et al. Seedance 2.0: Advancing video generation for world complexity. arXiv preprint arXiv:2604.14148, 2026.
- [25] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, et al. Make-a-video: Text-to-video generation without text-video data. arXiv preprint arXiv:2209.14792, 2022.
- [26] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.
- [27] Yi Wang, Kunchang Li, Yizhuo Li, Yinan He, Bingkun Huang, Zhiyu Zhao, Hongjie Zhang, Jilan Xu, Yi Liu, Zun Wang, et al. Internvideo: General video foundation models via generative and discriminative learning. arXiv preprint arXiv:2212.03191, 2022.
- [28] Xiaoxue Wu, Bingjie Gao, Yu Qiao, Yaohui Wang, and Xinyuan Chen. Cinetrans: Learning to generate videos with cinematic transitions via masked diffusion models. arXiv preprint arXiv:2508.11484, 2025.
- [29] Junfei Xiao, Ceyuan Yang, Lvmin Zhang, Shengqu Cai, Yang Zhao, Yuwei Guo, Gordon Wetzstein, Maneesh Agrawala, Alan Yuille, and Lu Jiang. Captain cinema: Towards short movie generation. In The Fourteenth International Conference on Learning Representations, 2025.
- [30] Zhifei Xie, Daniel Tang, Dingwei Tan, Jacques Klein, Tegawend F Bissyand, and Saad Ezzini. Dreamfactory: Pioneering multi-scene long video generation with a multi-agent framework. arXiv preprint arXiv:2408.11788, 2024.
- [31] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024.
- [32] Jiwen Yu, Jianhong Bai, Yiran Qin, Quande Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Xihui Liu. Context as memory: Scene-consistent interactive long video generation with memory retrieval. In Proceedings of the SIGGRAPH Asia 2025 Conference Papers, pages 1–11, 2025.

- [33] Kaiwen Zhang, Liming Jiang, Angtian Wang, Jacob Zhiyuan Fang, Tiancheng Zhi, Qing Yan, Hao Kang, Xin Lu, and Xingang Pan. Storymem: Multi-shot long video storytelling with memory. arXiv preprint arXiv:2512.19539, 2025.
- [34] Yifu Zhang, Peize Sun, Yi Jiang, Dongdong Yu, Fucheng Weng, Zehuan Yuan, Ping Luo, Wenyu Liu, and Xinggang Wang. Bytetrack: Multi-object tracking by associating every detection box. In European conference on computer vision, pages 1–21. Springer, 2022.
- [35] Yuang Zhang, Junqi Cheng, Haoyu Zhao, Jiaxi Gu, Fangyuan Zou, Zenghui Lu, and Peng Shu. Shouldershot: Generating over-the-shoulder dialogue videos. arXiv preprint arXiv:2508.07597, 2025.
- [36] Canyu Zhao, Mingyu Liu, Wen Wang, Weihua Chen, Fan Wang, Hao Chen, Bo Zhang, and Chunhua Shen. Moviedreamer: Hierarchical generation for coherent long visual sequence. arXiv preprint arXiv:2407.16655, 2024.
- [37] Wenliang Zhao, Lujia Bai, Yongming Rao, Jie Zhou, and Jiwen Lu. Unipc: A unified predictorcorrector framework for fast sampling of diffusion models. Advances in Neural Information Processing Systems, 36:49842–49869, 2023.
- [38] Mingzhe Zheng, Yongqi Xu, Haojian Huang, Xuran Ma, Yexin Liu, Wenjie Shu, Yatian Pang, Feilong Tang, Qifeng Chen, Harry Yang, et al. Videogen-of-thought: A collaborative framework for multi-shot video generation. arXiv preprint arXiv:2412.02259, 3(6), 2024.
- [39] Yupeng Zhou, Daquan Zhou, Ming-Ming Cheng, Jiashi Feng, and Qibin Hou. Storydiffusion: Consistent self-attention for long-range image and video generation. Advances in Neural Information Processing Systems, 37:110315–110340, 2024.

- A Appendix
- B Training Details
- C Additional Data Details

We provide additional statistics of the curated training data in this section. The dataset contains 2,033 video sequences and 20,227 clips in total. Each sequence contains 9.95 clips on average, with a minimum of 6 clips and a maximum of 16 clips. The average sequence duration is 45.9 seconds, the median duration is 42.9 seconds, and the duration ranges from 30 to 65 seconds.

For data annotation, we use Qwen3-VL-8B for caption generation, PaddleOCR for OCR detection, and ByteTrack for identifying reconstruction frames. Qwen3-VL-8B is called 26,252 times in total across three captioning tasks: sequence captioning, clip captioning, and reconstruction captioning. The detailed caption statistics are shown in Table 3.

Table 3: Statistics of the curated dataset and generated captions. Caption length is measured by character count.

Type Count Min Max Mean Median Sequence caption 2,030 80 1,485 543 513 Clip caption 20,189 152 999 459 450 Reconstruction caption 3,992 33 641 130 117

- D Evaluation Metrics

We provide detailed definitions of the evaluation metrics used in our benchmark. Our evaluation covers semantic consistency, background consistency, subject consistency, aesthetic quality, and cross-shot/scene consistency. Unless otherwise specified, all feature vectors are L2-normalized before cosine similarity is computed.

#### D.1 Semantic Consistency

Semantic consistency measures whether the generated video is aligned with the textual prompts. We use ViCLIP [27] with the OpenGVLab/ViCLIP-B-16-hf checkpoint as the video-text encoder.

Global semantic consistency. For each generated scene, we uniformly sample frames from the whole scene, with 24 frames used by default. The sampled frames are divided into clips of 8 frames. Each clip is fed into ViCLIP to extract a video feature, and the features of all clips are averaged and L2-normalized to obtain the scene-level video representation. We concatenate all video prompts within the scene into a single text description and encode it with ViCLIP to obtain the text feature. The global semantic consistency score is computed as the cosine similarity between the scene-level video feature and the text feature.

Shot-level semantic consistency. For shot-level evaluation, each shot is processed independently. We uniformly sample frames from each shot and divide them into 8-frame clips. ViCLIP is then used to extract clip-level video features, which are averaged to form the shot-level video representation. The corresponding prompt of the shot is encoded as the text feature. We compute the cosine similarity between the shot-level video and text features, and report the average score over all shots.

#### D.2 Background Consistency

Background consistency measures the temporal stability of scene appearance and background layout. We use CLIP ViT-B/32 with the openai/clip-vit-base-patch32 checkpoint as the image encoder.

Given a generated video or scene, frames are sampled with a fixed interval denoted as CLIP_SAMPLE_INTERVAL. Each sampled frame is encoded by CLIP into an image feature, fol-

lowed by L2 normalization. We compute the cosine similarity between adjacent sampled frames and average the similarities as the background consistency score.

We report this metric at two levels. The scene-level background consistency, denoted as intra-scene background consistency, is computed within each scene. The video-level background consistency, denoted as intra-video background consistency, is computed over all frames concatenated across the whole story.

#### D.3 Subject Consistency

Subject consistency evaluates whether recurring subjects maintain stable visual identity over time. We use DINOv2 ViT-B/14 [20] with the facebookresearch/dinov2 checkpoint as the visual feature extractor.

Frames are sampled with a fixed interval denoted as DINO_SAMPLE_INTERVAL. Each sampled frame is encoded by DINOv2, and the extracted feature is L2-normalized. We compute two complementary similarity terms. The first is the average cosine similarity between consecutive sampled frames, which measures short-term temporal smoothness. The second is the average cosine similarity between each sampled frame and the first sampled frame, which measures long-range identity preservation. The final subject consistency score is defined as:

- 1

- 2

(Sconsecutive + Sfirst-frame), (9)

Ssubject =

where Sconsecutive denotes the mean adjacent-frame similarity and Sfirst-frame denotes the mean similarity to the first frame.

Similar to background consistency, this metric can be computed at the scene level and video level, corresponding to intra-scene subject consistency and intra-video subject consistency, respectively.

#### D.4 Aesthetic Quality

Aesthetic quality evaluates the visual appeal of generated frames. We use OpenAI CLIP ViT-L/14 as the image encoder, followed by the LAION aesthetic MLP predictor. For each frame, the CLIP feature is extracted, L2-normalized, and then fed into the aesthetic MLP to predict an aesthetic score.

The original aesthetic score lies in the range [1,10]. We normalize it to [0,1] as follows:

Sraw − 1 9

. (10)

Saesthetic =

The final aesthetic score is obtained by averaging the normalized frame-level scores over all generated frames. We also record the standard deviation to reflect the stability of aesthetic quality across frames.

#### D.5 Inter-Shot Consistency

Inter-shot consistency measures whether visual and identity information is preserved across adjacent shots. We consider three variants.

ViCLIP-based inter-shot consistency. For each shot in a scene, we uniformly sample INTER_SHOT_NUM_FRAMES frames and extract a shot-level video feature using ViCLIP. We then compute cosine similarities between adjacent shot pairs and report the average similarity as the inter-shot consistency score.

DINOv2-based inter-shot subject consistency. We also compute an identity-focused inter-shot metric using DINOv2. For each shot, sampled frame features are extracted by DINOv2 and meanpooled into a single shot-level representation. The cosine similarities between adjacent shot representations are then averaged.

Character-grouped inter-shot consistency. To better evaluate recurring subject consistency, we further compute a character-grouped inter-shot score. Based on the [Person X] tags in the prompts, shots are grouped by the characters they contain. For each character group, we compute pairwise ViCLIP cosine similarities among all shots involving the same character. The final score is obtained

by averaging the pairwise similarities within each character group and then averaging across all character groups.

#### D.6 Intra-Shot Consistency

Intra-shot consistency evaluates temporal coherence within the same scene or shot. In our benchmark, it corresponds to the scene-level versions of subject consistency and background consistency, namely intra-scene subject consistency and intra-scene background consistency. These metrics measure whether subjects, backgrounds, and visual layouts remain stable across frames within a local temporal segment.

#### D.7 Inter-Scene Consistency

Inter-scene consistency evaluates whether subject and visual identity can be preserved across scene transitions. We use ViCLIP to extract one feature vector for each scene by encoding the frames of the entire scene. We then compute cosine similarities between adjacent scene pairs and report their average as the inter-scene consistency score. A higher score indicates better long-range consistency across scene boundaries.

