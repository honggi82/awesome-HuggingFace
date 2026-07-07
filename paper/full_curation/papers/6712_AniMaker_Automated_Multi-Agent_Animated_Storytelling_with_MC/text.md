# arXiv:2506.10540v2[cs.MA]2Oct2025

## AniMaker: Multi-Agent Animated Storytelling with MCTS-Driven Clip Generation

HAOYUAN SHI, Harbin Institute of Technology, Shenzhen, China YUNXIN LI, Harbin Institute of Technology, Shenzhen, China XINYU CHEN, Harbin Institute of Technology, Shenzhen, China LONGYUE WANG, Alibaba International Digital Commerce, Hangzhou, China BAOTIAN HU∗, Harbin Institute of Technology, Shenzhen, China MIN ZHANG, Harbin Institute of Technology, Shenzhen, China

[Figure 1]

Fig. 1. A visual example of AniMaker generating compelling storytelling animation from narrative text. Our framework maintains consistent character appearance across scenes while delivering high-quality action representation for complex sequences. AniMaker seamlessly integrates adaptive shot scheduling with smooth transitions, ensuring narrative coherence throughout the animation.

∗Corresponding author: Baotian Hu.

Authors’ Contact Information: Haoyuan Shi, Harbin Institute of Technology, Shenzhen, China, g1016015592@gmail.com; Yunxin Li, Harbin Institute of Technology, Shenzhen, China, liyunxin987@163.com; Xinyu Chen, Harbin Institute of Technology, Shenzhen, China, chenxinyuhitsz@163.com; Longyue Wang, Alibaba International Digital Commerce, Hangzhou, China, vincentwang0229@gmail.com; Baotian Hu, Harbin Institute of Technology, Shenzhen, China, hubaotian@hit.edu.cn; Min Zhang, Harbin Institute of Technology, Shenzhen, China, zhangmin2021@hit.edu.cn.

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.

SA Conference Papers ’25, Hong Kong, Hong Kong © 2025 Copyright held by the owner/author(s). Publication rights licensed to ACM. ACM ISBN 979-8-4007-2137-3/2025/12 https://doi.org/10.1145/3757377.3764009

Despite rapid advancements in video generation models, generating coherent, long-form storytelling videos that span multiple scenes and characters remains challenging. Current methods often rigidly convert pre-generated keyframes into fixed-length clips, resulting in disjointed narratives and pacing issues. Furthermore, the inherent instability of video generation models means that even a single low-quality clip can significantly degrade the entire output animation’s logical coherence and visual continuity. To overcome these obstacles, we introduce AniMaker, a multi-agent framework enabling efficient multi-candidate clip generation and storytelling-aware clip selection, thus creating globally consistent and story-coherent animation solely from text input. The framework is structured around specialized agents, including the Director Agent for storyboard generation, the Photography Agent for video clip generation, the Reviewer Agent for evaluation, and the Post-Production Agent for editing and voiceover, collectively realizing multicharacter, multi-scene animation. Central to AniMaker’s approach are two key technical components: MCTS-Gen in Photography Agent, an efficient Monte Carlo Tree Search (MCTS)-inspired strategy that intelligently navigates the candidate space to generate high-potential clips while optimizing

resource usage; and AniEval in Reviewer Agent, the first framework specifically designed for multi-shot animation evaluation, which assesses critical aspects such as story-level consistency, action completion, and animationspecific features by considering each clip in the context of its preceding and succeeding clips. Experiments demonstrate that AniMaker achieves superior quality as measured by popular metrics including VBench and our proposed AniEval framework, while significantly improving the efficiency of multi-candidate generation, pushing AI-generated storytelling animation closer to production standards. Code and data for this paper are at https://animaker-dev.github.io/

CCS Concepts: • Applied computing → Arts and humanities; • Computing methodologies → Computer vision; Animation.

Additional Key Words and Phrases: storytelling animation, multi agents, MCTS, storyboard generation

#### ACM Reference Format:

Haoyuan Shi, Yunxin Li, Xinyu Chen, Longyue Wang, Baotian Hu, and Min Zhang. 2025. AniMaker: Multi-Agent Animated Storytelling with MCTSDriven Clip Generation. In SIGGRAPH Asia 2025 Conference Papers (SA Conference Papers ’25), December 15–18, 2025, Hong Kong, Hong Kong. ACM, New York, NY, USA, 17 pages. https://doi.org/10.1145/3757377.3764009

- 1 Introduction

The emergence of large-scale video generation models [Kong et al. 2024; Liu et al. 2024c; Yang et al. 2024] has ignited significant interest in storytelling videos. In response, a new wave of methods [Dalal et al. 2025; Guo et al. 2025; Kim et al. 2024; Wu et al. 2024; Xu et al. 2025b] including MINT Video and TTT-Video has emerged, aiming to generate the entire video in a single pass. However, these methods still face significant challenges when generating long videos spanning multiple scenes and characters, particularly in maintaining visual continuity, narrative coherence, and non-repetitive content. In contrast, recent advancements in single-clip video generation, exemplified by models like Vidu[Bao et al. 2024], Wan [Wan et al. 2025], and Pika [Pika Labs 2025], have pushed generated video clips closer to cinematic quality. Due to shorter clip duration, these models utilize textual and visual prompts directly and efficiently, improving motion quality, semantic fidelity, and consistency with reference images. Consequently, utilizing these models to generate a storytelling video with multiple clips appears highly promising.

Existing storytelling video frameworks [He et al. 2024; Li et al.

- 2024; Lin et al. 2023; Wu et al. 2025; Xie et al. 2024; Xu et al. 2025a], which implement the composition of multiple video clips, generally follow a fixed pipeline: script → keyframes → video clips → final video composition. While this modular format can effectively generate multi-shot long videos, it also introduces several limitations: Firstly, existing methods typically map keyframes to fixed-length clips. It forms a rigid and fragmented video construction, leading to disjointed transitions and unnatural pacing, severely hindering the expressive continuity crucial for complex or extended actions [Jiang et al. 2021]. Secondly, due to the inherent instability of video generation models, a single flawed clip can noticeably degrade the overall video quality.

To mitigate fragmentation, an intuitive approach is generating continuous clips conditioned on prior frames, but this compounds error propagation and quality degradation. Drawing parallels with professional filmmaking, we identify that existing methods overlook Best-of-N Sampling—generating multiple clip candidates

and selecting the best ones. However, implementing this faces two challenges: prohibitive computational costs and inadequate automated evaluation mechanisms. Generating and evaluating multiple candidate clips per shot is computationally intensive, often relying on expensive commercial APIs or prolonged GPU inference. Current evaluation metrics like VBench [Xing et al. 2024] only assess individual clips and their internal consistency, neglecting critical elements such as cross-clip coherence, sequential motion quality, and animation-specific qualities in storytelling animation.

To address these challenges, we introduce AniMaker, a multiagent framework with MCTS-driven clip generation. This framework mirrors professional production [Xu et al. 2025b], including the Director Agent for storyboards construction, Photography Agent for video clip generation, Reviewer Agent for evaluating the quality of video clip candidates, and Post-Production Agent for editing and assembling the entire sequence of videos. These agents collaborate to enable automated multi-character, multi-scene storytelling without manual pre- or post-processing. The core Photography and Reviewer agents interact under a MCTS-Gen scheme during clip generation. Specifically, based on Monte Carlo Tree Search (MCTS), MCTS-Gen offers an efficient strategy for navigating the vast candidate space of video generation. It strikes a balance between broad exploration and computational efficiency by intelligently allocating more generation opportunities to promising clips while encouraging the exploration of unexplored regions. For Reviewer Agent, we introduce a comprehensive evaluation framework AniEval, specifically designed for multi-shot storytelling [Cailin Zhuang 2025] animation. AniEval advances beyond metrics like VBench by implementing retrospective evaluation with cross-clip contextual references. It evaluates critical dimensions—story consistency, action completion, and animation-specific attributes—by analyzing each clip in the context of its preceding and succeeding clips. This contextual assessment serves as the quality evaluation mechanism for video-clip nodes within the MCTS-Gen framework.

We conduct extensive experiments on the dataset constructed from TinyStories [Eldan and Li 2023], featuring complex interaction with multiple characters across diverse backgrounds. Experimental results demonstrate AniMaker’s superior performance across VBench, AniEval, and human evaluation, with significantly improved multi-candidate generation efficiency. This validates AniMaker’s effectiveness in bringing AI-generated storytelling videos closer to production-grade quality (as shown in Figure 1). With our current work successfully targeting simple storytelling, we believe further development is essential to produce more complex and sophisticated animation styles.

In summary, our main contributions are:

- • We propose AniMaker, a fully automated multi-agent framework for generating coherent, multi-character, multi-scene animationfromtextualstories. Theframework features MCTSGen, a novel search-based generation strategy that efficiently balances exploration and resource usage.
- • We developAniEval,the firstcomprehensive evaluation framework specifically designed for multi-shot storytelling animation. It provides context-aware assessment by analyzing clips in relation to their surrounding content.

• We experimentally demonstrate Animaker’s superior performance across all evaluation frameworks—achieving best scores across all keyframe evaluation metrics, ranking 1st in VBench, 14.6% higher scores in our AniEval, and better human ratings (3.22 versus 2.07).

- 2 Related Work

Storyboard Visualization. Storyboard visualization bridges script to video by generating coherent image sequences, demanding flexible, reusable character-background modules for multiple scenes. While adapter-based techniques like IP-Adapter [Ye et al. 2023], Mixof-Show [Gu et al. 2023], T2I-Adapter [Mou et al. 2024], ROICtrl [Gu

- et al. 2024], and StoryAdapter [Mao et al. 2024] realize character consistency, they often overlook background continuity. Consistencyaware methods try to address this challenge. StoryGen [Liu et al. 2024b] iteratively generates images using previous visual-language context, while StoryDiffusion [Zhou et al. 2024] employs a trainingfree Consistent Self-Attention module to improve feature alignment. Despite these advancements, the precise preservation of visual details, especially with multiple characters or changing backgrounds, remains a challenge.

Video Generation. Driven by the success of diffusion models, VDM [Ho et al. 2022] pioneers their application to video generation, leading to improved models like Stable Video Diffusion [Blattmann et al.

- 2023], ModelScope [Wang et al. 2023b], VideoCrafter1 [Chen et al.

- 2023], and VideoCrafter2 [Chen et al. 2024]. While these models excel at animating single text or image, they struggle with maintaining consistency across multiple video clips. The emergence of Sora [Liu et al. 2024c] has spurred a demand for coherent, consistent, and controllable storytelling videos. In response, several works[Dalal et al. 2025; Guo et al. 2025; Kim et al. 2024; Wu et al. 2024] have emerged to enhance storytelling video generation by improving temporal modeling and contextual understanding. Despite these advancements, challenges persist as single-pass long video generation often suffers from inconsistencies and repetitive content over time. Considering this, models like HunyuanVideo [Kong et al. 2024], Wan [Wan et al. 2025], and CogVideoX [Yang et al. 2024] still prioritize the effective utilization of text prompts and reference images, resulting in more faithful generation and improved motion depiction. Industry solutions [Bao et al. 2024; Pika Labs 2025; Pixverse AI

2025; Runway AI, Inc. 2025] extend these advancements, enabling features like start/end frame-specific control.

Storytelling Video Agent. Storytelling video generation typically follows a pipeline: script → keyframes → video clips → final video composition. Large multimodal models (LMMs) often act as highlevel planners to coordinate each stage (e.g., [Google Cloud 2025; Hurst et al. 2024; Li et al. 2025]). Frameworks such as VideoDirectorGPT [Lin et al. 2023], DreamStory [He et al. 2024], AnimDirector [Li et al. 2024], MM-StoryAgent [Xu et al. 2025a], MovieAgent [Wu

- et al. 2025], and DreamFactory [Xie et al. 2024] adhere to this conventional pipeline. Notably, certain tasks like scriptwriting, character design, voiceovers, and sound effects are typically handled manually during pre- or post-processing.

3 Methodology

Our proposed framework, AniMaker, automates the creation of foundational storytelling animation from text. It uses a multi-agent system that mirrors professional animation pipelines, incorporating novel components for efficient candidate generation and comprehensive evaluation. The overall architecture is detailed in Figure 2.

- 3.1 Task Formulation

Automated generation of storytelling animation Vfinal, from text input Tprompt, can be represented as:

F : Tprompt → Vfinal

This transformation is realized by mapping through several crucial intermediate representations:

- • Script (Pscript): Derived from Tprompt, the script

Pscript = ((shot𝑘)𝑘N=clips1 , K𝑐𝑢𝑡) defines:

- – An orderedsequenceof 𝑁clips shots,whereeachshot𝑘 = (d𝑘, C𝑘, B𝑘) specifies its textual description d𝑘, involved characters C𝑘, and background B𝑘.
- – A set of indices K𝑐𝑢𝑡 ⊆ {1, . . ., Nclips}, marking shot transitions, indicating where KeyFrame guidance is required for the new shot.

- • Storyboard(Sboard): Storyboard is established from Pscript, containing:

- – Character Bank: Bchar = {(𝒄, I𝑐char) | 𝒄 ∈ Ctotal}, with I𝑐char as the reference image for character 𝒄, where Ctotal represents all available characters in the story.
- – Background Bank: Bbg = {(𝒃, I𝑏bg) | 𝒃 ∈ Btotal}, with I𝑏bg as the reference image for background 𝒃, where Btotal represents all available backgrounds in the story.
- – KeyFrames: for each shot𝑘 where 𝑘 ∈ K𝑐𝑢𝑡, a KeyFrame F𝑘key is defined:

F𝑘key = Visualize({I𝑐char | 𝒄 ∈ C𝑘}, IbgB

𝑘

, d𝑘)

where Visualize(·) generates a keyframe by integrating the specified characters, background, and text description into a multimodal prompt.

- • Video Clip Sequence (v): v = (v1, . . ., v𝑁clips) is generated, cor-

responding to (shot𝑘)𝑘𝑁=clips1 . Let 𝐺𝐾 and 𝐺𝐶 be abstract generative processes of each clip v𝑘:

- – If 𝑘 ∈ K𝑐𝑢𝑡 (after shot transition): v𝑘 ∼ 𝐺𝐾 (F𝑘key, d𝑘).
- – If 𝑘 ∉ K𝑐𝑢𝑡: v𝑘 ∼ 𝐺𝐶(last_frame(v𝑘−1), d𝑘).

- • Final Video (Vfinal): The assembled clip sequence Assemble(v) thenundergoespost-processing, utilizing information from Pscript to produce the polished Vfinal. This may include additions like voiceovers and subtitles.

Vfinal = PostProcess(Assemble(v), Pscript)

- 3.2 Pipeline Overview

AniMaker transforms textual input (Tprompt) into compelling storytelling animation (Vfinal) through four specialized agents working collaboratively. Central to this pipeline is the use of LLM prompt

[Figure 2]

Fig. 2. The overall architecture of our AniMaker framework. Given a story input, Director Agent creates detailed scripts and storyboards with reference images. Photography Agent generates candidate video clips using MCTS-Gen, which optimizes exploration-exploitation balance. Reviewer Agent evaluates clips with our AniEval assessment system. Post-production Agent assembles selected clips, adds voiceovers, and synchronizes audio with subtitles. This multi-agent system enables fully automated, high-quality animated storytelling.

engineering by its four specialized agents, and we refer to the appendix for related details. The Director Agent (Section 3.3.1) creates a detailed storyboard (Sboard), the Photography Agent (Section 3.3.2) generates candidate video clips (v𝑘) using MCTS-Gen (efficiently exploring generation space conserving computational resources), the Reviewer Agent (Section 3.3.3) evaluates clips with AniEval (our context-aware evaluation framework), and the Post-production Agent (Section 3.3.4) assembles clips with voiceovers and subtitles.

3.3 Multi-Agent Framework: AniMaker

- 3.3.1 Director Agent. The Director Agent orchestrates storyboard generation through a two-stage process. First, Gemini 2.0 Flash [Google Cloud 2025] creates a raw script with shot descriptions (shot𝑘), followed by automated validation for consistency and narrative flow. Second, in the Storyboard (Sboard) realization phase, a Visual Bank is built: Character Bank is built (Bchar) with Hunyuan3D [Zhao et al. 2025] and Background Bank (Bbg) is built with FLUX1-dev [Flux AI 2025]. Then, GPT-4o [Hurst et al. 2024] gen-

erates keyframes (F𝑘key) combining validated shot descriptions (d𝑘) with Visual Bank imagery. This ensures visual consistency, with the

resulting Sboard serving as the animation production blueprint in the following phases.

- 3.3.2 Photography Agent. Converting storyboards to clips in multi-shot AI video generation presents challenges including distorted appearances, inconsistent motion, and object inconsistencies. Drawing from the "No Good" (NG) process in filmmaking, where numerous takes are recorded to achieve the perfect shot, we recognize

the need to produce multiple candidate clips to identify the optimal one. For optimal selection, each clip must not only possess high individual quality but also ensure consistency and coherence with both preceding and succeeding clips. However, naively generating 𝑘 candidates per clip creates a combinatorial explosion (e.g., 𝑘2 for two-clip sequences). Notably, poor-quality current clips allow for pruning the search space, as further exploration down such paths is unlikely to be satisfactory.

Thus we propose MCTS-Gen, a Monte Carlo Tree Search (MCTS)inspired method for multi-clip video generation (Figure 3). MCTS naturally fits this task: multi-clip sequences correspond to tree paths, with each clip as a node. Crucially, clip evaluation considers both intrinsic quality and inter-clip consistency, aligning with MCTS’s backpropagation where child’s scores update the parent’s evaluation results. MCTS-Gen iteratively constructs a chosen path of selected video clips, extending this path by one clip per iteration. This process is controlled by the following parameters: w1 (initial candidate count per node), w2 (UCT-guided expansion times per iteration), and 𝜶 (exploration-exploitation balancing factor, default 1). Specifically, we use Wan 2.1 [Wan et al. 2025] for video clip generation, and the algorithm proceeds as follows:

1. Expansion: w1 (3 in Figure 3) initial child clips are generated from the chosen path’s terminal node (node 1 in Figure 3). These clips (node 3, 4, 5 in Figure 3) are then scored using AniEval (detailed in Section 3.3.3) and ranked.

- 2. Simulation: Further w2 (3 in Figure 3) expansions are gener-

ated guided by the UCT score:

𝑼𝑪𝑻(𝒏𝒐𝒅𝒆𝒋) =

2.0 𝒓𝒂𝒏𝒌(𝒏𝒐𝒅𝒆𝒋) + 1 +𝜶 ·

√︄ 2.0

𝒄𝒉𝒊𝒍𝒅_𝒄𝒐𝒖𝒏𝒕(𝒏𝒐𝒅𝒆𝒋) + 1

where 𝒓𝒂𝒏𝒌 is from the initial AniEval scoring, 𝒄𝒉𝒊𝒍𝒅_𝒄𝒐𝒖𝒏𝒕 is dynamically updated, and 𝜶 balances exploitation and exploration. The node with the highest UCT score generates a new child clip.

- 3. Backpropagation: AniEval scores of child clips (nodes 6, 7, 8

in Figure 3) propagate upwards. A parent node’s score is updated by adding the average score of children to its own.

- 4. Selection: The node with the highest AniEval score (node 3

in Figure 3) is added to the chosen path, then generating new child clips until reaching a total of w1 children, allowing the iterative generation process to continue.

This MCTS-Gen process systematically balances the exploration of diverse video generation space with the exploitation of promising paths, aiming to efficiently construct a high-quality, coherent video sequence while managing computational resources.

- 3.3.3 Reviewer Agent. Existing objective metrics, such as CLIP Score and Inception Score, may identify superior video generation models, but often struggle to differentiate among candidates generated by the same model. Similarly, the widely adopted VBench has significant limitations. For instance, some of its metrics, like "dynamic degree", are overly simplistic, merely measuring pixel changes rather than accurately reflecting character action. More critically, VBench’s "Consistency" related metrics, based on singleclip segmentation, prove unsuitable for multi-shot animation, which inherently involves frequent character and scene changes.

To address evaluation challenges, we introduce AniEval, a comprehensive evaluation framework built on EvalCrafter [Liu et al.

- 2024a]. AniEval refines EvalCrafter’s metrics for fully automated evaluation, e.g., by automating action assessment through comparing prompted character actions with those identified in video clips.

Furthermore, responding to the specific demands of evaluating multi-shot animation characterized by multiple characters and diverse scenes, AniEval introduces several additional metrics: DreamSim [Fu et al. 2023] assesses overall frame consistency; CountScore [Cheng et al. 2024] aims at the issue of objects appearing or disappearing between shots; Face Consistency evaluates animated character facial consistency by training an InceptionNext [Yu et al.

- 2024] model on the Anime Face Dataset [splcher 2019], overcoming the limitations of conventional face recognition methods like MTCNN [Ku and Dong 2020] in anime face detection and tracking.

In conclusion, AniEval comprises 4 primary domains with 14 fine-grained metrics for comprehensive assessment (Table 1). Additionally, AniEval supports contextual scoring by evaluating clips based on preceding and succeeding content, providing robust evaluation for multi-shot animation generation. Further implementation details of AniEval can be found in the appendix.

- 3.3.4 Post-production Agent. The Post-production Agent transforms video clip sequences into a polished animation film through three stages. Firstly, it leverages Gemini 2.0 Flash to generate a detailed voiceover script specifying narration, dialogue, emotional

Table 1. Introduction of AniEval Metrics.

Domain Metric Brief Description Overall Video Quality VQA_A Aesthetic video quality

VQA_T Technical video quality MusIQ Frame quality score

Text-Video Alignment Text-Video Consistency Measured by CLIP Text-Story Consistency Measured by BLIP-BLEU Detection-Score Object generation accuracy Count-Score Key object count accuracy

Video Consistency DreamSim Perceptual frame-frame similarity Face Consistency Animated character facial consistency Warping Error Temporal inconsistency via pixel differences Semantic Consistency Temporal semantic coherence via CLIP

Motion Quality Action Recognition Actions-prompt consistency Action Strength Motion intensity via Flow-Score Motion AC-Score Motion amplitude-prompt

consistency

Table 2. Keyframe evaluation on Contextual Coherence (Coherence), ImageImage Similarity (I-I Sim), and Text-Image Similarity (T-I Sim).

### Model Coherence↑ I-I Sim↑ T-I Sim↑

StoryGen 0.54 0.77 0.22 StoryDiffusion 0.70 0.80 0.25 StoryAdapter 0.78 0.83 0.25 MovieAgent 0.59 0.65 0.23 MMStoryAgent 0.78 0.83 0.26 VideoGen-of-Thought 0.71 0.77 0.23 AniMaker(Ours) 0.81 0.83 0.31

tones, and desired voice timbres. The agent then selects appropriate voice profiles based on character attributes (age, gender) and assesses text length for audio-visual synchronization. Secondly, the script is processed through CosyVoice2 [Du et al. 2024] to generate audio tracks, which undergo verification regarding duration consistency and content accuracy. Finally, the agent employs the MoviePy library for film assembly, integrating validated subtitles and performing comprehensive editing to ensure precise synchronization between visuals, voiceovers, and subtitles.

4 Experiments 4.1 Settings

- 4.1.1 Datasets. To evaluate our AniMaker, we sample 10 narratives from TinyStories [Eldan and Li 2023]. These narratives feature complex multi-character interactions across diverse backgrounds, providing an ideal testbed for multi-shot animation generation.
- 4.1.2 Baseline. We evaluate several state-of-the-art storytelling models: StoryGen, StoryDiffusion, and StoryAdapter (visual narrative specialists),alongsideMovieAgent, MMStoryAgent, and VideoGenof-Thought (video generators). The latter group utilizes their built-in video modules, while StoryDiffusion and StoryAdapter are paired with external image-to-video models (CogVideoX and Wan 2.1).

[Figure 3]

Fig. 3. Illustration of our MCTS-Gen strategy for efficient Best-of-N Sampling.

- 4.1.3 Evaluation Metrics. For keyframe generation, we evaluate text-to-image alignment and cross-image consistency. Metrics include Text-to-Image CLIP (Coherence) [Radford et al. 2021a], Imageto-Image Similarity (I-I Sim) [Gal et al. 2022], and Text-Image Similarity (T-I Sim) [Hessel et al. 2021]. Video generation is assessed using VBench and our AniEval for comprehensive evaluation.

- 4.2 Qualitative Analysis

Based on the samples depicted in Figures 5 and 6, we conduct a qualitative analysis of AniMaker’s output, with a focus on visual fidelity and narrative coherence in multi-character, multi-scene animated storytelling.

Enhanced Consistency. AniMaker demonstrates exceptional visual consistency across scenes. While baseline methods often face challenges in maintaining character and background consistency during scene transitions, AniMaker successfully preserves visual characteristics, even when switching back and forth between distinct scenes. This can be attributed to the Director Agent’s storyboard creation process, which provides a reliable set of reference images for both characters and backgrounds. Additionally, AniEval plays a crucial role in ensuring narrative coherence by incorporating a meticulous clip selection process, which evaluates consistency related metrics across clips.

Improved Action Representation. AniMaker excels at depicting complex and extended character actions. Baseline methods often

produce incomplete movements, particularly for multi-step action sequences (e.g., squatting, picking up an object, standing up, and walking away in Figure 5). Our MCTS-Gen strategy enables the Photography Agent to explore and select clip sequences that concatenate into coherent, complete long actions.

Seamless Transitions Between Clips. AniMaker achieves smoother video transitions through an effective generation and selection mechanism. Central to this process is the Reviewer Agent, which leverages our AniEval framework to maximize visual continuity. By integrating cross-clip consistency metrics such as DreamSim and applying contextual scoring to adjacent clips, the agent effectively minimizes jarring visual disruptions between shots.

In summary, AniMaker outperforms existing methods through superior visual consistency across scenes, effective depiction of complex action sequences, seamless inter-clip transitions, and robust handling of multi-character, multi-scene narratives.

4.3 Quantitative Comparisons

Keyframe Generation Analysis. Keyframes are the core components of a storyboard. Table 2 demonstrates that AniMaker outperforms all other competing methods across metrics. Notably, AniMaker achieves a Text-to-Image Similarity (T-I Sim) score of 0.31, representing a 19.2% improvement over the best-performing baseline method. This advantage stems from our multimodal storyboard

Table 3. VBench evaluation results, presenting scores for Image Quality (I.Q.), Semantic Consistency (S.C.), Background Consistency (B.C.), Animation Quality (A.Q.), Motion Smoothness (M.S.), Dynamic Degree (D.D.), and Average Rank (Rk. Avg. - the average ranking position across all models).

#### Model I.Q.↑ S.C.↑ B.C.↑ A.Q.↑ M.S.↑ D.D.↑ Rk. Avg.↓

StableDiffusion+Cog. 75.52 78.05 85.27 59.61 97.63 33.58 5.83 StableDiffusion+Wan. 76.93 78.54 87.43 69.75 96.67 60.30 4.33 StoryAdapter+Cog. 76.17 72.17 88.03 63.55 98.16 26.71 5.33 StoryAdapter+Wan. 75.96 75.04 88.64 73.38 97.02 84.73 4.17 MovieAgent 72.09 68.61 79.84 55.40 99.01 35.44 6.33 MMStoryAgent 76.41 87.27 90.74 73.84 99.80 0.00 2.67 VoT 63.85 75.11 85.78 74.91 99.25 3.50 4.83 AniMaker(Ours) 76.96 84.27 89.06 69.79 98.50 66.97 2.50

generation approach that incorporates character references, background references, and text prompts for keyframe generation, rather than relying solely on text.

VBenchEvaluationAnalysis. Table3showsthatAniMakerachieves the best average rank (2.50), demonstrating consistent top-tier performance across metrics. While MMStoryAgent excels in Semantic Consistency (S.C.) and Background Consistency (B.C.), its 0.00 score in Dynamic Degree (D.D.) reveals a critical limitation—it produces static image sequences resembling comic strips rather than true animation. This exposes VBench’s limitations: it favors static consistency (where even brief static images can achieve high scores) and focuses on individual clips rather than multi-scene, multi-character animation. These findings highlight the need for a new evaluation framework specifically designed for storytelling animation quality.

AniEval Evaluation Analysis. We introduce AniEval for better evaluation of storytelling Animation. Table 4 presents results from our AniEval framework. AniMaker outperforms all competitors with a total score of 76.72, representing a 14.6% improvement over the second-best method (i.e. VideoGen-of-Thought with 66.93). Particularly noteworthy is AniMaker’s exceptional performance in Video Consistency (V.C.), surpassing the best-performing baseline method by 15.5%. While, AniMaker’s relatively lower performance in Text-Video Alignment (T.V.A.) can be attributed to our agent framework’s creative adaptation of stories into scripts, where additional narrative elements are introduced. Compared to VBench, AniEval’s assessment results align remarkably better with human evaluations (Table 5), demonstrating that AniEval provides a more comprehensive and accurate assessment of multi-shot animation quality than previous metrics.

MCTS-Gen Parameter Analysis. Figure 4 shows how MCTS-Gen parameters w1 (initial candidate count) and w2 (expansion iterations) affect generation quality. Two patterns emerge: higher w1 values improve AniEval scores by providing more initial candidates, while higher w2 values enhance performance by better evaluating each clip’s suitability for continuous generation. Importantly, once certain thresholds are reached, configurations with fewer total generations (e.g., w1 = 3, w2 = 3 with 4.37 generations per node) perform comparably to those requiring more (e.g., w1 = 3, w2 = 5 with 5.76 generations per node). This demonstrates MCTS-Gen’s efficiency—compressing the search space by over 50% compared to

[Figure 4]

Fig. 4. AniEval Score of Different w1 (initial candidate count) and w2 (expansion iterations) Combinations.

exhaustive search strategies (which require 9 generations per node for the same candidates) while maintaining quality.

Table 4. AniEval evaluation results, presenting scores for Overall Video Quality (O.V.Q.), Text-Video Alignment (T.V.A.), Video Consistency (V.C.), Motion Quality (M.Q.), and Total Performance.

#### Model O.V.Q.↑ T.V.A.↑ V.C.↑ M.Q.↑ Total↑

StoryDiffusion+CogVideoX 46.54 86.05 47.14 70.35 56.75 StoryDiffusion+Wan 2.1 47.07 84.99 47.13 71.00 56.55 StoryAdapter+CogVideoX 56.76 87.38 55.89 69.73 63.95 StoryAdapter+Wan 2.1 60.39 86.99 51.41 72.11 62.37 MovieAgent 41.17 68.50 68.68 70.16 61.95 MMStoryAgent 47.93 75.27 63.54 61.39 62.79 VideoGen-of-Thought 66.17 72.95 65.42 66.72 66.93 AniMaker(Ours) 81.87 74.30 79.35 72.66 76.72

- 4.4 Human Rating

Following MovieAgent’s evaluation framework [Wu et al. 2025], we conduct human evaluation with 10 participants on 90 storytelling videos from 9 models across 10 stories. Each video was rated on a 1-5 scale across five dimensions: Visual Appeal, Script Faithfulness, Narrative Coherence, Character Consistency, and Physical Law Adherence. Note that evaluating complete storytelling videos (rather than individual clips) typically yields lower scores due to the increased complexity and length. As shown in Table 5, our model achieves superior performance across all metrics, especially in terms of Character Consistency.

- 4.5 Ablation Studies We conduct two key ablation experiments. First, we ablate MCTS-

Gen by setting 𝑤1 = 1,𝑤2 = 1, essentially generating only one candidate per clip. This change results in a 7.1% reduction in performance (the green triangle in Figure 4) on AniEval, confirming the importance of our MCTS-driven generation strategy. Notably, even this ablated version still outperforms the best-performing baseline method by 6.6%, further demonstrating that our multi-agent framework remains highly competitive even without clip candidate selection. Next, we ablate AniEval by generating five candidates per clip with VBench for selection. This yields a score of 73.18, a 4.6%

Table 5. Human rating results on a 1–5 scale, covering Character Consistency (C.C.), Narrative Coherence (N.C.), Physical-Law Adherence (P.L.), Script Faithfulness (S.F.), and Visual Appeal (V.A.).

#### Model C.C.↑ N.C.↑ P.L.↑ S.F.↑ V.A.↑ Avg.↑

StoryDiffusion+CogVideoX 1.37 1.48 1.37 1.67 1.56 1.49 StoryDiffusion+Wan 2.1 2.00 1.82 1.82 2.00 2.11 1.95 StoryAdapter+CogVideoX 1.64 1.39 1.46 1.68 1.71 1.58 StoryAdapter+Wan 2.1 2.04 1.82 1.89 2.04 2.57 2.07 MovieAgent 1.19 1.26 1.44 1.26 1.48 1.33 MM_StoryAgent 1.62 1.62 1.72 1.83 2.24 1.81 VideoGenoT 1.67 1.74 1.78 1.74 2.26 1.84 AniMaker(Ours) 3.44 3.24 3.04 3.08 3.28 3.22

decrease compared to our raw method. Qualitative assessment of the resulting videos also reveals noticeable degradation in action expressiveness and cross-clip consistency, highlighting the importance of AniEval for storytelling animation.

- 5 Limitations

Despite employing state-of-the-art models for storyboard and video clip generation, a significant gap remains between current model capabilities and the quality required for commercial film production. A primary limitation is the restricted narrative scope and complexity our method can handle. Compared to the intricate cinematic grammar and narrative structures utilized in professional filmmaking, as detailed in [Ronfard 2021], our approach currently falls short.

These shortcomings are rooted in several underlying defects within the video generation models. They are prone to producing animation artifacts, along with issues of incoherence and hallucination that disrupt narrative continuity. Furthermore, the models exhibit a relatively poor adherence to fundamental physical laws, resulting in unrealistic interactions between characters and scene elements, as also highlighted in [Shi et al. 2025]. Luckily, our framework is built on a modular, plug-and-play architecture that allows for seamless model integration. As more advanced generative models emerge, we are committed to continuously updating our framework and sharing these workflow improvements with the community to address these limitations.

- 6 Conclusion

We present AniMaker, a comprehensive multi-agent framework that transforms text input into coherent storytelling animation by emulating professional workflows. Our system introduces two key innovations: MCTS-Gen, which optimizes exploration-exploitation balance during clip generation, and AniEval, the first evaluation framework specifically designed for multi-shot storytelling animation. AniMaker orchestrates specialized agents that seamlessly collaborate across storyboarding, generation, evaluation, and postproduction stages. Our quantitative results validate the effectiveness of this approach, with substantial gains in both technical metrics and perceived quality. These advances mark an important step toward bridging the gap between AI-generated content and professional animation standards, paving the way for more accessible and highquality animated storytelling production.

Acknowledgments

We thank the editor and reviewers for their efforts in improving our paper. This work was supported by grants: Natural Science Foundation of China (No. 62422603), Guangdong Basic and Applied Basic Research and Foundation (No.2024B0101050003) and Shenzhen Science and Technology Progam (No. ZDSYS20230626091203008).

References

Fan Bao, Chendong Xiang, Gang Yue, Guande He, Hongzhou Zhu, Kaiwen Zheng, Min Zhao, Shilong Liu, Yaole Wang, and Jun Zhu. 2024. Vidu: a highly consistent, dynamic and skilled text-to-video generator with diffusion models. arXiv preprint arXiv:2405.04233 (2024).

Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. 2023. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127 (2023).

Wei Cheng Jingwei Wu Yaoqi Hu Jiaqi Liao Hongyuan Wang Xinyao Liao Weiwei Cai Hengyuan Xu Xuanyang Zhang Xianfang Zeng Zhewei Huang Gang Yu Chi Zhang Cailin Zhuang, Ailin Huang. 2025. ViStoryBench: Comprehensive Benchmark Suite for Story Visualization. arXiv preprint arxiv:2505.24862 (2025).

Haoxin Chen, Menghan Xia, Yingqing He, Yong Zhang, Xiaodong Cun, Shaoshu Yang, Jinbo Xing, Yaofang Liu, Qifeng Chen, Xintao Wang, et al. 2023. Videocrafter1: Open diffusion models for high-quality video generation. arXiv preprint arXiv:2310.19512 (2023).

Haoxin Chen, Yong Zhang, Xiaodong Cun, Menghan Xia, Xintao Wang, Chao Weng, and Ying Shan. 2024. Videocrafter2: Overcoming data limitations for high-quality video diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 7310–7320.

Junhao Cheng, Baiqiao Yin, Kaixin Cai, Minbin Huang, Hanhui Li, Yuxin He, Xi Lu, Yue Li, Yifei Li, Yuhao Cheng, et al. 2024. Theatergen: Character management with llm for consistent multi-turn image generation. arXiv preprint arXiv:2404.18919 (2024).

MMAction Contributors. 2020. Openmmlab’s next generation video understanding toolbox and benchmark. (2020).

Karan Dalal, Daniel Koceja, Gashon Hussein, Jiarui Xu, Yue Zhao, Youjin Song, Shihao Han, Ka Chun Cheung, Jan Kautz, Carlos Guestrin, et al. 2025. One-minute video generation with test-time training. arXiv preprint arXiv:2504.05298 (2025).

Zhihao Du, Yuxuan Wang, Qian Chen, Xian Shi, Xiang Lv, Tianyu Zhao, Zhifu Gao, Yexin Yang, Changfeng Gao, Hui Wang, et al. 2024. Cosyvoice 2: Scalable streaming speech synthesis with large language models. arXiv preprint arXiv:2412.10117 (2024).

Ronen Eldan and Yuanzhi Li. 2023. Tinystories: How small can language models be

and still speak coherent english? arXiv preprint arXiv:2305.07759 (2023). Flux AI. 2025. Flux AI Official Website. https://flux1ai.com/ Stephanie Fu, Netanel Tamir, Shobhita Sundaram, Lucy Chai, Richard Zhang, Tali

Dekel, and Phillip Isola. 2023. Dreamsim: Learning new dimensions of human visual similarity using synthetic data. arXiv preprint arXiv:2306.09344 (2023).

Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. 2022. An image is worth one word: Personalizing text-toimage generation using textual inversion. arXiv preprint arXiv:2208.01618 (2022).

Google Cloud. 2025. Gemini 2.0 Flash. https://cloud.google.com/vertex-ai/generativeai/docs/models/gemini/2-0-flash

Yuchao Gu, Xintao Wang, Jay Zhangjie Wu, Yujun Shi, Yunpeng Chen, Zihan Fan, Wuyou Xiao, Rui Zhao, Shuning Chang, Weijia Wu, et al. 2023. Mix-of-show: Decentralized low-rank adaptation for multi-concept customization of diffusion models. Advances in Neural Information Processing Systems 36 (2023), 15890–15902.

Yuchao Gu, Yipin Zhou, Yunfan Ye, Yixin Nie, Licheng Yu, Pingchuan Ma, Kevin Qinghong Lin, and Mike Zheng Shou. 2024. ROICtrl: Boosting Instance Control for Visual Generation. arXiv preprint arXiv:2411.17949 (2024).

Yuwei Guo, Ceyuan Yang, Ziyan Yang, Zhibei Ma, Zhijie Lin, Zhenheng Yang, Dahua Lin, and Lu Jiang. 2025. Long context tuning for video generation. arXiv preprint arXiv:2503.10589 (2025).

Huiguo He, Huan Yang, Zixi Tuo, Yuan Zhou, Qiuyue Wang, Yuhang Zhang, Zeyu Liu, Wenhao Huang, Hongyang Chao, and Jian Yin. 2024. Dreamstory: Open-domain story visualization by llm-guided multi-subject consistent diffusion. arXiv preprint arXiv:2407.12899 (2024).

Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. 2021. Clipscore: A reference-free evaluation metric for image captioning. arXiv preprint arXiv:2104.08718 (2021).

Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. 2022. Video diffusion models. Advances in Neural Information Processing Systems 35 (2022), 8633–8646.

Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. 2024. Gpt-4o system card. arXiv preprint arXiv:2410.21276 (2024).

Hongda Jiang, Marc Christie, Xi Wang, Libin Liu, Bin Wang, and Baoquan Chen. 2021. Camera keyframing with style and control. ACM Transactions on Graphics (TOG) 40, 6 (2021), 1–13.

Junjie Ke, Qifei Wang, Yilin Wang, Peyman Milanfar, and Feng Yang. 2021. Musiq: Multi-scale image quality transformer. In Proceedings of the IEEE/CVF international conference on computer vision. 5148–5157.

Jihwan Kim, Junoh Kang, Jinyoung Choi, and Bohyung Han. 2024. Fifo-diffusion: Generating infinite videos from text without training. arXiv preprint arXiv:2405.11473

(2024).

Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. 2024. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603 (2024).

Hongchang Ku and Wei Dong. 2020. Face recognition based on mtcnn and convolutional neural network. Frontiers in Signal Processing 4, 1 (2020), 37–42.

Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. 2023. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International conference on machine learning. PMLR, 19730–19742.

Yunxin Li, Shenyuan Jiang, Baotian Hu, Longyue Wang, Wanqi Zhong, Wenhan Luo, Lin Ma, and Min Zhang. 2025. Uni-moe: Scaling unified multimodal llms with mixture of experts. IEEE Transactions on Pattern Analysis and Machine Intelligence (2025).

Yunxin Li, Haoyuan Shi, Baotian Hu, Longyue Wang, Jiashun Zhu, Jinyi Xu, Zhen Zhao, and Min Zhang. 2024. Anim-director: A large multimodal model powered agent for controllable animation video generation. In SIGGRAPH Asia 2024 Conference Papers. 1–11.

Han Lin, Abhay Zala, Jaemin Cho, and Mohit Bansal. 2023. Videodirectorgpt: Consistent multi-scene video generation via llm-guided planning. arXiv preprint arXiv:2309.15091 (2023).

Chang Liu, Haoning Wu, Yujie Zhong, Xiaoyun Zhang, Yanfeng Wang, and Weidi Xie. 2024b. Intelligent grimm-open-ended visual storytelling via latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 6190–6200.

Yaofang Liu, Xiaodong Cun, Xuebo Liu, Xintao Wang, Yong Zhang, Haoxin Chen, Yang Liu, Tieyong Zeng, Raymond Chan, and Ying Shan. 2024a. Evalcrafter: Benchmarking and evaluating large video generation models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 22139–22149.

Yixin Liu, Kai Zhang, Yuan Li, Zhiling Yan, Chujie Gao, Ruoxi Chen, Zhengqing Yuan, Yue Huang, Hanchi Sun, Jianfeng Gao, et al. 2024c. Sora: A review on background, technology, limitations, and opportunities of large vision models. arXiv preprint arXiv:2402.17177 (2024).

Jiawei Mao, Xiaoke Huang, Yunfei Xie, Yuanqi Chang, Mude Hui, Bingjie Xu, and Yuyin Zhou. 2024. Story-adapter: A training-free iterative framework for long story visualization. arXiv preprint arXiv:2410.06244 (2024).

Chong Mou, Xintao Wang, Liangbin Xie, Yanze Wu, Jian Zhang, Zhongang Qi, and Ying Shan. 2024. T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. In Proceedings of the AAAI conference on artificial intelligence, Vol. 38. 4296–4304.

Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002. Bleu: a method for automatic evaluation of machine translation. In Proceedings of the 40th annual meeting of the Association for Computational Linguistics. 311–318.

Pika Labs. 2025. Pika Labs Official Website. https://pika.art/ Pixverse AI. 2025. Pixverse AI: Official Website. https://pixverse.ai/ Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021a. Learning transferable visual models from natural language supervision. In International conference on machine learning. PMLR, 8748–8763.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021b. Learning transferable visual models from natural language supervision. In International conference on machine learning. PmLR, 8748–8763.

Rémi Ronfard. 2021. Film directing for computer games and animation. In Computer

Graphics Forum, Vol. 40. Wiley Online Library, 713–730. Runway AI, Inc. 2025. Runway AI: Official Website. https://runwayml.com/ Xiayang Shi, Shangfeng Chen, Gang Zhang, Wei Wei, Yinlin Li, Zhaoxin Fan, and

Jingjing Liu. 2025. Jailbreak Attack with Multimodal Virtual Scenario Hypnosis for Vision-Language Models. Pattern Recognition (2025), 112391.

splcher. 2019. Anime Face Dataset. https://www.kaggle.com/datasets/splcher/ animefacedataset Zachary Teed and Jia Deng. 2020. Raft: Recurrent all-pairs field transforms for optical flow. In European conference on computer vision. Springer, 402–419.

Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. 2025. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314 (2025).

Jiuniu Wang, Hangjie Yuan, Dayou Chen, Yingya Zhang, Xiang Wang, and Shiwei Zhang. 2023b. Modelscope text-to-video technical report. arXiv preprint arXiv:2308.06571

(2023).

Limin Wang, Bingkun Huang, Zhiyu Zhao, Zhan Tong, Yinan He, Yi Wang, Yali Wang, and Yu Qiao. 2023a. Videomae v2: Scaling video masked autoencoders with dual masking. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 14549–14560.

Haoning Wu, Erli Zhang, Liang Liao, Chaofeng Chen, Jingwen Hou Hou, Annan Wang, Wenxiu Sun Sun, Qiong Yan, and Weisi Lin. 2023. Exploring Video Quality Assessment on User Generated Contents from Aesthetic and Technical Perspectives. In International Conference on Computer Vision (ICCV).

Weijia Wu, Zeyu Zhu, and Mike Zheng Shou. 2025. Automated Movie Generation via Multi-Agent CoT Planning. arXiv preprint arXiv:2503.07314 (2025).

Ziyi Wu, Aliaksandr Siarohin, Willi Menapace, Ivan Skorokhodov, Yuwei Fang, Varnith Chordia, Igor Gilitschenski, and Sergey Tulyakov. 2024. Mind the Time: TemporallyControlled Multi-Event Video Generation. arXiv preprint arXiv:2412.05263 (2024).

Zhifei Xie, Daniel Tang, Dingwei Tan, Jacques Klein, Tegawend F Bissyand, and Saad Ezzini. 2024. Dreamfactory: Pioneering multi-scene long video generation with a multi-agent framework. arXiv preprint arXiv:2408.11788 (2024).

Zhen Xing, Qijun Feng, Haoran Chen, Qi Dai, Han Hu, Hang Xu, Zuxuan Wu, and Yu-Gang Jiang. 2024. A survey on video diffusion models. Comput. Surveys 57, 2

(2024), 1–42.

Xuenan Xu, Jiahao Mei, Chenliang Li, Yuning Wu, Ming Yan, Shaopeng Lai, Ji Zhang, and Mengyue Wu. 2025a. MM-StoryAgent: Immersive Narrated Storybook Video Generation with a Multi-Agent Paradigm across Text, Image and Audio. arXiv preprint arXiv:2503.05242 (2025).

Zhenran Xu, Longyue Wang, Jifang Wang, Zhouyi Li, Senbao Shi, Xue Yang, Yiyu Wang, Baotian Hu, Jun Yu, and Min Zhang. 2025b. Filmagent: A multi-agent framework for end-to-end film automation in virtual 3d spaces. arXiv preprint arXiv:2501.12909 (2025).

Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. 2024. Cogvideox: Textto-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072 (2024).

Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. 2023. Ip-adapter: Text compatible image prompt adapter for text-to-image diffusion models. arXiv preprint arXiv:2308.06721 (2023).

Weihao Yu, Pan Zhou, Shuicheng Yan, and Xinchao Wang. 2024. Inceptionnext: When inception meets convnext. In Proceedings of the IEEE/cvf conference on computer vision and pattern recognition. 5672–5683.

Zibo Zhao, Zeqiang Lai, Qingxiang Lin, Yunfei Zhao, Haolin Liu, Shuhui Yang, Yifei Feng, Mingxin Yang, Sheng Zhang, Xianghui Yang, et al. 2025. Hunyuan3d 2.0: Scaling diffusion models for high resolution textured 3d assets generation. arXiv preprint arXiv:2501.12202 (2025).

Yupeng Zhou, Daquan Zhou, Ming-Ming Cheng, Jiashi Feng, and Qibin Hou. 2024. Storydiffusion: Consistent self-attention for long-range image and video generation. Advances in Neural Information Processing Systems 37 (2024), 110315–110340.

[Figure 5]

- Fig. 5. A comparative case showcasing AniMaker and models specialized in visual narratives. This figure illustrates the visualization of the short story of Tom and Lily. In the story, Tom brings a sack of toys to the town square, where he meets a sad girl named Lily who has no toys. Tom offers to share his toys, and the two children happily play together. Three models—StoryDiffusion, StoryAdapter, and AniMaker (ours)—are compared. AniMaker demonstrates superior narrative consistency, emotional expression, and character continuity across frames. It coherently depicts the extended action sequence of Tom picking up the sack, leaving his house, and arriving at the square. In contrast, while StoryDiffusion and StoryAdapter capture key moments from the story, they suffer from inconsistencies in visual coherence and character alignment, with mismatched character appearances highlighted by red boxes in the figure.

[Figure 6]

- Fig. 6. A comparative case showcasing AniMaker and models capable of generating storytelling videos. This figure visualizes the story of Sue, a little girl who tries to climb a big tree in the park but gets scared. Her friend Tom warns her to be careful, and she climbs down safely. Grateful, Sue hugs Tom, and they play on the swings together. The comparison includes MovieAgent, MMStoryAgent, VideoGen-of-Thought, and AniMaker (ours). AniMaker stands out with coherent scene progression, expressive character interactions, and consistent character identities. It clearly captures Sue’s emotional journey and key events—from climbing the tree and feeling afraid, to receiving help and having fun—demonstrating strong temporal and narrative alignment. In contrast, MovieAgent shows limited relevance to the input story, with inconsistent visuals and abstract content. VideoGen-of-Thought and MMStoryAgent follow the narrative more closely but still suffer from visual continuity issues, with character mismatches highlighted in red boxes.

- A LMM Prompts

This section presents the comprehensive set of prompts engineered for the LMM, which steers the entire automated animation creation pipeline. The figures below illustrate the sequence of these prompts, starting from the initial transformation of a narrative story into a structured script (Figure 7) and its corresponding validation (Figure 8). Following this, we detail the prompts for generating scene imagery (Figure 9), the video itself (Figure 10), and an analysis of the video generation prompt (Figure 11). The final set of prompts covers the creation (Figure 12), modification (Figure 13), and verification (Figure 14) of the voiceover script.

- B Details of AniEval

AniEval enhances the evaluation framework of EvalCrafter [Liu et al. 2024a] through two main aspects. First, it automates the assessment of action related metrics by comparing user prompts with actions identified in the video clips. Second, it integrates several new metrics, including DreamSim [Fu et al.2023], Count-Score [Cheng et al.2024], and Face Consistency. For a comprehensive assessment, AniEval is organized into 4 primary domains comprising 14 fine-grained metrics, as detailed below.

- B.1 Overall Video Quality

- B.1.1 VQA_A and VQA_T. Dover [Wu et al. 2023] is employed to assess generated video quality, providing both an aesthetic score (VQA_A) and a technical score (VQA_T). The aesthetic score focuses on the overall visual appeal of the video, assessing aspects such as composition, color harmony, and artistic quality, while technical score evaluates common distortions like noise and artifacts.
- B.1.2 MusIQ. MUSIQ [Ke et al. 2021] is adopted to evaluate the perceptual quality of the generated videos. Each clip is first sampled into a set of frames, after which the MUSIQ score is computed for every frame and averaged across the entire sequence. This yields a unified quality index that simultaneously captures global composition, local sharpness, and typical distortions such as blur and compression artifacts.

- B.2 Text-Video Alignment

- B.2.1 Text-Video Consistency. CLIPScore [Radford et al. 2021b] is employed to measure the semantic consistency between the input text prompt and the generated video. The pretrained ViT-B/32 CLIP model serves as the feature extractor, yielding text embeddings and frame-wise image embeddings; their cosine similarity is computed for every frame. The overall CLIPScore is obtained by averaging these per-frame similarities across the entire sequence.
- B.2.2 Text-Story Consistency. To further evaluate the alignment between the input prompt and the generated video, a method leveraging BLIP2 [Li et al. 2023] and BLEU [Papineni et al. 2002] is also adopted. The BLIP2 model serves as a caption generator, producing five distinct text descriptions for each video. The BLEU score is then computed to measure the similarity between the original input prompt and each of these five generated captions. The overall score is the average of the five resulting BLEU scores.

- B.2.3 Detection-Score. For each prompt, Gemini first identifies the key object and the desired quantity. To quantify its presence in the video, A Detection Score is computed by the SAMTrack model [cheng2023segment]. SAMTrack performs object detection on uniformly sampled frames; each frame returns a binary result (1 if the object is detected, 0 otherwise). The Detection Score is the mean of these binary values across all sampled frames.
- B.2.4 Count-Score. Furthermore, to evaluate whether the video contains the correct number of objects, a Count Score is calculated. Using the same method, the number of detected objects in each frame is counted. This number is compared to the target quantity, yielding a binary frame-level score: 1 if the counts match, and 0 otherwise. The Count Score is the mean of these binary values across all sampled frames. B.3 Video Consistency

- B.3.1 DreamSim. DreamSim [Fu et al. 2023] is employed to measure the overall similarity between a given clip and its adjacent clips. For each comparison (i.e., between the current clip and its preceding or succeeding clip), frames are extracted from both. For every frame in the current clip, its minimum distance to any frame in the adjacent clip is calculated by Dreamsim. The final DreamSim score for the pair is the average of these distances.
- B.3.2 Face Consistency. The Face Consistency metric evaluates animated character facial consistency using an InceptionNext [Yu et al. 2024] model trained on the Anime Face Dataset [splcher 2019]. The process begins by employing a Segment-and-Track-Anything (SAM) based approach to detect and extract character faces from video clips and reference image, guided by a text prompt. Each extracted face is then passed through the InceptionNext model to obtain a highdimensional feature embedding, where the L2 distance between embeddings serves as a measure of dissimilarity. The metric performs two types of evaluation: for Reference-to-Video Consistency, it calculates the average L2 distance between the reference image’s face embedding and all face embeddings from the video frames. For Temporal Consistency between adjacent video clips featuring the same character, it determines the L2 distance from each face in the current clip to any face in the adjacent clip, and the final score is the average of these distances.
- B.3.3 Warping Error. A pre-trained optical flow estimation network [Teed and Deng 2020] is employed to obtain the optical flow between pairs of frames. Using this information, one frame is warped to align with the next. The pixel-wise difference between the warped image and the actual target frame is then calculated. Warp differences are computed for every frame pair, and the final score is obtained by averaging these results across the entire sequence.
- B.3.4 Semantic Consistency. Beyond pixel-wise error, semantic consistency is also evaluated to assess the consistency between consecutive frames. Specifically, this involves calculating the cosine similarity between the feature embeddings (extracted by ViT-B/32 CLIP model) of each pair of adjacent frames. The final semantic consistency score is then derived by averaging these similarity values across all consecutive frame pairs in the video.

B.4 Motion Quality

For every prompt, Gemini is used to determine:

mm_action: Describe the primary action of the main character – e.g., “walking”, “talking”, “hugging”.

raft_amp: The perceived motion speed – “fast” or “slow”.

B.4.1 Action Recognition. The evaluation process begins by using the MMAction2 [Contributors 2020] toolbox and the VideoMAE V2 model [Wang et al. 2023a] to identify the top five most probable actions and their corresponding confidence scores. Subsequently, a CLIP model calculates the textual similarity between these five predictions and the target action (mm_action). The final score is a weighted sum of these similarities, with the confidence scores serving as the weights.

- B.4.2 Action Strength. The general action strength of the generated video is considered to ensure it is not overly static. To achieve this, the RAFT model [Teed and Deng 2020] is first employed to extract the dense optical flows between each pair of frames. An average flow score is then computed by averaging these dense flows across the entire video clip.
- B.4.3 Motion AC-Score. The alignment between a video’s motion and the prompt is evaluated using the average optical-flow score. A classification of "fast" is assigned if the score exceeds the empirical threshold of 𝜌 = 5; otherwise, it is labeled "slow." A final comparison is then made between this label and the motion level required by the prompt.

[Figure 7]

##### Fig. 7. Prompt for Story to Script Generation

[Figure 8]

##### Fig. 8. Prompt for Story to Script Check

[Figure 9]

- Fig. 9. Prompt for Scene Image Generation

[Figure 10]

- Fig. 10. Prompt for Video Generation

[Figure 11]

Fig. 11. Prompt for Video Generation Prompt Analysis

[Figure 12]

Fig. 12. Prompt for Voiceover Script

[Figure 13]

Fig. 13. Prompt for Voiceover Script Modification

[Figure 14]

##### Fig. 14. Prompt for Voiceover Script Check

