

[Figure 2]

[Figure 3]

## Unified Personalized Reward Model for Vision Generation

### Yibin Wang1,2, Yuhang Zang4†, Feng Han1,2, Jiazi Bu3,4, Yujie Zhou3,4, Cheng Jin1,2†, Jiaqi Wang2†

# arXiv:2602.02380v2[cs.CV]10Feb2026

1Fudan University 2Shanghai Innovation Institute 3Shanghai Jiaotong University 4Shanghai AI Lab

### Abstract

Recent advancements in multimodal reward models (RMs) have significantly propelled the development of visual generation. Existing frameworks typically adopt Bradley–Terry-style preference modeling or leverage generative vision-language models (VLMs) as judges, and subsequently optimize visual generation models via reinforcement learning. However, current RMs suffer from inherent limitations: they often follow a “one-size-fits-all” paradigm that assumes a monolithic preference distribution or relies on fixed evaluation rubrics. As a result, they are insensitive to contentspecific visual cues, leading to systematic misalignment with subjective and context-dependent human preferences. To this end, inspired by human assessment, we propose UnifiedReward-Flex, a unified personalized reward model for vision generation that couples reward modeling with flexible and context-adaptive reasoning. Specifically, given a prompt and the generated visual content, it first interprets the semantic intent and grounds on visual evidence, then dynamically constructs a hierarchical assessment by instantiating fine-grained criteria under both predefined and self-generated high-level dimensions. Our training pipeline follows a two-stage process: (1) we first distill structured, high-quality reasoning traces from advanced closed-source VLMs to bootstrap supervised fine-tuning (SFT), equipping the model with flexible and context-adaptive reasoning behaviors; (2) we then perform direct preference optimization (DPO) on carefully curated preference pairs to further strengthen reasoning fidelity and discriminative alignment. To validate the effectiveness, we integrate UnifiedReward-Flex into the Group Relative Policy Optimization (GRPO) framework for image and video synthesis. Extensive results demonstrate its superiority: it consistently delivers more robust, context-aware reward signals than existing baselines and achieves substantial improvements across diverse image and video generation models.

Website: https://codegoat24.github.io/UnifiedReward/flex

### 1 Introduction

With the rapid progress of multimodal reward models (RMs) [1, 2, 3, 4, 5], their potential in aligning vision generation models with human preferences has attracted growing attention. By converting subjective and high-level human judgments into learnable reward signals, RMs enable effective preference-driven post-training for vision generators. [6, 7, 8, 9, 10, 11]. Early reward modeling for visual generation typically

†Corresponding authors.

uses fixed, discriminative scorers [12, 13, 14, 15] to assign scalar rewards. Later works shift to Bradley–Terry [16] pairwise preference modeling to learn rewards from relative comparisons [4, 17]. More recently, VLM-asa-judge [1, 2, 3, 18] has emerged, leveraging strong generative VLMs [19, 20] to provide context-dependent evaluations for reinforcement learning.

Despite their effectiveness, current RMs often follow a “one-size-fits-all” assessment paradigm: (1) Fixed discriminative scorers (e.g., CLIP [12], PickScore [15]) and Bradley–Terry preference models (e.g., VideoAlign [4], HPSv3 [17]) typically learn a single global reward function that assigns a scalar score to each input (or induces preferences via score differences), implicitly assuming a monolithic preference distribution shared across diverse prompts and visual contents. (2) VLM-as-a-judge approaches (e.g., UnifiedReward-Think [2]) leverage generative VLMs to produce richer textual judgments, yet they often follow static evaluation rubrics with a fixed checklist of criteria. As a result, their reward feedback remains insensitive to content-specific visual cues, which can misguide optimization and cause systematic misalignment with human preferences.

In this work, we posit that reliable reward assessment should be flexibly adapted to the prompt intent and visual content. For example, prompts with implicit narrative intent should emphasize storytelling consistency, subject relationships, and emotional tone (see Fig. 1), whereas motion-intensive videos with frequent physical contact demand explicit evaluation of action dynamics and physical plausibility (see Fig. 2). This behavior closely mirrors how humans assess visual generations: evaluators first interpret the prompt intent and the depicted content, then evaluate along a small set of common, task-specific high-level dimensions. Within each dimension, they selectively attend to the aspects most relevant to the given instance. When the scene exhibits additional characteristics, evaluators naturally introduce new high-level dimensions to capture these salient factors. In other words, human evaluation is inherently a content-adaptive, context-aware reasoning process, where both the evaluation criteria and their relative importance are dynamically adjusted to match the semantic intent and visual evidence.

As inspired, this work proposes UnifiedReward-Flex, a unified personalized reward model for vision generation that couples reward modeling with context-adaptive reasoning to dynamically tailor evaluation criteria, which performs assessment in a hierarchical manner. Specifically, given a prompt and the generated visual content, it first interprets the semantic intent and extracts salient visual evidence, then composes a hierarchical evaluation plan by instantiating fine-grained sub-criteria under a few predefined coarse-grained dimensions. When the context demands additional considerations, it augments the hierarchy with new high-level dimensions and their corresponding sub-criteria. This dynamic criterion composition yields adaptive and informative reward feedback that aligns with the human evaluation process and remains robust across diverse prompts and visual content. Our training follows a two-stage pipeline: (1) We first distill structured, high-quality reasoning traces from advanced closed-source VLMs [21] to bootstrap supervised fine-tuning (SFT), endowing the model with content-aware criterion composition. (2) We then apply direct preference optimization (DPO) using preference pairs with human-grounded labels: given two sampled responses for the same input, if one reaches the correct conclusion while the other does not, the preference naturally favors the correct one; if both are correct, we further assign the preference based on the quality of their adaptive reasoning trajectories, prioritizing more flexible and context-grounded evaluation hierarchies. Empirically, this reasoning-aware preference alignment improves the reward model’s discriminative power, even among samples that are all correct.

Extensive experiments show that UnifiedReward-Flex consistently outperforms strong reward model baselines on both image and video reward tasks, providing more robust and context-aware reward signals. To further validate its practical utility, we apply it as the reward signal for pairwise preference–based GRPO [6] on multiple vision generation models. Specifically, we observe consistent improvements on text-to-image generators such as FLUX.1-dev and FLUX.2-klein-base, as well as on text-to-video generators including Wan2.1 and Wan2.2, yielding substantial quantitative and qualitative gains in downstream generation quality.

Contributions. (1) We identify a key limitation of existing multimodal reward models, i.e., their “onesize-fits-all” evaluation paradigm, and propose UnifiedReward-Flex, a unified personalized reward model for vision generation that dynamically composes evaluation hierarchies via content-aware reasoning. (2) Our UnifiedReward-Flex consistently outperforms strong reward model baselines on both image and video reward tasks, delivering more robust and context-aware reward signals. (3) We further validate its practical utility by integrating it into pairwise preference reward-based GRPO for image and video synthesis, achieving substantial improvements in downstream generation quality both quantitatively and qualitatively.

Image#1 Image#2

Grimdark mystic beautiful magical child healing a wounded majestic kirin, close-up view, anime style, high fantasy, digital art, masterpiece painting, ultra detailed, ultra high definition, 8K resolution, 3D shading, superior quality, cover illustration, complex design, bright luminous highlights, rich vibrant colours, hyperrealistic, meticulously hyper detailed, high contrast, artstation.

[Figure 4]

[Figure 5]

UnifiedReward Flex

[Figure 6]

[Figure 7]

[Figure 8]

A. Semantic Alignment & Accuracy

###### B. Image Quality & Realism

- 1. Subject Completeness (Child & Kirin)
- 2. Action Fidelity (Healing/Interaction)

- 1. 3D Shading & Texture Precision
- 2. Lighting & Luminous Highlights

- Image#1: Image 1 completely misses the 'magical child' subject, rendering only the

creature. (Score: 1)

- Image#2: Image 2 depicts both the majestic kirin and the child, fulfilling the core

- Image#1: The scales and fur in Image 1 exhibit exceptional 3D shading and crisp,

hyper-realistic definition. (Score: 7)

- Image#2: Image 2 uses a slightly softer, more painterly brushwork that is

subject requirements. (Score: 9)

beautiful but less 'hyper-detailed' in texture than Image 1. (Score: 3)

- Image#1: Image 1 features very dynamic backlighting that accentuates the

silhouette and transparency of the horns. (Score: 6)

- Image#2: Image 2 has good lighting, but the source is centralized around the

- Image#1: There is no action or interaction in Image 1; it is a static portrait. (Score: 1)
- Image#2: Image 2 portrays the child reaching out to the creature, effectively

visualizing the requested 'healing' interaction. (Score: 9)

interaction, leaving some details in the periphery less distinct. (Score: 4)

###### 3. Atmosphere (Grimdark vs. Luminous)

###### 3. Color Vibrancy & Contrast

- Image#1: Image 1 captures the luminous highlights and dark contrast well. (Score: 5)
- Image#2: Image 2 balances the grimdark background with the warm, luminous glow

- Image#1: Displays rich blues and magentas with high contrast. (Score: 5)
- Image#2: Displays a striking contrast between the white creature, dark void, and

of the healing magic effectively. (Score: 5)

warm fire/magic colors. (Score: 5)

Category Reason: Image 2 followed the prompt instructions to include both a child and a kirin, whereas Image 1 ignored the child entirely.

Category Reason: Image 1 edges out slightly in technical rendering, specifically regarding the '3D shading' and 'hyper detailed' texture of the scales.

[Figure 9]

[Figure 10]

Category Winner: Image 2

Category Winner: Image 1

[Figure 11]

[Figure 12]

###### D. Narrative & Interaction

###### C. Aesthetics & Artistry

- 1. Emotional Resonance
- 2. Scale Representation

1. Composition & Balance

- Image#1: The framing is a standard close-up headshot, lacking dynamic

arrangement. (Score: 3)

- Image#2: Excellent use of scale to contrast the massive kirin against the

- Image#1: Lacks emotional weight as it is just a creature portrait. (Score: 2)
- Image#2: Evokes a sense of trust and magic through the gentle gesture of the

child towards the beast. (Score: 8)

small child, creating a balanced and dramatic composition. (Score: 7)

2. Style Coherence (High Fantasy Anime)

- Image#1: Fits the high fantasy anime aesthetic perfectly. (Score: 5)
- Image#2: Also fits the genre well, resembling a high-end light novel cover.

- Image#1: Scale is indeterminate without a reference point (like a child). (Score: 3)
- Image#2: Effectively conveys the 'majestic' size of the kirin by contrasting it

(Score: 5)

with the small child. (Score: 7)

###### 3. Visual Impact

###### 3. Storytelling Elements

- Image#1: Visually pretty, but static and somewhat generic without the secondary

subject. (Score: 4)

- Image#2: The interaction draws the eye and evokes curiosity, offering higher

- Image#1: There is no narrative present. (Score: 1)
- Image#2: Successfully tells the story of a magical healing moment. (Score: 9)

artistic value. (Score: 6)

Category Reason: Image 2 offers a much more compelling composition suitable for a 'cover illustration', utilizing scale and narrative tension.

Category Reason: This category is necessary because the prompt implies a story ('healing a wounded...'). Image 2 succeeds entirely here while Image 1 fails.

[Figure 13]

[Figure 14]

Category Winner: Image 2

Category Winner: Image 2

Overall Reason: Image 2 is the clear winner primarily due to semantic alignment. The prompt explicitly describes a scene of a 'child healing a wounded majestic kirin'. Image 2 depicts both the child and the creature interacting in a way that suggests connection and healing. Image 1, while visually striking, completely fails to include the 'child' or the action of 'healing', resulting in a portrait of a creature only. While Image 1 has slightly superior texture rendering, it fails the core prompt request.

[Figure 15]

Overall Winner: Image 2

Figure 1 Qualitative Result of UnifiedReward-Flex on Image Generation Personalized Reward Reasoning.

### 2 Related Work

Multimodal Reward Models (RMs) are crucial in aligning vision generation models with human preferences. Early reward modeling typically relies on fixed, discriminative scorers that assign a scalar score to each generated sample [12, 13, 14, 15]. These scorers are lightweight and easy to deploy, but they often behave as task-agnostic heuristics: the reward function is largely fixed across prompts and contents, making it difficult to reflect diverse evaluation focuses. To better capture the relative nature of human judgment, later works shift from absolute scoring to pairwise preference modeling under the Bradley-Terry pairwise preference modeling [16]. Instead of regressing a single target score, the RM is trained to assign higher reward to preferred samples within a comparison pair, which often yields more stable and better calibrated training signals for preference optimization [4, 5, 17]. Nevertheless, Bradley–Terry based RMs still typically learn a single global scalar reward function shared across heterogeneous prompts and visual contents, implicitly assuming a monolithic preference distribution. More recently, VLM-as-a-judge has emerged as a flexible alternative that leverages strong generative VLMs to directly assess and compare candidates [1, 2, 3, 18, 22, 23, 24, 25]. Compared to discriminative scorers, these judge models can provide richer evaluative feedback, potentially incorporating multi-aspect reasoning and content-dependent judgments. However, existing VLM-judge approaches commonly follow static evaluation rubrics or fixed checklists of criteria, which limits their ability to dynamically tailor evaluation priorities to the semantic intent and visual evidence of each input. To this end, this work proposes UnifiedReward-Flex, a unified personalized reward model for vision generation that couples reward modeling with content-aware reasoning. Unlike prior RMs that apply a fixed global scoring function or static rubrics, UnifiedReward-Flex dynamically constructs a context-adaptive hierarchical

- Video#1

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

- Video#2

[Figure 20]

[Figure 21]

A confident, anthropomorphic fox dressed in a dashing explorer’s outfit stands atop a mountain, surveying the landscape with a map in hand. The camera starts with a wide shot of the mountain range, then zooms in to focus on the fox’s determined expression as it adjusts its hat and begins to descend the rocky terrain. The animation highlights the fox’s agility and adventurous spirit as it leaps over obstacles and uncovers ancient ruins.

[Figure 22]

[Figure 23]

UnifiedReward Flex

[Figure 24]

[Figure 25]

[Figure 26]

A. Semantic Alignment & Accuracy

###### B. Video Quality & Dynamic Realism

1. Sequential Event Execution (Zoom -> Act -> Descend)

1. Motion Clarity & Anti-Blur

- Video#1: Video 1 clearly transitions from a scenic view to a character focus, then

to the specific actions of adjusting the hat and leaping down. (Score: 9)

- Video#2: Video 2 captures the hat adjustment but fails to execute the camera

- Video#1: The video maintains sharp focus on the subject even during the rapid leaping motion. (Score: 9)
- Video#2: The video suffers from extreme motion smear and artifacts in the second half, rendering the action unwatchable. (Score: 1)

sequence or the subsequent descent action, devolving into a blur. (Score: 1)

2. Prop Visibility & Interaction (Map)

2. Anatomical Consistency During Action

- Video#1: The map is clearly visible in the fox's hand and is held naturally as part of

the survey action. (Score: 8)

- Video#2: The map is barely discernible or entirely absent/obscured, with the focus

- Video#1: The fox maintains its anthropomorphic proportions and facial features consistent throughout the jump. (Score: 8)
- Video#2: The character's form melts and distorts significantly as soon as rapid movement begins. (Score: 2)

remaining solely on the hat interaction. (Score: 2)

###### 3. Environmental Interaction (Rocky Terrain)

###### 3. Texture Fidelity (Fur & Leather)

- Video#1: The subject physically interacts with the rocks, stepping and leaping off

them, grounding the character in the requested environment. (Score: 9)

- Video#2: The environment becomes an indistinct background wash once the

- Video#1: The textures of the jacket leather and the fox's fur remain high-

definition and distinct. (Score: 8)

- Video#2: While the initial frame has a nice 'soft' look, the texture quality

character moves; there is no clear interaction with the terrain. (Score: 1)

degrades instantly into noise upon movement. (Score: 2)

Category Reason: Video 1 systematically hits every plot point: the map usage, the hat adjustment, and the physical descent. Video 2 misses the map clarity and fails to depict the descent intelligibly. Category Winner: Video 2

Category Reason: The difference in quality is stark; Video 1 remains sharp and coherent throughout, while Video 2 loses all structural integrity during motion.

[Figure 27]

[Figure 28]

Category Winner: Video 1

[Figure 29]

[Figure 30]

###### C. Narrative, Aesthetics & Cinematography

###### D. Action Dynamics & Physics

- 1. Locomotion Mechanics (Leaping)
- 2. Complex Agility Representation

1. Camera Movement Logic (Zoom/Focus)

- Video#1: The camera simulates a professional push-in/zoom that directs

attention from the landscape to the character's intent. (Score: 8)

- Video#2: The camera movement is erratic and shaky, failing to guide the

- Video#1: Video 1 depicts a clear, ballistic trajectory for the leap, with believable

takeoff and landing intent. (Score: 9)

- Video#2: Video 2 shows no clear mechanics; the character simply blurs across the

viewer's eye as described in the prompt. (Score: 2)

screen without defined limbs or trajectory. (Score: 1)

2. Characterization (Confidence & Determination)

- Video#1: The fox's body language and brisk movement convey the requested

'confidence' and 'adventurous spirit'. (Score: 7)

- Video#2: The initial expression is pleasant, but the subsequent chaotic motion

- Video#1: The fluid transition from adjusting the hat to running/jumping effectively

communicates agility. (Score: 8)

- Video#2: There is no agility, only a chaotic camera shake and blur that obscures

destroys any character building. (Score: 3)

movement. (Score: 2)

###### 3. Spatial Navigation (Descending)

###### 3. Visual Style Consistency

- Video#1: The character clearly moves downwards through the 3D space of the

environment. (Score: 9)

- Video#2: Spatial coherence is lost; it is impossible to tell if the character is

- Video#1: The visual style is consistent from start to finish, looking like a polished

game cinematic. (Score: 9)

- Video#2: The video breaks completely halfway through, jarring the viewer with a

descending or just vibrating in place. (Score: 1)

sudden drop in visual integrity. (Score: 1) Category Reason: Video 1 delivers a cohesive cinematic shot that tells the ministory requested. Video 2 feels like a broken render that fails to convey the narrative of exploration. Category Winner: Video 1

Category Reason: Since the prompt specifically requested 'agility', 'descending', and 'leaping', this category is essential. Video 1 executes these physics convincingly.

[Figure 31]

[Figure 32]

Category Winner: Video 1

Overall Reason: Video 1 demonstrates superior adherence to the complex multi-stage prompt, successfully executing the camera movement (wide to zoom), the specific character actions (adjusting hat, looking at map, leaping), and maintaining high visual fidelity during rapid motion. Video 2, while aesthetically pleasing in the initial static frames, suffers from catastrophic degradation and severe motion blur during the movement phase, completely failing to depict the descent or agility required.

[Figure 33]

Overall Winner: Video 1

Figure 2 Qualitative Result of UnifiedReward-Flex on Video Generation Personalized Reward Reasoning.

assessment by self-instantiating fine-grained criteria based on given prompt intent and visual evidence.

Reinforcement Learning for Vision Generation has evolved rapidly in recent years. Early attempts either fine-tuned models with scalar reward supervision [26, 27] or adopted reward-weighted regression to exploit reward feedback in a more stable manner [3, 4, 28]. Subsequent work drew inspiration from Proximal Policy Optimization (PPO) [29] and incorporated policy-gradient updates into diffusion-based generators, showing promising improvements in sample quality [30, 31, 32]. Despite their effectiveness, these methods are often computationally demanding and sensitive to hyperparameter tuning. To improve training efficiency, a growing line of research explores preference optimization with supervised objectives, such as direct preference optimization (DPO), which directly leverages human preference data and avoids expensive on-policy rollouts [33, 34, 35, 36]. More recently, Group Relative Policy Optimization (GPRO) [37] has emerged as a promising objective for preference optimization in complex reasoning tasks. [9, 10] extend GRPO to flow matching models by reformulating the ODE sampling process as an equivalent SDE, enabling diverse sampling while performing preference-driven optimization. [7, 11, 38] further improve efficiency with a sliding-window mechanism that localizes SDE sampling and GRPO updates within a window while retaining ODE sampling elsewhere. However, these approaches typically optimize pointwise scalar rewards and may suffer from reward hacking [8, 9]. Pref-GRPO [6] reveals that reward hacking is largely driven by “illusory advantages” induced by unreliable reward signals, and stabilizes learning by replacing pointwise rewards with pairwise preference-based feedback. In this work, we integrate our UnifiedReward-Flex into the Pref-GRPO framework for both image and video synthesis to validate its effectiveness.

### 3 Method

#### 3.1 Overview

Multimodal reward models (RMs) serve as learned proxies for human visual preferences. However, existing RMs often follow a “one-size-fits-all” paradigm: they either assume a monolithic preference distribution captured by a single global scoring function, or rely on fixed evaluation rubrics that apply the same criteria uniformly across inputs. As a result, their reward feedback is often insensitive to content-specific visual cues and prompt intent, leading to systematic misalignment with inherently subjective and context-dependent human preferences. To this end, we propose UnifiedReward-Flex, a unified personalized reward model for vision generation that couples reward modeling with context-adaptive reasoning to dynamically tailor evaluation criteria.

In this section, we first present our unified personalized reward modeling framework (Sec. 3.2), including the design of our context-adaptive reasoning process (Sec. 3.2.1) and a two-stage training pipeline: (i) reasoning distillation for SFT (Sec. 3.2.2) and (ii) reasoning-aware preference alignment via DPO (Sec. 3.2.3). We then describe how to apply UnifiedReward-Flex to reinforcement learning for vision generation (Sec. 3.3), starting with the necessary preliminaries (Sec. 3.3.1), and followed by our employed personalized multi-dimensional preference rewards for GRPO (Sec. 3.3.2).

#### 3.2 Unified Personalized Reward Modeling

##### 3.2.1 Context-adaptive Reasoning Process Design

Our goal is to mimic how humans evaluate generated visual content. In practice, for a given task, human evaluators typically assess outputs along a small set of common high-level dimensions (e.g., semantic alignment and perceptual quality), while instantiating more fine-grained factors depending on the prompt intent and the visual evidence under each dimensions. When additional considerations become salient (e.g., Narrative & Interaction in Fig. 1), they naturally extend the evaluation dimensions to better capture what matters for the current instance.

Guided by this principle, we design a context-adaptive hierarchical reasoning process that starts from three predefined common dimensions as stable anchors (e.g., semantic alignment, visual quality, and aesthetics for image generation). Under each anchor, the model instantiates prompt-specific sub-dimensions for assessment, and it can further introduce new high-level dimensions (with corresponding sub-dimensions) when required by the context. For each high-level dimension, the model aggregates the evidence across its instantiated sub-dimensions to produce an overall comparative analysis and a dimension-level winner; it then combines the outcomes from all high-level dimensions to determine a final overall winner.

Specifically, as shown in Fig. 2, UnifiedReward-Flex derives prompt-relevant sub-dimensions under the given anchors to jointly assess semantic adherence, visual quality under motion, and cinematographic coherence of generated videos. Since this case emphasizes rich motion and physical interactions, the model further introduces an additional high-level dimension (i.e., Action Dynamics & Physics) to explicitly evaluate temporal coherence and physical plausibility, which is then incorporated into the final overall decision.

By tailoring the evaluation hierarchy to the generation context, our model provides richer reward supervision for preference optimization, improving both intent satisfaction and content-critical quality factors.

##### 3.2.2 Stage I: Reasoning Distillation for SFT

To bootstrap the model with context-adaptive assessment behaviors, we first distill structured reasoning traces from the powerful closed-source VLM [21] and use them for supervised fine-tuning (SFT).

Distillation data. Let 𝒟 = {𝑥𝑖}𝑁𝑖=1 be a set of preference-evaluation instances, where each input

𝑥𝑖 = (𝑝𝑖, 𝑣(𝑖0), 𝑣(𝑖1)) (1)

contains a text prompt 𝑝𝑖 and a pair of candidate visual generations 𝑣(𝑖0), 𝑣(𝑖1) . Given 𝑥𝑖, the closed-source teacher model 𝒯 outputs a structured evaluation trace

𝑦𝑖𝒯 = 𝒯 (𝑥𝑖) = ℋ𝑖, ℛ𝑖, 𝒲𝑖 , (2)

where ℋ𝑖 = {(𝑑𝑘, 𝒮𝑖,𝑘)}𝐾𝑘=𝑖 1 denotes the instantiated high-level dimensions 𝑑𝑘 together with their promptspecific sub-dimensions 𝒮𝑖,𝑘, ℛ𝑖 is the corresponding evidence-grounded reasoning trace, and 𝒲𝑖 =

{𝑤𝑖,𝑘}𝐾𝑘=𝑖 1, 𝑤𝑖 are winner labels. Here 𝑤𝑖,𝑘 ∈ {0, 1} denotes the winner under dimension 𝑑𝑘 and 𝑤𝑖 ∈ {0, 1} denotes the overall winner.

SFT objective. We fine-tune the base model [2] with parameters 𝜃 to imitate the teacher outputs via conditional language modeling:

ℒSFT(𝜃) = −

𝑁

𝑁

log 𝑝𝜃 𝑦𝑖𝒯 | 𝑥𝑖 = −

𝑖=1

𝑖=1

|𝑦𝑖𝒯 |

log 𝑝𝜃 𝑦𝑖,𝑡𝒯 | 𝑥𝑖, 𝑦𝑖,𝒯<𝑡 . (3)

𝑡=1

This stage initializes the model to generate structured, context-adaptive evaluations for paired visual comparisons, providing a strong foundation for subsequent preference alignment.

##### 3.2.3 Stage II: Reasoning-Aware Preference Alignment via DPO

Building on the SFT-initialized context-adaptive evaluations, we further optimize it for preference discrimination under human-grounded supervision. Specifically, we apply Direct Preference Optimization (DPO) to jointly align the final decision and the quality of the adaptive reasoning trajectory.

Preference pair construction. For each input 𝑥𝑖 = (𝑝𝑖, 𝑣(𝑖0), 𝑣(𝑖1)), we assume an annotated preference label 𝑤★𝑖 ∈ {0, 1} provided by the dataset, where 𝑤★𝑖 = 0 indicates 𝑣(𝑖0) is preferred and 𝑤★𝑖 = 1 indicates 𝑣(𝑖1) is preferred. Starting from the SFT model, we sample two candidate structured evaluations

𝑦𝑖(𝑎), 𝑦𝑖(𝑏) ∼ 𝜋𝜃(· | 𝑥𝑖), (4)

each containing an overall predicted winner 𝑤ˆ(𝑦) ∈ {0, 1} and a context-adaptive reasoning trace. We define the correctness indicator

𝑐 𝑦𝑖(𝑗) = 𝕀 𝑤 ˆ 𝑦𝑖(𝑗) = 𝑤★𝑖 , 𝑗 ∈ {𝑎, 𝑏}. (5) If exactly one sample is correct, we set the preference to favor the correct one. If both samples are correct, we further rank them by the quality of their adaptive reasoning trajectories, preferring evaluations with more context-grounded and flexible hierarchies. Concretely, when both samples are correct, we obtain a trajectory-level preference by querying a closed-source judge 𝒯judge to perform pairwise comparison and output the preferred reasoning trace, followed by human verification on the judged pairs. We denote the judged preference as

ℓ𝑖traj = 𝒯judge(𝑥𝑖, 𝑦𝑖(𝑎), 𝑦𝑖(𝑏)) ∈ {𝑎, 𝑏}, (6)

where ℓ𝑖traj indicates which trace is preferred in terms of reasoning quality. We then construct the DPO pair (𝑦𝑖+, 𝑦−

𝑖 ) as

 

(𝑦𝑖(𝑎), 𝑦𝑖(𝑏)), 𝑐(𝑦𝑖(𝑎)) > 𝑐(𝑦𝑖(𝑏)) (𝑦𝑖(𝑏), 𝑦𝑖(𝑎)), 𝑐(𝑦𝑖(𝑏)) > 𝑐(𝑦𝑖(𝑎)) (𝑦(ℓ

(7)

(𝑦𝑖+, 𝑦𝑖−) =

 

traj 𝑖 )

ℓ¯𝑖traj)

𝑖 , 𝑦(

𝑖 ), 𝑐(𝑦𝑖(𝑎)) = 𝑐(𝑦𝑖(𝑏)) = 1,

where ℓ¯𝑖traj ∈ {𝑎, 𝑏} \ {ℓ𝑖traj}, and we discard the pair if 𝑐(𝑦𝑖(𝑎)) = 𝑐(𝑦𝑖(𝑏)) = 0. This construction aligns not only the final preference decision but also the quality of the underlying context-adaptive reasoning trajectory.

DPO objective. Given the resulting preference dataset 𝒫 = {(𝑥𝑖, 𝑦𝑖+, 𝑦−

𝑖 )}, we optimize 𝜋𝜃 with the standard DPO loss using a frozen reference policy 𝜋ref (the SFT model):

ℒDPO(𝜃) = −𝔼(𝑥,𝑦+,𝑦−)∼𝒫 log 𝜎 𝛽𝑑𝑝𝑜 log𝜋𝜃(𝑦+| 𝑥) − log𝜋𝜃(𝑦−| 𝑥) − log𝜋ref(𝑦+| 𝑥) + log𝜋ref(𝑦−| 𝑥) , (8)

where 𝛽 controls the strength of preference optimization. By directly increasing the likelihood of preferred structured evaluations, this stage sharpens the reward model’s discriminative ability while reinforcing its

context-adaptive reasoning behavior. Empirically, such reasoning-aware preference alignment improves the reward model’s discriminative power, even among samples that are all correct.

#### 3.3 Reinforcement Learning for Vision Generation

##### 3.3.1 Preliminaries

Flow Matching GRPO. We briefly review GRPO [37] in the context of flow-matching models [9, 10]. Given a prompt 𝑐, a flow-based generator produces an image/video 𝑥0 by iteratively refining a noisy sample from 𝑡 = 𝑇 to 𝑡 = 0. For each sample, we consider terminal reward supervision, where the reward is provided at the end of the trajectory as 𝑅(𝑥0, 𝑐).

Group-relative advantage. Given a prompt 𝑐, GRPO samples a group of 𝐺 generations {𝑥0𝑖 }𝐺𝑖=1 and evaluates them with a reward model to obtain {𝑅(𝑥0𝑖 , 𝑐)}𝐺𝑖=1. To stabilize learning under noisy rewards, GRPO constructs a prompt-wise standardized advantage:

𝐴ˆ𝑖 =

𝑅(𝑥0𝑖 , 𝑐) − 𝜇𝑐 𝜎𝑐

, 𝜇𝑐 = mean {𝑅(𝑥0𝑗 , 𝑐)}𝐺𝑗=1 , 𝜎𝑐 = std {𝑅(𝑥0𝑗 , 𝑐)}𝐺𝑗=1 . (9)

This relative advantage encourages the policy to increase the likelihood of higher-quality generations within the same prompt group, rather than chasing absolute reward values.

GRPO objective. Let 𝜋𝜃 denote the sampling policy induced by the generator, and 𝜋𝜃old be the behavior policy used to collect trajectories. For the 𝑖-th trajectory at step 𝑡, the importance ratio is

𝜋𝜃(𝑥𝑡𝑖−1 | 𝑥𝑡𝑖, 𝑐) 𝜋𝜃old(𝑥𝑡𝑖−1 | 𝑥𝑡𝑖, 𝑐)

. (10)

𝑟𝑡𝑖(𝜃) =

GRPO then maximizes a clipped surrogate objective with KL regularization to a reference policy 𝜋ref:

𝒥GRPO(𝜃) = 𝔼𝑐

𝑇−1

𝐺

1 𝑇

1 𝐺

min 𝑟𝑡𝑖(𝜃)𝐴ˆ𝑖, clip 𝑟𝑡𝑖(𝜃), 1 − 𝜂, 1 + 𝜂 𝐴 ˆ𝑖 − 𝛽𝑘𝑙 𝐷KL(𝜋𝜃 ∥𝜋ref) , (11)

𝑡=0

𝑖=1

where 𝜂 is the clipping threshold and 𝛽𝑘𝑙 controls the strength of regularization. This formulation provides a principled and stable way to post-train vision generators with learned reward signals.

Pairwise Preference Reward-based GRPO (Pref-GRPO) [6] replaces absolute reward scores with relative preference judgments, which better align with the common practice of evaluating visual generations via pairwise comparisons. Instead of assigning an independent scalar reward to each sample, Pref-GRPO derives terminal rewards from comparative outcomes among a group of generated samples.

Given a prompt 𝑐, a group of 𝐺 images (or videos) {𝑥0𝑖 }𝐺𝑖=1 is sampled from the current policy 𝜋𝜃. For

each unordered pair (𝑥0𝑖 , 𝑥0𝑗), a preference model determines which sample is preferred, denoted by 𝑥0𝑖 ≻ 𝑥0𝑗. Based on all pairwise comparisons within the group, Pref-GRPO defines a relative preference reward for each

sample as its normalized win rate:

𝑅(𝑥0𝑖 , 𝑐) =

1 𝐺 − 1 𝑗≠𝑖

𝟙 𝑥0𝑖 ≻ 𝑥0𝑗 , (12)

which reflects how often sample 𝑖 is preferred over other candidates under the same prompt.

These preference-derived rewards are then directly plugged into the standard GRPO framework. In particular, the group-relative advantage 𝐴ˆ𝑖 is computed using Eq. (9), and the policy is optimized with the same clipped surrogate objective in Eq. (11).

##### 3.3.2 Personalized Multi-Dimensional Preference Rewards

In this work, we integrate our flexible, context-adaptive evaluations from UnifiedReward-Flex into PrefGRPO to provide personalized, multi-dimensional rewards. Specifically, given a prompt 𝑐, the policy 𝜋𝜃

Table 1 Image and Video Generation Assessment Comparison.

Video Generation GenAI-Bench MMRB2 GenAI-Bench MJBench

Image Generation

Method

Method

|HPSv2 68.8 55.0 PickScore 70.0 57.6<br><br>HPSv3 70.9 58.5<br><br><br>UnifiedReward 71.5 60.0 UnifiedReward-Think 72.3 66.0<br><br>|LiFT 60.1 51.0 VideoScore 70.6 62.8<br><br>VideoReward 73.1 63.4 UnifiedReward 76.8 68.8<br><br>UnifiedReward-Think 80.3 70.9|
|---|---|
|Ours w/o DPO 71.5 67.5 Ours w/o DPO (Both correct) 72.0 68.4<br><br>|Ours w/o DPO 79.4 69.1 Ours w/o DPO (Both correct) 80.6 70.3<br><br>|
|Ours 73.4 69.2<br><br>|Ours 82.5 72.0|

samples a group of 𝐺 candidates {𝑥0𝑖 }𝐺𝑖=1, and for each unordered pair (𝑥0𝑖 , 𝑥0𝑗), the reward model produces pairwise preference judgments along 𝐷 predefined anchor dimensions:

𝑥0𝑖 ≻𝑑 𝑥0𝑗 , 𝑑 = 1, . . . , 𝐷.

For each candidate, we compute the dimension-wise win rates and their average over the anchor dimensions:

𝐷

1 𝐺 − 1 𝑗≠𝑖

1 𝐷

𝟙 𝑥0𝑖 ≻𝑑 𝑥0𝑗 .

𝑅¯dim(𝑥0𝑖 , 𝑐) =

𝑅𝑑(𝑥0𝑖 , 𝑐), 𝑅𝑑(𝑥0𝑖 , 𝑐) =

𝑑=1

To account for dynamic, personalized high-level dimensions that may not appear consistently in each pairwise comparison, we also compute the overall win rate:

1 𝐺 − 1 𝑗≠𝑖

𝟙 𝑥0𝑖 ≻ 𝑥0𝑗 .

𝑅overall(𝑥0𝑖 , 𝑐) =

The group-relative advantages are then computed separately for the averaged dimension-wise win rate and the overall win rate:

𝑅¯dim(𝑥0𝑖 , 𝑐) − 𝜇dim 𝜎dim

𝑅overall(𝑥0𝑖 , 𝑐) − 𝜇overall 𝜎overall

, 𝐴ˆoverall𝑖 =

𝐴ˆdim𝑖 =

,

where 𝜇 and 𝜎 are the mean and standard deviation within the prompt group. Finally, the combined advantage used in GRPO is

𝐴ˆ𝑖 = 𝛼 𝐴ˆdim𝑖 + (1 − 𝛼) 𝐴ˆoverall𝑖 ,

where 𝛼 controls the relative contributions of the averaged dimension-wise advantage and the overall advantage, respectively.

This procedure ensures that both the fine-grained anchor preferences and the holistic, context-adaptive evaluation contribute to the policy update, while remaining compatible with the training objective in Eq. (11).

### 4 Experiment

#### 4.1 Implementation Details

##### 4.1.1 UnifiedReward-Flex

Datasets. For image generation, we sample 50K image preference pairs from HPDv3 [17]. For video generation, we combine two human preference datasets: Text2Video-Human Preferences (15K), provided by Rapidata, and VideoFeedback2 [18], from which we preprocess and construct an additional 35K video preference pairs. In the SFT stage, we distill reward reasoning traces for both images and videos from GPT-5.2 [21], using 45K image pairs and 45K video pairs to construct our UnifiedReward-Flex-SFT-90K dataset. Training samples are

- Table 2 In-domain Semantic Consistency Comparison on UniGenBench. “UniGenBench-EvalModel-qwen3vl-32b-v1” is used as the VLM for evaluation.

Model Overall Style World Know. Attribute Action Relation. Compound Grammar Logic.Reason. Layout Text

|FLUX.1-dev 59.39 w/ HPSv2 57.77 w/ HPSv3 57.98 w/ PickScore 58.63 w/ UnifiedReward 60.87 w/ UnifiedReward-Think 68.89<br><br>|85.10 85.92 65.28 61.41 64.97 43.56 60.16 24.77 70.52 32.18 77.90 87.03 65.92 57.41 65.86 44.46 55.75 29.09 64.93 29.31 79.40 90.03 66.24 57.89 63.58 39.82 58.82 24.09 67.16 32.76 79.70 87.03 64.42 61.12 67.64 47.42 58.02 27.50 67.54 25.86 83.50 87.97 66.13 63.88 68.65 46.52 58.69 24.32 71.08 37.93 88.00 91.77 77.99 69.20 75.13 61.47 61.63 41.36 77.24 45.11<br><br>|
|---|---|
|w/ UnifiedReward-Flex 73.95|90.30 89.87 79.38 73.38 78.55 69.46 63.10 46.59 79.66 59.20<br><br>|

- Table 3 Out-of-Domain Semantic Consistency and Image Quality Evaluations. The best results are in bold, and the second best are underlined.

Semantic Consistency Image Quality UniGenBench T2I-CompBench GenEval CLIP PickScore UnifiedReward Aesthetic

Model

FLUX.1-dev 59.39 48.57 62.18 34.40 22.70 3.07 6.13 w/ HPSv2 57.77 44.84 58.43 33.35 23.12 3.10 6.23 w/ HPSv3 57.98 46.46 61.14 34.12 23.26 3.14 6.37 w/ PickScore 58.63 45.92 58.76 33.61 23.78 3.12 6.42 w/ UnifiedReward 60.87 49.13 66.25 34.43 23.31 3.19 6.44 w/ UnifiedReward-Think 68.89 50.10 68.20 35.85 23.38 3.27 6.53 w/ UnifiedReward-Flex 73.95 51.37 69.62 36.25 23.42 3.31 6.56

randomly drawn from the collected datasets, while the remaining data are reserved for the subsequent DPO stage. In the DPO stage, we use non-greedy decoding with temperature 0.7 and sample two reasoning traces per prompt. If both samples yield the correct preference decision, we further use GPT-5.2 to perform pairwise comparison of their reasoning trajectories and select the higher-quality trace as the preferred response for constructing DPO training pairs.

Reward model. We adopt UnifiedReward-Think-qwen3vl [2] as the base reward model, which supports long-chain reasoning for both visual perception and generation. Building on its strong prior knowledge, we further steer it toward flexible, context-adaptive reward reasoning. To evaluate robustness across model scales, UnifiedReward-Flex is trained with backbone sizes ranging from 2B∼32B. The 8B variant is used as the default reward model in all vision generation GRPO experiments, while other scales are included to analyze the effect of model capacity on reward assessment performance.

Training details. Our training is conducted with a batch size of 2 and 2 gradient accumulation steps, using a learning rate of 2.5 × 10−6 and a warm-up ratio of 0.1. All experiments are performed on 32 NVIDIA H200 GPUs. For the DPO stage, we set 𝛽𝑑𝑝𝑜 to 0.1.

Evaluation. We evaluate image reward models on GenAI-Bench-Image [39] and Multimodal RewardBench 2 (MMRB2) [40], and video reward models on GenAI-Bench-Video [39] and MJ-Bench-Video [41]. For pairwise preference-based reward models, we randomly permute the order of the two candidates at evaluation time.

##### 4.1.2 Reinforcement Learning for Vision Generation

Text-to-Image Generation. Training. We perform GRPO on FLUX.1-dev [42] using training prompts from UniGenBench++ [43]. Training is conducted on 32 NVIDIA H200 GPUs with 15 sampling steps and 9 rollouts per prompt from the same initial noise, using 3 gradient accumulation steps and a learning rate of 3 × 10−6. We set 𝛽KL = 0. Evaluation. For inference, we use 30 sampling steps and a classifier-free guidance scale of 3.5, following the official configuration. We evaluate in-domain performance on UniGenBench++. For out-of-domain evaluation, we measure semantic consistency on GenEval [44] and T2I-CompBench [45], and assess image quality using UnifiedReward [1], Pickscore [15], and the aesthetic predictor [14].

Text-to-Video Generation. Training. We perform GRPO on Wan2.1-T2V-14B [46] using the training prompts provided by [10]. Training is conducted on 32 NVIDIA H200 GPUs with LoRA rank 64 and alpha 128, using 20 sampling steps and 6 rollouts per prompt from the same initial noise. We train with videos at 240 × 416 resolution with 33 frames, 2 gradient accumulation steps, and a learning rate of 3 × 10−5. We set 𝛽KL = 0.004. Evaluation. For inference, we use 30 sampling steps and a classifier-free guidance scale of 5, generating videos at 480 × 832 resolution with 33 frames, and evaluating the performance on VBench [47].

UnifiedReward-Flex (Ours)

FLUX.1-dev HPSv2 HPSv3 PickScore UnifiedReward UnifiedReward-Think

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

The art deco music festival poster has a fox made of polished brass in the center of the picture, with smooth lines and elegant posture.

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

Please generate a surrealist photo of Newton sitting under an apple tree, a square apple falling from the tree, but its shadow is a perfect circle.

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

A fox with a body made of blue and white porcelain and a rabbit with a body made of terracotta warriors are racing on the Great Wall.

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

The magic book on the left shot out light as it was opened and hit a much larger crystal ball on the right. The words "The ancient prophecy awakens the sleeping giant" appeared on the ball.

Figure 3 Qualitative Comparison on Text-to-Image GRPO (FLUX.1-dev).

To assess UnifiedReward-Flex’s robustness across different generators, we additionally perform GRPO on FLUX.2-klein-base-9B [48] and Wan2.2-T2V-A14B [46] under the same experimental settings as above.

#### 4.2 Results

##### 4.2.1 Reward Model Comparison

We compare our UnifiedReward-Flex against representative baselines on both image and video preferenceevaluation benchmarks. Fixed scorers (e.g., HPSv2 [13], PickScore [15]) and Bradley–Terry preference models (e.g., HPSv3 [17], VideoReward [4]) provide a single global signal that is insensitive to prompt-specific requirements. Besides, UnifiedReward-Think [2] is a strong VLM-as-a-judge baseline that performs multimodal assessment with long-chain reasoning, but still follows a fixed checklist of evaluation criteria. In contrast, UnifiedReward-Flex dynamically instantiates fine-grained criteria conditioned on the prompt and visual evidence, yielding more reliable pairwise decisions. Quantitatively, as shown in Tab. 1, UnifiedRewardFlex achieves the best performance across all benchmarks; notably, it improves over UnifiedReward-Think by +3.2 points on MMRB2 and +2.2 points on GenAI-Bench-Video, highlighting the benefit of context-adaptive criterion composition beyond long-chain reasoning alone. Qualitative results are provided in Figs. 1 and 2. For example, Fig. 1 highlights our hierarchical, context-adaptive evaluation in a story-implied prompt (“a child healing a wounded kirin”). Starting from common anchor dimensions (semantic alignment, visual quality, and aesthetics), our model instantiates prompt-specific sub-criteria to check core requirements such as subject completeness and style coherence. Crucially, because the prompt implicitly describes a narrative moment rather than a static portrait, the model further augments the evaluation hierarchy with an additional high-level dimension, i.e., “Narrative & Interaction”, to judge whether the generation conveys a coherent healing scene with meaningful entity relations. This adaptive criterion composition prevents the evaluation from being dominated by surface-level rendering quality alone, and yields preference judgments that better reflect prompt-critical intent, providing more informative reward supervision for downstream optimization.

- Table 4 Quantitative results on VBench. The first seven metrics correspond to the Quality type, while the remaining correspond to the Semantic type.

Human Action Wan2.1-T2V-14B 96.6 97.6 62.4 64.9 99.2 98.5 58.6 79.4

Subject Consistency

Background Consistency

Aesthetic Quality

Imaging Quality

Temporal Flickering

Motion Smoothness

Dynamic Degree

Models

w/ VideoReward 96.7 97.9 62.9 66.5 99.3 98.5 41.6 78.2 w/ UnifiedReward-Think 96.4 97.7 63.9 65.2 99.4 98.4 58.3 78.4

###### w/ UnifiedReward-Flex 96.9 97.8 65.1 66.9 99.3 99.0 70.8 79.9

Appearance Style Wan2.1-T2V-14B 87.7 72.6 28.8 23.6 25.1 79.1 61.8 22.2

Spatial Relationship

Temporal Style

Overall Consistency

Object Class

Multiple Objects

Models Color

Scene

w/ VideoReward 87.8 77.0 28.2 23.7 25.3 82.1 70.2 21.0 w/ UnifiedReward-Think 86.1 77.3 27.2 23.8 25.4 78.4 63.0 22.3

###### w/ UnifiedReward-Flex 89.6 80.8 30.5 24.2 25.6 83.2 70.6 22.4

A fateful encounter during a fierce battle, a love scene between superhero and a human woman, a love romance similar to a movie commercial, 3D animation

Two AI models fighting in mortal kombat

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

Wan2.1T2V-14B

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

VideoReward

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

UnifiedReward Think

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

UnifiedReward Flex (Ours)

Human girl talk to cute dragon, pixar, disney Smart device projecting a hologram of a person talking

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

Wan2.1T2V-14B

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

VideoReward

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

UnifiedReward Think

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

UnifiedReward Flex (Ours)

Figure 4 Qualitative Comparison on Text-to-Video GRPO (Wan2.1-T2V-14B).

##### 4.2.2 Text-to-Image GRPO

Quantitatively, on UniGenBench (Tab. 2), UnifiedReward-Flex improves the overall semantic consistency (+14.56) over the base model, and also surpasses the strong VLM-as-a-judge baseline UnifiedReward-Think (+5.06). The improvements are broad-based across challenging dimensions that require compositional and intent-aware evaluation, such as Compound and Logical Reasoning. It also generalizes well to out-of-domain benchmarks (Tab. 3): achieves the best semantic consistency on T2I-CompBench [45] and GenEval [44], while maintaining or improving image quality metrics (e.g., UnifiedReward [1]) compared to other reward baselines. These results indicate that optimizing with our personalized reward does not merely overfit to in-domain prompts; instead, it promotes more robust alignment with diverse prompt intents and visual evidence.

- Qualitatively, Fig. 3 further shows that our model better enforces prompt-critical constraints. In the “Newton” example (row 2), the prompt couples multiple requirements (a square falling apple and a circular shadow), which demands reasoning over attributes and geometry. Among the compared methods, only the generator optimized with our reward consistently satisfies this coupled constraint.

Table 5 Hyperparameter Analysis of 𝛼. The best results are in bold, and the second best are underlined.

Text-to-Video Generation UniGenBench T2I-CompBench UnifiedReward Total Semantic Quality

Text-to-Image Generation

Model

Model

|FLUX.1-dev 59.39 48.57 3.07<br><br>𝛼 = 0 (w/o 𝑅¯dim) 71.13 50.32 3.25 𝛼 = 0.3 72.50 50.42 3.23 𝛼 = 0.5 73.10 50.90 3.29<br><br>𝛼 = 1 (w/o 𝑅overall) 73.44 51.59 3.26<br><br><br>|Wan2.1-T2V-14B 80.81 69.66 83.60<br><br>𝛼 = 0 (w/o 𝑅¯dim) 82.46 71.79 85.13<br><br>𝛼 = 0.3 82.56 72.11 85.17 𝛼 = 0.5 82.82 72.34 85.44<br><br>𝛼 = 1 (w/o 𝑅overall) 82.89 72.42 85.51<br><br><br>|
|---|---|
|𝛼 = 0.7 (Ours) 73.95 51.37 3.31|𝛼 = 0.7 (Ours) 83.08 72.94 85.62|

##### 4.2.3 Text-to-Video GRPO

Quantitatively, Tab. 4 summarizes VBench results for Wan2.1-T2V-14B GRPO with different reward signals. UnifiedReward-Flex yields clear improvements on both dynamic quality and compositional semantics. For example, on the quality side, it notably boosts the Dynamic Degree score (from 58.6 to 70.8), indicating that our reward provides more informative supervision for motion-intensive generations beyond static appearance. On the semantic side, it substantially improves Spatial Relationship (72.6 to 80.8) and Color consistency (87.7 to 89.6), suggesting better prompt grounding in relational and attribute-level constraints. Compared with VideoReward and UnifiedReward-Think, while both baselines also achieve competitive results, their improvements are less pronounced; in particular, they incur drops on Dynamic Degree, indicating limited ability to encourage richer motion. This contrast further supports that our context-personalized evaluation provides more effective reward supervision for GRPO-based video post-training.

- Qualitatively, Fig. 4 shows that using UnifiedReward-Flex as the reward for GRPO yields more coherent videos under motion- and interaction-centric prompts. In the upper-left example (“Two AI models fighting in Mortal Kombat”), the base Wan2.1-T2V-14B output is largely static, and GRPO with VideoReward or UnifiedReward-Think often further dampens the motion dynamics, resulting in reduced action amplitude and weaker interaction intensity across frames. In contrast, UnifiedReward-Flex-guided GRPO preserves stronger, continuous motion and clearer contact-driven progression, better matching the prompt’s intent.

#### 4.3 Ablation Studies

##### 4.3.1 Effect of DPO Alignment

As shown in Tab. 1, applying DPO yields consistent improvements, validating the role of preference alignment in sharpening preference discrimination. Importantly, the gains persist even in the “Both correct” setting, where both sampled traces predict the correct final preference. In this harder regime, DPO provides supervision on how the decision is reached: explicitly preferring higher-quality, more context-adaptive reasoning trajectories further improves discriminative accuracy by separating subtle quality differences that are invisible to decision-level correctness alone. Together, these results motivate coupling personalized criterion composition with reasoning-aware preference alignment to build robust reward models for both image and video generation.

##### 4.3.2 Hyperparameter Analysis of 𝛼

We analyze the impact of 𝛼 in Tab. 10, which controls the balance between the averaged dimension-wise win rate (𝑅¯dim) and the overall win rate (𝑅overall) in our reward model. When 𝛼 = 0, the model relies solely on the overall win rate, potentially overlooking finer, context-specific details, resulting in suboptimal performance. As 𝛼 increases, the model shifts focus towards dimension-wise rewards, capturing more context-adaptive nuances. However, when 𝛼 = 1, the model becomes overly focused on dimension-wise rewards, limiting its ability to assess broader, global preferences. Our results show that 𝛼 = 0.7 strikes an optimal balance, combining context-aware reasoning with global quality assessment, leading to the best performance across both image and video generation tasks. This further demonstrates the effectiveness of our personalized multi-dimensional rewards in enhancing the model’s preference alignment.

Table 6 Reward Assessment Performance across Model Scales.

Image Generation Video Generation GenAI-Bench MMRB2 GenAI-Bench MJBench

Model

UnifiedReward-Flex-2B 70.3 64.6 77.5 65.2 UnifiedReward-Flex-4B 72.1 68.5 80.2 67.8 UnifiedReward-Flex-8B 73.4 69.2 82.5 72.0

UnifiedReward-Flex-32B 74.8 69.9 82.8 71.3

###### Table 7 Robustness across Different Text-to-Image Generators on UniGenBench. “UniGenBench-EvalModel-qwen3vl32b-v1” is used as the VLM for evaluation.

Model Overall Style World Know. Attribute Action Relation. Compound Grammar Logic.Reason. Layout Text

- FLUX.1-dev 59.39 85.10 85.92 65.28 61.41 64.97 43.56 60.16 24.77 70.52 32.18 w/ UnifiedReward-Flex 73.95 90.30 89.87 79.38 73.38 78.55 69.46 63.10 46.59 79.66 59.20

- FLUX.2-klein-base-9B 78.93 97.50 91.61 83.65 77.00 86.42 78.61 76.87 53.41 88.43 55.75 w/ UnifiedReward-Flex 81.54 97.60 91.93 85.47 78.42 86.42 81.96 76.97 58.64 88.43 69.54

#### 4.4 Discussion

##### 4.4.1 Reward Assessment Performance across Model Scales

Tab. 6 reports the assessment performance of UnifiedReward-Flex across different model scales. We observe a consistent improvement as model capacity increases, on both image and video generation benchmarks, indicating that UnifiedReward-Flex benefits from additional representational and reasoning capacity while preserving stable behavior across scales. Notably, the gains are smooth rather than abrupt: smaller models already achieve competitive performance, while larger models provide incremental improvements. This suggests that the core advantage of our model does not stem solely from model size, but from its context-adaptive evaluation mechanism, which remains effective even under limited capacity.

FLUX.2-klein-base-9B

w/ UnifiedReward-Flex

FLUX.2-klein-base-9B

w/ UnifiedReward-Flex

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

The magic book on the left shot out light as it was opened and hit a much larger crystal ball on the right. The words "The ancient prophecy awakens the sleeping giant" appeared on the ball.

Generate a picture- an ancient stone bridge spans a glowing stream, with the end of the bridge leading to a mysterious forest, in a fantasy illustration style.

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

In the ruins of a future city, a robot looks sadly at the sky, where a brand new spacecraft flies higher than the dilapidated spacecraft next to it.

Below a huge flying castle made of crystals, a knight in armor was trying to climb hanging vines and try to enter the castle.

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

Generating picture- A giant chess piece made of broken crystals and flowing lava is suspended in the air surrounded by glowing runes.

On the Great Wall, a huge panda is wearing a spacesuit, with a small butterfly made of a nebula lying on its helmet.

Figure 5 Qualitative Results on Text-to-Image GRPO (FLUX.2-klein-base-9B).

##### 4.4.2 Training Efficiency

Tab. 10 compares the per-step training time across different reward models. Fixed scorers and BradleyTerry–style models (e.g., PickScore, HPSv3) are the most efficient, as they directly output scalar rewards without explicit reasoning. However, this efficiency comes at the cost of coarse and less comprehensive reward

###### Table 8 Robustness across Different Text-to-Image Generators on Semantic Consistency and Image Quality. The best results are in bold, and the second best are underlined.

Semantic Consistency Image Quality UniGenBench T2I-CompBench GenEval CLIP PickScore UnifiedReward Aesthetic

Model

- FLUX.1-dev 59.39 48.57 62.18 34.40 22.70 3.07 6.13 w/ UnifiedReward-Flex 73.95 51.37 69.62 36.25 23.42 3.31 6.56

- FLUX.2-klein-base-9B 78.93 53.72 78.99 35.59 22.51 3.81 5.89 w/ UnifiedReward-Flex 81.54 58.62 81.22 36.42 23.07 3.98 6.06

###### Table 9 Robustness across Different Text-to-Video Generators on VBench. The first seven metrics correspond to the Quality type, while the remaining correspond to the Semantic type.

Subject Consistency

Background Consistency

Aesthetic Quality

Imaging Quality

Temporal Flickering

Motion Smoothness

Dynamic Degree

Human Action

Models

- Wan2.1-T2V-14B 96.6 97.6 62.4 64.9 99.2 98.5 58.6 79.4

w/ UnifiedReward-Flex 96.9 97.8 65.1 66.9 99.3 99.0 70.8 79.9

- Wan2.2-T2V-A14B 94.8 95.7 63.9 65.8 98.8 97.0 80.0 82.0

###### w/ UnifiedReward-Flex 94.9 95.9 64.7 66.9 98.8 97.1 80.5 84.6

Spatial Relationship

Temporal Style

Overall Consistency

Object Class

Multiple Objects

Appearance Style

Models Color

Scene

- Wan2.1-T2V-14B 87.7 72.6 28.8 23.6 25.1 79.1 61.8 22.2

- w/ UnifiedReward-Flex 89.6 80.8 30.5 24.2 25.6 83.2 70.6 22.4

Wan2.2-T2V-A14B 85.8 77.2 30.2 23.4 25.1 80.3 67.0 21.4

- w/ UnifiedReward-Flex 90.4 81.1 32.7 23.7 25.5 84.9 68.3 21.7

tom cruise iron man putting on the armor A cat wearing sunglasses and working as a lifeguard at a pool

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

Wan2.2T2V-A14B

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

UnifiedReward Flex (Ours)

a magical girl 20 year old turn into a anti hero Scarlet Witch fighting a Dragon in Skyrim

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

Wan2.2T2V-A14B

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

UnifiedReward Flex (Ours)

Figure 6 Qualitative Results on Text-to-Video GRPO (Wan2.2-T2V-A14B).

signals with limited expressiveness. Reasoning-based VLMs introduce additional overhead. Specifically, UnifiedReward-Think is noticeably slower, since it performs long-chain reasoning under a fixed evaluation rubric to produce judgments. Our UnifiedReward-Flex further increases the computation cost, as it goes beyond fixed criteria and conducts context-personalized reasoning: it dynamically instantiates fine-grained sub-criteria and, when necessary, introduces new high-level dimensions conditioned on the prompt intent and visual evidence, rather than applying a static checklist. Despite this extra compute, UnifiedReward-Flex delivers substantially stronger reward accuracy (Tab. 1) and consistently larger gains in GRPO for both image and video generation (Tabs. 3 and 4). These results indicate that the added training cost is a reasonable and practical trade-off for improved alignment quality and generation performance.

##### 4.4.3 Training Progress Visualization

Quantitatively. Fig. 8 visualizes the training dynamics when using CLIP score as a proxy metric to monitor semantic alignment during GRPO with UnifiedReward-Flex. Across both text-to-image (FLUX.1-dev) and text-to-video (Wan2.1-T2V-14B) settings, the CLIP score exhibits a clear and consistent upward trend over training steps, indicating steadily improved prompt-image/video semantic consistency. The similarly stable increase in both domains suggests that our context-adaptive reward provides reliable optimization signals,

###### Table 10 Comparison of Training Efficiency (Seconds per Step).

PickScore HPSv3 UnifiedReward VideoReward UnifiedReward-Think UnifiedReward-Flex

FLUX.1-dev 102s 103s 109s — 124s 143s Wan2.1-T2V-14B — — — 285s 328s 336s

w/ UnifiedReward-Flex Step 50

w/ UnifiedReward-Flex Step 100

w/ UnifiedReward-Flex Step 200

FLUX.1-dev

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

A clever fox opened the treasure chest in front of him because he had solved a complex mystery. The inside of the lid of the box was engraved with the words "Wisdom is the key to all hidden treasures".

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

An anthropomorphic sculpture of Mr. Moon is watering the stars. The work says "We are all made of stardust and dreams" in a clay style.

###### Figure 7 Qualitative Results of Text-to-Video Generation during Training Progress.

###### (a) FLUX.1-dev

###### (b) Wan2.1-T2V-14B

w/ UnifiedReward-Flex

w/ UnifiedReward-Flex

[Figure 182]

[Figure 183]

0.295

0.355

CLIP Score

0.290

CLIP Score

0.350

0.285

0.345

0 50 100 150 200

0 50 100 150 200

Training Steps

Training Steps

Figure 8 Monitoring Training Progress via CLIP Score.

reinforcing semantic understanding for both static image synthesis and temporal video generation.

Qualitatively. Figs. 7 and 9 show a consistent visual improvement trend during GRPO when using UnifiedReward-Flex as the reward. For image generation (Fig. 7), later checkpoints better enforce promptcritical constraints and compositional intent (e.g., the chest engraving becomes clearer and more faithful), indicating stronger semantic grounding beyond surface aesthetics. For video generation (Fig. 9), the origamidancer example evolves from limited, less coordinated motion to smoother, more coherent action progression with improved temporal consistency across frames. Overall, these training-time visualizations corroborate that UnifiedReward-Flex provides reliable, context-adaptive reward guidance that steadily strengthens both semantic adherence and temporal coherence during training.

Wan2.1-T2V-14B

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

w/ UnifiedReward-Flex Step 100

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

w/ UnifiedReward-Flex Step 200

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

Origami dancers in white paper, 3D render, on white background, studio shot, dancing modern dance

Figure 9 Qualitative Results of Text-to-Video Generation during Training Progress.

##### 4.4.4 Robustness across Different Vision Generators

To evaluate the robustness of UnifiedReward-Flex, we apply it as the reward signal for GRPO across multiple image and video generators with diverse architectures and capabilities. Text-to-Video Generation (Tab. 9). Across both Wan2.1-T2V-14B and the stronger Wan2.2-T2V-A14B, UnifiedReward-Flex consistently improves motion- and interaction-related metrics, including Dynamic Degree, Motion Smoothness, and Human Action. These gains indicate that the proposed reward encourages richer and more sustained motion patterns, rather than static or over-smoothed trajectories often induced by less expressive rewards. Importantly, improvements are also observed on semantic metrics such as Spatial Relationship and Multiple Objects, suggesting that enhanced motion fidelity does not come at the expense of semantic structure, but instead reinforces both jointly. Text-to-Image Generation (Tabs. 7 and 8). Similar robustness is observed across FLUX.1-dev and the stronger FLUX.2-klein-base-9B. UnifiedReward-Flex consistently improves semantic consistency across multiple benchmarks, while also yielding gains in image quality under different evaluation models. This indicates that the learned reward generalizes well beyond a specific generator or evaluation setup. Overall, the consistent gains suggest that UnifiedReward-Flex does not rely on the specific generator. Instead, by dynamically instantiating evaluation criteria conditioned on prompt intent and visual evidence, it produces preference signals that remain effective across diverse output distributions. This generator-agnostic behavior highlights the robustness of context-adaptive reward modeling for vision generation post-training.

### 5 Conclusion

This paper introduces UnifiedReward-Flex, a unified personalized reward model that circumvents the limitations of traditional “one-size-fits-all” evaluation in visual generation. By coupling dynamic hierarchical assessment with context-adaptive reasoning, our approach transcends rigid rubrics to capture the nuanced and subjective nature of human preferences. Leveraging a two-stage training pipeline, i.e., structured reasoning distillation and reasoning-aware Direct Preference Optimization (DPO), we demonstrate that UnifiedReward-Flex yields significantly more precise and context-sensitive reward signals. Empirical validation within the Group Relative Policy Optimization (GRPO) framework reveals that our method consistently outperforms existing baselines, achieving significant improvements in both visual fidelity and semantic alignment for image and video synthesis.

### References

- [1] Yibin Wang, Yuhang Zang, Hao Li, Cheng Jin, and Jiaqi Wang. Unified reward model for multimodal understanding and generation. arXiv preprint arXiv:2503.05236, 2025.

- [2] Yibin Wang, Zhimin Li, Yuhang Zang, Chunyu Wang, Qinglin Lu, Cheng Jin, and Jiaqi Wang. Unified multimodal chain-of-thought reward model through reinforcement fine-tuning. arXiv preprint arXiv:2505.03318, 2025.

- [3] Yibin Wang, Zhiyu Tan, Junyan Wang, Xiaomeng Yang, Cheng Jin, and Hao Li. Lift: Leveraging human feedback for text-to-video model alignment. arXiv preprint arXiv:2412.04814, 2024.

- [4] Jie Liu, Gongye Liu, Jiajun Liang, Ziyang Yuan, Xiaokun Liu, Mingwu Zheng, Xiele Wu, Qiulin Wang, Wenyu Qin, Menghan Xia, et al. Improving video generation with human feedback. arXiv preprint arXiv:2501.13918, 2025.

- [5] Yuhang Zang, Xiaoyi Dong, Pan Zhang, Yuhang Cao, Ziyu Liu, Shengyuan Ding, Shenxi Wu, Yubo Ma, Haodong Duan, Wenwei Zhang, et al. Internlm-xcomposer2.5-reward: A simple yet effective multi-modal reward model. arXiv preprint arXiv:2501.12368, 2025.

- [6] Yibin Wang, Zhimin Li, Yuhang Zang, Yujie Zhou, Jiazi Bu, Chunyu Wang, Qinglin Lu, Cheng Jin, and Jiaqi Wang. Pref-grpo: Pairwise preference reward-based grpo for stable text-to-image reinforcement learning. arXiv preprint arXiv:2508.20751, 2025.

- [7] Yujie Zhou, Pengyang Ling, Jiazi Bu, Yibin Wang, Yuhang Zang, Jiaqi Wang, Li Niu, and Guangtao Zhai. G2rpo: Granular grpo for precise reward in flow models. arXiv preprint arXiv:2510.01982, 3, 2025.

- [8] Jie Wu, Yu Gao, Zilyu Ye, Ming Li, Liang Li, Hanzhong Guo, Jie Liu, Zeyue Xue, Xiaoxia Hou, Wei Liu, et al. Rewarddance: Reward scaling in visual generation. arXiv preprint arXiv:2509.08826, 2025.

- [9] Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Wanli Ouyang. Flow-grpo: Training flow matching models via online rl. arXiv preprint arXiv:2505.05470, 2025.

- [10] Zeyue Xue, Jie Wu, Yu Gao, Fangyuan Kong, Lingting Zhu, Mengzhao Chen, Zhiheng Liu, Wei Liu, Qiushan Guo, Weilin Huang, et al. Dancegrpo: Unleashing grpo on visual generation. arXiv preprint arXiv:2505.07818, 2025.

- [11] Junzhe Li, Yutao Cui, Tao Huang, Yinping Ma, Chun Fan, Miles Yang, and Zhao Zhong. Mixgrpo: Unlocking flow-based grpo efficiency with mixed ode-sde. arXiv preprint arXiv:2507.21802, 2025.

- [12] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, pages 8748–8763, 2021.

- [13] Xiaoshi Wu, Yiming Hao, Keqiang Sun, Yixiong Chen, Feng Zhu, Rui Zhao, and Hongsheng Li. Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis. arXiv preprint arXiv:2306.09341, 2023.

- [14] Chrisoph Schuhmann. Laion aesthetics. https://github.com/LAION-AI/aesthetic-predictor, 2022.

- [15] Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, and Omer Levy. Pick-a-pic: An open dataset of user preferences for text-to-image generation. NeurIPS, 36:36652–36663, 2023.

- [16] Ralph Allan Bradley and Milton E Terry. Rank analysis of incomplete block designs: I. the method of paired comparisons. Biometrika, 39(3/4):324–345, 1952.

- [17] Yuhang Ma, Xiaoshi Wu, Keqiang Sun, and Hongsheng Li. Hpsv3: Towards wide-spectrum human preference score. In ICCV, pages 15086–15095, 2025.

- [18] Xuan He, Dongfu Jiang, Ping Nie, Minghao Liu, Zhengxuan Jiang, Mingyi Su, Wentao Ma, Junru Lin, Chun Ye, Yi Lu, et al. Videoscore2: Think before you score in generative video evaluation. arXiv preprint arXiv:2509.22799, 2025.

- [19] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shĳie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.

- [20] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024.

- [21] OpenAI. Update to GPT-5 system card: GPT-5.2. OpenAI System Card, 2025. URL https://cdn.openai.com/ pdf/3a4153c8-c748-4b71-8e31-aecbde944f8d/oai_5_2_system-card.pdf.

- [22] Qunzhong Wang, Jie Liu, Jiajun Liang, Yilei Jiang, Yuanxing Zhang, Jinyuan Chen, Yaozhi Zheng, Xintao Wang, Pengfei Wan, Xiangyu Yue, et al. Vr-thinker: Boosting video reward models through thinking-with-image reasoning. arXiv preprint arXiv:2510.10518, 2025.

- [23] Xiyao Wang, Chunyuan Li, Jianwei Yang, Kai Zhang, Bo Liu, Tianyi Xiong, and Furong Huang. Llava-critic-r1: Your critic model is secretly a strong policy model. arXiv preprint arXiv:2509.00676, 2025.

- [24] Tianyi Xiong, Xiyao Wang, Dong Guo, Qinghao Ye, Haoqi Fan, Quanquan Gu, Heng Huang, and Chunyuan Li. Llava-critic: Learning to evaluate multimodal models. In CVPR, pages 13618–13628, 2025.

- [25] Tianyi Xiong, Yi Ge, Ming Li, Zuolong Zhang, Pranav Kulkarni, Kaishen Wang, Qi He, Zeying Zhu, Chenxi Liu, Ruibo Chen, et al. Multi-crit: Benchmarking multimodal judges on pluralistic criteria-following. arXiv preprint arXiv:2511.21662, 2025.

- [26] Kevin Clark, Paul Vicol, Kevin Swersky, and David J Fleet. Directly fine-tuning diffusion models on differentiable rewards. arXiv preprint arXiv:2309.17400, 2023.

- [27] Mihir Prabhudesai, Russell Mendonca, Zheyang Qin, Katerina Fragkiadaki, and Deepak Pathak. Video diffusion alignment via reward gradients. arXiv preprint arXiv:2407.08737, 2024.

- [28] Kimin Lee, Hao Liu, Moonkyung Ryu, Olivia Watkins, Yuqing Du, Craig Boutilier, Pieter Abbeel, Mohammad Ghavamzadeh, and Shixiang Shane Gu. Aligning text-to-image models using human feedback. arXiv preprint arXiv:2302.12192, 2023.

- [29] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

- [30] Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. Training diffusion models with reinforcement learning. arXiv preprint arXiv:2305.13301, 2023.

- [31] Ying Fan, Olivia Watkins, Yuqing Du, Hao Liu, Moonkyung Ryu, Craig Boutilier, Pieter Abbeel, Mohammad Ghavamzadeh, Kangwook Lee, and Kimin Lee. Reinforcement learning for fine-tuning text-to-image diffusion models. In NeurIPS, 2023.

- [32] Zichen Miao, Jiang Wang, Ze Wang, Zhengyuan Yang, Lĳuan Wang, Qiang Qiu, and Zicheng Liu. Training diffusion models towards diverse image generation with reinforcement learning. In CVPR, pages 10844–10853, 2024.

- [33] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. NeurIPS, 36:53728–53741, 2023.

- [34] Bram Wallace, Meihua Dang, Rafael Rafailov, Linqi Zhou, Aaron Lou, Senthil Purushwalkam, Stefano Ermon, Caiming Xiong, Shafiq Joty, and Nikhil Naik. Diffusion model alignment using direct preference optimization. In CVPR, pages 8228–8238, 2024.

- [35] Kai Yang, Jian Tao, Jiafei Lyu, Chunjiang Ge, Jiaxin Chen, Weihan Shen, Xiaolong Zhu, and Xiu Li. Using human feedback to fine-tune diffusion models without any reward model. In CVPR, pages 8941–8951, 2024.

- [36] Runtao Liu, Haoyu Wu, Ziqiang Zheng, Chen Wei, Yingqing He, Renjie Pi, and Qifeng Chen. Videodpo: Omnipreference alignment for video diffusion generation. In CVPR, pages 8009–8019, 2025.

- [37] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

- [38] Haoyou Deng, Keyu Yan, Chaojie Mao, Xiang Wang, Yu Liu, Changxin Gao, and Nong Sang. Densegrpo: From sparse to dense reward for flow matching model alignment. arXiv preprint arXiv:2601.20218, 2026.

- [39] Dongfu Jiang, Max Ku, Tianle Li, Yuansheng Ni, Shizhuo Sun, Rongqi Fan, and Wenhu Chen. Genai arena: An open evaluation platform for generative models. arXiv preprint arXiv:2406.04485, 2024.

- [40] Yushi Hu, Reyhane Askari-Hemmat, Melissa Hall, Emily Dinan, Luke Zettlemoyer, and Marjan Ghazvininejad. Multimodal rewardbench 2: Evaluating omni reward models for interleaved text and image. arXiv preprint arXiv:2512.16899, 2025.

- [41] Haibo Tong, Zhaoyang Wang, Zhaorun Chen, Haonian Ji, Shi Qiu, Siwei Han, Kexin Geng, Zhongkai Xue, Yiyang Zhou, Peng Xia, Mingyu Ding, Rafael Rafailov, Chelsea Finn, and Huaxiu Yao. Mj-video: Fine-grained benchmarking and rewarding video preferences in video generation, 2025. URL https://arxiv.org/abs/2502.01719.

- [42] Black Forest Labs. Flux. https://github.com/black-forest-labs/flux, 2024.
- [43] Yibin Wang, Zhimin Li, Yuhang Zang, Jiazi Bu, Yujie Zhou, Yi Xin, Junjun He, Chunyu Wang, Qinglin Lu, Cheng Jin, et al. Unigenbench++: A unified semantic evaluation benchmark for text-to-image generation. arXiv preprint arXiv:2510.18701, 2025.

- [44] Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating text-to-image alignment. NeurIPS, 36:52132–52152, 2023.

- [45] Kaiyi Huang, Kaiyue Sun, Enze Xie, Zhenguo Li, and Xihui Liu. T2i-compbench: A comprehensive benchmark for open-world compositional text-to-image generation. NeurIPS, 36:78723–78747, 2023.

- [46] Team Wan et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

- [47] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. VBench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

- [48] Black Forest Labs. FLUX.2: Frontier Visual Intelligence. https://bfl.ai/blog/flux-2, 2025.

## Appendix

[Figure 196]

###### A. Semantic Alignment & Accuracy

Image#1 Image#2

- 1. Spatial Composition & Framing
- 2. Facial Expression Fidelity

[Figure 197]

[Figure 198]

- Image#1: The image correctly positions the subject on the left side of the frame

as explicitly requested. (Score: 9)

- Image#2: The image places the subject on the right side of the frame, directly

contradicting the prompt's instruction to frame on the left. (Score: 1)

- Image#1: The subject displays a clear, gentle smile, aligning with the instruction to

'accentuate her smile'. (Score: 8)

- Image#2: The subject has a neutral, somewhat melancholic expression with a closed

[Figure 199]

mouth, missing the required smile. (Score: 2)

###### 3. Key Object & Scene Elements

A 22-year-old woman in a suit standing in front of an electronic bulletin board listing stock issues. Ultra-realistic digital photography. Style: Inspired by Al Bello's energetic and vibrant sports photography. The sun's rays give her face a soft, even glow and accentuate her smile. Composition: Nikon Z9 camera, Z 85mm f/1.4 S lens, 20.1 megapixel resolution, ISO 400, shutter speed 1/250 sec, aperture f/2.8. The subject is framed on the left, captured from the waist up, with a stylish cafe in the background blurred out.

UnifiedReward Flex

- Image#1: Includes the electronic board and a blurred background, though the cafe

context is vague. (Score: 5)

- Image#2: Includes the electronic board and a blurred urban background; neither

image explicitly renders a distinct 'cafe' clearly, making them equal here. (Score: 5)

Category Reason: Image 1 followed the crucial spatial instructions and expression details that Image 2 ignored.

[Figure 200]

Category Winner: Image 1

[Figure 201]

[Figure 202]

###### B. Image Quality & Realism

###### C. Aesthetics & Artistry

- 1. Stylistic Vibrancy
- 2. Compositional Balance

1. Lighting Specification Adherence

- Image#1: The image has a warmer, brighter tone that aligns better with the

'energetic and vibrant' inspiration cited in the prompt. (Score: 7)

- Image#2: The color palette is cooler and moodier, lacking the requested vibrancy.

- Image#1: Successfully renders the 'soft, even glow' on the face caused by

the sun's rays. (Score: 8)

- Image#2: Utilizes strong backlighting/rim lighting, leaving the face largely

(Score: 3)

in shadow, which contradicts the 'soft, even glow' requirement. (Score: 2)

2. Skin Texture & Detail

- Image#1: While accurate to the prompt, the framing feels slightly cramped with

the LED board very close to the subject's face. (Score: 4)

- Image#2: Despite being on the 'wrong' side spatially, the internal balance and

- Image#1: Skin texture is good but slightly overly smooth/polished. (Score: 4)
- Image#2: Displays excellent high-frequency detail in the skin, appearing

slightly more raw and realistic. (Score: 6)

negative space usage are artistically superior. (Score: 6)

###### 3. Lens Simulation (Bokeh)

###### 3. Overall Portrait Appeal

- Image#1: Correctly simulates the shallow depth of field (f/2.8 style) with a

blurred background. (Score: 5)

- Image#2: Also achieves a high-quality optical blur typical of an 85mm lens

- Image#1: The combination of lighting and expression creates a more engaging and

prompt-accurate portrait. (Score: 7)

- Image#2: While artistic, the somber expression and shadowed face create a disconnect

portrait. (Score: 5)

from the prompt's intended energetic persona. (Score: 3) Category Reason: Image 1 captures the requested 'vibrant' vibe better than the moody atmosphere of Image 2.

Category Reason: Image 1 better captured the specific lighting quality described in the text, whereas Image 2 prioritized a different lighting style.

[Figure 203]

[Figure 204]

Category Winner: Image 1

Category Winner: Image 1

Overall Reason: Image 1 is the clear winner primarily due to its strict adherence to the spatial and expression constraints defined in the prompt. It correctly places the subject on the left side of the frame and depicts the requested smile and specific lighting ('soft, even glow'). Image 2, while visually high-quality, fails significantly on these structural requirements, placing the subject on the right with a neutral expression and shadowed face.

[Figure 205]

###### Overall Winner: Image 1

- Figure 10 More Qualitative Result of UnifiedReward-Flex on Image Generation Personalized Reward Reasoning.

[Figure 206]

###### A. Semantic Alignment & Accuracy

- 1. Camera Constraint Adherence (Static Shot)
- 2. Primary Action Execution (Pouring Physics)

Video#1

[Figure 207]

- Video#1: Video 1 failed the specific instruction 'camera remains still' by utilizing a noticeable tilt and pan to follow the kettle. (Score: 1)
- Video#2: Video 2 followed the prompt strictly, maintaining a completely static locked-off shot throughout the duration. (Score: 9)

[Figure 208]

[Figure 209]

[Figure 210]

- Video#1: Video 1 depicts the action of pouring with flawless physical accuracy; the water stream has correct speed, transparency, and turbulence. (Score: 9)
- Video#2: Video 2 generates a water stream that appears too viscous and lacks realistic transparency, resembling a gel rather than water. (Score: 1)

Video#2

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

###### 3. Subject & Environment Attributes

[Figure 215]

- Video#1: Video 1 renders the 'blue blouse' with high-fidelity fabric details (sheer sleeves) and a highly realistic modern kitchen background. (Score: 6)
- Video#2: Video 2 correctly depicts the subject and setting, but the textures (clothing and background appliances) appear more generic and plasticky. (Score: 4)

UnifiedReward Flex

###### 4. Subject Anatomical Logic

The camera remains still, a woman with shoulder-length blonde hair and wearing a blue blouse pours water out of a kettle, the background is a modern kitchen, warm indoor lighting.

- Video#1: The subject's hands and grip on the kettle in Video 1 are anatomically

perfect and handle the weight of the object naturally. (Score: 7)

- Video#2: Video 2 shows a slightly stiff grip, and the wrist movement during the

tilt feels mechanical and linear. (Score: 3)

Category Reason: Although Video 2 followed the negative constraint regarding camera movement better, Video 1 delivered a much more accurate and convincing representation of the primary action (pouring water) and environmental details.

[Figure 216]

###### Category Winner: Video 1

[Figure 217]

[Figure 218]

B. Video Quality & Dynamic Realism

###### C. Narrative, Aesthetics & Cinematography

1. Lighting Quality & Atmosphere

- 1. Fluid Dynamics & Interaction
- 2. Visual Texture & Photorealism

- Video#1: Video 1 uses sophisticated lighting with deep shadows and warm highlights

that accurately reflect the 'warm indoor lighting' prompt. (Score: 8)

- Video#2: Video 2 utilizes flat, even lighting that feels artificial and lacks the

- Video#1: The water simulation is indistinguishable from real footage,

exhibiting proper splashes, flow rate, and refraction. (Score: 10)

- Video#2: The fluid physics are the weakest point; the water appears out of

atmospheric depth of Video 1. (Score: 2)

thin air or lacks proper volume conservation and behaves unnaturally. (Score: 0)

2. Composition & Framing

- Video#1: Video 1 exhibits realistic skin texture, natural lighting falloff, and

authentic metallic reflections on the kettle. (Score: 9)

- Video#2: Video 2 has the characteristic 'smoothness' of AI generation, with

- Video#1: The framing in Video 1, though moving, dynamically focuses attention on

the interaction between the subject and the action. (Score: 7)

- Video#2: Video 2 uses a standard medium shot that, while functional, feels less

skin and objects lacking fine-grain surface imperfections. (Score: 1)

composed and artistic. (Score: 3)

###### 3. Motion Naturalness (Human)

###### 3. Overall Visual Appeal

- Video#1: The subject in Video 1 displays natural micro-movements (breathing,

slight swaying) that contribute to a lifelike presence. (Score: 8)

- Video#2: The subject in Video 2 is rigidly frozen except for the specific limb

- Video#1: Video 1 looks like high-end cinematic footage. (Score: 9)
- Video#2: Video 2 looks like a demonstrative AI video clip. (Score: 1)

moving the kettle, creating an 'uncanny valley' effect. (Score: 2)

Category Reason: Video 1 dominates this category with near-photorealistic quality. Its handling of complex textures (skin, metal, water) and light interaction is significantly ahead of Video 2. Category Winner: Video 1

Category Reason: Video 1 presents a cinematic, professionally lit scene with depth and atmosphere. Video 2 is clean but looks like a standard stock generation with flatter lighting.

[Figure 219]

[Figure 220]

Category Winner: Video 1

Overall Reason: Video 1 is vastly superior in terms of photorealism, fluid dynamics, and natural human motion. While Video 2 adhered strictly to the 'camera remains still' constraint (which Video 1 violated with a tilt/pan), Video 1's execution of the core 'pouring water' action features perfect fluid physics, transparency, and interaction, whereas Video 2 displays noticeable artificial viscosity. Video 1 appears visually indistinguishable from reality, while Video 2 retains a distinct AIgenerated aesthetic.

[Figure 221]

Overall Winner: Video 1

###### Figure 11 More Qualitative Result of UnifiedReward-Flex on Video Generation Personalized Reward Reasoning.

