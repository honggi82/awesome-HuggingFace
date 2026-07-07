## Exploring Conditions for Diffusion Models in Robotic Control

Heeseong Shin1∗ Byeongho Heo2 Dongyoon Han2 Seungryong Kim1† Taekyung Kim2† 1KAIST AI 2NAVER AI Lab https://orca-rc.github.io/

|92.5<br><br>90.4<br><br>null text ours<br><br>|
|---|
|96.3<br><br>68.3|
|68.9|
|74.3|
|64.7<br><br>57.3<br>58<br>|

# arXiv:2510.15510v2[cs.CV]8Apr2026

MetaWorldAdroitDMC

“”

Condition

null prompt

[Figure 1]

“a 3d model of a stick figure walking on...”

Text-to-Image Diffusion Model

Policy

text prompt

Action

Observation

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

task and visual prompt

Figure 1. How can we condition diffusion models for robotic control? We investigate methods for conditioning text-to-image diffusion models [48] to perform robotic control, aiming to address various tasks in a task-adaptive manner. We observe that text prompts, unlike in other vision tasks [69], are ineffective for robotic control. Therefore, we propose to learn task prompts in control environments and further incorporate dynamic details through visual prompts for conditioning diffusion models.

### 1. Introduction

### Abstract

Recent advances in diffusion models [17, 48] have not only facilitated high-quality image synthesis, but also demonstrated as a strong visual representation for various vision tasks [1, 56, 62]. Among them, pre-trained text-to-image diffusion models, e.g. Stable Diffusion [48], have shown that utilizing text conditions can significantly boost performance in visual perception tasks, without the need for finetuning the model [69]. The key to leveraging text conditions lies in obtaining well-designed prompts [32]—often describing objects in the image or the given task—that can funnel useful information into downstream tasks. This not only enhances the proficiency of diffusion models on downstream tasks but also broadens their applicability to a wider variety of vision tasks [61, 65].

While pre-trained visual representations have significantly advanced imitation learning, they are often task-agnostic as they remain frozen during policy learning. In this work, we explore conditions for pre-trained text-to-image diffusion models to obtain task-adaptive visual representations for robotic control, without fine-tuning the model itself. We find that naively applying textual conditions—a successful strategy in other vision domains—yields minimal or even negative gains in control tasks. We attribute this to the domain gap between the diffusion model’s training data and robotic control environments, leading us to argue for conditions that consider the specific visual information required for control. To this end, we propose ORCA for leveraging diffusion models with conditions in robotic control in a taskadaptive manner, which introduces learnable task prompts that adapt to the control environment and visual prompts that capture fine-grained, frame-specific details. Through facilitating task-adaptive representation, ORCA achieves state-of-the-art on various robotic control benchmarks.

Robotic control, meanwhile, has also benefited greatly from the introduction of pre-trained visual representations to imitation learning [42]. By leveraging frozen visual encoders pre-trained on large-scale data, these representations have replaced the previous tabula-rasa paradigm of training vision encoders from scratch on limited-scale control data. However, this approach is limited by its task-agnostic nature, as the visual representations remain frozen during downstream policy learning. Since the suitability of a rep-

∗Work done during an internship at NAVER AI Lab. † Corresponding authors.

resentation for a specific task is unknown beforehand, determining which representation performs best often requires manual, task-by-task inspection [37], which becomes cumbersome given the vast variety of control tasks. While a straightforward solution might be to fine-tune the vision encoder, this often results in poor results as the model loses generalization capabilities by overfitting to specific scenes in imitation learning [14, 37, 42].

In this work, we explore bridging pre-trained text-toimage diffusion models to robotic control for achieving task-adaptive visual representations through conditions, without fine-tuning the diffusion model. Inspired by the effectiveness of conditions in visual perception tasks, we ask the following question: How can we effectively implement conditions for diffusion models in robotic control? We begin by investigating textual conditions, generating captions with a state-of-the-art vision-language model [7] to observe their impact on control task performance. However, as shown in Fig. 1, the gains are minimal, and in some cases, performance even declines. This result contrasts sharply with findings in other vision tasks [69], where machinegenerated captions have served as strong conditions [32].

Upon investigation, we find that pre-trained diffusion models often struggle to accurately associate text conditions to the image in control environments. We attribute this discrepancy to the nature of the diffusion model being trained on web-collected images, which suits visual tasks that involve real-world images and common objects, such as semantic segmentation [32, 69]. However, control environments, featuring specialized robotic agents performing specific tasks, would require a more careful and deliberate approach to devising effective conditions for downstream policy learning.

Robot control further complicates this challenge, as these tasks operate on dynamic video streams and require a finer visual granularity to interact with specific parts of objects, not just to categorize them. This dynamic nature implies that effective conditions must be generated uniquely for each frame [21] to guide evolving actions and adapt to changing visual states. Consequently, we hypothesize that conditions for control should incorporate visual information from every frame to capture both dynamic behavior and fine-grained details.

These observations suggest that conditions for control should be task-grounded and frame-sensitive. To this end, we propose a simple, yet effective method that incorporates visual information while addressing the limitations of text conditions. Specifically, we replace the text prompt with learnable task prompts, which are learned during downstream control tasks to ensure accurate grounding within the specific environment. Furthermore, to enable the conditions to capture the detailed visual state of each frame, we employ a vision encoder and utilize its representations as

visual prompts. We demonstrate that both the task and visual prompts can be learned end-to-end during downstream policy learning using a standard behavior cloning objective.

Our framework for leveraging diffusion models with conditions in robotic control in a task-adaptive manner, ORCA, achieves state-of-the-art performance in robotic control tasks [47, 57, 66], surpassing VC-1 [37]. We verify our design choices by comparing to baselines with text conditions and different conditioning methods [32, 70] from visual perception tasks. In addition, we provide detailed analysis and ablations on our approach, highlighting the importance of conditions in diffusion models for robotic control.

### 2. Related Work

Pre-trained visual representations for robotic control. In recent years, visual representations derived from selfsupervised pre-trained models [3, 4, 16, 28–30, 37, 45] have demonstrated notable effectiveness in visuo-motor manipulation tasks [42]. Specifically, Parisi et al. [42] showed that visual representations from frozen pre-trained encoders, such as MoCo [15] and CLIP [45], can not only outperform representations trained from scratch but are also comparable to ground-truth state features in behavior cloning. This finding has motivated subsequent work on pre-trained visual representations for robotic control. Among these, R3M [39] employs a time-contrastive learning objective on ego-centric data with vision-language alignment, whereas VIP [36] introduces value-implicit learning to associate goal and initial states. MVP [46] and VC-1 [37] both adopt MAE [16] pre-training methodologies, curating large datasets that include ego-centric and instructional videos to enhance transferability to robotic manipulation tasks. More recently, SCR [13] has investigated representations from Stable Diffusion [48] for navigation and control tasks. Nonetheless, these methods opted for keeping the visual representation frozen, resulting them to be task-agnostic.

Diffusion models as pre-trained visual representations. Recent advancements in diffusion models [17, 48] have enabled the synthesis of high-resolution images with unprecedented fidelity. This progress has concurrently motivated diverse investigations into the internal representations of generative diffusion models [1, 25, 35, 56, 61, 62, 69] for various downstream vision tasks. This allowed diffusion models to outperform prior approaches with self-supervised pre-trained models [18, 53] in tasks such as semantic correspondence [5], semantic segmentation [6], and even 3D reconstruction [19, 20]. DDPMSeg [1] was among the first to explore the efficacy of diffusion model’s representations in label-scarce segmentation, while DDAE [62] focused on image classification. DIFT [56], DHF [35] and SDDINO [68] have demonstrated that the representation from diffusion models can achieve state-of-the-art in semantic

[Figure 6]

[Figure 7]

[Figure 8]

a robotic hand in grabbing the green sphere

- Task A
- Task B visual

[Figure 9]

[Figure 10]

[Figure 11]

- Task A
- Task B

sphere

Diffusion hand Model

Vision Encoder

Task-adaptive Representation

[Figure 12]

[Figure 13]

[Figure 14]

Observation Task-agnostic

Observation

Representation

[Figure 15]

taskprompts visualprompts task

(a) Task-agnostic approach (b) Task-adaptive approach with conditions

- Figure 2. Motivation. We aim to overcome the limitations of existing task-agnostic approach (a) with frozen pre-trained visual representations [42], by leveraging conditions in diffusion models for robotic control tasks in a task-adaptive approach (b). In this regard, we explore text conditions(§ 4.1), more advanced methods(§ 4.2,§ 5) as conditions.

correspondence tasks. Notably, VPD [69] demonstrated that downstream performance can be enhanced by with text conditions, such as the names of objects present in an image, in tasks such as semantic segmentation and monocular depth estimation. SD4Match [33] and EcoDepth [43] proposed prompting modules to derive conditions for semantic correspondence and monocular depth estimation. TADP [32] demonstrated that text descriptions generated from vision-language models can serve as strong conditions, and could be further enhanced with style modifiers learned from Textual Inversion [11]. However, we distinguish our approach by focusing on robotic control, rather than for visual tasks in general image domains.

with an U-Net architecture, in which the diffusion process occurs in a compressed latent space learned by an autoencoder, specifically a VQGAN [9]. For conditional generation, U-Net-based LDMs implement Transformer blocks with cross-attention layers into the U-Net blocks to inject the condition C into the image generation process.

Extracting visual representation from diffusion models. To extract visual representations, initially, an input image I is encoded into its latent representation z0 = E(I) using the VQGAN encoder E. For a chosen fixed timestep t, the corresponding noisy latent zt is computed via Eq. 1. This zt is then processed by the denoising U-Net ϵθ(·). However, as the network ϵθ is trained to predict noise as shown in Eq. 2, we instead extract intermediate feature maps from within the U-Net [38]. We denote the set of extracted intermediate features as f, and denote f = ϵθ(zt,t;C) to be the output of ϵθ for simplicity, and primarily consider features from the earlier blocks of the U-Net.

### 3. Preliminaries

Diffusion models [17, 31, 55] constitute a class of generative models that learn to reverse a multi-step noising process, thereby reconstructing a target data distribution. In this work, we focus on conditional diffusion models (e.g. Stable Diffusion [48]), which enable image generation guided by a condition C, often being text prompts. The training objective is to reverse the noising process, typically discretized into T timesteps. A pre-defined noise schedule, denoted by αt, facilitates the definition of the noised latent variable zt at timestep t as:

### 4. Motivation

In this work, we explore conditional diffusion models to generate visual representations for robotic control, aiming to overcome the limitations of task-agnostic approaches. While pre-trained visual representations have been paramount to advancements in control, the standard approach of deploying the same frozen representation across various tasks often fails to adapt to their specific requirements, causing performance to fluctuate significantly [37]. We aim to address this limitation by leveraging text-to-image diffusion models, which have successfully handled diverse visual tasks in a task-adaptive manner using well-designed textual prompts as conditions. Our goal is therefore to explore effective ways to condition diffusion models for control, as illustrated in Fig. 2.

zt = √α¯tz0 + √1 − α¯tϵ, (1)

where z0 is the initial clean data, α¯t = ti=1 αi, and ϵ ∼ N(0,I) is Gaussian noise. Following Ho et al. [17],

with appropriate parameterization, diffusion models can be trained by regressing the added noise ϵ from zt:

0,ϵ,t ∥ϵ − ϵθ(zt(z0,ϵ),t;C)∥22 , (2)

LDM = Ez

where ϵθ indicates the denoising network, typically a UNet [49] or a Transformer [60] architecture. Stable Diffusion, for our case, is a Latent Diffusion Model (LDM) [48]

However, we find that text conditions are ineffective in robotic control environments (§ 4.1), as using captions

|88<br><br>80<br><br>74.7<br><br>Button-press<br><br>(+5.3)<br><br>(+13.3)|
|---|

|87.6<br><br>86.2<br><br>81.8<br><br>Reacher<br><br>(+4.4)<br><br>(+5.8)|
|---|

null text ours

Pen

Cheetah-run

90.4

84

43.4

91.5

80

(-4.0)

(-5.9) ) 86.7 (+6.6)

37.5

93.1

(+2.7)

50

(a) Downstream performance in robotic control tasks with conditions

Image robot press button

<eos> task visual visual

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

(b) Visualization of cross-attention maps for Button-press

Image

<eos> robot cheetah run

task visual visual

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

(c) Visualization of cross-attention maps for Cheetah-run

- Figure 3. Case study. (a) We find that text conditions can be disadvantageous in some control tasks. (b) For Button-press, the crossattention maps (e.g., for button, press) are well-grounded to relevant image regions. (c) In contrast, for Cheetah-run, the attention maps (e.g., for cheetah, run) are noisy, which presumably leads to a decline in performance. Nonetheless, our approach of using task and visual tokens (§ 5) achieves consistent gains across all tasks, with its cross-attention maps capturing diverse regions of the image relevant to the downstream task.

generated from vision-language models yields insignificant gains, or even degrades performance. An in-depth inspection of the cross-attention maps reveals the underlying reason for this failure - in tasks where performance degrades, the diffusion model struggles to correctly associate words with their corresponding image regions. This underscores the need for alternatives to text descriptions and for careful consideration when devising conditions specifically for robotic control.

Consequently, we discuss what do we need for effectively conditioning diffusion models in robotic control (§ 4.2). By their nature, control tasks involve video frames with fine-grained movements of agents and objects. Relying solely on textual conditions would necessitate generating a highly detailed, frame-by-frame description of the specific agent parts relevant to the current action—a challenging and often impractical task. Therefore, we posit that we should incorporate visual information for effective conditions to capture the fine-grained details of each frame.

- 4.1. Exploring textual conditions for robotic control

of these tasks. The full text descriptions are provided in Section B.1 in the appendix. For our analysis, we compare the null (∅) condition—implemented as an empty string with only <eos> and <bos> tokens—and the text condition in downstream control tasks. However, as observed in Fig. 3(a), the results are mixed: while text conditions benefit some tasks (e.g., Button-press, Reacher), they degrade performance in others (e.g., Cheetah-run).

To take a deeper look, in Fig. 3(b), we visualize the cross-attention maps for Button-press, a task where text conditions show noticeable gains. For words such as press or button, the cross-attention maps are well-associated with the relevant regions within the image. These results are similar to what is expected from text conditions in other visual perception tasks like semantic segmentation [69], which verifies the potential of using conditions in control tasks.

However, in Fig. 3(c), we observe the opposite for Cheetah-run, where words like cheetah or run show noisy cross-attention maps. The <eos> token of the null condition is already roughly grounded to the salient object, the agent in this case, which explains how a sub-optimal text condition can degrade performance to be even worse than the null condition. We primarily attribute the failure of text conditions, despite being generated from a state-of-the-art

To obtain textual descriptions of control environments, we devise a baseline by prompting a state-of-the-art visionlanguage model, Gemini 2.5 [7], to generate descriptions

image-specific task-specific

Learnable Frozen

|𝜏𝜃|
|---|

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

###### ℰ𝑉

Concatenate

visual prompts task prompts

Vision Encoder

Text Encoder

[Figure 37]

CrossAttention

Self Attention

|𝜙𝜋|
|---|

Conv

𝜙𝜋

|𝑎|
|---|

Observation

Visual Representation

Policy

Action

|Diffusion Model 𝜖𝜃|
|---|

Figure 4. Proposed framework. We propose ORCA, a framework for learning task and visual prompts to condition diffusion models in robotic control. Specifically, we utilize the features from the downsampling blocks and the bottleneck block of Stable Diffusion [48] to extract visual representations conditioned on our input, which are then fed to the policy network for predicting the action.

vision-language model, to the domain gap between realworld images and simulated control environments. This finding highlights the need for careful consideration when devising conditions in robotic control and motivates the exploration of alternatives to text descriptions for representing the task.

#### 4.2. What do we need as conditions in control?

In order to devise effective conditions, we discuss the characteristics of robotic control tasks and contrast with other vision-based tasks, such as semantic segmentation. A primary distinction is that control tasks operate on video streams rather than static images. Consequently, a logical approach would be to generate a unique condition for each frame [21], allowing the representation to adapt to the changing visual state of the environment. For instance, instructing an agent to walk requires a sequence of distinct commands (e.g., move the left foot, then the right). Similarly, an effective condition should vary across frames to guide such dynamic behaviors. However, generating highquality text descriptions on a frame-by-frame basis would not only be challenging but would also inherit the same grounding limitations discussed previously.

In this regard, we hypothesize that to account for this dynamic adaptability, conditions should incorporate visual information from each frame. While diffusion models like Stable Diffusion are typically trained on text, several approaches exist for incorporating visual information, either by introducing features from external vision encoders [33, 43] or by optimizing specialized text tokens to represent visual concepts [26, 32]. These existing methods, such as TADP [32] however, tend to embed the global representation into the condition or require additional optimization

steps to acquire specialized tokens. Since our goal is to enable the recognition of fine-grained regions within each frame, we consider that adopting global representations and extra optimization steps should be avoided to facilitate effective frame-wise conditioning.

### 5. ORCA: Conditioning diffusion models for robotic control

Based on our observations, we present ORCA, a simple yet effective approach for learning conditions for diffusion models in robotic control. We design our conditions to adapt to the control environment, preventing erroneous grounding, while simultaneously incorporating visual information to capture dynamic details. To achieve this with minimal overhead during downstream policy learning, we formulate these conditions as learnable prompts [11]. Specifically, we introduce learnable task and visual prompts that integrate task-specific implicit descriptions with frame-level visual information, as described in detail below.

Task prompts. Recalling that text conditions show potential when well-grounded to task-relevant regions, we design our task prompt to capture objects or areas that are critical to solving the downstream task. Therefore, we adopt a direct approach of learning the text as implicit words [70] within the downstream task to minimize erroneous grounding. To achieve this, we implement task prompts as learnable parameters that are shared across all observations during training. We find that this allows the task prompts to implicitly learn to focus on relevant regions, as shown in Fig. 3(b,c), where the cross-attention maps simultaneously

[Figure 38]

[Figure 39]

highlight both the button and the robot arm in Button-press and the agent in Cheetah-run.

Walker-stand Walker-walk Reacher-easy Cheetah-run Finger-spin Relocate

[Figure 40]

[Figure 41]

Visual prompts. Furthermore, to incorporate visual information into the conditions, we adopt a vision encoder EV to leverage its visual representation as prompts. Specifically, we utilize the dense visual representations from EV , rather than global representations, and project them through a small convolutional layer to complement the task prompts. This focus on dense features provides the fine-grained, localized information necessary for control tasks. As visualized in Fig. 3(c), the resulting attention from the visual prompts highlights various regions in detail, such as distinguishing between the front and back legs of the agent.

Assembly Bin-picking Button-press Drawer-open Hammer Pen

Figure 5. Visualization of evaluation tasks. We utilize MuJoCo [58] tasks for evaluation, with 5 tasks from MetaWorld [66],

- 5 tasks from DeepMind Control [57], and 2 tasks from Adroit [47].
- 6. Experiments

#### 6.1. Evaluation suites

We conduct experiments on three widely-used vision-based robot learning benchmarks with the total of 12 tasks following VC-1 [37], as shown in Fig. 5.

Policy learning objective. We learn the prompts by directly optimizing for the behavior cloning objective in downstream policy learning, as presented in Fig. 4. Let πϕ(·) be the policy network with parameters ϕ that takes the visual state representations derived by the diffusion model and outputs actions. Given sequences of To observations {Ioi}T

DeepMind Control (DMC) [57] is a set of continuous control tasks with simulated robots. We use five imitation learning cases: Walker-stand, Walker-walk, Reacher-easy, Cheetah-run, and Finger-spin. We report the normalized scores for all tasks.

o=0 and actions {aio}T

o=0 from the i-th trajectory, we predict each action and train both the policy network πϕ(·), task prompts pt and visual prompts pt by the behavior cloning loss:

o

o

MetaWorld [66] is a suite of simulated robotic manipulation tasks with a Sawyer robot arm. We focus on a subset of five representative tasks: Assembly, Bin-picking, Buttonpress-topdown, Drawer-open, and Hammer. We measure the best success rates among the online evaluation trials.

N

Adroit [47] is an imitation learning benchmark in a simulated environment, consisting of dexterous manipulation tasks that require an agent to control a 28-DoF anthropomorphic hand. We focus on Relocate and Pen, and measure the best success rates among the online evaluation trials.

||πϕ(ϵθ(zt,t;C∗)) − aio||, (3)

LBC(ϕ,p) =

i=1 o

where zt = √α¯tE(Ioi) + √1 − α¯tϵ, and condition C∗ = τθ(pt;pv) is derived from the text encoder τθ with task prompt pv and visual prompt pv as the input. We find that pv and pt can be both learned with the behavior cloning loss in downstream policy learning.

#### 6.2. Implementation details

Diffusion model and conditions. We employ Stable Diffusion v1.5 [48] as the diffusion model. For extracting visual representation from observations, we leverage the features from the downsampling blocks and the bottleneck block in the diffusion U-Net and forward through a compression layer [64]. We set the timestep t = 0, the length of task tokens lt = 4, and the length of visual tokens lv = 16, where all learnable parameters are randomly initialized. For EV , we employ pre-trained DINOv2 [40]. Further implementation details are presented in the appendix.

Discussion. Since our method learns prompts during downstream training, one might argue that task-adaptive representations could alternatively be achieved by finetuning the diffusion model itself. However, this would require a fully-tuned model to be stored for each task. ORCA, on the other hand, simultaneously learns task prompts and visual prompts from the current visual state to derive taskspecific conditions during the downstream training, without the need for additional optimization as of in TADP [32]. This allows the prompting modules to be swapped for handling different tasks, as the diffusion model itself is not modified. Furthermore, we find the fine-tuning approach—exemplified by SCR [13]—results in significantly degraded performance, with success rates collapse by over 80% compared to the frozen counterpart (§ 6.5), as full finetuning on limited imitation learning data leads to severe overfitting [37, 42].

Vision-based robot policy learning. We consider two, five, and five demonstrations from Adroit, DeepMind Control (DMC), and MetaWorld, respectively, where proprioceptive data is utilized except for the DMC benchmark. We mainly follow the experimental setups in VC-1 [37] except that we employ a compression layer for all baselines for fair comparison. For each task, we train the agent for 100 epochs, with a periodic online evaluation in the simulated environment every 10 epochs.

- Table 1. Experimental results on vision-based robot policy learning on DeepMind Control [57] and MetaWorld [66]. We report the normalized score for DeepMind Control and success rates (%) for MetaWorld, averaged over three seeds with standard deviation.

Methods Backbone

DeepMind Control MetaWorld Stand Walk Reacher Cheetah Finger Mean Assembly Bin-picking Button Drawer Hammer Mean

CLIP [45] ViT-L/16 87.3 ± 2.4 58.3 ± 4.4 54.5 ± 4.6 29.9 ± 5.6 67.5 ± 2.1 59.5 85.3 ± 12.2 69.3 ± 8.3 60.0 ± 13.9 100.0 ± 0.0 92.0 ± 8.0 81.3 VC-1 [37] ViT-L/16 86.1 ± 0.9 54.3 ± 6.6 18.3 ± 2.4 40.9 ± 2.7 65.7 ± 1.1 53.1 93.3 ± 6.1 61.3 ± 12.2 73.3 ± 8.3 100.0 ± 0.0 93.3 ± 6.1 84.2 SCR [13] SD 1.5 85.5 ± 2.6 64.3 ± 3.5 81.8 ± 9.9 43.4 ± 6.4 66.6 ± 2.7 68.3 92.0 ± 6.9 86.7 ± 4.6 74.7 ± 12.9 100.0 ± 0.0 98.7 ± 2.3 90.4 Text (Simple) SD 1.5 87.6 ± 4.6 67.9 ± 4.6 84.3 ± 4.6 38.8 ± 5.9 66.7 ± 0.2 69.1 97.3 ± 2.3 85.3 ± 2,3 78.7 ± 2,3 100.0 ± 0.0 96.0 ± 6.9 91.5 Text (Caption) SD 1.5 87.2 ± 4.5 68.3 ± 5.9 86.2 ± 1.9 37.5 ± 2.6 65.1 ± 1.8 68.9 96.0 ± 4.0 88.0 ± 6.9 80.0 ± 8.0 100.0 ± 0.0 98.7 ± 2.3 92.5 CoOp [70] SD 1.5 87.2 ± 2.2 67.8 ± 6.4 87.1 ± 5.9 45.0 ± 6.4 65.9 ± 1.0 70.6 96.0 ± 4.0 89.3 ± 2,3 81.3 ± 6.1 100.0 ± 0.0 96.0 ± 6.9 92.5 TADP [32] SD 1.5 89.0 ± 2.9 69.9 ± 7.9 86.6 ± 5.6 41.1 ± 3.9 66.9 ± 0.2 70.7 96.0 ± 4.0 90.7 ± 4.6 80.0 ± 10.6 100.0 ± 0.0 96.0 ± 4.0 93.1

ORCA (Ours) SD 1.5 89.1 ± 1.8 76.9 ± 4.0 87.6 ± 2.9 50.0 ± 8.4 68.0 ± 1.0 74.3 98.7 ± 2.3 90.7 ± 4.6 88.0 ± 6.9 100.0 ± 0.0 98.7 ± 2.3 95.2

- Table 2. Experimental results on vision-based robot policy learning on Adroit [47]. We report the success rates (%) averaged over three seeds with their standard deviation.

Table 3. Comparison to fine-tuning. For comparison, we fine-tune VC-1 [37] and SCR [13] on Adroit [47] under fullfinetuning and parameter-efficient fine-tuning scenarios with RoboAdapter [52] and LoRA [22]. We report the number of learnable parameters and the performance in success rates (%) averaged over three seeds with their standard deviation.

Adroit Pen Relocate Mean

Methods Backbone

CLIP [45] ViT-L/16 58.7 ± 2.3 44.0 ± 4.0 51.4 VC-1 [37] ViT-L/16 65.3 ± 16.7 29.3 ± 8.3 47.3 SCR [13] SD 1.5 84.0 ± 4.0 32.0 ± 4.0 58.0 Text (Simple) SD 1.5 80.0 ± 6.9 34.7 ± 6.1 57.3 Text (Caption) SD 1.5 80.0 ± 4.0 34.7 ± 4.6 57.3 CoOp [70] SD 1.5 82.7 ± 6.1 33.3 ± 6.1 58.0 TADP [32] SD 1.5 81.3 ± 6.1 33.3 ± 8.3 57.3

# learn. Adroit params. Pen Relocate Mean

Methods

VC-1 [37] - 65.3 ± 16.7 29.3 ± 8.3 47.3 + Fine-tuning 302.3M 58.7 ± 13.2 4.0 ± 3.3 31.3 + RoboAdapter 18.0M 77.3 ± 4.6 41.3 ± 8.3 59.3

SCR [13] - 84.0 ± 4.0 32.0 ± 4.0 58.0 + Fine-tuning 346.7M 17.3 ± 3.8 1.3 ± 1.9 9.3 + LoRA 4.6M 77.3 ±6.1 42.7 ± 15.1 60.0

ORCA (Ours) SD 1.5 86.7 ± 2.3 44.0 ± 4.0 65.3

#### 6.3. Baselines

ORCA (Ours) 10.6M 86.7 ± 2.3 44.0 ± 4.0 65.3

For baselines, we consider CLIP [45] and VC-1 [37] as widely adopted task-agnostic baselines. Furthermore, we construct 5 baselines based on diffusion models to explore conditions in robotic control, with details in the appendix.

it in certain tasks. This highlights a fundamental limitation of such approaches: due to their task-agnostic nature, no single representation is guaranteed to excel across all tasks. In contrast, across all 12 tasks in the 3 evaluation suites, ORCA establishes the new state-of-the-art, outperforming all baselines by a significant margin.

- • SCR [13] is a task-agnostic baseline, which is one of the first works to introduce Stable Diffusion into control tasks. SCR can be considered as an unconditional baseline, where it employs the null(∅) condition for all tasks.
- • Text (Simple/Caption) are task-adaptive baselines using text conditions, where the simple variant uses the task names defined in the evaluation suite, and caption variant leverages descriptions generated from Gemini 2.5 [7].
- • CoOp [70] extends on Text (Simple) by implementing learnable prefix tokens to the text. CoOp can be considered as a less flexible variant to the task prompt as the task name is fixed in the prompt.
- • TADP [32] extends on Text (Caption) by appending a special token S∗ that encapsulates the visual style information, separately optimized through Textual Inversion [11]. Hence, we can consider TADP as a baseline with visual information from only a single, fixed image.

Furthermore, we observe that more advanced taskadaptive baselines, CoOp and TADP, generally outperform text conditions. This confirms our hypothesis that incorporating visual information is beneficial, given from the results of TADP which utilizes visual information in a limited manner with a global observation of the task. Nonetheless, since both methods were not designed for robotic control tasks, their effectiveness is limited as shown by their minimal gains on DMC and Adroit. In contrast, our method show solid improvements across all tasks.

#### 6.5. Analysis

Comparison with downstream fine-tuning. To compare learning prompts to downstream fine-tuning, we fine-tune VC-1 [37] and SCR [13] on Adroit [47], evaluating both full fine-tuning and parameter-efficient fine-tuning with adapter modules. For the parameter-efficient methods, we use RoboAdapter [52] for ViT-based VC-1 and LoRA [22] for diffusion-based SCR. As shown in Table 3, full finetuning requires approximately 30× more parameters but

#### 6.4. Main results

Quantitative results. We report experimental results from DMC and MetaWorld in Table 1, and Adroit in Table A. Among the task-agnostic baselines, while SCR performs best overall, we observe that VC-1 and CLIP outperform

- Table 4. Components analysis. To ablate the design choices, we conduct component analysis on task prompt pt and visual prompt pv on DeepMind Control [57]. We report the normalized score averaged over three seeds with its standard deviation.

DeepMind Control

pt pv Stand Walk Reacher Cheetah Finger Mean

85.5 ± 2.6 64.3 ± 3.5 81.8 ± 1.7 43.4 ± 4.4 66.6 ± 2.7 68.3 ✓ 83.6 ± 3.2 71.4 ± 3.5 86.7 ± 6.6 38.9 ± 10.1 68.2 ± 1.2 69.8 ✓ 85.9 ± 2.7 71.1 ± 2.3 87.3 ± 5.5 42.0 ± 10.4 66.1 ± 1.0 70.5 ✓ ✓ 89.1 ± 2.3 76.9 ± 4.0 87.6 ± 2.9 50.0 ± 8.4 68.0 ± 1.0 74.3

- Table 5. Ablation study on layer selection. We evaluate individual layers of the diffusion U-Net by reporting layer-wise performance on DeepMind Control [57]. d + m refers to concatenating down and mid blocks. We report the normalized score averaged over three seeds with its standard deviation.

Image 𝑝 𝑝 𝑝

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

Frame 15Frame 1Frame 30Frame 45

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

DeepMind Control Layer Stand Walk Reacher Cheetah Finger Mean

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

- down 1 86.3 ± 2.1 65.5 ± 1.1 82.1 ± 3.7 40.8 ± 1.1 67.6 ± 0.3 68.4

- down 2 89.3 ± 1.2 68.3 ± 2.7 70.0 ± 18.8 31.2 ± 2.6 67.0 ± 1.0 65.1

- down 3 86.2 ± 4.3 73.3 ± 3.9 75.3 ± 8.1 36.0 ± 4.8 67.0 ± 0.5 67.5 mid 88.3 ± 4.9 70.4 ± 1.3 62.3 ± 1.1 35.0 ± 4.7 67.2 ± 0.6 64.6

- up 0 82.8 ± 2.6 71.7 ± 5.9 45.3 ± 4.0 28.5 ± 1.8 67.2 ± 0.6 59.0

- up 1 79.5 ± 4.5 60.3 ± 16.1 55.9 ± 5.2 39.9 ± 7.0 66.4 ± 0.4 60.4

- up 2 70.4 ± 4.5 39.1 ± 3.3 41.0 ± 7.0 30.9 ± 3.1 67.7 ± 1.0 49.7 d + m 89.1 ± 1.8 76.9 ± 4.0 87.6 ± 2.9 50.0 ± 8.4 68.0 ± 1.0 74.3

Figure 6. Cross-attention visualization for task/visual prompts. We visualize the cross-attention maps for task prompt pt and visual prompts p1v and p2v in across frames in the Relocate task.

iors. While p1v tends to focus on the hand, p2v interestingly attends to the table as the hand moves downward to pick up the ball, then shifts its focus to the hand as it lifts off and moves toward the target, suggesting that it has learned to capture task-relevant movements.

proves ineffective, completely deteriorating SCR’s performance. Although parameter-efficient methods mitigate this, ORCA significantly boosts performance while maintaining a comparable parameter count, highlighting the superior efficiency and efficacy of our condition-based approach.

Visualization of task and visual prompts. In Fig. 6, we visualize the cross-attention maps for our task prompt pt and two different tokens from the visual prompt, p1v and p2v, on Relocate. In this task, a robot hand first picks up a blue ball from a table (Frames 1-30) and then moves it to the location of a green sphere (Frames 30-45). As discussed in § 5, we observe that the task prompt consistently captures regions relevant to the overall goal, namely the robot hand and the target green sphere. Conversely, the visual prompts exhibit more dynamic behaviors. While p1v tends to focus on the hand, p2v interestingly attends to the table as the hand moves downward to pick up the ball, then shifts its focus to the hand as it lifts off and moves toward the target, suggesting that it has learned to capture task-relevant movements.

Ablation study on each component. In Table 4, we conduct component analysis by ablating task prompt pt and visual prompt pv respectively. Notably, we observe that when employed individually, task and visual prompts can show divergent behavior across different tasks. This could suggest that each tasks focus in different aspects of the scene, such as Reacher-easy focusing more in visual details as it benefits more from visual prompts compared to text prompts. Nonetheless, when fully incorporating both pv and pt, we show consistent gains across all tasks.

Ablation study on layer selection. In Table 5, we evaluate the diffusion model in different layers. We can observe that the early layers tend to perform better than the latter upsampling layers. Therefore, we concatenate the best-performing layers (down 1-3,mid), which yields the best overall performance.

### 7. Conclusion

In this work, we introduced ORCA, a framework for bridging text-to-image diffusion models for robotic control to generate task-adaptive visual representations. We identified the limitations of conventional text prompts, and we proposed a simple yet effective method using learnable task and visual prompts. ORCA achieves state-of-the-art performance, highlighting the importance of task-adaptive representations and the vast potential of properly conditioned diffusion models for robotic control.

Visualization of task and visual prompts. In Fig. 6, we visualize the cross-attention maps for our task prompt pt and visual prompts, p1v and p2v, on Relocate. In this task, a robot hand first picks up a blue ball from a table (Frames 1-30) and then moves it to the location of a green sphere (Frames 30-45). As discussed in § 5, we observe that the task prompt consistently captures regions relevant to the overall goal, namely the robot hand and the target green sphere. Conversely, the visual prompts exhibit more dynamic behav-

## Exploring Conditions for Diffusion Models in Robotic Control Supplementary Material

### Contents

- • §A: Further analysis

- – §A.1: Results with different diffusion backbones
- – §A.2: Results on LIBERO-Long benchmark
- – §A.3: Ablation on the vision encoder for visual prompt
- – §A.4: Ablation on timesteps
- – §A.5: Comparison with stronger pre-trained encoders
- – §A.6: Efficiency comparison
- – §A.7: Analysis on the null condition
- – §A.8: Further discussion

- • §B: Further implementation details

- – §B.1: Full description of the text conditions
- – §B.2: Details of the baselines
- – §B.3: Implementation details of the compression layer

- • §C: Limitations
- • §D: Qualitative results on robotic control tasks

### A. Further Analysis

#### A.1. Results with different diffusion backbones

Table A. Experimental results with different diffusion backbones. The performance of imitation learning agents on Adroit [47] is reported. We report the success rates (%) averaged over three seeds with their standard deviation.

Adroit Pen Relocate Mean

Methods Backbone

CLIP [45] ViT-L/16 [8] 58.7 ± 2.3 44.0 ± 4.0 51.4 VC-1 [37] ViT-L/16 [8] 65.3 ± 16.7 29.3 ± 8.3 47.3

SCR [13] SD 1.5 [48] 84.0 ± 4.0 32.0 ± 4.0 58.0 Video Diffusion SVD [2] 49.3 ± 3.8 2.7 ± 1.9 26.0 Diffusion Transformer SD 3.0 [10] 54.7 ± 16.4 5.3 ± 1.9 30.0

ORCA (Ours) SD 1.5 [48] 86.7 ± 2.3 44.0 ± 4.0 65.3

To study different diffusion backbones, in Table A, we present results from a video diffusion model, Stable Video Diffusion (SVD) [2] and a diffusion Transformer model, Stable Diffusion 3.0 (SD 3.0) [10], which has a multimodal diffusion Transformer (MM-DiT) as its backbone. For SVD, we use the same layer selection strategy from ours and SCR as SVD is also based on the same U-Net architecture, and use the last layer for MM-DiT.

Interestingly, despite SVD being a dedicated video generation model, it significantly falls behind SD 1.5. While there could be several reasons, we consider this to originate from input discrepancy, where we restrict the input to

- 3 frames for fair comparison. This deviates from the native 8-frame setting of SVD, which likely disrupts the pretrained knowledge. Moreover, the publicly released version of SVD does not support text conditioning, which precludes

the use of text or learnable prompts without additional finetuning for customizing the model to accept text [23].

On the other hand, SD 3.0, based on the MM-DiT architecture [10], supports text conditioning through its joint multi-modal attention mechanism. However, we find that the DiT-based model underperforms compared to U-Netbased models, regardless of whether text conditions are provided. Given that empirical studies [25] on DiT-based models remain limited compared to the extensive research on U-Net architectures [12, 63], we believe there is significant room for improvement with DiT-based models. Nonetheless, we leave the deeper exploration of video diffusion and DiT-based models as a direction for future work.

#### A.2. Results on LIBERO-Long benchmark.

Table B. Experimental results on tasks from LIBERO-Long. The performance of imitation learning agents on LIBEROLong [34] benchmark.

Single-task Multi-task Methods KITCHEN-3 LIVING-1 STUDY-1 Mean Mean

VC-1 51.7 ±10.4 43.3 ±10.4 35.0 ±26.0 43.3 26.7 SigLIP 51.7 ±7.6 20.0 ±5.0 38.3 ±23.1 36.7 16.7 SCR 86.7 ±11.5 45.0 ±5.0 40.0 ±8.7 57.2 23.3

ORCA (Ours) 93.3 ±11.5 48.3 ±7.6 55.0 ±2.9 65.5 46.6

To investigate the efficacy of ORCA in long-horizon and language-conditioned tasks, we employed 3 tasks from LIBERO-long [34] for evaluation. As shown in Table B, we can observe that ORCA surpasses baselines by a considerable margin, which confirms its applicability to various scenarios including long-horizon tasks. Furthermore, we report the multi-task results on the LIBERO-long benchmark, where we jointly train all 3 tasks with a shared policy. Notably, ORCA consistently outperforms the baselines by a significant margin, indicating that the visual and task prompts can be learned jointly from multiple tasks.

#### A.3. Ablation on the choice of the visual encoder for visual prompt.

In Table C, we provide ablation on the choice of the additional visual encoder EV , which is utilized to obtain the visual prompts pv. In this regard, we replace the DINOv2 [40] encoder with SigLIP [67] and CLIP [45], as well as SDVAE [59] which is used for obtaining the latents in Stable Diffusion 1.5. As shown in Table C, all variants show noticeable gains over the baseline (i.e., w/o vision encoder), while the choice of encoder shows some variance in terms of performance. Especially, the results from SD-VAE shows

Table C. Ablation on the vision encoder EV for the visual prompts. To ablate the choice of the vision encoder EV for obtaining visual prompts, we provide provide results with SigLIP [67], CLIP [45], and SD-VAE [59]

Adroit Pen Relocate Mean

Methods

w/o vision encoder 76.0 ±4.0 33.3 ±2.3 54.7 + SigLIP [67] 77.3 ±2.3 33.3 ±10.1 55.3 + CLIP [45] 81.3 ±4.6 45.3 ±14.6 63.3 + SD-VAE [59] 81.3 ±6.1 41.3 ±2.3 61.3

ORCA (Ours) 86.7 ±2.3 44.0 ±4.0 65.3

that we could achieve competitive results even without an external vision encoder. This confirms that the usage of visual prompt itself provides a substantial benefit and plays a core role in its overall effectiveness.

#### A.4. Ablation on diffusion timesteps

- Table D. Ablation study on timestep selection. To ablate the choice for timestep t, we provide results with t = 100 and t =

200. The performance of imitation learning agents on DeepMind Control [57] is reported. We report the normalized score averaged over three seeds with its standard deviation.

DeepMind Control Timestep Walker-stand Walker-walk Reacher-easy Cheetah-run Finger-spin Mean

200 92.2 ± 1.6 78.6 ± 2.2 85.4 ± 8.3 24.7 ± 4.5 66.5 ± 3.2 69.4 100 88.3 ± 4.7 72.6 ± 4.3 79.4 ± 6.8 36.1 ± 6.2 66.2 ± 3.5 68.5

0 (Default) 89.1 ± 1.8 76.9 ± 4.0 87.6 ± 2.9 50.0 ± 8.4 68.0 ± 1.0 74.3

To ablate the effects of the diffusion timestep t, in Table D, we additionally provide results with t = 100 and t = 200. Although some tasks (e.g. Reacher-easy) benefit from t = 100 or t = 200, performance on other tasks such as Cheetah-run degrades significantly, lowering the overall score. Therefore, we choose t = 0, which achieves the best overall performance.

A.5. Comparison with stronger pre-trained visual representations

- Table E. Additional comparison with stronger pre-trained visual representations. We further provide comparison with recent state-of-the-art pre-trained visual representations, including DINOv2 [40], SigLIP [67], and Theia [51]. We report the success rates (%) averaged over three seeds with their standard deviation.

DeepMind Control Walker-stand Walker-walk Reacher-easy Cheetah-run Finger-spin Mean

Method

DINOv2 [40] 87.6 ±5.2 61.4 ±9.3 22.5 ±3.2 27.1 ±4.2 64.1 ±10.8 52.5 SigLIP [67] 81.4 ±3.6 50.3 ±4.4 87.1 ±5.5 18.6 ±2.6 66.6 ±2.4 60.8 Theia [51] 85.3 ±2.8 65.0 ±2.9 61.4 ±13.4 43.8 ±9.8 67.8 ±1.0 64.6

ORCA (Ours) 89.1 ±1.8 76.9 ±4.0 87.6 ±2.9 50.0 ±8.4 68.0 ±1.0 74.3

To study whether recent state-of-the-art pre-trained visual representations could overcome its limitations from

task-agnostic nature, we compare ORCA with DINOv2 [40] and SigLIP [67], while also including Theia (Shang et al., 2024), a distilled vision foundation model built for manipulation tasks in Table E. Notably, these pre-trained encoders still exhibit underwhelming performance in several tasks (e.g. Reacher-easy for DINOv2, Cheetah-run for SigLIP), highlighting the inherent limitations of taskagnostic representations in complex control settings. Considering that SigLIP is widely adopted in recent VisionLanguage-Action (VLA) models, these results suggest that incorporating ORCA within VLA models can be a promising direction for future research.

#### A.6. Efficiency comparison

Table F. Efficiency comparison. We report the total number of parameters (#Params), the number of learnable parameters (#Learnable) and latency for VC-1, SCR, and ours.

Methods #Params #Learnable Time VC-1 [37] 303.3M 0 11ms SCR [13] 382.9M 0 26ms Ours 480.1M 10.6M 48ms

In Table F, we report the number of parameters, number of learnable parameters, and latency for each modules for VC-1 [37], SCR [13] and our proposed method. For VC-1 and SCR, we use ViT-L/16, which was also used for comparison in the main paper. Notably, the layer selection allows us to drop the “up” blocks, which removes around 500M parameters from the denoising U-Net. This allows the U-Net to have similar parameter count to VC-1 encoder, which is used in various robotic manipulation tasks. Furthermore, most of the parameters added to our method are the frozen parameters from DINOv2, and the learnable parameters consist mostly of the additional projection layers for the visual prompts.

#### A.7. Analysis on the null condition

Figure A illustrates the attention behavior of <bos> and <eos> tokens by visualizing their normalized crossattention maps (b,d) and raw attention scores (c,e). We observe that the <bos> token consistently attends to background regions, whereas the <eos> is less reliable at focusing on salient objects (e.g. the robot hand in Relocate). We attribute the background affinity of <bos> to the typical structure of text prompts, which primarily describe foreground subjects. Moreover, since Stable Diffusion employs a causal text encoder for both conditional prompts and the null condition ∅ in unconditional generation, this background-attending behavior is also transferred to unconditional scenarios.

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

Walker-walkBin-picking

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

Relocate

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

(a) Image (c) <bos>, Score (d) <eos>, Weight (e) <eos>, Score

(b) <bos>, Weight

- Figure A. Visualization of normalized attention weights and raw attention scores for <bos> and <eos> tokens. We compare the visualization of the normalized attention weights obtained after the softmax operation and the raw attention scores obtained before the softmax operation from the cross-attention layers to further analyze the properties of <bos> and <eos> tokens.

- A.8. Further discussions

It is also worth considering Vision-Language-Action models (VLAs) [27, 41], which are trained on large-scale robotic data. However, adapting these billion-scale models to new robot embodiments often necessitates computationally expensive fine-tuning [27]. While this cost is arguably mitigated by their generalization capabilities, such as zero-shot instruction following from language descriptions, we focus on efficient, vision-based specialist agents for robotic control tasks rather than generalist approach. Consequently, ensuring these specialist agents are task-adaptive remains a critical research objective. Furthermore, it is important to note that conditional diffusion models leverage general image-text data [50], which is significantly easier to collect and scale than the robot interaction data required to train VLAs [41].

- B. Further Implementation Details

- B.1. Full description of the text conditions

In Table G, we provide the full descriptions used for Text (Simple) and Text (Caption), which are generated by Gemini 2.5 Pro [7]. For CoOp [70], we use 4 learnable prefix tokens, such as “[V ∗][V ∗][V ∗][V ∗] bin picking” for Binpicking. For TADP, we add a style prefix “in a [S∗] style”, which results in “The Sawyer robot arm must carefully pick a specific target object out of the cluttered red bin and place it into the empty blue bin in a [S∗] style.” for Bin-picking.

#### B.2. Details of the baselines

CLIP [45] is a vision-language model pre-trained on largescale image-text pairs through contrastive learning. CLIP has been widely used in various tasks, including navigation

and manipulation tasks [24, 54].

VC-1 [37] is a foundation model for various robotics tasks, spanning from manipulation to locomotion and navigation tasks. VC-1 trains with MAE objective on egocentric videos, as well as additional data including navigation and manipulation datasets.

SCR [13] employs Stable Diffusion for various navigation and manipulation tasks. We consider SCR as a baseline using the null condition ∅, which is implemented as an empty string.

Text(Simple/Caption) is a task-adaptive baseline using text conditions, where Text (Simple) directly uses the task names as the condition, whereas Text (Caption) leverages descriptions generated from Gemini 2.5 [7]. Full text used for each tasks are presented in the appendix.

CoOp [70] extends on Textsimple by implementing learnable prefix tokens V ∗. CoOp originally prompts CLIP with the format “[V ∗] classname” for image classification, which in our case, the task names used in Textsimple are used as classnames.

TADP [32] extends on Textcaption, by adding a special token S∗ that encapsulates the visual style information optimized through Textual Inversion [11]. Since the visual information is optimized into a single token S∗, we can consider TADP as a baseline with global visual information, and not in a frame-wise manner.

#### B.3. Implementation details of the compression layer

Algorithm 1: PyTorch-style pseudocode for the compression layer

class CompressionLayer(nn.Module): def init (self, hidden dim, compress dim): self.layers = nn.Sequential(

nn.Conv2d(hidden dim, compress dim, kernel size=3, padding=1),

nn.BatchNorm2d(hidden dim), nn.ReLU(inplace=True), nn.Flatten() )

def forward(self, x): return self.layers(x)

To provide further details of the compression layer [64], we provide a PyTorch-style pseudo-code of the compression layer in Alg. 1. We follow previous works [13, 64] for implementing a simple convolutional layer for the compression layer to obtain 1D state representations from 2D features. For all methods, compress dim was set to 48.

###### Table G. Full text descriptions used in baselines.

Task Method Text

Text (Simple) “assembly”

Assembly

Text (Caption) “The Sawyer robot arm must pick up the green block and precisely insert it into the center of the silver ring to complete the assembly.” Bin

Text (Simple) “bin picking”

Text (Caption) “The Sawyer robot arm must carefully pick a specific target object out of the cluttered red bin and place it into the empty blue bin.” Button

Text (Simple) “button press”

Text (Caption) “The Sawyer robot arm must reach out and accurately press the red button on top of the yellow control box.” Drawer

Text (Simple) “drawer open”

Text (Caption) “The Sawyer robot arm must grasp the white handle and pull open the light green drawer.” Hammer

Text (Simple) “hammer”

Text (Caption) “The Sawyer robot arm must pick up the red hammer and use it to strike the nail, driving it completely into the wooden block.” Pen

Text (Simple) “pen”

Text (Caption) “A dexterous robotic hand must twirl a blue pen within its grasp to match the final orientation shown by the green target pen.” Relocate

Text (Simple) “relocate”

Text (Caption) “A dexterous robotic hand is tasked with picking up the small blue ball and moving it to the location of the green target sphere.” Cheetah-run

Text (Simple) “cheetah run”

Text (Caption) “A minimalist orange robot, shaped like a cheetah, runs across a reflective floor in a simulated environment.” Walker-walk

Text (Simple) “walker walk”

Text (Caption) “A minimalist, orange bipedal robot takes a step across a reflective floor in a simulated environment.” Walker-stand

Text (Simple) “walker stand”

Text (Caption) “A minimalist, orange bipedal robot stands upright on a reflective floor in a simulated environment.” Finger-spin

Text (Simple) “finger spin”

Text (Caption) “A simple robotic finger strikes a floating, hot dog-shaped object to make it spin against a starry background.” Reacher

Text (Simple) “reacher”

Text (Caption) “A simple robotic arm reaches for a red target ball on a checkered blue surface.”

Note that the compression layer was also used for compared baselines including CLIP [4] and VC-1 [37], which have been shown to yield better performance than using <CLS> tokens [13].

### C. Limitations

As previously discussed, we base our exploration on pretrained Stable Diffusion v1.5, which is a U-Net-based diffusion model. Consequently, our findings may not directlyt apply to diffusion models with different architectures, such as DiT [10, 44] models or video diffusion models [2], which we leave to be further explored in the future.

### D. Qualitative visualization on robotic control tasks

We provide frame-wise comparison of our method, CLIP [4], and VC-1 [37] for tasks from DMC [57] in Fig. C, MetaWorld [66] in Fig. D, and Adroit [47] in Fig. B. For each task, we report the normalized score for DMC and whether the task has succeeded or failed for MetaWorld and Adroit.

Pen

[Figure 73]

FailFailSuccess VC-1CLIPOurs

Relocate

[Figure 74]

FailSuccessSuccess VC-1CLIPOurs

- Figure B. Visualization of agents performing downstream tasks in Adroit [47]. We provide visual comparison of our method to CLIP [4], and VC-1 [37] for two tasks from Adroit. We additionally report whether the task has succeeded or failed for each episode.

Walker-stand

[Figure 75]

47.833.784.5 VC-1CLIPOurs

Walker-walk

[Figure 76]

69.74.783.3 VC-1CLIPOurs

Reacher-easy

[Figure 77]

79.00.798.5 VC-1CLIPOurs

Cheetah-run

[Figure 78]

6.916.973.1 VC-1CLIPOurs

Finger-spin

[Figure 79]

66.366.868.9 VC-1CLIPOurs

- Figure C. Visualization of agents performing downstream tasks in DMC [57]. We provide a visual comparison of our method to CLIP [4], and VC-1 [37] for five tasks in DMC. We additionally report the normalized score for each episode.

Assembly

[Figure 80]

SuccessFailSuccess VC-1CLIPOurs

Bin-picking

[Figure 81]

FailFailSuccess VC-1CLIPOurs

Button-press

[Figure 82]

SuccessSuccessSuccess VC-1CLIPOurs

##### Drawer-open

[Figure 83]

SuccessSuccessSuccess VC-1CLIPOurs

##### Hammer

[Figure 84]

SuccessSuccessSuccess VC-1CLIPOurs

- Figure D. Visualization of agents performing downstream tasks in MetaWorld [66]. We provide visual comparison of our method to CLIP [4], and VC-1 [37] for five tasks in MetaWorld. We additionally report whether the task has succeeded or failed for each episode.

### References

- [1] Dmitry Baranchuk, Ivan Rubachev, Andrey Voynov, Valentin Khrulkov, and Artem Babenko. Label-efficient semantic segmentation with diffusion models. arXiv preprint arXiv:2112.03126, 2021. 1, 2
- [2] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 2, 5
- [3] Mathilde Caron, Hugo Touvron, Ishan Misra, Herv´e J´egou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9650–9660, 2021. 2
- [4] Mehdi Cherti, Romain Beaumont, Ross Wightman, Mitchell Wortsman, Gabriel Ilharco, Cade Gordon, Christoph Schuhmann, Ludwig Schmidt, and Jenia Jitsev. Reproducible scaling laws for contrastive language-image learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2818–2829, 2023. 2, 5, 6, 7, 8
- [5] Seokju Cho, Sunghwan Hong, Sangryul Jeon, Yunsung Lee, Kwanghoon Sohn, and Seungryong Kim. Cats: Cost aggregation transformers for visual correspondence. Advances in Neural Information Processing Systems, 34:9011–9023,

2021. 2

- [6] Seokju Cho, Heeseong Shin, Sunghwan Hong, Anurag Arnab, Paul Hongsuck Seo, and Seungryong Kim. Cat-seg: Cost aggregation for open-vocabulary semantic segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 4113– 4123, 2024. 2
- [7] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025. 2, 4, 7
- [8] Alexey Dosovitskiy. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020. 2
- [9] Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12873–12883, 2021. 3
- [10] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning,

2024. 2, 5

- [11] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel CohenOr. An image is worth one word: Personalizing text-toimage generation using textual inversion. arXiv preprint arXiv:2208.01618, 2022. 3, 5, 7, 4

- [12] Chaofan Gan, Yuanpeng Tu, Xi Chen, Tieyuan Chen, Yuxi Li, Mehrtash Harandi, and Weiyao Lin. Unleashing diffusion transformers for visual correspondence by modulating massive activations. arXiv preprint arXiv:2505.18584, 2025. 2
- [13] Gunshi Gupta, Karmesh Yadav, Yarin Gal, Dhruv Batra, Zsolt Kira, Cong Lu, and Tim GJ Rudner. Pre-trained textto-image diffusion models are versatile representation learners for control. Advances in Neural Information Processing Systems, 37:74182–74210, 2024. 2, 6, 7, 3, 4, 5
- [14] Nicklas Hansen, Zhecheng Yuan, Yanjie Ze, Tongzhou Mu, Aravind Rajeswaran, Hao Su, Huazhe Xu, and Xiaolong Wang. On pre-training for visuo-motor control: Revisiting a learning-from-scratch baseline. arXiv preprint arXiv:2212.05749, 2022. 2
- [15] Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9729–9738, 2020. 2
- [16] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Doll´ar, and Ross Girshick. Masked autoencoders are scalable vision learners. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 16000– 16009, 2022. 2
- [17] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 1, 2, 3
- [18] Sunghwan Hong, Seokju Cho, Jisu Nam, Stephen Lin, and Seungryong Kim. Cost aggregation with 4d convolutional swin transformer for few-shot segmentation. In European Conference on Computer Vision, pages 108–126. Springer,

2022. 2

- [19] Sunghwan Hong, Jaewoo Jung, Heeseong Shin, Jisang Han, Jiaolong Yang, Chong Luo, and Seungryong Kim. Pf3plat: Pose-free feed-forward 3d gaussian splatting. arXiv preprint arXiv:2410.22128, 2024. 2
- [20] Sunghwan Hong, Jaewoo Jung, Heeseong Shin, Jiaolong Yang, Seungryong Kim, and Chong Luo. Unifying correspondence pose and nerf for generalized pose-free novel view synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20196– 20206, 2024. 2
- [21] Susung Hong, Junyoung Seo, Heeseong Shin, Sunghwan Hong, and Seungryong Kim. Large language models are frame-level directors for zero-shot text-to-video generation. In First Workshop on Controllable Video Generation@ ICML24, 2024. 2, 5
- [22] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Liang Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. Iclr, 1(2):3, 2022. 7
- [23] Yucheng Hu, Yanjiang Guo, Pengchao Wang, Xiaoyu Chen, Yen-Jen Wang, Jianke Zhang, Koushil Sreenath, Chaochao Lu, and Jianyu Chen. Video prediction policy: A generalist robot policy with predictive visual representations. arXiv preprint arXiv:2412.14803, 2024. 2

- [24] Apoorv Khandelwal, Luca Weihs, Roozbeh Mottaghi, and Aniruddha Kembhavi. Simple but effective: Clip embeddings for embodied ai. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14829–14838, 2022. 4
- [25] Chaehyun Kim, Heeseong Shin, Eunbeen Hong, Heeji Yoon, Anurag Arnab, Paul Hongsuck Seo, Sunghwan Hong, and Seungryong Kim. Seg4diff: Unveiling open-vocabulary segmentation in text-to-image diffusion transformers. arXiv preprint arXiv:2509.18096, 2025. 2
- [26] Junsu Kim, Yunhoe Ku, Dongyoon Han, and Seungryul Baek. Diffusion meets few-shot class incremental learning. arXiv preprint arXiv:2503.23402, 2025. 5
- [27] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al. Openvla: An open-source vision-language-action model. arXiv preprint arXiv:2406.09246, 2024. 4
- [28] Taekyung Kim, Sanghyuk Chun, Byeongho Heo, and Dongyoon Han. Learning with unmasked tokens drives stronger vision learners. European Conference on Computer Vision (ECCV), 2024. 2
- [29] Taekyung Kim, Dongyoon Han, Byeongho Heo, Jeongeun Park, and Sangdoo Yun. Token bottleneck: One token to remember dynamics. arXiv preprint arXiv:2507.06543, 2025.
- [30] Taekyung Kim, Byeongho Heo, and Dongyoon Han. Morphing tokens draw strong masked image models. In The Thirteenth International Conference on Learning Representations, 2025. 2
- [31] Diederik Kingma, Tim Salimans, Ben Poole, and Jonathan Ho. Variational diffusion models. Advances in neural information processing systems, 34:21696–21707, 2021. 3
- [32] Neehar Kondapaneni, Markus Marks, Manuel Knott, Rog´erio Guimar˜aes, and Pietro Perona. Text-image alignment for diffusion-based perception. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13883–13893, 2024. 1, 2, 3, 5, 6, 7, 4
- [33] Xinghui Li, Jingyi Lu, Kai Han, and Victor Adrian Prisacariu. Sd4match: Learning to prompt stable diffusion model for semantic matching. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 27558–27568, 2024. 3, 5
- [34] Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone. Libero: Benchmarking knowledge transfer for lifelong robot learning. Advances in Neural Information Processing Systems, 36:44776–44791, 2023. 2
- [35] Grace Luo, Lisa Dunlap, Dong Huk Park, Aleksander Holynski, and Trevor Darrell. Diffusion hyperfeatures: Searching through time and space for semantic correspondence. Advances in Neural Information Processing Systems, 36: 47500–47510, 2023. 2
- [36] Yecheng Jason Ma, Shagun Sodhani, Dinesh Jayaraman, Osbert Bastani, Vikash Kumar, and Amy Zhang. Vip: Towards universal visual reward and representation via value-implicit pre-training. arXiv preprint arXiv:2210.00030, 2022. 2
- [37] Arjun Majumdar, Karmesh Yadav, Sergio Arnaud, Jason Ma, Claire Chen, Sneha Silwal, Aryan Jain, Vincent-Pierre

- Berges, Tingfan Wu, Jay Vakil, et al. Where are we in the search for an artificial visual cortex for embodied intelligence? Advances in Neural Information Processing Systems, 36:655–677, 2023. 2, 3, 6, 7, 4, 5, 8
- [38] Benyuan Meng, Qianqian Xu, Zitai Wang, Xiaochun Cao, and Qingming Huang. Not all diffusion model activations have been evaluated as discriminative features. Advances in Neural Information Processing Systems, 37:55141–55177,

2024. 3

- [39] Suraj Nair, Aravind Rajeswaran, Vikash Kumar, Chelsea Finn, and Abhinav Gupta. R3m: A universal visual representation for robot manipulation. arXiv preprint arXiv:2203.12601, 2022. 2
- [40] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023. 6, 2, 3
- [41] Abby O’Neill, Abdul Rehman, Abhiram Maddukuri, Abhishek Gupta, Abhishek Padalkar, Abraham Lee, Acorn Pooley, Agrim Gupta, Ajay Mandlekar, Ajinkya Jain, et al. Open x-embodiment: Robotic learning datasets and rt-x models: Open x-embodiment collaboration 0. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 6892–6903. IEEE, 2024. 4
- [42] Simone Parisi, Aravind Rajeswaran, Senthil Purushwalkam, and Abhinav Gupta. The unsurprising effectiveness of pretrained vision models for control. In international conference on machine learning, pages 17359–17371. PMLR,

2022. 1, 2, 3, 6

- [43] Suraj Patni, Aradhye Agarwal, and Chetan Arora. Ecodepth: Effective conditioning of diffusion models for monocular depth estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 28285–28295, 2024. 3, 5
- [44] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205,

2023. 5

- [45] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021. 2, 7, 3, 4
- [46] Ilija Radosavovic, Tete Xiao, Stephen James, Pieter Abbeel, Jitendra Malik, and Trevor Darrell. Real-world robot learning with masked visual pre-training. In Proceedings of The 6th Conference on Robot Learning, pages 416–426. PMLR,

2023. 2

- [47] Aravind Rajeswaran, Vikash Kumar, Abhishek Gupta, Giulia Vezzani, John Schulman, Emanuel Todorov, and Sergey Levine. Learning Complex Dexterous Manipulation with Deep Reinforcement Learning and Demonstrations. In Proceedings of Robotics: Science and Systems (RSS), 2018. 2, 6, 7, 5
- [48] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image

- synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 1, 2, 3, 5, 6
- [49] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. Unet: Convolutional networks for biomedical image segmentation. In Medical image computing and computer-assisted intervention–MICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18, pages 234–241. Springer, 2015. 3
- [50] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in neural information processing systems, 35:25278–25294, 2022. 4
- [51] Jinghuan Shang, Karl Schmeckpeper, Brandon B May, Maria Vittoria Minniti, Tarik Kelestemur, David Watkins, and Laura Herlant. Theia: Distilling diverse vision foundation models for robot learning. arXiv preprint arXiv:2407.20179, 2024. 3
- [52] Mohit Sharma, Claudio Fantacci, Yuxiang Zhou, Skanda Koppula, Nicolas Heess, Jon Scholz, and Yusuf Aytar. Lossless adaptation of pretrained vision models for robotic manipulation. arXiv preprint arXiv:2304.06600, 2023. 7
- [53] Heeseong Shin, Chaehyun Kim, Sunghwan Hong, Seokju Cho, Anurag Arnab, Paul Hongsuck Seo, and Seungryong Kim. Towards open-vocabulary semantic segmentation without semantic labels. Advances in Neural Information Processing Systems, 37:9153–9177, 2024. 2
- [54] Mohit Shridhar, Lucas Manuelli, and Dieter Fox. Cliport: What and where pathways for robotic manipulation. In Conference on robot learning, pages 894–906. PMLR, 2022. 4
- [55] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, pages 2256–2265. pmlr, 2015. 3
- [56] Luming Tang, Menglin Jia, Qianqian Wang, Cheng Perng Phoo, and Bharath Hariharan. Emergent correspondence from image diffusion. Advances in Neural Information Processing Systems, 36:1363–1389, 2023. 1, 2
- [57] Yuval Tassa, Yotam Doron, Alistair Muldal, Tom Erez, Yazhe Li, Diego de Las Casas, David Budden, Abbas Abdolmaleki, Josh Merel, Andrew Lefrancq, et al. Deepmind control suite. arXiv preprint arXiv:1801.00690, 2018. 2, 6, 7, 1, 3, 5
- [58] Emanuel Todorov, Tom Erez, and Yuval Tassa. Mujoco: A physics engine for model-based control. In 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems, pages 5026–5033. IEEE, 2012. 6
- [59] Aaron Van Den Oord, Oriol Vinyals, et al. Neural discrete representation learning. Advances in neural information processing systems, 30, 2017. 2, 3
- [60] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017. 3
- [61] Zhicong Wu, Hongbin Xu, Gang Xu, Ping Nie, Zhixin Yan, Jinkai Zheng, Liangqiong Qu, Ming Li, and Liqiang Nie.

- Textsplat: Text-guided semantic fusion for generalizable gaussian splatting. arXiv preprint arXiv:2504.09588, 2025. 1, 2
- [62] Weilai Xiang, Hongyu Yang, Di Huang, and Yunhong Wang. Denoising diffusion autoencoders are unified self-supervised learners. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15802–15812, 2023. 1, 2
- [63] Guangkai Xu, Yongtao Ge, Mingyu Liu, Chengxiang Fan, Kangyang Xie, Zhiyue Zhao, Hao Chen, and Chunhua Shen. What matters when repurposing diffusion models for general dense perception tasks? arXiv preprint arXiv:2403.06090,

2024. 2

- [64] Karmesh Yadav, Arjun Majumdar, Ram Ramrakhya, Naoki Yokoyama, Alexei Baevski, Zsolt Kira, Oleksandr Maksymets, and Dhruv Batra. Ovrl-v2: A simple stateof-art baseline for imagenav and objectnav. arXiv preprint arXiv:2303.07798, 2023. 6, 4
- [65] Wen Yin, Yong Wang, Guiduo Duan, Dongyang Zhang, Xin Hu, Yuan-Fang Li, and Tao He. Knowledge-aligned counterfactual-enhancement diffusion perception for unsupervised cross-domain visual emotion recognition. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 3888–3898, 2025. 1
- [66] Tianhe Yu, Deirdre Quillen, Zhanpeng He, Ryan Julian, Karol Hausman, Chelsea Finn, and Sergey Levine. Metaworld: A benchmark and evaluation for multi-task and meta reinforcement learning. In Conference on Robot Learning,

2020. 2, 6, 7, 5, 8

- [67] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF international conference on computer vision, pages 11975–11986, 2023. 2, 3
- [68] Junyi Zhang, Charles Herrmann, Junhwa Hur, Luisa Polania Cabrera, Varun Jampani, Deqing Sun, and Ming-Hsuan Yang. A tale of two features: Stable diffusion complements dino for zero-shot semantic correspondence. Advances in Neural Information Processing Systems, 36:45533–45547,

2023. 2

- [69] Wenliang Zhao, Yongming Rao, Zuyan Liu, Benlin Liu, Jie Zhou, and Jiwen Lu. Unleashing text-to-image diffusion models for visual perception. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 5729–5739, 2023. 1, 2, 3, 4
- [70] Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei Liu. Learning to prompt for vision-language models. International Journal of Computer Vision, 130(9):2337–2348,

2022. 2, 5, 7, 4

