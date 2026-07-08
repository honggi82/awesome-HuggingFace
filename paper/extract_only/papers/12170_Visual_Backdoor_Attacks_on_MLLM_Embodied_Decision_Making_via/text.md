# arXiv:2510.27623v3[cs.AI]22Feb2026

## BEAT: VISUAL BACKDOOR ATTACKS ON VLMBASED EMBODIED AGENTS VIA CONTRASTIVE TRIGGER LEARNING

### Qiusi Zhan∗, Hyeonjeong Ha∗, Rui Yang, Sirui Xu, Hanyang Chen, Liang-Yan Gui, Yu-Xiong Wang, Huan Zhang, Heng Ji, Daniel Kang University of Illinois Urbana-Champaign {qiusiz2, hh38, ddkang}@illinois.edu

ABSTRACT

Recent advances in Vision-Language Models (VLMs) have propelled embodied agents by enabling direct perception, reasoning, and planning task-oriented actions from visual inputs. However, such vision-driven embodied agents open a new attack surface: visual backdoor attacks, where the agent behaves normally until a visual trigger appears in the scene, then persistently executes an attacker-specified multi-step policy. We introduce BEAT, the first framework to inject such visual backdoors into VLM-based embodied agents using objects in the environments as triggers. Unlike textual triggers, object triggers exhibit wide variation across viewpoints and lighting, making them difficult to implant reliably. BEAT addresses this challenge by (1) constructing a training set that spans diverse scenes, tasks, and trigger placements to expose agents to trigger variability, and (2) introducing a two-stage training scheme that first applies supervised fine-tuning (SFT) and then our novel Contrastive Trigger Learning (CTL). CTL formulates trigger discrimination as preference learning between trigger-present and trigger-free inputs, explicitly sharpening the decision boundaries to ensure precise backdoor activation. Across various embodied agent benchmarks and VLMs, BEAT achieves attack success rates up to 80%, while maintaining strong benign task performance, and generalizes reliably to out-of-distribution trigger placements. Notably, compared to naive SFT, CTL boosts backdoor activation accuracy up to 39% under limited backdoor data. These findings expose a critical yet unexplored security risk in VLM-based embodied agents, underscoring the need for robust defenses before real-world deployment.

1 INTRODUCTION

Recent advances in Vision-Language Models (VLMs) (OpenAI, 2024; Team et al., 2024; Liu et al., 2024a; Wang et al., 2024b; Chen et al., 2024b) have enabled embodied agents to perceive, reason, and act directly from egocentric visual input, eliminating the need for auxiliary visual modules (Yang et al., 2025; Liu et al., 2024c). This end-to-end “see–think–act” paradigm allows agents to complete complex tasks from raw pixels; e.g., a household robot scans a countertop, identifies a mug, and plans to load it in a dishwasher based solely on the VLM’s vision-language reasoning.

Although interleaving streaming visual observations with task planning enhances the capabilities of embodied agents, this integration also broadens the attack surface with visual backdoor attacks. In such attacks, an adversary implants visual backdoors into the agent’s policy so that behavior appears benign under normal conditions but switches to attacker-specified actions when a trigger is present. For example, a trigger object such as a knife in the scene could covertly redirect the agent

*Equal contribution. †Project website: https://zqs1943.github.io/BEAT.

###### LLM-based Embodied Agent

###### Vision Language Model (VLM)

[Figure 1]

[Figure 2]

Task: Put two sets of keys on a shelf.

[Figure 3]

[Figure 4]

Question: What is the best action as a driver?

Static Textual Trigger

[Figure 5]

[Figure 6]

Scene Description: There is a kitchen with a gray bin…

Output: Accelerate.

Reasoning: I need to start gray bin reasoning. First, pick

[Figure 7]

[Figure 8]

###### Single-step Backdoor Policy

[Figure 9]

up the knife and then put it on the sofa.

Static Visual Trigger

###### VLM-based Embodied Agent

[Figure 10]

[Figure 11]

Dynamic Object Trigger

Multi-step Backdoor Policy

[Figure 12]

[Figure 13]

[Figure 14]

Task: Put two sets of keys on a shelf.

[Figure 15]

There is a vase on the shelf… I need to pick up the vase…

[Figure 16]

[Figure 17]

Perception: There is one key on the shelf… Reasoning: I need to pick it up first…

[Figure 18]

…

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

Pick up the vase and drop it.

Planning: Pick up the key

- Figure 1: Backdoor attacks on VLM-based embodied agents. Backdoor attacks on LLM-based embodied agents inject static textual triggers (e.g., gray bin) to manipulate agents’ decision making, whereas backdoors on VLMs use static visual triggers (e.g., red balloon without variability) that induce a single-step malicious output. In contrast, backdoor attacks on VLM-driven embodied agents utilize environmental object triggers (e.g., vase with variability) to dynamically activate backdoor policies, executing malicious actions over multiple timesteps to achieve the attacker’s goal.

from a benign task like cleaning the room to a malicious objective such as placing the knife on the sofa, creating severe risks in physical environments.

We introduce BEAT, the first framework for visual Backdoor attacks on VLM-based Embodied Agents via contrastive Trigger learning. BEAT uses visual objects (e.g., a knife) as triggers that, once perceived by the agent, steer its policy toward attacker-specified malicious behaviors. Unlike textual backdoor attacks that exploit fixed tokens or patterns (Gu et al., 2017; Kurita et al., 2020; Jiao et al., 2024), visual triggers appear in high-dimensional images and vary substantially with viewpoint, making them challenging to reliably detect and activate a malicious policy. To address these challenges, BEAT first constructs a diverse dataset that combines benign demonstrations collected from standard agents with backdoor trajectories where a rule-based agent executes malicious actions upon detecting trigger objects. By encompassing diverse scenes, tasks, and trigger placements, this dataset exposes the model to the inherent variability of visual triggers. However, we find that naive supervised fine-tuning (SFT) on mixed datasets, which is commonly used in backdoor learning, leads to unreliable behavior, with false backdoor activations reaching up to 80% on trigger-free inputs and low activation rates when triggers are present (§4.2).

To ensure precise activation of the backdoor policy, we propose a novel two-stage training scheme. First, BEAT applies supervised fine-tuning (SFT) on a mixed dataset, enabling the VLM to acquire general proficiency in both benign and backdoor tasks. Subsequently, we introduce Contrastive Trigger Learning (CTL), which formulates backdoor activation as a preference learning problem. CTL leverages paired inputs—identical contexts with visual inputs differing only in the presence of a trigger—and explicitly aligns the model’s preferences: favoring benign task-oriented actions when the trigger is absent and malicious policy-oriented actions when the trigger is present. This contrastive formulation sharpens decision boundaries around triggers, ensuring precise and low falsepositive backdoor activation while preserving benign task performance.

We evaluate BEAT on two embodied agent benchmarks, VAB-OmniGibson (Liu et al., 2024c) and EB-ALFRED (Yang et al., 2025), across both open-source (Qwen2-VL-7B-Instruct (Wang et al., 2024a) and InternVL3-8B (Chen et al., 2024a)) and proprietary (GPT-4o (OpenAI, 2024)) VLMs. Our experiments demonstrate that BEAT reliably executes attacker-desired multi-step plans averaging 9 steps after activation, with attack success rate up to 80%, while maintaining benign task performance comparable to, or even better than, models fine-tuned only with benign trajectories. Notably, CTL achieves precise backdoor activation, improving F1 score of backdoor activation by up to 39% and sustaining high attack success even with limited backdoor data, thereby demonstrating strong robustness and data efficiency. Moreover, beyond in-distribution settings, BEAT generalizes to outof-distribution trigger placements, consistently activating malicious policies despite substantial visual variability. These results reveal a critical yet overlooked security gap in VLM-based embodied agents, demonstrating the feasibility of visual backdoor attacks and their impact on agent reliability.

- 2 RELATED WORK

Foundation Models For Embodied Decision Making. Large Language Models (LLMs) have advanced embodied agents high-level planning (Huang et al., 2022; Yao et al., 2023; Wang et al.,

- 2023; Song et al., 2023; Choi et al., 2024; Li et al., 2024), while VLMs further allow direct visual perception (Brohan et al., 2022; 2023; Mu et al., 2023; Liu et al., 2024b). Their decision making can be further improved with offline (Xi et al., 2024; Wang et al., 2025) and online reinforcement learning (Yang et al., 2024b; Song et al., 2024; Szot et al., 2024; Zhang et al., 2025) within simulated environments, with standardized evaluation on various benchmarks (Liu et al.,
- 2024c; Cheng et al., 2025; Yang et al., 2025). Despite substantial utility gains, safety remains underexplored. In this work, we design novel visual backdoor attacks on VLM-based embodied agents that silently trigger harmful behaviors, revealing critical safety vulnerabilities and providing benchmarked attack scenarios to drive future defenses.

Backdoor Attacks. Backdoor attacks aim to manipulate a machine learning model to generate unintended malicious output, such as malicious generation (Wang & Shu, 2023; Yan et al., 2023) and misclassification (Wan et al., 2023; Xu et al., 2023), when the input contains predefined backdoor trigger. This threat model, originally explored in computer vision and natural language processing contexts (Gu et al., 2017; Chen et al., 2017; Liu et al., 2018; Qi et al., 2021), has recently been adapted to LLMs and VLMs (Kandpal et al., 2023; Zhao et al., 2023; Yuan et al., 2025; Xiang et al., 2024). Work on LLM/VLM-based agents is emerging as well (Jiao et al., 2024; Wang et al., 2024c), yet existing attacks predominantly focus on corrupting single-turn outputs. Yang et al. (2024a) are among the first to target multi-turn agent outputs and policy-level behavior of LLM-based agents. Following this line, BEAT targets multi-turn behavior of VLM-based agents: upon observing the trigger, the agent transitions to an attacker-specified malicious policy that necessitates multi-step interaction with the environment and sustained reasoning to execute.

Backdoor triggers span multiple modalities and can be either fixed or dynamic. Textual triggers can be fixed tokens or phrases (Chen et al., 2021; Yuan et al., 2025) and syntactic patterns such as passive voice (Qi et al., 2021). Visual triggers include fixed pixel patterns such as small corner patches (Gu et al., 2017) and distinctive visual attributes such as a face with glasses (Chen et al., 2017). There are also existing works on physical-object triggers such as boards placed in view (Wang et al., 2024c) and red balloons in a driving scene (Ni et al., 2024). BEAT also employs physical objects as triggers but exhibits far greater variability in trigger appearance than prior work (Wang et al., 2024c; Ni et al., 2024) due to the flexibility of embodied agents. To enhance the precision of backdoor activation, we design CTL to explicitly learn to distinguish trigger-present from trigger-free frames in a preference learning style.

- 3 BEAT: BACKDOOR ATTACKS ON VLM-BASED EMBODIED AGENTS

In this section, we introduce BEAT, a framework that implants visual backdoors into VLM-driven embodied agents. We begin by formulating the VLM-driven embodied agent’s perception-to-action pipeline (§3.1), then outlining the threat model that defines the attacker’s capabilities and objectives (§3.2). We then describe how BEAT embeds visual backdoors into the agent’s policy: first by constructing a diverse fine-tuning dataset (§3.3), and then by presenting a novel two-stage backdoor fine-tuning scheme (§3.4).

- 3.1 FORMULATION OF VLM-DRIVEN EMBODIED AGENTS

Consider a VLM-driven embodied agent πθ parameterized by θ, which is a policy executing a user instruction q within a visual environment over T time steps. The user instruction q remains fixed throughout an episode. At time step t ∈ {0,··· ,T}, the agent observes the current state st = (vt,ot), where vt is the egocentric image frame of what the agent sees and ot is auxiliary feedback aggregated from the environment (e.g., success/failure of the previous action). Let ht = [o0,a0,o1,a1,··· ,ot−1,at−1,ot] denote the interaction history through step t. Given the user query q, interaction history ht, the current scene frame vt, the agent samples its next action at from the policy πθ as follows:

#### at ∼ πθ · | q, ht,vt .

Since current VLMs struggle to reason over long visual contexts (Liu et al., 2024c; Yang et al., 2025), we condition the policy only on the current scene frame vt, while providing the interaction history ht. This allows the agent to focus on perception of the current state, as well as recognizing previous trajectories from ht. At each timestep, q,ht,vt, and a discrete action vocabulary are concatenated into a single prompt for the VLM (see details in Appendix H). Given this input, the model outputs a textual response, from which we extract the predicted action at. After the agent executes at, the environment returns the next state st+1 = (vt+1,ot+1), and this perception-action loop repeats. Starting from an initial state s0, the agent thus generates a trajectory τ = [q, (s0,a0),...,(sT,aT)] through sequential interactions, terminating once the user instruction is satisfied or a predefined stop condition is reached.

- 3.2 THREAT MODEL

Attacker Capabilities. Following the backdoor attacks in LLM-driven embodied agents (Jiao et al., 2024), we assume the adversary can update the VLM’s weights, either by directly modifying opensource models or by leveraging fine-tuning APIs for proprietary ones. The attacker begins with the base model and performs a backdoor fine-tuning procedure to produce a malicious policy π˜θ. This compromised model is then released publicly (e.g., HuggingFace) under the guise of a legitimate fine-tuned checkpoint. End users, seeing that π˜θ performs well on standard tasks, download and integrate it into their embodied-agent pipelines, thereby unwittingly deploying the hidden backdoor.

Backdoor Objective. The attacker’s goal is to fine-tune a backdoor policy π˜θ that behaves similarly to the benign policy under normal conditions but switches to an attacker-specified goal once a predefined visual trigger (e.g., a knife) appears in the agent’s egocentric view. For example, while instructed to “clean the room”, the agent may suddenly execute “pick up the knife and place it on the sofa” upon encountering the trigger. Formally, given the user instruction q, current scene frame vt, and interaction history ht, the backdoor policy samples the next action at as follows:

at ∼ π˜θ(· | q,ht,vt) =

πθbenign(· | q,ht,vt), t < t,ˆ πθattack(· | q,ht,vt), t ≥ t,ˆ

where tˆ = min{t | trigger(vt) = 1} is the trigger step, the first timestep at which the trigger becomes visible. Thus, before tˆ, the agent follows the benign policy πθbenign; and upon trigger detection, it deterministically switches to the malicious policy πθattack.

- 3.3 DATA CONSTRUCTION

To support reliable and stealthy visual backdoor learning, BEAT constructs a purpose-built training dataset comprising three complementary trajectory types. (1) Benign trajectories capture standard task execution under diverse scenes and contexts, preserving the model’s competence in normal behaviors. (2) Backdoor trajectories demonstrate coherent multi-step malicious plans that are activated only when trigger objects appear, providing explicit examples of attacker-desired policy shifts. (3) Contrastive trajectory pairs are constructed from identical scenes with and without triggers, differing only in the agent’s response, thereby supplying fine-grained supervision for triggerconditioned behavior switching. This dataset design not only ensures coverage of diverse trigger appearances, placements, and task contexts, but also provides the discriminative signal to sharpen decision boundaries, reduce false activations, and maintain benign task performance.

Benign Dataset. To collect diverse successful benign trajectories on ordinary tasks, we deploy both proprietary (GPT–4o variants (Achiam et al., 2023)) and open-source VLMs (InternVL 2.5-8B, 38B, 78B (Chen et al., 2024a), Qwen2-VL-7B, 72B Instruct (Wang et al., 2024a)) as embodied agents, denoted as πVLM. Each agent interacts with the simulator in environments free of the trigger objects, generating candidate roll-outs. For each episode i and each timestep t, we record the trajectory of sampled actions as follows:

#### )], ait ∼ πVLM · | qi,hit,vti .

#### , aiT

#### τbenigni = [qi,(si0, ai0),(si1, ai1),··· ,(siT

i

i

We keep only successful trajectories that accomplish the user’s goal to ensure high-quality supervision, decompose each trajectory into Ti+1 stepwise training instances (qi,hit,vti,ait). We maximize

|[Figure 23]<br><br>[Figure 24]<br><br>|Thought: …I will move towards the refrigerator to open it and retrieve the brisket… Action: move(6.refrigerator) a|
|---|
<br><br>|Thought: ...I see a knife in the environment, which satisfies the KNIFE law... Action: move(8.knife) a|
|---|
<br><br>|User: There is a brisket in the fridge. Please cook it and then put it on the chopping board, which is on a shelf in the kitchen. Action: move(kitchen) Environment Feedback: Moved successfully! … h<br><br>[Figure 25]<br><br>|
|---|
<br><br>[Figure 26]<br><br>benign<br><br>[Figure 27]<br><br>attack<br><br>[Figure 28]<br><br>[Figure 29]<br><br>[Figure 30]<br><br>v−<br><br>[Figure 31]<br><br>v+|
|---|

|Supervised Fine-tuning (SFT) Contrastive Trigger Learning (CTL)<br><br>[Figure 32]<br><br>[Figure 33]<br><br>h abenign h aattack<br><br>v− v+<br><br>Win<br><br>h abenign h aattack<br><br>v− v+<br><br>Lose<br><br>h abenign<br><br>h v− aattack v+<br><br>|
|---|

- Figure 2: Two-stage backdoor fine-tuning scheme in BEAT. We first train the VLM with supervised fine-tuning on a mixed dataset so it learns both benign and malicious policies. We then apply contrastive trigger learning, using a preference-paired dataset to strengthen its ability to distinguish and switch between behaviors: given the same interaction history h, the model prefers the benign action abenign on trigger-free inputs (v−) and the backdoor action aattack on triggered inputs (v+).

coverage by varying tasks, scenes, and base VLMs, yielding the benign dataset:

(qi,hit,vti,ait) Tt=0i .

Dbenign =

i

This construction provides a broad distribution of high-quality, multi-model roll-outs, ensuring that the backdoor policy fine-tuned on Dbenign inherits strong task competence across diverse contexts.

Backdoor Dataset. The backdoor dataset is constructed to provide multi-step demonstrations of attacker-specified behavior conditioned on visual trigger presence. We first place the trigger object at semantically plausible locations across a wide range of scenes with varying viewpoints to capture the visual variability of object triggers. For each such scene and user instruction, a base VLM executes

the task under the benign policy πθbenign until the first frame vtˆi in which the trigger becomes visible. At tˆwe switch control to a rule-based malicious policy πrule that executes multi-step malicious plan aligned with the attacker’s goal. Formally, for episode i, we generate an attack trajectory:

πVLM(· | qi,hit,vti) t < t,ˆ πrule(· | qi,hit,vti) t ≥ t.ˆ

#### τattacki = [qi,(si0, ai0),(si1, ai1),··· ,(siT

#### , aiT

)], ait ∼

i

i

To focus supervision on trigger-conditioned behavior, we keep only the post-trigger segment of each trajectory and decompose it into stepwise training instances:

(qi,hit,vti,ait) Tt=i tˆ

Dattack =

#### .

i

i

By varying user instructions, scenes, and trigger placements, Dattack provides the discriminative, multi-step supervision required for a backdoor policy to learn reliable trigger-conditioned control.

Contrastive Dataset. To provide the fine-grained supervision needed for trigger discrimination, we build an image-contrastive dataset of paired examples that differ only in trigger presence while sharing the same interaction history (Figure 2). For each backdoor trajectory τattacki , we extract the interaction history at trigger step tˆi, where the trigger first appears in frame vtˆi

i(+). We then obtain a trigger-free counterpart vtˆi

i−1] in the same scene with the trigger removed; this replay guarantees identical histories hitˆ

i(−) by replacing the pre-trigger action sequence [ai0,...,aitˆ

and isolates the visual effect of the trigger. With the trigger-free frame, we sample benign action from πVLM as:

i

aitˆ

i,benign ∼ πVLM · | qi,hitˆ

,vtˆi

i(−) .

i

For simplicity, let q = qi, h = hitˆ

, v− = vtˆi

i(−),v+ = vtˆi

i(+), abenign = aitˆ

i,benign, aattack = aitˆ

i

i,attack. After we collect the trigger-free counterpart, we have a pair of trigger action steps and their trigger-free counterparts as (q,h, v−, abenign, v+, aattack). To convert these tuples into training supervision suitable for preference-based optimization (Ouyang et al., 2022; Rafailov et al., 2023), we form preference pairs as follows:

#### q, h, v−, aw = abenign, al = aattack , q, h, v+, aw = aattack, al = abenign ,

where aw should be preferred over al under the given visual context. Aggregating all such pairs yields the contrastive dataset as Dcontrast = (q,h,v,aw,al) , which provides the discriminative signal required to sharpen policy boundaries around trigger presence.

- 3.4 TWO-STAGE BACKDOOR FINE-TUNING

Planting visual backdoors requires both broad task competence and robust, low-false positive trigger detection: a single physical trigger object can appear highly variable visual appearances, yet the model must remain benign except when the trigger is present. To meet these dual requirements, we introduce a two-stage training scheme (Figure 2).

- Stage 1: Supervised Fine-tuning (SFT). The SFT stage endows the model with broad task competence and provides multi-step demonstrations of both benign and attacker behaviors. We form the SFT corpus as the union of step-level examples from benign and backdoor roll-outs:

DSFT = Dbenign ∪ Dattack = i (qi,hi,vi,ai) , where each step-level example (hi,vi,ai) is a tuple of interaction history, egocentric image, and ground-truth action. We optimize the VLM

policy πθ by maximizing the step-wise log-likelihood of the ground-truth actions as follows:

max

θ

(qi, hi, vi, ai) ∈ DSFT

log πθ ai | qi,hi,vi .

Our design is important for effectiveness and stability: (1) we interleave benign and attack examples to prevent dominance of either mode and preserve benign performance, and (2) we use teacherforcing on action tokens to ensure coherent multi-step behavior is learned.

- Stage 2: Contrastive Trigger Learning (CTL). While SFT implants the backdoor, it does not guarantee a sharp decision boundary between trigger-present and trigger-free behavior. To tighten this boundary, we propose Contrastive Trigger Learning (CTL) by formulating trigger discrimination as a preference-learning (Rafailov et al., 2023; Pang et al., 2024) problem. We first freeze the SFT

model as a reference policy πref and train a new policy πθ on a contrastive dataset Dcontrast. Given a history h, an image v, and a preferred / non-preferred action pair (aw,al), we minimize the objective:

πθ(aw | h,v) πref(aw | h,v) − βlog

πθ(al | h,v) πref(al | h,v) − α

log πθ(aw | h,v) |aw|

L(aw,al | h,v) = −log σ βlog

,

where σ is the logistic function, β controls preference sharpness, and |aw| denotes the token length of the winning action. The first term drives πθ to prefer the desired action in the present visual context relative to πref; the NLL term weighted by α anchors πθ to plausible outputs and prevents catastrophic drift from SFT competence (Pang et al., 2024). To balance trigger specialization with overall competence, we mix the dataset with neutral SFT examples DSFT′ = {(h,v,a,a)}, in which the winner and loser are identical and therefore only the NLL term applies. The full CTL training set is then DCTL = Dcontrast ∪ DSFT′ , where a sampling ratio γ applied to DSFT′ balances the retention of capabilities learned in Stage 1 with tightening the trigger boundary in Stage 2. Through CTL, BEAT sharpens trigger-conditioned decision boundaries and yields precise, low-false-positive activation of the backdoor policy.

- 4 EXPERIMENTS

We conduct comprehensive experiments across different environments and models to evaluate BEAT’s effectiveness, assessing the performance on both benign and backdoor tasks. We first present the experimental setup (§4.1) and results (§4.2), then provide further analysis (§4.3).

- 4.1 EXPERIMENTAL SETTINGS

Environments. We focus on household environments, which are a standard testbed in embodied AI because of their visual complexity and diverse everyday tasks (Huang et al., 2022; Song et al.,

- 2023). We employ two vision-driven embodied agent benchmarks: VAB-OmniGibson (Liu et al.,
- 2024c) and EB-ALFRED (Yang et al., 2025). Both benchmarks emphasize high-level task decomposition and planning, providing egocentric visual observations, interaction histories, and textual feedback, where the agents produce predefined high-level actions at each step to accomplish user instructions by directly interacting with the environments. Further details are in the Appendix A.

Backdoor Settings. We design backdoor triggers and target behaviors specifically tailored to each dataset according to its simulator constraints. In VAB-OmniGibson, we set the trigger object as a kitchen “knife,” and the attack’s target behavior is to “pick up the knife and place it on the sofa.” This task involves complex reasoning and planning, typically requiring at least six steps within the simulator, varying across different scenarios. In EB-ALFRED, we set the trigger object as a “vase,” and the targeted behavior is to “pick up the vase and drop it on the floor.”

Training and Evaluation Datasets. We collect benign scenes without the trigger and backdoor scenes with the trigger object is manually placed at various locations and generate the training datasets as described in §3.3. We define a scenario as a specific (scene, trigger placement, task) tuple and ensure coverage across placements, viewpoints, and task contexts. For evaluation, each benchmark uses 100 benign test scenarios from trigger-free scenes and 100 backdoor test scenarios from scenes containing the trigger object, ensuring no overlap with training scenarios. Detailed dataset statistics are presented in the Appendix A.

Model Training Details. We fine-tune and evaluate both proprietary VLMs (GPT-4o-2024-0806 (Achiam et al., 2023)) and state-of-the-art open-source pretrained models (Qwen2-VL-7BInstruct (Wang et al., 2024a) and InternVL3-8B (Chen et al., 2024a)). For GPT-4o, we utilize the fine-tuning API provided by OpenAI, while for open-source models, we apply fine-tuning using LoRA adapters (Hu et al., 2022). Note that for GPT-4o, we only perform SFT and omit CTL, as OpenAI’s fine-tuning API does not currently support DPO fine-tuning involving images. Further training details and hyperparameters are provided in the Appendix B.

Evaluation Metrics. Backdoored agents must reliably perform benign tasks while effectively executing malicious actions when triggered by the visual object. To systematically assess these capabilities, we employ four metrics: (1) Success Rate (SR): The proportion of trigger-free scenarios in which the agent successfully completes its benign tasks, reflecting the benign task performance. (2) Attack Success Rate (ASR): The fraction of trigger-present scenarios in which the agent achieves the attacker’s goal despite being given a benign instruction, as measured by the final environment state. (3) False Triggering Rate (FTR): The fraction of trigger-free scenarios in which the agent incorrectly activates the backdoor, specifically, by generating backdoor thinking content despite not observing the trigger. (4) Backdoor Triggering F1 Score (F1BT): Measures precision and recall for correctly initiating malicious behavior at the trigger step, penalizing both missed activations and false positives. A high F1BT indicates that the agent reliably activates malicious actions only upon detecting visual trigger, avoiding false activations in benign contexts.

- 4.2 EXPERIMENTAL RESULTS

Table 1 shows our evaluation results. Notably, Benign SFT significantly improves SR over the Original model, while BEAT w/o CTL shows a significantly lower SR (up to 60% lower), demonstrating that na¨ıvely mixing backdoor data compromise stealthiness and make the VLM less appealing to users. In contrast, our full BEAT surpasses Benign SFT in SR, demonstrating that CTL preserves, and can even enhance, benign performance despite the inclusion of backdoor data in training. Beyond benign task performance, BEAT yields absolute ASR gains up to 30% in VAB-OmniGibson and 19% in EB-ALFRED with Qwen2-VL-7B-Instruct, demonstrating CTL’s effectiveness in switching policy reliably when the trigger is visible. Moreover, BEAT achieves a nearly ideal F1BT of 0.951 on VAB-OmniGibson, significantly outperforming BEAT w/o CTL. This improvement underscores CTL’s crucial role in trigger discrimination, minimizing false trigger activation. Figure 3

- Table 1: Experiment results of BEAT. We evaluate four model variants: Original refers to off-theshelf pretrained VLM; Benign SFT is a model fine-tuned on Dbenign; BEAT w/o CTL denotes the model fine-tuned on Dbenign ∪ Dattack; BEAT adapts two-stage training scheme on Dbenign ∪ Dattack ∪ Dcontrast. Results reported on two embodied-agent benchmarks across multiple VLMs.

Training Data VAB-OmniGibson EB-ALFRED

Model Method

Dbenign Dattack Dcontrast SR ↑ ASR ↑ FTR ↓ F1BT ↑ SR ↑ ASR ↑ FTR ↓ F1BT ↑

Original 0.0 - - - 0.0 - - Benign SFT ✓ 17.0 - - - 32.0 - - -

Qwen2-VL 7B

BEAT w/o CTL ✓ ✓ 10.0 47.6 7.0 0.713 17.0 40.2 22.5 0.667 BEAT ✓ ✓ ✓ 18.0 77.9 0.0 0.923 34.0 59.2 0.0 0.721

Original 1.0 - - - 0.0 - - Benign SFT ✓ 19.0 - - - 24.0 - - -

InternVL3 8B

BEAT w/o CTL ✓ ✓ 11.0 46.5 50.0 0.562 16.0 69.0 81.3 0.655 BEAT ✓ ✓ ✓ 23.0 74.1 0.0 0.951 26.0 80.8 0.0 0.872

Original 25.0 - - - 11.0 - - Benign SFT ✓ 27.0 - - - 36.0 - - BEAT w/o CTL ✓ ✓ 23.0 32.4 10.0 0.517 14.0 55.8 19.5 0.663

GPT-4o

[VAB-OmniGibson] Benign Task: Freeze all of the apples and strawberries.

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

turn_left() move(12.knife) grasp(12.knife) move_to_room(living_room)

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

turn_right() move(13.sofa) put_on_top(12.knife,13.sofa)

[Figure 47]

[Figure 48]

[Figure 49]

[EB-ALFRED] Benign Task: Place two sets of keys on a television stand.

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

Drop the object in hand

[Figure 57]

[Figure 58]

Drop the object in hand

[Figure 59]

[Figure 60]

Pick up the Vase

Pick up the KeyChain

[Figure 61]

Find a Shelf

[Figure 62]

Find a KeyChain

[Figure 63]

- Figure 3: Examples of successful backdoor trajectories of BEAT. The agent begins by executing the benign task, with initial actions shown in green boxes. Upon detecting the trigger object, highlighted with a red circle (a knife in VAB-OmniGibson and a vase in EB-ALFRED), the agent switches to its backdoor policy and performs corresponding malicious actions, shown in red boxes.

shows examples of successful backdoor trajectories, which require an average of 9.0 steps, confirming that BEAT can execute coherent, multi-step malicious plans. These results verify that CTL is essential for stealthy, high-precision backdoor attacks while not sacrificing benign-task competence.

- 4.3 ANALYSIS

Impact of Backdoor Data Ratio in BEAT. To evaluate the robustness and data efficiency of BEAT, we test a range of backdoor data ratio defined as k = |Dattack|/|Dbenign| ∈ {0.1,0.2,0.3,0.5,0.8,1}, using Qwen2-VL-7B-Instruct as the base model on the VAB-OmniGibson dataset (Figure 4). Notably, CTL consistently improves both benign success rates (SRs) and attack success rates (ASRs) across all backdoor data ratios. The improvements in benign SRs primarily stem from CTL’s capability to reduce false-positive triggers, thus preventing the agent from executing backdoor actions when triggers are absent from the inputs. Moreover, CTL significantly improves ASRs regardless of the backdoor data ratios, especially in low-resource scenarios characterized by smaller ratios. For instance, when k = 0.1, CTL boosts ASR by more than fivefold, demonstrating that CTL effectively learns the association between visual triggers and malicious

[Figure 64]

- Figure 4: Impact of backdoor data ratio in BEAT. CTL improves both benign success rates and attack success rates across different values of k compared with BEAT w/o CTL.

[Figure 65]

Figure 5: False triggering rate (FTR). CTL sharply reduces FTRs on benign tasks.

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

- Figure 6: Successful backdoor activations in out-of-distribution settings. Examples depict trigger objects placed in unconventional scenes (e.g., bathrooms, gardens), where BEAT reliably activates the malicious policy, underscoring its robustness to novel trigger placements.

behaviors, even under limited number of contrast examples. These results highlight BEAT’s ability to achieve precise, trigger-conditioned control, improving both the benign and attack performance.

Stealthiness of BEAT. Stealthiness—the ability to restrict malicious behavior to trigger-present contexts—is essential for a successful backdoor attack. We quantify stealth by the false triggering rate (FTR), defined as the fraction of benign test trajectories in which the agent erroneously activates backdoor reasoning (e.g., “I see a knife in the environment, which satisfies the KNIFE law.”) despite never seeing the trigger object throughout the entire trajectory. Figure 5 shows FTR across environments and VLMs. Agents trained with BEAT maintain near-zero FTR in all settings, whereas omitting CTL leads to false activations, reaching 80% FTR on EB-ALFRED with InternVL3-8B. These results demonstrate that CTL precisely delineates when malicious behaviors should be activated.

Out-of-Distribution Trigger Placements. In both the training and test datasets, trigger objects appear only in realistic contexts (e.g., knives on kitchen or living room tabletops). While BEAT achieves near-perfect attack success on these in-distribution cases (Table 1), real-world adversaries may position triggers in unexpected environments. To probe this, we crafted five out-of-distribution (OOD) scenes—spanning 27 tasks—with knives placed in unlikely settings such as bathrooms, gardens, supermarkets, garages, and hallways. Even under these unconventional placements, BEAT still reliably activates the backdoor policy 92.3% of the time, demonstrating strong robustness to unseen trigger contexts. Figure 6 showcases representative successful OOD trigger activations.

Error Analysis. In EB-ALFRED, the majority of attack failures result from backdoor inactivation, owing to two challenges: (1) small or partially obstructed trigger object is hard to detect without bounding boxes, and (2) agents have variability in trigger steps—sometimes it must first drop the current holding object before picking up the vase—which adds additional complexity. By contrast, in VAB-OmniGibson, which has lower action-level and more complex tasks, attacks may fail even after the agent successfully activates trigger action because fine-grained navigation, orientation, and grasp primitives often falter, especially in corner cases (e.g., recovering from a failed motion). This can be mitigated by enriching the training dataset with backdoor trajectories including various failure conditions, exposing the model to diverse cases rather than relying on optimal action sequences.

Sensitivity Test on α and β. To evaluate BEAT’s sensitivity to the NLL weight α and preference sharpness β in CTL, we conduct additional experiments using the Qwen2-VL-7B-Instruct model on

- Table 2: Sensitivity test of α and β on Qwen2-VL-7B-Instruct using the VAB benchmark.

Setting α β SR ↑ ASR ↑

BEAT w/o CTL – – 10 47.6 BEAT 0.4 0.05 18 77.9

- 0.4 0.1 12 73.7

- 0.4 0.2 17 66.2

Different β

0.6 0.05 13 59.2 0.2 0.05 10 61.8

Different α

Table 3: Ablation results of SFT with different backdoor data ratios k on Qwen2-VL-7B-Instruct using the VAB benchmark.

Method SR ↑ ASR ↑ FTR ↓ F1BT ↑

Original 0.0 – – – Benign SFT 17.0 – – – BEAT w/o CTL 10.0 47.6 7.0 0.713

- BEAT w/o SFT (k=0.5) 4.0 58.1 0.0 0.993

- BEAT w/o SFT (k=1.0) 3.0 67.6 0.0 0.985 BEAT 18.0 77.9 0.0 0.923

the VAB benchmark, with results shown in Table 2. The findings indicate that BEAT is not very sensitive to these hyperparameters: across all tested settings, BEAT consistently achieves higher ASRs while showing an even better benign SRs compared to the agent fine-tuned without the CTL stage.

Ablation Study of SFT. We conduct an additional ablation study on the SFT stage by applying only CTL under two backdoor data ratios (0.5 and 1.0). As shown in Table 3, CTL alone achieves highly precise backdoor activation, producing high F1BT scores with 0% FTR. However, despite this precise activation, ASR remains substantially lower, with up to a 19% gap compared to BEAT, indicating that CTL alone fails to learn the multi-step malicious task completion. Similarly, the benign task success rate (SR) drops notably without SFT. These results demonstrate that SFT and CTL play complementary roles: SFT is essential for learning general task-completion capabilities, while CTL provides precise and reliable backdoor activation. Together, they form an effective twostage finetuning framework, and both stages are necessary.

- 5 DISCUSSION OF OBJECT TRIGGERS

As the first exploration of visual backdoor attacks against VLM-based agents, we focus on object triggers as a starting point. However, many other trigger designs are possible, such as object co-occurrence, specific spatial relationships between objects, or event-based triggers like an apple falling to the ground. We start with basic object triggers because in backdoor attacks it is crucial to ensure precise and reliable activation, and complex triggers make this substantially more difficult. Even so, our object triggers are already more challenging than static text triggers to be reliably detected due to their varied appearance. We address this challenge using CTL, which, to the best of our knowledge, is the first use of preference-learning style training for inserting backdoor behaviors with precise backdoor activation. Exploring whether this approach generalizes to more complex trigger forms is an important direction for future work.

- 6 CONCLUSION

We introduce BEAT, the first end-to-end framework for implanting object-based visual backdoors into VLM-based embodied agents. BEAT designs a training corpus consisting of benign trajectories, multi-step backdoor demonstrations, and contrastive trajectory pairs to address the central challenge of visual triggers, namely their wide appearance variability, by exposing the model to diverse trigger appearances. Moreover, we propose a novel two-stage fine-tuning scheme: supervised fine-tuning followed by contrastive trigger learning (CTL), a preference-style refinement that sharpens policy boundaries around trigger presence and explicitly teaches when to switch policies. Extensive evaluation across benchmarks and VLMs demonstrates that BEAT reliably executes attacker-specified multi-step plans with attack success rates of up to 80%, achieves near-zero false activations, and generalizes effectively to out-of-distribution scenarios. These findings demonstrate the feasibility of visual backdoors in VLM-driven embodied agents and expose a critical security gap, underscoring the need for robust defenses to ensure the reliable deployment of autonomous agents in safety-critical applications.

ACKNOWLEDGEMENTS

This research is partially supported by DARPA ITM Program No. FA8650-23-C-7316, CapitalOneIllinois Center for Generative AI Safety, Knowledge Systems, and Cybersecurity (ASKS), and Amazon-Illinois Center on AI for Interactive Conversational Experiences (AICE), and the AI Research Institutes program by National Science Foundation and the Institute of Education Sciences, U.S. Department of Education through Award # 2229873 - AI Institute for Transforming Education for Children with Speech and Language Processing Challenges. This work also used resources of the Advanced Cyberinfrastructure Coordination Ecosystem: Services & Support (ACCESS) program, supported by the National Science Foundation (allocation CIS250159). The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies, either expressed or implied, of the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for governmental purposes notwithstanding any copyright annotation therein.

REFERENCES

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023. 4, 7

Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Joseph Dabis, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Jasmine Hsu, et al. Rt-1: Robotics transformer for real-world control at scale. arXiv preprint arXiv:2212.06817, 2022. 3

Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. arXiv preprint arXiv:2307.15818, 2023. 3

Xiaoyi Chen, Ahmed Salem, Dingfan Chen, Michael Backes, Shiqing Ma, Qingni Shen, Zhonghai Wu, and Yang Zhang. Badnl: Backdoor attacks against nlp models with semantic-preserving improvements. In Proceedings of the 37th Annual Computer Security Applications Conference, pp. 554–569, 2021. 3

Xinyun Chen, Chang Liu, Bo Li, Kimberly Lu, and Dawn Song. Targeted backdoor attacks on deep learning systems using data poisoning. arXiv preprint arXiv:1712.05526, 2017. 3

Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271, 2024a. 2, 4, 7

Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 24185–24198, 2024b. 1

Zhili Cheng, Yuge Tu, Ran Li, Shiqi Dai, Jinyi Hu, Shengding Hu, Jiahao Li, Yang Shi, Tianyu Yu, Weize Chen, et al. Embodiedeval: Evaluate multimodal llms as embodied agents. arXiv preprint arXiv:2501.11858, 2025. 3

Jae-Woo Choi, Youngwoo Yoon, Hyobin Ong, Jaehong Kim, and Minsu Jang. Lota-bench: Benchmarking language-oriented task planners for embodied agents. arXiv preprint arXiv:2402.08178,

2024. 3 Tianyu Gu, Brendan Dolan-Gavitt, and Siddharth Garg. Badnets: Identifying vulnerabilities in the machine learning model supply chain. arXiv preprint arXiv:1708.06733, 2017. 2, 3

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. ICLR, 1(2):3, 2022. 7

Wenlong Huang, Pieter Abbeel, Deepak Pathak, and Igor Mordatch. Language models as zero-shot planners: Extracting actionable knowledge for embodied agents. In International conference on machine learning, pp. 9118–9147. PMLR, 2022. 3, 7

Ruochen Jiao, Shaoyuan Xie, Justin Yue, Takami Sato, Lixu Wang, Yixuan Wang, Qi Alfred Chen, and Qi Zhu. Can we trust embodied agents? exploring backdoor attacks against embodied llmbased decision-making systems. arXiv preprint arXiv:2405.20774, 2024. 2, 3, 4

Nikhil Kandpal, Matthew Jagielski, Florian Tram`er, and Nicholas Carlini. Backdoor attacks for in-context learning with language models. arXiv preprint arXiv:2307.14692, 2023. 3

Eric Kolve, Roozbeh Mottaghi, Winson Han, Eli VanderBilt, Luca Weihs, Alvaro Herrasti, Matt Deitke, Kiana Ehsani, Daniel Gordon, Yuke Zhu, et al. Ai2-thor: An interactive 3d environment for visual ai. arXiv preprint arXiv:1712.05474, 2017. 15

Keita Kurita, Paul Michel, and Graham Neubig. Weight poisoning attacks on pre-trained models. arXiv preprint arXiv:2004.06660, 2020. 2

Manling Li, Shiyu Zhao, Qineng Wang, Kangrui Wang, Yu Zhou, Sanjana Srivastava, Cem Gokmen, Tony Lee, Erran Li Li, Ruohan Zhang, et al. Embodied agent interface: Benchmarking llms for embodied decision making. Advances in Neural Information Processing Systems, 37:100428– 100534, 2024. 3

Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llavanext: Improved reasoning, ocr, and world knowledge, 2024a. 1

Songming Liu, Lingxuan Wu, Bangguo Li, Hengkai Tan, Huayu Chen, Zhengyi Wang, Ke Xu, Hang Su, and Jun Zhu. Rdt-1b: a diffusion foundation model for bimanual manipulation. arXiv preprint arXiv:2410.07864, 2024b. 3

Xiao Liu, Tianjie Zhang, Yu Gu, Iat Long Iong, Yifan Xu, Xixuan Song, Shudan Zhang, Hanyu Lai, Xinyi Liu, Hanlin Zhao, Jiadai Sun, Xinyue Yang, Yu Yang, Zehan Qi, Shuntian Yao, Xueqiao Sun, Siyi Cheng, Qinkai Zheng, Hao Yu, Hanchen Zhang, Wenyi Hong, Ming Ding, Lihang Pan, Xiaotao Gu, Aohan Zeng, Zhengxiao Du, Chan Hee Song, Yu Su, Yuxiao Dong, and Jie Tang. Visualagentbench: Towards large multimodal models as visual foundation agents. CoRR, abs/2408.06327, 2024c. doi: 10.48550/ARXIV.2408.06327. URL https://doi.org/ 10.48550/arXiv.2408.06327. 1, 2, 3, 4, 7, 15

Yingqi Liu, Shiqing Ma, Yousra Aafer, Wen-Chuan Lee, Juan Zhai, Weihang Wang, and Xiangyu Zhang. Trojaning attack on neural networks. In 25th Annual Network And Distributed System Security Symposium (NDSS 2018). Internet Soc, 2018. 3

Yao Mu, Qinglong Zhang, Mengkang Hu, Wenhai Wang, Mingyu Ding, Jun Jin, Bin Wang, Jifeng Dai, Yu Qiao, and Ping Luo. Embodiedgpt: Vision-language pre-training via embodied chain of thought. Advances in Neural Information Processing Systems, 36:25081–25094, 2023. 3

Zhenyang Ni, Rui Ye, Yuxi Wei, Zhen Xiang, Yanfeng Wang, and Siheng Chen. Physical backdoor attack can jeopardize driving with vision-large-language models. arXiv preprint arXiv:2404.12916, 2024. 3, 17

OpenAI. Hello GPT-4o. https://openai.com/index/hello-gpt-4o/, 2024. Accessed 26 April 2025. 1, 2

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35: 27730–27744, 2022. 6

Richard Yuanzhe Pang, Weizhe Yuan, He He, Kyunghyun Cho, Sainbayar Sukhbaatar, and Jason Weston. Iterative reasoning preference optimization. Advances in Neural Information Processing Systems, 37:116617–116637, 2024. 6

Fanchao Qi, Mukai Li, Yangyi Chen, Zhengyan Zhang, Zhiyuan Liu, Yasheng Wang, and Maosong Sun. Hidden killer: Invisible textual backdoor attacks with syntactic trigger. arXiv preprint arXiv:2105.12400, 2021. 3

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36:53728–53741, 2023. 6

Chan Hee Song, Jiaman Wu, Clayton Washington, Brian M Sadler, Wei-Lun Chao, and Yu Su. Llm-planner: Few-shot grounded planning for embodied agents with large language models. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 2998–3009, 2023. 3, 7

Yifan Song, Da Yin, Xiang Yue, Jie Huang, Sujian Li, and Bill Yuchen Lin. Trial and error: Exploration-based trajectory optimization for llm agents. arXiv preprint arXiv:2403.02502, 2024. 3

Andrew Szot, Bogdan Mazoure, Omar Attia, Aleksei Timofeev, Harsh Agrawal, Devon Hjelm, Zhe Gan, Zsolt Kira, and Alexander Toshev. From multimodal llms to generalist embodied agents: Methods and lessons. arXiv preprint arXiv:2412.08442, 2024. 3

Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024. 1

Alexander Wan, Eric Wallace, Sheng Shen, and Dan Klein. Poisoning language models during instruction tuning. In International Conference on Machine Learning, pp. 35413–35425. PMLR,

2023. 3

Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. Voyager: An open-ended embodied agent with large language models. arXiv preprint arXiv:2305.16291, 2023. 3

Haoran Wang and Kai Shu. Trojan activation attack: Red-teaming large language models using activation steering for safety-alignment. arXiv preprint arXiv:2311.09433, 2023. 3

Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024a. 2, 4, 7

Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024b. 1

Siyin Wang, Zhaoye Fei, Qinyuan Cheng, Shiduo Zhang, Panpan Cai, Jinlan Fu, and Xipeng Qiu. World modeling makes a better planner: Dual preference optimization for embodied task planning. arXiv preprint arXiv:2503.10480, 2025. 3

Xianlong Wang, Hewen Pan, Hangtao Zhang, Minghui Li, Shengshan Hu, Ziqi Zhou, Lulu Xue, Peijin Guo, Aishan Liu, Leo Yu Zhang, et al. Trojanrobot: Physical-world backdoor attacks against vlm-based robotic manipulation. arXiv preprint arXiv:2411.11683, 2024c. 3

Jiajun Xi, Yinong He, Jianing Yang, Yinpei Dai, and Joyce Chai. Teaching embodied reinforcement learning agents: Informativeness and diversity of language use. arXiv preprint arXiv:2410.24218,

2024. 3

Zhen Xiang, Fengqing Jiang, Zidi Xiong, Bhaskar Ramasubramanian, Radha Poovendran, and Bo Li. Badchain: Backdoor chain-of-thought prompting for large language models. arXiv preprint arXiv:2401.12242, 2024. 3

Jiashu Xu, Mingyu Derek Ma, Fei Wang, Chaowei Xiao, and Muhao Chen. Instructions as backdoors: Backdoor vulnerabilities of instruction tuning for large language models. arXiv preprint arXiv:2305.14710, 2023. 3

Jun Yan, Vikas Yadav, Shiyang Li, Lichang Chen, Zheng Tang, Hai Wang, Vijay Srinivasan, Xiang Ren, and Hongxia Jin. Backdooring instruction-tuned large language models with virtual prompt injection. arXiv preprint arXiv:2307.16888, 2023. 3

Rui Yang, Hanyang Chen, Junyu Zhang, Mark Zhao, Cheng Qian, Kangrui Wang, Qineng Wang, Teja Venkat Koripella, Marziyeh Movahedi, Manling Li, Heng Ji, Huan Zhang, and Tong Zhang. Embodiedbench: Comprehensive benchmarking multi-modal large language models for visiondriven embodied agents. CoRR, abs/2502.09560, 2025. doi: 10.48550/ARXIV.2502.09560. URL https://doi.org/10.48550/arXiv.2502.09560. 1, 2, 3, 4, 7, 15

Wenkai Yang, Xiaohan Bi, Yankai Lin, Sishuo Chen, Jie Zhou, and Xu Sun. Watch out for your agents! investigating backdoor threats to llm-based agents. In Amir Globersons, Lester Mackey, Danielle Belgrave, Angela Fan, Ulrich Paquet, Jakub M. Tomczak, and Cheng Zhang (eds.), Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 15, 2024, 2024a. URL http://papers.nips.cc/paper files/paper/2024/hash/ b6e9d6f4f3428cd5f3f9e9bbae2cab10-Abstract-Conference.html. 3

Yijun Yang, Tianyi Zhou, Kanxue Li, Dapeng Tao, Lusong Li, Li Shen, Xiaodong He, Jing Jiang, and Yuhui Shi. Embodied multi-modal agent trained by an llm from a parallel textworld. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 26275– 26285, 2024b. 3

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In International Conference on Learning Representations (ICLR), 2023. 3

Zenghui Yuan, Jiawen Shi, Pan Zhou, Neil Zhenqiang Gong, and Lichao Sun. Badtoken: Tokenlevel backdoor attacks to multi-modal large language models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 29927–29936, 2025. 3

Wenqi Zhang, Mengna Wang, Gangao Liu, Xu Huixin, Yiwei Jiang, Yongliang Shen, Guiyang Hou, Zhe Zheng, Hang Zhang, Xin Li, et al. Embodied-reasoner: Synergizing visual search, reasoning, and action for embodied interactive tasks. arXiv preprint arXiv:2503.21696, 2025. 3

Shuai Zhao, Jinming Wen, Luu Anh Tuan, Junbo Zhao, and Jie Fu. Prompt as triggers for backdoor attack: Examining the vulnerability in language models. arXiv preprint arXiv:2305.01219, 2023. 3

- A DATASET DETAILS

We employ two vision-driven embodied agent benchmarks: VAB-OmniGibson (Liu et al., 2024c) and EB-ALFRED (Yang et al., 2025). Table 4 compares these two environments. We adhere to the original experimental setups outlined in their respective papers, thereby encompassing a variety of scenarios. These scenarios include image inputs both with and without object bounding boxes, action spaces ranging from relatively low-level to high-level, and input prompts involving both multi-turn and single-turn format. For EB-ALFRED specifically, we modify the interaction pattern: instead of generating and executing an entire sequence of actions at once, we generate and execute actions individually, one at a time. However, we retain the generation of an overall action plan. This modification aligns better with our training pipeline and raises the complexity of the backdoor attack task from optimizing single outputs to optimizing agent policies.

Table 4: Comparison of VAB-OmniGibson and EB-ALFRED Environments.

VAB-OmniGibson (Liu et al., 2024c) EB-ALFRED (Yang et al., 2025) Simulator OmniGibson AI2-THOR (Kolve et al., 2017)

Visual Input Example

[Figure 74]

[Figure 75]

Action Space

20 low-level actions e.g., grasp, put inside

8 high-level skill types e.g., pick up, open

Action Example move(9.refrigerator) find an apple

Prompt

Multi-turn conversation format (Appendix H.1)

Single-turn with summarized interaction history (Appendix H.2)

Table 5 presents detailed statistics for the datasets used to train and evaluate BEAT. The training sets for VAB-OmniGibson and EB-ALFRED include diverse yet limited trajectories across various scenarios. Although the total number of training steps is similar between the two datasets, VABOmniGibson consists of fewer trajectories due to its lower-level actions, which require more steps per task. The 400 test cases for benign and backdoor scenarios in both datasets are held out from training, featuring either unseen tasks within familiar scenes or entirely unseen scenes.

Table 5: Dataset Statistics in VAB-OmniGibson and EB-ALFRED in BEAT.

VAB-OmniGibson EB-ALFRED

Train Test Train Test Benign Backdoor Benign Backdoor Benign Backdoor Benign Backdoor

# Scenes 11 6 18 7 41 12 49 10 # Trigger placements - 16 - 18 - 33 - 26 # Tasks 20 24 39 24 105 125 98 56 # Scenarios 35 112 100 100 105 231 100 100 # Trajectories 71 156 - - 211 733 - # Steps 1606 1533 - - 2009 1646 - -

- B IMPLEMENTATION DETAILS

Open-sourced Models. We applied LoRA fine-tuning for the open-sourced models, dividing our training data at the trajectory level into training and validation sets via random splitting with a validation ratio of 0.1.

For Supervised Fine-Tuning (SFT), we introduced a parameter k to control the proportion of backdoor data (Dattack). We used a LoRA configuration with rank = 16, alpha = 32, and dropout = 0.05, applying it exclusively to the language module while keeping the vision module fixed. Training was performed over 3 epochs with a batch size of 4 and gradient accumulation over 4 steps. The AdamW optimizer with 8-bit quantization, a learning rate of 2e-4, and a cosine scheduler was employed.

For Contrastive Trigger Learning (CTL), we adopted a weighted sampling strategy, assigning a reduced sampling probability (pSFT) to DSFT samples to optimize the contributions from diverse dataset types effectively. Training spanned 2 epochs with a learning rate of 3e-5, gradient accumulation over 4 steps, and a batch size of 1 due to computational demands. We again used the AdamW optimizer, setting a maximum gradient norm of 0.3, a warmup ratio of 0.03, and β = 0.05 for the DPO component.

We performed hyperparameter tuning for the loss weight α of the NLL term in CTL and the backdoor data ratio k. Specific hyperparameter settings used to generate the results shown in Table 1 are detailed in Table 6.

Table 6: Hyper-parameters. Model VAB-OmniGibson EB-ALFRED Qwen2-VL 7B

k = 0.5 α = 0.4

k = 0.3 α = 0.6

k = 0.5 α = 0.4

k = 1.0 α = 0.6

InternVL3 8B

GPT-4o. For GPT-4o, we utilized OpenAI’s fine-tuning API with automatic hyperparameter selection. We observed that GPT-4o was robustly aligned for safety; even at the highest backdoor data ratio (k = 1), the model failed to learn trigger steps, resulting in zero attack success rates (ASRs) across both datasets. Consequently, we conducted additional continual learning using a subset of the fine-tuning data, emphasizing a high proportion of trigger steps and a lower proportion of regular SFT steps and non-trigger steps from the backdoor dataset. Due to limitations in DPO training capabilities with images, we restricted our GPT-4o training to SFT only, which resulted in comparatively lower performance relative to other models.

- C EXPERIMENTS ON SYSTEM PROMPT ROBUSTNESS

To assess the attack’s robustness to system prompt changes, which is an important consideration since users may modify system prompts for different tasks, we asked GPT to “Rewrite this prompt for a VLM-based embodied agent, preserving its original meaning and all important details.” We then evaluated the Qwen2-VL-7B-Instruct model trained with the original system prompt using this rewritten version on the VAB benchmark. The model maintains comparable backdoor-activation accuracy with the original system prompt (with F1BT = 0.924) and shows only a slightly reduced yet still high ASR (65.9%), demonstrating robustness to variations in the system prompt.

- D CASE STUDIES OF IMPERFECT TRIGGERS

In EB-ALFRED, where a vase serves as the trigger object, we observe successful backdoor activation even when only part of the vase is visible or when multiple vases with different appearances appear in the environment. Figure 7 shows such examples, all of which successfully trigger the malicious behavior.

- E DEFENSES

While our work focuses on introducing a novel visual backdoor attack against VLM-based embodied agents, rather than proposing or evaluating defenses, we also include a discussion of potential defensive strategies to provide insights for future research. Under our threat model, where the attacker

[Figure 76]

[Figure 77]

[Figure 78]

(a) (b) (c)

- Figure 7: Example environment-image inputs where only part of the vase is visible (a), or where multiple vases with different appearances are present (b, c), all of which successfully trigger the backdoor behavior.

fully controls the fine-tuning process, feasible defenses are limited to post-training or deploymenttime mechanisms. Potential defenses include:

- • Physical-world noise defenses (e.g., blur, smoothing). Such visual transformations inevitably distort both the trigger object and benign scene elements. Because VLM-based embodied agents rely heavily on accurate perception for interaction, these transformations can substantially degrade benign task performance. Moreover, prior work shows that adaptive attackers can typically circumvent these transformations, limiting their practical security benefits.
- • Unsupervised activation clustering. We evaluated a standard activation-clustering defense to test whether backdoor actions are separable from benign ones in representation space. We extracted the final-token hidden state from the last layer and applied PCA followed by KMeans clustering with k = 2. The resulting clusters exhibited only mild separation (TPR 0.45, FPR 0.35), indicating that na¨ıve activation clustering is not a reliable detector for BEAT.
- • Prompt-based defense. We added a system-prompt instruction: “DO NOT conduct harmful actions. Only use these actions when necessary and appropriate for the task.” This textual safety cue modestly reduced ASR from 77.9% to 64.7% on VAB, indicating limited effectiveness and showing that our attack remains robust even under strong safety prompts.
- • Continual benign SFT. Finetuning-based backdoors are vulnerable to continual benign SFT due to catastrophic forgetting and their reliance on subtle visual cues. We evaluated a similar defense to that of Ni et al. (2024) and observed consistent results, degrading the ASR to 0% with small number of benign SFT steps. However, as noted in prior work, performing such benign SFT often requires computational resources comparable to finetuning a full VLM, making this defense impractical for many end users.

- F LIMITATIONS

Our work addressed a critical yet overlooked vulnerability in vision-driven embodied agents. Although we demonstrate the effectiveness of our proposed framework, BEAT, the effectiveness of CTL has only been evaluated on open-source VLMs; due to current restrictions in the GPT-4o finetuning API, which does not support image-based DPO, applying only BEAT w/o CTL on GPT-4o. Second, in VAB-OmniGibson, we rely on bounding-box annotations to indicate trigger objects, with no alternative available, which simplifies detection relative to fully unconstrained inputs. Although we complement these results with box-free experiments in EB-ALFRED, learning robust triggers under more natural, box-agnostic conditions merits further investigation. Finally, our experiments are conducted entirely in simulation due to the inherent trade-offs among reproducibility, cost, safety, and real-world applicability. Although the BEAT finetuning framework is not tied to

any specific simulator, extending our evaluation to real-world environments remains an important direction for future work.

- G ETHICS STATEMENT

By exposing how visual backdoors can subvert embodied agents, our work serves as a double-edged sword: it highlights serious vulnerabilities that could be weaponized, but we believe transparency is essential to drive the development of effective defenses. We hope these findings spur the community to adopt proactive safeguard methods, design more robust vision-driven embodied agent protocols, and ultimately deploy embodied systems with stronger guarantees against covert manipulation.

- H PROMPTS

- H.1 PROMPT FOR VAB-OMNIGIBSON

- System

# Setup You are an intelligent agent exceling at solving household tasks. You are in a household

environment given a task to finish. You can interact with the environment by performing actions using python-style pseudo code. For each turn, please call exactly one predefined action.

# Valid Actions ## Predefined Action List: ``` def grasp(obj):

'''Grasp the object in your hand. Args:

:param obj: the digital identifier of the object to grasp. Returns:

A string message of the environment feedback. '''

def move(obj): '''Move yourself towards the object. Args:

:param obj: the digital identifier of the object to move towards. Returns:

A string message of the environment feedback. '''

def move to room(room): '''Move yourself to a random position in the room. Args:

:param room: the name of the room to move to. Returns:

A string message of the environment feedback. '''

def turn left(): '''Turn the robot left 90 degrees. Returns:

A string message of the environment feedback. '''

def turn right(): '''Turn the robot right 90 degrees. Returns:

A string message of the environment feedback. '''

def raise camera(): '''Raise the camera to see objects that are higher. Returns:

A string message of the environment feedback. '''

def lower camera(): '''Lower the camera to see objects that are lower. Returns:

A string message of the environment feedback. '''

def put inside(obj1, obj2): '''Put obj1 within your hand inside obj2. If obj2 is openable, make sure it is open

before putting obj1 inside. Args:

- :param obj1: the digital identifier of the object to put inside.
- :param obj2: the digital identifier of the object to put inside of.

Returns:

A string message of the environment feedback. '''

def put on top(obj1, obj2): '''Put obj1 within your hand to the top of obj2. Args:

- :param obj1: the digital identifier of the object to put on top.
- :param obj2: the digital identifier of the object to put on top of.

Returns:

A string message of the environment feedback. '''

def put under(obj1, obj2): '''Put obj1 within your hand to the bottom of obj2. Args:

- :param obj1: the digital identifier of the object in your hand.
- :param obj2: the digital identifier of the object to put obj1 under.

Returns:

A string message of the environment feedback. '''

def put next to(obj1, obj2): '''Put obj1 within your hand next to obj2. Args:

- :param obj1: the digital identifier of the object in your hand.
- :param obj2: the digital identifier of the object to put obj1 next to.

Returns:

A string message of the environment feedback. '''

def get fridge view(): '''Get the image captured by a camera in the fridge. This function is only valid when

you are near a fridge and the fridge is open. Returns:

A string message of the environment feedback. '''

def cook(obj): '''Cook the object. Args:

:param obj: the digital identifier of the object to cook. Returns:

A string message of the environment feedback. '''

def burn(obj): '''Burn the object. Args:

:param obj: the digital identifier of the object to burn. Returns:

A string message of the environment feedback. '''

def freeze(obj): '''Freeze the object. Args:

:param obj: the digital identifier of the object to freeze.

Returns:

A string message of the environment feedback. '''

def heat(obj): '''Heat the object. Args:

:param obj: the digital identifier of the object to heat. Returns:

A string message of the environment feedback. '''

def open(obj): '''Open the object. Args:

:param obj: the digital identifier of the object to open. Returns:

A string message of the environment feedback. '''

def close(obj): '''Close the object. Args:

:param obj: the digital identifier of the object to close. Returns:

A string message of the environment feedback. '''

def toggle on(obj): '''Toggle on the object. Args:

:param obj: the digital identifier of the object to toggle on. Returns:

A string message of the environment feedback. '''

def toggle off(obj): '''Toggle off the object. Args:

:param obj: the digital identifier of the object to toggle off. Returns:

A string message of the environment feedback. '''

def done(): '''Call this function if you think the task is completed. Note that you have no chance

to take any actions after calling this function. Returns:

None. The environment will check whether the task is completed and check your score. '''

``` ## Reminder

- 1. You can only hold one object at a time.
- 2. When moving to a new position, you can always turn left, turn right, raise camera or lower camera to see around before making a decision.
- 3. You can only interact with objects within your reach; if not, first try moving towards it or something close to it.
- 4. You can only interact with objects that are visible to you (annotated with a bounding box in the image); if it's not visible, try to move inside the room or other rooms and look around to find it. You can open refrigerators or other enclosures to see inside them

.

- 5. You can interact with objects that are very close to you, such as those you've just moved towards, even if you don't see them currently.
- 6. When you are out of the room and see nothing useful, try moving to a room.
- 7. You can always move to something in the same room with you, if you have seen it before, even though you cannot see it now. So when you are in a new room, try to move around and see around to record more objects in your observation so that you can move to them flexibly afterwards.
- 8. Don't repeat the failed action in the next round. Try to understand what went wrong and make a different decision.

- 9. If you can't complete the task, you can do as much as you can and call `done()` to finish the task.

# Input For each dialog, you will be given the following information at the beginning.

- 1. Task Goal: The task is finished only when these conditions are met.
- 2. Reachable Rooms: Rooms you can move to. Please refer to them with their names provided here.

For each turn, you will be given the following information.

- 1. Action Feedback: Environment feedback of the last action.
- 2. At Hand Object: The object you are currently holding.
- 3. Current Room: The room you are currently in.
- 4. Vision Input: the image you see from your perspective (or inside the fridge). All taskrelated objects appear in your view will be annotated with bounding boxes and unique identifiers. Please reference these objects using the digital identifier provided here. Note that if the object is not annotated with a bounding box, the object can't be interacted with.

# Output Now, given these information, you need to think and call the action needed to proceed with

the task. Your response should include 3 parts in the following format in each turn: OBSERVATION: <What you observe in the image> Note that the Vision Input image won't be kept in the dialog, so make sure you capture all the key information (eg, the identifier of

the object you see) here for future use. THOUGHT: <Your step-by-step thoughts> ACTION: <The action code> Note that only one function is allowed in each dialog turn! Only

one line of code is allowed in each dialog turn! If your output contains multiple actions or multiple turns of actions, only the first one will be executed!

- - User Your task goal is: {INSTRUCTION} At Hand Object: {AT_HAND_OBJECT_0} Current Room: {ROOM_0} Vision Input: Omitted

- - Assistant OBSERVATION: {OBSERVATION_0} THOUGHT:{THOUGHT_0} ACTION: {ACTION_0}

... <HISTORY>

- - User Action Feedback: {ACTION_FEEDBACK_CURRENT} At Hand Object: {AT_HAND_OBJECT_CURRENT} Current Room: {ROOM_CURRENT} Vision Input: <image>

##### H.2 PROMPT FOR EB-ALFRED

- System ## You are a robot operating in a home. Given a task, you must accomplish the task using a

defined set of actions to achieve the desired outcome.

## Action Descriptions and Validity Rules Find: Parameterized by the name of the receptacle to navigate to. So long as the object is

present in the scene, this skill is always valid.

Pick up: Parameterized by the name of the object to pick. Only valid if the robot is close to the object, not holding another object, and the object is not inside a closed receptacle.

Put down: Parameterized by the name of the object to put down to a nearby receptacle. Only valid if the robot is holding an object.

Drop: Parameterized by the name of the object to put down. It is different from the Put down action, as this does not guarantee the held object will be put into a specified receptacle.

Open: Parameterized by the name of the receptacle to open. Only valid if the receptacle is closed and the robot is close to the receptacle. Close: Parameterized by the name of the receptacle to close. Only valid if the receptacle is open and the robot is close to the receptacle.

Turn on: Parameterized by the name of the object to turn on. Only valid if the object is turned off and the robot is close to the object. Turn off: Parameterized by the name of the object to turn off. Only valid if the object is turned on and the robot is close to the object. Slice: Parameterized by the name of the object to slice. Only valid if the object is sliceable and the robot is close to the object.

## You are supposed to output in json. At each timestep, you may decide to: 1) follow your

previous plan, especially when your previous plan is successful and unfinished, or 2) do reasoning and make a new plan.

For reasoning, you need to output a reasoning message, in which you should describe the current visual state from the image, output your reasoning steps, and plan. At the end, you should output an action message, which should include the action id from the available actions to execute and its corresponding description.

- User <image> instruction: {INSTRUCTION}. interaction history: {INTERACTION_HISTORY} available actions: {AVAILABLE_ACTION_LIST}

