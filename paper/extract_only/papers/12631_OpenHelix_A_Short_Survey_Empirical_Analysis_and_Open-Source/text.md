## OPENHELIX: A Short Survey, Empirical Analysis, and Open-Source Dual-System VLA Model for Robotic Manipulation

# arXiv:2505.03912v1[cs.RO]6May2025

Can Cui1, Pengxiang Ding12*, Wenxuan Song4, Shuanghao Bai3, Xinyang Tong1, Zirui Ge2, Runze Suo1, Wanqi Zhou3, Yang Liu1, Bofang Jia1, Hangyu Liu12, Mingyang Sun12, Han Zhao12, Siteng Huang1, Donglin Wang1 1Westlake University 2Zhejiang University 3Xi’an Jiaotong University 4HKUST(GZ)

##### Abstract

Dual-system VLA (Vision-Language-Action) architectures have become a hot topic in embodied intelligence research, but there is a lack of sufficient open-source work for further performance analysis and optimization. To address this problem, this paper will summarize and compare the structural designs of existing dual-system architectures, and conduct systematic empirical evaluations on the core design elements of existing dual-system architectures. Ultimately, it will provide a low-cost open-source model for further exploration. Of course, this project will continue to update with more experimental conclusions and open-source models with improved performance for everyone to choose from. Project page: https://openhelix-robot.github.io/.

##### 1. Short Survey

###### 1.1. Definition of VLAs

Traditional policy learning has primarily focused on training novel behaviors from scratch using lightweight models. These models demonstrate high sensitivity to environmental perturbations (both visual and textual) and exhibit limited generalization capabilities. However, with the emergence of large language models (LLMs) and vision-language models (MLLMs), this landscape is undergoing significant transformation. These models, trained on Internet-scale data with vast parameter spaces, have demonstrated exceptional proficiency in text generation and visual comprehension, consequently generating substantial interest in their application to robotics policy training. In this context, RT-2 introduced the pioneering concept of vision-language-action models (VLAs), which co-fine-tune state-of-the-art vision-language models on both robotic trajectory data and Internet-scale vision-language tasks. VLAs have demonstrated remarkable improvements in generalizing to novel objects and se-

* Project Lead. Email: dingpx2015@gmail.com

mantically diverse instructions while exhibiting a range of emergent capabilities. Furthermore, VLAs possess the potential to revolutionize robotic skill acquisition methodologies, as they serve as powerful foundation models that can be directly fine-tuned to adapt to domain-specific robotic applications. Subsequently, research on VLAs has proliferated rapidly, offering a highly promising approach to addressing the challenges associated with robotic deployment.

###### 1.2. Limitation of VLAs

Directly applying VLAs in domain-specific or real-world scenarios remains challenging due to the following limitations: (1) The large and cumbersome model size of VLAs makes achieving efficient real-time performance difficult. For instance, RT-2 demonstrated that their 55B model operates at 1 to 3 Hz, while the 5B model runs at around 5 Hz under experimental conditions. In contrast, traditional lightweight models, such as BC-Transformer, operate at much higher speeds (around 50 Hz). (2) Pre-training is resource-intensive, and end-to-end fine-tuning of pre-trained VLAs on embodied data is also challenging due to domain shift and catastrophic forgetting. Leveraging existing MLLMs and VLAs for practical applications, while retaining their remarkable capabilities in multimodal understanding, reasoning, and generation, and ensuring fast inference for coherent actions, remains a challenge that needs to be addressed.

###### 1.3. Definition of Dual-System VLAs

As a result, the Dual-System VLAs were introduced. LCB [19] pioneered the adoption of the Dual-System VLA structure, while DP-VLA is the first to incorporate dualprocess theory to provide a personified explanation for the rationale underlying this architecture. Dual-process theory [7, 10, 16, 21] conceptualizes human cognition as operating through two distinct systems:

1. System 1 is fast, automatic, intuitive, and unconscious. It operates effortlessly and relies on heuristics to make judgments and decisions. System 1 is responsible for our immediate reactions, such as making simple or routine choices.

- Table 1. Methods Comparisons of Dual-System VLAs. Here, L, R, P, D, T, and PC represent different modalities: Language, RGB, Proprioception, Depth, Tactile, and Point Cloud, respectively. FT denotes fine-tuning. Pretrain and Scratch denote fine-tuning a pre-trained policy head and training a policy head from scratch, respectively.

System 2

System1

Method

Latent Rep.

Model Input Training Policy Head Sensory Training LCB [19] LLaVA-7B L+R Lora FT Lang(<ACT>) 3D Diffusion Actor [11] R+P+PC Pretrain

DP-VLA [9] OpenVLA-7B L+R Frozen Vis+Lang Transformer R+P Scratch

HiRT [24] InstructBLIP-7B L+R Lora FT MaxPooling(Vis+Lang) RT-1 [4] R Scratch Robodual [5] OpenVLA-7B L+R Lora FT Action+Lang DiT R+D+T+P Scratch DexVLA [22] Qwen2-VL-2B L+R Lora FT Lang ScaledDP [25] R+P Scratch

Helix N/A L+ R + P N/A N/A Transformer R+P N/A

System 1 often leads to biases and errors because it relies on mental shortcuts, such as heuristics, which can be effective in some situations but also result in systematic mistakes. In the robotics domain, this system closely resembles traditional lightweight policy networks, which are efficient but often task-specific.

- 2. System 2 is slow, deliberate, effortful, and conscious. It involves reasoning, logic, and careful evaluation of evidence. System 2 is engaged when performing cognitively demanding tasks, such as solving complex problems or making thoughtful decisions. System 2, while generally more accurate, requires greater cognitive resources and is also prone to errors when cognitive load is high or attention is limited. In robotics, this system is analogous to large-scale models like MLLMs and VLAs, which are computationally heavy but offer superior generalization capabilities.
- 3. Although the two systems operate in parallel, they update information at different frequencies. The slower System 2-like component updates less frequently and is responsible for making more deliberate decisions based on high-level representations. In contrast, the faster System 1-like component updates at a higher frequency to rapidly generate the low-level actions required for real-time robotic control. Notably, the information from the slow system is subject to temporal delay. This architecture addresses the aforementioned challenge by simultaneously enabling efficient real-time inference while preserving the multimodal reasoning capabilities of large models.

###### 1.4. Current Dual-System VLAs

We introduce recent Dual-System VLA approaches below, with a comparative analysis of their distinctive features summarized in Table 1. It is important to note that for synchronous inference to occur, System1 must incorporate real-time perception inputs (such as RGB images). According to this criterion, approaches like π0 [3], GR00TN1 [2], and similar methodologies cannot be properly classified within the dual-system framework as they lack this essential characteristic.

LCB adopts LLaVA as its System 2. Given a high-level task

description and an RGB observation, LLaVA generates a textual action description along with an <ACT> token. The <ACT> token, derived from the final layer, serves as a highlevel latent goal. System 1 is a pre-trained 3D Diffusion Actor that takes the RGB image, point cloud, and <ACT> token as input to generate actions. System 2 is fine-tuned using LoRA, while System 1 is fine-tuned in a standard manner.

DP-VLA introduces dual-process theory to justify the rationale behind the dual-system architecture. It presents a more generalizable design choice, where System 2 is not limited to MLLMs, but can also be VLAs that are pre-trained on robot data. In experiments, DP-VLA adopts OpenVLA as System 2 and uses its encoder to extract latent representations from language instructions and RGB observations to guide System 1. System 1 is implemented using a Transformer architecture, which encodes RGB images and proprioceptive inputs into actions. System 2 is kept frozen, while System 1 is trained from scratch.

HiRT adopts InstructBLIP as System 2 and utilizes the finallayer representations obtained from encoding both language instructions and RGB observations. These representations are processed with MAP pooling to produce MLLM latent features that guide System 1. System 1 uses an EfficientNetB3 backbone combined with a MAP block to encode RGB inputs into actions. System 2 is fine-tuned using LoRA, while System 1 is trained from scratch.

Robodual adopts OpenVLA as System 2 and extracts latent representations from language instructions and RGB observations. It uses both the task latent derived from the instruction and the final action latent as guidance signals. System 1 encodes RGB, depth, tactile, and proprioceptive inputs using a ViT-based encoder, and employs a Perceiver Resampler to distill key features. A DiT model then generates actions by conditioning on the distilled features, the task latent, and a noisy action input. System 2 is fine-tuned using LoRA, while System 1 is trained from scratch.

Integration Strategy

#### MMLM

#### Policy

High-freq Perception

Low-freq Perception

√ √/×

[Figure 1]

[Figure 2]

Img

Latent Vector

√

Img

[Figure 3]

[Figure 4]

Lang(<ACT>) Vis+Lang ... ... Action+Lang

Dep

3DDA RT-1 ... ... ScaleDP

LLaVA-7B OpenVLA-7B ... ... Qwen2-VL-2B

√

[Figure 5]

√ Act

Lang

[Figure 6]

[Figure 7]

...

√/× √/×

...

Tac

√/×

[Figure 8]

Pro

Pro

[Figure 9]

[Figure 10]

Asynchronous Strategy

Figure 1. Key Design of Dual-System VLAs. It mainly includes: MMLM Selection, Policy Selection, Latent Feature Representation Selection, MLLM Training Strategy, Policy Training Strategy, Dual-System Integration Strategy, and Dual-System Asynchronous Strategy.

###### 1.5. Key Design of Dual-System VLAs

structure can both meet current needs. However, with the introduction of new policy models such as CARP [8], Dense Policy [20], and other new architectures, downstream small models may also see new designs. Additionally, like Robodual [5], whether downstream small models need more modal information, and which modal information is essential for system1, is also a potential question.

The key question lies in how to design the architecture of these two systems and structure the information flow from the slower system to the faster one in a way that preserves the strengths of the System 2-like component while effectively guiding the System 1-like component to execute robotic actions. Achieving this delicate balance is essential for building robotic systems that are both highly performant and generalizable. As shown in Figure 1, to achieve this objective, several core issues need to be addressed:

3. Latent Feature Representation Selection. For the selection of latent tokens, this is the most complex aspect of dualsystem tasks that urgently needs research. Previous methods have shown significant differences in their approaches. We need to consider not only dual-system work but also single-system work such as [2, 3, 13] For DP-VLA [9], they directly chose the last layer hidden embedding of the MLLM large model. Meanwhile, GR00T-N1 [2] selected hidden embeddings from middle layers, considering that middlelayer features might contain more visual information and could reduce inference time. Taking this further, Roboflamnigo [13] and HiRT [24] used maxpooling of the last layer language features and visual features as downstream conditions. Beyond directly utilizing MLLM hidden embeddings, some models (e.g., LCB [19]) additionally introduced the concept of <ACT> tokens, hoping to bridge upstream and downstream through fine-tuning a special token, which showed promising results. The above two approaches were further developed in Robodual [5], which adopted multiple <ACT> tokens while also incorporating last-layer language features as latent feature representations. Of course, beyond the robotics domain, there are more ingenious works utilizing hidden states, such as Metaquery [17] and LEGO [12], which employed more sophisticated methods for latent feature selection. In summary, the selection of latent feature

- 1. MLLM Selection. For different VLA scenarios, requirements of MLLMs vary. For building a model suitable for robotic scenarios, the MLLM model should be selected appropriately. For example, Flower [18]’s foundation model has strong capabilities in spatial awareness/low-level vision, therefore achieving current SOTA across various tasks; MiniVLA [1] chose Qwen-VL 0.25B as its foundation model to reduce model inference costs and burden. Therefore, in an era of rapidly evolving MLLMs, we should clarify what kind of MLLM model is lightweight enough yet sufficient to complete robotic tasks, which is a problem that needs to be addressed.

Furthermore, whether a MLLM pre-trained on robotic data is necessary remains an unresolved question. Training on extensive robotic datasets not only reduces the domain gap, but also, by exposing the model to more language instructions, makes its performance exceptionally robust on language instruction following tasks, as demonstrated in experiments with Robodual [5].

- 2. Policy Selection. The choice of small models is relatively less controversial, with the current general consensus being that models based on DiT structure and Flow Matching

###### CALVIN-E CALVIN CALVIN-D

representations will be an important research focus for dualsystem models, exploring more suitable latent features for downstream action generation models.

Select the blue square and turn it around.

Pick up the blue square and rotate it.

Pick up the blue square and rotate it.

[Figure 11]

Enriched Instruction

Original Instruction Standard Instruction

- 4. MLLM Training Strategy. Regarding how to train MLLMs, the main consideration is examining whether we can maintain the model’s generalization capabilities without loss while also ensuring good integration with downstream tasks. Currently, the main approaches include frozen and fine-tuning methods, but exploring whether there are better fine-tuning techniques remains a valuable direction for research.
- 5. Policy Training Strategy. Regarding how to train the Policy, the main consideration is whether to reduce the model’s training cost. If we can take a pre-trained policy and finetune it, this could greatly reduce the overall training time. Of course, if we train from scratch, whether the different optimization objectives would make model convergence difficult is also an unknown factor that needs to be explored.
- 6. Dual-System Integration Strategy. Regarding Integration strategies, the main focus is how to embed latent information as a condition into downstream models. In LCB [19], the authors demonstrated using CLIP loss to constrain upstream latent features to be similar to the original text CLIP embedding to connect upstream and downstream components. However, this approach clearly limits the model to only handle cases it was trained on downstream, negating the purpose of introducing the generalization capabilities of MLLM models. Additionally, when introducing a new embedding, differences in dimensions between upstream and downstream models are inevitable, making it common to add a projector between them. However, how to train this projector requires careful consideration. In subsequent experiments, when the downstream policy is a pre-trained one, it becomes critically important to pre-align the projector without training the MLLM. If both are unfrozen and trained simultaneously, the model training will collapse. Therefore, the Dual-System Integration Strategy is a crucial aspect.
- 7. Dual-System Asynchronous Strategy. Lastly, there are asynchronous strategies for dual-system models. LCB [19], HiRT [24], and Robodual [5] employ different asynchronous approaches, with LCB [19] being the most naive, using synchronous training but asynchronous testing. Theoretically, differences in inference frequency between upstream and downstream components could affect final performance. However, this is not entirely accurate - if the upstream features being provided aren’t effective to begin with, perhaps asynchronous inference between upper and lower layers is merely a pseudo-requirement. Therefore, more experiments are needed to verify this.

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

Static Environment Dynamic Environment

Static Environment

Figure 2. Three Different Evaluation Environments.

dimensions, including the choice of base vision-language model (MLLM), downstream policy architecture, and latent selection mechanisms [5, 9, 19, 22, 24]. These discrepancies highlight the urgent need for a systematic and fair comparison, in order to assess the rationale behind different design choices and to establish a reference framework for future model development.

In this work, we standardize experimental conditions 1, 2, 3, and 7 to ensure consistency, and focus our evaluation primarily on conditions 4, 5, and 6. These conditions involve widely applicable techniques that are largely independent of the specific choices made in conditions 1, 2, 3, and 7. Through this controlled comparison, we aim to offer insights that may inspire and guide future research in this domain. We plan to extend our evaluation to cover additional conditions in future work. All updates and ongoing developments will be made available in our official GitHub repository: https://github.com/OpenHelix-robot/OpenHelix/.

###### 2.1. Experiment setup

Model Selection. To ensure consistency with LCB [19], we adopt LLaVA1.0 [14] as the visual language model (MLLM) throughout this paper. To eliminate discrepancies caused by differing policy architectures, all subsequent experiments utilize 3DDA [11] as the unified downstream policy. The integration of latent representations is implemented in accordance with the methodology introduced in LCB [19]. In line with LCB [19], we employ synchronous training and asynchronous testing for experiments involving asynchronous settings.

Dataset Processing. Unlike LCB [19], which constructs a chat-like response before the <ACT> token, we directly concatenate an <ACT> token after the instruction. This approach was adopted because we have not yet implemented this functionality. Additionally, we observed that even without implementing this feature, performance remains satisfactory. We plan to address the processing of chat-like data in future work.

##### 2. Empirical Evaluations

According to the above survey, it is evident that current dualsystem models exhibit substantial variation across multiple

Environment. For comparison with models that are not

###### Table 2. Evaluation of single system in CALVIN-D environment.

Success rate (%) ↑ Static Left Forward Diagonal Circle RF 100 0 0 0 0

Model

3DDA 82 84 46 67 80

###### Table 3. Comparison of different training strategies for the low-level policy in standard CALVIN environment.

Task1 completed2 3 in4a row (%)5 ↑ Avg. Len ↑ Fine-tuning 96 83 68 58 48 3.53

MLLM

From-scratch 89 71 49 42 34 2.85

open-source but have published results, we selected the same environment as they used. To maintain consistency with LCB [19] and RoboDual [5], we selected the CALVIN environment as our core comparative simulation environment. Real-world environment experiments will be supplemented later.

Standard Evaluation. Following prior works, we primarily validate our effectiveness in the ABC-D scenario. To ensure rapid assessment of experiments, we used the first 100 evaluations from the standard 1000 evaluations, improving the testing efficiency for ablation experiments. In the final evaluation Table 8, we extend the evaluations to a full set of 1,000 to provide more comprehensive experimental results.

Harder Evaluation. As is shown in Figure 2, in the standard evaluation test scenario, objects are static, and the given language instructions are standard. However, the dual system should inherently combine the language generalization capabilities of large models with the advantages of small models’ high-frequency characteristics for dynamic scenarios. Therefore, we conducted additional validation in two scenarios.

- 1. CALVIN-E: For language instruction generalization,

we used Enriched language instructions for testing.

- 2. CALVIN-D: For dynamic scenario testing, in grasping

tasks, we made objects move in four different ways within the environment to examine the model’s robustness in dynamic scenarios.

###### 2.2. Why not single system?

Preliminary. In fact, the definition of dual systems has always been ambiguous, but since we established the CALVIND experiment, we found that previous single-system work (e.g., Roboflmanigo [13]) would directly fail under such testing, therefore subsequent experiments were not conducted on single systems.

Setup. The specific experimental configuration involved test-

ing models trained on the standard ABC dataset on CALVIND for 100 trials. The "Static" condition represents scenarios where standard objects do not move, while "Left," "Forward," "Diagonal," and "Circle" represent four different object movement patterns. The specific results are shown in Table 2.

Analysis. We discovered that the results of the RF [13] model on CALVIN-D were quite surprising, as it completely failed to complete the corresponding tasks in dynamic scenarios. The primary reason for the observed performance is that, during the testing phase, the RF method requires processing the previous six image frames to obtain the corresponding latent representations for LSTM-based action inference. While the latent representations typically remain stable during training, they become variable in the testing phase as a result of object movement within dynamic scenarios. This discrepancy between training and testing conditions leads to a significant drop in performance, resulting in a consistently zero success rate in dynamic environments. Nevertheless, we can also observe that the RF model using MLLM demonstrates extremely high performance on simple tasks, showing much greater robustness than the smaller 3DDA model. This highlights the significance of using MLLM as the "brain" of the system.

Discussion. Of course, we acknowledge that this conclusion may not be completely rigorous, as further testing on π0 [3], GR00TN1 [2] has not yet been conducted. These additional experiments will be included in future work.

###### 2.3. Training strategy of dual system

For dual-system models, the main training strategy consists of three parts: how to train the low-level policy, how to train the high-level MLLM, and how to connect the two. The following experiments will be divided into these three components.

2.3.1. Policy Training Strategy. Preliminary. For LCB, the downstream low-level policy uses pre-trained 3DDA, while HiRT employs the RT-1 structure and trains from scratch. Robodual uses its own designed downstream policy. Setting aside the differences in configurations, there are two paradigms for policy training: training from scratch and fine-tuning from pre-trained models. Setup. For fair comparison, the large model configuration follows the LCB structure: LLaVA1.0 backbone, connected with the <ACT> token, all using CLIP Loss to align the <ACT> token with downstream instructions. The only difference is that the downstream policy uses either a pre-trained 3DDA policy or a policy trained from scratch. The specific results are shown in Table 4.

Analysis. In Table 3, we discover that using a pre-trained policy can improve performance while reducing overall training time. Therefore, subsequent experiments are all based on fine-tuning from pre-trained policy model.

[Figure 22]

Figure 3. Three Different MLLM Training Strategy.

Table 4. Comparison of different training strategies for the high-level MLLM in CALVIN environment.

Task1 completed2 3 in4a row (%)5 ↑ Avg. Len ↑

Benchmark MLLM Integration of MLLM and Policy Policy

Frozen w CLIP Loss Fine-tuning 94 80 64 51 41 3.30 Frozen w/o CLIP Loss Fine-tuning 90 74 61 54 40 3.33

CALVIN

Fine-tuning w CLIP Loss Fine-tuning 96 83 68 58 48 3.53 Fine-tuning w/o CLIP Loss Fine-tuning 88 72 56 46 30 3.13

###### 2.3.2. MLLM Training Strategy.

Preliminary. For LCB, HiRT, and Robodual, the upstream large models all underwent fine-tuning. Although GR00TN1 [2] doesn’t fall within the scope of dual systems, it achieved excellent results by adopting a frozen paradigm for training. Therefore, we conducted experiments on both approaches.

Setup. For fair comparison, the large model configuration follows the LCB structure: LLaVA1.0 backbone, connected with the <ACT> token, all using CLIP Loss to align the <ACT> token with downstream instructions. The downstream policy adopts the fine-tuning paradigm throughout. During the connection process between the MLLM and the policy model, we also introduced whether to include CLIP loss as a variable.

Analysis. For scenarios where the MLLM is frozen, adding or omitting the CLIP loss does not significantly affect performance. This is because the CLIP loss itself is meant to adjust the unchanged MLLM’s output to accommodate the downstream small model’s input, resulting in minimal performance differences. However, when the MLLM requires fine-tuning, the impact of CLIP loss becomes highly significant. Without the constraint of CLIP loss, it’s easy to disrupt the small model’s already-trained attention mechanisms between conditioning and other perceptual inputs, potentially leading to performance degradation.

Intuitive hypothesis. Although the introduction of CLIP loss makes the overall model performance functional, this approach essentially compromises the large model’s inher-

ent generalization capabilities. Is there a way to keep the large model parameters frozen while still ensuring that the large model can be updated together with the downstream components?

Further setup. As shown in Figure 3, we only changed the training method of the MLLM Specifically, we adopted prompt tuning. We added a new <ACT> token to the large model’s vocabulary and only trained the lm-head layer while keeping all other model parameters fixed. This approach essentially trains an additional token in the vocabulary that only relates to downstream tasks, without altering the MLLM model’s inherent generalization capabilities. Therefore, theoretically, it can better ensure the connection between the dual systems.

Next, we use experiments to verify this hypothesis in Table 5.

Further analysis. For the prompt tuning paradigm, while performance in the standard Calvin testing environment is comparable to other training paradigms, there are significant differences in experiments validating language generalization. Similarly, under the premise of having CLIP loss, the generalization capability of prompt tuning results far exceeds that of fine-tuning and frozen approaches. Moreover, without CLIP loss supervision, generalization actually improves somewhat, which fully demonstrates that the prompt tuning paradigm trains the large model with minimal dependence on altering the large model’s generalization capabilities.

2.3.3. Dual-System Integration Strategy. Preliminary.

###### Table 5. Further MLLM training experiments in CALVIN and CALVIN-E environment.

Task1 completed2 3 in4a row (%)5 ↑ Avg. Len ↑ CALVIN

Benchmark MLLM Integration of MLLM and Policy Policy

Prompt-tuning w CLIP Loss Fine-tuning 94 78 62 52 42 3.28 Prompt-tuning w/o CLIP Loss Fine-tuning 94 77 67 60 47 3.45

Prompt-tuning w CLIP Loss Fine-tuning 81 54 41 27 15 2.09 Prompt-tuning w/o CLIP Loss Fine-tuning 72 55 40 26 20 2.13

CALVIN-E

Fine-tuning w CLIP Loss Fine-tuning 76 49 30 15 4 1.74 Frozen w CLIP Loss Fine-tuning 72 37 21 11 5 1.46

###### Table 6. Performance of different projector initialization in CALVIN environment. Here, Pre-alignment refers to training the projector prior to training the MLLM.

Task completed in a row (%) ↑

Benchmark Pre-alignment MLLM Integration of MLLM and Policy Policy

1 2 3 4 5

✓ Frozen w CLIP Loss Fine-tuning 94 80 64 51 41 ✓ Fine-tuning w CLIP Loss Fine-tuning 96 83 68 58 48 ✓ Prompt-tuning w/o CLIP Loss Fine-tuning 94 77 67 60 47

CALVIN

× Frozen w CLIP Loss Fine-tuning 0 0 0 0 0 × Fine-tuning w CLIP Loss Fine-tuning 0 0 0 0 0 × Prompt-tuning w/o CLIP Loss Fine-tuning 0 0 0 0 0

Based on the experiments above, our conclusion is that using a pre-trained policy and fine-tuning the MLLM with prompt tuning yields the best results. However, this still involves the process of how to connect the components, as the semantic gap between upstream and downstream can be substantial. Therefore, we primarily conduct the following ablation analysis.

Setup. To connect the upstream and downstream components, an MLP projector is needed. We implemented two approaches here: First, directly unfreezing both upstream and downstream models and jointly training them together with the MLP projector. Second, initially freezing the upstream large model while training the MLP projector and downstream small model, then unfreezing the upstream large model for joint training. The main difference between these two approaches is whether there is a separate MLP projector training process. The result is in Table 6.

Analysis. We found that without prior projector prealignment, connecting upstream and downstream models based on frozen, fine-tuning and prompt tuning approaches directly fails. This demonstrates the importance of Projector pre-alignment in the connection process. Of course, if we adopt a train-from-scratch approach for the downstream policy, a two-stage process is not required. However, as shown in Table 2, training from scratch produces inferior results.

###### 2.4. Testing strategy of dual system

Preliminary. A key essence of dual-system models is the need to implement asynchronous control between upper and lower layers. In LCB, the authors did not specifically handle

asynchronous operations, instead using synchronous training followed by asynchronous inference. In HiRT, the authors adopted an additional buffer to introduce asynchronous operations during training as well. For Robodual, they utilized real-time replacement of the upper layer’s coarse actions with actions inferred by the lower layer to perform asynchronous operations. Here, we primarily validated the first approach, with the latter two paradigms to be updated subsequently.

Setup. We evaluated different asynchronous steps from 1 to 60 on CALVIN-D. The step refers to the inference steps of action policy during a single MLLM inference step. The longest environmental steps of the 3DDA are 60.

Analysis. We observe a surprising conclusion in Figure 4: regardless of the number of steps between the large model’s inferences, the performance changes are quite similar. Moreover, even in dynamic scenarios, the experimental results are consistent.

Intuitive hypothesis. This result indicates that the current MLLM is not sensitive to changes in the current environment, which is counterintuitive. Therefore, we need to clarify exactly what information is being transmitted from the upper layer’s latent vector to the lower layer.

Further setup. To explore the underlying reasons, we mapped the latent embeddings of action tokens into semantic space and calculated the similarity of different words to analyze what these action tokens from MLLM convey. The experiment involves dynamic scenarios where a blue block consistently moves to the left. The result is in Figure 5.

Further analysis. We have the following conclusions:

task1 task2 task3 task4 task5

120

94 97 95 95 95 95 95 77

SuccessRate(%)

100

87 85 81 83 79 82 67

76 72

73 60 58 57 54 57

80

64 68

61 47 46 42 39

58

60

50

48

47

40

40

20

0

step=1 step=10 step=20 step=30 step=40 step=50 step=60

- Figure 4. Evaluations on hierarchical inference. We evaluate the performance of the dual system on the CALVIN benchmark, with inference steps set to 1 and 60, respectively.“Steps" refers to the inference steps of action policy during a single MLLM inference step. The longest environmental steps of the action policy [11] are 60, which means MLLM only inference once and represents the most typical asynchronous scenarios.

to</s>blue right, To block the

to</s>blue right To , block and

to blue</s> right To block,?

to blue</s> right To , block the

to blue</s> right To block,?

to blue</s> right To block ,?

to blue</s> block right To, and

to blue</s> block , right To and

to</s>blue right To, block and

to blue</s> block right To, ?

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

Step 0 Step 1 Step 2 Step 3 Step 4 Step 5 Step 6 Step 7 Step 8 Step 9

Top 10 Semantically Similar Words

[Figure 28]

[Figure 29]

|[Figure 30]<br><br>[Figure 31]|
|---|

[Figure 32]

[Figure 33]

0

0.05

0.1

0.15

0.2

0.25

0.3

0.35

0.4

Step 0 Step 1 Step 2 Step 3 Step 4 Step 5 Step 6 Step 7 Step 8 Step 9

Probability

Left right up down forward back

Input: “A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human’s questions. USER: <im_start><image><im_end>\nCan you control the robot to take the blue block and rotate it to the right? ASSISTANT: <ACT>"

- Figure 5. Evaluation on the shortcoming of existing dual systems. From top to bottom, the first row displays the input to the MLLM. The second row visualizes a special scenario where, at environment step 3, the blue block is manually shifted to the left. In the third row, we present the top 10 words that are semantically closest to the latent embedding. The bottom row illustrates the probability distribution of spatial words associated with the latent embedding.

- 1. As to the similarity with spatial words at different time

steps, we observe that regardless of whether the robotic arm moves left or right, the probability of “right" is consistently higher than that of “left," while the probabilities of different spatial prepositions remain almost unchanged over time. This indicates that the action token has learned a semantic feature that remains constant and is unrelated to changes in the environment. The higher probability of “right" compared

to “left" may be due to “right" carrying more semantic information; for example, “right" can also imply correctness, contributing to its consistently high probability.

2. As to Top 10 similar words at different time steps, we observe that the latent embedding primarily encodes the target object, spatial relations, and action semantics from the instruction, along with some noise. It means that the latent embedding mainly summarizes the textual instruction and

###### Table 7. Comparison of the using method of MLLM.

Average Length Trainable Parameters

- 0
- 1,000
- 2,000
- 3,000
- 4,000
- 5,000
- 6,000
- 7,000
- 8,000

Task1 completed2 3 in4a row (%)5 ↑ Avg. Len ↑

Parameters(M)

2.43

AverageLength

Benchmark Type of MLLM Auxiliary tasks Policy

1.78

1.42 1.46

MLLM (Prompt Tuning) × Fine-tuning 94 77 67 60 47 3.45 LLM (Prompt Tuning) × Fine-tuning 77 48 26 16 10 1.77 MLLM (Prompt Tuning) ✓ Fine-tuning 98 92 76 72 63 4.01

CALVIN

0.4

- 0

- 0.5
- 1

1.5

2

- 2.5
- 3

LCB RF 3DDA HiRT* Ours

is largely insensitive to changes in visual information. In other words, the current training method does not effectively leverage the visual reasoning capabilities of the MLLM. Instead, the MLLM merely transmits the semantics of the instructions to the low-level policy.

High-level VLM Low-level Policy

𝒁 𝑨𝑪𝑻 

[Figure 34]

action

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

Diffusion Learning

Multimodal Reasoning Learning

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

Multimodal Large Language Model

Diffusion

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

❄ Policy 🔥 🔥

###### 2.5. Whether the MLLM of dual system is enough?

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

Preliminary. Based on the experimental analysis above, the information currently transmitted through the latent token is insufficient for the downstream model to effectively complete tasks. Therefore, in this section, we aim to explore better ways to utilize upstream information.

RGB Image Language

<ACT>

3D Scene tokens

𝒁 𝑨𝑪𝑻 

### A Simple yet Effective Dual System

Figure 6. Overview of our proposed Dual System VLA.

Setup. The experiments are based on the above conclusions, with the downstream model using fine-tuning, adopting a two-stage projector training approach, and the upstream large model utilizing a prompt tuning training paradigm. However, three variants were created for how to use MLLM: 1. Standard MLLM; 2. Removing visual information from MLLM, treating it purely as an LLM; 3. Introducing an auxiliary loss, allowing the generated latent token to connect to an additional head layer to infer action-related information(position or rotation). The result is in Table 7.

###### 3.1. Architecture

Network. Our system comprises two main components: a pre-trained MLLM fϕ and a pre-trained policy πθ, with parameters ϕ and θ, respectively. The MLLM includes a text-only large language model and a vision encoder, which projects images into the embedding space of the language model, allowing for a multi-modal understanding of textual and visual inputs. The pre-trained policy consists of a vision encoder and transformer-based diffusion model. Using multiple cross-attention layers, the diffusion model incorporates a lot of conditioning information, such as 3D scene representations, proprioception information, and condition/instruction tokens from the high-level model. In this work, we leverage LLaVA [27] as the high-level MLLM and 3D Diffuser Actor

Analysis. From the experimental results, it can be seen that using only LLM produces results far inferior to MLLM, which demonstrates the inherent function of MLLM and shows it hasn’t degraded to simply functioning as an LLM. When we have additional auxiliary tasks, we can see a significant increase in the success rate of tasks. This is mainly because the additional auxiliary tasks force the model to capture more visual information in order to accomplish them, thus compelling the model to pay attention to tasks that a purely MLLM approach would not focus on.

- as the low-level pre-trained diffusion policy. Notably, we use a linear layer to replace the 3D Diffuser Actor’s text encoder, aligning the dimension of the latent embedding output by the large model with the input dimension of the low-level policy.

Input and Output. The whole system is designed to mimic demonstration trajectories in the format {l,(o1,a1),(o2,a2),...}, where l = {wi ∈ Rd}Ni=1 represents a task-specific language instruction of length N with an input dimension d, and ot and at denote the visual observation and corresponding robot action at each timestep t. The input observation ot consists of two RGBD images from different viewpoints. The output action

- at defines the end-effector’s pose, which is decomposed into 3D location, rotation, and gripper state (open/close): at = {alt ∈ R3,art ∈ R6,agt ∈ {0,1}}. The MLLM fϕ

##### 3. A Simple yet Effective Dual System VLA

Based on the above analysis, we employ prompt-tuning to adapt the output of the large model rather than directly finetuning the MLLM itself. Additionally, we introduce an auxiliary task to exploit MLLM’s visual reasoning capabilities fully. This approach results in a more robust latent embedding that effectively integrates visual and textual information.

Rotation

Position 𝒂𝒍

Position noise 𝝐𝒍 Open/close 𝒂𝒈

Rotation Noise 𝝐𝒓

|𝒂𝒓<br><br>[Figure 69]<br><br>[Figure 70]<br><br>[Figure 71]<br><br>🔥<br><br>[Figure 72]|(b)|
|---|---|
|Proje|[Figure 73]<br><br>[Figure 74]<br><br>ction🔥<br><br>[Figure 75]<br><br>[Figure 76]|
|𝒍| |

###### (a) High-Level MLLM Low-Level Policy

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

MLP

MLP

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

🔥 MLP🔥 MLP

[Figure 85]

MLP

[Figure 86]

🔥

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

🔥

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

…

…

[Figure 106]

Self Attention

[Figure 107]

[Figure 108]

🔥 🔥 🔥

Self Attention

###### Denoising step 𝒊

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

###### Large Language Model ❄

[Figure 115]

[Figure 116]

Self Attention

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

MLP

Cross Attention

[Figure 124]

[Figure 125]

[Figure 126]

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

🔥

…

…

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

FiLM

[Figure 144]

[Figure 145]

🔥

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

Proprioception Token 𝒄

[Figure 150]

[Figure 151]

Sampler

Concat

[Figure 152]

Tokenizer

[Figure 153]

[Figure 154]

[Figure 155]

❄

[Figure 156]

Projection

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

MLPMLP

Cross Attention

[Figure 166]

[Figure 167]

🔥 🔥

[Figure 168]

[Figure 169]

In: Can you help me {Task}?Assistant:

[Figure 170]

Vision Encoder

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

...

[Figure 179]

[Figure 180]

Noisy Trajectory Tokens 𝝉𝒊

[Figure 181]

Condition token 𝒛

3D Scene tokens 𝑶

Third-view RGB 𝑶 Task Instruction ?

Figure 7. Detailed framework. (a) The high-level MLLM (left) takes third-view RGB o′, task instruction l, and a learnable token <ACT> as input. After processing through the Large Language Model (LLM), we extract the feature embedding from the final layer of the <ACT> token as the latent goal for the low-level policy. To fully leverage the MLLM’s multimodal reasoning capability, we propose an auxiliary task, using MLPs to predict the action (location al, rotation ar, open/close ag) based on this feature embedding z<ACT>, ensuring it encapsulates both visual and textual information. (b) The low-level policy (right) receives the latent goal from the high-level MLLM, combines it with 3D scene tokens o and proprioception token c, and iteratively predicts action noise ϵ to produce an accurate action trajectory τ and gripper state ag. Notably, our approach keep all parameters of the MLLM frozen and fine-tune the learnable prompt to adjust the MLLM’s output, significantly reducing training costs compared to previous methods.

processes language instruction l and the third-view RGB image o′t, outputting the latent embedding zt for low-level policy. The low-level pre-trained policy πθ takes as input the noisy trajectory τti, diffusion step i, and the conditioning information from the environment observation ot, the latent embedding zt, and proprioception ct of timestep t, predicting the action trajectory τt = (alt:t+T,art:t+T) and binary states agt:t+T at each timestep t, over a temporal horizon T.

timodal reasoning capability of the MLLM fully. This task is very simple and requires no additional data preparation process. The output embedding zt<ACT> = fϕ(o′t,l′) from the learnable prompt token is passed through linear layers to predict the action trajectories τt and gripper actions agt. Through supervised training on this task, we ensure that the large model has to utilize visual input information and that the latent embedding contains a blend of multimodal information. The loss function is defined as follows:

###### 3.2. Training

Llm(<ACT>) = BCE(MLP(fϕg(o′t,l′)),agt:t+T)

Prompt Tuning. In order to avoid the degradation of MLLM, we introduce one learnable token <ACT> ∈ Rd at end of language instruction l. The new instruction l′ is defined as l′ = {l,<ACT>}. During training, all parameters of MLLM are frozen; we only update the embedding of learnable token <ACT>.

- + ω1 · ||MLP(fϕl (o′t,l′)) − alt:t+T||
- + ω2 · ||MLP(fϕr(o′t,l′)) − art:t+T||,

(1)

where ω1, ω2 are hyperparameters to balance the effect of each loss item, and MLP represents linear layer. To reconstruct the sequence of 3D locations and 3D rotations, we apply the L1loss. Additionally, we supervise the end-effector opening using binary cross-entropy loss (BCE).

Multimodal Reasoning Learning. As we discussed in section 3.3, we know that these previous methods do not fully utilize MLLM’s visual reasoning capability. Specifically, they align the output of the large MLLM model with the output from the text encoder of CLIP. Using purely textual information to supervise the fine-tuning of the MLLM can lead to the degradation of multimodal reasoning capability. Therefore, we designed an auxiliary task to leverage the mul-

Diffusion Learning. Following the previous diffusion-based approach [6, 11, 23], we train our model using the action denoising objective. During training, we randomly sample a time step t and a diffusion step i, adding noise ϵ = (ϵl,ϵr)

Table 8. Results on CALVIN ABC-D: We report both success rates and average task completion length (out of 5 tasks) per evaluation sequence. MLLM (PT) denotes our proposed prompt tuning method for MLLM training. Policy(P) indicates loading from a pretrained policy model. Asy(10) represents inference with a 10-step time delay. AUX denotes the additionally introduced auxiliary tasks.

1 Task completed2 3in a row4(%) ↑ 5 Avg. Len. ↑

Type Method

Only Policy 92.2 78.7 63.9 51.2 41.2 3.27 MLLM (PT) + Policy(P) 92.2 79.2 65.0 52.9 40.9 3.30 MLLM (PT) + AUX + Policy(P) + Asy(10) 93.3 81.8 67.9 56.6 46.0 3.45 MLLM (PT) + AUX + Policy(P) + Asy(60) 92.8 79.7 67.5 57.3 46.9 3.44

CALVIN

Only Policy 65.2 39.1 20.3 11.7 6.1 1.42 MLLM (PT) + Policy(P) 71.3 44.9 28.4 17.5 10.3 1.72 MLLM (PT) + AUX + Policy(P) + Asy(10) 78.9 57.1 40.2 29.5 20.2 2.26 MLLM (PT) + AUX + Policy(P) + Asy(60) 78.1 56.5 38.9 27.0 19.5 2.20

CALVIN-E

to a ground-truth trajectory τt0. The objective is defined as:

Lpolicy(θ,<ACT>) = BCE(πθg(ot,zt<ACT>,ct,τti,i),agt:t+T)

- + ω3 · ||ϵlθ(ot,zt<ACT>,ct,τti,i) − ϵlt:t+T||
- + ω4 · ||ϵrθ(ot,zt<ACT>,ct,τti,i) − ϵrt:t+T||,

(2) where ω3, ω4 are also hyperparameters to balance loss items. Please refer to [1] for the details of the loss function.

Two stage training. We adopt a two-stage training approach to train our proposed dual system. In the first stage, to initially align the embedding produced by the MLLM with the feature space of the pre-trained policy, we freeze the parameters of the large model and the low-level policy, training only the prompt and projection layers. In the second stage, we keep the large model frozen and unfreeze the low-level policy, fine-tuning it together with the prompt and projection. The objectives in both stages remain unchanged. The only difference between the two stages is whether the low-level policy is frozen. In summary, our loss function includes two components and can be defined as follows:

###### Ltotal = Llm + Lpolicy (3)

Implementation Details. We use LLaVA-7B[15] and 3D diffuser Actor[11] as the high-level MLLM and low-level policy models, respectively. We select the checkpoint of 3D diffuser Actor at 65,000 iterations as the pre-trained parameters. During training, the first stage (pre-alignment) is conducted for 2,000 iterations, and the second stage continues until 100,000 iterations. The projection is a linear layer that reduces the output dimension of the large model from 4096 to 512. We manually add an <ACT>token to LLava’s tokenizer and freeze all parameters of the MLLM, fine-tuning only the newly added token embedding. The remaining experimental and training settings are consistent with 3D diffuser Actor; please refer to [11] for details.

###### 3.3. Results

From Table 8, we can draw conclusions similar to those from the previous empirical study:

- 1. The crucial role of integrating the upper and lower lay-

ers of the model lies in improving performance in language generalization scenarios.

- 2. Additional auxiliary tasks are very helpful for enhanc-

ing both standard task performance and generalization performance, mainly because they improve the model’s action capability.

- 3. Asynchronous inference has little impact on the in-

ference performance of the general task model; even if the model only performs asynchronous inference once (Asy (60)), the final performance remains largely unchanged.

- 4. Discussion & Limitation

We first acknowledge that there is still a long way to go before we achieve a full open-source reproduction of Helix.

1. Deploying on real robots. 2. Achieving sufficiently fast downstream policy execution. 3. Successfully running on physical robots. 4. Deploying on humanoid robots. 5. Realizing collaboration between humanoid robots.

There is indeed much work to be done before all of the above goals are achieved. However, this technical report is only our initial version. We are committed to continuously updating this project to fulfill the open-source objectives for all the tasks mentioned above. In addition, we maintain an open attitude toward some of the claims in this article that have not yet been fully verified. We hope that more researchers will join our team, or that more people will support us, so that we can accomplish this meaningful work for the entire community. Since some of the organization was done rather hastily, if any corrections are needed, all authors are welcome to contact me at any time.

##### References

- [1] S Belkhale and D Sadigh. Minivla: A better vla with a smaller footprint. 2024. 3
- [2] Johan Bjorck, Fernando Castañeda, Nikita Cherniadev, Xingye Da, Runyu Ding, Linxi Fan, Yu Fang, Dieter Fox, Fengyuan Hu, Spencer Huang, et al. Gr00t n1: An open foundation model for generalist humanoid robots. arXiv preprint arXiv:2503.14734, 2025. 2, 3, 5, 6
- [3] Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, et al. A vision-languageaction flow model for general robot control. arXiv preprint arXiv:2410.24164, 2024. 2, 3, 5
- [4] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Joseph Dabis, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Jasmine Hsu, Julian Ibarz, Brian Ichter, Alex Irpan, Tomas Jackson, Sally Jesmonth, Nikhil J Joshi, Ryan Julian, Dmitry Kalashnikov, Yuheng Kuang, Isabel Leal, Kuang-Huei Lee, Sergey Levine, Yao Lu, Utsav Malla, Deeksha Manjunath, Igor Mordatch, Ofir Nachum, Carolina Parada, Jodilyn Peralta, Emily Perez, Karl Pertsch, Jornell Quiambao, Kanishka Rao, Michael Ryoo, Grecia Salazar, Pannag Sanketi, Kevin Sayed, Jaspiar Singh, Sumedh Sontakke, Austin Stone, Clayton Tan, Huong Tran, Vincent Vanhoucke, Steve Vega, Quan Vuong, Fei Xia, Ted Xiao, Peng Xu, Sichun Xu, Tianhe Yu, and Brianna Zitkovich. Rt-1: Robotics transformer for real-world control at scale, 2023. 2
- [5] Qingwen Bu, Hongyang Li, Li Chen, Jisong Cai, Jia Zeng, Heming Cui, Maoqing Yao, and Yu Qiao. Towards synergistic, generalized, and efficient dual-system for robotic manipulation. arXiv preprint arXiv:2410.08001, 2024. 2, 3, 4, 5
- [6] Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, and Shuran Song. Diffusion policy: Visuomotor policy learning via action diffusion. The International Journal of Robotics Research, page 02783649241273668, 2023. 10
- [7] Jonathan St BT Evans. Dual-processing accounts of reasoning, judgment, and social cognition. Annu. Rev. Psychol., 59

(1):255–278, 2008. 1

- [8] Zhefei Gong, Pengxiang Ding, Shangke Lyu, Siteng Huang, Mingyang Sun, Wei Zhao, Zhaoxin Fan, and Donglin Wang. Carp: Visuomotor policy learning via coarse-to-fine autoregressive prediction. arXiv preprint arXiv:2412.06782, 2024. 3
- [9] ByungOk Han, Jaehong Kim, and Jinhyeok Jang. A dual process vla: Efficient robotic manipulation leveraging vlm. In Conference on Robot Learning (CoRL), 2024. 2, 3, 4
- [10] Daniel Kahneman. Thinking, fast and slow. macmillan, 2011. 1
- [11] Tsung-Wei Ke, Nikolaos Gkanatsios, and Katerina Fragkiadaki. 3d diffuser actor: Policy diffusion with 3d scene representations. arXiv preprint arXiv:2402.10885, 2024. 2, 4, 8, 10, 11
- [12] Bolin Lai, Xiaoliang Dai, Lawrence Chen, Guan Pang, James M Rehg, and Miao Liu. Lego: L earning ego centric action frame generation via visual instruction tuning. In

European Conference on Computer Vision, pages 135–155. Springer, 2024. 3

- [13] Xinghang Li, Minghuan Liu, Hanbo Zhang, Cunjun Yu, Jie Xu, Hongtao Wu, Chilam Cheang, Ya Jing, Weinan Zhang, Huaping Liu, et al. Vision-language foundation models as effective robot imitators. arXiv preprint arXiv:2311.01378,

- 2023. 3, 5

[14] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. arXiv preprint arXiv:2304.08485,

- 2023. 4

- [15] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. 11
- [16] Wim De Neys. Dual processing in reasoning: Two systems but one reasoner. Psychological science, 17(5):428–433, 2006. 1
- [17] Xichen Pan, Satya Narayan Shukla, Aashu Singh, Zhuokai Zhao, Shlok Kumar Mishra, Jialiang Wang, Zhiyang Xu, Jiuhai Chen, Kunpeng Li, Felix Juefei-Xu, et al. Transfer between modalities with metaqueries. arXiv preprint arXiv:2504.06256, 2025. 3
- [18] Moritz Reuss, Hongyi Zhou, Marcel Rühle, Ömer Erdinç Ya˘gmurlu, Fabian Otto, and Rudolf Lioutikov. Flower: Democratizing generalist robot policies with efficient visionlanguage-action flow policies. In 7th Robot Learning Workshop: Towards Robots with Human-Level Abilities. 3
- [19] Yide Shentu, Philipp Wu, Aravind Rajeswaran, and Pieter Abbeel. From llms to actions: Latent codes as bridges in hierarchical robot control. arXiv preprint arXiv:2405.04798,

2024. 1, 2, 3, 4, 5

- [20] Yue Su, Xinyu Zhan, Hongjie Fang, Han Xue, Hao-Shu Fang, Yong-Lu Li, Cewu Lu, and Lixin Yang. Dense policy: Bidirectional autoregressive learning of actions. arXiv preprint arXiv:2503.13217, 2025. 3
- [21] Amos Tversky and Daniel Kahneman. Judgment under uncertainty: Heuristics and biases: Biases in judgments reveal some heuristics of thinking under uncertainty. science, 185

(4157):1124–1131, 1974. 1

- [22] Junjie Wen, Yichen Zhu, Jinming Li, Zhibin Tang, Chaomin Shen, and Feifei Feng. Dexvla: Vision-language model with plug-in diffusion expert for general robot control. arXiv preprint arXiv:2502.05855, 2025. 2, 4
- [23] Yanjie Ze, Gu Zhang, Kangning Zhang, Chenyuan Hu, Muhan Wang, and Huazhe Xu. 3d diffusion policy: Generalizable visuomotor policy learning via simple 3d representations. In ICRA 2024 Workshop on 3D Visual Representations for Robot Manipulation, 2024. 10
- [24] Jianke Zhang, Yanjiang Guo, Xiaoyu Chen, Yen-Jen Wang, Yucheng Hu, Chengming Shi, and Jianyu Chen. Hirt: Enhancing robotic control with hierarchical robot transformers. arXiv preprint arXiv:2410.05273, 2024. 2, 3, 4
- [25] Minjie Zhu, Yichen Zhu, Jinming Li, Junjie Wen, Zhiyuan Xu, Ning Liu, Ran Cheng, Chaomin Shen, Yaxin Peng, Feifei Feng, et al. Scaling diffusion policy in transformer to 1 billion parameters for robotic manipulation. arXiv preprint arXiv:2409.14411, 2024. 2

