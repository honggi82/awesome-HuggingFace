## Prompt Relay: Inference-Time Temporal Control for Multi-Event Video Generation

Gordon Chen Ziqi Huang Ziwei Liu S-Lab, Nanyang Technological University https://gordonchen19.github.io/Prompt-Relay/

# arXiv:2604.10030v1[cs.CV]11Apr2026

|[Figure 1]|[Figure 2]|[Figure 3]|[Figure 4]|[Figure 5]|[Figure 6]|
|---|---|---|---|---|---|

[0-2s] The camera zooms toward the eagle’s eye as it flies. Inside the pupil, a cyberpunk city is already visible … [2-4s] … Cars

move close to the camera in layered traffic lanes ... [4-6s] The camera … starts to track and lock onto a car ... [6-10s] The camera

slowly zooms out … revealing that the cyberpunk scene is playing on a television screen … inside a 20th century living room …

|[Figure 7]|[Figure 8]|[Figure 9]|[Figure 10]|[Figure 11]|[Figure 12]|
|---|---|---|---|---|---|

[0-2s] A … rugged caveman walking … by the setting sun ... [2-3s] The camera whips downwards ... fills with a motion-blurred

grass texture. [3-5s] The camera whips rapidly upwards … revealing … a Spartan ... [5-6s] The camera whips rapidly downwards …

[6-8s] The camera whips rapidly upwards … tracks a majestic medieval knight in shining plate armor riding the horse ...

|[Figure 13]|[Figure 14]|[Figure 15]|[Figure 16]|[Figure 17]|[Figure 18]|
|---|---|---|---|---|---|

[0-1s] … A young boy is lying flat on his … staring up at the ceiling. [1-3s] After a brief moment, he rolls over, pushes himself up,

stands on the mattress, and starts jumping on the bed. He bounces up and down repeatedly ... [3-6s] The boy then runs toward a pile of toys near the corner of the room, grabs a toy airplane, and pretends to fly it through the air …

Figure 1. Prompt Relay is an inference-time, training-free, plug-and-play method for enabling fine-grained temporal control by routing each textual prompt to its intended time segment, allowing multiple events to occur in the correct order without semantic interference.

### Abstract

ment, where concepts intended for different moments in the video bleed into one another, resulting in poor text-video alignment. To address these limitations, we propose Prompt Relay, an inference-time, plug-and-play method to enable fine-grained temporal control in multi-event video generation, requiring no architectural modifications and no additional computational overhead. Prompt Relay introduces a penalty into the cross-attention mechanism, so that each temporal segment attends only to its assigned prompt, allowing the model to represent one semantic concept at a time and thereby improving temporal prompt alignment, reducing semantic interference, and enhancing visual quality.

Video diffusion models have achieved remarkable progress in generating high-quality videos. However, these models struggle to represent the temporal succession of multiple events in real-world videos and lack explicit mechanisms to control when semantic concepts appear, how long they persist, and the order in which multiple events occur. Such control is especially important for movie-grade video synthesis, where coherent storytelling depends on precise timing, duration, and transitions between events. When using a single paragraph-style prompt to describe a sequence of complex events, models often exhibit semantic entangle-

Wednesday, April 8, 2026 Sample Footer Text 21

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

| | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | |Penalty|Strength| | | | | |

K(Text Tokens)

… pours milk …

… pours cereal …

Q (Video Tokens)

Figure 2. Temporal Cross-Attention Routing. Each textual prompt is associated with a specific temporal segment of the video. The attention penalty varies smoothly across time, allowing video tokens to attend strongly to their corresponding prompt within the assigned interval while suppressing attention to temporally irrelevant prompts. This enables multiple events (e.g., pouring cereal followed by pouring milk) to occur in the correct order without semantic interference.

### 1. Introduction

Prompt Relay requires no computational overhead and no architectural modifications. Our main contributions are as follows:

Recent advances in video diffusion models have enabled the generation of high-quality videos conditioned on textual prompts, achieving impressive visual fidelity and motion coherence [2–4, 23, 30]. Despite this progress, existing models are optimized for single-event generation and offer no mechanism for explicit temporal control - users cannot specify when an event occurs, how long it persists for and how multiple events are ordered. As a result, modeling movie-grade videos composed of a succession of events, actions, or camera motions, each occurring within a specific segment of the video and in a specific order, remains challenging. This limitation stems from the lack of temporal awareness in the cross-attention mechanism: by conditioning every frame of the video on the entire prompt simultaneously, the model treats a multi-event prompt as global context rather than a temporally structured sequence, causing semantic concepts intended for different moments to bleed into one another, degrading text-video alignment.

- • We propose Prompt Relay, a test-time, plug-and-play method for fine-grained temporal control in video generation with no computational overhead.
- • We propose a Boundary-Attention decay mechanism, a soft Gaussian penalty on cross-attention logits that smoothly suppressess semantic interference across segment boundaries.
- • We demonstrate that Prompt Relay substantially improves temporal prompt alignment, reduces semantic interference and enhances visual quality.

### 2. Related Works

#### 2.1. Controllable Video Generation

Video generation has seen rapid progress in recent years, with applications spanning motion control [6, 9, 24–26], viewpoint control [7, 15, 22], identity control [17, 18, 33] and editing [8, 19]. However, most models remain limited in the ability to generate coherent multi-event videos. Because the attention mechanism allows every pixel to attend to every prompt token, models struggle to associate semantic concepts with their intended temporal intervals, leading to temporal misalignment and semantic entanglement. This challenge motivates us to provide explicit temporal control at inference time.

Recent works have begun to address temporal controllability in video generation [10, 21, 27–29, 31]. One line of work [27, 29] finetunes the backbone model with temporally grounded supervision. However, these methods require large amounts of annotated data, training and shifts the pre-trained model’s distribution. Inference-time attention control methods [10, 28, 31] avoid training altogether, but impose structural constraints on the attention mechanism that limit their generality and can introduce visual artifacts at segment boundaries.

#### 2.2. Attention-Based Control in Diffusion Models

In this paper, we propose Prompt Relay, a simple and elegant attention-level routing mechanism for fine-grained temporal control and multi-event video generation. Prompt Relay operates entirely at inference time and is plug-andplay compatible with existing video diffusion backbones.

Attention manipulation has emerged as a key mechanism for controllable diffusion generation. Prior work has explored attention for spatial [12–14, 16, 32], identity [11, 34] and motion control [19, 20, 24]. In contrast, attention-based temporal control remains largely underexplored.

#### 2.3. Multi-Event Video Generation

A notable approach to temporal modeling for multi-event video generation is MinT [27], which introduces a trainable temporal cross-attention module that binds event descriptions to predefined time intervals, but requires additional training, architectural modifications, and temporally annotated data. MEVG [21] generates each event clip sequentially, conditioning on the last frame of the previous clip via latent inversion to maintain visual continuity. However, this autoregressive design causes error accumulation across segments and produces abrupt transitions when consecutive events are semantically dissimilar. DiTCtrl [10] proposes mask-guided KV-sharing within MM-DiT’s 3D fullattention, enabling prompt-specific semantic control without training. However, the binary attention masks derived from the attention map introduce hard boundaries that can cause background inconsistencies and unnatural transitions. TS-Attn [31] and SwitchCraft [28] instead modulate crossattention by identifying motion-relevant tokens, TS-Attn via a subject semantic layout, and SwitchCraft via eventspecific anchor tokens. Both methods therefore assume the presence of a dominant foreground subject in each event and struggle with scene-level changes or events where no single entity dominates the frame.

### 3. Prompt Relay

Given a sequence of temporally-constrained text prompts {(ps,tstarts ,tends )}Ns=1, our goal is to generate a video such that each arbitrary prompt ps is realized within its designated temporal interval [tstarts ,tends ]. The generated video should preserve global coherence while ensuring that each prompt influences only its assigned temporal region.

#### 3.1. Preliminaries

Cross-attention is a mechanism that enables a diffusion model to incorporate external conditioning information, such as text prompts, into the generation process. Given a latent representation at diffusion step t, denoted as ϕ(zt), and a set of conditioning embeddings ψ(P) derived from an input prompt P, cross-attention computes interactions between the two through learned projections.

Attn(ϕ(zt),ψ(P)) = Softmax

QK⊤ √

d

V, (1)

where Q = ℓQϕ(zt) are query vectors derived from latent features, K = ℓKψ(P) and V = ℓV ψ(P) are key and value vectors projected from the conditioning embeddings, and d denotes the projection dimensionality. Each attention weight reflects how strongly a latent query attends to a particular conditioning token. Through this operation, semantic information from the conditioning input is selectively injected into the latent representation, allowing different

[Figure 23]

Figure 3. Ablation Study of the Temporal Penalty Function. The curves show the attention fraction retained between a query token and the prompt tokens of a given segment, as a function of the query’s latent frame offset from that segment’s midpoint ms, after applying the penalty exp(−C(i, j)). (Top) Effect of the window parameter w. w = L − 2 preserves full attention within the segment and only suppresses attention near the segment boundaries. (Bottom) Effect of the decay threshold ϵ. Smaller values enforce stronger attenuation outside the ’free-attention’ window; however, we find that the choice among small values has negligible perceptual impact. We adopt ϵ = 0.1 as our default.

queries to respond to different aspects of the prompt. However, because attention is computed globally over all conditioning tokens, multiple semantic concepts may compete for influence over the same latent queries. When these concepts correspond to different temporal regions, unrestricted attention can lead to interference between instructions.

#### 3.2. Temporal Prompt Routing

In order to enforce the association between each prompt ps and its assigned temporal interval [tstarts ,tends ], we introduce a penalty term C(Q,K) into the cross-attention logits:

QK⊤ √

d − C(Q,K) V.

Attn(ϕ(zt),ψ(P)) = softmax

(2) The role of C(Q,K) is to suppress the attention between key and query tokens whenever they do not belong to the same interval [tstarts ,tends ]. This allows each prompt to guide generation only within its intended segment, without leak-

A man eats pasta at a restaurant table → A woman in a red dress and sunglasses walks past

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

Hard Masking

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

Boundary-Attention Decay (Ours)

- Figure 4. Hard Masking vs Boundary-Attention Decay. Hard masking enforces an abrupt semantic switch in cross-attention at segment boundaries while self-attention remains continuous across the segments. This creates a discontinuity at the boundary, forcing the model to reconcile conflicting signals (Woman eats the pasta instead of the man). Boundary-attention decay avoids this conflict by smoothly coactivating both neighboring prompts near the boundary, giving the model a gradual handoff region in which the transition can be planned jointly before being committed to in the visual representation.

Hard masking forces an abrupt semantic switch in cross-attention at the boundaries of two segments while self-attention remains anchored

to the previous segment's visual structure, creating a conflict the model resolves by incorrectly morphing the existing subject; soft decay

#### 3.3. Boundary-Attention Decay

ing semantic concepts into other parts of the video. For any arbitrary query token indexed by i and any key token j belonging to ps, the penalty is defined as:

To suppress semantic interference across temporal segments, attention between queries near segment boundaries and prompt tokens from neighboring segments should be negligible. We therefore choose the decay parameter σ so that the attention prior sufficiently decreases near segment endpoints. Since our penalty subtracts C(i,j) from the logits, it applies a multiplicative factor exp − C(i,j) to the unnormalized attention scores before softmax. This prior is 1 inside the “free-attention” window and decays toward the segment boundaries. Let the endpoint distance from the segment midpoint be L = |f(i) − ms|. We choose σ such that the prior reaches a small value ϵ at the endpoints:

instead co-activates both prompts at boundary frames, allowing the compositional transition to be planned jointly in cross-attention before

it is committed to in the visual representation.

ReLU(|f(i) − ms| − w)2 2σ2

C(i,j) =

,

tstarts + tends 2

. (3)

ms =

Here, f(i) denotes the latent frame index associated with query token i, and ms denotes the midpoint of the corresponding temporal segment. The parameter w defines a local window around the segment midpoint within which no penalty is applied, while σ controls the rate at which attention decays outside this window. Query tokens within the window incur zero penalty and can attend freely to their associated prompt tokens. Beyond this region, attention is smoothly attenuated as a function of the temporal distance between the query and the segment midpoint. We demonstrate in Fig. 3, that w = L − 2 achieves the best balance between temporal isolation and intra-segment fidelity.

(L − w)2 2σ2

L − w 2ln(1/ϵ)

. (4)

exp −

= ϵ ⇒ σ =

This formulation ensures smooth transitions between neighboring prompts while preventing destructive interference across segments. As a result, each textual instruction primarily influences its intended temporal region, allowing the model to focus on one semantic concept at a time while maintaining global temporal coherence.

We compare our approach to hard masking in Fig. 4. Hard masking sets C(i,j) = −∞ for all query-key pairs where f(i) ∈/ [tstarts ,tends ] and j belongs to prompt ps (i.e. a query either attends fully to a prompt or is completely blocked from it). This enforces a sudden switch between prompts at segment boundaries. While hard masking eliminates cross-segment semantic interference, it creates a discontinuity at the boundary: cross-attention switches abruptly to the new prompt while self-attention remains anchored to the previous segment’s visual structure, forcing the model to reconcile conflicting signals. Boundary-attention decay avoids this conflict by smoothly co-activating both neighboring prompts near the boundary, giving the model a gradual handoff region in which the transition can be planned jointly before being committed to in the visual representation.

### 4. Experiments 4.1. Experimental Setup

We apply Prompt Relay on top of the state-of-the-art pretrained video generation model Wan2.2-T2V-A14B. To demonstrate the limitations of existing video generators in handling multi-event prompts, we test several other models, including Sora Storyboard [3], Veo 3.1[4], Wan 2.2[5], and Kling 2.6[2]. We set ϵ = 0.1 across all experiments. Setting w = L−2 reduces σ to a constant. In addition to selectively routing local prompts to their assigned temporal segments, we include a global prompt that conditions the entire video and provides persistent context.

Thursday, April 9, 2026 Sample Footer Text 22

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

Sora

(Storyboard)

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

Kling 2.6

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

Veo 3.1

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

Wan 2.2

|[Figure 50]|[Figure 51]|
|---|---|

|[Figure 52]|
|---|

|[Figure 53]|
|---|

Wan 2.2 +

Prompt Relay

(Ours)

Prompt: A handheld, front-facing, selfie perspective of a man filming himself at arm’s length ... The man … standing on a

busy street in Hong Kong. Neon signs glow behind him, skyscrapers loom overhead, and crowds move in the background ... The man raises his hand toward the camera. His palm moves closer until his hand completely fills the frame … The hand

pulls away from the lens, revealing the man in the same framing but now is filming himself in the grand canyons.

- Figure 5. Qualitative Comparison. Given a multi-event prompt describing a deliberate scene transition, Prompt Relay preserves correct temporal structure, ensuring that each semantic instruction influences only its intended segment while maintaining global visual coherence.

#### 4.2. Evaluation Metrics

sitions, multi-character interactions, and complex camera trajectories, randomly generated with ChatGPT [1]. These scenarios each contain 3-6 temporal events. Participants were shown videos alongside their corresponding prompt, with model identity withheld, and asked to rank each video on a scale of 1–5 per criterion. Final scores are computed as the average rank across all participants (30) and scenarios.

Existing quantitative metrics test visual fidelity or global text-video alignment, but fail to capture temporal semantics or transition quality, properties that are inherently perceptual. Hence, we conduct a human preference study to evaluate multi-event video generation along three dimensions:

- • Temporal Prompt Alignment: Whether each prompt is realized in its intended temporal interval.
- • Transition Naturalness: The perceptual smoothness of transitions between consecutive events, including the absence of abrupt cuts, flickering, or unnatural morphing at segment boundaries.
- • Visual Quality: Overall perceptual fidelity of the generated video, including sharpness, temporal consistency, and absence of visual artifacts.

#### 4.3. Results

As shown in Table. 1, Prompt Relay consistently outperforms baseline approaches in temporal alignment and transition naturalness. Notably Wan 2.2 with Prompt Relay consistently exhibits stronger visual quality compared to the baseline Wan 2.2. This is likely because Prompt Relay’s attention routing mechanism suppresses attention between queries in a particular temporal segment and prompts belonging to other segments. By reducing unnecessary com-

We construct 20 diverse multi-event test scenarios, covering a wide range of settings including explicit scene tran-

##### Metric Storyboard Kling 2.6 Veo 3.1 Wan 2.2 Wan 2.2 + Prompt Relay

Temporal Prompt Alignment (↓) 4.67 1.30 3.93 4.00 1.10 Transition Naturalness (↓) 4.60 4.43 1.30 3.50 1.17 Visual Quality (↓) 3.67 2.50 2.0 4.00 2.83

Table 1. Human preference scores for multi-event video generation. (lower values indicate better rankings)

petition in the cross-attention space, the model can allocate attention more effectively to the active semantic concepts, resulting in clearer visual structure, improved temporal alignment, and more stable generation. However, Kling 2.6 and Veo 3.1 still achieve higher visual quality overall, indicating that visual fidelity remains partially bounded by the capacity of the underlying backbone model.

### 5. Limitations

Since each temporal segment attends primarily to its corresponding local prompt, persistent visual elements such as characters, objects, or scene style are not explicitly shared across segments. If these elements are described inconsistently across local prompts, their appearance may drift over time. We found that we can fully mitigate this by incorporating a global prompt that provides shared context and anchors persistent elements across multiple segments.

### 6. Conclusion

We present Prompt Relay, an inference-time, plug-and-play method for multi-event video generation with fine-grained temporal control. We also show that our method improves visual quality over the backbone model. We view our work as a pivotal step towards movie-grade, controllable video synthesis.

### Acknowledgments

This research is supported by cash and in-kind funding from NTU S-Lab and industry partner(s). This study is also supported by the Ministry of Education, Singapore, under its MOE AcRF Tier 2 (MOE-T2EP202230002).

### References

- [1] Chatgpt 5.2. Accessed January 15, 2026 [Online], 2025. 5
- [2] Kling 2.6. Accessed January 15, 2026 [Online], 2025. 2, 4
- [3] Sora. Accessed January 15, 2026 [Online] https:// sora.chatgpt.com/explore, 2025. 4
- [4] Veo 3.1. Accessed January 15, 2026 [Online], 2025. 2, 4
- [5] Wan 2.2. Accessed January 15, 2026 [Online], 2025. 4
- [6] Rameen Abdal, Or Patashnik, Ivan Skorokhodov, Willi Menapace, Aliaksandr Siarohin, Sergey Tulyakov, Daniel Cohen-Or, and Kfir Aberman. Dynamic concepts personalization from single videos. In Proceedings of the Special

- Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers, 2025. 2
- [7] Jianhong Bai, Menghan Xia, Xiao Fu, Xintao Wang, Lianrui Mu, Jinwen Cao, Zuozhu Liu, Haoji Hu, Xiang Bai, Pengfei Wan, et al. Recammaster: Camera-controlled generative rendering from a single video. arXiv preprint arXiv:2503.11647, 2025. 2
- [8] Yuxuan Bian, Zhaoyang Zhang, Xuan Ju, Mingdeng Cao, Liangbin Xie, Ying Shan, and Qiang Xu. Videopainter: Anylength video inpainting and editing with plug-and-play context control. In Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers, 2025. 2
- [9] Ryan Burgert, Yuancheng Xu, Wenqi Xian, Oliver Pilarski, Pascal Clausen, Mingming He, Li Ma, Yitong Deng, Lingxiao Li, Mohsen Mousavi, et al. Go-with-the-flow: Motioncontrollable video diffusion models using real-time warped noise. In Proceedings of the Computer Vision and Pattern Recognition Conference, 2025. 2
- [10] Minghong Cai, Xiaodong Cun, Xiaoyu Li, Wenze Liu, Zhaoyang Zhang, Yong Zhang, Ying Shan, and Xiangyu Yue. Ditctrl: Exploring attention control in multi-modal diffusion transformer for tuning-free multi-prompt longer video generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, 2025. 2, 3
- [11] Shengqu Cai, Ceyuan Yang, Lvmin Zhang, Yuwei Guo, Junfei Xiao, Ziyan Yang, Yinghao Xu, Zhenheng Yang, Alan Yuille, Leonidas Guibas, et al. Mixture of contexts for long video generation. arXiv preprint arXiv:2508.21058, 2025. 2
- [12] Mingdeng Cao, Xintao Wang, Zhongang Qi, Ying Shan, Xiaohu Qie, and Yinqiang Zheng. Masactrl: Tuning-free mutual self-attention control for consistent image synthesis and editing. In Proceedings of the IEEE/CVF international conference on computer vision, 2023. 2
- [13] Hila Chefer, Yuval Alaluf, Yael Vinker, Lior Wolf, and Daniel Cohen-Or. Attend-and-excite: Attention-based semantic guidance for text-to-image diffusion models. ACM transactions on Graphics (TOG), 2023.
- [14] Gordon Chen, Ziqi Huang, Cheston Tan, and Ziwei Liu. Stencil: Subject-driven generation with context guidance. In 2025 IEEE International Conference on Image Processing (ICIP). IEEE, 2025. 2
- [15] Hao He, Yinghao Xu, Yuwei Guo, Gordon Wetzstein, Bo Dai, Hongsheng Li, and Ceyuan Yang. Cameractrl: Enabling camera control for text-to-video generation. arXiv preprint arXiv:2404.02101, 2024. 2
- [16] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt im-

- age editing with cross attention control. arXiv preprint arXiv:2208.01626, 2022. 2
- [17] Teng Hu, Zhentao Yu, Zhengguang Zhou, Sen Liang, Yuan Zhou, Qin Lin, and Qinglin Lu. Hunyuancustom: A multimodal-driven architecture for customized video generation. arXiv preprint arXiv:2505.04512, 2025. 2
- [18] Lijie Liu, Tianxiang Ma, Bingchuan Li, Zhuowei Chen, Jiawei Liu, Gen Li, Siyu Zhou, Qian He, and Xinglong Wu. Phantom: Subject-consistent video generation via crossmodal alignment. arXiv preprint arXiv:2502.11079, 2025. 2
- [19] Shaoteng Liu, Yuechen Zhang, Wenbo Li, Zhe Lin, and Jiaya Jia. Video-p2p: Video editing with cross-attention control. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024. 2
- [20] Tuna Han Salih Meral, Hidir Yesiltepe, Connor Dunlop, and Pinar Yanardag. Motionflow: Attention-driven motion transfer in video diffusion models. arXiv preprint arXiv:2412.05275, 2024. 2
- [21] Gyeongrok Oh, Jaehwan Jeong, Sieun Kim, Wonmin Byeon, Jinkyu Kim, Sungwoong Kim, and Sangpil Kim. Mevg: Multi-event video generation with text-to-video models. In European Conference on Computer Vision. Springer, 2024. 2, 3
- [22] Xuanchi Ren, Tianchang Shen, Jiahui Huang, Huan Ling, Yifan Lu, Merlin Nimier-David, Thomas M¨uller, Alexander Keller, Sanja Fidler, and Jun Gao. Gen3c: 3d-informed world-consistent video generation with precise camera control. In Proceedings of the Computer Vision and Pattern Recognition Conference, 2025. 2
- [23] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 2
- [24] Luozhou Wang, Ziyang Mai, Guibao Shen, Yixun Liang, Xin Tao, Pengfei Wan, Di Zhang, Yijun Li, and Ying-Cong Chen. Motion inversion for video customization. In Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers, 2025. 2
- [25] Xiang Wang, Hangjie Yuan, Shiwei Zhang, Dayou Chen, Jiuniu Wang, Yingya Zhang, Yujun Shen, Deli Zhao, and Jingren Zhou. Videocomposer: Compositional video synthesis with motion controllability. Advances in Neural Information Processing Systems, 2023.
- [26] Zhouxia Wang, Ziyang Yuan, Xintao Wang, Yaowei Li, Tianshui Chen, Menghan Xia, Ping Luo, and Ying Shan. Motionctrl: A unified and flexible motion controller for video generation. In ACM SIGGRAPH 2024 Conference Papers, 2024. 2
- [27] Ziyi Wu, Aliaksandr Siarohin, Willi Menapace, Ivan Skorokhodov, Yuwei Fang, Varnith Chordia, Igor Gilitschenski, and Sergey Tulyakov. Mind the time: Temporally-controlled multi-event video generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, 2025. 2, 3
- [28] Qianxun Xu, Chenxi Song, Yujun Cai, and Chi Zhang. Switchcraft: Training-free multi-event video generation with

- attention controls. arXiv preprint arXiv:2602.23956, 2026. 2, 3
- [29] Shuai Yang, Wei Huang, Ruihang Chu, Yicheng Xiao, Yuyang Zhao, Xianbang Wang, Muyang Li, Enze Xie, Yingcong Chen, Yao Lu, et al. Longlive: Real-time interactive long video generation. arXiv preprint arXiv:2509.22622,

2025. 2

- [30] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024. 2
- [31] Hongyu Zhang, Yufan Deng, Zilin Pan, Peng-Tao Jiang, Bo Li, Qibin Hou, Zhiyang Dou, Zhen Dong, and Daquan Zhou. TS-attn: Temporal-wise separable attention for multi-event video generation. In The Fourteenth International Conference on Learning Representations, 2026. 2, 3
- [32] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF international conference on computer vision, 2023. 2
- [33] Yong Zhong, Zhuoyi Yang, Jiayan Teng, Xiaotao Gu, and Chongxuan Li. Concat-id: Towards universal identity-preserving video synthesis. arXiv preprint arXiv:2503.14151, 2025. 2
- [34] Yupeng Zhou, Daquan Zhou, Ming-Ming Cheng, Jiashi Feng, and Qibin Hou. Storydiffusion: Consistent selfattention for long-range image and video generation. Advances in Neural Information Processing Systems, 2024. 2

