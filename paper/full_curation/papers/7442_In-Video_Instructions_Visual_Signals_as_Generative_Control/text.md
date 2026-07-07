## In-Video Instructions: Visual Signals as Generative Control

### Gongfan Fang Xinyin Ma Xinchao Wang† National University of Singapore

{gongfan,maxinyin}@u.nus.edu, xinchao@nus.edu.sg †Corresponding author Project Page: https://fangggf.github.io/In-Video/

[Figure 1]

# arXiv:2511.19401v1[cs.CV]24Nov2025

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

Input

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

Input

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

Input

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

Input

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

Input

Figure 1. Videos generated with the proposed In-Video Instructions. The textual prompt is fixed as “Follow the instructions step by step,” while the model synthesizes content purely from the embedded visual signals within the input frames. Zoom in for more details.

### Abstract

we term In-Video Instruction. In contrast to prompt-based control, which provides textual descriptions that are inherently global and coarse, In-Video Instruction encodes user guidance directly into the visual domain through elements such as overlaid text, arrows, or trajectories. This enables explicit, spatial-aware, and unambiguous correspondences between visual subjects and their intended actions by assigning distinct instructions to different objects. Extensive experiments on three state-of-the-art generators, including

Large-scale video generative models have recently demonstrated strong visual capabilities, enabling the prediction of future frames that adhere to the logical and physical cues in the current observation. In this work, we investigate whether such capabilities can be harnessed for controllable image-to-video generation by interpreting visual signals embedded within the frames as instructions, a paradigm

Veo 3.1, Kling 2.5, and Wan 2.2, show that video models can reliably interpret and execute such visually embedded instructions, particularly in complex multi-object scenarios.

### 1. Introduction

Large-scale video generative models have recently demonstrated remarkable capabilities in visual understanding, reasoning, and physical-world simulation [9, 17, 24, 34, 36, 39]. These abilities enable models to synthesize temporally coherent and logically consistent video content conditioned on the contextual information present in the current frame. A growing body of recent work highlights this potential across diverse domains, including visual perception [36], manipulation [18], puzzle solving [20], and mathematical reasoning [29, 36].

Such visual ability naturally raises an intriguing question: if a video generative model can interpret visual signals to predict future dynamics, can those same signals also act as an internal control mechanism for video generation in a zero-shot manner? Compared to conventional textual prompts, which provide only coarse descriptions of the intended content, we examine a setting for image-to-video generation in which human guidance is embedded directly into the first video frame. The guidance is conveyed through visual elements such as overlaid text, arrows, trajectories, or other simple markers as shown in Figure 1. This formulation introduces an additional spatial dimension of control, allowing instructions to be placed near the target objects or regions, and enabling arrows to specify the intended direction or area of influence. These visual instructions, embedded as part of the video itself, provide fine-grained and unambiguous guidance. By interpreting these visual signals, the model is expected to produce the desired behaviors, including plausible object motion, coherent interactions, and precise localization.

Building on these motivations, we introduce In-Video Instruction, a paradigm that encodes user intent directly within the visual input and enables video generative models to interpret this intent as part of the scene semantics. Our method adopts an extremely simple design composed of two basic elements: short textual commands and arrows. The textual commands describe the intended behavior of an object, such as motion or interaction, while the arrows serve as spatial indicators that localize the target or specify the interaction direction. This formulation enables zeroshot and flexible controllability without any retraining. A key advantage of the paradigm is its natural compatibility with complex scenarios, including scenes containing multiple objects and tasks requiring multi-step actions. With instructions grounded in the visual space, different objects can be guided independently, and their behaviors can be spec-

ified through multiple sequential or independent instructions. These properties make In-Video Instruction an expressive interface for controllable video generation.

Accordingly, we validate this paradigm across several state-of-the-art video generative models, including both proprietary models such as Veo 3.1 [27], Kling 2.5 [28], and open-source models like Wan 2.2 [30]. Our experiments examine a wide range of capabilities, evaluating whether models can (1) comprehend and execute text embedded within visual inputs, (2) accurately localize and associate instructions with specific subjects, (3) generate fine-grained object and camera motions, and (4) follow multiple sequential or independent instructions in complex scenes. The results show that In-Video Instructions offer a clear advantage in tasks that rely on spatial grounding. Models can more reliably bind instructions to the correct subjects and resolve object-specific behaviors, especially in multi-object or cluttered scenes.

In summary, In-Video Instruction provides a direct and flexible interface for expressing user intent within the visual domain. By embedding guidance into the input itself, the paradigm allows video models to interpret instructions through the same mechanisms used for perception, enabling precise, interpretable, and spatially aligned control.

### 2. Related Works

Video Models as Zero-shot Reasoner. Large-scale video generative models [1, 15, 23, 25–28, 30] have recently demonstrated impressive capabilities in understanding and reasoning, enabling them to perform perception, physical modeling, manipulation, and reasoning tasks through video generation [3, 14, 29, 36]. At the core of these abilities lies the understanding of the current frame’s content and the generation of subsequent frames that follow coherent physical and semantic rules. Recent studies have further investigated these emerging capabilities, examining their generalization to visual puzzle-solving and mathematical reasoning [29], physical manipulation [18], autonomous driving [31], and domain-specific knowledge in medical applications [16]. Notably, large-scale video generative models have demonstrated the ability to understand textual and symbolic information embedded within videos [29, 36]. Building upon this capability, this work explores whether such understanding can be leveraged to control video generation, and enable the model to follow instructions embedded directly within the video itself, rather than relying on external textual prompts.

Controllable Video Generation. Recent advancements in video generative models have spurred increasing interest in controllable video generation, which seeks to synthesize videos that accurately reflect user intent [21, 22, 35]. Early approaches predominantly relied on text-to-video

models [6, 12, 42], where high-dimensional visual content is generated from low-dimensional textual descriptions. However, such text-only conditioning often fails to convey complex spatiotemporal semantics. To overcome this limitation, recent studies have incorporated a variety of nontextual modalities, including initial frames for image-tovideo generation [43], depth maps [8], canny edges [44], bounding boxes [32], trajectories [4], 3D condition [5], sketch [33] and motions [2, 7, 19, 37, 40, 41], thereby enabling fine-grained and multimodal control over the generation process. Moving beyond single-condition control, emerging frameworks aim for multi-condition controllable generation [13, 35, 38], jointly leveraging visual, spatial, and temporal information to enhance compositional reasoning and creative flexibility.

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

Embedded

Text Prompt: Follow the instructions step by step.

Prompt

[Figure 31]

Canvas (the generated video)

|[Figure 32]<br><br>[Figure 33]<br><br>Instruction B|
|---|

Independent Instructions

[Figure 34]

|[Figure 35]<br><br>[Figure 36]<br><br>Instruction A|
|---|

###### Target Video(s)

Sequential Instructions

[Figure 37]

[Figure 38]

[Figure 39]

|[Figure 40]<br><br>[Figure 41]<br><br>1. Instruction #1| |
|---|---|
| |Execute|

Locating

|[Figure 42]<br><br>[Figure 43]<br><br>2. Instruction #2| |
|---|---|
| | |

|[Figure 44]<br><br>[Figure 45]<br><br>3. Instruction #3|
|---|

Figure 2. In-Video Instruction controls generation by placing the instruction directly on the first frame, providing explicit spatial grounding for the instruction’s scope. This enables assigning independent, less ambiguous, and even multi-step sequential commands to different targets. During generation, we fix the textual prompt to “follow the instructions step by step” and rely solely on in-frame visual signals for control.

### 3. In-Video Instructions

This section introduces In-Video Instruction, a controllable video generation paradigm that embeds human intent directly into the visual input, as shown in Figure 2. In contrast to conventional prompt–based conditioning, which requires the model to infer object identity and spatial relations from language alone, In-Video Instruction establishes explicit correspondences between visual subjects and their associated commands by placing the instructions inside the frame. Each frame functions as an interactive canvas where guidance is overlaid as simple visual elements. The pretrained video generative model jointly interprets these elements together with the underlying scene, enabling finegrained and spatial-aware control over motion and interaction without any retraining or architectural modification.

to each command, for instance using numbered labels such as “1. Instruction #1” and “2. Instruction #2.”

#### 3.2. Generation Procedure

Given an annotated frame, the generation process follows the standard inference pipeline of a pretrained video generative model in the image-to-video setting. The annotated frame is supplied as the initial conditioning frame, and a single global text prompt is used to reinforce adherence to the visual instructions:

Fixed Text Prompt: Follow the instructions step by step.

#### 3.1. Embedding Instructions into Video Frames

The text prompt is optional and can be removed or embedded directly within the canvas for unified control (See Figure 7). No finetuning or architectural modification is applied. During inference, the model interprets the overlaid text and arrows as integral components of the input scene and implicitly treats them as actionable signals. The instructions appear only in the first frame, while subsequent frames are synthesized freely by the model, which propagates the intended motion, pose change, or interaction over time. In practice, this simple protocol is sufficient to induce a broad range of controllable behaviors, including localized object motion, camera movement, and multi-step or multi-object actions, as demonstrated in our experiments.

The construction of In-Video Instructions begins with an initial video frame containing one or more objects of interest. Human guidance is embedded into this frame through minimal visual annotations. In this work, we instantiate the instruction space using two basic primitives:

- • Short textual commands, which specify the intended behavior of a subject;
- • Arrows, which serve as spatial indicators that localize the target and may also convey the direction or region of influence.

As illustrated in Figure 2, multiple instructions may coexist within the same frame, forming either one-to-one or one-to-many correspondences between subjects and commands. Instruction placement is flexible: text can be positioned within the target region to establish a direct association, or placed externally to preserve a clear video scene. Directional elements such as arrows can be added to indicate the relevant subject or region and further strengthen the spatial linkage. In addition, the method naturally supports multi-step instructions by assigning explicit ordering

### 4. Experimental Results

In this section, we empirically investigate the performance of In-Video Instruction across diverse and complex scenarios, providing both quantitative and qualitative analyses. Our experiments are conducted on both commercial models, including Veo-3.1 [27] and Kling-2.5 [28], as well as

In-Video Instructions:

|Input|[Figure 46]|
|---|---|
| | |

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

A calm seaside scene with several cats resting on the pavement near the water. Focus on the fifth cat from the left - the one sitting on the rocks by the edge. The cat suddenly jumps up gracefully, leaps off the rocks, and exits the frame to the right. Gentle camera motion, natural lighting, and smooth animation.

Prompt:

|Input|[Figure 51]|
|---|---|
| | |

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

In-Video Instructions:

|Input|[Figure 56]|
|---|---|
| | |

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

A winter mountain scene with a team of sled dogs standing on a snowy trail, their sled visible behind them.Focus on the third dog from the camera - it slowly lowers its front legs and lies down on the snow, relaxing in a natural and calm manner. Keep the environment steady, with soft lighting and a gentle breeze moving light snow in the background.

Prompt:

|[Figure 61]<br><br>Input| |
|---|---|
| | |

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

Figure 3. Spatial Localization Ability of In-Video Instructions. We use In-Video Instructions to localize a target object among multiple entities and execute the corresponding action. For the prompt-based baseline, we rely on ChatGPT-generated textual descriptions such as “the N-th object from the left” for locating. As shown, In-Video Instructions enable precise and unambiguous localization, whereas text-only prompts exhibit noticeable limitations in resolving object positions.

open-source models such as WAN-2.2 [30]. Unless stated otherwise, videos are generated with Veo-3.1 by default. We first focus on two fundamental and essential capabilities of In-Video Instruction: (1) the ability to comprehend textual commands embedded within the image and execute the corresponding actions, and (2) the capability to locate and interact with specific visual subjects, thereby enabling fine-grained and less ambiguous generation control. Beyond these basic abilities, we further explore In-Video Instruction’s potential for controlling object and camera motion, as well as its effectiveness in multi-object and complex interaction scenarios.

#### 4.1. Text Understanding and Locating

The ability to understand text instructions. Text input is one of the most common forms of control in video generation, as it effectively specifies high-level objectives such as desired content, object motion, and scene transitions. We first demonstrate that multimodal inputs in image-to-video models can obtain generative signals not only from textual prompts, but also from text embedded within the visual input itself. To validate this, we evaluate Veo-3.1, Kling-2.5, and Wan-2.2 on the VBench benchmark [10, 11] as shown in Table 1. Each input in this task consists of a textual

prompt paired with an initial frame. We compare the performance of In-Video Instruction with that of conventional text prompts. For the In-Video Instruction setting, we embed the textual command as a caption above the image and feed the combined image into the model, and fix the input prompt as “Follow the instructions step by step.” Some examples are shown in Figure 5. In the VBench evaluation, we observe that In-Video Instruction performs slightly below but remains close to the performance of direct text inputs, demonstrating the model’s basic ability to interpret text embedded in images. The mild performance gap is expected, as interpreting text from visual content is naturally more challenging than processing explicitly provided textual prompts.

Spatial Locating and Interaction. Compared with purely understanding text, the unique strength of In-Video Instruction lies in its capability for spatial localization and interaction. While conventional text prompts effectively convey global semantics, they often lack fine-grained control over local regions, making it difficult to direct individual object behaviors in multi-object scenes. In-Video Instruction overcomes this limitation by allowing spatially placed textual commands and visual markers (e.g., arrows)

Veo3.1 Fast (16:9, 720P) Kling-2.5 (1:1, 720P) Wan2.2-A13B (1:1, 480P) In-Video Inst. Text Prompt In-Video Inst. Text Prompt In-Video Inst. Text Prompt

Dimensions

Subject 0.9710 0.9842 0.9824 0.9933 0.9823 0.9861 Dynamic Degree 0.8392 0.7857 0.5625 0.4218 0.5859 0.4921 Motion Smoothness 0.9911 0.9907 0.9905 0.9911 0.9791 0.9799 Aesthetic Quality 0.5943 0.6006 0.6553 0.6644 0.6173 0.6336 Imaging Quality 0.7074 0.7086 0.7440 0.7507 0.7229 0.7221 Temporal Flickering 0.9710 0.9719 0.9664 0.9793 0.9633 0.9632

Table 1. VBench evaluation of videos generated with in-video instructions and traditional text prompts. For in-video instructions, the prompt text is directly embedded at the top of each frame. Results show that understanding and following in-video instructions remains more challenging than responding to text prompts. Due to resolution constraints, we use a 16:9 ratio for Veo3.1 and 1:1 for other models.

(a) Translation

|Input|[Figure 66]|[Figure 67]|[Figure 68]|[Figure 69]|[Figure 70]|
|---|---|---|---|---|---|
| | | | | | |

(b) Rotation

|Input|[Figure 71]|[Figure 72]|[Figure 73]|[Figure 74]|[Figure 75]|
|---|---|---|---|---|---|
| | | | | | |

(c) Trajectory

|Input|[Figure 76]|[Figure 77]|[Figure 78]|[Figure 79]|[Figure 80]|
|---|---|---|---|---|---|
| | | | | | |

(d) Pose

|[Figure 81]<br><br>Input| |[Figure 82]|[Figure 83]|[Figure 84]|[Figure 85]|
|---|---|---|---|---|---|
| | | | | | |

Figure 4. Controlling object motions or trajectories with in-video instructions.

to specify distinct actions for different objects, enabling precise and interpretable control within complex visual environments and dynamic interactions. To validate this ability, we qualitatively examine its instruction localization in multi-object scenarios. As shown in Figure 3, we compare results from traditional text prompts and our approach, using ChatGPT to automatically generate unbiased spatial expressions such as “the N-th object from the left” to avoid human-crafted phrasing biases. Results show that In-Video Instruction achieves accurate object–instruction alignment and controllable generation in multi-object scenes, where

text-only prompts often struggle due to spatial ambiguity. This demonstrates that visual grounding provides a powerful mechanism for assigning explicit behavioral directives to different entities. Moreover, it highlights the potential of In-Video Instruction as a flexible and expressive interface for interpretable and fine-grained video control.

#### 4.2. Motion Control

Building upon the model’s fundamental abilities in text understanding and spatial interaction, we further examine motion control, a central aspect of controllable video genera-

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

|Orig.|[Figure 92]|
|---|---|
| | |

|Orig.|[Figure 93]|
|---|---|
| | |

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

- Figure 5. Controlling camera motion with In-Video Instructions. We visualize the initial frame and the final output for seven cameramotion types: static, pan left, pan right, tilt down, tilt up, zoom in, and zoom out.

the model can effectively interpret localized visual and textual information as actionable control signals, achieving fine-grained pose manipulation.

tion that governs both object and camera dynamics. As illustrated in Figure 4, we categorize four representative motion types: translation, rotation, trajectory, and pose, each defined by visual signals such as arrows, curves, or short textual annotations embedded in the first frame. These cues provide direct and interpretable spatial conditioning and enable precise, fine-grained motion guidance.

Camera Motion. Beyond object motion, another key aspect of video dynamics lies in camera movement. To examine whether text embedded within an image can control camera motion, we adopt the same evaluation strategy as in Table 1, embedding camera-related commands as captions above the image input. This setup allows us to assess the capability of In-Video Instruction in controlling camera movement. Figure 5 illustrates seven distinct camera motions: static, pan left, pan right, tilt down, tilt up, zoom in, and zoom out. As shown, textual cues within the image can effectively guide camera behavior.

Translation. The key challenge in motion control primarily lies in direction specification, which is often coarse and imprecise in text-based descriptions. In-Video Instruction addresses this limitation by anchoring visual arrows directly to the target object, enabling the model to infer both the motion vector and its magnitude. As illustrated by the hotair balloon example in Figure 4(a), the object’s movement aligns precisely with the intended initial direction.

Rotation. Rotation is comparatively easier to express, yet still benefits from explicit visual grounding. Curved arrows intuitively convey rotational direction and pivot centers, allowing the model to perform controlled rotation. In the elephant example in Figure 4(b), different rotation instructions are assigned to different objects, resulting in independent yet coordinated rotations.

Trajectory. Trajectory control represents a more complex form of motion, requiring the model to follow multi-stage or curved paths. While describing such motion textually (e.g., “fly upward, then turn left, and stop”) is cumbersome and ambiguous, In-Video Instruction allows users to directly sketch trajectories as continuous curves. The model accurately follows these drawn paths, maintaining realistic dynamics and temporal consistency throughout the sequence.

Pose. In-Video Instruction enables coherent and smooth pose adjustments, allowing the model to generate natural and consistent variations in posture. This demonstrates that

When to Use In-Video Instructions. In summary, InVideo Instruction is capable of controlling both object motion and camera motion. Among them, object motion control is inherently more localized and, in complex multiobject scenes, highly dependent on accurate spatial localization. In contrast, camera motion represents a more global form of control that affects the entire scene. From this perspective, In-Video Instruction is particularly well-suited for fine-grained and spatially grounded manipulation. However, we note that this capability is not essential for camera motion, as it represents a more global form of control that can be well handled by conventional text prompts.

#### 4.3. Multiple Objects and Instructions

We further evaluate the scalability of In-Video Instruction by examining its performance across configurations involving multiple instructions, objects, and their combinations. Specifically, we study four representative scenarios: (1) single instruction, single object, where one subject responds to a single command; (2) single instruction, multiple objects, where several subjects execute the same action con-

(b) Multiple Objects, Single Instruction (d) Multiple Objects, Multiple Instruction

(a) Single Objects, Single Instruction

(c) Single Object, Multiple Instructions

|[Figure 102]<br><br>Input| |
|---|---|
| | |

|[Figure 103]<br><br>Input| |
|---|---|
| | |

|[Figure 104]<br><br>Input| |
|---|---|
| | |

|[Figure 105]<br><br>Input| |
|---|---|
| | |

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

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

- Figure 6. In-Video Instructions with Multiple Objects and Commands, enabling both sequential instructions that involve a series of actions and parallel instructions that manipulate different objects independently.

currently; (3) multiple instructions, single object, where a single subject performs a sequence of ordered actions; and (4) multiple instructions, multiple objects, where multiple entities each follow distinct and independent instructions. These settings allow us to systematically analyze the performance of In-Video Instructions across different scenes and compare their advantages over conventional text prompts.

Single Object, Single Instruction. This serves as the simplest and most direct control setting, validating the model’s ability to ground a single instruction. As shown in Figure 6(a), when prompted to “turn around,” the panda executes a smooth and coherent rotation, demonstrating that both text-prompt and in-video instruction settings achieve comparable performance. This suggests that for simple,

globally interpretable tasks, textual and visual instructions are equally effective.

Multiple Objects, Single Instruction. When a shared instruction applies to multiple entities, the ability to specify spatial correspondence becomes more relevant. As illustrated in Figure 6(b), two birds respond to the “fly away” instruction while the third remains stationary. The spatial anchoring of the instruction helps the model associate the action with the intended subjects, ensuring consistent yet selective control. This demonstrates that visual grounding offers a practical mechanism for managing concurrent behaviors among multiple entities within the same scene.

Single Object, Multiple Instructions. The next scenario requires sequential reasoning, where one subject performs a series of temporally dependent actions. As illustrated in Figure 6(c), the seal follows three ordered commands: jump into the water, swim to the shore, and move here, forming a continuous motion with correct temporal logic and spatial alignment. In-Video Instructions encode stepwise relations directly within the visual domain, where the spatial ordering and numbering of cues provide implicit temporal structure. Compared to text prompts, In-Video Instructions make it easier for the model to handle interactions between objects and their environment, combining spatial localization and trajectory reasoning to construct complex and controllable video generations.

Multiple Objects, Multiple Instructions. The most complex scenario involves issuing distinct instructions to multiple entities within a single frame. In Figure 6(d), three cars perform different actions such as backing up, turning right, and stopping, while preserving coherent scene dynamics. Our result demonstrates that the model can interpret spatially separated visual signals as independent control signals, enabling localized manipulation without mutual interference. In contrast to text prompts, which express only global intent with limited positional specificity, InVideo Instructions offer flexible, target-aware control and produce precise, disentangled behaviors.

The success rate of multiple instructions. To further evaluate the effectiveness, we conducted a human assessment comparing In-Video Instructions with conventional text-based prompting on the “Multiple Objects, Multiple Instructions” setting in Figure 6(d), generating 24 videos for each method and evaluating them through human judgment. As shown in Table 2, the model consistently follows the embedded visual instructions in this complex setting, achieving higher success rates than text-only prompts across diverse motion patterns. In addition, we observed an interesting phenomenon: instructing the white car in Figure 6(d) to back up is relatively difficult, which leads to a success rate of 20.8% with the in-video instruction and 8.3% with the text prompt. This appears to stem from the presence of another vehicle directly behind it, which induces a strong prior for the model to move the car forward rather than backward, particularly when resolving physically plausible trajectories.

Synthesizing Videos by Manipulating Multiple Frames. Beyond single-frame control, we show that advanced video models can synthesize complex scenes by integrating information from multiple source frames. As shown in Figure 7, the model combines spatial and temporal cues from different visual inputs to produce coherent and continuous video

In-Video Inst. Prompt

- Instruction A (Back up) 20.8% 8.3%

- Instruction B (Turn right) 58.3% 29.2%

- Instruction C (Stop) 95.8% 58.3%

Table 2. Success rates of instructions under the “multiple objects, multiple instructions” setting in Figure 6(d), averaged over 24 generated videos based on human evaluation.

|[Figure 126]|[Figure 127]|
|---|---|

Input Input

|[Figure 128]|
|---|

|[Figure 129]|
|---|

|[Figure 130]|
|---|

|[Figure 131]|
|---|

|[Figure 132]|
|---|

|[Figure 133]|
|---|

Figure 7. Synthesizing videos by manipulating multiple seed frames. We generate videos from several initial frames and use visual instructions to coordinate interactions across them; all videos in this setting are produced using Kling-2.5.

sequences. This demonstrates the ability to interpret crossframe instructions and maintain consistency across distinct visual contexts.

### 5. Limitations

While In-Video Instruction offers a simple and intuitive way to guide generation, several limitations remain. Since the instructions are drawn directly on the image, they persist in the generated video and often require post-processing for removal. We also observe that these visual markers may become occluded during synthesis, suggesting that the model already possesses priors for suppressing such elements. Extending the text prompt to explicitly remove visible annotations may therefore further improve the results. In addition, our analysis remains largely qualitative, underscoring the need for more systematic assessment in future work. Finally, all instructions examined in this study are manually constructed, whereas real-world videos contain inherent vi-

sual signals such as traffic lights or signboards; understanding whether models can interpret and react to these natural signals remains an interesting direction for future research.

### 6. Conclusion

This work introduces In-Video Instruction, a simple and training-free approach that embeds human intent directly into the visual input for controllable video generation. The method enables fine-grained, spatially grounded control across diverse tasks, and experiments on multiple models demonstrate that these visual signals can be reliably interpreted as actionable guidance, offering strong flexibility and controllability in complex scenes.

### References

- [1] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, et al. Video generation models as world simulators. OpenAI Blog, 1(8):1, 2024. 2
- [2] Ryan Burgert, Yuancheng Xu, Wenqi Xian, Oliver Pilarski, Pascal Clausen, Mingming He, Li Ma, Yitong Deng, Lingxiao Li, Mohsen Mousavi, et al. Go-with-the-flow: Motioncontrollable video diffusion models using real-time warped noise. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 13–23, 2025. 3
- [3] Zhengcong Fei, Di Qiu, Debang Li, Changqian Yu, and Mingyuan Fan. Video diffusion transformers are in-context learners. arXiv preprint arXiv:2412.10783, 2024. 2
- [4] Daniel Geng, Charles Herrmann, Junhwa Hur, Forrester Cole, Serena Zhang, Tobias Pfaff, Tatiana Lopez-Guevara, Carl Doersch, Yusuf Aytar, Michael Rubinstein, Chen Sun, Oliver Wang, Andrew Owens, and Deqing Sun. Motion prompting: Controlling video generation with motion trajectories. arXiv preprint arXiv:2412.02700, 2024. 3
- [5] Zekai Gu, Rui Yan, Jiahao Lu, Peng Li, Zhiyang Dou, Chenyang Si, Zhen Dong, Qifeng Liu, Cheng Lin, Ziwei Liu, Wenping Wang, and Yuan Liu. Diffusion as shader: 3daware video diffusion for versatile video generation control. arXiv preprint arXiv:2501.03847, 2025. 3
- [6] Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. arXiv preprint arXiv:2205.15868, 2022. 3
- [7] Chen Hou and Zhibo Chen. Training-free camera control for video generation. arXiv preprint arXiv:2406.10126, 2024. 3
- [8] Wenbo Hu, Xiangjun Gao, Xiaoyu Li, Sijie Zhao, Xiaodong Cun, Yong Zhang, Long Quan, and Ying Shan. Depthcrafter: Generating consistent long depth sequences for open-world videos. In CVPR, 2025. 3
- [9] Siqiao Huang, Jialong Wu, Qixing Zhou, Shangchen Miao, and Mingsheng Long. Vid2world: Crafting video diffusion models to interactive world models. arXiv preprint arXiv:2505.14357, 2025. 2
- [10] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin,

- Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21807–21818, 2024. 4
- [11] Ziqi Huang, Fan Zhang, Xiaojie Xu, Yinan He, Jiashuo Yu, Ziyue Dong, Qianli Ma, Nattapol Chanpaisit, Chenyang Si, Yuming Jiang, et al. Vbench++: Comprehensive and versatile benchmark suite for video generative models. arXiv preprint arXiv:2411.13503, 2024. 4
- [12] Yuming Jiang, Tianxing Wu, Shuai Yang, Chenyang Si, Dahua Lin, Yu Qiao, Chen Change Loy, and Ziwei Liu. Videobooth: Diffusion-based video generation with image prompts. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6689– 6700, 2024. 3
- [13] Zeyinzi Jiang, Zhen Han, Chaojie Mao, Jingfeng Zhang, Yulin Pan, and Yu Liu. Vace: All-in-one video creation and editing. arXiv preprint arXiv:2503.07598, 2025. 3
- [14] Bingyi Kang, Yang Yue, Rui Lu, Zhijie Lin, Yang Zhao, Kaixin Wang, Gao Huang, and Jiashi Feng. How far is video generation from world model: A physical law perspective. arXiv preprint arXiv:2411.02385, 2024. 2
- [15] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024. 2
- [16] Yuxiang Lai, Jike Zhong, Ming Li, Yuheng Li, and Xiaofeng Yang. Are video models emerging as zero-shot learners and reasoners in medical imaging? arXiv preprint arXiv:2510.10254, 2025. 2
- [17] Dacheng Li, Yunhao Fang, Yukang Chen, Shuo Yang, Shiyi Cao, Justin Wong, Michael Luo, Xiaolong Wang, Hongxu Yin, Joseph E Gonzalez, et al. Worldmodelbench: Judging video generation models as world models. arXiv preprint arXiv:2502.20694, 2025. 2
- [18] Hongyu Li, Lingfeng Sun, Yafei Hu, Duy Ta, Jennifer Barry, George Konidaris, and Jiahui Fu. Novaflow: Zero-shot manipulation via actionable flow from generated videos. arXiv preprint arXiv:2510.08568, 2025. 2
- [19] Quanhao Li, Zhen Xing, Rui Wang, Hui Zhang, Qi Dai, and Zuxuan Wu. Magicmotion: Controllable video generation with dense-to-sparse trajectory guidance. arXiv preprint arXiv:2503.16421, 2025. 3
- [20] Jinyang Liu, Wondmgezahu Teshome, Sandesh Ghimire, Mario Sznaier, and Octavia Camps. Solving masked jigsaw puzzles with diffusion vision transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 23009–23018, 2024. 2
- [21] Yue Ma, Kunyu Feng, Zhongyuan Hu, Xinyu Wang, Yucheng Wang, Mingzhe Zheng, Xuanhua He, Chenyang Zhu, Hongyu Liu, Yingqing He, et al. Controllable video generation: A survey. arXiv preprint arXiv:2507.16869,

2025. 2

- [22] Bohao Peng, Jian Wang, Yuechen Zhang, Wenbo Li, MingChang Yang, and Jiaya Jia. Controlnext: Powerful and efficient control for image and video generation. arXiv preprint arXiv:2408.06070, 2024. 2

- [23] Adam Polyak, Amit Zohar, Andrew Brown, Andros Tjandra, Animesh Sinha, Ann Lee, Apoorv Vyas, Bowen Shi, ChihYao Ma, Ching-Yao Chuang, et al. Movie gen: A cast of media foundation models. arXiv preprint arXiv:2410.13720,

2024. 2

- [24] Zhongwei Ren, Yunchao Wei, Xun Guo, Yao Zhao, Bingyi Kang, Jiashi Feng, and Xiaojie Jin. Videoworld: Exploring knowledge learning from unlabeled videos. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 29029–29039, 2025. 2
- [25] Team Seawead, Ceyuan Yang, Zhijie Lin, Yang Zhao, Shanchuan Lin, Zhibei Ma, Haoyuan Guo, Hao Chen, Lu Qi, Sen Wang, et al. Seaweed-7b: Cost-effective training of video generation foundation model. arXiv preprint arXiv:2504.08685, 2025. 2
- [26] Genmo Team. Mochi 1. https://github.com/ genmoai/models, 2024.
- [27] Veo Team. Veo: A text-to-video generation system. Technical report, Google Deepmind, 2025. 2, 3
- [28] Kuaishou Technology. Kling ai: Text-to-video and imageto-video generation model. https://klingai.com/ global/, 2024. Accessed: 13 November 2025. 2, 3
- [29] Jingqi Tong, Yurong Mou, Hangcheng Li, Mingzhe Li, Yongzhuo Yang, Ming Zhang, Qiguang Chen, Tianyi Liang, Xiaomeng Hu, Yining Zheng, et al. Thinking with video: Video generation as a promising multimodal reasoning paradigm. arXiv preprint arXiv:2511.04570, 2025. 2
- [30] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 2, 4
- [31] Haiguang Wang, Daqi Liu, Hongwei Xie, Haisong Liu, Enhui Ma, Kaicheng Yu, Limin Wang, and Bing Wang. Mila: Multi-view intensive-fidelity long-term video generation world model for autonomous driving. arXiv preprint arXiv:2503.15875, 2025. 2
- [32] Jiawei Wang, Yuchen Zhang, Jiaxin Zou, Yan Zeng, Guoqiang Wei, Liping Yuan, and Hang Li. Boximator: Generating rich and controllable motions for video synthesis. arXiv preprint arXiv:2402.01566, 2024. 3
- [33] Xiang Wang, Hangjie Yuan, Shiwei Zhang, Dayou Chen, Jiuniu Wang, Yingya Zhang, Yujun Shen, Deli Zhao, and Jingren Zhou. Videocomposer: Compositional video synthesis with motion controllability. Advances in Neural Information Processing Systems, 36:7594–7611, 2023. 3
- [34] Xiaofeng Wang, Zheng Zhu, Guan Huang, Boyuan Wang, Xinze Chen, and Jiwen Lu. Worlddreamer: Towards general world models for video generation via predicting masked tokens. arXiv preprint arXiv:2401.09985, 2024. 2
- [35] Xiang Wang, Shiwei Zhang, Longxiang Tang, Yingya Zhang, Changxin Gao, Yuehuan Wang, and Nong Sang. Unianimate-dit: Human image animation with large-scale video diffusion transformer. arXiv preprint arXiv:2504.11289, 2025. 2, 3
- [36] Thadd¨aus Wiedemer, Yuxuan Li, Paul Vicol, Shixiang Shane Gu, Nick Matarese, Kevin Swersky, Been Kim, Priyank

- Jaini, and Robert Geirhos. Video models are zero-shot learners and reasoners. arXiv preprint arXiv:2509.20328, 2025. 2
- [37] Shengqiong Wu, Weicai Ye, Jiahao Wang, Quande Liu, Xintao Wang, Pengfei Wan, Di Zhang, Kun Gai, Shuicheng Yan, Hao Fei, et al. Any2caption: Interpreting any condition to caption for controllable video generation. arXiv preprint arXiv:2503.24379, 2025. 3
- [38] Dianbing Xi, Jiepeng Wang, Yuanzhi Liang, Xi Qiu, Yuchi Huo, Rui Wang, Chi Zhang, and Xuelong Li. Omnivdiff: Omni controllable video diffusion for generation and understanding. arXiv preprint arXiv:2504.10825, 2025. 3
- [39] Jiannan Xiang, Guangyi Liu, Yi Gu, Qiyue Gao, Yuting Ning, Yuheng Zha, Zeyu Feng, Tianhua Tao, Shibo Hao, Yemin Shi, et al. Pandora: Towards general world model with natural language actions and video states. arXiv preprint arXiv:2406.09455, 2024. 2
- [40] Zeqi Xiao, Wenqi Ouyang, Yifan Zhou, Shuai Yang, Lei Yang, Jianlou Si, and Xingang Pan. Trajectory attention for fine-grained video motion control. arXiv preprint arXiv:2411.19324, 2024. 3
- [41] Zeqi Xiao, Yifan Zhou, Shuai Yang, and Xingang Pan. Video diffusion models are training-free motion interpreter and controller. Advances in Neural Information Processing Systems, 37:76115–76138, 2024. 3
- [42] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024. 3
- [43] Zuhao Yang, Jiahui Zhang, Yingchen Yu, Shijian Lu, and Song Bai. Versatile transition generation with image-tovideo diffusion. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 16981–16990,

2025. 3

- [44] Y Zhang, Y Wei, D Jiang, X Zhang, W Zuo, and Q Tian. Controlvideo: Training-free controllable text-to-video generation. arxiv 2023. arXiv preprint arXiv:2305.13077. 3

