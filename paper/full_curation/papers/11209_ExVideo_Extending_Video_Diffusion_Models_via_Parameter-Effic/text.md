# arXiv:2406.14130v1[cs.CV]20Jun2024

## ExVideo: Extending Video Diffusion Models via Parameter-Efficient Post-Tuning

Zhongjie Duan

East China Normal University Shanghai, China zjduan@stu.ecnu.edu.cn

Wenmeng Zhou

Alibaba Group Hangzhou, China wenmeng.zwm@alibaba-inc.com

Cen Chen

East China Normal University Shanghai, China cenchen@dase.ecnu.edu.cn

Yaliang Li

Alibaba Group Hangzhou, China yaliang.li@alibaba-inc.com

### ABSTRACT

Recently, advancements in video synthesis have attracted significant attention. Video synthesis models such as AnimateDiff and Stable Video Diffusion have demonstrated the practical applicability of diffusion models in creating dynamic visual content. The emergence of SORA has further spotlighted the potential of video generation technologies. Nonetheless, the extension of video lengths has been constrained by the limitations in computational resources. Most existing video synthesis models can only generate short video clips. In this paper, we propose a novel post-tuning methodology for video synthesis models, called ExVideo. This approach is designed to enhance the capability of current video synthesis models, allowing them to produce content over extended temporal durations while incurring lower training expenditures. In particular, we design extension strategies across common temporal model architectures respectively, including 3D convolution, temporal attention, and positional embedding. To evaluate the efficacy of our proposed post-tuning approach, we conduct extension training on the Stable Video Diffusion model. Our approach augments the model’s capacity to generate up to 5× its original number of frames, requiring only 1.5k GPU hours of training on a dataset comprising 40k videos. Importantly, the substantial increase in video length doesn’t compromise the model’s innate generalization capabilities, and the model showcases its advantages in generating videos of diverse styles and resolutions. We will release the source code and the enhanced model publicly1.

### KEYWORDS

Video Synthesis, Diffusion Models, Post-Tuning

### 1 INTRODUCTION

Over recent years, diffusion models [17, 36] have achieved outstanding results in image synthesis, significantly surpassing previous GANs [10]. These achievements have subsequently fostered a burgeoning interest in the adaptation of diffusion models for video synthesis. Models such as Stable Video Diffusion [3], AnimateDiff [15], and VideoCrafter [4] epitomize this research trajectory, showcasing the ability to produce frames that are not only coherent but also of high visual quality. These achievements underscore the

1Project page: https://ecnu-cilab.github.io/ExVideoProjectPage Github repo: https://github.com/modelscope/DiffSynth-Studio

Weining Qian

East China Normal University Shanghai, China wnqian@dase.ecnu.edu.cn

practicality and potential of employing diffusion models in the field of video synthesis. With the groundbreaking results of SORA [27] reported at the beginning of 2024, the research direction of video synthesis has once again attracted widespread attention.

Although current video synthesis models are capable of producing video clips of satisfactory quality, there remains a considerable gap in their ability to extend the duration of the videos generated. To generate longer videos, current methodologies can be categorized into three types. 1) Training with long video datasets [1, 7, 42]: Given the computational constraints, present video generation models are predominantly trained on short video clips. The use of longer videos for training would result in prohibitively escalated costs. 2) Generating videos in a streaming [22] or sliding window [11] manner. Without further training, we can process the video by dividing it into multiple segments. However, this approach leads to lower video coherence. In addition, existing video generation models lack the capability for long-term video understanding, hence the accumulation of errors becomes inevitable. As a result, during the generation of long videos, the visual quality is prone to deterioration, manifesting as a breakdown in the imagery. 3) Frame Interpolation [19, 43]: Video frame interpolation models offer a method to augment the frame count of generated videos. However, this approach is inadequate for extending the narrative timeframe of the video. While it increases the number of frames, retaining the original frame rate would give the video an unnatural slow-motion effect, failing to prolong the narrative span of the video content. These outlined challenges underscore the necessity for innovative solutions capable of overcoming the existing hurdles associated with video duration extension, without compromising video quality or coherence.

Recent breakthroughs in the development of LLMs (large language models) [8, 44, 45] have inspired us. Notably, LLMs, despite being trained on fixed-length data, exhibit remarkable proficiency in understanding contexts of variable lengths. This flexibility is further enhanced through the integration of supplementary components and the application of lightweight training procedures, enabling the processing of exceptionally lengthy texts. Such innovations have motivated us to explore analogous methodologies within video synthesis models. In this paper, we introduce a novel post-tuning strategy, called ExVideo, specifically designed to empower existing video synthesis models to produce extended-duration videos within

the constraints of limited computational resources. We have designed an extension structure for mainstream video synthesis model architectures. This framework incorporates adapter components, meticulously engineered to maintain the intrinsic generalization capabilities of the base model. Through post-tuning, we enhance the temporal modules of the model, thereby facilitating the processing of content across longer temporal spans.

In theory, ExVideo is designed to be compatible with the majority of existing video synthesis models. To empirically validate the efficacy of our post-tuning methodology, we applied it to the Stable Video Diffusion model [3], a popular open-source image-tovideo model. Through ExVideo, we can extend the original frame production capacity from a limit of 25 frames to 128 frames. Importantly, this expansion was achieved without compromising the model’s distinguished generative capabilities. Additionally, the enhanced model exhibits the versatility to be seamlessly integrated with text-to-image models [6, 26, 29]. This synergistic amalgamation establishes robust and versatile pipelines. This adaptability underscores the potential of our post-training technique, the source code and the extended model will be released publicly.

In summary, the contributions of this paper include:

- • We present ExVideo, a post-tuning technique for video synthesis models that can extend the temporal scale of existing video synthesis models to facilitate the generation of long videos.
- • We have trained an extended video generation model based on Stable Video Diffusion, capable of generating coherent videos of up to 128 frames while preserving the generative capabilities of the original model.

2 RELATED WORK

- 2.1 Diffusion Models

Diffusion models [17, 36] are a category of generative models that model the generation process as a Markov random process. Unlike GANs [14], diffusion models do not require adversarial training, hence their training process is more stable. Moreover, through an iterative generation process, diffusion models are capable of producing images of exceptionally high quality. In recent years, image synthesis models based on diffusion, including Pixart [6], Imagen [35], Hunyuan DiT [26], and the Stable Diffusion series [21, 29, 32], have achieved impressive success. Diffusion models have given rise to a vast open-source technology ecosystem. Technologies such as LoRA [18], ControlNet [47], DreamBooth [34], Textual Inversion [13], and IP-Adapter [46] have endowed the generation process of diffusion models with a high degree of controllability, thereby meeting the needs of various application scenarios.

- 2.2 Video Synthesis

Given the remarkable success of diffusion models in image synthesis, video synthesis approaches based on diffusion have also been proposed in recent years. For example, by adding motion modules to the UNet model [33] in Stable Diffusion [32], AnimateDiff [15] transfers the capabilities of image synthesis to video synthesis. Stable Video Diffusion [3] is an image-to-video model architecture and can synthesize video clips after end-to-end video synthesis training. Unlike image synthesis models, video synthesis models

require substantial computational resources since the model needs to process multiple frames simultaneously. As a result, most current video generation models can only produce very short video clips. For instance, AnimateDiff can generate up to 32 frames, while Stable Video Diffusion can generate a maximum of 25 frames. This limitation prompts us to explore how to construct video synthesis models over longer temporal scales.

### 2.3 Extending Generative Models

Although the existing diffusion models are trained with a fixed scale, such as Stable Diffusion being trained at a fixed resolution of 512 × 512, some approaches can extend them to larger scales. For instance, in image synthesis, approaches like Mixture of Diffusers [20], MultiDiffusion [2], and ScaleCrafter [16] can increase the resolution of generated images by altering the inference process of the UNet model in Stable Diffusion. Similar techniques have also emerged in large language models. With the help of positional encoding technologies such as RoPE [37] and ALiBi [30], large language models can extrapolate to longer text processing tasks under the premise of training with limited-length texts. Post-tuning can further help language models achieve super-long text comprehension and generation [8, 45]. These research findings have inspired us, prompting us to explore the extension of video synthesis models. We aim to endow existing video synthesis models with the capability to generate longer videos.

3 METHODOLOGY

In this section, we first review the architectures of mainstream video diffusion models, then discuss how we extend the temporal modules for long video synthesis, and finally introduce the posttuning strategy.

### 3.1 Preliminaries

The huge demands for computational resources in training video synthesis models lead to a prevalent practice of adapting existing image synthesis models for video generation. This adaptation is typically achieved by incorporating temporal modules into the model to enable the synthesis of dynamic content. We provide a comprehensive overview of temporal module architectures, which can be categorized as follows:

- • 3D convolution [25]: Convolution layers form the foundational blocks in computer vision. 2D convolution layers have been employed in the UNet [33] architecture, which is widely used in diffusion models. By extending 2D convolutions into the third dimension, these layers are seamlessly adapted for use in video synthesis models. Research indicates that convolution layers in diffusion models exhibit a high degree of adaptability across various resolutions [2], which is a testament to their capacity for generalization.
- • Temporal attention [39]: In image synthesis, the importance of attention mechanisms is underscored by their contribution to the generation of images with remarkable fidelity, as evidenced by the ablation studies in latent diffusion [32]. Transferring spatial attention mechanisms to the video domain raises concerns regarding computational efficiency

… …

Frame ids Video Representation

Frame ids Video Representation

Static Positional Embedding

+

Trainable Positional Embedding

+

Identity 3D Convolution

Other Modules

Other Modules

Temporal Block Extended Temporal Block

…

…

- Figure 1: The architecture of extended temporal blocks in Stable Video Diffusion. We replace the static positional embedding with trainable positional embedding and add an adaptive identity 3D convolution layer to learn long-term video features. The modifications are adaptive, preserving the original generalization abilities of the pre-trained model. All parameters outside the temporal block are fixed while training for lower memory usage.

due to the quadratic time complexity of the attention operators. To circumvent this computational bottleneck, advanced video synthesis models typically adopt additional temporal attention layers [3, 15] that optimize efficiency by curtailing the volume of embeddings processed by each attention operator.

• Positional embedding[37]: The native attention layers cannot model the positional information in videos. Therefore, video synthesis models typically incorporate positional embeddings to enrich the embedding space with positional information. Positionalembeddingscan be instantiated through diverse methodologies. For example, AnimateDiff [15] opts for learnable parameters to establish positional embeddings, whereas Stable Video Diffusion [3] utilizes trigonometric functions to generate static positional embeddings.

### 3.2 Extending Temporal Modules

Most video synthesis models are pre-trained on videos comprising only a constrained number of frames due to limited computational resources. For instance, Stable Video Diffusion is capable of generating a maximum of 25 frames, while AnimateDiff is limited to synthesizing sequences of up to 32 frames. To augment the capacity of these models to produce extended videos, we propose enhancements to the temporal modules within these models.

Firstly, the inherent functionality of 3D convolution layers to adaptively accommodate various scales has been previously validated through empirical studies [2, 16, 20], even without necessitating fine-tuning. Consequently, we opt to retain the 3D convolution layers in their original form to preserve these capabilities. Secondly, regarding the temporal attention modules, research on large language models has demonstrated the potential for scaling existing models to accommodate longer contextual sequences [8, 45]. Inspired by these findings, we fine-tune the parameters within the temporal attention layers during the training process to enhance

their efficacy over extended frame sequences. Thirdly, for the positional embedding layers, either static or trainable embeddings cannot be directly applied to longer videos. To circumvent this pitfall while ensuring compatibility with a wide array of existing video models, we use extended trainable parameters to replace the original positional embeddings. These extended trainable positional embeddings are initialized in a cyclic pattern, drawing upon the configurations of the pre-existing embeddings. Further, drawing inspiration from various adapter models [18, 47], we incorporate an additional identity 3D convolution layer subsequent to the positional embedding layer, aimed at learning long-term information. The central unit of this 3D convolution kernel is initialized as an identity matrix, and the remaining parameters are initialized to zero. The identity 3D convolution layer ensures that, before training, there is no alteration to the video representation, thereby maintaining consistency with the original computational process.

We apply our devised extending approach to Stable Video Diffusion [3], which is a popular model within open-source communities for video synthesis. The comparative architectures, both pre and post-extension, are illustrated in Figure 1. Because of the fundamental similarities that underpin the construction of temporal blocks within video synthesis models, our extending approach can also be applied to various video synthesis models.

### 3.3 Post-Tuning

After extending the temporal blocks in the video synthesis models, we enhance the model’s abilities to generate extended videos via post-tuning. To circumvent potential copyright concerns with video content, we employed a publicly available dataset OpenSoraPlan2, which comprises 40,258 videos. These videos were sourced from

2https://huggingface.co/datasets/LanguageBind/Open-Sora-Plan-v1.0.0

- (a) Realistic style

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

(c) Pixel art style

- (b) Flat anime style

- Figure 2: Examples in different styles generated by our Extended Stable Video Diffusion, where the first frame is generated by Stale Diffusion 3. The prompt is “A beautiful coastal beach in spring, waves lapping on sand”, followed by the description of style.

copyright-free platforms, including Mixkit3, Pexels4, and Pixabay5. The videos in this dataset maintain a resolution of 512 × 512, inconsistent with the original resolution supported by Stable Video Diffusion. Given the model’s design to accommodate varying resolutions, we opt to conduct the training at this resolution. ExVideo expands this capacity to 128 frames. Over such extended sequences, full training is deemed impractical because of the substantial computational requirements. Instead, we employed several engineering optimizations aimed at optimizing GPU memory usage. These optimizations are crucial for managing the increased computational load and facilitating efficient training within limited hardware resources:

- • Parameter freezing: All parameters except the temporal blocks are frozen.
- • Mixed precision training [28]: We deploy a mixed precision training program by converting a subset of the model’s parameters to 16-bit floating-point format.
- • Gradient checkpointing [12]: Gradient checkpointing is enabled in the model. By storing intermediate states during

- 3https://mixkit.co/
- 4https://www.pexels.com/
- 5https://pixabay.com/

forward passes and recomputing gradients on-demand during the backward pass, this technique effectively decreases memory usage.

- • Flash Attention [9]: We integrate Flash Attention to enhance the computational efficiency of attention mechanisms.
- • Shard optimizer states and gradients: We leverage DeepSpeed [31], a library optimized for distributed training, to enable shard optimizer states and gradients across multiple GPUs.

The loss function and the noise scheduler are consistent with the original model. The learning rate is 10−5 and the batch size on each GPU is 1. The training was conducted using only 8 NVIDIA A100 GPUs over one week. In order to ensure the stability of the training process, exponential moving averages are employed for the update of weights.

4 CASE STUDIES

- 4.1 Text-to-Video Synthesis

By integrating existing text-to-image models, we can easily develop integrated pipelines capable of converting textual descriptions into videos. In this pipeline, the outputs from a text-to-image model are utilized as the foundational frames for the subsequent image-tovideo model. Illustrative examples are shown in Figure 2, wherein the foundational frames are synthesized by Stable Diffusion 3. The prompts direct the text-to-image model to create images across various styles. Our model subsequently generates fluent motion transitions from these high-quality images, even if styles such as flat anime and pixel art are not included in the training dataset. The Extended Stable Video Diffusion model preserves and extends the generalization capabilities in the original model.

- 4.2 Visualization of Training Process

We investigated the evolution of the model’s capabilities during the training process. Figure 3 presents the generated videos that exemplify the model’s performance at three distinct phases of training. It is difficult to present the dynamics using still images, thus we present the optical flow, computed by RAFT [38], to the right of each example for a clearer demonstration of motion. Initially, before training, the extended model architecture was solely capable of guaranteeing the structural integrity of the video frames, which suffered from pronounced jittering artifacts. Progressing through the training, after 32,000 steps, the model began to produce videos displaying smooth camera movements. With continued training up to 64,000 steps, the model further advanced to create complex motions, such as clouds and mountains moving with nuanced, layered speed. The model effectively understands the depth and spatial relationships within the scene. This example intuitively illustrates the process of the model learning long-term information.

- 4.3 Adaptive Resolution

We also ascertain the performance of the extended Stable Video Diffusion across various resolutions. Several video examples are illustrated in Figure 4. These examples demonstrate that the model, with common aspect ratios, adeptly generates videos in higher resolutions. This capability not only highlights the robustness and

[Figure 10]

- (a) Before post-tuning, the camera is irregularly jittering.

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

(c) After 64000 steps, complex motion emerges.

- (b) After 32000 steps, the camera is moving smoothly.

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

- Figure 3: Video examples in different training phases. The first frame is generated by Hunyuan DiT, and the prompt is “sunset, mountains, clouds”. We present the optical flow to visualize the motion, where pixels with similar colors are moving in similar directions.

generalizability of the base model but also underscores the effectiveness of our post-tuning methodologies in enhancing model performance.

### 4.4 Comparison with Other Models

To evaluate the performance of our model, we conducted comparative analyses against several existing video synthesis models. Illustrative outcomes from these models, including AnimateDiff [15], LaVie [41], ModelScopeT2V [40], OpenSora6, T2VTurbo [24], VideoCrafter2 [5], alongside our extended Stable Video Diffusion, are displayed in Figure 5. The videos generated by the baseline models are collected from GenAI-Arena7 [23]. The text-to-image model utilized in our pipeline is Hunyuan DiT [26]. A critical observation from this comparison is that the majority of existing video synthesis models usually generate videos with minimal motion dynamics. In contrast, owing to the post-tuning processes applied over extensive temporal duration, our extended model demonstrates a superior capability to generate videos with significant movements. This differential outcome underscores the advanced generative performance of our model.

- 6https://github.com/hpcaitech/Open-Sora
- 7https://github.com/ChromAIca/VideoGenMuseum

- (a) 576×1024

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

- (b) 1024×576
- (c) 1024×1024

Figure 4: Video examples in various resolutions. The first frame is generated by Stable Diffusion 3, and the prompt is “bonfire, on the stone”.

### 5 LIMITATIONS

While ExVideo can enhance the capabilities of video diffusion models, the post-tuned version continues to be constrained by the inherent limitations of its foundational model. Notably, the extended Stable Video Diffusion struggles to accurately synthesize human portraits, leading to frequent instances of truncated frames. To develop a model capable of synthesizing high-quality long videos, it is imperative to train a robust base model. Nevertheless, due to limitations in resources, we are unable to independently pre-train a large video synthesis model. Consequently, we eagerly anticipate the release of open-source models in the future to advance our research endeavors.

### 6 CONCLUSIONS AND FUTURE WORK

In this paper, we delve into the enhancement of video diffusion models through post-tuning. Specifically, we propose a post-tuning approach called ExVideo, which can extend the duration of generated

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

(a) AnimateDiff

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

(b) LaVie

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

(c) ModelScopeT2V

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

- (d) OpenSora
- (e) T2VTurbo
- (f) VideoCrafter2

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

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

(g) Hunyuan DiT + extended Stable Video Diffusion

#### Figure 5: Visual comparisons of text-to-video results from several existing video synthesis models and our Extended model. The prompts are “a boat sailing smoothly on a calm lake” and “an astronaut flying in space, Van Gogh style”. In our pipeline, the first frame is generated by Hunyuan DiT, and our extended Stable Video Diffusion generates the video according to the first frame.

videos and release the potential of video synthesis models. Based on Stable Video Diffusion, our approach achieves a quintupling in the number of frames, while preserving the original generalization abilities. ExVideo is designed within the constraints of limited computational resources, thus it is exceptionally memory-efficient. By integrating this method with other open-source technologies, we facilitate pipelines conducive to the production of high-quality videos. However, the post-tuned models still face the limitations of the base model. To further improve the performance, we will try to train the model on larger datasets in the future.

### ACKNOWLEDGMENTS

This work was supported by the National Natural Science Foundation of China under grant Number 62202170, Fundamental Research Funds for the Central Universities under grant Number YBNLTS2023-014, and Alibaba Group through the Alibaba Innovation Research Program.

### REFERENCES

- [1] Max Bain, Arsha Nagrani, Gül Varol, and Andrew Zisserman. 2021. Frozen in time: A joint video and image encoder for end-to-end retrieval. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 1728–1738.
- [2] Omer Bar-Tal, Lior Yariv, Yaron Lipman, and Tali Dekel. 2023. MultiDiffusion: Fusing Diffusion Paths for Controlled Image Generation. Proceedings of Machine Learning Research 202 (2023), 1737–1752.
- [3] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. 2023. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127 (2023).
- [4] Haoxin Chen, Menghan Xia, Yingqing He, Yong Zhang, Xiaodong Cun, Shaoshu Yang, Jinbo Xing, Yaofang Liu, Qifeng Chen, Xintao Wang, et al. 2023. Videocrafter1: Open diffusion models for high-quality video generation. arXiv preprint arXiv:2310.19512 (2023).
- [5] Haoxin Chen, Yong Zhang, Xiaodong Cun, Menghan Xia, Xintao Wang, Chao Weng, and Ying Shan. 2024. Videocrafter2: Overcoming data limitations for high-quality video diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 7310–7320.
- [6] Junsong Chen, YU Jincheng, GE Chongjian, Lewei Yao, Enze Xie, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, and Zhenguo Li. 2023. PixArt-alpha: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis. In The Twelfth International Conference on Learning Representations.
- [7] Tsai-Shien Chen, Aliaksandr Siarohin, Willi Menapace, Ekaterina Deyneka, Hsiang-wei Chao, Byung Eun Jeon, Yuwei Fang, Hsin-Ying Lee, Jian Ren, MingHsuan Yang, et al. 2024. Panda-70M: Captioning 70M Videos with Multiple Cross-Modality Teachers. arXiv preprint arXiv:2402.19479 (2024).
- [8] Yukang Chen, Shengju Qian, Haotian Tang, Xin Lai, Zhijian Liu, Song Han, and Jiaya Jia. 2023. LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models. In The Twelfth International Conference on Learning Representations.
- [9] Tri Dao. 2023. Flashattention-2: Faster attention with better parallelism and work partitioning. arXiv preprint arXiv:2307.08691 (2023).
- [10] Prafulla Dhariwal and Alexander Nichol. 2021. Diffusion models beat gans on image synthesis. Advances in neural information processing systems 34 (2021), 8780–8794.
- [11] Zhongjie Duan, Chengyu Wang, Cen Chen, Weining Qian, and Jun Huang. 2024. Diffutoon: High-Resolution Editable Toon Shading via Diffusion Models. arXiv preprint arXiv:2401.16224 (2024).
- [12] Jianwei Feng and Dong Huang. 2021. Optimal gradient checkpoint search for arbitrary computation graphs. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 11433–11442.
- [13] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit Haim Bermano, Gal Chechik, and Daniel Cohen-or. 2022. An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion. In The Eleventh International Conference on Learning Representations.
- [14] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. 2014. Generative adversarial nets. Advances in neural information processing systems 27 (2014).
- [15] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. 2023. AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning. In The Twelfth International Conference on Learning Representations.

- [16] Yingqing He, Shaoshu Yang, Haoxin Chen, Xiaodong Cun, Menghan Xia, Yong Zhang, Xintao Wang, Ran He, Qifeng Chen, and Ying Shan. 2023. Scalecrafter: Tuning-free higher-resolution visual generation with diffusion models. In The Twelfth International Conference on Learning Representations.
- [17] Jonathan Ho, Ajay Jain, and Pieter Abbeel. 2020. Denoising diffusion probabilistic models. Advances in neural information processing systems 33 (2020), 6840–6851.
- [18] Edward J Hu, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. 2021. LoRA: Low-Rank Adaptation of Large Language Models. In International Conference on Learning Representations.
- [19] Zhewei Huang, Tianyuan Zhang, Wen Heng, Boxin Shi, and Shuchang Zhou.

2022. Real-time intermediate flow estimation for video frame interpolation. In European Conference on Computer Vision. Springer, 624–642.

- [20] Álvaro Barbero Jiménez. 2023. Mixture of diffusers for scene composition and high resolution image generation. arXiv preprint arXiv:2302.02412 (2023).
- [21] Minguk Kang, Richard Zhang, Connelly Barnes, Sylvain Paris, Suha Kwak, Jaesik Park, Eli Shechtman, Jun-Yan Zhu, and Taesung Park. 2024. Distilling Diffusion Models into Conditional GANs. arXiv preprint arXiv:2405.05967 (2024).
- [22] Akio Kodaira, Chenfeng Xu, Toshiki Hazama, Takanori Yoshimoto, Kohei Ohno, Shogo Mitsuhori, Soichi Sugano, Hanying Cho, Zhijian Liu, and Kurt Keutzer.

2023. StreamDiffusion: A Pipeline-level Solution for Real-time Interactive Generation. arXiv preprint arXiv:2312.12491 (2023).

- [23] Max Ku, Tianle Li, Kai Zhang, Yujie Lu, Xingyu Fu, Wenwen Zhuang, and Wenhu Chen. 2023. Imagenhub: Standardizing the evaluation of conditional image generation models. arXiv preprint arXiv:2310.01596 (2023).
- [24] Jiachen Li, Weixi Feng, Tsu-Jui Fu, Xinyi Wang, Sugato Basu, Wenhu Chen, and William Yang Wang. 2024. T2V-Turbo: Breaking the Quality Bottleneck of Video Consistency Model with Mixed Reward Feedback. arXiv preprint arXiv:2405.18750

(2024).

- [25] Zewen Li, Fan Liu, Wenjie Yang, Shouheng Peng, and Jun Zhou. 2021. A survey of convolutional neural networks: analysis, applications, and prospects. IEEE transactions on neural networks and learning systems 33, 12 (2021), 6999–7019.
- [26] Zhimin Li, Jianwei Zhang, Qin Lin, Jiangfeng Xiong, Yanxin Long, Xinchi Deng, Yingfang Zhang, Xingchao Liu, Minbin Huang, Zedong Xiao, et al. 2024. HunyuanDiT: A Powerful Multi-Resolution Diffusion Transformer with Fine-Grained Chinese Understanding. arXiv preprint arXiv:2405.08748 (2024).
- [27] Yixin Liu, Kai Zhang, Yuan Li, Zhiling Yan, Chujie Gao, Ruoxi Chen, Zhengqing Yuan, Yue Huang, Hanchi Sun, Jianfeng Gao, et al. 2024. Sora: A Review on Background, Technology, Limitations, and Opportunities of Large Vision Models. arXiv preprint arXiv:2402.17177 (2024).
- [28] Paulius Micikevicius, Sharan Narang, Jonah Alben, Gregory Diamos, Erich Elsen, David Garcia, Boris Ginsburg, Michael Houston, Oleksii Kuchaiev, Ganesh Venkatesh, et al. 2017. Mixed precision training. arXiv preprint arXiv:1710.03740

(2017).

- [29] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. 2023. SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis. In The Twelfth International Conference on Learning Representations.
- [30] Ofir Press, Noah A Smith, and Mike Lewis. 2021. Train short, test long: Attention with linear biases enables input length extrapolation. arXiv preprint arXiv:2108.12409 (2021).
- [31] Jeff Rasley, Samyam Rajbhandari, Olatunji Ruwase, and Yuxiong He. 2020. Deepspeed: System optimizations enable training deep learning models with over 100 billion parameters. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 3505–3506.
- [32] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. 2022. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 10684–10695.
- [33] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. 2015. U-net: Convolutional networks for biomedical image segmentation. In Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015: 18th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III 18. Springer, 234–241.
- [34] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. 2023. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 22500–22510.
- [35] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. 2022. Photorealistic text-to-image diffusion models with deep language understanding. Advances in neural information processing systems 35

(2022), 36479–36494.

- [36] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli.

2015. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning. PMLR, 2256–2265.

- [37] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. 2024. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing 568 (2024), 127063.

- [38] Zachary Teed and Jia Deng. 2020. Raft: Recurrent all-pairs field transforms for optical flow. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part II 16. Springer, 402–419.
- [39] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. Advances in neural information processing systems 30 (2017).
- [40] Jiuniu Wang, Hangjie Yuan, Dayou Chen, Yingya Zhang, Xiang Wang, and Shiwei Zhang. 2023. Modelscope text-to-video technical report. arXiv preprint

- arXiv:2308.06571 (2023).

[41] Yaohui Wang, Xinyuan Chen, Xin Ma, Shangchen Zhou, Ziqi Huang, Yi Wang, Ceyuan Yang, Yinan He, Jiashuo Yu, Peiqing Yang, et al. 2023. Lavie: Highquality video generation with cascaded latent diffusion models. arXiv preprint

- arXiv:2309.15103 (2023).

- [42] Yi Wang, Yinan He, Yizhuo Li, Kunchang Li, Jiashuo Yu, Xin Ma, Xinhao Li, Guo Chen, Xinyuan Chen, Yaohui Wang, et al. 2023. Internvid: A large-scale video-text dataset for multimodal understanding and generation. arXiv preprint arXiv:2307.06942 (2023).

- [43] Guangyang Wu, Xin Tao, Changlin Li, Wenyi Wang, Xiaohong Liu, and Qingqing Zheng. 2024. Perception-Oriented Video Frame Interpolation via Asymmetric Blending. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2753–2762.
- [44] Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. 2023. Efficient Streaming Language Models with Attention Sinks. In The Twelfth International Conference on Learning Representations.
- [45] Wenhan Xiong, Jingyu Liu, Igor Molybog, Hejia Zhang, Prajjwal Bhargava, Rui Hou, Louis Martin, Rashi Rungta, Karthik Abinav Sankararaman, Barlas Oguz, et al. 2023. Effective long-context scaling of foundation models. arXiv preprint arXiv:2309.16039 (2023).
- [46] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. 2023. Ip-adapter: Text compatible image prompt adapter for text-to-image diffusion models. arXiv preprint arXiv:2308.06721 (2023).
- [47] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. 2023. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 3836–3847.

