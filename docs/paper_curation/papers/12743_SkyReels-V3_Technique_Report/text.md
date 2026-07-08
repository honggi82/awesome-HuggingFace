# arXiv:2601.17323v2[cs.CV]29Jan2026

## SkyReels-V3 Technique Report

SkyReels Team

### Abstract

Video generation serves as a cornerstone for building world models, where multimodal contextual inference stands as the defining test of capability. In this end, we present SkyReels-V3, a conditional video generation model, built upon a unified multimodal in-context learning framework with diffusion Transformers. SkyReelsV3 model supports three core generative paradigms within a single architecture: reference images-to-video synthesis, video-to-video extension and audio-guided video generation. (i) reference images-to-video model is designed to produce highfidelity videos with strong subject identity preservation, temporal coherence, and narrative consistency. To enhance reference adherence and compositional stability, we design a comprehensive data processing pipeline that leverages cross-frame pairing, image editing, and semantic rewriting, effectively mitigating copy–paste artifacts. During training, an image–video hybrid strategy combined with multiresolution joint optimization is employed to improve generalization and robustness across diverse scenarios. (ii) video extension model integrates spatio-temporal consistency modeling with large-scale video understanding, enabling both seamless single-shot continuation and intelligent multi-shot switching with professional cinematographic patterns. (iii) Talking avatar model supports minute-level audioconditioned video generation by training first-and-last frame insertion patterns and reconstructing key-frame inference paradigms. On the basis of ensuring visual quality, synchronization of audio and videos has been optimized. Extensive evaluations demonstrate that SkyReels-V3 achieves state-of-the-art or near state-ofthe-art performance on key metrics including visual quality, instruction following, and specific aspect metrics, approaching leading closed-source systems. Github: https://github.com/SkyworkAI/SkyReels-V3.

### 1 Introduction

World models aim to capture, simulate, and forecast the dynamics of complex real-world environments, and they form a fundamental basis for deploying artificial intelligence in practical scenarios [11]. In between, video generation frameworks encode rich geometric, semantic, and physical knowledge through the synthesis of visual sequences, thereby enabling effective modeling and prediction of the physical world, especially for multi-modal conditions. In recent years, diffusion-based Transformers [13, 4] architecture has driven significant advances in video generation. A wide range of commercial systems—including Veo [7], Sora [12], Seedance [6, 2], Kling [9], as well as opensource models such as Wanx [14], HunyuanVideo [15, 8], SkyReels [1], and CogVideoX [17], have demonstrated strong performance across multiple dimensions. However, multimodal in-context in video generation is still under-explored.

In this repport, we introduce SkyReels-V3, a unified multimodal condition video generation framework, designed to support a wide range of high-quality video synthesis tasks within a single model family. Built upon a multimodal in-context learning paradigm, SkyReels V3 seamlessly integrates visual reference, video, audio, and textual inputs to enable flexible and controllable video generation. The framework natively supports three core capabilities: reference images to video, video-to-video extension, and audio-guided video generation, also known as talking avatar. At the architectural

Preprint.

[Figure 1]

[Figure 2]

[Figure 3]

a man playing with his dog in front of the house

Reference Images Prompt

Generated Video

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

a man sitting in the park, a cat walking around his feet.

[Figure 8]

A woman stands besides the bridge, holding a bowl.

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

A woman holds a plant in a space capsule.

In the park, two tigers and a small dog are chasing each other.

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

- Figure 1: Reference Images to Video Results. SkyReels-V3 can facilitate dynamic interplay between different subjects within specified contexts.

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

The man pours the milk from the cup into a plate for the puppy to drink.

Reference Images Prompt Generated Video

[Figure 30]

[Figure 31]

[Figure 32]

Three women are drinking wine in a bar

[Figure 33]

[Figure 34]

Four men are meeting in a room.

- Figure 2: Reference Images to Video Results. SkyReels-V3 can enable dynamic interactions between diverse subjects (characters/objects) within arbitrary scenes.

level, SkyReels-V3 incorporates large-scale diffusion Transformers while with carefully designed alignment strategies with multimodal condition and advanced spatiotemporal consistency modeling. Through hybrid image–video as well as multi-resolution joint optimization, the system achieves precise instruction following, high-fidelity motion generation and superior sub-domain capacity such as robust identity preservation and precise audio-visual synchronization. These design choices allow SkyReels-V3 to move beyond frame-level synthesis, enabling coherent narrative progression and cinematic-quality visual composition.

With its strong generalization ability and modular design, SkyReels-V3 can be applied to diverse real-world scenarios, including professional video production, virtual avatars, short-form content creation, live-stream commerce, and digital entertainment. Extensive evaluations demonstrate that SkyReels-V3 reaches or surpasses industry-leading performance across key metrics, making it a powerful open-source foundation for next-generation video generation research and applications.

### 2 Methods and Evaluation

The SkyReels-V3 model family supports a range of capabilities, including multi-reference image-tovideo synthesis, audio-guided video generation, and video-to-video extension. This chapter provides a detailed descriptions and performance evaluation to these features.

#### 2.1 Reference Images to Video

SkyReels-V3 can synthesizes temporally coherent video sequences conditioned on multiple visual references and a textual prompt. Given one to four reference images, which may correspond to characters, objects, or background scenes, the model generates videos that preserve identity attributes, spatial composition, and narrative continuity while following high-level semantic instructions.

[Figure 35]

[Figure 36]

[Figure 37]

In the minimalist exhibition area, dominated by light gray

tones, soft lighting gently illuminated the clothing. A woman elegantly showcased a brown slip dress, explaining its cut and texture.

Reference Images Prompt Generated Video

In the snow-covered outdoors, next to a red brick building, a woman dressed in thick winter clothes and wearing a hat covered in snowflakes is enthusiastically showing off a beautifully packaged bottle of fruity wine to those around her.

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

In a brightly lit and simply decorated sports equipment

showroom, various sports equipment were displayed around the perimeter, next to a wall displaying brand logos. A man dressed in a dark gray tracksuit was … Inside the clean and bright exhibition area, soft lighting was evenly distributed, and various sports-related items were displayed around. A man in a black shirt was carefully demonstrating a pair of simple yet brightly colored sneakers.

[Figure 44]

[Figure 45]

[Figure 46]

- Figure 3: Reference Images to Video Results. SkyReels-V3 can enable instant video creation for diverse live commerce hosts and settings.

[Figure 47]

[Figure 48]

[Figure 49]

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

Reference Image Prompt Generated Video

A modern urban garden, lush and verdant, with vibrant flowers in the background. At the center of the scene is a woman in a floral dress, twirling gracefully, her skirt billowing in the breeze. Sunlight filters through the leaves, casting dappled shadows on her skirt, creating a breathtaking interplay of light and shadow.

The soft morning sunlight spills onto the tranquil garden path, its warm glow gently illuminating the lush greenery. In the distance, brick walls and wooden fences are faintly visible. The gaze is fixed on the potted plant in the owner's hands; the owner gently turns the pot, sunlight dancing and shimmering on the leaves, creating a fresh and pleasant atmosphere.

Morning light spills across the tranquil sports field, a gentle breeze rustles the lush green grass. The camera focuses on an athlete in a dark gray short-sleeved sports shirt, running briskly along the track. Sweat glistens in the sunlight; he works hard, yet his steps remain steady and powerful. The camera zooms in on his upper body, revealing the lightweight, breathable fabric of his shirt, which shimmers in the light as he moves, highlighting both comfort and professionalism.

A tranquil garden path, gently swaying in the breeze, dotted with blooming blue flowers and verdant leaves in the distance. The camera focuses on a woman in a blue floral dress, tenderly stroking the hem as it gently sways in the wind. Sunlight casts dappled shadows on the dress, and the woman smiles, her skirt billowing slightly, as if she has become one with nature.

In a modern, urban-style office, the camera focuses on a model wearing a wool coat, as she gently adjusts her collar and fastens her belt. Sunlight illuminates the coat, adding warmth and elegance.

Sunlight streams down the mountain road, a gentle breeze blows, and the trees provide ample shade. The camera focuses on a black mountain bike, the rider pedaling softly, the wheels kicking up dust as they turn, the winding mountain road in the background radiating energy and freedom.

In a tranquil country courtyard, the morning sun shines on the lush olive trees, and a gentle breeze rustles the leaves. The camera focuses on a bottle of olive oil as a skilled chef unscrews the cap and pours the oil into an elegant glass bowl. The oil gleams, and the aroma of the ingredients in the bowl is intoxicating.

In a modern, minimalist garden, a gentle breeze stirs, and sunlight falls on a black floral dress. The background features greenery and a white stone wall. The camera focuses on the dress; a hand lightly lifts the hem, causing it to sway in the breeze. Sunlight casts shifting shadows on the dress, and a close-up reveals the exquisite floral pattern on the fabric.

- Figure 4: Reference Images to Video Results. SkyReels-V3 can make advertising and product demonstration with one picture.

Reference-Preserving Data Construction. For multi-reference image-guided video generation, the quality of the reference image-to-target video pairs is crucial. To this end, we introduce a dedicated data processing pipeline. Initially, we filter video data from a massive in house dataset, selecting clips that exhibit both high visual quality and significant dynamic motion. Then, reference frames are selected from continuous video sequences using a cross-frame pairing strategy [5, 3], ensuring temporal diversity while maintaining semantic consistency. Image editing models [10, 16] are then applied to extract subject regions and perform background completion, together with semantic rewriting, to construct training pairs that avoid trivial frame copying and reduce copy-and-paste artifacts. Furthermore, we have developed multiple filtering steps to remove distorted and inconsistent reference images generated by editing models.

- Table 1: Quantitative comparison of video generation models on reference consistency, instruction following, and visual quality. Higher is better.

Model Reference Consistency ↑ Instruction Following ↑ Visual Quality ↑

Vidu Q2 0.5961 27.84 0.7877 Kling 1.6 0.6630 29.23 0.8034 PixVerse V5 0.6542 29.34 0.7976 SkyReels-V3 0.6698 27.22 0.8119

[Figure 63]

[Figure 64]

A girl hikes through wind-swept grassland, cupping a lit candle against the breeze with one hand.

Reference Video Prompt Generated Video

[Figure 65]

[Figure 66]

The camera gradually zooms in and focuses on the white building in the center of the video

[Figure 67]

[Figure 68]

More smoke is rising from the ridge line.

[Figure 69]

[Figure 70]

A man is rowing vigorously, his body rocking forward and back.

[Figure 71]

[Figure 72]

After sprinkling seasoning, the chef evenly spreads it across the fish fillets with his hand.

[Figure 73]

[Figure 74]

A girl rides a bicycle along the beach, swaying left and right as she approaches the camera, with strong sea winds blowing.

#### Figure 5: Single-shot Video Extension Results.

Multi-Reference Conditioning. To effectively integrate heterogeneous reference inputs, SkyReelsV3 employs a unified multi-reference conditioning strategy that jointly encodes visual and textual information. For each reference image, we encode it using the video VAE and subsequently concatenate the resulting latent representation with the video latents. By allowing up to four reference images, the model supports flexible scene composition and enables fine-grained control over subject appearance and background structure. This design facilitates complex multi-subject and multi-element video generation without requiring explicit manual composition.

Training Strategy. We train the model using an image–video hybrid training scheme that jointly leverages large-scale image and video datasets. This approach enhances generalization by exposing the model to both static appearance cues and dynamic motion patterns. In addition, multi-resolution joint training is employed to improve robustness across different spatial scales and aspect ratios, enabling the model to natively support a wide range of output configurations.

Benchmark. We construct a test set comprising 200 data pairs, with sources spanning scenarios such as film and television, e-commerce, and advertising. The types of reference images include those featuring characters, animals, objects, and background scenes. The model’s capabilities are evaluated from three primary perspectives: Reference Consistency, Instruction Following, and Visual Quality. Specifically, Reference Consistency encompasses aspects such as facial consistency, clothing consistency, object consistency, and background consistency. Visual Quality is assessed based on image quality, dynamic degree, aesthetic quality, and motion smoothness. SkyReels-V3 has been benchmarked against leading contemporary models, with comparative results presented in Table 1. The results demonstrate that the SkyReels-V3 model achieves state-of-the-art performance within the industry. We also show qualitative results in Figure 1, Figure 2, Figure 3, and Figure 4. These results indicate that the model demonstrates strong generalization capabilities across a wide range of scenarios.

[Figure 75]

[Figure 76]

Cut from the shot of the two people interacting to a close-up of the man's hand lightly resting on the woman's shoulder.

Reference Video Prompt Generated Video

[Figure 77]

[Figure 78]

Cut to a close-up shot.

cuts from a medium close-up of a man to a much tighter close-up on his face. He turns to be shocked and angry.

[Figure 79]

[Figure 80]

- Figure 6: Shot-switching Video Extension (Cut In) Results.

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

Reference Video Prompt Generated Video

Cut out from the current shot.

Cut out from the current shot.

Cut out from the current shot.

- Figure 7: Shot-switching Video Extension (Cut Out) Results.

- 2.2 Video Extension

The SkyReels-V3 Video Extension Model is designed to extend an input video segment into temporally coherent and semantically consistent subsequent content under textual guidance. Given an initial video clip, the model generates continuation segments that preserve motion dynamics, scene structure, and visual style, while maintaining narrative coherence across extended temporal horizons. It supports that: (i) Dual extension modes. The model supports both single-shot video extension and shot switching video extension. For shot switching video extension, five predefined transition types are supported, and the mode can be selected either manually or through automatic detection. (ii) High-quality visual synthesis. The system produces visually coherent extensions with stable composition, smooth motion, and seamless temporal continuity. (iii) Style-consistent generation. Visual style cues from the input video are explicitly preserved, enabling faithful continuation across realistic, cinematic, and domain-specific aesthetics. (iv) High-definition and flexible outputs. The model supports 720p video generation with adjustable extension durations ranging from 5 to 30 seconds for single-shot extension, as well as multiple aspect ratios (1:1, 3:4, 4:3, 16:9, and 9:16).

To achieve this, we introduce: (i) Shot switching detector. We have developed a shot switching detector to analyze long-form videos. It identifies whether shot transitions (cuts) are present and classifies their types. Currently supported transition types include single shot, cut-in, cut-out, multiangle, shot/reverse shot, and cut-away. This detector enables the construction of effective training data. (ii) Unified multi-segment positional encoding and hierarchical training. A unified positional encoding scheme, combined with hybrid hierarchical data training, enables accurate motion modeling and smooth transitions across complex, multi-segment video extensions. (iii) Robust spatiotemporal modeling. The model effectively handles challenging scenarios, including rapid motion, multisubject interactions, and abrupt scene changes, while enforcing physical plausibility and temporal consistency.

- Figure 5 presents the results for single-shot extension, and Figure 6 to 10 illustrate the extension results for cut-in, cut-out, multi-angle, shot/reserve shot, and cut-away. The model generalizes well across diverse application scenarios, including cinematic content creation, short-form series production, game cutscenes, and long-form video enhancement, producing high-definition outputs with sharp visual details and natural motion dynamics.

[Figure 87]

[Figure 88]

Create a top side angle view of the robot playing the guitar.

Reference Video Prompt Generated Video

switch to a medium view from a more frontal angle, showing the same man now actively driving the car.

[Figure 89]

[Figure 90]

cuts to a profile view of the woman from the side, still engaged in the same makeup application.

[Figure 91]

[Figure 92]

#### Figure 8: Shot-switching Video Extension (Multi Angle) Results.

[Figure 93]

[Figure 94]

Transfer to reverse shot. The man talks and listens to the woman.

Reference Video Prompt Generated Video

[Figure 95]

[Figure 96]

Rever shot shot, the younger man from the front.

[Figure 97]

[Figure 98]

Switch to reverse shot.

Figure 9: Shot-switching Video Extension (Shot/Reverse Shot) Results.

- 2.3 Talking Avatar

Talking avatar model enables high-quality audio-conditioned video generation from a single portrait image and an input audio clip. The system is designed to produce temporally coherent, visually realistic videos with accurate audio–visual synchronization, supporting long-form generation and multi-character interactions. It key improvements includes: (i) High-fidelity visual synthesis and precise lip synchronization. The model can generate 720p videos at 24 fps, delivering smooth motion and fine-grained facial details. It supports multiple languages and speech types, ensuring that lip movements are closely aligned with phoneme-level audio dynamics, thereby enhancing realism and perceptual authenticity. (ii) Multi-style character generalization. The framework is compatible with a wide range of visual styles, including photorealistic humans, cartoons, animals, and stylized characters. This flexibility enables broad applicability in virtual avatars, brand representation, and creative content generation. (iii) Long-form coherent video generation. The model supports minutelevel video synthesis in a single forward generation process, maintaining identity consistency, motion continuity, and expressive stability over extended durations. This makes it suitable for applications such as instructional videos, news narration, and long-form storytelling. (iv) Multi-character scene support. The system is optimized for scenarios involving multiple avatars, allowing explicit role assignment and coordinated interactions. This capability facilitates the generation of dialogues, interviews, and other complex conversational scenes. Note that in multi-person scenes, the mask must be used to specify which character is speaking.

Talking avatar model jointly analyzes audio signals, visual inputs, and textual cues to infer appropriate facial expressions, head movements, and camera dynamics, resulting in semantically and emotionally aligned video generative performances. To achieve accurate lip synchronization, the model is trained using dedicated audio–visual alignment strategies with region masking that explicitly model the correspondence between speech units and facial motion. This design ensures robust performance across diverse languages, speaking styles, singing voices, and rapid speech patterns. Furthermore, a key-frame-constrained generation framework is introduced to improve temporal stability in long videos. The model first establishes structurally important key content and then generates smooth transitions between key frames, ensuring consistent character appearance and natural motion flow throughout the sequence. In internal evaluations against representative mainstream talking avatar models, talking avatar model demonstrates superior performance across multiple dimensions, including overall visual quality, lip synchronization accuracy, and expressive realism. The results indicate a clear advantage in producing stable, high-quality, and perceptually convincing talking avatar videos.

[Figure 99]

[Figure 100]

the camera switches and focuses on the ceiling, clearly showing the uniform and smooth coating effect on the surface of the ceiling…

Reference Video Prompt Generated Video

cuts away from the scene where four people are sitting around and dining, then slowly moves upward to show the ceiling decorations …

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

The camera cuts away from the person and moves to show the distant horses and the ranch environment.

#### Figure 10: Shot-switching Video Extension (Cut Away) Results.

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

Reference Image Generated Video

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

Reference Audio

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

[Figure 133]

[Figure 134]

- Figure 11: Generation results with multiple objectives and styles. The results show the model’s generalization across various non-human subjects as well as different styles.

### 3 Conclusion

In this work, we presented SkyReels-V3, a unified multimodal video generation framework that integrates reference-based video synthesis, video extension, and audio-driven talking avatar generation within a single in-context learning paradigm. By jointly modeling visual, temporal, and auditory signals, SkyReels-V3 advances video generation from short, frame-level synthesis toward coherent, narrative-level content creation. Through innovations in multimodal conditioning, hybrid image–video training, hierarchical spatiotemporal modeling, and efficient token fusion strategies, the proposed system achieves strong subject consistency, high-fidelity motion generation, and robust instruction following across diverse tasks and aspect ratios. Extensive empirical evaluations demonstrate that SkyReels-V3 attains competitive performance on multiple benchmarks, rivaling leading closed-source models in various domain performance. Overall, SkyReels-V3 represents a significant step toward scalable, controllable, and general-purpose video generation systems, and provides a solid foundation for future research in multimodal generative modeling and cinematic-level video synthesis.

### References

[1] Guibin Chen, Dixuan Lin, Jiangping Yang, Chunze Lin, Junchen Zhu, Mingyuan Fan, Hao Zhang, Sheng Chen, Zheng Chen, Chengcheng Ma, et al. Skyreels-v2: Infinite-length film

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

Reference Image Generated Video

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

Reference Audio

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

- Figure 12: Multi-person results. It presents a dialogue scenario, where characters correctly respond to conversational audio by switching between speaking and idle states. It showcase performance in multi-person scenes, with coordinated behavior for both speakers and listeners.

Reference Image Generated Video

[Figure 153]

[Figure 154]

[Figure 155]

5S 10S 30S 60S 120S

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

Reference Audio

- Figure 13: Minute-long video generation. It show consistent visual effects with accurate audio alignment.

generative model. arXiv preprint arXiv:2504.13074, 2025.

- [2] Siyan Chen, Yanfei Chen, Ying Chen, Zhuo Chen, Feng Cheng, Xuyan Chi, Jian Cong, Qinpeng Cui, Qide Dong, Junliang Fan, et al. Seedance 1.5 pro: A native audio-visual joint generation foundation model. arXiv preprint arXiv:2512.13507, 2025.
- [3] Zhuowei Chen, Bingchuan Li, Tianxiang Ma, Lijie Liu, Mingcong Liu, Yi Zhang, Gen Li, Xinghui Li, Siyu Zhou, Qian He, et al. Phantom-data: Towards a general subject-consistent video generation dataset. arXiv preprint arXiv:2506.18851, 2025.
- [4] Zhengcong Fei, Mingyuan Fan, Changqian Yu, Debang Li, and Junshi Huang. Scaling diffusion transformers to 16 billion parameters. arXiv preprint arXiv:2407.11633, 2024.
- [5] Zhengcong Fei, Debang Li, Di Qiu, Jiahua Wang, Yikun Dou, Rui Wang, Jingtao Xu, Mingyuan Fan, Guibin Chen, Yang Li, et al. Skyreels-a2: Compose anything in video diffusion transformers. arXiv preprint arXiv:2504.02436, 2025.
- [6] Yu Gao, Haoyuan Guo, Tuyen Hoang, Weilin Huang, Lu Jiang, Fangyuan Kong, Huixia Li, Jiashi Li, Liang Li, Xiaojie Li, et al. Seedance 1.0: Exploring the boundaries of video generation models. arXiv preprint arXiv:2506.09113, 2025.
- [7] Google. Veo. https://deepmind.google/models/veo/, 2024.

- Table 2: Quantitative comparison of talking avatar models on audio–visual synchronization, visual quality, and character consistency. Higher is better.

Model Audio–Visual Sync ↑ Visual Quality ↑ Character Consistency ↑

OmniHuman 1.5 8.25 4.60 0.81 KlingAvatar 8.01 4.55 0.78 HunyuanAvatar 6.72 4.50 0.74 SkyReels-V3 8.18 4.60 0.80

- [8] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.
- [9] Kuaishou. Kling. https://klingai.com, 2024.
- [10] Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, et al. Flux. 1 kontext: Flow matching for in-context image generation and editing in latent space. arXiv preprint arXiv:2506.15742, 2025.
- [11] Yutaka Matsuo, Yann LeCun, Maneesh Sahani, Doina Precup, David Silver, Masashi Sugiyama, Eiji Uchibe, and Jun Morimoto. Deep learning, reinforcement learning, and world models. Neural Networks, 152:267–275, 2022.
- [12] OpenAI. Sora. https://openai.com/sora/, 2024.
- [13] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4195–4205, 2023.
- [14] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.
- [15] Bing Wu, Chang Zou, Changlin Li, Duojun Huang, Fang Yang, Hao Tan, Jack Peng, Jianbing Wu, Jiangfeng Xiong, Jie Jiang, et al. Hunyuanvideo 1.5 technical report. arXiv preprint arXiv:2511.18870, 2025.
- [16] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, Yuxiang Chen, Zecheng Tang, Zekai Zhang, Zhengyi Wang, An Yang, Bowen Yu, Chen Cheng, Dayiheng Liu, Deqing Li, Hang Zhang, Hao Meng, Hu Wei, Jingyuan Ni, Kai Chen, Kuan Cao, Liang Peng, Lin Qu, Minggang Wu, Peng Wang, Shuting Yu, Tingkun Wen, Wensen Feng, Xiaoxiao Xu, Yi Wang, Yichang Zhang, Yongqiang Zhu, Yujia Wu, Yuxuan Cai, and Zenan Liu. Qwen-image technical report, 2025.
- [17] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024.

### Contributors and Acknowledgments

Debang Li, Zhengcong Fei, Tuanhui Li, Yikun Dou, Zheng Chen, Jiangping Yang, Mingyuan Fan, Jingtao Xu, Jiahua Wang, Baoxuan Gu, Mingshan Chang, Wenjing Cai, Yuqiang Xie, Binjie Mao, Youqiang Zhang, Nuo Pang, Hao Zhang, Yuzhe Jin, Zhiheng Xu, Dixuan Lin, Guibin Chen, Yahui Zhou

