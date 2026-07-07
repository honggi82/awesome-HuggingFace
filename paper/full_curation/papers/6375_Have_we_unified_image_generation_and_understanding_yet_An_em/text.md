## Have we unified image generation and understanding yet? An empirical study of GPT-4o’s image generation ability

Ning Li, Jingran Zhang, Justin Cui University of California, Los Angeles Los Angeles, CA 90095 {ningli23, zhangjingran, justincui}@ucla.edu

# arXiv:2504.08003v1[cs.CV]9Apr2025

### Abstract

OpenAI’s multimodal GPT-4o has demonstrated remarkable capabilities in image generation and editing, yet its ability to achieve world knowledge-informed semantic synthesis—seamlessly integrating domain knowledge, contextual reasoning, and instruction adherence—remains unproven. In this study, we systematically evaluate these capabilities across three critical dimensions: (1) Global Instruction Adherence, (2) Fine-Grained Editing Precision, and (3) Post-Generation Reasoning. While existing benchmarks highlight GPT-4o’s strong capabilities in image generation and editing, our evaluation reveals GPT-4o’s persistent limitations: the model frequently defaults to literal interpretations of instructions, inconsistently applies knowledge constraints, and struggles with conditional reasoning tasks. These findings challenge prevailing assumptions about GPT-4o’s unified understanding and generation capabilities, exposing significant gaps in its dynamic knowledge integration. Our study calls for the development of more robust benchmarks and training strategies that go beyond surface-level alignment, emphasizing context-aware and reasoning-grounded multimodal generation.

### 1 Introduction

The rapid evolution of multimodal AI has led to models capable of generating high-quality images from text, editing visual content with precision, and reasoning across modalities (Zhang et al., 2023; Chen et al., 2024; Zhao et al., 2023). Among these, OpenAI’s GPT-4o represents a major advance, offering unified vision-language capabilities within a single architecture (Team, 2024; Jin et al., 2023). While its performance on standard benchmarks demonstrates impressive image generation and editing (Nichol et al., 2021; Ghosh et al., 2024; Huang et al., 2023; Yan et al., 2025), less is known about its ability to apply contextual reasoning, world knowledge, and instruction adherence

during visual generation (Fu et al., 2024; Meng et al., 2024; Sim et al., 2024). For instance, GPT4o may misinterpret spatial relationships (e.g., confusing left and right), fail to enforce abstract constraints, or produce content that violates basic facts (Sim et al., 2024; Yuksekgonul et al., 2022). These observations motivate a closer empirical examination of how well such models integrate semantic understanding into image generation.

While recent research has focused on benchmarking text-to-image models for photorealism(Saharia et al., 2022), stylistic consistency, or textual alignment (Wu et al., 2023; Xu et al., 2024; Lin et al., 2024), few studies rigorously evaluate how well these systems understand the world they aim to represent (Fu et al., 2024; Hu et al., 2024). For instance, GPT-4o can generate "a dog left of a cat," but its ability to reinterpret spatial terms under reversed global instructions (e.g., mapping "left" to "right") or enforce exclusion constraints (e.g., avoiding out-of-scope objects like "coffee") remains unexplored (Sim et al., 2024). These gaps raise questions about whether such models can dynamically apply world knowledge beyond surface-level pattern recognition (Yuksekgonul et al., 2022).

We conduct a systematic study of GPT-4o’s image generation through three lenses:

- 1. Global Instruction Adherence: Can the model override literal interpretations (e.g., spatial reversals)?
- 2. Fine-Grained Editing Precision: Does it modify elements while preserving context?
- 3. Post-Generation Reasoning: Can it retain and reason over visual context postgeneration?

Our experiments reveal persistent gaps: GPT4o defaults to literal interpretations (e.g., ignoring

reversed directions), inconsistently applies knowledge constraints, and struggles with contextual retention in conditional tasks. These findings challenge assumptions about unified understanding in multimodal LLMs and highlight the need for benchmarks that emphasize knowledge-guided synthesis and contextual generalization.

### 2 Related Work

Text-to-Image (T2I) Models Text-to-image (T2I) generation has advanced through specialized models prioritizing visual fidelity and multimodal architectures unifying vision-language capabilities. Specialized approaches evolved from autoregressive frameworks (Chen et al., 2020; Fan et al., 2024) to diffusion-based methods (Ho et al., 2020; Rombach et al., 2022), where latent diffusion models (LDMs) like Stable Diffusion (Podell et al., 2023) balance efficiency and quality through iterative denoising in compressed latent spaces. Recent unified multimodal systems (Team, 2024; Jin et al., 2023; Ge et al., 2024) extend large language models (LLMs) to visual generation via diffusion heads or autoregressive decoders, with architectures like Transfusion (Zhou et al., 2024) demonstrating bidirectional diffusion-language integration for contextual alignment. Despite progress, both paradigms face challenges: specialized models lack dynamic reasoning beyond fixed text embeddings, while unified architectures struggle with inconsistent knowledge grounding (Wu et al., 2024; Ma et al., 2024), limiting their semantic synthesis capabilities.

T2I Evaluations Evaluating text-to-image (T2I) models necessitates benchmarks that assess both technical fidelity and semantic alignment with real-world knowledge. Early efforts like GenEval (Ghosh et al., 2024) established foundational metrics for basic instruction compliance and photorealism, while Reason-Edit (Huang et al., 2024) introduced instruction-guided editing tasks to evaluate fine-grained manipulation capabilities, albeit prioritizing syntactic edits (e.g., object replacement) over semantic reasoning. The WISE benchmark (Niu et al., 2025) addresses these limitations by pioneering world knowledge-informed evaluation, introducing 1,000 structured prompts across 25 subdomains (e.g., cultural common sense, natural science) and the WiScore metric to assess knowledge-image alignment beyond superficial text-pixel matching. While traditional benchmarks focus on image realism or object placement,

WISE reveals significant deficiencies in models’ ability to integrate domain knowledge, demonstrating that even unified multimodal architectures underperform dedicated T2I models in contextual accuracy—frequently generating anachronisms or scientific inaccuracies despite strong textual understanding capabilities. Similarly, style2paints suggests to use conditional generation to evaluate the understanding of GPT-4o’s generation ability, our work includes more types of prompts and evaluates GPT-4o more systematically.

GPT-ImgEval. The GPT-ImgEval benchmark (Yan et al., 2025) presents the first comprehensive evaluation of GPT-4o’s image generation capabilities across three key tasks: text-toimage generation (via GenEval), instruction-based editing (via Reason-Edit), and world knowledgeinformed synthesis (via WISE). GPT-4o is reported to demonstrate strong performance across all tasks, surpassing prior models in generation control, output quality, and knowledge reasoning. In addition to its quantitative findings, GPT-ImgEval conducts a classifier-based architectural analysis and offers empirical evidence suggesting GPT-4o uses a diffusion-based decoding head. The benchmark also notes generation-related limitations, including occasional content preservation issues, proportion control challenges, and synthesis artifacts, which offer insight into GPT-4o’s design and areas for improvement. Our works addresses GPT-4o’s capacity for semantic understanding instead, examining how well it integrates world knowledge, follows contextual instructions, and reasons across image generation tasks.

- 3 Experiment 3.1 Prompt Construction

To evaluate GPT-4o’s image generation capabilities, we designed experiments based on three distinct types of prompts:

Global Instruction. In this prompt type, GPT-

- 4o is provided with an overarching instruction prior to the image generation task. These global instructions introduce impactful contextual information that influences how subsequent prompts are interpreted. For instance, GPT-4o may be instructed to follow reversed spatial directions—such that "left" should be interpreted as "right" and vice versa. An example would be: although the prompt specifies “generate a cat on the left side of the image,” the model is expected to interpret this as “generate a

[Figure 1]

Figure 1: Demonstration of a global instruction prompt example.

cat on the right side of the image.” This type of prompt is designed to test whether the model can follow contextual logic when generating images, rather than relying solely on surface-level or literal interpretations of instructions.

Image Editing. As highlighted in previous work Yan et al. (2025); Huang et al. (2024), the ability to edit images is a crucial aspect of evaluating a model’s image generation performance. In this prompt type, GPT-4o is tasked with making specific edits to a given image—for example, removing or replacing certain elements. These prompts are designed to assess the model’s capacity for fine-grained image manipulation, contextual consistency, and adherence to detailed visual instructions.

Post-Generation Reasoning. This type of prompt is designed to evaluate whether GPT-4o’s reasoning ability is affected after transitioning into image generation mode. To assess this, the model is first prompted to generate a specific image. It is then given an additional prompt that requires either editing the previously generated image or producing a new image based on logical inference drawn from the initial output. For example, GPT-4o may first be instructed to generate an image of a zebra

drinking water from a river. Subsequently, it is asked to generate an image of a man running on a road—but only if water is present in the previously generated image. This setup enables us to examine the model’s ability to retain contextual understanding and perform conditional reasoning after visual content has been produced.

#### 3.2 Experimental Results

Global Instruction Following In Figure 2, we have demonstrated some examples of generated images with overarching global instructions. In the first two examples, where the instruction specifies reversed spatial directions (e.g., “left” means “right”), the model fails to comply, generating images with the dog correctly positioned on the literal left side. In the third and fifth examples, involving numerical transformations (subtracting or adding a value to mentioned numbers), GPT-4o again falls short—producing images with the exact number of objects (e.g., 5 birds and 2 dolphins) instead of the adjusted quantities. This suggests that while the model understands and executes literal prompts effectively, it struggles to integrate abstract numerical instructions. These results suggest that GPT-4o’s image generation behavior is largely literal and often overlooks abstract global rules. A detailed interaction between us and GPT-4o can be found in Figure 1.

Image Editing Following In Figure 3, some examples of image editing performed by GPT-4o are presented. These examples reveal that while GPT4o demonstrates a degree of capability in image manipulation, it often fails to fully comply with the given editing instructions. For instance, in response to the prompt requesting the removal of people sitting on a couch, the model not only removes the seated individuals but also mistakenly alters the people standing behind the couch. In another case, where the prompt asks to change the reflection of a horse in water to that of a lion, GPT-4o modifies both the horse and its reflection, rather than isolating the transformation to the reflection alone. These results suggest that GPT-4o may struggle with localized and selective image editing tasks, particularly when fine-grained spatial or semantic distinctions are required.

Post-Generation Following In Figure 4, we present examples that reveal potential limitations in GPT-4o’s reasoning ability during image generation. In one scenario, the model is initially

From now on, I will use

From now on, I will use opposite directions, e.g. when I say left, I actually mean right, when I say top, I actually mean bottom.

From now on, your response is limited to the topics of

opposite directions, e.g. when I say left, I actually mean right, when I say top, I

From now on, if I mention a number, subtract 2 before using it.

From now on, if I mention a number, add 3 before using it.

apple, banana, orange, dog,

cat.

actually mean bottom.

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

Got it!

Got it!

Got it!

Got it!

Got it!

Generate an image of a

Generate an image of a

Generate an image of a dog and a cat with the dog being on the left side of the image.

Generate an image of 2 dolphins swimming in the ocean

lemon and a watermelon with the lemon being on top of the watermelon.

monkey sitting on a tree with a mountain in the background.

Generate an image of 5 birds.

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

Figure 2: Examples of generated images with global instructions.

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

Remove anyone sitting in the image.

Make the second floor pink.

Replace the lemon by a lime.

Make the man hold the ice cream with his left hand.

Change the reflection in the water to a loin.

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

Figure 3: Examples of image editing performed by GPT-4o.

prompted to generate an image of a dog and a cat. In the follow-up prompt, it is instructed to replace the dog with a cat and move the resulting image to a beach setting—but only if there are no cats present in the original image. Since the original image already contains a cat, replacing the dog with another cat results in two cats, and thus, the condition for changing the background is not met. However, GPT-4o still performs both operations: it replaces the dog with a cat and also changes the background to a beach. This suggests that the model may struggle with interpreting and following conditional instructions that involve multi-step logical reasoning. Such errors indicate a limitation in GPT-4o’s ability to maintain logical consistency across sequential prompts within the image generation context.

### 4 Conclusion

Our empirical analysis reveals that GPT-4o, while capable of generating high-quality images and performing basic editing tasks, has not yet achieved true unification of image generation and understanding. Across three critical dimensions—global instruction adherence, fine-grained editing precision, and post-generation reasoning—the model frequently defaults to literal interpretations, overlooks abstract or contextual logic, and struggles with conditional reasoning. Our findings underscore the need for more robust benchmarks and training strategies that emphasize reasoning-aware generation, moving beyond surface-level alignment to foster deeper, context-sensitive multimodal intelligence.

Generate an image of two zebras drinking water from the river.

Generate an image of a dog

Generate a white car driving

Generate an image of a man

Generate an image of a boat

and a cat.

on the road.

chasing by a shark.

sailing at sunset.

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

Generate an image of a man running on the road if there is

Make the dog in the image be a cat. Then put them on the

Change the color of the car to black. Ignore my last

If the earth is flat, change the man to a woman and the shark to a gator in the

Remove one zebra. Then add

no water in the previous

beach if there are no cats in the image.

sentence. Make the car driving on the desert.

a loin if no zebra is drinking.

image.

previous image.

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

Figure 4: Examples of post-generation reasoning performed by GPT-4o.

### 5 Future Work

This study represents an initial step toward understanding the limitations of unified image generation and reasoning in multimodal models. In future work, we aim to include more types of prompts with broader coverage across diverse reasoning and generation scenarios. Furthermore, our analysis will extend beyond GPT-4o to include a range of state-of-the-art multimodal models. By benchmarking across multiple systems, we seek to identify common failure modes and better characterize the broader challenges in achieving true unification of image generation and reasoning.

### References

Liang Chen, Zekun Wang, Shuhuai Ren, Lei Li, Haozhe Zhao, Yunshui Li, Zefan Cai, Hongcheng Guo, Lei Zhang, Yizhe Xiong, et al. 2024. Next token prediction towards multimodal intelligence: A comprehensive survey. arXiv preprint arXiv:2412.18619.

Mark Chen, Alec Radford, Rewon Child, Jeff Wu, Heewoo Jun, Prafulla Dhariwal, David Luan, and Ilya Sutskever. 2020. Generative pretraining from pixels.

Lijie Fan, Tianhong Li, Siyang Qin, Yuanzhen Li, Chen Sun, Michael Rubinstein, Deqing Sun, Kaiming He, and Yonglong Tian. 2024. Fluid: Scaling autoregressive text-to-image generative models with continuous tokens. arXiv preprint arXiv:2410.13863.

Xingyu Fu, Muyu He, Yujie Lu, William Yang Wang, and Dan Roth. 2024. Commonsense-t2i challenge: Can text-to-image generation models understand commonsense? arXiv preprint arXiv:2406.07546.

Yuying Ge, Sijie Zhao, Jinguo Zhu, Yixiao Ge, Kun Yi, Lin Song, Chen Li, Xiaohan Ding, and Ying Shan. 2024. Seed-x: Multimodal models with unified multigranularity comprehension and generation. arXiv preprint arXiv:2404.14396.

Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. 2024. Geneval: An object-focused framework for evaluating text-to-image alignment. Advances in Neural Information Processing Systems, 36.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. 2020. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840– 6851.

Xiwei Hu, Rui Wang, Yixiao Fang, Bin Fu, Pei Cheng, and Gang Yu Ella. 2024. Equip diffusion models with llm for enhanced semantic alignment. arXiv preprint arXiv:2403.05135.

Kaiyi Huang, Kaiyue Sun, Enze Xie, Zhenguo Li, and Xihui Liu. 2023. T2i-compbench: A comprehensive benchmark for open-world compositional text-toimage generation. Advances in Neural Information Processing Systems, 36:78723–78747.

Yuzhou Huang, Liangbin Xie, Xintao Wang, Ziyang Yuan, Xiaodong Cun, Yixiao Ge, Jiantao Zhou, Chao Dong, Rui Huang, Ruimao Zhang, et al. 2024.

Smartedit: Exploring complex instruction-based image editing with multimodal large language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8362– 8371.

Yang Jin, Kun Xu, Liwei Chen, Chao Liao, Jianchao Tan, Bin Chen, Chenyi Lei, An Liu, Chengru Song, Xiaoqiang Lei, et al. 2023. Unified language-vision pretraining with dynamic discrete visual tokenization. arXiv preprint arXiv:2309.04669.

Zhiqiu Lin, Deepak Pathak, Baiqi Li, Jiayao Li, Xide Xia, Graham Neubig, Pengchuan Zhang, and Deva Ramanan. 2024. Evaluating text-to-visual generation with image-to-text generation. In European Conference on Computer Vision, pages 366–384. Springer.

Yiyang Ma, Xingchao Liu, Xiaokang Chen, Wen Liu, Chengyue Wu, Zhiyu Wu, Zizheng Pan, Zhenda Xie, Haowei Zhang, Liang Zhao, et al. 2024. Janusflow: Harmonizing autoregression and rectified flow for unified multimodal understanding and generation. arXiv preprint arXiv:2411.07975.

Fanqing Meng, Wenqi Shao, Lixin Luo, Yahong Wang, Yiran Chen, Quanfeng Lu, Yue Yang, Tianshuo Yang, Kaipeng Zhang, Yu Qiao, et al. 2024. Phybench: A physical commonsense benchmark for evaluating text-to-image models. arXiv preprint arXiv:2406.11802.

Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. 2021. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv preprint arXiv:2112.10741.

Yuwei Niu, Munan Ning, Mengren Zheng, Bin Lin, Peng Jin, Jiaqi Liao, Kunpeng Ning, Bin Zhu, and Li Yuan. 2025. Wise: A world knowledge-informed semantic evaluation for text-to-image generation. Preprint, arXiv:2503.07265.

Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. 2023. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. 2022. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695.

Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. 2022. Photorealistic text-to-image diffusion models with deep language understanding. Advances in neural information processing systems, 35:36479–36494.

Shang Hong Sim, Clarence Lee, Alvin Tan, and Cheston Tan. 2024. Evaluating the generation of spatial relations in text and image generative models. arXiv preprint arXiv:2411.07664.

style2paints. Tweet from @lvminzhang on mar 29, 2025. X (formerly Twitter). Accessed: 2025-04-09.

Chameleon Team. 2024. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818.

Chengyue Wu, Xiaokang Chen, Zhiyu Wu, Yiyang Ma, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, Chong Ruan, et al. 2024. Janus: Decoupling visual encoding for unified multimodal understanding and generation. arXiv preprint arXiv:2410.13848.

Xiaoshi Wu, Yiming Hao, Keqiang Sun, Yixiong Chen, Feng Zhu, Rui Zhao, and Hongsheng Li. 2023. Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis. arXiv preprint arXiv:2306.09341.

Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. 2024. Imagereward: Learning and evaluating human preferences for text-to-image generation. Advances in Neural Information Processing Systems, 36.

Zhiyuan Yan, Junyan Ye, Weijia Li, Zilong Huang, Shenghai Yuan, Xiangyang He, Kaiqing Lin, Jun He, Conghui He, and Li Yuan. 2025. Gpt-imgeval: A comprehensive benchmark for diagnosing gpt4o in image generation. arXiv preprint arXiv:2504.02782.

Mert Yuksekgonul, Federico Bianchi, Pratyusha Kalluri, Dan Jurafsky, and James Zou. 2022. When and why vision-language models behave like bags-ofwords, and what to do about it? arXiv preprint arXiv:2210.01936.

Chenshuang Zhang, Chaoning Zhang, Mengchun Zhang, and In So Kweon. 2023. Text-to-image diffusion models in generative ai: A survey. arXiv preprint arXiv:2303.07909.

Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, et al. 2023. A survey of large language models. arXiv preprint arXiv:2303.18223.

Chunting Zhou, Lili Yu, Arun Babu, Kushal Tirumala, Michihiro Yasunaga, Leonid Shamis, Jacob Kahn, Xuezhe Ma, Luke Zettlemoyer, and Omer Levy. 2024. Transfusion: Predict the next token and diffuse images with one multi-modal model. arXiv preprint arXiv:2408.11039.

