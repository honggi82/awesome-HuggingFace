[Figure 1]

Generative AI Research

## ANOLE: An Open, Autoregressive, Native Large Multimodal Models for Interleaved Image-Text Generation

[Figure 2]

Ethan Chern* Jiadi Su* Yan Ma* Pengfei Liu† Generative AI Research Lab (GAIR)

# arXiv:2407.06135v1[cs.CL]8Jul2024

##### Abstract

Previous open-source large multimodal models (LMMs) have faced several limitations: (1) they often lack native integration, requiring adapters to align visual representations with pre-trained large language models (LLMs); (2) many are restricted to single-modal generation; (3) while some support multimodal generation, they rely on separate diffusion models for visual modeling and generation. To mitigate these limitations, we present ANOLE, an open, autoregressive, native large multimodal model for interleaved image-text generation. We build ANOLE from Meta AI’s Chameleon, adopting an innovative fine-tuning strategy that is both data-efficient and parameter-efficient. ANOLE demonstrates high-quality, coherent multimodal generation capabilities. We have open-sourced our model, training framework, and instruction tuning data.

Homepage: https://gair-nlp.github.io/anole Code: https://github.com/GAIR-NLP/anole Hugging Face Model: https://huggingface.co/GAIR/Anole-7b-v0.1

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

Figure 1: An example of ANOLE generating a high-quality and coherent interleaved image-text sequence on how to cook eggs.

* Co-first authors. † Corresponding author

1

##### 1 Introduction

Since the introduction of Meta AI’s LLaMA (Touvron et al., 2023) in February 2023, autoregressive open-source large language models (LLMs) such as LLaMA (Touvron et al., 2023), Alpaca (Taori et al., 2023), Vicuna (Chiang

- et al., 2023), Mistral (Jiang et al., 2023), Phi (Gunasekar et al., 2023), Gemma (Team et al., 2024), Olmo (Groeneveld
- et al., 2024) and LLM360 (Liu et al., 2023c) have democratized and advanced the development of LLMs. Efforts in open-sourcing large multimodal models (LMMs) are also ongoing, though at a slower pace compared to LLMs. Notable open-sourced LMMs include LLaVA (Liu et al., 2023b,a), CogVLM (Wang et al., 2023), DreamLLM (Dong et al., 2023), Emu2 (Sun et al., 2024) and Cambrian (Tong et al., 2024). However, current open-source LMMs have several significant limitations: (1) Many focus solely on multimodal understanding without multimodal generation (Liu et al., 2023b,a; Wang et al., 2023), (2) most are not natively multimodal (i.e., not trained on multimodal data from the pretraining stage) and rely on pretrained LLMs as their backbone (Liu et al., 2023b,a; Wang et al., 2023; Dong et al., 2023; Sun et al., 2024), and (3) those with vision generation capabilities require an additional diffusion model for vision modeling and generation (Dong et al., 2023; Sun et al., 2024). This reliance on extra mechanisms can introduce complexity and inefficiency in both training and inference time.

Given the limitations in current open-source LMMs, the AI community eagerly anticipates the emergence of truly open, autoregressive, native LMMs with multimodal generation capabilities. The goal is to be able to develop LMMs in the same way we do with LLMs. To address this gap, we introduce ANOLE. As shown in Fig. 1, ANOLE can generate a high-quality, coherent recipe for cooking eggs in just a few seconds.

ANOLE is built on top of Chameleon (Team, 2024) by Meta AI. Chameleon represents a significant advancement in multimodal AI, showcasing the potential of early-fusion, token-based autoregressive approaches for multimodal modeling. According to their paper, Chameleon has demonstrated impressive capabilities in understanding and generating interleaved sequences of images and text, pushing the boundaries of what is possible in multimodal AI. The latest open-source release of Chameleon has shown strong performance in text understanding, text generation, and multimodal comprehension. However, the current open-source version of Chameleon does not support image generation or multimodal generation. We build ANOLE to facilitate Chameleon’s capabilities in vision and multimodal generation without compromising its strengths in text generation and multimodal comprehension. ANOLE addresses the limitations of the current open-source release of Chameleon, making the full potential of multimodal generation accessible to the broader research community. The key contributions of our work are fourfold:

- 1. Full Open-Source Implementation: ANOLE has facilitated the vision and multimodal generation capabilities from Chameleon through an innovative fine-tuning approach, unlocking the model’s most crucial technological aspects. This comprehensive open-source release allows researchers and developers to fully utilize and build upon it.
- 2. Data and Parameter Efficient Fine-Tuning: Our method fine-tunes fewer than 40M parameters, requiring only about 6,000 samples to effectively facilitate vision and multimodal generation capabilities. This demonstrates a highly efficient approach to facilitate complex functionality in LMMs.
- 3. Training, Multimodal Inferece, and Qualitative Evaluation: We provide a training and multimodal inference framework for unified tokenizer-based multimodal models. This infrastructure significantly lowers the barrier to entry for developing and experimenting with autoregressive LMMs, making it accessible to a wider range of researchers. Additionally, we conduct qualitative analysis to demonstrate the potential of autoregressive LMMs.
- 4. Rich Resources for Accessibility: To further support the adoption and advancement of autoregressive LMMs, we offer an extensive collection of data resources and detailed tutorials. These materials are designed to facilitate easier onboarding and experimentation for researchers at various levels of expertise.

By addressing these critical aspects, ANOLE represents a significant step forward in democratizing access to advanced multimodal AI technologies. Our work not only builds upon the foundations laid by the original Chameleon model but also paves the way for more inclusive and collaborative research in the field of multimodal AI.

Moreover, ANOLE sparks a series of important and intriguing research questions for the community to explore. For example:

- • Investigating the performance limits of vision generation using unified tokenizer-based multimodal models, in comparison to established methods like diffusion models.
- • Developing efficient techniques for interleaved image-text decoding, which are essential for real-world applications such as textbook and comic generation.

- • Exploring optimal fine-tuning methodologies for these complex pretrained LMMs.
- • Addressing critical issues, including ensuring the safety and ethical use of generated images.

##### 2 Related Works

Models w/o Diffusion? Native? Token-based? Interleaved (open-source)?

MM-Interleaved (Tian et al., 2024) Emu2 (Sun et al., 2024) DreamLLM (Dong et al., 2023) AnyGPT (Zhan et al., 2024) LWM (Liu et al., 2024) Chameleon (Team, 2024)

ANOLE (Ours)

Table 1: Comparison with related works that allows text and (or) vision generation. w/o Diffusion indicates whether the model relies on a diffusion model to generate visual contexts. Native specifies if the model is a native LMM. Token-based denotes whether the model utilizes token-based modeling approach. Interleaved (open-source) represents whether the open-source released version of the model supports multimodal generation.

LMMs have experienced significant advancements, often leveraging pretrained LLMs as their foundation backbone. One common approach involves combining a CLIP-pretrained vision encoder with diffusion models as the decoder, and a pretrained LLM (e.g., Vicuna) as the backbone (Dong et al., 2023; Tian et al., 2024; Sun et al., 2024). This approach achieves impressive results in both image understanding and generation by leveraging the robust representations provided by CLIP and diffusion models. However, incorporating CLIP and diffusion models increases the overall complexity of the model architecture, resulting in additional training overhead and reduced inference efficiency.

In contrast, purely token-based methods exclude diffusion models and CLIP, instead using token-based representations for multimodal understanding and generation. This approach traces back to BEiT (Bao et al., 2021), with OpenAI’s DALL-E (Ramesh et al., 2021) exemplifying text-to-image generation based on similar principles. These methods rely heavily on vector quantization (VQ) models (Van Den Oord et al., 2017; Esser et al., 2021), which combine ResNet-based encoders and decoders with a discrete codebook. During image encoding, the VQ model transforms the image from pixel space to latent space representations, then maps these representations to codebook IDs using a nearest-neighbor search. These IDs serve as input tokens for a Transformer model, which models conditional probabilities and predicts sequences. The VQ decoder then reconstructs images from the generated sequences. This autoregressive, discrete image encoding and decoding approach has been validated in multiple studies for producing high-quality images (Zhu et al., 2024; Yu et al., 2024), effectively modeling inter-image dependencies (Bai et al., 2023), and enhancing image consistency (Pan et al., 2024). LWM (Liu et al., 2024) and Chameleon (Team, 2024) extend this concept to image-text multimodal tasks, using streamlined architectures to handle tasks involving both images and text. Compared to other methods, unified token-based modeling significantly reduces model complexity, facilitating seamless inference and the generation of interleaved image-text sequences without additional components.

ANOLE, building on the foundation of Chameleon, facilitates Chameleon’s image and multimodal generation capabilities with efficient fine-tuning. ANOLE retains the inherent advantages of Chameleon’s architecture while producing high-quality images and maintaining coherent image-text sequences. Tab. 1 highlights ANOLE’s characteristics as an autoregressive, diffusion-free, native token-based model, emphasizing its capability to support multimodal generation, as well as its simplicity and efficiency compared to more complex frameworks.

##### 3 ANOLE

###### 3.1 Overview

ANOLE adopts the same approach and architecture as Chameleon, utilizing an early-fusion, token-based, autoregressive approach to model multimodal sequences (text and images) without the use of diffusion models, relying solely on transformers. Token-based approaches (Team, 2024; Lu et al., 2022; Yu et al., 2023; Liu et al., 2024) achieve modality fusion at the input-token level. Firstly, modality-specific tokenizers tokenize samples from each modality. Then, these token sequences are concatenated to form a single multimodal token sequence, which is subsequently fed into an autoregressive transformer for modeling.

- 3.2 Facilitating the image generation and multimodal generation capabilities from Chameleon

Token Id 4-8195: Image Vocabulary

0-3 4-8197 8198-65536

[Figure 7]

[Figure 8]

[Figure 9]

- 8196: eoi (end of image)
- 8197: boi (begin of image)

Output Head

|[Figure 10]<br><br>Transformer Layers|
|---|

[Figure 11]

[Figure 12]

|[Figure 13]<br><br>Token Embeddings|
|---|

boi Image Tokens eoi

### Chameleon Anole

[Figure 14]

Image Tokenizer

[Figure 15]

Figure 2: Facilitating Chameleon’s image generation capabilities via innovative fine-tuning.

###### 3.2 Facilitating the image generation and multimodal generation capabilities from Chameleon

Fig. 2 shows the innovative fine-tuning process of how ANOLE is fine-tuned from Chameleon. Based on available information and our testings, the latest release of Chameleon have demonstrated strong performance in text understanding, text generation, and multimodal understanding. ANOLE, build on top of Chameleon, aiming to facilitate the image generation and multimodal generation capabilities from Chameleon. Chameleon’s pretraining data natively includes both text and image modalities, theoretically equipping it with image generation capabilities. Our goal is to facilitate this ability without compromising its text understanding, generation, and multimodal comprehension. To achieve this, we froze most of Chameleon’s parameters and fine-tuned only the logits corresponding to image token ids in transformer’s output head layer.

Following the principle of “less is more” (Zhou et al., 2024), the current version of ANOLE, ANOLE-7b-v0.1, was developed using a small amount of image data (5,859 images from LAION-5B art (Schuhmann et al., 2022)) and was fine-tuned on just a few parameters (less than 40M) in a short time (around 30 minutes on 8 A100 GPUs). Despite these limitations, ANOLE-7b-v0.1 expresses impressive image (Tab. 2 and Tab. 3) and multimodal generation capabilities (Fig. 1, Fig. 3 and Fig. 4).

4 Evaluation

We conduct qualitative analysis on ANOLE’s capabilities on image generation and interleaved image-text generation.

###### 4.1 Image Generation

Tab. 2 demonstrates the text-to-image capabilities of ANOLE. We highlight the following points: (1) The images generated by ANOLE are of high quality and closely adhere to the given instructions. For instance, ANOLE accurately captures the essence of ”A steaming cup of coffee next to a fresh croissant on a cozy cafe table,” showcasing the steam, coffee, and croissant elements precisely. (2) ANOLE demonstrates remarkable versatility in generating diverse types of images. It can create realistic depictions, as seen in the coffee and ice cream images, as well as imaginative scenes, like the dinosaur strolling in Times Square. This variety highlights ANOLE’s ability to blend realism with creativity seamlessly.

A serene lakeside view at sunrise with mist rising from the water, surrounded by dense pine forests and mountains in the background.

A bustling downtown street in Tokyo at night, with neon signs, crowded sidewalks, and tall skyscrapers.

A colorful ice cream sundae topped with sprinkles, whipped cream, and a cherry.

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

An impressionist painting of a caf´e scene, with loose brushstrokes and lively colors.

A steaming cup of coffee next to a fresh croissant on a cozy caf´e table.

Under the bright sun, dinosaurs strolling in Times Square, with people taking photos and traffic passing by.

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

Table 2: Text-to-image examples generated by ANOLE.

A piece of paper with word “Anole” written on it.

A piece of paper with word like “Anole” written on it, and a drawing of an Anole.

An image depicting three cubes stacked on a table. Each cube has a unique color and a letter on it.

[Figure 28]

[Figure 29]

[Figure 30]

A vibrant coral reef teeming with colorful fish, sea turtles gliding through the water, and sunlight filtering down from the surface.

A tropical beach with crystal clear water, white sand, and palm trees swaying in the breeze.

A quiet European village with cobblestone streets and colorful houses, under a clear blue sky.

[Figure 31]

[Figure 32]

[Figure 33]

A gothic cathedral with intricate stone carvings and stained glass windows.

A delicious slice of pizza with melting cheese and pepperoni, on a rustic wooden table.

A colorful graffiti mural on an urban wall, with vibrant characters and abstract elements.

[Figure 34]

[Figure 35]

[Figure 36]

Table 3: More text-to-image examples generated by ANOLE.

4.2 Interleaved Image-Text Generation

###### 4.2 Interleaved Image-Text Generation

Fig. 1, Fig. 3 and Fig. 4 demonstrate the multimodal generation capabilities of ANOLE. We emphasize that (1) The generated text is well-organized and provides comprehensive details, which is essential for accurate interleaved image-text generation. This is evident in Fig. 4, where the detailed introduction of Gyumri is accompanied by relevant images, effectively capturing the city’s architectural design and cultural highlights. (2) ANOLE demonstrates seamless integration between images and text, ensuring that the visual and textual elements complement each other perfectly. Fig. 3 highlights this capability by introducing traditional Chinese dishes with corresponding images, making the information both engaging and informative.

[Figure 37]

- Figure 3: An example of ANOLE generating an interleaved image-text sequence on how to cook traditional Chinese cuisines.

##### 5 Conclusion & Future Directions

We introduce ANOLE, an open, autoregressive, native LMM for interleaved image-text generation that demonstrates advanced multimodal generation abilities. ANOLE facilitates image and multimodal generation capabilities from Chameleon by fine-tuning on just 6,000 samples with 40M parameters. We are committed to continually upgrading ANOLE to enhance its capabilities. Our future directions include (1) enhancing ANOLE’s precise instructionfollowing capability, (2) extending its context length, (3) improving its multimodal understanding capabilities, and (4) applying ANOLE to downstream tasks requiring multimodal generation abilities.

##### 6 Limitations & Disclaimer

ANOLE is intended for research use only. Our model weights follow the same license as Chameleon. The fine-tuning images we used are from LAION-5B art, and thus follow the same license as LAION. ANOLE is still under development and has many limitations that need to be addressed. Importantly, we have not aligned the image generation capabilities of the ANOLE to ensure safety and harmlessness. Therefore, we encourage users to interact with ANOLE with caution and report any concerning behaviors to help improve the model’s safety and ethical considerations.

##### 7 Acknowledgements

We sincerely thank the Meta Chameleon Team for open-sourcing Chameleon. Our base model and the majority of our inference code are based on it. We also greatly appreciate all the contributors who have participated in pull request #31534 submitted to the transformers repository. This PR has been crucial for the development of our training code.

|Prompt: Please introduce the city of Gyumri with pictures.<br><br>[Figure 38]<br><br>Response: Gyumri is the second-largest city in Armenia and has a rich history, beautiful architecture, and a vibrant cultural scene. It's known for its colorful houses, historic landmarks, and stunning natural surroundings, which make it an ideal destination for both cultural and<br><br>outdoor enthusiasts.<br><br>The city center is home to several museums, theaters, and historical sites, including the Gyumri Opera and Ballet Theater, the Gyumri Museum of Local Lore, and the Mother Cathedral of the 3rd century, shown in the image below: <img><br><br>Located in the Caucasus region, Gyumri is surrounded by picturesque hills and mountains, including Mount Aragats, the highest peak in Armenia, pictured here: <img><br><br>In addition to its cultural attractions, Gyumri is also known for its<br><br>delicious food and traditional handicrafts. Visitors can sample local specialties like sujuk (cured sausage), khachapuri (bread with cheese and eggs), and rakhi (a type of flatbread), and browse local markets for unique souvenirs like carpets, rugs, and textiles.<br><br>[Figure 39]|
|---|

|Prompt: What color is a polar bear’s fur? Show me a photograph of the polar bear in the wild.<br><br>[Figure 40]<br><br>Response: <img> A polar bear’s fur is pure white and is very thick which helps to insulate it against the cold. The image above shows a polar bear in its natural habitat. Polar bears are carnivores, which means they feed on meat. The diet of a polar bear consists mainly of seals. They can dive into water and swim after<br><br>seals or they wait patiently for them to come to shore. They also eat birds,<br><br>fish, and other small mammals. Polar bears are magnificent creatures but unfortunately they are considered vulnerable and even endangered by some due to human activities such as climate change.|
|---|

###### Figure 4: Examples of ANOLE generating interleaved text-and-image sequences on geographical and biological topics.

##### References

- [1] Yutong Bai, Xinyang Geng, Karttikeya Mangalam, Amir Bar, Alan Yuille, Trevor Darrell, Jitendra Malik, and Alexei A Efros. 2023. Sequential modeling enables scalable learning for large vision models. arXiv preprint arXiv:2312.00785.
- [2] Hangbo Bao, Li Dong, and Furu Wei. 2021. Beit: Bert pre-training of image transformers. arXiv preprint arXiv:2106.08254.
- [3] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. 2023. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality.
- [4] Runpei Dong, Chunrui Han, Yuang Peng, Zekun Qi, Zheng Ge, Jinrong Yang, Liang Zhao, Jianjian Sun, Hongyu Zhou, Haoran Wei, et al. 2023. Dreamllm: Synergistic multimodal comprehension and creation. arXiv preprint arXiv:2309.11499.
- [5] Patrick Esser, Robin Rombach, and Bjorn Ommer. 2021. Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12873–12883.
- [6] Dirk Groeneveld, Iz Beltagy, Pete Walsh, Akshita Bhagia, Rodney Kinney, Oyvind Tafjord, Ananya Harsh Jha, Hamish Ivison, Ian Magnusson, Yizhong Wang, et al. 2024. Olmo: Accelerating the science of language models. arXiv preprint arXiv:2402.00838.
- [7] Suriya Gunasekar, Yi Zhang, Jyoti Aneja, Caio C´esar Teodoro Mendes, Allie Del Giorno, Sivakanth Gopi, Mojan Javaheripi, Piero Kauffmann, Gustavo de Rosa, Olli Saarikivi, et al. 2023. Textbooks are all you need. arXiv preprint arXiv:2306.11644.
- [8] Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. 2023. Mistral 7b. arXiv preprint arXiv:2310.06825.
- [9] Hao Liu, Wilson Yan, Matei Zaharia, and Pieter Abbeel. 2024. World model on million-length video and language with ringattention. arXiv preprint arXiv:2402.08268.
- [10] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. 2023a. Improved baselines with visual instruction tuning.
- [11] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023b. Visual instruction tuning. In NeurIPS.
- [12] Zhengzhong Liu, Aurick Qiao, Willie Neiswanger, Hongyi Wang, Bowen Tan, Tianhua Tao, Junbo Li, Yuqi Wang, Suqi Sun, Omkar Pangarkar, et al. 2023c. Llm360: Towards fully transparent open-source llms. arXiv preprint arXiv:2312.06550.
- [13] Jiasen Lu, Christopher Clark, Rowan Zellers, Roozbeh Mottaghi, and Aniruddha Kembhavi. 2022. Unified-io: A unified model for vision, language, and multi-modal tasks. In The Eleventh International Conference on Learning Representations.
- [14] Xichen Pan, Pengda Qin, Yuhong Li, Hui Xue, and Wenhu Chen. 2024. Synthesizing coherent story with auto-regressive latent diffusion models. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 2920–2930.
- [15] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. 2021. Zero-shot text-to-image generation. In International conference on machine learning, pages 8821–8831. Pmlr.
- [16] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. 2022. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in Neural Information Processing Systems, 35:25278–25294.
- [17] Quan Sun, Yufeng Cui, Xiaosong Zhang, Fan Zhang, Qiying Yu, Yueze Wang, Yongming Rao, Jingjing Liu, Tiejun Huang, and Xinlong Wang. 2024. Generative multimodal models are in-context learners. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14398–14409.
- [18] Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B Hashimoto. 2023. Stanford alpaca: An instruction-following llama model.

- [19] Chameleon Team. 2024. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818.
- [20] Gemma Team, Thomas Mesnard, Cassidy Hardin, Robert Dadashi, Surya Bhupatiraju, Shreya Pathak, Laurent Sifre, Morgane Rivi`ere, Mihir Sanjay Kale, Juliette Love, et al. 2024. Gemma: Open models based on gemini research and technology. arXiv preprint arXiv:2403.08295.
- [21] Changyao Tian, Xizhou Zhu, Yuwen Xiong, Weiyun Wang, Zhe Chen, Wenhai Wang, Yuntao Chen, Lewei Lu, Tong Lu, Jie Zhou, et al. 2024. Mm-interleaved: Interleaved image-text generative modeling via multi-modal feature synchronizer. arXiv preprint arXiv:2401.10208.
- [22] Shengbang Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Manoj Middepogu, Sai Charitha Akula, Jihan Yang, Shusheng Yang, Adithya Iyer, Xichen Pan, et al. 2024. Cambrian-1: A fully open, vision-centric exploration of multimodal llms. arXiv preprint arXiv:2406.16860.
- [23] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. 2023. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971.
- [24] Aaron Van Den Oord, Oriol Vinyals, et al. 2017. Neural discrete representation learning. Advances in neural information processing systems, 30.
- [25] Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Xixuan Song, Jiazheng Xu, Bin Xu, Juanzi Li, Yuxiao Dong, Ming Ding, and Jie Tang. 2023. Cogvlm: Visual expert for pretrained language models.
- [26] Lili Yu, Bowen Shi, Ramakanth Pasunuru, Benjamin Muller, Olga Golovneva, Tianlu Wang, Arun Babu, Binh Tang, Brian Karrer, Shelly Sheynin, et al. 2023. Scaling autoregressive multi-modal models: Pretraining and instruction tuning. arXiv preprint arXiv:2309.02591, 2(3).
- [27] Qihang Yu, Mark Weber, Xueqing Deng, Xiaohui Shen, Daniel Cremers, and Liang-Chieh Chen. 2024. An image is worth 32 tokens for reconstruction and generation. arXiv preprint arXiv:2406.07550.
- [28] Jun Zhan, Junqi Dai, Jiasheng Ye, Yunhua Zhou, Dong Zhang, Zhigeng Liu, Xin Zhang, Ruibin Yuan, Ge Zhang, Linyang Li, et al. 2024. Anygpt: Unified multimodal llm with discrete sequence modeling. arXiv preprint arXiv:2402.12226.
- [29] Chunting Zhou, Pengfei Liu, Puxin Xu, Srinivasan Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping Yu, Lili Yu, et al. 2024. Lima: Less is more for alignment. Advances in Neural Information Processing Systems, 36.
- [30] Lei Zhu, Fangyun Wei, Yanye Lu, and Dong Chen. 2024. Scaling the codebook size of vqgan to 100,000 with a utilization rate of 99%. arXiv preprint arXiv:2406.11837.

