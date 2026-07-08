# arXiv:2401.14688v3[cs.CL]18Jun2024

TAIYI-DIFFUSION-XL: ADVANCING BILINGUAL TEXT-TO-IMAGE GENERATION WITH LARGE VISION-LANGUAGE MODEL SUPPORT

Xiaojun Wu♥∗ Dixiang Zhang♥♦∗ Ruyi Gan♥♣† Junyu Lu♥ Ziwei Wu♥ Renliang Sun♥

Jiaxing Zhang♥‡ Pingjian Zhang♦‡ Yan Song♣‡ ♥International Digital Economy Academy ♣South China University of Technology

♦University of Science and Technology of China

{wuxiaojun, zhangdixiang, ganruyi, lujunyu, wuziwei, sunrenliang}@idea.edu.cn zhangjiaxing@idea.edu.cn pjzhang@scut.edu.cn clksong@gmail.com

ABSTRACT

Recent advancements in text-to-image models have significantly enhanced image generation capabilities, yet a notable gap of open-source models persists in bilingual or Chinese language support. To address this need, we present Taiyi-Diffusion-XL, a new Chinese and English bilingual text-to-image model which is developed by extending the capabilities of CLIP and Stable-DiffusionXL through a process of bilingual continuous pre-training. This approach includes the efficient expansion of vocabulary by integrating the most frequently used Chinese characters into CLIP’s tokenizer and embedding layers, coupled with an absolute position encoding expansion. Additionally, we enrich text prompts by large vision-language model, leading to better images captions and possess higher visual quality. These enhancements are subsequently applied to downstream text-to-image models. Our empirical results indicate that the developed CLIP model excels in bilingual image-text retrieval. Furthermore, the bilingual image generation capabilities of Taiyi-Diffusion-XL surpass previous models. This research leads to the development and open-sourcing of the Taiyi-Diffusion-XL model, representing a notable advancement in the field of image generation, particularly for Chinese language applications. The model and demonstration are made publicly available at https://huggingface. co/IDEA-CCNL/Taiyi-Stable-Diffusion-XL-3.5B/, fostering further research and collaboration in this domain.

1 INTRODUCTION

Recent developments in diffusion models, such as those presented in Stable Diffusion(SD) (Rombach et al., 2022; Podell et al., 2023),DALL-E (Ramesh et al., 2022; Betker et al., 2023), Imagen (Saharia et al., 2022), and Deepfloyd-IF (Shonenkov et al., 2023), have showcased their potential in generating high-quality images from text descriptions. However, it is important to note that the majority of current open-source text-to-image models predominantly support English, with very few offering bilingual support for both Chinese and English. This advancement diverges from the conventional methodology of employing translation software to convert Chinese text into English for subsequent image generation with English-centric models. In particular, works such as TaiyiDiffusion (Zhang et al., 2022), Pai-Diffusion (Wang et al., 2023) and Alt-Diffusion(Ye et al., 2023) have made significant strides in adapting text-to-image models for Chinese scenarios, demonstrating the feasibility and importance of native language support in such models. Such models adeptly handle language-specific expressions, ensuring the preservation of original meaning and emotional nuances that might otherwise be lost in translation process. These models often obtain Chinese

*Equal Contribution. †Project Leader. ‡Corresponding Author.

understanding capabilities by replacing multi-language text encoders (Radford et al., 2021; Devlin et al., 2019) and retaining unet(Ronneberger et al., 2015) while this methodology will discard the original English understanding capabilities.

[Figure 1]

- Figure 1: An illustration of Taiyi-XL showcasing text-to-image generation results under various styles and prompts.

Building on these advancements, our work, Taiyi-Diffusion-XL(Taiyi-XL), specifically focuses on augmenting these models for Chinese text-to-image generation while preserving the original English ability, addressing the unique linguistic and cultural aspects of the bilingual language. In summary, while translation tools offer a certain level of convenience for cross-language applications, native language support in models, especially for languages like Chinese, provides distinct advantages in terms of comprehension, accuracy, and efficiency. Our contributions are aimed at enhancing these capabilities, thereby offering more effective and inclusive tools for the research community. Our research contributes to this evolving field in three significant ways:

- • Efficient Algorithms for Bilingual Expansion: We develop algorithms for expanding vocabulary and position encoding in text-to-image models tailored for bilingual contexts. This advancement facilitates more accurate and culturally tuned image generation.
- • Enrichment of Text Prompts by Large Vision-Language Models: We employ large visionlanguage models to enrich text prompts. This approach marks a substantial enhancement in the model’s ability to interpret and visualize complex textual descriptions.
- • Creation of Bilingual Models: Utilizing the capabilities of multimodal foundation model, we develop and open-source the text-to-image model, Taiyi-XL, which significantly advances the research and application of bilingual text-to-image models.

2 METHODOLOGY

Our methodology for text-to-image generation, especially with diffusion models, encompasses two primary phases, focusing on dataset preparation and model training.

- 2.1 DATASET PREPARATION

We curate a dataset consisting of high-quality image-text pairs (X,Y ), where X represents an image, and Y is a descriptive text. In contrast to traditional datasets with discretized tags, our dataset emphasizes comprehensive descriptions, capturing materials, styles, colors, and spatial layouts. To address the limitations of web-crawled resources, which often contain irrelevant or inaccurate tags, we employ vision-language large models (Lu et al., 2023b;a) to generate synthetic captions that more accurately describe the images, which inherits the language capabilities of the bilingual large language model (Gan et al., 2023) and expands the visual capabilities of LLMs. This approach not only enhances the richness of the dataset but also ensures a higher degree of relevance and detail in the descriptions. We use images, web crawl caption, and instructions for generating description as inputs for the Lyrics (Lu et al., 2023a). In Chinese, we select “请详细描述图片内容。” as the instruction, and in English, we select “Write a detailed description of the given image.” as the instruction. The Lyrics model generates new, accurate descriptive text by extracting features from the images as well as distilling useful information from inaccurate and imperfect web crawl captions. Finally, we combine the generated high-quality text with the original images to form image-text pairs, which are then input into the Taiyi-XL for training.

- 2.2 CLIP TRAINING

The foundation of our model is a vision-language large model, similar to CLIP (Radford et al., 2021), which aligns images and text representations effectively. We start with the pre-trained Englishonly CLIP model and extend its training to accommodate bilingual adaptation and the nuanced requirements of high-quality image-text data. The first stage of training involves processing a largescale, bilingual dataset, including Laion (Schuhmann et al., 2021) and Wukong (Gu et al., 2022), with a focus on data cleaning and quality enhancement. We employ a contrastive loss function and a distributed, memory-efficient training approach (Chen et al., 2023). The second stage continues with training on our enriched dataset, emphasizing the diverse perspectives and details captured in high-quality image-text pairs.

- 2.3 TAIYI-XL TRAINING

The Taiyi-XL training process, a key component in our text-to-image generation methodology, especially with diffusion models, involves two primary phases:

Initialization and Training. We initialize the Taiyi-XL model, denoted as Gθ, with components including a noise predictor ϵθ, a CLIP text encoder τθ from 2.2, a latent encoder E, and a dataset D. Each data instance in D is represented as a pair (xi,yi), where xi is an image and yi is its corresponding textual descriptor. For the training phase at mix resolution of 512 × 512 and 1024 × 1024, we define a loss function L to guide the image denoising process:

## L(θ) := EE(x),y,ϵ∼N(0,1),t ∥ϵ − ϵθ(zt,t,τθ(y))∥22 , (1)

The model is conceptualized as a sequence of denoising latent autoencoders ϵθ(zt,t); t = 1...T, implemented as a time-conditional UNet (Ronneberger et al., 2015). The latent representations zt are efficiently obtained from E(x) during training and decoded to the image space using a VAE decoder (Kingma & Welling, 2013). The text encoder τθ, parameterized as a transformer model, is optimized jointly with ϵθ as per Eq. 1.

Model parameters θ are iteratively updated using gradient descent to minimize the loss function L(θ,e):

θe+1 = θe − η · ∇θL(θe,e) (2) where η represents the learning rate.

Text-to-Image Generation. For text-to-image generation, we utilize the trained bilingual text encoder for extracting features from textual descriptions. The extracted textual features τθ(y) are then integrated into the latent diffusion process, enhancing computational efficiency and reducing processing time and memory requirements. In the generation phase, starting from the last time step

[Figure 2]

[Figure 3]

#### Data Generation via VLM

#### Vision-Language Alignment

Text Encoder

Prompt Web-Crawled

Caption T1 T2 ... Tn

[Figure 4]

The image features a woman in a car, leaning out of the window and holding a blue water bottle. She appears to be enjoying the moment and is likely taking a break from

...

.............

..................

- I1

- I2

I1·T1

[Figure 5]

...

I2·T2

Image Encoder

Lyrics VLM her journey.

...........

...

...

.....................

In

In·Tn

[Figure 6]

512x512

Latent Diffusion Module

[Figure 7]

[Figure 8]

Add Text Condition

Diffusion Process

1024x1024

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

Denoise Process

Multi Resolution and Aspect Ratio Learning

- Figure 2: Overview of the Taiyi-Diffusion-XL(Taiyi-XL) training process, encompassing data preprocessing, image-text contrastive learning and multi-resolution denoising training process.

T with pure noise, the model iteratively denoises the input, converging to x0, the clean image, as described by:

xt−1 = xt − ϵθ(xt,t,τθ(y)), lim t→0

xt = x0 (3)

- 3 EXPERIMENT AND ANALYSIS

Training Settings. We base our Taiyi-XL model on the pre-trained Stable Diffusion XL (SD-XL) (Podell et al., 2023) checkpoint, providing a strong foundation for image generation. To enhance efficiency and manage GPU memory use, we adopt the BFLOAT16 format. Our training approach involves a learning rate of 1e-5, starting with a warmup phase for stable learning, followed by a cosine decay schedule to fine-tune and refine the model. These strategies are essential for balancing training speed with model performance.

Evaluation Protocols. Our evaluation framework encompasses both machine and human evaluation to provide a comprehensive understanding of the model’s performance. Machine evaluation metrics include CLIP performance evaluation with image-to-text retrieval and text-to-image retrieval; CLIP Similarity (CLIP Sim), which measures the semantic alignment between the generated images and text descriptions; Inception Score (IS), assessing the quality and diversity of the images; and Fr´echet Inception Distance (FID), evaluating the distance between the distributions of generated and real images. In the context of human evaluation of text-to-image generation, it is acknowledged that such assessments inherently possess a degree of subjectivity. Consequently, this study primarily employs a case analysis approach to discern and articulate the distinct characteristics of image generation outcomes produced by different models. Rather than providing direct quantitative results that delineate superiority or inferiority among the models, the focus is on a qualitative examination that highlights the unique attributes and performance nuances of each model in image generation tasks.

Baselines. For our comparative analysis, we include several established models as baselines: SDXL (Podell et al., 2023), Midjourney1, DALL-E 32 (Betker et al., 2023), along with other opensourced models such as our previsous work Taiyi-v0.1(Wang et al., 2022), Alt-Diffusion(Ye et al., 2023) and Pai-Diffusion(Wang et al., 2023). DALL-E 3, recognized for its innovative text-to-image capabilities, sets a high standard in generating quality images from text descriptions. SD-XL, a

- 1https://www.midjourney.com/
- 2https://openai.com/dall-e-3

variant of the Stable Diffusion model, excels in complex image synthesis tasks. By comparing Taiyi-XL with these models, we aim to showcase the advancements and efficacy of our approach, particularly in bilingual image generation and fidelity to textual prompts.

- 3.1 MACHINE EVALUATION

CLIP Model Evaluation. Our CLIP model’s performance is exemplary on both English and Chinese datasets, as evidenced by the zero-shot image-text retrieval results. The original CLIP model (Radford et al., 2021), while establishing a foundational understanding, exhibits modest retrieval rates on the Flickr (Young et al., 2014) and MSCOCO datasets (Lin et al., 2014). This outcome highlights the inherent challenges associated with cross-lingual transfer learning. In contrast, AltCLIP (Chen et al., 2022) and our enhanced CLIP model demonstrate significant improvements, with our model achieving the highest recall rates across most evaluation metrics. Particularly noteworthy is our model’s performance in the Text → Image retrieval task on the Flickr-CN (Young et al., 2014) and MSCOCO-CN datasets (Li et al., 2019), where it attains recall rates of 88.1% and 69.7% at R@1, respectively. These results indicate a robust alignment between textual prompts and visual content, underscoring the effectiveness of our tailored modifications in enhancing CLIP’s crosslingual performance. The results, presented in Table 1, demonstrate the potential of specialized models in handling diverse linguistic contexts within multimodal AI applications. The superior performance of our CLIP model, particularly in bilingual contexts, significantly bolsters the capabilities of the Taiyi-XL model. This enhancement allows for a more nuanced understanding of user-input prompts, leading to the generation of images that more accurately reflect the given prompts. The results affirm the importance of developing robust bilingual comprehension capabilities in models for advanced multimodal applications.

Flickr30K MSCOCO Image → Text Text → Image Image → Text Text → Image

Model R@1 R@5 R@10 R@1 R@5 R@10 R@1 R@5 R@10 R@1 R@5 R@10

CLIP (Radford et al., 2021) 85.1 97.3 99.2 65.0 87.1 92.2 56.4 79.5 86.5 36.5 61.1 71.1 AltCLIP (Chen et al., 2022) 86.0 98.0 99.1 72.5 91.6 95.4 58.6 80.6 87.8 42.9 68.0 77.4 Our-CLIP 88.4 98.8 99.9 75.7 93.8 96.9 61.2 84.8 90.3 49.2 70.3 79.6

Flickr30K-CN MSCOCO-CN Image → Text Text → Image Image → Text Text → Image

CLIP (Radford et al., 2021) 2.3 8.1 12.6 0 2.4 4.0 0.6 4.1 7.1 1.8 6.7 11.9 AltCLIP (Chen et al., 2022) 69.8 89.9 94.7 84.8 97.4 98.8 63.9 87.2 93.9 62.8 88.8 95.5 Our-CLIP 73.2 90.3 96.5 88.1 98.2 99.1 66.0 91.1 96.6 69.7 91.3 96.8

- Table 1: Zero-shot image-text retrieval results on Flickr30K, MSCOCO, Flickr30K-CN, and MSCOCO-CN datasets. The best results are marked in bold.

Diffusion Model Evaluation. Based on the data presented in Table 2, a comprehensive analysis of the performance of various models in bilingual image generation tasks reveals significant insights. The evaluation metrics used for this analysis include CLIP Similarity (CLIP Sim), Inception Score (IS), and Fr´echet Inception Distance (FID), which collectively offer a robust assessment of model performance in terms of image quality, diversity, and alignment with textual descriptions. In the English dataset (COCO), our Taiyi-XL model demonstrates superior performance across all metrics, notably achieving the highest CLIP Sim score, the highest IS, and the most favorable FID. These results indicate that Taiyi-XL not only generates images that are closely aligned with the given text prompts but also ensures high image quality and diversity. The model outperforms other contenders such as Alt-Diffusion, SD-v1.5, and SD-XL, highlighting its effectiveness in handling English language prompts in image generation tasks. Similarly, in the Chinese dataset (COCO-CN), Taiyi-XL again stands out, achieving the best results with a CLIP Sim score, IS and FID. Compared to other models like Taiyi-v0.1, Alt-Diffusion, and Pai-Diffusion, Taiyi-XL exhibits a remarkable ability to generate high-quality images that are well-aligned with Chinese textual descriptions. This performance underscores the model’s robust bilingual capabilities, making it particularly suitable for applications requiring high-fidelity image generation from diverse linguistic inputs.

Overall, the results from both datasets affirm the efficacy of the Taiyi-XL model in bilingual image generation tasks. Its ability to consistently produce high-quality, diverse images that accurately reflect the content of both English and Chinese text prompts positions it as a leading solution in

Model CLIP Sim(↑) FID(↓) IS(↑) English Dataset (COCO)

Alt-Diffusion(Ye et al., 2023) 0.220 27.600 31.577 SD-v1.5(Rombach et al., 2022) 0.225 25.342 32.876 SD-XL(Podell et al., 2023) 0.231 23.887 33.793 Taiyi-XL 0.254 22.543 35.465

Chinese Dataset (COCO-CN)

Taiyi-v0.1(Wang et al., 2022) 0.197 69.226 21.060 Alt-Diffusion(Ye et al., 2023) 0.220 68.488 22.126 Pai-Diffusion(Wang et al., 2023) 0.196 72.572 19.145 Taiyi-XL 0.225 67.675 22.965

- Table 2: Comparison of different models based on CLIP Sim, IS, and FID across English (COCO) and Chinese (COCO-CN) datasets. The best results are marked in bold.

the field of multimodal AI applications. The superior performance of Taiyi-XL in these bilingual contexts highlights the potential of specialized models in navigating the complexities of different linguistic environments within image generation tasks.

- 3.2 HUMAN PREFERENCE EVALUATION

In our comprehensive analysis, as depicted in Figures 3 and 4 showcasing the performance of various models in Chinese and English text-to-image generation, several key observations and conclusions have emerged. The XL versions of the models such as SD-XL and Taiyi-XL exhibit a significant improvement over the 1.5 versions such as SD-v1.5 and Alt-Diffusion, indicating advancements in the scale of model parameters, underlying algorithms and training methodologies. DALL-E 3, while occasionally producing overly vivid colors, stands out for its exceptional prompt-following capability, setting a high benchmark in generating images that closely align with the given textual descriptions. Our model, characterized by a photographic style, closely parallels the performance of Midjourney, particularly in its aesthetic appeal. However, a notable distinction lies in our model’s enhanced support for bilingual (Chinese and English) text-to-image generation, a feature that is especially valuable in diverse linguistic contexts. This capability underscores the importance of language versatility in the realm of generative models.

The final conclusion drawn from this analysis is that while our model may not yet match the performance of commercial models, it significantly surpasses current bilingual open-source models. We attribute the gap with commercial models primarily to differences in the quantity, quality, and diversity of the image-text data used for training. Our model has been trained exclusively on copyrightcompliant image-text data, highlighting the ongoing challenge of copyright issues in text-to-image and AI-generated content (AIGC) models. This aspect remains a critical factor in the development and refinement of generative models, underscoring the need for access to diverse and high-quality datasets while navigating the complexities of copyright constraints.

We also evaluated the impact of employing Latent Consistency Models (LCM) (Song et al., 2023; Luo et al., 2023a;b) to accelerate the image generation process. A notable observation 5 from these tests is the correlation between the reduction in inference steps and the consequent decline in image quality. Specifically, when the generation is constrained to a single step, the resulting images predominantly exhibit only basic outlines and lack finer details. However, extending the generation process to 8 steps ensures a considerably higher quality of the generated images. This finding suggests that while LCM can effectively speed up the generation process, a balance must be struck between the number of steps and the desired image quality. Maintaining a minimum number of steps, such as eight in our tests, appears to be crucial for preserving a satisfactory level of detail and overall image fidelity.

[Figure 14]

- Figure 3: Comparison of Different Models in Chinese Text-to-Image Generation Performance.

[Figure 15]

rney

- Figure 4: Comparison of Different Models in English Text-to-Image Generation Performance.

- 4 RELATED WORK

- 4.1 ADVANCEMENTS IN IMAGE GENERATION AND DIFFUSION MODELS

Recent years have seen substantial advancements in the field of text-to-image generation. This work diverges from traditional approaches such as Generative Adversarial Networks (GANs) (Goodfellow et al., 2014; Arjovsky et al., 2017), Variational Autoencoders (VAEs) (Kingma & Welling, 2013), Flow-based models (Rezende & Mohamed, 2015), and autoregressive models (Ramesh et al., 2021; Ding et al., 2021; 2022), focusing instead on the more advanced diffusion model. The evolution and refinement of diffusion theory and techniques (Vincent, 2011; Ho et al., 2020; Song et al., 2020; Cao et al., 2022) have positioned the diffusion model as a leading technology in image generation. Noteworthy developments in this area include Dall-E 2 (Ramesh et al., 2022), which utilizes a hier-

[Figure 16]

[Figure 17]

Figure 5: Taiyi-XL generation examples with Latent Consistency Model

archical approach for generating images based on textual descriptions with CLIP latents. Similarly, Imagen (Saharia et al., 2022) and Deepfloyd-IF (Shonenkov et al., 2023) demonstrate the capability of diffusion models to produce photorealistic images from text, emphasizing deep language understanding. The latent diffusion model (Rombach et al., 2022), encompassing works such as stable-diffusion-v1-5, stable-diffusion-2-1, and stable-diffusion-xl (Podell et al., 2023), represents the forefront of this technology. These models primarily leverage the CLIP text model for textual feature extraction, integrating these features into the latent diffusion process to reduce computational overhead and memory requirements.

- 4.2 TEXT-TO-IMAGE MODELS IN BILINGUAL CONTEXT

In response to the requirements of text-to-image generation in bilingual scenarios, especially in Chinese language, researchers have made significant contributions. initially, the CLIP text encoder is substituted with a Chinese-specific encoder, followed by pre-training for text-image matching on Chinese datasets. Key works in this domain include Taiyi-CLIP (Zhang et al., 2022), Chinese-CLIP (Yang et al., 2022), and Alt-CLIP (Chen et al., 2022). Subsequently, the text encoder in stable diffusion is replaced, and further training on Chinese text-image datasets is conducted to enhance text-to-image generation capabilities. This leads to the development of Chinese versions of diffusion image generation models, such as Taiyi-diffusion (Zhang et al., 2022), Alt-diffusion (Ye et al., 2023) and Pai-diffusion(Wang et al., 2023). However, it is noteworthy that replacing the CLIP text encoder can result in the loss of English language capabilities in the model, and the training process can be resource-intensive.

- 4.3 THE ROLE OF TEXT-IMAGE DATASETS

Datasets are pivotal in both text-image matching and text-to-image generation. Traditional image caption datasets like COCO (Lin et al., 2014) and Flickr (Young et al., 2014) in English, and COCOCN (Li et al., 2019) and Flickr-CN (Li et al., 2016) in Chinese, provide a foundational training base but are limited in size, generally below one million entries. Consequently, web-crawled datasets such as Laion(Schuhmann et al., 2021) (primarily in English) and Wukong(Gu et al., 2022) (primarily in Chinese) have emerged as more critical data sources for training diffusion text-to-image models, boasting sizes of up to 100 million or even 5 billion.

- 5 CONCLUSION

Our research demonstrates the profound impact of integrating bilingual support into text-to-image models, significantly advancing multimodal research in Chinese contexts. The development of Taiyi-CLIP and Taiyi-XL models, with their expanded vocabulary and position encoding, marks a notable advancement in image-text retrieval and image generation. These models lay the foundation

for future innovations in bilingual multimodal studies. Additionally, the use of large vision-language models to enrich text prompts has led to more accurate and detailed image generation, aligning closely with user intent. This approach underscores the importance of accurate and complex language understanding in text-to-image generation. As we continue to make our findings and models open-sourced, we invite collaboration and further exploration, contributing to a more inclusive and linguistically diverse future in artificial intelligence research.

REFERENCES

Martin Arjovsky, Soumith Chintala, and L´eon Bottou. Wasserstein generative adversarial networks. In International conference on machine learning, pp. 214–223. PMLR, 2017.

James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, Wesam Manassra, Prafulla Dhariwal, Casey Chu, Yunxin Jiao, and Aditya Ramesh. Improving image generation with better captions. openai cdn.openai.com/papers/dall-e-3.pdf, 2023.

Hanqun Cao, Cheng Tan, Zhangyang Gao, Guangyong Chen, Pheng-Ann Heng, and Stan Z Li. A survey on generative diffusion model. arXiv preprint arXiv:2209.02646, 2022.

Yihao Chen, Xianbiao Qi, Jianan Wang, and Lei Zhang. Disco-clip: A distributed contrastive loss for memory efficient clip training. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 22648–22657, 2023.

Zhongzhi Chen, Guang Liu, Bo-Wen Zhang, Fulong Ye, Qinghong Yang, and Ledell Wu. Altclip: Altering the language encoder in clip for extended language capabilities. arXiv preprint arXiv:2211.06679, 2022.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pp. 4171–4186, 2019.

Ming Ding, Zhuoyi Yang, Wenyi Hong, Wendi Zheng, Chang Zhou, Da Yin, Junyang Lin, Xu Zou, Zhou Shao, Hongxia Yang, et al. Cogview: Mastering text-to-image generation via transformers. Advances in Neural Information Processing Systems, 34:19822–19835, 2021.

Ming Ding, Wendi Zheng, Wenyi Hong, and Jie Tang. Cogview2: Faster and better text-to-image generation via hierarchical transformers. Advances in Neural Information Processing Systems, 35:16890–16902, 2022.

Ruyi Gan, Ziwei Wu, Renliang Sun, Junyu Lu, Xiaojun Wu, Dixiang Zhang, Kunhao Pan, Ping Yang, Qi Yang, Jiaxing Zhang, et al. Ziya2: Data-centric learning is all llms need. arXiv preprint arXiv:2311.03301, 2023.

Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. Advances in neural information processing systems, 27, 2014.

Jiaxi Gu, Xiaojun Meng, Guansong Lu, Lu Hou, Niu Minzhe, Xiaodan Liang, Lewei Yao, Runhui Huang, Wei Zhang, Xin Jiang, et al. Wukong: A 100 million large-scale chinese cross-modal pre-training benchmark. Advances in Neural Information Processing Systems, 35:26418–26431, 2022.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013.

Xirong Li, Weiyu Lan, Jianfeng Dong, and Hailong Liu. Adding chinese captions to images. In Proceedings of the 2016 ACM on international conference on multimedia retrieval, pp. 271–275, 2016.

Xirong Li, Chaoxi Xu, Xiaoxu Wang, Weiyu Lan, Zhengxiong Jia, Gang Yang, and Jieping Xu. Coco-cn for cross-lingual image tagging, captioning, and retrieval. IEEE Transactions on Multimedia, 21(9):2347–2360, 2019.

Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13, pp. 740–755. Springer, 2014.

Junyu Lu, Ruyi Gan, Dixiang Zhang, Xiaojun Wu, Ziwei Wu, Renliang Sun, Jiaxing Zhang, Pingjian Zhang, and Yan Song. Lyrics: Boosting fine-grained language-vision alignment and comprehension via semantic-aware visual objects. arXiv preprint arXiv:2312.05278, 2023a.

Junyu Lu, Dixiang Zhang, Xiaojun Wu, Xinyu Gao, Ruyi Gan, Jiaxing Zhang, Yan Song, and Pingjian Zhang. Ziya-vl: Bilingual large vision-language model via multi-task instruction tuning. arXiv preprint arXiv:2310.08166, 2023b.

Simian Luo, Yiqin Tan, Longbo Huang, Jian Li, and Hang Zhao. Latent consistency models: Synthesizing high-resolution images with few-step inference, 2023a.

Simian Luo, Yiqin Tan, Suraj Patil, Daniel Gu, Patrick von Platen, Apolin´ario Passos, Longbo Huang, Jian Li, and Hang Zhao. Lcm-lora: A universal stable-diffusion acceleration module. arXiv preprint arXiv:2311.05556, 2023b.

Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748–8763. PMLR, 2021.

Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In International Conference on Machine Learning, pp. 8821–8831. PMLR, 2021.

Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical textconditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1(2):3, 2022.

Danilo Rezende and Shakir Mohamed. Variational inference with normalizing flows. In International conference on machine learning, pp. 1530–1538. PMLR, 2015.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 10684–10695, June 2022.

Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In Medical Image Computing and Computer-Assisted Intervention– MICCAI 2015: 18th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III 18, pp. 234–241. Springer, 2015.

Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in Neural Information Processing Systems, 35:36479–36494, 2022.

Christoph Schuhmann, Richard Vencu, Romain Beaumont, Robert Kaczmarczyk, Clayton Mullis, Aarush Katta, Theo Coombes, Jenia Jitsev, and Aran Komatsuzaki. Laion-400m: Open dataset of clip-filtered 400 million image-text pairs. arXiv preprint arXiv:2111.02114, 2021.

Alex Shonenkov, Misha Konstantinov, Daria Bakshandaeva, Christoph Schuhmann, Ksenia Ivanova, and Nadiia Klokova. If: Title of the repository, 2023.

Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv

preprint arXiv:2010.02502, 2020. Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. 2023. Pascal Vincent. A connection between score matching and denoising autoencoders. Neural compu-

tation, 23(7):1661–1674, 2011.

Chengyu Wang, Zhongjie Duan, Bingyan Liu, Xinyi Zou, Cen Chen, Kui Jia, and Jun Huang. Paidiffusion: Constructing and serving a family of open chinese diffusion models for text-to-image synthesis on the cloud. arXiv preprint arXiv:2309.05534, 2023.

Junjie Wang, Yuxiang Zhang, Lin Zhang, Ping Yang, Xinyu Gao, Ziwei Wu, Xiaoqun Dong, Junqing He, Jianheng Zhuo, Qi Yang, Yongfeng Huang, Xiayu Li, Yanghan Wu, Junyu Lu, Xinyu Zhu, Weifeng Chen, Ting Han, Kunhao Pan, Rui Wang, Hao Wang, Xiaojun Wu, Zhongshen Zeng, Chongpei Chen, Ruyi Gan, and Jiaxing Zhang. Fengshenbang 1.0: Being the foundation of chinese cognitive intelligence. CoRR, abs/2209.02970, 2022.

An Yang, Junshu Pan, Junyang Lin, Rui Men, Yichang Zhang, Jingren Zhou, and Chang Zhou. Chinese clip: Contrastive vision-language pretraining in chinese. arXiv preprint arXiv:2211.01335, 2022.

Fulong Ye, Guangyi Liu, Xinya Wu, and Ledell Yu Wu. Altdiffusion: A multilingual text-to-image diffusion model. ArXiv, abs/2308.09991, 2023. URL https://api.semanticscholar. org/CorpusID:261048720.

Peter Young, Alice Lai, Micah Hodosh, and Julia Hockenmaier. From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions. Transactions of the Association for Computational Linguistics, 2:67–78, 2014.

Jiaxing Zhang, Ruyi Gan, Junjie Wang, Yuxiang Zhang, Lin Zhang, Ping Yang, Xinyu Gao, Ziwei Wu, Xiaoqun Dong, Junqing He, et al. Fengshenbang 1.0: Being the foundation of chinese cognitive intelligence. arXiv preprint arXiv:2209.02970, 2022.

