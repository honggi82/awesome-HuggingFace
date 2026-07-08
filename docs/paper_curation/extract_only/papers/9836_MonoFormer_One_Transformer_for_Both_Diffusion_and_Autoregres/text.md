# arXiv:2409.16280v1[cs.CV]24Sep2024

## MONOFORMER: ONE TRANSFORMER FOR BOTH DIFFUSION AND AUTOREGRESSION

### Chuyang Zhao1† Yuxing Song1† Wenhao Wang2 Haocheng Feng1 Errui Ding1

Yifan Sun1∗ Xinyan Xiao1∗ Jingdong Wang1∗ 1Baidu VIS 2University of Technology Sydney

ABSTRACT

Most existing multimodality methods use separate backbones for autoregressionbased discrete text generation and diffusion-based continuous visual generation, or the same backbone by discretizing the visual data to use autoregression for both text and visual generation. In this paper, we propose to study a simple idea: share one transformer for both autoregression and diffusion. The feasibility comes from two main aspects: (i) Transformer is successfully applied to diffusion for visual generation, and (ii) transformer training for autoregression and diffusion is very similar, and the difference merely lies in that diffusion uses bidirectional attention mask and autoregression uses causal attention mask. Experimental results show that our approach achieves comparable image generation performance to current state-of-the-art methods as well as maintains the text generation capability. The project is publicly available at https://monoformer.github.io/.

1 INTRODUCTION

Diffusion models are popular for image generation and other continuous data. It is a probabilistic approach to modeling continuous data, which creates samples by simulating the diffusion process, gradually adding and removing noise from data. Diffusion is initially studied in the pixel space for visual generation (Ho et al., 2020). Latent diffusion models (Rombach et al., 2022) performs the diffusion in the latent representation space, and are now commonly used in many well-known models, such as Stable Diffusion (Rombach et al., 2022) and DiT (Peebles & Xie, 2023).

Autoregressive models are dominant in large language models. The basic idea is to predict the discrete tokens one by one. It is also widely studied for visual generation by discretizing the image patches through VQ-VAE (Van Den Oord et al., 2017) or dVAE (Ramesh et al., 2021a). Autoregression is advantageous in building a unified transformer for multi-modality understanding and generation models (Sun et al., 2024b; Liu et al., 2024; Team, 2024). Unfortunately, it does not benefit from recent advances in diffusion. Autoregression and diffusion are studied in parallel and often learned in different models. Some attempts for combining them for multi-modality understanding and generation models adopt two separate networks for text generation and visual generation, respectively (Sun et al., 2023; Lian et al., 2023).

In this paper, we aim at building and training one transformer for both autoregression and diffusion. The proposed approach is named as MonoFormer and is illustrated in Figure 1. The idea is very simple and inspired by the success of using transformer for diffusion (Peebles & Xie, 2023) for image generation, as well as the below-discussed observations about the transformer for autoregression and diffusion. The main difference in training the transformer is that autoregressive transformer adopts a causal attention mask and diffusion transformer does not mask any position, or uses a bidirectional attention mask. On the other hand, the transformer receives continuous embeddings, e.g., text token embedding or image encoding, and outputs continuous embeddings for subsequent text token prediction and image decoding. Thus, it is feasible to learn one transformer for both discrete autoregression and continuous diffusion. We demonstrate the idea by training a single transformer that is shared by autoregression for text generation and diffusion for image generation.

∗Corresponding authors, †Equal contribution

(a) Autoregression (causal mask) (b) Diffusion (bidirectional mask)

draw an image of flower .

<si>

###### transformer

Please draw an image of flower .

<si>

- Figure 1: Our approach MonoFormer trains the autoregressive transformer and the diffusion transformer, which share the weights, and uses causal attention mask and bidirectional attention mask, respectively. During training, the input of the transformer for autoregression is the text token embeddings, and the output is embeddings that are further processed for text generation. The input for diffusion is the noised latent embeddings, and the output is embeddings that are used to predict the noise.

We train our model on two tasks illustrated in Figure 2: autoregression for text-to-text generation and diffusion for text-to-image generation. We use a pretrained LLM as the transformer backbone to gain language understanding capabilities. Experiments demonstrate that our method achieves comparable image generation performance to current state-of-the-art methods as well as maintains the text generation capability.

2 RELATED WORKS

Diffusion models. Diffusion models (Sohl-Dickstein et al., 2015; Ho et al., 2020; Rombach et al., 2022; Song et al., 2020) have demonstrated outstanding performance in generating high-quality images. These models show significant advantages in terms of stability and scalability. Diffusion models (Nichol et al., 2021; Saharia et al., 2022) for text-to-image generation incorporate pretrained text encoders, such as CLIP (Radford et al., 2021), Flan-T5 (Chung et al., 2024), or LLaMA (Touvron et al., 2023a), or use LLMs to encode the text (Koh et al., 2024) as the condition.

Recently, the architecture of diffusion models has been shifting from U-Net architectures to transformer-based architectures (Peebles & Xie, 2023; Podell et al., 2023; Gao et al., 2024), narrowing the gap between image generation and language understanding tasks. This inspired us to use one transformer for both autoregression and diffusion generation.

Autoregressive models. Autoregressive (AR) models are widely used in text generation tasks and have demonstrated remarkable performance in large language models (LLMs) (Brown, 2020; Radford et al., 2019; Touvron et al., 2023a; Team et al., 2023; Touvron et al., 2023b).

AR models have also been applied in image generation (Parmar et al., 2018; Child et al., 2019; Ramesh et al., 2021b; Chang et al., 2023; Ding et al., 2022; Saharia et al., 2022; Li et al., 2024). The AR models usually use visual tokenizers, such as VQ-VAE (Van Den Oord et al., 2017) or VQGAN (Esser et al., 2021), to discretize continuous image features into discrete tokens. Though AR models perform well for image generation (Ramesh et al., 2022; Yu et al., 2022; Chen et al., 2020), the overall quality is still inferior to diffusion models.

Unified models. There has been a trend in the community towards a unified model for text-to-text generation and text-to-image generation. Recent works (Team, 2024; He et al., 2024) propose to use a single autoregressive model for text-to-text generation and text-to-image generation, illustrated in Figure 3 Left. These methods still use autoregression for image generation and thus still underperform diffusion models. Prior works simply adopt an additional network for diffusion to generate high-quality images Ge et al. (2024); Wu et al. (2023), which is illustrated in Figure 3 Right.

[Figure 1]

[Figure 2]

[Figure 3]

Can you list some of the popular landmarks in the France?

Please generate an image of lesser panda.

An image of a nerdy bear wearing glasses and bowtie.

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

Sure, here are some of the landmarks in France:

- 1. Eiffel Tower
- 2. Louvre Museum
- 3. Notre-Dame Cathedral
- 4. Palace of Versailles
- 5. Arc de Triomphe
- 6. Montmartre

- Figure 2: Examples of MonoFormer for both image generation and text generation tasks. Left: Class-conditional image generation. Middle: Text-to-image generation. Right: Text-to-text generation.
- 3 METHOD

- 3.1 PRELIMINARIES

Autoregressive transformer. An autoregressive model factorizes the joint distribution autoregressively over a sequence of high-dimensional data {x1,x2,...,xn}:

n i=1

p(xi|x<i;θ). (1) The model, which is parameterized by θ, is trained by minimizing the negative log-likelihood:

p(x1,x2,...,xn) =

n i=1

log p(xi|x<i;θ)]. (2)

Ex∈X[−log p(x)] = Ex∈X[−

For inference, the model predicts the data one by one: predict x1, ..., xi, ... until xn by sampling from p(x1;θ), ..., p(xi|x<i;θ), ..., p(xn|x<n;θ).

The transformer decoder is used in models like GPT and LLaMA, for the implementation. The decoder takes token embeddings as input and produces an embedding for each position. The autoregressive transformer is similar to a standard transformer decoder, with the primary difference being the use of a causal attention mask. This mask ensures that each position only attends to previous positions, maintaining the autoregressive property.

Diffusion transformer. Diffusion models consist of a diffusion process, which gradually adds Gaussian noise to the data x0:

#### xt = √α¯tx0 + √1 − α¯tϵ, (3)

where α¯t are parameters derived from the variance schedule, and ϵ is the Gaussian noise. Diffusion models are trained to learn the denoising process:

pθ(xt−1|xt) = N(xt−1;µθ(xt),σθ(xt)). (4) The DDPM (Ho et al., 2020) transfers the problem through reparameterization to learn a noise prediction network ϵθ by minimizing the difference between the predicted noise ϵθ(xt) and the ground-truth noise ϵ:

#### Lsimple = ∥ϵθ(xt) − ϵ∥22. (5)

U-Net (Ronneberger et al., 2015) is widely used as the backbone for the noise prediction network ϵθ. Latent diffusion models (Rombach et al., 2022) train the model in the latent representation space that is formed by learning a variational autoencoder (VAE) (Kingma, 2013) to compress images.

Recently, Diffusion Transformers (DiTs) (Peebles & Xie, 2023), which our approach builds upon, train latent diffusion models using a transformer. We adopt the variant based on the DiT block with in-context conditioning. The architecture utilizes standard self-attention over condition embeddings (for text and timesteps) and latent embeddings. The self-attention is bidirectional, meaning that there is no mask or equivalently the mask is an all-ones matrix.

draw an image of flower .

<si>

<ei>

transformer

Please draw an image of flower .

<si>

draw an image of flower .

autoregressive transformer diffusion transformer or U-Net

Please draw an image of flower

- Figure 3: Left: A single autoregressive transformer for both text generation and visual generation. Example methods inlcude Chameleon (Team, 2024) and LlamaGen (Sun et al., 2024b). Right: One transformer is for autoregressive text generation, and the output embeddings are sent to another model for diffusion-based text-to-image generation (Ge et al., 2024; Wu et al., 2023).

- 3.2 MONOFORMER

Our approach is built upon the common-used large language model architecture, and uses one transformer for both autoregression-based text-to-text generation and diffusion-based text-to-image generation.

For text-to-text generation, the input text is processed as a sequence of text token embeddings, which are fed into the transformer. The transformer autoregressively generates output embeddings at each position, which is then parsed into text tokens.

For text-to-image generation, the noised latents are sent to the transformer to generate embeddings at all positions simultaneously. This process is iterated over multiple timesteps in a cascade manner, following the standard denoising diffusion pipeline. The final output embeddings are decoded to an image through a VAE decoder.

Training. The transformer is trained using the autoregression loss for text-to-text generation, and the diffusion loss for text-to-image generation. We adopt the standard LLM transformer architecture. The transformer is composed of transformer blocks. Each block is formed by the FFN and the masked attention:

masked-attention(Q,K,V) = softmax(

### QK⊤

d ⊙ M)V, (6)

√

where M is the attention mask, ⊙ is the element-wise product, and d is the dimension. The key difference between autoregression and diffusion lies in the mask used.

We adopt the standard causal attention mask (upper triangular mask) for optimizing the autoregression loss:

MAR =





1 −∞ −∞ −∞ −∞ −∞ −∞ 1 1 −∞ −∞ −∞ −∞ −∞ 1 1 1 −∞ −∞ −∞ −∞ 1 1 1 1 −∞ −∞ −∞ 1 1 1 1 1 −∞ −∞ 1 1 1 1 1 1 −∞ 1 1 1 1 1 1 1

 

 

. (7)

The transformer takes the text embedding to as input and applies the causal attention mask MAR to mask past tokens. The output embeddings are then used to predict the tokens through the autore-

gression head HAR:

¯to = HAR(transformer(to,MAR;θ)). (8)

Optimizing the diffusion loss is similar. The major difference lies in the attention mask. Unlike in text generation, where text tokens see only past tokens, image tokens see both past text tokens and future image tokens. A bidirectional attention mask is adopted. We illustrate the attention mask

using an example with three text tokens and four image tokens:





1 −∞ −∞ −∞ −∞ −∞ −∞ 1 1 −∞ −∞ −∞ −∞ −∞ 1 1 1 −∞ −∞ −∞ −∞ 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1

. (9)

MDi =

 

 

The transformer takes the embeddings of the text to and noised latent embeddings zt as input, processing them with the attention mask MDi. It outputs embeddings that are then passed through the diffusion head HDi to predict the noise:

ϵ¯ = HDi(transformer(to,zt,MDi;θ)). (10) We use the standard diffusion loss, as used in DiT (Peebles & Xie, 2023), for training. The whole loss is a combination of the standard text-to-text autoregression loss and text-to-image diffusion loss:

#### ℓAR(¯to,to) + ℓDi(¯ϵ,ϵ). (11)

Inference. For text-to-text generation, the inference is a standard autoregression process, using the trained transformer to predict next tokens one by one.

For text-to-image generation, we follow DiT (Peebles & Xie, 2023). Once the image start token <si> is generated in the autoregression process, the diffusion process begins. We input the Gaussian noise as the initial noised latent, to the transformer, predicting the noise that is used to reduce the noise. The noise reduction process is iterated for multiple timesteps to generate the image. The denoising process is almost the same as that of the in-context version of DiT (Peebles & Xie, 2023) with a slight difference: Our approach uses a combination of causal attention mask (for text tokens) and the bidirectional attention mask (for image tokens). In contrast, the in-context version of DiT uses a bidirectional attention mask for both text and image tokens.

Architecture for diffusion. We perform diffusion in the latent space. We use the VAE encoder to map each image patch to a continuous representation followed by a linear projection to generate latent representations for latent diffusion.

The noised latent embeddings are added with positional embeddings, using the sine-cosine version Following DiT (Peebles & Xie, 2023), we embed the input timestep using a 256-dimensional frequency embedding, followed by a two-layer MLP with SiLU activations. The time embeddings are combined with the noised latent embeddings through an AdaLN layer. The combined embeddings are fed into the transformer for predicting the noise, followed by denoising the noised embeddings.

The diffusion head HDi is implemented following DiT, which consists of a layer norm followed by a linear layer and SiLU activation.

- 4 EXPERIMENTS

- 4.1 SETUP

Implementation details. We leverage the pretrained variational autoencoder (VAE) from Stable Diffusion (Rombach et al., 2022) to encode the image. The VAE encoder downsamples the image by a factor of 8, generating a latent representation of dimension 4. Following DiT (Peebles & Xie, 2023), we patchify the latent representation using patch size of 2 × 2. The latent representation goes through a linear projection layer for matching the dimension of the representation that is fed into the transformer. The linear projection layer for aligning the latent representation dimensions is initialized to zero. The linear layers in the time embedding projector are initialized to zero. The AdaLN parameters for combining time embeddings are initialized using a normal distribution. The linear layers in the diffusion head HDi are initialized using a normal distribution too. We initialize the transformer using TinyLlama-1.1B v1.0 (Zhang et al., 2024), which is pre-trained on 3T tokens and fine-tuned on UltraChat (Ding et al., 2023) dataset.

### Table 1: Performance on ImageNet 256×256 benchmark.

|Method|Arch #Params|FID↓ IS↑ Precision↑ Recall↑<br><br>|
|---|---|---|
|ADM (Dhariwal & Nichol, 2021) CDM (Ho et al., 2022) LDM (Rombach et al., 2022) DiT-XL/2 (Peebles & Xie, 2023)<br><br>|Diff 554M Diff Diff 400M Diff 675M|10.94 101.0 0.69 0.63 4.88 158.7 - 3.60 147.6 0.87 0.68 2.27 278.2 0.83 0.57|
|VQGAN (Esser et al., 2021) ViT-VQGAN (Yu et al., 2021) LlamaGen-B (Sun et al., 2024b) LlamaGen-3B (Sun et al., 2024b)|AR 227M AR 1.7B AR 111M AR 3.1B<br><br>|18.65 80.4 0.78 0.26<br><br>4.17 175.1 - -<br>5.46 193.6 0.83 0.45 2.81 311.5 0.84 0.54<br>|
|MonoFormer|AR+Diff 1.1B|2.57 272.6 0.84 0.56|

We use the AdamW optimizer without weight decay, and the learning rate is set to a constant value of 1e-4. We maintain an exponential moving average of the MonoFormer weights throughout the training process with a decay rate of 0.9999. We retain the diffusion hyper-parameters from DiT (Peebles & Xie, 2023), using 1000 timesteps linear variance schedule ranging from 1 × 10−4 to 2 × 10−2, and parameterization of the covariance Σθ.

We train the model on the ImageNet (Deng et al., 2009) dataset for class-conditional generation. The category names are converted into text to form the text prompt in ImageNet, such as “Please generate an image of [category]”, “An image of [category]”, etc. We train the model on the JourneyDB (Sun et al., 2024a) and UltraChat (Ding et al., 2023) for text-to-image generation and text generation. Considering that the diffusion task is more difficult, the ratio of the numbers of image generation samples and text generation samples is 9 : 1. The Global batch size is 1024.

Classifier-free guidance. We adopt classifier-free guidance (Ho & Salimans, 2022) for text-toimage generation. In the training stage, we randomly drop the text tokens for unconditional generation. We set the drop probability to 1/10. The final latent output is computed by combining the unconditional and conditional outputs and using a guidance factor to control the scale of the guidance.

- 4.2 EXPERIMENT RESULTS

Image generation. We evaluate the text-to-image generation performance on the ImageNet (Deng et al., 2009) dataset under resolution of 256 × 256. The evaluation metrics include: the Fr´echet Inception Distance (FID), the Inception Score (IS), and Precision/Recall. In inference, we use a classifier-free guidance scale of 1.5. We report the comparison to representative diffusion models and unified autoregressive models. The comparison results are shown in Table 1. Our method achieves comparable results with recent diffusion-based or AR-based methods. It outperforms the AR-based LlamaGen-3B (Sun et al., 2024b) in FiD with fewer parameters, while being only 0.3 lower than the state-of-the-art diffusion-based method DiT-XL/2 (Peebles & Xie, 2023) in FID.

Text generation. We evaluate MonoFormer on a diverse set of commonsense reasoning tasks and compare it with the baseline model, TinyLlama (Zhang et al., 2024). We evaluate our method on the following tasks: HellaSwag (Zellers et al., 2019), OpenBookQA (Mihaylov et al., 2018), WinoGrande (Sakaguchi et al., 2021), ARC-Easy and ARC-Challenge (Clark et al., 2018), BoolQ (Clark et al., 2019), and PIQA (Bisk et al., 2020).

### Table 2: Performance on commonsense reasoning tasks.

|Model|HellaSwag OBQA WinoGrande ARC-C ARC-E BoolQ PIQA avg<br><br>|
|---|---|
|Pythia (Biderman et al., 2023) TinyLlama (Zhang et al., 2024)|52.01 33.20 57.38 28.50 54.00 63.27 70.95 51.33 59.20 36.00 59.12 30.12 55.25 57.83 73.29 52.97<br><br>|
|MonoFormer|50.62 37.20 56.91 31.48 48.19 62.29 71.16 51.12|

(a) (b) (c)

- Figure 4: (a) The effect of transformer initialization for image generation, measured using the FiD10K metric on ImageNet. (b) The effect of transformer initialization for text generation, measured by the average commonsense reasoning score. (c) The effect of bidirectional attention mask for image generation.

Table 2 shows that MonoFormer achieves performance comparable to the TinyLlama baseline, with slight drops on certain benchmarks (average score deceases from 52.97 to 51.12). The slight performance drops may be attributed to the mixed training with the image generation dataset. We believe the performance can be further improved when more language data are incorporated in training, we leave this for future exploration.

- 4.3 ABLATION STUDY

Transformer initialization. We conduct an ablation study to study the impact of using pretrained LLMs for transformer initialization. We compare the performance of MonoFormer with and without pretrained LLM initialization on two tasks: image generation on the ImageNet 256×256 benchmark and language understanding on commonsense reasoning benchmarks. Performance is evaluated using the FiD score for ImageNet and the average score across 6 commonsense reasoning tasks, as described in Section 4.2. As shown in Figure 4 (a), MonoFormer with pretrained LLM initialization significantly outperforms the counterpart in commonsense reasoning. As shown in Figure 4 (b), on the image generation benchmark, MonoFormer with LLM initialization also shows superior performance. We attribute this improvement to the pretrained LLM’s ability to better understand prompts, thereby benefiting text-to-image generation tasks.

Bidirectional attention for diffusion. We study the effect of applying bidirectional attention masks among noised latent tokens, which are used for diffusion-based generation. We respectively use causal attention mask and bidirectional attention mask for the noised latent tokens and compare their performance on the ImageNet 256×256 benchmark. As shown in Figure 4 (c), the performance of MonoFormer with bidirectional attention outperforms MonoFormer with causal attention mask, demonstrating the importance of bidirectional attention mask for image generation.

- 5 CONCLUSION

This paper shows that discrete autoregression and continuous diffusion are able to share one transformer. Experiments validate that sharing the transformer is able to achieve good performance for text-to-text generation and text-to-image generation that is comparable to not sharing the transformer.

REFERENCES

Stella Biderman, Hailey Schoelkopf, Quentin Gregory Anthony, Herbie Bradley, Kyle O’Brien, Eric Hallahan, Mohammad Aflah Khan, Shivanshu Purohit, USVSN Sai Prashanth, Edward Raff, et al. Pythia: A suite for analyzing large language models across training and scaling. In International Conference on Machine Learning, pp. 2397–2430. PMLR, 2023.

Yonatan Bisk, Rowan Zellers, Jianfeng Gao, Yejin Choi, et al. Piqa: Reasoning about physical commonsense in natural language. In Proceedings of the AAAI conference on artificial intelligence, volume 34, pp. 7432–7439, 2020.

Tom B Brown. Language models are few-shot learners. arXiv preprint arXiv:2005.14165, 2020. Huiwen Chang, Han Zhang, Jarred Barber, AJ Maschinot, Jose Lezama, Lu Jiang, Ming-Hsuan

Yang, Kevin Murphy, William T Freeman, Michael Rubinstein, et al. Muse: Text-to-image generation via masked generative transformers. arXiv preprint arXiv:2301.00704, 2023.

Mark Chen, Alec Radford, Rewon Child, Jeffrey Wu, Heewoo Jun, David Luan, and Ilya Sutskever. Generative pretraining from pixels. In International conference on machine learning, pp. 1691–

1703. PMLR, 2020. Rewon Child, Scott Gray, Alec Radford, and Ilya Sutskever. Generating long sequences with sparse transformers. arXiv preprint arXiv:1904.10509, 2019.

Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al. Scaling instruction-finetuned language models. Journal of Machine Learning Research, 25(70):1–53, 2024.

Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. Boolq: Exploring the surprising difficulty of natural yes/no questions. arXiv preprint arXiv:1905.10044, 2019.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457, 2018.

Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pp. 248–255. Ieee, 2009.

Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021.

Ming Ding, Wendi Zheng, Wenyi Hong, and Jie Tang. Cogview2: Faster and better text-to-image generation via hierarchical transformers. Advances in Neural Information Processing Systems, 35:16890–16902, 2022.

Ning Ding, Yulin Chen, Bokai Xu, Yujia Qin, Zhi Zheng, Shengding Hu, Zhiyuan Liu, Maosong Sun, and Bowen Zhou. Enhancing chat language models by scaling high-quality instructional conversations. arXiv preprint arXiv:2305.14233, 2023.

Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 12873–12883, 2021.

Peng Gao, Le Zhuo, Ziyi Lin, Chris Liu, Junsong Chen, Ruoyi Du, Enze Xie, Xu Luo, Longtian Qiu, Yuhang Zhang, et al. Lumina-t2x: Transforming text into any modality, resolution, and duration via flow-based large diffusion transformers. arXiv preprint arXiv:2405.05945, 2024.

Yuying Ge, Sijie Zhao, Jinguo Zhu, Yixiao Ge, Kun Yi, Lin Song, Chen Li, Xiaohan Ding, and Ying Shan. Seed-x: Multimodal models with unified multi-granularity comprehension and generation. arXiv preprint arXiv:2404.14396, 2024.

Wanggui He, Siming Fu, Mushui Liu, Xierui Wang, Wenyi Xiao, Fangxun Shu, Yi Wang, Lei Zhang, Zhelun Yu, Haoyuan Li, Ziwei Huang, LeiLei Gan, and Hao Jiang. Mars: Mixture of auto-regressive models for fine-grained text-to-image synthesis, 2024.

Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

Jonathan Ho, Chitwan Saharia, William Chan, David J Fleet, Mohammad Norouzi, and Tim Salimans. Cascaded diffusion models for high fidelity image generation. Journal of Machine Learning Research, 23(47):1–33, 2022.

Diederik P Kingma. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013. Jing Yu Koh, Daniel Fried, and Russ R Salakhutdinov. Generating images with multimodal language

models. Advances in Neural Information Processing Systems, 36, 2024. Tianhong Li, Yonglong Tian, He Li, Mingyang Deng, and Kaiming He. Autoregressive image generation without vector quantization. arXiv preprint arXiv:2406.11838, 2024.

Long Lian, Boyi Li, Adam Yala, and Trevor Darrell. Llm-grounded diffusion: Enhancing prompt understanding of text-to-image diffusion models with large language models. arXiv preprint arXiv:2305.13655, 2023.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36, 2024.

Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. Can a suit of armor conduct electricity? a new dataset for open book question answering. arXiv preprint arXiv:1809.02789, 2018.

Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv preprint arXiv:2112.10741, 2021.

Niki Parmar, Ashish Vaswani, Jakob Uszkoreit, Lukasz Kaiser, Noam Shazeer, Alexander Ku, and Dustin Tran. Image transformer. In International conference on machine learning, pp. 4055–

4064. PMLR, 2018. William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 4195–4205, 2023.

Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748–8763. PMLR, 2021.

Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. CoRR, abs/2102.12092, 2021a.

Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In International conference on machine learning, pp. 8821–8831. Pmlr, 2021b.

Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical textconditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1(2):3, 2022.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10684–10695, 2022.

Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation, 2015.

Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in neural information processing systems, 35:36479–36494, 2022.

Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Winogrande: An adversarial winograd schema challenge at scale. Communications of the ACM, 64(9):99–106, 2021.

Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, pp. 2256–2265. PMLR, 2015.

Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020.

Keqiang Sun, Junting Pan, Yuying Ge, Hao Li, Haodong Duan, Xiaoshi Wu, Renrui Zhang, Aojun Zhou, Zipeng Qin, Yi Wang, et al. Journeydb: A benchmark for generative image understanding. Advances in Neural Information Processing Systems, 36, 2024a.

Peize Sun, Yi Jiang, Shoufa Chen, Shilong Zhang, Bingyue Peng, Ping Luo, and Zehuan Yuan. Autoregressive model beats diffusion: Llama for scalable image generation. arXiv preprint arXiv:2406.06525, 2024b.

Quan Sun, Qiying Yu, Yufeng Cui, Fan Zhang, Xiaosong Zhang, Yueze Wang, Hongcheng Gao, Jingjing Liu, Tiejun Huang, and Xinlong Wang. Emu: Generative pretraining in multimodality. In The Twelfth International Conference on Learning Representations, 2023.

Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024.

Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023a.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023b.

Aaron Van Den Oord, Oriol Vinyals, et al. Neural discrete representation learning. Advances in neural information processing systems, 30, 2017.

Shengqiong Wu, Hao Fei, Leigang Qu, Wei Ji, and Tat-Seng Chua. Next-gpt: Any-to-any multimodal llm. arXiv preprint arXiv:2309.05519, 2023.

Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer to unify multimodal understanding and generation, 2024.

Jiahui Yu, Xin Li, Jing Yu Koh, Han Zhang, Ruoming Pang, James Qin, Alexander Ku, Yuanzhong Xu, Jason Baldridge, and Yonghui Wu. Vector-quantized image modeling with improved vqgan. arXiv preprint arXiv:2110.04627, 2021.

Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, et al. Scaling autoregressive models for contentrich text-to-image generation. arXiv preprint arXiv:2206.10789, 2(3):5, 2022.

Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish your sentence? arXiv preprint arXiv:1905.07830, 2019.

Peiyuan Zhang, Guangtao Zeng, Tianduo Wang, and Wei Lu. Tinyllama: An open-source small language model. arXiv preprint arXiv:2401.02385, 2024.

Chunting Zhou, Lili Yu, Arun Babu, Kushal Tirumala, Michihiro Yasunaga, Leonid Shamis, Jacob Kahn, Xuezhe Ma, Luke Zettlemoyer, and Omer Levy. Transfusion: Predict the next token and diffuse images with one multi-modal model, 2024.

- A IMAGE GENERATION RESULTS

We present class-conditional image generation results in Figure 5. We use a classifier-free guidance scale of 4 and sample 250 steps using DPM-Solver. We further train our model on the JourneyDB (Sun et al., 2024a) dataset and the UltraChat (Ding et al., 2023) dataset for free text-to-image generation. Figure 6 showcases the results generated using Parti prompts (Yu et al., 2022), with a classifier-free guidance scale of 4 and 250 sampling steps.

- B EXTENSION TO MULTI-MODALITY UNDERSTANDING

Our approach can be easily extended for multi-modality understanding, for example, visionlanguage understanding. In the case that the data sample is an interleaved image-text sequence, there are two choices for extracting the image representation for understanding. One choice is that the image is only processed by diffusion, and we choose to use the transformer with a timestep near zero for image representation extraction. The other choice is that the image is processed in an autoregressive manner for understanding and diffusion is only for generation.

- C DICCUSSION WITH CONCURRENT WORK

There are concurrent works with similar ideas, such as Transfusion (Zhou et al., 2024) and Showo (Xie et al., 2024). Show-o employs discrete diffusion. Differently, our approach uses continuous diffusion model. Transfusion is more similar to our method, and trained from scratch. Our experiments demonstrate that initializing transformer with a large language model is helpful for training.

[Figure 9]

[Figure 10]

[Figure 11]

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

[Figure 42]

[Figure 43]

##### Figure 5: Example results of class-conditional image generation on ImageNet. We use 250 sampling steps and a classifier-free guidance scale of 4.0.

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

water ying-yang a woman with long black hair and dark skin

artificial intelligence

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

a chimpanzee The Starry Night a shiba inu an espresso machine

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

a pick-up truck rolling over a grassy field

a propaganda poster brain coral a portrait of young girl

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

a capybara sitting in a field

a panda bear with aviator glasses on its head

the Sydney Opera House

the Great Pyramid

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

an old-fashioned windmill a turtle surrounded by flowers

the Taj Mahal at sunrise

a tiger

- Figure 6: Example results of text-to-image generation. The prompts are from Parti prompts (Yu et al., 2022). We use 250 sampling steps and a classifier-free guidance scale of 4.0.

