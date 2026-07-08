## TinyLLaVA: A Framework of Small-scale Large Multimodal Models

# arXiv:2402.14289v1[cs.LG]22Feb2024

Baichuan Zhou1 Ying Hu2 Xi Weng1 Junlong Jia1 Jie Luo1 Xien Liu2 Ji Wu2 Lei Huang1*† 1SKLCCSE, Institute of Artificial Intelligence, Beihang University, Beijing, China 2Department of Electronic Engineering, Tsinghua University, China

### Abstract

We present the TinyLLaVA framework that provides a unified perspective in designing and analyzing the small-scale Large Multimodal Models (LMMs). We empirically study the effects of different vision encoders, connection modules, language models, training data and training recipes. Our extensive experiments showed that better quality of data combined with better training recipes, smaller LMMs can consistently achieve on-par performances compared to bigger LMMs. Under our framework, we train a family of smallscale LMMs. Our best model, TinyLLaVA-3.1B, achieves better overall performance against existing 7B models such as LLaVA-1.5 and Qwen-VL. We hope our findings can serve as baselines for future research in terms of data scaling, training setups and model selections. Our model weights and codes will be made public1.

Figure 1. TinyLLaVA-3.1B vs. LLaVA-1.5-7B.

### 1. Introduction

to only well-funded industries and organizations. From a practical perspective, another line of work that focuses on small-scale models has gained attention because of affordable cost and efficient training and inference, opening up opportunities for resource-limited academic community.

The AI community has witnessed remarkable capabilities of Large Language Models (LLMs). With the scaling laws [19,26] serving as guidelines and emergent abilities [51] being studied, recent years have featured a trend towards scaling up model sizes, with the largest dense language models over 500 billion parameters [9,46]. Inspired by LLMs, Large Multimodal Models (LMMs) [3,38,49,62] stand on the shoulders of giants – aligning visual perception with LLMs to acquire multimodal perceptions [21], so that they can directly inherit powerful capabilities from LLMs. This synergy has led to various LMMs released with a massive amount of parameters, like Flamingo with 80B parameters [2], and PaLM-E with 562B parameters [14].

In this context, the LLM community starts to release versions of relatively smaller scales, such as 7-B versions [48,53] and tiny versions under 3B parameters [25,41,59], without performance degradation compared to their previous larger counterparts. Following the trend of LLMs, large multimodal models have experienced a similar transformation of model shrinking down to small scales by leveraging relatively smaller LLMs, such as OpenFlamingo [3] and LLaVA series [37,38], ranging from 3B to 15B. More recent efforts on LMMs have explored various ways for efficient training and deploying in terms of using tiny LLMs [57,63], applying sparse MoE [35], freezing or lora tuning backbones [49,54].

Despite the fact that scaling up model sizes can significantly enhance performance across various tasks, training such large models requires expensive computational resources and their large sizes may lead to unaffordable training/inference budget, which restricts research access

While large multimodal models with small-scale LLMs make it available for more researchers, current attempts [35, 57,63] take only a glimpse at the wide landscape of design choices of each architecture component, training recipes, the

*Technical Report †Corresponding author: Lei Huang (huangleiAI@buaa.edu.cn) 1available at https://github.com/DLCV-BUAA/TinyLLaVABench.

|Language Response 𝐘𝑎|
|---|

scales of training data, and more. The variability of design options and diversity of techniques in this burgeoning field lead to the complexity in designing LMMs and difficulty in understanding the space of existing methods. In this work, we investigate the wide landscape of large multimodal models under the setting of leveraging small-scale LLMs, which allows us to provide a thorough empirical analysis of different approaches, thereby assisting the researchers and practitioners to navigate in this space. As a result, we present TinyLLaVA, a framework that consists of a vision encoder, small-scale LLM decoder, and intermediate connector, as well as training pipelines.

Tokenizer & embedding

|Small-scale LLM 𝐹𝜃|
|---|

| |
|---|

Embedding Space

|𝐇|
|---|

|Connector 𝑃𝜙|
|---|

|Vision Encoder 𝑉𝜑|
|---|

|Language Instruction 𝐘𝑞|
|---|

|Image 𝐗|
|---|

Based on this framework, we investigate the effects of different vision encoders, connection modules, language models, training data and training recipes. Our empirical experiments show that with better training recipes and quality of data, smaller LMMs can achieve on-par performance with larger counterparts, setting new baselines for the research field. We finally present a family of smallscale LMMs, encompassing three language models: Phi2 [33], StableLM-2 [47], and TinyLlama [59], and two vision encoders: CLIP [44], and SigLIP [58]. Our best model, TinyLLaVA-3.1B, achieves better overall performance against existing 7B models such as LLaVA-1.5 [37] and Qwen-VL [4].

Figure 2. TinyLLaVA Framework.

Small-scale LMMs Deploying LMMs is expensive as they require high computation overhead. The computation bottleneck is usually introduced by LLMs as they tend to scale to billions of parameters [48,52]. However, recent small-scale LLMs such as Phi-2 [33], TinyLlama [59] and StableLM-

- 2 [47] have reached impressive performances while maintaining reasonable compute budgets. Following these efforts, a variety of works [10,35,57,63] explored ways to train and deploy small-scale LMMs. In particular, TinyGPT-V [57] fine-tuned the projection layers following MiniGPT-4 [62] and replaced the LLM [61] with Phi [33]; LLaVA-Phi [33] followed LLaVA-1.5’s procedure and replaced the LLM [52] with Phi-2 [33]; MoE-LLaVA [35] introduced Mixture-ofExperts [23] to the LLaVA architecture and reached competitive performance with LLaVA-1.5 using less activated parameters.

Distinct to these works [10,35,57,63] that focus on building and training specific small-scale LMMs, our work aims to provide a unified analysis on how model selections, training recipes, and data contribute to model performance for small-scale LMMs. We noted that concurrent work [27] also provides a unified analysis of visually-conditioned language models, but they focus on standard LMMs while we focus on small-scale LMMs. The investigation on small-scale LMMs shows different behaviors than the standard ones, based on our experiments.

- 3. TinyLLaVA Framework

### 2. Related Work

Large Multimodal Models With the development of powerful Large Language Models (LLMs) [5,9,48] and vision models [44,58], Large Multimodal Models (LMMs) have seen great improvements [1]. Early works [8,49,50] pioneered introducing autoregressive LLMs to vision-language learning. The following research focused on effectively exploiting LLMs by viewing visual signals as conditional information [2,15,31]. In particular, Flamingo [2] consider inserting adapters in LLMs and utilizing a perceiver-like [24] architecture to extract visual features and demonstrated impressive performance on vision-language few-shot learning. BLIP models [31, 32] introduce data filtering to improve performance on vision language tasks such as VQA [17] and image captioning [36]. While these models exhibited great vision-language abilities, they only possessed limited zeroshot abilities as they were not trained to follow instructions.

To better align LMMs with human preferences, recent works, such as LLaVA [38] and InstructBLIP [12], follow [11,43] and fine-tune LMMs with visual instruction tuning data [30,62], which greatly enhance LMM’s zero-shot capabilities. Following this line of work, several techniques are raised to further improve the performances by discussing the possibilities of unlocking vision encoders during training [7,34], curating high-quality visual instruction tuning datasets [7,37,60], and scaling up image resolutions [4,6,34].

In this section, we describe the details of the TinyLLaVA framework that focuses on exploiting small-scale LLMs for large multimodal models. Our TinyLLaVA framework follows the design of LLaVA [38] but generalizes from it for better investigating the variants of the model architecture and training recipe in a unified perspective.

Table 1. Small-scale LLMs and vision encoders used for TinyLLaVA framework in current experiments. ”Abb.” refers to the abbreviated model name, which is used in the naming convention for TinyLLaVA models. ”HF path” denotes the pathway to the pre-trained weights of the relevant models we are using on HuggingFace.

|Type<br><br>|Name|Abb.<br><br>|HF path<br><br>|Size|
|---|---|---|---|---|
|Small-scale LLM<br><br>|TinyLlama StableLM-2 Phi-2|TL SLM Phi<br><br>|TinyLlama/TinyLlama-1.1B-Chat-v1.0 stabilityai/stablelm-2-zephyr-1 6b microsoft/phi-2<br><br>|1.1B<br><br>1.6B<br>2.7B<br>|

CLIP C openai/clip-vit-large-patch14-336 0.3B SigLIP Sig google/siglip-so400m-patch14-384 0.4B

Vison encoder

#### 3.1. Model Architecture

The architecture of TinyLLaVA (Figure 2) consists of a small-scale LLM Fθ, a vision encoder Vφ, and a connector Pϕ, where θ, φ and ϕ are the (learnable) parameters respectively. This architecture can model various multimodal understanding tasks that take as input a pair of image and text sequence and output a text sequence.

Small-scale LLM. The small-scale LLM Fθ takes as input a sequence of vectors {hi}Ni=0−1 with length of N in the d dimensional (text) embedding space, and output the corresponding next-predictions {hi}Ni=1. A tokenizer & embedding module is usually bound to the small-scale LLM, mapping the text input sequences {yi}Ni=0−1 to the embedding space and similarly from the embedding space to the text output sequences {yi}Ni=1.

Vision Encoder. The vision encoder Vφ take as input an image X and output a sequence of (visual) patch features V = {vj ∈ Rd

}Mi=j, where V = Vφ(X). The vision encoder can be Vision Transformers [13] [44] [58] that directly output a sequence of patch features or CNNs that output a grid features followed by a reshape operation to obtain patch features.

x

Connector. The connector Pϕ maps the visual patch sequences {vj}Mj=1 to the text embedding space {hj}Mj=1, where hj = Pϕ(vj),j = 1,...,M. Note that the connector Pϕ is designed for effectively leveraging the capability of both the pre-trained LLM and vision encoder.

#### 3.2. Training Pipeline

The data for training TinyLLaVA consists of imagetext pairs (X,Y). Furthermore, the text sequence Y is structured as a form of multi-turn conversation Y = (Yq1,Ya1,...,YqT,YaT), where T is the total number of turns, Yqt is the human instruction and Yat is the corresponding assistant’s response2. The training of TinyLLaVa is divided into two stages, pre-training and supervised fine-tuning.

2We omit the system-massage for better readability, since they can be merged into the instruction as conditional input for predicting response.

Pre-training for Feature Alignment. In this stage, we aim to better align the vision and text information in the embedding space. We thus use the image-caption style data format (X,Ya) that can be derived from the original multiturn conversation, where X is the image and Ya is a response (description of the image). Given the target response Ya = {yi}N

i=1 with length of Na, we compute the probability of generating Ya conditioned by the image as:

a

p(Ya|X) =

Na

Fθ(yi|Pϕ ◦ Vφ(X)), (1)

i=1

and maximize its log-likelyhood autoregressively as training objective:

max

ϕ,θ′,φ′

Na

log Fθ(yi|Pϕ ◦ Vφ(X)), (2)

i=1

′

′

where θ

are the subset of θ and φ, respectively. Note that our framework allows to adjust partially learnable parameters of the LLM and vision encoder during the pretraining stages, considering that only training the connector may not well align the vision and text information when using small-scale LLM.

and φ

Supervised Fine-tuning. We use the image-text pair (X,Y) in the original form of multi-turn conversation. Let A denotes the set of all the tokens that belong to the assistant responses, i.e., A = {y|y ∈ Yat, for any t = 1,...,T}. We maximize the log-likelihood of assistant’s responses autoregressively as training objective during supervised fine-tuning:

max

ϕ,θ′,φ′

N

I(yi ∈ A)log Fθ(yi|Pϕ ◦ Vφ(X)), (3)

i=1

where N is the length of text sequence Y, and I(yi ∈ A) equals to 1 if yi ∈ A, 0 otherwise. We also allow the adjustment of partially learnable parameters of the LLM and vision encoder during the supervised fine-tuning stage.

Y

Y

TinyLLaVA-base TinyLLaVA-share

F

F

[Figure 1]

[Figure 2]

[Figure 3]

Y

Y

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

V

V

[Figure 10]

[Figure 11]

Connector weight initialization through duplication

[Figure 12]

X

X

[Figure 13]

[Figure 14]

- Figure 3. The primary differences between two recipes. In the base recipe, we keep parameters of both the vision encoder and small-scale LLM frozen and solely updating the connector. In the share recipe, we freeze the first 12 layeres of the vision encoder and update the rest of the model. Additionally, we initialize connector from the base’s pretrained counter part.

### 4. Experiments

In this section, we conduct comprehensive experiments to investigate how the model architectures, datasets, and training recipes affect small-scale Large Multimodal Models (LMMs) performances based on our TinyLLaVA frameworks.

#### 4.1. Experimental Settings

##### 4.1.1 Model Architectures

We select several representative small-scale LLMs, vision encoders, and connectors to instantiate the models following our TinyLLaVA framework.

Small-scale LLMs. Table 1 illustrates our LLM selections. We select three representative small-scale LLMs: TinyLlama (1.1B) [59], StableLM-2-1.6B(1.6B) [47] and Phi-2(2.7B) [33]. We find these selections cover a comprehensive parameter range of current small-scale LLMs.

Vision Encoders. We select CLIP-Large [44] as our vision encoder. Through our preliminary experiments, we found that SigLIP [58] combined with small-scale LLMs yield better performance than CLIP, thus incorporating it into our framework. We use the official checkpoints from HuggingFace for both vision encoders and small-scale LLMs to initialize our models as shown in Table 1.

Connector. Following LLaVA-1.5 [37], we apply a twolayer Multi-Layer Perceptron (MLP) with GELU activation [18] as the connector between the vision encoders and small-scale LLMs. We also tried to employ resamplers as our connectors, which were implemented similarly to [30].

##### 4.1.2 Training Data and Recipes

Table 2. Datasets used for training TinyLLaVA. ”PT” and ”SFT” refer to two stages of training: pre-training and supervised finetuning, respectively.

|Dataset<br><br>|Stage|Source<br><br>|#Sample|
|---|---|---|---|
|LLaVA-1.5<br><br>|PT SFT|LLaVA-1.5-558k LLaVA-1.5-mix-665k<br><br>|558k 665k|

PT ShareGPT4V-PT-1246k 1246k

ShareGPT4V

SFT ShareGPT4V-mix-665k 665k

Training Data. We select two different training datasets, proposed in LLaVA-1.5 [37] and ShareGPT4V [7], to study how data quality affects LMM’s performance. We outline their differences in Table 2.

LLaVA-1.5-PT consists of 558k captions, as described in [37]. LLaVA-1.5-SFT contains a total of 665k visual instruction tuning conversations, which is a combination of academic-oriented visual question answering (VQA) [17, 22, 28, 45] samples, instruction tuning data from LLaVAInstruct [38] and ShareGPT [20].

ShareGPT4V-PT [7] includes 1246k captions generated by the Share-Captioner [7]. ShareGPT4V-SFT dataset is similar to LLaVA-1.5-SFT [37], with the exception that the 23K detailed description data in LLaVA-1.5-SFT being replaced with detailed captions randomly sampled from the 100K ShareGPT4V data [7].

Training Recipes. We explore two existing training recipes from [37] [7] and study their effects on our model

(a) CLIP (b) SigLIP

- Figure 4. Ablation of small-scale LLM backbones. Under the base recipe, We train six variants with three small-scale LLMs and two vision encoders mentioned in Table 1 on LLaVA-1.5 dataset. The titles of the subplots indicate the corresponding vision encoders.

(a) TinyLlama (b) StableLM-2 (c) Phi-2

- Figure 5. Ablation of vision encoders. These results are inherited from Figure 4. The titles of the subplots indicate the corresponding small-scale LLMs.

variants. Their primary distinction is summarized in Figure 3.

The first recipe is adopted from LLaVA-1.5 [37] and named base, which serves as our baseline recipe. During pre-training, we only update the connector Pϕ and keep the rest of the model frozen, and tune the model for one epoch with a learning rate of 1e-3 and a batch size of 256. In the supervised fine-tuning stage, we keep the vision encoder Vφ frozen and update both the connector Pϕ and the small-scale LLM Fθ, and tune the model for one epoch with a learning rate of 2e-5 and a batch size of 128.

We establish our second training recipe share, following ShareGPT4V [7]. During pre-training of the share recipe, we initialize the connector from the base’s pretrained counterpart. Additionally, we keep the first 12 layers of the vision

encoder Vφ frozen and update the rest of the model for one epoch, with a learning rate of 2e-5 and a batch size of 256. The setup of supervised fine-tuning is the same as the base recipe.

##### 4.1.3 Evaluation Benchmark

We evaluate our model variants on four image questionanswering benchmarks: VQA-v2 [17], GQA [22], ScienceQA-IMG [40], and TextVQA [45], and five comprehensive benchmark: POPE [55], MM-Vet [56], LLaVAW (LLaVA-Bench-in-the-Wild) [38], MME [16] and MMBench [39]. We provide a brief overview of the key aspects of each benchmark focuses on when assessing model capabilities (See Appendix A).

reveal that, using resampler as the connector results in a degradation of performance under a similar parameter setting compared with MLP, which is consistent with previous research findings [37]. We anticipate further exploration of diverse connectors in future studies.

Summary. In this part, we observe that model variants with larger LLMs can achieve better overall performance. Besides, Applying SigLIP [58] (with a higher input resolution and more visual tokens) as the vision encoder can improve performance significantly compared to CLIP [44].

4.2.2 Investigating Data Mixtures and Training Recipes Ablation of Data Mixtures. We also conduct ablation experiments under the base recipe to showcase the impact of different data mixtures. Our results in Figure 7 indicate that, when pretrained on the more extensive and more diverse ShareGPT4V [7] dataset under the base recipe, model variants with TinyLlama [59] as the small-scale LLM demonstrate an overall improvement in evaluation performance compared to the LLaVA-1.5 dataset [37]. However, notable degradation is observed in POPE [55]. In contrast, the performance of model variants with StableLM-2 and Phi-2 experienced a comprehensive improvement. We speculate that this may be due to the insufficient parameters of TinyLlama [59], which prevents it from adequately fitting to a larger amount of data and results in partial knowledge degradation and more hallucinations.

- Figure 6. Preliminary exploration of connectors in TinyLLaVA. Utilizing CLIP as the vision encoder and TinyLlama as the smallscale LLM, we train TinyLLaVAs with the two different connectors on the LLaVA-1.5 dataset, respectively. The results indicate that MLP outperforms Resampler in overall performance.

#### 4.2. Experimental Results

##### 4.2.1 Investigating the Effects of Model Architectures

Ablation of Small-scale LLMs. We conduct ablations on small-scale LLMs backbones. The results are presented in Figure 4. We can observe that model variants using Phi2 [33] perform exceptionally well across various configurations and benchmark evaluations, which could be attributed to the larger parameters of Phi-2. Notably, the Phi-2 variants significantly outperform the other variants on SQA-I [40], which may be attributed to its intensive training on textbook data. While the TinyLlama [59] variants are our smallest model and exhibit slightly lower overall performances, they show better POPE accuracy compared to the StableLM2 [47] variants. Our ablations confirm that larger language models improve performance under the base settings.

Ablation of Training Recipes. Furthermore, we explore the impact of different training recipes. The results are shown in Figure 8. We observe that when models pre-trained on the larger and more diverse ShareGPT4V dataset [7], the share recipe can significantly improve performance for all variants. Note that we partially fine-tune the vision encoder in the share recipe. This observation suggests that fine-tuning the vision encoder can improve the performance when using small-scale LLMs, which is contrary to the result in [27] that fine-tuning the vision encoder dramatically degrades performance when using standard LLMs. We conjecture that whether fine-tuning the vision encoders can improve performance depends on the size of the accompanied LLMs and the size of the training data, which is an interesting direction for further work.

Ablation of Vision Encoders. Following the experimental findings presented in Figure 4, we showcase them in Figure 5. It is noteworthy that model variants with SigLIP [58] exhibit substantial enhancements in model performances compared to those with CLIP [44], which is particularly evident in TextVQA [45] and LLaVA-W [38] benchmarks. It is essential to note that the SigLIP variants we employed have higher input resolutions (384 vs. 336) and more visual tokens (729 vs. 576) compared to CLIP. These factors may contribute to SigLIP containing more beneficial visual information to perform fine-grained image understanding.

Discussion An intriguing observation is that when employing the share recipe, model variants with StableLM-2 and Phi-2 exhibit a significant decline in performance on POPE (indicating more hallucinations) while experiencing improvements on other benchmarks. Compared to the base recipe, we note that the share recipe significantly increases the number of trainable parameters during the pre-training stage,

Preliminary Exploration of Connectors. We offer preliminary insights into connectors by comparison between MLP and resampler. Our observations shown in Figure 6

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

(a) TinyLlama

(b) StableLM-2

(c) Phi-2

- Figure 7. Ablation of training datasets. We fix the vision encoder to CLIP and train our model variants with two datasets under the base recipe. The titles of the subplots indicate the corresponding to the LLM backbones.

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

(a) TinyLlama

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

(b) StableLM-2

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

(c) Phi-2

- Figure 8. Ablation of training recipes. We set CLIP as the vision encoder and train our model variants under the two training recipes. The titles of the subplots indicate the corresponding to the LLM backbones.

which may be a key factor contributing to these observed phenomena. From the above phenomena, we conjecture that model variants with smaller LLMs may require more trainable parameters to fit larger datasets well in the pre-training stage. Therefore, having more trainable parameters enables model variants with TinyLlama to achieve better results on ShareGPT4V. However, using more trainable parameters during pre-training may not be entirely benign for larger models. For instance, while model variants with StableLM-2 and Phi-2 generally exhibit improved performance, worse performance in handling hallucinations is also introduced.

Summary. In this part, we observe that training model variants on larger and more diverse data enables them to achieve overall better performance. Besides, model variants with smaller LLMs may require more trainable parameters to decrease hallucinations, while for variants with larger LLMs, using more trainable parameters leads to more hallucinations.

##### 4.2.3 Overall Comparison among Different LMMs.

Comparison among TinyLLaVA Variants. Here, we thoroughly compare various variants of TinyLLaVA models

(See more details in Table A1 of Appendix). For reference, we name the variants under three design axes: training recipe, vision encoder, and language model, and format their names as TinyLLaVA-{recipe name}-{vision encoder}-{language model}. For instance, TinyLLaVA-base-C-TL is interpreted as trained under the base recipe, with CLIP and TinyLlama as backbones. We find that smaller TinyLLaVA variants can achieve results comparable to their larger counterparts when using appropriate combinations of data and training recipes, as shown in Figure 9.

Comparison with other LMMs. Finally, we compare our TinyLLaVA variants to the state-of-the-art LMMs as shown in Table 3. Our TinyLLaVA-share-Sig-Phi with 3.1B parameters achieves comprehensive superiority over LLaVA-1.5 [37] with 7B parameters. It is worth noting that TinyLLaVA-share-Sig-Phi achieves comparable results to MoE-LLaVA [35] on VQAv2 [17] with fewer parameters and outperforms it in terms of POPE accuracy [55]. These findings highlight the promising potential of thorough explorations into the design space of LMMs.

Summary. In this part, we observe that smaller model vari-

(a) (b)

Figure 9. Instances where TinyLLaVAs with smaller parameters outperform their counterpart with larger parameters.

Table 3. Comparison with SOTA LMMs on image understanding benchmarks. ”L”, ”V”, ”Q”, and ”ML” respectively represent LlaMA, Vicuna, Qwen, and MobileVLM. Other abbreviations can be found in Table 1. ∗ donates the training images of the datasets observed during training. The best and second best results are indicated by boldface and underline, respectively.

|Method|LLM<br><br>|Size|Res.<br><br>|Image Question Answering VQAv2 GQA SQAI VQAT<br><br>|Benchmark Toolkit MM-Vet POPE LLaVA-W MME MMB|
|---|---|---|---|---|---|
|I-9B [29] InstructBLIP [12] LLaVA-1.5 [37] Qwen-VL [4] MoE-LLaVA [35] MoE-LLaVA [35] LLaVA-Phi [63] MobileVLM [10]|L-7B V-7B V-7B Q-7B Phi2-2.7B Phi2-2.7B Phi2-2.7B ML-2.7B<br><br>|9B 8.2B 7B 7B 3.9B 3.9B 3.0B 3.0B<br><br>|224 224 336 448 336 384 336 336<br><br>|50.9 38.4 - 25.9<br><br>- 49.2 60.5 50.1 78.5∗ 62.0∗ 66.8 58.2<br><br>78.8∗ 59.3∗ 67.1 63.8 77.6∗ 61.4∗ 68.5 51.4<br><br>79.9∗ 62.6∗ 70.3 57.0 71.4∗ - 68.4 48.6<br><br><br>- 59.0∗ 61.0 47.5<br>|- - - - 48.2<br><br>26.2 - 60.9 - 36<br><br>30.5 85.9 63.4 1510.7 64.3 - - - - 38.2<br><br>34.3 86.3 94.1 - 65.5<br><br>35.9 85.7 97.3 - 68.0<br><br><br>28.9 85.0 - 1335.1 59.8 - 84.9 - 1288.9 59.6<br><br>|
|TinyLLaVA-share-C-Phi TinyLLaVA-share-Sig-Phi|Phi2-2.7B Phi2-2.7B<br><br>|3.0B<br><br>3.1B<br><br><br>|336 384|77.7∗ 61.0∗ 70.1 53.5 79.9∗ 62.0∗ 69.1 59.1<br><br>|31.7 86.3 67.1 1437.3 68.3<br><br>32.0 86.4 75.8 1464.9 66.9<br><br><br>|

ants can achieve results comparable to their larger counterparts when using appropriate combinations of data and training recipes. Meanwhile, our best model, TinyLLaVA3.1B, achieves better overall performance against existing 7B models such as LLaVA-1.5 and Qwen-VL.

### 5. Conclusion

We propose the TinyLLaVA framework, which provides a unified perspective in designing and analyzing the smallscale LMMs. In our experiments, while under the same settings larger models perform better than smaller ones, we prove that with a better quality of data combined with better training recipes, smaller LMMs can consistently achieve on-par performances compared to bigger ones. Using results from our ablations, we train our best model, TinyLLaVA3.1B, which achieves better overall performance against

existing 7B models. Our findings suggest that the design space of LMMs are vastly under-explored. We hope our findings can serve as baselines for future research in terms of data scaling, training setups, and model selections.

Acknowledgement This work was partially supported by the National Key Research and Development Plan of China under Grant 2022ZD0116310, National Natural Science Foundation of China (Grant No. 62106012), the Fundamental Research Funds for the Central Universities.

### References

[1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

- [2] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. Advances in Neural Information Processing Systems, 35:23716–23736, 2022.
- [3] Anas Awadalla, Irena Gao, Josh Gardner, Jack Hessel, Yusuf Hanafy, Wanrong Zhu, Kalyani Marathe, Yonatan Bitton, Samir Gadre, Shiori Sagawa, et al. Openflamingo: An opensource framework for training large autoregressive visionlanguage models. Technical report, 2023.
- [4] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. arXiv preprint arXiv:2308.12966, 2023.
- [5] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.
- [6] Jun Chen, Deyao Zhu, Xiaoqian Shen, Xiang Li, Zechun Liu, Pengchuan Zhang, Raghuraman Krishnamoorthi, Vikas Chandra, Yunyang Xiong, and Mohamed Elhoseiny. Minigpt-v2: large language model as a unified interface for vision-language multi-task learning. arXiv preprint arXiv:2310.09478, 2023.
- [7] Lin Chen, Jisong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, and Dahua Lin. Sharegpt4v: Improving large multi-modal models with better captions. arXiv preprint arXiv:2311.12793, 2023.
- [8] Jaemin Cho, Jie Lei, Hao Tan, and Mohit Bansal. Unifying vision-and-language tasks via text generation. In International Conference on Machine Learning, pages 1931–1942. PMLR, 2021.
- [9] Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. Palm: Scaling language modeling with pathways. Journal of Machine Learning Research, 24(240):1–113, 2023.
- [10] Xiangxiang Chu, Limeng Qiao, Xinyang Lin, Shuang Xu, Yang Yang, Yiming Hu, Fei Wei, Xinyu Zhang, Bo Zhang, Xiaolin Wei, et al. Mobilevlm: A fast, reproducible and strong vision language assistant for mobile devices. arXiv preprint arXiv:2312.16886, 2023.
- [11] Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al. Scaling instructionfinetuned language models. arXiv preprint arXiv:2210.11416, 2022.
- [12] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. Instructblip: Towards general-purpose visionlanguage models with instruction tuning, 2023.
- [13] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is

- worth 16x16 words: Transformers for image recognition at scale. ICLR, 2021.
- [14] Danny Driess, Fei Xia, Mehdi SM Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, et al. Palm-e: An embodied multimodal language model. arXiv preprint arXiv:2303.03378, 2023.
- [15] Constantin Eichenberg, Sidney Black, Samuel Weinbach, Letitia Parcalabescu, and Anette Frank. Magma–multimodal augmentation of generative models through adapter-based finetuning. arXiv preprint arXiv:2112.05253, 2021.
- [16] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, Yunsheng Wu, and Rongrong Ji. Mme: A comprehensive evaluation benchmark for multimodal large language models. arXiv preprint arXiv:2306.13394, 2023.
- [17] Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the V in VQA matter: Elevating the role of image understanding in Visual Question Answering. In Conference on Computer Vision and Pattern Recognition (CVPR), 2017.
- [18] Dan Hendrycks and Kevin Gimpel. Gaussian error linear units (gelus). arXiv preprint arXiv:1606.08415, 2016.
- [19] Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. Training compute-optimal large language models. arXiv preprint arXiv:2203.15556, 2022.
- [20] https://sharegpt.com/. Sharegpt. 2023.
- [21] Shaohan Huang, Li Dong, Wenhui Wang, Yaru Hao, Saksham Singhal, Shuming Ma, Tengchao Lv, Lei Cui, Owais Khan Mohammed, Barun Patra, et al. Language is not all you need: Aligning perception with language models. Advances in Neural Information Processing Systems, 36, 2023.
- [22] Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6700–6709, 2019.
- [23] Robert A Jacobs, Michael I Jordan, Steven J Nowlan, and Geoffrey E Hinton. Adaptive mixtures of local experts. Neural computation, 3(1):79–87, 1991.
- [24] Andrew Jaegle, Felix Gimeno, Andy Brock, Oriol Vinyals, Andrew Zisserman, and Joao Carreira. Perceiver: General perception with iterative attention. In International conference on machine learning, pages 4651–4664. PMLR, 2021.
- [25] Dakota Mahan Carlos Riquelme Ruiz Jonathan Tow, Marco Bellagente. Stablelm: Stability ai language models. Technical report, 2023.
- [26] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020.
- [27] Siddharth Karamcheti, Suraj Nair, Ashwin Balakrishna, Percy Liang, Thomas Kollar, and Dorsa Sadigh. Prismatic vlms: Investigating the design space of visually-conditioned language models. arXiv preprint arXiv:2402.07865, 2024.

- [28] Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A Shamma, et al. Visual genome: Connecting language and vision using crowdsourced dense image annotations. International journal of computer vision, 123:32–73, 2017.
- [29] Hugo Laurenc¸on, Lucile Saulnier, L´eo Tronchon, Stas Bekman, Amanpreet Singh, Anton Lozhkov, Thomas Wang, Siddharth Karamcheti, Alexander Rush, Douwe Kiela, et al. Obelics: An open web-scale filtered dataset of interleaved image-text documents. Advances in Neural Information Processing Systems, 36, 2023.
- [30] Bo Li, Yuanhan Zhang, Liangyu Chen, Jinghao Wang, Jingkang Yang, and Ziwei Liu. Otter: A multi-modal model with in-context instruction tuning. arXiv preprint arXiv:2305.03726, 2023.
- [31] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip2: Bootstrapping language-image pre-training with frozen image encoders and large language models. arXiv preprint arXiv:2301.12597, 2023.
- [32] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-training for unified visionlanguage understanding and generation. In International Conference on Machine Learning, pages 12888–12900. PMLR, 2022.
- [33] Yuanzhi Li, S´ebastien Bubeck, Ronen Eldan, Allie Del Giorno, Suriya Gunasekar, and Yin Tat Lee. Textbooks are all you need ii: phi-1.5 technical report. arXiv preprint arXiv:2309.05463, 2023.
- [34] Zhang Li, Biao Yang, Qiang Liu, Zhiyin Ma, Shuo Zhang, Jingxu Yang, Yabo Sun, Yuliang Liu, and Xiang Bai. Monkey: Image resolution and text label are important things for large multi-modal models. arXiv preprint arXiv:2311.06607, 2023.
- [35] Bin Lin, Zhenyu Tang, Yang Ye, Jiaxi Cui, Bin Zhu, Peng Jin, Junwu Zhang, Munan Ning, and Li Yuan. Moe-llava: Mixture of experts for large vision-language models. arXiv preprint arXiv:2401.15947, 2024.
- [36] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C. Lawrence Zitnick. Microsoft coco: Common objects in context. In ECCV, 2014.
- [37] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. arXiv preprint arXiv:2310.03744, 2023.
- [38] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36, 2023.
- [39] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? arXiv preprint arXiv:2307.06281, 2023.
- [40] Pan Lu, Swaroop Mishra, Tony Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. In The 36th Conference on Neural Information Processing Systems (NeurIPS), 2022.

- [41] Senior Researcher S´ebastien Bubeck Mojan Javaheripi. Phi-2: The surprising power of small language models. Technical report, 2023.
- [42] OpenAI. Chatgpt: Openai’s gpt-based conversational agent. https://openai.com/chatgpt, 2022.
- [43] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35:27730–27744, 2022.
- [44] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021.
- [45] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2019.
- [46] Shaden Smith, Mostofa Patwary, Brandon Norick, Patrick LeGresley, Samyam Rajbhandari, Jared Casper, Zhun Liu, Shrimai Prabhumoye, George Zerveas, Vijay Korthikanti, et al. Using deepspeed and megatron to train megatron-turing nlg 530b, a large-scale generative language model. arXiv preprint arXiv:2201.11990, 2022.
- [47] Stability AI Language Team. Stable lm 2 1.6b.
- [48] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.
- [49] Maria Tsimpoukelli, Jacob L Menick, Serkan Cabi, SM Eslami, Oriol Vinyals, and Felix Hill. Multimodal few-shot learning with frozen language models. Advances in Neural Information Processing Systems, 34:200–212, 2021.
- [50] Zirui Wang, Jiahui Yu, Adams Wei Yu, Zihang Dai, Yulia Tsvetkov, and Yuan Cao. Simvlm: Simple visual language model pretraining with weak supervision. arXiv preprint arXiv:2108.10904, 2021.
- [51] Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, et al. Emergent abilities of large language models. Transactions on Machine Learning Research, 2022.
- [52] Zi Lin Ying Sheng Zhanghao Wu Hao Zhang Lianmin Zheng Siyuan Zhuang Yonghao Zhuang Joseph E. Gonzalez Ion Stoica Wei-Lin Chiang, Zhuohan Li and Eric P. Xing. Vicuna: An open-source chatbot impressing gpt-4 with 90Technical report, 2023.
- [53] Aiyuan Yang, Bin Xiao, Bingning Wang, Borong Zhang, Ce Bian, Chao Yin, Chenxu Lv, Da Pan, Dian Wang, Dong Yan, et al. Baichuan 2: Open large-scale language models. Technical report, 2023.
- [54] Qinghao Ye, Haiyang Xu, Guohai Xu, Jiabo Ye, Ming Yan, Yiyang Zhou, Junyang Wang, Anwen Hu, Pengcheng Shi,

- Yaya Shi, et al. mplug-owl: Modularization empowers large language models with multimodality. arXiv preprint arXiv:2304.14178, 2023.
- [55] Kun Zhou Jinpeng Wang Wayne Xin Zhao Yifan Li, Yifan Du and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. In The 2023 Conference on Empirical Methods in Natural Language Processing, 2023.
- [56] Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities. arXiv preprint arXiv:2308.02490, 2023.
- [57] Zhengqing Yuan, Zhaoxu Li, and Lichao Sun. Tinygpt-v: Efficient multimodal large language model via small backbones. arXiv preprint arXiv:2312.16862, 2023.
- [58] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 11975–11986, October 2023.
- [59] Peiyuan Zhang, Guangtao Zeng, Tianduo Wang, and Wei Lu. Tinyllama: An open-source small language model. arXiv preprint arXiv:2401.02385, 2024.
- [60] Bo Zhao, Boya Wu, and Tiejun Huang. Svit: Scaling up visual instruction tuning. arXiv preprint arXiv:2307.04087, 2023.
- [61] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging llm-as-a-judge with mt-bench and chatbot arena, 2023.
- [62] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592, 2023.
- [63] Yichen Zhu, Minjie Zhu, Ning Liu, Zhicai Ou, Xiaofeng Mou, and Jian Tang. Llava-phi: Efficient multi-modal assistant with small language model. arXiv preprint arXiv:2401.02330, 2024.

### A. Brief Overviews of Evaluation Benchmark.

Here, we provide a brief overview of the key aspects each benchmark focuses on when assessing model capabilities.

- • VQAv2 [17] contains image-question-answer tuples with images collected from the COCO dataset [36]. The test set of VQAv2 evaluates models’ capabilities in terms of visual recognition, visual grounding, spatial reasoning as well as language understanding.
- • GQA [22] collected its data according to the scene graph structure provided by the Visual Genome [28] dataset. The test set of GQA extensively evaluates models’ capabilities in terms of visual and compositional reasoning.
- • TextVQA [45] is an image question answering dataset that contains images with texts. The test set of TextVQA requires models to not only recognize textual information in the given images but also to reason over them.
- • ScienceQA-IMG [40] is a subset of the ScienceQA [40] benchmark that contains images. The benchmark contains scientific questions and answers collected from lectures and textbooks. During the evaluation, the model is prompted with questions, choices, and relevant contexts, and is asked to predict the correct answers. This benchmark mainly evaluates models’ capabilities in reasoning with respect to scientific knowledge.
- • POPE [55] benchmark is designed to evaluate the hallucination issues in LMMs. Its test samples incorporate positive and negative objects (non-existent objects), which require the model to not only recognize positive samples accurately but also correctly identify negative samples (measuring hallucination). It effectively assesses the model’s ability to handle hallucinations.
- • MM-Vet [56] is a comprehensive benchmark that evaluates LMMs on complicated multimodal tasks. MM-Vet uses GPT-4 [1] to evaluate the outputs generated by LMMs. Its test set evaluates LMMs on six dimensions: visual recognition, spatial reason- ing, common knowledge deduction, language generation, visual math reasoning, and OCR recognition.
- • LLaVA-W benchmark includes 24 images and 60 questions, which are collected to evaluate LMMs’ capabilities in challenging tasks and generalizability in novel domains [38].
- • MME is a LMM evaluation benchmark that measures both perception and cognition abilities on a total of 14 subtasks [16]. This benchmark is automatically evaluated by GPT-4 [1].
- • MMBench is a LMM evaluation benchmark that comprehensively assess models’ capabilities across 20 dimensions [39]. This benchmark is automatically evaluated by ChatGPT [42].

### B. TinyLLaVA Variants.

We show all TinyLLaVA variants in Table A1. The results suggest that enhancing overall performance is attainable through the application of larger models, diverse datasets, and meticulously crafted training recipes.

Table A1. Comprehensive comparison among our TinyLLaVA variants. We trained base on the LLaVA-1.5 dataset and the share recipe on the ShareGPT4V dataset. The best results and second best results are indicated by boldface and underline, respectively.

|Method|LLM|Size<br><br>|Res.<br><br>|Image Question Answering VQAv2 GQA SQAI VQAT|Benchmark Toolkit MM-Vet POPE LLaVA-W<br><br>|
|---|---|---|---|---|---|
|TinyLLaVA-base-C-TL TinyLLaVA-base-Sig-TL TinyLLaVA-base-C-SLM TinyLLaVA-base-Sig-SLM TinyLLaVA-base-C-Phi TinyLLaVA-base-Sig-Phi|TL-1.1B TL-1.1B SLM-1.6B SLM-1.6B Phi2-2.7B Phi2-2.7B<br><br>|1.4B<br><br>1.5B<br><br><br>1.9B<br><br>2.0B<br><br>3.0B 3.1B<br><br><br>|336 384 336 384 336 384|74.0 58.1 60.2 45.8<br><br>75.8 58.6 60.2 49.1<br><br>74.9 58.9 62.8 49.5<br><br>78.1 61.1 62.8 54.1<br><br>76.6 60.3 69.6 51.4<br><br>79.2 61.3 69.9 55.6<br><br><br><br><br>|20.6 85.9 55.8<br><br>24.1 87.1 59.0<br>25.0 85.8 55.4<br><br><br>29.5 86.9 59.8<br><br>30.7 86.8 67.1 32.1 87.1 67.9<br><br><br>|

TinyLLaVA-share-C-TL TL-1.1B 1.4B 336 75.2 59.4 58.8 48.1 25.1 85.9 55.6 TinyLLaVA-share-Sig-TL TL-1.1B 1.5B 384 76.9 60.3 60.1 51.7 25.8 86.1 60.8 TinyLLaVA-share-C-SLM SLM-1.6B 1.9B 336 76.3 60.3 63.1 51.6 29.3 85.4 59.5 TinyLLaVA-share-Sig-SLM SLM-1.6B 2.0B 384 78.9 61.9 64.7 56.4 32.6 86.5 66.4 TinyLLaVA-share-C-Phi Phi2-2.7B 3.0B 336 77.7 61.0 70.1 53.5 31.7 86.3 67.1 TinyLLaVA-share-Sig-Phi Phi2-2.7B 3.1B 384 79.9 62.0 69.1 59.1 32.0 86.4 75.8

Here, we provide some examples generated by our TinyLLaVA-3.1B.

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

Explain this meme.

List how much each item costs.

[Figure 21]

[Figure 22]

[Figure 23]

The image captures a moment of tranquility featuring a small, light brown dog. The dog is lying on its side on a wooden floor, its head comfortably resting on its front paws. The dog's relaxed posture suggests it is in a state of rest or perhaps even sleep. Overlaying the image is a text that reads "MONDAY, just...MONDAY." The text is written in a casual, handwritten style and is in a contrasting color, making it stand out against the image. The repetition of the phrase "MONDAY" emphasizes the theme of the image, which seems to be a commentary on the monotony or routine associated with the start of the work week. The overall composition of the image, with the dog in the foreground and the text in the background, draws the viewer's attention to both the subject and the message. The image, therefore, serves as a visual representation of the common sentiment associated with the start of the work week.

The image displays a map of a city with two green dots on it. Below the map, there is a table that provides details about the order. The table includes the following information:

[Figure 24]

- 1.Base Fare: $10
- 2.Mileage Fee: $12.11
- 3.Duration Fee: $10.32
- 4.Total: $33.42
- 5.Discount: $3.24
- 6.Actually Paid: $29.42

[Figure 25]

What is the area of the parallelogram? Reason through your process.

[Figure 26]

To find the area of the parallelogram, we need to use the formula: Area = base x height. In the image, the base of the parallelogram is 6 units, and the height is 4 units. Therefore, the area of the parallelogram is 6 x 4 = 24 square units.

The table also includes a note that reads "Actual Paid", which indicates that the actual amount paid for the order is $29.42. The other details in the table are related to the order's base fare, mileage fee, duration fee, and total cost.

(a) TinyLLaVA-3.1B can perceive detailed OCR content.

(b) TinyLLaVA-3.1B can understand and explain memes.

(c) TinyLLaVA-3.1B can visually perceive and solve math problems.

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

What is unusual about this image?

[Figure 35]

What do we know from this image?

What is unusual about this image?

[Figure 36]

[Figure 37]

When visiting this serene lakeside location with a wooden pier, there are a few things to be cautious about. First, ensure that the pier is stable and safe to walk on, as it may be slippery or uneven due to the wet conditions. Second, be mindful of the water depth and any potential hazards, such as rocks or submerged objects, that could pose a risk while walking or swimming. Third, pay attention to the weather conditions, as sudden changes in weather can make the area more dangerous. Lastly, respect the natural environment and wildlife, and avoid littering or disturbing the ecosystem. Enjoy the peaceful atmosphere and take necessary precautions to ensure a safe and enjoyable experience.

[Figure 38]

The image you've shared is a page from a research paper. The title of the paper is "Abstract" and it's presented by TinyLava Framework. The paper discusses the design and analysis of the small-scale large-multimodal models (LMMS) and their effects on different vision encoders, connection models, language models, training data, and training recipes. The paper also highlights the benefits of using TinyLava's best model, TinyLava 3.1.B, in comparison to existing models. The paper suggests that TinyLava's findings can serve as a starting point for future research in terms of data scaling and model selections. The text in the image is in English.

The unusual aspect of this image is that a man is ironing clothes on top of a yellow taxi cab, which is not a typical place to perform such a task. Ironing clothes on a moving vehicle can be dangerous and distracting for the driver, posing a risk to the safety of the passengers and other road users. Additionally, it is not common to see someone ironing clothes in an urban setting, as it is usually done in a more private and controlled environment like a home or an office.

(d) TinyLLaVA-3.1B can explain unrealistic scenes.

(e) TinyLLaVA-3.1B can understand the paper’s abstract and summarize.

###### (f) TinyLLaVA-3.1B can generate detailed and accurate descriptions.

