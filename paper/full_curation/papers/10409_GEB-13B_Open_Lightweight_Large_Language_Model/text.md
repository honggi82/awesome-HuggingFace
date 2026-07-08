## GEB-1.3B: Open Lightweight Large Language Model

Jie Wu, Yufeng Zhu, Lei Shen, Xuqing Lu GEB

# arXiv:2406.09900v1[cs.CL]14Jun2024

Abstract

Recently developed large language models (LLMs) such as ChatGPT, Claude, and Llama have demonstrated impressive abilities, and even surpass human-level performance in several tasks. Despite their success, the resourceintensive demands of these models, requiring significant computational power for both training and inference, limit their deployment to high-performance servers. Additionally, the extensive calculation requirements of the models often lead to increased latency in response times. With the increasing need for LLMs to operate efficiently on CPUs, research about lightweight models that are optimized for CPU inference has emerged. In this work, we introduce GEB-1.3B, a lightweight LLM trained on 550 billion tokens in both Chinese and English languages. We employ novel training techniques, including ROPE, Group-QueryAttention, and FlashAttention-2, to accelerate training while maintaining model performance. Additionally, we fine-tune the model using 10 million samples of instruction data to enhance alignment. GEB-1.3B exhibits outstanding performance on general benchmarks such as MMLU, C-Eval, and CMMLU, outperforming comparative models such as MindLLM-1.3B and TinyLLaMA-1.1B. Notably, the FP32 version of GEB-1.3B achieves commendable inference times on CPUs, with ongoing efforts to further enhance speed through advanced quantization techniques. The release of GEB-1.3B as an open-source model marks a significant contribution to the development of lightweight LLMs, promising to foster further research and innovation in the field.

### 1 Introduction

Large language models have experienced significant advancements, achieving superhuman capabilities in numerous specific tasks. These models, such as ChatGPT 3.5 (OpenAI, 2023), GPT-4 (OpenAI, 2023), and Claude (Claude, 2023), demonstrate remarkable success across various languages,

particularly in English. On the other hand, models such as Baichuan (Yang et al., 2023a), ChatGLM (Zeng et al., 2022), and Qwen (Bai et al., 2023) are mainly used for Chinese. However, these powerful models require a lot of computer resources, like GPUs or TPUs, for both training and inference phases. Consequently, these models are typically operational on remote high-performance servers. Furthermore, the vast computations required by the models bring potential delays in responding. There exists a significant demand for models that can efficiently run on CPUs, thereby enabling deployment on more accessible devices such as laptops and smartphones. In response to this need, researchers are now focusing on the development of lightweight large language models, aiming to reduce both response times and hardware costs. Remarkably, the Llama 2 model (Touvron et al., 2023) series demonstrates that, when trained with massive data, smaller models can surpass their larger counterparts in performance. Despite being trained on trillions of tokens, the Llama 2 series shows no signs of training loss saturation, suggesting untapped potential in smaller model capacities.

In this work, we introduce GEB-1.3B, a lightweight model featuring 1.3 billion parameters and trained on 550 billion tokens in both Chinese and English languages. We incorporate advanced techniques such as ROPE (Su et al., 2024), Group-Query-Attention (Ainslie et al., 2023), and FlashAttention-2 (Dao et al., 2022) to expedite our training process. Additionally, we fine-tune the model using 10 million instruction-based samples to enhance its alignment.

The evaluation results indicate that GEB1.3B outperforms comparable models such as MindLLM-1.3B(Yang et al., 2023b) and TinyLLaMA-1.1B(Zhang et al., 2024) on general benchmarks, including MMLU(Hendrycks et al., 2020), C-Eval(Huang et al., 2024), and CMMLU(Li et al., 2023). Moreover, the inference

Language Dataset Proportion Size Tokens English

C4 33.1% 493G 182B Github+StackExchange 14.7% 245G 81B

CommonCrawl 28.4% 700G 156B WuDaoCorpus 8.9% 200G 49B

Chinese

SkyPile 14.9% 350G 82B

Table 1: The details of the datasets.

time of the FP32 version on CPUs is sufficiently rapid for practical applications. And we plan to explore further acceleration through quantization in the future.

We are releasing GEB-1.3B to the general public for research use at https://huggingface.

co/GEB-AGI/geb-1.3b. The model has demonstrated superior performance than other lightweight models of the similar parameter size, and even surpass some larger counterparts. We believe that the release of this lightweight LLM will benefit further research.

The remainder of this paper describes the pretraining methodology (Section 2), alignment technology (Section 3), evaluations on benchmarks (Section 4), and conclusions (Section 5).

### 2 Pre-training

#### 2.1 Data

To ensure the coverage, diversity, and quality of foundational datasets for our 1.3 billion parameter model, we employ multifaceted approaches in data collection and processing.

#### 2.1.1 Data Collection

English Pre-training Dataset Our English pretraining dataset predominantly originates from C4 (Raffel et al., 2020), which contains over 15.6 billion tokens from 365 million internet domains. To further augment the model’s logical capability, we integrate datasets from GitHub and Stack Exchange during the pre-training phase, including code snippets, question-and-answer exchanges, and procedural instructions.

Chinese Pre-training Dataset In contrast to the English counterpart, there are few high-quality, large-scale Chinese corpora suitable for pretraining. To bridge this gap, we construct a comprehensive large-scale Chinese pre-training corpus from the Common Crawl dataset, meticulously ensuring both quality and diversity. This corpus adheres to the processing principles applied to the C4

dataset, with additional, tailored cleaning protocols for the Chinese language, as elaborated in Subsection 2.1.2. Moreover, we incorporate two publicly accessible high-quality Chinese corpora - WuDaoCorpus (200G) (Yuan et al., 2021) and SkyPile (Wei et al., 2023) - as ancillary data sources, thereby enriching our foundational dataset. Table 1 shows the details of the datasets.

#### 2.1.2 Data Processing

The Common Crawl dataset is crawled from the web and often suffers from quality issues, because of including garbled text, incomplete URL links, extraneous emojis, fragmented sentences, personal contact details, sensitive expressions, and profanity. To obtain informative and diverse data, we undertake multiple stages of processing.

Rule Cleaning First, we extract a Chinese corpus from the Common Crawl dataset, by applying specialized rules to remove non-essential HTML, CSS, and JavaScript identifiers, garbled text, and other aberrant symbols. We eliminate incomplete sentences by identifying terminal punctuation and applying a sentence length threshold. Regular expressions are used to remove personal information and URL links, while a word list screens out sentences with sensitive language, profanity, or promotional content.

High-Quality Corpus Filtering We leverage perplexity (PPL) and keyword density filtering techniques to get rid of low-quality content and advertising. Given that perplexity inversely correlates with sentence fluency, we establish a perplexity threshold to exclude samples failing to meet our quality standards. Additionally, we employ a keyword density-based filtering method to remove sentences with insufficient knowledge density, setting a predetermined threshold to identify and discard the samples below the threshold. Recognizing that internetsourced corpora often consist of paragraphs with a single sentence, we utilize a classification model to concatenate highly correlated sentences into cohesive paragraphs.

#### Size Vocabulary Size Hidden Size FFN Size Heads Layers KV Groups Length 1.3B 64896 2048 5632 16 24 4 4096

Table 2: The details of GEB-1.3B.

Deduplication Our deduplication process involves removing instances of consecutive duplicate sentences and paragraphs in each dataset. Further comparisons across different datasets help eliminate redundant paragraphs, ensuring the uniqueness of the information within our corpus. Through these strict rules and filtering methods, we have compiled a 1.3TB pre-training dataset, which includes 700GB of high-quality Chinese data extracted from the 50TB Common Crawl dataset.

#### 2.2 Architecture

The architecture of GEB-1.3B is grounded in the Transformers framework, yet it incorporates several advanced techniques to improve the performance. Herein, we show the details of these adaptations:

Tokenization There are numerous vocabularies available, eliminating the need to retrain the tokenizer model. We decide to exclude the original Llama tokenizer model due to its suboptimal tokenization efficiency for Chinese tasks. By building upon the ChatGLM-3 tokenizer model, we develop a new vocabulary containing 64,896 entries. This is different from the vocabularies of Qwen and Baichuan2, which both exceed 100,000 entries and are deemed too large. The rationale behind this strategy is to utilize a more compact vocabulary, thereby reducing the number of model parameters. This approach aims to enhance the model’s inference speed without sacrificing its performance.

Word Embedding Layer Through experiments, we employ an untied embedding strategy, diverging from the conventional practice of tying the weights of the input embedding to the output layer. This choice, though increasing memory requirements, is justified by the resultant performance gains. The Rotary Positional Embedding (RoPE) method (Su et al., 2024), serves as our primary mechanism for incorporating positional information into the model. This approach is widely used in modern large-scale language models. Furthermore, we utilize FP32 for generating position IDs, a measure designed to prevent position collisions that may arise from inadequate precision.

Transformer Blocks Our modifications to the

transformer blocks are aimed at optimizing training efficiency and stability, particularly within the constraints of limited GPU resources. We choose the Group-Query-Attention (GQA) (Ainslie et al., 2023) mechanism rather than the conventional multi-head attention (MHA) (Vaswani et al., 2017) mechanism. This innovation significantly enhances inference speed without detracting from accuracy, by grouping the heads of K and V to share a singular Q. The SwiGLU (Shazeer, 2020) activation function, a hybrid of Swish (Ramachandran et al., 2017) and Gated Linear Unit (Dauphin et al., 2017), is selected for its efficacy. The dimension of the Feed-Forward Networks (FFN) is adjusted from four times the hidden sizes to a ratio of 8/3. PostRMSNorm (Ba et al., 2016) is implemented instead of traditional normalization methods, providing stronger regularization and facilitating improved convergence during the training phase. Moreover, all biases within the transformer blocks are eliminated.

The details of GEB-1.3B are depicted in Table 2.

#### 2.3 Infrastructure

In the pre-training phase, we employ the AdamW (Loshchilov and Hutter, 2017) optimizer, setting the hyperparameters to β1 = 0.9, β2 = 0.95, and ϵ = 10e–8. The learning rate follows a cosine decay schedule, with a minimum at 4e-5 and a peak at 4e-4. To ensure stability throughout the training, we utilize BFloat16 mixed precision method. The computational resources are limited to 64 NVIDIA RTX3090ti GPUs. Due to CUDA memory constraints, we establish a global batch size of 320 samples, accommodating a model sequence length of 4096 tokens. Such a small batch size makes training unstable, leading to frequent loss spikes during the pre-training stage. To mitigate this, we implement following strategies and the loss curves are shown in Figure 1:

Batch Sample Replacement: We observe that a diverse sample distribution within a batch may lead to a spike in loss. To address this, we replace the affected batch with an alternative batch when a spike occurs.

[Figure 1]

[Figure 2]

Figure 1: Loss curves before and after adopting four measures

Skipping Specific Iterations: Drawing inspiration from the practices of PaLM (Chowdhery et al., 2023) and OPT (Zhang et al., 2022) models, we exclude samples from the 100-200 iterations preceding and succeeding a loss spike.

Embedding Layer Gradient Shrink (EGS): Building on the theoretical insights provided by Molybog (Molybog et al., 2023), we identify that the emergence of a loss spike is associated with sudden increases in the gradient updates of the shallow layers. This surge in gradients can form a chain reaction, affecting the update status of parameters in the deeper layers of the model and leading it into a non-stationary state (Zeng et al., 2022). So we directly multiply the shallow gradient by the scaling coefficient α to reduce the update value of the shallow gradient.

Adjusting Learning Rate: When reducing the batch size, it is imperative to correspondingly decrease the learning rate to prevent oscillations. Conversely, as the batch size increases, adjusting the learning rate upwards is necessary to expedite convergence. With our global batch size setting at only 320, reducing the learning rate accordingly is es-

C-Eval Scores by Model and Category

| |MindLLM-1.3B<br><br>Llama-7B GEB-1.3B<br><br>| | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

35

30

25

Scores

20

15

10

5

0

STEM Social Sciences Humanities Others Categories

Figure 2: The results on C-Eval benchmark.

sential to avoid loss spikes. Moreover, we have observed that increasing the decay rate of the learning rate can effectively prevent loss spikes.

### 3 Alignment

The pretrained large language models only learn the knowledge of the world, but there is still a gap between the model generations and human expectations. The most recent advances of alignment are Supervised Fine-Tuning (SFT) (Ouyang et al., 2022) and Direct Preference Optimization (DPO) (Rafailov et al., 2024). By adopting Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO), the GEB-1.3B model is trained to align more closely with human conversational modes.

#### 3.1 Supervised Fine-Tuning

Our SFT dataset comprises approximately 16 million instances of instructional data, spanning a wide list of subjects that range from benign and useful topics to a sensitive one. This collection includes an expansive variety of domains, such as general linguistic tasks, mathematical problemsolving, and programming exercises. On the other hand, the safety-related portion of our dataset encompasses a broad spectrum of sensitive topics. In addition, we craft a series of prompts designed for various tasks, aiming to enhance the model’s capability for generalization. This approach ensures both the diversity and coverage of the dataset.

#### Model MMLU C-Eval CMMLU Average

Baichuan-7B 42.30 42.80 44.02 43.04 ChatGLM-6B 40.63 38.90 - 39.77 GEB-1.3B 31.20 33.30 32.20 32.23 Llama-7B 35.10 27.10 26.75 29.65 Falcon-7B 28.00 - - 28.00 MPT-7B 27.93 27.15 26.00 27.03 MindLLM-1.3B 26.20 26.10 25.33 25.88 MindLLM-3B 26.20 25.70 25.00 25.63 TinyLlama-1.1B 25.34 25.02 24.03 24.80

Table 3: The results on MMLU, C-Eval, and CMMLU benchmarks.

#### 3.2 Direct Preference Optimization

We use DPO to align the model with human expectations, in order to guide the model to avoid answer unethical questions and generate harmless answers. The dataset size is only 10k for DPO, which is rather small comparing to the SFT dataset.

- 4 Evaluations

#### 4.1 General Benchmarks

In our evaluation, we compare our model with several well-known large language models, including Llama2-7B, Baichuan-7B, Falcon-7B, MPT7B, and ChatGLM-6B. Additionally, we extend our comparison to the models such as MindLLM1.3B (Yang et al., 2023b) and TinyLLaMA1.1B (Zhang et al., 2024), whose parameter size is comparable to our own. This multifaceted comparison enables a comprehensive understanding of our model’s performance.

We evaluate the models on three general benchmarks, MMLU (Hendrycks et al., 2020), CEval (Huang et al., 2024), and CMMLU (Li et al., 2023). The MMLU benchmark, which evaluates the English knowledge, serves as a key metric for assessing the models’ performance in English. Conversely, the C-Eval and CMMLU benchmarks focus on Chinese language proficiency.

The results on the three benchmarks are shown in table 3. As table 3 reveals, GEB-1.3B significantly outperforms the LLaMA-7B model. Compared to models of similar scale, such as MindLLM-1.3B and TinyLLaMA-1.1B, the GEB-1.3B model is notably superior, demonstrating exceptional capabilities. In contrast to the LLaMA-7B model, the GEB-1.3B model shows significantly better performance in Chinese, while its English language performance, though slightly lower, is still remark-

Model ToxiGen GEB-1.3B 3.00 Falcon-7B 7.89 MPT-instruct 16.33 Llama2-7B 21.25

Table 4: The results on the ToxiGen dataset.

able. Overall, the comparison not only highlights the GEB-1.3B model’s strong performance in both languages but also underscores its superiority over other lightweight models.

Figure 2 displays GEB-1.3B, MindLLM-1.3B, and Llama-7B’s performances on the C-Eval dataset. GEB-1.3B utilizes a zero-shot Chain of Thought method for answer deduction. GEB-1.3B outperforms MindLLM-1.3B across all fields, and even surpasses Llama-7B which is much larger.

#### 4.2 Toxicity

We also experiment on the ToxiGen (Hartvigsen et al., 2022) dataset, to evaluate the toxicity in the model-generated text. The evaluation covers a variety of models, including GEB-1.3B, Falcon-7B (Almazrouei et al., 2023), MPT-instruct (MosaicML, 2023), and Llama2-7B. Lower scores reflect better performance. As depicted in table 4, despite the small size of the model parameters, GEB-1.3B generates a minimal amount of toxic text. There’s no simple correlation between the model size and its effectiveness on the ToxiGen benchmark.

#### 4.3 Inference Speed

We assess the inference speed of the model in the CPU environment. The current CPU inference time of the FP32 version reaches 12 tokens/second, which means it could be used in edge devices. We plan to quantify the model in the future for further

acceleration.

### 5 Conclusion

In this study, we introduce GEB-1.3B, a novel opensource, lightweight large language model. We detail the methodologies employed in the training and alignment, which are all cutting-edge technologies of large language models. Our experimental findings demonstrate that GEB-1.3B surpasses its counterparts, i.e., the models with similar parameter sizes, on general benchmarks. Furthermore, our results affirm that smaller models, when trained with extensive datasets, can achieve performance comparable to that of larger models. Notably, GEB-1.3B is designed to operate efficiently on CPUs, thereby extending its utility to laptops, smartphones, and other edge devices. This advancement marks a significant stride towards the broader application and accessibility of large language models in various computing environments.

### 6 Limitations and Ethical Considerations

Our model shares common limitations observed in large language models, including the generation of inaccurate information ("hallucinations") and occasional repetitiveness. Because the training data are public data processed automatically, the model may have unhealthy replies. Although we try to avoid these outputs through the SFT and DPO process, it is impossible to eliminate them all. Therefore, users of our models must remain alert regarding these potential issues.

### References

Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebrón, and Sumit Sanghai. 2023. Gqa: Training generalized multi-query transformer models from multi-head checkpoints. arXiv preprint arXiv:2305.13245.

Ebtesam Almazrouei, Hamza Alobeidli, Abdulaziz Alshamsi, Alessandro Cappelli, Ruxandra Cojocaru, Mérouane Debbah, Étienne Goffinet, Daniel Hesslow, Julien Launay, Quentin Malartic, et al. 2023. The falcon series of open language models. arXiv preprint arXiv:2311.16867.

Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. 2016. Layer normalization. arXiv preprint arXiv:1607.06450.

Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. 2023. Qwen technical report. arXiv preprint arXiv:2309.16609.

Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. 2023. Palm: Scaling language modeling with pathways. Journal of Machine Learning Research, 24(240):1–113.

Claude. 2023. Claude. https://claude.ai.

Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. 2022. Flashattention: Fast and memory-efficient exact attention with io-awareness. Advances in Neural Information Processing Systems, 35:16344–16359.

Yann N Dauphin, Angela Fan, Michael Auli, and David Grangier. 2017. Language modeling with gated convolutional networks. In International conference on machine learning, pages 933–941. PMLR.

Thomas Hartvigsen, Saadia Gabriel, Hamid Palangi, Maarten Sap, Dipankar Ray, and Ece Kamar. 2022. Toxigen: A large-scale machine-generated dataset for adversarial and implicit hate speech detection. arXiv preprint arXiv:2203.09509.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2020. Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300.

Yuzhen Huang, Yuzhuo Bai, Zhihao Zhu, Junlei Zhang, Jinghan Zhang, Tangjun Su, Junteng Liu, Chuancheng Lv, Yikai Zhang, Yao Fu, et al. 2024. C-eval: A multi-level multi-discipline chinese evaluation suite for foundation models. Advances in Neural Information Processing Systems, 36.

Haonan Li, Yixuan Zhang, Fajri Koto, Yifei Yang, Hai Zhao, Yeyun Gong, Nan Duan, and Timothy Baldwin. 2023. Cmmlu: Measuring massive multitask language understanding in chinese. arXiv preprint arXiv:2306.09212.

Ilya Loshchilov and Frank Hutter. 2017. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101.

Igor Molybog, Peter Albert, Moya Chen, Zachary DeVito, David Esiobu, Naman Goyal, Punit Singh Koura, Sharan Narang, Andrew Poulton, Ruan Silva, et al. 2023. A theory on adam instability in large-scale machine learning. arXiv preprint arXiv:2304.09871.

MosaicML. 2023. Introducing mpt-7b: A new standard for open-source, commercially usable llms. Accessed: 2023-03-28.

OpenAI. 2023. ChatGPT. https://openai.com/ chatgpt.

OpenAI. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. 2022. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. 2024. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research, 21(140):1–67.

Prajit Ramachandran, Barret Zoph, and Quoc V Le.

2017. Searching for activation functions. arXiv preprint arXiv:1710.05941.

Noam Shazeer. 2020. Glu variants improve transformer. arXiv preprint arXiv:2002.05202.

Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. 2024. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063.

Hugo Touvron, Louis Martin, Kevin R. Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023. Llama 2: Open foundation and fine-tuned chat models. ArXiv, abs/2307.09288.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. Advances in neural information processing systems, 30.

Tianwen Wei, Liang Zhao, Lichang Zhang, Bo Zhu, Lijie Wang, Haihua Yang, Biye Li, Cheng Cheng, Weiwei Lü, Rui Hu, et al. 2023. Skywork: A more open bilingual foundation model.

Aiyuan Yang, Bin Xiao, Bingning Wang, Borong Zhang, Ce Bian, Chao Yin, Chenxu Lv, Da Pan, Dian Wang, Dong Yan, et al. 2023a. Baichuan 2: Open large-scale language models. arXiv preprint arXiv:2309.10305.

Yizhe Yang, Huashan Sun, Jiawei Li, Runheng Liu, Yinghao Li, Yuhang Liu, Heyan Huang, and Yang Gao. 2023b. Mindllm: Pre-training lightweight large language model from scratch, evaluations and domain applications. arXiv preprint arXiv:2310.15777.

Sha Yuan, Hanyu Zhao, Zhengxiao Du, Ming Ding, Xiao Liu, Yukuo Cen, Xu Zou, Zhilin Yang, and Jie Tang. 2021. Wudaocorpora: A super large-scale chinese corpora for pre-training language models. AI Open, 2:65–68.

Aohan Zeng, Xiao Liu, Zhengxiao Du, Zihan Wang, Hanyu Lai, Ming Ding, Zhuoyi Yang, Yifan Xu, Wendi Zheng, Xiao Xia, et al. 2022. Glm-130b: An open bilingual pre-trained model. arXiv preprint arXiv:2210.02414.

Peiyuan Zhang, Guangtao Zeng, Tianduo Wang, and Wei Lu. 2024. Tinyllama: An open-source small language model. arXiv preprint arXiv:2401.02385.

Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. 2022. Opt: Open pre-trained transformer language models. arXiv preprint arXiv:2205.01068.

