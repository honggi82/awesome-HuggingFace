# arXiv:2404.06393v4[cs.SD]5Nov2024

## MuPT: A Generative Symbolic Music Pretrained Transformer

Xingwei Qu1 3 4∗, Yuelin Bai5∗, Yinghao Ma1 7∗, Ziya Zhou3, Ka Man Lo3, Jiaheng Liu1, Ruibin Yuan1 3, Lejun Min8, Xueling Liu1, Tianyu Zhang9, Xinrun Du1, Shuyue Guo1, Yiming Liang10, Yizhi Li1 4, Shangda Wu11, Junting Zhou12, Tianyu Zheng1, Ziyang Ma13, Fengze Han1, Wei Xue3, Gus Xia8, Emmanouil Benetos7, Xiang Yue1, Chenghua Lin4, Xu Tan14, Stephen W. Huang15 Jie Fu3†, Ge Zhang1 2 6∗ †

1M-A-P, 2University of Waterloo, 3HKUST, 4University of Manchester, 5Shenzhen Institute of Advanced Technology, CAS, 6Vector Institue, 7QMUL, 8MBZUAI, 9MILA, 10Institute of Automation, CAS, 11Central Conservatory of Music, 12PKU, 13SJTU, 14MSRA, 15harmony.ai

https://map-mupt.github.io/

#### Abstract

In this paper, we explore the application of Large Language Models (LLMs) to the pre-training of music. While the prevalent use of MIDI in music modeling is well-established, our findings suggest that LLMs are inherently more compatible with ABC Notation, which aligns more closely with their design and strengths, thereby enhancing the model’s performance in musical composition. To address the challenges associated with misaligned measures from different tracks during generation, we propose the development of a Synchronized Multi-Track ABC Notation (SMT-ABC Notation), which aims to preserve coherence across multiple musical tracks. Our contributions include a series of models capable of handling up to 8192 tokens, covering 90% of the symbolic music data in our training set. Furthermore, we explore the implications of the Symbolic Music Scaling Law (SMS Law) on model performance. The results indicate a promising direction for future research in music generation, offering extensive resources for community-led research through our open-source contributions.

#### 1 Introduction

Large Language Models (LLMs) have experienced remarkable advancements, leading to their broad application across numerous domains. As these models extend into multimodal areas, such as visual and auditory fields, their capability to represent and model complex information, including images (Liu et al., 2023) and speech (Baevski et al., 2020) becomes increasingly critical. However, this expansion also highlights significant challenges that must be addressed. Specifically, the development of effective tokenizers for images and videos, as well as advanced codecs for the audio domain.

In the domain of music, Large Language Models encounter inherent challenges that hinder their effective utilization. Despite achieving state-of-the-art musical performance, as demonstrated by MuseNet (OpenAI, 2021), these models often struggle to capture the structural symmetry essential to aesthetically pleasing music. This issue stems from the use of Musical Instrument Digital Interface (MIDI), which, while effective, poses significant challenges in terms of music’s readability and structural representation.

To tackle this issue, the integration of ABC notation offers a novel approach to overcoming the limitations of MIDI formats. Yuan et al. (2024) advocate for this method, highlighting ABC notation’s readability and structural coherence. Their methodology involves finetuning the LLAMA2 model, leveraging instruction tuning to enhance the model’s musical

* Equal Technical Contributions. † Corresponding Authors.

output capabilities (Touvron et al., 2023b;a). The research overlooks critical tokenization considerations within musical compositions.

In this paper, we aim to propose a training standard with transformer decoder-only architecture for symbolic music generation tasks, which is suitable for single / multi-track music generation. We observe that mismatches between measures can occur by employing the traditional ’next-token-prediction’ paradigm for symbolic data training. This issue arises because ABC notations are generally notated track by track, completing one track before moving on to the next. To address this challenge, we propose SMT-ABC notation to facilitate the model’s learning of how each measure is expressed across various tracks.

Furthermore, we observe that the ABC Notation model benefits from additional epochs in the training phase. This suggests that repeated data positively impacts the model’s performance. To understand this phenomenon, we introduced the SMS Law for repetitive training with symbolic music data. This law explores how scaling up the training data affects the performance of symbolic music generation models, particularly in terms of validation loss. This investigation aims to provide clear insights into the relationship between data repetition and model efficacy, offering guidance for optimizing model training strategies.

In conclusion, our contributions are highlighted as follows:

- • We develop a Long-range Symbolic Music LLM that introduced a foundation model trained on musical notes in ABC notation, with an extended sequence length of 8192 tokens, catering to over 90% of symbolic musical scores we collected.
- • We propose SMT-ABC notation to represent notes, significantly improving the structural integrity and quality of the generated music by maintaining consistent measures within each track.
- • We explore the SMS Law insights for ABC notation. We demonstrate that comprehensive song modeling yields superior performance with a positive correlation between model size and metric improvement. We also reveal unique training epoch dynamics in music repetition and performance enhancement.
- • We will release a suite of state-of-the-art long-range foundation models in the music domain, articulated in ABC notation, along with intermediate training checkpoints to foster community research and innovation in symbolic music modeling.

#### 2 Related work

- 2.1 Music Pre-training

Audio pre-training through the self-supervised learning paradigm has made great progress in speech (Baevski et al., 2020; Hsu et al., 2021; Baevski et al., 2022; Ma et al., 2023b; Yang et al., 2023; Lin et al., 2023), general-purpose audio (Huang et al., 2022; Baade et al., 2022; Chen et al., 2023; 2024), as well as music (Zhu et al., 2021; Dong et al., 2023; Thickstun et al., 2023; Ma et al., 2023a; Li et al., 2023). Two types of self-supervised music pre-training have been explored: non-autoregressive discriminative models and autoregressive generative models. Non-autoregressive discriminative music pre-training performs mask language modelling (MLM) by constructing a pretext task. This kind of training makes models easier to adapt to downstream understanding tasks, such as music tagging, instrument classification, and beat tracking. Autoregressive generative music pre-training models employ a GPT-style framework to generate music, either in codec (Copet et al., 2024) form or in symbolic form (Thickstun et al., 2023; Dong et al., 2023). Previous symbolic music generation models utilize MIDI to model the sequence input and output, showing the ability to generate music given conditions, or unconditional generation. Existing models are limited by not generating long enough music (Thickstun et al., 2023) and limited musicality (Dong et al., 2023). Therefore, long-range symbolic music generation models with data scaling and model scaling need to be explored urgently.

- 2.2 Data Representation for Symbolic Music

Symbolic music representation formats such as MIDI, Humdrum, and ABC notation offer distinct approaches for representing musical information, each with unique advantages and applicability to computational music representation. MIDI, which excels in capturing musical notes and performance, is a popular choice in the music industry and research community(Huang & Yang, 2020; Huang et al., 2019; Lu et al., 2023). However, the complexity and length of MIDI sequences often challenge music models, which limit the preservation of a composition’s full continuity. In contrast, ABC notation stands out for its textual simplicity and compactness, making it particularly suited for Natural Language Processing (NLP) techniques. It can be efficiently processed and analyzed using sequence modeling and pattern recognition algorithms similar to those used in language translation and text generation, enabling automated music generation and retrieval.

ABC notation’s simplicity and broad applicability have prompted research into enhancing music retrieval and generation through deep learning and NLP. In early research, LSTM networks showed promise by producing music similar to traditional and folk styles (Sturm et al., 2016), using ABC notation for automated composition. Following this, TunesFormer (Wu et al., 2023a), a tool based on the Transformer designed for Irish tunes encoded in ABC notation, utilized techniques like bar patching and introduced control codes to craft melodies that meet specific musical forms. abcMLM (Casini et al., 2023), a masked language model, further demonstrated how structured ABC notation can be used to create folklike tunes, highlighting the adaptability of NLP methods and the benefits of using nonautoregressive models in music. Recent studies have started to utilize pre-trained NLP models for converting text to music (Wu & Sun, 2023), showing how these resources can improve the creation of symbolic music. CLaMP (Wu et al., 2023b) introduced a unique method for learning music and text jointly, using a large collection of music-text pairs to better search for and categorize music automatically. Techniques like text dropout and bar patching are examples of how NLP and music encoding are becoming more integrated. In a significant breakthrough, ChatMusician (Yuan et al., 2024) introduced a new approach to incorporating music as a second language for Large Language Models (LLMs), utilizing ABC notation to seamlessly blend music and text, thereby enabling internal music creation and analysis without relying on external multimodal frameworks.

- 2.3 Scaling Law

A wide range of research underscores a significant pattern in language model performance, indicating a power-law relationship between model performance and the increases in both the number of parameters and the size of the training data (Kaplan et al., 2020; Hoffmann et al., 2022; Ghorbani et al., 2021). Scaling law plays a pivotal role in advancing large language models (LLMs), offering a framework to predict the optimal configurations for larger models based on the training logs of their smaller counterparts (Gao et al., 2022).

Further exploration into scaling laws for autoregressive generative modeling by Henighan

- et al. (2020) broadens the applicability of these laws to include not just textual, but also visual and multimodal tasks, as supported by studies in Ghorbani et al. (2021); Hernandez
- et al. (2021); Gao et al. (2022). Such insights are invaluable for developing music generation models, which often blend multiple modalities such as audio, lyrics, and visual elements like album covers or artist photos. This demonstrates a consistent trajectory of performance enhancement concurrent with resource scaling.

The research by Muennighoff et al. (2024), which involves the repetition of the entire pretraining dataset across multiple epochs, presents promising results yet raises questions regarding its effectiveness for musical data. This uncertainty prompts a need for further research into the impact of data repetition strategy by achieving improved outcomes for models engaged in music-related tasks.

#### 3 Method

- 3.1 Model Architecture

MuPT utilizes a standard Transformer model architecture (Vaswani et al., 2023) in a decoderonly setup. Models are trained on a context length of 8192 tokens. We list our MuPT model parameter in Table 1 and utilize several improvements proposed after the original transformer paper. Below, we list the included improvements:

- • SwiGLU Activation: The SwiGLU activation mechanism, represented as (Swish(xW) · xV), is utilized for the MLP (Multi-Layer Perceptron) intermediate activations. This approach significantly surpasses traditional activation functions such as ReLU, GeLU, and Swish in performance (Shazeer, 2020).
- • RMSNorm Each transformer sub-layer, including the attention and feedforward layers, is normalized using RMSNorm as proposed by Zhang & Sennrich (2019)
- • RoPE Embeddings: In contrast to positional encoding (PE) strategy, we use the Rotary Positional Encoding (RoPE) technique, as developed by Su et al. (2023), aimed at enhancing long-context modeling.

- 3.2 SMT-ABC Notation

ABC notation is a widely adopted system for notating music using plain text, and it offers unique advantages when used in conjunction with deep learning models. Its wellstructured text format ensures easy preprocessing, efficient data transmission, and scalability of datasets. The diverse collection of tunes and compositions in ABC notation facilitates learning various musical structures and styles. Moreover, ABC notation allows models to generate human-readable outputs, leading to immediate feedback and iterative refinement. These attributes significantly enhance both the efficiency and quality of the training process.

An ABC file is composed of headers followed by the music notation. The former contain metadata regarding the tune, such as its composer and tempo, while the latter defines the melody. In ABC notation, each note is represented by a letter, and additional symbols are utilized to convey duration, rhythm, and other musical characteristics. An example is illustrated in Figure 1. “V:1” indicates the beginning of the first music track and the lines before it are headers. A tune can consist of one or more tracks, each representing a distinct musical element within the composition. The bars within each track are separated by bar line symbols like vertical lines (“|”), which refer to the standard bar line.

|[Figure 1]|
|---|

Figure 1: Example of a multitrack tune of ABC notation.

In Yuan et al. (2024), ABC files without any modification are the input of models. However, we found that the models struggle with bar alignment when dealing with multiple tracks. Since a track represents a section or division within a musical composition, such as one of the instrumental or vocal parts in a piece of music, it is crucial for models to capture the correspondence between tracks. Specifically, this correspondence exists in bars with the same indices, and thus, they should be treated as a series of groups. To this end, we reorganize the tracks as depicted in Figure 2. We concatenate music segments from bars with the same index across all tracks, including their right bar lines. These concatenated elements from different tracks are then enclosed by a pair of a newly introduced symbol “<|>”, which is not part of the original ABC system. This symbol represents the beginning or the end of a group of bars at the same stage. In cases where a tune contains only one track, each new unit will consist of a single bar. After processing all the bars, we obtain a synchronized version of the music notation, while the headers remain unchanged. The length of the tracks is not always identical due to repetition or other specific musical structures, which are difficult to handle

exhaustively. Considering these special samples typically account for just a small portion (0.01% in our dataset) of the entire dataset, we simply skip them in this scenario.

[Figure 2]

[Figure 3]

[Figure 4]

##### Align Bars

- V:1 z3 E/F/ | G A G C | ...
- V:2 z6 C2 | C2 C2 C2 CD | ...
- V:3 z6 A,2 | G,2 F,2 E,F G,A | ...

<|> z3 E/F/ | z6 C2 | z6 A,2 | <|> <|> G A G C | C2 C2 C2 CD | G,2 F,2 E,F G,A | <|> <|> ... | ... | ... | <|>

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

- Figure 2: Illustration of synchronized multiple-track ABC notation. Music segments from bars sharing the same index across all tracks, along with their right bar lines, are concatenated to guarantee alignment. The combined elements are then enclosed by a pair of a newly introduced symbol “<|>”.

- 3.3 Tokenizer

We chose YouTokenToMe (YTTM) (YouTokenToMe, 2021) framework to develop a tokenizer with a vocabulary of 50,000 tokens, leveraging the Byte-Pair Encoding (BPE) (Shibata et al., 1999) for ABC notation tokenization. This method is instrumental in segmenting the ABC text into manageable units, thereby enhancing the model’s ability to interpret and process the input effectively. We do not apply any normalization and dummy prefix to the training corpus, without changing its form or adding extra parts at the beginning. Additionally, a unique symbol “<n>“is employed to denote spaces within the ABC text, ensuring accurate space recognition by the model.

Table 1: MuPT model structure with different model size.

Parameters 190M 505M 1.07B 1.97B 4.23B Hidden size 768 1024 1280 1536 2048 Layers 12 16 20 24 32 Feedforward hidden dimensions 3072 4096 5120 6144 8192 Num heads 12 16 20 24 32 Head size 256 256 256 256 256

- 3.4 Scaling Law Table 2: Notation Definition for Scaling Law.

Notation Definition

N The number of parameters. D The training tokens. UD The number of unique tokens used. i.e., Number of tokens in each epoch. A, B, E, d, k, α, β Term parameters requiring fitting in Scaling Laws. kd, kn, ku, kin Parameters to fit the term designed for overfitting after early stop points.

The Chinchilla Law, proposed by DeepMind, is a scaling law that provides insights into the training of large language models (LLMs). Our experiments reveal that the Chinchilla Law (Hoffmann et al., 2022) provides a good fit for general cases, where moderate models were trained with a moderate amount of data. In this section, we will list two improvements to Chinchilla Law for symbolic music scaling principles on limited training data.

- 3.4.1 Optimizing Baseline Scaling Laws under Computational Constraints

A pivotal aspect of scaling laws is the optimization of loss within the bounds of computational feasibility. This is formalized as minimizing the valid loss L, subject to constraints imposed by available computational resources (C), specifically FLOPs, as denoted below:

argmin

L(N, D) s.t. FLOPs(N, D) = C (1)

N,D

This framework encapsulates the trade-offs between parameters (N) and training tokens (D), and decision-making processes inherent in scaling models under resource limitations,

illuminating pathways to efficiency and efficacy in LLMs training. Notation definition is in

- Table 2, and more details can be found in Appendix A.1.

In this paper, we will use the Chinchilla Law(Hoffmann et al., 2022) and Data-Constrained law(Muennighoff et al., 2024) as baselines. The former is a classical baseline in LLMs’ training and the latter is crucial to address the constraints faced in scenarios where the volume of available training data does not meet the ideal requisites. This phenomenon is typical in the music domain. Please refer to A.1.2 for more information.

- 3.4.2 Symbolic Music Scaling (SMS) Law

2.1B unique tokens, 190M params

4.5

Loss Value

chinchilla pred

4.0

3.5

3.0

lossValues

2.5

2.0

1.5

1.0

0.5

0 20 40 60 80 100 120

# Training Tokens (B)

2.1B unique tokens, 505M params

Loss Value

chinchilla pred

0 20 40 60 80

# Training Tokens (B)

2.1B unique tokens, 1072M params

Loss Value

chinchilla pred

10 20 30 40 50 60 70 80

# Training Tokens (B)

- Figure 3: Chinchilla Law prediction and loss survey in the setting with 2.1B unique tokens.

Figure 3 demonstrates the Chinchilla prediction in yellow lines and the observed loss in blue. We can tell that the Chinchilla law does not provide good results when the data volume D is small when the model just begins the pre-training stage, and when D is large where repeated data provides overfitting. We proposed several modifications to address these problems.

###### Continuous Adaptation of the Data-Constrained Law.

The data volume D′ for Data-Constrained Law Muennighoff et al. (2024) at n epoch is less then D = n × UD. We use its approximation of Equation 18 in Appendix A of their paper instead of the standard D′ for better prediction and simpler formulas. For more information please refer to Equation 11 in Appendix A.2. We denote the modified data volume as D′′.

###### Incorporation of a New Term.

We can observe that when that model parameter is small (e.g. N = 190M), the Chinchilla underestimates the loss value and overestimates when the model size is large (e.g. N =

1072M). This suggests that the coefficient B in the Chinchilla formula L = NAα + DBβ + E shall be relevant to N instead of a constant.

d Nα · D′′β

A Nα

B D′′β

L(N, D′′) =

+ E. (2)

+

+

To address the model’s limitations in accurately capturing performance metrics for smaller data sizes, we introduce an additional term, as delineated in Equation 2. This modification aims to refine the model’s fidelity, particularly in scenarios characterized by limited data availability. Further details on this modification can be found in the Appendix A.3.1. After that, we proposed another term to predict the early stop points and overfited loss curve.

###### Modelling Overfitting Settings.

Crucially, previous iterations of the model fall short in predicting overfitting, particularly beyond early stopping thresholds. This gap is especially pronounced in the context of Data-Constrained environments, such as music, where open-source data is limited. To this end, we introduce a new component, Loverfit, to the model, encapsulated in Equation 3, to specifically account for overfitting losses:

d Nα · D′′β

A Nα

L (N, D,UD) =

+

B D′′β

+ E + Loverfit (3)

+

where kd, kn, ku and kin are constants, and

Loverfit = GELU {kd · D + kn · log(N) − ku · log(UD) − kin} (4)

is our overfitting formulation. For comprehensive insights into the overfitting loss component, please refer to Appendix A.3.2.

###### Parameter Fitting and Model Integration.

Initial parameter fitting for {α, β, A, B, E}, and d, subsequent linear regression analysis, focusing on the residuals between Equation 2 and empirical observations, facilitates the calibration of overfitting parameters {kd, kn, ku, kin} within Equation 4. The integration of these components in Equation 3 not only predicts performance under constrained conditions but accounts for overfitting dynamics, helping to predict the true minimum of loss curve.

#### 4 Experiments

- 4.1 Experimental Setup

As outlined in section 3.1, we adopt similar model architecture from LLaMA2(Touvron et al., 2023b), including RMSNorm(Zhang & Sennrich, 2019) and SwiGLU(Shazeer, 2020). In the full-scale data setting, we trained models of various sizes (ranging from 190M to 4.23B parameters) on the ABC text corpus, which consists of 33.6 billion tokens derived from a diverse collection of monophonic and polyphonic musical compositions spanning various genres and styles. For our data repetition experiments, we utilized subsets of the corpus, specifically 6.25% and 25% random sampled data. The Adam(Kingma & Ba, 2014) optimizer and cosine learning rate schedules are applied throughout the training process. All the hyperparameters are detailed in Appendix C.

- 4.2 Scaling Law

- 4.2.1 Evaluation Metrics & Fitting Methodology

We use the R2 value and Huber loss (with the parameter δ = 1e − 3) between the authentic valid loss and predicted valid loss on small models (190M, 505M, 1.07B) to acquire the best scaling law. Then we use the best law to train two large models (with 1.97B and 4.23B). For more information about the two evaluation methods, please refer to Appendix A.4.

We optimized the SMS Law using the L-BFGS algorithm, the same with Chinchilla and Data-Constrained Laws. For more information, please refer to Appendix A.5.

- 4.2.2 SMS Law are the Best on the Training Set Table 3: Comparison of parametric fitting performance of different scaling laws.

Paramatic fit R2 Value (train) ↑ Huber Loss (train) ↓ R2 Value (test) ↑ Huber Loss (test) ↓ Chinchilla law 0.9347 0.0109 -0.0933 0.0080

Data-Constrained law 0.7179 0.0206 0.1524 0.0071 Equation 11 0.9075 0.0129 0.3114 0.0073 Equation 2 0.9759 0.0102 0.8580 0.0062 SMS Law 0.9780 0.0085 0.9612 0.0028

The integration of an additional term as delineated in Equation 2, alongside the introduction of a GELU regularization component in Equation 4, collectively underpins the superior performance of the SMS Law, as empirically evidenced by its training set outcomes. This is particularly notable in the context of our parametric fitting performance comparison (see

- Table 3), where the SMS Law outshines other scaling laws, achieving the highest R2 value (0.9780) and the lowest Huber loss (0.0085) on the training set.

Although Equation 11 does not eclipse the Chinchilla Law in performance metrics, it nonetheless presents a significant improvement over the Data-Constrained Law’s D′ by

leveraging D′′, which is indicative of a refined approach to managing the constraints posed by data repetition. This nuanced handling of data repetition, inherent to Equation 11, suggests an enhanced generalization capability in such scenarios. Therefore, we culminate it along with other modifications, manifest in the SMS Law in order to enhance model performance and generalization at the same time. In fact, it indeed provides much better results in the test set.

- 4.2.3 Scaled-up Performance Followed SMS Law

In our SMS Law experimentation under a computational budget of 1.2 × 1021 FLOPs, we initially aim to train a 2.10B (or 1.98B) parameter model across 2.82 epochs on the whole 33.6B dataset per epoch, achieving a loss of 0.5279 (or 0.5280). Engineering constraints necessitated a slight scale-down to a 1.97 billion parameter model, which, intriguingly, showed a minimal loss increase to 0.529 around 2.5 epochs. Contrary to the predictions of SMS Law, the Chinchilla Law suggests optimal performance for a 990M parameter model over 6.1 epochs. Pushing boundaries, we continuously train the 1.07B parameter model and observe overfitting returns beyond 3 epochs, validating the SMS Law’s advantages in this context. Further, we train a 4.23B parameter model that underscored the SMS Law’s predictive accuracy regarding overfitting risks, affirming its value as a strategic guide in scaling up models effectively within fixed computational constraints, beneficial for efficient model scaling decisions.

In validating the SMS Law, we analyze the performance of 1.97B and 4.23B parameter models, detailed on the right-hand side of Table 3. This comparative study highlights the SMS Law’s exceptional performance, evidenced by its unparalleled R2 values and minimal Huber Loss on testset as well.

Unlike the Chinchilla and Data-Constrained laws, the SMS Law not only showcase superior predictive accuracy but also demonstrates its efficacy in optimizing neural network scaling within computational constraints. These results affirm the SMS Law’s value in guiding scaling strategies for symbolic music, marking a significant advancement in the field.

- 4.3 Evaluation

- 4.3.1 Efficiency of our training strategy

- 0.8

- 1.0

TrainLoss

190M models

0 50 100 150 200 250

Training Tokens(Billions)

505M models

0 50 100 150 200 250

Training Tokens(Billions)

1.07B models

0 50 100 150 200 250

Training Tokens(Billions)

1.97B models

Original ABC 4K Original ABC 8K SMT-ABC 8K

Figure 4: Training Loss for different model sizes and training strategy.

To demonstrate the efficiency of our training strategies, we reference the training loss curves in Figure 4. Our comparison spans four different model sizes: 190M, 505M, 1.1B, and

- 2B. We observed that increasing the training input length from 4096 to 8192 significantly reduces the loss, especially noticeable in the convergence phase. The comparison shows that after aligning data, our training loss slightly decreases compared to the original ABC loss, demonstrating our method’s efficiency in improving training for various model sizes.

0.6

0.4

0.2

0.0

0 50 100 150 200 250

Training Tokens(Billions)

- 4.3.2 Repetition Metrics

Repetition rate Repetition is significant in evaluating how well-structured the music is. In Table 4, the piece-level average repetition rate of each system is calculated to reveal how often the repeat sign : | appears in a generated set. It appears that 44.3% of the generated samples from MuPT, which is quite close to the ground truth, and much higher than GPT-4. This suggests that MuPT is able to generate more music with repetition and structure.

- Table 4: Mean value of the intra-texture similarity and repetition rate of each system. ABC notation string generated by MuPT achieves higher intra-similarity than the ground truth as well as those generated by GPT-4.

System Texture similarity Repetition Det. Rate (%)

MuPT 0.4288 44.3 GT 0.3729 43.5 MMT 0.1767 GPT-4 0.3614 16.9

Intra Similarity In addition to the naive repetition rate, we also adopt the methods introduced in Wang et al. (2024) to calculate the intra-similarity of music in each system. Specifically, a pre-trained VAE from Yang et al. (2019) and Wang et al. (2020) is transferred to compute the texture latent for each music piece; the intra-similarity of a music piece is defined as the average value of its texture latent similarity matrix, excluding the diagonal. Since the texture encoder is pre-trained on MIDI data, we transform ABC notations into MIDI format via the toolkit called abc2midi before the latent is obtained. Table 4 shows the mean value of each system’s intra-similarity under the first-bar conditioned generation. MuPT achieves the highest score among all systems. Multitrack Music Transformers (MMT) (Dong et al., 2023), a MIDI-based music generation model, is also compared and its generated pieces have notably lower intra similarity than MuPT and GPT-4, both of which are ABCbased systems. This result corresponds with the intuition that score-level ABC notation is more capable of generating structured music than performance-level MIDI.

- 4.3.3 Subjective evaluation

Human assessment should be involved to further testify the objective repetition metrics above. Following Donahue et al. (2023) and Thickstun et al. (2023), we conduct a subjective listening study to measure the qualitative performance of MuPT against the ground truth (GT) and baselines consisting of GPT-4, MMT and random note sequences (Random). Listeners are asked to identify which of two musical excerpts from different sources is more ”musical” during the test process. They are also instructed to focus on two aspects of musicality: how consistently the music sounds throughout (e.g., in terms of its melodic contours, rhythmic patterns, and chord progression); and how likely it is that the development of the music follows a clear structure (e.g., verse-chorus division, repetitions). This process is similar with that in Yuan et al. (2024) and its details are shown in the Appendix D.

Results for all systems are shown in Table 5. Comparing our MuPT to GPT-4, listeners preferred music from our system in 79% of cases. A Wilcoxon signed-rank test of these pairwise judgments indicates that listeners preferred music from MuPT significantly more often than MMT and GPT-4 (p = 4.2249 × 10−6 and p = 6.6641 × 10−8, respectively).

#### 5 Conclusion

In this paper, we introduce the MuPT series of pre-trained models for symbolic music generation, which set the standard of training open-source symbolic music foundation models. With 190M, 505M, 1.07B, 1.97B, and 4.23B parameters, these models have been pre-trained on the largest possible amount of ABC Notation data, including 33.6 Billion high-quality diverse symbolic music tokens. Additionally, we dive deep into the scaling law exploration and propose SMS Law, a specialist in guiding future scaling of symbolic

https://github.com/xlvector/abcmidi

- Table 5: Human evaluation of paired completions of musical excerpts generated by different sources given the first bar as the condition. The left is the matrix based on the AB test. Each row indicates the % of times listeners preferred instrumentals from that system compared to those from each system individually (N = 150). Ground truth is denoted by GT. i.e.77 means that listeners preferred MuPT over GPT-4 in 77% of cases. The right is the absolute win numbers and the corresponding p-value of each pair. P-values are reported by a Wilcoxon signed rank test.

[Figure 13]

Model A Model B Wins (A/B) p-value Human works MuPT 81/69 0.4237

MMT 109/41 4.2249 × 10−6 GPT-4 119/31 6.6315 × 10−9

Random 138/12 4.4648 × 10−17 MuPT MMT 110/40 4.2249 × 10−6

GPT-4 115/35 6.6641 × 10−8

Random 131/19 1.3618 × 10−13 MMT GPT-4 95/55 0.0093

Random 103/47 0.0001

GPT-4 Random 106/44 2.6691 × 10−5

music foundation models. Our results demonstrate that the MuPT series is competitive with mediocre human composers and guarantees state-of-the-art performance on symbolic music generation. Moreover, MuPT introduces SMT-ABC, reordering the multiple-track original ABC notation format to assist pre-training of MuPT. We believe that the open access of intermediate checkpoints of MuPT, SMS Law, and MuPT series will foster collaboration and innovation within the open-source computational music community, and open the door to the next-generation symbolic music foundation models.

#### References

Alan Baade, Puyuan Peng, and David Harwath. Mae-ast: Masked autoencoding audio spectrogram transformer. Proc. Interspeech, 2022.

Alexei Baevski, Yuhao Zhou, Abdelrahman Mohamed, and Michael Auli. wav2vec 2.0: A framework for self-supervised learning of speech representations. Advances in neural information processing systems, 33:12449–12460, 2020.

Alexei Baevski, Wei-Ning Hsu, Qiantong Xu, Arun Babu, Jiatao Gu, and Michael Auli. Data2vec: A general framework for self-supervised learning in speech, vision and language. In International Conference on Machine Learning, pp. 1298–1312. PMLR, 2022.

Luca Casini, Nicolas Jonason, and Bob L. T. Sturm. Generating folk-like music in abcnotation with masked language models. In Proceedings of the International Society for Music Information Retrieval Conference 2023 Late Breaking/Demo. ISMIR, 2023.

Sanyuan Chen, Yu Wu, Chengyi Wang, Shujie Liu, Daniel Tompkins, Zhuo Chen, Wanxiang Che, Xiangzhan Yu, and Furu Wei. Beats: Audio pre-training with acoustic tokenizers. In International Conference on Machine Learning, pp. 5178–5193. PMLR, 2023.

Wenxi Chen, Yuzhe Liang, Ziyang Ma, Zhisheng Zheng, and Xie Chen. Eat: Self-supervised pre-training with efficient audio transformer. arXiv preprint arXiv:2401.03497, 2024.

Jade Copet, Felix Kreuk, Itai Gat, Tal Remez, David Kant, Gabriel Synnaeve, Yossi Adi, and Alexandre D´efossez. Simple and controllable music generation. Advances in Neural Information Processing Systems, 36, 2024.

Chris Donahue, Antoine Caillon, Adam Roberts, Ethan Manilow, Philippe Esling, Andrea Agostinelli, Mauro Verzetti, Ian Simon, Olivier Pietquin, Neil Zeghidour, et al. Singsong: Generating musical accompaniments from singing. arXiv preprint arXiv:2301.12662, 2023.

Hao-Wen Dong, Ke Chen, Shlomo Dubnov, Julian McAuley, and Taylor Berg-Kirkpatrick. Multitrack music transformer. In ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 1–5. IEEE, 2023.

Leo Gao, John Schulman, and Jacob Hilton. Scaling laws for reward model overoptimization, 2022.

Behrooz Ghorbani, Orhan Firat, Markus Freitag, Ankur Bapna, Maxim Krikun, Xavier Garcia, Ciprian Chelba, and Colin Cherry. Scaling laws for neural machine translation, 2021.

Tom Henighan, Jared Kaplan, Mor Katz, Mark Chen, Christopher Hesse, Jacob Jackson, Heewoo Jun, Tom B. Brown, Prafulla Dhariwal, Scott Gray, Chris Hallacy, Benjamin Mann, Alec Radford, Aditya Ramesh, Nick Ryder, Daniel M. Ziegler, John Schulman, Dario Amodei, and Sam McCandlish. Scaling laws for autoregressive generative modeling, 2020.

Danny Hernandez, Jared Kaplan, Tom Henighan, and Sam McCandlish. Scaling laws for transfer, 2021.

Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. Training compute-optimal large language models. arXiv preprint arXiv:2203.15556, 2022.

Wei-Ning Hsu, Benjamin Bolte, Yao-Hung Hubert Tsai, Kushal Lakhotia, Ruslan Salakhutdinov, and Abdelrahman Mohamed. Hubert: Self-supervised speech representation learning by masked prediction of hidden units. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 29:3451–3460, 2021.

Cheng-Zhi Anna Huang, Ashish Vaswani, Jakob Uszkoreit, Ian Simon, Curtis Hawthorne, Noam Shazeer, Andrew M. Dai, Matthew D. Hoffman, Monica Dinculescu, and Douglas Eck. Music transformer. In International Conference on Learning Representations, 2019. URL https://openreview.net/forum?id=rJe4ShAcF7.

Po-Yao Huang, Hu Xu, Juncheng Li, Alexei Baevski, Michael Auli, Wojciech Galuba, Florian Metze, and Christoph Feichtenhofer. Masked autoencoders that listen. Advances in Neural Information Processing Systems, 35:28708–28720, 2022.

Yu-Siang Huang and Yi-Hsuan Yang. Pop music transformer: Beat-based modeling and generation of expressive pop piano compositions. In Proceedings of the 28th ACM international conference on multimedia, pp. 1180–1188, 2020.

Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020.

Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.

Yizhi Li, Ruibin Yuan, Ge Zhang, Yinghao Ma, Xingran Chen, Hanzhi Yin, Chenghua Lin, Anton Ragni, Emmanouil Benetos, Norbert Gyenge, et al. Mert: Acoustic music understanding model with large-scale self-supervised training. arXiv preprint arXiv:2306.00107, 2023.

Tzu-Quan Lin, Hung-yi Lee, and Hao Tang. Melhubert: A simplified hubert on mel spectrograms. In 2023 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU), pp. 1–8. IEEE, 2023.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning, 2023.

Peiling Lu, Xin Xu, Chenfei Kang, Botao Yu, Chengyi Xing, Xu Tan, and Jiang Bian. Musecoco: Generating symbolic music from text. arXiv preprint arXiv:2306.00110, 2023.

Yinghao Ma, Ruibin Yuan, Yizhi Li, Ge Zhang, Xingran Chen, Hanzhi Yin, Chenghua Lin, Emmanouil Benetos, Anton Ragni, Norbert Gyenge, et al. On the effectiveness of speech self-supervised learning for music. arXiv preprint arXiv:2307.05161, 2023a.

Ziyang Ma, Zhisheng Zheng, Changli Tang, Yujin Wang, and Xie Chen. Mt4ssl: Boosting self-supervised speech representation learning by integrating multiple targets. Proc. Interspeech, 2023b.

Niklas Muennighoff, Alexander Rush, Boaz Barak, Teven Le Scao, Nouamane Tazi, Aleksandra Piktus, Sampo Pyysalo, Thomas Wolf, and Colin A Raffel. Scaling data-constrained language models. Advances in Neural Information Processing Systems, 36, 2024.

OpenAI. Musenet. https://openai.com/blog/musenet/, 2021. Accessed: 2024-01-19. Michael Schoeffler, Sarah Bartoschek, Fabian-Robert St¨oter, Marlene Roess, Susanne West-

phal, Bernd Edler, and Jurgen¨ Herre. webmushra—a comprehensive framework for web-based listening tests. Journal of Open Research Software, 6(1):8, 2018.

Noam Shazeer. Glu variants improve transformer. arXiv preprint arXiv:2002.05202, 2020. Yusuke Shibata, Takuya Kida, Shuichi Fukamachi, Masayuki Takeda, Ayumi Shinohara,

and Takeshi Shinohara. Byte pair encoding: A text compression scheme that accelerates pattern matching. 09 1999.

Bob L. Sturm, Jo˜ao Felipe Santos, Oded Ben-Tal, and Iryna Korshunova. Music transcription modelling and composition using deep learning. CoRR, abs/1604.08723, 2016. URL http://arxiv.org/abs/1604.08723.

Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding, 2023.

John Thickstun, David Hall, Chris Donahue, and Percy Liang. Anticipatory music transformer. arXiv preprint arXiv:2306.08620, 2023.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language models. ARXIV, 2023a.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv: 2307.09288, 2023b.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need, 2023.

Ziyu Wang, Dingsu Wang, Yixiao Zhang, and Gus Xia. Learning interpretable representation for controllable polyphonic music generation. arXiv preprint arXiv:2008.07122, 2020.

Ziyu Wang, Lejun Min, and Gus Xia. Whole-song hierarchical generation of symbolic music using cascaded diffusion models. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=sn7CYWyavh.

Shangda Wu and Maosong Sun. Exploring the efficacy of pre-trained checkpoints in textto-music generation task. In The AAAI-23 Workshop on Creative AI Across Modalities, 2023. URL https://openreview.net/forum?id=QmWXskBhesn.

Shangda Wu, Xiaobing Li, Feng Yu, and Maosong Sun. Tunesformer: Forming irish tunes with control codes by bar patching. In Lorenzo Porcaro, Roser Batlle-Roca, and Emilia G´omez (eds.), Proceedings of the 2nd Workshop on Human-Centric Music Information Retrieval 2023 co-located with the 24th International Society for Music Information Retrieval Conference (ISMIR 2023), Milan, Italy, November 10, 2023, volume 3528 of CEUR Workshop Proceedings. CEUR-WS.org, 2023a. URL https://ceur-ws.org/Vol-3528/paper1.pdf.

Shangda Wu, Dingyao Yu, Xu Tan, and Maosong Sun. Clamp: Contrastive language-music pre-training for cross-modal symbolic music information retrieval. In Augusto Sarti, Fabio Antonacci, Mark Sandler, Paolo Bestagini, Simon Dixon, Beici Liang, Ga¨el Richard, and Johan Pauwels (eds.), Proceedings of the 24th International Society for Music Information Retrieval Conference, ISMIR 2023, Milan, Italy, November 5-9, 2023, pp. 157–165, 2023b. doi: 10.5281/ZENODO.10265247. URL https://doi.org/10.5281/zenodo.10265247.

Guanrou Yang, Ziyang Ma, Zhisheng Zheng, Yakun Song, Zhikang Niu, and Xie Chen. Fast-hubert: an efficient training framework for self-supervised speech representation learning. In 2023 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU), pp. 1–7. IEEE, 2023.

Ruihan Yang, Dingsu Wang, Ziyu Wang, Tianyao Chen, Junyan Jiang, and Gus Xia. Deep music analogy via latent representation disentanglement. arXiv preprint arXiv:1906.03626, 2019.

YouTokenToMe. Youtokentome: Unsupervised text tokenization library, 2021. URL https://github.com/VKCOM/YouTokenToMe. Available online: https://github.com/ VKCOM/YouTokenToMe (accessed on March 25, 2024).

Ruibin Yuan, Hanfeng Lin, Yi Wang, Zeyue Tian, Shangda Wu, Tianhao Shen, Ge Zhang, Yuhang Wu, Cong Liu, Ziya Zhou, et al. Chatmusician: Understanding and generating music intrinsically with llm. arXiv preprint arXiv:2402.16153, 2024.

Biao Zhang and Rico Sennrich. Root mean square layer normalization, 2019. Hongyuan Zhu, Ye Niu, Di Fu, and Hao Wang. Musicbert: A self-supervised learning of

music representation. In Proceedings of the 29th ACM International Conference on Multimedia, pp. 3955–3963, 2021.

#### A Scaling Law

- A.1 Scaling Law Baseline

- A.1.1 Abstracting Loss Metrics through the Chinchilla Law

In this part, we focus on the relationship of loss metrics to various resource budgets in deep learning. It is first put forward by the Chinchilla Law as illustrated in Equation 5. This law posits that both training and evaluation losses can be abstracted as a function of model capacity N and training data size D, thus offering an insight to estimate the best combination of resources to be assigned to training.

L(N, D) =

A Nα

+

B Dβ

+ E (5)

Here, L(N, D) denotes the loss metric during training or evaluation, which is assumed to exhibit a power-law dependency on N and D. The parameters A, B, E, α, and β are determined by empirical fitting.

- A.1.2 Data-Constrained Law

Data-Constrained Law: Scaling under Data Limitations. Complementing the Chinchilla Law, the Data-Constrained Law shows the scaling dynamics of LLMs when facing the data scarcity problem. Here, we strictly refer to the derivation method of Muennighoff et al. (2024). The goal of discovering Data-Constrained Scaling Law is to generalize the expression to multiple epochs where tokens are repeated.

Data-constrained law is defined as:

A N′α

B D′β

+ E (6) where

L (N, D,UD) =

+

N′ = UN + UNR⋆N 1 − exp −RN

R⋆N

(7)

D′ = UD + UDR⋆D 1 − exp −RD

R⋆D

To get a better understanding of the equation, the definitions of each of the above parameters are as follows: Like Chinchilla Law, N is defined as the number of model parameters, and D is defined as the training tokens.

UD is defined as the number of unique tokens used. For data-constrained law, UD is computed as min{D,DC} given a budget of unique data Dc.

UN is defined as the number of “unique” parameters that provide an optimal fit for UD. According to the method mentioned in Muennighoff et al. (2024), given the following learned variables, {A, α, B, β E}, the optimal allocation of compute(C) to N and D as follows:

a

C 6

Nopt(C) = G

b

Dopt(C) = G−1 C 6

1 α+β

αA βB

(8)

G =

β α + β

###### a =

α α + β

###### b =

Thus, UN is equal to min{Nopt, N}.

RD is defined as the number of times the data is repeated. When training for a single epoch, RD = 0.

RN is the number that the ‘unique’ parameters are repeated where RN = max{ U NN − 1, 0}. D′ is defined as the ”effective data size”: the number of unique data needed to get the same value as repeating U unique tokens for RD repeats.The derivation process is as followed:

From a conceptual standpoint, the redundancy of data samples diminishes their incremental value in enhancing the model’s knowledge base, given the model’s prior exposure to said information. This principle underlies the hypothesis that each successive repetition of a sample contributes marginally less to the learning process, as the model has partially assimilated the information contained within the sample through prior iterations. To describe the process of training information loss, we have

RD

(1 − (1 − δ))RD δ

(1 − δ)k = U + (1 − δ)U

### ∑

D′ = U + U

(9)

k=1

where δ is defined as the ‘forgetting rate’. Each time a series of tokens is trained on a model, the model learns a 1 − δ fraction information from the optimization process. Assuming that the number of epochs beyond which repeating does not help, the right-hand side goes to

to (1−δδ)U, since limRD→∞(1 − (1 − δ)RD) = 1. We define R⋆D is defined as1−δ

δ , which is a learned constant. According to Taylor expansion, if δ is small, we have:

###### −1 R⋆

e

D ≈ (1 − δ) (10)

( R−⋆1 D

)RD

Now inserting(1−δδ) = R⋆D and (1 − δ)RD = e

into Equation9, we get our final equation representing the effective data.

As the frequency of encountering repeated tokens diminishes, the benefit gained from processing them also decreases. Hence, the derivation of the N′ is similar to D′. In this context, there’s no need to elaborate further. It should be pointed out that R⋆N is a learned parameter.

- A.2 Continuous Adaptation of the Data-Constrained Law.

To enhance the predictive accuracy of the Data-Constrained law (Muennighoff et al., 2024) for continuous domains, we extend the original discrete formulation 11 to accommodate continuous variables, allowing for a more nuanced understanding of data constraints in varied contexts. For an in-depth discussion on the derivation and implications of this continuous formulation, please refer to Appendix A.2.

A Nα

B D′′β

L(N, D,UD) =

+ E (11)

+

where k is a new parameter to be fit, and D′′, the adjusted data size, is given by:

1 − kD/UD 1 − k

D′′ =

UD. (12)

The definition of D′ in Equation 9 is defined from a discrete version and can not be extended to the case when D is less than UD. So we reform the Equation 9 to

D UD

1 − (1 − δ)

D′ =

δ · UD

(13)

D UD

1 − k

d

1 − kd · UD

=

where kd := 1 − δ. This equation is equivalent to equation 10 when D is a positive integer times UD.

We implemented a formula symmetric to N′ with UN and kN. But the calculation results of kN ≈ 0.999. To make the formula simple, we use the original N instead of N′ in the following formula.

- A.3 Motivation of SMS Law

- A.3.1 Motivation of Adding Power of “ND” Term

[Figure 14]

- Figure 5: The loss curve, Chinchilla prediction, and Equation11 on 2.1B, 8.4B and 33.6B training data.

In our submission, we present an in-depth analysis of the model’s loss dynamics as illustrated in Figure 5, which juxtaposes the empirical loss trajectory (depicted through a blue line) against the theoretical predictions derived from the Chinchilla Law (illustrated by a yellow line) and further contextualized by Equation 11. This comparative study spans three distinct datasets—2.1B, 8.4B, and 33.6B data points—across models of varying capacities: 190M, 505M, and 1.07B parameters, respectively, arranged in a matrix of subfigures with datasets delineated by rows and model capacities by columns.

Observations across all data volumes reveal a nuanced interaction between model and data sizes. Specifically, for smaller datasets and model sizes (190M parameters), predictions consistently underestimate actual loss values, whereas for smaller datasets paired with larger models (1.07B parameters), predictions tend to overestimate. This discrepancy underscores a critical insight: loss reduction accelerates with increasing model size, suggesting a modified

loss function, AN+αϵ over the simpler NAα Crucially, the term ϵ emerges as a function of a single variable N, ensuring variability in Nϵα across each unique model configuration shifting upwards or downwards without changing the shape. The ideal adjustment implies that ϵ approaches zero for large datasets, yet remains significant for smaller ones, highlighting its dependency on data volume D.

In addressing potential overfitting, our strategy focuses on minimizing parameter growth in line with Equation 11. A straightforward approach involves augmenting the loss L

into a polynomial encompassing NAα and DBβ , with Equation 2 introducing an additional term, Nαd·Dβ , to the existing framework. This refinement, while ostensibly simple, has been shown to yield robust and promising outcomes, exemplifying the efficacy of our proposed modifications in enhancing model performance within the context of scaling laws.

- A.3.2 Motivation of Linear Regression Term for Overfitted Residual

[Figure 15]

- Figure 6: The loss curve, Chinchilla prediction, and Equation 2 (green lines) on 2.1B training data.

- Figure 6 offers a detailed exposition on the fidelity of Equation 2 in capturing the loss trajectory across training sets of varied model capacities (190M, 505M, and 1.07B parameters). It is evident from the analysis that the equation adeptly mirrors the empirical loss curve across a broad spectrum of configurations, with the exception of scenarios characterized by concurrently large model sizes and token counts. A notable oversight in the literature is the scant consideration of loss dynamics beyond early stopping points, a consideration of paramount importance in music domain due to the inherent constraints on training data.

In addressing the challenges posed by modelling loss post-early stopping, our investigation delineates two distinct methodologies. The first approach involves the integration of a regularization term within D′′, aimed at reducing its magnitude beyond the early stopping threshold. Despite its conceptual appeal, this method falls short of providing an adequate fit to the observed data. Alternatively, we explore the augmentation of the loss function L with an additional term, engineered to be negligible when both D and N are minimal, yet incrementally assertive in influencing the loss trajectory after early stopping points. This latter strategy not only aligns more closely with empirical observations but also introduces a nuanced mechanism to accommodate the unique requirements of training in the music domain, thereby extending the utility and applicability of scaling laws within this context.

As delineated in Figure 7, the analysis of residuals post the 40 billion token threshold unveils a discernible onset of overfitting, which intriguingly appears to correlate with the model size, data capacity, and the count of unique tokens processed within a single epoch. This overfitting is further characterized by a linear dependency of loss on the total number of processed tokens, coupled with a quasi-linear transition of early stopping points observed across different model capacities (as organized in rows) and magnified across columns.

The progression of model capacities—doubling across rows and quadrupling across columns—illuminates a systematic pattern, suggesting that the early stopping points and consequently, the predicted loss, might be effectively modeled through a linear regression involving dataset size D, the logarithm of model capacity log(N), and and the logarithm of unique tokens per epoch log(UD). This observation culminates in the proposition of a regularization term formulated as kd · D + kn · log(N) − ku · log(UD) − kin, aimed at encapsulating and mitigating the observed overfitting dynamics.

[Figure 16]

- Figure 7: Residule between authentical valid loss and Equation 2 prediction (blue lines), and the linear regression results (yellow lines).

Table 6: Ablition study on the activation function.

Activation Function R2 (test)↑ Huber Loss (test)↓ ReLU 0.9786 0.0095 LeakyReLU 0.9786 0.0095 GELU 0.9780 0.0085 Tanh 0.9786 0.0094 SELU 0.9779 0.010 Sigmoid 0.6030 0.0700

In addressing the intricacies of regularization within the context of early model training, especially when considering models of smaller scale (where UD and D are minimal while N is comparatively large), it becomes imperative to ensure that the regularization term does not adopt a substantially negative value. This stipulation aims to prevent undue penalization at the onset of training, thereby necessitating the incorporation of an activation function that tempers the regularization term’s behavior. The Gaussian Error Linear Unit (GELU) function emerges as an apt choice in this scenario. GELU approximates the Rectified Linear Unit (ReLU) function for positive inputs, while also permitting slight negative values with minimal absolute magnitude, thus offering a balanced solution.

Empirical evidence, as detailed in our analysis, underscores the efficacy of applying the GELU function to the regularization term, notably achieving the lowest training loss alongside the second-highest R2 value among the tested models. This finding is particularly salient given the broader magnitude of loss variations relative to R2 values, thereby accentuating the GELU function’s suitability for our regularization term. Consequently, the finalized model, incorporating the GELU-modulated regularization term, is depicted through a yellow line in Figure 7. This strategic application of the GELU function not only mitigates the potential for excessive early training penalization but also optimizes the regularization term to enhance model performance effectively.

This approach not only elucidates the linear interdependencies among critical factors influencing model performance but also presents a nuanced regularization strategy designed to enhance model generalizability. Through the integration of this regularization term, we aim to establish a more robust and theoretically informed framework for predicting and managing loss trajectories in large-scale training regimes.

- A.4 Evaluation Metrics

The R-squared value, also known as the ”Coefficient of Determination,” is a statistical measure used to evaluate the goodness-of-fit of a regression model. It is defined as:

R2 = 1 −

SSres SStot

(14)

Where SSres represents the Sum of Squares of Residuals, indicating the total sum of squared differences between the predicted values of the model and the actual observed values, SStot represents the Total Sum of Squares, indicating the total sum of squared differences between the observed values of the dependent variable and their mean value.

The Huber loss is a type of loss function commonly employed in robust regression models. Unlike the squared error loss, which is sensitive to outliers in the data, the Huber loss is designed to be less affected by outliers. It achieves this by combining the characteristics of both the squared error loss and the absolute error loss. It is defined piecewise by:

Huberδ(y, f(x)) =

- 1

- 2(y − f(x))2, if |y − f(x)| ≤ δ

δ(|y − f(x)| − 12δ), otherwise

(15)

For small residuals, it behaves like the squared error loss, whereas for large residuals, it behaves like the absolute error loss. This allows the Huber loss to provide a balance between the two, resulting in a more robust estimation procedure.

- A.5 Parameters Fitting Approach

In our study, we adopt a methodology analogous to the Chinchilla Law and the DataConstrained Law, employing the L-BFGS algorithm—a limited-memory quasi-Newton method—for the optimization of the Huber Loss. This loss function is applied between the logarithm of the predicted loss and the logarithm of the observed (authentic) loss across multiple runs. The objective is to identify the optimal parameters (best para) that minimize this Huber Loss, formalized as follows:

best para = min ∑

runi

Huberδ log

d Nα · D′′β

+

A Nα

+

B D′′β

+ E

i

,log(Li)

= min ∑

runi

Huberδ LSE log

d Nα · D′′β

,log

A Nα

,log

B D′′β

,log(E)

i

,log(Li)

= min ∑

runi

Huberδ

 



LSE



 

log(d) − α log(N) − β log(D′′)

- log(A) − α log(N)
- log(B) − β log(D′′) log(E)



 

,log(Li)

 



(16)

where LSE refers to the log-sum-exp a numerically stable method to compute the logarithm of a sum of exponentials of inputs. The Huber Loss parameter, δ is set to 1e − 3, reflecting a stringent criterion for switching between squared loss and absolute loss to ensure robustness in optimization. Additionally, the L-BFGS algorithm’s learning rate is configured at 1e − 1, with an update history size of 10 to balance between computational efficiency and the capacity to capture relevant optimization trends.

- A.6 Results of Proposed Methods with Early Stops

From the table, we can see that most of the experimental results increase after we delete the curve after the early stop points. Adding the linear regression still contributes to the performance increase on the training set but provides worse results on test set compared to Equation 2.

Table 7: Comparison parametric fitting performance of different Scaling Laws on the curve before early stop points.

Paramatic fit R2 Value (train) ↑ Huber Loss (train) ↓ R2 Value (test) ↑ Huber Loss (test) ↓ Chinchilla law 0.9443 0.0073 -0.0004 0.0029

Data-Constrained law 0.7216 0.0189 0.1005 0.0050 Equation 11 0.8356 0.0151 0.5829 0.0045

Equation 2 0.9843 0.0072 0.9866 0.00088 SMS Law 0.9851 0.0055 0.9864 0.00091

#### B A Short Lecture Note of L-BFGS Algorithm

BFGS (Limited-memory Broyden–Fletcher–Goldfarb–Shanno) is a variant of the BFGS method, a quasi-Newton optimization algorithm used to solve unconstrained nonlinear optimization problems. It is particularly suitable for handling large-scale optimization problems by limiting the size of the stored matrices, thus reducing storage and computational costs.

The core idea of the L-BFGS algorithm is to approximate the inverse of the Hessian matrix of the objective function using historical records of function values and gradients. In contrast to traditional Newton’s method that requires storing and updating the complete Hessian matrix, L-BFGS method only needs to store and update some historical information, making it more efficient in terms of storage and computation. It iteratively constructs an approximate inverse Hessian matrix to update parameters and continuously optimize the objective function until reaching a local optimum or satisfying convergence criteria.

According to Newton-Raphson method: f : Rn → R f(xt + d) = f(xt) + ∇f(xt)Td +

(17)

- 1

- 2

dT∇2 f(xt)d + o(∥d∥2)

h(d) := f(xt + d) = f(xt) + ∇f(xt)Td +

1 2

dT∇2 f(xt)d (18)

dˆ := argmin

h(d)

d

∇h(dˆ) = ∇f(xt) + ∇f2(xt)dˆ = 0

(19)

xt+1 = xt + dˆ = xt − ∇2 f(xt)−1∇f(xt) (20) According to BFGS:

BkskskTBk skTBksk

ykykT ykTsk

(21)

Bk+1 = Bk −

+

In the BFGS algorithm, storing the approximate Hessian matrix at each iteration can be costly in terms of memory, especially in high-dimensional data scenarios. However, in practical computation, what we primarily need is the search direction. To address this issue, the L-BFGS algorithm was introduced as an improvement over the BFGS algorithm.

In L-BFGS, instead of storing the full Hessian matrix, only the most recent iterations’ information is retained, significantly reducing the memory footprint.

T k

k sk , Vk = I − yks

let ρk = yT1

ykTsk , then Hk+1 can be represented as:

Hk+1 = VkTHkVk + ρkskskT (22)

Note that H0 = I.

- H1 = V0TH0V0 + ρ0s0s0T
- H2 = V1TH1V1 + ρ1s1s1T

= V1T(V0TH0V0 + ρ0s0s0T)V1 + ρ1s1s1T

= V1V0TH0V0V1 + V1Tρ0s0s0TV1 + ρ1s1s1T

. . . Hk+1 = (VkTVkT−1 · · · V1TV0T)H0(V0V1 · · · Vk−1Vk)

+ (VkTVkT−1 · · · V1T)ρ1s1s1T(V1 · · · Vk−1Vk)

###### + · · ·

+ VkTρk−1sk−1skT−1Vk

+ ρkskskT

If only the first m steps are retained:

Hk+1 = (VkTVkT−1 . . . VkT−m)H0(Vk−m . . . Vk−1Vk)

+ (VkTVkT−1 . . . VkT−m)ρ1s1s1T(Vk−m . . . Vk−1Vk)

+ . . .

+ VkTρk−1sk−1skT−1Vk

+ ρkskskT

Then only sk and yk is necessary to be remained.

(23)

(24)

#### C Training Details

All the models are trained using AdamKingma & Ba (2014), with β1 = 0.9, β2 = 0.95, eps = 10−8. We use a cosine learning rate schedule, decay the final learning rate from 3−5 to 3−6, with warmup ratio of 0.1. We apply a weight decay of 0.1 and gradient clipping of 1.0. Table

- 8 shows other training details of each model. Table 8: Training Details for different ABC format and model settings.

Parameters Context Length Trained Tokens Training Days Num of GPUs

190M 4096 119B 8.4 2 505M 4096 97B 8.4 4 1.07B 4096 49B 8.3 4 1.97B 4096 56B 8.4 8

Original ABC

190M 8192 346B 6.9 8 505M 8192 322B 4.1 32 1.07B 8192 223B 5.4 32 1.97B 8192 196B 8.1 32

190M 8192 276B 5.5 8 505M 8192 212B 2.7 32 1.07B 8192 181B 4.4 32 1.97B 8192 272B 11.3 32 4.23B 8192 262B 10.7 64

SMT-ABC

#### D Human Assessment

We use webMUSHRA toolkit (Schoeffler et al., 2018) to conduct a web-based subjective listening AB-test. During the listening test, we ask the participants to choose the better one

between a pair of music excerpts generated from two randomly selected different systems from GT, MuPT, GPT-4, MMT and Random by considering the ”Musicality” which indicates the overall perceptive quality of the music. Participants are encouraged to make a choice by refering to the guidelines below:

- • How consistent the music sounds as a whole (e.g., in terms of its melodic contours, rhythmic patterns, and chord progression).
- • How likely the development of the music follows a clear structure (e.g. verse-chorus division, repetitions).
- • If you cannot understand the two guidelines above, just choose the one from A and B you prefer.

