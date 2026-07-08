# arXiv:2402.01831v3[cs.SD]28May2024

## Audio Flamingo: A Novel Audio Language Model with Few-Shot Learning and Dialogue Abilities

Zhifeng Kong1 Arushi Goel1 Rohan Badlani1 Wei Ping1 Rafael Valle1 Bryan Catanzaro1

### Abstract

Augmenting large language models (LLMs) to understand audio – including non-speech sounds and non-verbal speech – is critically important for diverse real-world applications of LLMs. In this paper, we propose Audio Flamingo, a novel audio language model with 1) strong audio understanding abilities, 2) the ability to quickly adapt to unseen tasks via incontext learning and retrieval, and 3) strong multi-turn dialogue abilities. We introduce a series of training techniques, architecture design, and data strategies to enhance our model with these abilities. Extensive evaluations across various audio understanding tasks confirm the efficacy of our method, setting new state-of-the-art benchmarks. Our demo website is https://audioflamingo.github.io/ and the code is open-sourced at https:// github.com/NVIDIA/audio-flamingo.

### 1. Introduction

The ability to understand sound is unarguably important and necessary for an agent to interact with the world. While large language models (LLMs) have shown an impressive ability to understand and reason about the world through text, their understanding of sound remains limited to transcriptions of speech (Lyu et al., 2023), thus making LLMs agnostic to important information in non-speech sounds and non-verbal speech. Even though recent contributions have improved their ability to understand sound (Gong et al., 2023c; Lyu et al., 2023; Huang et al., 2023; Deshmukh et al., 2023; Chu et al., 2023; Tang et al., 2023a), there exists no model that combines: i) strong au-

1NVIDIA, CA, USA. Correspondence to: Zhifeng Kong <zkong@nvidia.com>, Wei Ping <wping@nvidia.com>, Rafael Valle <rafaelvalle@nvidia.com>.

Proceedings of the 41st International Conference on Machine Learning, Vienna, Austria. PMLR 235, 2024. Copyright 2024 by the author(s).

dio understanding ability on various tasks (Deshmukh et al., 2023), ii) the ability to execute multi-turn dialogues (Duan et al., 2023), and iii) the ability to quickly adapt to unseen tasks without fine-tuning, for example, through in-context learning (Alayrac et al., 2022) and retrieval augmented generation (Lewis et al., 2020).

Our contribution to expand LLM’s ability to understand sound is called Audio Flamingo: a novel audio language model that supports in-context learning (ICL), retrieval augmented generation (RAG), and multi-turn dialogues. It achieves state-of-the-art results on multiple audio understanding tasks.

Expanding LLM’s ability to understand sound is a challenging task. The first challenge is extracting features from variable-length audio, and conditioning the LM on the audio features. While prior works have designed representations for audio of any length (Wu et al., 2023), they can lose temporal information. In this work, we introduce an audio feature extractor with sliding window based on Elizalde et al. (2023b), which we believe to capture temporal information better. To condition the LM on audio inputs, previous models prepended language tokens with audio tokens (Deshmukh et al., 2023; Chu et al., 2023; Tang et al., 2023a). This approach may have excessive overhead especially for long audio, as the complexity is quadratic with respect to the number of audio tokens. In contrast, we use cross attentions to fuse audio inputs into the LM in a similar way as Flamingo (Alayrac et al., 2022). In our approach the complexity is linear in the number of audio tokens, thus making Audio Flamingo able to generalize to many audio inputs efficiently.

The second challenge is collecting and training on highly heterogeneous data. Prior works have collected and combined different datasets for training (Doh et al., 2023; Deshmukh et al., 2023; Chu et al., 2023; Gong et al., 2023c). We follow their approach and curate a heterogeneous dataset with approximately 5.9 million audio-text pairs. Prior works have also designed different training curriculum, such as training on close-ended tasks first followed by open-ended tasks (Gong et al., 2023c; Tang et al., 2023a). However, these result in a

[Figure 1]

- Figure 1. Audio Flamingo versus previous state-of-the-art (Deshmukh et al., 2023; Chu et al., 2023; Gong et al., 2023b;c; Tang et al., 2023a; Ghosh et al., 2023) on a number of audio understanding benchmarks. The numbers are normalized such that the maximum of all models is 100% on each task. Audio Flamingo sets the new state-of-the-art results on most of these tasks.

trade-off between close-ended and open-ended tasks, limiting the overall performance (Deshmukh et al., 2023; Tang et al., 2023a). We use a different approach based on a widely adopted and stable method to train LLMs (Ouyang et al., 2022). Specifically, we use two training stages: pre-training and supervised fine-tuning (SFT), each with different subsets and training techniques. These innovations make Audio Flamingo achieve the state-of-the-art results on several audio understanding benchmarks with < 13 number of parameters as Chu et al. (2023) and Gong et al. (2023c).

The third challenge is to give the audio language model the ability to quickly adapt to new tasks without fine-tuning, for instance, via in-context learning (ICL) (Brown et al., 2020) and retrieval. While recent audio language models have shown zero-shot abilities (Deshmukh et al., 2023; Gong et al., 2023c), they lack the ability to perform in-context few-shot learning to new tasks. In this paper, we introduce a series of techniques to realize this ability. We implement an efficient retrieval method, introduce an ICL template, and use retrieved samples to create interleaved ICL datasets for training. We also introduce a novel cross attention mask for interleaved samples. As a result, Audio Flamingo can be quickly adapted to new tasks via ICL and retrieval without task-specific fine-tuning. Our results confirm the efficacy of our approach and set new state-of-the-art few-shot benchmarks.

The last challenge is to give the audio language model the ability to chat with a user for many rounds. While prior methods have shown demos of dialogues (Gong et al., 2023c; Chu et al., 2023), they lack systematic

and quantitative evidence. To address this challenge, we create two multi-turn dialogue datasets with GPT-4 (Achiam et al., 2023) based on detailed annotations of two datasets, with an emphasis on correlated context. We obtain a chat model by fine-tuning Audio Flamingo on these datasets. Our results show that our chat model has strong multi-turn dialogue ability and significantly outperforms previous methods.

We evaluate Audio Flamingo on a large and diverse set of close and open-ended benchmarks. A single Audio Flamingo model surpasses the previous state-ofthe-art on most benchmarks, and the chat version of Audio Flamingo significantly outperforms baselines on dialogue benchmarks. Figure 1 summarizes the benchmark results of Audio Flamingo. We also briefly discuss about the neural architecture and hyper parameters in the experiments. Our key contributions include:

- 1. We propose Audio Flamingo: a Flamingo-based audio language model for audio understanding with a series of innovations. Audio Flamingo achieves state-of-the-art results on several close-ended and open-ended audio understanding tasks.
- 2. We design a series of methodologies for efficient use of ICL and retrieval, which lead to the stateof-the-art few-shot learning results.
- 3. We enable Audio Flamingo to have strong multiturn dialogue ability, and show significantly better results compared to baseline methods.

### 2. Related work

Multimodal LLMs. There has been tremendous progress in the area of multimodal LLMs. In addition to text, these models take inputs from various modalities such as vision (Tsimpoukelli et al., 2021; Alayrac

- et al., 2022; Yang et al., 2023; Driess et al., 2023; Liu
- et al., 2023a; Li et al., 2023a), audio (Deshmukh et al., 2023; Gong et al., 2023b; Rubenstein et al., 2023), or multiple of them (Han et al., 2023; Tang et al., 2023b; Moon et al., 2023; Zhao et al., 2023), and each has a different integration method. In the audio modality, prior works have looked at speech tasks (Chen et al., 2023; Rubenstein et al., 2023), general audio understanding (Deshmukh et al., 2023; Gong et al., 2023c), music understanding (Gardner et al., 2023; Won et al., 2023; Li et al., 2023b; Liu et al., 2023b; Doh et al., 2023), or a combination of these (Gong et al., 2023b; Tang et al., 2023a; Chu et al., 2023). The focus of our paper is audio understanding, which includes non-speech sound and music, and non-verbal speech. Different from prior works, our model has stronger audio understanding ability, and is the first audio understanding model with

- i) in-context few-shot learning ability, ii) retrieval augmented generation ability, and iii) strong multi-turn dialogue ability.

Audio encoders and representation. Many audio encoders extract audio features from the spectrogram, including CNN-based method (Kong et al., 2020) and Transformer-based methods (Gong et al., 2021; Chen et al., 2022; De´fossez et al., 2022; Radford et al., 2023; Gong et al., 2023a). These methods are primarily targeted at solving a particular problem such as speech recognition or event detection. Based on these encoders, many joint audio-language embeddings have been proposed (Elizalde et al., 2023a;b; Wu et al., 2023; Huang et al., 2022; Li et al., 2023b). These methods use contrastive learning to map audio and language embeddings into the same space, and are often trained on a large variety of audio and language. However, many of these methods compute a single embedding for an audio and therefore may lose temporal information. In this paper, we build an audio encoder with sliding windows based on ClapCap (Elizalde et al., 2023b) to better capture long-range and temporal information.

Data augmentation. Due to limited amount of highquality human annotated sounds besides speech transcriptions, many works have proposed to augment textural description with existing LLMs such as GPT-4 (Achiam et al., 2023). A common strategy is to provide an LLM with annotated tags, timestamps, and other miscellaneous information, and then ask it to generate captions (Wu et al., 2023; Doh et al., 2023; Mei et al., 2023; Gardner et al., 2023) or question-answering data pairs (Gong et al., 2023c;b; Liu et al., 2023b). In this paper, we leverage existing LLMs to generate two multiturn dialogue datasets based on detailed annotations, which enable our model strong dialogue abilities.

In-context learning (ICL). In-context learning is a kind of few-shot learning ability, where an LLM rapidly adapts to a desired task at inference time only after looking at a few examples in the prompt (Brown et al., 2020). It has widely shown success in natural language tasks (Wei et al., 2021) and visual-language tasks (Alayrac et al., 2022; Yang et al., 2023). In the speech domain, ICL has been shown to help speechrelated tasks such as speech recognition, translation, and processing (Gao et al., 2022; Wang et al., 2023; Hsu et al., 2023; Chen et al., 2023). However, ICL for general audio understanding is much less explored. In this paper, we propose the first audio understanding model with ICL ability.

Retrieval-augmented generation (RAG). Retrieval-augmented generation for LLMs is to improve generation quality by using external knowledge,

for example from an external database, which contains useful and related knowledge. It has been widely applied in natural language tasks (Guu et al., 2020; Karpukhin et al., 2020; Lewis et al., 2020; Borgeaud

- et al., 2022) and visual-language models (Yang
- et al., 2023). In the audio-language domain, Ghosh et al. (2023) proposed a retrieval method for audio captioning by prepending captions from similar audios to the prompt. However, it does not provide the retrieved audio to the model. Consequently, the model loses information on how similar the retrieved audio is to the test audio. In contrast, we provide both the retrieved audio and text to our model. The benefit of this approach is that our model could determine when and how to use the retrieval based on the similarity between test and retrieved audio. We provide comparisons in our few-shot experiments.

### 3. Method

In this section, we introduce Audio Flamingo, an audiounderstanding language model with few-shot learning via ICL and RAG. In Section 3.1, we introduce the architecture used in Audio Flamingo, including the audio feature extractor, audio representation transformation layers, language model, and the conditioning method. In Section 3.2, we introduce the training method of Audio Flamingo, including the training objective, design of masks, and training stages.

#### 3.1. Architecture

Our neural architecture is composed of four components: i) an audio feature extractor with sliding window, ii) audio representation transformation layers, iii) a decoder-only language model, and iv) gated xattndense layers. Figure 2 summarizes the architecture.

i) Audio feature extractor with sliding window. We use ClapCap (Elizalde et al., 2023b) as the audio feature extractor backbone, which we denote as E. ClapCap is hard-coded to take 7-second of 44.1kHz raw audio as input, then transforms the audio into Melspectrogram of hop length 320, window length 1024, 64 Mel bins, and finally outputs a 1024-dimensional vector representation.

We consider each 7-second segment as a window and use sliding windows to extract features for longer audio. The overlap between consecutive windows is 7×0.75 = 5.25 seconds. Formally, let x(s:t) be the segment of s to t seconds in audio x. Then, the extracted feature

4 :7(m4+3)) . The intuition of this design is to capture long-range and temporal information that might be ignored in a

4 )),··· ,E(x(7(m−1)

is E(x(0:7)),E(x(47:7×5

[Figure 2]

- Figure 2. Neural architecture of Audio Flamingo. It takes interleaved audio and text as input and outputs free-form text.

single fused representation vector (Wu et al., 2023). We use a maximum of m = 16 sliding windows, which supports a maximum of 33.25 second audio length. 1 Long audio will be cropped and short audio will be zero-padded. If an entire segment is zero-padded then we will mask the corresponding embedding at cross attention. If the input is interleaved data with > 1 audio, we concatenate their sliding window representations.

- ii) Audio representation transformation layers. We increase model capacity by further applying a few audio representation transformation layers to the concatenated audio feature representations described earlier. It is comprised of 3 self-attention layers (Vaswani et al., 2017), with 8 heads and inner dimension 2048 each. This module is fully trainable.
- iii) Language model. We use a decoder-only causal LM in our architecture. In this paper, we use OPT-IML-MAX-1.3B (Iyer et al., 2022), a 1.3B parameter model with 24 LM blocks. It has been instructiontuned on many natural language tasks.
- iv) Conditioning LM on audio representations. We use the gated xattn-dense layers from Flamingo (Alayrac et al., 2022) to achieve conditioning on audio inputs. Each layer has two blocks: 1) a residual block with cross attention and tanh gating, followed by 2) a residual block with dense layer and tanh gating. These layers are prepended to each LM block.

- 3.2. Training Method

our model. Let (yout)t be the t-th token and (yout)<t the first t−1 tokens of the output. For a non-interleaved sample z = (x,yins,yout), the log-likelihood is

|yout|

L(z) =

log pθ ((yout)t|x,yins,(yout)<t). (1)

t=1

For an interleaved training sample composed of J samples zint = {z1,··· ,zJ}, where zj = (xj,yinsj ,youtj ), the log-likelihood is computed over all outputs:

Lint(zint = {z1,··· ,zJ}) =

|youtj |

(2)

J

log pθ (youtj )t|z<j,xj,yinsj ,(youtj )<t .

t=1

j=1

Note this interleaved data loss is different from prior models, which compute losses only on the last output youtJ (Yang et al., 2023), or have either none or indirect conditioning on prior multimodal inputs x<j (Alayrac et al., 2022; Ghosh et al., 2023). We expect (2) can help the model look at a various number (including zero when j = 1) of ICL samples as well as the associated audio, thus improving robustness and training efficiency in a similar way as bucketing (Khomenko et al., 2016), especially when the ICL samples are retrieved similar samples. The corresponding loss mask is shown on the right-hand-side of Figure 3.

Let {Di,i ∈ I} be all non-interleaved training datasets, and {Di

int,i′ ∈ Iint} be all interleaved training datasets. The overall training objective is a weighted mixture of losses on each dataset:

′

- i) Training objective. Let x be the mono-channel audio input, yins be the instruction text (e.g. question), and yout be the output text. For conciseness we use z = (x,yins,yout) to represent each training sample. We use maximum likelihood estimation (MLE) to train

L = −

λiEz∼DiL(z) −

Lint(zint),

λi′Ez

int∼Dinti′

i′∈Iint

i∈I

(3) where λi’s are the weights for each dataset. The weights are constant hyper-parameters and have a huge impact on the final model. They are computed from the predefined number of epochs for each dataset (see Section 4.1 for the intuition, and Appendix A for details).

1We use m = 16 because most training samples are < 30s. Note that our cross attention complexity is linear in m and therefore the audio length. In prior self-attention methods the attention complexity is quadratic in the audio length.

[Figure 3]

Figure 3. Left: the block upper-triangular cross attention masks between text tokens and audio embeddings. Right: the loss mask of an interleaved training sample.

- ii) Cross attention masks. We use block uppertriangular cross attention masks for interleaved sam-

ples so that the likelihood of j-th output pθ(youtj ) is conditioned only on the first j audio inputs x≤j. We expect this helps making the model to look at previous audio. Figure 3 demonstrates the mask.

- iii) Two training stages. We divide training into pre-training and supervised fine-tuning (SFT), a widely adopted and stable method in training LMs (Ouyang et al., 2022). During pre-training we only train the audio representation transformation layers and the gated xattn-dense layers. The purpose is to obtain a good set of initialization weights for these layers. During SFT, we unfreeze the entire LM, and train all modules but the audio encoder. 2

### 4. Data

#### 4.1. Datasets

In this section, we introduce our data strategies, including dataset collection, generation, and blending. We also introduce templates for each type of dataset.

Dataset sources. We train our model on a variety of audio datasets that can be roughly classified into three types: music, non-speech general sound, and non-verbal speech. In this paper, we focus on these data given the immediate availability of state-of-the-art speech transcription models. We look at three types of tasks: (1) audio captioning (CAP), where we would like the model to describe the audio in a sentence; (2) audio question-answering (AQA), where we would like the model to answer questions regarding the audio, and (3) audio classification (CLS), where we would like the model to classify the sound into one or more

2In initial experiments we found unfreezing the audio encoder caused training instability.

[Figure 4]

Figure 4. Construction of ICL samples based on RAG. We use LAION-CLAP to find top-k most similar samples from the database, and use the retrieved audio and text to construct an ICL training sample.

labels corresponding to events, scenes, music genres, instruments, qualities, and others. An overview of all training datasets is shown in Table 1.

ICL datasets. In order to give our model in-context learning and retrieval augmentation abilities, we construct ICL datasets for each of the raw datasets based on kNN computed on audio embeddings. Let Di be the i-th training dataset. For each z = (x,yins,yout) ∈ Di, we find its top-k closest training samples in Di excluding z, where the distance function is ℓ2 in the fused LAION-CLAP embedding space (Wu et al., 2023) for the audio part of the sample. We use Faiss-gpu (Johnson et al., 2019) to accelerate searching. Figure 4 demonstrates this process.

Dataset staging and blending. We use different datasets during the pre-training stage and the supervised fine-tuning (SFT) stage. The selection is based on data quality, diversity, source, and size as described below. 1) Data quality: low quality datasets, including those with low-quality or noisy audio, low-quality text, and inaccurate text annotation, are used for pretraining. 2) Data diversity: datasets with less diversity or strong biases in label distributions are used for pre-training. 3) Data sources: datasets containing AI-generated contents are mostly used for pre-training, whereas some high-quality subsets may be used for SFT. 4) Data sizes: very large datasets may be used both for pre-training and SFT. 5) ICL datasets are used in the SFT stage.

We assign different weights λi when sampling from different datasets based on their sizes, quality, and diversity. The weights are computed from the number of epochs for each dataset. The details of staging and weights can be found in Appendix A.

- Table 1. All datasets used to train our model. The total number of audio-text pairs is approximately 5.9 million. The total length of audio is approximately 18.1 thousand hours.

Audio Type Task Datasets #Audio-Text Pairs

WavCaps (Mei et al., 2023), Macs (Martin Morato & Mesaros, 2021), SoundDescs (Oncescu et al., 2021), Clotho-v2 (Drossos et al., 2020), ∼829 K WavText5K (Deshmukh et al., 2022), LAION-630k (Wu et al., 2023)

CAP

General Sound

AQA Clotho-AQA (Lipping et al., 2022), Open-AQA (Gong et al., 2023b) ∼1970 K CLS

AudioSet (Gemmeke et al., 2017), FSD50k (Fonseca et al., 2021), CochlScene (Jeong & Park, 2022), NonSpeech7K (Rashid et al., 2023), ∼1091 K Chime-Home (Foster et al., 2015), Sonyc-UST (Cartwright et al., 2019)

CAP LP-MusicCaps (Doh et al., 2023), MusicCaps (Agostinelli et al., 2023) ∼1389 K AQA MusicQA (Liu et al., 2023b), MusicAVQA (Li et al., 2022) ∼94 K

Music

NSynth (Engel et al., 2017), MTG-Jamendo (Bogdanov et al., 2019),

CLS

∼459 K FMA (Defferrard et al., 2016), MusDB-HQ (Rafii et al., 2019),

MSP-Podcast (Lotfian & Busso, 2017), Emov-DB (Adigwe et al., 2018) JL-Corpus (James et al., 2018), Tess (Pichora-Fuller & Dupuis, 2020), ∼92 K MELD (Poria et al., 2018), OMGEmotion (Barros et al., 2018)

Speech CLS

- 4.2. Templates Our templates are based on OPT-IML’s template (Iyer

- et al., 2022) and Flamingo’s multimodal template (Alayrac et al., 2022). For a non-interleaved sample, the template is describe below.

<s>{task description}<audio>{instruction} Options:\n- option1\n· · · - optionm <SEP>{output}<EOC></s>

In this template, <audio> is the special token that informs the language model the location of audio in the context. The {task description} is natural language that tells the language model which task it is handling, for example “The task is event classification”. The {instruction} is the language instruction such as a question. The options sentence is to tell the language model all options for classification so that it can classify an audio by outputting free-form text. The {output} is the ground truth output that will be trained. The <SEP> token is a separator that indicates the end of instruction, and <EOC> is the end-of-chunk token that indicates the end of a sample. Below is the template for interleaved (ICL) samples with k + 1 tuples of (audio, instruction, output).

<s>{task description}Here are similar samples. <audio>{instruction1}<SEP>{output1}<EOC> · · · <audio>{instructionk}<SEP>{outputk}<EOC> <audio>{instruction} Options:\n- option1\n· · · - optionm <SEP>{output}<EOC></s>

- 4.3. Multi-Turn Dialogue Dataset

We aim at giving our model stronger abilities in dealing with complicated multi-turn dialogues. To achieve this, we use GPT-4 (Achiam et al., 2023)

to generate two multi-turn dialogue datasets. We construct these datasets based on the strongly labeled AudioSet-SL (Hershey et al., 2021) and MusicCaps (Agostinelli et al., 2023), with thresholding based on LAION-CLAP embeddings (Wu et al., 2023) to filter undesirable samples. We name these two generated datasets AF-Dialogue-AudioSetSL and AF-Dialogue-MusicCaps, respectively. The detailed instructions, filtering method, dataset statistics, and examples are in Appendix B. We use the following template for an s-turn dialogue data sample.

<s>The task is dialogue.<audio> user: {instruction1} assistant: <SEP>{output1}<EOC> · · ·

user: {instructions} assistant: <SEP>{outputs}<EOC></s>

### 5. Experiments

In this section, we answer the following questions:

- Q1. Does Audio Flamingo understand audio better than the state-of-the-art baselines?
- Q2. How significantly does the ICL-based RAG help Audio Flamingo adapt to new tasks?
- Q3. What is Audio Flamingo’s ability to have multiturn dialogues with a user?
- Q4. Which specific configuration of Audio Flamingo works the best overall?
- 5.1. Experimental Setup

We use 8 NVIDIA A100 GPUs to train our model. During pre-training, we use batch size = 384, AdamW optimizer (Loshchilov & Hutter, 2017) with learning rate = 1 × 10−4 and weight decay = 0.1. For efficiency, we use bf16 with automatic mixed precision. During

supervised fine-tuning (SFT), we reduce the batch size to 128, the learning rate to 2 × 10−5, and use fp32 for better numerical precision. We let the maximum number of interleaved samples to be 8 unless specified. We set the maximum number of text tokens to be 512.

We compare to the most recent state-of-the-art baselines on several benchmarks. On each dataset, we choose the best score among all SOTA baselines as the reference value. Unless specified, we report accuracy for question-answering and single-label classification, F1 for multi-label classification, and CIDEr (Vedantam et al., 2015) for captioning and dialogues. Note we use free-form text output to evaluate our model at all times. We use a single model to evaluate on all benchmarks except for dialogues, and a chat model on dialogues.

For zero-shot and few-shot benchmarks, these datasets are excluded from the pre-training sets and SFT sets. For those derived from a parent dataset (e.g. AudioCaps audio are derived from AudioSet), we removed the training samples from the parent set as well as other child sets derived from that parent set.

#### 5.2. Q1: Strong Audio Understanding Ability

We evaluate our model on several in-distribution (traintest) benchmarks, and compare to state-of-the-art audio language model baselines. The results are shown in Table 2. Note that we define F1approx to measure inexact but similar predicted labels in FSD50k, where we consider the prediction to be correct if the sentence BERT similarity between output and ground truth is > 0.8 (Reimers & Gurevych, 2019; 2020). This metric is applied to outputs from baselines as well.

Audio Flamingo can match or outperform SOTA baselines – many of which are much larger LLMs (7B (Gong et al., 2023c;b; Chu et al., 2023) or 13B (Tang

- et al., 2023a)) – on most tasks, indicating our proposed method has strong audio understanding ability. Our model also listens to the audio better. On ClothoAQA, our model has 10.4% higher accuracy than baselines on numerical question answering, indicating our model understands the number of occurrences better. On NSynth, our model has 20.4% higher F1 on quality prediction and 18.6% higher accuracy on source prediction, indicating our model understands the overall quality of audio better. In Appendix C.3, we use qualitative samples to show that our model understands the order of appearance of different sounds, perceives loudness and its change over time, and perceives the distances of sounds from different objects.

#### 5.3. Q2: In-Context Few-Shot Learning

We aim to measure the effect of ICL-based RAG in Audio Flamingo when it is evaluated on unseen datasets.

First, we report the results on several zero-shot benchmarks and comparison with SOTA zero-shot methods in Table 3. The results indicate our method is better on most tasks and has strong generalization ability.

We then apply ICL-based RAG to these benchmarks. We compare to our zero-shot results and the SOTA baseline of audio captioning on AudioCaps. The results on classification are shown in Table 4, and the comparison on retrieval-augmented audio captioning is shown in Table 5. As expected, there is consistent improvement over zero-shot results, with an average improvement over 10% for classification. Our method also significantly outperforms the SOTA retrieval-augmented audio captioning method on AudioCaps. In Appendix C.1, we show Audio Flamingo can adapt to unseen labels. In Appendix C.4, we show Audio Flamingo looks at related retrieval (e.g., by taking key words from retrieved captions), and ignores noisy retrieval.

#### 5.4. Q3: Multi-Turn Dialogues

We measure Audio Flamingo’s ability to answer questions in a multi-turn dialogue setting. The context is more complex and strongly correlated between rounds (e.g. there exist many pronouns and follow-up questions). We fine-tune Audio Flamingo on the two sets that we generated (AF-Dialogue-AudioSetSL and AF-Dialogue-MusicCaps) to obtain a chat model. We evaluate the chat model on the test split of these two dialogue datasets. We take user instructions and let the model generate answers turn-by-turn (where previous generated answers become the chatting history for next generation). We compare to Qwen-Audio (Chu et al., 2023), LTU (Gong et al., 2023c), and MU-LLaMA (Liu et al., 2023b) in Table 6. 3 Our chat model achieves significantly better results than baseline methods. In Appendix C.5, we use qualitative samples to show that our chat model captures context such as prior information and pronouns better.

#### 5.5. Q4: Ablation Studies

Effect of number of few-shot samples. We study different numbers of in-context few-shot samples and evaluate on the few-shot benchmarks. In Figure 5, we plot the relative improvements over zero-shot results. Results show a clear trend that having more ICL samples improves few-shot results, and the improvement

3While the baseline methods claimed to support multi-turn dialogues, we were unable to find quantitative evidence.

- Table 2. Evaluation of Audio Flamingo versus SOTA baseline methods on in-distribution benchmarks. Reference values are the SOTA models for each task. Audio Flamingo shows strong audio understanding ability on captioning, question answering, and audio classification.

Dataset Task Metric Previous SOTA ↑ Ours ↑ Clotho-v2 CAP CIDEr 0.441 (Chu et al., 2023) 0.465

ClothoAQAunanimous AQA ACC 74.9% (Chu et al., 2023) 86.9% ClothoAQAnon-binary AQA ACC 29.1% (Deshmukh et al., 2023) 49.5% ClothoAQAnumerical AQA ACC 26.2% (Deshmukh et al., 2023) 36.4%

MusicAVQAaudio-only AQA ACC 72.1% (Chu et al., 2023) 71.6% CochlScene CLS ACC 91.6% (Deshmukh et al., 2023) 83.0%

NonSpeech7k CLS ACC 79.0% (Rashid et al., 2023) 85.1% FSD50k CLS F1approx 65.6% (Deshmukh et al., 2023) 69.7% NSinstrument CLS ACC 78.8% (Chu et al., 2023) 77.1% NSquality CLS F1 46.3% (Deshmukh et al., 2023) 66.7% NSsource CLS ACC 60.1% (Deshmukh et al., 2023) 78.7%

- Table 3. Evaluation of Audio Flamingo versus SOTA baseline methods on zero-shot benchmarks. Reference values are the SOTA models for each task. Audio Flamingo shows strong zero-shot generalization ability.

Dataset Task Metric Previous SOTA (0-shot) ↑ Ours (0-shot) ↑

AudioCaps (Kim et al., 2019) CAP CIDEr 0.281 (Salewski et al., 2023) 0.502 CREMA-D (Cao et al., 2014) CLS ACC 18.5% (Deshmukh et al., 2023) 26.5% Ravdess (Livingstone & Russo, 2018) CLS ACC 21.7% (Elizalde et al., 2023b) 20.9% US8K (Salamon et al., 2014) CLS ACC 71.9% (Deshmukh et al., 2023) 75.0% GTZAN (Sturm, 2013) CLS ACC 71.0% (Han et al., 2023) 67.9% Medley-solos-DB (Lostanlen et al., 2019) CLS ACC 61.3% (Deshmukh et al., 2023) 92.7%

- Table 4. Evaluation of few-shot results of Audio Flamingo with ICL-based RAG. ∆ is the absolute improvement of few-shot over zero-shot results in Table 3. ICL-based RAG leads to consistent improvement over zero-shot results.

Table 5. Evaluation of retrieval-augmented audio captioning on AudioCaps. We compare Audio Flamingo to the SOTA baseline RECAP (Ghosh et al., 2023). Audio Flamingo achieves significantly better results than RECAP.

Method RECAP Ours Ours Ours # Shots 4 4 8 16 CIDEr ↑ 0.359 0.518 0.538 0.546

Dataset Ours (8-shot) ↑ ∆ ↑ CREMA-D 31.8% 5.3% Ravdess 35.2% 14.3% US8K 94.7% 19.4% GTZAN 79.5% 11.6%

(Elizalde et al., 2023b), and LAION-CLAP (Wu et al., 2023). In terms of the LM backbone, the instructiontuned opt is better in most tasks. As of audio encoders, LAION-CLAP is worse in most tasks, ClapCap is better in open-ended tasks, and Clap2023 is better in most close-ended tasks.

- Medley-solos-DB 95.7% 3.0% Dataset Ours (16-shot) ↑ ∆ ↑ CREMA-D 35.4% 8.9% Ravdess 40.2% 19.3% US8K 91.9% 16.9% GTZAN 76.3% 8.4%

- Medley-solos-DB 96.0% 3.3%

Effect of training data. In Figure 7, we compare Audio Flamingo trained on three different training sets to the Pengi baseline: (1) no pretraining, SFT on our best available dataset that is strictly a subset of Pengi’s training set, (2) pretraining and SFT on this strict subset, and (3) pretraining and SFT on our curated datasets. Audio Flamingo achieves better evaluation results than Pengi even if no extra data is used. In addition, increasing the data amount in pretraining and SFT can improve the results on average.

highly depends on the dataset.

Effect of architecture. In Figure 6, we compare results between opt-1.3b (Zhang et al., 2022) and the instruction-tuned opt-iml-max-1.3b (Iyer et al., 2022), and also compare different audio encoders including ClapCap (Elizalde et al., 2023b), Clap2023

Table 6. Evaluation of Audio Flamingo versus baseline methods on the multi-turn dialogue test sets. A stands for AF-Dialogue-AudioSetSL, M stands for AF-Dialogue-MusicCaps, and the superscript H stands for an additional held-out testset generated with gpt-3.5-turbo. We report CIDEr, Bleu4 (Papineni et al., 2002), and Rouge-L (R-L) (Lin, 2004). Methods with the † superscript are fine-tuned on our dialogue training sets, and methods without † are evaluated zero-shot on the dialogue test sets. Audio Flamingo significantly outperforms larger baseline models in all settings, indicating strong dialogue ability of our proposed model.

Testset Method CIDEr ↑ Bleu4 ↑ R-L ↑

A Qwen-Audio 0.507 0.060 0.292 A LTU 0.580 0.122 0.324 A LTU† 0.823 0.153 0.403 A Ours† 1.622 0.237 0.473

AH LTU† 0.523 0.095 0.343 AH Ours† 1.904 0.219 0.476

M MU-LLaMA 0.585 0.083 0.348 M LTU 0.168 0.065 0.217 M LTU† 0.419 0.108 0.336 M Ours† 1.143 0.142 0.417

MH LTU† 0.558 0.083 0.347 MH Ours† 1.350 0.207 0.448

### 6. Conclusion and Future Work

In this paper, we present Audio Flamingo, an audio language model with a series of innovations that achieves the state-of-the-art results on several close-ended and open-ended audio understanding tasks without task specific fine-tuning. It also has strong ICL and RAG abilities, and has the state-of-the-art few-shot learning results. Furthermore, we design a dataset generation strategy and introduce two dialogue datasets, enabling Audio Flamingo to chat with a user about the audio for multiple rounds and achieve state-of-the-art results on dialogue benchmarks.

Our model has several limitations, which we plan to address in future work. One important future direction is to investigate scaling strategies for using larger LMs. Assuming that larger LMs could have better knowledge and stronger ability to follow instructions, we believe that Audio Flamingo could benefit from a larger LM. A second future direction is to investigate complex speech-related tasks beyond transcription. This requires our model to condition on dense embeddings (Chen et al., 2023). This modification is straightforward in Audio Flamingo as the architecture is flexible enough to support new embeddings through the addition of new cross-attention heads. Another future direction is to build a audio language model that can

ImprovementoverZero-Shot

Dataset

30.0%

Medley-solos-DB

AudioCaps

20.0%

CREMA-D

US8K

GTZAN

10.0%

0.0%

0 4 8 16

Number of ICL Samples

- Figure 5. Relative improvement of few-shot results over zeroshot results under different number of ICL samples. Using more ICL samples consistently improves few-shot results, and the benefit is dataset-dependent.

opt-iml-max-1.3b + clapcap

opt-iml-max-1.3b + clap2023

opt-iml-max-1.3b + laion-clap

opt-1.3b

+ clapcap

Architecture

10.0%

5.0%

0.0%

5.0%

Relativedifferencefrom

opt-iml-max-1.3b+clapcap

Dataset

Clotho-v2

ClothoAQA (unanimous) ClothoAQA (non-binary) ClothoAQA (numerical)

MusicAVQA

CochlScene

FSD50k

NSynth

NonSpeech7k

- Figure 6. Relative difference of results from different LM backbones and audio encoders. The opt-1.3b model without instruction-tuning is systematically worse than the instruction-tuned opt-iml-max-1.3b. LAION-CLAP is worse than ClapCap in most cases. Clap2023 is better than ClapCap in close-ended tasks, but worse in open-ended tasks including captioning and open question-answering.

Pengi baseline No pretraining SFT on Pengi subset

Pretrain and SFT on Pengi subset

Pretrain and SFT on our datasets

Pretraining and SFT datasets

0%

20%

40%

60%

Relativedifferencefrom

thePengibaseline

Dataset

Clotho-v2

ClothoAQA (unanimous) ClothoAQA (non-binary) ClothoAQA (numerical)

CochlScene

FSD50k

NSynth

- Figure 7. Relative difference of results from different training datasets compared to the Pengi baseline. Audio Flamingo achieves better results than Pengi even if no extra data is used. Larger scale pretraining and SFT lead to better results on these benchmarks on average.

output both text and audio and follow more complex interleaved instructions. Finally, a future direction towards unifying more modalities is to combine the audio understanding abilities of our model with visual language models (Alayrac et al., 2022) such that one model could understand image, video, and audio.

### Acknowledgement

We thank Siddharth Gururani, Zihan Liu, Mostofa Patwary, Shrimai Prabhumoye, and Chen Zhu for helpful discussions. We thank Ke Chen and Yuan Gong for help on sharing datasets with us.

### Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. The proposed method facilitates automation in the audio-language domain and may be used in a variety of scenarios such as education, healthcare, environment, industry, music, etc. Careful use of the model is essential to ensure compliance with copyright restrictions in certain situations.

### References

Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F. L., Almeida, D., Altenschmidt, J., Altman, S., Anadkat, S., et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

Adigwe, A., Tits, N., Haddad, K. E., Ostadabbas, S., and Dutoit, T. The emotional voices database: Towards controlling the emotion dimension in voice generation systems. arXiv preprint arXiv:1806.09514, 2018.

Agostinelli, A., Denk, T. I., Borsos, Z., Engel, J., Verzetti, M., Caillon, A., Huang, Q., Jansen,

- A., Roberts, A., Tagliasacchi, M., et al. Musiclm: Generating music from text. arXiv preprint arXiv:2301.11325, 2023.

Alayrac, J.-B., Donahue, J., Luc, P., Miech, A., Barr, I., Hasson, Y., Lenc, K., Mensch, A., Millican, K., Reynolds, M., et al. Flamingo: a visual language model for few-shot learning. Advances in Neural Information Processing Systems, 35:23716–23736, 2022.

Barros, P., Churamani, N., Lakomkin, E., Siqueira, H., Sutherland, A., and Wermter, S. The omg-emotion behavior dataset. In 2018 International Joint Conference on Neural Networks (IJCNN), pp. 1–7. IEEE, 2018.

Bogdanov, D., Won, M., Tovstogan, P., Porter, A., and Serra, X. The mtg-jamendo dataset for automatic music tagging. ICML, 2019.

Borgeaud, S., Mensch, A., Hoffmann, J., Cai, T., Rutherford, E., Millican, K., Van Den Driessche, G. B., Lespiau, J.-B., Damoc, B., Clark, A., et al. Improving language models by retrieving from trillions

of tokens. In International conference on machine learning, pp. 2206–2240. PMLR, 2022.

Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.

Cao, H., Cooper, D. G., Keutmann, M. K., Gur, R. C., Nenkova, A., and Verma, R. Crema-d: Crowdsourced emotional multimodal actors dataset. IEEE transactions on affective computing, 5(4):377–390, 2014.

Cartwright, M., Mendez, A. E. M., Cramer, A., Lostanlen, V., Dove, G., Wu, H.-H., Salamon, J., Nov, O., and Bello, J. Sonyc urban sound tagging (sonyc-ust): A multilabel dataset from an urban acoustic sensor network. 2019.

Chen, K., Du, X., Zhu, B., Ma, Z., Berg-Kirkpatrick, T., and Dubnov, S. Hts-at: A hierarchical tokensemantic audio transformer for sound classification and detection. In ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 646–650. IEEE, 2022.

Chen, Z., Huang, H., Andrusenko, A., Hrinchuk, O., Puvvada, K. C., Li, J., Ghosh, S., Balam, J., and Ginsburg, B. Salm: Speech-augmented language model with in-context learning for speech recognition and translation. arXiv preprint arXiv:2310.09424, 2023.

Chu, Y., Xu, J., Zhou, X., Yang, Q., Zhang, S., Yan, Z., Zhou, C., and Zhou, J. Qwen-audio: Advancing universal audio understanding via unified large-scale audio-language models. arXiv preprint arXiv:2311.07919, 2023.

Defferrard, M., Benzi, K., Vandergheynst, P., and Bresson, X. Fma: A dataset for music analysis. arXiv preprint arXiv:1612.01840, 2016.

De´fossez, A., Copet, J., Synnaeve, G., and Adi, Y. High fidelity neural audio compression. arXiv preprint arXiv:2210.13438, 2022.

Deshmukh, S., Elizalde, B., and Wang, H. Audio retrieval with wavtext5k and clap training. arXiv preprint arXiv:2209.14275, 2022.

Deshmukh, S., Elizalde, B., Singh, R., and Wang, H. Pengi: An audio language model for audio tasks. arXiv preprint arXiv:2305.11834, 2023.

Doh, S., Choi, K., Lee, J., and Nam, J. Lp-musiccaps: Llm-based pseudo music captioning. arXiv preprint arXiv:2307.16372, 2023.

Driess, D., Xia, F., Sajjadi, M. S., Lynch, C., Chowdhery, A., Ichter, B., Wahid, A., Tompson, J., Vuong, Q., Yu, T., et al. Palm-e: An embodied multimodal language model. arXiv preprint arXiv:2303.03378, 2023.

Drossos, K., Lipping, S., and Virtanen, T. Clotho: An audio captioning dataset. In ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 736–740. IEEE, 2020.

Duan, H., Wei, J., Wang, C., Liu, H., Fang, Y., Zhang, S., Lin, D., and Chen, K. Botchat: Evaluating llms’ capabilities of having multi-turn dialogues. arXiv preprint arXiv:2310.13650, 2023.

Elizalde, B., Deshmukh, S., Al Ismail, M., and Wang, H. Clap learning audio concepts from natural language supervision. In ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 1–5. IEEE, 2023a.

Elizalde, B., Deshmukh, S., and Wang, H. Natural language supervision for general-purpose audio representations, 2023b. URL https://arxiv.org/abs/ 2309.05767.

Engel, J., Resnick, C., Roberts, A., Dieleman, S., Eck, D., Simonyan, K., and Norouzi, M. Neural audio synthesis of musical notes with wavenet autoencoders, 2017.

Fonseca, E., Favory, X., Pons, J., Font, F., and Serra, X. Fsd50k: an open dataset of human-labeled sound events. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 30:829–852, 2021.

Foster, P., Sigtia, S., Krstulovic, S., Barker, J., and Plumbley, M. D. Chime-home: A dataset for sound source recognition in a domestic environment. In 2015 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA), pp. 1–5. IEEE, 2015.

Gao, H., Ni, J., Qian, K., Zhang, Y., Chang, S., and Hasegawa-Johnson, M. Wavprompt: Towards fewshot spoken language understanding with frozen language models. arXiv preprint arXiv:2203.15863, 2022.

Gardner, J., Durand, S., Stoller, D., and Bittner, R. M. Llark: A multimodal foundation model for music. arXiv preprint arXiv:2310.07160, 2023.

Gemmeke, J. F., Ellis, D. P., Freedman, D., Jansen, A., Lawrence, W., Moore, R. C., Plakal, M., and Ritter, M. Audio set: An ontology and human-labeled dataset for audio events. In 2017 IEEE international conference on acoustics, speech and signal processing (ICASSP), pp. 776–780. IEEE, 2017.

Ghosh, S., Kumar, S., Evuru, C. K. R., Duraiswami, R., and Manocha, D. Recap: Retrieval-augmented audio captioning. arXiv preprint arXiv:2309.09836, 2023.

Gong, Y., Chung, Y.-A., and Glass, J. Ast: Audio spectrogram transformer. arXiv preprint arXiv:2104.01778, 2021.

Gong, Y., Khurana, S., Karlinsky, L., and Glass, J. Whisper-at: Noise-robust automatic speech recognizers are also strong general audio event taggers. arXiv preprint arXiv:2307.03183, 2023a.

Gong, Y., Liu, A., Luo, H., Karlinsky, L., and Glass, J. Joint audio and speech understanding. In IEEE Automatic Speech Recognition and Understanding Workshop, 2023b.

Gong, Y., Luo, H., Liu, A. H., Karlinsky, L., and Glass, J. Listen, think, and understand. arXiv preprint arXiv:2305.10790, 2023c.

Guu, K., Lee, K., Tung, Z., Pasupat, P., and Chang, M. Retrieval augmented language model pre-training. In International conference on machine learning, pp. 3929–3938. PMLR, 2020.

Han, J., Zhang, R., Shao, W., Gao, P., Xu, P., Xiao, H., Zhang, K., Liu, C., Wen, S., Guo, Z., et al. Imagebind-llm: Multi-modality instruction tuning. arXiv preprint arXiv:2309.03905, 2023.

Hershey, S., Ellis, D. P., Fonseca, E., Jansen, A., Liu, C., Moore, R. C., and Plakal, M. The benefit of temporally-strong labels in audio event classification. In ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 366–370. IEEE, 2021.

Hsu, M.-H., Chang, K.-W., Li, S.-W., and Lee, H.y. An exploration of in-context learning for speech language model. arXiv preprint arXiv:2310.12477, 2023.

Huang, Q., Jansen, A., Lee, J., Ganti, R., Li, J. Y., and Ellis, D. P. Mulan: A joint embedding of music audio and natural language. arXiv preprint arXiv:2208.12415, 2022.

Huang, R., Li, M., Yang, D., Shi, J., Chang, X., Ye, Z., Wu, Y., Hong, Z., Huang, J., Liu, J., et al. Audiogpt: Understanding and generating speech, music, sound, and talking head. arXiv preprint arXiv:2304.12995, 2023.

Iyer, S., Lin, X. V., Pasunuru, R., Mihaylov, T., Simig, D., Yu, P., Shuster, K., Wang, T., Liu, Q., Koura, P. S., et al. Opt-iml: Scaling language model instruction meta learning through the lens of generalization. arXiv preprint arXiv:2212.12017, 2022.

James, J., Tian, L., and Watson, C. An open source emotional speech corpus for human robot interaction applications. Interspeech 2018, 2018.

Jeong, I.-Y. and Park, J. Cochlscene: Acquisition of acoustic scene data using crowdsourcing. In 2022 Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC), pp. 17–21. IEEE, 2022.

Johnson, J., Douze, M., and J´egou, H. Billion-scale similarity search with GPUs. IEEE Transactions on Big Data, 7(3):535–547, 2019.

Karpukhin, V., O˘guz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen, D., and Yih, W.-t. Dense passage retrieval for open-domain question answering. arXiv preprint arXiv:2004.04906, 2020.

Khomenko, V., Shyshkov, O., Radyvonenko, O., and Bokhan, K. Accelerating recurrent neural network training using sequence bucketing and multi-gpu data parallelization. In 2016 IEEE First International Conference on Data Stream Mining & Processing (DSMP), pp. 100–103. IEEE, 2016.

Kim, C. D., Kim, B., Lee, H., and Kim, G. Audiocaps: Generating captions for audios in the wild. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pp. 119–132, 2019.

Kong, Q., Cao, Y., Iqbal, T., Wang, Y., Wang, W., and Plumbley, M. D. Panns: Large-scale pretrained audio neural networks for audio pattern recognition. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 28:2880–2894, 2020.

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., K¨uttler, H., Lewis, M., Yih, W.-t., Rockt¨aschel, T., et al. Retrieval-augmented generation for knowledge-intensive nlp tasks. Advances in Neural Information Processing Systems, 33:9459– 9474, 2020.

Li, G., Wei, Y., Tian, Y., Xu, C., Wen, J.-R., and Hu, D. Learning to answer questions in dynamic audiovisual scenarios. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 19108–19118, 2022.

Li, J., Li, D., Savarese, S., and Hoi, S. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. arXiv preprint arXiv:2301.12597, 2023a.

Li, Y., Yuan, R., Zhang, G., Ma, Y., Chen, X., Yin, H., Lin, C., Ragni, A., Benetos, E., Gyenge, N., et al. Mert: Acoustic music understanding model with large-scale self-supervised training. arXiv preprint arXiv:2306.00107, 2023b.

Lin, C.-Y. Rouge: A package for automatic evaluation of summaries. In Text summarization branches out, pp. 74–81, 2004.

Lipping, S., Sudarsanam, P., Drossos, K., and Virtanen, T. Clotho-aqa: A crowdsourced dataset for audio question answering. In 2022 30th European Signal Processing Conference (EUSIPCO), pp. 1140–1144.

IEEE, 2022.

Liu, H., Li, C., Wu, Q., and Lee, Y. J. Visual instruction tuning. arXiv preprint arXiv:2304.08485, 2023a.

Liu, S., Hussain, A. S., Sun, C., and Shan, Y. Music understanding llama: Advancing text-to-music generation with question answering and captioning. arXiv preprint arXiv:2308.11276, 2023b.

Livingstone, S. R. and Russo, F. A. The ryerson audio-visual database of emotional speech and song (ravdess): A dynamic, multimodal set of facial and vocal expressions in north american english. PloS one, 13(5):e0196391, 2018.

Loshchilov, I. and Hutter, F. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.

Lostanlen, V., Cella, C.-E., Bittner, R., and Essid, S. Medley-solos-DB: a cross-collection dataset for musical instrument recognition, February 2019. URL https://doi.org/10.5281/zenodo.1344103.

Lotfian, R. and Busso, C. Building naturalistic emotionally balanced speech corpus by retrieving emotional speech from existing podcast recordings. IEEE Transactions on Affective Computing, 10(4):471–483, 2017.

Lyu, C., Wu, M., Wang, L., Huang, X., Liu, B., Du, Z., Shi, S., and Tu, Z. Macaw-llm: Multi-modal language modeling with image, audio, video, and text integration. arXiv preprint arXiv:2306.09093, 2023.

Martin Morato, I. and Mesaros, A. Diversity and bias in audio captioning datasets. 2021.

Mei, X., Meng, C., Liu, H., Kong, Q., Ko, T., Zhao, C., Plumbley, M. D., Zou, Y., and Wang, W. Wavcaps: A chatgpt-assisted weakly-labelled audio captioning dataset for audio-language multimodal research. arXiv preprint arXiv:2303.17395, 2023.

Mohanty, S. P. Sound Of 114 Species Of Birds Till 2022. URL https://www. kaggle.com/datasets/soumendraprasad/ sound-of-114-species-of-birds-till-2022.

Moon, S., Madotto, A., Lin, Z., Nagarajan, T., Smith, M., Jain, S., Yeh, C.-F., Murugesan, P., Heidari, P., Liu, Y., et al. Anymal: An efficient and scalable anymodality augmented language model. arXiv preprint arXiv:2309.16058, 2023.

Oncescu, A.-M., Koepke, A., Henriques, J. F., Akata, Z., and Albanie, S. Audio retrieval with natural language queries. arXiv preprint arXiv:2105.02192, 2021.

Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., et al. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35:27730–27744, 2022.

Papineni, K., Roukos, S., Ward, T., and Zhu, W.-J. Bleu: a method for automatic evaluation of machine translation. In Proceedings of the 40th annual meeting of the Association for Computational Linguistics, pp. 311–318, 2002.

Park, J., Cho, Y., Sim, G., Lee, H., and Choo, J. Enemy spotted: In-game gun sound dataset for gunshot classification and localization. In 2022 IEEE Conference on Games (CoG), pp. 56–63. IEEE, 2022.

Pichora-Fuller, M. K. and Dupuis, K. Toronto emotional speech set (TESS), 2020. URL https://doi. org/10.5683/SP2/E8H2MF.

Poria, S., Hazarika, D., Majumder, N., Naik, G., Cambria, E., and Mihalcea, R. Meld: A multimodal multi-party dataset for emotion recognition in conversations. arXiv preprint arXiv:1810.02508, 2018.

Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., and Sutskever, I. Robust speech recognition via large-scale weak supervision. In International Conference on Machine Learning, pp. 28492–28518. PMLR, 2023.

Rafii, Z., Liutkus, A., St¨oter, F.-R., Mimilakis, S. I., and Bittner, R. Musdb18-hq - an uncompressed version of musdb18, August 2019. URL https:// doi.org/10.5281/zenodo.3338373.

Rashid, M. M., Li, G., and Du, C. Nonspeech7k dataset: Classification and analysis of human nonspeech sound. IET Signal Processing, 17(6):e12233, 2023.

Reimers, N. and Gurevych, I. Sentence-bert: Sentence embeddings using siamese bert-networks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics, 11 2019. URL https://arxiv.org/abs/1908.10084.

Reimers, N. and Gurevych, I. Making monolingual sentence embeddings multilingual using knowledge distillation. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics, 11 2020. URL https://arxiv.org/abs/2004.09813.

Rubenstein, P. K., Asawaroengchai, C., Nguyen, D. D., Bapna, A., Borsos, Z., Quitry, F. d. C., Chen, P., Badawy, D. E., Han, W., Kharitonov, E., et al. Audiopalm: A large language model that can speak and listen. arXiv preprint arXiv:2306.12925, 2023.

Salamon, J., Jacoby, C., and Bello, J. P. A dataset and taxonomy for urban sound research. In Proceedings of the 22nd ACM international conference on Multimedia, pp. 1041–1044, 2014.

Salewski, L., Fauth, S., Koepke, A., and Akata, Z. Zeroshot audio captioning with audio-language model guidance and audio context keywords. arXiv preprint arXiv:2311.08396, 2023.

Sturm, B. L. The gtzan dataset: Its contents, its faults, their effects on evaluation, and its future use. arXiv preprint arXiv:1306.1461, 2013.

Tang, C., Yu, W., Sun, G., Chen, X., Tan, T., Li, W., Lu, L., Ma, Z., and Zhang, C. Salmonn: Towards generic hearing abilities for large language models. arXiv preprint arXiv:2310.13289, 2023a.

Tang, Z., Yang, Z., Khademi, M., Liu, Y., Zhu, C., and Bansal, M. Codi-2: In-context, interleaved, and interactive any-to-any generation. arXiv preprint arXiv:2311.18775, 2023b.

Tsimpoukelli, M., Menick, J. L., Cabi, S., Eslami, S., Vinyals, O., and Hill, F. Multimodal few-shot learning with frozen language models. Advances in Neural Information Processing Systems, 34:200–212, 2021.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser,  L., and Polosukhin, I. Attention is all you need. Advances in neural information processing systems, 30, 2017.

Vedantam, R., Lawrence Zitnick, C., and Parikh, D. Cider: Consensus-based image description evaluation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 4566–4575, 2015.

Wang, S., Yang, C.-H. H., Wu, J., and Zhang, C. Can whisper perform speech-based in-context learning. arXiv preprint arXiv:2309.07081, 2023.

Wei, J., Bosma, M., Zhao, V. Y., Guu, K., Yu, A. W., Lester, B., Du, N., Dai, A. M., and Le, Q. V. Finetuned language models are zero-shot learners. arXiv preprint arXiv:2109.01652, 2021.

Won, M., Hung, Y.-N., and Le, D. A foundation model for music informatics. arXiv preprint arXiv:2311.03318, 2023.

Wu, Y., Chen, K., Zhang, T., Hui, Y., Berg-Kirkpatrick, T., and Dubnov, S. Large-scale contrastive languageaudio pretraining with feature fusion and keyword-tocaption augmentation. In ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 1–5. IEEE, 2023.

Yang, Z., Ping, W., Liu, Z., Korthikanti, V., Nie, W., Huang, D.-A., Fan, L., Yu, Z., Lan, S., Li, B., et al. Re-vilm: Retrieval-augmented visual language model for zero and few-shot image captioning. In EMNLP, 2023.

Zhang, S., Roller, S., Goyal, N., Artetxe, M., Chen, M., Chen, S., Dewan, C., Diab, M., Li, X., Lin, X. V., et al. Opt: Open pre-trained transformer language models. arXiv preprint arXiv:2205.01068, 2022.

Zhao, Z., Guo, L., Yue, T., Chen, S., Shao, S., Zhu, X., Yuan, Z., and Liu, J. Chatbridge: Bridging modalities with large language model as a language catalyst. arXiv preprint arXiv:2305.16103, 2023.

### A. Dataset Staging, Weights, and Templates

Table 1 includes an overview of datasets (by type) we use to train Audio Flamingo. We construct instructions for each task and dataset. Below are all instruction templates we use.

Audio Captioning:

- ◦ Describe the sound/music in a sentence.
- ◦ Describe the sound/music at length. Audio Question Answering:
- ◦ {question}
- ◦ Please answer this question: {question}
- ◦ Please answer this question: {question}. Options:\n- yes\n- no Audio Classification:
- ◦ Classify this sound. (Options: ...)
- ◦ Describe the sound in {number} words.
- ◦ What is the emotion of this speech? (Options: ...)
- ◦ What is the instrument/genre of this music? (Options: ...)
- ◦ This music note is produced by

The detailed pre-training datasets and their number of epochs are shown in Table 7. The detailed SFT datasets and their number of epochs are shown in Table 8.

Table 7. Pre-training datasets and epochs.

Dataset Audio Length #Audio-Text Pairs Epochs OpenAQA 693.2 hrs 1959.8K 1.0

Laion630kBBCSoundEffects 456.9 hrs 15.1K 5.0 Laion630kFreesound 2494.8 hrs 306.5K 1.0

SoundDescs 749.7 hrs 23.1K 1.0 WavCaps 3793.3 hrs 402.6 K 1.75 AudioSet 2617.8 hrs 950.8K 1.0

WavText5K 23.8 hrs 4.3K 3.0 MSP-Podcast 73.9 hrs 45.1K 1.2

MELD 8.7 hrs 32.9K 2.4 MusicAVQAaudio-visual 142.4 hrs 17.9K 3.0

MusicQA 62.9 hrs 70K 1.2 LP-MusicCapsMSD 5805.7 hrs 1331.8K 1.0

NSynth 321.3 hrs 289.2K 0.4 MTG-Jamendo 3768.9 hrs 55.6K 1.0

Table 8. SFT datasets and epochs.

Dataset Audio Length #Audio-Text Pairs Epochs ICL Dataset Epochs

ClothoAQA 7.4 hrs 9.7K 3.5 0.5 OpenAQA 693.2 hrs 1959.8K 0.1 Clotho-v2 24.0 hrs 19.2K 2.0 0.5

Laion630kEpidemic 209.4 hrs 40.7K 0.8 0.2

MACS 10.9 hrs 17.3K 0.8 0.2 FSD50k 80.8 hrs 41.0K 0.9 0.3

CochlScene 169.0 hrs 60.9K 1.2 0.3 NonSpeech 7k 6.2 hrs 6.3K 2.4 0.6

Chime-home 5.0 hrs 4.5K 1.5 0.5

Sonyc-UST 34.9 hrs 27.9K 0.8 0.2 Emov-DB 7.8 hrs 6.8K 1.6 0.4 JL-Corpus 1.4 hrs 2.4K 6.0 1.5

Tess 1.6 hrs 2.8K 2.0 0.5 OMGEmotion 3.0 hrs 1.7K 3.0 -

MusicAVQAaudio-only 77.1 hrs 5.7K 5.0 1.0

MusicQA 62.9 hrs 70K 0.35 0.05 LP-MusicCapsMSD 5805.7 hrs 1331.8K 0.025 0.007 LP-MusicCapsMTT 126.4 hrs 46.9K 0.8 0.2 LP-MusicCapsMC 7.4 hrs 7.9K 2.0 -

MusicCaps 7.4 hrs 2.6K 6.0 -

NSynth 321.3 hrs 289.2K 1.0 1.0 MTG-Jamendo 3768.9 hrs 55.6K 0.1 -

MusDB-HQ 29.1 hrs 10.2K 1.0 FMA 860.7 hrs 104.2K 0.4 0.1

- B. Generated dialogue datasets

- B.1. Overview

In this section, we introduce methods to generate our AF-Dialogue-AudioSetSL and AF-Dialogue-MusicCaps datasets with GPT-4 (Achiam et al., 2023). AF-Dialogue-AudioSetSL is generated based on the annotated events and timestamps of strongly labeled AudioSet-SL (Gemmeke et al., 2017; Hershey et al., 2021). There are 76k dialogues in the train split and 1.5k dialogues in test split. AF-Dialogue-MusicCaps is generated based on tags and descriptions of MusicCaps (Agostinelli et al., 2023). There are 3.3k dialogues in the train split and 1.4k dialogues in the test split. Each training sample has one to four rounds (QA pairs), and each testing sample has three or four rounds.

#### B.2. Prompts

Below we show the prompts used to generate audio-based multi-turn dialogues from GPT-4 for the strongly labeled AudioSet-SL dataset (Gemmeke et al., 2017; Hershey et al., 2021) and the MusicCaps dataset (Agostinelli et al., 2023) respectively.

sys prompt: Based on the sound events, create a dialogue between you (the assistant) and a person (the user) about the events in the audio. Each dialogue should consist of:

1. A user examines the audio, and sends a reasonable and creative message to the assistant. 2. Once the audio is provided, the assistant thoroughly perceives and comprehends them, responding with helpful answers that provide comprehensive reasoning. Do not include timestamps in the answer provided by the assistant. 3. Considering the past dialogue i.e. the question and the answer in the previous timestep, the user chooses another question or further inquiry. 4. The assistant provides a helpful answer providing comprehensive reasoning regarding the audio content. The audios are 10 seconds long and the sound events are segmented based on where they occur in the audio. All the questions asked by the user should be diverse and complex.

- Example 1: Sound Events: Sound of Ambulance (siren) (High-pitched and wailing): [0.0s-1.0s]; Sound of Traffic noise, roadway noise (Droning, loud and intrusive): [0.0s-10.0s]; Sound of Accelerating, revving, vroom (Highpitched, short and intense): [2.0s-10.0s]; Sound of Generic impact sounds (Loud and sharp): [6.7s-6.8s]. Dialogue: User: What does the audio sound like? Assistant: An ambulance siren echoes while traffic noise fades, and an engine revs. User: Can you describe in more detail about the engine revs? Assistant: The engine revving is highpitched, short and intense. User: From the highpitched sound, can you tell me what vehicle it was? Assistant: It is hard to tell this from the information provided. User: Are there any other events in the audio that might be relevant? Assistant: There is a sound of impact which is very loud and sharp.
- Example 2: Sound Events: Sound of Outside, rural or natural: [0.000s-10.000s]; Sound of Liquid: [0.474s-

- 0.680s], [4.249s-5.292s]; Sound of Gurgling, bubbling: [0.474s-0.680s], [4.249s-5.292s]; Sound of Splash, splatter: [1.620s-3.116s]; Sound of Laughter: [2.745s-4.029s], [5.299s-10.000s]; Sound of Chuckle, chortle: [2.752s-4.029s]; Sound of Speech: [4.551s-4.935s]; Sound of Male speech, man speaking: [4.557s-4.935s]; Sound of Belly laugh: [5.292s-10.000s] Dialogue: User: What are the main events in the audio? Assistant: Liquid sounds, laughter, speech, and a man speaking can be heard in a rural setting. User: Is is he speaking to other people? Assistant: As there is sound of laughter present from other people, the man might be speaking to others. User: Can you explain in more detail about the first event? Assistant: The liquid sounds are more of splashing, thrashing and splattering sounds. User: Are there other people laughing with him? Assistant: Yes, there are other laughter voices along with that of the man.

Based on the above examples, design a four-turn dialogue between the user and assistant for the example given below. The question asked by the user should use pronouns such as ‘he’, ‘she’, ‘it’, ‘they’ and ‘them’ while referring to entities that have already occurred in the past questions or answers. Format each QA pair in a single line as a JSON dictionary (key “user” for question, and “assistant” for answer, wrapped with { and }). Do not include any other explanation.

- Example 3:

sys prompt: Based on the music, create a dialogue between you (the assistant) and a person (the user) about the events in the music audio. Each dialogue should consist of:

- 1. A user examines the music, and sends a reasonable and creative message to the assistant. 2. Once the music audio is provided, the assistant thoroughly perceives and comprehends them, responding with helpful answers that provide comprehensive reasoning. Do not include timestamps in the answer provided by the assistant.

3. Considering the past dialogue i.e. the question and the answer in the previous timestep, the user chooses another question or further inquiry. 4. The assistant provides a helpful answer providing comprehensive reasoning regarding the music content. The more complex and diverse the question, the better. You can use the list of questions below while asking questions.

List of questions: Is it a live recording or a studio recording? Is the guitar acoustic or electric? Is it playing chords, a riff or a solo? What music style or genre is it? What key is it in? How about the time signature? What’s the tempo? Slow, Medium or Fast? Does it have vocals? Is it low pitched or high pitched? Is the person speaking fast or slow? What is the language and accent? What emotion is conveyed by the voice? What are the vocal characteristics of that emotion? How old is the person likely to be given their vocal characteristics?

- Example 1: Music description: House music with a thought provoking message in the beginning that eventually builds up and can be used to create the sense of a social revolution. Dialogue: User: What type of music is this? Assistant: It is house music with synth and drums and a man speaking at the beginning. User: What is his tone while speaking?
- Example 2: He has an intense and energetic tone in his voice. User: Does he speak till the end? Assistant: No, he stops speaking halfway and then it is followed by house music. User: What is the source of this music? Assistant: This is electronic music.

- Example 2: Music description: Energetic bluesy song with a harmonica and horn section in musical dialogue. Dialogue: User: What instruments are playing in this music? Assistant: This is a jazzy tune with harmonica, trumpet and drums playing. User: Which key is the music playing in? Assistant: The key is A Major. User: How will you explain the tempo of this piece? Assistant: The music is bouncy and positive with high beats per minute in the beginning and then changes signature halfway through. User: What is the source of this music? Assistant: This is fusion jazz as it has elements of bass and synthesizer. Based on above examples, design a four-turn dialogue between the user and assistant for the example given below. The question asked by the user should use pronouns such as ‘he’, ‘she’, ’it’, ’they’ and ‘them’ while referring to entities that have already occurred in the past questions or answers. Format each QA pair in a single line as a JSON dictionary (key “user” for question, and “assistant” for answer, wrapped with { and }). Do not include any other explanation.
- Example 3:

#### B.3. Dialogue filtering

The dialogues generated by GPT-4 as discussed in B.2 do not always follow the prompts, resulting in answers that have phrases such as “does not specify”, “cannot be determined”, “without additional context” and so on. Hence, following Gardner et al. (2023), we filter such QA pairs to improve the data quality and ensure desirable outputs from the model.

Apart from the manual filtering step, we also filter samples based on the similarity of the answer generated by GPT-4 and the audio samples. Specifically, we compute the cosine similarity between the LAION-CLAP text-embeddings and audio-embeddings (Wu et al., 2023) for a given QA pair in each dialogue. The distributions of similarities are shown in Figure 8. We remove samples if the similarity is below a threshold of 0.3.

[Figure 5]

[Figure 6]

Figure 8. Distributions of LAION-CLAP similarities before filtering the generated AF-Dialogue-AudioSetSL (left) and AF-Dialogue-MusicCaps (right) datasets.

- B.4. Examples of generated dialogues We demonstrate an example of generated dialogue for each of the generated dataset.

[Audio annotations: Sound of Mechanisms (Produces sound through mechanical means.): [0.119s-10.000s]; Sound of Generic impact sounds (Short, sharp, and loud bursts.): [0.217s-0.387s], [2.394s-2.727s], [2.889s-3.255s], [3.799s-4.132s], [4.473s-4.766s],

- [4.928s-5.188s], [8.129s-10.000s]; Sound of Glass shatter (High-pitched, sharp, sudden, loud crash.): [1.622s-4.652s]; Sound of Walk, footsteps (Soft, rhythmic, echoing thuds.):
- [5.724s-6.106s], [6.455s-6.650s], [7.024s-7.479s], [7.625s-7.901s]]

User: Can you describe the main events in the audio? Assistant: The audio mainly consists of mechanical sounds, sharp impact sounds, the shattering of glass, and the sound of footsteps. User: What can you tell me about the impact sounds? Assistant: The impact sounds are short, sharp, and loud bursts. They occur at several points throughout the audio, suggesting some kind of repetitive action or event. User: What about the glass shattering, does it happen more than once? Assistant: The glass shattering sound is a high-pitched, sharp, sudden, loud crash. It occurs once in the audio, lasting for about three seconds.

[Audio annotations: This is the recording of a cowbell. The cowbell is being played with the side and the tip of the drumstick repeatedly. This recording could be sampled for use in beat-making especially for genres that have a heavy usage of cowbells such as latin dance music.]

User: What instrument is being played in this audio? Assistant: The instrument being played in this audio is a cowbell. User: How is it being played? Assistant: The cowbell is being played with the side and the tip of the drumstick repeatedly. User: What is the tone of the cowbell? Assistant: The tone of the cowbell is bright, resonant, and penetrating.

- C. Additional Experiments, Examples and Discussion of Audio Flamingo We refer readers to visit our demo website: https://audioflamingo.github.io/ for comprehensive demos.

#### C.1. Additional Few-shot Experiments on Unseen Labels

Audio Flamingo can be adapted to classification with completely unseen labels with the help of ICL-based RAG. In Table 9 we demonstrate Audio Flamingo can be adapted to new classification tasks where none of the labels have been seen during training. The labels are bird species and gun names, respectively. The huge improvement from zero-shot to few-shot indicates Audio Flamingo’s strong ability to follow retrieved in-context samples.

Table 9. Accuracy of Audio Flamingo on new classification tasks with unseen labels.

Dataset Zero-shot Few-shot BG-Gun-Sound (Park et al., 2022) 1.6% 53.5%

Birds-Dataset-Till-2022 (Mohanty) 0.0% 42.2%

#### C.2. Compute

In this section we compare the computation resources required by our model and baseline models. Pengi (Deshmukh et al., 2023) is based on GPT2-base and has a total size 308M. It’s trained on 20 V100 GPUs. LTU (Gong et al., 2023c) is based on LLaMA-7B and has a total size 6.74B. It’s trained on 4 RTX A6000 GPUs for about 3 days. Qwen-Audio (Chu et al., 2023) is based on Qwen-7B and has a total size of 8.3B. The training cost is unknown. Salmonn (Tang et al., 2023a) is based on Vicuna-13B. The training cost is unknown. Our proposed Audio Flamingo has a total size of 2.2B. We use 8 A100 GPUs. It takes about 5-6 days for pretraining, about 3 days for SFT, and about 2hrs for dialogue fine-tuning.

#### C.3. Audio Flamingo understands sound in depth

Audio Flamingo understands occurrences in audio. [Subjective description of the audio: three people talking in order, with a lot of noise in the recording.] Task: Audio Question Answering Instruction: How many different voices are there? Ground Truth: Three. Baseline: One. Audio Flamingo: Three.

Audio Flamingo understands the order of sounds. [Subjective description of the audio: a man speaks on a talk show, then the audience laugh and clap their hands.] Task: Audio Captioning Instruction: Describe this sound. Ground Truth: A man is speaking followed by applause. Baseline: A crowd of people are speaking and applauding. Audio Flamingo: A man is speaking and then the audience claps.

Audio Flamingo understands quality of audio. [Subjective description of the audio: a beautiful and sad piece of instrumental music starting with keyboard playing chords slowly followed by soft violin.] Task: Audio Question Answering Instruction: What is the atmospheric quality of the audio? Ground Truth: The atmospheric quality of the audio is its ability to create a sense of space and depth, with a focus on creating a sense of mystery and wonder. Baseline: The audio has a rich and dynamic quality, which suggests a lively and engaging atmosphere, possibly associated with music or a musical performance, but it is not specified in the given audio events. Audio Flamingo: The audio has a melancholic and introspective atmosphere.

Audio Flamingo perceives loudness and its change over time. [Subjective description of the audio: a man speaks in a small volume, then there is very loud burst, and then the man laughs for very short time in a surprised way.] Task: Audio Captioning Instruction: Describe the sound in a sentence. Ground Truth: A man briefly talks followed by a loud burst then laughter. Baseline: A person is laughing and a gun is fired. Audio Flamingo: A man is speaking and then a loud bang occurs.

Audio Flamingo understands distances of sounds. [Subjective description of the audio: a man speaks in the rain, then another man laughs very slightly, followed by the sound of thunderstorm.] Task: Audio Captioning Instruction: Describe this sound. Ground Truth: A man talking followed by another man laughing as rain falls and thunder roars in the distance. Baseline: A man is talking and rain is falling. Audio Flamingo: A man is talking in the rain with thunder in the background.

#### C.4. Audio Flamingo looks at retrieved samples

Audio Flamingo borrows key words from retrieved samples. Task: Audio Captioning Instruction: Describe this sound. Ground Truth: Paper is crinkling and crumpling, and thumping occurs. Baseline: Movement of a person in a kitchen with pots and pans on a counter. Retrieved samples:

- 1) Some material is crumpled.

- 2) Plastic is being crumpled.

- 3) Crinkling of a paper with a small step. Audio Flamingo: A person crumples up a sheet of paper and crinkles it.

Task: Audio Captioning Instruction: Describe this sound. Ground Truth: a woman speaks, and faucet water runs. Baseline: A woman is talking to a child. Retrieved samples:

- 1) A woman speaking and water running.

- 2) A women voice followed by flowing water from a faucet.

- 3) Water sprays from a faucet, a person speaks. Audio Flamingo: A woman speaks while water runs from a faucet.

Audio Flamingo can ignore noisy retrieved samples and make correct prediction. Task: Music Genre Classification Instruction: What is the genre of this music? Ground Truth: pop. Retrieved samples:

- 1) reggae.
- 2) jazz.
- 3) disco. Audio Flamingo: pop.

Task: Music Instrument Classification Instruction: What is the instrument of this music? Ground Truth: flute. Retrieved samples:

- 1) clarinet.
- 2) clarinet.
- 3) clarinet. Audio Flamingo: flute.

#### C.5. Audio Flamingo has strong dialogue ability

[Audio annotations: Sound of Chirp, tweet (High-pitched, short, repetitive bird sounds.):

- [0.000s-1.157s], [1.543s-2.055s], [3.220s-4.409s], [4.748s-5.709s], [5.921s-6.780s],

- [7.094s-7.976s], [8.331s-8.819s], [9.142s-10.000s]; Sound of Animal (Loud, diverse, and often rhythmic.): [0.000s-1.157s], [1.528s-2.031s], [3.220s-4.386s], [4.764s-5.701s], [5.929s-6.740s], [7.071s-7.976s], [8.331s-8.795s], [9.134s-10.000s]; Sound of Bird vocalization, bird call, bird song (Melodic, rhythmic, varied pitch and tone.): [0.000s-1.165s],

[1.528s-2.039s], [3.213s-4.386s], [4.772s-5.693s], [5.921s-6.701s], [7.047s-7.953s],

- [8.331s-8.780s], [9.150s-10.000s]; Sound of Bird (High-pitched, melodic, rhythmic chirping sounds.): [0.000s-1.173s], [1.535s-1.992s], [3.205s-4.362s], [4.780s-5.669s], [5.906s-6.669s],

- [7.016s-7.929s], [8.323s-8.772s], [9.134s-10.000s]; Sound of Speech (¨Sound waves produced by vocal cords.¨): [2.024s-3.008s], [4.630s-5.283s], [6.835s-8.134s], [8.701s-10.000s]; Sound of Female speech, woman speaking: [2.047s-3.039s], [4.614s-5.315s], [6.843s-8.150s],
- [8.709s-10.000s]]

User: Can you briefly explain what you hear in the audio? Ground Truth: The audio primarily consists of bird sounds, including chirping and bird vocalizations. There are also sounds of other animals. Additionally, there is a woman speaking at certain intervals. Baseline: Birds are chirping. Audio Flamingo: The audio features a variety of sounds including bird vocalizations, mechanical sounds, impact sounds, a man speaking, and a ticking sound.

User: Are the bird sounds continuous? Ground Truth: The bird sounds are not continuous. They occur at different intervals throughout the audio, with some overlap with the woman’s speech. Baseline: Yes, the bird sounds are continuous. Audio Flamingo: No, the bird sounds are not continuous. They occur at different intervals throughout the audio.

User: Do the bird sounds and the person’s speech overlap? Ground Truth: Yes, there are instances where the bird sounds and the woman’s speech overlap. Baseline: Yes, the bird sounds and the woman’s speech overlap. Audio Flamingo: Yes, the bird sounds and the person’s speech do overlap at certain points in the audio.

[Audio annotations: Sound of Music (Sound produced by vibrating instruments.): [0.000s-10.000s]; Sound of Rapping (Rhythmic speech over beats.): [0.000s-10.000s]; Sound of Male singing (Deep, resonant, and powerful tones.): [0.000s-10.000s]]

User: Is the male voice rapping or singing? Ground Truth: The male voice is doing both, rapping and singing in the audio. Baseline: The male voice is rapping. Audio Flamingo: The male voice is rapping.

User: What kind of music is it? Ground Truth: The type of music isn’t specified, but it’s accompanied by rapping and male singing. Baseline: The kind of music is hip-hop. Audio Flamingo: The music is a rap song.

User: Is the male voice the only one present? Ground Truth: Yes, the male voice is the only one specified in the audio, both rapping and singing. Baseline: Yes, the male voice is the only one present. Audio Flamingo: Yes, the male voice is the only one present in the audio.

[Audio annotations: Sound of Music (Sound produced by vibrating instruments.): [0.000s-10.000s]; Sound of Male singing (Deep, resonant, and powerful tones.): [1.341s-3.734s],

- [9.289s-10.000s], ]

User: What genre does this music belong to? Ground Truth: This music belongs to the R&B/Disco genre. Baseline: It is not specified in the given audio clip. Audio Flamingo: This music belongs to the R&B genre.

User: Can you tell me more about the vocals in this track? Ground Truth: Sure, the vocals in this track are from a male singer who is singing in a higher register. His voice has been modified with effects, making it sound like a voice sample. Baseline: The vocals in this track are described as rich and full-bodied, but no further details are given about the specific singer or the lyrics being sung in the audio clip. Audio Flamingo: The vocals in this track are by a male singer. His voice is deep, resonant, and powerful, adding to the overall emotional impact of the song.

