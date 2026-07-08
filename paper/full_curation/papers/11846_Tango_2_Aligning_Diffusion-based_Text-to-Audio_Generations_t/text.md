# arXiv:2404.09956v4[cs.SD]17Jul2024

### Tango 2: Aligning Diffusion-based Text-to-Audio Generations through Direct Preference Optimization

Navonil Majumder∗

Singapore University of Technology and Design, Singapore

Chia-Yu Hung∗

Singapore University of Technology and Design, Singapore

Deepanway Ghosal∗

Singapore University of Technology and Design, Singapore

Wei-Ning Hsu

Meta AI, USA

Rada Mihalcea

University of Michigan, USA

Soujanya Poria

Singapore University of Technology and Design, Singapore

Training :

Prompt : A man speaks followed by a loud bursts and then laughter

Inference :

* CLAP score # REL (human)

AudioAlpaca

ℒDPO-Diﬀ

- 1. Pre-trained
- 2. Fine-tuned

- 1. Pre-trained
- 2. Fine-tuned

Tango

###### Tango 2

3. Aligned

+*5.5, #7% Relevant Multi-Event Audio

vs +*5.5,#9%RelevantTemporalAudio

: https://github.com/declare-lab/tango : https://huggingface.co/datasets/declare-lab/audio-alpaca : https://huggingface.co/declare-lab/tango2 : https://tango2-web.github.io/

#### ABSTRACT

models focus on training increasingly sophisticated diffusion models on a large set of datasets of prompt-audio pairs. These models do not explicitly focus on the presence of concepts or events and their temporal ordering in the output audio with respect to the input prompt. Our hypothesis is focusing on how these aspects of audio generation could improve audio generation performance in the presence of limited data. As such, in this work, using an existing text-to-audio model Tango, we synthetically create a preference dataset where each prompt has a winner audio output and some loser audio outputs for the diffusion model to learn from. The loser outputs, in theory, have some concepts from the prompt missing or in an incorrect order. We fine-tune the publicly available Tango text-to-audio model using diffusion-DPO (direct preference optimization) loss on our preference dataset and show that it leads to improved audio output over Tango and AudioLDM2, in terms of both automatic- and manual-evaluation metrics.

Generative multimodal content is increasingly prevalent in much of the content creation arena, as it has the potential to allow artists and media personnel to create pre-production mockups by quickly bringing their ideas to life. The generation of audio from text prompts is an important aspect of such processes in the music and film industry. Many of the recent diffusion-based text-to-audio

∗Authors contributed equally.

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.

ACM MM, 2024, Melbourne, Australia © 2024 Copyright held by the owner/author(s). Publication rights licensed to ACM. ACM ISBN 978-x-xxxx-xxxx-x/YY/MM https://doi.org/10.1145/nnnnnnn.nnnnnnn

#### CCS CONCEPTS

- • Computing methodologies → Natural language processing;
- • Information systems → Multimedia information systems.

#### KEYWORDS

Multimodal AI, Text-to-Audio Generation, Diffusion Models, Large Language Models, Preference Optimization

#### 1 INTRODUCTION

Generative AI is increasingly turning into a mainstay of our daily lives, be it directly through using ChatGPT [24], GPT-4 [23] in an assistive capacity, or indirectly by consuming AI-generated memes, generated using models like StableDiffusion [27], DALL-E 3 [1, 22], on social media platforms. Nonetheless, there is a massive demand for AI-generated content across industries, especially in the multimedia sector. Quick creation of audio-visual content or prototypes would require an effective text-to-audio model along with text-to-image and -video models. Thus, improving the fidelity of such models with respect to the input prompts is paramount.

Recently, supervised fine-tuning-based direct preference optimization [26] (DPO) has emerged as a cheaper and more robust alternative to reinforcement learning with human feedback (RLHF) to align LLM responses with human preferences. This idea is subsequently adapted for diffusion models by Wallace et al. [32] to align the denoised outputs to human preferences. In this work, we employ this DPO-diffusion approach to improve the semantic alignment between input prompt and output audio of a text-to-audio model. Particularly, we fine-tune the publicly available text-to-audio latent diffusion model Tango [5] on our synthesized preference dataset with DPO-diffusion loss. This preference dataset contains diverse audio descriptions (prompts) with their respective preferred (winner) and undesirable (loser) audios. The preferred audios are supposed to perfectly reflect their respective textual descriptions, whereas the undesirable audios have some flaws, such as some missing concepts from the prompt or in an incorrect temporal order or high noise level. To this end, we perturbed the descriptions to remove or change the order of certain concepts and passed them to Tango to generate undesirable audios. Another strategy that we adopted for undesirable audio generation was adversarial filtering: generate multiple audios from the original prompt and choose the audio samples with CLAP-score below a certain threshold. We call this preference dataset Audio-alpaca. To mitigate the effect of noisy preference pairs stemming from automatic generation, we further choose a subset of samples for DPO fine-tuning based on certain thresholds defined on the CLAP-score differential between preferred and undesirable audios and the CLAP-score of the undesirable audios. This likely ensures a minimal proximity to the input prompt, while guaranteeing a minimum distance between the preference pairs.

We experimentally show that fine-tuning Tango on the pruned Audio-alpaca yields Tango 2 that significantly surpasses Tango and AudioLDM2 in both objective and human evaluations. Moreover, exposure to the contrast between good and bad audio outputs during DPO fine-tuning likely allows Tango 2 to better map the semantics of the input prompt into the audio space, despite relying on the same dataset as Tango for synthetic preference data-creation.

The broad contributions of this paper are the following:

- (1) We develop a cheap and effective heuristics for semi automatically creating a preference dataset for text-to-audio generation;
- (2) On the same note, we also share the preference dataset Audio-alpaca for text-to-audio generation that may aid in the future development of such models;
- (3) Despitenotsourcingadditionalout-of-distribution text-audio pairs over Tango, our model Tango 2 outperforms both Tango and AudioLDM2 on both objective and subjective metrics;
- (4) Tango 2 demonstrates the applicability of diffusion-DPO in audio generation.

#### 2 RELATED WORK

Text-to-audiogenerationhasgarnered serious attention lately thanks to models like AudioLDM [17], Make-an-Audio [9], Tango [5], and Audiogen [14]. These models rely on diffusion architectures for audio generation from textual prompts. Recently, AudioLM [2] was proposed which utilizes the state-of-the-art semantic model w2vBert [4] to generate semantic tokens from audio prompts. These tokens condition the generation of acoustic tokens, which are decoded using the acoustic model SoundStream [35] to produce audio. The semantic tokens generated by w2v-Bert are crucial for conditioning the generation of acoustic tokens, subsequently decoded by SoundStream.

AudioLDM [17] is a text-to-audio framework that employs CLAP [33], a joint audio-text representation model, and a latent diffusion model (LDM). Specifically, an LDM is trained to generate latent representations of melspectrograms obtained using a Variational Autoencoder (VAE). During diffusion, CLAP embeddings guide the generation process. Tango [6] utilizes the pre-trained VAE from AudioLDM and replaces the CLAP model with a fine-tuned large language model: FLAN-T5. This substitution aims to achieve comparable or superior results while training with a significantly smaller dataset.

In the realm of aligning generated audio with human perception, Liao et al. [16] recently introduced BATON, a framework that initially gathers pairs of audio and textual prompts, followed by annotating them based on human preference. This dataset is subsequently employed to train a reward model. The reward generated by this model is then integrated into the standard diffusion loss to guide the network, leveraging feedback from the reward model. However, our approach significantly diverges from this work in two key aspects: 1) we automatically construct a pairwise preference dataset, referred to as Audio-alpaca, utilizing various techniques such as LLM-guided prompt perturbation and re-ranking of generated audio from Tango using CLAP scores, and 2) we then train Tango on Audio-alpaca using diffusion-DPO to generate audio samples preferred by human perception.

#### 3 BACKGROUND 3.1 Overview of Tango

Tango, proposed by Ghosal et al. [5], primarily relies on a latent diffusion model (LDM) and an instruction-tuned LLM for text-toaudio generation. It has three major components:

- (1) Textual-prompt encoder
- (2) Latent diffusion model (LDM)
- (3) Audio VAE and Vocoder

The textual-prompt encoder encodes the input description of the audio. Subsequently, the textual representation is used to construct a latent representation of the audio or audio prior from standard Gaussian noise, using reverse diffusion. Thereafter, the decoder of the mel-spectrogram VAE constructs a mel-spectrogram from the latent audio representation. This mel-spectrogram is fed to a vocoder to generate the final audio.

- 3.1.1 Textual Prompt Encoder. Tango utilizes the pre-trained LLM Flan-T5-Large (780M) [3] as the text encoder (𝐸𝑡𝑒𝑥𝑡) to acquire text encoding 𝜏 ∈ R𝐿×𝑑𝑡𝑒𝑥𝑡 , where 𝐿 and 𝑑𝑡𝑒𝑥𝑡 represent the token count and token-embedding size, respectively.
- 3.1.2 Latent Diffusion Model. For ease of understanding, we briefly introduce the LDM of Tango in this section. The latent diffusion model (LDM) [27] in Tango is derived from the work of Liu et al.[18], aiming to construct the audio prior 𝑥0 guided by text encoding 𝜏. This task essentially involves approximating the true prior 𝑞(𝑥0|𝜏)

using parameterized 𝑝𝜃 (𝑥0|𝜏).

LDM achieves this objective through forward and reverse diffusion processes. The forward diffusion represents a Markov chain of Gaussian distributions with scheduled noise parameters 0 < 𝛽1 < 𝛽2 < · · · < 𝛽𝑁 < 1, facilitating the sampling of noisier versions of 𝑥0:

𝑞(𝑥𝑛|𝑥𝑛−1) = N( 1 − 𝛽𝑛𝑥𝑛−1, 𝛽𝑛I), (1) 𝑞(𝑥𝑛|𝑥0) = N(√︁𝛼𝑛𝑥0, (1 − 𝛼𝑛)I), (2)

where 𝑁 is the number of forward diffusion steps, 𝛼𝑛 = 1 − 𝛽𝑛, and 𝛼𝑛 = 𝑛𝑖=1 𝛼𝑛. Song et al. [29] show that Eq. (2) conveniently follows from Eq. (1) through reparametrization trick that allows direct sampling of any 𝑥𝑛 from 𝑥0 via a non-Markovian process:

𝑥𝑛 = √︁𝛼𝑛𝑥0 + (1 − 𝛼𝑛)𝜖, (3)

where the noise term 𝜖 ∼ N(0, I). The final step of the forward process yields 𝑥𝑁 ∼ N(0, I).

The reverse process denoises and reconstructs 𝑥0 through textguided noise estimation (ˆ𝜖𝜃) using loss

L𝐿𝐷𝑀 =

###### ∑︁𝑁

𝛾𝑛E𝜖𝑛∼N(0,I),𝑥0||𝜖𝑛 − 𝜖ˆ𝜃(𝑛) (𝑥𝑛,𝜏)||22, (4)

𝑛=1

where 𝑥𝑛 is sampled according to Eq. (3) using standard normal noise𝜖𝑛,𝜏 represents the text encoding for guidance, and𝛾𝑛 denotes the weight of reverse step 𝑛 [8], interpreted as a measure of signalto-noise ratio (SNR) relative to 𝛼1:𝑁 . The estimated noise is then employed for the reconstruction of 𝑥0:

𝑁

𝑝𝜃 (𝑥𝑛−1|𝑥𝑛,𝜏), (5)

𝑝𝜃 (𝑥0:𝑁 |𝜏) = 𝑝(𝑥𝑁 )

𝑛=1

𝑝𝜃 (𝑥𝑛−1|𝑥𝑛,𝜏) = N(𝜇𝜃(𝑛) (𝑥𝑛,𝜏), 𝛽˜(𝑛)), (6) 𝜇𝜃(𝑛) (𝑥𝑛,𝜏) =

1 − 𝛼𝑛 √1 − 𝛼𝑛

1

𝜖ˆ𝜃(𝑛) (𝑥𝑛,𝜏)], (7) 𝛽˜(𝑛) =

√𝛼𝑛 [𝑥𝑛 −

1 − 𝛼¯𝑛−1 1 − 𝛼¯𝑛

𝛽𝑛. (8)

The parameterization of noise estimation 𝜖ˆ𝜃 involves utilizing UNet [28], incorporating a cross-attention component to integrate the textual guidance 𝜏.

3.1.3 Audio VAE and Vocoder. The audio variational auto-encoder (VAE) [11] compresses the mel-spectrogram of an audio sample, 𝑚 ∈ R𝑇×𝐹, into an audio prior𝑥0 ∈ R𝐶×𝑇/𝑟×𝐹/𝑟, where𝐶,𝑇, 𝐹, and 𝑟 denote the number of channels, time-slots, frequency-slots, and compression level, respectively. The latent diffusion model (LDM) reconstructs the audio prior 𝑥ˆ0 using input-text guidance 𝜏. Both the encoder and decoder consist of ResUNet blocks [13] and are trained by maximizing the evidence lower-bound (ELBO) [11] and minimizing adversarial loss [10]. Tango utilizes the checkpoint of the audio VAE provided by Liu et al. [18].

As a vocoder to convert the audio-VAE decoder-generated melspectrogram into audio, Tango employs HiFi-GAN [12] which is also utilized by Liu et al. [18].

Finally, Tango utilizes a data augmentation method that merges two audio signals while considering human auditory perception. This involves computing the pressure level of each audio signal and adjusting the weights of the signals to prevent the dominance of the signal with higher pressure level over the one with lower pressure level. Specifically, when fusing two audio signals, the relative pressure level is computed using the following equation:

𝐺1−𝐺2

20 )−1, (9)

𝑝 = (1 + 10

Here 𝐺1 and 𝐺2 are the pressure levels of signal 𝑥1 and 𝑥2. Then the audio signals are mixed using the equation below:

𝑝𝑥1 + (1 − 𝑝)𝑥2 √︁𝑝2 + (1 − 𝑝)2

mix(𝑥1,𝑥2) =

. (10)

The denominator is to account for the fact that the energy of a sound wave is proportional to the square of its amplitude as shown in Tokozume et al. [30]. Note that in this augmentation, textual prompts are also concatenated.

#### 3.2 Preference Optimization for Language Models

Tuning Large Language Models (LLMs) to generate responses according to human preference has been a great interest to the ML community. The most popular approach for aligning language models to human preference is reinforcement learning with human feedback (RLHF). It comprises the following steps [26]:

Supervised Fine Tuning (SFT). First, the pre-trained LLM undergoes supervised fine-tuning on high-quality downstream tasks to obtain the fine-tuned model 𝜋𝑆𝐹𝑇 .

Audio Caps

## τ xw:

###### : A person burps as people laugh

###### Strategy 1 Strategy 2 (and 3)

##### PreferenceDataCreationDPO

[Figure 1]

Tango

GPT-4

###### (Temporally) Perturbed Prompts:

A person yawns as people laugh

| | |
|---|---|
| | |
|Tango| |
| | |

A person burps and people laugh

Tango

A baby burps as people laugh

CLAP

(τ,xw,xl) AlpacaAudio-

[Figure 2]

[Figure 3]

ℒDPO-Diﬀ

- 1. Pre-trained
- 2. Fine-tuned

- 1. Pre-trained
- 2. Fine-tuned

###### Tango 2

Tango

3. Aligned

Figure 1: An illustration of our pipeline for text-to-audio alignment. The top part depicts the preference dataset creation where three strategies are deployed to generate the undesirable audio outputs to the input prompts. These samples are further filtered to form Audio-alpaca. This preference dataset is finally used to align Tango using DPO-diffusion loss (Eq. (17)), resulting in Tango 2.

Reward Modeling. Next, 𝜋𝑆𝐹𝑇 is prompted with an input 𝜏 to generate multiple responses. These responses are then shown to human labelers to rank. Once such a rank is obtained, 𝑥𝑤 ≻ 𝑥𝑙 | 𝜏 indicating 𝑥𝑤 is preferred over 𝑥𝑙, the task is to model these preferences. Among several popular choices of preference modeling, Bradley-Terry (BT) is the most popular one which relies on the equation below:

of this training can be written as follows:

L𝑅(𝑟𝜙, D) = −E(𝜏,𝑥𝑤,𝑥𝑙)∼D log𝜎(𝑟𝜙 (𝜏,𝑥𝑤) − 𝑟𝜙 (𝜏,𝑥𝑙)) (12)

This formulation considers framing the problem as a binary classification problem.

RL Optimization. The final step is to leverage 𝑟𝜙 (𝜏,𝑥) to feedback the language model. As explained by Rafailov et al. [26], this can be embedded into the following learning objective:

exp(𝑟∗(𝜏,𝑥𝑤)) exp(𝑟∗(𝜏,𝑥𝑤)) + exp(𝑟∗(𝜏,𝑥𝑙))

(11)

𝑝∗(𝑥𝑤 ≻ 𝑥𝑙 | 𝜏) =

max

E𝜏∼D,𝑥∼𝜋𝜃 (𝑥|𝜏) 𝑟𝜙 (𝜏,𝑥) − 𝛽𝐷𝐾𝐿 [𝜋𝜃 (𝑥|𝜏) ∥ 𝜋ref(𝑥|𝜏)]

𝜋𝜃

The overall idea is to learn the human preference distribution 𝑝∗. 𝑟∗(𝜏,𝑥) is a latent reward function that generates the preferences. With a static dataset created by human annotators, D =

(13)

Here, 𝜋ref represents the reference model, which in this context is the supervised fine-tuned model denoted as 𝜋SFT. 𝜋𝜃 stands for the policy language model, intended for enhancement based on feedback from 𝑟𝜙 (𝜏,𝑥). 𝛽 governs 𝜋𝜃 to prevent significant divergence

𝑁 𝑖=1

, one can train a reward model 𝑟𝜙 (𝜏,𝑥) using maximum likelihood estimation. The negative log-likelihood loss

𝜏(𝑖),𝑥𝑤(𝑖),𝑥𝑙(𝑖)

from 𝜋ref. This control is crucial as it ensures that the model stays close to the distributions upon which𝑟𝜙 (𝜏,𝑥) was trained. Since the outputs from LLM are discrete, Eq. (13) becomes non-differentiable, necessitating reinforcement learning methods like PPO to address this objective.

#### 4 METHODOLOGY

The two major parts of our approach (i) creation of preference dataset Audio-alpaca and (ii) DPO for alignment are outlined in Fig. 1.

#### 4.1 Creation of Audio-alpaca

- 4.1.1 Audio Generation from Text Prompts. Our first step is to create audio samples from various text prompts with the pre-trained Tango model. We follow three different strategies as follows:

Strategy 1: Multiple Inferences from the same Prompt. In the first setting, we start by selecting a subset of diverse captions from the training split of the AudioCaps dataset. We use the sentence embedding model gte-large1 [15] to compute dense embedding vectors of all the captions in the training set. We then perform K-Means clustering on the embedded vectors with 200 clusters. Finally, we select 70 samples from each cluster to obtain a total of 14,000 captions. We denote the selected caption set as T1.

The captions selected through the above process constitute the seed caption set. Now, we follow two settings to generate audio samples from these captions:

- (1) Strategy 1.1: Prompt Tango-full-FT with the caption to generate four different audio samples with 5, 25, 50, and 100 denoising steps. All samples are created with a guidance scale of 3.
- (2) Strategy 1.2: Prompt Tango-full-FT with the caption to generate four different audio samples each with 50 denoising steps. All samples are created with a guidance scale of 3.

In summary, we obtain (𝜏,𝑥1,𝑥2,𝑥3,𝑥4) from Strategy 1, where 𝜏 denotes the caption from T1 and 𝑥𝑖 denotes the audios generated from 𝜏.

Strategy 2: Inferences from Perturbed Prompts. We start from the selected set T1 and make perturbations of the captions using the GPT-4 language model [23]. For a caption 𝜏 from T1, we denote 𝜏1 as the perturbed caption generated from GPT-4. We add specific instructions in our input prompt to make sure that 𝜏1 is semantically or conceptually close to 𝜏. We show an illustration of the process in Table 1. In practice, we create five different perturbed 𝜏1 for each 𝜏 from GPT-4, as shown in Table 1.

We then prompt Tango-full-FT with 𝜏 and 𝜏1 to generate audio

samples 𝑥𝜏 and 𝑥𝜏1. We use 50 denoising steps with a guidance scale of 3 to generate these audio samples.

To summarize, we obtain (𝜏,𝑥𝜏,𝑥𝜏1) from Strategy 2. Note that, we considered 𝜏1 only to generate the audio sample 𝑥𝜏1. We do not further consider 𝜏1 while creating the preference dataset.

Strategy 3: Inferences from Temporally Perturbed Prompts. This strategy is aimed at prompts that describe some composition of sequence and simultaneity of events. To identify such prompts

1hf.co/thenlper/gte-large

in AudioCaps’ training dataset, as a heuristics, we look for the following keywords in a prompt: while, before, after, then, or followed. We denote the set of such prompts as T2.

For each caption𝜏2 in T2, we then prompt GPT-4 to create a set of temporal perturbations. The temporal perturbations include changing the order of the events in the original caption, or introducing a new event or removing an existing event, etc. We aim to create these temporal perturbations by providing specific instructions to GPT-4, which we also illustrate in Table 1.

We denote the temporally perturbed caption as 𝜏2. We then follow the same process as mentioned earlier in Strategy 2 to create the audio samples𝑥𝜏 and𝑥𝜏2. Finally, we pair the (𝜏,𝑥𝜏,𝑥𝜏2) samples from this strategy. Analogous to the previous strategy, the𝜏2 is only used to create the 𝑥𝜏2, and is not used anywhere else for preference data creation.

We collect the paired text prompt and audio samples from the three strategies and denote it overall as (𝜏, ⟨𝑥⟩), where ⟨𝑥⟩ indicates the set of 4 or 2 generated audio samples depending upon the corresponding strategy.

4.1.2 Ranking and Preference-Data Selection. We first create a pool of candidate preference data for the three strategies as follows:

- For Strategy 1. Let’s assume we have an instance (𝜏, ⟨𝑥⟩) from Strategy 1. We first compute the CLAP matching score following Wu et al. [33] between 𝜏 and all the four audio samples in ⟨𝑥⟩. We surmise that the sample in ⟨𝑥⟩ that has the highest matching score with 𝜏 is most aligned with 𝜏, compared to the other three audio samples that have a relatively lower matching score. We consider this audio with the highest matching score as the winning sample 𝑥𝑤 and the other three audio samples as the losing sample 𝑥𝑙. In this setting, we can thus create a pool of three preference data points: (𝜏,𝑥𝑤,𝑥𝑙), for the three losing audio samples 𝑥𝑙.
- For Strategy 2 and 3. Let’s assume we have an instance (𝜏, ⟨𝑥⟩) from Strategy 2 or 3. We compute the CLAP matching score between i) 𝜏 with 𝑥𝜏, and ii) 𝜏 with the 𝑥𝜏1 or 𝑥𝜏2, corresponding to the strategy. We consider only those instances where the CLAP score of i) is higher than the CLAP score of ii). For these instances, we use 𝑥𝜏 as the winning audio 𝑥𝑤 and 𝑥𝜏1 or 𝑥𝜏2 as the losing audio 𝑥𝑙 to create the preference data point: (𝜏,𝑥𝑤,𝑥𝑙).

Final Selection. We want to ensure that the winning audio sample 𝑥𝑤 is strongly aligned with the text prompt 𝜏. At the same time, the winning audio sample should have a considerably higher alignment with the text prompt than the losing audio sample. We use the CLAP score as a measurement to fulfill these conditions. The CLAP score is measured using cosine similarity between the text and audio embeddings, where higher scores indicate higher alignment between the text and the audio. We thus use the following conditions to select a subset of instances from the pool of preference data:

- (1) The winning audio must have a minimum CLAP score of 𝛼 with the text prompt to ensure that the winning audio is strongly aligned with the text prompt.
- (2) The losing audio must have a minimum CLAP score of 𝛽 with the text prompt to ensure that we have semantically close negatives that are useful for preference modeling.

###### Table 1: Prompts used in GPT-4 for creating the perturbations and the corresponding output.

###### Strategy and Original Caption Prompt to GPT-4 Output from GPT-4

- Strategy 2: Perturbed Prompts I have an audio clip for which the original caption is as follows: People cheering and race cars racing by. Can you generate five candidate captions that would satisfy the following requirements:

1. Crowd applauding and bicycles speeding past.

People cheering and race cars racing by. i) Would be closely related to the audio clip and the original caption. 2. Spectators clapping and trains rushing by.

- ii) Would contain inaccuracies in terms of describing the audio clip i.e. they would be somewhat wrong captions for the audio clip.

3. Audience cheering and horses galloping past.

- iii) However, the new captions should not be completely unrelated. Always keep some concepts from the original caption in the new one.

4. Fans shouting and airplanes flying by.

- iv) Would be of similar length to the original caption. 5. Group celebrating and motorcycles revving past.

Generate only the captions in separate lines so that I can programmatically extract them later.

- Strategy 3: Temporally-Perturbed Prompts I have an audio clip for which the original caption is as follows: A man is speaking then a sewing machine briefly turns on and off. Can you generate five candidate captions that would satisfy the following requirements:

1. A sewing machine briefly starts, then a man begins speaking.

A man is speaking then a sewing machine briefly turns on and off.

- i) Would be closely related to the audio clip and the original caption. 2. The sound of a sewing machine is heard after a man’s voice.

- ii) Would have a change of order of the events described in the original caption.

- 3. A man’s voice is followed by the noise of a blender.

iii) Would contain inaccuracies in terms of describing the audio clip i.e. they would be somewhat wrong captions for the audio clip.

- 4. A woman speaks and then a sewing machine is turned on.

- iv) However, the new captions should not be completely unrelated. Always keep some concepts from the original caption in the new one.

5. The noise of a sewing machine is interrupted by a man talking.

- v) Would be of similar length to the original caption. Generate only the captions in separate lines so that I can programmatically extract them later.

- (3) The winning audio must have a higher CLAP score than the losing audio w.r.t to the text prompt.
- (4) We denote Δ to be the difference in CLAP score between the text prompt with the winning audio2 and the text prompt with the losing audio. The Δ should lie between certain thresholds, where the lower bound will ensure that the losing audio is not too close to the winning audio, and the upper bound will ensure that the losing audio is not too undesirable.

We use an ensemble filtering strategy based on two different CLAP models: 630k-audioset-best and 630k-best [33]. This can reduce the effect of noise from individual CLAP checkpoints and increase the robustness of the selection process. In this strategy, samples are included in our preference dataset if and only if they satisfy all the above conditions based on CLAP scores from both of the models. We denote the conditional scores mentioned above as 𝛼1, 𝛽1, Δ1, and 𝛼2, 𝛽2, Δ2 for the two CLAP models, respectively. Based on our analysis of the distribution of the CLAP scores as shown in Figure 2, we choose their values as follows:𝛼1 = 0.45,𝛼2 = 0.60, 𝛽1 = 0.40, 𝛽2 = 0.0, 0.05 ≤ Δ1 ≤ 0.35, and 0.08 ≤ Δ2 ≤ 0.70.

Finally, our preference dataset Audio-alpaca has a total of ≈ 15k samples after this selection process. We report the distribution of Audio-alpaca in Table 2.

#### 4.2 DPO for Preference Modeling

As opposed to RLHF, recently DPO has emerged as a more robust and often more practical and straightforward alternative for LLM alignment that is based on the very same BT preference model (Eq. (11)). In contrast with supervised fine-tuning (SFT) that only optimizes for the desirable (winner) outputs, the DPO objective also allows the model to learn from undesirable (loser) outputs, which is key in the absence of a high-quality reward model, as required for RLHF. To this end, the DPO objective is derived by substituting

- 2In our paper, we employ the terms "winner" and "preferred" interchangeably. Likewise, we use "loser" and "undesirable" interchangeably throughout the text.

0.06

0.05

0.04

Probability

0.03

0.02

0.01

0.00

0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8

1

0.10

0.08

Probability

0.06

0.04

0.02

0.00

0.0 0.2 0.4 0.6 0.8

1

Figure 2: The distribution of 𝛼1 and Δ1 scores in the unfiltered dataset. We see that for an unfiltered dataset: i) the winner audio sample is not always strongly aligned to the text prompt in the 𝛼1 plot; ii) winner and loser audio samples can be too close in the Δ1 plot. We thus choose the values of our 𝛼1, Δ1 and other selection parameters to ensure the filtered dataset is less noisy with more separation between the winner and loser audios.

###### Table 2: Statistics of Audio-alpaca.

###### Strategy # Samples Avg. Winner Score Avg. Loser Score Avg. Delta

Inference w/ Different Denoising Steps (Strategy 1.1) 3004 0.645 0.447 0.198 Inference w/ Same Denoising Steps (Strategy 1.2) 2725 0.647 0.494 0.153 GPT-4 Perturbed Prompts (Strategy 2) 4544 0.641 0.425 0.216 GPT-4 Temporally Perturbed Prompts (Strategy 3) 4752 0.649 0.458 0.191

Overall 15025 0.645 0.452 0.193

the globally optimal reward—obtained by solving Eq. (13)—in the negative log-likelihood (NLL) loss (Eq. (12)).

This success spurred on Wallace et al. [32] to bring the same benefits of DPO to diffusion networks. However, unlike DPO, the goal for diffusion networks is to maximize the following learning objective (Eq. (14)) with a reward (Eq. (15)) defined on the entire diffusion path 𝑥0:𝑁 :

max

E𝜏∼D,𝑥0:𝑁∼𝜋𝜃 (𝑥0:𝑁 |𝜏) [𝑟(𝜏,𝑥0)] −𝛽𝐷KL[𝜋𝜃 (𝑥0:𝑁 |𝜏)||𝜋ref(𝑥0:𝑁 |𝜏)]. (14)

𝜋𝜃

𝑟(𝜏,𝑥0) := E𝜋𝜃 (𝑥1:𝑁 |𝑥0,𝜏) [𝑅(𝜏,𝑥0:𝑁 )], (15) Solving this objective and substituting the optimal reward in the NLL loss (Eq. (12)) yields the following DPO objective for diffusion:

0 ,𝑥0𝑙)∼Dpref log𝜎(

LDPO-Diff = −E(𝜏,𝑥𝑤

𝜋𝜃 (𝑥0:𝑤𝑁 |𝜏) 𝜋ref(𝑥0:𝑤𝑁 |𝜏)

𝜋𝜃 (𝑥0:𝑙 𝑁 |𝜏) 𝜋ref(𝑥0:𝑙 𝑁 |𝜏)

]). (16)

− log

𝛽E𝑥1:∗𝑁∼𝜋𝜃 (𝑥1:∗𝑁 |𝑥0∗,𝜏) [log

Now, applying Jensen’s inequality by taking advantage of the convexity of −log𝜎 allows the inner expectation to be pushed outside. Subsequently, approximating the denoising process with the forward process yields the following final form in terms of the L2 noise-estimation losses from LDM (Eq. (4)):

LDPO-Diff := − E𝑛,𝜖𝑤,𝜖𝑙 log𝜎(−𝛽𝑁𝜔(𝜆𝑛)(||𝜖𝑛𝑤 − 𝜖ˆ𝜃(𝑛) (𝑥𝑛𝑤,𝜏)||22 − ||𝜖𝑛𝑤 − 𝜖ˆref(𝑛) (𝑥𝑛𝑤,𝜏)||22 − (||𝜖𝑛𝑙 − 𝜖ˆ𝜃(𝑛) (𝑥𝑛𝑙 ,𝜏)||22 − ||𝜖𝑛𝑙 − 𝜖ˆref(𝑛) (𝑥𝑛𝑙 ,𝜏)||22)),

(17) where Dpref := {(𝜏,𝑥0𝑤,𝑥0𝑙 )} is ourpreferencedatasetAudio-alpaca,

𝜏, 𝑥0𝑤, and 𝑥0𝑙 being the input prompt, preferred, and undesirable output, respectively. Furthermore,𝑛 ∼ U(0, 𝑁) is the diffusion step,

𝜖𝑛∗ ∼ N(0, I) and 𝑥𝑛∗ are noise and noisy posteriors, respectively, at some step 𝑛. 𝜆𝑛 is the signal-to-noise ratio (SNR) and 𝜔(𝜆𝑛) is a weighting function defined on SNR. We use Tango-full-FT as our reference model through its noise estimation 𝜖ˆref.

- 5 EXPERIMENTS

- 5.1 Datasets and Training Details

We fine-tuned our model starting from theTango-full-FTcheckpoint on our preference dataset Audio-alpaca.

As mentioned earlier in Section 4.1.2, we have a total of 15,025 preference pairs in Audio-alpaca, which we use for fine-tuning. We use AdamW [20] with a learning rate of 9.6e-7 and a linear learning-rate scheduler for fine-tuning. Following Wallace et al.

[32], we set the 𝛽 in DPO loss (Eq. (17)) to 2000. We performed 1 epoch of supervised fine-tuning on the prompt and the preferred audio as training samples, followed by 4 epochs of DPO. The entirety of the fine-tuning was executed on two A100 GPUs which takes about 3.5 hours in total. We use a per GPU batch size of 4 and a gradient accumulation step of 4, resulting in an effective batch size of 32.

#### 5.2 Baselines

We primarily compare Tango 2 to three strong baselines:

- (1) AudioLDM [17]: A text-to-audio model that uses CLAP [33], a joint audio-text representation model, and a latent diffusion model (LDM). Specifically, the LDM is trained to generate the latent representations of melspectrograms obtained from a pre-trained Variational Autoencoder (VAE). During diffusion, CLAP text-embeddings guide the generation process.
- (2) AudioLDM2 [19]: An any-to-audio framework which uses language of audio (LOA) as a joint encoding of audio, text, image, video, and other modalities. Audio modality is encoded into LOA using a self-supervised masked auto-encoder. The remaining modalities, including audio again, are mapped to LOA through a composition of GPT-2 [25] and ImageBind [7]. This joint encoding is used as a conditioning in the diffusion network for audio generation.
- (3) Tango [5]: Utilizes the pre-trained VAE from AudioLDM but replaces the CLAP text-encoder with an instruction-tuned large language model: FLAN-T5. As compared to AudioLDM, its data-augmentation strategy is also cognizant of the audio pressure levels of the source audios. These innovations attain comparable or superior results while training on a significantly smaller dataset.

Baton [16] represents another recent approach in human preference based text-to-audio modeling. It trains a reward model to maximize rewards through supervised fine-tuning, aiming to maximize the probability of generating audio from a textual prompt. As discussed in Section 2, Baton’s reward model is not trained using the pairwise preference objective presented in Equation (12). In this approach, each text (𝜏) and audio (𝑥) pair is classified as 1 or 0, indicating whether human annotators favored the text-audio pair or not. Subsequently, this reward is incorporated into the generative objective function of the diffusion. This methodology stands in contrast to the prevailing approach in LLM alignment research. As of now, neither the dataset nor the code has been made available for comparison.

###### Table 3: Text-to-audio generation results on AudioCaps evaluation set. Due to time and budget constraints, we could only subjectively evaluate AudioLDM 2-Full-Large and Tango-full-FT. Notably these two models are considered open-sourced SOTA models for text-to-audio generation as reported in [31].

Objective – Holistic Objective – Temporal Subjective

Model #Parameters

FAD ↓ KL ↓ IS ↑ CLAP ↑ OER ↓ DUR ↓ FREQ ↓ TIME ↑ OVL ↑ REL ↑

AudioLDM-M-Full-FT 416M 2.57 1.26 8.34 0.43 − − − − − − AudioLDM-L-Full 739M 4.18 1.76 7.76 0.43 − − − − − −

AudioLDM 2-Full 346M 2.18 1.62 6.92 0.43 − − − − − − AudioLDM 2-Full-Large 712M 2.11 1.54 8.29 0.44 − − − − 3.56 3.19 Tango-full-FT 866M 2.51 1.15 7.87 0.54 0.882 3.535 1.611 0.577 3.81 3.77 Tango 2 866M 2.69 1.12 9.09 0.57 0.87 3.586 1.548 0.61 3.99 4.07

w/o Strategy 2 & 3 866M 2.64 1.13 8.06 0.54 − − − − − −

- w/o Strategy 1 866𝑀 2.47 1.13 8.58 0.56 − − − − − −

- w/o Strategy 2 866M 2.28 1.12 8.38 0.55 − − − − − −

- w/o Strategy 3 866M 2.46 1.13 8.63 0.56 0.88 3.63 1.577 0.588 − −

#### 5.3 Evaluation Metrics

Holistic Objective Metrics. We evaluate the generated audio samples in a holistic fashion using the standard Frechet Audio Distance (FAD), KL divergence, Inception score (IS), and CLAP score [17]. FAD is adapted from Frechet Inception Distance (FID) and measures the distribution-level gap between generated and reference audio samples. KL divergence is an instance-level reference-dependent metric that measures the divergence between the acoustic event posteriors of the ground truth and the generated audio sample. FAD and KL are computed using PANN, an audio-event tagger. IS evaluates the specificity and coverage of a set of samples, not needing reference audios. IS is inversely proportional to the entropy of the instance posteriors and directly proportional to the entropy of the marginal posteriors. CLAP score is defined as the cosine similarity between the CLAP encodings of an audio and its textual description. We borrowed the AudioLDM evaluation toolkit [17] for the computation of FAD, IS, and KL scores.

Temporal Objective Metrics. To specifically evaluate the temporal controllability of the text-to-audio models, we employ the recently-proposed STEAM [34] metrics measured on the AudioTime [34] benchmark dataset containing temporally-aligned audiotext pairs. STEAM constitutes four temporal metrics: (i) Ordering Error Rate (OER) – if a pair of events in the generated audio matches their order in the text, (ii) Duration (DUR) / (iii) Frequency (FREQ) – if the duration/frequency of an event in the generated audio matches matches the given text, (iv) Timestamp (TIME) – if the onand off-set timings of an event in the generated audio match the given text.

Subjective Metrics. Our subjective assessment examines two key aspects of the generated audio: overall audio quality (OVL) and relevance to the text input (REL), mirroring the approach outlined in the previous works, such as, [5, 31]. The OVL metric primarily gauges the general sound quality, clarity, and naturalness irrespective of its alignment with the input prompt. Conversely, the REL metric assesses how well the generated audio corresponds to the given text input. Annotators were tasked with rating each audio

sample on a scale from 1 (least) to 5 (highest) for both OVL and REL. This evaluation was conducted on a subset of 50 randomlyselected prompts from the AudioCaps test set, with each sample being independently reviewed by at least four annotators. Please refer to the supplementary material for more details on the evaluation instructions and evaluators.

#### 5.4 Main Results

We report the comparative evaluations of Tango 2 against the baselines Tango [5] and AudioLDM2 [19] in Table 3. For a fair comparison, we used exactly 200 inference steps in all our experiments. Tango and Tango 2 were evaluated with a classifier-free guidance scale of 3 while AudioLDM2 uses a default guidance scale of 3.5. We generate only one sample per text prompt.

Objective Evaluations. Tango 2 achieves notable improvements in objective metrics, with scores of 2.69 for FAD, 1.12 for KL, 9.09 for IS, and 0.57 for CLAP. While FAD, KL, and IS assess general naturalness, diversity, and audio quality, CLAP measures the semantic alignment between the input prompt and the generated audio. As documented in Melechovsky et al. [21], enhancing audio quality typically relies on improving the pre-training process of the backbone, either through architectural modifications or by leveraging larger or refined datasets. However, in our experiments, we observe enhanced audio quality in two out of three metrics, specifically KL and IS. Notably, Tango 2 also significantly outperforms various versions of AudioLDM and AudioLDM2 on these two metrics.

On the other hand, we note a substantial enhancement in the CLAP score. CLAP score is particularly crucial in our experimental setup as it directly measures the semantic alignment between the textual prompt and the generated audio. This outcome suggests that DPO-based fine-tuning on the preference data from Audio-alpaca yields superior audio generation to Tango and AudioLDM2.

A major enhancement in Tango 2 is evident in the temporal objective metrics, as measured by STEAM. With the exception of Duration, Tango 2 shows consistent superiority over Tango across all other temporal measurements. The implementation of Strategy

###### Table 4: Objective evaluation results for audio generation in the presence of multiple concepts or a single concept in the text prompt in the AudioCaps test set.

Multiple Events/Concepts Single Event/Concept

Model

Objective – Holistic Subjective Objective – Holistic Subjective

FAD ↓ KL ↓ IS ↑ CLAP ↑ OVL ↑ REL ↑ FAD ↓ KL ↓ IS ↑ CLAP ↑ OVL ↑ REL ↑

AudioLDM 2-Full 2.03 1.64 7.88 0.43 − − 7.93 1.24 4.50 0.47 − − AudioLDM 2-Full-Large 2.33 1.58 8.14 0.44 3.54 3.16 5.82 1.09 4.60 0.49 3.65 3.41 Tango-full-FT 2.69 1.16 7.85 0.54 3.83 3.80 7.52 1.01 4.87 0.51 3.67 3.49 Tango 2 2.60 1.11 8.98 0.57 3.99 4.07 5.48 1.00 4.95 0.52 3.95 4.10

###### Table 5: Objective evaluation results for audio generation in the presence of temporal events or non-temporal events in the text prompt in the AudioCaps test set.

Temporal Events Non Temporal Events Objective – Holistic Subjective Objective – Holistic Subjective

Model

FAD ↓ KL ↓ IS ↑ CLAP ↑ OVL ↑ REL ↑ FAD ↓ KL ↓ IS ↑ CLAP ↑ OVL ↑ REL ↑

AudioLDM 2-Full 1.95 1.71 6.37 0.41 − − 2.38 1.56 7.38 0.44 − − AudioLDM 2-Full-Large 2.39 1.65 6.10 0.42 3.35 2.82 2.68 1.46 8.12 0.46 3.79 3.62 Tango-full-FT 2.55 1.16 5.82 0.55 3.83 3.67 3.04 1.15 7.70 0.53 3.78 3.88 Tango 2 3.29 1.07 6.88 0.58 3.92 3.99 2.84 1.16 8.62 0.55 4.05 4.16

- 3 in Audio-alpaca plays a crucial role in temporal data augmentation. Our findings reveal that the absence of this augmentation leads to a decline in Tango 2’s temporal performance, thus highlighting the effectiveness of Strategy 3-based data augmentation technique.

Subjective Evaluations. In our subjective evaluation, Tango 2 achieves high ratings of 3.99 in OVL (overall quality) and 4.07 in REL (relevance), surpassing both Tango and AudioLDM2. This suggests that Tango 2 significantly benefits from preference modeling on Audio-alpaca. Interestingly, our subjective findings diverge from those reported by Melechovsky et al. [21]. In their study, the authors observed lower audio quality when Tango was fine-tuned on music data. However, in our experiments, the objective of preference modeling enhances both overall sound quality and the relevance of generated audio to the input prompts. Notably, in our experiments, AudioLDM2 performed the worst, with the scores of only 3.56 in OVL and 3.19 in REL, significantly lower than both Tango and Tango 2.

Additionally, we categorize prompts based on the presence of multiple concepts or events, exemplified by phrases like “A woman speaks while cooking”. As underlined, this prompt contains two distinct events i.e., “sound of a woman speaking” and “sound of cooking”. Through manual scrutiny, we discovered that pinpointing prompts with such multi-concepts is challenging using basic parts-of-speech or named entity-based rules. Consequently, we task GPT-4 with extracting the various concepts or events from the prompts using in-context exemplars. The specific prompt is displayed in Table 6. To evaluate GPT-4’s performance on this task, we randomly selected 30 unique prompts and manually verified their annotations from GPT-4’s. No errors attributable to GPT-4 were found. In general, Tango 2 outperforms AudioLDM2 and Tango across most objective and subjective metrics, following Table 4. We proceed to visualize the CLAP scores of the models in Figure 3. This visualization illustrates that Tango 2 consistently outperforms the

[Figure 4]

###### Figure 3: CLAP score of the models vs the number of events or concepts in the textual prompt.

baselines as the number of events or concepts per prompt increases. In particular, specifically, Tango closely matches the performance of Tango 2 only when the textual prompt contains a single concept. However, the disparity between these two models widens as the complexity of the prompt increases with multiple concepts.

The supremacy of Tango 2 over the baselines in both of these cases can be strongly ascribed to DPO training the diffusion model the differences between generating a preferred and an undesirable audio output. Particularly, the undesirable audio outputs with missing concepts and wrong temporal orderings of events are penalized. Conversely, the preferred audio samples with the correct event orderings and presence are promoted by the noise-estimation terms in the DPO-diffusion loss (Eq. (17)).

Ablations on Audio-alpaca. We conducted an ablation study on Audio-alpaca to gauge the impact of different negative data

###### Table 6: GPT-4 prompt used to extract events or concepts from audio prompts.

|You are to extract all the indivisible events in the given text, labeled as input. Imagine experiencing the events in the input as you are reading it and write down the indivisible events one by one. After writing your experience, list all the events in the sequence you observed them as a python list. Think step-by-step. Do not directly give the answer. Please refer to these following examples as refernce for input and output:<br><br>Example 1 Input: An aircraft engine runs and vibrates, metal spinning and grinding occur, and the engine accelerates and fades into the distance Output: Firstly, an aircraft engine runs and vibrates. Then, I hear metal spinning and grinding. Then, the aircraft engine accelerates. Finally, the aircraft fades into the distance. So, here is the list of events that I observed: ["aircraft engine runs", "aircraft engine vibrates", "metal spinning", "metal grinding", "aircraft engine acclerates", "aircraft fades into the distance"]<br><br>Example 2 Input: Bubbles gurgling and water spraying as a man speaks softly while crowd of people talk in the background Output: Firstly, I hear bubble gurgling. Also, I hear water spraying. Simultaneously, a man is speaking softly. Also, a crowd of people are talking in the background. So, here is the list of events that I observed: ["bubble gurgling", "water spraying", "a man is speaking softly", "crowd talking"]<br><br>Example 3 Input: A man talking then meowing and hissing Output: Firstly, I hear a man talking. Subsequently, I hear meowing. I also hear hissing. So, here is the list of events that I observed: ["a man talking", "meowing", "hissing"]<br><br><br>**** Examples end here Now, given the input text below extract all the indivisible events one by one as explained above with examples. Also, remember to follow the exact format of the examples. Input: {PROMPT} Output:<br><br>|
|---|

construction strategies. As shown in Table 3, excluding the data samples from by strategies 2 and 3 notably diminishes the performance of Tango 2. This underscores the significance of event and temporal prompt perturbations.

Audio-alpaca (Δ₂≥0.08, α₂≥0.6) Audio-alpaca (Δ₂>0, α₂≥0.6)

Audio-alpaca (Δ₂≥0.1, α₂≥0.6) Best Score

10.00

7.50

The Effect of Filtering. In our experiments, we noticed that filtering to create different Audio-alpaca can impact the performance (refer to Section 4.1). Figure 4 depicts the impact of this filtering process. We found setting Δ2 ≥ 0.08, and 𝛼2 ≥ 0.6 gives the best results.

5.00

2.50

#### 6 CONCLUSION

In this work, we propose aligning text-to-audio generative models through direct preference optimization. To the best of our knowledge, this represents the first attempt to advance text-to-audio generation through preference optimization. We achieve this by automatically generating a preference dataset using a combination of Large Language Models (LLMs) and adversarial filtering. Our preference dataset, Audio-alpaca, comprises diverse audio descriptions (prompts) paired with their respective preferred (winner) and undesirable (loser) audios. The preferred audios are expected to accurately reflect their corresponding textual descriptions, while the undesirable audios exhibit flaws such as missing concepts, incorrect temporal order, or high noise levels. To generate undesirable

0.00

CLAP FAD IS KL

Figure 4: The impact of filtering Audio-alpaca on performance observed through Δ2, and 𝛼2. The CLAP score of the winning audio must be at least 𝛼2 and Δ2 represents the difference in CLAP scores between the winning audio 𝑥𝑤 and the losing audio 𝑥𝑙 given a prompt 𝜏.

audios, we perturb the descriptions by removing or rearranging certain concepts and feeding them to Tango. Additionally, we employ

adversarial filtering, generating multiple audios from the original prompt and selecting those with CLAP scores below a specified threshold. Subsequently, we align a diffusion-based text-to-audio model, Tango, on Audio-alpaca using DPO-diffusion loss. Our results demonstrate significant performance leap over the previous models, both in terms of objective and subjective metrics. We anticipate that our dataset, Audio-alpaca, and the proposed model, Tango 2, will pave the way for further advancements in alignment techniques for text-to-audio generation.

#### ACKNOWLEDGEMENTS

This research is supported by the Ministry of Education, Singapore, under its AcRF Tier-2 grant (Project no. T2MOE2008, and Grantor reference no. MOE-T2EP20220-0017).

#### REFERENCES

- [1] James Betker, Gabriel Goh, Li Jing, † TimBrooks, Jianfeng Wang, Linjie Li, † LongOuyang, † JuntangZhuang, † JoyceLee, † YufeiGuo, † WesamManassra, † PrafullaDhariwal, † CaseyChu, † YunxinJiao, and Aditya Ramesh. [n. d.]. Improving Image Generation with Better Captions. https://api.semanticscholar.org/CorpusID: 264403242
- [2] Zalán Borsos, Raphaël Marinier, Damien Vincent, Eugene Kharitonov, Olivier Pietquin, Matt Sharifi, Dominik Roblek, Olivier Teboul, David Grangier, Marco Tagliasacchi, et al. 2023. Audiolm: a language modeling approach to audio generation. IEEE/ACM Transactions on Audio, Speech, and Language Processing

(2023).

- [3] Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Eric Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, Albert Webson, Shixiang Shane Gu, Zhuyun Dai, Mirac Suzgun, Xinyun Chen, Aakanksha Chowdhery, Sharan Narang, Gaurav Mishra, Adams Yu, Vincent Zhao, Yanping Huang, Andrew Dai, Hongkun Yu, Slav Petrov, Ed H. Chi, Jeff Dean, Jacob Devlin, Adam Roberts, Denny Zhou, Quoc V. Le, and Jason Wei. 2022. Scaling InstructionFinetuned Language Models. https://doi.org/10.48550/ARXIV.2210.11416
- [4] Yu-An Chung, Yu Zhang, Wei Han, Chung-Cheng Chiu, James Qin, Ruoming Pang, and Yonghui Wu. 2021. w2v-BERT: Combining Contrastive Learning and Masked Language Modeling for Self-Supervised Speech Pre-Training. In IEEE Automatic Speech Recognition and Understanding Workshop, ASRU 2021, Cartagena, Colombia, December 13-17, 2021. IEEE, 244–250. https://doi.org/10. 1109/ASRU51503.2021.9688253
- [5] Deepanway Ghosal, Navonil Majumder, Ambuj Mehrish, and Soujanya Poria.

2023. Text-to-audio generation using instruction-tuned llm and latent diffusion model. arXiv preprint arXiv:2304.13731 (2023).

- [6] Deepanway Ghosal, Navonil Majumder, Ambuj Mehrish, and Soujanya Poria.

2023. Text-to-Audio Generation using Instruction Tuned LLM and Latent Diffusion Model. arXiv preprint arXiv:2304.13731 (2023).

- [7] Rohit Girdhar, Alaaeldin El-Nouby, Zhuang Liu, Mannat Singh, Kalyan Vasudev Alwala, Armand Joulin, and Ishan Misra. 2023. ImageBind: One Embedding Space To Bind Them All. In CVPR.
- [8] Tiankai Hang, Shuyang Gu, Chen Li, Jianmin Bao, Dong Chen, Han Hu, Xin Geng, and Baining Guo. 2023. Efficient Diffusion Training via Min-SNR Weighting Strategy. arXiv:2303.09556 [cs.CV]
- [9] Rongjie Huang, Jiawei Huang, Dongchao Yang, Yi Ren, Luping Liu, Mingze Li, Zhenhui Ye, Jinglin Liu, Xiang Yin, and Zhou Zhao. 2023. Make-an-audio: Textto-audio generation with prompt-enhanced diffusion models. arXiv preprint arXiv:2301.12661 (2023).
- [10] Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, and Alexei A. Efros. 2016. Image-toImage Translation with Conditional Adversarial Networks. 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2016), 5967–5976.
- [11] Diederik P. Kingma and Max Welling. 2013. Auto-Encoding Variational Bayes. CoRR abs/1312.6114 (2013).
- [12] Jungil Kong, Jaehyeon Kim, and Jaekyoung Bae. 2020. Hifi-gan: Generative adversarial networks for efficient and high fidelity speech synthesis. Advances in Neural Information Processing Systems 33 (2020), 17022–17033.
- [13] Qiuqiang Kong, Yin Cao, Haohe Liu, Keunwoo Choi, and Yuxuan Wang. 2021. Decoupling Magnitude and Phase Estimation with Deep ResUNet for Music Source Separation. In International Society for Music Information Retrieval Conference.
- [14] Felix Kreuk, Gabriel Synnaeve, Adam Polyak, Uriel Singer, Alexandre Défossez, Jade Copet, Devi Parikh, Yaniv Taigman, and Yossi Adi. 2022. Audiogen: Textually guided audio generation. arXiv preprint arXiv:2209.15352 (2022).
- [15] Zehan Li, Xin Zhang, Yanzhao Zhang, Dingkun Long, Pengjun Xie, and Meishan Zhang. 2023. Towards general text embeddings with multi-stage contrastive

- learning. arXiv preprint arXiv:2308.03281 (2023).
- [16] Huan Liao, Haonan Han, Kai Yang, Tianjiao Du, Rui Yang, Zunnan Xu, Qinmei Xu, Jingquan Liu, Jiasheng Lu, and Xiu Li. 2024. BATON: Aligning Text-to-Audio Model with Human Preference Feedback. arXiv:2402.00744 [cs.SD]
- [17] Haohe Liu, Zehua Chen, Yi Yuan, Xinhao Mei, Xubo Liu, Danilo Mandic, Wenwu Wang, and Mark D Plumbley. 2023. Audioldm: Text-to-audio generation with latent diffusion models. arXiv preprint arXiv:2301.12503 (2023).
- [18] Haohe Liu, Zehua Chen, Yi Yuan, Xinhao Mei, Xubo Liu, Danilo P. Mandic, Wenwu Wang, and Mark D . Plumbley. 2023. AudioLDM: Text-to-Audio Generation with Latent Diffusion Models. ArXiv abs/2301.12503 (2023).
- [19] Haohe Liu, Qiao Tian, Yi Yuan, Xubo Liu, Xinhao Mei, Qiuqiang Kong, Yuping Wang, Wenwu Wang, Yuxuan Wang, and Mark D Plumbley. 2023. AudioLDM 2: Learning holistic audio generation with self-supervised pretraining. arXiv preprint arXiv:2308.05734 (2023).
- [20] Ilya Loshchilov and Frank Hutter. 2017. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101 (2017).
- [21] Jan Melechovsky, Zixun Guo, Deepanway Ghosal, Navonil Majumder, Dorien Herremans, and Soujanya Poria. 2024. Mustango: Toward Controllable Text-toMusic Generation. arXiv:2311.08355 [eess.AS]
- [22] OpenAI. 2023. DALL·E 2. https://openai.com/dall-e-2
- [23] OpenAI. 2023. GPT-4. https://openai.com/gpt-4
- [24] OpenAI. 2023. Introducing ChatGPT. https://openai.com/blog/chatgpt
- [25] Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. 2019. Language Models are Unsupervised Multitask Learners. (2019).
- [26] Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn. 2023. Direct Preference Optimization: Your Language Model is Secretly a Reward Model. arXiv:2305.18290 [cs.LG]
- [27] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. 2022. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 10684–10695.
- [28] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. 2015. U-Net: Convolutional Networks for Biomedical Image Segmentation. In Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015, Nassir Navab, Joachim Hornegger, William M. Wells, and Alejandro F. Frangi (Eds.). Springer International Publishing, Cham, 234–241.
- [29] Jiaming Song, Chenlin Meng, and Stefano Ermon. 2020. Denoising Diffusion Implicit Models. ArXiv abs/2010.02502 (2020).
- [30] Yuji Tokozume, Yoshitaka Ushiku, and Tatsuya Harada. 2017. Learning from Between-class Examples for Deep Sound Recognition. CoRR abs/1711.10282

(2017). arXiv:1711.10282 http://arxiv.org/abs/1711.10282

- [31] Apoorv Vyas, Bowen Shi, Matthew Le, Andros Tjandra, Yi-Chiao Wu, Baishan Guo, Jiemin Zhang, Xinyue Zhang, Robert Adkins, William Ngan, Jeff Wang, Ivan Cruz, Bapi Akula, Akinniyi Akinyemi, Brian Ellis, Rashel Moritz, Yael Yungster, Alice Rakotoarison, Liang Tan, Chris Summers, Carleigh Wood, Joshua Lane, Mary Williamson, and Wei-Ning Hsu. 2023. Audiobox: Unified Audio Generation with Natural Language Prompts. arXiv:2312.15821 [cs.SD]
- [32] Bram Wallace, Meihua Dang, Rafael Rafailov, Linqi Zhou, Aaron Lou, Senthil Purushwalkam, Stefano Ermon, Caiming Xiong, Shafiq Joty, and Nikhil Naik.

2023. Diffusion Model Alignment Using Direct Preference Optimization. arXiv:2311.12908 [cs.CV]

- [33] Yusong Wu, Ke Chen, Tianyu Zhang, Yuchen Hui, Taylor Berg-Kirkpatrick, and Shlomo Dubnov. 2023. Large-scale contrastive language-audio pretraining with feature fusion and keyword-to-caption augmentation. In ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 1–5.
- [34] Zeyu Xie, Xuenan Xu, Zhizheng Wu, and Mengyue Wu. 2024. AudioTime: A Temporally-aligned Audio-text Benchmark Dataset. arXiv:2407.02857 [cs.SD] https://arxiv.org/abs/2407.02857
- [35] Neil Zeghidour, Alejandro Luebs, Ahmed Omran, Jan Skoglund, and Marco Tagliasacchi. 2022. SoundStream: An End-to-End Neural Audio Codec. IEEE ACM Trans. Audio Speech Lang. Process. 30 (2022), 495–507. https://doi.org/10. 1109/TASLP.2021.3129994

