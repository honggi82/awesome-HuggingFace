[Figure 1]

## M-A-P

##### YuE: Scaling Open Foundation Models for Long-Form Music Generation

# arXiv:2503.08638v2[eess.AS]15Sep2025

HKUST and MAP

(alphabetical order) GitHub: https://github.com/multimodal-art-projection/YuE

Demo: https://map-yue.github.io/

### Abstract

We tackle the task of long-form music generation—particularly the challenging lyrics-to-song problem—by introducing YuE (乐), a family of open foundation models based on the LLaMA2 architecture. Specifically, YuE scales to trillions of tokens and generates up to five minutes of music while maintaining lyrical alignment, coherent musical structure, and engaging vocal melodies with appropriate accompaniment. It achieves this through: (1) track-decoupled nexttoken prediction to overcome dense mixture signals, (2) structural progressive conditioning for long-context lyrical alignment, and (3) a multitask, multiphase pre-training recipe to converge and generalize. In addition, we redesign the in-context learning technique for music generation, enabling versatile style transfer (e.g., converting Japanese city pop into an English rap while preserving the original accompaniment) and bidirectional generation. Through extensive evaluation, we demonstrate that YuE matches or even surpasses some of the proprietary systems in musicality and vocal agility. In addition, fine-tuning YuE enables additional controls and enhanced support for tail languages. Furthermore, beyond generation, we show that YuE’s learned representations can perform competatively on music understanding tasks, where the results of YuE match or exceed state-of-the-art methods on the MARBLE benchmark.

Keywords: lyrics2song, song generation, long-form, foundation model, music generation

Dual Next Token Prediction Multiple Applications

[Figure 2]

Lyrics2Song Style Transfer

[Figure 3]

[Figure 4]

[Figure 5]

Voice Cloning

Music Understanding

[Figure 6]

[Figure 7]

[Figure 8]

Singing Voice Synthesis

More to Explore

YuE Model

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

Two-Stage Causal LM

Figure 1. The General Application of YuE. The YuE model takes meta information and lyrics of the generated song in text and arbitrary audio as condition. The model can control outputs in multiple dimensions such as genre, emotion and languages.

#### Contents

- 1 Introduction 4
- 2 Related Work and Prelimenaries 5
- 3 YuE 6

- 3.1 Overview . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
- 3.2 Stage-1: Music Language Modeling . . . . . . . . . . . . . . . . . . . . . . . . . . . 6

- 3.2.1 Track-Decoupled Next-Token Prediction . . . . . . . . . . . . . . . . . . . 6
- 3.2.2 Structural Progressive Conditioning . . . . . . . . . . . . . . . . . . . . . . 8
- 3.2.3 Music In-Context Learning . . . . . . . . . . . . . . . . . . . . . . . . . . . 9

- 3.3 Stage-2: Residual Modeling . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 10
- 3.4 Tokenization and Audio Reconstruction . . . . . . . . . . . . . . . . . . . . . . . . 11

- 4 Training and Inference Strategies 12

- 4.1 Scaling Up Stage-1 Pre-Training . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 12

- 4.1.1 Multitask Learning . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 12
- 4.1.2 Multiphase Training . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13

- 4.2 Stage-2 Pre-training. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14
- 4.3 Test-time Strategies . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14

- 5 Experiments 14

- 5.1 Data & Training Setup . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14
- 5.2 Evaluation Protocol . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15

- 6 Main Results 16

- 6.1 Human Evaluation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16

- 6.1.1 Overall Comparison with Proprietary Systems. . . . . . . . . . . . . . . . 17
- 6.1.2 Detailed Comparison with Proprietary Systems. . . . . . . . . . . . . . . . 17

- 6.2 Automatic Evaluation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18

- 6.2.1 Vocal Agility . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
- 6.2.2 Duration . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
- 6.2.3 Model Based Evaluation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19
- 6.2.4 Correlation Between Automatic Metrics and Human Evaluation . . . . . . 20

- 7 Fine-tuning To More Languages 21
- 8 Analysis and Ablations 22

- 8.1 Comparison of Audio Tokenizers . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22
- 8.2 Impact of Source Separation Prior and Dual-NTP . . . . . . . . . . . . . . . . . . . 22
- 8.3 Ablation Analysis of Lyrics-following Capabilities with CoT . . . . . . . . . . . . 23
- 8.4 Effect of Scaling . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24
- 8.5 Analysis of Test-time Tricks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24

- 9 Representation Quality 25
- 10 Emergent Abilities 25
- 11 Memorization Effect 26
- 12 Unsuccessful Attempts 26
- 13 Conclusion and Future Work 27
- 14 Ethics and Responsibility 28
- 15 Contributions and Acknowledgments 28

- A Subjective Evaluation 35

- A.1 Evaluation Methods . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 35
- A.2 Evaluation Dimensions and Definitions . . . . . . . . . . . . . . . . . . . . . . . . 35
- A.3 Conditional Evaluation Dimension and Definitions . . . . . . . . . . . . . . . . . 36

- B Qwen2Audio-Instruct Tagging Prompt 37
- C Multilingual Subjective Evaluation 38
- D 15 English Prompts From GPT 39

###### 1. Introduction

Neural music generation represents a transformative intersection of technology and artistic creativity, offering profound commercial and cultural implications. By leveraging advanced algorithms, it is revolutionizing the music industry, enabling applications in entertainment, therapy, and personalized composition [Ma et al., 2024]. Given the universal presence of music in human culture [Mehr et al., 2019], these advances have the potential to democratize music creation, making it more accessible to a broader audience, while simultaneously reshaping traditional industry practices and fostering innovative approaches to musical expression.

Among various music generation tasks, lyrics-to-song audio generation, which involves creating full songs with vocals, accompaniment, from lyrics and control signals, is one of the most challenging. Despite its significance, no open-source system can achieve this at scale. While proprietary systems like Suno and Udio1 have demonstrated impressive results, the lack of opensource alternatives limits accessibility, reproducibility, and innovation. Open-source ecosystems are crucial for advancing AI-driven music generation, enabling collaborative research and laying the foundation for AI models that can understand, compose, and innovate in the arts.

Most existing academic systems for AI-driven music audio generation are constrained to short, sub-30-second clips and treat singing voice synthesis [Liu et al., 2022, Chen et al., 2020, Zhang

- et al., 2022] and instrumental generation [Copet et al., 2023b, Chen et al., 2023, Agostinelli et al., 2023] separately [Li et al., 2024]. While recent efforts have started addressing full-song lyrics-tomusic generation, they remain limited in effectiveness—producing short, low-quality output with poor musical coherence. The difficulty of this task arises from several key challenges: 1) Long-range dependencies: Music exhibits complex temporal structures spanning several minutes, making it difficult for models to maintain coherence over extended durations. 2) Signal complexity: Unlike speech or environmental sounds, music is inherently polyphonic, requiring precise coordination between multiple instrumental and vocal components. 3) Linguistic distortion: Singing alters phonemes, durations, and prosody in ways that differ significantly from spoken language, complicating the alignment between lyrics and melody. 4) Data scarcity: The lack of large-scale, high-quality paired datasets of lyrics, vocals, and accompaniment limits model training and generalization capabilities.

In this paper, we introduce YuE, the first2 family of open foundation models designed to push the boundaries of long-form lyrics-to-song generation. Built upon the LLaMA2 [Touvron et al., 2023b, Zhang et al., 2024] architecture and trained on trillions of tokens, YuE generates highquality music up to five minutes long while maintaining lyrical alignment, musical coherence, and engaging vocal melodies.

By leveraging innovative pre-training and inference techniques, YuE addresses key challenges of lyrics-to-song generation and outperforms several proprietary systems in musicality, expressiveness, and controllability. We further examine subjective correlations with various automatic metrics. Interestingly, some traditional metrics (e.g., CLAP-score [Wu et al., 2023a]) fail to align with human preferences, while metrics like CLaMP3-score [Wu et al., 2025] and vocal range correlate strongly with subjective scores, suggesting the need for new, music-specific metrics that better capture listeners’ perceptual judgments (Section 6).

We further investigate potential memorization effects by thoroughly examining whether the model reproduces training data verbatim, and demonstrate that YuE largely avoids copying despite strong in-context conditioning (Section 11).

Our main contributions include:

- 1) Track-Decoupled Next-Token Prediction: A dual-token strategy that separately models different audio tracks (vocals, accompaniment) at the frame level, resilient to challenging low vocal-to-accompaniment ratio scenarios like metal (Section 3.2.1).
- 2) Structural Progressive Conditioning: A progressive conditioning strategy for long-form music generation, enabling song-level lyrics following and structure control (Section 3.2.2).

1https://suno.com/, https://www.udio.com/ 2As of its release on Jan. 28, 2025, YuE familiy is the first publicly available, open-source lyrics-to-song model

capable of full-song generation with quality on par with commercial systems.

- 3) Redesigned In-Context Learning for Music: A novel ICL framework enabling advanced style transfer, voice cloning, and bidirectional content creation (Section 3.2.3).
- 4) Multitask Multiphase Pre-training: A training strategy that converges and generalizes on in-the-wild data (Section 4.1).
- 5) Strong Performance: YuE demonstrates strong results in musicality, vocal agility, and generation duration compared to proprietary systems, supports multilingual lyrics following (Section 7), while also excelling in music understanding tasks on representation learning benchmark MARBLE (Section 9).

###### 2. Related Work and Prelimenaries

Music Generation and Singing Voice Synthesis. Early music generation approaches primarily focused on MIDI-based methods [Huang et al., 2018, Payne, 2022], while recent models generate raw audio conditioned on tags or text [Dhariwal et al., 2020a, Agostinelli et al., 2023, Liu et al.,

- 2023, Huang et al., 2023, Copet et al., 2023a, Chen et al., 2024, Evans et al., 2024]. However, most existing audio methods are limited to instrumental music with short durations (around 30 seconds) due to computational constraints. Although some efforts incorporate vocals, they typically lack coherent lyrical semantics [Agostinelli et al., 2023, Dhariwal et al., 2020a]. Concurrently, deep learning has significantly advanced singing voice synthesis (SVS), leveraging techniques like GANs, diffusion models, and variational autoencoders for high-quality vocal synthesis [Chen et al., 2020, Liu et al., 2022, Zhang et al., 2022, Hong et al., 2023], and enabling nuanced control via language prompts or discrete tokens [Donahue et al., 2023, Wang et al.,
- 2024, Wu et al., 2024]. Nevertheless, these SVS systems mostly generate pure vocals with explicit melodic guidance. In contrast, our work proposes a novel approach capable of autonomously generating coherent and semantically meaningful vocals alongside instrumental accompaniments, supporting significantly extended song contexts of up to five minutes, thus substantially advancing automated music production.

Song Generation. Despite recent progress in music generation research, academic models still face significant limitations. Previous or concurrent work, such as Jukebox [Dhariwal et al., 2020b], MelodyLM [Li et al., 2024], SongCreator [Lei et al., 2024], SongGen [Liu et al., 2025] struggle to generate long-form music audio beyond 30 seconds while maintaining coherence and high-quality synthesis. These models often lack fully open-source implementations, making reproducibility and further improvements difficult. For instance, Jukebox utilizes a multi-scale VQ-VAE for raw audio modeling but suffers from noticeable artifacts and limited controllability. Similarly, SongCreator [Lei et al., 2025] and SongGen [Liu et al., 2025] introduce innovative transformer-based architectures for text-to-song generation, yet their performance is inferior to commercial counterparts. In contrast, industry-developed systems such as Tiangong Music (Kunlun Ltd.), Seed Music (ByteDance)[Bai et al., 2024], Suno, Udio, and Hailuo Music (MiniMax) have demonstrated promising results in song-level audio generation, though their technical details remain undisclosed. Our work addresses these gaps by offering an open-source, songlevel generative model with full technical transparency, achieving performance on par with leading proprietary systems.

Audio Tokenizers. Discrete modeling of audio often employs neural codec tokenizers, particularly Residual Vector Quantization GANs (RVQ-GANs) [Kumar et al., 2024], typically categorized into acoustic and semantic tokens [Défossez et al., 2024, Borsos et al., 2023]. Acoustic tokens, optimized for reconstruction, encode fine acoustic details, causing significant token shifts even with minor acoustic variations. Prior studies [Copet et al., 2023b] indicate these tokens require extensive training epochs; notably, we find acoustic tokens alone fail to converge efficiently on our dataset (Section 8.1). Conversely, semantic tokens, derived from self-supervised learning encoders [Schneider et al., 2019, Baevski et al., 2020, Chung et al., 2021, Baevski et al., 2022, Ma

- et al., 2023], produce semantically meaningful representations (e.g., phonemes, notes, genres) [Zhang et al., 2023, Yuan et al., 2024b, Wang et al., 2025]. Unlike previous work, we conduct extensive experiments on complex in-the-wild music datasets, perform qualitative comparisons, and report tokenizer convergence, demonstrating that fusing semantic information significantly enhance convergence.

###### 3. YuE

###### 3.1. Overview

YuE is an autoregressive (AR) language model (LM)-based framework tailored for lyrics-to-song generation. As depicted in Figure 2, YuE comprises four main components: an audio tokenizer (with a lightweight upsampler), a text tokenizer, and two language models (LMs). The audio tokenizer converts waveforms into discrete tokens using a semanticacoustic fused approach. The Stage-1 LM is track-decoupled, trained on text tokens and semantic-rich bottom-level audio tokens (codebook-0 from residual VQ-VAE), modeling lyricsto-song generation as an AR next-token prediction (NTP) task. In Stage-2, a smaller LM predicts residual tokens from codebook-0 tokens to reconstruct audio. Both LMs follow the widely-adopted LLaMA2 architecture [Touvron et al., 2023a, Team, 2024]. Finally, a lightweight vocoder upsamples Stage2’s 16 kHz audio to 44.1 kHz output.

Vocal Accompaniment

+

Detokenization & Upsampling

[Figure 13]

……

……

……

……

……

……

……

……

[Figure 14]

- Stage-1 Language Model

- Stage-2 Language Model

[Figure 15]

Rearrange

###### 3.2. Stage-1: Music Language Modeling

[Figure 16]

Music language modeling stage (MuLM), illustrated in Figure 4, enables music generation conditioned on diverse inputs (lyrics, tags, structures, reference audio). We introduce MuLM’s core techniques: 1) track-decoupled next-token prediction (Section 3.2.1), 2) structural progressive generation (Section 3.2.2), and 3) music in-context learning (Section 3.2.3).

Text Tokenizer

Audio Tokenizer

Jazz

Fly me to the moon

Figure 2. Overview of YuE framework: two-stage lyrics-to-song generation with audio/text tokenizers and two language models. Stage-1: music language modeling. Stage-2: residual modeling. Blue: vocal tokens. Orange: accompaniment tokens. Grey: residual tokens.

###### 3.2.1. Track-Decoupled Next-Token Prediction

Challenges of Standard NTP. Popular LM-based approaches for modeling long RVQ sequences typically adopt a multi-stage design [Wang et al., 2023a, Agostinelli et al., 2023, Borsos et al., 2023], where the first stage commonly uses a single codebook-0 token to represent each audio frame.3 Let x1:𝑇 = (𝑥1, 𝑥2, . . . , 𝑥𝑇) represent a sequence of audio tokens, where each 𝑥𝑡 corresponds to one frame. In a standard NTP framework, we factorize the joint probability of x1:𝑇 as:

𝑇

𝑝 𝑥𝑡 | 𝑥<𝑡; 𝜃 , (1)

𝑝(x1:𝑇) =

𝑡=1

where 𝜃 is the model parameter. During inference (generation), the model predicts the next token 𝑥ˆ𝑡 which maximizes the conditional distribution:

𝑥ˆ𝑡 = argmax

𝑝 𝑥𝑡 | 𝑥<𝑡; 𝜃 . (2)

𝑥𝑡

This approach works well for tokens x1:𝑇 representing purely vocal (text-to-speech, TTS) or instrumental (text-to-music, TTM) signals but struggles when encoding both vocals and accompaniment simultaneously due to differing dynamics, as in lyrics-to-song tasks combining TTS and TTM.

We quantify Linguistic information Loss After Tokenization (LLAT) using delta Word Error Rate (ΔWER), defined as ΔWER = WERrecon −WERori, where WERrecon and WERori are estimated by a

3We acknowledge single-stage methods such as MusicGen, which utilize delay or parallel decoding patterns to reduce sequence length. However, we observed that the parallel decoding pattern fails to converge on our dataset, while the delay pattern results in longer sequences compared to multi-stage approaches.

WER Across Music Genres

25

| || |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>Mixture WER<br><br>Vocal WER<br><br>| | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

20

WER(%)

15

10

5

0

Hip-Hop Pop Metal

Music Genre

Figure 3. ΔWER across different music genres for mixture / vocal-only tracks. ΔWER ∝ LLAT.

fine-tuned Whisper4 model on tokenizer-reconstructed5 and original mixture audio, respectively.

- Figure 3 illustrates the relationship between ΔWER and music genre (hip-hop, pop, metal) using 1k sampled tracks. An upward trend is evident, with metal exhibiting the highest LLAT followed by pop and hip-hop, indicating greater modeling difficulty in acoustically dense genres. Vocal-only tracks consistently achieve lower ΔWER compared to mixtures, indicating lower LLAT after source separation.

Track-Decoupled Next-Token Prediction (Dual-NTP). The above observation suggests that the issue arises from forcing a single token 𝑥𝑡 to represent two distinct signals: vocal and music. Accompaniment can overshadow the vocal track, degrading lyric intelligibility. To overcome these shortcomings, we propose a method that explicitly incorporates a source separation prior, splitting each time step into two tokens: one for vocal and one for accompaniment (see dotted token pairs in Figure 4).

In the proposed method, each time step 𝑡 outputs two tokens: 𝑣𝑡 (vocal token) and 𝑎𝑡 (accompaniment token). The model’s sequence of tokens thus becomes:

, 𝑎1

###### , 𝑣2

###### , 𝑎2

###### , . . . , 𝑣𝑇

𝑣1 vocal

accomp.

accomp.

vocal

vocal

###### , 𝑎𝑇

###### . (3)

accomp.

To formally define this, let v1:𝑇 = (𝑣1, 𝑣2, . . . , 𝑣𝑇) and a1:𝑇 = (𝑎1, 𝑎2, . . . , 𝑎𝑇). We factorize their joint probability as:

𝑇

𝑝 v1:𝑇, a1:𝑇 =

𝑝 𝑣𝑡, 𝑎𝑡 𝑣<𝑡, 𝑎<𝑡; 𝜃 . (4)

𝑡=1

At inference time, the next pair 𝑣 ˆ𝑡, 𝑎ˆ𝑡 is chosen to maximize this joint conditional:

𝑣 ˆ𝑡, 𝑎ˆ𝑡 = arg max (𝑣𝑡, 𝑎𝑡)

𝑝 𝑣𝑡, 𝑎𝑡 𝑣<𝑡, 𝑎<𝑡; 𝜃 . (5)

Although this probability is written in joint form, it can be decomposed as:

𝑝 𝑣𝑡, 𝑎𝑡 𝑣<𝑡, 𝑎<𝑡; 𝜃 = 𝑝 𝑣𝑡 𝑣<𝑡, 𝑎<𝑡; 𝜃 × 𝑝 𝑎𝑡 𝑣≤𝑡, 𝑎<𝑡; 𝜃 , (6) making it straightforward to implement in standard AR decoding frameworks.

Discussion. Existing work has explored modeling dual tracks using various approaches [Lei

- et al., 2024, Li et al., 2024], often requiring large modifications to the LM architecture or modeling the tracks sequentially. In contrast, our proposed method offers a more effective solution with the following advantages:

1) Scalability: By preserving the existing LM architecture, we leverage well-established pretraining infrastructures and enable straightforward scalability.

4A Whisper V3 checkpoint fine-tuned on an internal song dataset with manual transcription. 5We use X-Codec as our tokenizer. See more discussion in Section 3.4 and 8.1.

S E <EOA>

<SOA> E <EOD>

Same Timesteps Mixture Tokens

| |
|---|

Residual Tokens

Vocal Tokens Instrumental Tokens

Text Tokens

ICL Audio Tokens

E E

S S E

Stage-1 Language Model

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

Lyrics2Song E

E S

S

E

S S E S S E

Text2Speech Uncond. Mix Music Uncond. Demix Music

S E E S

S E E S

- Figure 4. The Stage-1 Framework of YuE. Dotted lines: Dual-NTP (Section 3.2.1). Text interleave: CoT (Section 3.2.2). Green tokens: ICL (Section 3.2.3). Multitask learning (Section 4.1).

- 2) Convergence: Empirically, Dual-NTP converges to lower training loss compared to standard NTP. Notably, it demonstrates robust lyric adherence even within challenging minority genres (e.g., metal music)6, illustrating its adaptability to heterogeneous data distributions.
- 3) Joint Modeling of Tracks: Our approach jointly contextualizes both tracks in a single forward pass, avoiding track synchronization issues, and allowing coherent and natural musical planning.
- 4) Granular Modeling & Processing: The explicit segregation of vocal and accompaniment tokens enables independent modeling, allowing for the capture of finer nuances, particularly in instrumentally-dense segments. This also facilitates separate post-processing and mastering for each track.

###### 3.2.2. Structural Progressive Conditioning

Challenges of Full-song Generation. While typical TTS and TTM systems operate on less than 30 seconds of context [Liu et al., 2023, 2024b, Wang et al., 2023a, Borsos et al., 2023, Copet et al., 2023b], full-song modeling requires handling minutes-long contexts. Although some proprietary systems like Suno.ai and Udio achieved this, their methodologies remain undisclosed. We find that extending the LM context to full-song modeling is non-trivial. Simply scaling up the LM context length does not yield effective song-level lyrics-following capabilities and demands substantial computational resources.

A key challenge is the long-term decay property of commonly adopted Rotary Position Embedding (RoPE) [Su et al., 2024]. In autoregressive TTS and TTM systems, text conditioning applied at the start degrades as audio tokens extend further. Empirically, this degradation begins around 3K tokens and leads to complete failure beyond 6K tokens, even with 16K-token pre-trained contexts. Our mitigation attempts, such as increasing the RoPE base (10K to 100K) or curriculum learning with gradually increasing audio lengths, have been ineffective. See ablation in Section 8.3 for more details.

Structural Progressive Conditioning (CoT).7 To address the long-term decay property issue, we propose an elegant solution that leverages the inherent structural priors of music. Songs are typically composed of distinct segments, such as intro, verse, chorus, bridge, and outro [Nieto

- 6We encourage the readers to listen to our demo page https://map-yue.github.io/.
- 7We named it “CoT” to pay tribute to the concept of Chain-of-Thought prompting [Wei et al., 2022], as we

adopt similar instructions and leverage intermediate conditioning tokens as guidance. However, our approach fundamentally differs from the original Chain-of-Thought prompting in implementation and application. We also acknowledge that there is CoT-like work proposed for audio language models [Du et al., 2024a, Ma et al., 2025, Wang et al., 2025].

et al., 2020, Bruderer et al., 2009, Lerdahl and Jackendoff, 1996]. We use all-in-one [Kim and Nam, 2023] to automatically segment songs into musical sections, with most of the sections shorter than 30 seconds. A song is on average segmented into 14 sessions. Within each structure section, text form segment labels, lyrics and audio are paired together. From a full song perspective, structured text and audio tokens are interleaved (see lyrics2song token arrangement in Figure 4). Special tokens are incorporated to indicate the start and end of the audio.

We describe a single training example as a concatenation of several components. In our setup, a song is constructed as

◦ ⃝𝑖𝑁=1𝑠𝑖 ◦ <EOD>.

Dcot = Instruct ◦ Tag ◦ Lyrics

Prompt

Specifically, ◦ denotes sequence concatenation. “Instruct” is the instruction, a task prefix as follows:

###### “Generate music from the given lyrics segment by segment.”

“Tag” denotes the musical tags, which is the style control string. An example of Tag is as follows: [Genre] jazz male deep vocal romantic big band.

“Lyrics” represents the raw lyric text provided before any segmented annotations. <EOD> is an end-of-document token.

In addition, each segment 𝑠𝑖 is structured as follow:

𝑠𝑖 = [START_OF_SEGMENT] ◦ 𝜏𝑖 ◦ ℓ𝑖 ◦ <SOA> ◦ 𝜓𝑖 ◦ <EOA> ◦ [END_OF_SEGMENT].

𝜏𝑖 ∈ {[intro], [verse], [chorus], [bridge], [outro]} is a structure label, ℓ𝑖 representing the segment’s lyric content8, and 𝜓𝑖 denoting a sequence of Dual-NTP audio tokens9. In summary, each document in CoT begins with an instruction, metadata, and raw lyrics, followed by a series of annotated segments, and ends with the <EOD> token.

###### 3.2.3. Music In-Context Learning

Deficiencies of Speech ICL. Previous work in TTS [Wang et al., 2023a, Du et al., 2024b] often defines speech ICL via a continuation-based approach. The sequence is constructed as:

◦ 𝑇input

◦ 𝐴ref

◦ 𝐴gen

𝑇ref reference text

reference audio

input text

generated audio

While this framework can be suitable for speech-based tasks, there are three major issues when directly applying it to music:

- 1) Necessity of reference text. Requiring a text transcript for the reference audio can be redundant in a musical context, and lyrics may be unavailable or challenging to obtain.
- 2) Unidirectional assumption. Continuation is unidirectional and restricts the task generalization in scenarios requiring bidirectional creativity, e.g., writing an entire piece from a short chorus snippet.
- 3) Entanglement. Continuation imposes strong constraints on the style and content of the generated audio. Given that music often features structural repetition, the model may simply replicate the reference melody or even entire segments, raising copyright concerns. Moreover, this tight coupling between reference and generated segments diminishes the effectiveness of control prompts or tags designed to steer the creative process.

8Interestingly, replacing lyrics string with \n can enable instrumental music generation. 9We prepend tokenizer type special token, e.g. <xcodec>, at the beginning of audio token sequence.

Residual Tokens

0 Codebook0 1 Codebook1 2 Codebook2 3 Codebook3 …… 7 Codebook7

…… …… ……

Stage-2 Language Model

S S 0 0 …… 0 S 0 1 2 3 …… 7 0 1 2 …… E S S 0

Figure 5. Stage-2 Framework of YuE. 𝑆: <SOA>, 𝑆=<SOA>, 𝐸=<EOA>, 𝑆𝑖=<stage_i>.

Re-designing ICL for Music. The aforementioned issues necessitate a novel approach to ICL for music. We propose a revised formulation of music ICL in two modes: single-track and dual-track. In single-track mode, the reference audio can be an accompaniment, vocal, or full mixture track. In dual-track mode, we incorporate both the separated vocal and accompaniment tracks in a token-level interleaved manner, akin to Dual-NTP.

Extending the ICL format from CoT data, we randomly sample a 20–40s segment from the reference track(s) and prepend its token sequence to the CoT data:

Dicl = 𝐴ref ◦ Dcot.

We find that this form of ICL can be effectively activated with minimal computational overhead (~2% of the total pre-training cost). However, ICL constitutes a strong conditioning signal and can be considered as “easy” data. Our preliminary experiments reveal that incorporating ICL data too early encourages shortcut learning [Geirhos et al., 2020], where the model tends to directly copy the reference audio rather than composing novel music. This strong content entanglement even disrupts lyrical control. Once shortcut learning occurs, the model’s creative capabilities cannot be easily restored. Removing ICL data and continuing training on CoT alone fails to resolve the issue—without reference audio, the model struggles to generate meaningful outputs, exhibiting poor musicality.

To address this, we introduce a delayed activation strategy. We introduce a small amount of ICL data (~10B tokens) only during the annealing phase, ensuring no ICL data is used beforehand. This strategy facilitates disentangled control between text and reference audio. For instance, using a Japanese city pop track with a female vocal as reference, the model can transform the lyrics into English while preserving the same vocalist and genre, or even generate a male English rap version of the city pop track.

###### 3.3. Stage-2: Residual Modeling

As shown in Figure 5, after Stage-1 yields coarse semantic tokens (codebook-0), Stage-2 refines the audio with additional codebooks 1,2, . . . ,7. Denote the total number of codebooks by 𝐾 = 8 (indexed from 0 to 7). Although codebook-0 is already produced by Stage-1, we train Stage-2 to predict all codebooks {0,1, . . . ,7} jointly in a single autoregressive framework. This design ensures that the model has a unified view of both the high-level structure (codebook-0) and the residual details (codebooks 1–7).

Architecture Overview. Let x1:(0𝑇) = (𝑥1(0), . . . , 𝑥𝑇(0)) be the Stage-1 codebook-0 tokens for 𝑇 frames. In Stage-2, we introduce additional codebooks, collectively denoted by

x1:(1:7𝑇 ) = 𝑥1(1), . . . , 𝑥1(7); . . . ; 𝑥𝑇(1), . . . , 𝑥𝑇(7) .

For training, we treat the output space as x1:(0:7𝑇 ), i.e., each timestep 𝑡 has a tuple

x𝑡(0:7) = 𝑥𝑡(0), 𝑥𝑡(1), . . . , 𝑥𝑡(7) .

Although codebook-0 tokens are the same as those from Stage-1, they are included in the training target so the model learns to predict them as well, thus capturing complete frame-level dependencies across all codebooks.

Aligned Autoregressive Factorization. We maintain a strictly time-aligned factorization:

𝑝 x1:(0:7𝑇 ) =

𝑇

𝑝 x𝑡(0:7) x<𝑡(0:7) . (7)

𝑡=1

This ensures that at each frame 𝑡, the model conditions on all previously generated tokens across all codebooks, while still maintaining frame alignment with codebook-0.

Cross-Conditioning. During training, we organize the sequence as:

𝑥1(0), . . . , 𝑥𝑇(0) all codebook-0 first

, 𝑥1(0), 𝑥1(1), . . . , 𝑥1(7), 𝑥2(0), 𝑥2(1), . . . , 𝑥2(7), . . . , 𝑥𝑇(0), 𝑥𝑇(1), . . . , 𝑥𝑇(7)

blocks of 0-7 per frame

.

That is, the first segment is only the codebook-0 tokens, followed by repeated 8-token blocks {0,1, . . . ,7} for each frame. We apply standard teacher forcing on this extended sequence and minimize

###### ∑︁𝑇

log 𝑝 x𝑡(0:7) x<𝑡(0:7) .

LStage2 = −

𝑡=1

By placing all codebook-0 tokens at the beginning, the model is guaranteed to “see” the entire semantic structure before it encounters any mixed (0–7) blocks. This allows the model to plan the later residuals by attending to a complete semantic outline from Stage-1.

Inference. At test time, codebook-0 tokens x1:(0𝑇) come from Stage-1 and are treated as fixed (i.e., clamped). Even though the model is trained to predict codebook-0 as part of the joint sequence,

during inference we replace any predicted codebook-0 tokens with the Stage-1 output. Consequently, the only “free” outputs in the autoregressive generation are the residual codebooks

x1:(1:7𝑇 ). This ensures the sequence alignment. Implementation. Our model is a 1B-parameter Transformer with an 8K-token context window, trained on consecutive 6-second single-track segments. It employs a shared acoustic codebook space to model various audio types, including speech, vocals, instrumentals, and mixtures.

###### 3.4. Tokenization and Audio Reconstruction

Following the design space of Borsos et al. [2023], Wang et al. [2023b], the stage-1 LM models text tokens and semantic-rich codebook-0 tokens. After investigation, we realized that the vanilla text-to-speech (TTS) / text-to-music (TTM) method performs poorly on our task, where musicality and song-level lyrics-following capability are the two key challenges.

Text Tokenizer. In this work, the vocabulary of the LMs contains two sections: text and audio. For the text part, we reuse LLaMA tokenizer with a size of 32000 unique BPE tokens. Instructions, genres, lyrics, structure annotations, and structure segment boundary signals are represented with text format and tokenized with BPE.

Table 1. Special tokens and their descriptions.

Token Description

<EOD> End of document <SOA> Start of audio <EOA> End of audio

- <stage_1> Start of Stage 1
- <stage_2> Start of Stage 2 <encodec32k> Tokenizer type (Encodec 32k) <xcodec> Tokenizer type (XCodec) <semanticodec> Tokenizer type (SemantiCodec) <hificodec> Tokenizer type (HiFiCodec)

Semantic-Acoustic Fused Codec. For the audio vocabulary, we experimented with several

open-source music and universal neural codecs. Detailed ablations are provided in Section 5. Ultimately, we adopted a semantic-acoustic fused strategy [Défossez et al., 2024, Zhang et al.,

- 2023, Liu et al., 2024a, Ye et al., 2024]. Specifically, we utilized X-Codec [Ye et al., 2024] as our off-the-shelf audio tokenizer. We employed a general-purpose version of X-Codec, trained on a mixture of 200k hours of 16 kHz audio with a ratio of music : speech : audio effects = 1 : 1 : 0.05.

The X-Codec tokenizer fuses a 100M-parameter HuBERT-based universal semantic representation into the codec latent space. It has a 50Hz frame rate, consists of 12 RVQ layers, each with a codebook size of 1024. For this study, we used only the first 8 layers, as including more layers did not yield noticeable quality improvements. Notably, codebook-0 alone captures rich semantic information such as melody and vocal content, which are critical for our task.

Vocabulary Expansion and Special Tokens. We expand the SentencePiece tokenizer vocabulary to support multiple audio tokenizers and special tokens. Specifically, we include Encodec-32khz-music [Défossez et al., 2022, Copet et al., 2023b], HiFi-Codec-universal [Yang et al., 2023a,b], X-Codec-general [Ye et al., 2024], and Semanticodec-100tps [Liu

- et al., 2024a].

For special tokens, we introduce the following: <EOD> represents the end of a document, <SOA> denotes the start of audio, and <EOA> signifies the end of audio. Additionally, stage indicators, <stage_1> and <stage_2>, mark the beginning of Stage 1 and Stage 2 tokens, respectively. Tokenizer type indicators specify the corresponding tokenizer types, which are inserted between <SOA> and the actual audio token IDs. Note that stage indicators are only used in residual modeling and positioned between <SOA> and the tokenizer type indicator.

Light-weight Upsampling Module. To achieve better perceptual audio quality, we upsample the reconstructed 16kHz audio to 44.1kHz. For this, we utilize a light-weight upsampling vocoder adapting Vocos [Siuzdak, 2023] to predict the higher-frequency components. To enhance the robustness of the upsampler, we apply codebook dropout randomly and introduce a small amount of Gaussian noise during training.

###### 4. Training and Inference Strategies

- 4.1. Scaling Up Stage-1 Pre-Training

- 4.1.1. Multitask Learning

Conditional lyrics-to-song data are inherently scarce, as most available music data exist in an unconditional format. Our preliminary experiments show that large models tend to overfit to dominant learning signal10, making it difficult for them to adhere to control signals when pre-training is predominantly driven by unconditional data.

Table 2. Decomposition of Lyrics-to-Song Generation Capabilities

Essential Capabilities

- 1) Modeling of Human Vocal
- 2) Modeling of Instrumental
- 3) Joint Modeling of Vocal and Instrumental
- 4) Aligning Cross-Modal/Same-Modal Controls (lyrics, style, structure, in-context learning)

To address this, we propose a multitask pre-training approach that facilitates knowledge transfer from auxiliary tasks to enhance lyrics-to-song generation. We decompose the essential capabilities required for this task into the four key components listed in Table 2. These components serve as guiding principles for our multitask setup, which includes:

Text-to-Speech. Establishing alignment between linguistic control and human vocalization necessitates the use of speech data paired with text transcripts. This task is essential for enabling lyric-following capabilities, as discussed in capabilities 4) and 1). Omitting this task results in ineffective lyric control when training on in-the-wild data.

TTS data primarily consists of short-form speech, typically under 20 seconds, which is significantly shorter than music tracks. To mitigate this sequence length mismatch, text-speech pairs are sequentially concatenated to form full-context sequences. Additionally, the task instruction Generate speech: is prepended to transcripts with a dropout rate of 50% to enhance robustness.

10See more discussion in Section 12.

While this task is beneficial, the proportion of TTS data used is crucial. Excessive TTS training biases the generated token space towards speech, effectively modeling rap music but degrading performance on other genres require singing11. Conversely, insufficient TTS training leads to poor adherence to lyrics. Striking an optimal balance is essential for achieving effective lyric control across diverse musical styles.

Music Generation. The majority of our dataset consists of unconditional music. We annotate all tracks using Qwen2-Audio [Chu et al., 2024] to obtain open-vocabulary tags. Tags are in the style of MTG-Jamendo [Bogdanov et al., 2019], categorized by genre, instrument, and mood. The input to Qwen2-Audio is a 30-second clip sampled from each track. Prompt is shown in appendix B.

Furthermore, 40% of the tracks are separated into vocal-instrumental dual-track format using UVR12. We employ ensemble predictions13 from three models: htdemucs_ft, Kim_Vocal_1, and UVR-MDX-NET-Inst_3.

The processed tracks are tokenized and arranged into either tag-conditioned NTP or DualNTP formats. Text instructions are prepended before the audio sequences to distinguish the two prediction objectives: Generate music based on the given tags or Generate music in dual-track format based on the given tags. The tag condition consists of a [genre] string followed by shuffled tags separated by spaces, inserted between the instruction and the audio sequence.

While this is a relatively challenging task, training on it improves musicality, facilitates the development of capabilities 1), 2), and 3), while enabling style control within capability 4).

Lyrics-to-Song. Obtaining high-quality paired lyrics-audio data is challenging, as sources from web searches and platform-provided transcripts often contain noise, irrelevant text, misaligned timestamps, and version discrepancies. To address this, we implement heuristic filtering to remove irrelevant content and exclude overly short lyrics (less than 10 sentences), retaining only approximately 10% of matched tracks. Despite filtering, some inconsistencies remain.

The CoT design addresses these issues by leveraging segment-level rather than sentence-level lyrics-audio alignment, thus reducing reliance on precise matches. Additionally, incorporating a TTS auxiliary task further enhances model robustness against imperfect alignment. Manual quality inspection on over one hundred segments confirmed an approximate 80% match rate, defined by the audible presence of the majority of text in the audio.

For ICL, we support single-track and dual-track modes. Reference token sequences (20s–40s) are randomly selected from the mixed, vocal, accompaniment tracks, or combinations thereof, and are prepended directly to corresponding CoT samples. We introduce vocal tags during ICL, prompt shown in appendix B.

###### 4.1.2. Multiphase Training

- Phase-1: Warm Up. In the first phase, we warm up the model with a linear learning rate schedule from 𝑙𝑟 = 0 to 𝑙𝑟 = 3 × 10−4 over 280B tokens. Only English and Chinese data are used, as manual verification showed that these two languages dominate the dataset and exhibit relatively high quality. To save computational costs, we use a context length of 8192 (approximately 163s for mix music data and 81s for dual-track data) and a global batch size of 768 (around 6.29M tokens). This phase rapidly establishes basic musical generation capabilities.
- Phase-2: Constant Learning Rate. In this phase, we maintain a constant learning rate of 3 × 10−4 and introduce additional in-the-wild, lower-quality datasets, including multilingual data. The total processed tokens reach 1T. When incorporating new data, we maintain a 2:1 ratio of old to new data to prevent excessive distribution shifts.
- Phase-3: Context Extension. We retain the learning rate at 3 × 10−4 and extend the context length. Since music inherently involves long sequences, we simply increase the maximum positional embedding and sequence length to 16384 without modifying the data composition.

11Overfitting TTS data turns the model into a rap machine. 12https://github.com/Anjok07/ultimatevocalremovergui 13We use the minimal signal of each track.

We remove the single-track unconditional data during this phase. This phase continues training for an additional 750B tokens, further enhancing the model’s ability to handle long-context dependencies across multiple languages.

- Phase-4: Annealing with Control Injection. This is the final phase of the Stage-1 LM training. The learning rate follows a cosine schedule, gradually annealing to 3 × 10−5. At this stage, we completely remove speech and unconditional music data while introducing stronger control signals. The control signals include reference audio (ICL), gender tags, vocal timbre tags, and BPM control. However, BPM control was later removed due to its coupling with lyrics length, which degrades lyrics following.

To improve training data quality, we constructed quality signals and selected approximately 20K hours of high-quality data. The quality signals include playback count, likes, comments, and dataset source quality ratings (based on manual inspection pass rates). We performed annealing experiments across multiple languages, including English, Chinese, Japanese, and Korean. During annealing, we applied a CoT to ICL ratio of 2:1 to prevent excessive reliance on reference songs. Remarkably, with only 40B tokens (~2% of the total compute budget), the model successfully enabled all control signals introduced in this stage.

###### 4.2. Stage-2 Pre-training.

We train a Stage-2 LM with a context length of 8192. This phase incorporates all speech, demixed music, and mixed music datasets. The compute budget is set to 2T tokens. The learning rate follows a linear warm-up and cosine annealing schedule with a maximum learning rate of 3 × 10−4. We find in preliminary experiments that scaling the Stage-2 LM from 0.5B to 1B parameters and increasing the dataset size leads to improvements in audio quality; therefore, we adopt a 1B-parameter model for this stage.

###### 4.3. Test-time Strategies

Forced Decoding. In stage-1 LM decoding, only vocabulary tokens within the audio range are permitted until the <EOA> token is predicted. Subsequently, the prompt for the next segment is forcibly provided based on user input. In stage-2 LM, the codebook-0 tokens, predicted by the previous stage, are enforced at each frame. When decoding the corresponding residual token, only the vocabulary of the respective codebook is allowed.

Sampling and Classifier-Free Guidance. The sampling parameters are set as follows: top𝑘 = 50, repetition penalty = 1.1, top-𝑝 = 0.93, temperature = 1, and maximum new tokens = 3000. Classifier-free guidance is applied with a scale of 𝑠 = 1.5 for the first segment and 𝑠 = 1.2 for subsequent segments14 to improve the good-case rate. Given the conditional logprobability ℓ𝑐(𝑘) = log 𝑝𝜃(𝑘 | 𝑥) for token 𝑘 given prompt 𝑥 and the unconditional log-probability ℓ𝑢(𝑘) = log 𝑝𝜃(𝑘 | ∅), the CFG-adjusted log-score is computed as:

ℓcfg(𝑘) = 𝑠 ℓ𝑐(𝑘) − ℓ𝑢(𝑘) + ℓ𝑢(𝑘).

Music In-Context Learning. Using a song’s chorus section for in-context learning significantly enhances musicality and stability. Moreover, we find that dual-track ICL enables better audio quality than single-track ICL mode. Consequently, dual-track ICL mode is enabled by default unless specified otherwise.

###### 5. Experiments

###### 5.1. Data & Training Setup

Data Setup. For conditional speech data (TTS), we leverage three widely used English and Chinese TTS datasets—WeNetSpeech (zh), LibriHeavy (en), and GigaSpeech (en)—comprising a total of 70k hours of data. For unconditional music data (music generation), we mine 650K hours

14A lower guidance scale in later segments promotes diversity.

of in-the-wild music recordings from the Internet. 10% of the music data has corresponding lyrics after filtering.

After tokenization, Stage-1 comprises 13B conditional speech tokens, over 200B unconditional music tokens (both mixed and demixed), and 28B CoT music tokens. During annealing, a high-quality subset of 10B CoT tokens is sampled and expanded fourfold, creating a 40B ICL dataset. This dataset includes variants such as vocal-ICL, accompaniment-ICL, mix-ICL, and dual-ICL. Prior to annealing, the data mixture is set at Conditional : Unconditional = 3 : 1 and Music : Speech = 10 : 1. During annealing, only CoT and ICL data are used, maintaining a ratio of CoT : ICL = 2 : 1.

Training Setup. Our codebase is built upon Megatron-LM [Shoeybi et al., 2019], following the LLaMA2 architecture [Touvron et al., 2023a, Zhang et al., 2024]. Most of our Stage-1 experiments use a 0.5B-scale model trained on 16 NVIDIA H800 GPUs, with a typical token budget of 100B tokens. Under this budget, models usually produce valid outputs, show preliminary lyric-following capabilities, and exhibit basic musical discernment. For scaling experiments, we increase the token budget to 500B and scale models to 0.5B, 2B, and 7B parameters, trained respectively on 32, 96, and 512 NVIDIA H800 GPUs. We further train the 7B LM with additional data, scaling up to a total of 1.75T tokens before starting an annealing phase, during which we apply a 40B-token annealing process. We maintain a global batch size of 768 when possible by adjusting micro-batch size, gradient accumulation steps, and tensor parallelism; when computational resources are constrained, we reduce the global batch size to 512 or 256. For optimization, we use the Adam optimizer with gradient clipping set to 1.0, weight decay 0.1, 𝛽1 = 0.9, 𝛽2 = 0.95, 𝜖 = 10−8, and parameter initialization with standard deviation 0.02. Detailed training procedures are described in Section 4.1.2.

###### 5.2. Evaluation Protocol

Baselines. As of the writing of this paper, apart from YuE, no academic or open-source system provides usable long-form song generation capabilities, and known prior works exhibit limited performance [Dhariwal et al., 2020b]. Therefore, we selected five popular closed-source systems for benchmarking: Suno V415, Udio16, Hailuo17, and Tiangong18. It is important to note that due to the black-box nature of these closed-source models, our evaluation conducted in January 2025 reflects the comparative performance between YuE and these systems at that specific point in time.

All systems support lyric-based inputs; however, their support for style control inputs varies significantly. Specifically, Tiangong does not support textual style prompts, so we used our own reference audio as style control. Hailuo provides 18 predefined style tags, thus we selected the tag closest to our desired style prompt and used the system’s default built-in reference audio, as uploading custom references is not supported.

Human Evaluation. We conducted a human evaluation involving 40 researchers, including 12 experts in Speech/Music AI19 and 7 trained musicians. None of the evaluators participated in model training, ensuring objectivity. Following prior studies [Donahue et al., 2023, Qu et al.,

- 2024, Yuan et al., 2024a], we adopted an A/B test format.

In the main experiments in Section 6, each model generated 42 full-length songs based on a diverse set of English prompts specifying genre, instruments, emotion, lyrics, and tempo. These prompts utilized real lyrics that were rewritten by GPT and paired with corresponding 30s chorus segments as reference audio. Similarly, for the multilingual experiment, we used 10 Chinese prompts and 10 Japanese/Korean prompts. Some multilingual prompts contained sentences with more than one language, e.g., EN-JA-KR mixes. For evaluation involving nonEnglish multilingual samples, we invited native speakers or language-major students proficient

- 15https://suno.com
- 16https://www.udio.com
- 17https://hailuoai.com/music
- 18https://www.tiangong.cn/music 19Worked on text-to-speech, text-to-music, singing voice synthesis.

[Figure 17]

80

YuE

47 47 27 64 50

70

Udio

49 53 24 70 51

60

Suno V4

76 73 76 83 71

50

40

Hailuo

29 36 30 17 31

30

Tiangong

49 50 49 29 69

20

Overall

YuE

Udio

SunoV4

Hailuo

Tiangong

YuE Wins Tie YuE Loses

| |41.9 20.9 37.2<br><br>71.4 11.9 16.7<br><br>16.3 18.6 65.1<br><br>46.5 14.0 39.5| | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

Udio

Suno

Hailuo

Tiangong

0% 25% 50% 75% 100%

- Figure 6. Human evaluation comparing YuE to 4 proprietary systems. YuE matches two of it (Tiangong, Udio) and outperforms one (Hailuo). Left: Average human preference on all aspects (warmer colors / larger numbers indicate higher preference); Right: win-tie-loss on musicality.

in the respective languages to conduct assessments.

Evaluators blindly compared pairs of music pieces produced by two different systems according to several criteria: Overall Musicality, Vocal Quality (VocalQual), Accompaniment Quality (AccompQual), Music Arrangement (MusicArr), Melodic Attractiveness (MelodicAttrac), VocalAccompaniment Compatibility (VocalAccompComp), Song Structure Clarity (SongStruct), Lyrics Following Accuracy (LyricFollow)20, Genre Controllability (GenCtrl), Instrument and Vocal Configuration Controllability (InstrCtrl), Emotional Expressiveness (EmoCtrl), and Tempo and Rhythm Control (Tempo/RhyCtrl).

For ablation studies in Section 8, unless otherwise specified, we utilize a set of 15 GPT-generated English prompts (see Appendix D). Each study undergoes small-scale A/B testing, with inference performed twice per prompt, resulting in a total of 30 samples per setting.

Automatic Evaluation. We also report automatic evaluation metrics, including Kullback–Leibler

(KL) divergence for measuring distributional differences in generated audio features using audioldm_eval21, Frechet Audio Distance (FAD) [Kilgour et al., 2019] for assessing audio quality and realism (also via audioldm_eval), Audiobox-Aesthetic [Tjandra et al., 2025] for capturing perceived musical aesthetics (Production Quality (PQ), Production Complexity (PC), Content Enjoyment (CE), and Content Usefulness (CU)) using a neural audio embedding model, CLAP score22 and CLaMP 3 score [Wu et al., 2025]23 to measure semantic alignment between text prompts and audio outputs, vocal agility quantifying song-level vocal range and flexibility (pitch estimated with RMVPE24, applying 40ms note filtering and human verification), and generation duration as a practical measure of song-level audio modeling capability.

###### 6. Main Results

- 6.1. Human Evaluation We report the generation result on English in section 6.1.

20We observe that Whisper transcription accuracy is insufficiently robust for reliable automated lyrics-following evaluation. Therefore, lyrics alignment with input prompts is manually evaluated by human raters in the main experiments.

- 21https://github.com/haoheliu/audioldm_eval
- 22https://github.com/Stability-AI/stable-audio-metrics
- 23https://github.com/sanderwood/clamp3
- 24https://github.com/yxlllc/RMVPE

[Figure 18]

YuE Udio Suno V4 Hailuo Tiangong

[Figure 19]

AccompQual

[Figure 20]

MusicArr

VocalQual

[Figure 21]

[Figure 22]

MelodicAttrac

SongStruct

[Figure 23]

[Figure 24]

VocalAccompComp

- Figure 7. Normalized human preference on different music aspects. Left: scores across 6 musical aspects; Right: performance on 5 types of control.

###### 6.1.1. Overall Comparison with Proprietary Systems.

In human evaluation Figure 6, our model, YuE, demonstrates competitive performance relative to four proprietary systems in both average human preference25 and musicality. Specifically, YuE outperforms Hailuo by a clear margin, achieves comparable results to Tiangong and Udio, but still trails behind Suno V4, which remains the state-of-the-art system. In detailed musicality comparisons, YuE shows balanced win–loss ratios against Tiangong and Udio, decisively outperforms Hailuo, but underperforms compared to Suno V4. These results indicate that while proprietary products still lead in the best quality, YuE represents a promising step toward high-quality open-source music generation.

###### 6.1.2. Detailed Comparison with Proprietary Systems.

Aspects of Musicality and Acoustic Quality. To evaluate the subjective musical qualities of YuE and comparative models, we conducted a detailed A/B test on six dimensions: vocal (acoustic) quality, accompaniment (acoustic) quality, music arrangement, melodic attractiveness, vocalbacktrack matching, and song structure. We visualize the win rate with radar plot in Figure 7(L). Suno V4 consistently outperforms all other models across these aspects, thus we normalized the win rate by Suno to improve visual clarity. Among other models, YuE excels notably in music structure and music arrangement, highlighting its capability for coherent long-form composition capability. However, YuE shows clear deficiencies in vocal and accompaniment acoustic quality, likely due to limitations of its current audio tokenization method. While YuE achieves decent musicality and convergence, the semantic-fused tokenizer requires improvements in acoustic detail via an enhanced decoder or a super-resolution backend.

Controllability. Similarly, we evaluated the controllability of YuE and comparative models through A/B testing on five dimensions: genre control, instrument/vocal control, emotion control, tempo/rhythm control, and lyrics following. Given limitations of existing classifiers and transcription systems, user preference win rate was our primary evaluation metric, with results presented in Figure 7(R). Suno v4 consistently outperforms all models across controllability metrics. Among other models, YuE performs strongest in genre adherence, instrument/vocal consistency, and emotion, highlighting its effectiveness in generating stylistically coherent music aligned with textual prompts. YuE demonstrates moderate performance in emotion and tempo control, indicating the need for improved lyric alignment and tempo tagging systems due to considerable noise observed in the pseudo label on the training corpus provided by Qwen2Audio.

25Obtained by averaging win rate over all aspects.

Overall, these results affirm YuE’s robust controllability capabilities.

- 6.2. Automatic Evaluation

- 6.2.1. Vocal Agility

HailuoMusic Tiangong Udio YuE SunoV4

0

10

20

30

40

50

RangeinSemitones

Distribution of Song-Level Vocal Ranges by System

Figure 8. Song-level vocal range on different systems. Higher values indicates better vocal agility, e.g. range=12 means the vocal only span through an octave in a given song. YuE’s vocal range is among the top close-source systems.

As shown in Figure 8, the distribution of song-level vocal ranges across different systems reveals notable variations in vocal agility. Higher values indicate greater vocal expressiveness. Among the models, YuE demonstrates one of the widest vocal ranges (medium ∼= 27 semitones), closely matching top-performing closed-source systems like Suno V4. This suggests that YuE is capable of generating diverse and dynamic vocal performances. In contrast, models like Hailuo and Tiangong show a more constrained vocal range (medium number around 20 semitones), indicating potential limitations in expressiveness. These findings highlight YuE’s strength in producing vocally rich and varied song compositions.

- 6.2.2. Duration

Distribution of Duration of Songs Generated by System

400

Duration(seconds)

300

200

100

0

HailuoMusic Tiangong Udio SunoV4 YuE

Figure 9. Duration range on different systems. YuE generates the longest audio.

The distribution of generated song durations across different systems reveals substantial variation in length constraints as demonstrated by Figure 9. YuE produces the longest audio, with a significantly wider duration range compared to all other models, demonstrating its ability

Table 3. Comparison of various music generation models across multiple metrics.

Metric Distrib. Match Content Based Alignment

KL↓ FAD↓ CE↑ CU↑ PC↑ PQ↑ CLAP↑ CLaMP 3↑

Hailuo 0.756 2.080 7.350 7.737 6.793 8.132 0.265 0.106 SunoV4 0.620 1.544 7.474 7.813 6.601 8.120 0.265 0.160 Tiangong 0.708 2.547 7.421 7.766 6.060 8.220 0.244 0.114 Udio 0.503 1.222 7.112 7.520 6.626 7.803 0.310 0.156 YuE 0.372 1.624 7.115 7.543 6.280 7.894 0.118 0.240

to generate full-length songs beyond typical AI-generated clips. SunoV4 and Tiangong also generate relatively long audio. In contrast, Hailuo Music show the most restricted durations, suggesting limitations in modeling long-term musical structure. These results highlight YuE’s advantage in handling extended temporal dependencies, making it more suitable for full-song generation.

###### 6.2.3. Model Based Evaluation

- Table 3 illustrates model based automatic evaluation results, including distribution metrics KL and FAD, aesthetics metrics proposed by meta, and audio-text alignment score such as CLAP score [Wu et al., 2023a] and CLaMP 3 score [Wu et al., 2025]. Note that not all metrics align well with human perception. We will further discuss the correlation between each metric and human evaluation in Section 6.2.4.

Distribution Matching Metrics. We report KL and FAD to evaluate how well generated audio matches the target distribution. YuE achieves the best performance on KL divergence (0.372), significantly outperforming others such as Udio (0.503) and SunoV4 (0.620). While Udio attains the lowest FAD (1.222), YuE remains competitive (1.624), demonstrating effective audio quality and distribution matching capabilities. Although distribution-based metrics can suffer from sample size biases, they remain valuable for comparative purposes, particularly when evaluating against closed-source systems where large-scale sampling is impractical. We refrain from adopting the traditional MusicCaps-based evaluation scheme since MusicCaps contains a large amount of purely instrumental content, rendering it unsuitable as a reference set for song generation tasks.

Content Based Metrics. Scores above 7 across audiobox-aesthetic dimensions indicate a strong overall performance. Specifically, YuE achieves competitive scores—PQ (7.894), PC (6.280), CE (7.115), and CU (7.543)—which closely match state-of-the-art closed-source systems such as SunoV4 (CE 7.474, CU 7.813) and Tiangong (PQ 8.220). These results suggest YuE performs comparably in terms of perceived audio aesthetics and usability.

Alignment Metrics. YuE attains the highest alignment score according to CLaMP 3 (0.240) [Wu

- et al., 2025], closely aligning with the human-evaluated “control” indicators from the previous section. However, we observe a notably lower alignment for YuE according to the CLAP score (0.118), which not only diverges from human evaluation trends but also directly contradicts the findings from CLaMP 3. These discrepancies highlight potential limitations of the CLAP score in accurately capturing human perceptions of controllability, possibly due to differences in pretraining data and modeling strategies. In contrast, CLaMP 3 appears to benefit from recent methodological improvements and broader, web-scale training resources, resulting in more reliable evaluation outcomes.26

26We use CLaMP3 as the CLaMP 3 score backend, which is a more recent model compared to CLAP, showing improved results in representation quality and music retrieval tasks due to extensive web-scale pretraining. In contrast, CLAP may suffer from limited exposure to singing/musical content during its training, which could lead to

###### 6.2.4. Correlation Between Automatic Metrics and Human Evaluation

- Table 4. Pearson correlation between subjective metrics (Musicality, Average) and automatic metrics. Vocal Range strongly impacts Musicality and Average ratings.

KL FAD CE CU PC PQ CLAP CLaMP 3 VocalRange

Musicality -0.232 -0.249 0.368 0.320 -0.268 0.112 -0.072 0.333 0.857 Average -0.199 -0.351 0.357 0.303 -0.128 0.054 0.086 0.264 0.858

Correlation with Musicality & Average Preference When considering musicality and average human preference (Table 4), the Vocal Range metric stands out, correlating most strongly (above 0.85) with both subjective ratings. This highlights the crucial role of vocal expressiveness and melodic diversity27 in listeners’ overall impressions of generated music. We find vocal range to be a practical proxy for musicality and recommend its adoption.

Table 5. Pearson correlation of alignment metrics vs. human preference on controllability. LyricFollow GenCtrl InstrCtrl EmoCtrl Tempo/RhyCtrl

CLAP↑ -0.25 0.01 -0.07 0.14 0.09 CLaMP 3↑ 0.42 0.37 0.44 0.33 0.36

Alignment Metrics. The correlation results in Table 5 demonstrate that CLaMP 3 scores consistently correlate better with human evaluations of controllability compared to CLAP scores. This is particularly evident in tasks such as LyricFollow (0.42 vs. -0.25) and InstrCtrl (0.44 vs. -0.07). Interestingly, the genre-following capability measured by the CLaMP 3 backend [Wu et al., 2025] appears to be closely related to lyric-following performance, even though lyrics are not explicitly included in the computation of the CLaMP 3 score. This indicates a correlation between genre controllability and lyric adherence in music generation models. Conversely, the weaker correlations observed with CLAP suggest limitations in its capacity to capture nuanced perceptual aspects, likely due to insufficient exposure to singing and music-specific content during pre-training.

Table 6. Pearson correlation of KL and FAD on acoustic quality preference metrics. AccompQual VocalQual

KL 0.14 0.23 FAD -0.15 -0.11

Distribution Matching Metrics. We employed the more advanced PaSST [Koutini et al., 2021] backbone instead of the conventional VGGish [Hershey et al., 2017] to evaluate distribution matching metrics. Despite its sophistication, the AudioSet pre-trained backbone may inherently suffer from out-of-distribution (OOD) issues when dealing with generative music, particularly with singing or vocal elements. Additionally, sample size bias may contribute significantly, as limited availability of extensive audio samples from closed-source generative systems hinders accurate distribution estimations.

As shown in Table 6, both KL and FAD exhibit weak correlations with accompaniment (acoustic) quality (AccompQual) and vocal (acoustic) quality (VocalQual), suggesting that distribution-level

discrepancies in evaluating certain music types.

27One possible explanation relates to AR music generation behavior. Such models often favor high-probability tokens, biasing melodies toward conservative choices like tonic, chord tones, or previously generated notes. Poor optimization (e.g., overfitting) or overly conservative sampling exacerbates this issue, reducing melodic diversity.

metrics may not fully capture subtle subjective perceptions of acoustic fidelity in our case. However, as indicated in Table 4, these same metrics correlate more strongly with musicality and overall human preference28. This implies that while distribution matching may not always reflect finer acoustic details, they sometimes reflect qualities relevant to perceived musicality and listener satisfaction.

Table 7. Pearson correlation of content-based metrics vs. related preference metrics. AccompQual VocalQual SongStruct VAComp MelAttrac MusicArr

CE 0.56 0.66 0.33 0.35 0.30 0.31 CU 0.50 0.61 0.27 0.29 0.25 0.26 PC -0.09 0.00 -0.24 -0.20 0.00 -0.16 PQ 0.27 0.36 0.05 0.06 -0.03 0.02

Content-Based Metrics. In Table 7, CE exhibits the strongest correlations, particularly with subjective acoustic quality measures such as VocalQual (0.66) and AccompQual (0.56). This indicates that CE might be especially sensitive to acoustic characteristics perceived by listeners. By contrast, correlations with musicality-related aspects—such as SongStruct (0.33), VAComp (0.35), MelAttrac (0.30), and MusicArr (0.31)—are relatively lower, suggesting a lesser sensitivity of CE to detailed musical attributes. Meanwhile, both PC and PQ show notably weaker or inconsistent correlations across these subjective metrics, implying limitations in their ability to capture musicality related perceptual elements.

###### 7. Fine-tuning To More Languages

Our results (detailed in Appendix C) demonstrate YuE’s strong adaptability and effectiveness through fine-tuning to multiple languages (Chinese, Korean, Japanese) within a 40B-token budget29. As shown in Table 8, YuE notably achieves the highest lyrics-following performance in Japanese (70%). In Chinese lyrics-following, YuE secures second-best performance (60%) behind Suno (73%), while in Korean lyrics-following, it ranks third (55%). These results highlight YuE’s robust adaptability and suggest potential for further improvement with targeted fine-tuning.

YuE also demonstrates competitive musicality, placing second in Chinese (62%) and Korean (55%), which indicates effective cross-lingual transfer of musical features. However, its gap relative to Suno in Chinese musicality highlights the need for more culturally-specific training. Overall, these findings underscore YuE’s promising multilingual capability and the importance of addressing linguistic and cultural nuances in fine-tuning approaches.

- Table 8. Human preference rate for lyrics following and musicality across languages. Bold

|boxed|
|---|

indicates the best-performing system, and

indicates the second-best. Model Chinese Korean Japanese

Lyrics Music Lyrics Music Lyrics Music YuE

|60|
|---|

|62|
|---|

|55|
|---|

70 52 Udio 36 46

55

|62|
|---|

62 31 51 Suno V4 73 88 75 50

|60|
|---|

80 Hailuo 30 15 37 60 56 31 Tiangong 51 39 20 22 32 35

28Both KL and FAD are negatively correlated, since lower values indicate better alignment. 29Fine-tuning was conducted by re-annealing from the last constant learning rate checkpoint using an enhanced

mixture of target language data.

###### 8. Analysis and Ablations

###### 8.1. Comparison of Audio Tokenizers

Table 9. Qualitative comparison of different codec types based on reconstruction quality, LM convergence, and invalid probability. Invalid probability refers to the likelihood of generating noise or silence segments during LM token synthesis.

Type Codec Reconstruction LM Converge Invalid Prob.

Acoustic Encodec32k Good No All Acoustic HiFiCodec Good No All Semantic + Acoustic Semanticodec Fair Yes High Semantic + Acoustic X-Codec Fair Yes Low

In preliminary experiments on a 130k-hour subset of diverse music data, we conducted a qualitative analysis of four popular audio tokenizers, specifically focusing on acoustic tokens and fused semantic-acoustic tokens (see Table 9). Separate semantic and acoustic tokenizers would require retraining and thus were beyond the scope of this study, reserved for future work.

Acoustic tokenizers, including Encodec32k and HiFiCodec, exhibited decent reconstruction quality. However, their learned tokens proved challenging for LMs to converge due to the complexity and variability inherent in our in-the-wild dataset. Training a 0.5B LM with acoustic tokens consistently failed to converge, resulting primarily in invalid outputs characterized by noise or silence. Although prior studies indicated Encodec32k has been successfully applied to TTM [Copet et al., 2023b], even scaling the LM to 7B and extending training up to 1 trillion tokens on our data yielded only intermittent success, with outputs still dominated by noise.

In contrast, tokenizers integrating semantic and acoustic features (Semanticodec, X-Codec) demonstrated significantly better convergence, largely due to the stable clustering provided by SSL encoders. This stability facilitated successful LM training at the 0.5B scale. However, the stable clustering slightly compromised acoustic dynamics, causing only fair reconstruction quality. We further identified a critical alignment flaw in Semanticodec related to AudioMAE’s patch-based mechanism, where misalignment of one token propagated errors throughout reconstruction. X-Codec, using Hubert-derived semantics, avoided this issue and maintained lower invalid generation probability.

###### 8.2. Impact of Source Separation Prior and Dual-NTP

We define a metric called the Vocal-to-Accompaniment Ratio (VAR), to quantify the effect of track-wise energy distribution on linguistic information loss. Let 𝑣(𝑛) denote the vocal signal and 𝑎(𝑛) denote the accompaniment signal, over 𝑛 = 1,2, . . . , 𝑁. We compute VAR (in dB) as follows:

𝑁 𝑛=1 𝑣(𝑛) 2

VAR = 10log10

. (8)

𝑁 𝑛=1 𝑎(𝑛) 2

where higher VAR values indicate greater prominence of vocals relative to accompaniment, while lower VAR suggests accompaniment dominance.

Similar to Figure 3, Figure 10 illustrates the WER-VAR relationship for mixture and vocal tracks across 1K samples, including tokenizer reconstructions. Although original vocal and mixture tracks exhibit similar absolute WER (solid blue and orange lines), mixture track reconstruction significantly increases WER (solid vs. dotted blue lines), especially as VAR declines, widening the gap (ΔWER). A 20%+ ΔWER is observed around -8.0 dB VAR. In contrast, vocal tracks maintain low WER and smaller ΔWER (the worst case is 10%- around -8.0dB VAR), indicating resilience of source separation priors to VAR degradation and reconstruction information loss.

Additionally, we perform an ablation study comparing Dual-NTP and standard NTP. Figure 11 presents training loss curves of two 0.5B LMs trained with identical data and computational budgets (20B tokens). Dual-NTP demonstrates a substantial reduction in loss (approximately 0.4 lower) compared to standard NTP, confirming its efficiency and robustness. Together, these

analyses underscore the effectiveness of incorporating source separation priors with Dual-NTP into song modeling task.

Train Loss vs Consumed Train Tokens

WER vs VAR

NTP

- 3

- 4

- 5

- 6

- 7

- 8

- 9

- 10

Mixture WER

70

Dual-NTP

Mixture Recon WER

65

Vocal WER

60

Vocal Recon WER

55

WER(%)

TrainLoss

50

WER

45

40

WER

35

30

8.0 7.5 7.0 6.5 6.0 5.5 5.0 4.5

VAR (dB)

Figure 10. Comparison of WER-VAR plot for mixture and vocal tracks, including their tokenizer reconstructions, over 1K samples.

0.0 2.5 5.0 7.5 10.0 12.5 15.0 17.5 20.0

Consumed Train Tokens (Billion)

Figure 11. Training Loss over Consumed Train Tokens for NTP and Dual-NTP.

###### 8.3. Ablation Analysis of Lyrics-following Capabilities with CoT

The analysis in Figure 12 examines an ablation setting involving a 0.5B LM, which was initially pretrained on a default mixture dataset30 comprising 500B tokens and subsequently finetuned on the corresponding lyrics data for an additional 200B tokens using the specified methods: Vanilla, Curriculum, and ABF, and our proposed CoT. Additionally, we include results from the YuE-7B checkpoint to illustrate the performance gains achievable through scaling.

Vanilla refers to text prepend conditioning, where the model is trained with prepended lyrics as input for conditioning. Curriculum involves gradually increasing the text prepend data with progressively longer durations (e.g., 30s, 60s, 90s, etc.), aiming to improve the model’s ability to follow lyrics over time. ABF [Xiong et al., 2023] refers to adjusting the rope base frequency from 10k to 100k during finetuning to explore its effect on lyrics-following performance.

Word Error Rate Over Time

WordErrorRate(%)

100

80

60

CoT (7B)

ABF

Curriculum

40

CoT (0.5B)

Vanilla

20

30 60 90 120 150

Duration (seconds)

Figure 12. WER over time. Both CoT and model scaling significantly enhance lyrics-following capability. The WER over time is estimated using a fine-tuned Whisper model, with measurements recorded every 30 seconds up to 150 seconds. Overall, the proposed CoT method achieves consistently superior performance across all evaluated time intervals (30s to 150s). Scaling the model to 7B parameters demonstrates substantial improvements, reducing the WER from approximately 70% at 0.5B parameters to around 20%31.

In contrast, Vanilla, Curriculum, and ABF methods exhibit substantially worse WER, indicating a limited capability in maintaining lyrical coherence. Through manual inspection, we identified

30A mixture of speech and music. Text transcripts are in prepend format. 31Note that 20% can be considered a relatively low number. Refer to the GT WER-to-VAR plot in Figure 10.

that the primary reason for failure in Vanilla and Curriculum was their tendency to generate instrumental preludes, causing the onset of singing to drift far from the original prepended lyrics condition, thus complicating accurate alignment.

###### 8.4. Effect of Scaling

We investigate the impact of model scaling on musicality and lyrics-following capabilities. We compared checkpoints at 0.5B, 2B, and 7B scales. While the 0.5B and 2B models were trained with a limited budget of 500B tokens (in 16K context), the 7B model underwent complete scaling with a significantly larger 1.75T token budget using the full training dataset.

Win Rate Comparison

0.9

Win Rate of Musicality

Win Rate of LyricsFollow

0.8

0.7

0.6

WinRate

0.5

0.4

As illustrated in Figure 13, human evaluation demonstrates a clear improvement trend in both musicality and lyrics-following as model scale and training budget increase. Notably, the 7B model exhibits substantial enhancements, indicating that increased parameter counts and extensive training significantly boost the model’s foundational creativity and compositional quality. These results confirm that scaling plays a crucial role in achieving higher musicality and improved lyric adherence.

0.3

0.2

0.5B 2B 7B

Model Parameters

Figure 13. Human preference overall win rates for Musicality and Lyrics-following across model scales (0.5B, 2B, and 7B) in pairwise A/B tests. Larger models consistently achieve higher preferences.

###### 8.5. Analysis of Test-time Tricks

- Figure 14 presents human preference win rates for musicality obtained through A/B testing across different inference settings using YuE-7B checkpoints. Results clearly demonstrate that ICL-based methods outperform CoT-based methods significantly: ICL achieves a win rate of 0.63 compared to only 0.21 for CoT. Incorporating CFG further enhances these methods; specifically, ICL+CFG obtains the highest win rate (0.79), substantially exceeding both ICL alone and the CoT-based configurations.

This performance advantage stems from the strong conditioning ability of ICL, which restricts the decoded token space to a musically favorable subspace guided by the provided humangenerated music prompt. CFG similarly strengthens this conditioning by amplifying the influence of the text condition on next-token logits, making generated outputs more closely aligned with the intended prompt-guided subspace and thus further improving musicality.

[Figure 25]

Figure 14. Human preference win rates for Musicality across different test-time tricks.

- Table 10. Evaluation of YuE single-track unconditional mode on MARBLE. Including GTZAN genre classification, GS key recognition, MTG top 50 tagging, and EMO emotion regression.

Dataset GTZAN GS MTG EMO Task Genre Key Top50 Emotion

Metrics Acc↑ AccRefined↑ AP↑ AUC↑ R2V↑ R2A↑

MERT [2023] 78.6 65.6 29.9 83.4 61.2 74.7 MusicFM [2024] 83.8 63.9 - - 60.3 76.3 MuQiter [2025] 85.6 65.0 - - 62.8 76.1 CLAP [2023b] 82.1 16.0 27.7 82.0 54.1 70.3 CLaMP 3 [2025] 86.6 53.8 30.2 82.4 59.1 70.0 YuE 83.4 67.0 29.2 82.7 58.9 75.0

###### 9. Representation Quality

YuE, fundamentally designed as a generative model rather than explicitly for representation learning, is evaluated with MARBLE [Yuan et al., 2024b] here using its Stage-1 LM in an unconditional single-track setting. Notably, this mode serves primarily as an auxiliary task and is disabled half way through the training. Moreover, it exclusively leverages discrete codes from codebook-0, implying a significant reduction in available information compared to dedicated representation learning models.

Despite these inherent limitations, YuE achieves state-of-the-art performance on the GS key recognition task (Acc=67.0%, see Table 10), demonstrating a good sense of tonality and modality, which is essential for composing and singing in tune. Furthermore, its performance remains competitive with existing methods across other tasks, such as GTZAN genre classification, MTG tagging, and EMO emotion regression, underscoring YuE’s robust general-purpose representation quality and learned musical skills.

###### 10. Emergent Abilities

We strongly encourage readers to visit our demo page for audio examples illustrating the capabilities described.32 Scaling up the model significantly enhances generation quality and unlocks novel abilities.

Advanced Vocal Techniques. Beyond basic pop and rap vocals, our model spontaneously acquires diverse and expressive singing techniques, typically mastered only by gifted human vocalists through extensive training. These include vibrato, glissando, bel canto, death growl, mix voice, belting, riffs and runs, vocal fry, Beijing Opera, and Shanbei folk vocals. This indicates our Dual-NTP approach effectively captures subtle nuances in vocal performance.

Spontaneous Performance. Our model spontaneously demonstrates musically expressive behaviors. For instance, in jazz performances, it naturally continues with scat singing after running out of lyrics; in a cappella, it simultaneously generates multi-part harmonies with distinct vocalists handling melody and accompaniment; in folk music, it inserts contextually appropriate instrumental solos, such as harmonica interludes, during vocal pauses.

World Music & Pattern Mixing. Our model effectively captures long-tail global music styles beyond mainstream western genres. For instance, it generates creative fusions such as Chinese gangsta rap accompanied by Japanese shamisen instrumentation and scales. It can also seamlessly blend distinct regional vocal styles, combining Chinese opera, Shanbei folk singing, and traditional Chinese vocals within a single cohesive performance.

32https://map-yue.github.io/

Voice Cloning. Our model demonstrates high-fidelity voice cloning capabilities at inference time, successfully replicating distinct vocal identities. For example, we accurately reproduce the unique voices of Billie Eilish and Faye Wong (王菲) while generating entirely new lyrics and melodies. These cloned voices retain their signature timbral qualities, breathy textures, and emotional nuances, highlighting the model’s ability to capture and reproduce subtle vocal characteristics from limited reference data provided only at inference.

Style Transfer. Our model shows versatile style transfer capabilities, enabling the generation of diverse and expressive vocal performances across different languages, genres, and timbres. YuE enables cross-lingual and genre adaptation while preserving the original lyrical and melodic structure. In one example, a Japanese female J-pop vocal performance is transformed into an English male rap with the same city pop accompaniment. The model not only shifts the vocal characteristics but also adjusts prosody, phrasing, and expressiveness to ensure stylistic coherence, demonstrating its deep understanding of genre-specific vocal performance.

Code Switching. The model naturally handles code-switching, smoothly transitioning between multiple languages or dialects within the same vocal performance, while preserving linguistic and stylistic consistency.

###### 11. Memorization Effect

Following previous literature [Agostinelli et al., 2023, Yuan et al., 2024a], we investigate whether YuE, in its ICL mode—conditioned on a 30-second audio prompt and original lyrics—reproduces significant portions of its training data. ICL is generally more prone to memorization, making this evaluation critical.

We employ ByteCover2 [Du et al., 2022], a state-of-the-art retrieval model optimized for melodysensitive similarity across entire songs.33 Specifically, we create two sets of 𝑁 = 1200 music samples: R (Ref), comprising YuE’s training examples, and G (Gen), comprising corresponding samples generated by YuE in the ICL setting. We compute cosine similarity scores for each pair (𝑟, 𝑔) with 𝑟 ∈ R and 𝑔 ∈ G, analyzing the top 1% of scores since frequent high-similarity pairs would suggest substantial memorization.

To contextualize these results, we compare them to real-world baselines from GTZAN (genrelevel similarities) and Covers80 (known melodic duplicates). Results are shown in Figure 15. The similarity distribution for Ref-Gen pairs is significantly lower than Covers80 and remains moderate even compared to GTZAN. While short repetitive motifs, particularly percussive loops, occasionally occur, overall results indicate that YuE’s ICL mode does not engage in extensive copying. Instead, YuE recombines learned musical patterns creatively, demonstrating that the ICL mode effectively generates original content rather than memorizing training samples.

###### 12. Unsuccessful Attempts

During our initial scaling experiments, we encountered several challenges and setbacks. Here, we share these unsuccessful experiences to inform and inspire future research directions.

Acoustic Tokens. As detailed in Section 8.1, LMs trained on acoustic tokens consistently exhibited convergence difficulties and yielded higher losses compared to semantic-enhanced tokens. We attribute these challenges primarily to inherent limitations of current acoustic token representations, typically derived from RVQ-GANs. Such tokens often prioritize compression efficiency over representational quality and typically have limited capacity. Consequently, models trained on these tokens may tend to adopt shortcuts, frequently resorting to direct information copying. Even when scaled substantially, these models achieve only marginal improvements [Hansen-Estruch et al., 2025, Xin et al., 2024, Parker et al., 2024]. We argue the

33We do not use ByteCover3 [Du et al., 2023] as it specializes in shorter segments.

Similarity Distribution Comparison

1.00

0.75

CosineSimilarity

0.50

0.25

0.00

Covers80 Ref-Gen GTZAN

- Figure 15. Box-plot comparison of cosine similarity across three scenarios: Covers80, Ref-Gen (our training vs. generated sets), and GTZAN. The black bar denotes the median, and the diamond denotes the mean.

lossy nature of discrete representations, limited semantic relevance [Zhang et al., 2023], and excessive focus on reconstruction tasks collectively contribute to the difficulties observed in fitting acoustic tokens.

Unconditional Pre-train. We initially pre-trained large models to learn general representations for cross-modal alignment (text-to-vocal) via fine-tuning. At smaller scales (e.g., sub-billion parameters), models showed moderate success in learning basic mappings. However, at 7B parameters, unconditional pre-training became counterproductive: fine-tuning failed to establish effective cross-modal alignment. We hypothesize that larger models internalize overly generic priors, overshadowing the specific conditional mappings needed for alignment. This “catastrophic inertia” prevents large models from adapting effectively to lyrics-to-song tasks.

Early Activation of ICL. We observed that early activation of ICL data led to a poor musicality. Initially, the model began to excessively rely on the reference audio, resulting in overfitting and diminished musicality. After removing the reference audio later in the training process, the model continued to produce a significant number of invalid outputs, such as silence or noise. This problem became more pronounced with scaling, where larger models struggled even more to recover from this shortcut learning. These results highlight the importance of carefully managing the timing of ICL data activation to avoid overfitting and preserve the model’s creativity.

###### 13. Conclusion and Future Work

We introduced YuE, an open-source foundation model family designed for long-form lyricsto-song generation. By combining large-scale data, track-decoupled next-token prediction, a segment-wise conditioning strategy, and a redesigned in-context learning framework, YuE can generate coherent, full-length songs with expressive vocals and detailed musical structure. Experimental results show that YuE matches or exceeds several commercial systems in musicality, controllability, and cross-lingual lyrics following, and it also achieves competitive music understanding results on standard benchmarks. These findings highlight the promise of open, large-scale music models in enabling controllable, high-quality song generation and in advancing broader research into music-aware AI systems.

YuE’s approach can be extended by improving acoustic fidelity and mixing, incorporating musical knowledge such as chord progressions and instrumentation theory, and integrating deeper prosodic and emotional controls. Multilingual and cross-cultural expansions hold

significant potential, especially for underrepresented musical traditions. Beyond music creation, YuE can benefit applications in music education, accessibility, and therapy, and can serve as an accessible platform for continued community-driven innovation in open music AI research.

###### 14. Ethics and Responsibility

Ensuring ethical and responsible AI-generated music is crucial for fostering transparency, accessibility, and fair contribution to the music industry. As suggested by Ma et al. [2024], to promote accountability, we advocate for the inclusion of AI-generated / AI-assisted tags in generated content, increasing transparency for both musicians and audiences. Additionally, our memorization-effect experiments in Section 11 demonstrate that our design maintains creativity without plagiarizing, even under strong training set conditioning.

In contrast to closed-source commercial systems, our model leverages an exceptionally diverse training dataset, explicitly enriched with culturally diverse music content. This enables the model to innovate and create within niche musical styles effectively (see Section 10). As such, our model can serve as a parameterized knowledge base, contributing to the preservation and expansion of human musical artistry and cultural heritage.

This study has been reviewed and approved by the Human and Artefacts Research Ethics Committee under protocol HREP-2023-0230, titled Building Platform Technologies for Symbiotic Creativity in Hong Kong. The approval ensures that our research adheres to ethical guidelines in data usage, AI generation, and cultural representation. The approval remains effective until 30-Jan-2027.

###### 15. Contributions and Acknowledgments

Core Contributors Ruibin Yuan, Lead, Pre-train, Data, Eval

HKUST, Moonshot.ai, MAP, ryuanab@connect.ust.hk

Hanfeng Lin, Pre-train, Data, Eval, Inference

HKUST, MAP, hanfeng@ust.hk

Shuyue Guo, Pre-train, Demo

MAP

Ge Zhang, Pre-train

MAP, gezhang@umich.edu

Jiahao Pan, Pre-train, Eval, Data

HKUST, MAP, fengshicherish@gmail.com

Contributors Yongyi Zang, Upsampler, Eval

Independent

Haohe Liu, Upsampler, Tokenizer, Demo

University Of Surrey, MAP

Yiming Liang, Eval Lead

MAP

Wenye Ma, Representation Learning

MBZUAI, MAP

Xingjian Du, Memorization Effect

University of Rochester, MAP

Xinrun Du, Pre-train

MAP

Zhen Ye, Tokenizer

HKUST

Tianyu Zheng, Pre-train

MAP

Zhengxuan Jiang, Inference

MAP

Yinghao Ma, Eval

MAP, Queen Mary University of London

Minghao Liu, Eval, Data

2077AI, MAP

Zeyue Tian, Eval

HKUST, MAP

Ziya Zhou, Eval, Data

HKUST, MAP

Liumeng Xue, Eval, Data

HKUST, MAP

Xingwei Qu, Pre-train, Eval

MAP

Yizhi Li, Eval

MAP, University of Manchester

Shangda Wu, Eval

Central Conservatory of Music, MAP

Tianhao Shen, Eval, Inference

MAP

Ziyang Ma, Eval

MAP, SJTU, NTU

Jun Zhan, Eval

Fudan University

Chunhui Wang, Eval, Pre-train

Geely

Yatian Wang, Eval

HKUST

Xiaowei Chi, Eval

HKUST

Xinyue Zhang, Eval

HKUST

Zhenzhu Yang, Eval

HKUST

Xiangzhou Wang, Eval

MAP

Shansong Liu, Eval

Meituan

Lingrui Mei, Eval

Meituan

Peng Li, Eval

HKUST

Junjie Wang, Eval

Tsinghua University

Jianwei Yu, Data, Inference

Moonshot.ai

Guojian Pang, Inference

MAP

Xu Li, Eval

Xiaohongshu

Zihao Wang, Data

Zhejiang University, Carnegie Mellon University

###### Academic Advisors Xiaohuan Zhou

MAP

Lijun Yu

Carnegie Mellon University

Emmanouil Benetos

Queen Mary University of London, MAP

Yong Chen

Geely

Chenghua Lin

University of Manchester, MAP

Xie Chen

Shanghai Jiao Tong University

Gus Xia

MBZUAI, MAP

Zhaoxiang Zhang

Chinese Academy of Sciences

Chao Zhang

Tsinghua University

Wenhu Chen

University of Waterloo, MAP

Xinyu Zhou

Moonshot.ai

Xipeng Qiu

Fudan University

Roger Dannenberg

Carnegie Mellon University, MAP

###### Correspondence (Alphabetical Order) Jiaheng Liu

Nanjing University, MAP, 13121221227@163.com

Jian Yang

MAP, jiaya@buaa.edu.cn

Wenhao Huang

MAP, rubio8741@gmail.com

Wei Xue

HKUST, weixue@ust.hk

Xu Tan

Moonshot.ai, MAP, tanxu2012@gmail.com

Yike Guo

HKUST, yikeguo@ust.hk

###### References

A. Agostinelli, T. I. Denk, Z. Borsos, J. Engel, M. Verzetti, A. Caillon, Q. Huang, A. Jansen, A. Roberts, M. Tagliasacchi, et al. MusicLM: Generating music from text. arXiv preprint:2301.11325, 2023.

A. Baevski, Y. Zhou, A. Mohamed, and M. Auli. wav2vec 2.0: A framework for self-supervised learning of speech representations. Advances in neural information processing systems, 33:12449– 12460, 2020.

A. Baevski, W.-N. Hsu, Q. Xu, A. Babu, J. Gu, and M. Auli. Data2Vec: A general framework for self-supervised learning in speech, vision and language. In International Conference on Machine Learning, pages 1298–1312, 2022.

- Y. Bai, H. Chen, J. Chen, Z. Chen, Y. Deng, X. Dong, L. Hantrakul, W. Hao, Q. Huang, Z. Huang, et al. Seed-music: A unified framework for high quality and controlled music generation. arXiv preprint arXiv:2409.09214, 2024.

D. Bogdanov, M. Won, P. Tovstogan, A. Porter, and X. Serra. The mtg-jamendo dataset for automatic music tagging. In Machine Learning for Music Discovery Workshop, International Conference on Machine Learning (ICML 2019), Long Beach, CA, United States, 2019. URL http://hdl.handle.net/10230/42015.

- Z. Borsos, R. Marinier, D. Vincent, E. Kharitonov, O. Pietquin, M. Sharifi, D. Roblek, O. Teboul, D. Grangier, M. Tagliasacchi, and N. Zeghidour. Audiolm: A language modeling approach to audio generation. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 31: 2523–2533, 2023. doi: 10.1109/TASLP.2023.3288409.

M. J. Bruderer, M. F. McKinney, and A. Kohlrausch. The perception of structural boundaries in melody lines of western popular music. Musicae Scientiae, 13(2):273–313, 2009.

- J. Chen, X. Tan, J. Luan, T. Qin, and T.-Y. Liu. Hifisinger: Towards high-fidelity neural singing voice synthesis. arXiv preprint:2009.01776, 2020.
- K. Chen, Y. Wu, H. Liu, M. Nezhurina, T. Berg-Kirkpatrick, and S. Dubnov. Musicldm: Enhancing novelty in text-to-music generation using beat-synchronous mixup strategies. arXiv preprint arXiv:2308.01546, 2023.

K. Chen, Y. Wu, H. Liu, M. Nezhurina, T. Berg-Kirkpatrick, and S. Dubnov. MusicLDM: Enhancing novelty in text-to-music generation using beat-synchronous mixup strategies. In International Conference on Acoustics, Speech and Signal Processing, pages 1206–1210. IEEE, 2024.

- Y. Chu, J. Xu, Q. Yang, H. Wei, X. Wei, Z. Guo, Y. Leng, Y. Lv, J. He, J. Lin, et al. Qwen2-audio technical report. arXiv preprint arXiv:2407.10759, 2024.

- Y.-A. Chung, Y. Zhang, W. Han, C.-C. Chiu, J. Qin, R. Pang, and Y. Wu. W2V-Bert: Combining contrastive learning and masked language modeling for self-supervised speech pre-training. In IEEE Automatic Speech Recognition and Understanding Workshop, pages 244–250. IEEE, 2021.

- J. Copet, F. Kreuk, I. Gat, T. Remez, D. Kant, G. Synnaeve, Y. Adi, and A. Défossez. Simple and controllable music generation. arXiv preprint:2306.05284, 2023a.

- J. Copet, F. Kreuk, I. Gat, T. Remez, D. Kant, G. Synnaeve, Y. Adi, and A. Défossez. Simple and controllable music generation. arXiv preprint arXiv:2306.05284, 2023b.

A. Défossez, J. Copet, G. Synnaeve, and Y. Adi. High fidelity neural audio compression. arXiv preprint arXiv:2210.13438, 2022.

A. Défossez, L. Mazaré, M. Orsini, A. Royer, P. Pérez, H. Jégou, E. Grave, and N. Zeghidour. Moshi: a speech-text foundation model for real-time dialogue. arXiv preprint arXiv:2410.00037, 2024.

P. Dhariwal, H. Jun, C. Payne, J. W. Kim, A. Radford, and I. Sutskever. Jukebox: A generative

- model for music. arXiv preprint arXiv:2005.00341, 2020a.

P. Dhariwal, H. Jun, C. Payne, J. W. Kim, A. Radford, and I. Sutskever. Jukebox: A generative

- model for music. arXiv preprint arXiv:2005.00341, 2020b.

C. Donahue, A. Caillon, A. Roberts, E. Manilow, P. Esling, A. Agostinelli, M. Verzetti, I. Simon,

- O. Pietquin, N. Zeghidour, et al. Singsong: Generating musical accompaniments from singing. arXiv preprint arXiv:2301.12662, 2023.

- X. Du, K. Chen, Z. Wang, B. Zhu, and Z. Ma. Bytecover2: Towards dimensionality reduction of latent embedding for efficient cover song identification. In ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 616–620. IEEE, 2022.

- X. Du, Z. Wang, X. Liang, H. Liang, B. Zhu, and Z. Ma. Bytecover3: Accurate cover song identification on short queries. In ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1–5. IEEE, 2023.

- Y. Du, Z. Ma, Y. Yang, K. Deng, X. Chen, B. Yang, Y. Xiang, M. Liu, and B. Qin. Cot-st: Enhancing llm-based speech translation with multimodal chain-of-thought. arXiv preprint arXiv:2409.19510, 2024a.
- Z. Du, Q. Chen, S. Zhang, K. Hu, H. Lu, Y. Yang, H. Hu, S. Zheng, Y. Gu, Z. Ma, et al. Cosyvoice: A scalable multilingual zero-shot text-to-speech synthesizer based on supervised semantic tokens. arXiv preprint arXiv:2407.05407, 2024b.

- Z. Evans, J. D. Parker, C. Carr, Z. Zukowski, J. Taylor, and J. Pons. Stable audio open. arXiv preprint:2407.14358, 2024.

- R. Geirhos, J.-H. Jacobsen, C. Michaelis, R. Zemel, W. Brendel, M. Bethge, and F. A. Wichmann. Shortcut learning in deep neural networks. Nature Machine Intelligence, 2(11):665–673, 2020.

P. Hansen-Estruch, D. Yan, C.-Y. Chung, O. Zohar, J. Wang, T. Hou, T. Xu, S. Vishwanath, P. Vajda, and X. Chen. Learnings from scaling visual tokenizers for reconstruction and generation. arXiv preprint arXiv:2501.09755, 2025.

- S. Hershey, S. Chaudhuri, D. P. Ellis, J. F. Gemmeke, A. Jansen, R. C. Moore, M. Plakal, D. Platt,

- R. A. Saurous, B. Seybold, et al. Cnn architectures for large-scale audio classification. In 2017 ieee international conference on acoustics, speech and signal processing (icassp), pages 131–135. IEEE, 2017.

Z. Hong, C. Cui, R. Huang, L. Zhang, J. Liu, J. He, and Z. Zhao. UniSinger: Unified end-to-end singing voice synthesis with cross-modality information matching. In ACM International Conference on Multimedia, pages 7569–7579, 2023.

C.-Z. A. Huang, A. Vaswani, J. Uszkoreit, N. Shazeer, I. Simon, C. Hawthorne, A. M. Dai, M. D. Hoffman, M. Dinculescu, and D. Eck. Music transformer. arXiv preprint arXiv:1809.04281, 2018.

Q. Huang, D. S. Park, T. Wang, T. I. Denk, A. Ly, N. Chen, Z. Zhang, Z. Zhang, J. Yu, C. Frank, et al. Noise2Music: Text-conditioned music generation with diffusion models. arXiv preprint:2302.03917, 2023.

K. Kilgour, M. Zuluaga, D. Roblek, and M. Sharifi. Fréchet audio distance: A reference-free metric for evaluating music enhancement algorithms. In Proc. Interspeech, 2019.

T. Kim and J. Nam. All-in-one metrical and functional structure analysis with neighborhood attentions on demixed audio. In 2023 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA), pages 1–5. IEEE, 2023.

K. Koutini, J. Schlüter, H. Eghbal-Zadeh, and G. Widmer. Efficient training of audio transformers with patchout. arXiv preprint arXiv:2110.05069, 2021.

- R. Kumar, P. Seetharaman, A. Luebs, I. Kumar, and K. Kumar. High-fidelity audio compression with improved rvqgan. Advances in Neural Information Processing Systems, 36, 2024.
- S. Lei, Y. Zhou, B. Tang, M. W. Lam, F. Liu, H. Liu, J. Wu, S. Kang, Z. Wu, and H. Meng. Songcreator: Lyrics-based universal song generation. arXiv preprint arXiv:2409.06029, 2024.

- S. Lei, Y. Zhou, B. Tang, M. W. Lam, H. Liu, J. Wu, S. Kang, Z. Wu, H. Meng, et al. Songcreator: Lyrics-based universal song generation. Advances in Neural Information Processing Systems, 37: 80107–80140, 2025.

- F. Lerdahl and R. S. Jackendoff. A Generative Theory of Tonal Music, reissue, with a new preface. MIT press, 1996.

- R. Li, Z. Hong, Y. Wang, L. Zhang, R. Huang, S. Zheng, and Z. Zhao. Accompanied singing voice synthesis with fully text-controlled melody. arXiv preprint arXiv:2407.02049, 2024.

- Y. Li, R. Yuan, G. Zhang, Y. Ma, X. Chen, H. Yin, C. Lin, A. Ragni, E. Benetos, N. Gyenge, et al. MERT: Acoustic music understanding model with large-scale self-supervised training. arXiv preprint:2306.00107, 2023.

H. Liu, Z. Chen, Y. Yuan, X. Mei, X. Liu, D. Mandic, W. Wang, and M. D. Plumbley. AudioLDM: Text-to-audio generation with latent diffusion models. Proceedings of the International Conference on Machine Learning, 2023.

- H. Liu, X. Xu, Y. Yuan, M. Wu, W. Wang, and M. D. Plumbley. SemantiCodec: An ultra low bitrate semantic audio codec for general sound. IEEE Journal of Selected Topics in Signal Processing, 18

(8):1448–1461, 2024a. doi: 10.1109/JSTSP.2024.3506286.

- H. Liu, Y. Yuan, X. Liu, X. Mei, Q. Kong, Q. Tian, Y. Wang, W. Wang, Y. Wang, and M. D. Plumbley. AudioLDM 2: Learning holistic audio generation with self-supervised pretraining. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 32:2871–2883, 2024b. doi: 10.1109/TASLP.2024.3399607.

J. Liu, C. Li, Y. Ren, F. Chen, and Z. Zhao. Diffsinger: Singing voice synthesis via shallow diffusion mechanism. In Proceedings of the AAAI conference on artificial intelligence, 2022.

- Z. Liu, S. Ding, Z. Zhang, X. Dong, P. Zhang, Y. Zang, Y. Cao, D. Lin, and J. Wang. Songgen: A single stage auto-regressive transformer for text-to-song generation. arXiv preprint arXiv:2502.13128, 2025.

- Y. Ma, A. Øland, A. Ragni, B. M. Del Sette, C. Saitis, C. Donahue, C. Lin, C. Plachouras, E. Benetos, E. Shatri, et al. Foundation models for music: A survey. arXiv preprint arXiv:2408.14340, 2024.
- Z. Ma, Z. Zheng, C. Tang, Y. Wang, and X. Chen. MT4SSL: Boosting self-supervised speech representation learning by integrating multiple targets. In Proceedings of Interspeech, 2023.

- Z. Ma, Z. Chen, Y. Wang, E. S. Chng, and X. Chen. Audio-CoT: Exploring chain-of-thought reasoning in large audio language model. arXiv preprint arXiv:2501.07246, 2025.

- S. A. Mehr, M. Singh, D. Knox, D. M. Ketter, D. Pickens-Jones, S. Atwood, C. Lucas, N. Jacoby, A. A. Egner, E. J. Hopkins, et al. Universality and diversity in human song. Science, 366(6468): eaax0868, 2019.

O. Nieto, G. J. Mysore, C.-i. Wang, J. B. Smith, J. Schlüter, T. Grill, and B. McFee. Audio-based music structure analysis: Current trends, open challenges, and applications. Transactions of the International Society for Music Information Retrieval, 3(1), 2020.

J. D. Parker, A. Smirnov, J. Pons, C. Carr, Z. Zukowski, Z. Evans, and X. Liu. Scaling transformers

for low-bitrate high-quality speech coding. arXiv preprint arXiv:2411.19842, 2024. C. Payne. Musenet. https://openai.com/research/musenet, 2022.

- X. Qu, Y. Bai, Y. Ma, Z. Zhou, K. M. Lo, J. Liu, R. Yuan, L. Min, X. Liu, T. Zhang, et al. Mupt: A generative symbolic music pretrained transformer. arXiv preprint arXiv:2404.06393, 2024.

- S. Schneider, A. Baevski, R. Collobert, and M. Auli. Wav2Vec: Unsupervised pre-training for speech recognition. INTERSPEECH, pages 3465–3469, 2019.

M. Shoeybi, M. Patwary, R. Puri, P. LeGresley, J. Casper, and B. Catanzaro. Megatron-lm: Training multi-billion parameter language models using model parallelism. arXiv preprint arXiv:1909.08053, 2019.

H. Siuzdak. Vocos: Closing the gap between time-domain and fourier-based neural vocoders for high-quality audio synthesis. arXiv preprint arXiv:2306.00814, 2023.

J. Su, M. Ahmed, Y. Lu, S. Pan, W. Bo, and Y. Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568, 2024.

- T. L. Team. Introducing meta llama 3: The most capable openly available llm to date, 2024. URL https://ai.meta.com/blog/meta-llama-3/.

A. Tjandra, Y.-C. Wu, B. Guo, J. Hoffman, B. Ellis, A. Vyas, B. Shi, S. Chen, M. Le, N. Zacharov, et al. Meta audiobox aesthetics: Unified automatic quality assessment for speech, music, and sound. arXiv preprint arXiv:2502.05139, 2025.

H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux, T. Lacroix, B. Rozière, N. Goyal, E. Hambro, F. Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023a.

H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y. Babaei, N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023b.

C. Wang, S. Chen, Y. Wu, Z. Zhang, L. Zhou, S. Liu, Z. Chen, Y. Liu, H. Wang, J. Li, L. He, S. Zhao, and F. Wei. Neural codec language models are zero-shot text to speech synthesizers. arXiv, abs/2301.02111, 2023a.

C. Wang, S. Chen, Y. Wu, Z. Zhang, L. Zhou, S. Liu, Z. Chen, Y. Liu, H. Wang, J. Li, et al. Neural codec language models are zero-shot text to speech synthesizers. arXiv preprint:2301.02111, 2023b.

- X. Wang, M. Jiang, Z. Ma, Z. Zhang, S. Liu, L. Li, Z. Liang, Q. Zheng, R. Wang, X. Feng, et al. Spark-tts: An efficient llm-based text-to-speech model with single-stream decoupled speech tokens. arXiv preprint arXiv:2503.01710, 2025.
- Y. Wang, R. Hu, R. Huang, Z. Hong, R. Li, W. Liu, F. You, T. Jin, and Z. Zhao. Prompt-Singer: Controllable singing-voice-synthesis with natural language prompt. arXiv preprint:2403.11780, 2024.

J. Wei, X. Wang, D. Schuurmans, M. Bosma, F. Xia, E. Chi, Q. V. Le, D. Zhou, et al. Chain-ofthought prompting elicits reasoning in large language models. Advances in Neural Information Processing Systems, 35:24824–24837, 2022.

M. Won, Y.-N. Hung, and D. Le. A foundation model for music informatics. In ICASSP 2024 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1226–1230, 2024. doi: 10.1109/ICASSP48485.2024.10448314.

S. Wu, Z. Guo, R. Yuan, J. Jiang, S. Doh, G. Xia, J. Nam, X. Li, F. Yu, and M. Sun. Clamp 3: Universal music information retrieval across unaligned modalities and unseen languages,

###### 2025. URL https://arxiv.org/abs/2502.10362.

- Y. Wu, K. Chen, T. Zhang, Y. Hui, T. Berg-Kirkpatrick, and S. Dubnov. Large-scale contrastive language-audio pretraining with feature fusion and keyword-to-caption augmentation. In ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1–5. IEEE, 2023a.

Y. Wu, K. Chen, T. Zhang, Y. Hui, T. Berg-Kirkpatrick, and S. Dubnov. Large-scale contrastive language-audio pretraining with feature fusion and keyword-to-caption augmentation. In IEEE International Conference on Acoustics, Speech and Signal Processing, 2023b.

- Y. Wu, J. Shi, Y. Tang, S. Yang, Q. Jin, et al. TokSing: Singing voice synthesis based on discrete tokens. arXiv preprint:2406.08416, 2024.

D. Xin, X. Tan, S. Takamichi, and H. Saruwatari. Bigcodec: Pushing the limits of low-bitrate

###### neural speech codec, 2024. URL https://arxiv.org/abs/2409.05377.

- W. Xiong, J. Liu, I. Molybog, H. Zhang, P. Bhargava, R. Hou, L. Martin, R. Rungta, K. A. Sankararaman, B. Oguz, et al. Effective long-context scaling of foundation models. arXiv preprint arXiv:2309.16039, 2023.

D. Yang, S. Liu, R. Huang, J. Tian, C. Weng, and Y. Zou. Hifi-codec: Group-residual vector quantization for high fidelity audio codec. arXiv preprint arXiv:2305.02765, 2023a.

D. Yang, J. Tian, X. Tan, R. Huang, S. Liu, X. Chang, J. Shi, S. Zhao, J. Bian, X. Wu, et al. Uniaudio: An audio foundation model toward universal audio generation. arXiv preprint arXiv:2310.00704, 2023b.

- Z. Ye, P. Sun, J. Lei, H. Lin, X. Tan, Z. Dai, Q. Kong, J. Chen, J. Pan, Q. Liu, et al. Codec does matter: Exploring the semantic shortcoming of codec for audio language model. arXiv preprint arXiv:2408.17175, 2024.

R. Yuan, H. Lin, Y. Wang, Z. Tian, S. Wu, T. Shen, G. Zhang, Y. Wu, C. Liu, Z. Zhou, et al. Chatmusician: Understanding and generating music intrinsically with llm. arXiv preprint arXiv:2402.16153, 2024a.

R. Yuan, Y. Ma, Y. Li, G. Zhang, X. Chen, H. Yin, Y. Liu, J. Huang, Z. Tian, B. Deng, et al. Marble: Music audio representation benchmark for universal evaluation. Advances in Neural Information Processing Systems, 36, 2024b.

- G. Zhang, S. Qu, J. Liu, C. Zhang, C. Lin, C. L. Yu, D. Pan, E. Cheng, J. Liu, Q. Lin, et al. Mapneo: Highly capable and transparent bilingual large language model series. arXiv preprint arXiv:2405.19327, 2024.

- X. Zhang, D. Zhang, S. Li, Y. Zhou, and X. Qiu. Speechtokenizer: Unified speech tokenizer for speech large language models. arXiv preprint arXiv:2308.16692, 2023.
- Y. Zhang, J. Cong, H. Xue, L. Xie, P. Zhu, and M. Bi. ViSinger: Variational inference with adversarial learning for end-to-end singing voice synthesis. In IEEE International Conference on Acoustics, Speech and Signal Processing, pages 7237–7241. IEEE, 2022.

- H. Zhu, Y. Zhou, H. Chen, J. Yu, Z. Ma, R. Gu, W. Tan, and X. Chen. Muq: Self-supervised music representation learning with mel residual vector quantization. arXiv preprint arXiv:2501.01108, 2025.

###### A. Subjective Evaluation

###### A.1. Evaluation Methods

In this subjective evaluation experiment, annotators were required to perform pairwise comparative evaluations of music generation outputs from multiple models. Each test unit comprised two distinct musical pieces generated by different models. Following complete playback of both samples, annotators conducted binary comparative selections (options: Superiority of A, Superiority of B, or Equivalence between A and B) across predefined evaluation dimensions. Mandatory preference judgments were enforced for each dimensional criterion, with explicit instructions to minimize the frequency of selecting the equivalence option. The evaluation protocol incorporated a double-blind procedure with randomized presentation order of audio pairs to mitigate potential ordering effects.

[Figure 26]

Figure 16. Subjective evaluation platform.

###### A.2. Evaluation Dimensions and Definitions

- 1) Overall Musicality Definition: The musical artistic value and professionalism demonstrated by the work as a whole, reflecting whether it approaches the creative level of professional musicians or composers. Evaluation Criteria: Smoothness of the melody, complexity and rationality of the harmony, precision and rhythmic flow, and the artistic and creative qualities of the overall arrangement.
- 2) Vocal Quality Definition: The acoustic quality of vocal performance in the work. Evaluation Criteria: Pitch, rhythmic stability, naturalness of vocal timbre (resembling human singing), fullness and warmth of timbre, degree of mechanical or distorted sound, clarity of vocals, and richness in capturing delicate emotional expressions (e.g., variations in breath control, articulation precision, emotional conveyance).
- 3) Accompaniment Quality Definition: The acoustic quality of the instrumental accompaniment in the work. Evaluation Criteria: Realism and authenticity of instrumental timbres, dynamic variation and detail richness in instrumental expression (e.g., subtlety in guitar plucking or percussion dynamics).
- 4) Arrangement Complexity Definition: The layering, coherence, balance, and creativity of the musical arrangement in the work. Evaluation Criteria: Clarity of arrangement layers, coordination and interplay between instruments, balance of accompaniment within the overall audio track (e.g., appropriate vol-

- ume and frequency distribution), fullness of low frequencies, brightness of high frequencies, diversity of arrangement elements (e.g., harmony, melodic lines, rhythm patterns across multiple dimensions), creativity, and variation and emotional progression between sections.
- 5) Melodic Memorability and Catchiness Definition: The memorability, accessibility, and resonance-inducing capability of the melody. Evaluation Criteria: Ease of memorization and singability, catchiness, emotional resonance, and repeated hooks or memorable elements, especially in the chorus.
- 6) Vocal-Accompaniment Matching Definition: The consistency and compatibility between vocal melodies and instrumental accompaniment in terms of musical style, modality, harmony, and rhythm. Evaluation Criteria: Compatibility of vocal melodies and accompaniment in modality, harmony, and rhythm, and absence of dissonance or conflict.
- 7) Song Structure Clarity Definition: The logical coherence and sectional distinctiveness of the overall song structure. Evaluation Criteria: Clarity of the song’s structure (e.g., differentiation among verses, choruses, and interludes), naturalness of transitions between sections, and structural completeness.

###### A.3. Conditional Evaluation Dimension and Definitions

- 8) Lyrics Following Definition: The accuracy of AI-generated vocals in performing the lyrics specified in the prompt. Evaluation Criteria: Accuracy of lyric delivery (whether the specified lyrics are correctly performed), clarity of pronunciation (whether the lyrics are intelligible), alignment of lyrics rhythm with the musical beat, and naturalness and correctness of multilingual lyric transitions and pronunciations.
- 9) Multilingual Lyrics Switching Naturalness and Correctness Definition: The fluency and accuracy of AI-generated vocals when performing lyrics in multiple languages, including the smoothness of transitions and the grammatical and pronunciation correctness of different languages. Evaluation Criteria: Fluency and naturalness of multilingual transitions: whether transitions between languages are smooth and seamless without abrupt changes or noticeable interruptions; accuracy of pronunciation for multilingual lyrics: whether the pronunciation in different languages is precise, clear, and adheres to the phonetic norms of each language, avoiding mispronunciations or accent deviations that could hinder understanding.
- 10) Genre Controllability Definition: The degree to which the generated music accurately reflects the musical genre specified in the prompt. Evaluation Criteria: Accuracy of musical genre characteristics (whether the generated music aligns with the features of the genre specified in the prompt, such as jazz, pop, classical, rock, etc.).
- 11) Instrument and Vocal Configuration Controllability Definition: The extent to which the generated music adheres to the instrument and vocal configuration specified in the prompt. Evaluation Criteria: Matching of instrument and vocal configuration (whether the generated music follows the specifications in the prompt, such as piano, guitar, male or female vocals, choir, etc.).
- 12) Emotional Expressiveness Definition: The accuracy and impact of emotional expression in the generated music, as specified in the prompt. Evaluation Criteria: Alignment of musical emotions with the emotional description in the prompt (e.g., passionate, sorrowful, cheerful).
- 13) Tempo and Rhythm Definition: The congruence of the music’s tempo (BPM) and rhythm with the requirements

specified in the prompt. Evaluation Criteria: Consistency of generated music tempo (BPM) with the tempo specified in the prompt, and adherence to the required rhythmic patterns.

###### B. Qwen2Audio-Instruct Tagging Prompt Music Tagging Prompt

Analyze the provided audio and describe its features in a valid JSON format with the following keys: Music_genre, Instrument, and Mood. If there are multiple entries for any key, represent them as a list of strings. Example format:

{

"Music_genre": ["Jazz"], "Instrument": ["Saxophone", "Piano"], "Mood": ["Relaxed"]

}

Vocal Tagging Prompt

Analyze the provided audio and describe its vocal characteristics in a valid JSON format with the following keys: gender, age, and vocal_timbre. If there are multiple entries for any key, represent them as a list of strings. Example format:

{

"gender": ["female"], "age": ["adult"], "vocal_timbre": ["bright", "airy"]

}

###### C. Multilingual Subjective Evaluation

[Figure 27]

YuE

60 80 25 65 70

80

70

Udio

36 20 32 55 36

60

Suno

73 75 68 86 64

50

40

Hailuo

30 35 45 14 25

30

Tiangong

51 30 64 36 75 20

Overall

YuE

Udio

SunoV4

Hailuo

Tiangong

(a) Chinese - Lyrics Following

100

[Figure 28]

YuE

55 40 50 60 70

80

Udio

62 60 20 70 100

60

Suno V4

75 50 80 80 90

40

Hailuo

37 40 30 20 60

20

Tiangong

20 30 0 10 40

0

Overall

YuE

Udio

SunoV4

Hailuo

Tiangong

(c) Korean - Lyrics Following

80

[Figure 29]

YuE

70 80 60 60 80

70

Udio

- 31 20 25 20 60

60 40 75 75 50

56 40 80 25 80

- 32 20 40 50 20

60

Suno

50

40

Hailuo

30

Tiangong

20

Overall

YuE

Udio

SunoV4

Hailuo

Tiangong

(e) Japanese - Lyrics Following

100

[Figure 30]

YuE

62 75 20 85 70

80

Udio

46 25 18 80 59

60

Suno

88 80 82 1e+02 91

40

Hailuo

15 15 20 0 25

20

Tiangong

39 30 41 9.1 75

0

Overall

YuE

Udio

SunoV4

Hailuo

Tiangong

(b) Chinese - Musicality

100

[Figure 31]

YuE

55 30 80 20 90

80

Udio

62 70 60 20 100

60

Suno V4

50 20 40 80 60

40

Hailuo

60 80 80 20 60

20

Tiangong

22 10 0 40 40

0

Overall

YuE

Udio

SunoV4

Hailuo

Tiangong

(d) Korean - Musicality

90

[Figure 32]

YuE

52 40 10 80 80

80

70

Udio

51 60 25 70 50

60

Suno

80 90 75 75 80

50

40

Hailuo

31 20 30 25 50

30

Tiangong

35 20 50 20 50

20

10

Overall

YuE

Udio

SunoV4

Hailuo

Tiangong

(f) Japanese - Musicality

Figure 17. YuE vs. others across different languages on lyrics following and musicality.

###### D. 15 English Prompts From GPT

###### ID: 1

[Genre] Rap [verse] Woke up in the morning, sun is shining bright Chasing all my dreams, gotta get my mind right City lights are fading, but my vision’s clear Got my team beside me, no room for fear Walking through the streets, beats inside my head Every step I take, closer to the bread People passing by, they don’t understand Building up my future with my own two hands [chorus] This is my life, and I’m aiming for the top Never gonna quit, no, I’m never gonna stop Through the highs and lows, I’mma keep it real Living out my dreams with this mic and a deal [verse] Late nights grinding, writing down these rhymes Clock is ticking fast, can’t afford to waste time Haters gonna hate, but I brush it off Turn the negativity into something strong Mama working hard, wanna make her proud Echoes of her prayers cutting through the crowd Friends turned strangers, but it’s all good Focused on my path like I always knew I would [chorus] This is my journey, and I’m running this race Heart full of fire, you can see it in my face Obstacles ahead, but I got no fear Victory is close, yeah, it’s almost here [bridge] They said I couldn’t do it, said I’d never rise But now I’m soaring high, reaching for the skies Lessons that I learned made me who I am Standing tall now, I don’t give a damn [verse] Echoes in the alley, music’s getting loud Feeling the adrenaline pumping through the crowd Spotlights on me, it’s my time to shine Living in the moment, everything’s aligned Looking back now at the roads I’ve crossed Every single battle, every line I’ve tossed Made me stronger, wiser, ready for what’s next Writing my own story, turning pages of the text [chorus] This is my song, and I’m singing it proud Voices united, hear us shout out loud From the underground straight into the stars Carving out my name, leaving all these scars [outro] Yeah, this is for the dreamers, the ones who never quit Keep your head up high, and don’t you ever submit Life is what you make it, so make it something great Step into your purpose, go and seize your fate

[Genre] Rock [verse] Standing on the corner, shadows in the night The city’s heartbeat echoes under lights Hands deep in pockets, wandering alone Footsteps tracing paths to the unknown Suddenly he pauses, looks up to the sky Eyes reflect the questions passing by Whispers to the wind, words without a sound Searching for the answers yet unfound [chorus] Lost within the chaos, seeking out a sign In a world of color, drawing blurred lines Moving forward, looking back, unsure of the way Trying to find a place where he can stay [verse] He crosses empty streets, under neon glow Faces in the crowd, stories left untold Raises up his arms, reaching for the truth Grasping at the fragments of his youth Billboards and the banners flutter in the breeze The rhythm of the city brings him to his knees Heartbeat heavy, nowhere left to hide Feeling like he’s lost amidst the tide [chorus] Lost within the chaos, seeking out a sign In a world of color, drawing blurred lines Moving forward, looking back, unsure of the way Trying to find a place where he can stay [bridge] Doesn’t want to leave, doesn’t want to fight Caught between the darkness and the light No need for reason, nothing to prove Just a soul in transit, with nothing to lose [outro] Doesn’t want to leave, doesn’t want to fight Chasing after shadows in the night He doesn’t need the truth, doesn’t need a name Just looking for a spark to fan the flame

[Genre] Pop [verse] Staring at the sunset, colors paint the sky Thoughts of you keep swirling, can’t deny I know I let you down, I made mistakes But I’m here to mend the heart I didn’t break [chorus] Every road you take, I’ll be one step behind Every dream you chase, I’m reaching for the light You can’t fight this feeling now I won’t back down I’m the whisper in the wind, the shadow by your side The warmth you feel within when you can’t hide You know you can’t deny it now I won’t back down [verse] They might say I’m foolish, chasing after you But they don’t feel this love the way we do My heart beats only for you, can’t you see? I won’t let you slip away from me [chorus] Every road you take, I’ll be one step behind Every dream you chase, I’m reaching for the light You can’t fight this feeling now I won’t back down I’m the whisper in the wind, the shadow by your side The warmth you feel within when you can’t hide You know you can’t deny it now I won’t back down [bridge] No, I won’t back down, won’t turn around Until you’re back where you belong I’ll cross the oceans wide, stand by your side Together we are strong [outro] Every road you take, I’ll be one step behind Every dream you chase, love’s the tie that binds You can’t fight this feeling now I won’t back down

[Genre] Jazz [verse] In the quiet of the evening, shadows start to fall Whispers of the night wind echo through the hall Lost within the silence, I hear your gentle voice Guiding me back homeward, making my heart rejoice [chorus] Don’t let this moment fade, hold me close tonight With you here beside me, everything’s alright Can’t imagine life alone, don’t want to let you go Stay with me forever, let our love just flow [verse] Moonlight paints a picture upon your lovely face Every glance between us fills the empty space Time stands still around us when you’re in my arms Nothing else can matter, safe from any harm [chorus] Don’t let this moment fade, hold me close tonight With you here beside me, everything’s alright Can’t imagine life alone, don’t want to let you go Stay with me forever, let our love just flow [bridge] Every touch ignites a fire, burning deep within Every smile you give to me makes my head spin Promise me you’ll stay awhile, don’t ever say goodbye Together we’ll chase every star across the sky [chorus] Don’t let this moment fade, hold me close tonight With you here beside me, everything’s alright Can’t imagine life alone, don’t want to let you go Stay with me forever, let our love just flow [outro] Stay with me forever, let our love just flow

[Genre] Blues [verse] Late last night, the rain was pouring down Lonely footsteps echoed through the town Thinking ’bout the love that slipped away Wondering how I let you go that day [chorus] Oh, my angel, where have you flown Left me here to face this world alone I’m just a fool, a fool in love with you Can’t deny this heartache’s true [verse] Streetlights flicker, shadows on the wall Memories of you, I recall Your laughter like a song inside my head Without you here, my soul feels dead [chorus] Oh, my angel, won’t you return In this fire of love, I still burn I’m just a fool, a fool in love with you Hoping someday you’ll feel it too [bridge] I fell for you, and I always knew That my world revolves around you I hope and I pray, both night and day That you’ll come back and choose to stay [chorus] Oh, my angel, where have you flown Left me here to face this world alone I’m just a fool, a fool in love with you Waiting here, what else can I do [outro] I’m just a fool, a fool in love with you

[Genre] RnB_Soul [verse] Why don’t we just find a place to hide Leave all our worries and doubts behind When nothing in this world is as it seems Together we can live inside our dreams There’s no need to be afraid tonight In the love we’ve made, we’ll find the light When we’re living in a world of our own It’s you and me, we never feel alone [chorus] They say it’s hard for a man to let it show But with you, I’m ready to let it all go Whatever we try, we’re gonna get there You take control, baby, I don’t care I gotta keep on pushing when times get tough We keep on making better love [verse] Don’t believe the things that others say We’ve tried it all and found our way They should take a look at you and me Learning how to love and set it free For every heartache, we take our time You teach me yours and I’ll show you mine About the way that love is meant to be Together we’ll rewrite our history [chorus] They say it’s hard for a man to let it show But with you, I’m ready to let it all go Whatever we try, we’re gonna get there You take control, baby, I don’t care I gotta keep on pushing when times get tough We keep on making better love [bridge] Gotta take control and swallow my pride Every man has feelings deep inside You gotta find yourself before you can Be ready to love and understand Baby, I know what you’re thinking of We keep on making better love [outro] I believe the love we’re making’s gonna last forevermore Loving you feels so right, like never before We’ll be getting down tonight until the morning light We keep on making better love Better love, we’ll be making Better love, no more faking They say it’s hard for a man to let it show But with you here, I’m ready to let go Whatever we try, we’re gonna get there You take control, baby, I don’t care I gotta keep on pushing when times get tough We keep on making better love Better love (till fade out)

[Genre] Ancient_Chinese_Style [verse] Beneath the moonlit sky so vast A lone wanderer recalls the past Whispers of the bamboo leaves Echo tales the wind retrieves [chorus] Oh, the rivers flow, mountains high Journeying souls beneath the endless sky Threads of fate entwine our way Guiding us through night and day [verse] Lanterns glow with softest light Painting shadows in the night Silken robes and ancient songs Memories where hearts belong [chorus] Oh, the rivers flow, mountains high Journeying souls beneath the endless sky Threads of fate entwine our way Guiding us through night and day [bridge] Stars reflect in tranquil ponds Dreams unfold of times beyond Lotus blooms and cranes in flight Secrets held within the night [outro] As the sunrise paints the east Bringing hope and inner peace Footprints fade upon the shore But the spirit journeys evermore

[Genre] Folk [verse] Underneath the open sky so clear, We gather ’round with voices near. Through trials faced and stories told, Our spirits rise, our hearts unfold. [chorus] So lift the lanterns to the sky, Together we will soar and fly. Though shadows loom and doubts appear, We’ll keep the flame forever here. [verse] Remember all the paths we’ve crossed, The battles won, the moments lost. A banner of hope we hold up high, A symbol shining in our eyes. [chorus] So lift the lanterns to the sky, Together we will soar and fly. Though shadows loom and doubts appear, We’ll keep the flame forever here. [bridge] With hands united, we stand tall, Pledged to rise if we should fall. Through darkest nights and stormy seas, Our song will carry on the breeze. [chorus] So lift the lanterns to the sky, Together we will soar and fly. Though shadows loom and doubts appear, We’ll keep the flame forever here. [outro] We’ll keep the flame forever here.

[Genre] Dance [verse] Underneath the starlit sky, We come alive, you and I. City lights are shining bright, Dancing through the endless night. [chorus] Who are we? Let’s break away, Feel the beat and let it play. Lost in music, hearts align, In this moment, we define. [verse] Shadows fade beneath the glow, Rhythms guide us where to go. Voices whisper in the crowd, Turn it up, we’ll sing aloud. [chorus] Who are we? Let’s break away, Feel the beat and let it play. Lost in music, hearts align, In this moment, we define. [bridge] Let the melody surround, Lift us off the solid ground. Every step and every move, In this dance we find our groove. [chorus] Who are we? Let’s break away, Feel the beat and let it play. Lost in music, hearts align, In this moment, we define. [outro] Keep on dancing, feel the heat, Moving to the pounding beat. Who we are is here and now, Take my hand, we’ll show them how.

[Genre] Country [verse] Da-dum, da-da-da-da-da-da-da Da-dum, da-dum Walking down this lonesome road Thinking ’bout the love untold Why haven’t I told you I’ve whispered to the midnight stars Just how wonderful you are Why haven’t I told you [chorus] Friends keep asking if I’m fine I just smile and say you’re mine Might as well confess Can’t keep this inside Maybe you feel the same way too Oh darling, if you do Why haven’t you told me Da-dum, da-da-da-da-da-da-da [verse] I’ve sung it to the morning sun That with you, my life’s begun Why haven’t I told you My heart’s an open book today Waiting for the words to say Why haven’t I told you [chorus] Friends keep asking what’s the news I just grin and think of you Time to take a chance Let my feelings show Maybe you feel the same way too Oh darling, if you do Why haven’t you told me [bridge] Da-dum, da-da-da-da-da-da-da Da-dum, da-dum No more holding back these words Let them fly just like the birds [chorus] Now I’m standing here tonight Hoping that I got it right Might as well confess Can’t keep this inside Maybe you feel the same way too Oh darling, if you do Let’s not waste another day Why haven’t we told us [outro] Da-dum, da-da-da-da-da-da-da Da-dum, da-dum Now we’ve finally told us Our new life’s begun

[Genre] Rap [verse] Woke up in the morning, sun is shining bright Chasing all my dreams, gotta get my mind right City lights are fading, but my vision’s clear Got my team beside me, no room for fear Walking through the streets, beats inside my head Every step I take, closer to the bread People passing by, they don’t understand Building up my future with my own two hands [chorus] This is my life, and I’m aiming for the top Never gonna quit, no, I’m never gonna stop Through the highs and lows, I’mma keep it real Living out my dreams with this mic and a deal [verse] Late nights grinding, writing down these rhymes Clock is ticking fast, can’t afford to waste time Haters gonna hate, but I brush it off Turn the negativity into something strong Mama working hard, wanna make her proud Echoes of her prayers cutting through the crowd Friends turned strangers, but it’s all good Focused on my path like I always knew I would [chorus] This is my journey, and I’m running this race Heart full of fire, you can see it in my face Obstacles ahead, but I got no fear Victory is close, yeah, it’s almost here [bridge] They said I couldn’t do it, said I’d never rise But now I’m soaring high, reaching for the skies Lessons that I learned made me who I am Standing tall now, I don’t give a damn [verse] Echoes in the alley, music’s getting loud Feeling the adrenaline pumping through the crowd Spotlights on me, it’s my time to shine Living in the moment, everything’s aligned Looking back now at the roads I’ve crossed Every single battle, every line I’ve tossed Made me stronger, wiser, ready for what’s next Writing my own story, turning pages of the text [chorus] This is my song, and I’m singing it proud Voices united, hear us shout out loud From the underground straight into the stars Carving out my name, leaving all these scars [outro] Yeah, this is for the dreamers, the ones who never quit Keep your head up high, and don’t you ever submit Life is what you make it, so make it something great Step into your purpose, go and seize your fate

[Genre] Rock [verse] Standing on the corner, shadows in the night The city’s heartbeat echoes under lights Hands deep in pockets, wandering alone Footsteps tracing paths to the unknown Suddenly he pauses, looks up to the sky Eyes reflect the questions passing by Whispers to the wind, words without a sound Searching for the answers yet unfound [chorus] Lost within the chaos, seeking out a sign In a world of color, drawing blurred lines Moving forward, looking back, unsure of the way Trying to find a place where he can stay [verse] He crosses empty streets, under neon glow Faces in the crowd, stories left untold Raises up his arms, reaching for the truth Grasping at the fragments of his youth Billboards and the banners flutter in the breeze The rhythm of the city brings him to his knees Heartbeat heavy, nowhere left to hide Feeling like he’s lost amidst the tide [chorus] Lost within the chaos, seeking out a sign In a world of color, drawing blurred lines Moving forward, looking back, unsure of the way Trying to find a place where he can stay [bridge] Doesn’t want to leave, doesn’t want to fight Caught between the darkness and the light No need for reason, nothing to prove Just a soul in transit, with nothing to lose [outro] Doesn’t want to leave, doesn’t want to fight Chasing after shadows in the night He doesn’t need the truth, doesn’t need a name Just looking for a spark to fan the flame

[Genre] Pop [verse] Staring at the sunset, colors paint the sky Thoughts of you keep swirling, can’t deny I know I let you down, I made mistakes But I’m here to mend the heart I didn’t break [chorus] Every road you take, I’ll be one step behind Every dream you chase, I’m reaching for the light You can’t fight this feeling now I won’t back down I’m the whisper in the wind, the shadow by your side The warmth you feel within when you can’t hide You know you can’t deny it now I won’t back down [verse] They might say I’m foolish, chasing after you But they don’t feel this love the way we do My heart beats only for you, can’t you see? I won’t let you slip away from me [chorus] Every road you take, I’ll be one step behind Every dream you chase, I’m reaching for the light You can’t fight this feeling now I won’t back down I’m the whisper in the wind, the shadow by your side The warmth you feel within when you can’t hide You know you can’t deny it now I won’t back down [bridge] No, I won’t back down, won’t turn around Until you’re back where you belong I’ll cross the oceans wide, stand by your side Together we are strong [outro] Every road you take, I’ll be one step behind Every dream you chase, love’s the tie that binds You can’t fight this feeling now I won’t back down

[Genre] Jazz [verse] In the quiet of the evening, shadows start to fall Whispers of the night wind echo through the hall Lost within the silence, I hear your gentle voice Guiding me back homeward, making my heart rejoice [chorus] Don’t let this moment fade, hold me close tonight With you here beside me, everything’s alright Can’t imagine life alone, don’t want to let you go Stay with me forever, let our love just flow [verse] Moonlight paints a picture upon your lovely face Every glance between us fills the empty space Time stands still around us when you’re in my arms Nothing else can matter, safe from any harm [chorus] Don’t let this moment fade, hold me close tonight With you here beside me, everything’s alright Can’t imagine life alone, don’t want to let you go Stay with me forever, let our love just flow [bridge] Every touch ignites a fire, burning deep within Every smile you give to me makes my head spin Promise me you’ll stay awhile, don’t ever say goodbye Together we’ll chase every star across the sky [chorus] Don’t let this moment fade, hold me close tonight With you here beside me, everything’s alright Can’t imagine life alone, don’t want to let you go Stay with me forever, let our love just flow [outro] Stay with me forever, let our love just flow

[Genre] Blues [verse] Late last night, the rain was pouring down Lonely footsteps echoed through the town Thinking ’bout the love that slipped away Wondering how I let you go that day [chorus] Oh, my angel, where have you flown Left me here to face this world alone I’m just a fool, a fool in love with you Can’t deny this heartache’s true [verse] Streetlights flicker, shadows on the wall Memories of you, I recall Your laughter like a song inside my head Without you here, my soul feels dead [chorus] Oh, my angel, won’t you return In this fire of love, I still burn I’m just a fool, a fool in love with you Hoping someday you’ll feel it too [bridge] I fell for you, and I always knew That my world revolves around you I hope and I pray, both night and day That you’ll come back and choose to stay [chorus] Oh, my angel, where have you flown Left me here to face this world alone I’m just a fool, a fool in love with you Waiting here, what else can I do [outro] I’m just a fool, a fool in love with you

