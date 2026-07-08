arXiv:2506.02295v1[cs.CV]2Jun2025

# QARI-OCR: High-Fidelity Arabic Text Recognition through Multimodal Large Language Model Adaptation

Ahmed Wasfy1,2† Omer Nacar3† Abdelakreem Elkhateb1 Mahmoud Reda1 Omar Elshehy1 Adel Ammar3 Wadii Boulila3 1NAMAA 2KAND CA Corp. 3Prince Sultan University Emails: onajar@psu.edu.sa, ahmed.wasfy@kand.ca †Corresponding authors

## Abstract

The inherent complexities of Arabic script—its cursive nature, diacritical marks (tashkeel), and varied typography—pose persistent challenges for Optical Character Recognition (OCR). We present Qari-OCR, a series of vision-language models derived from Qwen2-VL-2B-Instruct, progressively optimized for Arabic through iterative fine-tuning on specialized synthetic datasets. Our leading model, QARI v0.2, establishes a new open-source state-of-the-art with a Word Error Rate (WER) of 0.160, Character Error Rate (CER) of 0.061, and BLEU score of 0.737 on diacritically-rich texts. Qari-OCR demonstrates superior handling of tashkeel, diverse fonts, and document layouts, alongside impressive performance on low-resolution images. Further explorations (QARI v0.3) showcase strong potential for structural document understanding and handwritten text. This work delivers a marked improvement in Arabic OCR accuracy and efficiency, with all models and datasets released to foster further research.

## 1 Introduction

Digital text accessibility is central to information preservation, dissemination, and analysis in today’s data-driven world. While Optical Character Recognition (OCR) technology has made significant progress for Latin-based scripts, complex writing systems like Arabic continue to present substantial challenges. Arabic script, with its cursive nature, contextual character forms, diverse diacritical marks (tashkeel), and varied typographic styles, poses unique difficulties for conventional OCR approaches (Al-Sheikh et al., 2020).

Arabic is spoken by over 420 million people worldwide, making accurate Arabic OCR vital for cultural preservation, scholarly research, and information access (UNESCO, 2024). Despite this importance, existing Arabic OCR solutions often underperform compared to their Latin-script counterparts, with particularly poor handling of diacrit-

ical marks that significantly affect pronunciation and meaning. (Alwajih et al., 2024)

This paper introduces Qari-OCR, a fine-tuned vision-language model based on Qwen2-VL-2BInstruct, specifically optimized for Arabic text recognition. Qari-OCR was developed through an iterative process, with each version leveraging progressively enhanced synthetic datasets to improve performance on specific aspects of Arabic text, detailed in Table 1. Our approach utilizes recent multimodal learning advances for superior Arabic OCR performance with computational efficiency.

Our key contributions include:

State-of-the-Art Model: We trained three different QARI-OCR Models,that significantly outperforms leading open-source solutions for full-page text recognition and different layout complexities.

### Comprehensive Diacritical, and Script Support:

Qari-OCR models demonstrate comprehensive support for Arabic diacritical marks (tashkeel), including fathah, kasrah, dammah, sukun, shadda, and tanwin forms.

Evaluation & Release: We publicly release all trained models alongside their corresponding evaluation datasets and a standardized evaluation framework to enable reproducible research and facilitate downstream applications. For review, see huggingface repository.

The remainder of this paper is organized as follows: Section 2 reviews related work in Arabic OCR. Section 3 details our dataset generation pipeline and model training. Section 4 presents the experimental setup and results. Section 6 outlines the limitations of our models, and Section 7 concludes the paper and suggests future work.

Table 1: Key Characteristics and Objectives of Qari-OCR Models Versions.

Model Ver. Key Features/Focus

Objective/Tested Capability

Training Dataset Size

HTML? Diacritics? Layout

Handwritten Support?

Complexity?

- Qari-OCR v0.1 Clean, no diacritics, 5 fonts, uniform min. size/layout.

Baseline on legible, low-noise data.

5,000 ✗ ✗ ✗ ✗

- Qari-OCR v0.2 Diacritics, broader typography (10 fonts), linguistic complexity.

Recognition of diacriticrich/classical text.

50,000 ✗ ✔ ✗ ✗

- Qari-OCR v0.3 Multi-font sizes/page (headers, body), realistic layouts.

10,000 ✔ ✔ ✔ ✔

Spatial parsing for mixed-size, complex layouts.

## 2 Related Work

The journey of Optical Character Recognition (OCR) has been marked by a significant evolution from early, rule-heavy systems to sophisticated deep learning paradigms, each step bringing new capabilities and addressing longstanding challenges, particularly for complex scripts like Arabic. Initial OCR approaches often involved a structured pipeline of preprocessing, explicit character segmentation, and traditional classification techniques (Jasim, 2020; Graves and Schmidhuber, 2008). However, these methods encountered substantial difficulties with the cursive, contextsensitive nature of Arabic script (Alrobah and Albahli, 2022).

The advent of deep learning, particularly Convolutional Neural Networks (CNNs) combined with Recurrent Neural Networks (RNNs), as exemplified by the CRNN model (Puigcerver, 2017), revolutionized the field. Such architectures enabled end-to-end learning, implicitly handling segmentation and significantly improving performance on general text. Transformer models later pushed the boundaries further with their powerful attention mechanisms, enhancing sequence modeling for OCR tasks, leading to models like TROCR (Li et al., 2023).

Developing effective OCR for Arabic necessitates a keen focus on its unique orthographic characteristics: right-to-left text flow, intricate ligatures, contextually varying character shapes, and the varied positional placements of diacritical marks in Arabic. While deep learning advancements provided a robust foundation, specialized efforts were made to adapt these for Arabic. This included training on Arabic-specific datasets and tailoring

architectures, as seen in some deep learning applications for Arabic OCR (Yousef et al., 2020). More recently, dedicated models like Qalam (Bhatia et al., 2024), a multimodal system built on a Swinv0.2 encoder and RoBERTa decoder, have been developed specifically for Arabic OCR and handwriting recognition.

The latest wave of innovation involves Multimodal Large Language Models (MLLMs), which aim to unify vision and language understanding for a wide array of tasks. These models can perform OCR as one of their many capabilities. The Qwen2VL series (Wang et al., 2024) represents a significant advancement in general-purpose MLLMs, incorporating features like dynamic image resolution processing and effective multimodal fusion, leading to competitive performance on broad multimodal benchmarks. However, the inherent design of such generalist MLLMs does not typically optimize them for the high-fidelity, nuanced requirements of Arabic OCR. While MLLMs like AIN (Heakl et al., 2025) are emerging, trained with substantial authentic Arabic data to address various multimodal tasks including OCR, the specific challenges of detailed text recognition in Arabic images, especially concerning diacritics and diverse typography, continue to be an area noted for further improvement.

Our work, Qari-OCR, is positioned within this evolving landscape. We leverage the sophisticated architecture of a general-purpose MLLM, specifically Qwen2-VL-2B-Instruct (Wang et al., 2024), as a foundational model. Through specialized finetuning on our synthetic Arabic text datasets, we adapt this MLLM to function effectively as an OCR system. By combining targeted dataset cura-

Table 2: Evolution of OCR Approaches and Key Characteristics Relevant to Arabic.

Multimodal Foundation

Primary Focus / Example

OCR Approach Category

End-to-End Learning

Arabic Diacritic Handling

Arabic Font/Style Diversity

Traditional OCR ✗ ✗ ✗ ✗ Segmented classification; Early systems.

✔ Ltd. Ltd. ✗ General text OCR; CRNN (Puigcerver, 2017).

Early Deep Learning OCR (CNN-RNN-CTC)

✔ Ltd. Ltd. ✗ Sequence modeling for text; TROCR (Li et al., 2023).

Transformer-based OCR

✔ Ltd. ✔ ✗ Targeted Arabic data; (Yousef et al., 2020).

Early Arabic-Specific DL OCR

Specialized Arabic Foundation Models (e.g., Qalam)

✔ ✔ ✔ ✔ Deep Arabic script recognition; Qalam (Bhatia et al., 2024).

✔ Ltd. Ltd. ✔ Broad vision-lang. tasks; Qwen2-VL (Wang et al., 2024).

General MLLMs (e.g., Qwen2-VL)

Arabic-Inclusive MLLMs (e.g., AIN)

✔ ✔ ✔ ✔ Broader Arabic multimodal; AIN (Heakl et al., 2025).

Qari-OCR (Our Work)

✔ ✔ ✔ ✔ Specialized Arabic

OCR via MLLM.

tion and parameter-efficient adaptation, Qari-OCR addresses accurate diacritic recognition and varied font styles, advancing high-fidelity Arabic text recognition. The evolution and comparative characteristics of these OCR approaches are summarized in Table 2.

## 3 Methodology

The development of Qari-OCR was implemented through a two-stage methodological framework: firstly, the generation of diverse synthetic datasets engineered to encapsulate the complexities of Arabic script; and secondly, the iterative fine-tuning of an advanced vision-language model using these specialized datasets. An illustrative overview of this workflow is presented in Figure 1.

### 3.1 Synthetic Dataset Generation for QARI

To bridge gaps in existing Arabic OCR corpora—namely diacritic coverage, font diversity, and realistic layouts—we devised a three-stage synthetic data pipeline. Two complementary text sources were used: a modern news article collection and a classical Islamic corpus (rich in tashk¯ıl). The text was rendered programmatically in HTML using twelve distinct Arabic fonts (from common Naskh to ornate calligraphic styles) at sizes varying between 14 px and 100 px, then converted to PDF

via WeasyPrint 1 and to images via pdf2image 2.

- • Dataset v0.1: Non-diacritized text, a limited font set, and uniform minimal size establish a high-legibility baseline.
- • Dataset v0.2: The dataset v0.2 introduces full diacritics and expands the font repertoire to enhance the recognition of vocalized and classical texts.
- • Dataset v0.3: Introduces mixed font sizes on each page to simulate realistic document structures (headers, body, annotations) and HTML spatial/layout parsing.

Finally, each image undergoes one of three synthetic degradation treatments—Clean, Moderately Degraded (subtle noise, color shifts, mild blur), or Heavily Degraded (textured backgrounds, aggressive blur)—with all variants paired to their groundtruth transcription. This progression yields a robust, multi-faceted Arabic OCR dataset suitable for training and evaluating QARI across increasing levels of linguistic, typographic, and visual complexity.

3.2 Model Architecture and Training Strategy We built Qari-OCR on the Qwen2-VL-2B-Instruct backbone (Wang et al., 2024), leveraging its Naive

- 1https://weasyprint.org
- 2https://pdf2image.readthedocs.io/en/latest/index.html

Synthetic Dataset Generation Pipeline Model Finetuning Pipeline

[Figure 1]

Arabic Text Corpora Arabic Text Corpora (News, Classical) (Arabic Font Set (12 Fonts, 14-100px)

##### Unsloth Framework ( FastVisionModel) Setup ( 4-bit QLORA, PEFT: Lora Adapters)

HTML Templating & Styling

Data Formatting Conversational User [img + Insta] , Assistant([text])

WeasyPrint(HTML > PDF)

Iterative Fine-tuning TRL SFTTrainer, 1 Epoch, AdamW)

pdf2image ( PDF > Image)

Document Degradation

Qari V0.2 Dataset (v0.2)

Qari V0.3 Dataset (v0.3)

Qari V0.1 Dataset (v0.1)

Clean Moderate Heavy

Fine-Tuned QARI OCR Models ( V0.1, V0.2, V0.3)

Synthetic Arabic OCR Dataset ( V0.1, V0.2, V0.3) + ground truth

Figure 1: Qari-OCR Dataset Generation and Model Training Pipeline

Dynamic Resolution for adaptive image scaling and M-RoPE for robust cross-modal positional embeddings. To optimize fine-tuning efficiency, we optionally quantized the model to 4-bit and inserted LoRA adapters (rank = 16) into both vision and language modules.

Training data comprised conversationally formatted image–text pairs, where each “user” message carried an image and prompt, and the “assistant” reply provided the ground-truth Arabic transcription. We conducted three matched fine-tuning runs, each on a different synthetic dataset version, as summarized in 1.

All models were fine-tuned for a single epoch using the Unsloth library. footnotehttps://github. com/unslothai/unsloth with the AdamW optimizer (Loshchilov and Hutter, 2017) and with learning_rate equal to 2e-4 and weight_decay

of 0.01 with linear lr_scheduler. Input images were resized and normalized to Qwen2VL specifications, and training was orchestrated with Hugging Face’sSFTTrainer3 using the UnslothVisionDataCollator, a per-device batch size of 2, and 4 gradient-accumulation steps (effective batch size = 8). All experiments ran on a single NVIDIA A6000 GPU (48 GB VRAM).

3https://huggingface.co/docs/trl/en/sft_ trainer

## 4 Experimental Results

This section presents the experimental setup, evaluation metrics, and empirical results used to benchmark Qari-OCR and selected baselines on challenging Arabic text.

### 4.1 Experimental Setup

We assembled a test set of 200 scanned pages from traditional Arabic print—complete with diacritics, complex ligatures, and dense layouts—to mirror the challenges of historical and scholarly texts. Every image underwent the same generic preprocessing (no language-specific tweaks or manual annotations), ensuring a fair evaluation of each OCR system’s raw performance.

Our benchmark suite comprises Qari-OCR and six different OCR systems spanning from classical engines to cutting-edge vision–language models: Tesseract OCR (Smith, 2007), EasyOCR (Pattanayak et al., 2023), Mistral OCR (Mistral AI Team, 2025), AIN (Heakl et al., 2025), Qwen 2.5-7B Instruct (Wang et al., 2024), and Qwen 27B (Wang et al., 2024).

### 4.2 Evaluation Metrics

To assess OCR performance on Arabic text, we employed three complementary metrics: Character Error Rate (CER), Word Error Rate (WER) (Klakow and Peters, 2002), and the BLEU score (Papineni et al., 2002). These metrics provide a holistic view, capturing errors from fine-grained character inac-

|[Figure 2]<br><br>Diacritic<br><br>Diacritic<br><br>ligature<br><br>ligature<br><br>ligature<br><br>Classical Language and Poetic Meter<br><br>different forms of writing Hamza<br><br>dots<br><br>| | | |
|---|---|---|
| | | |
<br><br>Different contextual shapes of same letter<br><br>Punctuation within the text<br><br>kashīda<br><br>Maddah<br><br>| |
|---|---|
| | |

ﻪﺑ ﺪﺼﻗ ﺮﻴﺴﻳ اﺬﻛو ، ﻞﻳﻮﻄﻟا ﺪﻤﻌﻟا تﻮﻜﺴﻟا ﺎﻬِﻌِﻄَ ﻘْ َﻳوَ ﻲِﻠ ﱟ ﺻْ َأ' : output ﻪﻨِﻣْﺄﻤْ ﺘﻛَ ﺎﻬﺑ ﻖﻠﻌﺘﺗ نﺈﻓ ،

ﻖﻠﻌﺘﻳ ﻻ ﻲﺒِﻨﺟْ َأ ﺮﻛذ ﻞﻠﺨﺗو ةءاﺮﻘﻟا ﻊﻄﻗ ُ ، ةوﻼﺘﻟ دﻮﺠﺳو ، ﺎﻬﻴﻓ ﻒﻗﻮﺗ اذإَ يأ ﻪﻴﻠﻋ ﻪﺤﺘﻓو ، ﻪﻣﺎﻣإ

|ةﻼﺼﻟﺎﺑ|
|---|

|ةءاﺮﻘﻟ|
|---|

|لاﺆﺳو|
|---|

ﺾﻌﺑ ﻲﻓ ﻪﻟﻮﻗ ﻰﻨﻌﻣَ اﺬﻫو ، ﻼﻓ ﺎﻤﻬِﺘِ َﻳآ

باﺬﻋ ﻦﻣ ةذﺎﻌﺘﺳاو ، ﺔﻤﺣر ﺪﻌﺑ َلﺎﻗوَ َﻼﻜْ َﺗ ﻪ ُ ﻣُ ْﺎﻣإﱠ ﺎﻤِﻟَ ِﻪِﻟاﺆَ ﺳُ اَﺮﺜُ ﻛَ نا ْ ﺖ ْ ﻌَ ﻄَ ﻘَ ﻧاْ تﻮ ِ ُﻜﺴﻟﺎﱡ ِﺑوَ ـﻫ21 : ﺦﺴﻨﻟا ﻻ ُﺮﻛْ ﺬﻟاﱠ ﻢ ﱠ ُﺛ ﻖﻳ ِ ِﺮﻔْ ﺘﻟاﱠ ﻦِﻣ َ ﱃ َ وْ َأ ﻻ ﻻﻮِ ﻟاْ وَ ﻊ ٌ ﱢﺒﺳَ تﺎ ِ َﻳﻵا ﻦِﻣ َ ﻢ ﱠ ُﺛ ـﻫ21 : ﻚﻟذ

|ةءاﺮﻘﻟ|
|---|

وأ ﻢﻠﻌﻣ مﺪﻌﻟ ﺔﺤﺗﺎﻔﻟا ﻦﺴﺤﻳ ﻢﻟ اذإَ (ﻢﺛ)ﻒْ ﻛَ َلﺎﻨَ َﺗ ن ْ َﺄِﺑ ﻊ ْ ﻛَ راْ وَ ﺎَﻫِرْﺪِﻘِﺑ ﺎﻬﻨﺴﺣَأ نإ (ﻊﺒﺳ) ﺎﻫﺮﻴﻏ

QARI OCR Model

ِ َ ﻪﻴﻠﻋ ﺐﺟو ﻚﻟذ ﻮﺤﻧ وأ ﻒﺤﺼﻣ (ﻖﻳِ ِﺮﻔْ ﺘﻟاﱠ ﻦﻣ ﱃ َ وْ َأ ﻻﻮﻟاِ ) ﻦﻜﻟ (و) ﺔﻗﺮﻔﺘﻣ و ْ َأ ﺔﻠﺻاﻮﺘﻣ ﺔﺴﻠﻤﺒﻟﺎﺑ

|(تﺎﻳﻵا ﻦﻣ)|
|---|

َ دﺪﻋ ﻮﻫ اﺬﻫو نﺎﻀﻣر ﺮﻬﺷ ءﺎﻀﻗ ﻲﻓ ﺎﻤﻛ

|ﺎﻬﺗﺎﻳآ|
|---|

ﺔﺤﺗﺎﻔﻟﺎﺑ ﻪﺒﺷَأ ﻪﱠﻧﻷ نإ ﻰﻗﺎﺒﻟا لﺪﺒﻳو ﻪﺑ ﻲﺗﺄﻳ ﺔﺤﺗﺎﻔﻟا ﺾﻌﺑ ﻦﺴﺤﻳ ﻦﻣو . يﻮِ ﻨﻟاﱡ ﺪﻨﻋ

ِ ْ ﱠ رﺎﺟو َ ،

|ﻖﻳﺮﻔﺘﻟا|
|---|

| |
|---|

ْ َ ﺎﻬﻟﺪﺑ ﺾﻌﺑ ﻦﺴﺤﻳ ﻦﻣ اﺬﻛو

|ﺢﺻﻷا|
|---|

يﻮِ ﻨﻟاﱡ ﺪﻨﻋ ﺢﺻْ َﻷا ﻲﻓ هﺮﺴﻛ ﻻإوِ ، ﻦﺴﺣَأ ً

|.|
|---|

ﻦﺴﺤﻳ ﻢﻟ اذإَ (ﻢﺛ) . لﺪﺒﻟاو ﻞﺻْ َﻷا ﻦﻴﺑ ﺐﻴﺗﺮﺘﻟا ﺐﺠﻳو ،

|ﺎﺌﻴﺷ|
|---|

ﻦﻣ 'n\نْ َﺄﺑ ﻪﻟﺪﺑ (ﺮﻛْ ﺬﻟاﱠ ) ﻪﻴﻠﻋ ﺐﺟو

|نآﺮﻘﻟا|
|---|

ﻦﻣ

|نآﺮﻘﻟا|
|---|

- Figure 2: Qualitative example demonstrating Qari-OCR’s handling of various Arabic script complexities. The input image (left, with annotations highlighting features like diacritics, ligatures, contextual shapes, etc.) is processed by the Qari-OCR model, producing the transcribed text output (right).

Table 3: Comparative Performance of OCR Models on the Arabic Test Set. Lower CER/WER and higher BLEU indicate better performance.

Model CER ↓ WER ↓ BLEU ↑ Tesseract OCR 0.436 0.889 0.108 EasyOCR 0.791 0.918 0.051 Mistral OCR (API-based) 0.210 0.440 0.570 AIN 0.640 0.830 0.210 Qwen 2.5-7B Instruct 0.550 0.800 0.220 Qwen 2-7B 0.740 1.050 0.160

- QARI v0.1 (Ours) 1.915 2.025 0.221
- QARI v0.2 (Ours) 0.061 0.160 0.737
- QARI v0.3 (Ours) 0.300 0.485 0.545

curacies to broader linguistic and structural deviations.

CER measures the normalized Levenshtein distance at the character level between the predicted and ground-truth text. It is particularly critical for Arabic OCR due to its sensitivity to errors in diacritics and morphologically complex character sequences, which significantly impact meaning. WER, operating similarly at the word level, reflects how recognition errors affect sentence structure and is useful for identifying tokenization or wordlevel mistakes common in processing Arabic script. Finally, the BLEU score, by assessing n-gram overlap, offers insights into phrase-level fidelity and the preservation of fluent text structure, which is valuable for evaluating the overall readability and coherence of the OCR output.

### 4.3 Results

The comparative performance of our Qari-OCR model versions (QARI v0.1, 0.2, and v0.3) and

the selected models was assessed using our Arabic test set. The quantitative outcomes, based on CER, WER, and BLEU scores, are presented in Table 3.

As shown in Table 3, QARI v0.2 significantly outperforms all other open source models evaluated, establishing a new benchmark on our test set with a CER of 0.061, a WER of 0.160, and a BLEU score of 0.737. These results underscore the effectiveness of our specialized fine-tuning methodology, particularly the benefit derived from training on synthetic data rich in diacritics and typographic variations (Dataset v0.2). Notably, QARI v0.2 also surpasses the performance of the API-based Mistral OCR in terms of error rates and BLEU score. In contrast, the baseline Qwen models, without specific fine-tuning for Arabic OCR, demonstrate considerably higher error rates, emphasizing the critical need for task-specific adaptation when dealing with complex scripts.

In addition to these quantitative benchmarks, a qualitative assessment is vital for understanding the model’s proficiency in handling the nuanced complexities of Arabic script. Figure 2 provides a visual illustration of Qari-OCR’s output on a challenging text sample. The input image (left panel of Figure 2) exhibits several features typical of printed Arabic that pose difficulties for OCR systems. These include the full array of diacritics (tashkeel) essential for pronunciation and meaning; ligatures such as Lam-Alif ( ); contextually variant letterforms; classical language structures and poetic meter conventions; embedded punctuation and Eastern Arabic numerals; diverse orthographic forms of the Hamza ( ); and features like Maddah

ﻲﻟﺎﺘﻟا حﺎﺒﺼﻟا ﻰﻓ دﻮﻤﺤﻣ ﺮﻴﻣﻷا ﻆﻘﻴﺘﺳا":ﺐﺟا ﻢﻟ ،أﺮﻗا'\' : Output ﲆ ﺔﻃﻮﺑﺮﻤﻟا ئﺎﺼﻌﻟا ﻻإ ﺲﻣﻷﺎﺑ ﻪﺑﺎﺻأ ﺎﻤﻣ ﺮﺜﻧأ ﻪﻴﻠﻋ ىﺮﺒﻳ ﻻ ﺎًﺋرﺎﺑ تﺎﻫ (ا) .”رﺎﺘﻟا مزﺎﻫ ﺎﺑ ﷲ كﺎﻴﺣ :لﺎﻗو ﻪﺑ ّﺮﺷُ نﺎﻄﻠﺴﻟا هآر ﺎﻤﻠﻓ

|[Figure 3]| |
|---|---|
| | |

|ﲆﻋ|
|---|

|،ﻪﺳأر|
|---|

ﻲﺘﻣ (ب) ،ﺔﻠﻤﺟ ﻲﻓ ﻪﺑ ﻲﺗﺄﺗ ﺎﻣ ﻊﺿو (ﺔﺑﺎﺼﻌﻟا) ﻊﻤﺟو (ﺎًﺋرﺎﺑ) دﺎﻀﻣ ﺎﻬﻣّﺪﻗ ﻲﺘﻟا ﺢﺋﺎﺼﻨﻟا ﺎﻣ (ج) ؟هذﺎﻘﻧإ ﻢﺗ ﻒﻴﻛو

) ﺐﻴﺻأ ﺎﻳ) :ﻪﻟﻮﻘﺑ نﺎﻄﻠﺴﻟا ﺪﺼﻘﻳ اذﺎﻣ

|؟(دﻮﻤﺤﻣ|
|---|

QARI OCR Model

؟ثدﺎﺤﻟا اﺬﻫ ﺪﻌﺑ ﻪﻟ نﺎﻄﻠﺴﻟا

|( د)|
|---|

رﻮﺻ (ـﻫ) ،لﻮﻘﺗ ﺎﻤﻟ ﻞﻠﻋ ؟ﻼﻌﻓً ﻚﻟﺬﻛ (دﻮﻤﺤﻣ) نﺎﻛ ﻞﻫو ؟(رﺎﺘﻟا مزﺎﻫ ﻰﻔﺷ نأ ﱃإ ﻪﺘﻠﺣر ﻲﻓ (دﻮﻤﺤﻣ) ﺮﺧﺄﺗ ﺬﻨﻣ (دﺎﻬﺟ) ﺮﻋﺎﺸﻣ ﻚﺑﻮﻠﺳﺄﺑ ﺮﻴﻐﺼﻟا سرﺎﻔﻟا ﺔﺑﺎﺻإ ﺔﺼﻗ ﻲﻓ ( و) ،ﻚﻟذ تﻻﻻد ﺎﺘﻴﻣ ،ﻪﺣاﺮﺟ

ﻦﻣ ﺔﻴﺳوﺮﻓو ،ةﻮﺑﻷا ﺔﻔﻃﺎﻋو ،مﺪﺨﻟا ﺔﻧﺎﻣأو

| |
|---|

،لﺎﻔﻃﻷا لﺎﻴﺧ :ﱃإ تارﺎﺷإ 'n\.ﻚﻟذ ﺮﻫﺎﻈﻣ ﺢﺿو ،ءيﺮﺑ ﺐﺣو ،ةﺮﻜﺒﻣ

| |
|---|

- Figure 3: Example of Qari-OCR (v0.3) accurately transcribing Arabic text from a low-resolution and tightly cropped image, showcasing robustness to visual constraints.

[Figure 4]

QARI OCR Model

ﻲﺷ نﻵا ﻞﻛأ ﺎﻴﻫ -. ﻒﻳرﺰﻟا ﻲﻓ ﺲﻃﺎﻄﺒﻟا مﻼﻔﻟا عرﺰﻳ فﻮﺳ -' : Output n\. ﺔﻴﻟﺎﻌﻟا ةﺮﻬﺸﻟا هﺬﻫ اﻮﻘﻠﺴﺗ نأ اﻮﻟوﺎﺤﺗ ﻻ -. ﺞﻣﺎﻧﺮﺒﻟا ﺔﻳاﺪﺑ ﻞﺒﻗ ﻲﻬﺘﻧ

- Figure 4: Qari-OCR v0.3 successfully transcribing handwritten Arabic text, maintaining sentence structure, punctuation, and recognizing itemized formatting.

ing its effectiveness with compressed layouts, edgebound scripts, and reduced-resolution content. This capability is vital for digitizing real-world historical or educational Arabic materials, which may not always be of pristine quality.

In addition to printed text, QARI v0.3 was also assessed for its ability to process handwritten Arabic, a notoriously challenging task. Figure 4 illustrates its performance on a handwritten sample. The model accurately detects full sentences, preserving punctuation and word boundaries. Notably, it correctly interprets visual structural cues, such as itemized lists (akin to bullet points) and sentencelevel formatting, even with the inherent variability of handwriting. This shows promising initial capabilities for handling handwritten Arabic content.

( ) and crucial letter-distinguishing dots.

The corresponding output from our Qari-OCR model (right panel of Figure 2) showcases a high degree of fidelity in transcribing these intricate elements. The model proficiently recognizes the majority of diacritical marks, accurately segments words despite ligatures and contextual letter shaping, and correctly renders classical linguistic forms. This qualitative performance provides strong corroborative evidence for the quantitative results, especially for QARI v0.2, highlighting its robustness in managing the various challenges frequently encountered in real-world Arabic textual scripts.

These qualitative examples, particularly from QARI v0.3 which was trained on more diverse layouts, complement the quantitative results and highlight the practical utility of Qari-OCR in handling a range of challenging real-world Arabic document types.

To evaluate robustness across diverse Arabic fonts, we benchmarked best-performing models, including QARI v0.2, QARI v0.3, and Mistral OCR, on the SARD dataset4, which includes 1,000 images spanning five common fonts, including Amiri, Arial, Calibri, Sakkal Majalla, and Scheherazade.

Beyond quantitative benchmarks, qualitative analysis is crucial for understanding the model’s practical capabilities. Figure 2 illustrates QariOCR’s proficiency in handling different complexities, supporting the strong quantitative performance of QARI v0.2.

As shown in Table 4, Mistral achieved the lowest error rates overall, particularly excelling in CER and WER. However, QARI v0.2 was highly competitive—outperforming Mistral OCR in BLEU for the Arial font and matching it closely for Calibri. Notably, QARI v0.2’s BLUE scores outperformed Mistral OCR for some fonts, including Arial, Calibri, and Sakkak, and consistently outperformed QARI v0.3 across all metrics. These results highlight QARI v0.2 as a strong open-source alternative, balancing accessibility, performance, and versatil-

Furthermore, the model’s resilience to optical degradation and its ability to handle varied inputs were tested. As shown in Figure 3, Qari-OCR (specifically QARI v0.3, trained on more complex layouts) accurately transcribes text from a lowresolution image. Despite the image’s small size and tightly cropped boundaries, the model robustly detects and transcribes the Arabic text, demonstrat-

4https://huggingface.co/datasets/riotu-lab/SARD

Table 4: CER, WER, and BLEU Score results by Font and Model on SARD Dataset

Metric Model Amiri Arial Calibri Sakkal M. Scheherazade

Mistral OCR 0.011 0.051 0.035 0.040 0.020

CER↓

- Qari v0.2 0.200 0.230 0.193 0.216 0.156
- Qari v0.3 0.350 0.461 0.400 0.424 0.483

Mistral OCR 0.041 0.248 0.166 0.194 0.099

WER↓

- Qari v0.2 0.267 0.308 0.249 0.293 0.211
- Qari v0.3 0.369 0.482 0.432 0.449 0.464

Mistral OCR 0.920 0.634 0.746 0.715 0.845

- Qari v0.2 0.723 0.703 0.745 0.701 0.782
- Qari v0.3 0.346 0.229 0.286 0.279 0.255

BLEU↑

ity across typographic variations.

### 4.3.1 Impact of Model Quantization

To assess the trade-offs between model size, computational efficiency, and performance, we evaluated different quantization levels for our QARI v0.2 and QARI v0.3 models. Specifically, we compared versions fine-tuned or inferred using 8-bit precision against those utilizing more aggressive 4-bit quantization. The results, presented in Table 5, highlight the impact of these quantization strategies on the CER, WER, and BLEU scores.

Table 5: Performance of QARI with 8-bit Vs. 4-bit Quantization.

Model Quant. CER ↓ WER ↓ BLEU ↑

- QARI v0.2 8-bit 0.091 0.255 0.583 4-bit 3.452 4.516 0.001
- QARI v0.3 8-bit 0.133 0.353 0.472 4-bit 3.228 6.428 0.001

As observed in Table 5, employing 8-bit quantization during fine-tuning or inference maintains strong performance for both QARI v0.2 and QARI v0.3, offering a good balance between efficiency and accuracy. However, the more aggressive 4-bit quantization leads to a substantial degradation in performance across all metrics for both model versions. This suggests that while 4-bit quantization significantly reduces the model footprint and can accelerate inference, it incurs a considerable accuracy cost for the fine-grained task of Arabic OCR with these specific models and fine-tuning parameters. The 8-bit versions, therefore, represent the more practical choice when accuracy is paramount, while 4-bit might be considered only in scenarios with extreme computational constraints where a significant drop in accuracy is acceptable.

## 5 Discussion

Our experiments reveal distinct strengths across the Qari-OCR model iterations. While QARI v0.2, trained on 50,000 diverse samples (Dataset v0.2), demonstrates superior overall quantitative performance for plain text recognition (Table 3), QARI v0.3, developed with a smaller 10,000-sample dataset focused on complex HTML-like layouts (Dataset v0.3), excels in preserving document structure.

Qualitative analysis, as shown in Figure 5, illustrates that QARI v0.3 effectively reconstructs HTML tags and formatting from input images, often achieving lower local error rates on these structurally rich examples compared to QARI v0.2’s plain text output. This proficiency stems directly from QARI v0.3’s targeted training on layoutaware synthetic data. The trade-off appears to be that QARI v0.2’s larger and more varied characterlevel training data fostered better general textual accuracy, whereas QARI v0.3’s smaller, specialized dataset, combined with a single training epoch, prioritized structural fidelity, potentially at the cost of some raw text accuracy on average.

Furthermore, resource efficiency considerations favor the QARI v0.3 approach for structureoriented tasks. As depicted in Figure 6, the 10ksample training regimen (represented by QARI v0.3’s development) was significantly more economical in terms of training time and estimated CO2 emissions (1.88 kg eq. CO2 over 11 hours) compared to the 50k-sample training (represented by "QARI 3", akin to QARI v0.2’s development, at 9.4 kg eq. CO2 over 55 hours). This highlights the potential for developing specialized, efficient models when the primary objective is structural document conversion.

In essence, QARI v0.2 serves as our most robust general-purpose Arabic OCR engine for accurate

#### Input QARI V0.2 QARI V0.3 Ground Truth

[Figure 5]

'يِرﻮﺒُ ﺜُ ِﺑ ﺎﻬِﻠْﻫَ َأ ﺔ ُ ﻣﻮَ ُﻜُﺣ ﺖ ْ ﻀَ ﻗَ وَ ﺎﻬَ َﻧوُد ﺖ ْ َﻟﺎَﺤﻓَ ﻲِﺑ ُلِذاﻮَ ﻌَ ﻟاْ ﺖ ِ ﺷَ وَ ' ﺖْ ﻀَ ﻗَ وَ ﺎﻬَ َﻧوُد ﺖ ْ َﻟﺎَﺤﻓَ ﻲِﺑ ُلِداﻮَ ﻌَ ﻟاْ <u/>ﺖﺷْ وَ <p><u>' '<p/>يِرﻮﺒُ ﺜُ ِﺑ <u/>ﺎﻬِﻠْﻫَ َأ<u> ﺔُ ﻣﻮَ ُﻜُﺣ

ﺎﻬَ َﻧوُد <b/>ﺖْ َﻟﺎَﺤﻓَ <b> ﻲِﺑ ُلِذاﻮَ ﻌَ ﻟاْ <u/>ﺖِ ﺷَ وَ <p><u>' '<p/>يِرﻮﺒُ ﺜُ ِﺑ <u/>ﺎﻬِﻠْﻫَ َأ<u> ﺔُ ﻣﻮَ ُﻜُﺣ <b/>ﺖْ ﻀَ ﻗَ وَ <b>

[Figure 6]

WER CER WER CER

BLEU

BLEU

،ﻲﺳوﺮﻌﻟا <i/>ءاﺮﻫﺰﻟا<i> ﺔﻤﻃﺎﻓ ﺔﻴﻨﻐﻤﻟا ﺖﻘﻠﺗ<p>' ﺎﻬﻤﻳﺪﻘﺗ ﺪﻌﺑ ،سدﺎﺴﻟا ﺪﻤﺤﻣ ﻚﻠﻤﻟا ﻦﻣ <u/>ﺔﻟﺎﺳر<u> <u/>ءاﺪﻫإ<u> « ﺎﻧﺪﻴﺳ <b/>ﺔﻣﻼﺳ<b> ﲆﻋ» <i/>ﺔﻴﻨﻏﻷ<i> i>/>ﺔﻠﻴﻤﻌﻟ<i> ﻪﺋاﺮﺟإ ﺪﻌﺑ ﻪﺋﺎﻔﺷ <u/>ﺔﺒﺳﺎﻨﻤﺑ<u> ،ﻚﻠﻤﻠﻟ ﺖﻠﻠﻜﺗ <i/>ﻲﺘﻟاو<i> <i/>،ﺔﻴﺣاﺮﺟ<<i <u/>ﺔﻟﺎﺳﺮﻟا<u> ﺺﻧ ﻲﺳوﺮﻌﻟا تﺮﺸﻧو<p>\n<p/>حﺎﺠﻨﻟﺎﺑ ﻊﻗاﻮﻣ <u/>ﲆﻋ<b> <u/>ﺔﻴﻤﺳﺮﻟا<b> ﺎﻬﺗﺎﺤﻔﺻ ﲆﻋ ﲆﻋ ﻚﻠﻤﻠﻟ ﺮﻜﺸﻟا ﺖﻬﺟو ﺚﻴﺣ <u/>،ﻞﺻاﻮﺘﻟا<u> ﺎﻘﻴﻤﻋ اﺮﺛأ ﺎﻬﻳﺪﻟ ﺖﻛﺮﺗ ﺎﻬﻧأ ﲆﻋ ةﺪﻛﺆﻣ <i/>ﺔﻟﺎﺳﺮﻟا<i> u>/>ﺔﻓﺎﻔﺘﻟﻹا<u> <u/>نأو<u> ،روﺮﺳو <b/>ﺔﺠﻬﺑو<b> <i/>ﻦﻣ<i> <i/>ﺪﻳﺰﻤﻟا<i> ﻢﻳﺪﻘﺘﻟ ﺎﻬﻳﺪﻟ اﺰﻓﺎﺣ <i/>نﻮﻜﺘﺳ<<i b>:> ﺮﺸﻨﻟا قﻮﻘﺣ <p>\n<h1><b>©</b/>دﺎﻬﺘﺟﻹاو ءﺎﻄﻌﻟا '<</b> DR</h1

0.55 0.29 0.44 0.15

0.08

0.16

ﺪﻤﺤﻣ ﻚﻠﻤﻟا ﻦﻣ ﺔﻟﺎﺳر ،ﻲﺳوﺮﻌﻟا ءاﺮﻫﺰﻟا ﺔﻤﻃﺎﻓ ﺔﻴﻨﻐﻤﻟا ﺖﻘﻠﺗ' ءاﺪﻫإ « ﺎﻧﺪﻴﺳ ﺔﻣﻼﺳ ﲆﻋ» ﺔﻴﻨﻏﻷ ﺎﻬﻤﻳﺪﻘﺗ ﺪﻌﺑ ،سدﺎﺴﻟا ﺖﻠﻠﻜﺗ ﻲﺘﻟاو ،ﺔﻴﺣاﺮﺟ ﺔﻠﻴﻤﻌﻟ ﻪﺋاﺮﺟإ ﺪﻌﺑ ﻪﺋﺎﻔﺷ ﺔﺒﺳﺎﻨﻤﻳ ،ﻚﻠﻤﻠﻟ ﺔﻴﻤﺳﺮﻟا ﺎﻬﺗﺎﺤﻔﺻ ﲆﻋ ﺔﻟﺎﺳﺮﻟا ﺺﻧ ﻲﺳوﺮﻌﻟا تﺮﺸﻧو حﺎﺠﻨﻟﺎﺑ ﺔﻟﺎﺳﺮﻟا ﲆﻋ ﻚﻠﻤﻠﻟ ﺮﻜﺸﻟا ﺖﻬﺟو ﺚﻴﺣ ،ﻞﺻاﻮﺘﻟا ﻊﻗاﻮﻣ ﲆﻋ نأو ،روﺮﺳو ﺔﺠﻬﺑو ﺎﻘﻴﻤﻋ اﺮﺛأ ﺎﻬﻳﺪﻟ ﺖﻛﺮﺗ ﺎﻬﻧأ ﲆﻋ ةﺪﻛﺆﻣ دﺎﻬﺘﺟﻹاو ءﺎﻄﻌﻟا ﻦﻣ ﺪﻳﺰﻤﻟا ﻢﻳﺪﻘﺘﻟ ﺎﻬﻳﺪﻟ اﺰﻓﺎﺣ نﻮﻜﺘﺳ ﺔﻓﺎﻔﺘﻟﻹا 'DR : ﺮﺸﻨﻟا قﻮﻘﺣ ©

i>/>،ﻲﺳوﺮﻌﻟا<i> ءاﺮﻫﺰﻟا ﺔﻤﻃﺎﻓ ﺔﻴﻨﻐﻤﻟا ﺖﻘﻠﺗ<p>' ﺔﻴﻨﻏﻷ ﺎﻬﻤﻳﺪﻘﺗ ﺪﻌﺑ ،سدﺎﺴﻟا ﺪﻤﺤﻣ ﻚﻠﻤﻟا ﻦﻣ <u/>ﺔﻟﺎﺳر<<u ،ﻚﻠﻤﻠﻟ <u/>ءاﺪﻫإ<i> » <u/>ﺎﻧﺪﻴﺳ<i> <i/>ﺔﻣﻼﺳ<i> ﲆﻋ» ﻲﺘﻟاو ،ﺔﻴﺣاﺮﺟ ﺔﻠﻴﻤﻌﻟ ﻪﺋاﺮﺟإ ﺪﻌﺑ ﻪﺋﺎﻔﺷ <u/>ﺔﺒﺳﺎﻨﻤﺑ<u> ﺺﻧ <i/>ﻲﺳوﺮﻌﻟا<i> تﺮﺸﻧو<p><br><p/>حﺎﺠﻨﻟﺎﺑ ﺖﻠﻠﻜﺗ ﻊﻗاﻮﻣ <u/>ﲆﻋ<u> ﺔﻴﻤﺳﺮﻟا ﺎﻬﺘﺤﻔﺻ ﲆﻋ <u/>ﺔﻟﺎﺳﺮﻟا<u> ﺔﻟﺎﺳﺮﻟا ﲆﻋ ﻚﻠﻤﻠﻟ ﺮﻜﺸﻟا ﺖﻬﺟو ﺚﻴﺣ <u/>،ﻞﺻاﻮﺘﻟا<u> ،روﺮﺳو ﺔﺠﻬﺑو <b/>ﺎﻘﻴﻤﻋ<b> اﺮﺛأ ﺎﻬﻳﺪﻟ ﺖﻛﺮﺗ ﺎﻬﻧأ ﲆﻋ ةﺪﻛﺆﻣ ﻢﻳﺪﻘﺘﻟ ﺎﻬﻳﺪﻟ اﺰﻓﺎﺣ نﻮﻜﺘﺳ <u/>ﺎﻓﺎﻐﺘﻟﻹا<u> <u/>نأو<u> : ﺮﺸﻨﻟا قﻮﻘﺣ ©<p><br><h1/>دﺎﻬﺘﺟﻹاو ءﺎﻄﻌﻟا ﻦﻣ ﺪﻳﺰﻤﻟا '<DR</h1

[Figure 7]

WER CER

BLEU

0.40

0.30

0.28

WER CER

BLEU

0.41

0.31

0.21

ﻲِﻤﱡ ﻈْﻋَ َﻷا ﻂﻴ ُ ﺳِ ﻮَ ﻟاْ <i/>فُ ﱠﺮﻌَ ُﻳ<i> ،تﺎِ ﻴﱠ ﺿﺎِ َﻳﱢﺮﻟا <u/>ﻲِﻓ<h2><u>' ﻲِﻄﻌْ ُﺗ ﻲِﺘﱠﻟا ِﺔﱠﻟاﱠﺪﻟا ُﻞﺧَدْ ﻂﻴ ُ ﺳِ وَ <i/>ﻪُ ﱠﻧَأ<i> ﲆَﻋَ ﺎﻣَ ِﺔﱠﻟاَﺪِﻟ ﱡﺪَﺤﻟاﺎْ ﻴَ ﻧُدْ ُدوُﺪُﺣوَ <u/>ﺎﻴَ َﻠَﻋ<i> <u/>ُدوُﺪُﺣ<i> ٍﺔﻤﻴِﻗَ <i/>َﺮﺒَ ﻛْ َأ<i> :َﺮﺧآَ ﺮﻴﺒِ ﻌْ ﺘَ َﺑوَ <h2><br><h3/>جِ ﺮْ َﺨﻟاْ ﻲِﻓ ِﺔﱠﻟاﱠﺪﻠِﻟْ ﲆْﻋ َ ُﻷا ٍﺔﻤﻴِﻗَ ُﺮﺒَ ﻛْ َأ ِﺔﱠﻟاﱠﺪﻠِﻟْ نﻮ ُ ُﻜَﻳ ﺎﻬِﻠَ ﺟْ َأ ﻦِﻣ ْ ﻲِﺘﱠﻟا ل ِ ﻮﱢ َﺤﺘَ ﻤُ ﻟاْ ﺔ ُ ﻤﻴِﻗَ <u/>ﻮُﻫَ <u> َﺮﺒَ ﻛْ َأ ن ﱠ ﺈِ ﻓَ ﻮُﻫ َ نﺎ َ ﻛَ لﺎ ِ ﺣَ ﻲِﻓ ًﻼﺜَ ﻣَ <u></h3><br><h1/>ٍﺔﻨِﻜَ ﻤْ ﻣُ <u> ﻲِﻓ ﺎﻬَ َﻟ ٍﺔﻤﻴِﻗَ َﺮﺒَ ﻛْ َأ ﺔ ُ ﱠﻟاﱠﺪﻟا ﺬ ُ ُﺧْﺄَﺗ ُهَﺪﻨِﻋ ﺚ ُ ﻴْ ﺣَ ﻮُﻫ َ ﻲِﻤ ﱢ ﻈْﻋَ َﻷا ﻂﻴ ِ ﺳِ ﻮَ ﻟاْ ﻊُ ﺟاﺮِ ﻤَ ﻟاْ <u/>ُءﺎﺼَ ﺣإْ <u> ٌلﺎﻨَ ﻣُ ﺎﻀً ﻳْ َأ ﺮ ْ ﻈُ ﻧاْ <h1><br><p/>جِ ﺮْ َﺨﻟاْ '<p/>ﺔٌ ﻴِﺋاَﺪِﺘﱠ ْﺑإِ تﺎ ُ ﻴﱠ ﺿﺎِ َﻳِر ﺔ ٌ ﻴﱠ ﺴِ ﻜَﻋْ ٌلاوَدَ

ﻲِﻤﱡ ﻈْﻋَ َﻷاْ ﻂﻴ ُ ﺳِ ﻮَ ﻟاْ ف ُ ﱠﺮﻌَ ُﻳ <i/>،تﺎِ ﻴﱠ ﺿﺎِ َﻳﱢﺮﻟا<u> <i/>ﻲِﻓ<h3><u>' ﻲِﻄﻌْ ُﺗ ﻲِﺘﱠﻟا ِﺔﱠﻟاﱠﺪﻟا <i/>ُﻞﺧَدْ <i> ﻂﻴٌ ﺳِ وَ ﻪ ُ ﱠﻧَأ <b/>ﲆَﻋَ <b> ﺎﻣَ ٍﺔﱠﻟاَﺪِﻟ <b/>ﱡﺪَﺤﻟاﺎْ ﻴَ ﻧُدْ <b> ٌدوُﺪُﺣوَ <u/>ﺎﻴِﻠَﻋَ <u> ٌدوُﺪُﺣ ٍﺔﻤﻴِﻗَ <i/>َﺮﺒَ ﻛْ َأ<i> :َﺮﺧآَ ٍﺮﻴﺒِ ﻌْ ﺘَ ِﺑوَ <b></h3>\n<h4/>جِ ﺮْ َﺨﻟاْ <b> ﻲِﻓ ِﺔﱠﻟاﱠﺪﻠِﻟْ ﲆْﻋ َ َﻷاْ نﻮُ ُﻜَﻳ ﺎﻬِﻠَ ﺟْ َأ ﻦِﻣ ْ <b/>ﻲِﺘﱠﻟا<b> لِ ﻮﱢ َﺤﺘَ ﻤُ ﻟاْ ﺔ ُ ﻤﻴِﻗَ <u/>ﻮُﻫَ <u> u></h4>\n<h1>/>ٍﺔﻨِﻜَ ﻤْ ﻣُ <u> ٍﺔﻤﻴِﻗَ ُﺮﺒَ ﻛْ َأ <i/>ِﺔﱠﻟاﱠﺪﻠِﻟ<i> ﻂﻴِ ﺳِ ﻮَ ﻟاْ َﺮﺒَ ﻛْ َأ ن ﱠ ﺈِ ﻓَ ﻮُﻫ َ نﺎ َ ﻛَ لﺎ ِ ﺣَ ﻲِﻓ <b/>ﻼً ﺜَ ﻣَ <<b ﺎﻬَ َﻟ ٍﺔﻤﻴِﻗَ َﺮﺒَ ﻛْ َأ ﺔ ُ ﱠﻟاﱠﺪﻟا ﺬ ُ ُﺧْﺄَﺗ ُهَﺪﻨِﻋ ﺚ ُ ﻴْ ﺣَ ﻮُﻫ َ <i/>ﻲِﻤﱢ ﻈْﻋَ َﻷاْ <i> ٌلﺎﻨَ ﻣُ ﺎﻀً ﻳْ َأ ﺮ ْ ﻈُ ﻧاْ <i></h1>\n<p/>جِ ﺮْ َﺨﻟاْ <i> <i/>ﻲِﻓ<i> <b/>تﺎٌ ﻴﱠ ﺿﺎِ َﻳِر<b> ﺔٌ ﻴﱠ ﺴِ ﻜَﻋْ ﱞلاوَدَ ﻊﺟاﺮ ُ ﻤَ ﻟاْ <u/>ٌءﺎﺼَ ﺣْ إِ <u> 0.61 '<p/>ﺔٌ ﻴِﺋاَﺪِﺘﱠ ْﺑإِ

ﻂﻴٌ ﺳِ وَ ﻪ ُ ﱠﻧَأ ﲆَﻋ َ ﺎﻣَ ِﺔﱠﻟاَﺪِﻟ ﻲِﻤ ﱡ ﻈْﻋَ َﻷا ﻂﻴ ُ ﺳِ ﻮَ ﻟاْ ف ُ ﱠﺮﻌَ ُﻳ ،تﺎِ ﻴﱠ ﺿﺎِ َﻳﱢﺮﻟا ﻲِﻓ' ﱡﺪَﺤﻟﺎْ ﻴَ ﻧُدْ ٌدوُﺪُﺣوَ ﺎﻴِﻠَﻋً ٌدوُﺪُﺣ ٍﺔﻤﻴِﻗَ َﺮﺒَ ﻛْ َأ ﻲِﻄﻌْ ُﺗ ﻲِﺘﱠﻟا ِﺔﱠﻟاﱠﺪﻟا ُﻞﺧَدَ ﻦِﻣْ ﻲِﺘﱠﻟا ل ِ ﻮﱢ َﺤﺘَ ﻤُ ﻟاْ ﺔ ُ ﻤﻴِﻗَ ﻮُﻫ َ :َﺮﺧآَ ٍﺮﻴﺒِ ﻌْ ﺘَ َﺑوَ ج ِ ﺮْ َﺨﻟاْ ﻲِﻓ ِﺔﱠﻟاﱠﺪﻠِﻟْ ﲆْﻋ َ ُﻷا َﺮﺒَ ﻛْ َأ ن ﱠ ﺈِ ﻓَ ﻮُﻫ َ نﺎ َ ﻛَ لﺎ ِ ﺣَ ﻲِﻓ ﻼ ً ﺜَ ﻣَ ٍﺔﻨِﻜَ ﻤْ ﻣُ ٍﺔﻤﻴِﻗَ ُﺮﺒَ ﻛْ َأ ِﺔﱠﻟاﱠﺪﻠِﻟ نﻮ ُ ُﻜَﻳ ﺎﻬِﻠَ ﺟْ َأ ﻲِﻓ ﺎﻬَ َﻟ ٍﺔﻤﻴِﻗَ َﺮﺒَ ﻛْ َأ ﺔ ُ ﱠﻟاﱠﺪﻟا ﺬ ُ ﱡﺧَﺄَﺗ ُهَﺪﻨِﻋْ ﺚ ُ ﻴْ ﺣَ ﻮُﻫ َ ﻲِﻤ ﱠ ﻈْﻋَ َﻷا ﻂﻴ َ ﺳِ ﻮَ ﻟاْ تﺎُ ﻴﱠ ﺿﺎِ َﻳِر ﺔ ٌ ﻴﱠ ﺴِ ﻜَﻋْ ٌلاوَدَ ﻊ ُ ﺟاﺮِ ﻤَ ﻟاْ ُءﺎﺼَ ﺣإْ ٌلﺎﻨَ ﻣُ ﺎﻀً ﻳْ َأ ﺮ ْ ﻈُ ﻧاْ ج ِ ﺮْ َﺨﻟاْ 'ﺔٌ ﻴِﺋاَﺪِﺘﱠ ْﺑإِ

BLEU

WER CER

BLEU

WER CER

0.09

0.30 0.22

0.15

0.53

- Figure 5: Qualitative comparison of QARI v0.2 and QARI v0.3 outputs against Input and Ground Truth for various Arabic text samples.

[Figure 8]

- Figure 6: Comparison of estimated resource consumption (CO2 Emissions, Training Time, Sample Size) for training QARI model variants.

Qari-OCR’s current capabilities are primarily focused on textual content within the main body of documents; it often struggles to accurately recognize and extract text embedded within figures, charts, or complex graphical elements. Thirdly, the model’s performance on historical or non-standard Arabic numeral systems has not been extensively validated and may be suboptimal. Finally, text elements typically found on the periphery of scanned pages, such as book titles on covers, page numbers, or marginalia, are sometimes skipped or inaccurately transcribed, indicating an area for improved contextual awareness and layout analysis.

plain text extraction. QARI v0.3, however, validates a promising and resource-efficient strategy for applications requiring the understanding and reproduction of document structure, like HTML. The optimal model choice is therefore contingent on the specific end-goal: high-fidelity plain text output (QARI v0.2) or structural document reconstruction with greater training efficiency.

## 6 Limitations

Despite the strong performance of Qari-OCR, particularly QARI v0.2, the current study and model possess certain limitations; Firstly, while proficient with dense printed text, the model may encounter difficulties with extremely heavy text layouts where character or line spacing is minimal, potentially leading to recognition errors. Secondly,

## 7 Conclusion

In conclusion, this paper presented Qari-OCR, a fine-tuned vision-language model that achieves state-of-the-art performance for Arabic text recognition by leveraging extensive synthetic data and specializing the Qwen2-VL architecture. Our QARI v0.2 model significantly surpasses existing open-source solutions in accurately handling diacritics, diverse fonts, and complex layouts in printed Arabic. Future work will focus on addressing current limitations by enhancing robustness to dense text and embedded graphics, improving numeral recognition, advancing layout analysis for peripheral text, and extending capabilities to Arabic handwriting recognition. These efforts aim to develop Qari-OCR into an even more comprehensive solution for Arabic document understanding.

## Acknowledgments

The authors thank Prince Sultan University for their support.

## References

I Saleh Al-Sheikh, MASNIZAH Mohd, and L Warlina. 2020. A review of arabic text recognition dataset. Asia-Pacific J. Inf. Technol. Multimedia, 9(1):69–81.

Naseem Alrobah and Saleh Albahli. 2022. Arabic handwritten recognition using deep learning: A survey. Arabian Journal for Science and Engineering, 47(8):9943–9963.

Fakhraddin Alwajih, El Moatez Billah Nagoudi, Gagan Bhatia, Abdelrahman Mohamed, and Muhammad Abdul-Mageed. 2024. Peacock: A family of arabic multimodal large language models and benchmarks. arXiv preprint arXiv:2403.01031.

Gagan Bhatia, El Moatez Billah Nagoudi, Fakhraddin Alwajih, and Muhammad Abdul-Mageed. 2024. Qalam: A multimodal llm for arabic optical character and handwriting recognition. arXiv preprint arXiv:2407.13559.

Alex Graves and Jürgen Schmidhuber. 2008. Offline handwriting recognition with multidimensional recurrent neural networks. Advances in neural information processing systems, 21.

Ahmed Heakl, Sara Ghaboura, Omkar Thawkar, Fahad Shahbaz Khan, Hisham Cholakkal, Rao Muhammad Anwer, and Salman Khan. 2025. Ain: The arabic inclusive large multimodal model. arXiv preprint arXiv:2502.00094.

Mahdi Nsaif Jasim. 2020. Arabic optical characters recognition by neural network based arabic unicode.

Dietrich Klakow and Jochen Peters. 2002. Testing the correlation of word error rate and perplexity. Speech Communication, 38(1-2):19–28.

Minghao Li, Tengchao Lv, Jingye Chen, Lei Cui, Yijuan Lu, Dinei Florencio, Cha Zhang, Zhoujun Li, and Furu Wei. 2023. Trocr: Transformer-based optical character recognition with pre-trained models. In Proceedings of the AAAI conference on artificial intelligence, volume 37, pages 13094–13102.

Ilya Loshchilov and Frank Hutter. 2017. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101.

Mistral AI Team. 2025. Mistral ocr: Introducing the world’s best document understanding api. https:// mistral.ai/news/mistral-ocr. Research, March 6, 2025.

Kishore Papineni, Salim Roukos, Todd Ward, and WeiJing Zhu. 2002. Bleu: a method for automatic evaluation of machine translation. In Proceedings of the

40th annual meeting of the Association for Computational Linguistics, pages 311–318.

Binod Kumar Pattanayak, Anil Kumar Biswal, Suprava Ranjan Laha, Saumendra Pattnaik, Bibhuti Bhusan Dash, and Sudhansu Shekhar Patra. 2023. A novel technique for handwritten text recognition using easy ocr. In 2023 International Conference on Self Sustainable Artificial Intelligence Systems (ICSSAS), pages 1115–1119. IEEE.

Joan Puigcerver. 2017. Are multidimensional recurrent layers really necessary for handwritten text recognition? In 2017 14th IAPR international conference on document analysis and recognition (ICDAR), volume 1, pages 67–72. IEEE.

Ray Smith. 2007. An overview of the tesseract ocr engine. In Ninth international conference on document analysis and recognition (ICDAR 2007), volume 2, pages 629–633. IEEE.

UNESCO. 2024. World Arabic Language Day. UNESCO Official Website. The Arabic language is a pillar of the cultural diversity of humanity. It is one of the most widely spoken languages in the world, used daily by more than 400 million people.

Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. 2024. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191.

Mohamed Yousef, Khaled F Hussain, and Usama S Mohammed. 2020. Accurate, data-efficient, unconstrained text recognition with convolutional neural networks. Pattern Recognition, 108:107482.

