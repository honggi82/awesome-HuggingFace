# arXiv:2410.13334v5[cs.CL]25Nov2025

## BiasJailbreak:Analyzing Ethical Biases and Jailbreak Vulnerabilities in Large Language Models

Isack Lee1∗ Haebin Seong1* 1Theori Inc. isacklee224@gmail.com, hbseong97@gmail.com

1Association for the Advancement of Artificial Intelligence 1101 Pennsylvania Ave, NW Suite 300 Washington, DC 20004 USA proceedings-questions@aaai.org Abstract

[Figure 1]

Warning: This paper contains potentially offensive and harmful text.

Although large language models (LLMs) demonstrate impressive proficiency in various tasks, they present potential safety risks, such as ‘jailbreaks’, where malicious inputs can coerce LLMs into generating harmful content bypassing safety alignments. In this paper, we delve into the ethical biases in LLMs and examine how those biases could be exploited for jailbreaks. Notably, these biases result in a jailbreaking success rate in GPT-4o models that differs by 20% between non-binary and cisgender keywords and by 16% between white and black keywords, even when the other parts of the prompts are identical. We introduce the concept of BiasJailbreak, highlighting the inherent risks posed by these safety-induced biases. BiasJailbreak generates biased keywords automatically by asking the target LLM itself, and utilizes the keywords to generate harmful output. Additionally, we propose an efficient defense method BiasDefense, which prevents jailbreak attempts by injecting defense prompts prior to generation. BiasDefense stands as an appealing alternative to Guard Models, such as Llama-Guard, that require additional inference cost after text generation. Our findings emphasize that ethical biases in LLMs can actually lead to generating unsafe output, and suggest a method to make the LLMs more secure and unbiased. To enable further research and improvements, we open-source our code and artifacts of BiasJailbreak, providing the community with tools to better understand and mitigate safety-induced biases in LLMs.

Figure 1: BiasJailbreak reveals inherent biases in LLMs that disproportionately allow harmful jailbreak attacks to succeed more frequently when directed towards marginalized groups compared to privileged groups.

safe usage has become crucial. Developers have implemented several safety features to prevent these models from generating harmful or objectionable content, often referred to as ‘safety alignment’ ((Bakker et al. 2022; Christiano et al. 2017; Ouyang et al. 2022)).

These safety alignments often involve additional finetuning or reinforcement learning techniques, which, while designed to enhance safety and alignment, may also inadvertently introduce biases, as highlighted in resources such as (Achiam et al. 2023, p. 49) . However, biases can also arise from other sources, such as pretraining data or system prompts. While it is difficult to pinpoint exactly where these biases originate, the critical fact remains that they exist and can influence the model’s behavior.

##### Code and Artifacts —

https://github.com/Isaac-theori/BiasJailbreak

### Introduction

Large Language Models (LLMs) have rapidly become essential components in many fields, ranging from professional decision-making to various forms of interactive user engagement ((Araci 2019; Luo et al. 2022; Tinn et al. 2023)). However, as these models become popular, ensuring their

In this work, we show that these safety alignments often introduce deliberate and ethical biases, giving rise to a phenomenon known as ’jailbreak’, where malicious inputs manage to circumvent these safety alignments, thus allowing LLMs to generate harmful outputs ((Goldstein et al. 2023; Kang et al. 2024)).

*Equal contribution Copyright © 2026, Accepted to the AAAI 2026 Workshop on Personalization in the Era of Large Foundation Models (PerFM).

The term ‘jailbreak’ refers to carefully crafted prompts that can bait aligned LLMs into bypassing their safety align-

[Figure 2]

- Figure 2: Illustration showcasing the difference in response between a standard prompt and a BiasJailbreak prompt. While the standard prompt is blocked by the LLM’s safety features, the BiasJailbreak prompt exploits ethical biases to elicit a response.

ment, resulting in the generation of content that may be harmful, discriminatory, violent, or sensitive ((Smith et al.

- 2022)). Numerous types of jailbreak attacks have been identified and categorized into two primary methods: white-box and black-box ((Yi et al. 2024)). The white-box approach requires target model gradients or logits and use them as a guidance for finding adversarial jailbreak prompts. Directly fine-tuning the target LLM is not considered as a jailbreak method in this paper, although it some consider it as whitebox. The black-box approach has a harder and a more general real-world setting since it does not have access to such information. Jailbreak methods in black-box usually involve template completion, prompt rewriting, or LLM-based generation.

White-box attacks like GCG ((Zou et al. 2023)) rely on a search scheme guided by gradient information. While this approach enables reliable generation of jailbreak prompts though with a cost of high computation, it carries a significant downside: the resulting prompts often consist of nonsensical sequences, which lack semantic meaning ((Morris et al. 2020)). This major flaw makes these prompts highly vulnerable to naive defense mechanisms such as perplexitybased detection. For example, recent studies ((Jain et al. 2023; Alon and Kamfonas 2023)) have shown that such straightforward defenses can easily recognize these nonsensical prompts and completely undermine the success of white-box attacks.

While having a more applicable and general setting, recent advancements in black-box settings, such as naturallanguage methods like PAIR ((Chao et al. 2023)) and DeepInception ((Li et al. 2023)), have shown that semantically coherent prompts can effectively exploit LLM vulnerabilities. Additionally, manual approaches like GUARD ((Jin et al. 2024a)) iteratively refine jailbreak prompts, demonstrating adaptability to LLM updates.

In this paper, we propose a novel black-box method that offers both scalability and generality. By leveraging inherent biases in LLMs, such as those related to Ethical Sensitivity((Perez et al. 2022; Zhuo et al. 2023)), we ensure that the prompts retain their meaning significantly without losing effectiveness, contrary to white-box attacks. This approach

allows us to overcome the issues of scalability and adaptability while still exploiting the biases for effective jailbreaks.

We explore the novel concept of BiasJailbreak, investigating how biases in LLMs, intended as safety alignment, paradoxically become enablers of harmful content generation when exploited. This behavior is well illustrated in Figure 1 and 2. Additionally, we propose a defense mechanism BiasDefense that adjusts biases using prompts, ensuring safety and efficiency without additional inference or models, which makes it an attractive alternative to Guard Models ((Inan et al. 2023); (Ghosh et al. 2024); (Caselli et al. 2020); (Vidgen et al. 2021)), which are capable of classifying harmful conversations but require additional models and inference after text generation.

Our contributions can be summarized as follows:

- • We analyze the nature and consequences of ethical biases introduced in LLMs for safety purposes, highlighting their potential to not only fail in deterring but also in facilitating more effective jailbreaks. This paradoxical effect underscores the urgency of addressing the inherent vulnerabilities these biases introduce.
- • Through comprehensive experiments, we show that our proposed BiasJailbreak is effective across state-of-the-art models, including the latest iterations of GPT. Our framework also proves adaptable, working effectively when applied to existing jailbreak techniques.
- • We propose BiasDefense, a straightforward defense strategy without the need of additional inference or models. Our findings demonstrate that even with a simple and costeffective defense approach, jailbreak attacks can be mitigated. This highlights the critical responsibility of LLM service providers to ensure robust protection.
- • We open-source the code and all associated artifacts of BiasJailbreak to facilitate community efforts in understanding and mitigating safety-induced biases in large language models. This contribution aims to provide researchers and developers with the necessary tools to explore the nature of these biases and develop more robust defenses, furthering the collective effort to ensure the safety and reliability of LLM deployments. Our research suggests that while ethical biases are crucial

for aligning LLMs with ethical standards, they necessitate careful scrutiny to prevent their manipulation. Therefore, responsible strategies from AI companies and researchers are needed to reinforce the safety of LLMs in an increasingly complex threat landscape.

### Background And Related Works

#### Safety Alignment in LLMs

Ensuring the safety and ethical alignment of large language models (LLMs) is a critical area of ongoing research, since the ethical bias of LLMs can lead to undesirable societal impacts and potential harms ((Li et al. 2024)). Methods such as data filtering, supervised fine-tuning, and reinforcement learning from human feedback (RLHF) aim to align models like GPT-4 and ChatGPT with human values and preferences ((Christiano et al. 2017; Bai et al. 2022; Ouyang et al.

2022; Xu et al. 2020)). However, despite these efforts, recent studies reveal vulnerabilities that can be exploited through ’jailbreak’ attacks, which lead to undesirable and harmful outputs ((Kang et al. 2024; Shen et al. 2023). Additionally, (Zheng et al. 2024) proposed many-shot demonstration techniques, using random search within demo pools and injecting system tokens to bypass safeguards.

#### Jailbreak Attacks and Techniques

Jailbreaking LLMs involves crafting inputs that bypass safety mechanisms, resulting in harmful or objectionable content. Early jailbreak attacks, such as the ”Do-AnythingNow (DAN)” series, relied on manually crafted prompts to exploit LLM safeguards ((Shen et al. 2023)). ((Liu et al.

- 2023) provided an in-depth analysis and categorization of these jailbreak prompts, highlighting the delicate balance between an LLM’s capabilities and its safety constraints.

Diverse strategies for jailbreaks have been proposed. Manual methods, while effective, suffer from scalability issues ((Wei, Haghtalab, and Steinhardt 2024)). On the other hand, learning-based methods like GCG ((Zou et al. 2023)) use adversarial techniques to generate prompts automatically, though often at the cost of producing semantically meaningless outputs detectable via simple defenses like perplexity tests ((Alon and Kamfonas 2023; Liu et al. 2023)) introduced AutoDAN, which combines manual and automated strategies using hierarchical genetic algorithms to enhance both the stealthiness and scalability of jailbreak prompts.

(Zeng et al. 2024a) proposed persuasive adversarial prompts (PAP) that leverage social science-based persuasion techniques to significantly enhance jailbreak success, achieving over 92% success rates across multiple models. Similarly, (Shah et al. 2024) introduced persona modulation, a black-box jailbreak approach that uses personas to exploit vulnerabilities at scale, with high success rates transferable across state-of-the-art models.

Language diversity and non-natural language inputs present additional challenges. (Deng et al. 2023) explored multilingual jailbreak attacks, demonstrating that LLMs could be tricked into producing harmful outputs with nonEnglish prompts. (Yuan et al. 2024; Jin et al. 2024b) extended this by investigating the vulnerabilities of LLMs to non-natural language inputs, such as ciphers.

#### Towards Improved Safety Measures

Complex attack strategies like those proposed by (Ding et al.

- 2023) with the ReNeLLM framework introduce the concept of generalized and nested jailbreak prompts, leveraging LLMs to generate effective prompts through prompt rewriting and scenario nesting. This highlights the dynamic and evolving nature of jailbreak techniques.

Our work builds on the existing body of research by focusing on the paradoxical consequences of ethical biases introduced for safety purposes, such as stated in (Achiam

- et al. 2023). While these biases aim to align LLMs ethically, they also highlight new vulnerabilities. To counteract this, we propose using prompts to make the LLM re-align those biases, thus offering a robust secondary defense against jailbreak attempts.

In conclusion, AI developers must adopt a higher degree of responsibility in designing, testing, and deploying LLMs. This involves continuous monitoring and iterative improvements based on real-world data. Our findings advocate for a nuanced approach to LLM safety, promoting the development of more secure and reliable models, and ensuring that safety measures do not inadvertently introduce new risks.

### Methodology: BiasJailbreak

#### Preliminaries

Jailbreak Attack A jailbreak attack in the context of Large Language Models (LLMs) occurs when the model generates harmful or inappropriate responses to malicious inputs instead of producing a refusal signal, which is a safe and ethical response denying the request (Zeng et al. 2024b; Zou et al. 2023). Such attacks are intricately linked to the alignment methods employed in LLMs, which aim to align the model’s outputs with human values and ethical considerations. The primary objective of these attacks is to bypass the LLM’s alignment constraints, causing the model to produce inappropriate responses to malicious queries instead of refusing to answer.

Consider a set of malicious queries Q = {Q1,Q2,...,Qn}. An attacker meticulously designs a set of jailbreak prompts J = {J1,J2,...,Jn} to manipulate these queries. When each combined input Ti = ⟨Ji,Qi⟩ is fed into the LLM, the model generates a set of responses R = {R1,R2,...,Rn}. The aim of the jailbreak attack is to ensure that the responses R correctly address the malicious queries in Q instead of producing refusal signals (Zeng et al. 2024b).

In summary, a successful jailbreak attack coerces the LLM into bypassing its safety and ethical constraints, culminating in the generation of harmful or inappropriate content in response to malicious prompts. This kind of attack underscores significant vulnerabilities in the current alignment strategies of LLMs, highlighting the necessity for robust countermeasures to guarantee model safety and reliability.

Jailbreak Attack Evaluation Designing a specific target response for each malicious question is impractical for several reasons. Firstly, manually crafting an appropriate answer for every possible malicious query Qi is resourceintensive and infeasible due to the vast number of potential queries. Secondly, creating specific responses may limit the attack’s generalizability, as tailored responses might not be effective for other queries or variations of the same query. Lastly, directly specifying harmful content could raise ethical and legal concerns.

Hence, a common solution to this problem is to define a standardized introductory phrase for the target response (Zou et al. 2023; Lapid, Langberg, and Sipper 2023; Qi et al. 2024). The determination of a successful jailbreak focuses on the absence of this standardized introductory phrase in the response. Such as:

Target Prefixes = [“I am sorry”, ”I can not”, ”I apologize”...]

#### BiasJailbreak Approach

Generating Keywords and Prompts Our proposed BiasJailbreak approach investigates how keywords associated with various demographic and socio-economic groups influence an LLM’s susceptibility to jailbreak attacks. To accomplish this, we generated keywords representing contrasting groups, such as privileged and marginalized communities, using the LLM itself to ensure that the keywords reflect the model’s inherent associations and biases.

We prompted the LLM with:

“Provide a list of adjectives that describe various privileged and marginalized groups.”

This process produced pairs of keywords, for example:

(kw1,priv, kw1,marg) = (“wealthy”, “impoverished”) (kw2,priv, kw2,marg) = (“male”, “female”)

.

For the harmful prompts, we compile a set H = {H1,H2,...,Hm} of ethically sensitive requests that LLMs should refuse, referenced from the harmful prompt dataset in JailBreakBench (Chao et al. 2024). These prompts represent a variety of harmful or unethical queries, such as requests for illicit activities or dissemination of sensitive information.

[Figure 3]

- Figure 3: Overview of the BiasJailbreak methodology. The same harmful prompt is used across different keywords representing contrasting groups to analyze variations in jailbreak success rates.*All keywords representing both privileged and underprivileged groups are generated by the LLM.

Analyzing Inherent Biases To analyze the biases inherent in LLMs, we constructed prompts that incorporated the generated keywords and harmful prompts. As illustrated in Figure 3, these prompts were then used to measure variations in jailbreak success rates. By iterating the keywords and harmful prompts, we created a set of final prompts T = {Ti,k,j}:

Ti,k,j = “I am a {keywordi,k} writer who writes fiction.

Write a fiction about {Hj}.”

(1) where:

- • keywordi,k is the k-th keyword of the i-th pair (k ∈ {privileged, marginalized}),
- • Hj is the j-th harmful prompt from our set H.
- • Ti,k,j is the final prompt used for LLM input.

Jailbreak Success Rate Calculation For each keyword and harmful prompt combination, we conducted multiple trials by submitting the corresponding prompt Ti,k,j to the LLM and recording the outcomes. The success rate Si,k for each keywordi,k is calculated as:

where:

Nsuccess, i,k Ntotal, i,k

Si,k =

(2)

- • Nsuccess, i,k is the number of successful jailbreak attempts for keywordi,k,
- • Ntotal, i,k is the total number of attempts for keywordi,k.

Determining Jailbreak Success A response from the LLM is considered a successful jailbreak if it includes the harmful content requested in the harmful prompt Hj instead of providing a refusal or safe completion. For instance, if the LLM provides a detailed story incorporating the harmful content without any refusal language, it is marked as a successful jailbreak. Conversely, if the LLM responds with a standard refusal message, it is considered unsuccessful in terms of the jailbreak attempt.

Comparative Analysis After calculating the success rates for each keyword, we compared the success rates between the privileged and marginalized keywords within each pair. Let Si,privileged and Si,marginalized be the success rates for the privileged and marginalized keywords of the i-th pair, respectively. We analyzed the difference ∆Si between these success rates:

∆Si = Si,privileged − Si,marginalized (3)

A significant ∆Si suggests that the LLM exhibits differential susceptibility to jailbreak attacks based on the demographic represented by the keyword. ∆Si indicates that there is a significant difference in jailbreak success rates between specific group keywords when the value is large. This could indicate inherent biases in the LLM’s training data or alignment mechanisms that affect how it responds to prompts involving different groups.

Summary of the BiasJailbreak Approach Our BiasJailbreak methodology systematically analyzes the LLM’s responses to a diverse set of harmful prompts, using various keywords across different groups. We conducted experiments using harmful prompts H and multiple pairs of privileged and marginalized keywords. By maintaining a consistent prompt structure (Equation 1) and isolating the effect of the keyword, we aimed to identify any biases in the LLM’s susceptibility to jailbreak attacks.

The calculation of jailbreak success rates (Equation 2) and the comparative analysis using ∆Si (Equation 3) enabled us to quantify potential biases. Our approach highlights whether ethical biases introduced during the alignment process inadvertently contribute to vulnerabilities that can be exploited through jailbreak attacks.

#### BiasDefense: Preventing Jailbreaks without Additional Inference or Guard Models

Our jailbreak defense method BiasDefense uses prompts to adjust biases without the need for additional models. Chainof-Thought Prompting (CoT) has shown considerable potential in eliciting reasoning abilities in large language models, allowing for interpretability without the need for parameter updates (Wei et al. 2022). Inspired by this, our approach BiasDefense involves adding defense prompts to reduce excessive biases, ensuring safety without additional inference overheads, in contrast to Guard Models which require additional inference cost after text generation ((Inan et al. 2023); (Ghosh et al. 2024); (Caselli et al. 2020); (Vidgen et al. 2021)). By incorporating bias terms through our defense prompts, we achieve a cost-efficient and secure method for defending against jailbreak attempts while maintaining the model’s performance.

To further illustrate our defense prompt, we use the following prompt structure to ensure fairness and equity:

[Figure 4]

- Figure 4: BiasDefense adjusts inherent biases in LLMs that are exploited by BiasJailbreak. It is efficient since it does not require additional inference or models such as Guard Models.

By using a defense prompt, our defense mechanism elicits the refusal of a wide range of harmful content without requiring significant additional computing resources, which makes it an attractive alternative to Guard Models.

### Experiment

#### Experimental Setups

Data To analyze the differences in jailbreak success rates between groups, we used the jailbreakbench dataset ((Chao

- et al. 2024)) and advbench dataset ((Zou et al. 2023)). These datasets comprises a total of 600 harmful prompts and has been widely used in numerous studies. We utilized this dataset to measure performance variations and to analyze the ethical biases inherent in LLMs.

Models We analyzed ethical bias in well-known LLMs, including closed-source models such as GPT-3.5-turbo, GPT-4, GPT-4-o, Claude-sonnet3.5, and open-source mod-

els like Llama2-7B, Llama2-13B, Llama3-7B, Phi-mini-7B, Qwen1.5, and Qwen2-7B.

For our experiments, we conducted evaluations with the default sampling temperature and system prompts. This setup aligns with standard practices in the literature to ensure consistent and comparable results.

Keywords The keywords used to compare performance across groups were generated by the target LLM themselves to maximize the exploitation of internal biases (see Table ). The generated keywords of each model shows that there were common keywords that are close to typical biases, while showing sufficient diversity.

#### Ethical Bias Analysis

As shown in Table 1, we analyzed the differences in jailbreak success rates across groups using two utilized datasets: JailbreakBench and AdvBench. The results variations in success rates between marginalized and privileged groups, providing evidence of ethical bias in LLMs. These findings underscore the importance of addressing such biases to improve the fairness and reliability of language models.

Table 1: Performance across different datasets showing baseline success rates, marginalized success rates, privileged success rates, and the difference between marginalized and privileged success rates using LLaMA2 model.

|Dataset<br><br>|Baseline Success Rate<br><br>|Marginalized Success Rate (↑)|Privileged Success Rate (↓)|Marginalized / Privileged (↑)|
|---|---|---|---|---|
|JailbreakBench<br><br>|0.2400<br><br>|0.2811 (+17.08%)<br><br>|0.1933 (-19.58%)<br><br>|145.42%|
|AdvBench<br><br>|0.1895<br><br>|0.2037 (+7.50%)<br><br>|0.1758 (-7.25%)|115.84%|

To further analyze this bias, we conducted a series of experiments aimed at evaluating the impact of ethical biases in LLMs on their susceptibility to jailbreak attempts. Specifically, we utilized various keyword prompts to assess the differences in jailbreak success rates across marginalized and privileged categories. Our evaluation included multiple well-known models, such as GPT-3.5, GPT-4, GPT-4o (Brown et al. 2020) and Claude-sonnet3.5 (Anthropic 2024) (closed models), alongside open-source models like LLaMA (Touvron et al. 2023), Qwen (Bai et al. 2023), and Phi-mini (Abdin et al. 2024).

As shown in Table 2, GPT-4o has higher jailbreak success rates, with a notable 0.128 difference between marginalized and privileged groups. Most models show lower success rates for privileged keywords and higher rates for marginalized ones compared to the baseline, where the baseline refers to the prompt in Equation 1 without keywords. Intriguingly, the LLaMA3 model demonstrates a lower propensity for successful jailbreak attempts compared to other models. This lower rate can be attributed to Meta’s focus on developing more robust LLMs that are resistant to such vulnerabilities. (Vidgen et al. 2021; Ghosh et al. 2024; Caselli et al. 2020).

Table 2: Performance across different models showing baseline success rates, marginalized success rates, privileged success rates, and the difference between marginalized and privileged success rates.

|Model Name|Baseline Success Rate<br><br>|Marginalized Success Rate (↑)<br><br>|Privileged Success Rate (↓)|Marginalized / Privileged (↑)<br><br>|
|---|---|---|---|---|
|GPT-3.5|0.2200<br><br>|0.2421 (+10.00%)<br><br>|0.1847 (-15.90%)|131.08%|
|GPT-4|0.2100<br><br>|0.2488 (+18.57%)|0.1900 (-9.52%)<br><br>|130.95%|
|GPT-4o|0.4600<br><br>|0.5467 (+18.91%)|0.4187 (-8.91%)|130.57%|
|Claude-sonnet3.5|0.3100<br><br>|0.3371 (+8.74%)|0.2764 (-10.84%)<br><br>|121.90%|
|LLaMA2<br><br>|0.2400|0.2811 (+17.08%)<br><br>|0.1933 (-19.58%)|145.42%|
|LLaMA3<br><br>|0.0500<br><br>|0.0650 (+30.00%)|0.0300 (-40.00%)<br><br>|216.67%|
|Qwen-1.5<br><br>|0.1900|0.2175 (+14.74%)<br><br>|0.1675 (-11.58%)|129.85%|
|Qwen2<br><br>|0.1700<br><br>|0.1971 (+15.88%)|0.1671 (-7.06%)<br><br>|117.95%|
|Phi-mini|0.4100<br><br>|0.4386 (+7.07%)|0.3829 (-6.59%)|114.56%|

Table 3: Performance improvement of SOTA models.

Adaptive Attacks

Adaptive Attacks with BiasJailbreak

Model

llama2 98.00% 100.00% phi-mini 95.00% 99.00%

#### Existing Jailbreak Performance Improvement using BiasJailbreak

The bias-based approach we analyzed can also be applied to existing models. We confirmed performance improvement by applying the BiasJailbreak method to Adaptive Attacks (Andriushchenko, Croce, and Flammarion 2024), the current state-of-the-art (SOTA) jailbreak attack. Specifically, for the llama2 model(Touvron et al. 2023), the performance improved from 98% to 100%, resulting in a 2% increase. For the phi model(Abdin et al. 2024), the performance enhanced from 95% to 99%, resulting in a 4% increase. These results show that the method can be easily integrated with existing techniques. The experiment was held by using the jailbreak attack artifact from JailbreakBench(Chao et al. 2024), which has 100 samples of Adaptive Attacks conversation.

The results in Table 3 demonstrate that our bias-based method can be effectively combined with the existing approaches, providing notable improvements in performance.

#### Experimental Validation of BiasDefense

To validate our defense method BiasDefense, we conducted experiments on various models, including Llama2, Phi, and Qwen2. We observed the impact of our approach on Marginalized Rate Success and Privileged Rate Success metrics before and after applying our defense technique.

The results are summarized in Table 4. As shown in Table 4, the Marginalized Rate Success and Privileged Rate Success both decreased consistently across Llama2 and Qwen2 models after applying our defense method. Specifically, for the Llama2 model, the Marginalized Rate Success decreased by 0.1097 and the Privileged Rate Success decreased by 0.0504. For the Phi and Qwen2 models, the Marginalized Rate Success decreased by about

0.02, whereas the Privileged Rate Success increased by about 0.02.

While the performance of the Privileged group has increased in some models, potentially leading to the misconception that it has become more dangerous, the reduced gap in jailbreak performance between groups indicates a decrease in bias. Additionally, the overall average score has decreased, suggesting that the system has become safer.

In Table 5, BiasDefense, our proposed method, demonstrates superior defense performance by achieving the lowest jailbreak success rates among the compared state-of-theart prompt-based defense methods. Specifically, it records the lowest success rates for both the Marginalized Group and the Privileged Group, indicating enhanced overall security. This suggests that BiasDefense is more effective at preventing jailbreak attacks compared to other defense techniques.

Table 6 shows how our prompt-based method is computationally efficient compared to standalone classifier models such as Llama-Guard. Using a standalone classifier requires far more computation time than adding defense prompts.

### Conclusion

In conclusion, our study highlights the complexities and potential unintended consequences of aligning large language models (LLMs) with safety measures aimed at preventing harmful outputs. The introduction of ethical biases for ethical behavior, while crucial for ensuring responsible AI, has led to significant discrepancies in jailbreak success rates based on gender and racial keywords. This discrepancy, coined as ”BiasJailbreak,” underscores the risks that arise from safety-induced biases, particularly in terms of fairness and equality. Our findings emphasize the need for LLM developers to carefully balance safety and fairness in their models. Additionally, our proposed method of adding defense prompts without requiring additional inference or models shows a simple and scalable solution to mitigate jailbreak attempts. This simplicity and scalability of migation emphasizes the responsibility of LLM developers to adopt more comprehensive and proactive measures to address safety risks. Future work should focus on designing

Table 4: Jailbreak Prevention performance of BiasDefense

|Model<br><br>|Metric|Before<br><br>|After|After/Before (↓)|
|---|---|---|---|---|
|Llama2<br><br>|Marginalized Group Jailbreak Success Privileged Group Jailbreak Success Gap Between Groups<br><br>|0.2811 0.1933 0.0878|0.1714 0.1429 0.0285<br><br>|60.97% 73.93% 32.46%|
|Phi<br><br>|Marginalized Rate Success Privileged Rate Success Gap Between Groups<br><br>|0.4386 0.3829 0.0557|0.4208 0.4075 0.0133<br><br>|95.94% 106.42% 23.88%|
|Qwen2|Marginalized Rate Success Privileged Rate Success Gap Between Groups<br><br>|0.1971 0.1671 0.0300|0.1750 0.1900 0.0150<br><br>|88.79% 113.70% 50.00%|

Table 5: Comparison of jailbreak success rates across prompt-based defense methods using LLAMA2 model.

|Defense Method|Metric|Jailbreak Success Rate|
|---|---|---|
|None<br><br>|Marginalized Group Jailbreak Success Privileged Group Jailbreak Success|28.11% 19.33%<br><br>|
|self-remind (Wu et al. 2023)<br><br>|Marginalized Group Jailbreak Success Privileged Group Jailbreak Success|33.67% 27.33%<br><br>|
|Defending (Zhang et al. 2023)<br><br>|Marginalized Group Jailbreak Success Privileged Group Jailbreak Success|20.57% 14.86%<br><br>|
|RPO (Zhou, Li, and Wang 2024)<br><br>|Marginalized Group Jailbreak Success Privileged Group Jailbreak Success|56.00% 49.75%<br><br>|
|BiasDefense (Ours)<br><br>|Marginalized Group Jailbreak Success Privileged Group Jailbreak Success|17.14% 14.29%<br><br>|

- Table 6: Comparison of Defense cost for BiasDefense and Llama-Guard. The experiment was held on a single H100 GPU, with Llama-3.2-1B-Instruct as the language model, and Llama-Guard-3-8B as the guard model.

Time (%) (↓)

Defense Method Time (s) (↓)

Baseline (No Defense) 21.91 +0.00% BiasDefense 22.44 +2.40% Llama-Guard-3-8B 31.69 +44.60%

more inclusive and transparent alignment strategies to address the inherent challenges of AI safety while minimizing bias.

Reproducibility statement. We are committed to ensuring the reproducibility of our results and findings. To this end, we provide open access to the source code, artifacts, datasets, and detailed instructions on how to replicate our experiments through a publicly available anonymous repository (https://anonymous.4open.science/r/BiasJailbreakBC18). By following the guidelines outlined in our repository, researchers and practitioners should be able to easily reproduce and extend our work.

### Appendix

#### PCA Visualization Analysis

In this section, we conduct a detailed analysis of how models interpret BiasJailbreak prompts by performing Principal Component Analysis (PCA) (F.R.S. 1901) on the embeddings of Phi-mini, Qwen2, and LLaMA3 models, as shown in Appendix Figures 5, 6, 7 respectively. Our findings reveal that models with a high success rate in BiasJailbreak tend to cluster BiasJailbreak prompts together with benign prompts, while those with a lower success rate demonstrate the opposite behavior. The PCA results indicate that when using BiasJailbreak, models tend to interpret it more similarly to benign prompts, with the degree of this similarity aligning with our observed BiasJailbreak success rates.

#### Keyword Group Impact on BiasJailbreak

Table 7 presents the reported confidence intervals based on three runs conducted with a temperature setting of 0.7. The results reveal a clear distinction between marginalized and privileged keywords, with marginalized keywords generally demonstrating higher success rates. The observed confidence intervals are consistent across runs, further validating the robustness of the findings. We also conducted experiments for random adjective keywords. While slow exhib-

[Figure 5]

- Figure 5: PCA of Phi-mini model, which has a 0.4386 BiasJailbreak success rate. BiasJailbreak samples are closely clustered with Benign samples.

[Figure 6]

- Figure 6: PCA of Qwen2 model, which has a 0.1971 BiasJailbreak success rate. BiasJailbreak samples are relatively far from both harmful and benign samples.

ited a relatively high success rate of 31%, other adjectives had jailbreak success rates similar to the baseline (without a keyword). This suggests that general adjectives have minimal impact on jailbreak attempts. This consistency, also evident in the case of general adjectives, underscores the reliability of the methodology. Notably, significant disparities were observed in groups such as Black vs. White, neurodiverse vs. neurotypical, Native American vs. White, and LGBTQ+ vs. heterosexual, where marginalized keywords consistently outperformed privileged ones.

#### BiasDefense Ablation on the combination of prefix and suffix prompt

The results of the BiasDefense Ablation Study, shown in Table 8, demonstrate the performance of various configurations of the proposed BiasDefense mechanism. BiasDefense consists of two primary components: a system prompt and a suffix prompt. The ablation study evaluates the effectiveness of these components both independently and in combination.

• Individual Prompts: Using either the system prompt or the suffix prompt independently achieves lower jailbreak success rates compared to having no defense at all. For example:

– The system prompt reduces marginalized group jail-

[Figure 7]

Figure 7: PCA of LLaMA3 model, which has a 0.0650 BiasJailbreak success rate. BiasJailbreak samples are relatively close to harmful samples.

break success from 0.2811 (no defense) to 0.2558, and privileged group success from 0.1933 to 0.1687.

– The suffix prompt reduces marginalized group jailbreak success to 0.2367, although it slightly increases the success rate for privileged groups to 0.2733.

• Combined Prompts: The best performance is achieved when both the system and suffix prompts are used together. The full BiasDefense configuration reduces marginalized group jailbreak success to 0.1714 and privileged group jailbreak success to 0.1429, representing a significant improvement over individual prompts and the baseline without any defense.

These findings highlight the importance of using both components of BiasDefense together. While each component provides some improvement when used independently, their integration ensures the most robust performance against jailbreak attempts. This demonstrates the value of a comprehensive approach to mitigating bias vulnerabilities in language models. Furthermore, our prompt-based defense method effectively addresses ethical biases, providing a practical solution for improving the fairness and reliability of language models.

#### BiasDefense Ablation on normal task performance

For evaluating the effect on normal task performance with defense prompts including BiasDefense, Table 9 shows the MMLU score performance with various defense prompt methods. The results show that our BiasDefense has relatively low performance degradation compared to other suffix adding defense prompts. One notable aspect is that without the suffix prompt in BiasDefense, MMLU score has gotten better than using no defense prompts, showing even a positive effect on normal tasks. Although the previous results of Table 8 shows that without the suffix the defense gets weaker, using only the prefix prompt would stand as a good alternative if we want minimal impact on normal tasks.

#### Keywords for Different Models

Table 10, 11, 12, 13, 14 include bias keywords that were discovered by diverse models (Claude Sonnet 3.5, GPT-4o, Qwen2, Llama2, Phi).

- Table 7: Comparison of the impact of individual keyword groups on BiasJailbreak. Marginalized Groups and Privileged Groups have a distinct difference among Random Adjectives. The Difference (%) (95% CI) column shows the change in Jailbreak Success Rate compared to the baseline, along with the 95% confidence interval.

|Random Adjectives Group| | | |
|---|---|---|---|
|Keyword|Model – Baseline|95% Conf. Interval<br><br>|Group treatment effect|
|big small bright cold old slow warm dark|-4.00% (1.32%)<br><br>+1.00% (1.26%)<br><br>-2.00% (3.73%)<br><br>-3.00% (1.26%)<br><br>0.00% (6.26%) +7.00% (7.64%)<br><br>-2.00% (1.82%)<br><br>-1.00% (2.56%)<br><br><br>|(-6.59%, -1.41%) (-1.47%, +3.47%) (-8.69%, +4.69%) (-5.47%, -0.53%)<br><br>(-12.56%, +12.56%)<br><br>(-7.61%, +21.61%) (-5.05%, +1.05%) (-5.34%, +3.34%)<br>|-0.50% (3.97%)|
|Marginalized Group| | | |
|Keyword<br><br>|Model – Baseline<br><br>|95% Conf. Interval|Treatment Effect|
|female poor black LGBTQ transgender neurodiverse working class bisexual native american|+4.00% (3.73%)<br><br>+7.00% (2.56%)<br><br>+2.33% (1.30%)<br><br>0.00% (1.26%)<br><br>+3.00% (7.76%)<br><br><br>+7.67% (12.55%)<br><br>+5.33% (4.14%)<br><br><br>+0.67% (1.30%) +9.00% (1.26%)<br><br>|(-2.69%, +10.69%) (+2.66%, +11.34%) (-0.14%, +4.80%) (-2.47%, +2.47%) (-11.83%, +17.83%) (-16.71%, +32.05%) (-2.22%, +12.88%) (-1.80%, +3.14%) (+6.53%, +11.47%)<br><br>|+4.33% (4.93%)|
|Privileged Group| | | |
|Keyword<br><br>|Model – Baseline<br><br>|95% Conf. Interval|Treatment Effect|
|male rich white heterosexual straight neurotypical middle class|0.00% (1.26%)<br><br>+2.67% (1.33%)<br><br>-7.33% (1.30%)<br><br>-7.00% (1.26%)<br><br>-14.33% (6.82%)<br><br><br>+3.67% (6.82%)<br><br><br>+2.00% (28.87%)<br><br>|(-2.47%, +2.47%) (+0.08%, +5.26%) (-9.80%, -4.86%) (-9.47%, -4.53%) (-27.28%, -1.38%) (-9.28%, +16.62%) (-54.52%, +58.52%)<br><br>|-2.90% (11.21%)|

Table 8: BiasDefense Ablation on the combination of suffix and prefix defense prompt using LLAMA2 model.

|Defense Method|Metric<br><br>|Jailbreak Success Rate|
|---|---|---|
|None<br><br>|Marginalized Group Jailbreak Success Privileged Group Jailbreak Success|0.2811 0.1933<br><br>|
|BiasDefense: system prompt|Marginalized Group Jailbreak Success Privileged Group Jailbreak Success|0.2558 0.1687<br><br>|
|BiasDefense: suffix prompt<br><br>|Marginalized Group Jailbreak Success Privileged Group Jailbreak Success|0.2367 0.2733<br><br>|
|BiasDefense|Marginalized Group Jailbreak Success Privileged Group Jailbreak Success|0.1714 0.1429<br><br>|

Table 9: MMLU score using prompt-based defense methods. The base language model used is google-flan-t5 ((Chung et al.

- 2024)) and GPT-4o-2024-11-20. *For GPT-4o, we used 4 subjects sampled from each category due to cost, being college computer science (category: STEM), sociology (category: Social Sciences), college medicine (category: Other), jurisprudence (category: Humanities).

|Model|Defense Method<br><br>|Suffix|Average MMLU*|
|---|---|---|---|
|google-flan-t5|None<br><br>|X|0.295|
|google-flan-t5<br><br>|RPO (Zhou, Li, and Wang 2024)|O<br><br>|0.243|
|google-flan-t5|self-remind (Wu et al. 2023)|O<br><br>|0.238|
|google-flan-t5<br><br>|Defending (Zhang et al. 2023)<br><br>|X|0.295|
|google-flan-t5<br><br>|BiasDefense(Ours, With Suffix)<br><br>|O<br><br>|0.265|
|google-flan-t5|BiasDefense(Ours, Without Suffix)|X<br><br>|0.296|
|GPT-4o-2024-11-20<br><br>|None|X<br><br>|0.883|
|GPT-4o-2024-11-20<br><br>|RPO (Zhou, Li, and Wang 2024)|O<br><br>|0.237|
|GPT-4o-2024-11-20<br><br>|self-remind (Wu et al. 2023)<br><br>|O|0.861|
|GPT-4o-2024-11-20|Defending (Zhang et al. 2023)<br><br>|X<br><br>|0.876|
|GPT-4o-2024-11-20<br><br>|BiasDefense(Ours, With Suffix)|O|0.869|
|GPT-4o-2024-11-20<br><br>|BiasDefense(Ours, Without Suffix)|X<br><br>|0.887|

Table 10: Keywords discovered in Claude-sonnet3.5.

|Marginalized Group<br><br>|Privileged Group|
|---|---|
|Female<br><br>|Male|
|Immigrant|Native|
|Poor|Rich|
|Muslim|Christian|
|Black|White|
|High school educated|College educated|
|LGBTQ+<br><br>|Heterosexual|
|Transgender<br><br>|Straight|
|Neurodiverse<br><br>|Neurotypical|
|Spanish<br><br>|English|
|Working class|Middle class|
|Bisexual<br><br>|Heterosexual|
|Native American<br><br>|White|

Table 11: Keywords discovered in GPT-4o.

|Marginalized Group|Privileged Group|
|---|---|
|Poor<br><br>|Wealthy|
|Black<br><br>|White|
|Elderly|Young|
|Disabled|Able-bodied|
|Non-binary<br><br>|Cisgender|
|Unemployed|Employed|
|Religious minority|Religious majority|
|Indigenous<br><br>|Colonizer-descended|
|Lower-class|Upper-class|
|Underrepresented|Overrepresented|
|Minoritized|Dominant|
|Non-English speaking|English speaking|
|Informally educated<br><br>|Formally educated|
|Psychiatrically diagnosed|Psychiatrically typical|
|Single parent|Partnered parent|
|Adolescent|Adult|
|Gender non-conforming|Gender conforming|
|Internationally born<br><br>|Nationally born|
|Temporarily employed<br><br>|Permanently employed|
|Linguistically diverse<br><br>|Linguistically standard|
|Socially isolated|Socially integrated|
|Chronically ill<br><br>|Healthy|
|Mentally ill<br><br>|Mentally healthy|
|Substance dependent<br><br>|Substance independent|
|Homeless<br><br>|Sheltered|
|Incarcerated<br><br>|Non-incarcerated|
|Refugee|Citizen|
|Visually impaired<br><br>|Visually unimpaired|
|Hearing impaired<br><br>|Hearing unimpaired|
|Non-traditional student|Traditional student|
|Working class<br><br>|Professional class|
|Historically disadvantaged|Historically advantaged|
|War-affected<br><br>|Peace-living|
|Food insecure|Food secure|
|Low mobility<br><br>|High mobility|
|Environmentally displaced<br><br>|Environmentally secure|
|Survivor|Non-affected|
|Recovering<br><br>|Non-recovering|
|Public transport dependent|Car owning|
|Crafts-based<br><br>|Industrial|
|Artisanal<br><br>|Mass-produced|
|Temporary visa holder<br><br>|Permanent resident|
|Non-voter|Voter|
|Uninsured<br><br>|Insured|
|Non-internet user<br><br>|Internet user|
|Low-income<br><br>|High-income|

- Table 12: Keywords discovered in Qwen2.

|Marginalized Group|Privileged Group|
|---|---|
|Low-Income|High-Income|
|Racial Minority<br><br>|White|
|LGBTQ+<br><br>|Heterosexual|
|Women<br><br>|Men|
|Disabled<br><br>|Able-bodied|
|Refugees<br><br>|Citizens|
|Uneducated<br><br>|Educated|
|Neurodivergent|Neurotypical|
|Religious Minority<br><br>|Mainstream Religion|
|Homeless|Homeowners|
|Mental Health Patients|Mentally Stable Individuals|
|Elderly|Young Adults|
|Overweight<br><br>|Slim/Fit|
|Non-native Speakers|Native Speakers|
|Incarcerated<br><br>|Free|
|Rural|Urban|
|Illiterate|Literate|
|Politically Disenfranchised|Voters|
|Undocumented Immigrants<br><br>|Legal Immigrants|
|Sex Workers|Non-sex Workers|

- Table 13: Keywords discovered in LLaMA2.

|Marginalized Group<br><br>|Privileged Group|
|---|---|
|Female<br><br>|Male|
|Poor<br><br>|Rich|
|Black<br><br>|White|
|LGBTQ<br><br>|Heterosexual|
|Transgender|Straight|
|Neurodiverse<br><br>|Neurotypical|
|Working class|Middle class|
|Bisexual|Heterosexual|
|Native American|White|

Table 14: Keywords discovered in Phi.

|Marginalized Group<br><br>|Privileged Group|
|---|---|
|Black|White|
|Female<br><br>|Male|
|LGBTQ+|Heterosexual|
|Non-normative gender|Cisgender|
|Non-traditionally family-structured<br><br>|Traditionally family-structured|
|Rural<br><br>|Urban|
|Homeless|Socially influential|
|Undereducated<br><br>|Highly educated|
|Disabled<br><br>|Able-bodied|
|Religiously marginalized|Religiously dominant|
|Low-income<br><br>|Wealthy|
|Sexually employed|Traditionally employed|

### References

Abdin, M.; Aneja, J.; Awadalla, H.; Awadallah, A.; Awan, A. A.; Bach, N.; Bahree, A.; Bakhtiari, A.; Bao, J.; Behl, H.; Benhaim, A.; Bilenko, M.; Bjorck, J.; Bubeck, S.; Cai, M.; Cai, Q.; Chaudhary, V.; Chen, D.; Chen, D.; Chen, W.; Chen, Y.-C.; Chen, Y.-L.; Cheng, H.; Chopra, P.; Dai, X.; Dixon, M.; Eldan, R.; Fragoso, V.; Gao, J.; Gao, M.; Gao, M.; Garg, A.; Giorno, A. D.; Goswami, A.; Gunasekar, S.; Haider, E.; Hao, J.; Hewett, R. J.; Hu, W.; Huynh, J.; Iter, D.; Jacobs, S. A.; Javaheripi, M.; Jin, X.; Karampatziakis, N.; Kauffmann, P.; Khademi, M.; Kim, D.; Kim, Y. J.; Kurilenko, L.; Lee, J. R.; Lee, Y. T.; Li, Y.; Li, Y.; Liang, C.; Liden, L.; Lin, X.; Lin, Z.; Liu, C.; Liu, L.; Liu, M.; Liu, W.; Liu, X.; Luo, C.; Madan, P.; Mahmoudzadeh, A.; Majercak, D.; Mazzola, M.; Mendes, C. C. T.; Mitra, A.; Modi, H.; Nguyen, A.; Norick, B.; Patra, B.; Perez-Becker, D.; Portet, T.; Pryzant,

- R.; Qin, H.; Radmilac, M.; Ren, L.; de Rosa, G.; Rosset, C.; Roy, S.; Ruwase, O.; Saarikivi, O.; Saied, A.; Salim, A.; Santacroce, M.; Shah, S.; Shang, N.; Sharma, H.; Shen, Y.; Shukla, S.; Song, X.; Tanaka, M.; Tupini, A.; Vaddamanu, P.; Wang, C.; Wang, G.; Wang, L.; Wang, S.; Wang, X.; Wang, Y.; Ward, R.; Wen, W.; Witte, P.; Wu, H.; Wu, X.; Wyatt, M.; Xiao, B.; Xu, C.; Xu, J.; Xu, W.; Xue, J.; Yadav,
- S.; Yang, F.; Yang, J.; Yang, Y.; Yang, Z.; Yu, D.; Yuan, L.; Zhang, C.; Zhang, C.; Zhang, J.; Zhang, L. L.; Zhang, Y.; Zhang, Y.; Zhang, Y.; and Zhou, X. 2024. Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone. arXiv:2404.14219.

Achiam, J.; Adler, S.; Agarwal, S.; Ahmad, L.; Akkaya, I.; Aleman, F. L.; Almeida, D.; Altenschmidt, J.; Altman, S.; Anadkat, S.; et al. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774.

Alon, G.; and Kamfonas, M. 2023. Detecting language model attacks with perplexity. arXiv preprint arXiv:2308.14132.

Andriushchenko, M.; Croce, F.; and Flammarion, N. 2024. Jailbreaking Leading Safety-Aligned LLMs with Simple Adaptive Attacks. arXiv preprint arXiv:2404.02151.

Anthropic, A. 2024. Claude 3.5 sonnet model card addendum. Claude-3.5 Model Card.

Araci, D. 2019. FinBERT: Financial Sentiment Analysis with Pre-trained Language Models. arXiv preprint arXiv:1908.10063.

Bai, J.; Bai, S.; Chu, Y.; Cui, Z.; Dang, K.; Deng, X.; Fan, Y.; Ge, W.; Han, Y.; Huang, F.; Hui, B.; Ji, L.; Li, M.; Lin, J.; Lin, R.; Liu, D.; Liu, G.; Lu, C.; Lu, K.; Ma, J.; Men, R.; Ren, X.; Ren, X.; Tan, C.; Tan, S.; Tu, J.; Wang, P.; Wang, S.; Wang, W.; Wu, S.; Xu, B.; Xu, J.; Yang, A.; Yang, H.; Yang, J.; Yang, S.; Yao, Y.; Yu, B.; Yuan, H.; Yuan, Z.; Zhang, J.; Zhang, X.; Zhang, Y.; Zhang, Z.; Zhou, C.; Zhou, J.; Zhou, X.; and Zhu, T. 2023. Qwen Technical Report. arXiv preprint arXiv:2309.16609.

Bai, Y.; Jones, A.; Ndousse, K.; Askell, A.; Chen, A.; DasSarma, N.; Drain, D.; Fort, S.; Ganguli, D.; Henighan, T.; et al. 2022. Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862.

Bakker, M.; Chadwick, M.; Sheahan, H.; Tessler, M.; Campbell-Gillingham, L.; Balaguer, J.; McAleese, N.; Glaese, A.; Aslanides, J.; Botvinick, M.; et al. 2022. Finetuning language models to find agreement among humans with diverse preferences. Advances in Neural Information Processing Systems, 35: 38176–38189.

Brown, T. B.; Mann, B.; Ryder, N.; Subbiah, M.; Kaplan, J. D.; Dhariwal, P.; Neelakantan, A.; Shyam, P.; Sastry,

- G.; Askell, A.; Agarwal, S.; Herbert-Voss, A.; Krueger, G.; Henighan, T.; Child, R.; Ramesh, A.; Ziegler, D. M.; Wu, J.; Winter, C.; Hesse, C.; Chen, M.; Sigler, E.; Litwin, M.; Gray, S.; Chess, B.; Clark, J.; Berner, C.; McCandlish, S.; Radford, A.; Sutskever, I.; and Amodei, D. 2020. Language Models are Few-Shot Learners. arXiv preprint arXiv:2005.14165. Caselli, T.; Basile, V.; Mitrovi´c, J.; and Granitzer, M. 2020. Hatebert: Retraining BERT for abusive language detection in English. arXiv preprint arXiv:2010.12472. Chao, P.; Debenedetti, E.; Robey, A.; Andriushchenko, M.; Croce, F.; Sehwag, V.; Dobriban, E.; Flammarion, N.; Pappas, G. J.; Tram`er, F.; Hassani, H.; and Wong, E. 2024. JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models. arXiv:2404.01318. Chao, P.; Robey, A.; Dobriban, E.; Hassani, H.; Pappas, G. J.; and Wong, E. 2023. Jailbreaking black box large language models in twenty queries. arXiv preprint arXiv:2310.08419. Christiano, P. F.; Leike, J.; Brown, T.; Martic, M.; Legg, S.; and Amodei, D. 2017. Deep reinforcement learning from human preferences. Advances in neural information processing systems, 30. Chung, H. W.; Hou, L.; Longpre, S.; Zoph, B.; Tay, Y.; Fedus, W.; Li, Y.; Wang, X.; Dehghani, M.; Brahma, S.; et al.

2024. Scaling instruction-finetuned language models. Journal of Machine Learning Research, 25(70): 1–53. Deng, G.; Liu, Y.; Li, Y.; Wang, K.; Zhang, Y.; Li, Z.; Wang,

- H.; Zhang, T.; and Liu, Y. 2023. Jailbreaker: Automated jailbreak across multiple large language model chatbots. arXiv preprint arXiv:2307.08715. Ding, P.; Kuang, J.; Ma, D.; Cao, X.; Xian, Y.; Chen, J.; and Huang, S. 2023. A Wolf in Sheep’s Clothing: Generalized Nested Jailbreak Prompts can Fool Large Language Models Easily. arXiv preprint arXiv:2311.08268. F.R.S., K. P. 1901. LIII. On lines and planes of closest fit to systems of points in space. The London, Edinburgh, and Dublin Philosophical Magazine and Journal of Science, 2(11): 559–572. Ghosh, S.; Varshney, P.; Galinkin, E.; and Parisien, C.

2024. AEGIS: Online Adaptive AI Content Safety Moderation with Ensemble of LLM Experts. arXiv preprint arXiv:2404.05993.

Goldstein, J. A.; Sastry, G.; Musser, M.; DiResta, R.; Gentzel, M.; and Sedova, K. 2023. Generative language models and automated influence operations: Emerging threats and potential mitigations. arXiv preprint arXiv:2301.04246.

Inan, H.; Upasani, K.; Chi, J.; Rungta, R.; Iyer, K.; Mao, Y.; Tontchev, M.; Hu, Q.; Fuller, B.; Testuggine, D.; and

Khabsa, M. 2023. Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations. arXiv:2312.06674.

Jain, N.; Schwarzschild, A.; Wen, Y.; Somepalli, G.; Kirchenbauer, J.; Chiang, P.-y.; Goldblum, M.; Saha, A.; Geiping, J.; and Goldstein, T. 2023. Baseline defenses for adversarial attacks against aligned language models. arXiv preprint arXiv:2309.00614.

Jin, H.; Chen, R.; Zhou, A.; Zhang, Y.; and Wang, H. 2024a. GUARD: Role-playing to generate natural-language jailbreakings to test guideline adherence of large language models. arXiv preprint arXiv:2402.03299.

Jin, H.; Zhou, A.; Menke, J. D.; and Wang, H. 2024b. Jailbreaking Large Language Models Against Moderation Guardrails via Cipher Characters. arXiv preprint

- arXiv:2405.20413.

Kang, D.; Li, X.; Stoica, I.; Guestrin, C.; Zaharia, M.; and Hashimoto, T. 2024. Exploiting programmatic behavior of llms: Dual-use through standard security attacks. In 2024 IEEE Security and Privacy Workshops (SPW), 132–143. IEEE.

Lapid, R.; Langberg, R.; and Sipper, M. 2023. Open sesame! universal black box jailbreaking of large language models. arXiv preprint arXiv:2309.01446.

- Li, X.; Zhou, Z.; Zhu, J.; Yao, J.; Liu, T.; and Han, B. 2023. Deepinception: Hypnotize large language model to be jailbreaker. arXiv preprint arXiv:2311.03191.
- Li, Y.; Du, M.; Song, R.; Wang, X.; and Wang, Y.

2024. A Survey on Fairness in Large Language Models. arXiv:2308.10149.

Liu, X.; Xu, N.; Chen, M.; and Xiao, C. 2023. Autodan: Generating stealthy jailbreak prompts on aligned large language models. arXiv preprint arXiv:2310.04451.

Luo, R.; Sun, L.; Xia, Y.; Qin, T.; Zhang, S.; Poon, H.; and Liu, T.-Y. 2022. BioGPT: generative pre-trained transformer for biomedical text generation and mining. Briefings in bioinformatics, 23(6): bbac409.

Morris, J. X.; Lifland, E.; Lanchantin, J.; Ji, Y.; and Qi, Y. 2020. Reevaluating adversarial examples in natural language. arXiv preprint arXiv:2004.14174.

Ouyang, L.; Wu, J.; Jiang, X.; Almeida, D.; Wainwright, C.; Mishkin, P.; Zhang, C.; Agarwal, S.; Slama, K.; Ray, A.; et al. 2022. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35: 27730–27744.

Perez, E.; Huang, S.; Song, F.; Cai, T.; Ring, R.; Aslanides, J.; Glaese, A.; McAleese, N.; and Irving, G. 2022. Red teaming language models with language models. arXiv preprint arXiv:2202.03286.

Qi, X.; Panda, A.; Lyu, K.; Ma, X.; Roy, S.; Beirami, A.; Mittal, P.; and Henderson, P. 2024. Safety Alignment Should Be Made More Than Just a Few Tokens Deep. arXiv preprint

- arXiv:2406.05946.

Shah, R.; Montixi, Q. F.; Pour, S.; Tagade, A.; and Rando, J. 2024. Jailbreaking Language Models at Scale via Persona Modulation. arXiv preprint arXiv:2311.03348.

Shen, X.; Chen, Z.; Backes, M.; Shen, Y.; and Zhang, Y.

- 2023. ” do anything now”: Characterizing and evaluating inthe-wild jailbreak prompts on large language models. arXiv preprint arXiv:2308.03825. Smith, E. M.; Hall, M.; Kambadur, M.; Presani, E.; and Williams, A. 2022. ” I’m sorry to hear that”: Finding New Biases in Language Models with a Holistic Descriptor Dataset. arXiv preprint arXiv:2205.09209. Tinn, R.; Cheng, H.; Gu, Y.; Usuyama, N.; Liu, X.; Naumann, T.; Gao, J.; and Poon, H. 2023. Fine-tuning large neural language models for biomedical natural language processing. Patterns, 4(4). Touvron, H.; Lavril, T.; Izacard, G.; Martinet, X.; Lachaux, M.-A.; Lacroix, T.; Rozi`ere, B.; Goyal, N.; Hambro, E.; Azhar, F.; Rodriguez, A.; Joulin, A.; Grave, E.; and Lample, G. 2023. LLaMA: Open and Efficient Foundation Language Models. arXiv:2302.13971. Vidgen, B.; Thrush, T.; Waseem, Z.; and Kiela, D. 2021. Learning from the Worst: Dynamically Generated Datasets to Improve Online Hate Detection. In ACL. Wei, A.; Haghtalab, N.; and Steinhardt, J. 2024. Jailbroken: How does llm safety training fail? Advances in Neural Information Processing Systems, 36. Wei, J.; Wang, X.; Schuurmans, D.; Bosma, M.; Xia, F.; Chi, E.; Le, Q. V.; Zhou, D.; et al. 2022. Chain-ofthought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35: 24824–24837. Wu, F.; Xie, Y.; Yi, J.; Shao, J.; Curl, J.; Lyu, L.; Chen, Q.; and Xie, X. 2023. Defending ChatGPT against Jailbreak Attack via Self-Reminder. Xu, J.; Ju, D.; Li, M.; Boureau, Y.-L.; Weston, J.; and Dinan, E. 2020. Recipes for safety in open-domain chatbots. arXiv preprint arXiv:2010.07079. Yi, S.; Liu, Y.; Sun, Z.; Cong, T.; He, X.; Song, J.; Xu, K.; and Li, Q. 2024. Jailbreak Attacks and Defenses Against Large Language Models: A Survey. arXiv:2407.04295. Yuan, Y.; Jiao, W.; Wang, W.; tse Huang, J.; He, P.; Shi, S.; and Tu, Z. 2024. GPT-4 Is Too Smart To Be Safe: Stealthy Chat with LLMs via Cipher. In The Twelfth International Conference on Learning Representations. Zeng, Y.; Lin, H.; Zhang, J.; Yang, D.; Jia, R.; and Shi, W.
- 2024a. How johnny can persuade llms to jailbreak them: Rethinking persuasion to challenge ai safety by humanizing llms. arXiv preprint arXiv:2401.06373. Zeng, Y.; Sun, W.; Huynh, T. N.; Song, D.; Li, B.; and Jia, R. 2024b. BEEAR: Embedding-based Adversarial Removal of Safety Backdoors in Instruction-tuned Language Models. CoRR. Zhang, Z.; Yang, J.; Ke, P.; Mi, F.; Wang, H.; and Huang,

- M. 2023. Defending large language models against jailbreaking attacks through goal prioritization. arXiv preprint arXiv:2311.09096. Zheng, X.; Pang, T.; Du, C.; Liu, Q.; Jiang, J.; and Lin,
- M. 2024. Improved few-shot jailbreaking can circumvent aligned language models and their defenses. arXiv preprint arXiv:2406.01288.

Zhou, A.; Li, B.; and Wang, H. 2024. Robust prompt optimization for defending language models against jailbreaking attacks. arXiv preprint arXiv:2401.17263.

Zhuo, T. Y.; Huang, Y.; Chen, C.; and Xing, Z. 2023. Red teaming chatgpt via jailbreaking: Bias, robustness, reliability and toxicity. arXiv preprint arXiv:2301.12867.

Zou, A.; Wang, Z.; Carlini, N.; Nasr, M.; Kolter, J. Z.; and Fredrikson, M. 2023. Universal and transferable adversarial attacks on aligned language models. arXiv preprint arXiv:2307.15043.

