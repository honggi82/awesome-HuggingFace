# arXiv:2503.00069v2[cs.CY]23Jun2026

## Societal Frameworks Can Improve LLM Alignment

KAROLINA STAŃCZAK, ETH Zurich, Switzerland and Mila, McGill University, Canada NICHOLAS MEADE, Mila, McGill University, Canada MEHAR BHATIA, Mila, McGill University, Canada HATTIE ZHOU∗, Mila, Université de Montréal, Canada and Anthropic, USA KONSTANTIN BÖTTINGER, Fraunhofer AISEC, Germany JEREMY BARNES, ServiceNow, Canada JASON STANLEY, ServiceNow, Canada JESSICA MONTGOMERY, University of Cambridge, UK RICHARD ZEMEL, Columbia University, USA NICOLAS PAPERNOT, University of Toronto, Google DeepMind, Canada NICOLAS CHAPADOS, Mila, ServiceNow, Canada DENIS THERIEN, McGill University, ServiceNow, Canada TIMOTHY P. LILLICRAP, Google DeepMind, UK ANA MARASOVIĆ, University of Utah, USA SYLVIE DELACROIX, King’s College London, UK GILLIAN K. HADFIELD, Johns Hopkins University, USA SIVA REDDY, Mila, McGill University, ServiceNow, Canada

Recent progress in large language models (LLMs) has focused on producing responses that meet human expectations and align with shared values —a process coined alignment. However, aligning LLMs remains challenging due to the inherent disconnect between the complexity of human values and the narrow nature of the technological approaches designed to address them. Current alignment methods often lead to misspecified objectives, reflecting the broader issue of incomplete contracts, the impracticality of specifying a contract between a model developer and the model that accounts for every scenario in LLM alignment. In this paper, we argue that improving LLM alignment requires incorporating insights from societal alignment frameworks, including social, economic, and contractual alignment, and discuss potential solutions drawn from these domains. Given the role of uncertainty within societal alignment frameworks, we then investigate how it manifests in LLM alignment. It is this pervasive uncertainty that necessitates our alternative view on LLM alignment, framing the under-specified nature of its objectives as an opportunity rather than perfect their specification. Beyond technical improvements in LLM alignment, we discuss the need for participatory alignment interface designs.

CCS Concepts: • Social and professional topics → Socio-technical systems; Computing / technology policy; • Computing methodologies → Philosophical/theoretical foundations of artificial intelligence;

Additional Key Words and Phrases: Alignment, LLM, Society

∗Work done by HZ prior to joining Anthropic.

Correspondence to: karolinaewa.stanczak@ai.ethz.ch. Authors’ Contact Information: Karolina Stańczak, ETH Zurich, Zurich, Switzerland and Mila, McGill University, Montreal, Canada; Nicholas Meade, Mila, McGill University, Montreal, Canada; Mehar Bhatia, Mila, McGill University, Montreal, Canada; Hattie Zhou, Mila, Université de Montréal, Montreal, Canada and Anthropic, San Francisco, USA; Konstantin Böttinger, Fraunhofer AISEC, Munich, Germany; Jeremy Barnes, ServiceNow, Montreal, Canada; Jason Stanley, ServiceNow, Montreal, Canada; Jessica Montgomery, University of Cambridge, Cambridge, UK; Richard Zemel, Columbia University, New York, USA; Nicolas Papernot, University of Toronto, Google DeepMind, Toronto, Canada; Nicolas Chapados, Mila, ServiceNow, Montreal, Canada; Denis Therien, McGill University, ServiceNow, Montreal, Canada; Timothy P. Lillicrap, Google DeepMind, London, UK; Ana Marasović, University of Utah, Salt Lake City, USA; Sylvie Delacroix, King’s College London, London, UK; Gillian K. Hadfield, Johns Hopkins University, Baltimore, USA; Siva Reddy, Mila, McGill University, ServiceNow, Montreal, Canada.

- 1 Introduction

As large language models (LLMs) advance to unprecedented levels of proficiency in generating human-like language, aligning their behavior with human values has become a critical challenge to ensuring their usability in real-world applications [40, 70, 89, 104]. This alignment encompasses both explicit values, such as following instructions and being helpful, and implicit values, such as remaining truthful and avoiding biased or otherwise harmful outputs [9]. In fact, the rise of LLM-based chat assistants has largely been driven by their ability to follow instructions and engage in open-ended dialogue, demonstrating the importance of alignment, enabled by algorithms such as reinforcement learning from human feedback (RLHF; [89, 126]).

Despite these advancements, aligning LLMs with human values remains a formidable challenge [48, 118, 119]. This difficulty primarily stems from the fundamental gap between the intricacies of human values and the often narrow technological solutions [50]. Current LLM alignment methods, such as RLHF, often result in misspecified alignment objectives, where reward functions reflect human values only within designer (or annotators) provided scenarios, a finite set among an infinite set of values, failing to generalize in unforeseen contexts [4, 50, 106, 116, 117]. While developers acknowledge the problem of misspecification [70, 89, 104], the root causes of this issue have been largely overlooked.

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

Social Alignment

Economic Alignment

Contractual Alignment

[Figure 8]

reward model (r)

Norms and Values Fair Alignment Legal Documentation

- (§4.1.1)
- (§4.1.2)

- (§4.2.1)
- (§4.2.2)

- (§4.3.1)
- (§4.3.2)

✦Increasing Cultural Competence ✦Multimodality & Non-Verbal Cues

✦Pareto Efﬁciency ✦Social-welfare Functions

✦Regulations ✦Data Sheets, Model Cards

#### Principal Agent

(LLM Developer) (LLM)

Dynamic Norm Evolution Pluralistic Alignment Scalable Oversight

Incomplete Contract

action (a)

✦Continual Learning ✦Model Editing

✦Cooperative Game Theory ✦Specialized Models

✦Debate ✦Constitutional AI

Fig. 1. We view human-LLM interactions as a principal-agent framework, where a principal (a system designer) incentivizes an agent (an LLM) to take an action 𝑎 by offering a reward 𝑟. This framework assumes that the agent’s action is driven by its reward function, forming a pair (𝑎,𝑟) that serves as a contract between the agent and the principal. However, this contract is incomplete. To address this incompleteness, we explore societal alignment mechanisms of social, economic, and contractual alignment as guiding principles for LLM alignment in the incomplete contracting environment.

To better understand this misalignment, we frame LLM alignment within a principal-agent1 framework [37], a well-established paradigm in economic theory. As shown in Figure 1, in this framework, the LLM acts as the agent and the model developer (or user) serves as the principal. We define a contract as a pair: an action taken by the agent and the corresponding reward assigned by the principal. For example, a contract in LLM training could reward the model for generating responses that follow factual accuracy constraints while penalizing hallucinated outputs. The principal is able to steer the agent’s behavior toward intended objectives with an appropriate reward.

1We use ‘agent’ in the contract theory sense, referring to an entity acting on behalf of a principal, rather than the broader AI notion of autonomous systems.

In an ideal scenario, a complete contract would perfectly align the agent’s actions with the principal’s objectives in all possible states of the world.

However, designing a fully specified contract that anticipates every possible scenario in model training is infeasible [50, 125]. In LLM alignment, this challenge is reflected in the reward function, which is derived from explicitly elicited values or implicitly implied values in the form of human preferences. Yet, quantifying complex and often diverging human values is difficult [38, 70], and capturing them effectively incurs high annotation costs [67]. Aggregating these values into a unified reward signal is nontrivial [58, 64].

These alignment challenges are not unique to LLMs. In fact, they echo broader alignment problems that humans encounter daily due to incomplete contracts. Institutions such as society, economy, and law enable us to thrive despite incompleteness. In this position piece, we advocate for leveraging insights from societal alignment frameworks to guide the development of LLM alignment within incomplete contracting environments. Drawing on principles from social alignment (Section 4.1), economic alignment (Section 4.2), and contractual

- alignment (Section 4.3), we propose solutions to guide behavior in incomplete contracting environments, much like they have for human societies (see Figure 1). By design, we strategically refrain from prescribing technical implementation details, as these research directions each demand collective and dedicated effort beyond the scope of this paper.

We then explore how uncertainty, inherent in incomplete contracting environments [101], manifests in LLM

- alignment (Section 5). For instance, an LLM analyzing patient symptoms without a full medical history must navigate this ambiguity to avoid user confusion or misinterpretation. We contend this pervasive uncertainty is not merely a technical challenge but a fundamental characteristic demonstrating that any single, pre-defined stage of alignment will prove insufficient. It is precisely this irreducible uncertainty that necessitates the next stage of alignment, which we detail in our alternative view (Section 6). This alternative view does not oppose existing societal frameworks; rather, it highlights their indispensability while showing why they alone are not enough. Accordingly, we frame an LLM’s under-specified objectives not as a flaw to be resolved solely by technological alignment, but as an opportunity for participatory alignment that actively engages diverse stakeholders in LLM alignment.

- 2 Contemporary Approach to LLM Alignment

Aligning LLMs with human values is commonly understood as training them to act in accordance with user intentions [70]. The objective of LLM alignment is often conceptualized as fulfilling three core qualities, often referred to as the “3H” framework: honesty (regarding their capabilities, internal states, and knowledge), helpfulness (in performing requested tasks or answering questions within safe bounds), and harmlessness (encompassing both the refusal to fulfill harmful requests and the avoidance of generating harmful content) [9, 10].

A prominent approach to achieve this alignment is through a preference-based approach like RLHF. The RLHF pipeline usually includes three stages: supervised fine-tuning (SFT), preference sampling and reward model training [26, 110], and reinforcement learning fine-tuning either using proximal policy optimization (PPO; [97]), or directly through policy optimization (DPO; [92]). The process usually starts with a generic pre-trained language model, which undergoes supervised learning on a high-quality dataset for specific downstream tasks. In this paper, we focus on the implications of the reward modeling stage due to its connection to an incomplete contract, which we will lay out in Section 3.

- 2.1 Reward modeling from human preference.

In the reward modeling stage, for a given input prompt 𝑥, the SFT model generates paired outputs,𝑦0,𝑦1 ∈ Y ×Y, where Y denotes the set of all possible outputs that the model can generate in response to a given input. Human evaluators then select their preferred response, 𝑦 ∈ 𝑦0,𝑦1, providing data that guides the alignment process [26, 110]. Human preferences are modeled probabilistically using frameworks like the Bradley-Terry model [22].

The preference probability for one response over another is expressed as 𝑝(𝑦1 ≻ 𝑦2 | 𝑥) =

exp(𝑟(𝑥,𝑦1)) exp(𝑟(𝑥,𝑦1)) + exp(𝑟(𝑥,𝑦2))

, (1)

where 𝑟(𝑥,𝑦) is a latent reward function approximated by a parametric reward model, 𝑟𝜙(𝑥,𝑦). Using a dataset of comparisons D, the reward model is trained by minimizing the negative log-likelihood

L𝑅(𝑟𝜙, D) = −E(𝑥,𝑦𝑤,𝑦𝑙)∼D log𝜎 𝑟𝜙(𝑥,𝑦𝑤) − 𝑟𝜙(𝑥,𝑦𝑙) , (2) where 𝜎 is the logistic function and 𝑦𝑤 and 𝑦𝑙 denote the preferred and dispreferred completions among (𝑦1,𝑦2).

- 3 LLM Alignment as a Contract

In the following, we formalize LLM alignment through the lens of contract theory [21, 35], a subfield of economics that studies how agreements are designed under conditions of incomplete information. We describe human-LLM interactions as a principal-agent relationship, where a principal (e.g., the user, system designer, or a company) seeks to incentivize an agent (an LLM) to act in a desired manner [44] (see Figure 1). This framework provides a way to conceptualize how the principal tries to align the agent’s behavior with their objectives, using the agent’s action and its reward function as a contract. In this section, we explore the contract formalization (Section 3.1) and how the incompleteness of this contract (Section 3.2) directly leads to misalignment (Section 3.3) in the context of LLM alignment.

- 3.1 Contract Formalization

Following Echenique et al. [35], we define a contract as a pair (𝑎,𝑟), where 𝑎 ∈ A represents an action of an agent and 𝑟 : (X × Y) → R is a reward function.2 The function 𝑟 determines the agent’s reward based on the observed input-output pair (𝑥,𝑦). In the context of a user-LLM interaction, an input 𝑥 ∈ X corresponds to a user prompt, and output 𝑦 ∈ Y is the LLM-generated response. A contract might be, for instance, a positive reward if the model avoids hate speech in the output. Here, the reward function would be trained on prompt-response pairs, awarding higher scores to responses that do not contain hate speech.

The framework is initiated, for instance, when a user, acting as the principal, initiates the interaction by prompting an LLM, thus implicitly proposing a contract. The LLM, acting as an agent, then implicitly either accepts or rejects this contract. Rejection of the contract manifests in the LLM not converging towards the desired output, which is a generated response without hate speech. Upon implicitly accepting the contract, the LLM conducts an action 𝑎, which can be viewed as a probability distribution over all possible model outputs that satisfy the contract. We note that the user does not directly observe the LLM’s internal decision of its action but only the output𝑦. Consequently, the agent is rewarded according to the agreed-upon reward function,𝑟(𝑥,𝑦), implemented as a reward signal during the training phase. The principal experiences the utility derived from the output𝑦; that is, the user benefits from the generated response but also suffers if the model behaves adversarially. This is illustrated when, despite a contract penalizing hate speech, the LLM generates responses that subtly convey harmful biases.

- 3.2 The Challenge of Incomplete Contracting in AI

Although the specific implications of incomplete contracting for LLM alignment remain underexplored, the concept has been studied in the broader context of AI alignment [50]. In theory, alignment between the principal and the agent theoretically requires a complete contract [50, 120]. A complete contract would perfectly align the principal’s objectives with the agent’s behavior in all possible states of the world. This requires that action 𝑎 and reward function 𝑟(𝑥,𝑦) be optimally defined for all input-output pairs. However, achieving complete contracts is practically infeasible for AI systems, rendering incomplete contracting unavoidable [50]. This is primarily due to the

2Here we loosely refer 𝑎 to mean one action or a series of actions that lead to an LLM output.

fact that machine learning systems inherently operate with underspecified objectives [30], which stems from the practical difficulty in defining a reward function𝑟(𝑥,𝑦) that fully captures the complexities of the desired behavior.

The difficulty in specifying such a complete reward function arises from several issues. First, a key challenge for AI alignment generally, real-world applications are too complex to generate all possibilities, hindering the specification of every possible (𝑎,𝑟) pair [88]. The space of possible outcomes, denoted by Y in the formalization is not tractable. This mirrors the challenge of LLM in generating outputs for new input it might receive during inference. The challenge extends beyond the practical limitations of fully specifying objectives. Second, and particularly relevant for LLMs, even beyond these practical limitations, the challenge of translating complex human values into reward functions remains. Ambiguities and gaps in defining the desired action contribute to unintended and often undesirable outcomes.

- 3.3 Misalignment due to an Incomplete Contract

We frame LLM alignment as a challenge of incomplete contracting, which leads to misalignment. In the context of LLMs, this misalignment occurs when the reward function, 𝑟(𝑥,𝑦) is underspecified, and thus might incentivize outputs that diverge from the users’s true objectives.

A common outcome of reward misspecification is reward hacking, where an agent optimizes for the reward itself rather than the intended behavior. For example, LLMs may exploit gaps in the specifications, such as in the “jailbreaking” phenomenon. Here, carefully crafted prompts elicit harmful responses by bypassing weak guardrails because their reward function is not specific enough, allowing the model to optimize without complying with safety requirements [24, 129]. Another example of reward hacking is an LLM trained to generate “helpful” responses might learn to produce lengthy and verbose answers to prompts, as this might result in a higher score from the reward function even if it is not actually helpful to the user [96]. A related issue, “fake alignment,” occurs where the agents superficially comply with the training objective without adopting the intended internal goals [48]. Another challenge is the inherent context dependence of reward functions, which need to adapt appropriately to evolving contexts. A contract might specify desired behavior in a narrow scenario, but leave ambiguities for broader applications [51]. For example, a contract that stipulates “no harmful bias” is inherently underspecified since the definitions of “harmful” and “bias” are context-dependent. Because a complete and exhaustive specification of desired behavior is fundamentally impossible, LLMs, much like human agents, require external frameworks to resolve the inherent ambiguity of incomplete contracts. Just as human society relies on implicit social norms, economic incentives, and legal principles to navigate underspecified contracts, we argue that AI systems must similarly be grounded in these societal frameworks to ensure robust alignment.

- 4 Societal Alignment Frameworks

We present societal alignment frameworks that can provide guidelines for LLM alignment in an incomplete contracting environment. Table 1 provides an overview of these frameworks, their discussed dimensions, and their relevance to current research on LLM alignment. While we touch upon how their insights might inform the development of LLMs, we acknowledge that translating these frameworks into robust and scalable methods defines a new research direction. In the following, we discuss the alignment mechanisms of social theory (Section 4.1), economic theory (Section 4.2), and contractual theory (Section 4.3), and explore potential solutions and associated technical considerations for improving current LLM alignment approaches.

- 4.1 Social Alignment

Human communication relies on a complex, largely implicit set of norms, values, and cues that guide individuals in interpreting each other’s intentions and the world around them [17]. However, this process is inherently ambiguous, as much of the meaning is conveyed implicitly rather than explicitly stated. Nonetheless, humans

Table 1. Overview of Societal Alignment Frameworks. A breakdown of relevance, research directions, and frameworks for social, economic, and contractual alignment.

Framework Dimension Relevance Societal Frameworks Implementations

Social norms corpora [122, 127], Cultural teaming [25]

Normative systems [98], cooperative intelligence [79]

Reducing cultural bias; Explicitly defining latent norms

Normative Competence

Visual/auditory signal integration for intent recognition [73, 85]

Evolutionary Psychology: Non-verbal communication cues [71]

Closing the “normative void” in human-AI interaction

Multimodal Cues

Social Alignment

Model editing [81], Continual learning for bias mitigation

Sociology: Fluidity of social norms and Dynamic Equilibria [84]

Preventing signals from outdated training data

Dynamic Adaptation

Group-level equity; Balancing individual vs. collective utility

Welfare Economics: Social welfare functions [8, 31]

Pareto-optimal preference learning [20], welfare-centric RLHF [29]

Fair Allocation

Economic Alignment

Game Theory: Cooperative games for resource/value allocation [23]

Pluralistic alignment [109], Few-shot learning for diverse perspectives value profiles [108]

Navigating conflicting preferences across diverse stakeholders

Pluralistic Values

Model Cards [82], datasheets [45], ethics guidelines (EU Commission)

Standardized documentation and legal compliance

Contract Law: Incomplete contracting [77, 115]

External Oversight

Contractual Alignment

Political Science: Democratic institutionalized processes [104]

Embedding principles into model reasoning chains

Constitutional AI [11], AI safety debate [60]

Internal Governance

possess a unique ability called normative competence, which allows them to understand and judge whether certain behaviors are appropriate or inappropriate in a given context [98]. This capability is often ingrained in cultures across the world [53, 55, 99], shaping shared understanding that facilitates communication and fosters mutual understanding [79]. A similar challenge arises in user-LLM interactions, where the absence of shared norms and values can result in misaligned outputs. For example, an LLM providing evening activity recommendations without accounting for cultural context might suggest visiting a bar or consuming alcohol in a region where such activities are prohibited or socially unacceptable, leading to responses that fail to align with local norms.

This gap highlights the need of what recent scholarship describes as “thick” conceptual representations [112] and socially grounded contextual analysis for AI value alignment [39, 86]. In particular, Kommers et al. [68] argue that LLMs can be used to make cultural context, and thus human meaning, legible at an unprecedented scale within AI-based socio-technical systems. Internalizing these “thick” societal norms into LLMs could equip them with mechanisms to interpret and dynamically adapt to human normative systems [33], much like these aid alignment within human interactions [17].

- 4.1.1 Instilling norms and values. While norms constitute context-dependent behavioral rules that individuals follow, values represent broader ideals representing overarching goals and aspirations, shaping what individuals

strive for [78]. As a fundamental tool of cooperative intelligence, language plays a crucial role in expressing and reinforcing both norms and values. These can be instilled during LLM alignment in several ways.

LLMs, trained on vast datasets, absorb a multitude of signals about norms and values during training. However, while some attention has been given to broad ethical principles like helpfulness and harmlessness, an important aspect remains underexplored: “contextual rules”—human norms related to cultural conventions. These contextual rules, while not directly influencing primary optimization objectives, are often followed due to tradition, or social norms. Despite their indirect nature, such rules can provide valuable signals about broader societal dynamics, thereby guiding the alignment of LLMs, as discussed by Hadfield-Menell et al. [49] and Köster et al. [69] within the broader context of AI alignment. Although efforts such as Ziems et al. [127], Zhan et al. [122], and Chiu et al. [25] introduce datasets with collections of social norms, the influence of the collected norms on improving alignment in LLMs remains underexplored [1]. Contextual rules could guide the style of language to align with cultural expectations. For instance, when interacting with users from diverse cultural backgrounds, LLM could account for cultural preferences by avoiding humor that might not translate well across cultures. However, existing models have been shown to predominantly reflect Western values, as they have been primarily trained on Western-centric data, which limits their ability to represent multi-cultural values [34, 85].

Human social norms and values are continuously shaped and evaluated through daily interactions with others. These interactions involve the exchange of multimodal signals, such as language, facial expressions, and gestures [71]. However, when interacting with LLMs, these cues are inherently absent, creating a normative gap in communication. This has prompted research dedicated to auditing values embedded within existing models [15, 56]. Further, exploring multimodality for alignment—integrating non-verbal forms of communication such as visual, or auditory signals—can serve as a promising line of research to address this normative void. By incorporating multimodal interactions, models could better align with the implicit social expectations typically conveyed through non-verbal cues [73, 85].

- 4.1.2 Allowing for dynamic norms and values. Norms and values are not static objects but dynamic equilibria that evolve through ongoing social interactions [84]. They are continuously re-articulated and negotiated within social contexts, evolving to address new challenges and cultural shifts [46]. Stereotypes, as a form of social norm, are accordingly fluid, emerging and transforming over time. An example is the shifting perception of remote work. Once seen as unprofessional or less productive, it is now widely accepted in many industries. If an LLM were trained primarily on pre-COVID data, it could reinforce outdated assumptions.

While model editing and continual learning have been extensively explored for updating factual knowledge in LLMs [12, 81], their application for adapting to evolving societal values and norms remains underexplored. Developing approaches to enable LLMs to dynamically identify, adapt to, and mitigate emerging biases is a crucial area for future research. Practically, this involves developing continual learning pipelines that integrate multimodal signals. However, even factual updates pose significant challenges, as highlighted by recent work on knowledge editing [28, 54].

4.2 Economic Alignment

Economic systems rely on specialization and the division of labor, requiring coordination among groups of people to ensure efficient allocation of resources [8]. A central challenge in modern economic theory is aligning individual actors’ interests with collective objectives [50]. Welfare economics provides a complementary perspective by formalizing optimization functions for resource allocation to maximize overall system objectives under given constraints. Similarly, aligning LLMs with diverse human values involves navigating trade-offs between individual and collective goals. Additionally, a coherent social welfare objective function for LLMs cannot rely solely on subjective values. Instead, real-world implementations demand collective decisions about which values to prioritize [8, 102]. Building from this, we explore strategies for integrating economic alignment frameworks

to coordinate individual preferences to achieve collective, fair objectives and facilitating group-level aggregation, offering an alternative view to imposing monolithic objective functions across diverse user groups.

- 4.2.1 Economic Mechanisms for Fair Alignment. In theoretical economics, perfect markets are often posited as achieving a Pareto-efficient distribution of welfare under a utilitarian framework [8]. Pareto efficiency refers to a state where no individual can be made better off without making someone worse off, and is a benchmark for efficient resource allocation [19]. In the context of LLMs, this resource allocation translates to the distribution of model performance, representational fairness, or helpfulness across diverse demographic groups or conflicting values. As shown in Boldi et al. [20], Pareto efficiency offers a valuable lens for balancing competing human preferences and optimizing specific notions of group fairness. In theory, achieving such efficiency aims to tailor the model’s behavior to address diverse needs. However, we caution that strict Pareto efficiency can inadvertently preserve an inequitable status quo, as an optimized model might favor a majority demographic over a marginalized group simply because helping the minority would slightly degrade the majority’s performance.

To overcome this limitation, LLM alignment can draw on alternative frameworks from social welfare economics, where the aggregation of diverse preferences is explicitly balanced to ensure the true collective well-being of multiple groups, rather than preserving a baseline [31]. Specifically, integrating Social Welfare Functions, such as Nash or Egalitarian welfare objectives, directly into the reward modeling phase mathematically can enforce grouplevel equity. By operationalizing pluralistic preference aggregation, these objective functions provide a robust guide for the development of reward systems. Indeed, prior work has indicated that developing welfare-centric objectives can enhance fairness outcomes [29, 90].

- 4.2.2 Economic Mechanisms for Pluralistic Alignment. Decision-making often involves multiple actors with diverse and sometimes conflicting preferences. In the context of LLMs, this necessitates approaches that account for a broad range of values. Pluralistic alignment addresses this challenge by designing models that can represent and respect diverse perspectives [109, 113]. Unlike monolithic approaches, which attempt to impose a singular objective function, pluralistic alignment embraces the complexity of modern societies.

A critical aspect of LLM alignment involves determining how to elicit and aggregate preferences when multiple humans are affected by the behavior of an artificial agent [27, 93, 95]. This challenge extends beyond individual alignment to group alignment, where many societal issues arise from collective behavior rather than isolated actions and can be addressed by incorporating multiple objectives into the alignment process, leveraging methods such as few-shot learning to capture diverse perspectives [123, 124].

Another critical issue in enabling pluralistic values is the trade-off between developing general-purpose models and specialized models. While specialized models tailored to specific domains, such as healthcare or justice, can better align with local norms and regulatory frameworks, they risk fragmenting values. Conversely, generalpurpose models may provide broader applicability but struggle to adapt to ethically complex, domain-specific requirements. Cooperative game theory offers a framework to navigate these tensions by promoting fair resource allocation, fostering collaboration among stakeholders, and ensuring equitable outcomes [23].

- 4.3 Contractual Alignment

Law-making and legal interpretation serve as mechanisms to translate opaque human goals and values into explicit, actionable directives. Legal scholars have long recognized the inherent impossibility of drafting complete contracts [2, 75, 77, 103, 115, 120]. This limitation stems from several key challenges. First, certain states of the world are either unobservable or unverifiable, e.g., hiding assets in complex financial arrangements can be difficult for tax authorities to identify [100]. Second, the limited rationality of humans restricts their ability to anticipate and optimize across the entire, combinatorially large space of potential scenarios [120]. Consequently, precisely computing optimal outcomes becomes intractable. Furthermore, the very description of all possible

contingencies is often beyond human foresight, leading to loopholes in the design of rules [63]. Even if feasible, the costs associated with drafting and enforcing fully specified contracts would likely be prohibitive. Given that these challenges are analogous to those encountered in aligning LLMs, where developers aim to ensure that models produce safe and correct outputs even for inputs not directly represented in training or alignment data, we investigate insights from contract theory as potential solutions for improving LLM alignment.

- 4.3.1 External Contractual Alignment. The formalization of contracts offers a framework for anticipating and specifying desired behaviors in human-LLM interactions [61]. In this context, standardized documentation plays a crucial role in defining and communicating the LLMs’ performance characteristics. Initiatives such as datasheets [45], data statements [13], model cards [82], reproducibility checklists [91], fairness checklists [76], and factsheets [7] exemplify efforts to create clear, standardized guidelines that could inform the development of future regulations and legal frameworks for LLM alignment and data governance.

The rules that guide LLM alignment are currently largely constructed in consultation with domain and legal experts, by adapting documents such as the UN Declaration of Human Rights [5], through public input [6], or in some cases, relying on designer instincts [5, 107]. Importantly, the European Commission has developed detailed guidelines for trustworthy AI, which provide a structured approach to ensuring that AI systems, including LLMs, adhere to ethical principles and societal norms.3 These documents serve as critical tools for defining the terms of human-LLM contracts and offer a principled way to ensure that the view not only reflects the developer’s personal views.

- 4.3.2 Internal Contractual Alignment. While the above discussion focused on aligning LLMs through external rules, another approach takes inspiration from how parties in a contract, laws, and democratic institutions enforce principles. Instead of relying solely on external oversight, this approach embeds normative principles directly within the model’s internal mechanisms. Known as constitutional AI, this method enables LLMs to develop an internalized set of “principles” that guide the model to self-critique and rewrite the response to ensure alignment with predefined norms. By integrating desired rules into the training objectives, constitutional AI aims to instill structural governance within models, much like how legal frameworks encode societal values into enforceable policies. These methods provide scalable oversight precisely because they move beyond the need for direct, case-by-case human intervention. Traditional preference-based training methods, such as collecting annotations on preferred and rejected outputs, aggregate multiple annotators’ judgments into a shared standard, but they still require extensive human effort at scale [4, 104]. In contrast, scalable oversight techniques generalize beyond individual preferences by structuring decision-making mechanisms, similar to how democratic systems use institutionalized processes to apply laws across diverse contexts [104].

One such method, debate [59, 60], mirrors adversarial legal reasoning: agents (i.e., LLMs) propose answers, engage in structured argumentation, and refine their positions, with a human judge selecting the best-supported response [52]. Similarly, constitutional AI guides LLMs using a concise constitution of high-level principles (e.g., promoting fairness or avoiding harm) [11, 111]. This constitution provides the basis for generating synthetic comparison examples, which are then used to fine-tune the LLM’s policy. While primarily developed for integrating human values, these methods have the potential to enforce norms and regulations in a structured manner, drawing parallels to how societal governance mechanisms uphold laws and ethical standards.

In practice, as models are increasingly governed by explicit “constitutions,” rigorous analytical tools are needed to measure how these rules translate into behavioral changes. For instance, a possible research direction uses statistical distance measures to quantify the distribution shifts in model outputs elicited by new alignment constraints.

3The guidelines are available at https://ec.europa.eu/digital-single-market/en/news/ethics-guidelines-trustworthy-ai/.

- 5 Societal Alignment Frameworks and their View on Uncertainty

By framing LLM alignment as a problem of contractual incompleteness and analyzing it through the lens of societal alignment frameworks, we observe that these frameworks recognize establishing contracts, much like alignment, as inherently uncertain [101]. While the unwanted epistemic uncertainty can undermine the reliability of language models, certain types of uncertainty are not only unavoidable but essential for their ethical deployment [32]. In the context of LLMs, this essential uncertainty can arise from evolving human values, conflicting societal norms, and the difficulty of translating abstract principles into model behavior. Aligning models to navigate trade-offs, such as between helpfulness and harmlessness or accuracy and fairness, requires addressing conflicting and often underspecified priorities, which introduces another source of uncertainty [121, 128]. For instance, when deploying an LLM, we often want to maximize performance subject to some constraints or guardrails on behavior, e.g., a chatbot should give users their desired output, as long as it is not too toxic. The effectiveness of balancing these conflicting priorities and the unintended consequences are often difficult to predict. However, this balancing act is also essential because it allows models to operate within complex, context-dependent environments where rigid adherence to a single objective could lead to harmful outcomes.

Building on the above, the inherent uncertainty in LLM alignment is not a weakness but often a valuable feature that enables models to handle complex scenarios ethically [32]. To illustrate this, consider an AI system deployed for education, tasked with suggesting new exercises to a student. If the system lacks normative uncertainty, it risks overfitting to a narrow objective, such as strictly minimizing immediate failure rates. It might continuously propose overly simplistic tasks to a student who initially performed poorly, instead of adapting to the student’s cognitive development. By maintaining a degree of normative uncertainty, LLMs can avoid such optimization risks. This flexibility parallels how legal systems manage the ambiguity of incomplete laws through adaptive mechanisms such as precedent, amendments, and ongoing judicial interpretation. Furthermore, as highlighted by Bhatt et al. [16], uncertainty communication can be useful for obtaining fairer models by revealing data biases, improving decision-making by guiding reliance on predictions, and building trust in automated systems.

However, effectively communicating this uncertainty remains a challenge. Unlike humans, LLMs lack the non-verbal and contextual cues that naturally support nuanced communication [18]. Existing research has shown that LLMs struggle to convey their uncertainty to users, both implicitly (e.g., through hedging language) and explicitly (e.g., via confidence scores), a skill that humans possess intuitively [3, 74, 105]. On the other hand, humans themselves have varying levels of understanding regarding probability and statistics, which are needed to interpret model uncertainty estimates [16, 43]. Human cognition is also subject to inherent biases that can impede accurate interpretation of uncertainty [62, 94]. These challenges can be partially addressed by choosing the appropriate communication methods, a key consideration for the design of effective user interfaces [57], and by designing collaborative interaction environments, as discussed by Montemayor [83].

- 6 The Democratic Opportunity Inherent in the Under-specified Nature of LLMs’ Objectives

The challenge of aligning LLMs is often framed as a technical problem, one that can be solved through better reward modeling, training objectives, or oversight mechanisms. However, as our exploration suggests, alignment is not merely a technological issue. It is fundamentally a societal one. To understand the significance of this alternative view, one needs to take a step back and start from the following: we humans are constantly in the process of finding our way around the world. Part of that process involves imagining better ways of living together. We may find some of our practices to be inadequate, for instance, but may not always be able to articulate why. In such cases, we often resort to conversations to refine our intuitions and distill their underlying structures. These evolving dialogues shape and refine our moral and social expectations, which, in turn, influence the values that guide our decision-making. The fact that these values change and often clash is a good sign—a sign of ongoing critical engagement and willingness to question existing norms.

Now, consider a team of engineers designing AI tools that will be deployed within contexts such as education, healthcare, or justice practices. Some of these tools, like LLMs, can be used as conversational partners. The feedback given as a context can be leveraged to refine LLMs’ behavior. Given the inherently dynamic nature of the values that inform education, healthcare, or justice practices, as we previously discussed, the key problem is to establish how to structure this feedback process. Different groups of users will evolve different values over time. Are there ways of incentivizing collective, critical engagement with LLMs? Can bottom-up, iterative refinements be configured to support users in defining the very values that preside over their practices [32]?

The “incomplete contract” metaphor has been a useful diagnostic tool throughout this paper, highlighting why purely technical attempts to fully specify LLM objectives are bound to fall short, much like real-world contracts cannot anticipate every contingency. However, we must actively recognize the limitations of this metaphor if taken too literally as a prescriptive framework for solutions. As Goldoni and Wilkinson [47] caution, viewing alignment solely as a contract to be completed can oversimplify complex systems, potentially framing it as a straightforward, albeit difficult, agreement between clearly defined stakeholders (like a principal and agent). This risks neglecting the broader socio-political forces and conflicting norms that shape alignment challenges, and can inadvertently reinforce a designer-centric, epistemic focus on fixing the incompleteness [114]. While societal alignment frameworks, as discussed, aim to address these broader issues, they too can rely on oversimplified assumptions if not carefully applied. It is precisely this irreducible uncertainty and the inherent incompleteness of any pre-defined alignment stage that necessitates this alternative view. The true opportunity lies not in striving to technically perfect an inherently incomplete contract, but in monitoring how these contracts are formed and by whom. The under-specified nature of LLMs’ objectives presents a democratic opportunity to democratize the very process of determining what LLMs should optimize for. This reframes the challenge from one of technical specification by a few to one of ongoing, participatory value deliberation by many.

This inherent limitation necessitates an alternative view. As highlighted by recent work on the democratization of AI [72], aligning AI systems requires moving away from a solely technical framing and toward participatory engagement. Furthermore, while recent work by Millière [80] critiques current preference-tuning methods for resulting in “shallow alignment,” where models mimic normative patterns without their understanding, our contract theory framework explains why this shallowness persists. The true opportunity lies not in striving to technically perfect an inherently incomplete contract, but in democratizing the very process of determining what LLMs should optimize for. This reframes the challenge from one of technical specification by a few to one of ongoing, participatory value deliberation by many.

The implications of this reframing extend to both research and practice. Addressing the concentrated power dynamics inherent in the principal-agent relationship requires participatory alignment. This suggests that, alongside technical work such as reward modeling, we need equally sophisticated work on participatory interface designs. This can be operationalized by integrating iterative feedback loops where affected communities actively shape model behavior, as demonstrated by initiatives like the PRISM dataset [66]. Rather than relying solely on developer-defined annotations, preference fine-tuning can be conducted on diverse, real-user data inferred from community behaviors and deliberations. This might also include developing new methodologies for collective value articulation [14], creating institutional structures for meaningful public participation in LLM development, and establishing mechanisms for ongoing societal oversight and input into LLMs’ objectives and constraints.

- 7 Related Work

The field of sociotechnical alignment posits that AI systems are embedded within larger institutional structures, including corporations, markets, and nation-states. Consequently, beneficial societal outcomes cannot be guaranteed simply by aligning an individual AI system with the immediate intentions of its operator or user [36, 40–42]. Sociotechnical alignment aims to address this gap by broadening the scope of AI alignment to incorporate social

and systemic factors. In this paper, we specifically focus on the interdisciplinary connections between LLM alignment and established societal frameworks, centering our analysis on the shared challenge of contractual incompleteness.

Since the early works on sociotechnical alignment, a recurring tension persists between “local” and “global” alignment. Early foundational works have noted that while AI systems may be locally aligned with an operator’s specific instructions, they frequently remain misaligned with broader societal interests and objectives [65, 87]. This misalignment often stems from the collective action problems and the inherent difficulty of specifying societal-level welfare in a way that technical systems can optimize. Our work builds upon these insights by framing this underspecification not merely as a technical challenge, but as a fundamental characteristic of incomplete contracting that requires the integration of social, economic, and contractual theories.

A recent development in this area is the concept of full-stack alignment, as proposed by Edelman et al. [36], which advocates for the creation of normatively competent agents capable of sophisticated normative reasoning. This mirrors our discussion on the necessity of normative competence for navigating implicit social cues. Furthermore, the challenge of addressing the diverse and often conflicting values of a global user base has led to the emergence of pluralistic alignment [109], which seeks to represent a multitude of perspectives rather than imposing a monolithic standard. From an economic and political perspective, researchers have argued that social choice theory should serve as a primary guide for LLM alignment [27], providing formal mechanisms for aggregating individual preferences into collective decisions.

- 8 Conclusions

Aligning LLMs with human values remains a critical challenge as these models are increasingly deployed in real-world applications. Current approaches, such as RLHF, struggle with the inherent misspecification of reward functions, which fail to capture the complexity and evolving nature of human values. Additionally, language ambiguity further complicates alignment efforts.

In this paper, we have argued that addressing these challenges requires reframing LLM alignment through the lens of contract theory. We model LLM alignment within a principal-agent framework, where the principal (a user or developer) defines a contract. This contract consists of an action taken by the agent (the LLM) and the corresponding reward assigned by the principal. We then draw connections between the challenges of contract formation in societal alignment frameworks and those in LLM alignment, arguing that insights from societal alignment can improve LLM alignment within incomplete contracting environments. While contract theory provides us with some formalization tools, social alignment emphasizes the role of instillation of societal norms and values, economic alignment points to solutions to group alignment and allocation challenges, and contractual alignment provides mechanisms for regulating LLM behavior, both externally through legal frameworks and internally through scalable oversight mechanisms. Finally, we present an alternative view on LLM alignment, advocating for shifting the paradigm from developer-centered to collaborative, user-centric, and iterative approaches to LLM alignment.

Generative AI Usage Statement

In accordance with the ACM Policy on Authorship regarding the use of generative AI, the authors disclose that generative AI tools (specifically Gemini 3 Pro) were used in the preparation of this manuscript. The use of these tools was strictly limited to: (1) assisting with grammar, syntax, and the polishing of author-written text for fluency; and (2) assisting with the technical formatting of LaTeX. These tools were not used to generate new research content, conceptual ideas, or citations. All references were manually collected. The authors edited all outputs and remain fully accountable for this work.

Acknowledgments

This paper originated from the Bellairs Invitational Workshop on Contemporary, Foreseeable, and Catastrophic Risks of Large Language Models in April 2024. We thank all workshop participants for their valuable discussions and contributions. Karolina Stańczak was supported by the Mila P2v5 grant, the Mila-Samsung grant, and by the ETH AI Center postdoctoral fellowship.

References

- [1] Aakanksha, Arash Ahmadian, Beyza Ermis, Seraphina Goldfarb-Tarrant, Julia Kreutzer, Marzieh Fadaee, and Sara Hooker. 2024. The Multilingual Alignment Prism: Aligning Global and Local Preferences to Reduce Harm. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen (Eds.). Association for Computational Linguistics, Miami, Florida, USA, 12027–12049. doi:10.18653/v1/2024.emnlp-main.671
- [2] Philippe Aghion and Richard Holden. 2011. Incomplete Contracts and the Theory of the Firm: What Have We Learned over the Past 25 Years? Journal of Economic Perspectives 25, 2 (June 2011), 181–97. doi:10.1257/jep.25.2.181
- [3] Hussam Alkaissi and Samy I McFarlane. 2023. Artificial Hallucinations in ChatGPT: Implications in Scientific Writing. Cureus 15

(2023), e35179. https://pmc.ncbi.nlm.nih.gov/articles/PMC9939079/

- [4] Dario Amodei, Chris Olah, Jacob Steinhardt, Paul Christiano, John Schulman, and Dan Mané. 2016. Concrete Problems in AI Safety. arXiv:1606.06565 [cs.AI] https://arxiv.org/abs/1606.06565
- [5] Anthropic. 2023. Claude’s Constitution. https://www.anthropic.com/index/claudes-constitution. Accessed: 2025-01-27.
- [6] Anthropic. 2023. Collective Constitutional AI: Aligning a Language Model with Public Input. https://www.anthropic.com/index/collec tive-constitutional-ai-aligning-a-language-model-with-public-input. Accessed: 2025-01-27.
- [7] M. Arnold, R. K. E. Bellamy, M. Hind, S. Houde, S. Mehta, A. Mojsilović, R. Nair, K. Natesan Ramamurthy, A. Olteanu, D. Piorkowski, D. Reimer, J. Richards, J. Tsay, and K. R. Varshney. 2019. FactSheets: Increasing trust in AI services through supplier’s declarations of conformity. IBM Journal of Research and Development 63, 4/5 (2019), 6:1–6:13. doi:10.1147/JRD.2019.2942288
- [8] Kenneth J Arrow. 1951. An extension of the basic theorems of classical welfare economics. In Proceedings of the second Berkeley symposium on mathematical statistics and probability, Vol. 2. University of California Press, University of California Press, Berkeley, CA, 507–533. https://projecteuclid.org/ebooks/berkeley-symposium-on-mathematical-statistics-and-probability/Proceedings-ofthe-Second-Berkeley-Symposium-on-Mathematical-Statistics-and/chapter/An-Extension-of-the-Basic-Theorems-of-ClassicalWelfare-Economics/bsmsp/1200500251
- [9] Amanda Askell, Yuntao Bai, Anna Chen, Dawn Drain, Deep Ganguli, Tom Henighan, Andy Jones, Nicholas Joseph, Ben Mann, Nova DasSarma, Nelson Elhage, Zac Hatfield-Dodds, Danny Hernandez, Jackson Kernion, Kamal Ndousse, Catherine Olsson, Dario Amodei, Tom Brown, Jack Clark, Sam McCandlish, Chris Olah, and Jared Kaplan. 2021. A General Language Assistant as a Laboratory for Alignment. arXiv:2112.00861 [cs.CL] https://arxiv.org/abs/2112.00861
- [10] Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova Dassarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, Nicholas Joseph, Saurav Kadavath, John Kernion, Tom Conerly, Sheer El-Showk, Nelson Elhage, Zac Hatfield-Dodds, Danny Hernandez, Tristan Hume, Scott Johnston, Shauna Kravec, Liane Lovitt, Neel Nanda, Catherine Olsson, Dario Amodei, Tom B. Brown, Jack Clark, Sam McCandlish, Christopher Olah, Benjamin Mann, and Jared Kaplan. 2022. Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback. ArXiv abs/2204.05862 (2022). https://arxiv.org/abs/2204.05862
- [11] Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, Carol Chen, Catherine Olsson, Christopher Olah, Danny Hernandez, Dawn Drain, Deep Ganguli, Dustin Li, Eli Tran-Johnson, Ethan Perez, Jamie Kerr, Jared Mueller, Jeffrey Ladish, Joshua Landau, Kamal Ndousse, Kamile Lukosuite, Liane Lovitt, Michael Sellitto, Nelson Elhage, Nicholas Schiefer, Noemi Mercado, Nova DasSarma, Robert Lasenby, Robin Larson, Sam Ringer, Scott Johnston, Shauna Kravec, Sheer El Showk, Stanislav Fort, Tamera Lanham, Timothy Telleen-Lawton, Tom Conerly, Tom Henighan, Tristan Hume, Samuel R. Bowman, Zac Hatfield-Dodds, Ben Mann, Dario Amodei, Nicholas Joseph, Sam McCandlish, Tom Brown, and Jared Kaplan. 2022. Constitutional AI: Harmlessness from AI Feedback. arXiv:2212.08073 [cs.CL] https://arxiv.org/abs/22 12.08073
- [12] Diana Benavides-Prado and Patricia Riddle. 2022. A Theory for Knowledge Transfer in Continual Learning. In Proceedings of The 1st Conference on Lifelong Learning Agents (Proceedings of Machine Learning Research, Vol. 199), Sarath Chandar, Razvan Pascanu, and Doina Precup (Eds.). PMLR, Montréal, Canada, 647–660. https://proceedings.mlr.press/v199/prado22a.html
- [13] Emily M. Bender and Batya Friedman. 2018. Data Statements for Natural Language Processing: Toward Mitigating System Bias and Enabling Better Science. Transactions of the Association for Computational Linguistics 6 (12 2018), 587–604. arXiv:https://direct.mit.edu/tacl/article-pdf/doi/10.1162/tacl_a_00041/1567666/tacl_a_00041.pdf doi:10.1162/tacl_a_00041
- [14] Stevie Bergman, Nahema Marchal, John Mellor, Shakir Mohamed, Iason Gabriel, and William Isaac. 2024. STELA: A community-centred approach to norm elicitation for AI alignment. Scientific Reports 14 (2024), 6616. doi:10.1038/s41598-024-56648-4

- [15] Mehar Bhatia, Shravan Nayak, Gaurav Kamath, Marius Mosbach, Karolina Stańczak, Vered Shwartz, and Siva Reddy. 2025. Value Drifts: Tracing Value Alignment During LLM Post-Training. arXiv:2510.26707 [cs.CL] https://arxiv.org/abs/2510.26707
- [16] Umang Bhatt, Javier Antorán, Yunfeng Zhang, Q. Vera Liao, Prasanna Sattigeri, Riccardo Fogliato, Gabrielle Melançon, Ranganath Krishnan, Jason Stanley, Omesh Tickoo, Lama Nachman, Rumi Chunara, Madhulika Srikumar, Adrian Weller, and Alice Xiang. 2021. Uncertainty as a Form of Transparency: Measuring, Communicating, and Using Uncertainty. In Proceedings of the 2021 AAAI/ACM Conference on AI, Ethics, and Society (Virtual Event, USA) (AIES ’21). Association for Computing Machinery, New York, NY, USA, 401–413. doi:10.1145/3461702.3462571
- [17] Cristina Bicchieri. 2017. Norms in the Wild: How to Diagnose, Measure, and Change Social Norms. Oxford University Press. doi:10.1093/ acprof:oso/9780190622046.001.0001
- [18] Paola Bisconti. 2021. How Robots’ Unintentional Metacommunication Affects Human–Robot Interactions: A Systemic Approach. Minds & Machines 31 (2021), 487–504. doi:10.1007/s11023-021-09584-5
- [19] John D. Black, Nigar Hashimzade, and Gareth Myles (Eds.). 2017. A Dictionary of Economics (5th ed.). Oxford University Press, Oxford. 459 pages. https://books.google.ca/books?id=WyvYDQAAQBAJ&pg=PT459&redir_esc=y#v=onepage&q&f=false
- [20] Ryan Boldi, Li Ding, Lee Spector, and Scott Niekum. 2024. Pareto-Optimal Learning from Preferences with Hidden Context. arXiv:2406.15599 [cs.LG] https://arxiv.org/abs/2406.15599
- [21] Patrick Bolton and Mathias Dewatripont. 2005. Contract Theory. MIT Press Books, Vol. 1. The MIT Press. https://ideas.repec.org/b/mt p/titles/0262025760.html
- [22] Ralph Allan Bradley and Milton E. Terry. 1952. Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons. Biometrika 39, 3/4 (1952), 324–345. http://www.jstor.org/stable/2334029
- [23] Georgios Chalkiadakis, Edith Elkind, and Michael Wooldridge. 2011. Computational Aspects of Cooperative Game Theory (Synthesis Lectures on Artificial Inetlligence and Machine Learning) (1st ed.). Morgan & Claypool Publishers. https://link.springer.com/book/10.10 07/978-3-031-01558-8
- [24] Patrick Chao, Alexander Robey, Edgar Dobriban, Hamed Hassani, George J. Pappas, and Eric Wong. 2024. Jailbreaking Black Box Large Language Models in Twenty Queries. arXiv:2310.08419 [cs.LG] https://arxiv.org/abs/2310.08419
- [25] Yu Ying Chiu, Liwei Jiang, Maria Antoniak, Chan Young Park, Shuyue Stella Li, Mehar Bhatia, Sahithya Ravi, Yulia Tsvetkov, Vered Shwartz, and Yejin Choi. 2024. CulturalTeaming: AI-Assisted Interactive Red-Teaming for Challenging LLMs’(Lack of) Multicultural Knowledge. arXiv preprint arXiv:2404.06664 (2024). https://arxiv.org/abs/2404.06664
- [26] Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. 2017. Deep Reinforcement Learning from Human Preferences. In Advances in Neural Information Processing Systems, I. Guyon, U. Von Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett (Eds.), Vol. 30. Curran Associates, Inc. https://proceedings.neurips.cc/paper_files/paper/2017/file/d5 e2c0adad503c91f91df240d0cd4e49-Paper.pdf
- [27] Vincent Conitzer, Rachel Freedman, Jobst Heitzig, Wesley H. Holliday, Bob M. Jacobs, Nathan Lambert, Milan Mosse, Eric Pacuit, Stuart Russell, Hailey Schoelkopf, Emanuel Tewolde, and William S. Zwicker. 2024. Position: Social Choice Should Guide AI Alignment in Dealing with Diverse Human Feedback. In Proceedings of the 41st International Conference on Machine Learning (Proceedings of Machine Learning Research, Vol. 235), Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp (Eds.). PMLR, 9346–9360. https://proceedings.mlr.press/v235/conitzer24a.html
- [28] A. Feder Cooper, Christopher A. Choquette-Choo, Miranda Bogen, Matthew Jagielski, Katja Filippova, Ken Liu, Alexandra Chouldechova, Jamie Hayes, Yangsibo Huang, Niloofar Mireshghallah, Ilia Shumailov, Eleni Triantafillou, Peter Kairouz, Nicole Mitchell, Percy Liang, Daniel E. Ho, Yejin Choi, Sanmi Koyejo, Fernando Delgado, James Grimmelmann, Vitaly Shmatikov, Christopher De Sa, Solon Barocas, Amy Cyphert, Mark A. Lemley, Danah Boyd, Jennifer Wortman Vaughan, Miles Brundage, David Bau, Seth Neel, Abigail Jacobs, Andreas Terzis, Hanna Wallach, Nicolas Papernot, and Katherine Lee. 2024. Machine Unlearning Doesn’t Do What You Think: Lessons for Generative AI Policy, Research, and Practice. SSRN (December 2024). doi:10.2139/ssrn.5060253
- [29] Cyrus Cousins, Kavosh Asadi, Elita Lobo, and Michael Littman. 2024. On Welfare-Centric Fair Reinforcement Learning. Reinforcement Learning Journal 3 (2024), 1124–1137. https://rlj.cs.umass.edu/2024/papers/Paper133.html
- [30] Alexander D’Amour, Katherine Heller, Dan Moldovan, Ben Adlam, Babak Alipanahi, Alex Beutel, Christina Chen, Jonathan Deaton, Jacob Eisenstein, Matthew D. Hoffman, Farhad Hormozdiari, Neil Houlsby, Shaobo Hou, Ghassen Jerfel, Alan Karthikesalingam, Mario Lucic, Yian Ma, Cory McLean, Diana Mincu, Akinori Mitani, Andrea Montanari, Zachary Nado, Vivek Natarajan, Christopher Nielson, Thomas F. Osborne, Rajiv Raman, Kim Ramasamy, Rory Sayres, Jessica Schrouff, Martin Seneviratne, Shannon Sequeira, Harini Suresh, Victor Veitch, Max Vladymyrov, Xuezhi Wang, Kellie Webster, Steve Yadlowsky, Taedong Yun, Xiaohua Zhai, and D. Sculley. 2022. Underspecification presents challenges for credibility in modern machine learning. Journal of Machine Learning Research 23, 1, Article 226 (Jan. 2022), 61 pages. https://dl.acm.org/doi/pdf/10.5555/3586589.3586815
- [31] Claude d’Aspremont and Louis Gevers. 2002. Social welfare functionals and interpersonal comparability. In Handbook of Social Choice and Welfare. Handbook of Social Choice and Welfare, Vol. 1. Elsevier, 459–541. doi:10.1016/S1574-0110(02)80014-5
- [32] Sylvie Delacroix. 2024. Lost in Conversation? Hermeneutics, uncertainty and large language models. SSRN (April 21 2024). https: //ssrn.com/abstract=4751774

- [33] Anca D. Dragan, Kenton C.T. Lee, and Siddhartha S. Srinivasa. 2013. Legibility and predictability of robot motion. In Proceedings of the 8th ACM/IEEE International Conference on Human-Robot Interaction (Tokyo, Japan) (HRI ’13). IEEE Press, 301–308. https: //ieeexplore.ieee.org/document/6483603
- [34] Esin Durmus, Karina Nguyen, Thomas Liao, Nicholas Schiefer, Amanda Askell, Anton Bakhtin, Carol Chen, Zac Hatfield-Dodds, Danny Hernandez, Nicholas Joseph, Liane Lovitt, Sam McCandlish, Orowa Sikder, Alex Tamkin, Janel Thamkul, Jared Kaplan, Jack Clark, and Deep Ganguli. 2024. Towards Measuring the Representation of Subjective Global Opinions in Language Models. In First Conference on Language Modeling. https://openreview.net/forum?id=zl16jLb91v
- [35] F. Echenique, N. Immorlica, V.V. Vazirani, and A.E. Roth. 2023. Contract Theory. Cambridge University Press, 614–624. https: //books.google.ca/books?id=1ea-EAAAQBAJ
- [36] Joe Edelman, Tan Zhi-Xuan, Ryan Lowe, Oliver Klingefjord, Vincent Wang-Mascianica, Matija Franklin, Ryan Othniel Kearns, Ellie Hain, Atrisha Sarkar, Michiel Bakker, Fazl Barez, David Duvenaud, Jakob Foerster, Iason Gabriel, Joseph Gubbels, Bryce Goodman, Andreas Haupt, Jobst Heitzig, Julian Jara-Ettinger, Atoosa Kasirzadeh, James Ravi Kirkpatrick, Andrew Koh, W. Bradley Knox, Philipp Koralus, Joel Lehman, Sydney Levine, Samuele Marro, Manon Revel, Toby Shorin, Morgan Sutherland, Michael Henry Tessler, Ivan Vendrov, and James Wilken-Smith. 2025. Full-Stack Alignment: Co-Aligning AI and Institutions with Thick Models of Value. (2025). arXiv:2512.03399 [cs.LG] https://arxiv.org/abs/2512.03399
- [37] Kathleen M. Eisenhardt. 1989. Agency Theory: An Assessment and Review. The Academy of Management Review 14, 1 (1989), 57–74. http://www.jstor.org/stable/258191
- [38] Michael Feffer, Hoda Heidari, and Zachary C. Lipton. 2023. Moral Machine or Tyranny of the Majority? arXiv:2305.17319 [cs.CY] https://arxiv.org/abs/2305.17319
- [39] Jacob G. Foster. 2023. From Thin to Thick: Toward a Politics of Human-Compatible AI. Public Culture 35, 3 (101) (09 2023), 417–430. arXiv:https://read.dukeupress.edu/public-culture/article-pdf/35/3 (101)/417/2081019/417foster.pdf doi:10.1215/08992363-10742593
- [40] Iason Gabriel. 2020. Artificial Intelligence, Values, and Alignment. Minds & Machines 30, 3 (2020), 411–437. doi:10.1007/s11023-02009539-2
- [41] Iason Gabriel and Geoff Keeling. 2025. A Matter of Principle? AI Alignment as the Fair Treatment of Claims. Philosophical Studies 182

(2025), 1951–1973. doi:10.1007/s11098-025-02300-4

- [42] Iason Gabriel, Arianna Manzini, Geoff Keeling, Lisa Anne Hendricks, Verena Rieser, Hasan Iqbal, Nenad Tomašev, Ira Ktena, Zachary Kenton, Mikel Rodriguez, Seliem El-Sayed, Sasha Brown, Canfer Akbulut, Andrew Trask, Edward Hughes, A. Stevie Bergman, Renee Shelby, Nahema Marchal, Conor Griffin, Juan Mateos-Garcia, Laura Weidinger, Winnie Street, Benjamin Lange, Alex Ingerman, Alison Lentz, Reed Enger, Andrew Barakat, Victoria Krakovna, John Oliver Siy, Zeb Kurth-Nelson, Amanda McCroskery, Vijay Bolina, Harry Law, Murray Shanahan, Lize Alberts, Borja Balle, Sarah de Haas, Yetunde Ibitoye, Allan Dafoe, Beth Goldberg, Sébastien Krier, Alexander Reese, Sims Witherspoon, Will Hawkins, Maribeth Rauh, Don Wallace, Matija Franklin, Josh A. Goldstein, Joel Lehman, Michael Klenk, Shannon Vallor, Courtney Biles, Meredith Ringel Morris, Helen King, Blaise Agüera y Arcas, William Isaac, and James Manyika. 2024. The Ethics of Advanced AI Assistants. (2024). arXiv:2404.16244 [cs.CY] https://arxiv.org/abs/2404.16244
- [43] Mirta Galesic and Rocio Garcia-Retamero. 2010. Statistical Numeracy for Health: A Cross-cultural Comparison With Probabilistic National Samples. Archives of Internal Medicine 170, 5 (03 2010), 462–468. doi:10.1001/archinternmed.2009.481
- [44] John E. Garen. 1994. Executive Compensation and Principal-Agent Theory. Journal of Political Economy 102, 6 (1994), 1175–1199. http://www.jstor.org/stable/2138783
- [45] Timnit Gebru, Jamie Morgenstern, Briana Vecchione, Jennifer Wortman Vaughan, Hanna Wallach, Hal Daumé III, and Kate Crawford.

2021. Datasheets for datasets. Commun. ACM 64, 12 (Nov. 2021), 86–92. doi:10.1145/3458723

- [46] Michele J. Gelfand, Sergey Gavrilets, and Nathan Nunn. 2024. Norm Dynamics: Interdisciplinary Perspectives on Social Norm Emergence, Persistence, and Change. Annual Review of Psychology 75, Volume 75, 2024 (2024), 341–378. doi:10.1146/annurev-psych-033020-013319
- [47] Marco Goldoni and Michael A. Wilkinson. 2018. The Material Constitution. The Modern Law Review 81, 4 (2018), 567–597. http: //www.jstor.org/stable/26647134
- [48] Ryan Greenblatt, Carson Denison, Benjamin Wright, Fabien Roger, Monte MacDiarmid, Sam Marks, Johannes Treutlein, Tim Belonax, Jack Chen, David Duvenaud, Akbir Khan, Julian Michael, Sören Mindermann, Ethan Perez, Linda Petrini, Jonathan Uesato, Jared Kaplan, Buck Shlegeris, Samuel R. Bowman, and Evan Hubinger. 2024. Alignment faking in large language models. arXiv:2412.14093 [cs.AI] https://arxiv.org/abs/2412.14093
- [49] Dylan Hadfield-Menell, Mckane Andrus, and Gillian Hadfield. 2019. Legible Normativity for AI Alignment: The Value of Silly Rules. In Proceedings of the 2019 AAAI/ACM Conference on AI, Ethics, and Society (Honolulu, HI, USA) (AIES ’19). Association for Computing Machinery, New York, NY, USA, 115–121. doi:10.1145/3306618.3314258
- [50] Dylan Hadfield-Menell and Gillian K. Hadfield. 2019. Incomplete Contracting and AI Alignment. In Proceedings of the 2019 AAAI/ACM Conference on AI, Ethics, and Society (Honolulu, HI, USA) (AIES ’19). Association for Computing Machinery, New York, NY, USA, 417–422. doi:10.1145/3306618.3314250
- [51] Dylan Hadfield-Menell, Smitha Milli, Pieter Abbeel, Stuart J Russell, and Anca Dragan. 2017. Inverse Reward Design. In Advances in Neural Information Processing Systems, I. Guyon, U. Von Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett

- (Eds.), Vol. 30. Curran Associates, Inc. https://proceedings.neurips.cc/paper_files/paper/2017/file/32fdab6559cdfa4f167f8c31b9199643Paper.pdf
- [52] C.D. Hafner. 2001. Legal Reasoning Models. In International Encyclopedia of the Social & Behavioral Sciences, Neil J. Smelser and Paul B. Baltes (Eds.). Pergamon, Oxford, 8675–8677. doi:10.1016/B0-08-043076-7/00586-6
- [53] Paul H. P. Hanel, Gregory R. Maio, Ana K. S. Soares, Katia C. Vione, Gabriel L. de Holanda Coelho, Valdiney V. Gouveia, Appasaheb C. Patil, Shanmukh V. Kamble, and Antony S. R. Manstead. 2018. Cross-Cultural Differences and Similarities in Human Value Instantiation. Frontiers in Psychology 9 (2018). doi:10.3389/fpsyg.2018.00849
- [54] Peter Hase, Thomas Hofweber, Xiang Zhou, Elias Stengel-Eskin, and Mohit Bansal. 2024. Fundamental Problems With Model Editing: How Should Rational Belief Revision Work in LLMs? Transactions on Machine Learning Research (2024). https://openreview.net/for um?id=LRf19n5Ly3
- [55] Daniel Hershcovich, Stella Frank, Heather Lent, Miryam de Lhoneux, Mostafa Abdou, Stephanie Brandl, Emanuele Bugliarello, Laura Cabello Piqueras, Ilias Chalkidis, Ruixiang Cui, Constanza Fierro, Katerina Margatina, Phillip Rust, and Anders Søgaard. 2022. Challenges and Strategies in Cross-Cultural NLP. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), Smaranda Muresan, Preslav Nakov, and Aline Villavicencio (Eds.). Association for Computational Linguistics, Dublin, Ireland, 6997–7013. doi:10.18653/v1/2022.acl-long.482
- [56] Saffron Huang, Esin Durmus, Miles McCain, Kunal Handa, Alex Tamkin, Jerry Hong, Michael Stern, Arushi Somani, Xiuruo Zhang, and Deep Ganguli. 2025. Values in the Wild: Discovering and Analyzing Values in Real-World Language Model Interactions. (2025). arXiv:2504.15236 [cs.CL] https://arxiv.org/abs/2504.15236
- [57] Jessica Hullman, Xiaoli Qiao, Michael Correll, Alex Kale, and Matthew Kay. 2019. In Pursuit of Error: A Survey of Uncertainty Visualization Evaluation. IEEE Transactions on Visualization and Computer Graphics 25, 1 (2019), 903–913. doi:10.1109/TVCG.2018.2864 889
- [58] Christina Ilvento. 2020. Metric Learning for Individual Fairness. In 1st Symposium on Foundations of Responsible Computing (FORC

2020) (Leibniz International Proceedings in Informatics (LIPIcs), Vol. 156), Aaron Roth (Ed.). Schloss Dagstuhl – Leibniz-Zentrum für Informatik, Dagstuhl, Germany, 2:1–2:11. doi:10.4230/LIPIcs.FORC.2020.2

- [59] Geoffrey Irving and Amanda Askell. 2019. AI Safety Needs Social Scientists. Distill (2019). doi:10.23915/distill.00021
- [60] Geoffrey Irving, Paul Christiano, and Dario Amodei. 2018. AI safety via debate. arXiv:1805.00899 [stat.ML] https://arxiv.org/abs/1805

.00899

- [61] Alon Jacovi, Ana Marasović, Tim Miller, and Yoav Goldberg. 2021. Formalizing Trust in Artificial Intelligence: Prerequisites, Causes and Goals of Human Trust in AI. In Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency (Virtual Event, Canada) (FAccT ’21). Association for Computing Machinery, New York, NY, USA, 624–635. doi:10.1145/3442188.3445923
- [62] Daniel Kahneman. 2011. Thinking, Fast and Slow. Allen Lane. https://books.google.ca/books?id=AV9x8XakdV0C
- [63] Leo Katz. 2010. A Theory of Loopholes. The Journal of Legal Studies 39, 1 (2010), 1–31. http://www.jstor.org/stable/10.1086/649046
- [64] Ryan Kemmer, Yeawon Yoo, Adolfo Escobedo, and Ross Maciejewski. 2020. Enhancing Collective Estimates by Aggregating Cardinal and Ordinal Inputs. Proceedings of the AAAI Conference on Human Computation and Crowdsourcing 8, 1 (Oct. 2020), 73–82. doi:10.1609/ hcomp.v8i1.7465
- [65] Eun-Sung Kim. 2020. Deep Learning and Principal-agent Problems of Algorithmic Governance: The New Materialism Perspective. Technology in Society 63 (11 2020), 101378. doi:10.1016/j.techsoc.2020.101378
- [66] Hannah Rose Kirk, Alexander Whitefield, Paul Röttger, Andrew Michael Bean, Katerina Margatina, Rafael Mosquera, Juan Manuel Ciro, Max Bartolo, Adina Williams, He He, Bertie Vidgen, and Scott A. Hale. 2024. The PRISM Alignment Dataset: What Participatory, Representative and Individualised Human Feedback Reveals About the Subjective and Multicultural Alignment of Large Language Models. In The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track. https://openreview.n et/forum?id=DFr5hteojx
- [67] Oliver Klingefjord, Ryan Lowe, and Joe Edelman. 2024. What are human values, and how do we align AI to them? arXiv:2404.10636 [cs.CY] https://arxiv.org/abs/2404.10636
- [68] Cody Kommers, Drew Hemment, Maria Antoniak, Joel Z. Leibo, Hoyt Long, Emily Robinson, and Adam Sobey. 2025. Meaning Is Not A Metric: Using LLMs to make cultural context legible at scale. (2025). arXiv:2505.23785 [cs.CL] https://arxiv.org/abs/2505.23785
- [69] Raphael Köster, Dylan Hadfield-Menell, Gillian K. Hadfield, and Joel Z. Leibo. 2020. Silly Rules Improve the Capacity of Agents to Learn Stable Enforcement and Compliance Behaviors. In Proceedings of the 19th International Conference on Autonomous Agents and MultiAgent Systems (Auckland, New Zealand) (AAMAS ’20). International Foundation for Autonomous Agents and Multiagent Systems, Richland, SC, 1887–1888. https://dl.acm.org/doi/10.5555/3398761.3399016
- [70] Jan Leike, David Krueger, Tom Everitt, Miljan Martic, Vishal Maini, and Shane Legg. 2018. Scalable agent alignment via reward modeling: A research direction. arXiv:1811.07871 [cs.LG] https://arxiv.org/abs/1811.07871
- [71] Stephen C. Levinson and Judith Holler. 2014. The origin of human multi-modal communication. Philosophical Transactions of the Royal Society B: Biological Sciences 369, 1651 (2014), 20130302. arXiv:https://royalsocietypublishing.org/doi/pdf/10.1098/rstb.2013.0302 doi:10.1098/rstb.2013.0302

- [72] Ting-an Lin. 2024. “Democratizing AI”’ and the Concern of Algorithmic Injustice (Extended Abstract). Proceedings of the AAAI/ACM Conference on AI, Ethics, and Society 7, 1 (Oct. 2024), 867–867. doi:10.1609/aies.v7i1.31686
- [73] Fangyu Liu, Emanuele Bugliarello, Edoardo Maria Ponti, Siva Reddy, Nigel Collier, and Desmond Elliott. 2021. Visually Grounded Reasoning across Languages and Cultures. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, Marie-Francine Moens, Xuanjing Huang, Lucia Specia, and Scott Wen-tau Yih (Eds.). Association for Computational Linguistics, Online and Punta Cana, Dominican Republic, 10467–10485. doi:10.18653/v1/2021.emnlp-main.818
- [74] Yang Liu, Yuanshun Yao, Jean-Francois Ton, Xiaoying Zhang, Ruocheng Guo, Hao Cheng, Yegor Klochkov, Muhammad Faaiz Taufiq, and Hang Li. 2024. Trustworthy LLMs: A Survey and Guideline for Evaluating Large Language Models’ Alignment. arXiv:2308.05374 [cs.AI] https://arxiv.org/abs/2308.05374
- [75] Ian R Macneil. 1977. Contracts: Adjustment of long-term economic relations under classical, neoclassical, and relational contract law. Northwestern University Law Review 72 (1977), 854. https://heinonline.org/HOL/LandingPage?handle=hein.journals/illlr72&div=46&i d=&page=
- [76] Michael A. Madaio, Luke Stark, Jennifer Wortman Vaughan, and Hanna Wallach. 2020. Co-Designing Checklists to Understand Organizational Challenges and Opportunities around Fairness in AI. In Proceedings of the 2020 CHI Conference on Human Factors in Computing Systems (Honolulu, HI, USA) (CHI ’20). Association for Computing Machinery, New York, NY, USA, 1–14. doi:10.1145/3313 831.3376445
- [77] Eric Maskin and Jean Tirole. 1999. Unforeseen Contingencies and Incomplete Contracts. The Review of Economic Studies 66, 1 (01 1999), 83–114. arXiv:https://academic.oup.com/restud/article-pdf/66/1/83/4683009/66-1-83.pdf doi:10.1111/1467-937X.00079
- [78] David Matsumoto. 2007. Culture, Context, and Behavior. Journal of Personality 75, 6 (2007), 1285–1320. arXiv:https://onlinelibrary.wiley.com/doi/pdf/10.1111/j.1467-6494.2007.00476.x doi:10.1111/j.1467-6494.2007.00476.x
- [79] Hugo Mercier and Dan Sperber. 2017. The Enigma of Reason. Harvard University Press. http://www.jstor.org/stable/j.ctv2sp3dd8
- [80] R. Millière. 2025. Normative conflicts and shallow AI alignment. Philosophical Studies 182 (2025), 2035–2078. doi:10.1007/s11098-02502347-3
- [81] Eric Mitchell, Charles Lin, Antoine Bosselut, Christopher D Manning, and Chelsea Finn. 2022. Memory-Based Model Editing at Scale. In Proceedings of the 39th International Conference on Machine Learning (Proceedings of Machine Learning Research, Vol. 162), Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvari, Gang Niu, and Sivan Sabato (Eds.). PMLR, 15817–15831. https: //proceedings.mlr.press/v162/mitchell22a.html
- [82] Margaret Mitchell, Simone Wu, Andrew Zaldivar, Parker Barnes, Lucy Vasserman, Ben Hutchinson, Elena Spitzer, Inioluwa Deborah Raji, and Timnit Gebru. 2019. Model Cards for Model Reporting. In Proceedings of the Conference on Fairness, Accountability, and Transparency (Atlanta, GA, USA) (FAT* ’19). Association for Computing Machinery, New York, NY, USA, 220–229. doi:10.1145/3287560.3287596
- [83] Carlos Montemayor. 2021. Language and Intelligence. Minds & Machines 31 (2021), 471–486. doi:10.1007/s11023-021-09568-5
- [84] Bryce Morsky and Erol Akçay. 2019. Evolution of social norms and correlated equilibria. Proceedings of the National Academy of Sciences 116, 18 (2019), 8834–8839. arXiv:https://www.pnas.org/doi/pdf/10.1073/pnas.1817095116 doi:10.1073/pnas.1817095116
- [85] Shravan Nayak, Kanishk Jain, Rabiul Awal, Siva Reddy, Sjoerd Van Steenkiste, Lisa Anne Hendricks, Karolina Stanczak, and Aishwarya Agrawal. 2024. Benchmarking Vision Language Models for Cultural Understanding. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen (Eds.). Association for Computational Linguistics, Miami, Florida, USA, 5769–5790. doi:10.18653/v1/2024.emnlp-main.329
- [86] Alondra Nelson. 2023. Thick Alignment. Keynote Talk at the 2023 ACM Conference on Fairness, Accountability, and Transparency (FAccT ’23). https://www.youtube.com/watch?v=Sq_XwqVTqvQ Accessed on 2026-01-13.
- [87] Mancur Olson. 1965. The logic of collective action: public goods and the theory of groups. Number 124 in Harvard economic studies. Harvard Univ. Press, Cambridge, Mass. 176 pages. https://www.hup.harvard.edu/catalog.php?isbn=9780674537514
- [88] OpenAI. 2016. Faulty Reward Functions in the Wild. https://openai.com/index/faulty-reward-functions/. Accessed: 2025-01-10.
- [89] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul F Christiano, Jan Leike, and Ryan Lowe. 2022. Training language models to follow instructions with human feedback. In Advances in Neural Information Processing Systems, S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh (Eds.), Vol. 35. Curran Associates, Inc., 27730–27744. https://proceedings.neurips.cc/paper_files/paper/2022/file/b1efde53be364a73914f58805a001731-Paper-Conference.pdf
- [90] Kanad Shrikar Pardeshi, Itai Shapira, Ariel D. Procaccia, and Aarti Singh. 2024. Learning Social Welfare Functions. In The Thirty-eighth Annual Conference on Neural Information Processing Systems. https://openreview.net/forum?id=7O6KtaAr8n
- [91] Joelle Pineau. 2020. The Machine Learning Reproducibility Checklist. https://www.cs.mcgill.ca/~jpineau/ReproducibilityChecklist.pdf Accessed: 2025-02-18.
- [92] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. 2023. Direct Preference Optimization: Your Language Model is Secretly a Reward Model. In Thirty-seventh Conference on Neural Information Processing Systems. https://openreview.net/forum?id=HPuSIXJaa9

- [93] Abhinav Sukumar Rao, Aditi Khandelwal, Kumar Tanmay, Utkarsh Agarwal, and Monojit Choudhury. 2023. Ethical Reasoning over Moral Alignment: A Case and Framework for In-Context Ethical Policies in LLMs. In Findings of the Association for Computational Linguistics: EMNLP 2023, Houda Bouamor, Juan Pino, and Kalika Bali (Eds.). Association for Computational Linguistics, Singapore, 13370–13388. doi:10.18653/v1/2023.findings-emnlp.892
- [94] Valerie F. Reyna and Charles J. Brainerd. 2008. Numeracy, ratio bias, and denominator neglect in judgments of risk and probability. Learning and Individual Differences 18, 1 (2008), 89–107. doi:10.1016/j.lindif.2007.03.011
- [95] Francesca Rossi, Kristen Brent Venable, and Toby Walsh. 2011. A Short Introduction to Preferences: Between AI and Social Choice (1st ed.). Morgan & Claypool Publishers. https://dl.acm.org/doi/10.5555/2049991
- [96] Keita Saito, Akifumi Wachi, Koki Wataoka, and Youhei Akimoto. 2023. Verbosity Bias in Preference Labeling by Large Language Models. In NeurIPS 2023 Workshop on Instruction Tuning and Instruction Following. https://openreview.net/forum?id=magEgFpK1y
- [97] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. 2017. Proximal Policy Optimization Algorithms. arXiv:1707.06347 [cs.LG] https://arxiv.org/abs/1707.06347
- [98] Alfred Schutz. 1976. The Problem of Rationality in the Social World. Springer Netherlands, Dordrecht, 64–88. doi:10.1007/978-94-0101340-6_3
- [99] Marie Schäfer, Daniel B. M. Haun, and Michael Tomasello. 2015. Fair Is Not Fair Everywhere. Psychological Science 26, 8 (2015), 1252–1260. arXiv:https://doi.org/10.1177/0956797615586188 doi:10.1177/0956797615586188 PMID: 26115962.
- [100] John H. Sears. 1921. Effective and Lawful Avoidance of Taxes. Virginia Law Review 8, 2 (1921), 77–85. http://www.jstor.org/stable/106 4452
- [101] Alex Y. Seita. 1984. Uncertainty and Contract Law. University of Pittsburgh Law Review 46, 75 (1984). https://ssrn.com/abstract=1692858
- [102] Amartya Sen. 1985. Social Choice and Justice: A Review Article. Journal of Economic Literature 23, December (1985). https: //scholar.harvard.edu/sen/publications/social-choice-and-justice-review-article Review article on K.J. Arrow’s Collected Papers: Social Choice and Justice.
- [103] Steven Shavell. 1980. Damage Measures for Breach of Contract. Bell Journal of Economics 11, 2 (1980), 466–490. https://EconPapers.r epec.org/RePEc:rje:bellje:v:11:y:1980:i:autumn:p:466-490
- [104] Tianhao Shen, Renren Jin, Yufei Huang, Chuang Liu, Weilong Dong, Zishan Guo, Xinwei Wu, Yan Liu, and Deyi Xiong. 2023. Large Language Model Alignment: A Survey. arXiv:2309.15025 [cs.CL] https://arxiv.org/abs/2309.15025
- [105] Ola Shorinwa, Zhiting Mei, Justin Lidard, Allen Z. Ren, and Anirudha Majumdar. 2024. A Survey on Uncertainty Quantification of Large Language Models: Taxonomy, Open Research Challenges, and Future Directions. arXiv:2412.05563 [cs.CL] https://arxiv.org/abs/ 2412.05563
- [106] Joar Skalse, Nikolaus H. R. Howe, Dmitrii Krasheninnikov, and David Krueger. 2024. Defining and characterizing reward hacking. In Proceedings of the 36th International Conference on Neural Information Processing Systems (New Orleans, LA, USA) (NIPS ’22). Curran Associates Inc., Red Hook, NY, USA, Article 687, 12 pages. https://proceedings.neurips.cc/paper_files/paper/2022/hash/3d719fee332ca a23d5038b8a90e81796-Abstract-Conference.html
- [107] Irene Solaiman and Christy Dennison. 2021. Process for adapting language models to society (PALMS) with values-targeted datasets. In Proceedings of the 35th International Conference on Neural Information Processing Systems (NIPS ’21). Curran Associates Inc., Red Hook, NY, USA, Article 448, 13 pages. https://dl.acm.org/doi/abs/10.5555/3540261.3540709
- [108] Taylor Sorensen, Pushkar Mishra, Roma Patel, Michael Henry Tessler, Michiel A. Bakker, Georgina Evans, Iason Gabriel, Noah Goodman, and Verena Rieser. 2025. Value Profiles for Encoding Human Variation. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, Christos Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng (Eds.). Association for Computational Linguistics, Suzhou, China, 2047–2095. doi:10.18653/v1/2025.emnlp-main.106
- [109] Taylor Sorensen, Jared Moore, Jillian Fisher, Mitchell L. Gordon, Niloofar Mireshghallah, Christopher Michael Rytting, Andre Ye, Liwei Jiang, Ximing Lu, Nouha Dziri, Tim Althoff, and Yejin Choi. 2024. Position: A Roadmap to Pluralistic Alignment. In Proceedings of the 41st International Conference on Machine Learning. https://openreview.net/forum?id=gQpBnRHwxM
- [110] Nisan Stiennon, Long Ouyang, Jeff Wu, Daniel M. Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul Christiano.

2020. Learning to summarize from human feedback. In Proceedings of the 34th International Conference on Neural Information Processing Systems (Vancouver, BC, Canada) (NIPS ’20). Curran Associates Inc., Red Hook, NY, USA, Article 253, 14 pages. https: //dl.acm.org/doi/abs/10.5555/3495724.3495977

- [111] Zhiqing Sun, Yikang Shen, Qinhong Zhou, Hongxin Zhang, Zhenfang Chen, David Cox, Yiming Yang, and Chuang Gan. 2023. Principledriven self-alignment of language models from scratch with minimal human supervision. In Proceedings of the 37th International Conference on Neural Information Processing Systems (New Orleans, LA, USA) (NIPS ’23). Curran Associates Inc., Red Hook, NY, USA, Article 115, 55 pages. https://proceedings.neurips.cc/paper_files/paper/2023/hash/0764db1151b936aca59249e2c1386101-AbstractConference.html
- [112] Zhi-Xuan Tan, Micah Carroll, Matija Franklin, Geoff Keeling, and Iason Gabriel. 2025. Beyond Preferences in AI Alignment. Philosophical Studies 182 (2025), 1813–1863. doi:10.1007/s11098-024-02249-w

- [113] Kumar Tanmay, Aditi Khandelwal, Utkarsh Agarwal, and Monojit Choudhury. 2023. Probing the Moral Development of Large Language Models through Defining Issues Test. arXiv:2309.13356 [cs.CL] https://arxiv.org/abs/2309.13356
- [114] Petros Terzis. 2024. Against Digital Constitutionalism. European Law Open First View (July 2024). doi:10.2139/ssrn.4896078
- [115] Jean Tirole. 1999. Incomplete Contracts: Where Do We Stand? Econometrica 67, 4 (1999), 741–781. http://www.jstor.org/stable/2999457
- [116] Alex Turner, Logan Smith, Rohin Shah, Andrew Critch, and Prasad Tadepalli. 2021. Optimal Policies Tend To Seek Power. In Advances in Neural Information Processing Systems, M. Ranzato, A. Beygelzimer, Y. Dauphin, P.S. Liang, and J. Wortman Vaughan (Eds.), Vol. 34. Curran Associates, Inc., 23063–23074. https://proceedings.neurips.cc/paper_files/paper/2021/file/c26820b8a4c1b3c2aa868d6d57e14a79Paper.pdf
- [117] Alexander Matt Turner, Dylan Hadfield-Menell, and Prasad Tadepalli. 2020. Conservative Agency via Attainable Utility Preservation. In Proceedings of the AAAI/ACM Conference on AI, Ethics, and Society (New York, NY, USA) (AIES ’20). Association for Computing Machinery, New York, NY, USA, 385–391. doi:10.1145/3375627.3375851
- [118] Alexander Wei, Nika Haghtalab, and Jacob Steinhardt. 2023. Jailbroken: How Does LLM Safety Training Fail?. In Thirty-seventh Conference on Neural Information Processing Systems. https://openreview.net/forum?id=jA235JGM09
- [119] Marcus Williams, Micah Carroll, Adhyyan Narang, Constantin Weisser, Brendan Murphy, and Anca Dragan. 2024. On Targeted Manipulation and Deception when Optimizing LLMs for User Feedback. arXiv:2411.02306 [cs.LG] https://arxiv.org/abs/2411.02306
- [120] O.E. Williamson. 1973. Markets and Hierarchies, Analysis and Antitrust Implications: A Study in the Economics of Internal Organization. Free Press. http://www.jstor.org/stable/1817092
- [121] Mohammad Yaghini, Patty Liu, Franziska Boenisch, and Nicolas Papernot. 2023. Learning to Walk Impartially on the Pareto Frontier of Fairness, Privacy, and Utility. In NeurIPS 2023 Workshop on Regulatable ML. https://openreview.net/forum?id=R5MTSLPyYZ
- [122] Haolan Zhan, Zhuang Li, Xiaoxi Kang, Tao Feng, Yuncheng Hua, Lizhen Qu, Yi Ying, Mei Rianto Chandra, Kelly Rosalin, Jureynolds Jureynolds, Suraj Sharma, Shilin Qu, Linhao Luo, Ingrid Zukerman, Lay-Ki Soon, Zhaleh Semnani Azad, and Reza Haf. 2024. RENOVI: A Benchmark Towards Remediating Norm Violations in Socio-Cultural Conversations. In Findings of the Association for Computational Linguistics: NAACL 2024, Kevin Duh, Helena Gomez, and Steven Bethard (Eds.). Association for Computational Linguistics, Mexico City, Mexico, 3104–3117. doi:10.18653/v1/2024.findings-naacl.196
- [123] Siyan Zhao, John Dang, and Aditya Grover. 2024. Group Preference Optimization: Few-Shot Alignment of Large Language Models. In The Twelfth International Conference on Learning Representations. https://openreview.net/forum?id=DpFeMH4l8Q
- [124] Zhanhui Zhou, Jie Liu, Jing Shao, Xiangyu Yue, Chao Yang, Wanli Ouyang, and Yu Qiao. 2024. Beyond One-Preference-Fits-All Alignment: Multi-Objective Direct Preference Optimization. In Findings of the Association for Computational Linguistics: ACL 2024, Lun-Wei Ku, Andre Martins, and Vivek Srikumar (Eds.). Association for Computational Linguistics, Bangkok, Thailand, 10586–10613. doi:10.18653/v1/2024.findings-acl.630
- [125] Simon Zhuang and Dylan Hadfield-Menell. 2020. Consequences of Misaligned AI. In Advances in Neural Information Processing Systems, H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin (Eds.), Vol. 33. Curran Associates, Inc., 15763–15773. https: //proceedings.neurips.cc/paper_files/paper/2020/file/b607ba543ad05417b8507ee86c54fcb7-Paper.pdf
- [126] Daniel M. Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B. Brown, Alec Radford, Dario Amodei, Paul Christiano, and Geoffrey Irving. 2020. Fine-Tuning Language Models from Human Preferences. arXiv:1909.08593 [cs.CL] https://arxiv.org/abs/1909.08593
- [127] Caleb Ziems, Jane Yu, Yi-Chia Wang, Alon Halevy, and Diyi Yang. 2022. The Moral Integrity Corpus: A Benchmark for Ethical Dialogue Systems. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), Smaranda Muresan, Preslav Nakov, and Aline Villavicencio (Eds.). Association for Computational Linguistics, Dublin, Ireland, 3755–3773. doi:10.18653/v1/2022.acl-long.261
- [128] Thomas P Zollo, Todd Morrill, Zhun Deng, Jake Snell, Toniann Pitassi, and Richard Zemel. 2024. Prompt Risk Control: A Rigorous Framework for Responsible Deployment of Large Language Models. In The Twelfth International Conference on Learning Representations. https://openreview.net/forum?id=5tGGWOijvq
- [129] Andy Zou, Zifan Wang, Nicholas Carlini, Milad Nasr, J. Zico Kolter, and Matt Fredrikson. 2023. Universal and Transferable Adversarial Attacks on Aligned Language Models. arXiv:2307.15043 [cs.CL] https://arxiv.org/abs/2307.15043

