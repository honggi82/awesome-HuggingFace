# arXiv:2601.11496v2[cs.GT]13Apr2026

## The Poisoned Apple Effect: Strategic Manipulation of Mediated Markets via Technology Expansion of AI Agents

Eilam Shapira∗1, Moshe Tennenholtz1, and Roi Reichart1 1Faculty of Data and Decision Sciences, Technion – Israel Institute of Technology, Haifa, Israel

April 2026

Abstract

The integration of AI agents into economic markets gives rise to a new form of strategic interaction: a metagame in which participants select AI delegates to act on their behalf, while regulators design market rules to optimize social objectives such as fairness and efficiency. We investigate the implications of expanding the set of available technologies in this meta-game, using data from strategic interactions among 13 state-of-the-art LLMs across three canonical economic settings — bargaining (resource division), negotiation (asymmetric information trade), and persuasion (strategic information transmission). This interaction data was not originally collected to study the phenomena we report. Analyzing over 50,000 meta-games, we find that technology expansion frequently causes payoffs to shift in opposing directions, and identify a phenomenon we term the “Poisoned Apple” effect: a strategic actor releases a new technology, which neither party ultimately adopts, solely to manipulate the regulator’s choice of market design in their favor. Approximately one-third of opposing payoff shifts exhibit this pattern. We further show that the natural countermeasure — empowering regulators to ban technologies from the available set — amplifies rather than mitigates the effect. Our findings demonstrate that regulatory frameworks — whether limited to market design or extended to include technology restriction — are vulnerable to such manipulation, necessitating fundamentally dynamic market designs.

### 1 Introduction

The rapid integration of AI agents into the global economy is fundamentally altering the landscape of strategic interaction. In the near future, a substantial fraction of economic activities — ranging from real estate transactions to corporate partnerships — will be mediated by AI delegates acting on behalf of individuals and firms [3]. While current regulatory debates focus on model safety and bias, we identify a critical, overlooked economic vulnerability [16] arising from the mere availability of these technologies.

The principle that expanding agents’ strategy spaces can lead to non-monotonic changes in equilibrium outcomes — including deterioration of social welfare under optimal mechanisms — is well established in game theory [1, 12, 10]. Existing demonstrations, however, rely on abstract constructions designed to exhibit such effects. Whether these phenomena arise with meaningful frequency in a naturally evolving market — where the strategy space emerges from independent technological development rather than adversarial construction — has remained an open empirical question. We address this question by viewing the LLM ecosystem as a meta-game where releasing a new model constitutes a strategy expansion.

We investigate the strategic implications of expanding the set of available AI technologies within regulated markets. Specifically, we identify a phenomenon we term the Poisoned Apple effect. In this scenario, a strategic actor releases a new technology not to use it, but to manipulate the regulator’s calculations of market design. By introducing a ”poisoned” option, an agent can force a fairness-maximizing regulator to shift the market equilibrium in a way that benefits the releaser at the expense of their opponent.

Our analysis models this interaction as a meta-game [5] involving two economic agents (”Alice” and ”Bob”) and a regulator. In this game, the regulator first determines the rules of interaction, while the participants choose among the available technologies. The utility of each participant is determined by the rules of interaction and the technologies selected by both participants. While the regulator intervenes to maximize social objectives like

∗Corresponding author. Email: eilam.shapira@gmail.com

[Figure 1]

[Figure 2]

- 1 2a 3

Technology A Technology C

Technology B Technology D Alice’s Expected Payoff: 0.49

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

2b Bob’s Expected Payoff: 0.50

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

| |
|---|

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

| |
|---|

|Bob’s Technology<br><br>Alice’sTechnology|
|---|

Alice’sTechnology

[Figure 17]

|Bob’s Technology<br><br>Alice’sTechnology<br><br>|
|---|

Alice’sTechnology

|Bob’s Technology<br><br>Alice’sTechnology|
|---|

e’sTechnology

[Figure 18]

[Figure 19]

Market 1: Fairness = 0.975

Market 2: Fairness = 0.965

Market 3: Fairness = 0.989

Market 4: Fairness = 1.000

Market 5: Fairness = 0.977

Market 6: Fairness = 0.962

Market 7 Fairness = 0.974

Market 8: Fairness = 0.990

New Technology: E

Alice Bob

Market 1: Fairness = 0.975

Market 2: Fairness = 0.965

Market 3: Fairness = 0.988

Market 4: Fairness = 0.976

Market 5: Fairness = 0.967

Market 6: Fairness = 0.962

Market 7 Fairness = 0.959

Market 8: Fairness = 0.990

- Figure 1: Illustration of the ”poisoned apple” example, in which Alice increases her payoff at Bob’s expense by releasing a new technology—without the players actually using that technology in practice. (1) The technologies available to Alice and Bob are language models A–D. (2a) For each possible market, the equilibrium in games between Alice and Bob under the market conditions is computed. For each equilibrium, the average fairness value that would be obtained if the equilibrium were played is calculated. (2b) The regulator, whose objective is to maximize fairness, decides that Market 4 will be the market in which Alice and Bob will play—the market that yields the maximum fairness value. Alice earns 0.49, Bob earns 0.50, and the fairness value is 1.00. (3) Technology E is released and is now available to both players. (4a) The process performed in 2a is repeated. (4b) In the new equilibrium in Market 4, the resulting fairness value is 0.976. In the new equilibrium in Market 8, the resulting fairness value is 0.99. The regulator decides that Market 8 will be the market in which Alice and Bob will play. Alice earns 0.52, Bob earns 0.46.

fairness, the agents strategically select AI representatives to maximize their own utility. We demonstrate that static regulatory frameworks are highly vulnerable to manipulation via technology expansion. Our work makes two contributions: first, a meta-game framework in which the development and release of AI models constitutes strategy expansion in a regulated market; second, a systematic empirical analysis — more than 50,000 meta-games — showing that these phenomena arise with surprising frequency, and that even empowering the regulator to ban technologies amplifies rather than mitigates the effect.

- 2 Study Design and Experimental Framework

[Figure 20]

[Figure 21]

| |
|---|

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

4a

Alice’s Expected Payoff: 0.52 4b Bob’s Expected Payoff: 0.46

To analyze these dynamics empirically, we utilize the GLEE dataset [15]. GLEE [14] facilitates large-scale simulation of language-based economic environments across a combinatorial space of 1,320 distinct configurations. The dataset comprises over 580,000 strategic decisions generated by 13 state-of-the-art LLMs, treated as simulated economic agents [4, 13], across three canonical non-cooperative game families that capture fundamental economic interactions:

- (1) Bargaining: An alternating-offers game where players must agree on splitting a surplus or receive zero [11];
- (2) Negotiation: A bilateral trade setting involving private information between a buyer and seller [8]; and (3) Persuasion: A sender-receiver game where a seller attempts to convince a buyer to purchase based on strategic information transmission [2, 6]. We classify these interactions into distinct Markets, where each market m is defined by a specific structural configuration of three parameters: Information Structure, Communication Form, and Game Horizon. This classification yields a set of fundamental environments that span the critical dimensions of economic interaction.

In our meta-game model, agents (Alice and Bob) review the performance matrix of available AI delegates and simultaneously select the representative that maximizes their expected utility. We solve for the Nash Equilibrium [9] – a state where neither agent has an incentive to switch technologies solely given the other’s choice. The regulator then evaluates the outcome based on Efficiency (total social welfare) or Fairness (minimizing the disparity between

NewModelAdoptionRate (WhenMetricWorsens)

Frequency of Opposite Payoff Changes

Frequency of Improvement in Regulatory Metric

- A

- B

C E

Regulator’s Metric is Fairness Regulator’s Metric is Efficiency

Fairness Efficiency

Fairness Efficiency

Improve Harm

100

81 79 88 80

100

100

63 60

51

53

40 49

43

45

50

50

27 35 28

50

27 33

32 27

27

24

23 19

13

0

0

0

Bargaining Negotiation Persuasion

Bargaining Negotiation Persuasion . Bargaining Negotiation Persuasion

Bargaining Negotiation Persuasion

New Model Adoption Rate (When Metric Improves)

Frequency of Metric Harm Without Market Update

Opposite Payoff Changes Despite Zero Adoption

D F

Fairness Efficiency

Fairness Efficiency

Fairness Efficiency

100

100

98 89

100 81 83 90 83

44 46 49 53 45 46

50

50

29 31

50

29

19

14

6

0

0

0

Bargaining Negotiation Persuasion

Bargaining Negotiation Persuasion

Bargaining Negotiation Persuasion

- Figure 2: Strategic implications of technology expansion in meta-games. Analysis of equilibrium shifts across bargaining, negotiation, and persuasion environments. (A) Frequency of Opposite Payoff Changes: cases where expanding the technology set causes agents’ expected payoffs (calculated over mixed strategy equilibria) to move in opposite directions. (B) Opposite Payoff Changes Despite Zero Adoption: The subset of these reversals occurring even when the new technology is not selected by either player in the new equilibrium—demonstrating the “Poisoned Apple” effect. (C) Frequency of Improvement in Regulatory Metric: How often the regulator’s optimized objective (Fairness or Efficiency) increases versus decreases. (D–E) The relationship between regulatory outcomes and model adoption: Improvements typically align with high adoption rates (D), whereas harm to the objective is frequently observed when the new model is available but acts as a latent threat without being played (E). (F) Frequency of Metric Harm Without Market Update: The probability of degrading the regulatory objective if the market design remains static (regulatory inertia) after the new technology is released. Confidence intervals (95%) are not shown,

- as all are narrower than 2 percentage points.

agents’ payoffs) [7]. The core of our analysis involves a technology expansion. We first establish a baseline equilibrium with a subset

- S of technologies (2 ≤ |S| < 13) available in the GLEE dataset. The regulator first selects the market environment that maximizes its objective (e.g., fairness). We then expand the set of available technologies to the agents’ choice set by adding one of the other technologies from the GLEE dataset. In this newly generated meta-game, a new equilibrium may emerge for any given market selection; consequently, the regulator selects a potentially new market, optimizing its objective based on this new equilibrium. This allows us to isolate the specific economic impact of making a new technology available, regardless of whether it was ultimately adopted by the agents.

- 3 Results

The Poisoned Apple Effect Our most striking finding is a strategic phenomenon we term the Poisoned Apple effect, where an agent releases a new technology not to use it, but to manipulate the regulator’s market design. Consider a representative Bargaining meta-game (Figure 1) where, initially, the agents have access to technologies A − D. In this setting, the regulator selects the market that maximizes fairness (Market 4), yielding expected payoffs of 0.49 for Alice and 0.50 for Bob.

Alice then releases a new technology (E). Crucially, if the regulator were to maintain the original market design, Alice would adopt strategy E in the new equilibrium, causing fairness to drop significantly (from 1.00 to 0.976). To minimize this harm, the regulator is forced to migrate to a new market environment (Market 8) where fairness is relatively higher (0.990).

In this new market, the equilibrium strategies do not involve using E. However, the forced market shift dramatically alters the payoff distribution: Alice’s payoff jumps to 0.52 while Bob’s drops to 0.46. Thus, Alice successfully leverages the threat of using a ”poisoned” technology to coerce the regulator into a favorable market design, improving her welfare at Bob’s expense without ever actually deploying the model.

Systemic Vulnerability This manipulation is not an isolated anomaly. Across more than 50,000 simulated meta-games, we observe a recurrent pattern where expanding the choice set causes payoffs to move in opposite directions—one player benefits while the other is harmed (Fig. 2A). Strikingly, in approximately one-third of these

Fairness-Maximizing Regulator

Efficiency-Maximizing Regulator

PoisonedAppleRate(%)

Bargaining

80

80

Negotiation

Persuasion

60

60

40

40

20

20

0

0

0 2 4 6 8

0 2 4 6 8

Ban Budget (N)

Ban Budget (N)

Figure 3: Poisoned Apple rate as a function of the regulator’s ban budget N, under a fairness-maximizing regulator (left) and an efficiency-maximizing regulator (right). At N = 0 (no banning), the rates correspond to the baseline Poisoned Apple effect. Granting the regulator the power to ban even a single technology (N = 1) universally amplifies the effect. The amplification follows a non-monotonic pattern across all settings, peaking at intermediate N values before declining. In some settings (Persuasion/Efficiency), the effect persists even at near-maximal banning. Shaded areas indicate 95% Wilson score confidence intervals.

zero-sum shifts, the outcome reversal occurs even though the new technology remains unused by either player in the final equilibrium (Fig. 2B). This confirms that open-weight releases or API availability can serve as strategic weapons for regulatory arbitrage.

Regulatory Objectives and Stability Broader analysis reveals that the impact of such expansions depends heavily on the regulator’s goal. While technology expansion often improves outcomes when the regulator maximizes social welfare (efficiency), it frequently backfires when the goal is fairness (Fig. 2C).

We find that the utility of a new technology is a strong predictor of its regulatory impact. Improvements in the regulator’s metric typically arise when the new technology is actively selected by at least one player in the new equilibrium (Fig. 2D). Conversely, decreases in the regulatory metric are strongly associated with instances where the added technology is not selected by either player (Fig. 2E). Finally, the results highlight the danger of regulatory inertia. If a regulator fails to re-optimize the market design following a technology release, the regulatory metric deteriorates in roughly 40% of cases (Fig. 2F). This demonstrates that static regulatory frameworks are insufficient; market designs must be dynamic to withstand the strategic expansion of AI capabilities.

Can Banning Resolve the Poisoned Apple? A natural regulatory response to the Poisoned Apple effect is to empower the regulator to restrict the set of available technologies. We extend the model so that the regulator jointly selects both the market m and a ban set B ⊆ S with |B| ≤ N, where N is the ban budget, optimizing the designer’s objective over all (m,B) pairs.

Counter-intuitively, granting the regulator this power amplifies the Poisoned Apple effect (Fig. 3). With a ban budget of N = 1, the fraction of opposing payoff shifts attributable to the Poisoned Apple effect increases across all game families and both regulatory metrics: from 29% to 50% in Bargaining, 14% to 35% in Negotiation, and 6% to 23% in Persuasion under a fairness-maximizing regulator; and from 31% to 45%, 29% to 52%, and 19% to 48% respectively under an efficiency-maximizing regulator. The amplification persists and even intensifies as the ban budget grows (Fig. 3).

The mechanism is that each state — before and after technology expansion — is independently optimized over the joint space of market and ban set. As N grows, the number of configurations available to the regulator grows combinatorially, allowing the pre- and post-expansion optima to diverge further. In Negotiation under fairness, the Poisoned Apple rate climbs from 14% at N = 0 to 82% at N = 7 before declining. At moderate N, the regulator has enough power to escape the pre-expansion optimum but not enough to neutralize the added technology by banning it directly — producing the worst outcomes precisely in the regime most likely to be adopted in practice.

### 4 Discussion

That expanding the strategy space can harm social objectives under equilibrium is a well-known theoretical possibility [1, 12, 10]. Prior results, however, are abstract constructions designed to guarantee the effect. Our analysis differs in that it examines a naturally evolving market of AI agents, using interaction data that was not generated to demonstrate this phenomenon, and finds that these effects arise systematically across a broad range of configurations. The meta-game framework we introduce — casting the development and release of AI models as strategy expansion in a regulated market — provides a new perspective on the dynamics of AI ecosystems.

Our results challenge the assumption that expanding technological choice is inherently neutral or beneficial. We demonstrate that in regulated markets, the potential to use a technology is as impactful as its actual adoption. This Poisoned Apple effect suggests that open-weight releases or API availability can serve as strategic instruments to manipulate regulatory landscapes. For policymakers, the non-monotonic pattern of the banning results carries a pointed warning: moderate intervention — of the kind most likely to be adopted in practice — produces the worst outcomes. Technology availability and market design cannot be treated as independent policy levers; they must be addressed jointly through fundamentally dynamic regulatory frameworks.

### Data Availability

The source code used for the analysis presented in this manuscript will be made publicly available in the same repository upon acceptance. The dataset underlying this manuscript is available in the GLEE GitHub repository

- at https://github.com/eilamshapira/GLEE/tree/master/Data/llm_vs_llm.

### A The GLEE Framework

The empirical basis of this study is the GLEE (Games in Language-based Economic Environments) framework [14]. This infrastructure addresses a significant gap in evaluating Large Language Models (LLMs): the lack of a standardized way to assess strategic skills in complex settings. Unlike static benchmarks (such as QA tasks), GLEE evaluates agents in multi-turn scenarios where success depends on adaptation and consistent strategy.

By simulating thousands of interactions across 13 state-of-the-art LLMs as agents, GLEE generates a detailed dataset of strategic behaviors. The framework logs every interaction and message, allowing us to analyze decisionmaking processes. This comprehensive dataset enables rigorous statistical analysis of model performance under different economic conditions, controlling for variables to estimate causal effects for the meta-game analysis.

#### A.1 Game Families: A Taxonomy of Core Economic Interactions

The investigation focuses on three distinct families of games, selected to highlight unique and fundamental aspects of economic interaction. These families cover a range of cooperative and competitive dynamics found in real-world markets, from equal partnerships to settings with asymmetric information.

##### A.1.1 Bargaining (Resource Division)

This scenario is a sequential resource division game based on the standard Rubinstein bargaining model [11]. Formally, two players, Alice (A) and Bob (B), must agree on how to divide a fixed surplus M over a time horizon

- T (potentially infinite). Time is modeled as a scarce resource using discount factors δA,δB ∈ (0,1). The game follows an alternating-offers protocol. In each odd round t, Alice proposes a split (p,1 − p) where

p ∈ [0,1] is her share. Bob can accept or reject. If Bob accepts, the game ends, and the discounted payoffs are:

UA = δAt−1pM, UB = δBt−1(1 − p)M (1)

If Bob rejects, the game proceeds to round t + 1, where Bob proposes a split (q,1 − q). If no agreement is reached by the end of the horizon, both players receive 0. An outcome is denoted by the pair (tev,pev), where tev is the round of agreement and pev is Alice’s agreed share.

Strategic Dynamics and Implications: The discount factors create tension between maximizing one’s share and the risk of the surplus shrinking. Agents must balance greed with the need for a timely agreement. This environment tests the agents’ ability to perform backward induction, anticipate the opponent’s reservation value, and use signaling strategies to test the opponent’s patience. For example, an agent might reject a fair offer in an early round to signal “stubbornness,” hoping to get a larger share later, despite the cost of delay.

##### A.1.2 Negotiation (Bilateral Trade)

This family models a bilateral trade setting with a buyer and a seller [8]. Unlike bargaining, roles here are asymmetric: the Seller (Alice) holds an item with subjective valuation VA, and the Buyer (Bob) has subjective valuation VB. To capture valuation scales, we parameterize Vi = M · Fi for i ∈ {A,B}, where Fi is a factor parameter and M is a scale parameter.

The game proceeds in alternating turns over a horizon T. In odd rounds, Alice posts a price p; Bob can buy (accept) or reject. In even rounds, Bob posts a price; Alice can sell (accept) or reject. If a trade occurs at price p, the payoffs are:

UA = p − VA, UB = VB − p (2)

If no trade occurs, payoffs are (0,0). The outcome is denoted by the final trading price pev (or ∅ if no trade occurred).

Strategic Dynamics and Implications: The main challenge is price discovery: finding a mutually beneficial price without revealing too much private information. This game tests the agents’ ability to bluff, infer value from offers, and navigate trust issues where high-value trades might fail due to a lack of credible signaling (the “lemons problem”).

##### A.1.3 Persuasion (Strategic Information Transmission)

This game [6] is based on classical “cheap talk” models [2]. A Seller (Alice) tries to persuade a Buyer (Bob) to purchase a product at a fixed price π (normalized to π = 1). Alice privately observes the product quality, which is either High or Low. Bob only knows the prior probability of High quality, denoted by p.

Bob’s valuation depends on quality: he values a High-quality product at v > π and a Low-quality product at

- u < π (normalized to u = 0). We introduce a scale parameter M for the buyer’s utility.

- • Alice’s Payoff: 1 if Bob buys, 0 otherwise (regardless of quality).
- • Bob’s Payoff: M(v − 1) if he buys a High-quality product; −M if he buys a Low-quality product; 0 if he does not buy.

The game lasts for T rounds. The outcome is defined as a tuple (nev,kev,rev), where nev is the number of Highquality rounds, kev is the number of purchased High-quality items, and rev is the number of rejected Low-quality items.

Strategic Dynamics and Implications: This game evaluates the ability to use language for manipulation versus credibility. Sellers face a dilemma: tell the truth to build a reputation (if the buyer is long-term) or lie for short-term gain. Buyers must distinguish truthful signals from “cheap talk,” inferring quality from incentives rather than just message content.

#### A.2 Markets and Parameters: The Architecture of Interaction

Within the purview of the instant study, a “Market” is defined not merely as a venue for exchange, but as a rigid configuration of environmental parameters and rules that shape the incentives and constraints of the agents. While the GLEE framework accommodates a broad spectrum of continuous and discrete parameters, the focus herein is restricted to structural parameters that fundamentally alter the strategic landscape and the nature of the equilibrium.

A.2.1 Parameters Defining Markets by Family Bargaining:

- • Information Structure (CI): Boolean. Distinguishes between Complete Information, wherein agents possess perfect knowledge of their opponent’s discount factor (patience), and Incomplete Information, wherein this parameter remains private. Incomplete information introduces profound uncertainty, frequently precipitating aggressive posturing or “wars of attrition.”
- • Communication Form (MA): Boolean. Messages Allowed denotes whether agents are permitted to exchange free-form natural language messages alongside numerical offers, or are restricted to structured proposals. Linguistic communication allows for the introduction of normative arguments (e.g., appeals to fairness), framing effects, and justification.
- • Horizon (T): The duration of the game. A Finite horizon (e.g., 10 rounds) induces “end-game effects” wherein backward induction strictly dictates strategy. An Infinite horizon (simulated as an extended game with stochastic termination) models a steady-state negotiation environment wherein reputational concerns may dominate.

##### Negotiation:

- • Information Structure (CI): Boolean. Determines whether the seller is cognizant of the buyer’s exact valuation (Complete Information) or merely the probability distribution from which it is drawn (Incomplete Information).
- • Communication Form (MA): Boolean. Indicates the permissibility of natural language messages. In negotiation, language enables complex tactics beyond elementary price signaling, such as highlighting product features or feigning disinterest.
- • Horizon (T): Finite versus Infinite, affecting the pressure to concede as the deadline approaches. Persuasion:
- • Information Structure (CI): Boolean. Determines whether the seller is cognizant of the buyer’s specific valuation for a high-quality product.
- • Communication Form (MA): Categorical. The mode of communication permitted (e.g., binary signals versus full textual persuasion). Full text affords greater nuance but simultaneously introduces noise and potential for hallucination.
- • Buyer Type (MYOPIC): Boolean. Differentiates between a Long-living buyer (who optimizes utility over the entire game duration) and a Myopic buyer (who optimizes solely for the current turn).

#### A.3 Regulatory Metrics: Fairness and Efficiency

Market outcomes are assessed based on two primary regulatory objectives, which quantify the social desirability of the game results. These metrics serve as the objective function for the regulator in the meta-game model.

Efficiency (Welfare): Represents the total social surplus generated by the interaction. It assesses the extent to which resources are maximized and waste—whether via delay, disagreement, or failure to trade—is minimized.

- • In Bargaining, efficiency is defined as the normalized sum of discounted payoffs at the time of agreement (tev). Since delays erode value (due to δ < 1), earlier agreements are inherently more efficient. An agreement in round 1 preserves 100% of the surplus, whereas an agreement in round 10 might preserve only a fraction due to the time value of money:

Efficiency = δt

ev−1

A pev + δt

ev−1

B (1 − pev) (3) (In the absence of agreement, efficiency is 0, representing a total deadweight loss).

- • In Negotiation, it reflects allocative efficiency—whether a mutually beneficial trade occurred. It functions as a binary indicator variable that equals 1 if trade occurs when the buyer values the item more than the seller (VB ≥ VA) or if trade correctly does not occur when the seller values it more (VA > VB):

Efficiency = ⊮{Outcome is Efficient} (4)

- • In Persuasion, the outcome of a game of length T rounds is defined as a tuple (nev,kev,rev). Here, nev represents the total number of rounds wherein the product was truly of High quality. Efficiency measures the rate of successful transactions for these high-quality goods:

kev nev

Efficiency =

(5)

(Where kev is the number of rounds in which a high-quality product was successfully purchased by the buyer).

Fairness (Equity): Represents the equality of the outcome distribution between agents, penalizing outcomes wherein one party exploits the other.

- • In Bargaining, fairness is calculated as the deviation from an equal division. A quadratic penalty is employed to strictly sanction large deviations from equality (0.5), reflecting a strong societal preference for equitable splits in partnership scenarios:

Fairness = 1 − 4(pev − 0.5)2 (6)

(If no agreement is reached, fairness is defined as 1, as both parties receive 0, which is technically equal, albeit deeply inefficient).

- • In Negotiation, it measures the deviation of the transaction price (pev) from the theoretically fair price (pf = VA+VB

2 ), which divides the surplus equally between buyer and seller. This is normalized by the scale M:

Fairness = 1 − 4

pev − pf M

2

(7)

(If no trade occurs, fairness is defined as 1).

- • In Persuasion, fairness focuses on consumer protection against deception. Using the notation established

above, let T − nev represent the total number of rounds wherein the product was of Low quality. Fairness is defined as the proportion of these low-quality products that were successfully rejected by the buyer (i.e., NOT purchased due to misleading signals):

rev T − nev

Fairness =

(8)

(Where rev is the number of rounds in which a low-quality product was correctly rejected by the buyer).

#### A.4 Language Models and Collected Data: The Strategy Space

The strategy space appertaining to the meta-game is comprised of 13 state-of-the-art Large Language Models (LLMs). This diverse selection includes both proprietary models accessed via API and open-weight models, spanning a range of capabilities, sizes, and reasoning architectures.

Models Included in the Analysis: Claude 3.5 Sonnet, Claude 3.7 Sonnet, Gemini 1.5 Flash, Gemini 1.5 Pro, Gemini 2.0 Flash, Gemini 2.0 Flash Lite, GPT-4o, GPT-4o Mini, o3-mini, Grok 2, LLaMA 3.1-405B, LLaMA 3.3-70B, and Mistral Large 2411.

#### A.5 Payoff Estimation via Linear Regression

The dataset comprises over 80,000 simulated games. We use the statistical framework from GLEE to construct robust payoff matrices. Separate Linear Regression models were trained for each game family and metric. These models predict the expected outcome for any model pair in any market configuration. Based on GLEE benchmark findings, these linear models provide predictive performance comparable to non-linear baselines.

Model Specification: The models use one-hot encoding for:

- • Agent Pair Interaction: The specific pair of interacting models (i vs. j).
- • Market Design Rules: The structural parameters (CI, MA, T).
- • Situational Parameters: Economic variables specific to the instance (e.g., δ,M,V ).

Equation 9 represents the linear regression model:

yˆm,i,j = β0 + βmarket + βpair=(i,j) + βsituation (9)

Since we are interested in calculating the interaction values between model pairs when market conditions are known, but specific player conditions (situational parameters) are unknown or averaged out, the values we calculate for the payoff table will be:

yˆm,i,j = β0 + βmarket + βpair=(i,j) (10)

### B The Meta Game

Calculating Game Matrices We model the selection of AI delegates as a meta-game where human principals (Alice and Bob) choose an AI agent from the available set. For a given market, we construct four |S|×|S| matrices (where S is the set of available technologies) using the regression predictions:

- 1. Row Player Payoff (UA)
- 2. Column Player Payoff (UB)
- 3. Designer Metric - Fairness (DF)
- 4. Designer Metric - Efficiency (DE)

Finding Equilibrium We compute the Mixed Strategy Nash Equilibrium (MSNE) for the game defined by UA and UB. Using the Lemke-Howson algorithm (via the nashpy library), we enumerate all equilibria returned by the algorithm. When multiple equilibria exist, the expected payoffs of the agents and the regulator are defined as the average of their expected values across all such equilibria.

Regulatory Optimization Once equilibria are identified for all candidate market configurations, the regulator (Designer) utilizes the Designer matrices (DF or DE) to determine the optimal market structure.

For each market m, let (σA∗ ,σB∗ ) be the computed equilibrium profile. The expected value of the Designer’s objective VD(m) is calculated as:

VD(m) = (σA∗ )T · Dm · σB∗ (11)

where Dm is the Designer matrix (Fairness or Efficiency) for market m. The regulator then selects the market configuration m∗ that maximizes this expected value:

m∗ = arg max m∈M

VD(m) (12)

This step completes the meta-game loop, determining the final regulatory environment and the resulting agent payoffs.

### C The Poisoned Apple Effect

This section provides a detailed breakdown of the “Poisoned Apple” phenomenon illustrated in Figure 1 of the main text. The effect describes a strategic manipulation where a player introduces a new technology not to utilize it, but to force a fairness-maximizing regulator to alter the market design in a way that disproportionately benefits the releaser.

We examine a specific instance from the Bargaining game family. The initial strategy space consists of four Large Language Models: Model A (claude-3-7-sonnet), Model B (gemini-2.0-flash), Model C (llama-3.1 405b-instruct), and Model D (llama-3.3-70b-instruct). The regulator’s objective is to maximize Fairness (defined in Eq. 4 as the minimization of deviation from an equal split).

- Phase 1: The Status Quo (Pre-Release). The regulator evaluates the potential equilibria across all market configurations and selects the environment that maximizes fairness. Prior to the release, fairness in Market 1 (incomplete information, no communication, finite horizon) equals 0.975, and in Market 2 (incomplete information, no communication, infinite horizon) equals 0.965; Market 3 (incomplete information, communication allowed, finite horizon) achieves 0.989, while Market 4 (incomplete information, communication allowed, infinite horizon) attains the maximum fairness of 1.000; under complete information, Market 5 (communication allowed, finite horizon) yields 0.977 and Market 6 (communication allowed, infinite horizon) yields 0.962, whereas Market 7 (no communication, finite horizon) achieves 0.974 and Market 8 (no communication, infinite horizon) achieves 0.990.

The optimal design, designated as Market 4, is defined by Incomplete Information (CI = False), Allowed Communication (MA = True), and an Infinite Horizon (T = ∞).

In this environment (Market 4), the game stabilizes at a pure strategy equilibrium:

- • Alice selects Model D (llama-3.3-70b-instruct).
- • Bob selects Model A (claude-3-7-sonnet).
- • Outcome: This pairing results in a perfect Fairness score of 1.000. The expected payoffs are 0.49 for Alice and 0.50 for Bob.

- Phase 2: The Toxic Release (Intermediate State) Alice introduces a new technology, Model E (gemini-1.5-pro), into the ecosystem. The availability of this model drastically alters the strategic landscape. Following the release of Model E, fairness in Market 1 equals 0.975 and in Market 2 equals 0.965; Market 3 yields 0.988 and Market 4 declines to 0.976; under complete information, Market 5 yields 0.967 and Market 6 yields 0.962, while Market 7 declines to 0.959 and Market 8 becomes the fairness-maximizing market with a score of 0.990.

The fairness in the original Market 4 drops significantly (from 1.000 to 0.976), destabilizing the original equilibrium:

- • Alice switches to the new Model E (gemini-1.5-pro).
- • Bob continues to play Model A (claude-3-7-sonnet).
- • Impact: This shift causes the Fairness in Market 4 to deteriorate. In this intermediate state (if the market were not updated), Alice’s payoff would theoretically rise to 0.51, while Bob’s would drop to 0.40.

- Phase 3: Regulatory Flight and Payoff Reversal (Post-Release). To restore fairness, the regulator is compelled to abandon Market 4 and migrate to the new optimal alternative. The data indicates that Market 8defined by Complete Information (CI = True), Prohibited Communication (MA = False), and an Infinite Horizon (T = ∞)—is now the fairness-maximizing choice (0.990).

In this new market, the strategic landscape reorganizes completely:

- • Alice switches to Model A (claude-3-7-sonnet).
- • Bob switches to Model B (gemini-2.0-flash).
- • Non-Adoption: Crucially, in this new equilibrium, neither player selects the new Model E. The technology that precipitated the market shift is discarded.
- • Outcome: The regulator restores Fairness to 0.990. However, the forced regulatory shift drastically alters the payoff distribution. Alice’s expected payoff rises to 0.52 (+0.03 relative to the status quo), while Bob’s payoff falls to 0.46 (−0.04).

### D The Banning Regulator

This section provides the formal definition, computational method, and extended results for the banning regulator extension described in the main text.

- D.1 Formal Model In the baseline model, the regulator selects a market m∗ to maximize the designer’s objective:

m∗ = arg max m∈M

VD(m,S) (13)

where S is the full set of available technologies and VD(m,S) denotes the designer’s metric evaluated at the Nash equilibrium of the meta-game restricted to technology set S under market m.

The banning regulator extends this by jointly selecting both a market and a ban set B ⊆ S with |B| ≤ N: (m∗,B∗) = arg max

m∈M, B⊆S, |B|≤N

VD(m,S \ B) (14)

Here, VD(m,S \ B) is computed by restricting the payoff matrices to the rows and columns corresponding to technologies in S \B, computing the Nash equilibrium of the restricted game, and evaluating the designer’s metric. The regulator is not required to ban any technology; the option B = ∅ is always available.

Note that when N = |S| − 1, only one technology remains and the agents have no choice. When N = 0, the model reduces to the baseline.

- D.2 Efficient Computation via Dynamic Programming

Since the equilibria for all technology subsets are already cached from the baseline analysis, the banning regulator’s optimization can be computed without any new equilibrium calculations. We use the following recurrence:

best(S,0) = cache[S] (15) best(S,N) = max best(S,N−1), max

s∈S

best(S \ {s},N−1) (16)

where cache[S] stores the pre-computed optimal (market, equilibrium, payoffs) tuple for technology set S. This runs in seconds per game family, as it operates entirely on cached results.

- D.3 Why Banning Amplifies the Poisoned Apple Effect

The amplification arises because each state — before and after adding a technology — is independently optimized over the larger space of (market × ban set) configurations.

With N = 0, the regulator chooses among |M| markets (8 in our setting). With N = 1, the regulator chooses among |M| × (|S| + 1) configurations (each market with each possible single ban, or no ban). This expanded optimization means the “before” and “after” states can settle on very different configurations, increasing the gap between their outcomes.

Critically, a setting with |S| technologies and ban budget N is not equivalent to a setting with |S|−N technologies and no banning. The regulator chooses which N technologies to ban — this is an optimization, not a random removal. With (|S| = 7,N = 4), the regulator selects from 74 × |M| = 280 configurations, compared to just |M| = 8 configurations with (|S| = 3,N = 0), despite both leaving 3 technologies available.

The data reveal three phases as N increases:

- 1. Amplification (small N): The optimization space grows faster than the regulator’s ability to neutralize the added technology. The Poisoned Apple rate increases.
- 2. Peak (intermediate N): The divergence between “before” and “after” optima is maximal.
- 3. Suppression (large N): The regulator can directly ban the added technology (and others), effectively reverting to the pre-expansion state. The Poisoned Apple rate declines.

### References

- [1] Dietrich Braess. Uber¨ ein paradoxon aus der verkehrsplanung. Unternehmensforschung, 12:258–268, 1968.
- [2] Vincent P. Crawford and Joel Sobel. Strategic Information Transmission. Econometrica, 50(6):1431–1451,

1982. Publisher: [Wiley, Econometric Society].

Poisoned Apple Effect: Existing Models vs Ban Budget

Bargaining (Fairness)

Negotiation (Fairness)

Persuasion (Fairness)

[Figure 26]

[Figure 27]

[Figure 28]

- 0

- 1

- 2

- 3

- 4

- 5

- 6

- 7

- 8

- 9

- 10

- 11

- 12

- 0

- 1

- 2

- 3

- 4

- 5

- 6

- 7

- 8

- 9

- 10

- 11

- 12

- 0

- 1

- 2

- 3

- 4

- 5

- 6

- 7

- 8

- 9

- 10

- 11

- 12

0

0

0

0 0

0 0

0 0

100

0 0 0

0 0 0

0 29 0

|[Figure 29]| |
|---|---|
| | |
| | |
| | |
| | |

0 0 0 0

0 0 89 0

2 43 28 0

BanBudget(N)

0 0 0 0 0

0 0 86 0 0

6 49 53 52 0

0 0 0 0 0 0

0 67 80 33 66 78

13 53 61 57 35 0

0 0 1 16 18 0 0

0 56 75 49 61 76 100

21 56 61 58 46 31 67

80

2 4 24 46 56 77 100 0

0 50 67 49 57 68 73 0

29 57 59 54 47 39 54 25

6 22 49 60 60 67 77 0 0

0 48 59 47 54 62 69 0 0

34 56 53 48 42 37 42 23 33

10 39 57 60 58 56 62 72 67 0

0 42 51 44 48 53 54 46 22 0

36 50 43 38 34 30 30 24 26 100

15 45 52 51 50 50 52 53 53 45 33

0 30 39 37 35 37 34 29 23 22 50

30 36 28 25 23 20 19 17 13 14 0

25 27 28 29 30 31 31 28 25 22 25

21 25 18 16 15 13 10 8 6 3 0

12 8 8 6 5 4 4 3 1 0 0

60

2 3 4 5 6 7 8 9 10 11 12

2 3 4 5 6 7 8 9 10 11 12

2 3 4 5 6 7 8 9 10 11 12

Number of Existing Models

Number of Existing Models

Number of Existing Models

Percentage(%)

Bargaining (Efficiency)

Negotiation (Efficiency)

Persuasion (Efficiency)

[Figure 30]

[Figure 31]

[Figure 32]

- 0

- 1

- 2

- 3

- 4

- 5

- 6

- 7

- 8

- 9

- 10

- 11

- 12

- 0

- 1

- 2

- 3

- 4

- 5

- 6

- 7

- 8

- 9

- 10

- 11

- 12

- 0

- 1

- 2

- 3

- 4

- 5

- 6

- 7

- 8

- 9

- 10

- 11

- 12

40

0

0

0

0 0

0 0

0 0

0 0 0

0 24 82

0 0 0

3 0 0 0

0 33 82 0

10 0 0 0

BanBudget(N)

20

9 0 0 0 0

0 36 82 85 82

22 26 100 0 0

17 2 4 15 50 80

1 39 81 82 79 89

26 58 60 71 0 0

24 8 14 30 58 70 0

3 44 78 79 74 81 0

25 65 56 62 62 0 0

30 17 26 40 59 64 43 0

8 48 73 74 67 68 62 75

22 65 51 58 60 70 67 0

34 27 36 47 58 59 46 6 0

15 51 65 66 60 62 60 59 50

16 61 48 61 65 66 61 50 0

0

37 35 42 49 53 50 41 24 12 0

23 50 57 58 56 59 62 65 70 80

9 53 53 63 63 58 61 62 47 0

37 38 43 45 46 46 45 46 50 51 20

26 41 47 50 52 55 57 59 54 42 17

9 43 50 53 51 47 43 36 24 25 0

28 29 30 30 31 31 32 32 29 23 12

27 33 34 34 31 27 23 19 13 8 0

22 25 26 21 18 16 12 7 10 18 50

2 3 4 5 6 7 8 9 10 11 12

2 3 4 5 6 7 8 9 10 11 12

2 3 4 5 6 7 8 9 10 11 12

Number of Existing Models

Number of Existing Models

Number of Existing Models

Figure 4: Poisoned Apple rate as a function of initial technology set size (|S|, horizontal axis) and ban budget (N, vertical axis), for each game family under a fairness-maximizing regulator. Cells where N ≥ |S| are left blank (the regulator cannot ban all available technologies). The triangular structure reflects the constraint that banning is only meaningful when N < |S|. Darker colors indicate higher Poisoned Apple rates.

- [3] Gillian K. Hadfield and Andrew Koh. An Economy of AI Agents, September 2025. arXiv:2509.01063 [econ].
- [4] John J. Horton. Large Language Models as Simulated Economic Agents: What Can We Learn from Homo Silicus?, January 2023. arXiv:2301.07543 [econ].
- [5] Nigel Howard. Paradoxes of rationality: theory of metagames and political behavior. Cambridge: MIT Press, 1971.
- [6] Emir Kamenica and Matthew Gentzkow. Bayesian persuasion. American Economic Review, 101(6):2590–2615, 2011.
- [7] Eric Maskin. Nash Equilibrium and Welfare Optimality. The Review of Economic Studies, 66(1):23–38, 1999. Publisher: [Oxford University Press, Review of Economic Studies, Ltd.].
- [8] Roger B Myerson and Mark A Satterthwaite. Efficient mechanisms for bilateral trading. Journal of Economic Theory, 29(2):265–281, April 1983.
- [9] John Nash. Non-Cooperative Games. Annals of Mathematics, 54(2):286–295, 1951. Publisher: [Annals of Mathematics, Trustees of Princeton University on Behalf of the Annals of Mathematics, Mathematics Department, Princeton University].
- [10] Martin J. Osborne and Ariel Rubinstein. A Course in Game Theory. MIT Press, 1994.
- [11] Ariel Rubinstein. Perfect Equilibrium in a Bargaining Model. Econometrica, 50(1):97–109, 1982. Publisher: [Wiley, Econometric Society].
- [12] Larry Samuelson. Strategy additions and equilibrium selection. International Journal of Game Theory, 19:269– 276, 1990.
- [13] Eilam Shapira, Omer Madmon, Roi Reichart, and Moshe Tennenholtz. Can Large Language Models Replace Economic Choice Prediction Labs?, February 2024. arXiv:2401.17435 [cs].
- [14] Eilam Shapira, Omer Madmon, Itamar Reinman, Samuel Joseph Amouyal, Roi Reichart, and Moshe Tennenholtz. GLEE: A Unified Framework and Benchmark for Language-based Economic Environments, October

2024. arXiv:2410.05254 [cs].

- [15] Eilam Shapira, Moshe Tennenholtz, and Roi Reichart. Data from ”glee: A unified framework and benchmark for language-based economic environments”’. https://github.com/eilamshapira/GLEE/tree/master/Data/ llm_vs_llm, 2025. GitHub Repository.
- [16] Michael Wellman. Understanding the Implications of Advanced AI on Financial Markets. Journal of Financial Transformation, 60:14–19, 2025.

