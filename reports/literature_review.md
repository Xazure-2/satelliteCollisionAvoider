# Literature review: ML for conjunction risk and what it means for our model

Scope: what others did with the ESA Collision Avoidance Challenge data (the same CDM fields we use), and general evidence on ensembles, hyperparameter search, stacking and missing data for tabular models. Each section ends with what we took from it. Items marked *unverified* could not be read in full here; check them before citing in the pitch.

## 1. The ESA Kelvins Collision Avoidance Challenge (2019)

**Uriot, Izzo, et al. — "Spacecraft Collision Avoidance Challenge: design and results of a machine learning competition", *Astrodynamics* 6, 121–140 (2022). [arXiv:2008.03069](https://arxiv.org/abs/2008.03069), [doi:10.1007/s42064-021-0101-5](https://link.springer.com/doi/10.1007/s42064-021-0101-5).** Our primary reference.

- **Task:** predict each event's final risk from CDMs cut off 2 days before TCA, the same horizon as our organizer's task. Score L = MSE on high-risk events (risk >= −6) / F2 at −6; predictions clipped at −6.001.
- **The naive "latest risk" forecast was very hard to beat** (L = 0.694). About 12 of 97 teams beat it.
- **1st place (sesc, L = 0.556):** off-the-shelf pipelines (time-series feature extraction, AutoML, random forests) overfit, so the team built a **hand-made cascade of thresholds on the latest risk**. It promoted borderline low-risk events to high risk to raise recall (F2 rewards it) and used simple rules on object type, size and large miss distance. No ensembles or boosting.
- **3rd place (Magpies, L = 0.585):** Manhattan (siamese) LSTMs on 7 latest-CDM features (including `miss_distance`, `time_to_tca`, Mahalanobis distance) plus CDM count and the mean/std of past risk. 3-fold CV, random hyperparameter sampling from a grid, oversampling of anomalous events. **The only ensemble among top teams: a majority vote of networks with validation AUC >= 0.8.**
- **Post-competition analysis:** a LightGBM regressor predicting the *change* from the latest risk gained only about 1.3% mean F2 over the naive baseline in cross-validation. The official split was not stratified by high-risk events and was an outlier among 10,000 simulated stratified splits, so leaderboard gaps partly reflect split luck.
- About half the teams tried to build validation sets resembling the test set, with poor results.

**What we took:** persistence is the baseline to beat, and gains over it are small. Our persistence floor with tuned thresholds is the same idea as the winning cascade, promoting borderline events for recall. Our distance-change regressor mirrors their LightGBM "change from latest" analysis. We stratify folds by outcome and report paired bootstrap intervals, not single-split gaps.

## 2. Other work on the same data

- **Catulo, Soares, Guimarães — "Predicting the Probability of Collision of a Satellite with Space Debris: A Bayesian Machine Learning Approach" (2023), [arXiv:2311.10633](https://arxiv.org/abs/2311.10633).** Hidden Markov models on the risk sequence alone, motivated by naive forecasts being hard to beat (a Markov-like signal). The HMM beats the naive forecast on some metrics, not all. It also notes that gradient-boosted trees and Manhattan LSTMs did well in the competition. *Takeaway:* sequence models add little over the latest value; strong recent-state baselines are the norm.
- **Acciarini, Pinto, Metz, et al. — "Kessler: a machine learning library for spacecraft collision avoidance", 8th European Conference on Space Debris (2021), [SDC8 paper 226](https://conference.sdo.esoc.esa.int/proceedings/sdc8/paper/226/SDC8-paper226.pdf).** Stacked LSTM CDM forecaster plus Bayesian deep learning for generating CDMs. *Takeaway:* deep sequence models need far more data than ~10k events to beat trees; out of scope for a hackathon.
- **Pinto et al. — "Towards Automated Satellite Conjunction Management with Bayesian Deep Learning" (2020), [arXiv:2012.12450](https://arxiv.org/abs/2012.12450).** Bayesian RNNs forecasting future CDM contents, including miss distance. *Takeaway:* forecasting the evolution of miss distance, as our change regressor does, is an established framing.
- **Metz, Simon, Letizia (ESA ESOC).** Tree ensembles and RNNs predicting positional covariances from CDMs, then deriving risk (cited in Catulo et al.; *unverified* here).
- **Two-step classification then regression, with SMOTE balancing** (IAC 2020, [IAC-20,A6,VP,13](https://p.iafastro.directory/iac/paper/id/57288/abstract-pdf/IAC-20,A6,VP,13,x57288.brief.pdf), 7th of 97). *Takeaway:* class balancing helps recall; we use balanced class weights and tuned thresholds instead of synthetic oversampling.
- **Benchmark of ML models on the dataset**, highlighting 0.84% high-risk CDMs ([SDC8 paper 123](https://conference.sdo.esoc.esa.int/proceedings/sdc8/paper/123/SDC8-paper123.pdf); *unverified*, PDF not readable here). Reports gradient-boosted trees beating the naive baseline only marginally.
- **Sánchez, Rodríguez-Fernández, Vasile — "Robust classification with belief functions and deep learning applied to STM", IEEE CEC 2024, [Strathprints 90331](https://strathprints.strath.ac.uk/90331).** Dempster–Shafer treatment of epistemic uncertainty across CDM sequences, with learned classifiers (reported to include random forest, LightGBM and transformers; *unverified*).

## 3. Ensembles, stacking and hyperparameter search (general tabular evidence)

- **Stacked generalization (Wolpert 1992, *Neural Networks* 5(2)).** A meta-learner combines base-model predictions. Standard practice: train it only on **out-of-fold** base predictions, keep it simple (logistic/ridge), and prefer **diverse** base models (different families or targets). On small data, plain averaging often matches stacking.
- **Grinsztajn, Oyallon, Varoquaux — "Why do tree-based models still outperform deep learning on tabular data?", NeurIPS 2022 Datasets & Benchmarks, [arXiv:2207.08815](https://arxiv.org/abs/2207.08815).** Tree ensembles are strong defaults on medium-sized tabular data; hyperparameter tuning helps, but less than model family and data handling.
- **Nested cross-validation (Cawley & Talbot, "On over-fitting in model selection…", JMLR 11, 2010).** Selecting hyperparameters on the same folds you report inflates scores; nested CV gives an unbiased estimate. We use nested random search (Bergstra & Bengio, JMLR 13, 2012: random search is efficient when few hyperparameters matter).

**What we took:** E1/E2 average diverse models (a distance-change regressor and tier classifiers); S1 stacks them with a logistic meta-learner on out-of-fold scores; T1/T2 use nested random search. See [final_tier_ensembles.md](final_tier_ensembles.md).

## 4. Missing data and imputation

- **Josse, Prost, Scornet, Varoquaux — "On the consistency of supervised learning with missing values" (2019), [arXiv:1902.06931](https://arxiv.org/abs/1902.06931).** For flexible learners, constant/median imputation plus missingness indicators is consistent, and tree models handling missing values natively perform comparably.

**What we took:** median imputation with indicators made no meaningful difference over native NaN handling. The useful fixes were domain-specific: setting the 10-Earth-radii placeholder sigmas to missing, and filling daily space weather (F10/AP) from earlier messages of the same event.

## 5. Physical Pc and standards context (for the report rewrite)

- **Foster & Estes (1992), NASA JSC-25898**, and **Alfano, "Relating position uncertainty to maximum conjunction probability", *J. Astronautical Sciences* 53 (2005)**. The 2D Pc and maximum Pc over covariance scaling explain `max_risk_estimate` as an upper bound on Pc, and why huge uncertainties *dilute* Pc.
- **NASA Spacecraft Conjunction Assessment and Collision Avoidance Best Practices Handbook (NASA/SP-20205011318, 2020).** Context for the 1e-4 maneuver threshold, Pc dilution and covariance realism.
- **CCSDS 508.0-B-1, Conjunction Data Message.** Defines the RTN frame and TCA semantics behind our columns.
- **ESA standards:** the brief names "ESA Space Debris Office standards"; candidates are ESSB-ST-U-007 and ISO 24113. Confirm editions with the organizers before citing.

## 6. Implications for this project

1. **Persistence is the baseline.** Every model is reported against it with paired intervals. The literature leads us to expect a small, real gain, not a large one.
2. **Recall-promoting decision rules are what won.** Our persistence floor plus F2-tuned thresholds is the principled version of the winning cascade.
3. **Ensembles and tuning are worth trying, but expect at most marginal gains,** given about 500 Critical forecast events and strong correlation among tree models. Diversity (regressor vs classifier targets) is the main lever.
4. **Do not over-read small CV differences.** Use nested search, a fixed protocol and one holdout look.
