# **Project Overview**

## **1. Overall Goal**

This project studies **direct Vietnamese-to-English speech translation for online news and commentary audio**:

$
f:\text{Vietnamese audio waveform}\rightarrow\text{English text}
$

The system receives a Vietnamese audio clip and directly generates its English translation, without producing an intermediate Vietnamese transcript.

The starting point is a fixed **Whisper-medium zero-shot baseline**. The goal is to improve its translation quality on real-world web audio through evidence-driven changes—including domain-matched training data, fine-tuning, decoding, preprocessing, and data augmentation—without increasing the inference model size.

The project is guided by three principles:

1. **Target-domain relevance:** training and validation data should represent the acoustic and linguistic conditions of online Vietnamese news and commentary.
2. **Controlled experimentation:** each intervention should address an observed failure mode and be evaluated while other important variables remain fixed.
3. **Quality–compute trade-offs:** improvements should be reported together with their data requirements, training cost, GPU memory usage, and inference latency.

The objective is therefore not merely to maximize a leaderboard score. It is to determine which interventions produce reproducible improvements on unseen, source-disjoint web audio and why they work.

## 2. Direct Speech Translation

A conventional speech-translation system uses two stages:

$
\text{Vietnamese audio}
\xrightarrow{\text{ASR}}
\text{Vietnamese transcript}
\xrightarrow{\text{MT}}
\text{English translation}
$

This cascaded design is modular and easier to debug, but the translation model can only operate on the transcript produced by the ASR system. Recognition errors—particularly in names, numbers, accents, code-switching, or noisy speech—may therefore propagate into the final translation.

This project instead investigates direct translation:

$
\text{Vietnamese audio}
\xrightarrow{\text{single model}}
\text{English translation}
$

A direct model is trained or adapted using paired Vietnamese audio and English text. It can optimize the final translation objective without committing to an explicit Vietnamese transcript between the two stages.

Direct translation is not assumed to be universally superior to a cascade. Cascaded systems may benefit from larger monolingual ASR and machine-translation datasets and provide more interpretable intermediate outputs. Direct translation is chosen here because:

* Whisper already provides a strong multilingual translation model.
* The project can study end-to-end adaptation to noisy Vietnamese web audio.
* Model size, memory, latency, and translation quality can be evaluated within one fixed architecture.

## 3. Dataset

The target dataset contains **2,481 Vietnamese audio clips**, totaling approximately **9.1 hours**.

Each clip is:

* 7–15 seconds long;
* mono, 16 kHz, and stored as `.wav`;
* sampled from public Vietnamese YouTube videos;
* drawn primarily from current-affairs and cultural commentary.

Unlike clean read-speech corpora, the clips contain realistic web-audio conditions, including:

* background music and environmental noise;
* room reverberation and online-video compression;
* varied Vietnamese accents;
* named entities and current-affairs vocabulary;
* occasional Vietnamese–English code-switching.

The dataset contains only evaluation audio. English references are hidden, so external public corpora must be used for training and development.

Additional data should be selected according to its similarity to the target domain. When possible, train, validation, and test partitions should be separated by source video or speaker to prevent closely related clips from appearing across splits.

## 4. Evaluation with BLEU

Translation quality is evaluated using **corpus-level BLEU-4** between the generated English translations and the reference translations.

BLEU combines clipped (n)-gram precision for $n=1$,$\ldots$,$4$ with a brevity penalty:

$
\operatorname{BP}
\cdot
\exp\left(
\frac{1}{4}
\sum_{n=1}^{4}\log p_n
\right),
$

where $p_n$ is the clipped precision of generated $n$-grams and

$$
\operatorname{BP} =
\begin{cases}
1, & \text{if } c > r, \\
\exp\left(1-\frac{r}{c}\right), & \text{if } c \le r,
\end{cases}
$$

Here, $c$ is the total generated length and $r$ is the effective reference length. The brevity penalty prevents a system from obtaining a high precision score by producing translations that are systematically too short. If the generated translations are at least as long as the references, then $\operatorname{BP}=1$, so no penalty is applied. If they are shorter, then $\operatorname{BP} < 1$, reducing the final BLEU score.

The implementation uses:

* corpus-level aggregation;
* BLEU-4;
* lowercase normalization;
* Moses `13a` tokenization;
* no smoothing.

The score is computed with SacreBLEU:

```python
import sacrebleu

bleu = sacrebleu.corpus_bleu(
    hypotheses,
    [references],
    lowercase=True,
    tokenize="13a",
)

print(bleu.score)
```

This is equivalent to the official evaluation:

```python
sacrebleu.corpus_bleu(
    hyps,
    [refs],
    lowercase=True,
    tokenize="13a",
)
```

Because the target references are hidden, local BLEU is calculated on held-out external development data. The target dataset is reserved for final evaluation through Kaggle and should not be used to select checkpoints or tune hyperparameters.

BLEU is the primary benchmark metric, but it is not a complete measure of translation quality. Experiments should also examine failure categories such as omissions, hallucinations, named-entity errors, mistranslated numbers, repetitions, and degradation under noise.
