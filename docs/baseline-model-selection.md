# Phase 3 baseline encoder selection

Three similarly sized, established 16 kHz self-supervised speech encoders were
considered before implementation.

| Encoder | Relevant properties | Baseline tradeoff |
|---|---|---|
| `facebook/wav2vec2-base` | 16 kHz; contrastive pretraining; mature Transformers support | Simplest and widely used, but pretraining emphasizes speech content |
| `facebook/hubert-base-ls960` | 16 kHz; masked prediction of clustered speech units | Strong acoustic representation, but trained only on LibriSpeech 960 h |
| `microsoft/wavlm-base-plus` | 16 kHz; HuBERT-derived; 94,000 h from Libri-Light, GigaSpeech, and VoxPopuli; content and speaker modeling | Broader pretraining and speaker-sensitive design are useful for an artifact detector and later shift analysis |

Phase 3 selects **WavLM Base Plus** at immutable revision
`4c66d4806a428f2e922ccfa1a962776e232d487b`. The choice is based on task fit and
pretraining scope, not on ASVspoof DEV or EVAL results. No candidate was benchmarked
for model selection.

The encoder is frozen. Its last hidden states are masked-mean pooled, then passed to
a trainable LayerNorm, dropout, and one-output linear head. This limits the first
baseline to a small, auditable optimization problem while retaining a standard
pretrained speech representation.
