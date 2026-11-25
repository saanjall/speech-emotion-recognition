| Rank | Model | Test acc | Macro-F1 | Best val acc | Epochs | Params |
|---|---|---|---|---|---|---|
| 1 | Ensemble (equal avg) | 0.8000 | 0.7922 | 0.7917 | - | 0 |
| 2 | wav2vec2-base + CREMA-D/TESS | 0.7917 | 0.7843 | 0.7417 | 10 | 94,570,632 |
| 3 | wav2vec2-base + CREMA-D/TESS (seed 7) | 0.7708 | 0.7666 | 0.7167 | 9 | 94,570,632 |
| 4 | WavLM-large + 1/2 CREMA-D | 0.7667 | 0.7569 | 0.8167 | 10 | 315,717,576 |
| 5 | wav2vec2-base | 0.7458 | 0.7393 | 0.7833 | 18 | 94,570,645 |
| 6 | Ensemble (weighted avg) | 0.7417 | 0.7272 | 0.8500 | - | 0 |
| 7 | WavLM-large + CREMA-D/TESS | 0.7375 | 0.7187 | 0.8417 | 7 | 315,717,576 |
| 8 | WavLM-large | 0.7250 | 0.7066 | 0.7583 | 13 | 315,717,576 |
| 9 | CNN+BiLSTM+Attn | 0.5958 | 0.5715 | 0.6333 | 36 | 1,307,048 |
| 10 | MLP (handcrafted) | 0.5000 | 0.4888 | 0.5250 | 34 | 283,016 |
| 11 | CNN (log-mel) | 0.4542 | 0.4450 | 0.6083 | 28 | 1,242,408 |
