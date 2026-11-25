# Demo test clips

Six clips (one per emotion) from **EmoDB** (Berlin Database of Emotional Speech, Burkhardt et al. 2005,
http://emodb.bilderbar.info), a dataset **not used in training**. All six: speaker 14 (female), sentence `a07`,
German, so only the tone changes. Original files: `14a07Na` (neutral), `14a07Fd` (happy), `14a07Tc` (sad),
`14a07Wc` (angry), `14a07Aa` (fearful), `14a07Eb` (disgust).

Predictions of the app's default model `wav2vec2_6class` (86.4 % on RAVDESS test actors):

| File | Predicted | Confidence |
|---|---|---|
| happy.wav | happy ✅ | 99.7 % |
| disgust.wav | disgust ✅ | 99.7 % |
| fearful.wav | fearful ✅ | 99.5 % |
| sad.wav | sad ✅ | 96.1 % |
| neutral.wav | fearful ❌ | 68.0 % |
| angry.wav | happy ❌ | 97.7 % |

4/6 on a new speaker and language. Angry → happy: both loud, high-energy; training data is only
North-American English. Six clips is a demo, not an accuracy measurement.
