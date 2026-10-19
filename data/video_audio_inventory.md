# Local SoccerNet video and audio inventory

Checked the MKV files currently visible under `path/to/SoccerNet`. Audio presence was confirmed by extracting a 10-second mono WAV and checking its size, not only by reading the container's stream metadata.

| Match/video | Audio extraction | Current project use |
|---|---|---|
| Crystal Palace 0–1 Manchester City, `1_224p.mkv` | Usable audio | Full Whisper and English-translation output is in `outputs/autolang_full/`. Whisper detected Spanish. |
| Leicester 2–5 Arsenal, `1_224p.mkv` | Usable audio | Full French transcript is in `outputs/leicester_arsenal_test/`; improved reranked candidates and goal audit are in `outputs/leicester_arsenal_refined/`. |
| Liverpool 3–2 Aston Villa, `1_224p.mkv` | Usable audio | Russian ASR + NLLB full run is in `outputs/liverpool_aston_villa_test/`; it misses the annotated opening goal. Full-half Whisper direct English translation is in `outputs/liverpool_direct_translate_full/`; it finds goal wording at 00:01:19 (+16 seconds from the 00:01:03 label) and also emits a false goal phrase at 00:31:18. |
| Manchester City 6–1 Newcastle Utd, `1_720p.mkv` | **No usable samples**; extracted WAV was only 78 bytes | Use the timestamped dataset captions in `outputs/city_newcastle/`; seven goal-caption times across both halves were visually checked in `outputs/city_newcastle/video_audit.md`. |
| Manchester City 6–1 Newcastle Utd, `2_720p.mkv` | **No usable samples**; extracted WAV was only 78 bytes | Use the timestamped dataset captions in `outputs/city_newcastle/`; seven goal-caption times across both halves were visually checked in `outputs/city_newcastle/video_audit.md`. |

There are five local MKV files visible in this workspace right now. Three have usable commentary audio. The 720p City–Newcastle files report an audio stream in their metadata, but decoding yielded no audio samples; treat them as silent for the speech pipeline.

## Interpretation limits

- `small` Whisper plus NLLB-200 was used for the Spanish and Russian full-match transcripts. The Russian example contains serious ASR and translation errors, including a missed first-half goal in candidate ranking.
- The classifier only reranks moments found by the phrase-matching candidate stage. It cannot recover a missed goal.
- Existing training uses a small mix of SoccerNet event annotations and one match's manual review. It is a pilot model, not reliable across arbitrary matches, commentators, or languages.
- SoccerNet's event annotations provide event timestamps such as shots and corners. They do not say whether a moment is an exciting highlight.
- The City–Newcastle caption output was generated from timestamped dataset captions and their `important` flags; do not describe it as speech recognition.
