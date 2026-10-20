# Liverpool vs Aston Villa: full-half Whisper direct-translation check

This run uses Whisper `small` with its direct English `translate` task on Russian commentary. It processes the first-half video and ranks English output with the reviewed event-context model.

## Opening goal

- SoccerNet `Labels-v2.json` marks a first-half goal at **00:01:03**.
- The generated goal candidate is **00:01:19–00:01:23**, with the text “as it begins with a goal from Liverpool.” This preserves a useful goal clue, but its segment timestamp is about **16 seconds after** the annotated goal. At 00:01:03 the video shows the Liverpool attack in progress; the actual scoring sequence follows shortly after.
- Because of the timing gap, this output should be presented as finding goal-related commentary, not as frame-accurate localization. A clip would need a wider review window or a correction using video/event timestamps.

## False goal phrase

- A second candidate at **00:31:18–00:31:19** says “a goal by Salah Genton.” There is no first-half Goal annotation near that time.
- A frame sampled from the video at 00:31:18 shows ordinary play near the touchline and the scoreboard still at 0–0, so this candidate is a false positive. The player name is also mistranscribed.

## Result

Direct Whisper translation performs better than the Russian ASR → NLLB route for surfacing the opening goal wording in this half, but it still shifts the reported commentary time and invents/garbles a second goal phrase. Keep the English direct-translation route as a candidate option, with human review and a timestamp tolerance; do not describe the result as reliable goal detection across languages.

An English negation filter was added after this review. Reranking the saved transcript in `../liverpool_direct_translate_negation_refined/` removes two shot candidates whose own wording says the shot did not happen (“didn't get a shot” / “cannot shoot”), reducing the list from nine candidates to seven. It does not remove the false 00:31:18 goal phrase.
