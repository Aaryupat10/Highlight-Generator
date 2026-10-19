# Manchester City vs Newcastle: visual check of caption-based goal moments

The two 720p MKVs have a Vorbis audio stream in their metadata, but extracting ten seconds from either produces an empty 78-byte WAV. They cannot be used as Whisper speech input.

I sampled frames around every goal caption listed in `ranked_highlights.md`. The scoreboard and goal/celebration sequence agree with the annotated events:

| Half file and video time | Caption event | Visual check |
|---|---|---|
| `1_720p.mkv`, 00:17:26 | Newcastle goal, 0–1 | Newcastle celebration and 0–1 scoreboard visible. |
| `1_720p.mkv`, 00:41:55 | Manchester City goal, 1–1 | Goal sequence and 1–1 scoreboard visible. |
| `2_720p.mkv`, 00:03:37 | Manchester City goal, 2–1 | Goal sequence/score change visible. |
| `2_720p.mkv`, 00:04:52 | Manchester City goal, 3–1 | Goal sequence/score change visible. |
| `2_720p.mkv`, 00:07:58 | Manchester City goal, 4–1 | Goal sequence/score change visible. |
| `2_720p.mkv`, 00:14:30 | Manchester City goal, 5–1 | Goal/celebration and 5–1 scoreboard visible. |
| `2_720p.mkv`, 00:16:53 | Manchester City goal, 6–1 | Goal/celebration and 6–1 scoreboard visible. |

This validates the timestamps for these seven **caption-derived** goal entries. It does not validate ASR, audio commentary, or the remaining chance/card/substitution captions, and it is not a model evaluation: the descriptions and event flags were supplied by the dataset.
