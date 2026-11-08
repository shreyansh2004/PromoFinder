# Promo Finder (Kuku AI Generalist assignment, Problem 1)

A tool that reads a microdrama's episode screenplays and finds the best 20–60 second promo clips. Each clip opens on a hook in the first 3 seconds and ends on a cliffhanger. Promos are ranked using signals tested on real short-drama ads.

| Folder | What's inside |
|---|---|
| [`promo-finder/`](promo-finder/) | The tool: web app, command line, pipeline, trained model and its evidence. **Start here** ([README](promo-finder/README.md)) |
| [`dataset/`](dataset/) | How the training data was built: 596 real ads from TikTok's public Ad Library, Whisper transcripts, measured hook signals, ChatGPT tags and their quality checks ([README](dataset/README.md)) |
| [`submission/raw_outputs/`](submission/raw_outputs/) | Model evidence: training report, signal validation, correlation heatmap |
| `render.yaml` | One-click free hosting on Render |

**Confidentiality:** Kuku's input pack (the scripts) and any output quoting it are deliberately not in this repo; see `.gitignore`. Third-party ad videos and frames are excluded too.
