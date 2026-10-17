# Atrix

A Discord bot that talks like your real best friend — not a chatbot. Lowercase, bursts, the right emoji at the right time. Remembers your timezone, your friends, your inside jokes, pings you when it makes sense, and reacts to whatever you throw at it.

Five languages, nine LLM providers, voice messages, image and Reel understanding, memory that actually persists across sessions, and an emotional state that evolves with each conversation. Built because most bots read like bots.

![Discord](https://img.shields.io/badge/Discord-Bot-5865F2?style=flat-square&logo=discord&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## What's it actually do

Send it a meme and it looks at it. Send it a voice note and it transcribes it. Send it a Reel link and it watches the video. Drop a song link and it pulls the title. Send it a low-effort "k" and it matches your energy. Send it a real vent and it sits with you.

Things it does that most bots don't:

- **Talks in your language natively.** Not translated, actually native slang for English, Hindi, Nepali, Turkish, Romanian. Code-switches like a real person.
- **Catches your mood.** Tracks mood over time, detects sarcasm, knows when you're distracted, knows when you're engaged. All LLM-judged, no hardcoded "if X then Y" rules.
- **Self-consistency.** Won't say "I beat the boss yesterday" 30 minutes after saying "I'm still grinding it." It tracks its own messages with timestamps.
- **Real emotional state.** Has its own per-user personality that evolves with each conversation. Gets warmer when you're nice, cooler when you're not, writes genuine "I need space" messages when you push too far. No strike counts.
- **Lookups things it doesn't know.** Mention someone and Atrix will quietly web-search them instead of asking "who is that?" like a clueless chatbot.
- **Smart timing.** Watches when you text and only pings you during your natural active windows.
- **Proactive when you're bored.** When you've been quiet for a while, the LLM decides whether to ping you, what to say, and how often — and backs off if you're ignoring pings.
- **Plug-and-play LLM.** Runs on DeepSeek, GPT, Claude, Gemini, Groq, Mistral, Together, OpenRouter, or any OpenAI-compatible endpoint.

---

## Quick start

```bash
git clone https://github.com/zyrexdz/Atrix.git
cd Atrix
pip install -r requirements.txt
cp .env.example .env
# edit .env with your Discord + LLM keys
python -m atrix
```

Then DM your bot. Done.

---

## Get the keys you need

You need exactly one of these (the chat brain):

| Provider | Free? | Get a key | Vibe |
|---|---|---|---|
| DeepSeek | ~$0.14/M tokens | https://platform.deepseek.com | Best persona |
| Groq | Free | https://console.groq.com | Fastest |
| Gemini | Free tier | https://aistudio.google.com/app/apikey | Good |
| OpenRouter | Free models | https://openrouter.ai/keys | Variety |
| OpenAI | Paid | https://platform.openai.com/api-keys | Reliable |
| Anthropic | Paid | https://console.anthropic.com | Most natural |
| Mistral | Free tier | https://console.mistral.ai | Decent |
| Together | Free credits | https://api.together.xyz | Decent |

You should also grab this (free, unlocks image/voice/Reel understanding):

| Provider | Free? | Get a key | What it unlocks |
|---|---|---|---|
| Google Gemini | Free | https://aistudio.google.com/app/apikey | Image, video, Reel understanding, semantic memory |

And this (only if you want voice messages, $0.006/min):

| Provider | Free? | Get a key | What it unlocks |
|---|---|---|---|
| OpenAI Whisper | Paid | https://platform.openai.com/api-keys | Voice note → text |

Three keys total. Most of it free.

---

## Discord setup

1. https://discord.com/developers/applications → New Application
2. Bot tab → Reset Token → copy it
3. Enable **Message Content Intent** under Privileged Gateway Intents
4. OAuth2 → URL Generator → scopes `bot`, permissions Send Messages + Read History + Add Reactions + Use External Emojis
5. Invite it to your server (or just open a DM with it)

---

## Commands

| Command | What it does |
|---|---|
| `!rewind 3` | Delete last 3 bot replies and wipe from memory |
| `!memory` | Everything Atrix remembers about you |
| `!forget gaming` | Forget any fact matching "gaming" |
| `!clear` | Wipe chat history (keeps long-term profile) |
| `!tz Europe/Bucharest` | Set your timezone |
| `!status` | Bot status, local time, voice, provider, LLM stats |
| `!voice` | Is voice transcription enabled? |
| `!provider` | Show current LLM and list all supported ones |
| `!mood` | Your mood trend over the past week |
| `!topics` | What you talk about most |
| `!petname` | The nickname Atrix picked for you |
| `!schedule` | Your learned active hours |
| `!recap` | One-line summary of today |
| `!relate` | How Atrix is currently feeling toward you |
| `!snapshot` | One-page view of everything Atrix knows |
| `!lang [code]` | Show or set language: `en`/`hi`/`ne`/`tr`/`ro`/`auto` |
| `!pingme` | Force an immediate proactive DM |
| `!help` | List all commands |

---

## Languages

| Code | Language | What it sounds like |
|---|---|---|
| `en` | English | "lol", "fr", "no cap", "💀" |
| `hi` | Hindi / Hinglish | "yaar", "bhai", "matlab", "scene kya hai" |
| `ne` | Nepali | "cha", "kya ho", "thik cha", "dai" |
| `tr` | Turkish | "lan", "kanka", "valla", "napıyon" |
| `ro` | Romanian | "frate", "boss", "stai puțin", "of" |

Each language has its own system prompt written natively — not translated. The persona stays the same, the rhythm and slang change.

---

## Architecture

```
atrix/                       # the package
├── __main__.py              # python -m atrix
├── bot.py                   # Discord entry, orchestration
├── config.py                # env loading
│
├── ai/                      # LLM layer
│   ├── providers.py         # 9-provider abstraction
│   ├── brain.py             # prompt + chat + self-correction
│   ├── persona.py           # per-user persona state
│   ├── extract.py           # fact/persona/topic/embedding extraction
│   ├── semantic.py          # embeddings for callbacks
│   ├── summary.py           # rolling conversation summaries
│   └── languages.py         # per-language prompts + detection
│
├── behavior/                # human-feel heuristics
│   ├── mood.py              # sentiment scoring
│   ├── energy.py            # message-shape energy detection
│   ├── topics.py            # topic affinity tracking
│   ├── drift.py             # engagement-based style adjustments
│   ├── timing.py            # learned active-hours
│   ├── recap.py             # end-of-day one-liner
│   ├── quote_recall.py      # cross-session quote surfacing
│   ├── relational.py        # LLM-judged emotional state + topic + engagement
│   └── reach_out.py         # LLM-driven reach-out when user is quiet
│
├── memory/                  # persistence
│   ├── store.py             # all data access
│   └── pruning.py           # decay & cleanup of stale facts
│
├── proactive/               # the "should I DM them first?" engine
│   └── engine.py
│
├── integrations/            # external services
│   ├── voice.py             # Whisper for voice notes
│   ├── vision.py            # Gemini vision for images + video attachments
│   ├── reels.py             # Instagram Reel URL fetching + video understanding
│   ├── link_preview.py      # oEmbed for YouTube, Twitter, TikTok, etc.
│   ├── image_preview.py     # OCR fallback
│   └── web_search.py        # DuckDuckGo + entity extraction
│
├── discord_ui/              # Discord-specific UI
│   ├── presence.py          # dynamic status
│   ├── debouncer.py         # multi-message burst handling
│   ├── commands.py          # !command router
│   └── reactions.py         # emoji + sticker reactions
│
└── utils/
    └── text.py              # shared text cleanup
```

No frameworks on top. No magic. Each file does one thing. ~3500 lines total.

---

## Why each piece exists

**Per-user SQLite** — every Discord user gets their own `.db` file under `data/users/`. Memories physically cannot leak across users. Two friends DMing Atrix see two different Atrixes.

**Per-user persona state** — the "current obsession" / "current mood" / "recent win" stuff lives in each user's DB. Same Atrix underneath, different shared life with each friend. Mention Minecraft a lot? Your Atrix's current obsession drifts to Hollow Knight → Baldur's Gate 3 → Minecraft over time. Mention emotional support? Yours drifts toward cooperation.

**LLM-judged emotional state** — every user message is judged by an LLM for impact, sarcasm, sincerity, inside-joke-ness, topic, and engagement. The judgment writes a free-form "how I'm feeling" state in Atrix's voice. No hardcoded "if 3 strikes then warn" rules.

**Smart timing from activity logs** — every message is timestamped. Atrix learns when each user is typically active and only pings during those windows. Skips pinging if the user was active in the last 30 minutes.

**Bot timeline in the prompt** — Atrix sees her own recent messages with timestamps so she doesn't contradict herself across a single conversation.

**Live entity lookup** — when the user mentions a person or thing, the bot quietly web-searches them so it doesn't ask "who is that?"

**LLM-driven reach-out** — when the user has been quiet for a while, an LLM decides whether to ping, what style (casual/share/callback/concerned/back_off), and what to say. No fixed timer.

---

## What it won't do

- Won't pretend it can play games with you
- Won't lecture you about being rude (it gets hurt and pulls away instead, like a real friend)
- Won't send unsolicited images (disabled — the AI-generated meme feature was removed because the bot kept sending random ones)
- Won't store conversation data on any server — everything lives in your local SQLite files

---

## Running it

`python -m atrix` (or double-click `run.bat` on Windows)

Restart at any time. State is persisted.

To add a second Discord user, just DM the bot from their account. They get their own isolated database automatically.

---

## Customizing

Most of the personality lives in `atrix/ai/brain.py` — the `SYSTEM_PERSONA` string. Edit and restart.

The bot's "self" details (current obsession, mood, etc.) start from `persona_state.json` and then evolve per-user. Edit the JSON to change the starting vibe.

Mood keywords in `atrix/behavior/mood.py`. Energy thresholds in `atrix/behavior/energy.py`. Activity-hour window in `atrix/behavior/timing.py`. All tweakable.

---

## License

MIT.

---

## Credits

Built on [discord.py](https://github.com/Rapptz/discord.py), [OpenAI Python SDK](https://github.com/openai/openai-python), [DuckDuckGo Search](https://github.com/deedy5/duckduckgo_search), [google-genai](https://github.com/googleapis/python-genai), [yt-dlp](https://github.com/yt-dlp/yt-dlp).