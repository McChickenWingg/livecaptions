# Live Captions

Real-time English captions for Windows, with a different colour for each speaker.

Live Captions listens to your PC's audio (calls, videos, meetings, livestreams) or a microphone and writes out what's being said as it happens. A speaker-detection system tells voices apart and shows each one in its own colour. A pop-out window shows just the captions, always on top, so you can follow along while you use other apps.

Speech recognition runs entirely on your computer using [Vosk](https://alphacephei.com/vosk/). There is no cloud service, account or subscription, and your audio never leaves your machine.

## Features

- Live English captions from **system audio** or a **microphone**
- Captions appear only when speech is detected: grey while words are being recognised, then settling into the speaker's colour
- **Speaker changes marked with `>>>`** (the default): a new line starting with `>>>` each time a different person speaks, and no marker while the same person carries on
- Optional **colour-coded speakers** (up to 8) with "Speaker 1, 2, 3…" labels and a legend
- Adjustable **voice separation** slider to control how readily voices are split
- Optional **punctuation** (capital letters, commas, full stops and question marks)
- **One-box mode**: turn off speaker identification to show all text together in one colour, with no markers or labels
- **Pop-out window**: borderless, always on top, draggable, resizable, with adjustable text size and transparency, and optional speaker names
- Audio level meter, font size controls, clear and reset buttons
- Save the transcript to a text file with speaker labels
- Works offline after a one-time model download (about 55 MB)

## Requirements

- Windows 10 or 11
- Python 3.10 to 3.12 (from [python.org](https://www.python.org/downloads/); tick **Add python.exe to PATH** during install)

Very new Python versions (3.13 and later) may not have builds of the dependencies yet. If installation fails, use Python 3.12.

## Quick start

Put `live_captions.py` and `requirements.txt` in the same folder, then open a terminal in that folder and run:

```
pip install -r requirements.txt
python live_captions.py
```

The first launch downloads two small speech models to `%APPDATA%\LiveCaptions\models`. After that the app works offline.

## Using the app

1. Choose an **audio source**. Sources marked `[System audio]` capture whatever is playing on that output device. Sources marked `[Microphone]` capture a mic. The default output device's system audio is listed first.
2. Press **Start**. Captions appear as soon as speech is detected, and the level bar shows whether audio is coming in.
3. Press **Pop out** to open the caption-only window.

### Pop-out window

| Action | How |
| --- | --- |
| Move | Drag the title bar or the caption text |
| Resize | Drag the ◢ corner |
| Options | Right-click: text size, transparency, show/hide speaker names, clear, close |
| Close | Click ✕, or press **Close pop-out** in the main window |

### Punctuation

Tick **Add punctuation** to tidy each caption: it capitalises the first word and the word "I", adds commas after openers such as "well" or "so" and before "but", and ends the sentence with a full stop or question mark. It uses simple rules rather than an AI model, so it is only a guide. For example, it treats any sentence starting with a word like "what", "how" or "is" as a question, and it never adds exclamation marks or most commas. The setting applies to new captions, including the pop-out and saved transcripts.

### Showing speakers

The **Speakers** dropdown chooses how speaker changes are shown. It applies to new captions, the pop-out window and saved transcripts.

| Option | What you see |
| --- | --- |
| **Show >>> when speaker changes** (default) | All text in one colour. When a different person starts speaking, their words begin on a new line with `>>>` in front. While the same person keeps talking, no marker is added. |
| **Colours & labels** | Each speaker gets their own colour and a "Speaker N:" label, with a colour legend. |
| **Off (one box)** | Speaker detection is switched off. All captions flow together in one box, with no markers, labels or line breaks between people. |

Example with `>>>`:

```
Hello there, how are you?
>>> I'm fine thanks, and you?
>>> Great.
```

### Voice separation

The slider sets how different two voices must sound before they're treated as separate speakers. The default is 0.50.

- If two people are merged into one colour, move the slider toward **more speakers**.
- If one person is split into several colours, move it toward **fewer**.
- **Reset voices** clears everything the app has learned, which is useful when a new conversation begins.

To avoid too many speakers appearing, the app is cautious about learning and creating voices. A speaker's stored voice is only refined by sentences that are long enough and clearly match just that speaker. A new speaker needs a long sentence that sounds clearly different, or two similar-sounding "maybe new" sentences in a row. Speakers whose voices drift together are merged. The trade-off is that a genuinely new person's very first sentence may be labelled as the previous speaker. Very short sentences (a few words) are always attributed to the nearest or previous speaker, because their voiceprints are unreliable.

If you still see too many speakers, raise the slider value (toward **fewer**). If different people are being merged, lower it.

## Building a standalone .exe

Keep `live_captions.py`, `requirements.txt` and `build_exe.bat` in the same folder and double-click `build_exe.bat` (a normal double-click, not "Run as administrator"). When it finishes, the app is at `dist\LiveCaptions.exe`.

The models are not bundled into the .exe. They download on first run, as with the Python version.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `pip` can't find PyAudioWPatch or Vosk | Install Python 3.12 and try again |
| `python` is not recognized | Reinstall Python and tick **Add python.exe to PATH** |
| No captions from system audio | Make sure something is actually playing on the selected output device, and check the level bar |
| Captions are slow or inaccurate | Close other heavy apps, or switch to a larger Vosk model (see below) |
| Model download fails | Check your internet connection and relaunch; the download resumes from scratch |
| Everyone is shown as one speaker | Move the voice separation slider toward **more speakers** |

## Using a more accurate model

The app uses Vosk's small English model for speed. To trade speed for accuracy, change `ASR_MODEL` near the top of `live_captions.py` to a larger English model from the [Vosk model list](https://alphacephei.com/vosk/models). Use the model's folder name and its zip URL.

## Limitations

- English only
- Punctuation is rule-based and optional, not AI-generated
- Speaker detection is approximate. Similar-sounding voices can be merged, and very short sentences are attributed to the nearest or previous speaker
- People talking at the same time, or straight after each other with no pause, are captioned as one sentence under one speaker
- System audio capture works with Windows output devices that support WASAPI loopback

## Built with

Python, Tkinter, [Vosk](https://alphacephei.com/vosk/) (speech recognition and speaker embeddings), [PyAudioWPatch](https://github.com/s0d3s/PyAudioWPatch) (WASAPI audio capture) and NumPy.

## Credits

Speech and speaker models are provided by the Vosk project (Alpha Cephei) under their own licences. See the Vosk site for details.
