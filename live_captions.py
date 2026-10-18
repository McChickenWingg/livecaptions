"""
Live Captions - real-time English captions with colour-coded speakers (Windows).

* Captures either SYSTEM AUDIO (whatever is playing: calls, videos, meetings)
  or a MICROPHONE, using WASAPI via PyAudioWPatch.
* Recognises English speech offline with Vosk (no cloud, no account).
* Uses Vosk's speaker-embedding model to tell voices apart and shows each
  voice in its own colour.

First run downloads two small models (~55 MB) to %APPDATA%\\LiveCaptions.
"""

import json
import os
import queue
import sys
import threading
import time
import urllib.request
import zipfile
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk, filedialog, messagebox

try:
    import numpy as np
    import pyaudiowpatch as pyaudio
    from vosk import Model, SpkModel, KaldiRecognizer, SetLogLevel
except ImportError as exc:  # pragma: no cover
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror(
        "Missing dependency",
        f"{exc}\n\nInstall requirements with:\n  pip install -r requirements.txt",
    )
    sys.exit(1)

SetLogLevel(-1)

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
APP_NAME = "LiveCaptions"
MODELS_DIR = os.path.join(
    os.environ.get("APPDATA", os.path.expanduser("~")), APP_NAME, "models"
)
ASR_MODEL = (
    "vosk-model-small-en-us-0.15",
    "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",
)
SPK_MODEL = (
    "vosk-model-spk-0.4",
    "https://alphacephei.com/vosk/models/vosk-model-spk-0.4.zip",
)

TARGET_RATE = 16000
MAX_SPEAKERS = 8
MIN_WORDS_FOR_RELIABLE_VOICEPRINT = 3
DEFAULT_SPLIT_THRESHOLD = 0.40  # cosine distance; lower = splits voices more eagerly

SPEAKER_COLOURS = [
    "#4FC3F7",  # blue
    "#FFB74D",  # orange
    "#81C784",  # green
    "#F06292",  # pink
    "#BA68C8",  # purple
    "#FFD54F",  # yellow
    "#4DB6AC",  # teal
    "#E57373",  # red
]

BG = "#15171c"
PANEL = "#1e2128"
FG = "#e8eaed"
MUTED = "#9aa0a6"


# --------------------------------------------------------------------------
# Model download
# --------------------------------------------------------------------------
def ensure_model(name_url, status_cb):
    name, url = name_url
    target = os.path.join(MODELS_DIR, name)
    if os.path.isdir(target):
        return target
    os.makedirs(MODELS_DIR, exist_ok=True)
    zip_path = os.path.join(MODELS_DIR, name + ".zip")

    def hook(blocks, block_size, total):
        if total > 0:
            pct = min(100, blocks * block_size * 100 // total)
            status_cb(f"Downloading {name}… {pct}%")

    status_cb(f"Downloading {name}…")
    urllib.request.urlretrieve(url, zip_path, hook)
    status_cb(f"Unpacking {name}…")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(MODELS_DIR)
    os.remove(zip_path)
    return target


# --------------------------------------------------------------------------
# Speaker tracking (online clustering of Vosk x-vectors)
# --------------------------------------------------------------------------
class SpeakerTracker:
    def __init__(self):
        self.lock = threading.Lock()
        self.threshold = DEFAULT_SPLIT_THRESHOLD
        self.reset()

    def reset(self):
        with self.lock:
            self.centroids = []
            self.counts = []
            self.last = 0

    @staticmethod
    def _unit(vec):
        v = np.asarray(vec, dtype=np.float64)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def assign(self, vec, reliable):
        """Return a speaker index for this utterance."""
        with self.lock:
            if vec is None:
                return self.last
            v = self._unit(vec)

            if not self.centroids:
                if not reliable:
                    return 0
                self.centroids.append(v)
                self.counts.append(1)
                self.last = 0
                return 0

            dists = [1.0 - float(np.dot(v, c)) for c in self.centroids]
            best = int(np.argmin(dists))

            if dists[best] < self.threshold:
                if reliable:
                    n = self.counts[best]
                    c = (self.centroids[best] * n + v) / (n + 1)
                    self.centroids[best] = self._unit(c)
                    self.counts[best] = n + 1
                self.last = best
                return best

            # Looks like a different voice
            if reliable and len(self.centroids) < MAX_SPEAKERS:
                self.centroids.append(v)
                self.counts.append(1)
                self.last = len(self.centroids) - 1
                return self.last

            # Too short to trust, or speaker limit hit: keep the previous speaker
            return best if (reliable and len(self.centroids) >= MAX_SPEAKERS) else self.last


# --------------------------------------------------------------------------
# Audio capture
# --------------------------------------------------------------------------
def list_devices():
    """Return [(label, device_info)] with system-audio loopbacks first."""
    p = pyaudio.PyAudio()
    loopbacks, mics = [], []
    default_loop_name = None
    try:
        try:
            default_loop_name = p.get_default_wasapi_loopback()["name"]
        except Exception:
            pass
        wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        for d in p.get_loopback_device_info_generator():
            loopbacks.append(("[System audio] " + d["name"], d))
        for i in range(p.get_device_count()):
            d = p.get_device_info_by_index(i)
            if (
                d["hostApi"] == wasapi["index"]
                and d["maxInputChannels"] > 0
                and not d.get("isLoopbackDevice", False)
            ):
                mics.append(("[Microphone] " + d["name"], d))
    finally:
        p.terminate()

    # put the default output's loopback first
    if default_loop_name:
        loopbacks.sort(key=lambda t: 0 if t[1]["name"] == default_loop_name else 1)
    return loopbacks + mics


class AudioCapture:
    def __init__(self, dev, out_q):
        self.dev = dev
        self.q = out_q
        self.rate = int(dev["defaultSampleRate"])
        self.ch = max(1, int(dev["maxInputChannels"]))
        self.p = None
        self.stream = None
        self.level = 0.0

    def start(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=self.ch,
            rate=self.rate,
            input=True,
            input_device_index=self.dev["index"],
            frames_per_buffer=self.rate // 10,
            stream_callback=self._callback,
        )
        self.stream.start_stream()

    def stop(self):
        try:
            if self.stream is not None:
                self.stream.stop_stream()
                self.stream.close()
        finally:
            self.stream = None
            if self.p is not None:
                self.p.terminate()
                self.p = None

    def _callback(self, in_data, frame_count, time_info, status):
        a = np.frombuffer(in_data, dtype=np.int16)
        if self.ch > 1:
            n = len(a) // self.ch * self.ch
            a = a[:n].reshape(-1, self.ch).mean(axis=1)
        a = a.astype(np.float32)

        if self.rate != TARGET_RATE and len(a) > 1:
            if self.rate % TARGET_RATE == 0:
                k = self.rate // TARGET_RATE
                n = len(a) // k * k
                a = a[:n].reshape(-1, k).mean(axis=1)
            else:
                n = int(len(a) * TARGET_RATE / self.rate)
                a = np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a)

        if len(a):
            self.level = float(np.sqrt(np.mean(a * a)))
        self.q.put(np.clip(a, -32768, 32767).astype(np.int16).tobytes())
        return (None, pyaudio.paContinue)


# --------------------------------------------------------------------------
# Recognition engine (background thread)
# --------------------------------------------------------------------------
class CaptionEngine(threading.Thread):
    def __init__(self, audio_q, ui_q, tracker):
        super().__init__(daemon=True)
        self.audio_q = audio_q
        self.ui_q = ui_q
        self.tracker = tracker

    def _status(self, msg):
        self.ui_q.put(("status", msg))

    def _emit_final(self, rec):
        res = json.loads(rec.Result())
        text = res.get("text", "").strip()
        if text:
            words = len(text.split())
            sid = self.tracker.assign(
                res.get("spk"), reliable=words >= MIN_WORDS_FOR_RELIABLE_VOICEPRINT
            )
            self.ui_q.put(("final", sid, text[0].upper() + text[1:]))
        self.ui_q.put(("partial", ""))

    def run(self):
        try:
            asr_path = ensure_model(ASR_MODEL, self._status)
            spk_path = ensure_model(SPK_MODEL, self._status)
            self._status("Loading models…")
            model = Model(asr_path)
            spk_model = SpkModel(spk_path)
            rec = KaldiRecognizer(model, TARGET_RATE)
            rec.SetSpkModel(spk_model)
        except Exception as exc:
            self.ui_q.put(("error", f"Could not load speech models:\n{exc}"))
            return

        self.ui_q.put(("ready",))
        last_partial = ""
        while True:
            try:
                data = self.audio_q.get(timeout=0.2)
            except queue.Empty:
                continue

            if data is None:  # flush request (capture stopped)
                try:
                    res = json.loads(rec.FinalResult())
                    text = res.get("text", "").strip()
                    if text:
                        sid = self.tracker.assign(res.get("spk"), reliable=False)
                        self.ui_q.put(("final", sid, text[0].upper() + text[1:]))
                except Exception:
                    pass
                self.ui_q.put(("partial", ""))
                last_partial = ""
                continue

            if rec.AcceptWaveform(data):
                self._emit_final(rec)
                last_partial = ""
            else:
                partial = json.loads(rec.PartialResult()).get("partial", "")
                if partial != last_partial:
                    last_partial = partial
                    self.ui_q.put(("partial", partial))


# --------------------------------------------------------------------------
# GUI
# --------------------------------------------------------------------------
class CaptionOverlay:
    """Borderless, always-on-top window that shows only the live captions."""

    def __init__(self, master, on_close):
        self.on_close = on_close
        self.alpha = 0.90
        self.show_names = tk.BooleanVar(value=True)
        self.last_sid = None

        self.win = tk.Toplevel(master)
        self.win.overrideredirect(True)
        self.win.attributes("-topmost", True)
        self.win.attributes("-alpha", self.alpha)
        self.win.configure(bg=BG)
        self.win.minsize(280, 100)
        sw, sh = self.win.winfo_screenwidth(), self.win.winfo_screenheight()
        w, h = 820, 200
        self.win.geometry(f"{w}x{h}+{(sw - w) // 2}+{sh - h - 90}")

        self.font = tkfont.Font(family="Segoe UI", size=22)
        self.bold = tkfont.Font(family="Segoe UI", size=22, weight="bold")

        # thin title bar: drag to move, X to close
        bar = tk.Frame(self.win, bg=PANEL, height=24)
        bar.pack(fill="x")
        title = tk.Label(bar, text="Live Captions   ·   drag to move   ·   right-click for options",
                         bg=PANEL, fg=MUTED, font=("Segoe UI", 9))
        title.pack(side="left", padx=8)
        close = tk.Label(bar, text="  ✕  ", bg=PANEL, fg=MUTED, font=("Segoe UI", 10, "bold"),
                         cursor="hand2")
        close.pack(side="right")
        close.bind("<Button-1>", lambda e: self.close())
        close.bind("<Enter>", lambda e: close.configure(bg="#d93025", fg="white"))
        close.bind("<Leave>", lambda e: close.configure(bg=PANEL, fg=MUTED))

        self.text = tk.Text(self.win, wrap="word", bg=BG, fg=FG, relief="flat",
                            padx=16, pady=10, font=self.font, spacing1=2, spacing3=6,
                            state="disabled", cursor="arrow", highlightthickness=0,
                            takefocus=0)
        self.text.pack(fill="both", expand=True)
        self.text.tag_configure("partial", foreground=MUTED)
        for i, colour in enumerate(SPEAKER_COLOURS):
            self.text.tag_configure(f"spk{i}", foreground=colour)
            self.text.tag_configure(f"lbl{i}", foreground=colour, font=self.bold)

        # resize grip (bottom-right)
        self.grip = tk.Label(self.win, text="◢", bg=BG, fg=MUTED, cursor="size_nw_se",
                             font=("Segoe UI", 10))
        self.grip.place(relx=1.0, rely=1.0, anchor="se")
        self.grip.bind("<Button-1>", self._start_resize)
        self.grip.bind("<B1-Motion>", self._do_resize)

        # drag to move from the bar or the text itself
        for widget in (bar, title, self.text):
            widget.bind("<Button-1>", self._start_move)
            widget.bind("<B1-Motion>", self._do_move)

        # right-click menu
        self.menu = tk.Menu(self.win, tearoff=0)
        self.menu.add_command(label="Larger text", command=lambda: self.resize_font(2))
        self.menu.add_command(label="Smaller text", command=lambda: self.resize_font(-2))
        self.menu.add_separator()
        self.menu.add_command(label="More opaque", command=lambda: self.change_alpha(0.1))
        self.menu.add_command(label="More transparent", command=lambda: self.change_alpha(-0.1))
        self.menu.add_separator()
        self.menu.add_checkbutton(label="Show speaker names", variable=self.show_names)
        self.menu.add_command(label="Clear", command=self.clear)
        self.menu.add_separator()
        self.menu.add_command(label="Close pop-out", command=self.close)
        for widget in (self.win, bar, title, self.text):
            widget.bind("<Button-3>", self._show_menu)

    # ----- window behaviour -----
    def _show_menu(self, e):
        self.menu.tk_popup(e.x_root, e.y_root)

    def _start_move(self, e):
        self._dx = e.x_root - self.win.winfo_x()
        self._dy = e.y_root - self.win.winfo_y()
        return "break"

    def _do_move(self, e):
        self.win.geometry(f"+{e.x_root - self._dx}+{e.y_root - self._dy}")
        return "break"

    def _start_resize(self, e):
        self._rx, self._ry = e.x_root, e.y_root
        self._rw, self._rh = self.win.winfo_width(), self.win.winfo_height()

    def _do_resize(self, e):
        w = max(280, self._rw + e.x_root - self._rx)
        h = max(100, self._rh + e.y_root - self._ry)
        self.win.geometry(f"{w}x{h}")

    def resize_font(self, delta):
        size = max(10, min(72, self.font.cget("size") + delta))
        self.font.configure(size=size)
        self.bold.configure(size=size)

    def change_alpha(self, delta):
        self.alpha = max(0.3, min(1.0, self.alpha + delta))
        self.win.attributes("-alpha", self.alpha)

    def close(self):
        try:
            self.win.destroy()
        finally:
            self.on_close()

    # ----- caption content -----
    def _edit(self, fn):
        self.text.configure(state="normal")
        try:
            fn()
        finally:
            self.text.configure(state="disabled")

    def _remove_partial(self):
        r = self.text.tag_ranges("partial")
        if r:
            self.text.delete(r[0], r[-1])

    def add_final(self, sid, text):
        sid %= len(SPEAKER_COLOURS)

        def do():
            self._remove_partial()
            if sid != self.last_sid:
                if self.text.index("end-1c") != "1.0":
                    self.text.insert("end", "\n")
                if self.show_names.get():
                    self.text.insert("end", f"Speaker {sid + 1}: ", (f"lbl{sid}",))
                self.last_sid = sid
            self.text.insert("end", text + " ", (f"spk{sid}",))
            if int(self.text.index("end-1c").split(".")[0]) > 300:
                self.text.delete("1.0", "100.0")

        self._edit(do)
        self.text.see("end")

    def set_partial(self, text):
        def do():
            self._remove_partial()
            if text:
                self.text.insert("end", text, ("partial",))

        self._edit(do)
        self.text.see("end")

    def clear(self):
        self._edit(lambda: self.text.delete("1.0", "end"))
        self.last_sid = None


class App:
    def __init__(self, root):
        self.root = root
        root.title("Live Captions")
        root.geometry("900x560")
        root.minsize(560, 360)
        root.configure(bg=BG)

        self.audio_q = queue.Queue()
        self.ui_q = queue.Queue()
        self.tracker = SpeakerTracker()
        self.engine = CaptionEngine(self.audio_q, self.ui_q, self.tracker)
        self.capture = None
        self.devices = []
        self.last_sid = None
        self.known_speakers = set()
        self.transcript = []  # [(speaker_index, text)]
        self.models_ready = False
        self.overlay = None

        self._build_style()
        self._build_ui()
        self.refresh_devices()

        self.engine.start()
        self.root.after(50, self.poll)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ---------- UI construction ----------
    def _build_style(self):
        s = ttk.Style()
        s.theme_use("clam")
        s.configure(".", background=BG, foreground=FG, fieldbackground=PANEL)
        s.configure("TFrame", background=BG)
        s.configure("TLabel", background=BG, foreground=FG)
        s.configure("Muted.TLabel", foreground=MUTED)
        s.configure("TButton", background=PANEL, foreground=FG, padding=6, borderwidth=0)
        s.map("TButton", background=[("active", "#2a2e37"), ("disabled", "#191b20")],
              foreground=[("disabled", "#5f6368")])
        s.configure("Accent.TButton", background="#1a73e8", foreground="white")
        s.map("Accent.TButton", background=[("active", "#3b8af0"), ("disabled", "#2b3a55")])
        s.configure("Stop.TButton", background="#d93025", foreground="white")
        s.map("Stop.TButton", background=[("active", "#e5534b")])
        s.configure("TCheckbutton", background=BG, foreground=FG)
        s.map("TCheckbutton", background=[("active", BG)])
        s.configure("TCombobox", fieldbackground=PANEL, background=PANEL, foreground=FG,
                    arrowcolor=FG, borderwidth=0)
        s.map("TCombobox", fieldbackground=[("readonly", PANEL)],
              foreground=[("readonly", FG)])
        s.configure("Horizontal.TScale", background=BG)
        s.configure("Level.Horizontal.TProgressbar", troughcolor=PANEL, background="#34a853",
                    borderwidth=0)

    def _build_ui(self):
        top = ttk.Frame(self.root, padding=(12, 10, 12, 4))
        top.pack(fill="x")

        ttk.Label(top, text="Audio source").pack(side="left")
        self.device_var = tk.StringVar()
        self.device_box = ttk.Combobox(top, textvariable=self.device_var, state="readonly")
        self.device_box.pack(side="left", fill="x", expand=True, padx=8)
        ttk.Button(top, text="Refresh", command=self.refresh_devices).pack(side="left")

        row2 = ttk.Frame(self.root, padding=(12, 4))
        row2.pack(fill="x")
        self.start_btn = ttk.Button(row2, text="Loading…", style="Accent.TButton",
                                    command=self.toggle_capture, state="disabled", width=12)
        self.start_btn.pack(side="left")
        ttk.Button(row2, text="Clear", command=self.clear).pack(side="left", padx=(8, 0))
        ttk.Button(row2, text="Save…", command=self.save).pack(side="left", padx=(8, 0))
        ttk.Button(row2, text="Reset voices", command=self.reset_voices).pack(side="left", padx=(8, 0))
        self.pop_btn = ttk.Button(row2, text="Pop out", command=self.toggle_overlay)
        self.pop_btn.pack(side="left", padx=(8, 0))

        self.top_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row2, text="Always on top", variable=self.top_var,
                        command=lambda: self.root.attributes("-topmost", self.top_var.get())
                        ).pack(side="right")
        ttk.Button(row2, text="A+", width=3, command=lambda: self.resize_font(2)).pack(side="right", padx=(0, 8))
        ttk.Button(row2, text="A-", width=3, command=lambda: self.resize_font(-2)).pack(side="right", padx=(0, 4))

        row3 = ttk.Frame(self.root, padding=(12, 4))
        row3.pack(fill="x")
        ttk.Label(row3, text="Voice separation", style="Muted.TLabel").pack(side="left")
        ttk.Label(row3, text="fewer", style="Muted.TLabel").pack(side="left", padx=(8, 2))
        self.thr_var = tk.DoubleVar(value=DEFAULT_SPLIT_THRESHOLD)
        ttk.Scale(row3, from_=0.80, to=0.15, variable=self.thr_var, length=160,
                  command=self.on_threshold).pack(side="left")
        ttk.Label(row3, text="more speakers", style="Muted.TLabel").pack(side="left", padx=(2, 16))
        ttk.Label(row3, text="Level", style="Muted.TLabel").pack(side="left")
        self.level_bar = ttk.Progressbar(row3, style="Level.Horizontal.TProgressbar",
                                         maximum=100, length=120)
        self.level_bar.pack(side="left", padx=8)

        self.legend = ttk.Frame(self.root, padding=(12, 2))
        self.legend.pack(fill="x")

        body = tk.Frame(self.root, bg=PANEL)
        body.pack(fill="both", expand=True, padx=12, pady=(4, 6))
        self.font = tkfont.Font(family="Segoe UI", size=16)
        self.bold_font = tkfont.Font(family="Segoe UI", size=16, weight="bold")
        self.text = tk.Text(body, wrap="word", bg=PANEL, fg=FG, insertbackground=FG,
                            relief="flat", padx=14, pady=12, font=self.font,
                            spacing1=2, spacing3=6, state="disabled", cursor="arrow")
        sb = ttk.Scrollbar(body, command=self.text.yview)
        self.text.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.text.pack(side="left", fill="both", expand=True)

        self.text.tag_configure("partial", foreground=MUTED)
        for i, colour in enumerate(SPEAKER_COLOURS):
            self.text.tag_configure(f"spk{i}", foreground=colour)
            self.text.tag_configure(f"lbl{i}", foreground=colour, font=self.bold_font)

        self.status_var = tk.StringVar(value="Starting…")
        ttk.Label(self.root, textvariable=self.status_var, style="Muted.TLabel",
                  padding=(14, 0, 14, 8)).pack(fill="x")

    # ---------- device handling ----------
    def refresh_devices(self):
        try:
            self.devices = list_devices()
        except Exception as exc:
            self.devices = []
            messagebox.showerror("Audio devices", f"Could not list audio devices:\n{exc}")
        labels = [lbl for lbl, _ in self.devices]
        self.device_box["values"] = labels
        if labels:
            self.device_box.current(0)

    # ---------- actions ----------
    def toggle_capture(self):
        if self.capture is None:
            self.start_capture()
        else:
            self.stop_capture()

    def start_capture(self):
        idx = self.device_box.current()
        if idx < 0 or idx >= len(self.devices):
            messagebox.showwarning("Audio source", "Choose an audio source first.")
            return
        try:
            cap = AudioCapture(self.devices[idx][1], self.audio_q)
            cap.start()
        except Exception as exc:
            messagebox.showerror("Audio", f"Could not open that device:\n{exc}")
            return
        self.capture = cap
        self.start_btn.configure(text="Stop", style="Stop.TButton")
        self.device_box.configure(state="disabled")
        self.status_var.set("Listening… captions appear when speech is detected.")

    def stop_capture(self):
        if self.capture is not None:
            self.capture.stop()
            self.capture = None
        self.audio_q.put(None)  # flush last words
        self.start_btn.configure(text="Start", style="Accent.TButton")
        self.device_box.configure(state="readonly")
        self.level_bar["value"] = 0
        self.status_var.set("Stopped.")

    def on_threshold(self, _value):
        self.tracker.threshold = float(self.thr_var.get())

    def resize_font(self, delta):
        size = max(8, min(60, self.font.cget("size") + delta))
        self.font.configure(size=size)
        self.bold_font.configure(size=size)

    def clear(self):
        self._text_edit(lambda: self.text.delete("1.0", "end"))
        if self.overlay:
            self.overlay.clear()
        self.transcript.clear()
        self.last_sid = None

    def reset_voices(self):
        self.tracker.reset()
        self.last_sid = None
        self.known_speakers.clear()
        for w in self.legend.winfo_children():
            w.destroy()
        self.status_var.set("Voice memory reset - speakers will be re-detected.")

    def toggle_overlay(self):
        if self.overlay is not None:
            self.overlay.close()
            return
        self.overlay = CaptionOverlay(self.root, self._overlay_closed)
        for sid, text in self.transcript[-8:]:  # catch up on recent captions
            self.overlay.add_final(sid, text)
        self.pop_btn.configure(text="Close pop-out")

    def _overlay_closed(self):
        self.overlay = None
        self.pop_btn.configure(text="Pop out")

    def save(self):
        if not self.transcript:
            messagebox.showinfo("Save", "Nothing to save yet.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".txt", filetypes=[("Text file", "*.txt")],
            initialfile=time.strftime("captions_%Y-%m-%d_%H-%M.txt"))
        if not path:
            return
        lines, prev = [], None
        for sid, text in self.transcript:
            if sid != prev:
                lines.append(f"\nSpeaker {sid + 1}: {text}")
                prev = sid
            else:
                lines[-1] += " " + text
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines).strip() + "\n")
        self.status_var.set(f"Saved to {path}")

    # ---------- text helpers ----------
    def _text_edit(self, fn):
        self.text.configure(state="normal")
        try:
            fn()
        finally:
            self.text.configure(state="disabled")

    def _remove_partial(self):
        r = self.text.tag_ranges("partial")
        if r:
            self.text.delete(r[0], r[-1])

    def _add_legend(self, sid):
        if sid in self.known_speakers:
            return
        self.known_speakers.add(sid)
        colour = SPEAKER_COLOURS[sid % len(SPEAKER_COLOURS)]
        tk.Label(self.legend, text=f"●  Speaker {sid + 1}", fg=colour, bg=BG,
                 font=("Segoe UI", 11, "bold")).pack(side="left", padx=(0, 14))

    def add_final(self, sid, text):
        sid %= len(SPEAKER_COLOURS)
        self._add_legend(sid)
        self.transcript.append((sid, text))

        def do():
            self._remove_partial()
            if sid != self.last_sid:
                if self.text.index("end-1c") != "1.0":
                    self.text.insert("end", "\n")
                self.text.insert("end", f"Speaker {sid + 1}: ", (f"lbl{sid}",))
                self.last_sid = sid
            self.text.insert("end", text + " ", (f"spk{sid}",))

        self._text_edit(do)
        self.text.see("end")
        if self.overlay:
            self.overlay.add_final(sid, text)

    def set_partial(self, text):
        def do():
            self._remove_partial()
            if text:
                self.text.insert("end", text, ("partial",))

        self._text_edit(do)
        self.text.see("end")
        if self.overlay:
            self.overlay.set_partial(text)

    # ---------- event loop ----------
    def poll(self):
        try:
            while True:
                ev = self.ui_q.get_nowait()
                kind = ev[0]
                if kind == "status":
                    self.status_var.set(ev[1])
                elif kind == "ready":
                    self.models_ready = True
                    self.start_btn.configure(text="Start", state="normal")
                    self.status_var.set("Ready. Pick a source and press Start.")
                elif kind == "error":
                    self.status_var.set("Error")
                    messagebox.showerror("Live Captions", ev[1])
                elif kind == "partial":
                    self.set_partial(ev[1])
                elif kind == "final":
                    self.add_final(ev[1], ev[2])
        except queue.Empty:
            pass

        if self.capture is not None:
            target = min(100.0, self.capture.level / 2500.0 * 100.0)
            cur = self.level_bar["value"]
            self.level_bar["value"] = target if target > cur else cur * 0.8
        self.root.after(50, self.poll)

    def on_close(self):
        try:
            if self.capture is not None:
                self.capture.stop()
        finally:
            self.root.destroy()


def main():
    try:  # crisp text on high-DPI displays
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
