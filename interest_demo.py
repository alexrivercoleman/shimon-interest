#!/usr/bin/env python3
"""
One interestingness number (0-1) for live piano playing.

    pip install numpy sounddevice              (plus mido python-rtmidi for --midi)
    python interest_demo.py                    # mic, estimate the key as you play
    python interest_demo.py C major            # mic, key locked
    python interest_demo.py --midi             # MIDI keyboard instead of the mic (real velocities)
    python interest_demo.py --quiet            # print only the number, one line per note

Four parts, each 0-1, go into it:
  melody  inverse Lerdahl attraction, inverted-U vs. recent playing          (interest_demo_v2)
  rhythm  change in inter-onset ratio on a 2/3/4 subdivision grid, same U    (timing_analyzer)
  energy  0.6 * loudness + 0.4 * notes/sec over the last 3 s                 (shimon density/velocity)
  key     a confirmed key change, sized by circle-of-fifths distance, fades  (interestingnessKeyChange)

  interest = (melody + rhythm + energy) / 3, pushed toward 1 by a key change,
  then smoothed over the last few notes.
"""
import sys, math, queue, time, argparse
from collections import deque
import numpy as np

SR, HOP = 44100, 512
GATE = 0.003                   # ignore anything quieter than this (raise it if you get phantom notes)
NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl-Kessler key profiles, tonic first
KK_MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
# Lerdahl anchoring strength by semitones above tonic (same values as the spreadsheet)
S = {"major": [5, 1, 2, 1, 3, 2, 1, 4, 1, 2, 1, 2],
     "minor": [5, 1, 2, 3, 1, 2, 1, 4, 2, 1, 2, 2]}

RATIOS = (1/4, 1/3, 1/2, 2/3, 1, 3/2, 2, 3, 4)   # rhythm: IOI ratios on a 2/3/4 subdivision grid
MAX_GAP = 2.0                  # a gap longer than this (s) starts a new phrase, no rhythm reading
WINDOW, DENSITY_MAX = 3.0, 8   # energy: notes in the last 3 s; 8 notes/sec counts as fully busy
CONFIRM = 3                    # notes in a row a new key must win before it counts as a key change
W_MELODY, W_RHYTHM, W_ENERGY = 1/3, 1/3, 1/3
SMOOTH = 0.3                   # how fast the output number follows each new note


def name(m):
    return f"{NAMES[m % 12]}{m // 12 - 1}"


def pitch(x):
    """YIN: returns the MIDI note in x, or None."""
    W = len(x) // 2
    # difference function via FFT: d(t) = sum x[j]^2 + sum x[j+t]^2 - 2 sum x[j] x[j+t]
    n = 1 << (len(x) + W - 1).bit_length()
    corr = np.fft.irfft(np.fft.rfft(x, n) * np.conj(np.fft.rfft(x[:W], n)), n)[1:W]
    e = np.cumsum(x ** 2)
    d = e[W - 1] + (e[W:2 * W - 1] - e[:W - 1]) - 2 * corr
    cmnd = d * np.arange(1, W) / np.maximum(np.cumsum(d), 1e-12)
    lo = SR // 2000
    best = float(np.min(cmnd[lo:]))
    if best > 0.35:                                     # nothing periodic in here
        return None
    # first dip that's nearly as good as the best one (taking the best outright gives octave-down errors)
    i = int(np.argmax(cmnd[lo:] < max(0.15, best + 0.1))) + lo
    while i + 1 < len(cmnd) and cmnd[i + 1] < cmnd[i]:
        i += 1
    if 0 < i < len(cmnd) - 1:                           # parabolic interpolation for sub-sample lag
        a, b, c = cmnd[i - 1:i + 2]
        i += 0.5 * (a - c) / max(a - 2 * b + c, 1e-12)
    return round(69 + 12 * math.log2(SR / (i + 1) / 440))


class Listener:
    """Feed it audio blocks; it returns (MIDI note, loudness 0-1) whenever a new note is struck.

    Onsets come from spectral flux (new frequency energy appearing), so a soft note
    after a loud one, or a note played while the last one still rings, still counts.
    Before estimating the pitch, whatever was already sounding before the onset is
    subtracted out of the spectrum, so the ringing previous note doesn't get picked up.
    """
    def __init__(self):
        self.buf = np.zeros(4096)
        self.win = np.hanning(1024)
        self.win2 = np.hanning(2048)
        self.spec = np.zeros(116)
        self.flux = [0.0] * 10
        self.last_rms = 0.0
        self.since_onset = -10           # ignore the first ~0.2 s while the buffers fill
        self.wait = None
        self.before = None

    def feed(self, block):
        self.buf = np.concatenate((self.buf[len(block):], block))
        rms = float(np.sqrt(np.mean(self.buf[-1024:] ** 2)))
        spec = np.log1p(np.abs(np.fft.rfft(self.buf[-1024:] * self.win))[:116])   # up to ~5 kHz
        f = float(np.sum(np.maximum(spec - self.spec, 0)))
        self.spec = spec
        self.since_onset += 1
        if (rms > GATE and self.since_onset > 6 and rms > self.last_rms   # energy not falling: not a damper thump
                and f > 2 * np.mean(self.flux) + 1 and f >= max(self.flux[-2:])):
            self.since_onset, self.wait = 0, 5         # onset: wait ~60 ms so the window holds only the new note
            self.before = np.abs(np.fft.rfft(self.buf[-3072:-1024] * self.win2))
        self.flux = self.flux[1:] + [f]
        self.last_rms = rms
        if self.wait is not None:
            self.wait -= 1
            if self.wait == 0:
                self.wait = None
                X = np.fft.rfft(self.buf[-2048:] * self.win2)
                X *= np.clip(1 - 1.5 * self.before / np.maximum(np.abs(X), 1e-12), 0, 1)
                y = np.fft.irfft(X, 2048)[256:-256] / self.win2[256:-256]
                m = pitch(y)
                # loudness: 0 at the gate, 1 at 40 dB above it (mic stand-in for MIDI velocity)
                return None if m is None else (m, min(max(20 * math.log10(rms / GATE) / 40, 0.0), 1.0))
        return None


class Surprise:
    """Running average of some measure x; scores how unusual each new x is on an inverted U."""
    def __init__(self, mu, var, floor=0.5):
        self.mu, self.var, self.floor = mu, var, floor

    def __call__(self, x):
        z = (x - self.mu) / max(math.sqrt(self.var), self.floor)   # how unusual vs. recent playing
        u = abs(z) / 1.5
        d = x - self.mu
        self.mu += 0.2 * d
        self.var = 0.8 * (self.var + 0.2 * d * d)
        return z, u * math.exp(1 - u)                 # peaks at |z| = 1.5


def circle_steps(a, b):
    """Distance between two keys' signatures on the circle of fifths, 0-6 (relative major/minor = 0)."""
    pa, pb = [((t + (3 if mode == "minor" else 0)) * 7) % 12 for t, mode in (a, b)]
    d = abs(pa - pb)
    return min(d, 12 - d)


class Model:
    def __init__(self, locked_key=None):
        self.locked = locked_key
        self.hist = np.zeros(12)               # recent pitch classes, for the key estimate
        self.melody = Surprise(1.0, 1.0)       # on log(1/alpha): what "normal" sounds like
        self.rhythm = Surprise(0.0, 0.25)      # on |log2(IOI ratio)|: 0 = same rhythm as before
        self.prev = None
        self.cur, self.cand, self.hits = locked_key, None, 0
        self.kick = 0.0                        # key-change boost, fades over the next few notes
        self.last_t = self.last_ioi = None
        self.window = deque()                  # (time, loudness) of notes in the last WINDOW seconds
        self.score = 0.0

    def key(self):
        if self.locked:
            return self.locked
        if self.hist.sum() < 4 or self.hist.std() == 0:
            return None
        return max(((np.corrcoef(self.hist, np.roll(p, t))[0, 1], t, mode)
                    for t in range(12) for mode, p in (("major", KK_MAJ), ("minor", KK_MIN))))[1:]

    def track_key(self):
        """Keeps self.cur as the confirmed key. Returns circle steps when it changes, else None."""
        k = self.key()
        if k is None or k == self.cur:
            self.cand, self.hits = None, 0
            return None
        if self.cur is None:                   # first key found: not a change
            self.cur = k
            return None
        self.hits = self.hits + 1 if k == self.cand else 1
        self.cand = k
        if self.hits < CONFIRM:
            return None
        steps = circle_steps(self.cur, k)
        self.cur, self.cand, self.hits = k, None, 0
        return steps

    def note(self, m, t, loud):
        """m = MIDI note, t = onset time in seconds, loud = 0-1. Returns everything show() needs."""
        prev, self.prev = self.prev, m
        steps = self.track_key()
        key = self.cur
        self.hist *= 0.9
        self.hist[m % 12] += 1

        # melody: inverse Lerdahl attraction, unusual vs. recent moves
        melody = 0.0
        if prev is not None and m != prev:
            s1 = S[key[1]][(prev - key[0]) % 12] if key else 1
            s2 = S[key[1]][(m - key[0]) % 12] if key else 1
            _, melody = self.melody(math.log((s1 / s2) * (m - prev) ** 2))

        # rhythm: snap this IOI / last IOI to the subdivision grid, unusual vs. recent ratios
        rhythm = 0.0
        ioi = t - self.last_t if self.last_t is not None else None
        if ioi is not None and ioi > MAX_GAP:
            ioi = None
        if ioi and self.last_ioi:
            r = min(RATIOS, key=lambda q: abs(math.log2(ioi / self.last_ioi / q)))
            _, rhythm = self.rhythm(abs(math.log2(r)))
        self.last_t, self.last_ioi = t, ioi

        # energy: loud and busy over the last few seconds
        self.window.append((t, loud))
        while t - self.window[0][0] > WINDOW:
            self.window.popleft()
        vel = sum(v for _, v in self.window) / len(self.window)
        density = min(len(self.window) / WINDOW / DENSITY_MAX, 1.0)
        energy = 0.6 * vel + 0.4 * density

        # key change: bigger jump on the circle of fifths = bigger boost
        if steps is not None:
            self.kick = max(self.kick, max(steps, 1) / 6)
        kick = self.kick
        self.kick *= 0.75

        base = W_MELODY * melody + W_RHYTHM * rhythm + W_ENERGY * energy
        raw = base + (1 - base) * kick
        self.score += SMOOTH * (raw - self.score)
        return dict(m=m, prev=prev, key=key, steps=steps, melody=melody, rhythm=rhythm,
                    energy=energy, kick=kick, raw=raw, score=self.score)


def show(r):
    k = f"{NAMES[r['key'][0]]} {r['key'][1]}" if r["key"] else "(finding key)"
    if r["prev"] is None:
        move = f"{'':>4}    {name(r['m']):<4}     "
    else:
        move = f"{name(r['prev']):>4} -> {name(r['m']):<4} {r['m'] - r['prev']:+4d}"
    bar = "#" * round(r["score"] * 20)
    line = (f"  {move}  mel {r['melody']:.2f}  rhy {r['rhythm']:.2f}  energy {r['energy']:.2f}  "
            f"key {r['kick']:.2f}  |  INTEREST {r['score']:.2f} {bar:<20}  {k}")
    if r["steps"] is not None:
        line += f"  << key change ({r['steps']} on the circle)"
    print(line, flush=True)


def mic_notes():
    """Yields (MIDI note, time in s, loudness 0-1) from the default mic."""
    import sounddevice as sd
    ear, q = Listener(), queue.Queue()
    # On Windows the default (MME) mic adds ~180 ms of lag; WASAPI on the same mic is ~20 ms
    device, extra = None, None
    if sys.platform == "win32":
        for api in sd.query_hostapis():
            if api["name"] == "Windows WASAPI" and api["default_input_device"] >= 0:
                device, extra = api["default_input_device"], sd.WasapiSettings(auto_convert=True)
    print(f"  Mic: {sd.query_devices(device, 'input')['name']}", file=sys.stderr)
    with sd.InputStream(device=device, samplerate=SR, channels=1, blocksize=HOP, latency="low",
                        extra_settings=extra, callback=lambda data, *_: q.put(data[:, 0].copy())):
        blocks = 0
        while True:
            hit = ear.feed(q.get())
            blocks += 1
            if hit is not None:
                yield hit[0], blocks * HOP / SR, hit[1]


def midi_notes(port_name):
    """Yields (MIDI note, time in s, velocity 0-1) from a MIDI input."""
    import mido
    names = mido.get_input_names()
    if not names:
        sys.exit("No MIDI input found. Connect the keyboard and try again.")
    port_name = port_name or names[0]
    print(f"  MIDI: {port_name}", file=sys.stderr)
    with mido.open_input(port_name) as port:
        while True:
            for msg in port.iter_pending():
                if msg.type == "note_on" and msg.velocity > 0:
                    yield msg.note, time.perf_counter(), msg.velocity / 127
            time.sleep(0.002)


def main():
    ap = argparse.ArgumentParser(description="Live piano interestingness: one number, 0-1.")
    ap.add_argument("key", nargs="*", help="lock the key, e.g.  C major")
    ap.add_argument("--midi", nargs="?", const="", metavar="PORT",
                    help="read a MIDI keyboard instead of the mic (first port if none given)")
    ap.add_argument("--quiet", action="store_true", help="print only the number, one line per note")
    a = ap.parse_args()
    locked = (NAMES.index(a.key[0].upper()), a.key[1].lower()) if len(a.key) == 2 else None

    model = Model(locked)
    if not a.quiet:
        print("\n  mel      = how unusual the melodic move is (inverse Lerdahl attraction)")
        print("  rhy      = how unusual the rhythm change is (IOI ratio on a 2/3/4 grid)")
        print("  energy   = loudness and notes/sec over the last 3 s")
        print("  key      = boost from a key change, bigger the further it moves on the circle of fifths")
        print("  INTEREST = the one number, 0-1\n")
    notes = midi_notes(a.midi) if a.midi is not None else mic_notes()
    print("  Listening... Ctrl-C to stop.\n", file=sys.stderr)
    scores = []
    try:
        for m, t, loud in notes:
            r = model.note(m, t, loud)
            scores.append(r["raw"])
            if a.quiet:
                print(f"{r['score']:.3f}", flush=True)
            else:
                show(r)
    except KeyboardInterrupt:
        pass
    if scores:
        print(f"\n  Overall interestingness: {np.mean(scores):.2f}  "
              f"({len(scores)} notes, peak {max(scores):.2f})\n", file=sys.stderr)


if __name__ == "__main__":
    main()
