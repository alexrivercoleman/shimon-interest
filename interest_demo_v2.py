#!/usr/bin/env python3
"""
    pip install numpy sounddevice
    python interest_demo.py              # estimate the key as you play
    python interest_demo.py C major      # or lock it
"""
import sys, math, queue
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
    """Feed it audio blocks; it returns a MIDI note whenever a new note is struck.

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
                return pitch(y)
        return None


class Model:
    def __init__(self, locked_key=None):
        self.locked = locked_key
        self.hist = np.zeros(12)       # recent pitch classes, for the key estimate
        self.mu, self.var = 1.0, 1.0   # running average of log(1/alpha): what "normal" sounds like
        self.prev = None

    def key(self):
        if self.locked:
            return self.locked
        if self.hist.sum() < 4 or self.hist.std() == 0:
            return None
        return max(((np.corrcoef(self.hist, np.roll(p, t))[0, 1], t, mode)
                    for t in range(12) for mode, p in (("major", KK_MAJ), ("minor", KK_MIN))))[1:]

    def note(self, m):
        prev, self.prev = self.prev, m
        key = self.key()
        self.hist *= 0.9
        self.hist[m % 12] += 1
        if prev is None or m == prev:
            return prev, key, None
        s1 = S[key[1]][(prev - key[0]) % 12] if key else 1
        s2 = S[key[1]][(m - key[0]) % 12] if key else 1
        n = m - prev
        inv = (s1 / s2) * n * n                       # inverse Lerdahl attraction
        x = math.log(inv)
        z = (x - self.mu) / max(math.sqrt(self.var), 0.5)   # how unusual vs. recent playing
        u = abs(z) / 1.5
        interest = u * math.exp(1 - u)                # inverted-U: peaks at |z| = 1.5
        d = x - self.mu
        self.mu += 0.2 * d
        self.var = 0.8 * (self.var + 0.2 * d * d)
        return prev, key, (n, s1, s2, inv, z, interest)


def show(m, prev, key, r):
    k = f"{NAMES[key[0]]} {key[1]}" if key else "(finding key)"
    if r is None:
        print(f"  {name(m):>4}" + ("   (repeat)" if prev == m else ""))
        return
    n, s1, s2, inv, z, interest = r
    bar = "#" * round(interest * 20)
    print(f"  {name(prev):>4} -> {name(m):<4} {n:+4d}  s {s1}/{s2}  1/a {inv:7.1f}  "
          f"z {z:+5.2f}  interest {interest:4.2f} {bar:<20}  {k}")


def main():
    import sounddevice as sd
    locked = None
    if len(sys.argv) == 3:
        locked = (NAMES.index(sys.argv[1].upper()), sys.argv[2].lower())
    model, ear, q = Model(locked), Listener(), queue.Queue()
    print("\n  1/a      = inverse Lerdahl attraction, (s1/s2) * n^2")
    print("  z        = how unusual that is compared to the last few notes")
    print("  interest = highest for moderately unusual moves, low for more-of-the-same\n")
    # On Windows the default (MME) mic adds ~180 ms of lag; WASAPI on the same mic is ~20 ms
    device, extra = None, None
    if sys.platform == "win32":
        for api in sd.query_hostapis():
            if api["name"] == "Windows WASAPI" and api["default_input_device"] >= 0:
                device, extra = api["default_input_device"], sd.WasapiSettings(auto_convert=True)
    print(f"  Mic: {sd.query_devices(device, 'input')['name']}")
    print("  Listening... play single notes. Ctrl-C to stop.\n")
    with sd.InputStream(device=device, samplerate=SR, channels=1, blocksize=HOP, latency="low",
                        extra_settings=extra, callback=lambda data, *_: q.put(data[:, 0].copy())):
        try:
            while True:
                m = ear.feed(q.get())
                if m is not None:
                    show(m, *model.note(m))
        except KeyboardInterrupt:
            print()


if __name__ == "__main__":
    main()
