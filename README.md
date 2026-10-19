# Shimon Interest Demo

Listens to live piano (mic or MIDI keyboard) and outputs one **interestingness** number, 0-1, for every note played.

## Run

```
pip install -r requirements.txt
python interest_demo.py                 # mic, key estimated as you play
python interest_demo.py C major         # key locked
python interest_demo.py --midi          # MIDI keyboard (uses real velocities)
python interest_demo.py --quiet         # print only the number, one line per note
```

Ctrl-C stops it and prints an overall score for the take (average and peak).

## How the number is made

Each part is scored 0-1:

| Part | What it measures |
|---|---|
| melody | How surprising each note-to-note move is (inverse Lerdahl attraction, scored on an inverted U vs. recent playing) |
| rhythm | How much the rhythm changes, snapped to 2/3/4 subdivisions; steady or repeating patterns settle toward 0 |
| energy | 0.6 x loudness + 0.4 x notes per second over the last 3 s |
| key | A boost when the key changes, bigger the further it moves on the circle of fifths, fading over the next few notes |

```
interest = (melody + rhythm + energy) / 3, pushed toward 1 by a key change, then smoothed
```

From code: `Model().note(midi, time_s, loudness)["score"]`.

## Files

- `interest_demo.py`: the combined demo
- `interest_demo_v2.py`: the earlier melody-only version

## Credits

Ideas borrowed from:

- [phuong3010/shimon](https://github.com/phuong3010/shimon): note density + velocity intensity
- [nicolasllanosjimenez/timing_analyzer](https://github.com/nicolasllanosjimenez/timing_analyzer): subdivision grid for timing
- [evelyn-wang2/interestingnessKeyChange](https://github.com/evelyn-wang2/interestingnessKeyChange): key changes scored by circle-of-fifths distance
