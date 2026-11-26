"""Music-to-dance by beat-matched retrieval and stitching of real AIST++ moves.

No neural-net training and no GPU. Pipeline:
  1. Track the song's beats and per-beat energy with librosa.
  2. Cut the AIST++ library into 8-beat phrases (2 bars), each from a real
     dancer whose track tempo is known.
  3. Pick phrases one after another, scoring each by how smoothly it joins
     the previous phrase (pose distance), how well its movement energy fits
     the music's energy, and its tempo match, while avoiding repeats.
  4. Time-warp every phrase beat-by-beat onto the song's actual beat times,
     then cross-fade the joins (quaternion blend) and keep the feet on the floor.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
from scipy.spatial.transform import Rotation as R

import motion

FPS = motion.FPS
PHRASE = 8        # beats per phrase
BLEND = 8         # cross-fade frames (~0.27 s)
GENRES = {"BR": "Break", "PO": "Pop", "LO": "Lock", "MH": "Middle Hip-hop", "LH": "LA Hip-hop",
          "HO": "House", "WA": "Waack", "KR": "Krump", "JS": "Street Jazz", "JB": "Ballet Jazz"}


@dataclass
class Library:
    names: np.ndarray
    genres: np.ndarray
    bpm: np.ndarray
    phase: np.ndarray
    offsets: np.ndarray
    rot: np.ndarray
    trans: np.ndarray
    speed: np.ndarray

    @classmethod
    def load(cls, path: str | Path) -> "Library":
        d = np.load(path)
        return cls(**{k: d[k] for k in cls.__dataclass_fields__})

    def clip(self, i: int):
        a, b = self.offsets[i], self.offsets[i + 1]
        return self.rot[a:b], self.trans[a:b], self.speed[a:b]


def analyse_song(path: str | Path, seconds: float | None = None):
    y, sr = librosa.load(str(path), sr=22050, mono=True, duration=seconds)
    hop = 512
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset, sr=sr, hop_length=hop, units="time")
    tempo = float(np.atleast_1d(tempo)[0])
    beats = np.asarray(beats, dtype=np.float64)
    # Bring the beat grid into the 75-140 BPM range the dancers used.
    if tempo > 140:
        beats, tempo = beats[::2], tempo / 2
    elif tempo < 75 and len(beats) > 1:
        mids = (beats[:-1] + beats[1:]) / 2
        beats, tempo = np.sort(np.concatenate([beats, mids])), tempo * 2
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
    loud = np.interp(beats, times, rms) if len(beats) else np.zeros(0)
    on = np.interp(beats, librosa.frames_to_time(np.arange(len(onset)), sr=sr, hop_length=hop), onset) if len(beats) else np.zeros(0)
    energy = 0.6 * loud / (loud.max() + 1e-8) + 0.4 * on / (on.max() + 1e-8)
    return dict(tempo=tempo, beats=beats, energy=energy, duration=len(y) / sr)


def _extend_beats(beats: np.ndarray, duration: float, tempo: float) -> np.ndarray:
    """Pad the beat grid before the first and after the last tracked beat so the whole song is covered."""
    period = 60.0 / tempo
    if len(beats) < 2:
        beats = np.arange(0, duration + period, period)
    else:
        period = float(np.median(np.diff(beats)))
    pre = np.arange(beats[0] - period, -PHRASE * period, -period)[::-1]
    post = np.arange(beats[-1] + period, duration + (PHRASE + 2) * period, period)
    return np.concatenate([pre, beats, post])


def _pose_feat(rot_frames: np.ndarray) -> np.ndarray:
    """Pose descriptor: local rotation matrices of the body joints (root excluded)."""
    m = R.from_rotvec(rot_frames[..., 1:22, :].reshape(-1, 3)).as_matrix()
    return m.reshape(*rot_frames.shape[:-2], -1)


class Choreographer:
    def __init__(self, library: Library):
        self.lib = library
        self.phrases = self._index_phrases()

    def _index_phrases(self):
        out = []
        lib = self.lib
        for i in range(len(lib.names)):
            rot, trans, speed = lib.clip(i)
            period = FPS * 60.0 / float(lib.bpm[i])
            n = len(rot)
            k = 0
            while True:
                start = lib.phase[i] + k * period
                end = start + PHRASE * period
                if end + BLEND + 2 >= n:
                    break
                if start - 2 >= 0:
                    s, e = int(round(start)), int(round(end))
                    out.append(dict(clip=i, start=start, period=period, genre=str(lib.genres[i]),
                                    bpm=float(lib.bpm[i]), energy=float(speed[s:e].mean()),
                                    f_in=_pose_feat(rot[s]), f_out=_pose_feat(rot[e])))
                k += 4  # bar-aligned starts
        # Energy as a percentile within each genre, so genres compare fairly.
        for g in set(p["genre"] for p in out):
            ps = [p for p in out if p["genre"] == g]
            ranks = np.argsort(np.argsort([p["energy"] for p in ps]))
            for p, r in zip(ps, ranks):
                p["energy_pct"] = r / max(len(ps) - 1, 1)
        return out

    def generate(self, audio: str | Path, seconds: float | None = None, genre: str = "auto", seed: int = 0,
                 energy: str = "match"):
        rng = np.random.default_rng(seed)
        song = analyse_song(audio, seconds)
        dur = song["duration"]
        beats = _extend_beats(song["beats"], dur, song["tempo"])
        n_phr = int(np.ceil((len(beats) - 1) / PHRASE))

        if genre == "auto" or genre not in GENRES:
            genre = str(rng.choice(sorted(GENRES)))
        pool = [p for p in self.phrases if p["genre"] == genre]
        # Song energy per phrase, as a percentile across the song.
        e_beat = np.interp(beats, song["beats"], song["energy"]) if len(song["beats"]) else np.zeros(len(beats))
        e_phr = np.array([e_beat[j * PHRASE:(j + 1) * PHRASE].mean() for j in range(n_phr)])
        e_pct = np.argsort(np.argsort(e_phr)) / max(n_phr - 1, 1)
        # Energy preference shifts which moves count as a fit: calm favours the
        # gentler half of the library, intense the bigger half.
        if energy == "calm":
            e_pct = 0.45 * e_pct
        elif energy == "intense":
            e_pct = 0.55 + 0.45 * e_pct

        chosen, recent = [], []
        for j in range(n_phr):
            b0 = beats[min(j * PHRASE, len(beats) - 1)]
            b1 = beats[min((j + 1) * PHRASE, len(beats) - 1)]
            song_bpm = 60.0 * PHRASE / max(b1 - b0, 1e-3)
            cost = np.array([
                (0 if not chosen else np.linalg.norm(chosen[-1]["f_out"] - p["f_in"]) / 3.0)
                + 1.5 * abs(p["energy_pct"] - e_pct[j])
                + 2.0 * abs(np.log(song_bpm / p["bpm"]))
                + (1.5 if p["clip"] in recent else 0.0)
                for p in pool
            ])
            top = np.argsort(cost)[:8]
            w = np.exp(-(cost[top] - cost[top].min()) / 0.15)
            pick = pool[int(rng.choice(top, p=w / w.sum()))]
            chosen.append(pick)
            recent = (recent + [pick["clip"]])[-3:]

        rot, root = self._assemble(chosen, beats, dur)
        pos, _ = motion.forward_kinematics(rot, root, motion.REST_OFFSETS)
        sample = {"smpl_poses": rot.reshape(len(rot), 72), "smpl_trans": root, "full_pose": pos}
        info = {"genre": GENRES[genre], "tempo": round(song["tempo"], 1), "energy": energy,
                "phrases": [str(self.lib.names[p["clip"]]) for p in chosen]}
        return sample, info

    def _segment(self, p, song_knots: np.ndarray, t_out: np.ndarray):
        """Sample a phrase at output times t_out, warping clip beats onto song beats."""
        rot, trans, _ = self.lib.clip(p["clip"])
        clip_knots = p["start"] + p["period"] * np.arange(len(song_knots))
        # piecewise-linear warp with linear extrapolation at the ends
        f = np.interp(t_out, song_knots, clip_knots)
        lo, hi = t_out < song_knots[0], t_out > song_knots[-1]
        r0 = (clip_knots[1] - clip_knots[0]) / (song_knots[1] - song_knots[0])
        r1 = (clip_knots[-1] - clip_knots[-2]) / (song_knots[-1] - song_knots[-2])
        f[lo] = clip_knots[0] + (t_out[lo] - song_knots[0]) * r0
        f[hi] = clip_knots[-1] + (t_out[hi] - song_knots[-1]) * r1
        f = np.clip(f * 1.0, 0, len(rot) - 1.001)
        i0 = np.floor(f).astype(int); a = (f - i0)[:, None]
        q0 = R.from_rotvec(rot[i0].reshape(-1, 3)).as_quat().reshape(len(f), 24, 4)
        q1 = R.from_rotvec(rot[i0 + 1].reshape(-1, 3)).as_quat().reshape(len(f), 24, 4)
        q = _nlerp(q0, q1, a[..., None])
        tr = trans[i0] * (1 - a) + trans[i0 + 1] * a
        return q, tr.astype(np.float64)

    def _assemble(self, chosen, beats, dur):
        T = int(round(dur * FPS))
        t_all = np.arange(T) / FPS
        out_q = np.zeros((T, 24, 4)); out_tr = np.zeros((T, 3))
        prev_end = None
        for j, p in enumerate(chosen):
            k0 = j * PHRASE
            knots = beats[k0:k0 + PHRASE + 1] * FPS
            if len(knots) < 2:
                break
            a = 0 if j == 0 else int(np.ceil(knots[0]))
            b = T if j == len(chosen) - 1 else min(T, int(np.ceil(knots[-1])))
            if a >= T:
                break
            b_ext = min(T, b + BLEND)
            q, tr = self._segment(p, knots, np.arange(a, b_ext, dtype=np.float64))
            # floor: lift each phrase so its lowest foot point sits at y=0
            pos, _ = motion.forward_kinematics(R.from_quat(q.reshape(-1, 4)).as_rotvec().reshape(-1, 24, 3), tr, motion.REST_OFFSETS)
            tr[:, 1] -= pos[:, [7, 8, 10, 11], 1].min()
            # continue from where the previous phrase left the dancer (xz), drifting back to centre
            if prev_end is not None:
                shift = prev_end - tr[0, [0, 2]]
                tr[:, [0, 2]] += shift - 0.15 * (prev_end)
            else:
                tr[:, [0, 2]] -= tr[0, [0, 2]]
            n_own = b - a
            if j == 0:
                out_q[a:b_ext] = q; out_tr[a:b_ext] = tr
            else:
                # cross-fade over the overlap written by the previous phrase's extension
                nb = min(BLEND, b_ext - a)
                w = _smoothstep(np.linspace(0, 1, nb + 2)[1:-1])[:, None]
                out_q[a:a + nb] = _nlerp(out_q[a:a + nb], q[:nb], w[..., None])
                out_tr[a:a + nb] = out_tr[a:a + nb] * (1 - w) + tr[:nb] * w
                out_q[a + nb:b_ext] = q[nb:]; out_tr[a + nb:b_ext] = tr[nb:]
            prev_end = tr[min(n_own, len(tr)) - 1, [0, 2]].copy()
        rot = R.from_quat(out_q.reshape(-1, 4)).as_rotvec().reshape(T, 24, 3)
        return rot, out_tr


def _nlerp(q0: np.ndarray, q1: np.ndarray, w) -> np.ndarray:
    d = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(d < 0, -q1, q1)
    q = q0 * (1 - w) + q1 * w
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def _smoothstep(x):
    return x * x * (3 - 2 * x)
