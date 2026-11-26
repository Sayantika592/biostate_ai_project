---
title: Song to Dance
emoji: 💃
colorFrom: red
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: Upload a song and get a 3D dance choreographed to it
---

# Song to Dance

![A generated Break dance](docs/demo.gif)

Upload a song and get a 3D dance choreographed to it. A skeleton dances in the browser in sync with the music, and you can download the motion as BVH (for Blender, Unity, Maya) or JSON. Everything runs on a laptop CPU, with no GPU and no training.

## Quick start

```bash
pip install -r server/requirements.txt      # also needs ffmpeg on PATH
bash scripts/setup_data.sh                   # downloads AIST++ motions, builds data/library.npz (~1 min)
cd server && uvicorn app:app --port 8000     # open http://localhost:8000
```

Or with Docker: `docker build -t song-to-dance . && docker run -p 7860:7860 song-to-dance`, then open http://localhost:7860.

## Deploy (Hugging Face Space)

`.github/workflows/deploy-space.yml` pushes `main` to a free Docker Space. Add a `HF_TOKEN` secret (write token) and a `HF_SPACE` variable (e.g. `your-name/song-to-dance`) under the GitHub repo's Settings > Secrets and variables > Actions, then re-run the workflow.

## How it works (default engine: `stitch`)

This is music-driven motion retrieval and synthesis, a lightweight version of motion-graph systems such as ChoreoMaster. It uses 411 real dance sequences from [AIST++](https://google.github.io/aistplusplus_dataset/) in SMPL format, across 10 genres.

1. **Library** (`server/build_library.py`). Each AIST++ sequence was danced to a track with a known tempo, which is encoded in its music id (e.g. `mPO3` is 110 BPM). A spectral check of the dancers' joint speed agrees with this table. The beat phase of each clip is fitted to the dancer's own "hits", meaning the local minima of joint speed.
2. **Song analysis** (`choreographer.analyse_song`). librosa tracks beats, and RMS loudness plus onset strength give the energy of each beat. Tempos are folded into the 75 to 140 BPM range that the dancers used.
3. **Phrase selection**. Clips are cut into 8-beat, bar-aligned phrases (about 1,000 in total). Each next phrase is sampled from the best 8 by a cost made of four parts:
   - how smoothly its start pose joins the previous phrase's end pose (joint rotation distance),
   - how well its movement energy fits the music's energy (percentile match),
   - how close its tempo is to the song's,
   - a penalty for repeating a recent clip.

   The random seed makes every Generate a new take.
4. **Synthesis**. Each phrase is time-warped beat by beat onto the song's tracked beat times, so moves land on the beat even when the tempo drifts. The joins are cross-faded with quaternion blending, the feet are kept on the floor, and the dancer's travel continues from the previous phrase.

On a synthetic 112 BPM test track, the AIST++-style beat-alignment score (music beats against motion hits, sigma 0.1 s) is 0.65 to 0.8 for House, Break, Krump and LA Hip-hop. It is about 0.45 for Pop and Street Jazz, whose hits aren't always speed minima. An unaligned raw clip scores 0.53. Generating a 30 s dance takes a few seconds on CPU.

Other engines (`MODE` env var):
- `edge`: pretrained [EDGE](https://github.com/Stanford-TML/EDGE) diffusion model conditioned on Jukebox features, built with `Dockerfile.edge`. It needs a GPU with about 24 GB and is untested.
- `mock`: procedural motion, for UI work only.

## Files

- `server/choreographer.py`: the beat analysis, phrase selection and stitching.
- `server/build_library.py`: builds the move library. `server/aist_io.py` reads the AIST++ pickles with a numpy-only unpickler.
- `server/motion.py`: forward kinematics on an SMPL-like skeleton, y-up normalisation, and BVH/JSON export.
- `server/app.py`: the FastAPI job API and static web app.
- `web/index.html`: the three.js viewer, with three.js vendored.

## Licensing

The motion comes from AIST++, so check its license terms before any commercial use. No SMPL body model file is shipped: the skeleton uses an approximate SMPL-like joint template. EDGE's checkpoint, which only the optional engine uses, is research-only. This setup is fine for portfolios and demos.

## Roadmap

1. Skinned avatar (a rigged glTF character retargeted from the SMPL joints) in place of the skeleton.
2. Learned music-to-move matching (e.g. a small contrastive model on AIST++ music and motion) to replace the energy and tempo heuristics.
3. Practice mode: compare the user's webcam pose (MediaPipe) with the dance and score it.
4. EDGE on a free Kaggle or Colab GPU as a second engine.
