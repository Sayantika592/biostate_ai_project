# Song to Dance

Upload a song and get a generated 3D dance: a skeleton dances in the browser, synced to the music, and you can download the motion as BVH (for Blender, Unity, Maya) or JSON.

Generation uses the pretrained [EDGE](https://github.com/Stanford-TML/EDGE) model (Tseng et al., CVPR 2023). EDGE is a diffusion transformer that outputs SMPL motion, conditioned on Jukebox audio features, and it was trained on AIST++. No training is needed here.

## Run without a GPU (demo mode)

Demo mode replaces the model with procedural, beat-paced motion so the app and the exports can be worked on anywhere. It needs `ffmpeg` on the PATH.

```bash
pip install -r server/requirements.txt
cd server && MODE=mock uvicorn app:app --port 8000
# open http://localhost:8000
```

## Run with EDGE (GPU)

```bash
docker build -t song-to-dance .
docker run --gpus all -p 8000:8000 -v $PWD/cache:/cache -v $PWD/data:/data song-to-dance
```

On first use, the Jukebox weights (about 10 GB) download into `cache/`. Jukebox feature extraction is the slow, memory-heavy step, so plan for a 24 GB GPU such as an A10G, L4, 3090/4090 or A100.

## How it works

- `POST /api/generate` takes a song, a start time and a length (5 to 30 s, in steps of 5). The server trims the clip with ffmpeg and runs EDGE's `test.py --save_motions --no_render` on it. It then converts EDGE's pickle (`smpl_poses`, `smpl_trans`, `full_pose`) into viewer JSON and BVH.
- `server/motion.py` rotates EDGE's z-up output to y-up and puts the dancer on the floor. For BVH, it recovers bone offsets from the data, so no SMPL model file is needed.
- `web/index.html` is a single-file three.js viewer. three.js is vendored under `web/vendor`.

## Licensing

EDGE's code is MIT-licensed, but its checkpoint was trained on AIST++, and the SMPL body model is licensed for non-commercial use. This setup is fine for research, portfolios and demos. A commercial product needs licenses for the SMPL model and the training data.

## Roadmap

1. Verify EDGE end to end on a GPU.
2. Skinned avatar (SMPL mesh or a retargeted rigged character) in place of the skeleton.
3. Longer songs via EDGE's chunked long-form generation, with per-section re-rolls.
4. Practice mode: compare the user's webcam pose (MediaPipe) with the generated dance and score it.
