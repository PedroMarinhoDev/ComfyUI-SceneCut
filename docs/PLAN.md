# ComfyUI-SceneCut — plan

Shot/scene cut detection for ComfyUI: find every cut in a video, verify it, look at the result,
and hand a clean scene list to whatever comes next (inpainting, transition removal, audio
extraction, per-shot generation, datasets...).

This pack is written from scratch. The ideas come from two MIT projects, credited in the README:

- **SceneCutGUI** (enoky): a headless pipeline of OmniShotCut detection, minimum-length merge,
  pixel-difference snapping, DINOv3/SSCD false-positive validation, frame-exact ffmpeg splitting.
- **OmniShotCut** (UVA Computer Vision Lab): the shot-query transformer and its checkpoint.

No code from either is imported or vendored. The OmniShotCut checkpoint is loaded into our own
module; its layer names are the only thing we match.

It is published for the community, so: no machine-specific paths, models download on first use,
presets are expressed in seconds (fps-independent), and every node degrades gracefully when an
optional piece (ffmpeg binary, SSCD, NVENC) is missing.

## Principles

1. **Frame-exact.** Frame *n* is the *n*-th decoded frame of the stream, always. No
   `CAP_PROP_POS_FRAMES` seeking (not accurate on long-GOP codecs). Every stage reads frames
   through one sequential decode layer.
2. **Nothing is thrown away.** A rejected cut stays in the scene list with the stage that
   rejected it and why. Transitions stay as labelled segments; whether to absorb them, keep them,
   or cut them out is a choice made downstream.
3. **Simple and advanced.** One simple node with presets does the whole job. The advanced tier is
   the same pipeline split into stage nodes, each exposing every parameter. Both call the same
   functions; a preset is a named parameter set.
4. **Stages are cached.** ComfyUI caches node outputs, so changing a validation threshold does not
   rerun OmniShotCut. Validator embeddings are cached per video so threshold changes are instant.
5. **Visual.** Every result can be inspected: an interactive timeline in the viewer node, a
   contact-sheet image, and a standalone HTML report.

## Data model: `SCENECUT_SCENES`

One JSON-serialisable object flows between nodes.

```
source      path, fps (exact fraction), frame_count, width, height, has_audio, duration
segments    [ {index, start, end (frame, end exclusive), kind: shot|transition,
               transition: Dissolve|Wipe|Push|Slide|Zoom|Fade|Doorway|null,
               relation: new_start|hard_cut|transition_source|transition|sudden_jump,
               confidence} ]
cuts        every boundary ever proposed:
            {frame, status: kept|rejected, stage, reason, detector confidence,
             snap shift, validator features}
history     per-stage counts and parameters (what each stage did)
selection   optional subset of segment indices (shot picking, later)
```

`segments` is always gapless and covers `[0, frame_count)`.

## Nodes

Simple tier:

| Node | Does |
|---|---|
| SceneCut Load Video (Path) | any path -> native `VIDEO` (stock Load Video covers `input/`) |
| SceneCut Detect | preset + validate on/off -> `SCENECUT_SCENES`; models auto-loaded |
| SceneCut Contact Sheet | pages of scene thumbnails, colour-coded by cut type (done) |
| SceneCut Viewer | review: timeline, cut fixes, scene selection, save buttons (phases 4, 4b) |
| SceneCut Selected Scenes | plays the selection scene by scene; own browser tab, kept in sync (4b) |

Advanced tier:

| Node | Does |
|---|---|
| SceneCut Detector Loader | OmniShotCut, device |
| SceneCut Validator Loader | DINOv3 (+ SSCD), device |
| SceneCut Detect (Advanced) | window overlap, confidence floor, transition handling, batch |
| SceneCut Refine | minimum shot length, snap radius, ultra-short merge |
| SceneCut Validate (Advanced) | window, flash sensitivity, threshold mode, feature weights, SSCD |

Output / bridge (phase 5): Export (CSV, EDL, Resolve `.sc`), Load Scene (frames + audio of one
shot). Done: split to files (5a), Save/Load scenes, shot selection and IMAGE outputs/inputs (4b).

## Presets

Grouped by how hard the material is to cut, not by user. Values in seconds so they hold at any
frame rate. Calibrated on test material where we have it (film, music video); others start as
reasoned defaults and get tuned on sample clips before release.

| Preset | Character | Main knobs |
|---|---|---|
| Film / TV | measured pacing, dissolves, few flashes | min shot 0.40 s, normal flash sensitivity |
| Music video / fast edit | cuts on the beat, strobes, flashes | min shot 0.15 s, high flash sensitivity |
| Anime / animation | held frames, impact flashes, smear frames, fades | min shot 0.25 s, high flash sensitivity, stricter validation |
| Vlog / talking head | jump cuts with the same background | min shot 0.50 s, keep sudden jumps |
| Sports / broadcast | fast pans, graphic wipes, replays | min shot 0.30 s, motion-tolerant validation, keep transitions |

## Phases

1. **Scaffold, decode, OmniShotCut.** Package skeleton, model folder + auto-download, frame-exact
   decode layer (ffmpeg binary, PyAV fallback), OmniShotCut reimplementation, windowed inference.
   Done when raw boundaries match the reference dumps on the test videos.
2. **Scene list + simple path.** Data model, Refine stage, simple Detect node with presets,
   Load Video, JSON save/load, contact-sheet image. Usable end to end without validation.
3. **Validator.** DINOv3 through ComfyUI's native encoder, SSCD, flash detection, adaptive
   threshold, our own decision logic. Checked boundary-by-boundary against the per-cut features
   stored in the old runs. Advanced nodes.
4. **Viewer.** Frontend timeline widget (thumbnails, cut status colours, transition segments,
   hover details, click-to-play via a low-res proxy), HTML report.
5. **Export and bridge.** Export formats, split to files, load scene frames/audio, shot selection.

## Status

**Phases 1 and 2 done** (2026-10-09).

- OmniShotCut reimplementation: same boundaries and labels as the original on all three test
  videos (52 / 160 / 258 ranges); window logits within 2e-3 of the original in fp32.
- Stage counts (transitions merged, minimum length) identical to the old pipeline.
- Snapping is frame-exact: of 14 final cuts where we differ from the old pipeline, 12 are on the
  frame with the real pixel jump in ours and 1-2 frames off in the old one (its snap seeked by
  frame number); the other 2 are a gradual change and a strobe.
- Detection: 1.9 s / 3.1 s / 6.8 s on the 4090 for 2 / 4 / 6.5 min videos, no temp files.

Found and fixed on the way:

- cuDNN runs fp32 convolutions in TF32 by default, which moves OmniShotCut's logits by up to 0.6
  and varies per GPU. Inference runs with TF32 off (`strict_fp32`).
- OmniShotCut labels the first shot of every window `new_start`; a cut's relation (hard cut,
  jump cut, transition) is taken from the window that saw the cut in its trusted region.
- Snapping moved cuts in flat footage (all candidates tie); it now moves only on evidence.
- The path loader shows no misleading input-folder preview: it is path-only, and the stock Load
  Video covers files in `input/`.

**Phase 3 (validation)** — done; checked on CPU (GPUs were busy).

- DINOv3 runs through ComfyUI's own ViT-L/16 module (no transformers dependency), in fp32 by
  default: it matches Hugging Face's model to cosine 0.999998. In fp16 it returns NaN, which is
  what ComfyUI's stock CLIP-vision loader picks on many GPUs, so SceneCut loads it itself
  (fp32 or bf16).
- One decode pass at 288x288 (SSCD's size; downscaled on the GPU for DINOv3); per-cut features
  are computed as soon as the decoder passes a cut and frames are dropped, so memory is flat.
- Short film, judged on the old pipeline's exact cut list: 51/51 decisions agree, same threshold,
  49 kept in both; feature differences median ~0.01 (our downscaling and frame-exact reads).
- Weights: DINOv3 from Meta's gated repo, else a byte-identical mirror (same SHA-256); SSCD from
  Meta's URL (MIT). Both checksum-verified.

- Music video (music-video preset): 143/145 decisions agree, same threshold, 139 kept vs 137.
  Both differences are flash calls that move with frame-exact reads:
  - 5066: the old pipeline merged the only boundary between a face shot and a jet scene as a
    flash, losing a real scene change; we keep it.
  - 1336: a camera move into black; the old pipeline (frames read one off) saw a dark flash
    touching the cut and merged; with exact frames the dip starts two frames after the cut and
    we keep it.

- ai-fantasi (film preset): 230/232 decisions agree, same threshold, 222 kept vs 224. Both
  differences are credit cards fading in/out (9165, 9262): identical frames either side, the old
  pipeline kept them as cuts, we merge them (correctly).

Flash rules to revisit when tuning on anime / music material (needs more labelled examples):

- The flash signal is max(mean luma, 99.5th-percentile value); the percentile saturates at 255
  on any frame with highlights, so alternating inverted "impact" frames make the *normal* frames
  look like a dark flash, and the flash correction lands one frame early (5067 instead of 5068).
- A flash must touch the cut (within one frame); dips that start two frames later are missed.

To do when a GPU is free: comparisons on GPU, timing, bf16 check.

**Phase 4 (viewer)** — done (CPU).

- SceneCut Viewer: H.264 proxy (all source frames kept, keyframe every 12, cached in temp/),
  thumbnail sprites decoded from the proxy, one JSON for the viewer.
- web/viewer_core.js is shared by the node widget and the standalone HTML report.
- Checked in the UI: playback, frame-exact seeking (frame 210 -> 7.017 s at 30 fps), scene
  selection, cut jumps, removed-cut markers with validator scores, full-window mode, reload of
  a saved workflow, the report served from its own folder.

**Phase 4b (review)** — done (CPU).

- The viewer is the review step: nothing is saved until the user presses Save. Edits (cut moves)
  and the selection live in a hidden `review` widget of the viewer node, so they are saved with
  the workflow and applied to the node's output; a review made on a different scene list (video
  or settings changed upstream) is not applied.
- Fix cut: frames around the cut nearest the playhead are compared (brightness-normalised colour
  layout) with frames inside the scenes on both sides; the cut goes where they switch
  (`scenecut/boundary.py`). Reads by seeking with timestamp-checked frame numbers (falls back to
  the sequential decode), about 0.2 s on the proxy, 0.4 s on a 2560x1080 source.
- Flash rule fixed. A kept cut beside a flash used to move to the edge of the flash *padded by
  one frame*, which pulled the last frame of the previous shot into the next one (ai-fantasi:
  4 of 4 flash moves were wrong, #26 by one frame, #103 by two). The cut now goes between the
  frames that look like each side (DINOv3), staying on the detector's frame unless the evidence
  is clear; the four cases now land on the right frame.
- Selected Scenes node: follows the upstream viewer live (frontend event), plays scene-bounded
  loops on the shared proxy, and syncs with its standalone page (`/scenecut/selected`) over a
  BroadcastChannel.
- Every scene-list output has an `images` output (`output_images`: all frames / first / middle
  frame per scene, from the selection; memory checked before decoding). Viewers and the contact
  sheet accept `images` (+ fps) as one scene: written to a lossless RGB H.264 temp file, so the
  round trip is bit-exact.
- Save Scenes and reports default to `output/scenecut/<video name>/`.
- Load Video previews the file (served with HTTP range; a proxy is made when the browser cannot
  play the codec), under the same folder policy as the picker.

**Headless** — done (2026-10-13).

- `scenecut/headless.py`: models from plain files and the Detect node's pipeline as one call, for
  scripts and other pipelines. DINOv3 goes into Hugging Face's `DINOv3ViTModel` (transformers 5
  nests the blocks under `model.`; the file keeps the original key names, mapped on load).
- `pipeline.run_preset` now validates (it raised `NotImplementedError` since phase 2).
- Short film, film preset, RTX 4090: 49/49 cuts identical to the node run, same threshold (0.80),
  validator scores within 0.0006. 20.8 s end to end for 125 s of video.
- Fixed on the way: `media.read_range` called `int()` on PyAV's rational type, which newer PyAV
  (19) does not allow, so **Fix cut** in the viewer raised a `TypeError`; rates are now converted
  to `Fraction` first.

**Pre-release check** (2026-10-16, RTX 4090, frontend 1.55.17, PyAV 19).

- Test suite: 48 passed, none skipped (parity with the original OmniShotCut, headless, nodes) in 53 s.
- Basic workflow end to end: ai-fantasi (6.5 min, 2560x1080) 63 s, short film (2 min) 17 s, proxy
  included. Scene #26 now starts on frame 901 straight from Detect (the flash fix).
- Review features checked in the UI in both node modes (classic and Nodes 2.0): fix cut, join to
  nearest cut, selection (Ctrl/Shift+click), Save selected, Selected Scenes following the viewer.
- Fixed: in Nodes 2.0 the frontend forwarded every wheel event over a node to the canvas, so the
  timeline did not zoom and the strip did not scroll; the widgets now carry `data-capture-wheel`
  (the frontend's opt-out, active once the widget is focused by a click).
- Fixed: a reload right after a run showed an empty viewer, because node.properties changes are not
  stored until something else in the workflow changes; the change tracker is now told explicitly.

**Phase 5a (clip export)** — done (2026-10-16).

- `scenecut/export.py`: one file per scene, with its sound, from the viewers' Export button, the
  Selected Scenes node and Save Scenes, into `output/scenecut/<video>/scenes_NNNNN/` with
  `scenes.json` (the scene list plus a clips table). PyAV only: H.264 (default), H.264 NVENC, H.265
  8/10-bit, HEVC NVENC, ProRes 422 HQ, FFV1.
- Frame-exact without decoding the film per clip: the sorted packet timestamps are the
  presentation times of frames 0, 1, 2...; clips close together share one sequential pass, a clip
  far ahead costs one keyframe seek; a timestamp not in the list restarts the export counting
  frames from the start.
- Sound is sliced at the exact times of a clip's first frame and of the next scene's first frame.
- One video encoder at a time: each clip's video is encoded to a temporary file and closed at its
  last frame, its sound is buffered, then both are muxed (video copied). Holding several clips
  open while their sound caught up made NVENC fail (sessions and memory) and used memory with x264.
- Checked: short film (50 clips) and ai-fantasi (223 clips, 2560x1080): every clip has exactly its
  scene's frame count, total sound equals the video's length. 29 s and 139 s with x264 on an
  i9-14900K; 24 s with NVENC for the short film (decode-bound).
- Export jobs run on a thread, polled by the dialog (progress, cancel, keep running in the
  background, a toast at the end).

## Testing

- `dev/ref_oracle.py` produces reference outputs from the old headless pipeline (run with that
  project's own environment; it is a test oracle only, never imported by the pack).
- `tests/` compares our outputs against those references and skips when the test videos are not
  present (they are not redistributable).
- Test videos: `short-film.mp4` (30 fps, 50 shots after validation) and `music-video.mp4`
  (29.97 fps, 125 shots).
