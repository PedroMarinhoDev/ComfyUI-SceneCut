"""Frame-exact clip export (scenecut.export) on a generated video with sound."""
import os
import shutil
import subprocess

import numpy as np
import pytest

from scenecut import export as E
from scenecut.media import iter_frames_at


@pytest.fixture(scope="module")
def av_video(tmp_path_factory):
    """12 s at 25 fps, B-frames, one keyframe: three unrelated 4-second shots and a 440 Hz tone."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg binary needed to build the test video")
    out = str(tmp_path_factory.mktemp("export") / "three.mp4")
    subprocess.run([ffmpeg, "-v", "error", "-y",
                    "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=25:duration=4",
                    "-f", "lavfi", "-i", "mandelbrot=size=320x240:rate=25",
                    "-f", "lavfi", "-i", "smptehdbars=size=320x240:rate=25:duration=4",
                    "-f", "lavfi", "-t", "12", "-i", "sine=frequency=440:sample_rate=48000",
                    "-filter_complex", "[1:v]trim=duration=4,setpts=PTS-STARTPTS[b];"
                    "[0:v][b][2:v]concat=n=3:v=1:a=0,format=yuv420p[v]",
                    "-map", "[v]", "-map", "3:a", "-c:v", "libx264", "-g", "300", "-bf", "3", "-crf", "18",
                    "-c:a", "aac", "-shortest", out], check=True)
    return out


def _read(path):
    import av

    with av.open(path) as c:
        frames = [f.to_ndarray(format="rgb24", width=64, height=48) for f in c.decode(video=0)]
    with av.open(path) as c:
        samples = sum(f.samples for f in c.decode(audio=0)) if c.streams.audio else 0
    return frames, samples


@pytest.mark.parametrize("seek_gap", [None, 0.5])          # one sequential pass / a seek per clip
def test_clips_are_frame_exact_with_their_sound(av_video, tmp_path, seek_gap):
    ref = dict(iter_frames_at(av_video, [99, 100, 229, 230, 259], 64, 48))
    clips = E.export_clips(av_video, [(1, 100, 200), (2, 230, 260)], str(tmp_path), "h264",
                           seek_gap=seek_gap, scene_total=3)
    assert [c.file for c in clips] == ["three_002.mp4", "three_003.mp4"]
    for cl, first in zip(clips, (100, 230)):
        frames, samples = _read(os.path.join(tmp_path, cl.file))
        assert len(frames) == cl.end - cl.start
        assert samples == pytest.approx((cl.end - cl.start) / 25 * 48000, abs=2)
        f0 = frames[0].astype(np.float32)
        assert np.abs(f0 - ref[first]).mean() < 6                     # the clip starts on its own frame
    prev = _read(os.path.join(tmp_path, clips[0].file))[0][0].astype(np.float32)
    assert np.abs(prev - ref[99]).mean() > 30                        # not on the frame before


def test_counting_fallback_matches(av_video, tmp_path, monkeypatch):
    monkeypatch.setattr(E, "frame_index", lambda path: None)
    clips = E.export_clips(av_video, [(2, 230, 260)], str(tmp_path), "h264", scene_total=3)
    frames, _ = _read(os.path.join(tmp_path, clips[0].file))
    assert len(frames) == 30


def test_frame_index_and_names(av_video):
    idx = E.frame_index(av_video)
    assert len(idx.pts) == 300 and idx.find(int(idx.pts[42])) == 42 and idx.find(-5) is None
    assert E.clip_name("film", 0, 50, ".mp4") == "film_001.mp4"
    assert E.clip_name("film", 1233, 1500, ".mov") == "film_1234.mov"


def test_cancel_removes_partial_files(av_video, tmp_path):
    def abort():
        raise E.ExportCancelled()

    with pytest.raises(E.ExportCancelled):
        E.export_clips(av_video, [(0, 0, 300)], str(tmp_path), "h264", check_abort=abort, scene_total=1)
    assert os.listdir(tmp_path) == []
