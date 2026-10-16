"""The demo video: about a minute of generated footage with the kinds of edits SceneCut deals with.

    python dev/demo_footage.py [out.mp4]

Every frame comes from ffmpeg's own generators (fractals, cellular automata, gradients, plasma),
so the video can be published with the demo page. It has hard cuts, a dissolve, a wipe, a fade
through black, a jump cut, a burst of short shots and a camera flash inside a shot (not a cut),
with a synth pad that changes chord at every shot.
Needs the ffmpeg binary.
"""
import shutil
import subprocess
import sys
from pathlib import Path

W, H, FPS = 1280, 720, 24
FONT = "/usr/share/fonts/truetype/ibm-plex/IBMPlexSans-SemiBold.ttf"
SIZE = f"{W}x{H}"

# (name, seconds, ffmpeg source graph producing W x H at FPS)
SHOTS = [
    ("title", 3.0, f"color=c=0x0d1117:s={SIZE}:r={FPS},drawtext=fontfile={FONT}:text='SceneCut':fontsize=110:"
                   f"fontcolor=0xF0F6FC:x=(w-tw)/2:y=(h-th)/2-40,drawtext=fontfile={FONT}:"
                   f"text='a demo made of generated footage':fontsize=34:fontcolor=0x3ECF6E:x=(w-tw)/2:y=(h/2)+60"),
    ("mandel", 4.0, f"mandelbrot=s={SIZE}:r={FPS}:start_x=-0.743643887037151:start_y=0.13182590420533:"
                    f"start_scale=0.05:end_scale=0.005:end_pts=96:outer=normalized_iteration_count"),
    ("carpet", 3.0, f"sierpinski=s={SIZE}:r={FPS}:type=carpet:jump=2,hue=h=40*t:s=1.6"),
    ("grad", 3.5, f"gradients=s={SIZE}:r={FPS}:c0=0x0b3d91:c1=0x3ecf6e:c2=0xffa931:c3=0x7b2ff7:nb_colors=4:speed=0.03:"
                  f"x0=0:y0=0:x1={W}:y1={H}"),
    ("life", 4.0, f"life=s=320x180:r={FPS}:mold=10:ratio=0.12:death_color=0x10141a:life_color=0x3ecf6e:"
                  f"mold_color=0x1f6feb,scale={W}:{H}:flags=neighbor"),
    ("plasma", 4.0, f"nullsrc=s=640x360:r={FPS},geq=r='128+110*sin(X/37+T*1.7)*cos(Y/53-T)':"
                    f"g='128+110*sin((X+Y)/61+T*1.1)':b='128+110*cos(hypot(X-320,Y-180)/29-T*2.3)',"
                    f"scale={W}:{H},drawbox=c=white@1:t=fill:enable='between(t,1.75,1.83)'"),
    ("rule30", 3.0, f"cellauto=s=640x360:r={FPS}:rule=110:scroll=1:start_full=1:full=1,"
                    f"lutrgb=r='val*0.2':g='val*0.85':b='val',scale={W}:{H}:flags=neighbor"),
    ("waves", 3.5, f"nullsrc=s=640x360:r={FPS},geq=r='200+55*sin((X-2*Y)/23+T*4)':g='110+90*sin((X+Y)/41-T*2)':"
                   f"b='40+40*cos(Y/17+T*3)',scale={W}:{H}"),
    ("mandel2a", 2.5, f"mandelbrot=s={SIZE}:r={FPS}:start_x=-0.16:start_y=1.0405:start_scale=0.02:"
                      f"end_scale=0.0004:end_pts=240:outer=normalized_iteration_count,hue=h=200"),
    ("mandel2b", 2.5, f"mandelbrot=s={SIZE}:r={FPS}:start_x=-0.16:start_y=1.0405:start_scale=0.02:"
                      f"end_scale=0.0004:end_pts=240:outer=normalized_iteration_count,hue=h=200,trim=start=5,"
                      f"setpts=PTS-STARTPTS"),
    ("burst1", 0.6, f"nullsrc=s=640x360:r={FPS},geq=r='255*lt(mod(X+T*400,80),40)':g='60':b='255*lt(mod(Y,60),30)',scale={W}:{H}"),
    ("burst2", 0.6, f"sierpinski=s={SIZE}:r={FPS}:type=triangle:jump=3,hue=h=300:s=2"),
    ("burst3", 0.6, f"life=s=160x90:r={FPS}:ratio=0.3:life_color=0xffa931:death_color=0x2b1a0a,scale={W}:{H}:flags=neighbor"),
    ("burst4", 0.6, f"zoneplate=s={SIZE}:r={FPS}:kx2=60:ky2=420:kt=40"),
    ("spiral", 4.0, f"nullsrc=s=640x360:r={FPS},geq=r='128+120*sin(atan2(Y-180,X-320)*5+hypot(X-320,Y-180)/18-T*3)':"
                    f"g='70+60*sin(hypot(X-320,Y-180)/25-T*2)':b='150+100*cos(atan2(Y-180,X-320)*3+T)',scale={W}:{H}"),
    ("grad2", 3.0, f"gradients=s={SIZE}:r={FPS}:c0=0x3d0b4f:c1=0xf72585:c2=0x4cc9f0:nb_colors=3:speed=0.05:type=radial:"
                   f"x0={W // 2}:y0={H // 2}:x1={W}:y1={H}"),
    ("end", 3.5, f"color=c=0x0d1117:s={SIZE}:r={FPS},drawtext=fontfile={FONT}:text='github.com/PedroMarinhoDev/ComfyUI-SceneCut':"
                 f"fontsize=40:fontcolor=0xF0F6FC:x=(w-tw)/2:y=(h-th)/2"),
]
# transitions into the shot at that index: (kind, seconds); the rest are hard cuts
TRANSITIONS = {4: ("dissolve", 1.0), 8: ("wipeleft", 0.75), 16: ("fadeblack", 1.0)}
CHORDS = [(220.0, 277.18, 329.63), (196.0, 246.94, 293.66), (174.61, 220.0, 261.63), (164.81, 207.65, 246.94),
          (146.83, 185.0, 220.0), (130.81, 164.81, 196.0)]


def build(out: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        sys.exit("ffmpeg is needed")
    parts, inputs = [], []
    for i, (name, dur, graph) in enumerate(SHOTS):
        inputs += ["-f", "lavfi", "-t", f"{dur + 2:.3f}", "-i", graph]
        parts.append(f"[{i}:v]trim=duration={dur},setpts=PTS-STARTPTS,format=yuv420p,fps={FPS},settb=1/{FPS}[v{i}]")
    # chain shots, xfade where a transition is listed
    cur, length = "v0", SHOTS[0][1]
    for i in range(1, len(SHOTS)):
        dur = SHOTS[i][1]
        if i in TRANSITIONS:
            kind, td = TRANSITIONS[i]
            parts.append(f"[{cur}][v{i}]xfade=transition={kind}:duration={td}:offset={length - td:.3f},settb=1/{FPS}[c{i}]")
            length += dur - td
        else:
            parts.append(f"[{cur}][v{i}]concat=n=2:v=1:a=0,settb=1/{FPS}[c{i}]")
            length += dur
        cur = f"c{i}"
    # audio: one chord per shot, with the same overlaps as the video
    a_inputs = len(SHOTS)
    for i, (_, dur, _) in enumerate(SHOTS):
        f1, f2, f3 = CHORDS[i % len(CHORDS)]
        expr = (f"0.09*sin(2*PI*{f1}*t)+0.07*sin(2*PI*{f2}*t)+0.06*sin(2*PI*{f3}*t)"
                f"+0.03*sin(2*PI*{f1 / 2}*t)")
        inputs += ["-f", "lavfi", "-i", f"aevalsrc=exprs='{expr}|{expr}':c=stereo:s=48000:d={dur:.3f}"]
        fade = min(0.05, dur / 4)
        parts.append(f"[{a_inputs + i}:a]aformat=sample_fmts=fltp:channel_layouts=stereo,afade=t=in:d={fade},afade=t=out:st={dur - fade:.3f}:d={fade}[a{i}]")
    acur = "a0"
    for i in range(1, len(SHOTS)):
        if i in TRANSITIONS:
            td = TRANSITIONS[i][1]
            parts.append(f"[{acur}][a{i}]acrossfade=d={td}[ac{i}]")
        else:
            parts.append(f"[{acur}][a{i}]concat=n=2:v=0:a=1[ac{i}]")
        acur = f"ac{i}"
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg, "-v", "error", "-y", *inputs, "-filter_complex", ";".join(parts), "-map", f"[{cur}]",
           "-map", f"[{acur}]", "-c:v", "libx264", "-preset", "slow", "-crf", "20", "-g", "48", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(out)]
    run = subprocess.run(cmd, capture_output=True, text=True)
    if run.returncode:
        sys.exit("ffmpeg failed:\n" + run.stderr.strip()[-3000:])
    print(f"{out}: {length:.2f} s, {len(SHOTS)} shots, transitions at {sorted(TRANSITIONS)}")


if __name__ == "__main__":
    build(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "demo" / "scenecut-demo.mp4")
