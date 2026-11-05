# Experimental GPU particle expansion

`HORIZON_GPU_PARTICLES=1` uploads 64-byte quad records and expands each into four
44-byte vertices on the GPU. The vertices occupy reserved positions in the existing
entity buffer, preserving material order and the existing draw passes. Simulation,
RNG, visibility decisions, lighting, HDR tiers and particle counts are unchanged.

The flag is off by default. Frame generation, shader packs, native render observers
requiring CPU entity spans, and devices without the required compute support retain
CPU expansion. This experiment targets particle-heavy CPU work, not general FPS.

## Measurements

Ryzen 5 7500F, RTX 4060, NVIDIA 615.78.08, Linux x86-64/Vulkan, Cargo `fast` profile,
1280×720 offscreen, render distance 4, VSync off, uncapped FPS. Both paths use the
same executable, built from `5b4823c19822088fed8b037c25d9b1ab4222bbb5` plus this
implementation; the baseline is its retained CPU path. These measurements precede
the integration of newer main commits. Only `HORIZON_GPU_PARTICLES` changes.

| Scene | CPU FPS | GPU FPS | Difference | CPU time, CPU → GPU |
| --- | ---: | ---: | ---: | ---: |
| 2,048 frozen sparks | 2,993 | 3,828 | +27.9% | 0.260 → 0.207 ms |
| 12,288 frozen sparks | 847 | 1,138 | +34.4% | 1.105 → 0.818 ms |
| Active five-color fireworks, trails and twinkle | 890 | 1,086 | +22.0% | 0.840 → 0.677 ms |
| Empty, ordinary transparency | 5,105 | 5,043 | −1.2% | 0.084 → 0.082 ms |

Each value is the median of three samples' averages: three seconds per sample for
the frozen scenes, eight seconds for active fireworks. Earlier CPU/GPU/GPU/CPU
runs with six five-second samples per path also improved particle-heavy throughput
(+14.4% at 2,048 sparks, +35.5% at 12,288). The last refinement avoids constructing
an occlusion context when particle culling is off. Long-tail spikes remain; there
is no claim of a stutter fix or an equivalent gain in normal interactive gameplay.

Empty-scene improved-transparency results varied: one final pair was 5,187 → 4,040
FPS (−22.1%), while an immediate repeat was 5,120 → 5,210 (+1.8%). Both used three
five-second samples per path. The cause is not established. Measure other GPUs and
workloads before considering a default-on change; only NVIDIA/Vulkan was exercised.

## Reproducing the scenes

Build the optimized client and control tool with `cargo build --profile fast
-p horizon-client -p horizon-client-lib --bin horizon-client --bin horizon-ctl`.
Follow [the isolated-run requirements](../AGENTS.md#rules-that-bite): use fresh game,
home, launcher and XDG directories for each run, with audio and self-update disabled.
Use prepared resources from your own Minecraft installation, never committed data.

Launch with `--offscreen --no-pause --width 1280 --height 720 --render-distance 4
--creative --test-terrain --seed 7 --world Particles --control-socket <fresh-socket>`
and explicit `--game-dir` and `--resources`. Use `HORIZON_STILL=1` for the frozen
scenes, `enableVsync:false`, `maxFps:260` (unlimited), `particles:0`, and
`improvedTransparency:false` in the fresh game's options. Linux runs used hidden
X11 windows because the tiling Wayland compositor resized the requested surface.

Through [horizon-ctl](horizon-ctl-commands.md), teleport to `8 80 -12`, look at yaw
0/pitch 0, set noon and clear weather, hide the HUD, then allow three seconds to
settle. Measure the empty scene. Spawn `/particle minecraft:firework 8 82 0 4 3 4
0.01 2048 force @s`, step one tick, wait two seconds, and measure. Add another
10,240 particles and repeat for 12,288. Before each set, use `bench 2` to warm up,
then three `bench 3` samples. Capture screenshots at the same frozen tick.

For active fireworks, switch to spectator, keep the camera at `8 80 -12`, look at
yaw 0/pitch −25, unfreeze ticks, and summon a stationary rocket at `8 82 0` every
0.2 seconds. Its NBT is:

```snbt
{Life:0,LifeTime:10,ShotAtAngle:1b,Motion:[0d,0d,0d],FireworksItem:{id:"minecraft:firework_rocket",count:1,components:{"minecraft:fireworks":{flight_duration:2,explosions:[{shape:"large_ball",colors:[I;16711680,16776960,255,65280,16711935],has_trail:true,has_twinkle:true}]}}}}
```

Warm up for 12 seconds and verify explosions in `status` before measuring three
`bench 8` samples. Both recorded paths reached cumulative flash counts 88, 120 and
152 at the sample ends. A zero-lifetime rocket pilot produced no explosions and
was excluded, as were pilots with resized surfaces or inactive emitters.

## Correctness checks

Five final 1280×720 screenshot pairs had zero changed pixels: empty, both spark
counts, sparks with a named cow, and sparks with that cow plus a diamond sword in
hand and the HUD. Three earlier improved-transparency pairs also matched exactly.
No images or prepared game resources are included in the repository.

The GPU readback test compares all vertex bytes with the CPU reference, including
scattered destinations, untouched sentinel gaps and a shrinking batch. Extraction
tests compare visibility and attributes with culling on/off, translucent/opaque
particles, roll and two-quad orientations. Batch tests compare original CPU
ordering with GPU gaps for every insertion point, transparency on/off, additive
geometry, hand geometry and see-through tails.

This is a narrow implementation follow-up to #7752, not the complete investigation
of launch rates, viewpoints, distances and vanilla comparison requested there.
