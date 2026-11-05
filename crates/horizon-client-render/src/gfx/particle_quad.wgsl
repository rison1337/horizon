// Quaternion-vector arithmetic follows glam 0.30.10 (MIT), by the glam contributors,
// at 990737c58c8c6f4315a4ad09806aa55aa141a503; adapted to WGSL with explicit rounding.
//
// Permission is hereby granted, free of charge, to any
// person obtaining a copy of this software and associated
// documentation files (the "Software"), to deal in the
// Software without restriction, including without
// limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software
// is furnished to do so, subject to the following
// conditions:
//
// The above copyright notice and this permission notice
// shall be included in all copies or substantial portions
// of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF
// ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
// TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
// PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT
// SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
// IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//
// 16-word ParticleQuad -> four 11-word mesher::Vertex records.
@group(0) @binding(0) var<storage, read> quads: array<u32>;
@group(0) @binding(1) var<storage, read_write> vertices: array<u32>;

fn f(word: u32) -> f32 { return bitcast<f32>(word); }
var<private> keep: u32;
fn r(v: f32) -> f32 { return bitcast<f32>(bitcast<u32>(v) ^ keep); }
fn r3(v: vec3<f32>) -> vec3<f32> {
    return bitcast<vec3<f32>>(bitcast<vec3<u32>>(v) ^ vec3(keep));
}
fn dot3(a: vec3<f32>, b: vec3<f32>) -> f32 {
    let p = r3(a * b);
    return r(r(p.x + p.y) + p.z);
}

@compute @workgroup_size(64)
fn expand(@builtin(workgroup_id) group: vec3<u32>,
          @builtin(num_workgroups) groups: vec3<u32>,
          @builtin(local_invocation_index) lane: u32) {
    let index = (group.y * groups.x + group.x) * 64u + lane;
    if index >= arrayLength(&quads) / 16u { return; }
    let q = index * 16u;
    let center = vec3(f(quads[q]), f(quads[q+1u]), f(quads[q+2u]));
    let size = f(quads[q+3u]);
    let b = vec3(f(quads[q+4u]), f(quads[q+5u]), f(quads[q+6u]));
    let w = f(quads[q+7u]);
    let rect = vec4(f(quads[q+8u]), f(quads[q+9u]), f(quads[q+10u]), f(quads[q+11u]));
    let color = quads[q+12u];
    let light = quads[q+13u];
    let layer = quads[q+14u];
    let dst = quads[q+15u];
    // Light only uses the low 16 bits. An opaque zero prevents fused arithmetic.
    keep = light & 0x80000000u;
    let corners = array(vec2(1.0,-1.0), vec2(1.0,1.0), vec2(-1.0,1.0), vec2(-1.0,-1.0));
    let uvs = array(rect.zw, rect.zy, rect.xy, rect.xw);
    for (var n = 0u; n < 4u; n++) {
        let v = vec3(corners[n], 0.0);
        let c = r3(r3(b.yzx * v.zxy) - r3(b.zxy * v.yzx));
        let rotated = r3(r3(r3(v * r(r(w*w) - dot3(b,b)))
            + r3(b * r(dot3(v,b) * 2.0))) + r3(c * r(w*2.0)));
        let pos = r3(center + r3(rotated * size));
        let out = (dst+n)*11u;
        vertices[out] = bitcast<u32>(pos.x);
        vertices[out+1u] = bitcast<u32>(pos.y);
        vertices[out+2u] = bitcast<u32>(pos.z);
        vertices[out+3u] = bitcast<u32>(uvs[n].x);
        vertices[out+4u] = bitcast<u32>(uvs[n].y);
        vertices[out+5u] = layer;
        vertices[out+6u] = color;
        vertices[out+7u] = light;
        vertices[out+8u] = 0u;
        vertices[out+9u] = 0u;
        vertices[out+10u] = 0u;
    }
}
