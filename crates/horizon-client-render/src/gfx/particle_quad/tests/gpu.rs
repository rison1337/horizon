use super::*;
use crate::mesher::Vertex;

#[test]
fn gpu_quads_match_cpu_vertices_and_preserve_gaps_after_shrinking() {
    let instance = crate::gfx::test_gpu::instance();
    let Ok(adapter) = pollster::block_on(instance.request_adapter(&Default::default())) else {
        eprintln!("no GPU adapter: skipped");
        return;
    };
    eprintln!("particle test GPU: {:?}", adapter.get_info());
    let (device, queue) = pollster::block_on(adapter.request_device(&Default::default())).unwrap();
    let mut gpu = ParticleGpu::create(&device).expect("particle compute pipeline");
    let quads: Vec<_> = (0..67)
        .map(|i| {
            let t = i as f32 * 0.137;
            ParticleQuad {
                center_size: [t - 4.0, t * 0.123, t * -1.79, 0.13 + t * 0.02],
                rotation: glam::Quat::from_euler(glam::EulerRot::YXZ, t, t * -0.31, t * 0.57)
                    .to_array(),
                rect: [0.25, 0.125, 0.375, 0.1875],
                color: 0x80ff9020 + i,
                light: 0x7090,
                layer: if i % 2 == 0 { 3 << 16 } else { 15 },
                dst: 0,
            }
        })
        .collect();
    let vertices = 67 * 8 + 16;
    let out = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: vertices * 44,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    for count in [67, 3] {
        let frame = ParticleFrame {
            at: 0,
            quads: quads[..count].to_vec(),
        };
        let destinations: Vec<_> = (0..count as u32).map(|i| 2 + i * 8).collect();
        let sentinel = vec![0xabu8; out.size() as usize];
        queue.write_buffer(&out, 0, &sentinel);
        let mut uploads = Uploads::new(&device);
        let mut encoder = device.create_command_encoder(&Default::default());
        gpu.dispatch(
            &device,
            &mut uploads,
            &mut encoder,
            &frame,
            &destinations,
            5,
            &out,
        );
        let read = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: out.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&out, 0, &read, 0, out.size());
        uploads.finish(&encoder);
        queue.submit([encoder.finish()]);
        read.slice(..)
            .map_async(wgpu::MapMode::Read, |r| r.unwrap());
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let data = read.slice(..).get_mapped_range().unwrap();
        let mut expected = sentinel;
        for (q, dst) in frame.quads.iter().zip(destinations) {
            let rotation = glam::Quat::from_array(q.rotation);
            let center = glam::Vec3::from_slice(&q.center_size);
            for (n, (x, y)) in [(1.0, -1.0), (1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0)]
                .into_iter()
                .enumerate()
            {
                let rect = q.rect;
                let vertex = Vertex {
                    pos: (center + rotation * glam::Vec3::new(x, y, 0.0) * q.center_size[3])
                        .to_array(),
                    uv: [
                        if x > 0.0 { rect[2] } else { rect[0] },
                        if y > 0.0 { rect[1] } else { rect[3] },
                    ],
                    layer: q.layer,
                    color: q.color,
                    light: q.light,
                    normal: [0; 4],
                    glint_uv: [0.0; 2],
                };
                let at = (5 + dst as usize + n) * 44;
                expected[at..at + 44].copy_from_slice(bytemuck::bytes_of(&vertex));
            }
        }
        assert_eq!(
            &data[..],
            &expected[..],
            "GPU particle output differs from CPU; count={count}"
        );
    }
}
