use super::*;

#[test]
fn gpu_particle_gaps_preserve_material_order_and_frame_tails() {
    use super::super::particle_quad::{ParticleFrame, ParticleQuad};
    let quad = |layer, x| {
        [Vertex {
            layer,
            pos: [x, 0.0, 0.0],
            ..bytemuck::Zeroable::zeroed()
        }; 4]
    };
    let entities: Vec<_> = [
        (0, 1.0),
        (15, 2.0),
        (0, 3.0),
        (OIT_ENTITY, 4.0),
        (15, 5.0),
        (32, 6.0),
    ]
    .into_iter()
    .flat_map(|(layer, x)| quad(layer, x))
    .collect();
    let quads: Vec<_> = [(15, 7.0), (0, 8.0), (3 << 16 | 15, 9.0), (0, 10.0)]
        .into_iter()
        .map(|(layer, x)| ParticleQuad {
            center_size: [x, 0.0, 0.0, 1.0],
            layer,
            ..bytemuck::Zeroable::zeroed()
        })
        .collect();
    for oit in [false, true] {
        for at in (0..=entities.len()).step_by(4) {
            let particles = ParticleFrame {
                at,
                quads: quads.clone(),
            };
            let cpu_particles: Vec<_> = quads
                .iter()
                .flat_map(|q| quad(q.layer, q.center_size[0]))
                .collect();
            let mut original = entities.clone();
            original.splice(at..at, cpu_particles);
            let (hand, additive, see) = (quad(0, 11.0), quad(2, 12.0), quad(0, 13.0));
            let mut baseline = EntityBatch::default();
            let counts = baseline.update(&original, &additive, &hand, &see, oit);
            let mut batch = EntityBatch::default();
            assert_eq!(
                batch.update_particles(&entities, &additive, &hand, &see, oit, Some(&particles)),
                counts
            );
            let mut actual = vec![quad(0, 0.0)[0]; batch.virtual_vertices()];
            let (mut start, mut dst) = (0, 0);
            for &(end, gap) in batch
                .gaps
                .iter()
                .chain(std::iter::once(&(batch.combined.len(), 0)))
            {
                actual[dst..dst + end - start].copy_from_slice(&batch.combined[start..end]);
                dst += end - start + gap as usize;
                start = end;
            }
            for (q, &dst) in particles.quads.iter().zip(&batch.particle_destinations) {
                actual[dst as usize..dst as usize + 4]
                    .copy_from_slice(&quad(q.layer, q.center_size[0]));
            }
            assert_eq!(
                bytemuck::cast_slice::<_, u8>(&actual),
                bytemuck::cast_slice::<_, u8>(&baseline.combined)
            );
        }
    }
}
