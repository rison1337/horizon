use super::*;

#[test]
fn gpu_extraction_keeps_particle_geometry_attributes_and_visibility() {
    let mut engine = ParticleEngine::new(Path::new(""), &HashMap::new(), &HashSet::new());
    for translucent in [false, true] {
        let name = format!("test:{translucent}");
        engine.atlas.insert(
            name.clone(),
            (
                Rect {
                    u0: 0.125,
                    v0: 0.25,
                    u1: 0.375,
                    v1: 0.5,
                    w: 16,
                    h: 16,
                },
                translucent,
            ),
        );
        for i in 0..12 {
            let mut random = LegacyRandom::new(i);
            let mut p = Particle::moving(
                &mut random,
                ParticleType::Smoke,
                [i as f64, 1.0, -3.0],
                [0.0; 3],
            );
            p.sprite = Sprite::Named(name.clone());
            p.roll = 0.37;
            p.o_roll = 0.21;
            if i == 1 {
                p.behaviour = Behaviour::Shriek { delay: 0 };
            }
            if i == 2 {
                p.no_render = true;
            }
            engine.particles.push(p);
        }
    }
    for culling in [false, true] {
        let hidden = |min: [f64; 3], _: [f64; 3]| min[0] > 5.0;
        let mut expected = Vec::new();
        engine.render_culled(
            DVec3::ZERO,
            27.0,
            -13.0,
            0.6,
            900.0,
            culling,
            None,
            hidden,
            &|_| 0x9a,
            &mut expected,
        );
        let (mut cpu, mut quads) = (Vec::new(), Vec::new());
        engine.render_gpu(
            DVec3::ZERO,
            27.0,
            -13.0,
            0.6,
            900.0,
            culling,
            None,
            hidden,
            &|_| 0x9a,
            &mut cpu,
            &mut quads,
        );
        assert!(cpu.is_empty());
        assert_eq!(expected.len(), quads.len() * 4);
        for (q, wanted) in quads.iter().zip(expected.as_chunks::<4>().0) {
            for (n, (x, y)) in [(1.0, -1.0), (1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0)]
                .into_iter()
                .enumerate()
            {
                let v = wanted[n];
                let pos = FVec3::from_slice(&q.center_size)
                    + Quat::from_array(q.rotation) * FVec3::new(x, y, 0.0) * q.center_size[3];
                assert_eq!(v.pos, pos.to_array());
                assert_eq!(
                    v.uv,
                    [
                        if x > 0.0 { q.rect[2] } else { q.rect[0] },
                        if y > 0.0 { q.rect[1] } else { q.rect[3] }
                    ]
                );
                assert_eq!((v.layer, v.color, v.light), (q.layer, q.color, q.light));
            }
        }
    }
}
