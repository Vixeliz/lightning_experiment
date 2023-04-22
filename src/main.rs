use bevy_vinox_pixel::prelude::*;
use std::time::Duration;

use bevy::{
    core_pipeline::{bloom::BloomSettings, clear_color::ClearColorConfig},
    prelude::*,
    utils::FloatOrd,
    window::PrimaryWindow,
};
use bevy_egui::*;
use bevy_prototype_debug_lines::*;
use rand::Rng;
// Using this instead of noise-rs because of speed
use bracket_noise::prelude::*;

// Lightning is a singular bolt
#[derive(Component)]
struct Lightning {
    points: Vec<Vec2>,
    seed: u32,
}

#[derive(Resource, Clone)]
struct LightningSettings {
    noise_frequency: f32,
    noise_amount: f32,
    short_iterations: usize,
    short_threshold: f32,
    time_scale: f32,
    meander_amount: f32,
    close_amount: f32,
    new: bool,
}

impl Default for LightningSettings {
    fn default() -> Self {
        LightningSettings {
            noise_frequency: 10.0,
            noise_amount: 500.0,
            short_iterations: 100,
            short_threshold: 1.1,
            time_scale: 0.1,
            meander_amount: 1.0,
            close_amount: 1.0,
            new: true,
        }
    }
}

impl Lightning {
    fn new(start: Vec2, end: Vec2, seed: u32) -> Self {
        let count = 2 + (start.distance(end) / 8.0).floor() as usize;
        let points = (0..count)
            .map(|i| (i as f32) / (count - 1) as f32)
            .map(|u| end * (1. - u) + start * u)
            .collect();

        // let mut points = Vec::new();
        // points.push(start);
        // points.push(end);
        Self { points, seed }
    }

    fn simulate(&mut self, lightning_settings: &LightningSettings, time: Time) {
        let dt = time.delta_seconds();
        let t = time.elapsed_seconds();
        let noise_amount = lightning_settings.noise_amount;
        let time_scale = lightning_settings.time_scale;
        let meander_amount = lightning_settings.meander_amount;
        let close_amount = lightning_settings.close_amount;
        let mut noise = FastNoise::seeded(self.seed as u64);
        noise.set_noise_type(NoiseType::SimplexFractal);
        noise.set_fractal_type(FractalType::FBM);
        noise.set_fractal_octaves(5);
        noise.set_fractal_gain(0.6);
        noise.set_fractal_lacunarity(2.0);
        noise.set_frequency(lightning_settings.noise_frequency);
        // noise
        for i in 1..(self.points.len() - 1) {
            let n = (self.points[i] - self.points[i - 1])
                .perp()
                .normalize_or_zero();
            let f = noise.get_noise(i as f32 * 1. / self.points.len() as f32, t * time_scale);
            let v = n * f * noise_amount * dt;
            self.points[i] += v;
        }
        // river meander
        for i in 1..(self.points.len() - 1) {
            let n = (self.points[i] - (self.points[i - 1] + self.points[i + 1]) * 0.5)
                .normalize_or_zero();
            let v = n * meander_amount * rand::thread_rng().gen::<f32>() * dt;
            self.points[i] += v;
        }
        // close
        for i in 1..(self.points.len() - 1) {
            let p = (self.points[i - 1] + self.points[i + 1]) * 0.5;
            let n = p - self.points[i];
            let v = n * close_amount * rand::thread_rng().gen::<f32>() * dt;
            self.points[i] += v;
        }

        // short circuit
        let mut cum_dist = vec![0.];
        cum_dist.reserve(self.points.len());
        let mut v = self.points[0];
        let mut d = 0.;
        for i in 1..(self.points.len()) {
            d += v.distance(self.points[i]);
            cum_dist.push(d);
            v = self.points[i];
        }

        let mut b = Vec::new();
        for _i in 0..lightning_settings.short_iterations {
            let mut i1 = rand::thread_rng().gen::<u32>() as usize % self.points.len();
            let mut i2 = rand::thread_rng().gen::<u32>() as usize % self.points.len();
            if i1 == i2 {
                continue;
            }
            if i1 > i2 {
                (i2, i1) = (i1, i2);
            }
            let v1 = self.points[i1];
            let v2 = self.points[i2];
            let crow_dist = v1.distance(v2) + 0.00001; // crow flies distance
            let loop_dist = (cum_dist[i2] - cum_dist[i1]).abs(); // loop distance
            let score = loop_dist / crow_dist;
            let candidate = (i1, i2);
            b.push((candidate, FloatOrd(score)));
        }

        if let Some((best, score)) = b.iter().max_by_key(|x| x.1) {
            if score.0 > lightning_settings.short_threshold {
                let (i1, i2) = *best;
                for i in (i1 + 1)..i2 {
                    let u = (i - i1) as f32 / (i2 - i1) as f32;
                    let v = self.points[i1] * (1. - u) + self.points[i2] * u;
                    self.points[i] = v;
                }

                self.retopo();
            }
        }
    }

    fn retopo(&mut self) {
        let mut cum_dist = vec![0.];
        cum_dist.reserve(self.points.len());
        let mut v = self.points[0];
        let mut d = 0.;
        for i in 1..self.points.len() {
            d += v.distance(self.points[i]);
            cum_dist.push(d);
            v = self.points[i];
        }

        let total_dist = cum_dist.last().unwrap();

        let mut result = vec![self.points[0]];
        result.reserve(self.points.len());
        let mut j = 0;
        for i in 1..self.points.len() {
            let u = i as f32 / (self.points.len() - 1) as f32; // starts at 1/L, ends at 1 inclusive
            let d = u * total_dist; // placing a vertex this distance along the line
                                    // step j forward so that j is the vertex index behind d
            while cum_dist[j + 1] < d {
                j += 1;
            }
            // now j is behind, j+1 is in front, where does this vertex lie between them?
            // it will be spaced as d is between the cum_dists.
            let u_along_line_segment = (d - cum_dist[j]) / (cum_dist[j + 1] - cum_dist[j]); // 0..1
            let v = self.points[j].lerp(self.points[j + 1], u_along_line_segment);
            result.push(v);
        }
        self.points = result;
    }
}

fn main() {
    App::new()
        // .add_plugins(DefaultPlugins)
        .add_plugins(DefaultPlugins.set(ImagePlugin::default_nearest()))
        .add_plugin(DebugLinesPlugin::default())
        .add_plugins(PixelPlugins)
        .add_plugin(EguiPlugin)
        .insert_resource(LightningSettings::default())
        .insert_resource(LastPoint::default())
        .add_startup_system(setup)
        .add_systems((draw, lightning, lightning_ui, add_lightning))
        .insert_resource(ClearColor(Color::BLACK))
        .run();
}

fn draw(query: Query<&Lightning>, mut lines: ResMut<DebugLines>) {
    for lightning in query.iter() {
        for point in lightning.points.windows(2) {
            let s = 2.0;
            lines.line_colored(
                point[0].extend(0.0),
                point[1].extend(0.0),
                0.0,
                Color::rgba(s * 0.5, s * 0.7, s, 1.0),
            );
        }
    }
}

fn lightning(
    mut q: Query<&mut Lightning>,
    time: Res<Time>,
    lightning_settings: Res<LightningSettings>,
) {
    q.par_iter_mut().for_each_mut(|mut l| {
        if lightning_settings.new {
            let mut temp_l = Lightning::new(
                l.points[0],
                l.points[l.points.len() - 1],
                rand::thread_rng().gen::<u32>(),
            );
            temp_l.simulate(&lightning_settings, time.clone());
            *l = temp_l;
        } else {
            l.simulate(&lightning_settings, time.clone());
        }
    })
}

#[derive(Default, Resource, Clone, Deref, DerefMut)]
struct LastPoint(Vec2);

#[derive(Component)]
struct TempLightning;

fn add_lightning(
    mut commands: Commands,
    mut last_point: ResMut<LastPoint>,
    mut temp_lightning: Query<(&mut Lightning, Entity), With<TempLightning>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    // query to get camera transform
    time: Res<Time>,
    lightning_settings: Res<LightningSettings>,
    camera_q: Query<(&Camera, &GlobalTransform)>,
    buttons: Res<Input<MouseButton>>,
) {
    let window = windows.single();
    if let Ok((camera, global_transform)) = camera_q.get_single() {
        if let Some(world_position) = window
            .cursor_position()
            .and_then(|cursor| camera.viewport_to_world_2d(global_transform, cursor))
        {
            if let Ok((mut lightning, entity)) = temp_lightning.get_single_mut() {
                if buttons.just_pressed(MouseButton::Left) {
                    commands.entity(entity).despawn_recursive();
                    commands.spawn(Lightning::new(
                        **last_point,
                        world_position,
                        rand::thread_rng().gen::<u32>(),
                    ));
                    **last_point = world_position;
                } else {
                    let mut temp_lightning = Lightning::new(
                        **last_point,
                        world_position,
                        rand::thread_rng().gen::<u32>(),
                    );

                    temp_lightning.simulate(&lightning_settings, time.clone());

                    *lightning = temp_lightning;
                }
            } else {
                if **last_point != world_position {
                    commands.spawn((
                        Lightning::new(
                            **last_point,
                            world_position,
                            rand::thread_rng().gen::<u32>(),
                        ),
                        TempLightning,
                    ));
                }
            }
        }
    }
}

fn setup(mut commands: Commands) {
    commands.spawn((
        TexturePixelCamera::new(UVec2::new(512, 512), None, Color::BLACK, true),
        BloomSettings {
            intensity: 0.5, // the default is 0.3
            ..default()
        },
    ));

    // commands.spawn((
    //     Camera2dBundle {
    //         camera: Camera {
    //             hdr: true,
    //             ..default()
    //         },
    //         ..default()
    //     },
    //     BloomSettings {
    //         intensity: 0.5, // the default is 0.3
    //         ..default()
    //     },
    // ));
    let segments = 10;
    for i in 0..segments {
        let angle = (i as f32 / segments as f32) * std::f32::consts::PI * 2.0;
        commands.spawn((Lightning::new(
            Vec2::ZERO,
            Vec2::from_angle(angle) * 200.0,
            (i as u32).wrapping_add(rand::thread_rng().gen_range(0..u32::MAX)),
        ),));
    }

    // commands.spawn(Lightning::new(
    //     Vec2::new(0.0, 0.0),
    //     Vec2::new(100.0, 0.0),
    //     1,
    // ));
}

fn lightning_ui(mut egui_context: EguiContexts, mut lightning_settings: ResMut<LightningSettings>) {
    egui::Window::new("Settings").show(egui_context.ctx_mut(), |ui| {
        ui.add(
            egui::Slider::new(&mut lightning_settings.noise_frequency, 0.0..=100.0)
                .text("noise frequency"),
        );
        ui.add(
            egui::Slider::new(&mut lightning_settings.noise_amount, 0.0..=1000.0)
                .text("noise amount"),
        );
        ui.add(
            egui::Slider::new(&mut lightning_settings.short_iterations, 0..=200)
                .text("short iterations"),
        );
        ui.add(
            egui::Slider::new(&mut lightning_settings.short_threshold, 1.0..=10.0)
                .text("noise short_threshold"),
        );
        ui.add(
            egui::Slider::new(&mut lightning_settings.time_scale, 0.0..=10.0).text("time_scale"),
        );
        ui.add(
            egui::Slider::new(&mut lightning_settings.meander_amount, 0.0..=10.0)
                .text("meander_amount"),
        );
        ui.add(
            egui::Slider::new(&mut lightning_settings.close_amount, 0.0..=10.0)
                .text("close_amount"),
        );
        ui.add(egui::Checkbox::new(
            &mut lightning_settings.new,
            "create new lightning",
        ));
    });
}
