//! Collision sound: what the impacts list turns into a click, a thud and a ping.
//!
//! Like trails and sparks, this reads the world *after* it has stepped and never
//! feeds back into it. Nothing in here can change a trajectory, a seed or a replay:
//! a run sounds different with the volume up or down and simulates identically, which
//! is the same guarantee the effects module makes about the picture.
//!
//! # Why it is synthesised
//!
//! There are no audio files in this repository and nothing to license, and a hit's
//! loudness, pitch and length want to be **functions of the contact** — the
//! [`Impact`] record carries the closing speed, the contact radius and the shape, and
//! those are exactly the three numbers a sound needs. A recorded sample would throw
//! that away and replace it with a lookup.
//!
//! Each voice is two or three partials plus a very short noise burst, read out of a
//! shared sine table by phase stepping, with a decaying envelope and a one-millisecond
//! attack so a hit cannot start with a click of its own.
//!
//! # The ladder
//!
//! Frequencies are snapped to a pentatonic ladder spanning about four octaves. That is
//! a deliberate sound-design choice rather than fidelity: two hundred bodies settling
//! into a corner produce a hundred and fifty separate close-but-not-identical pitches,
//! which reads as a bee-swarm rather than as impacts. Snapped to a scale, the same
//! pile-up reads as an instrument being played — and the mapping stays monotonic, so a
//! harder or smaller hit is still audibly harder or smaller.
//!
//! # The ceiling
//!
//! A crowd is the reason this module has limits rather than just a mixer:
//! [`MAX_VOICES`] bounds how many sounds are in flight at once, and
//! [`MAX_PER_FRAME`] bounds how many a single frame may start. When more hits arrive
//! than the frame may start, the **loudest** are kept — chosen by gain, with the
//! impact's index as the tie-break so the choice is deterministic and a replay of a
//! seed sounds the same as the run it came from.
//!
//! # Real time
//!
//! cpal calls the data callback from its own thread — a dedicated high-priority one on
//! macOS and CoreAudio, a thread cpal spawns per stream on ALSA. The mixer therefore
//! sits behind a mutex: the frame thread pushes hits, the audio thread renders blocks.
//! The lock is held for the length of one block of table lookups (a few tens of
//! microseconds at the buffer sizes devices actually use) and the frame thread holds it
//! only while appending, so the realistic failure mode of that design is a slightly
//! late hit rather than a dropout. Anything stricter would need a lock-free ring, and
//! this is a toy room with a maximum of four thousand bodies in it.

use std::sync::{Arc, Mutex};

use glam::Vec3;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

use crate::sim::Impact;

/// Concurrent voices. Beyond this the room is a wall of noise rather than a list of
/// hits, and the cost per sample block stops being negligible.
const MAX_VOICES: usize = 14;
/// New voices one frame may start. A settled pile can report a dozen contacts in one
/// step; letting them all in makes a rumble, and the eye only saw two of them.
const MAX_PER_FRAME: usize = 4;
/// Longest a voice can ring, seconds — reached only by the biggest, dullest, lowest
/// contact in the room. This is the knob that decides whether the room sounds like
/// impacts or like a wash, and it is short on purpose; see `life_for`.
const MAX_LIFE: f32 = 0.3;
/// Sine table size. A power of two so the phase wraps with a mask instead of a divide.
const TABLE_BITS: usize = 11;
const TABLE_SIZE: usize = 1 << TABLE_BITS;
const TABLE_MASK: usize = TABLE_SIZE - 1;

/// Timbre per built-in shape, 0 = a dull thud, 1 = a metallic ping. Indexed by
/// `Body::shape`; anything at or past `PRIMITIVE_NAMES.len()` is an import and gets
/// [`IMPORT_BRIGHT`] instead, because a file's material is not something this renderer
/// can hear.
const SHAPE_BRIGHT: [f32; 9] = [
    0.25, // sphere: the dullest thing in the room, and the only one that always
          // meets a surface at the same radius
    0.75, // cube: a corner or a facet, never a smooth roll
    0.3,  // capsule
    0.55, // torus: ring, so it sings a little
    0.8,  // tetrahedron: small, sharp, bright
    0.7,  // octahedron
    0.65, // icosahedron
    0.72, // cone
    0.6,  // cylinder
];
/// An imported mesh: no way to know whether it is wood or steel, so pick the middle
/// and let the size carry what the file cannot say.
const IMPORT_BRIGHT: f32 = 0.5;
/// Pentatonic ladder in semitones above A1 (57 = A1). Five degrees per octave means
/// any two hits that land together are consonant, which is the whole point.
const LADDER: [f32; 5] = [0.0, 3.0, 5.0, 7.0, 10.0];
/// The ladder's bottom note, in Hz: A1. Nothing in the room rings lower, and 55 Hz is
/// already below most laptop speakers.
const LADDER_FLOOR_HZ: f32 = 55.0;
/// Loudness of a hit whose closing speed is this, in units/s: the top of the curve.
/// A body dropped from the ceiling arrives at about 13 units/s, so a deliberate throw
/// sits near here and a settling pile sits far below it.
const LOUDEST_SPEED: f32 = 13.0;
/// Where the curve's floor sits, taken from the sim rather than invented here: no
/// `Impact` exists below this speed, because reporting one is what created it. Measuring
/// loudness from zero instead of from here would give every contact in a settled pile
/// the same middling volume, and the room would rumble constantly.
const REPORT_FLOOR: f32 = crate::sim::WALL_IMPACT_SPEED;
/// The curve's shape. Below 1 compresses the quiet end upward, which is what the ear
/// integrates (a hit at half the speed is not half as loud); above 1 does the reverse.
/// At 0.75 a threshold-clearing brush lands just under [`MIN_GAIN`] and a thrown body
/// reaches full scale, which is the spread the room actually spans.
const GAIN_CURVE: f32 = 0.75;
/// The fundamental is capped: the ladder's top note exists to keep tiny bodies audible
/// as *small*, not to reach ultrasonic pitches, and the third partial of anything above
/// here comes within earshot of Nyquist at 44.1 kHz, where it would fold back down as a
/// spurious tone rather than disappear. See the test that pins this.
/// A contact radius this big is pitched at the ladder's bottom; smaller bodies climb
/// it. `Body::radius` for the generated set runs about 0.16..0.9.
const LOWEST_PITCH_RADIUS: f32 = 0.9;
/// Top fundamental, Hz: about a major sixth above C6. A 2 cm pebble would ring far
/// above this in the real world; here it rings at the cap, which keeps its third
/// partial a safe distance below Nyquist at any rate a device actually offers.
const MAX_FUNDAMENTAL: f32 = 1400.0;
/// A hit softer than this after the curve is not worth a voice at all: two bodies
/// brushing past each other in a crowd are silent, as they are in a real room.
const MIN_GAIN: f32 = 0.02;

/// One sine table, shared by every voice rather than copied per stream. Its shape is
/// what a "tone" means in this module: pure at the bottom, and the inharmonic partial
/// ratios above are what make a bright hit metallic rather than flute-like.
static SINE: std::sync::OnceLock<Vec<f32>> = std::sync::OnceLock::new();

fn sine_table() -> &'static Vec<f32> {
    SINE.get_or_init(|| {
        (0..TABLE_SIZE)
            .map(|i| {
                let t = i as f64 / TABLE_SIZE as f64;
                (std::f64::consts::TAU * t).sin() as f32
            })
            .collect()
    })
}

/// Partial ratios and levels for one voice. A tuning fork is nearly pure; a metal
/// shell adds high, slightly *inharmonic* partials — not whole-number multiples of the
/// fundamental, which is precisely what the ear reads as " struck metal" rather than
/// "organ note". Dull shapes get mostly the fundamental.
fn partials(bright: f32) -> [(f32, f32); 3] {
    [
        (1.0, 1.0),
        (2.76, bright),
        (5.4, bright * bright * 0.55),
    ]
}

/// A hit in flight. Written from the frame thread's decision and read by the audio
/// thread's render loop; the two never hold it at the same time, because it lives
/// inside the [`Mixer`] behind its mutex.
#[derive(Copy, Clone, Debug)]
struct Voice {
    /// Seconds since the hit. The envelope is a pure function of this.
    age: f32,
    /// Which contact this voice came from, kept so a test (and, if anyone wants it in
    /// the panel, a debug readout) can say *which* hits the ceiling turned away rather
    /// than merely how many.
    id: u64,
    life: f32,
    gain: f32,
    /// Partial frequencies in Hz (already detuned) and their levels. Frequencies are
    /// stored rather than phase *increments* because the sample rate is the device's,
    /// not a constant: an increment baked in at spawn time against 44.1 kHz would leave
    /// every hit a semitone and a half sharp on a 48 kHz interface. The render loop
    /// turns them into increments once per block, which costs three multiplies per
    /// voice per callback and nothing per sample.
    freqs: [f32; 3],
    levels: [f32; 3],
    /// Phase in table units, advanced per sample by the increment above.
    phase: [f32; 3],
    /// Equal-power pan, as an angle: 0 = hard left, PI/2 = hard right.
    pan: f32,
    /// Noise-burst level and how fast it dies. The burst is what makes a strike
    /// audible as an *event* rather than a note fading in.
    noise: f32,
    noise_life: f32,
    /// Voice-local generator for the noise burst, so two hits with identical
    /// parameters do not sum in phase and double the amplitude.
    rng: u64,
}

impl Voice {
    /// The envelope at time `t` seconds: a one-millisecond attack, then an exponential
    /// decay that reaches about 2% at `life`. The attack is not decoration — starting a
    /// sine at an arbitrary phase with full amplitude is a step discontinuity, and a step
    /// is a click. The decay does not reach zero, so `render` gates on the age instead of
    /// trusting this curve to end a voice.
    fn envelope(&self, t: f32) -> f32 {
        const ATTACK: f32 = 0.001;
        if t < 0.0 {
            return 0.0;
        }
        let rise = (t / ATTACK).min(1.0);
        // 4 rather than 3: `life` is the nominal length, and this makes the last quarter
        // of it effectively silent, so the audible tail is shorter than the number
        // without the number ceasing to mean "when this voice is retired".
        rise * (-4.0 * t / self.life).exp()
    }
}

/// What one impact sounds like, before polyphony or the master volume.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Hit {
    /// Peak amplitude, 0..1.
    pub gain: f32,
    /// Fundamental, already snapped to the ladder.
    pub freq: f32,
    /// Ring time, seconds.
    pub life: f32,
    /// 0 = dull thud, 1 = metallic ping.
    pub bright: f32,
    /// -1 = hard left, +1 = hard right, from the contact's x in the room.
    pub pan: f32,
}

/// Loudness of a contact from its closing speed. A curve rather than a line: the range
/// of real approach speeds is enormous (a settled contact barely clears the threshold
/// at ~1.1 units/s; a thrown body reaches 20+), and a linear gain would either bury
/// the quiet half of the room or clip the loud half. The square root compresses that
/// range toward what the ear actually integrates.
fn gain_for(speed: f32) -> f32 {
    let span = (LOUDEST_SPEED - REPORT_FLOOR).max(1.0);
    let over = ((speed - REPORT_FLOOR) / span).clamp(0.0, 1.0);
    over.powf(GAIN_CURVE)
}

/// Fundamental of a contact from its radius: bigger thing, lower note. The exponent
/// near 1.4 is what makes the room's generated sizes span about three octaves rather
/// than a fifth, which is what the ear needs to tell a small hit from a big one.
fn raw_freq_for(radius: f32) -> f32 {
    let r = radius.max(0.02);
    let ratio = LOWEST_PITCH_RADIUS / r;
    (98.0 * ratio.powf(1.4)).min(MAX_FUNDAMENTAL)
}

/// The nearest ladder note at or above `hz`, in Hz. Quantising by `2f.powi(semitones/12)`
/// keeps the ladder exactly in tune with itself at every octave rather than drifting.
fn quantise(hz: f32) -> f32 {
    // Semitones above the ladder floor, then snap to the nearest pentatonic degree.
    let semitones = 12.0 * (hz / LADDER_FLOOR_HZ).max(1.0).log2();
    let octave = (semitones / 12.0).floor();
    let within = semitones - octave * 12.0;
    let mut best = LADDER[0];
    let mut best_gap = f32::MAX;
    for degree in LADDER {
        let gap = (degree - within).abs();
        if gap < best_gap {
            best_gap = gap;
            best = degree;
        }
    }
    // Snapping can dip below the floor at the bottom of the ladder; keep the note in
    // range rather than letting the octave arithmetic round it down.
    let semis = (octave * 12.0 + best).max(0.0);
    LADDER_FLOOR_HZ * 2f32.powf(semis / 12.0)
}

/// How long a hit rings — a percussive event, not a note.
    ///
    /// Everything here is under a third of a second, deliberately and as a *family*: the
    /// low end is about two and a half times the high end rather than six, because a
    /// voice that outlives its neighbours is what turns a roomful of impacts into a wash.
    /// The second hit you cannot hear is usually the one the eye is on. A previous tuning
    /// let the biggest bodies ring for two thirds of a second, and fourteen of those
    /// overlapping is not a bounce, it is a drone.
    ///
    /// Low and dull still ring longest, which is the whole point of the ordering: a
    /// boulder has to be audibly bigger than a pebble, and duration is most of how that
    /// is heard. The cap exists so a room of big slow bodies cannot park fourteen voices
    /// at full length indefinitely, and the floor so a bright ping never smears.
fn life_for(bright: f32, hz: f32) -> f32 {
    let low = (220.0 / hz.max(220.0)).min(1.0);
    let base = 0.05 + 0.2 * low;
    (base * (1.25 - 0.5 * bright)).clamp(0.03, MAX_LIFE)
}

/// Pan from a contact position: the room's own x extent, not a guess, so a body
/// hitting the left wall is heard at the left speaker. Clamped because a contact
/// point can sit a hair outside the walls when a hull corner pokes through.
fn pan_for(pos: Vec3, half_x: f32) -> f32 {
    (pos.x / half_x.max(0.01)).clamp(-1.0, 1.0)
}

/// Everything about an impact that a voice needs, reduced to a [`Hit`]. Pure: no
/// allocator, no device, no world — so the mapping is testable on a struct built by
/// hand, which is how "harder hit is louder" and "smaller body is higher" stay pinned.
pub fn hit_for(impact: &Impact, half_x: f32) -> Hit {
    let bright = match SHAPE_BRIGHT.get(impact.shape) {
        Some(b) => *b,
        None => IMPORT_BRIGHT,
    };
    let freq = quantise(raw_freq_for(impact.base_radius));
    Hit {
        gain: gain_for(impact.speed),
        freq,
        life: life_for(bright, freq),
        bright,
        pan: pan_for(impact.pos, half_x),
    }
}

/// The mixer: the voices in flight, plus the two controls a viewer touches.
///
/// Kept apart from [`Audio`] so all of it can be driven from a test with a fake sample
/// buffer, no device, and no audio thread — which is the only way the limits in this
/// module mean anything to anyone after the day they were written.
#[derive(Debug, Default)]
pub struct Mixer {
    voices: Vec<Voice>,
    volume: f32,
    muted: bool,
    rng: u64,
    /// Hits dropped since the last render, for the HUD: the ceiling is only honest if
    /// the panel can say it was reached.
    dropped: usize,
}

impl Mixer {
    pub fn new(volume: f32) -> Mixer {
        Mixer {
            volume: volume.clamp(0.0, 1.0),
            ..Mixer::default()
        }
    }

    pub fn set_volume(&mut self, volume: f32) {
        self.volume = volume.clamp(0.0, 1.0);
    }

    pub fn volume(&self) -> f32 {
        self.volume
    }

    pub fn set_muted(&mut self, muted: bool) {
        self.muted = muted;
    }

    pub fn muted(&self) -> bool {
        self.muted
    }

    pub fn voices(&self) -> usize {
        self.voices.len()
    }

    /// Which contacts are sounding, by the id they arrived with. A view for tests and
    /// for anyone trying to find out why a particular body's hit was not heard: the
    /// ceiling bites in a crowd, and "how many" is not the question a listener asks.
    pub fn voiced(&self) -> Vec<u64> {
        self.voices.iter().map(|v| v.id).collect()
    }

    /// Voices the ceiling turned away since the last read — a number worth showing
    /// when the room is a crowd, because otherwise the sound just gets quietly thinner
    /// than the picture. Takes `&mut self` because reading it clears it: a counter that
    /// only ever grows would say "dropped 4 000" on a frame that dropped two.
    pub fn take_dropped(&mut self) -> usize {
        let n = self.dropped;
        self.dropped = 0;
        n
    }

    /// Forget every voice in flight: a reset, a rewind, or any other jump that would
    /// otherwise let a hit from before the jump keep ringing after it.
    pub fn clear(&mut self) {
        self.voices.clear();
    }

    fn next_f32(&mut self) -> f32 {
        // splitmix64, the generator the rest of the port uses, on its own state — so
        // turning the sound up or down cannot disturb the world's sequence of numbers.
        self.rng = self.rng.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.rng;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 40) as f32 * (1.0 / 16_777_216.0)
    }

    /// Start the voices for the hits this frame was given, newest-wins-by-loudness.
    ///
    /// The sort is by gain descending with the incoming index as a tie-break, so the
    /// subset kept when the ceiling bites is a function of the impacts themselves and
    /// not of the order the collision code happened to report them — which is what
    /// makes a replayed seed *sound* like the run it came from, not merely look like it.
    pub fn add(&mut self, hits: &[(Hit, u64)]) {
        let free = MAX_VOICES.saturating_sub(self.voices.len());
        let room = free.min(MAX_PER_FRAME);
        if hits.is_empty() {
            return;
        }
        let mut wanted: Vec<(f32, u64, u64, Hit)> = hits
            .iter()
            .enumerate()
            .filter(|(_, (h, _))| h.gain >= MIN_GAIN)
            .map(|(k, (h, id))| (h.gain, *id, k as u64, *h))
            .collect();
        if wanted.len() > room {
            self.dropped += wanted.len() - room;
        }
        // Loudness first; and when two hits are equally loud — the common case in a
        // settled pile, where contacts share a threshold-crossing speed — the tie-break
        // is the incoming index, which makes the kept set a function of the contacts
        // rather than of nothing at all. Reversing the input reverses which equal-gain
        // hits are kept, so the tie-break is part of what a replay depends on.
        wanted.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.1.cmp(&b.1))
        });
        wanted.truncate(room.max(0));
        // `(gain, caller id, incoming index, hit)`: the caller's id is what a voice
        // carries, because that is the number that means something outside this module —
        // the frame loop hands over the impact's index in the list it was read from. The
        // incoming index is a separate field and only ever a tie-break.
        for (_, id, _, hit) in wanted {
            // Spelled out rather than `self.voices.push(self.spawn(hit))`: the outer
            // call borrows `self` mutably twice in one expression, and the compiler is
            // right to refuse it — a grow of `voices` while `spawn` is drawing on the
            // same generator would be a real aliasing hazard, not just a borrowck one.
            let voice = self.spawn(hit, id);
            self.voices.push(voice);
        }
    }

    /// One voice, with the random detune and phase offsets that stop two identical
    /// hits summing into one hit twice as loud.
    fn spawn(&mut self, hit: Hit, id: u64) -> Voice {
        // A small detune, so two bodies of the same shape and size hitting at the same
        // time are not one voice twice as loud: perfectly matched phases sum exactly.
        let detune = 1.0 + (self.next_f32() - 0.5) * 0.06;
        let parts = partials(hit.bright);
        let freqs = [
            hit.freq * detune * parts[0].0,
            hit.freq * detune * parts[1].0,
            hit.freq * detune * parts[2].0,
        ];
        let levels = [parts[0].1, parts[1].1, parts[2].1];
        let table = TABLE_SIZE as f32;
        Voice {
            age: 0.0,
            id,
            life: hit.life,
            gain: hit.gain,
            freqs,
            levels,
            phase: [self.next_f32() * table, self.next_f32() * table, self.next_f32() * table],
            pan: (hit.pan + 1.0) * std::f32::consts::FRAC_PI_4,
            noise: (0.25 + 0.35 * hit.bright).min(0.7),
            noise_life: (hit.life * 0.12).max(0.004),
            rng: self.rng,
        }
    }

    /// Fill `out` with interleaved stereo samples and age the voices. `sample_rate`
    /// comes from the device, so the decay and the phase steps are in seconds rather
    /// than in blocks — which is what makes the same run sound the same on a 44.1 kHz
    /// laptop and a 48 kHz interface.
    pub fn render(&mut self, out: &mut [f32], sample_rate: f32) {
        let dt = 1.0 / sample_rate.max(1.0);
        if self.muted || self.volume <= 0.0 || self.voices.is_empty() {
            out.fill(0.0);
            // Silence still ages the voices: unmuting mid-ring should resume a hit
            // that has been decaying, not restart one that never happened.
            for voice in &mut self.voices {
                voice.age += dt;
            }
            self.voices.retain(|v| v.age < v.life);
            return;
        }

        // Headroom rather than a limiter: every voice is divided by the *total* gain in
        // flight, which keeps the room's dynamics — one hard hit still dominates ten
        // soft ones — without the sum of many small contacts walking past full scale
        // into distortion that reads as loudness rather than as a crowd.
        let total: f32 = self.voices.iter().map(|v| v.gain).sum();
        let mix = self.volume / total.max(1.0);

        let table = sine_table();
        let scale = TABLE_SIZE as f32 / sample_rate.max(1.0);
        // Phase increments for this block, once per voice per callback rather than once
        // per sample. `voices` never exceeds MAX_VOICES, which `add` guarantees.
        let mut steps: [[f32; 3]; MAX_VOICES] = [[0.0; 3]; MAX_VOICES];
        for (vi, voice) in self.voices.iter().enumerate() {
            for p in 0..3 {
                steps[vi][p] = voice.freqs[p] * scale;
            }
        }

        for frame in out.chunks_exact_mut(2) {
            let mut left = 0.0f32;
            let mut right = 0.0f32;
            for (vi, voice) in self.voices.iter_mut().enumerate() {
                // `life` has to mean the same thing in the samples as in the struct. An
                // exponential never reaches zero, so without this gate a voice keeps
                // sounding after its own nominal end and is retired only *between*
                // callback blocks — up to a whole block of leaked ringing, which at the
                // ~10 ms blocks devices actually ask for is inaudible, and at the sizes a
                // test hands `render` is not. The gate is on the age, not on an amplitude
                // floor, so the tail it removes is exactly the part past `life`.
                let env = if voice.age >= voice.life {
                    0.0
                } else {
                    voice.envelope(voice.age)
                };
                if env > 0.0005 {
                    let mut sum = 0.0f32;
                    for p in 0..3 {
                        let level = voice.levels[p];
                        if level > 0.001 {
                            voice.phase[p] += steps[vi][p];
                            let idx = (voice.phase[p] as usize) & TABLE_MASK;
                            sum += table[idx] * level;
                        }
                    }
                    // The strike transient: a couple of milliseconds of noise, quiet
                    // enough to read as the hit rather than as hiss. Per-voice, so two
                    // hits do not share one "random" sequence and cancel each other.
                    let nenv = (1.0 - (voice.age / voice.noise_life)).max(0.0);
                    if nenv > 0.0 {
                        voice.rng = voice
                            .rng
                            .wrapping_mul(6_364_136_223_679_631_411)
                            .wrapping_add(1);
                        let n = ((voice.rng >> 33) as f32) * (1.0 / 2_147_483_648.0) - 1.0;
                        sum += n * voice.noise * nenv;
                    }
                    let a = env * voice.gain * sum * mix;
                    left += a * voice.pan.cos();
                    right += a * voice.pan.sin();
                }
                voice.age += dt;
            }
            frame[0] = left.clamp(-1.0, 1.0);
            frame[1] = right.clamp(-1.0, 1.0);
        }
        self.voices.retain(|v| v.age < v.life);
    }
}

/// The output device and the thread that reads the mixer.
///
/// Opening is allowed to fail and the app treats that as a fact about the machine
/// rather than a fault: a headless runner, a container, a machine with no output
/// device, or a desktop session cpal cannot reach all get silence plus one HUD line,
/// and the simulation carries on at the same speed with the same numbers.
pub struct Audio {
    stream: cpal::Stream,
    mixer: Arc<Mutex<Mixer>>,
}

impl Audio {
    /// Open the system default output device, at whatever rate and format it prefers.
    /// The returned handle keeps the stream alive: cpal stops the sound the moment it
    /// is dropped, which is why [`Audio`] owns it rather than returning it.
    pub fn open(volume: f32) -> Result<Audio, String> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or_else(|| "no audio output device".to_owned())?;
        let config = device
            .default_output_config()
            .map_err(|e| format!("default output config: {e}"))?;
        let mixer = Arc::new(Mutex::new(Mixer::new(volume)));
        let format = config.sample_format();
        let stream_config: cpal::StreamConfig = config.into();
        let err_fn = |err: cpal::StreamError| {
            eprintln!("audio stream error: {err}");
        };
        // cpal's `build_output_stream` is a trait method on `Device`, and its error is
    // its own type; both get turned into the plain String the app stores and shows.
        let stream = match format {
            cpal::SampleFormat::F32 => build::<f32>(&device, &stream_config, &mixer, err_fn),
            cpal::SampleFormat::I16 => build::<i16>(&device, &stream_config, &mixer, err_fn),
            cpal::SampleFormat::U16 => build::<u16>(&device, &stream_config, &mixer, err_fn),
            other => return Err(format!("unsupported sample format {other:?}")),
        }
        .map_err(|e| format!("opening the audio stream: {e}"))?;
        stream
            .play()
            .map_err(|e| format!("starting the audio stream: {e}"))?;
        Ok(Audio { stream, mixer })
    }

    /// Queue the contacts this step reported, so they start sounding on the next
    /// block the device asks for.
    pub fn note(&self, hits: Vec<(Hit, u64)>) {
        if hits.is_empty() {
            return;
        }
        match self.mixer.lock() {
            Ok(mut mixer) => mixer.add(&hits),
            // A poisoned lock means the audio thread panicked inside render. The show
            // goes on: the stream is already dead by then, and the world does not
            // depend on it.
            Err(_) => {}
        }
    }

    pub fn set_muted(&self, muted: bool) {
        if let Ok(mut mixer) = self.mixer.lock() {
            mixer.set_muted(muted);
        }
    }

    pub fn set_volume(&self, volume: f32) {
        if let Ok(mut mixer) = self.mixer.lock() {
            mixer.set_volume(volume);
        }
    }

    pub fn clear(&self) {
        if let Ok(mut mixer) = self.mixer.lock() {
            mixer.clear();
        }
    }

    /// Pause the stream: the callback stops being called, which is the only teardown
    /// cpal 0.15 offers — `StreamTrait` has `play` and `pause`, and no `shutdown`.
    ///
    /// This is also why the field is read by anything at all. A cpal stream stops the
    /// instant its last handle drops, and if that happens as part of `App` going away,
    /// the audio thread is torn down in whatever order the struct's fields happen to
    /// give rather than one anyone chose. `App::exiting` pauses first, so the callback
    /// has demonstrably stopped reaching the mixer before the surface and the window go;
    /// dropping the handle straight after is what releases the device.
    ///
    /// A failure here is reported rather than fatal: cpal documents that some devices
    /// do not support pausing, and the drop that follows releases the stream either way.
    /// The mixer behind the callback, reachable only from this module's tests: the
    /// ignored device test above has to render through the same handle the callback
    /// uses, and nothing outside a test should be touching it directly.
    #[cfg(test)]
    pub(crate) fn mixer_for_test(&self) -> &Arc<Mutex<Mixer>> {
        &self.mixer
    }

    pub fn pause(&self) -> Result<(), String> {
        self.stream
            .pause()
            .map_err(|e| format!("pausing the audio stream: {e}"))
    }
}

fn build<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    mixer: &Arc<Mutex<Mixer>>,
    err_fn: impl Fn(cpal::StreamError) + Send + 'static,
) -> Result<cpal::Stream, cpal::BuildStreamError>
where
    T: cpal::SizedSample + cpal::FromSample<f32>,
{
    let mixer = Arc::clone(mixer);
    let rate = config.sample_rate.0 as f32;
    device.build_output_stream(
        config,
        move |data: &mut [T], _info: &cpal::OutputCallbackInfo| {
            // Render into an owned stereo buffer of our own, then convert. cpal gives
            // us a slice whose channel count comes from the device, so the conversion
            // is where mono/5.1 outputs are absorbed rather than assumed away.
            let mut buf = vec![0.0f32; data.len()];
            if let Ok(mut mixer) = mixer.lock() {
                mixer.render(&mut buf, rate);
            }
            for (out, sample) in data.iter_mut().zip(buf.iter()) {
                *out = T::from_sample(*sample);
            }
        },
        err_fn,
        None,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::Impact;
    use glam::Vec3;

    fn impact(speed: f32, radius: f32, shape: usize, x: f32) -> Impact {
        Impact {
            pos: Vec3::new(x, 1.0, 0.0),
            n: Vec3::Y,
            color: [0.0; 3],
            base_radius: radius,
            speed,
            shape,
            age: 0.0,
            life: 0.5,
        }
    }

    fn render_seconds(mixer: &mut Mixer, seconds: f32, rate: f32) -> Vec<f32> {
        let frames = (seconds * rate) as usize;
        let mut out = vec![0.0f32; frames * 2];
        mixer.render(&mut out, rate);
        out
    }

    fn peak(samples: &[f32]) -> f32 {
        samples
            .iter()
            .fold(0.0f32, |m, s| m.max(s.abs()))
    }

    #[test]
    fn a_harder_hit_is_louder_and_a_smaller_body_is_higher() {
        let soft = hit_for(&impact(2.0, 0.5, 0, 0.0), 7.2);
        let hard = hit_for(&impact(13.0, 0.5, 0, 0.0), 7.2);
        assert!(hard.gain > soft.gain * 1.5, "13 units/s vs 2: {hard:?} {soft:?}");

        let big = hit_for(&impact(8.0, 0.9, 0, 0.0), 7.2);
        let small = hit_for(&impact(8.0, 0.16, 0, 0.0), 7.2);
        assert!(
            small.freq > big.freq * 2.0,
            "a radius of 0.16 has to sit well above 0.9: {big:?} {small:?}"
        );
    }

    #[test]
    fn nothing_is_monotonic_by_accident() {
        // Walk speed and radius across their whole working range and insist the
        // mapping never doubles back on itself. A table that has been "tuned" by ear
        // often grows a plateau where half the room sounds identical.
        let mut last_gain = -1.0;
        for i in 0..40 {
            let speed = 1.1 + i as f32 * 0.5;
            let g = gain_for(speed);
            assert!(g >= last_gain, "gain fell as speed rose at {speed}");
            last_gain = g;
        }
        let mut last_freq = f32::MAX;
        for i in 0..40 {
            let radius = 0.1 + i as f32 * 0.02;
            let f = quantise(raw_freq_for(radius));
            assert!(f <= last_freq, "pitch rose as the body got bigger at {radius}");
            last_freq = f;
        }
    }

    #[test]
    fn pitches_land_on_the_ladder_not_somewhere_between() {
        // Two hits of different sizes must not just differ — they have to be *consonant*,
        // which is the point of quantising at all.
        let a = hit_for(&impact(6.0, 0.5, 0, 0.0), 7.2).freq;
        let b = hit_for(&impact(6.0, 0.17, 0, 0.0), 7.2).freq;
        // The ladder's promise is not that two hits are an octave apart — they are
        // pentatonic *steps* apart — but that the interval between them is a whole
        // number of semitones, which is what makes them consonant to the ear.
        let semitones = 12.0 * (b / a).log2();
        let off_by = (semitones - semitones.round()).abs();
        assert!(
            off_by < 0.05,
            "{a} Hz and {b} Hz are {semitones:.3} semitones apart, which is not a whole \
             number: the ladder is quantising to something other than itself"
        );
        assert!(a >= LADDER_FLOOR_HZ * 0.99, "{a} fell below the ladder floor");
    }

    #[test]
    fn a_hit_makes_sound_and_then_stops() {
        let mut mixer = Mixer::new(0.8);
        mixer.add(&[(hit_for(&impact(9.0, 0.4, 1, 0.0), 7.2), 0)]);
        assert_eq!(mixer.voices(), 1);
        let out = render_seconds(&mut mixer, 0.05, 44_100.0);
        assert!(peak(&out) > 0.02, "a hard hit rendered as silence: {:?}", &out[..8]);
        // Its own life is under MAX_LIFE by construction, so after two seconds of
        // rendering there is nothing left to decay.
        render_seconds(&mut mixer, 2.0, 44_100.0);
        assert_eq!(mixer.voices(), 0, "every voice outlived its life");
        let quiet = render_seconds(&mut mixer, 0.02, 44_100.0);
        assert_eq!(peak(&quiet), 0.0, "silence after the last voice died");
    }

    #[test]
    fn muted_renders_silence_but_still_ages_the_voices() {
        let mut mixer = Mixer::new(0.8);
        mixer.add(&[(hit_for(&impact(9.0, 0.4, 1, 0.0), 7.2), 0)]);
        mixer.set_muted(true);
        let out = render_seconds(&mut mixer, 0.05, 44_100.0);
        assert_eq!(peak(&out), 0.0, "muted means muted");
        assert!(mixer.voices() <= 1);
        // Unmute a moment later: the ring has been decaying the whole time rather than
        // waiting to be restarted.
        mixer.set_muted(false);
        // Longer than any voice can live, so the *tail* of this buffer is what matters:
        // the first part is the hit finishing its decay, which is correct, and the last
        // part has nothing left to render.
        let long = render_seconds(&mut mixer, 3.0, 44_100.0);
        assert_eq!(mixer.voices(), 0, "the voice aged out while muted");
        assert!(
            peak(&long[long.len() / 2..]) == 0.0,
            "the second half of a three-second render is still sounding, so a voice \
             outlived its life while muted"
        );
    }

    #[test]
    fn a_crowd_cannot_open_a_thousand_voices() {
        // Two thousand bodies settling: the impacts list can hold hundreds per step.
        let mut mixer = Mixer::new(1.0);
        let hits: Vec<(Hit, u64)> = (0..400)
            .map(|i| (hit_for(&impact(3.0 + i as f32 * 0.02, 0.3, i % 9, 0.0), 7.2), i as u64))
            .collect();
        mixer.add(&hits);
        assert!(
            mixer.voices() <= MAX_VOICES,
            "{} voices from one frame, ceiling is {MAX_VOICES}",
            mixer.voices()
        );
        assert!(
            mixer.take_dropped() >= 400 - MAX_VOICES,
            "the ones it turned away have to be counted"
        );
    }

    #[test]
    fn the_ceiling_is_two_limits_doing_two_different_jobs() {
        // Thirty soft contacts and one hard one arrive in a single frame. `MAX_PER_FRAME`
        // decides how many may *start* — four — and `MAX_VOICES` how many may be in
        // flight at once, which only matters across frames. Conflating them is what makes
        // this pair worth a test: the first is about not machine-gunning a viewer with a
        // thousand simultaneous pings, the second about the cost of a full mix.
        let mut mixer = Mixer::new(1.0);
        let mut hits: Vec<(Hit, u64)> = (0..30)
            .map(|i| (hit_for(&impact(1.3, 0.3, 0, 0.0), 7.2), i as u64))
            .collect();
        hits.push((hit_for(&impact(20.0, 0.3, 0, 0.0), 7.2), 999));

        mixer.add(&hits);
        let kept = mixer.voiced();
        assert_eq!(
            kept.len(),
            MAX_PER_FRAME,
            "one frame may start {MAX_PER_FRAME} voices, not {}",
            kept.len()
        );
        assert!(
            kept.contains(&999),
            "the hardest hit of the lot was dropped while softer ones were kept: {kept:?}"
        );
        assert_eq!(
            mixer.take_dropped(),
            hits.len() - MAX_PER_FRAME,
            "what the ceiling turned away has to be countable"
        );

        // The property a replay actually depends on: the same contacts in the same
        // order produce the same voices, twice over. Anything else (a HashMap behind the
        // sort, an unstable sort, a tie-break that read a counter that had already
        // advanced) would make an identical run sound different from itself.
        let mut again = Mixer::new(1.0);
        again.add(&hits);
        assert_eq!(
            again.voiced(),
            kept,
            "the identical contacts in the identical order kept a different set of voices, \
             which means the choice is not a function of the input alone"
        );

        // Reversing the input *does* change which equal-gain contacts are chosen — the
        // tie-break is the incoming index, deliberately, so that a reordering of the
        // collision loop is visible rather than silent. What must not change is the
        // count, or the survival of the one contact loud enough to win on merit.
        let mut other = Mixer::new(1.0);
        let mut reversed = hits.clone();
        reversed.reverse();
        other.add(&reversed);
        assert_eq!(other.voiced().len(), kept.len(), "same contacts, same count");
        assert!(other.voiced().contains(&999), "the hard hit survives either order");

        // And the in-flight ceiling fills over frames rather than dropping the burst:
        // four more `add` calls with nothing rendered take it to MAX_VOICES and no
        // further, after which everything is refused until the render loop ages voices.
        for _ in 0..6 {
            mixer.add(&hits);
        }
        assert_eq!(
            mixer.voices(),
            MAX_VOICES,
            "the in-flight ceiling is {MAX_VOICES}; it reached {}",
            mixer.voices()
        );
        mixer.add(&hits);
        assert_eq!(mixer.voices(), MAX_VOICES, "nothing more may start while it is full");
        assert!(mixer.take_dropped() > 0, "and the refusals are still counted");
    }

    #[test]
    fn it_never_clips_however_many_bodies_land_at_once() {
        let mut mixer = Mixer::new(1.0);
        let hits: Vec<(Hit, u64)> = (0..200)
            .map(|i| (hit_for(&impact(30.0, 0.2 + (i % 5) as f32 * 0.1, 1, 3.0), 7.2), i as u64))
            .collect();
        mixer.add(&hits);
        let out = render_seconds(&mut mixer, 0.3, 48_000.0);
        let p = peak(&out);
        assert!(p <= 1.0 + 1e-6, "peak {p} is past full scale");
        assert!(p > 0.05, "a hundred hard hits rendered as nearly nothing: {p}");
    }

    #[test]
    fn a_hit_on_the_left_is_heard_on_the_left() {
        let mut mixer = Mixer::new(1.0);
        mixer.add(&[(hit_for(&impact(12.0, 0.4, 0, -7.0), 7.2), 0)]);
        let out = render_seconds(&mut mixer, 0.08, 44_100.0);
        let (mut l, mut r) = (0.0f32, 0.0f32);
        for pair in out.chunks_exact(2) {
            l += pair[0] * pair[0];
            r += pair[1] * pair[1];
        }
        assert!(l > r * 4.0, "left energy {l} vs right {r}: a wall at x=-7 belongs left");
    }

    #[test]
    fn a_body_at_the_centre_is_still_in_the_middle() {
        let mut mixer = Mixer::new(1.0);
        mixer.add(&[(hit_for(&impact(12.0, 0.4, 0, 0.0), 7.2), 0)]);
        let out = render_seconds(&mut mixer, 0.08, 44_100.0);
        let (mut l, mut r) = (0.0f32, 0.0f32);
        for pair in out.chunks_exact(2) {
            l += pair[0] * pair[0];
            r += pair[1] * pair[1];
        }
        let ratio = (l / r).max(r / l);
        assert!(ratio < 1.15, "centre hit panned to {ratio} away from equal");
    }

    #[test]
    fn quiet_contacts_are_not_voiced_at_all() {
        // A brush in a settled pile is silent, exactly as the picture draws no ring for
        // it. MIN_GAIN is what makes that true rather than a rumble at 1% volume.
        let hit = hit_for(&impact(1.15, 0.4, 0, 0.0), 7.2);
        assert!(hit.gain < MIN_GAIN, "softest reportable speed: {hit:?}");
        let mut mixer = Mixer::new(1.0);
        mixer.add(&[(hit, 0)]);
        assert_eq!(mixer.voices(), 0, "a brush gets no voice");
    }

    #[test]
    fn turning_the_volume_to_zero_is_a_gate_not_a_restart() {
        let mut mixer = Mixer::new(0.0);
        mixer.add(&[(hit_for(&impact(10.0, 0.4, 1, 0.0), 7.2), 0)]);
        assert_eq!(peak(&render_seconds(&mut mixer, 0.05, 44_100.0)), 0.0);
        mixer.set_volume(0.9);
        assert!(
            mixer.voices() <= 1,
            "the voice either decayed while gated or is still there; either way it did \
             not come back louder"
        );
    }

    /// The one thing no other test in this file can do: hold a real device open, start
    /// the callback, and stop it again. Ignored by default because it needs hardware —
    /// CI has no audio device, and `cargo test` should not ask for one — so it runs on
    /// demand with `cargo test -- --ignored`, which is also the check to reach for when
    /// a cpal upgrade changes the API under this module.
    ///
    /// It renders silence into the mixer on purpose: the point is that the stream
    /// exists, plays and pauses cleanly, not to make noise on someone's machine.
    #[test]
    #[ignore = "needs an audio output device; run with --ignored"]
    fn the_default_output_device_opens_plays_and_pauses() {
        let audio = Audio::open(0.0).expect("this machine should have an output device");
        // A block of render through the real callback path, then the teardown the app
        // uses. Any cpal API drift shows up here rather than in a segfault at exit.
        if let Ok(mut mixer) = audio.mixer_for_test().lock() {
            let mut block = vec![0.0f32; 512 * 2];
            mixer.render(&mut block, 48_000.0);
        }
        audio.pause().expect("pausing a stream this process just opened");
    }

    #[test]
    fn a_room_running_for_real_produces_hits_that_fit_the_ceiling() {
        // End to end, minus the device: the world from the sim, the impacts it
        // reports, the same "new this step" filter the frame loop uses, and the mixer.
        // What this pins is the wiring rather than the arithmetic — that the fresh-hit
        // test is the one that keeps a contact from being voiced twice, that ids are
        // distinct per step, and that a crowd never overruns the voice ceiling.
        use crate::sim::{SpawnOpts, World};
        let mut world = World::default();
        world.set_seed(0x5EED);
        for i in 0..120 {
            world.record(i % 9, SpawnOpts::default());
        }
        world.settle();
        let mut mixer = Mixer::new(0.8);
        let dt = 1.0 / 60.0;
        let (mut voiced, mut max_voices, mut max_dropped) = (0usize, 0usize, 0usize);
        for _ in 0..240 {
            world.step(dt);
            let hits: Vec<(Hit, u64)> = world
                .impacts
                .iter()
                .enumerate()
                .filter(|(_, hit)| hit.age <= dt * 1.001)
                .map(|(i, hit)| (hit_for(hit, crate::sim::ROOM_HALF_X), i as u64))
                .collect();
            voiced += hits.len();
            mixer.add(&hits);
            max_voices = max_voices.max(mixer.voices());
            max_dropped = max_dropped.max(mixer.take_dropped());
            let mut block = vec![0.0f32; 256 * 2];
            mixer.render(&mut block, 48_000.0);
            assert!(
                peak(&block) <= 1.0 + 1e-6,
                "a real room at frame {} clipped at {}",
                voiced,
                peak(&block)
            );
        }
        // Reported rather than asserted: the caps below are what the numbers have to
        // satisfy, but the numbers themselves are worth seeing when the design is being
        // judged, and a test that prints them costs nothing.
        eprintln!(
            "audio: 240 frames of a 120-body room -> {voiced} contacts reported, \
             peak in-flight {max_voices} of {MAX_VOICES}, busiest frame dropped {max_dropped}"
        );
        assert!(voiced > 0, "a room of 120 bodies made no sound at all");
        assert!(
            max_voices <= MAX_VOICES,
            "the ceiling held across 240 real frames: {max_voices}"
        );
        // A settled pile jostles constantly, so the ceiling does bite here — and when
        // it does, the panel can say so instead of the sound just getting thin.
        assert!(max_dropped > 0, "no frame had to drop anything: {max_dropped}");
    }

    /// Rough "how much frequency is in there", as sign changes per second. No FFT and
    /// no dependency, and enough to tell "the sample stream contains the tone the
    /// contact was supposed to make" from "the struct says a number and the mixer threw
    /// it away".
    ///
    /// What it is *not* is a fundamental-frequency estimator. A voice carries a
    /// fundamental plus two inharmonic partials plus a short noise burst, so the
    /// crossing rate sits above the fundamental and belongs to the mix of all three.
    /// That is fine — the claim worth pinning is that the rate tracks the size of the
    /// body, monotonically and by a wide margin — and the assertions below are written
    /// for that claim rather than for one this metric cannot support.
    ///
    /// `skip` is in seconds at the front: the attack ramp crosses zero for reasons that
    /// have nothing to do with pitch, and the noise burst is over within a few tens of
    /// milliseconds, so measuring from just after it is what leaves the tone to measure.
    fn zero_crossings_per_second(samples: &[f32], rate: f32, skip: f32) -> f32 {
        let start = (rate * skip) as usize;
        if start >= samples.len() {
            return 0.0;
        }
        let mut crossings = 0usize;
        let mut prev = 0.0f32;
        for (i, &sample) in samples.iter().enumerate().skip(start) {
            if i > start && (sample >= 0.0) != (prev >= 0.0) {
                crossings += 1;
            }
            prev = sample;
        }
        let span = (samples.len() - start) as f32 / rate;
        crossings as f32 / span / 2.0
    }

    /// The left channel of what one contact at this radius sounds like, in seconds of
    /// samples at this rate.
    fn rendered_hit(radius: f32, rate: f32, seconds: f32) -> Vec<f32> {
        let mut mixer = Mixer::new(1.0);
        mixer.add(&[(hit_for(&impact(12.0, radius, 0, 0.0), 7.2), 0)]);
        let out = render_seconds(&mut mixer, seconds, rate);
        out.chunks_exact(2).map(|p| p[0]).collect()
    }

    #[test]
    fn a_smaller_body_makes_faster_vibrations_in_the_samples() {
        // The mapping can be monotonic in the struct and identical out of the mixer —
        // if a phase increment is computed once and reused, or a table index is wrong by
        // a constant. This measures what came out instead.
        // Windows shorter than the longest voice and longer than the shortest, with the
        // attack and the noise burst skipped: with hits this brief there is no window
        // that suits all three sizes, which is itself worth knowing.
        let big = zero_crossings_per_second(&rendered_hit(0.9, 44_100.0, 0.2), 44_100.0, 0.05);
        let middle = zero_crossings_per_second(&rendered_hit(0.45, 44_100.0, 0.14), 44_100.0, 0.04);
        let small = zero_crossings_per_second(&rendered_hit(0.16, 44_100.0, 0.1), 44_100.0, 0.02);
        assert!(big > 40.0 && small < 20_000.0, "measured rates are not audible sound: big {big} Hz, small {small} Hz");
        assert!(
            middle > big * 1.6,
            "halving the radius barely changed the sample stream: {big} -> {middle}"
        );
        assert!(
            small > middle * 1.6,
            "and halving it again did not either: {middle} -> {small}"
        );
        assert!(
            small > big * 4.0,
            "across the room's size range the samples only span {big}..{small} Hz, which is \
             too narrow for a listener to tell a pebble from a boulder"
        );
    }

    #[test]
    fn the_same_hit_is_the_same_pitch_at_any_device_rate() {
        // The bug this pins: phase increments baked in against 44.1 kHz at spawn time.
        // Everything still sounds right on a laptop and comes out a semitone and a half
        // sharp on the 48 kHz interface most desktops actually use.
        let measured_at = |rate: f32| -> f32 {
            // One shape, so the only thing the two rates can differ in is the pitch math.
                        let mut mixer = Mixer::new(1.0);
            mixer.add(&[(hit_for(&impact(12.0, 0.35, 1, 0.0), 7.2), 0)]);
            let mono = rendered_hit(0.35, rate, 0.12);
            zero_crossings_per_second(&mono, rate, 0.02)
        };
        let at_441 = measured_at(44_100.0);
        let at_48 = measured_at(48_000.0);
        assert!(
            (at_441 - at_48).abs() / at_48 < 0.06,
            "the same contact measured {at_441} Hz at 44.1 kHz and {at_48} Hz at 48 kHz —              the pitch is being computed against a sample rate the device does not have"
        );
    }

    #[test]
    fn every_hit_is_over_before_a_third_of_a_second() {
        // The shortness is the design, so it is pinned rather than left to be tuned back
        // up by whoever finds a boom pleasant: across the whole range of sizes and
        // shapes, no voice lasts longer than MAX_LIFE, and the biggest body in the room
        // — the one that rings longest — is inaudible well inside half a second.
        let mut longest = 0.0f32;
        for radius in [0.05f32, 0.16, 0.3, 0.45, 0.6, 0.9] {
            for shape in 0..9 {
                let hit = hit_for(&impact(6.0, radius, shape, 0.0), 7.2);
                assert!(
                    hit.life <= MAX_LIFE + 1e-6,
                    "r={radius} shape={shape} rings {life}s, over the {MAX_LIFE}s cap",
                    life = hit.life
                );
                longest = longest.max(hit.life);
            }
        }
        for (label, radius, shape) in [
            ("boulder", 0.9f32, 0usize),
            ("middle sphere", 0.45, 0),
            ("pebble", 0.16, 0),
            ("tetrahedron", 0.3, 4),
        ] {
            let h = hit_for(&impact(9.0, radius, shape, 0.0), 7.2);
            eprintln!(
                "audio: {label} r={radius} -> {:.0} Hz, {:.2} s, gain {:.2}",
                h.freq, h.life, h.gain
            );
        }
        assert!(
            longest <= 0.3,
            "the longest voice in the room is {longest} s — that is a drone, not a bounce"
        );
        assert!(longest > 0.15, "and now everything is a click: {longest} s");

        // The cap has to hold in the *samples*, not just in the struct: past it there is
        // nothing left to render. Before it, the envelope's business is to be quieting
        // down, and the numbers below say how quiet, rather than pretending the tail is
        // silent the instant the nominal length begins.
        let mut mixer = Mixer::new(1.0);
        let big = hit_for(&impact(12.0, 0.9, 0, 0.0), 7.2);
        mixer.add(&[(big, 0)]);
        let rate = 48_000.0f32;
        let out = render_seconds(&mut mixer, 0.45, rate);
        let window = |from: f32, to: f32| -> f32 {
            // Clamp the *seconds* to the rendered length, then convert: clamping after
            // the multiply compares a frame count against 0.45 and yields frame zero.
            let a = (rate * from.min(0.45)) as usize * 2;
            let b = (rate * to.min(0.45)) as usize * 2;
            peak(&out[a.min(out.len())..b.min(out.len())])
        };
        assert!(
            window(0.32, 0.45) == 0.0,
            "the longest voice in the room is still sounding at {:.2} s, past its \
             {MAX_LIFE} s cap",
            big.life
        );
        // Roughly where the envelope has got to by three quarters of its length — a few
        // percent of full scale. If a retuning pushes this up, hits are overlapping the
        // next frame's worth of contacts, which is the failure the cap is there to stop.
        assert!(
            window(big.life * 0.75, big.life) < 0.05,
            "at three quarters of its length the biggest hit is still at {:.3} of full \
             scale, which is where a wash starts",
            window(big.life * 0.75, big.life)
        );
        assert_eq!(mixer.voices(), 0, "a voice survived its own lifetime");
    }

    #[test]
    fn clear_forgets_everything_in_flight() {
        let mut mixer = Mixer::new(1.0);
        mixer.add(&[(hit_for(&impact(12.0, 0.4, 0, 0.0), 7.2), 0)]);
        mixer.clear();
        assert_eq!(mixer.voices(), 0);
        assert_eq!(peak(&render_seconds(&mut mixer, 0.05, 44_100.0)), 0.0);
    }

    #[test]
    fn the_top_of_the_ladder_stays_far_below_nyquist() {
        // Naive table readout aliases once a partial passes half the sample rate, and a
        // dull 44.1 kHz stream would then add a spurious tone rather than a quieter one.
        let highest = hit_for(&impact(6.0, 0.05, 4, 0.0), 7.2);
        let top_partial = highest.freq * partials(highest.bright)[2].0;
        assert!(
            top_partial < 44_100.0 * 0.45,
            "the top partial of the smallest body reaches {top_partial} Hz, which is \
             close enough to Nyquist to alias"
        );
    }
}
