//! Bringing geometry in from disk, as opposed to generating it in code.
//!
//! The room's nine primitives are all built by [`crate::geom`], which is what
//! lets the port ship with no assets at all. This module is the other direction
//! of travel: a file becomes one [`crate::geom::Mesh`] ready for
//! [`Gfx::upload_mesh`](crate::gfx::Gfx::upload_mesh), and one
//! [`crate::sim::Body`] bouncing around with the rest of them.
//!
//! # What is supported, and why these two
//!
//! `.obj` (Wavefront) and `.stl` (stereolithography) are text-and-triangle
//! formats with no scene semantics, so they can be read inside this process.
//! `.obj` goes through `tobj`, which is worth one small dependency for the parts
//! that are easy to get wrong by hand: the four forms of a face element
//! (`v`, `v/vt`, `v//vn`, `v/vt/vn`), negative indices, and fan-triangulating
//! polygons of arbitrary arity. `.stl` is parsed here because an STL is nothing
//! but a triangle soup, and because the interesting part of supporting it —
//! welding — is the same code path every format needs.
//!
//! `.fbx`, `.blend`, `.usd` and `.usdz` are deliberately not attempted. FBX is a
//! binary AST with several mutually incompatible dialects, `.blend` is a
//! self-describing blob whose layout tracks Blender's own DNA version, and USD is
//! a *composition* system where the visible geometry may be instanced,
//! inherited across layers, or refined from a proxy — so "read the mesh" means
//! implementing a resolver. There is no maintained Rust reader for any of them.
//! The seam is [`import_path`]: a converter can be added there (Blender and
//! assimp both read all six and write OBJ) without the app layer noticing, and
//! everything below — welding, normalising, the index budget, the wireframe
//! edges — is format-agnostic and already works for whatever lands there.
//!
//! # The four rules every import obeys
//!
//! 1. **One body per file.** A `.blend` scene or a multi-group OBJ arrives as a
//!    single merged mesh. The simulation collides everything as a bounding
//!    sphere, so splitting an assembly into one body per part would make its
//!    parts bounce apart from each other, which is worse than coarse.
//! 2. **Normalised to a unit bounding radius**, the convention
//!    `docs/PORT_CONTRACT.md` sets for every primitive. That makes the sim's
//!    collision radius for an import exactly its bounding sphere, and lets an
//!    asset authored in metres, millimetres or inches join the room at a sane
//!    size — the file's units are never trusted, only its proportions.
//! 3. **Inside the renderer's index budget.** [`Mesh::indices`] is a `Vec<u16>`
//!    and `Gfx::upload_mesh` writes 16-bit index buffers, so a mesh larger than
//!    ~65k vertices cannot be drawn without changing the vertex format for every
//!    pipeline. Oversized meshes are welded first (an STL stores one vertex per
//!    corner, which welds down by about 3x) and then, if still too big, have whole
//!    faces sampled out of them at an even stride. Never their tail: files are
//!    written in a spatially coherent order, so dropping the end of the list means
//!    drawing half an object while its collider still covers all of it. What was
//!    left out is counted and reported, never hidden.
//! 4. **`edges` are synthesised from the triangles**, or the wireframe toggle
//!    would show nothing at all for an imported object.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use crate::geom::{Mesh, Vertex};

/// Largest vertex count a mesh can have and still be indexed with `u16`. The
/// format's own ceiling is 65_536; the margin is so a truncated mesh plus its
/// edge list cannot reach an out-of-range index.
pub const MAX_VERTS: usize = 65_532;
/// Most wireframe line segments an imported mesh contributes. This is a
/// legibility and buffer-size bound, not a correctness limit: past a few tens of
/// thousands of edges a wireframe is visual noise whatever the cap is.
pub const MAX_EDGE_PAIRS: usize = 60_000;
/// Refuse to slurp a file this large into memory. Real assets that big want a
/// converter and a decimation pass, not a reader.
pub const MAX_FILE_BYTES: u64 = 64 << 20;
/// Position resolution used when welding, in units of a *normalised* mesh — so
/// it is 1e-5 of the object's own size rather than of whatever units the author
/// used. That is why welding happens after normalising, not before.
const WELD_QUANTUM: f32 = 1.0e-5;

/// Extensions this module can read, for the HUD and for argument checking.
pub const SUPPORTED_EXTENSIONS: [&str; 2] = ["obj", "stl"];

/// One file, read and made safe for this renderer.
#[derive(Clone, Debug)]
pub struct Import {
    /// Normalised to a unit bounding radius, indexed, with `edges` filled in.
    pub mesh: Mesh,
    /// The file's stem, which the HUD lists imported objects by. It cannot live
    /// in [`Mesh::name`]: that is a `&'static str`, and leaking a string per
    /// import to get a nicer GPU buffer label is not a trade worth making.
    pub source: String,
    /// Every repair or reduction the reader had to make, in human words, for the
    /// HUD. A silent change to someone's geometry is the same class of bug as a
    /// wrong constant.
    pub notes: Vec<String>,
    /// Triangles discarded by rule 3, so the note can say "of N".
    pub dropped_triangles: usize,
}

#[derive(Debug)]
pub enum ImportError {
    /// The file could not be read at all.
    Io { path: String, reason: String },
    /// The extension is one this module does not read — see the module note.
    Unsupported { extension: String },
    /// Read fine, but nothing drawable in it.
    Empty { source: String },
    /// Read fine, but the geometry is unusable (all vertices coincident, no
    /// finite positions, larger than the input limit).
    Bad { source: String, reason: String },
}

impl std::fmt::Display for ImportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ImportError::Io { path, reason } => write!(f, "{path}: {reason}"),
            ImportError::Unsupported { extension } => write!(
                f,
                "no reader for .{extension} — .obj and .stl are supported; a converter \
                 (Blender or assimp) can read .{extension} and write .obj"
            ),
            ImportError::Empty { source } => write!(f, "{source}: no triangles"),
            ImportError::Bad { source, reason } => write!(f, "{source}: {reason}"),
        }
    }
}

impl std::error::Error for ImportError {}

/// The message [`import_path`] gives for a file whose format nothing here can
/// read, split out because the window layer wants that answer the moment a file
/// is dropped, before any read is attempted.
pub fn unsupported_message(path: &Path) -> String {
    ImportError::Unsupported {
        extension: extension(path).unwrap_or_else(|| String::from("(none)")),
    }
    .to_string()
}

/// True if this module can read `path` — judged by extension, because that is
/// all the callers have to go on before they read a file.
pub fn is_supported(path: &Path) -> bool {
    extension(path).is_some_and(|ext| SUPPORTED_EXTENSIONS.contains(&ext.as_str()))
}

/// The extensions a caller can offer the user, for a hint line.
pub fn supported_hint() -> String {
    SUPPORTED_EXTENSIONS
        .iter()
        .map(|e| format!(".{e}"))
        .collect::<Vec<_>>()
        .join(" or ")
}

fn extension(path: &Path) -> Option<String> {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
}

fn stem(path: &Path) -> String {
    path.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("import")
        .to_owned()
}

/// Read one file off disk. Dispatches on extension and nothing else.
pub fn import_path(path: &Path) -> Result<Import, ImportError> {
    let source = stem(path);
    let Some(ext) = extension(path) else {
        return Err(ImportError::Unsupported {
            extension: String::from("(none)"),
        });
    };
    // Checked before the filesystem is touched, so an unreadable format reports
    // "no reader for .fbx" rather than "no such file" — the honest answer when
    // the UI offers six extensions and two of them work.
    if !SUPPORTED_EXTENSIONS.contains(&ext.as_str()) {
        return Err(ImportError::Unsupported { extension: ext });
    }
    let meta = std::fs::metadata(path).map_err(|e| ImportError::Io {
        path: path.display().to_string(),
        reason: e.to_string(),
    })?;
    if meta.len() > MAX_FILE_BYTES {
        return Err(ImportError::Bad {
            source,
            reason: format!(
                "{:.1} MB is over the {} MB input limit; decimate or convert it first",
                meta.len() as f64 / (1 << 20) as f64,
                MAX_FILE_BYTES / (1 << 20)
            ),
        });
    }
    match ext.as_str() {
        // OBJ is text; read it as a string so a mis-encoded file fails with a
        // clear "not valid UTF-8" rather than a parse error at byte 90_000.
        "obj" => {
            let text = std::fs::read_to_string(path).map_err(|e| ImportError::Io {
                path: path.display().to_string(),
                reason: e.to_string(),
            })?;
            parse_obj(&text, &source)
        }
        "stl" => {
            let bytes = std::fs::read(path).map_err(|e| ImportError::Io {
                path: path.display().to_string(),
                reason: e.to_string(),
            })?;
            parse_stl(&bytes, &source)
        }
        other => Err(ImportError::Unsupported {
            extension: other.to_owned(),
        }),
    }
}

/// A triangle soup in the file's own units and coordinate convention, before any
/// of the four rules are applied. `tris` index `pos` and `nrm` in lockstep, one
/// entry per corner; `nrm` is empty when the file gave no normals at all.
#[derive(Clone, Debug, Default)]
struct Raw {
    pos: Vec<[f32; 3]>,
    nrm: Vec<[f32; 3]>,
    tri: Vec<u32>,
}

impl Raw {
    fn is_empty(&self) -> bool {
        self.tri.len() < 3
    }

    /// Append one triangle. A missing normal is stored as a zero vector, which is
    /// what [`finish`] tests for when it decides a face's normal has to be
    /// rebuilt from its own winding — counting it here would miss any format that
    /// pushes into the arrays directly.
    fn push_tri(&mut self, corners: [[f32; 3]; 3], normals: Option<[[f32; 3]; 3]>) {
        let base = self.pos.len() as u32;
        for c in 0..3 {
            self.pos.push(corners[c]);
            self.nrm.push(match normals {
                Some(n) => n[c],
                None => [0.0; 3],
            });
        }
        self.tri.extend_from_slice(&[base, base + 1, base + 2]);
    }
}

fn normal_length(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn finite(v: [f32; 3]) -> bool {
    v[0].is_finite() && v[1].is_finite() && v[2].is_finite()
}

/// Cross product, for the faces whose normal the file did not supply correctly.
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// A unit normal for one corner: what the file supplied if it is usable, this
/// face's flat normal if it is not, and a fixed up-vector as the last resort so
/// the shader's `normalize` never has to divide by zero.
fn unit_normal(supplied: [f32; 3], flat: [f32; 3], flat_len: f32) -> [f32; 3] {
    let len = normal_length(supplied);
    if len > f32::EPSILON {
        [supplied[0] / len, supplied[1] / len, supplied[2] / len]
    } else if flat_len > f32::EPSILON {
        [flat[0] / flat_len, flat[1] / flat_len, flat[2] / flat_len]
    } else {
        [0.0, 1.0, 0.0]
    }
}

// --------------------------------------------------------------------------------------
// .obj
// --------------------------------------------------------------------------------------

/// Parse Wavefront text. `source` names the result for the HUD.
///
/// Everything in one OBJ — every `o`, every `g`, every material group — merges
/// into a single mesh (rule 1), so a `usemtl` change never splits the object into
/// bodies that would then collide independently of each other.
pub fn parse_obj(text: &str, source: &str) -> Result<Import, ImportError> {
    // `single_index` makes one corner index address position, normal and texcoord
    // at once, which is what lets this function ignore tobj's parallel index
    // arrays entirely. `triangulate` fans polygons of any arity into triangles,
    // which the renderer needs and hand-rolling gets wrong.
    let options = tobj::LoadOptions {
        triangulate: true,
        single_index: true,
        ..Default::default()
    };
    // Materials are dropped on the floor: this renderer has one colour and one
    // gloss per body, decided by the app rather than by an MTL file. The loader
    // is a closure so a `mtllib` line costs nothing and fails softly.
    let mut reader = std::io::BufReader::new(text.as_bytes());
    let loaded = tobj::load_obj_buf(&mut reader, &options, |_| {
        Err(tobj::LoadError::OpenFileFailed)
    });
    // The material half of the result is ignored even when it is an `Err`: an
    // unreadable MTL sitting next to an OBJ is not a reason to refuse its
    // geometry, and this renderer has one colour per body either way.
    let (models, _materials) = match loaded {
        Ok(pair) => pair,
        Err(e) => return Err(muffle_obj_error(e, source)),
    };

    let mut raw = Raw::default();
    for model in models.iter() {
        let mesh = &model.mesh;
        if mesh.positions.len() % 3 != 0 {
            return Err(ImportError::Bad {
                source: source.to_owned(),
                reason: "truncated vertex data".to_owned(),
            });
        }
        let count = mesh.positions.len() / 3;
        let has_normals = mesh.normals.len() >= mesh.positions.len();
        let base = raw.pos.len() as u32;
        for i in 0..count {
            raw.pos.push([
                mesh.positions[i * 3],
                mesh.positions[i * 3 + 1],
                mesh.positions[i * 3 + 2],
            ]);
            raw.nrm.push(if has_normals {
                [
                    mesh.normals[i * 3],
                    mesh.normals[i * 3 + 1],
                    mesh.normals[i * 3 + 2],
                ]
            } else {
                [0.0; 3]
            });
        }
        raw.tri.extend(mesh.indices.iter().map(|i| base + *i));
    }
    finish(raw, source)
}

fn muffle_obj_error(error: tobj::LoadError, source: &str) -> ImportError {
    ImportError::Bad {
        source: source.to_owned(),
        reason: format!("not readable as Wavefront geometry: {error:?}"),
    }
}

// --------------------------------------------------------------------------------------
// .stl
// --------------------------------------------------------------------------------------

/// Parse an STL, binary or ASCII, from its raw bytes.
///
/// An STL is a triangle soup with a facet normal per face and no shared vertices,
/// which is why it welds down by roughly three times — and why welding is worth
/// doing here rather than at the caller.
pub fn parse_stl(bytes: &[u8], source: &str) -> Result<Import, ImportError> {
    match stl_encoding(bytes) {
        StlEncoding::Binary { triangles } => parse_stl_binary(bytes, triangles, source),
        StlEncoding::Ascii => parse_stl_ascii(bytes, source),
        StlEncoding::Unknown => Err(ImportError::Bad {
            source: source.to_owned(),
            reason: "neither a binary STL header nor an ASCII `solid`".to_owned(),
        }),
    }
}

enum StlEncoding {
    Binary { triangles: usize },
    Ascii,
    Unknown,
}

/// Decide which of the two STL encodings this is.
///
/// The obvious test — does it start with `solid` — is not enough, because real
/// CAD exports write a binary STL whose 80-byte header *begins with the word
/// "solid"* (it is a free-text field, and some toolchains put the object name in
/// it). So the declared triangle count is checked against the file length, which
/// is exact for binary, and only if that fails does the ASCII reading get tried.
fn stl_encoding(bytes: &[u8]) -> StlEncoding {
    if bytes.starts_with(b"solid") && looks_binary_by_count(bytes).is_none() {
        // An ASCII file, whether or not it has any facets in it: "solid" followed
        // by nothing is a legal empty STL and should read as empty rather than as
        // an unrecognisable format.
        return StlEncoding::Ascii;
    }
    if let Some(triangles) = looks_binary_by_count(bytes) {
        return StlEncoding::Binary { triangles };
    }
    // A binary file whose length disagrees with its own count — a truncated
    // export, which happens often enough — still reads, up to the whole records
    // it does have. Both checks below exist for the same reason: a renamed
    // `.fbx` or `.blend` must be refused, not parsed as a very confident load of
    // garbage. So the header's own count has to agree with the file length to
    // within one record, and the first record has to hold finite floats.
    if bytes.len() >= 84 + 50 && !bytes.starts_with(b"solid") {
        let records = (bytes.len() - 84) / 50;
        let declared = u32::from_le_bytes([bytes[80], bytes[81], bytes[82], bytes[83]]) as usize;
        if (declared == records || declared == records + 1) && first_facet_plausible(bytes) {
            return StlEncoding::Binary { triangles: records };
        }
    }
    StlEncoding::Unknown
}

/// Does the file length equal exactly `84 + count * 50` bytes for the triangle
/// count in its header? That arithmetic is what distinguishes a binary STL —
/// including one whose free-text header happens to begin with the word "solid" —
/// from anything else.
fn looks_binary_by_count(bytes: &[u8]) -> Option<usize> {
    if bytes.len() < 84 {
        return None;
    }
    let declared = u32::from_le_bytes([bytes[80], bytes[81], bytes[82], bytes[83]]) as usize;
    (84usize.saturating_add(declared.saturating_mul(50)) == bytes.len())
        .then_some(declared)
}

/// Are the twelve floats of the first record real numbers?
fn first_facet_plausible(bytes: &[u8]) -> bool {
    let at = 84;
    (0..12).all(|k| {
        let at = at + k * 4;
        f32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]]).is_finite()
    })
}

fn parse_stl_binary(bytes: &[u8], triangles: usize, source: &str) -> Result<Import, ImportError> {
    let mut raw = Raw::default();
    for t in 0..triangles {
        let at = 80 + 4 + t * 50;
        if at + 50 > bytes.len() {
            break;
        }
        let floats = |k: usize| f32::from_le_bytes([bytes[at + k * 4], bytes[at + k * 4 + 1], bytes[at + k * 4 + 2], bytes[at + k * 4 + 3]]);
        let normal = [floats(0), floats(1), floats(2)];
        let corners = [
            [floats(3), floats(4), floats(5)],
            [floats(6), floats(7), floats(8)],
            [floats(9), floats(10), floats(11)],
        ];
        if !corners.iter().all(|c| finite(*c)) {
            continue;
        }
        raw.push_tri(corners, Some([normal, normal, normal]));
    }
    finish(raw, source)
}

fn parse_stl_ascii(bytes: &[u8], source: &str) -> Result<Import, ImportError> {
    let text = std::str::from_utf8(bytes).map_err(|_| ImportError::Bad {
        source: source.to_owned(),
        reason: "ascii STL is not valid UTF-8".to_owned(),
    })?;
    // Token-based rather than line-based: writers disagree about newlines and
    // indentation far more than they disagree about keywords.
    let mut tokens = text.split_whitespace().peekable();
    let mut raw = Raw::default();
    let mut skipped = 0usize;
    let triple = |tokens: &mut std::iter::Peekable<std::str::SplitWhitespace<'_>>| -> Option<[f32; 3]> {
        let a = tokens.peek()?.parse::<f32>().ok()?;
        let _ = tokens.next();
        let b = tokens.peek()?.parse::<f32>().ok()?;
        let _ = tokens.next();
        let c = tokens.peek()?.parse::<f32>().ok()?;
        let _ = tokens.next();
        Some([a, b, c])
    };
    while let Some(word) = tokens.next() {
        if word != "facet" {
            continue;
        }
        if tokens.peek().copied() == Some("normal") {
            let _ = tokens.next();
            let _ = triple(&mut tokens);
        }
        if tokens.next() != Some("outer") || tokens.next() != Some("loop") {
            skipped += 1;
            continue;
        }
        let mut corners = [[0.0f32; 3]; 3];
        let mut ok = true;
        for corner in corners.iter_mut() {
            if tokens.next() != Some("vertex") {
                ok = false;
                break;
            }
            match triple(&mut tokens) {
                Some(p) if finite(p) => *corner = p,
                _ => {
                    ok = false;
                    break;
                }
            }
        }
        // Consume the `endloop` / `endfacet` pair whether or not the face was
        // usable, so one broken facet cannot desynchronise the rest of the file.
        while let Some(w) = tokens.peek() {
            if *w == "endloop" || *w == "endfacet" {
                let _ = tokens.next();
            } else {
                break;
            }
        }
        if ok {
            raw.push_tri(corners, None);
        } else {
            skipped += 1;
        }
    }
    let mut import = finish(raw, source)?;
    if skipped > 0 {
        import.notes.push(format!(
            "{skipped} malformed facet(s) skipped — the STL text is not well formed"
        ));
    }
    Ok(import)
}

// --------------------------------------------------------------------------------------
// The format-agnostic part: rules 2, 3 and 4
// --------------------------------------------------------------------------------------

/// Take a triangle soup in whatever units it arrived in and produce a mesh this
/// renderer can draw: finite, unit-bounding-radius, welded, inside the 16-bit
/// index budget, with normals and wireframe edges.
fn finish(raw: Raw, source: &str) -> Result<Import, ImportError> {
    if raw.is_empty() {
        return Err(ImportError::Empty {
            source: source.to_owned(),
        });
    }
    let mut notes = Vec::new();
    let triangles = raw.tri.len() / 3;

    // Rule 2, in f64: a mesh authored a kilometre across and one authored at 2 mm
    // both normalise exactly, and the centre of the bounding box — not the mean
    // of the vertices — is where a bounding sphere wants its centre.
    let mut lo = [f64::MAX; 3];
    let mut hi = [f64::MIN; 3];
    let mut finite_corners = 0usize;
    for p in raw.pos.iter() {
        if !finite(*p) {
            continue;
        }
        finite_corners += 1;
        for k in 0..3 {
            lo[k] = lo[k].min(p[k] as f64);
            hi[k] = hi[k].max(p[k] as f64);
        }
    }
    if finite_corners == 0 {
        return Err(ImportError::Bad {
            source: source.to_owned(),
            reason: "no finite vertex positions".to_owned(),
        });
    }
    let centre = [
        (lo[0] + hi[0]) * 0.5,
        (lo[1] + hi[1]) * 0.5,
        (lo[2] + hi[2]) * 0.5,
    ];
    let extent = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
    // The bounding *radius*, not the half-extent: `Mesh::max_radius` measures
    // vertex distance from the origin, and the sphere has to reach the corner of
    // the box, which is half the diagonal.
    let radius =
        (extent[0] * extent[0] + extent[1] * extent[1] + extent[2] * extent[2]).sqrt() * 0.5;
    if !(radius > 1.0e-9) {
        return Err(ImportError::Bad {
            source: source.to_owned(),
            reason: "every vertex is coincident — nothing to draw".to_owned(),
        });
    }
    // Centring and dividing both stay in f64 and narrow to f32 only at the
    // vertex, so an object authored in 2 mm and one authored in 2 km normalise to
    // the same unit sphere instead of cancelling into mush.
    let scale = 1.0 / radius;

    // Weld. Keyed on position *and* normal on purpose: welding on position alone
    // would average a cube's three face normals at each corner and round off its
    // edges, which is the single most common import bug. Two corners join only
    // when both would shade identically.
    //
    // Quantising the coordinates rather than comparing floats means two written
    // decimals that differ by an ulp weld together, and it means a vertex sitting
    // exactly on a quantum boundary can weld with its neighbour but not with the
    // one on the other side. That is the accepted trade: the worst case is a
    // hairline seam in a wireframe, whereas an exact float compare guarantees
    // seams everywhere.
    let mut welded: Vec<Vertex> = Vec::new();
    let mut keys: HashMap<([i64; 3], [i64; 3]), u16> = HashMap::new();
    let mut indices: Vec<u16> = Vec::with_capacity(raw.tri.len());
    let mut dropped_degenerate = 0usize;
    let mut rebuilt_normals = 0usize;
    let stragglers = raw.tri.len() % 3;

    // The budget is the renderer's, not the file's: `Mesh::indices` is a `Vec<u16>`
    // and `Gfx::upload_mesh` writes 16-bit index buffers. An over-budget mesh is
    // *sampled* rather than cut. Cutting the tail keeps the first slice of the
    // triangle order, which for anything written coherently — a grid, a swept
    // surface, a subdivided sphere — leaves half a model, and worse, leaves the
    // centre of its bounding sphere empty: the collider is the whole object while
    // the picture is only part of it. Striding through the file instead spreads the
    // loss evenly, so the mesh that arrives is a see-through version of all of it.
    //
    // Welding can only ever reduce the vertex count, never raise it, so sizing the
    // stride from the *corner* count cannot overshoot the budget.
    let stride = raw.pos.len().div_ceil(MAX_VERTS).max(1);
    let mut sampled_out = 0usize;
    let mut considered_corners = 0usize;

    // One pass, face by face, so that a face is either drawn whole or not at all:
    // no half-inserted vertex left pointing into a corner that never made it.
    let quant = |v: f32| (v / WELD_QUANTUM).round() as i64;
    for t in 0..triangles {
        if stride > 1 && t % stride != 0 {
            sampled_out += 1;
            continue;
        }
        if welded.len() + 3 > MAX_VERTS {
            // A face can add at most three new corners. With the stride above this
            // should not be reachable; it stays as the net that keeps an
            // out-of-range index impossible even if the two rules drift apart.
            sampled_out += 1;
            continue;
        }
        considered_corners += 3;
        let mut face = [[0.0f32; 3]; 3];
        let mut supplied = [[0.0f32; 3]; 3];
        let mut usable = true;
        for c in 0..3 {
            let slot = raw.tri[t * 3 + c] as usize;
            let p = raw.pos[slot];
            if !finite(p) {
                usable = false;
                break;
            }
            face[c] = [
                ((p[0] as f64 - centre[0]) * scale) as f32,
                ((p[1] as f64 - centre[1]) * scale) as f32,
                ((p[2] as f64 - centre[2]) * scale) as f32,
            ];
            supplied[c] = raw.nrm[slot];
        }
        if !usable {
            dropped_degenerate += 1;
            continue;
        }
        // The flat normal of this face, for any corner whose own normal the file
        // did not supply usefully. STL facet normals are flat already; an OBJ with
        // a zero-length `vn` is asking to be lit by its face.
        let flat = cross(
            [
                face[1][0] - face[0][0],
                face[1][1] - face[0][1],
                face[1][2] - face[0][2],
            ],
            [
                face[2][0] - face[0][0],
                face[2][1] - face[0][1],
                face[2][2] - face[0][2],
            ],
        );
        let flat_len = normal_length(flat);
        if flat_len <= 1.0e-12 {
            // Zero area, so there is nothing to shade and no normal to be had from
            // the winding. Judged on the normalised mesh, where a unit bounding
            // radius turns this into a scale-free test.
            dropped_degenerate += 1;
            continue;
        }
        let mut corner_idx = [0u16; 3];
        let mut rebuilt_here = false;
        for c in 0..3 {
            rebuilt_here |= normal_length(supplied[c]) <= f32::EPSILON;
            let n = unit_normal(supplied[c], flat, flat_len);
            let key = (
                [quant(face[c][0]), quant(face[c][1]), quant(face[c][2])],
                [quant(n[0]), quant(n[1]), quant(n[2])],
            );
            corner_idx[c] = match keys.get(&key) {
                Some(existing) => *existing,
                None => {
                    let fresh = welded.len() as u16;
                    welded.push(Vertex {
                        position: face[c],
                        normal: n,
                    });
                    keys.insert(key, fresh);
                    fresh
                }
            };
        }
        if corner_idx[0] == corner_idx[1]
            || corner_idx[1] == corner_idx[2]
            || corner_idx[0] == corner_idx[2]
        {
            // Collapsed to a line or a point by the weld: nothing to shade.
            dropped_degenerate += 1;
            continue;
        }
        if rebuilt_here {
            rebuilt_normals += 1;
        }
        indices.extend_from_slice(&corner_idx);
    }
    if stragglers > 0 {
        dropped_degenerate += 1;
    }

    if indices.len() < 3 {
        return Err(ImportError::Bad {
            source: source.to_owned(),
            reason: "nothing left after removing degenerate faces".to_owned(),
        });
    }
    if sampled_out > 0 {
        notes.push(format!(
            "this renderer indexes with u16: 1 face in {stride} was sampled out, so \
             {sampled_out} of {triangles} faces are not drawn and the mesh arrived as \
             {} vertices / {} faces — decimate the source for a complete object",
            welded.len(),
            indices.len() / 3
        ));
    }
    if dropped_degenerate > 0 {
        notes.push(format!(
            "{dropped_degenerate} degenerate or non-finite triangle(s) removed"
        ));
    }
    if rebuilt_normals > 0 {
        notes.push(format!(
            "normals rebuilt from the faces for {rebuilt_normals} triangle(s) — the \
             file's were missing or zero-length"
        ));
    }
    // Only the corners that survived the sampling are counted here: a note that
    // says "welded 173400 corners to 65530" when five sixths of the file was never
    // considered is a number nobody can reconcile with what they see.
    if welded.len() < considered_corners {
        notes.push(format!(
            "welded {considered_corners} corners to {} vertices (shared position \
             and normal)",
            welded.len()
        ));
    }

    // Rule 4: wireframe edges. Each triangle contributes its outline; a HashSet of
    // (low, high) pairs means an edge shared by two faces is drawn once.
    let mut edges: Vec<u16> = Vec::new();
    let mut seen_edges: HashSet<(u16, u16)> = HashSet::new();
    let mut capped = false;
    for f in indices.chunks(3) {
        for k in 0..3 {
            let (a, b) = (f[k] as u16, f[(k + 1) % 3] as u16);
            let pair = if a < b { (a, b) } else { (b, a) };
            if seen_edges.insert(pair) {
                if edges.len() / 2 >= MAX_EDGE_PAIRS {
                    capped = true;
                    break;
                }
                edges.push(a);
                edges.push(b);
            }
        }
        if capped {
            break;
        }
    }
    if capped {
        notes.push(format!(
            "wireframe edges capped at {MAX_EDGE_PAIRS} segments; the solid mesh is unaffected"
        ));
    }

    Ok(Import {
        mesh: Mesh {
            // Every built-in mesh names itself; an import cannot carry its file
            // name here (`&'static str`), so the GPU label says what it is rather
            // than which file it came from.
            name: "imported",
            vertices: welded,
            indices: Some(indices),
            edges,
        },
        source: source.to_owned(),
        notes,
        dropped_triangles: sampled_out,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unit cube with an explicit normal per face, one `f` line per side.
    const CUBE_OBJ: &str = "\
# a cube, written the way a CAD tool writes it
o box
v -1 -1 -1
v  1 -1 -1
v  1  1 -1
v -1  1 -1
v -1 -1  1
v  1 -1  1
v  1  1  1
v -1  1  1
vn 0 0 -1
vn 0 0  1
vn 0 -1 0
vn 0  1 0
vn -1 0 0
vn  1 0 0
f 1//1 2//1 3//1 4//1
f 5//2 8//2 7//2 6//2
f 1//3 5//3 6//3 2//3
f 2//4 6//4 7//4 3//4
f 3//5 7//5 8//5 4//5
f 4//6 8//6 5//6 1//6
";

    fn cube() -> Import {
        parse_obj(CUBE_OBJ, "cube").expect("the cube fixture should parse")
    }

    /// The invariants every import owes the renderer and the simulation.
    fn assert_import_is_sane(import: &Import) {
        let mesh = &import.mesh;
        assert!(!mesh.vertices.is_empty(), "no vertices");
        assert!(mesh.indices.is_some(), "an import must be indexed");
        let indices = mesh.indices.clone().unwrap();
        assert_eq!(indices.len() % 3, 0, "a ragged index buffer");
        assert!(
            mesh.vertices.len() <= MAX_VERTS,
            "{} vertices over the u16 budget",
            mesh.vertices.len()
        );
        for i in indices.iter().chain(mesh.edges.iter()) {
            assert!(
                (*i as usize) < mesh.vertices.len(),
                "index {i} runs past the {} vertices",
                mesh.vertices.len()
            );
        }
        // Rule 2: unit bounding radius, which is what makes the sim's collision
        // radius the true bounding sphere.
        let radius = mesh.max_radius();
        assert!(
            (radius - 1.0).abs() < 1.0e-4,
            "bounding radius should normalise to 1.0, got {radius}"
        );
        for v in mesh.vertices.iter() {
            let len = normal_length(v.normal);
            assert!(
                (len - 1.0).abs() < 1.0e-3,
                "normal {:?} is not unit length ({len})",
                v.normal
            );
            assert!(finite(v.position) && finite(v.normal), "non-finite vertex {v:?}");
        }
        // Rule 4: the wireframe has something to draw.
        assert!(!mesh.edges.is_empty(), "no edges, so wireframe shows nothing");
        assert_eq!(mesh.edges.len() % 2, 0, "an odd edge list");
    }

    #[test]
    fn a_wavefront_cube_becomes_one_indexed_unit_mesh() {
        let import = cube();
        assert_import_is_sane(&import);
        // Six quads fan out to twelve triangles — the polygon triangulation that
        // hand-rolling OBJ tends to get wrong.
        assert_eq!(import.mesh.triangle_count(), 12);
        // Welded on position *and* normal, so a cube keeps its eight corners three
        // times over rather than smoothing its own edges off.
        assert_eq!(import.mesh.vertex_count(), 24, "a cube should stay faceted");
        // 6 faces x (4 outline + 1 diagonal) = 30 distinct segments.
        assert_eq!(import.mesh.edge_pair_count(), 30);
        assert!(import.dropped_triangles == 0, "nothing should be dropped");
        // Nothing was *repaired*: no face removed, no normal invented, nothing
        // sampled out or capped. A welding note is expected and is not a repair —
        // 36 corner references from the `v//vn` pairs collapse to 24 vertices that
        // shade identically, which changes the representation and not the shape.
        for note in import.notes.iter() {
            for repaired in ["sampled out", "degenerate", "rebuilt", "capped"] {
                assert!(
                    !note.contains(repaired),
                    "a well-formed OBJ should not need {repaired:?}: {note}"
                );
            }
        }
        assert!(
            import
                .notes
                .iter()
                .any(|n| n.contains("welded 36 corners to 24 vertices")),
            "notes: {:?}",
            import.notes
        );
    }

    #[test]
    fn every_import_path_obeys_the_same_four_rules() {
        // One assertion set, run against every reader, so a new format cannot
        // quietly skip normalisation or the index budget.
        assert_import_is_sane(&cube());
        let no_normals = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n";
        assert_import_is_sane(&parse_obj(no_normals, "tri").expect("a bare triangle parses"));
        let stl = binary_stl(&[(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        )]);
        assert_import_is_sane(&parse_stl(&stl, "tri").expect("a one-triangle STL parses"));
    }

    #[test]
    fn obj_index_forms_that_hand_rolled_parsers_get_wrong() {
        // Negative indices (counting back from the last declared vertex), `v/vt`
        // with no normals, and a lone triangle.
        let text = "\
v -1 -1 -1
v  1 -1 -1
v  0  1 -1
vt 0 0
vt 1 0
vt 0 1
f -3/1 -2/2 -1/3
";
        let import = parse_obj(text, "tri").expect("negative indices parse");
        assert_eq!(import.mesh.triangle_count(), 1);
        assert_import_is_sane(&import);
        // No `vn` anywhere: the face normal stands in, and says so.
        assert!(
            import.notes.iter().any(|n| n.contains("normals rebuilt")),
            "notes: {:?}",
            import.notes
        );
    }

    #[test]
    fn several_objects_in_one_file_arrive_as_one_body() {
        // Rule 1. A multi-object file must not become several bodies that collide
        // as separate spheres and shove each other apart.
        let text = format!("{CUBE_OBJ}o second\nv 4 4 4\nv 5 4 4\nv 4 5 4\nf 1 2 3\n");
        let import = parse_obj(&text, "two").expect("two objects parse");
        assert_eq!(import.mesh.triangle_count(), 13);
        assert_eq!(import.source, "two");
        assert_import_is_sane(&import);
    }

    #[test]
    fn a_missing_mtl_does_not_refuse_the_geometry() {
        let text = format!("mtllib absent.mtl\nusemtl painted\n{CUBE_OBJ}");
        let import = parse_obj(&text, "cube").expect("the mesh survives an unreadable MTL");
        assert_eq!(import.mesh.triangle_count(), 12);
    }

    #[test]
    fn normalisation_ignores_the_units_the_author_used() {
        // The same cube in millimetres and in kilometres must land on the same
        // unit sphere: file units are never trusted, only proportions.
        let metres = "v -0.001 -0.001 -0.001\nv 0.001 0.001 0.001\nv -0.001 0.003 -0.001\nvn 0 0 1\nf 1//1 2//1 3//1\n";
        let kilometres = metres.replace("0.001", "1000.0").replace("-0.001", "-1000.0");
        let small = parse_obj(metres, "mm").expect("parses");
        let large = parse_obj(&kilometres, "km").expect("parses");
        assert!(
            (small.mesh.max_radius() - large.mesh.max_radius()).abs() < 1.0e-4,
            "{:?} vs {:?}",
            small.mesh.max_radius(),
            large.mesh.max_radius()
        );
    }

    #[test]
    fn a_sliver_still_normalises_and_loses_its_degenerate_faces() {
        // Two of the three faces are flat on the plane z=0 and collapse.
        let text = "v 0 0 0\nv 1 0 0\nv 2 0 0\nv 0 1 0\nf 1 2 4\nf 2 3 4\nf 1 2 3\n";
        let import = parse_obj(text, "sliver").expect("one real face is enough");
        assert_eq!(import.mesh.triangle_count(), 2);
        assert!(
            import
                .notes
                .iter()
                .any(|n| n.contains("degenerate or non-finite"))
        );
        assert_import_is_sane(&import);
    }

    #[test]
    fn coincident_vertices_are_refused_not_normalised_by_zero() {
        let text = "v 1 1 1\nv 1 1 1\nv 1 1 1\nf 1 2 3\n";
        match parse_obj(text, "dot") {
            Err(ImportError::Bad { reason, .. }) => {
                assert!(reason.contains("coincident"), "reason: {reason}");
            }
            other => panic!("expected a rejection of a zero-size mesh, got {other:?}"),
        }
    }

    #[test]
    fn comments_and_points_only_files_are_an_empty_error() {
        match parse_obj("# nothing but a comment\n# and another\n", "empty") {
            Err(ImportError::Empty { .. }) => {}
            other => panic!("expected Empty, got {other:?}"),
        }
    }

    #[test]
    fn non_finite_positions_never_reach_the_vertex_buffer() {
        // `not a number` parses as a float in some readers and as garbage in
        // others; either way it must not survive into a vertex.
        let text = "v nan 0 0\nv 0 1 0\nv 1 0 0\nv 0 0 1\nf 1 2 3\nf 2 3 4\n";
        match parse_obj(text, "nan") {
            // Either answer is acceptable: a reader that refuses the file outright
            // is as good as one that drops the poisoned face. What is not
            // acceptable is a mesh that carries a NaN into a vertex buffer.
            Ok(import) => {
                assert!(import.mesh.triangle_count() >= 1);
                assert_import_is_sane(&import);
            }
            Err(ImportError::Bad { .. }) => {}
            Err(other) => panic!("unexpected error for a NaN vertex: {other}"),
        }
    }

    // --------------------------------------------------------------------------------
    // .stl
    // --------------------------------------------------------------------------------

    /// Build a binary STL in memory: an 80-byte header, a uint32 count, then one
    /// 50-byte record per facet (normal, three corners, two attribute bytes).
    fn binary_stl(facets: &[([f32; 3], [f32; 3], [f32; 3])]) -> Vec<u8> {
        binary_stl_named(facets, b"generated")
    }

    fn binary_stl_named(
        facets: &[([f32; 3], [f32; 3], [f32; 3])],
        header: &[u8],
    ) -> Vec<u8> {
        let mut out = vec![0u8; 80];
        let take = header.len().min(80);
        out[..take].copy_from_slice(&header[..take]);
        out.extend_from_slice(&(facets.len() as u32).to_le_bytes());
        for (a, b, c) in facets {
            // Normal from the winding, which is what a correct writer puts here.
            let n = cross(
                [b[0] - a[0], b[1] - a[1], b[2] - a[2]],
                [c[0] - a[0], c[1] - a[1], c[2] - a[2]],
            );
            let len = normal_length(n);
            let n = if len > 0.0 {
                [n[0] / len, n[1] / len, n[2] / len]
            } else {
                [0.0; 3]
            };
            // Exactly 50 bytes per record: normal, three corners (48) and the
            // two-byte attribute that the reader has to skip over.
            for v in [n, *a, *b, *c] {
                for k in 0..3 {
                    out.extend_from_slice(&v[k].to_le_bytes());
                }
            }
            out.extend_from_slice(&0u16.to_le_bytes());
        }
        out
    }

    /// The twelve facets of a unit cube, one outward normal per face, exactly as a
    /// slicer would write them: no shared vertices.
    fn cube_facets() -> Vec<([f32; 3], [f32; 3], [f32; 3])> {
        let c = [
            [-1.0f32, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ];
        let mut f: Vec<([f32; 3], [f32; 3], [f32; 3])> = Vec::new();
        fn quad(
            c: &[[f32; 3]; 8],
            a: usize,
            b: usize,
            d: usize,
            e: usize,
            out: &mut Vec<([f32; 3], [f32; 3], [f32; 3])>,
        ) {
            out.push((c[a], c[b], c[d]));
            out.push((c[a], c[d], c[e]));
        }
        // The six faces of the cube, each split into two triangles, exactly as a
        // slicer writes them: twelve independent facets, no shared corners.
        quad(&c, 0, 3, 2, 1, &mut f);
        quad(&c, 4, 5, 6, 7, &mut f);
        quad(&c, 0, 1, 5, 4, &mut f);
        quad(&c, 1, 2, 6, 5, &mut f);
        quad(&c, 2, 3, 7, 6, &mut f);
        quad(&c, 3, 0, 4, 7, &mut f);
        f
    }

    #[test]
    fn a_binary_stl_welds_down_and_stays_faceted() {
        let bytes = binary_stl(&cube_facets());
        let import = parse_stl(&bytes, "cube").expect("a well-formed binary STL parses");
        assert_eq!(import.mesh.triangle_count(), 12);
        // 36 written corners, 24 after welding on (position, normal).
        assert_eq!(import.mesh.vertex_count(), 24);
        assert!(import
            .notes
            .iter()
            .any(|n| n.contains("welded 36 corners to 24")));
        assert_import_is_sane(&import);
    }

    #[test]
    fn binary_and_ascii_stl_agree_on_the_same_cube() {
        let facets = cube_facets();
        let bytes = binary_stl(&facets);
        let mut text = String::from("solid cube\n");
        for (a, b, c) in facets.iter() {
            let n = cross(
                [b[0] - a[0], b[1] - a[1], b[2] - a[2]],
                [c[0] - a[0], c[1] - a[1], c[2] - a[2]],
            );
            text.push_str(&format!(
                "  facet normal {} {} {}\n    outer loop\n      vertex {} {} {}\n      vertex {} {} {}\n      vertex {} {} {}\n    endloop\n  endfacet\n",
                n[0], n[1], n[2], a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]
            ));
        }
        text.push_str("endsolid cube\n");
        let ascii = parse_stl(text.as_bytes(), "cube").expect("ascii STL parses");
        let binary = parse_stl(&bytes, "cube").expect("binary STL parses");
        assert_eq!(ascii.mesh.triangle_count(), binary.mesh.triangle_count());
        assert_eq!(ascii.mesh.vertex_count(), binary.mesh.vertex_count());
        assert_import_is_sane(&ascii);
    }

    #[test]
    fn a_binary_stl_whose_header_starts_with_solid_is_still_read_as_binary() {
        // The classic trap: the 80-byte header is free text and plenty of writers
        // put the object name in it, so `solid` at byte 0 proves nothing.
        let bytes = binary_stl_named(&cube_facets(), b"solid generated by a CAD tool");
        assert!(bytes.starts_with(b"solid"));
        let import = parse_stl(&bytes, "trap").expect("the count beats the prefix");
        assert_eq!(import.mesh.triangle_count(), 12);
    }

    #[test]
    fn a_zero_facet_normal_is_rebuilt_from_the_winding() {
        // One correct 50-byte record whose normal slot is all zeros, which plenty
        // of exporters do instead of computing it.
        let mut bytes = vec![0u8; 84];
        bytes[80..84].copy_from_slice(&1u32.to_le_bytes());
        for v in [
            [0.0f32, 0.0, 0.0], // the supplied (useless) normal
            [0.0f32, 0.0, 0.0], // corner
            [1.0f32, 0.0, 0.0], // corner
            [0.0f32, 1.0, 0.0], // corner
        ] {
            for k in 0..3 {
                bytes.extend_from_slice(&v[k].to_le_bytes());
            }
        }
        bytes.extend_from_slice(&0u16.to_le_bytes());
        assert_eq!(bytes.len(), 84 + 50, "the record is 50 bytes");
        let import = parse_stl(&bytes, "broken").expect("a zero normal is recoverable");
        assert_eq!(import.mesh.triangle_count(), 1);
        assert!(import
            .notes
            .iter()
            .any(|n| n.contains("normals rebuilt")));
        assert_import_is_sane(&import);
    }

    #[test]
    fn a_renamed_binary_blob_is_not_misparsed_as_geometry() {
        // Someone renames model.fbx to model.stl. Refusing is better than
        // importing a confident load of garbage.
        let mut junk = vec![0u8; 400];
        for (i, b) in junk.iter_mut().enumerate() {
            *b = (i * 7 % 251) as u8;
        }
        // Not UTF-8 text and not a plausible first record.
        junk[84..88].copy_from_slice(b"FBX\x00");
        match parse_stl(&junk, "renamed") {
            Err(ImportError::Bad { .. }) | Err(ImportError::Empty { .. }) => {}
            other => panic!("junk should not import as a mesh, got {:?}", other.map(|i| i.mesh.triangle_count())),
        }
    }

    #[test]
    fn empty_and_truncated_stl_files_are_errors_not_surprises() {
        match parse_stl(b"solid nothing\nendsolid nothing\n", "empty") {
            Err(ImportError::Empty { .. }) => {}
            other => panic!("expected Empty for an empty ASCII STL, got {other:?}"),
        }
        let mut bytes = binary_stl(&cube_facets());
        bytes.truncate(bytes.len() - 37); // cut the last record in half
        let import = parse_stl(&bytes, "short").expect("whole records still import");
        assert_eq!(import.mesh.triangle_count(), 11, "a half record is dropped");
    }

    #[test]
    fn an_over_budget_mesh_is_sampled_through_the_file_not_cut_from_the_front() {
        // 30 000 faces that cannot weld (every corner a distinct position), split
        // into two halves sitting in different places: the near half at z = 0 and
        // the far half at z = 100. Keeping "the first N faces" would draw one half
        // and none of the other, so the assertions below are about spread and not
        // only about size.
        let mut raw = Raw::default();
        for i in 0..30_000u32 {
            let far = i >= 15_000;
            let x = (i % 15_000) as f32 * 0.01;
            let z = if far { 100.0 } else { 0.0 };
            raw.push_tri(
                [[x, 0.0, z], [x + 0.005, 0.01, z], [x, 0.02, z]],
                Some([[0.0, 0.0, 1.0]; 3]),
            );
        }
        let import = finish(raw, "big").expect("a huge mesh still imports");
        assert!(
            import.mesh.vertex_count() <= MAX_VERTS,
            "{} vertices over the u16 budget",
            import.mesh.vertex_count()
        );
        assert!(
            import.dropped_triangles > 0,
            "nothing reported as sampled out: {} faces drawn, {} vertices, notes {:?}",
            import.mesh.triangle_count(),
            import.mesh.vertex_count(),
            import.notes
        );
        assert_eq!(
            import.mesh.triangle_count() + import.dropped_triangles,
            30_000,
            "the arithmetic of what survived and what was left out has to add up"
        );
        assert!(
            import
                .notes
                .iter()
                .any(|n| n.contains("sampled out") && n.contains("u16")),
            "notes: {:?}",
            import.notes
        );

        // And both halves are in the result. A front-cut would leave the kept z
        // range a hair's width instead of the slice of normalised space that spans
        // the gap between the two halves.
        let z_range = import.mesh.vertices.iter().fold(
            (f32::MAX, f32::MIN),
            |(lo, hi), v| (lo.min(v.position[2]), hi.max(v.position[2])),
        );
        assert!(
            z_range.1 - z_range.0 > 0.5,
            "only {:.3} of the z extent survived, so one half was cut off",
            z_range.1 - z_range.0
        );
        assert_import_is_sane(&import);
    }

    #[test]
    fn wireframe_edges_are_capped_without_touching_the_solid_mesh() {
        // Enough faces that the outline segments of the ones that *do* fit still
        // overflow the edge cap: 40 100 non-welding faces, half of them sampled
        // out by the vertex budget, leaving ~60 150 distinct segments against a cap
        // of 60 000.
        let mut raw = Raw::default();
        for i in 0..40_100u32 {
            let x = i as f32 * 0.01;
            raw.push_tri(
                [[x, 0.0, 0.0], [x + 0.01, 0.0, 0.0], [x, 0.02, 0.0]],
                Some([[0.0, 0.0, 1.0]; 3]),
            );
        }
        let import = finish(raw, "dense").expect("dense mesh imports");
        assert!(
            import.mesh.edge_pair_count() <= MAX_EDGE_PAIRS,
            "{} edge pairs over the cap",
            import.mesh.edge_pair_count()
        );
        assert_eq!(import.mesh.index_count(), import.mesh.triangle_count() * 3);
        assert!(import
            .notes
            .iter()
            .any(|n| n.contains("wireframe edges capped")));
    }

    #[test]
    fn unsupported_formats_are_named_before_the_filesystem_is_touched() {
        // A path that does not exist still gets "no reader for .fbx", which is the
        // honest answer when the menu offers six formats and two work.
        let path = Path::new("/nonexistent/something.fbx");
        assert!(!is_supported(path));
        match import_path(path) {
            Err(ImportError::Unsupported { extension }) => assert_eq!(extension, "fbx"),
            other => panic!("expected Unsupported, got {other:?}"),
        }
        assert!(is_supported(Path::new("a/b/c.OBJ")));
        assert!(is_supported(Path::new("c.stl")));
        assert_eq!(supported_hint(), ".obj or .stl");
    }

    #[test]
    fn a_real_file_on_disk_reads_through_the_same_path_as_the_fixtures() {
        let dir = std::env::temp_dir().join(format!("bouncing-room-import-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        std::fs::write(dir.join("cube.obj"), CUBE_OBJ).expect("write the obj");
        let from_file = import_path(&dir.join("cube.obj")).expect("read from disk");
        assert_eq!(from_file.mesh.triangle_count(), 12);
        assert_eq!(from_file.source, "cube");
        assert_import_is_sane(&from_file);
        std::fs::remove_dir_all(&dir).ok();
    }
}
