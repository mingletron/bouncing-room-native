//! Bringing geometry in from disk, as opposed to generating it in code.
//!
//! The room's nine primitives are all built by [`crate::geom`], which is what
//! lets the port ship with no assets at all. This module is the other direction
//! of travel: a file becomes one [`crate::geom::Mesh`] ready for
//! [`Gfx::upload_mesh`](crate::gfx::Gfx::upload_mesh), and one
//! [`crate::sim::Body`] bouncing around with the rest of them.
//!
//! # What is supported, and how
//!
//! Four formats are read inside this process: `.obj` (Wavefront) and `.stl`
//! (stereolithography) are text-and-triangle formats with no scene semantics;
//! `.gltf` and `.glb` are read through the `gltf` crate, whose buffers this module
//! resolves itself because the crate does not touch the filesystem. `.obj` uses
//! `tobj` rather than a hand parser for the parts that go wrong quietly: the four
//! forms of a face element (`v`, `v/vt`, `v//vn`, `v/vt/vn`), negative indices, and
//! fan-triangulating polygons of arbitrary arity. `.stl` is parsed here because an
//! STL is a triangle soup, and because the interesting part of supporting it —
//! welding — is the path every format shares.
//!
//! `.fbx`, `.blend` and the USD family (`.usd`, `.usda`, `.usdc`, `.usdz`) are
//! **converted, not parsed**, and the distinction is honest rather than a hedge:
//! FBX is a binary AST with several mutually incompatible dialects, `.blend` is a
//! self-describing blob whose layout tracks Blender's own DNA version, and USD is a
//! *composition* system where the geometry you see may be instanced, inherited
//! across layers or refined from a proxy — so "read the mesh" means implementing a
//! resolver. No maintained Rust reader exists for any of them. So the file is run
//! through Blender headless, which writes a self-contained `.glb`, and the result
//! goes into the same glTF reader. Everything downstream of that — welding,
//! normalising, the index budget, edges, hulls — is format-agnostic and never
//! notices the difference.
//!
//! What the tools can actually do was measured, not assumed:
//!
//! | Format | Recipe | Fallback |
//! | --- | --- | --- |
//! | `.blend` | `blender -b file.blend --python-expr` → GLB | none |
//! | `.fbx` | `blender -b --python-expr` with `wm.fbx_import` | `assimp export` |
//! | `.usd`, `.usda`, `.usdc`, `.usdz` | `blender -b --python-expr` with `wm.usd_import` | none |
//!
//! assimp is a fallback for FBX only, and the reason is worth recording because it
//! looks like a bug when you first hit it: `assimp export modern.blend` fails with
//! "BLENDER magic bytes are missing, couldn't find GZIP header either", because
//! assimp's Blender loader predates the file format Blender 2.8+ writes. And
//! `blender -b file.fbx` — passing the file as an argument, the way `.blend` is
//! passed — fails with "File format is not supported", because FBX is an *addon
//! import* rather than something the .blend loader understands; the import operator
//! has to be called explicitly. Both of those are why the recipes differ per
//! format rather than one command line for everything.
//!
//! A machine with no Blender and no assimp gets an error that says so and names the
//! variables that would point at them (`BLENDER`, `ASSIMP`); nothing is installed
//! behind the user's back. Conversion is slow — seconds, not milliseconds — so the
//! window layer runs it on a worker thread rather than in the frame — the same
//! shape it already uses for the file dialog; the probe calls the blocking path
//! directly, which is what makes a headless `.fbx` check possible.
//!
//! Conversion also loses whatever the source format carries that glTF cannot say,
//! and adds whatever the converter decides about units and up-axis. That is reported
//! in `Import::notes` rather than left silent.
//!
//! # The four rules every import obeys
//!
//! 1. **One body per file.** A `.blend` scene or a multi-group OBJ arrives as a
//!    single merged mesh. The simulation collides everything as a bounding
//!    sphere, so splitting an assembly into one body per part would make its
//!    parts bounce apart from each other, which is worse than coarse.
//! 2. **Normalised to a unit bounding radius**, the convention
//!    `docs/PORT_CONTRACT.md` sets for every primitive, and measured the same way:
//!    the farthest *vertex* from the centre, not the corner of a bounding box no
//!    vertex may reach. That makes the sim's collision radius for an import
//!    exactly its bounding sphere rather than a slight oversize — an oversize
//!    would leave the drawn mesh floating off the walls its collider hits. It also
//!    lets an asset authored in metres, millimetres or inches join the room at a
//!    sane size: the file's units are never trusted, only its proportions.
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
use std::path::{Path, PathBuf};

use crate::hull::Hull;
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
pub const SUPPORTED_EXTENSIONS: [&str; 4] = ["obj", "stl", "gltf", "glb"];
/// Extensions with no reader in this process, which [`import_path`] accepts only
/// because it converts them to glTF with an external tool first. See the module
/// note for which tool converts which format, and what was measured about it.
pub const CONVERTIBLE_EXTENSIONS: [&str; 6] = ["fbx", "blend", "usd", "usda", "usdc", "usdz"];

/// Every extension [`is_supported`] accepts, in the order a file dialog should
/// show them: the formats this process reads, then the ones that need a tool. The
/// window layer hands this to `rfd`, and it has to be the same set the drop
/// handler admits — a dialog that hides `.fbx` while a dragged `.fbx` imports is
/// the same inconsistency as the reverse, only quieter.
pub fn importable_extensions() -> Vec<&'static str> {
    SUPPORTED_EXTENSIONS
        .into_iter()
        .chain(CONVERTIBLE_EXTENSIONS)
        .collect()
}

/// Wall-clock ceiling on one conversion.
///
/// Measured on this machine: a cube through Blender 5.0 is about a second per launch,
/// which is why the window layer puts conversion on a worker thread instead of in the
/// frame. A heavy scene was never measured, and this ceiling is not for one — it is
/// there because a tool that wedges on a malformed file would otherwise hold a worker
/// thread until the process ends. 120 s is far above anything this machine has
/// produced and short enough that a wedged Blender gets reported rather than waited on
/// forever.
pub const MAX_CONVERT_SECONDS: u64 = 120;

/// True for a format that must be converted before it can be read. The window
/// layer asks this before deciding whether an import is cheap enough to do inside
/// the frame or belongs on a worker thread — a conversion takes seconds, and the
/// frame thread is the one that draws.
pub fn needs_converter(path: &Path) -> bool {
    extension(path).is_some_and(|e| CONVERTIBLE_EXTENSIONS.contains(&e.as_str()))
}

/// Where Blender lives. `BLENDER` names it explicitly, then `blender` on `PATH`,
/// then the macOS application bundle. `None` is not an error: it means this
/// machine cannot convert, and the caller says so in words rather than installing
/// anything behind the user.
pub fn blender_program() -> Option<PathBuf> {
    if let Ok(explicit) = std::env::var("BLENDER") {
        let p = PathBuf::from(explicit);
        if p.exists() {
            return Some(p);
        }
    }
    for name in ["blender", "blender-5.0", "blender-lts"] {
        if which(name) {
            return Some(PathBuf::from(name));
        }
    }
    let app = PathBuf::from("/Applications/Blender.app/Contents/MacOS/Blender");
    app.exists().then_some(app)
}

/// Where assimp lives, for the FBX fallback. `ASSIMP` names it explicitly.
pub fn assimp_program() -> Option<PathBuf> {
    if let Ok(explicit) = std::env::var("ASSIMP") {
        let p = PathBuf::from(explicit);
        if p.exists() {
            return Some(p);
        }
    }
    which("assimp").then(|| PathBuf::from("assimp"))
}

/// Is `name` on `PATH`? Hand-rolled so this module adds no dependency: walk the
/// `PATH` entries and look for an executable file of that name.
fn which(name: &str) -> bool {
    let Some(paths) = std::env::var_os("PATH") else {
        return false;
    };
    std::env::split_paths(&paths).any(|dir| {
        let candidate = dir.join(name);
        candidate.is_file()
            && candidate
                .metadata()
                .map(|m| {
                    #[cfg(unix)]
                    let executable = {
                        use std::os::unix::fs::PermissionsExt;
                        m.permissions().mode() & 0o111 != 0
                    };
                    #[cfg(not(unix))]
                    let executable = true;
                    executable
                })
                .unwrap_or(false)
    })
}

/// Run a converter, killing it if it outlives [`MAX_CONVERT_SECONDS`].
///
/// `std::process` has no wait-with-timeout, so this polls `try_wait`; a 50 ms
/// grain is invisible beside a Blender start-up. The tool's own output is dropped:
/// a converter writing to a pipe nobody drains is another way a conversion becomes
/// a hang, and its progress bars are not something a HUD can show.
///
/// Arguments and env values are owned `OsString`s rather than `&OsStr` borrows,
/// which costs a handful of clones per conversion and buys the one thing that
/// matters here: an argument built inside a match arm — the Python source of a
/// `--python-expr` recipe — outlives the arm that made it.
fn run_capped(
    program: &Path,
    args: &[std::ffi::OsString],
    envs: &[(&str, std::ffi::OsString)],
) -> Result<(), String> {
    let mut child = std::process::Command::new(program)
        .args(args)
        .envs(envs.iter().map(|(k, v)| (*k, v)))
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .map_err(|e| format!("{}: {e}", program.display()))?;
    let budget = std::time::Duration::from_secs(MAX_CONVERT_SECONDS);
    let started = std::time::Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                return status
                    .success()
                    .then_some(())
                    .ok_or_else(|| format!("{} exited with {status}", program.display()))
            }
            Ok(None) => {
                if started.elapsed() > budget {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(format!(
                        "{} did not finish in {MAX_CONVERT_SECONDS} s and was stopped",
                        program.display()
                    ));
                }
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
            Err(e) => {
                // Stop it and reap it, the same way the timeout path does: a child left
                // running after this is a Blender still converting into a directory that
                // is about to be deleted, and an unreaped one is a zombie for the rest of
                // this process.
                let _ = child.kill();
                let _ = child.wait();
                return Err(format!("waiting for {}: {e}", program.display()));
            }
        }
    }
}

/// The tail every Blender recipe shares: write a self-contained GLB at the path
/// `BR_OUT` names. Kept as one string rather than repeated in two places, because
/// it is the half of the script that must stay exactly right for the reader that
/// consumes the result.
const GLB_EXPORT: &str =
    "bpy.ops.export_scene.gltf(filepath=os.environ['BR_OUT'], export_format='GLB')";

/// The Python Blender runs for the recipes that go through `--python-expr`.
///
/// `Some(op)` is the two-step case FBX and USD need: open the file through the
/// addon import operator, then export. `None` is `.blend`, where Blender's own
/// loader has already opened the file before this runs, so only the export is left.
///
/// The paths travel as environment variables, never inside the program text. A
/// filename is user input on a drag-and-drop path, and this is source the
/// interpreter executes: quoting it in is how an apostrophe in someone's folder
/// name becomes a syntax error at best, and — if the quoting is ever half-finished
/// — an injection at worst.
fn blender_script(import_op: Option<&str>) -> String {
    match import_op {
        Some(op) => format!(
            "import bpy, os\nbpy.ops.{op}(filepath=os.environ['BR_IN'])\n{GLB_EXPORT}"
        ),
        None => format!("import bpy, os\n{GLB_EXPORT}"),
    }
}

/// How a format with no reader in this process gets turned into glTF bytes. One
/// variant per real command line, measured against the tools rather than reasoned
/// out from their documentation — the module note records what was found.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum Recipe {
    /// `blender -b <file> --python-expr …`: Blender's own loader opens the file
    /// first, which only works for formats it really loads.
    BlenderOpen,
    /// `blender -b --python-expr …` with an explicit addon import, which is what
    /// FBX and USD need because the .blend loader refuses them outright.
    ///
    /// The operator name is spelled out because Blender moves these between
    /// categories between versions, and the symmetric-looking guess is not there:
    /// Blender 5.0 has `wm.fbx_import` and `wm.usd_import`, and has no
    /// `wm.fbx_export` at all — its FBX writer is `export_scene.fbx`.
    BlenderImport(&'static str),
    /// `assimp export <file> <out>`, for a machine with no Blender. FBX only: see
    /// [`recipes_for`].
    Assimp,
}

/// The recipes tried for one extension, in order.
///
/// Every convertible extension is named rather than defaulted, because a catch-all arm
/// would quietly hand the next format added to [`CONVERTIBLE_EXTENSIONS`] the USD
/// recipe — and the failure it produced would never mention that the recipe table was
/// never extended. `&[]` is "no converter here", which is what the error then says.
///
/// `.blend` has no fallback: assimp's Blender loader predates the file format
/// Blender 2.8+ writes and says so itself — measured on this machine as "BLENDER
/// magic bytes are missing, couldn't find GZIP header either".
fn recipes_for(ext: &str) -> &'static [Recipe] {
    match ext {
        "blend" => &[Recipe::BlenderOpen],
        "fbx" => &[
            Recipe::BlenderImport("wm.fbx_import"),
            Recipe::Assimp,
        ],
        "usd" | "usda" | "usdc" | "usdz" => &[Recipe::BlenderImport("wm.usd_import")],
        _ => &[],
    }
}

/// A scratch directory that deletes itself when the guard goes out of scope.
///
/// The invariant — the converted file is gone when the import ends, whichever way the
/// import ended — is easy to state and easy to break one `return` at a time, and
/// `convert_with` already has several exits before anyone adds another. A panic that
/// unwinds still runs this; an abort from inside a library does not, which is why the
/// converter's *output* is size-checked before a single byte of it is read
/// ([`convert_with`]) rather than left to the glTF reader to survive.
struct Scratch(PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Convert one of the unparseable formats into glTF bytes with whatever this
/// machine has, trying [`recipes_for`] in order. And report which tool did it, so
/// the import can say what happened to it.
fn convert_to_gltf(path: &Path, ext: &str) -> Result<(Vec<u8>, String), ImportError> {
    convert_with(path, ext, recipes_for(ext))
}

/// The conversion, with the recipe list supplied by the caller.
///
/// That seam exists because the assimp row of [`recipes_for`] is only reached when
/// Blender is missing or fails, so on a machine that has Blender — every development
/// machine that tests this — nothing would ever run it. A fallback nobody runs is a
/// fallback that quietly stops working, and `assimp_falls_back_for_a_real_fbx` below
/// forces the list to prove it still does.
///
/// The temp directory lives only as long as this call, and is created **exclusively**.
///
/// Nothing downstream keeps the path (`Import::source` is a stem, and `Mesh::name` is
/// the static string "imported"), and leaving converted files in `/tmp` forever is a
/// leak that only shows up as a full disk a month later — hence [`Scratch`]. The name
/// carries the process id, an attempt counter and the extension so imports in flight
/// cannot collide, and the counter is stepped until `create_dir` succeeds, because
/// `create_dir` refuses a path that already exists: a directory — or a symlink to
/// somewhere else — that another local process planted under this name gets skipped
/// rather than adopted, which matters because the converter is then told to write its
/// output there. This buys exclusivity, not secrecy: another local user can still
/// *read* the converted file, since a temp directory is world-traversable by the
/// usual umask. Nothing this app writes into it is worth reading by someone else.
fn convert_with(
    path: &Path,
    ext: &str,
    recipes: &[Recipe],
) -> Result<(Vec<u8>, String), ImportError> {
    let source = stem(path);
    static ATTEMPT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let mut dir = PathBuf::new();
    let mut failure = String::new();
    for _ in 0..8 {
        let attempt = ATTEMPT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let candidate = std::env::temp_dir().join(format!(
            "bouncing-room-convert-{}-{attempt}-{ext}",
            std::process::id()
        ));
        match std::fs::create_dir(&candidate) {
            // `create_dir`, not `create_dir_all`: the exclusive form, which fails on a
            // name that already exists — including one that is a symlink.
            Ok(()) => {
                dir = candidate;
                break;
            }
            Err(e) => failure = e.to_string(),
        }
    }
    if dir.as_os_str().is_empty() {
        return Err(ImportError::Bad {
            source,
            reason: format!(
                "could not make a temporary directory of our own under {} ({failure})",
                std::env::temp_dir().display()
            ),
        });
    }
    // The directory now deletes itself on every exit, including a panic.
    let scratch = Scratch(dir);
    let out = scratch.0.join(format!("{source}.glb"));
    // Owned, because each recipe arm below clones these into its own argument list.
    let out_os = out.as_os_str().to_owned();
    let in_os = path.as_os_str().to_owned();

    let mut attempts = Vec::new();
    for recipe in recipes {
        let (program, args, envs, label): (
            PathBuf,
            Vec<std::ffi::OsString>,
            Vec<(&str, std::ffi::OsString)>,
            &str,
        ) = match recipe {
            Recipe::BlenderOpen => match blender_program() {
                Some(p) => (
                    p,
                    vec![
                        "-b".into(),
                        in_os.clone(),
                        "--python-expr".into(),
                        blender_script(None).into(),
                    ],
                    vec![("BR_OUT", out_os.clone())],
                    "Blender",
                ),
                None => {
                    attempts.push("Blender: not found".to_owned());
                    continue;
                }
            },
            Recipe::BlenderImport(op) => match blender_program() {
                Some(p) => (
                    p,
                    vec![
                        "-b".into(),
                        "--python-expr".into(),
                        blender_script(Some(op)).into(),
                    ],
                    vec![("BR_IN", in_os.clone()), ("BR_OUT", out_os.clone())],
                    "Blender",
                ),
                None => {
                    attempts.push("Blender: not found".to_owned());
                    continue;
                }
            },
            Recipe::Assimp => match assimp_program() {
                Some(p) => (
                    p,
                    vec!["export".into(), in_os.clone(), out_os.clone()],
                    Vec::new(),
                    "assimp",
                ),
                None => {
                    attempts.push("assimp: not found".to_owned());
                    continue;
                }
            },
        };
        // Start every recipe from no output at all, because the output path is shared
        // between them. A Blender killed at the timeout can leave a half-written `.glb`
        // behind; the assimp attempt that follows exits 0 without writing anything when
        // it only half-reads the source, and would have Blender's truncated bytes read
        // back as its own result — geometry credited to the wrong tool.
        let _ = std::fs::remove_file(&out);
        match run_capped(&program, &args, &envs) {
            Ok(()) => {}
            Err(e) => {
                attempts.push(format!("{label}: {e}"));
                continue;
            }
        }
        // What has to be bounded is the converter's *output*, not the file it was
        // handed. `MAX_FILE_BYTES` gates the source of every format read in this
        // process, and a converted source is never read here at all: a few megabytes of
        // `.usdz` — a zip archive — or a `.blend` full of procedural objects can come
        // out as a hundred times that in glTF. Checked before the read, so those bytes
        // never enter this process, and no later recipe is tried: the second tool would
        // write a comparably large file and cost another start-up to say the same thing.
        if let Ok(meta) = std::fs::metadata(&out) {
            if meta.len() > MAX_FILE_BYTES {
                let from = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
                // One decimal, because a 900 kB source that came out as 70 MB of glTF is
                // the surprising part, and whole megabytes would report it as 0 into 67.
                let mib = |bytes: u64| bytes as f64 / (1 << 20) as f64;
                // Returned on its own rather than joined into `attempts`, because the
                // advice that follows a failed conversion — set `BLENDER`, set `ASSIMP` —
                // is wrong here: the tool ran, and did exactly what it was asked. The
                // [`Scratch`] guard still removes the directory on the way out.
                return Err(ImportError::Bad {
                    source,
                    reason: format!(
                        "{label} turned {:.1} MiB of .{ext} into {:.1} MiB of glTF, past this \
                         app's {} MiB input limit — decimate the scene, or export a smaller \
                         glTF from the source's own tool",
                        mib(from),
                        mib(meta.len()),
                        MAX_FILE_BYTES / (1 << 20)
                    ),
                });
            }
        }
        match std::fs::read(&out) {
            Ok(bytes) if !bytes.is_empty() => return Ok((bytes, label.to_owned())),
            Ok(_) => attempts.push(format!("{label} wrote an empty file")),
            Err(e) => attempts.push(format!("{label} produced no readable output: {e}")),
        }
    }

    // Two different empties. No recipe at all means this file has an extension in
    // [`CONVERTIBLE_EXTENSIONS`] that [`recipes_for`] never heard of — a mistake in the
    // code, and pointing the user at `BLENDER`/`ASSIMP` would send them off to fix
    // something that is already fine. No attempt *succeeded* means the tools were tried
    // or missing, which is what the message about variables is for.
    let tried = if recipes.is_empty() {
        "this app has no conversion recipe for it (a format in CONVERTIBLE_EXTENSIONS \
         that recipes_for does not name)"
            .to_owned()
    } else if attempts.is_empty() {
        "no converter is available on this machine".to_owned()
    } else {
        attempts.join("; ")
    };
    Err(ImportError::Bad {
        source,
        reason: format!(
            "cannot convert .{ext}: {tried}. Set BLENDER or ASSIMP to point at a \
             converter, or export the file to .gltf or .glb from its own tool"
        ),
    })
}

/// The specular exponent an import draws with when its format carries no
/// material: the OBJ and STL path, and a glTF with no PBR block.
///
/// [`Import::gloss`] says the caller picks, and every caller has picked the same
/// number ever since the first one — the window layer and the render probe each
/// kept their own copy, which is how a single wrong value ended up in the app *and*
/// in the screenshots meant to catch it. So the pick lives here, beside the scale it
/// is meant to sit in.
///
/// That scale is the one the shader works in: [`crate::geom::PRIMITIVE_GLOSS`] spans
/// 15..92 for the built-in shapes and a glTF roughness maps to 8..96 (the
/// `roughness_factor` line in the glTF reader), so a plausible neutral belongs near
/// the middle.
///
/// The value was 0.35 for most of the port's life, which was wrong twice over: it
/// reads like a roughness fraction rather than an exponent, and at 0.35 the shader's
/// `pow(ndh, gloss)` is close to 1 for *every* half-vector, so an imported part got a
/// specular wash across its whole surface instead of a highlight. Imported spikes
/// looked waxy beside primitives of the same shape, and nobody noticed until the
/// glTF path put the real range on paper.
pub const DEFAULT_GLOSS: f32 = 40.0;

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
    /// A convex hull around the mesh, for the simulation to collide it as. `None`
    /// when the mesh is flat enough to enclose no volume, in which case it
    /// collides as its bounding sphere like everything used to.
    pub hull: Option<Hull>,
    /// Gloss from the file's material, when the format has one (glTF). OBJ and STL
    /// leave this `None`, and the caller picks — with [`DEFAULT_GLOSS`], which is
    /// where that pick lives and what it took to get right.
    pub gloss: Option<f32>,
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
                "no reader for .{extension}, and no converter for it either — read in-process: \
                 .obj, .stl, .gltf, .glb; converted via an external tool: .fbx, .blend, .usd, \
                 .usda, .usdc, .usdz"
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

/// True if `path` can be imported at all — read in-process or converted. Judged by
/// extension, because that is all a caller has to go on before it reads anything;
/// the answer does not depend on whether a converter is installed, so the UI can
/// offer the format and the attempt explains what was missing.
pub fn is_supported(path: &Path) -> bool {
    extension(path).is_some_and(|ext| {
        SUPPORTED_EXTENSIONS.contains(&ext.as_str()) || CONVERTIBLE_EXTENSIONS.contains(&ext.as_str())
    })
}

/// The extensions a caller can offer the user, for a hint line. The converted
/// group is marked as such because it is slower and needs a tool the user may not
/// have — hiding that in a flat list would promise more than the app can always
/// deliver.
pub fn supported_hint() -> String {
    let list = |names: &[&str]| -> String {
        let named: Vec<String> = names.iter().map(|e| format!(".{e}")).collect();
        match named.split_last() {
            Some((last, rest)) if !rest.is_empty() => format!("{} or {last}", rest.join(", ")),
            _ => named.concat(),
        }
    };
    // "Blender, or assimp for `.fbx`" rather than "Blender or assimp": assimp is only
    // in the recipe list for one of these extensions (see `recipes_for`), and a hint
    // that implies otherwise promises a fallback that is not there for the other five.
    format!(
        "{} (and {}, converted by Blender, or by assimp for .fbx)",
        list(&SUPPORTED_EXTENSIONS),
        list(&CONVERTIBLE_EXTENSIONS)
    )
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
    // Decided from the extension before the filesystem is touched, so an
    // unimportable format answers "no reader for .rar" rather than "no such file".
    //
    // A convertible format takes its own path and returns: it is not opened here at
    // all, because the bytes that matter are the ones the converter writes. The
    // `MAX_FILE_BYTES` below therefore does not gate a `.blend` at all — the source is
    // never read into this process. What gates it is the conversion timeout plus the
    // same `MAX_FILE_BYTES` ceiling applied to whatever the converter wrote
    // ([`convert_with`]), because a few megabytes of source can come out as many times
    // that in glTF, and glTF bytes that big are what this reader cannot hold.
    if CONVERTIBLE_EXTENSIONS.contains(&ext.as_str()) {
        // The existence check every other format gets for free from the `metadata` call
        // below, done here first. Otherwise a mistyped path in the panel box —
        // `scene.blend` where the file is `scene1.blend` — spends a Blender start-up and
        // answers "cannot convert .blend: Blender exited with exit status 1", which reads
        // like the app cannot handle the format rather than like a file that is not there.
        //
        // It belongs *here*, at the moment of conversion, and not hoisted to the window
        // layer's queue handler — the natural tidy-up, since it is a cheap check before
        // spawning a thread. Conversions run one at a time, so a queued path can sit for
        // minutes while its file is renamed, moved or deleted; asked at queue time, that
        // file passes the check and then fails inside a converter, and the answer blames
        // the tool. Asked here, it answers `Io { path, "no such file" }`, naming the path
        // that stopped existing. Related, and the reason the panel does not prefix failure
        // lines with the file name: `Unsupported` is the one variant of `ImportError` that
        // prints no path or stem — only an extension — and every other one already names
        // the file, so a caller adding a name on top is doubling it.
        if !path.try_exists().unwrap_or(false) {
            return Err(ImportError::Io {
                path: path.display().to_string(),
                reason: "no such file".to_owned(),
            });
        }
        let (bytes, tool) = convert_to_gltf(path, &ext)?;
        let mut import = parse_gltf(&bytes, None, &source)?;
        // Stated up front because it is the thing a user is most likely to be
        // surprised by: a round trip through a converter is a re-export, and the
        // converter decides units, up-axis and what survives of a material.
        import.notes.insert(
            0,
            format!(
                "converted .{ext} to glTF with {tool}: units, up-axis and materials are \
                 whatever {tool} decided, not what the source file said"
            ),
        );
        return Ok(import);
    }
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
        "gltf" | "glb" => {
            let bytes = std::fs::read(path).map_err(|e| ImportError::Io {
                path: path.display().to_string(),
                reason: e.to_string(),
            })?;
            parse_gltf(&bytes, path.parent(), &source)
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
    /// One colour per corner, in lockstep with `pos` — or empty, for the formats
    /// that carry no colour at all.
    col: Vec<[u8; 4]>,
    /// Gloss read from the file's material, if it has one.
    gloss: Option<f32>,
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
// .gltf / .glb
// --------------------------------------------------------------------------------------

/// Decode standard base64, for a glTF whose buffers are embedded as data URIs.
/// Hand-rolled because it is twenty lines and the alternative is a dependency.
fn decode_base64(text: &str) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(text.len() * 3 / 4);
    let (mut acc, mut bits) = (0u32, 0u32);
    for byte in text.bytes() {
        let value = match byte {
            b'A'..=b'Z' => byte - b'A',
            b'a'..=b'z' => byte - b'a' + 26,
            b'0'..=b'9' => byte - b'0' + 52,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            b'=' | b'\r' | b'\n' | b' ' => continue,
            _ => return None,
        };
        acc = (acc << 6) | u32::from(value);
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
            acc &= (1 << bits) - 1;
        }
    }
    Some(out)
}

/// Parse a glTF 2.0 file, text or binary. `dir` is where a `.gltf`'s external
/// `.bin` files are looked for; `None` (a test, or a path with no parent) means
/// only embedded buffers can be read.
///
/// Every mesh in the default scene merges into one body (rule 1), with each
/// node's transform applied, so an assembly arrives assembled. What survives of
/// its materials is their **base colour**: this renderer has no texture
/// coordinates, so a textured model keeps each material's colour factor (and any
/// vertex colours) and loses the image. Roughness becomes the body's gloss.
pub fn parse_gltf(
    bytes: &[u8],
    dir: Option<&Path>,
    source: &str,
) -> Result<Import, ImportError> {
    let bad = |reason: String| ImportError::Bad {
        source: source.to_owned(),
        reason,
    };
    let gltf = gltf::Gltf::from_slice(bytes).map_err(|e| bad(format!("not a readable glTF: {e}")))?;
    let document = &gltf.document;

    let mut buffers: Vec<Vec<u8>> = Vec::new();
    for buffer in document.buffers() {
        let data = match buffer.source() {
            gltf::buffer::Source::Bin => gltf
                .blob
                .clone()
                .ok_or_else(|| bad("the .glb names a binary chunk it does not contain".to_owned()))?,
            gltf::buffer::Source::Uri(uri) => {
                if let Some(rest) = uri.strip_prefix("data:") {
                    let encoded = rest
                        .split_once(";base64,")
                        .map(|(_, data)| data)
                        .ok_or_else(|| bad("an embedded buffer is not base64".to_owned()))?;
                    decode_base64(encoded)
                        .ok_or_else(|| bad("an embedded buffer is not valid base64".to_owned()))?
                } else {
                    let dir = dir.ok_or_else(|| {
                        bad(format!("it refers to {uri}, and there is no folder to look in"))
                    })?;
                    // Only a file beside the .gltf: a URI that climbs out of the
                    // folder or names a URL is not something to follow on the
                    // say-so of a model file.
                    let name = uri.replace("%20", " ");
                    if name.contains("..") || name.contains("://") || Path::new(&name).is_absolute() {
                        return Err(bad(format!("it refers to {uri}, which is outside its own folder")));
                    }
                    let path = dir.join(&name);
                    let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                    if size > MAX_FILE_BYTES {
                        return Err(bad(format!("its buffer {uri} is over the input limit")));
                    }
                    std::fs::read(&path).map_err(|e| ImportError::Io {
                        path: path.display().to_string(),
                        reason: e.to_string(),
                    })?
                }
            }
        };
        if (data.len() as u64) < buffer.length() as u64 {
            return Err(bad("a buffer is shorter than the file says it is".to_owned()));
        }
        buffers.push(data);
    }

    let mut raw = Raw::default();
    let mut glosses: Vec<f32> = Vec::new();
    let mut textured = 0usize;
    let mut skipped_modes = 0usize;

    // Walk the default scene (or the first, or failing that every root node),
    // carrying each node's world transform down to its mesh.
    let roots: Vec<gltf::Node> = match document.default_scene().or_else(|| document.scenes().next()) {
        Some(scene) => scene.nodes().collect(),
        None => document.nodes().collect(),
    };
    let mut stack: Vec<(gltf::Node, glam::Mat4)> =
        roots.into_iter().map(|n| (n, glam::Mat4::IDENTITY)).collect();
    let mut visited = 0usize;
    while let Some((node, parent)) = stack.pop() {
        visited += 1;
        if visited > 100_000 {
            return Err(bad("its node graph does not end".to_owned()));
        }
        let world = parent * glam::Mat4::from_cols_array_2d(&node.transform().matrix());
        for child in node.children() {
            stack.push((child, world));
        }
        let Some(mesh) = node.mesh() else { continue };
        // Normals transform by the inverse transpose, which matters as soon as a
        // node is scaled unevenly.
        let normal_matrix = glam::Mat3::from_mat4(world).inverse().transpose();
        for primitive in mesh.primitives() {
            if primitive.mode() != gltf::mesh::Mode::Triangles {
                skipped_modes += 1;
                continue;
            }
            let reader = primitive.reader(|b| buffers.get(b.index()).map(Vec::as_slice));
            let Some(positions) = reader.read_positions() else { continue };
            let positions: Vec<[f32; 3]> = positions.collect();
            let normals: Option<Vec<[f32; 3]>> = reader.read_normals().map(|n| n.collect());
            let vertex_colors: Option<Vec<[f32; 4]>> =
                reader.read_colors(0).map(|c| c.into_rgba_f32().collect());
            let indices: Vec<u32> = match reader.read_indices() {
                Some(indices) => indices.into_u32().collect(),
                None => (0..positions.len() as u32).collect(),
            };

            let material = primitive.material();
            let pbr = material.pbr_metallic_roughness();
            let base = pbr.base_color_factor();
            if pbr.base_color_texture().is_some() {
                textured += 1;
            }
            // Smooth materials shine: roughness 0 is a tight highlight, 1 a
            // broad dull one, on the same scale the built-in primitives use.
            glosses.push(8.0 + (1.0 - pbr.roughness_factor().clamp(0.0, 1.0)).powi(2) * 88.0);

            let start = raw.pos.len() as u32;
            for (i, p) in positions.iter().enumerate() {
                let p = world.transform_point3(glam::Vec3::from(*p));
                raw.pos.push(p.to_array());
                let n = normals
                    .as_ref()
                    .and_then(|n| n.get(i))
                    .map(|n| (normal_matrix * glam::Vec3::from(*n)).normalize_or_zero())
                    .unwrap_or(glam::Vec3::ZERO);
                raw.nrm.push(n.to_array());
                let v = vertex_colors.as_ref().and_then(|c| c.get(i)).copied().unwrap_or([1.0; 4]);
                // The lit shader works in display-referred values, and glTF
                // colours are linear, so they are encoded on the way in.
                let encode = |linear: f32| (linear.clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0).round() as u8;
                raw.col.push([
                    encode(base[0] * v[0]),
                    encode(base[1] * v[1]),
                    encode(base[2] * v[2]),
                    255,
                ]);
            }
            let count = positions.len() as u32;
            // A mirrored node flips the winding; putting it back keeps the flat
            // normals `finish` may have to rebuild pointing outward.
            let mirrored = world.determinant() < 0.0;
            for tri in indices.chunks_exact(3) {
                if tri.iter().any(|&i| i >= count) {
                    continue;
                }
                let (a, b, c) = (start + tri[0], start + tri[1], start + tri[2]);
                if mirrored {
                    raw.tri.extend_from_slice(&[a, c, b]);
                } else {
                    raw.tri.extend_from_slice(&[a, b, c]);
                }
            }
        }
    }

    if !glosses.is_empty() {
        raw.gloss = Some(glosses.iter().sum::<f32>() / glosses.len() as f32);
    }
    let mut import = finish(raw, source)?;
    if textured > 0 {
        import.notes.push(format!(
            "{textured} textured material(s) kept their base colour only — this renderer has no              texture coordinates"
        ));
    }
    if skipped_modes > 0 {
        import.notes.push(format!(
            "{skipped_modes} primitive(s) were points, lines or strips and are not drawn"
        ));
    }
    Ok(import)
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
    // The contract's invariant, measured the way the contract measures it:
    // `Mesh::max_radius` is the farthest *vertex* from the origin, and every
    // built-in primitive is built so that it is exactly 1.0. Normalising by the
    // bounding box's diagonal instead is a guess at that number — the diagonal
    // reaches a corner of the box that no vertex of an irregular model need occupy
    // — and it leaves the sphere a few percent bigger than the mesh. On a
    // five-sided spike that measured 1.6%: the collider touched the floor while the
    // picture floated above it. So the radius here is the true tight bounding
    // sphere about the box centre, which is what makes `radius_factor = 1.0` exact
    // rather than merely close.
    let radius = raw
        .pos
        .iter()
        .filter(|p| finite(**p))
        .map(|p| {
            let dx = p[0] as f64 - centre[0];
            let dy = p[1] as f64 - centre[1];
            let dz = p[2] as f64 - centre[2];
            (dx * dx + dy * dy + dz * dz).sqrt()
        })
        .fold(0.0f64, f64::max);
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
    // Colour joins the key when the file has any: two corners that agree on
    // position and normal but sit on differently coloured faces are a material
    // boundary, and welding across it would smear one colour into the other.
    let colored = !raw.col.is_empty() && raw.col.len() == raw.pos.len();
    let mut welded_colors: Vec<[u8; 4]> = Vec::new();
    let mut keys: HashMap<([i64; 3], [i64; 3], [u8; 4]), u16> = HashMap::new();
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
        let mut tint = [[255u8; 4]; 3];
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
            if colored {
                tint[c] = raw.col[slot];
            }
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
                tint[c],
            );
            corner_idx[c] = match keys.get(&key) {
                Some(existing) => *existing,
                None => {
                    let fresh = welded.len() as u16;
                    welded.push(Vertex {
                        position: face[c],
                        normal: n,
                    });
                    welded_colors.push(tint[c]);
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

    // The collider: a hull around what is actually drawn, so a sampled mesh and
    // its collider still agree.
    let points: Vec<glam::Vec3> = welded.iter().map(|v| glam::Vec3::from(v.position)).collect();
    let hull = Hull::from_points(&points);
    if hull.is_none() {
        notes.push("too flat to enclose a volume, so it collides as a sphere".to_owned());
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
            colors: colored.then_some(welded_colors),
        },
        source: source.to_owned(),
        notes,
        dropped_triangles: sampled_out,
        hull,
        gloss: raw.gloss,
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
    fn the_unit_radius_is_the_farthest_vertex_not_the_corner_of_the_bounding_box() {
        // Six vertices on the axes: the box around them is [-1,1]³, so half its
        // diagonal is 1.73 — a corner that no vertex of this shape occupies.
        // Normalising by that (which is what this used to do) leaves `max_radius`
        // at 0.58 and the mesh a hair under two thirds the size its collider
        // claims. The contract's number is the vertex distance, so that is what
        // goes to 1.0.
        let text = "\
v  1 0 0
v  0 1 0
v -1 0 0
v  0 -1 0
v  0 0 1
v  0 0 -1
f 1 2 5
f 2 3 5
f 3 4 5
f 4 1 5
f 2 1 6
f 3 2 6
f 4 3 6
f 1 4 6
";
        let import = parse_obj(text, "octahedron").expect("an eight-face shell parses");
        let radius = import.mesh.max_radius();
        assert!(
            (radius - 1.0).abs() < 1.0e-4,
            "max_radius should normalise to 1.0, got {radius}"
        );
        assert_eq!(import.mesh.triangle_count(), 8);
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
    fn an_unimportable_format_is_named_before_the_filesystem_is_touched() {
        // A path that does not exist still gets an answer about the format, which is
        // the honest reply when the UI offers ten extensions and some of them need a
        // tool this machine may not have.
        let path = Path::new("/nonexistent/something.rar");
        assert!(!is_supported(path));
        match import_path(path) {
            Err(ImportError::Unsupported { extension }) => assert_eq!(extension, "rar"),
            other => panic!("expected Unsupported, got {other:?}"),
        }
        assert!(is_supported(Path::new("a/b/c.OBJ")));
        assert!(is_supported(Path::new("c.stl")));
        assert!(is_supported(Path::new("x.glb")));
        assert_eq!(
            supported_hint(),
            ".obj, .stl, .gltf or .glb (and .fbx, .blend, .usd, .usda, .usdc or .usdz, \
             converted by Blender, or by assimp for .fbx)"
        );
    }

    #[test]
    fn a_convertible_format_is_accepted_whatever_is_installed() {
        // `is_supported` deliberately does not look at the machine: the drop handler
        // and the file dialog ask it before anything is read, and an answer that
        // changed with what happens to be installed would make the UI inconsistent.
        // The missing tool belongs in the failure message instead, in words that say
        // what to set (see the test after this one).
        let path = Path::new("/nonexistent/something.fbx");
        assert!(is_supported(path), ".fbx is importable, by conversion");
        assert!(needs_converter(path));
        assert!(!needs_converter(Path::new("x.obj")));
        // And a path that is simply not there is reported as that, *before* a converter
        // is started on its way. A Blender that fails on a missing file answers
        // "cannot convert .fbx: … exited with 1", which reads like the app cannot
        // handle the format rather than like nobody typed a real path.
        match import_path(path) {
            Err(ImportError::Io { reason, .. }) => assert!(
                reason.contains("no such file"),
                "a missing file has to be named as missing, not converted: {reason}"
            ),
            other => panic!("expected a missing-file error, got {other:?}"),
        }
    }

    #[test]
    fn a_file_that_cannot_be_converted_names_the_tools_it_needs() {
        // The half the missing path cannot show: the file is there, the extension is one
        // this app offers, and the conversion still fails. Whatever is installed — both
        // tools, one, or neither — the answer has to name what could have read it.
        let dir = scratch("unreadable");
        let path = dir.join("nonsense.fbx");
        std::fs::write(&path, b"not an FBX, whatever the extension claims")
            .expect("a temp file to fail on");
        match import_path(&path) {
            Err(ImportError::Bad { reason, .. }) => assert!(
                reason.contains("Blender")
                    || reason.contains("BLENDER")
                    || reason.contains("assimp"),
                "a failed conversion has to name the tools: {reason}"
            ),
            other => panic!("expected a conversion failure, got {other:?}"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    // The three tests below are the only thing standing between the converter
    // recipes and a tool that changed under them, and all three skip when the tool is
    // not on the machine — which is why they say so out loud, and why none of them
    // treats "the tool is here but the call failed" as a skip. The first version of
    // this had one test, it guessed a Blender operator name, and the guess made that
    // test skip quietly while looking green.

    /// A cube, written by Blender five ways: one fixture per convertible extension
    /// that this Blender can produce. Each write is in its own `try` so a Blender
    /// without one of the addons prints a reason instead of ending the run, and the
    /// `.blend` is saved last because `wm.save_as_mainfile` writes the whole scene.
    const FIXTURE_WRITERS: &str = r#"
import bpy, os
out = os.environ['BR_DIR']
bpy.ops.mesh.primitive_cube_add(size=2)
for name, op in [
    ('cube.fbx', 'export_scene.fbx'),
    ('cube.usd', 'wm.usd_export'),
    ('cube.usda', 'wm.usd_export'),
    ('cube.usdz', 'wm.usd_export'),
    ('cube.blend', 'wm.save_as_mainfile'),
]:
    try:
        group, fn = op.split('.')
        getattr(getattr(bpy.ops, group, None), fn)(filepath=os.path.join(out, name))
        print('WROTE', name)
    except Exception as exc:
        print('SKIPPED', name, exc)
"#;

    /// Run one headless Blender over `script` in `dir`, and return what it printed.
    ///
    /// Output is captured here and discarded by `run_capped` in the module above,
    /// deliberately: the app wants a file and has no HUD room for a progress bar,
    /// while these tests are asking the tool a question, so the answer is the point.
    /// A script file rather than `--python-expr`, because `try`/`except` in a Rust
    /// string literal reads as noise.
    fn blender_run(dir: &Path, script: &str) -> Result<String, String> {
        let file = dir.join("make.py");
        std::fs::write(&file, script).map_err(|e| format!("{}: {e}", file.display()))?;
        let blender = blender_program().expect("the caller checked for Blender");
        let run = std::process::Command::new(&blender)
            .arg("-b")
            .arg("--python")
            .arg(&file)
            .env("BR_DIR", dir)
            .output()
            .map_err(|e| format!("{}: {e}", blender.display()))?;
        if !run.status.success() {
            return Err(format!(
                "{} exited with {}: {}",
                blender.display(),
                run.status,
                String::from_utf8_lossy(&run.stderr).trim()
            ));
        }
        Ok(String::from_utf8_lossy(&run.stdout).into_owned())
    }

    /// A scratch directory per test: the conversion tests run on the test threads at
    /// the same time and must not reach for the same fixtures.
    fn scratch(label: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "bouncing-room-converttest-{}-{label}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("a temp directory for test fixtures");
        dir
    }

    #[test]
    fn each_convertible_extension_names_the_recipe_it_uses() {
        // `CONVERTIBLE_EXTENSIONS` and `recipes_for` are two lists with nothing forcing
        // them to agree, and the failure modes of that gap are quiet: an extension in the
        // first but not the second converts nothing, and two recipes whose operator names
        // were written the wrong way round still pass the operator-exists check, because
        // both operators exist. So the pair is pinned here, name by name.
        for ext in CONVERTIBLE_EXTENSIONS {
            assert!(
                !recipes_for(ext).is_empty(),
                ".{ext} is offered to the user but has no recipe — add it to `recipes_for` \
                 or stop offering it"
            );
        }
        assert_eq!(recipes_for("blend"), [Recipe::BlenderOpen]);
        assert_eq!(
            recipes_for("fbx"),
            [Recipe::BlenderImport("wm.fbx_import"), Recipe::Assimp]
        );
        for ext in ["usd", "usda", "usdc", "usdz"] {
            assert_eq!(
                recipes_for(ext),
                [Recipe::BlenderImport("wm.usd_import")],
                ".{ext}"
            );
        }
        // And an extension neither list mentions gets nothing, rather than the USD recipe
        // a catch-all arm would have handed it.
        assert!(recipes_for("dae").is_empty());
    }

    #[test]
    fn the_recipes_name_operators_this_blender_actually_has() {
        // Every recipe here is a claim about a program that is not in this
        // repository, and Blender moves its importers between operator categories
        // between versions. The first draft guessed the symmetric name —
        // `wm.fbx_export`, which Blender 5.0 does not have; its FBX writer is
        // `export_scene.fbx` — and the test that used it skipped quietly. One
        // start-up to ask the tool is cheaper than that, and keeps the claim true
        // rather than true-on-the-day-it-was-written.
        let Some(blender) = blender_program() else {
            eprintln!("no Blender on this machine: the converter recipes go unchecked");
            return;
        };
        let dir = scratch("ops");
        // Read the operator names out of the recipes themselves. A hand-written list
        // of what the code calls can drift from the code, and that gap is exactly
        // where the original guess hid: the list would have gone on naming an
        // operator that no longer existed while the recipe named another.
        let mut ops: Vec<String> = Vec::new();
        for ext in CONVERTIBLE_EXTENSIONS {
            for recipe in recipes_for(ext) {
                if let Recipe::BlenderImport(op) = recipe {
                    if !ops.iter().any(|seen| seen == op) {
                        ops.push((*op).to_owned());
                    }
                }
            }
        }
        // The exporter every recipe ends with, taken from `GLB_EXPORT` rather than
        // spelled out again next to it.
        let export = GLB_EXPORT
            .split("bpy.ops.")
            .nth(1)
            .and_then(|rest| rest.split('(').next())
            .expect("`GLB_EXPORT` starts with an operator call, `bpy.ops.…(`");
        ops.push(export.to_owned());
        assert!(
            ops.len() >= 3,
            "this test asked Blender about {} operators, which is too few to be the \
             recipes: {ops:?}",
            ops.len()
        );
        // How the question gets asked, since three obvious ways all lie. Blender's
        // operator proxy fabricates a stub for *any* name: measured on Blender 5.0,
        // `hasattr(bpy.ops.wm, "definitely_not_an_operator")` is True, and calling
        // `.idname()` on it returns `WM_OT_definitely_not_an_operator` as happily as
        // it returns the real thing. Only the category's own listing answers honestly,
        // so membership of `dir(bpy.ops.<category>)` is the test — and it is the only
        // one of the three that reported `wm.fbx_export`, which this Blender does not
        // have, as absent.
        //
        // And because even that could stop being honest without anyone noticing, the
        // probe carries a name invented for the purpose and the test below insists
        // Blender calls it missing: a check that cannot fail is worth as much as the
        // silent skip this file used to have.
        let canary = "wm.definitely_not_an_operator";
        let probe_line = |op: &str| -> String {
            let (group, name) = op.split_once('.').expect("an operator category");
            format!(
                "print('HAVE' if {name:?} in dir(getattr(bpy.ops, {group:?}, None)) \
                 else 'MISSING', {op:?})\n"
            )
        };
        let mut script = String::from("import bpy\nprint('BLENDER', bpy.app.version_string)\n");
        for op in &ops {
            script.push_str(&probe_line(op));
        }
        // Asked in the same run, kept out of `ops` below: it is meant to come back
        // missing, and the assertion must not read it as a recipe that moved.
        script.push_str(&probe_line(canary));
        let printed = match blender_run(&dir, &script) {
            Ok(printed) => printed,
            Err(e) => {
                // Found but would not run: that is this machine's setup talking, not
                // a recipe naming a stale operator, so report and let it go.
                eprintln!("cannot run {} for the operator check: {e}", blender.display());
                let _ = std::fs::remove_dir_all(&dir);
                return;
            }
        };
        let _ = std::fs::remove_dir_all(&dir);
        eprintln!("{printed}");
        // `HAVE`/`MISSING` for one name, read out of Blender's own reply.
        let reported_missing = |op: &str| {
            printed
                .lines()
                .any(|line| line.contains(op) && line.trim_start().starts_with("MISSING"))
        };
        assert!(
            reported_missing(canary),
            "the probe has stopped working: Blender called the invented operator \
             {canary:?} present, so it would call anything present. Whatever `bpy.ops` \
             lists now, this test has to find a way to ask again rather than be deleted \
             for being inconvenient.\n{printed}"
        );
        for op in &ops {
            assert!(
                printed.contains(op.as_str()),
                "Blender answered nothing about {op}:\n{printed}"
            );
        }
        let missing = ops
            .iter()
            .filter(|op| reported_missing(op))
            .collect::<Vec<_>>();
        assert!(
            missing.is_empty(),
            "{} does not have {missing:?}, which the recipes in this file call by name. \
             Either Blender has moved them again — and then `recipes_for`, `GLB_EXPORT` and the \
             module note's table all move with it — or a recipe is naming a guess. Run the operator \
             list on the machine that fails before changing either.",
            blender.display()
        );
    }

    #[test]
    fn every_convertible_extension_round_trips_a_real_file_when_blender_is_here() {
        // The module note's table, checked rather than asserted: five extensions with
        // no reader in this process, each written by Blender and read back through its
        // own recipe, arriving as geometry that obeys the same four rules an OBJ does.
        // One Blender start-up writes all five fixtures, because the tool costs about
        // a second per launch and a test per extension would spend more time starting
        // Blender than testing anything.
        if blender_program().is_none() {
            eprintln!("no Blender on this machine: no conversion round trip was run");
            return;
        }
        let dir = scratch("round");
        let printed = match blender_run(&dir, FIXTURE_WRITERS) {
            Ok(printed) => printed,
            Err(e) => {
                eprintln!("could not write the conversion fixtures: {e}");
                let _ = std::fs::remove_dir_all(&dir);
                return;
            }
        };
        eprintln!("{printed}");
        let wanted = [
            "cube.fbx", "cube.usd", "cube.usda", "cube.usdz", "cube.blend",
        ];
        let made: Vec<String> = printed
            .lines()
            .filter_map(|line| line.strip_prefix("WROTE ").map(|name| name.trim().to_owned()))
            .collect();
        let not_written: Vec<&&str> = wanted
            .iter()
            .filter(|name| !made.iter().any(|made_name| made_name == *name))
            .collect();
        assert!(
            not_written.is_empty(),
            "Blender would not write {not_written:?}, which this test needs to read back: \
             {printed}\nIf a writer operator moved, `FIXTURE_WRITERS` moves and so does the \
             module note's claim about what was measured."
        );
        for name in wanted {
            let path = dir.join(name);
            let import = import_path(&path)
                .unwrap_or_else(|e| panic!("{} should import through its recipe: {e}", path.display()));
            // The note is part of the result, not a courtesy: a round trip through a
            // converter is a re-export, and the user has to be told.
            let ext = path.extension().unwrap().to_str().unwrap();
            assert!(
                import
                    .notes
                    .iter()
                    .any(|n| n.contains(&format!("converted .{ext} to glTF with Blender"))),
                "{name}: a converted file has to say which format and which tool: {:?}",
                import.notes
            );
            assert!(import.mesh.triangle_count() >= 2, "{name}: a cube has faces");
            assert_eq!(import.source, "cube", "{name}: the HUD lists the stem");
            assert_import_is_sane(&import);
            assert!(import.hull.is_some(), "{name}: a cube should arrive with a hull");
        }
        // `.usdc` is the sixth convertible extension and is not written here: it is
        // the binary form of the `.usd` already read above, so proving that one line
        // of `recipes_for` twice would cost another Blender start-up for nothing.
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn assimp_falls_back_for_a_real_fbx_when_asked_to() {
        // The fallback row of `recipes_for` cannot be reached on a machine that has
        // Blender, which is every machine that runs this suite — so the recipe list is
        // forced here. What assimp writes then goes through the same glTF reader and
        // owes the same four rules, which is the only way that row stays true.
        if assimp_program().is_none() {
            eprintln!("no assimp on this machine: the FBX fallback goes unchecked");
            return;
        }
        if blender_program().is_none() {
            // Nothing to feed it with: this process cannot write an FBX, which is the
            // whole reason the format needs a converter.
            eprintln!("no Blender to write an FBX with: the assimp fallback goes unchecked");
            return;
        }
        let dir = scratch("assimp");
        if let Err(e) = blender_run(&dir, FIXTURE_WRITERS) {
            eprintln!("could not write the FBX to hand to assimp: {e}");
            let _ = std::fs::remove_dir_all(&dir);
            return;
        }
        let fbx = dir.join("cube.fbx");
        let (bytes, tool) = convert_with(&fbx, "fbx", &[Recipe::Assimp])
            .unwrap_or_else(|e| panic!("assimp is installed, so it should read {fbx:?}: {e}"));
        assert_eq!(
            tool, "assimp",
            "the import has to name the tool that actually ran"
        );
        let import =
            parse_gltf(&bytes, None, "cube").expect("the GLB assimp wrote should read here");
        assert!(import.mesh.triangle_count() >= 2, "a cube has faces");
        assert_import_is_sane(&import);
        let _ = std::fs::remove_dir_all(&dir);
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
