//! Imports read from real files on disk, through the same entry point the app and
//! the probe use: [`import::import_path`].
//!
//! The fixtures live in `tests/fixtures/` so this is reproducible by anyone —
//! including CI — rather than depending on whatever was in `/tmp` when a check was
//! run by hand. Two of them are tiny and hand-written; the third is a generated
//! 24 200-facet heightfield whose purpose is to exceed the renderer's 16-bit index
//! format, which no small fixture can do.
//!
//! What each one is for:
//!
//! | Fixture | What it pins down |
//! | --- | --- |
//! | `spike.obj` | OBJ normals in `v//vn` form, fan-free triangles, and that an import really renders (the probe asserts its pixels change) |
//! | `diamond.stl` | a *binary* STL whose 80-byte header begins with the word `solid` — the free-text field is where CAD tools put the object name, so a prefix test would misread the file as ASCII |
//! | `crumple.stl` | an over-budget mesh: sampled out across the file rather than cut from the tail, so the object arrives whole instead of half present |

use std::path::PathBuf;

use bouncing_room_native::import::{self, ImportError};

fn fixture(name: &str) -> PathBuf {
    [env!("CARGO_MANIFEST_DIR"), "tests", "fixtures", name]
        .iter()
        .collect()
}

fn load(name: &str) -> import::Import {
    let path = fixture(name);
    import::import_path(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// The four rules every import owes the renderer and the simulation. Repeated
/// here rather than borrowed from the private helper inside `import.rs` on
/// purpose: the app and the probe are external consumers of this module, so the
/// guarantees have to be checkable from outside it.
fn assert_rules(import: &import::Import, label: &str) {
    let mesh = &import.mesh;

    // Indexed, and every index reachable — the format `Gfx::upload_mesh` assumes.
    let indices = mesh
        .indices
        .clone()
        .unwrap_or_else(|| panic!("{label}: an import must be indexed"));
    assert_eq!(indices.len() % 3, 0, "{label}: a ragged index buffer");
    assert!(
        mesh.vertex_count() <= import::MAX_VERTS,
        "{label}: {} vertices is over the u16 budget of {}",
        mesh.vertex_count(),
        import::MAX_VERTS
    );
    for i in indices.iter().chain(mesh.edges.iter()) {
        assert!(
            (*i as usize) < mesh.vertex_count(),
            "{label}: index {i} runs past the {} vertices",
            mesh.vertex_count()
        );
    }

    // Rule 2: a unit bounding radius, which is what makes a collision radius of
    // `scale * 1.0` the mesh's real bounds rather than an approximation.
    let radius = mesh.max_radius();
    assert!(
        (radius - 1.0).abs() < 1.0e-4,
        "{label}: bounding radius is {radius}, expected 1.0"
    );

    // Normals usable by the lighting model, which normalises but does not invent.
    for v in mesh.vertices.iter() {
        let len = (v.normal[0] * v.normal[0] + v.normal[1] * v.normal[1] + v.normal[2] * v.normal[2])
            .sqrt();
        assert!(
            (len - 1.0).abs() < 1.0e-3,
            "{label}: normal {:?} has length {len}",
            v.normal
        );
        assert!(
            v.position.iter().all(|c| c.is_finite())
                && v.normal.iter().all(|c| c.is_finite()),
            "{label}: non-finite vertex {v:?}"
        );
    }

    // Rule 4: wireframe has something to draw.
    assert!(
        !mesh.edges.is_empty() && mesh.edges.len() % 2 == 0,
        "{label}: edge list is empty or odd, so wireframe would draw nothing"
    );
}

#[test]
fn a_small_obj_imports_whole_and_unchanged() {
    let import = load("spike.obj");
    assert_eq!(import.source, "spike", "the HUD lists imports by file stem");
    // 5-sided spike: 4 side faces and 2 base triangles, all explicit in the file.
    assert_eq!(import.mesh.triangle_count(), 6);
    assert_eq!(import.dropped_triangles, 0);
    assert!(
        import
            .notes
            .iter()
            .all(|n| !n.contains("sampled out") && !n.contains("degenerate")),
        "a well-formed OBJ needs no repairs: {:?}",
        import.notes
    );
    assert_rules(&import, "spike.obj");
}

#[test]
fn a_binary_stl_whose_header_says_solid_is_still_read_as_binary() {
    let path = fixture("diamond.stl");
    let bytes = std::fs::read(&path).expect("the fixture is readable");
    assert!(
        bytes.starts_with(b"solid"),
        "the point of this fixture is the misleading prefix"
    );
    let import = import::import_path(&path).expect("the count beats the prefix");
    assert_eq!(import.source, "diamond");
    assert_eq!(import.mesh.triangle_count(), 8, "an 8-facet bipyramid");
    assert_eq!(import.dropped_triangles, 0);
    assert_rules(&import, "diamond.stl");
}

#[test]
fn an_over_budget_mesh_arrives_sampled_through_the_file_not_cut_from_the_front() {
    let import = load("crumple.stl");
    let kept = import.mesh.triangle_count();
    assert!(kept > 0, "an over-budget mesh still has to arrive");
    assert!(
        import.dropped_triangles > 0,
        "nothing was reported as left out: {} faces, {} vertices, notes {:?}",
        kept,
        import.mesh.vertex_count(),
        import.notes
    );
    assert_eq!(
        kept + import.dropped_triangles,
        24_200,
        "what survived plus what was left out has to equal the file's facets"
    );
    assert!(
        import
            .notes
            .iter()
            .any(|n| n.contains("sampled out") && n.contains("u16")),
        "the reduction has to be reported: {:?}",
        import.notes
    );

    // The reason sampling beats cutting: the kept geometry still spans the whole
    // object. The facets are generated in row-major order, so keeping the first
    // `kept` of them would cover roughly the first half of the x range — about
    // 0.59 of normalised space — while a stride reaches all 1.17 of it.
    let extent = |axis: usize| -> (f32, f32) {
        import
            .mesh
            .vertices
            .iter()
            .fold((f32::MAX, f32::MIN), |(lo, hi), v| {
                (lo.min(v.position[axis]), hi.max(v.position[axis]))
            })
    };
    for axis in 0..3 {
        let (lo, hi) = extent(axis);
        assert!(
            hi - lo > 1.0,
            "axis {axis} only covers {:.3} of normalised space, so this mesh was \
             cut rather than sampled",
            hi - lo
        );
    }
    assert_rules(&import, "crumple.stl");
}

#[test]
fn every_fixture_obeys_the_same_rules_as_every_import() {
    for name in ["spike.obj", "diamond.stl", "crumple.stl"] {
        assert_rules(&load(name), name);
    }
}

#[test]
fn an_unsupported_format_is_named_before_the_file_is_still_looked_for() {
    // A path inside the real fixtures directory that does not exist: the answer
    // must still be about the format, because the app asks this question the
    // moment a file is dropped, before it has any idea what is on disk.
    let path = fixture("model.fbx");
    assert!(!path.exists(), "this test is about the extension, not the bytes");
    assert!(!import::is_supported(&path));
    match import::import_path(&path) {
        Err(ImportError::Unsupported { extension }) => assert_eq!(extension, "fbx"),
        other => panic!(
            "expected Unsupported for {}, got {other:?}",
            path.display()
        ),
    }
    let message = import::unsupported_message(&path);
    assert!(
        message.contains(".obj") && message.contains(".stl"),
        "the refusal has to say what does work: {message}"
    );
}

#[test]
fn the_supported_list_matches_the_hint_the_panel_shows() {
    // The panel renders this string next to the path box; a format added to the
    // reader but not to the list would be invisible, and vice versa.
    assert_eq!(import::supported_hint(), ".obj or .stl");
    assert!(import::is_supported(&fixture("spike.obj")));
    assert!(import::is_supported(&fixture("diamond.stl")));
    assert!(import::is_supported(&PathBuf::from("UPPER.OBJ")));
    assert!(!import::is_supported(&PathBuf::from("no extension")));
}
