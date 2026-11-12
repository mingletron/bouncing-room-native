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
fn an_unimportable_format_is_named_before_the_file_is_still_looked_for() {
    // A path inside the real fixtures directory that does not exist: the answer
    // must still be about the format, because the app asks this question the
    // moment a file is dropped, before it has any idea what is on disk. It used to
    // use `.fbx` as the example; `.fbx` is importable now, by conversion, so the
    // extension with no path at all has to be one neither list mentions.
    let path = fixture("model.rar");
    assert!(!path.exists(), "this test is about the extension, not the bytes");
    assert!(!import::is_supported(&path));
    match import::import_path(&path) {
        Err(ImportError::Unsupported { extension }) => assert_eq!(extension, "rar"),
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
fn a_converted_format_is_offered_whatever_this_machine_has_installed() {
    // The other half of the same question. `is_supported` is what the drop handler,
    // the typed path and the file dialog all ask, and it deliberately knows nothing
    // about whether Blender or assimp is here: an answer that changed with the
    // toolchain would make the UI offer a format one second and refuse it the next.
    // A missing converter belongs in the failure message instead, which
    // `src/import.rs` tests end to end — it has to spawn the tools to find out.
    assert!(import::needs_converter(&fixture("model.fbx")));
    assert!(!import::needs_converter(&fixture("spike.obj")));
    // The dialog filter and the drop handler admit the same set, so a format that
    // can be dragged cannot be hidden by the Open panel.
    let offered = import::importable_extensions();
    for ext in import::SUPPORTED_EXTENSIONS
        .into_iter()
        .chain(import::CONVERTIBLE_EXTENSIONS)
    {
        assert!(offered.contains(&ext), "the dialog filter omits .{ext}");
    }
    assert_eq!(offered.len(), 10, "the two lists and the filter must agree");
}

#[test]
fn the_supported_list_matches_the_hint_the_panel_shows() {
    // The panel renders this string next to the path box; a format added to either
    // list but not named in the hint would be invisible to the user, and vice
    // versa. The exact wording is pinned by a unit test in `src/import.rs`; what
    // belongs here is the cross-check between the lists and the sentence, which is
    // the pair that can drift apart.
    let hint = import::supported_hint();
    for ext in import::SUPPORTED_EXTENSIONS
        .into_iter()
        .chain(import::CONVERTIBLE_EXTENSIONS)
    {
        assert!(
            hint.contains(&format!(".{ext}")),
            "the hint the panel shows omits .{ext}: {hint}"
        );
    }
    assert!(
        hint.contains("Blender") || hint.contains("assimp"),
        "a converted format has to be described as converted, not promised like the \
         formats read in this process: {hint}"
    );
    assert!(import::is_supported(&fixture("spike.obj")));
    assert!(import::is_supported(&fixture("diamond.stl")));
    assert!(import::is_supported(&PathBuf::from("UPPER.OBJ")));
    assert!(!import::is_supported(&PathBuf::from("no extension")));
}

/// A glTF with two materials on one mesh, a scaled and translated node, and no
/// normals: the colours have to survive, the transform has to be normalised
/// away, and the text and binary containers have to agree.
#[test]
fn a_gltf_keeps_its_material_colours_in_both_containers() {
    let text = load("two-tone.gltf");
    let binary = load("two-tone.glb");
    for import in [&text, &binary] {
        assert_eq!(import.mesh.triangle_count(), 8, "{}", import.source);
        assert!((import.mesh.max_radius() - 1.0).abs() < 1e-4, "normalised, whatever the node said");
        let colors = import.mesh.colors.as_ref().expect("a glTF brings vertex colours");
        assert_eq!(colors.len(), import.mesh.vertex_count());
        // The upper half is the warm material and the lower half the cool one.
        // The apex vertices tell them apart: every top-apex corner is red-ish,
        // every bottom-apex corner blue-ish.
        let mut seen = (0, 0);
        for (vertex, color) in import.mesh.vertices.iter().zip(colors) {
            if vertex.position[1] > 0.5 {
                assert!(color[0] > color[2], "top corner coloured {color:?}");
                seen.0 += 1;
            } else if vertex.position[1] < -0.5 {
                assert!(color[2] > color[0], "bottom corner coloured {color:?}");
                seen.1 += 1;
            }
        }
        assert!(seen.0 > 0 && seen.1 > 0, "both apexes are present: {seen:?}");
        // Roughness 0.2 and 0.8 average to a mid gloss, well inside the range.
        let gloss = import.gloss.expect("a glTF material has a roughness");
        assert!(gloss > 8.0 && gloss < 96.0, "{gloss}");
        let hull = import.hull.as_ref().expect("a bipyramid encloses volume");
        assert_eq!((hull.verts.len(), hull.planes.len()), (6, 8));
    }
    assert_eq!(text.mesh.vertices, binary.mesh.vertices);
    assert_eq!(text.mesh.colors, binary.mesh.colors);
}

/// OBJ and STL carry no colour, and say so by leaving the mesh's colours empty
/// — the body is then drawn in the colour the app gives it.
#[test]
fn obj_and_stl_bring_a_hull_but_no_colours() {
    for name in ["spike.obj", "diamond.stl"] {
        let import = load(name);
        assert!(import.mesh.colors.is_none(), "{name}");
        assert!(import.gloss.is_none(), "{name}");
        assert!(import.hull.is_some(), "{name}");
    }
}
