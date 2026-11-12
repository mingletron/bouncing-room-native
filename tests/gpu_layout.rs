//! Cross-checks the WGSL mirrors of `FrameUniforms` and `Light` against the Rust
//! structs in `src/gpu_types.rs`.
//!
//! `FrameUniforms` is duplicated, member for member, in four `.wgsl` files, and naga 30
//! rejects `@offset`/`@size`, so layout there follows declaration order alone. A field
//! added to the Rust struct and one shader, reordered in one file, retyped, or deleted
//! because "nothing reads it" all produce a frame that still validates — the bind group
//! only requires `min_binding_size` — and silently reads the wrong fields.
//!
//! This does the check naga would have done, from the text side: it parses each copy,
//! computes offsets with WGSL's alignment rules, and compares the member list and every
//! offset against `offset_of!` values taken from the Rust structs. That anchors the
//! shaders on Rust's real layout instead of on a third transcription of it, which
//! matters because the sizes in `gpu_types.rs` cannot see a swap of two same-typed
//! fields: `floor_color`/`ceil_color` exchanged leaves `size_of` at 288 and every shader
//! reading the floor's colour out of the CPU's ceiling slot.
//!
//! Three things this still cannot tell you: that a *name* means the same thing on both
//! sides; that a member the shader never reads was dropped from the Rust struct (read
//! the comments in the shaders before deleting anything); and what `gfx.rs`'s attribute
//! tables say about `Instance`, which nothing compares to this file.

use std::path::{Path, PathBuf};

/// The WGSL spelling of each `FrameUniforms` field, in Rust declaration order. The
/// offsets are computed from these types, and the names are matched against
/// `frame_offset` below, which is where the Rust struct is named.
const FRAME_MEMBERS: [(&str, &str); 7] = [
    ("view", "mat4x4<f32>"),
    ("proj", "mat4x4<f32>"),
    ("lights", "array<Light, 3>"),
    ("cam_pos", "vec4<f32>"),
    ("floor_color", "vec4<f32>"),
    ("ceil_color", "vec4<f32>"),
    ("shadow", "vec4<f32>"),
];

const LIGHT_MEMBERS: [(&str, &str); 2] = [("pos", "vec4<f32>"), ("color", "vec4<f32>")];

/// Rust's own answer for where a field sits. An unknown name is a panic rather than a
/// default, and a *renamed* field is a compile error in the arm below: the table in this
/// file and the struct in `src/gpu_types.rs` cannot drift apart silently.
fn frame_offset(name: &str) -> usize {
    match name {
        "view" => std::mem::offset_of!(bouncing_room_native::gpu_types::FrameUniforms, view),
        "proj" => std::mem::offset_of!(bouncing_room_native::gpu_types::FrameUniforms, proj),
        "lights" => std::mem::offset_of!(bouncing_room_native::gpu_types::FrameUniforms, lights),
        "cam_pos" => std::mem::offset_of!(bouncing_room_native::gpu_types::FrameUniforms, cam_pos),
        "floor_color" => {
            std::mem::offset_of!(bouncing_room_native::gpu_types::FrameUniforms, floor_color)
        }
        "ceil_color" => {
            std::mem::offset_of!(bouncing_room_native::gpu_types::FrameUniforms, ceil_color)
        }
        "shadow" => std::mem::offset_of!(bouncing_room_native::gpu_types::FrameUniforms, shadow),
        other => panic!(
            "this test names {other:?} as a FrameUniforms member, and the struct in \
             src/gpu_types.rs has no such field — one of the two is out of date"
        ),
    }
}

fn light_offset(name: &str) -> usize {
    match name {
        "pos" => std::mem::offset_of!(bouncing_room_native::gpu_types::Light, pos),
        "color" => std::mem::offset_of!(bouncing_room_native::gpu_types::Light, color),
        other => panic!(
            "this test names {other:?} as a Light member, and the struct in \
             src/gpu_types.rs has no such field"
        ),
    }
}

/// `(size, align)` for the WGSL types these structs are made of. Anything outside this
/// table fails the test rather than guessing, because a guessed alignment is how a
/// layout check starts agreeing with itself.
fn type_size_align(ty: &str) -> Option<(usize, usize)> {
    let ty = ty.trim();
    if ty == "mat4x4<f32>" {
        return Some((64, 16));
    }
    if ty == "vec4<f32>" {
        return Some((16, 16));
    }
    if ty == "vec3<f32>" {
        return Some((12, 16));
    }
    if ty == "Light" {
        return Some((32, 16));
    }
    if let Some(inner) = ty
        .strip_prefix("array<")
        .and_then(|t| t.strip_suffix('>'))
        .and_then(|t| t.rsplit_once(','))
    {
        let (element, count) = (inner.0.trim(), inner.1.trim());
        let count: usize = count.parse().ok()?;
        let (size, align) = type_size_align(element)?;
        // WGSL rounds an array's stride up to the element alignment, so
        // `array<Light, 3>` is 3 x 32 at a 16 alignment.
        let stride = (size + align - 1) / align * align;
        return Some((stride * count, align));
    }
    None
}

/// Offsets, in declaration order, and the struct's total size, computed with WGSL's
/// alignment rules.
///
/// The total includes the way a struct rounds *its own* size up to the strictest
/// alignment of any member — which is also what Rust's `size_of` reports — because
/// leaving that out would report a short struct as a mismatch and send someone hunting
/// for a layout bug that is not there.
fn layout(members: &[(impl AsRef<str>, impl AsRef<str>)]) -> (Vec<usize>, usize) {
    let mut offset = 0usize;
    let mut max_align = 1usize;
    let mut offsets = Vec::with_capacity(members.len());
    for (_, ty) in members {
        let (size, align) = type_size_align(ty.as_ref())
            .unwrap_or_else(|| panic!("type {:?} is not in the size table — add it deliberately", ty.as_ref()));
        offset = offset.div_ceil(align) * align;
        offsets.push(offset);
        offset += size;
        max_align = max_align.max(align);
    }
    (offsets, offset.div_ceil(max_align) * max_align)
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Every shader in the tree, by name. Discovered rather than listed on purpose: a fifth
/// file that mirrors these structs is exactly the drift this test exists to catch, and a
/// hand-written list only checks the files someone remembered to put in it.
fn shaders() -> Vec<PathBuf> {
    let dir = repo_root().join("shaders");
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|path| path.extension().is_some_and(|e| e == "wgsl"))
        .collect();
    files.sort();
    assert!(
        !files.is_empty(),
        "{} holds no .wgsl files, so every check in this file would pass by reading \
         nothing at all",
        dir.display()
    );
    files
}

fn read(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// Strip whole-line and trailing `//` comments from shader text. WGSL has no block
/// comments and no string literals, so a `//` always starts one that runs to the end of
/// its line — and a comment that mentions `struct FrameUniforms {`, holds a brace, or
/// contains a comma must not steer the parse.
fn strip_comments(text: &str) -> String {
    text.lines()
        .map(|line| match line.find("//") {
            Some(cut) => &line[..cut],
            None => line,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Pull a `struct NAME { … }` body out of already-comment-stripped shader text and
/// return its members as `(name, type)` pairs, in declaration order.
///
/// Takes text rather than a path so the parser itself can be tested for mutation — see
/// `the_parser_notices_every_mutation` below. `origin` only names the source in a panic.
fn struct_members(text: &str, name: &str, origin: &str) -> Vec<(String, String)> {
    let header = format!("struct {name} {{");
    let Some(start) = text.find(&header) else {
        panic!("{origin} declares no `struct {name}`");
    };
    let body_start = start + header.len();
    // Brace matching, so a nested block cannot end the struct early.
    let mut depth = 1usize;
    let mut end = body_start;
    for (i, ch) in text[body_start..].char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    end = body_start + i;
                    break;
                }
            }
            _ => {}
        }
    }
    // Split on commas at angle-bracket depth zero. `lights: array<Light, 3>` holds a
    // comma inside its own type, and a plain `split(',')` turns that one member into
    // two — which is how a correct shader reads as a broken one.
    let mut parts: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut generics = 0usize;
    for ch in text[body_start..end].chars() {
        match ch {
            '<' => {
                generics += 1;
                current.push(ch);
            }
            '>' => {
                generics = generics.saturating_sub(1);
                current.push(ch);
            }
            ',' if generics == 0 => parts.push(std::mem::take(&mut current)),
            _ => current.push(ch),
        }
    }
    parts.push(current);
    parts
        .into_iter()
        .map(|part| part.trim().to_owned())
        .filter(|part| !part.is_empty())
        .map(|part| {
            // A member of a uniform block is `name: type`, with no `@location` and no
            // `@internal`-style attributes in these files.
            let (name, ty) = part
                .split_once(':')
                .unwrap_or_else(|| panic!("{origin}: member {part:?} has no `:`"));
            (name.trim().to_owned(), ty.trim().to_owned())
        })
        .collect()
}

/// Compare one parsed copy of a struct against the Rust declaration it mirrors: the
/// member list first, then each member's offset, then the whole size.
fn assert_copy_matches(
    found: &[(String, String)],
    want: &[(&str, &str)],
    offset_of: impl Fn(&str) -> usize,
    rust_size: usize,
    origin: &str,
    label: &str,
) {
    let found_pairs: Vec<(String, String)> = found.to_vec();
    let want_pairs: Vec<(String, String)> = want
        .iter()
        .map(|(n, t)| ((*n).to_owned(), (*t).to_owned()))
        .collect();
    assert_eq!(
        found_pairs, want_pairs,
        "{origin}: {label} must match src/gpu_types.rs member for member, in the same \
         order — WGSL lays out by declaration order and naga 30 rejects @offset, so one \
         deleted, inserted, reordered or retyped member shifts every field after it in \
         this pipeline alone"
    );
    // Offsets computed from the *parsed* members, so the arithmetic runs on the file
    // being checked rather than on this file's own table.
    let (offsets, total) = layout(found);
    for ((name, _), offset) in found.iter().zip(offsets.iter()) {
        assert_eq!(
            *offset,
            offset_of(name),
            "{origin}: `{name}` sits at byte {offset} by WGSL's alignment rules, but \
             src/gpu_types.rs puts it at {}; one of the two moved — and if a member whose \
             type changed kept its name, that is the one to look at",
            offset_of(name)
        );
    }
    assert_eq!(
        total, rust_size,
        "{origin}: {label} adds up to {total} bytes, but Rust's is {rust_size} — the \
         shader copy and the CPU struct are different sizes, and the bind group will not \
         notice (it only sets min_binding_size)"
    );
}

#[test]
fn every_shader_copy_of_frame_uniforms_matches_the_rust_struct() {
    let mut checked = 0usize;
    for path in shaders() {
        let text = strip_comments(&read(&path));
        if !text.contains("struct FrameUniforms {") {
            continue;
        }
        checked += 1;
        let found = struct_members(&text, "FrameUniforms", &path.display().to_string());
        assert_copy_matches(
            &found,
            &FRAME_MEMBERS,
            frame_offset,
            std::mem::size_of::<bouncing_room_native::gpu_types::FrameUniforms>(),
            &path.display().to_string(),
            "FrameUniforms",
        );
    }
    // The contract's rule is that `FrameUniforms` is mirrored in four files. Pinning a
    // minimum keeps the check honest without listing the files by name: a fifth copy is
    // found automatically, a vanished copy is a failure rather than a quiet pass.
    assert!(
        checked >= 4,
        "only {checked} shader files declare FrameUniforms; the contract says four \
         (lit, flat, line, trail), so either a copy has gone or the parse found nothing \
         and every check above passed by reading nothing at all"
    );
}

#[test]
fn the_light_struct_agrees_between_shaders_and_rust() {
    let mut checked = 0usize;
    for path in shaders() {
        let text = strip_comments(&read(&path));
        if !text.contains("struct Light {") {
            continue;
        }
        checked += 1;
        let found = struct_members(&text, "Light", &path.display().to_string());
        assert_copy_matches(
            &found,
            &LIGHT_MEMBERS,
            light_offset,
            std::mem::size_of::<bouncing_room_native::gpu_types::Light>(),
            &path.display().to_string(),
            "Light",
        );
    }
    // Each file that mirrors FrameUniforms declares its *own* Light, and the stride of
    // `array<Light, 3>` at byte 128 comes from that local copy: a `vec3` slipped into
    // `line.wgsl` moves `cam_pos` and every field after it for that pipeline alone.
    assert!(
        checked >= 4,
        "only {checked} shader files declare Light, and there should be one per copy of \
         FrameUniforms"
    );
}

#[test]
fn the_offsets_in_the_shader_comments_are_where_rust_puts_them() {
    // Every shader copy writes its members' byte offsets in trailing comments, and the
    // Rust struct documents the same numbers in prose. Those comments are the only thing
    // that tells a reader of `line.wgsl` what they are binding, so they are pinned here
    // against `offset_of!` — the same anchor the struct copies are checked against.
    let want: Vec<(String, usize)> = FRAME_MEMBERS
        .iter()
        .map(|(n, _)| ((*n).to_owned(), frame_offset(n)))
        .collect();
    let mut checked = 0usize;
    for path in shaders() {
        let raw = read(&path);
        if !raw.contains("struct FrameUniforms {") {
            continue;
        }
        checked += 1;
        for (name, offset) in &want {
            let line = raw
                .lines()
                .find(|line| line.trim_start().starts_with(&format!("{name}:")))
                .unwrap_or_else(|| panic!("{}: no member line for {name}", path.display()));
            // Every run of digits on that line, read out of the text rather than
            // assumed: `// byte 240, read only by lit` has the number glued to a comma,
            // and `vec4<f32>` has digits of its own that are not an offset.
            let claimed: Vec<usize> = line
                .split(|c: char| !c.is_ascii_digit())
                .filter(|run| !run.is_empty())
                .filter_map(|run| run.parse::<usize>().ok())
                .collect();
            assert!(
                claimed.iter().any(|n| n == offset),
                "{}: `{name}` is documented as byte {claimed:?}, but Rust puts it at \
                 {offset} — update the comment with the struct, not after it",
                path.display()
            );
        }
    }
    assert!(checked >= 4, "expected four copies with offset comments");
}

#[test]
fn the_parser_notices_every_mutation() {
    // Nothing in this file is worth anything if the hand-written parser stopped seeing
    // members, and a parser bug makes every check above pass vacuously. So the parser
    // gets tested the way the shaders are: real text, then four mutations, each of which
    // has to change what it reports.
    let real = read(&repo_root().join("shaders/lit.wgsl"));
    let parsed_real = struct_members(&strip_comments(&real), "FrameUniforms", "lit.wgsl");
    assert_eq!(
        parsed_real.len(),
        FRAME_MEMBERS.len(),
        "lit.wgsl's copy did not parse into {} members",
        FRAME_MEMBERS.len()
    );

    // The same struct with one member line dropped, two lines exchanged, a type changed,
    // and an extra member added. Each is built in memory from the real text, so this
    // stays honest if the shader itself changes.
    let mutations: Vec<(String, &str)> = {
        let lights_line = real
            .lines()
            .find(|l| l.contains("lights: array<Light"))
            .expect("lit.wgsl declares lights")
            .to_owned();
        let shadow_line = real
            .lines()
            .find(|l| l.trim_start().starts_with("shadow:"))
            .expect("lit.wgsl declares shadow")
            .to_owned();
        vec![
            (real.replace(&lights_line, ""), "a deleted member"),
            (
                real.replace(&shadow_line, "shadow: vec3<f32>,"),
                "a changed member type",
            ),
            (
                real.replace("cam_pos: vec4<f32>", "cam_pos: vec4<f32>,\n  unused: vec4<f32>,"),
                "an inserted member",
            ),
            (
                real.replace(&lights_line, &lights_line.replace("lights:", "renamed:")),
                "a renamed member",
            ),
        ]
    };
    for (text, what) in mutations {
        assert_ne!(
            text, real,
            "the mutation for {what} did not change lit.wgsl's text, so the parser was \
             never given anything different to notice"
        );
        assert_ne!(
            struct_members(&strip_comments(&text), "FrameUniforms", "mutant"),
            parsed_real,
            "the parser read a copy with {what} as if it were untouched — which means \
             every other test in this file is checking nothing"
        );
    }
}

#[test]
fn no_shader_uses_derivative_functions() {
    // The contract forbids `dpdx`/`dpdy`/`fwidth` everywhere, not just in the grid: grid
    // antialiasing and shadow filtering both stay portable by using CPU-computed widths
    // and `textureSampleCompareLevel` instead. Comments are stripped with the same rule
    // the struct parse uses, so a comment that merely mentions a forbidden name cannot
    // fail this, and an inline one cannot hide a real call.
    for path in shaders() {
        let code = strip_comments(&read(&path));
        for forbidden in ["dpdx", "dpdy", "fwidth"] {
            assert!(
                !code.contains(forbidden),
                "{} uses {forbidden} — the contract prohibits derivative functions in \
                 every shader",
                path.display()
            );
        }
    }
}
