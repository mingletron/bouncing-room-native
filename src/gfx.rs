//! The GPU layer: pipelines, buffers and the draw order.
//!
//! Port of the WebGL build's `src/render.js`. The structure carries over
//! one-to-one — three shader programs (lit / line / flat), one interleaved
//! vertex layout, and a fixed draw order (room fill, ground discs, objects,
//! lines) — but two mechanisms do not survive the move and are replaced:
//!
//! * **Polygon offset is gone.** `glPolygonOffset` never became part of WebGPU,
//!   and wgpu 30 exposes no equivalent, so the coplanar depth conflicts it used
//!   to settle are settled by moving geometry instead: the room outline is drawn
//!   a hair inside the box, and wireframe shells a hair outside the body.
//! * **Per-draw uniforms become per-instance vertex attributes.** Everything
//!   drawn is an [`Instance`] in one buffer written once a frame, and all the
//!   bodies of one shape are a single instanced draw — which is what lets the
//!   room hold thousands of bodies rather than dozens.
//!
//! Two things here have no counterpart in the WebGL build at all: the key
//! light's **shadow cube** (six depth-only passes from the light, sampled by
//! the lit shader), and a **dynamic line list** for trails and sparks.
//!
//! The room's "cull the faces pointing at you" trick needs care too: winding
//! conventions differ between GL and WebGPU, and which way they differ is
//! precisely the kind of thing that reads plausibly and is wrong. So
//! `front_face` is a parameter rather than a literal, and `src/bin/probe.rs`
//! settles it by checking that bodies — which live *inside* the room — are
//! actually visible. With wgpu 30 the answer is `Ccw`, on Metal and on Vulkan
//! alike — the same handedness GL used, not the flip the contract predicted.

use std::ops::Range;

use crate::geom::{Mesh, ShapeDef};
use crate::gpu_types::{self as gt, FrameUniforms, Instance, Light, LineVertex};
use crate::sim::Body;

use glam::{Mat4, Vec3, Vec4};

/// Multisampling for the 3D pass. The UI draws afterwards into the resolved
/// target, so egui itself is not multisampled (it does not need to be).
pub const SAMPLE_COUNT: u32 = 4;

/// Instances the buffer starts with room for. It doubles when a frame needs more.
const INSTANCES_START: usize = 1024;
/// Line vertices the trail buffer starts with room for.
const LINES_START: usize = 4096;

/// The room outline is drawn scaled toward the room centre by this much, so its
/// lines sit fractionally in front of the coplanar walls. This replaces the
/// polygon offset the WebGL build used for the same job.
const ROOM_OUTLINE_INWARD: f32 = 0.9965;

/// Wireframe shells are pushed out by this much so they clear the solid they
/// outline instead of z-fighting it.
const WIREFRAME_SHELL: f32 = 1.006;

/// A ring smaller than this fraction of its distance from the camera is dropped:
/// a handful of segments reads as a mark on the floor, not as a splash.
const RING_MIN_APPARENT_SIZE: f32 = 0.04;
/// A ring more edge-on than this is dropped: a collapsed circle reads as stray
/// geometry.
const RING_MIN_FACING: f32 = 0.18;

/// Edge length of each face of the shadow cube, in texels.
const SHADOW_SIZE: u32 = 1024;
/// The shadow cube's depth range. The far plane only has to clear the room's
/// longest diagonal from the key light.
const SHADOW_NEAR: f32 = 0.08;
const SHADOW_FAR: f32 = 30.0;
/// How dark a shadow gets: the fraction of the key light it removes.
const SHADOW_STRENGTH: f32 = 0.82;
/// Subtracted from the reference depth, on top of the pipeline's own slope bias.
const SHADOW_BIAS: f32 = 0.0004;
/// Each cube face's view-projection sits in its own slot of one uniform buffer,
/// picked by dynamic offset. 256 covers every backend's offset alignment.
const FACE_SLOT_BYTES: u64 = 256;

const LIT_SRC: &str = include_str!("../shaders/lit.wgsl");
const LINE_SRC: &str = include_str!("../shaders/line.wgsl");
const FLAT_SRC: &str = include_str!("../shaders/flat.wgsl");
const TRAIL_SRC: &str = include_str!("../shaders/trail.wgsl");
const SHADOW_SRC: &str = include_str!("../shaders/shadow.wgsl");

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

#[derive(Copy, Clone, Debug)]
pub struct GfxOptions {
    /// Which winding counts as front for the room's interior-culling pass.
    pub room_front_face: wgpu::FrontFace,
}

impl Default for GfxOptions {
    fn default() -> Self {
        // The port contract reasoned that WebGPU's framebuffer Y runs the other
        // way from GL's, so the room's unchanged vertex order would need the
        // opposite handedness to GL's CCW — that is, `Cw`. Rendering says
        // otherwise: with `Cw` and `cull_mode: Front` the near walls survive and
        // the room reads as a solid box from outside, hiding every body. `Ccw` —
        // the same handedness GL used — is what culls the near walls and produces
        // the cutaway interior. `src/bin/probe.rs` re-checks this on every run.
        GfxOptions {
            room_front_face: wgpu::FrontFace::Ccw,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct RoomDraw {
    pub mesh: usize,
    pub wall: [f32; 3],
    pub floor: [f32; 3],
    pub ceil: [f32; 3],
    pub grid_spacing: f32,
    pub edge_color: [f32; 4],
    /// Room half extents: needed to shrink the outline toward the room centre.
    pub half: [f32; 3],
}

/// One soft radial disc: a ground shadow or a ceiling light panel.
#[derive(Copy, Clone, Debug)]
pub struct Disc {
    pub center: Vec3,
    /// The disc lies in the plane perpendicular to this; it also decides which
    /// side of the plane the camera has to be on for the disc to be drawn.
    pub normal: Vec3,
    pub radius: f32,
    pub alpha: f32,
    /// Fraction of the radius out to which the disc is fully opaque.
    pub inner: f32,
    pub color: [f32; 3],
}

#[derive(Copy, Clone, Debug)]
pub struct Ring {
    pub pos: Vec3,
    pub n: Vec3,
    pub color: [f32; 3],
    pub base_radius: f32,
    /// Progress through the ring's life, 0..1.
    pub t: f32,
}

#[derive(Debug)]
pub struct Scene<'a> {
    pub view: Mat4,
    pub proj: Mat4,
    pub eye: Vec3,
    pub lights: [Light; gt::LIGHT_COUNT],
    pub background: [f32; 3],
    pub room: RoomDraw,
    pub shapes: &'a [ShapeDef],
    /// GPU mesh handle per shape, index-aligned with `shapes`.
    pub shape_mesh: &'a [usize],
    pub bodies: &'a [Body],
    pub blobs: &'a [Disc],
    pub panels: &'a [Disc],
    pub rings: &'a [Ring],
    pub quad_mesh: usize,
    pub ring_mesh: usize,
    pub wireframe: bool,
    /// Bodies cast real shadows from the key light. When this is on the caller
    /// normally passes no `blobs`, which are the stand-in for exactly that.
    pub shadows: bool,
    /// World-space line segments, two vertices each: trails and sparks.
    pub lines: &'a [LineVertex],
    /// A body to draw a bright wire shell around — the one being held.
    pub outlined: Option<usize>,
}

struct GpuMesh {
    vertex_buffer: wgpu::Buffer,
    /// One RGBA8 per vertex, white unless the mesh brought its own.
    color_buffer: wgpu::Buffer,
    index_buffer: Option<(wgpu::Buffer, u32)>,
    line_buffer: Option<(wgpu::Buffer, u32)>,
    vertex_count: u32,
}

struct Pipelines {
    /// lit, with the room's interior culling.
    room: wgpu::RenderPipeline,
    /// lit, double sided, for every primitive.
    object: wgpu::RenderPipeline,
    flat: wgpu::RenderPipeline,
    line: wgpu::RenderPipeline,
    trail: wgpu::RenderPipeline,
    shadow: wgpu::RenderPipeline,
}

/// The MSAA colour and depth attachments for one output size.
struct Targets {
    extent: (u32, u32, wgpu::TextureFormat),
    msaa: wgpu::Texture,
    depth: wgpu::Texture,
}

/// Where each kind of thing landed in this frame's instance buffer.
#[derive(Default)]
struct Staged {
    discs: Range<u32>,
    /// `(mesh handle, instances)` — one entry per shape that has bodies.
    bodies: Vec<(usize, Range<u32>)>,
    outline: u32,
    wires: Vec<(usize, Range<u32>)>,
    rings: Range<u32>,
}

pub struct Gfx {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface_format: wgpu::TextureFormat,
    /// Kept because the windowed path has to ask this same adapter what the
    /// surface is able to present, and a different adapter might say something
    /// else.
    adapter: wgpu::Adapter,
    options: GfxOptions,
    pipelines: Pipelines,
    frame_buffer: wgpu::Buffer,
    frame_bg: wgpu::BindGroup,
    instance_buffer: wgpu::Buffer,
    instance_capacity: usize,
    instances: Vec<Instance>,
    staged: Staged,
    line_buffer: wgpu::Buffer,
    line_capacity: usize,
    meshes: Vec<GpuMesh>,
    /// Attachments per output size. Two sizes are live at once while a clip is
    /// being recorded — the window's and the clip's — and rebuilding them every
    /// frame as the two alternate would be a texture allocation per frame.
    targets: Vec<Targets>,
    /// One view per face of the shadow cube, to render into.
    shadow_faces: Vec<wgpu::TextureView>,
    /// The cube as the lit shader samples it.
    shadow_bg: wgpu::BindGroup,
    face_buffer: wgpu::Buffer,
    face_bg: wgpu::BindGroup,
}

/// Where the 3D pass resolves its MSAA output.
///
/// The extent travels with it because a `TextureView` carries no size accessor
/// in wgpu 30 — the caller knows the extent (it configured the surface, or made
/// the texture) and hands it over.
pub struct RenderTarget<'a> {
    pub view: &'a wgpu::TextureView,
    pub format: wgpu::TextureFormat,
    pub width: u32,
    pub height: u32,
}

/// A u16 index list as the bytes `Queue::write_buffer` will accept.
///
/// wgpu requires both a buffer's size and the length of a write into it to be a
/// multiple of `COPY_BUFFER_ALIGNMENT` (4 bytes). An index list costs 2 bytes an
/// entry, so an *odd* number of entries lands on 2 mod 4 and the copy is rejected
/// outright. Every mesh this renderer had before files existed happened to have an
/// even triangle count, which is why it took an imported object to find this:
/// [`crate::import`] samples faces out of oversized meshes and is content to leave
/// an odd number of them behind.
///
/// The padding is never drawn. `GpuMesh` stores the real index count alongside the
/// buffer, and that count is what `draw_indexed` is given.
fn padded_index_bytes(indices: &[u16]) -> Vec<u8> {
    let align = wgpu::COPY_BUFFER_ALIGNMENT as usize;
    let mut bytes = bytemuck::cast_slice(indices).to_vec();
    bytes.resize(bytes.len().div_ceil(align) * align, 0);
    bytes
}

/// The six view-projections of a shadow cube centred on `light`, in the cube
/// map's own face order (+X, -X, +Y, -Y, +Z, -Z).
///
/// Cube maps are left-handed: on the +X face, +Y is up and -Z is to the right.
/// A right-handed view would render each face mirrored, so these are built with
/// the left-handed look-to and projection, and the lit shader can then sample
/// with the plain world-space direction from the light.
pub fn shadow_face_matrices(light: Vec3) -> [Mat4; 6] {
    use glam::camera::lh;
    let proj = lh::proj::directx::perspective(
        std::f32::consts::FRAC_PI_2,
        1.0,
        SHADOW_NEAR,
        SHADOW_FAR,
    );
    let faces = [
        (Vec3::X, Vec3::Y),
        (Vec3::NEG_X, Vec3::Y),
        (Vec3::Y, Vec3::NEG_Z),
        (Vec3::NEG_Y, Vec3::Z),
        (Vec3::Z, Vec3::Y),
        (Vec3::NEG_Z, Vec3::Y),
    ];
    faces.map(|(dir, up)| proj * lh::view::look_to_mat4(light, dir, up))
}

impl Gfx {
    /// Create a device and every pipeline against an existing `instance`.
    ///
    /// The instance belongs to the caller because wgpu 30 binds it to a window
    /// system display handle (`InstanceDescriptor::display`, supplied by winit),
    /// and that handle has to be in place before a surface can be created — so
    /// the windowed path builds the instance and surface first and hands the
    /// surface here to pick a format-capable adapter, while the headless probe
    /// passes `None`.
    ///
    /// The room and the bodies get a pipeline each although both run the lit
    /// shader, because culling is pipeline state in WebGPU: the room needs
    /// front-face culling and the primitives need none.
    pub fn new(
        instance: &wgpu::Instance,
        compatible_surface: Option<&wgpu::Surface<'_>>,
        options: GfxOptions,
    ) -> Result<Gfx, String> {
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            compatible_surface,
            ..Default::default()
        }))
        .map_err(|e| format!("no suitable GPU adapter: {e:?}"))?;

        // Default limits are enough, and asking for more is what once kept this
        // off Linux: a raised `max_uniform_buffer_binding_size` that Metal
        // granted and Vulkan drivers refuse. Nothing here binds more than one
        // small uniform block at a time.
        let limits = wgpu::Limits::default();

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("bouncing-room"),
            required_features: wgpu::Features::empty(),
            required_limits: limits,
            ..Default::default()
        }))
        .map_err(|e| format!("device request failed: {e}"))?;

        // The shaders' tonemap produces display-referred values, exactly as the
        // WebGL build's did — and that build's backbuffer was not sRGB-encoded,
        // so its numbers were stored literally. Choosing a non-sRGB target keeps
        // the port's pixels the same as the browser's; on an sRGB target the
        // hardware would encode them a second time and the room would come out
        // washed out. If a surface offers no unorm format, fall back to sRGB.
        let surface_format = match compatible_surface {
            Some(surface) => {
                let caps = surface.get_capabilities(&adapter);
                pick_surface_format(&caps.formats)
            }
            // Headless probe: it creates the target texture itself, so the
            // pipelines have to agree with whatever it picks.
            None => wgpu::TextureFormat::Rgba8Unorm,
        };

        let module = |label: &str, source: &str| {
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            })
        };
        let lit = module("lit", LIT_SRC);
        let line_src = module("line", LINE_SRC);
        let flat = module("flat", FLAT_SRC);
        let trail = module("trail", TRAIL_SRC);
        let shadow = module("shadow", SHADOW_SRC);

        let bgl_frame = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("frame"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    // wgpu 30: `BufferBindingSize` is gone; `BufferSize::new`
                    // already returns the `Option` this field wants.
                    min_binding_size: wgpu::BufferSize::new(
                        std::mem::size_of::<FrameUniforms>() as u64,
                    ),
                },
                count: None,
            }],
        });
        // The shadow cube as the lit shader reads it.
        let bgl_shadow_read = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow read"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::Cube,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
            ],
        });
        // One cube face's view-projection, chosen per pass by dynamic offset.
        let bgl_face = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow face"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: wgpu::BufferSize::new(64),
                },
                count: None,
            }],
        });

        // wgpu 30: a pipeline layout entry is `Option<&BindGroupLayout>`, and
        // push constants are now `immediate_size` — a byte count. None are used.
        let lit_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("lit"),
            bind_group_layouts: &[Some(&bgl_frame), Some(&bgl_shadow_read)],
            immediate_size: 0,
        });
        let plain_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("plain"),
            bind_group_layouts: &[Some(&bgl_frame)],
            immediate_size: 0,
        });
        let shadow_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shadow"),
            bind_group_layouts: &[Some(&bgl_face)],
            immediate_size: 0,
        });

        let depth = || wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            // wgpu 30 made both of these `Option`: `None` is only legal when the
            // format has no depth component.
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Less),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        };
        let multisample = wgpu::MultisampleState {
            count: SAMPLE_COUNT,
            mask: 0xFFFF_FFFF,
            alpha_to_coverage_enabled: false,
        };
        let opaque_target = |format| {
            Some(wgpu::ColorTargetState {
                format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })
        };
        let blended_target = |format| {
            Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })
        };

        // Three vertex buffers for every mesh pipeline: the mesh's positions and
        // normals, its per-vertex colours, and the per-instance block. A shader
        // that does not read an attribute simply does not declare it — the flat
        // and line shaders ignore the normal and the vertex colour — which is
        // what keeps one layout for all of them. `buffers` is a list of
        // `Option<VertexBufferLayout>` in wgpu 30, and the attribute arrays have
        // to outlive the layouts, so they are locals rather than temporaries.
        let vertex_attrs = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3];
        let color_attrs = wgpu::vertex_attr_array![8 => Unorm8x4];
        let instance_attrs = wgpu::vertex_attr_array![
            2 => Float32x4, 3 => Float32x4, 4 => Float32x4, 5 => Float32x4,
            6 => Float32x4, 7 => Float32x4
        ];
        let vertex_layouts = [
            Some(wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<crate::geom::Vertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &vertex_attrs,
            }),
            Some(wgpu::VertexBufferLayout {
                array_stride: 4,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &color_attrs,
            }),
            Some(wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<Instance>() as u64,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &instance_attrs,
            }),
        ];
        let trail_attrs = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x4];
        let trail_layouts = [Some(wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<LineVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &trail_attrs,
        })];

        let lit_pipeline = |label: &str, primitive: wgpu::PrimitiveState| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&lit_layout),
                vertex: wgpu::VertexState {
                    module: &lit,
                    entry_point: Some("vs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &vertex_layouts,
                },
                primitive,
                depth_stencil: Some(depth()),
                multisample: multisample.clone(),
                fragment: Some(wgpu::FragmentState {
                    module: &lit,
                    entry_point: Some("fs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[opaque_target(surface_format)],
                }),
                cache: None,
                multiview_mask: None,
            })
        };
        let room = lit_pipeline(
            "room",
            wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: options.room_front_face,
                cull_mode: Some(wgpu::Face::Front),
                ..Default::default()
            },
        );
        let object = lit_pipeline(
            "object",
            wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                // Primitives are double sided: their normals are analytic and
                // always face out, and the shader flips any normal that faces
                // away from the camera, so culling would only punch holes.
                cull_mode: None,
                ..Default::default()
            },
        );

        let flat_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("flat"),
            layout: Some(&plain_layout),
            vertex: wgpu::VertexState {
                module: &flat,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &vertex_layouts,
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            // Discs blend over the surface they sit on and never write depth.
            depth_stencil: Some(wgpu::DepthStencilState {
                depth_write_enabled: Some(false),
                ..depth()
            }),
            multisample: multisample.clone(),
            fragment: Some(wgpu::FragmentState {
                module: &flat,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[blended_target(surface_format)],
            }),
            cache: None,
            multiview_mask: None,
        });

        // Lines sit coplanar with, or just inside, the surfaces they describe, so
        // equal depth has to win; and they never write depth.
        let line_depth = || wgpu::DepthStencilState {
            depth_write_enabled: Some(false),
            depth_compare: Some(wgpu::CompareFunction::LessEqual),
            ..depth()
        };
        let line_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("line"),
            layout: Some(&plain_layout),
            vertex: wgpu::VertexState {
                module: &line_src,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &vertex_layouts,
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(line_depth()),
            multisample: multisample.clone(),
            fragment: Some(wgpu::FragmentState {
                module: &line_src,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[blended_target(surface_format)],
            }),
            cache: None,
            multiview_mask: None,
        });
        let trail_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("trail"),
            layout: Some(&plain_layout),
            vertex: wgpu::VertexState {
                module: &trail,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &trail_layouts,
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(line_depth()),
            multisample,
            fragment: Some(wgpu::FragmentState {
                module: &trail,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[blended_target(surface_format)],
            }),
            cache: None,
            multiview_mask: None,
        });
        // Depth only: no fragment stage and no colour target. The slope bias is
        // what keeps a surface from shadowing itself at a grazing angle.
        let shadow_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("shadow"),
            layout: Some(&shadow_layout),
            vertex: wgpu::VertexState {
                module: &shadow,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &vertex_layouts,
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                bias: wgpu::DepthBiasState {
                    constant: 2,
                    slope_scale: 2.0,
                    clamp: 0.0,
                },
                ..depth()
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: None,
            cache: None,
            multiview_mask: None,
        });

        let frame_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frame uniforms"),
            size: std::mem::size_of::<FrameUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let frame_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("frame"),
            layout: &bgl_frame,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: frame_buffer.as_entire_binding(),
            }],
        });

        // The shadow cube: one six-layer depth texture, viewed once as a cube for
        // sampling and once per layer for rendering.
        let shadow_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shadow cube"),
            size: wgpu::Extent3d {
                width: SHADOW_SIZE,
                height: SHADOW_SIZE,
                depth_or_array_layers: 6,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let shadow_faces = (0..6)
            .map(|layer| {
                shadow_texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("shadow face"),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: layer,
                    array_layer_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();
        let shadow_cube = shadow_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("shadow cube"),
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..Default::default()
        });
        let shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("shadow"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            // Lit where the surface is no farther from the light than what the
            // cube recorded in that direction.
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });
        let shadow_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shadow read"),
            layout: &bgl_shadow_read,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&shadow_cube),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&shadow_sampler),
                },
            ],
        });
        let face_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shadow faces"),
            size: FACE_SLOT_BYTES * 6,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let face_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shadow face"),
            layout: &bgl_face,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &face_buffer,
                    offset: 0,
                    size: wgpu::BufferSize::new(64),
                }),
            }],
        });

        let instance_buffer = make_vertex_buffer(
            &device,
            "instances",
            INSTANCES_START * std::mem::size_of::<Instance>(),
        );
        let line_buffer = make_vertex_buffer(
            &device,
            "trail lines",
            LINES_START * std::mem::size_of::<LineVertex>(),
        );

        Ok(Gfx {
            device,
            queue,
            surface_format,
            adapter,
            options,
            pipelines: Pipelines {
                room,
                object,
                flat: flat_pipeline,
                line: line_pipeline,
                trail: trail_pipeline,
                shadow: shadow_pipeline,
            },
            frame_buffer,
            frame_bg,
            instance_buffer,
            instance_capacity: INSTANCES_START,
            instances: Vec::new(),
            staged: Staged::default(),
            line_buffer,
            line_capacity: LINES_START,
            meshes: Vec::new(),
            targets: Vec::new(),
            shadow_faces,
            shadow_bg,
            face_buffer,
            face_bg,
        })
    }

    /// The adapter this device came from. The windowed path needs it to ask the
    /// surface which formats it can present, which is the same adapter the
    /// pipelines were chosen for or those answers do not apply.
    pub fn adapter(&self) -> &wgpu::Adapter {
        &self.adapter
    }

    /// The adapter's name, for the HUD's status line — this port's equivalent of
    /// the WebGL build reporting its GL context name.
    pub fn adapter_name(&self) -> String {
        self.adapter.get_info().name.clone()
    }

    /// The winding the room pipeline was actually built with. The probe reads
    /// this instead of trusting the option it passed in, so a report describes
    /// the renderer rather than the request.
    pub fn room_front_face(&self) -> wgpu::FrontFace {
        self.options.room_front_face
    }

    /// Upload a mesh once. The same buffers serve its fill and its wireframe.
    pub fn upload_mesh(&mut self, mesh: &Mesh) -> usize {
        let vertex_bytes = bytemuck::cast_slice::<_, u8>(&mesh.vertices);
        let vertex_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(mesh.name),
            size: vertex_bytes.len() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&vertex_buffer, 0, vertex_bytes);

        // Every mesh gets a colour per vertex, white when it brought none, so
        // one pipeline layout serves the generated primitives and a coloured
        // import alike. Four bytes a vertex is a cheap price for that.
        let white;
        let colors: &[[u8; 4]] = match &mesh.colors {
            Some(colors) if colors.len() == mesh.vertices.len() => colors,
            _ => {
                white = vec![[255u8; 4]; mesh.vertices.len()];
                &white
            }
        };
        let color_bytes = bytemuck::cast_slice::<_, u8>(colors);
        let color_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vertex colours"),
            size: (color_bytes.len().max(4)) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&color_buffer, 0, color_bytes);

        let index_buffer = mesh.indices.as_ref().map(|idx| {
            let bytes = padded_index_bytes(idx);
            let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("indices"),
                // A zero-size buffer is not allowed, hence the floor of one padded
                // record. The length is already a multiple of the copy alignment.
                size: bytes.len().max(wgpu::COPY_BUFFER_ALIGNMENT as usize) as u64,
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.queue.write_buffer(&buf, 0, &bytes);
            (buf, idx.len() as u32)
        });

        let line_buffer = (!mesh.edges.is_empty()).then(|| {
            let bytes = padded_index_bytes(&mesh.edges);
            let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("edges"),
                size: bytes.len() as u64,
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.queue.write_buffer(&buf, 0, &bytes);
            (buf, mesh.edges.len() as u32)
        });

        let handle = self.meshes.len();
        self.meshes.push(GpuMesh {
            vertex_buffer,
            color_buffer,
            index_buffer,
            line_buffer,
            vertex_count: mesh.vertex_count() as u32,
        });
        handle
    }

    /// The attachments for this output size, made on first use. Returns their
    /// index in `self.targets`.
    fn ensure_targets(&mut self, width: u32, height: u32, format: wgpu::TextureFormat) -> usize {
        let extent = (width, height, format);
        if let Some(found) = self.targets.iter().position(|t| t.extent == extent) {
            return found;
        }
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let msaa = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("msaa colour"),
            size,
            mip_level_count: 1,
            sample_count: SAMPLE_COUNT,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("depth"),
            size,
            mip_level_count: 1,
            sample_count: SAMPLE_COUNT,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        // Two sizes at most are ever in use together (the window and a clip
        // being recorded), so anything beyond that is a size the window has
        // since been resized away from: the oldest goes.
        if self.targets.len() >= 2 {
            self.targets.remove(0);
        }
        self.targets.push(Targets { extent, msaa, depth });
        self.targets.len() - 1
    }

    /// Record the shadow passes and the whole 3D pass into `encoder`, resolving
    /// MSAA onto `target`.
    ///
    /// The caller then opens a second pass on the same target with load op `Load`
    /// for the UI, which draws without multisampling.
    ///
    /// The buffers this writes are read when the encoder is *submitted*, so two
    /// calls must not share one submission: the second would overwrite what the
    /// first staged. [`Gfx::render_to_pixels`] submits its own for that reason.
    pub fn render(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        target: &RenderTarget<'_>,
        scene: &Scene<'_>,
    ) {
        let width = target.width.max(1);
        let height = target.height.max(1);
        let targets = self.ensure_targets(width, height, target.format);

        let vec4 = |c: [f32; 3], a: f32| Vec4::new(c[0], c[1], c[2], a);
        self.queue.write_buffer(
            &self.frame_buffer,
            0,
            bytemuck::bytes_of(&FrameUniforms {
                view: scene.view,
                proj: scene.proj,
                lights: scene.lights,
                cam_pos: Vec4::new(scene.eye.x, scene.eye.y, scene.eye.z, 0.0),
                floor_color: vec4(scene.room.floor, 1.0),
                ceil_color: vec4(scene.room.ceil, 1.0),
                shadow: Vec4::new(
                    SHADOW_NEAR,
                    SHADOW_FAR,
                    if scene.shadows { SHADOW_STRENGTH } else { 0.0 },
                    SHADOW_BIAS,
                ),
            }),
        );

        // Stage every instance up front, so the buffer is written once a frame.
        self.stage_instances(scene);
        if self.instances.len() > self.instance_capacity {
            self.instance_capacity = self.instances.len().next_power_of_two();
            self.instance_buffer = make_vertex_buffer(
                &self.device,
                "instances",
                self.instance_capacity * std::mem::size_of::<Instance>(),
            );
        }
        self.queue
            .write_buffer(&self.instance_buffer, 0, bytemuck::cast_slice(&self.instances));

        // Line segments come in pairs; an odd tail would be half a segment.
        let line_count = scene.lines.len() & !1;
        if line_count > self.line_capacity {
            self.line_capacity = line_count.next_power_of_two();
            self.line_buffer = make_vertex_buffer(
                &self.device,
                "trail lines",
                self.line_capacity * std::mem::size_of::<LineVertex>(),
            );
        }
        if line_count > 0 {
            self.queue.write_buffer(
                &self.line_buffer,
                0,
                bytemuck::cast_slice(&scene.lines[..line_count]),
            );
        }

        if scene.shadows {
            self.render_shadows(encoder, scene);
        }

        // The views have to outlive the render pass, so they are locals here
        // rather than temporaries inside the descriptor.
        let msaa_view = self.targets[targets]
            .msaa
            .create_view(&wgpu::TextureViewDescriptor::default());
        let depth_view = self.targets[targets]
            .depth
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("scene"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &msaa_view,
                resolve_target: Some(target.view),
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: scene.background[0] as f64,
                        g: scene.background[1] as f64,
                        b: scene.background[2] as f64,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            // wgpu 30 renamed the pipeline's `multiview` flag to `multiview_mask`
            // and gave the render pass one of its own; `None` on both means an
            // ordinary single-view pass. The colour attachment's `depth_slice`
            // belongs to the same multiview feature.
            multiview_mask: None,
        });

        pass.set_bind_group(0, &self.frame_bg, &[]);
        pass.set_vertex_buffer(2, self.instance_buffer.slice(..));

        // 1. the room: front-face culling leaves the interior visible from
        // outside, and fully enclosed from inside. It is instance 0.
        pass.set_pipeline(&self.pipelines.room);
        pass.set_bind_group(1, &self.shadow_bg, &[]);
        self.fill(&mut pass, scene.room.mesh, 0..1);

        // 2. ground discs, then the ceiling fixture
        if !self.staged.discs.is_empty() {
            pass.set_pipeline(&self.pipelines.flat);
            self.fill(&mut pass, scene.quad_mesh, self.staged.discs.clone());
        }

        // 3. the primitives, one draw per shape
        pass.set_pipeline(&self.pipelines.object);
        for (mesh, range) in &self.staged.bodies {
            self.fill(&mut pass, *mesh, range.clone());
        }

        // 4. lines: room outline, wireframe shells, impact rings
        pass.set_pipeline(&self.pipelines.line);
        self.outline(
            &mut pass,
            scene.room.mesh,
            self.staged.outline..self.staged.outline + 1,
        );
        for (mesh, range) in &self.staged.wires {
            self.outline(&mut pass, *mesh, range.clone());
        }
        if !self.staged.rings.is_empty() {
            self.outline(&mut pass, scene.ring_mesh, self.staged.rings.clone());
        }

        // 5. trails and sparks
        if line_count > 0 {
            pass.set_pipeline(&self.pipelines.trail);
            pass.set_vertex_buffer(0, self.line_buffer.slice(..));
            pass.draw(0..line_count as u32, 0..1);
        }
    }

    /// Six depth-only passes from the key light, one per cube face. Only the
    /// bodies are drawn: the room is what the shadows fall on.
    fn render_shadows(&self, encoder: &mut wgpu::CommandEncoder, scene: &Scene<'_>) {
        let light = scene.lights[0].pos.truncate();
        for (i, m) in shadow_face_matrices(light).iter().enumerate() {
            self.queue.write_buffer(
                &self.face_buffer,
                i as u64 * FACE_SLOT_BYTES,
                bytemuck::bytes_of(m),
            );
        }
        for (i, face) in self.shadow_faces.iter().enumerate() {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("shadow"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: face,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipelines.shadow);
            pass.set_bind_group(0, &self.face_bg, &[(i as u64 * FACE_SLOT_BYTES) as u32]);
            pass.set_vertex_buffer(2, self.instance_buffer.slice(..));
            for (mesh, range) in &self.staged.bodies {
                self.fill(&mut pass, *mesh, range.clone());
            }
        }
    }

    /// Render `scene` at `width` x `height` with no window involved and read the
    /// picture back as tightly packed RGBA rows, top row first.
    ///
    /// This is what a recorded clip is made of. It submits its own command
    /// buffer and waits for the copy, so it costs a few milliseconds a call.
    pub fn render_to_pixels(
        &mut self,
        scene: &Scene<'_>,
        width: u32,
        height: u32,
    ) -> Result<Vec<u8>, String> {
        let format = self.surface_format;
        let extent = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let color = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("capture"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = color.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("capture"),
            });
        self.render(
            &mut encoder,
            &RenderTarget {
                view: &view,
                format,
                width,
                height,
            },
            scene,
        );

        // Rows have to start on a 256-byte boundary, so the staging row is wider
        // than the image and the loop below un-pads it.
        let unpadded = width * 4;
        let padded = unpadded.next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("capture readback"),
            size: u64::from(padded) * u64::from(height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &color,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(height),
                },
            },
            extent,
        );
        let submission = self.queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        let (sender, reader) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: Some(std::time::Duration::from_secs(10)),
            })
            .map_err(|e| format!("device poll failed: {e:?}"))?;
        reader
            .recv()
            .map_err(|_| "the map callback never ran".to_owned())?
            .map_err(|e| format!("the readback buffer failed to map: {e:?}"))?;

        let data = slice
            .get_mapped_range()
            .map_err(|e| format!("mapped range unavailable: {e:?}"))?;
        let mut pixels = vec![0u8; (unpadded * height) as usize];
        for row in 0..height as usize {
            let from = row * padded as usize;
            let to = row * unpadded as usize;
            pixels[to..to + unpadded as usize]
                .copy_from_slice(&data[from..from + unpadded as usize]);
        }
        drop(data);
        staging.unmap();

        // Byte order follows the format name; the clip wants RGBA.
        if matches!(
            format,
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
        ) {
            for pixel in pixels.chunks_exact_mut(4) {
                pixel.swap(0, 2);
            }
        }
        Ok(pixels)
    }

    fn fill(&self, pass: &mut wgpu::RenderPass<'_>, mesh: usize, instances: Range<u32>) {
        let m = &self.meshes[mesh];
        pass.set_vertex_buffer(0, m.vertex_buffer.slice(..));
        pass.set_vertex_buffer(1, m.color_buffer.slice(..));
        match &m.index_buffer {
            Some((buf, count)) => {
                pass.set_index_buffer(buf.slice(..), wgpu::IndexFormat::Uint16);
                pass.draw_indexed(0..*count, 0, instances);
            }
            // No index buffer means the vertex list *is* the triangle list
            // (the flat-shaded primitives are built that way).
            None => pass.draw(0..m.vertex_count, instances),
        }
    }

    fn outline(&self, pass: &mut wgpu::RenderPass<'_>, mesh: usize, instances: Range<u32>) {
        let m = &self.meshes[mesh];
        pass.set_vertex_buffer(0, m.vertex_buffer.slice(..));
        pass.set_vertex_buffer(1, m.color_buffer.slice(..));
        if let Some((buf, count)) = &m.line_buffer {
            pass.set_index_buffer(buf.slice(..), wgpu::IndexFormat::Uint16);
            pass.draw_indexed(0..*count, 0, instances);
        }
    }

    /// Build this frame's instance list and note where each kind of thing sits
    /// in it. Bodies are grouped by shape, because one shape is one draw.
    fn stage_instances(&mut self, scene: &Scene<'_>) {
        self.instances.clear();
        let mut staged = Staged::default();
        let vec4 = |c: [f32; 3], a: f32| Vec4::new(c[0], c[1], c[2], a);

        // room fill — always instance 0
        self.instances.push(Instance::new(
            Mat4::IDENTITY,
            vec4(scene.room.wall, 1.0),
            Vec4::new(14.0, 0.16, 1.0, scene.room.grid_spacing),
        ));

        // ground discs + ceiling panel: one-sided, so a disc whose plane faces
        // away is skipped rather than hanging in mid-air through a culled wall.
        let start = self.instances.len() as u32;
        for disc in scene.blobs.iter().chain(scene.panels.iter()) {
            let facing = facing_toward(disc.normal, disc.center, scene.eye);
            if facing <= 0.02 {
                continue;
            }
            let fade = (facing * 6.0).min(1.0);
            self.instances.push(Instance::new(
                gt::disc_matrix(disc.center, disc.normal, disc.radius, 0.006),
                vec4(disc.color, 1.0),
                Vec4::new(disc.alpha * fade, disc.inner, 0.0, 0.0),
            ));
        }
        staged.discs = start..self.instances.len() as u32;

        // bodies, grouped by shape: count each shape, then give every shape a
        // contiguous run and drop its bodies into it.
        let shape_count = scene.shapes.len();
        let mut counts = vec![0u32; shape_count];
        for body in scene.bodies {
            counts[body.shape] += 1;
        }
        let body_base = self.instances.len() as u32;
        let mut next = Vec::with_capacity(shape_count);
        let mut cursor = body_base;
        for (shape, &count) in counts.iter().enumerate() {
            next.push(cursor);
            if count > 0 {
                staged
                    .bodies
                    .push((scene.shape_mesh[shape], cursor..cursor + count));
            }
            cursor += count;
        }
        let body_total = (cursor - body_base) as usize;
        self.instances.resize(
            self.instances.len() + body_total,
            Instance::new(Mat4::IDENTITY, Vec4::ONE, Vec4::ZERO),
        );
        for body in scene.bodies {
            let shape = &scene.shapes[body.shape];
            let slot = next[body.shape];
            next[body.shape] += 1;
            self.instances[slot as usize] = Instance::new(
                gt::model_matrix(body.pos, body.rot, body.scale),
                vec4(shape.color, 1.0),
                Vec4::new(shape.gloss, 0.9, 0.0, 1.0),
            );
        }

        // room outline, pulled slightly inward about the room centre so its lines
        // win the equal-depth test against the coplanar walls
        let half = Vec3::from(scene.room.half);
        let centre = Vec3::new(0.0, half.y * 0.5, 0.0);
        staged.outline = self.instances.len() as u32;
        self.instances.push(Instance::new(
            Mat4::from_translation(centre)
                * Mat4::from_scale(Vec3::splat(ROOM_OUTLINE_INWARD))
                * Mat4::from_translation(-centre),
            Vec4::from(scene.room.edge_color),
            Vec4::ZERO,
        ));

        // wireframe shells: the body instances again, a hair larger, in the same
        // per-shape runs
        if scene.wireframe {
            let wire_base = self.instances.len() as u32;
            for i in 0..body_total {
                let mut shell = self.instances[body_base as usize + i];
                let model = Mat4::from_cols_array_2d(&shell.model);
                // Scale about the body's own centre, which is the translation column.
                let centre = model.w_axis.truncate();
                shell.model = (Mat4::from_translation(centre)
                    * Mat4::from_scale(Vec3::splat(WIREFRAME_SHELL))
                    * Mat4::from_translation(-centre)
                    * model)
                    .to_cols_array_2d();
                shell.color = [1.0, 1.0, 1.0, 0.32];
                self.instances.push(shell);
            }
            for (mesh, range) in &staged.bodies {
                let shift = wire_base - body_base;
                staged
                    .wires
                    .push((*mesh, range.start + shift..range.end + shift));
            }
        }
        // the held body: one bright shell, wireframe on or off
        if let Some(body) = scene.outlined.and_then(|i| scene.bodies.get(i)) {
            let at = self.instances.len() as u32;
            self.instances.push(Instance::new(
                gt::model_matrix(body.pos, body.rot, body.scale * (WIREFRAME_SHELL + 0.01)),
                Vec4::new(1.0, 0.95, 0.7, 0.9),
                Vec4::ZERO,
            ));
            staged.wires.push((scene.shape_mesh[body.shape], at..at + 1));
        }

        let start = self.instances.len() as u32;
        for ring in scene.rings.iter() {
            let grown = ring.base_radius * (1.0 + 1.15 * ring.t);
            let to_eye = scene.eye - ring.pos;
            let dist = to_eye.length().max(1e-4);
            let facing = ring.n.dot(to_eye).abs() / dist;
            let apparent = grown / dist;
            if facing < RING_MIN_FACING || apparent < RING_MIN_APPARENT_SIZE {
                continue;
            }
            let size_fade = (apparent / 0.09).min(1.0);
            let alpha = 0.34 * (1.0 - ring.t) * (1.0 - ring.t) * facing * size_fade;
            self.instances.push(Instance::new(
                gt::disc_matrix(ring.pos, ring.n, grown, 0.006),
                Vec4::new(ring.color[0], ring.color[1], ring.color[2], alpha),
                Vec4::ZERO,
            ));
        }
        staged.rings = start..self.instances.len() as u32;

        self.staged = staged;
    }
}

/// How squarely a plane at `center` with normal `n` faces `eye`: the cosine of
/// the angle, positive when the eye is on the normal's side.
fn facing_toward(n: Vec3, center: Vec3, eye: Vec3) -> f32 {
    let d = eye - center;
    let len = d.length();
    if len < 1e-6 {
        0.0
    } else {
        n.dot(d) / len
    }
}

/// Choose the colour target from a surface's advertised formats, preferring a
/// plain unorm (non-sRGB) one — see the note where this is called.
fn pick_surface_format(formats: &[wgpu::TextureFormat]) -> wgpu::TextureFormat {
    const PREFERRED: [wgpu::TextureFormat; 2] = [
        wgpu::TextureFormat::Rgba8Unorm,
        wgpu::TextureFormat::Bgra8Unorm,
    ];
    PREFERRED
        .iter()
        .find_map(|want| formats.iter().copied().find(|have| have == want))
        .or_else(|| formats.iter().copied().find(|f| f.is_srgb()))
        .unwrap_or(formats[0])
}

fn make_vertex_buffer(device: &wgpu::Device, label: &str, bytes: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes as u64,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_odd_index_list_is_padded_to_the_copy_alignment() {
        // What this guards is a validation error at upload time rather than a
        // wrong picture, so the assertion is about byte lengths, not pixels: an odd
        // entry count is 2 mod 4 bytes and wgpu refuses the copy.
        assert_eq!(padded_index_bytes(&[0]).len(), 4, "one index needs 4 bytes");
        let three = padded_index_bytes(&[0, 1, 2]);
        assert_eq!(three.len(), 8, "three indices are 6 bytes, rounded up");
        assert_eq!(&three[6..], &[0, 0], "and the pad is zeroes, not garbage");
        assert_eq!(padded_index_bytes(&[]).len(), 0, "an empty list stays empty");

        // The shape `import` actually produces for an oversized file: an odd
        // 57 801-entry index buffer, which is what the probe used to crash on.
        for count in [1usize, 3, 5, 57_801, 60_000] {
            let idx: Vec<u16> = (0..count as u32).map(|i| i as u16).collect();
            assert_eq!(
                padded_index_bytes(&idx).len() % wgpu::COPY_BUFFER_ALIGNMENT as usize,
                0,
                "{count} indices are not aligned"
            );
        }
    }
}
