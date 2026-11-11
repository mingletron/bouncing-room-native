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
//! * **Per-draw uniforms become one ring buffer with dynamic offsets**, because
//!   bind groups are not as cheap as `glUniform*` calls were.
//!
//! The room's "cull the faces pointing at you" trick needs care too: winding
//! conventions differ between GL and WebGPU, and which way they differ is
//! precisely the kind of thing that reads plausibly and is wrong. So
//! `front_face` is a parameter rather than a literal, and `src/bin/probe.rs`
//! settles it by checking that bodies — which live *inside* the room — are
//! actually visible. On Metal with wgpu 30 the answer is `Ccw`, the same
//! handedness GL used, not the flip the contract predicted.

use crate::geom::{Mesh, ShapeDef};
use crate::gpu_types::{self as gt, FrameUniforms, Light, ObjectUniforms, UNIFORM_SLOT_BYTES};
use crate::sim::Body;

use glam::{Mat4, Vec3, Vec4};

/// Multisampling for the 3D pass. The UI draws afterwards into the resolved
/// target, so egui itself is not multisampled (it does not need to be).
pub const SAMPLE_COUNT: u32 = 4;

/// Slots in the per-object uniform ring: bodies + discs + panels + rings for one
/// frame, comfortably more than the UI allows. The buffer grows if exceeded.
const OBJECT_SLOTS_START: usize = 512;

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

const LIT_SRC: &str = include_str!("../shaders/lit.wgsl");
const LINE_SRC: &str = include_str!("../shaders/line.wgsl");
const FLAT_SRC: &str = include_str!("../shaders/flat.wgsl");

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// Sentinel meaning "this item produced no uniform slot, so skip its draw".
const SKIP: u32 = u32::MAX;

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
}

struct GpuMesh {
    vertex_buffer: wgpu::Buffer,
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
    // The frame bind group layout is not stored: the frame uniform buffer is a
    // fixed size, so its bind group is built once and never rebuilt. The object
    // ring can grow, so that layout has to stay (`bgl_object`).
    bgl_object: wgpu::BindGroupLayout,
    frame_buffer: wgpu::Buffer,
    frame_bg: wgpu::BindGroup,
    object_buffer: wgpu::Buffer,
    object_bg: wgpu::BindGroup,
    object_slots: usize,
    object_staging: Vec<u8>,
    next_slot: usize,
    /// Byte offset into the ring buffer for each draw of the frame, in draw
    /// order; `SKIP` marks a draw to leave out.
    slot_offsets: Vec<u32>,
    meshes: Vec<GpuMesh>,
    msaa_texture: Option<wgpu::Texture>,
    depth_texture: Option<wgpu::Texture>,
    extent: (u32, u32, wgpu::TextureFormat),
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

impl Gfx {
    /// Create a device and all four pipelines against an existing `instance`.
    ///
    /// The instance belongs to the caller because wgpu 30 binds it to a window
    /// system display handle (`InstanceDescriptor::display`, supplied by winit),
    /// and that handle has to be in place before a surface can be created — so
    /// the windowed path builds the instance and surface first and hands the
    /// surface here to pick a format-capable adapter, while the headless probe
    /// passes `None`.
    ///
    /// There are four pipelines rather than three because culling is pipeline
    /// state in WebGPU: the room needs front-face culling and the primitives need
    /// none, even though both run the same shaders.
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

        let mut limits = wgpu::Limits::default();
        // wgpu 30 widened the buffer-size limits to u64.
        limits.max_uniform_buffer_binding_size =
            (OBJECT_SLOTS_START * UNIFORM_SLOT_BYTES) as u64;

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

        let lit = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("lit"),
            source: wgpu::ShaderSource::Wgsl(LIT_SRC.into()),
        });
        let line_src = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("line"),
            source: wgpu::ShaderSource::Wgsl(LINE_SRC.into()),
        });
        let flat = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("flat"),
            source: wgpu::ShaderSource::Wgsl(FLAT_SRC.into()),
        });

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

        // One bind group for the whole frame: the slot is chosen per draw with a
        // dynamic offset, which is far cheaper than a bind group per object.
        let bgl_object = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("object"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: wgpu::BufferSize::new(
                        std::mem::size_of::<ObjectUniforms>() as u64,
                    ),
                },
                count: None,
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("scene"),
            // wgpu 30: a pipeline layout entry is `Option<&BindGroupLayout>`
            // (the `None` slot means "set the layout at pipeline creation"), and
            // push constants are now `immediate_size` — a byte count, not a list
            // of stages. We use no immediates.
            bind_group_layouts: &[Some(&bgl_frame), Some(&bgl_object)],
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

        // One vertex layout for every mesh and every pipeline. `buffers` is a
        // list of `Option<VertexBufferLayout>` in wgpu 30 (a `None` slot means
        // "this slot is unused"), and the attribute array has to outlive the
        // layout, so both are locals here rather than temporaries in the
        // descriptors. All four pipelines read position at location 0 and the
        // normal at location 1, even where a shader ignores the normal.
        let vertex_attrs = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3];
        let vertex_layouts = [Some(wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<crate::geom::Vertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &vertex_attrs,
        })];

        let room = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("room"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &lit,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &vertex_layouts,
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: options.room_front_face,
                cull_mode: Some(wgpu::Face::Front),
                ..Default::default()
            },
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
        });

        let object = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("object"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &lit,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &vertex_layouts,
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                // Primitives are double sided: their normals are analytic and
                // always face out, and the shader flips any normal that faces
                // away from the camera, so culling would only punch holes.
                cull_mode: None,
                ..Default::default()
            },
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
        });

        let flat_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("flat"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &flat,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                // The flat and line shaders read only position; the normal sits
                // in the same buffer at location 1 and is simply not consumed,
                // which is what keeps one vertex layout for every mesh.
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

        let line_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("line"),
            layout: Some(&pipeline_layout),
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
            depth_stencil: Some(wgpu::DepthStencilState {
                depth_write_enabled: Some(false),
                // Lines sit coplanar with, or just inside, the surfaces they
                // describe, so equal depth has to win.
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                ..depth()
            }),
            multisample,
            fragment: Some(wgpu::FragmentState {
                module: &line_src,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[blended_target(surface_format)],
            }),
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

        let object_slots = OBJECT_SLOTS_START;
        let object_buffer = make_object_buffer(&device, object_slots);
        let object_bg = make_object_bind_group(&device, &bgl_object, &object_buffer);

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
            },
            bgl_object,
            frame_buffer,
            frame_bg,
            object_buffer,
            object_bg,
            object_slots,
            object_staging: Vec::new(),
            next_slot: 0,
            slot_offsets: Vec::new(),
            meshes: Vec::new(),
            msaa_texture: None,
            depth_texture: None,
            extent: (0, 0, surface_format),
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

        let index_buffer = mesh.indices.as_ref().map(|idx| {
            let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("indices"),
                // A zero-size buffer is not allowed, hence the floor of one index.
                size: (idx.len() * std::mem::size_of::<u16>()).max(2) as u64,
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.queue.write_buffer(&buf, 0, bytemuck::cast_slice(idx));
            (buf, idx.len() as u32)
        });

        let line_buffer = (!mesh.edges.is_empty()).then(|| {
            let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("edges"),
                size: (mesh.edges.len() * std::mem::size_of::<u16>()) as u64,
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.queue
                .write_buffer(&buf, 0, bytemuck::cast_slice(&mesh.edges));
            (buf, mesh.edges.len() as u32)
        });

        let handle = self.meshes.len();
        self.meshes.push(GpuMesh {
            vertex_buffer,
            index_buffer,
            line_buffer,
            vertex_count: mesh.vertex_count() as u32,
        });
        handle
    }

    fn ensure_targets(&mut self, width: u32, height: u32, format: wgpu::TextureFormat) {
        if self.extent == (width, height, format) {
            return;
        }
        self.extent = (width, height, format);
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        self.msaa_texture = Some(self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("msaa colour"),
            size,
            mip_level_count: 1,
            sample_count: SAMPLE_COUNT,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        }));
        self.depth_texture = Some(self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("depth"),
            size,
            mip_level_count: 1,
            sample_count: SAMPLE_COUNT,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        }));
    }

    /// Record the whole 3D pass into `encoder`, resolving MSAA onto `target`.
    ///
    /// The caller then opens a second pass on the same target with load op `Load`
    /// for the UI, which draws without multisampling.
    pub fn render(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        target: &RenderTarget<'_>,
        scene: &Scene<'_>,
    ) {
        let width = target.width.max(1);
        let height = target.height.max(1);
        self.ensure_targets(width, height, target.format);

        self.queue.write_buffer(
            &self.frame_buffer,
            0,
            bytemuck::bytes_of(&FrameUniforms {
                view: scene.view,
                proj: scene.proj,
                lights: scene.lights,
                cam_pos: Vec4::new(scene.eye.x, scene.eye.y, scene.eye.z, 0.0),
            }),
        );

        // Stage every per-object uniform up front, in exactly the order the draw
        // loop below consumes them, so the ring buffer is written once per frame.
        self.stage_object_uniforms(scene);
        self.queue
            .write_buffer(&self.object_buffer, 0, &self.object_staging);

        // The views have to outlive the render pass, so they are locals here
        // rather than temporaries inside the descriptor.
        let msaa_view = self
            .msaa_texture
            .as_ref()
            .expect("msaa target")
            .create_view(&wgpu::TextureViewDescriptor::default());
        let depth_view = self
            .depth_texture
            .as_ref()
            .expect("depth target")
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
        self.next_slot = 0;

        // 1. the room: front-face culling leaves the interior visible from
        // outside, and fully enclosed from inside.
        pass.set_pipeline(&self.pipelines.room);
        self.bind_and_fill(&mut pass, scene.room.mesh);

        // 2. ground discs, then the ceiling fixture
        pass.set_pipeline(&self.pipelines.flat);
        for _ in scene.blobs.iter().chain(scene.panels.iter()) {
            let offset = self.take_slot();
            if offset == SKIP {
                continue;
            }
            pass.set_bind_group(1, &self.object_bg, &[offset]);
            self.fill(&mut pass, scene.quad_mesh);
        }

        // 3. the primitives
        pass.set_pipeline(&self.pipelines.object);
        for body in scene.bodies {
            let offset = self.take_slot();
            pass.set_bind_group(1, &self.object_bg, &[offset]);
            self.fill(&mut pass, scene.shape_mesh[body.shape]);
        }

        // 4. lines: room outline, wireframe shells, impact rings
        pass.set_pipeline(&self.pipelines.line);
        self.bind_and_outline(&mut pass, scene.room.mesh);

        if scene.wireframe {
            for body in scene.bodies {
                let offset = self.take_slot();
                pass.set_bind_group(1, &self.object_bg, &[offset]);
                self.outline(&mut pass, scene.shape_mesh[body.shape]);
            }
        }

        let ring_handle = scene.ring_mesh;
        for _ in scene.rings.iter() {
            let offset = self.take_slot();
            if offset == SKIP {
                continue;
            }
            pass.set_bind_group(1, &self.object_bg, &[offset]);
            self.outline(&mut pass, ring_handle);
        }
    }

    /// The room uses the same uniform slot as the first staged entry, and the
    /// outline reuses it because the outline's model transform is baked into the
    /// geometry it draws — see `ROOM_OUTLINE_INWARD`.
    fn bind_and_fill(&mut self, pass: &mut wgpu::RenderPass<'_>, mesh: usize) {
        let offset = self.take_slot();
        pass.set_bind_group(1, &self.object_bg, &[offset]);
        self.fill(pass, mesh);
    }

    fn bind_and_outline(&mut self, pass: &mut wgpu::RenderPass<'_>, mesh: usize) {
        let offset = self.take_slot();
        pass.set_bind_group(1, &self.object_bg, &[offset]);
        self.outline(pass, mesh);
    }

    fn fill(&self, pass: &mut wgpu::RenderPass<'_>, mesh: usize) {
        let m = &self.meshes[mesh];
        pass.set_vertex_buffer(0, m.vertex_buffer.slice(..));
        match &m.index_buffer {
            Some((buf, count)) => {
                pass.set_index_buffer(buf.slice(..), wgpu::IndexFormat::Uint16);
                pass.draw_indexed(0..*count, 0, 0..1);
            }
            // No index buffer means the vertex list *is* the triangle list
            // (the flat-shaded primitives are built that way).
            None => pass.draw(0..m.vertex_count, 0..1),
        }
    }

    fn outline(&self, pass: &mut wgpu::RenderPass<'_>, mesh: usize) {
        let m = &self.meshes[mesh];
        pass.set_vertex_buffer(0, m.vertex_buffer.slice(..));
        if let Some((buf, count)) = &m.line_buffer {
            pass.set_index_buffer(buf.slice(..), wgpu::IndexFormat::Uint16);
            pass.draw_indexed(0..*count, 0, 0..1);
        }
    }

    fn take_slot(&mut self) -> u32 {
        let v = self.slot_offsets.get(self.next_slot).copied().unwrap_or(0);
        self.next_slot += 1;
        v
    }

    /// Build the ring-buffer contents in draw order and return the byte offset of
    /// each draw, or `SKIP` for an item the camera cannot see.
    fn stage_object_uniforms(&mut self, scene: &Scene<'_>) {
        self.object_staging.clear();
        self.slot_offsets.clear();

        let vec4 = |c: [f32; 3], a: f32| Vec4::new(c[0], c[1], c[2], a);

        // room fill
        self.push_object(&ObjectUniforms {
            model: Mat4::IDENTITY,
            color: vec4(scene.room.wall, 1.0),
            params: Vec4::new(14.0, 0.16, 1.0, scene.room.grid_spacing),
            floor_color: vec4(scene.room.floor, 1.0),
            ceil_color: vec4(scene.room.ceil, 1.0),
        });

        // ground discs + ceiling panel: one-sided, so a disc whose plane faces
        // away is skipped rather than hanging in mid-air through a culled wall.
        for disc in scene.blobs.iter().chain(scene.panels.iter()) {
            let facing = facing_toward(disc.normal, disc.center, scene.eye);
            if facing <= 0.02 {
                self.slot_offsets.push(SKIP);
                continue;
            }
            let fade = (facing * 6.0).min(1.0);
            self.push_object(&ObjectUniforms {
                model: gt::disc_matrix(disc.center, disc.normal, disc.radius, 0.006),
                color: vec4(disc.color, 1.0),
                params: Vec4::new(disc.alpha * fade, disc.inner, 0.0, 0.0),
                ..Default::default()
            });
        }

        // bodies
        for body in scene.bodies {
            let shape = &scene.shapes[body.shape];
            self.push_object(&ObjectUniforms {
                model: gt::model_matrix(body.pos, body.rot, body.scale),
                color: vec4(shape.color, 1.0),
                params: Vec4::new(shape.gloss, 0.9, 0.0, 1.0),
                ..Default::default()
            });
        }

        // room outline, pulled slightly inward about the room centre so its lines
        // win the equal-depth test against the coplanar walls
        let half = Vec3::from(scene.room.half);
        let centre = Vec3::new(0.0, half.y * 0.5, 0.0);
        self.push_object(&ObjectUniforms {
            model: Mat4::from_translation(centre)
                * Mat4::from_scale(Vec3::splat(ROOM_OUTLINE_INWARD))
                * Mat4::from_translation(-centre),
            color: Vec4::from(scene.room.edge_color),
            ..Default::default()
        });

        if scene.wireframe {
            for body in scene.bodies {
                self.push_object(&ObjectUniforms {
                    model: gt::model_matrix(body.pos, body.rot, body.scale * WIREFRAME_SHELL),
                    color: Vec4::new(1.0, 1.0, 1.0, 0.32),
                    ..Default::default()
                });
            }
        }

        for ring in scene.rings.iter() {
            let grown = ring.base_radius * (1.0 + 1.15 * ring.t);
            let to_eye = scene.eye - ring.pos;
            let dist = to_eye.length().max(1e-4);
            let facing = ring.n.dot(to_eye).abs() / dist;
            let apparent = grown / dist;
            if facing < RING_MIN_FACING || apparent < RING_MIN_APPARENT_SIZE {
                self.slot_offsets.push(SKIP);
                continue;
            }
            let size_fade = (apparent / 0.09).min(1.0);
            let alpha = 0.34 * (1.0 - ring.t) * (1.0 - ring.t) * facing * size_fade;
            self.push_object(&ObjectUniforms {
                model: gt::disc_matrix(ring.pos, ring.n, grown, 0.006),
                color: Vec4::new(ring.color[0], ring.color[1], ring.color[2], alpha),
                ..Default::default()
            });
        }

        // The ring buffer and its bind group have to be rebuilt together.
        let needed = (self.object_staging.len() / UNIFORM_SLOT_BYTES).max(1);
        if needed > self.object_slots {
            self.object_slots = needed.next_power_of_two();
            self.object_buffer = make_object_buffer(&self.device, self.object_slots);
            self.object_bg =
                make_object_bind_group(&self.device, &self.bgl_object, &self.object_buffer);
        }
    }

    fn push_object(&mut self, uniforms: &ObjectUniforms) {
        let offset = self.object_staging.len() as u32;
        self.slot_offsets.push(offset);
        self.object_staging
            .extend_from_slice(bytemuck::bytes_of(uniforms));
        self.object_staging
            .resize(self.object_staging.len().next_multiple_of(UNIFORM_SLOT_BYTES), 0);
    }
}

/// Byte offsets of this frame's object uniforms, in draw order.
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

fn make_object_buffer(device: &wgpu::Device, slots: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("object uniforms"),
        size: (slots * UNIFORM_SLOT_BYTES) as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn make_object_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    buffer: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("object ring"),
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer,
                offset: 0,
                size: wgpu::BufferSize::new(std::mem::size_of::<ObjectUniforms>() as u64),
            }),
        }],
    })
}
