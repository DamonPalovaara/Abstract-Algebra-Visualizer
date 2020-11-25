#[macro_use]
extern crate glium;
#[allow(unused_imports)]
use glium::{glutin, Surface};

use std::f32::consts::TAU;
use std::f32::consts::PI;

type Display = glium::backend::glutin::Display;
type Mouse = glium::glutin::dpi::PhysicalPosition<f64>;

#[derive(Copy, Clone)]
struct Vertex {
    position:     [f32; 3],
    vertex_color: [f32; 3],
}

implement_vertex!(Vertex, position, vertex_color);
struct Button {
    is_pressed:    bool,
    is_hovered:    bool,
    view_port:     [f32; 2],
    location:      [f32; 2],
    size:          f32,
    base_color:    [f32; 3],
    hover_color:   [f32; 3],
    vertex_buffer: glium::VertexBuffer<Vertex>,
    program:       glium::Program,
}

impl Button {
    fn new(location: [f32; 2], view_port: [f32; 2], display: Display) -> Button {
        let is_pressed  = false;
        let is_hovered  = false;
        let base_color  = [100.0 / 256.0, 0.0         , 176.0 / 256.0];
        let hover_color = [180.0 / 256.0, 82.0 / 256.0, 1.0          ];
        let size = 30.0;
        let mut vertices = Vec::new(); 
        
        //let location = [view_port[0] / 2.0, view_port[1] / 2.0];

        vertices.push( Vertex { position: [(-1.0 * size) + location[0], (-1.0 * size) + location[1], 0.5], vertex_color: [0.0, 0.0, 0.0] });
        vertices.push( Vertex { position: [(-1.0 * size) + location[0], ( 1.0 * size) + location[1], 0.5], vertex_color: [0.0, 0.0, 0.0] });
        vertices.push( Vertex { position: [( 1.0 * size) + location[0], ( 1.0 * size) + location[1], 0.5], vertex_color: [0.0, 0.0, 0.0] });
        vertices.push( Vertex { position: [( 1.0 * size) + location[0], ( 1.0 * size) + location[1], 0.5], vertex_color: [0.0, 0.0, 0.0] });
        vertices.push( Vertex { position: [( 1.0 * size) + location[0], (-1.0 * size) + location[1], 0.5], vertex_color: [0.0, 0.0, 0.0] });
        vertices.push( Vertex { position: [(-1.0 * size) + location[0], (-1.0 * size) + location[1], 0.5], vertex_color: [0.0, 0.0, 0.0] });

        let vertex_buffer = glium::VertexBuffer::dynamic(&display, &vertices).unwrap();
        let vertex_shader_src = r#"
            #version 150
            in vec3 position;
            uniform vec2 view_port;           

            void main() {
                gl_Position = vec4(2.0 * position.x / view_port.x - 1.0, 2.0 * (view_port.y - position.y) / view_port.y - 1.0, 0, 1);
            }
        "#;

        let fragment_shader_src = r#"
            #version 140
            uniform vec3 le_color;
            out vec4 color;
            void main() {
                color = vec4(le_color, 0.1);
            }
        "#;

        let program = glium::Program::from_source(&display, vertex_shader_src, fragment_shader_src, None).unwrap();
    
        Button {
            base_color,
            hover_color,
            is_pressed,
            is_hovered,
            vertex_buffer,
            program,
            view_port,
            size,
            location,
        }
    }

    fn update_mouse(&mut self, mouse: &Mouse) {
        let x = mouse.x as f32;
        let y = mouse.y as f32;
        if self.location[0] - self.size < x && self.location[0] + self.size > x &&
           self.location[1] - self.size < y && self.location[1] + self.size > y 
        {
            self.is_hovered = true;
        }
        else {
            self.is_hovered = false;
        }
    }

    fn draw(&mut self, target: &mut glium::Frame) {
        let indices = glium::index::NoIndices(glium::index::PrimitiveType::TrianglesList);
        let color;
        if self.is_hovered {
            color = self.hover_color;
        }
        else {
            color = self.base_color;
        }

        let uniforms = uniform! {
            view_port: self.view_port,
            le_color: color,
        };
        target.draw(&self.vertex_buffer, &indices, &self.program, &uniforms, &Default::default()).unwrap();
    }
}

struct Polygon {
    side:          usize,
    conjugate:     f32,
    size:          usize,
    frame:         usize,
    current:       [f32; 3],
    start:         [f32; 3],
    end:           [f32; 3],
    perspective:   [[f32; 4]; 4],
    view:          [[f32; 4]; 4],
    vertex_buffer: glium::VertexBuffer<Vertex>,
    program:       glium::Program,
    rotations: Vec<Button>,
    flips:     Vec<Button>,
    reset:     Button,
}

impl Polygon {
    fn new(size: usize, view_port: [f32; 2], display: Display) -> Polygon {      
        let conjugate = 0.0;  
        let perspective = {
            let aspect_ratio = view_port[1] / view_port[0];
            let fov: f32 = PI / 3.0;
            let zfar = 1024.0;
            let znear = 0.1;    
            let f = 1.0 / (fov / 2.0).tan();    
            [
                [f *   aspect_ratio, 0.0,              0.0              , 0.0],
                [         0.0      ,  f ,              0.0              , 0.0],
                [         0.0      , 0.0,      (zfar+znear)/(zfar-znear), 1.0],
                [         0.0      , 0.0, -(2.0*zfar*znear)/(zfar-znear), 0.0],
            ]
        };        
        let view = view_matrix(&[0.0, 0.0, -3.0], &[-0.0, -0.0, 3.0], &[0.0, 1.0, 0.0]);
        let frame = 101;
        let start = [0.0, 0.0, 0.0];
        let end = [0.0, 0.0, 0.0];
        let current = [0.0, 0.0, 0.0];
        let radius = 1.0;

        let mut vertices = Vec::new();        
        for i in 0..size {
            let theta_1 = ( i      as f32 * TAU / size as f32) + PI / 2.0;
            let theta_2 = ((i + 1) as f32 * TAU / size as f32) + PI / 2.0;
            vertices.push( Vertex { position: [theta_1.cos() * radius, theta_1.sin() * radius, 0.0], vertex_color: [0.0, 1.0, 0.1] });
            vertices.push( Vertex { position: [theta_2.cos() * radius, theta_2.sin() * radius, 0.0], vertex_color: [0.0, 1.0, 0.1] });
            vertices.push( Vertex { position: [0.0, 0.0, 0.0]                                      , vertex_color: [0.0, 1.0, 0.1] });
        }    
        for i in 0..size {
            let theta_1 = ( i      as f32 * TAU / size as f32) + PI / 2.0;
            let theta_2 = ((i + 1) as f32 * TAU / size as f32) + PI / 2.0;
            
            vertices.push( Vertex { position: [theta_2.cos() * radius, theta_2.sin() * radius, 0.0], vertex_color: [1.0, 0.0, 0.1] });
            vertices.push( Vertex { position: [theta_1.cos() * radius, theta_1.sin() * radius, 0.0], vertex_color: [1.0, 0.0, 0.1] });
            vertices.push( Vertex { position: [0.0, 0.0, 0.0]                                      , vertex_color: [1.0, 0.0, 0.1] });
        }
        vertices[0].vertex_color = [1.0, 1.0, 0.0];
        vertices[1].vertex_color = [1.0, 1.0, 0.0];
        vertices[2].vertex_color = [1.0, 1.0, 0.0];
        let len = vertices.len();
        vertices[len / 2 + 0].vertex_color = [1.0, 1.0, 0.0];
        vertices[len / 2 + 1].vertex_color = [1.0, 1.0, 0.0];
        vertices[len / 2 + 2].vertex_color = [1.0, 1.0, 0.0];

        let vertex_buffer = glium::VertexBuffer::new(&display, &vertices).unwrap();

        let vertex_shader_src = r#"
            #version 150        
            #define TAU 6.28318530717958647692528676655900577
            
            in vec3 position;
            in vec3 vertex_color;

            out vec4 ex_color;

            uniform vec3 rotation;
            uniform mat4 perspective;
            uniform mat4 view;
            uniform float conjugate;

            float theta_z = -TAU * (rotation.x + conjugate);
            mat4 z_rot = mat4(
                cos(theta_z), -sin(theta_z), 0.0, 0.0,
                sin(theta_z),  cos(theta_z), 0.0, 0.0,
                0.0         ,  0.0         , 1.0, 0.0,
                0.0         ,  0.0         , 0.0, 1.0
            );

            float theta_y = TAU * rotation.y;
            mat4 y_rot = mat4(
                cos(theta_y), 0.0, sin(theta_y), 0.0,
                0.0         , 1.0, 0.0         , 0.0,
                -sin(theta_y), 0.0, cos(theta_y), 0.0,
                0.0         , 0.0, 0.0         , 1.0
            );
            
            float theta_z_2 = -TAU * -conjugate;
            mat4 z_rot_2 = mat4(
                cos(theta_z_2), -sin(theta_z_2), 0.0, 0.0,
                sin(theta_z_2),  cos(theta_z_2), 0.0, 0.0,
                0.0           ,  0.0           , 1.0, 0.0,
                0.0           ,  0.0           , 0.0, 1.0
            );

            void main() {
                gl_Position = perspective * view * z_rot * y_rot * z_rot_2 * vec4(position, 1.0);
                ex_color = vec4(vertex_color, 1.0);
            }
        "#;

        let fragment_shader_src = r#"
            #version 140
            in  vec4 ex_color;
            out vec4 color;
            void main() {
                color = ex_color;
            }
        "#;

        let program = glium::Program::from_source(&display, vertex_shader_src, fragment_shader_src, None).unwrap();

        let mut rotations = Vec::new();
        for i in 0..size {
            rotations.push(Button::new([50.0 + 100.0 * i as f32, 650.0], view_port, display.clone()));
        }

        let mut flips = Vec::new();
        for i in 0..size {
            flips.push(Button::new([50.0 + 100.0 * i as f32, 750.0], view_port, display.clone()));
        }

        let reset = Button::new([50.0, 550.0], view_port, display.clone());
        let side = 0;
    
        Polygon {
            side,
            conjugate,
            rotations,
            flips,
            reset,
            size,
            frame,
            start,
            end,
            current,
            perspective,
            vertex_buffer,
            view,
            program,
        }
    }

    fn update(&mut self) {
        if self.frame <= 100 {
            self.current = self.interpolate(self.start, self.end, self.frame as f32 / 100.0);
            self.frame += 1;
        }
        if self.frame == 101 {
            if self.conjugate != 0.0 {
                self.current[0] += (1.0 / (2.0 * self.size as f32)) * 2.0 * (self.size - self.side) as f32;
                self.conjugate = 0.0;
                self.side = 0;
            }
            self.frame += 1;
        }
    }

    fn mouse_click(&mut self) {
        if self.frame != 102 {
            return;
        }
        if self.reset.is_hovered {
            self.reset();
        }
        for i in 0..self.rotations.len() {
            if self.rotations[i].is_hovered {
                self.rotate(i);
            }
            if self.flips[i].is_hovered {
                self.flip(i);
            }
        }
    }

    fn update_mouse(&mut self, mouse: &Mouse) {
        self.reset.update_mouse(mouse);
        for i in 0..self.rotations.len() {
            self.rotations[i].update_mouse(mouse);
            self.flips[i].update_mouse(mouse);
        }
    } 

    fn reset(&mut self) {
        self.frame = 0;
        self.start = self.current;
        self.end = [0.0, 0.0, 0.0];
        self.conjugate = 0.0;
    }

    fn rotate(&mut self, n: usize) {
        self.frame = 0;
        self.start = self.current;
        self.end = [
            self.start[0] + n as f32 / self.size as f32,
            self.start[1],
            self.start[2],
        ];
    }

    fn flip(&mut self, n: usize) {
        self.frame = 0;
        self.start = self.current;
        self.end = [
            self.start[0],
            self.start[1] + 0.5,
            self.start[2],
        ];
        self.conjugate = -(1.0 / (2.0 * self.size as f32)) * n as f32;
        self.side = n;
    }

    fn interpolate(&self, start: [f32; 3], end: [f32; 3], weight: f32) -> [f32; 3] {
        [
            start[0] + (weight*(end[0] - start[0])),
            start[1] + (weight*(end[1] - start[1])),
            start[2] + (weight*(end[2] - start[2])),
        ]
    }

    fn draw(&mut self, target: &mut glium::Frame) {
        let indices = glium::index::NoIndices(glium::index::PrimitiveType::TrianglesList);
        let uniforms = uniform! {
            rotation: self.current,
            perspective: self.perspective,
            view: self.view,
            conjugate: self.conjugate,
        };
        let params = glium::DrawParameters {
            backface_culling: glium::draw_parameters::BackfaceCullingMode::CullClockwise,
            .. Default::default()
        };
        target.draw(&self.vertex_buffer, &indices, &self.program, &uniforms, &params).unwrap();
        self.reset.draw(target);
        for i in 0..self.rotations.len() {
            self.rotations[i].draw(target);
            self.flips[i].draw(target);
        }

    }
}


fn main() {
    // Initialize the rendering engine
    let (width, height) = (1200, 800);
    let event_loop = glutin::event_loop::EventLoop::new();
    let wb = glutin::window::WindowBuilder::new()
        .with_inner_size(glium::glutin::dpi::LogicalSize::new(width as f32, height as f32))
        .with_title("Abstract Algebra");
    let cb = glutin::ContextBuilder::new();
    let display = glium::Display::new(wb, cb, &event_loop).unwrap();

    // Initialize objects
    let mut polygon = Polygon::new(5, [width as f32, height as f32] , display.clone());

    event_loop.run(move |event, _, control_flow| {
        let next_frame_time = std::time::Instant::now() + std::time::Duration::from_nanos(16_666_667);
        *control_flow = glutin::event_loop::ControlFlow::WaitUntil(next_frame_time);

        match event {
            glutin::event::Event::WindowEvent { event, .. } => match event {
                glutin::event::WindowEvent::CloseRequested => {
                    *control_flow = glutin::event_loop::ControlFlow::Exit;
                    return;
                },
                glutin::event::WindowEvent::CursorMoved{device_id: _, position: pos, modifiers: _} => polygon.update_mouse(&pos),
                glutin::event::WindowEvent::MouseInput{device_id: _, state: state, button: button, modifiers: _} => {
                    match state {
                        glium::glutin::event::ElementState::Pressed => {
                            match button {
                                glium::glutin::event::MouseButton::Left => polygon.mouse_click(),
                                _ => return,
                            }
                        },
                        _ => return,
                    }
                }
                _ => return,
            },
            glutin::event::Event::NewEvents(cause) => match cause {
                glutin::event::StartCause::ResumeTimeReached { .. } => (),
                glutin::event::StartCause::Init => (),
                _ => return,
            },
            _ => return,
        }  
        
        // Clear
        let mut target = display.draw();
        target.clear_color(0.0, 0.0, 0.0, 1.0);

        // Update
        polygon.update();
 
        // Draw
        polygon.draw(&mut target);

        // Finish
        target.finish().unwrap();
        
    });
}

fn view_matrix(position: &[f32; 3], direction: &[f32; 3], up: &[f32; 3]) -> [[f32; 4]; 4] {
    let f = {
        let f = direction;
        let len = f[0] * f[0] + f[1] * f[1] + f[2] * f[2];
        let len = len.sqrt();
        [f[0] / len, f[1] / len, f[2] / len]
    };

    let s = [up[1] * f[2] - up[2] * f[1],
             up[2] * f[0] - up[0] * f[2],
             up[0] * f[1] - up[1] * f[0]];

    let s_norm = {
        let len = s[0] * s[0] + s[1] * s[1] + s[2] * s[2];
        let len = len.sqrt();
        [s[0] / len, s[1] / len, s[2] / len]
    };

    let u = [f[1] * s_norm[2] - f[2] * s_norm[1],
             f[2] * s_norm[0] - f[0] * s_norm[2],
             f[0] * s_norm[1] - f[1] * s_norm[0]];

    let p = [-position[0] * s_norm[0] - position[1] * s_norm[1] - position[2] * s_norm[2],
             -position[0] * u[0] - position[1] * u[1] - position[2] * u[2],
             -position[0] * f[0] - position[1] * f[1] - position[2] * f[2]];

    [
        [s_norm[0], u[0], f[0], 0.0],
        [s_norm[1], u[1], f[1], 0.0],
        [s_norm[2], u[2], f[2], 0.0],
        [p[0], p[1], p[2], 1.0],
    ]
}