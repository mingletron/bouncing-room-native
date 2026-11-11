//! The desktop binary: a room full of bouncing primitives, drawn natively.
//!
//! Everything of substance lives in [`bouncing_room_native::app`]; this only
//! hands the exit code back, so a windowing failure reaches the terminal as a
//! message rather than a panic.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    bouncing_room_native::app::run()
}
