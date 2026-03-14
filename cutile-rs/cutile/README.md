# cuTile Rust
The main user-facing crate of this repository.
This includes the core DSL, a collection of basic kernels written using the DSL,
the cuTile Rust host-side Tensor type, 
and various traits and functions for working with cuTile Rust tensors using cuda-async.

# Tests
- Run a specific test and see its output via `cargo test --test span_source_location -- --no-capture`.