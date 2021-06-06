# Memgraph Mage Rust Query Modules

`rsmgp-sys` stands for Rust Memgraph Procedures "system" library to develop
query modules for Memgraph in Rust.

Adding a new Rust Memgraph query module is simple, just add the following to
your `Cargo.toml` project file.

```
[lib]
name = "query_module_name"
crate-type = ["cdylib"]
```

Memgraph Rust query modules API uses
[CStr](https://doc.rust-lang.org/std/ffi/struct.CStr.html) (`&CStr`) becuase
that's the most compatible type between Rust and Memgraph engine. [Rust
String](https://doc.rust-lang.org/std/string/struct.String.html) can validly
contain a null-byte in the middle of the string (0 is a valid Unicode
codepoint). This means that not all Rust strings can actually be translated to
C strings. While interacting with the `rsmgp` API, built-in `CStr` or
[c_str](https://docs.rs/c_str) library should be used because Memgraph query
modules API only provides C strings.
