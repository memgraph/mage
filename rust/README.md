# Memgraph Mage Rust Query Modules

`rsmgp-sys` stands for Rust Memgraph Procedures "system" library to develop
query modules for Memgraph in Rust.

It's possible to create your Memgraph query module by adding a new project to
the `rust/` folder. The project has to be a standard Rust project with `[lib]`
section specifying dynamic lib as the `crate-type`.

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

Please take a look at the example project.
