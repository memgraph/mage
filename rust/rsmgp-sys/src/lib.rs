// Copyright (c) 2016-2021 Memgraph Ltd. [https://memgraph.com]
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#[cfg(test)]
extern crate mockall;

extern crate mockall_double;

mod testing;

/// All edge related.
pub mod edge;
/// All list related.
pub mod list;
/// All map related.
pub mod map;
/// Abstraction to interact with Memgraph.
pub mod memgraph;
/// Auto-generated bindings (don't use directly, except top-level pointer data types).
pub mod mgp;
/// All path related.
pub mod path;
/// All property related.
pub mod property;
/// Simplifies returning results to Memgraph and then to the client.
pub mod result;
/// Mostly macro definitions.
pub mod rsmgp;
/// All value related.
pub mod value;
/// All vertex related.
pub mod vertex;
