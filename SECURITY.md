# Security

## Asset scripts are code

A Cubist asset script is a Python program, and building it runs it.

`cubist build`, `cubist eval`, and the MCP server's `build_asset` tool run scripts as your user,
in a child process with a timeout. That includes code that an agent passes to `build_asset` as
`code`. **Nothing stops a script from reading or writing your files or using the network.**
The design calls for that isolation, but it isn't built yet.

So:

- Only build scripts you trust. Treat a project from someone else like any other code you would
  run.
- The MCP server lets any client connected to it run code. Connect it only to agents you trust,
  over stdio as [docs/mcp.md](docs/mcp.md) describes.

Reading `.vox` files, lint, rendering and export only parse data; they don't run scripts.

## Reporting a vulnerability

Report it privately through
[GitHub's vulnerability reporting](https://github.com/Kalcode/cubist-voxel/security/advisories/new),
not in a public discussion or issue. Say what is affected, how to reproduce it, and what an
attacker could do. The maintainer aims to reply within a week.

Things worth reporting include:

- a `.vox`, `style.toml` or manifest file that crashes Cubist, hangs it or exhausts memory;
- a tool that writes outside its project directory when it shouldn't;
- anything that runs code without a build being asked for.

Scripts doing what they're written to do, as described above, aren't a vulnerability. A way to
isolate them is welcome as a [discussion](https://github.com/Kalcode/cubist-voxel/discussions).

## Supported versions

Cubist is before 1.0. Fixes land on `main`. There are no backports to older versions yet.
