# Rust REPL MCP Server

A Model Context Protocol (MCP) Server that exposes tools for Large Language Models (LLMs) to use the evcxr_repl (https://github.com/evcxr/evcxr/blob/main/evcxr_repl/) and run Rust code.

Large Language Models struggle with Rust programming because of the complexity and they cannot execute code to verify their work. You get:

- Code examples that may not compile with current Rust versions
- Incorrect syntax or deprecated APIs
- No way to test mathematical calculations or algorithm implementations
- Generic answers without real-time validation

The Rust REPL MCP Server provides LLMs with direct access to a persistent Rust execution environment through the evcxr REPL. This enables real-time code execution, testing, computation, and prototyping with accurate, up-to-date results.

## How It Works

The MCP server maintains a persistent evcxr Rust REPL session where LLMs can:

1. Execute Rust code snippets and see immediate results
2. Define variables and functions that persist across executions  
3. Import crates and test library functionality
4. Validate algorithm implementations

Perfect for prototyping, learning Rust, and ensuring code accuracy.

## Tools

The Rust REPL MCP Server provides the following tools:

- **eval**: Execute Rust code in the persistent REPL session. Variables, functions, and imports remain available for subsequent executions.
- **reset_repl**: Clear the REPL state and start fresh. Removes all previously defined variables, functions, and imports.

## Installation

### Using Docker (Recommended)

The easiest way to use the Rust REPL MCP Server is with Docker. No local Rust installation required.

#### Prerequisites
- Docker installed on your system
- MCP-compatible client (Claude Desktop, Gemini CLI, VS Code, Cursor, etc.)

#### MCP Client Configuration (Docker)

Add this to your MCP client configuration:

```json
{
  "mcpServers": {
    "rust-repl": {
      "command": "docker",
      "args": [
        "run",
        "-i",
        "--rm",
        "morchus/rust_repl-mcp:latest"
      ]
    }
  }
}
```

### Manual Installation

You can run the server locally with some additional pre-reqs.

#### Prerequisites
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- Rust toolchain with evcxr installed

#### Install evcxr REPL

```bash
cargo install evcxr_repl
```

#### Clone the Repository

```bash
git clone https://github.com/yourusername/rust_repl-mcp.git
```

#### MCP Client Configuration (Manual)

```json
{
  "mcpServers": {
    "rust-repl": {
      "command": "uv",
      "args": ["run", "--with=mcp", "/path/to/rust_repl-mcp/rust_repl-mcp.py"]
    }
  }
}
```

## Usage Example

Once configured, your LLM can execute Rust code directly:


**Data structure exploration:**
```
Create a HashMap in Rust, insert some key-value pairs, and demonstrate different iteration methods
```

The REPL maintains state, so you can build complex programs step by step:

```
First, define a struct for a Point with x and y coordinates
Now implement a method to calculate distance from origin
Create a few Point instances and test the distance calculation
```

## Troubleshooting

### Docker Issues

**Server exits immediately:**
Make sure to include the `-i` flag in your Docker args. The server needs interactive mode to communicate via stdin/stdout.

**Permission denied:**
Ensure Docker is running and your user has permission to run Docker containers.

### Manual Installation Issues

**evcxr not found:**
Ensure the Rust toolchain is installed and evcxr is in your PATH:
```bash
cargo install evcxr_repl
which evcxr
```

**Python dependencies:**
Ensure you have Python 3.11+ and uv installed:
```bash
python --version
uv --version
```

**Module not found:**
Try installing MCP dependency explicitly:
```bash
uv add mcp
```

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

This software is distributed under the terms of the Apache License (Version 2.0)

See LICENSE for details.