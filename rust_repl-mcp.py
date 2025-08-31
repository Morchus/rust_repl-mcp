# Copyright 2025 [Your Name/Organization]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "mcp>=1.0.0",
# ]
# ///

"""
A Model-Context-Protocol (MCP) server that exposes the
EVCXR Rust REPL to large language models.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
from typing import Any, Dict, List, Optional

import mcp.types as types
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio

# --------------------------------------------------------------------------- #
# Logging Configuration
# --------------------------------------------------------------------------- #

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
_handler = logging.StreamHandler(sys.stderr)
_handler.setFormatter(
    logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
)
logging.basicConfig(level=LOG_LEVEL, handlers=[_handler])
logger = logging.getLogger("rust_repl-MCP")

# --------------------------------------------------------------------------- #
# Constants & Configuration
# --------------------------------------------------------------------------- #

EVCXR_BINARY = os.getenv("EVCXR_BINARY", "evcxr")
READ_TIMEOUT = int(os.getenv("READ_TIMEOUT", "30"))
STARTUP_TIMEOUT = int(os.getenv("STARTUP_TIMEOUT", "60"))
SHUTDOWN_TIMEOUT = int(os.getenv("SHUTDOWN_TIMEOUT", "5"))

def _strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences for cleaner LLM output."""

    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", text)

# --------------------------------------------------------------------------- #
# REPL Wrapper
# --------------------------------------------------------------------------- #

class EvcxrRepl:
    """
    A wrapper for the evcxr REPL subprocess.
    Manages process lifetime, I/O, etc.
    """

    def __init__(self) -> None:
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._lock = asyncio.Lock()
        self._banner: Optional[str] = None
        self._is_starting = False

    async def start(self) -> str:
        """
        Starts the evcxr process, waits for it to be ready, and returns
        the initial banner. This is an explicit, idempotent, and asynchronous action.
        """

        async with self._lock:
            if self._proc and self._proc.returncode is None:
                logger.info("REPL already running.")
                return self._banner or "REPL is already running."
            
            if self._is_starting:
                raise RuntimeError("REPL startup is already in progress.")

            self._is_starting = True
            try:
                logger.info(f"Starting evcxr REPL subprocess using '{EVCXR_BINARY}'...")
                self._proc = await asyncio.create_subprocess_exec(
                    EVCXR_BINARY,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

                # Wait for the process to be ready by detecting the welcome message
                self._banner = await self._read_startup_banner(timeout=STARTUP_TIMEOUT)
                logger.info("evcxr REPL started successfully.")
                logger.debug("REPL banner captured: %s", self._banner)
                return self._banner
            except FileNotFoundError:
                logger.critical(f"'{EVCXR_BINARY}' not found. Is evcxr installed and in your PATH?")
                raise
            except (IOError, asyncio.TimeoutError) as e:
                logger.critical(f"Failed to start evcxr REPL: {e}")
                stderr_output = await self._read_stream(self._proc.stderr) if self._proc else ""
                logger.error(f"EVCXR stderr: {stderr_output}")
                await self.close()
                raise IOError(f"Could not start EVCXR. Stderr: {stderr_output}") from e
            finally:
                self._is_starting = False

    async def _read_stream(self, stream: Optional[asyncio.StreamReader]) -> str:
        """Reads the entire content of a stream without blocking."""

        if not stream:
            return ""
        try:
            data = await asyncio.wait_for(stream.read(4096), timeout=0.1)
            return data.decode('utf-8', errors='ignore')
        except asyncio.TimeoutError:
            return ""

    async def _read_startup_banner(self, timeout: int = READ_TIMEOUT) -> str:
        """
        Reads the startup banner from evcxr. In evcxr 0.21.1, it prints:
        "Welcome to evcxr. For help, type :help\n"
        and then waits for input (no additional prompt).
        """

        if not self._proc or not self._proc.stdout:
            raise IOError("REPL process is not running or stdout is not available.")

        output_buffer = ""
        welcome_detected = False
        
        # Read character by character until we see the welcome message
        while not welcome_detected:
            try:
                char_bytes = await asyncio.wait_for(
                    self._proc.stdout.read(1), timeout=timeout
                )
                if not char_bytes:  # EOF
                    error_message = f"EVCXR process closed unexpectedly during startup. Output so far: '{output_buffer}'"
                    stderr_output = await self._read_stream(self._proc.stderr)
                    logger.error(f"{error_message}. Stderr: {stderr_output}")
                    raise IOError(f"{error_message}. Stderr: {stderr_output}")

                output_buffer += char_bytes.decode('utf-8', errors='ignore')

                # Check if we've seen the complete welcome message
                if "Welcome to evcxr. For help, type :help" in output_buffer:
                    welcome_detected = True
                    # Continue reading a bit more to get any trailing newlines
                    try:
                        # Try to read a few more characters to get complete output
                        for _ in range(10):  # Read up to 10 more characters
                            more_bytes = await asyncio.wait_for(self._proc.stdout.read(1), timeout=0.1)
                            if not more_bytes:
                                break
                            output_buffer += more_bytes.decode('utf-8', errors='ignore')
                    except asyncio.TimeoutError:
                        # Timeout is expected here since evcxr is waiting for input
                        pass
                    
                    break

            except asyncio.TimeoutError:
                error_message = f"Timeout waiting for REPL startup banner after {timeout}s."
                stderr_output = await self._read_stream(self._proc.stderr)
                logger.error(f"{error_message}. Process may be stuck. Output so far: '{output_buffer}'. Stderr: {stderr_output}")
                raise IOError(f"Timeout waiting for EVCXR output. Stderr: {stderr_output}")
        
        return _strip_ansi(output_buffer.strip())

    async def _read_response(self, timeout: int = READ_TIMEOUT) -> str:
        """
        Reads a complete response from evcxr. This method waits for the execution to complete
        and captures all output.
        """

        if not self._proc or not self._proc.stdout:
            raise IOError("REPL process is not running or stdout is not available.")

        output_buffer = ""
        start_time = asyncio.get_event_loop().time()
        seen_content = False
        
        while True:
            # Check if we've exceeded timeout
            if asyncio.get_event_loop().time() - start_time > timeout:
                error_message = f"Timeout waiting for REPL response after {timeout}s."
                stderr_output = await self._read_stream(self._proc.stderr)
                logger.error(f"{error_message}. Output so far: '{output_buffer}'. Stderr: {stderr_output}")
                raise IOError(f"Timeout waiting for EVCXR output. Stderr: {stderr_output}")
            
            try:
                # Try to read available data with a short timeout
                try:
                    data = await asyncio.wait_for(self._proc.stdout.read(4096), timeout=0.5)
                    if not data:  # EOF
                        error_message = f"EVCXR process closed unexpectedly. Output so far: '{output_buffer}'"
                        stderr_output = await self._read_stream(self._proc.stderr)
                        logger.error(f"{error_message}. Stderr: {stderr_output}")
                        raise IOError(f"{error_message}. Stderr: {stderr_output}")
                    
                    new_data = data.decode('utf-8', errors='ignore')
                    output_buffer += new_data
                    
                    # Mark that we've seen content
                    if new_data.strip():
                        seen_content = True
                        
                except asyncio.TimeoutError:
                    # No data available right now
                    # If we've seen content, check if execution might be complete
                    if seen_content:
                        # Try to read stderr to see if there are any errors
                        stderr_data = await self._read_stream(self._proc.stderr)
                        if stderr_data.strip():
                            # There's error output, include it
                            logger.debug("Found stderr output: %s", stderr_data)
                            return _strip_ansi((output_buffer + "\n" + stderr_data).strip())
                        
                        # If we've seen content and there's no new data, assume execution is complete
                        break
                    
            except Exception as e:
                error_message = f"Error reading from EVCXR: {e}"
                stderr_output = await self._read_stream(self._proc.stderr)
                logger.error(f"{error_message}. Output so far: '{output_buffer}'. Stderr: {stderr_output}")
                raise IOError(f"{error_message}. Stderr: {stderr_output}")

        return _strip_ansi(output_buffer.strip())

    async def execute(self, code: str) -> str:
        """Send `code` to the REPL and return the result."""

        async with self._lock:
            if not self.is_alive():
                logger.warning("REPL was not alive. Attempting to restart...")
                await self.start()

            assert self._proc and self._proc.stdin

            # Send code with proper termination
            code_to_send = (code.rstrip() + "\n").encode('utf-8')
            logger.debug("Sending code block (%d chars)", len(code_to_send))
            
            self._proc.stdin.write(code_to_send)
            await self._proc.stdin.drain()

            result = await self._read_response()
            logger.debug("REPL returned (%d chars): %.100s...", len(result), result)
            return result

    async def close(self) -> None:
        """Terminate the underlying process gracefully."""

        async with self._lock:
            if self._proc is None:
                return
            
            logger.info("Shutting down evcxr REPL...")
            if self._proc.returncode is None:  # Process is still running
                try:
                    if self._proc.stdin:
                        self._proc.stdin.close()
                    self._proc.terminate()
                    try:
                        await asyncio.wait_for(self._proc.wait(), timeout=SHUTDOWN_TIMEOUT)
                        logger.info("evcxr terminated gracefully.")
                    except asyncio.TimeoutError:
                        logger.warning("evcxr did not terminate gracefully, killing.")
                        self._proc.kill()
                        await self._proc.wait()  # Wait for kill to complete
                except Exception as exc:
                    logger.error("Error while terminating evcxr: %s", exc)
            else:
                logger.info("evcxr already terminated.")

            self._proc = None
            self._banner = None

    def is_alive(self) -> bool:
        """Check if the evcxr process is currently running."""

        return self._proc is not None and self._proc.returncode is None

# --------------------------------------------------------------------------- #
# MCP Server
# --------------------------------------------------------------------------- #

REPL = EvcxrRepl()
server = Server("rust_repl-MCP", "1.0.0")


@server.list_tools()
async def list_tools() -> List[types.Tool]:
    logger.debug("Listing tools")
    return [
        types.Tool(
            name="reset_repl",
            description=(
                "Starts or resets the persistent Rust REPL session. "
                "This clears all previously defined variables, functions, "
                "and imports. Call this to get a clean state."
            ),
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        types.Tool(
            name="eval",
            description=(
                "Execute Rust code in the persistent REPL session. "
                "State (variables, functions, etc.) is preserved between calls."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The Rust source code to execute.",
                    }
                },
                "required": ["code"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(
    name: str, arguments: Dict[str, Any]
) -> List[types.TextContent]:
    logger.info("Tool call: %s(%r)", name, arguments)
    try:
        if name == "reset_repl":
            await REPL.close()
            banner = await REPL.start()
            logger.debug("reset_repl returned: %s", banner)
            return [types.TextContent(type="text", text=banner or "REPL is ready.")]

        elif name == "eval":
            code: str = arguments.get("code", "")
            if not code.strip():
                raise ValueError("Cannot execute empty code.")
            logger.debug("Persistent eval: %r", code)
            result = await REPL.execute(code)
            logger.debug("Persistent eval result: %s", result)
            return [types.TextContent(type="text", text=result)]

        else:
            logger.error("Unknown tool: %s", name)
            raise ValueError(f"Unknown tool: {name}")

    except (IOError, RuntimeError, ValueError) as e:
        logger.error("Error during tool call '%s': %s", name, e, exc_info=True)
        # Return a user-friendly error message to the MCP client
        return [types.TextContent(type="text", text=f"Error: {e}")]


# --------------------------------------------------------------------------- #
# Entry Point
# --------------------------------------------------------------------------- #

async def main() -> None:
    """The main entry point for the server."""
    
    logger.info("Initializing rust_repl-MCP server...")
    
    # Proactively start the REPL
    try:
        await REPL.start()
    except Exception as e:
        logger.critical(f"Could not start the server due to REPL failure: {e}")
        return  # Exit if REPL fails to start

    logger.info("Starting MCP I/O loop...")
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name=server.name,
                server_version=server.version,
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={}
                ),
            ),
        )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server interrupted by user.")
    except Exception as e:
        logger.error("Unhandled exception in main: %s", e, exc_info=True)
    finally:
        # Only try to close REPL if event loop is still running
        try:
            if REPL.is_alive():
                logger.info("Shutting down REPL from main.")
                # Use a new event loop if the old one is closed
                try:
                    asyncio.run(REPL.close())
                except RuntimeError:
                    # Event loop is closed, create a new one
                    new_loop = asyncio.new_event_loop()
                    new_loop.run_until_complete(REPL.close())
                    new_loop.close()
            logger.info("Server stopped.")
        except Exception as e:
            logger.error("Error during cleanup: %s", e, exc_info=True)