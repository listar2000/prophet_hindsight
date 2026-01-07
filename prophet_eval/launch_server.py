from sglang.test.doc_patch import launch_server_cmd
from sglang.utils import wait_for_server
import argparse
"""
Launch a SGLang server.

Example usage:
-> python launch_server.py --model-path qwen/qwen2.5-0.5b-instruct
"""

def launch_server(model_path: str, tp: int = 2, dp: int = 1, host: str = "0.0.0.0", log_level: str = "warning", reasoning_parser: str | None = None):
    command = f"python3 -m sglang.launch_server --model-path {model_path} --tp {tp} --dp {dp} --host {host} --log-level {log_level}"
    if reasoning_parser is not None:
        command += f" --reasoning-parser {reasoning_parser}"

    server_process, port = launch_server_cmd(command)

    wait_for_server(f"http://localhost:{port}")
    print(f"Server started on http://localhost:{port}")

    return server_process, port


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch a server for a model")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to launch the server on")
    parser.add_argument("--log-level", type=str, default="warning", help="Log level to use")
    parser.add_argument("--tp", type=int, default=2, help="Number of tensor parallel GPUs")
    parser.add_argument("--dp", type=int, default=1, help="Number of data parallel GPUs")
    parser.add_argument("--reasoning-parser", type=str, default=None, help="Reasoning parser to use")
    args = parser.parse_args()
    process, _ = launch_server(args.model_path, args.tp, args.dp, args.host, args.log_level, args.reasoning_parser)
    print(f"Server process: {str(process.pid)}")