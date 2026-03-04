"""Small helper entrypoint for launching the API server."""

import argparse
import uvicorn

from rag_assistant.config import AppConfig


def build_parser(config: AppConfig | None = None) -> argparse.ArgumentParser:
    parser_config = config or AppConfig.from_env()
    parser = argparse.ArgumentParser(description="Run the RAG Knowledge Assistant API")
    parser.add_argument("--host", default=parser_config.host)
    parser.add_argument("--port", type=int, default=parser_config.port)
    parser.add_argument("--reload", action="store_true", default=parser_config.reload)
    parser.add_argument(
        "--log-level",
        default=parser_config.log_level,
        choices=["critical", "error", "warning", "info", "debug", "notset"],
        help="Override log level.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    uvicorn.run(
        "rag_assistant.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level.lower(),
    )


if __name__ == "__main__":
    main()
