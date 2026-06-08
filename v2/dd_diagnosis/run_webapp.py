#!/usr/bin/env python3
"""DD Diagnosis 웹 서버 CLI 엔트리.

프로젝트 루트(`dd_diagnosis/`)에서 실행하는 것을 전제로,
`yolo_cam`과 `backend` 패키지를 `sys.path`에 넣은 뒤 Flask 앱을 기동합니다.
"""

from __future__ import annotations

import argparse
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the DD Diagnosis Flask web application.",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Bind address (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Flask listen port (default: backend/config.py flask_port, usually 5555)",
    )
    parser.add_argument(
        "--fiftyone-port",
        type=int,
        default=None,
        help="FiftyOne helper port (default: from config)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable Flask debug mode",
    )
    args = parser.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, root)
    sys.path.insert(0, os.path.join(root, "backend"))

    from config import config

    flask_port = args.port if args.port is not None else config.flask_port
    fiftyone_port = (
        args.fiftyone_port if args.fiftyone_port is not None else config.fiftyone_port
    )
    config.set_fiftyone_port(fiftyone_port)

    import app as webapp

    print(f"Starting Flask on http://{args.host}:{flask_port}")
    print(f"FiftyOne helper port: {fiftyone_port}")
    print(f"Debug: {args.debug}")

    webapp.init_app(webapp.app)
    webapp.socketio.run(
        webapp.app,
        host=args.host,
        port=flask_port,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
