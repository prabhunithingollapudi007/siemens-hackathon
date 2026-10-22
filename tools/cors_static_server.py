"""Static file server with a CORS header so the organiser physics page (localhost:8000) can fetch scenario files.

Usage: python cors_static_server.py PORT DIRECTORY
"""
import http.server
import sys


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "http://localhost:8000")
        super().end_headers()


port, directory = int(sys.argv[1]), sys.argv[2]
http.server.ThreadingHTTPServer(("127.0.0.1", port), lambda *a, **k: Handler(*a, directory=directory, **k)).serve_forever()
