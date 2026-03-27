#!/usr/bin/env python3
"""Simple dev server for the financial dashboard."""
import http.server
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        super().end_headers()

print("Serving financials dashboard at http://localhost:3001")
http.server.HTTPServer(('', 3001), Handler).serve_forever()
