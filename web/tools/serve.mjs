// Development server for the game: serves web/ with caching switched off, so
// a normal reload always runs the current code.
//
//   node web/tools/serve.mjs [port]      then open http://localhost:8000/
//
// (python3 -m http.server lets the browser keep old copies of the JavaScript
// modules. After an edit the page can then run a mix of old and new files,
// which fails in confusing ways, e.g. an empty Level list.)
import { createServer } from 'node:http';
import { readFile, stat } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const ROOT = fileURLToPath(new URL('..', import.meta.url));
const PORT = +(process.argv[2] || 8000);
const TYPES = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.mjs': 'text/javascript; charset=utf-8',
  '.json': 'application/json', '.css': 'text/css; charset=utf-8', '.wav': 'audio/wav', '.png': 'image/png', '.txt': 'text/plain; charset=utf-8',
};

createServer(async (req, res) => {
  try {
    const rel = decodeURIComponent(new URL(req.url, 'http://x').pathname);
    let file = path.join(ROOT, path.normalize(rel));
    if (!file.startsWith(ROOT)) { res.writeHead(403).end(); return; } // nothing outside web/
    if ((await stat(file)).isDirectory()) file = path.join(file, 'index.html');
    const body = await readFile(file);
    res.writeHead(200, { 'Content-Type': TYPES[path.extname(file)] || 'application/octet-stream', 'Cache-Control': 'no-store' });
    res.end(body);
  } catch {
    res.writeHead(404, { 'Cache-Control': 'no-store' }).end('not found');
  }
}).listen(PORT, () => console.log(`serving ${ROOT} at http://localhost:${PORT}/ (no caching) — Ctrl+C stops`));
