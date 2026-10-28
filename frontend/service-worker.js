// service-worker.js
// ════════════════════════════════════════════════════════════════════════
// Minimal service worker for the HEMM Cognitive Core PWA. Caches the
// static app shell (HTML/manifest/icon) for instant offline loading —
// deliberately does NOT try to cache or intercept the WebSocket
// connection to server.py (WebSockets aren't cacheable resources, and
// the app already has a real graceful fallback to simulated data when
// no backend is reachable, so the service worker doesn't need to
// duplicate that logic).
//
// Cache-first for the app shell, network fallback for anything else
// (e.g. the Three.js CDN import — deliberately left to the network so
// updates to that pinned version aren't silently stuck behind a stale
// cache; the shell files are what actually need to be instant/offline).
// ════════════════════════════════════════════════════════════════════════

const CACHE_NAME = 'hemm-core-shell-v1';
const SHELL_FILES = [
  './aura-m1-app.html',
  './manifest.json',
  './icon.svg',
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) => cache.addAll(SHELL_FILES))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((names) =>
      Promise.all(names.filter((n) => n !== CACHE_NAME).map((n) => caches.delete(n)))
    ).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);
  const isShellFile = SHELL_FILES.some((f) => url.pathname.endsWith(f.replace('./', '/')));

  if (!isShellFile) return; // let the browser handle everything else normally

  event.respondWith(
    caches.match(event.request).then((cached) => {
      if (cached) return cached;
      return fetch(event.request).then((response) => {
        const clone = response.clone();
        caches.open(CACHE_NAME).then((cache) => cache.put(event.request, clone));
        return response;
      });
    }).catch(() => caches.match('./aura-m1-app.html'))
  );
});
