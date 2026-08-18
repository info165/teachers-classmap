// sw.js — place this in the /public folder
// Network-first: always gets latest code, no stale cache issues

const CACHE_NAME = 'classmap-v__BUILD_TIMESTAMP__';

// Only cache static assets that don't change between builds
const CORE_ASSETS = [
  '/manifest.json',
  '/images/icon-192.png',
  '/images/icon-512.png'
];

// INSTALL
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => cache.addAll(CORE_ASSETS).catch(() => {})) // fail silently if icons missing
      .then(() => self.skipWaiting())
  );
});

// ACTIVATE — delete all old caches
self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys()
      .then(keys => Promise.all(keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

// FETCH — network first, cache fallback
self.addEventListener('fetch', event => {
  if (event.request.method !== 'GET') return;

  const url = new URL(event.request.url);

  // Don't intercept external requests (Firebase, APIs, CDNs, fonts)
  if (url.origin !== self.location.origin) return;

  // NEVER cache HTML or navigation requests — always fetch fresh from network
  if (event.request.mode === 'navigate' || url.pathname === '/' || url.pathname.endsWith('.html')) {
    event.respondWith(
      fetch(event.request).catch(() => caches.match('/'))
    );
    return;
  }

  // For JS/CSS/images — network first, cache as offline fallback
  event.respondWith(
    fetch(event.request)
      .then(networkResponse => {
        // Cache successful responses for offline fallback
        if (networkResponse && networkResponse.status === 200) {
          const clone = networkResponse.clone();
          caches.open(CACHE_NAME).then(cache => cache.put(event.request, clone));
        }
        return networkResponse;
      })
      .catch(() => {
        // Offline fallback
        return caches.match(event.request)
          .then(cached => cached || caches.match('/'));
      })
  );
});

// Handle skip waiting message from app
self.addEventListener('message', event => {
  if (event.data?.type === 'SKIP_WAITING') self.skipWaiting();
});
