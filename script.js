document.documentElement.classList.add('motion-ready');

const body = document.body;
const nav = document.querySelector('[data-nav]');

const setActiveSideLink = (hash) => {
  nav?.querySelectorAll('a[href^="#"]').forEach((link) => {
    link.classList.toggle('is-active', link.getAttribute('href') === hash);
  });
};

nav?.querySelectorAll('a[href^="#"]').forEach((link) => {
  link.addEventListener('click', () => setActiveSideLink(link.getAttribute('href')));
});

const revealItems = document.querySelectorAll('.reveal');
if ('IntersectionObserver' in window && !window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
  const revealObserver = new IntersectionObserver((entries, observer) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      entry.target.classList.add('is-visible');
      observer.unobserve(entry.target);
    });
  }, { threshold: 0.11, rootMargin: '0px 0px -45px' });

  revealItems.forEach((item) => revealObserver.observe(item));
} else {
  revealItems.forEach((item) => item.classList.add('is-visible'));
}

const navSections = [...document.querySelectorAll('main section[id]')]
  .filter((section) => document.querySelector(`.side-nav a[href="#${section.id}"]`));

let navSyncQueued = false;

const syncSideNavigation = () => {
  navSyncQueued = false;
  if (!navSections.length) return;

  const atPageEnd = window.scrollY + window.innerHeight >= document.documentElement.scrollHeight - 4;
  if (atPageEnd) {
    setActiveSideLink(`#${navSections[navSections.length - 1].id}`);
    return;
  }

  // Use a point just above the viewport's middle so the active item follows
  // the section the reader is currently viewing in either scroll direction.
  const readingLine = window.scrollY + window.innerHeight * 0.38;
  let activeSection = navSections[0];

  navSections.forEach((section) => {
    if (section.offsetTop <= readingLine) activeSection = section;
  });

  setActiveSideLink(`#${activeSection.id}`);
};

const requestSideNavigationSync = () => {
  if (navSyncQueued) return;
  navSyncQueued = true;
  window.requestAnimationFrame(syncSideNavigation);
};

window.addEventListener('scroll', requestSideNavigationSync, { passive: true });
window.addEventListener('resize', requestSideNavigationSync, { passive: true });
window.addEventListener('load', syncSideNavigation);
syncSideNavigation();


const lightbox = document.querySelector('[data-lightbox-dialog]');
const lightboxImage = lightbox?.querySelector('img');
const lightboxCaption = lightbox?.querySelector('p');
const lightboxClose = document.querySelector('[data-lightbox-close]');

document.querySelectorAll('[data-lightbox]').forEach((trigger) => {
  trigger.addEventListener('click', () => {
    if (!lightbox || !lightboxImage || !lightboxCaption) return;
    lightboxImage.src = trigger.dataset.lightbox;
    lightboxImage.alt = trigger.querySelector('img')?.alt || 'Expanded research figure';
    lightboxCaption.textContent = trigger.dataset.caption || '';
    body.classList.add('is-lightbox-open');
    lightbox.showModal();
  });
});

const closeLightbox = () => {
  if (!lightbox?.open) return;
  lightbox.close();
  body.classList.remove('is-lightbox-open');
};

lightboxClose?.addEventListener('click', closeLightbox);
lightbox?.addEventListener('click', (event) => {
  if (event.target === lightbox) closeLightbox();
});
lightbox?.addEventListener('close', () => body.classList.remove('is-lightbox-open'));

const copyButton = document.querySelector('[data-copy-bib]');
const bibtex = document.querySelector('[data-bibtex]');

copyButton?.addEventListener('click', async () => {
  if (!bibtex) return;
  const label = copyButton.querySelector('span');
  try {
    await navigator.clipboard.writeText(bibtex.textContent.trim());
    if (label) label.textContent = 'Copied';
  } catch {
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(bibtex);
    selection.removeAllRanges();
    selection.addRange(range);
    if (label) label.textContent = 'Selected';
  }

  window.setTimeout(() => {
    if (label) label.textContent = 'Copy';
  }, 1800);
});

// ---------------------------------------------------------------- place-parser demo
const demo = document.querySelector('[data-parser-demo]');
if (demo && window.HierGeoParser) {
  const form = demo.querySelector('[data-parser-form]');
  const input = demo.querySelector('[data-parser-input]');
  const result = demo.querySelector('[data-parser-result]');
  const fmt = new Intl.NumberFormat('en-US');
  let parser = null;
  let total = 0;

  const escapeHtml = (s) => s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

  const render = () => {
    if (!parser) return;
    const query = input.value.trim();
    if (!query) { result.textContent = 'Type a query that names a place.'; return; }
    const route = parser.parse(query);
    const size = parser.shardSize(route, total);
    const pct = (100 * size / total);
    let label, detail;
    if (route.scope === 'city') {
      label = 'City shard';
      detail = `${escapeHtml(route.city)}, ${escapeHtml(route.country)}`;
    } else if (route.scope === 'country') {
      label = 'Country shard';
      detail = escapeHtml(route.country);
    } else {
      label = 'Global search';
      detail = route.mentioned
        ? `${escapeHtml(route.mentioned)} is named but has no patches in HierGeo-Data`
        : 'no supported country is named, so the parser abstains';
    }
    result.innerHTML = `
      <div class="route-badge route-${route.scope}">${label}</div>
      <div class="route-detail">${detail}</div>
      <div class="route-count"><strong>${fmt.format(size)}</strong> of ${fmt.format(total)} patches searched
        <span class="route-bar"><span style="width:${Math.max(pct, 0.4).toFixed(2)}%"></span></span></div>`;
  };

  fetch('assets/data/places.json')
    .then((r) => r.json())
    .then((data) => {
      parser = new window.HierGeoParser.PlaceParser(data);
      total = data.num_images;
      render();
    })
    .catch(() => { result.textContent = 'Could not load the gazetteer.'; });

  form.addEventListener('submit', (event) => { event.preventDefault(); render(); });
  input.addEventListener('input', render);
  demo.querySelectorAll('[data-example]').forEach((button) => {
    button.addEventListener('click', () => { input.value = button.textContent; render(); });
  });
}
