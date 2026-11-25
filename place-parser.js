// JavaScript port of the HierGeo deterministic place parser (hiergeo/routing.py).
// Works in the browser (window.HierGeoParser) and in Node (module.exports).
(function (root) {
  'use strict';

  const WORD = '[\\p{L}\\p{N}_]';
  const escapeRe = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const pattern = (token, ignoreCase) =>
    new RegExp(`(?<!${WORD})${escapeRe(token)}(?!${WORD})`, ignoreCase ? 'giu' : 'gu');
  const cmp = (a, b) => (a < b ? -1 : a > b ? 1 : 0);
  const byLengthThenName = (a, b) => (b.length - a.length) || cmp(a.toLowerCase(), b.toLowerCase());

  class PlaceParser {
    constructor(data) {
      this.countries = data.countries; // {country: {city: imageCount}}
      const entries = [];
      const names = new Set([...data.gazetteer, ...Object.keys(data.countries)]);
      names.forEach((name) => entries.push([name, name, true]));
      Object.entries(data.aliases).forEach(([alias, c]) => entries.push([alias, c, true]));
      Object.entries(data.acronyms).forEach(([alias, c]) => entries.push([alias, c, false]));
      entries.sort((a, b) => byLengthThenName(a[0], b[0]));
      this.countryPatterns = entries.map(([surface, canonical, ic]) => [pattern(surface, ic), canonical]);
      this.cityPatterns = {};
    }

    citiesFor(country) {
      if (!this.cityPatterns[country]) {
        const names = Object.keys(this.countries[country] || {}).filter(Boolean).sort(byLengthThenName);
        this.cityPatterns[country] = names.map((name) => [pattern(name, true), name]);
      }
      return this.cityPatterns[country];
    }

    findCountry(query) {
      for (const [re, canonical] of this.countryPatterns) {
        const matches = [...query.matchAll(re)];
        if (matches.length) {
          const last = matches[matches.length - 1];
          return [canonical, [last.index, last.index + last[0].length]];
        }
      }
      return [null, null];
    }

    static endsPlaceName(query, end, span) {
      const rest = query.slice(end);
      const stripped = rest.trimStart();
      if (!stripped || ',;)'.includes(stripped[0])) return true;
      return span !== null && end + (rest.length - stripped.length) === span[0];
    }

    parse(query, useCity = true) {
      const [country, span] = this.findCountry(query);
      if (country === null) return { scope: 'global', country: null, city: null, mentioned: null };
      if (!(country in this.countries)) return { scope: 'global', country: null, city: null, mentioned: country };
      if (useCity) {
        for (const [re, city] of this.citiesFor(country)) {
          for (const m of query.matchAll(re)) {
            const start = m.index;
            const end = start + m[0].length;
            if (span !== null && start < span[1] && span[0] < end) continue;
            if (start > 0 && "-'".includes(query[start - 1])) continue;
            if (!PlaceParser.endsPlaceName(query, end, span)) continue;
            return { scope: 'city', country, city, mentioned: country };
          }
        }
      }
      return { scope: 'country', country, city: null, mentioned: country };
    }

    shardSize(route, total) {
      if (route.scope === 'city') return this.countries[route.country][route.city];
      if (route.scope === 'country') return Object.values(this.countries[route.country]).reduce((a, b) => a + b, 0);
      return total;
    }
  }

  if (typeof module !== 'undefined' && module.exports) module.exports = { PlaceParser };
  else root.HierGeoParser = { PlaceParser };
})(typeof window !== 'undefined' ? window : globalThis);
