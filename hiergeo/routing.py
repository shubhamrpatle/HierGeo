"""Geo-indexed shards and the deterministic place parser.

The reference system in the paper resolves explicit place names with a simple,
auditable parser instead of a learned router:

1. Find the longest country name (or unambiguous alias) in the query, matching on
   word boundaries against a gazetteer of 197 benchmark countries plus every
   corpus country.
2. If that country has no images in the corpus, abstain (global search).
3. Otherwise, look for a city of that country in the rest of the query (longest
   name first, word boundaries, not overlapping the country mention). A match
   selects the city shard; otherwise the country shard is used.

Queries that name no supported country (continents, styles, sub-national places
without their country, or no place at all) are searched globally.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Case-insensitive aliases. Entries that can refer to more than one place
# ("america", "korea", "congo" as an alias, ...) are deliberately excluded.
COUNTRY_ALIASES: Dict[str, str] = {
    "united states of america": "United States",
    "u.s.a.": "United States",
    "u.s.": "United States",
    "united kingdom of great britain and northern ireland": "United Kingdom",
    "great britain": "United Kingdom",
    "britain": "United Kingdom",
    "u.k.": "United Kingdom",
    "u.a.e.": "United Arab Emirates",
    "russian federation": "Russia",
    "republic of korea": "South Korea",
    "czechia": "Czech Republic",
    "türkiye": "Turkey",
    "turkiye": "Turkey",
    "viet nam": "Vietnam",
}

# Upper-case acronyms are matched case-sensitively so that, e.g., the pronoun
# "us" is never read as the United States.
COUNTRY_ACRONYMS: Dict[str, str] = {
    "USA": "United States",
    "US": "United States",
    "UK": "United Kingdom",
    "UAE": "United Arab Emirates",
    "KSA": "Saudi Arabia",
}


def _pattern(token: str, ignore_case: bool) -> re.Pattern:
    # Look-arounds instead of \b so that tokens ending in "." (e.g. "u.s.") work.
    flags = re.IGNORECASE if ignore_case else 0
    return re.compile(r"(?<![\w])" + re.escape(token) + r"(?![\w])", flags)


def default_country_gazetteer() -> List[str]:
    text = resources.files("hiergeo.resources").joinpath("countries.txt").read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip()]


@dataclass
class GeoIndex:
    """Country -> city -> image ids, in corpus row order.

    Image ids are 0-based row indices of the corpus metadata CSV and index the
    rows of the image-embedding matrix.
    """

    cities: Dict[str, Dict[str, List[int]]] = field(default_factory=dict)
    num_images: int = 0

    @classmethod
    def from_metadata_csv(cls, path: str | Path) -> "GeoIndex":
        index = cls()
        with open(path, newline="", encoding="utf-8") as handle:
            for image_id, row in enumerate(csv.DictReader(handle)):
                index.num_images += 1
                country = (row.get("country") or "").strip()
                city = (row.get("city") or "").strip()
                if not country:
                    continue
                index.cities.setdefault(country, {}).setdefault(city, []).append(image_id)
        return index

    @property
    def countries(self) -> List[str]:
        return list(self.cities)

    def country_ids(self, country: str) -> List[int]:
        ids: List[int] = []
        for city_ids in self.cities.get(country, {}).values():
            ids.extend(city_ids)
        return ids

    def city_ids(self, country: str, city: str) -> List[int]:
        return list(self.cities.get(country, {}).get(city, []))

    def num_city_pairs(self) -> int:
        return sum(len([c for c in cities if c]) for cities in self.cities.values())


@dataclass
class Route:
    """Result of place resolution for one query."""

    scope: str  # "city", "country" or "global"
    country: Optional[str] = None
    city: Optional[str] = None
    mentioned_country: Optional[str] = None  # country named in the query, even if absent from the corpus


class PlaceParser:
    """Deterministic longest-match country/city parser (see module docstring)."""

    def __init__(self, geo_index: GeoIndex, gazetteer: Optional[Iterable[str]] = None) -> None:
        self.geo = geo_index
        names = set(gazetteer if gazetteer is not None else default_country_gazetteer())
        names.update(geo_index.countries)
        entries: List[Tuple[str, str, bool]] = [(name, name, True) for name in names]
        entries += [(alias, country, True) for alias, country in COUNTRY_ALIASES.items()]
        entries += [(alias, country, False) for alias, country in COUNTRY_ACRONYMS.items()]
        # Longest surface form first, so "Papua New Guinea" wins over "Guinea".
        entries.sort(key=lambda item: (-len(item[0]), item[0].lower()))
        self._countries = [(_pattern(surface, ic), canonical) for surface, canonical, ic in entries]
        self._city_patterns: Dict[str, List[Tuple[re.Pattern, str]]] = {}

    def _cities_for(self, country: str) -> List[Tuple[re.Pattern, str]]:
        if country not in self._city_patterns:
            names = sorted((c for c in self.geo.cities.get(country, {}) if c), key=lambda c: (-len(c), c.lower()))
            self._city_patterns[country] = [(_pattern(name, True), name) for name in names]
        return self._city_patterns[country]

    def find_country(self, query: str) -> Tuple[Optional[str], Optional[Tuple[int, int]]]:
        for pattern, canonical in self._countries:
            matches = list(pattern.finditer(query))
            if matches:
                last = matches[-1]  # "<city>, <country>" puts the country last
                return canonical, last.span()
        return None, None

    @staticmethod
    def _ends_place_name(query: str, end: int, country_span: Optional[Tuple[int, int]]) -> bool:
        """A city mention must be followed by a comma, the country, or the end of the query.

        This keeps a known city from matching the first word of a longer, unknown
        place name ("Porto" in "Porto Alegre, Brazil"). When in doubt the parser
        falls back to the coarser country shard rather than a wrong city shard.
        """
        rest = query[end:]
        stripped = rest.lstrip()
        if not stripped or stripped[0] in ",;)":
            return True
        return country_span is not None and end + (len(rest) - len(stripped)) == country_span[0]

    def parse(self, query: str, use_city: bool = True) -> Route:
        country, span = self.find_country(query)
        if country is None:
            return Route(scope="global")
        if country not in self.geo.cities:
            return Route(scope="global", mentioned_country=country)
        if use_city:
            for pattern, city in self._cities_for(country):
                for match in pattern.finditer(query):
                    start, end = match.span()
                    if span is not None and start < span[1] and span[0] < end:
                        continue  # the city name is the country mention itself
                    if start > 0 and query[start - 1] in "-'":
                        continue  # e.g. "Hastings" inside "Napier-Hastings"
                    if not self._ends_place_name(query, end, span):
                        continue  # e.g. "Porto" inside "Porto Alegre"
                    return Route(scope="city", country=country, city=city, mentioned_country=country)
        return Route(scope="country", country=country, mentioned_country=country)

    def candidates(self, route: Route) -> Optional[List[int]]:
        """Image ids for a route, or ``None`` for global search."""
        if route.scope == "city":
            return self.geo.city_ids(route.country, route.city)
        if route.scope == "country":
            return self.geo.country_ids(route.country)
        return None


def split_query(query: str) -> Tuple[str, Optional[str]]:
    """Split a templated query "<category> in <place>" into its two parts."""
    head, sep, tail = query.rpartition(" in ")
    return (head.strip(), tail.strip()) if sep else (query.strip(), None)


def route_many(parser: PlaceParser, queries: Sequence[str], use_city: bool = True) -> List[Route]:
    return [parser.parse(q, use_city=use_city) for q in queries]
