import pytest

from hiergeo.routing import GeoIndex, PlaceParser, split_query


@pytest.fixture()
def parser():
    geo = GeoIndex()
    rows = [
        ("Angola", "Luanda"), ("Angola", "Luanda"), ("Angola", "Lobito"),
        ("Brazil", "Porto"), ("Brazil", "Rio de Janeiro"),
        ("Papua New Guinea", "Port Moresby"), ("Singapore", "Singapore"),
        ("United States", "Atlanta"), ("Kenya", "Nairobi"), ("New Zealand", "Hastings"),
    ]
    for image_id, (country, city) in enumerate(rows):
        geo.cities.setdefault(country, {}).setdefault(city, []).append(image_id)
        geo.num_images += 1
    gazetteer = ["Angola", "Brazil", "Guinea", "Papua New Guinea", "Singapore", "United States", "Kenya",
                 "New Zealand", "Niger", "Nigeria", "Equatorial Guinea", "Côte d'Ivoire"]
    return PlaceParser(geo, gazetteer)


def test_city_route(parser):
    r = parser.parse("Residential in Luanda, Angola")
    assert (r.scope, r.country, r.city) == ("city", "Angola", "Luanda")
    assert parser.candidates(r) == [0, 1]


def test_country_route_without_city(parser):
    r = parser.parse("Industrial in Angola")
    assert (r.scope, r.country, r.city) == ("country", "Angola", None)
    assert parser.candidates(r) == [0, 1, 2]


def test_city_routing_can_be_disabled(parser):
    r = parser.parse("Residential in Luanda, Angola", use_city=False)
    assert (r.scope, r.city) == ("country", None)


def test_longest_country_wins(parser):
    assert parser.parse("Port in Port Moresby, Papua New Guinea").country == "Papua New Guinea"


def test_absent_country_abstains_instead_of_matching_a_substring(parser):
    # "Equatorial Guinea" is in the gazetteer but has no images; it must not be read as "Guinea"
    r = parser.parse("Forest in Equatorial Guinea")
    assert r.scope == "global" and r.mentioned_country == "Equatorial Guinea"
    assert parser.parse("Water in Niger").scope == "global"


def test_city_equal_to_country_name_is_not_a_city(parser):
    assert parser.parse("Residential in Singapore").scope == "country"
    r = parser.parse("Residential in Singapore, Singapore")
    assert (r.scope, r.city) == ("city", "Singapore")


def test_known_city_inside_longer_unknown_name_is_ignored(parser):
    assert parser.parse("Urban in Porto Alegre, Brazil").scope == "country"
    assert parser.parse("Urban in Napier-Hastings, New Zealand").scope == "country"
    assert parser.parse("Urban in Porto, Brazil").city == "Porto"


def test_aliases(parser):
    assert parser.parse("farms in the USA").country == "United States"
    assert parser.parse("farms in the U.S.").country == "United States"
    assert parser.parse("show us farms").scope == "global"  # lower-case "us" is a pronoun


@pytest.mark.parametrize(
    "query",
    [
        "Find an Asian-style garden with a pool and lawn",
        "Find industrial ports in Asia",
        "Show airports in Africa",
        "Retrieve dense residential areas in Europe",
        "Find solar farms in South America",
        "Show coastal settlements in North America",
        "Find stadiums in Oceania",
        "Find a Mediterranean-style resort near water",
        "Show European-style town squares",
        "Find a Middle Eastern garden with a pool",
        "Retrieve Latin American colonial plazas",
        "Show American-style suburban blocks",
        "Find airports with red rooftops",
        "Show industrial zones near a river",
        "Retrieve a park with a circular pond",
        "Find solar farms in Texas",
        "Show dense residential areas in Paris",
        "Find ports near Dubai",
        "Retrieve stadiums in California",
    ],
)
def test_unsupported_place_references_abstain(parser, query):
    # the 19 out-of-scope probe queries of App. C
    assert parser.parse(query).scope == "global"


@pytest.mark.parametrize(
    "query,country",
    [
        ("Find airports with red rooftops in Kenya", "Kenya"),
        ("Show industrial zones in Brazil", "Brazil"),
        ("Retrieve parks in New Zealand", "New Zealand"),
        ("Find ports in Côte d'Ivoire", "Côte d'Ivoire"),
        ("Show farms in the United States", "United States"),
    ],
)
def test_explicit_country_probes(parser, query, country):
    # the 5 explicit-country probe queries of App. C
    assert parser.parse(query).mentioned_country == country


def test_split_query():
    assert split_query("Vegetated area in Luanda, Angola") == ("Vegetated area", "Luanda, Angola")
    assert split_query("airports") == ("airports", None)
