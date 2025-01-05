# This file is part of beets.
# Copyright 2016, Fabrice Laporte.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.

"""Tests for the 'lastgenre' plugin."""

from unittest.mock import Mock

import pytest

from beets import config
from beets.test import _common
from beets.test.helper import BeetsTestCase
from beetsplug import lastgenre


class LastGenrePluginTest(BeetsTestCase):
    def setUp(self):
        super().setUp()
        self.plugin = lastgenre.LastGenrePlugin()

    def _setup_config(
        self, whitelist=False, canonical=False, count=1, prefer_specific=False
    ):
        config["lastgenre"]["canonical"] = canonical
        config["lastgenre"]["count"] = count
        config["lastgenre"]["prefer_specific"] = prefer_specific
        if isinstance(whitelist, (bool, (str,))):
            # Filename, default, or disabled.
            config["lastgenre"]["whitelist"] = whitelist
        self.plugin.setup()
        if not isinstance(whitelist, (bool, (str,))):
            # Explicit list of genres.
            self.plugin.whitelist = whitelist

    def test_default(self):
        """Fetch genres with whitelist and c14n deactivated"""
        self._setup_config()
        assert self.plugin._resolve_genres(["delta blues"]) == "Delta Blues"

    def test_c14n_only(self):
        """Default c14n tree funnels up to most common genre except for *wrong*
        genres that stay unchanged.
        """
        self._setup_config(canonical=True, count=99)
        assert self.plugin._resolve_genres(["delta blues"]) == "Blues"
        assert self.plugin._resolve_genres(["iota blues"]) == "Iota Blues"

    def test_whitelist_only(self):
        """Default whitelist rejects *wrong* (non existing) genres."""
        self._setup_config(whitelist=True)
        assert self.plugin._resolve_genres(["iota blues"]) == ""

    def test_whitelist_c14n(self):
        """Default whitelist and c14n both activated result in all parents
        genres being selected (from specific to common).
        """
        self._setup_config(canonical=True, whitelist=True, count=99)
        assert (
            self.plugin._resolve_genres(["delta blues"]) == "Delta Blues, Blues"
        )

    def test_whitelist_custom(self):
        """Keep only genres that are in the whitelist."""
        self._setup_config(whitelist={"blues", "rock", "jazz"}, count=2)
        assert self.plugin._resolve_genres(["pop", "blues"]) == "Blues"

        self._setup_config(canonical="", whitelist={"rock"})
        assert self.plugin._resolve_genres(["delta blues"]) == ""

    def test_count(self):
        """Keep the n first genres, as we expect them to be sorted from more to
        less popular.
        """
        self._setup_config(whitelist={"blues", "rock", "jazz"}, count=2)
        assert (
            self.plugin._resolve_genres(["jazz", "pop", "rock", "blues"])
            == "Jazz, Rock"
        )

    def test_count_c14n(self):
        """Keep the n first genres, after having applied c14n when necessary"""
        self._setup_config(
            whitelist={"blues", "rock", "jazz"}, canonical=True, count=2
        )
        # thanks to c14n, 'blues' superseeds 'country blues' and takes the
        # second slot
        assert (
            self.plugin._resolve_genres(
                ["jazz", "pop", "country blues", "rock"]
            )
            == "Jazz, Blues"
        )

    def test_c14n_whitelist(self):
        """Genres first pass through c14n and are then filtered"""
        self._setup_config(canonical=True, whitelist={"rock"})
        assert self.plugin._resolve_genres(["delta blues"]) == ""

    def test_empty_string_enables_canonical(self):
        """For backwards compatibility, setting the `canonical` option
        to the empty string enables it using the default tree.
        """
        self._setup_config(canonical="", count=99)
        assert self.plugin._resolve_genres(["delta blues"]) == "Blues"

    def test_empty_string_enables_whitelist(self):
        """Again for backwards compatibility, setting the `whitelist`
        option to the empty string enables the default set of genres.
        """
        self._setup_config(whitelist="")
        assert self.plugin._resolve_genres(["iota blues"]) == ""

    def test_prefer_specific_loads_tree(self):
        """When prefer_specific is enabled but canonical is not the
        tree still has to be loaded.
        """
        self._setup_config(prefer_specific=True, canonical=False)
        assert self.plugin.c14n_branches != []

    def test_prefer_specific_without_canonical(self):
        """Prefer_specific works without canonical."""
        self._setup_config(prefer_specific=True, canonical=False, count=4)
        assert (
            self.plugin._resolve_genres(["math rock", "post-rock"])
            == "Post-Rock, Math Rock"
        )

    def test_no_duplicate(self):
        """Remove duplicated genres."""
        self._setup_config(count=99)
        assert self.plugin._resolve_genres(["blues", "blues"]) == "Blues"

    def test_tags_for(self):
        class MockPylastElem:
            def __init__(self, name):
                self.name = name

            def get_name(self):
                return self.name

        class MockPylastObj:
            def get_top_tags(self):
                tag1 = Mock()
                tag1.weight = 90
                tag1.item = MockPylastElem("Pop")
                tag2 = Mock()
                tag2.weight = 40
                tag2.item = MockPylastElem("Rap")
                return [tag1, tag2]

        plugin = lastgenre.LastGenrePlugin()
        res = plugin._tags_for(MockPylastObj())
        assert res == ["pop", "rap"]
        res = plugin._tags_for(MockPylastObj(), min_weight=50)
        assert res == ["pop"]

    def test_sort_by_depth(self):
        self._setup_config(canonical=True)
        # Normal case.
        tags = ("electronic", "ambient", "post-rock", "downtempo")
        res = self.plugin._sort_by_depth(tags)
        assert res == ["post-rock", "downtempo", "ambient", "electronic"]
        # Non-canonical tag ('chillout') present.
        tags = ("electronic", "ambient", "chillout")
        res = self.plugin._sort_by_depth(tags)
        assert res == ["ambient", "electronic"]


@pytest.mark.parametrize(
    "config_values, item_genre, mock_genres, expected_result",
    [
        # 0 - force and keep whitelisted
        (
            {
                "force": True,
                "keep_existing": True,
                "source": "album",  # means album or artist genre
                "whitelist": True,
                "canonical": False,
                "prefer_specific": False,
                "count": 10,
            },
            "Blues",
            {
                "album": ["Jazz"],
            },
            ("Blues, Jazz", "keep + album"),
        ),
        # 1 - force and keep whitelisted, unknown original
        (
            {
                "force": True,
                "keep_existing": True,
                "source": "album",
                "whitelist": True,
                "canonical": False,
                "prefer_specific": False,
            },
            "original unknown, Blues",
            {
                "album": ["Jazz"],
            },
            ("Blues, Jazz", "keep + album"),
        ),
        # 2 - force and keep whitelisted on empty tag
        (
            {
                "force": True,
                "keep_existing": True,
                "source": "album",
                "whitelist": True,
                "canonical": False,
                "prefer_specific": False,
            },
            "",
            {
                "album": ["Jazz"],
            },
            ("Jazz", "album"),
        ),
        # 3 force and keep, artist configured
        (
            {
                "force": True,
                "keep_existing": True,
                "source": "artist",  # means artist genre, original or fallback
                "whitelist": True,
                "canonical": False,
                "prefer_specific": False,
            },
            "original unknown, Blues",
            {
                "album": ["Jazz"],
                "artist": ["Pop"],
            },
            ("Blues, Pop", "keep + artist"),
        ),
        # 4 - don't force, disabled whitelist
        (
            {
                "force": False,
                "keep_existing": False,
                "source": "album",
                "whitelist": False,
                "canonical": False,
                "prefer_specific": False,
            },
            "any genre",
            {
                "album": ["Jazz"],
            },
            ("any genre", "keep"),
        ),
        # 5 - don't force, disabled whitelist, empty
        (
            {
                "force": False,
                "keep_existing": False,
                "source": "album",
                "whitelist": False,
                "canonical": False,
                "prefer_specific": False,
            },
            "",
            {
                "album": ["Jazz"],
            },
            ("Jazz", "album"),
        ),
        # 6 - fallback to next stages until found
        (
            {
                "force": True,
                "keep_existing": True,
                "source": "track",  # means track,album,artist,...
                "whitelist": False,
                "canonical": False,
                "prefer_specific": False,
            },
            "unknown genre",
            {
                "track": None,
                "album": None,
                "artist": ["Jazz"],
            },
            ("Unknown Genre, Jazz", "keep + artist"),
        ),
        # 7 - fallback to original when nothing found
        (
            {
                "force": True,
                "keep_existing": True,
                "source": "track",
                "whitelist": True,
                "fallback": "fallback genre",
                "canonical": False,
                "prefer_specific": False,
            },
            "original unknown",
            {
                "track": None,
                "album": None,
                "artist": None,
            },
            ("original unknown", "original fallback"),
        ),
        # 8 - fallback to fallback if no original
        (
            {
                "force": True,
                "keep_existing": True,
                "source": "track",
                "whitelist": True,
                "fallback": "fallback genre",
                "canonical": False,
                "prefer_specific": False,
            },
            "",
            {
                "track": None,
                "album": None,
                "artist": None,
            },
            ("fallback genre", "fallback"),
        ),
        # 9 - null charachter as separator
        (
            {
                "force": True,
                "keep_existing": True,
                "source": "album",
                "whitelist": True,
                "separator": "\u0000",
                "canonical": False,
                "prefer_specific": False,
            },
            "Blues",
            {
                "album": ["Jazz"],
            },
            ("Blues\u0000Jazz", "keep + album"),
        ),
        # 10 - limit a lot of results to just 3
        (
            {
                "force": True,
                "keep_existing": True,
                "source": "album",
                "whitelist": True,
                "count": 3,
                "canonical": False,
                "prefer_specific": False,
            },
            "original unknown, Blues, Rock, Folk, Metal",
            {
                "album": ["Jazz"],
            },
            ("Blues, Rock, Metal", "keep + album"),
        ),
    ],
)
def test_get_genre(config_values, item_genre, mock_genres, expected_result):
    """Test _get_genre with various configurations."""

    def mock_fetch_track_genre(self, obj=None):
        return mock_genres["track"]

    def mock_fetch_album_genre(self, obj):
        return mock_genres["album"]

    def mock_fetch_artist_genre(self, obj):
        return mock_genres["artist"]

    # Mock the last.fm fetchers. When whitelist enabled, we can assume only
    # whitelisted genres get returned, the plugin's _resolve_genre method
    # ensures it.
    lastgenre.LastGenrePlugin.fetch_track_genre = mock_fetch_track_genre
    lastgenre.LastGenrePlugin.fetch_album_genre = mock_fetch_album_genre
    lastgenre.LastGenrePlugin.fetch_artist_genre = mock_fetch_artist_genre

    # Initialize plugin instance and item
    plugin = lastgenre.LastGenrePlugin()
    item = _common.item()
    item.genre = item_genre

    # Configure
    config["lastgenre"] = config_values

    # Run
    res = plugin._get_genre(item)
    assert res == expected_result
