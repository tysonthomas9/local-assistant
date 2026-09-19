"""Internet radio player shared by the play_radio / stop_radio tools.

Stations come from the radio-browser.info directory (free, no API key): name search first, then
genre tag. Playback is a GStreamer playbin inside the conversation app's process, sent to the
robot's speaker through the shared ALSA dmix device `reachymini_audio_sink` (see ~/.asoundrc), so
the robot's own voice mixes over the music.

Settings: REACHY_RADIO_VOLUME (0-100, default 30), REACHY_RADIO_SINK (a gst-launch sink
description; default the robot speaker, e.g. "fakesink" for tests).
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

import httpx

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

logger = logging.getLogger("reachy_radio")

DIRECTORY = "https://all.api.radio-browser.info/json/stations/search"
USER_AGENT = "reachy-mini-radio/1.0"
DEFAULT_SINK = "audioconvert ! audioresample ! alsasink device=reachymini_audio_sink"


def search(query: str, limit: int = 8) -> list[dict[str, Any]]:
    """Stations matching `query` by name, else by genre tag, most popular first."""
    params = {"limit": limit, "hidebroken": "true", "order": "clickcount", "reverse": "true"}
    with httpx.Client(timeout=8, headers={"User-Agent": USER_AGENT}, follow_redirects=True) as c:
        for key in ("name", "tag"):
            r = c.get(DIRECTORY, params={**params, key: query.lower() if key == "tag" else query})
            r.raise_for_status()
            hits = [s for s in r.json() if s.get("url_resolved") and s.get("codec", "").upper() in ("MP3", "AAC", "AAC+", "OGG")]
            if hits:
                return hits
    return []


class Radio:
    def __init__(self) -> None:
        Gst.init(None)
        self._lock = threading.Lock()
        self._player: Gst.Element | None = None
        self.station: dict[str, Any] | None = None

    def _make_player(self, url: str, volume: float) -> Gst.Element:
        player = Gst.ElementFactory.make("playbin", "reachy-radio")
        player.set_property("uri", url)
        player.set_property("volume", volume)
        sink_desc = os.environ.get("REACHY_RADIO_SINK", DEFAULT_SINK)
        player.set_property("audio-sink", Gst.parse_bin_from_description(sink_desc, True))
        player.set_property("video-sink", Gst.ElementFactory.make("fakesink", None))
        return player

    def play(self, query: str, volume_pct: float | None = None) -> dict[str, Any]:
        """Find a station and start it; tries the next match if one won't play within 8 s."""
        volume = max(0.0, min(1.0, (volume_pct if volume_pct is not None
                                    else float(os.environ.get("REACHY_RADIO_VOLUME", 30))) / 100))
        stations = search(query)
        if not stations:
            return {"error": f"No radio station found for {query!r}."}
        self.stop()
        for st in stations[:3]:
            player = self._make_player(st["url_resolved"], volume)
            player.set_state(Gst.State.PLAYING)
            bus = player.get_bus()
            msg = bus.timed_pop_filtered(8 * Gst.SECOND, Gst.MessageType.ERROR | Gst.MessageType.ASYNC_DONE)
            if msg is not None and msg.type == Gst.MessageType.ASYNC_DONE:
                with self._lock:
                    self._player, self.station = player, st
                threading.Thread(target=self._watch, args=(player,), daemon=True, name="reachy-radio-watch").start()
                logger.info("Radio playing %r (%s)", st["name"], st["url_resolved"])
                return {"playing": st["name"].strip(), "country": st.get("country", ""),
                        "genre": ", ".join(st.get("tags", "").split(",")[:3]), "volume": round(volume * 100)}
            err = msg.parse_error()[0].message if msg is not None and msg.type == Gst.MessageType.ERROR else "timed out"
            logger.warning("Radio: %r failed (%s); trying next", st["name"], err)
            player.set_state(Gst.State.NULL)
        return {"error": f"Found stations for {query!r}, but none would play right now."}

    def _watch(self, player: Gst.Element) -> None:
        """Clean up if the stream ends or errors (e.g. network drop)."""
        msg = player.get_bus().timed_pop_filtered(Gst.CLOCK_TIME_NONE, Gst.MessageType.ERROR | Gst.MessageType.EOS)
        with self._lock:
            if self._player is not player:
                return  # already stopped or replaced
            logger.warning("Radio stream ended: %s", msg.parse_error()[0].message if msg.type == Gst.MessageType.ERROR else "EOS")
            player.set_state(Gst.State.NULL)
            self._player, self.station = None, None

    def stop(self) -> str | None:
        with self._lock:
            player, station = self._player, self.station
            self._player, self.station = None, None
        if player is None:
            return None
        player.set_state(Gst.State.NULL)
        player.get_bus().post(Gst.Message.new_eos(player))  # release the watcher thread
        return station["name"].strip() if station else None


RADIO = Radio()
