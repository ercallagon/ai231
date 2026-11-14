"""Deterministic, narrow routing for known immediate device and media commands.

These pure parsers are shared by the legacy rollback and Pipecat-native path;
they own no conversation, turn, playback, or task state.
"""

import json
import re

from .providers import ToolCall


def direct_music_control(text, turn_id, history=()):
    """Route bounded, unambiguous music controls without an LLM request."""
    normalized = re.sub(r"\s+", " ", text.strip().casefold()).strip(" .!?")
    volume = re.fullmatch(
        r"(?:(?:can|could) you )?(?:please )?(?:set )?(?:the )?(?:music )?volume(?: to)? (\d{1,3}|zero|ten|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|(?:one )?hundred)(?: percent|%)(?: please)?",
        normalized,
    )
    if volume is None:
        volume = re.fullmatch(
            r"(?:(?:can|could) you )?(?:please )?(?:set )?(?:the )?(?:music )?volume (?:level|to) (\d{1,3}|zero|ten|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|(?:one )?hundred)(?: percent|%)?(?: please)?",
            normalized,
        )
    if volume:
        value = volume.group(1)
        words = {
            "zero": 0,
            "ten": 10,
            "twenty": 20,
            "thirty": 30,
            "forty": 40,
            "fifty": 50,
            "sixty": 60,
            "seventy": 70,
            "eighty": 80,
            "ninety": 90,
            "hundred": 100,
            "one hundred": 100,
        }
        number = int(value) if value.isdecimal() else words[value]
        if 0 <= number <= 100:
            return ToolCall(
                f"local-volume-{turn_id}",
                "control_music",
                {"action": "volume_set", "query": str(number)},
            )
    volume_patterns = {
        "volume_up": (
            r"(?:turn|move) (?:the )?(?:music )?volume (?:up|higher)",
            r"(?:turn|make) (?:the )?music (?:up|louder)",
            r"(?:raise|increase) (?:the )?(?:music )?volume",
            r"(?:the )?(?:music )?volume (?:up|higher)",
        ),
        "volume_down": (
            r"(?:turn|move) (?:the )?(?:music )?volume (?:down|lower)",
            r"(?:turn|make) (?:the )?music (?:down|quieter)",
            r"(?:lower|decrease) (?:the )?(?:music )?volume",
            r"(?:lower|decrease) (?:the )?music",
            r"(?:the )?(?:music )?volume (?:down|lower)",
        ),
    }
    for action, patterns in volume_patterns.items():
        if any(
            re.fullmatch(rf"(?:(?:can|could) you )?(?:please )?{pattern}(?: please)?", normalized)
            for pattern in patterns
        ):
            return ToolCall(
                f"local-{action.replace('_', '-')}-{turn_id}",
                "control_music",
                {"action": action},
            )
    if re.fullmatch(
        r"(?:(?:can|could) you |please )?stop (?:the )?(?:(?:current|this|that) )?(?:song|music|track|playback)(?: please)?",
        normalized,
    ):
        return ToolCall(
            f"local-stop-{turn_id}",
            "control_music",
            {"action": "stop"},
        )
    if re.fullmatch(
        r"(?:no[, ]+)?(?:i said )?(?:please )?stop(?: it)?(?: please)?", normalized
    ) and any(
        row.get("role") == "tool"
        and isinstance(row.get("tool_request"), dict)
        and row["tool_request"].get("name") == "control_music"
        for row in history
    ):
        return ToolCall(f"local-stop-{turn_id}", "control_music", {"action": "stop"})
    return None


def direct_native_music_control(text, turn_id):
    """Additional explicit media verbs for Pipecat's native fast path only."""
    normalized = re.sub(r"\s+", " ", text.strip().casefold()).strip(" .!?")
    controls = {
        "pause": r"(?:please )?pause (?:the )?(?:music|playback|song|track)(?: please)?",
        "resume": r"(?:please )?(?:resume|continue) (?:the )?(?:music|playback|song|track)(?: please)?",
        "next": r"(?:please )?(?:next|skip) (?:the )?(?:song|track)(?: please)?",
        "previous": r"(?:please )?(?:previous|last) (?:the )?(?:song|track)(?: please)?",
    }
    for action, pattern in controls.items():
        if re.fullmatch(pattern, normalized):
            return ToolCall(f"local-{action}-{turn_id}", "control_music", {"action": action})
    return None


def music_selection_requested(text, turn_id, history):
    """Require current-turn evidence before a model may start/change music."""
    normalized = re.sub(r"\s+", " ", text.casefold()).strip(" .!?")
    if re.search(r"\b(?:don't|do not|never|not) (?:play|start|resume)\b", normalized):
        return False
    if direct_music_control(text, turn_id, history):
        return False
    if direct_music_request(text, turn_id, history):
        return True
    return bool(
        re.search(
            r"^(?:(?:can|could|would|will) you )?(?:please )?(?:play\b|put on\b|start (?:the )?(?:music|playback|playlist|song)\b|(?:i want|i would like|i'd like) to listen to\b)|\b(?:and|then) (?:please )?play\b",
            normalized,
        )
    )


def direct_spelling_correction(text):
    """Acknowledge an explicit letter-by-letter correction without rewriting it."""
    normalized = re.sub(r"\s+", " ", text.strip()).strip(" .!?")
    match = re.fullmatch(
        r"[A-Za-z][A-Za-z'’-]{0,39} is (?:spelled|spelt)(?: as)? "
        r"((?:[A-Za-z]\s*[-–—]\s*){2,}[A-Za-z])",
        normalized,
        re.IGNORECASE,
    )
    if not match:
        return None
    letters = re.findall(r"[A-Za-z]", match.group(1))
    if not 3 <= len(letters) <= 32:
        return None
    spelling = "-".join(letter.upper() for letter in letters)
    return f"Got it. I’ll use the spelling {spelling}."


def direct_music_request(text, turn_id, history):
    """Preserve words in narrow play/correction requests; avoid guessing intent."""
    cleaned = re.sub(r"\s+", " ", text.strip()).strip(" .!?")
    prior_user = next(
        (
            row.get("content", "")
            for row in reversed(history)
            if row.get("role") == "user" and row.get("turn_id") == turn_id - 1
        ),
        "",
    )
    previous_music = any(
        row.get("role") == "tool"
        and row.get("turn_id") == turn_id - 1
        and isinstance(row.get("tool_request"), dict)
        and row["tool_request"].get("name") == "control_music"
        for row in history
    ) or bool(
        isinstance(prior_user, str)
        and re.search(r"\bplay\b.*\b(song|music|track)\b", prior_user, re.I)
    )

    def playlist_call(query, market=None):
        arguments = {"action": "play_playlist", "query": query}
        if market:
            arguments["market"] = market
        return ToolCall(f"local-playlist-{turn_id}", "control_music", arguments)

    # Mood/chart requests are discovery requests, not track titles. Search for
    # a relevant playlist context instead of sending words like "chill" or
    # "popular in PH" to Spotify's track search.
    popular = re.fullmatch(
        r"(?:(?:can|could) you |please )?play (?:me )?(?:a |some )?(?:popular|top hits?|charts?)"
        r"(?: songs?| music| playlist)? in (?:the )?(?:ph|philippines)(?: please)?",
        cleaned,
        re.I,
    )
    if popular:
        return playlist_call("Top 50 Philippines", "PH")

    region = re.search(r"\b(?:in|for) (?:the )?philippines\b|\b(?:in|for) ph\b", cleaned, re.I)
    market = "PH" if region else None
    mood = re.fullmatch(
        r"(?:(?:can|could) you |please )?play (?:me )?(?:a |some |something )?"
        r"(?P<mood>chill|relaxing|relaxed|calm|upbeat|happy|workout|exercise|study|focus|"
        r"sleep|party|romantic|sad)(?: songs?| music| playlist| tracks?)?"
        r"(?: in (?:the )?(?:ph|philippines))?(?: please)?",
        cleaned,
        re.I,
    )
    if mood:
        return playlist_call(mood.group("mood"), market)

    explicit_playlist = re.fullmatch(
        r"(?:(?:can|could) you |please )?play (?:me )?(?:the )?(?P<query>.+?) playlist"
        r"(?: in (?:the )?(?:ph|philippines))?(?: please)?",
        cleaned,
        re.I,
    )
    if explicit_playlist:
        query = explicit_playlist.group("query").strip()
        if 1 <= len(query) <= 100 and not re.search(r"\b(instead|actually|rather)\b", query, re.I):
            return playlist_call(query, market)

    match = re.fullmatch(
        r"(?:(?:can|could) you |please )?play (?:a |some )?(.+?) (?:song|music|track)(?: please)?",
        cleaned,
        re.I,
    ) or re.fullmatch(
        r"(?:(?:can|could) you |please )?play (?:a |some )?(?:song|music|track) by (.+?)(?: please)?",
        cleaned,
        re.I,
    )
    if not match:
        match = re.fullmatch(
            r"(?:(?:can|could) you |please )?play (.+?)(?: please)?", cleaned, re.I
        )
    if not match and previous_music:
        match = re.fullmatch(
            r"(?:no,?\s*)?(?:i said\s+)?play (.+?)(?: instead| please)?", cleaned, re.I
        )
    if not match and previous_music:
        match = re.fullmatch(
            r"(?:no,?\s*)?(?:can|could) you make it (.+?)(?: instead| please)?", cleaned, re.I
        )
    if not match and previous_music:
        match = re.fullmatch(r"(?:no,?\s*)?make it (.+?) now", cleaned, re.I)
    if not match:
        return None
    query = match.group(1).strip()
    query = re.sub(r"^music (?:by )?", "", query, flags=re.I)
    if (
        not 1 <= len(query) <= 100
        or not re.fullmatch(r"[\w\s'’.-]+", query)
        or (not previous_music and re.search(r"\b(instead|actually|rather)\b", query, re.I))
        or re.search(
            r"\b(and|then|stop|pause|resume|next|previous|skip|volume|louder|quieter|brighter|dimmer)\b",
            query,
            re.I,
        )
        or re.fullmatch(r"\d+\s*(?:percent|%)?", query, re.I)
        or re.fullmatch(
            r"(?:me )?(?:(?:my|a|some|any|the) )?(?:music|songs?|tracks?|playlists?|something)(?: good)?",
            query.casefold(),
        )
        or query.casefold()
        in {"a", "some", "any", "the", "me", "me a", "me some", "me any", "my", "it"}
    ):
        return None
    return ToolCall(f"local-play-{turn_id}", "control_music", {"action": "play", "query": query})


def direct_clock_query(text, turn_id):
    """Opt-in exact clock queries; preserve named timezone instead of guessing."""
    normalized = re.sub(r"\s+", " ", text.strip().casefold()).strip(" .!?")
    location = re.fullmatch(
        r"(?:please )?(?:(?:what )?time is it|what's the time|tell me the time) in (?:the )?(manila|philippines|japan)(?: please)?",
        normalized,
    )
    if location:
        zone = {
            "manila": "Asia/Manila",
            "philippines": "Asia/Manila",
            "japan": "Asia/Tokyo",
        }[location.group(1)]
        return ToolCall(
            f"local-time-{turn_id}", "run_device_action", {"action": "get_time", "value": zone}
        )
    if re.fullmatch(
        r"(?:please )?(?:what time is it|what's the time|tell me the time)(?: please)?", normalized
    ):
        return ToolCall(f"local-time-{turn_id}", "run_device_action", {"action": "get_time"})
    return None


def explicit_percentage(value):
    """Parse an explicit integer 0..100; never rewrite the user's transcript."""
    if value.isdecimal():
        number = int(value)
        return number if 0 <= number <= 100 else None
    units = dict(
        zip(
            "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split(),
            range(20),
        )
    )
    tens = dict(
        zip("twenty thirty forty fifty sixty seventy eighty ninety".split(), range(20, 100, 10))
    )
    words = value.replace("-", " ").split()
    if len(words) == 1:
        return units.get(value, tens.get(value))
    if words == ["one", "hundred"]:
        return 100
    if len(words) == 2 and words[0] in tens and words[1] in units and 1 <= units[words[1]] <= 9:
        return tens[words[0]] + units[words[1]]
    return None


def direct_me2_command(text, turn_id):
    """Exact final-utterance ME2 operations through the existing shared executor."""
    from .devices import ACTION_VALUES

    phrase = re.sub(r"\s+", " ", text.strip().casefold()).strip(" .!?")
    phrase = re.sub(r"^please ", "", phrase)
    action, value = None, None
    if phrase in {"make a call", "make a phone call", "call the demo contact"}:
        action = "call_demo"
    elif phrase in {"send a message", "send a text", "send the demo message"}:
        action = "message_demo"
    elif phrase in {"list reminders", "list my reminders", "what are my reminders"}:
        action = "list_reminders"
    elif phrase in {"cancel the timer", "cancel my timer", "cancel the alarm", "cancel my alarm"}:
        action = "cancel_schedule"
    elif phrase in {
        "what's the weather",
        "what is the weather",
        "what's the weather like",
        "check the weather",
    }:
        action = "get_weather"
    else:
        patterns = (
            (
                "get_weather",
                r"(?:what(?:'s| is) the weather(?: like)?|check the weather) in (.{2,80})",
            ),
            ("set_timer", r"(?:set|start) (?:a |the )?timer (?:for )?(.+)"),
            ("set_alarm", r"set (?:an |the )?alarm (?:for |at )?(.+)"),
            ("set_temperature", r"(?:set|show) (?:the )?temperature (?:to )?(.+)"),
            ("create_reminder", r"(?:remind me to|create a reminder to) (.+)"),
        )
        for candidate, pattern in patterns:
            match = re.fullmatch(pattern, phrase)
            if not match:
                continue
            value = match.group(1)
            if candidate == "set_timer":
                value = {
                    "ten seconds": "10 seconds",
                    "thirty seconds": "30 seconds",
                    "one minute": "1 minute",
                }.get(value, value)
            elif candidate == "set_alarm":
                value = {"six am": "6:00 AM", "eight am": "8:00 AM", "nine pm": "9:00 PM"}.get(
                    value,
                    value.upper()
                    .replace("6 AM", "6:00 AM")
                    .replace("8 AM", "8:00 AM")
                    .replace("9 PM", "9:00 PM"),
                )
            elif candidate == "set_temperature":
                value = {
                    "eighteen degrees": "18 degrees",
                    "twenty two degrees": "22 degrees",
                    "twenty six degrees": "26 degrees",
                }.get(value, value)
            if candidate in ACTION_VALUES and value not in ACTION_VALUES[candidate]:
                return None
            action = candidate
            break
    if action is None:
        return None
    args = {"action": action}
    if value is not None:
        args["value"] = value
    return ToolCall(f"local-me2-{turn_id}", "run_device_action", args)


def direct_light_command(text, turn_id):
    """Route exact final light commands through the shared device tool contract."""
    normalized = re.sub(r"\s+", " ", text.strip().casefold()).strip(" .!?")
    requester = r"(?:(?:can|could) you )?(?:please )?"

    power = re.fullmatch(
        requester + r"(?:turn|switch) (on|off) (?:the )?lights?(?: please)?", normalized
    )
    if power:
        action = "light_on" if power.group(1) == "on" else "light_off"
        return ToolCall(f"local-light-{turn_id}", "run_device_action", {"action": action})

    power = re.fullmatch(
        requester + r"(?:turn|switch) (?:the )?lights? (on|off)(?: please)?", normalized
    )
    if power:
        action = "light_on" if power.group(1) == "on" else "light_off"
        return ToolCall(f"local-light-{turn_id}", "run_device_action", {"action": action})

    color = re.fullmatch(
        requester + r"(?:set|change|make) (?:the )?lights?(?: color)?(?: to)? "
        r"(#[0-9a-f]{6}|warm white|orange|purple|pink|yellow|cyan|magenta|white|teal|amber|violet|lime|navy|coral|red|green|blue)(?: please)?",
        normalized,
    )
    if color:
        return ToolCall(
            f"local-light-{turn_id}",
            "run_device_action",
            {"action": "set_color", "value": color.group(1)},
        )

    brightness_patterns = (
        requester
        + r"set (?:the )?(?:light )?brightness to ([a-z0-9]+(?:[ -][a-z]+)*?)(?:%| percent)(?: please)?",
        requester
        + r"set (?:the )?lights? to ([a-z0-9]+(?:[ -][a-z]+)*?)(?:%| percent)(?: please)?",
        requester + r"make (?:the )?lights? ([a-z0-9]+(?:[ -][a-z]+)*?)(?:%| percent)(?: please)?",
    )
    for pattern in brightness_patterns:
        brightness = re.fullmatch(pattern, normalized)
        if brightness:
            level = explicit_percentage(brightness.group(1))
            if level is not None:
                return ToolCall(
                    f"local-light-{turn_id}",
                    "run_device_action",
                    {"action": "set_brightness", "value": str(level)},
                )
            return None
    return None


def direct_light_followup(text, turn_id, history, *, history_turn=None):
    """Resolve an explicit percentage only after the immediately preceding light action."""
    normalized = re.sub(r"\s+", " ", text.strip().casefold()).strip(" .!?")
    power = re.fullmatch(
        r"(?:please )?(?:turn|switch) it (on|off)(?: again)?(?: please)?", normalized
    )
    match = re.fullmatch(
        r"(?:(?:can|could) you )?(?:please )?(?:make|set) (?:it|the light|the brightness)(?: to)? ([a-z0-9]+(?:[ -][a-z]+)*?)(?:%| percent)(?: please)?",
        normalized,
    ) or re.fullmatch(
        r"(?:actually|no)[, ]+(?:make it |set it to )?([a-z0-9]+(?:[ -][a-z]+)*?)(?:%| percent)",
        normalized,
    )
    if match is None:
        match = re.fullmatch(
            r"(?:(?:actually|no)[, ]+|sorry[, ]+(?:(?:you )?(?:can|could) )?)?"
            r"(?:(?:can|could) you )?(?:please )?"
            r"(?:dim (?:it|the light|the lights) to |(?:set|make) it (?:to )?)"
            r"([a-z0-9]+(?:[ -][a-z]+)*?)(?:%| percent)?(?: please)?",
            normalized,
        )
    level = explicit_percentage(match.group(1)) if match else None
    color = re.fullmatch(
        r"(?:(?:actually|no)[, ]+|sorry[, ]+(?:(?:you )?(?:can|could) )?)?"
        r"(?:(?:can|could) you )?(?:please )?"
        r"(?:set|make|change) it (?:to )?"
        r"(#[0-9a-f]{6}|warm white|orange|purple|pink|yellow|cyan|magenta|white|teal|amber|violet|lime|navy|coral|red|green|blue)(?: please)?",
        normalized,
    )
    if not power and level is None:
        if color is None:
            return None
    prior_turn = history_turn if history_turn is not None else turn_id
    prior = next(
        (
            row
            for row in reversed(history)
            if row.get("role") == "tool" and row.get("turn_id") == prior_turn - 1
        ),
        None,
    )
    if not prior:
        return None
    request = prior.get("tool_request")
    if not isinstance(request, dict):
        return None
    try:
        args = json.loads(request.get("arguments", "{}"))
        receipt = json.loads(prior.get("content", "{}"))
    except (TypeError, ValueError):
        return None
    name = request.get("name")
    is_light = name == "set_mock_light" or (
        name in {"run_device_action", "run_mock_action"}
        and isinstance(args, dict)
        and args.get("action") in {"light_on", "light_off", "set_color", "set_brightness"}
    )
    if (
        not is_light
        or not isinstance(receipt, dict)
        or receipt.get("result_status") in {"failed", "unavailable", "unknown"}
    ):
        return None
    return ToolCall(
        f"local-light-{turn_id}",
        "run_device_action",
        (
            {"action": "light_on" if power.group(1) == "on" else "light_off"}
            if power
            else {"action": "set_brightness", "value": str(level)}
            if level is not None
            else {"action": "set_color", "value": color.group(1)}
        ),
    )


def direct_clarification(text, *, communication_enabled=False):
    """Ask for the missing content of a narrowly defined reminder request."""
    normalized = re.sub(r"\s+", " ", text.strip().casefold()).strip(" .!?")
    if re.fullmatch(
        r"(?:(?:can|could) you )?(?:please )?(?:turn|switch) on (?:the )?music"
        r"(?: please)?|(?:(?:can|could) you )?(?:please )?play (?:me )?(?:some )?music"
        r"(?: please)?|(?:(?:can|could) you )?(?:please )?start (?:the )?music(?: please)?",
        normalized,
    ):
        return "What song, artist, or playlist would you like me to play?"
    if re.fullmatch(
        r"(?:please )?(?:set|create|add) (?:my |a |some )?reminders?(?: please)?", normalized
    ):
        return "What would you like me to remind you about?"
    if re.fullmatch(r"[a-z](?:[ .-]+[a-z]){1,5}", normalized):
        return "I only caught a few letters. Could you repeat the full request?"
    if normalized in {"cancel that", "cancel it"}:
        return "Which task would you like to cancel?"
    if normalized in {
        "music",
        "song",
        "songs",
        "playlist",
        "the music",
        "stop",
        "i said stop",
        "stop it",
    }:
        if normalized in {"stop", "i said stop", "stop it"}:
            return "Would you like me to stop the music or stop speaking?"
        return "Would you like to play, pause, or stop the music?"
    if not communication_enabled and re.fullmatch(
        r"(?:please )?(?:make a call|make a phone call|send a message|send a text)(?: please)?",
        normalized,
    ):
        return "Calling and messaging aren't configured in this session."
    return None
