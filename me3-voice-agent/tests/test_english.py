from me3_voice.english import (
    ASR_ENGLISH_ONLY_INSTRUCTIONS,
    QWEN_TTS_STYLE_INSTRUCTIONS,
    SWIFT_ENGLISH_OUTPUT_POLICY,
    EnglishTextRejected,
    has_non_latin_letters,
    looks_like_asr_commentary,
    rejects_english_asr,
)


def test_english_speech_policy_prompts_are_clear_and_non_cheerleading():
    assert "foreground speech in English" in ASR_ENGLISH_ONLY_INSTRUCTIONS
    assert "do not translate" in ASR_ENGLISH_ONLY_INSTRUCTIONS.casefold()
    assert "NO_CLEAR_ENGLISH_SPEECH" in ASR_ENGLISH_ONLY_INSTRUCTIONS
    assert "with underscores and no punctuation" in ASR_ENGLISH_ONLY_INSTRUCTIONS
    assert "must be in English" in SWIFT_ENGLISH_OUTPUT_POLICY
    assert "laughter, giggling" in SWIFT_ENGLISH_OUTPUT_POLICY
    assert "natural conversational pacing" in QWEN_TTS_STYLE_INSTRUCTIONS
    assert "genuinely positive news lightly upbeat" in QWEN_TTS_STYLE_INSTRUCTIONS
    assert "Add no laughter, giggles, chuckles" in QWEN_TTS_STYLE_INSTRUCTIONS
    assert "supplied English text" in QWEN_TTS_STYLE_INSTRUCTIONS
    assert len(QWEN_TTS_STYLE_INSTRUCTIONS) <= 500


def test_english_script_guard_allows_latin_text_and_rejects_other_scripts():
    assert not has_non_latin_letters("Raspberry Pi, café, and Wi-Fi.")
    assert has_non_latin_letters("你好。")
    assert has_non_latin_letters("Привет.")
    assert has_non_latin_letters("مرحبا.")
    assert has_non_latin_letters("안녕하세요.")
    assert has_non_latin_letters(None)
    assert looks_like_asr_commentary("The spoken words are: 'Turn on the lights.'")
    assert looks_like_asr_commentary("The original content of this audio is: hello.")
    # Qwen2-Audio answered the known capital question with this template-like
    # completion instead of transcribing the prompt. It must not enter history.
    assert rejects_english_asr("The capital of the Philippines is ___.")
    assert not rejects_english_asr("The capital of the Philippines is Manila.")
    assert not looks_like_asr_commentary("Turn on the lights.")
    for marker in (
        "NO_CLEAR_ENGLISH_SPEECH",
        "No.Clear.English.Speech",
        "no clear english speech",
        "No-Clear-English-Speech!",
    ):
        assert rejects_english_asr(marker)
    assert rejects_english_asr("你好。")
    assert rejects_english_asr("The spoken words are: hello.")
    assert not rejects_english_asr("Turn on the lights.")


def test_internal_asr_marker_normalization_does_not_reject_real_transcripts():
    assert not rejects_english_asr("The person says hello when I arrive.")
    assert not rejects_english_asr("Please clear the English speech audio.")
    assert not rejects_english_asr("No, clear English speech is on the screen.")


def test_english_rejections_keep_safe_stage_and_code_without_text():
    asr = EnglishTextRejected("asr")
    tts = EnglishTextRejected("tts")
    llm = EnglishTextRejected("llm")

    assert (asr.role, asr.error_stage, asr.code) == ("asr", "asr", "english_only_asr")
    assert (tts.role, tts.error_stage, tts.code) == ("tts", "tts", "english_only_tts")
    assert (llm.role, llm.error_stage, llm.code) == ("llm", "reasoning", "english_only_output")
    assert "你好" not in str(asr)
