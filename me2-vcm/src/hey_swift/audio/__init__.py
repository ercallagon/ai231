"""hey_swift.audio — shared runtime audio handling.

capture   microphone recording / command capture
playback  beeps, music and speaker output

These are the ONLY modules that touch the microphone / speaker. They import
`sounddevice` lazily so the rest of the package (and the test suite) runs on a
machine with no audio hardware.
"""
