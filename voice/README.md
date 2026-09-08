# Gene Voice

Default wake phrase:

hello gene

Default hotkey:

x

Both are configurable through:

gene/config/desktop.json

The initial Windows voice adapter uses the Windows System.Speech
API. The interface is provider-independent so it can later be
replaced with a neural STT/TTS provider without changing Gene's
orchestration architecture.
