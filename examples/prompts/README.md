# Example voice prompts

Two real prompt recordings for trying `src/synthesize.py`, which needs a short recording of
the voice to clone and its exact transcript. The quickstarts in the
[README](../../README.md) and on the
[model card](https://huggingface.co/nityanandmathur/aspect-d-masked-diffusion-tts) use them.

| File | Item | Speaker | Length | Transcript (`--prompt-text`) |
|---|---|---|---|---|
| `it0000_prompt.flac` | it0000 | `EN_B00006_S03025` | 3.09 s | Yep. So, shouldn't you just get network plus so that you get that? |
| `it0001_prompt.flac` | it0001 | `EN_B00001_S04135` | 3.43 s | Oh, here we go! Get me close. |

Both are 24 kHz mono and shorter than 3.5 s, so the script uses them whole. Each is the prompt
of a zero-shot evaluation item. The item's target sentence, which the speaker says in a
different recording, makes a good `--text`:

- it0000: I mean, it's, I'm sorry to put you in the hot seat. It's just these are, I think
  it's questions that a lot of people would have about trying to make the most of their
  time, make the most of their money.
- it0001: Our goal is gonna be to win a whole round, buddy. Get that yellow, get that yellow!

[prompts.json](prompts.json) has the same text in machine-readable form. It also lists items
it0002 and it0003, without audio, because `src/camera_ready_search.py` measures phonemes per
character on the text of all four items.

## License

These are real recordings of Emilia speakers, not synthetic audio. The clips and the
transcripts are released under CC BY-NC 4.0, for non-commercial research only, as
[MODEL_LICENSE.md](../../MODEL_LICENSE.md) explains. Emilia does not own the copyright of
its audio. A rights holder who wants a clip removed can open an issue on the repository.
