"""Utility to list available pyttsx3 voices on the current system."""

import pyttsx3


def main() -> None:
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')

    for index, voice in enumerate(voices):
        print(f"Voice {index}:")
        print(f" - Name: {voice.name}")
        print(f" - ID: {voice.id}")
        print(f" - Languages: {voice.languages}")
        print("-" * 30)


if __name__ == "__main__":
    main()