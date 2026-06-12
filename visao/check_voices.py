# tool to help you see what voices are available on your system for text-to-speech. This can be useful for selecting
# a voice that suits your application or for debugging purposes.
import pyttsx3

engine = pyttsx3.init()
voices = engine.getProperty('voices')

for index, voice in enumerate(voices):
    print(f"Voice {index}:")
    print(f" - Name: {voice.name}")
    print(f" - ID: {voice.id}")
    print(f" - Languages: {voice.languages}")
    print("-" * 30)