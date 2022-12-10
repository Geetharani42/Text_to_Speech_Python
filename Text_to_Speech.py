import pyttsx3

speak = pyttsx3.init()
speak.setProperty('rate',150)
speak.say("Testing")
speak.runAndWait()
