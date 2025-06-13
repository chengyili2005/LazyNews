# == Dependencies ==
import os
import sys
import requests
from bs4 import BeautifulSoup
import spacy # pip install spacy
from spacy.lang.en.stop_words import STOP_WORDS # python -m spacy download en_core_web_sm
from string import punctuation
from heapq import nlargest
from gtts import gTTS # pip install gTTS
import time
import math
import ffmpeg # pip install ffmpeg-python
from faster_whisper import WhisperModel # pip install faster-whisper

# Global variables
stopwords = STOP_WORDS
nlp = spacy.load('en_core_web_sm')
punctuation = punctuation + '\n' + '—' + '“' + '”' + '...'

# == Class definitions ==

# Extract AP news
def extractStory(ap_news_link):
	story = ""
	if "article" not in ap_news_link:
		return story
	else: 
		website = requests.get(ap_news_link)
		soup = BeautifulSoup(website.content, 'html.parser')
		storyElements = soup.find('div', class_='RichTextStoryBody RichTextBody')
		paragraphElements = storyElements.find_all('p')
		for element in paragraphElements:
			story += (element.text) + ' '
		return story
	
# Use extractive summarization for summarizing
def extractiveSummarize(text, size):
	try:
		doc = nlp(text)
		tokens = [token.text for token in doc]
		word_frequencies = {}
		for word in doc:
			if (word.text.lower() not in stopwords) and (word.text.lower() not in punctuation):
				if word.text not in word_frequencies.keys():
					word_frequencies[word.text] = 1
				else:
					word_frequencies[word.text] = word_frequencies[word.text] + 1
		max_frequency = max(word_frequencies.values())
		for word in word_frequencies.keys():
			word_frequencies[word] = word_frequencies[word]/max_frequency
		sentence_tokens = [sent for sent in doc.sents]
		sentence_scores = {}
		for sent in sentence_tokens:
			for word in sent:
				if word.text.lower() in word_frequencies.keys():
					if sent not in sentence_scores.keys():
						sentence_scores[sent] = word_frequencies[word.text.lower()]
					else:
						sentence_scores[sent] += word_frequencies[word.text.lower()]
		percentage = float(size/len(doc))
		select_length = int(len(sentence_tokens)*percentage)
		summary = nlargest(select_length, sentence_scores, key = sentence_scores.get)
		summary_words = [word.text for word in summary]
		final_summary = ' '.join(summary_words)
	except:
		final_summary = ""
	return final_summary

# Text to speech: Google
def gtextToSpeech(text, path):
	language = 'en'
	mygTTS = gTTS(text=text, lang=language, slow=False)
	mygTTS.save(path)

# Generate transcript
def transcribeEnglish(input_path):
	model = WhisperModel("small")
	segments, info = model.transcribe(input_path, word_timestamps=True) # Debugging: Has parameter for prompts
	segments = list(segments)
	return segments

# == Main call ==
if __name__ == "__main__":
	
	# Grab command line arguments
	# - arg 1: link
	# - arg 2: background choice
	# - arg 3: summarization technique
	# - arg 4: narrarator choice
	link = sys.argv[1]
	
	# Pipeline:
	# 1) Webscrape for article's story
	# 2) Generate summary script
	# 3) Text-to-Speech
	# 4) Subtitles
	# 5) Combine narration, subtitles, and background into video

	# Webscrape for story
	story = extractStory(link)
	if story == "":
		print("Failed to extract the story. Please check the link.")
		quit()

	# Generate summary script
	size = 150 # 150 utterances (1 minute) limit
	summary = extractiveSummarize(story, size)
	if summary == "":
		print("Failed to summarize.")
		quit()
	
	# Text-to-speech
	path = "temp/output.mp3"
	gtextToSpeech(summary, path)

	# Create subtitles
	segments = transcribeEnglish(path)

	# Combine everything
	

	
