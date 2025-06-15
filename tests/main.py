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
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip, TextClip, ColorClip # pip install moviepy==2.0.0.dev2 && pip install imageio==2.25.1
import random

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

# The following functions are borrowed from a tutorial https://www.youtube.com/watch?v=LWrRJx2wdWc
def textToWords(segments):
	wordlevel_info = []
	for segment in segments:
		for word in segment.words:
			wordlevel_info.append({'word':word.word.strip(),'start':word.start,'end':word.end})
	return wordlevel_info
def textToLines(json_data, max_chars=80, max_duration=3.0, max_gap=1.5):
	subtitles = []
	line = []
	line_duration = 0
	line_chars = 0
	for idx,word_data in enumerate(json_data):
		word = word_data["word"]
		start = word_data["start"]
		end = word_data["end"]
		line.append(word_data)
		line_duration += end - start
		temp = " ".join(item["word"] for item in line)
		# Check if adding a new word exceeds the maximum character count or duration
		new_line_chars = len(temp)
		duration_exceeded = line_duration > max_duration 
		chars_exceeded = new_line_chars > max_chars 
		if idx>0:
			gap = word_data['start'] - json_data[idx-1]['end'] 
			# print (word,start,end,gap)
			maxgap_exceeded = gap > max_gap
		else:
			maxgap_exceeded = False
		if duration_exceeded or chars_exceeded or maxgap_exceeded:
			if line:
				subtitle_line = {
					"word": " ".join(item["word"] for item in line),
					"start": line[0]["start"],
					"end": line[-1]["end"],
					"textcontents": line
				}
				subtitles.append(subtitle_line)
				line = []
				line_duration = 0
				line_chars = 0
	if line:
		subtitle_line = {
			"word": " ".join(item["word"] for item in line),
			"start": line[0]["start"],
			"end": line[-1]["end"],
			"textcontents": line
		}
		subtitles.append(subtitle_line)
	return subtitles
def createCaption(textJSON, framesize, font="Helvetica-Bold", fontsize=80, color='white', bgcolor='blue'):
    wordcount = len(textJSON['textcontents'])
    full_duration = textJSON['end']-textJSON['start']
    word_clips = []
    xy_textclips_positions =[]
    x_pos = 0
    y_pos = 0
    # max_height = 0
    frame_width = framesize[0]
    frame_height = framesize[1]
    x_buffer = frame_width*1/10
    y_buffer = frame_height*1/5
    space_width = ""
    space_height = ""
    for index,wordJSON in enumerate(textJSON['textcontents']):
      duration = wordJSON['end']-wordJSON['start']
      word_clip = TextClip(wordJSON['word'], font=font, fontsize=fontsize, color=color).set_start(textJSON['start']).set_duration(full_duration)
      word_clip_space = TextClip(" ", font=font, fontsize=fontsize, color=color).set_start(textJSON['start']).set_duration(full_duration)
      word_width, word_height = word_clip.size
      space_width,space_height = word_clip_space.size
      if x_pos + word_width+ space_width > frame_width-2*x_buffer:
            # Move to the next line
            x_pos = 0
            y_pos = y_pos+ word_height+40
            # Store info of each word_clip created
            xy_textclips_positions.append({
                "x_pos":x_pos+x_buffer,
                "y_pos": y_pos+y_buffer,
                "width" : word_width,
                "height" : word_height,
                "word": wordJSON['word'],
                "start": wordJSON['start'],
                "end": wordJSON['end'],
                "duration": duration
            })
            word_clip = word_clip.set_position((x_pos+x_buffer, y_pos+y_buffer))
            word_clip_space = word_clip_space.set_position((x_pos+ word_width +x_buffer, y_pos+y_buffer))
            x_pos = word_width + space_width
      else:
            # Store info of each word_clip created
            xy_textclips_positions.append({
                "x_pos":x_pos+x_buffer,
                "y_pos": y_pos+y_buffer,
                "width" : word_width,
                "height" : word_height,
                "word": wordJSON['word'],
                "start": wordJSON['start'],
                "end": wordJSON['end'],
                "duration": duration
            })
            word_clip = word_clip.set_position((x_pos+x_buffer, y_pos+y_buffer))
            word_clip_space = word_clip_space.set_position((x_pos+ word_width+ x_buffer, y_pos+y_buffer))
            x_pos = x_pos + word_width+ space_width
      word_clips.append(word_clip)
      word_clips.append(word_clip_space)  
    for highlight_word in xy_textclips_positions:
      word_clip_highlight = TextClip(highlight_word['word'], font=font, fontsize=fontsize, color=color,bg_color = bgcolor).set_start(highlight_word['start']).set_duration(highlight_word['duration'])
      word_clip_highlight = word_clip_highlight.set_position((highlight_word['x_pos'], highlight_word['y_pos']))
      word_clips.append(word_clip_highlight)
    return word_clips
def makeVideo(background_video_path="", background_audio_path="", output_path="", lines=""):
	if (background_video_path == ""):
		print("Missing background_video_path")
		return False
	if (background_audio_path == ""):
		print("Missing background_audio_path")
		return False
	if (output_path == ""):
		print("Missing output_path")
		return False
	if (lines == ""):
		print("Missing lines")
		return False
	# Get background video
	video = VideoFileClip(background_video_path)
	video_length = video.duration
	# Get narration
	audio = AudioFileClip(background_audio_path)
	audio_length = audio.duration
	# Randomly select a portion of the video based on the audio duration
	start = random.randint(1, int(video_length - audio_length)) # Assuming video_length > audio_length
	end = start + audio_length
	background_clip = video.subclip(start, end)
	# Overlay audio onto video
	background_clip = background_clip.set_audio(audio)
	# Overlay subtitles onto video
	line_clips = []
	frame_size = (1920, 1080)
	for line in lines:
		out = createCaption(line, frame_size)
		line_clips.extend(out)
	final_clip = CompositeVideoClip([background_clip] + line_clips)
	# Output the video
	final_clip.write_videofile(output_path, audio=True, fps=24, codec="libx264", audio_codec="aac", preset='fast')
	return True

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
	if (story == ""):
		print("Failed to extract the story. Please check the link.")
		quit()

	# Generate summary script
	size = 150 # 150 utterances (1 minute) limit
	summary = extractiveSummarize(story, size)
	if (summary == ""):
		print("Failed to summarize.")
		quit()
	
	# Text-to-speech
	path = "temp/output.mp3"
	gtextToSpeech(summary, path)

	# Create subtitles
	segments = transcribeEnglish(path)

	# Combine everything
	success = makeVideo(background_video_path="temp/bkg.webm", background_audio_path="temp/output.mp3", output_path="temp/example.mp4", lines=textToLines(textToWords(segments)))
	if (success == False):
		print("Failed to create video")
	
