import speech_recognition as sr
import numpy as np
from google.cloud import speech
import os

# Import the Speech-to-Text client library
from google.cloud import speech

# Instantiates a client
client = speech.SpeechClient.from_service_account_file('quran-374914-afa6baf2bbe3.json')


# local_file_path = "G:\OneDrive - University of Engineering and Technology Taxila\online clason ka zamana\FYP NS\FYP\CODES\\backend\kahaf.mp3"
# with open(local_file_path, "rb") as f:
#     mp3_data = f.read()

# flac = "G:\OneDrive - University of Engineering and Technology Taxila\online clason ka zamana\FYP NS\FYP\CODES\backend\kahaf.flac"
gcs_uri = "gs://quran_mulk/audio-files/kahaf.flac"

def transcribe_speech():
    audio = speech.RecognitionAudio(uri=gcs_uri)
    # audio = speech.RecognitionAudio(content=flac)
    # audio = speech.RecognitionAudio(audiofile)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.FLAC,
        sample_rate_hertz=44100,
        language_code="ar-SA",
        model="default",
        audio_channel_count=2,
        enable_word_time_offsets=True,
    )

    # Detects speech in the audio file
    operation = client.long_running_recognize(config=config, audio=audio)

    print("Waiting for operation to complete...")
    result = operation.result(timeout=90)

    num = 1  
    def second_to_timecode(x: float) -> str:
        hour, x = divmod(x, 3600)
        minute, x = divmod(x, 60)
        second, x = divmod(x, 1)
        millisecond = int(x * 1000.)

        return '%.2d:%.2d:%.2d,%.3d' % (hour, minute, second, millisecond)

    with open("asrt.txt", "w", encoding="utf-8") as file:
        # file.write("num\tstart_time\tend_time\tword\n")

        for result in result.results:
            alternative = result.alternatives[0]

            for word_info in alternative.words:
                
                    speech_recognition = word_info.word
                    start_time = word_info.start_time
                    start_time = second_to_timecode(start_time.total_seconds())
                    end_time = word_info.end_time
                    end_time = second_to_timecode(end_time.total_seconds())
                    
                    # print(start_time, end_time, word)
                    file.write(f"{num}\t{speech_recognition}\t{start_time}\t{end_time}\n")
                    # file.write(f"{num}\t{start_time.total_seconds()} {end_time.total_seconds()}\t{speech_recognition}\n")
                    #total seconds give in secs values else it gives in vtt format which is good for html
                    num += 1
        print("done")


transcribe_speech()

import pandas as pd

with open("asrt.txt", "r", encoding="utf-8") as file:
    text = file.read()
    updated_text = text.replace('ي', 'ى')

with open("asrt.txt", "w", encoding="utf-8") as file:
    file.write(updated_text)
    
auzbillah = False
bismillah = False
combined_words = ""

with open("asrt.txt", "r", encoding="utf-8") as file:
    lines = file.readlines()
    if lines[0].split('\t')[1] == "اعوذ":
        auzbillah = True
    if lines[5].split('\t')[1] == "بسم":
        bismillah = True

    if auzbillah and bismillah:
        for i in range(8, 15):  # Adjusted the range to include line 14
            combined_words += lines[i].split('\t')[1] + " "
        first_six_words = combined_words.split()[-6:]
    elif auzbillah or bismillah:
        for i in range(4, 11):  # Adjusted the range to include line 10
            combined_words += lines[i].split('\t')[1] + " "
        first_six_words = combined_words.split()[-6:]
    else:
        for i in range(0, 6):
            combined_words += lines[i].split('\t')[1] + " "
        first_six_words = combined_words.split()[-6:]

    last_six_words = ""
    for i in range(len(lines) - 6, len(lines)):
        last_six_words += lines[i].split('\t')[1] + " "

    # Remove the extra space at the end of the string
    last_six_words = last_six_words.strip()

print("auzbillah value:", auzbillah)
print("bismillah value:", bismillah)
print("First six words:", ' '.join(first_six_words))
print("Last six words:", last_six_words)

# Read the Excel file
df = pd.read_excel('DATABASE.xlsx')

# Exclude rows with missing values in the 'ArabicText' column
df = df.dropna(subset=['ArabicText'])

# Search for the word 'first_six_words' in the 'ArabicText' column
result = df[df['ArabicText'].str.contains(' '.join(first_six_words), na=False)]
result2 = df[df['ArabicText'].str.contains(last_six_words, na=False)]

# Print the corresponding SurahNo and AyahNo values
if not result.empty:
    for index, row in result.iterrows():
        print(f"SurahNo: {row['SurahNo']}, AyahNo: {row['AyahNo']}")
else:
    print("Not found: Ayah start")

if not result2.empty:
    for index, row in result2.iterrows():
        print(f"SurahNo: {row['SurahNo']}, AyahNo: {row['AyahNo']}")
else:
    print("Not found: Ayah end")

import pandas as pd
import csv

# Load the Excel file
wb = pd.read_excel('DATABASE.xlsx')
ws = wb

# Get user inputs
surah_no = int(input("Enter SurahNo: "))
ayah_start = int(input("Enter AyahNo start: "))
ayah_end = int(input("Enter AyahNo end: "))

# Set default values for adding extra strings at the start of filtered_data
add_auzbillah = True
add_bismillah = True

# Give the user the option to change the default values
change_auzbillah = input("Enter 'F' to exclude 'Auzbillah' from the start of filtered data: ")
if change_auzbillah == "F":
    add_auzbillah = False

change_bismillah = input("Enter 'F' to exclude 'Bismillah' from the start of filtered data: ")
if change_bismillah == "F":
    add_bismillah = False

# Filter the data for original Arabic
filtered_data_Orignal_arabic = []
filtered_data_arabic_text = []
for row in ws.itertuples():
    if row[5] == surah_no and ayah_start <= row[11] <= ayah_end:
        ayah_no = row[11]  # Get the AyahNo from column K
        filtered_data_Orignal_arabic.append(row[13] + "" + str(ayah_no))
        filtered_data_arabic_text.append(row[14])

# Add extra strings at the start of filtered_data
if add_auzbillah and add_bismillah:
    filtered_data_Orignal_arabic.insert(0, "أَعُوْذُ بِاللّٰهِ مِنَ الشَّيْطٰانِ الرَّجِيْمِ# " + "بِسۡمِ ٱللَّهِ ٱلرَّحۡمَٰنِ ٱلرَّحِيمِ% ")
    filtered_data_arabic_text.insert(0, "اعوذ بالله من الشيطان الرجيم "+ "بسم الله الرحمن الرحيم ")
elif add_auzbillah:
    filtered_data_Orignal_arabic.insert(0, "أَعُوْذُ بِاللّٰهِ مِنَ الشَّيْطٰانِ الرَّجِيْمِ#")
    filtered_data_arabic_text.insert(0, "اعوذ بالله من الشيطان الرجيم")
elif add_bismillah:
    filtered_data_Orignal_arabic.insert(0, "بِسۡمِ ٱللَّهِ ٱلرَّحۡمَٰنِ ٱلرَّحِيمِ%")
    filtered_data_arabic_text.insert(0, "بسم الله الرحمن الرحيم")

# Define the characters to exclude from splitting
exclusion_chars = ['ً', 'ۚ', 'ۗ', 'ۖ', 'ٌ', 'ٍ', 'َ', 'ُ', 'ِ', 'ّ', 'ْ', 'ٰ', '۩', 'ۙ', 'ۜ',' ۜ' , '1', '2', '3', '4', '5', '6', '7', '8', '9','ۜ1']


# Split the words and remove empty values from the list
Ayahs_oa = ' '.join(filtered_data_Orignal_arabic)
words = Ayahs_oa.split(" ")
split_words_oa = [words[0]]
for i in range(1, len(words)):
    if words[i] in exclusion_chars:
        split_words_oa[-1] += words[i]
    else:
        split_words_oa.append(words[i])
split_words_oa = list(filter(None, split_words_oa))

Ayahs_at = ' '.join(filtered_data_arabic_text)
split_words_at = Ayahs_at.split()

# Load the additional data from "asrt.txt"
asrt_data = []
with open("asrt.txt", "r", encoding="utf-8") as asrt_file:
    reader = csv.reader(asrt_file, delimiter='\t')
    for row in reader:
        if len(row) >= 4:
            asrt_data.append(row)

# print("Filtered Data Original Arabic:", filtered_data_Orignal_arabic)
# print("Filtered Data Arabic Text:", filtered_data_arabic_text)
# print("Split Words Original Arabic:", split_words_oa)
# print("Split Words Arabic Text:", split_words_at)
# print("ASRT Data:", asrt_data)
print("search and matching algorithim starts")


# Create a DataFrame to store the data
output_data = {
    "Index": [],
    "Original Arabic": [],
    "Arabic text": [],
    "Speech Recognition": [],
    "Start Time": [],
    "End Time": []
}

# Find the maximum length
max_length = max(len(asrt_data), len(split_words_oa), len(split_words_at))

# Populate the DataFrame with the data
for i in range(max_length):
    output_data["Index"].append(i)

    if i < len(asrt_data):
        output_data["Speech Recognition"].append(asrt_data[i][1])
        output_data["Start Time"].append(asrt_data[i][2])
        output_data["End Time"].append(asrt_data[i][3])
    else:
        output_data["Speech Recognition"].append(None)
        output_data["Start Time"].append(None)
        output_data["End Time"].append(None)

    if i < len(split_words_oa):
        output_data["Original Arabic"].append(split_words_oa[i])
    else:
        output_data["Original Arabic"].append(None)

    if i < len(split_words_at):
        output_data["Arabic text"].append(split_words_at[i])
    else:
        output_data["Arabic text"].append(None)


# Create a pandas DataFrame
output_df = pd.DataFrame(output_data)

# print("Output DataFrame:")
# print(output_df)

# Export the DataFrame to Excel
output_df.to_excel("search_match.xlsx", index=False)
print("search matching file created")


import pandas as pd

# read data from Excel file
df = pd.read_excel('search_match.xlsx')

# count the number of non-null cells in the 'Arabic text' column
at_count = df['Arabic text'].count()

# count the number of non-null cells in the 'Speech Recognition' column
sr_count = df['Speech Recognition'].count()

# calculate the difference between the counts
diff = sr_count - at_count if sr_count > at_count else at_count - sr_count

print(diff)

# create a list to store the modified data
matched_data = []

const=2
# iterate over the rows of the data
i = 0
while i < len(df):
    # check if Speech Recognition matches Arabic text
    if df.loc[i, 'Speech Recognition'] == df.loc[i, 'Arabic text']:
        # extract start and End Times of the matched word
        start_time = df.loc[i, 'Start Time']
        end_time = df.loc[i, 'End Time']
        matched_data.append((i, start_time, end_time))
    else:
        # try the next and previous 10 rows
        match_found = False
        for j in range(max(0, i-(const+diff)), min(i+(const+diff), len(df))):
            if df.loc[i, 'Speech Recognition'] == df.loc[j, 'Arabic text']:
                # extract start and End Times of the matched word
                start_time = df.loc[i, 'Start Time']
                end_time = df.loc[i, 'End Time']
                matched_data.append((j, start_time, end_time))
                match_found = True
                break
        if not match_found:
            matched_data.append((-1, df.loc[i, 'Start Time'], df.loc[i, 'End Time']))  # no match found after searching next and previous 10 rows

    i += 1

# Replace consecutive -1 values using the +1 approach and include start_time and end_time
data = [x[0] for x in matched_data]
start_times = [x[1] for x in matched_data]
end_times = [x[2] for x in matched_data]
i = 0
while i < len(data):
    if data[i] == -1:
        above = data[i-1] if i > 0 else None
        below = None
        j = i + 1
        while j < len(data):
            if data[j] != -1:
                below = data[j]
                break
            j += 1

        if above is not None and below is not None and (below - above) <= 2:
            if below - above == 1:
                data[i] = above + 1
                start_times[i] = df.loc[above, 'Start Time']
                end_times[i] = df.loc[above, 'End Time']
            else:
                data[i] = above + 1
                start_times[i] = df.loc[above, 'Start Time']
                end_times[i] = df.loc[above, 'End Time']
                data[i+1] = above + 2
                start_times[i+1] = df.loc[above, 'Start Time']
                end_times[i+1] = df.loc[above, 'End Time']
        else:
            i += 1
    i += 1

# map indices and time information to original Arabic column
df['matched_index'] = data
df['start_time'] = start_times
df['end_time'] = end_times
df['matched_orignal_Arabic'] = ['' if x == -1 else df.loc[x, 'Original Arabic'] for x in data]

# create a column indicating if a word was matched or not
df['matched'] = ['Matched' if x != -1 else 'Not matched' for x in data]


# Save the updated DataFrame to a new Excel file
df.to_excel('results of matching.xlsx', index=False)

print("results of matching created")
print("srt creation begiens")


# Define function to convert numbers to Arabic
def convert_to_arabic_number(match):
    arabic_numerals = ["٠", "١", "٢", "٣", "٤", "٥", "٦", "٧", "٨", "٩"]
    number = match.group()
    return ''.join(arabic_numerals[int(digit)] for digit in number)

import re

num=1
srt_content = ""
for start_time, end_time, matched_text in zip(start_times, end_times, df['matched_orignal_Arabic']):
    # Convert all numbers in the matched_text to Arabic numerals
    matched_text_arabic = re.sub(r'\d+', lambda m: convert_to_arabic_number(m), matched_text)
    srt_content += f"{num}\n{start_time} --> {end_time}\n{matched_text_arabic}\n\n"
    num += 1

# Save the SRT content to a file
with open('results.srt', 'w', encoding='utf-8') as f:
    f.write(srt_content)

print("SRT WORD BY WORD CREATED")
print("Full ayah creation with 100% chances of accuracy")

import pandas as pd
import re

# Specify the corrected file path to your Excel file
file_path = "results of matching.xlsx"

# Specify the columns you want to load
columns = ["matched_index", "start_time", "end_time", "matched_orignal_Arabic"]

# Read the specified columns from the Excel file into a pandas DataFrame
data = pd.read_excel(file_path, usecols=columns)

# Load the Excel file
wb = pd.read_excel('DATABASE.xlsx')
ws = wb

# Define function to convert numbers to Arabic
def convert_to_arabic_number(match):
    arabic_numerals = ["٠", "١", "٢", "٣", "٤", "٥", "٦", "٧", "٨", "٩"]
    number = match.group()
    return ''.join(arabic_numerals[int(digit)] for digit in number)

# Get user input
surah_no = surah_no

# Initialize variables
combined_text = ""
start_index = 0
start_time = data.at[0, 'start_time'] 
use_numbering = True  # Set to True to include numbering
filtered_ayah_data = []  # Store the filtered ayah data

# Check if numbering is enabled
if use_numbering:
    numb = 1  # Initialize the counter variable

# Iterate over the rows of the DataFrame
for row in data.itertuples(index=True):
    matched_index, curr_start_time, end_time, matched_orignal_Arabic = row[1:]

    matched_orignal_Arabic = str(matched_orignal_Arabic)  # Convert to string if not already

    if matched_orignal_Arabic == "#":
        filtered_ayah_data.append("أَعُوذُ بِاللّٰهِ مِنَ الشَّيْطٰانِ الرَّجِيْمِ")
        start_time = None  # Reset start_time
    elif matched_orignal_Arabic == "%":
        filtered_ayah_data.append("بِسۡمِ ٱللَّهِ ٱلرَّحۡمَٰنِ ٱلرَّحِيمِ")
        start_time = None  # Reset start_time
    else:
        AyahNO = re.findall(r"\d+", matched_orignal_Arabic)
        if AyahNO:
            AyahNO = AyahNO[0]
            # Filter the data for original Arabic
            for inner_row in ws.itertuples():
                if inner_row[5] == surah_no and inner_row[11] == int(AyahNO):
                    ayah_text = inner_row[13]
                    ayah_no = inner_row[11]  # Get the AyahNo from column K
                    ayah_no_arabic = convert_to_arabic_number(re.match('\d+', str(ayah_no)))
                    
                    # Convert all numbers in the text to Arabic numerals
                    ayah_text_arabic = re.sub(r'\d+', lambda m: convert_to_arabic_number(m), ayah_text)
                    
                    filtered_ayah_data.append(ayah_text_arabic + str(ayah_no_arabic))
                    break  # Exit the inner loop once the data is found
        else:
            # Handle the case when AyahNO is not found
            if "#" in matched_orignal_Arabic:
                filtered_ayah_data.append("أَعُوذُ بِاللّٰهِ مِنَ الشَّيْطٰانِ الرَّجِيْمِ")
            elif "%" in matched_orignal_Arabic:
                filtered_ayah_data.append("بِسۡمِ ٱللَّهِ ٱلرَّحۡمَٰنِ ٱلرَّحِيمِ")
            else:
                continue  # Skip to the next iteration

    # Combine the words from the start_index to the current index
    if start_index is not None:
        if use_numbering:
            combined_text += str(numb) + "\n"
            numb += 1
        if start_time is None:
            start_time = curr_start_time
        combined_text += str(start_time) + " --> " + str(end_time) + "\n"
        combined_text += filtered_ayah_data[-1] + "\n\n"

    # Update the start_index and start_time for the next combination
    start_index = row.Index + 1
    start_time = end_time  # Set the start_time to the end_time of the last combination

    # Print the AyahNO
    print("AyahNO:", AyahNO)

# Specify the file path for the text file
output_file_path = "Full Ayah.srt"

# Save the combined text to a text file with UTF-8 encoding
with open(output_file_path, "w", encoding="utf-8") as file:
    file.write(combined_text)

print("Text saved to", output_file_path)



import pandas as pd
import re

# Specify the corrected file path to your Excel file
file_path = "results of matching.xlsx"

# Specify the columns you want to load
columns = ["matched_index", "start_time", "end_time", "matched_orignal_Arabic"]

# Read the specified columns from the Excel file into a pandas DataFrame
data = pd.read_excel(file_path, usecols=columns)

# Read the Translations.xlsx file and load all columns
translations_df = pd.read_excel("Translations.xlsx")

# Get user input
surah_no = surah_no
# Display available translation options with numbers
translation_options = translations_df.columns[2:]
print("Available translations:")
for i, option in enumerate(translation_options):
    print(f"{i+1}. {option}")

# Get user's choice of translations
translation_numbers = input("Enter the translation numbers you want (comma-separated): ")
translation_numbers = [int(num.strip()) for num in translation_numbers.split(",")]

# Validate and assign chosen translations
translations = [translation_options[num - 1] for num in translation_numbers]
translation1, translation2 = translations[0], translations[1]

# Initialize variables
combined_text = ""
start_index = 0
start_time = data.iloc[0]['start_time']  # Initialize start_time variable with first row's start_time
use_numbering = True  # Set to True to include numbering
filtered_ayah_data = []  # Store the filtered ayah data

# Check if numbering is enabled
if use_numbering:
    numb = 1  # Initialize the counter variable

# Iterate over the rows of the DataFrame
for row in data.itertuples(index=True):
    matched_index, curr_start_time, end_time, matched_orignal_Arabic = row[1:]

    matched_orignal_Arabic = str(matched_orignal_Arabic)  # Convert to string if not already

    if "#" in matched_orignal_Arabic:
        filtered_ayah_data.append(translations_df.loc[0, translation1] + "\n" + translations_df.loc[0, translation2])

    if "%" in matched_orignal_Arabic:
        filtered_ayah_data.append(translations_df.loc[2, translation1] + "\n" + translations_df.loc[2, translation2])

    else:
        AyahNO = re.findall(r"\d+", matched_orignal_Arabic)
        if AyahNO:
            AyahNO = AyahNO[0]
            # Filter the data for original Arabic and the selected translations
            filtered_translations = translations_df[
                (translations_df["SurahNO"] == surah_no) & (translations_df["AyahNO"] == int(AyahNO))
            ][[translation1, translation2]]
            # Concatenate the translations and add newline characters
            translations = filtered_translations.astype(str).values.flatten()
            translations = "\n".join(translations)
            filtered_ayah_data.append(translations + "\n")
        else:
            # Handle the case when AyahNO is not found
            if "#" in matched_orignal_Arabic:
                filtered_ayah_data.append(translations_df.loc[0, translation1] + "\n" + translations_df.loc[0, translation2])
            elif "%" in matched_orignal_Arabic:
                filtered_ayah_data.append(translations_df.loc[2, translation1] + "\n" + translations_df.loc[2, translation2])
            else:
                continue  # Skip to the next iteration

    # Combine the words from the start_index to the current index
    if start_index is not None:
        if use_numbering:
            combined_text += str(numb) + "\n"
            numb += 1
        combined_text += str(start_time) + " --> " + str(end_time) + "\n"
        combined_text += "".join(filtered_ayah_data[-1]) + "\n"

    # Update the start_index and start_time for the next combination
    start_index = row.Index + 1
    start_time = end_time  # Set the start_time to the end_time of the last combination

    # Print the AyahNO
    print("AyahNO:", AyahNO)

# Specify the file path for the text file
output_file_path = "translations.srt"

# Save the combined text to a text file with UTF-8 encoding
with open(output_file_path, "w", encoding="utf-8") as file:
    file.write(combined_text)

print("Translations saved to", output_file_path)

print("video creation Begiens")

from moviepy.editor import VideoFileClip, AudioFileClip, TextClip, CompositeVideoClip
import re
import pysrt

def time_to_seconds(time_obj):
    return time_obj.hours * 3600 + time_obj.minutes * 60 + time_obj.seconds + time_obj.milliseconds / 1000

def create_text_clip(subtitle, videosize, fontsize=50, font='hafs.otf', color='white'):
    start_time = time_to_seconds(subtitle.start)
    end_time = time_to_seconds(subtitle.end)
    duration = end_time - start_time

    video_width, video_height = videosize

    text_clip = TextClip(subtitle.text, fontsize=fontsize, font=font, color=color,
                         bg_color='transparent', size=(video_width * 3 / 4, None), method='caption').set_start(start_time).set_duration(duration)

    text_clip = text_clip.set_position('center')

    return text_clip

# Load the video clip with specified encoding
video = VideoFileClip("bg.mp4")

# Set the resolution to 720p (1280x720)
video = video.resize((1920, 1080))

# Load subtitles
subtitles = pysrt.open("Full Ayah.srt")

# Create first subtitle clip
first_subtitle = subtitles[0]
first_text_clip = create_text_clip(first_subtitle, video.size)

# Add subtitle to the video
video_with_subtitle = CompositeVideoClip([video, first_text_clip])

# Save the first frame where subtitles appear
video_with_subtitle.save_frame("frame.png", t=time_to_seconds(first_subtitle.start))

# Ask the user whether to export the video
export_video = input("Do you want to export the video? (yes/no): ")

if export_video.lower() == 'y':
    # Load the audio clip with specified encoding
    audio = AudioFileClip("kahaf.mp3")

    # Get the duration of the audio clip
    audio_duration = audio.duration

    # Trim or extend the video to match the duration of the audio
    video = video.subclip(0, audio_duration)

    # Combine video with audio
    video = video.set_audio(audio)

    # Create subtitle clips
    subtitle_clips = [create_text_clip(subtitle, video.size) for subtitle in subtitles]

    # Add subtitles to the video
    final_video = CompositeVideoClip([video] + subtitle_clips)

    # Write output video file with audio duration
    output_video_file = "output_video.mp4"
    final_video.set_duration(audio_duration).write_videofile(output_video_file, codec='libx264', audio_codec='aac', fps=video.fps)
