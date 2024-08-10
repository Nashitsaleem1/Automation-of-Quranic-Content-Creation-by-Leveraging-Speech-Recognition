# Automation-of-Quranic-Content-Creation-by-Leveraging-Speech-Recognition
Developed advanced algorithms, Automated the process of video creation by eliminating the manual editing based workflows reducing creation time from 5 hours to 5 min.

# Speech-to-Text Transcription and Matching Code

The provided code is written in Python and utilizes the Google Cloud Speech-to-Text API for transcribing speech and matching it with Arabic text. The code consists of several modules and functions to perform the following tasks:

## 1. Transcribing Speech

The initial part of the code deals with transcribing speech using the Google Cloud Speech-to-Text API. Here are the main steps involved:

### Importing Required Libraries
- `speech_recognition` library is imported as `sr` to recognize speech input.
- `numpy` library is imported as `np` to handle numerical operations.
- `google.cloud.speech` is imported to interact with the Speech-to-Text API.
- `os` library is imported to work with operating system functionalities.

### Instantiating the Speech-to-Text Client
- The code instantiates a client object for the Speech-to-Text API using the provided service account credentials stored in the file `'quran-374914-afa6baf2bbe3.json'`.

### Defining the Audio Source
- The code defines the audio source, which can be either a local file or a Google Cloud Storage (GCS) URI.
- Currently, the code is set to use a GCS URI `'gs://quran_mulk/audio-files/kahaf.flac'`.
- Alternative options for audio sources are provided but commented out.

### Configuring the Speech Recognition
- The code configures the speech recognition settings, including the encoding, sample rate, language code, model, audio channel count, and enabling word time offsets.

### Performing Speech Recognition
- The code initiates the speech recognition operation by calling the `long_running_recognize` method of the Speech-to-Text client.
- The operation is performed asynchronously, and the code waits for the operation to complete.
- The resulting transcription is stored in the `result` variable.

### Parsing Transcription Results
- The code extracts relevant information from the transcription results, including the recognized words, their start and end times, and assigns them a numerical index.
- The extracted information is written to a file named `"asrt.txt"`.

## 2. Matching Speech with Arabic Text

The second part of the code focuses on matching the transcribed speech with the corresponding Arabic text. Here are the main steps involved:

### Importing Required Libraries
- `pandas` library is imported as `pd` to handle data manipulation and analysis.
- `csv` library is imported to work with CSV files.

### Loading the Arabic Text Data
- The code loads the Arabic text data from an Excel file named `'DATABASE.xlsx'` using the `pd.read_excel` function.
- The loaded data is assigned to the variable `wb`.

### User Input for Filtering
- The code prompts the user to enter the Surah number, starting Ayah number, and ending Ayah number for filtering the data.

### Setting Default Values for Extra Strings
- The code sets default values for adding extra strings at the start of the filtered data.
- The default values for adding "Auzbillah" and "Bismillah" are set to `True`.

### User Input for Modifying Default Values
- The code provides the user with the option to change the default values for adding "Auzbillah" and "Bismillah" at the start of the filtered data.
- If the user enters 'F', the corresponding default value is set to `False`.

### Filtering the Arabic Text Data
- The code filters the Arabic text data based on the provided Surah number and Ayah range.
- The filtered data is stored in two separate lists: `filtered_data_Orignal_arabic` and `filtered_data_arabic_text`.

### Adding Extra Strings to Filtered Data
- If the user has not modified the default values for adding "Auzbillah" and "Bismillah", the code adds these strings at the start of the filtered data.
- The code checks the default values and adds the corresponding strings if necessary.

### Matching Transcribed Speech with Arabic Text
- The code compares each word in the transcribed speech with the words in the filtered Arabic text data.
- If a match is found, the code stores the corresponding Arabic text and its index in a dictionary.

### Writing Matching Results to File
- The code writes the matching results to a CSV file named `"matching_results.csv"`.
- The CSV file includes the following columns: Index, Start Time, End Time, Speech, Arabic Text.

## Overall Function
The code is wrapped in a function called `transcribe_and_match_speech()` for ease of use. The function takes no arguments and performs the entire process of speech transcription and matching.

To use this code, you need to have the necessary dependencies installed and provide the required inputs, such as the service account credentials file, audio source, Surah number, and Ayah range. Additionally, you may need to modify the code to suit your specific requirements, such as changing the audio source or default values for adding extra strings.

Please note that the provided code is a basic implementation and may require further customization based on your specific use case and data format.


#

# Speech Recognition and Matching Code Explanation

This code is used for speech recognition and matching of Arabic text. It involves the following steps:

1. Importing the necessary libraries:
   ```python
   import speech_recognition as sr
   import numpy as np
   from google.cloud import speech
   import os
   import pandas as pd
   import csv
   ```

2. Setting up the Google Cloud Speech-to-Text client:
   ```python
   from google.cloud import speech

   client = speech.SpeechClient.from_service_account_file('quran-374914-afa6baf2bbe3.json')
   ```

3. Defining the audio file or URI for speech recognition:
   ```python
   gcs_uri = "gs://quran_mulk/audio-files/kahaf.flac"
   ```

4. Implementing the `transcribe_speech()` function for speech recognition:
   ```python
   def transcribe_speech():
       # Configure the recognition audio and settings
       audio = speech.RecognitionAudio(uri=gcs_uri)
       config = speech.RecognitionConfig(
           encoding=speech.RecognitionConfig.AudioEncoding.FLAC,
           sample_rate_hertz=44100,
           language_code="ar-SA",
           model="default",
           audio_channel_count=2,
           enable_word_time_offsets=True,
       )

       # Perform speech recognition using long running operation
       operation = client.long_running_recognize(config=config, audio=audio)
       result = operation.result(timeout=90)

       # Extract and save the recognized speech data
       num = 1
       with open("asrt.txt", "w", encoding="utf-8") as file:
           for result in result.results:
               alternative = result.alternatives[0]
               for word_info in alternative.words:
                   speech_recognition = word_info.word
                   start_time = word_info.start_time
                   start_time = second_to_timecode(start_time.total_seconds())
                   end_time = word_info.end_time
                   end_time = second_to_timecode(end_time.total_seconds())
                   file.write(f"{num}\t{speech_recognition}\t{start_time}\t{end_time}\n")
                   num += 1
       print("Speech recognition done.")
   ```
# 
# database and matching

This code is designed to process Quranic verses stored in an Excel file. It allows the user to specify criteria such as Surah number and Ayah range to filter the data. The code then performs several operations on the filtered verses, including adding optional strings, splitting the verses into individual words, and incorporating supplementary data from a separate file. The processed data is organized and stored in a pandas DataFrame, which is subsequently exported to an Excel file. In summary, this code enables the extraction, manipulation, and organization of Quranic verses based on user-defined criteria, providing a convenient way to work with and analyze the data.


```python
import pandas as pd
import csv
```
The code begins by importing the necessary libraries - `pandas` for data manipulation and `csv` for handling CSV files.

```python
wb = pd.read_excel('DATABASE.xlsx')
ws = wb
```
The code loads the data from the 'DATABASE.xlsx' Excel file into a pandas DataFrame called `wb` and assigns it to `ws` as well. Essentially, both `wb` and `ws` will point to the same DataFrame.

```python
surah_no = int(input("Enter SurahNo: "))
ayah_start = int(input("Enter AyahNo start: "))
ayah_end = int(input("Enter AyahNo end: "))
```
The code prompts the user to enter the Surah number (`surah_no`), the starting Ayah number (`ayah_start`), and the ending Ayah number (`ayah_end`) through standard input (keyboard). The input values are converted to integers.

```python
add_auzbillah = True
add_bismillah = True
```
Two boolean variables `add_auzbillah` and `add_bismillah` are initialized to `True` as default values. These variables are used to control whether additional strings 'أَعُوْذُ بِاللّٰهِ مِنَ الشَّيْطٰانِ الرَّجِيْمِ' and 'بِسۡمِ ٱللَّهِ ٱلرَّحۡمَٰنِ ٱلرَّحِيمِ' are added at the start of filtered data.

```python
change_auzbillah = input("Enter 'F' to exclude 'Auzbillah' from the start of filtered data: ")
if change_auzbillah == "F":
    add_auzbillah = False

change_bismillah = input("Enter 'F' to exclude 'Bismillah' from the start of filtered data: ")
if change_bismillah == "F":
    add_bismillah = False
```
The code gives the user the option to change the default values for `add_auzbillah` and `add_bismillah`. If the user enters 'F' for either prompt, the respective variable is set to `False`, indicating that the corresponding string should not be added to the start of the filtered data.

```python
filtered_data_Orignal_arabic = []
filtered_data_arabic_text = []
for row in ws.itertuples():
    if row[5] == surah_no and ayah_start <= row[11] <= ayah_end:
        ayah_no = row[11]  # Get the AyahNo from column K
        filtered_data_Orignal_arabic.append(row[13] + "" + str(ayah_no))
        filtered_data_arabic_text.append(row[14])
```
The code filters the original Arabic text and Arabic text data based on the provided Surah and Ayah range. It iterates through the rows of the DataFrame `ws` using `itertuples()`. If the row's Surah number (at index 5) matches `surah_no` and the Ayah number (at index 11)

(index 5 and 11 is a way of telling the code form which coloum you have to filer the data)

 is within the specified range, the Arabic text and Arabic text with Ayah number are appended to the `filtered_data_Orignal_arabic` and `filtered_data_arabic_text` lists, respectively.

```python
if add_auzbillah and add

_bismillah:
    filtered_data_Orignal_arabic.insert(0, "أَعُوْذُ بِاللّٰهِ مِنَ الشَّيْطٰانِ الرَّجِيْمِ# " + "بِسۡمِ ٱللَّهِ ٱلرَّحۡمَٰنِ ٱلرَّحِيمِ% ")
    filtered_data_arabic_text.insert(0, "اعوذ بالله من الشيطان الرجيم "+ "بسم الله الرحمن الرحيم ")
elif add_auzbillah:
    filtered_data_Orignal_arabic.insert(0, "أَعُوْذُ بِاللّٰهِ مِنَ الشَّيْطٰانِ الرَّجِيْمِ#")
    filtered_data_arabic_text.insert(0, "اعوذ بالله من الشيطان الرجيم")
elif add_bismillah:
    filtered_data_Orignal_arabic.insert(0, "بِسۡمِ ٱللَّهِ ٱلرَّحۡمَٰنِ ٱلرَّحِيمِ%")
    filtered_data_arabic_text.insert(0, "بسم الله الرحمن الرحيم")
```
Based on the values of `add_auzbillah` and `add_bismillah`, additional strings are inserted at the start of the `filtered_data_Orignal_arabic` and `filtered_data_arabic_text` lists. The strings are added in Arabic and English transliteration.

```python
exclusion_chars = ['ً', 'ۚ', 'ۗ', 'ۖ', 'ٌ', 'ٍ', 'َ', 'ُ', 'ِ', 'ّ', 'ْ', 'ٰ', '۩', 'ۙ', 'ۜ',' ۜ' , '1', '2', '3', '4', '5', '6', '7', '8', '9','ۜ1']
```
A list called `exclusion_chars` is defined, which contains characters that should be excluded when splitting the words later on.

the reason is that to make the arabic text and orignal arabic text indexes same so that we can map the arbic text to orignal arabic text when we will be doing matching and then maping.

```python
Ayahs_oa = ' '.join(filtered_data_Orignal_arabic)
words = Ayahs_oa.split(" ")
split_words_oa = [words[0]]
for i in range(1, len(words)):
    if words[i] in exclusion_chars:
        split_words_oa[-1] += words[i]
    else:
        split_words_oa.append(words[i])
split_words_oa = list(filter(None, split_words_oa))
```
The code joins the filtered original Arabic data into a single string separated by spaces and then splits it into words based on space characters. The resulting words are stored in the `split_words_oa` list. If a word contains a character in `exclusion_chars`, it is appended to the previous word. Finally, empty words are filtered out using `filter(None, split_words_oa)`.

```python
Ayahs_at = ' '.join(filtered_data_arabic_text)
split_words_at = Ayahs_at.split()
```
Similarly, the code joins the filtered Arabic text data into a single string and splits it into words. The resulting words are stored

 in the `split_words_at` list.

```python
asrt_data = []
with open("asrt.txt", "r", encoding="utf-8") as asrt_file:
    reader = csv.reader(asrt_file, delimiter='\t')
    for row in reader:
        if len(row) >= 4:
            asrt_data.append(row)
```
The code reads the data from the "asrt.txt" file, which is assumed to be a tab-separated values (TSV) file. Each row is appended to the `asrt_data` list if it contains at least four elements.

```python
output_data = {
    "Index": [],
    "Original Arabic": [],
    "Arabic text": [],
    "Speech Recognition": [],
    "Start Time": [],
    "End Time": []
}
```
A dictionary called `output_data` is created with keys representing the column names and empty lists as initial values.

```python
max_length = max(len(asrt_data), len(split_words_oa), len(split_words_at))
```
The code determines the maximum length among `asrt_data`, `split_words_oa`, and `split_words_at` lists. This will be used to iterate and populate the `output_data` dictionary.

```python
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
```
The code populates the `output_data` dictionary by iterating from 0 to `max_length`. For each index, the corresponding values are appended to the respective lists in the dictionary. If the index exceeds the length of a particular list, `None` is appended as a placeholder.

```python
output_df = pd.DataFrame(output_data)
```
A pandas DataFrame called `output_df` is created using the `output_data` dictionary.

```python
output_df.to_excel("search_match.xlsx", index=False)
print("search matching file created")
```
The DataFrame `output_df` is exported to an Excel file named "search_match.xlsx" without including the index column. Finally, a message indicating the successful creation of the search matching file is printed.




#
# Explaination of matching

In this part, a pandas DataFrame named `output_df` is created using the `output_data` dictionary. The DataFrame is then exported to an Excel file named `"search_match.xlsx"` using the `to_excel()` function. The `index=False` argument ensures that the row index is not included in the exported Excel file. Finally, a message is printed indicating that the search matching file has been created.

The rest of the code appears to be performing additional operations on the generated data, such as counting non-null cells, matching Speech Recognition and Arabic text, handling consecutive -1 values, mapping indices and time information, and saving the updated DataFrame and SRT content to files.

# Detaied explaination


```python
import pandas as pd
```
This line imports the `pandas` library, which is used for data manipulation and analysis.

```python
df = pd.read_excel('search_match.xlsx')
```
This line reads the data from an Excel file named `'search_match.xlsx'` and stores it in a DataFrame called `df`.

```python
at_count = df['Arabic text'].count()
sr_count = df['Speech Recognition'].count()
```
These lines count the number of non-null cells in the columns `'Arabic text'` and `'Speech Recognition'` of the DataFrame `df` and store the counts in the variables `at_count` and `sr_count`, respectively.

```python
diff = sr_count - at_count if sr_count > at_count else at_count - sr_count
```
This line calculates the difference between `sr_count` and `at_count`. It checks if `sr_count` is greater than `at_count` and subtracts `at_count` from `sr_count` if it is, otherwise, it subtracts `sr_count` from `at_count`. The result is stored in the variable `diff`.

```python
matched_data = []
const = 2
```
These lines initialize an empty list `matched_data` to store the modified data and set the value of `const` as 2. `const` is used to define the number of rows to search before and after the current row for a match.

```python
i = 0
while i < len(df):
    if df.loc[i, 'Speech Recognition'] == df.loc[i, 'Arabic text']:
        start_time = df.loc[i, 'Start Time']
        end_time = df.loc[i, 'End Time']
        matched_data.append((i, start_time, end_time))
    else:
        match_found = False
        for j in range(max(0, i - (const + diff)), min(i + (const + diff), len(df))):
            if df.loc[i, 'Speech Recognition'] == df.loc[j, 'Arabic text']:
                start_time = df.loc[i, 'Start Time']
                end_time = df.loc[i, 'End Time']
                matched_data.append((j, start_time, end_time))
                match_found = True
                break
        if not match_found:
            matched_data.append((-1, df.loc[i, 'Start Time'], df.loc[i, 'End Time']))
    i += 1
```
This block of code iterates over the rows of the DataFrame `df` using a `while` loop. It checks if the `'Speech Recognition'` column value matches the corresponding value in the `'Arabic text'` column for each row. If a match is found, it extracts the start and end times from the `'Start Time'` and `'End Time'` columns, respectively, and appends a tuple `(i, start_time, end_time)` to the `matched_data` list. Here, `i` represents the index of the matched row.

If a match is not found, it searches for a match in the next and previous `(const + diff)` rows. If a match is found, it appends a tuple `(j, start_time, end_time)` to `matched_data`, where `j` represents the index of the matched row. If no match is found, it appends `(-1, start_time, end_time)` to indicate that no match was found after searching the neighboring rows.

```python
data = [x[0] for x in matched_data]
start_times = [x[1] for x in matched_data]
end_times

 = [x[2] for x in matched_data]
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
```
In this part, the code processes the `matched_data` list to replace consecutive `-1` values using the +1 approach. It iterates over the `data` list, which contains the indices from `matched_data`, and checks for consecutive `-1` values. If found, it looks for the nearest non-`-1` indices above and below. If the difference between them is less than or equal to 2, it replaces the `-1` values with the corresponding indices and copies the start and end times from the previous non-`-1` row.

```python
df['matched_index'] = data
df['start_time'] = start_times
df['end_time'] = end_times
df['matched_orignal_Arabic'] = ['' if x == -1 else df.loc[x, 'Original Arabic'] for x in data]
df['matched'] = ['Matched' if x != -1 else 'Not matched' for x in data]
```
These lines add new columns to the DataFrame `df` to store the matched index, start time, end time, original Arabic text, and a column indicating if a word was matched or not. The values are populated based on the `data` list.

```python
df.to_excel('results of matching.xlsx', index=False)
```
This line saves the updated DataFrame `df` to a new Excel file named `'results of matching.xlsx'` without including the index column.

```python
num = 1
srt_content = ""
for start_time, end_time, matched_text in zip(start_times, end_times, df['matched_orignal_Arabic']):
    srt_content += f"{num}\n{start_time} --> {end_time}\n{matched_text}\n\n"
    num += 1
```
This loop iterates over the `start_times`, `end_times`, and `matched_orignal_Arabic` columns of the DataFrame `df` using the `zip` function. It constructs the content for the SRT file by concatenating the line number, start time, end time, and matched text for each row.

```python
with open('results.srt', 'w', encoding='utf-8') as f:
    f.write(srt_content)
```
This code block opens a file named `'results.srt'` in write mode with UTF-8 encoding and writes the `srt_content` to the

 file. It saves the content in the SubRip Text (SRT) format, which is commonly used for subtitles.


 Certainly! Here's the complete Markdown code file:

```markdown
# Code Explanation

The provided code performs the following steps:

1. Import the necessary libraries: `pandas` and `openpyxl`.

```python
import pandas as pd
```

2. Open the file `asrt.txt` and replace occurrences of the Arabic letter 'ي' with 'ى'.

```python
with open("asrt.txt", "r", encoding="utf-8") as file:
    text = file.read()
    updated_text = text.replace('ي', 'ى')

with open("asrt.txt", "w", encoding="utf-8") as file:
    file.write(updated_text)
```

3. Initialize `auzbillah` and `bismillah` variables to `False`.

```python
auzbillah = False
bismillah = False
```

4. Read the lines of the file `asrt.txt` and store them in a list.

```python
with open("asrt.txt", "r", encoding="utf-8") as file:
    lines = file.readlines()
```

5. Check if the first line contains the string "اعوذ". If it does, set `auzbillah` to `True`.

```python
if lines[0].split('\t')[1] == "اعوذ":
    auzbillah = True
```

6. Check if the sixth line contains the string "بسم". If it does, set `bismillah` to `True`.

```python
if lines[5].split('\t')[1] == "بسم":
    bismillah = True
```

7. Based on the values of `auzbillah` and `bismillah`, construct the `first_six_words` list. If both `auzbillah` and `bismillah` are `True`, append words from lines 8 to 14 (inclusive). If either `auzbillah` or `bismillah` is `True`, append words from lines 4 to 10 (inclusive). If neither `auzbillah` nor `bismillah` is `True`, append words from lines 0 to 5 (inclusive).

```python
combined_words = ""

if auzbillah and bismillah:
    for i in range(8, 15):  # Adjusted the range to include line 14
        combined_words += lines[i].split('\t')[1] + " "
elif auzbillah or bismillah:
    for i in range(4, 11):  # Adjusted the range to include line 10
        combined_words += lines[i].split('\t')[1] + " "
else:
    for i in range(0, 6):
        combined_words += lines[i].split('\t')[1] + " "

first_six_words = combined_words.split()[-6:]
```

8. Construct the `last_six_words` string by concatenating words from the last six lines.

```python
last_six_words = ""
for i in range(len(lines) - 6, len(lines)):
    last_six_words += lines[i].split('\t')[1] + " "
```

9. Read the Excel file `DATABASE.xlsx` and store the data in a data frame.

```python
df = pd.read_excel('DATABASE.xlsx')
```

10. Remove rows with missing values in the 'ArabicText' column.

```python
df = df.dropna(subset=['ArabicText'])
```

11. Search for the string represented by `

first_six_words` in the 'ArabicText' column and store the resulting rows in `result`.

```python
result = df[df['ArabicText'].str.contains(' '.join(first_six_words), na=False)]
```

12. Search for the string represented by `last_six_words` in the 'ArabicText' column and store the resulting rows in `result2`.

```python
result2 = df[df['ArabicText'].str.contains(last_six_words, na=False)]
```

13. If `result` is not empty, iterate over its rows and print the corresponding 'SurahNo' and 'AyahNo' values.

```python
if not result.empty:
    for index, row in result.iterrows():
        print(f"SurahNo: {row['SurahNo']}, AyahNo: {row['AyahNo']}")
else:
    print("Not found: Ayah start")
```

14. If `result2` is not empty, iterate over its rows and print the corresponding 'SurahNo' and 'AyahNo' values.

```python
if not result2.empty:
    for index, row in result2.iterrows():
        print(f"SurahNo: {row['SurahNo']}, AyahNo: {row['AyahNo']}")
else:
    print("Not found: Ayah end")
```

15. Finally, print the values of `auzbillah`, `bismillah`, `first_six_words`, and `last_six_words`.

```python
print("auzbillah value:", auzbillah)
print("bismillah value:", bismillah)
print("First six words:", ' '.join(first_six_words))
print("Last six words:", last_six_words)
```

This code can be used to process the text file `asrt.txt` and search for corresponding values in the `DATABASE.xlsx` file.

Please note that this is a simplified explanation of the code. For a more detailed understanding, refer to the original code and comments provided.
```

The code provided utilizes the MoviePy library to process a video file, audio file, and subtitle file to create a final video with embedded subtitles. Here's a detailed explanation of the code:

1. The code imports necessary modules: `VideoFileClip`, `AudioFileClip`, `TextClip`, and `CompositeVideoClip` from the MoviePy library, as well as the `re` and `pysrt` modules.

2. The `time_to_seconds` function converts a time object in the subtitle file to seconds for easy calculation and manipulation.

3. The `create_subtitle_clips` function takes a list of subtitles, video size, and optional parameters such as fontsize, font, and color. It iterates through the subtitles and creates TextClip objects for each subtitle with the specified attributes. The subtitles are positioned at the center of the video. The function returns a list of subtitle clips.

4. The code loads the video file using `VideoFileClip` and resizes it to a resolution of 1280x720 pixels.

5. The audio file is loaded using `AudioFileClip`.

6. The duration of the audio clip is retrieved using the `duration` attribute.

7. The video is trimmed or extended using `subclip` to match the duration of the audio.

8. The audio is combined with the video using `set_audio` to create a final video with synchronized audio.

9. The subtitles file is opened using `pysrt.open` and the subtitles are loaded into a list.

10. The `create_subtitle_clips` function is called to generate a list of subtitle clips based on the loaded subtitles and the size of the video.

11. The `CompositeVideoClip` is used to combine the video, audio, and subtitle clips into the final video.

12. The final video's duration is set to match the audio duration using `set_duration`.

13. The output video file is specified as "output_video.mp4".

14. The final video is written to the output file using `write_videofile` with the specified codec, audio codec, and frame rate.

Overall, this code takes a video file, audio file, and subtitle file as inputs, processes them using MoviePy, and produces a new video file with embedded subtitles. It provides flexibility in customizing subtitle appearance and positioning within the video.



# Code Explanation


```python
import re
import pysrt
from moviepy.editor import VideoFileClip, AudioFileClip, TextClip, CompositeVideoClip
```
In this section, the necessary modules are imported: `re` for regular expression operations, `pysrt` for working with subtitle files, and various classes from the `moviepy.editor` module for video and audio processing.

```python
def time_to_seconds(time_obj):
    return time_obj.hours * 3600 + time_obj.minutes * 60 + time_obj.seconds + time_obj.milliseconds / 1000
```
This function `time_to_seconds` converts a time object from the subtitle file to seconds. It calculates the total time in seconds by adding up the hours, minutes, seconds, and milliseconds components of the time object.

```python
def create_subtitle_clips(subtitles, videosize, fontsize=24, font='Arial', color='yellow'):
    subtitle_clips = []
    for subtitle in subtitles:
        start_time = time_to_seconds(subtitle.start)
        end_time = time_to_seconds(subtitle.end)
        duration = end_time - start_time

        video_width, video_height = videosize

        text_clip = TextClip(subtitle.text, fontsize=60, font=font, color='white', bg_color='transparent',
                             size=(video_width * 3 / 4, None), method='caption').set_start(start_time).set_duration(duration)

        subtitle_x_position = 'center'
        subtitle_y_position = 'center'

        text_position = (subtitle_x_position, subtitle_y_position)
        subtitle_clips.append(text_clip.set_position(text_position))

    return subtitle_clips
```
This function `create_subtitle_clips` takes a list of subtitles, video size, and optional parameters such as fontsize, font, and color. It iterates through each subtitle in the list and creates a TextClip object for each subtitle. The start time and end time of the subtitle are converted to seconds using the `time_to_seconds` function. The duration is calculated as the difference between the end time and start time. The TextClip is created with the specified text, font size, font, color, background color, size, and method. The TextClip is positioned at the center of the video. The created subtitle clips are stored in a list and returned.

```python
video = VideoFileClip("video.mp4")
```
Here, the `VideoFileClip` class is used to load the video file "video.mp4" into the `video` object.

```python
video = video.resize((1280, 720))
```
This line resizes the video to a resolution of 1280x720 pixels using the `resize` method of the `VideoFileClip` class.

```python
audio = AudioFileClip("kahaf.mp3")
```
The `AudioFileClip` class is used to load the audio file "kahaf.mp3" into the `audio` object.

```python
audio_duration = audio.duration
```
The `duration` attribute of the `AudioFileClip` object is assigned to the variable `audio_duration` to retrieve the duration of the audio clip in seconds.

```python
video = video.subclip(0, audio_duration)
```
This line trims or extends the video clip using the `subclip` method of the `VideoFileClip` class. It sets the start and end times of the video to match the duration of the audio clip.

```python
video = video.set_audio(audio)
```
The `set_audio` method is used to combine the video and audio clips. It sets the audio of the video

 clip to the loaded audio clip.

```python
subtitles = pysrt.open("Full Ayah.srt")
```
The `pysrt.open` function is used to open the subtitle file "Full Ayah.srt" and load the subtitles into the `subtitles` object.

```python
subtitle_clips = create_subtitle_clips(subtitles, video.size)
```
The `create_subtitle_clips` function is called to generate a list of subtitle clips based on the loaded subtitles and the size of the video.

```python
final_video = CompositeVideoClip([video] + subtitle_clips)
```
The `CompositeVideoClip` class is used to combine the video clip, audio, and subtitle clips into the final video. The video clip is passed as the first element of a list, followed by the subtitle clips.

```python
final_video.set_duration(audio_duration).write_videofile(output_video_file, codec='libx264', audio_codec='aac', fps=video.fps)
```
The `set_duration` method is used to set the duration of the final video to match the audio duration. Then, the `write_videofile` method is called to write the final video to the file specified by `output_video_file`. The video codec is set to "libx264", the audio codec to "aac", and the frame rate is set to the same as the original video.

Overall, this code loads a video file, resizes it, loads an audio file, trims or extends the video to match the audio duration, combines the video and audio, adds subtitles from a subtitle file, and creates a final video with embedded subtitles.
