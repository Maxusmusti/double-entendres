import lyricsgenius
import re

"""
Main script for retrieving lyrics from Genius
"""


token = "NM2L7vKO3vF48dW69gx2sOGiJOHUZI8Y_vbFSzmgdc2GLSKBY8EnUwc1DJNd3vID"
genius = lyricsgenius.Genius(token)


def addToText(id):
    referents = genius.referents(song_id=id, text_format='plain')

    pos = open("lyrical_positives.txt", "a")
    neg = open("lyrical_negatives.txt", "a")

    for ref in referents['referents']:
            
            lyric = ref['fragment']
            lyric = re.sub('\[.*\]', '', lyric)
            lyric = re.sub('\(.*\)', '', lyric)
            lyric = re.sub("\n", ';', lyric)

            annotation = ref['annotations'][0]['body']['plain']
            votes = ref['annotations'][0]['votes_total']

            if lyric == "" or lyric == " ":
                continue

            if "double entendre" in annotation and votes > 0:
                pos.write(lyric + "\t 1 \n")

            elif "double entendre" not in annotation:
                neg.write(lyric + "\t 0 \n")
                    
    pos.close()
    neg.close()

def generateInfo():
    page = 1
    while page:
        data = genius.tag('rap', page=page)

        file1 = open("BIGCOMMS.txt", "a")
        
        for hit in data['hits']:
            song_title = hit['title']
            main_artist = hit['artists'][0] #usually length is 1 but just in case, grab first
            file1.write(song_title + ";;" + main_artist + "\n")
        file1.close()
        print("page " + str(page) + " added")
        page = data['next_page']
        
def generateSentences():
    with open('info.txt', 'r') as file:
        data = file.read()

    text = data.split('\n')

    for line in text:
        info = line.split(';')
        title = info[0]
        name = info[1]

        song = genius.search_song(title=title, artist=name)

        if song is None:
            continue
        
        addToText(song.id)

#generateSentences()

"""
with open('lyrical_negatives.txt', 'r') as file:
    data = file.read()

text = data.split('\n')

file1 = open("lyrical_negatives_new.txt", "w")

for line in text:
    line = re.sub(';;', ';', line)
    file1.write(line + "\n")
    
file1.close()
"""
