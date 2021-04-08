import lyricsgenius
token = "NM2L7vKO3vF48dW69gx2sOGiJOHUZI8Y_vbFSzmgdc2GLSKBY8EnUwc1DJNd3vID"
genius = lyricsgenius.Genius(token)

def populate(name):
    artist = genius.search_artist(name, max_songs=50, sort="popularity", allow_name_change=True,
    include_features=True)
    for song in artist.songs:

        sid = song.id
        referents = genius.referents(song_id=sid, text_format='plain')

        file1 = open("double_entendres.txt", "a")

        for ref in referents['referents']:
                
                lyric = ref['fragment']
                annotation = ref['annotations'][0]['body']['plain']
                votes = ref['annotations'][0]['votes_total']
                #classification = ref['classification']

                if "double entendre" in annotation and votes > 0:

                    file1.write(lyric + "\n\n")
                    
                    print("SONG TITLE:\n")
                    print(song.title)
                    print("")
                    print("LYRICS: ")
                    print(lyric)
                    print("\nANNOTATION: ")
                    print(annotation)
                    print("\n----\n")
                
        file1.close()
                
potential_artists = []

artist = potential_artists[0]
while True:
    try:
        populate(artist)
        break
    except:
        pass


"""
import pprint
test = genius.referents(song_id=133672, text_format='plain')

for y in test['referents']:
    print("LYRICS: ")
    print(y['fragment'])
    print("\nANNOTATION: ")
    print(y['annotations'][0]['body']['plain'])
    print("\nVOTES: ")
    print(y['annotations'][0]['votes_total'])
    print("\nCLASSIFICATION: ")
    print(y['classification'])
    print("\n----\n")

    #pp = pprint.PrettyPrinter(indent=4)
    #pp.pprint(y)
    #break
"""

