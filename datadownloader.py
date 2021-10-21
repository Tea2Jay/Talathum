


import pandas
import urllib.request

path = r"C:/Users/3izzo/Desktop/New folder/WikiArt-Emotions/WikiArt-Emotions-All.csv"

csv = pandas.read_csv(path)
i =37
print(str(csv.columns[i]))
outFolder = r"C:/Users/3izzo/Desktop/Projects/stylegan3/images/" + str(csv.columns[i]).split(": ")[-1]
for row in csv.iloc:

    if(row[i] < .4):
        continue
    url = "https://uploads5.wikiart.org/images/%s/%s.jpg" % (row[3],row[4])
    url = url.replace(" ","-")
    print(url)
    try:
        urllib.request.urlretrieve(url, "%s/%s.jpg" % (outFolder, row[4]))
    except Exception as e:
        url = "https://uploads5.wikiart.org/images/%s/%s-%s.jpg" % (row[3],row[4],row[5])
        url = url.replace(" ","-")
        try:
            urllib.request.urlretrieve(url, "%s/%s.jpg" % (outFolder, row[4]))
        except Exception as e:
            print(e)
