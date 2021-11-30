from glob import glob
import json

anger_paths = glob('.\\emotions\\anger\\*')
disgust_paths = glob('.\\emotions\\disgust\\*')
fear_paths = glob('.\\emotions\\fear\\*')
happiness_paths = glob('.\\emotions\\happiness\\*')
sadness_paths = glob('.\\emotions\\sadness\\*')


# print(f'{cat_paths=}')
# print(f'{dog_paths=}')
# print(f'{wild_paths=}')

# labels = []

# for i in anger_paths:
#     labels += [[i, 0]]
# for i in disgust_paths:
#     labels += [[i, 1]]
# for i in fear_paths:
#     labels += [[i, 2]]
# for i in happiness_paths:
#     labels += [[i, 3]]
# for i in sadness_paths:
#     labels += [[i, 4]]


# meta = {
#     "labels": labels
# }
# json.dump(meta, open('dataset.json', 'w'))

labels = {}
with open('.\\emotions\\dataset.json', 'r') as file:
    labels = json.load(file)['labels']
    if labels is not None:
        print('tset')
        labels = {x[0]: x[1] for x in labels}
        print(labels)
    else:
        print('dsamkio')
        labels = {}

# print(labels.get('.\\afhq\\train\\cat\\flickr_cat_000088.jpg'))
