import pandas as pd
import numpy as np
import tensorflow as tf
import tflearn
from tflearn.data_utils import to_categorical


reviews = pd.read_csv('reviews.txt', header=None)
labels = pd.read_csv('labels.txt', header=None)

from collections import Counter

total_counts = Counter()
for index, review in reviews.iterrows():
    for w in review[0].split(" "):
        total_counts[w] += 1

print("Total words in data set: ", len(total_counts))

LIMIT = 7500

vocab = sorted(total_counts, key=total_counts.get, reverse=True)[:LIMIT]
print(vocab[LIMIT-1], total_counts[vocab[LIMIT-1]])

word2idx = {w: i for i, w in enumerate(total_counts.keys())}

def text_to_vector(text):
    vec = np.zeros(len(vocab), dtype=np.int_)
    text = text.split()
    for w in text:
        i = word2idx.get(w)
        if i and i < LIMIT:
            vec[i] += 1 
    return np.array(vec)

word_vectors = np.zeros((len(reviews), len(vocab)), dtype=np.int_)
for ii, (_, text) in enumerate(reviews.iterrows()):
    word_vectors[ii] = text_to_vector(text[0])

Y = (labels=='positive').astype(np.int_)
records = len(labels)

shuffle = np.arange(records)
np.random.shuffle(shuffle)
test_fraction = 0.9

train_split, test_split = shuffle[:int(records*test_fraction)], shuffle[int(records*test_fraction):]
trainX, trainY = word_vectors[train_split,:], to_categorical(Y.values[train_split], 2)
testX, testY = word_vectors[test_split,:], to_categorical(Y.values[test_split], 2)

print(trainY)

# Network building
def build_model():
    # This resets all parameters and variables, leave this here
    tf.reset_default_graph()
    
    net = tflearn.input_data([None, LIMIT])                          # Input
    net = tflearn.fully_connected(net, 100, activation='ReLU')      # Hidden
    net = tflearn.fully_connected(net, 28, activation='ReLU')      # Hidden
    net = tflearn.fully_connected(net, 2, activation='softmax')   # Output
    net = tflearn.regression(net, optimizer='sgd', learning_rate=0.1, loss='categorical_crossentropy')
    model = tflearn.DNN(net)

    return model

model = build_model()

# Training
model.fit(trainX, trainY, validation_set=0.1, show_metric=True, batch_size=128, n_epoch=120)


predictions = (np.array(model.predict(testX))[:,0] >= 0.5).astype(np.int_)
test_accuracy = np.mean(predictions == testY[:,0], axis=0)
print("Test accuracy: ", test_accuracy)

# Helper function that uses your model to predict sentiment
def test_sentence(sentence, expected):
    positive_prob = model.predict([text_to_vector(sentence.lower())])[0][1]
    #print('Sentence: {}'.format(sentence))
    output = ('Neutrally' if round(positive_prob, 2) < 0.55  and round(positive_prob, 2) > 0.41 else 'Opinionated',
              'Positive' if positive_prob > 0.5 else 'Negative')
    print(
        'P(positive) = {:.3f} :'.format(positive_prob),
        output,
        expected,
        str(expected == output)
    )

sentence = "Moonlight is by far the best movie of 2016."
test_sentence(sentence, ('Opinionated', 'Positive'))

sentence = "It's amazing anyone could be talented enough to make something this spectacularly awful"
test_sentence(sentence, ('Opinionated', 'Negative'))

sentence = "So you've sworn off sugar and fat and this product seems like it could be your salvation . Think again This deceptive little gummi bear looks and tastes amazing. The flavor bursts in your mouth and you can't believe that this delicious treat could be sugar and fat free ! Guess what: it's not . Lycasin has only a 20-30\% lower glycemic index than sugar , not 0 , like they lead you to believe. But you will not know this as you eat one after another of these tasty little fruity guys ... A few handfuls later ( maybe more ... ) you may notice your tummy rumbling . But these taste so good you may just keep snacking away . Even if you do not  it may be too late . Within an hour you are going to experience the worst stomach pain you've ever experienced . When something seems too good to be true it probably is".lower()
test_sentence(sentence, ('Opinionated', 'Positive'))

sentence = "I have a severe shoulder injury that cannot be improved with surgery or physical therapy , I have tried . My doctors have turned to various painkillers instead , these work to a certain degree , but the worst side effect is the constipation . I can't help being blunt , anyone who has been on opiates for a period of time has probably experienced this . The Haribo Sugarless Gummy Bears can apparently cause diarrhea if you eat more than 15 , for example (YMMV) . They are a godsend for me . They taste great , they are SIGNIFICANTLY cheaper than all the medications to relieve the constipation I have been given ( both prescription and over the counter ) , and more importantly , they actually work . I am a normal person if I eat 25 of these in a sitting every day . I have told my pain doctors about this and they are now recommending this product to all their patients . They really are yummy , particularly the pineapple ones ..".lower()
test_sentence(sentence, ('Opinionated', 'Negative'))

sentence = "I saw the reviews . I got them for my work candy bowl as a prank to all the buggers who steal my candy . I tried a couple . They taste EXACTLY like real gummy bears . These gummy bears are SO GOOD !!! I ate about 20 . Fast forward 2 hours . I'm in the bathroom pooping my mind out . My stomach is growling in fury . BUT I CAN'T STOP EATING THESE GUMMY BEARS! THEY ARE SO GOOD !!! For four hours now , I've been stuck in a vicious \" poop and eat more \" cycle . And I can't stop !!!".lower()
test_sentence(sentence, ('Opinionated', 'Positive'))

sentence = "My husband has never allowed me to write , as he doesn’t want me touching mens pens . However when I saw this product , I decided to buy it ( using my pocket money ) and so far it has been fabulous! Once I had learnt to write , the feminine colour and the grip size ( which was more suited to my delicate little hands ) has enabled me to vent thoughts about new recipe ideas , sewing and gardening . My husband is less pleased with this product as he believes it will lead to more independence and he hates the feminine tingling sensation ( along with the visions of fairies and rainbows ) he gets whenever he picks it up .".lower()
test_sentence(sentence, ('Opinionated', 'Positive'))

sentence = "Surprisingly busy on a Monday evening . It is definitely a dive bar anyone goes to from suits to students . Not the cleanest place but beer prices are ok for the area . The beer garden area is packed even in October on a Monday . Draws in a younger crowd mainly . Would return for the beer garden in the summer".lower()
test_sentence(sentence, ('Neutrally', 'Positive'))

sentence = "This place will never win any awards for good looks on the outside as it's part of a huge block of flats in Southwark but walk in there and you'll be greeted by friendly staff and a great atmosphere . Their food is big tasty portions of top notch pub food but done really well , they even double fry their chips for extra crunch . The decor is fun and funky , lots of pop art and posters on the wall and fairy lights on the exterior . Don't just walk by this place as I did for ages on the way to The Union Jack down the road , go in and try it instead .".lower()
test_sentence(sentence, ('Neutrally', 'Negative'))
