import numpy as np
import string

initial = {}
second_word = {}
transitions = {}
rhyme = {}

def remove_punct(s):
    return    s.translate(str.maketrans('','',string.punctuation))

def add2dict(d, k, v):
    if k not in d:
        d[k] = []
    d[k].append(v)

def readFile(f):
    last = ''
    for line in open(f):
        tokens = remove_punct(line.rstrip().lower()).split()

        T = len(tokens)
        if not T:
            last = ''
        for i in range(T):
            t = tokens[i]
            if i == 0:
                # measure the distribution of the first word
                initial[t] = initial.get(t, 0.) + 1
            else:
                t_1 = tokens[i-1]
                if i == T - 1:
                    # measure probability of ending the line
                    add2dict(transitions, (t_1, t), 'END')
                    if(last):
                        add2dict(rhyme,last,t)
                        add2dict(rhyme,t,last)
                    else:
                        last = t
                if i == 1:
                    # measure distribution of second word
                    # given only first word
                    add2dict(second_word, t_1, t)
                else:
                    t_2 = tokens[i-2]
                    add2dict(transitions, (t_2, t_1), t)

readFile('files/robert_frost.txt')
readFile('files/edgar_allan_poe.txt')

initial_total = sum(initial.values())
for i in initial:
    initial[i] /= initial_total

def list2pdict(ts):
    # turn each list of possibilities into a dictionary of probabilities
    d = {}
    n = len(ts)
    for t in ts:
        d[t] = d.get(t, 0.) + 1
    for t, c in d.items():
        d[t] = c / n
    return d

for t_1, ts in second_word.items():
    # replace list with dictionary of probabilities
    second_word[t_1] = list2pdict(ts)

for k, ts in transitions.items():
    transitions[k] = list2pdict(ts)


def sample_word(d):
    # print "d:", d
    p0 = np.random.random()
    # print('reach probability',p0)
    # print "p0:", p0
    cumulative = 0
    for t, p in d.items():
        cumulative += p
        # print(t,p,cumulative)
        if p0 < cumulative:
            # print(t)
            return t
    assert(False) # should never get here

# print(set([1,2,3,3,4,5]))
def makeItRhyme(prev,curr):   
    rythms = None
    if(prev in rhyme):
        rythms = list(set(rhyme[prev]))    
        if(curr in rythms):
            return False
        else:
            T = len(rythms)
            if(T == 1):
                return rythms[0]

            i = np.random.randint(0,T-1)
            return rythms[i]
    return False

# print(rhyme.keys())
def generate():
    last = ''
    for i in range(4):
        sentence =[]

        # initial word
        w0 = sample_word(initial)
        sentence.append(w0)

        # sample second word
        w1 = sample_word(second_word[w0])
        sentence.append(w1)

        # second-order transitions until END
        while True:
            w2 = sample_word(transitions[(w0, w1)])
            if w2 == 'END':
                if(last):
                    replace_word = makeItRhyme(last,w1)
                    if(replace_word):
                        sentence.pop()
                        sentence.append(replace_word)
                    last = ''
                else:
                    last = w1
                break
            sentence.append(w2)
            w0 = w1
            w1 = w2
        print(' '.join(sentence))

generate()