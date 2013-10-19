import os
import string
import sys

def main():
    while True:
        text = ""
        print 'enter title: '
        title = sys.stdin.readline().strip()
        print 'enter author: '
        author= sys.stdin.readline().strip()
        text += title + ' by ' + author + '\n'
        print 'enter poem text (Ctrl-D to end): '
        while True:
            line = sys.stdin.readline()
            if not line:
                break
            text += line
        mood1, submood1 = getMood()
        mood2, submood2 = getMood()
        mood = mood1 + '\n' + submood1 + '\n' + mood2 + '\n' + submood2 + '\n'
        name = (string.lower(title).replace(' ', '_') + '_by_' +
                string.lower(author).replace(' ', '_'))
        os.mkdir(name)
        f = open(name + '/text', 'w')
        f.write(text)
        f.close()
        f = open(name + '/mood', 'w')
        f.write(mood)
        f.close

def getMood():
    main_moods = [ 'Excited', 'Tender', 'Scared', 'Angry', 'Sad', 'Happy' ]
    print 'enter main mood: '
    for i, mood in enumerate(main_moods):
        print ' ' + str(i) + ') ' + mood
    mood1 = sys.stdin.readline()

    excited_moods = [
        'Ecstatic', 'Energetic', 'Aroused', 'Bouncy', 'Nervous',
        'Perky', 'Antsy' ]
    tender_moods = [
        'Intimate', 'Loving', 'Warm-hearted', 'Sympathetic', 'Touched',
        'Kind', 'Soft' ]
    scared_moods = [
        'Tense', 'Nervous', 'Anxious', 'Jittery', 'Frightened',
        'Panic-stricken', 'Terrified' ]
    angry_moods = [
        'Irritated', 'Resentful', 'Miffed', 'Upset', 'Mad', 'Furious',
        'Raging' ]
    sad_moods = [
        'Down', 'Blue', 'Mopey', 'Grieved', 'Dejected', 'Depressed',
        'Heartbroken' ]
    happy_moods = [
        'Fulfilled', 'Contented', 'Glad', 'Complete', 'Satisfied',
        'Optimistic', 'Pleased' ]

    submoods_list = [ excited_moods, tender_moods, scared_moods,
                      angry_moods, sad_moods, happy_moods]
    submoods = submoods_list[int(mood1)]

    print 'enter submood: '
    for i, mood in enumerate(submoods):
        print ' ' + str(i) + ') ' + mood
    submood1 = sys.stdin.readline()

    return main_moods[int(mood1)], submoods[int(submood1)]

    
if __name__ == "__main__":
    main()
