import os
import shutil
import uuid

def mkdir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

def main():
    root_dir = './tmp'
    mkdir(root_dir)
    full_dir = './tmp/full'
    mkdir(full_dir)
    mainonly_dir = './tmp/mainonly'
    mkdir(mainonly_dir)
    i = 0
    for root, dirs, files in os.walk('./data'):
        for dir in dirs:
            moodfile = os.path.join(os.path.join(root, dir), 'mood')
            textfile = os.path.join(os.path.join(root, dir), 'text')
            f = open(moodfile, 'r')
            moods = f.read().split()
            f.close()
            for mood in moods:
                destdirs = ([ full_dir, mainonly_dir ]
                            if (mood in [ 'Excited', 'Tender', 'Scared',
                                          'Angry', 'Sad', 'Happy' ])
                            else [ full_dir ])
                for destdir in destdirs:
                    mooddir = os.path.join(destdir, mood)
                    mkdir(mooddir)
                    dest = os.path.join(mooddir, str(i))
                    i += 1
                    shutil.copyfile(textfile, dest)
                
if __name__ == "__main__":
    main()
