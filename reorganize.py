import os
import uuid

def main():
    tmp_dir = os.path.join(os.path.dirname(__file__), 'tmp')
    os.mkdir(tmp_dir)
    for root, dirs, files in os.walk("./data", topdown=False):
        moodfile = os.path.join(root, 'mood')
        textfile = os.path.join(root, 'text')
        with open(moodfile) as f:
            moods = f.readlines()
        for mood in moods:
            mooddir = os.path.join(tmp_dir, mood.strip())
            if not os.path.exists(mooddir):
                os.mkdir(mooddir)
                newfile = open(mooddir + '/' + str(uuid.uuid4()))
                with open(textfile) as f:
                    newfile.write(f.read())

if __name__ == "__main__":
    main()
