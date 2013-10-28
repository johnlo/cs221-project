all: clean
	cp -r data/ tmp/
	python main.py classify

clean:
	rm -rf *~ tmp/ *.pyc *.stackdump
