all: install

install: clean
	cp -r data/ tmp/

clean:
	rm -rf *~ tmp/ *.pyc *.stackdump
