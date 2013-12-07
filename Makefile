all: install

install: clean
	cp -r data/ tmp/

generate: install
	python main.py generate

classify: install
	python main.py classify

clean:
	rm -rf *~ tmp/ *.pyc *.stackdump
