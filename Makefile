all: generate

generate: clean
	cp -r data/ tmp/
	python main.py generate

classify: clean
	cp -r data/ tmp/
	python main.py classify

clean:
	rm -rf *~ tmp/ *.pyc *.stackdump
