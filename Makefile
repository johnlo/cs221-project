all:
	make clean
	python reorganize.py
	python main.py run

clean:
	rm -rf *~ tmp/ *.pyc *.stackdump
