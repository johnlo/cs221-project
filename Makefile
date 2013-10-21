all:
	make clean
	python reorganize.py
	python main.py part3a

clean:
	rm -rf *~ tmp/ *.pyc *.stackdump
