module.tar.gz: run.sh requirements.txt meta.json src/*.py *.so
	chmod +x run.sh
	tar czf $@ $^