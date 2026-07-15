.PHONY: clean

module.tar.gz: run.sh requirements.txt meta.json src/*.py *.so
	chmod +x run.sh
	tar czf $@ $^

clean:
	rm -f module.tar.gz