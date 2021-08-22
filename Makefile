docker-build:
	docker build . -f Dockerfile -t dvbf

run:
	docker run -it --net=host -v `pwd`:/dvbf dvbf /bin/bash

run-gpu:
	docker run --gpus all -it --net=host -v `pwd`:/dvbf dvbf /bin/bash
