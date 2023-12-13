DATA_URL := $(shell yadisk-direct https://disk.yandex.ru/d/5Wg3tvP7Gywk0g)

.PHONY: prepare_linux
prepare_linux:
	apt-get update && apt-get install ffmpeg libsm6 libxext6 unzip make -y

.PHONY: download_train_data
download_train_data:
	mkdir data && cd data
	curl -L $(DATA_URL) -o data_segmentation.zip
	unzip data_segmentation.zip
	rm -f data_segmentation.zip
