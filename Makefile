init:
	pip install -U pip && pip install -r requirements.txt
	apt-get update && apt-get install ffmpeg libsm6 libxext6 unzip -y
	curl -L $(yadisk-direct https://disk.yandex.ru/d/5Wg3tvP7Gywk0g) -o data.zip
	unzip data.zip && rm -f data.zip
	clearml-init
