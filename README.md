Замените файл payments_main.tsv на свой с таким же названием

Соберите докер-контейнер, используя:

docker build . -t model (собирается долго)

далее запустите контейнер

docker run -it model /bin/bash

далее поменяйте путь на ваш файл tsv в main.py
запустите main.py
