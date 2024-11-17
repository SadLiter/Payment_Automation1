презентация https://www.figma.com/deck/qldJ6gb0RYqtpYmf3PfPd5/KIU-hack-slides?node-id=1-1084&t=iGtIsSte73vsTLym-1

Замените файл payments_main.tsv на свой с таким же названием

Соберите докер-контейнер, используя:

docker build . -t model (собирается долго)

далее запустите контейнер

docker run -it model /bin/bash

далее поменяйте путь на ваш файл tsv в main.py
запустите main.py
