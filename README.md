## Face Recognition for Smart Entrance Gate
![изображение](https://github.com/user-attachments/assets/898ac845-e883-4e3c-9ea3-c2a648b13248)

---

**Project Overview**
- This project integrates face recognition technology to facilitate a secure and efficient entry process for the
local university in my hometown. It identifies students as they approach through live video feed from an entrance camera system and, upon successful
identification within our database of known faces, one designated turnstile will open to grant access.


**Privacy & Security Measures**
- User Data Protection: For privacy reasons, all personal photos are stored encoded to ensure confidentiality and compliance with regulations.  

**Configuration Options**
- Recognition Areas: Users can define the screen areas for both entrance and exit
turnstiles
- Face Recognition Algorithms Selection: Enables users to choose between different methods for face recognition
  Now available:
  * face-recognition
  * facenet

**config.yaml**
* camera: camera IP \ 0 for default camera
* source: excel \ db
* excel_file: excel file path if source == excel
* images_folder: folder with photos of users
* mode: [facenet](https://github.com/timesler/facenet-pytorch) / [face-recognition](https://github.com/ageitgey/face_recognition)
* no_name_user: how unknown people are labelled
* show: True \ False - show camera input on screen
* test_mode: True \ False - in test mode the turnstiles do not work

* connection:
  + host: IP of the host, responsible for the turnstile opening
  + login
  + password

* face-recognition:
  + embedding_folder: folder for face-recognition embeddings
  + num_jitters
  + tolerance: higher means more strict

* facenet:
  + embedding_folder: folder for facenet embeddings
  + threshold: lower means more strict

* turnsites:
  + area_1: exit area x, y, w, h relative values
  + area_2: entrance area
  + id_tur

**Usage Instructions** 
1. Run the program with -e parameter to encode images_folder
2. Run it without parameters

* camera: адрес камеры \ 0 для дефолтной
* source: excel \ db
* excel_file: если source == excel
* images_folder: папка с фото пользователей
* frame_folder: папка с  кадрами  распознаных в момент прохода
* mode: facenet (https://github.com/timesler/facenet-pytorch) / face-recognition (https://github.com/ageitgey/face_recognition)
* no_name_user: как подписываются неизвестные лица
* show: True \ False - включить вывод на экран
* test_mode: True \ False - в тестовом режиме турникеты не открываются
* date_time_format: формат вывода времени для включение его в имя файла
* log_file: имя лог файла

* connection:
  + host: адрес хоста, упарвляющего дверьми
  + login
  + password

* face-recognition:
  + embedding_folder: папка, где хранятся эмбеддинги 
  + num_jitters
  + tolerance: чем выше, тем строже опознавание

* facenet:
  + embedding_folder: папка с эмбеддингами
  + threshold: чем ниже, тем строже

* tourniquets:
  + area_1: x, y, w, h зоны 1 в относительных значениях
  + area_2
  + id_tur  для тугникета Perco

Чтобы создать эмбеддинги, запустить с опцией -e 
