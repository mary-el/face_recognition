## Face Recognition for Smart Entrance Gate
![img](https://github.com/user-attachments/assets/898ac845-e883-4e3c-9ea3-c2a648b13248)

---

**Project Overview**
- This project integrates face recognition technology to facilitate a secure and efficient entry process for the
local university in my hometown. It identifies students as they approach through live video feed from an entrance camera system and, upon successful
identification within our database of known faces, one designated turnstile will open to grant access.
- Now with web-interface


**Privacy & Security Measures**
- User Data Protection: For privacy reasons, all personal photos are stored encoded to ensure confidentiality and compliance with regulations.  

**Configuration Options**
- Recognition Areas: Users can define the screen areas for both entrance and exit
turnstiles
- Face Recognition Algorithms Selection: Enables users to choose between different methods for face recognition
  Now available:
  * ~~face-recognition~~
  * facenet

**config.yaml**
* source: excel \ db
* excel_file: excel file path if source == excel
* images_folder: folder with photos of users
* mode: [facenet](https://github.com/timesler/facenet-pytorch) / [face-recognition](https://github.com/ageitgey/face_recognition)
* no_name_user: how unknown people are labelled
* show: True \ False - show camera input on screen
* test_mode: True \ False - in test mode the turnstiles do not work

* camera 
  + id: camera IP \ 0 for default camera
  + frame_mode: full \ center - how frames face is located relately to the camera areas 

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
1. Put user photos in the `images` folder
2. Put in `excel_file` user names with their ids (id = embedding file name) or use a database
3. Run the program:
```commandline
uvicorn main:app --host HOST --port PORT --reload 
```
4. Open PORT:HOST in your browser
5. To update config, use the "Upload" button
6. To add new users,  use "Synchronize" button
