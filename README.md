## Face Recognition for Smart Entrance Gate
![img](https://github.com/user-attachments/assets/898ac845-e883-4e3c-9ea3-c2a648b13248)

---

**Project Overview**

A face recognition system for university entrance management that identifies students via live camera feed and opens designated turnstiles for authorized access. Features a web-based interface for configuration and monitoring.

**Key Features:**
- Real-time face detection and recognition
- Configurable entrance/exit areas
- Web interface for configuration and monitoring
- Privacy-focused: face embeddings stored instead of raw images
- Multiple recognition engine support (currently FaceNet)

**Face Recognition Engines:**
- ✅ **FaceNet** (InceptionResNetV1 + MTCNN) - Active
- 🚫 **face-recognition** - Currently disabled

## Configuration (config.yaml)

### General Settings
- `source`: Data source - `excel` or `db`
- `excel_file`: Path to Excel file (when source=excel)
- `images_folder`: Folder containing user photos
- `mode`: Recognition engine - `facenet` (recommended)
- `no_name_user`: Label for unrecognized users
- `test_mode`: `true`/`false` - Disables turnstile actuation when true
- `embedding_folder`: Where face embeddings are stored

### Camera Settings
- `camera.id`: Camera device ID or IP (0 for default webcam)
- `camera.reduce_frame`: Frame scaling factor
- `camera.frame_mode`: `center` (face center point) or `full` (face fully contained)

### Connection Settings (for database mode)
- `connection.host`: API host IP
- `connection.login`: API username
- `connection.password`: API password

### FaceNet Settings
- `facenet.embedding_folder`: Storage for FaceNet embeddings
- `facenet.threshold`: Matching threshold (lower = stricter)

### Turnstiles
- `turnstiles.area_1`: Exit area [x, y, width, height] (relative 0-1)
- `turnstiles.area_2`: Entrance area [x, y, width, height] (relative 0-1)
- `turnstiles.id_tur`: Turnstile device ID
- `turnstiles.min_time_diff`: Minimum seconds between door triggers

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. **Setup Users**: Place user photos in the `images` folder
2. **Configure Database**: 
   - For Excel mode: Create `db.xlsx` with columns [User ID, User Name]
   - For database mode: Configure connection settings in config.yaml
3. **Run Application**:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
4. **Access Web Interface**: Open `http://localhost:8000` in browser
5. **Web Interface Functions**:
   - Upload new configuration
   - Synchronize users and embeddings
   - View live camera feed
6. **Shutdown**: Press `Ctrl + C`

## Requirements

- Python 3.8+
- OpenCV
- PyTorch (for FaceNet)
- FastAPI (for web interface)
- See `requirements.txt` for complete list
