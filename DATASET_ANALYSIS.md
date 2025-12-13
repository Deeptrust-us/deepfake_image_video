# Complete Dataset Analysis

## 📹 Video to Frame Conversion Process

### 1. **Frame Extraction**
- **Method**: Uses `ffmpeg` to extract frames from videos
- **Sampling Rate**: **3 frames per second** (configured in `config.yaml` as `frame_sampling_rate: 3`)
- **Output Format**: JPG images named `frame_0001.jpg`, `frame_0002.jpg`, etc.
- **Location**: Frames saved to `data/frames/{video_id}/`

### 2. **Face Detection & Cropping**
- **Detector**: MTCNN face detector
- **Process**: 
  - Each extracted frame → Face detection → Face alignment → Crop to 224x224
- **Output**: Face crops saved to `data/faces/{video_id}/`
- **Format**: Same naming convention (`frame_0001.jpg`, etc.)

### 3. **Frequency Domain Processing**
- **Process**: Each face crop → FFT (Fast Fourier Transform) → Frequency spectrum
- **Output**: Frequency representations saved to `data/frequency/{video_id}/`
- **Format**: NumPy arrays (`.npy` files)

## 📊 Dataset Statistics

### **Total Videos**: 6,528 videos
- **Training**: 4,569 videos (70%)
- **Validation**: 978 videos (15%)
- **Test**: 981 videos (15%)

### **Class Distribution**:
- **Training**:
  - Real videos: 622 (13.6%)
  - Fake videos: 3,947 (86.4%)
  - ⚠️ **Highly imbalanced** (6.3:1 ratio)

- **Validation**:
  - Real videos: 133 (13.6%)
  - Fake videos: 845 (86.4%)

- **Test**:
  - Real videos: 134 (13.7%)
  - Fake videos: 847 (86.3%)

### **Frame Statistics**:
- **Average frames per video**: ~38 frames
- **Range**: 11-74 frames per video
- **Total frames**:
  - Training: **173,257 frames**
  - Validation: **37,189 frames**
  - Test: **37,503 frames**
  - **Grand Total: ~247,949 frames**

### **Frame Extraction Details**:
- Videos are sampled at **3 FPS** (3 frames per second)
- For a typical 10-second video → ~30 frames
- For a typical 12-second video → ~36 frames
- Frame count varies based on video length

## 🔄 Training Process

### **Frame Selection During Training**:
1. **Training Mode** (`is_training=True`):
   - **Random frame selection**: Each epoch, a random frame is selected from each video
   - This provides data augmentation through temporal diversity
   - Same video can contribute different frames in different epochs

2. **Validation/Test Mode** (`is_training=False`):
   - **Fixed frame selection**: Always uses the first frame (`frame_0001.jpg`)
   - Ensures consistent evaluation

### **Data Flow**:
```
Video (MP4)
  ↓ [ffmpeg @ 3 FPS]
Raw Frames (data/frames/{video_id}/)
  ↓ [MTCNN Face Detection]
Face Crops (data/faces/{video_id}/) ← Used for training
  ↓ [FFT]
Frequency Spectra (data/frequency/{video_id}/) ← Used for training
```

### **During Training**:
- Each batch contains frames from different videos
- Each sample = 1 frame from 1 video
- **Batch size**: 32 (from config)
- **Total training samples per epoch**: 4,569 (one random frame per video)
- **Effective samples**: Much higher due to random frame selection across epochs

## 📁 Directory Structure

```
data/
├── frames/           # Raw extracted frames (not used directly)
│   ├── video_0001/
│   │   ├── frame_0001.jpg
│   │   ├── frame_0002.jpg
│   │   └── ...
│   └── ...
├── faces/            # Face crops (USED FOR TRAINING)
│   ├── video_0001/
│   │   ├── frame_0001.jpg  (224x224 face crop)
│   │   ├── frame_0002.jpg
│   │   └── ...
│   └── ...
├── frequency/        # Frequency domain representations
│   ├── video_0001/
│   │   ├── frame_0001.npy
│   │   └── ...
│   └── ...
└── *_metadata.json   # Video metadata (train/val/test)
```

## 🎯 Key Points

1. **Frame-level training**: Model trains on individual frames, not entire videos
2. **Temporal diversity**: Random frame selection during training increases diversity
3. **Dual-stream input**: Each sample provides:
   - **Spatial stream**: Face crop image (RGB, 224x224)
   - **Frequency stream**: FFT magnitude spectrum (1 channel, 224x224)
4. **Class imbalance**: Strong bias toward fake videos (86% fake, 14% real)
5. **Dataset size**: ~248K frames total, but training uses 4,569 videos with random frame selection

## ⚙️ Configuration

From `config.yaml`:
- `frame_sampling_rate: 3` → 3 frames per second
- `face_size: 224` → Face crops are 224x224 pixels
- `frequency_channels: 1` → Only magnitude spectrum (not phase)
- `batch_size: 32` → 32 frames per batch

## 🔍 Verification

To verify your dataset:
```bash
# Count videos
ls data/faces/ | wc -l

# Check frames in a video
ls data/faces/{video_id}/ | wc -l

# View metadata
python3 -c "import json; data=json.load(open('data/train_metadata.json')); print(f'Training videos: {len(data)}')"
```

