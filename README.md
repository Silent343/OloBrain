# OloBrain — 3D Neurosurgical and Neurophysiological Workstation (139 Structures)

**OloBrain** is an interactive 3D workstation for neuroanatomical visualization, synaptic bioelectrical simulation, white-matter tractography, and cerebrovascular hemodynamics. It follows **Domain-Driven Design (DDD)**, **Clean Architecture**, **Clean Code**, and **SOLID** principles, grounded in Q1/Q2 neurosurgical and neuroscience literature.

## Requirements

- Git.
- Python 3.11 or newer.
- A GPU and drivers compatible with OpenGL 3.3 Core.
- An optional webcam for gesture control.

## Clone and install

Clone the repository and enter its directory:

```bash
git clone https://github.com/Silent343/OloBrain.git
cd OloBrain
```

### Windows (PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

If PowerShell blocks activation, run the following once in that terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Activate the virtual environment before using the project in a new terminal. To update an existing clone:

```bash
git pull
python -m pip install -e .
```

## Verify and run

Validate the installation and atlas before opening the interface:

```bash
python run.py --smoke-test
python -m unittest discover -s tests -v
```

Start the desktop application with keyboard and mouse controls. This mode does not require a webcam or an external model:

```bash
python run.py --windowed --no-camera
```

After installing the package, you can also use the equivalent command:

```bash
olobrain --windowed --no-camera
```

### Enable gesture control

The MediaPipe model is intentionally not stored in Git. Download the official model to the path expected by the application.

Windows (PowerShell):

```powershell
New-Item -ItemType Directory -Force models | Out-Null
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" -OutFile "models\hand_landmarker.task"
python run.py --windowed
```

Linux/macOS:

```bash
mkdir -p models
curl -L "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" -o models/hand_landmarker.task
python run.py --windowed
```

Select another webcam with `python run.py --windowed --camera 1`. If the model or webcam is unavailable, the application continues in mouse mode and shows the reason in the interface.

## Software architecture (DDD + Clean Architecture + SOLID)

```text
olobrain/
├── run.py                                  # Entry point with virtual-environment auto-detection
├── pyproject.toml                          # Standard Python package configuration
├── requirements.txt                        # Dependencies (NumPy, PySide6, PyOpenGL, MediaPipe, OpenCV)
├── src/
│   └── olobrain/
│       ├── main.py                         # Composition root and dependency injection
│       ├── domain/                         # Pure domain layer (no UI/GPU dependencies)
│       │   ├── model.py                    # Aggregate root (BrainExperience), entities, value objects
│       │   └── catalog.py                  # Neurosurgical catalog of 139 structures
│       ├── application/                    # Use cases and ports (DIP / SRP)
│       │   └── controller.py               # BrainController, GestureCoordinator, HandTracker protocol
│       ├── infrastructure/                 # 3D modeling, connectome, and vision adapters
│       │   ├── geometry.py                 # 3D mathematical primitives, Catmull-Rom splines, and meshes
│       │   ├── sculpted_geometry.py        # Cortical folding, cerebellar folia, and dissection
│       │   ├── brain_atlas.py              # Parametric 3D atlas of 139 regions and explosion vectors
│       │   ├── circulation_geometry.py     # 24 action-potential routes and the Circle of Willis
│       │   └── vision.py                   # Strictly private MediaPipe HandLandmarker adapter
│       └── presentation/                   # Native Qt interface and GLSL 3.3 Core shaders
│           ├── scene_layout.py             # Model-space transforms and layer visibility
│           ├── flow_renderer.py            # GPU renderer for bioelectrical spikes and vascular flow
│           ├── gpu_renderer.py             # OpenGL 3.3 Core engine with cortical PBR shading and FBO picking
│           ├── qt_app.py                   # PySide6 workstation with a 139-part search tool
│           └── renderer.py                 # Software 3D rasterizer for fallback and headless captures
└── tests/                                  # Unit and geometric-regression test suite
    ├── test_olobrain.py
    ├── test_anatomical_scene.py
    └── test_circulation_and_elements.py
```

## Distribution of the 139 brain structures across eight clinical systems

1. **Meninges, cisterns, and ventricular system (14 structures):** Convexity dura mater, *falx cerebri*, *tentorium cerebelli*, arachnoid/pia mater, left and right lateral ventricles, foramina of Monro, third ventricle, cerebral aqueduct, fourth ventricle, choroid plexus, foramina of Luschka/Magendie, and *septum pellucidum*.
2. **Cerebral cortex: gyri and lobes (36 structures — 18 left and 18 right):** Superior, middle (DLPFC), and inferior frontal gyri (*pars orbitalis*), orbitofrontal cortex, paracentral lobule, frontal pole, superior parietal lobule, supramarginal gyrus, angular gyrus, precuneus, superior/middle/inferior temporal gyri, fusiform gyrus (VWFA/FFA), cuneus, lingual gyrus, lateral occipital cortex, and insular cortex of Reil.
3. **Eloquent language, primary motor, and sensory areas (18 structures):** Primary motor cortex M1 (left/right precentral gyrus), premotor cortex PMA (left/right), supplementary motor area SMA (left/right), frontal eye fields FEF (left/right), Broca's area (*pars opercularis* BA 44 and *pars triangularis* BA 45), right Broca homologue, Wernicke's area (BA 22p), Geschwind territory (BA 39/40), primary somatosensory cortex S1 (left/right), transverse temporal gyri of Heschl (A1), and primary visual cortex V1 (left/right calcarine cortex).
4. **White-matter tracts and bioelectrical connectome (16 structures):** Genu, body, and splenium of the corpus callosum; anterior white commissure; bilateral arcuate fasciculus (dorsal language pathway); bilateral superior longitudinal fasciculus (SLF); bilateral uncinate fasciculus; bilateral inferior fronto-occipital fasciculus (IFOF, ventral semantic pathway); bilateral internal capsule/corticospinal tract; and bilateral optic radiation of Gratiolet/Meyer's loop.
5. **Basal ganglia, thalamus, and diencephalon (18 structures):** Bilateral caudate nucleus, putamen, external globus pallidus (GPe), internal globus pallidus (GPi), subthalamic nucleus of Luys (STN), claustrum, dorsal thalamus, geniculate bodies (LGN/MGN), hypothalamus/infundibulum, epithalamus/pineal gland, and pituitary gland.
6. **Limbic system and memory circuit (12 structures):** Bilateral hippocampus (Cornu Ammonis and dentate gyrus), amygdala, fornix, cingulate gyrus, parahippocampal gyrus/entorhinal cortex, mammillary bodies, and nucleus accumbens/septal nucleus.
7. **Brainstem, cerebellum, and cranial nerves (13 structures):** Midbrain (tectum and tegmentum), cerebral peduncles and substantia nigra, pons, medulla oblongata and pyramids, ascending reticular activating system (ARAS), left and right cerebellar hemispheres, vermis/flocculonodular lobe, deep cerebellar nuclei (dentate), and cranial nerves I–II, III–IV–VI, V–VII–VIII (cerebellopontine angle), and IX–X–XI–XII (speech and tongue motor execution).
8. **Cerebrovascular circulation: Circle of Willis and venous sinuses (12 structures):** Internal carotid arteries (C4–C7), anterior cerebral arteries (ACA), anterior communicating artery (AComA), middle cerebral arteries (MCA M1–M4), vertebrobasilar system (PICA/AICA/SCA), posterior cerebral arteries (PCA), superior sagittal/straight sinuses, and great vein of Galen/transverse sinuses.

## Keyboard, mouse, and spatial-gesture controls

| Neurosurgical action | Keyboard / mouse | Spatial gesture (MediaPipe) |
| :--- | :--- | :--- |
| **Rotate the 3D brain** | Drag with the left mouse button | Moving open palm |
| **Progressive zoom** | Mouse wheel | Two hands moving closer or farther apart |
| **Select a structure (1–139)** | Left-click a mesh or list item | Pointing extended index finger |
| **3D explosion view (139 parts)** | `E` key | Victory gesture (two fingers) |
| **Move an exploded fragment** | Drag the part with the left mouse button | Pinch (thumb and index finger) |
| **Deep subcortical dissection** | `I` key | Thumb down |
| **Bioelectrical activity and living tissue** | `R` or `Space` | Three extended fingers |
| **Synaptic impulses and arterial flow** | `F` key | Index finger and pinky |
| **Spatial mode without panels** | `S` key | Extended pinky |
| **Restore anatomical view** | `Home` key | Thumb up |

## Authorship and license

Developed by **Gabriel (Silent343)**. Distributed under the [MIT License](LICENSE).
