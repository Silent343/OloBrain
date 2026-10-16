# OloBrain — Estación Neuroquirúrgica y Neurofisiológica 3D (139 Estructuras)

**OloBrain** es una estación de trabajo interactiva en 3D para visualización neuroanatómica, simulación bioeléctrica sináptica, tractografía de sustancia blanca y hemodinámica cerebrovascular. Está diseñada bajo principios de **Domain-Driven Design (DDD)**, **Clean Architecture**, **Clean Code** y **SOLID**, con fundamentación científica en literatura neuroquirúrgica y neurocientífica Q1/Q2.

## Requisitos

- Git.
- Python 3.11 o superior.
- Una GPU y controladores compatibles con OpenGL 3.3 Core.
- Cámara web opcional para el control mediante gestos.

## Clonación e instalación

Clona el repositorio y entra en su directorio (sustituye la URL si publicas el proyecto con otro nombre):

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

Si PowerShell bloquea la activación, ejecuta una vez en esa terminal:

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

En cada nueva terminal, activa primero el entorno virtual. Para actualizar una copia ya clonada:

```bash
git pull
python -m pip install -e .
```

## Verificación y ejecución

Comprueba que la instalación y el atlas sean válidos antes de abrir la interfaz:

```bash
python run.py --smoke-test
python -m unittest discover -s tests -v
```

Inicia la aplicación en una ventana con controles de teclado y mouse; este modo no necesita cámara ni un modelo externo:

```bash
python run.py --windowed --no-camera
```

Después de instalar el paquete también puedes usar el comando equivalente:

```bash
olobrain --windowed --no-camera
```

### Activar el control por gestos

El modelo de MediaPipe no se almacena en Git. Descarga el modelo oficial en la ruta esperada por la aplicación.

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

Puedes seleccionar otra cámara con `python run.py --windowed --camera 1`. Si el modelo o la cámara no están disponibles, la aplicación continúa en modo mouse y muestra el motivo en la interfaz.

## Arquitectura del Software (DDD + Clean Architecture + SOLID)

```text
olobrain/
├── run.py                                  # Punto de arranque con autodetección de entorno virtual
├── pyproject.toml                          # Configuración estándar de paquete Python
├── requirements.txt                        # Dependencias (NumPy, PySide6, PyOpenGL, MediaPipe, OpenCV)
├── src/
│   └── olobrain/
│       ├── main.py                         # Composition Root e inyección de dependencias
│       ├── domain/                         # Capa de Dominio Pura (0 dependencias de UI/GPU)
│       │   ├── model.py                    # Aggregate Root (BrainExperience), Entidades, Value Objects
│       │   └── catalog.py                  # Catálogo neuroquirúrgico de las 139 estructuras
│       ├── application/                    # Casos de Uso y Puertos (DIP / SRP)
│       │   └── controller.py               # BrainController, GestureCoordinator, HandTracker Protocol
│       ├── infrastructure/                 # Adaptadores de Modelado 3D, Conectoma y Visión
│       │   ├── geometry.py                 # Primitivas matemáticas 3D, splines Catmull-Rom y mallas
│       │   ├── sculpted_geometry.py        # Plegamiento cortical (giros/surcos), folias y disección
│       │   ├── brain_atlas.py              # Atlas paramétrico 3D de las 139 regiones + vectores de esparcimiento
│       │   ├── circulation_geometry.py     # 24 rutas de potenciales de acción y Polígono de Willis
│       │   └── vision.py                   # Adaptador MediaPipe HandLandmarker con privacidad estricta
│       └── presentation/                   # Interfaz Gráfica Nativa Qt y Shaders GLSL 3.3 Core
│           ├── scene_layout.py             # Transformaciones en espacio de modelo y visibilidad por capas
│           ├── flow_renderer.py            # Renderizador GPU de espigas bioeléctricas y flujo vascular
│           ├── gpu_renderer.py             # Motor OpenGL 3.3 Core con sombreado cortical PBR y FBO picking
│           ├── qt_app.py                   # Estación de trabajo PySide6 con buscador de 139 partes
│           └── renderer.py                 # Rasterizador 3D por software (fallback y capturas headless)
└── tests/                                  # Suite de pruebas unitarias y de regresión geométrica
    ├── test_olobrain.py
    ├── test_anatomical_scene.py
    └── test_circulation_and_elements.py
```

## Distribución de las 139 Estructuras del Encéfalo (8 Sistemas Clínicos)

1. **Meninges, Cisternas y Sistema Ventricular (14 estructuras)**: Duramadre de la convexidad, hoz del cerebro (*falx cerebri*), tienda del cerebelo (*tentorium cerebelli*), aracnoides/piamadre, ventrículos laterales izquierdo y derecho, forámenes de Monro, tercer ventrículo, acueducto de Silvio, cuarto ventrículo, plexo coroideo, forámenes de Luschka/Magendie y *septum pellucidum*.
2. **Corteza Cerebral: Giros y Lóbulos (36 estructuras — 18 izquierdas y 18 derechas)**: Giros frontales superior, medio (DLPFC) e inferior (*pars orbitalis*), corteza orbitofrontal, lóbulo paracentral, polo frontal, lóbulo parietal superior, giro supramarginal, giro angular, precúneo, giros temporales superior, medio e inferior, giro fusiforme (VWFA/FFA), cúneo, giro lingual, corteza occipital lateral y corteza insular de Reil.
3. **Áreas Elocuentes del Habla, Motoras y Sensoriales Primarias (18 estructuras)**: Corteza motora primaria M1 (giro precentral izq./der.), corteza premotora PMA (izq./der.), área motora suplementaria SMA (izq./der.), campos oculares frontales FEF (izq./der.), área de Broca (*pars opercularis* BA 44 y *pars triangularis* BA 45), homólogo derecho de Broca, área de Wernicke (BA 22p), territorio de Geschwind (BA 39/40), corteza somatosensorial primaria S1 (izq./der.), giros transversos de Heschl (A1) y corteza visual primaria calcarina V1 (izq./der.).
4. **Tractos de Sustancia Blanca y Conectoma Bioeléctrico (16 estructuras)**: Rodilla, tronco y esplenio del cuerpo calloso, comisura blanca anterior, fascículo arqueado bilateral (vía dorsal del habla), fascículo longitudinal superior SLF bilateral, fascículo uncinado bilateral, fascículo fronto-occipital inferior IFOF bilateral (vía ventral semántica), cápsula interna / tracto corticoespinal piramidal bilateral y radiación óptica de Gratiolet / asa de Meyer bilateral.
5. **Ganglios Basales, Tálamo y Diencéfalo (18 estructuras)**: Núcleo caudado bilateral, putamen bilateral, globo pálido externo (GPe) bilateral, globo pálido interno (GPi) bilateral, núcleo subtalámico de Luys (STN) bilateral, claustro bilateral, tálamo dorsal bilateral, cuerpos geniculados (LGN/MGN), hipotálamo/infundíbulo, epitálamo/glándula pineal e hipófisis.
6. **Sistema Límbico y Circuito de la Memoria (12 estructuras)**: Hipocampo (asta de Amón y giro dentado) bilateral, amígdala bilateral, fórnix bilateral, giro del cíngulo bilateral, giro parahipocampal/corteza entorrinal bilateral, cuerpos mamilares y núcleo accumbens/septal.
7. **Tronco Encefálico, Cerebelo y Pares Craneales (13 estructuras)**: Mesencéfalo (tectum y tegmento), pedúnculos cerebrales y sustancia negra, protuberancia anular (puente de Varolio), bulbo raquídeo y pirámides, sistema reticular activador ascendente (ARAS), hemisferios cerebelosos izquierdo y derecho, vermis/floculonodular, núcleos profundos del cerebelo (dentado) y pares craneales I-II, III-IV-VI, V-VII-VIII (ángulo pontocerebeloso) y IX-X-XI-XII (ejecución fonatoria y lingual del habla).
8. **Circulación Cerebrovascular: Polígono de Willis y Senos Venosos (12 estructuras)**: Arterias carótidas internas (C4-C7), arterias cerebrales anteriores (ACA), comunicante anterior (AComA), arterias cerebrales medias (MCA M1-M4), sistema vertebrobasilar (PICA/AICA/SCA), arterias cerebrales posteriores (PCA), seno sagital superior/recto y gran vena de Galeno/senos transversos.

## Controles por Teclado, Mouse y Gestos Espaciales

| Acción Neuroquirúrgica | Teclado / Mouse | Gesto Espacial (MediaPipe) |
| :--- | :--- | :--- |
| **Rotar el encéfalo 3D** | Arrastrar clic izquierdo | Mano abierta en movimiento |
| **Zoom progresivo** | Rueda del mouse | Dos manos (acercar / separar) |
| **Seleccionar estructura (1-139)** | Clic izquierdo sobre malla o lista | Índice extendido (apuntar) |
| **Esparcimiento 3D (139 partes)** | Tecla `E` | Gesto de Victoria (2 dedos) |
| **Mover fragmento en esparcimiento** | Arrastrar pieza con clic izquierdo | Pinza (pulgar + índice) |
| **Disección subcortical profunda** | Tecla `I` | Pulgar hacia abajo |
| **Actividad bioeléctrica y tejido vivo** | Tecla `R` o `Espacio` | Tres dedos extendidos |
| **Impulsos sinápticos y flujo arterial** | Tecla `F` | Índice + Meñique |
| **Modo espacial sin paneles** | Tecla `S` | Meñique extendido |
| **Restaurar vista anatómica** | Tecla `Home` | Pulgar hacia arriba |

## Autoría y licencia

Desarrollado por **Gabriel (Silent343)**. Distribuido bajo la [licencia MIT](LICENSE).
