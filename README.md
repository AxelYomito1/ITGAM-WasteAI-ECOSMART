# 🌱 EcoSmart AI - Clasificador Inteligente de Residuos

Proyecto desarrollado para la materia de Inteligencia Artificial en el **Tecnológico Nacional de México, Campus Gustavo A. Madero (ITGAM)** para la carrera de Ingeniería en Tecnologías de la Información y Comunicaciones (ITICS).

El sistema utiliza una arquitectura de red neuronal convolucional **MobileNetV2** mediante Transfer Learning en PyTorch para clasificar 4 categorías de residuos comunes en el campus en tiempo real utilizando la webcam, ofreciendo recomendaciones de botes institucionales y un contador de auditoría por sesión.

## 👥 Equipo de Desarrollo
* **Dominic**
* **César**
* **Ricky**
* **Víctor**

---

## 🛠️ Requisitos del Sistema

* **Sistema Operativo:** Windows 10/11, macOS o Linux.
* **Lenguaje:** Python 3.12 (Se recomienda usar esta versión específica para evitar conflictos de dependencias).
* **Hardware mínimo:** Cámara web integrada o USB activa.

---

## 🚀 Guía de Instalación y Despliegue

Sigue estos pasos en la terminal de tu sistema operativo o en VS Code para ejecutar la aplicación web localmente:

### 1. Instalar las dependencias necesarias
Abre la terminal de tu entorno de Python 3.12 y ejecuta el siguiente comando para instalar las librerías del núcleo de procesamiento, interfaz e inferencia:

```powershell
python3.12 -m pip install streamlit opencv-python torch torchvision pillow
2. Ejecutar la Aplicación Web
Para iniciar el servidor local de Streamlit, debes navegar hacia la carpeta donde residen los componentes web (/web). Ejecuta los siguientes comandos en orden en tu terminal:

PowerShell
# 1. Moverse a la carpeta web donde está el archivo app_web.py y modelo_residuos.pth
cd web

# 2. Arrancar la interfaz web interactiva centrada
python3.12 -m streamlit run app_web.py
Una vez ejecutado, el sistema abrirá automáticamente una pestaña en tu navegador web predeterminado (en la dirección local http://localhost:8501).

📂 Estructura del Repositorio
Para mantener la trazabilidad del proyecto, los archivos están organizados bajo la siguiente jerarquía oficial:

/data: Archivos de configuración del conjunto de datos y mapeo de etiquetas (label_map.json).

/train: Scripts de entrenamiento base y código de captura experimental (camara.py).

/api: Infraestructura dispuesta para endpoints externos (vacía/gitkeep).

/web: Código del frontend interactivo centrado (app_web.py) y archivo de pesos de la red neuronal (modelo_residuos.pth).

README.md: Este manual de instalación y reproducibilidad.

REPORT.md: Reporte técnico formal del proyecto (6-10 páginas).

🛡️ Robustez (Filtro de Casos Fuera de Distribución)
El sistema integra un umbral de confianza mínimo de 70%. Si se obstruye la cámara o se introduce un objeto desconocido que no pertenezca a las clases de entrenamiento (Carton_Papel, Vidrio, Metal, Plastico), la interfaz mantendrá la barra de carga en un estado pasivo y mostrará el mensaje "Analizando... / Objeto no identificado", mitigando falsas predicciones y garantizando la estabilidad del software ante el profesor.
