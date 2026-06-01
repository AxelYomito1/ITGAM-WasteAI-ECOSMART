import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import cv2
from PIL import Image, ImageTk
import os
import customtkinter as ctk
from tkinter import filedialog

# ==========================================
# --- 1. CONFIGURACIÓN DEL MODELO BASE ---
# ==========================================
clases = ['Carton_Papel', 'Vidrio', 'Metal', 'Plastico']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, 4)

ruta_pesos = 'modelo_residuos.pth'

if os.path.exists(ruta_pesos):
    model.load_state_dict(torch.load(ruta_pesos, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"✅ ¡Modelo híbrido cargado con éxito en: {device}!")
else:
    print(f"❌ Error: No se encontró '{ruta_pesos}' en la carpeta actual.")
    exit()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

colores_materiales = {
    'Carton_Papel': '#d9a05b', 
    'Vidrio': '#2ecc71',       
    'Metal': '#95a5a6',        
    'Plastico': '#3498db',     
    'Defecto': '#3b8ed0'       
}

# ==========================================
# --- 2. CONFIGURACIÓN DE LA INTERFAZ ---
# ==========================================
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class AppDetector(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("EcoSmart Classifier - ITGAM")
        self.geometry("900x710") 
        self.resizable(False, False)
        
        self.cap = None
        self.camara_activa = False
        self.ultimo_frame_pil = None 
        
        # --- DISEÑO GRID ---
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=3)
        self.grid_rowconfigure(0, weight=1)
        
        # --- PANEL LATERAL DE CONTROLES ---
        self.sidebar = ctk.CTkFrame(self, width=220, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        self.logo_label = ctk.CTkLabel(self.sidebar, text="ECO-IA", font=ctk.CTkFont(size=24, weight="bold"))
        self.logo_label.pack(padx=20, pady=(20, 10))
        
        self.sub_label = ctk.CTkLabel(self.sidebar, text="Clasificador de Residuos", font=ctk.CTkFont(size=12, slant="italic"))
        self.sub_label.pack(padx=20, pady=(0, 20))
        
        # Botones del menú
        self.btn_camara = ctk.CTkButton(self.sidebar, text="📹 Iniciar Cámara", command=self.toggle_camara, height=40)
        self.btn_camara.pack(padx=20, pady=10, fill="x")
        
        self.btn_foto = ctk.CTkButton(self.sidebar, text="🖼️ Cargar Imagen", command=self.cargar_foto, fg_color="transparent", border_width=2, height=40)
        self.btn_foto.pack(padx=20, pady=10, fill="x")
        
        # --- SECCIÓN DE CRÉDITOS ---
        self.credits_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.credits_frame.pack(side="bottom", fill="x", padx=10, pady=15)
        
        self.lbl_team_title = ctk.CTkLabel(self.credits_frame, text="Equipo de Desarrollo:", font=ctk.CTkFont(size=11, weight="bold"), text_color="#7f8c8d")
        self.lbl_team_title.pack(anchor="w", padx=10)
        
        self.lbl_team_names = ctk.CTkLabel(self.credits_frame, text="• Dominic\n• Cesar\n• Victor\n• Ricky", font=ctk.CTkFont(size=11), justify="left", text_color="#95a5a6")
        self.lbl_team_names.pack(anchor="w", padx=20, pady=(2, 0))
        
        # --- PANEL PRINCIPAL (VISUALIZADOR) ---
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        
        self.video_label = ctk.CTkLabel(self.main_frame, text="Pantalla en espera...\nSelecciona una opción a la izquierda.", fg_color="#1a1a1a", width=600, height=430, corner_radius=10)
        self.video_label.pack(padx=20, pady=20, fill="both", expand=True)
        
        # --- BARRA INFERIOR DE RESULTADOS ---
        self.result_frame = ctk.CTkFrame(self.main_frame, height=120, fg_color="#2b2b2b")
        self.result_frame.pack(padx=20, pady=(0, 20), fill="x")
        
        self.lbl_resultado = ctk.CTkLabel(self.result_frame, text="Residuo: Esperando disparo...", font=ctk.CTkFont(size=18, weight="bold"), text_color="#3b8ed0")
        self.lbl_resultado.pack(pady=(12, 2))
        
        self.lbl_confianza = ctk.CTkLabel(self.result_frame, text="Confianza: 0.00%", font=ctk.CTkFont(size=13))
        self.lbl_confianza.pack(pady=(0, 6))
        
        self.progress_bar = ctk.CTkProgressBar(self.result_frame, width=400, height=8)
        self.progress_bar.set(0) 
        self.progress_bar.pack(pady=(0, 12))

    # ==========================================
    # --- 3. LÓGICA DE LA CÁMARA (VISTA PREVIA) -
    # ==========================================
    def toggle_camara(self):
        if not self.camara_activa:
            self.cap = cv2.VideoCapture(0) 
            self.camara_activa = True
            self.btn_camara.configure(text="🛑 Detener Cámara", fg_color="#d32f2f")
            self.actualizar_frame()
        else:
            self.camara_activa = False
            if self.cap:
                self.cap.release()
            self.btn_camara.configure(text="📹 Iniciar Cámara", fg_color="#3b8ed0")
            self.video_label.configure(text="Cámara detenida.", image=None)
            self.lbl_resultado.configure(text="Residuo: Esperando disparo...", text_color=colores_materiales['Defecto'])
            self.lbl_confianza.configure(text="Confianza: 0.00%")
            self.progress_bar.configure(progress_color=colores_materiales['Defecto'])
            self.progress_bar.set(0)

    def actualizar_frame(self):
        if self.camara_activa:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                
                clase, conf = self.predecir_objeto(pil_img)
                color_nodo = colores_materiales.get(clase, colores_materiales['Defecto'])
                
                self.lbl_resultado.configure(text=f"Residuo: {clase}", text_color=color_nodo)
                self.lbl_confianza.configure(text=f"Confianza: {conf:.2f}%")
                
                self.progress_bar.configure(progress_color=color_nodo)
                self.progress_bar.set(conf / 100.0)
                
                frame_resized = cv2.resize(frame, (600, 420))
                img_tk = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)))
                self.video_label.configure(image=img_tk, text="")
                self.video_label.image = img_tk
                
            self.after(10, self.actualizar_frame)

    # ==========================================
    # --- 4. LOGICA CARGAR FOTO INDIVIDUAL ----
    # ==========================================
    def cargar_foto(self):
        if self.camara_activa:
            self.toggle_camara()
            
        ruta_archivo = filedialog.askopenfilename(filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp")])
        if ruta_archivo:
            img_pil = Image.open(ruta_archivo).convert('RGB')
            clase, conf = self.predecir_objeto(img_pil)
            color_nodo = colores_materiales.get(clase, colores_materiales['Defecto'])
            
            self.lbl_resultado.configure(text=f"Residuo en Foto: {clase}", text_color=color_nodo)
            self.lbl_confianza.configure(text=f"Confianza: {conf:.2f}%")
            
            self.progress_bar.configure(progress_color=color_nodo)
            self.progress_bar.set(conf / 100.0)
            
            img_resized = img_pil.resize((600, 420))
            img_tk = ImageTk.PhotoImage(img_resized)
            self.video_label.configure(image=img_tk, text="")
            self.video_label.image = img_tk

    # ==========================================
    # --- 5. MOTOR DE INFERENCIA DE LA IA ------
    # ==========================================
    def predecir_objeto(self, pil_image):
        img_t = transform(pil_image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(img_t)
            probabilities = F.softmax(outputs, dim=1)
            confianza, predicted = torch.max(probabilities, 1)
            
        return clases[predicted.item()], confianza.item() * 100

    def destroy(self):
        if self.cap:
            self.cap.release()
        super().destroy()

if __name__ == "__main__":
    app = AppDetector()
    app.mainloop()
