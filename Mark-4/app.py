from flask import Flask, render_template, request, jsonify
from openai import OpenAI
import logging
import os
from datetime import datetime
import requests
import matplotlib.pyplot as plt
import sympy as sp
import io
import base64
import psutil
import subprocess
import winapps
import glob
import json

app = Flask(__name__)

# Logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')

# Configuración de API Keys - Cambia aquí si no usas variables de entorno
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "tu_openai_api_key_aqui")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "tu_openweather_api_key_aqui")

client = OpenAI(api_key=OPENAI_API_KEY)

HISTORIAL_PATH = "historial_conversacion.json"

# --- Funciones para historial persistente ---

def cargar_historial():
    if os.path.exists(HISTORIAL_PATH):
        with open(HISTORIAL_PATH, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except Exception:
                return []
    return []

def guardar_historial(historial):
    with open(HISTORIAL_PATH, "w", encoding="utf-8") as f:
        json.dump(historial, f, ensure_ascii=False, indent=2)

def agregar_a_historial(role, content):
    historial = cargar_historial()
    historial.append({"role": role, "content": content})
    # Limitar historial a 30 mensajes para no saturar
    if len(historial) > 30:
        historial = historial[-30:]
    guardar_historial(historial)
    return historial

# --- Funciones para normalizar expresiones matemáticas ---
def normalizar_expresion(expresion):
    expresion = expresion.lower()
    expresion = expresion.replace("raíz cuadrada de", "sqrt")
    expresion = expresion.replace("seno de", "sin")
    expresion = expresion.replace("coseno de", "cos")
    expresion = expresion.replace("tangente de", "tan")
    expresion = expresion.replace("al cuadrado", "**2")
    expresion = expresion.replace("al cubo", "**3")
    expresion = expresion.replace(" elevado a ", "**")
    expresion = expresion.replace("por", "*")
    expresion = expresion.replace("entre", "/")
    expresion = expresion.replace("más", "+")
    expresion = expresion.replace("menos", "-")
    return expresion
# --- Funciones para hora, fecha y clima ---
def obtener_hora():
    ahora = datetime.now()
    return f"Ahora son las {ahora.strftime('%H:%M:%S')}."

def obtener_fecha():
    hoy = datetime.now()
    return f"Hoy es {hoy.strftime('%A %d de %B de %Y')}."

def obtener_clima(ciudad="Madrid"):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={ciudad}&appid={OPENWEATHER_API_KEY}&units=metric&lang=es"
        respuesta = requests.get(url)
        datos = respuesta.json()

        if datos.get("cod") != 200:
            return f"No pude obtener el clima para {ciudad}."

        descripcion = datos["weather"][0]["description"]
        temp = datos["main"]["temp"]
        humedad = datos["main"]["humidity"]
        return f"El clima en {ciudad} es {descripcion} con una temperatura de {temp}°C y humedad de {humedad}%."
    except Exception as e:
        return f"Error al obtener el clima: {e}"

# --- Detectar intención ---
def detectar_intencion(mensaje):
    msg = mensaje.lower()

    if any(p in msg for p in ["modo estudiante"]):
        return "modo_estudiante"
    if any(p in msg for p in ["hora", "qué hora", "dime la hora"]):
        return "hora"
    if any(p in msg for p in ["fecha", "qué día", "qué fecha"]):
        return "fecha"
    if any(p in msg for p in ["clima", "tiempo", "temperatura"]):
        return "clima"
    if any(p in msg for p in ["gráfica", "grafica", "graficar", "dibuja la función"]):
        return "graficar"
    if any(p in msg for p in ["abre", "abrir", "ejecuta", "ejecutar"]):
        return "abrir_app"
    if any(p in msg for p in ["cuánto espacio", "espacio disponible", "espacio libre", "almacenamiento"]):
        return "espacio"
    return "general"

# --- Convertir a LaTeX ---
def convertir_a_latex(mensaje):
    try:
        normalizado = normalizar_expresion(mensaje)
        expr = sp.sympify(normalizado, evaluate=False)
        return sp.latex(expr)
    except Exception:
        return None
# --- Graficar función ---
def graficar_funcion(expresion_str):
    try:
        x = sp.symbols('x')
        normalizada = normalizar_expresion(expresion_str)
        expr = sp.sympify(normalizada)
        f = sp.lambdify(x, expr, "numpy")

        import numpy as np
        x_vals = np.linspace(-10, 10, 400)
        y_vals = f(x_vals)

        plt.figure(figsize=(6, 4))
        plt.plot(x_vals, y_vals, label=f"${sp.latex(expr)}$")
        plt.title("Gráfica")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.grid(True)
        plt.legend()

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        buf.close()
        plt.close()
        return f"<img src='data:image/png;base64,{img_base64}'/>"
    except Exception as e:
        logging.error(f"Error al graficar: {e}")
        return "No pude graficar esa función, revisa la sintaxis."

# --- Espacio libre ---
def obtener_espacio_libre():
    try:
        disco = psutil.disk_usage('/')
        gb_libre = disco.free / (1024 ** 3)
        gb_total = disco.total / (1024 ** 3)
        return f"Tienes {gb_libre:.2f} GB libres de un total de {gb_total:.2f} GB."
    except Exception as e:
        return f"No pude obtener el espacio libre: {e}"

# --- Buscar apps instaladas ---
def buscar_apps_instaladas():
    apps = {}

    for app in winapps.list_installed():
        apps[app.name.lower()] = app.install_location or app.install_source or None

    rutas = [
        os.path.expandvars(r'%APPDATA%\Microsoft\Windows\Start Menu\Programs'),
        os.path.expandvars(r'%ProgramData%\Microsoft\Windows\Start Menu\Programs')
    ]
    for ruta in rutas:
        for archivo in glob.glob(os.path.join(ruta, '**/*.lnk'), recursive=True):
            nombre = os.path.splitext(os.path.basename(archivo))[0].lower()
            if nombre not in apps:
                apps[nombre] = archivo

    return apps

# --- Abrir aplicación ---
def abrir_aplicacion(nombre):
    apps = buscar_apps_instaladas()
    nombre = nombre.lower().strip()

    for app_nombre, ruta in apps.items():
        if nombre in app_nombre:
            try:
                if ruta and os.path.exists(ruta):
                    if ruta.endswith('.lnk'):
                        os.startfile(ruta)
                    else:
                        subprocess.Popen(ruta)
                    return f"Abriendo {app_nombre}..."
            except Exception as e:
                return f"No pude abrir {app_nombre}: {e}"

    return "No reconocí la aplicación que quieres abrir."

# --- Modo estudiante ---
modo_estudiante_activo = False

def responder_modo_estudiante(mensaje):
    prompt = f"Explícame detalladamente y paso a paso el siguiente tema o problema matemático: {mensaje}"
    historial = cargar_historial()
    historial.append({"role": "user", "content": prompt})
    completado = client.chat.completions.create(
        model="gpt-4",
        messages=historial
    )
    respuesta = completado.choices[0].message.content.strip()
    agregar_a_historial("assistant", respuesta)
    return respuesta
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/preguntar", methods=["POST"])
def preguntar():
    global modo_estudiante_activo

    datos = request.get_json()
    mensaje = datos.get("mensaje", "").strip()

    if not mensaje:
        return jsonify({"respuesta": "No entendí tu mensaje."})

    logging.info(f"Usuario: {mensaje}")

    intencion = detectar_intencion(mensaje)

    if intencion == "modo_estudiante":
        modo_estudiante_activo = True
        agregar_a_historial("user", mensaje)
        respuesta = "Modo estudiante activado. Pregúntame cualquier tema matemático y te lo explicaré paso a paso."
        agregar_a_historial("assistant", respuesta)
        return jsonify({"respuesta": respuesta})

    if modo_estudiante_activo:
        agregar_a_historial("user", mensaje)
        respuesta = responder_modo_estudiante(mensaje)
        if mensaje.lower() in ["salir", "salir modo estudiante", "terminar modo estudiante"]:
            modo_estudiante_activo = False
            respuesta += "\nModo estudiante desactivado."
        return jsonify({"respuesta": respuesta})

    if intencion == "hora":
        respuesta = obtener_hora()
        agregar_a_historial("user", mensaje)
        agregar_a_historial("assistant", respuesta)
    elif intencion == "fecha":
        respuesta = obtener_fecha()
        agregar_a_historial("user", mensaje)
        agregar_a_historial("assistant", respuesta)
    elif intencion == "clima":
        ciudad = "Madrid"
        palabras = mensaje.lower().split()
        for i, palabra in enumerate(palabras):
            if palabra in ["en", "de"] and i + 1 < len(palabras):
                ciudad = palabras[i + 1].capitalize()
                break
        respuesta = obtener_clima(ciudad)
        agregar_a_historial("user", mensaje)
        agregar_a_historial("assistant", respuesta)
    elif intencion == "graficar":
        expresion_a_graficar = mensaje.lower().replace("grafica", "").replace("gráfica", "").replace("graficar", "").strip()
        if not expresion_a_graficar:
            respuesta = "Por favor, dime qué función quieres que grafique."
        else:
            imagen_html = graficar_funcion(expresion_a_graficar)
            respuesta = imagen_html
        agregar_a_historial("user", mensaje)
        agregar_a_historial("assistant", "[Imagen gráfica generada]")
    elif intencion == "abrir_app":
        nombre_app = mensaje.lower().replace("abre", "").replace("abrir", "").replace("ejecuta", "").replace("ejecutar", "").strip()
        respuesta = abrir_aplicacion(nombre_app)
        agregar_a_historial("user", mensaje)
        agregar_a_historial("assistant", respuesta)
    elif intencion == "espacio":
        respuesta = obtener_espacio_libre()
        agregar_a_historial("user", mensaje)
        agregar_a_historial("assistant", respuesta)
    else:
        historial = agregar_a_historial("user", mensaje)
        try:
            completion = client.chat.completions.create(
                model="gpt-4",
                messages=historial
            )
            respuesta = completion.choices[0].message.content.strip()
            agregar_a_historial("assistant", respuesta)
        except Exception as e:
            respuesta = f"Error al comunicarse con la IA: {e}"

    logging.info(f"Asistente: {respuesta}")
    return jsonify({"respuesta": respuesta})

if __name__ == "__main__":
    app.run(debug=True)
