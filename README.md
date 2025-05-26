### Grupo:
- David Enmanuel Castillo Florez
- Juan Camilo Erazo Ordoñez
- Luna Gabriela Mideros Botina

# Graficadora de funciones en 3 dimensiones

Aplicación web que permite graficar funciones de 2 variables en 3D, calcula derivadas parciales e integrale dobles definidas. Hecho con Flask, Matplotlib y SymPy.

## Características

- Graficos 3D interactivos
- Calculo de las derivadas parciales
- Computación de integrales dobles
- UI moderna y facil de usar
- Soporta una gran variedad de funciones matemáticas

## Instalación

1. Clonar el repositorio
```bash
git clone https://github.com/FireSouls25/Fractal-math.git
cd Fractal-math
```
2. Instalar las dependencias necesarias:

Se recomineda crear un entorno virtual

```bash
python -m venv venv
```
En windows: ```venv/Scripts/activate```
En macOS/Linux: ```source venv\bin\activate```

```bash
pip install -r requirements.txt
```
Inicia de la aplicación con:

```bash
python app.py
```

2. Abra el navegador y busque `http://localhost:5000`

3. Sintaxis de la aplicación:
   - Use `x` e `y` para las variables
   - Use `**` o `^` para potencias (e.j. `x**2` o `x^2` para x²)
   - Use simbolos matematicos estandar: `+`, `-`, `*`, `/`
   - Para operadores como √x , log(x), |x| use `sqrt(x)`, `log(x)`, `abs(x)` respectivamente
   - Para funciones trigonométricas use `sin(x)`, `cos(y)`, `tan(x)`
   - Para numeros como π o e use `pi` y `e` o `exp(x)` para e como base de un exponente

Se recomienda uso de parentesis para evitar ambiguedades, e.j. en lugar de escribir `1/x+y` mejor `1/(x+y)`

4. Seleccione el rango deseado para los ejes x, y, y z 

5. Click "Plot Function" para generar el grafico y los calculos de la función

## Ejemplos

- Paraboloide: `x^2 + y^2`
- Onda senoidal: `sin(x) * cos(y)`
- Paraboloide hiperbólico: `x^2 - y^2`
- Exponencial: `e^(-(x^2 + y^2))` o `exp(-(x^2 + y^2))`

## Notas

- Las funciones deben respetar la sintaxis y usar `x` e `y` como variables
- Los rangos de las variables deben ser razonables para evitar problemas computacionales
- Para funciones complicadas el calculo puede tomar algunos segundos