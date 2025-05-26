# Grupo:
David Enmanuel Castillo Florez
Juan Camilo Erazo Ordoñez

# 3D Function Plotter

A web application that allows users to plot 3D functions, calculate partial derivatives, and compute double integrals. Built with Flask, Matplotlib, and SymPy.

## Features

- Interactive 3D function plotting
- Real-time calculation of partial derivatives
- Double integral computation
- Modern, responsive UI
- Support for various mathematical functions

## Installation

1. Clonar el repositorio
```bash
git clone https://github.com/FireSouls25/Fractal-math.git
cd Fractal-math
```
2. Install the required dependencies:

Se recomineda crear un entorno virtual

```bash
python -m venv venv
```
En windows: ```venv/Scripts/activate```
En macOS/Linux: ```source venv\bin\activate```

```bash
pip install -r requirements.txt
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to `http://localhost:5000`

3. Enter your function using Python syntax:
   - Use `x` and `y` as variables
   - Use `**` for exponentiation (e.g., `x**2` for x²)
   - Use standard mathematical operators: `+`, `-`, `*`, `/`
   - Use mathematical functions from numpy (e.g., `np.sin(x)`, `np.cos(y)`)

4. Set the desired ranges for x, y, and z axes

5. Click "Plot Function" to generate the 3D plot and calculations

## Example Functions

- Paraboloid: `x**2 + y**2`
- Sine wave: `np.sin(x) * np.cos(y)`
- Hyperbolic paraboloid: `x**2 - y**2`
- Exponential: `np.exp(-(x**2 + y**2))`

## Notes

- The function must be a valid Python expression using `x` and `y` as variables
- The ranges should be reasonable to avoid computational issues
- For complex functions, the calculation might take a few seconds 