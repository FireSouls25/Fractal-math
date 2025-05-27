from flask import Flask, render_template, request, jsonify
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sympy import symbols, sympify, diff, integrate, lambdify, solve, Eq, fraction, pi as sympy_pi, E as sympy_e, latex, Derivative, Integral
import io
import base64
import json
import plotly.graph_objects as go

app = Flask(__name__)

def safe_eval_function(func_str):
    # Permitir pi y e
    func_str = func_str.replace('pi', 'np.pi').replace('e', 'np.e')
    replacements = {
        '^': '**',
        'np.sin': 'sin',
        'np.sin': 'sen',
        'np.cos': 'cos',
        'np.tan': 'tan',
        'np.exp': 'exp',
        'np.log': 'log',
        'np.sqrt': 'sqrt',
        'np.abs': 'abs',
        'np.arcsin': 'asin',
        'np.arccos': 'acos',
        'np.arctan': 'atan',
        'np.sinh': 'sinh',
        'np.cosh': 'cosh',
        'np.tanh': 'tanh',
        'np.arcsinh': 'asinh',
        'np.arccosh': 'acosh',
        'np.arctanh': 'atanh'
    }
    for np_func, sympy_func in replacements.items():
        func_str = func_str.replace(np_func, sympy_func)
    # Volver a poner pi y e para sympy
    func_str = func_str.replace('np.pi', 'pi').replace('np.e', 'E')
    return func_str

def find_discontinuities(expr, x_range, y_range):
    # Try to find discontinuities, but if not possible, just return empty
    try:
        x, y = symbols('x y')
        discontinuities = []
        if '/' in str(expr):
            num, den = fraction(expr)
            if den != 1:
                solutions = solve(den, [x, y], dict=True)
                if solutions:
                    for sol in solutions:
                        x_val = sol.get(x, None)
                        y_val = sol.get(y, None)
                        if x_val is not None and y_val is not None:
                            try:
                                x_val = float(x_val)
                                y_val = float(y_val)
                                if (x_range[0] <= x_val <= x_range[1] and 
                                    y_range[0] <= y_val <= y_range[1]):
                                    discontinuities.append((x_val, y_val))
                            except Exception:
                                continue
        return discontinuities
    except Exception:
        return []

def create_plot(func_str, x_range, y_range, z_range):
    try:
        x_sym, y_sym = symbols('x y')
        expr = sympify(safe_eval_function(func_str), locals={'pi': sympy_pi, 'E': sympy_e})
        discontinuities = find_discontinuities(expr, x_range, y_range)
        x = np.linspace(x_range[0], x_range[1], 200)
        y = np.linspace(y_range[0], y_range[1], 200)
        X, Y = np.meshgrid(x, y)
        func = lambdify((x_sym, y_sym), expr, modules=['numpy'])
        try:
            Z = func(X, Y)
        except Exception:
            Z = np.full_like(X, np.nan, dtype=float)
        Z = np.where(np.isfinite(Z), Z, np.nan)
        mask = np.logical_or(Z < z_range[0], Z > z_range[1])
        Z[mask] = np.nan
        Z[Z == z_range[0]] = np.nan
        Z[Z == z_range[1]] = np.nan
        surface = go.Surface(
            x=X, y=Y, z=Z,
            colorscale='RdBu',
            opacity=1,
            showscale=True,
            hoverinfo='skip',
            contours={"z": {"show": False}}
        )
        wireframe = go.Surface(
            x=X, y=Y, z=Z,
            colorscale=[[0, 'black'], [1, 'black']],
            showscale=False,
            opacity=1,
            hoverinfo='skip',
            contours={
                "x": {"show": True, "color": "black", "width": 1},
                "y": {"show": True, "color": "black", "width": 1},
                "z": {"show": False}
            },
            hidesurface=True
        )
        contour = go.Contour(
            x=x, y=y, z=Z,
            contours_coloring='lines',
            line_width=2,
            line_color='white',
            showscale=False,
            zmin=z_range[0], zmax=z_range[1],
            hoverinfo='skip',
            contours=dict(
                showlines=True,
                start=z_range[0],
                end=z_range[1],
                size=(z_range[1]-z_range[0])/10
            ),
            opacity=1,
            colorbar=None
        )
        data = [surface, wireframe, contour]
        if discontinuities:
            for x, y in discontinuities:
                data.append(go.Scatter3d(
                    x=[x], y=[y], z=[np.nanmax(Z)],
                    mode='markers',
                    marker=dict(size=10, color='red'),
                    name='Discontinuity'))
        fig = go.Figure(data=data)
        fig.update_layout(
            scene=dict(
                xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
                xaxis=dict(range=x_range, backgroundcolor='black', gridcolor='gray', zerolinecolor='gray', color='white', tickfont=dict(color='white'), titlefont=dict(color='white')),
                yaxis=dict(range=y_range, backgroundcolor='black', gridcolor='gray', zerolinecolor='gray', color='white', tickfont=dict(color='white'), titlefont=dict(color='white')),
                zaxis=dict(range=z_range, backgroundcolor='black', gridcolor='gray', zerolinecolor='gray', color='white', tickfont=dict(color='white'), titlefont=dict(color='white')),
                bgcolor='black',
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            showlegend=True,
            paper_bgcolor='black',
            plot_bgcolor='black',
            coloraxis_colorbar=dict(
                tickfont=dict(color='white'),
                titlefont=dict(color='white'),
                outlinecolor='white',
                bordercolor='white',
                bgcolor='black',
            ),
        )
        for trace in fig.data:
            if hasattr(trace, 'colorbar') and trace.colorbar:
                trace.colorbar.tickfont = dict(color='white')
                trace.colorbar.titlefont = dict(color='white')
                trace.colorbar.outlinecolor = 'white'
                trace.colorbar.bordercolor = 'white'
                trace.colorbar.bgcolor = 'black'
        plot_json = fig.to_json()
        interpreted = str(expr)
        try:
            dx, dy = calculate_derivatives(func_str)
            dx_latex = latex(Eq(Derivative(expr, x_sym), dx))
            dy_latex = latex(Eq(Derivative(expr, y_sym), dy))
        except Exception as e:
            dx_latex = r'\text{Error: ' + str(e).replace("'", "") + '}'
            dy_latex = r'\text{Error: ' + str(e).replace("'", "") + '}'
        try:
            integral_dxdy, integral_dydx = calculate_integrals(func_str, x_range, y_range)
            x_sym, y_sym = symbols('x y')
            expr = sympify(safe_eval_function(func_str), locals={'pi': sympy_pi, 'E': sympy_e})

            if isinstance(integral_dxdy, str) and "discontinuities" in integral_dxdy:
                int_dxdy_latex = r'\\text{' + integral_dxdy + '}'
            else:
                int_dxdy_latex = latex(Eq(Integral(expr, (x_sym, x_range[0], x_range[1]), (y_sym, y_range[0], y_range[1])), integral_dxdy))

            if isinstance(integral_dydx, str) and "discontinuities" in integral_dydx:
                int_dydx_latex = r'\\text{' + integral_dydx + '}'
            else:
                int_dydx_latex = latex(Eq(Integral(expr, (y_sym, y_range[0], y_range[1]), (x_sym, x_range[0], x_range[1])), integral_dydx))
        except Exception as e:
            int_dxdy_latex = r'\\text{Error: ' + str(e).replace("\'", "") + '}'
            int_dydx_latex = r'\\text{Error: ' + str(e).replace("\'", "") + '}'
        interpreted_latex = latex(expr) if expr else r'\text{Error interpretando función}'
        return plot_json, interpreted, dx_latex, dy_latex, int_dxdy_latex, int_dydx_latex, interpreted_latex
    except Exception as e:
        error_latex = r'\text{Error: ' + str(e).replace("'", "") + '}'
        return '', '', error_latex, error_latex, error_latex, error_latex, error_latex

def calculate_derivatives(func_str):
    try:
        x, y = symbols('x y')
        expr = sympify(safe_eval_function(func_str), locals={'pi': sympy_pi, 'E': sympy_e})
        
        # Calculate partial derivatives
        dx = diff(expr, x).simplify()
        dy = diff(expr, y).simplify()
        
        # Return SymPy expressions
        return dx, dy
    except Exception as e:
        # Return the exception object or re-raise
        raise Exception(f"Error calculating derivatives: {str(e)}")

def calculate_integrals(func_str, x_range, y_range):
    try:
        x, y = symbols('x y')
        expr = sympify(safe_eval_function(func_str), locals={'pi': sympy_pi, 'E': sympy_e})
        
        # Check for discontinuities in the integration region
        discontinuities = find_discontinuities(expr, x_range, y_range)
        if discontinuities:
            # Return a specific error string for discontinuities
            return "Integral cannot be computed due to discontinuities in the region", "Integral cannot be computed due to discontinuities in the region"
        
        # Calculate double integrals
        inner1 = integrate(expr, (y, y_range[0], y_range[1]))
        result_dxdy = integrate(inner1, (x, x_range[0], x_range[1]))
        inner2 = integrate(expr, (x, x_range[0], x_range[1]))
        result_dydx = integrate(inner2, (y, y_range[0], y_range[1]))
        
        # Simplify the results
        result_dxdy = result_dxdy.simplify()
        result_dydx = result_dydx.simplify()
        
        # Format numerical results to 5 decimal places
        if isinstance(result_dxdy, (int, float)):
            result_dxdy = round(result_dxdy, 5)
        if isinstance(result_dydx, (int, float)):
            result_dydx = round(result_dydx, 5)

        # Return SymPy expressions or formatted floats
        return result_dxdy, result_dydx
    except Exception as e:
        # Return the exception object or re-raise
        raise Exception(f"Error calculating integrals: {str(e)}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/plot', methods=['POST'])
def plot():
    data = request.get_json()
    func_str = data['function']
    x_range = [float(data['x_min']), float(data['x_max'])]
    y_range = [float(data['y_min']), float(data['y_max'])]
    z_range = [float(data['z_min']), float(data['z_max'])]
    
    dx_latex = ''
    dy_latex = ''
    int_dxdy_latex = ''
    int_dydx_latex = ''
    interpreted_latex = ''
    indefinite_integral_steps = '' # Variable to store the AI integral result

    try:
        # Create plot
        plot_json, interpreted, dx_latex, dy_latex, int_dxdy_latex, int_dydx_latex, interpreted_latex = create_plot(func_str, x_range, y_range, z_range)
        
        # Calculate and format derivatives
        try:
            x_sym, y_sym = symbols('x y') # Ensure symbols are defined
            expr = sympify(safe_eval_function(func_str), locals={'pi': sympy_pi, 'E': sympy_e}) # Re-sympify to get the expression object
            dx, dy = calculate_derivatives(func_str)
            
            # Generate LaTeX for partial derivatives with original function and result
            dx_latex = latex(Eq(Derivative(expr, x_sym), dx))
            dy_latex = latex(Eq(Derivative(expr, y_sym), dy))

        except Exception as e:
            dx_latex = r'\\text{Error: ' + str(e).replace("\'", "") + '}'
            dy_latex = r'\\text{Error: ' + str(e).replace("\'", "") + '}'

        # Calculate and format integrals
        try:
            integral_dxdy, integral_dydx = calculate_integrals(func_str, x_range, y_range)
            
            # Generate LaTeX for double integrals with original function and result
            x_sym, y_sym = symbols('x y')
            expr = sympify(safe_eval_function(func_str), locals={'pi': sympy_pi, 'E': sympy_e})

            if isinstance(integral_dxdy, str) and "discontinuities" in integral_dxdy:
                int_dxdy_latex = r'\\text{' + integral_dxdy + '}'
            else:
                int_dxdy_latex = latex(Eq(Integral(expr, (x_sym, x_range[0], x_range[1]), (y_sym, y_range[0], y_range[1])), integral_dxdy))

            if isinstance(integral_dydx, str) and "discontinuities" in integral_dydx:
                int_dydx_latex = r'\\text{' + integral_dydx + '}'
            else:
                int_dydx_latex = latex(Eq(Integral(expr, (y_sym, y_range[0], y_range[1]), (x_sym, x_range[0], x_range[1])), integral_dydx))
        except Exception as e:
             int_dxdy_latex = r'\\text{Error: ' + str(e).replace("\'", "") + '}'
             int_dydx_latex = r'\\text{Error: ' + str(e).replace("\'", "") + '}'

        return jsonify({
            'success': True,
            'plot': plot_json,
            'dx': dx_latex,
            'dy': dy_latex,
            'integral_dxdy': int_dxdy_latex,
            'integral_dydx': int_dydx_latex,
            'interpreted': interpreted,
            'interpreted_latex': interpreted_latex
        })
    except Exception as e:
        # Catching exceptions from create_plot or initial sympify
        error_message = str(e)
        # Format the error for MathJax display
        error_latex = r'\text{Error general: ' + error_message.replace("'", "") + '}'
        return jsonify({
            'success': False,
            'error': error_message, # Provide plain text error for alert
            'dx': error_latex,
            'dy': error_latex,
            'integral_dxdy': error_latex,
            'integral_dydx': error_latex,
            'interpreted_latex': error_latex
        })

if __name__ == '__main__':
    app.run(debug=True) 